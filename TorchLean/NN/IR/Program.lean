/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Eval
public import TorchLean.NN.IR.Scalar
public import TorchLean.Tensor.Op.Storage
import FloatLib.Floats.ExecFloat.Conversion.Proof

/-!
# Typed programs of registered operators

A typed program (`IR.Program`) is a sequence of steps with its outputs and one scalar per
step. A step is an input, constant or operator node of `IR.Graph`, a conversion of one output
to another element type (`cast`), or a call of a registered implementation with its declared
signature (`call`). Its scalar is the element type of the step's value, frozen and detached ports,
and none for a step without such ports. Boolean, integer and index ports carry their elements
natively (`Bool`, `BitVec width`, `Port.IndexTuple bounds`), so they need no scalar. `Typed.ofPort`
gives the type of an output from its port and its step's scalar, and `Typed.dtype?` its element
type, none for indices. `Typed.ofDType?` gives the type of a value from its element type and shape.

A runtime (`IR.Program.Machine`) supplies the carrier, storage, arithmetic and optional capabilities
of each scalar element type, the conversions between element types, and the reader of interchange
words. `Machine.reference` uses the reference carriers of `TorchLean.NN.IR.Scalar` and
`IR.Reference.cast`. `IR.Program.check R` is the structural check of a program under `R`, and
`Payload.check` the check of its constants, custom programs and calls against it.

`IR.Program.run R P args` checks the program and the payload, then runs every step in order, unused
ones included. An input or a constant is read at its port. An operator runs at its step's scalar: a
custom operation by its checked program, and another operator by `Op.invoke` at the scalar's
arithmetic and capabilities. Where the scalar has no arithmetic or the step has no scalar, it runs
by its storage forward `Op.storageForward?`, the rule at `Float` of an operator whose ports are all
Boolean, integer or index ports, so no `Context` is built. A conversion runs entry by entry with
`R.cast`, and a call by the payload's implementation, whose result must have the step's type. The
run merges the conversion flags of every step, and `IR.Program.eval` reads the outputs.

A program whose steps are all operator-graph nodes, with one scalar or none, has an operator
graph (`IR.Program.graph?`). Once the program and its payload pass their checks, it runs on its
arguments as its graph. At a scalar `d` with arithmetic, every step's outputs are the graph's at
`d` with the payload's constants and registered custom programs (`Payload.graph`), and no flag is
raised (`run_graph?_ok`). Without a scalar, every step's outputs are the graph's at any scalar
type, as the same Boolean, integer and index values (`run_graph?_none`). Neither theorem assumes
that every operator has a rule: a step that fails fails in both.
-/

@[expose] public section

namespace TorchLean

open Port
open FloatLib.Floats.ExecFloat (ConversionStatus)
open FloatLib.Floats.Formats.BinaryInterchange.Model (IEEERoundingMode)

/-- Whether an integer port has a positive width, and `true` at every other port, which has no
width. -/
def Port.widthPositive (p : Port) : Bool :=
  match p.kind with
  | .integer _ width => 0 < width
  | _ => true

/-- A traversal of a nonempty list reads at a nonempty list of ports exactly when its first action's
result reads at the first port and the rest at the rest. -/
theorem Values.mapM_ofErased_cons {α : Type} [Storage α] {δ : Type}
    (h : δ → Except String (Op.Value α)) (p : Port) (ps : List Port) (a : δ) (l : List δ)
    (x : p.Carrier α) (xs : Values α ps) :
    (List.mapM h (a :: l) >>= Values.ofErased (p :: ps)) = .ok (x, xs) ↔
      (h a >>= p.read) = .ok x ∧ (l.mapM h >>= Values.ofErased ps) = .ok xs := by
  rw [List.mapM_cons]
  cases ha : h a with
  | error e => simp [bind, Except.bind]
  | ok y =>
    cases hl : l.mapM h with
    | error e => simp [bind, Except.bind]
    | ok ys =>
      simp only [bind, Except.bind, pure, Except.pure, Values.ofErased]
      cases p.read y <;> cases Values.ofErased ps ys <;> simp

/-- Reading two traversals at the ports of a list succeeds with the same values exactly when,
port by port, reading their actions' results at the port does. -/
theorem Values.ofErased_mapM_congr {α : Type} [Storage α] {β γ : Type}
    {f : Port × β → Except String (Op.Value α)} {g : γ → Except String (Op.Value α)} :
    ∀ {l : List (Port × β)} {bs : List γ},
      List.Forall₂ (fun pa b => ∀ x, (f pa >>= pa.1.read) = .ok x ↔
        (g b >>= pa.1.read) = .ok x) l bs →
      ∀ xs, (l.mapM f >>= Values.ofErased (l.map Prod.fst)) = .ok xs ↔
        (bs.mapM g >>= Values.ofErased (l.map Prod.fst)) = .ok xs := by
  intro l bs hl
  induction hl with
  | nil => intro xs; rfl
  | cons hab _ ih =>
      rintro ⟨x, xs⟩
      exact (mapM_ofErased_cons f _ _ _ _ x xs).trans
        ((and_congr (hab x) (ih xs)).trans (mapM_ofErased_cons g _ _ _ _ x xs).symm)

/-- A traversal of port-value pairs that reads at their ports succeeds only where each pair's
action reads at its port. -/
theorem Values.mapM_ofErased_mem {α : Type} [Storage α] {δ : Type}
    (h : Port × δ → Except String (Op.Value α)) :
    ∀ {l : List (Port × δ)} {ps : List Port}, l.map Prod.fst = ps → ∀ {xs : Values α ps},
      (l.mapM h >>= Values.ofErased ps) = .ok xs → ∀ pv ∈ l, ∃ x, (h pv >>= pv.1.read) = .ok x
  | [], _, _, _, _ => by simp
  | (p, a) :: l, _, rfl, xs, hl => by
      obtain ⟨ha, hl⟩ := (mapM_ofErased_cons h p (l.map Prod.fst) (p, a) l xs.1 xs.2).1 hl
      intro pv hpv
      rcases List.mem_cons.1 hpv with rfl | hpv
      · exact ⟨xs.1, ha⟩
      · exact mapM_ofErased_mem h rfl hl pv hpv

namespace IR.Program

/-- The type of one entry at an element type: `Bool` for Booleans, `BitVec width` for an integer
type of known width, and the runtime's carrier otherwise. -/
abbrev Entry (Scalar : DType → Type) : DType → Type
  | .bool => Bool
  | .integer _ (some width) => BitVec width
  | dtype => Scalar dtype

/-- The entries of the reference carriers are the reference carriers. -/
theorem entry_reference (d : DType) : Entry Reference.Carrier d = Reference.Carrier d := by
  cases d with
  | integer _ width => cases width <;> rfl
  | _ => rfl

/-- `Reference.cast` on the entries of the reference carriers. -/
def Entry.cast (source target : DType) (rounding : IEEERoundingMode)
    (x : Entry Reference.Carrier source) :
    Except String (Entry Reference.Carrier target × ConversionStatus) :=
  (Reference.cast source target rounding (_root_.cast (entry_reference source) x)).map
    fun (y, flags) => (_root_.cast (entry_reference target).symm y, flags)

/-- A runtime of typed programs: a carrier, storage, arithmetic and capabilities per scalar
element type, conversions between element types, and the reader of interchange words. -/
structure Machine where
  /-- The carrier of each scalar element type. -/
  Carrier : DType → Type
  /-- The storage of each carrier. -/
  storage : (d : DType) → Storage (Carrier d)
  /-- Whether the runtime implements an element type. -/
  supported : DType → Bool
  /-- The arithmetic of a scalar element type, if it has one. -/
  arithmetic : (d : DType) → Option (Context (Carrier d))
  /-- The optional capabilities of a scalar element type. -/
  capabilities : (d : DType) → Op.Capabilities (Carrier d)
  /-- Convert one element and return the conversion flags it raises, or an error. -/
  cast : (source target : DType) → IEEERoundingMode → Entry Carrier source →
    Except String (Entry Carrier target × ConversionStatus)
  /-- Decode one interchange word of a scalar element type. -/
  read : (d : DType) → Nat → Except String (Carrier d)

namespace Machine

/-- The carrier at a step's scalar, and `Empty` at a step without one, which reads no scalar. -/
abbrev CarrierAt (R : Machine) : Option DType → Type
  | some d => R.Carrier d
  | none => Empty

/-- The storage of `CarrierAt`. -/
abbrev storageAt (R : Machine) : (d : Option DType) → Storage (R.CarrierAt d)
  | some d => R.storage d
  | none => instArrayStorage Empty

/-- The reference runtime: configured binary arithmetic, native Booleans and integer words, and
`Reference.cast` between them. -/
def reference : Machine where
  Carrier := Reference.Carrier
  storage := Reference.storage
  supported := Reference.supported
  arithmetic := Reference.arithmetic
  capabilities := Reference.capabilities
  cast := Entry.cast
  read := Reference.read

/-- The reference runtime converts by `Reference.cast` on the reference carriers. -/
theorem cast_reference (source target : DType) (rounding : IEEERoundingMode)
    (x : Entry Reference.Carrier source) :
    reference.cast source target rounding x =
      (Reference.cast source target rounding (_root_.cast (entry_reference source) x)).map
        fun (y, flags) => (_root_.cast (entry_reference target).symm y, flags) := rfl

/-- The check of the scalar `d` of step `i`: a valid element type other than Boolean or integer
that `R` supports. -/
def checkScalar (R : Machine) (i : Nat) (d : DType) : Except String Unit := do
  d.check
  unless d.isScalar do throw s!"IR check: node {i}: {d} is a port sort, not a scalar"
  unless R.supported d do throw s!"IR check: node {i}: unsupported scalar {d}"

/-- The scalar check passes exactly at a valid scalar element type that `R` supports. -/
theorem checkScalar_eq_ok {R : Machine} {i : Nat} {d : DType} :
    R.checkScalar i d = .ok () ↔
      d.check = .ok () ∧ d.isScalar = true ∧ R.supported d = true := by
  unfold checkScalar
  cases hc : d.check with
  | error e => simp [bind, Except.bind]
  | ok u =>
      cases u
      cases hs : d.isScalar <;> cases hr : R.supported d <;>
        simp [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw]

end Machine

/-- The element type at a port: the port, and the scalar of a value, frozen or detached port. -/
structure Typed where
  /-- The port. -/
  port : Port
  /-- The scalar of a value, frozen or detached port, and none for any other port. -/
  scalar : Option DType
  deriving DecidableEq, Repr

namespace Typed

/-- The type of a port `p` of a step with scalar `d`: `d` only where the port carries it. -/
def ofPort (d : Option DType) (p : Port) : Typed :=
  { port := p, scalar := if p.kind.isScalar then d else none }

/-- The element type of a typed port: its scalar, `.bool`, or its port's integer type, and none at
an index port. -/
def dtype? (t : Typed) : Option DType :=
  match t.port.kind with
  | .value | .frozen | .detached => t.scalar
  | .bool => some .bool
  | .integer signed width => some (.integer signed (some width))
  | .index _ => none

/-- The type of a value of element type `d` at shape `s`: a Boolean port, an integer port of the
type's positive width, or a value port at the scalar `d`. It is `none` at an unbounded integer type,
a zero width or an element type that is no scalar. -/
def ofDType? (d : DType) (s : Shape) : Option Typed :=
  match d with
  | .bool => some ⟨.bool s, none⟩
  | .integer signed (some width) =>
      if 0 < width then some ⟨.integer signed width s, none⟩ else none
  | d => if d.isScalar then some ⟨.value s, some d⟩ else none

/-- A type that `ofDType? d s` returns has element type `d` and shape `s`, and it is no integer
port of width zero. -/
theorem ofDType?_eq_some {d : DType} {s : Shape} {t : Typed}
    (h : ofDType? d s = some t) :
    t.dtype? = some d ∧ t.port.shape = s ∧ t.port.widthPositive = true := by
  unfold ofDType? at h
  split at h
  · cases h
    exact ⟨rfl, rfl, rfl⟩
  · split at h
    · cases h
      simp_all [dtype?, Port.widthPositive]
    · cases h
  · split at h
    · cases h
      exact ⟨rfl, rfl, rfl⟩
    · cases h

/-- The tensors of a typed port under runtime `R`. -/
abbrev Carrier (R : Machine) (t : Typed) : Type :=
  @Port.Carrier (R.CarrierAt t.scalar) (R.storageAt t.scalar) t.port

/-- The check of a declared argument or result type at step `i`: a positive integer width, and a
scalar exactly at a value, frozen or detached port, a valid scalar element type that `R`
supports. -/
def check (t : Typed) (R : Machine) (i : Nat) : Except String Unit := do
  unless t.port.widthPositive do throw s!"IR check: node {i}: an integer port of width zero"
  match t.scalar, t.port.kind.isScalar with
  | some d, true => R.checkScalar i d
  | none, false => pure ()
  | _, _ => throw s!"IR check: node {i}: a declared type with a scalar exactly at value ports"

end Typed

/-- A type that `ofDType? d s` returns is `Typed.ofPort` of its scalar and its port. -/
theorem Typed.ofPort_of_ofDType {d : DType} {s : Shape} {τ : Typed}
    (h : Typed.ofDType? d s = some τ) : Typed.ofPort τ.scalar τ.port = τ := by
  unfold ofDType? at h
  split at h
  · cases h
    rfl
  · split at h
    · cases h
      rfl
    · cases h
  · split at h
    · cases h
      simp [Typed.ofPort, Kind.isScalar]
    · cases h

/-- One step of a typed program: an operator-graph node, a conversion of its source's elements
to the element type of `port`, or a registered call with its declared signature. -/
inductive Node where
  /-- An input, constant or operator node. -/
  | node (n : IR.Node)
  /-- Convert the output `source` with `rounding` to `port`. -/
  | cast (rounding : IEEERoundingMode) (source : Ref) (port : Port)
  /-- Call `name` on `args`, whose declared types are `signature`, returning `port`. -/
  | call (name : Lean.Name) (args : Array Ref) (signature : List Typed) (port : Port)
  deriving DecidableEq, Repr

/-- The ports a step produces. -/
def Node.outs : Node → List Port
  | .node n => n.outs
  | .cast _ _ port | .call _ _ _ port => [port]

/-- The ports of a step that can carry its scalar: an operator's input and output ports, the port
of an input or constant, the result port of a conversion or call. -/
def Node.ports : Node → List Port
  | .node (.input _ p) | .node (.const p) => [p]
  | .node (.op ⟨ins, outs, _⟩ _) => ins ++ outs
  | .cast _ _ port | .call _ _ _ port => [port]

/-- The ports of a step that must have a positive width where they are integer ports: its ports
and, for a call, the ports of its declared parameters. -/
def Node.widthPorts (n : Node) : List Port :=
  match n with
  | .call _ _ signature _ => n.ports ++ signature.map (·.port)
  | _ => n.ports

/-- The name, input shapes and output shape of a custom operation step, and none for any other
step. -/
def Node.custom? : Node → Option (String × List Shape × Shape)
  | .node (.op ⟨_, _, .custom name inputs output⟩ _) => some (name, inputs, output)
  | _ => none

end IR.Program

/-- A typed program: its steps in evaluation order (a step's id is its position), its outputs,
and the scalar of each step. -/
structure IR.Program where
  /-- The steps. -/
  nodes : Array Program.Node
  /-- The outputs, in order, repeats allowed. -/
  outputs : Array Ref
  /-- The scalar of each step's value, frozen and detached ports (none for a step without them). -/
  scalars : Array (Option DType)
  deriving DecidableEq, Repr

namespace IR.Program

open Values (mapM_ofErased_cons)

variable (p : Program)

/-- The scalar of step `i`. -/
def scalar (i : Nat) : Option DType := p.scalars[i]?.join

/-- The port of output `r` of a step before step `i`. -/
def portBefore (i : Nat) (r : Ref) : Option Port :=
  if r.node < i then (p.nodes[r.node]?).bind fun n => n.outs[r.out]? else none

/-- The type of output `r` of a step before step `i`. -/
def typedBefore (i : Nat) (r : Ref) : Option Typed :=
  (p.portBefore i r).map (Typed.ofPort (p.scalar r.node))

/-- The input index of each input step, in step order. -/
def inputIndices : List Nat :=
  p.nodes.toList.filterMap fun | .node (.input k _) => some k | _ => none

/-- The number of arguments. -/
def arity : Nat := p.inputIndices.length

/-- The step reading argument `k`. -/
def input? (k : Nat) : Option Nat :=
  p.nodes.findIdx? fun | .node (.input k' _) => k' == k | _ => false

/-- The check of a step's scalar: present exactly when the step has a value, frozen or detached
port, and then a valid element type other than Boolean or integer that `R` supports. -/
def checkScalar (R : Machine) (i : Nat) (n : Node) : Except String Unit :=
  match p.scalar i, n.ports.any (·.kind.isScalar) with
  | some d, true => R.checkScalar i d
  | none, false => pure ()
  | some _, false => throw s!"IR check: node {i}: a scalar for a node without value ports"
  | none, true => throw s!"IR check: node {i}: no scalar for a node with value ports"

/-- The check of step `i`: an operator-graph node passes `Node.check` and reads its value, frozen
and detached inputs at its own scalar. A conversion reads a Boolean, integer or scalar output and
writes one of the same shape, at types that `R` supports, and a call matches its declared
signature. -/
def checkNode (R : Machine) (i : Nat) : Node → Except String Unit
  | .node n => do
      n.check (p.portBefore i) i
      if let .op ⟨ins, _, _⟩ refs := n then
        for (r, q) in refs.toList.zip ins do
          if q.kind.isScalar && p.scalar r.node != p.scalar i then
            throw s!"IR check: node {i}: output {r.out} of node {r.node} has another scalar"
  | .cast _ r port => do
      let some t := p.typedBefore i r |
        throw s!"IR check: node {i}: no earlier output {r.out} of node {r.node}"
      let some source := t.dtype? | throw s!"IR check: node {i}: a conversion of indices"
      unless port.kind matches .value | .bool | .integer .. do
        throw s!"IR check: node {i}: a conversion to {repr port}"
      let some target := (Typed.ofPort (p.scalar i) port).dtype? |
        throw s!"IR check: node {i}: a conversion without a target scalar"
      unless port.shape = t.port.shape do
        throw s!"IR check: node {i}: a conversion changes the shape"
      unless R.supported source && R.supported target do
        throw s!"IR check: node {i}: unsupported conversion {source} to {target}"
  | .call name refs signature _ => do
      if name.isAnonymous then throw s!"IR check: node {i}: a call without a name"
      unless refs.size = signature.length do
        throw s!"IR check: node {i}: {refs.size} arguments for {signature.length} parameters"
      for s in signature do s.check R i
      for (r, s) in refs.toList.zip signature do
        match p.typedBefore i r with
        | some t =>
            unless s.port.accepts t.port && s.scalar == t.scalar do
              throw s!"IR check: node {i}: output {r.out} of node {r.node} does not match \
                its declared type"
        | none => throw s!"IR check: node {i}: no earlier output {r.out} of node {r.node}"

/-- The structural check of a typed program under `R`, first error first: one scalar per step
(`checkScalar`) and positive integer widths (`widthPorts`) at every step, then each step by
`checkNode` in step order, then nonempty outputs naming existing outputs, then input indices
`0, ..., arity - 1`, each once. -/
def check (R : Machine) : Except String Unit := do
  unless p.scalars.size = p.nodes.size do
    throw s!"IR check: {p.scalars.size} scalars for {p.nodes.size} nodes"
  for i in List.range p.nodes.size do
    match p.nodes[i]? with
    | some n => do
        p.checkScalar R i n
        unless n.widthPorts.all (·.widthPositive) do
          throw s!"IR check: node {i}: an integer port of width zero"
    | none => pure ()
  for i in List.range p.nodes.size do
    match p.nodes[i]? with
    | some n => p.checkNode R i n
    | none => pure ()
  if p.outputs.isEmpty then throw "IR check: a program without outputs"
  for r in p.outputs do
    unless (p.portBefore p.nodes.size r).isSome do
      throw s!"IR check: no output {r.out} of node {r.node}"
  let indices := p.inputIndices
  unless (List.range indices.length).all fun k => indices.count k == 1 do
    throw "IR check: the input indices are not 0, ..., arity - 1, each once"

/-- A value of a typed program under `R`: its type and its tensor. -/
structure Value (R : Machine) where
  /-- The type. -/
  typed : Typed
  /-- The tensor. -/
  tensor : typed.Carrier R

/-- A registered call: its declared argument and result types and its implementation. -/
structure Call (R : Machine) where
  /-- The argument types. -/
  inputs : List Typed
  /-- The result type. -/
  output : Typed
  /-- The implementation, with the conversion flags it raises. -/
  run : Array (Value R) → Except String (Value R × ConversionStatus)

/-- A registered custom operation: its scalar, input shapes and output shape, and its checked
program at that scalar. -/
structure Custom (R : Machine) where
  /-- The scalar. -/
  scalar : DType
  /-- The input shapes. -/
  inputs : List Shape
  /-- The output shape. -/
  output : Shape
  /-- The checked program. -/
  program : Kernel.Program (R.Carrier scalar)

/-- The payload of a typed program, one entry per step: the constant of a constant step, the
registered custom operation of a custom step, the implementation of a call. -/
structure Payload (R : Machine) where
  /-- The constants. -/
  consts : Array (Option (Value R))
  /-- The custom operations. -/
  customs : Array (Option (Custom R))
  /-- The calls. -/
  calls : Array (Option (Call R))

/-- Registered implementations: custom operations by their wire name, calls by name. -/
structure Env (R : Machine) where
  /-- The custom operations. -/
  custom : String → Option (Custom R)
  /-- The calls. -/
  call : Lean.Name → Option (Call R)

/-- The check of a payload against a program: one entry of each kind per step, a constant exactly
at each constant step with the step's type, a custom operation exactly at each custom step with
the step's scalar, input shapes and output shape, a call exactly at each call step with the
declared signature and result type. -/
def Payload.check {R : Machine} (P : Payload R) (p : Program) : Except String Unit := do
  unless P.consts.size = p.nodes.size && P.customs.size = p.nodes.size &&
      P.calls.size = p.nodes.size do
    throw "IR payload: not one entry of each kind per node"
  for i in [0:p.nodes.size] do
    match p.nodes[i]? with
    | some n => do
        match n, P.consts[i]?.join with
        | .node (.const port), some v =>
            unless v.typed = Typed.ofPort (p.scalar i) port do
              throw s!"IR payload: node {i}: a constant of another type"
        | .node (.const _), none => throw s!"IR payload: node {i}: no constant"
        | _, some _ => throw s!"IR payload: node {i}: a constant for a node that reads none"
        | _, none => pure ()
        match n.custom?, P.customs[i]?.join with
        | some (_, inputs, output), some c =>
            unless some c.scalar = p.scalar i && c.inputs = inputs && c.output = output do
              throw s!"IR payload: node {i}: a custom operation of another scalar or interface"
        | some _, none => throw s!"IR payload: node {i}: no custom operation"
        | none, some _ => throw s!"IR payload: node {i}: a custom operation for another node"
        | none, none => pure ()
        match n, P.calls[i]?.join with
        | .call _ _ signature port, some c =>
            unless c.inputs = signature && c.output = Typed.ofPort (p.scalar i) port do
              throw s!"IR payload: node {i}: a call of another signature"
        | .call .., none => throw s!"IR payload: node {i}: no call"
        | _, some _ => throw s!"IR payload: node {i}: a call for a node that calls none"
        | _, none => pure ()
    | none => pure ()

namespace Machine

/-- The storage of the entries of an element type: the runtime's at a scalar type, Lean's at
Booleans and integer words. -/
@[instance_reducible]
def entryStorage (R : Machine) : (d : DType) → Storage (Entry R.Carrier d)
  | .bool => inferInstanceAs (Storage Bool)
  | .integer _ (some width) => inferInstanceAs (Storage (BitVec width))
  | .integer signed none => R.storage (.integer signed none)
  | .context => R.storage .context
  | .binary format => R.storage (.binary format)
  | .decimal format => R.storage (.decimal format)
  | .posit bits => R.storage (.posit bits)
  | .rational => R.storage .rational
  | .real => R.storage .real
  | .complex component => R.storage (.complex component)
  | .named name parameters => R.storage (.named name parameters)

/-- At a scalar element type the entries are the runtime's carrier, with its storage. -/
theorem entry_carrier (R : Machine) {d : DType} (h : d.isScalar = true) :
    (⟨Entry R.Carrier d, R.entryStorage d⟩ : Σ α : Type, Storage α) =
      ⟨R.Carrier d, R.storage d⟩ := by
  cases d with
  | bool => simp [DType.isScalar] at h
  | integer _ width => simp [DType.isScalar] at h
  | _ => rfl

/-- The conversion of every entry of a tensor of `source` entries to `target`, in row-major order,
the conversion flags merged. -/
def castTensor (R : Machine) (source target : DType) (rounding : IEEERoundingMode) {s : Shape}
    (x : @Tensor (Entry R.Carrier source) s (R.entryStorage source)) :
    Except String (@Tensor (Entry R.Carrier target) s (R.entryStorage target) × ConversionStatus) :=
  letI := R.entryStorage source
  letI := R.entryStorage target
  let compute : StateT ConversionStatus (Except String) (Tensor (Entry R.Carrier target) s) :=
    Tensor.generateFlatM s fun index => do
      let value := Tensor.Rep.getFlat x index
      let (converted, flags) ← R.cast source target rounding value
      modify fun accumulated => accumulated.merge flags
      pure converted
  compute.run {}

/-- A tensor transported along an equality of carriers with their storage. -/
def tensorCast {a b : Σ α : Type, Storage α} (h : a = b) {s : Shape} (x : @Tensor a.1 s a.2) :
    @Tensor b.1 s b.2 :=
  h ▸ x

/-- Transporting a tensor back along an equality of carriers undoes transporting it. -/
@[simp] theorem tensorCast_symm {a b : Σ α : Type, Storage α} (h : a = b) {s : Shape}
    (x : @Tensor a.1 s a.2) : tensorCast h.symm (tensorCast h x) = x := by
  subst h
  rfl

/-- Transporting a tensor along two equalities transports it along their composite. -/
@[simp] theorem tensorCast_tensorCast {a b c : Σ α : Type, Storage α} (h : a = b) (h' : b = c)
    {s : Shape} (x : @Tensor a.1 s a.2) :
    tensorCast h' (tensorCast h x) = tensorCast (h.trans h') x := by
  subst h h'
  rfl

/-- Transporting a tensor along an equality of a carrier with itself keeps it. -/
@[simp] theorem tensorCast_self {a : Σ α : Type, Storage α} (h : a = a) {s : Shape}
    (x : @Tensor a.1 s a.2) : tensorCast h x = x :=
  rfl

/-- Converting the entries of a tensor transported along an equality of carriers, by a
conversion that agrees with another up to transports, is converting the tensor's entries by the
other and transporting the result. -/
theorem convert_tensorCast {a b c e : Σ α : Type, Storage α} (ha : a = c) (hb : b = e)
    {s : Shape} (f : a.1 → Except String (b.1 × ConversionStatus))
    (g : c.1 → Except String (e.1 × ConversionStatus))
    (hfg : ∀ x, g (_root_.cast (congrArg Sigma.fst ha) x) =
      (f x).map fun (y, flags) => (_root_.cast (congrArg Sigma.fst hb) y, flags))
    (x : @Tensor a.1 s a.2) :
    (letI := c.2
     letI := e.2
     (Tensor.generateFlatM s fun index => do
        let value := Tensor.Rep.getFlat (tensorCast ha x) index
        let (converted, flags) ← g value
        modify fun accumulated => accumulated.merge flags
        pure converted : StateT ConversionStatus (Except String) (Tensor e.1 s)).run {}) =
      (letI := a.2
       letI := b.2
       (Tensor.generateFlatM s fun index => do
          let value := Tensor.Rep.getFlat x index
          let (converted, flags) ← f value
          modify fun accumulated => accumulated.merge flags
          pure converted : StateT ConversionStatus (Except String) (Tensor b.1 s)).run {}).map
        fun (y, flags) => (tensorCast hb y, flags) := by
  subst ha hb
  obtain rfl : g = f := funext fun x => (hfg x).trans (by cases f x <;> rfl)
  exact (congrFun Except.map_id _).symm

end Machine

namespace Value

variable {R : Machine}

/-- The value as an operator value at the scalar `d`, with its port and tensor. A value at a value,
frozen or detached port reads only at its own scalar, and a Boolean, integer or index value reads
at any. -/
def toOp (v : Value R) (d : Option DType) :
    Except String (@Op.Value (R.CarrierAt d) (R.storageAt d)) :=
  match v with
  | ⟨⟨⟨.index bounds, s⟩, _⟩, x⟩ => pure ⟨.index bounds s, x⟩
  | ⟨⟨⟨.bool, s⟩, _⟩, x⟩ => pure ⟨.bool s, x⟩
  | ⟨⟨⟨.integer signed width, s⟩, _⟩, x⟩ => pure ⟨.integer signed width s, x⟩
  | ⟨⟨p, e⟩, x⟩ =>
      if h : e = d then pure (h ▸ (⟨p, x⟩ : @Op.Value (R.CarrierAt e) (R.storageAt e)))
      else throw s!"IR: a value at {repr e} read at {repr d}"

/-- The value of an operator value at the scalar `d`. -/
def ofOp (d : Option DType) : @Op.Value (R.CarrierAt d) (R.storageAt d) → Value R
  | ⟨⟨.index bounds, s⟩, x⟩ => ⟨⟨.index bounds s, none⟩, x⟩
  | ⟨⟨.bool, s⟩, x⟩ => ⟨⟨.bool s, none⟩, x⟩
  | ⟨⟨.integer signed width, s⟩, x⟩ => ⟨⟨.integer signed width s, none⟩, x⟩
  | ⟨p, x⟩ => ⟨⟨p, d⟩, x⟩

/-- The value of an operator value has the type of its port at the scalar. -/
theorem ofOp_typed (d : Option DType) (w : @Op.Value (R.CarrierAt d) (R.storageAt d)) :
    (ofOp d w).typed = Typed.ofPort d w.1 := by
  rcases w with ⟨⟨dt, s⟩, x⟩
  cases dt <;> rfl

/-- The entries of a Boolean, integer or scalar value as entries of its element type. -/
def entries (v : Value R) :
    Except String (Σ d : DType, Σ s : Shape, @Tensor (Entry R.Carrier d) s (R.entryStorage d)) :=
  match v with
  | ⟨⟨⟨.bool, s⟩, _⟩, x⟩ => pure ⟨.bool, s, x⟩
  | ⟨⟨⟨.integer signed width, s⟩, _⟩, x⟩ => pure ⟨.integer signed (some width), s, x⟩
  | ⟨⟨⟨.index _, _⟩, _⟩, _⟩ => throw "IR: a conversion of indices"
  | ⟨⟨⟨_, _⟩, none⟩, _⟩ => throw "IR: a value without a scalar"
  | ⟨⟨⟨.value, s⟩, some d⟩, x⟩ | ⟨⟨⟨.frozen, s⟩, some d⟩, x⟩
  | ⟨⟨⟨.detached, s⟩, some d⟩, x⟩ =>
      if h : d.isScalar then
        pure ⟨d, s, Machine.tensorCast (R.entry_carrier h).symm
          (x : @Tensor (R.Carrier d) s (R.storage d))⟩
      else throw s!"IR: {d} is a port sort, not a scalar"

/-- The value of type `t` holding entries of `target` of shape `s`, where `t` is a Boolean,
integer or scalar port of that element type and shape. -/
def ofEntries (t : Typed) (target : DType) {s : Shape}
    (y : @Tensor (Entry R.Carrier target) s (R.entryStorage target)) : Except String (Value R) :=
  if hs : t.port.shape = s then
    match t, target, hs, y with
    | ⟨⟨.bool, _⟩, _⟩, .bool, rfl, y => pure ⟨⟨.bool _, none⟩, y⟩
    | ⟨⟨.integer signed width, _⟩, _⟩, .integer signed' (some width'), rfl, y =>
        if h : signed = signed' ∧ width = width' then by
          obtain ⟨rfl, rfl⟩ := h
          exact pure ⟨⟨.integer signed width _, none⟩, y⟩
        else throw "IR: a conversion to another integer type"
    | ⟨⟨.value, _⟩, some d⟩, target, rfl, y =>
        if h : target = d ∧ d.isScalar then by
          obtain ⟨rfl, hd⟩ := h
          exact pure ⟨⟨.value _, some target⟩, (Machine.tensorCast (R.entry_carrier hd) y :
            @Tensor (R.Carrier target) _ (R.storage target))⟩
        else throw "IR: a conversion to another scalar"
    | _, _, _, _ => throw "IR: a conversion to a port without its element type"
  else throw "IR: a conversion changes the shape"

/-- Entries of an element type and shape make a value of the type of that element type and
shape, whose entries they are. -/
theorem ofEntries_ofDType {d : DType} {s : Shape} {τ : Program.Typed}
    (hτ : Program.Typed.ofDType? d s = some τ)
    (y : @Tensor (Program.Entry R.Carrier d) s (R.entryStorage d)) :
    ∃ w, Value.ofEntries τ d y = .ok w ∧ w.typed = τ ∧ w.entries = .ok ⟨d, s, y⟩ := by
  unfold Program.Typed.ofDType? at hτ
  split at hτ
  · cases hτ
    exact ⟨⟨⟨.bool s, none⟩, y⟩, by simp [Value.ofEntries, pure, Except.pure], rfl, rfl⟩
  · split at hτ
    · cases hτ
      rename_i signed width _
      exact ⟨⟨⟨.integer signed width s, none⟩, y⟩, by simp [Value.ofEntries, pure, Except.pure],
        rfl, rfl⟩
    · cases hτ
  · split at hτ
    · cases hτ
      rename_i hd
      refine ⟨⟨⟨.value s, some d⟩, Machine.tensorCast (R.entry_carrier hd) y⟩,
        by simp [Value.ofEntries, hd, pure, Except.pure], rfl, ?_⟩
      simp [Value.entries, hd, pure, Except.pure]
    · cases hτ

/-- A value of the type of an element type and shape is made back from its entries. -/
theorem ofEntries_entries {w : Value R} {d : DType} {s : Shape}
    {y : @Tensor (Program.Entry R.Carrier d) s (R.entryStorage d)}
    (hτ : Program.Typed.ofDType? d s = some w.typed) (he : w.entries = .ok ⟨d, s, y⟩) :
    Value.ofEntries w.typed d y = .ok w := by
  obtain ⟨τ, x⟩ := w
  simp only at hτ ⊢
  unfold Program.Typed.ofDType? at hτ
  split at hτ
  · cases hτ
    simp [Value.entries, pure, Except.pure] at he
    subst he
    simp [Value.ofEntries, pure, Except.pure]
  · split at hτ
    · cases hτ
      simp [Value.entries, pure, Except.pure] at he
      subst he
      simp [Value.ofEntries, pure, Except.pure]
    · cases hτ
  · split at hτ
    · cases hτ
      rename_i hd
      simp [Value.entries, hd, pure, Except.pure] at he
      subst he
      simp [Value.ofEntries, hd, pure, Except.pure]
    · cases hτ

end Value

/-- The outputs of operator `o` of step `i` at the scalar `d`, without arithmetic: its storage
forward after its attribute check and the reading of its inputs, or an error where it has none. -/
def runStorage (R : Machine) (i : Nat) (d : Option DType) {ins outs : List Port}
    (o : Op ins outs) (inputs : List (@Op.Value (R.CarrierAt d) (R.storageAt d))) :
    Except String (List (@Op.Value (R.CarrierAt d) (R.storageAt d))) := do
  letI := R.storageAt d
  do
  let some f := o.storageForward? |
    throw s!"IR: node {i} ({(Op.encode ⟨_, _, o⟩).tag}): no arithmetic at {repr d} \
      and no rule without arithmetic"
  o.check
  let xs ← Values.ofErased ins inputs
  pure (Values.toErased outs (← f _ xs))

/-- The outputs of the operator of step `i` at the scalar `d` on the outputs `refs` of earlier
steps. A custom operation runs its checked program. Another operator is invoked at the scalar's
arithmetic and capabilities, and where it has no rule there (a capability it reads is absent or
does not fit) the error names the step, the operator and the scalar. Without arithmetic, or
without a scalar, the operator runs its storage forward. -/
def runOp (R : Machine) (P : Payload R) (earlier : Array (List (Value R))) (i : Nat)
    (refs : Array Ref) :
    (d : Option DType) → Operator → Except String (List (@Op.Value (R.CarrierAt d) (R.storageAt d)))
  | some e, ⟨_, _, .custom name shapes output⟩ => do
      let some c := P.customs[i]?.join |
        throw s!"IR: custom {name}: no checked program for node {i}"
      if h : c.scalar = e then
        letI := R.storage e
        do
        let values ← refs.mapM fun r => do someTensor? (← (← r.read earlier).toOp (some e))
        pure [⟨.value output, ← customOutput (h ▸ c.program) name shapes output values⟩]
      else throw s!"IR: custom {name}: a program at another scalar for node {i}"
  | some e, ⟨_, _, o⟩ => do
      let inputs ← refs.toList.mapM fun r => do (← r.read earlier).toOp (some e)
      match R.arithmetic e with
      | some ctx =>
          letI := R.storage e
          letI := ctx
          letI := R.capabilities e
          if (o.rule? (α := R.Carrier e)).isNone then
            throw s!"IR: node {i} ({(Op.encode ⟨_, _, o⟩).tag}): no rule at {e}"
          else o.invoke (pure inputs)
      | none => runStorage R i (some e) o inputs
  | none, ⟨_, _, o⟩ => do
      let inputs ← refs.toList.mapM fun r => do (← r.read earlier).toOp none
      runStorage R i none o inputs

/-- The outputs of step `i` from the arguments and the outputs of earlier steps, with the flags a
conversion or a call raises. An input reads its argument and a constant its payload value at its
port (`Port.read`), an operator runs by `runOp`, a conversion converts every entry with `R.cast`,
and a call runs the payload's implementation, whose result must have the step's type. -/
def step (R : Machine) (P : Payload R) (args : Array (Value R))
    (earlier : Array (List (Value R))) (i : Nat) :
    Node → Except String (List (Value R) × ConversionStatus)
  | .node (.input k port) => do
      let some a := args[k]? | throw s!"IR: no input {k}"
      letI := R.storageAt (p.scalar i)
      do
      let x ← port.read (← a.toOp (p.scalar i))
      pure ([Value.ofOp (p.scalar i) ⟨port, x⟩], {})
  | .node (.const port) => do
      let some a := P.consts[i]?.join | throw s!"IR: no constant for node {i}"
      letI := R.storageAt (p.scalar i)
      do
      let x ← port.read (← a.toOp (p.scalar i))
      pure ([Value.ofOp (p.scalar i) ⟨port, x⟩], {})
  | .node (.op o refs) => do
      let results ← runOp R P earlier i refs (p.scalar i) o
      pure (results.map (Value.ofOp (p.scalar i)), {})
  | .cast rounding r port => do
      let ⟨source, _, x⟩ ← (← r.read earlier).entries
      let t := Typed.ofPort (p.scalar i) port
      let some target := t.dtype? | throw s!"IR: node {i}: a conversion without a target scalar"
      let (y, flags) ← R.castTensor source target rounding x
      pure ([← Value.ofEntries t target y], flags)
  | .call name refs _ port => do
      let some c := P.calls[i]?.join | throw s!"IR: no implementation of {name} for node {i}"
      let (v, flags) ← c.run (← refs.mapM (·.read earlier))
      unless v.typed = Typed.ofPort (p.scalar i) port do
        throw s!"IR: node {i}: {name} returned a value of another type"
      pure ([v], flags)

/-- Checks the program, the payload and the number of arguments, then runs every step in order,
unused ones included, and returns the outputs of each step with the conversion flags of all the
steps merged. -/
def run (R : Machine) (P : Payload R) (args : Array (Value R)) :
    Except String (Array (List (Value R)) × ConversionStatus) := do
  p.check R
  P.check p
  unless args.size = p.arity do
    throw s!"IR: {args.size} arguments for {p.arity} inputs"
  p.nodes.toList.foldlM (fun (earlier, flags) n => do
    let (vs, raised) ← p.step R P args earlier earlier.size n
    pure (earlier.push vs, flags.merge raised)) (#[], {})

/-- The values of the program's outputs, in order, and the conversion flags of its run. -/
def eval (R : Machine) (P : Payload R) (args : Array (Value R)) :
    Except String (List (Value R) × ConversionStatus) := do
  let (values, flags) ← p.run R P args
  pure (← p.outputs.toList.mapM (·.read values), flags)

section Graph

variable {p : Program} {R : Machine} {P : Payload R} {args : Array (Value R)}

/-- A value written at a scalar and read back at that scalar is unchanged. -/
theorem Value.toOp_ofOp (d : Option DType)
    (w : @Op.Value (R.CarrierAt d) (R.storageAt d)) :
    (Value.ofOp d w).toOp d = .ok w := by
  rcases w with ⟨⟨dt, s⟩, x⟩
  cases dt <;> simp [Value.ofOp, Value.toOp, pure, Except.pure]

/-- A value written at a scalar reads without one exactly where it is a Boolean, integer or index
value, as the same value. -/
theorem Value.toOp_ofOp_none (d : DType)
    (w : @Op.Value (R.CarrierAt (some d)) (R.storageAt (some d)))
    (z : @Op.Value (R.CarrierAt none) (R.storageAt none)) :
    (Value.ofOp (some d) w).toOp none = .ok z ↔ Op.NativeRel z w := by
  rcases w with ⟨⟨dt, s⟩, x⟩
  cases dt <;>
    simp only [Value.ofOp, Value.toOp, pure, Except.pure, throw, throwThe, MonadExceptOf.throw,
      reduceCtorEq, Except.ok.injEq, false_iff, reduceDIte]
  all_goals first
    | (intro h; cases h)
    | (constructor
       · rintro rfl; constructor
       · intro h; cases h; rfl)

/-- Related Boolean, integer and index values are the same program value. -/
theorem Value.ofOp_native {e e' : Option DType}
    {z : @Op.Value (R.CarrierAt e) (R.storageAt e)}
    {w : @Op.Value (R.CarrierAt e') (R.storageAt e')} (h : Op.NativeRel z w) :
    Value.ofOp e z = Value.ofOp e' w := by
  cases h <;> rfl

/-- Related lists of Boolean, integer and index values are the same program values. -/
theorem Value.map_ofOp_native {e e' : Option DType}
    {zs : List (@Op.Value (R.CarrierAt e) (R.storageAt e))}
    {ws : List (@Op.Value (R.CarrierAt e') (R.storageAt e'))}
    (h : List.Forall₂ Op.NativeRel zs ws) :
    zs.map (Value.ofOp e) = ws.map (Value.ofOp e') := by
  induction h with
  | nil => rfl
  | cons hzw _ ih => simp [Value.ofOp_native hzw, ih]

/-- A value read at a scalar is at the value's port. -/
theorem Value.toOp_port {v : Value R} {e : Option DType}
    {z : @Op.Value (R.CarrierAt e) (R.storageAt e)} (h : v.toOp e = .ok z) :
    z.1 = v.typed.port := by
  rcases v with ⟨⟨⟨dt, s⟩, sc⟩, t⟩
  cases dt <;> simp only [Value.toOp, pure, Except.pure, Except.ok.injEq] at h
  all_goals first
    | (subst h; rfl)
    | (split at h
       · rename_i hc
         subst hc
         cases h
         rfl
       · cases h)

/-- A Boolean, integer or index value reads at every scalar as the same value. -/
theorem Value.toOp_native {v : Value R} (hv : v.typed.port.kind.isScalar = false)
    {e e' : Option DType} {z : @Op.Value (R.CarrierAt e) (R.storageAt e)}
    (h : v.toOp e = .ok z) :
    ∃ z', v.toOp e' = .ok z' ∧ Op.NativeRel z z' := by
  rcases v with ⟨⟨⟨dt, s⟩, sc⟩, t⟩
  cases dt <;> simp [Kind.isScalar] at hv <;>
    simp only [Value.toOp, pure, Except.pure, Except.ok.injEq] at h ⊢ <;> subst h <;>
    exact ⟨_, rfl, by constructor⟩

/-- A value read at a Boolean, integer or index port at one scalar reads there at any scalar as
the same tensor. -/
theorem Value.read_native {a : Value R} {port : Port} (hp : port.kind.isScalar = false)
    {e e' : Option DType} {x : @Port.Carrier (R.CarrierAt e) (R.storageAt e) port}
    (h : (a.toOp e >>= @Port.read (R.CarrierAt e) (R.storageAt e) port) = .ok x) :
    ∃ y, (a.toOp e' >>= @Port.read (R.CarrierAt e') (R.storageAt e') port) = .ok y ∧
      Op.NativeRel (⟨port, x⟩ : @Op.Value (R.CarrierAt e) (R.storageAt e)) ⟨port, y⟩ := by
  obtain ⟨z, hz, hx⟩ := bind_eq_ok.1 h
  have hap : a.typed.port.kind.isScalar = false := by
    rw [← Value.toOp_port hz, Port.fst_eq_of_read hp hx]
    exact hp
  obtain ⟨z', hz', hzz⟩ := Value.toOp_native hap (e' := e') hz
  obtain ⟨y, hy, hxy⟩ := Port.read_native hzz port hx
  exact ⟨y, bind_eq_ok.2 ⟨z', hz', hy⟩, hxy⟩

/-- Boolean, integer and index values written at a scalar read without one as the same values. -/
private theorem Value.mapM_toOp_ofOp_none (d : DType) :
    ∀ {ws : List (@Op.Value (R.CarrierAt (some d)) (R.storageAt (some d)))},
      (∀ w ∈ ws, w.1.kind.isScalar = false) →
      ∃ zs, List.Forall₂ (fun w z => (Value.ofOp (some d) w).toOp none = .ok z) ws zs ∧
        List.Forall₂ Op.NativeRel zs ws
  | [], _ => ⟨[], .nil, .nil⟩
  | ⟨⟨dt, s⟩, x⟩ :: ws, h => by
      obtain ⟨zs, hzs, hr⟩ := Value.mapM_toOp_ofOp_none d (ws := ws)
        (fun v hv => h v (by simp [hv]))
      have hw := h ⟨⟨dt, s⟩, x⟩ (by simp)
      cases dt <;> simp [Kind.isScalar] at hw <;>
        exact ⟨_ :: zs, .cons ((Value.toOp_ofOp_none d _ _).2 (by constructor)) hzs,
          .cons (by constructor) hr⟩

/-- The operator-graph node of a step that is one, and none for a conversion or a call. -/
def Node.node? : Node → Option IR.Node
  | .node n => some n
  | _ => none

/-- Steps that are all operator-graph nodes are those nodes. -/
theorem Node.mapM_node? : ∀ {l : List Node} {ns : List IR.Node},
    l.mapM Node.node? = some ns → l = ns.map .node
  | [], ns, h => by simp at h; subst h; rfl
  | .node n :: l, ns, h => by
      simp only [List.mapM_cons, Node.node?] at h
      cases hl : l.mapM Node.node? with
      | none => simp [hl] at h
      | some ns' =>
          simp only [hl] at h
          cases h
          simp [Node.mapM_node? hl]
  | .cast .. :: _, _, h => by simp [List.mapM_cons, Node.node?] at h
  | .call .. :: _, _, h => by simp [List.mapM_cons, Node.node?] at h

/-- The operator graph of a program whose steps are all operator-graph nodes and whose scalars
other than none agree, paired with that scalar, or with none where every scalar is none. -/
def graph? (p : Program) : Option (Graph × Option DType) := do
  let nodes ← p.nodes.toList.mapM Node.node?
  let scalars := p.scalars.toList.filterMap id
  if scalars.all (some · == scalars.head?) then pure (⟨nodes.toArray, p.outputs⟩, scalars.head?)
  else none

/-- A program with a graph has the graph's nodes as its steps and its outputs, and every scalar
it has is the graph's. -/
theorem graph?_eq_some {g : Graph} {sd : Option DType} (h : p.graph? = some (g, sd)) :
    p.nodes = g.nodes.map .node ∧ g.outputs = p.outputs ∧
      ∀ i e, p.scalar i = some e → sd = some e := by
  unfold graph? at h
  cases hn : p.nodes.toList.mapM Node.node? with
  | none => simp [hn] at h
  | some ns =>
      simp only [hn, Option.bind_eq_bind, Option.bind_some] at h
      split at h
      · rename_i hall
        simp only [pure, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨?_, rfl, fun i e he => ?_⟩
        · apply Array.ext'
          simp [Node.mapM_node? hn]
        · have hmem : e ∈ p.scalars.toList.filterMap id := by
            simp only [scalar, Option.join_eq_some_iff] at he
            exact List.mem_filterMap.2
              ⟨some e, Array.mem_toList_iff.2 (Array.mem_of_getElem? he), rfl⟩
          exact (beq_iff_eq.1 (List.all_eq_true.1 hall e hmem)).symm
      · cases h

/-- The payload of a program's operator graph at the scalar `d`: the constants read at `d` and
the registered custom programs at `d`. -/
def Payload.graph (P : Payload R) (d : DType) : @IR.Payload (R.Carrier d) (R.storage d) :=
  letI := R.storage d
  { const? := fun i => P.consts[i]?.join.bind fun v => (v.toOp (some d)).toOption
    custom? := fun i => P.customs[i]?.join.bind fun c =>
      if h : c.scalar = d then some (h ▸ c.program) else none }

/-- A checked program passes the scalar check and the step check at every step. -/
theorem check_step (h : p.check R = .ok ()) {i : Nat} {n : Node} (hn : p.nodes[i]? = some n) :
    p.checkScalar R i n = .ok () ∧ p.checkNode R i n = .ok () := by
  have hi : i < p.nodes.size := by
    by_contra hc
    rw [Array.getElem?_eq_none (Nat.le_of_not_lt hc)] at hn
    cases hn
  unfold check at h
  dsimp only at h
  split at h
  · obtain ⟨_, h₁, h⟩ := bind_eq_ok.1 h
    obtain ⟨_, h₂, -⟩ := bind_eq_ok.1 h
    have k₁ := forIn_eq_ok ?_ h₁ i (List.mem_range.2 hi)
    · have k₂ := forIn_eq_ok ?_ h₂ i (List.mem_range.2 hi)
      · simp only [hn] at k₁ k₂
        obtain ⟨_, k₁, -⟩ := bind_eq_ok.1 k₁
        obtain ⟨_, k₂, -⟩ := bind_eq_ok.1 k₂
        exact ⟨k₁, k₂⟩
      · intro b
        split
        · intro hb
          obtain ⟨_, -, hb⟩ := bind_eq_ok.1 hb
          cases hb
        · intro hb
          cases hb
    · intro b
      split
      · intro hb
        obtain ⟨_, -, hb⟩ := bind_eq_ok.1 hb
        split at hb
        · cases hb
        · obtain ⟨_, -, hb⟩ := bind_eq_ok.1 hb
          cases hb
      · intro hb
        cases hb
  · obtain ⟨_, h, -⟩ := bind_eq_ok.1 h
    cases h

/-- The scalar check of a step without a scalar: its ports are Boolean, integer or index ports. -/
theorem checkScalar_none {i : Nat} {n : Node} (h : p.checkScalar R i n = .ok ())
    (hs : p.scalar i = none) : n.ports.all (fun q => !q.kind.isScalar) := by
  unfold checkScalar at h
  rw [hs] at h
  cases hany : n.ports.any (fun q => q.kind.isScalar) with
  | false => simpa using hany
  | true =>
      rw [hany] at h
      simp [throw, throwThe, MonadExceptOf.throw] at h

/-- The step check of an operator node: one reference per input port. -/
theorem checkNode_op {i : Nat} {o : Operator} {refs : Array Ref}
    (h : p.checkNode R i (.node (.op o refs)) = .ok ()) : refs.size = o.1.length := by
  rcases o with ⟨ins, outs, op⟩
  obtain ⟨_, h, -⟩ := bind_eq_ok.1 h
  unfold IR.Node.check at h
  by_contra hne
  simp [hne, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- An operator whose ports are all Boolean, integer or index ports is not a custom operation,
whose output is a value. -/
private theorem not_custom_of_native {ins outs : List Port} {op : Op ins outs}
    (hnat : (ins ++ outs).all (fun q => !q.kind.isScalar)) :
    ∀ (name : String) (shapes : List Shape) (output : Shape), ins = shapes.map Port.value →
      HEq (⟨outs, op⟩ : Σ outs, Op ins outs)
        (⟨[Port.value output], Op.custom name shapes output⟩ :
          Σ outs, Op (shapes.map Port.value) outs) → False := by
  intro name shapes output hins hh
  subst hins
  obtain ⟨rfl, -⟩ := Sigma.mk.inj_iff.1 (eq_of_heq hh)
  simp [Kind.isScalar] at hnat

/-- The run of a checked program with a checked payload and its arguments is its fold over the
steps. -/
theorem run_eq_foldlM (hcheck : p.check R = .ok ()) (hP : P.check p = .ok ())
    (hsize : args.size = p.arity) :
    p.run R P args = p.nodes.toList.foldlM (fun (earlier, flags) n => do
        let (vs, raised) ← p.step R P args earlier earlier.size n
        pure (earlier.push vs, flags.merge raised)) (#[], {}) := by
  simp only [run, hcheck, hP, hsize]
  rfl

/-- Running steps from the flags `f₀` merges `f₀` into the flags of running them from none: the
outputs do not depend on the flags. -/
theorem foldlM_step_merge (l : List Node) (earlier : Array (List (Value R)))
    (f₀ : ConversionStatus) :
    l.foldlM (fun (earlier, flags) n => do
        let (vs, raised) ← p.step R P args earlier earlier.size n
        pure (earlier.push vs, flags.merge raised)) (earlier, f₀) =
      (fun r => (r.1, f₀.merge r.2)) <$> l.foldlM (fun (earlier, flags) n => do
        let (vs, raised) ← p.step R P args earlier earlier.size n
        pure (earlier.push vs, flags.merge raised)) (earlier, {}) := by
  induction l generalizing earlier f₀ with
  | nil => simp [ConversionStatus.merge_empty]
  | cons n l ih =>
    simp only [List.foldlM_cons]
    cases hs : p.step R P args earlier earlier.size n with
    | error e => rfl
    | ok r =>
      obtain ⟨vs, raised⟩ := r
      simp only [bind, Except.bind, pure, Except.pure] at ih ⊢
      rw [ih, ih (f₀ := ConversionStatus.merge {} raised), Functor.map_map]
      congr 1
      funext r
      simp [ConversionStatus.merge_assoc]

variable {β : Type} [Storage β] [Context β] [Op.Capabilities β]

/-- The storage forward of an operator that has one succeeds exactly when its attribute check, the
reading of its inputs at its ports and the forward succeed, and returns the forward's outputs. -/
theorem runStorage_eq_ok {R : Machine} {i : Nat} {d : Option DType} {ins outs : List Port}
    {o : Op ins outs}
    {f : ∀ (α : Type) [Storage α], Values α ins → Except String (Values α outs)}
    (hf : o.storageForward? = some f)
    {inputs outputs : List (@Op.Value (R.CarrierAt d) (R.storageAt d))} :
    runStorage R i d o inputs = .ok outputs ↔
      ∃ xs ys, o.check = .ok () ∧ @Values.ofErased _ (R.storageAt d) ins inputs = .ok xs ∧
        @f _ (R.storageAt d) xs = .ok ys ∧
          outputs = @Values.toErased _ (R.storageAt d) outs ys := by
  unfold runStorage
  simp only [hf]
  cases hc : o.check with
  | error err => simp [bind, Except.bind]
  | ok u =>
    cases hx : @Values.ofErased _ (R.storageAt d) ins inputs with
    | error err => simp [bind, Except.bind]
    | ok xs =>
      cases hy : @f _ (R.storageAt d) xs with
      | error err => simp [bind, Except.bind, hy]
      | ok ys => simp [bind, Except.bind, hy, pure, Except.pure, eq_comm]

/-- The storage forward of an operator without a scalar on the inputs `xp` and its invocation on
the inputs `xg` both fail or both succeed with related outputs, when the inputs correspond. -/
private theorem runStorage_invoke {i : Nat} {ins outs : List Port} (o : Op ins outs)
    (hnat : (ins ++ outs).all (fun p => !p.kind.isScalar))
    {xp : Except String (List (@Op.Value (R.CarrierAt none) (R.storageAt none)))}
    {xg : Except String (List (Op.Value β))}
    (h₁ : ∀ zs, xp = .ok zs → ∃ ws, xg = .ok ws ∧ List.Forall₂ Op.NativeRel zs ws)
    (h₂ : ∀ ws, xg = .ok ws → (∀ w ∈ ws, w.1.kind.isScalar = false) →
      ∃ zs, xp = .ok zs ∧ List.Forall₂ Op.NativeRel zs ws) :
    (∀ rs, (xp >>= runStorage R i none o) = .ok rs →
      ∃ rs', o.invoke xg = .ok rs' ∧ List.Forall₂ Op.NativeRel rs rs') ∧
    (∀ rs', o.invoke xg = .ok rs' →
      ∃ rs, (xp >>= runStorage R i none o) = .ok rs ∧ List.Forall₂ Op.NativeRel rs rs') := by
  obtain ⟨f, hf⟩ := Option.isSome_iff_exists.1 (Op.storageForward?_isSome_of_native o hnat)
  have hrun : ∀ zs rs, runStorage R i none o zs = .ok rs ↔ ∃ xs ys, o.check = .ok () ∧
      Values.ofErased ins zs = .ok xs ∧ f _ xs = .ok ys ∧ rs = Values.toErased outs ys :=
    fun _ _ => runStorage_eq_ok hf
  refine ⟨fun rs h => ?_, fun rs' h => ?_⟩
  · obtain ⟨zs, hzs, h⟩ := bind_eq_ok.1 h
    obtain ⟨ws, hws, hzw⟩ := h₁ zs hzs
    obtain ⟨xs, ys, hc, hx, hy, rfl⟩ := (hrun zs rs).1 h
    obtain ⟨xs', hx', hxx⟩ := Values.ofErased_native ins hzw hx
    obtain ⟨ys', hy', hyy⟩ := Op.storageForward_native o hf hxx hy
    refine ⟨_, ?_, hyy⟩
    rw [hws]
    exact (Op.invoke_storageForward o hf).2 ⟨xs', ys', hc, hx', hy', rfl⟩
  · obtain ⟨ws, -, -, -, hws, -⟩ := Op.invoke_eq_ok h
    rw [hws] at h
    obtain ⟨xs', ys', hc, hx', hy', rfl⟩ := (Op.invoke_storageForward o hf).1 h
    have hins : ins.all (fun p => !p.kind.isScalar) := by
      rw [List.all_append, Bool.and_eq_true] at hnat
      exact hnat.1
    obtain ⟨zs, hzs, hzw⟩ := h₂ ws hws (Values.native_of_ofErased ins hins hx')
    obtain ⟨xs, hx, hxx⟩ := Values.ofErased_native ins (Op.NativeRel.forall₂_symm hzw) hx'
    obtain ⟨ys, hy, hyy⟩ := Op.storageForward_native o hf hxx hy'
    exact ⟨_, bind_eq_ok.2 ⟨zs, hzs, (hrun zs _).2 ⟨xs, ys, hc, hx, hy, rfl⟩⟩,
      Op.NativeRel.forall₂_symm hyy⟩

/-- A program whose steps are the nodes `l` runs as the graph run of `l` when each step does: from
outputs `Z.map (·.map F)` related row by row to the graph's outputs, each step and its node both
succeed, with related outputs and no flags, or both fail. -/
private theorem run_nodes {X : Type} (F : X → Value R) (rel : X → Op.Value β → Prop)
    (GP : IR.Payload β) (gargs : List (Op.Value β)) :
    ∀ (l : List IR.Node) (Z : Array (List X)) (W : Array (List (Op.Value β)))
      (fl : ConversionStatus), List.Forall₂ (List.Forall₂ rel) Z.toList W.toList →
    (∀ j n, l[j]? = some n → ∀ (Z' : Array (List X)) (W' : Array (List (Op.Value β))),
      Z'.size = Z.size + j → List.Forall₂ (List.Forall₂ rel) Z'.toList W'.toList →
      (∀ vs fl', p.step R P args (Z'.map (·.map F)) (Z.size + j) (.node n) = .ok (vs, fl') →
        ∃ xs ws, n.eval GP gargs W' (Z.size + j) = .ok ws ∧ List.Forall₂ rel xs ws ∧
          vs = xs.map F ∧ fl' = {}) ∧
      (∀ ws, n.eval GP gargs W' (Z.size + j) = .ok ws →
        ∃ xs, p.step R P args (Z'.map (·.map F)) (Z.size + j) (.node n) = .ok (xs.map F, {}) ∧
          List.Forall₂ rel xs ws)) →
    (∀ E fl', (l.map Node.node).foldlM (fun (earlier, flags) n => do
        let (vs, raised) ← p.step R P args earlier earlier.size n
        pure (earlier.push vs, flags.merge raised)) (Z.map (·.map F), fl) = .ok (E, fl') →
      ∃ (Z' : Array (List X)) (W' : Array (List (Op.Value β))),
        runNodes GP gargs W l = .ok W' ∧
        List.Forall₂ (List.Forall₂ rel) Z'.toList W'.toList ∧ E = Z'.map (·.map F) ∧ fl' = fl) ∧
    (∀ W', runNodes GP gargs W l = .ok W' →
      ∃ Z' : Array (List X), (l.map Node.node).foldlM (fun (earlier, flags) n => do
        let (vs, raised) ← p.step R P args earlier earlier.size n
        pure (earlier.push vs, flags.merge raised)) (Z.map (·.map F), fl) =
          .ok (Z'.map (·.map F), fl) ∧ List.Forall₂ (List.Forall₂ rel) Z'.toList W'.toList)
  | [], Z, W, fl, hrel, _ => by
      refine ⟨fun E fl' h => ?_, fun W' h => ?_⟩
      · simp only [List.map_nil, List.foldlM_nil, pure, Except.pure, Except.ok.injEq,
          Prod.mk.injEq] at h
        exact ⟨Z, W, rfl, hrel, h.1.symm, h.2.symm⟩
      · simp only [runNodes_nil, Except.ok.injEq] at h
        subst h
        exact ⟨Z, rfl, hrel⟩
  | n :: l, Z, W, fl, hrel, hstep => by
      have hstep₀ := hstep 0 n rfl Z W rfl hrel
      simp only [Nat.add_zero] at hstep₀
      have hsize : (Z.map (·.map F)).size = Z.size := Array.size_map ..
      have hW : W.size = Z.size := by simpa using hrel.length_eq.symm
      have ih (xs : List X) (ws : List (Op.Value β)) (hx : List.Forall₂ rel xs ws) :=
        run_nodes F rel GP gargs l (Z.push xs) (W.push ws) fl
          (by simpa only [Array.toList_push] using List.rel_append hrel (.cons hx .nil))
          (fun j m hm Z' W' hZ' hr => by
            have := hstep (j + 1) m (by simpa using hm) Z' W' (by simp at hZ' ⊢; omega) hr
            simpa [Nat.add_assoc, Nat.add_comm 1 j] using this)
      refine ⟨fun E fl' h => ?_, fun W' h => ?_⟩
      · rw [List.map_cons, List.foldlM_cons] at h
        obtain ⟨⟨E₁, fl₁⟩, h₁, h⟩ := bind_eq_ok.1 h
        obtain ⟨⟨vs, raised⟩, hv, h₁⟩ := bind_eq_ok.1 h₁
        rw [hsize] at hv
        obtain ⟨xs, ws, hw, hx, rfl, rfl⟩ := hstep₀.1 vs raised hv
        cases h₁
        rw [ConversionStatus.merge_empty, ← Array.map_push] at h
        obtain ⟨Z', W', hW', hr, rfl, rfl⟩ := (ih xs ws hx).1 E fl' h
        refine ⟨Z', W', ?_, hr, rfl, rfl⟩
        rw [runNodes_cons, hW, hw]
        exact hW'
      · rw [runNodes_cons, hW] at h
        obtain ⟨ws, hw, h⟩ := bind_eq_ok.1 h
        obtain ⟨xs, hv, hx⟩ := hstep₀.2 ws hw
        obtain ⟨Z', hZ', hr⟩ := (ih xs ws hx).2 W' h
        refine ⟨Z', ?_, hr⟩
        rw [List.map_cons, List.foldlM_cons]
        refine bind_eq_ok.2 ⟨((Z.push xs).map (·.map F), fl), ?_, hZ'⟩
        refine bind_eq_ok.2 ⟨(xs.map F, {}), by rw [hsize]; exact hv, ?_⟩
        simp [pure, Except.pure, ConversionStatus.merge_empty, Array.map_push]

section Common

variable {d : DType} {ctx : Context (R.Carrier d)}

/-- At the common scalar `d`, a step whose scalar is `d`, or none with Boolean, integer and index
ports only, runs as its node in the graph at `d`: both fail, or both succeed with the node's outputs
at `d` and no flags. -/
private theorem step_some (hctx : R.arithmetic d = some ctx)
    {args' : List (@Op.Value (R.Carrier d) (R.storage d))}
    (hargs : List.Forall₂ (fun a v => a.toOp (some d) = .ok v) args.toList args')
    {i : Nat} {n : IR.Node}
    (hs : p.scalar i = some d ∨
      (p.scalar i = none ∧ (Node.node n).ports.all (fun q => !q.kind.isScalar)))
    (harity : ∀ o refs, n = .op o refs → refs.size = o.1.length)
    (W : Array (List (@Op.Value (R.Carrier d) (R.storage d))))
    (vs : List (Value R)) (fl : ConversionStatus) :
    p.step R P args (W.map (·.map (Value.ofOp (some d)))) i (.node n) = .ok (vs, fl) ↔
      ∃ ws, @IR.Node.eval (R.Carrier d) (R.storage d) ctx (R.capabilities d) (P.graph d)
          args' W i n = .ok ws ∧ vs = ws.map (Value.ofOp (some d)) ∧ fl = {} := by
  cases n with
  | input k port =>
      have hk := forall₂_getElem? hargs k
      rw [Array.getElem?_toList] at hk
      simp only [step, IR.Node.eval]
      rcases hs with hs | ⟨hs, hp⟩
      · rw [hs]
        generalize args[k]? = oa at hk ⊢
        generalize args'[k]? = ov at hk ⊢
        cases hk with
        | none => simp [throw, throwThe, MonadExceptOf.throw]
        | some ha =>
            rename_i a v
            constructor
            · intro h
              obtain ⟨v', hv', h⟩ := bind_eq_ok.1 h
              rw [ha] at hv'
              cases hv'
              obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
              cases h
              exact ⟨_, bind_eq_ok.2 ⟨x, hx, rfl⟩, rfl, rfl⟩
            · rintro ⟨ws, h, rfl, rfl⟩
              obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
              cases h
              exact bind_eq_ok.2 ⟨_, ha, bind_eq_ok.2 ⟨_, hx, rfl⟩⟩
      · rw [hs]
        have hp' : port.kind.isScalar = false := by simpa [Node.ports] using hp
        generalize args[k]? = oa at hk ⊢
        generalize args'[k]? = ov at hk ⊢
        cases hk with
        | none => simp [throw, throwThe, MonadExceptOf.throw]
        | some ha =>
            rename_i a v
            constructor
            · intro h
              obtain ⟨z, hz, h⟩ := bind_eq_ok.1 h
              obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
              cases h
              obtain ⟨y, hy, hxy⟩ :=
                Value.read_native hp' (e' := some d) (bind_eq_ok.2 ⟨z, hz, hx⟩)
              obtain ⟨v', hv', hy⟩ := bind_eq_ok.1 hy
              rw [ha] at hv'
              cases hv'
              exact ⟨_, bind_eq_ok.2 ⟨y, hy, rfl⟩, by simp [Value.ofOp_native hxy], rfl⟩
            · rintro ⟨ws, h, rfl, rfl⟩
              obtain ⟨y, hy, h⟩ := bind_eq_ok.1 h
              cases h
              obtain ⟨x, hx, hyx⟩ :=
                Value.read_native hp' (e' := none) (bind_eq_ok.2 ⟨_, ha, hy⟩)
              obtain ⟨z, hz, hx⟩ := bind_eq_ok.1 hx
              refine bind_eq_ok.2 ⟨_, hz, bind_eq_ok.2 ⟨_, hx, ?_⟩⟩
              simp [Value.ofOp_native hyx, pure, Except.pure]
  | const port =>
      simp only [step, IR.Node.eval, Payload.graph]
      rcases hs with hs | ⟨hs, hp⟩
      · rw [hs]
        cases P.consts[i]?.join with
        | none => simp [throw, throwThe, MonadExceptOf.throw]
        | some a =>
            cases ha : a.toOp (some d) with
            | error e =>
                simp only [Option.bind_some, ha, Except.toOption]
                constructor
                · intro h
                  obtain ⟨_, h, -⟩ := bind_eq_ok.1 h
                  cases h
                · rintro ⟨ws, h, -⟩
                  cases h
            | ok v =>
                simp only [Option.bind_some, ha, Except.toOption]
                constructor
                · intro h
                  obtain ⟨v', hv', h⟩ := bind_eq_ok.1 h
                  cases hv'
                  obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
                  cases h
                  exact ⟨_, bind_eq_ok.2 ⟨_, hx, rfl⟩, rfl, rfl⟩
                · rintro ⟨ws, h, rfl, rfl⟩
                  obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
                  cases h
                  exact bind_eq_ok.2 ⟨_, rfl, bind_eq_ok.2 ⟨_, hx, rfl⟩⟩
      · rw [hs]
        have hp' : port.kind.isScalar = false := by simpa [Node.ports] using hp
        cases P.consts[i]?.join with
        | none => simp [throw, throwThe, MonadExceptOf.throw]
        | some a =>
            simp only [Option.bind_some]
            constructor
            · intro h
              obtain ⟨z, hz, h⟩ := bind_eq_ok.1 h
              obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
              cases h
              obtain ⟨y, hy, hxy⟩ :=
                Value.read_native hp' (e' := some d) (bind_eq_ok.2 ⟨z, hz, hx⟩)
              obtain ⟨v, hv, hy⟩ := bind_eq_ok.1 hy
              rw [hv]
              exact ⟨_, bind_eq_ok.2 ⟨y, hy, rfl⟩, by simp [Value.ofOp_native hxy], rfl⟩
            · rintro ⟨ws, h, rfl, rfl⟩
              cases ha : a.toOp (some d) with
              | error e =>
                  simp only [ha, Except.toOption] at h
                  cases h
              | ok v =>
                  simp only [ha, Except.toOption] at h
                  obtain ⟨y, hy, h⟩ := bind_eq_ok.1 h
                  cases h
                  obtain ⟨x, hx, hyx⟩ :=
                    Value.read_native hp' (e' := none) (bind_eq_ok.2 ⟨_, ha, hy⟩)
                  obtain ⟨z, hz, hx⟩ := bind_eq_ok.1 hx
                  refine bind_eq_ok.2 ⟨_, hz, bind_eq_ok.2 ⟨_, hx, ?_⟩⟩
                  simp [Value.ofOp_native hyx, pure, Except.pure]
  | op o refs =>
      rcases o with ⟨ins, outs, op⟩
      let _ : Storage (R.Carrier d) := R.storage d
      let _ : Context (R.Carrier d) := ctx
      let _ : Op.Capabilities (R.Carrier d) := R.capabilities d
      have hread : ∀ r : Ref,
          (do let v ← r.read (W.map (·.map (Value.ofOp (some d)))); v.toOp (some d)) =
            r.read W := by
        intro r
        rw [Ref.read_map]
        cases r.read W with
        | error e => rfl
        | ok w => exact Value.toOp_ofOp (some d) w
      simp only [step, IR.Node.eval]
      rcases hs with hs | ⟨hs, hp⟩
      · rw [hs]
        by_cases hc : ∃ (name : String) (shapes : List Shape) (output : Shape),
            ins = shapes.map Port.value ∧
            HEq (⟨outs, op⟩ : Σ outs, Op ins outs)
              (⟨[Port.value output], Op.custom name shapes output⟩ :
                Σ outs, Op (shapes.map Port.value) outs)
        · obtain ⟨name, shapes, output, rfl, hh⟩ := hc
          obtain ⟨rfl, hop⟩ := Sigma.mk.inj_iff.1 (eq_of_heq hh)
          cases hop
          have harity' : refs.size = shapes.length := by
            simpa using harity _ refs rfl
          have hread' : ∀ r : Ref,
              (do
                let v ← r.read (W.map (·.map (Value.ofOp (some d))))
                let w ← v.toOp (some d)
                someTensor? w) = (do let w ← r.read W; someTensor? w) := by
            intro r
            rw [← hread r, bind_assoc]
          rw [runOp.eq_1, evalOp.eq_1]
          simp only [runCustom, Payload.graph, harity', hread', ↓reduceIte]
          cases P.customs[i]?.join with
          | none => simp [throw, throwThe, MonadExceptOf.throw]
          | some c =>
              by_cases h : c.scalar = d
              · subst h
                simp only [Option.bind_some, dite_true]
                constructor
                · intro h₁
                  obtain ⟨rs, h₂, h₃⟩ := bind_eq_ok.1 h₁
                  cases h₃
                  obtain ⟨vals, hm, h₄⟩ := bind_eq_ok.1 h₂
                  obtain ⟨x, hx, h₅⟩ := bind_eq_ok.1 h₄
                  cases h₅
                  exact ⟨_, bind_eq_ok.2 ⟨x, bind_eq_ok.2 ⟨vals, hm, hx⟩, rfl⟩,
                    rfl, rfl⟩
                · rintro ⟨ws, h₁, rfl, rfl⟩
                  obtain ⟨x, h₂, h₃⟩ := bind_eq_ok.1 h₁
                  cases h₃
                  obtain ⟨vals, hm, hx⟩ := bind_eq_ok.1 h₂
                  exact bind_eq_ok.2 ⟨_, bind_eq_ok.2 ⟨vals, hm,
                    bind_eq_ok.2 ⟨x, hx, rfl⟩⟩, rfl⟩
              · simp [h, throw, throwThe, MonadExceptOf.throw]
        · have hc' : ∀ (name : String) (shapes : List Shape) (output : Shape),
              ins = shapes.map Port.value →
              HEq (⟨outs, op⟩ : Σ outs, Op ins outs)
                (⟨[Port.value output], Op.custom name shapes output⟩ :
                  Σ outs, Op (shapes.map Port.value) outs) → False :=
            fun name shapes output h₁ h₂ => hc ⟨name, shapes, output, h₁, h₂⟩
          rw [runOp.eq_2 _ _ _ _ _ _ _ _ _ hc', evalOp.eq_2 _ _ _ _ _ _ _ hc']
          simp only [hread, hctx]
          constructor
          · intro h₁
            obtain ⟨rs, h₂, h₃⟩ := bind_eq_ok.1 h₁
            cases h₃
            obtain ⟨ws, hws, h₄⟩ := bind_eq_ok.1 h₂
            split at h₄
            · cases h₄
            · refine ⟨rs, ?_, rfl, rfl⟩
              rw [hws]
              exact h₄
          · rintro ⟨ws, h₁, rfl, rfl⟩
            obtain ⟨values, xs, rule, -, hi, -, hs', -, -⟩ := Op.invoke_eq_ok h₁
            refine bind_eq_ok.2 ⟨ws, bind_eq_ok.2 ⟨values, hi, ?_⟩, rfl⟩
            rw [hs']
            rw [hi] at h₁
            exact h₁
      · rw [hs]
        have hnat : (ins ++ outs).all (fun q => !q.kind.isScalar) := by
          simpa [Node.ports] using hp
        rw [runOp.eq_3, evalOp.eq_2 _ _ _ _ _ _ _ (not_custom_of_native hnat)]
        have hread₀ : ∀ r : Ref,
            (do let v ← r.read (W.map (·.map (Value.ofOp (some d)))); v.toOp none) =
              (r.read W >>= fun w => (Value.ofOp (some d) w).toOp none) := by
          intro r
          rw [Ref.read_map]
          cases r.read W <;> rfl
        simp only [hread₀]
        have key := runStorage_invoke (R := R) (i := i) op hnat
          (xp := refs.toList.mapM fun r => r.read W >>= fun w => (Value.ofOp (some d) w).toOp none)
          (xg := refs.toList.mapM (·.read W))
          (fun zs h => by
            obtain ⟨ws, hws, hr⟩ := mapM_bind_eq_ok.1 h
            refine ⟨ws, hws, ?_⟩
            exact (hr.imp fun w z h => (Value.toOp_ofOp_none d w z).1 h).flip)
          (fun ws hws hnw => by
            obtain ⟨zs, hzs, hr⟩ := Value.mapM_toOp_ofOp_none d hnw
            exact ⟨zs, mapM_bind_eq_ok.2 ⟨ws, hws, hzs⟩, hr⟩)
        constructor
        · intro h₁
          obtain ⟨rs, h₂, h₃⟩ := bind_eq_ok.1 h₁
          cases h₃
          obtain ⟨rs', h₄, hr⟩ := key.1 rs h₂
          exact ⟨rs', h₄, Value.map_ofOp_native hr, rfl⟩
        · rintro ⟨ws, h₁, rfl, rfl⟩
          obtain ⟨rs, h₂, hr⟩ := key.2 ws h₁
          refine bind_eq_ok.2 ⟨rs, h₂, ?_⟩
          simp [Value.map_ofOp_native (e := none) (e' := some d) hr, pure, Except.pure]

end Common

section Native

variable {α : Type} [Storage α] [Context α] [Op.Capabilities α]

/-- A step without a scalar, whose ports are Boolean, integer and index ports, runs as its node in
a graph at any scalar type `α`: from outputs related row by row to the graph's, both fail, or both
succeed with the same Boolean, integer and index values and no flags. -/
private theorem step_none {gargs : List (Op.Value α)}
    (hargs : ∀ k : Nat, Option.Rel (fun a w => ∃ z, a.toOp none = .ok z ∧ Op.NativeRel z w)
      args[k]? gargs[k]?)
    {GP : IR.Payload α} {i : Nat} {n : IR.Node}
    (hGP : ∀ port, n = .const port → Option.Rel Op.NativeRel
      ((P.consts[i]?.join).bind fun v => (v.toOp none).toOption) (GP.const? i))
    (hs : p.scalar i = none)
    (hp : (Node.node n).ports.all (fun q => !q.kind.isScalar))
    {Z : Array (List (@Op.Value (R.CarrierAt none) (R.storageAt none)))}
    {W : Array (List (Op.Value α))}
    (hZW : List.Forall₂ (List.Forall₂ Op.NativeRel) Z.toList W.toList) :
    (∀ vs fl, p.step R P args (Z.map (·.map (Value.ofOp none))) i (.node n) = .ok (vs, fl) →
      ∃ zs ws, n.eval GP gargs W i = .ok ws ∧ List.Forall₂ Op.NativeRel zs ws ∧
        vs = zs.map (Value.ofOp none) ∧ fl = {}) ∧
    (∀ ws, n.eval GP gargs W i = .ok ws →
      ∃ zs, p.step R P args (Z.map (·.map (Value.ofOp none))) i (.node n) =
          .ok (zs.map (Value.ofOp none), {}) ∧ List.Forall₂ Op.NativeRel zs ws) := by
  cases n with
  | input k port =>
      have hk := hargs k
      simp only [step, IR.Node.eval]
      rw [hs]
      generalize args[k]? = oa at hk ⊢
      generalize gargs[k]? = ow at hk ⊢
      cases hk with
      | none =>
          exact ⟨fun _ _ h => by simp [throw, throwThe, MonadExceptOf.throw] at h,
            fun _ h => by simp [throw, throwThe, MonadExceptOf.throw] at h⟩
      | some ha =>
          rename_i a w
          obtain ⟨z, hz, hzw⟩ := ha
          refine ⟨fun vs fl h => ?_, fun ws h => ?_⟩
          · obtain ⟨z', hz', h⟩ := bind_eq_ok.1 h
            rw [hz] at hz'
            cases hz'
            obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
            cases h
            obtain ⟨y, hy, hxy⟩ := Port.read_native hzw port hx
            exact ⟨_, _, bind_eq_ok.2 ⟨y, hy, rfl⟩, .cons hxy .nil, rfl, rfl⟩
          · obtain ⟨y, hy, h⟩ := bind_eq_ok.1 h
            cases h
            obtain ⟨x, hx, hyx⟩ := Port.read_native hzw.symm port hy
            exact ⟨_, bind_eq_ok.2 ⟨_, hz, bind_eq_ok.2 ⟨_, hx, rfl⟩⟩,
              .cons hyx.symm .nil⟩
  | const port =>
      have hc := hGP port rfl
      simp only [step, IR.Node.eval]
      rw [hs]
      generalize GP.const? i = og at hc ⊢
      cases ha : P.consts[i]?.join with
      | none =>
          rw [ha] at hc
          cases hc
          exact ⟨fun _ _ h => by simp [throw, throwThe, MonadExceptOf.throw] at h,
            fun _ h => by simp [throw, throwThe, MonadExceptOf.throw] at h⟩
      | some a =>
          rw [ha] at hc
          simp only [Option.bind_some] at hc
          cases hz : a.toOp none with
          | error e =>
              rw [hz] at hc
              simp only [Except.toOption] at hc
              cases hc
              refine ⟨fun _ _ h => ?_,
                fun _ h => by simp [throw, throwThe, MonadExceptOf.throw] at h⟩
              obtain ⟨z', hz', -⟩ := bind_eq_ok.1 h
              rw [hz] at hz'
              cases hz'
          | ok z =>
              rw [hz] at hc
              simp only [Except.toOption] at hc
              cases hc with
              | some hzw =>
                  rename_i w
                  refine ⟨fun vs fl h => ?_, fun ws h => ?_⟩
                  · obtain ⟨z', hz', h⟩ := bind_eq_ok.1 h
                    rw [hz] at hz'
                    cases hz'
                    obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
                    cases h
                    obtain ⟨y, hy, hxy⟩ := Port.read_native hzw port hx
                    exact ⟨_, _, bind_eq_ok.2 ⟨y, hy, rfl⟩, .cons hxy .nil, rfl, rfl⟩
                  · obtain ⟨y, hy, h⟩ := bind_eq_ok.1 h
                    cases h
                    obtain ⟨x, hx, hyx⟩ := Port.read_native hzw.symm port hy
                    exact ⟨_, bind_eq_ok.2 ⟨_, hz, bind_eq_ok.2 ⟨_, hx, rfl⟩⟩,
                      .cons hyx.symm .nil⟩
  | op o refs =>
      rcases o with ⟨ins, outs, op⟩
      have hnat : (ins ++ outs).all (fun q => !q.kind.isScalar) := by
        simpa [Node.ports] using hp
      simp only [step, IR.Node.eval]
      rw [hs, runOp.eq_3, evalOp.eq_2 _ _ _ _ _ _ _ (not_custom_of_native hnat)]
      have hread : ∀ r : Ref,
          (do let v ← r.read (Z.map (·.map (Value.ofOp none))); v.toOp none) = r.read Z := by
        intro r
        rw [Ref.read_map]
        cases r.read Z with
        | error e => rfl
        | ok z => exact Value.toOp_ofOp none z
      simp only [hread]
      obtain ⟨hm₁, hm₂⟩ := Ref.mapM_read_rel hZW refs.toList
      have key := runStorage_invoke (R := R) (i := i) op hnat
        (xp := refs.toList.mapM (·.read Z)) (xg := refs.toList.mapM (·.read W)) hm₁
        (fun ws hws _ => hm₂ ws hws)
      refine ⟨fun vs fl h => ?_, fun ws h => ?_⟩
      · obtain ⟨rs, h₂, h₃⟩ := bind_eq_ok.1 h
        cases h₃
        obtain ⟨rs', h₄, hr⟩ := key.1 rs h₂
        exact ⟨rs, rs', h₄, hr, rfl, rfl⟩
      · obtain ⟨rs, h₂, hr⟩ := key.2 ws h
        exact ⟨rs, bind_eq_ok.2 ⟨rs, h₂, rfl⟩, hr⟩

end Native

/-- Outputs related by equality row by row are equal. -/
private theorem eq_of_forall₂_eq {X : Type} {Z W : Array (List X)}
    (h : List.Forall₂ (List.Forall₂ Eq) Z.toList W.toList) : Z = W := by
  simp only [List.forall₂_eq_eq_eq] at h
  exact Array.toList_inj.1 h

/-- A checked program whose steps form the graph `g` at one scalar `d` with arithmetic `ctx` runs as
`g` at `d`, with the payload's constants and registered custom programs at `d` and the arguments
read at `d`: the outputs of every step are the graph's at `d`, and no flag is raised. -/
theorem run_graph?_ok {g : Graph} {d : DType} {ctx : Context (R.Carrier d)}
    (hg : p.graph? = some (g, some d)) (hctx : R.arithmetic d = some ctx)
    (hcheck : p.check R = .ok ()) (hP : P.check p = .ok ()) (hsize : args.size = p.arity)
    {args' : List (@Op.Value (R.Carrier d) (R.storage d))}
    (hargs : args.toList.mapM (·.toOp (some d)) = .ok args')
    (vs : Array (List (Value R))) (flags : ConversionStatus) :
    p.run R P args = .ok (vs, flags) ↔
      ∃ ws, @Graph.run (R.Carrier d) (R.storage d) ctx (R.capabilities d) g (P.graph d) args' =
          .ok ws ∧ vs = ws.map (·.map (Value.ofOp (some d))) ∧ flags = {} := by
  obtain ⟨hnodes, -, hsc⟩ := graph?_eq_some hg
  have hlist : p.nodes.toList = g.nodes.toList.map Node.node := by
    rw [hnodes, Array.toList_map]
  let _ : Storage (R.Carrier d) := R.storage d
  let _ : Context (R.Carrier d) := ctx
  let _ : Op.Capabilities (R.Carrier d) := R.capabilities d
  have key := run_nodes (p := p) (P := P) (args := args) (Value.ofOp (some d)) Eq (P.graph d)
    args' g.nodes.toList #[] #[] {} .nil ?_
  · rw [run_eq_foldlM hcheck hP hsize, hlist]
    obtain ⟨h₁, h₂⟩ := key
    constructor
    · intro h
      obtain ⟨Z', W', hW', hr, rfl, rfl⟩ := h₁ vs flags (by simpa using h)
      obtain rfl := eq_of_forall₂_eq hr
      exact ⟨Z', hW', rfl, rfl⟩
    · rintro ⟨ws, hw, rfl, rfl⟩
      obtain ⟨Z', hZ', hr⟩ := h₂ ws hw
      obtain rfl := eq_of_forall₂_eq hr
      simpa using hZ'
  · intro j n hn Z' W' hZ' hr
    obtain rfl := eq_of_forall₂_eq hr
    simp only [Array.size_empty, Nat.zero_add] at hZ' ⊢
    have hpn : p.nodes[j]? = some (.node n) := by
      rw [hnodes, Array.getElem?_map, ← Array.getElem?_toList, hn]
      rfl
    obtain ⟨hcs, hcn⟩ := check_step hcheck hpn
    have hs : p.scalar j = some d ∨
        (p.scalar j = none ∧ (Node.node n).ports.all (fun q => !q.kind.isScalar)) := by
      cases he : p.scalar j with
      | none => exact .inr ⟨rfl, checkScalar_none hcs he⟩
      | some e =>
          cases hsc j e he
          exact .inl rfl
    have harity : ∀ o refs, n = .op o refs → refs.size = o.1.length := by
      intro o refs hno
      subst hno
      exact checkNode_op hcn
    have hstep := step_some (P := P) hctx (mapM_eq_ok_iff.1 hargs) hs harity Z'
    refine ⟨fun vs fl h => ?_, fun ws h => ?_⟩
    · obtain ⟨ws, hw, rfl, rfl⟩ := (hstep vs fl).1 h
      exact ⟨ws, ws, hw, List.forall₂_same.2 fun _ _ => rfl, rfl, rfl⟩
    · exact ⟨ws, (hstep _ _).2 ⟨ws, h, rfl, rfl⟩, List.forall₂_same.2 fun _ _ => rfl⟩

/-- A checked program whose steps form the graph `g` without a scalar runs as `g` at any scalar
type `α`, with constants at `g`'s constant nodes and arguments related to the program's: the
outputs of every step are the graph's, the same Boolean, integer and index values, and no flag is
raised. -/
theorem run_graph?_none {g : Graph} (hg : p.graph? = some (g, none))
    (hcheck : p.check R = .ok ()) (hP : P.check p = .ok ()) (hsize : args.size = p.arity)
    {α : Type} [Storage α] [Context α] [Op.Capabilities α] {GP : IR.Payload α}
    (hGP : ∀ j port, g.nodes[j]? = some (.const port) → Option.Rel Op.NativeRel
      ((P.consts[j]?.join).bind fun v => (v.toOp none).toOption) (GP.const? j))
    {args₀ : List (@Op.Value (R.CarrierAt none) (R.storageAt none))}
    (hargs : args.toList.mapM (·.toOp none) = .ok args₀)
    {gargs : List (Op.Value α)} (hrel : List.Forall₂ Op.NativeRel args₀ gargs)
    (vs : Array (List (Value R))) (flags : ConversionStatus) :
    p.run R P args = .ok (vs, flags) ↔
      ∃ (zs : Array (List (@Op.Value (R.CarrierAt none) (R.storageAt none))))
        (ws : Array (List (Op.Value α))), g.run GP gargs = .ok ws ∧
        List.Forall₂ (List.Forall₂ Op.NativeRel) zs.toList ws.toList ∧
        vs = zs.map (·.map (Value.ofOp none)) ∧ flags = {} := by
  obtain ⟨hnodes, -, hsc⟩ := graph?_eq_some hg
  have hlist : p.nodes.toList = g.nodes.toList.map Node.node := by
    rw [hnodes, Array.toList_map]
  have hargs' : ∀ k : Nat, Option.Rel (fun a w => ∃ z, a.toOp none = .ok z ∧ Op.NativeRel z w)
      args[k]? gargs[k]? := by
    intro k
    have h₁ := forall₂_getElem? (mapM_eq_ok_iff.1 hargs) k
    have h₂ := forall₂_getElem? hrel k
    rw [Array.getElem?_toList] at h₁
    generalize args[k]? = oa at h₁ ⊢
    generalize args₀[k]? = oz at h₁ h₂
    generalize gargs[k]? = ow at h₂ ⊢
    cases h₁ with
    | none =>
        cases h₂
        exact .none
    | some ha =>
        cases h₂ with
        | some hzw => exact .some ⟨_, ha, hzw⟩
  have key := run_nodes (p := p) (P := P) (args := args) (Value.ofOp none) Op.NativeRel GP gargs
    g.nodes.toList #[] #[] {} .nil ?_
  · rw [run_eq_foldlM hcheck hP hsize, hlist]
    obtain ⟨h₁, h₂⟩ := key
    constructor
    · intro h
      obtain ⟨Z', W', hW', hr, rfl, rfl⟩ := h₁ vs flags (by simpa using h)
      exact ⟨Z', W', hW', hr, rfl, rfl⟩
    · rintro ⟨zs, ws, hw, hr, rfl, rfl⟩
      obtain ⟨Z', hZ', hr'⟩ := h₂ ws hw
      have hu : Relator.LeftUnique (@Op.NativeRel (R.CarrierAt none) α (R.storageAt none) ‹_›) :=
        fun _ _ _ => Op.NativeRel.unique
      obtain rfl := Array.toList_inj.1 (hu.forall₂.forall₂ hr' hr)
      simpa using hZ'
  · intro j n hn Z' W' hZ' hr
    simp only [Array.size_empty, Nat.zero_add] at hZ' ⊢
    have hpn : p.nodes[j]? = some (.node n) := by
      rw [hnodes, Array.getElem?_map, ← Array.getElem?_toList, hn]
      rfl
    obtain ⟨hcs, -⟩ := check_step hcheck hpn
    have hs : p.scalar j = none := by
      cases he : p.scalar j with
      | none => rfl
      | some e => cases hsc j e he
    exact step_none hargs' (fun port h => hGP j port (by rw [← Array.getElem?_toList, hn, h]))
      hs (checkScalar_none hcs hs) hr

end Graph

/-- A step of `Op.bitcast` between integer words of one width retypes its operand's words. -/
theorem step_bitcast_integer {R : Machine} {q : Program} {P : Payload R} {args : Array (Value R)}
    {earlier : Array (List (Value R))} {i : Nat} {signed signed' : Bool} {width : Nat}
    {s : Shape} {r : Ref} (hsc : q.scalar i = none)
    {x : @Port.Carrier (R.CarrierAt none) (R.storageAt none) (.integer signed width s)}
    (hr : r.read earlier = .ok ⟨⟨.integer signed width s, none⟩, x⟩) :
    q.step R P args earlier i
        (.node (.op ⟨_, _, Op.bitcast (s := s) width (.integer signed)
          (.integer signed')⟩ #[r])) =
      .ok ([⟨⟨.integer signed' width s, none⟩, x⟩], {}) := by
  have hf : (Op.bitcast (s := s) width (.integer signed)
      (.integer signed')).storageForward? = some fun _ _ xs => pure (xs.1, ()) := rfl
  have hc : (Op.bitcast (s := s) width (.integer signed)
      (.integer signed')).check = .ok () := rfl
  simp only [step]
  rw [hsc]
  simp only [runOp, runStorage, hf, hc]
  simp [hr, Value.toOp, Values.ofErased, Values.toErased, bind, Except.bind, pure, Except.pure]
  erw [Port.read_self]
  rfl

/-- A step of `Op.bitcast` from the scalar to integer words at the scalar `e`, whose words have
`width` bits, reads every entry's word. -/
theorem step_bitcast_scalar_integer {R : Machine} {q : Program} {P : Payload R}
    {args : Array (Value R)} {earlier : Array (List (Value R))} {i : Nat} {e : DType}
    {signed : Bool} {width : Nat} {s : Shape} {r : Ref} (hsc : q.scalar i = some e)
    {ctx : Context (R.Carrier e)} (hctx : R.arithmetic e = some ctx) {B : Bits (R.Carrier e)}
    (hB : (R.capabilities e).bits? = some B) (hw : B.width = width)
    {x : @Tensor (R.Carrier e) s (R.storage e)}
    (hr : r.read earlier = .ok ⟨⟨.value s, some e⟩, x⟩) :
    q.step R P args earlier i
        (.node (.op ⟨_, _, Op.bitcast (s := s) width .scalar (.integer signed)⟩ #[r])) =
      .ok ([⟨⟨.integer signed width s, none⟩,
        @Tensor.map _ (R.storage e) _ _ _ (fun y => (B.toBits y).cast hw) x⟩], {}) := by
  simp only [step]
  rw [hsc]
  simp only [runOp, hctx]
  subst hw
  have hc : (Op.bitcast (s := s) B.width .scalar (.integer signed)).check = .ok () := rfl
  simp [hr, Value.toOp, Op.invoke, hc, Op.rule?, hB, Values.ofErased,
    Values.toErased, Op.Rule.check, Port.read, bind, Except.bind, pure,
    Except.pure]
  rfl

/-- A step of `Op.bitcast` from integer words to the scalar at the scalar `e`, whose words have
`width` bits, reads every word as a value. -/
theorem step_bitcast_integer_scalar {R : Machine} {q : Program} {P : Payload R}
    {args : Array (Value R)} {earlier : Array (List (Value R))} {i : Nat} {e : DType}
    {signed : Bool} {width : Nat} {s : Shape} {r : Ref} (hsc : q.scalar i = some e)
    {ctx : Context (R.Carrier e)} (hctx : R.arithmetic e = some ctx) {B : Bits (R.Carrier e)}
    (hB : (R.capabilities e).bits? = some B) (hw : B.width = width)
    {x : Tensor (BitVec width) s} (hr : r.read earlier = .ok ⟨⟨.integer signed width s, none⟩, x⟩) :
    q.step R P args earlier i
        (.node (.op ⟨_, _, Op.bitcast (s := s) width (.integer signed) .scalar⟩ #[r])) =
      .ok ([⟨⟨.value s, some e⟩,
        @Tensor.map _ _ _ (R.storage e) _ (fun b : BitVec width => B.ofBits (b.cast hw.symm)) x⟩],
        {}) := by
  simp only [step]
  rw [hsc]
  simp only [runOp, hctx]
  subst hw
  have hc : (Op.bitcast (s := s) B.width (.integer signed) .scalar).check = .ok () := rfl
  simp [hr, Value.toOp, Op.invoke, hc, Op.rule?, hB, Values.ofErased,
    Values.toErased, Op.Rule.check, Port.read, bind, Except.bind, pure,
    Except.pure]
  rfl

/-- A step of `Op.decodeIndex` at no scalar, on integer words of positive width, outputs the
index tuples they hold (`Op.wordIndices`) where every bound is positive or the output has no
entries, failing where a component lies outside its bound, and fails without a rule otherwise. -/
theorem step_decodeIndex {R : Machine} {q : Program} {P : Payload R}
    {args : Array (Value R)} {earlier : Array (List (Value R))} {i : Nat} {bounds : List Nat}
    {signed : Bool} {width : Nat} {s : Shape} {r : Ref} (hsc : q.scalar i = none) (hw : 0 < width)
    {x : @Port.Carrier (R.CarrierAt none) (R.storageAt none)
      (.integer signed width (Port.irShape ⟨.index bounds, s⟩))}
    (hr : r.read earlier =
      .ok ⟨⟨.integer signed width (Port.irShape ⟨.index bounds, s⟩), none⟩, x⟩) :
    q.step R P args earlier i
        (.node (.op ⟨_, _, Op.decodeIndex (s := s) bounds signed width⟩ #[r])) =
      if (∀ b ∈ bounds, 0 < b) ∨ s.size = 0 then
        (Op.wordIndices signed x).map fun y => ([⟨⟨.index bounds s, none⟩, y⟩], {})
      else .error "op: no rule (the operator's attribute check fails, or decodeIndex has none at \
        a zero bound with entries)" := by
  have hc : (Op.decodeIndex (s := s) bounds signed width).check = .ok () := by
    simp [Op.check, hw, pure, Except.pure]
  by_cases hb : (∀ b ∈ bounds, 0 < b) ∨ s.size = 0
  · have hf : (Op.decodeIndex (s := s) bounds signed width).storageForward? =
        some (fun _ _ xs => do
          (Op.wordIndices signed xs.1).map fun _ => ()
          pure (Op.wordIndicesOr hb signed xs.1, ())) := by
      unfold Op.storageForward?
      rw [Op.rule?_decodeIndex hb]
      exact dite_eq_left ⟨rfl, rfl⟩
    simp only [step]
    rw [hsc]
    simp only [runOp, runStorage, hf, hc, hb, ↓reduceIte]
    simp [hr, Value.toOp, Values.ofErased, Values.toErased, bind, Except.bind, pure, Except.pure]
    erw [Port.read_self]
    dsimp only
    unfold Op.wordIndicesOr
    cases Op.wordIndices signed x <;> rfl
  · have hf : (Op.decodeIndex (s := s) bounds signed width).storageForward? =
        some (fun _ _ _ => throw "op: no rule (the operator's attribute check fails, or \
          decodeIndex has none at a zero bound with entries)") := by
      unfold Op.storageForward?
      rw [Op.rule?_decodeIndex_eq_none hb]
      exact dite_eq_left ⟨rfl, rfl⟩
    simp only [step]
    rw [hsc]
    simp only [runOp, runStorage, hf, hc, hb, ↓reduceIte]
    simp [hr, Value.toOp, Values.ofErased, Values.toErased, bind, Except.bind, pure, Except.pure]
    all_goals (try erw [Port.read_self])
    all_goals rfl

/-- A step of `Op.encodeIndex` at no scalar, into words of positive width, outputs the words of
the components of the index tuples it reads (`Op.indexWords`), failing where a component does not
fit the integer type (`Op.checkIndexWords`). -/
theorem step_encodeIndex {R : Machine} {q : Program} {P : Payload R} {args : Array (Value R)}
    {earlier : Array (List (Value R))} {i : Nat} {bounds : List Nat} {signed : Bool}
    {width : Nat} {s : Shape} {r : Ref} (hsc : q.scalar i = none) (hw : 0 < width)
    {y : @Port.Carrier (R.CarrierAt none) (R.storageAt none) ⟨.index bounds, s⟩}
    (hr : r.read earlier = .ok ⟨⟨.index bounds s, none⟩, y⟩) :
    q.step R P args earlier i
        (.node (.op ⟨_, _, Op.encodeIndex (s := s) bounds signed width⟩ #[r])) =
      (Op.checkIndexWords signed width y).map fun _ =>
        ([⟨⟨.integer signed width (Port.irShape ⟨.index bounds, s⟩), none⟩,
          Op.indexWords width y⟩], {}) := by
  have hf : (Op.encodeIndex (s := s) bounds signed width).storageForward? =
      some (fun _ _ xs => do
        Op.checkIndexWords signed width xs.1
        pure (Op.indexWords width xs.1, ())) := rfl
  have hc : (Op.encodeIndex (s := s) bounds signed width).check = .ok () := by
    simp [Op.check, hw, pure, Except.pure]
  simp only [step]
  rw [hsc]
  simp only [runOp, runStorage, hf, hc]
  simp [hr, Value.toOp, Values.ofErased, Values.toErased, bind, Except.bind, pure, Except.pure]
  erw [Port.read_self]
  dsimp only
  cases Op.checkIndexWords signed width y <;> rfl

/-- A successful step of `Op.decodeIndex` at no scalar outputs one value and no flags. -/
theorem step_decodeIndex_eq_ok {R : Machine} {q : Program} {P : Payload R}
    {args : Array (Value R)} {earlier : Array (List (Value R))} {i : Nat} {bounds : List Nat}
    {signed : Bool} {width : Nat} {s : Shape} {r : Ref} (hsc : q.scalar i = none)
    {vs : List (Value R)} {raised : ConversionStatus}
    (h : q.step R P args earlier i
      (.node (.op ⟨_, _, Op.decodeIndex (s := s) bounds signed width⟩ #[r])) =
        .ok (vs, raised)) :
    raised = {} ∧ ∃ u, vs = [u] := by
  simp only [step] at h
  rw [hsc] at h
  simp only [runOp, runStorage, Op.storageForward?] at h
  obtain ⟨results, hres, h⟩ := bind_eq_ok.1 h
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  refine ⟨rfl, ?_⟩
  obtain ⟨_, -, hres⟩ := bind_eq_ok.1 hres
  obtain ⟨xs, -, hres⟩ := bind_eq_ok.1 hres
  obtain ⟨ys, -, hres⟩ := bind_eq_ok.1 hres
  obtain ⟨zs, -, hres⟩ := bind_eq_ok.1 hres
  cases hres
  exact ⟨_, rfl⟩

/-- Values read at Boolean, integer and index ports at one scalar exactly where they read there at
another, as related values. -/
theorem mapM_toOp_native {R : Machine} {γ : Type} (g : γ → Except String (Value R))
    (d₁ d₂ : Option DType) :
    ∀ (ps : List Port) (l : List γ) {xs : @Values (R.CarrierAt d₁) (R.storageAt d₁) ps},
      ps.all (fun p => !p.kind.isScalar) →
      (l.mapM (fun c => do (← g c).toOp d₁) >>= @Values.ofErased _ (R.storageAt d₁) ps) =
        .ok xs →
      ∃ ys, (l.mapM (fun c => do (← g c).toOp d₂) >>= @Values.ofErased _ (R.storageAt d₂) ps) =
          .ok ys ∧
        List.Forall₂ (@Op.NativeRel _ _ (R.storageAt d₁) (R.storageAt d₂))
          (@Values.toErased _ (R.storageAt d₁) ps xs) (@Values.toErased _ (R.storageAt d₂) ps ys)
  | [], [], _, _, _ => ⟨(), rfl, .nil⟩
  | [], a :: l, _, _, h => by
    exfalso
    revert h
    rw [List.mapM_cons]
    cases (do (← g a).toOp d₁ : Except String _) <;>
      cases l.mapM (fun c => do (← g c).toOp d₁) <;>
      simp [bind, Except.bind, pure, Except.pure, Values.ofErased]
  | _ :: _, [], _, _, h => by
    simp [bind, Except.bind, pure, Except.pure, Values.ofErased] at h
  | p :: ps, a :: l, ⟨x, xs⟩, hn, h => by
    simp only [List.all_cons, Bool.and_eq_true] at hn
    obtain ⟨ha, hl⟩ := (@mapM_ofErased_cons _ (R.storageAt d₁) _ _ p ps a l x xs).1 h
    have ha' : ((g a >>= (·.toOp d₁)) >>= @Port.read _ (R.storageAt d₁) p) = .ok x := ha
    cases hg : g a with
    | error err => simp [hg, bind, Except.bind] at ha'
    | ok w =>
      simp only [hg, bind, Except.bind] at ha'
      obtain ⟨y, hy, hxy⟩ := Value.read_native (e' := d₂) (by simpa using hn.1) ha'
      obtain ⟨ys, hys, hrel⟩ := mapM_toOp_native g d₁ d₂ ps l hn.2 hl
      refine ⟨(y, ys), (@mapM_ofErased_cons _ (R.storageAt d₂) _ _ p ps a l y ys).2 ⟨?_, hys⟩,
        .cons hxy hrel⟩
      simpa [hg, bind, Except.bind] using hy

/-- An operator step without a scalar, all of whose ports are Boolean, integer or index ports,
succeeds exactly when invoking the operator on its inputs read at any scalar does, raising no
flags. -/
theorem step_op_native {R : Machine} {q : Program} {P : Payload R} {args : Array (Value R)}
    {earlier : Array (List (Value R))} {i : Nat} {e : DType} {ins outs : List Port}
    {o : Op ins outs} {refs : Array Ref} (hsc : q.scalar i = none)
    (hn : (ins ++ outs).all (fun p => !p.kind.isScalar)) (ctx : Context (R.Carrier e))
    {vs : List (Value R)} {flags : FloatLib.Floats.ExecFloat.ConversionStatus} :
    q.step R P args earlier i (.node (.op ⟨ins, outs, o⟩ refs)) = .ok (vs, flags) ↔
      flags = {} ∧ ∃ outputs, @Op.invoke (R.Carrier e) (R.storage e) ctx
          (R.capabilities e) ins outs o
          (refs.toList.mapM fun r => do (← r.read earlier).toOp (some e)) = .ok outputs ∧
        vs = outputs.map (Value.ofOp (some e)) := by
  obtain ⟨f, hf⟩ := Option.isSome_iff_exists.1 (Op.storageForward?_isSome_of_native o hn)
  have hin : ins.all (fun p => !p.kind.isScalar) := by
    simp only [List.all_append, Bool.and_eq_true] at hn
    exact hn.1
  let _ : Storage (R.Carrier e) := R.storage e
  let _ : Context (R.Carrier e) := ctx
  let _ : Op.Capabilities (R.Carrier e) := R.capabilities e
  simp only [step]
  rw [hsc, runOp.eq_3]
  constructor
  · intro h
    cases hN : refs.toList.mapM (fun r => do (← r.read earlier).toOp none) with
    | error err =>
      rw [hN] at h
      cases h
    | ok inputs' =>
      rw [hN] at h
      simp only [bind, Except.bind] at h
      cases hR : runStorage R i none o inputs' with
      | error err =>
        rw [hR] at h
        cases h
      | ok results' =>
        rw [hR] at h
        simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨xs', ys', hc, hx', hy', rfl⟩ := (runStorage_eq_ok hf).1 hR
        obtain ⟨xs, hx, hrel⟩ := mapM_toOp_native (Ref.read earlier) none (some e) ins
          refs.toList hin (by rw [hN]; exact hx')
        cases hS : refs.toList.mapM (fun r => do (← r.read earlier).toOp (some e)) with
        | error err => rw [hS] at hx; cases hx
        | ok inputs =>
          rw [hS] at hx
          obtain ⟨zs, hz, hzrel⟩ := Op.storageForward_native o hf hrel hy'
          exact ⟨rfl, _, (Op.invoke_storageForward o hf).2 ⟨xs, zs, hc, hx, hz, rfl⟩,
            Value.map_ofOp_native hzrel⟩
  · rintro ⟨rfl, outputs, h, rfl⟩
    obtain ⟨inputs, -, -, hc, hS, -, -, -, -⟩ := Op.invoke_eq_ok h
    rw [hS] at h
    obtain ⟨xs, zs, -, hx, hz, rfl⟩ := (Op.invoke_storageForward o hf).1 h
    obtain ⟨xs', hx', hrel⟩ := mapM_toOp_native (Ref.read earlier) (some e) none ins refs.toList hin
      (by rw [hS]; exact hx)
    obtain ⟨ys', hy', hyrel⟩ := Op.storageForward_native o hf hrel hz
    cases hN : refs.toList.mapM (fun r => do (← r.read earlier).toOp none) with
    | error err => rw [hN] at hx'; cases hx'
    | ok inputs' =>
      rw [hN] at hx'
      have hR := (runStorage_eq_ok (R := R) (i := i) hf).2 ⟨xs', ys', hc, hx', hy', rfl⟩
      simp only [bind, Except.bind]
      rw [hR]
      simp only [pure, Except.pure, Value.map_ofOp_native hyrel]

/-- An operator step that is not a custom operation, at a scalar with arithmetic, succeeds exactly
when invoking the operator on its inputs read at that scalar does, raising no flags. -/
theorem step_op_arithmetic {R : Machine} {q : Program} {P : Payload R} {args : Array (Value R)}
    {earlier : Array (List (Value R))} {i : Nat} {e : DType} {ins outs : List Port}
    {o : Op ins outs} {refs : Array Ref} (hsc : q.scalar i = some e)
    {ctx : Context (R.Carrier e)} (hctx : R.arithmetic e = some ctx)
    (hcus : (Node.node (.op ⟨ins, outs, o⟩ refs)).custom? = none)
    {vs : List (Value R)} {flags : ConversionStatus} :
    q.step R P args earlier i (.node (.op ⟨ins, outs, o⟩ refs)) = .ok (vs, flags) ↔
      flags = {} ∧ ∃ outputs, @Op.invoke (R.Carrier e) (R.storage e) ctx
          (R.capabilities e) ins outs o
          (refs.toList.mapM fun r => do (← r.read earlier).toOp (some e)) = .ok outputs ∧
        vs = outputs.map (Value.ofOp (some e)) := by
  have hnc : ∀ (name : String) (shapes : List Shape) (output : Shape),
      ins = shapes.map Port.value →
        (⟨outs, o⟩ : (outs : List Port) × Op ins outs) ≍
          (⟨[Port.value output], Op.custom name shapes output⟩ :
            (outs : List Port) × Op (shapes.map Port.value) outs) → False := by
    intro name shapes output h₁ h₂
    subst h₁
    obtain ⟨rfl, h₃⟩ := Sigma.mk.inj (eq_of_heq h₂)
    cases eq_of_heq h₃
    simp [Node.custom?] at hcus
  simp only [step]
  rw [hsc, runOp.eq_2 _ _ _ _ _ _ _ _ _ hnc]
  simp only [hctx]
  let _ : Storage (R.Carrier e) := R.storage e
  let _ : Context (R.Carrier e) := ctx
  let _ : Op.Capabilities (R.Carrier e) := R.capabilities e
  cases hm : refs.toList.mapM (fun r => do (← r.read earlier).toOp (some e)) with
  | error err =>
    simp only [bind, Except.bind, reduceCtorEq, false_iff, not_and, not_exists]
    intro _ outputs h
    obtain ⟨_, _, _, _, hi, _⟩ := Op.invoke_eq_ok h
    cases hi
  | ok inputs =>
    simp only [bind, Except.bind]
    by_cases hs : (o.rule? (α := R.Carrier e)).isNone = true
    · simp only [hs, ite_true, throw, throwThe, MonadExceptOf.throw, reduceCtorEq, false_iff,
        not_and, not_exists]
      intro _ outputs h
      obtain ⟨_, _, rule, _, _, _, hr, _⟩ := Op.invoke_eq_ok h
      simp [hr] at hs
    · simp only [hs, pure, Except.pure]
      cases o.invoke (.ok inputs) <;> simp [eq_comm, and_comm]

end IR.Program

end TorchLean
