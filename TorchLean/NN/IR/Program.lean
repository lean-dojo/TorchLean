/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Eval
public import TorchLean.NN.IR.Scalar
public import TorchLean.Tensor.Op.Storage

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

`TorchLean.NN.IR.Program.Graph` runs a program whose steps are all operator-graph nodes, with one
scalar or none, as its operator graph, and `TorchLean.NN.IR.Program.Step` states the steps of
`bitcast`, `decodeIndex` and `encodeIndex` and an operator step as the operator's invocation.
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

end IR.Program

end TorchLean
