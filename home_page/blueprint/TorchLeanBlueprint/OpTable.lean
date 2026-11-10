import VersoManual
-- Every module that registers an operator certificate, so that the gradient cells and
-- `opCertKinds` see all of them, and the definitions the other cells read.
import TorchLean.NN.Op.Certificates
import TorchLean.NN.Op.Pointwise
import TorchLean.NN.Op.Softmax
import TorchLean.NN.Op.LayerNorm
import TorchLean.NN.Op.BatchNorm
import TorchLean.Verify.NumericalCertificate.Enclosure
import TorchLean.Tensor.Op.Need
import TorchLean.NN.LibTorch.Kernel
import TorchLean.Verify.CROWN.Op.Cert
import TorchLean.Verify.CROWN.Op.ExecCROWN

/-!
# Operator tables read off the definitions

The `operators` code block renders a table with one row for each operator it names, a constructor
of `TorchLean.Op`. Every cell of a row is read off a definition at that constructor while the page
is elaborated:

* Rule: the capability of the scalar that `Op.rule?` reads for the rule (`Op.need`), and the
  attributes at which the operator has no rule (`Op.hasRule`).
* CUDA: whether the LibTorch table has an entry for the operator
  (`TorchLean.LibTorch.Kernel.ofOp`).
* Gradient: the kinds of the certificates registered for the pullback at each value input
  (`TorchLean.Op.certificates`), and whether the rule has a forward mode (`Rule.jvp?`), with the
  kinds of the certificates registered for it.
* Bounds: which bound propagations have an arm for the operator: the certificate checker
  (`TorchLean.CROWN.Op.cert?`), interval propagation (`TorchLean.CROWN.Op.ibp?`) and the
  backward sweep of CROWN (`TorchLean.CROWN.Op.backwardRule`), and whether an `enclosure`
  certificate is registered for the operator.

A cell is computed by reducing the definition at the constructor applied to fresh attributes, with
every element attribute at the model's scalar. A match on an attribute splits the attribute into
its constructors, so a cell says whether a property holds at every attribute, at none or at some.
A bound propagation has an arm for an operator when its definition returns a result at some
attributes and inputs. A definition that does not reduce to a decided cell stops the build with
the operator's name.

The role `opListed` checks that the `operators` blocks of a page name every constructor and gives
their number, `opCount` gives the number of constructors, and `opCertKinds` the number of
registered certificates of each kind.
-/

namespace TorchLeanBlueprint.OpTable

open Lean Meta

/-- What a reduction ends at: a constructor, an input, a declaration it stops at, or a term it is
stuck on, with the reason. -/
inductive Leaf where
  | ctor (name : Name)
  | data
  | stop (name : Name)
  | unknown (why : String)
  deriving Inhabited, BEq

/-- The leaf, as a phrase of an error message. -/
def Leaf.text : Leaf → String
  | .ctor n => s!"the constructor {n}"
  | .data => "an input"
  | .stop n => s!"a call of {n}"
  | .unknown why => why

/-- A reduction: a result, a split over the values of an attribute, where every child is a region
of the attributes, or a split over the inputs, where the children are alternatives. -/
inductive Tree where
  | leaf (l : Leaf)
  | attr (children : Array Tree)
  | data (children : Array Tree)
  deriving Inhabited

/-- The leaves of a reduction. -/
partial def Tree.leaves : Tree → Array Leaf
  | .leaf l => #[l]
  | .attr cs | .data cs => cs.foldl (· ++ ·.leaves) #[]

/-- Where a property of the result holds: at every attribute, at none, at some. -/
inductive Cover where
  | every
  | never
  | some
  | vacuous
  | unknown (why : String)
  deriving Inhabited, BEq

/-- The cover, as a phrase of an error message. -/
def Cover.text : Cover → String
  | .every => "at every attribute"
  | .never => "at no attribute"
  | .some => "at some attributes"
  | .vacuous => "at no attribute its hypotheses allow"
  | .unknown why => why

/-- Some input has the property: an input split holds where one of its children holds. -/
partial def Tree.cover (pos : Leaf → Option Bool) : Tree → Cover
  | .leaf l =>
    match pos l with
    | Option.some true => .every
    | Option.some false => .never
    | none => .unknown l.text
  | .attr cs =>
    let rs := (cs.map (·.cover pos)).filter (· != .vacuous)
    if rs.isEmpty then .vacuous else
    if rs.contains .some || rs.contains .every && rs.contains .never then .some
    else match rs.find? (· matches .unknown _) with
      | Option.some u => u
      | none => if rs.all (· == .never) then .never else .every
  | .data cs =>
    let rs := (cs.map (·.cover pos)).filter (· != .vacuous)
    if rs.isEmpty then .vacuous else
    if rs.contains .every then .every
    else if rs.all (· == .never) then .never
    else if (rs.filter (· == .some)).size == 1 && rs.all (fun r => r == .some || r == .never) then
      .some
    else match rs.find? (· matches .unknown _) with
      | Option.some u => u
      | none => .unknown "two inputs at some attributes"

/-- Some leaf has the property: a leaf where it holds, every leaf where it fails, or unknown. -/
partial def Tree.reach (pos : Leaf → Option Bool) : Tree → Cover
  | .leaf l =>
    match pos l with
    | Option.some true => .every
    | Option.some false => .never
    | none => .unknown l.text
  | .attr cs | .data cs =>
    let rs := cs.map (·.reach pos)
    if rs.contains .every then .every
    else match rs.find? (· matches .unknown _) with
      | Option.some u => u
      | none => if rs.isEmpty then .vacuous else .never

/-- The context of a reduction. -/
structure Ctx where
  /-- The inputs: the references of a node and the fields of a split over inputs. -/
  data : FVarIdSet
  /-- The declarations the reduction stops at. -/
  stops : NameSet := {}
  /-- The property whose cover is wanted: the exploration stops where the cover is decided. -/
  pos : Option (Leaf → Option Bool) := none
  /-- The heartbeats the exploration may use. -/
  budget : Nat := 20000000
  /-- The heartbeats at the start of the exploration. -/
  start : Nat := 0
  /-- What the attributes satisfy: the propositions among them, and `0 < n` for one in `Fin n`. -/
  facts : Array Expr := #[]
  /-- Whether the property is wanted at some attributes and some inputs rather than at every
  attribute for some input. -/
  reach : Bool := false

/-- The counters of a reduction, and the depth of each attribute a split introduced. -/
structure St where
  fuel : Nat := 50000
  leaves : Nat := 0
  depth : FVarIdMap Nat := {}

abbrev M := ReaderT Ctx (StateRefT St MetaM)

/-- What a term is stuck on: an input, an attribute, or neither, with the reason. -/
inductive Reason where
  | data
  | attr (x : FVarId)
  | none (why : String)

/-- The loops the explorer keeps folded and reads by their bodies. -/
def loops : NameSet := NameSet.empty |>.insert ``Vector.ofFnM |>.insert ``Array.ofFnM

/-- One delta step at the head of `e`: `unfoldDefinition?`, and otherwise the definition's value,
for the matchers and the recursive definitions whose smart unfolding is stuck. -/
def delta? (e : Expr) : MetaM (Option Expr) := do
  if let some e' ← unfoldDefinition? e then return some e'
  let .const name us := e.getAppFn | return none
  if loops.contains name then return none
  let some info@(.defnInfo _) := (← getEnv).find? name | return none
  return some ((← instantiateValueLevelParams info us).betaRev e.getAppRevArgs)

/-- `whnf`, with a matcher it leaves stuck at its restricted transparency unfolded and reduced
again, and a cast along an equation whose sides do not reduce read as its value. -/
partial def whnfM (e : Expr) (fuel : Nat := 16) : MetaM Expr := do
  let e ← whnf e
  if fuel == 0 then return e
  let .const name _ := e.getAppFn | return e
  if let some (.recInfo r) := (← getEnv).find? name then
    let args := e.getAppArgs
    if r.k && r.numMinors == 1 && args.size > r.getMajorIdx then
      if let some minor := args[r.numParams + r.numMotives]? then
        return ← whnfM (mkAppN minor args[r.getMajorIdx + 1:]) (fuel - 1)
    return e
  unless ← isMatcher name do return e
  let some e' ← delta? e | return e
  whnfM e' (fuel - 1)

/-- What the reduction of `e` is stuck on. -/
partial def reason (e : Expr) (fuel : Nat := 256) : M Reason := do
  if fuel == 0 then return .none "reason fuel"
  let e ← whnfCore e
  match e.getAppFn with
  | .fvar x => return if (← read).data.contains x then .data else .attr x
  | .proj _ i b =>
    let b ← whnfM b
    if let some field ← projectCore? b i then
      return ← reason (mkAppN field e.getAppArgs) (fuel - 1)
    reason b (fuel - 1)
  | .const name _ =>
    match (← getEnv).find? name with
    | some (.recInfo r) =>
      let args := e.getAppArgs
      -- a cast along an equation whose sides do not reduce: its value
      if r.k && r.numMinors == 1 then
        if let some minor := args[r.numParams + r.numMotives]? then
          return ← reason (mkAppN minor args[r.getMajorIdx + 1:]) (fuel - 1)
      match args[r.getMajorIdx]? with
      | some major =>
        match ← whnfM major with
        | .lit (.natVal n) =>
          -- a numeral major: the recursor at the constructor it stands for
          let c := if n == 0 then mkConst ``Nat.zero
            else mkApp (mkConst ``Nat.succ) (mkRawNatLit (n - 1))
          reason (mkAppN e.getAppFn (args.set! r.getMajorIdx c)) (fuel - 1)
        | major => reason major (fuel - 1)
      | none => return .none s!"a partial application of {name}"
    | some (.ctorInfo _) => return .none s!"the constructor {name}"
    | _ =>
      match ← delta? e with
      | some e' => reason e' (fuel - 1)
      | none => return .none s!"the constant {name}"
  | .mvar _ => return .none "a metavariable"
  | f => return .none s!"a term of kind {f.ctorName}"

/-- The fact an attribute `x` gives: its type if it is a proposition, `0 < n` if it is in
`Fin n`. -/
def factOf (x : Expr) : MetaM (Option Expr) := do
  let type := (← inferType x).cleanupAnnotations
  if ← isProp type then return some type
  let type ← whnf type
  if type.isAppOfArity ``Fin 1 then
    return some (← mkAppM ``LT.lt #[mkNatLit 0, type.appArg!])
  return none

/-- `p` reduces to false. -/
def refuted (p : Expr) : MetaM Bool := do
  try
    let inst ← synthInstance (mkApp (mkConst ``Decidable) p)
    return (← whnf (mkApp2 (mkConst ``Decidable.decide) p inst)).isConstOf ``Bool.false
  catch _ => return false

/-- The inductive type of an attribute a split may enumerate: no indices, and not recursive except
`List` and `Nat`. -/
def splittable (type : Expr) : MetaM (Option (InductiveVal × List Level × Array Expr)) := do
  if ← isProp type then return none
  let type ← whnf type
  let .const name us := type.getAppFn | return none
  let some (.inductInfo info) := (← getEnv).find? name | return none
  if info.numIndices != 0 || info.isUnsafe then return none
  if info.isRec && !(name == ``List || name == ``Nat) then return none
  return some (info, us, type.getAppArgs)

/-- `k` at each constructor of the attribute `x`, applied to fresh attributes, three deep. -/
partial def split (x : FVarId) (e : Expr) (k : Expr → M Tree) : M (Option Tree) := do
  let some (info, us, params) ← splittable (← inferType (.fvar x)) | return none
  let d := ((← get).depth.get? x).getD 0
  if d ≥ 3 then return none
  let mut out := #[]
  for c in info.ctors do
    let ctor := mkAppN (mkConst c us) params
    let rest ← instantiateForall (← inferType (mkConst c us)) params
    let t? ← forallTelescope rest fun fields _ => do
      let value := mkAppN ctor fields
      let facts := ((← read).facts.map (·.replaceFVar (.fvar x) value)) ++
        (← fields.filterMapM (fun f => (factOf f : MetaM (Option Expr))))
      if ← facts.anyM (fun f => (refuted f : MetaM Bool)) then return none
      modify fun s =>
        { s with depth := fields.foldl (fun m f => m.insert f.fvarId! (d + 1)) s.depth }
      withReader ({ · with facts }) do
        return some (← k (e.replaceFVar (.fvar x) value))
    let some t := t? | continue
    out := out.push t
    if let some p := (← read).pos then
      if (← read).reach then
        if (Tree.attr out).reach p == .every then break
      else if (Tree.attr out).cover p == .some then break
  return some (.attr out)

/-- An option is some. -/
def posSome : Leaf → Option Bool
  | .ctor n =>
    if n == ``Option.some then some true else if n == ``Option.none then some false else none
  | .data => some true
  | _ => none

/-- A term `m` stuck on an input, or in a search for some value a decision on inputs, at each
constructor of its type with inputs as the fields, and `k` at `rebuild` of each. -/
def dataSplit? (m : Expr) (rebuild : Expr → Expr) (k : Expr → M Tree) : M (Option Tree) := do
  let type ← whnf (← inferType m)
  unless (← read).reach && m.hasAnyFVar (← read).data.contains &&
      (type.isConstOf ``Bool || type.isAppOf ``Decidable) do
    let .data ← reason m | return none
  let .const tname us := type.getAppFn | return none
  let some (.inductInfo info) := (← getEnv).find? tname | return none
  if info.isRec || info.numIndices != 0 then return none
  let params := type.getAppArgs
  let mut out := #[]
  for c in info.ctors do
    let rest ← instantiateForall (← inferType (mkConst c us)) params
    let t ← forallTelescope rest fun fields _ =>
      withReader (fun ctx => { ctx with data := fields.foldl (·.insert ·.fvarId!) ctx.data }) do
        k (rebuild (mkAppN (mkAppN (mkConst c us) params) fields))
    out := out.push t
    if let some p := (← read).pos then
      if (← read).reach then
        if (Tree.data out).reach p == .every then break
      else if (Tree.data out).cover p == .every then break
  return some (.data out)

/-- In a search for some value, a fold over a list that does not reduce to a constructor, or over
`List.finRange n` for a length `n` with attributes in it: its initial value, or its step at an input
accumulator and an input element. -/
def foldl? (e : Expr) (rebuild : Expr → Expr) (k : Expr → M Tree) : M (Option Tree) := do
  unless (← read).reach && e.isAppOfArity ``List.foldl 5 do return none
  let args := e.getAppArgs
  let open_ := args[4]!.isAppOfArity ``List.finRange 1 && args[4]!.appArg!.hasFVar
  unless open_ do
    let l ← whnfM args[4]!
    if l.isAppOf ``List.nil || l.isAppOf ``List.cons then return none
  let first ← k (rebuild args[3]!)
  if let some p := (← read).pos then
    if first.reach p == .every then return some (.data #[first])
  let step ← withLocalDeclD `acc args[0]! fun acc => withLocalDeclD `x args[1]! fun x =>
    withReader (fun ctx => { ctx with data := (ctx.data.insert acc.fvarId!).insert x.fvarId! }) do
      k (rebuild (mkApp2 args[2]! acc x))
  return some (.data #[first, step])

mutual

/-- The reduction of `e` followed to the term it is stuck on, rewritten there, and `k` at `rebuild`
of the result: a major premise or a structure `whnfM` reduces to a constructor is that constructor,
a cast is its value, a comparison `Nat.beq a b`, `Nat.ble a b` or `List.beq a b` (for an equality
that decides `=`) of definitionally equal sides is true, and a loop in `Option` whose body is some
at every index is some of a fresh input. -/
partial def unstick? (e : Expr) (rebuild : Expr → Expr) (k : Expr → M Tree) (fuel : Nat := 256) :
    M (Option Tree) := do
  if fuel == 0 then return none
  let e ← whnfCore e
  let f := e.getAppFn
  let args := e.getAppArgs
  match f with
  | .proj s i b =>
    let b ← whnfM b
    if let some field ← projectCore? b i then return some (← k (rebuild (mkAppN field args)))
    unstick? b (fun b' => rebuild (mkAppN (.proj s i b') args)) k (fuel - 1)
  | .const name _ =>
    if let some t ← foldl? e rebuild k then return some t
    if (name == ``Nat.beq || name == ``Nat.ble) && args.size == 2 then
      if ← isDefEq args[0]! args[1]! then return some (← k (rebuild (mkConst ``Bool.true)))
    -- lists compared by an equality that decides `=`
    if name == ``List.beq && args.size == 4 && args[1]!.isAppOf ``instBEqOfDecidableEq then
      if ← isDefEq args[2]! args[3]! then return some (← k (rebuild (mkConst ``Bool.true)))
    if loops.contains name then
      unless args.size ≥ 2 && (← whnf args[0]!).isConstOf ``Option do return none
      let body := args.back!
      let .forallE _ dom _ _ ← whnf (← inferType body) | return none
      let t ← withLocalDeclD `j dom fun j => do
        let facts := (← read).facts ++ (← (factOf j : MetaM (Option Expr))).toArray
        if ← facts.anyM (fun f => (refuted f : MetaM Bool)) then return Tree.attr #[]
        withReader ({ · with facts, pos := some posSome }) (explore (mkApp body j))
      let type ← whnf (← inferType e)
      let some u := type.getAppFn.constLevels!.head? | return none
      let value := type.appArg!
      let c := if (← read).reach then t.reach posSome else t.cover posSome
      if (← read).reach && c == .never then
        return some (.attr #[← k (rebuild (mkApp (mkConst ``Option.none [u]) value))])
      unless c == .every || c == .vacuous do
        return some (.leaf (.unknown s!"the body of a loop {name} is some {c.text}"))
      return ← withLocalDeclD `loop value fun v =>
        withReader (fun ctx => { ctx with data := ctx.data.insert v.fvarId! }) do
          return some (.attr #[← k (rebuild (mkApp2 (mkConst ``Option.some [u]) value v))])
    match (← getEnv).find? name with
    | some (.recInfo r) =>
      if r.k && r.numMinors == 1 && args.size > r.getMajorIdx then
        if let some minor := args[r.numParams + r.numMotives]? then
          return some (← k (rebuild (mkAppN minor args[r.getMajorIdx + 1:])))
      let some major := args[r.getMajorIdx]? | return none
      let m ← whnfM major
      let reduced := mkAppN f (args.set! r.getMajorIdx m)
      if m.isRawNatLit then return some (← k (rebuild reduced))
      if let .const mc _ := m.getAppFn then
        if let some (.ctorInfo _) := (← getEnv).find? mc then return some (← k (rebuild reduced))
      let rebuild' := fun m => rebuild (mkAppN f (args.set! r.getMajorIdx m))
      if let some t ← unstick? m rebuild' k (fuel - 1) then return some t
      dataSplit? m rebuild' k
    | some (.ctorInfo _) => return none
    | _ =>
      let some e' ← delta? e | return none
      unstick? e' rebuild k (fuel - 1)
  | _ => return none

/-- The reduction tree of `e`. -/
partial def explore (e : Expr) : M Tree := do
  let s ← get
  let leaf (l : Leaf) : M Tree := do
    modify fun s => { s with leaves := s.leaves + 1 }
    return .leaf l
  if s.fuel == 0 then return ← leaf (.unknown "the exploration ran out of fuel")
  if (← IO.getNumHeartbeats) - (← read).start > (← read).budget then
    return ← leaf (.unknown "the exploration ran out of heartbeats")
  if s.leaves ≥ 512 then return ← leaf (.unknown "the exploration reached 512 leaves")
  set { s with fuel := s.fuel - 1 }
  let e ← whnfCore e
  -- `e` at each constructor of its major premise `m`, the argument `idx`. The fresh fields are
  -- inputs when `m` is stuck on an input
  let branch (idx : Nat) (m : Expr) (input : Bool) (fallback : Leaf) : M Tree := do
    let type ← whnf (← inferType m)
    let .const tname us := type.getAppFn | leaf fallback
    let some (.inductInfo info) := (← getEnv).find? tname | leaf fallback
    if info.isRec || info.numIndices != 0 then return ← leaf fallback
    let params := type.getAppArgs
    let f := e.getAppFn
    let args := e.getAppArgs
    let mut out := #[]
    for c in info.ctors do
      let ctor := mkAppN (mkConst c us) params
      let rest ← instantiateForall (← inferType (mkConst c us)) params
      let t ← forallTelescope rest fun fields _ =>
        let inputs (ctx : Ctx) : Ctx :=
          if input then { ctx with data := fields.foldl (·.insert ·.fvarId!) ctx.data } else ctx
        withReader inputs do
          explore (mkAppN f (args.set! idx (mkAppN ctor fields)))
      out := out.push t
      if let some p := (← read).pos then
        if (← read).reach then
          if (Tree.data out).reach p == .every then break
        else
          if input && (Tree.data out).cover p == .every then break
          if !input && (Tree.attr out).cover p == .some then break
    return if input then .data out else .attr out
  match e.getAppFn with
  | .fvar x => leaf (if (← read).data.contains x then .data else .unknown "an attribute")
  | .lit _ => leaf (.ctor `literal)
  | .lam .. => leaf (.ctor `function)
  | .proj _ i b =>
    let b ← whnfM b
    if let some field ← projectCore? b i then return ← explore (mkAppN field e.getAppArgs)
    match ← reason b with
    | .data => leaf .data
    | .attr x =>
      match ← split x e explore with
      | some t => return t
      | none => leaf (.unknown s!"a projection of the attribute {(← x.getDecl).userName}")
    | .none why => leaf (.unknown s!"a projection of {why}")
  | .const name _ =>
    if (← read).stops.contains name then return ← leaf (.stop name)
    if let some t ← foldl? e id explore then return t
    match (← getEnv).find? name with
    | some (.ctorInfo _) => leaf (.ctor name)
    | some (.recInfo r) =>
      let idx := r.getMajorIdx
      let args := e.getAppArgs
      match args[idx]? with
      | none => leaf (.unknown s!"a partial application of {name}")
      | some major =>
        let m ← whnfM major
        if let .const mc _ := m.getAppFn then
          if let some (.ctorInfo _) := (← getEnv).find? mc then
            return ← explore (mkAppN e.getAppFn (args.set! idx m))
        if m.isRawNatLit then
          return ← explore (mkAppN e.getAppFn (args.set! idx m))
        -- in a search for some value, a decision on inputs takes either value
        if (← read).reach && m.hasAnyFVar (← read).data.contains then
          let type ← whnf (← inferType m)
          if type.isConstOf ``Bool || type.isAppOf ``Decidable then
            return ← branch idx m true .data
        if let some t ← unstick? m (fun m' => mkAppN e.getAppFn (args.set! idx m')) explore then
          return t
        -- a cast along an equation whose sides do not reduce: its value
        if r.k && r.numMinors == 1 then
          if let some minor := args[r.numParams + r.numMotives]? then
            return ← explore (mkAppN minor args[idx + 1:])
        match ← reason m with
        | .data => branch idx m true .data
        | .attr x =>
          if let some t ← split x e explore then return t
          branch idx m false (.unknown s!"{name} on the attribute {(← x.getDecl).userName}")
        | .none why => leaf (.unknown s!"{name} on {why}")
    | _ =>
      match ← delta? e with
      | some e' => explore e'
      | none => leaf (.unknown s!"the constant {name}")
  | f => leaf (.unknown s!"a term of kind {f.ctorName}")

end

/-- The facts of the attributes of the operator `op`. -/
def factsOf (op : Expr) : MetaM (Array Expr) :=
  op.getAppArgs.filterMapM fun x => if x.isFVar then factOf x else pure none

/-- The reduction tree of `e`, with the free variables `data` as inputs. -/
def run (data : Array Expr) (e : Expr) (stops : NameSet := {})
    (pos : Option (Leaf → Option Bool) := none) (facts : Array Expr := #[]) (reach := false) :
    MetaM Tree := do
  let ctx : Ctx := { data := data.foldl (fun s x => s.insert x.fvarId!) {}, stops, pos,
                     start := ← IO.getNumHeartbeats, facts, reach }
  -- the loops stay folded, but the arithmetic of `Nat` unfolds, its division by zero included
  withCanUnfoldPred (fun cfg info =>
      if loops.contains info.name then pure false
      else if info.name.getPrefix == `Nat then pure true else canUnfoldDefault cfg info) do
    (explore e |>.run ctx).run' {}

/-- The entries of a list of ports, after reducing it to its constructors. -/
partial def portList (l : Expr) : MetaM (Option (Array Expr)) := do
  let l ← whnf l
  if l.isAppOfArity ``List.nil 1 then return some #[]
  if l.isAppOfArity ``List.cons 3 then
    let some rest ← portList l.appArg! | return none
    return some (#[l.appFn!.appArg!] ++ rest)
  return none

/-- The input and output ports of the operator `op`. -/
def portsOf (op : Expr) : MetaM (Expr × Expr) := do
  let ty ← whnfR (← inferType op)
  let #[ins, outs] := ty.getAppArgs | throwError "not an operator: {ty}"
  return (ins, outs)

/-- `k` at the constructor `c` applied to fresh attributes. With `scalar`, each element attribute is
the scalar and a list attribute the inputs are computed from has two entries. -/
def withOp (c : Name) (scalar : Bool) (k : Expr → MetaM α) : MetaM α := do
  let info ← getConstInfoCtor c
  forallTelescope info.type fun xs _ => do
    let mut op := mkAppN (mkConst c (info.levelParams.map fun _ => Level.zero)) xs
    unless scalar do return ← k op
    for x in xs do
      let t := (← inferType x).cleanupAnnotations
      if t.isConstOf ``TorchLean.Op.Element then
        op := op.replaceFVar x (mkConst ``TorchLean.Op.Element.scalar)
      else if t.isConstOf ``TorchLean.Op.NumericElement then
        op := op.replaceFVar x (mkConst ``TorchLean.Op.NumericElement.scalar)
    let (ins, _) ← portsOf op
    if (← portList ins).isSome then return ← k op
    for x in xs do
      let t ← whnf (← inferType x)
      if t.isAppOfArity ``List 1 then
        let elem := t.appArg!
        let found ← withLocalDeclD `a elem fun a => withLocalDeclD `b elem fun b => do
          let op' := op.replaceFVar x (← mkListLit elem [a, b])
          let (ins', _) ← portsOf op'
          if (← portList ins').isSome then return some (← k op') else return none
        if let some r := found then return r
    k op

/-- Which input ports of `op` hold values, the ports a gradient flows into. -/
def valueInputs (op : Expr) : MetaM (Option (Array Bool)) := do
  let (ins, _) ← portsOf op
  let some ports ← portList ins | return none
  ports.mapM fun p => do
    let p ← whnf p
    if p.isAppOfArity ``TorchLean.Port.mk 2 then
      return (← whnf p.appFn!.appArg!).isConstOf ``TorchLean.Port.Kind.value
    return false

/-- The number of binders of `t` before the first one of type `n`. -/
def countBefore (t : Expr) (n : Name) : Nat := go t 0 where
  go : Expr → Nat → Nat
    | .forallE _ d b _, k => if d.isConstOf n then k else go b (k + 1)
    | _, k => k

/-- The operator `op` packed with its ports, as a graph node holds it. -/
def operatorOf (op : Expr) : MetaM Expr := do
  let ty ← whnfR (← inferType op)
  let #[ins, outs] := ty.getAppArgs | throwError "not an operator: {ty}"
  let ports ← inferType ins
  let opC := ty.getAppFn
  let u := Level.zero
  let inner := mkApp4 (mkConst ``Sigma.mk [u, u]) ports
    (.lam `outs ports (mkApp2 opC ins (.bvar 0)) .default) outs op
  let outer := Expr.lam `ins ports (mkApp2 (mkConst ``Sigma [u, u]) ports
    (.lam `outs ports (mkApp2 opC (.bvar 1) (.bvar 0)) .default)) .default
  return mkApp4 (mkConst ``Sigma.mk [u, u]) ports outer ins inner

/-- The reduction tree of the bound propagation `fn` at a node of `op` on fresh references, with
its other arguments fresh and the references as inputs. -/
def boundLeaves (fn : Name) (op : Expr) (nIns : Nat) (stops : NameSet := {})
    (pos : Option (Leaf → Option Bool) := none) (reach := true) : MetaM Tree := do
  let info ← getConstInfo fn
  let c := mkConst fn (info.levelParams.map fun _ => Level.zero)
  let n := countBefore info.type ``TorchLean.IR.Operator
  forallBoundedTelescope info.type (some n) fun xs _ => do
    let refTy := mkConst ``TorchLean.IR.Ref
    withLocalDeclsD ((List.range nIns).toArray.map fun i =>
        (Name.mkSimple s!"r{i}", fun _ => pure refTy)) fun rs => do
      let sigma ← operatorOf op
      let e := mkAppN c (xs ++ #[sigma, ← mkListLit refTy rs.toList])
      run (xs ++ rs) e stops pos (← factsOf op) reach

/-- The reduction tree of whether the rule of `op` has a forward mode, at a capability record
that holds every capability. -/
def jvpLeaves (op : Expr) (pos : Option (Leaf → Option Bool) := none) : MetaM Tree := do
  let info ← getConstInfo ``TorchLean.Op.rule?
  let spec := mkConst ``TorchLean.Op.rule? (info.levelParams.map fun _ => Level.zero)
  let rec count : Expr → Nat → Nat
    | .forallE _ d b _, k => if d.isAppOf ``TorchLean.Op.Capabilities then k else count b (k + 1)
    | _, k => k
  forallBoundedTelescope info.type (some (count info.type 0)) fun xs _ => do
    let α := xs[0]!
    let mkTy ← instantiateForall (← inferType (mkConst ``TorchLean.Op.Capabilities.mk)) #[α]
    let fieldTys ← forallTelescope mkTy fun fs _ => fs.mapM fun f => do
      return (← inferType f).appArg!
    withLocalDeclsD (fieldTys.map fun t => (`c, fun _ => pure t)) fun cs => do
      let caps := mkAppN (mkApp (mkConst ``TorchLean.Op.Capabilities.mk) α)
        ((cs.zip fieldTys).map fun (c, t) => mkApp2 (mkConst ``Option.some [Level.zero]) t c)
      let (ins, outs) ← portsOf op
      let app := mkAppN spec (xs ++ #[caps, ins, outs, op])
      let ruleTy := (← inferType app).appArg!
      let has ← withLocalDeclD `r ruleTy fun r => do
        mkLambdaFVars #[r] (← mkAppM ``Option.isSome #[← mkAppM ``TorchLean.Op.Rule.jvp? #[r]])
      run (xs ++ cs) (← mkAppM ``Option.any #[has, app]) {} pos (← factsOf op)

/-- The fields of a rule a statement reads. -/
structure RuleFields where
  forward : Bool
  backward : Bool
  prepare : Bool
  jvp : Bool

/-- The fields of `TorchLean.Op.Rule` that `type` reads. -/
def ruleFields (type : Expr) : MetaM RuleFields := do
  let fields := getStructureFields (← getEnv) ``TorchLean.Op.Rule
  let reads (field : Name) : Bool :=
    (type.find? fun e => e.isAppOf (``TorchLean.Op.Rule ++ field) ||
      match e with
      | .proj owner position _ => owner == ``TorchLean.Op.Rule && fields[position]? == some field
      | _ => false).isSome
  return { forward := reads `forward, backward := reads `backward, prepare := reads `prepare,
           jvp := reads `jvp? }

/-- The declarations at which the backward sweep of CROWN leaves an operator without an arm: it
discharges the coefficient against the box, or it ends. -/
def crownStops : NameSet :=
  ({} : NameSet).insert ``TorchLean.CROWN.Op.consumeAt
    |>.insert ``TorchLean.CROWN.Directed.State.fail

/-- A Boolean is true. -/
def posBool (l : Leaf) : Option Bool :=
  if l == .ctor ``Bool.true then some true else if l == .ctor ``Bool.false then some false else none

/-- An option is some. -/
def posOpt (l : Leaf) : Option Bool :=
  if l == .ctor ``Option.some then some true else if l == .ctor ``Option.none then some false
  else none

/-- A box is found: some, or a box read from the inputs. -/
def posBox (l : Leaf) : Option Bool := if l == .data then some true else posOpt l

/-- The backward sweep passes the coefficient on through an arm of its own. -/
def posOwn (l : Leaf) : Option Bool :=
  if l == .ctor ``TorchLean.CROWN.Directed.State.mk then some true
  else if l matches .stop _ then some false else none

/-- The class of the scalar that holds each capability an operator can need. -/
def needClasses : List (Name × String) :=
  [(``TorchLean.Op.Need.rounding, "Rounding"), (``TorchLean.Op.Need.erf, "Erf"),
   (``TorchLean.Op.Need.inverseTrig, "InverseTrig"),
   (``TorchLean.Op.Need.decidableEq, "DecidableEq"),
   (``TorchLean.Op.Need.cbrt, "Cbrt"), (``TorchLean.Op.Need.nextafter, "NextAfter"),
   (``TorchLean.Op.Need.bits, "Bits")]

/-- The rule cell: the capabilities `Op.rule?` reads for the rule, and where there is no rule. -/
def ruleCell (c : Name) (need hasRule : Tree) : MetaM String := do
  let ruled := hasRule.cover posBool
  match ruled with
  | .never =>
    unless c == ``TorchLean.Op.custom do throwError "{c} has no rule at any attribute"
    return "from its node's program (`Op.customRule`)"
  | .every | .some => pure ()
  | other => throwError "whether {c} has a rule does not reduce: {other.text}"
  let mut caps : Array String := #[]
  let mut free := false
  for l in need.leaves do
    let .ctor n := l | throwError "the capability {c} needs does not reduce: {l.text}"
    if n == ``TorchLean.Op.Need.none then
      free := true
    else
      let some cls := needClasses.lookup n
        | throwError "the capability {c} needs does not reduce: {l.text}"
      unless caps.contains cls do caps := caps.push cls
  let text :=
    if caps.isEmpty then "every scalar"
    else if caps.size == 1 && !free then s!"`{caps[0]!}`"
    else ", ".intercalate (caps.toList.map fun cls => s!"`{cls}` at some attributes")
  return if ruled == .some then text ++ "; no rule at some attributes" else text

/-- The CUDA cell: whether the LibTorch table has an entry. -/
def cudaCell (c : Name) (kernel : Tree) : MetaM String :=
  match kernel.cover posOpt with
  | .every => pure "yes"
  | .never => pure "no"
  | .some => pure "at some attributes"
  | other => throwError "the LibTorch entry of {c} does not reduce: {other.text}"

/-- The certificates registered for `c`, with the fields of the rule each one reads. An `enclosure`
certificate reads the forward map, and every other one the pullback, the prepared state or the
forward mode. -/
def certsOf (c : Name) : MetaM (Array (TorchLean.Op.CertEntry × RuleFields)) := do
  (TorchLean.Op.certificates (← getEnv) c).mapM fun entry => do
    let fields ← ruleFields (← getConstInfo entry.certificate).type
    if entry.kind == `enclosure then
      unless fields.forward do
        throwError "the certificate {entry.certificate} of {c} reads no forward map of the rule"
    else unless fields.backward || fields.prepare || fields.jvp do
      throwError "the certificate {entry.certificate} of {c} reads no gradient of the rule"
    return (entry, fields)

/-- The kinds `kinds` holds, in the order of `TorchLean.Op.certKinds`, as code. -/
def kindsText (kinds : Array Name) : String :=
  ", ".intercalate (TorchLean.Op.certKinds.filter kinds.contains |>.map fun k => s!"`{k}`")

/-- `input 2`, `inputs 0 and 1`, `inputs 0, 1 and 2`. -/
def inputList (ks : List Nat) : String :=
  match ks.reverse with
  | [] => "no input"
  | [k] => s!"input {k}"
  | last :: rest => s!"inputs {", ".intercalate (rest.reverse.map toString)} and {last}"

/-- The gradient cell: the kinds of the certificates registered for the pullback at each value
input, and whether the rule has a forward mode, with the kinds registered for it. -/
def gradientCell (c : Name) (values : Array Bool) (jvp : Tree) : MetaM String := do
  let certs ← certsOf c
  let kindsAt (k : Nat) : String :=
    let kinds := certs.filterMap fun (entry, f) =>
      if !f.jvp && (f.backward || f.prepare) && (entry.inputs.isEmpty || entry.inputs.contains k)
      then some entry.kind else none
    if kinds.isEmpty then "no certificate" else kindsText kinds
  let mut groups : Array (String × Array Nat) := #[]
  for k in [:values.size] do
    unless values[k]! do continue
    let t := kindsAt k
    match groups.findIdx? (·.1 == t) with
    | some i => groups := groups.modify i fun (t, ks) => (t, ks.push k)
    | none => groups := groups.push (t, #[k])
  let pullback := match groups.toList with
    | [] => "no value input"
    | [(t, _)] => t
    | gs => "; ".intercalate (gs.map fun (t, ks) => s!"{inputList ks.toList}: {t}")
  let tangentKinds := certs.filterMap fun (entry, f) => if f.jvp then some entry.kind else none
  let tangent ← match jvp.cover posBool with
    | .every => pure "forward mode"
    | .some => pure "forward mode at some attributes"
    | .never =>
      unless tangentKinds.isEmpty do
        throwError "{c} has certificates of a forward mode and no forward mode"
      return pullback
    | other => throwError "whether {c} has a forward mode does not reduce: {other.text}"
  let kinds := if tangentKinds.isEmpty then "" else s!" ({kindsText tangentKinds})"
  return s!"{pullback}; {tangent}{kinds}"

/-- The bounds cell: the bound propagations with an arm for the operator, then `enclosure` when an
`enclosure` certificate is registered for it. -/
def boundsCell (c : Name) (cert ibp crown : Tree) : MetaM String := do
  let mut out := #[]
  let arms := [("checker", cert, posBox), ("IBP", ibp, posBox), ("CROWN", crown, posOwn)]
  for (label, t, pos) in arms do
    match t.reach pos with
    | .every => out := out.push label
    | .never | .vacuous => pure ()
    | other => throwError "the {label} arm of {c} does not reduce: {other.text}"
  let cell := if out.isEmpty then "none" else ", ".intercalate out.toList
  let enclosed := (← certsOf c).any (·.1.kind == `enclosure)
  return if enclosed then s!"{cell}; `enclosure`" else cell

/-- The cells of the row of the operator `c`: its name, rule, CUDA, gradient and bounds. -/
def row (c : Name) : MetaM (Array String) := withOp c true fun op => do
  let facts ← factsOf op
  let need ← run #[] (← mkAppM ``TorchLean.Op.need #[op]) {} none facts
  let hasRule ← run #[] (← mkAppM ``TorchLean.Op.hasRule #[op]) {} posBool facts
  let kernel ← run #[] (← mkAppM ``TorchLean.LibTorch.Kernel.ofOp #[op]) {} posOpt facts
  let some values ← valueInputs op | throwError "the input ports of {c} do not reduce"
  let jvp ← jvpLeaves op posBool
  let n := values.size
  let cert ← boundLeaves ``TorchLean.CROWN.Op.cert? op n {} posBox
  let ibp ← boundLeaves ``TorchLean.CROWN.Op.ibp? op n {} posBox
  let crown ← boundLeaves ``TorchLean.CROWN.Op.backwardRule op n crownStops posOwn
  return #[s!"`{c.getString!}`", ← ruleCell c need hasRule, ← cudaCell c kernel,
    ← gradientCell c values jvp, ← boundsCell c cert ibp crown]

/-- The constructors of `TorchLean.Op`, in declaration order. -/
def opCtors : CoreM (List Name) := do
  let some (.inductInfo table) := (← getEnv).find? ``TorchLean.Op
    | throwError "`TorchLean.Op` is not an inductive type"
  return table.ctors

/-- The operators the `operators` blocks of the page have listed. -/
initialize listedExt : EnvExtension (Array Name) ← registerEnvExtension (pure #[])

end TorchLeanBlueprint.OpTable

open Lean
open Verso Doc Elab ArgParse
open Verso.Genre Manual

namespace TorchLeanBlueprint.OpTable

/-- `text` as inlines, a span between backquotes as code. -/
def cellInlines (text : String) : Array (Verso.Doc.Inline Manual) := Id.run do
  let mut out : Array (Verso.Doc.Inline Manual) := #[]
  let mut code := false
  for s in text.splitOn "`" do
    unless s.isEmpty do out := out.push (if code then .code s else .text s)
    code := !code
  return out

/-- The header of an operator table. -/
def header : Array String := #["Operator", "Rule", "CUDA", "Gradient", "Bounds"]

/-- The operator table with the rows `rows`, as the `table` directive renders a table. -/
def operatorTable (rows : Array (Array String)) : Verso.Doc.Block Manual :=
  let cell (text : String) : Verso.Doc.ListItem (Verso.Doc.Block Manual) :=
    ⟨#[.para (cellInlines text)]⟩
  .other (Block.table header.size true none none) #[.ul ((#[header] ++ rows).flatten.map cell)]

end TorchLeanBlueprint.OpTable

open TorchLeanBlueprint.OpTable

/-- A table with a row for each operator the block names, by constructor of `TorchLean.Op`,
separated by spaces or lines. -/
@[code_block_expander operators]
def operators : CodeBlockExpander
  | args, str => do
    ArgParse.run .done args
    let ctors ← opCtors
    let mut rows := #[]
    for word in str.getString.splitToList Char.isWhitespace do
      if word.isEmpty then continue
      let c := ``TorchLean.Op ++ word.toName
      unless ctors.contains c do throwErrorAt str "`{word}` is not an operator of `TorchLean.Op`"
      if (listedExt.getState (← getEnv)).contains c then
        throwErrorAt str "`{word}` is listed twice on this page"
      modifyEnv (listedExt.modifyState · (·.push c))
      rows := rows.push (← row c)
    pure #[← ``(operatorTable $(quote rows))]

/-- `{opCount}[]`: the number of operators of `TorchLean.Op`. -/
@[role_expander opCount]
def opCount : RoleExpander
  | args, _ => do
    ArgParse.run .done args
    pure #[← ``(Verso.Doc.Inline.text $(quote (toString (← opCtors).length)))]

/-- `{opListed}[]`: the number of operators of `TorchLean.Op`, after a check that the `operators`
blocks above it on the page name every one. -/
@[role_expander opListed]
def opListed : RoleExpander
  | args, _ => do
    ArgParse.run .done args
    let listed := listedExt.getState (← getEnv)
    let ctors ← opCtors
    let missing := (ctors.filter (!listed.contains ·)).map (·.getString!)
    unless missing.isEmpty do
      throwError "the operator tables of this page omit {", ".intercalate missing}"
    pure #[← ``(Verso.Doc.Inline.text $(quote (toString ctors.length)))]

/-- `{opCertKinds}[]`: the number of registered certificates of each kind, in the order of
`TorchLean.Op.certKinds`. -/
@[role_expander opCertKinds]
def opCertKinds : RoleExpander
  | args, _ => do
    ArgParse.run .done args
    let entries := TorchLean.Op.certExtension.getState (← getEnv)
    let counts := TorchLean.Op.certKinds.map fun k =>
      (toString (entries.filter (·.kind == k)).size, k.toString)
    let mut out := #[]
    let mut i := 0
    for (n, k) in counts do
      if i > 0 then
        let sep := if i + 1 == counts.length then " and " else ", "
        out := out.push (← ``(Verso.Doc.Inline.text $(quote sep)))
      out := out.push (← ``(Verso.Doc.Inline.text $(quote (n ++ " "))))
      out := out.push (← ``(Verso.Doc.Inline.code $(quote k)))
      i := i + 1
    pure out
