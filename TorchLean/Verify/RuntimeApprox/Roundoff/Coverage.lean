/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Broadcast -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Convolution -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.InverseTrigonometric -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Literal -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.OnlineSoftmax -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Power -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Products -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Softmax -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Tensor -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxReplay
public import TorchLean.Verify.RuntimeApprox.Roundoff.Routes
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Composite -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Movement -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Ordering -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.ReduceOrder -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Reduction -- shake: keep
public meta import Lean.Elab.DeclarationRange
public meta import TorchLean.Tensor.Tactic.Init
public import TorchLean.Tensor.Tactic.Init

/-!
# Roundoff coverage of registered operators

`#roundoff_coverage family := operator, ...` runs `roundoff` on every route of the declarations
it is given, as `TorchLean.Verify.RuntimeApprox.Roundoff.Routes` reads them from their code, and
records the outcome of each. A registered operator may also be listed with some attributes fixed,
as `TorchLean.Op.crossEntropy (mean := true)`: it then has its routes once more with those
attributes the given terms, named after them and the route (`mean true forward`). The entry
`every M` lists the public definitions of the imported modules `M` and under it whose value is
the scalar or a tensor, and that no earlier coverage has recorded (`moduleFunctions`). A route that
applies an already registered operator specification to local variables shares that
specification's outcome.

Each route is analyzed universally over the declaration's shapes and parameters, at one
coordinate of the tensor or at the scalar it computes, for a configured scalar over an arbitrary
format with the hypothesis that the format is IEEE, once for an absolute bound and once for a
mixed bound. The theorem keeps an instance or a hypothesis of the route, such as the scalar's
storage or the hypothesis that the format is IEEE, only where its statement or proof uses it
(`dropUnused`): one without the IEEE hypothesis holds at every format, the float8 formats whose
encodings are not IEEE included. A successful analysis is declared as a theorem in the current
namespace, under the declaration's name less the root namespace the two share (`theoremName`),
unless the theorem of an earlier route, in this module or one it imports, has the same statement:
the route then records that theorem (`sameStatement?`), so identical routes, such as the two
gradients of `add` or an operator and the tensor function it applies, share one theorem. Its
premises are classified as finite inputs, finite intermediate values, range conditions on inputs,
conditions on computed values and others. A theorem whose budget or reference decodes a computed
value, or whose proof uses an enclosure in place of a rounding theorem, is recorded with that
caveat: it holds, but it is not an a priori bound.

A configured route whose analysis exceeds the heartbeat budget of a declaration is analyzed once
more when the operations its definitions call have propagation theorems. A route whose first
analysis succeeds is not analyzed again and keeps its theorem. A propagation theorem is the analysis
of a route of such an operation, found within two levels of tensor-returning definitions, under the
hypothesis that each input tensor or scalar lies within real budgets of real references, so that its
reference reads those references instead of the input. The statement of
`TorchLean.Normalization.normalize` reads each input tensor at one coordinate, the output coordinate
or its broadcast position, and its propagation theorem bounds each input at that coordinate alone,
with a real reference and real budgets (`localizedCallees`). The second analysis applies these
theorems as rules and supplies their hypotheses from the analysis of the arguments. A proof that
takes one of them from its own conditions, as `roundoff` binds the premises it does not discharge,
is rejected. The second analysis and the reading of its proof each have a heartbeat budget of their
own.
The propagation theorems and auxiliary lemmas a proof uses, directly or through those it keeps, are
declared with it. A propagation theorem whose statement an earlier one has, generated in the module
or declared in it or in one it imports, is not declared again: the composite applies the earlier
one (`recordPropagation`), whose own arms and primitive theorems it reads, as the declaring module
recorded them (`propagationExtension`). Its record names the propagation theorems, inherits their
primitive theorems and says whether its reference decodes a computed value.
`#roundoff_propagation mode := operation, ...` declares the propagation theorems of every
configured route of the listed operations in one mode, absolute or mixed, outside any composite,
each unless an earlier one has its statement.

A declaration may take functions over the scalar, such as the fields of the operator specifications
that `TorchLean.VJPOp.compose` composes, or the fields of a capability instance the scalar does not
provide, such as the `InverseTrig` of `TorchLean.Tensor.atan2`. A route that applies one is analyzed
under a contract for it: a real model, a domain on the arguments' real values and budgets, each a
variable of the theorem, and the hypothesis that at finite arguments in the domain its result is
finite with mixed error `(u, η)` against the model, or in absolute analysis differs from the model
by at most `ε`. Mixed analysis reads the hypothesis as it reads a generated theorem. The theorem
keeps the contracts its proof uses. They are recorded as the premise class `function contract`, and
the conditions on their domains as `contract domain`. Such a theorem holds for every function that
meets its contract and says nothing about one that does not.

The analysis also uses the rules of thirteen modules in `Rules/`. `Tensor` has the tensor reads and
shape casts, `Softmax` the softmax reads, `OnlineSoftmax` the bounds of the online softmax schedule,
`Products` the replays of scaled products and `Movement` the data-movement reads. `Ordering` has the
cumulative products, extrema and sorts, `Reduction` the reductions and scatters, `Composite` the
composites along any axes and `Broadcast` the bound on broadcast reductions. `Convolution` has the
convolution and pooling reads, `Literal` the literals, `Power` the powers and `Elementwise` the
elementwise reads and exact integral roundings. It also uses the softmax normalization and pullback
of `TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxReplay`.

Native `Float32`, read into binary32 through `ExecFloat.Binary.ofFloat32`, and the rounded reals
`TorchLean.FP32` are analyzed once each, and each route's entry states what stops the analysis. The
entries are stored in an environment extension, which the coverage check of
`scripts/checks/Library.lean` reads.
-/

public section

namespace TorchLean.Roundoff

/-- A bound at `t` on `f` against `r` gives the bound at every point on `f` against `f` updated to
`r` at `t`, within the budget `e` at `t` and `0` elsewhere. -/
theorem abs_sub_update_le {C : Sort*} [DecidableEq C] (f : C → ℝ) (t : C) {r e : ℝ}
    (h : |f t - r| ≤ e) (c : C) :
    |f c - Function.update f t r c| ≤ Function.update (fun _ => (0 : ℝ)) t e c := by
  rcases eq_or_ne c t with rfl | hc
  · simpa using h
  · simp [Function.update_of_ne hc]

/-- A mixed error at `t` of `f` against `r` gives the mixed error at every point of `f` against `f`
updated to `r` at `t`, within the budgets `ρ` and `α` at `t` and `0` elsewhere. -/
theorem mixedError_update {C : Sort*} [DecidableEq C] (f : C → ℝ) (t : C) {r ρ α : ℝ}
    (h : FloatLib.Numerics.MixedError (f t) r ρ α) (c : C) :
    FloatLib.Numerics.MixedError (f c) (Function.update f t r c)
      (Function.update (fun _ => (0 : ℝ)) t ρ c) (Function.update (fun _ => (0 : ℝ)) t α c) := by
  rcases eq_or_ne c t with rfl | hc
  · simpa using h
  · simp [Function.update_of_ne hc]

end TorchLean.Roundoff

end

public meta section

namespace TorchLean.Roundoff

open Lean Meta Elab Term Command
open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Roundoff
  (conjuncts mentions normalizeRule primitiveRoundings primitiveEnclosures roundoffRules Session)
open Einops (instantiateOuterLets)

/-- The outcome of analyzing one route. -/
inductive CoverageStatus where
  /-- A theorem without premises whose reference and budget decode inputs only. -/
  | proved
  /-- A theorem with premises or caveats, each recorded. -/
  | conditional
  /-- The analysis found an operation without a numerical rule. -/
  | unsupported
  /-- The analysis stopped for another reason. -/
  | failed
  deriving Inhabited, BEq

/-- The record of one analyzed route. -/
structure CoverageEntry where
  /-- The family the operator is registered in. -/
  family : Name
  /-- The registered declaration. -/
  operator : Name
  /-- `forward`, `backward`, `value`, a tape gradient such as `x gradient`, or the constructor a
  route takes, such as `sum`. It is empty when the declaration is not analyzed. -/
  route : String
  /-- The scalar the route was analyzed for. -/
  backend : String
  /-- `absolute` or `mixed`. -/
  mode : String
  /-- The outcome of the analysis. -/
  status : CoverageStatus
  /-- The generated theorem, or `.anonymous`. -/
  evidence : Name := .anonymous
  /-- The real reference of the generated bound, pretty-printed. -/
  reference : String := ""
  /-- The number of distinct subterms of the reference. -/
  referenceSize : Nat := 0
  /-- The number of distinct subterms of the generated budgets, or `0` when each is the numeral
  `0`. -/
  budget : Nat := 0
  /-- The number of premises of each class. -/
  premises : Array (String × Nat) := #[]
  /-- Why the theorem is weaker than an a priori bound. -/
  caveats : Array String := #[]
  /-- The primitive theorems the proof uses: correctly rounded functions and enclosures. -/
  contracts : Array Name := #[]
  /-- Whether the proof uses the hypothesis that the format is IEEE, which the theorem then
  keeps. -/
  usesIEEE : Bool := false
  /-- The functions the declaration takes whose contracts the theorem assumes. -/
  functions : Array String := #[]
  /-- The analysis' message for an unsupported or failed route. -/
  reason : String := ""
  /-- The operator specification and route whose outcome this route shares. -/
  link? : Option (Name × String) := none
  /-- The generated theorems of other routes that the proof applies as rules. -/
  arms : Array Name := #[]
  deriving Inhabited

/-- The routes analyzed by `#roundoff_coverage`, in the order they were analyzed. -/
initialize coverageExtension :
    SimplePersistentEnvExtension CoverageEntry (Array CoverageEntry) ←
  registerSimplePersistentEnvExtension {
    addEntryFn := Array.push
    addImportedFn := fun modules => modules.foldl (· ++ ·) #[] }

/-! ## Function contracts -/

/-- A value a function variable takes or returns. -/
inductive ValueKind where
  /-- The scalar. -/
  | scalar
  /-- A tensor of `shape` over the scalar. -/
  | tensor (shape : Expr)
  /-- A pair of such values. -/
  | pair (first second : ValueKind)

/-- The kind of the values of `type` over `scalar`, if it has one. -/
partial def valueKind? (scalar type : Expr) : MetaM (Option ValueKind) := do
  if ← withReducible (isDefEq type scalar) then return some .scalar
  let type ← whnfR type
  if type.isAppOfArity ``Tensor.Rep 3 then
    if ← withReducible (isDefEq type.getAppArgs[0]! scalar) then
      return some (.tensor type.getAppArgs[1]!)
  if type.isAppOfArity ``Prod 2 then
    let some first ← valueKind? scalar type.appFn!.appArg! | return none
    let some second ← valueKind? scalar type.appArg! | return none
    return some (.pair first second)
  return none

/-- A function that a route may apply and that the declaration takes as a parameter: a variable or
a field of a structure variable, from scalars and tensors over the scalar to one such value,
selected by pair projections from its result. -/
structure FunctionComponent where
  /-- The variable, or the field of a structure variable. -/
  function : Expr
  /-- The type and kind of each of its arguments. -/
  parameters : Array (Expr × ValueKind)
  /-- The pair projections selecting the value from its result, the first applied first. -/
  projections : Array Nat
  /-- The kind of the selected value: a scalar or a tensor. -/
  result : ValueKind
  /-- The function as the record names it, such as `f`, `g.forward` or `backward.1`. -/
  label : String
  /-- The prefix of the names of its contract's variables. -/
  stem : String

/-- The scalars and tensors in a result of kind `kind`, with the pair projections that select them
and the suffixes of their labels and stems. -/
private def selections : ValueKind → Array (Array Nat × String × String × ValueKind)
  | .pair first second =>
    (selections first).map (fun (path, label, stem, kind) =>
        (#[0] ++ path, ".1" ++ label, "Fst" ++ stem, kind)) ++
      (selections second).map fun (path, label, stem, kind) =>
        (#[1] ++ path, ".2" ++ label, "Snd" ++ stem, kind)
  | kind => #[(#[], "", "", kind)]

/-- The components of `function` when it maps scalars and tensors over `scalar` to such values or
pairs of them. -/
def functionComponents (scalar function : Expr) (label stem : String) :
    MetaM (Array FunctionComponent) := do
  forallTelescope (← inferType function) fun arguments result => do
    if arguments.isEmpty then return #[]
    let dependent (type : Expr) := type.hasAnyFVar fun id => arguments.any (·.fvarId! == id)
    let mut parameters := #[]
    for argument in arguments do
      let type ← inferType argument
      if dependent type then return #[]
      let some kind ← valueKind? scalar type | return #[]
      if kind matches .pair .. then return #[]
      parameters := parameters.push (type, kind)
    if dependent result then return #[]
    let some kind ← valueKind? scalar result | return #[]
    return (selections kind).map fun (projections, labelSuffix, stemSuffix, result) =>
      { function, parameters, projections, result, label := label ++ labelSuffix
        stem := stem ++ stemSuffix }

/-- The function components of a variable of the declaration: the variable itself, or the fields
of a structure variable that is not a tensor. A class is taken only when `capability` is set: an
instance of a class of scalar functions, such as `InverseTrig`, that the scalar does not provide is
a variable of the route, and its fields are functions it applies. -/
def variableComponents (scalar parameter : Expr) (capability : Bool := false) :
    MetaM (Array FunctionComponent) := do
  let name := (← parameter.fvarId!.getUserName).eraseMacroScopes.toString
  let direct ← functionComponents scalar parameter name name
  unless direct.isEmpty do return direct
  let type ← whnfR (← inferType parameter)
  if (← valueKind? scalar type).isSome then return #[]
  let some owner := type.getAppFn.constName? | return #[]
  let environment ← getEnv
  unless isStructure environment owner && (capability || !isClass environment owner) do
    return #[]
  (getStructureFields environment owner).mapIdx (·, ·) |>.flatMapM fun (index, field) =>
    functionComponents scalar (.proj owner index parameter) s!"{name}.{field}"
      (name ++ field.toString.capitalize)

/-- Run `k` on a new variable `name` for the coordinates of a tensor of `shape`. -/
private def withCoordinate {α : Type} (shape : Expr) (name : Name) (k : Expr → TermElabM α) :
    TermElabM α := do
  withLocalDeclD name (← mkAppM ``Shape.Coord #[shape]) k

/-- The finiteness premise and the real value of an argument of kind `kind`: for a tensor, the
finiteness of each coordinate and the function decoding its coordinates. -/
private def decodeArgument (kind : ValueKind) (argument : Expr) : TermElabM (Expr × Expr) := do
  let .tensor shape := kind |
    return (← Roundoff.Tuple.finiteness argument, ← mkAppM ``ExecFloat.Binary.toReal #[argument])
  withCoordinate shape `j fun coordinate => do
    let entry ← mkAppM ``Tensor.Rep.get #[argument, coordinate]
    return (← mkForallFVars #[coordinate] (← Roundoff.Tuple.finiteness entry),
      ← mkLambdaFVars #[coordinate] (← mkAppM ``ExecFloat.Binary.toReal #[entry]))

/-- The contract of `component` for its real `model`, its `domain` and its budgets: at finite
arguments whose real values are in the domain, the selected value, at each coordinate of a tensor,
is finite with mixed error `(u, η)` against the model, or in absolute analysis differs from it by
at most `ε`. -/
private def contractStatement (mixed : Bool) (component : FunctionComponent)
    (model domain : Expr) (budgets : Array Expr) : TermElabM Expr := do
  let names := #[`x, `y, `z, `w]
  let declarations := component.parameters.mapIdx fun index (type, _) =>
    (names[index]?.getD (.mkSimple s!"x{index + 1}"), type)
  withLocalDeclsDND declarations fun arguments => do
    let mut premises := #[]
    let mut decoded := #[]
    for argument in arguments, (_, kind) in component.parameters do
      let (premise, value) ← decodeArgument kind argument
      premises := premises.push premise
      decoded := decoded.push value
    let mut result := mkAppN component.function arguments
    for projection in component.projections do
      result ← mkAppM (if projection == 0 then ``Prod.fst else ``Prod.snd) #[result]
    let bound (value reference : Expr) : MetaM Expr := do
      let observed ← mkAppM ``ExecFloat.Binary.toReal #[value]
      if mixed then
        mkAppM ``And #[← Roundoff.Tuple.finiteness value,
          ← mkAppM ``FloatLib.Numerics.MixedError #[observed, reference, budgets[0]!, budgets[1]!]]
      else
        mkAppM ``LE.le #[← mkAppM ``abs #[← mkAppM ``HSub.hSub #[observed, reference]],
          budgets[0]!]
    let conclusion ← match component.result with
      | .tensor shape => withCoordinate shape `i fun coordinate => do
        let value ← mkAppM ``Tensor.Rep.get #[result, coordinate]
        let point := decoded.push coordinate
        mkForallFVars #[coordinate]
          (← mkArrow (mkAppN domain point) (← bound value (mkAppN model point)))
      | _ => do mkArrow (mkAppN domain decoded) (← bound result (mkAppN model decoded))
    mkForallFVars arguments (← premises.foldrM (fun premise body => mkArrow premise body)
      conclusion)

/-- Run `k` on the variables of the contract of `component`, its hypothesis last. They are a real
model and a domain over the real values of the arguments (and the coordinate of a tensor result),
the mixed budgets `(u, η)` or in absolute analysis the bound `ε`, and the contract. -/
def withContract {α : Type} (mixed : Bool) (component : FunctionComponent)
    (k : Array Expr → TermElabM α) : TermElabM α := do
  let real := Lean.mkConst ``Real
  let decoded ← component.parameters.mapM fun
    | (_, .tensor shape) => do mkArrow (← mkAppM ``Shape.Coord #[shape]) real
    | _ => pure real
  let over (codomain : Expr) : MetaM Expr := do
    let codomain ← match component.result with
      | .tensor shape => do mkArrow (← mkAppM ``Shape.Coord #[shape]) codomain
      | _ => pure codomain
    decoded.foldrM (fun domain codomain => mkArrow domain codomain) codomain
  let name (suffix : String) : Name := .mkSimple (component.stem ++ suffix)
  let withBudgets (body : Array Expr → TermElabM α) : TermElabM α :=
    if mixed then
      withLocalDeclD (name "Relative") real fun relative =>
        withLocalDeclD (name "Absolute") real fun absolute => body #[relative, absolute]
    else withLocalDeclD (name "Error") real fun error => body #[error]
  withLocalDeclD (name "Model") (← over real) fun model => do
  withLocalDeclD (name "Domain") (← over (mkSort .zero)) fun domain => do
  withBudgets fun budgets => do
  withLocalDeclD (name "Contract") (← contractStatement mixed component model domain budgets)
    fun contract => k (#[model, domain] ++ budgets.push contract)

/-- Bind the contracts of `components` in turn and run `k` on their variables. -/
private def bindContracts {α : Type} (mixed : Bool) :
    List FunctionComponent → Array (Array Expr) → (Array (Array Expr) → TermElabM α) →
      TermElabM α
  | [], bound, k => k bound
  | component :: rest, bound, k =>
    withContract mixed component fun binders => bindContracts mixed rest (bound.push binders) k

/-- Run `k` on `computation` with the contracts of the function components of its variables, each
bound after the variable it constrains, with the positions of the new binders and, for each
contract hypothesis, its position and the function's label. -/
def withContracts {α : Type} (computation : Expr) (mixed : Bool)
    (k : Expr → Array Nat → Array (Nat × String) → TermElabM α) : TermElabM α :=
  lambdaTelescope computation fun arguments body => do
    let scalar ← inferType body.appArg!
    let mut owned := #[]
    for h : index in [:arguments.size] do
      let argument := arguments[index]
      if ← isProp (← inferType argument) then continue
      let capability := (← argument.fvarId!.getDecl).binderInfo.isInstImplicit
      for component in ← variableComponents scalar argument capability do
        owned := owned.push (index, component)
    bindContracts mixed (owned.toList.map (·.2)) #[] fun bound => do
      let mut binders := #[]
      let mut added := #[]
      let mut hypotheses := #[]
      for h : index in [:arguments.size] do
        binders := binders.push arguments[index]
        for ((owner, component), contract) in owned.zip bound do
          if owner != index then continue
          added := added ++ (Array.range contract.size).map (binders.size + ·)
          hypotheses := hypotheses.push (binders.size + contract.size - 1, component.label)
          binders := binders ++ contract
      k (← mkLambdaFVars binders body) added hypotheses

/-- The positions of the instances and hypotheses among the local variables `arguments`, such as
the scalar's storage and the hypothesis that the format is IEEE: a generated theorem keeps them
only where its statement or proof uses them (`dropUnused`). -/
def optionalArguments (arguments : Array Expr) : MetaM (Array Nat) :=
  (Array.range arguments.size).filterM fun index => do
    let argument := arguments[index]!
    return (← argument.fvarId!.getDecl).binderInfo.isInstImplicit ||
      (← isProp (← inferType argument))

/-- `proof` and its type `type`, whose first `arity` binders are the computation's arguments,
without the binders at `optional` that neither uses, with the positions of the binders they keep
and the labels of the contract hypotheses at `hypotheses` among these. -/
def dropUnused (proof type : Expr) (arity : Nat) (optional : Array Nat)
    (hypotheses : Array (Nat × String) := #[]) : MetaM (Expr × Expr × Array Nat × Array String) :=
  lambdaBoundedTelescope proof arity fun arguments body => do
    if arguments.size != arity then
      return (proof, type, Array.range arity, hypotheses.map (·.2))
    let statement ← instantiateForall type arguments
    let mut used := collectFVars (collectFVars {} body) statement
    let mut kept := #[]
    let mut labels := #[]
    for index in (List.range arity).reverse do
      let argument := arguments[index]!
      if optional.contains index && !used.fvarSet.contains argument.fvarId! then continue
      kept := kept.push index
      if let some (_, label) := hypotheses.find? (·.1 == index) then labels := labels.push label
      used := collectFVars used (← instantiateMVars (← inferType argument))
    kept := kept.reverse
    let binders := kept.map (arguments[·]!)
    return (← mkLambdaFVars binders body, ← mkForallFVars binders statement, kept, labels.reverse)

/-! ## Generated statements -/

/-- A local variable, a field or coordinate of one, a literal, the model of one of these, the
binary32 reading `ExecFloat.Binary.ofFloat32` of one of these, or a `match` each of whose
alternatives is one of these, such as a read at an optional source coordinate that is zero in the
padding. The variables an alternative binds are not inputs (`bound`). -/
private partial def isInputRead (value : Expr) (bound : Array Expr := #[]) : MetaM Bool := do
  match value.consumeMData with
  | .fvar id => return !bound.contains (.fvar id) && (← id.getDecl).value?.isNone
  | .proj _ _ source => isInputRead source bound
  | value =>
    if value.isAppOfArity ``Tensor.Rep.get 5 then
      return ← isInputRead value.getAppArgs[3]! bound
    if let some app ← matchMatcherApp? value then
      return ← app.alts.allM fun alt =>
        lambdaTelescope alt fun binders body => isInputRead body (bound ++ binders)
    if value.isAppOf ``ExecFloat.Binary.toModel then return ← isInputRead value.appArg! bound
    if value.isAppOfArity ``ExecFloat.Binary.ofFloat32 4 then
      return ← isInputRead value.appFn!.appArg! bound
    if value.isAppOf ``OfNat.ofNat || value.isAppOf ``OfScientific.ofScientific then return true
    let some name := value.getAppFn.constName? | return false
    let some field ← getProjectionFnInfo? name | return false
    if field.fromClass || value.getAppNumArgs != field.numParams + 1 then return false
    isInputRead value.getAppArgs[field.numParams]! bound

/-- The full applications of `heads` in `term` and its projections out of the structures
`structures`, visiting each distinct subterm once. -/
private partial def applicationsOf (heads : Array Name) (term : Expr)
    (structures : Array Name := #[]) : Array Expr :=
  (visit term ({}, #[])).2
where
  visit (term : Expr) : Std.HashSet Expr × Array Expr → Std.HashSet Expr × Array Expr
    | (seen, found) =>
      if seen.contains term then (seen, found) else
      let state := (seen.insert term, found)
      match term with
      | .app .. =>
        let state := if heads.any term.isAppOf then (state.1, state.2.push term) else state
        term.getAppArgs.foldl (fun state argument => visit argument state)
          (visit term.getAppFn state)
      | .lam _ type body _ | .forallE _ type body _ => visit body (visit type state)
      | .letE _ type value body _ => visit body (visit value (visit type state))
      | .proj name _ inner =>
        visit inner (if structures.contains name then (state.1, state.2.push term) else state)
      | .mdata _ inner => visit inner state
      | _ => state

/-- Whether `term` reads a computed scalar: a decoding or classification of a scalar that is not
an input read, a comparison of floating-point scalars with an operand that is neither an input read
nor a negated literal, or the error of a `TrackedSum`, which decodes its partial sums. A reference
that branches on such a comparison depends on the rounded computation. -/
def decodesComputedValue (term : Expr) : MetaM Bool := do
  unless (applicationsOf #[``TrackedSum.error] term).isEmpty do return true
  let observations := applicationsOf #[``ExecFloat.Binary.toReal, ``ExecFloat.Binary.toModel,
    ``ExecFloat.Binary.isFinite, ``ExecFloat.Binary.isNaN, ``ExecFloat.Binary.isInfinite,
    ``ExecFloat.Binary.isZero, ``FloatLib.Floats.Formats.Flocq.NF.val] term
    (structures := #[``FloatLib.Floats.Formats.Flocq.NF])
  if ← observations.anyM fun
    | .proj _ _ value => return !(← isInputRead value)
    | observation => return !(← isInputRead observation.appArg!) then
    return true
  let comparisons := applicationsOf #[``LT.lt, ``LE.le, ``GT.gt, ``GE.ge, ``BEq.beq, ``bne,
    ``Ord.compare, ``Eq, ``Ne] term
  for comparison in comparisons do
    let arguments := comparison.getAppArgs
    let first := if comparison.isAppOfArity ``Eq 3 || comparison.isAppOfArity ``Ne 3 then 1 else 2
    unless arguments.size == first + 2 do continue
    let scalar ← withCanUnfoldPred (fun config info =>
      if info.name == ``ExecFloat then pure false else canUnfoldDefault config info)
      (whnfD arguments[0]!)
    unless scalar.isAppOf ``ExecFloat || scalar.isConstOf ``Float32 ||
        scalar.isAppOf ``FloatLib.Floats.Formats.Flocq.NF do continue
    for value in arguments.extract first do
      let negativeLiteral := value.isAppOf ``Neg.neg &&
        (value.appArg!.isAppOf ``OfNat.ofNat || value.appArg!.isAppOf ``OfScientific.ofScientific)
      unless negativeLiteral || (← isInputRead value) do return true
  return false

/-- The number of distinct subterms of `terms`. -/
private partial def distinctSubterms (terms : Array Expr) : Nat :=
  (terms.foldl (fun seen term => visit term seen) {}).size
where
  visit (term : Expr) (seen : Std.HashSet Expr) : Std.HashSet Expr :=
    if seen.contains term then seen else
    let seen := seen.insert term
    match term with
    | .app function argument => visit argument (visit function seen)
    | .lam _ type body _ | .forallE _ type body _ => visit body (visit type seen)
    | .letE _ type value body _ => visit body (visit value (visit type seen))
    | .mdata _ inner | .proj _ _ inner => visit inner seen
    | _ => seen

/-- The classes of generated premises, the most informative first. The contracts of functions
the declaration takes are counted apart, as the class `function contract`. -/
def premiseClasses : Array String :=
  #["finite intermediate", "computed condition", "other", "contract domain", "range",
    "finite input", "IEEE descriptor"]

/-- The class of a generated premise: the first class in `premiseClasses` of a conjunct of its
conclusion, under its binders and hypotheses. -/
def premiseClass (premise : Expr) : MetaM String :=
  forallTelescope premise fun _ body => do
    let classes ← (conjuncts body).mapM fun fact => do
      if let some (_, left, _) := fact.eq? then
        if left.isAppOf ``FloatFormat.isIEEE then return "IEEE descriptor"
        if left.isAppOf ``ExecFloat.Binary.isFinite || left.isAppOf ``Model.isFinite then
          return if ← isInputRead left.appArg! then "finite input" else "finite intermediate"
      if ← decodesComputedValue fact then return "computed condition"
      -- The domain of a function's contract is a predicate variable of the theorem.
      if fact.getAppFn.isFVar then return "contract domain"
      if fact.isAppOf ``LT.lt || fact.isAppOf ``LE.le then return "range"
      return "other"
    return (premiseClasses.find? classes.contains).getD "other"

/-- The reference and budgets of a generated conclusion: an absolute bound, or a mixed bound
with or without the finiteness of the computed value. -/
def boundParts (conclusion : Expr) : Option (Expr × Array Expr) :=
  let bound := if conclusion.isAppOfArity ``And 2 then conclusion.appArg! else conclusion
  if bound.isAppOfArity ``FloatLib.Numerics.MixedError 4 then
    let arguments := bound.getAppArgs
    some (arguments[1]!, #[arguments[2]!, arguments[3]!])
  else if bound.isAppOfArity ``LE.le 4 then
    let difference := bound.getAppArgs[2]!.appArg!
    if difference.isAppOfArity ``HSub.hSub 6 then
      some (difference.appArg!, #[bound.appArg!])
    else none
  else none

/-- A propagation theorem generated in a module: its statement and proof, the positions of its
input-bound hypotheses among its binders, and the propagation theorems, auxiliary lemmas and
primitive theorems its proof uses. -/
structure PropagationTheorem where
  /-- The statement. -/
  type : Expr
  /-- The proof. -/
  proof : Expr
  /-- The positions of the input-bound hypotheses among the statement's binders. -/
  bounds : Array Nat
  /-- The propagation theorems the proof applies. -/
  arms : Array Name
  /-- The auxiliary lemmas the analysis added that the proof uses (`addedDeclarations`). -/
  auxiliary : Array ConstantInfo
  /-- The primitive theorems, correctly rounded functions and enclosures, that the proof and the
  auxiliary lemmas use. -/
  primitives : Array Name

/-- A declared propagation theorem as the modules importing it read it: its name, the positions of
its input-bound hypotheses, the propagation theorems its proof applies and the primitive theorems
its proof and auxiliary lemmas use. Their proofs are not visible to an importing module. -/
structure DeclaredPropagation where
  /-- The theorem. -/
  name : Name
  /-- The positions of the input-bound hypotheses among the statement's binders. -/
  bounds : Array Nat
  /-- The propagation theorems the proof applies. -/
  arms : Array Name
  /-- The primitive theorems the proof and its auxiliary lemmas use. -/
  primitives : Array Name

/-- The record of `generated` declared as `name`. -/
def PropagationTheorem.declaration (generated : PropagationTheorem) (name : Name) :
    DeclaredPropagation :=
  { name, bounds := generated.bounds, arms := generated.arms, primitives := generated.primitives }

/-- The propagation theorems generated in each module, by the name they are applied under: their
own, or that of an earlier propagation theorem with their statement (`recordPropagation`), and the
imported ones the module applies, as their declaring modules recorded them (`recordDeclared`). A
composite that does not prove drops the declarations it made, and a later one declares them again
from here. -/
initialize generatedPropagations : IO.Ref (Std.HashMap (Name × Name) PropagationTheorem) ←
  IO.mkRef {}

/-- Why the propagation theorem of a route was not generated, by module, name and a hash of the
registered rules it was analyzed with, on which the analysis depends. -/
initialize propagationFailures : IO.Ref (Std.HashMap (Name × Name × UInt64) String) ←
  IO.mkRef {}

/-- The propagation theorem `arm` generated in this module. The environment may not hold its proof
yet while the kernel checks it. -/
def propagationTheorem? (arm : Name) : CoreM (Option PropagationTheorem) := do
  return (← generatedPropagations.get).get? (← getMainModule, arm)

/-- The propagation theorems declared by `#roundoff_coverage` and `#roundoff_propagation`, in a
module and the modules it imports. -/
initialize propagationExtension :
    SimplePersistentEnvExtension DeclaredPropagation (Array DeclaredPropagation) ←
  registerSimplePersistentEnvExtension {
    addEntryFn := Array.push
    addImportedFn := fun modules => modules.foldl (· ++ ·) #[] }

/-- The earlier propagation theorem with the same statement that the name of a route's propagation
theorem stands for in each module (`recordPropagation`). -/
initialize propagationAliases : IO.Ref (Std.HashMap (Name × Name) Name) ← IO.mkRef {}

/-- The propagation theorem that `name` stands for in this module: an earlier one with the same
statement (`recordPropagation`), or `name`. -/
def propagationName (name : Name) : CoreM Name := do
  return ((← propagationAliases.get).get? (← getMainModule, name)).getD name

/-- A propagation theorem other than `name` whose statement is `type`: one generated in this module,
or one declared in it or in a module it imports (`propagationExtension`). -/
def samePropagation? (name : Name) (type : Expr) : CoreM (Option Name) := do
  let module ← getMainModule
  let found := (← generatedPropagations.get).toList.find? fun ((owner, arm), generated) =>
    owner == module && arm != name && generated.type == type
  if let some ((_, arm), _) := found then return some arm
  let environment ← getEnv
  return (propagationExtension.getState environment).findSome? fun declared =>
    if declared.name != name && (environment.find? declared.name).any (·.type == type) then
      some declared.name
    else none

/-- Record the declared propagation theorem `name` and those its proof applies, unless this module
generated them, as their declaring modules recorded them (`propagationExtension`): each proved by
its own constant, with no auxiliary lemmas to declare. -/
partial def recordDeclared (name : Name) : CoreM Unit := do
  let module ← getMainModule
  if (← generatedPropagations.get).contains (module, name) then return
  let environment ← getEnv
  let (some declared, some info) :=
      ((propagationExtension.getState environment).find? (·.name == name), environment.find? name) |
    throwError "roundoff: no declared propagation theorem {name}"
  generatedPropagations.modify (·.insert (module, name)
    { type := info.type, proof := mkConst name, bounds := declared.bounds, arms := declared.arms
      auxiliary := #[], primitives := declared.primitives })
  declared.arms.forM recordDeclared

/-- Record `generated`, the propagation theorem of a route analyzed for `name` in this module, and
return the name to apply it under: that of an earlier propagation theorem with its statement
(`samePropagation?`), which `name` then stands for, or `name`. The earlier theorem keeps its own
record (`recordDeclared`): its proof may apply other propagation and primitive theorems than
`generated`'s. -/
def recordPropagation (name : Name) (generated : PropagationTheorem) : CoreM Name := do
  let module ← getMainModule
  let some earlier ← samePropagation? name generated.type |
    generatedPropagations.modify (·.insert (module, name) generated)
    return name
  propagationAliases.modify (·.insert (module, name) earlier)
  recordDeclared earlier
  return earlier

/-- The constants of `terms` and of the statements and values of `auxiliary`. -/
def usedConstants (terms : Array Expr) (auxiliary : Array ConstantInfo := #[]) : NameSet :=
  let bodies := auxiliary.foldl (fun bodies info => (bodies.push info.type) ++ info.value?.toArray)
    terms
  bodies.foldl (fun used term => term.getUsedConstants.foldl (·.insert ·) used) {}

/-- The primitive theorems, correctly rounded functions and enclosures, among the constants of
`terms` and of the auxiliary lemmas `auxiliary`. -/
def primitivesOf (terms : Array Expr) (auxiliary : Array ConstantInfo := #[]) : Array Name :=
  let used := usedConstants terms auxiliary
  (primitiveRoundings ++ primitiveEnclosures).filter used.contains

/-- The propagation theorems among the constants of `terms` and of the auxiliary lemmas
`auxiliary`. -/
def directArms (terms : Array Expr) (auxiliary : Array ConstantInfo := #[]) :
    CoreM (Array Name) := do
  let generated ← generatedPropagations.get
  let module ← getMainModule
  return (usedConstants terms auxiliary).foldl (init := #[]) fun arms constant =>
    if generated.contains (module, constant) then arms.push constant else arms

/-- `arm` and the propagation theorems its proof applies, directly or through others, after those
already in `state`, each after the ones its proof applies. -/
private partial def closeArm (generated : Std.HashMap (Name × Name) PropagationTheorem)
    (module arm : Name) (state : NameSet × Array Name) : NameSet × Array Name :=
  if state.1.contains arm then state else
  let state := (state.1.insert arm, state.2)
  let dependencies := (generated.get? (module, arm)).map (·.arms) |>.getD #[]
  let (seen, order) := dependencies.foldl (fun state arm => closeArm generated module arm state)
    state
  (seen, order.push arm)

/-- The propagation theorems that `terms` and the auxiliary lemmas `auxiliary` apply, directly or
through the proofs and auxiliary lemmas of others, each after the ones its own proof applies. -/
def armClosure (terms : Array Expr) (auxiliary : Array ConstantInfo := #[]) :
    CoreM (Array Name) := do
  let generated ← generatedPropagations.get
  let module ← getMainModule
  return ((← directArms terms auxiliary).foldl
    (fun state arm => closeArm generated module arm state) ({}, #[])).2

/-- Describe the generated theorem `type`, whose first `arity` binders are the computation's
arguments it keeps, with proof `proof` and the auxiliary lemmas `auxiliary` it keeps. The contracts
of `functions` are among those binders. -/
def describeTheorem (entry : CoverageEntry) (arity : Nat) (type proof : Expr)
    (functions : Array String) (auxiliary : Array ConstantInfo := #[]) : MetaM CoverageEntry :=
  forallBoundedTelescope type arity fun _ body => do
    let body := instantiateOuterLets body
    let (premises, conclusion) := match body with
      | .forallE _ conditions conclusion _ =>
        if conclusion.hasLooseBVars then (#[], body) else (conjuncts conditions, conclusion)
      | _ => (#[], body)
    let some (reference, budgets) := boundParts conclusion |
      throwError "#roundoff_coverage: unexpected generated conclusion {conclusion}"
    let mut classes : Array (String × Nat) :=
      if functions.isEmpty then #[] else #[("function contract", functions.size)]
    for premise in premises do
      let kind ← premiseClass premise
      classes := match classes.findIdx? (·.1 == kind) with
        | some index => classes.modify index fun (kind, count) => (kind, count + 1)
        | none => classes.push (kind, 1)
    -- A composed proof also uses the primitive theorems of the propagation theorems it applies and
    -- of the auxiliary lemmas they keep.
    let mut used := usedConstants #[proof] auxiliary
    for arm in entry.arms do
      if let some generated ← propagationTheorem? arm then
        used := generated.primitives.foldl (·.insert ·) used
    let contracts := (primitiveRoundings ++ primitiveEnclosures).filter used.contains
    let mut caveats := #[]
    if ← budgets.anyM decodesComputedValue then
      caveats := caveats.push "budget decodes computed values"
    if ← decodesComputedValue reference then
      caveats := caveats.push "reference decodes computed values"
    if primitiveEnclosures.any used.contains then
      caveats := caveats.push "enclosure in place of a rounding theorem"
    let apriori := caveats.isEmpty && classes.isEmpty
    let usesIEEE ← lambdaBoundedTelescope proof arity fun arguments value =>
      arguments.anyM fun argument => do
        let isHypothesis := ((← inferType argument).eq?.map fun (_, left, _) =>
          left.isAppOf ``FloatFormat.isIEEE).getD false
        return isHypothesis && mentions value (· == argument.fvarId!)
    return { entry with
      status := if apriori then .proved else .conditional
      reference := (← ppExpr reference).pretty 1000000
      referenceSize := distinctSubterms #[reference]
      budget := if budgets.all (·.nat? == some 0) then 0 else distinctSubterms budgets
      premises := classes, caveats, contracts, usesIEEE, functions }

/-! ## Analysis -/

/-- The proof `roundoff_proof%` generates for `computation`, in mixed mode or not. The
elaboration records no info trees: nothing in the source corresponds to the generated terms. -/
def generateProof (computation : Expr) (mixed : Bool) : TermElabM Expr :=
  withEnableInfoTree false do
    let term ← exprToSyntax computation
    let generated ← if mixed then `(roundoff_proof% mixed $term) else `(roundoff_proof% $term)
    let proof ← withoutErrToSorry <| elabTerm generated none
    synthesizeSyntheticMVarsNoPostponing
    instantiateMVars proof

/-- The message of a failed analysis on one line. -/
def failureReason (error : Exception) : TermElabM String := do
  if error.isMaxHeartbeat then
    return "the analysis exceeded the default heartbeat budget of a declaration"
  let text ← error.toMessageData.toString
  return " ".intercalate (text.splitOn "\n" |>.map (·.trimAscii.toString) |>.filter (· ≠ ""))

/-- The last component of the name of a route's theorem: `forward_error` and
`forward_mixedError` for the configured scalar, `forward_nativeError` and `forward_roundedError`
for the other scalars. -/
def theoremSuffix (route : String) (backend : CoverageBackend) (mixed : Bool) : Name :=
  let kind := match backend, mixed with
    | .configured, false => "error"
    | .configured, true => "mixedError"
    | .native, _ => "nativeError"
    | .rounded, _ => "roundedError"
  .mkSimple s!"{route.replace " " "_"}_{kind}"

/-! ## Composition -/

/-- The name of the propagation theorem of `route` in this mode, in snake case, as
`forward_error_of_input_bounds` or `none_two_pass_mixed_error_of_input_bounds`. -/
def propagationSuffix (route : String) (mixed : Bool) : Name :=
  let snake (word : String) : String := word.foldl (fun text c =>
    if c.isUpper then text ++ (if text.isEmpty then "" else "_") ++ c.toLower.toString
    else text.push c) ""
  let words := (route.splitOn " ").filter (· ≠ "") ++ [if mixed then "mixedError" else "error"]
  .mkSimple ("_".intercalate (words.map snake) ++ "_of_input_bounds")

/-- The name of the theorem `suffix` about `declaration` in the namespace `space`: `space`, the
name of `declaration` less the root namespace it shares with `space`, and `suffix`, as
`TorchLean.Coverage.Elementwise.Op.exp.forward_error` for `forward_error` of `TorchLean.Op.exp`. -/
def theoremName (space declaration suffix : Name) : Name :=
  space ++ declaration.replacePrefix space.getRoot .anonymous ++ suffix

/-- The record of the first route recorded by `#roundoff_coverage`, in this module or one it
imports, whose theorem's statement is `type`. -/
def sameStatement? (type : Expr) : CoreM (Option CoverageEntry) := do
  let environment ← getEnv
  return (coverageExtension.getState environment).find? fun entry =>
    (environment.find? entry.evidence).any (·.type == type)

/-- The record of the route `entry` that records the theorem of the earlier route `earlier`: the
operator, route, scalar and mode of `entry` with the theorem of `earlier` and the description
`describeTheorem` gave it from the proof of its declaring module, which a module importing it
cannot read. -/
def CoverageEntry.reusing (entry earlier : CoverageEntry) : CoverageEntry :=
  { earlier with
    family := entry.family, operator := entry.operator, route := entry.route,
    backend := entry.backend, mode := entry.mode, link? := entry.link? }

/-- Whether `type` may hold values of `scalar`: it mentions `scalar` once reduced, or it is an
inductive type, other than those in `visited`, one of whose constructors has a field that may. -/
partial def holdsScalar (scalar type : Expr) (visited : NameSet := {}) : MetaM Bool := do
  let type ← whnf type
  if (type.find? (· == scalar)).isSome then return true
  let .const name levels := type.getAppFn | return false
  if visited.contains name then return false
  let some (.inductInfo info) := (← getEnv).find? name | return false
  let parameters := type.getAppArgs.extract 0 info.numParams
  info.ctors.anyM fun constructor => do
    let fields ← inferType (mkAppN (mkConst constructor levels) parameters)
    forallTelescopeReducing fields fun fields _ => fields.anyM fun field => do
      holdsScalar scalar (← inferType field) (visited.insert name)

/-- `computation` with, after each input that is a tensor over its scalar or a scalar, real
references and budgets (functions of the coordinate for a tensor) and the hypothesis `hinput`
that the input lies within them: `|toReal (input c) - inputReference c| ≤ inputError c` at every
coordinate, or in mixed analysis `MixedError (toReal (input c)) (inputReference c)
(inputRelative c) (inputAbsolute c)`, with the positions of these hypotheses among its binders.
There is none when another argument may hold values of the scalar (`holdsScalar`), such as an
optional tensor or a family of tensors, whose error it would not bound. -/
def withInputBounds (computation : Expr) (mixed : Bool) : MetaM (Option (Expr × Array Nat)) :=
  lambdaTelescope computation fun arguments body => do
    let observe := body.appFn!
    let scalar ← inferType body.appArg!
    let rec bind (index : Nat) (binders : Array Expr) (bounds : Array Nat) :
        MetaM (Option (Expr × Array Nat)) := do
      if h : index < arguments.size then
        let input := arguments[index]
        let binders := binders.push input
        let declaration ← input.fvarId!.getDecl
        let type ← instantiateMVars declaration.type
        if declaration.binderInfo.isInstImplicit || (← isProp type) ||
            !(← holdsScalar scalar type) then
          return ← bind (index + 1) binders bounds
        let whnfType ← whnfR type
        let tensor := whnfType.isAppOfArity ``Tensor.Rep 3 &&
          whnfType.getAppArgs[0]! == scalar
        unless tensor || type == scalar do return none
        -- A tensor is read at a coordinate, with the storage its type names. A scalar is its value.
        let read ← if tensor then do
            let get ← mkConstWithFreshMVarLevels ``Tensor.Rep.get
            let (parameters, _, _) ← forallMetaTelescopeReducing (← inferType get) (some 4)
            unless ← isDefEq parameters[3]! input do
              throwError "roundoff: the input {input} is not read as a tensor of its type"
            instantiateMVars (mkAppN get parameters)
          else pure input
        let coordinate? ← if tensor then
            forallTelescope (← inferType read) fun points _ => some <$> inferType points[0]!
          else pure none
        let real ← match coordinate? with
          | some coordinate => mkArrow coordinate (mkConst ``Real)
          | none => pure (mkConst ``Real)
        let stem := declaration.userName.eraseMacroScopes.toString
        let parts := if mixed then #["Reference", "Relative", "Absolute"]
          else #["Reference", "Error"]
        withLocalDeclsDND (parts.map fun part => (.mkSimple (stem ++ part), real)) fun reals => do
          let boundAt (value : Expr) (values : Array Expr) : MetaM Expr := do
            if mixed then
              mkAppM ``FloatLib.Numerics.MixedError #[value, values[0]!, values[1]!, values[2]!]
            else
              let difference ← mkAppM ``HSub.hSub #[value, values[0]!]
              mkAppM ``LE.le #[← mkAppM ``abs #[difference], values[1]!]
          let statement ← match coordinate? with
            | some coordinate => withLocalDeclD `c coordinate fun point => do
                mkForallFVars #[point]
                  (← boundAt (mkApp observe (mkApp read point)) (reals.map (mkApp · point)))
            | none => boundAt (mkApp observe read) reals
          withLocalDeclD (.mkSimple ("h" ++ stem)) statement fun hypothesis =>
            let binders := binders ++ reals
            bind (index + 1) (binders.push hypothesis) (bounds.push binders.size)
      else return some (← mkLambdaFVars binders body, bounds)
    bind 0 #[] #[]

/-- The tensor-returning definitions, other than instances, among the constants of `term`. -/
def tensorDefinitions (term : Expr) : MetaM (Array Name) := do
  let mut found := #[]
  for constant in term.getUsedConstants do
    let some (.defnInfo info) := (← getEnv).find? constant | continue
    if constant.isInternal || (← isInstance constant) then continue
    let returnsTensor ← forallTelescope info.type fun _ result => do
      let result ← whnfR result
      return result.isAppOf ``Tensor.Rep || (result.isAppOf ``Prod &&
        (result.find? fun part => part.isConstOf ``Tensor.Rep ||
          part.isConstOf ``Tensor).isSome)
    if returnsTensor then found := found.push constant
  return found

/-- The definition producing the container that the observed value `body` reads, through
projections of a tuple. -/
partial def readProducer? (body : Expr) : Option Name :=
  let rec producer : Expr → Option Name
    | .proj _ _ inner => producer inner.consumeMData
    | term =>
      if term.isAppOf ``Prod.fst || term.isAppOf ``Prod.snd then
        producer term.appArg!.consumeMData
      else term.getAppFn.constName?
  let read := body.appArg!.consumeMData
  if read.isAppOfArity ``Tensor.Rep.get 5 then
    producer read.getAppArgs[3]!.consumeMData
  else none

/-- `constant` after the definitions it calls among `calls`, after those already in `state`. -/
private partial def callOrder (calls : Std.HashMap Name (Array Name)) (constant : Name)
    (state : NameSet × Array Name) : NameSet × Array Name :=
  if state.1.contains constant then state else
  let state := (state.1.insert constant, state.2)
  let (seen, order) := ((calls.get? constant).getD #[]).foldl
    (fun state callee => callOrder calls callee state) state
  (seen, order.push constant)

/-- The tensor-returning definitions that the definitions of the observed computation call,
within `depth` levels of their values by the shortest path, each after the ones it calls: the
operations a composite is built from. The search passes through tensor-returning definitions only,
and the definition producing the observed container, the composite itself, is not among them. -/
def callees (computation : Expr) (depth : Nat := 2) : MetaM (Array Name) := do
  let body ← lambdaTelescope computation fun _ body => pure body
  let roots ← tensorDefinitions body
  -- Breadth first, so that each definition is expanded with the most levels left to it.
  let mut queue := roots.map (·, depth)
  let mut reached : NameSet := roots.foldl (·.insert ·) {}
  let mut calls : Std.HashMap Name (Array Name) := {}
  let mut index := 0
  while index < queue.size do
    let (constant, left) := queue[index]!
    index := index + 1
    if left == 0 then continue
    let found ← match (← getEnv).find? constant |>.bind (·.value?) with
      | some value => tensorDefinitions value
      | none => pure #[]
    calls := calls.insert constant found
    for callee in found do
      unless reached.contains callee do
        reached := reached.insert callee
        queue := queue.push (callee, left - 1)
  let order := (roots.foldl (fun state root => callOrder calls root state) ({}, #[])).2
  let own := readProducer? body
  return order.filter (some · != own)

/-- The value a generated conclusion bounds: `actual` in `|actual - reference| ≤ error` or in a
mixed bound with or without the finiteness of the computed value. -/
def boundedValue? (conclusion : Expr) : Option Expr :=
  let bound := if conclusion.isAppOfArity ``And 2 then conclusion.appArg! else conclusion
  if bound.isAppOfArity ``FloatLib.Numerics.MixedError 4 then some (bound.getArg! 0)
  else if bound.isAppOfArity ``LE.le 4 && (bound.getArg! 2).isApp then
    let difference := (bound.getArg! 2).appArg!
    if difference.isAppOfArity ``HSub.hSub 6 then some (difference.getArg! 4) else none
  else none

/-- Whether the propagation theorem `name`, read as the analysis indexes it, with the bound of a
mixed statement and its reads normalized by `session`, serves a composite: it bounds a read of a
tensor that an operation produces, not of one of its inputs, which would match every read, and its
reference reads an input reference, unlike that of a constant. -/
def isArm (session : Session) (name : Name) : MetaM Bool := do
  let rule ← mkConstWithFreshMVarLevels name
  let bound ← forallTelescopeReducing (← inferType rule) fun arguments conclusion => do
    if conclusion.isAppOfArity ``And 2 then
      mkLambdaFVars arguments (← mkAppM ``And.right #[mkAppN rule arguments])
    else pure rule
  let normalized ← normalizeRule session bound
  forallTelescopeReducing (← inferType normalized) fun binders conclusion => do
    let conclusion := instantiateOuterLets conclusion
    let (some (reference, _), some observed) := (boundParts conclusion, boundedValue? conclusion) |
      return false
    unless observed.isAppOf ``ExecFloat.Binary.toReal && observed.isApp do return false
    let read := observed.appArg!.consumeMData
    unless read.isAppOfArity ``Tensor.Rep.get 5 do return false
    unless read.getAppArgs[3]!.consumeMData.getAppFn.isConst do return false
    let references ← binders.filterM fun binder => do
      return (← binder.fvarId!.getUserName).toString.endsWith "Reference"
    return references.any fun binder => reference.containsFVar binder.fvarId!

/-- `term` without metadata, eta expansion and the transports `Eq.mpr`, `Eq.mp`, `cast` and `id`
around it. -/
partial def stripTransports (term : Expr) : Expr :=
  let term := term.consumeMData.eta
  if term.isAppOfArity ``Eq.mpr 4 || term.isAppOfArity ``Eq.mp 4 || term.isAppOfArity ``cast 4 ||
      term.isAppOfArity ``id 2 then stripTransports term.appArg!
  else term

/-- Whether `term` is one of the conditions a generated theorem binds together, as `roundoff`
binds the premises it does not discharge: up to transports, a left part of a chain of right parts
of their conjunction. -/
partial def isCondition (term : Expr) : Bool :=
  let rec evidence (term : Expr) : Bool :=
    let term := stripTransports term
    if term.isBVar || term.isFVar then true
    else term.isAppOfArity ``And.right 3 && evidence term.appArg!
  let term := stripTransports term
  term.isAppOfArity ``And.left 3 && evidence term.appArg!

/-- The propagation theorems among `generated` whose input-bound hypotheses an application in
`term` takes from conditions (`isCondition`), after those in `state`. -/
private partial def conditionedArms (generated : Std.HashMap (Name × Name) PropagationTheorem)
    (module : Name) (term : Expr) (state : Std.HashSet Expr × Array Name) :
    Std.HashSet Expr × Array Name :=
  if state.1.contains term then state else
  let found : Bool := match term.getAppFn with
    | .const name _ => (generated.get? (module, name)).any fun arm =>
        !state.2.contains name && arm.bounds.any fun position =>
          (term.getAppArgs[position]?.map isCondition).getD false
    | _ => false
  let state := (state.1.insert term, if found then state.2.push term.getAppFn.constName! else
    state.2)
  match term with
  | .app function argument =>
    conditionedArms generated module argument (conditionedArms generated module function state)
  | .lam _ type body _ | .forallE _ type body _ =>
    conditionedArms generated module body (conditionedArms generated module type state)
  | .letE _ type value body _ =>
    conditionedArms generated module body
      (conditionedArms generated module value (conditionedArms generated module type state))
  | .mdata _ inner | .proj _ _ inner => conditionedArms generated module inner state
  | _ => state

/-- The propagation theorems whose input-bound hypotheses `proof`, with its `let` bindings
substituted and its applied functions reduced, and the auxiliary lemmas `auxiliary` take from
their conditions instead of an analysis of the input: the bounds they did not supply. -/
def unsuppliedBounds (proof : Expr) (auxiliary : Array ConstantInfo := #[]) :
    MetaM (Array Name) := do
  let generated ← generatedPropagations.get
  let module ← getMainModule
  let bodies := auxiliary.foldl (fun bodies info => bodies ++ info.value?.toArray) #[proof]
  let mut state := ({}, #[])
  for body in bodies do
    let body ← Core.betaReduce (← zetaReduce (← instantiateMVars body))
    state := conditionedArms generated module body state
  return state.2

/-- `name` and the declarations its type and value use, after those in `state`, that `environment`
holds and `base` does not, other than the propagation theorems of `generated`, each after the ones
it uses. -/
private partial def addedFrom (environment base : Environment)
    (generated : Std.HashMap (Name × Name) PropagationTheorem) (module name : Name)
    (state : NameSet × Array ConstantInfo) : NameSet × Array ConstantInfo :=
  if state.1.contains name || base.contains name || generated.contains (module, name) then state
  else
    let state := (state.1.insert name, state.2)
    match environment.find? name with
    | none => state
    | some info =>
      let used := info.type.getUsedConstants ++ (info.value?.map (·.getUsedConstants)).getD #[]
      let (seen, order) := used.foldl
        (fun state name => addedFrom environment base generated module name state) state
      (seen, order.push info)

/-- The declarations that the constants of `terms` use, directly or through others, that the
environment holds and `base` does not, other than propagation theorems, each after the ones it
uses: the auxiliary lemmas, such as those of simplification sets, that an analysis added. -/
def addedDeclarations (base : Environment) (terms : Array Expr) :
    CoreM (Array ConstantInfo) := do
  let environment ← getEnv
  let generated ← generatedPropagations.get
  let module ← getMainModule
  return (terms.foldl (fun state term => term.getUsedConstants.foldl
    (fun state name => addedFrom environment base generated module name state) state)
    ({}, #[])).2

/-- The matchers whose splitter or equation `name` may be, given the auxiliary lemmas
`declarations`. `Match.getEquationsFor` declares them, when a case analysis splits an application
of the matcher, under the private prefix of the current module after removing the matcher's own
(`mkPrivateName`), so their names do not tell apart private matchers of one name in different
modules. The matcher is the constant of that name, with or without a private prefix, that a
declaration of `declarations` under the same prefix uses: the left side of each equation applies
it, as does the splitter of a matcher without overlapping patterns. When none does, as for the
splitter of a matcher with overlapping patterns alone, they are the matchers of that name under no
private prefix or the private prefix of one of the modules of `environment`. -/
def auxiliaryMatchers (environment : Environment) (declarations : Array ConstantInfo)
    (name : Name) : Array Name :=
  match privateToUserName? name with
  | some (.str matcher suffix) =>
    if suffix == "splitter" || suffix.startsWith "eq_" then
      let used := declarations.foldl (init := #[]) fun used info =>
        if info.name.getPrefix == name.getPrefix then
          used ++ info.type.getUsedConstants ++ (info.value?.map (·.getUsedConstants)).getD #[]
        else used
      match used.find? fun constant =>
          privateToUserName constant == matcher && isMatcherCore environment constant with
      | some found => #[found]
      | none =>
        let candidates := (environment.allImportedModuleNames.push environment.mainModule).map
          (mkPrivateNameCore · matcher)
        (candidates.push matcher).filter (isMatcherCore environment)
    else #[]
  | _ => #[]

/-- Declare again those of the auxiliary lemmas `declarations` that the environment does not hold,
as a restored state lacks them. The splitter and equations of a matcher are realized again
(`Match.getEquationsFor`) instead: declared by `addDecl`, they would be in the environment without
the matcher's entry in the equations' extension, so every later case analysis of the matcher would
fail. Their matcher is the one `auxiliaryMatchers` finds, and each one realized must have the
statement recorded. -/
def restoreDeclarations (declarations : Array ConstantInfo) : CoreM Unit := do
  for declaration in declarations do
    if (← getEnv).contains declaration.name then continue
    match auxiliaryMatchers (← getEnv) declarations declaration.name with
    | #[] => pure ()
    | #[matcher] =>
      discard <| (Match.getEquationsFor matcher).run'
      if let some realized := (← getEnv).find? declaration.name then
        unless realized.type == declaration.type do
          throwError "roundoff: {declaration.name}, realized again from {matcher}, has another \
            statement than the one recorded"
        continue
    | matchers =>
      throwError "roundoff: {declaration.name} may be the splitter of any of {matchers}"
    match declaration with
    | .thmInfo value => addDecl (.thmDecl value)
    | .defnInfo value => addDecl (.defnDecl value)
    | _ => throwError "roundoff: {declaration.name} is no theorem or definition to declare again"

/-- Declare the propagation theorems `arms`, in their order, that the environment does not hold,
each after the auxiliary lemmas its proof uses. -/
def declareArms (arms : Array Name) : CoreM Unit := do
  for arm in arms do
    if (← getEnv).contains arm then continue
    let some generated ← propagationTheorem? arm |
      throwError "roundoff: no propagation theorem {arm}"
    restoreDeclarations generated.auxiliary
    addDecl (.thmDecl
      { name := arm, levelParams := [], type := generated.type, value := generated.proof })
    modifyEnv (propagationExtension.addEntry · (generated.declaration arm))
    addDocStringCore arm "Error bound of a route of an operation under the hypothesis that each \
      input lies within real budgets of real references: a propagation theorem generated by \
      `#roundoff_coverage`."

/-- `term`, under binders whose `let` values are `values` (`none` for other binders), with these
values substituted, or `none` when it uses the variable of another binder. -/
def closeTerm (values : Array (Option Expr)) (term : Expr) : Option Expr := Id.run do
  if !term.hasLooseBVars then return some term
  if term.looseBVarRange > values.size then return none
  for index in [0:term.looseBVarRange] do
    if term.hasLooseBVar index && values[values.size - 1 - index]!.isNone then return none
  return some (term.instantiateRev (values.map (·.getD (.sort .zero))))

private partial def readSitesAux (targets : Array FVarId) (values : Array (Option Expr))
    (term : Expr) : StateT (Array Expr × Std.HashSet Expr) Option Unit := do
  unless term.hasFVar do return
  if !term.hasLooseBVars then
    if (← get).2.contains term then return
    modify fun (sites, seen) => (sites, seen.insert term)
  if let .fvar id := term.getAppFn then
    if targets.contains id then
      let #[site] := term.getAppArgs | failure
      let some site := closeTerm values site | failure
      modify fun (sites, seen) => (sites.push site, seen)
      return
  match term with
  | .app function argument =>
    readSitesAux targets values function
    readSitesAux targets values argument
  | .lam _ type body _ | .forallE _ type body _ =>
    readSitesAux targets values type
    readSitesAux targets (values.push none) body
  | .letE _ type value body _ =>
    readSitesAux targets values type
    readSitesAux targets values value
    readSitesAux targets (values.push (closeTerm values value)) body
  | .mdata _ inner | .proj _ _ inner => readSitesAux targets values inner
  | _ => pure ()

/-- The coordinates at which `term` applies the functions `targets`, with its `let` variables
replaced by their values, or `none` when it uses one of them otherwise. -/
def readSites (targets : Array FVarId) (term : Expr) : Option (Array Expr) :=
  ((readSitesAux targets #[] term).run (#[], {})).map (·.2.1)

/-- `term` with each application of one of the functions `targets` replaced by the matching entry
of `values`. -/
def replaceReads (targets : Array FVarId) (values : Array Expr) (term : Expr) : Expr :=
  term.replace fun part => match part.getAppFn with
    | .fvar id => if part.isApp then (targets.idxOf? id).map (values[·]!) else none
    | _ => none

/-- The propagation theorem `type`, whose input bounds sit at `bounds` among its first `arity`
binders, with each tensor input that its statement reads at one coordinate only bounded at that
coordinate. That bound, a real reference, real budgets and their hypothesis, comes after the other
binders, and the proof applies `proof` to the input's values updated at the coordinate
(`abs_sub_update_le`, or `mixedError_update` when `mixed`). -/
def localizeBounds (type proof : Expr) (arity : Nat) (bounds : Array Nat) (mixed : Bool) :
    MetaM (Expr × Expr × Array Nat) :=
  forallBoundedTelescope type arity fun binders statement => do
    if binders.size != arity then return (type, proof, bounds)
    let parts := if mixed then 3 else 2
    let mut localized : Array (Nat × Expr) := #[]
    for position in bounds do
      if position < parts then continue
      let hypothesis := binders[position]!
      unless (← inferType hypothesis).isForall do continue
      let functions := (binders.extract (position - parts) position).map (·.fvarId!)
      let some sites := readSites functions statement | continue
      let some site := sites[0]? | continue
      unless ← sites.allM (isDefEq site ·) do continue
      -- Another binder whose type names the input's references keeps them universal.
      let dropped := functions.push hypothesis.fvarId!
      let elsewhere ← binders.anyM fun binder => do
        if dropped.contains binder.fvarId! then return false
        return dropped.any (← inferType binder).containsFVar
      unless elsewhere do localized := localized.push (position, site)
    if localized.isEmpty then return (type, proof, bounds)
    let mut declarations : Array (Name × BinderInfo × (Array Expr → MetaM Expr)) := #[]
    for (position, site) in localized do
      let hypothesis := binders[position]!
      let olds := binders.extract (position - parts) position
      for old in olds do
        declarations := declarations.push (← old.fvarId!.getUserName, .default,
          fun _ => pure (mkConst ``Real))
      let start := declarations.size - parts
      let targets := olds.map (·.fvarId!)
      let body := (← inferType hypothesis).bindingBody!.instantiate1 site
      declarations := declarations.push (← hypothesis.fvarId!.getUserName, .default,
        fun added => pure (replaceReads targets (added.extract start (start + parts)) body))
    let attempt : MetaM (Expr × Expr × Array Nat) := withLocalDecls declarations fun added => do
      let mut reals := #[]
      let mut functions := #[]
      let mut replacements : Array (FVarId × Expr) := #[]
      let mut equations := #[]
      for (position, site) in localized, index in [0:localized.size] do
        let hypothesis := binders[position]!
        let bound ← inferType hypothesis
        let coordinate := bound.bindingDomain!
        let targets := (binders.extract (position - parts) position).map (·.fvarId!)
        let points := added.extract (index * (parts + 1)) (index * (parts + 1) + parts)
        let value ← withLocalDeclD `c coordinate fun c => do
          let some value := boundedValue? (bound.bindingBody!.instantiate1 c) |
            throwError "roundoff: the input bound {bound} bounds no value"
          mkLambdaFVars #[c] value
        let level ← getLevel coordinate
        let decidable := mkApp (mkConst ``Classical.typeDecidableEq [level]) coordinate
        let transfer := if mixed then ``mixedError_update else ``abs_sub_update_le
        let point := added[index * (parts + 1) + parts]!
        let transferred := mkAppN (mkConst transfer [level])
          (#[coordinate, decidable, value, site] ++ points |>.push point)
        let updated ← forallBoundedTelescope (← inferType transferred) (some 1) fun c bound => do
          let some (reference, budgets) := boundParts bound |
            throwError "roundoff: the transferred bound {bound} has no budgets"
          (#[reference] ++ budgets).mapM fun part => return (← mkLambdaFVars c part).eta
        for function in updated, real in points do
          let arguments := function.getAppArgs
          unless function.isAppOfArity ``Function.update 6 do
            throwError "roundoff: {function} is no update"
          equations := equations.push (mkAppN
            (mkConst ``Function.update_self function.getAppFn.constLevels!)
            #[arguments[0]!, arguments[1]!, arguments[2]!, site, real, arguments[3]!])
        reals := reals ++ points
        functions := functions ++ targets
        replacements := (replacements ++ targets.zip updated).push (hypothesis.fvarId!, transferred)
      let localStatement := replaceReads functions reals statement
      let dropped := replacements.map (·.1)
      if dropped.any localStatement.containsFVar then
        throwError "roundoff: the statement reads a localized input otherwise"
      let arguments := binders.map fun binder =>
        ((replacements.find? (·.1 == binder.fvarId!)).map (·.2)).getD binder
      -- `mkAppN` does not check that the replacements have the types of the binders they fill.
      let mut remaining := type
      for argument in arguments, binder in binders do
        let .forallE _ domain body _ := remaining | throwError "roundoff: too many arguments"
        if dropped.contains binder.fvarId! then
          unless ← isDefEq (← inferType argument) domain do
            throwError "roundoff: {argument} does not fill {binder}"
        remaining := body.instantiate1 argument
      let mut equality ← mkEqRefl (← mkLambdaFVars reals localStatement)
      for equation in equations do
        equality ← mkCongr equality equation
      let value ← mkExpectedTypeHint (← mkEqMP equality (mkAppN proof arguments)) localStatement
      let kept := binders.filter (!dropped.contains ·.fvarId!)
      let positions := bounds.map fun position =>
        match localized.findIdx? (·.1 == position) with
        | some index => kept.size + (index + 1) * (parts + 1) - 1
        | none => (kept.idxOf? binders[position]!).getD position
      let all := kept ++ added
      return (← mkForallFVars all localStatement, ← mkLambdaFVars all value, positions)
    try attempt catch error =>
      if ← isTracingEnabledFor `roundoff then
        addTrace `roundoff m!"the input bounds stay universal: {error.toMessageData}"
      return (type, proof, bounds)

/-- The callees whose propagation theorems take the bound of each input their statement reads at
one coordinate at that coordinate only (`localizeBounds`). `Normalization.normalize` reads its
statistics, scale and shift at the broadcast position of the output coordinate, so a composite
applying it bounds them on the output coordinate's slice only. -/
def localizedCallees : Array Name := #[`TorchLean.Normalization.normalize]

/-- The propagation theorem of the configured route `route` of a callee: the analysis of its
computation under the input bounds of `withInputBounds`, with the lemmas added since `base`. The
analysis must take no function contract and apply the input bounds of the propagation theorems it
uses. The theorem keeps every input bound and the other instances and hypotheses of the route that
its statement or proof uses (`dropUnused`), and when `localize` the bounds of inputs it reads at one
coordinate are taken there (`localizeBounds`). -/
def propagationTheoremOf (base : Environment) (route : Route) (mixed localize : Bool) :
    TermElabM PropagationTheorem := do
  let (computation, added) ← withContracts route.computation mixed
    fun computation added _ => pure (computation, added)
  unless added.isEmpty do throwError "roundoff: the route applies a function under a contract"
  let some (lifted, bounds) ← withInputBounds computation mixed |
    throwError "roundoff: an argument holds values of the scalar but is no tensor or scalar"
  let proof ← generateProof lifted mixed
  let type ← instantiateMVars (← inferType proof)
  let (arity, optional) ← lambdaTelescope lifted fun binders _ =>
    return (binders.size, (← optionalArguments binders).filter (!bounds.contains ·))
  let (proof, type, kept, _) ← dropUnused proof type arity optional
  let bounds := bounds.filterMap kept.idxOf?
  let auxiliary ← addedDeclarations base #[type, proof]
  let unsupplied ← unsuppliedBounds proof auxiliary
  unless unsupplied.isEmpty do throwError "roundoff: the input bounds of {unsupplied} are premises"
  let arms ← directArms #[type, proof] auxiliary
  let (type, proof, bounds) ← if localize then localizeBounds type proof kept.size bounds mixed
    else pure (type, proof, bounds)
  return { type, proof, bounds, arms, auxiliary, primitives := primitivesOf #[proof] auxiliary }

/-- The propagation theorems of the callees of `computation` (`callees`) in this mode, registered
as rules, and why the others are not generated. Each is the propagation theorem
(`propagationTheoremOf`) of a configured route of a callee, analyzed with its own heartbeat budget
and declared in the current namespace under `theoremName` with `propagationSuffix`, or an earlier
one with its statement (`recordPropagation`). A callee contributes when every route of it has one
and serves a composite (`isArm`). -/
def propagationArms (base : Environment) (computation : Expr) (mixed : Bool) :
    TermElabM (Array Name × Array String) := do
  let module ← getMainModule
  let reason (error : Exception) : TermElabM String := failureReason error
  -- The search, each callee's routes and each analysis have heartbeat budgets of their own.
  let found ← tryCatchRuntimeEx (Except.ok <$> withCurrHeartbeats (callees computation))
    (pure ∘ Except.error)
  if let .error error := found then return (#[], #[s!"the search for callees: {← reason error}"])
  let .ok found := found | return (#[], #[])
  let mut arms := #[]
  let mut failures := #[]
  for callee in found do
    -- A function returning several tensors has a route for each.
    let routes ← tryCatchRuntimeEx (Except.ok <$> withCurrHeartbeats do
          let routes ← operatorRoutes .configured callee
          if routes.any (!·.isEmpty) then return routes
          componentRoutes .configured callee)
        (pure ∘ Except.error)
    if let .error error := routes then
      failures := failures.push s!"{callee}: {← reason error}"
      continue
    let .ok (some routes) := routes | continue
    -- A rule makes its producer a boundary of normalization, so a callee is an arm only when
    -- every route of it has a propagation theorem: the analysis would not read the others.
    let rules := hash (roundoffRules.getState (← getEnv))
    let mut theorems := #[]
    let mut failure? := none
    for route in routes do
      let mut name ← propagationName
        (theoremName (← getCurrNamespace) callee (propagationSuffix route.route mixed))
      if (← propagationTheorem? name).isNone then
        if (← getEnv).contains name then
          failure? := some s!"{name}: declared by other means"
          break
        if let some known := (← propagationFailures.get).get? (module, name, rules) then
          failure? := some known
          break
        match ← tryCatchRuntimeEx (Except.ok <$> withCurrHeartbeats
            (propagationTheoremOf base route mixed (localizedCallees.contains callee)))
            (pure ∘ Except.error) with
        | .ok generated => name ← recordPropagation name generated
        | .error error =>
          let known := s!"{name}: {← reason error}"
          propagationFailures.modify (·.insert (module, name, rules) known)
          failure? := some known
          break
      let declared ← tryCatchRuntimeEx (Except.ok <$> withCurrHeartbeats do
          let some generated ← propagationTheorem? name | return
          declareArms ((← armClosure #[generated.proof] generated.auxiliary).push name))
        (pure ∘ Except.error)
      if let .error error := declared then
        failure? := some s!"{name}: {← reason error}"
        break
      theorems := theorems.push name
    if let some failure := failure? then
      failures := failures.push failure
      continue
    -- The analysis indexes a rule by its normalized statement, so that is the one to check.
    let before := roundoffRules.getState (← getEnv)
    modifyEnv (roundoffRules.modifyState · (· ++ theorems))
    let checked ← tryCatchRuntimeEx (Except.ok <$> withCurrHeartbeats do
        let session ← Roundoff.Session.prepare
        theorems.findM? fun name => return !(← isArm session name))
      (pure ∘ Except.error)
    let rejected? ← match checked with
      | .ok rejected? => pure <| rejected?.map fun name =>
        s!"{name}: its normalized conclusion does not read an operation"
      | .error error => do pure (some s!"{callee}: {← reason error}")
    if let some failure := rejected? then
      modifyEnv (roundoffRules.modifyState · fun _ => before)
      failures := failures.push failure
      continue
    arms := arms ++ theorems
  return (arms, failures)

/-- Analyze one route, declare its theorem as `name` when the analysis succeeds, and describe the
outcome. A first analysis whose statement the theorem of an earlier route has (`sameStatement?`)
records that theorem, with that route's description of it (`CoverageEntry.reusing`), instead of
declaring another. The theorem keeps the binders at `added`, those of the contracts of functions,
whose hypotheses are at `hypotheses`, and the instances and hypotheses of the computation
(`optionalArguments`) only where its statement or proof uses them. A configured route over the
heartbeat budget, and only such a route, is analyzed once more when its callees have propagation
theorems (`propagationArms`), with these as rules, keeping the ones its proof applies
(`armClosure`) as its arms; that second analysis always declares `name`. That analysis and the
reading of its proof each have a budget of their own. -/
def coverRoute (entry : CoverageEntry) (computation : Expr) (mixed : Bool) (name : Name)
    (ref : Syntax) (added : Array Nat := #[]) (hypotheses : Array (Nat × String) := #[]) :
    TermElabM CoverageEntry := do
  let saved ← saveState
  let base ← getEnv
  let analyze : TermElabM (Expr × Expr × Nat × Array String) := do
    let proof ← generateProof computation mixed
    let type ← instantiateMVars (← inferType proof)
    let (arity, optional) ← lambdaTelescope computation fun arguments _ =>
      return (arguments.size, added ++ (← optionalArguments arguments))
    let (proof, type, kept, functions) ← dropUnused proof type arity optional hypotheses
    return (type, proof, kept.size, functions)
  let declare (result : Expr × Expr × Nat × Array String) := do
    addDecl (.thmDecl { name, levelParams := [], type := result.1, value := result.2.1 })
    addDocStringCore name s!"Bound on the {entry.mode} error of the `{entry.route}` route of \
      `{entry.operator}` against its real reference, for the {entry.backend} scalar, generated \
      by `#roundoff_coverage`."
    return (name, result)
  let reuse (result : Expr × Expr × Nat × Array String) := do
    match ← sameStatement? result.1 with
    | some earlier => pure (earlier.evidence, result)
    | none => declare result
  let mut generated ← tryCatchRuntimeEx (Except.ok <$> (analyze >>= reuse))
    (pure ∘ Except.error)
  let mut arms := #[]
  let mut auxiliary := #[]
  let mut composed := ""
  let mut scope? := none
  if let .error error := generated then
    if error.isMaxHeartbeat && entry.backend == CoverageBackend.configured.label then
      saved.restore
      -- Later declarations get the macro scopes the first attempt left, as without a second one.
      let scope := (← getThe Core.State).nextMacroScope
      scope? := some scope
      try
        let (candidates, failures) ← propagationArms base computation mixed
        if ← isTracingEnabledFor `roundoff then
          for failure in failures do
            addTrace `roundoff m!"no propagation theorem for {name}: {failure}"
        let missing := if failures.isEmpty then "" else
          s!" ({failures.size} callees whose propagation theorems failed)"
        if candidates.isEmpty then
          unless failures.isEmpty do
            composed := s!"; no propagation theorem of its callees{missing}"
        else
          -- The propagation theorems and auxiliary lemmas the proof uses are kept, and nothing
          -- else of the attempt.
          let attempt ← tryCatchRuntimeEx (Except.ok <$> do
              let result ← withCurrHeartbeats analyze
              withCurrHeartbeats do
                let kept ← addedDeclarations base #[result.1, result.2.1]
                let unsupplied ← unsuppliedBounds result.2.1 kept
                unless unsupplied.isEmpty do
                  throwError "roundoff: the input bounds of {unsupplied} are premises"
                return (result, ← armClosure #[result.1, result.2.1] kept, kept))
            (pure ∘ Except.error)
          saved.restore
          match attempt with
          | .ok (result, used, kept) =>
            arms := used
            auxiliary := kept
            generated ← tryCatchRuntimeEx (Except.ok <$> withCurrHeartbeats do
                declareArms used
                restoreDeclarations kept
                declare result)
              (pure ∘ Except.error)
          | .error second =>
            composed := s!"; with the propagation theorems of {candidates.size} routes of its \
              callees as rules{missing}: {← failureReason second}"
      finally
        modifyThe Core.State fun state => { state with nextMacroScope := scope }
  let described ← match generated with
    | .error error => do
      let reason := (← failureReason error) ++ composed
      saved.restore
      let unsupported := reason.startsWith "roundoff: no numerical rule" ||
        reason.startsWith "roundoff: no proved arithmetic rule"
      pure { entry with status := if unsupported then .unsupported else .failed, reason }
    | .ok (evidence, type, proof, arity, functions) => do
      if evidence == name then addDeclarationRangesFromSyntax name ref
      -- A theorem of an earlier route keeps that route's description, not one of this proof.
      match (coverageExtension.getState (← getEnv)).find? (·.evidence == evidence) with
      | some earlier => pure (entry.reusing earlier)
      | none =>
        let entry := { entry with evidence, status := .conditional, arms }
        tryCatchRuntimeEx
          (withCurrHeartbeats <| describeTheorem entry arity type proof functions auxiliary)
          fun error => return { entry with
            caveats := #[s!"the statement was not classified: {← failureReason error}"] }
  -- Describing a composed theorem prints it, which takes macro scopes too.
  if let some scope := scope? then
    modifyThe Core.State fun state => { state with nextMacroScope := scope }
  return described

/-- Analyze every route of `operator` for every backend and mode, and record each. A route that
applies a registered operator specification to local variables is recorded with that
specification's outcome instead of being analyzed again. A configured route is analyzed under
the contracts of the functions its variables supply. The attributes in `fixed` are the given
terms. -/
def coverOperator (family operator : Name) (ref : Syntax) (fixed : Array (Name × Term) := #[]) :
    CommandElabM Unit := do
  let record (entry : CoverageEntry) := modifyEnv (coverageExtension.addEntry · entry)
  for backend in #[CoverageBackend.configured, .native, .rounded] do
    let base : CoverageEntry :=
      { family, operator, route := "", backend := backend.label, mode := ""
        status := .unsupported }
    let outcome ← liftTermElabM <|
      tryCatchRuntimeEx (Except.ok <$> operatorRoutes backend operator fixed)
        (pure ∘ Except.error)
    match outcome with
    | .error error =>
      record { base with status := .failed, reason := ← liftTermElabM (failureReason error) }
    | .ok none => record { base with reason := "the scalar type is fixed" }
    | .ok (some #[]) => record { base with reason := "its definition records no readable tensor" }
    | .ok (some routes) =>
      for route in routes do
        -- Each route is analyzed in a metavariable context of its own, so a universe that the
        -- route leaves open, as one of a type other than the scalar's, is fixed at 0.
        let route := { route with
          computation := route.computation.replaceLevel fun | .mvar _ => some .zero | _ => none }
        for mixed in backend.modes do
          let mode := if mixed then "mixed" else "absolute"
          let entry := { base with route := route.route, mode }
          let recorded := coverageExtension.getState (← getEnv)
          let link? := route.link?.filter (·.1 != operator)
          let linked := link?.bind fun (target, targetRoute) =>
            recorded.find? fun previous =>
              previous.operator == target && previous.route == targetRoute &&
                previous.backend == entry.backend && previous.mode == entry.mode
          match linked with
          | some previous => record { previous with family, operator, route := route.route, link? }
          | none =>
            let name :=
              theoremName (← getCurrNamespace) operator (theoremSuffix route.route backend mixed)
            -- The contracts are bound apart, so that the analysis keeps its own heartbeat budget.
            let (computation, added, hypotheses) ← if backend matches .configured then
                liftTermElabM <| withContracts route.computation mixed fun computation added
                  hypotheses => pure (computation, added, hypotheses)
              else pure (route.computation, #[], #[])
            record (← liftTermElabM <| withRef ref <|
              coverRoute entry computation mixed name ref added hypotheses)

/-- The public definitions of the imported modules `root` and under it whose type, after its
arguments, is a type variable, as the scalar of a generic function is, or a tensor, in name
order. Instances, projections, auxiliary declarations and the declarations that an earlier
`#roundoff_coverage` has recorded, in this module or one it imports, are left out. -/
def moduleFunctions (root : Name) : CommandElabM (Array Name) := do
  let env ← getEnv
  let recorded := coverageExtension.getState env
  let mut found := #[]
  for (name, info) in env.constants.map₁.toList do
    unless info matches .defnInfo _ do continue
    let some index := env.getModuleIdxFor? name | continue
    unless root.isPrefixOf env.header.moduleNames[index.toNat]! do continue
    if isPrivateName name || name.isInternalDetail || isAuxRecursor env name ||
        isNoConfusion env name || isInstanceCore env name ||
        (env.getProjectionFnInfo? name).isSome || recorded.any (·.operator == name) then
      continue
    let value ← liftTermElabM <| forallTelescopeReducing info.type fun _ result => do
      let result ← whnfR result
      pure (result.isFVar || result.getAppFn.isConstOf ``TorchLean.Tensor)
    if value then found := found.push name
  return found.qsort (·.toString < ·.toString)

/-- An entry of `#roundoff_coverage`: a declaration, a registered operator with some attributes
fixed, as `Op.crossEntropy (mean := true)`, or `every M`, the declarations `moduleFunctions M`
finds. -/
syntax coverageEntry := (&"every " ident) <|> (ident (" (" ident " := " term ")")*)

/-- `#roundoff_coverage family := operator, ...` analyzes every route of each registered
operator with `roundoff` and records the outcome, by family. An operator listed with fixed
attributes has those routes again with the attributes fixed, and `every M` stands for each
scalar or tensor function of the modules `M` and those under it (`moduleFunctions`). -/
syntax (name := coverage) "#roundoff_coverage" (ppLine ident " := " coverageEntry,+)+ :
  command

@[command_elab coverage, inherit_doc coverage]
def elabCoverage : CommandElab
  | `(#roundoff_coverage $[$families := $entries,*]*) => do
    for family in families, group in entries do
      for entry in group.getElems do
        match entry with
        | `(coverageEntry| every $root:ident) =>
          let names ← moduleFunctions root.getId
          if names.isEmpty then
            throwErrorAt root "no imported module under {root.getId} has such a function"
          for name in names do
            coverOperator family.getId name entry
        | `(coverageEntry| $operator:ident $[ ($names := $values)]*) =>
          let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo operator
          coverOperator family.getId name entry (names.map (·.getId) |>.zip values)
        | _ => throwUnsupportedSyntax
  | _ => throwUnsupportedSyntax

/-- Declare in the current namespace the propagation theorem (`propagationTheoremOf`) of every
configured route of `callee` in this mode, after the propagation theorems its proof applies, unless
an earlier one has its statement (`recordPropagation`), which is declared instead if it is not yet,
and then nothing the analysis added is kept. A route whose theorem an analysis earlier in this
module generated, as an arm of a composite, is not analyzed again: that theorem is declared if it is
not yet. Each analysis has a heartbeat budget of its own, and one that fails is an error. -/
def declarePropagations (callee : Name) (mixed : Bool) (ref : Syntax) : TermElabM Unit := do
  let routes ← withCurrHeartbeats do
    let routes ← operatorRoutes .configured callee
    if routes.any (!·.isEmpty) then pure routes else componentRoutes .configured callee
  let some routes := routes | throwError "roundoff: {callee} has no configured routes"
  for route in routes do
    let name := theoremName (← getCurrNamespace) callee (propagationSuffix route.route mixed)
    -- The analysis of a composite earlier in this module may have generated it as an arm.
    if let some earlier ← propagationTheorem? name then
      declareArms ((← armClosure #[earlier.proof] earlier.auxiliary).push name)
      continue
    let base ← getEnv
    let generated ← withCurrHeartbeats <|
      propagationTheoremOf base route mixed (localizedCallees.contains callee)
    let applied ← recordPropagation name generated
    if applied != name then
      -- The earlier theorem is applied instead, so nothing this analysis added is kept.
      setEnv base
      let some earlier ← propagationTheorem? applied |
        throwError "roundoff: no propagation theorem {applied}"
      declareArms ((← armClosure #[earlier.proof] earlier.auxiliary).push applied)
      continue
    declareArms (← armClosure #[generated.proof] generated.auxiliary)
    restoreDeclarations generated.auxiliary
    addDecl (.thmDecl { name, levelParams := [], type := generated.type, value := generated.proof })
    modifyEnv (propagationExtension.addEntry · (generated.declaration name))
    addDocStringCore name "Error bound of a route of an operation under the hypothesis that each \
      input lies within real budgets of real references: a propagation theorem declared by \
      `#roundoff_propagation`."
    addDeclarationRangesFromSyntax name ref

/-- `#roundoff_propagation mode := operation, ...` declares, in the current namespace, the
propagation theorems of every configured route of each operation in `mode`, `absolute` or
`mixed`, as the second analysis of a composite declares those of its callees
(`propagationArms`), each unless an earlier one has its statement. -/
syntax (name := propagation) "#roundoff_propagation" (ppLine ident " := " ident,+)+ : command

@[command_elab propagation, inherit_doc propagation]
def elabPropagation : CommandElab
  | `(#roundoff_propagation $[$modes := $groups,*]*) => do
    for mode in modes, group in groups do
      let mixed ← match mode.getId with
        | `absolute => pure false
        | `mixed => pure true
        | _ => throwErrorAt mode "#roundoff_propagation: the mode is `absolute` or `mixed`"
      for operation in group.getElems do
        let callee ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo operation
        liftTermElabM <| withRef operation <| declarePropagations callee mixed operation
  | _ => throwUnsupportedSyntax

end TorchLean.Roundoff
