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
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Literal -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.OnlineSoftmax -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Power -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Products -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Softmax -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Tensor -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.SoftmaxReplay
public import TorchLean.NN.VJPOp
public import TorchLean.Tensor.Op.Rule
public import TorchLean.Tensor.SomeTensor
public import TorchLean.NN.Tape
public import FloatLib.Floats.Formats.IEEE754.Native
public import TorchLean.Verify.Floats.FP32
public import TorchLean.Tensor.NF
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Composite -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Movement -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Ordering -- shake: keep
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Reduction -- shake: keep
public meta import Lean.Elab.DeclarationRange
public meta import TorchLean.Tensor.Tactic.Init
public import TorchLean.Tensor.Tactic.Init

/-!
# Roundoff coverage of registered operators

`#roundoff_coverage family := operator, ...` runs `roundoff` on every route of the declarations
it is given and records the outcome of each. An operator specification has two routes, its
forward map and its pullback. A tape constructor has the forward value of the node it records
and one route for each contribution its backward closure sends to a parent. The routes are read
from its code, with the values it saves. A tape constructor that records a value of any type, such
as `TorchLean.Tape.leaf`, is read on the CPU tape, whose values are tensors over the scalar packed
with their shapes as `TorchLean.SomeTensor`. A function returning tensors, such as the sum that
accumulates gradients, has one route for each tensor it returns. A function returning a tensor or
a scalar, such as a general tensor operator, has one route, `value`, read from its result. A
tensor of indices is read through the cast of its entry into the scalar, and a tensor of Booleans,
such as a comparison, through `1` or `0`. A function returning a pair of tensors, such as the
values and indices of `TorchLean.Tensor.sort`, has the routes `values` and `indices`, read from its
two components. A parameter of such a function whose type is a nonrecursive inductive type with
several constructors, and which a later instance parameter depends on or the function matches on,
such as the operation of
`TorchLean.Tensor.reduce`, is taken one constructor at a time with the constructor's fields as
variables: the function then has one route per constructor, named after it. An operator
with such a parameter, such as the reduction of `TorchLean.Op.pool`, has its routes once for each
constructor, named after the constructor and the route. A registered operator may also be listed
with some attributes fixed, as `TorchLean.Op.crossEntropy (mean := true)`: it then has its routes
once more with those attributes the given terms, named after them and the route
(`mean true forward`). A route that applies an already registered operator specification to local
variables shares that specification's outcome.

A constructor of `TorchLean.Op` has the routes of its arm of `TorchLean.Op.rule?`: the forward value
at each output, the tangent of each value output under the forward-mode rule where the arm has one
and no other route computes it, the gradient of each value input from the dense pullback, the
prepared pullback where it is other code, and the routes of an alias rule. A frozen scalar input is
a scalar variable, a capability the scalar lacks is an instance variable, the element of a
generalized operator is the scalar (its Boolean and integer elements compute without rounding), and
an attribute of a nonrecursive inductive type other than `Bool` and `Option` is taken one
constructor at a time.

Each route is analyzed universally over the declaration's shapes and parameters, at one
coordinate of the tensor or at the scalar it computes, for a configured scalar over an arbitrary
format with the hypothesis that the format is IEEE, once for an absolute bound and once for a
mixed bound. A successful analysis is declared as a theorem in the current namespace, under the
declaration's name less the root namespace the two share (`theoremName`). Its premises are
classified as finite inputs, finite intermediate values, range conditions on inputs, conditions on
computed values and others. A theorem whose budget or reference decodes a computed value, or whose
proof uses an enclosure in place of a rounding theorem, is recorded with that caveat: it holds, but
it is not an a priori bound. Whether the proof uses the IEEE hypothesis decides whether the outcome
carries over to other formats.

A configured route whose analysis exceeds the heartbeat budget of a declaration is analyzed once
more when the operations its definitions call have propagation theorems. A route whose first
analysis succeeds is not analyzed again and keeps its theorem. A propagation theorem is the analysis
of a route of such an operation, found within two levels of tensor-returning definitions, under the
hypothesis that each input tensor or scalar lies within real budgets of real references, so that its
reference reads those references instead of the input. The statement of
`TorchLean.Normalization.normalize` reads each input tensor at one coordinate, the output coordinate
or its broadcast position, and its propagation theorem bounds each input at that coordinate alone,
with a real reference and real budgets (`localizedCallees`). The second analysis, with a budget of
its own, applies these theorems as rules and supplies their hypotheses from the analysis of the
arguments. A proof that takes one of them from its own conditions, as `roundoff` binds the premises
it does not discharge, is rejected.
The propagation theorems and auxiliary lemmas a proof uses, directly or through those it keeps, are
declared with it. Its record names the propagation theorems, inherits their primitive theorems and
says whether its reference decodes a computed value.
`#roundoff_propagation mode := operation, ...` declares the propagation theorems of every
configured route of the listed operations in one mode, absolute or mixed, outside any composite.

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
  /-- Whether the proof uses the hypothesis that the format is IEEE. -/
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

/-! ## Scalars -/

/-- The scalars a route is analyzed for. -/
inductive CoverageBackend where
  /-- `ExecFloat (Configured.Family format code plan)` over an arbitrary format. -/
  | configured
  /-- Lean's native `Float32`, decoded through `ExecFloat.Binary.ofFloat32` into binary32. -/
  | native
  /-- The rounded reals `FP32`: binary32 precision, no upper exponent bound. -/
  | rounded

namespace CoverageBackend

/-- The backend's name in the record. -/
def label : CoverageBackend → String
  | .configured => "configured"
  | .native => "native Float32"
  | .rounded => "rounded FP32"

/-- The modes a route is analyzed in. One analysis of the other scalars states what stops it:
the analysis has no rule for their operations in either mode. -/
def modes : CoverageBackend → Array Bool
  | .configured => #[false, true]
  | .native | .rounded => #[false]

/-- Run `k` with the local variables the scalar depends on, the hypotheses on them and the
scalar: a tensor storage and, for the configured scalar, its format, storage plan, representation
and codec, and the hypothesis that the format is IEEE. -/
def withScalar {α : Type} (backend : CoverageBackend)
    (k : Array Expr → Array Expr → Expr → TermElabM α) : TermElabM α := do
  let withStorage (scalar : Expr) (body : Expr → TermElabM α) := do
    withLocalDecl (← mkFreshUserName `inst) .instImplicit (← mkAppM ``Storage #[scalar])
      body
  match backend with
  | .native => withStorage (mkConst ``Float32) fun storage => k #[storage] #[] (mkConst ``Float32)
  | .rounded =>
    let scalar : Expr := .const ``FP32 []
    withStorage scalar fun storage => k #[storage] #[] scalar
  | .configured =>
    withLocalDecl `format .implicit (mkConst ``FloatFormat) fun format => do
    withLocalDecl `plan .implicit (← mkAppM ``Configured.StoragePlan #[format]) fun plan => do
    withLocalDecl `code .implicit (mkSort Level.one) fun code => do
    let codec ← mkAppM ``ExecFloat.ModelCodec #[plan, ← mkAppM ``Model #[format], code]
    withLocalDecl (← mkFreshUserName `inst) .instImplicit codec fun codecInstance => do
    let family ← exprToSyntax (← mkAppM ``Configured.Family #[format, code, plan])
    let scalar ← instantiateMVars (← elabType (← `(FloatLib.Floats.ExecFloat $family)))
    withStorage scalar fun storage => do
    let ieee ← mkEq (← mkAppM ``FloatFormat.isIEEE #[format]) (mkConst ``Bool.true)
    withLocalDeclD `hformat ieee fun hformat =>
      k #[format, plan, code, codecInstance, storage] #[hformat] scalar

/-- The real value of a scalar of the backend. -/
def decode (backend : CoverageBackend) (value : Expr) : TermElabM Expr := do
  match backend with
  | .configured => mkAppM ``ExecFloat.Binary.toReal #[value]
  | .rounded => mkAppM ``FloatLib.Floats.Formats.Flocq.NF.val #[value]
  | .native =>
    let value ← exprToSyntax value
    elabTerm (← `(FloatLib.Floats.ExecFloat.Binary.toReal
      (FloatLib.Floats.ExecFloat.Binary.ofFloat32 $value : FloatLib.Floats.Binary32))) none

end CoverageBackend

/-- The shape of `type` when it is a tensor of indices, `Tensor (Fin n) s`. -/
def indexTensorShape? (type : Expr) : MetaM (Option Expr) := do
  let type ← whnfR type
  unless type.isAppOfArity ``Tensor.Rep 3 do return none
  return if (← whnfR type.getAppArgs[0]!).isAppOfArity ``Fin 1 then some type.getAppArgs[1]!
    else none

/-- The shape of `type` when it is a tensor of Booleans, `Tensor Bool s`. -/
def boolTensorShape? (type : Expr) : MetaM (Option Expr) := do
  let type ← whnfR type
  unless type.isAppOfArity ``Tensor.Rep 3 do return none
  return if (← whnfR type.getAppArgs[0]!).isConstOf ``Bool then some type.getAppArgs[1]! else none

/-- Run `k` on the constructor `name` of the inductive type `type`, applied to the parameters of
`type` and to new variables for its fields, and on those variables. -/
def withConstructor {α : Type} (type : Expr) (name : Name)
    (k : Expr → Array Expr → TermElabM α) : TermElabM α := do
  let type ← whnfR type
  let some (.ctorInfo info) := (← getEnv).find? name |
    throwError "#roundoff_coverage: {name} is not a constructor"
  let head := mkAppN (.const name type.getAppFn.constLevels!)
    (type.getAppArgs.extract 0 info.numParams)
  forallTelescope (← inferType head) fun fields _ => k (mkAppN head fields) fields

/-- Apply `operator` to `scalar` at its first type argument, to every instance that synthesis
finds and, at the binder positions of `cases`, to the given constructors. With `packed`, the first
type argument is `SomeTensor scalar` instead, and an argument of that type is the packing
`SomeTensor.ofTensor` of a tensor variable of an implicit shape `s`. Run `k` on the result
with the remaining arguments, the constructors' fields and these shapes and tensors as local
variables, or on `none` when `operator` takes no scalar type and does not return a tensor of
indices. -/
partial def withScalarInstance {α : Type} (operator scalar : Expr) (cases : Array (Nat × Name))
    (k : Option (Expr × Array Expr) → TermElabM α) (packed : Bool := false) : TermElabM α :=
  visit operator #[] false 0
where
  visit (value : Expr) (parameters : Array Expr) (applied : Bool) (position : Nat) :
      TermElabM α := do
    let type ← whnfR (← inferType value)
    let .forallE name domain _ info := type |
      k (if applied || (← indexTensorShape? type).isSome then some (value, parameters) else none)
    let domain := domain.cleanupAnnotations
    if !applied && domain.isType then
      let argument ← if packed then mkAppOptM ``SomeTensor #[scalar, none] else pure scalar
      return ← visit (mkApp value argument) parameters true (position + 1)
    if info.isInstImplicit then
      if let some argument ← synthInstance? domain then
        return ← visit (mkApp value argument) parameters applied (position + 1)
    if let some (_, constructor) := cases.find? (·.1 == position) then
      return ← withConstructor domain constructor fun argument fields =>
        visit (mkApp value argument) (parameters ++ fields) applied (position + 1)
    if packed && domain.isAppOfArity ``SomeTensor 2 then
      return ← withLocalDecl `s .implicit (mkConst ``Shape) fun shape => do
        let tensorType ← mkAppOptM ``Tensor #[scalar, shape, none]
        withLocalDecl name info tensorType fun tensor => do
          visit (mkApp value (← mkAppM ``SomeTensor.ofTensor #[tensor]))
            (parameters ++ #[shape, tensor]) applied (position + 1)
    withLocalDecl name info domain fun parameter =>
      visit (mkApp value parameter) (parameters.push parameter) applied (position + 1)

/-! ## Routes -/

/-- A route of a registered declaration. -/
structure Route where
  /-- The route's name, as in `CoverageEntry.route`. -/
  route : String
  /-- The real observation of one coordinate of the tensor the route computes, abstracted over
  the variables the tensor depends on and the coordinate. `operatorRoutes` also abstracts the
  scalar's variables before them and its hypotheses after them. -/
  computation : Expr
  /-- The operator specification and route that the tensor applies to local variables. -/
  link? : Option (Name × String) := none

/-- The variables among `locals` that `term` depends on, directly or through the types of
other such variables, and the instance parameters about these variables, in their order. An
instance such as nonempty-axis evidence stays available to the analysis even when `term` does not
mention it. -/
private def dependencies (locals : Array Expr) (term : Expr) : MetaM (Array Expr) := do
  let mut needed := collectFVars {} (← instantiateMVars term)
  for x in locals.reverse do
    if needed.fvarSet.contains x.fvarId! then
      needed := collectFVars needed (← instantiateMVars (← inferType x))
  let about (x : Expr) : MetaM Bool := do
    unless (← x.fvarId!.getDecl).binderInfo.isInstImplicit do return false
    let mentioned := (collectFVars {} (← instantiateMVars (← inferType x))).fvarIds.filter
      fun id => locals.any (·.fvarId! == id)
    return !mentioned.isEmpty && mentioned.all needed.fvarSet.contains
  locals.filterM fun x => return needed.fvarSet.contains x.fvarId! || (← about x)

/-- The operator specification and route that `tensor` applies to distinct local variables. -/
def linkedRoute? (tensor : Expr) : Option (Name × String) := do
  let arguments := tensor.getAppArgs
  let (route, tensors) ← if tensor.isAppOfArity ``VJPOp.forward 6 then
      some ("forward", arguments.extract 5)
    else if tensor.isAppOfArity ``VJPOp.backward 7 then
      some ("backward", arguments.extract 5)
    else none
  guard (tensors.all (·.isFVar) && tensors.toList.eraseDups.length == tensors.size)
  let operator ← arguments[4]!.getAppFn.constName?
  return (operator, route)

/-- `term` with the beta redexes at its head reduced, also through metadata on the function. -/
private partial def betaHead (term : Expr) : Expr :=
  let function := term.getAppFn.consumeMData
  if term.isApp && function.isLambda then betaHead (function.betaRev term.getAppRevArgs)
  else term

/-- `term` with `let`s, beta redexes and projections of pairs at its head reduced. -/
private partial def reducePairs (term : Expr) : Expr :=
  match betaHead term with
  | .letE _ _ value body _ => reducePairs (body.instantiate1 value)
  | .proj ``Prod index pair => component index pair (.proj ``Prod index pair)
  | term =>
    if term.isAppOfArity ``Prod.fst 3 then component 0 term.appArg! term
    else if term.isAppOfArity ``Prod.snd 3 then component 1 term.appArg! term
    else term
where
  /-- The component `index` of `pair` when it reduces to a pair, otherwise `term`. -/
  component (index : Nat) (pair term : Expr) : Expr :=
    let pair := reducePairs pair
    if pair.isAppOfArity ``Prod.mk 4 then reducePairs pair.getAppArgs[2 + index]! else term

/-- The route `route` that observes `tensor` at a new coordinate, abstracted over that coordinate
and the variables among `locals` that `tensor` depends on. -/
def observeTensor (backend : CoverageBackend) (locals : Array Expr) (route : String)
    (tensor : Expr) : TermElabM Route := do
  let tensor := reducePairs (← Core.betaReduce (← instantiateMVars tensor))
  let type ← whnfR (← inferType tensor)
  unless type.isAppOfArity ``Tensor.Rep 3 do
    throwError "#roundoff_coverage: the recorded value {tensor} is not a tensor"
  let shape := type.getAppArgs[1]!
  withLocalDeclD `index (← mkAppM ``Shape.Coord #[shape]) fun index => do
    let value ← backend.decode (← mkAppM ``Tensor.Rep.get #[tensor, index])
    let computation ← mkLambdaFVars ((← dependencies locals tensor).push index) value
    return { route, computation, link? := linkedRoute? tensor }

/-- The forward map and the pullback of an operator specification. -/
def forwardBackwardRoutes (backend : CoverageBackend) (scalar operator : Expr)
    (locals : Array Expr) :
    TermElabM (Array Route) := do
  let #[_, _, input, output] := (← whnfR (← inferType operator)).getAppArgs |
    throwError "#roundoff_coverage: {operator} is not an operator specification"
  let tensor (shape : Expr) := mkAppOptM ``Tensor #[scalar, shape, none]
  let forward ← withLocalDeclD `input (← tensor input) fun x => do
    observeTensor backend (locals.push x) "forward"
      (← mkAppM ``VJPOp.forward #[operator, x])
  let backward ← withLocalDeclD `input (← tensor input) fun x => do
    withLocalDeclD `upstream (← tensor output) fun seed => do
      observeTensor backend (locals ++ #[x, seed]) "backward"
        (← mkAppM ``VJPOp.backward #[operator, x, seed])
  return #[forward, backward]

/-- Whether `type`, after its binders, is the result of recording a tape node: the extended tape
and the node's id, possibly as a `Result`. -/
def recordsNode (type : Expr) : MetaM Bool :=
  forallTelescope type fun _ result => do
    let result ← whnfR result
    let result := if result.isAppOfArity ``Except 2 then result.appArg! else result
    return result.isAppOfArity ``Prod 2 && result.appFn!.appArg!.isAppOf ``Tape

/-- Whether `type`, after its binders, is a tape node, such as the result of `Tape.Node.leaf`. -/
def buildsNode (type : Expr) : MetaM Bool :=
  forallTelescope type fun _ result => return (← whnfR result).isAppOfArity ``Tape.Node 1

/-- Whether `operator` records a node on a tape whose values have the type of its first type
parameter, a value of any type, as `Tape.leaf` does. -/
def recordsAnyValue (operator : Name) : MetaM Bool := do
  let type := (← getConstInfo operator).type
  unless ← recordsNode type do return false
  forallTelescope type fun parameters result => do
    let some values ← parameters.findM? fun x => return (← inferType x).isType | return false
    let result ← whnfR result
    let result := if result.isAppOfArity ``Except 2 then result.appArg! else result
    let tape := result.appFn!.appArg!
    return tape.isAppOfArity ``Tape 1 && tape.appArg! == values

/-- `term` with the definition at its head unfolded and applied to its arguments. -/
def unfoldHead? (term : Expr) : MetaM (Option Expr) := do
  let .const name levels := term.getAppFn | return none
  let some (.defnInfo info) := (← getEnv).find? name | return none
  return some ((← instantiateValueLevelParams (.defnInfo info) levels).beta term.getAppArgs)

/-- The parent a contribution is sent to, named after the variable holding its node id. -/
def parentName (id : Expr) : MetaM String := do
  let name := (← id.fvarId!.getUserName).eraseMacroScopes.toString
  return if name.endsWith "Id" && name.length > 2 then (name.dropEnd 2).toString else name

/-- The tensors that `term` records, with the variables bound on the way to them: the forward
value and the contributions to parents of each tape node it builds and, with `values`, every
tensor it returns. Tape constructors, the definitions of `Tape` that build a node and pattern
matches that `term` uses are unfolded, and a tuple matched by `let (a, b) := p` is read through its
projections. -/
partial def recordedTensors (backend : CoverageBackend) (values : Bool) (term : Expr)
    (locals : Array Expr) : TermElabM (Array Route) := do
  match betaHead term with
  | .lam name type body info =>
    withLocalDecl name info type fun x =>
      recordedTensors backend values (body.instantiate1 x) (locals.push x)
  | .letE _ _ value body _ => recordedTensors backend values (body.instantiate1 value) locals
  | .mdata _ inner => recordedTensors backend values inner locals
  | term@(.app ..) =>
    let arguments := term.getAppArgs
    if term.isAppOfArity ``SomeTensor.ofTensor 4 && values then
      return #[← observeTensor backend locals "value" arguments.back!]
    if term.isAppOfArity ``Prod.mk 4 && arguments[2]!.isFVar &&
        arguments[3]!.isAppOfArity ``SomeTensor.ofTensor 4 then
      let route := s!"{← parentName arguments[2]!} gradient"
      return #[← observeTensor backend locals route arguments[3]!.appArg!]
    if term.isAppOf ``Tape.Node.mk then
      let fields := getStructureFields (← getEnv) ``Tape.Node
      let field (name : Name) := arguments[1 + (fields.idxOf? name).get!]!
      let value := field `value
      let forward ← if value.isAppOfArity ``SomeTensor.ofTensor 4 then
          pure #[← observeTensor backend locals "forward" value.appArg!]
        else pure #[]
      return forward ++ (← recordedTensors backend values (field `backward) locals)
    if term.isAppOf ``Prod.casesOn && arguments.size ≥ 5 then
      let pair := arguments[3]!
      let next := mkAppN (mkApp2 arguments[4]! (← mkAppM ``Prod.fst #[pair])
        (← mkAppM ``Prod.snd #[pair])) (arguments.extract 5)
      return ← recordedTensors backend values next locals
    if let some name := term.getAppFn.constName? then
      let type ← inferType term.getAppFn
      if (← isMatcher name) || (``Tape).isPrefixOf name &&
          ((← recordsNode type) || (← buildsNode type)) then
        if let some unfolded ← unfoldHead? term then
          return ← recordedTensors backend values unfolded locals
    arguments.foldlM (init := #[]) fun found argument => do
      return found ++ (← recordedTensors backend values argument locals)
  | _ => return #[]

/-- The variables that a `match` in `term` takes as discriminants. -/
private partial def matchedVariables (environment : Environment) : Expr → Array FVarId
  | term@(.app ..) =>
    let arguments := term.getAppArgs
    let own := match Meta.isMatcherAppCore? environment term with
      | some info =>
        (arguments.extract (info.numParams + 1) (info.numParams + 1 + info.numDiscrs)).filterMap
          fun discriminant => if discriminant.isFVar then some discriminant.fvarId! else none
      | none => #[]
    arguments.foldl (fun found argument => found ++ matchedVariables environment argument)
      (own ++ matchedVariables environment term.getAppFn)
  | .lam _ type body _ | .forallE _ type body _ =>
    matchedVariables environment type ++ matchedVariables environment body
  | .letE _ type value body _ =>
    matchedVariables environment type ++ matchedVariables environment value ++
      matchedVariables environment body
  | .mdata _ inner | .proj _ _ inner => matchedVariables environment inner
  | _ => #[]

/-- The explicit parameters of `operator` that its routes take one constructor at a time, with
their binder positions and constructors: those of a nonrecursive inductive type with several
constructors that a later instance parameter depends on or that the definition matches on. -/
def casedParameters (operator : Name) : MetaM (Array (Nat × List Name)) := do
  let environment ← getEnv
  let some (.defnInfo info) := environment.find? operator | return #[]
  lambdaTelescope info.value fun parameters body => do
    let matched := matchedVariables environment body
    let mut found := #[]
    for h : index in [:parameters.size] do
      let parameter := parameters[index]
      let declaration ← parameter.fvarId!.getDecl
      unless declaration.binderInfo.isExplicit do continue
      if ← isProp declaration.type then continue
      let .const typeName _ := (← whnfR declaration.type).getAppFn | continue
      let some (.inductInfo type) := environment.find? typeName | continue
      if type.isRec || type.numIndices != 0 || type.ctors.length < 2 then continue
      let constrains ← (parameters.extract (index + 1)).anyM fun later => do
        return (← later.fvarId!.getDecl).binderInfo.isInstImplicit &&
          (← instantiateMVars (← inferType later)).containsFVar parameter.fvarId!
      if constrains || matched.contains parameter.fvarId! then
        found := found.push (index, type.ctors)
    return found

/-- Whether `operator` returns a tensor or a value of one of its type parameters. -/
def returnsValue (operator : Name) : MetaM Bool := do
  forallTelescope (← getConstInfo operator).type fun parameters result => do
    let result ← whnfR result
    return result.isAppOfArity ``Tensor.Rep 3 || parameters.contains result

/-- The route `route` that observes `value`: a tensor at a new coordinate, a tensor of indices at
a new coordinate through the cast of its entry into `scalar`, a tensor of Booleans through `1` or
`0` in `scalar`, or a scalar. -/
def observeValue (backend : CoverageBackend) (scalar : Expr) (locals : Array Expr)
    (route : String) (value : Expr) : TermElabM Route := do
  if let some shape ← indexTensorShape? (← inferType value) then
    return ← withLocalDeclD `index (← mkAppM ``Shape.Coord #[shape]) fun index => do
      let entry ← mkAppM ``Fin.val #[← mkAppM ``Tensor.Rep.get #[value, index]]
      let cast ← mkAppOptM ``Nat.cast #[scalar, none, entry]
      let computation ← mkLambdaFVars ((← dependencies locals value).push index)
        (← backend.decode cast)
      return { route, computation }
  if let some shape ← boolTensorShape? (← inferType value) then
    return ← withLocalDeclD `index (← mkAppM ``Shape.Coord #[shape]) fun index => do
      let entry ← mkAppM ``Tensor.Rep.get #[value, index]
      let one ← mkAppOptM ``OfNat.ofNat #[scalar, mkRawNatLit 1, none]
      let zero ← mkAppOptM ``OfNat.ofNat #[scalar, mkRawNatLit 0, none]
      let indicator ← mkAppM ``ite #[← mkEq entry (mkConst ``Bool.true), one, zero]
      let computation ← mkLambdaFVars ((← dependencies locals value).push index)
        (← backend.decode indicator)
      return { route, computation }
  if ← withReducible (isDefEq (← inferType value) scalar) then
    let computation ← mkLambdaFVars (← dependencies locals value) (← backend.decode value)
    return { route, computation }
  observeTensor backend locals route value

/-- The choices of constructors for the cased parameters of `operator`, each with its binder
position, and the name of each choice: the constructors' names, or `""` for the one choice of an
operator without cased parameters. -/
def constructorChoices (operator : Name) : MetaM (List (List (Nat × Name) × String)) := do
  let choices := (← casedParameters operator).foldl (init := [[]])
    fun choices (position, constructors) =>
      choices.flatMap fun choice => constructors.map fun constructor =>
        choice ++ [(position, constructor)]
  return choices.map fun choice =>
    (choice, " ".intercalate (choice.map fun (_, constructor) => constructor.getString!))

/-- The routes of a function returning a tensor or a scalar: one, `value`, or one for each choice
of constructors of its cased parameters, named after them. -/
def valueRoutes (backend : CoverageBackend) (operator : Name) :
    TermElabM (Option (Array Route)) := do
  let mut routes := #[]
  for (choice, label) in ← constructorChoices operator do
    let name := if choice.isEmpty then "value" else label
    let constant ← mkConstWithFreshMVarLevels operator
    let some route ← backend.withScalar fun context hypotheses scalar =>
        withScalarInstance constant scalar choice.toArray fun
          | none => return none
          | some (specialized, parameters) => do
            let route ← observeValue backend scalar parameters name specialized
            let computation ← lambdaTelescope route.computation fun arguments body => do
              instantiateMVars (← mkLambdaFVars (context ++ arguments ++ hypotheses) body)
            return some { route with computation }
      | return none
    routes := routes.push route
  return some routes

/-- Whether `operator` returns a pair of tensors, such as the values and indices of a sort. -/
def returnsTensorPair (operator : Name) : MetaM Bool := do
  forallTelescope (← getConstInfo operator).type fun _ result => do
    let result ← whnfR result
    unless result.isAppOfArity ``Prod 2 do return false
    let isTensor (type : Expr) : MetaM Bool := do
      return (← whnfR type).isAppOfArity ``Tensor.Rep 3
    return (← isTensor result.appFn!.appArg!) && (← isTensor result.appArg!)

/-- The routes of a function returning a pair of tensors: `values`, observing the first, and
`indices`, observing the second, each as `observeValue` reads it. -/
def pairRoutes (backend : CoverageBackend) (operator : Name) :
    TermElabM (Option (Array Route)) := do
  let mut routes := #[]
  for (name, projection) in #[("values", ``Prod.fst), ("indices", ``Prod.snd)] do
    let constant ← mkConstWithFreshMVarLevels operator
    let some route ← backend.withScalar fun context hypotheses scalar =>
        withScalarInstance constant scalar #[] fun
          | none => return none
          | some (specialized, parameters) => do
            let route ← observeValue backend scalar parameters name
              (← mkAppM projection #[specialized])
            let computation ← lambdaTelescope route.computation fun arguments body => do
              instantiateMVars (← mkLambdaFVars (context ++ arguments ++ hypotheses) body)
            return some { route with computation }
      | return none
    routes := routes.push route
  return some routes

/-- The projections, from the outside in, reaching each tensor of a nested pair type `type`, or
`none` when a component is neither a tensor nor a pair. -/
partial def tensorComponents (type : Expr) : MetaM (Option (Array (List Name))) := do
  let type ← whnfR type
  if type.isAppOfArity ``Tensor.Rep 3 then return some #[[]]
  unless type.isAppOfArity ``Prod 2 do return none
  let some first ← tensorComponents type.appFn!.appArg! | return none
  let some second ← tensorComponents type.appArg! | return none
  return some (first.map (``Prod.fst :: ·) ++ second.map (``Prod.snd :: ·))

/-- The routes of a function returning a nested pair of tensors: `component i`, observing the
`i`-th tensor from the left, as `observeValue` reads it. -/
def componentRoutes (backend : CoverageBackend) (operator : Name) :
    TermElabM (Option (Array Route)) := do
  let some components ← forallTelescope (← getConstInfo operator).type fun _ result =>
      tensorComponents result
    | return none
  let mut routes := #[]
  for h : index in [:components.size] do
    let constant ← mkConstWithFreshMVarLevels operator
    let some route ← backend.withScalar fun context hypotheses scalar =>
        withScalarInstance constant scalar #[] fun
          | none => return none
          | some (specialized, parameters) => do
            let value ← components[index].foldlM (fun value projection =>
              mkAppM projection #[value]) specialized
            let route ← observeValue backend scalar parameters s!"component {index}" value
            let computation ← lambdaTelescope route.computation fun arguments body => do
              instantiateMVars (← mkLambdaFVars (context ++ arguments ++ hypotheses) body)
            return some { route with computation }
      | return none
    routes := routes.push route
  return some routes

/-! ## Registered operators -/

/-- Whether `operator` is a constructor of the registered operators `TorchLean.Op`. The names of
the operator table are read at run time, so this module does not depend on the table. -/
def isRegisteredOperator (operator : Name) : CoreM Bool := do
  return match (← getEnv).find? operator with
    | some (.ctorInfo info) => info.induct == `TorchLean.Op
    | _ => false

/-- The elements of a list of ports, after reducing it to its constructors. -/
partial def portList (ports : Expr) : MetaM (Array Expr) := do
  let ports ← whnf ports
  if ports.isAppOfArity ``List.cons 3 then
    return #[ports.getAppArgs[1]!] ++ (← portList ports.getAppArgs[2]!)
  if ports.isAppOfArity ``List.nil 1 then return #[]
  throwError "#roundoff_coverage: the ports {ports} are not a list of known length"

/-- The element sort of a port, reduced to its constructor, and the port's shape. -/
def portParts (port : Expr) : MetaM (Expr × Expr) := do
  let port ← whnf port
  unless port.isAppOfArity ``Port.mk 2 do
    throwError "#roundoff_coverage: the port {port} is not a constructor application"
  return (← whnf port.getAppArgs[0]!, port.getAppArgs[1]!)

/-- The tensor type a port of sort `dtype` and `shape` carries over `scalar`: the scalar for a
value, frozen or detached port, `Fin n` for an index port with one bound `n`, the index tuple of its
bounds for an index port with several, `Bool` for a Boolean port, and `BitVec width` for an
integer port. -/
def portTensor (scalar dtype shape : Expr) : MetaM Expr := do
  let element ← if dtype.isConstOf ``Port.Kind.value ||
      dtype.isConstOf ``Port.Kind.frozen || dtype.isConstOf ``Port.Kind.detached then
      pure scalar
    else if dtype.isAppOfArity ``Port.Kind.index 1 then do
      let tuple ← mkAppM ``Port.IndexTuple #[dtype.appArg!]
      let reduced ← whnf tuple
      pure (if reduced.isAppOfArity ``Fin 1 then reduced else tuple)
    else if dtype.isConstOf ``Port.Kind.bool then pure (mkConst ``Bool)
    else if dtype.isAppOfArity ``Port.Kind.integer 2 then
      pure (mkApp (mkConst ``BitVec) dtype.appArg!)
    else throwError "#roundoff_coverage: the element sort {dtype} is not a constructor"
  mkAppOptM ``Tensor #[element, shape, none]

/-- `term` with every local definition in it, `let` and `have`, replaced by its value, as the
routes of tape constructors read their saved values. -/
def instantiateLocalDefinitions (term : Expr) : MetaM Expr :=
  Meta.transform term (pre := fun e => do
    match e with
    | .letE _ _ value body _ => return .visit (body.instantiate1 value)
    | _ =>
      if e.isAppOfArity ``letFun 4 then return .visit (e.appArg!.beta #[e.appFn!.appArg!])
      return .continue)

/-- The classes of the comparisons an arm reads through the scalar's `Context`. -/
def comparisonClasses : Array Name := #[``BEq, ``LT, ``LE, ``Ord, ``Decidable]

/-- `term` with each comparison instance it applies replaced by the instance that synthesis finds
for its type, where the two are definitionally equal: the comparisons an arm reads through the
scalar's `Context` become the scalar's own, as in the routes of operators that take their
instances directly. -/
def synthesizeInstances (term : Expr) : MetaM Expr :=
  Meta.transform term (post := fun e => do
    if e.isFVar || e.hasLooseBVars || e.hasMVar then return .continue
    let type ← inferType e
    let some name ← isClass? type | return .continue
    unless comparisonClasses.contains name do return .continue
    let some synthesized ← synthInstance? type | return .continue
    if synthesized == e then return .continue
    unless ← withNewMCtxDepth (isDefEq synthesized e) do return .continue
    return .done synthesized)

/-- `term` with every projection of a pair it applies to a pair reduced, and the entry of a scalar
tensor built from a value read as that value. -/
def reduceTuples (term : Expr) : MetaM Expr :=
  Meta.transform term (post := fun e => do
    if e.isAppOfArity ``Tensor.item 3 &&
        e.appArg!.isAppOfArity ``Tensor.scalar 3 then
      return .visit e.appArg!.appArg!
    let pick (index : Nat) (pair : Expr) : Option Expr :=
      if pair.isAppOfArity ``Prod.mk 4 then some pair.getAppArgs[2 + index]! else none
    let arguments := e.getAppArgs
    let reduced :=
      if e.isAppOf ``Prod.fst && arguments.size ≥ 3 then
        (pick 0 arguments[2]!).map (mkAppN · (arguments.extract 3))
      else if e.isAppOf ``Prod.snd && arguments.size ≥ 3 then
        (pick 1 arguments[2]!).map (mkAppN · (arguments.extract 3))
      else match e.getAppFn with
        | .proj ``Prod index pair => (pick index pair).map (mkAppN · arguments)
        | _ => none
    return match reduced with
      | some term => .visit term.headBeta
      | none => .done e)

/-- Run `k` on new variables for the tensors of `ports` that `keep` selects, named `name`
with the port's position, and on the nested pair of their values, with `missing` at the others. -/
partial def withPortValues {α : Type} (scalar : Expr) (ports : Array Expr) (name : String)
    (keep : Expr → Bool) (missing : Expr) (k : Array Expr → Expr → TermElabM α) :
    TermElabM α :=
  go 0 #[] #[]
where
  go (position : Nat) (bound : Array Expr) (values : Array Expr) : TermElabM α := do
    if position < ports.size then
      let (dtype, shape) ← portParts ports[position]!
      if keep dtype then
        -- A frozen scalar input is a scalar variable, as a declaration captures a scalar.
        if dtype.isConstOf ``Port.Kind.frozen && (← whnf shape).isAppOfArity ``List.nil 1 then
          withLocalDeclD (.mkSimple s!"{name}{position}") scalar fun c => do
            go (position + 1) (bound.push c) (values.push (← mkAppM ``Tensor.scalar #[c]))
        else
          withLocalDeclD (.mkSimple s!"{name}{position}") (← portTensor scalar dtype shape)
            fun x => go (position + 1) (bound.push x) (values.push x)
      else go (position + 1) bound (values.push missing)
    else
      let tuple ← values.foldrM (init := mkConst ``Unit.unit) fun value rest =>
        mkAppM ``Prod.mk #[value, rest]
      k bound tuple

/-- The component at `position` of a nested pair. -/
def tupleComponent (tuple : Expr) (position : Nat) : MetaM Expr := do
  let mut rest := tuple
  for _ in [:position] do rest ← mkAppM ``Prod.snd #[rest]
  mkAppM ``Prod.fst #[rest]

/-- The flags of `count` inputs that all need a gradient, as a nested pair of literals. -/
def allFlags (count : Nat) : MetaM Expr := do
  let mut flags := Lean.mkConst ``Unit.unit
  for _ in [:count] do flags ← mkAppM ``Prod.mk #[mkConst ``Bool.true, flags]
  return flags

/-- The flags of `count` inputs with only the input at `position` needing a gradient. -/
def onlyFlag (count position : Nat) : MetaM Expr := do
  let mut flags := Lean.mkConst ``Unit.unit
  for index in (List.range count).reverse do
    let flag := if index == position then ``Bool.true else ``Bool.false
    flags ← mkAppM ``Prod.mk #[Lean.mkConst flag, flags]
  return flags

/-- A field of the structure instance `value` of the structure `name` with `parameters`
parameters, or its projection reduced. -/
def structureField (name : Name) (parameters : Nat) (value : Expr) (field : Name) :
    MetaM Expr := do
  let fields := getStructureFields (← getEnv) name
  let some index := fields.idxOf? field | throwError "#roundoff_coverage: no field {field}"
  if value.isAppOf (name ++ `mk) then return value.getAppArgs[parameters + index]!
  whnfR (← mkAppM (name ++ field) #[value])

/-- A field of the rule `rule`, a structure instance of `TorchLean.Op.Rule`. -/
def ruleField (rule : Expr) (field : Name) : MetaM Expr :=
  structureField ``Op.Rule 4 rule field

/-- Run `k` on the capabilities `Op.rule?` reads at `scalar` and on the variables it introduced: an
instance where synthesis finds one, and otherwise a new instance variable, as the routes of a
declaration that takes the instance treat it (its functions get contracts). -/
partial def withCapabilities {β : Type} (scalar : Expr) (k : Expr → Array Expr → TermElabM β) :
    TermElabM β := do
  let constructor ← mkConstWithFreshMVarLevels `TorchLean.Op.Capabilities.mk
  let type ← instantiateForall (← inferType constructor) #[scalar]
  let classes ← forallTelescope type fun fields _ => fields.mapM fun field => do
    return (← whnfR (← inferType field)).appArg!
  go (mkApp constructor scalar) classes.toList #[]
where
  go (value : Expr) (classes : List Expr) (introduced : Array Expr) : TermElabM β := do
    match classes with
    | [] => k value introduced
    | capability :: rest =>
      if let some instance_ ← synthInstance? capability then
        go (mkApp value (← mkAppM ``Option.some #[instance_])) rest introduced
      else
        withLocalDecl `capability .instImplicit capability fun instance_ => do
          go (mkApp value (← mkAppM ``Option.some #[instance_])) rest (introduced.push instance_)

/-- `TorchLean.Op.rule?` at `scalar`, with the capabilities `capabilities`, applied to `op`. -/
def applyOpRule (scalar capabilities op : Expr) : MetaM Expr := do
  let spec ← mkConstWithFreshMVarLevels `TorchLean.Op.rule?
  forallTelescopeReducing (← inferType spec) fun binders _ => do
    let mut arguments : Array (Option Expr) := #[]
    for h : position in [:binders.size] do
      let domain ← inferType binders[position]
      if position == 0 then arguments := arguments.push (some scalar)
      else if position + 1 == binders.size then arguments := arguments.push (some op)
      else if domain.getAppFn.constName? == some `TorchLean.Op.Capabilities then
        arguments := arguments.push (some capabilities)
      else arguments := arguments.push none
    mkAppOptM `TorchLean.Op.rule? arguments

/-- The scalar element when `domain` is the element of a generalized operator (`Op.Element` or
`Op.NumericElement`): its routes are the scalar operator's, as the Boolean and integer elements
compute without rounding. -/
def scalarElement? (domain : Expr) : MetaM (Option Expr) := do
  match (← whnfR domain).getAppFn.constName? with
  | some `TorchLean.Op.Element => return some (mkConst `TorchLean.Op.Element.scalar)
  | some `TorchLean.Op.NumericElement => return some (mkConst `TorchLean.Op.NumericElement.scalar)
  | _ => return none

/-- The constructors an attribute of a registered operator is taken through one at a time: those of
a nonrecursive inductive type other than `Bool`, `Option` and an element (`scalarElement?`) with
several constructors, such as the reduction of a pooling. -/
def splitConstructors? (domain : Expr) : MetaM (Option (List Name)) := do
  if ← isProp domain then return none
  if (← scalarElement? domain).isSome then return none
  let .const typeName _ := (← whnfR domain).getAppFn | return none
  if typeName == ``Bool || typeName == ``Option then return none
  let some (.inductInfo type) := (← getEnv).find? typeName | return none
  if type.isRec || type.numIndices != 0 || type.ctors.length < 2 then return none
  return some type.ctors

/-- The binder positions of the explicit attributes of a type `List τ` that occur in a list of ports
of an operator of type `type`, with attributes `attributes`, that `portList` cannot read, such as
the input sizes of a concatenation, whose ports are mapped from them. The test is syntactic: such
an attribute is taken at two entries without a proof that it fixes the number of ports. -/
def portListAttributes (attributes : Array Expr) (type : Expr) : MetaM (Array Nat) := do
  let type ← whnf type
  let mut found := #[]
  for ports in type.getAppArgs do
    let known ← try
        discard <| portList ports
        pure true
      catch _ => pure false
    if known then continue
    for h : position in [:attributes.size] do
      let declaration ← attributes[position].fvarId!.getDecl
      unless declaration.binderInfo.isExplicit do continue
      unless (← whnfR declaration.type).isAppOfArity ``List 1 do continue
      if ports.containsFVar declaration.fvarId && !found.contains position then
        found := found.push position
  return found

/-- The choices for the split attributes of the registered operator `constant`, by binder position,
each with its label: a constructor for an attribute of a nonrecursive inductive type
(`splitConstructors?`), labelled by its name, and `List.cons` for a list attribute that the ports
are mapped from (`portListAttributes`), taken at two entries (`withListEntries`) and labelled `two`
and the binder's name. The attributes named in `fixed` are not split. -/
partial def attributeChoices (constant : Expr) (fixed : Array Name := #[]) :
    MetaM (List (List (Nat × Name) × String)) := do
  forallTelescope (← inferType constant) fun attributes type => do
    let lists ← portListAttributes attributes type
    let mut choices : List (List (Nat × Name) × List String) := [([], [])]
    for h : position in [:attributes.size] do
      let declaration ← attributes[position].fvarId!.getDecl
      unless declaration.binderInfo.isExplicit do continue
      if fixed.contains declaration.userName then continue
      if lists.contains position then
        let label := s!"two {declaration.userName.eraseMacroScopes}"
        choices := choices.map fun (choice, labels) =>
          (choice ++ [(position, ``List.cons)], labels ++ [label])
        continue
      let some constructors ← splitConstructors? declaration.type | continue
      choices := choices.flatMap fun (choice, labels) => constructors.map fun c =>
        (choice ++ [(position, c)], labels ++ [c.getString!])
    return choices.map fun (choice, labels) => (choice, " ".intercalate labels)

/-- Run `k` on the list of two new variables of the element type of the list type `type`, named
after `name`, and on those variables. -/
def withListEntries {α : Type} (type : Expr) (name : Name)
    (k : Expr → Array Expr → TermElabM α) : TermElabM α := do
  let element := (← whnfR type).appArg!
  withLocalDeclD (name.appendIndexAfter 0) element fun first =>
    withLocalDeclD (name.appendIndexAfter 1) element fun second => do
      k (← mkListLit element [first, second]) #[first, second]

/-- Apply the registered operator `constant` to fresh variables for its attributes and run `k` on
the operator and the variables. At the binder positions of `choice` the arguments are the given
constructors with their fields as variables, or for `List.cons` a list of two variables
(`withListEntries`). An element of a generalized operator is the scalar (`scalarElement?`), and the
attributes named in `fixed` are the given terms, elaborated at the attribute's type. -/
partial def withAttributes {β : Type} (constant : Expr) (choice : List (Nat × Name))
    (k : Expr → Array Expr → TermElabM β) (fixed : Array (Name × Term) := #[]) : TermElabM β :=
  visit constant #[] 0
where
  visit (value : Expr) (attributes : Array Expr) (position : Nat) : TermElabM β := do
    let type ← whnfR (← inferType value)
    let .forallE name domain _ info := type | k value attributes
    if let some (_, term) := fixed.find? (·.1 == name) then
      let argument ← Term.elabTermEnsuringType term domain
      Term.synthesizeSyntheticMVarsNoPostponing
      return ← visit (mkApp value (← instantiateMVars argument)) attributes (position + 1)
    if let some (_, constructor) := choice.find? (·.1 == position) then
      if constructor == ``List.cons then
        return ← withListEntries domain name fun argument entries =>
          visit (mkApp value argument) (attributes ++ entries) (position + 1)
      return ← withConstructor domain constructor fun argument fields =>
        visit (mkApp value argument) (attributes ++ fields) (position + 1)
    if let some scalar ← scalarElement? domain then
      return ← visit (mkApp value scalar) attributes (position + 1)
    withLocalDecl name info domain fun variable_ =>
      visit (mkApp value variable_) (attributes.push variable_) (position + 1)

/-- The routes of a registered operator, read from the arm of `TorchLean.Op.rule?` at the operator
applied to its attributes: its forward value at each output, `forward` for one output and
`forward k` for the output at position `k`, the tangent of each value output under the rule's
forward-mode rule where it has one, `jvp` or `jvp k`, and the gradient of each value input from the
rule's dense pullback, `backward` for one input and `backward k` otherwise. Where the prepared
pullback at every input active computes a gradient by other code than the dense one, that gradient
is the route `prepared backward k`. A tangent with the computation of another route is not a route
of its own, as that route's theorem states it. An arm with an alias rule also has the routes of its
prepared node with the two positions the same variable and every input active: `alias forward` and
`alias backward`, the one contribution it sends. A capability the scalar lacks is an instance
variable of the routes.
The attributes in `fixed` are the given terms, and every route is named after them first. -/
def registeredOperatorRoutes (backend : CoverageBackend) (operator : Name)
    (fixed : Array (Name × Term) := #[]) : TermElabM (Option (Array Route)) := do
  let constant ← mkConstWithFreshMVarLevels operator
  let binders ← forallTelescope (← inferType constant) fun attributes _ =>
    attributes.mapM fun binder => return (← binder.fvarId!.getDecl).userName
  for (name, _) in fixed do
    unless binders.contains name do
      throwError "#roundoff_coverage: {operator} has no attribute {name}"
  let fixedLabel ← fixed.mapM fun (name, term) => do
    return s!"{name} {(← PrettyPrinter.ppTerm term).pretty}"
  let fixedLabel := " ".intercalate fixedLabel.toList
  let mut all : Array Route := #[]
  for (choice, label) in ← attributeChoices constant (fixed.map (·.1)) do
    let label := " ".intercalate ([fixedLabel, label].filter (!·.isEmpty))
    let some routes ← backend.withScalar fun context hypotheses scalar =>
        withCapabilities scalar fun capabilities introduced =>
        withAttributes constant choice (fixed := fixed) fun op attributes => do
          let some routes ← armRoutes context hypotheses scalar capabilities
              (introduced ++ attributes) op | return none
          return some (routes.map fun route =>
            if label.isEmpty then route else { route with route := s!"{label} {route.route}" })
      | return none
    all := all ++ routes
  return some all
where
  /-- The routes of the arm of `op`, whose variables are `parameters`. -/
  armRoutes (context hypotheses : Array Expr) (scalar capabilities : Expr)
      (parameters : Array Expr) (op : Expr) : TermElabM (Option (Array Route)) := do
      let type ← whnf (← inferType op)
      let #[ins, outs] := type.getAppArgs |
        throwError "#roundoff_coverage: {operator} is not a registered operator"
      let spec ← whnf (← applyOpRule scalar capabilities op)
      unless spec.isAppOfArity ``Option.some 2 do
        throwError "#roundoff_coverage: {operator} has no rule in the operator table"
      let rule := spec.appArg!
      let inputs ← portList ins
      let outputs ← portList outs
      let isValue (dtype : Expr) := dtype.isConstOf ``Port.Kind.value
      let close (route : Route) : TermElabM Route := do
        let computation ← lambdaTelescope route.computation fun arguments body => do
          instantiateMVars (← mkLambdaFVars (context ++ arguments ++ hypotheses) body)
        return { route with computation }
      let normalize (term : Expr) : MetaM Expr := do
        synthesizeInstances (← reduceTuples (← Core.betaReduce
          (← instantiateLocalDefinitions (← instantiateMVars term))))
      let forward ← ruleField rule `forward
      let backward ← ruleField rule `backward
      let prepare ← ruleField rule `prepare
      let flags ← allFlags inputs.size
      withPortValues scalar inputs "input" (fun _ => true) (mkConst ``Unit.unit)
          fun xs inputTuple => do
        let mut routes := #[]
        let values := forward.beta #[inputTuple]
        for h : position in [:outputs.size] do
          let name := if outputs.size == 1 then "forward" else s!"forward {position}"
          let value ← normalize (← tupleComponent values position)
          routes := routes.push (← close (← observeValue backend scalar (parameters ++ xs) name
            value))
        -- The tangent of each value output under the forward-mode rule, where the arm has one.
        let tangentRule ← whnf (← ruleField rule `jvp?)
        let forwardRoutes ← if tangentRule.isAppOfArity ``Option.some 2 then
            withPortValues scalar inputs "tangent" isValue (mkConst ``Unit.unit)
                fun tangentInputs tangentTuple => do
              let tangents := tangentRule.appArg!.beta #[inputTuple, tangentTuple]
              let mut found := routes
              for h : position in [:outputs.size] do
                let (dtype, _) ← portParts outputs[position]
                unless isValue dtype do continue
                let name := if outputs.size == 1 then "jvp" else s!"jvp {position}"
                let tangent ← normalize (← tupleComponent tangents position)
                found := found.push (← close (← observeTensor backend
                  (parameters ++ xs ++ tangentInputs) name tangent))
              return found
          else pure routes
        withPortValues scalar outputs "upstream" isValue (mkConst ``Unit.unit)
            fun seeds seedTuple => do
          let gradients := backward.beta #[inputTuple, seedTuple]
          let prepared ← whnf (prepare.beta #[inputTuple, flags])
          unless prepared.isAppOfArity ``Op.Prepared.mk 6 do
            throwError "#roundoff_coverage: the prepared node {prepared} is not a structure \
              instance"
          let contributions := prepared.getAppArgs[5]!.beta #[seedTuple]
          let mut routes := forwardRoutes
          for h : position in [:inputs.size] do
            let (dtype, _) ← portParts inputs[position]
            unless isValue dtype do continue
            let suffix := if inputs.size == 1 then "" else s!" {position}"
            let gradient ← normalize (← tupleComponent gradients position)
            routes := routes.push (← close (← observeTensor backend (parameters ++ xs ++ seeds)
              s!"backward{suffix}" gradient))
            let contribution ← whnf (← normalize (← tupleComponent contributions position))
            unless contribution.isAppOfArity ``Option.some 2 do
              throwError "#roundoff_coverage: the prepared contribution {contribution} of an \
                active input is not `some`"
            let compact ← normalize contribution.appArg!
            unless compact == gradient do
              routes := routes.push (← close (← observeTensor backend
                (parameters ++ xs ++ seeds) s!"prepared backward{suffix}" compact))
            -- The contribution with only this input active, which a prepared pullback may compute
            -- by other code than with every input active.
            let alone ← onlyFlag inputs.size position
            let preparedAlone ← whnf (prepare.beta #[inputTuple, alone])
            let contributionAlone ← whnf (← normalize (← tupleComponent
              (preparedAlone.getAppArgs[5]!.beta #[seedTuple]) position))
            unless contributionAlone.isAppOfArity ``Option.some 2 do
              throwError "#roundoff_coverage: the prepared contribution {contributionAlone} of the \
                only active input is not `some`"
            let compactAlone ← normalize contributionAlone.appArg!
            unless compactAlone == compact || compactAlone == gradient do
              routes := routes.push (← close (← observeTensor backend
                (parameters ++ xs ++ seeds) s!"prepared alone backward{suffix}" compactAlone))
          -- A tangent with the computation of another route would state that route's theorem.
          routes := routes.filter fun route => !route.route.startsWith "jvp" ||
            !routes.any fun other => !other.route.startsWith "jvp" &&
              other.computation == route.computation
          let coalesced ← whnf (← ruleField rule `alias?)
          unless coalesced.isAppOfArity ``Option.some 2 do return some routes
          aliasRoutes scalar parameters close normalize coalesced.appArg! inputs outputs
            fun aliasRoutes => return some (routes ++ aliasRoutes)
  /-- The routes of the alias rule `coalesced`: its prepared node with the input at `second` the
  variable of the input at `first`, every input active. -/
  aliasRoutes {β : Type} (scalar : Expr) (parameters : Array Expr)
      (close : Route → TermElabM Route) (normalize : Expr → MetaM Expr)
      (coalesced : Expr) (inputs outputs : Array Expr) (k : Array Route → TermElabM β) :
      TermElabM β := do
    unless coalesced.isAppOf ``Op.AliasRule.mk do
      throwError "#roundoff_coverage: the alias rule {coalesced} is not a structure instance"
    let field := structureField ``Op.AliasRule 5 coalesced
    let some first ← (evalNat (← whnf (← mkAppM ``Fin.val #[← field `first]))).run |
      throwError "#roundoff_coverage: the alias position of {coalesced} is not a numeral"
    let some second ← (evalNat (← whnf (← mkAppM ``Fin.val #[← field `second]))).run |
      throwError "#roundoff_coverage: the alias position of {coalesced} is not a numeral"
    unless first != second && first < inputs.size && second < inputs.size do
      throwError "#roundoff_coverage: alias positions {first} and {second} are not two inputs"
    let remaining := inputs.eraseIdx! second
    let flags ← allFlags inputs.size
    let prepare ← field `prepare
    withPortValues scalar remaining "input" (fun _ => true) (mkConst ``Unit.unit)
        fun xs _ => do
      -- The values with the input at `second` the variable of the input at `first`. `xs` lacks the
      -- position `second`, so the positions after it are one lower.
      let at_ (position : Nat) : Expr := if position < second then xs[position]! else
        xs[position - 1]!
      let values := (List.range inputs.size).toArray.map fun position =>
        if position == second then at_ first else at_ position
      let tuple ← values.foldrM (init := mkConst ``Unit.unit) fun value rest =>
        mkAppM ``Prod.mk #[value, rest]
      let prepared ← whnf (prepare.beta #[tuple, flags])
      unless prepared.isAppOfArity ``Op.Prepared.mk 6 do
        throwError "#roundoff_coverage: the alias node {prepared} is not a structure instance"
      let value := prepared.getAppArgs[4]!
      let pullback := prepared.getAppArgs[5]!
      let mut routes := #[]
      for h : position in [:outputs.size] do
        let name := if outputs.size == 1 then "alias forward" else s!"alias forward {position}"
        let output ← normalize (← tupleComponent value position)
        routes := routes.push (← close (← observeValue backend scalar (parameters ++ xs) name
          output))
      let isValue (dtype : Expr) := dtype.isConstOf ``Port.Kind.value
      let found := routes
      withPortValues scalar outputs "upstream" isValue (mkConst ``Unit.unit)
          fun seeds seedTuple => do
        let contribution ← whnf (← normalize (← tupleComponent (pullback.beta #[seedTuple])
          first))
        unless contribution.isAppOfArity ``Option.some 2 do
          throwError "#roundoff_coverage: the alias contribution {contribution} is not `some`"
        let gradient ← normalize contribution.appArg!
        k (found.push (← close (← observeTensor backend (parameters ++ xs ++ seeds)
          "alias backward" gradient)))

/-- The routes of `operator` for `backend`, as closed real computations, or `none` when the operator
takes no scalar type. A registered operator, such as `TorchLean.Op.pool`, has the routes of
`registeredOperatorRoutes`. An operator specification has a forward and a backward route, and a
tape constructor has the forward value of its node and a route for each parent gradient. A function
returning tensors in a structure, such as `SomeTensor`, has a route for each tensor it returns.
These routes come once for each choice of constructors of the cased parameters, with the choice's
name before the route's. A function returning a pair of tensors has the routes of `pairRoutes`, and
a function returning a tensor or a scalar those of `valueRoutes`. -/
def operatorRoutes (backend : CoverageBackend) (operator : Name)
    (fixed : Array (Name × Term) := #[]) : TermElabM (Option (Array Route)) := do
  if ← isRegisteredOperator operator then return ← registeredOperatorRoutes backend operator fixed
  unless fixed.isEmpty do
    throwError "#roundoff_coverage: fixed attributes apply to registered operators only"
  if ← returnsValue operator then return ← valueRoutes backend operator
  if ← returnsTensorPair operator then return ← pairRoutes backend operator
  let packed ← recordsAnyValue operator
  let mut all : Array Route := #[]
  for (choice, label) in ← constructorChoices operator do
    let constant ← mkConstWithFreshMVarLevels operator
    let some routes ← backend.withScalar fun context hypotheses scalar =>
        withScalarInstance constant scalar choice.toArray (packed := packed) fun
          | none => return none
          | some (specialized, parameters) => do
            let type ← whnfR (← inferType specialized)
            let routes ← if type.isAppOf ``VJPOp then
                forwardBackwardRoutes backend scalar specialized parameters
              else
                let some body ← unfoldHead? specialized |
                  throwError "#roundoff_coverage: cannot unfold {operator}"
                recordedTensors backend (!(← recordsNode type)) body parameters
            -- A tensor sent to one parent on several paths of the code is one route. Distinct
            -- tensors sent to the same parent are numbered. The scalar's hypotheses come last.
            let mut distinct : Array (String × Route) := #[]
            for route in routes do
              let computation ← lambdaTelescope route.computation fun arguments body => do
                instantiateMVars (← mkLambdaFVars (context ++ arguments ++ hypotheses) body)
              let same := distinct.filter (·.1 == route.route)
              if same.any (·.2.computation == computation) then continue
              let name := if same.isEmpty then route.route else s!"{route.route} {same.size + 1}"
              let name := if choice.isEmpty then name else s!"{label} {name}"
              distinct := distinct.push (route.route, { route with route := name, computation })
            return some (distinct.map (·.2))
      | return none
    all := all ++ routes
  return some all

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

/-- `proof` and its type `type`, whose first `arity` binders are the computation's arguments,
without the contract binders at `added` that neither uses, with the new arity and the labels of the
contract hypotheses at `hypotheses` they keep. -/
def dropUnusedContracts (proof type : Expr) (arity : Nat) (added : Array Nat)
    (hypotheses : Array (Nat × String)) : MetaM (Expr × Expr × Nat × Array String) :=
  lambdaBoundedTelescope proof arity fun arguments body => do
    let statement ← instantiateForall type arguments
    let mut used := collectFVars (collectFVars {} body) statement
    let mut dropped := #[]
    let mut labels := #[]
    for index in (List.range arguments.size).reverse do
      let argument := arguments[index]!
      if added.contains index && !used.fvarSet.contains argument.fvarId! then
        dropped := dropped.push index
        continue
      if let some (_, label) := hypotheses.find? (·.1 == index) then labels := labels.push label
      used := collectFVars used (← instantiateMVars (← inferType argument))
    let kept := (Array.range arguments.size).filterMap fun index =>
      if dropped.contains index then none else arguments[index]?
    return (← mkLambdaFVars kept body, ← mkForallFVars kept statement, kept.size, labels.reverse)

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

/-- Whether `term` decodes or classifies a scalar that is not an input read. The configured
scalar and native `Float32` decode through `ExecFloat.Binary.toReal`, `toModel` and `isFinite`,
and the rounded reals through their field `NF.val`, which may appear as a projection. The error of
a `TrackedSum` decodes the partial sums it was formed from. -/
def decodesComputedValue (term : Expr) : MetaM Bool := do
  unless (applicationsOf #[``TrackedSum.error] term).isEmpty do return true
  let observations := applicationsOf #[``ExecFloat.Binary.toReal, ``ExecFloat.Binary.toModel,
    ``ExecFloat.Binary.isFinite, ``FloatLib.Floats.Formats.Flocq.NF.val] term
    (structures := #[``FloatLib.Floats.Formats.Flocq.NF])
  observations.anyM fun
    | .proj _ _ value => return !(← isInputRead value)
    | observation => return !(← isInputRead observation.appArg!)

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
input-bound hypotheses among its binders, and the propagation theorems and auxiliary lemmas its
proof uses. -/
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

/-- The propagation theorems generated in each module, by name. A composite that does not prove
drops the declarations it made, and a later one declares them again from here. -/
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

/-- The constants of `terms` and of the statements and values of `auxiliary`. -/
def usedConstants (terms : Array Expr) (auxiliary : Array ConstantInfo := #[]) : NameSet :=
  let bodies := auxiliary.foldl (fun bodies info => (bodies.push info.type) ++ info.value?.toArray)
    terms
  bodies.foldl (fun used term => term.getUsedConstants.foldl (·.insert ·) used) {}

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
arguments, with proof `proof` and the auxiliary lemmas `auxiliary` it keeps. The contracts of
`functions` are among those binders. -/
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
        used := used ++ usedConstants #[generated.proof] generated.auxiliary
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

/-- Declare again those of the auxiliary lemmas `declarations` that the environment does not hold,
as a restored state lacks them. -/
def restoreDeclarations (declarations : Array ConstantInfo) : CoreM Unit := do
  for declaration in declarations do
    if (← getEnv).contains declaration.name then continue
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
uses, and when `localize` the bounds of inputs it reads at one coordinate are taken there
(`localizeBounds`). -/
def propagationTheoremOf (base : Environment) (route : Route) (mixed localize : Bool) :
    TermElabM PropagationTheorem := do
  let (computation, added) ← withContracts route.computation mixed
    fun computation added _ => pure (computation, added)
  unless added.isEmpty do throwError "roundoff: the route applies a function under a contract"
  let some (lifted, bounds) ← withInputBounds computation mixed |
    throwError "roundoff: an argument holds values of the scalar but is no tensor or scalar"
  let proof ← generateProof lifted mixed
  let type ← instantiateMVars (← inferType proof)
  let auxiliary ← addedDeclarations base #[type, proof]
  let unsupplied ← unsuppliedBounds proof auxiliary
  unless unsupplied.isEmpty do throwError "roundoff: the input bounds of {unsupplied} are premises"
  let arms ← directArms #[type, proof] auxiliary
  let arity ← lambdaTelescope lifted fun binders _ => pure binders.size
  let (type, proof, bounds) ← if localize then localizeBounds type proof arity bounds mixed
    else pure (type, proof, bounds)
  return { type, proof, bounds, arms, auxiliary }

/-- The propagation theorems of the callees of `computation` (`callees`) in this mode, registered
as rules, and why the others are not generated. Each is the propagation theorem
(`propagationTheoremOf`) of a configured route of a callee, analyzed with its own heartbeat budget
and declared in the current namespace under `theoremName` with `propagationSuffix`. A callee
contributes when every route of it has one and serves a composite (`isArm`). -/
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
      let name := theoremName (← getCurrNamespace) callee (propagationSuffix route.route mixed)
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
        | .ok generated => generatedPropagations.modify (·.insert (module, name) generated)
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
outcome. The binders at `added` belong to the contracts of functions, whose hypotheses are at
`hypotheses`, and the theorem keeps those its proof uses. A configured route over the heartbeat
budget, and only such a route, is analyzed once more when its callees have propagation theorems
(`propagationArms`), with these as rules and a budget of its own, keeping the ones its proof applies
(`armClosure`) as its arms. -/
def coverRoute (entry : CoverageEntry) (computation : Expr) (mixed : Bool) (name : Name)
    (ref : Syntax) (added : Array Nat := #[]) (hypotheses : Array (Nat × String) := #[]) :
    TermElabM CoverageEntry := do
  let saved ← saveState
  let base ← getEnv
  let analyze : TermElabM (Expr × Expr × Nat × Array String) := do
    let proof ← generateProof computation mixed
    let type ← instantiateMVars (← inferType proof)
    let arity ← lambdaTelescope computation fun arguments _ => pure arguments.size
    let (proof, type, arity, functions) ← if added.isEmpty then pure (proof, type, arity, #[])
      else dropUnusedContracts proof type arity added hypotheses
    return (type, proof, arity, functions)
  let declare (result : Expr × Expr × Nat × Array String) := do
    addDecl (.thmDecl { name, levelParams := [], type := result.1, value := result.2.1 })
    addDocStringCore name s!"Bound on the {entry.mode} error of the `{entry.route}` route of \
      `{entry.operator}` against its real reference, for the {entry.backend} scalar, generated \
      by `#roundoff_coverage`."
    return result
  let mut generated ← tryCatchRuntimeEx (Except.ok <$> (analyze >>= declare))
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
          let attempt ← tryCatchRuntimeEx (Except.ok <$> withCurrHeartbeats do
              let result ← analyze
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
    | .ok (type, proof, arity, functions) => do
      addDeclarationRangesFromSyntax name ref
      let entry := { entry with evidence := name, status := .conditional, arms }
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

/-- An entry of `#roundoff_coverage`: a declaration, or a registered operator with some attributes
fixed, as `Op.crossEntropy (mean := true)`. -/
syntax coverageEntry := ident (" (" ident " := " term ")")*

/-- `#roundoff_coverage family := operator, ...` analyzes every route of each registered
operator with `roundoff` and records the outcome, by family. An operator listed with fixed
attributes has those routes again with the attributes fixed. -/
syntax (name := coverage) "#roundoff_coverage" (ppLine ident " := " coverageEntry,+)+ :
  command

@[command_elab coverage, inherit_doc coverage]
def elabCoverage : CommandElab
  | `(#roundoff_coverage $[$families := $entries,*]*) => do
    for family in families, group in entries do
      for entry in group.getElems do
        let `(coverageEntry| $operator:ident $[ ($names := $values)]*) := entry |
          throwUnsupportedSyntax
        let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo operator
        coverOperator family.getId name entry (names.map (·.getId) |>.zip values)
  | _ => throwUnsupportedSyntax

/-- Declare in the current namespace the propagation theorem (`propagationTheoremOf`) of every
configured route of `callee` in this mode, after the propagation theorems its proof applies. Each
analysis has a heartbeat budget of its own, and one that fails is an error. -/
def declarePropagations (callee : Name) (mixed : Bool) (ref : Syntax) : TermElabM Unit := do
  let routes ← withCurrHeartbeats do
    let routes ← operatorRoutes .configured callee
    if routes.any (!·.isEmpty) then pure routes else componentRoutes .configured callee
  let some routes := routes | throwError "roundoff: {callee} has no configured routes"
  for route in routes do
    let name := theoremName (← getCurrNamespace) callee (propagationSuffix route.route mixed)
    let base ← getEnv
    let generated ← withCurrHeartbeats <|
      propagationTheoremOf base route mixed (localizedCallees.contains callee)
    generatedPropagations.modify (·.insert (← getMainModule, name) generated)
    declareArms (← armClosure #[generated.proof] generated.auxiliary)
    restoreDeclarations generated.auxiliary
    addDecl (.thmDecl { name, levelParams := [], type := generated.type, value := generated.proof })
    addDocStringCore name "Error bound of a route of an operation under the hypothesis that each \
      input lies within real budgets of real references: a propagation theorem declared by \
      `#roundoff_propagation`."
    addDeclarationRangesFromSyntax name ref

/-- `#roundoff_propagation mode := operation, ...` declares, in the current namespace, the
propagation theorems of every configured route of each operation in `mode`, `absolute` or
`mixed`, as the second analysis of a composite declares those of its callees
(`propagationArms`). -/
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
