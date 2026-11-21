/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff
public import TorchLean.NN.VJPOp
public import TorchLean.Tensor.Op.Rule
public import TorchLean.Tensor.SomeTensor
public import TorchLean.NN.Tape
public import FloatLib.Floats.Formats.IEEE754.Native
public import TorchLean.Verify.Floats.FP32
public import TorchLean.Tensor.NF

/-!
# Routes of registered declarations

`#roundoff_coverage` (`TorchLean.Verify.RuntimeApprox.Roundoff.Coverage`) analyzes the routes that
this module reads from the code of a declaration: each observes one coordinate of a tensor the
declaration computes, or the scalar it computes, in the reals, as a function of its variables, for
the configured scalar over an arbitrary format, native `Float32` or the rounded reals `FP32`
(`CoverageBackend`).

An operator specification has two routes, its forward map and its pullback. A tape constructor has
the forward value of the node it records and one route for each contribution its backward closure
sends to a parent. The routes are read from its code, with the values it saves. A tape constructor
that records a value of any type, such as `TorchLean.Tape.leaf`, is read on the CPU tape, whose
values are tensors over the scalar packed with their shapes as `TorchLean.SomeTensor`. A function
returning tensors, such as the sum that accumulates gradients, has one route for each tensor it
returns. A function returning a tensor or a scalar, such as a general tensor operator, has one
route, `value`, read from its result. A tensor of indices is read through the cast of its entry
into the scalar, and a tensor of Booleans, such as a comparison, through `1` or `0`. A function
returning a pair of tensors, such as the values and indices of `TorchLean.Tensor.sort`, has the
routes `values` and `indices`, read from its two components. A parameter of such a function whose
type is a nonrecursive inductive type with several constructors, and which a later instance
parameter depends on or the function matches on, such as the operation of `TorchLean.Tensor.reduce`,
is taken one constructor at a time with the constructor's fields as variables: the function then has
one route per constructor, named after it. An operator with such a parameter, such as the reduction
of `TorchLean.Op.pool`, has its routes once for each constructor, named after the constructor and
the route.

A constructor of `TorchLean.Op` has the routes of its arm of `TorchLean.Op.rule?`: the forward value
at each output, the tangent of each value output under the forward-mode rule where the arm has one
and no other route computes it, the gradient of each value input from the dense pullback, the
prepared pullback where it is other code, and the routes of an alias rule. A frozen scalar input is
a scalar variable, a capability the scalar lacks is an instance variable, the element of a
generalized operator is the scalar (its Boolean and integer elements compute without rounding), and
an attribute of a nonrecursive inductive type other than `Bool` and `Option` is taken one
constructor at a time.
-/

public meta section

namespace TorchLean.Roundoff

open Lean Meta Elab Term Command
open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

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

/-- The shape of `type` when it is a tensor of integer words, `Tensor (BitVec w) s`. -/
def wordTensorShape? (type : Expr) : MetaM (Option Expr) := do
  let type ← whnfR type
  unless type.isAppOfArity ``Tensor.Rep 3 do return none
  return if (← whnfR type.getAppArgs[0]!).isAppOfArity ``BitVec 1 then some type.getAppArgs[1]!
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
a new coordinate through the cast of its entry into `scalar`, a tensor of integer words through the
cast of the two's-complement value of its entry, a tensor of Booleans through `1` or `0` in
`scalar`, or a scalar. -/
def observeValue (backend : CoverageBackend) (scalar : Expr) (locals : Array Expr)
    (route : String) (value : Expr) : TermElabM Route := do
  if let some shape ← indexTensorShape? (← inferType value) then
    return ← withLocalDeclD `index (← mkAppM ``Shape.Coord #[shape]) fun index => do
      let entry ← mkAppM ``Fin.val #[← mkAppM ``Tensor.Rep.get #[value, index]]
      let cast ← mkAppOptM ``Nat.cast #[scalar, none, entry]
      let computation ← mkLambdaFVars ((← dependencies locals value).push index)
        (← backend.decode cast)
      return { route, computation }
  if let some shape ← wordTensorShape? (← inferType value) then
    return ← withLocalDeclD `index (← mkAppM ``Shape.Coord #[shape]) fun index => do
      let entry ← mkAppM ``BitVec.toInt #[← mkAppM ``Tensor.Rep.get #[value, index]]
      let cast ← mkAppOptM ``Rat.cast #[scalar, none,
        ← mkAppOptM ``Int.cast #[mkConst ``Rat, none, entry]]
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

/-- The value of an attribute of type `domain` that is taken without a label: the scalar element
for the element of a generalized operator (`Op.Element` or `Op.NumericElement`), as the Boolean and
integer elements compute without rounding, and the left fold for a reduction order
(`Tensor.ReduceOrder`), as another order is taken by fixing the attribute (the `fixed` of
`registeredOperatorRoutes`). -/
def defaultAttribute? (domain : Expr) : MetaM (Option Expr) := do
  match (← whnfR domain).getAppFn.constName? with
  | some `TorchLean.Op.Element => return some (mkConst `TorchLean.Op.Element.scalar)
  | some `TorchLean.Op.NumericElement => return some (mkConst `TorchLean.Op.NumericElement.scalar)
  | some `TorchLean.Tensor.ReduceOrder =>
      return some (mkConst `TorchLean.Tensor.ReduceOrder.leftFold)
  | _ => return none

/-- The constructors an attribute of a registered operator is taken through one at a time: those of
a nonrecursive inductive type other than `Bool`, `Option` and one with a default
(`defaultAttribute?`) with several constructors, such as the reduction of a pooling. -/
def splitConstructors? (domain : Expr) : MetaM (Option (List Name)) := do
  if ← isProp domain then return none
  if (← defaultAttribute? domain).isSome then return none
  let .const typeName _ := (← whnfR domain).getAppFn | return none
  if typeName == ``Bool || typeName == ``Option then return none
  let some (.inductInfo type) := (← getEnv).find? typeName | return none
  if type.isRec || type.numIndices != 0 || type.ctors.length < 2 then return none
  return some type.ctors

/-- The binder positions of the explicit attributes of a type `List τ` that occur in a list of ports
of an operator of type `type`, with attributes `attributes`, that `portList` cannot read, such as
the input sizes of a concatenation, whose ports are mapped from them. The test is syntactic: such
an attribute is taken at a few lengths (`listLengths`) without a proof that it fixes the number of
ports. -/
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

/-- How a split attribute of a registered operator is taken: as a constructor of its inductive
type, with its fields as variables, or as a list of `count` variables. -/
inductive AttributeChoice where
  | constructor (name : Name)
  | entries (count : Nat)
  deriving BEq, Inhabited

/-- The lengths at which the list attribute `list` of a registered operator with attributes
`attributes` is taken. When the type of an attribute is an inductive family with the index
`list.length`, they are the values of that index at the family's constructors, sorted without
repeats, so a list of orders of a special function is taken at each number of arguments a function
has. Otherwise, or when an index of a constructor is not a numeral, the list is taken at two
entries. -/
def listLengths (attributes : Array Expr) (list : Expr) : MetaM (List Nat) := do
  for binder in attributes do
    let type ← whnfR (← inferType binder)
    let .const typeName _ := type.getAppFn | continue
    let some (.inductInfo info) := (← getEnv).find? typeName | continue
    let some index := (type.getAppArgs.extract info.numParams).findIdx? fun index =>
        index.isAppOfArity ``List.length 2 && index.appArg! == list | continue
    let lengths ← info.ctors.mapM fun constructor => do
      forallTelescope (← getConstInfo constructor).type fun _ result =>
        (evalNat (result.getArg! (info.numParams + index))).run
    if let some lengths := lengths.mapM id then
      return lengths.mergeSort.eraseDups
  return [2]

/-- The word for a count below ten, and its digits from ten on. -/
def countWord (count : Nat) : String :=
  ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"].getD count
    (toString count)

/-- The choices for the split attributes of the registered operator `constant`, by binder position,
each with its label: with `constructors`, a constructor for an attribute of a nonrecursive
inductive type (`splitConstructors?`), labelled by its name, and for a list attribute that the
ports are mapped from (`portListAttributes`), a list of variables at each of its lengths
(`listLengths`, `withListEntries`), labelled by the count in words and the binder's name, such as
`two sizes`. The attributes named in `fixed` are not split. -/
partial def attributeChoices (constant : Expr) (fixed : Array Name := #[])
    (constructors : Bool := true) : MetaM (List (List (Nat × AttributeChoice) × String)) := do
  forallTelescope (← inferType constant) fun attributes type => do
    let lists ← portListAttributes attributes type
    let mut choices : List (List (Nat × AttributeChoice) × List String) := [([], [])]
    for h : position in [:attributes.size] do
      let declaration ← attributes[position].fvarId!.getDecl
      unless declaration.binderInfo.isExplicit do continue
      if fixed.contains declaration.userName then continue
      if lists.contains position then
        let name := declaration.userName.eraseMacroScopes
        let lengths ← listLengths attributes attributes[position]
        choices := choices.flatMap fun (choice, labels) => lengths.map fun count =>
          (choice ++ [(position, .entries count)], labels ++ [s!"{countWord count} {name}"])
        continue
      unless constructors do continue
      let some cs ← splitConstructors? declaration.type | continue
      choices := choices.flatMap fun (choice, labels) => cs.map fun c =>
        (choice ++ [(position, .constructor c)], labels ++ [c.getString!])
    return choices.map fun (choice, labels) => (choice, " ".intercalate labels)

/-- Run `k` on the list of `count` new variables of the element type of the list type `type`,
named after `name` with their positions, and on those variables. -/
def withListEntries {α : Type} (type : Expr) (name : Name) (count : Nat)
    (k : Expr → Array Expr → TermElabM α) : TermElabM α := do
  visit (← whnfR type).appArg! count #[]
where
  /-- Add `remaining` more variables of type `element` after `entries`. -/
  visit (element : Expr) : Nat → Array Expr → TermElabM α
    | 0, entries => do k (← mkListLit element entries.toList) entries
    | remaining + 1, entries =>
        withLocalDeclD (name.appendIndexAfter entries.size) element fun entry =>
          visit element remaining (entries.push entry)

/-- Apply the registered operator `constant` to fresh variables for its attributes and run `k` on
the operator and the variables. At the binder positions of `choice` the arguments are the given
constructors with their fields as variables, or lists of the given numbers of variables
(`withListEntries`). An element of a generalized operator is the scalar and a reduction order the
left fold (`defaultAttribute?`), and the attributes named in `fixed` are the given terms, elaborated
at the attribute's type, each hole of a term (`?width`) a variable of its own. -/
partial def withAttributes {β : Type} (constant : Expr) (choice : List (Nat × AttributeChoice))
    (k : Expr → Array Expr → TermElabM β) (fixed : Array (Name × Term) := #[]) : TermElabM β :=
  visit constant #[] 0
where
  visit (value : Expr) (attributes : Array Expr) (position : Nat) : TermElabM β := do
    let type ← whnfR (← inferType value)
    let .forallE name domain _ info := type | k value attributes
    if let some (_, term) := fixed.find? (·.1 == name) then
      let argument ← Term.elabTermEnsuringType term domain
      Term.synthesizeSyntheticMVarsNoPostponing
      return ← withHoles argument attributes fun argument attributes =>
        visit (mkApp value argument) attributes (position + 1)
    match choice.find? (·.1 == position) with
    | some (_, .entries count) =>
        return ← withListEntries domain name count fun argument entries =>
          visit (mkApp value argument) (attributes ++ entries) (position + 1)
    | some (_, .constructor constructor) =>
        return ← withConstructor domain constructor fun argument fields =>
          visit (mkApp value argument) (attributes ++ fields) (position + 1)
    | none => pure ()
    if let some default_ ← defaultAttribute? domain then
      return ← visit (mkApp value default_) attributes (position + 1)
    withLocalDecl name info domain fun variable_ =>
      visit (mkApp value variable_) (attributes.push variable_) (position + 1)
  /-- Assign each metavariable left in `argument` a new variable named after it, added to
  `attributes`. The metavariable gives up its name, so the term elaborates afresh at each route. -/
  withHoles (argument : Expr) (attributes : Array Expr) (k : Expr → Array Expr → TermElabM β) :
      TermElabM β := do
    let argument ← instantiateMVars argument
    let some hole := (← getMVars argument)[0]? | k argument attributes
    let holeType ← instantiateMVars (← hole.getType)
    let holeName := (← hole.getDecl).userName
    hole.setUserName .anonymous
    withLocalDeclD holeName holeType fun variable_ => do
      hole.assign variable_
      withHoles argument (attributes.push variable_) k

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
The attributes in `fixed` are the given terms, each hole of a term (`?width`) a variable of the
routes, and every route is named after them first. -/
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

end TorchLean.Roundoff
