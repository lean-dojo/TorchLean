/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Check.Einsum
public import TorchLean.Tensor.Einops.Check.Pack
public import TorchLean.Tensor.Promotion
public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Positivity.Finset
import Mathlib.Tactic.Ring.RingNF
public meta import TorchLean.Tensor.Datatype
public meta import TorchLean.Tensor.Einops.Check.Einsum
public meta import TorchLean.Tensor.Einops.Check.Pack
public meta import TorchLean.Tensor.Einops.Check.Transform
public meta import TorchLean.Tensor.Native.Construction
public meta import TorchLean.Tensor.Einops.Elab.Syntax
public meta import Lean.Elab.Tactic -- shake: keep
meta import Mathlib.Tactic.Positivity -- shake: keep
meta import Mathlib.Tactic.Ring -- shake: keep
public import TorchLean.Tensor.Einops.Check.ParseShape -- shake: keep
public import TorchLean.Tensor.Einops.Semantics.Transform.RearrangeExpand -- shake: keep
public import TorchLean.Tensor.Einops.Semantics.Transform.Reduction -- shake: keep
import Mathlib.Tactic.FinCases
import TorchLean.Tensor.Datatype
import TorchLean.Tensor.Native.Construction
import TorchLean.Tensor.Einops.Elab.Syntax

/-!
# Shared elaboration infrastructure

The elaborators of every operation family share the metaprogramming helpers here. They inspect
dependent tensor types, keep symbolic natural-number dimensions intact, build heterogeneous
tensor families and emit checker certificates that carry proofs. The literal term grammar has
its own module, `TorchLean.Tensor.Einops.Elab.Syntax`.

The helpers sit in the namespace `TorchLean.Einops.Elab` as implementation infrastructure,
outside the user API, and every operation module calls them directly.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

deriving instance Lean.ToExpr for Span
deriving instance Lean.ToExpr for Located
deriving instance Lean.ToExpr for Axis
deriving instance Lean.ToExpr for CompositeAxis
deriving instance Lean.ToExpr for Expression
deriving instance Lean.ToExpr for TransformPattern
deriving instance Lean.ToExpr for EinsumPattern
deriving instance Lean.ToExpr for PackPattern
deriving instance Lean.ToExpr for TransformKind
deriving instance Lean.ToExpr for AxisId
deriving instance Lean.ToExpr for EinsumAxis
deriving instance Lean.ToExpr for NormalForm

section

open Tensor (Rep)

/-- Render a singular or plural label for a diagnostic source range. -/
def pluralColumns (length : Nat) : String :=
  if length = 1 then "column" else "columns"

/-- Draw a caret marker beneath the source range selected by a diagnostic. -/
def patternMarker (span : Span) : String :=
  String.ofList <|
    List.replicate span.offset ' ' ++
      List.replicate (max 1 span.length) '^'

/-- Raise an elaboration error that prints the pattern with carets under the offending range. -/
def throwPatternDiagnostic {α : Type}
    (operation phase source : String) (message : String)
    (span : Span) : TermElabM α :=
  throwError
    "{operation} {phase} error at {pluralColumns span.length} \
      {span.offset}-{span.stop}:\n\
      {message}\n\
      {source}\n\
      {patternMarker span}"

/--
Expose the elements of a statically known list while retaining symbolic
element expressions.
-/
partial def staticListElements? (value : Expr) :
    MetaM (Option (List Expr)) := do
  let value ← withTransparency .all <| whnf value
  if value.isAppOfArity ``List.nil 1 then
    return some []
  if value.isAppOfArity ``List.cons 3 then
    let arguments := value.getAppArgs
    let some tail ← staticListElements? arguments[2]! | return none
    return some (arguments[1]! :: tail)
  return none

/-- Read natural-number expressions only when every dimension is concrete. -/
def concreteNatExpressions? (dimensions : List Expr) :
    MetaM (Option Shape) := do
  let mut concrete : Shape := []
  for dimension in dimensions do
    let dimension ← withTransparency .all <| whnf dimension
    let some value ← getNatValue? dimension | return none
    concrete := List.concat concrete value
  return some concrete

/--
Read shape-expression spines as ordinary shapes only when every dimension
reduces to a natural-number literal.
-/
def concreteShapes? (shapes : List (List Expr)) :
    MetaM (Option (List Shape)) := do
  let mut concreteShapes : List Shape := []
  for dimensions in shapes do
    let some shape ← concreteNatExpressions? dimensions | return none
    concreteShapes := concreteShapes.concat shape
  return some concreteShapes

/-- Compute the length of a statically known list without inspecting its elements. -/
def concreteListLength? (value : Expr) :
    MetaM (Option Nat) :=
  return (← staticListElements? value).map List.length

/-- Evaluate an integer expression when reduction at full transparency exposes a literal. -/
def concreteIntExpression? (value : Expr) :
    MetaM (Option Int) := do
  if let some integer ← getIntValue? value then
    return some integer
  let value ← withTransparency .all <| whnf value
  if value.isAppOfArity ``Int.ofNat 1 then
    let some natural ← getNatValue? value.appArg! | return none
    return some natural
  if value.isAppOfArity ``Int.negSucc 1 then
    let some natural ← getNatValue? value.appArg! | return none
    return some (Int.negSucc natural)
  getIntValue? value

/-- Read a statically written list whose integer elements all reduce to literals. -/
def concreteIntList? (value : Expr) :
    MetaM (Option (List Int)) := OptionT.run do
  let elements ← OptionT.mk (staticListElements? value)
  elements.mapM fun element => OptionT.mk (concreteIntExpression? element)

/-- Read concrete unpack metadata as a statically written list of integer lists. -/
def concreteRequestedShapes? (value : Expr) :
    MetaM (Option RequestedShapes) := OptionT.run do
  let shapes ← OptionT.mk (staticListElements? value)
  shapes.mapM fun shape => OptionT.mk (concreteIntList? shape)

/--
Return the static shape, element type and storage of a `TorchLean.Tensor` type, in that order.

The public tensor abbreviation is unfolded to its packed representation first. Plain coordinate
functions are not accepted as tensors.
-/
def tensorTypeInfo (tensorSyntax : Syntax) (tensorType : Expr) :
    TermElabM (Expr × Expr × Expr) := do
  let tensorType ← withTransparency .reducible <| whnf tensorType
  unless tensorType.isAppOfArity ``Tensor.Rep 3 do
    throwErrorAt tensorSyntax
      "expected a `TorchLean.Tensor α shape`, but the term has type\
        {indentExpr tensorType}"
  let arguments := tensorType.getAppArgs
  return (arguments[1]!, arguments[0]!, arguments[2]!)

/--
Elaborate one tensor and return it with its scalar type, storage and dimension list, keeping
symbolic dimension expressions.

A nonempty list of equal-shaped tensors is stacked into one tensor whose leading axis is the
statically known list length.
-/
def elaborateTensor (tensorSyntax : Syntax) :
    TermElabM (Expr × Expr × Expr × List Expr) := do
  let input ← elabTermAndSynthesize tensorSyntax none
  let inputType ← instantiateMVars (← inferType input)
  let inputType ← withTransparency .reducible <| whnf inputType
  let (tensor, tensorType, leadingDimensions) ←
    if inputType.isAppOfArity ``List 1 then
      let componentType := inputType.appArg!
      let some componentCount ← concreteListLength? input
        | throwErrorAt tensorSyntax
            "a list tensor input must have a statically reducible length"
      if componentCount = 0 then
        throwErrorAt tensorSyntax
          "a list tensor input must contain at least one tensor"
      let stackedTensor ←
        mkAppM ``Tensor.Rep.stackList #[input]
      pure (stackedTensor, componentType, [mkNatLit componentCount])
    else
      pure (input, inputType, [])
  let (shape, scalarType, storage) ← tensorTypeInfo tensorSyntax tensorType
  let some dimensions ← staticListElements? shape
    | throwErrorAt tensorSyntax
        "the tensor shape must have a statically known list structure; its \
          dimensions may be symbolic natural-number expressions"
  -- Sealed certificates expand let-bound locals, so a let-bound operand whose value is still
  -- being elaborated (inside a binary operator in a `do` block, for example) would leave its
  -- pending metavariables in the generated term: elaborate again once they are solved.
  if (← instantiateMVars (← zetaReduce (← instantiateMVars tensor))).hasMVar then
    tryPostpone
  return (tensor, scalarType, storage, leadingDimensions ++ dimensions)

/-- Elaborate named supplementary axis lengths as natural-number expressions. -/
def supplementaryExpr
    (lengthSyntax : Array (TSyntax `einopsAxisLength)) :
    TermElabM (List (String × Expr)) := do
  let mut lengths : List (String × Expr) := []
  for item in lengthSyntax do
    let (axisName, length) ←
      match item with
      | `(einopsAxisLength| $axis:ident := $length:term) =>
          pure (axis.getId.eraseMacroScopes.toString, length)
      | `(einopsAxisLength| $axis:str := $length:term) =>
          pure (axis.getString, length)
      | _ => throwUnsupportedSyntax
    let lengthExpr ←
      elabTermEnsuringType length (mkConst ``Nat)
    synthesizeSyntheticMVarsNoPostponing
    let lengthExpr ← instantiateMVars lengthExpr
    lengths := lengths.concat (axisName, lengthExpr)
  return lengths

/-- Reify supplementary axis lengths when every supplied expression is concrete. -/
def concreteSupplementary?
    (supplementary : List (String × Expr)) :
    MetaM (Option SupplementaryLengths) := do
  let mut concrete : SupplementaryLengths := []
  for (name, length) in supplementary do
    let length ← withTransparency .all <| whnf length
    let some value ← getNatValue? length | return none
    concrete := concrete.concat (name, value)
  return some concrete

/--
Apply a structure constructor to one more argument after checking the argument against the
dependent field type.
-/
def applyConstructorArgument (constructor constructorType argument : Expr) :
    MetaM (Expr × Expr) := do
  let constructorType ← whnf constructorType
  let .forallE _ expectedType resultType _ := constructorType
    | throwError "internal error: certificate constructor has too few fields"
  let argumentType ← inferType argument
  unless ← isDefEq expectedType argumentType do
    throwError
      "internal error: certificate field has type{indentExpr argumentType}\n\
        but expected{indentExpr expectedType}"
  return (mkApp constructor argument, resultType.instantiate1 argument)

/--
Seal a generated certificate in a kernel-checked auxiliary theorem.

The module-qualified theorem kind prevents exported terms in independently
compiled modules from producing the same `_einops_N` declaration names.
-/
def sealCertificate (proposition certificate : Expr) : MetaM Expr := do
  let theoremKind := (← getMainModule) ++ `_einops
  mkAuxTheorem proposition certificate
    (zetaDelta := true) (kind? := theoremKind)

/--
Close a generated value over its free locals in a compiled auxiliary definition named from `stem`.
Let-bound locals stay parameters (`zetaDelta := false`), so the caller computes each one once.
-/
def sealAuxDefinition (stem : Name) (value : Expr) : TermElabM Expr := do
  mkAuxDefinitionFor (← mkAuxName stem) value (zetaDelta := false)

/--
Build a proof-valued structure from its parameters, the fields already proved, and `decide`
proofs, checked by kernel reduction, for the remaining fields.
-/
def buildCertificate (constructorName : Name)
    (parameters fixedFields : Array Expr) (decidableFields : Nat) :
    MetaM Expr := do
  let mut constructor ← mkConstWithFreshMVarLevels constructorName
  let mut constructorType ← inferType constructor
  for parameter in parameters do
    (constructor, constructorType) ←
      applyConstructorArgument constructor constructorType parameter
  for field in fixedFields do
    (constructor, constructorType) ←
      applyConstructorArgument constructor constructorType field
  for _ in [0:decidableFields] do
    let fieldType ← whnf constructorType
    let .forallE _ proposition _ _ := fieldType
      | throwError
          "internal error: certificate constructor has too few decidable fields"
    -- Keep the proof typed by the unreduced field proposition, so downstream
    -- tactics can treat reflected certificates abstractly without unfolding
    -- parser and normalization definitions to recover their type.
    let certificate ← withTransparency .all <| mkDecideProof proposition
    let certificate ← sealCertificate proposition certificate
    (constructor, constructorType) ←
      applyConstructorArgument constructor constructorType certificate
  return constructor

/--
Prove a generated proposition with a focused tactic and seal the kernel certificate in an auxiliary
theorem, reporting failure as a missing user hypothesis. With `clearContext`, the goal first drops
every local its statement does not mention, so the certificate cannot capture unrelated loop
coordinates, and a failure is an internal error.
-/
def certifyWithTactic (description : String) (proposition : Expr)
    (tactic : Syntax) (clearContext := false) : TermElabM Expr := do
  let invariantGoal ← mkFreshExprSyntheticOpaqueMVar proposition
  let focusedGoal ←
    if clearContext then
      invariantGoal.mvarId!.withContext do
        return (← invariantGoal.mvarId!.tryClearMany' (← getLCtx).getFVarIds).1
    else
      pure invariantGoal.mvarId!
  let remainingGoals ←
    Lean.Elab.Tactic.run focusedGoal do
      discard <| observing? do
        let result ←
          Mathlib.Tactic.withResetServerInfo <|
            Lean.Elab.Tactic.evalTactic tactic
        if result.result?.isNone || result.msgs.hasErrors then
          throwError "generated certificate tactic failed"
  unless remainingGoals.isEmpty do
    let target ← instantiateMVars (← remainingGoals[0]!.getType)
    if clearContext then
      throwError "internal error: could not prove {description}:{indentExpr target}"
    throwError
      "could not prove {description}:{indentExpr target}\n\
        Add the shape equality, broadcasting, positivity, or divisibility \
        hypothesis needed by this operation."
  let certificate ← instantiateMVars invariantGoal
  if certificate.hasMVar then
    throwError
      "internal error: the proof of {description} still contains metavariables"
  if clearContext && certificate.hasSorry then
    throwError
      "internal error: the proof of {description} contains a synthetic sorry"
  let proposition ← instantiateMVars proposition
  sealCertificate proposition certificate

/--
Discharge a general symbolic shape invariant using the arithmetic and
finite-family vocabulary shared by transformations and packing.
-/
def certifyGeneratedInvariant (description : String)
    (proposition : Expr) : TermElabM Expr := do
  let tactic ←
    `(tactic|
      simp_all (config := { zeta := true }) [
        ParseShape.AxisMatches,
        PackPattern.explicitRank,
        PackPattern.starShape,
        Pack.starShape,
        Pack.metadata,
        Pack.segmentLengths,
        Pack.packedAxisLength,
        Pack.output,
        TransformPlan.allAxes,
        AxisTuple.groupedShape,
        TransformPlan.inferredInput,
        TransformPlan.inferredOutput,
        TransformPlan.literalAxesAgree,
        TransformPlan.normalized,
        Transform.reducedAxes,
        Transform.reductionShape,
        Transform.reductionFiberSize,
        NormalForm.inputAxes,
        NormalForm.outputAxes,
        PartialAxisLengths.seed,
        PartialAxisLengths.set,
        List.lookup,
        CompositeAxis.semanticAxes,
        Expression.ellipsisCount,
        Expression.flatAxes,
        Nat.div_mul_cancel,
        Nat.dvd_iff_mod_eq_zero,
        Nat.mod_add_div,
        Nat.mul_div_cancel,
        Nat.mul_div_cancel_left,
        Nat.mul_div_left,
        Nat.mul_div_right,
        Nat.mul_mod,
        Nat.mul_mod_left,
        Nat.mul_mod_right,
        Nat.sub_add_cancel,
        Nat.zero_mod] <;>
      first
      | assumption
      | rfl
      | exact Nat.mul_mod_left _ _
      | exact Nat.mul_mod_right _ _
      | exact Or.inl (Nat.mul_div_left _ (by positivity))
      | exact Or.inl (Nat.mul_div_right _ (by positivity))
      | positivity
      | omega
      -- Statically written tensor families quantify over `Fin n`, and
      -- enumerating that index exposes each symbolic shape equality.
      | (intro component; fin_cases component <;> simp_all)
      | ring
      | aesop)
  certifyWithTactic description proposition tactic

/-- Human-readable description of a logical transformation axis. -/
def _root_.TorchLean.Einops.AxisId.description : AxisId → String
  | .named name => s!"axis '{name}'"
  | .anonymous value _ => s!"anonymous axis {value}"
  | .ellipsis index => s!"ellipsis axis {index}"

/-- `assignments` with `(axis, length)` appended, or `assignments` itself when it has an entry for
`axis`. -/
def appendAxisExpression {κ : Type} [BEq κ] (assignments : List (κ × Expr))
    (axis : κ) (length : Expr) :
    List (κ × Expr) :=
  if (assignments.lookup axis).isSome then
    assignments
  else
    assignments.concat (axis, length)

/-- Construct a left-associated natural-number product in source order. -/
def natProductExpr : List Expr → MetaM Expr
  | [] => pure (mkNatLit 1)
  | first :: rest =>
      rest.foldlM (init := first) fun product factor =>
        mkAppM ``Nat.mul #[product, factor]

/-- Construct a source-order natural-number sum expression. -/
def natSumExpr : List Expr → MetaM Expr
  | [] => pure (mkNatLit 0)
  | first :: rest =>
      rest.foldlM (init := first) fun total summand =>
        mkAppM ``Nat.add #[total, summand]

/--
Build index `index` of `Fin count` as `Fin.succ` applied `index` times to the zero of
`Fin (count - index)`.

`Fin.cases` consumes this successor form and unfolding `Fin.foldl` produces it, so generated
terms stay definitionally transparent to proof tactics, which a modular `OfNat` numeral does not.
-/
def finIndexExpr (count index : Nat) : MetaM Expr := do
  unless index < count do
    throwError "internal error: index {index} is outside count {count}"
  let remainingType ← mkAppM ``Fin #[mkNatLit (count - index)]
  let mut value ← mkNumeral remainingType 0
  for _ in [:index] do
    value ← mkAppM ``Fin.succ #[value]
  return value

/-- Resolve a list of logical axes to their symbolic length expressions. -/
def axisExpressions (assignments : List (AxisId × Expr))
    (axes : List AxisId) : MetaM (List Expr) := do
  let mut lengths : List Expr := []
  for axis in axes do
    let some length := assignments.lookup axis
      | throwError
          "internal error: {AxisId.description axis} has no symbolic length"
    lengths := lengths.concat length
  return lengths

/-- Multiply the symbolic lengths assigned to a group of elementary axes. -/
def axisProductExpr (assignments : List (AxisId × Expr))
    (axes : List AxisId) : MetaM Expr := do
  natProductExpr (← axisExpressions assignments axes)

/--
Flatten multiplication syntax after reducible normalization, preserving each
factor expression for definitional matching.
-/
partial def natMultiplicationFactors (value : Expr) :
    MetaM (List Expr) := do
  let value := (← instantiateMVars value).consumeMData
  if value.isAppOfArity ``Nat.mul 2 then
    let arguments := value.getAppArgs
    return (← natMultiplicationFactors arguments[0]!) ++
      (← natMultiplicationFactors arguments[1]!)
  let reduced ← withTransparency .all <| whnf value
  if reduced.isAppOfArity ``Nat.mul 2 then
    let arguments := reduced.getAppArgs
    return (← natMultiplicationFactors arguments[0]!) ++
      (← natMultiplicationFactors arguments[1]!)
  return [reduced]

/-- Remove one definitionally equal factor from a candidate multiset. -/
def eraseDefinitionalFactor? (factor : Expr) :
    List Expr → MetaM (Option (List Expr))
  | [] => pure none
  | candidate :: candidates => do
      if ← withTransparency .reducible <| isDefEq factor candidate then
        return some candidates
      let some remaining ← eraseDefinitionalFactor? factor candidates
        | return none
      return some (candidate :: remaining)

/-- Remove every requested factor by definitional equality, respecting multiplicity. -/
def eraseDefinitionalFactors? (factors : List Expr) :
    List Expr → MetaM (Option (List Expr))
  | dimensions => do
      let mut remaining := dimensions
      for factor in factors do
        let some next ← eraseDefinitionalFactor? factor remaining
          | return none
        remaining := next
      return some remaining

/--
Recover the missing factor when the tensor type already displays a product.

For example, if the physical dimension is syntactically `height * width` and
`height` is known, the remaining expression `width` is a valid axis length
even when `height = 0`. Falling back to natural-number division would demand
positivity and lose information that the dependent tensor type carries.
-/
def explicitMissingFactor? (dimension : Expr)
    (knownLengths : List Expr) : MetaM (Option Expr) := do
  let dimensionFactors ← natMultiplicationFactors dimension
  let knownFactors ←
    knownLengths.foldlM (init := []) fun factors length =>
      return factors ++ (← natMultiplicationFactors length)
  let some remaining ←
      eraseDefinitionalFactors? knownFactors dimensionFactors
    | return none
  return some (← natProductExpr remaining)

/-- Construct the type-level shape list represented by symbolic dimensions. -/
def shapeExpr (dimensions : List Expr) : MetaM Expr :=
  mkListLit (mkConst ``Nat) dimensions

/-- Construct a list of already elaborated type-level shape expressions. -/
def shapesExpr (shapes : List Expr) : MetaM Expr := do
  let shapeType ← mkAppM ``List #[mkConst ``Nat]
  mkListLit shapeType shapes

/--
Transport a generated tensor from a checker-indexed result shape to the
compact dimension list computed by the surface operation.

When the two shapes are not definitionally equal, as can happen with symbolic
dimensions, `Einops.Elab.certifyGeneratedInvariant` proves the equality. The cast keeps
the full checker certificate out of inferred types.
-/
def castTensorToCompactShape (result targetShape : Expr) :
    TermElabM Expr := do
  let resultType ← withTransparency .reducible <| whnf (← inferType result)
  unless resultType.isAppOfArity ``Tensor.Rep 3 do
    throwError
      "internal error: expected a generated tensor result, but found\
        {indentExpr resultType}"
  let arguments := resultType.getAppArgs
  -- Only results whose scalar type lies in `Type` are transported. The others keep the generated
  -- plan type exactly, so the proof terms of universe-polymorphic laws carry no presentation cast.
  unless (← getDecLevel arguments[0]!) == .zero do
    return result
  let sourceShape := arguments[1]!
  let agreement ← mkEq sourceShape targetShape
  let shapeAgreement ←
    if ← withTransparency .all <| isDefEq sourceShape targetShape then
      withTransparency .all <|
        mkExpectedTypeHint (← mkEqRefl sourceShape) agreement
    else
      certifyGeneratedInvariant
        "that the generated tensor has the displayed shape" agreement
  mkAppM ``TorchLean.Tensor.castShape #[result, shapeAgreement]

/--
Construct the public tensor type corresponding to an internal list-shaped
dimension expression.

Generated kernels work on `Rep`, while inferred declaration types keep the
`Tensor α [dims]` spelling that users and editor tooling see.
-/
def publicTensorType (scalarType shape storage : Expr) : MetaM Expr := do
  let tensorConstant := Lean.mkConst ``TorchLean.Tensor [← getDecLevel scalarType]
  return mkAppN tensorConstant #[scalarType, shape, storage]

/--
Give an inferred tensor result its compact public type without changing its
value or its checked implementation.

A known expected type wins, and the result is checked against it. Otherwise the
scalar and storage come from the generated `Rep`, and `shape` is the compact
shape expression computed by the public operation.
-/
def exposePublicTensorType (result shape : Expr)
    (expectedType? : Option Expr) : TermElabM Expr := do
  if let some expectedType := expectedType? then
    let expectedType ← instantiateMVars expectedType
    unless expectedType.isMVar do
      return ← ensureHasType (some expectedType) result
  let resultType ←
    withTransparency .reducible <| whnf (← inferType result)
  unless resultType.isAppOfArity ``Tensor.Rep 3 do
    throwError
      "internal error: expected a generated tensor result, but found\
        {indentExpr resultType}"
  let arguments := resultType.getAppArgs
  -- A scalar type outside `Type` keeps the generated `Rep` type, as the universe-polymorphic
  -- laws expect.
  unless (← getDecLevel arguments[0]!) == .zero do
    return result
  let publicType ← publicTensorType arguments[0]! shape arguments[2]!
  withTransparency .reducible <| mkExpectedTypeHint result publicType

/-- Extract a tensor shape from an informative, non-metavariable expected type. -/
def expectedTensorShape? (expectedType? : Option Expr) :
    MetaM (Option Expr) := do
  let some expectedType := expectedType? | return none
  let expectedType ← instantiateMVars expectedType
  if expectedType.isMVar then
    return none
  let expectedType ← withTransparency .reducible <| whnf expectedType
  if expectedType.isAppOfArity ``Tensor.Rep 3 then
    return some expectedType.getAppArgs[1]!
  return none

/-- Extract the value from a checker result already proved to be successful. -/
def extractSuccessfulResult (checkedResult : Expr) : MetaM Expr := do
  let checkedOption ← mkAppM ``Except.toOption #[checkedResult]
  let isSome ← mkAppM ``Option.isSome #[checkedOption]
  let hIsSome ←
    mkDecideProof (← mkEq isSome (mkConst ``Bool.true))
  mkAppM ``Option.get #[checkedOption, hIsSome]

/--
Check that the checker's result on the reflected input reduces to `.ok checked`,
seal that equality once as an auxiliary theorem, and return `result` unchanged.

Keeping the proof out of the generated value avoids repeating a potentially
large checker proposition during downstream type checking, and the proof has
no computational or dependent use.
-/
def attachCheckerAgreement (checkedResult checked result : Expr) :
    MetaM Expr := do
  let checkedResultType ← whnf (← inferType checkedResult)
  unless checkedResultType.isAppOfArity ``Except 2 do
    throwError "internal error: checker did not return an `Except` value"
  let resultArguments := checkedResultType.getAppArgs
  let successfulResult ←
    mkAppOptM ``Except.ok #[
      some resultArguments[0]!, some resultArguments[1]!, some checked]
  let agreement ← mkEq checkedResult successfulResult
  let hAgreement ← mkAppM ``Eq.refl #[checkedResult]
  let agreementType ← inferType hAgreement
  unless ← isDefEq agreementType agreement do
    throwError
      "internal error: reflected certificate disagrees with checker computation"
  let _ ← sealCertificate agreement hAgreement
  return result

/--
Eliminate a finite tensor index into one branch per heterogeneous operand,
retaining each branch's exact dependent shape.
-/
partial def buildTensorFamilyCases (scalarType storage : Expr)
    (inputShapes : List Expr) (inputTensors : List Expr)
    (tensorIndex : Expr) :
    MetaM Expr := do
  match inputShapes, inputTensors with
  | [], [] =>
      let shape ← mkAppM ``List.get #[← shapesExpr inputShapes, tensorIndex]
      let resultType :=
        mkAppN (mkConst ``Rep [← getDecLevel scalarType]) #[
          scalarType, shape, storage]
      mkAppOptM ``Fin.elim0 #[some resultType, some tensorIndex]
  | _ :: remainingShapes, inputTensor :: remainingTensors =>
      let remainingCount := remainingShapes.length
      let remainingFin := mkApp (mkConst ``Fin) (mkNatLit remainingCount)
      let remainingFamily ←
        withLocalDeclD `remainingIndex remainingFin fun remainingIndex => do
          let body ←
            buildTensorFamilyCases scalarType storage remainingShapes
              remainingTensors remainingIndex
          mkLambdaFVars #[remainingIndex] body
      let tensorCount := mkNatLit (remainingCount + 1)
      let tensorFin := mkApp (mkConst ``Fin) tensorCount
      let motive ←
        withLocalDeclD `selectedTensor tensorFin fun selectedTensor => do
          let shape ← mkAppM ``List.get #[
            ← shapesExpr inputShapes, selectedTensor]
          let tensorType :=
            mkAppN (mkConst ``Rep [← getDecLevel scalarType]) #[
              scalarType, shape, storage]
          mkLambdaFVars #[selectedTensor] tensorType
      mkAppOptM ``Fin.cases #[
        some (mkNatLit remainingCount), some motive, some inputTensor,
        some remainingFamily, some tensorIndex]
  | _, _ =>
      throwError
        "internal error: tensor expressions and checked shapes \
          have different lengths"

/-- Build the dependent `Fin n`-indexed family of heterogeneous input tensors. -/
def buildTensorFamily (scalarType storage : Expr)
    (inputShapes : List Expr) (inputTensors : List Expr) : MetaM Expr := do
  let tensorCount := mkNatLit inputShapes.length
  let tensorIndexType := mkApp (mkConst ``Fin) tensorCount
  withLocalDeclD `tensorIndex tensorIndexType fun tensorIndex => do
    let body ←
      buildTensorFamilyCases scalarType storage inputShapes inputTensors
        tensorIndex
    mkLambdaFVars #[tensorIndex] body

/-- One original tensor operand and its pending conversion to the common scalar type. -/
private structure CommonScalarInput where
  /-- Syntax this operand came from, for error positions. -/
  sourceSyntax : Syntax
  /-- The operand's tensor expression. -/
  tensor : Expr
  /-- Original scalar type, the domain of its conversion. -/
  scalarType : Expr
  /-- Original `Storage` instance, retained when no conversion is needed. -/
  storage : Expr
  /-- The operand's shape, outermost dimension first. -/
  dimensions : List Expr
  /-- Coercion from the operand's own scalar type to the common one. -/
  conversion : Expr

/--
Fallback used only by the meta-level array operations whose indices are
checked by the surrounding promotion fold.
-/
private instance : Inhabited CommonScalarInput where
  default := {
    sourceSyntax := Inhabited.default
    tensor := Inhabited.default
    scalarType := Inhabited.default
    storage := Inhabited.default
    dimensions := Inhabited.default
    conversion := Inhabited.default
  }

/-- Whether an operand is a `let`-bound local, which the consumer receives staged. -/
private def isLetVariable (tensor : Expr) : MetaM Bool := do
  match tensor with
  | .fvar fvarId => return (← fvarId.getDecl).isLet
  | _ => return false

/-- The coercion used when an operand is already at the common scalar type. -/
private def identityConversion (scalarType : Expr) : MetaM Expr :=
  withLocalDeclD `value scalarType fun value =>
    mkLambdaFVars #[value] value

/-- Project one side's coercion out of a `Promotion` instance. -/
private def promotionProjection (projection : Name) (leftType rightType
    outputType promotion : Expr) : Expr :=
  mkAppN (mkConst projection) #[
    leftType, rightType, outputType, promotion]

/--
Elaborate a nonempty tensor family and convert mixed element types to one common
type before operation-specific checking.

Promotion folds the element types from left to right to find the common type
`γ`. Each operand of type `α` then converts once, by the left conversion of
`ElementPromotion α γ γ`, which is the conversion binary arithmetic applies.
Homogeneous families keep their original tensors and storage.
-/
def elaborateCommonScalarFamily (operation member : String)
    (tensorSyntax : Array (TSyntax `term)) :
    TermElabM (List Expr × List (List Expr) × Expr × Expr × List Bool) := do
  let mut inputs : Array CommonScalarInput := #[]
  for tensorIndex in [:tensorSyntax.size] do
    let currentSyntax := tensorSyntax[tensorIndex]!
    let (tensor, scalarType, storage, dimensions) ←
      elaborateTensor currentSyntax
    inputs := inputs.push {
      sourceSyntax := currentSyntax
      tensor
      scalarType
      storage
      dimensions
      conversion := ← identityConversion scalarType
    }
  let some first := inputs[0]?
    | throwError "internal error: `{operation}` received no tensors"
  let mut homogeneous := true
  for input in inputs[1:] do
    unless ← isDefEq first.scalarType input.scalarType do
      homogeneous := false
    if homogeneous then
      unless ← isDefEq first.storage input.storage do
        throwErrorAt input.sourceSyntax
          "{operation} {member} uses incompatible tensor storage"
  if homogeneous then
    pure
      (inputs.toList.map (·.tensor), inputs.toList.map (·.dimensions),
        first.scalarType, first.storage,
        ← inputs.toList.mapM fun input => isLetVariable input.tensor)
  else
    let mut commonType := first.scalarType
    for inputIndex in [1:inputs.size] do
      let input := inputs[inputIndex]!
      let outputTypeMVar ← mkFreshExprMVar (mkSort (.succ .zero))
      let promotionType ←
        mkAppM ``ElementPromotion #[
          commonType, input.scalarType, outputTypeMVar]
      try
        discard <| withRef input.sourceSyntax <| synthInstance promotionType
      catch _ =>
        throwErrorAt input.sourceSyntax
          "{operation} {member} {inputIndex} with element type\
            {indentExpr input.scalarType}\n\
            cannot be promoted with the preceding common element type\
            {indentExpr commonType}\n\
            Register an `ElementPromotion` instance for this ordered pair, or \
            convert the tensor explicitly with `Tensor.cast`."
      let outputType ← instantiateMVars outputTypeMVar
      commonType ← withTransparency .reducible <| whnf outputType
    -- Converting through an intermediate common type would round twice, so each
    -- operand converts once from its own type to the final one.
    for inputIndex in [:inputs.size] do
      let input := inputs[inputIndex]!
      if ← withTransparency .reducible <| isDefEq input.scalarType commonType then
        continue
      let outputTypeMVar ← mkFreshExprMVar (mkSort (.succ .zero))
      let promotionType ←
        mkAppM ``ElementPromotion #[
          input.scalarType, commonType, outputTypeMVar]
      let promotion ←
        try
          withRef input.sourceSyntax <| synthInstance promotionType
        catch _ =>
          throwErrorAt input.sourceSyntax
            "{operation} {member} {inputIndex} with element type\
              {indentExpr input.scalarType}\n\
              has no `ElementPromotion` to the common element type\
              {indentExpr commonType}\n\
              Register an `ElementPromotion` instance for this ordered pair, or \
              convert the tensor explicitly with `Tensor.cast`."
      let outputType ← instantiateMVars outputTypeMVar
      unless ← isDefEq outputType commonType do
        throwErrorAt input.sourceSyntax
          "{operation} {member} {inputIndex} with element type\
            {indentExpr input.scalarType}\n\
            promotes with the common element type{indentExpr commonType}\n\
            to{indentExpr outputType}\n\
            instead of to the common element type itself."
      inputs := inputs.set! inputIndex {
        input with
        conversion :=
          promotionProjection ``ElementPromotion.left input.scalarType
            commonType commonType promotion
      }
    let storageType ← mkAppM ``Storage #[commonType]
    let commonStorage ←
      try
        synthInstance storageType
      catch _ =>
        throwError
          "{operation} promoted its inputs to element type\
            {indentExpr commonType}\n\
            but no `Storage` instance is available for that type"
    let commonLevel ← getDecLevel commonType
    let mut convertedTensors : List Expr := []
    let mut stagedConversions : List Bool := []
    for input in inputs do
      let identity ← identityConversion input.scalarType
      let conversionIsIdentity ←
        withTransparency .all <| isDefEq input.conversion identity
      let scalarUnchanged ←
        withTransparency .reducible <| isDefEq input.scalarType commonType
      let storageUnchanged ←
        withTransparency .reducible <| isDefEq input.storage commonStorage
      if conversionIsIdentity && scalarUnchanged && storageUnchanged then
        convertedTensors := convertedTensors.concat input.tensor
        stagedConversions := stagedConversions.concat (← isLetVariable input.tensor)
      else
        let sourceLevel ← getDecLevel input.scalarType
        let shape ← shapeExpr input.dimensions
        let converted :=
          mkAppN (mkConst ``TorchLean.Tensor.map [sourceLevel, commonLevel]) #[
            input.scalarType, input.storage, commonType, commonStorage, shape,
            input.conversion, input.tensor]
        convertedTensors := convertedTensors.concat converted
        stagedConversions := stagedConversions.concat true
    pure
      (convertedTensors, inputs.toList.map (·.dimensions), commonType,
        commonStorage, stagedConversions)

/--
Elaborate a common-scalar tensor family and share every required conversion
around the generated consumer.

The continuation receives local tensor variables in place of repeated conversion expressions. Each
conversion, and each `let`-bound operand, goes through `Native.stage`, whose no-inline boundary
keeps it out of every downstream scalar read. By `Native.stage_eq`, staging leaves the value
unchanged.
-/
def withCommonScalarFamily (operation member : String)
    (tensorSyntax : Array (TSyntax `term))
    (body :
      List Expr → List (List Expr) → Expr → Expr → TermElabM Expr) :
    TermElabM Expr := do
  let (tensors, shapes, scalarType, storage, stagedConversions) ←
    elaborateCommonScalarFamily operation member tensorSyntax
  unless stagedConversions.any id do
    return ← body tensors shapes scalarType storage
  let rec
    /-- Stage converted tensors in source order before entering the consumer. -/
    visit (remaining : List Expr) (remainingStages : List Bool) (index : Nat)
        (bound : List Expr) : TermElabM Expr := do
      match remaining, remainingStages with
      | [], [] => body bound shapes scalarType storage
      | tensor :: remaining, shouldStage :: remainingStages =>
          unless shouldStage do
            return ←
              visit remaining remainingStages (index + 1)
                (bound.concat tensor)
          let name := Name.mkSimple s!"converted{member.capitalize}{index}"
          withLocalDeclD name (← inferType tensor) fun localTensor => do
            let result ←
              visit remaining remainingStages (index + 1)
                (bound.concat localTensor)
            let consumer ← mkLambdaFVars #[localTensor] result
            mkAppM ``Native.stage #[tensor, consumer]
      | _, _ =>
          throwError
            "internal error: `{operation}` promotion staging metadata is \
              misaligned"
  visit tensors stagedConversions 0 []

end

end TorchLean.Einops.Elab
