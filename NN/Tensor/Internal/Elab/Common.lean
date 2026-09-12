/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Check.Einsum
public import NN.Tensor.Internal.Check.Pack
public import NN.Tensor.Internal.Representation.Promotion
public import Mathlib.Algebra.Order.Field.Basic
meta import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Positivity.Finset
import Mathlib.Tactic.Ring.RingNF
public meta import NN.Spec.Core.Tensor.Core
public meta import NN.Tensor.Internal.Check.Einsum
public meta import NN.Tensor.Internal.Check.Normalize
public meta import NN.Tensor.Internal.Check.Pack
public meta import NN.Tensor.Internal.Check.Transform
public meta import NN.Tensor.Internal.Elab.Native.Tensor
public meta import NN.Tensor.Internal.Elab.Syntax
public meta import Lean.Elab.Tactic -- shake: keep
public meta import Lean.Elab.Tactic.Omega -- shake: keep
public meta import Lean.Meta.LitValues -- shake: keep
public meta import Aesop -- shake: keep
meta import Mathlib.Tactic.Positivity -- shake: keep
meta import Mathlib.Tactic.Ring -- shake: keep
public meta import NN.Tensor.Internal.Syntax.Span -- shake: keep
public import NN.Tensor.Internal.Check.ParseShape -- shake: keep
public import NN.Tensor.Internal.Semantics.Transform -- shake: keep

/-!
# Shared elaboration infrastructure

This module contains the common metaprogramming machinery used by every
operation family. Literal term grammar lives in `TorchLean.Tensor.Internal.Elab.Syntax`.
These internal utilities inspect dependent tensor types, preserve symbolic
natural-number dimensions, construct heterogeneous tensor families, and emit
proof-bearing checker certificates.

The helpers live in `TorchLean.Tensor.Internal.Elab.Impl` because they are implementation
infrastructure rather than user API. Operation modules share them directly;
there is no alternate tensor representation or compatibility elaborator.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

deriving instance Lean.ToExpr for Syntax.Span
deriving instance Lean.ToExpr for Syntax.Located
deriving instance Lean.ToExpr for Syntax.Axis
deriving instance Lean.ToExpr for Syntax.CompositeAxis
deriving instance Lean.ToExpr for Syntax.Expression
deriving instance Lean.ToExpr for Syntax.TransformPattern
deriving instance Lean.ToExpr for Syntax.EinsumPattern
deriving instance Lean.ToExpr for Syntax.PackPattern
deriving instance Lean.ToExpr for Check.TransformKind
deriving instance Lean.ToExpr for Check.AxisId
deriving instance Lean.ToExpr for Check.EinsumAxis
deriving instance Lean.ToExpr for Check.NormalizedTransform

namespace Impl

/-- Render a singular or plural label for a diagnostic source range. -/
def pluralColumns (length : Nat) : String :=
  if length = 1 then "column" else "columns"

/-- Draw a caret marker beneath the source range selected by a diagnostic. -/
def patternMarker (span : Syntax.Span) : String :=
  String.ofList <|
    List.replicate span.offset ' ' ++
      List.replicate (max 1 span.length) '^'

/-- Raise an elaboration error containing the pattern and its precise source range. -/
def throwPatternDiagnostic {α : Type}
    (operation phase source : String) (message : String)
    (span : Syntax.Span) : TermElabM α :=
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
    concrete := concrete.concat value
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
partial def concreteListLength? (value : Expr) :
    MetaM (Option Nat) := do
  let value ← withTransparency .all <| whnf value
  if value.isAppOfArity ``List.nil 1 then
    return some 0
  if value.isAppOfArity ``List.cons 3 then
    let some tailLength ← concreteListLength? value.getAppArgs[2]!
      | return none
    return some (tailLength + 1)
  return none

/-- Evaluate an integer expression when kernel reduction exposes a literal. -/
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
partial def concreteIntList? (value : Expr) :
    MetaM (Option (List Int)) := do
  let value ← withTransparency .all <| whnf value
  if value.isAppOfArity ``List.nil 1 then
    return some []
  if value.isAppOfArity ``List.cons 3 then
    let arguments := value.getAppArgs
    let some head ← concreteIntExpression? arguments[1]! | return none
    let some tail ← concreteIntList? arguments[2]! | return none
    return some (head :: tail)
  return none

/-- Read concrete unpack metadata as a statically written list of integer lists. -/
partial def concreteRequestedShapes? (value : Expr) :
    MetaM (Option Check.RequestedShapes) := do
  let value ← withTransparency .all <| whnf value
  if value.isAppOfArity ``List.nil 1 then
    return some []
  if value.isAppOfArity ``List.cons 3 then
    let arguments := value.getAppArgs
    let some head ← concreteIntList? arguments[1]! | return none
    let some tail ← concreteRequestedShapes? arguments[2]! | return none
    return some (head :: tail)
  return none

/--
Recover the element type, static shape, and storage selected by
`TorchLean.Tensor`.

The canonical public tensor abbreviation is unfolded before inspecting its
packed representation. Arbitrary coordinate functions are not accepted as
tensors.
-/
def tensorTypeInfo (tensorSyntax : Syntax) (tensorType : Expr) :
    TermElabM (Expr × Expr × Expr) := do
  let tensorType ← withTransparency .reducible <| whnf tensorType
  unless tensorType.isAppOfArity ``TorchLean.Tensor.Internal.Rep 3 do
    throwErrorAt tensorSyntax
      "expected a `TorchLean.Tensor α shape`, but the term has type\
        {indentExpr tensorType}"
  let arguments := tensorType.getAppArgs
  return (arguments[1]!, arguments[0]!, arguments[2]!)

/--
Elaborate one tensor and expose its static list structure while preserving
symbolic dimension expressions.

A nonempty list of equal-shaped tensors is accepted as a convenient stacked
tensor and contributes its statically known list length as a leading axis.
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
        mkAppM ``TorchLean.Tensor.Internal.Rep.stackList #[input]
      pure (stackedTensor, componentType, [mkNatLit componentCount])
    else
      pure (input, inputType, [])
  let (shape, scalarType, storage) ← tensorTypeInfo tensorSyntax tensorType
  let some dimensions ← staticListElements? shape
    | throwErrorAt tensorSyntax
        "the tensor shape must have a statically known list structure; its \
          dimensions may be symbolic natural-number expressions"
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
          pure (axis.getId.toString, length)
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
    MetaM (Option Check.SupplementaryLengths) := do
  let mut concrete : Check.SupplementaryLengths := []
  for (name, length) in supplementary do
    let length ← withTransparency .all <| whnf length
    let some value ← getNatValue? length | return none
    concrete := concrete.concat (name, value)
  return some concrete

/--
Apply one explicit structure-constructor argument after checking its dependent
field type.
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
Build a proof-valued structure using explicit parameters, any fields that are
already proved, and kernel-reduced decisions for the remaining fields.
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
    -- Keep the proof typed by the unreduced field proposition. Downstream
    -- tactics can then treat reflected certificates abstractly without
    -- unfolding parser and normalization definitions to recover its type.
    let certificate ← withTransparency .all <| mkDecideProof proposition
    let certificate ← sealCertificate proposition certificate
    (constructor, constructorType) ←
      applyConstructorArgument constructor constructorType certificate
  return constructor

/--
Run a focused tactic on a generated proposition and seal the resulting kernel
certificate in an auxiliary theorem.

Callers choose the smallest tactic vocabulary appropriate for their invariant.
Failure is reported as a missing user hypothesis rather than hidden behind an
unchecked cast or axiom.
-/
def certifyWithTactic (description : String) (proposition : Expr)
    (tactic : Syntax) : TermElabM Expr := do
  let invariantGoal ← mkFreshExprSyntheticOpaqueMVar proposition
  let remainingGoals ←
    Lean.Elab.Tactic.run invariantGoal.mvarId! do
      discard <| observing? do
        let result ←
          Mathlib.Tactic.withResetServerInfo <|
            Lean.Elab.Tactic.evalTactic tactic
        if result.result?.isNone || result.msgs.hasErrors then
          throwError "generated certificate tactic failed"
  unless remainingGoals.isEmpty do
    let target ← instantiateMVars (← remainingGoals[0]!.getType)
    throwError
      "could not prove {description}:{indentExpr target}\n\
        Add the shape equality, broadcasting, positivity, or divisibility \
        hypothesis needed by this operation."
  let certificate ← instantiateMVars invariantGoal
  if certificate.hasMVar then
    throwError
      "internal error: the proof of {description} still contains metavariables"
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
        Check.ParseShape.AxisMatches,
        Check.packFixedRank,
        Check.packStarShape,
        Check.CheckedPack.starShape,
        Check.CheckedPack.metadata,
        Check.CheckedPack.segmentLengths,
        Check.CheckedPack.packedAxisLength,
        Check.CheckedPack.output,
        Check.TransformPlan.allAxes,
        Check.TransformPlan.groupLength,
        Check.TransformPlan.inferredInput,
        Check.TransformPlan.inferredOutput,
        Check.TransformPlan.literalAxesAgree,
        Check.TransformPlan.normalized,
        Check.CheckedTransform.reducedAxes,
        Check.CheckedTransform.reductionShape,
        Check.CheckedTransform.reductionFiberSize,
        Check.NormalizedTransform.inputAxes,
        Check.NormalizedTransform.outputAxes,
        Check.PartialAxisLengths.seed,
        Check.PartialAxisLengths.set,
        Check.SupplementaryLengths.lookup?,
        Syntax.CompositeAxis.semanticAxes,
        Syntax.Expression.ellipsisCount,
        Syntax.Expression.flatAxes,
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
      -- Statically written tensor families quantify over `Fin n`.
      -- Enumerating that index exposes each symbolic shape equality.
      | (intro component; fin_cases component <;> simp_all)
      | ring
      | aesop)
  certifyWithTactic description proposition tactic

/-- Human-readable description of a logical transformation axis. -/
def axisDescription : Check.AxisId → String
  | .named name => s!"axis '{name}'"
  | .anonymous value _ => s!"anonymous axis {value}"
  | .ellipsis index => s!"ellipsis axis {index}"

/-- Look up the symbolic length currently assigned to a logical axis. -/
def axisExpression? (assignments : List (Check.AxisId × Expr))
    (axis : Check.AxisId) : Option Expr :=
  (assignments.find? fun assignment => assignment.1 == axis).map Prod.snd

/-- Append a symbolic axis assignment unless that axis was assigned earlier. -/
def appendAxisExpression (assignments : List (Check.AxisId × Expr))
    (axis : Check.AxisId) (length : Expr) :
    List (Check.AxisId × Expr) :=
  if (axisExpression? assignments axis).isSome then
    assignments
  else
    assignments.concat (axis, length)

/-- Construct a balanced-by-source-order natural-number product expression. -/
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

/-- Resolve a list of logical axes to their symbolic length expressions. -/
def axisExpressions (assignments : List (Check.AxisId × Expr))
    (axes : List Check.AxisId) : MetaM (List Expr) := do
  let mut lengths : List Expr := []
  for axis in axes do
    let some length := axisExpression? assignments axis
      | throwError
          "internal error: {axisDescription axis} has no symbolic length"
    lengths := lengths.concat length
  return lengths

/-- Multiply the symbolic lengths assigned to a group of elementary axes. -/
def axisProductExpr (assignments : List (Check.AxisId × Expr))
    (axes : List Check.AxisId) : MetaM Expr := do
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

/--
Flatten addition syntax after reducible normalization, preserving each
summand expression for definitional matching.
-/
partial def natAdditionSummands (value : Expr) :
    MetaM (List Expr) := do
  let value := (← instantiateMVars value).consumeMData
  if value.isAppOfArity ``Nat.add 2 then
    let arguments := value.getAppArgs
    return (← natAdditionSummands arguments[0]!) ++
      (← natAdditionSummands arguments[1]!)
  let reduced ← withTransparency .all <| whnf value
  if reduced.isAppOfArity ``Nat.add 2 then
    let arguments := reduced.getAppArgs
    return (← natAdditionSummands arguments[0]!) ++
      (← natAdditionSummands arguments[1]!)
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
even when `height = 0`. Falling back immediately to natural-number division
would unnecessarily demand positivity and would lose information deliberately
present in the dependent tensor type.
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

/--
Recover the part of a packed length not occupied by known segments.

Addition is treated modulo association and order, but only definitionally
equal summands are removed. This preserves expressions already present in the
tensor type without asking the kernel to choose a subtraction normal form.
-/
def explicitResidual? (total : Expr) (known : List Expr) :
    MetaM (Option Expr) := do
  let totalSummands ← natAdditionSummands total
  let knownSummands ←
    known.foldlM (init := []) fun summands length =>
      return summands ++ (← natAdditionSummands length)
  let some remaining ←
      eraseDefinitionalFactors? knownSummands totalSummands
    | return none
  return some (← natSumExpr remaining)

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

The equality is definitional for generated plans, but the explicit transport
prevents inferred declaration types from retaining the complete checker
certificate.
-/
def castTensorToCompactShape (result targetShape : Expr) :
    TermElabM Expr := do
  let resultType ← withTransparency .reducible <| whnf (← inferType result)
  unless resultType.isAppOfArity ``TorchLean.Tensor.Internal.Rep 3 do
    throwError
      "internal error: expected a generated tensor result, but found\
        {indentExpr resultType}"
  let arguments := resultType.getAppArgs
  -- This transport only serves the public `Tensor` API, whose scalar lives
  -- in `Type`. Universe-polymorphic internal laws retain the exact generated
  -- plan type so their proof terms are not wrapped in a presentation cast.
  unless (← getDecLevel arguments[0]!) == .zero do
    return result
  let sourceShape := arguments[1]!
  let shapeAgreement ←
    withTransparency .all <|
      mkExpectedTypeHint (← mkEqRefl sourceShape)
        (← mkEq sourceShape targetShape)
  mkAppM ``TorchLean.Tensor.Internal.Rep.castShape #[
    shapeAgreement, result]

/--
Construct the public tensor type corresponding to an internal list-shaped
dimension expression.

Generated kernels use `Rep` directly, but inferred declaration types should
retain the ordinary `Tensor α [dims]` spelling seen by users and editor
tooling.
-/
def publicTensorType (scalarType shape storage : Expr) : MetaM Expr := do
  let publicShape ← mkAppM ``Spec.Shape.ofList #[shape]
  let tensorConstant := Lean.mkConst ``TorchLean.Tensor
  return mkAppN tensorConstant #[scalarType, publicShape, storage]

/--
Give an inferred tensor result its compact public type without changing its
value or its checked implementation.

An explicit expected type remains authoritative. Otherwise the scalar and
storage are recovered from the generated `Rep`, while `shape` is the compact
shape expression computed by the public operation.
-/
def exposePublicTensorType (result shape : Expr)
    (expectedType? : Option Expr) : TermElabM Expr := do
  if expectedType?.isSome then
    return ← ensureHasType expectedType? result
  let resultType ←
    withTransparency .reducible <| whnf (← inferType result)
  unless resultType.isAppOfArity ``TorchLean.Tensor.Internal.Rep 3 do
    throwError
      "internal error: expected a generated tensor result, but found\
        {indentExpr resultType}"
  let arguments := resultType.getAppArgs
  -- The public `Tensor` alias currently lives in `Type`, while the internal
  -- representation also supports proof scalars in arbitrary universes.
  -- Preserve that more general internal type in universe-polymorphic laws.
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
  if expectedType.isAppOfArity ``TorchLean.Tensor.Internal.Rep 3 then
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
Register evidence that a reflected certificate is exactly the ordinary
checker's result.

The agreement is checked once as a named theorem. It is deliberately not
embedded as a nondependent `let` in every generated value: doing so repeats a
potentially large checker proposition during downstream type checking even
though the proof has no computational or dependent use.
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

/-- One operand of a tensor family after promotion, with the coercion it needs applied. -/
private structure CommonScalarInput where
  /-- Syntax this operand came from, for error positions. -/
  sourceSyntax : Syntax
  /-- The operand's tensor expression. -/
  tensor : Expr
  /-- Scalar type the operand is being read at, after promotion. -/
  scalarType : Expr
  /-- `Storage` instance for that scalar type. -/
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

/-- The coercion used when an operand is already at the common scalar type. -/
private def identityConversion (scalarType : Expr) : MetaM Expr :=
  withLocalDeclD `value scalarType fun value =>
    mkLambdaFVars #[value] value

/-- Compose two coercions, for operands promoted in more than one step. -/
private def composeConversion (outer inner domain : Expr) : MetaM Expr :=
  withLocalDeclD `value domain fun value => do
    let converted := mkApp inner value
    mkLambdaFVars #[value] (mkApp outer converted)

/-- Project one side's coercion out of a `Promotion` instance. -/
private def promotionProjection (projection : Name) (leftType rightType
    outputType promotion : Expr) : Expr :=
  mkAppN (mkConst projection) #[
    leftType, rightType, outputType, promotion]

/--
Elaborate a nonempty tensor family and convert mixed element types to one common
type before operation-specific checking.

Promotion folds from left to right. Each input conversion is composed during
that fold and materialized at most once after the final common type is known.
Homogeneous families retain their original tensors and storage exactly.
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
        first.scalarType, first.storage, inputs.toList.map fun _ => false)
  else
    let mut commonType := first.scalarType
    for inputIndex in [1:inputs.size] do
      let input := inputs[inputIndex]!
      let outputTypeMVar ← mkFreshExprMVar (mkSort (.succ .zero))
      let promotionType ←
        mkAppM ``ElementPromotion #[
          commonType, input.scalarType, outputTypeMVar]
      let promotion ←
        try
          withRef input.sourceSyntax <| synthInstance promotionType
        catch _ =>
          throwErrorAt input.sourceSyntax
            "{operation} {member} {inputIndex} with element type\
              {indentExpr input.scalarType}\n\
              cannot be promoted with the preceding common element type\
              {indentExpr commonType}\n\
              Register an `ElementPromotion` instance for this ordered pair, or \
              convert the tensor explicitly with `Tensor.cast`."
      let outputType ← instantiateMVars outputTypeMVar
      let outputType ← withTransparency .reducible <| whnf outputType
      let leftConversion :=
        promotionProjection ``ElementPromotion.left commonType input.scalarType
          outputType promotion
      let rightConversion :=
        promotionProjection ``ElementPromotion.right commonType input.scalarType
          outputType promotion
      for previousIndex in [:inputIndex] do
        let previous := inputs[previousIndex]!
        inputs := inputs.set! previousIndex {
          previous with
          conversion := ←
            composeConversion leftConversion previous.conversion
              previous.scalarType
        }
      inputs := inputs.set! inputIndex {
        input with
        conversion := rightConversion
      }
      commonType := outputType
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
        stagedConversions := stagedConversions.concat false
      else
        let sourceLevel ← getDecLevel input.scalarType
        let shape ← shapeExpr input.dimensions
        let converted :=
          mkAppN (mkConst ``Rep.map [sourceLevel, commonLevel]) #[
            input.scalarType, input.storage, commonType, commonStorage, shape,
            input.conversion, input.tensor]
        convertedTensors := convertedTensors.concat converted
        stagedConversions := stagedConversions.concat true
    pure
      (convertedTensors, inputs.toList.map (·.dimensions), commonType,
        commonStorage, stagedConversions)

/--
Elaborate a common-scalar tensor family and share every required conversion
around
the generated consumer.

The continuation sees local tensor variables rather than repeated conversion
expressions. Each conversion is passed through `nativeStage`, whose no-inline
boundary ensures native execution materializes it once before entering the
consumer rather than sinking it into every downstream scalar read.
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
            mkAppM ``nativeStage #[tensor, consumer]
      | _, _ =>
          throwError
            "internal error: `{operation}` promotion staging metadata is \
              misaligned"
  visit tensors stagedConversions 0 []

end Impl

end TorchLean.Tensor.Internal.Elab
