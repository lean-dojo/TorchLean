/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Einops.Elab.Pull
public meta import TorchLean.Tensor.Einops.Elab.Reduce
public meta import TorchLean.Tensor.Einops.Elab.Transform.View
public import TorchLean.Tensor.Einops.Elab.Pull
public import TorchLean.Tensor.Einops.Elab.Reduce
public import TorchLean.Tensor.Einops.Elab.Syntax
public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Einops.Elab.Transform.View

/-!
# Elaboration of transformations and parse-shape expressions

The elaborators for `rearrange`, `expand`, `reduce` and `parse_shape` each take one of two
paths. With concrete dimensions the executable checker runs and reports source-located
diagnostics. Symbolic dimensions produce the same checked transformation type, with every shape
equation proved in the caller's Lean context.

Built-in reductions fold each fiber from left to right in row-major order. `sum`, `prod`,
`min`, and `max` fold with the step and seed of the `ReduceOp` of `Tensor.reduce`, read from its
`ReduceOp.Scalar` instance, so `min` and `max` are IEEE 754-2019 `minimum` and `maximum` for every
scalar with a `Context`. A scalar without that instance (such as `Nat` for `min` and `max`, or a
type above `Type`) uses its own `+`, `*`, `Min`, or `Max`. `mean` divides the `sum` fold by the
fiber size: by the cast size at a division ring of characteristic zero, and with
`TorchLean.Context.divNat`, as `Tensor.mean` does, at every other scalar with a `Context`. A scalar
with neither is rejected. `Lowering.reduceFold_mean` rewrites the first form to `Reduction.mean`.
`Lowering.reduceFold_ordered_correct` rewrites the second to an ordered reduction, which
`TorchLean.Tensor.Einops.Lowering.Reduce.Mean` proves equal to `Reduction.mean` under
`LawfulContext` and to `Tensor.mean` when every axis is reduced.
A fold without a seed, such as the library's `min` and `max`, starts from the first entry of each
fiber, so it and `mean` require nonempty fibers. Custom reducers consume a `Multiset` and may
return a different scalar type, so they expose no accumulation order.
-/

public meta section

namespace TorchLean.Einops.Elab

open TorchLean.Tensor
open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta
open AxisId Native

/--
Parse a transformation pattern and report failures against the original
source span.
-/
private def parsedTransformPattern (operation source : String) :
    TermElabM TransformPattern :=
  match TransformPattern.parse source with
  | .error diagnostic =>
      throwPatternDiagnostic operation "pattern" source
        diagnostic.message diagnostic.span
  | .ok pattern => pure pattern

/--
Reject duplicate or unused supplementary axis lengths before solving symbolic
shape equations.
-/
private def validateSymbolicSupplementary (operation source : String)
    (pattern : TransformPattern)
    (normalized : NormalForm)
    (supplementary : List (String × Expr)) : TermElabM Unit := do
  let names := supplementary.map Prod.fst
  unless names.Nodup do
    throwPatternDiagnostic operation "shape" source
      "an axis length was supplied more than once" pattern.span
  let usedNames :=
    (normalized.inputAxes ++ normalized.outputAxes).filterMap fun axis =>
      match axis with
      | .named name => some name
      | _ => none
  for name in names do
    unless name ∈ usedNames do
      throwPatternDiagnostic operation "shape" source
        s!"supplied axis length '{name}' is not used by the transformation"
        pattern.span

/--
Seed symbolic axis lengths from the numeric axes of the pattern and the user's `with`
assignments.
-/
private def seedSymbolicAssignments
    (normalized : NormalForm)
    (supplementary : List (String × Expr)) :
    List (AxisId × Expr) :=
  let literalAssignments :=
    (normalized.inputAxes ++ normalized.outputAxes).foldl
      (init := []) fun assignments axis =>
        match axis with
        | .anonymous value _ =>
            appendAxisExpression assignments axis (mkNatLit value)
        | _ => assignments
  supplementary.foldl (init := literalAssignments) fun assignments item =>
    appendAxisExpression assignments (.named item.1) item.2

/--
Recover the one unknown factor of a grouped physical dimension. When the dimension is written as a
product that contains the known lengths, `Einops.Elab.explicitMissingFactor?` reads off the
remaining factor. Otherwise the factor is the quotient by the known product, after proofs that this
product is positive and divides the dimension.
-/
private def inferredAxisExpression (dimension : Expr)
    (knownLengths : List Expr) : TermElabM Expr := do
  if knownLengths.isEmpty then
    return dimension
  if let some factor ← explicitMissingFactor? dimension knownLengths then
    return factor
  let knownProduct ← natProductExpr knownLengths
  let positive ← mkAppM ``LT.lt #[mkNatLit 0, knownProduct]
  let _ ←
    certifyGeneratedInvariant "that the known grouped-axis product is positive"
      positive
  let remainder ← mkAppM ``Nat.mod #[dimension, knownProduct]
  let divisible ← mkEq remainder (mkNatLit 0)
  let _ ←
    certifyGeneratedInvariant "that the physical dimension is exactly divisible"
      divisible
  mkAppM ``Nat.div #[dimension, knownProduct]

/--
Solve the finite system of shape equations induced by the pattern.

Each constraint equates one physical dimension with the product of a logical
axis group. Repeated passes matter: an expected output type can determine one
axis, which can then make a grouped input equation uniquely solvable. The
algorithm assigns an axis only when every other factor in one equation is
known, so it never invents a decomposition for an underdetermined product.
-/
private def resolveSymbolicConstraints
    (initial : List (AxisId × Expr))
    (constraints : List (List AxisId × Expr)) :
    TermElabM (List (AxisId × Expr)) := do
  let mut assignments := initial
  let mut changed := true
  while changed do
    changed := false
    for (group, dimension) in constraints do
      let unknownAxes :=
        group.filter fun axis => (assignments.lookup axis).isNone
      match unknownAxes with
      | [unknown] =>
          let knownAxes := group.filter fun axis => axis != unknown
          let knownLengths ← axisExpressions assignments knownAxes
          let inferred? : Option Expr ←
            try
              pure (some (← inferredAxisExpression dimension knownLengths))
            catch _ =>
              pure none
          if let some inferred := inferred? then
            assignments := appendAxisExpression assignments unknown inferred
            changed := true
      | _ => pure ()
  return assignments

/--
Build the total axis-length function stored by a transformation plan. Unresolved axes map to
zero, and the caller rejects them before this point.
-/
private def axisLengthExpr (assignments : List (AxisId × Expr)) :
    MetaM Expr := do
  let mut partialAxisLengths ←
    mkAppM ``PartialAxisLengths.seed #[
      Lean.toExpr ([] : SupplementaryLengths)]
  for (axis, length) in assignments do
    partialAxisLengths ←
      mkAppM ``PartialAxisLengths.set #[
        partialAxisLengths, Lean.toExpr axis, length]
  withLocalDeclD `axis (mkConst ``AxisId) fun axis => do
    let assignedLength ← mkAppM' partialAxisLengths #[axis]
    let totalLength ←
      mkAppM ``Option.getD #[assignedLength, mkNatLit 0]
    mkLambdaFVars #[axis] totalLength

/--
Reuse rank-only normalization data with the caller's symbolic input shape and
rebuild its structural validity certificate.
-/
private def reifySymbolicNormalization
    (normalization : CheckedNormalization) (inputShape : Expr) :
    MetaM (Expr × Expr) := do
  let normalized := normalization.value
  let normalizedExpr ←
    mkAppM ``NormalForm.mk #[
      Lean.toExpr normalized.kind,
      Lean.toExpr normalized.source,
      inputShape,
      Lean.toExpr normalized.ellipsisRank,
      Lean.toExpr normalized.ellipsisAxes,
      Lean.toExpr normalized.inputGroups,
      Lean.toExpr normalized.outputGroups]
  let normalizedValid ←
    buildCertificate ``NormalForm.Valid.mk
      #[normalizedExpr] #[] 13
  let checkedNormalization ←
    mkAppM ``CheckedNormalization.mk #[
      normalizedExpr, normalizedValid]
  return (checkedNormalization, normalizedValid)

/--
Construct a checked transformation whose dimensions may contain arbitrary
natural-number expressions.

Normalization depends only on the statically visible rank, so it is first run
against a dummy shape of that rank. The resulting structural certificate is
then reified with the actual symbolic input shape. Input dimensions, supplied
axis lengths, and a statically visible expected output shape form a finite
system of product equations. Every equation used by the final plan is proved
in the caller's local context.
-/
private def symbolicCheckedTransformExpr (operation source : String)
    (kind : TransformKind) (pattern : TransformPattern)
    (dimensions : List Expr) (supplementary : List (String × Expr))
    (expectedType? : Option Expr) : TermElabM (Expr × Expr) := do
  let dummyShape := List.replicate dimensions.length 1
  let normalization ←
    match TransformPattern.normalize kind pattern dummyShape with
    | .error diagnostic =>
        throwPatternDiagnostic operation "shape" source
          diagnostic.message diagnostic.span
    | .ok normalization => pure normalization
  let normalized := normalization.value
  validateSymbolicSupplementary operation source pattern normalized supplementary
  let inputShape ← shapeExpr dimensions
  let mut constraints := normalized.inputGroups.zip dimensions
  let mut expectedOutputShape? : Option Expr := none
  if let some expectedShape ← expectedTensorShape? expectedType? then
    if let some expectedDimensions ← staticListElements? expectedShape then
      if expectedDimensions.length = normalized.outputGroups.length then
        constraints := constraints ++
          normalized.outputGroups.zip expectedDimensions
        expectedOutputShape? := some expectedShape
  let assignments ←
    resolveSymbolicConstraints
      (seedSymbolicAssignments normalized supplementary) constraints
  let unresolved :=
    (normalized.inputAxes ++ normalized.outputAxes).eraseDups.filter fun axis =>
      (assignments.lookup axis).isNone
  unless unresolved.isEmpty do
    let axes := String.intercalate ", " (unresolved.map description)
    throwPatternDiagnostic operation "shape" source
      s!"could not determine {axes}; supply an axis length or an expected \
        output tensor type that determines it"
      pattern.span
  let axisLength ← axisLengthExpr assignments
  let inferredOutputDimensions ←
    normalized.outputGroups.mapM fun group => do
      axisProductExpr assignments group
  let inferredOutputShape ← shapeExpr inferredOutputDimensions
  let outputShape := expectedOutputShape?.getD inferredOutputShape
  let (checkedNormalization, normalizedValid) ←
    reifySymbolicNormalization normalization inputShape
  let plan ←
    mkAppM ``TransformPlan.mk #[
      checkedNormalization, axisLength, outputShape]
  let literalAxes ←
    mkAppM ``TransformPlan.literalAxesAgree #[plan]
  let hLiteralAxes ←
    certifyGeneratedInvariant "that anonymous axes retain their literal lengths"
      (← mkEq literalAxes (mkConst ``Bool.true))
  let inferredInputShape ←
    mkAppM ``TransformPlan.inferredInput #[plan]
  let hInputShape ←
    certifyGeneratedInvariant
      "that the logical input axes reconstruct the tensor shape"
      (← mkEq inferredInputShape inputShape)
  let planInferredOutputShape ←
    mkAppM ``TransformPlan.inferredOutput #[plan]
  let hOutputShape ←
    certifyGeneratedInvariant
      "that the logical output axes reconstruct the result shape"
      (← mkEq outputShape planInferredOutputShape)
  let planValid ←
    buildCertificate ``TransformPlan.Valid.mk #[plan]
      #[normalizedValid, hLiteralAxes, hInputShape, hOutputShape] 0
  let checked ← mkAppM ``Transform.mk #[plan, planValid]
  return (checked, outputShape)

/--
Rebuild a concrete checker result with kernel-checked validity proofs.

The elaborator has already run the checker, so the rebuilt term records its normalization,
axis lengths and output shape, then decides the normalization and plan invariants.
-/
private def reifyCheckedTransform (checked : Transform) :
    TermElabM Expr := do
  let normalized := checked.value.normalization.value
  let normalizedExpr := Lean.toExpr normalized
  let normalizedValid ←
    buildCertificate ``NormalForm.Valid.mk
      #[normalizedExpr] #[] 13
  let normalization ←
    mkAppM ``CheckedNormalization.mk #[
      normalizedExpr,
      normalizedValid]
  let mut partialAxisLengths ←
    mkAppM ``PartialAxisLengths.seed #[
      Lean.toExpr ([] : SupplementaryLengths)]
  for axis in checked.value.allAxes.eraseDups do
    partialAxisLengths ←
      mkAppM ``PartialAxisLengths.set #[
        partialAxisLengths,
        Lean.toExpr axis,
        Lean.toExpr (checked.value.axisLength axis)]
  let axisLength ←
    withLocalDeclD `axis (mkConst ``AxisId) fun axis => do
      let assignedLength ← mkAppM' partialAxisLengths #[axis]
      let totalLength ←
        mkAppM ``Option.getD #[assignedLength, mkNatLit 0]
      mkLambdaFVars #[axis] totalLength
  let plan ←
    mkAppM ``TransformPlan.mk #[
      normalization,
      axisLength,
      Lean.toExpr checked.value.output]
  let planValid ←
    buildCertificate ``TransformPlan.Valid.mk
      #[plan] #[normalizedValid] 3
  mkAppM ``Transform.mk #[plan, planValid]

/--
Construct kernel-checked evidence that a reflected reduction plan has
positive fibers.
-/
private def certifyPositiveReductionFiber (checked : Expr) : TermElabM Expr := do
  let fiberSize ←
    mkAppM ``Transform.reductionFiberSize #[checked]
  let positive ← mkAppM ``LT.lt #[mkNatLit 0, fiberSize]
  certifyGeneratedInvariant "that every reduction fiber is nonempty" positive

/--
Choose one dimension of the concrete witness for the structural `parse_shape` checker. A
position holding a single numeric axis gets its literal length, and every other position gets
`1`.
-/
private def parseShapeDummyDimension :
    Option CompositeAxis → Nat
  | none => 1
  | some axis =>
      match axis.semanticAxes with
      | [] => 1
      | [item] =>
          match item.value with
          | .anonymous value _ => value
          | .named _ | .unit | .ellipsis => 1
      | _ => 1

/--
Emit the ordered `(name, dimension)` bindings selected by a checked
`parse_shape` expression.
-/
private def parseShapeBindingsExpr
    (expanded : List (Option CompositeAxis))
    (dimensions : List Expr) : MetaM Expr := do
  let mut bindings : List Expr := []
  for (patternAxis, dimension) in expanded.zip dimensions do
    match patternAxis with
    | some axis =>
        match axis.semanticAxes with
        | [item] =>
            match item.value with
            | .named name =>
                unless name == "_" do
                  bindings := bindings.concat <|
                    ← mkAppM ``Prod.mk #[Lean.toExpr name, dimension]
            | _ => pure ()
        | _ => pure ()
    | none => pure ()
  let bindingType ← mkAppM ``Prod #[mkConst ``String, mkConst ``Nat]
  mkListLit bindingType bindings

/--
Elaborate the input, supplementary lengths, parser, checker, and reflected
certificate shared by rearrange, repeat, and reduce.

Returning ordinary values in a tuple keeps the metaprogramming path shared
without adding another public plan representation.
-/
private def elaborateTransform (operation : String)
    (kind : TransformKind) (tensorSyntax : Syntax)
    (sourceSyntax : TSyntax `str)
    (lengthSyntax : Array (TSyntax `einopsAxisLength))
    (expectedType? : Option Expr) :
    TermElabM
      (Expr × Expr × Option Transform × Expr × Expr) := do
  let (tensor, scalarType, _, dimensions) ← elaborateTensor tensorSyntax
  let supplementary ← supplementaryExpr lengthSyntax
  let source := sourceSyntax.getString
  let pattern ← parsedTransformPattern operation source
  let concreteShape? ← concreteNatExpressions? dimensions
  let concreteSupplementary? ← concreteSupplementary? supplementary
  match concreteShape?, concreteSupplementary? with
  | some concreteShape, some concreteSupplementary =>
      match TransformPattern.check kind pattern concreteShape concreteSupplementary with
      | .ok checked =>
          let checkedExpr ← reifyCheckedTransform checked
          let outputShape := Lean.toExpr checked.value.output
          return (tensor, scalarType, some checked, checkedExpr, outputShape)
      | .error diagnostic =>
          if diagnostic.code = .cannotInferAxis ||
              diagnostic.code = .missingAxisLength then
            if (← expectedTensorShape? expectedType?).isSome then
              try
                let (checked, outputShape) ←
                  symbolicCheckedTransformExpr operation source kind pattern
                    dimensions supplementary expectedType?
                return (tensor, scalarType, none, checked, outputShape)
              catch _ =>
                throwPatternDiagnostic operation "shape" source
                  diagnostic.message diagnostic.span
            else
              throwPatternDiagnostic operation "shape" source
                diagnostic.message diagnostic.span
          else
            throwPatternDiagnostic operation "shape" source
              diagnostic.message diagnostic.span
  | _, _ =>
      let (checked, outputShape) ←
        symbolicCheckedTransformExpr operation source kind pattern dimensions
          supplementary expectedType?
      return (tensor, scalarType, none, checked, outputShape)

/--
Compile a transformation against either its fused source or one materialized
predecessor.

When inherited flat-index arithmetic becomes expensive, the predecessor is bound outside the
generated callback through `Native.stage`. Its certified native kernel then runs before the
consumer and stays out of each scalar callback. The proof is about values, and
`Native.stage_eq` states that staging leaves the value unchanged.
-/
private def withTransformSource
    (inputShape tensor : Expr)
    (compile :
      Expr → Expr → Expr → Expr → Option (Expr × Expr) → Bool →
        TermElabM Expr) :
    TermElabM Expr := do
  let identitySource (sourceTensor : Expr) : TermElabM Expr := do
    let coordinateType ← mkAppM ``Shape.Coord #[inputShape]
    let inputMap ←
      withLocalDeclD `inputCoordinate coordinateType fun inputCoordinate =>
        mkLambdaFVars #[inputCoordinate] inputCoordinate
    let (inputFlatMap, hInputMap) ←
      directFlatProjection inputShape inputMap
    compile inputMap inputFlatMap hInputMap sourceTensor none false
  if let some
      (inputMap, inputFlatMap, hInputMap, sourceTensor, _, logicalTensor,
        hLogicalTensor) ←
      fusedTransformInput? inputShape tensor then
    if shouldFuseFlatIndex inputFlatMap then
      return ←
        compile inputMap inputFlatMap hInputMap sourceTensor
          (some (logicalTensor, hLogicalTensor)) true
    let tensorType ← inferType tensor
    let consumer ←
      withLocalDeclD `materializedTransform tensorType fun materialized => do
        let body ← identitySource materialized
        mkLambdaFVars #[materialized] body
    return ← mkAppM ``stage #[tensor, consumer]
  identitySource tensor

/--
Choose the compact public transformation represented by one native kernel and
prove that the fused flat pullback has the same value.

Direct kernels reference the ordinary public lowering. Fused kernels reference
that same lowering applied to the preceding logical tensor, which keeps the
implementation-oriented flat pullback out of tactics and reports.
-/
private def transformSemanticReference
    (semantic checked hKind hAxes inputMap inputFlatMap hInputMap
      sourceTensor : Expr)
    (logicalInput? : Option (Expr × Expr)) (isFused : Bool)
    (publicTransform directCorrect fusedCorrect : Name) :
    TermElabM (Expr × Expr) := do
  if !isFused then
    let reference ←
      mkAppM publicTransform #[checked, hKind, sourceTensor]
    let hSemanticReference ←
      mkAppM ``Eq.symm #[
        ← mkAppM directCorrect #[checked, hKind, sourceTensor]]
    return (reference, hSemanticReference)
  let some (logicalTensor, hLogicalTensor) := logicalInput?
    | throwError
        "internal error: a fused transform has no logical input certificate"
  let reference ←
    mkAppM publicTransform #[checked, hKind, logicalTensor]
  let fused ←
    mkAppM ``Lowering.transformFused #[
      checked, hAxes, inputMap, inputFlatMap, hInputMap, sourceTensor]
  let hSemanticFused ←
    withTransparency .all <|
      mkExpectedTypeHint (← mkEqRefl semantic)
        (← mkEq semantic fused)
  let hFusedPublic ←
    mkAppM fusedCorrect #[
      checked, hKind, inputMap, inputFlatMap, hInputMap, sourceTensor]
  let hFlatPull ←
    mkAppM ``Rep.pullFlat_eq_pull #[
      inputFlatMap, inputMap, hInputMap, sourceTensor]
  let hLogicalPull ←
    mkAppM ``Eq.trans #[hLogicalTensor, hFlatPull]
  let logicalTensorType ← inferType logicalTensor
  let transform ←
    withLocalDeclD `logicalInput logicalTensorType fun logicalInput => do
      let body ←
        mkAppM publicTransform #[checked, hKind, logicalInput]
      mkLambdaFVars #[logicalInput] body
  let hPublicInputs ←
    mkAppM ``congrArg #[transform, hLogicalPull]
  let hFusedReference ←
    mkAppM ``Eq.trans #[
      hFusedPublic, ← mkAppM ``Eq.symm #[hPublicInputs]]
  let hSemanticReference ←
    mkAppM ``Eq.trans #[hSemanticFused, hFusedReference]
  return (reference, hSemanticReference)

/--
Elaborate `parse_shape`, returning named symbolic dimensions only after the
pattern's rank, literals, ellipsis, and wildcard obligations are proved.
-/
@[term_elab TorchLean.Tensor.parseShapeStx]
def elabParseShape : TermElab := fun stx expectedType? => withRef stx do
  let `(parse_shape $tensorSyntax:term $sourceSyntax:str) := stx
    | throwUnsupportedSyntax
  let (_, _, _, dimensions) ← elaborateTensor tensorSyntax
  let source := sourceSyntax.getString
  let parsedExpression ←
    match Expression.parse source .parseShape with
    | .error diagnostic =>
        throwPatternDiagnostic "parse_shape" "pattern" source
          diagnostic.message diagnostic.span
    | .ok expression => pure expression
  let expanded :=
    ParseShape.expandAxes parsedExpression dimensions.length
  let dummyShape :=
    if expanded.length == dimensions.length then expanded.map parseShapeDummyDimension
    else List.replicate dimensions.length 1
  let checkedShape :=
    (← concreteNatExpressions? dimensions).getD dummyShape
  match ParseShape.check parsedExpression checkedShape with
  | .error diagnostic =>
      throwPatternDiagnostic "parse_shape" "shape" source
        diagnostic.message diagnostic.span
  | .ok _ => pure ()
  let inputShape ← shapeExpr dimensions
  let axesMatch ←
    mkAppM ``List.Forall₂ #[
      mkConst ``ParseShape.AxisMatches,
      Lean.toExpr expanded,
      inputShape]
  let hAxesMatch ←
    certifyGeneratedInvariant
      "that parse_shape literals agree with the tensor dimensions"
      axesMatch
  let result ← parseShapeBindingsExpr expanded dimensions
  -- The rank-compatible witness makes every structural parser/checker
  -- decision reducible, while `hAxesMatch` checks the actual dimensions.
  let checkedResult ←
    mkAppM ``ParseShape.check #[
      Lean.toExpr parsedExpression, Lean.toExpr checkedShape]
  let structuralEvidence ← extractSuccessfulResult checkedResult
  let structuralEvidenceType ← inferType structuralEvidence
  let result :=
    mkLet `parseShapeStructuralCheck structuralEvidenceType
      structuralEvidence result (nondep := true)
  let result :=
    mkLet `parseShapeAxesMatch axesMatch hAxesMatch result
      (nondep := true)
  ensureHasType expectedType? result

/--
Elaborate a rearrangement or repeat as the flat pullback certified by its checked plan of kind
`kind`. `publicTransform` is the public lowering of that kind, and `directCorrect` and
`fusedCorrect` relate it to the direct and fused flat pullbacks.
-/
private def elabPullTransform (operation : String) (kind : TransformKind)
    (tensorSyntax : Syntax) (sourceSyntax : TSyntax `str)
    (lengthSyntax : Array (TSyntax `einopsAxisLength)) (expectedType? : Option Expr)
    (publicTransform directCorrect fusedCorrect : Name) : TermElabM Expr := do
  let (tensor, _, concreteChecked?, checked, outputShape) ←
    elaborateTransform operation kind tensorSyntax sourceSyntax lengthSyntax
      expectedType?
  let hKind ← mkAppM ``Eq.refl #[Lean.toExpr kind]
  let checkedType ← inferType checked
  let result ← withLetDecl `checkedTransform checkedType checked fun checkedVar => do
    let (inputShape, _) ← checkedTransformShapes checkedVar
    let hAxes ← inputAxesSubsetProof kind checkedVar hKind
    let body ← withTransformSource inputShape tensor fun
        inputMap inputFlatMap hInputMap sourceTensor logicalInput?
        isFused => do
      let checkedFlatMap ←
        mkAppM ``Transform.inputFlatIndexOfOutput #[
          checkedVar, hAxes]
      let semanticFlatMap ←
        if isFused then
          mkAppM ``Function.comp #[inputFlatMap, checkedFlatMap]
        else
          pure checkedFlatMap
      let semantic ←
        mkAppM ``Rep.pullFlat #[semanticFlatMap, sourceTensor]
      let (reference, hSemanticReference) ←
        transformSemanticReference
          semantic checkedVar hKind hAxes inputMap inputFlatMap hInputMap
          sourceTensor logicalInput? isFused
          publicTransform directCorrect fusedCorrect
      match ← compileNativeTransform? checkedVar hAxes inputFlatMap
          sourceTensor reference hSemanticReference isFused concreteChecked? with
      | some native => pure native
      | none => pure reference
    mkLetFVars (generalizeNondepLet := false) #[checkedVar] body
  let result ← castTensorToCompactShape result outputShape
  exposePublicTensorType result outputShape expectedType?

/--
Elaborate rearrangement syntax to the coordinate equivalence certified by its
symbolic or concrete transformation plan.
-/
@[term_elab TorchLean.Tensor.rearrangeStx]
def elabRearrange : TermElab := fun stx expectedType? => withRef stx do
  let `(rearrange $tensorSyntax:term $sourceSyntax:str
      $[with $lengthSyntax:einopsAxisLength,*]?) := stx
    | throwUnsupportedSyntax
  let lengthSyntax :=
    match lengthSyntax with
    | none => #[]
    | some entries => entries.getElems
  elabPullTransform "rearrange" .rearrange tensorSyntax sourceSyntax lengthSyntax
    expectedType? ``Lowering.rearrange ``Lowering.rearrange_eq_pullFlat
    ``Lowering.transformFused_eq_rearrange

/--
Elaborate expand syntax to the generic pullback along its checked coordinate
projection, including symbolic new-axis lengths.
-/
@[term_elab TorchLean.Tensor.expandStx]
def elabExpand : TermElab := fun stx expectedType? => withRef stx do
  let `(expand $tensorSyntax:term $sourceSyntax:str
      $[with $lengthSyntax:einopsAxisLength,*]?) := stx
    | throwUnsupportedSyntax
  let lengthSyntax :=
    match lengthSyntax with
    | none => #[]
    | some entries => entries.getElems
  elabPullTransform "expand" .expand tensorSyntax sourceSyntax lengthSyntax
    expectedType? ``Lowering.expand ``Lowering.expand_eq_pullFlat
    ``Lowering.transformFused_eq_expand

/--
Elaborate built-in or user-supplied reduction syntax to the generic finite
fiber reduction, proving nonemptiness for reducers without an empty case.
-/
@[term_elab TorchLean.Tensor.reduceStx]
def elabReduce : TermElab := fun stx expectedType? => withRef stx do
  let `(reduce $tensorSyntax:term $sourceSyntax:str
      by $reduction:einopsReduction
      $[with $lengthSyntax:einopsAxisLength,*]?) := stx
    | throwUnsupportedSyntax
  let reductionName? :=
    match reduction with
    | `(einopsReduction| $name:ident) => some name.getId.eraseMacroScopes
    | _ => none
  let lengthSyntax :=
    match lengthSyntax with
    | none => #[]
    | some entries => entries.getElems
  let (tensor, scalarType, concreteChecked?, checked, outputShape) ←
    elaborateTransform "reduce" .reduce tensorSyntax sourceSyntax lengthSyntax
      expectedType?
  let scalarType ← whnf (← instantiateMVars scalarType)
  let hKind ←
    mkAppM ``Eq.refl #[mkConst ``TransformKind.reduce]
  let (inputShape, _) ← checkedTransformShapes checked
  let fusedInput? ← fusedTransformInput? inputShape tensor
  let flatReader? ←
    match fusedInput? with
    | none => pure none
    | some (_, inputFlatMap, _, sourceTensor, hTensor, logicalTensor,
        hLogicalTensor) => do
        let inputSize ← mkAppM ``Shape.size #[inputShape]
        let inputIndexType ← mkAppM ``Fin #[inputSize]
        let read ←
          withLocalDeclD `inputIndex inputIndexType fun inputIndex => do
            let sourceIndex := mkApp inputFlatMap inputIndex
            let value ←
              mkAppM ``Rep.getFlat #[sourceTensor, sourceIndex]
            mkLambdaFVars #[inputIndex] value
        let hRead ←
          mkAppM ``Rep.getFlat_eq_of_eq_pullFlat #[
            tensor, sourceTensor, inputFlatMap, hTensor]
        let hLogicalRead ←
          mkAppM ``Rep.getFlat_eq_of_eq_pullFlat #[
            logicalTensor, sourceTensor, inputFlatMap, hLogicalTensor]
        let hLogicalSemantic ←
          mkAppM ``Eq.trans #[
            hLogicalTensor, ← mkAppM ``Eq.symm #[hTensor]]
        pure <| some
          (read, hRead, hLogicalRead, sourceTensor, inputFlatMap,
            logicalTensor, hLogicalSemantic)
  let nativeReaderData ←
    match flatReader? with
    | some (read, _, hLogicalRead, sourceTensor, inputFlatMap,
        logicalTensor, hLogicalSemantic) =>
        pure
          (read, hLogicalRead, sourceTensor, inputFlatMap, logicalTensor,
            hLogicalSemantic)
    | none => do
        let inputSize ← mkAppM ``Shape.size #[inputShape]
        let inputIndexType ← mkAppM ``Fin #[inputSize]
        let inputFlatMap ←
          withLocalDeclD `inputIndex inputIndexType fun inputIndex =>
            mkLambdaFVars #[inputIndex] inputIndex
        let read ←
          withLocalDeclD `inputIndex inputIndexType fun inputIndex => do
            let value ← mkAppM ``Rep.getFlat #[tensor, inputIndex]
            mkLambdaFVars #[inputIndex] value
        let hRead ←
          withLocalDeclD `inputIndex inputIndexType fun inputIndex => do
            let value := mkApp read inputIndex
            mkLambdaFVars #[inputIndex] (← mkEqRefl value)
        pure
          (read, hRead, tensor, inputFlatMap, tensor, ← mkEqRefl tensor)
  let scalarInstance (className : Name) : TermElabM Expr := do
    let instanceType ← mkAppM className #[scalarType]
    withRef reduction <| synthInstance instanceType
  let ensureBooleanInput (reductionName : Name) : TermElabM Unit := do
    unless ← isDefEq scalarType (mkConst ``Bool) do
      throwErrorAt reduction
        "einops reduction '{reductionName}' requires a Boolean tensor, but \
          the input scalar type is{indentExpr scalarType}"
  let scalarBinaryFunction (operation : Name) : TermElabM Expr :=
    withLocalDeclD `total scalarType fun total =>
      withLocalDeclD `value scalarType fun value => do
        let body ← mkAppM operation #[total, value]
        mkLambdaFVars #[total, value] body
  -- The step and seed of `Tensor.reduce`'s `reduceOp`, read from the fields of its
  -- `ReduceOp.Scalar` instance so that einops and `Tensor.reduce` use one step and seed. A
  -- scalar without the instance (such as one above `Type`) steps with its own `operation` from
  -- the numeral `fallbackSeed`. Without a seed the fold starts from a fiber's first entry and
  -- steps the others, where `Tensor.reduce` also steps the first with itself, so a one-entry
  -- fiber of a signaling NaN stays signaling here.
  let builtinFold (reduceOp className operation : Name) (fallbackSeed : Option Nat) :
      TermElabM (Expr × Option Expr) := do
    let level ← instantiateLevelMVars (← getDecLevel scalarType)
    let reduceInstance? ←
      if level == .zero then
        let instanceType :=
          mkApp2 (mkConst ``ReduceOp.Scalar) (mkConst reduceOp) scalarType
        try some <$> synthInstance instanceType catch _ => pure none
      else
        pure none
    match reduceInstance? with
    | some reduceInstance =>
        let some step ← project? reduceInstance 0
          | throwError "internal error: could not read the step of{indentExpr reduceInstance}"
        let some seed ← project? reduceInstance 1
          | throwError "internal error: could not read the seed of{indentExpr reduceInstance}"
        let seed ← whnf seed
        if seed.isAppOfArity ``Option.some 2 then
          pure (step, some seed.appArg!)
        else if seed.isAppOfArity ``Option.none 1 then
          pure (step, none)
        else
          throwError "internal error: the seed of{indentExpr reduceInstance}\nis not a constructor"
    | none =>
        let _ ← scalarInstance className
        let seed ← fallbackSeed.mapM fun seed => Expr.ofNat scalarType seed
        pure (← scalarBinaryFunction operation, seed)
  let identityFinalizer : TermElabM Expr :=
    withLocalDeclD `total scalarType fun total =>
      withLocalDeclD `cardinality (mkConst ``Nat) fun cardinality =>
        mkLambdaFVars #[total, cardinality] total
  -- At a division ring of characteristic zero the finalizer divides by the count cast to the
  -- scalar, the form that `Lowering.reduceFold_mean` rewrites to `Reduction.mean`. Every
  -- other scalar with a `Context` divides with `Context.divNat`, as `Tensor.mean` does. Under
  -- `LawfulContext`, `Lowering.reduceFold_ordered_correct` and then
  -- `Semantics.denoteOrderedReduce_divNat_eq_mean` rewrite that form to `Reduction.mean`. In a
  -- binary format `Context.divNat` rounds the exact quotient once, where dividing by the count
  -- rounded to the format rounds twice at a count above `2 ^ 24` (binary32) or `2 ^ 53`
  -- (binary64) that the format does not represent.
  let scalarContext? : TermElabM (Option Expr) := do
    try some <$> scalarInstance ``Context catch _ => pure none
  let meanFinalizer (context? : Option Expr) : TermElabM Expr :=
    withLocalDeclD `total scalarType fun total =>
      withLocalDeclD `cardinality (mkConst ``Nat) fun cardinality => do
        let result ←
          match context? with
          | some context =>
              pure <| mkApp4 (mkConst ``Context.divNat) scalarType context total cardinality
          | none => mkAppM ``HDiv.hDiv #[total, ← mkCoe scalarType cardinality]
        mkLambdaFVars #[total, cardinality] result
  let elaborateFoldReduction
      (step initial finish : Expr) : TermElabM Expr := do
    let (read, hRead, sourceTensor, inputFlatMap, logicalTensor,
      hLogicalSemantic) := nativeReaderData
    let nativeResult? ←
      match concreteChecked? with
      | some _ =>
          compileNativeReduceFold? step initial finish checked
            hKind logicalTensor tensor hLogicalSemantic read hRead
            sourceTensor inputFlatMap
      | none => pure none
    match nativeResult? with
    | some nativeResult => pure nativeResult
    | none =>
        match flatReader? with
        | none =>
            mkAppM ``Lowering.reduceFold #[
              step, initial, finish, checked, hKind, tensor]
        | some (read, hRead, _, _, _, _, _) =>
            mkAppM ``Lowering.reduceFoldView #[
              step, initial, finish, checked, hKind, tensor, read, hRead]
  let rejectEmptyFibers (reductionName : Name) : TermElabM Unit := do
    if let some checkedValue := concreteChecked? then
      if checkedValue.reductionFiberSize = 0 then
        throwErrorAt reduction
          "einops reduction '{reductionName}' is undefined on empty \
            fibers: the product of the removed-axis lengths is zero"
  let elaborateBuiltinReduction (reductionName : Name) (step : Expr)
      (seed : Option Expr) : TermElabM Expr := do
    match seed with
    | some initial =>
        elaborateFoldReduction step initial (← identityFinalizer)
    | none =>
        rejectEmptyFibers reductionName
        let hPositiveFiber ← certifyPositiveReductionFiber checked
        match flatReader? with
        | none =>
            mkAppM ``Lowering.reduceNonemptyFold #[
              step, checked, hKind, hPositiveFiber, tensor]
        | some (read, hRead, _, _) =>
            mkAppM ``Lowering.reduceNonemptyFoldView #[
              step, checked, hKind, hPositiveFiber, tensor, read, hRead]
  let elaborateCustomReduction : TermElabM Expr := do
    let aggregateSyntax :=
      match reduction with
      | `(einopsReduction| $name:ident) => name.raw
      | `(einopsReduction| ($aggregate:term)) => aggregate.raw
      | _ => reduction.raw
    let inputMultisetType ← mkAppM ``Multiset #[scalarType]
    let outputScalarType ← mkFreshTypeMVar
    let aggregateType ← mkArrow inputMultisetType outputScalarType
    let aggregate ←
      withRef aggregateSyntax <|
        elabTerm aggregateSyntax (some aggregateType)
    synthesizeSyntheticMVarsNoPostponing
    let aggregate ← instantiateMVars aggregate
    match flatReader? with
    | none =>
        mkAppM ``Lowering.reduce #[
          aggregate, checked, hKind, tensor]
    | some (read, hRead, _, _, _, _, _) =>
        mkAppM ``Lowering.reduceView #[
          aggregate, checked, hKind, tensor, read, hRead]
  let result ←
    match reductionName? with
    | some `any =>
        ensureBooleanInput `any
        elaborateFoldReduction
          (mkConst ``Bool.or)
          (mkConst ``Bool.false)
          (← identityFinalizer)
    | some `all =>
        ensureBooleanInput `all
        elaborateFoldReduction
          (mkConst ``Bool.and)
          (mkConst ``Bool.true)
          (← identityFinalizer)
    | some `mean =>
        rejectEmptyFibers `mean
        let (step, seed) ← builtinFold ``ReduceOp.sum ``Add ``HAdd.hAdd (some 0)
        let some initial := seed
          | throwErrorAt reduction
              "einops reduction 'mean' sums from the seed of `ReduceOp.Scalar .sum`, which \
                is `none` at{indentExpr scalarType}"
        let _ ← scalarInstance ``Div
        let context? ←
          try
            let _ ← scalarInstance ``DivisionRing
            let additiveMonoidWithOne ←
              scalarInstance ``AddMonoidWithOne
            let charZeroType ←
              mkAppOptM ``CharZero #[
                some scalarType, some additiveMonoidWithOne]
            let _ ← withRef reduction <| synthInstance charZeroType
            pure none
          catch fieldError =>
            match ← scalarContext? with
            | some context => pure (some context)
            | none => throw fieldError
        let _ ← certifyPositiveReductionFiber checked
        elaborateFoldReduction step initial (← meanFinalizer context?)
    | some reductionName =>
        let builtin? : Option (Name × Name × Name × Option Nat) :=
          match reductionName with
          | `sum => some (``ReduceOp.sum, ``Add, ``HAdd.hAdd, some 0)
          | `prod => some (``ReduceOp.prod, ``Mul, ``HMul.hMul, some 1)
          | `min => some (``ReduceOp.min, ``Min, ``min, none)
          | `max => some (``ReduceOp.max, ``Max, ``max, none)
          | _ => none
        match builtin? with
        | some (reduceOp, className, operation, fallbackSeed) =>
            let (step, seed) ← builtinFold reduceOp className operation fallbackSeed
            elaborateBuiltinReduction reductionName step seed
        | none => elaborateCustomReduction
    | none => elaborateCustomReduction
  let result ← castTensorToCompactShape result outputShape
  exposePublicTensorType result outputShape expectedType?

end TorchLean.Einops.Elab
