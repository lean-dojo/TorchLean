/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Check.ParseShape
public import NN.Tensor.Internal.Check.ParseShape
public meta import NN.Tensor.Internal.Elab.Native.Pull
public meta import NN.Tensor.Internal.Elab.Native.Reduce
public meta import NN.Tensor.Internal.Elab.Transform.View
public meta import NN.Tensor.Internal.Semantics.Transform.Geometry
public import NN.Tensor.Internal.Syntax.Diagnostic
public meta import NN.Tensor.Internal.Syntax.Parser.Expression.Roundtrip
public meta import NN.Tensor.Internal.Syntax.Parser.Transform
public import NN.Tensor.Internal.Lowering.Rearrange -- shake: keep
public import NN.Tensor.Internal.Lowering.Reduce.View -- shake: keep
public import NN.Tensor.Internal.Lowering.Repeat -- shake: keep
public import NN.Tensor.Internal.Lowering.TransformFusion -- shake: keep
public import NN.Tensor.Internal.Representation.Reduction -- shake: keep

/-!
# Elaboration of transformations and parse-shape expressions

This module implements `rearrange`, `expand`, `reduce`, and `parse_shape`.
Concrete dimensions use the executable checker for precise source
diagnostics. Symbolic dimensions construct the same checked transformation
type and prove every shape equation in the caller's Lean context.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab

open TorchLean.Tensor
open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta
open Impl

/--
Parse a transformation pattern and report failures against the original
source span.
-/
private def parsedTransformPattern (operation source : String) :
    TermElabM Syntax.TransformPattern :=
  match Syntax.parseTransformPattern source with
  | .error diagnostic =>
      throwPatternDiagnostic operation "pattern" source
        diagnostic.message diagnostic.span
  | .ok pattern => pure pattern

/--
Reject duplicate or unused supplementary axis lengths before solving symbolic
shape equations.
-/
private def validateSymbolicSupplementary (operation source : String)
    (pattern : Syntax.TransformPattern)
    (normalized : Check.NormalizedTransform)
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
Seed symbolic axis lengths from literal axes and explicit user assignments.
-/
private def seedSymbolicAssignments
    (normalized : Check.NormalizedTransform)
    (supplementary : List (String × Expr)) :
    List (Check.AxisId × Expr) :=
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
Recover the one unknown factor in a grouped physical dimension, preferring an
explicit product factor before introducing division obligations.
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
    (initial : List (Check.AxisId × Expr))
    (constraints : List (List Check.AxisId × Expr)) :
    TermElabM (List (Check.AxisId × Expr)) := do
  let mut assignments := initial
  let mut changed := true
  while changed do
    changed := false
    for (group, dimension) in constraints do
      let unknownAxes :=
        group.filter fun axis => (axisExpression? assignments axis).isNone
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
Build the total dependent axis-length function stored by a transformation
plan; unresolved axes map to zero and are rejected before this point.
-/
private def axisLengthExpr (assignments : List (Check.AxisId × Expr)) :
    MetaM Expr := do
  let mut partialAxisLengths ←
    mkAppM ``Check.PartialAxisLengths.seed #[
      Lean.toExpr ([] : Check.SupplementaryLengths)]
  for (axis, length) in assignments do
    partialAxisLengths ←
      mkAppM ``Check.PartialAxisLengths.set #[
        partialAxisLengths, Lean.toExpr axis, length]
  withLocalDeclD `axis (mkConst ``Check.AxisId) fun axis => do
    let assignedLength ← mkAppM' partialAxisLengths #[axis]
    let totalLength ←
      mkAppM ``Option.getD #[assignedLength, mkNatLit 0]
    mkLambdaFVars #[axis] totalLength

/--
Reuse rank-only normalization data with the caller's symbolic input shape and
rebuild its structural validity certificate.
-/
private def reifySymbolicNormalization
    (normalization : Check.CheckedNormalization) (inputShape : Expr) :
    MetaM (Expr × Expr) := do
  let normalized := normalization.value
  let normalizedExpr ←
    mkAppM ``Check.NormalizedTransform.mk #[
      Lean.toExpr normalized.kind,
      Lean.toExpr normalized.source,
      inputShape,
      Lean.toExpr normalized.ellipsisRank,
      Lean.toExpr normalized.ellipsisAxes,
      Lean.toExpr normalized.inputGroups,
      Lean.toExpr normalized.outputGroups]
  let normalizedValid ←
    buildCertificate ``Check.NormalizedTransform.Valid.mk
      #[normalizedExpr] #[] 13
  let checkedNormalization ←
    mkAppM ``Check.CheckedNormalization.mk #[
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
    (kind : Check.TransformKind) (pattern : Syntax.TransformPattern)
    (dimensions : List Expr) (supplementary : List (String × Expr))
    (expectedType? : Option Expr) : TermElabM (Expr × Expr) := do
  let dummyShape := List.replicate dimensions.length 1
  let normalization ←
    match Check.normalize kind pattern dummyShape with
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
      (axisExpression? assignments axis).isNone
  unless unresolved.isEmpty do
    let axes := String.intercalate ", " (unresolved.map axisDescription)
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
    mkAppM ``Check.TransformPlan.mk #[
      checkedNormalization, axisLength, outputShape]
  let literalAxes ←
    mkAppM ``Check.TransformPlan.literalAxesAgree #[plan]
  let hLiteralAxes ←
    certifyGeneratedInvariant "that anonymous axes retain their literal lengths"
      (← mkEq literalAxes (mkConst ``Bool.true))
  let inferredInputShape ←
    mkAppM ``Check.TransformPlan.inferredInput #[plan]
  let hInputShape ←
    certifyGeneratedInvariant
      "that the logical input axes reconstruct the tensor shape"
      (← mkEq inferredInputShape inputShape)
  let planInferredOutputShape ←
    mkAppM ``Check.TransformPlan.inferredOutput #[plan]
  let hOutputShape ←
    certifyGeneratedInvariant
      "that the logical output axes reconstruct the result shape"
      (← mkEq outputShape planInferredOutputShape)
  let planValid ←
    buildCertificate ``Check.TransformPlan.Valid.mk #[plan]
      #[normalizedValid, hLiteralAxes, hInputShape, hOutputShape] 0
  let checked ← mkAppM ``Check.CheckedTransform.mk #[plan, planValid]
  return (checked, outputShape)

/--
Extract a successful checker result in the generated term.

The elaborator has already run the checker to select a diagnostic or continue.
This expression reruns the accepted computation, and the proof required by
`Option.get` makes its success part of the kernel-checked term.
-/
private def reifyCheckedTransform (checked : Check.CheckedTransform) :
    TermElabM Expr := do
  let normalized := checked.value.normalization.value
  let normalizedExpr := Lean.toExpr normalized
  let normalizedValid ←
    buildCertificate ``Check.NormalizedTransform.Valid.mk
      #[normalizedExpr] #[] 13
  let normalization ←
    mkAppM ``Check.CheckedNormalization.mk #[
      normalizedExpr,
      normalizedValid]
  let mut partialAxisLengths ←
    mkAppM ``Check.PartialAxisLengths.seed #[
      Lean.toExpr ([] : Check.SupplementaryLengths)]
  for axis in checked.value.allAxes.eraseDups do
    partialAxisLengths ←
      mkAppM ``Check.PartialAxisLengths.set #[
        partialAxisLengths,
        Lean.toExpr axis,
        Lean.toExpr (checked.value.axisLength axis)]
  let axisLength ←
    withLocalDeclD `axis (mkConst ``Check.AxisId) fun axis => do
      let assignedLength ← mkAppM' partialAxisLengths #[axis]
      let totalLength ←
        mkAppM ``Option.getD #[assignedLength, mkNatLit 0]
      mkLambdaFVars #[axis] totalLength
  let plan ←
    mkAppM ``Check.TransformPlan.mk #[
      normalization,
      axisLength,
      Lean.toExpr checked.value.output]
  let planValid ←
    buildCertificate ``Check.TransformPlan.Valid.mk
      #[plan] #[normalizedValid] 3
  mkAppM ``Check.CheckedTransform.mk #[plan, planValid]

/--
Construct kernel-checked evidence that a reflected reduction plan has
positive fibers.
-/
private def certifyPositiveReductionFiber (checked : Expr) : TermElabM Expr := do
  let fiberSize ←
    mkAppM ``Check.CheckedTransform.reductionFiberSize #[checked]
  let positive ← mkAppM ``LT.lt #[mkNatLit 0, fiberSize]
  certifyGeneratedInvariant "that every reduction fiber is nonempty" positive

/--
Choose a rank-compatible concrete witness for the structural `parse_shape`
checker while preserving literal-axis requirements.
-/
private def parseShapeDummyDimension :
    Option Syntax.CompositeAxis → Nat
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
    (expanded : List (Option Syntax.CompositeAxis))
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
    (kind : Check.TransformKind) (tensorSyntax : Syntax)
    (sourceSyntax : TSyntax `str)
    (lengthSyntax : Array (TSyntax `einopsAxisLength))
    (expectedType? : Option Expr) :
    TermElabM
      (Expr × Expr × Option Check.CheckedTransform × Expr × Expr) := do
  let (tensor, scalarType, _, dimensions) ← elaborateTensor tensorSyntax
  let supplementary ← supplementaryExpr lengthSyntax
  let source := sourceSyntax.getString
  let pattern ← parsedTransformPattern operation source
  let concreteShape? ← concreteNatExpressions? dimensions
  let concreteSupplementary? ← concreteSupplementary? supplementary
  match concreteShape?, concreteSupplementary? with
  | some concreteShape, some concreteSupplementary =>
      match Check.checkTransform kind pattern concreteShape concreteSupplementary with
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

When inherited flat-index arithmetic becomes expensive, the predecessor is
bound outside the generated callback. This evaluates its certified native
kernel once instead of rebuilding it for every output scalar.
-/
private def withTransformSource
    (inputShape tensor : Expr)
    (compile :
      Expr → Expr → Expr → Expr → Option (Expr × Expr) → Bool →
        TermElabM Expr) :
    TermElabM Expr := do
  let identitySource (sourceTensor : Expr) : TermElabM Expr := do
    let coordinateType ← mkAppM ``Coord #[inputShape]
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
    return ← mkAppM ``nativeStage #[tensor, consumer]
  identitySource tensor

/--
Choose the compact public transformation represented by one native kernel and
prove that the fused flat pullback has the same value.

Direct kernels reference the ordinary public lowering. Fused kernels reference
that same lowering applied to the preceding logical tensor, rather than
exposing an implementation-oriented flat pullback to tactics and reports.
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
    mkAppM ``Lowering.transformTensorFused #[
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
    match Syntax.parseExpression source .parseShape with
    | .error diagnostic =>
        throwPatternDiagnostic "parse_shape" "pattern" source
          diagnostic.message diagnostic.span
    | .ok expression => pure expression
  let expanded :=
    Check.ParseShape.expandAxes parsedExpression dimensions.length
  let dummyShape := expanded.map parseShapeDummyDimension
  let checkedShape :=
    (← concreteNatExpressions? dimensions).getD dummyShape
  match Check.checkParseShape parsedExpression checkedShape with
  | .error diagnostic =>
      throwPatternDiagnostic "parse_shape" "shape" source
        diagnostic.message diagnostic.span
  | .ok _ => pure ()
  let inputShape ← shapeExpr dimensions
  let axesMatch ←
    mkAppM ``List.Forall₂ #[
      mkConst ``Check.ParseShape.AxisMatches,
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
    mkAppM ``Check.checkParseShape #[
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
  let (tensor, _, concreteChecked?, checked, outputShape) ←
    elaborateTransform "rearrange" .rearrange tensorSyntax sourceSyntax
      lengthSyntax expectedType?
  let hKind ←
    mkAppM ``Eq.refl #[mkConst ``Check.TransformKind.rearrange]
  let checkedType ← inferType checked
  let result ← withLetDecl `checkedTransform checkedType checked fun checkedVar => do
    let (inputShape, _) ← checkedTransformShapes checkedVar
    let hAxes ← rearrangeAxesProof checkedVar hKind
    let body ← withTransformSource inputShape tensor fun
        inputMap inputFlatMap hInputMap sourceTensor logicalInput?
        isFused => do
      let checkedFlatMap ←
        mkAppM ``Check.CheckedTransform.inputFlatIndexOfOutput #[
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
          ``Lowering.rearrangeTensor
          ``Lowering.rearrangeTensor_eq_pullFlat
          ``Lowering.transformTensorFused_rearrange_correct
      match ← compileNativeTransform? checkedVar hAxes inputFlatMap
          sourceTensor reference hSemanticReference isFused concreteChecked? with
      | some native => pure native
      | none => pure reference
    mkLetFVars (generalizeNondepLet := false) #[checkedVar] body
  let result ← castTensorToCompactShape result outputShape
  exposePublicTensorType result outputShape expectedType?

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
  let (tensor, _, concreteChecked?, checked, outputShape) ←
    elaborateTransform "expand" .repeat tensorSyntax sourceSyntax lengthSyntax
      expectedType?
  let hKind ←
    mkAppM ``Eq.refl #[mkConst ``Check.TransformKind.repeat]
  let checkedType ← inferType checked
  let result ← withLetDecl `checkedTransform checkedType checked fun checkedVar => do
    let (inputShape, _) ← checkedTransformShapes checkedVar
    let hAxes ← repeatAxesProof checkedVar hKind
    let body ← withTransformSource inputShape tensor fun
        inputMap inputFlatMap hInputMap sourceTensor logicalInput?
        isFused => do
      let checkedFlatMap ←
        mkAppM ``Check.CheckedTransform.inputFlatIndexOfOutput #[
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
          ``Lowering.repeatTensor
          ``Lowering.repeatTensor_eq_pullFlat
          ``Lowering.transformTensorFused_repeat_correct
      match ← compileNativeTransform? checkedVar hAxes inputFlatMap
          sourceTensor reference hSemanticReference isFused concreteChecked? with
      | some native => pure native
      | none => pure reference
    mkLetFVars (generalizeNondepLet := false) #[checkedVar] body
  let result ← castTensorToCompactShape result outputShape
  exposePublicTensorType result outputShape expectedType?

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
    | `(einopsReduction| $name:ident) => some name.getId
    | _ => none
  let lengthSyntax :=
    match lengthSyntax with
    | none => #[]
    | some entries => entries.getElems
  let (tensor, scalarType, concreteChecked?, checked, outputShape) ←
    elaborateTransform "reduce" .reduce tensorSyntax sourceSyntax lengthSyntax
      expectedType?
  let scalarType ← whnf (← instantiateMVars scalarType)
  let isFloat32 := scalarType.isConstOf ``Float32
  let isFloat := scalarType.isConstOf ``Float
  let hKind ←
    mkAppM ``Eq.refl #[mkConst ``Check.TransformKind.reduce]
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
          mkAppM ``Lowering.source_getFlat_eq_of_eq_pullFlat #[
            tensor, sourceTensor, inputFlatMap, hTensor]
        let hLogicalRead ←
          mkAppM ``Lowering.source_getFlat_eq_of_eq_pullFlat #[
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
  let identityFinalizer : TermElabM Expr :=
    withLocalDeclD `total scalarType fun total =>
      withLocalDeclD `cardinality (mkConst ``Nat) fun cardinality =>
        mkLambdaFVars #[total, cardinality] total
  let meanFinalizer : TermElabM Expr :=
    withLocalDeclD `total scalarType fun total =>
      withLocalDeclD `cardinality (mkConst ``Nat) fun cardinality => do
        let scalarCardinality ←
          if isFloat32 then
            mkAppM ``Float32.ofNat #[cardinality]
          else if isFloat then
            mkAppM ``Float.ofNat #[cardinality]
          else
            mkCoe scalarType cardinality
        let result ← mkAppM ``HDiv.hDiv #[total, scalarCardinality]
        mkLambdaFVars #[total, cardinality] result
  let elaborateFoldReduction
      (step initial finish : Expr) : TermElabM Expr := do
    let (read, hRead, sourceTensor, inputFlatMap, logicalTensor,
      hLogicalSemantic) := nativeReaderData
    let nativeResult? ←
      match concreteChecked? with
      | some checkedValue =>
          Impl.compileNativeReduceFold? step initial finish checked
            hKind logicalTensor tensor hLogicalSemantic read hRead
            sourceTensor inputFlatMap checkedValue
      | none => pure none
    match nativeResult? with
    | some nativeResult => pure nativeResult
    | none =>
        match flatReader? with
        | none =>
            mkAppM ``Lowering.reduceFoldTensor #[
              step, initial, finish, checked, hKind, tensor]
        | some (read, hRead, _, _, _, _, _) =>
            mkAppM ``Lowering.reduceFoldTensorFromFlat #[
              step, initial, finish, checked, hKind, tensor, read, hRead]
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
        mkAppM ``Lowering.reduceTensor #[
          aggregate, checked, hKind, tensor]
    | some (read, hRead, _, _, _, _, _) =>
        mkAppM ``Lowering.reduceTensorFromFlat #[
          aggregate, checked, hKind, tensor, read, hRead]
  let result ←
    match reductionName? with
    | some `sum =>
        let _ ← scalarInstance ``Add
        elaborateFoldReduction
          (← scalarBinaryFunction ``HAdd.hAdd)
          (← Expr.ofNat scalarType 0)
          (← identityFinalizer)
    | some `prod =>
        let _ ← scalarInstance ``Mul
        elaborateFoldReduction
          (← scalarBinaryFunction ``HMul.hMul)
          (← Expr.ofNat scalarType 1)
          (← identityFinalizer)
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
    | some reductionName =>
        unless [`mean, `min, `max].contains reductionName do
          return ← elaborateCustomReduction
        match concreteChecked? with
        | some checkedValue =>
            if checkedValue.reductionFiberSize = 0 then
              throwErrorAt reduction
                "einops reduction '{reductionName}' is undefined on empty \
                  fibers; the product of the removed-axis lengths is zero"
        | none => pure ()
        match reductionName with
        | `mean =>
            let _ ← scalarInstance ``Add
            let _ ← scalarInstance ``Div
            unless isFloat32 || isFloat do
              let _ ← scalarInstance ``DivisionRing
              let additiveMonoidWithOne ←
                scalarInstance ``AddMonoidWithOne
              let charZeroType ←
                mkAppOptM ``CharZero #[
                  some scalarType, some additiveMonoidWithOne]
              let _ ← withRef reduction <| synthInstance charZeroType
            let _ ← certifyPositiveReductionFiber checked
            elaborateFoldReduction
              (← scalarBinaryFunction ``HAdd.hAdd)
              (← Expr.ofNat scalarType 0)
              (← meanFinalizer)
        | `min =>
            let _ ← scalarInstance ``Min
            let hPositiveFiber ← certifyPositiveReductionFiber checked
            let step ← scalarBinaryFunction ``min
            match flatReader? with
            | none =>
                mkAppM ``Lowering.reduceNonemptyFoldTensor #[
                  step, checked, hKind, hPositiveFiber, tensor]
            | some (read, hRead, _, _) =>
                mkAppM ``Lowering.reduceNonemptyFoldTensorFromFlat #[
                  step, checked, hKind, hPositiveFiber, tensor, read, hRead]
        | _ =>
            let _ ← scalarInstance ``Max
            let hPositiveFiber ← certifyPositiveReductionFiber checked
            let step ← scalarBinaryFunction ``max
            match flatReader? with
            | none =>
                mkAppM ``Lowering.reduceNonemptyFoldTensor #[
                  step, checked, hKind, hPositiveFiber, tensor]
            | some (read, hRead, _, _) =>
                mkAppM ``Lowering.reduceNonemptyFoldTensorFromFlat #[
                  step, checked, hKind, hPositiveFiber, tensor, read, hRead]
    | none => elaborateCustomReduction
  let result ← castTensorToCompactShape result outputShape
  exposePublicTensorType result outputShape expectedType?

end TorchLean.Tensor.Internal.Elab
