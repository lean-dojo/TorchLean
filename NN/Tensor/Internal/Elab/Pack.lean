/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Packing
public import NN.Tensor.Internal.Elab.Native.Pack.Dispatch
public import NN.Tensor.Internal.Elab.Native.Pack.Unpack
public meta import NN.Tensor.Internal.Syntax.Parser.Pack
public meta import NN.Tensor.Internal.Elab.Common -- shake: keep
public meta import NN.Tensor.Internal.Elab.Native.Pack -- shake: keep
public import NN.Tensor.Internal.Lowering.Pack -- shake: keep

/-!
# Elaboration of pack and unpack

This module implements heterogeneous-rank `pack` and `unpack`. It preserves
symbolic component dimensions, verifies fixed-axis agreement, proves exact
segment partitioning, and returns the dependent tensor family dictated by the
checked metadata.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab

open TorchLean.Tensor
open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta
open Impl

/-- Parse a pack pattern and attach source-precise diagnostics to failures. -/
private def parsedPackPattern (operation source : String) :
    TermElabM Syntax.PackPattern :=
  match Syntax.parsePackPattern source with
  | .error diagnostic =>
      throwPatternDiagnostic operation "pattern" source
        diagnostic.message diagnostic.span
  | .ok pattern => pure pattern

/-- Rebuild a concrete checker result as an explicit `CheckedPack` certificate. -/
private def reifyCheckedPack (checked : Check.CheckedPack) :
    TermElabM Expr :=
  buildCertificate ``Check.CheckedPack.mk #[
    Lean.toExpr checked.pattern,
    Lean.toExpr checked.inputShapes,
    Lean.toExpr checked.leadingShape,
    Lean.toExpr checked.trailingShape] #[] 3

/--
Construct symbolic pack metadata from component dimensions and certify the
shared fixed prefix and suffix.
-/
private def checkedPackFromDimensionsExpr
    (pattern : Syntax.PackPattern) (inputDimensions : List (List Expr))
    (leadingDimensions trailingDimensions : List Expr) :
    TermElabM Expr := do
  let inputShapeExpressions ←
    inputDimensions.mapM fun dimensions => shapeExpr dimensions
  let inputShapes ← shapesExpr inputShapeExpressions
  let leadingShape ← shapeExpr leadingDimensions
  let trailingShape ← shapeExpr trailingDimensions
  let patternExpr := Lean.toExpr pattern
  let leadingRank ← mkAppM ``List.length #[leadingShape]
  let hLeadingRank ←
    certifyGeneratedInvariant "that the leading pack region has the stated rank"
      (← mkEq leadingRank (mkNatLit pattern.before.length))
  let trailingRank ← mkAppM ``List.length #[trailingShape]
  let hTrailingRank ←
    certifyGeneratedInvariant "that the trailing pack region has the stated rank"
      (← mkEq trailingRank (mkNatLit pattern.after.length))
  let componentCount := mkNatLit inputShapeExpressions.length
  let componentType := mkApp (mkConst ``Fin) componentCount
  let inputShapesProposition ←
    withLocalDeclD `component componentType fun component => do
      let inputShape ← mkAppM ``List.get #[inputShapes, component]
      let starShape ←
        mkAppM ``Check.packStarShape #[patternExpr, inputShape]
      let leadingAndStar ←
        mkAppM ``List.append #[leadingShape, starShape]
      let decomposed ←
        mkAppM ``List.append #[leadingAndStar, trailingShape]
      let equality ← mkEq inputShape decomposed
      mkForallFVars #[component] equality
  let hInputShapes ←
    certifyGeneratedInvariant
      "that every packed tensor has the shared fixed dimensions"
      inputShapesProposition
  let checked ←
    buildCertificate ``Check.CheckedPack.mk #[
      patternExpr, inputShapes, leadingShape, trailingShape]
      #[hLeadingRank, hTrailingRank, hInputShapes] 0
  return checked

/--
Resolve the optional unpack dimension and retain every arithmetic fact needed
to justify the exact packed-axis partition.
-/
private def resolveSymbolicUnpackShapes (packedAxisLength : Expr)
    (requestedShapes : List (List (Option Expr))) :
    TermElabM (List (List Expr) × List (Expr × Expr)) := do
  let inferredShape? :=
    requestedShapes.find? fun shape =>
      shape.any fun dimension => dimension.isNone
  match inferredShape? with
  | none =>
      let resolvedShapes :=
        requestedShapes.map fun shape => shape.filterMap id
      let segmentLengths ←
        resolvedShapes.mapM fun shape => natProductExpr shape
      let totalLength ← natSumExpr segmentLengths
      let partition ← mkEq totalLength packedAxisLength
      let hPartition ←
        certifyGeneratedInvariant
          "that the requested unpack segments partition the packed axis"
          partition
      return (resolvedShapes, [(partition, hPartition)])
  | some inferredShape =>
      let knownShapes :=
        requestedShapes.filter fun shape =>
          shape.all fun dimension => dimension.isSome
      let knownSegmentLengths ←
        knownShapes.mapM fun shape =>
          natProductExpr (shape.filterMap id)
      let knownTotal ← natSumExpr knownSegmentLengths
      let knownDimensions := inferredShape.filterMap id
      let explicitResidual? ←
        explicitResidual? packedAxisLength knownSegmentLengths
      let explicitInferred? ←
        match explicitResidual? with
        | none => pure none
        | some residual =>
            explicitMissingFactor? residual knownDimensions
      if let some inferred := explicitInferred? then
        let resolvedShapes :=
          requestedShapes.map fun shape =>
            shape.map fun dimension => dimension.getD inferred
        let segmentLengths ←
          resolvedShapes.mapM fun shape => natProductExpr shape
        let totalLength ← natSumExpr segmentLengths
        let partition ← mkEq totalLength packedAxisLength
        let hPartition ←
          certifyGeneratedInvariant
            "that the explicitly inferred unpack segments partition the packed axis"
            partition
        return (resolvedShapes, [(partition, hPartition)])
      let knownTotalBound ← mkAppM ``LE.le #[knownTotal, packedAxisLength]
      let hKnownTotalBound ←
        certifyGeneratedInvariant
          "that the known unpack segments fit inside the packed axis"
          knownTotalBound
      let knownFactor ← natProductExpr knownDimensions
      let knownFactorPositive ←
        mkAppM ``LT.lt #[mkNatLit 0, knownFactor]
      let hKnownFactorPositive ←
        certifyGeneratedInvariant
          "that the known factor around the inferred unpack dimension is positive"
          knownFactorPositive
      let residual ←
        mkAppM ``Nat.sub #[packedAxisLength, knownTotal]
      let remainder ← mkAppM ``Nat.mod #[residual, knownFactor]
      let divisible ← mkEq remainder (mkNatLit 0)
      let hDivisible ←
        certifyGeneratedInvariant
          "that the remaining packed length is divisible by the known factor"
          divisible
      let inferred ← mkAppM ``Nat.div #[residual, knownFactor]
      let resolvedShapes :=
        requestedShapes.map fun shape =>
          shape.map fun dimension => dimension.getD inferred
      let segmentLengths ←
        resolvedShapes.mapM fun shape => natProductExpr shape
      let totalLength ← natSumExpr segmentLengths
      let partition ← mkEq totalLength packedAxisLength
      let hPartition ←
        certifyGeneratedInvariant
          "that the inferred unpack segments partition the packed axis"
          partition
      return (resolvedShapes,
        [(knownTotalBound, hKnownTotalBound),
         (knownFactorPositive, hKnownFactorPositive),
         (divisible, hDivisible),
         (partition, hPartition)])

/--
Construct symbolic unpack metadata while using a concrete rank witness for
the parser and structural checker.
-/
private def symbolicCheckedUnpackExpr (source : String)
    (pattern : Syntax.PackPattern) (packedDimensions : List Expr)
    (requestedShapes : List (List (Option Expr))) :
    TermElabM (Expr × Expr × List (Expr × Expr)) := do
  let expectedRank :=
    pattern.before.length + 1 + pattern.after.length
  let dummyRequestedShapes : Check.RequestedShapes :=
    requestedShapes.map fun shape =>
      shape.map fun dimension =>
        if dimension.isNone then -1 else 1
  if packedDimensions.length != expectedRank then
    match
        Check.checkUnpack pattern
          (List.replicate packedDimensions.length 1)
          dummyRequestedShapes with
    | .error diagnostic =>
        throwPatternDiagnostic "unpack" "shape" source
          diagnostic.message diagnostic.span
    | .ok _ =>
        throwError "internal error: symbolic unpack rank validation succeeded"
  let dummyResolvedShapes :=
    requestedShapes.map fun shape =>
      shape.map fun _ => 1
  let dummyPackedAxisLength :=
    (dummyResolvedShapes.map Shape.size).sum
  let dummyPackedShape :=
    (List.replicate expectedRank 1).set pattern.before.length
      dummyPackedAxisLength
  match Check.checkUnpack pattern dummyPackedShape dummyRequestedShapes with
  | .error diagnostic =>
      throwPatternDiagnostic "unpack" "shape" source
        diagnostic.message diagnostic.span
  | .ok _ => pure ()
  let packedAxisLength := packedDimensions[pattern.before.length]!
  let (resolvedStarShapes, arithmeticFacts) ←
    resolveSymbolicUnpackShapes packedAxisLength requestedShapes
  let leadingDimensions :=
    packedDimensions.take pattern.before.length
  let trailingDimensions :=
    packedDimensions.drop (pattern.before.length + 1)
  let inputDimensions :=
    resolvedStarShapes.map fun starShape =>
      leadingDimensions ++ starShape ++ trailingDimensions
  let checked ←
    checkedPackFromDimensionsExpr pattern inputDimensions leadingDimensions
      trailingDimensions
  let checkedResult ←
    mkAppM ``Check.checkUnpack #[
      Lean.toExpr pattern,
      Lean.toExpr dummyPackedShape,
      Lean.toExpr dummyRequestedShapes]
  let structuralEvidence ← extractSuccessfulResult checkedResult
  let structuralEvidenceType ← inferType structuralEvidence
  let checked :=
    mkLet `unpackStructuralCheck structuralEvidenceType structuralEvidence
      checked (nondep := true)
  let outputShape ← mkAppM ``Check.CheckedPack.output #[checked]
  let packedShape ← shapeExpr packedDimensions
  let outputShapeAgreement ← mkEq outputShape packedShape
  let hOutputShape ←
    certifyGeneratedInvariant
      "that unpack metadata reconstructs the packed tensor shape"
      outputShapeAgreement
  return (checked, hOutputShape, arithmeticFacts)

/--
Transport a packed tensor across the certified equality between its declared
shape and the shape reconstructed by unpack metadata.
-/
private def castTensorToShape
    (scalarType storage packedTensor shapeEquality : Expr) : MetaM Expr := do
  let equalityType ← withTransparency .reducible <| whnf (← inferType shapeEquality)
  let some (_, outputShape, _) := equalityType.eq?
    | throwError "internal error: expected an unpack shape equality"
  let tensorConstant := Lean.mkConst ``Rep [← getDecLevel scalarType]
  let outputTensorType :=
    mkAppN tensorConstant #[scalarType, outputShape, storage]
  let packedTensorType ← inferType packedTensor
  if ← withTransparency .reducible <|
      isDefEq packedTensorType outputTensorType then
    return packedTensor
  let shapeType ← mkAppM ``List #[mkConst ``Nat]
  let tensorFamily ←
    withLocalDeclD `shape shapeType fun shape => do
      let tensorType :=
        mkAppN tensorConstant #[scalarType, shape, storage]
      mkLambdaFVars #[shape] tensorType
  let reversedEquality ← mkAppM ``Eq.symm #[shapeEquality]
  let tensorTypeEquality ←
    mkAppM ``congrArg #[tensorFamily, reversedEquality]
  mkAppM ``cast #[tensorTypeEquality, packedTensor]

/--
Retain generated arithmetic certificates as nondependent lets in the emitted
term, so the kernel checks them without changing its result type.
-/
private def attachGeneratedFacts
    (facts : List (Expr × Expr)) (result : Expr) : Expr :=
  facts.foldr (init := result) fun fact body =>
    mkLet `generatedShapeFact fact.1 fact.2 body (nondep := true)

/--
Construct symbolic pack metadata and retain the successful structural checker
computation in the generated term.
-/
private def symbolicCheckedPackExpr (operation source : String)
    (pattern : Syntax.PackPattern) (inputDimensions : List (List Expr)) :
    TermElabM Expr := do
  let dummyShapes :=
    inputDimensions.map fun dimensions =>
      List.replicate dimensions.length 1
  match Check.checkPack pattern dummyShapes with
  | .error diagnostic =>
      throwPatternDiagnostic operation "shape" source
        diagnostic.message diagnostic.span
  | .ok _ => pure ()
  let firstDimensions :=
    inputDimensions.headD []
  let leadingDimensions :=
    firstDimensions.take pattern.before.length
  let trailingDimensions :=
    firstDimensions.drop
      (firstDimensions.length - pattern.after.length)
  let checked ←
    checkedPackFromDimensionsExpr pattern inputDimensions leadingDimensions
      trailingDimensions
  let patternExpr := Lean.toExpr pattern
  let checkedResult ←
    mkAppM ``Check.checkPack #[
      patternExpr, Lean.toExpr dummyShapes]
  let structuralEvidence ← extractSuccessfulResult checkedResult
  let structuralEvidenceType ← inferType structuralEvidence
  return mkLet `packStructuralCheck structuralEvidenceType structuralEvidence
    checked (nondep := true)

/-- Compute the compact public output shape of a pack expression. -/
private def packedOutputShapeExpr (pattern : Syntax.PackPattern)
    (inputDimensions : List (List Expr)) : MetaM Expr := do
  let firstDimensions := inputDimensions.headD []
  let leadingDimensions :=
    firstDimensions.take pattern.before.length
  let trailingDimensions :=
    firstDimensions.drop
      (firstDimensions.length - pattern.after.length)
  let starDimensions :=
    inputDimensions.map fun dimensions =>
      (dimensions.drop pattern.before.length).take
        (dimensions.length - Check.packFixedRank pattern)
  let segmentLengths ← starDimensions.mapM natProductExpr
  let packedAxisLength ← natSumExpr segmentLengths
  let packedAxisLength ← withTransparency .all <| whnf packedAxisLength
  shapeExpr (leadingDimensions ++ packedAxisLength :: trailingDimensions)

/--
Construct the user-facing pack result. Generated checked plans remain in the
value, while the inferred type displays only the public tensor shape and
ordinary natural-number metadata.
-/
private def exposePackResult
    (scalarType storage packedTensor componentShapes outputShape : Expr)
    (expectedType? : Option Expr) : TermElabM Expr := do
  let packedTensor ← castTensorToCompactShape packedTensor outputShape
  let publicOutputShape ← mkAppM ``Spec.Shape.ofList #[outputShape]
  let packedTensor ←
    if expectedType?.isSome then
      pure packedTensor
    else
      exposePublicTensorType packedTensor outputShape none
  let publicComponentShapes ←
    mkAppM ``List.map #[mkConst ``Spec.Shape.ofList, componentShapes]
  let publicComponentShapes ←
    if expectedType?.isSome then
      pure publicComponentShapes
    else
      let metadataType ← mkAppM ``List #[mkConst ``Spec.Shape]
      withTransparency .reducible <|
        mkExpectedTypeHint publicComponentShapes metadataType
  let result ←
    mkAppOptM ``TorchLean.Tensor.Packed.mk #[
      some scalarType, some publicOutputShape, some publicComponentShapes, some storage,
      some packedTensor]
  ensureHasType expectedType? result

/--
Elaborate a nonempty heterogeneous component family to verified packing,
returning the packed tensor together with exact star-shape metadata.
-/
@[term_elab TorchLean.Tensor.packStx]
def elabPack : TermElab := fun stx expectedType? => withRef stx do
  let `(pack $tensorSyntax:term,* $sourceSyntax:str) := stx
    | throwUnsupportedSyntax
  withCommonScalarFamily "pack" "component" tensorSyntax.getElems fun
      inputTensors inputShapes scalarType storage => do
    let inputShapeExpressions ←
      inputShapes.mapM fun dimensions => shapeExpr dimensions
    let source := sourceSyntax.getString
    let pattern ← parsedPackPattern "pack" source
    let concreteInputShapes? ← concreteShapes? inputShapes
    let outputShape ← packedOutputShapeExpr pattern inputShapes
    let inputFamily ←
      buildTensorFamily scalarType storage inputShapeExpressions inputTensors
    match concreteInputShapes? with
    | none =>
        let checked ←
          symbolicCheckedPackExpr "pack" source pattern inputShapes
        let packedTensor ←
          mkAppOptM ``Lowering.packTensor #[
            some scalarType, some storage, some checked, some inputFamily]
        let metadata ←
          mkAppM ``Check.CheckedPack.metadata #[checked]
        let result ←
          exposePackResult scalarType storage packedTensor metadata outputShape
            expectedType?
        synthesizeSyntheticMVarsNoPostponing
        instantiateMVars result
    | some concreteInputShapes =>
        let checkedValue ←
          match Check.checkPack pattern concreteInputShapes with
          | .error diagnostic =>
              throwPatternDiagnostic "pack" "shape" source
                diagnostic.message diagnostic.span
          | .ok checked => pure checked
        let compilerChecked ← reifyCheckedPack checkedValue
        let checkedType ← inferType compilerChecked
        withLetDecl `checkedPack checkedType compilerChecked fun checked => do
          let packedTensor ←
            match ←
                compileNativePack? compilerChecked checked inputFamily
                  checkedValue with
            | some packedTensor => pure packedTensor
            | none =>
              mkAppOptM ``Lowering.packTensor #[
                  some scalarType, some storage, some checked,
                  some inputFamily]
          let metadata ←
            mkAppM ``Check.CheckedPack.metadata #[checked]
          let result ←
            exposePackResult scalarType storage packedTensor metadata outputShape
              expectedType?
          synthesizeSyntheticMVarsNoPostponing
          let result ← instantiateMVars result
          let checkedResult ←
            mkAppM ``Check.checkPack #[
              Lean.toExpr pattern, Lean.toExpr concreteInputShapes]
          let result ←
            attachCheckerAgreement checkedResult checked result
          mkLetFVars (generalizeNondepLet := false) #[checked] result

/--
Elaborate unpack metadata, infer its optional `-1` dimension, and return the
heterogeneous component family certified by the reconstructed packed shape.
-/
@[term_elab TorchLean.Tensor.unpackStx]
def elabUnpack : TermElab := fun stx expectedType? => withRef stx do
  let `(unpack $packedSyntax:term $sourceSyntax:str) := stx
    | throwUnsupportedSyntax
  let packed ← elabTermAndSynthesize packedSyntax none
  let packedType ← instantiateMVars (← inferType packed)
  let packedType ← withTransparency .reducible <| whnf packedType
  unless packedType.isAppOfArity ``TorchLean.Tensor.Packed 4 do
    throwErrorAt packedSyntax
      "expected a value returned by `pack`, but the term has type\
        {indentExpr packedType}"
  let packedArguments := packedType.getAppArgs
  let scalarType := packedArguments[0]!
  let packedShape := packedArguments[1]!
  let componentShapes := packedArguments[2]!
  let storage := packedArguments[3]!
  let packedTensor ←
    mkAppOptM ``TorchLean.Tensor.Packed.tensor #[
      some scalarType, some packedShape, some componentShapes, some storage,
      some packed]
  let packedShapeList ← mkAppM ``Spec.Shape.toList #[packedShape]
  let some dimensions ← staticListElements? packedShapeList
    | throwErrorAt packedSyntax
        "the packed tensor shape must have a statically known list structure"
  let some naturalShapeExpressions ← staticListElements? componentShapes
    | throwErrorAt packedSyntax
        "packed component shapes must have a statically known list structure"
  let integerListType ← mkAppM ``List #[mkConst ``Int]
  let mut integerShapeExpressions : List Expr := []
  let mut symbolicRequestedShapes : List (List (Option Expr)) := []
  for naturalShape in naturalShapeExpressions do
    let naturalShapeList ← mkAppM ``Spec.Shape.toList #[naturalShape]
    let some shapeDimensions ← staticListElements? naturalShapeList
      | throwErrorAt packedSyntax
          "every packed component shape must have a statically known list structure"
    let integerDimensions ←
      shapeDimensions.mapM fun dimension =>
        mkAppM ``Int.ofNat #[dimension]
    let integerShape ←
      mkListLit (mkConst ``Int) integerDimensions
    integerShapeExpressions := integerShapeExpressions.concat integerShape
    symbolicRequestedShapes :=
      symbolicRequestedShapes.concat (shapeDimensions.map some)
  let requestedShapes ←
    mkListLit integerListType integerShapeExpressions
  let source := sourceSyntax.getString
  let pattern ← parsedPackPattern "unpack" source
  let concretePackedShape? ← concreteNatExpressions? dimensions
  let concreteRequestedShapes? ←
    concreteRequestedShapes? requestedShapes
  let (checked, hOutputShape, arithmeticFacts,
      checkedResult?, checkedValue?) ←
    match concretePackedShape?, concreteRequestedShapes? with
    | some packedShape, some concreteRequestedShapes =>
        let checkedValue ←
          match Check.checkUnpack pattern packedShape concreteRequestedShapes with
          | .error diagnostic =>
              throwPatternDiagnostic "unpack" "shape" source
                diagnostic.message diagnostic.span
          | .ok checked => pure checked
        let checked ← reifyCheckedPack checkedValue
        let outputShape ←
          mkAppM ``Check.CheckedPack.output #[checked]
        let packedShapeExpr ← shapeExpr dimensions
        let outputShapeAgreement ← mkEq outputShape packedShapeExpr
        let hOutputShape ←
          certifyGeneratedInvariant
            "that checked unpack metadata reconstructs the packed shape"
            outputShapeAgreement
        let checkedResult ←
          mkAppM ``Check.checkUnpack #[
            Lean.toExpr pattern, Lean.toExpr packedShape, requestedShapes]
        pure
          (checked, hOutputShape, [], some checkedResult,
            some checkedValue)
    | _, _ =>
        let (checked, hOutputShape, arithmeticFacts) ←
          symbolicCheckedUnpackExpr source pattern dimensions
            symbolicRequestedShapes
        pure (checked, hOutputShape, arithmeticFacts, none, none)
  let packedTensor ←
    castTensorToShape scalarType storage packedTensor hOutputShape
  let result ←
    match checkedValue? with
    | some checkedValue =>
        match ←
            compileNativeUnpack? scalarType storage checked packedTensor
              checkedValue with
        | some result => pure result
        | none =>
            mkAppOptM ``Lowering.unpackTensor #[
              some scalarType, some storage, some checked, some packedTensor]
    | none =>
        mkAppOptM ``Lowering.unpackTensor #[
          some scalarType, some storage, some checked, some packedTensor]
  let result ← ensureHasType expectedType? result
  synthesizeSyntheticMVarsNoPostponing
  let result ← instantiateMVars result
  let result := attachGeneratedFacts arithmeticFacts result
  match checkedResult? with
  | none => pure result
  | some checkedResult =>
      attachCheckerAgreement checkedResult checked result

end TorchLean.Tensor.Internal.Elab
