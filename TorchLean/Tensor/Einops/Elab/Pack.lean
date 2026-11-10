/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Elab.Pack.Result
public import TorchLean.Tensor.Einops.Elab.Pack.Dispatch
public import TorchLean.Tensor.Einops.Elab.Pack.Unpack
public meta import TorchLean.Tensor.Einops.Elab.Pack.Dispatch -- shake: keep
public meta import TorchLean.Tensor.Einops.Elab.Pack.Unpack -- shake: keep
public import TorchLean.Tensor.Einops.Elab.Syntax

/-!
# Elaboration of pack and unpack

`pack` and `unpack` accept components of different ranks. The elaborators keep symbolic
component dimensions, verify that the fixed axes agree, prove that the segments partition the
packed axis, and return the dependent tensor family that the checked metadata dictates.

Scalar promotion picks one common scalar type before the family is built, and `unpack` returns
the components at that common type with their original shapes. Symbolic dimensions need a
statically known shape-list structure.
-/

public meta section

namespace TorchLean.Einops.Elab

open TorchLean.Tensor
open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/-- Parse a pack pattern, reporting failures with source-located diagnostics. -/
private def parsedPackPattern (operation source : String) :
    TermElabM PackPattern :=
  match PackPattern.parse source with
  | .error diagnostic =>
      throwPatternDiagnostic operation "pattern" source
        diagnostic.message diagnostic.span
  | .ok pattern => pure pattern

/-- Rebuild a concrete checker result as a `Einops.Pack` term, deciding its proof fields. -/
private def reifyCheckedPack (checked : Pack) :
    TermElabM Expr :=
  buildCertificate ``Pack.mk #[
    Lean.toExpr checked.pattern,
    Lean.toExpr checked.inputShapes,
    Lean.toExpr checked.leadingShape,
    Lean.toExpr checked.trailingShape] #[] 3

/--
Construct symbolic pack metadata from component dimensions and certify the
shared fixed prefix and suffix.
-/
private def checkedPackFromDimensionsExpr
    (pattern : PackPattern) (inputDimensions : List (List Expr))
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
        mkAppM ``PackPattern.starShape #[patternExpr, inputShape]
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
    buildCertificate ``Pack.mk #[
      patternExpr, inputShapes, leadingShape, trailingShape]
      #[hLeadingRank, hTrailingRank, hInputShapes] 0
  return checked

/--
Construct symbolic unpack metadata while using a concrete rank witness for
the parser and structural checker.
-/
private def symbolicCheckedUnpackExpr (source : String)
    (pattern : PackPattern) (packedDimensions : List Expr)
    (requestedShapes : List (List Expr)) :
    TermElabM (Expr × Expr × List (Expr × Expr)) := do
  let expectedRank :=
    pattern.before.length + 1 + pattern.after.length
  let dummyRequestedShapes : RequestedShapes :=
    requestedShapes.map fun shape =>
      shape.map fun _ => 1
  if packedDimensions.length != expectedRank then
    match
        PackPattern.checkUnpack pattern
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
  match PackPattern.checkUnpack pattern dummyPackedShape dummyRequestedShapes with
  | .error diagnostic =>
      throwPatternDiagnostic "unpack" "shape" source
        diagnostic.message diagnostic.span
  | .ok _ => pure ()
  let packedAxisLength := packedDimensions[pattern.before.length]!
  let segmentLengths ← requestedShapes.mapM fun shape => natProductExpr shape
  let totalLength ← natSumExpr segmentLengths
  let partition ← mkEq totalLength packedAxisLength
  let hPartition ←
    certifyGeneratedInvariant
      "that the unpack metadata partitions the packed axis" partition
  let leadingDimensions :=
    packedDimensions.take pattern.before.length
  let trailingDimensions :=
    packedDimensions.drop (pattern.before.length + 1)
  let inputDimensions :=
    requestedShapes.map fun starShape =>
      leadingDimensions ++ starShape ++ trailingDimensions
  let checked ←
    checkedPackFromDimensionsExpr pattern inputDimensions leadingDimensions
      trailingDimensions
  let checkedResult ←
    mkAppM ``PackPattern.checkUnpack #[
      Lean.toExpr pattern,
      Lean.toExpr dummyPackedShape,
      Lean.toExpr dummyRequestedShapes]
  let structuralEvidence ← extractSuccessfulResult checkedResult
  let structuralEvidenceType ← inferType structuralEvidence
  let checked :=
    mkLet `unpackStructuralCheck structuralEvidenceType structuralEvidence
      checked (nondep := true)
  let outputShape ← mkAppM ``Pack.output #[checked]
  let packedShape ← shapeExpr packedDimensions
  let outputShapeAgreement ← mkEq outputShape packedShape
  let hOutputShape ←
    certifyGeneratedInvariant
      "that unpack metadata reconstructs the packed tensor shape"
      outputShapeAgreement
  return (checked, hOutputShape, [(partition, hPartition)])

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
    (pattern : PackPattern) (inputDimensions : List (List Expr)) :
    TermElabM Expr := do
  let dummyShapes :=
    inputDimensions.map fun dimensions =>
      List.replicate dimensions.length 1
  match PackPattern.check pattern dummyShapes with
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
    mkAppM ``PackPattern.check #[
      patternExpr, Lean.toExpr dummyShapes]
  let structuralEvidence ← extractSuccessfulResult checkedResult
  let structuralEvidenceType ← inferType structuralEvidence
  return mkLet `packStructuralCheck structuralEvidenceType structuralEvidence
    checked (nondep := true)

/-- Compute the compact public output shape of a pack expression. -/
private def packedOutputShapeExpr (pattern : PackPattern)
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
        (dimensions.length - PackPattern.explicitRank pattern)
  let segmentLengths ← starDimensions.mapM natProductExpr
  let packedAxisLength ← natSumExpr segmentLengths
  let packedAxisLength ← withTransparency .all <| whnf packedAxisLength
  shapeExpr (leadingDimensions ++ packedAxisLength :: trailingDimensions)

/--
Construct the user-facing pack result. The value carries the generated checked plans, while
the inferred type shows only the public tensor shape and ordinary natural-number metadata.
-/
private def exposePackResult
    (scalarType storage packedTensor componentShapes outputShape : Expr)
    (expectedType? : Option Expr) : TermElabM Expr := do
  let packedTensor ← castTensorToCompactShape packedTensor outputShape
  let packedTensor ←
    if expectedType?.isSome then
      pure packedTensor
    else
      exposePublicTensorType packedTensor outputShape none
  let publicComponentShapes ←
    if expectedType?.isSome then
      pure componentShapes
    else
      let metadataType ← mkAppM ``List #[mkConst ``Shape]
      withTransparency .reducible <|
        mkExpectedTypeHint componentShapes metadataType
  let result ←
    mkAppOptM ``TorchLean.Tensor.Packed.mk #[
      some scalarType, some outputShape, some publicComponentShapes, some storage,
      some packedTensor]
  ensureHasType expectedType? result

/--
Elaborate a verified pack of a nonempty component family whose members may differ in rank and
scalar type. The result holds the packed tensor together with the star shape of each component.
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
          mkAppOptM ``Lowering.pack #[
            some scalarType, some storage, some checked, some inputFamily]
        let metadata ←
          mkAppM ``Pack.metadata #[checked]
        let result ←
          exposePackResult scalarType storage packedTensor metadata outputShape
            expectedType?
        synthesizeSyntheticMVarsNoPostponing
        instantiateMVars result
    | some concreteInputShapes =>
        let checkedValue ←
          match PackPattern.check pattern concreteInputShapes with
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
              mkAppOptM ``Lowering.pack #[
                  some scalarType, some storage, some checked,
                  some inputFamily]
          let metadata ←
            mkAppM ``Pack.metadata #[checked]
          let result ←
            exposePackResult scalarType storage packedTensor metadata outputShape
              expectedType?
          synthesizeSyntheticMVarsNoPostponing
          let result ← instantiateMVars result
          let checkedResult ←
            mkAppM ``PackPattern.check #[
              Lean.toExpr pattern, Lean.toExpr concreteInputShapes]
          let result ←
            attachCheckerAgreement checkedResult checked result
          mkLetFVars (generalizeNondepLet := false) #[checked] result

/--
Read the component shapes stored in the type of a `pack` result and return the component
family, certified by the proof that those shapes reconstruct the packed shape.
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
  let some dimensions ← staticListElements? packedShape
    | throwErrorAt packedSyntax
        "the packed tensor shape must have a statically known list structure"
  let some naturalShapeExpressions ← staticListElements? componentShapes
    | throwErrorAt packedSyntax
        "packed component shapes must have a statically known list structure"
  let integerListType ← mkAppM ``List #[mkConst ``Int]
  let mut integerShapeExpressions : List Expr := []
  let mut symbolicRequestedShapes : List (List Expr) := []
  for naturalShape in naturalShapeExpressions do
    let some shapeDimensions ← staticListElements? naturalShape
      | throwErrorAt packedSyntax
          "every packed component shape must have a statically known list structure"
    let integerDimensions ←
      shapeDimensions.mapM fun dimension =>
        mkAppM ``Int.ofNat #[dimension]
    let integerShape ←
      mkListLit (mkConst ``Int) integerDimensions
    integerShapeExpressions := integerShapeExpressions.concat integerShape
    symbolicRequestedShapes :=
      symbolicRequestedShapes.concat shapeDimensions
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
          match PackPattern.checkUnpack pattern packedShape concreteRequestedShapes with
          | .error diagnostic =>
              throwPatternDiagnostic "unpack" "shape" source
                diagnostic.message diagnostic.span
          | .ok checked => pure checked
        let checked ← reifyCheckedPack checkedValue
        let outputShape ←
          mkAppM ``Pack.output #[checked]
        let packedShapeExpr ← shapeExpr dimensions
        let outputShapeAgreement ← mkEq outputShape packedShapeExpr
        let hOutputShape ←
          certifyGeneratedInvariant
            "that checked unpack metadata reconstructs the packed shape"
            outputShapeAgreement
        let checkedResult ←
          mkAppM ``PackPattern.checkUnpack #[
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
    mkAppM ``TorchLean.Tensor.castShape #[packedTensor, ← mkAppM ``Eq.symm #[hOutputShape]]
  let result ←
    match checkedValue? with
    | some checkedValue =>
        match ←
            compileNativeUnpack? scalarType storage checked packedTensor
              checkedValue with
        | some result => pure result
        | none =>
            mkAppOptM ``Lowering.unpack #[
              some scalarType, some storage, some checked, some packedTensor]
    | none =>
        mkAppOptM ``Lowering.unpack #[
          some scalarType, some storage, some checked, some packedTensor]
  let result ← ensureHasType expectedType? result
  synthesizeSyntheticMVarsNoPostponing
  let result ← instantiateMVars result
  let result := attachGeneratedFacts arithmeticFacts result
  match checkedResult? with
  | none => pure result
  | some checkedResult =>
      attachCheckerAgreement checkedResult checked result

end TorchLean.Einops.Elab
