/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Laws.PackIndex
public meta import Mathlib.Algebra.GroupWithZero.Nat
public import NN.Tensor.Internal.Elab.Einsum.Kernel.Index
public meta import NN.Tensor.Internal.Elab.Native.Index
public meta import NN.Tensor.Internal.Elab.Native.Pull -- shake: keep

/-!
# Certified native pack dispatch

Concrete pack plans compile to one native output loop and a balanced segment
dispatcher. Every leaf reads one component buffer through the verified inverse
unpack index, so component count and tensor rank do not create semantic cases.

Symbolic or nonportable shapes retain the general coordinate lowering.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/-- One nonempty concrete component segment used by native pack dispatch. -/
private abbrev PackSegment :=
  Nat × (Nat × (Nat × Shape))

/-- Name one generated native pack callback for compact elaboration. -/
private def sealNativePackCallback (callback : Expr) : TermElabM Expr := do
  let name ← mkAuxName `_einops_native_pack_callback
  mkAuxDefinitionFor name callback (zetaDelta := false)

/--
Name the completed native packed tensor before composing its semantic proof.

The callback and its pointwise certificate are already compact constants, but
the dependent tensor builder still carries both through its result type. A
separate executable definition keeps the public kernel wrapper shallow while
retaining the same native loop.
-/
private def sealNativePackImplementation (implementation : Expr) :
    TermElabM Expr := do
  let name ← mkAuxName `_einops_native_pack_implementation
  mkAuxDefinitionFor name implementation (zetaDelta := false)

/-- Name a proof-only pack reference so reconstruction theorems stay shallow. -/
private def sealNativePackReference (reference : Expr) : TermElabM Expr := do
  let name ← mkAuxName `_einops_native_pack_reference
  mkAuxDefinitionFor name reference (zetaDelta := false)

/--
Compile one leaf of a native pack dispatcher.

The leaf's path assumptions identify its component segment. The generated
source index is direct row-major arithmetic, while the returned theorem
identifies the direct component read with the independent pack denotation.
-/
private def compileNativePackLeaf
    (checked inputFamily outputBuffer outputSize outputIndexNat outputFin
      hNativeOutputIndex positionNat hPositionDecomposition : Expr)
    (checkedValue : Check.CheckedPack)
    (componentCount trailingSize packedAxisLength : Nat)
    (segment : PackSegment) (pathAssumptions : Array Expr)
    (expectedValue : Expr) :
    TermElabM (Expr × Expr) := do
  let (componentIndex, offset, segmentLength, starShapeValue) := segment
  let componentType ← mkAppM ``Fin #[mkNatLit componentCount]
  let component ← mkNumeral componentType componentIndex
  let specializedChecked ← zetaReduce checked
  let inputShapes ←
    mkAppM ``Check.CheckedPack.inputShapes #[specializedChecked]
  let inputShape ← mkAppM ``List.get #[inputShapes, component]
  let inputSize ← mkAppM ``Shape.size #[inputShape]
  let literalInputSize :=
    mkNatLit (Shape.size checkedValue.inputShapes[componentIndex]!)
  let leadingSize := Shape.size checkedValue.leadingShape
  let literalLeadingSize := mkNatLit leadingSize
  let literalOutputSize :=
    mkNatLit (leadingSize * (trailingSize * packedAxisLength))
  let literalStarShape := Lean.toExpr starShapeValue
  let literalTrailingShape := Lean.toExpr checkedValue.trailingShape
  let literalPackedAxisLength := mkNatLit packedAxisLength
  let literalOffset := mkNatLit offset
  let checkedStarShape ←
    mkAppM ``Check.CheckedPack.starShape #[
      specializedChecked, component]
  let checkedTrailingShape ←
    mkAppM ``Check.CheckedPack.trailingShape #[specializedChecked]
  let checkedPackedAxisLength ←
    mkAppM ``Check.CheckedPack.packedAxisLength #[specializedChecked]
  let hStarShape ←
    certifyGeneratedInvariant
      "that a native pack segment has the checked star shape"
      (← mkEq literalStarShape checkedStarShape)
  let hTrailingShape ←
    certifyGeneratedInvariant
      "that native pack uses the checked trailing shape"
      (← mkEq literalTrailingShape checkedTrailingShape)
  let hPackedAxisLength ←
    certifyGeneratedInvariant
      "that native pack uses the checked packed-axis length"
      (← mkEq literalPackedAxisLength checkedPackedAxisLength)
  let componentValue ← mkAppM ``Fin.val #[component]
  let checkedSegmentLengths ←
    mkAppM ``Check.CheckedPack.segmentLengths #[specializedChecked]
  let checkedPrefix ←
    mkAppM ``List.take #[componentValue, checkedSegmentLengths]
  let checkedPrefixSum ← mkAppM ``List.sum #[checkedPrefix]
  let hLiteralPrefix ←
    certifyGeneratedInvariant
      "that a native pack segment has the checked prefix offset"
      (← mkEq literalOffset checkedPrefixSum)
  let hCheckedPrefix ←
    mkAppM ``Lowering.Pack.Impl.componentOffset_eq_sum_take #[
      specializedChecked, component]
  let hOffset ←
    mkAppM ``Eq.trans #[
      hLiteralPrefix, ← mkAppM ``Eq.symm #[hCheckedPrefix]]
  let directIndex ←
    mkAppM ``Lowering.Pack.Impl.packDirectComponentIndex #[
      literalStarShape, literalTrailingShape,
      literalPackedAxisLength, literalOffset, outputIndexNat]
  let (compiledIndexValue, hCompiledIndexValue) ←
    compileRowMajorIndex directIndex #[
      ``Lowering.Pack.Impl.packDirectComponentIndex]
  let hLower ←
    certifyNativeArithmeticFact
      "that a packed output lies after its selected component prefix"
      (← mkLE literalOffset positionNat)
      pathAssumptions
  let hUpper ←
    certifyNativeArithmeticFact
      "that a packed output lies before the end of its selected component"
      (← mkLT positionNat (mkNatLit (offset + segmentLength)))
      pathAssumptions
  let hTrailing ←
    mkDecideProof (← mkLT (mkNatLit 0) (mkNatLit trailingSize))
  let hLiteralOutputSize ←
    certifyGeneratedInvariant
      "that native pack uses the checked output flat size"
      (← mkEq literalOutputSize outputSize)
  let hCheckedOutputBound ← mkAppM ``Fin.isLt #[outputFin]
  let hLiteralOutputBound ←
    nativeLoopIndexBound outputIndexNat hLiteralOutputSize
      hCheckedOutputBound
  let hDirectInputBound ←
    mkAppM ``Lowering.Pack.Impl.packDirectComponentIndex_lt #[
      literalStarShape, literalTrailingShape,
      literalPackedAxisLength, literalOffset,
      outputIndexNat, literalLeadingSize,
      hTrailing, hLiteralOutputBound, hLower, hUpper]
  let hDirectInputBound ←
    withTransparency .all <|
      mkExpectedTypeHint hDirectInputBound
        (← mkLT directIndex literalInputSize)
  let hLiteralInputBound ←
    indexBoundFromValueEquality literalInputSize
      hCompiledIndexValue hDirectInputBound
  let hInputSize ←
    certifyGeneratedInvariant
      "that a native pack component has its checked flat size"
      (← mkEq literalInputSize inputSize)
  let inputBoundPredicate ←
    withLocalDeclD `bound (mkConst ``Nat) fun bound => do
      mkLambdaFVars #[bound] (← mkLT compiledIndexValue bound)
  let hInputBoundEquality ←
    mkAppM ``congrArg #[inputBoundPredicate, hInputSize]
  let hInputBound ←
    mkAppM ``Eq.mp #[hInputBoundEquality, hLiteralInputBound]
  let inputIndex ←
    mkAppOptM ``Fin.mk #[
      some inputSize, some compiledIndexValue, some hInputBound]
  let hInputValue ←
    mkEqRefl compiledIndexValue
  let familyTensor ←
    withTransparency .all <| whnf (mkApp inputFamily component)
  let useNativeInputIndex ←
    supportsNativeInputIndex <|
      checkedValue.inputShapes[componentIndex]!.map Lean.toExpr
  let (sourceValue, hSourceValue) ←
    compileInputRead familyTensor inputSize
      compiledIndexValue compiledIndexValue inputIndex hInputValue
      useNativeInputIndex (some hInputBound)
      #[hLiteralOutputBound, hNativeOutputIndex, hPositionDecomposition,
        hLower, hUpper, hLiteralInputBound]
  let hIndex ←
    mkAppM
      ``Lowering.Pack.Impl.unpackFlatIndex_eq_of_specializedPackDirectComponentIndex #[
      specializedChecked, component, inputIndex, outputFin,
      literalStarShape, literalTrailingShape,
      literalPackedAxisLength, literalOffset,
      hStarShape, hTrailingShape, hPackedAxisLength, hOffset,
      hCompiledIndexValue, hTrailing, hLower, hUpper]
  let hSemanticValue ←
    mkAppM ``Lowering.Pack.Impl.getFlat_denotePack_of_unpackFlatIndex_eq #[
        specializedChecked, inputFamily, component,
        inputIndex, outputFin, hIndex]
  let hValue ← mkAppM ``Eq.trans #[hSourceValue, hSemanticValue]
  let hValue ←
    withTransparency .all <|
      mkExpectedTypeHint hValue (← mkEq sourceValue expectedValue)
  let sourceValue := sourceValue.consumeMData
  unless sourceValue.isAppOfArity ``Rep.getFlatUSize 6 do
    throwError
      "internal error: a native pack leaf is not a native tensor read"
  let sourceArguments := sourceValue.getAppArgs
  let sourceTensor := sourceArguments[3]!
  let sourceIndex := sourceArguments[4]!
  let hSourceIndex := sourceArguments[5]!
  let nextOutput ←
    mkAppM ``nativeTensorCopyAt #[
      sourceTensor, sourceIndex, hSourceIndex, outputBuffer]
  let hNextOutput ←
    mkAppM ``nativeTensorCopyAt_eq_push_of_getFlatUSize_eq #[
      sourceTensor, sourceIndex, hSourceIndex, outputBuffer,
      expectedValue, hValue]
  return (nextOutput, hNextOutput)

/--
Compile a balanced segment decision tree for one native packed output.

Every recursive split compares the packed-axis position with the first offset
of the right subtree. Leaves therefore receive enough path facts to certify
their component interval and native subtraction.
-/
private partial def compileNativePackDispatch
    (checked inputFamily outputBuffer outputSize outputIndexNat outputFin
      hNativeOutputIndex nativePosition positionNat hNativePosition
      hPositionDecomposition : Expr)
    (checkedValue : Check.CheckedPack)
    (componentCount trailingSize packedAxisLength : Nat)
    (segments : List PackSegment) (pathAssumptions : Array Expr)
    (expectedValue : Expr) :
    TermElabM (Expr × Expr) := do
  match segments with
  | [] =>
      throwError
        "internal error: native pack dispatch has no reachable component"
  | [segment] =>
      compileNativePackLeaf checked inputFamily outputBuffer outputSize
        outputIndexNat outputFin hNativeOutputIndex positionNat
        hPositionDecomposition
        checkedValue componentCount trailingSize
        packedAxisLength segment pathAssumptions expectedValue
  | _ =>
      let split := segments.length / 2
      let left := segments.take split
      let right := segments.drop split
      let threshold := right.head!.2.1
      let some (nativeThreshold, hNativeThreshold) ←
          certifiedNativeIndexValue? (mkNatLit threshold) #[]
        | throwError
            "internal error: a native pack segment threshold is not portable"
      let condition ← mkLT nativePosition nativeThreshold
      let leftResult : Expr × Expr ←
        withLocalDeclD `hLeft condition fun hLeft => do
          let hLeftNat ←
            mkAppM ``nat_lt_of_native_lt_of_eq #[
              nativePosition, nativeThreshold,
              positionNat, mkNatLit threshold,
              hNativePosition, hNativeThreshold, hLeft]
          let (leftValue, hLeftValue) ←
            compileNativePackDispatch checked inputFamily outputBuffer outputSize
              outputIndexNat outputFin hNativeOutputIndex
              nativePosition positionNat hNativePosition
              hPositionDecomposition checkedValue
              componentCount trailingSize packedAxisLength left
              (pathAssumptions.push hLeftNat) expectedValue
          let leftFunction ← mkLambdaFVars #[hLeft] leftValue
          let hLeftFunction ← mkLambdaFVars #[hLeft] hLeftValue
          return (leftFunction, hLeftFunction)
      let leftFunction := leftResult.1
      let hLeftFunction := leftResult.2
      let rightResult : Expr × Expr ←
        withLocalDeclD `hRight (mkNot condition) fun hRight => do
          let hRightNat ←
            mkAppM ``nat_le_of_not_native_lt_of_eq #[
              nativePosition, nativeThreshold,
              positionNat, mkNatLit threshold,
              hNativePosition, hNativeThreshold, hRight]
          let (rightValue, hRightValue) ←
            compileNativePackDispatch checked inputFamily outputBuffer outputSize
              outputIndexNat outputFin hNativeOutputIndex
              nativePosition positionNat hNativePosition
              hPositionDecomposition checkedValue
              componentCount trailingSize packedAxisLength right
              (pathAssumptions.push hRightNat) expectedValue
          let rightFunction ← mkLambdaFVars #[hRight] rightValue
          let hRightFunction ← mkLambdaFVars #[hRight] hRightValue
          return (rightFunction, hRightFunction)
      let rightFunction := rightResult.1
      let hRightFunction := rightResult.2
      let value ←
        mkAppM ``dite #[condition, leftFunction, rightFunction]
      let expectedOutput ←
        mkAppM ``Storage.push #[outputBuffer, expectedValue]
      let hValue ←
        mkAppM ``dite_eq_of_branch_eq #[
          condition, leftFunction, rightFunction, expectedOutput,
          hLeftFunction, hRightFunction]
      return (value, hValue)

/- Compile one concrete pack plan to a native output loop. -/
private def compileNativePackCore?
    (compilerChecked checked inputFamily : Expr)
    (checkedValue : Check.CheckedPack) :
    TermElabM (Option Expr) := do
  let componentCount := checkedValue.inputShapes.length
  let trailingSize := Shape.size checkedValue.trailingShape
  let packedAxisLength := checkedValue.segmentLengths.sum
  let segments : List PackSegment :=
    checkedValue.segmentLengths.zipIdx.filterMap fun
      (segmentLength, componentIndex) =>
        if segmentLength = 0 then
          none
        else
          some
            (componentIndex,
              (checkedValue.segmentLengths.take componentIndex).sum,
              segmentLength,
              Check.packStarShape checkedValue.pattern
                checkedValue.inputShapes[componentIndex]!)
  if trailingSize = 0 || packedAxisLength = 0 || segments.isEmpty then
    return none
  let specializedChecked ← zetaReduce compilerChecked
  let outputShape ←
    mkAppM ``Check.CheckedPack.output #[specializedChecked]
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let some (outputBound, hOutputBound) ← nativeLoopBound? outputSize
    | return none
  let rawCompilerReference ←
    mkAppM ``Semantics.denotePack #[specializedChecked, inputFamily]
  let compilerReference ← sealNativePackReference rawCompilerReference
  let values ← mkAppM ``Rep.flatten #[compilerReference]
  let componentType ← mkAppM ``Fin #[mkNatLit componentCount]
  let firstComponent ← mkNumeral componentType segments.head!.1
  let firstTensor ←
    withTransparency .all <| whnf (mkApp inputFamily firstComponent)
  let firstBuffer ← mkAppM ``Rep.buffer #[firstTensor]
  let outputBufferType ← inferType firstBuffer
  let some (nativeStep, hStep) ←
    withLocalDeclD `output outputBufferType fun outputBuffer => do
      withLocalDeclD `outputIndex (mkConst ``USize) fun outputIndex => do
        let outputIndexNat ← mkAppM ``USize.toNat #[outputIndex]
        let outputIndexBoundType ← mkLT outputIndexNat outputSize
        withLocalDeclD `hOutputIndex outputIndexBoundType fun hOutputIndex => do
          let hNativeOutputIndex ←
            nativeLoopIndexBound outputIndexNat hOutputBound hOutputIndex
          let outputFin ←
            mkAppOptM ``Fin.mk #[
              some outputSize, some outputIndexNat, some hOutputIndex]
          let positionNat ←
            mkAppM ``Nat.mod #[
              ← mkAppM ``Nat.div #[
                outputIndexNat, mkNatLit trailingSize],
              mkNatLit packedAxisLength]
          let hPackedAxisPositive ←
            mkDecideProof <| ← mkLT (mkNatLit 0) (mkNatLit packedAxisLength)
          let hPositionBound ←
            mkAppM ``Nat.mod_lt #[
              ← mkAppM ``Nat.div #[
                outputIndexNat, mkNatLit trailingSize],
              hPackedAxisPositive]
          let hPositionDecomposition ←
            mkAppM ``MixedRadix.div_eq_mod_add_mul_div #[
              outputIndexNat, mkNatLit trailingSize,
              mkNatLit packedAxisLength]
          let some (nativePosition, hNativePosition) ←
              certifiedNativeIndexValue? positionNat #[hNativeOutputIndex]
            | return none
          let expectedValue := mkApp values outputFin
          let (nextOutput, hNextOutput) ←
            compileNativePackDispatch specializedChecked inputFamily
              outputBuffer outputSize outputIndexNat outputFin
              hNativeOutputIndex nativePosition positionNat hNativePosition
              hPositionDecomposition
              checkedValue componentCount trailingSize packedAxisLength
              segments #[hPositionBound] expectedValue
          let nativeStep ←
            mkLambdaFVars #[outputBuffer, outputIndex, hOutputIndex]
              nextOutput
          let hStep ←
            mkLambdaFVars #[outputBuffer, outputIndex, hOutputIndex]
              hNextOutput
          return some (nativeStep, hStep)
    | return none
  let nativeStep ← sealNativePackCallback nativeStep
  let hStepType ← inferType hStep
  let hStep ← sealCertificate hStepType hStep
  let implementation ←
    mkAppM ``nativeTensorOfCopyFn #[
      outputBound, hOutputBound, nativeStep, values, hStep]
  let implementation ← sealNativePackImplementation implementation
  let hNativeImplementation ←
    mkAppM ``nativeTensorOfCopyFn_correct #[
      outputBound, hOutputBound, nativeStep, values, hStep]
  let flatReference ← mkAppM ``Rep.unflatten #[values]
  let hNativeImplementationType ← mkEq implementation flatReference
  let hNativeImplementation ←
    withTransparency .all <|
      mkExpectedTypeHint hNativeImplementation hNativeImplementationType
  let hNativeImplementation ←
    sealCertificate hNativeImplementationType hNativeImplementation
  let hReconstruction ←
    mkAppM ``Rep.unflatten_flatten #[compilerReference]
  let hReconstructionType ← mkEq flatReference compilerReference
  let hReconstruction ←
    withTransparency .all <|
      mkExpectedTypeHint hReconstruction hReconstructionType
  let hReconstruction ←
    sealCertificate hReconstructionType hReconstruction
  let rawReference ←
    mkAppM ``Semantics.denotePack #[checked, inputFamily]
  let reference ← sealNativePackReference rawReference
  let hReferenceType ← mkEq compilerReference reference
  let hReference ←
    withTransparency .all <|
      mkExpectedTypeHint (← mkEqRefl rawReference) hReferenceType
  let hReference ← sealCertificate hReferenceType hReference
  let hImplementation ←
    mkAppM ``Eq.trans #[hNativeImplementation, hReconstruction]
  let hImplementation ←
    mkAppM ``Eq.trans #[hImplementation, hReference]
  let hImplementationType ← inferType hImplementation
  let hImplementation ←
    sealCertificate hImplementationType hImplementation
  return some <| ← mkAppM ``certifiedNativeTensor #[
    implementation, hImplementation]

/--
Try to compile a concrete pack plan to a native output loop.

The generated dispatcher supports any number of components. Native pack is an
optional proof-producing optimization: if its word-arithmetic certificates
cannot be constructed, elaboration transactionally restores its state and
retains the general verified coordinate lowering. Symbolic, zero-volume, and
nonportable plans also return `none`.
-/
def compileNativePack?
    (compilerChecked checked inputFamily : Expr)
    (checkedValue : Check.CheckedPack) :
    TermElabM (Option Expr) := do
  match ←
      observing?
        (compileNativePackCore? compilerChecked checked inputFamily
          checkedValue) with
  | some result => return result
  | none => return none

end TorchLean.Tensor.Internal.Elab.Impl
