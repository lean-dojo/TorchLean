/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Mathlib.Basic.Logic.Basic
public import TorchLean.Tensor.Einops.Lowering.Pack.Index
public import TorchLean.Tensor.Einops.Elab.Kernel.Index
public meta import TorchLean.Tensor.Einops.Elab.Pull -- shake: keep
import TorchLean.Tensor.Einops.Elab.Index
import TorchLean.Tensor.Native.Construction

/-!
# Certified native pack dispatch

A concrete pack plan compiles to one native output loop and a balanced dispatcher over the
component segments. Every leaf reads one component buffer through the verified inverse unpack
index, so neither the component count nor the tensor rank adds a semantic case.

Symbolic shapes, and shapes too large for portable 32-bit native indices, keep the general
coordinate lowering.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta
open Tensor (Rep)

/--
A nonempty component segment for native pack dispatch: the component index, its offset on the
packed axis, its length there and its star shape.
-/
private abbrev PackSegment :=
  Nat × (Nat × (Nat × Shape))

/--
Compile one leaf of a native pack dispatcher.

The leaf's path assumptions identify its component segment. The generated
source index is direct row-major arithmetic, while the returned theorem
identifies the direct component read with the independent pack denotation.
-/
private def compileNativePackLeaf
    (checked inputFamily buffer outputSize outputIndexNat outputFin
      hNativeOutputIndex positionNat hPositionDecomposition : Expr)
    (checkedValue : Pack)
    (componentCount trailingSize packedAxisLength : Nat)
    (segment : PackSegment) (pathAssumptions : Array Expr)
    (expectedValue : Expr) :
    TermElabM (Expr × Expr) := do
  let (componentIndex, offset, segmentLength, starShapeValue) := segment
  let componentType ← mkAppM ``Fin #[mkNatLit componentCount]
  let component ← mkNumeral componentType componentIndex
  let specializedChecked ← zetaReduce checked
  let inputShapes ←
    mkAppM ``Pack.inputShapes #[specializedChecked]
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
    mkAppM ``Pack.starShape #[
      specializedChecked, component]
  let checkedTrailingShape ←
    mkAppM ``Pack.trailingShape #[specializedChecked]
  let checkedPackedAxisLength ←
    mkAppM ``Pack.packedAxisLength #[specializedChecked]
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
    mkAppM ``Pack.segmentLengths #[specializedChecked]
  let checkedPrefix ←
    mkAppM ``List.take #[componentValue, checkedSegmentLengths]
  let checkedPrefixSum ← mkAppM ``List.sum #[checkedPrefix]
  let hLiteralPrefix ←
    certifyGeneratedInvariant
      "that a native pack segment has the checked prefix offset"
      (← mkEq literalOffset checkedPrefixSum)
  let hCheckedPrefix ←
    mkAppM ``Pack.componentOffset_eq_sum_take #[
      specializedChecked, component]
  let hOffset ←
    mkAppM ``Eq.trans #[
      hLiteralPrefix, ← mkAppM ``Eq.symm #[hCheckedPrefix]]
  let directIndex ←
    mkAppM ``Pack.packDirect #[
      literalStarShape, literalTrailingShape,
      literalPackedAxisLength, literalOffset, outputIndexNat]
  let (compiledIndexValue, hCompiledIndexValue) ←
    compileRowMajorIndex directIndex #[
      ``Pack.packDirect]
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
    mkAppM ``Pack.packDirect_lt #[
      literalStarShape, literalTrailingShape,
      literalPackedAxisLength, literalOffset,
      outputIndexNat, literalLeadingSize,
      hTrailing, hLiteralOutputBound, hLower, hUpper]
  let hDirectInputBound ←
    withTransparency .all <|
      mkExpectedTypeHint hDirectInputBound
        (← mkLT directIndex literalInputSize)
  let hLiteralInputBound ←
    mkAppM ``lt_of_eq_of_lt #[hCompiledIndexValue, hDirectInputBound]
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
      ``Pack.unpackFlat_eq_of_packDirect' #[
      specializedChecked, component, inputIndex, outputFin,
      literalStarShape, literalTrailingShape,
      literalPackedAxisLength, literalOffset,
      hStarShape, hTrailingShape, hPackedAxisLength, hOffset,
      hCompiledIndexValue, hTrailing, hLower, hUpper]
  let hSemanticValue ←
    mkAppM ``Pack.getFlat_denotePack_of_eq #[
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
    mkAppM ``Native.tensorCopyAt #[
      sourceTensor, sourceIndex, hSourceIndex, buffer]
  let hNextOutput ←
    mkAppM ``Native.tensorCopyAt_eq_push_of_eq #[
      sourceTensor, sourceIndex, hSourceIndex, buffer,
      expectedValue, hValue]
  return (nextOutput, hNextOutput)

/--
Compile a balanced decision tree over the component segments for one packed output entry.

Each split tests whether the packed-axis position lies below the first offset of the right half.
The comparisons on its path give every leaf the facts it needs to certify its component interval
and the native subtraction of its offset.
-/
private partial def compileNativePackDispatch
    (checked inputFamily buffer outputSize outputIndexNat outputFin
      hNativeOutputIndex nativePosition positionNat hNativePosition
      hPositionDecomposition : Expr)
    (checkedValue : Pack)
    (componentCount trailingSize packedAxisLength : Nat)
    (segments : List PackSegment) (pathAssumptions : Array Expr)
    (expectedValue : Expr) :
    TermElabM (Expr × Expr) := do
  match segments with
  | [] =>
      throwError
        "internal error: native pack dispatch has no reachable component"
  | [segment] =>
      compileNativePackLeaf checked inputFamily buffer outputSize
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
            mkAppM ``Native.nat_lt_of_lt_of_eq #[
              nativePosition, nativeThreshold,
              positionNat, mkNatLit threshold,
              hNativePosition, hNativeThreshold, hLeft]
          let (leftValue, hLeftValue) ←
            compileNativePackDispatch checked inputFamily buffer outputSize
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
            mkAppM ``Native.nat_le_of_not_lt_of_eq #[
              nativePosition, nativeThreshold,
              positionNat, mkNatLit threshold,
              hNativePosition, hNativeThreshold, hRight]
          let (rightValue, hRightValue) ←
            compileNativePackDispatch checked inputFamily buffer outputSize
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
        mkAppM ``Storage.push #[buffer, expectedValue]
      let hBranches ←
        mkAppM ``And.intro #[hLeftFunction, hRightFunction]
      let branchRule ←
        mkAppOptM ``dite_eq_iff' #[
          some (← inferType expectedOutput), some condition, none,
          some expectedOutput, some leftFunction, some rightFunction]
      let hValue ←
        mkAppM ``Iff.mpr #[branchRule, hBranches]
      return (value, hValue)

/- Compile one concrete pack plan to a native output loop. -/
private def compileNativePackCore?
    (compilerChecked checked inputFamily : Expr)
    (checkedValue : Pack) :
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
              PackPattern.starShape checkedValue.pattern
                checkedValue.inputShapes[componentIndex]!)
  if trailingSize = 0 || packedAxisLength = 0 || segments.isEmpty then
    return none
  let specializedChecked ← zetaReduce compilerChecked
  let outputShape ←
    mkAppM ``Pack.output #[specializedChecked]
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let some (outputBound, hOutputBound) ← nativeLoopBound? outputSize
    | return none
  let rawCompilerReference ←
    mkAppM ``Semantics.denotePack #[specializedChecked, inputFamily]
  -- Naming the proof-only reference keeps the reconstruction theorems shallow.
  let compilerReference ←
    sealAuxDefinition `_einops_native_pack_reference rawCompilerReference
  let values ← mkAppM ``Rep.getFlat #[compilerReference]
  let componentType ← mkAppM ``Fin #[mkNatLit componentCount]
  let firstComponent ← mkNumeral componentType segments.head!.1
  let firstTensor ←
    withTransparency .all <| whnf (mkApp inputFamily firstComponent)
  let firstBuffer ← mkAppM ``Rep.buffer #[firstTensor]
  let bufferType ← inferType firstBuffer
  let some (nativeStep, hStep) ←
    withLocalDeclD `output bufferType fun buffer => do
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
              buffer outputSize outputIndexNat outputFin
              hNativeOutputIndex nativePosition positionNat hNativePosition
              hPositionDecomposition
              checkedValue componentCount trailingSize packedAxisLength
              segments #[hPositionBound] expectedValue
          let nativeStep ←
            mkLambdaFVars #[buffer, outputIndex, hOutputIndex]
              nextOutput
          let hStep ←
            mkLambdaFVars #[buffer, outputIndex, hOutputIndex]
              hNextOutput
          return some (nativeStep, hStep)
    | return none
  let nativeStep ← sealAuxDefinition `_einops_native_pack_callback nativeStep
  let hStepType ← inferType hStep
  let hStep ← sealCertificate hStepType hStep
  let implementation ←
    mkAppM ``Native.tensorOfCopyFn #[
      outputBound, hOutputBound, nativeStep, values, hStep]
  -- The callback and its pointwise certificate are compact constants already, but the dependent
  -- tensor builder carries both through its result type. Naming the completed native tensor
  -- keeps the public kernel wrapper shallow and runs the same native loop.
  let implementation ←
    sealAuxDefinition `_einops_native_pack_implementation implementation
  let hNativeImplementation ←
    mkAppM ``Native.tensorOfCopyFn_correct #[
      outputBound, hOutputBound, nativeStep, values, hStep]
  let flatReference ← mkAppM ``Rep.ofFlatFn #[values]
  let hNativeImplementationType ← mkEq implementation flatReference
  let hNativeImplementation ←
    withTransparency .all <|
      mkExpectedTypeHint hNativeImplementation hNativeImplementationType
  let hNativeImplementation ←
    sealCertificate hNativeImplementationType hNativeImplementation
  let hReconstruction ←
    mkAppM ``Rep.ofFlatFn_getFlat #[compilerReference]
  let hReconstructionType ← mkEq flatReference compilerReference
  let hReconstruction ←
    withTransparency .all <|
      mkExpectedTypeHint hReconstruction hReconstructionType
  let hReconstruction ←
    sealCertificate hReconstructionType hReconstruction
  let rawReference ←
    mkAppM ``Semantics.denotePack #[checked, inputFamily]
  let reference ← sealAuxDefinition `_einops_native_pack_reference rawReference
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
  return some <| ← mkAppM ``Native.certifiedTensor #[
    implementation, hImplementation]

/--
Try to compile a concrete pack plan to a native output loop.

The generated dispatcher handles any number of components. Native pack is an optional
optimization that produces its own proof: if a word-arithmetic certificate cannot be built,
`observing?` restores the elaborator state and the caller keeps the general verified coordinate
lowering. Symbolic and nonportable plans return `none`, as do plans with no nonempty segment or a
zero-sized trailing shape. A zero-sized leading shape can use an empty native output loop.
-/
def compileNativePack?
    (compilerChecked checked inputFamily : Expr)
    (checkedValue : Pack) :
    TermElabM (Option Expr) := do
  return (← observing?
    (compileNativePackCore? compilerChecked checked inputFamily checkedValue)).join

end TorchLean.Einops.Elab
