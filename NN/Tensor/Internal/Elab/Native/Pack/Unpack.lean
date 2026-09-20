/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Laws.PackIndex
public import NN.Tensor.Internal.Lowering.Pack
public meta import NN.Tensor.Internal.Elab.Native.Pull
public import NN.Tensor.Internal.Elab.Common
public import NN.Tensor.Internal.Elab.Native.Pull
import NN.Tensor.Internal.Elab.Einsum.Kernel.Index
import NN.Tensor.Internal.Elab.Native.Index

/-!
# Certified native unpack

Concrete unpack plans compile each component to one native flat pullback from
the packed buffer. The generated source index is specialized row-major
arithmetic; the independent packed-coordinate equivalence remains the
correctness reference.

Symbolic or nonportable shapes retain the general coordinate lowering.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Construct a component index in the canonical form consumed by `Fin.cases`.

Using successors instead of a modular `OfNat` numeral keeps generated
dependent tensor families definitionally transparent to proof tactics.
-/
private def componentIndexExpr
    (componentCount componentIndex : Nat) : MetaM Expr := do
  unless componentIndex < componentCount do
    throwError
      "internal error: unpack component index {componentIndex} is outside \
        component count {componentCount}"
  let remainingCount := componentCount - componentIndex
  let remainingType ← mkAppM ``Fin #[mkNatLit remainingCount]
  let mut component ← mkNumeral remainingType 0
  for _ in [:componentIndex] do
    component ← mkAppM ``Fin.succ #[component]
  return component

/--
Compile one checked unpack component to a specialized flat-index map.

The returned equality relates the executable map to the operation-level
`unpackFlatIndex`, which is independently proved equal to the packed
coordinate semantics.
-/
private def compileUnpackFlatMap
    (checked : Expr) (checkedValue : Check.CheckedPack)
    (componentIndex componentCount : Nat) :
    TermElabM (Expr × Expr) := do
  let specializedChecked ← zetaReduce checked
  let component ← componentIndexExpr componentCount componentIndex
  let inputShapeValue := checkedValue.inputShapes[componentIndex]!
  let starShapeValue :=
    Check.packStarShape checkedValue.pattern inputShapeValue
  let trailingShapeValue := checkedValue.trailingShape
  let packedAxisLengthValue := checkedValue.segmentLengths.sum
  let offsetValue :=
    (checkedValue.segmentLengths.take componentIndex).sum
  let inputShapes ←
    mkAppM ``Check.CheckedPack.inputShapes #[specializedChecked]
  let inputShape ← mkAppM ``List.get #[inputShapes, component]
  let outputShape ←
    mkAppM ``Check.CheckedPack.output #[specializedChecked]
  let inputSize ← mkAppM ``Shape.size #[inputShape]
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let inputIndexType ← mkAppM ``Fin #[inputSize]
  withLocalDeclD `inputIndex inputIndexType fun inputIndex => do
    let inputValue ← mkAppM ``Fin.val #[inputIndex]
    let literalStarShape := Lean.toExpr starShapeValue
    let literalTrailingShape := Lean.toExpr trailingShapeValue
    let literalPackedAxisLength := mkNatLit packedAxisLengthValue
    let literalOffset := mkNatLit offsetValue
    let checkedStarShape ←
      mkAppM ``Check.CheckedPack.starShape #[
        specializedChecked, component]
    let checkedTrailingShape ←
      mkAppM ``Check.CheckedPack.trailingShape #[specializedChecked]
    let checkedPackedAxisLength ←
      mkAppM ``Check.CheckedPack.packedAxisLength #[specializedChecked]
    let checkedOffset ←
      mkAppM ``Lowering.Pack.Impl.componentOffset #[
        specializedChecked, component]
    let componentValue ← mkAppM ``Fin.val #[component]
    let checkedSegmentLengths ←
      mkAppM ``Check.CheckedPack.segmentLengths #[specializedChecked]
    let checkedPrefix ←
      mkAppM ``List.take #[componentValue, checkedSegmentLengths]
    let checkedPrefixSum ← mkAppM ``List.sum #[checkedPrefix]
    let literalDirectIndex ←
      mkAppM ``Lowering.Pack.Impl.unpackDirectLinearIndex #[
        literalStarShape,
        literalTrailingShape,
        literalPackedAxisLength,
        literalOffset,
        inputValue]
    let hStarShape ←
      certifyGeneratedInvariant
        "that the specialized unpack star shape matches the checked plan"
        (← mkEq literalStarShape checkedStarShape)
    let hTrailingShape ←
      certifyGeneratedInvariant
        "that the specialized unpack trailing shape matches the checked plan"
        (← mkEq literalTrailingShape checkedTrailingShape)
    let hPackedAxisLength ←
      certifyGeneratedInvariant
        "that the specialized unpack packed length matches the checked plan"
        (← mkEq literalPackedAxisLength checkedPackedAxisLength)
    let hLiteralPrefix ←
      certifyGeneratedInvariant
        "that the specialized unpack prefix matches the checked segment list"
        (← mkEq literalOffset checkedPrefixSum)
    let hCheckedPrefix ←
      mkAppM ``Lowering.Pack.Impl.componentOffset_eq_sum_take #[
        specializedChecked, component]
    let hOffset ←
      mkAppM ``Eq.trans #[
        hLiteralPrefix, ← mkAppM ``Eq.symm #[hCheckedPrefix]]
    let hLiteralChecked ←
      mkAppOptM ``Lowering.Pack.Impl.unpackDirectLinearIndex_congr #[
        some literalStarShape, some checkedStarShape,
        some literalTrailingShape, some checkedTrailingShape,
        some literalPackedAxisLength, some checkedPackedAxisLength,
        some literalOffset, some checkedOffset, some inputValue,
        some hStarShape, some hTrailingShape,
        some hPackedAxisLength, some hOffset]
    let (compiledValue, hCompiledLiteral) ←
      compileRowMajorIndex literalDirectIndex #[
        ``Lowering.Pack.Impl.unpackDirectLinearIndex]
    let hCompiledDirect ←
      mkAppM ``Eq.trans #[hCompiledLiteral, hLiteralChecked]
    let hDirectCompact ←
      mkAppM ``Lowering.Pack.Impl.unpackDirectLinearIndex_eq #[
        specializedChecked, component, inputIndex]
    let hCompiledCompact ←
      mkAppM ``Eq.trans #[hCompiledDirect, hDirectCompact]
    let compactBound ←
      mkAppM ``Lowering.Pack.Impl.unpackLinearIndex_lt #[
        specializedChecked, component, inputIndex]
    let compiledBound ←
      indexBoundFromValueEquality outputSize hCompiledCompact compactBound
    let compiledIndex ←
      mkAppOptM ``Fin.mk #[
        some outputSize, some compiledValue, some compiledBound]
    let compactFlatIndex ←
      mkAppM ``Lowering.Pack.Impl.unpackFlatIndex #[
        specializedChecked, component, inputIndex]
    let hCompiledIndex ←
      finIndexEquality outputSize compiledIndex compactFlatIndex
        hCompiledCompact
    let flatMap ← mkLambdaFVars #[inputIndex] compiledIndex
    let hFlatMap ← mkLambdaFVars #[inputIndex] hCompiledIndex
    let originalCompactFlatMap ←
      mkAppM ``Lowering.Pack.Impl.unpackFlatIndex #[
        checked, component]
    let expected ←
      withLocalDeclD `index inputIndexType fun index =>
        mkEq (mkApp flatMap index)
          (mkApp originalCompactFlatMap index) >>= fun equality =>
        mkForallFVars #[index] equality
    let hFlatMap ←
      withTransparency .all <| mkExpectedTypeHint hFlatMap expected
    return (flatMap, hFlatMap)

/--
Compile one unpack component by copying its contiguous row blocks.

The optimization is optional. Concrete nonempty blocks use one native outer
loop and one runtime array-slice traversal per leading coordinate; plans whose
shape arithmetic cannot be certified retain the scalar pullback compiler.
-/
private def compileNativeUnpackSlices?
    (scalarType storage componentShape packedTensor flatMap : Expr)
    (checkedValue : Check.CheckedPack) (componentIndex : Nat) :
    TermElabM (Option (Expr × Expr)) := do
  let leadingSize := Shape.size checkedValue.leadingShape
  let starShape :=
    Check.packStarShape checkedValue.pattern
      checkedValue.inputShapes[componentIndex]!
  let trailingSize := Shape.size checkedValue.trailingShape
  let rowLength := Shape.size starShape * trailingSize
  let sourceStride := checkedValue.segmentLengths.sum * trailingSize
  let sourceOffset :=
    (checkedValue.segmentLengths.take componentIndex).sum * trailingSize
  if leadingSize = 0 || rowLength = 0 then
    return none
  let literalLeadingSize := mkNatLit leadingSize
  let literalRowLength := mkNatLit rowLength
  let literalSourceStride := mkNatLit sourceStride
  let literalSourceOffset := mkNatLit sourceOffset
  let outputSize ← mkAppM ``Shape.size #[componentShape]
  let hShapeSize ←
    certifyGeneratedInvariant
      "that a native unpack component is a rectangle of contiguous rows"
      (← mkEq
        (← mkAppM ``Nat.mul #[literalLeadingSize, literalRowLength])
        outputSize)
  let some (rowBound, hRowBound) ← nativeLoopBound? literalLeadingSize
    | return none
  let packedTensorType ← whnf (← inferType packedTensor)
  let packedTensorType := packedTensorType.consumeMData
  unless packedTensorType.isAppOfArity ``Rep 3 do
    throwError "internal error: native unpack received a non-tensor input"
  let packedShape := packedTensorType.getAppArgs[1]!
  let packedSize ← mkAppM ``Shape.size #[packedShape]
  let rowType ← mkAppM ``Fin #[literalLeadingSize]
  let columnType ← mkAppM ``Fin #[literalRowLength]
  let hRanges ←
    withLocalDeclD `row rowType fun row => do
      let rowValue ← mkAppM ``Fin.val #[row]
      let start ←
        mkAppM ``Nat.add #[
          ← mkAppM ``Nat.mul #[rowValue, literalSourceStride],
          literalSourceOffset]
      let stop ← mkAppM ``Nat.add #[start, literalRowLength]
      let hRange ←
        certifyGeneratedInvariant
          "that a native unpack row lies inside the packed buffer"
          (← mkLE stop packedSize)
      mkLambdaFVars #[row] hRange
  let values ←
    withLocalDeclD `inputIndex (← mkAppM ``Fin #[outputSize])
        fun inputIndex => do
      let value ←
        mkAppM ``Rep.getFlat #[
          packedTensor, mkApp flatMap inputIndex]
      mkLambdaFVars #[inputIndex] value
  let hValues ←
    withLocalDeclD `row rowType fun row => do
      withLocalDeclD `column columnType fun column => do
        let rowValue ← mkAppM ``Fin.val #[row]
        let columnValue ← mkAppM ``Fin.val #[column]
        let sourceIndexValue ←
          mkAppM ``Nat.add #[
            ← mkAppM ``Nat.add #[
              ← mkAppM ``Nat.mul #[rowValue, literalSourceStride],
              literalSourceOffset],
            columnValue]
        let hSourceIndex ←
          certifyGeneratedInvariant
            "that a native unpack slice index lies inside the packed buffer"
            (← mkLT sourceIndexValue packedSize)
        let sourceIndex ←
          mkAppOptM ``Fin.mk #[
            some packedSize, some sourceIndexValue, some hSourceIndex]
        let rectangularIndex ←
          mkAppM ``rectangularIndex #[
            literalLeadingSize, literalRowLength, row, column]
        let inputIndex ←
          mkAppM ``Fin.cast #[hShapeSize, rectangularIndex]
        let runtimeSourceIndex := mkApp flatMap inputIndex
        let tactic ←
          `(tactic|
            simp_all (config := { zeta := true }) [
              rectangularIndex,
              Lowering.Pack.Impl.unpackDirectLinearIndex,
              Lowering.Pack.Impl.unpackDirectLinearIndex_row_column,
              Shape.size]
            <;> omega)
        let hIndex ←
          certifyWithTactic
            "that contiguous unpack slices implement the checked flat map"
            (← mkEq sourceIndex runtimeSourceIndex) tactic
        let sourceValue ←
          mkAppM ``Rep.getFlat #[packedTensor, sourceIndex]
        let expectedValue := mkApp values inputIndex
        let getFlatFunction ←
          withLocalDeclD `index (← mkAppM ``Fin #[packedSize]) fun index => do
            let value ←
              mkAppM ``Rep.getFlat #[packedTensor, index]
            mkLambdaFVars #[index] value
        let hValue ←
          mkAppM ``congrArg #[
            getFlatFunction, hIndex]
        let hValue ←
          withTransparency .all <|
            mkExpectedTypeHint hValue (← mkEq sourceValue expectedValue)
        mkLambdaFVars #[row, column] hValue
  let hRangesType ← inferType hRanges
  let hRanges ← sealCertificate hRangesType hRanges
  let hValuesType ← inferType hValues
  let hValues ← sealCertificate hValuesType hValues
  let implementation ←
    mkAppM ``nativeTensorOfSlices #[
      literalLeadingSize, literalRowLength, literalSourceStride,
      literalSourceOffset, hShapeSize, rowBound, hRowBound,
      packedTensor, values, hRanges, hValues]
  let hImplementation ←
    mkAppM ``nativeTensorOfSlices_correct #[
      literalLeadingSize, literalRowLength, literalSourceStride,
      literalSourceOffset, hShapeSize, rowBound, hRowBound,
      packedTensor, values, hRanges, hValues]
  let compiled ←
    mkAppOptM ``Rep.pullFlat #[
      some scalarType, some storage, none, none,
      some flatMap, some packedTensor]
  let hImplementation ←
    withTransparency .all <|
      mkExpectedTypeHint hImplementation
        (← mkEq implementation compiled)
  return some (implementation, hImplementation)

/--
Compile every component of one concrete unpack plan to native flat loops.

The result keeps the same dependent component family as ordinary `unpack`;
only component storage construction changes.
-/
def compileNativeUnpack?
    (scalarType storage checked packedTensor : Expr)
    (checkedValue : Check.CheckedPack) :
    TermElabM (Option Expr) := do
  let componentCount := checkedValue.inputShapes.length
  let packedDimensions := checkedValue.output.map Lean.toExpr
  let mut components : List Expr := []
  for componentIndex in [:componentCount] do
    let component ← componentIndexExpr componentCount componentIndex
    let componentShape :=
      Lean.toExpr checkedValue.inputShapes[componentIndex]!
    let (flatMap, hFlatMap) ←
      compileUnpackFlatMap checked checkedValue componentIndex componentCount
    let sliceResult? ←
      observing? <|
        compileNativeUnpackSlices? scalarType storage componentShape
          packedTensor flatMap checkedValue componentIndex
    let some (implementation, hImplementation) ←
        match sliceResult?.join with
        | some result => pure (some result)
        | none =>
            compileNativePullFlat? componentShape packedTensor flatMap
              (nativeSourceDimensions? := some packedDimensions)
      | return none
    let compiled ←
      mkAppOptM ``Rep.pullFlat #[
        some scalarType, some storage, none, none,
        some flatMap, some packedTensor]
    let compactFlatMap ←
      mkAppM ``Lowering.Pack.Impl.unpackFlatIndex #[
        checked, component]
    let hCompiledCompact ←
      mkAppOptM ``Rep.pullFlat_congr #[
        some scalarType, some storage, none, none,
        some flatMap, some compactFlatMap, some hFlatMap,
        some packedTensor]
    let unpacked ←
      mkAppOptM ``Lowering.unpackTensor #[
        some scalarType, some storage, some checked, some packedTensor]
    let reference := mkApp unpacked component
    let hReferenceCompact ←
      mkAppOptM
        ``Lowering.Pack.Impl.denoteUnpack_component_eq_pullFlat #[
          some scalarType, some storage, some checked,
          some packedTensor, some component]
    let hCompiledSemantic ←
      mkAppM ``Eq.trans #[
        hCompiledCompact, ← mkAppM ``Eq.symm #[hReferenceCompact]]
    let hUnpackCorrect ←
      mkAppOptM ``Lowering.unpackTensor_correct #[
        some scalarType, some storage, some checked, some packedTensor]
    let hReferenceSemantic ←
      mkAppM ``congrFun #[hUnpackCorrect, component]
    let hCompiledReference ←
      mkAppM ``Eq.trans #[
        hCompiledSemantic, ← mkAppM ``Eq.symm #[hReferenceSemantic]]
    let result ←
      mkAppM ``nativeTensorKernel #[
        reference, compiled, implementation,
        hCompiledReference, hImplementation]
    components := components.concat result
  let componentShapes := checkedValue.inputShapes.map Lean.toExpr
  return some <|
    ← buildTensorFamily scalarType storage componentShapes components

end TorchLean.Tensor.Internal.Elab.Impl
