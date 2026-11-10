/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Lowering.Pack.Index
public import TorchLean.Tensor.Einops.Lowering.Pack
public meta import TorchLean.Tensor.Einops.Elab.Pull
public import TorchLean.Tensor.Einops.Elab
public import TorchLean.Tensor.Einops.Elab.Pull
import TorchLean.Tensor.Einops.Elab.Kernel.Index
import TorchLean.Tensor.Einops.Elab.Index

/-!
# Certified native unpack

Concrete unpack plans compile each component to a native kernel over the packed buffer. We first
try `compileNativeSlices?`, which copies one contiguous block of the packed buffer per leading
coordinate, and use the entry-by-entry `compileNativePullFlat?` when the slice copy does not apply
or its certificate fails. Each kernel is proved equal to its component of `Lowering.unpack`,
and plans with symbolic or nonportable shapes use that lowering directly.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta
open Tensor (Rep)

/--
Compile one checked unpack component to a specialized flat-index map.

The returned equality relates the executable map to the operation-level
`Pack.unpackFlat`, which is independently proved equal to the packed
coordinate semantics.
-/
private def compileUnpackFlatMap
    (checked : Expr) (checkedValue : Pack)
    (componentIndex componentCount : Nat) :
    TermElabM (Expr × Expr) := do
  let specializedChecked ← zetaReduce checked
  let component ← finIndexExpr componentCount componentIndex
  let inputShapeValue := checkedValue.inputShapes[componentIndex]!
  let starShapeValue :=
    PackPattern.starShape checkedValue.pattern inputShapeValue
  let trailingShapeValue := checkedValue.trailingShape
  let packedAxisLengthValue := checkedValue.segmentLengths.sum
  let offsetValue :=
    (checkedValue.segmentLengths.take componentIndex).sum
  let inputShapes ←
    mkAppM ``Pack.inputShapes #[specializedChecked]
  let inputShape ← mkAppM ``List.get #[inputShapes, component]
  let outputShape ←
    mkAppM ``Pack.output #[specializedChecked]
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
      mkAppM ``Pack.starShape #[
        specializedChecked, component]
    let checkedTrailingShape ←
      mkAppM ``Pack.trailingShape #[specializedChecked]
    let checkedPackedAxisLength ←
      mkAppM ``Pack.packedAxisLength #[specializedChecked]
    let checkedOffset ←
      mkAppM ``Pack.componentOffset #[
        specializedChecked, component]
    let componentValue ← mkAppM ``Fin.val #[component]
    let checkedSegmentLengths ←
      mkAppM ``Pack.segmentLengths #[specializedChecked]
    let checkedPrefix ←
      mkAppM ``List.take #[componentValue, checkedSegmentLengths]
    let checkedPrefixSum ← mkAppM ``List.sum #[checkedPrefix]
    let literalDirectIndex ←
      mkAppM ``Pack.unpackDirect #[
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
      mkAppM ``Pack.componentOffset_eq_sum_take #[
        specializedChecked, component]
    let hOffset ←
      mkAppM ``Eq.trans #[
        hLiteralPrefix, ← mkAppM ``Eq.symm #[hCheckedPrefix]]
    let hLiteralChecked ←
      mkAppOptM ``Pack.unpackDirect_congr #[
        some literalStarShape, some checkedStarShape,
        some literalTrailingShape, some checkedTrailingShape,
        some literalPackedAxisLength, some checkedPackedAxisLength,
        some literalOffset, some checkedOffset, some inputValue,
        some hStarShape, some hTrailingShape,
        some hPackedAxisLength, some hOffset]
    let (compiledValue, hCompiledLiteral) ←
      compileRowMajorIndex literalDirectIndex #[
        ``Pack.unpackDirect]
    let hCompiledDirect ←
      mkAppM ``Eq.trans #[hCompiledLiteral, hLiteralChecked]
    let hDirectCompact ←
      mkAppM ``Pack.unpackDirect_eq #[
        specializedChecked, component, inputIndex]
    let hCompiledCompact ←
      mkAppM ``Eq.trans #[hCompiledDirect, hDirectCompact]
    let compactBound ←
      mkAppM ``Pack.unpackIndex_lt #[
        specializedChecked, component, inputIndex]
    let compiledBound ←
      mkAppM ``lt_of_eq_of_lt #[hCompiledCompact, compactBound]
    let compiledIndex ←
      mkAppOptM ``Fin.mk #[
        some outputSize, some compiledValue, some compiledBound]
    let compactFlatIndex ←
      mkAppM ``Pack.unpackFlat #[
        specializedChecked, component, inputIndex]
    let hCompiledIndex ←
      finIndexEquality outputSize compiledIndex compactFlatIndex
        hCompiledCompact
    let flatMap ← mkLambdaFVars #[inputIndex] compiledIndex
    let hFlatMap ← mkLambdaFVars #[inputIndex] hCompiledIndex
    let originalCompactFlatMap ←
      mkAppM ``Pack.unpackFlat #[
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
Compile one unpack component to `compileNativeSlices?` with one contiguous block of the packed
buffer per leading coordinate, starting at the component's segment offset in its packed row.

`simp` and `omega` with the `Pack.unpackDirect` lemmas prove that each slice index is the
index the specialized flat map selects.
-/
private def compileNativeUnpackSlices?
    (componentShape packedTensor flatMap : Expr)
    (checkedValue : Pack) (componentIndex : Nat) :
    TermElabM (Option (Expr × Expr)) := do
  let starShape :=
    PackPattern.starShape checkedValue.pattern
      checkedValue.inputShapes[componentIndex]!
  let trailingSize := Shape.size checkedValue.trailingShape
  compileNativeSlices? componentShape packedTensor flatMap
      (Shape.size checkedValue.leadingShape)
      (Shape.size starShape * trailingSize)
      (checkedValue.segmentLengths.sum * trailingSize)
      ((checkedValue.segmentLengths.take componentIndex).sum * trailingSize)
    fun _ _ sourceIndex mappedIndex => do
      -- Rewrite the `finProdFinEquiv` value first: once `simp_all` evaluates the
      -- literal product in its type, the value lemma does not match.
      let tactic ←
        `(tactic|
          simp only [Equiv.toFun_as_coe, Fin.val_cast,
            finProdFinEquiv_apply_val] <;>
          simp_all (config := { zeta := true }) [
            Pack.unpackDirect,
            Pack.unpackDirect_row_column,
            Shape.size]
          <;> omega)
      certifyWithTactic
        "that contiguous unpack slices implement the checked flat map"
        (← mkEq sourceIndex mappedIndex) tactic

/--
Compile every component of one concrete unpack plan to native flat loops.

The result has the same dependent component family as ordinary `unpack`. Each component is a
`Native.tensorKernel` whose reference is the matching component of `Einops.Lowering.unpack`.
-/
def compileNativeUnpack?
    (scalarType storage checked packedTensor : Expr)
    (checkedValue : Pack) :
    TermElabM (Option Expr) := do
  let componentCount := checkedValue.inputShapes.length
  let packedDimensions := checkedValue.output.map Lean.toExpr
  let mut components : List Expr := []
  for componentIndex in [:componentCount] do
    let component ← finIndexExpr componentCount componentIndex
    let componentShape :=
      Lean.toExpr checkedValue.inputShapes[componentIndex]!
    let (flatMap, hFlatMap) ←
      compileUnpackFlatMap checked checkedValue componentIndex componentCount
    let sliceResult? ←
      observing? <|
        compileNativeUnpackSlices? componentShape packedTensor flatMap
          checkedValue componentIndex
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
      mkAppM ``Pack.unpackFlat #[
        checked, component]
    let hCompiledCompact ←
      mkAppOptM ``Rep.pullFlat_congr #[
        some scalarType, some storage, none, none,
        some flatMap, some compactFlatMap, some hFlatMap,
        some packedTensor]
    let unpacked ←
      mkAppOptM ``Lowering.unpack #[
        some scalarType, some storage, some checked, some packedTensor]
    let reference := mkApp unpacked component
    let hReferenceCompact ←
      mkAppOptM
        ``Pack.denoteUnpack_eq_pullFlat #[
          some scalarType, some storage, some checked,
          some packedTensor, some component]
    let hCompiledSemantic ←
      mkAppM ``Eq.trans #[
        hCompiledCompact, ← mkAppM ``Eq.symm #[hReferenceCompact]]
    let hUnpackCorrect ←
      mkAppOptM ``Lowering.unpack_correct #[
        some scalarType, some storage, some checked, some packedTensor]
    let hReferenceSemantic ←
      mkAppM ``congrFun #[hUnpackCorrect, component]
    let hCompiledReference ←
      mkAppM ``Eq.trans #[
        hCompiledSemantic, ← mkAppM ``Eq.symm #[hReferenceSemantic]]
    let result ←
      mkAppM ``Native.tensorKernel #[
        reference, compiled, implementation,
        hCompiledReference, hImplementation]
    components := components.concat result
  let componentShapes := checkedValue.inputShapes.map Lean.toExpr
  return some <|
    ← buildTensorFamily scalarType storage componentShapes components

end TorchLean.Einops.Elab
