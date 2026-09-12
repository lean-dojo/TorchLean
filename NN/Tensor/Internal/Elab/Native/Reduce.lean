/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Lowering.Reduce.View
public import NN.Tensor.Internal.Elab.Native.Tensor -- shake: keep
public meta import NN.Tensor.Internal.Elab.Einsum.Kernel.Affine
public meta import NN.Tensor.Internal.Elab.Einsum.Kernel.Index
public meta import NN.Tensor.Internal.Elab.Native.ReductionIndex

/-!
# Certified native reduction loops

Concrete reductions use one native output loop and one native fiber loop.
The pointwise reader certificate connects direct source-buffer reads to the
existing row-major reduction semantics, so scalar operation order is
unchanged.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Elab.Impl

universe u v w

open Check
open Lowering
open Lowering.Reduce.Impl

/--
Reduce every fiber with native output and fiber counters.

The executable callback receives both counters as `USize`. Its pointwise
certificate is erased and proves agreement with the existing flat-reader
reduction at the corresponding bounded indices.
-/
@[inline] def nativeReduceFoldTensorFromFlat
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β)
    (finish : β → Nat → γ)
    (checked : CheckedTransform)
    (_hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (_hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex)
    (outputBound : USize)
    (hOutputBound :
      outputBound.toNat = Shape.size checked.value.output)
    (fiberBound : USize)
    (hFiberBound :
      fiberBound.toNat = Shape.size checked.reductionShape)
    (nativeRead :
      (outputIndex : USize) →
      outputIndex.toNat < Shape.size checked.value.output →
      (fiberIndex : USize) →
      fiberIndex.toNat < Shape.size checked.reductionShape →
      α)
    (hNativeRead :
      ∀ outputIndex hOutputIndex fiberIndex hFiberIndex,
        nativeRead outputIndex hOutputIndex fiberIndex hFiberIndex =
          read
            (reductionInputFlatIndex checked
              (Coord.unlinearize
                ⟨outputIndex.toNat, hOutputIndex⟩)
              ⟨fiberIndex.toNat, hFiberIndex⟩)) :
    checked.OutputTensor γ :=
  nativeTensorOfFlatFn outputBound hOutputBound
    (fun outputIndex hOutputIndex =>
      finish
        (nativeFinFoldl
          (Shape.size checked.reductionShape)
          fiberBound hFiberBound
          (fun total fiberIndex hFiberIndex =>
            step total <|
              nativeRead outputIndex hOutputIndex fiberIndex hFiberIndex)
          initial)
        checked.reductionFiberSize)
    (fun outputIndex =>
      finish
        (reductionFoldlFromFlat step initial checked read <|
          Coord.unlinearize outputIndex)
        checked.reductionFiberSize)
    (by
      intro outputIndex hOutputIndex
      apply congrArg
        (fun total => finish total checked.reductionFiberSize)
      apply nativeFinFoldl_eq_fin_foldl_of_eq
      intro total fiberIndex hFiberIndex
      rw [hNativeRead outputIndex hOutputIndex fiberIndex hFiberIndex])

/--
The native ordered reduction is exactly the existing flat-reader reduction.
-/
theorem nativeReduceFoldTensorFromFlat_correct
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β)
    (finish : β → Nat → γ)
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex)
    (outputBound : USize)
    (hOutputBound :
      outputBound.toNat = Shape.size checked.value.output)
    (fiberBound : USize)
    (hFiberBound :
      fiberBound.toNat = Shape.size checked.reductionShape)
    (nativeRead :
      (outputIndex : USize) →
      outputIndex.toNat < Shape.size checked.value.output →
      (fiberIndex : USize) →
      fiberIndex.toNat < Shape.size checked.reductionShape →
      α)
    (hNativeRead :
      ∀ outputIndex hOutputIndex fiberIndex hFiberIndex,
        nativeRead outputIndex hOutputIndex fiberIndex hFiberIndex =
          read
            (reductionInputFlatIndex checked
              (Coord.unlinearize
                ⟨outputIndex.toNat, hOutputIndex⟩)
              ⟨fiberIndex.toNat, hFiberIndex⟩)) :
    nativeReduceFoldTensorFromFlat step initial finish checked hKind
        inputTensor read hRead outputBound hOutputBound fiberBound
        hFiberBound nativeRead hNativeRead =
      reduceFoldTensorFromFlat step initial finish checked hKind
        inputTensor read hRead := by
  apply nativeTensorOfFlatFn_correct

end TorchLean.Tensor.Internal.Elab.Impl

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/-- Name one generated native reduction callback for compact elaboration. -/
def sealNativeReductionCallback (callback : Expr) :
    TermElabM Expr := do
  let name ← mkAuxName `_einops_native_reduce_callback
  mkAuxDefinitionFor name callback (zetaDelta := true)

/--
Compile an ordered flat-reader reduction to nested native loops.

The source map may be the identity map of an ordinary tensor or a certified
map recovered from a preceding shape-only transform.
-/
def compileNativeReduceFold?
    (step initial finish checked hKind inputTensor semanticInputTensor
      hInputTensor read hRead sourceTensor inputFlatMap : Expr)
    (checkedValue : Check.CheckedTransform) :
    TermElabM (Option Expr) := do
  let value ← mkAppM ``Check.CheckedTransform.value #[checked]
  let outputShape ← mkAppM ``Check.TransformPlan.output #[value]
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let fiberShape ←
    mkAppM ``Check.CheckedTransform.reductionShape #[checked]
  let fiberSize ← mkAppM ``Shape.size #[fiberShape]
  let valueNormalized ←
    mkAppM ``Check.TransformPlan.normalized #[value]
  let logicalInputShape ←
    mkAppM ``Check.NormalizedTransform.input #[valueNormalized]
  let logicalInputSize ←
    mkAppM ``Shape.size #[logicalInputShape]
  let some (outputBound, hOutputBound) ← nativeLoopBound? outputSize
    | return none
  let some (fiberBound, hFiberBound) ← nativeLoopBound? fiberSize
    | return none
  let some (_logicalInputBound, hLogicalInputBound) ←
      nativeLoopBound? logicalInputSize
    | return none
  let sourceTensorType ← whnf (← inferType sourceTensor)
  let sourceTensorType := sourceTensorType.consumeMData
  unless sourceTensorType.isAppOfArity ``Rep 3 do
    throwError "internal error: a native reduction has a non-tensor source"
  let sourceShape := sourceTensorType.getAppArgs[1]!
  let sourceSize ← mkAppM ``Shape.size #[sourceShape]
  let sourceDimensions? ← staticListElements? sourceShape
  let useNativeSourceIndex ←
    match sourceDimensions? with
    | some sourceDimensions =>
        supportsNativeInputIndex sourceDimensions
    | none => pure false
  let nativeReadData ←
    withLocalDeclD `outputIndex (mkConst ``USize) fun outputIndex => do
      let outputIndexNat ← mkAppM ``USize.toNat #[outputIndex]
      let outputIndexBoundType ← mkLT outputIndexNat outputSize
      withLocalDeclD `hOutputIndex outputIndexBoundType fun hOutputIndex => do
        let hNativeOutputIndex ←
          nativeLoopIndexBound outputIndexNat hOutputBound hOutputIndex
        let outputFin ←
          mkAppOptM ``Fin.mk #[
            some outputSize, some outputIndexNat, some hOutputIndex]
        withLocalDeclD `fiberIndex (mkConst ``USize) fun fiberIndex => do
          let fiberIndexNat ← mkAppM ``USize.toNat #[fiberIndex]
          let fiberIndexBoundType ← mkLT fiberIndexNat fiberSize
          withLocalDeclD `hFiberIndex fiberIndexBoundType fun hFiberIndex => do
            let hNativeFiberIndex ←
              nativeLoopIndexBound fiberIndexNat hFiberBound hFiberIndex
            let fiberFin ←
              mkAppOptM ``Fin.mk #[
                some fiberSize, some fiberIndexNat, some hFiberIndex]
            let (compiledLogicalValue, logicalIndex, hCompiledLogicalValue) ←
              compileReductionLogicalIndex checked checkedValue
                outputFin fiberFin outputIndexNat fiberIndexNat
            let logicalIndexBound ← mkAppM ``Fin.isLt #[logicalIndex]
            let compiledLogicalBound ←
              indexBoundFromValueEquality logicalInputSize
                hCompiledLogicalValue logicalIndexBound
            let hNativeCompiledLogicalBound ←
              nativeLoopIndexBound compiledLogicalValue
                hLogicalInputBound compiledLogicalBound
            let compiledLogicalIndex ←
              mkAppOptM ``Fin.mk #[
                some logicalInputSize, some compiledLogicalValue,
                some compiledLogicalBound]
            let hCompiledLogicalIndex ←
              finIndexEquality logicalInputSize compiledLogicalIndex
                logicalIndex hCompiledLogicalValue
            let sourceIndex := mkApp inputFlatMap compiledLogicalIndex
            let semanticSourceIndex := mkApp inputFlatMap logicalIndex
            let sourceIndexValueFunction ←
              withLocalDeclD `logicalIndex
                  (← mkAppM ``Fin #[logicalInputSize]) fun index => do
                let sourceIndex := mkApp inputFlatMap index
                let value ← mkAppM ``Fin.val #[sourceIndex]
                mkLambdaFVars #[index] value
            let hSourceIndex ←
              mkAppM ``congrArg #[
                sourceIndexValueFunction, hCompiledLogicalIndex]
            let sourceIndexValue ← mkAppM ``Fin.val #[sourceIndex]
            let normalizedSourceIndexValue ← zetaReduce sourceIndexValue
            let (optimizedSourceIndexValue, hSourceIndexOptimized) ←
              simplifyAffineIndex normalizedSourceIndexValue
            let hOptimizedRuntimeSourceIndex ←
              mkAppM ``Eq.symm #[hSourceIndexOptimized]
            let hOptimizedSourceIndex ←
              mkAppM ``Eq.trans #[
                hOptimizedRuntimeSourceIndex, hSourceIndex]
            let sourceIndexBound ← mkAppM ``Fin.isLt #[sourceIndex]
            let optimizedSourceIndexBound ←
              indexBoundFromValueEquality sourceSize
                hOptimizedRuntimeSourceIndex sourceIndexBound
            let (sourceValue, hSourceValue) ←
              compileInputRead sourceTensor sourceSize
                optimizedSourceIndexValue optimizedSourceIndexValue
                semanticSourceIndex hOptimizedSourceIndex
                useNativeSourceIndex
                (some optimizedSourceIndexBound)
                #[
                  hNativeOutputIndex, hNativeFiberIndex,
                  hNativeCompiledLogicalBound]
            let nativeRead ←
              mkLambdaFVars #[
                outputIndex, hOutputIndex, fiberIndex, hFiberIndex]
                sourceValue
            let expectedValue := mkApp read logicalIndex
            let hSourceValue ←
              withTransparency .all <|
                mkExpectedTypeHint hSourceValue
                  (← mkEq sourceValue expectedValue)
            let hNativeRead ←
              mkLambdaFVars #[
                outputIndex, hOutputIndex, fiberIndex, hFiberIndex]
                hSourceValue
            return (nativeRead, hNativeRead)
  let (nativeRead, hNativeRead) := nativeReadData
  let nativeRead ← sealNativeReductionCallback nativeRead
  let hNativeReadType ← inferType hNativeRead
  let hNativeRead ← sealCertificate hNativeReadType hNativeRead
  let implementation ←
    mkAppM ``nativeReduceFoldTensorFromFlat #[
      step, initial, finish, checked, hKind, inputTensor, read, hRead,
      outputBound, hOutputBound, fiberBound, hFiberBound,
      nativeRead, hNativeRead]
  let compiled ←
    mkAppM ``Lowering.reduceFoldTensorFromFlat #[
      step, initial, finish, checked, hKind, inputTensor, read, hRead]
  let reference ←
    mkAppM ``Lowering.reduceFoldTensor #[
      step, initial, finish, checked, hKind, inputTensor]
  let hImplementation ←
    mkAppM ``nativeReduceFoldTensorFromFlat_correct #[
      step, initial, finish, checked, hKind, inputTensor, read, hRead,
      outputBound, hOutputBound, fiberBound, hFiberBound,
      nativeRead, hNativeRead]
  let hCompiled ←
    mkAppM ``Lowering.reduceFoldTensorFromFlat_eq #[
      step, initial, finish, checked, hKind, inputTensor, read, hRead]
  let nativeKernel ← mkAppM ``nativeTensorKernel #[
    reference, compiled, implementation, hCompiled, hImplementation]
  if inputTensor == semanticInputTensor then
    return some nativeKernel
  let inputTensorType ← inferType inputTensor
  let reduceInput ←
    withLocalDeclD `logicalInput inputTensorType fun logicalInput => do
      let body ←
        mkAppM ``Lowering.reduceFoldTensor #[
          step, initial, finish, checked, hKind, logicalInput]
      mkLambdaFVars #[logicalInput] body
  let hReference ← mkAppM ``congrArg #[reduceInput, hInputTensor]
  let hNativeReference ←
    mkAppM ``nativeTensorKernel_correct #[
      reference, compiled, implementation, hCompiled, hImplementation]
  let hNativeSemantic ←
    mkAppM ``Eq.trans #[hNativeReference, hReference]
  return some <| ← mkAppM ``certifiedNativeTensor #[
    nativeKernel, hNativeSemantic]

end TorchLean.Tensor.Internal.Elab.Impl
