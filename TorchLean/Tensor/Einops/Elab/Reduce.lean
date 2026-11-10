/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Lowering.Reduce.View
public import TorchLean.Tensor.Native.Construction -- shake: keep
public meta import TorchLean.Tensor.Einops.Elab.Kernel.Affine
public meta import TorchLean.Tensor.Einops.Elab.Kernel.Index
public meta import TorchLean.Tensor.Einops.Elab.Reduce.Index
public import TorchLean.Tensor.Einops.Elab.Kernel.Affine
public import TorchLean.Tensor.Einops.Elab.Kernel.Index
public import TorchLean.Tensor.Einops.Elab.Reduce.Index

/-!
# Certified native reduction loops

Concrete reductions use one native output loop and one native fiber loop. The
pointwise reader certificate connects source-buffer reads to the row-major
reduction semantics, so the scalar operation order is unchanged.
-/

@[expose] public section

namespace TorchLean.Native

universe u v w

open Einops
open Lowering
open Lowering.Reduction
open Shape (Coord)

/--
Reduce every fiber with native output and fiber counters.

The executable callback receives both counters as `USize`. Its pointwise
certificate is erased and proves agreement with the existing flat-reader
reduction at the corresponding bounded indices.
-/
@[inline] def reduceFoldView
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β)
    (finish : β → Nat → γ)
    (checked : Transform)
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
            (inputFlatIndex checked
              (Coord.unlinearize
                ⟨outputIndex.toNat, hOutputIndex⟩)
              ⟨fiberIndex.toNat, hFiberIndex⟩)) :
    checked.OutputTensor γ :=
  tensorOfFlatFn outputBound hOutputBound
    (fun outputIndex hOutputIndex =>
      finish
        (finFoldl
          (Shape.size checked.reductionShape)
          fiberBound hFiberBound
          (fun total fiberIndex hFiberIndex =>
            step total <|
              nativeRead outputIndex hOutputIndex fiberIndex hFiberIndex)
          initial)
        checked.reductionFiberSize)
    (fun outputIndex =>
      finish
        (foldlFromFlat step initial checked read <|
          Coord.unlinearize outputIndex)
        checked.reductionFiberSize)
    (by
      intro outputIndex hOutputIndex
      apply congrArg
        (fun total => finish total checked.reductionFiberSize)
      apply finFoldl_eq_fin_foldl_of_eq
      intro total fiberIndex hFiberIndex
      rw [hNativeRead outputIndex hOutputIndex fiberIndex hFiberIndex])

/--
The native ordered reduction is exactly the existing flat-reader reduction.
-/
theorem reduceFoldView_correct
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β)
    (finish : β → Nat → γ)
    (checked : Transform)
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
            (inputFlatIndex checked
              (Coord.unlinearize
                ⟨outputIndex.toNat, hOutputIndex⟩)
              ⟨fiberIndex.toNat, hFiberIndex⟩)) :
    reduceFoldView step initial finish checked hKind
        inputTensor read hRead outputBound hOutputBound fiberBound
        hFiberBound nativeRead hNativeRead =
      Lowering.reduceFoldView step initial finish checked hKind
        inputTensor read hRead := by
  apply tensorOfFlatFn_correct

end TorchLean.Native

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Compile an ordered flat-reader reduction to nested native loops.

The source map may be the identity map of an ordinary tensor or a certified
map recovered from a preceding shape-only transform.

Nonconcrete or nonportable output, fiber, and logical-input sizes return
`none`. Once compilation begins, certificate errors propagate to the caller.
This function does not transactionally catch failed optimizations.
-/
def compileNativeReduceFold?
    (step initial finish checked hKind inputTensor semanticInputTensor
      hInputTensor read hRead sourceTensor inputFlatMap : Expr) :
    TermElabM (Option Expr) := do
  let value ← mkAppM ``Transform.value #[checked]
  let outputShape ← mkAppM ``TransformPlan.output #[value]
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let fiberShape ←
    mkAppM ``Transform.reductionShape #[checked]
  let fiberSize ← mkAppM ``Shape.size #[fiberShape]
  let valueNormalized ←
    mkAppM ``TransformPlan.normalized #[value]
  let logicalInputShape ←
    mkAppM ``NormalForm.input #[valueNormalized]
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
  unless sourceTensorType.isAppOfArity ``Tensor.Rep 3 do
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
              compileReductionLogicalIndex checked
                outputFin fiberFin outputIndexNat fiberIndexNat
            let logicalIndexBound ← mkAppM ``Fin.isLt #[logicalIndex]
            let compiledLogicalBound ←
              mkAppM ``lt_of_eq_of_lt #[hCompiledLogicalValue, logicalIndexBound]
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
              mkAppM ``lt_of_eq_of_lt #[
                hOptimizedRuntimeSourceIndex, sourceIndexBound]
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
  let nativeRead ← sealAuxDefinition `_einops_native_reduce_callback nativeRead
  let hNativeReadType ← inferType hNativeRead
  let hNativeRead ← sealCertificate hNativeReadType hNativeRead
  let implementation ←
    mkAppM ``Native.reduceFoldView #[
      step, initial, finish, checked, hKind, inputTensor, read, hRead,
      outputBound, hOutputBound, fiberBound, hFiberBound,
      nativeRead, hNativeRead]
  let compiled ←
    mkAppM ``Lowering.reduceFoldView #[
      step, initial, finish, checked, hKind, inputTensor, read, hRead]
  let reference ←
    mkAppM ``Lowering.reduceFold #[
      step, initial, finish, checked, hKind, inputTensor]
  let hImplementation ←
    mkAppM ``Native.reduceFoldView_correct #[
      step, initial, finish, checked, hKind, inputTensor, read, hRead,
      outputBound, hOutputBound, fiberBound, hFiberBound,
      nativeRead, hNativeRead]
  let hCompiled ←
    mkAppM ``Lowering.reduceFoldView_eq #[
      step, initial, finish, checked, hKind, inputTensor, read, hRead]
  let nativeKernel ← mkAppM ``Native.tensorKernel #[
    reference, compiled, implementation, hCompiled, hImplementation]
  if inputTensor == semanticInputTensor then
    return some nativeKernel
  let inputTensorType ← inferType inputTensor
  let reduceInput ←
    withLocalDeclD `logicalInput inputTensorType fun logicalInput => do
      let body ←
        mkAppM ``Lowering.reduceFold #[
          step, initial, finish, checked, hKind, logicalInput]
      mkLambdaFVars #[logicalInput] body
  let hReference ← mkAppM ``congrArg #[reduceInput, hInputTensor]
  let hNativeReference ←
    mkAppM ``Native.tensorKernel_correct #[
      reference, compiled, implementation, hCompiled, hImplementation]
  let hNativeSemantic ←
    mkAppM ``Eq.trans #[hNativeReference, hReference]
  return some <| ← mkAppM ``Native.certifiedTensor #[
    nativeKernel, hNativeSemantic]

end TorchLean.Einops.Elab
