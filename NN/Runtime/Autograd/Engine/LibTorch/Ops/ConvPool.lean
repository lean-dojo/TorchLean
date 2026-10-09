/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops.Core
public import NN.Spec.Layers.Pooling.Spatial
public import NN.Tensor.Conversion
public import NN.Runtime.Autograd.Engine.LibTorch.ConvPool

/-!
# CUDA Tape Operations: Convolution and Pooling

Every entry point rejects zero strides and geometry that cannot cross the `UInt32` ABI, validates
input lengths through `requireValue`, and checks forward native outputs before recording them on the
tape. Pointer lifetime and device ownership remain responsibilities of the native buffer layer.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-! ## Convolution and pooling -/

/-- Shared rank, kernel, and stride checks, in the order required by every native spatial op. -/
@[inline] def Internal.validateSpatialGeometry {d : Nat} (opName : String)
    (kernel stride : TorchLean.Tensor Nat [d]) : Result Unit := do
  if d = 0 then
    throw s!"autograd: cuda: {opName}: d=0 is not supported"
  if d > 3 then
    throw s!"autograd: cuda: {opName}: spatial rank must be at most 3"
  if !decide (∀ i : Fin d, kernel.getScalar i ≠ 0) then
    throw s!"autograd: cuda: {opName}: kernel_size must be > 0"
  if !decide (∀ i : Fin d, stride.getScalar i ≠ 0) then
    throw s!"autograd: cuda: {opName}: stride must be > 0"

/-- Convert spatial metadata to the native ABI, checking axis sizes before element counts. -/
@[inline] def Internal.spatialMetadata {d : Nat} (opName : String)
    (inSpatial kernel stride padding : TorchLean.Tensor Nat [d]) :
    Result (Array Nat × Array Nat × Array Nat × Array Nat) := do
  let inputDims := Array.ofFn (fun i : Fin d => inSpatial.getScalar i)
  let kernelDims := Array.ofFn (fun i : Fin d => kernel.getScalar i)
  let strides := Array.ofFn (fun i : Fin d => stride.getScalar i)
  let pads := Array.ofFn (fun i : Fin d => padding.getScalar i)
  validateU32Dimensions opName inputDims
  validateU32Dimensions opName kernelDims
  validateU32Dimensions opName strides
  validateU32Dimensions opName pads
  let _ ← AnyBuffer.numelU32 (Shape.ofList (inSpatial.to (List Nat)))
  let _ ← AnyBuffer.numelU32 (Shape.ofList (kernel.to (List Nat)))
  pure (inputDims, kernelDims, strides, pads)

/-- One-, two-, or three-dimensional convolution through LibTorch. -/
@[inline] def conv
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (t : Tape) (kernelId biasId inputId : Nat) :
  Result (Tape × Nat) := do
  Internal.validateSpatialGeometry "conv" kernel stride

  let inC32 ← AnyBuffer.natToU32Checked inC
  let outC32 ← AnyBuffer.natToU32Checked outC
  let (inSpatialArr, kernelSpatialArr, strideArr, paddingArr) ←
    Internal.spatialMetadata "conv" inSpatial kernel stride padding

  let kernelShape : Shape :=
    Shape.ofList (outC :: inC :: kernel.to (List Nat))
  let inputShape : Shape :=
    Shape.ofList (inC :: inSpatial.to (List Nat))
  let outSpatial : TorchLean.Tensor Nat [d] :=
    Spec.convOutSpatial inSpatial kernel stride padding
  let _ ← AnyBuffer.numelU32 (Shape.ofList (outSpatial.to (List Nat)))
  let outShape : Shape :=
    Shape.ofList (outC :: outSpatial.to (List Nat))
  let _ ← AnyBuffer.numelU32 outShape

  let kernelBuf ← requireValue (t := t) kernelId kernelShape
  let biasBuf ← requireValue (t := t) biasId (.dim outC .scalar)
  let inputBuf ← requireValue (t := t) inputId inputShape

  let y :=
    torchleanConvFwdCuda inputBuf kernelBuf biasBuf
      inSpatialArr kernelSpatialArr strideArr paddingArr
      inC32 outC32
  let output ← AnyBuffer.validate { s := outShape, buf := y }

  let node : Node :=
    { name := some "conv"
      value := output
      requiresGrad := (t.getNode? kernelId).any (·.requiresGrad) ||
        (t.getNode? biasId).any (·.requiresGrad) ||
        (t.getNode? inputId).any (·.requiresGrad)
      parents := #[kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dKernel, dBias, dInput) :=
          torchleanConvBwdCuda inputBuf kernelBuf dLdy.buf
            inSpatialArr kernelSpatialArr strideArr paddingArr
            inC32 outC32
        pure #[
          (kernelId, { s := kernelShape, buf := dKernel }),
          (biasId, { s := .dim outC .scalar, buf := dBias }),
          (inputId, { s := inputShape, buf := dInput })
        ] }
  pure (t.addNode node)

/-- One-, two-, or three-dimensional transpose convolution through LibTorch. -/
@[inline] def convTranspose
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (t : Tape) (kernelId biasId inputId : Nat) :
  Result (Tape × Nat) := do
  Internal.validateSpatialGeometry "conv_transpose" kernel stride

  let inC32 ← AnyBuffer.natToU32Checked inC
  let outC32 ← AnyBuffer.natToU32Checked outC
  let (inSpatialArr, kernelSpatialArr, strideArr, paddingArr) ←
    Internal.spatialMetadata "conv_transpose" inSpatial kernel stride padding

  -- NOTE: for transposed conv, kernel layout is `(inC, outC, kernelSpatial...)`.
  let kernelShape : Shape :=
    Shape.ofList (inC :: outC :: kernel.to (List Nat))
  let inputShape : Shape :=
    Shape.ofList (inC :: inSpatial.to (List Nat))
  let outSpatial : TorchLean.Tensor Nat [d] :=
    Spec.convTransposeOutSpatial inSpatial kernel stride padding
  let _ ← AnyBuffer.numelU32 (Shape.ofList (outSpatial.to (List Nat)))
  let outShape : Shape :=
    Shape.ofList (outC :: outSpatial.to (List Nat))
  let _ ← AnyBuffer.numelU32 outShape

  let kernelBuf ← requireValue (t := t) kernelId kernelShape
  let biasBuf ← requireValue (t := t) biasId (.dim outC .scalar)
  let inputBuf ← requireValue (t := t) inputId inputShape

  let y :=
    torchleanConvTransposeFwdCuda inputBuf kernelBuf biasBuf
      inSpatialArr kernelSpatialArr strideArr paddingArr
      inC32 outC32
  let output ← AnyBuffer.validate { s := outShape, buf := y }

  let node : Node :=
    { name := some "conv_transpose"
      value := output
      requiresGrad := (t.getNode? kernelId).any (·.requiresGrad) ||
        (t.getNode? biasId).any (·.requiresGrad) ||
        (t.getNode? inputId).any (·.requiresGrad)
      parents := #[kernelId, biasId, inputId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dKernel, dBias, dInput) :=
          torchleanConvTransposeBwdCuda inputBuf kernelBuf dLdy.buf
            inSpatialArr kernelSpatialArr strideArr paddingArr
            inC32 outC32
        pure #[
          (kernelId, { s := kernelShape, buf := dKernel }),
          (biasId, { s := .dim outC .scalar, buf := dBias }),
          (inputId, { s := inputShape, buf := dInput })
        ] }
  pure (t.addNode node)

/-- One-, two-, or three-dimensional max pooling through LibTorch. -/
@[inline] def maxPool
    {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  Internal.validateSpatialGeometry "max_pool" kernel stride

  let inC32 ← AnyBuffer.natToU32Checked C
  let (inSpatialArr, kernelArr, strideArr, paddingArr) ←
    Internal.spatialMetadata "max_pool" inSpatial kernel stride padding

  let inputShape : Shape :=
    Shape.ofList (C :: inSpatial.to (List Nat))
  let outSpatial : TorchLean.Tensor Nat [d] :=
    Spec.poolOutSpatialPad inSpatial kernel stride padding
  let _ ← AnyBuffer.numelU32 (Shape.ofList (outSpatial.to (List Nat)))
  let outShape : Shape :=
    Shape.ofList (C :: outSpatial.to (List Nat))
  let _ ← AnyBuffer.numelU32 outShape

  let xBuf ← requireValue (t := t) xId inputShape
  let y :=
    torchleanMaxPoolFwdCuda xBuf inSpatialArr kernelArr strideArr paddingArr inC32
  let output ← AnyBuffer.validate { s := outShape, buf := y }

  let node : Node :=
    { name := some "max_pool"
      value := output
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanMaxPoolBwdCuda xBuf dLdy.buf
            inSpatialArr kernelArr strideArr paddingArr inC32
        pure #[(xId, { s := inputShape, buf := dx })] }
  pure (t.addNode node)

/--
One-, two-, or three-dimensional smooth max pooling.

`beta` is checked in the input dtype, where conversion can underflow a nonzero value to zero
or overflow a finite value to infinity.
-/
@[inline] def smoothMaxPool
    {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (t : Tape) (xId : Nat) (beta : Float) : Result (Tape × Nat) := do
  Internal.validateSpatialGeometry "smooth_max_pool" kernel stride

  let inC32 ← AnyBuffer.natToU32Checked C
  let (inSpatialArr, kernelArr, strideArr, paddingArr) ←
    Internal.spatialMetadata "smooth_max_pool" inSpatial kernel stride padding

  let inputShape : Shape :=
    Shape.ofList (C :: inSpatial.to (List Nat))
  let outSpatial : TorchLean.Tensor Nat [d] :=
    Spec.poolOutSpatialPad inSpatial kernel stride padding
  let _ ← AnyBuffer.numelU32 (Shape.ofList (outSpatial.to (List Nat)))
  let outShape : Shape :=
    Shape.ofList (C :: outSpatial.to (List Nat))
  let _ ← AnyBuffer.numelU32 outShape

  let xBuf ← requireValue (t := t) xId inputShape
  let effectiveBeta := if Buffer.dtype xBuf == .float32 then beta.toFloat32.toFloat else beta
  if !effectiveBeta.isFinite || effectiveBeta == 0 then
    throw "autograd: cuda: smooth_max_pool: beta must be finite and nonzero"
  let y :=
    torchleanSmoothMaxPoolFwdCuda xBuf beta
      inSpatialArr kernelArr strideArr paddingArr
      inC32
  let output ← AnyBuffer.validate { s := outShape, buf := y }

  let node : Node :=
    { name := some "smooth_max_pool"
      value := output
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanSmoothMaxPoolBwdCuda xBuf dLdy.buf beta
            inSpatialArr kernelArr strideArr paddingArr
            inC32
        pure #[(xId, { s := inputShape, buf := dx })] }
  pure (t.addNode node)

/-- One-, two-, or three-dimensional average pooling through LibTorch. -/
@[inline] def avgPool
    {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  Internal.validateSpatialGeometry "avg_pool" kernel stride

  let inC32 ← AnyBuffer.natToU32Checked C
  let (inSpatialArr, kernelArr, strideArr, paddingArr) ←
    Internal.spatialMetadata "avg_pool" inSpatial kernel stride padding

  let inputShape : Shape :=
    Shape.ofList (C :: inSpatial.to (List Nat))
  let outSpatial : TorchLean.Tensor Nat [d] :=
    Spec.poolOutSpatialPad inSpatial kernel stride padding
  let _ ← AnyBuffer.numelU32 (Shape.ofList (outSpatial.to (List Nat)))
  let outShape : Shape :=
    Shape.ofList (C :: outSpatial.to (List Nat))
  let _ ← AnyBuffer.numelU32 outShape

  let xBuf ← requireValue (t := t) xId inputShape
  let y :=
    torchleanAvgPoolFwdCuda xBuf
      inSpatialArr kernelArr strideArr paddingArr
      inC32
  let output ← AnyBuffer.validate { s := outShape, buf := y }

  let node : Node :=
    { name := some "avg_pool"
      value := output
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let dx :=
          torchleanAvgPoolBwdCuda dLdy.buf
            inSpatialArr kernelArr strideArr paddingArr
            inC32
        pure #[(xId, { s := inputShape, buf := dx })] }
  pure (t.addNode node)
end Tape

end LibTorch
end Autograd
end Runtime
