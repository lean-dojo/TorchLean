/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.Core
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Pooling.Spatial
public import NN.Tensor.Conversion
public import NN.Runtime.Autograd.Engine.Cuda.ConvPool

/-!
# CUDA Tape Operations: Convolution and Pooling

Every entry point rejects zero strides and geometry that cannot cross the `UInt32` ABI, validates
input lengths through `requireValue`, and checks forward native outputs before recording them on the
tape. Pointer lifetime and device ownership remain responsibilities of the native buffer layer.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-! ## Arbitrary-rank convolution and pooling -/

/-- Rank-polymorphic convolution via the CUDA ConvPool FFI (spatial rank $\le 8$). -/
def conv
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (t : Tape) (kernelId biasId inputId : Nat) :
  Result (Tape × Nat) := do
  if d = 0 then
    throw "autograd: cuda: conv: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: conv: rank too large (max 8)"
  if !decide (∀ i : Fin d, kernel.getScalar i ≠ 0) then
    throw "autograd: cuda: conv: kernel_size must be > 0"
  if !decide (∀ i : Fin d, stride.getScalar i ≠ 0) then
    throw "autograd: cuda: conv: stride must be > 0"

  let inC32 ← AnyBuffer.natToU32Checked inC
  let outC32 ← AnyBuffer.natToU32Checked outC
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.getScalar i)
  let kernelSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.getScalar i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.getScalar i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.getScalar i)
  validateU32Dimensions "conv" inSpatialArr
  validateU32Dimensions "conv" kernelSpatialArr
  validateU32Dimensions "conv" strideArr
  validateU32Dimensions "conv" paddingArr
  let _ ← AnyBuffer.numelU32 (Shape.ofList (inSpatial.to (List Nat)))
  let _ ← AnyBuffer.numelU32 (Shape.ofList (kernel.to (List Nat)))

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

/-- Rank-polymorphic transpose convolution via the CUDA ConvPool FFI (spatial rank $\le 8$). -/
def convTranspose
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (t : Tape) (kernelId biasId inputId : Nat) :
  Result (Tape × Nat) := do
  if d = 0 then
    throw "autograd: cuda: conv_transpose: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: conv_transpose: rank too large (max 8)"
  if !decide (∀ i : Fin d, kernel.getScalar i ≠ 0) then
    throw "autograd: cuda: conv_transpose: kernel_size must be > 0"
  if !decide (∀ i : Fin d, stride.getScalar i ≠ 0) then
    throw "autograd: cuda: conv_transpose: stride must be > 0"

  let inC32 ← AnyBuffer.natToU32Checked inC
  let outC32 ← AnyBuffer.natToU32Checked outC
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.getScalar i)
  let kernelSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.getScalar i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.getScalar i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.getScalar i)
  validateU32Dimensions "conv_transpose" inSpatialArr
  validateU32Dimensions "conv_transpose" kernelSpatialArr
  validateU32Dimensions "conv_transpose" strideArr
  validateU32Dimensions "conv_transpose" paddingArr
  let _ ← AnyBuffer.numelU32 (Shape.ofList (inSpatial.to (List Nat)))
  let _ ← AnyBuffer.numelU32 (Shape.ofList (kernel.to (List Nat)))

  -- NOTE: for transposed conv, kernel layout is `(inC, outC, kernelSpatial...)`.
  let kernelShape : Shape :=
    Shape.ofList (inC :: outC :: kernel.to (List Nat))
  let inputShape : Shape :=
    Shape.ofList (inC :: inSpatial.to (List Nat))
  let outSpatial : TorchLean.Tensor Nat [d] :=
    TorchLean.Tensor.ofFn (fun a =>
      Spec.convTransposeOutDim
        (inSpatial.getScalar a) (kernel.getScalar a) (stride.getScalar a) (padding.getScalar a))
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

/-- Rank-polymorphic max pooling via the CUDA ConvPool FFI (spatial rank $\le 8$). -/
def maxPool
    {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  if d = 0 then
    throw "autograd: cuda: max_pool: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: max_pool: rank too large (max 8)"
  if !decide (∀ i : Fin d, kernel.getScalar i ≠ 0) then
    throw "autograd: cuda: max_pool: kernel_size must be > 0"
  if !decide (∀ i : Fin d, stride.getScalar i ≠ 0) then
    throw "autograd: cuda: max_pool: stride must be > 0"

  let inC32 ← AnyBuffer.natToU32Checked C
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.getScalar i)
  let kernelArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.getScalar i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.getScalar i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.getScalar i)
  validateU32Dimensions "max_pool" inSpatialArr
  validateU32Dimensions "max_pool" kernelArr
  validateU32Dimensions "max_pool" strideArr
  validateU32Dimensions "max_pool" paddingArr
  let _ ← AnyBuffer.numelU32 (Shape.ofList (inSpatial.to (List Nat)))
  let _ ← AnyBuffer.numelU32 (Shape.ofList (kernel.to (List Nat)))

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
Rank-polymorphic smooth max pooling via the CUDA ConvPool FFI (spatial rank at most eight).

`beta` is checked after conversion to `Float32`, because conversion can underflow a nonzero `Float`
to zero or overflow a finite one to infinity.
-/
def smoothMaxPool
    {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (t : Tape) (xId : Nat) (beta : Float) : Result (Tape × Nat) := do
  let beta32 := Float.toFloat32 beta
  if !beta32.isFinite || beta32 == (0.0 : Float32) then
    throw "autograd: cuda: smooth_max_pool: beta must be finite and nonzero"
  if d = 0 then
    throw "autograd: cuda: smooth_max_pool: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: smooth_max_pool: rank too large (max 8)"
  if !decide (∀ i : Fin d, kernel.getScalar i ≠ 0) then
    throw "autograd: cuda: smooth_max_pool: kernel_size must be > 0"
  if !decide (∀ i : Fin d, stride.getScalar i ≠ 0) then
    throw "autograd: cuda: smooth_max_pool: stride must be > 0"

  let inC32 ← AnyBuffer.natToU32Checked C
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.getScalar i)
  let kernelArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.getScalar i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.getScalar i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.getScalar i)
  validateU32Dimensions "smooth_max_pool" inSpatialArr
  validateU32Dimensions "smooth_max_pool" kernelArr
  validateU32Dimensions "smooth_max_pool" strideArr
  validateU32Dimensions "smooth_max_pool" paddingArr
  let _ ← AnyBuffer.numelU32 (Shape.ofList (inSpatial.to (List Nat)))
  let _ ← AnyBuffer.numelU32 (Shape.ofList (kernel.to (List Nat)))

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

/-- Rank-polymorphic average pooling via the CUDA ConvPool FFI (spatial rank $\le 8$). -/
def avgPool
    {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  if d = 0 then
    throw "autograd: cuda: avg_pool: d=0 is not supported"
  if d > 8 then
    throw "autograd: cuda: avg_pool: rank too large (max 8)"
  if !decide (∀ i : Fin d, kernel.getScalar i ≠ 0) then
    throw "autograd: cuda: avg_pool: kernel_size must be > 0"
  if !decide (∀ i : Fin d, stride.getScalar i ≠ 0) then
    throw "autograd: cuda: avg_pool: stride must be > 0"

  let inC32 ← AnyBuffer.natToU32Checked C
  let inSpatialArr : Array Nat := Array.ofFn (fun i : Fin d => inSpatial.getScalar i)
  let kernelArr : Array Nat := Array.ofFn (fun i : Fin d => kernel.getScalar i)
  let strideArr : Array Nat := Array.ofFn (fun i : Fin d => stride.getScalar i)
  let paddingArr : Array Nat := Array.ofFn (fun i : Fin d => padding.getScalar i)
  validateU32Dimensions "avg_pool" inSpatialArr
  validateU32Dimensions "avg_pool" kernelArr
  validateU32Dimensions "avg_pool" strideArr
  validateU32Dimensions "avg_pool" paddingArr
  let _ ← AnyBuffer.numelU32 (Shape.ofList (inSpatial.to (List Nat)))
  let _ ← AnyBuffer.numelU32 (Shape.ofList (kernel.to (List Nat)))

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

end Cuda
end Autograd
end Runtime
