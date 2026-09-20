/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Convert
public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Torch.Core.TensorTransfer
public import NN.Spec.Core.Tensor.SomeTensor

/-!
# CUDA Tensor Storage Bridge

Adapt the public `TensorTransfer` capability to the float32 row-major storage owned by the eager
CUDA tape.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean

namespace Internal
namespace CudaBridge

/-- Upload a tensor to CUDA float32 storage through its runtime transfer representation. -/
def toAnyBuffer {α : Type} [TorchLean.Storage α] [TensorTransfer α] {s : Shape}
    (tensor : Tensor α s) : IO Runtime.Autograd.Cuda.AnyBuffer := do
  let host ← TensorTransfer.toFloatTensor tensor
  let values := Runtime.Autograd.Cuda.Convert.flattenFloat host
  let buffer ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO values
  pure { s := s, buf := buffer }

/-- Download a buffer with the requested shape, checking its element count. -/
def ofBuffer {α : Type} [TorchLean.Storage α] [TensorTransfer α] {s : Shape}
    (buffer : Runtime.Autograd.Cuda.Buffer) : IO (Tensor α s) := do
  let values := Runtime.Autograd.Cuda.Buffer.toFloatArray buffer
  match Runtime.Autograd.Cuda.Convert.unflattenFloat? (s := s) values with
  | some tensor =>
      TensorTransfer.ofFloatTensor tensor
  | none =>
      throw <| IO.userError <|
        s!"torch: cuda: bad buffer length (expected {Spec.Shape.size s}, " ++
          s!"got {values.size})"

/-- Download a shape-erased buffer through the scalar type's runtime transfer representation. -/
def ofAnyBuffer {α : Type} [TorchLean.Storage α] [TensorTransfer α]
    (stored : Runtime.Autograd.Cuda.AnyBuffer) : IO (Spec.SomeTensor α) := do
  pure <| Spec.SomeTensor.ofTensor (← ofBuffer (α := α) (s := stored.s) stored.buf)

end CudaBridge
end Internal
end Torch
end Autograd
end Runtime
