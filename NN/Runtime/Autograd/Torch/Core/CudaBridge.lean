/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Convert
public import NN.Runtime.Autograd.Engine.LibTorch.Tape
public import NN.Runtime.Autograd.Torch.Core.TensorTransfer
public import NN.Spec.Core.Tensor.SomeTensor

/-!
# CUDA Tensor Storage Bridge

Adapt the public `TensorTransfer` capability to dtype-carrying row-major storage owned by the eager
CUDA tape.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean

namespace Internal
namespace CudaBridge

/-- Upload using the scalar's native dtype, preserving the tensor's shape. -/
def toAnyBuffer {α : Type} [TorchLean.Storage α] [TensorTransfer α] {s : Shape}
    (tensor : Tensor α s) : IO Runtime.Autograd.LibTorch.AnyBuffer := do
  if let some scalar := TensorTransfer.encoding? (α := α) then
    if let .binary format := scalar.precision then
      let bytes ← IO.ofExcept (scalar.pack tensor)
      let buffer ← Runtime.Autograd.LibTorch.Buffer.ofEncodedIO bytes format
        scalar.precision.bytes.toUInt64
      return { s := s, buf := buffer }
  let host ← TensorTransfer.toFloatTensor tensor
  let values := Runtime.Autograd.LibTorch.Convert.flattenFloat host
  let dtype ← TensorTransfer.dtype (α := α)
  let buffer ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO values dtype
  pure { s := s, buf := buffer }

/-- Download a buffer with the requested shape, checking its element count. -/
def ofBuffer {α : Type} [TorchLean.Storage α] [TensorTransfer α] {s : Shape}
    (buffer : Runtime.Autograd.LibTorch.Buffer) : IO (Tensor α s) := do
  if let some scalar := TensorTransfer.encoding? (α := α) then
    if let .binary format := scalar.precision then
      unless Runtime.Autograd.LibTorch.Buffer.format? buffer == some format do
        throw <| IO.userError "torch: buffer format does not match the requested scalar"
      return ← IO.ofExcept <| scalar.unpack s
        (← Runtime.Autograd.LibTorch.Buffer.toEncodedIO buffer)
  unless Runtime.Autograd.LibTorch.Buffer.dtype buffer == (← TensorTransfer.dtype (α := α)) do
    throw <| IO.userError "torch: native buffer dtype does not match the requested scalar"
  let values := Runtime.Autograd.LibTorch.Buffer.toFloatArray buffer
  match Runtime.Autograd.LibTorch.Convert.unflattenFloat? (s := s) values with
  | some tensor =>
      TensorTransfer.ofFloatTensor tensor
  | none =>
      throw <| IO.userError <|
        s!"torch: cuda: bad buffer length (expected {Spec.Shape.size s}, " ++
          s!"got {values.size})"

/-- Download a shape-erased buffer through the scalar type's runtime transfer representation. -/
def ofAnyBuffer {α : Type} [TorchLean.Storage α] [TensorTransfer α]
    (stored : Runtime.Autograd.LibTorch.AnyBuffer) : IO (Spec.SomeTensor α) := do
  pure <| Spec.SomeTensor.ofTensor (← ofBuffer (α := α) (s := stored.s) stored.buf)

end CudaBridge
end Internal
end Torch
end Autograd
end Runtime
