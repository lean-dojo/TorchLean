/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.LibTorch.Tape
public import TorchLean.NN.TensorTransfer
public import TorchLean.Tensor.SomeTensor

/-!
# CUDA Tensor Storage Bridge

Adapt the public `TensorTransfer` capability to the dtype-carrying row-major buffers owned by the
eager CUDA tape. A scalar with a binary configured encoding travels as its packed words, and every
other scalar travels as `Float` values converted to its native dtype.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

namespace TensorTransfer

/-- Upload a tensor with its shape, as packed configured words or in the scalar's native dtype. -/
def toSomeBuffer {α : Type} [Storage α] [TensorTransfer α] {s : Shape}
    (tensor : Tensor α s) : IO LibTorch.SomeBuffer := do
  if let some scalar := TensorTransfer.encoding? (α := α) then
    if let .binary format := scalar.precision then
      let bytes ← IO.ofExcept (scalar.pack tensor)
      let buffer ← LibTorch.Buffer.ofEncodedIO bytes format
        scalar.precision.bytes.toUInt64
      return { s := s, buf := buffer }
  let host ← TensorTransfer.toFloatTensor tensor
  let values := host.to FloatArray
  let dtype ← TensorTransfer.dtype (α := α)
  let buffer ← LibTorch.Buffer.ofFloatArrayIO values dtype
  pure { s := s, buf := buffer }

/-- Download a buffer as a tensor of shape `s`, checking its format or dtype and its length. -/
def ofBuffer {α : Type} [Storage α] [TensorTransfer α] {s : Shape}
    (buffer : LibTorch.Buffer) : IO (Tensor α s) := do
  if let some scalar := TensorTransfer.encoding? (α := α) then
    if let .binary format := scalar.precision then
      unless LibTorch.Buffer.format? buffer == some format do
        throw <| IO.userError "torch: buffer format does not match the requested scalar"
      return ← IO.ofExcept <| scalar.unpack s
        (← LibTorch.Buffer.toEncodedIO buffer)
  unless LibTorch.Buffer.dtype buffer == (← TensorTransfer.dtype (α := α)) do
    throw <| IO.userError "torch: native buffer dtype does not match the requested scalar"
  let values := LibTorch.Buffer.toFloatArray buffer
  match (Tensor.from values).reshape? s with
  | some tensor =>
      TensorTransfer.ofFloatTensor tensor
  | none =>
      throw <| IO.userError <|
        s!"torch: cuda: bad buffer length (expected {Shape.size s}, " ++
          s!"got {values.size})"

/-- Download a shape-erased buffer through the scalar type's runtime transfer representation. -/
def ofSomeBuffer {α : Type} [Storage α] [TensorTransfer α]
    (stored : LibTorch.SomeBuffer) : IO (SomeTensor α) := do
  pure <| SomeTensor.ofTensor (← ofBuffer (α := α) (s := stored.s) stored.buf)

end TensorTransfer
end TorchLean
