/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.IEEE754.Native
public import NN.Runtime.Autograd.Engine.LibTorch.Trusted
public import NN.Spec.Core.Tensor.Core
public import NN.Kernel.Scalar

/-!
# Runtime Tensor Transfer

Backend-neutral conversion between executable tensors and the host `Float` representation used at
native runtime boundaries.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean
open FloatLib.Floats

/--
Conversion between an executable scalar type and the host `Float` representation used at native
runtime boundaries.

Native transfer and host inspection are separate operations. A scalar semantics may allow every
element to be inspected as `Float` without claiming that a native tensor has the same arithmetic or
storage representation.
-/
class TensorTransfer (α : Type) [TorchLean.Storage α] where
  /-- Native dtype, when this carrier can use the existing native tensor runtime.
  The default keeps custom scalar types on CPU rather than narrowing them through `Float`.
  This does not test device availability or certify native arithmetic. -/
  dtype? : Option Runtime.Autograd.LibTorch.Dtype := none
  /-- Lossless configured encoding for custom arithmetic on the existing tape. Native tensor
  kernels and optimizers do not acquire this capability merely by supporting a byte encoding. -/
  encoding? : Option (NN.Kernel.Scalar α) := none
  /-- Encode a tensor in the host representation used by native runtimes. -/
  toFloatTensor : {s : Shape} → Tensor α s → IO (Tensor Float s)
  /-- Decode a tensor received from a native runtime without changing its shape. -/
  ofFloatTensor : {s : Shape} → Tensor Float s → IO (Tensor α s)
  /-- Convert one scalar to the host representation used by native runtimes. -/
  toFloat : α → IO Float
  /-- Read a tensor for host-side inspection without claiming a native storage representation. -/
  readFloatTensor : {s : Shape} → Tensor α s → IO (Tensor Float s) := toFloatTensor

/-- Whether a scalar has a native tape representation. Device availability is checked separately. -/
def TensorTransfer.supportsGpu {α : Type} [Storage α] [TensorTransfer α] : Bool :=
  (TensorTransfer.dtype? (α := α)).isSome

/-- Whether custom arithmetic can retain this scalar's configured words on the GPU tape. -/
def TensorTransfer.supportsEncodedGpu {α : Type} [Storage α] [TensorTransfer α] : Bool :=
  match TensorTransfer.encoding? (α := α) with
  | some scalar => match scalar.precision with
    | .binary _ => scalar.precision.supportsGpu
    | .native _ => false
  | none => false

/-- Native dtype required at a transfer boundary; unsupported carriers must stay on CPU. -/
def TensorTransfer.dtype {α : Type} [Storage α] [TensorTransfer α] :
    IO Runtime.Autograd.LibTorch.Dtype :=
  match TensorTransfer.dtype? (α := α) with
  | some dtype => pure dtype
  | none => throw <| IO.userError
      "torch: this scalar has no native tensor representation; run it on CPU"

/-- `Float` retains binary64 on the native tape, without an intermediate binary32 conversion. -/
instance (priority := 1000) : TensorTransfer Float where
  dtype? := some .float64
  toFloatTensor := fun tensor => pure tensor
  ofFloatTensor := fun tensor => pure tensor
  toFloat := pure

/-- Native binary32 transfers preserve the runtime's float32 wire representation. -/
instance (priority := 1000) : TensorTransfer Float32 where
  dtype? := some .float32
  toFloatTensor := fun tensor => pure (TorchLean.Tensor.map Float32.toFloat tensor)
  ofFloatTensor := fun tensor => pure (TorchLean.Tensor.map Float.toFloat32 tensor)
  toFloat := fun x => pure x.toFloat

/--
Host-side conversion for FloatLib's configured IEEE-754 binary32 scalar.

The configured arithmetic uses software kernels. Scalar and tensor readback to `Float` remain
available for reports and checkpoints, but this does not select native CUDA arithmetic. Finite
binary32 values embed exactly in binary64; the native conversion canonicalizes NaN payloads.
-/
instance (priority := 1000) :
    TensorTransfer (ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)) where
  encoding? := some inferInstance
  toFloatTensor := fun {_s} _ =>
    throw <| IO.userError
      "torch: configured binary32 supports host readback; select native arithmetic for CUDA"
  ofFloatTensor := fun {_s} _ =>
    throw <| IO.userError
      "torch: configured binary32 supports host readback; select native arithmetic for CUDA"
  toFloat := fun x => pure (ExecFloat.Binary.toFloat32 x).toFloat
  readFloatTensor := fun tensor => pure <|
    TorchLean.Tensor.map (fun x => (ExecFloat.Binary.toFloat32 x).toFloat) tensor

/-- Configured scalar encodings do not imply support for the native operator catalogue. -/
instance (priority := 100) (α : Type) [TorchLean.Storage α] [NN.Kernel.Scalar α] :
    TensorTransfer α where
  encoding? := some inferInstance
  toFloatTensor := fun {_s} _ =>
    throw <| IO.userError "torch: configured words cannot be converted to a native tensor"
  ofFloatTensor := fun {_s} _ =>
    throw <| IO.userError "torch: native tensors cannot supply configured words"
  toFloat := fun _ =>
    throw <| IO.userError "torch: inspect this scalar in its own format"

/--
CPU-preserving fallback for scalar types without a native tensor representation.

Add a higher-priority `TensorTransfer α` instance when a scalar type has an honest host `Float`
encoding. CPU execution does not invoke these failing transfer operations.
-/
instance (priority := 10) (α : Type) [TorchLean.Storage α] : TensorTransfer α where
  toFloatTensor := fun {_s} _ =>
    throw <| IO.userError <|
      "torch: this scalar type cannot be converted to a native tensor; " ++
        "use Float for native execution or run this scalar on CPU"
  ofFloatTensor := fun {_s} _ =>
    throw <| IO.userError <|
      "torch: native tensors cannot be converted to this scalar type; " ++
        "use Float for native execution or run this scalar on CPU"
  toFloat := fun _ =>
    throw <| IO.userError <|
      "torch: this scalar type cannot be converted to a native scalar; " ++
        "use Float for native execution or run this scalar on CPU"

end Torch
end Autograd
end Runtime
