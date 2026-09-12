/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32.Compare
public import NN.Spec.Core.Tensor.Core

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

/--
Conversion between an executable scalar type and the host `Float` representation used at native
runtime boundaries.

Native transfer and host inspection are separate operations. A scalar semantics may allow every
element to be inspected as `Float` without claiming that a native tensor has the same arithmetic or
storage representation.
-/
class TensorTransfer (α : Type) [TorchLean.Storage α] where
  /-- Encode a tensor in the host representation used by native runtimes. -/
  toFloatTensor : {s : Shape} → Tensor α s → IO (Tensor Float s)
  /-- Decode a tensor received from a native runtime without changing its shape. -/
  ofFloatTensor : {s : Shape} → Tensor Float s → IO (Tensor α s)
  /-- Convert one scalar to the host representation used by native runtimes. -/
  toFloat : α → IO Float
  /-- Read a tensor for host-side inspection without claiming a native storage representation. -/
  readFloatTensor : {s : Shape} → Tensor α s → IO (Tensor Float s) := toFloatTensor

/-- Encode every tensor element with the supplied scalar conversion. -/
def TensorTransfer.toFloatTensorWith {α : Type} [TorchLean.Storage α]
    (encode : α → Float) {s : Shape}
    (tensor : Tensor α s) : IO (Tensor Float s) :=
  pure <| TorchLean.Tensor.map encode tensor

/-- Decode every tensor element with the supplied scalar conversion. -/
def TensorTransfer.ofFloatTensorWith {α : Type} [TorchLean.Storage α]
    (decode : Float → α) {s : Shape}
    (tensor : Tensor Float s) : IO (Tensor α s) :=
  pure <| TorchLean.Tensor.map decode tensor

/-- `Float` transfers preserve the runtime's host representation. -/
instance (priority := 1000) : TensorTransfer Float where
  toFloatTensor := fun tensor => pure tensor
  ofFloatTensor := fun tensor => pure tensor
  toFloat := pure

/-- Native binary32 transfers preserve the runtime's float32 wire representation. -/
instance (priority := 1000) : TensorTransfer Float32 where
  toFloatTensor := TensorTransfer.toFloatTensorWith Float32.toFloat
  ofFloatTensor := TensorTransfer.ofFloatTensorWith Float.toFloat32
  toFloat := fun x => pure x.toFloat

/--
Host-side conversion for TorchLean's executable IEEE-754 binary32 scalar.

`IEEE32Exec` is a Lean-defined bit-level scalar semantics, not a native float32 wire format. Scalar
and tensor readback to `Float` remain available for reports and checkpoints, but bulk native
transfer is unsupported.
-/
instance (priority := 1000) : TensorTransfer TorchLean.Floats.IEEE754.IEEE32Exec where
  toFloatTensor := fun {_s} _ =>
    throw <| IO.userError
      "torch: IEEE32Exec has host-side scalar conversion only; use Float for native tensor transfer"
  ofFloatTensor := fun {_s} _ =>
    throw <| IO.userError
      "torch: IEEE32Exec has host-side scalar conversion only; use Float for native tensor transfer"
  toFloat := fun x => pure (TorchLean.Floats.IEEE754.IEEE32Exec.toFloat x)
  readFloatTensor := TensorTransfer.toFloatTensorWith
    TorchLean.Floats.IEEE754.IEEE32Exec.toFloat

/--
CPU-preserving fallback for scalar types without a native tensor representation.

Add a higher-priority `TensorTransfer α` instance when a scalar type has an honest host `Float`
encoding. CPU execution does not invoke these failing transfer operations.
-/
instance (priority := 10) (α : Type) [TorchLean.Storage α] : TensorTransfer α where
  toFloatTensor := fun {_s} _ =>
    throw <| IO.userError <|
      "torch: this scalar type has no TensorTransfer tensor encoding; " ++
        "use Float for native execution or run this scalar on CPU"
  ofFloatTensor := fun {_s} _ =>
    throw <| IO.userError <|
      "torch: this scalar type has no TensorTransfer tensor decoding; " ++
        "use Float for native execution or run this scalar on CPU"
  toFloat := fun _ =>
    throw <| IO.userError <|
      "torch: this scalar type has no TensorTransfer scalar conversion; " ++
        "use Float for native execution or run this scalar on CPU"

end Torch
end Autograd
end Runtime
