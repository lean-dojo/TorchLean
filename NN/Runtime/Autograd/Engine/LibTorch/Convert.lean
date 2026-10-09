/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA helpers: row-major conversions between spec tensors and `FloatArray`.

Motivation:
- CUDA buffers (`Runtime.Autograd.LibTorch.Buffer`) retain their native real dtype.
- The LibTorch bridge interprets buffers in row-major order for a given `Spec.Shape`.
- Tensor storage is traversed in logical row-major order.

This module fixes a single layout convention for CUDA interop:
the last axis varies fastest (row-major / C-order).

We provide conversions for:
- `TorchLean.Tensor Float s` ↔ `FloatArray`
- `TorchLean.Tensor Bool s` → `FloatArray` (Bool masks encoded as `1.0` for `true`, `0.0` for
  `false`)
-/

module

public import NN.Spec.Core.Tensor.Core

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

open Spec TorchLean

namespace Convert

/-!
### Flatten (`TorchLean.Tensor → FloatArray`)

Row-major order with outermost axis first and innermost axis last.
-/

/-- Flatten a `TorchLean.Tensor Float s` into a row-major `FloatArray` (CUDA-compatible). -/
def flattenFloat {s : Shape} (t : Tensor Float s) : FloatArray :=
  TorchLean.Tensor.Internal.Rep.foldl (fun acc value => acc.push value)
    (FloatArray.emptyWithCapacity (Spec.Shape.size s)) t

/-- Flatten a `TorchLean.Tensor Bool s` mask to `FloatArray` as `0.0/1.0` values in row-major
order. -/
def flattenBoolMask {s : Shape} (mask : Tensor Bool s) : FloatArray :=
  TorchLean.Tensor.Internal.Rep.foldl (fun acc value => acc.push (if value then 1.0 else 0.0))
    (FloatArray.emptyWithCapacity (Spec.Shape.size s)) mask

/-!
### Unflatten (`FloatArray → TorchLean.Tensor`)

These functions assume row-major order. The public operations check the expected length and return
`none` on mismatch. Runtime code that already owns the buffer-size invariant calls the internal
workers directly.
-/

namespace Internal

/-- Read a tensor out of `a` starting at `offset`, trusting the caller for the bounds. The checked
entry point is `unflattenFloat?`; this one exists so a batched download can slice one array. -/
def unflattenFloat {s : Shape} (a : FloatArray) (offset : Nat) : Tensor Float s :=
  TorchLean.Tensor.Internal.Rep.ofFlatFn fun index => a.get! (offset + index.val)

end Internal

/-- Unflatten a row-major `FloatArray` into a `TorchLean.Tensor Float s` when `a.size` matches. -/
def unflattenFloat? {s : Shape} (a : FloatArray) : Option (Tensor Float s) :=
  if a.size = Spec.Shape.size s then
    some (Internal.unflattenFloat (s := s) a 0)
  else
    none

end Convert

end LibTorch
end Autograd
end Runtime
