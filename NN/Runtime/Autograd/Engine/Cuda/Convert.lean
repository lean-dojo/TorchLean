/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA helpers: row-major conversions between spec tensors and `FloatArray`.

Motivation:
- CUDA buffers (`Runtime.Autograd.Cuda.Buffer`) are contiguous float32 arrays.
- Many CUDA kernels interpret buffers in row-major order for a given `Spec.Shape`.
- `TorchLean.Tensor` is a functional/nested representation that does not commit to a layout.

This module fixes a single layout convention for CUDA interop:
outermost-first recursion, where the last axis varies fastest (row-major / C-order).

We provide conversions for:
- `TorchLean.Tensor Float s` ↔ `FloatArray`
- `TorchLean.Tensor Bool s` ↔ `FloatArray` (Bool masks encoded as `1.0` for `true`, `0.0` for
  `false`)
-/

module

public import NN.Spec.Core.Tensor.Core

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec TorchLean

namespace Convert

namespace Internal

/-!
### Flatten (`TorchLean.Tensor → FloatArray`)

Row-major order with outermost axis first and innermost axis last.
-/

/-- Append the elements of a tensor to an accumulator in row-major order. -/
def flattenFloat : {s : Shape} → Tensor Float s → FloatArray → FloatArray
  | _, tensor, out =>
      TorchLean.Tensor.Internal.Rep.foldl (fun acc value => acc.push value) out tensor

end Internal

/-- Flatten a `TorchLean.Tensor Float s` into a row-major `FloatArray` (CUDA-compatible). -/
def flattenFloat {s : Shape} (t : Tensor Float s) : FloatArray :=
  Internal.flattenFloat (s := s) t (FloatArray.emptyWithCapacity (Spec.Shape.size s))

namespace Internal

/-- Accumulating worker for `flattenBoolMask`: `true` becomes `1.0`, `false` becomes `0.0`, since
the CUDA side has no boolean buffers and reads masks as floats. -/
def flattenBoolMask : {s : Shape} → Tensor Bool s → FloatArray → FloatArray
  | _, tensor, out =>
      TorchLean.Tensor.Internal.Rep.foldl
        (fun acc value => acc.push (if value then 1.0 else 0.0)) out tensor

end Internal

/-- Flatten a `TorchLean.Tensor Bool s` mask to `FloatArray` as `0.0/1.0` values in row-major
order. -/
def flattenBoolMask {s : Shape} (mask : Tensor Bool s) : FloatArray :=
  Internal.flattenBoolMask (s := s) mask (FloatArray.emptyWithCapacity (Spec.Shape.size s))

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

end Cuda
end Autograd
end Runtime
