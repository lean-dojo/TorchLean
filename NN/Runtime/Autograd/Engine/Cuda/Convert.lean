/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA helpers: row-major conversions between spec tensors and `FloatArray`.

Motivation:
- CUDA buffers (`Runtime.Autograd.Cuda.Buffer`) are contiguous float32 arrays.
- Many CUDA kernels interpret buffers in row-major order for a given `Spec.Shape`.
- `Spec.Tensor` is a functional/nested representation that does not commit to a layout.

This module fixes a single layout convention for CUDA interop:
outermost-first recursion, where the last axis varies fastest (row-major / C-order).

We provide conversions for:
- `Spec.Tensor Float s` ↔ `FloatArray`
- `Spec.Tensor Bool s` ↔ `FloatArray` (Bool masks encoded as `1.0` for `true`, `0.0` for `false`)
-/

module

public import NN.Spec.Core.Tensor.Core

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec

namespace Convert

/-!
### Flatten (`Spec.Tensor → FloatArray`)

Row-major order with outermost axis first and innermost axis last.
-/

/--
Worker for `flattenFloat`.

The accumulator lets us append in one left-to-right row-major traversal instead of repeatedly
concatenating intermediate arrays.
-/
def flattenFloatAux : {s : Shape} → Tensor Float s → FloatArray → FloatArray
  | .scalar, .scalar x, out => out.push x
  | .dim n s, .dim f, out =>
      (List.finRange n).foldl (fun acc i => flattenFloatAux (s := s) (f i) acc) out

/-- Flatten a `Spec.Tensor Float s` into a row-major `FloatArray` (CUDA-compatible). -/
def flattenFloat {s : Shape} (t : Tensor Float s) : FloatArray :=
  flattenFloatAux (s := s) t (FloatArray.emptyWithCapacity (Shape.size s))

/-- Worker for `flattenBoolMask`; true is encoded as `1.0`, false as `0.0`. -/
def flattenBoolMaskAux : {s : Shape} → Tensor Bool s → FloatArray → FloatArray
  | .scalar, .scalar b, out => out.push (if b then 1.0 else 0.0)
  | .dim n s, .dim f, out =>
      (List.finRange n).foldl (fun acc i => flattenBoolMaskAux (s := s) (f i) acc) out

/-- Flatten a `Spec.Tensor Bool s` mask to `FloatArray` as `0.0/1.0` values in row-major order. -/
def flattenBoolMask {s : Shape} (mask : Tensor Bool s) : FloatArray :=
  flattenBoolMaskAux (s := s) mask (FloatArray.emptyWithCapacity (Shape.size s))

/-!
### Unflatten (`FloatArray → Spec.Tensor`)

These functions assume row-major order. The `?` variants check the expected length and return
`none` on mismatch.
-/

/--
Worker for `unflattenFloatUnsafe`.

`offset` is the row-major starting position for the current tensor subtree. The caller is
responsible for ensuring the array contains at least `offset + Shape.size s` elements.
-/
def unflattenFloatAux : {s : Shape} → FloatArray → (offset : Nat) → Tensor Float s
  | .scalar, a, offset =>
      Tensor.scalar (a.get! offset)
  | .dim n s, a, offset =>
      Tensor.dim (fun i : Fin n =>
        unflattenFloatAux (s := s) a (offset + i.val * Shape.size s))

/-- Unflatten a row-major `FloatArray` into a `Spec.Tensor Float s` (unsafe: assumes correct size). -/
def unflattenFloatUnsafe {s : Shape} (a : FloatArray) : Tensor Float s :=
  unflattenFloatAux (s := s) a 0

/-- Unflatten a row-major `FloatArray` into a `Spec.Tensor Float s` when `a.size` matches. -/
def unflattenFloat? {s : Shape} (a : FloatArray) : Option (Tensor Float s) :=
  if a.size = Shape.size s then
    some (unflattenFloatUnsafe (s := s) a)
  else
    none

/--
Worker for `unflattenBoolMaskUnsafe`.

The CUDA mask convention is deliberately simple: zero means `false`, any nonzero value means `true`.
-/
def unflattenBoolMaskAux : {s : Shape} → FloatArray → (offset : Nat) → Tensor Bool s
  | .scalar, a, offset =>
      Tensor.scalar (a.get! offset != 0.0)
  | .dim n s, a, offset =>
      Tensor.dim (fun i : Fin n =>
        unflattenBoolMaskAux (s := s) a (offset + i.val * Shape.size s))

/-- Unflatten a `FloatArray` into a `Spec.Tensor Bool s` mask (unsafe: nonzero = true). -/
def unflattenBoolMaskUnsafe {s : Shape} (a : FloatArray) : Tensor Bool s :=
  unflattenBoolMaskAux (s := s) a 0

/-- Unflatten a `FloatArray` into a `Spec.Tensor Bool s` mask when `a.size` matches. -/
def unflattenBoolMask? {s : Shape} (a : FloatArray) : Option (Tensor Bool s) :=
  if a.size = Shape.size s then
    some (unflattenBoolMaskUnsafe (s := s) a)
  else
    none

end Convert

end Cuda
end Autograd
end Runtime
