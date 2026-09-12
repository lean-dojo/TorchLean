/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Core

/-!
# Linear algebra primitives (spec layer)

This file defines the basic matrix/vector operations used across model specifications:

- `matMulSpec` (matrix × matrix)
- `matVecMulSpec` (matrix × vector)
- `vecMatMulSpec` (vector × matrix)
- `outerProductSpec`

All operations are *shape-indexed* in their types, so misuse is caught by elaboration.

These are kept simple, “obvious” definitions (folding over `List.finRange`) so that:

- they are easy to reason about in proofs, and
- they can be instantiated over many scalar backends (`Float`, `ℚ`, `IEEE32Exec`, `ℝ`, …).

PyTorch analogies:

- `matMulSpec A B` is `A @ B`
- `matVecMulSpec A v` is `A @ v`
- `vecMatMulSpec v A` is `v @ A`
- `outerProductSpec a b` is like `a.unsqueeze(1) * b.unsqueeze(0)` (result is `(m,n)`).
-/

@[expose] public section


open TorchLean

namespace Spec

/--
Create an identity matrix (n x n).

Notes:
- The `n = 0` case is an empty matrix; it still exists as a well-typed tensor.
- We use `i.val == j.val` rather than `DecidableEq (Fin n)` to keep the definition directly
  executable across backends.
-/
def identityTensorSpec {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (n : Nat) : Tensor α [n, n] :=
  TorchLean.Tensor.Internal.Rep.ofFn fun coordinate =>
    if coordinate.1.val == coordinate.2.1.val then 1 else 0

/--
Matrix multiplication (m x n) @ (n x p) = (m x p).

This is the simplest definitional version: sum over the shared `n` dimension.
For performance-oriented runtime code, use the runtime layer; this spec is about clarity and proofs.
-/
def matMulSpec {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α] {m n p : Nat}
    (A : Tensor α [m, n]) (B : Tensor α [n, p]) : Tensor α [m, p] :=
  TorchLean.Tensor.Internal.Rep.ofFn fun coordinate =>
    (List.finRange n).foldl
      (fun sum k => sum + get2 A coordinate.1 k * get2 B k coordinate.2.1)
      0

/-- Matrix-vector multiplication (m x n) @ (n) = (m). -/
def matVecMulSpec {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α] {m n : Nat}
    (A : Tensor α [m, n]) (v : Tensor α [n]) : Tensor α [m] :=
  TorchLean.Tensor.Internal.Rep.ofFn fun coordinate =>
    (List.finRange n).foldl
      (fun sum k => sum + get2 A coordinate.1 k * v.getScalar k)
      0

/-- Rank-one tensor by matrix multiplication: `(m) @ (m x n) = (n)`. -/
def vecMatMulSpec {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α] {m n : Nat}
    (v : Tensor α [m]) (A : Tensor α [m, n]) : Tensor α [n] :=
  TorchLean.Tensor.Internal.Rep.ofFn fun coordinate =>
    (List.finRange m).foldl
      (fun sum i => sum + v.getScalar i * get2 A i coordinate.1)
      0

/-- Outer product (m) otimes (n) = (m x n). -/
def outerProductSpec {α : Type} [TorchLean.Storage α] [Mul α]
    {m n : Nat} (a : Tensor α [m]) (b : Tensor α [n]) :
    Tensor α [m, n] :=
  TorchLean.Tensor.Internal.Rep.ofFn fun coordinate =>
    a.getScalar coordinate.1 * b.getScalar coordinate.2.1

/-- A coordinate of an outer product is the product of the corresponding vector entries. -/
@[simp] theorem get2_outerProductSpec {α : Type} [TorchLean.Storage α]
    [Mul α] {m n : Nat}
    (left : Tensor α [m]) (right : Tensor α [n])
    (i : Fin m) (j : Fin n) :
    get2 (outerProductSpec left right) i j =
      Tensor.getScalar left i * Tensor.getScalar right j := by
  simp [outerProductSpec, get2, Tensor.getScalar, get, Tensor.unstack, Tensor.item]

end Spec
