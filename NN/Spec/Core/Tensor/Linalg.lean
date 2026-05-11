/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Core

/-!
# Linear algebra primitives (spec layer)

This file defines the basic matrix/vector operations used across the model zoo:

- `matMulSpec` (matrix × matrix)
- `matVecMulSpec` (matrix × vector)
- `vecMatMulSpec` (vector × matrix)
- `outerProductSpec`

All operations are *shape-indexed* in their types, so misuse is caught by elaboration.

These are intentionally simple, “obvious” definitions (folding over `List.finRange`) so that:

- they are easy to reason about in proofs, and
- they can be instantiated over many scalar backends (`Float`, `ℚ`, `IEEE32Exec`, `ℝ`, …).

PyTorch analogies:

- `matMulSpec A B` is `A @ B`
- `matVecMulSpec A v` is `A @ v`
- `vecMatMulSpec v A` is `v @ A`
- `outerProductSpec a b` is like `a.unsqueeze(1) * b.unsqueeze(0)` (result is `(m,n)`).
-/

@[expose] public section


namespace Spec

/--
Create an identity matrix (n x n).

Notes:
- The `n = 0` case is an empty matrix; it still exists as a well-typed tensor.
- We use `i.val == j.val` rather than `DecidableEq (Fin n)` to keep the definition directly
  executable across backends.
-/
def identityTensorSpec {α : Type} [Zero α] [One α] : ∀ (n : Nat), Tensor α (.dim n (.dim n
  .scalar))
  | 0 => Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0))
  -- Empty identity tensor for 0 dimensions
  | Nat.succ _ =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        Tensor.scalar (if i.val == j.val then 1 else 0)))

/--
Matrix multiplication (m x n) @ (n x p) = (m x p).

This is the simplest definitional version: sum over the shared `n` dimension.
For performance-oriented runtime code, use the runtime layer; this spec is about clarity and proofs.
-/
def matMulSpec {α : Type} [Add α] [Mul α] [Zero α] {m n p : Nat} (A : Tensor α (.dim m (.dim n
  .scalar)))
    (B : Tensor α (.dim n (.dim p .scalar))) : Tensor α (.dim m (.dim p .scalar)) :=
  match A, B with
  | Tensor.dim rowsA, Tensor.dim rowsB =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        Tensor.scalar (
          (List.finRange n).foldl (fun sum k =>
            match rowsA i, rowsB k with
            | Tensor.dim colsA, Tensor.dim colsB =>
              match colsA k, colsB j with
              | Tensor.scalar a, Tensor.scalar b => sum + a * b) 0)))

/-- Matrix-vector multiplication (m x n) @ (n) = (m). -/
def matVecMulSpec {α : Type} [Add α] [Mul α] [Zero α] {m n : Nat} (A : Tensor α (.dim m (.dim n
  .scalar)))
    (v : Tensor α (.dim n .scalar)) : Tensor α (.dim m .scalar) :=
  match A, v with
  | Tensor.dim rowsA, Tensor.dim valuesV =>
    Tensor.dim fun i =>
      match rowsA i with
      | Tensor.dim colsA =>
        (List.finRange n).foldl
          (fun (acc : Tensor α .scalar) (k : Fin n) =>
            match acc, colsA k, valuesV k with
            | Tensor.scalar s, Tensor.scalar ak, Tensor.scalar vk =>
              Tensor.scalar (s + ak * vk))
          (Tensor.scalar 0)

/-- Vector-matrix multiplication (m) @ (m x n) = (n). -/
def vecMatMulSpec {α : Type} [Add α] [Mul α] [Zero α] {m n : Nat} (v : Tensor α (.dim m .scalar))
    (A : Tensor α (.dim m (.dim n .scalar))) : Tensor α (.dim n .scalar) :=
  match v, A with
  | Tensor.dim valuesV, Tensor.dim rowsA =>
    Tensor.dim (fun j =>
      Tensor.scalar (
        (List.finRange m).foldl (fun sum i =>
          match valuesV i, rowsA i with
          | Tensor.scalar vi, Tensor.dim colsA =>
            match colsA j with
            | Tensor.scalar aij => sum + vi * aij) 0))

/-- Outer product (m) otimes (n) = (m x n). -/
def outerProductSpec {α : Type} [Mul α] {m n : Nat} (a : Tensor α (.dim m .scalar)) (b : Tensor α
  (.dim n .scalar)) :
    Tensor α (.dim m (.dim n .scalar)) :=
  match a, b with
  | Tensor.dim f1, Tensor.dim f2 =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        match f1 i, f2 j with
        | Tensor.scalar x, Tensor.scalar y => Tensor.scalar (x * y)))

end Spec
