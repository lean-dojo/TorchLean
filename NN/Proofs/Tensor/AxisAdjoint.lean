/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Folds
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Inner products under axis permutations

Moving an axis changes the order of a tensor's coordinates. It leaves the sum of coordinate
products unchanged. These identities apply at every rank, including shapes with empty axes,
and let a rowwise adjoint calculation pass through the permutations used by softmax.
-/

@[expose] public section

namespace Proofs

open Spec TorchLean
open TorchLean.Tensor
open scoped BigOperators

noncomputable section

/-- A tensor inner product is the sum of the inner products of its leading slices. -/
theorem dot_eq_sum_unstack {n : Nat} {s : Shape} (x y : Tensor ℝ (.dim n s)) :
    Spec.dot x y = ∑ i : Fin n, Spec.dot (x.unstack i) (y.unstack i) := by
  simp only [Spec.dot_eq_tensorAlgebra_dot, TensorAlgebra.dot,
    List.finRange_foldl_add_eq_finset_sum]

/-- Transporting both tensors through the same shape equality preserves their inner product. -/
theorem dot_cast_shape {s t : Shape} (h : s = t) (x y : Tensor ℝ s) :
    Spec.dot (h ▸ x) (h ▸ y) = Spec.dot x y := by
  cases h
  rfl

/-- Exchanging adjacent axes preserves the inner product at any depth. -/
theorem dot_swapAdjacentAxes {s : Shape} (x y : Tensor ℝ s) (depth : Nat) :
    Spec.dot (swapAdjacentAxes x depth) (swapAdjacentAxes y depth) = Spec.dot x y := by
  induction depth generalizing s with
  | zero =>
      cases s with
      | scalar => rfl
      | dim n s =>
          cases s with
          | scalar => rfl
          | dim m t =>
              simp only [swapAdjacentAxes, dot_eq_sum_unstack, Tensor.unstack_dim]
              exact Finset.sum_comm
  | succ depth ih =>
      cases s with
      | scalar => rfl
      | dim n s =>
          simp only [swapAdjacentAxes, dot_eq_sum_unstack, Tensor.unstack_dim, ih]

/-- A sequence of adjacent swaps preserves the inner product, without a rank restriction. -/
theorem dot_permuteByAdjacentSwaps {s : Shape} (x y : Tensor ℝ s) (swaps : List Nat) :
    Spec.dot (permuteByAdjacentSwaps x swaps) (permuteByAdjacentSwaps y swaps) =
      Spec.dot x y := by
  induction swaps generalizing s with
  | nil => rfl
  | cons depth swaps ih =>
      exact (ih (swapAdjacentAxes x depth) (swapAdjacentAxes y depth)).trans
        (dot_swapAdjacentAxes x y depth)

end

end Proofs
