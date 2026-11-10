/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Fin
public import TorchLean.Tensor.Axes

/-!
# Row-major strides

`Shape.stride s d` is the number of entries between consecutive indices along axis `d` of a
row-major tensor of shape `s`. A flat index is the sum over the axes of each index times its stride
(`Shape.Coord.linearize_val_eq_sum`). `Shape.weightedIndex` computes such a sum, with any weights,
from the flat index itself (`Shape.weightedIndex_ofFn`), so a compiled kernel can move entries
without building coordinates.
-/

@[expose] public section

open TorchLean.Shape (rank)

namespace TorchLean.Shape

/-- The row-major stride of axis `d` of `s`, the product of the extents after `d`, as PyTorch's
`x.stride(d)` for a contiguous `x`. At a zero extent after `d` ours is `0` and torch 2.8.0's isn't:
`torch.empty(2, 0, 3).stride()` is `(3, 3, 1)` and ours `(0, 3, 1)`, on a tensor with no entries. -/
def stride (s : Shape) (d : Fin s.rank) : Nat :=
  Shape.size (s.drop (d.val + 1))

/-- The outermost axis of `n :: s` has stride `size s`. -/
@[simp] theorem stride_cons_zero (n : Nat) (s : Shape) (h : 0 < Shape.rank (n :: s)) :
    stride (n :: s) ⟨0, h⟩ = Shape.size s :=
  rfl

/-- An inner axis of `n :: s` has its stride in `s`. -/
@[simp] theorem stride_cons_succ (n : Nat) (s : Shape) (d : Fin s.rank) :
    stride (n :: s) d.succ = stride s d :=
  rfl

/-- Peel a flat index from the outermost axis: an axis with tail size `size` and weight `weight`
contributes `i / size * weight`, and the rest reads `i % size`. -/
def weightedIndex : List (Nat × Nat) → Nat → Nat
  | [], _ => 0
  | (size, weight) :: rest, i => i / size * weight + weightedIndex rest (i % size)

namespace Coord

/-- A row-major flat index is the sum over the axes of the index along each axis times its
stride. -/
theorem linearize_val_eq_sum {s : Shape} (c : s.Coord) :
    (Coord.linearize c).val = ∑ d, (get c d).val * s.stride d := by
  induction s with
  | scalar =>
      have h := (Coord.linearize (s := []) c).isLt
      simp only [Shape.size_nil, Nat.lt_one_iff] at h
      rw [h]
      exact (Fin.sum_univ_zero _).symm
  | dim n s ih =>
      obtain ⟨i, c⟩ := c
      rw [Coord.linearize_cons_val, ih c]
      change _ = ∑ d : Fin (s.rank + 1), _
      rw [Fin.sum_univ_succ]
      simp only [get_cons_succ, stride_cons_succ]
      change _ = i.val * Shape.size s + _
      rw [Nat.mul_comm, Nat.add_comm]

end Coord

/-- `weightedIndex` with the strides of `s` and weights `w` sends a flat index to the sum over the
axes of its index along each axis times the weight of that axis. -/
theorem weightedIndex_ofFn {s : Shape} (w : Fin s.rank → Nat)
    (i : Fin (Shape.size s)) :
    weightedIndex (List.ofFn fun d => (s.stride d, w d)) i.val =
      ∑ d, (Coord.get (Coord.unlinearize i) d).val * w d := by
  rw [Coord.unlinearizeFast_eq]
  induction s with
  | scalar => exact (Fin.sum_univ_zero _).symm
  | dim n s ih =>
      erw [List.ofFn_succ (n := s.rank), Fin.sum_univ_succ (n := s.rank)]
      simp only [stride_cons_succ, weightedIndex]
      have hs : 0 < Shape.size s :=
        Nat.pos_of_ne_zero fun h => by
          have := i.isLt
          simp [h] at this
      erw [ih (fun d => w d.succ) ⟨i.val % Shape.size s, Nat.mod_lt _ hs⟩]
      rfl
