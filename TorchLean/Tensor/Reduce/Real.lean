/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ShapeChange.Flatten -- shake: keep

/-!
# Sums and folds of tensors

Over any commutative additive monoid, `Tensor.sum` is the finite sum of a tensor's coordinates,
and `dot_eq_rep_dot` identifies the executable row-major fold `Tensor.dot` with the pairing
`dot`. Over `ℝ` we split a sum along the outer dimension and give the coordinate formula of
`Tensor.matmul`. These facts concern exact sums and do not permit reassociating a floating-point
reduction. For any step function we also unfold the slice fold `Tensor.foldlEntries.go` one slice
at a time.
-/

@[expose] public section

namespace TorchLean.Tensor

open scoped BigOperators
open Rep (dot)

/-! ## Folds and finite sums -/

/-- Peel one slice from the proof-facing fold starting at `k`, when `k < n`. -/
theorem foldl_go_of_lt {α β : Type} [Storage α]
    (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : k < n) :
    Tensor.foldlEntries.go f n s values k acc =
      Tensor.foldlEntries.go f n s values (k + 1) (Tensor.foldl f acc (values ⟨k, hk⟩)) := by
  rw [Tensor.foldlEntries.go, Tensor.foldlEntries.go,
    List.drop_eq_getElem_cons (by simpa using hk)]
  simp

/-- Starting the proof-facing slice fold at or beyond its length returns the accumulator. -/
theorem foldl_go_of_not_lt {α β : Type} [Storage α]
    (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : ¬ k < n) :
    Tensor.foldlEntries.go f n s values k acc = acc := by
  have hLength : (List.finRange n).length ≤ k := by
    simpa using Nat.le_of_not_gt hk
  rw [Tensor.foldlEntries.go, List.drop_eq_nil_of_le hLength]
  rfl

/-- Entry `(i, j)` of the matrix product `Tensor.matmul A B` is the sum over `k` of
`A i k * B k j`. -/
theorem get2_matmul_eq_sum {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p]) (i : Fin m) (j : Fin p) :
  get2 (Tensor.matmul A B) i j = ∑ k : Fin n, (get2 A i k) * (get2 B k j) := by
  rw [get2_eq_item_unstack]
  simp only [Tensor.item, Tensor.matmul_eq_ofFn,
    Tensor.unstack_apply, Tensor.Rep.get_ofFn]
  change
    (List.finRange n).foldl
        (fun total k => total + get2 A i k * get2 B k j) 0 =
      ∑ k : Fin n, get2 A i k * get2 B k j
  rw [Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]

/-- Over a commutative additive monoid, `Tensor.sum t` is the finite sum of the coordinates
of `t`. -/
theorem sum_eq_coord_sum {α : Type} [Storage α] [AddCommMonoid α] {s : Shape} (t : Tensor α s) :
    Tensor.sum t = ∑ coordinate : s.Coord, t coordinate := by
  classical
  have hFlat :
      t.data.toList.sum =
        ∑ index : Fin (Shape.size s), t.getFlat index := by
    have hSize := Tensor.Rep.data_size t
    calc
      t.data.toList.sum =
          ∑ index : Fin t.data.size, t.data.toList[index.val] := by
        simpa only [Array.length_toList] using
          (Fin.sum_univ_getElem t.data.toList).symm
      _ = ∑ index : Fin (Shape.size s), t.getFlat index := by
        apply Fintype.sum_equiv (finCongr hSize)
        intro index
        rw [Array.getElem_toList]
        simpa using
          (Tensor.Rep.data_getFlat t (Fin.cast hSize index))
  calc
    Tensor.sum t = t.data.toList.sum := by
      rw [Tensor.sum_eq_foldl,
        Tensor.Rep.foldl_eq_data_foldl, ← Array.foldl_toList,
        ← List.sum_eq_foldl]
    _ = ∑ index : Fin (Shape.size s), t.getFlat index := hFlat
    _ = ∑ coordinate : s.Coord, t coordinate := by
      symm
      simpa [Tensor.Rep.get, Shape.Coord.linearize] using
        ((Shape.Coord.equivFin s).sum_comp
          (fun index => t.getFlat index))

/-- The sum of a tensor of shape `.dim n s` is the sum over `i : Fin n` of the sums of its slices
`Tensor.unstack t i`. -/
theorem sum_dim {n : Nat} {s : Shape} (t : Tensor ℝ (.dim n s)) :
  Tensor.sum t = ∑ i : Fin n, Tensor.sum (Tensor.unstack t i) := by
  classical
  rw [sum_eq_coord_sum, Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro i _
  rw [sum_eq_coord_sum]
  apply Finset.sum_congr rfl
  intro coordinate _
  exact (Tensor.unstack_apply t i coordinate).symm

/-- Over a commutative additive monoid, the fold `Tensor.dot` is the pairing `dot`. -/
theorem dot_eq_rep_dot {α : Type} [Storage α] [AddCommMonoid α] [Mul α] {s : Shape}
    (a b : Tensor α s) :
    Tensor.dot a b = Rep.dot a b := by
  rw [Tensor.dot, sum_eq_coord_sum]
  simp only [Rep.dot, mul_def, Tensor.zipWith_apply]

end TorchLean.Tensor
