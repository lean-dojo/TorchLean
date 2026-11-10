/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Rep.Fiber.Differential
public import TorchLean.Tensor.Euclidean
public import TorchLean.Tensor.Reduce.Backward
public import TorchLean.Tensor.Layers.Normalization
public import Mathlib.Analysis.InnerProductSpace.Adjoint

/-!
# The adjoint of broadcasting

`Tensor.broadcast x t dims h` reads `x` at the coordinate `broadcastSource dims h d` of `s` for
each coordinate `d` of `t`, so it is `Rep.pull` along `broadcastSource dims h`
(`broadcast_eq_pull`). Over a commutative additive monoid, `Tensor.reduceBroadcast dims h` sums its
argument over each fiber of that map (`reduceBroadcast_apply`), so it is
`Rep.push` along it (`reduceBroadcast_eq_push`). Hence `reduceBroadcast` is the adjoint of
`broadcast` for the finite dot product `Rep.dot` over a non-unital non-associative
semiring (`dot_reduceBroadcast`), and for the Euclidean inner product over the
reals (`inner_broadcast`). These hold at every placement and shape, including scalar
shapes, zero extents and empty fibers.

Over the reals, the sum over every axis but `dim` is the adjoint of broadcasting along `dim`
(`inner_broadcast_axis`), so it is `reduceBroadcast` at that placement
(`reduce_otherAxes_eq_reduceBroadcast`). `broadcastCLM` is broadcasting as a
continuous linear map on real tensors, with `reduceBroadcast` as its adjoint.
-/

@[expose] public section

open scoped BigOperators RealInnerProductSpace
open TorchLean.Shape (rank_cons)

namespace TorchLean.Tensor

/-- The coordinate of `s` that `broadcast _ t dims h` reads at the coordinate `d` of `t`. -/
def broadcastSource {s t : Shape} (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) (d : t.Coord) : s.Coord :=
  Shape.Coord.ofFun fun i =>
    if hi : s.extent i = 1 then ⟨0, by omega⟩
    else Fin.cast ((h.extent_eq i).resolve_left hi).symm (Shape.Coord.get d (dims i))

/-- Broadcasting is the pullback `Rep.pull` along `broadcastSource dims h`. -/
theorem broadcast_eq_pull {α : Type} [Storage α] {s : Shape} (x : Tensor α s) (t : Shape)
    (dims : Fin s.rank → Fin t.rank) (h : Shape.BroadcastDims s t dims) :
    broadcast x t dims h = Rep.pull (broadcastSource dims h) x :=
  rfl

private theorem index_broadcastSource {s t : Shape} (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) (d : t.Coord) {k : Nat} (hk : k < s.rank) :
    Shape.Coord.index (broadcastSource dims h d) k =
      if s.extent ⟨k, hk⟩ = 1 then 0 else Shape.Coord.index d (dims ⟨k, hk⟩).val := by
  rw [broadcastSource, Shape.Coord.index_ofFun, dite_eq_left hk]
  split_ifs with hi
  · rfl
  · rw [Fin.val_cast, Shape.Coord.val_get]

private theorem index_cons_of_ne_zero {n : Nat} {t : Shape} (i : Fin n) (d : t.Coord) {v : Nat}
    (hv : v ≠ 0) : Shape.Coord.index (s := .dim n t) (i, d) v = Shape.Coord.index d (v - 1) := by
  cases v with
  | zero => exact absurd rfl hv
  | succ v => rfl

/-- With the outer axis of `n :: t` unplaced, `broadcastSource` ignores the outer index. -/
private theorem broadcastSource_cons_of_avoids {s : Shape} {n : Nat} {t : Shape}
    (dims : Fin s.rank → Fin (Shape.dim n t).rank) (h : Shape.BroadcastDims s (.dim n t) dims)
    (avoids : ∀ i, (dims i).val ≠ 0) (i : Fin n) (d : t.Coord) :
    broadcastSource dims h (i, d) =
      broadcastSource (Shape.BroadcastDims.shiftDims dims avoids) (h.shift avoids) d := by
  apply Shape.Coord.ext_index
  intro k hk
  rw [index_broadcastSource _ _ _ hk, index_broadcastSource _ _ _ hk,
    index_cons_of_ne_zero i d (avoids ⟨k, hk⟩)]
  rfl

/-- With the outer axis of `m :: s` placed at the outer axis of `n :: t`, `broadcastSource` maps
the outer index to itself, or to zero at extent one, and the inner coordinate by the inner
placement. -/
private theorem broadcastSource_cons_eq_iff {m n : Nat} {s t : Shape}
    (dims : Fin (Shape.dim m s).rank → Fin (Shape.dim n t).rank)
    (h : Shape.BroadcastDims (.dim m s) (.dim n t) dims)
    (placed : (dims ⟨0, Nat.succ_pos _⟩).val = 0) (i : Fin n) (d : t.Coord) (j : Fin m)
    (c : s.Coord) :
    broadcastSource dims h (i, d) = (j, c) ↔ (if m = 1 then 0 else i.val) = j.val ∧
      broadcastSource (Shape.BroadcastDims.shiftDims (fun k => dims k.succ) h.avoids_succ)
        h.tail d = c := by
  have head : (broadcastSource dims h (i, d)).1.val = if m = 1 then 0 else i.val := by
    have e := index_broadcastSource dims h (i, d) (k := 0) (Nat.succ_pos _)
    change _ = if m = 1 then 0 else Shape.Coord.index (s := .dim n t) (i, d)
      (dims ⟨0, Nat.succ_pos _⟩).val at e
    rw [Shape.Coord.index_zero] at e
    rw [e, placed]
    rfl
  have inner : (broadcastSource dims h (i, d)).2 =
      broadcastSource (Shape.BroadcastDims.shiftDims (fun k => dims k.succ) h.avoids_succ)
        h.tail d := by
    apply Shape.Coord.ext_index
    intro k hk
    have hk1 : k + 1 < (Shape.dim m s).rank := by simp only [Shape.rank_cons]; omega
    have e := index_broadcastSource dims h (i, d) hk1
    rw [Shape.Coord.index_succ] at e
    have hv : (dims ⟨k + 1, hk1⟩).val ≠ 0 := h.avoids_succ ⟨k, hk⟩
    rw [e, index_broadcastSource _ _ _ hk, index_cons_of_ne_zero i d hv]
    rfl
  constructor
  · intro e
    rw [e] at head inner
    exact ⟨head.symm, inner.symm⟩
  · intro ⟨e1, e2⟩
    exact Prod.ext (Fin.ext (head.trans e1)) (inner.trans e2)

/-- An entry of a sum over the outer axis, cast to the shape without it, is the finite sum of
the entries along that axis. -/
private theorem reduce_sum_zero_apply_eq_sum {α : Type} [Storage α] [AddCommMonoid α]
    {n : Nat} {t : Shape} (g : Tensor α (.dim n t)) (d : t.Coord) :
    castShape (target := t) (reduce .sum g [⟨0, Nat.succ_pos _⟩] false)
        (Shape.reduce_singleton _ _) d =
      ∑ i : Fin n, g (i, d) := by
  have e : castShape (target := t) (reduce .sum g [⟨0, Nat.succ_pos _⟩] false)
        (Shape.reduce_singleton _ _) =
      Rep.ofFn fun c : t.Coord => sum (Tensor.ofFn fun i : Fin n => g (i, c)) :=
    castShape_reduce_sum_zero g _ _
  rw [e, Rep.get_ofFn, sum_eq_coord_sum]
  show ∑ c : Fin n × PUnit, _ = _
  rw [Fintype.sum_prod_type]
  simp only [ofFn_apply, Finset.univ_unique, Finset.sum_singleton]

/-- Under a condition on the inner coordinate only, the sum over `n :: t` of the selected entries
of `g` is the sum of the selected entries of its outer-axis reduction. -/
private theorem sum_ite_cons {α : Type} [Storage α] [AddCommMonoid α] {β : Type} [DecidableEq β]
    {n : Nat} {t : Shape} (g : Tensor α (.dim n t)) (r : t.Coord → β) (c : β) :
    (∑ d : Shape.Coord (.dim n t), if r d.2 = c then g d else 0) =
      ∑ d : t.Coord, if r d = c then
        castShape (target := t) (reduce .sum g [⟨0, Nat.succ_pos _⟩] false)
          (Shape.reduce_singleton _ _) d
      else 0 := by
  show (∑ d : Fin n × t.Coord, _) = _
  rw [Fintype.sum_prod_type, Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro d _
  rw [reduce_sum_zero_apply_eq_sum]
  split_ifs <;> simp

private theorem reduceBroadcast_apply' {α : Type} [Storage α] [AddCommMonoid α]
    {s t : Shape} (dims : Fin s.rank → Fin t.rank) (h : Shape.BroadcastDims s t dims)
    (g : Tensor α t) (c : s.Coord) :
    reduceBroadcast dims h g c =
      ∑ d : t.Coord, if broadcastSource dims h d = c then g d else 0 := by
  induction dims, h, g using reduceBroadcast.induct with
  | case1 g dims h =>
      rw [reduceBroadcast]
      show g c = ∑ d : PUnit, _
      simp only [Finset.univ_unique, Finset.sum_singleton, ite_true]
  | case2 _ _ _ dims => exact (dims ⟨0, Nat.succ_pos _⟩).elim0
  | case3 n t g dims h ih =>
      rw [reduceBroadcast, ih, ← sum_ite_cons g _ c]
  | case4 s n t g dims h placed ih =>
      obtain ⟨j, c⟩ := c
      rw [reduceBroadcast, dite_eq_left placed, dite_eq_left rfl]
      simp only [Tensor.dim_apply]
      rw [ih j c, Fin.cast_eq_self]
      symm
      show (∑ d : Fin n × t.Coord, _) = _
      rw [Fintype.sum_prod_type, Finset.sum_eq_single j]
      · apply Finset.sum_congr rfl
        intro d _
        have hj : (if n = 1 then 0 else j.val) = j.val := by
          have := j.isLt
          split_ifs <;> omega
        simp only [broadcastSource_cons_eq_iff dims h placed, hj, true_and,
          Tensor.unstack_apply]
      · intro i _ hij
        apply Finset.sum_eq_zero
        intro d _
        rw [ite_eq_right_iff, broadcastSource_cons_eq_iff dims h placed]
        intro ⟨e, _⟩
        have := i.isLt
        have := j.isLt
        exact absurd (Fin.ext (by split_ifs at e <;> omega)) hij
      · intro hj
        exact absurd (Finset.mem_univ _) hj
  | case5 m s n t hExtent g dims h placed ih =>
      obtain ⟨j, c⟩ := c
      have hm : m = 1 := by
        rcases h.extent_eq ⟨0, Nat.succ_pos _⟩ with e | e
        · exact e
        · rw [show dims ⟨0, Nat.succ_pos _⟩ = ⟨0, Nat.succ_pos _⟩ from Fin.ext placed] at e
          exact (hExtent e).elim
      rw [reduceBroadcast, dite_eq_left placed, dite_eq_right hExtent]
      simp only [Tensor.dim_apply]
      rw [ih c, ← sum_ite_cons g _ c]
      apply Finset.sum_congr rfl
      intro ⟨i, d⟩ _
      have hj : (if m = 1 then 0 else i.val) = j.val := by
        have := j.isLt
        rw [ite_eq_left hm]
        omega
      simp only [broadcastSource_cons_eq_iff dims h placed, hj, true_and]
  | case6 m s n t g dims h unplaced ih =>
      rw [reduceBroadcast, dite_eq_right unplaced, ih, ← sum_ite_cons g _ c]
      apply Finset.sum_congr rfl
      intro d _
      rw [show d = (d.1, d.2) from rfl,
        broadcastSource_cons_of_avoids dims h (h.avoids_of_head unplaced)]

/-- Over a commutative additive monoid, an entry of `reduceBroadcast dims h g` is the sum of `g`
over the coordinates of `t` at which `broadcast _ t dims h` reads that entry. -/
theorem reduceBroadcast_apply {α : Type} [Storage α] [AddCommMonoid α]
    {s t : Shape} (dims : Fin s.rank → Fin t.rank) (h : Shape.BroadcastDims s t dims)
    (g : Tensor α t) (c : s.Coord) :
    reduceBroadcast dims h g c = ∑ d : {d : t.Coord // broadcastSource dims h d = c}, g d.1 := by
  rw [reduceBroadcast_apply', ← Finset.sum_filter,
    Finset.sum_subtype (p := fun d => broadcastSource dims h d = c) _ (fun d => by simp)]

/-- Over a commutative additive monoid, `reduceBroadcast dims h` is the fiber sum
`Rep.push` along `broadcastSource dims h`. -/
theorem reduceBroadcast_eq_push {α : Type} [Storage α] [AddCommMonoid α] {s t : Shape}
    (dims : Fin s.rank → Fin t.rank) (h : Shape.BroadcastDims s t dims) (g : Tensor α t) :
    reduceBroadcast dims h g = Rep.push (broadcastSource dims h) g := by
  apply Rep.ext
  intro c
  rw [reduceBroadcast_apply, Rep.push_apply]

/-- Over a `NonUnitalNonAssocSemiring`, `reduceBroadcast dims h` is the adjoint of
`broadcast _ t dims h` for the finite dot product `Rep.dot`. -/
theorem dot_reduceBroadcast {α : Type} [Storage α]
    [NonUnitalNonAssocSemiring α] {s t : Shape} (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) (g : Tensor α t) (x : Tensor α s) :
    Rep.dot (reduceBroadcast dims h g) x = Rep.dot g (broadcast x t dims h) := by
  rw [reduceBroadcast_eq_push, broadcast_eq_pull, Rep.dot_push_eq_dot_pull]

/-- Over the reals, `reduceBroadcast dims h` is the adjoint of `broadcast _ t dims h` for the
Euclidean inner product. -/
theorem inner_broadcast {s t : Shape} (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) (x : Tensor ℝ s) (g : Tensor ℝ t) :
    ⟪broadcast x t dims h, g⟫ = ⟪x, reduceBroadcast dims h g⟫ := by
  rw [real_inner_comm g, real_inner_comm (reduceBroadcast dims h g), inner_eq_sum, inner_eq_sum]
  exact (dot_reduceBroadcast dims h g x).symm


/-- Over the reals, broadcasting along `dim` and summing over every other axis are adjoint for the
Euclidean inner product. -/
theorem inner_broadcast_axis {s : Shape} (dim : Fin s.rank)
    (v : Tensor ℝ [s.extent dim]) (w : Tensor ℝ s) :
    ⟪broadcast v s _ (s.broadcastDims_axis dim), w⟫ =
      ⟪v, castShape (Tensor.reduce .sum w (s.otherAxes [dim]) false)
        (Shape.eraseAxes_otherAxes_singleton s dim)⟫ := by
  rw [inner_eq_sum, inner_eq_sum]
  simp only [reduce_sum_otherAxes_apply, broadcast_axis_apply, getScalar_eq_apply,
    Finset.mul_sum, mul_ite, mul_zero]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro c _
  show _ = ∑ o : Fin (s.extent dim) × PUnit, _
  rw [Fintype.sum_prod_type]
  simp only [Finset.univ_unique, Finset.sum_singleton]
  rw [Finset.sum_ite_eq, ite_eq_left (Finset.mem_univ _)]

/-- Over the reals, the sum over every axis but `dim` is the pullback of broadcasting along
`dim`. -/
theorem reduce_otherAxes_eq_reduceBroadcast {s : Shape} (dim : Fin s.rank)
    (w : Tensor ℝ s) :
    castShape (Tensor.reduce .sum w (s.otherAxes [dim]) false)
        (Shape.eraseAxes_otherAxes_singleton s dim) =
      reduceBroadcast (fun _ => dim) (s.broadcastDims_axis dim) w := by
  apply ext_inner_left ℝ
  intro v
  rw [← inner_broadcast_axis, inner_broadcast]

/-- Broadcasting to `t` along `dims`, as a continuous linear map on real tensors. -/
noncomputable def broadcastCLM {s : Shape} (t : Shape) (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) : Tensor ℝ s →L[ℝ] Tensor ℝ t :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => broadcast x t dims h
      map_add' := fun x y => Rep.ext fun c => by
        simp only [broadcast_apply, Tensor.add_apply]
      map_smul' := fun a x => Rep.ext fun c => by
        simp only [broadcast_apply, Rep.smul_apply, RingHom.id_apply] }

/-- `broadcastCLM t dims h` applied to `x` is `broadcast x t dims h`. -/
@[simp] theorem broadcastCLM_apply {s : Shape} (t : Shape) (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) (x : Tensor ℝ s) :
    broadcastCLM t dims h x = broadcast x t dims h :=
  rfl

/-- The adjoint of the broadcasting map is `reduceBroadcast`. -/
theorem broadcastCLM_adjoint_apply {s : Shape} (t : Shape) (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) (g : Tensor ℝ t) :
    (broadcastCLM t dims h).adjoint g = reduceBroadcast dims h g := by
  apply ext_inner_left ℝ
  intro x
  rw [ContinuousLinearMap.adjoint_inner_right, broadcastCLM_apply,
    inner_broadcast]

end TorchLean.Tensor
