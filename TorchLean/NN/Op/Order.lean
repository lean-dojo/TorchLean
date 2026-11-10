/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tactic.OpDeriv
public import TorchLean.NN.FDeriv.Reindex
public import TorchLean.NN.Op.Select
public import TorchLean.Tensor.Op.Cert
public import TorchLean.Tensor.Sort
public import TorchLean.Tensor.Cumulative

/-!
# Derivative certificates of the order-statistic operators

We prove that the stored backward of `Op.sort`, `Op.topk`, `Op.cummax` and `Op.cummin` is the
adjoint of the derivative of the value output of the real forward, on the exact set where this
holds. For each operator a `convention` certificate states the stored backward at every real input
in closed form, so its routing at ties is on record. The index outputs are index ports and carry
no gradient.

Each value output reads its input along its own index output (`sort_fst_apply`, `topk_fst`,
`cummax_fst_apply`). Where the comparisons that pick those indices are strict, they persist near
the input, so the indices are locally constant and the forward agrees near the input with a fixed
coordinate pullback (`hasPullbackAt_of_eventuallyEq_pull`). Its adjoint is the sum over the fibers
of the read, and we show that the stored backward folds are that sum at every input
(`sortBackward_eq_sumFibers`, `topkBackward_eq_sumFibers`, `cummaxBackward_eq_sumFibers`). For
`topk` only the first `k` sorted positions are read, and ties placed after them leave the indices
of those positions locally constant (`hasPullbackAt_topk`).
-/

@[expose] public section

open scoped RealInnerProductSpace
open Filter Topology

namespace TorchLean

namespace HasPullbackAt

variable {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [NormedAddCommGroup F] [InnerProductSpace ℝ F] {f : E → F} {x : E}

/-- A map that agrees near `x` with a continuous linear map `L` has derivative `L` at `x`, and
every transpose `T` of `L` is its pullback there. -/
theorem of_eventuallyEq_clm (L : E →L[ℝ] F) (T : F → E) (hT : ∀ u δ, ⟪L u, δ⟫ = ⟪u, T δ⟫)
    (hf : f =ᶠ[𝓝 x] L) : HasPullbackAt f L T x :=
  ⟨L.hasFDerivAt.congr_of_eventuallyEq hf, hT⟩

end HasPullbackAt

namespace Shape

/-- The prefix slice of `k ≤ s.extent dim` entries along `dim` is valid. -/
theorem sliceValid_prefix {s : Shape} {dim : Fin s.rank} {k : Nat} (h : k ≤ s.extent dim) :
    SliceValid s [dim] [0] [k] [1] := by
  simpa using sliceValid_singleton s dim 0 k (by simpa using h)

end Shape

namespace Shape.Coord

/-- Replacing the index of `c` along `dim` by `i` gives `c` exactly when `i` is that index. -/
theorem update_eq_self_iff {s : Shape} (c : s.Coord) (dim : Fin s.rank)
    (i : Fin (s.extent dim)) : update c dim i = c ↔ i = get c dim := by
  constructor
  · intro h
    simpa using congrArg (fun c' => get c' dim) h
  · rintro rfl
    exact update_get c dim

end Shape.Coord

namespace Tensor

variable {s : Shape}

/-! ## Locally constant reads -/

/-- A map of real tensors that agrees near `x` with the coordinate pullback `Rep.pull σ` has
derivative `Rep.pullCLM σ` at `x` and pullback `Rep.sumFibers σ`. -/
theorem hasPullbackAt_of_eventuallyEq_pull {t : Shape} {f : Tensor ℝ s → Tensor ℝ t}
    {x : Tensor ℝ s} (σ : t.Coord → s.Coord) (hf : f =ᶠ[𝓝 x] Rep.pull σ) :
    HasPullbackAt f (Rep.pullCLM σ) (Rep.sumFibers σ) x :=
  HasPullbackAt.of_eventuallyEq_clm _ _ (Rep.inner_pull_sumFibers σ) hf

/-- The inputs `y` such that `x a < x b` implies `y a < y b` for all coordinates `a` and `b` form a
neighbourhood of `x`. -/
theorem eventually_forall_lt_of_lt (x : Tensor ℝ s) :
    ∀ᶠ y in 𝓝 x, ∀ a b : s.Coord, x a < x b → y a < y b := by
  refine Filter.eventually_all.mpr fun a => Filter.eventually_all.mpr fun b => ?_
  by_cases hab : x a < x b
  · exact ((coordinateCLM a).continuous.continuousAt.eventually_lt
      (coordinateCLM b).continuous.continuousAt hab).mono fun _ hy _ => hy
  · exact Filter.Eventually.of_forall fun _ h => absurd h hab

/-! ## Folds and fiber sums -/

/-- A left fold that overwrites the accumulator with `g j` at each `j` satisfying `p` returns its
start when no element of the list satisfies `p`. -/
private theorem foldl_ite_of_forall_not {ι β : Type*} (p : ι → Prop) [DecidablePred p]
    (g : ι → β) : ∀ (l : List ι) (a : β), (∀ j ∈ l, ¬p j) →
      l.foldl (fun acc j => if p j then g j else acc) a = a
  | [], _, _ => rfl
  | i :: l, a, h => by
    rw [List.foldl_cons, ite_eq_right (h i List.mem_cons_self)]
    exact foldl_ite_of_forall_not p g l a fun j hj => h j (List.mem_cons_of_mem _ hj)

/-- From zero, a left fold that overwrites the accumulator with `g j` at each `j` satisfying `p`
is the sum of those `g j`, when no two elements of the list both satisfy `p`. -/
private theorem foldl_ite_eq_sum {ι β : Type*} [AddCommMonoid β] (p : ι → Prop)
    [DecidablePred p] (g : ι → β) : ∀ l : List ι, l.Pairwise (fun i j => ¬(p i ∧ p j)) →
      l.foldl (fun acc j => if p j then g j else acc) 0 =
        (l.map fun j => if p j then g j else 0).sum
  | [], _ => rfl
  | i :: l, h => by
    rw [List.pairwise_cons] at h
    by_cases hi : p i
    · have hl : ∀ j ∈ l, ¬p j := fun j hj hj' => h.1 j hj ⟨hi, hj'⟩
      simp only [List.foldl_cons, List.map_cons, List.sum_cons, ite_eq_left hi]
      rw [foldl_ite_of_forall_not p g l _ hl, List.sum_eq_zero, add_zero]
      intro b hb
      obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hb
      exact ite_eq_right (hl j hj)
    · simp only [List.foldl_cons, List.map_cons, List.sum_cons, ite_eq_right hi, zero_add]
      exact foldl_ite_eq_sum p g l h.2

/-- Over `List.finRange n`, from zero, a left fold that overwrites the accumulator with `g j` at
each `j` satisfying `p` is `∑ j, if p j then g j else 0`, when at most one index satisfies `p`. -/
private theorem foldl_finRange_ite_eq_sum {β : Type*} [AddCommMonoid β] {n : Nat}
    (p : Fin n → Prop) [DecidablePred p] (g : Fin n → β) (hp : ∀ i j, p i → p j → i = j) :
    (List.finRange n).foldl (fun acc j => if p j then g j else acc) 0 =
      ∑ j, if p j then g j else 0 := by
  rw [foldl_ite_eq_sum p g _ (List.Pairwise.imp (fun hij h => hij (hp _ _ h.1 h.2))
    (List.nodup_finRange n)), Fin.sum_univ_def]

/-- Over `List.finRange n`, from zero, a left fold that adds `g j` at each `j` satisfying `p` is
`∑ j, if p j then g j else 0`. -/
private theorem foldl_finRange_add_ite_eq_sum {β : Type*} [AddCommMonoid β] {n : Nat}
    (p : Fin n → Prop) [DecidablePred p] (g : Fin n → β) :
    (List.finRange n).foldl (fun acc j => if p j then acc + g j else acc) 0 =
      ∑ j, if p j then g j else 0 := by
  rw [Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]
  exact congrArg (fun step => List.foldl step 0 (List.finRange n))
    (funext fun acc => funext fun j => by split_ifs <;> simp)

/-- A sum over the coordinates `c'` of `s` of the terms for which `c'` with index `idx c'` along
`dim` is `c` is a sum over the fiber of `c` along `dim`. -/
private theorem sum_ite_update_eq {β : Type*} [AddCommMonoid β] (dim : Fin s.rank)
    (idx : s.Coord → Fin (s.extent dim)) (F : s.Coord → β) (c : s.Coord) :
    (∑ c', if Shape.Coord.update c' dim (idx c') = c then F c' else 0) =
      ∑ j, if idx (Shape.Coord.update c dim j) = Shape.Coord.get c dim then
        F (Shape.Coord.update c dim j) else 0 := by
  refine (Fintype.sum_of_injective (fun j => Shape.Coord.update c dim j) (fun i j hij => ?_)
    _ _ (fun c' hc' => ?_) (fun j => ?_)).symm
  · simpa using congrArg (fun c' => Shape.Coord.get c' dim) hij
  · refine ite_eq_right fun h => hc' ⟨Shape.Coord.get c' dim, ?_⟩
    show Shape.Coord.update c dim (Shape.Coord.get c' dim) = c'
    rw [← h, Shape.Coord.update_update, Shape.Coord.update_get]
  · simp only [Shape.Coord.update_update, Shape.Coord.update_eq_self_iff]

/-! ## Sort -/

section Sorting

/-- Two real tensors whose fibers along `dim` compare their entries in the same way have the same
`sort` indices. -/
theorem sort_snd_eq_of_lt_iff (x y : Tensor ℝ s) (dim : Fin s.rank) (descending : Bool)
    (h : ∀ (c : s.Coord) (i j : Fin (s.extent dim)),
      y (Shape.Coord.update c dim i) < y (Shape.Coord.update c dim j) ↔
        x (Shape.Coord.update c dim i) < x (Shape.Coord.update c dim j)) :
    (sort y dim descending).2 = (sort x dim descending).2 :=
  Rep.ext fun c => by
    rw [sort_snd_apply, sort_snd_apply]
    simp only [beq_self_eq_true, gt_iff_lt, h]

/-- Along the fiber through `c`, distinct indices have distinct `sort` indices. -/
theorem sort_snd_update_injective {α : Type} [Storage α] [BEq α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] (x : Tensor α s) (dim : Fin s.rank)
    (descending : Bool) (c : s.Coord) :
    Function.Injective fun j => (sort x dim descending).2 (Shape.Coord.update c dim j) :=
  fun i j hij => List.inj_on_of_nodup_map
    ((perm_sort_snd x dim descending c).nodup_iff.mpr (List.nodup_finRange _))
    (List.mem_finRange i) (List.mem_finRange j) hij

/-- Near a real input each of whose fibers along `dim` has pairwise distinct entries, every input
compares the entries of each fiber along `dim` as the input does. -/
theorem eventually_update_lt_iff (x : Tensor ℝ s) (dim : Fin s.rank)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      x (Shape.Coord.update c dim i) ≠ x (Shape.Coord.update c dim j)) :
    ∀ᶠ y in 𝓝 x, ∀ (c : s.Coord) (i j : Fin (s.extent dim)),
      y (Shape.Coord.update c dim i) < y (Shape.Coord.update c dim j) ↔
        x (Shape.Coord.update c dim i) < x (Shape.Coord.update c dim j) := by
  filter_upwards [eventually_forall_lt_of_lt x] with y hy c i j
  refine ⟨fun hlt => ?_, hy _ _⟩
  by_contra hge
  rcases eq_or_ne i j with rfl | hij
  · exact lt_irrefl _ hlt
  · exact lt_asymm hlt (hy _ _ (lt_of_le_of_ne (not_lt.mp hge) (hx c j i hij.symm)))

/-- Over the reals, `sortBackward` sums the incoming gradient over the fibers of the read of
`sort`, the map `c ↦ update c dim ((sort x dim descending).2 c)`. -/
theorem sortBackward_eq_sumFibers (x g : Tensor ℝ s) (dim : Fin s.rank) (descending : Bool) :
    sortBackward x g dim descending =
      Rep.sumFibers (fun c => Shape.Coord.update c dim ((sort x dim descending).2 c)) g := by
  refine Rep.ext fun c => ?_
  rw [sortBackward_apply, Rep.sumFibers, Rep.get_ofFn, sum_ite_update_eq]
  exact foldl_finRange_ite_eq_sum _ _ fun i j hi hj =>
    sort_snd_update_injective x dim descending c (hi.trans hj.symm)

/-- At a real input each of whose fibers along `dim` has pairwise distinct entries, the values of
`sort` agree near the input with the read along its indices, so they have that read as derivative
and the fiber sum along it as pullback. -/
theorem hasPullbackAt_sort (x : Tensor ℝ s) (dim : Fin s.rank) (descending : Bool)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      x (Shape.Coord.update c dim i) ≠ x (Shape.Coord.update c dim j)) :
    HasPullbackAt (fun y : Tensor ℝ s => (sort y dim descending).1)
      (Rep.pullCLM fun c => Shape.Coord.update c dim ((sort x dim descending).2 c))
      (Rep.sumFibers fun c => Shape.Coord.update c dim ((sort x dim descending).2 c)) x := by
  refine hasPullbackAt_of_eventuallyEq_pull _ ?_
  filter_upwards [eventually_update_lt_iff x dim hx] with y hy
  refine Rep.ext fun c => ?_
  show (sort y dim descending).1 c = Rep.pull _ y c
  rw [sort_fst_apply, Rep.pull_apply, sort_snd_eq_of_lt_iff x y dim descending hy]

end Sorting

/-! ## Running extrema -/

section Running

/-- At a real input in which every tie `x i = x j` with `i < j` of a fiber along `dim` has an
entry before `j` above `x j`, every other index of the prefix through `c` has an entry strictly
below the entry at the index of `cummax` at `c`. -/
private theorem lt_cummax_of_ne (x : Tensor ℝ s) (dim : Fin s.rank)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i < j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        ∃ l < j, x (Shape.Coord.update c dim j) < x (Shape.Coord.update c dim l))
    (c : s.Coord) (i : Fin (s.extent dim)) (hi : i.val ≤ (Shape.Coord.get c dim).val)
    (hne : i ≠ (cummax x dim).2 c) :
    x (Shape.Coord.update c dim i) < x (Shape.Coord.update c dim ((cummax x dim).2 c)) := by
  have hle : ∀ l : Fin (s.extent dim), l.val ≤ (Shape.Coord.get c dim).val →
      x (Shape.Coord.update c dim l) ≤ x (Shape.Coord.update c dim ((cummax x dim).2 c)) := by
    intro l hl
    have h := le_cummax x dim c id (fun _ => beq_self_eq_true _) (fun _ _ => Iff.rfl) l hl
    rw [cummax_fst_apply] at h
    exact h
  refine lt_of_le_of_ne (hle i hi) fun heq => ?_
  rcases lt_or_gt_of_ne hne with h | h
  · obtain ⟨l, hl, hlt⟩ := hx c i _ h heq
    exact absurd hlt (not_lt.mpr (hle l (Nat.le_of_lt (Nat.lt_of_lt_of_le hl
      (cummax_snd_le x dim c)))))
  · obtain ⟨l, hl, hlt⟩ := hx c _ i h heq.symm
    exact absurd (heq.symm.trans_lt hlt) (not_lt.mpr (hle l (Nat.le_of_lt
      (Nat.lt_of_lt_of_le hl hi))))

/-- At a real input in which every tie `x i = x j` with `i < j` of a fiber along `dim` has an
entry before `j` below `x j`, every other index of the prefix through `c` has an entry strictly
above the entry at the index of `cummin` at `c`. -/
private theorem cummin_lt_of_ne (x : Tensor ℝ s) (dim : Fin s.rank)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i < j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        ∃ l < j, x (Shape.Coord.update c dim l) < x (Shape.Coord.update c dim j))
    (c : s.Coord) (i : Fin (s.extent dim)) (hi : i.val ≤ (Shape.Coord.get c dim).val)
    (hne : i ≠ (cummin x dim).2 c) :
    x (Shape.Coord.update c dim ((cummin x dim).2 c)) < x (Shape.Coord.update c dim i) := by
  have hle : ∀ l : Fin (s.extent dim), l.val ≤ (Shape.Coord.get c dim).val →
      x (Shape.Coord.update c dim ((cummin x dim).2 c)) ≤ x (Shape.Coord.update c dim l) := by
    intro l hl
    have h := cummin_le x dim c id (fun _ => beq_self_eq_true _) (fun _ _ => Iff.rfl) l hl
    rw [cummin_fst_apply] at h
    exact h
  refine lt_of_le_of_ne (hle i hi) fun heq => ?_
  rcases lt_or_gt_of_ne hne with h | h
  · obtain ⟨l, hl, hlt⟩ := hx c i _ h heq.symm
    exact absurd hlt (not_lt.mpr (hle l (Nat.le_of_lt (Nat.lt_of_lt_of_le hl
      (cummin_snd_le x dim c)))))
  · obtain ⟨l, hl, hlt⟩ := hx c _ i h heq
    exact absurd (hlt.trans_eq heq.symm) (not_lt.mpr (hle l (Nat.le_of_lt
      (Nat.lt_of_lt_of_le hl hi))))

/-- A real tensor `y` whose entry at the index of `cummax x` at `c` is strictly above its other
entries of the prefix through `c` has the same `cummax` index at `c`. -/
private theorem cummax_snd_eq_of_lt (x y : Tensor ℝ s) (dim : Fin s.rank) (c : s.Coord)
    (hy : ∀ i : Fin (s.extent dim), i.val ≤ (Shape.Coord.get c dim).val →
      i ≠ (cummax x dim).2 c →
        y (Shape.Coord.update c dim i) < y (Shape.Coord.update c dim ((cummax x dim).2 c))) :
    (cummax y dim).2 c = (cummax x dim).2 c := by
  by_contra hne
  have h := le_cummax y dim c id (fun _ => beq_self_eq_true _) (fun _ _ => Iff.rfl)
    ((cummax x dim).2 c) (cummax_snd_le x dim c)
  rw [cummax_fst_apply] at h
  exact absurd (hy _ (cummax_snd_le y dim c) hne) (not_lt.mpr h)

/-- A real tensor `y` whose entry at the index of `cummin x` at `c` is strictly below its other
entries of the prefix through `c` has the same `cummin` index at `c`. -/
private theorem cummin_snd_eq_of_lt (x y : Tensor ℝ s) (dim : Fin s.rank) (c : s.Coord)
    (hy : ∀ i : Fin (s.extent dim), i.val ≤ (Shape.Coord.get c dim).val →
      i ≠ (cummin x dim).2 c →
        y (Shape.Coord.update c dim ((cummin x dim).2 c)) < y (Shape.Coord.update c dim i)) :
    (cummin y dim).2 c = (cummin x dim).2 c := by
  by_contra hne
  have h := cummin_le y dim c id (fun _ => beq_self_eq_true _) (fun _ _ => Iff.rfl)
    ((cummin x dim).2 c) (cummin_snd_le x dim c)
  rw [cummin_fst_apply] at h
  exact absurd (hy _ (cummin_snd_le y dim c) hne) (not_lt.mpr h)

/-- Over the reals, `cummaxBackward` sums the incoming gradient over the fibers of the read of
`cummax`, the map `c ↦ update c dim ((cummax x dim).2 c)`. -/
theorem cummaxBackward_eq_sumFibers (x g : Tensor ℝ s) (dim : Fin s.rank) :
    cummaxBackward x g dim =
      Rep.sumFibers (fun c => Shape.Coord.update c dim ((cummax x dim).2 c)) g := by
  refine Rep.ext fun c => ?_
  rw [cummaxBackward_apply, Rep.sumFibers, Rep.get_ofFn, sum_ite_update_eq]
  exact foldl_finRange_add_ite_eq_sum _ _

/-- Over the reals, `cumminBackward` sums the incoming gradient over the fibers of the read of
`cummin`, the map `c ↦ update c dim ((cummin x dim).2 c)`. -/
theorem cumminBackward_eq_sumFibers (x g : Tensor ℝ s) (dim : Fin s.rank) :
    cumminBackward x g dim =
      Rep.sumFibers (fun c => Shape.Coord.update c dim ((cummin x dim).2 c)) g := by
  refine Rep.ext fun c => ?_
  rw [cumminBackward_apply, Rep.sumFibers, Rep.get_ofFn, sum_ite_update_eq]
  exact foldl_finRange_add_ite_eq_sum _ _

/-- At a real input in which every tie `x i = x j` with `i < j` of a fiber along `dim` has an
entry before `j` above `x j`, the values of `cummax` agree near the input with the read along its
indices, so they have that read as derivative and the fiber sum along it as pullback. -/
theorem hasPullbackAt_cummax (x : Tensor ℝ s) (dim : Fin s.rank)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i < j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        ∃ l < j, x (Shape.Coord.update c dim j) < x (Shape.Coord.update c dim l)) :
    HasPullbackAt (fun y : Tensor ℝ s => (cummax y dim).1)
      (Rep.pullCLM fun c => Shape.Coord.update c dim ((cummax x dim).2 c))
      (Rep.sumFibers fun c => Shape.Coord.update c dim ((cummax x dim).2 c)) x := by
  refine hasPullbackAt_of_eventuallyEq_pull _ ?_
  filter_upwards [eventually_forall_lt_of_lt x] with y hy
  refine Rep.ext fun c => ?_
  show (cummax y dim).1 c = Rep.pull _ y c
  rw [cummax_fst_apply, Rep.pull_apply, cummax_snd_eq_of_lt x y dim c fun i hi hne =>
    hy _ _ (lt_cummax_of_ne x dim hx c i hi hne)]

/-- At a real input in which every tie `x i = x j` with `i < j` of a fiber along `dim` has an
entry before `j` below `x j`, the values of `cummin` agree near the input with the read along its
indices, so they have that read as derivative and the fiber sum along it as pullback. -/
theorem hasPullbackAt_cummin (x : Tensor ℝ s) (dim : Fin s.rank)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i < j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        ∃ l < j, x (Shape.Coord.update c dim l) < x (Shape.Coord.update c dim j)) :
    HasPullbackAt (fun y : Tensor ℝ s => (cummin y dim).1)
      (Rep.pullCLM fun c => Shape.Coord.update c dim ((cummin x dim).2 c))
      (Rep.sumFibers fun c => Shape.Coord.update c dim ((cummin x dim).2 c)) x := by
  refine hasPullbackAt_of_eventuallyEq_pull _ ?_
  filter_upwards [eventually_forall_lt_of_lt x] with y hy
  refine Rep.ext fun c => ?_
  show (cummin y dim).1 c = Rep.pull _ y c
  rw [cummin_fst_apply, Rep.pull_apply, cummin_snd_eq_of_lt x y dim c fun i hi hne =>
    hy _ _ (cummin_lt_of_ne x dim hx c i hi hne)]

end Running

/-! ## Top-k -/

section TopK

variable {dim : Fin s.rank} {k : Nat}

/-- The coordinate that the prefix slice of `k` entries along `dim` reads has an index below `k`
along `dim`. -/
theorem val_get_sliceSource_lt (hv : Shape.SliceValid s [dim] [0] [k] [1])
    (c : (s.slice [dim] [0] [k] [1]).Coord) :
    (Shape.Coord.get (sliceSource hv c) dim).val < k := by
  have hlt := Shape.Coord.index_lt c (k := dim.val) (by simp)
  rw [Shape.extent_slice] at hlt
  rw [val_get_sliceSource]
  simpa using hlt

/-- The prefix slice of `k` entries along `dim` reads at `Shape.prefixCoord s dim k c j` the
coordinate `c` with index `j` along `dim`. -/
theorem sliceSource_prefixCoord (h : k ≤ s.extent dim) (hv : Shape.SliceValid s [dim] [0] [k] [1])
    (c : s.Coord) (j : Fin k) :
    sliceSource hv (Shape.prefixCoord s dim k c j) = Shape.Coord.update c dim (Fin.castLE h j) := by
  apply Shape.Coord.ext_val
  intro d
  by_cases hd : d = dim
  · subst hd
    simp [sliceSource, Shape.prefixCoord]
  · rw [Shape.Coord.get_update_of_ne _ _ hd]
    simp [sliceSource, Shape.prefixCoord, hd]

/-- An index of `topk` is the index of `sort` at the coordinate that the prefix slice reads. -/
theorem topk_snd_apply_eq_sort_snd {α : Type} [Storage α] [BEq α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] (x : Tensor α s) (largest : Bool)
    (h : k ≤ s.extent dim) (c : (s.slice [dim] [0] [k] [1]).Coord) :
    (topk x dim k largest h).2 c =
      (sort x dim largest).2 (sliceSource (Shape.sliceValid_prefix h) c) :=
  Rep.pull_apply (sliceSource (Shape.sliceValid_prefix h)) (sort x dim largest).2 c

/-- If `σ` permutes `Fin n` so that a key `a` decreases along it, and every tie of `a` has at least
`k` keys strictly above it, then below rank `k` the key at each rank is strictly above the keys at
all later ranks. -/
private theorem lt_of_le_card_filter {n : Nat} {σ : Fin n → Fin n} (hσ : Function.Injective σ)
    {a : Fin n → ℝ} (ha : ∀ p q : Fin n, p ≤ q → a (σ q) ≤ a (σ p))
    (htie : ∀ i j, i ≠ j → a i = a j → k ≤ (Finset.univ.filter fun l => a i < a l).card)
    {p q : Fin n} (hp : p.val < k) (hpq : p < q) : a (σ q) < a (σ p) := by
  refine lt_of_le_of_ne (ha p q hpq.le) fun heq => ?_
  have hcard := htie (σ q) (σ p) (fun h => hpq.ne (hσ h).symm) heq
  have hsub : (Finset.univ.filter fun l => a (σ q) < a l) ⊆ (Finset.Iio p).image σ := by
    intro l hl
    obtain ⟨r, rfl⟩ := Finite.surjective_of_injective hσ l
    rw [Finset.mem_filter] at hl
    refine Finset.mem_image.mpr ⟨r, Finset.mem_Iio.mpr ?_, rfl⟩
    by_contra hr
    exact absurd (ha p r (not_lt.mp hr)) (not_le.mpr (heq.symm.trans_lt hl.2))
  have hle := (Finset.card_le_card hsub).trans Finset.card_image_le
  rw [Fin.card_Iio] at hle
  omega

/-- Let `σ` and `τ` permute `Fin n`, with a key `b` decreasing along `τ`. If below rank `k` the key
`a` at each rank of `σ` is strictly above the keys at all later ranks, and `b` keeps every strict
comparison of `a`, then `τ` agrees with `σ` below rank `k`. -/
private theorem eq_of_lt_of_val_lt {n : Nat} {σ τ : Fin n → Fin n} (hσ : Function.Injective σ)
    (hτ : Function.Injective τ) {a b : Fin n → ℝ}
    (hb : ∀ p q : Fin n, p ≤ q → b (τ q) ≤ b (τ p))
    (hstrict : ∀ p q : Fin n, p.val < k → p < q → a (σ q) < a (σ p))
    (hab : ∀ i j, a i < a j → b i < b j) (p : Fin n) (hp : p.val < k) : τ p = σ p := by
  revert hp
  refine (wellFounded_lt (α := Fin n)).induction (C := fun p => p.val < k → τ p = σ p) p ?_
  intro p ih hp
  obtain ⟨r, hr⟩ := Finite.surjective_of_injective hτ (σ p)
  obtain ⟨q, hq⟩ := Finite.surjective_of_injective hσ (τ p)
  have hpr : p ≤ r := not_lt.mp fun hrp =>
    (ne_of_lt hrp) (hσ ((ih r hrp (lt_trans (Fin.lt_def.mp hrp) hp)).symm.trans hr))
  have hpq : p ≤ q := not_lt.mp fun hqp =>
    (ne_of_lt hqp) (hτ ((ih q hqp (lt_trans (Fin.lt_def.mp hqp) hp)).trans hq))
  rcases hpq.lt_or_eq with hlt | rfl
  · have h1 := hab _ _ (hstrict p q hp hlt)
    rw [hq, ← hr] at h1
    exact absurd (hb p r hpr) (not_le.mpr h1)
  · exact hq.symm

/-- At a real input in which every tie of a fiber along `dim` has at least `k` entries strictly
above it (strictly below it when `largest = false`), every input that keeps the strict comparisons
of the input has the same `sort` indices at the coordinates with index below `k` along `dim`. -/
private theorem sort_snd_eq_of_val_lt (x y : Tensor ℝ s) (largest : Bool)
    (hlarge : largest = true → ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        k ≤ (Finset.univ.filter fun l =>
          x (Shape.Coord.update c dim i) < x (Shape.Coord.update c dim l)).card)
    (hsmall : largest = false → ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        k ≤ (Finset.univ.filter fun l =>
          x (Shape.Coord.update c dim l) < x (Shape.Coord.update c dim i)).card)
    (hy : ∀ a b : s.Coord, x a < x b → y a < y b) (c : s.Coord)
    (hc : (Shape.Coord.get c dim).val < k) :
    (sort y dim largest).2 c = (sort x dim largest).2 c := by
  have key : ∀ p : Fin (s.extent dim), p.val < k →
      (sort y dim largest).2 (Shape.Coord.update c dim p) =
        (sort x dim largest).2 (Shape.Coord.update c dim p) := by
    cases largest with
    | true =>
      have hx : ∀ p q : Fin (s.extent dim), p ≤ q →
          x (Shape.Coord.update c dim ((sort x dim true).2 (Shape.Coord.update c dim q))) ≤
            x (Shape.Coord.update c dim ((sort x dim true).2 (Shape.Coord.update c dim p))) :=
        fun p q hpq => by
          have h := le_sort_fst x dim c id (fun _ => beq_self_eq_true _) (fun _ _ => Iff.rfl) hpq
          simpa only [sort_fst_apply, Shape.Coord.update_update, id_eq] using h
      refine eq_of_lt_of_val_lt (sort_snd_update_injective x dim true c)
        (sort_snd_update_injective y dim true c) (a := fun i => x (Shape.Coord.update c dim i))
        (b := fun i => y (Shape.Coord.update c dim i)) (fun p q hpq => ?_)
        (fun p q hp hpq => lt_of_le_card_filter (sort_snd_update_injective x dim true c)
          (a := fun i => x (Shape.Coord.update c dim i)) hx (hlarge rfl c) hp hpq)
        (fun i j => hy _ _)
      have h := le_sort_fst y dim c id (fun _ => beq_self_eq_true _) (fun _ _ => Iff.rfl) hpq
      simpa only [sort_fst_apply, Shape.Coord.update_update, id_eq] using h
    | false =>
      have hx : ∀ p q : Fin (s.extent dim), p ≤ q →
          -x (Shape.Coord.update c dim ((sort x dim false).2 (Shape.Coord.update c dim q))) ≤
            -x (Shape.Coord.update c dim ((sort x dim false).2 (Shape.Coord.update c dim p))) :=
        fun p q hpq => by
          have h := sort_fst_le x dim c id (fun _ => beq_self_eq_true _) (fun _ _ => Iff.rfl) hpq
          simpa only [sort_fst_apply, Shape.Coord.update_update, id_eq, neg_le_neg_iff] using h
      refine eq_of_lt_of_val_lt (sort_snd_update_injective x dim false c)
        (sort_snd_update_injective y dim false c) (a := fun i => -x (Shape.Coord.update c dim i))
        (b := fun i => -y (Shape.Coord.update c dim i)) (fun p q hpq => ?_)
        (fun p q hp hpq => lt_of_le_card_filter (sort_snd_update_injective x dim false c)
          (a := fun i => -x (Shape.Coord.update c dim i)) hx (fun i j hij heq => ?_) hp hpq)
        (fun i j hij => neg_lt_neg (hy _ _ (neg_lt_neg_iff.mp hij)))
      · have h := sort_fst_le y dim c id (fun _ => beq_self_eq_true _) (fun _ _ => Iff.rfl) hpq
        simpa only [sort_fst_apply, Shape.Coord.update_update, id_eq, neg_le_neg_iff] using h
      · simpa only [neg_lt_neg_iff] using hsmall rfl c i j hij (neg_inj.mp heq)
  simpa only [Shape.Coord.update_get] using key (Shape.Coord.get c dim) hc

/-- Over the reals, `topkBackward` sums the incoming gradient over the fibers of the read of
`topk`, the map `c ↦ update (sliceSource _ c) dim ((topk x dim k largest h).2 c)`. -/
theorem topkBackward_eq_sumFibers (x : Tensor ℝ s) (dim : Fin s.rank) (k : Nat) (largest : Bool)
    (h : k ≤ s.extent dim) (g : Tensor ℝ (s.slice [dim] [0] [k] [1])) :
    topkBackward x dim k largest h g = Rep.sumFibers (fun c => Shape.Coord.update
      (sliceSource (Shape.sliceValid_prefix h) c) dim ((topk x dim k largest h).2 c)) g := by
  have hidx : ∀ (c : s.Coord) (j : Fin k),
      (topk x dim k largest h).2 (Shape.prefixCoord s dim k c j) =
        (sort x dim largest).2 (Shape.Coord.update c dim (Fin.castLE h j)) := fun c j =>
    slice_prefixCoord (sort x dim largest).2 dim k h (Shape.sliceValid_prefix h) c j
  refine Rep.ext fun c => ?_
  rw [topkBackward_apply, Rep.sumFibers, Rep.get_ofFn]
  refine (foldl_finRange_ite_eq_sum _ _ fun i j hi hj => Fin.ext ?_).trans ?_
  · have hij := sort_snd_update_injective x dim largest c
      (((hidx c i).symm.trans hi).trans (hj.symm.trans (hidx c j)))
    have hval := congrArg Fin.val hij
    exact hval
  refine Fintype.sum_of_injective (Shape.prefixCoord s dim k c) (fun i j hij => ?_) _ _
    (fun c' hc' => ?_) (fun j => ?_)
  · have hs := congrArg (fun c' => (Shape.Coord.get (sliceSource (Shape.sliceValid_prefix h) c')
      dim).val) hij
    simp only [sliceSource_prefixCoord h, Shape.Coord.get_update_self, Fin.val_castLE] at hs
    exact Fin.ext hs
  · refine ite_eq_right fun hc => hc' ⟨⟨(Shape.Coord.get
      (sliceSource (Shape.sliceValid_prefix h) c') dim).val, val_get_sliceSource_lt _ c'⟩, ?_⟩
    apply sliceSource_injective (Shape.sliceValid_prefix h)
    beta_reduce at hc
    rw [sliceSource_prefixCoord h, ← hc, Shape.Coord.update_update,
      Shape.Coord.update_eq_self_iff]
    exact Fin.ext rfl
  · refine if_congr ?_ rfl rfl
    rw [sliceSource_prefixCoord h, Shape.Coord.update_update, Shape.Coord.update_eq_self_iff]

/-- At a real input in which every tie of a fiber along `dim` has at least `k` entries strictly
above it (strictly below it when `largest = false`), the values of `topk` agree near the input
with the read along its indices, so they have that read as derivative and the fiber sum along it
as pullback. -/
theorem hasPullbackAt_topk (x : Tensor ℝ s) (dim : Fin s.rank) (k : Nat) (largest : Bool)
    (h : k ≤ s.extent dim)
    (hlarge : largest = true → ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        k ≤ (Finset.univ.filter fun l =>
          x (Shape.Coord.update c dim i) < x (Shape.Coord.update c dim l)).card)
    (hsmall : largest = false → ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        k ≤ (Finset.univ.filter fun l =>
          x (Shape.Coord.update c dim l) < x (Shape.Coord.update c dim i)).card) :
    HasPullbackAt (fun y : Tensor ℝ s => (topk y dim k largest h).1)
      (Rep.pullCLM fun c => Shape.Coord.update (sliceSource (Shape.sliceValid_prefix h) c) dim
        ((topk x dim k largest h).2 c))
      (Rep.sumFibers fun c => Shape.Coord.update (sliceSource (Shape.sliceValid_prefix h) c) dim
        ((topk x dim k largest h).2 c)) x := by
  refine hasPullbackAt_of_eventuallyEq_pull _ ?_
  filter_upwards [eventually_forall_lt_of_lt x] with y hy
  refine Rep.ext fun c => ?_
  show (topk y dim k largest h).1 c = Rep.pull _ y c
  rw [Rep.pull_apply, topk_fst y dim k largest h (Shape.sliceValid_prefix h), slice_eq_pull,
    Rep.pull_apply, sort_fst_apply, sort_snd_eq_of_val_lt x y largest hlarge hsmall hy _
      (val_get_sliceSource_lt _ c), topk_snd_apply_eq_sort_snd]

end TopK

end Tensor

/-! ## Pullback rules -/

namespace HasPullbackAt

open Tensor

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s : Shape} {x : E}
  {f : E → Tensor ℝ s} {Df : E →L[ℝ] Tensor ℝ s} {bf : Tensor ℝ s → E}

/-- At a point where each fiber of `f x` along `dim` has pairwise distinct entries, the values of
the real `sort` of `f` along `dim` have the read along the indices of `sort (f x)` as derivative
and the stored `sortBackward` at `f x` as pullback, composed with those of `f`. -/
@[fun_prop]
theorem sort_fst (dim : Fin s.rank) (descending : Bool)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      f x (Shape.Coord.update c dim i) ≠ f x (Shape.Coord.update c dim j))
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => (Tensor.sort (f y) dim descending).1)
      ((Rep.pullCLM fun c =>
        Shape.Coord.update c dim ((Tensor.sort (f x) dim descending).2 c)).comp Df)
      (fun δ => bf (Tensor.sortBackward (f x) δ dim descending)) x :=
  ((hasPullbackAt_sort (f x) dim descending hx).comp hf).congr_deriv rfl fun δ =>
    congrArg bf (sortBackward_eq_sumFibers (f x) δ dim descending).symm

/-- At a point where every tie `f x i = f x j` with `i < j` of a fiber along `dim` has an entry
before `j` above it, the values of the real `cummax` of `f` along `dim` have the read along the
indices of `cummax (f x)` as derivative and the stored `cummaxBackward` at `f x` as pullback,
composed with those of `f`. -/
@[fun_prop]
theorem cummax_fst (dim : Fin s.rank)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i < j →
      f x (Shape.Coord.update c dim i) = f x (Shape.Coord.update c dim j) →
        ∃ l < j, f x (Shape.Coord.update c dim j) < f x (Shape.Coord.update c dim l))
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => (Tensor.cummax (f y) dim).1)
      ((Rep.pullCLM fun c => Shape.Coord.update c dim ((Tensor.cummax (f x) dim).2 c)).comp Df)
      (fun δ => bf (Tensor.cummaxBackward (f x) δ dim)) x :=
  ((hasPullbackAt_cummax (f x) dim hx).comp hf).congr_deriv rfl fun δ =>
    congrArg bf (cummaxBackward_eq_sumFibers (f x) δ dim).symm

/-- At a point where every tie `f x i = f x j` with `i < j` of a fiber along `dim` has an entry
before `j` below it, the values of the real `cummin` of `f` along `dim` have the read along the
indices of `cummin (f x)` as derivative and the stored `cumminBackward` at `f x` as pullback,
composed with those of `f`. -/
@[fun_prop]
theorem cummin_fst (dim : Fin s.rank)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i < j →
      f x (Shape.Coord.update c dim i) = f x (Shape.Coord.update c dim j) →
        ∃ l < j, f x (Shape.Coord.update c dim l) < f x (Shape.Coord.update c dim j))
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => (Tensor.cummin (f y) dim).1)
      ((Rep.pullCLM fun c => Shape.Coord.update c dim ((Tensor.cummin (f x) dim).2 c)).comp Df)
      (fun δ => bf (Tensor.cumminBackward (f x) δ dim)) x :=
  ((hasPullbackAt_cummin (f x) dim hx).comp hf).congr_deriv rfl fun δ =>
    congrArg bf (cumminBackward_eq_sumFibers (f x) δ dim).symm

-- When it registers the next rule, `fun_prop` would reduce the first component of `Tensor.topk`
-- to the `Tensor.slice` of its definition and file the rule there, where `op_deriv`, which leaves
-- `Tensor.topk` folded, never looks. With `Tensor.topk` irreducible the rule is filed under
-- `Prod.fst`, as those of `sort`, `cummax` and `cummin` are.
attribute [local irreducible] Tensor.topk in
/-- At a point where every tie of a fiber of `f x` along `dim` has at least `k` entries strictly
above it (strictly below it when `largest = false`), the values of the real `topk` of `f` along
`dim` have the read along the indices of `topk (f x)` as derivative and the stored `topkBackward`
at `f x` as pullback, composed with those of `f`. -/
@[fun_prop]
theorem topk_fst (dim : Fin s.rank) (k : Nat) (largest : Bool) (h : k ≤ s.extent dim)
    (hlarge : largest = true → ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      f x (Shape.Coord.update c dim i) = f x (Shape.Coord.update c dim j) →
        k ≤ (Finset.univ.filter fun l =>
          f x (Shape.Coord.update c dim i) < f x (Shape.Coord.update c dim l)).card)
    (hsmall : largest = false → ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      f x (Shape.Coord.update c dim i) = f x (Shape.Coord.update c dim j) →
        k ≤ (Finset.univ.filter fun l =>
          f x (Shape.Coord.update c dim l) < f x (Shape.Coord.update c dim i)).card)
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => (Tensor.topk (f y) dim k largest h).1)
      ((Rep.pullCLM fun c => Shape.Coord.update (sliceSource (Shape.sliceValid_prefix h) c) dim
        ((Tensor.topk (f x) dim k largest h).2 c)).comp Df)
      (fun δ => bf (Tensor.topkBackward (f x) dim k largest h δ)) x :=
  ((hasPullbackAt_topk (f x) dim k largest h hlarge hsmall).comp hf).congr_deriv rfl fun δ =>
    congrArg bf (topkBackward_eq_sumFibers (f x) dim k largest h δ).symm

end HasPullbackAt

/-! ## Certificates -/

-- The rules of `sort`, `cummax`, `cummin` and `topk` above all apply to `Prod.fst`, so `fun_prop`
-- tries each of them on every value goal of the four. With the four operators irreducible here,
-- the rule of another operator fails at once, also at default transparency.
attribute [local irreducible] Tensor.sort Tensor.cummax Tensor.cummin Tensor.topk

/-- The forward of `Op.sort dim descending` is differentiable at a real input each of whose fibers
along `dim` has pairwise distinct entries, and its backward is the adjoint of the derivative of the
value output there. The set is exact: at a tie the first sorted position of the tie group reads the
minimum of the group near the input (the maximum when descending), which has no derivative, as at
`x = (0, 0)` on the shape `[2]`. -/
@[op_cert TorchLean.Op.sort derivative]
theorem sort_backward_eq_adjoint {s : Shape} (dim : Fin s.rank) (descending : Bool)
    (x : Tensor ℝ s)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      x (Shape.Coord.update c dim i) ≠ x (Shape.Coord.update c dim j))
    (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.sort dim descending).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.sort dim descending).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, (), ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.sort dim descending).rule? (α := ℝ) |>.get rfl).forward (x', ())).1)
        x).adjoint δ := by
  op_deriv

/-- At every real input, the backward of `Op.sort dim descending` sends the incoming gradient at
each sorted position to the entry of the input that the position reads, through the index output
of `Tensor.sort`. At a tie group of an ascending sort the value is `δ` paired with the left
partial at the group's lowest index and with the right partial at its highest index, and at the
indices between them it differs from both pairings for some `δ`, with left and right swapped when
descending. -/
@[op_cert TorchLean.Op.sort convention]
theorem sort_backward_eq {s : Shape} (dim : Fin s.rank) (descending : Bool) (x δ : Tensor ℝ s) :
    (((Op.sort dim descending).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, (), ())).1 =
      Tensor.Rep.sumFibers
        (fun c => Shape.Coord.update c dim ((Tensor.sort x dim descending).2 c)) δ := by
  simp only [Op.rule?_sort, Option.get_some]
  exact Tensor.sortBackward_eq_sumFibers x δ dim descending

/-- The forward of `Op.cummax dim` is differentiable at a real input in which every tie
`x i = x j` with `i < j` of a fiber along `dim` has an entry before `j` above `x j`, and its
backward is the adjoint of the derivative of the value output there. The set is exact: where
`x j` equals the maximum of the entries before it, output `j` is near the input the maximum of a
tie group, which has no derivative, as at `x = (0, 0, 2)` at output `1`. -/
@[op_cert TorchLean.Op.cummax derivative]
theorem cummax_backward_eq_adjoint {s : Shape} (dim : Fin s.rank) (x : Tensor ℝ s)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i < j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        ∃ l < j, x (Shape.Coord.update c dim j) < x (Shape.Coord.update c dim l))
    (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.cummax dim).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.cummax dim).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, (), ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.cummax dim).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x).adjoint δ := by
  op_deriv

/-- At every real input, the backward of `Op.cummax dim` adds the incoming gradient at each
coordinate to the entry of the input at the index output of `Tensor.cummax` there, the latest
maximizer of the prefix. At a tie each output adds its `δ` times its right partial in the latest
maximizer and times its left partial in the other maximizers of the prefix. -/
@[op_cert TorchLean.Op.cummax convention]
theorem cummax_backward_eq {s : Shape} (dim : Fin s.rank) (x δ : Tensor ℝ s) :
    (((Op.cummax dim).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, (), ())).1 =
      Tensor.Rep.sumFibers
        (fun c => Shape.Coord.update c dim ((Tensor.cummax x dim).2 c)) δ := by
  simp only [Op.rule?_cummax, Option.get_some]
  exact Tensor.cummaxBackward_eq_sumFibers x δ dim

/-- The forward of `Op.cummin dim` is differentiable at a real input in which every tie
`x i = x j` with `i < j` of a fiber along `dim` has an entry before `j` below `x j`, and its
backward is the adjoint of the derivative of the value output there. The set is exact: where
`x j` equals the minimum of the entries before it, output `j` is near the input the minimum of a
tie group, which has no derivative, as at `x = (0, 0, 2)` at output `1`. -/
@[op_cert TorchLean.Op.cummin derivative]
theorem cummin_backward_eq_adjoint {s : Shape} (dim : Fin s.rank) (x : Tensor ℝ s)
    (hx : ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i < j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        ∃ l < j, x (Shape.Coord.update c dim l) < x (Shape.Coord.update c dim j))
    (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.cummin dim).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.cummin dim).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, (), ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.cummin dim).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x).adjoint δ := by
  op_deriv

/-- At every real input, the backward of `Op.cummin dim` adds the incoming gradient at each
coordinate to the entry of the input at the index output of `Tensor.cummin` there, the latest
minimizer of the prefix. At a tie each output adds its `δ` times its left partial in the latest
minimizer and times its right partial in the other minimizers of the prefix. -/
@[op_cert TorchLean.Op.cummin convention]
theorem cummin_backward_eq {s : Shape} (dim : Fin s.rank) (x δ : Tensor ℝ s) :
    (((Op.cummin dim).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, (), ())).1 =
      Tensor.Rep.sumFibers
        (fun c => Shape.Coord.update c dim ((Tensor.cummin x dim).2 c)) δ := by
  simp only [Op.rule?_cummin, Option.get_some]
  exact Tensor.cumminBackward_eq_sumFibers x δ dim

/-- The forward of `Op.topk dim k largest h` is differentiable at a real input in which every tie
`x i = x j` with `i ≠ j` of a fiber along `dim` has at least `k` entries strictly above `x i`
(strictly below it when `largest = false`), and its backward is the adjoint of the derivative of
the value output there. The set is exact: off it a tie group starts within the first `k` sorted
positions, and near the input the output there is the maximum of the group (the minimum when
`largest = false`), which has no derivative, as at `x = (1, 1)` with `k = 1`. -/
@[op_cert TorchLean.Op.topk derivative]
theorem topk_backward_eq_adjoint {s : Shape} (dim : Fin s.rank) (k : Nat) (largest : Bool)
    (h : k ≤ s.extent dim) (x : Tensor ℝ s)
    (hlarge : largest = true → ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        k ≤ (Finset.univ.filter fun l =>
          x (Shape.Coord.update c dim i) < x (Shape.Coord.update c dim l)).card)
    (hsmall : largest = false → ∀ (c : s.Coord) (i j : Fin (s.extent dim)), i ≠ j →
      x (Shape.Coord.update c dim i) = x (Shape.Coord.update c dim j) →
        k ≤ (Finset.univ.filter fun l =>
          x (Shape.Coord.update c dim l) < x (Shape.Coord.update c dim i)).card)
    (δ : Tensor ℝ (s.slice [dim] [0] [k] [1])) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.topk dim k largest h).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.topk dim k largest h).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, (), ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.topk dim k largest h).rule? (α := ℝ) |>.get rfl).forward (x', ())).1)
        x).adjoint δ := by
  op_deriv

/-- At every real input, the backward of `Op.topk dim k largest h` sends the incoming gradient at
each position of the prefix to the entry of the input that the position reads, through the index
output of `Tensor.topk`, and is zero at the other entries. At a tie group of a fiber with
`largest = true` the value is `δ` paired with the right partial at the group's lowest index,
with the left partial at its highest index and at its members past the first `k` sorted positions,
and at the others it differs from both pairings for some `δ`, with left and right swapped when
`largest = false`. -/
@[op_cert TorchLean.Op.topk convention]
theorem topk_backward_eq {s : Shape} (dim : Fin s.rank) (k : Nat) (largest : Bool)
    (h : k ≤ s.extent dim) (x : Tensor ℝ s) (δ : Tensor ℝ (s.slice [dim] [0] [k] [1])) :
    (((Op.topk dim k largest h).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, (), ())).1 =
      Tensor.Rep.sumFibers (fun c => Shape.Coord.update
        (Tensor.sliceSource (Shape.sliceValid_prefix h) c) dim
        ((Tensor.topk x dim k largest h).2 c)) δ := by
  simp only [Op.rule?_topk, Option.get_some]
  exact Tensor.topkBackward_eq_sumFibers x dim k largest h δ

end TorchLean
