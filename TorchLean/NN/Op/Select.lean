/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.PullbackTensor
public import TorchLean.Tensor.Op.Table.Elementwise
public import TorchLean.Tensor.Op.Table.Movement
public import TorchLean.Tensor.Op.Table.Random
public import TorchLean.NN.FDeriv.Reindex
public import TorchLean.Tensor.Op.Cert

/-!
# Derivative certificates of the selection operators

We prove that the backward of `Op.indexAt`, `Op.fullLike`, `Op.dropout`, `Op.embedding`,
`Op.slice` and `Op.gather` is the adjoint of the derivative of the forward in the value input,
and that the backward of `Op.select` is the adjoint in its two value inputs together, at every
argument of the operators. Each forward is linear in its value inputs over the reals, or constant
in it for `Op.fullLike`. For each tensor function of one value input we give a continuous linear
map, the inner product identity with the stored backward, and a `fun_prop` rule of
`HasPullbackAt`.

A slice reads its input through an injective map of coordinates (`Tensor.sliceSource`), so its
adjoint writes each upstream entry back where the slice read it and zero elsewhere, which is
`Tensor.sliceBackward` (`Tensor.inner_slice`). With a fixed mask and probability, the masked
input of dropout multiplies each entry by a slope, and the slope map is self-adjoint
(`Tensor.inner_masked`). An embedding reads its table at the row named by the index and the column
of the output coordinate (`Tensor.embeddingSource`). Its stored backward folds the upstream rows
into the rows their indices name, which is the sum over the fibers of that map, the adjoint of
the read (`Tensor.inner_embedding`). A gather reads its input through `Tensor.gatherSource`, and
its stored backward adds each upstream entry into the coordinate that `Op.gatherSources` names,
which is again the sum over the fibers of the read (`Tensor.scatter_gatherSources_eq_sumFibers`,
`Tensor.inner_gather`). A selection with a fixed condition scales its first value input by the
indicator of the condition and its second by the indicator of the negation, so
`HasPullbackAt.select` follows from `HasPullbackAt.pointwise₂`.
-/

@[expose] public section

open scoped RealInnerProductSpace

namespace TorchLean

namespace Tensor

variable {s : Shape}

/-! ## Slicing -/

section Slice

variable {dims : List (Fin s.rank)} {starts stops steps : List Nat}

/-- The coordinate of `s` that `slice` reads at the coordinate `c` of the slice. -/
def sliceSource (h : Shape.SliceValid s dims starts stops steps)
    (c : (s.slice dims starts stops steps).Coord) : s.Coord :=
  Shape.Coord.ofFun fun d =>
    ⟨Shape.valueAt dims starts 0 d + Shape.valueAt dims steps 1 d *
        (Shape.Coord.get c (d.cast (s.rank_slice dims starts stops steps).symm)).val,
      h.add_mul_lt d _⟩

/-- A slice is the pullback of its input along `sliceSource`. -/
theorem slice_eq_pull (h : Shape.SliceValid s dims starts stops steps) (x : Tensor ℝ s) :
    slice x dims starts stops steps h = Rep.pull (sliceSource h) x :=
  rfl

/-- The index of `sliceSource h c` along `e` is the start plus the step times the index of `c`. -/
theorem val_get_sliceSource (h : Shape.SliceValid s dims starts stops steps)
    (c : (s.slice dims starts stops steps).Coord) (e : Fin s.rank) :
    (Shape.Coord.get (sliceSource h c) e).val =
      Shape.valueAt dims starts 0 e + Shape.valueAt dims steps 1 e * Shape.Coord.index c e.val := by
  rw [sliceSource, Shape.Coord.get_ofFun]
  exact congrArg (Shape.valueAt dims starts 0 e + Shape.valueAt dims steps 1 e * ·)
    (Shape.Coord.val_get c _)

/-- The index of `sliceSource h c` at axis number `k` is the start plus the step times the index of
`c` there. -/
theorem index_sliceSource (h : Shape.SliceValid s dims starts stops steps)
    (c : (s.slice dims starts stops steps).Coord) {k : Nat} (hk : k < s.rank) :
    Shape.Coord.index (sliceSource h c) k =
      Shape.valueAt dims starts 0 ⟨k, hk⟩ + Shape.valueAt dims steps 1 ⟨k, hk⟩ *
        Shape.Coord.index c k := by
  have hc := val_get_sliceSource h c ⟨k, hk⟩
  rw [Shape.Coord.val_get] at hc
  exact hc

/-- Distinct coordinates of a slice read distinct coordinates of its input. -/
theorem sliceSource_injective (h : Shape.SliceValid s dims starts stops steps) :
    Function.Injective (sliceSource h) := by
  intro a b hab
  apply Shape.Coord.ext_index
  intro k hk
  have hk' : k < s.rank := s.rank_slice dims starts stops steps ▸ hk
  have hidx := congrArg (fun z => Shape.Coord.index z k) hab
  simp only [index_sliceSource h _ hk'] at hidx
  exact Nat.eq_of_mul_eq_mul_left (h.step_pos _) (Nat.add_left_cancel hidx)

/-- A fiber sum of `sliceSource` at a coordinate read at `a` is the upstream entry at `a`. -/
theorem sum_ite_sliceSource_eq (h : Shape.SliceValid s dims starts stops steps)
    (δ : Tensor ℝ (s.slice dims starts stops steps)) {a : (s.slice dims starts stops steps).Coord}
    {c : s.Coord} (ha : sliceSource h a = c) :
    ∑ b, (if sliceSource h b = c then δ b else 0) = δ a := by
  rw [Finset.sum_eq_single_of_mem a (Finset.mem_univ _) fun b _ hb =>
    ite_eq_right fun hbc => hb (sliceSource_injective h (hbc.trans ha.symm))]
  exact ite_eq_left ha

/-- Slicing a real tensor, as a continuous linear map. -/
noncomputable def sliceCLM (dims : List (Fin s.rank)) (starts stops steps : List Nat)
    (h : Shape.SliceValid s dims starts stops steps) :
    Tensor ℝ s →L[ℝ] Tensor ℝ (s.slice dims starts stops steps) :=
  Rep.pullCLM (sliceSource h)

/-- `sliceCLM dims starts stops steps h` applied to `y` is the slice of `y`. -/
@[simp] theorem sliceCLM_apply (h : Shape.SliceValid s dims starts stops steps) (y : Tensor ℝ s) :
    sliceCLM dims starts stops steps h y = slice y dims starts stops steps h :=
  rfl

/-- `sliceBackward` writes each upstream entry back at the coordinate the slice read it from:
it is the fiber sum along `sliceSource`. -/
theorem sliceBackward_eq_sumFibers (h : Shape.SliceValid s dims starts stops steps)
    (δ : Tensor ℝ (s.slice dims starts stops steps)) :
    sliceBackward dims starts stops steps δ = Rep.sumFibers (sliceSource h) δ := by
  have hr := s.rank_slice dims starts stops steps
  have key : ∀ a b m : Nat, 0 < b → a ≤ m → (m - a) % b = 0 → a + b * ((m - a) / b) = m :=
    fun a b m _ ham hmod => by
      rw [Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hmod), Nat.add_sub_cancel' ham]
  refine Rep.ext fun c => ?_
  rw [sliceBackward, Rep.sumFibers, Rep.get_ofFn, Rep.get_ofFn]
  split_ifs with hc
  · refine (sum_ite_sliceSource_eq h δ ?_).symm
    apply Shape.Coord.ext_index
    intro k hk
    have hk' : k < (s.slice dims starts stops steps).rank := hr ▸ hk
    obtain ⟨h1, h2, -⟩ := hc ⟨k, hk'⟩
    rw [Shape.Coord.val_get] at h1 h2
    rw [index_sliceSource h _ hk, Shape.Coord.index_ofFun, dite_eq_left hk', Fin.val_mk,
      Shape.Coord.val_get]
    exact key _ _ _ (h.step_pos _) h1 h2
  · symm
    refine Finset.sum_eq_zero fun b _ => ite_eq_right fun hbc => hc fun d => ?_
    subst hbc
    have hb := (Shape.Coord.get b d).isLt
    rw [Shape.Coord.val_get] at hb
    simp only [val_get_sliceSource, Nat.add_sub_cancel_left, Nat.mul_mod_right,
      Nat.mul_div_cancel_left _ (h.step_pos _)]
    exact ⟨Nat.le_add_right _ _, trivial, hb⟩

/-- `sliceBackward` is the adjoint of `slice` for the Euclidean inner product. -/
theorem inner_slice (h : Shape.SliceValid s dims starts stops steps) (u : Tensor ℝ s)
    (δ : Tensor ℝ (s.slice dims starts stops steps)) :
    ⟪slice u dims starts stops steps h, δ⟫ = ⟪u, sliceBackward dims starts stops steps δ⟫ := by
  rw [slice_eq_pull, sliceBackward_eq_sumFibers h]
  exact Rep.inner_pull_sumFibers _ u δ

end Slice

/-! ## Dropout -/

/-- The masked input of dropout over the reals, `Dropout.masked p mask`, as a continuous linear
map: the slope map that scales a kept entry by `(1 - p)⁻¹` and a dropped entry by zero, and every
entry by zero when `p ≥ 1`. -/
noncomputable def maskedCLM (p : ℝ) (mask : Tensor Bool s) : Tensor ℝ s →L[ℝ] Tensor ℝ s :=
  Op.slopeCLM fun c => if Context.geBool p 1 then 0 else if mask c then (1 - p)⁻¹ else 0

/-- `maskedCLM p mask` applied to `y` is `Dropout.masked p mask y`. -/
@[simp] theorem maskedCLM_apply (p : ℝ) (mask : Tensor Bool s) (y : Tensor ℝ s) :
    maskedCLM p mask y = Dropout.masked p mask y := by
  refine Rep.ext fun c => ?_
  rw [maskedCLM, Op.slopeCLM_apply, Dropout.masked]
  by_cases hp : Context.geBool p 1
  · rw [ite_eq_left hp, ite_eq_left hp, zero_mul, full_apply]
  · rw [ite_eq_right hp, ite_eq_right hp, zipWith_apply]
    by_cases hm : mask c
    · rw [ite_eq_left hm, ite_eq_left hm, div_eq_inv_mul]
    · rw [ite_eq_right hm, ite_eq_right hm, zero_mul, mul_zero]

/-- The masked input of dropout is self-adjoint for the Euclidean inner product. -/
theorem inner_masked (p : ℝ) (mask : Tensor Bool s) (u δ : Tensor ℝ s) :
    ⟪Dropout.masked p mask u, δ⟫ = ⟪u, Dropout.masked p mask δ⟫ := by
  rw [← maskedCLM_apply, ← maskedCLM_apply, maskedCLM, ← ContinuousLinearMap.adjoint_inner_right,
    Op.slopeCLM_adjoint]

/-! ## Embedding -/

section Embedding

variable {num dim : Nat} {t : Shape}

/-- The table coordinate that `embedding` reads at `c`: the row that the index at the leading part
of `c` names, and the column of `c`. -/
def embeddingSource (indices : Tensor (Fin num) t) (c : (t ++ [dim]).Coord) :
    Shape.Coord [num, dim] :=
  (indices (Shape.Coord.appendEquiv t [dim] c).1, (Shape.Coord.appendEquiv t [dim] c).2)

/-- Splitting a coordinate reindexed along `t ++ e = t ++ e'` gives the same leading part. -/
private theorem appendEquiv_reindexEq_fst {t e e' : Shape} (h : t ++ e = t ++ e')
    (c : (t ++ e).Coord) :
    (Shape.Coord.appendEquiv t e' (Shape.Coord.reindexEq h c)).1 =
      (Shape.Coord.appendEquiv t e c).1 := by
  refine Shape.Coord.ext_index fun k hk => ?_
  rw [Shape.Coord.index_appendEquiv_fst _ _ _ k (by rwa [← Shape.rank_eq_length]),
    Shape.Coord.index_reindexEq,
    Shape.Coord.index_appendEquiv_fst _ _ _ k (by rwa [← Shape.rank_eq_length])]

/-- Splitting a coordinate reindexed along `t ++ e = t ++ e'` gives a trailing part with the same
indices. -/
private theorem index_appendEquiv_reindexEq_snd {t e e' : Shape} (h : t ++ e = t ++ e')
    (c : (t ++ e).Coord) (k : Nat) :
    Shape.Coord.index (Shape.Coord.appendEquiv t e' (Shape.Coord.reindexEq h c)).2 k =
      Shape.Coord.index (Shape.Coord.appendEquiv t e c).2 k := by
  rw [Shape.Coord.index_appendEquiv_snd, Shape.Coord.index_reindexEq,
    Shape.Coord.index_appendEquiv_snd]

/-- An entry of an embedding is the table entry at `embeddingSource`. -/
theorem embedding_apply {α : Type} [Storage α] (w : Tensor α [num, dim])
    (indices : Tensor (Fin num) t) (c : (t ++ [dim]).Coord) :
    embedding w indices c = w (embeddingSource indices c) := by
  simp only [embedding, gather, Rep.pull_apply]
  refine congrArg (Rep.get w) (Shape.Coord.ext_index fun k hk => ?_)
  rw [Shape.GatherValid.source, Shape.Coord.index_ofFun, dite_eq_left hk]
  match k, hk with
  | 0, hk =>
    split
    · erw [Fin.val_cast, Tensor.map_apply]
      rw [Fin.val_cast]
      exact congrArg (fun u => (Rep.get indices u).val) (appendEquiv_reindexEq_fst _ c)
    · rename_i ha
      exact absurd (List.mem_singleton_self _) ha
  | 1, hk =>
    split
    · rename_i ha
      exact absurd (congrArg Fin.val (List.mem_singleton.mp ha)) Nat.one_ne_zero
    split
    · rename_i hp
      simp [Shape.noBatch] at hp
    · rw [Fin.val_cast]
      exact index_appendEquiv_reindexEq_snd (e' := [dim]) rfl c 0
  | k + 2, hk => exact absurd hk (by simp)

/-- An embedding reads its table through `embeddingSource`. -/
theorem embedding_eq_pull {α : Type} [Storage α] (w : Tensor α [num, dim])
    (indices : Tensor (Fin num) t) :
    embedding w indices = Rep.pull (embeddingSource indices) w :=
  Rep.ext fun c => by rw [embedding_apply, Rep.pull_apply]

/-- A left fold from zero that adds `f i` at each `i` is the sum of `f`. -/
private theorem foldl_add_eq_sum {β : Type} [AddCommMonoid β] {n : Nat} (step : β → Fin n → β)
    (f : Fin n → β) (h : ∀ acc i, step acc i = acc + f i) :
    (List.finRange n).foldl step 0 = ∑ i, f i := by
  rw [Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]
  exact congrArg (fun g => List.foldl g 0 (List.finRange n))
    (funext fun acc => funext fun i => h acc i)

/-- Over the reals, the stored pullback of an embedding sums the gradient over the fibers of
`embeddingSource`. -/
theorem embeddingBackward_eq_sumFibers (indices : Tensor (Fin num) t)
    (δ : Tensor ℝ (t ++ [dim])) :
    embeddingBackward indices δ = Rep.sumFibers (embeddingSource indices) δ := by
  refine Rep.ext fun c => ?_
  obtain ⟨a, b, ⟨⟩⟩ := c
  simp only [embeddingBackward, scatter, Rep.sumFibers, Rep.get_ofFn]
  erw [full_apply]
  refine (foldl_add_eq_sum _ (fun i => if indices (Shape.Coord.unlinearize i) = a then
      δ ((Shape.Coord.appendEquiv t [dim]).symm (Shape.Coord.unlinearize i, (b, PUnit.unit)))
    else 0) fun acc i => ?_).trans ?_
  · split
    · rename_i hc
      erw [Tensor.map_apply] at hc
      have ha : indices (Shape.Coord.unlinearize i) = a :=
        Fin.ext (congrArg Fin.val (congrFun (Option.some.inj hc) ⟨0, Nat.zero_lt_one⟩))
      rw [ite_eq_left ha]
      rfl
    · rename_i hc
      have ha : indices (Shape.Coord.unlinearize i) ≠ a := fun ha => hc (by
        erw [Tensor.map_apply]
        refine congrArg some (funext fun k => ?_)
        obtain ⟨j, hj⟩ := k
        obtain rfl : j = 0 := Nat.lt_one_iff.mp hj
        exact Fin.ext (congrArg Fin.val ha))
      rw [ite_eq_right ha, add_zero]
  · refine ((Shape.Coord.equivFin t).symm.sum_comp (fun u => if indices u = a then
      δ ((Shape.Coord.appendEquiv t [dim]).symm (u, (b, PUnit.unit))) else 0)).trans ?_
    refine Eq.trans ?_ ((Shape.Coord.appendEquiv t [dim]).symm.sum_comp _)
    rw [Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun u _ => ?_
    by_cases ha : indices u = a
    · simp [embeddingSource, ha]
    · simp [embeddingSource, ha]

/-- The real embedding with fixed indices, as a continuous linear map of the table. -/
noncomputable def embeddingCLM (indices : Tensor (Fin num) t) :
    Tensor ℝ [num, dim] →L[ℝ] Tensor ℝ (t ++ [dim]) :=
  Rep.pullCLM (embeddingSource indices)

/-- `embeddingCLM indices` applied to `w` is `embedding w indices`. -/
@[simp] theorem embeddingCLM_apply (indices : Tensor (Fin num) t) (w : Tensor ℝ [num, dim]) :
    embeddingCLM indices w = embedding w indices :=
  (embedding_eq_pull w indices).symm

/-- Over the reals, the adjoint of an embedding in its table is `embeddingBackward`. -/
theorem inner_embedding (indices : Tensor (Fin num) t) (w : Tensor ℝ [num, dim])
    (δ : Tensor ℝ (t ++ [dim])) :
    ⟪embedding w indices, δ⟫ = ⟪w, embeddingBackward indices δ⟫ := by
  rw [embedding_eq_pull, embeddingBackward_eq_sumFibers]
  exact Rep.inner_pull_sumFibers _ w δ

end Embedding

/-! ## Gathering -/

section Gather

variable {t : Shape} {dims : List (Fin s.rank)} {batchDims : List (Fin s.rank × Fin t.rank)}

/-- The coordinate of `s` that `gather` at the index tuples `tuples` reads at the coordinate `c`
of its output: `h.source` at the tuple of the leading part of `c`, with both parts of `c`. -/
def gatherSource (h : Shape.GatherValid s t dims batchDims)
    (tuples : Tensor (Einops.AxisTuple s.extent dims) t)
    (c : (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)).Coord) : s.Coord :=
  let parts := Shape.Coord.appendEquiv t (s.eraseAxes (dims ++ batchDims.map Prod.fst)) c
  h.source (tuples parts.1) parts.1 parts.2

/-- A gather into `t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)` is the pullback of its
input along `gatherSource`. -/
theorem gather_eq_pull {α : Type} [Storage α] (x : Tensor α s)
    (h : Shape.GatherValid s t dims batchDims) (tuples : Tensor (Einops.AxisTuple s.extent dims) t)
    (hout : t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst) =
      t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)) :
    gather x dims batchDims tuples _ h hout = Rep.pull (gatherSource h tuples) x :=
  Rep.ext fun c => by
    simp only [gather_apply, Rep.pull_apply, Shape.Coord.reindexEq_self, gatherSource]

/-- Along a remaining axis, the coordinate `h.source p ct cr` holds the index of `cr` there. -/
private theorem get_source_otherAxes (h : Shape.GatherValid s t dims batchDims)
    (p : Einops.AxisTuple s.extent dims) (ct : t.Coord)
    (cr : (s.eraseAxes (dims ++ batchDims.map Prod.fst)).Coord)
    (j : Fin (s.otherAxes (dims ++ batchDims.map Prod.fst)).length) :
    Shape.Coord.get (h.source p ct cr) ((s.otherAxes (dims ++ batchDims.map Prod.fst)).get j) =
      (Einops.AxisTuple.coordEquiv s.extent (s.otherAxes (dims ++ batchDims.map Prod.fst)))
        cr j := by
  have hmem := List.get_mem (s.otherAxes (dims ++ batchDims.map Prod.fst)) j
  rw [Shape.mem_otherAxes, List.mem_append, not_or] at hmem
  obtain ⟨hdims, hbatch⟩ := hmem
  have hnone : batchDims.find? (fun q => q.1 = (s.otherAxes _).get j) = none :=
    List.find?_eq_none.mpr fun q hq hqa =>
      hbatch (List.mem_map.mpr ⟨q, hq, of_decide_eq_true hqa⟩)
  apply Fin.ext
  rw [Shape.GatherValid.source, Shape.Coord.get_ofFun, dite_eq_right hdims]
  split
  · rename_i hq
    exact absurd (hnone.symm.trans hq) (by simp)
  · simp only [Fin.val_cast]
    have key : ∀ i i' : Fin (s.otherAxes (dims ++ batchDims.map Prod.fst)).length, i = i' →
        ((Einops.AxisTuple.coordEquiv s.extent
          (s.otherAxes (dims ++ batchDims.map Prod.fst))) cr i).val =
        ((Einops.AxisTuple.coordEquiv s.extent
          (s.otherAxes (dims ++ batchDims.map Prod.fst))) cr i').val := by
      rintro i i' rfl
      rfl
    exact key _ _ (Fin.ext ((Shape.nodup_otherAxes s _).idxOf_getElem j.val j.isLt))

/-- At the index coordinate `u`, `Op.gatherSources` stores the indices that the coordinate read by
`gather` has along the indexed and batch axes, whatever the remaining part `cr`. -/
private theorem gatherSources_apply_eq_source (h : Shape.GatherValid s t dims batchDims)
    (tuples : Tensor (Einops.AxisTuple s.extent dims) t) (u : t.Coord)
    (cr : (s.eraseAxes (dims ++ batchDims.map Prod.fst)).Coord) :
    Op.gatherSources h tuples u = some fun k =>
      Shape.Coord.get (h.source (tuples u) u cr) ((dims ++ batchDims.map Prod.fst).get k) := by
  obtain ⟨hdims, hfst, hdisj⟩ := List.nodup_append.mp h.nodup
  rw [Op.gatherSources, Rep.get_ofFn]
  refine congrArg some (funext fun k => Fin.ext ?_)
  rw [Shape.GatherValid.source, Shape.Coord.get_ofFun]
  by_cases hk : k.val < dims.length
  · have hget : (dims ++ batchDims.map Prod.fst).get k = dims[k.val] :=
      List.getElem_append_left hk
    have ha : (dims ++ batchDims.map Prod.fst).get k ∈ dims := by
      rw [hget]
      exact List.getElem_mem hk
    have hidx : dims.idxOf ((dims ++ batchDims.map Prod.fst).get k) = k.val := by
      rw [hget, hdims.idxOf_getElem]
    simp only [dite_eq_left hk, dite_eq_left ha, Fin.val_cast]
    have key : ∀ i i' : Fin dims.length, i = i' → (tuples u i).val = (tuples u i').val := by
      rintro i i' rfl
      rfl
    exact key _ _ (Fin.ext hidx.symm)
  · have hj : k.val - dims.length < batchDims.length := by
      have := k.isLt
      simp only [List.length_append, List.length_map] at this
      omega
    have hget :
        (dims ++ batchDims.map Prod.fst).get k = (batchDims[k.val - dims.length]'hj).1 := by
      rw [List.get_eq_getElem, List.getElem_append_right (Nat.le_of_not_lt hk), List.getElem_map]
    have ha : (dims ++ batchDims.map Prod.fst).get k ∉ dims := fun hmem =>
      hdisj _ hmem _ (List.mem_map_of_mem (List.getElem_mem hj)) hget
    simp only [dite_eq_right hk, dite_eq_right ha, Fin.val_cast]
    split
    · rename_i q hq
      have hq0 := List.find?_some hq
      have hq1 : q.1 = (dims ++ batchDims.map Prod.fst).get k := of_decide_eq_true hq0
      obtain rfl : q = batchDims[k.val - dims.length]'hj :=
        List.inj_on_of_nodup_map hfst (List.mem_of_find?_eq_some hq) (List.getElem_mem hj)
          (hq1.trans hget)
      simp only [Fin.val_cast]
    · rename_i hnone
      exact (List.find?_eq_none.mp hnone _ (List.getElem_mem hj)
        (decide_eq_true hget.symm)).elim

/-- The coordinate that `gather` reads at the index coordinate `u` and the remaining part `r` is
`c` exactly when `Op.gatherSources` stores the indices of `c` at `u` and `r` holds the other
indices of `c`. -/
private theorem source_eq_iff (h : Shape.GatherValid s t dims batchDims)
    (tuples : Tensor (Einops.AxisTuple s.extent dims) t) (u : t.Coord)
    (r : (s.eraseAxes (dims ++ batchDims.map Prod.fst)).Coord) (c : s.Coord) :
    h.source (tuples u) u r = c ↔
      Op.gatherSources h tuples u =
          some (fun k => Shape.Coord.get c ((dims ++ batchDims.map Prod.fst).get k)) ∧
        r = (Einops.AxisTuple.coordEquiv s.extent
          (s.otherAxes (dims ++ batchDims.map Prod.fst))).symm
          fun k => Shape.Coord.get c ((s.otherAxes (dims ++ batchDims.map Prod.fst)).get k) := by
  constructor
  · rintro rfl
    refine ⟨gatherSources_apply_eq_source h tuples u r, ?_⟩
    exact (Equiv.eq_symm_apply (Einops.AxisTuple.coordEquiv s.extent
      (s.otherAxes (dims ++ batchDims.map Prod.fst)))).mpr
      (funext fun j => (get_source_otherAxes h _ u r j).symm)
  · rintro ⟨hc, hr⟩
    have hD := congrFun (Option.some.inj
      ((gatherSources_apply_eq_source h tuples u r).symm.trans hc))
    refine Shape.Coord.ext fun a => ?_
    by_cases ha : a ∈ dims ++ batchDims.map Prod.fst
    · obtain ⟨k, rfl⟩ := List.get_of_mem ha
      exact hD k
    · obtain ⟨j, rfl⟩ := List.get_of_mem (Shape.mem_otherAxes.mpr ha)
      rw [get_source_otherAxes, hr]
      exact congrFun (Equiv.apply_symm_apply _ _) j

/-- Over the reals, the stored pullback of `gather`, the sum-scatter of the upstream gradient at
the tuples of `Op.gatherSources`, is the sum over the fibers of `gatherSource`. -/
theorem scatter_gatherSources_eq_sumFibers (h : Shape.GatherValid s t dims batchDims)
    (tuples : Tensor (Einops.AxisTuple s.extent dims) t)
    (δ : Tensor ℝ (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst))) :
    scatter .sum (full s 0) (dims ++ batchDims.map Prod.fst) (Op.gatherSources h tuples) δ =
      Rep.sumFibers (gatherSource h tuples) δ := by
  refine Rep.ext fun c => ?_
  simp only [scatter, Rep.sumFibers, Rep.get_ofFn]
  erw [full_apply]
  refine (foldl_add_eq_sum _ (fun i =>
      if Op.gatherSources h tuples (Shape.Coord.unlinearize i) =
          some (fun k => Shape.Coord.get c ((dims ++ batchDims.map Prod.fst).get k)) then
        δ ((Shape.Coord.appendEquiv t (s.eraseAxes (dims ++ batchDims.map Prod.fst))).symm
          (Shape.Coord.unlinearize i, (Einops.AxisTuple.coordEquiv s.extent
            (s.otherAxes (dims ++ batchDims.map Prod.fst))).symm
            fun k => Shape.Coord.get c ((s.otherAxes (dims ++ batchDims.map Prod.fst)).get k)))
      else 0) fun acc i => ?_).trans ?_
  · split_ifs
    · rfl
    · exact (add_zero acc).symm
  · refine ((Shape.Coord.equivFin t).symm.sum_comp (fun u =>
        if Op.gatherSources h tuples u =
            some (fun k => Shape.Coord.get c ((dims ++ batchDims.map Prod.fst).get k)) then
          δ ((Shape.Coord.appendEquiv t (s.eraseAxes (dims ++ batchDims.map Prod.fst))).symm
            (u, (Einops.AxisTuple.coordEquiv s.extent
              (s.otherAxes (dims ++ batchDims.map Prod.fst))).symm
              fun k => Shape.Coord.get c ((s.otherAxes (dims ++ batchDims.map Prod.fst)).get k)))
        else 0)).trans ?_
    refine Eq.trans ?_
      ((Shape.Coord.appendEquiv t (s.eraseAxes (dims ++ batchDims.map Prod.fst))).symm.sum_comp _)
    rw [Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun u _ => ?_
    simp only [gatherSource, Equiv.apply_symm_apply]
    rw [Fintype.sum_eq_single (α := (s.eraseAxes (dims ++ batchDims.map Prod.fst)).Coord)
      ((Einops.AxisTuple.coordEquiv s.extent (s.otherAxes (dims ++ batchDims.map Prod.fst))).symm
        fun k => Shape.Coord.get c ((s.otherAxes (dims ++ batchDims.map Prod.fst)).get k))]
    · by_cases hu : Op.gatherSources h tuples u =
          some (fun k => Shape.Coord.get c ((dims ++ batchDims.map Prod.fst).get k))
      · rw [ite_eq_left hu, ite_eq_left]
        exact (source_eq_iff h tuples u _ c).mpr ⟨hu, rfl⟩
      · rw [ite_eq_right hu, ite_eq_right]
        exact fun hc => hu ((source_eq_iff h tuples u _ c).mp hc).1
    · intro r hr
      exact ite_eq_right fun hc => hr ((source_eq_iff h tuples u r c).mp hc).2

/-- The real gather at fixed index tuples, as a continuous linear map of its input. -/
noncomputable def gatherCLM (h : Shape.GatherValid s t dims batchDims)
    (tuples : Tensor (Einops.AxisTuple s.extent dims) t) :
    Tensor ℝ s →L[ℝ] Tensor ℝ (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)) :=
  Rep.pullCLM (gatherSource h tuples)

/-- `gatherCLM h tuples` applied to `y` is the gather of `y` at `tuples`. -/
@[simp] theorem gatherCLM_apply (h : Shape.GatherValid s t dims batchDims)
    (tuples : Tensor (Einops.AxisTuple s.extent dims) t) (y : Tensor ℝ s) :
    gatherCLM h tuples y =
      gather y dims batchDims tuples (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)) h :=
  (gather_eq_pull y h tuples _).symm

/-- Over the reals, the adjoint of `gather` in its input is the sum-scatter of the upstream
gradient at the tuples of `Op.gatherSources`. -/
theorem inner_gather (h : Shape.GatherValid s t dims batchDims)
    (tuples : Tensor (Einops.AxisTuple s.extent dims) t) (u : Tensor ℝ s)
    (δ : Tensor ℝ (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst))) :
    ⟪gather u dims batchDims tuples (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)) h, δ⟫ =
      ⟪u, scatter .sum (full s 0) (dims ++ batchDims.map Prod.fst)
        (Op.gatherSources h tuples) δ⟫ := by
  rw [gather_eq_pull, scatter_gatherSources_eq_sumFibers]
  exact Rep.inner_pull_sumFibers _ u δ

end Gather

end Tensor

namespace HasPullbackAt

open Tensor

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s : Shape} {x : E}
  {f : E → Tensor ℝ s} {Df : E →L[ℝ] Tensor ℝ s} {bf : Tensor ℝ s → E}

/-- Slicing, pulled back by `sliceBackward`. -/
@[fun_prop]
theorem slice (dims : List (Fin s.rank)) (starts stops steps : List Nat)
    (h : Shape.SliceValid s dims starts stops steps) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.slice (f y) dims starts stops steps h)
      ((sliceCLM dims starts stops steps h).comp Df)
      (fun δ => bf (sliceBackward dims starts stops steps δ)) x := by
  simpa only [sliceCLM_apply] using
    clm (sliceCLM dims starts stops steps h) (sliceBackward dims starts stops steps)
      (fun u δ => by rw [sliceCLM_apply, inner_slice]) hf

/-- The masked input of dropout with a fixed probability and mask, pulled back by itself. -/
@[fun_prop]
theorem masked (p : ℝ) (mask : Tensor Bool s) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Dropout.masked p mask (f y)) ((maskedCLM p mask).comp Df)
      (fun δ => bf (Dropout.masked p mask δ)) x := by
  simpa only [maskedCLM_apply] using
    clm (maskedCLM p mask) (Dropout.masked p mask)
      (fun u δ => by rw [maskedCLM_apply, inner_masked]) hf

/-- An embedding with fixed indices, pulled back in its table by `embeddingBackward`. -/
@[fun_prop]
theorem embedding {num dim : Nat} {t : Shape} {g : E → Tensor ℝ [num, dim]}
    {Dg : E →L[ℝ] Tensor ℝ [num, dim]} {bg : Tensor ℝ [num, dim] → E}
    (indices : Tensor (Fin num) t) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.embedding (g y) indices) ((embeddingCLM indices).comp Dg)
      (fun δ => bg (Tensor.embeddingBackward indices δ)) x := by
  simpa only [embeddingCLM_apply] using
    clm (embeddingCLM indices) (embeddingBackward indices)
      (fun u δ => by rw [embeddingCLM_apply, inner_embedding]) hg

/-- Gathering at fixed index tuples, pulled back by the sum-scatter of the cotangent at the tuples
of `Op.gatherSources`. -/
@[fun_prop]
theorem gather {t : Shape} (dims : List (Fin s.rank)) (batchDims : List (Fin s.rank × Fin t.rank))
    (tuples : Tensor (Einops.AxisTuple s.extent dims) t) (h : Shape.GatherValid s t dims batchDims)
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt
      (fun y => Tensor.gather (f y) dims batchDims tuples
        (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst)) h)
      ((gatherCLM h tuples).comp Df)
      (fun δ => bf (Tensor.scatter .sum (Tensor.full s 0) (dims ++ batchDims.map Prod.fst)
        (Op.gatherSources h tuples) δ)) x := by
  simpa only [gatherCLM_apply] using
    clm (gatherCLM h tuples)
      (Tensor.scatter .sum (Tensor.full s 0) (dims ++ batchDims.map Prod.fst)
        (Op.gatherSources h tuples))
      (fun u δ => by rw [gatherCLM_apply, inner_gather]) hf

/-- Selecting by a fixed condition between `f y` where it holds and `g y` elsewhere: the
derivative scales `Df` by the indicator of the condition and `Dg` by that of its negation, and the
pullback sends the two parts of `selectBackward` to `bf` and `bg`. -/
@[fun_prop]
theorem select (cond : Tensor Bool s) {g : E → Tensor ℝ s} {Dg : E →L[ℝ] Tensor ℝ s}
    {bg : Tensor ℝ s → E} (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.select cond (f y) (g y))
      ((Op.slopeCLM fun c => if cond c then 1 else 0).comp Df +
        (Op.slopeCLM fun c => if cond c then 0 else 1).comp Dg)
      (fun δ => bf (Tensor.selectBackward cond δ).1 + bg (Tensor.selectBackward cond δ).2) x := by
  have h₀ : ∀ δ, Op.slopeCLM (fun c => if cond c then 1 else 0) δ =
      (Tensor.selectBackward cond δ).1 := fun δ =>
    Rep.ext fun c => by cases hc : cond c <;> simp [Tensor.selectBackward, Tensor.map, hc]
  have h₁ : ∀ δ, Op.slopeCLM (fun c => if cond c then 0 else 1) δ =
      (Tensor.selectBackward cond δ).2 := fun δ =>
    Rep.ext fun c => by cases hc : cond c <;> simp [Tensor.selectBackward, Tensor.map, hc]
  refine (pointwise₂ (d₀ := fun c => if cond c then 1 else 0)
    (d₁ := fun c => if cond c then 0 else 1) (Tensor.select cond)
    (fun c p => if cond c then p.1 else p.2) (fun y z c => Tensor.select_apply cond y z c)
    (fun c => ?_) hf hg).congr_deriv rfl fun δ => by rw [h₀, h₁]
  cases hc : cond c
  · simpa [hc] using hasFDerivAt_snd
  · simpa [hc] using hasFDerivAt_fst

end HasPullbackAt

/-! ## Certificates -/

/-- The forward of `Op.indexAt` is differentiable in its input, and its backward is the adjoint of
that derivative. -/
@[op_cert TorchLean.Op.indexAt derivative]
theorem indexAt_backward_eq_adjoint {s : Shape} (dim : Fin s.rank) (index : Fin (s.extent dim))
    (x : Tensor ℝ s) (δ : Tensor ℝ (s.eraseIdx dim)) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.indexAt dim index).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.indexAt dim index).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.indexAt dim index).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x).adjoint δ := by
  simp only [Op.rule?_indexAt, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- The forward of `Op.fullLike` is constant in its value input, so its derivative there is zero,
and its backward, zero, is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.fullLike derivative]
theorem fullLike_backward_eq_adjoint {s : Shape} (x : Tensor ℝ s) (value : Tensor ℝ [])
    (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.fullLike (s := s)).rule? (α := ℝ) |>.get rfl).forward (x', value, ())).1) x ∧
    (((Op.fullLike (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, value, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.fullLike (s := s)).rule? (α := ℝ) |>.get rfl).forward (x', value, ())).1)
        x).adjoint δ := by
  simp only [Op.rule?_fullLike, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · exact Tensor.Rep.ext fun c => by
      simp only [Tensor.map_apply, Tensor.Rep.zero_apply]

/-- With a fixed sample and probability, the forward of `Op.dropout` is differentiable in its
value input, and its backward is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.dropout derivative]
theorem dropout_backward_eq_adjoint {s : Shape} (x sample : Tensor ℝ s) (p : Tensor ℝ [])
    (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.dropout (s := s)).rule? (α := ℝ) |>.get rfl).forward (x', sample, p, ())).1) x ∧
    (((Op.dropout (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, sample, p, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.dropout (s := s)).rule? (α := ℝ) |>.get rfl).forward (x', sample, p, ())).1)
        x).adjoint δ := by
  simp only [Op.rule?_dropout, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- The forward of `Op.embedding` is differentiable in its table at every index tensor, and its
backward in the table is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.embedding derivative]
theorem embedding_backward_eq_adjoint {num dim : Nat} {t : Shape} (w : Tensor ℝ [num, dim])
    (indices : Tensor (Fin num) t) (δ : Tensor ℝ (t ++ [dim])) :
    DifferentiableAt ℝ (fun w' =>
        (((Op.embedding (num := num) (dim := dim) (t := t)).rule? (α := ℝ) |>.get rfl).forward
          (w', indices, ())).1) w ∧
    (((Op.embedding (num := num) (dim := dim) (t := t)).rule? (α := ℝ) |>.get rfl).backward
        (w, indices, ()) (δ, ())).1 =
      (fderiv ℝ (fun w' =>
        (((Op.embedding (num := num) (dim := dim) (t := t)).rule? (α := ℝ) |>.get rfl).forward
          (w', indices, ())).1) w).adjoint δ := by
  simp only [Op.rule?_embedding, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- The forward of `Op.slice` is differentiable in its input, and its backward, the upstream
gradient written back where the slice read it, is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.slice derivative]
theorem slice_backward_eq_adjoint {s : Shape} (dims : List (Fin s.rank))
    (starts stops steps : List Nat) (h : Shape.SliceValid s dims starts stops steps)
    (x : Tensor ℝ s) (δ : Tensor ℝ (s.slice dims starts stops steps)) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.slice dims starts stops steps h).rule? (α := ℝ) |>.get rfl).forward
          (x', ())).1) x ∧
    (((Op.slice dims starts stops steps h).rule? (α := ℝ) |>.get rfl).backward
        (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.slice dims starts stops steps h).rule? (α := ℝ) |>.get rfl).forward
          (x', ())).1) x).adjoint δ := by
  simp only [Op.rule?_slice, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- The forward of `Op.gather` is differentiable in its input at every index tensor, and its
backward in the input, the sum-scatter of the upstream gradient at the gathered coordinates, is
the adjoint of that derivative. -/
@[op_cert TorchLean.Op.gather derivative]
theorem gather_backward_eq_adjoint {s t : Shape} (dims : List (Fin s.rank))
    (batchDims : List (Fin s.rank × Fin t.rank)) (h : Shape.GatherValid s t dims batchDims)
    (x : Tensor ℝ s) (index : Tensor (Port.IndexTuple (dims.map s.extent)) t)
    (δ : Tensor ℝ (t ++ s.eraseAxes (dims ++ batchDims.map Prod.fst))) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.gather dims batchDims h).rule? (α := ℝ) |>.get rfl).forward
          (x', index, ())).1) x ∧
    (((Op.gather dims batchDims h).rule? (α := ℝ) |>.get rfl).backward
        (x, index, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.gather dims batchDims h).rule? (α := ℝ) |>.get rfl).forward
          (x', index, ())).1) x).adjoint δ := by
  simp only [Op.rule?_gather, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- At a fixed condition, the forward of `Op.select` is differentiable in its two value inputs
together, on their L2 product, and its pair of pullbacks is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.select derivative 1 2]
theorem select_prod_backward_eq_adjoint {s : Shape} (cond : Tensor Bool s) (x y δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.select (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (cond, p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (x, y)) ∧
    WithLp.toLp 2
        ((((Op.select (s := s)).rule? (α := ℝ) |>.get rfl).backward
          (cond, x, y, ()) (δ, ())).2.1,
        (((Op.select (s := s)).rule? (α := ℝ) |>.get rfl).backward
          (cond, x, y, ()) (δ, ())).2.2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.select (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (cond, p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (x, y))).adjoint δ := by
  simp only [Op.rule?_select, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · refine withLp_prod_ext ?_ ?_ <;>
      simp only [WithLp.add_fst, WithLp.add_snd, WithLp.toLp_fst, WithLp.toLp_snd, add_zero,
        zero_add]

end TorchLean
