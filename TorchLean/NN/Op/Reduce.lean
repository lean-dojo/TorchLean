/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.PullbackTensor
public import TorchLean.NN.FDeriv.Reindex
public import TorchLean.NN.Op.Pointwise
public import TorchLean.Tensor.Op.Cert

/-!
# Derivative certificates of the reductions

We prove that the backward of `Op.cumsum`, `Op.reduce` and `Op.mean` is the adjoint of the
derivative of the forward in the value input. The reduction over `dims` reads the input at
`Shape.reduceSource o r` for an output coordinate `o` and a reduced coordinate `r`, and this map is
a bijection onto the input coordinates whose inverse sends `c` to the coordinate it broadcasts from
and to its indices along the listed axes. A sum is therefore `reduceBroadcast` along
`Shape.reduceAxes` (`reduce_sum_eq_reduceBroadcast`), whose adjoint is the broadcast that
`Tensor.reduceBackward` stores. A mean is a sum scaled by `meanDenominator`.

A cumulative sum along `dim` adds the entries of a fiber up to an index, and its stored backward
adds the gradient from an index up, so the adjoint identity is a swap of the two indices along
`dim`. A product is smooth, and its stored backward multiplies the gradient by the product of the
other entries of the fiber, which is its partial derivative. A maximum or a minimum is
differentiable exactly where every fiber has a unique extremal entry
(`differentiableAt_reduce_max_iff`). There it agrees near the input with the read of the extremal
entries, and the stored backward sends each gradient to that entry. At a tie we move one of the
tied entries along a line, on which the forward is a translate of `max t 0` (or `min t 0`), and
this function is not differentiable at zero. At every input, ties included, the stored backward
splits the gradient at an output evenly over the entries of its fiber equal to the extremum
(`Op.reduce_backward_eq`).
-/

@[expose] public section

open scoped RealInnerProductSpace

namespace TorchLean

namespace Tensor

variable {s : Shape}

/-! ## Fibers of a reduction -/

section Fibers

variable {dims : List (Fin s.rank)} {keepdim : Bool}

/-- The indices of the coordinate that broadcasting reads. -/
private theorem index_broadcastSource_of_lt {s t : Shape} (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) (d : t.Coord) {k : Nat} (hk : k < s.rank) :
    Shape.Coord.index (broadcastSource dims h d) k =
      if s.extent ⟨k, hk⟩ = 1 then 0 else Shape.Coord.index d (dims ⟨k, hk⟩).val := by
  rw [broadcastSource, Shape.Coord.index_ofFun, dite_eq_left hk]
  split_ifs with hi
  · rfl
  · rw [Fin.val_cast, Shape.Coord.val_get]

/-- An output axis of a reduction placed at a listed axis has extent one. -/
private theorem extent_reduce_of_mem (j : Fin (s.reduce dims keepdim).rank)
    (hj : s.reduceAxes dims keepdim j ∈ dims) : (s.reduce dims keepdim).extent j = 1 := by
  cases keepdim with
  | false => exact absurd hj (Shape.mem_otherAxes.mp (List.getElem_mem _))
  | true =>
    simp only [Shape.extent, Shape.reduce, Bool.cond_true, Shape.keepAxes, List.getElem_ofFn]
    exact ite_eq_left hj

/-- The position of the axis an output axis of a reduction is placed at is that output axis. -/
private theorem keptIndex_reduceAxes (j : Fin (s.reduce dims keepdim).rank) :
    s.keptIndex dims keepdim (s.reduceAxes dims keepdim j) = j.val := by
  cases keepdim with
  | false => exact (s.nodup_otherAxes dims).idxOf_getElem _ _
  | true => rfl

/-- The position of an unlisted axis is an output axis of the reduction. -/
private theorem keptIndex_lt {a : Fin s.rank} (ha : a ∉ dims) :
    s.keptIndex dims keepdim a < (s.reduce dims keepdim).rank := by
  cases keepdim with
  | false =>
    have h := List.idxOf_lt_length_iff.mpr (Shape.mem_otherAxes.mpr ha)
    simpa [Shape.reduce, Shape.eraseAxes, Shape.rank_eq_length, Shape.keptIndex] using h
  | true => simpa [Shape.reduce, Shape.keepAxes, Shape.rank_eq_length, Shape.keptIndex] using a.isLt

/-- The output axis at the position of an unlisted axis is placed at that axis. -/
private theorem reduceAxes_keptIndex {a : Fin s.rank} (ha : a ∉ dims) :
    s.reduceAxes dims keepdim ⟨s.keptIndex dims keepdim a, keptIndex_lt ha⟩ = a := by
  cases keepdim with
  | false => exact List.getElem_idxOf _
  | true => rfl

/-- The output axis at the position of an unlisted axis has the extent of that axis. -/
private theorem extent_reduce_keptIndex {a : Fin s.rank} (ha : a ∉ dims) :
    (s.reduce dims keepdim).extent ⟨s.keptIndex dims keepdim a, keptIndex_lt ha⟩ =
      s.extent a := by
  cases keepdim with
  | false =>
    exact (s.extent_keptAxis dims _).trans (congrArg s.extent (List.getElem_idxOf _))
  | true =>
    show (s.keepAxes dims).extent ⟨a.val, _⟩ = s.extent a
    simp only [Shape.extent, Shape.keepAxes, List.getElem_ofFn]
    simp [ha]

/-- The coordinate of the selected shape that `c` lies over, with the indices of `c` along the
listed axes. -/
def listedCoord (dims : List (Fin s.rank)) (c : s.Coord) : (s.selectAxes dims).Coord :=
  Shape.Coord.ofFun fun i => Fin.cast (s.extent_listedAxis dims i).symm
    (Shape.Coord.get c (s.listedAxis dims i))

/-- The indices of `listedCoord dims c` are those of `c` along the listed axes. -/
private theorem index_listedCoord (dims : List (Fin s.rank)) (c : s.Coord) (k : Nat) :
    Shape.Coord.index (listedCoord dims c) k =
      if h : k < (s.selectAxes dims).rank then
        Shape.Coord.index c (s.listedAxis dims ⟨k, h⟩).val
      else 0 := by
  rw [listedCoord, Shape.Coord.index_ofFun]
  split_ifs
  · simp [Shape.Coord.val_get]
  · rfl

/-- A coordinate read by a reduction broadcasts from the output coordinate it is read for. -/
private theorem broadcastSource_reduceSource (o : (s.reduce dims keepdim).Coord)
    (r : (s.selectAxes dims).Coord) :
    broadcastSource (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim)
      (s.reduceSource dims keepdim o r) = o := by
  apply Shape.Coord.ext_index
  intro k hk
  rw [index_broadcastSource_of_lt _ _ _ hk]
  split_ifs with h1
  · have h2 := Shape.Coord.index_lt o hk
    omega
  · have hnot : s.reduceAxes dims keepdim ⟨k, hk⟩ ∉ dims := fun hm =>
      h1 (extent_reduce_of_mem _ hm)
    rw [Shape.index_reduceSource, dite_eq_left (Fin.isLt _)]
    simp only [Fin.eta, hnot, ↓reduceIte, keptIndex_reduceAxes]

/-- A coordinate read by a reduction lies over the reduced coordinate it is read at. -/
private theorem listedCoord_reduceSource (o : (s.reduce dims keepdim).Coord)
    (r : (s.selectAxes dims).Coord) :
    listedCoord dims (s.reduceSource dims keepdim o r) = r := by
  apply Shape.Coord.ext_index
  intro k hk
  have hlen : k < (s.sortedAxes dims).length := by
    simpa [Shape.selectAxes, Shape.rank_eq_length] using hk
  have hmem : (s.sortedAxes dims)[k] ∈ dims :=
    Shape.mem_sortedAxes.mp (List.getElem_mem hlen)
  have hnd : (s.sortedAxes dims).Nodup := (List.nodup_finRange s.rank).filter _
  rw [index_listedCoord, dite_eq_left hk, Shape.index_reduceSource, dite_eq_left (Fin.isLt _)]
  simp only [Shape.listedAxis, Fin.eta, hmem, ↓reduceIte, hnd.idxOf_getElem k hlen]

/-- A coordinate is read by a reduction at the coordinate it broadcasts from and the reduced
coordinate it lies over. -/
private theorem reduceSource_broadcastSource_listedCoord (c : s.Coord) :
    s.reduceSource dims keepdim
      (broadcastSource (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c)
      (listedCoord dims c) = c := by
  apply Shape.Coord.ext_index
  intro k hk
  rw [Shape.index_reduceSource, dite_eq_left hk]
  split_ifs with hd
  · have hmem : (⟨k, hk⟩ : Fin s.rank) ∈ s.sortedAxes dims := Shape.mem_sortedAxes.mpr hd
    have hlt : (s.sortedAxes dims).idxOf ⟨k, hk⟩ < (s.selectAxes dims).rank := by
      simpa [Shape.selectAxes, Shape.rank_eq_length] using List.idxOf_lt_length_iff.mpr hmem
    rw [index_listedCoord, dite_eq_left hlt]
    simp only [Shape.listedAxis, List.getElem_idxOf]
  · rw [index_broadcastSource_of_lt _ _ _ (keptIndex_lt hd), reduceAxes_keptIndex hd,
      extent_reduce_keptIndex hd]
    split_ifs with h1
    · have h2 := Shape.Coord.index_lt c hk
      omega
    · rfl

/-- Over a commutative additive monoid, an entry of a sum over `dims` is the finite sum of the
entries of its fiber. -/
theorem reduce_sum_apply_eq_sum {α : Type} [Storage α] [AddCommMonoid α] (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool) (o : (s.reduce dims keepdim).Coord) :
    (reduce .sum x dims keepdim) o = ∑ r, x (s.reduceSource dims keepdim o r) := by
  have hfold : ∀ {n : Nat} (f : Fin n → α),
      (List.finRange n).foldl (fun acc i => acc + f i) 0 = ∑ i, f i := fun f => by
    rw [Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]
  rw [reduce_sum_apply, hfold]
  exact (Shape.Coord.equivFin _).symm.sum_comp (fun r => x (s.reduceSource dims keepdim o r))

/-- Over a commutative additive monoid, a sum over `dims` is `reduceBroadcast` along the axes the
output broadcasts back to. -/
theorem reduce_sum_eq_reduceBroadcast {α : Type} [Storage α] [AddCommMonoid α] (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool) :
    reduce .sum x dims keepdim =
      reduceBroadcast (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) x := by
  refine Rep.ext fun o => ?_
  rw [reduce_sum_apply_eq_sum, reduceBroadcast_apply]
  refine Fintype.sum_equiv
    { toFun := fun r => ⟨s.reduceSource dims keepdim o r, broadcastSource_reduceSource o r⟩
      invFun := fun c => listedCoord dims c.1
      left_inv := fun r => listedCoord_reduceSource o r
      right_inv := fun c => Subtype.ext ?_ } _ _ fun r => rfl
  have h3 := reduceSource_broadcastSource_listedCoord (dims := dims) (keepdim := keepdim) c.1
  rw [c.2] at h3
  exact h3

/-- A left fold from one that multiplies by `g i` at each `i` outside `p` is the product of `g`
over the indices outside `p`. -/
private theorem foldl_mul_ite_eq_prod {β : Type} [CommMonoid β] {n : Nat} (p : Fin n → Prop)
    [DecidablePred p] (g : Fin n → β) :
    (List.finRange n).foldl (fun acc i => if p i then acc else acc * g i) 1 =
      ∏ i, if p i then 1 else g i := by
  rw [Fin.prod_univ_def, List.prod_eq_foldl, List.foldl_map]
  exact congrArg (fun step => List.foldl step 1 (List.finRange n))
    (funext fun acc => funext fun i => by rw [mul_ite, mul_one])

/-- Over a commutative monoid, an entry of a product over `dims` is the finite product of the
entries of its fiber. -/
theorem reduce_prod_apply_eq_prod {α : Type} [Storage α] [CommMonoid α] (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool) (o : (s.reduce dims keepdim).Coord) :
    (reduce .prod x dims keepdim) o = ∏ r, x (s.reduceSource dims keepdim o r) := by
  have hfold : ∀ {n : Nat} (f : Fin n → α),
      (List.finRange n).foldl (fun acc i => acc * f i) 1 = ∏ i, f i := fun f => by
    rw [Fin.prod_univ_def, List.prod_eq_foldl, List.foldl_map]
  rw [reduce_prod_apply, hfold]
  exact (Shape.Coord.equivFin _).symm.prod_comp (fun r => x (s.reduceSource dims keepdim o r))

/-- The coordinates of the fiber of `c` other than `c` are read at reduced coordinates other than
`listedCoord dims c`. -/
private theorem prod_fiber_ite_eq {β : Type} [CommMonoid β] (c : s.Coord)
    (g : (s.selectAxes dims).Coord → β) :
    (∏ r, if s.reduceSource dims keepdim
        (broadcastSource (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c) r = c
      then 1 else g r) = ∏ r ∈ Finset.univ.erase (listedCoord dims c), g r := by
  rw [← Finset.prod_erase_mul _ _ (Finset.mem_univ (listedCoord dims c)),
    ite_eq_left (reduceSource_broadcastSource_listedCoord (dims := dims) (keepdim := keepdim) c),
    mul_one]
  refine Finset.prod_congr rfl fun r hr => ite_eq_right fun hc => Finset.ne_of_mem_erase hr ?_
  exact (listedCoord_reduceSource _ r).symm.trans (congrArg (listedCoord dims) hc)

end Fibers

/-! ## Sums and means -/

section Sum

variable (dims : List (Fin s.rank)) (keepdim : Bool)

/-- The real sum over `dims`, as a continuous linear map. -/
noncomputable def reduceSumCLM : Tensor ℝ s →L[ℝ] Tensor ℝ (s.reduce dims keepdim) :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => reduce .sum x dims keepdim
      map_add' := fun x y => Rep.ext fun o => by
        simp only [reduce_sum_apply_eq_sum, Tensor.add_apply, Finset.sum_add_distrib]
      map_smul' := fun a x => Rep.ext fun o => by
        simp only [reduce_sum_apply_eq_sum, Rep.smul_apply, RingHom.id_apply, smul_eq_mul,
          Finset.mul_sum] }

/-- `reduceSumCLM dims keepdim` applied to `x` is `reduce .sum x dims keepdim`. -/
@[simp] theorem reduceSumCLM_apply (x : Tensor ℝ s) :
    reduceSumCLM dims keepdim x = reduce .sum x dims keepdim :=
  rfl

/-- Over the reals, the adjoint of a sum over `dims` is the broadcast back along
`Shape.reduceAxes`. -/
theorem inner_reduce_sum (u : Tensor ℝ s) (δ : Tensor ℝ (s.reduce dims keepdim)) :
    ⟪reduce .sum u dims keepdim, δ⟫ =
      ⟪u, broadcast δ s (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim)⟫ := by
  rw [reduce_sum_eq_reduceBroadcast, real_inner_comm, ← inner_broadcast]
  exact real_inner_comm _ _

/-- Over the reals, a mean over `dims` is the sum over `dims` scaled by the inverse of
`meanDenominator (s.selectAxes dims)`. -/
theorem mean_eq_smul (x : Tensor ℝ s) :
    mean x dims keepdim =
      ((meanDenominator (s.selectAxes dims) : ℕ) : ℝ)⁻¹ • reduce .sum x dims keepdim := by
  refine Rep.ext fun o => ?_
  simp only [mean, Tensor.map_apply, Context.divNat_real, Rep.smul_apply, smul_eq_mul,
    meanDenominator]
  rw [div_eq_inv_mul]

/-- Over the reals, the stored pullback of a mean is the broadcast back scaled by the inverse of
`meanDenominator (s.selectAxes dims)`. -/
theorem meanBackward_eq_smul (δ : Tensor ℝ (s.reduce dims keepdim)) :
    meanBackward dims keepdim δ =
      ((meanDenominator (s.selectAxes dims) : ℕ) : ℝ)⁻¹ •
        broadcast δ s (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) := by
  refine Rep.ext fun c => ?_
  simp only [meanBackward, Tensor.map_apply, Context.divNat_real, Rep.smul_apply,
    smul_eq_mul, meanDenominator]
  rw [div_eq_inv_mul]

/-- The real mean over `dims`, as a continuous linear map. -/
noncomputable def meanCLM : Tensor ℝ s →L[ℝ] Tensor ℝ (s.reduce dims keepdim) :=
  ((meanDenominator (s.selectAxes dims) : ℕ) : ℝ)⁻¹ • reduceSumCLM dims keepdim

/-- `meanCLM dims keepdim` applied to `x` is `mean x dims keepdim`. -/
@[simp] theorem meanCLM_apply (x : Tensor ℝ s) : meanCLM dims keepdim x = mean x dims keepdim :=
  (mean_eq_smul dims keepdim x).symm

/-- Over the reals, the adjoint of a mean over `dims` is `meanBackward`. -/
theorem inner_mean (u : Tensor ℝ s) (δ : Tensor ℝ (s.reduce dims keepdim)) :
    ⟪mean u dims keepdim, δ⟫ = ⟪u, meanBackward dims keepdim δ⟫ := by
  rw [mean_eq_smul, meanBackward_eq_smul, real_inner_smul_left, real_inner_smul_right,
    inner_reduce_sum]

/-- Over the reals, the adjoint of the sum over `dims` of a tangent scaled entrywise by `w` scales
the broadcast back entrywise by `w`. -/
theorem inner_reduce_sum_slopeCLM (w : s.Coord → ℝ) (u : Tensor ℝ s)
    (δ : Tensor ℝ (s.reduce dims keepdim)) :
    ⟪reduce .sum (Op.slopeCLM w u) dims keepdim, δ⟫ =
      ⟪u, Op.slopeCLM w (broadcast δ s (s.reduceAxes dims keepdim)
        (s.reduce_broadcastDims dims keepdim))⟫ := by
  rw [inner_reduce_sum]
  simp only [inner_eq_sum, Op.slopeCLM_apply]
  exact Finset.sum_congr rfl fun c _ => by ring

end Sum

/-! ## Products -/

section Prod

variable (dims : List (Fin s.rank)) (keepdim : Bool)

/-- The partial derivative of a real product over `dims` at `x` in the entry `c`, the product of
the other entries of the fiber of `c`. -/
noncomputable def reduceProdSlope (x : Tensor ℝ s) (c : s.Coord) : ℝ :=
  ∏ r ∈ Finset.univ.erase (listedCoord dims c),
    x (s.reduceSource dims keepdim
      (broadcastSource (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c) r)

/-- At the entry read at `o` and `r`, `reduceProdSlope` is the product of the entries of the fiber
of `o` read at the reduced coordinates other than `r`. -/
theorem reduceProdSlope_reduceSource (x : Tensor ℝ s) (o : (s.reduce dims keepdim).Coord)
    (r : (s.selectAxes dims).Coord) :
    reduceProdSlope dims keepdim x (s.reduceSource dims keepdim o r) =
      ∏ r' ∈ Finset.univ.erase r, x (s.reduceSource dims keepdim o r') := by
  rw [reduceProdSlope, listedCoord_reduceSource, broadcastSource_reduceSource]

/-- The derivative of the real product over `dims` at `x`: the sum over `dims` of the tangent
scaled entrywise by `reduceProdSlope`. -/
noncomputable def reduceProdCLM (x : Tensor ℝ s) :
    Tensor ℝ s →L[ℝ] Tensor ℝ (s.reduce dims keepdim) :=
  (reduceSumCLM dims keepdim).comp (Op.slopeCLM (reduceProdSlope dims keepdim x))

/-- `reduceProdCLM dims keepdim x` applied to `v` is the sum over `dims` of `v` scaled entrywise by
`reduceProdSlope`. -/
@[simp] theorem reduceProdCLM_apply (x v : Tensor ℝ s) :
    reduceProdCLM dims keepdim x v =
      reduce .sum (Op.slopeCLM (reduceProdSlope dims keepdim x) v) dims keepdim :=
  rfl

/-- The real product over `dims` has derivative `reduceProdCLM dims keepdim x` at every `x`. -/
theorem hasFDerivAt_reduce_prod (x : Tensor ℝ s) :
    HasFDerivAt (fun y : Tensor ℝ s => reduce .prod y dims keepdim)
      (reduceProdCLM dims keepdim x) x := by
  have hc : ∀ o : (s.reduce dims keepdim).Coord,
      HasFDerivAt (fun y : Tensor ℝ s => (reduce .prod y dims keepdim) o)
        (∑ r, (∏ r' ∈ Finset.univ.erase r, x (s.reduceSource dims keepdim o r')) •
          coordinateCLM (s.reduceSource dims keepdim o r)) x := fun o => by
    have h := HasFDerivAt.finsetProd (u := Finset.univ) (x := x)
      fun r _ => (coordinateCLM (s.reduceSource dims keepdim o r)).hasFDerivAt
    simp only [coordinateCLM_apply] at h
    simpa only [reduce_prod_apply_eq_prod] using h
  refine (hasFDerivAt_of_coordinates (fun y : Tensor ℝ s => reduce .prod y dims keepdim) _ x
    hc).congr_fderiv ?_
  refine ContinuousLinearMap.ext fun v => Rep.ext fun o => ?_
  rw [assemble_apply, reduceProdCLM_apply, reduce_sum_apply_eq_sum, _root_.sum_apply]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [_root_.smul_apply, coordinateCLM_apply, smul_eq_mul, Op.slopeCLM_apply,
    reduceProdSlope_reduceSource]

/-- Over the reals, the stored pullback of a product over `dims` at `x` scales the broadcast back
of the gradient entrywise by `reduceProdSlope`. -/
theorem reduceBackward_prod (x : Tensor ℝ s)
    (h : ReduceOp.prod = .max ∨ ReduceOp.prod = .min → ∀ d ∈ dims, 0 < s.extent d)
    (δ : Tensor ℝ (s.reduce dims keepdim)) :
    reduceBackward .prod x dims keepdim h δ =
      Op.slopeCLM (reduceProdSlope dims keepdim x)
        (broadcast δ s (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim)) := by
  refine Rep.ext fun c => ?_
  simp only [reduceBackward, Rep.get_ofFn, Op.slopeCLM_apply]
  rw [mul_comm]
  congr 1
  have hfold := List.foldl_map (f := fun i => s.reduceSource dims keepdim
      (broadcastSource (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c)
      (Shape.Coord.unlinearize i))
    (g := fun acc source => if source = c then acc else acc * x source)
    (l := List.finRange (Shape.size (s.selectAxes dims))) (init := 1)
  refine hfold.trans ((foldl_mul_ite_eq_prod _ _).trans ?_)
  refine ((Shape.Coord.equivFin _).symm.prod_comp (fun r =>
    if s.reduceSource dims keepdim (broadcastSource (s.reduceAxes dims keepdim)
      (s.reduce_broadcastDims dims keepdim) c) r = c then 1
    else x (s.reduceSource dims keepdim (broadcastSource (s.reduceAxes dims keepdim)
      (s.reduce_broadcastDims dims keepdim) c) r))).trans ?_
  exact prod_fiber_ite_eq c _

/-- Over the reals, a product over `dims` has at `x` the derivative `reduceProdCLM` and the stored
pullback `reduceBackward .prod`. -/
theorem hasPullbackAt_reduce_prod (x : Tensor ℝ s) :
    HasPullbackAt (fun y : Tensor ℝ s => reduce .prod y dims keepdim)
      (reduceProdCLM dims keepdim x) (fun δ => reduceBackward .prod x dims keepdim (by simp) δ)
      x where
  hasFDerivAt := hasFDerivAt_reduce_prod dims keepdim x
  inner_eq u δ := by
    rw [reduceProdCLM_apply, inner_reduce_sum_slopeCLM, reduceBackward_prod]

end Prod

/-! ## Maxima and minima -/

section Extremum

variable (dims : List (Fin s.rank)) (keepdim : Bool)

/-- Over the reals, an entry ties an extremum exactly when they are equal. -/
private theorem extremumTie_real (a b : ℝ) : extremumTie a b = true ↔ a = b := by
  simp [extremumTie]

/-- In a fiber whose entry read at `r` differs from all its other entries, exactly one entry ties
the entry read at `r`. -/
private theorem countP_fiber_eq_one (x : Tensor ℝ s) (o : (s.reduce dims keepdim).Coord)
    (r : (s.selectAxes dims).Coord)
    (hr : ∀ r', r' ≠ r →
      x (s.reduceSource dims keepdim o r') ≠ x (s.reduceSource dims keepdim o r)) :
    ((List.finRange (Shape.size (s.selectAxes dims))).map fun i =>
        s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)).countP
      (fun source => extremumTie (x source) (x (s.reduceSource dims keepdim o r))) = 1 := by
  have hunl : Shape.Coord.unlinearize (Shape.Coord.equivFin (s.selectAxes dims) r) = r :=
    (Shape.Coord.equivFin _).symm_apply_apply r
  rw [List.countP_map]
  refine (List.countP_congr fun i _ => ?_).trans (List.count_eq_one_of_mem
    (List.nodup_finRange _) (List.mem_finRange (Shape.Coord.equivFin (s.selectAxes dims) r)))
  show extremumTie (x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)))
      (x (s.reduceSource dims keepdim o r)) = true ↔
    (i == Shape.Coord.equivFin (s.selectAxes dims) r) = true
  rw [extremumTie_real, beq_iff_eq]
  constructor
  · intro he
    by_contra hne
    refine hr _ (fun hi => hne ?_) he
    rw [← hi]
    exact ((Shape.Coord.equivFin _).apply_symm_apply i).symm
  · intro he
    rw [he, hunl]

/-- Over the reals, the entries of the fiber of `o` that tie `v` are as many as the reduced
coordinates `q` at which the fiber of `o` reads `v`. -/
private theorem countP_fiber_eq_card (x : Tensor ℝ s) (o : (s.reduce dims keepdim).Coord)
    (v : ℝ) :
    ((List.finRange (Shape.size (s.selectAxes dims))).map fun i =>
        s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)).countP
      (fun source => extremumTie (x source) v) =
      (Finset.univ.filter fun q => x (s.reduceSource dims keepdim o q) = v).card := by
  rw [List.countP_map]
  refine (List.countP_congr (q := fun i =>
    decide (x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)) = v))
    fun i _ => ?_).trans ?_
  · show extremumTie (x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))) v = true ↔
      decide (x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)) = v) = true
    rw [extremumTie_real, decide_eq_true_iff]
  · rw [← (List.nodup_finRange _).card_eq_countP (P := fun i =>
      x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)) = v), List.toFinset_finRange]
    exact Finset.card_equiv (Shape.Coord.equivFin _).symm fun i => by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      rfl

/-- A left fold of `max` from a value at most `v j`, over a list in which every value is at most
`v j`, is `v j` when `j` is in the list or the initial value is `v j`. -/
private theorem foldl_max_eq {ι : Type} (v : ι → ℝ) (j : ι) (hv : ∀ i, v i ≤ v j) :
    ∀ (l : List ι) (a : ℝ), a ≤ v j → (j ∈ l ∨ a = v j) →
      l.foldl (fun acc i => max acc (v i)) a = v j
  | [], _, _, h => h.resolve_left (by simp)
  | i :: l, a, ha, h => by
    rw [List.foldl_cons]
    refine foldl_max_eq v j hv l _ (max_le ha (hv i)) ?_
    rcases h with h | h
    · rcases List.mem_cons.mp h with h | h
      · exact Or.inr (by rw [← h]; exact max_eq_right ha)
      · exact Or.inl h
    · exact Or.inr (by rw [h]; exact max_eq_left (hv i))

/-- A left fold of `min` from a value at least `v j`, over a list in which every value is at least
`v j`, is `v j` when `j` is in the list or the initial value is `v j`. -/
private theorem foldl_min_eq {ι : Type} (v : ι → ℝ) (j : ι) (hv : ∀ i, v j ≤ v i) :
    ∀ (l : List ι) (a : ℝ), v j ≤ a → (j ∈ l ∨ a = v j) →
      l.foldl (fun acc i => min acc (v i)) a = v j
  | [], _, _, h => h.resolve_left (by simp)
  | i :: l, a, ha, h => by
    rw [List.foldl_cons]
    refine foldl_min_eq v j hv l _ (le_min ha (hv i)) ?_
    rcases h with h | h
    · rcases List.mem_cons.mp h with h | h
      · exact Or.inr (by rw [← h]; exact min_eq_right ha)
      · exact Or.inl h
    · exact Or.inr (by rw [h]; exact min_eq_left (hv i))

/-- Where the entry read at `r` is a largest entry of the fiber of `o`, the real maximum over `dims`
at `o` is that entry. -/
theorem reduce_max_apply_of_le (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) (r : (s.selectAxes dims).Coord)
    (hr : ∀ q, x (s.reduceSource dims keepdim o q) ≤ x (s.reduceSource dims keepdim o r)) :
    (reduce .max x dims keepdim fun _ => hpos) o = x (s.reduceSource dims keepdim o r) := by
  have hunl : Shape.Coord.unlinearize (Shape.Coord.equivFin (s.selectAxes dims) r) = r :=
    (Shape.Coord.equivFin _).symm_apply_apply r
  have hv : ∀ i, x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)) ≤
      x (s.reduceSource dims keepdim o
        (Shape.Coord.unlinearize (Shape.Coord.equivFin (s.selectAxes dims) r))) :=
    fun i => by
      rw [hunl]
      exact hr _
  rw [reduce_max_apply]
  simp only [Context.maximum_real]
  rw [foldl_max_eq (fun i => x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))) _ hv
    _ _ (hv _) (Or.inl (List.mem_finRange _)), hunl]

/-- Where the entry read at `r` is a smallest entry of the fiber of `o`, the real minimum over
`dims` at `o` is that entry. -/
theorem reduce_min_apply_of_le (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) (r : (s.selectAxes dims).Coord)
    (hr : ∀ q, x (s.reduceSource dims keepdim o r) ≤ x (s.reduceSource dims keepdim o q)) :
    (reduce .min x dims keepdim fun _ => hpos) o = x (s.reduceSource dims keepdim o r) := by
  have hunl : Shape.Coord.unlinearize (Shape.Coord.equivFin (s.selectAxes dims) r) = r :=
    (Shape.Coord.equivFin _).symm_apply_apply r
  have hv : ∀ i,
      x (s.reduceSource dims keepdim o
        (Shape.Coord.unlinearize (Shape.Coord.equivFin (s.selectAxes dims) r))) ≤
        x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)) :=
    fun i => by
      rw [hunl]
      exact hr _
  rw [reduce_min_apply]
  simp only [Context.minimum_real]
  rw [foldl_min_eq (fun i => x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))) _ hv
    _ _ (hv _) (Or.inl (List.mem_finRange _)), hunl]

/-- Where the entry read at `r` is larger than every other entry of the fiber of `o`, the real
maximum over `dims` at `o` is that entry. -/
theorem reduce_max_apply_of_lt (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) (r : (s.selectAxes dims).Coord)
    (hr : ∀ r', r' ≠ r →
      x (s.reduceSource dims keepdim o r') < x (s.reduceSource dims keepdim o r)) :
    (reduce .max x dims keepdim fun _ => hpos) o = x (s.reduceSource dims keepdim o r) :=
  reduce_max_apply_of_le dims keepdim x hpos o r fun q => by
    by_cases hq : q = r
    · rw [hq]
    · exact (hr q hq).le

/-- Where the entry read at `r` is smaller than every other entry of the fiber of `o`, the real
minimum over `dims` at `o` is that entry. -/
theorem reduce_min_apply_of_lt (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) (r : (s.selectAxes dims).Coord)
    (hr : ∀ r', r' ≠ r →
      x (s.reduceSource dims keepdim o r) < x (s.reduceSource dims keepdim o r')) :
    (reduce .min x dims keepdim fun _ => hpos) o = x (s.reduceSource dims keepdim o r) :=
  reduce_min_apply_of_le dims keepdim x hpos o r fun q => by
    by_cases hq : q = r
    · rw [hq]
    · exact (hr q hq).le

/-- The weight of the entry `c` in `reduceMaxCLM`, which is the derivative of a real maximum over
`dims` at `x` when every fiber of `x` has a unique largest entry (`hasFDerivAt_reduce_max`): one if
`x c` is larger than every other entry of the fiber of `c`, zero otherwise. -/
noncomputable def reduceMaxSlope (x : Tensor ℝ s) (c : s.Coord) : ℝ :=
  if ∀ r, r ≠ listedCoord dims c → x (s.reduceSource dims keepdim
      (broadcastSource (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c) r) <
      x c then 1 else 0

/-- The weight of the entry `c` in `reduceMinCLM`, which is the derivative of a real minimum over
`dims` at `x` when every fiber of `x` has a unique smallest entry (`hasFDerivAt_reduce_min`): one
if `x c` is smaller than every other entry of the fiber of `c`, zero otherwise. -/
noncomputable def reduceMinSlope (x : Tensor ℝ s) (c : s.Coord) : ℝ :=
  if ∀ r, r ≠ listedCoord dims c → x c < x (s.reduceSource dims keepdim
      (broadcastSource (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c) r)
    then 1 else 0

/-- At the entry read at `o` and `r`, `reduceMaxSlope` is one if that entry is larger than every
other entry of the fiber of `o`, zero otherwise. -/
theorem reduceMaxSlope_reduceSource (x : Tensor ℝ s) (o : (s.reduce dims keepdim).Coord)
    (r : (s.selectAxes dims).Coord) :
    reduceMaxSlope dims keepdim x (s.reduceSource dims keepdim o r) =
      if ∀ r', r' ≠ r →
        x (s.reduceSource dims keepdim o r') < x (s.reduceSource dims keepdim o r)
      then 1 else 0 := by
  rw [reduceMaxSlope, listedCoord_reduceSource, broadcastSource_reduceSource]

/-- At the entry read at `o` and `r`, `reduceMinSlope` is one if that entry is smaller than every
other entry of the fiber of `o`, zero otherwise. -/
theorem reduceMinSlope_reduceSource (x : Tensor ℝ s) (o : (s.reduce dims keepdim).Coord)
    (r : (s.selectAxes dims).Coord) :
    reduceMinSlope dims keepdim x (s.reduceSource dims keepdim o r) =
      if ∀ r', r' ≠ r →
        x (s.reduceSource dims keepdim o r) < x (s.reduceSource dims keepdim o r')
      then 1 else 0 := by
  rw [reduceMinSlope, listedCoord_reduceSource, broadcastSource_reduceSource]

/-- The sum over `dims` of the tangent scaled entrywise by `reduceMaxSlope`, which is the
derivative of the real maximum at inputs whose fibers have unique largest entries
(`hasFDerivAt_reduce_max`). -/
noncomputable def reduceMaxCLM (x : Tensor ℝ s) :
    Tensor ℝ s →L[ℝ] Tensor ℝ (s.reduce dims keepdim) :=
  (reduceSumCLM dims keepdim).comp (Op.slopeCLM (reduceMaxSlope dims keepdim x))

/-- `reduceMaxCLM dims keepdim x` applied to `v` is the sum over `dims` of `v` scaled entrywise by
`reduceMaxSlope`. -/
@[simp] theorem reduceMaxCLM_apply (x v : Tensor ℝ s) :
    reduceMaxCLM dims keepdim x v =
      reduce .sum (Op.slopeCLM (reduceMaxSlope dims keepdim x) v) dims keepdim :=
  rfl

/-- The sum over `dims` of the tangent scaled entrywise by `reduceMinSlope`, which is the
derivative of the real minimum at inputs whose fibers have unique smallest entries
(`hasFDerivAt_reduce_min`). -/
noncomputable def reduceMinCLM (x : Tensor ℝ s) :
    Tensor ℝ s →L[ℝ] Tensor ℝ (s.reduce dims keepdim) :=
  (reduceSumCLM dims keepdim).comp (Op.slopeCLM (reduceMinSlope dims keepdim x))

/-- `reduceMinCLM dims keepdim x` applied to `v` is the sum over `dims` of `v` scaled entrywise by
`reduceMinSlope`. -/
@[simp] theorem reduceMinCLM_apply (x v : Tensor ℝ s) :
    reduceMinCLM dims keepdim x v =
      reduce .sum (Op.slopeCLM (reduceMinSlope dims keepdim x) v) dims keepdim :=
  rfl

/-- At a real input each of whose fibers has an entry larger than all its other entries, the
maximum over `dims` has derivative `reduceMaxCLM dims keepdim x`. -/
theorem hasFDerivAt_reduce_max (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (hx : ∀ o, ∃ r, ∀ r', r' ≠ r →
      x (s.reduceSource dims keepdim o r') < x (s.reduceSource dims keepdim o r)) :
    HasFDerivAt (fun y : Tensor ℝ s => reduce .max y dims keepdim fun _ => hpos)
      (reduceMaxCLM dims keepdim x) x := by
  choose m hm using hx
  have hev : ∀ᶠ y in nhds x, ∀ o r, r ≠ m o →
      y (s.reduceSource dims keepdim o r) < y (s.reduceSource dims keepdim o (m o)) := by
    refine Filter.eventually_all.mpr fun o => Filter.eventually_all.mpr fun r => ?_
    by_cases hr : r = m o
    · exact Filter.Eventually.of_forall fun _ h => absurd hr h
    · exact ((coordinateCLM (s.reduceSource dims keepdim o r)).continuous.continuousAt.eventually_lt
        (coordinateCLM (s.reduceSource dims keepdim o (m o))).continuous.continuousAt
        (hm o r hr)).mono fun _ hy _ => hy
  refine ((Rep.pullCLM fun o =>
    s.reduceSource dims keepdim o (m o)).hasFDerivAt.congr_of_eventuallyEq ?_).congr_fderiv ?_
  · filter_upwards [hev] with y hy
    refine Rep.ext fun o => ?_
    rw [Rep.pullCLM_apply, Rep.pull_apply]
    exact reduce_max_apply_of_lt dims keepdim y hpos o (m o) (hy o)
  · refine ContinuousLinearMap.ext fun v => Rep.ext fun o => ?_
    rw [Rep.pullCLM_apply, Rep.pull_apply, reduceMaxCLM_apply, reduce_sum_apply_eq_sum]
    refine Eq.trans ?_ (Fintype.sum_eq_single (m o) fun r hr => ?_).symm
    · rw [Op.slopeCLM_apply, reduceMaxSlope_reduceSource, ite_eq_left (hm o), one_mul]
    · rw [Op.slopeCLM_apply, reduceMaxSlope_reduceSource,
        ite_eq_right fun hall => lt_asymm (hm o r hr) (hall (m o) (Ne.symm hr)), zero_mul]

/-- At a real input each of whose fibers has an entry smaller than all its other entries, the
minimum over `dims` has derivative `reduceMinCLM dims keepdim x`. -/
theorem hasFDerivAt_reduce_min (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (hx : ∀ o, ∃ r, ∀ r', r' ≠ r →
      x (s.reduceSource dims keepdim o r) < x (s.reduceSource dims keepdim o r')) :
    HasFDerivAt (fun y : Tensor ℝ s => reduce .min y dims keepdim fun _ => hpos)
      (reduceMinCLM dims keepdim x) x := by
  choose m hm using hx
  have hev : ∀ᶠ y in nhds x, ∀ o r, r ≠ m o →
      y (s.reduceSource dims keepdim o (m o)) < y (s.reduceSource dims keepdim o r) := by
    refine Filter.eventually_all.mpr fun o => Filter.eventually_all.mpr fun r => ?_
    by_cases hr : r = m o
    · exact Filter.Eventually.of_forall fun _ h => absurd hr h
    · exact ((coordinateCLM
        (s.reduceSource dims keepdim o (m o))).continuous.continuousAt.eventually_lt
        (coordinateCLM (s.reduceSource dims keepdim o r)).continuous.continuousAt
        (hm o r hr)).mono fun _ hy _ => hy
  refine ((Rep.pullCLM fun o =>
    s.reduceSource dims keepdim o (m o)).hasFDerivAt.congr_of_eventuallyEq ?_).congr_fderiv ?_
  · filter_upwards [hev] with y hy
    refine Rep.ext fun o => ?_
    rw [Rep.pullCLM_apply, Rep.pull_apply]
    exact reduce_min_apply_of_lt dims keepdim y hpos o (m o) (hy o)
  · refine ContinuousLinearMap.ext fun v => Rep.ext fun o => ?_
    rw [Rep.pullCLM_apply, Rep.pull_apply, reduceMinCLM_apply, reduce_sum_apply_eq_sum]
    refine Eq.trans ?_ (Fintype.sum_eq_single (m o) fun r hr => ?_).symm
    · rw [Op.slopeCLM_apply, reduceMinSlope_reduceSource, ite_eq_left (hm o), one_mul]
    · rw [Op.slopeCLM_apply, reduceMinSlope_reduceSource,
        ite_eq_right fun hall => lt_asymm (hm o r hr) (hall (m o) (Ne.symm hr)), zero_mul]

/-- Moving an input along the tensor that is one at the entry read at `o` and `r` and zero
elsewhere moves only that entry of the fiber of `o`. -/
private theorem exists_fiber_line (x : Tensor ℝ s) (o : (s.reduce dims keepdim).Coord)
    (r : (s.selectAxes dims).Coord) :
    ∃ e : Tensor ℝ s, ∀ (t : ℝ) q, (x + t • e) (s.reduceSource dims keepdim o q) =
      x (s.reduceSource dims keepdim o q) + if q = r then t else 0 := by
  refine ⟨(coordinateEquiv s).symm fun c => if c = s.reduceSource dims keepdim o r then 1 else 0,
    fun t q => ?_⟩
  rw [← coordinateCLM_apply (s.reduceSource dims keepdim o q) (x + t • _), map_add, map_smul,
    coordinateCLM_apply, coordinateCLM_apply, coordinateEquiv_symm_apply, smul_eq_mul]
  by_cases hq : q = r
  · rw [ite_eq_left (congrArg (s.reduceSource dims keepdim o) hq), ite_eq_left hq, mul_one]
  · rw [ite_eq_right fun h => hq ((listedCoord_reduceSource o q).symm.trans
      ((congrArg (listedCoord dims) h).trans (listedCoord_reduceSource o r))), ite_eq_right hq,
      mul_zero]

/-- A real maximum over `dims` is not differentiable at an input with a fiber whose largest entry,
read at `r`, is also read at another `r'`. -/
theorem not_differentiableAt_reduce_max (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) {r r' : (s.selectAxes dims).Coord} (hne : r' ≠ r)
    (hle : ∀ q, x (s.reduceSource dims keepdim o q) ≤ x (s.reduceSource dims keepdim o r))
    (heq : x (s.reduceSource dims keepdim o r') = x (s.reduceSource dims keepdim o r)) :
    ¬ DifferentiableAt ℝ (fun y : Tensor ℝ s => reduce .max y dims keepdim fun _ => hpos) x := by
  intro hd
  obtain ⟨e, hy⟩ := exists_fiber_line dims keepdim x o r
  have hline : ∀ t : ℝ, (reduce .max (x + t • e) dims keepdim fun _ => hpos) o =
      x (s.reduceSource dims keepdim o r) + max t 0 := fun t => by
    rcases le_total 0 t with ht | ht
    · have hm : ∀ q, (x + t • e) (s.reduceSource dims keepdim o q) ≤
          (x + t • e) (s.reduceSource dims keepdim o r) := fun q => by
        rw [hy, hy, ite_eq_left rfl]
        by_cases hq : q = r
        · exact le_of_eq (by rw [ite_eq_left hq, hq])
        · rw [ite_eq_right hq, add_zero]
          linarith [hle q]
      rw [reduce_max_apply_of_le dims keepdim _ hpos o r hm, hy, ite_eq_left rfl, max_eq_left ht]
    · have hm : ∀ q, (x + t • e) (s.reduceSource dims keepdim o q) ≤
          (x + t • e) (s.reduceSource dims keepdim o r') := fun q => by
        rw [hy, hy, ite_eq_right hne, add_zero, heq]
        by_cases hq : q = r
        · rw [ite_eq_left hq, hq]
          linarith
        · rw [ite_eq_right hq, add_zero]
          exact hle q
      rw [reduce_max_apply_of_le dims keepdim _ hpos o r' hm, hy, ite_eq_right hne, add_zero, heq,
        max_eq_right ht, add_zero]
  have hd0 : DifferentiableAt ℝ (fun y : Tensor ℝ s => reduce .max y dims keepdim fun _ => hpos)
      (x + (0 : ℝ) • e) := by
    rwa [zero_smul, add_zero]
  have hl : DifferentiableAt ℝ (fun t : ℝ => x + t • e) 0 := by fun_prop
  have hg : DifferentiableAt ℝ (fun t : ℝ => x (s.reduceSource dims keepdim o r) + max t 0) 0 :=
    (DifferentiableAt.fun_comp' (0 : ℝ) (coordinateCLM o).differentiableAt
      (DifferentiableAt.fun_comp' (f := fun t : ℝ => x + t • e) (0 : ℝ) hd0
        hl)).congr_of_eventuallyEq (Filter.Eventually.of_forall fun t => (hline t).symm)
  refine not_differentiableAt_abs_zero ?_
  have habs : (_root_.abs : ℝ → ℝ) = fun t =>
      2 * (x (s.reduceSource dims keepdim o r) + max t 0) -
        2 * x (s.reduceSource dims keepdim o r) - t := funext fun t => by
    rcases le_total 0 t with ht | ht
    · rw [abs_of_nonneg ht, max_eq_left ht]
      ring
    · rw [abs_of_nonpos ht, max_eq_right ht]
      ring
  rw [habs]
  exact ((hg.const_mul 2).sub_const _).sub differentiableAt_id

/-- A real minimum over `dims` is not differentiable at an input with a fiber whose smallest
entry, read at `r`, is also read at another `r'`. -/
theorem not_differentiableAt_reduce_min (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) {r r' : (s.selectAxes dims).Coord} (hne : r' ≠ r)
    (hle : ∀ q, x (s.reduceSource dims keepdim o r) ≤ x (s.reduceSource dims keepdim o q))
    (heq : x (s.reduceSource dims keepdim o r') = x (s.reduceSource dims keepdim o r)) :
    ¬ DifferentiableAt ℝ (fun y : Tensor ℝ s => reduce .min y dims keepdim fun _ => hpos) x := by
  intro hd
  obtain ⟨e, hy⟩ := exists_fiber_line dims keepdim x o r
  have hline : ∀ t : ℝ, (reduce .min (x + t • e) dims keepdim fun _ => hpos) o =
      x (s.reduceSource dims keepdim o r) + min t 0 := fun t => by
    rcases le_total 0 t with ht | ht
    · have hm : ∀ q, (x + t • e) (s.reduceSource dims keepdim o r') ≤
          (x + t • e) (s.reduceSource dims keepdim o q) := fun q => by
        rw [hy, hy, ite_eq_right hne, add_zero, heq]
        by_cases hq : q = r
        · rw [ite_eq_left hq, hq]
          linarith
        · rw [ite_eq_right hq, add_zero]
          exact hle q
      rw [reduce_min_apply_of_le dims keepdim _ hpos o r' hm, hy, ite_eq_right hne, add_zero, heq,
        min_eq_right ht, add_zero]
    · have hm : ∀ q, (x + t • e) (s.reduceSource dims keepdim o r) ≤
          (x + t • e) (s.reduceSource dims keepdim o q) := fun q => by
        rw [hy, hy, ite_eq_left rfl]
        by_cases hq : q = r
        · exact le_of_eq (by rw [ite_eq_left hq, hq])
        · rw [ite_eq_right hq, add_zero]
          linarith [hle q]
      rw [reduce_min_apply_of_le dims keepdim _ hpos o r hm, hy, ite_eq_left rfl, min_eq_left ht]
  have hd0 : DifferentiableAt ℝ (fun y : Tensor ℝ s => reduce .min y dims keepdim fun _ => hpos)
      (x + (0 : ℝ) • e) := by
    rwa [zero_smul, add_zero]
  have hl : DifferentiableAt ℝ (fun t : ℝ => x + t • e) 0 := by fun_prop
  have hg : DifferentiableAt ℝ (fun t : ℝ => x (s.reduceSource dims keepdim o r) + min t 0) 0 :=
    (DifferentiableAt.fun_comp' (0 : ℝ) (coordinateCLM o).differentiableAt
      (DifferentiableAt.fun_comp' (f := fun t : ℝ => x + t • e) (0 : ℝ) hd0
        hl)).congr_of_eventuallyEq (Filter.Eventually.of_forall fun t => (hline t).symm)
  refine not_differentiableAt_abs_zero ?_
  have habs : (_root_.abs : ℝ → ℝ) = fun t =>
      2 * x (s.reduceSource dims keepdim o r) -
        2 * (x (s.reduceSource dims keepdim o r) + min t 0) + t := funext fun t => by
    rcases le_total 0 t with ht | ht
    · rw [abs_of_nonneg ht, min_eq_right ht]
      ring
    · rw [abs_of_nonpos ht, min_eq_left ht]
      ring
  rw [habs]
  exact ((hg.const_mul 2).const_sub _).add differentiableAt_id

/-- A real maximum over `dims` is differentiable at `x` exactly when every fiber of `x` has an
entry larger than all its other entries. -/
theorem differentiableAt_reduce_max_iff (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => reduce .max y dims keepdim fun _ => hpos) x ↔
      ∀ o, ∃ r, ∀ r', r' ≠ r →
        x (s.reduceSource dims keepdim o r') < x (s.reduceSource dims keepdim o r) := by
  refine ⟨fun hd o => ?_, fun hx =>
    (hasFDerivAt_reduce_max dims keepdim x hpos hx).differentiableAt⟩
  obtain ⟨r, -, hr⟩ := Finset.exists_max_image Finset.univ
    (fun q => x (s.reduceSource dims keepdim o q))
    ⟨Shape.Coord.unlinearize ⟨0, Shape.size_selectAxes_pos hpos⟩, Finset.mem_univ _⟩
  exact ⟨r, fun r' hr' => (hr r' (Finset.mem_univ _)).lt_of_ne fun heq =>
    not_differentiableAt_reduce_max dims keepdim x hpos o hr' (fun q => hr q (Finset.mem_univ _))
      heq hd⟩

/-- A real minimum over `dims` is differentiable at `x` exactly when every fiber of `x` has an
entry smaller than all its other entries. -/
theorem differentiableAt_reduce_min_iff (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => reduce .min y dims keepdim fun _ => hpos) x ↔
      ∀ o, ∃ r, ∀ r', r' ≠ r →
        x (s.reduceSource dims keepdim o r) < x (s.reduceSource dims keepdim o r') := by
  refine ⟨fun hd o => ?_, fun hx =>
    (hasFDerivAt_reduce_min dims keepdim x hpos hx).differentiableAt⟩
  obtain ⟨r, -, hr⟩ := Finset.exists_min_image Finset.univ
    (fun q => x (s.reduceSource dims keepdim o q))
    ⟨Shape.Coord.unlinearize ⟨0, Shape.size_selectAxes_pos hpos⟩, Finset.mem_univ _⟩
  exact ⟨r, fun r' hr' => (hr r' (Finset.mem_univ _)).lt_of_ne' fun heq =>
    not_differentiableAt_reduce_min dims keepdim x hpos o hr' (fun q => hr q (Finset.mem_univ _))
      heq hd⟩

/-- An entry of the stored pullback of a real maximum, with the fiber read through
`broadcastSource`. -/
private theorem reduceBackward_max_apply (x : Tensor ℝ s)
    (h : ReduceOp.max = .max ∨ ReduceOp.max = .min → ∀ d ∈ dims, 0 < s.extent d)
    (δ : Tensor ℝ (s.reduce dims keepdim)) (c : s.Coord) :
    reduceBackward .max x dims keepdim h δ c =
      if extremumTie (x c) (broadcast (reduce .max x dims keepdim fun _ => h (Or.inl rfl)) s
          (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c) then
        Context.divNat
          (broadcast δ s (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c)
          (((List.finRange (Shape.size (s.selectAxes dims))).map fun i =>
              s.reduceSource dims keepdim (broadcastSource (s.reduceAxes dims keepdim)
                (s.reduce_broadcastDims dims keepdim) c) (Shape.Coord.unlinearize i)).countP
            fun source => extremumTie (x source)
              (broadcast (reduce .max x dims keepdim fun _ => h (Or.inl rfl)) s
                (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c))
      else 0 :=
  Rep.get_ofFn _ c

/-- An entry of the stored pullback of a real minimum, with the fiber read through
`broadcastSource`. -/
private theorem reduceBackward_min_apply (x : Tensor ℝ s)
    (h : ReduceOp.min = .max ∨ ReduceOp.min = .min → ∀ d ∈ dims, 0 < s.extent d)
    (δ : Tensor ℝ (s.reduce dims keepdim)) (c : s.Coord) :
    reduceBackward .min x dims keepdim h δ c =
      if extremumTie (x c) (broadcast (reduce .min x dims keepdim fun _ => h (Or.inr rfl)) s
          (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c) then
        Context.divNat
          (broadcast δ s (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c)
          (((List.finRange (Shape.size (s.selectAxes dims))).map fun i =>
              s.reduceSource dims keepdim (broadcastSource (s.reduceAxes dims keepdim)
                (s.reduce_broadcastDims dims keepdim) c) (Shape.Coord.unlinearize i)).countP
            fun source => extremumTie (x source)
              (broadcast (reduce .min x dims keepdim fun _ => h (Or.inr rfl)) s
                (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim) c))
      else 0 :=
  Rep.get_ofFn _ c

/-- At every real input, the stored pullback of a maximum over `dims` gives the entry read at `o`
and `r` the gradient at `o` divided by the number of entries of the fiber of `o` equal to it when it
is a largest entry of that fiber, and zero otherwise. -/
theorem reduceBackward_max_reduceSource (x : Tensor ℝ s)
    (h : ReduceOp.max = .max ∨ ReduceOp.max = .min → ∀ d ∈ dims, 0 < s.extent d)
    (δ : Tensor ℝ (s.reduce dims keepdim)) (o : (s.reduce dims keepdim).Coord)
    (r : (s.selectAxes dims).Coord) :
    reduceBackward .max x dims keepdim h δ (s.reduceSource dims keepdim o r) =
      if ∀ q, x (s.reduceSource dims keepdim o q) ≤ x (s.reduceSource dims keepdim o r) then
        δ o / (Finset.univ.filter fun q =>
          x (s.reduceSource dims keepdim o q) = x (s.reduceSource dims keepdim o r)).card
      else 0 := by
  rw [reduceBackward_max_apply]
  simp only [broadcast_eq_pull, Rep.pull_apply, broadcastSource_reduceSource]
  obtain ⟨m, -, hm⟩ := Finset.exists_max_image Finset.univ
    (fun q => x (s.reduceSource dims keepdim o q))
    ⟨Shape.Coord.unlinearize ⟨0, Shape.size_selectAxes_pos (h (Or.inl rfl))⟩, Finset.mem_univ _⟩
  rw [reduce_max_apply_of_le dims keepdim x (h (Or.inl rfl)) o m
    (fun q => hm q (Finset.mem_univ _)), countP_fiber_eq_card dims keepdim x o,
    Context.divNat_real]
  by_cases hr : ∀ q, x (s.reduceSource dims keepdim o q) ≤ x (s.reduceSource dims keepdim o r)
  · have he : x (s.reduceSource dims keepdim o r) = x (s.reduceSource dims keepdim o m) :=
      le_antisymm (hm r (Finset.mem_univ _)) (hr m)
    rw [ite_eq_left ((extremumTie_real _ _).mpr he), ite_eq_left hr, he]
  · have hne : ¬extremumTie (x (s.reduceSource dims keepdim o r))
        (x (s.reduceSource dims keepdim o m)) = true := fun ht =>
      hr fun q => (hm q (Finset.mem_univ _)).trans_eq ((extremumTie_real _ _).mp ht).symm
    rw [ite_eq_right hne, ite_eq_right hr]

/-- At every real input, the stored pullback of a minimum over `dims` gives the entry read at `o`
and `r` the gradient at `o` divided by the number of entries of the fiber of `o` equal to it when it
is a smallest entry of that fiber, and zero otherwise. -/
theorem reduceBackward_min_reduceSource (x : Tensor ℝ s)
    (h : ReduceOp.min = .max ∨ ReduceOp.min = .min → ∀ d ∈ dims, 0 < s.extent d)
    (δ : Tensor ℝ (s.reduce dims keepdim)) (o : (s.reduce dims keepdim).Coord)
    (r : (s.selectAxes dims).Coord) :
    reduceBackward .min x dims keepdim h δ (s.reduceSource dims keepdim o r) =
      if ∀ q, x (s.reduceSource dims keepdim o r) ≤ x (s.reduceSource dims keepdim o q) then
        δ o / (Finset.univ.filter fun q =>
          x (s.reduceSource dims keepdim o q) = x (s.reduceSource dims keepdim o r)).card
      else 0 := by
  rw [reduceBackward_min_apply]
  simp only [broadcast_eq_pull, Rep.pull_apply, broadcastSource_reduceSource]
  obtain ⟨m, -, hm⟩ := Finset.exists_min_image Finset.univ
    (fun q => x (s.reduceSource dims keepdim o q))
    ⟨Shape.Coord.unlinearize ⟨0, Shape.size_selectAxes_pos (h (Or.inr rfl))⟩, Finset.mem_univ _⟩
  rw [reduce_min_apply_of_le dims keepdim x (h (Or.inr rfl)) o m
    (fun q => hm q (Finset.mem_univ _)), countP_fiber_eq_card dims keepdim x o,
    Context.divNat_real]
  by_cases hr : ∀ q, x (s.reduceSource dims keepdim o r) ≤ x (s.reduceSource dims keepdim o q)
  · have he : x (s.reduceSource dims keepdim o r) = x (s.reduceSource dims keepdim o m) :=
      le_antisymm (hr m) (hm r (Finset.mem_univ _))
    rw [ite_eq_left ((extremumTie_real _ _).mpr he), ite_eq_left hr, he]
  · have hne : ¬extremumTie (x (s.reduceSource dims keepdim o r))
        (x (s.reduceSource dims keepdim o m)) = true := fun ht =>
      hr fun q => ((extremumTie_real _ _).mp ht).trans_le (hm q (Finset.mem_univ _))
    rw [ite_eq_right hne, ite_eq_right hr]

/-- At a real input each of whose fibers has an entry larger than all its other entries, the
stored pullback of a maximum over `dims` scales the broadcast back of the gradient entrywise by
`reduceMaxSlope`. -/
theorem reduceBackward_max (x : Tensor ℝ s)
    (h : ReduceOp.max = .max ∨ ReduceOp.max = .min → ∀ d ∈ dims, 0 < s.extent d)
    (hx : ∀ o, ∃ r, ∀ r', r' ≠ r →
      x (s.reduceSource dims keepdim o r') < x (s.reduceSource dims keepdim o r))
    (δ : Tensor ℝ (s.reduce dims keepdim)) :
    reduceBackward .max x dims keepdim h δ =
      Op.slopeCLM (reduceMaxSlope dims keepdim x)
        (broadcast δ s (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim)) := by
  choose m hm using hx
  refine Rep.ext fun c => ?_
  obtain ⟨o, r, rfl⟩ : ∃ o r, s.reduceSource dims keepdim o r = c :=
    ⟨_, _, reduceSource_broadcastSource_listedCoord c⟩
  rw [reduceBackward_max_apply, Op.slopeCLM_apply]
  simp only [broadcast_eq_pull, Rep.pull_apply, broadcastSource_reduceSource]
  rw [reduce_max_apply_of_lt dims keepdim x (h (Or.inl rfl)) o (m o) (hm o),
    reduceMaxSlope_reduceSource]
  by_cases hr : r = m o
  · subst hr
    rw [ite_eq_left ((extremumTie_real _ _).mpr rfl), ite_eq_left (hm o),
      countP_fiber_eq_one dims keepdim x o (m o) fun r' hr' => (hm o r' hr').ne,
      Context.divNat_real, Nat.cast_one, div_one, one_mul]
  · rw [ite_eq_right fun ht => (hm o r hr).ne ((extremumTie_real _ _).mp ht),
      ite_eq_right fun hall => lt_asymm (hm o r hr) (hall (m o) (Ne.symm hr)), zero_mul]

/-- At a real input each of whose fibers has an entry smaller than all its other entries, the
stored pullback of a minimum over `dims` scales the broadcast back of the gradient entrywise by
`reduceMinSlope`. -/
theorem reduceBackward_min (x : Tensor ℝ s)
    (h : ReduceOp.min = .max ∨ ReduceOp.min = .min → ∀ d ∈ dims, 0 < s.extent d)
    (hx : ∀ o, ∃ r, ∀ r', r' ≠ r →
      x (s.reduceSource dims keepdim o r) < x (s.reduceSource dims keepdim o r'))
    (δ : Tensor ℝ (s.reduce dims keepdim)) :
    reduceBackward .min x dims keepdim h δ =
      Op.slopeCLM (reduceMinSlope dims keepdim x)
        (broadcast δ s (s.reduceAxes dims keepdim) (s.reduce_broadcastDims dims keepdim)) := by
  choose m hm using hx
  refine Rep.ext fun c => ?_
  obtain ⟨o, r, rfl⟩ : ∃ o r, s.reduceSource dims keepdim o r = c :=
    ⟨_, _, reduceSource_broadcastSource_listedCoord c⟩
  rw [reduceBackward_min_apply, Op.slopeCLM_apply]
  simp only [broadcast_eq_pull, Rep.pull_apply, broadcastSource_reduceSource]
  rw [reduce_min_apply_of_lt dims keepdim x (h (Or.inr rfl)) o (m o) (hm o),
    reduceMinSlope_reduceSource]
  by_cases hr : r = m o
  · subst hr
    rw [ite_eq_left ((extremumTie_real _ _).mpr rfl), ite_eq_left (hm o),
      countP_fiber_eq_one dims keepdim x o (m o) fun r' hr' => (hm o r' hr').ne',
      Context.divNat_real, Nat.cast_one, div_one, one_mul]
  · rw [ite_eq_right fun ht => (hm o r hr).ne' ((extremumTie_real _ _).mp ht),
      ite_eq_right fun hall => lt_asymm (hm o r hr) (hall (m o) (Ne.symm hr)), zero_mul]

/-- Over the reals, a maximum over `dims` at an input each of whose fibers has an entry larger than
all its other entries has the derivative `reduceMaxCLM` and the stored pullback
`reduceBackward .max`. -/
theorem hasPullbackAt_reduce_max (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (hx : ∀ o, ∃ r, ∀ r', r' ≠ r →
      x (s.reduceSource dims keepdim o r') < x (s.reduceSource dims keepdim o r)) :
    HasPullbackAt (fun y : Tensor ℝ s => reduce .max y dims keepdim fun _ => hpos)
      (reduceMaxCLM dims keepdim x)
      (fun δ => reduceBackward .max x dims keepdim (fun _ => hpos) δ) x where
  hasFDerivAt := hasFDerivAt_reduce_max dims keepdim x hpos hx
  inner_eq u δ := by
    rw [reduceMaxCLM_apply, inner_reduce_sum_slopeCLM, reduceBackward_max dims keepdim x _ hx]

/-- Over the reals, a minimum over `dims` at an input each of whose fibers has an entry smaller
than all its other entries has the derivative `reduceMinCLM` and the stored pullback
`reduceBackward .min`. -/
theorem hasPullbackAt_reduce_min (x : Tensor ℝ s) (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (hx : ∀ o, ∃ r, ∀ r', r' ≠ r →
      x (s.reduceSource dims keepdim o r) < x (s.reduceSource dims keepdim o r')) :
    HasPullbackAt (fun y : Tensor ℝ s => reduce .min y dims keepdim fun _ => hpos)
      (reduceMinCLM dims keepdim x)
      (fun δ => reduceBackward .min x dims keepdim (fun _ => hpos) δ) x where
  hasFDerivAt := hasFDerivAt_reduce_min dims keepdim x hpos hx
  inner_eq u δ := by
    rw [reduceMinCLM_apply, inner_reduce_sum_slopeCLM, reduceBackward_min dims keepdim x _ hx]

end Extremum

/-! ## Cumulative sums -/

section Cumsum

variable {α : Type} [Storage α] [AddCommMonoid α]

/-- A left fold over `Tensor.ofFn f` is the fold of `f` over `List.finRange n`. -/
private theorem foldl_ofFn {β γ : Type} [Storage β] {n : Nat} (step : γ → β → γ) (init : γ)
    (f : Fin n → β) :
    Tensor.foldl step init (Tensor.ofFn f) =
      (List.finRange n).foldl (fun acc i => step acc (f i)) init := by
  rw [Tensor.ofFn, Rep.ofFn_eq_map_coordinates, Rep.foldl_map,
    ← Fin.foldl_eq_foldl_finRange, fin_foldl_eq_coordinates]

/-- Over a commutative additive monoid, an entry of `cumsum x dim` is the sum of the entries of
its fiber along `dim` with index at most its own. -/
theorem cumsum_apply_eq_sum (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    cumsum x dim c = ∑ i, if i ≤ Shape.Coord.get c dim then x (Shape.Coord.update c dim i)
      else 0 := by
  have hfold : ∀ {n : Nat} (f : Fin n → α),
      (List.finRange n).foldl (fun acc i => acc + f i) 0 = ∑ i, f i := fun f => by
    rw [Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]
  rw [cumsum_apply, foldl_ofFn, hfold, ← Finset.sum_filter]
  have hc := (Shape.Coord.get c dim).isLt
  refine Finset.sum_nbij' (fun i => Fin.castLE hc i)
    (fun j => ⟨min j.val (Shape.Coord.get c dim).val, by omega⟩) ?_ ?_ ?_ ?_ ?_
  · intro i _
    have hi := i.isLt
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fin.le_iff_val_le_val,
      Fin.val_castLE]
    omega
  · intro j _
    exact Finset.mem_univ _
  · intro i _
    have hi := i.isLt
    exact Fin.ext (by simp only [Fin.val_castLE]; omega)
  · intro j hj
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fin.le_iff_val_le_val] at hj
    exact Fin.ext (by simp only [Fin.val_castLE]; omega)
  · intro i _
    rfl

omit [AddCommMonoid α] in
/-- Flipping along the single axis `dim` reads the reversed index along `dim`. -/
private theorem flip_singleton_apply (y : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    flip y [dim] c = y (Shape.Coord.update c dim (Shape.Coord.get c dim).rev) := by
  rw [flip_apply]
  congr 1
  apply Shape.Coord.ext
  intro d
  rw [Shape.Coord.get_ofFun]
  by_cases hd : d = dim
  · subst hd
    rw [ite_eq_left (List.mem_singleton_self _), Shape.Coord.get_update_self]
  · rw [ite_eq_right fun h => hd (List.mem_singleton.mp h), Shape.Coord.get_update_of_ne _ _ hd]

/-- Over a commutative additive monoid, an entry of `cumsumBackward g dim` is the sum of the
entries of its fiber along `dim` with index at least its own. -/
theorem cumsumBackward_apply_eq_sum (g : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    cumsumBackward g dim c =
      ∑ j, if Shape.Coord.get c dim ≤ j then g (Shape.Coord.update c dim j) else 0 := by
  rw [cumsumBackward_apply, flip_singleton_apply, cumsum_apply_eq_sum]
  simp only [flip_singleton_apply, Shape.Coord.update_update, Shape.Coord.get_update_self]
  refine Fintype.sum_equiv Fin.revPerm _ _ fun i => ?_
  rw [Fin.revPerm_apply]
  exact if_congr Fin.le_rev_iff rfl rfl

end Cumsum

/-- The real cumulative sum along `dim`, as a continuous linear map. -/
noncomputable def cumsumCLM (dim : Fin s.rank) : Tensor ℝ s →L[ℝ] Tensor ℝ s :=
  LinearMap.toContinuousLinearMap
    { toFun := fun x => cumsum x dim
      map_add' := fun x y => Rep.ext fun c => by
        simp only [cumsum_apply_eq_sum, Tensor.add_apply, ← Finset.sum_add_distrib, ite_add_ite,
          add_zero]
      map_smul' := fun a x => Rep.ext fun c => by
        simp only [cumsum_apply_eq_sum, Rep.smul_apply, RingHom.id_apply, smul_eq_mul,
          Finset.mul_sum, mul_ite, mul_zero] }

/-- `cumsumCLM dim` applied to `x` is `cumsum x dim`. -/
@[simp] theorem cumsumCLM_apply (dim : Fin s.rank) (x : Tensor ℝ s) :
    cumsumCLM dim x = cumsum x dim :=
  rfl

/-- Over the reals, the adjoint of the cumulative sum along `dim` is `cumsumBackward`. -/
theorem inner_cumsum (dim : Fin s.rank) (u δ : Tensor ℝ s) :
    ⟪cumsum u dim, δ⟫ = ⟪u, cumsumBackward δ dim⟫ := by
  rw [inner_eq_sum, inner_eq_sum]
  simp only [cumsum_apply_eq_sum, cumsumBackward_apply_eq_sum, Finset.sum_mul, Finset.mul_sum,
    ite_mul, mul_ite, zero_mul, mul_zero]
  rw [← Fintype.sum_prod_type', ← Fintype.sum_prod_type']
  have hinv : Function.Involutive fun p : s.Coord × Fin (s.extent dim) =>
      (Shape.Coord.update p.1 dim p.2, Shape.Coord.get p.1 dim) := fun p => by
    simp only [Shape.Coord.update_update, Shape.Coord.update_get, Shape.Coord.get_update_self]
  refine Fintype.sum_equiv (hinv.toPerm _) _ _ fun p => ?_
  simp only [Function.Involutive.coe_toPerm, Shape.Coord.update_update, Shape.Coord.update_get,
    Shape.Coord.get_update_self]

end Tensor

namespace HasPullbackAt

open Tensor

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s : Shape} {x : E}
  {f : E → Tensor ℝ s} {Df : E →L[ℝ] Tensor ℝ s} {bf : Tensor ℝ s → E}

/-- A sum over `dims`, pulled back by the broadcast back along `Shape.reduceAxes`. -/
@[fun_prop]
theorem reduce_sum (dims : List (Fin s.rank)) (keepdim : Bool) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.reduce .sum (f y) dims keepdim)
      ((reduceSumCLM dims keepdim).comp Df)
      (fun δ => bf (Tensor.broadcast δ s (s.reduceAxes dims keepdim)
        (s.reduce_broadcastDims dims keepdim))) x := by
  simpa only [reduceSumCLM_apply] using
    clm (reduceSumCLM dims keepdim)
      (fun δ => Tensor.broadcast δ s (s.reduceAxes dims keepdim)
        (s.reduce_broadcastDims dims keepdim))
      (fun u δ => by rw [reduceSumCLM_apply, inner_reduce_sum]) hf

/-- A real product over `dims`, pulled back by the stored `reduceBackward .prod` at `f x`. -/
@[fun_prop]
theorem reduce_prod (dims : List (Fin s.rank)) (keepdim : Bool) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.reduce .prod (f y) dims keepdim)
      ((reduceProdCLM dims keepdim (f x)).comp Df)
      (fun δ => bf (Tensor.reduceBackward .prod (f x) dims keepdim (by simp) δ)) x :=
  (hasPullbackAt_reduce_prod dims keepdim (f x)).comp hf

/-- A real maximum over `dims`, at a point where each fiber of `f x` has an entry larger than all
its other entries, pulled back by the stored `reduceBackward .max` at `f x`. -/
@[fun_prop]
theorem reduce_max (dims : List (Fin s.rank)) (keepdim : Bool)
    (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (hx : ∀ o, ∃ r, ∀ r', r' ≠ r →
      f x (s.reduceSource dims keepdim o r') < f x (s.reduceSource dims keepdim o r))
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.reduce .max (f y) dims keepdim fun _ => hpos)
      ((reduceMaxCLM dims keepdim (f x)).comp Df)
      (fun δ => bf (Tensor.reduceBackward .max (f x) dims keepdim (fun _ => hpos) δ)) x :=
  (hasPullbackAt_reduce_max dims keepdim (f x) hpos hx).comp hf

/-- A real minimum over `dims`, at a point where each fiber of `f x` has an entry smaller than all
its other entries, pulled back by the stored `reduceBackward .min` at `f x`. -/
@[fun_prop]
theorem reduce_min (dims : List (Fin s.rank)) (keepdim : Bool)
    (hpos : ∀ d ∈ dims, 0 < s.extent d)
    (hx : ∀ o, ∃ r, ∀ r', r' ≠ r →
      f x (s.reduceSource dims keepdim o r) < f x (s.reduceSource dims keepdim o r'))
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.reduce .min (f y) dims keepdim fun _ => hpos)
      ((reduceMinCLM dims keepdim (f x)).comp Df)
      (fun δ => bf (Tensor.reduceBackward .min (f x) dims keepdim (fun _ => hpos) δ)) x :=
  (hasPullbackAt_reduce_min dims keepdim (f x) hpos hx).comp hf

/-- A mean over `dims`, pulled back by `meanBackward`. -/
@[fun_prop]
theorem mean (dims : List (Fin s.rank)) (keepdim : Bool) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.mean (f y) dims keepdim) ((meanCLM dims keepdim).comp Df)
      (fun δ => bf (Tensor.meanBackward dims keepdim δ)) x := by
  simpa only [meanCLM_apply] using
    clm (meanCLM dims keepdim) (meanBackward dims keepdim)
      (fun u δ => by rw [meanCLM_apply, inner_mean]) hf

/-- A cumulative sum along `dim`, pulled back by `cumsumBackward`. -/
@[fun_prop]
theorem cumsum (dim : Fin s.rank) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.cumsum (f y) dim) ((cumsumCLM dim).comp Df)
      (fun δ => bf (Tensor.cumsumBackward δ dim)) x := by
  simpa only [cumsumCLM_apply] using
    clm (cumsumCLM dim) (fun δ => cumsumBackward δ dim)
      (fun u δ => by rw [cumsumCLM_apply, inner_cumsum]) hf

end HasPullbackAt

/-! ## Certificates -/

/-- The backward of `Op.cumsum dim` is the adjoint of the derivative of its forward at every
input. -/
@[op_cert TorchLean.Op.cumsum derivative]
theorem cumsum_backward_eq_adjoint {s : Shape} (dim : Fin s.rank) (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.cumsum dim).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.cumsum dim).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.cumsum dim).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x).adjoint δ := by
  simp only [Op.rule?_cumsum, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- The backward of `Op.mean dims keepdim` is the adjoint of the derivative of its forward at
every input. -/
@[op_cert TorchLean.Op.mean derivative]
theorem mean_backward_eq_adjoint {s : Shape} (dims : List (Fin s.rank)) (keepdim : Bool)
    (x : Tensor ℝ s) (δ : Tensor ℝ (s.reduce dims keepdim)) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.mean dims keepdim).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.mean dims keepdim).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.mean dims keepdim).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x).adjoint
        δ := by
  simp only [Op.rule?_mean, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- The backward of `Op.reduce op dims keepdim h` is the adjoint of the derivative of its forward
at `x`, for a sum or a product at every input. For a maximum (a minimum) we assume that every
fiber of `x` has an entry larger (smaller) than all its other entries, which is exactly where the
forward is differentiable (`differentiableAt_reduce_max_iff`, `differentiableAt_reduce_min_iff`). -/
@[op_cert TorchLean.Op.reduce derivative]
theorem reduce_backward_eq_adjoint {s : Shape} (op : Tensor.ReduceOp) (dims : List (Fin s.rank))
    (keepdim : Bool) (h : op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d) (x : Tensor ℝ s)
    (hmax : op = .max → ∀ o, ∃ r, ∀ r', r' ≠ r →
      x (s.reduceSource dims keepdim o r') < x (s.reduceSource dims keepdim o r))
    (hmin : op = .min → ∀ o, ∃ r, ∀ r', r' ≠ r →
      x (s.reduceSource dims keepdim o r) < x (s.reduceSource dims keepdim o r'))
    (δ : Tensor ℝ (s.reduce dims keepdim)) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.reduce op dims keepdim h).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.reduce op dims keepdim h).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.reduce op dims keepdim h).rule? (α := ℝ) |>.get rfl).forward (x', ())).1)
        x).adjoint δ := by
  simp only [Op.rule?_reduce, Option.get_some]
  cases op with
  | sum =>
    apply HasPullbackAt.differentiableAt_and
    · fun_prop
    · rfl
  | prod =>
    apply HasPullbackAt.differentiableAt_and
    · fun_prop
    · rfl
  | max =>
    exact HasPullbackAt.differentiableAt_and
      (Tensor.hasPullbackAt_reduce_max dims keepdim x (h (Or.inl rfl)) (hmax rfl)) rfl
  | min =>
    exact HasPullbackAt.differentiableAt_and
      (Tensor.hasPullbackAt_reduce_min dims keepdim x (h (Or.inr rfl)) (hmin rfl)) rfl

/-- At every real input, the stored backward of a maximum (a minimum) over `dims` gives the entry
read at `o` and `r` the value `δ o` divided by the number of entries of the fiber of `o` equal to it
when it is a largest (smallest) entry of that fiber, and zero otherwise. At a tie the tied entries
thus split `δ o` evenly, so `x = (1, 1)` reduced over its one axis with `δ = 1` gets
`(1 / 2, 1 / 2)`. -/
@[op_cert TorchLean.Op.reduce convention]
theorem Op.reduce_backward_eq {s : Shape} (op : Tensor.ReduceOp) (dims : List (Fin s.rank))
    (keepdim : Bool) (h : op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d) (x : Tensor ℝ s)
    (δ : Tensor ℝ (s.reduce dims keepdim)) (o : (s.reduce dims keepdim).Coord)
    (r : (s.selectAxes dims).Coord) :
    (op = .max →
      (((Op.reduce op dims keepdim h).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1
          (s.reduceSource dims keepdim o r) =
        if ∀ q, x (s.reduceSource dims keepdim o q) ≤ x (s.reduceSource dims keepdim o r) then
          δ o / (Finset.univ.filter fun q =>
            x (s.reduceSource dims keepdim o q) = x (s.reduceSource dims keepdim o r)).card
        else 0) ∧
    (op = .min →
      (((Op.reduce op dims keepdim h).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1
          (s.reduceSource dims keepdim o r) =
        if ∀ q, x (s.reduceSource dims keepdim o r) ≤ x (s.reduceSource dims keepdim o q) then
          δ o / (Finset.univ.filter fun q =>
            x (s.reduceSource dims keepdim o q) = x (s.reduceSource dims keepdim o r)).card
        else 0) := by
  simp only [Op.rule?_reduce, Option.get_some]
  constructor
  · rintro rfl
    exact Tensor.reduceBackward_max_reduceSource dims keepdim x h δ o r
  · rintro rfl
    exact Tensor.reduceBackward_min_reduceSource dims keepdim x h δ o r

end TorchLean
