/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.InnerProductSpace.ProdL2
public import TorchLean.NN.FDeriv.PullbackTensor
public import TorchLean.NN.FDeriv.Reindex
public import TorchLean.NN.Op.Pointwise
public import TorchLean.Tensor.Op.Cert
public import TorchLean.Tensor.Op.Table.Movement

/-!
# Derivative certificates of the scatter

We prove that the real forward of `Op.scatter` at the default element, as a map of the base and
the updates jointly on their L2 product, has the stored backward as the adjoint of its derivative.
This holds at every input for the sum, the replacement and the product, and for the maximum and
the minimum where every output entry has a unique extremal participant, which is exactly where
they are differentiable. At ties we state the stored backward of the maximum and the minimum in
closed form, as a convention: the tied participants share the cotangent evenly.

We write every rule as the weighted map `scatterCLM`, which scales the base and the updates
entrywise and adds every update to the entry it lands on, and we register `@[fun_prop]` rules for
the tensor scatters so that `op_deriv` composes through them.
-/

@[expose] public section

open scoped RealInnerProductSpace Topology

open Filter

namespace TorchLean

namespace Tensor

variable {s t : Shape}

/-! ## The coordinate an update lands on -/

/-- The entry of `AxisTuple.coordEquiv` at the axis numbered `k` is the index of the coordinate
along that axis. -/
private theorem val_coordEquiv {ι : Type} (length : ι → Nat) :
    ∀ (axes : List ι) (c : Shape.Coord (axes.map length)) (k : Fin axes.length),
      ((Einops.AxisTuple.coordEquiv length axes c) k).val = Shape.Coord.index c k.val
  | [], _, k => k.elim0
  | _ :: axes, c, k => by
    obtain ⟨hd, tl⟩ := c
    refine Fin.cases rfl (fun k => ?_) k
    exact val_coordEquiv length axes tl k

section Target

variable (dims : List (Fin s.rank)) (h : dims.Nodup)
  (indices : Tensor (Einops.AxisTuple s.extent dims) t)

/-- The coordinate of `s` on which `Tensor.scatter` along `dims` writes the update at `u` for the
index tuples `indices`, which is the coordinate that `scatterUpdatesBackward` reads for `u`. -/
def scatterTarget (u : (t ++ s.eraseAxes dims).Coord) : s.Coord :=
  let parts := Shape.Coord.appendEquiv t
    (s.eraseAxes (dims ++ ([] : List (Fin s.rank × Fin t.rank)).map Prod.fst))
    (Shape.Coord.reindexEq (by simp) u)
  (⟨by simpa using h, by simp, by simp⟩ : Shape.GatherValid s t dims []).source
    (indices parts.1) parts.1 parts.2

/-- The update gradient that `scatterUpdatesBackward` stores for a sum reads the gradient at the
coordinate the update lands on. -/
theorem scatterUpdatesBackward_sum_apply {α : Type} [Storage α] [Zero α] (g : Tensor α s)
    (u : (t ++ s.eraseAxes dims).Coord) :
    scatterUpdatesBackward .sum dims h indices g u = g (scatterTarget dims h indices u) := by
  simp only [scatterUpdatesBackward, gather_apply, scatterTarget]

/-- The index tuple along `dims` of the coordinate an update lands on is the tuple at the leading
part of the update coordinate. -/
theorem get_scatterTarget_dims (u : (t ++ s.eraseAxes dims).Coord) :
    (fun k => Shape.Coord.get (scatterTarget dims h indices u) (dims.get k)) =
      indices (Shape.Coord.appendEquiv t (s.eraseAxes dims) u).1 := by
  funext k
  have hval : ∀ (c c' : t.Coord) (i j : Fin dims.length), c = c' → i = j →
      (indices c i).val = (indices c' j).val := by
    rintro c c' i j rfl rfl
    rfl
  apply Fin.ext
  simp only [scatterTarget, Shape.GatherValid.source, Shape.Coord.get_ofFun, List.get_mem,
    ↓reduceDIte, Fin.val_cast]
  refine hval _ _ _ _ (Shape.Coord.ext_index fun j hj => ?_) (Fin.ext (List.get_idxOf h k))
  rw [Shape.Coord.index_appendEquiv_fst _ _ _ j (by rwa [← Shape.rank_eq_length]),
    Shape.Coord.index_reindexEq,
    Shape.Coord.index_appendEquiv_fst _ _ _ j (by rwa [← Shape.rank_eq_length])]

/-- The coordinate of `s.eraseAxes dims` made of the indices of `c` along the axes outside
`dims`. -/
def scatterRest (c : s.Coord) : (s.eraseAxes dims).Coord :=
  (Einops.AxisTuple.coordEquiv s.extent (s.otherAxes dims)).symm
    fun k => Shape.Coord.get c ((s.otherAxes dims).get k)

/-- The coordinate of the updates that `Tensor.scatter` along `dims` combines into the entry `c` at
the index `i` of the index tensor: `Shape.Coord.unlinearize i` followed by `scatterRest dims c`. -/
def scatterUpdate (c : s.Coord) (i : Fin (Shape.size t)) : (t ++ s.eraseAxes dims).Coord :=
  (Shape.Coord.appendEquiv t (s.eraseAxes dims)).symm
    (Shape.Coord.unlinearize i, scatterRest dims c)

/-- Splitting `scatterUpdate dims c i` gives the coordinate of `i` and `scatterRest dims c`. -/
@[simp] theorem appendEquiv_scatterUpdate (c : s.Coord) (i : Fin (Shape.size t)) :
    Shape.Coord.appendEquiv t (s.eraseAxes dims) (scatterUpdate dims c i) =
      (Shape.Coord.unlinearize i, scatterRest dims c) :=
  Equiv.apply_symm_apply _ _

/-- The index at position `j` of `scatterRest dims c` is the index of `c` along the axis at
position `j` of `s.otherAxes dims`. -/
private theorem index_scatterRest (c : s.Coord) (j : Nat) (hj : j < (s.otherAxes dims).length) :
    Shape.Coord.index (scatterRest dims c) j =
      (Shape.Coord.get c ((s.otherAxes dims).get ⟨j, hj⟩)).val := by
  have key := val_coordEquiv s.extent (s.otherAxes dims)
    ((Einops.AxisTuple.coordEquiv s.extent (s.otherAxes dims)).symm
      fun k => Shape.Coord.get c ((s.otherAxes dims).get k)) ⟨j, hj⟩
  rw [Equiv.apply_symm_apply] at key
  exact key.symm

/-- The indices outside `dims` of the coordinate an update lands on are the trailing part of the
update coordinate. -/
theorem scatterRest_scatterTarget (u : (t ++ s.eraseAxes dims).Coord) :
    scatterRest dims (scatterTarget dims h indices u) =
      (Shape.Coord.appendEquiv t (s.eraseAxes dims) u).2 := by
  refine Shape.Coord.ext_index fun j hj => ?_
  have hj' : j < (s.otherAxes dims).length := by
    simpa [Shape.eraseAxes, Shape.rank_eq_length] using hj
  rw [index_scatterRest dims _ j hj', Shape.Coord.index_appendEquiv_snd]
  have hk : (s.otherAxes dims).get ⟨j, hj'⟩ ∉ dims := Shape.mem_otherAxes.mp (List.get_mem _ _)
  have hidx : List.idxOf ((s.otherAxes dims).get ⟨j, hj'⟩)
      (s.otherAxes (dims ++ ([] : List (Fin s.rank × Fin t.rank)).map Prod.fst)) = j := by
    simpa using List.get_idxOf (Shape.nodup_otherAxes s dims) ⟨j, hj'⟩
  simp only [scatterTarget, Shape.GatherValid.source, Shape.Coord.get_ofFun, hk, ↓reduceDIte]
  split
  · rename_i hb
    simp at hb
  · erw [Fin.val_cast, val_coordEquiv, Shape.Coord.index_appendEquiv_snd,
      Shape.Coord.index_reindexEq]
    exact congrArg (fun n => Shape.Coord.index u (n + List.length t)) hidx

/-- A coordinate of `s` is determined by its index tuple along `dims` and by `scatterRest`. -/
theorem eq_of_get_dims_of_scatterRest {c c' : s.Coord}
    (hkey : (fun k => Shape.Coord.get c (dims.get k)) = fun k => Shape.Coord.get c' (dims.get k))
    (hrest : scatterRest dims c = scatterRest dims c') : c = c' := by
  refine Shape.Coord.ext fun d => ?_
  by_cases hd : d ∈ dims
  · obtain ⟨k, rfl⟩ := List.mem_iff_get.mp hd
    exact congrFun hkey k
  · obtain ⟨k, rfl⟩ := List.mem_iff_get.mp (Shape.mem_otherAxes.mpr hd)
    exact congrFun
      ((Einops.AxisTuple.coordEquiv s.extent (s.otherAxes dims)).symm.injective hrest) k

/-- The update `scatterUpdate dims c i` lands on `c` when the index tuple at `i` is that of `c`. -/
theorem scatterTarget_scatterUpdate {c : s.Coord} {i : Fin (Shape.size t)}
    (hi : indices (Shape.Coord.unlinearize i) = fun k => Shape.Coord.get c (dims.get k)) :
    scatterTarget dims h indices (scatterUpdate dims c i) = c := by
  refine eq_of_get_dims_of_scatterRest dims ?_ ?_
  · rw [get_scatterTarget_dims, appendEquiv_scatterUpdate]
    exact hi
  · rw [scatterRest_scatterTarget, appendEquiv_scatterUpdate]

/-- Every update coordinate is the `scatterUpdate` at the entry it lands on and at its index. -/
theorem scatterUpdate_scatterTarget (u : (t ++ s.eraseAxes dims).Coord) :
    scatterUpdate dims (scatterTarget dims h indices u)
        (Shape.Coord.linearize (Shape.Coord.appendEquiv t (s.eraseAxes dims) u).1) = u := by
  rw [scatterUpdate, Shape.Coord.unlinearize_linearize, scatterRest_scatterTarget]
  exact (Shape.Coord.appendEquiv t (s.eraseAxes dims)).symm_apply_apply u

/-- `scatterUpdate dims c` is injective. -/
theorem scatterUpdate_injective (c : s.Coord) :
    Function.Injective (scatterUpdate (t := t) dims c) := by
  intro i j hij
  have := congrArg (fun v => Shape.Coord.linearize
    (Shape.Coord.appendEquiv t (s.eraseAxes dims) v).1) hij
  simpa using this

/-- The update coordinates that land on `c` are the images under `scatterUpdate dims c` of the
indices whose tuple is that of `c` along `dims`. -/
private theorem filter_scatterTarget_eq (c : s.Coord) :
    (Finset.univ.filter fun u => scatterTarget dims h indices u = c) =
      (Finset.univ.filter fun i : Fin (Shape.size t) =>
          indices (Shape.Coord.unlinearize i) = fun k => Shape.Coord.get c (dims.get k)).map
        ⟨scatterUpdate dims c, scatterUpdate_injective dims c⟩ := by
  ext u
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_map,
    Function.Embedding.coeFn_mk]
  constructor
  · rintro rfl
    exact ⟨_, by rw [Shape.Coord.unlinearize_linearize, get_scatterTarget_dims],
      scatterUpdate_scatterTarget dims h indices u⟩
  · rintro ⟨i, hi, rfl⟩
    exact scatterTarget_scatterUpdate dims h indices hi

/-- A sum over the indices whose tuple is that of `c` is a sum over the update coordinates that
land on `c`, when the summands agree at `scatterUpdate dims c`. -/
theorem sum_ite_scatterUpdate {M : Type*} [AddCommMonoid M] (c : s.Coord)
    (G : Fin (Shape.size t) → M) (F : (t ++ s.eraseAxes dims).Coord → M)
    (hG : ∀ i, indices (Shape.Coord.unlinearize i) = (fun k => Shape.Coord.get c (dims.get k)) →
      G i = F (scatterUpdate dims c i)) :
    (∑ i, if indices (Shape.Coord.unlinearize i) = (fun k => Shape.Coord.get c (dims.get k))
        then G i else 0) = ∑ u, if scatterTarget dims h indices u = c then F u else 0 := by
  rw [← Finset.sum_filter, ← Finset.sum_filter, filter_scatterTarget_eq dims h indices c,
    Finset.sum_map]
  exact Finset.sum_congr rfl fun i hi => hG i (Finset.mem_filter.mp hi).2

end Target

/-! ## Folds -/

section Folds

variable {ι : Type*} (p : ι → Prop) [DecidablePred p] (f : ι → ℝ)

/-- A left fold whose step `m` is addition, applied at the entries satisfying `p`, is the seed
plus the sum of those entries. -/
private theorem foldl_ite_add (m : ℝ → ℝ → ℝ) (hm : ∀ a b, m a b = a + b) (l : List ι)
    (x : ℝ) : l.foldl (fun acc i => if p i then m acc (f i) else acc) x =
      x + (l.map fun i => if p i then f i else 0).sum := by
  induction l generalizing x with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    rw [ih]
    by_cases hi : p i
    · rw [ite_eq_left hi, ite_eq_left hi, hm, add_assoc]
    · rw [ite_eq_right hi, ite_eq_right hi, zero_add]

/-- A left fold whose step `m` is multiplication, applied at the entries satisfying `p`, is the
seed times the product of those entries. -/
private theorem foldl_ite_mul (m : ℝ → ℝ → ℝ) (hm : ∀ a b, m a b = a * b) (l : List ι)
    (x : ℝ) : l.foldl (fun acc i => if p i then m acc (f i) else acc) x =
      x * (l.map fun i => if p i then f i else 1).prod := by
  induction l generalizing x with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldl_cons, List.map_cons, List.prod_cons]
    rw [ih]
    by_cases hi : p i
    · rw [ite_eq_left hi, ite_eq_left hi, hm, mul_assoc]
    · rw [ite_eq_right hi, ite_eq_right hi, one_mul]

/-- A left fold whose step `m` is the maximum, applied at the entries satisfying `p`, is at most
`y` exactly when the seed and those entries are. -/
private theorem foldl_ite_max_le_iff (m : ℝ → ℝ → ℝ) (hm : ∀ a b, m a b = max a b) (y : ℝ)
    (l : List ι) (x : ℝ) :
    l.foldl (fun acc i => if p i then m acc (f i) else acc) x ≤ y ↔
      x ≤ y ∧ ∀ i ∈ l, p i → f i ≤ y := by
  induction l generalizing x with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldl_cons]
    rw [ih]
    by_cases hi : p i
    · simp only [hi, ↓reduceIte, hm, max_le_iff, List.mem_cons, forall_eq_or_imp,
        true_implies, and_assoc]
    · simp only [hi, ↓reduceIte, List.mem_cons, forall_eq_or_imp, false_implies, true_and]

/-- A left fold whose step `m` is the minimum, applied at the entries satisfying `p`, is at least
`y` exactly when the seed and those entries are. -/
private theorem le_foldl_ite_min_iff (m : ℝ → ℝ → ℝ) (hm : ∀ a b, m a b = min a b) (y : ℝ)
    (l : List ι) (x : ℝ) :
    y ≤ l.foldl (fun acc i => if p i then m acc (f i) else acc) x ↔
      y ≤ x ∧ ∀ i ∈ l, p i → y ≤ f i := by
  induction l generalizing x with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldl_cons]
    rw [ih]
    by_cases hi : p i
    · simp only [hi, ↓reduceIte, hm, le_min_iff, List.mem_cons, forall_eq_or_imp,
        true_implies, and_assoc]
    · simp only [hi, ↓reduceIte, List.mem_cons, forall_eq_or_imp, false_implies, true_and]

/-- A left fold whose step `m` is the maximum, applied at the entries satisfying `p`, is `y` when
the seed and those entries are at most `y` and one of them is `y`. -/
private theorem foldl_ite_max_eq (m : ℝ → ℝ → ℝ) (hm : ∀ a b, m a b = max a b) (l : List ι)
    (x y : ℝ) (hx : x ≤ y) (hf : ∀ i ∈ l, p i → f i ≤ y)
    (hy : x = y ∨ ∃ i ∈ l, p i ∧ f i = y) :
    l.foldl (fun acc i => if p i then m acc (f i) else acc) x = y := by
  have hlow := (foldl_ite_max_le_iff p f m hm _ l x).mp le_rfl
  refine le_antisymm ((foldl_ite_max_le_iff p f m hm y l x).mpr ⟨hx, hf⟩) ?_
  rcases hy with rfl | ⟨i, hi, hpi, rfl⟩
  · exact hlow.1
  · exact hlow.2 i hi hpi

/-- A left fold whose step `m` is the minimum, applied at the entries satisfying `p`, is `y` when
the seed and those entries are at least `y` and one of them is `y`. -/
private theorem foldl_ite_min_eq (m : ℝ → ℝ → ℝ) (hm : ∀ a b, m a b = min a b) (l : List ι)
    (x y : ℝ) (hx : y ≤ x) (hf : ∀ i ∈ l, p i → y ≤ f i)
    (hy : x = y ∨ ∃ i ∈ l, p i ∧ f i = y) :
    l.foldl (fun acc i => if p i then m acc (f i) else acc) x = y := by
  have hup := (le_foldl_ite_min_iff p f m hm _ l x).mp le_rfl
  refine le_antisymm ?_ ((le_foldl_ite_min_iff p f m hm y l x).mpr ⟨hx, hf⟩)
  rcases hy with rfl | ⟨i, hi, hpi, rfl⟩
  · exact hup.1
  · exact hup.2 i hi hpi

end Folds

/-- A left fold over `List.finRange n` whose step `m` returns the update, applied at the indices
satisfying `p`, is the seed when no index satisfies `p`, and otherwise `f` at the last such
index. -/
private theorem foldl_ite_replace (m : ℝ → ℝ → ℝ) (hm : ∀ a b, m a b = b) :
    ∀ {n : Nat} (p : Fin n → Prop) [DecidablePred p] (f : Fin n → ℝ) (x : ℝ),
      (List.finRange n).foldl (fun acc i => if p i then m acc (f i) else acc) x =
        (if ∀ i, ¬ p i then x else 0) +
          ∑ i, if p i then (if ∀ j, j ≤ i ∨ ¬ p j then f i else 0) else 0
  | 0, p, _, f, x => by simp
  | n + 1, p, _, f, x => by
    have h1 : ∀ i : Fin n, (∀ j : Fin (n + 1), j ≤ i.castSucc ∨ ¬ p j) ↔
        (∀ j : Fin n, j ≤ i ∨ ¬ p j.castSucc) ∧ ¬ p (Fin.last n) := fun i => by
      rw [Fin.forall_fin_succ']
      simp only [Fin.castSucc_le_castSucc_iff, not_le.mpr (Fin.castSucc_lt_last i), false_or]
    rw [List.finRange_succ_last, List.foldl_append]
    simp only [List.foldl_map, List.foldl_cons, List.foldl_nil]
    rw [foldl_ite_replace m hm (fun i => p i.castSucc) (fun i => f i.castSucc) x,
      Fin.sum_univ_castSucc]
    by_cases hl : p (Fin.last n)
    · have hG : ∀ i : Fin n, (if p i.castSucc then
          (if ∀ j : Fin (n + 1), j ≤ i.castSucc ∨ ¬ p j then f i.castSucc else 0) else 0) = 0 :=
        fun i => ite_eq_right_iff.mpr fun _ => ite_eq_right fun hall => ((h1 i).mp hall).2 hl
      rw [ite_eq_left hl, hm, ite_eq_right fun hall => hall _ hl,
        Finset.sum_congr rfl fun i _ => hG i, Finset.sum_const_zero, ite_eq_left hl,
        ite_eq_left fun j => Or.inl (Fin.le_last j), zero_add, zero_add]
    · rw [ite_eq_right hl, ite_eq_right hl, add_zero]
      congr 1
      · simp only [Fin.forall_fin_succ', hl, not_false_eq_true, and_true]
      · exact Finset.sum_congr rfl fun i _ => by
          simp only [h1 i, hl, not_false_eq_true, and_true]

/-- The number of indices of `List.finRange n` at which `q` holds is the number of elements of
`Fin n` at which it holds. -/
private theorem countP_finRange {n : Nat} (q : Fin n → Bool) :
    (List.finRange n).countP q = (Finset.univ.filter fun j => q j = true).card := by
  rw [← List.toFinset_finRange, (List.nodup_finRange n).card_eq_countP]
  simp only [Bool.decide_eq_true]

/-- A product over a filter of `Fin n` without `i` is the product over `Fin n` of the entries
other than `i` that satisfy the filter. -/
private theorem prod_erase_filter {n : Nat} (q : Fin n → Prop) [DecidablePred q] (F : Fin n → ℝ)
    (i : Fin n) :
    ∏ j ∈ (Finset.univ.filter q).erase i, F j = ∏ j, if j ≠ i ∧ q j then F j else 1 := by
  rw [← Finset.prod_filter, ← Finset.filter_ne', Finset.filter_filter]
  exact Finset.prod_congr (Finset.filter_congr fun j _ => and_comm) fun _ _ => rfl

/-! ## The weighted scatter -/

section Weighted

variable (dims : List (Fin s.rank)) (h : dims.Nodup)
  (indices : Tensor (Einops.AxisTuple s.extent dims) t)

/-- The continuous linear map that scales the base by `a` and the updates by `w` entrywise and
adds every scaled update to the entry it lands on. -/
noncomputable def scatterCLM (a : s.Coord → ℝ) (w : (t ++ s.eraseAxes dims).Coord → ℝ) :
    WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)) →L[ℝ] Tensor ℝ s :=
  (Op.slopeCLM a).comp (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ (t ++ s.eraseAxes dims))) +
    (Rep.pullCLM (scatterTarget dims h indices)).adjoint.comp
      ((Op.slopeCLM w).comp (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ (t ++ s.eraseAxes dims))))

/-- The entry of `scatterCLM` at `c` is the scaled base entry plus the scaled updates that land
on `c`. -/
theorem scatterCLM_apply (a : s.Coord → ℝ) (w : (t ++ s.eraseAxes dims).Coord → ℝ)
    (p : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) (c : s.Coord) :
    scatterCLM dims h indices a w p c =
      a c * p.fst c + ∑ v, if scatterTarget dims h indices v = c then w v * p.snd v else 0 := by
  simp only [scatterCLM, _root_.add_apply, ContinuousLinearMap.comp_apply, Rep.pullCLM_adjoint,
    Rep.sumFibers, Tensor.add_apply, Rep.get_ofFn, Op.slopeCLM_apply, WithLp.fstL_apply,
    WithLp.sndL_apply]

/-- `scatterCLM` pairs with the map that scales a cotangent by `a` on the base and, on the updates,
reads it where each update lands and scales it by `w`. -/
theorem inner_scatterCLM (a : s.Coord → ℝ) (w : (t ++ s.eraseAxes dims).Coord → ℝ)
    (p : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) (δ : Tensor ℝ s) :
    ⟪scatterCLM dims h indices a w p, δ⟫ =
      ⟪p, WithLp.toLp 2 (Op.slopeCLM a δ,
        Op.slopeCLM w (Rep.pull (scatterTarget dims h indices) δ))⟫ := by
  have hp : scatterCLM dims h indices a w p = Op.slopeCLM a p.fst +
      (Rep.pullCLM (scatterTarget dims h indices)).adjoint (Op.slopeCLM w p.snd) := rfl
  calc ⟪scatterCLM dims h indices a w p, δ⟫
      = ⟪Op.slopeCLM a p.fst, δ⟫ +
          ⟪(Rep.pullCLM (scatterTarget dims h indices)).adjoint (Op.slopeCLM w p.snd), δ⟫ := by
        rw [hp, inner_add_left]
    _ = ⟪p.fst, Op.slopeCLM a δ⟫ +
          ⟪p.snd, Op.slopeCLM w (Rep.pull (scatterTarget dims h indices) δ)⟫ := by
        rw [← ContinuousLinearMap.adjoint_inner_right (Op.slopeCLM a), Op.slopeCLM_adjoint,
          ContinuousLinearMap.adjoint_inner_left, Rep.pullCLM_apply,
          ← ContinuousLinearMap.adjoint_inner_right (Op.slopeCLM w), Op.slopeCLM_adjoint]
    _ = _ := rfl

/-- `scatterCLM` is its own derivative, with the pullback of `inner_scatterCLM`. -/
theorem hasPullbackAt_scatterCLM (a : s.Coord → ℝ) (w : (t ++ s.eraseAxes dims).Coord → ℝ)
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) :
    HasPullbackAt (scatterCLM dims h indices a w) (scatterCLM dims h indices a w)
      (fun δ => WithLp.toLp 2 (Op.slopeCLM a δ,
        Op.slopeCLM w (Rep.pull (scatterTarget dims h indices) δ))) x :=
  ⟨(scatterCLM dims h indices a w).hasFDerivAt, inner_scatterCLM dims h indices a w⟩

/-- A map that agrees with `scatterCLM` near `x` has it as derivative at `x`, with any pullback
equal to that of `inner_scatterCLM`. -/
theorem hasPullbackAt_of_scatterCLM
    {F : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)) → Tensor ℝ s}
    {b : Tensor ℝ s → WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))}
    (a : s.Coord → ℝ) (w : (t ++ s.eraseAxes dims).Coord → ℝ)
    {x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))}
    (hF : ∀ᶠ p in 𝓝 x, ∀ c, F p c = scatterCLM dims h indices a w p c)
    (hb : ∀ δ, b δ = WithLp.toLp 2 (Op.slopeCLM a δ,
      Op.slopeCLM w (Rep.pull (scatterTarget dims h indices) δ))) :
    HasPullbackAt F (scatterCLM dims h indices a w) b x :=
  ((hasPullbackAt_scatterCLM dims h indices a w x).congr_of_eventuallyEq
    (hF.mono fun _ hp => Rep.ext hp)).congr_deriv rfl fun δ => (hb δ).symm

end Weighted

/-! ## The sum and the replacement -/

section Linear

variable (dims : List (Fin s.rank)) (h : dims.Nodup)
  (indices : Tensor (Einops.AxisTuple s.extent dims) t)
  {P : Tensor (Option (Einops.AxisTuple s.extent dims)) t}

/-- An entry of a sum scatter at the index tuples `indices` is the base entry plus the updates
that land on it. -/
theorem scatter_sum_apply_eq (hP : ∀ j, P j = some (indices j)) (b : Tensor ℝ s)
    (u : Tensor ℝ (t ++ s.eraseAxes dims)) (c : s.Coord) :
    scatter .sum b dims P u c =
      b c + ∑ v, if scatterTarget dims h indices v = c then u v else 0 := by
  rw [scatter_apply]
  refine (foldl_ite_add _ (fun i => u (scatterUpdate dims c i))
    (ScatterReduce.Scalar.combine (op := .sum)) (fun _ _ => rfl) _ _).trans ?_
  rw [← Fin.sum_univ_def]
  refine congrArg (b c + ·) ((Finset.sum_congr rfl fun i _ => ?_).trans
    (sum_ite_scatterUpdate dims h indices c (fun i => u (scatterUpdate dims c i)) u
      fun _ _ => rfl))
  simp only [hP, Option.some.injEq]

/-- A sum scatter at the index tuples `indices`, as a map of the base and the updates jointly, is
`scatterCLM` with unit weights, and the stored pullback, the cotangent on the base and its read
where each update lands on the updates, pairs with it. -/
theorem hasPullbackAt_scatter_sum (hP : ∀ j, P j = some (indices j))
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) :
    HasPullbackAt (fun p => scatter .sum p.fst dims P p.snd)
      (scatterCLM dims h indices (fun _ => 1) (fun _ => 1))
      (fun δ => WithLp.toLp 2 (δ, scatterUpdatesBackward .sum dims h indices δ)) x :=
  hasPullbackAt_of_scatterCLM dims h indices _ _
    (Eventually.of_forall fun p c => by
      rw [scatter_sum_apply_eq dims h indices hP, scatterCLM_apply]
      simp only [one_mul])
    fun δ => congrArg (WithLp.toLp 2) (Prod.ext (Rep.ext fun c => by
      simp only [Op.slopeCLM_apply, one_mul]) (Rep.ext fun v => by
      simp only [Op.slopeCLM_apply, one_mul, Rep.pull_apply, scatterUpdatesBackward_sum_apply]))

/-- The weight of a base entry in a replacing scatter: one when no index tuple is that of the
entry, zero otherwise. -/
noncomputable def scatterReplaceBaseWeight (c : s.Coord) : ℝ :=
  if ∀ i : Fin (Shape.size t),
      ¬ indices (Shape.Coord.unlinearize i) = fun k => Shape.Coord.get c (dims.get k)
  then 1 else 0

/-- The weight of an update in a replacing scatter: one when no later index tuple in row-major
order is the same, zero otherwise. -/
noncomputable def scatterReplaceWeight (v : (t ++ s.eraseAxes dims).Coord) : ℝ :=
  if ∀ j, j ≤ Shape.Coord.linearize (Shape.Coord.appendEquiv t (s.eraseAxes dims) v).1 ∨
      ¬ indices (Shape.Coord.unlinearize j) =
        indices (Shape.Coord.appendEquiv t (s.eraseAxes dims) v).1
  then 1 else 0

/-- An entry of a replacing scatter at the index tuples `indices` is the base entry if no update
lands on it, plus the last update in row-major order that lands on it. -/
theorem scatter_replace_apply_eq (hP : ∀ j, P j = some (indices j)) (b : Tensor ℝ s)
    (u : Tensor ℝ (t ++ s.eraseAxes dims)) (c : s.Coord) :
    scatter .replace b dims P u c = scatterReplaceBaseWeight dims indices c * b c +
      ∑ v, if scatterTarget dims h indices v = c then scatterReplaceWeight dims indices v * u v
        else 0 := by
  rw [scatter_apply]
  refine (foldl_ite_replace (ScatterReduce.Scalar.combine (op := .replace)) (fun _ _ => rfl) _
    (fun i => u (scatterUpdate dims c i)) _).trans ?_
  simp only [hP, Option.some.injEq]
  refine congrArg₂ (· + ·) ?_ (sum_ite_scatterUpdate dims h indices c _ _ fun i hi => ?_)
  · simp only [scatterReplaceBaseWeight, ite_mul, one_mul, zero_mul]
  · simp only [scatterReplaceWeight, appendEquiv_scatterUpdate, Shape.Coord.linearize_unlinearize,
      hi, ite_mul, one_mul, zero_mul]

/-- The update gradient that `scatterUpdatesBackward` stores for a replacement is the replacement
weight times the gradient where the update lands. -/
theorem scatterUpdatesBackward_replace_apply (g : Tensor ℝ s) (v : (t ++ s.eraseAxes dims).Coord) :
    scatterUpdatesBackward .replace dims h indices g v =
      scatterReplaceWeight dims indices v * g (scatterTarget dims h indices v) := by
  have hread := scatterUpdatesBackward_sum_apply dims h indices g v
  simp only [scatterUpdatesBackward] at hread ⊢
  simp only [Rep.get_ofFn, scatterReplaceWeight, List.all_eq_true, List.mem_finRange,
    true_implies, Bool.or_eq_true, decide_eq_true_eq, Bool.not_eq_true', decide_eq_false_iff_not,
    ite_mul, one_mul, zero_mul, hread]

/-- A replacing scatter at the index tuples `indices`, as a map of the base and the updates
jointly, is `scatterCLM` with the replacement weights, and the stored pullback pairs with it. -/
theorem hasPullbackAt_scatter_replace (hP : ∀ j, P j = some (indices j))
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) :
    HasPullbackAt (fun p => scatter .replace p.fst dims P p.snd)
      (scatterCLM dims h indices (scatterReplaceBaseWeight dims indices)
        (scatterReplaceWeight dims indices))
      (fun δ => WithLp.toLp 2 (scatter .replace δ dims P (full _ 0),
        scatterUpdatesBackward .replace dims h indices δ)) x :=
  hasPullbackAt_of_scatterCLM dims h indices _ _
    (Eventually.of_forall fun p c => by
      rw [scatter_replace_apply_eq dims h indices hP, scatterCLM_apply])
    fun δ => congrArg (WithLp.toLp 2) (Prod.ext (Rep.ext fun c => by
      rw [scatter_replace_apply_eq dims h indices hP, Op.slopeCLM_apply]
      simp only [full_apply, mul_zero, ite_self, Finset.sum_const_zero, add_zero])
      (Rep.ext fun v => by
        rw [Op.slopeCLM_apply, Rep.pull_apply, scatterUpdatesBackward_replace_apply]))

end Linear

/-! ## The product -/

section Product

variable (dims : List (Fin s.rank)) (h : dims.Nodup)
  (indices : Tensor (Einops.AxisTuple s.extent dims) t)
  {P : Tensor (Option (Einops.AxisTuple s.extent dims)) t}

/-- The product of the updates that land on the entry `c` of a product scatter at the index tuples
`indices`. -/
noncomputable def scatterProdBaseWeight (u : Tensor ℝ (t ++ s.eraseAxes dims)) (c : s.Coord) :
    ℝ :=
  ∏ i, if indices (Shape.Coord.unlinearize i) = (fun k => Shape.Coord.get c (dims.get k)) then
    u (scatterUpdate dims c i) else 1

/-- The weight of the update at `v` in the derivative of a product scatter: the base entry it
lands on times the other updates that land there. -/
noncomputable def scatterProdWeight (b : Tensor ℝ s) (u : Tensor ℝ (t ++ s.eraseAxes dims))
    (v : (t ++ s.eraseAxes dims).Coord) : ℝ :=
  b (scatterTarget dims h indices v) *
    ∏ j, if j ≠ Shape.Coord.linearize (Shape.Coord.appendEquiv t (s.eraseAxes dims) v).1 ∧
        indices (Shape.Coord.unlinearize j) =
          indices (Shape.Coord.appendEquiv t (s.eraseAxes dims) v).1 then
      u ((Shape.Coord.appendEquiv t (s.eraseAxes dims)).symm
        (Shape.Coord.unlinearize j, (Shape.Coord.appendEquiv t (s.eraseAxes dims) v).2))
    else 1

/-- An entry of a product scatter at the index tuples `indices` is the base entry times the
updates that land on it. -/
theorem scatter_prod_apply_eq (hP : ∀ j, P j = some (indices j)) (b : Tensor ℝ s)
    (u : Tensor ℝ (t ++ s.eraseAxes dims)) (c : s.Coord) :
    scatter .prod b dims P u c = b c * scatterProdBaseWeight dims indices u c := by
  rw [scatter_apply]
  refine (foldl_ite_mul _ (fun i => u (scatterUpdate dims c i))
    (ScatterReduce.Scalar.combine (op := .prod)) (fun _ _ => rfl) _ _).trans ?_
  rw [← Fin.prod_univ_def]
  refine congrArg (b c * ·) (Finset.prod_congr rfl fun i _ => ?_)
  simp only [hP, Option.some.injEq]

/-- The pullback that `scatterProdBackward` stores, in closed form: it scales the cotangent by
`scatterProdBaseWeight` on the base and, on the updates, reads it where each update lands and
scales it by `scatterProdWeight`. -/
theorem scatterProdBackward_eq (b : Tensor ℝ s) (u : Tensor ℝ (t ++ s.eraseAxes dims))
    (δ : Tensor ℝ s) :
    scatterProdBackward b dims h indices u δ =
      (Op.slopeCLM (scatterProdBaseWeight dims indices u) δ,
        Op.slopeCLM (scatterProdWeight dims h indices b u)
          (Rep.pull (scatterTarget dims h indices) δ)) := by
  refine Prod.ext (Rep.ext fun c => ?_) (Rep.ext fun v => ?_)
  · simp only [scatterProdBackward, Op.slopeCLM_apply]
    exact (scatter_prod_apply_eq dims indices (fun j => Tensor.map_apply some indices j) δ u
      c).trans (mul_comm _ _)
  · simp only [scatterProdBackward, Rep.get_ofFn, Op.slopeCLM_apply, Rep.pull_apply,
      scatterUpdatesBackward_sum_apply, scatterProdWeight]
    rw [foldl_ite_mul _ _ (· * ·) (fun _ _ => rfl), ← Fin.prod_univ_def]
    ring

/-- A product scatter at the index tuples `indices`, as a map of the base and the updates jointly,
has at `x` the derivative `scatterCLM` with the product weights at `x`. -/
theorem hasFDerivAt_scatter_prod (hP : ∀ j, P j = some (indices j))
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) :
    HasFDerivAt (fun p => scatter .prod p.fst dims P p.snd)
      (scatterCLM dims h indices (scatterProdBaseWeight dims indices x.snd)
        (scatterProdWeight dims h indices x.fst x.snd)) x := by
  refine (hasFDerivAt_of_coordinates _ (fun c => (coordinateCLM c).comp
    (scatterCLM dims h indices (scatterProdBaseWeight dims indices x.snd)
      (scatterProdWeight dims h indices x.fst x.snd))) x fun c => ?_).congr_fderiv
    (ContinuousLinearMap.ext fun q => Rep.ext fun c => by
      rw [assemble_apply, ContinuousLinearMap.comp_apply, coordinateCLM_apply])
  have h1 := ((coordinateCLM c).comp
    (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ (t ++ s.eraseAxes dims)))).hasFDerivAt (x := x)
  have h2 := HasFDerivAt.finsetProd (x := x)
    (u := Finset.univ.filter fun i : Fin (Shape.size t) =>
      indices (Shape.Coord.unlinearize i) = fun k => Shape.Coord.get c (dims.get k))
    (g := fun i (p : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) =>
      p.snd (scatterUpdate dims c i))
    fun i _ => ((coordinateCLM (scatterUpdate dims c i)).comp
      (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ (t ++ s.eraseAxes dims)))).hasFDerivAt
  refine ((h1.mul h2).congr_of_eventuallyEq (Eventually.of_forall fun p => ?_)).congr_fderiv
    (ContinuousLinearMap.ext fun q => ?_)
  · exact (scatter_prod_apply_eq dims indices hP p.fst p.snd c).trans
      (congrArg (p.fst c * ·) (Finset.prod_filter _ _).symm)
  · simp only [_root_.add_apply, _root_.smul_apply, _root_.sum_apply,
      ContinuousLinearMap.comp_apply, coordinateCLM_apply, WithLp.fstL_apply, WithLp.sndL_apply,
      smul_eq_mul, scatterCLM_apply]
    rw [add_comm, Finset.prod_filter, Finset.mul_sum, Finset.sum_filter]
    refine congrArg (_ + ·) (sum_ite_scatterUpdate dims h indices c _ _ fun i hi => ?_)
    simp only [scatterProdWeight, scatterTarget_scatterUpdate dims h indices hi,
      appendEquiv_scatterUpdate, Shape.Coord.linearize_unlinearize, hi]
    exact (congrArg (x.fst c * ·) (congrArg (· * q.snd (scatterUpdate dims c i))
      (prod_erase_filter _ (fun j => x.snd (scatterUpdate dims c j)) i))).trans
      (mul_assoc _ _ _).symm

/-- A product scatter at the index tuples `indices`, as a map of the base and the updates jointly,
has the stored pullback `scatterProdBackward` at every point, zeros included. -/
theorem hasPullbackAt_scatter_prod (hP : ∀ j, P j = some (indices j))
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) :
    HasPullbackAt (fun p => scatter .prod p.fst dims P p.snd)
      (scatterCLM dims h indices (scatterProdBaseWeight dims indices x.snd)
        (scatterProdWeight dims h indices x.fst x.snd))
      (fun δ => WithLp.toLp 2 (scatterProdBackward x.fst dims h indices x.snd δ)) x :=
  HasPullbackAt.congr_deriv ⟨hasFDerivAt_scatter_prod dims h indices hP x,
    inner_scatterCLM dims h indices _ _⟩ rfl fun δ =>
    congrArg (WithLp.toLp 2) (scatterProdBackward_eq dims h indices x.fst x.snd δ).symm

end Product

/-! ## The maximum and the minimum -/

section Extremum

variable (dims : List (Fin s.rank)) (h : dims.Nodup)
  (indices : Tensor (Einops.AxisTuple s.extent dims) t)
  {P : Tensor (Option (Einops.AxisTuple s.extent dims)) t}

/-- The number of updates that land on `c` and equal `y c`. -/
noncomputable def scatterTies (u : Tensor ℝ (t ++ s.eraseAxes dims)) (y : Tensor ℝ s)
    (c : s.Coord) : ℕ :=
  (Finset.univ.filter fun v => scatterTarget dims h indices v = c ∧ u v = y c).card

/-- The share of the cotangent that the base entry at `c` receives in the stored pullback of a
scatter maximum or minimum with output `y`: one over the number of participants equal to `y c`
when the base entry is one of them, zero otherwise. -/
noncomputable def scatterShareBase (b : Tensor ℝ s) (u : Tensor ℝ (t ++ s.eraseAxes dims))
    (y : Tensor ℝ s) (c : s.Coord) : ℝ :=
  if b c = y c then ((scatterTies dims h indices u y c : ℝ) + 1)⁻¹ else 0

/-- The share of the cotangent that the update at `v` receives in the stored pullback of a scatter
maximum or minimum with output `y`: one over the number of participants equal to `y` where it
lands when the update is one of them, zero otherwise. -/
noncomputable def scatterShare (b : Tensor ℝ s) (u : Tensor ℝ (t ++ s.eraseAxes dims))
    (y : Tensor ℝ s) (v : (t ++ s.eraseAxes dims).Coord) : ℝ :=
  if u v = y (scatterTarget dims h indices v) then
    (((if b (scatterTarget dims h indices v) = y (scatterTarget dims h indices v) then
      scatterTies dims h indices u y (scatterTarget dims h indices v) + 1
    else scatterTies dims h indices u y (scatterTarget dims h indices v) : ℕ) : ℝ))⁻¹
  else 0

/-- The pullback that `scatterExtremumBackward` stores for a scatter maximum or minimum with output
`y`, in closed form: the participants of an entry that equal `y` there share the cotangent evenly,
through `scatterShareBase` and `scatterShare`. -/
theorem scatterExtremumBackward_eq (b y : Tensor ℝ s) (u : Tensor ℝ (t ++ s.eraseAxes dims))
    (δ : Tensor ℝ s) :
    scatterExtremumBackward y b dims h indices u δ =
      (Op.slopeCLM (scatterShareBase dims h indices b u y) δ,
        Op.slopeCLM (scatterShare dims h indices b u y)
          (Rep.pull (scatterTarget dims h indices) δ)) := by
  have hties (c : s.Coord) : (List.finRange (Shape.size t)).countP (fun j =>
      decide (indices (Shape.Coord.unlinearize j) = fun k => Shape.Coord.get c (dims.get k)) &&
        u (scatterUpdate dims c j) == y c) = scatterTies dims h indices u y c := by
    rw [countP_finRange, scatterTies, Finset.card_filter, Finset.card_filter]
    simp only [Bool.and_eq_true, decide_eq_true_eq, LawfulContext.beq_iff, ite_and]
    exact sum_ite_scatterUpdate dims h indices c _ _ fun _ _ => rfl
  refine Prod.ext (Rep.ext fun c => ?_) (Rep.ext fun v => ?_)
  · simp only [scatterExtremumBackward, Rep.get_ofFn, Op.slopeCLM_apply, extremumTie,
      Context.isNaN_real, Bool.and_false, Bool.or_false, LawfulContext.beq_iff,
      LawfulContext.divNat_eq]
    erw [hties c]
    rw [scatterShareBase]
    by_cases hb : b c = y c
    · simp only [ite_eq_left hb]
      rw [Nat.cast_add_one, div_eq_inv_mul]
    · simp only [ite_eq_right hb, zero_mul]
  · simp only [scatterExtremumBackward, Rep.get_ofFn, Op.slopeCLM_apply, Rep.pull_apply,
      scatterUpdatesBackward_sum_apply, extremumTie, Context.isNaN_real, Bool.and_false,
      Bool.or_false, LawfulContext.beq_iff, LawfulContext.divNat_eq]
    erw [hties (scatterTarget dims h indices v)]
    rw [scatterShare]
    by_cases hv : u v = y (scatterTarget dims h indices v)
    · simp only [ite_eq_left hv]
      rw [div_eq_inv_mul]
    · simp only [ite_eq_right hv, zero_mul]

/-- An entry of a scatter maximum at the index tuples `indices` is `m` when the participants of
the entry are at most `m` and one of them is `m`. -/
theorem scatter_max_apply_eq (hP : ∀ j, P j = some (indices j)) (b : Tensor ℝ s)
    (u : Tensor ℝ (t ++ s.eraseAxes dims)) (c : s.Coord) (m : ℝ) (hb : b c ≤ m)
    (hu : ∀ v, scatterTarget dims h indices v = c → u v ≤ m)
    (hm : b c = m ∨ ∃ v, scatterTarget dims h indices v = c ∧ u v = m) :
    scatter .max b dims P u c = m := by
  rw [scatter_apply]
  refine foldl_ite_max_eq _ (fun i => u (scatterUpdate dims c i))
    (ScatterReduce.Scalar.combine (op := .max)) (fun a b => Context.maximum_real a b) _ _ _ hb
    (fun i _ hi => hu _ (scatterTarget_scatterUpdate dims h indices
      (Option.some.inj ((hP _).symm.trans hi)))) ?_
  rcases hm with hm | ⟨v, rfl, rfl⟩
  · exact Or.inl hm
  · refine Or.inr ⟨Shape.Coord.linearize (Shape.Coord.appendEquiv t (s.eraseAxes dims) v).1,
      List.mem_finRange _, ?_, congrArg u (scatterUpdate_scatterTarget dims h indices v)⟩
    rw [hP, Shape.Coord.unlinearize_linearize, get_scatterTarget_dims]

/-- An entry of a scatter minimum at the index tuples `indices` is `m` when the participants of
the entry are at least `m` and one of them is `m`. -/
theorem scatter_min_apply_eq (hP : ∀ j, P j = some (indices j)) (b : Tensor ℝ s)
    (u : Tensor ℝ (t ++ s.eraseAxes dims)) (c : s.Coord) (m : ℝ) (hb : m ≤ b c)
    (hu : ∀ v, scatterTarget dims h indices v = c → m ≤ u v)
    (hm : b c = m ∨ ∃ v, scatterTarget dims h indices v = c ∧ u v = m) :
    scatter .min b dims P u c = m := by
  rw [scatter_apply]
  refine foldl_ite_min_eq _ (fun i => u (scatterUpdate dims c i))
    (ScatterReduce.Scalar.combine (op := .min)) (fun a b => Context.minimum_real a b) _ _ _ hb
    (fun i _ hi => hu _ (scatterTarget_scatterUpdate dims h indices
      (Option.some.inj ((hP _).symm.trans hi)))) ?_
  rcases hm with hm | ⟨v, rfl, rfl⟩
  · exact Or.inl hm
  · refine Or.inr ⟨Shape.Coord.linearize (Shape.Coord.appendEquiv t (s.eraseAxes dims) v).1,
      List.mem_finRange _, ?_, congrArg u (scatterUpdate_scatterTarget dims h indices v)⟩
    rw [hP, Shape.Coord.unlinearize_linearize, get_scatterTarget_dims]

/-- Near `x`, every strict inequality at `x` between an entry of the base and an update, or
between two updates, holds. -/
private theorem eventually_lt_scatter
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) :
    ∀ᶠ p in 𝓝 x, (∀ c v, x.snd v < x.fst c → p.snd v < p.fst c) ∧
      (∀ c v, x.fst c < x.snd v → p.fst c < p.snd v) ∧
      ∀ v v', x.snd v' < x.snd v → p.snd v' < p.snd v := by
  have hfst (c : s.Coord) := ((coordinateCLM c).comp
    (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ (t ++ s.eraseAxes dims)))).continuous.continuousAt
    (x := x)
  have hsnd (v : (t ++ s.eraseAxes dims).Coord) := ((coordinateCLM v).comp
    (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ (t ++ s.eraseAxes dims)))).continuous.continuousAt
    (x := x)
  have key {f g : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)) → ℝ}
      (hf : ContinuousAt f x) (hg : ContinuousAt g x) :
      ∀ᶠ p in 𝓝 x, f x < g x → f p < g p := by
    by_cases hlt : f x < g x
    · exact (hf.eventually_lt hg hlt).mono fun _ hp _ => hp
    · exact Eventually.of_forall fun _ hx => absurd hx hlt
  exact (eventually_all.2 fun c => eventually_all.2 fun v => key (hsnd v) (hfst c)).and
    ((eventually_all.2 fun c => eventually_all.2 fun v => key (hfst c) (hsnd v)).and
      (eventually_all.2 fun v => eventually_all.2 fun v' => key (hsnd v') (hsnd v)))

/-- At a point `p` near `x`, where every entry of a scatter maximum has a unique largest
participant, the entry at `c` of the scatter maximum is that of `scatterCLM` with the shares at
`x`. -/
private theorem scatter_max_eq_scatterCLM (hP : ∀ j, P j = some (indices j))
    {x p : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))} (c : s.Coord)
    (hc : (∀ v, scatterTarget dims h indices v = c → x.snd v < x.fst c) ∨
      ∃ v, scatterTarget dims h indices v = c ∧ x.fst c < x.snd v ∧
        ∀ v', scatterTarget dims h indices v' = c → v' ≠ v → x.snd v' < x.snd v)
    (hp1 : ∀ v, x.snd v < x.fst c → p.snd v < p.fst c)
    (hp2 : ∀ v, x.fst c < x.snd v → p.fst c < p.snd v)
    (hp3 : ∀ v v', x.snd v' < x.snd v → p.snd v' < p.snd v) :
    scatter .max p.fst dims P p.snd c =
      scatterCLM dims h indices
        (scatterShareBase dims h indices x.fst x.snd (scatter .max x.fst dims P x.snd))
        (scatterShare dims h indices x.fst x.snd (scatter .max x.fst dims P x.snd)) p c := by
  rw [scatterCLM_apply]
  rcases hc with hc | ⟨v0, hv0, hb0, huniq⟩
  · have hy : scatter .max x.fst dims P x.snd c = x.fst c :=
      scatter_max_apply_eq dims h indices hP _ _ c _ le_rfl (fun v hv => (hc v hv).le)
        (Or.inl rfl)
    have hties : scatterTies dims h indices x.snd (scatter .max x.fst dims P x.snd) c = 0 := by
      rw [scatterTies, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      rintro v - ⟨hv, hvy⟩
      exact (hc v hv).ne (hvy.trans hy)
    rw [scatter_max_apply_eq dims h indices hP _ _ c (p.fst c) le_rfl
        (fun v hv => (hp1 v (hc v hv)).le) (Or.inl rfl), scatterShareBase, ite_eq_left hy.symm,
      hties, Nat.cast_zero, zero_add, inv_one, one_mul, Finset.sum_eq_zero, add_zero]
    intro v _
    refine ite_eq_right_iff.mpr fun hv => ?_
    rw [scatterShare, hv, hy, ite_eq_right (hc v hv).ne, zero_mul]
  · have hle : ∀ v, scatterTarget dims h indices v = c → x.snd v ≤ x.snd v0 := fun v hv => by
      rcases eq_or_ne v v0 with rfl | hne
      exacts [le_rfl, (huniq v hv hne).le]
    have hy : scatter .max x.fst dims P x.snd c = x.snd v0 :=
      scatter_max_apply_eq dims h indices hP _ _ c _ hb0.le hle (Or.inr ⟨v0, hv0, rfl⟩)
    have hties : scatterTies dims h indices x.snd (scatter .max x.fst dims P x.snd) c = 1 := by
      rw [scatterTies, Finset.card_eq_one]
      refine ⟨v0, Finset.eq_singleton_iff_unique_mem.mpr
        ⟨Finset.mem_filter.mpr ⟨Finset.mem_univ _, hv0, hy.symm⟩, fun v hv => ?_⟩⟩
      obtain ⟨-, hvc, hvy⟩ := Finset.mem_filter.mp hv
      by_contra hne
      exact (huniq v hvc hne).ne (hvy.trans hy)
    rw [scatter_max_apply_eq dims h indices hP _ _ c (p.snd v0) (hp2 v0 hb0).le
        (fun v hv => by
          rcases eq_or_ne v v0 with rfl | hne
          exacts [le_rfl, (hp3 v0 v (huniq v hv hne)).le])
        (Or.inr ⟨v0, hv0, rfl⟩), scatterShareBase, hy, ite_eq_right hb0.ne, zero_mul, zero_add,
      Finset.sum_eq_single v0, ite_eq_left hv0, scatterShare, hv0, hy, ite_eq_left rfl,
      ite_eq_right hb0.ne, hties, Nat.cast_one, inv_one, one_mul]
    · intro v _ hne
      refine ite_eq_right_iff.mpr fun hvc => ?_
      rw [scatterShare, hvc, hy, ite_eq_right (huniq v hvc hne).ne, zero_mul]
    · exact fun hn => absurd (Finset.mem_univ v0) hn

/-- At a point `p` near `x`, where every entry of a scatter minimum has a unique smallest
participant, the entry at `c` of the scatter minimum is that of `scatterCLM` with the shares at
`x`. -/
private theorem scatter_min_eq_scatterCLM (hP : ∀ j, P j = some (indices j))
    {x p : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))} (c : s.Coord)
    (hc : (∀ v, scatterTarget dims h indices v = c → x.fst c < x.snd v) ∨
      ∃ v, scatterTarget dims h indices v = c ∧ x.snd v < x.fst c ∧
        ∀ v', scatterTarget dims h indices v' = c → v' ≠ v → x.snd v < x.snd v')
    (hp1 : ∀ v, x.snd v < x.fst c → p.snd v < p.fst c)
    (hp2 : ∀ v, x.fst c < x.snd v → p.fst c < p.snd v)
    (hp3 : ∀ v v', x.snd v' < x.snd v → p.snd v' < p.snd v) :
    scatter .min p.fst dims P p.snd c =
      scatterCLM dims h indices
        (scatterShareBase dims h indices x.fst x.snd (scatter .min x.fst dims P x.snd))
        (scatterShare dims h indices x.fst x.snd (scatter .min x.fst dims P x.snd)) p c := by
  rw [scatterCLM_apply]
  rcases hc with hc | ⟨v0, hv0, hb0, huniq⟩
  · have hy : scatter .min x.fst dims P x.snd c = x.fst c :=
      scatter_min_apply_eq dims h indices hP _ _ c _ le_rfl (fun v hv => (hc v hv).le)
        (Or.inl rfl)
    have hties : scatterTies dims h indices x.snd (scatter .min x.fst dims P x.snd) c = 0 := by
      rw [scatterTies, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      rintro v - ⟨hv, hvy⟩
      exact (hc v hv).ne' (hvy.trans hy)
    rw [scatter_min_apply_eq dims h indices hP _ _ c (p.fst c) le_rfl
        (fun v hv => (hp2 v (hc v hv)).le) (Or.inl rfl), scatterShareBase, ite_eq_left hy.symm,
      hties, Nat.cast_zero, zero_add, inv_one, one_mul, Finset.sum_eq_zero, add_zero]
    intro v _
    refine ite_eq_right_iff.mpr fun hv => ?_
    rw [scatterShare, hv, hy, ite_eq_right (hc v hv).ne', zero_mul]
  · have hle : ∀ v, scatterTarget dims h indices v = c → x.snd v0 ≤ x.snd v := fun v hv => by
      rcases eq_or_ne v v0 with rfl | hne
      exacts [le_rfl, (huniq v hv hne).le]
    have hy : scatter .min x.fst dims P x.snd c = x.snd v0 :=
      scatter_min_apply_eq dims h indices hP _ _ c _ hb0.le hle (Or.inr ⟨v0, hv0, rfl⟩)
    have hties : scatterTies dims h indices x.snd (scatter .min x.fst dims P x.snd) c = 1 := by
      rw [scatterTies, Finset.card_eq_one]
      refine ⟨v0, Finset.eq_singleton_iff_unique_mem.mpr
        ⟨Finset.mem_filter.mpr ⟨Finset.mem_univ _, hv0, hy.symm⟩, fun v hv => ?_⟩⟩
      obtain ⟨-, hvc, hvy⟩ := Finset.mem_filter.mp hv
      by_contra hne
      exact (huniq v hvc hne).ne' (hvy.trans hy)
    rw [scatter_min_apply_eq dims h indices hP _ _ c (p.snd v0) (hp1 v0 hb0).le
        (fun v hv => by
          rcases eq_or_ne v v0 with rfl | hne
          exacts [le_rfl, (hp3 v v0 (huniq v hv hne)).le])
        (Or.inr ⟨v0, hv0, rfl⟩), scatterShareBase, hy, ite_eq_right hb0.ne', zero_mul, zero_add,
      Finset.sum_eq_single v0, ite_eq_left hv0, scatterShare, hv0, hy, ite_eq_left rfl,
      ite_eq_right hb0.ne', hties, Nat.cast_one, inv_one, one_mul]
    · intro v _ hne
      refine ite_eq_right_iff.mpr fun hvc => ?_
      rw [scatterShare, hvc, hy, ite_eq_right (huniq v hvc hne).ne', zero_mul]
    · exact fun hn => absurd (Finset.mem_univ v0) hn

/-- A scatter maximum at the index tuples `indices`, as a map of the base and the updates jointly,
has at `x` the derivative `scatterCLM` with the shares at `x`, and the stored pullback pairs with
it, when every entry has a unique largest participant at `x`. That condition is exact by
`differentiableAt_scatter_max_iff`. -/
theorem hasPullbackAt_scatter_max (hP : ∀ j, P j = some (indices j))
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)))
    (hx : ∀ c, (∀ v, scatterTarget dims h indices v = c → x.snd v < x.fst c) ∨
      ∃ v, scatterTarget dims h indices v = c ∧ x.fst c < x.snd v ∧
        ∀ v', scatterTarget dims h indices v' = c → v' ≠ v → x.snd v' < x.snd v) :
    HasPullbackAt (fun p => scatter .max p.fst dims P p.snd)
      (scatterCLM dims h indices
        (scatterShareBase dims h indices x.fst x.snd (scatter .max x.fst dims P x.snd))
        (scatterShare dims h indices x.fst x.snd (scatter .max x.fst dims P x.snd)))
      (fun δ => WithLp.toLp 2 (scatterExtremumBackward (scatter .max x.fst dims P x.snd) x.fst
        dims h indices x.snd δ)) x :=
  hasPullbackAt_of_scatterCLM dims h indices _ _
    ((eventually_lt_scatter dims x).mono fun _ hp c =>
      scatter_max_eq_scatterCLM dims h indices hP c (hx c) (hp.1 c) (hp.2.1 c) hp.2.2)
    fun δ => congrArg (WithLp.toLp 2) (scatterExtremumBackward_eq dims h indices _ _ _ δ)

/-- A scatter minimum at the index tuples `indices`, as a map of the base and the updates jointly,
has at `x` the derivative `scatterCLM` with the shares at `x`, and the stored pullback pairs with
it, when every entry has a unique smallest participant at `x`. That condition is exact by
`differentiableAt_scatter_min_iff`. -/
theorem hasPullbackAt_scatter_min (hP : ∀ j, P j = some (indices j))
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)))
    (hx : ∀ c, (∀ v, scatterTarget dims h indices v = c → x.fst c < x.snd v) ∨
      ∃ v, scatterTarget dims h indices v = c ∧ x.snd v < x.fst c ∧
        ∀ v', scatterTarget dims h indices v' = c → v' ≠ v → x.snd v < x.snd v') :
    HasPullbackAt (fun p => scatter .min p.fst dims P p.snd)
      (scatterCLM dims h indices
        (scatterShareBase dims h indices x.fst x.snd (scatter .min x.fst dims P x.snd))
        (scatterShare dims h indices x.fst x.snd (scatter .min x.fst dims P x.snd)))
      (fun δ => WithLp.toLp 2 (scatterExtremumBackward (scatter .min x.fst dims P x.snd) x.fst
        dims h indices x.snd δ)) x :=
  hasPullbackAt_of_scatterCLM dims h indices _ _
    ((eventually_lt_scatter dims x).mono fun _ hp c =>
      scatter_min_eq_scatterCLM dims h indices hP c (hx c) (hp.1 c) (hp.2.1 c) hp.2.2)
    fun δ => congrArg (WithLp.toLp 2) (scatterExtremumBackward_eq dims h indices _ _ _ δ)

/-- A scatter maximum, as a map of the base and the updates jointly, is not differentiable at an
`x` where the largest participant of the entry at `c`, the update at `v0`, ties with the base entry
or with another update that lands on `c`. -/
theorem not_differentiableAt_scatter_max (hP : ∀ j, P j = some (indices j))
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) (c : s.Coord)
    {v0 : (t ++ s.eraseAxes dims).Coord} (hv0 : scatterTarget dims h indices v0 = c)
    (hb : x.fst c ≤ x.snd v0) (hu : ∀ v, scatterTarget dims h indices v = c → x.snd v ≤ x.snd v0)
    (htie : x.fst c = x.snd v0 ∨
      ∃ v, v ≠ v0 ∧ scatterTarget dims h indices v = c ∧ x.snd v = x.snd v0) :
    ¬ DifferentiableAt ℝ (fun p => scatter .max p.fst dims P p.snd) x := by
  intro hd
  let e : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)) :=
    WithLp.toLp 2 (0, Rep.ofFn fun v => if v = v0 then 1 else 0)
  have hfst (τ : ℝ) : (x + τ • e).fst = x.fst := by
    simp only [e, WithLp.add_fst, WithLp.smul_fst, WithLp.toLp_fst, smul_zero, add_zero]
  have hsnd (τ : ℝ) (v : (t ++ s.eraseAxes dims).Coord) :
      (x + τ • e).snd v = x.snd v + if v = v0 then τ else 0 := by
    simp only [e, WithLp.add_snd, WithLp.smul_snd, WithLp.toLp_snd, Tensor.add_apply,
      Tensor.Rep.smul_apply, Rep.get_ofFn, smul_eq_mul, mul_ite, mul_one, mul_zero]
  have hline (τ : ℝ) :
      scatter .max (x + τ • e).fst dims P (x + τ • e).snd c = x.snd v0 + max τ 0 := by
    rw [hfst]
    rcases le_total 0 τ with hτ | hτ
    · rw [max_eq_left hτ]
      refine scatter_max_apply_eq dims h indices hP _ _ c _ (hb.trans (le_add_of_nonneg_right hτ))
        (fun v hv => ?_) (Or.inr ⟨v0, hv0, by rw [hsnd, ite_eq_left rfl]⟩)
      rw [hsnd]
      by_cases hv' : v = v0
      · subst hv'
        rw [ite_eq_left rfl]
      · rw [ite_eq_right hv', add_zero]
        exact (hu v hv).trans (le_add_of_nonneg_right hτ)
    · rw [max_eq_right hτ, add_zero]
      refine scatter_max_apply_eq dims h indices hP _ _ c _ hb (fun v hv => ?_) ?_
      · rw [hsnd]
        by_cases hv' : v = v0
        · subst hv'
          rw [ite_eq_left rfl]
          exact add_le_of_nonpos_right hτ
        · rw [ite_eq_right hv', add_zero]
          exact hu v hv
      · rcases htie with htie | ⟨v1, hne, hv1, heq⟩
        · exact Or.inl htie
        · exact Or.inr ⟨v1, hv1, by rw [hsnd, ite_eq_right hne, add_zero, heq]⟩
  have hd0 : DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)) =>
      scatter .max p.fst dims P p.snd) (x + (0 : ℝ) • e) := by
    rwa [zero_smul, add_zero]
  have hl : DifferentiableAt ℝ (fun τ : ℝ => x + τ • e) 0 := by fun_prop
  have hg : DifferentiableAt ℝ (fun τ : ℝ => x.snd v0 + max τ 0) 0 :=
    (DifferentiableAt.fun_comp' (0 : ℝ) (coordinateCLM c).differentiableAt
      (DifferentiableAt.fun_comp' (f := fun τ : ℝ => x + τ • e) (0 : ℝ) hd0
        hl)).congr_of_eventuallyEq (Eventually.of_forall fun τ => (hline τ).symm)
  refine not_differentiableAt_abs_zero ?_
  have habs : (_root_.abs : ℝ → ℝ) = fun τ => 2 * (x.snd v0 + max τ 0) - 2 * x.snd v0 - τ :=
    funext fun τ => by
      rcases le_total 0 τ with hτ | hτ
      · rw [abs_of_nonneg hτ, max_eq_left hτ]
        ring
      · rw [abs_of_nonpos hτ, max_eq_right hτ]
        ring
  rw [habs]
  exact ((hg.const_mul 2).sub_const _).sub differentiableAt_id

/-- A scatter minimum, as a map of the base and the updates jointly, is not differentiable at an
`x` where the smallest participant of the entry at `c`, the update at `v0`, ties with the base
entry or with another update that lands on `c`. -/
theorem not_differentiableAt_scatter_min (hP : ∀ j, P j = some (indices j))
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) (c : s.Coord)
    {v0 : (t ++ s.eraseAxes dims).Coord} (hv0 : scatterTarget dims h indices v0 = c)
    (hb : x.snd v0 ≤ x.fst c) (hu : ∀ v, scatterTarget dims h indices v = c → x.snd v0 ≤ x.snd v)
    (htie : x.fst c = x.snd v0 ∨
      ∃ v, v ≠ v0 ∧ scatterTarget dims h indices v = c ∧ x.snd v = x.snd v0) :
    ¬ DifferentiableAt ℝ (fun p => scatter .min p.fst dims P p.snd) x := by
  intro hd
  let e : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)) :=
    WithLp.toLp 2 (0, Rep.ofFn fun v => if v = v0 then 1 else 0)
  have hfst (τ : ℝ) : (x + τ • e).fst = x.fst := by
    simp only [e, WithLp.add_fst, WithLp.smul_fst, WithLp.toLp_fst, smul_zero, add_zero]
  have hsnd (τ : ℝ) (v : (t ++ s.eraseAxes dims).Coord) :
      (x + τ • e).snd v = x.snd v + if v = v0 then τ else 0 := by
    simp only [e, WithLp.add_snd, WithLp.smul_snd, WithLp.toLp_snd, Tensor.add_apply,
      Tensor.Rep.smul_apply, Rep.get_ofFn, smul_eq_mul, mul_ite, mul_one, mul_zero]
  have hline (τ : ℝ) :
      scatter .min (x + τ • e).fst dims P (x + τ • e).snd c = x.snd v0 + min τ 0 := by
    rw [hfst]
    rcases le_total 0 τ with hτ | hτ
    · rw [min_eq_right hτ, add_zero]
      refine scatter_min_apply_eq dims h indices hP _ _ c _ hb (fun v hv => ?_) ?_
      · rw [hsnd]
        by_cases hv' : v = v0
        · subst hv'
          rw [ite_eq_left rfl]
          exact le_add_of_nonneg_right hτ
        · rw [ite_eq_right hv', add_zero]
          exact hu v hv
      · rcases htie with htie | ⟨v1, hne, hv1, heq⟩
        · exact Or.inl htie
        · exact Or.inr ⟨v1, hv1, by rw [hsnd, ite_eq_right hne, add_zero, heq]⟩
    · rw [min_eq_left hτ]
      refine scatter_min_apply_eq dims h indices hP _ _ c _ ((add_le_of_nonpos_right hτ).trans hb)
        (fun v hv => ?_) (Or.inr ⟨v0, hv0, by rw [hsnd, ite_eq_left rfl]⟩)
      rw [hsnd]
      by_cases hv' : v = v0
      · subst hv'
        rw [ite_eq_left rfl]
      · rw [ite_eq_right hv', add_zero]
        exact (add_le_of_nonpos_right hτ).trans (hu v hv)
  have hd0 : DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)) =>
      scatter .min p.fst dims P p.snd) (x + (0 : ℝ) • e) := by
    rwa [zero_smul, add_zero]
  have hl : DifferentiableAt ℝ (fun τ : ℝ => x + τ • e) 0 := by fun_prop
  have hg : DifferentiableAt ℝ (fun τ : ℝ => x.snd v0 + min τ 0) 0 :=
    (DifferentiableAt.fun_comp' (0 : ℝ) (coordinateCLM c).differentiableAt
      (DifferentiableAt.fun_comp' (f := fun τ : ℝ => x + τ • e) (0 : ℝ) hd0
        hl)).congr_of_eventuallyEq (Eventually.of_forall fun τ => (hline τ).symm)
  refine not_differentiableAt_abs_zero ?_
  have habs : (_root_.abs : ℝ → ℝ) = fun τ => 2 * x.snd v0 - 2 * (x.snd v0 + min τ 0) + τ :=
    funext fun τ => by
      rcases le_total 0 τ with hτ | hτ
      · rw [abs_of_nonneg hτ, min_eq_right hτ]
        ring
      · rw [abs_of_nonpos hτ, min_eq_left hτ]
        ring
  rw [habs]
  exact ((hg.const_mul 2).const_sub _).add differentiableAt_id

/-- A scatter maximum, as a map of the base and the updates jointly, is differentiable at `x`
exactly when every entry has a unique largest participant at `x`. -/
theorem differentiableAt_scatter_max_iff (hP : ∀ j, P j = some (indices j))
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) :
    DifferentiableAt ℝ (fun p => scatter .max p.fst dims P p.snd) x ↔
      ∀ c, (∀ v, scatterTarget dims h indices v = c → x.snd v < x.fst c) ∨
        ∃ v, scatterTarget dims h indices v = c ∧ x.fst c < x.snd v ∧
          ∀ v', scatterTarget dims h indices v' = c → v' ≠ v → x.snd v' < x.snd v := by
  refine ⟨fun hd c => ?_, fun hx =>
    (hasPullbackAt_scatter_max dims h indices hP x hx).hasFDerivAt.differentiableAt⟩
  by_cases hlt : ∀ v, scatterTarget dims h indices v = c → x.snd v < x.fst c
  · exact Or.inl hlt
  obtain ⟨v1, hv1, hle1⟩ : ∃ v, scatterTarget dims h indices v = c ∧ x.fst c ≤ x.snd v := by
    by_contra hno
    exact hlt fun v hv => not_le.mp fun hle => hno ⟨v, hv, hle⟩
  obtain ⟨v0, hv0mem, hv0max⟩ := Finset.exists_max_image
    (Finset.univ.filter fun v => scatterTarget dims h indices v = c) x.snd
    ⟨v1, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hv1⟩⟩
  have hv0 : scatterTarget dims h indices v0 = c := (Finset.mem_filter.mp hv0mem).2
  have hu (v) (hv : scatterTarget dims h indices v = c) : x.snd v ≤ x.snd v0 :=
    hv0max v (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hv⟩)
  rcases (hle1.trans (hu v1 hv1)).lt_or_eq with hb | hb
  · refine Or.inr ⟨v0, hv0, hb, fun v' hv' hne => (hu v' hv').lt_of_ne fun heq =>
      not_differentiableAt_scatter_max dims h indices hP x c hv0 hb.le hu
        (Or.inr ⟨v', hne, hv', heq⟩) hd⟩
  · exact absurd hd
      (not_differentiableAt_scatter_max dims h indices hP x c hv0 hb.le hu (Or.inl hb))

/-- A scatter minimum, as a map of the base and the updates jointly, is differentiable at `x`
exactly when every entry has a unique smallest participant at `x`. -/
theorem differentiableAt_scatter_min_iff (hP : ∀ j, P j = some (indices j))
    (x : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims))) :
    DifferentiableAt ℝ (fun p => scatter .min p.fst dims P p.snd) x ↔
      ∀ c, (∀ v, scatterTarget dims h indices v = c → x.fst c < x.snd v) ∨
        ∃ v, scatterTarget dims h indices v = c ∧ x.snd v < x.fst c ∧
          ∀ v', scatterTarget dims h indices v' = c → v' ≠ v → x.snd v < x.snd v' := by
  refine ⟨fun hd c => ?_, fun hx =>
    (hasPullbackAt_scatter_min dims h indices hP x hx).hasFDerivAt.differentiableAt⟩
  by_cases hlt : ∀ v, scatterTarget dims h indices v = c → x.fst c < x.snd v
  · exact Or.inl hlt
  obtain ⟨v1, hv1, hle1⟩ : ∃ v, scatterTarget dims h indices v = c ∧ x.snd v ≤ x.fst c := by
    by_contra hno
    exact hlt fun v hv => not_le.mp fun hle => hno ⟨v, hv, hle⟩
  obtain ⟨v0, hv0mem, hv0min⟩ := Finset.exists_min_image
    (Finset.univ.filter fun v => scatterTarget dims h indices v = c) x.snd
    ⟨v1, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hv1⟩⟩
  have hv0 : scatterTarget dims h indices v0 = c := (Finset.mem_filter.mp hv0mem).2
  have hu (v) (hv : scatterTarget dims h indices v = c) : x.snd v0 ≤ x.snd v :=
    hv0min v (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hv⟩)
  rcases ((hu v1 hv1).trans hle1).lt_or_eq with hb | hb
  · refine Or.inr ⟨v0, hv0, hb, fun v' hv' hne => (hu v' hv').lt_of_ne' fun heq =>
      not_differentiableAt_scatter_min dims h indices hP x c hv0 hb.le hu
        (Or.inr ⟨v', hne, hv', heq⟩) hd⟩
  · exact absurd hd
      (not_differentiableAt_scatter_min dims h indices hP x c hv0 hb.le hu (Or.inl hb.symm))

end Extremum

end Tensor

/-! ## Rules for `fun_prop` -/

namespace HasPullbackAt

open Tensor

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s t : Shape} {x : E}
  {β : Type} [Storage β]

/-- A sum scatter of `f y` with the updates `g y` at the index tuples `sel` of `index`, pulled
back by the cotangent on the base and its read where each update lands on the updates. -/
@[fun_prop]
theorem scatter_sum (dims : List (Fin s.rank)) (h : dims.Nodup)
    (sel : β → Einops.AxisTuple s.extent dims) (index : Tensor β t) {f : E → Tensor ℝ s}
    {g : E → Tensor ℝ (t ++ s.eraseAxes dims)} {Df : E →L[ℝ] Tensor ℝ s}
    {Dg : E →L[ℝ] Tensor ℝ (t ++ s.eraseAxes dims)} {bf : Tensor ℝ s → E}
    {bg : Tensor ℝ (t ++ s.eraseAxes dims) → E} (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt
      (fun y => Tensor.scatter .sum (f y) dims (Tensor.map (fun i => some (sel i)) index) (g y))
      ((scatterCLM dims h (Tensor.map sel index) (fun _ => 1) (fun _ => 1)).comp
        ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
          (Df.prod Dg)))
      (fun δ => bf δ + bg (scatterUpdatesBackward .sum dims h (Tensor.map sel index) δ)) x :=
  (hasPullbackAt_scatter_sum dims h (Tensor.map sel index)
    (fun j => (Tensor.map_apply (fun i => some (sel i)) index j).trans
      (congrArg some (Tensor.map_apply sel index j).symm))
    _).comp (toLp_prod hf hg)

/-- A replacing scatter of `f y` with the updates `g y` at the index tuples `sel` of `index`,
pulled back by the stored `scatter .replace` of the cotangent with zero updates on the base and by
`scatterUpdatesBackward .replace` on the updates. -/
@[fun_prop]
theorem scatter_replace (dims : List (Fin s.rank)) (h : dims.Nodup)
    (sel : β → Einops.AxisTuple s.extent dims) (index : Tensor β t) {f : E → Tensor ℝ s}
    {g : E → Tensor ℝ (t ++ s.eraseAxes dims)} {Df : E →L[ℝ] Tensor ℝ s}
    {Dg : E →L[ℝ] Tensor ℝ (t ++ s.eraseAxes dims)} {bf : Tensor ℝ s → E}
    {bg : Tensor ℝ (t ++ s.eraseAxes dims) → E} (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt
      (fun y => Tensor.scatter .replace (f y) dims (Tensor.map (fun i => some (sel i)) index)
        (g y))
      ((scatterCLM dims h (Tensor.map sel index)
          (scatterReplaceBaseWeight dims (Tensor.map sel index))
          (scatterReplaceWeight dims (Tensor.map sel index))).comp
        ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
          (Df.prod Dg)))
      (fun δ => bf (Tensor.scatter .replace δ dims (Tensor.map (fun i => some (sel i)) index)
          (Tensor.full _ 0)) +
        bg (scatterUpdatesBackward .replace dims h (Tensor.map sel index) δ)) x :=
  (hasPullbackAt_scatter_replace dims h (Tensor.map sel index)
    (fun j => (Tensor.map_apply (fun i => some (sel i)) index j).trans
      (congrArg some (Tensor.map_apply sel index j).symm))
    _).comp (toLp_prod hf hg)

/-- A product scatter of `f y` with the updates `g y` at the index tuples `sel` of `index`, pulled
back by the stored `scatterProdBackward` at `f x` and `g x`, zeros included. -/
@[fun_prop]
theorem scatter_prod (dims : List (Fin s.rank)) (h : dims.Nodup)
    (sel : β → Einops.AxisTuple s.extent dims) (index : Tensor β t) {f : E → Tensor ℝ s}
    {g : E → Tensor ℝ (t ++ s.eraseAxes dims)} {Df : E →L[ℝ] Tensor ℝ s}
    {Dg : E →L[ℝ] Tensor ℝ (t ++ s.eraseAxes dims)} {bf : Tensor ℝ s → E}
    {bg : Tensor ℝ (t ++ s.eraseAxes dims) → E} (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt
      (fun y => Tensor.scatter .prod (f y) dims (Tensor.map (fun i => some (sel i)) index) (g y))
      ((scatterCLM dims h (Tensor.map sel index)
          (scatterProdBaseWeight dims (Tensor.map sel index) (g x))
          (scatterProdWeight dims h (Tensor.map sel index) (f x) (g x))).comp
        ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
          (Df.prod Dg)))
      (fun δ => bf (scatterProdBackward (f x) dims h (Tensor.map sel index) (g x) δ).1 +
        bg (scatterProdBackward (f x) dims h (Tensor.map sel index) (g x) δ).2) x :=
  (hasPullbackAt_scatter_prod dims h (Tensor.map sel index)
    (fun j => (Tensor.map_apply (fun i => some (sel i)) index j).trans
      (congrArg some (Tensor.map_apply sel index j).symm))
    _).comp (toLp_prod hf hg)

/-- A scatter maximum of `f y` with the updates `g y` at the index tuples `sel` of `index`, pulled
back by the stored `scatterExtremumBackward` at `f x` and `g x`, when every entry has a unique
largest participant there. That condition is sufficient, and it is exact for the scatter itself by
`differentiableAt_scatter_max_iff`. -/
@[fun_prop]
theorem scatter_max (dims : List (Fin s.rank)) (h : dims.Nodup)
    (sel : β → Einops.AxisTuple s.extent dims) (index : Tensor β t) {f : E → Tensor ℝ s}
    {g : E → Tensor ℝ (t ++ s.eraseAxes dims)} {Df : E →L[ℝ] Tensor ℝ s}
    {Dg : E →L[ℝ] Tensor ℝ (t ++ s.eraseAxes dims)} {bf : Tensor ℝ s → E}
    {bg : Tensor ℝ (t ++ s.eraseAxes dims) → E}
    (hx : ∀ c, (∀ v, scatterTarget dims h (Tensor.map sel index) v = c → g x v < f x c) ∨
      ∃ v, scatterTarget dims h (Tensor.map sel index) v = c ∧ f x c < g x v ∧
        ∀ v', scatterTarget dims h (Tensor.map sel index) v' = c → v' ≠ v → g x v' < g x v)
    (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt
      (fun y => Tensor.scatter .max (f y) dims (Tensor.map (fun i => some (sel i)) index) (g y))
      ((scatterCLM dims h (Tensor.map sel index)
          (scatterShareBase dims h (Tensor.map sel index) (f x) (g x)
            (Tensor.scatter .max (f x) dims (Tensor.map (fun i => some (sel i)) index) (g x)))
          (scatterShare dims h (Tensor.map sel index) (f x) (g x)
            (Tensor.scatter .max (f x) dims (Tensor.map (fun i => some (sel i)) index)
              (g x)))).comp
        ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
          (Df.prod Dg)))
      (fun δ => bf (scatterExtremumBackward
          (Tensor.scatter .max (f x) dims (Tensor.map (fun i => some (sel i)) index) (g x))
          (f x) dims h (Tensor.map sel index) (g x) δ).1 +
        bg (scatterExtremumBackward
          (Tensor.scatter .max (f x) dims (Tensor.map (fun i => some (sel i)) index) (g x))
          (f x) dims h (Tensor.map sel index) (g x) δ).2) x :=
  HasPullbackAt.comp (f := fun y => WithLp.toLp 2 (f y, g y))
    (hasPullbackAt_scatter_max dims h (Tensor.map sel index)
      (fun j => (Tensor.map_apply (fun i => some (sel i)) index j).trans
        (congrArg some (Tensor.map_apply sel index j).symm))
      (WithLp.toLp 2 (f x, g x)) hx) (toLp_prod hf hg)

/-- A scatter minimum of `f y` with the updates `g y` at the index tuples `sel` of `index`, pulled
back by the stored `scatterExtremumBackward` at `f x` and `g x`, when every entry has a unique
smallest participant there. That condition is sufficient, and it is exact for the scatter itself
by `differentiableAt_scatter_min_iff`. -/
@[fun_prop]
theorem scatter_min (dims : List (Fin s.rank)) (h : dims.Nodup)
    (sel : β → Einops.AxisTuple s.extent dims) (index : Tensor β t) {f : E → Tensor ℝ s}
    {g : E → Tensor ℝ (t ++ s.eraseAxes dims)} {Df : E →L[ℝ] Tensor ℝ s}
    {Dg : E →L[ℝ] Tensor ℝ (t ++ s.eraseAxes dims)} {bf : Tensor ℝ s → E}
    {bg : Tensor ℝ (t ++ s.eraseAxes dims) → E}
    (hx : ∀ c, (∀ v, scatterTarget dims h (Tensor.map sel index) v = c → f x c < g x v) ∨
      ∃ v, scatterTarget dims h (Tensor.map sel index) v = c ∧ g x v < f x c ∧
        ∀ v', scatterTarget dims h (Tensor.map sel index) v' = c → v' ≠ v → g x v < g x v')
    (hf : HasPullbackAt f Df bf x) (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt
      (fun y => Tensor.scatter .min (f y) dims (Tensor.map (fun i => some (sel i)) index) (g y))
      ((scatterCLM dims h (Tensor.map sel index)
          (scatterShareBase dims h (Tensor.map sel index) (f x) (g x)
            (Tensor.scatter .min (f x) dims (Tensor.map (fun i => some (sel i)) index) (g x)))
          (scatterShare dims h (Tensor.map sel index) (f x) (g x)
            (Tensor.scatter .min (f x) dims (Tensor.map (fun i => some (sel i)) index)
              (g x)))).comp
        ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
          (Df.prod Dg)))
      (fun δ => bf (scatterExtremumBackward
          (Tensor.scatter .min (f x) dims (Tensor.map (fun i => some (sel i)) index) (g x))
          (f x) dims h (Tensor.map sel index) (g x) δ).1 +
        bg (scatterExtremumBackward
          (Tensor.scatter .min (f x) dims (Tensor.map (fun i => some (sel i)) index) (g x))
          (f x) dims h (Tensor.map sel index) (g x) δ).2) x :=
  HasPullbackAt.comp (f := fun y => WithLp.toLp 2 (f y, g y))
    (hasPullbackAt_scatter_min dims h (Tensor.map sel index)
      (fun j => (Tensor.map_apply (fun i => some (sel i)) index j).trans
        (congrArg some (Tensor.map_apply sel index j).symm))
      (WithLp.toLp 2 (f x, g x)) hx) (toLp_prod hf hg)

end HasPullbackAt

/-! ## The certificates of `Op.scatter` -/

namespace Op

open Tensor

/-- At the default element, the forward of `Op.scatter` is differentiable in the base and the
updates together, on their L2 product, and its pair of pullbacks is the adjoint of that
derivative, for the sum, the replacement and the product at every input, and for the maximum
(minimum) where every output entry has a unique largest (smallest) participant. That set is
exact, since off it the forward has no derivative (`differentiableAt_scatter_max_iff` and
`differentiableAt_scatter_min_iff`), as for the base `(0)` and one update `0` landing on it. -/
@[op_cert TorchLean.Op.scatter derivative 0 2]
theorem scatter_prod_backward_eq_adjoint {s t : Shape} (rule : ScatterReduce)
    (dims : List (Fin s.rank)) (h : dims.Nodup) (base : Tensor ℝ s)
    (index : Tensor (Port.IndexTuple (dims.map s.extent)) t)
    (updates : Tensor ℝ (t ++ s.eraseAxes dims))
    (hmax : rule = .max → ∀ c,
      (∀ v, scatterTarget dims h (Tensor.map (Op.axisTuples _ dims) index) v = c →
        updates v < base c) ∨
      ∃ v, scatterTarget dims h (Tensor.map (Op.axisTuples _ dims) index) v = c ∧
        base c < updates v ∧
        ∀ v', scatterTarget dims h (Tensor.map (Op.axisTuples _ dims) index) v' = c → v' ≠ v →
          updates v' < updates v)
    (hmin : rule = .min → ∀ c,
      (∀ v, scatterTarget dims h (Tensor.map (Op.axisTuples _ dims) index) v = c →
        base c < updates v) ∨
      ∃ v, scatterTarget dims h (Tensor.map (Op.axisTuples _ dims) index) v = c ∧
        updates v < base c ∧
        ∀ v', scatterTarget dims h (Tensor.map (Op.axisTuples _ dims) index) v' = c → v' ≠ v →
          updates v < updates v')
    (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)) =>
        (((Op.scatter (t := t) rule dims h).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, index, p.snd, ())).1)
      (WithLp.toLp 2 (base, updates)) ∧
    WithLp.toLp 2
        ((((Op.scatter (t := t) rule dims h).rule? (α := ℝ) |>.get rfl).backward
          (base, index, updates, ()) (δ, ())).1,
        (((Op.scatter (t := t) rule dims h).rule? (α := ℝ) |>.get rfl).backward
          (base, index, updates, ()) (δ, ())).2.2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)) =>
        (((Op.scatter (t := t) rule dims h).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, index, p.snd, ())).1)
        (WithLp.toLp 2 (base, updates))).adjoint δ := by
  have hP (j : t.Coord) : Tensor.map (fun i => some (Op.axisTuples _ dims i)) index j =
      some (Tensor.map (Op.axisTuples _ dims) index j) :=
    (Tensor.map_apply _ index j).trans (congrArg some (Tensor.map_apply _ index j).symm)
  simp only [Op.rule?_scatter, Option.get_some]
  cases rule with
  | replace =>
    exact HasPullbackAt.differentiableAt_and (hasPullbackAt_scatter_replace dims h _ hP _) rfl
  | sum => exact HasPullbackAt.differentiableAt_and (hasPullbackAt_scatter_sum dims h _ hP _) rfl
  | prod =>
    exact HasPullbackAt.differentiableAt_and (hasPullbackAt_scatter_prod dims h _ hP _) rfl
  | max =>
    exact HasPullbackAt.differentiableAt_and
      (hasPullbackAt_scatter_max dims h _ hP _ (hmax rfl)) rfl
  | min =>
    exact HasPullbackAt.differentiableAt_and
      (hasPullbackAt_scatter_min dims h _ hP _ (hmin rfl)) rfl

/-- At the default element and every real input, the stored backward of `Op.scatter` by a
maximum (minimum) splits `δ c` evenly between the participants of the output entry `c` that equal
the result there, the base entry and the updates landing on it, and gives the others zero. With
two tied participants each receives `δ c / 2`, which is `δ c` times the mean of the one-sided
partial derivatives one and zero, and with `k > 2` each receives `δ c / k`, which for `δ c ≠ 0` is
not. -/
@[op_cert TorchLean.Op.scatter convention]
theorem scatter_backward_eq {s t : Shape} (rule : ScatterReduce) (dims : List (Fin s.rank))
    (h : dims.Nodup) (base : Tensor ℝ s) (index : Tensor (Port.IndexTuple (dims.map s.extent)) t)
    (updates : Tensor ℝ (t ++ s.eraseAxes dims)) (δ : Tensor ℝ s) :
    (rule = .max →
      ((((Op.scatter (t := t) rule dims h).rule? (α := ℝ) |>.get rfl).backward
          (base, index, updates, ()) (δ, ())).1,
        (((Op.scatter (t := t) rule dims h).rule? (α := ℝ) |>.get rfl).backward
          (base, index, updates, ()) (δ, ())).2.2.1) =
      (Op.slopeCLM (scatterShareBase dims h (Tensor.map (Op.axisTuples _ dims) index) base
          updates (Tensor.scatter .max base dims
            (Tensor.map (fun i => some (Op.axisTuples _ dims i)) index) updates)) δ,
        Op.slopeCLM (scatterShare dims h (Tensor.map (Op.axisTuples _ dims) index) base updates
          (Tensor.scatter .max base dims
            (Tensor.map (fun i => some (Op.axisTuples _ dims i)) index) updates))
          (Rep.pull (scatterTarget dims h (Tensor.map (Op.axisTuples _ dims) index)) δ))) ∧
    (rule = .min →
      ((((Op.scatter (t := t) rule dims h).rule? (α := ℝ) |>.get rfl).backward
          (base, index, updates, ()) (δ, ())).1,
        (((Op.scatter (t := t) rule dims h).rule? (α := ℝ) |>.get rfl).backward
          (base, index, updates, ()) (δ, ())).2.2.1) =
      (Op.slopeCLM (scatterShareBase dims h (Tensor.map (Op.axisTuples _ dims) index) base
          updates (Tensor.scatter .min base dims
            (Tensor.map (fun i => some (Op.axisTuples _ dims i)) index) updates)) δ,
        Op.slopeCLM (scatterShare dims h (Tensor.map (Op.axisTuples _ dims) index) base updates
          (Tensor.scatter .min base dims
            (Tensor.map (fun i => some (Op.axisTuples _ dims i)) index) updates))
          (Rep.pull (scatterTarget dims h (Tensor.map (Op.axisTuples _ dims) index)) δ))) := by
  simp only [Op.rule?_scatter, Option.get_some]
  constructor
  · rintro rfl
    exact scatterExtremumBackward_eq dims h _ base _ updates δ
  · rintro rfl
    exact scatterExtremumBackward_eq dims h _ base _ updates δ

end Op

end TorchLean
