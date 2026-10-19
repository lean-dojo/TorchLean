/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Softmax
public import NN.Spec.Layers.MoE
import Mathlib.Algebra.BigOperators.Field

/-!
# Mixture of experts

The facts behind the choices in `NN.Spec.Layers.MoE`, in four groups.

* Routing keeps exactly `min k E` experts, and every kept expert beats every expert it displaces
  (`card_kept`, `beats_of_kept`), whenever the comparison of the scores is a strict weak order
  (`StrictWeak`). Real numbers satisfy this, and so do IEEE numbers other than NaN, on which `+0`
  and `-0` tie.
* Which experts are kept depends only on how the scores compare (`kept_congr`). Over `ℝ` the
  softmax preserves every comparison, so routing on probabilities, as Mixtral does, keeps the
  experts with the largest logits (`kept_softmax`), and the gates are the softmax of the kept
  logits (`getScalar_gatesVecSpec_softmax`). Rounding breaks this with floats: in binary32 the
  logits `0` and `2⁻³⁰` have the same probability, so routing on probabilities keeps expert `0`
  and routing on logits keeps expert `1`.
* Over `ℝ` the dense layer, which runs every expert, equals Mixtral's sparse block, which runs only
  the kept ones (`forwardSpec_eq_sparseForwardSpec`).
* A token's output depends only on that token, so a batch can be split across devices anywhere.
  A capacity per device, as in GShard and Switch, breaks this: two tokens routed to the same
  expert are both kept on two devices, but on one device the second is dropped
  (`capacity_depends_on_devices`). So does a balancing loss computed on each device's tokens
  (`balance_depends_on_devices`). Computed per work unit, a fixed set of tokens, both are
  functions of the unit again.
-/

@[expose] public section

open scoped BigOperators

namespace NN.Proofs.Models.MoE

open Spec Spec.MoE TorchLean TorchLean.Tensor

/-! ## Routing -/

section Routing

variable {α : Type} [Context α]

theorem gtBool_eq_true {x y : α} : Context.gtBool x y = true ↔ x > y := by
  simp [Context.gtBool]

theorem gtBool_eq_false {x y : α} : Context.gtBool x y = false ↔ ¬ x > y := by
  rw [← gtBool_eq_true, Bool.not_eq_true]

theorem beats_eq_true {E : ℕ} (scores : Fin E → α) (j i : Fin E) :
    beats scores j i = true ↔ scores j > scores i ∨ (j < i ∧ ¬ scores i > scores j) := by
  simp [beats, gtBool_eq_true, gtBool_eq_false]

/-- The comparison of `scores` is a strict weak order: no score is above itself, `>` is
transitive, and whenever `a > c` every other score is below `a` or above `c`. -/
structure StrictWeak {E : ℕ} (scores : Fin E → α) : Prop where
  irrefl : ∀ i, ¬ scores i > scores i
  trans : ∀ a b c, scores a > scores b → scores b > scores c → scores a > scores c
  split : ∀ a b c, scores a > scores c → scores a > scores b ∨ scores b > scores c

/-- Real scores are compared by a strict weak order. -/
theorem strictWeak_real {E : ℕ} (scores : Fin E → ℝ) : StrictWeak scores where
  irrefl _ := lt_irrefl _
  trans _ _ _ hab hbc := lt_trans hbc hab
  split a b c hac := by
    rcases lt_or_ge (scores b) (scores a) with h | h
    · exact Or.inl h
    · exact Or.inr (lt_of_lt_of_le hac h)

variable {E : ℕ} {scores : Fin E → α}

theorem beats_self (h : StrictWeak scores) (i : Fin E) : beats scores i i = false := by
  cases hb : beats scores i i
  · rfl
  · rcases (beats_eq_true scores i i).mp hb with hb | ⟨hlt, _⟩
    · exact absurd hb (h.irrefl i)
    · exact absurd hlt (lt_irrefl i)

/-- Of two different experts, one beats the other. -/
theorem beats_total (scores : Fin E → α) {i j : Fin E} (hij : i ≠ j) :
    beats scores i j = true ∨ beats scores j i = true := by
  rw [beats_eq_true, beats_eq_true]
  by_cases h₁ : scores i > scores j
  · exact Or.inl (Or.inl h₁)
  by_cases h₂ : scores j > scores i
  · exact Or.inr (Or.inl h₂)
  rcases lt_or_gt_of_ne hij with h | h
  · exact Or.inl (Or.inr ⟨h, h₂⟩)
  · exact Or.inr (Or.inr ⟨h, h₁⟩)

theorem beats_trans (h : StrictWeak scores) {a b c : Fin E} (hab : beats scores a b = true)
    (hbc : beats scores b c = true) : beats scores a c = true := by
  rw [beats_eq_true] at hab hbc ⊢
  rcases hab with hab | ⟨hab, nba⟩ <;> rcases hbc with hbc | ⟨hbc, ncb⟩
  · exact Or.inl (h.trans _ _ _ hab hbc)
  · rcases h.split a c b hab with h' | h'
    · exact Or.inl h'
    · exact absurd h' ncb
  · rcases h.split b a c hbc with h' | h'
    · exact absurd h' nba
    · exact Or.inl h'
  · refine Or.inr ⟨lt_trans hab hbc, fun hca => ?_⟩
    rcases h.split c b a hca with h' | h'
    · exact ncb h'
    · exact nba h'

theorem rank_lt (h : StrictWeak scores) (i : Fin E) : rank scores i < E := by
  unfold rank
  calc _ < (Finset.univ : Finset (Fin E)).card :=
        Finset.card_lt_card (Finset.filter_ssubset.mpr ⟨i, Finset.mem_univ i, by
          simp [beats_self h]⟩)
    _ = E := by simp

/-- An expert that beats another has the smaller rank. -/
theorem rank_lt_rank (h : StrictWeak scores) {i i' : Fin E} (hb : beats scores i i' = true) :
    rank scores i < rank scores i' := by
  unfold rank
  have hsub : (Finset.univ.filter fun j => beats scores j i = true) ⊆
      Finset.univ.filter fun j => beats scores j i' = true := by
    intro j hj
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj ⊢
    exact beats_trans h hj hb
  exact Finset.card_lt_card
    ((Finset.ssubset_iff_of_subset hsub).mpr ⟨i, by simp [hb], by simp [beats_self h]⟩)

/-- Different experts have different ranks, so the ranks are `0, …, E - 1` in some order. -/
theorem rank_injective (h : StrictWeak scores) : Function.Injective (rank scores) := by
  intro i i' heq
  by_contra hne
  rcases beats_total scores hne with hb | hb
  · exact (rank_lt_rank h hb).ne heq
  · exact (rank_lt_rank h hb).ne heq.symm

private theorem card_filter_val_lt (E k : ℕ) :
    (Finset.univ.filter fun r : Fin E => r.val < k).card = min k E := by
  rw [← Finset.card_range (min k E), ← Finset.card_map Fin.valEmbedding]
  congr 1
  ext x
  simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Fin.valEmbedding_apply,
    Finset.mem_range, lt_min_iff]
  constructor
  · rintro ⟨r, hr, rfl⟩
    exact ⟨hr, r.isLt⟩
  · rintro ⟨hk, hE⟩
    exact ⟨⟨x, hE⟩, hk, rfl⟩

/-- Routing keeps exactly `min k E` experts. -/
theorem card_kept (h : StrictWeak scores) (k : ℕ) :
    (Finset.univ.filter fun i => kept k scores i = true).card = min k E := by
  let r : Fin E → Fin E := fun i => ⟨rank scores i, rank_lt h i⟩
  have hr : Function.Bijective r :=
    Finite.injective_iff_bijective.mp fun i i' hii' => rank_injective h (congrArg Fin.val hii')
  rw [← card_filter_val_lt E k]
  exact Finset.card_equiv (Equiv.ofBijective r hr) fun i => by simp [kept, r]

/-- The list of kept experts has `min k E` entries. -/
theorem length_keptExperts (h : StrictWeak scores) (k : ℕ) :
    (keptExperts k scores).length = min k E := by
  rw [← card_kept h k, keptExperts,
    ← List.toFinset_card_of_nodup ((List.nodup_finRange E).filter _)]
  congr 1
  ext i
  simp

/-- Every kept expert beats every expert that is not kept. -/
theorem beats_of_kept (h : StrictWeak scores) {k : ℕ} {i i' : Fin E}
    (hi : kept k scores i = true) (hi' : kept k scores i' = false) : beats scores i i' = true := by
  simp only [kept, decide_eq_true_eq, decide_eq_false_iff_not, not_lt] at hi hi'
  have hne : i ≠ i' := by
    rintro rfl
    omega
  rcases beats_total scores hne with hb | hb
  · exact hb
  · exact absurd (rank_lt_rank h hb) (by omega)

/-- Over `ℝ`: a kept expert's score is at least that of every expert it displaces, and larger
unless the displaced expert has the larger index. -/
theorem kept_dominates {scores : Fin E → ℝ} {k : ℕ} {i i' : Fin E}
    (hi : kept k scores i = true) (hi' : kept k scores i' = false) :
    scores i' < scores i ∨ (scores i' = scores i ∧ i < i') := by
  rcases (beats_eq_true scores i i').mp (beats_of_kept (strictWeak_real scores) hi hi') with
    h | ⟨hlt, hle⟩
  · exact Or.inl h
  · rcases lt_or_ge (scores i') (scores i) with h | h
    · exact Or.inl h
    · exact Or.inr ⟨le_antisymm (not_lt.mp hle) h, hlt⟩

/-- Two score vectors that compare alike keep the same experts. -/
theorem kept_congr {β : Type} [Context β] {t : Fin E → β}
    (h : ∀ i j, t i > t j ↔ scores i > scores j) (k : ℕ) : kept k t = kept k scores := by
  have hb : beats t = beats scores := by
    funext j i
    apply Bool.eq_iff_iff.mpr
    rw [beats_eq_true, beats_eq_true, h, h]
  have hr : rank t = rank scores := by
    funext i
    unfold rank
    congr 1
    ext j
    simp only [Finset.mem_filter, hb]
  funext i
  simp only [kept, hr]

end Routing

/-! ## Gates over the reals -/

section Gates

variable {E : ℕ}

/-- Over `ℝ` the softmax preserves every comparison, so routing on the probabilities keeps the
experts with the largest logits. -/
theorem kept_softmax (k : ℕ) (logits : Tensor ℝ [E]) :
    kept k (fun e => (Activation.softmaxVecSpec logits).getScalar e) =
      kept k fun e => logits.getScalar e := by
  cases E with
  | zero =>
      funext i
      exact i.elim0
  | succ n =>
      apply kept_congr
      intro i j
      rw [Proofs.getScalar_softmaxVecSpec_eq_exp_div, Proofs.getScalar_softmaxVecSpec_eq_exp_div]
      have hZ : 0 < ∑ j, Real.exp (logits.getScalar j) :=
        Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty
      rw [gt_iff_lt, gt_iff_lt, div_lt_div_iff_of_pos_right hZ, Real.exp_lt_exp]

private theorem getScalar_topKMaskVecSpec (k : ℕ) (scores : Tensor ℝ [E]) (e : Fin E) :
    (topKMaskVecSpec k scores).getScalar e =
      if kept k (fun j => scores.getScalar j) e then 1 else 0 := by
  simp [topKMaskVecSpec]

/-- The kept experts as a finite set. -/
noncomputable abbrev keptSet (k : ℕ) (scores : Fin E → ℝ) : Finset (Fin E) :=
  Finset.univ.filter fun e => kept k scores e = true

/-- Over `ℝ`, the gate of expert `e` is its probability divided by the total probability of the
kept experts, or zero when `e` is not kept. -/
theorem getScalar_gatesVecSpec (k : ℕ) (logits : Tensor ℝ [E]) (e : Fin E) :
    let p := fun j => (Activation.softmaxVecSpec logits).getScalar j
    (gatesVecSpec k logits).getScalar e =
      if kept k p e then p e / ∑ j ∈ keptSet k p, p j else 0 := by
  intro p
  have hsum : TorchLean.Tensor.sumSpec
      (mulSpec (Activation.softmaxVecSpec logits)
        (topKMaskVecSpec k (Activation.softmaxVecSpec logits))) = ∑ j ∈ keptSet k p, p j := by
    rw [Spec.sum_spec_vec, Finset.sum_filter]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [mulSpec, TorchLean.Tensor.getScalar_map2Spec, getScalar_topKMaskVecSpec]
    split <;> simp [p]
  simp only [gatesVecSpec]
  rw [hsum]
  simp only [divSpec, TorchLean.Tensor.getScalar_map2Spec, mulSpec, getScalar_topKMaskVecSpec]
  simp only [Spec.replicate, TorchLean.Tensor.getScalar_const, Tensor.item_scalar]
  split <;> simp [p]

/-- Over `ℝ`, the gates are the softmax of the logits of the kept experts, the experts with the
largest logits. -/
theorem getScalar_gatesVecSpec_softmax (k : ℕ) (logits : Tensor ℝ [E]) (e : Fin E) :
    let l := fun j => logits.getScalar j
    (gatesVecSpec k logits).getScalar e =
      if kept k l e then Real.exp (l e) / ∑ j ∈ keptSet k l, Real.exp (l j) else 0 := by
  intro l
  have hset :
      keptSet k (fun j => (Activation.softmaxVecSpec logits).getScalar j) = keptSet k l := by
    simp only [keptSet, kept_softmax, l]
  rw [getScalar_gatesVecSpec, hset, kept_softmax]
  cases E with
  | zero => exact e.elim0
  | succ n =>
      split
      · simp only [Proofs.getScalar_softmaxVecSpec_eq_exp_div, ← Finset.sum_div]
        have hZ : (∑ j, Real.exp (logits.getScalar j)) ≠ 0 :=
          (Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty).ne'
        rw [div_div_div_cancel_right₀ hZ]
      · rfl

/-- Over `ℝ`, with at least one expert kept, the gates are nonnegative and sum to one. -/
theorem gatesVecSpec_simplex (k : ℕ) (hk : 0 < k) (hE : 0 < E) (logits : Tensor ℝ [E]) :
    (∀ e, 0 ≤ (gatesVecSpec k logits).getScalar e) ∧
      ∑ e, (gatesVecSpec k logits).getScalar e = 1 := by
  set l := fun j => logits.getScalar j
  have hpos : 0 < ∑ j ∈ keptSet k l, Real.exp (l j) := by
    apply Finset.sum_pos (fun _ _ => Real.exp_pos _)
    rw [← Finset.card_pos, card_kept (strictWeak_real l)]
    exact lt_min hk hE
  refine ⟨fun e => ?_, ?_⟩
  · rw [getScalar_gatesVecSpec_softmax]
    split
    · exact div_nonneg (Real.exp_pos _).le hpos.le
    · exact le_rfl
  · simp_rw [getScalar_gatesVecSpec_softmax]
    rw [← Finset.sum_filter, ← Finset.sum_div]
    exact div_self hpos.ne'

end Gates

/-! ## The dense layer is the sparse block -/

section Dense

variable {E : ℕ} {s : Shape}

private theorem foldl_add_apply {ι : Type} (f : ι → Tensor ℝ s) (l : List ι) (a : Tensor ℝ s)
    (c : s.Coord) :
    (l.foldl (fun acc e => addSpec acc (f e)) a) c = a c + (l.map fun e => f e c).sum := by
  induction l generalizing a with
  | nil => simp
  | cons e l ih =>
      rw [List.foldl_cons, ih]
      simp [addSpec, add_assoc]

private theorem sum_map_filter {ι : Type} (q : ι → Bool) (f : ι → ℝ) (l : List ι) :
    ((l.filter q).map f).sum = (l.map fun e => if q e then f e else 0).sum := by
  induction l with
  | nil => rfl
  | cons e l ih =>
      by_cases he : q e <;> simp [he, ih]

private theorem sum_finRange (f : Fin E → ℝ) : ((List.finRange E).map f).sum = ∑ e, f e :=
  (Fin.sum_univ_def f).symm

/-- One coordinate of the combined output, over `ℝ`. -/
theorem combineSpec_apply (gates : Tensor ℝ [E]) (outputs : Fin E → Tensor ℝ s) (c : s.Coord) :
    combineSpec gates outputs c = ∑ e, gates.getScalar e * outputs e c := by
  rw [combineSpec, foldl_add_apply, sum_finRange]
  simp [mulSpec]

private theorem foldl_add_scalar {ι : Type} (f : ι → ℝ) (l : List ι) (a : ℝ) :
    l.foldl (fun acc e => acc + f e) a = a + (l.map f).sum := by
  induction l generalizing a with
  | nil => simp
  | cons e l ih =>
      rw [List.foldl_cons, ih]
      simp [add_assoc]

/-- Over `ℝ`, the dense layer computes exactly Mixtral's sparse block. -/
theorem forwardSpec_eq_sparseForwardSpec {σ : Shape} (k : ℕ)
    (router : Tensor ℝ σ → Tensor ℝ [E]) (experts : Fin E → Tensor ℝ σ → Tensor ℝ s)
    (x : Tensor ℝ σ) :
    forwardSpec k router experts x = sparseForwardSpec k router experts x := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro c
  let p := fun j => (Activation.softmaxVecSpec (router x)).getScalar j
  have htotal : (keptExperts k p).foldl (fun acc e => acc + p e) 0 = ∑ j ∈ keptSet k p, p j := by
    rw [foldl_add_scalar, zero_add, keptExperts, sum_map_filter, sum_finRange, Finset.sum_filter]
  show combineSpec (gatesVecSpec k (router x)) (fun e => experts e x) c =
    ((keptExperts k p).foldl
      (fun acc e => addSpec acc (mulSpec (experts e x) (replicate (Tensor.scalar
        (p e / (keptExperts k p).foldl (fun acc e => acc + p e) 0)))))
      (replicate (Tensor.scalar 0))) c
  rw [combineSpec_apply, foldl_add_apply, htotal, keptExperts, sum_map_filter, sum_finRange]
  simp only [replicate_apply, Tensor.item_scalar, zero_add]
  refine Finset.sum_congr rfl fun e _ => ?_
  rw [getScalar_gatesVecSpec]
  split <;> simp [mulSpec, replicate_apply, p, mul_comm]

end Dense

/-! ## Capacity and balancing -/

section Capacity

/-- Which tokens of `group`, each given by the expert it is routed to, a capacity of `C` tokens
per expert keeps: a token is kept when fewer than `C` earlier tokens of the group went to its
expert. -/
def capacityKeeps {E : ℕ} (C : ℕ) (group : List (Fin E)) : Array Bool :=
  (group.mapIdx fun t e => decide ((group.take t).count e < C)).toArray

/-- The capacity Switch gives each expert for a group of `n` tokens and `E` experts with capacity
factor one, `⌈n / E⌉`. -/
def switchCapacity (E n : ℕ) : ℕ := (n + E - 1) / E

/-- Two tokens routed to the first of two experts. On one device the capacity is one, so the
second token is dropped; split over two devices, each device has capacity one and keeps its
token. -/
theorem capacity_depends_on_devices :
    capacityKeeps (switchCapacity 2 2) ([0, 0] : List (Fin 2)) = #[true, false] ∧
      capacityKeeps (switchCapacity 2 1) ([0] : List (Fin 2)) ++
        capacityKeeps (switchCapacity 2 1) ([0] : List (Fin 2)) = #[true, true] := by
  decide

/-- Switch's balancing loss on a group of tokens, each given by its expert and the router's
probabilities: `E` times the sum over experts of the fraction of tokens sent to the expert times
its mean probability. -/
def switchBalance {E : ℕ} (group : List (Fin E × (Fin E → ℚ))) : ℚ :=
  E * ∑ e, ((group.countP fun t => t.1 = e : ℕ) / group.length : ℚ) *
    ((group.map fun t => t.2 e).sum / group.length)

/-- Two tokens, each sent to the expert it prefers with probability `3/4`. On one device the
balancing loss is `1`; split over two devices, each device's loss is `3/2`, and so is their
mean. -/
theorem balance_depends_on_devices :
    let a : Fin 2 × (Fin 2 → ℚ) := (0, ![3 / 4, 1 / 4])
    let b : Fin 2 × (Fin 2 → ℚ) := (1, ![1 / 4, 3 / 4])
    switchBalance [a, b] = 1 ∧ switchBalance [a] = 3 / 2 ∧ switchBalance [b] = 3 / 2 := by
  norm_num [switchBalance, Fin.sum_univ_two]

end Capacity

end NN.Proofs.Models.MoE
