/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.FactorizationsDecision
public import NN.Proofs.Tensor.Basic.FactorizationsOrthonormal
public import Mathlib.Probability.Distributions.Gaussian.Real
public import Mathlib.MeasureTheory.Constructions.Pi
public import Mathlib.MeasureTheory.Measure.Map
public import Mathlib.MeasureTheory.Measure.Typeclasses.Probability

/-!
# CHD `Z_test`: the distributional layer (`interpolatory.py`)

[`FactorizationsDecision`](./FactorizationsDecision.lean) proved the `Z_test` thresholds are
*well-posed* — each of `Z_low`/`Z_high` is a genuine order statistic of the per-sample `noise`,
lying in `[0,1]`, with `Z_low ≤ Z_high`, and the chooser consumes `Z_low`. That is the
*deterministic* half of the test. This file closes the **distributional** half, in two pieces that
are honestly provable over Mathlib v4.30.0:

* **Finite-sample calibration (counting).** The operational meaning of "`Z_low` is the 5th
  percentile" is that the threshold's *own* empirical false-positive rate is controlled: among the
  `N` null draws, **at most `⌊N/20⌋ ≈ 5%`** score strictly below `Z_low`
  (`zLow_null_exceedance_le`), and **at most `N-1-⌊19N/20⌋ ≈ 5%`** score strictly above `Z_high`
  (`zHigh_null_exceedance_le`). These are exact consequences of order-statistic sortedness — a
  sorted list has at most `k` entries below its `k`-th element — and need no probability theory.

* **The Gaussian null law (measure theory).** CHD draws the null samples i.i.d. standard Gaussian.
  We model that draw as `nullGaussian n`, the product of `n` standard normals on `Fin n → ℝ`
  (`Measure.pi (fun _ => gaussianReal 0 1)`), a genuine probability measure. The per-sample `noise`
  is a *measurable* map (`measurable_noiseMap`), so its **null law** `noiseLaw` is a probability
  measure (`IsProbabilityMeasure`) **supported in `[0,1]`** (`noiseLaw_Icc_eq_one`) — the verified
  `varNoiseFn ∈ [0,1]` bound, lifted to the law. `sampleNoisesFn_eq_noiseMap` identifies CHD's
  executable per-draw statistic with this measurable map, tying the counting layer to the measure.

Scope honesty: what remains genuinely *research-grade* (beyond Mathlib v4.30.0) is the
*asymptotic* calibration — that the empirical 5%/95% percentiles converge to the true quantiles of
`noiseLaw` (Glivenko–Cantelli / DKW), and that, under exchangeability of a fresh null draw with the
sample, the false-positive rate is exactly the rank level `k/(N+1)`. Those need an empirical-process
theory Mathlib does not yet carry; we do not stub them with `sorry`. The finite-sample false-positive
*bound* proved here is the exact, non-asymptotic statement the test actually guarantees.
-/

@[expose] public section

namespace Spec.Factorization

open Spec.Factorization.Reconstruction
open MeasureTheory ProbabilityTheory

variable {n : Nat}

/-! ## Finite-sample calibration: order-statistic tail counts

A sorted list has at most `k` entries strictly below its `k`-th element, and at most
`length - 1 - k` entries strictly above it. Pushed through the sort-is-a-permutation invariance of
`countP`, this bounds how many of the `N` null draws fall on the wrong side of a percentile
threshold — the test's empirical false-positive rate. -/

/-- In an ascending-sorted list, at most `k` entries are strictly below a cutoff `c ≤ s[k]`: every
entry from index `k` onward is `≥ s[k] ≥ c`, so all sub-`c` entries live in the length-`k` prefix. -/
private theorem sortedLE_countP_lt_le {s : List ℝ} (hs : s.SortedLE) {k : Nat}
    (hk : k < s.length) {c : ℝ} (hc : c ≤ s[k]) :
    s.countP (fun x => decide (x < c)) ≤ k := by
  conv_lhs => rw [← List.take_append_drop k s, List.countP_append]
  have hge : ∀ x ∈ s.drop k, c ≤ x := by
    intro x hx
    rw [List.mem_iff_getElem] at hx
    obtain ⟨i, hi, rfl⟩ := hx
    rw [List.getElem_drop]
    exact le_trans hc (hs.getElem_le_getElem_of_le (Nat.le_add_right k i))
  have hdrop : (s.drop k).countP (fun x => decide (x < c)) = 0 := by
    rw [List.countP_eq_zero]
    intro x hx
    simp only [decide_eq_true_eq]
    exact not_lt.mpr (hge x hx)
  have htake : (s.take k).countP (fun x => decide (x < c)) ≤ k := by
    refine le_trans List.countP_le_length ?_
    rw [List.length_take]; exact Nat.min_le_left _ _
  rw [hdrop, Nat.add_zero]; exact htake

/-- In an ascending-sorted list, at most `length - 1 - k` entries are strictly above a cutoff
`s[k] ≤ c`: every entry up to index `k` is `≤ s[k] ≤ c`, so all super-`c` entries live in the
length-`(length-(k+1))` suffix. -/
private theorem sortedLE_countP_gt_le {s : List ℝ} (hs : s.SortedLE) {k : Nat}
    (hk : k < s.length) {c : ℝ} (hc : s[k] ≤ c) :
    s.countP (fun x => decide (c < x)) ≤ s.length - 1 - k := by
  conv_lhs => rw [← List.take_append_drop (k + 1) s, List.countP_append]
  have htake : (s.take (k + 1)).countP (fun x => decide (c < x)) = 0 := by
    rw [List.countP_eq_zero]
    intro x hx
    rw [List.mem_iff_getElem] at hx
    obtain ⟨i, hi, rfl⟩ := hx
    have hi' : i < k + 1 := by
      rw [List.length_take] at hi; exact lt_of_lt_of_le hi (Nat.min_le_left _ _)
    rw [List.getElem_take]
    simp only [decide_eq_true_eq]
    exact not_lt.mpr (le_trans (hs.getElem_le_getElem_of_le (Nat.lt_succ_iff.mp hi')) hc)
  have hdrop : (s.drop (k + 1)).countP (fun x => decide (c < x)) ≤ s.length - (k + 1) := by
    refine le_trans List.countP_le_length ?_
    rw [List.length_drop]
  have heq : s.length - (k + 1) = s.length - 1 - k := by rw [Nat.sub_sub, Nat.add_comm]
  rw [htake, Nat.zero_add]
  exact le_trans hdrop (le_of_eq heq)

/-- The ascending-`(· ≤ ·)` mergeSort of the family `a`, whose `k`-th entry is `kthSmallestFn a k`. -/
private theorem kthSmallestFn_eq_getElem {N : Nat} (a : Fin N → ℝ) {k : Nat}
    (hks : k < (((List.finRange N).map a).mergeSort (· ≤ ·)).length) :
    Spec.kthSmallestFn a k = (((List.finRange N).map a).mergeSort (· ≤ ·))[k] := by
  rw [kthSmallestFn_eq_sorted_getD, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hks,
    Option.getD_some]

/-- **At most `k` of the family's values are strictly below its `k`-th order statistic.** Sorting is
a permutation (so `countP` is unchanged) and the sorted list has at most `k` entries below `s[k]`. -/
theorem kthSmallestFn_strictBelow_count_le {N : Nat} (a : Fin N → ℝ) {k : Nat} (hk : k < N) :
    ((List.finRange N).map a).countP (fun x => decide (x < Spec.kthSmallestFn a k)) ≤ k := by
  have hlen : (((List.finRange N).map a).mergeSort (· ≤ ·)).length = N := by
    rw [List.length_mergeSort, List.length_map, List.length_finRange]
  have hks : k < (((List.finRange N).map a).mergeSort (· ≤ ·)).length := by rw [hlen]; exact hk
  rw [kthSmallestFn_eq_getElem a hks,
    ← List.Perm.countP_eq _ (List.mergeSort_perm ((List.finRange N).map a) (· ≤ ·))]
  exact sortedLE_countP_lt_le List.sortedLE_mergeSort hks (le_refl _)

/-- **At most `N-1-k` of the family's values are strictly above its `k`-th order statistic.** -/
theorem kthSmallestFn_strictAbove_count_le {N : Nat} (a : Fin N → ℝ) {k : Nat} (hk : k < N) :
    ((List.finRange N).map a).countP (fun x => decide (Spec.kthSmallestFn a k < x)) ≤ N - 1 - k := by
  have hlen : (((List.finRange N).map a).mergeSort (· ≤ ·)).length = N := by
    rw [List.length_mergeSort, List.length_map, List.length_finRange]
  have hks : k < (((List.finRange N).map a).mergeSort (· ≤ ·)).length := by rw [hlen]; exact hk
  rw [kthSmallestFn_eq_getElem a hks,
    ← List.Perm.countP_eq _ (List.mergeSort_perm ((List.finRange N).map a) (· ≤ ·))]
  have hcount := sortedLE_countP_gt_le List.sortedLE_mergeSort hks (le_refl _)
  rw [hlen] at hcount
  exact hcount

/-! ### The `Z_test` empirical false-positive bounds -/

/-- **`Z_low` controls the lower-tail false-positive rate.** At most `⌊N/20⌋ ≈ 5%` of the `N` null
draws score strictly below the empirical `Z_low` threshold — exactly the rank that defines the 5th
percentile. This is the finite-sample, non-asymptotic guarantee the significance test carries. -/
theorem zLow_null_exceedance_le {n N : Nat} (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ)
    (samples : Fin N → Fin n → ℝ) (hN : 0 < N) :
    ((List.finRange N).map (Spec.sampleNoisesFn Λ V γ samples)).countP
        (fun x => decide (x < Spec.zLowFn Λ V γ samples)) ≤ Spec.zLowIdx N := by
  rw [Spec.zLowFn]
  exact kthSmallestFn_strictBelow_count_le _ (zLowIdx_lt hN)

/-- **`Z_high` controls the upper-tail false-positive rate.** At most `N-1-⌊19N/20⌋ ≈ 5%` of the `N`
null draws score strictly above the empirical `Z_high` threshold. -/
theorem zHigh_null_exceedance_le {n N : Nat} (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ)
    (samples : Fin N → Fin n → ℝ) (hN : 0 < N) :
    ((List.finRange N).map (Spec.sampleNoisesFn Λ V γ samples)).countP
        (fun x => decide (Spec.zHighFn Λ V γ samples < x)) ≤ N - 1 - Spec.zHighIdx N := by
  rw [Spec.zHighFn]
  exact kthSmallestFn_strictAbove_count_le _ (zHighIdx_lt hN)

/-! ## The Gaussian null law

CHD's `Z_test` draws each null sample i.i.d. standard Gaussian. We model one draw as
`nullGaussian n`: the product of `n` standard normals on `Fin n → ℝ`. The per-sample `noise` is a
measurable map, so its pushforward — the null law of the statistic — is a probability measure
concentrated on `[0,1]`. -/

noncomputable section

/-- The per-draw `noise` statistic as a map on raw draws `s : Fin n → ℝ` (one null sample):
`noiseMap Λ V γ s = varNoiseFn Λ γ (Vᵀ·s)`, the same functional `Z_test` scores each draw with. -/
noncomputable def noiseMap (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) : (Fin n → ℝ) → ℝ :=
  fun s => Spec.varNoiseFn Λ γ (Spec.projFn V s)

/-- CHD's executable per-draw null statistic is exactly `noiseMap` applied to that draw. This bridges
the counting layer (`sampleNoisesFn`) to the measure-theoretic model. -/
theorem sampleNoisesFn_eq_noiseMap {N : Nat} (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ)
    (samples : Fin N → Fin n → ℝ) (j : Fin N) :
    Spec.sampleNoisesFn Λ V γ samples j = noiseMap Λ V γ (samples j) := rfl

/-- A `dotFn` whose entries each depend measurably on a parameter is measurable in that parameter
(it is the finite sum `∑ₖ f k · g k`). -/
private theorem measurable_dotFn₂ {β : Type*} [MeasurableSpace β] {f g : β → Fin n → ℝ}
    (hf : ∀ k, Measurable (fun b => f b k)) (hg : ∀ k, Measurable (fun b => g b k)) :
    Measurable (fun b => Spec.dotFn (f b) (g b)) := by
  simp_rw [fun b => dotFn_eq_sum (f b) (g b)]
  exact Finset.measurable_sum _ (fun k _ => (hf k).mul (hg k))

/-- **The per-draw `noise` statistic is measurable.** It is a ratio of finite sums of products of the
(measurable) draw coordinates, hence Borel-measurable. -/
theorem measurable_noiseMap (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) :
    Measurable (noiseMap Λ V γ) := by
  have hproj : ∀ k, Measurable (fun s : Fin n → ℝ => Spec.projFn V s k) := fun k =>
    measurable_dotFn₂ (fun _ => measurable_const) (fun j => measurable_pi_apply j)
  have hpc : ∀ k, Measurable
      (fun s : Fin n → ℝ => Spec.projFn V s k * Spec.ridgeCoeffFn Λ γ k) := fun k =>
    (hproj k).mul measurable_const
  exact (measurable_dotFn₂ hpc hpc).div (measurable_dotFn₂ hpc hproj)

/-- The standard Gaussian draw of a single `Z_test` null sample: `n` i.i.d. standard normals on
`Fin n → ℝ`. A genuine probability measure (the product of probability measures). -/
noncomputable def nullGaussian (n : Nat) : Measure (Fin n → ℝ) :=
  Measure.pi (fun _ : Fin n => gaussianReal 0 1)

instance instIsProbabilityMeasureNullGaussian (n : Nat) : IsProbabilityMeasure (nullGaussian n) := by
  unfold nullGaussian; infer_instance

/-- The **null law** of the `Z_test` statistic: the pushforward of the standard-Gaussian draw under
the per-sample `noise`. This is the distribution `Z_low`/`Z_high` are percentiles of. -/
noncomputable def noiseLaw (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) : Measure ℝ :=
  (nullGaussian n).map (noiseMap Λ V γ)

/-- **The null law is a probability measure** (pushforward of one under a measurable map). -/
instance instIsProbabilityMeasureNoiseLaw (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) :
    IsProbabilityMeasure (noiseLaw Λ V γ) :=
  Measure.isProbabilityMeasure_map (measurable_noiseMap Λ V γ).aemeasurable

/-- **The null `noise` distribution lives entirely in `[0,1]`.** Every draw's statistic is in `[0,1]`
(the verified `varNoiseFn_nonneg`/`varNoiseFn_le_one`), so the law assigns full mass to `[0,1]` —
the percentiles `Z_low`/`Z_high` are therefore percentiles of a genuine `[0,1]`-valued random
variable. -/
theorem noiseLaw_Icc_eq_one {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ)
    (V : Fin n → Fin n → ℝ) :
    noiseLaw Λ V γ (Set.Icc 0 1) = 1 := by
  rw [noiseLaw, Measure.map_apply (measurable_noiseMap Λ V γ) measurableSet_Icc]
  have hpre : noiseMap Λ V γ ⁻¹' Set.Icc 0 1 = Set.univ := by
    ext s
    simp only [Set.mem_preimage, Set.mem_Icc, Set.mem_univ, iff_true]
    exact ⟨varNoiseFn_nonneg hΛ hγ _, varNoiseFn_le_one hΛ hγ _⟩
  rw [hpre, measure_univ]

end

end Spec.Factorization
