/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.FactorizationsZTest
public import Mathlib.Probability.Independence.InfinitePi
public import Mathlib.MeasureTheory.Integral.IntegrableOn
public import Mathlib.Probability.StrongLaw
public import Mathlib.Probability.CDF
public import Mathlib.MeasureTheory.Integral.Bochner.Set
public import Mathlib.Probability.Moments.SubGaussian

/-!
# CHD `Z_test`: asymptotic calibration — i.i.d. scaffold, empirical-CDF consistency, pointwise concentration and quantile transfer (steps a–d)

[`FactorizationsZTest`](./FactorizationsZTest.lean) modelled a *single* `Z_test` null draw as
`nullGaussian n` (the product of `n` standard normals on `Fin n → ℝ`) and proved the per-draw
`noise` statistic measurable, with null law `noiseLaw` a probability measure on `[0,1]`. That is
enough for the finite-sample false-positive bound, but the *asymptotic* calibration —
empirical 5%/95% percentiles converging to the true quantiles of `noiseLaw` — needs the whole
**i.i.d. sequence** of null draws, not one of them.

This file builds that sequence and proves it i.i.d.: the scaffold the asymptotic statements
(Glivenko–Cantelli via the SLLN, the Hoeffding per-`t` rate) are applications of. Concretely:

* **The sequence measure.** `nullSeqGaussian n := Measure.infinitePi (fun _ : ℕ => nullGaussian n)`
  on `ℕ → (Fin n → ℝ)` — countably many independent copies of one null draw, a genuine probability
  measure (`instIsProbabilityMeasureNullSeqGaussian`).

* **The `i`-th draw's statistic.** `nullNoise Λ V γ i ω := noiseMap Λ V γ (ω i)` — the same
  measurable `noiseMap` from `FactorizationsZTest`, read off the `i`-th coordinate.

* **i.i.d.** The coordinate evaluations are independent under the product measure, and composing
  with the measurable `noiseMap` preserves it (`nullNoise_iIndepFun`, and its pairwise corollary
  `nullNoise_pairwise_indepFun` in the exact shape `strong_law_ae_real` consumes). Each draw is
  measure-preservingly the same standard-Gaussian draw, so each has the *same* law `noiseLaw`
  (`nullNoise_hasLaw`, `nullNoise_identDistrib`). Every draw's noise lies in `[0,1]`
  (`nullNoise_mem_Icc`), hence is integrable (`integrable_nullNoise`).

So `nullNoise` is an i.i.d. real sequence, each with law `noiseLaw`, valued in `[0,1]` and
integrable — exactly the three hypotheses (`hint`/`hindep`/`hident`) the strong law of large
numbers and the Hoeffding tail take. This scaffold is the only genuinely *new* measure-theory
plumbing; the empirical-CDF consistency and concentration statements (steps b–d of the plan) are
applications of it, and the *uniform* Glivenko–Cantelli / DKW–Massart sharp constant and the
exchangeability rank rate remain genuinely research-grade (flagged, never `sorry`'d).

**Step (b) — pointwise consistency of the empirical CDF** is the first such application, proved
here. Fix a threshold `t`. The threshold indicators `nullBelow Λ V γ t i ω = 𝟙[nullNoise i ω ≤ t]`
inherit the i.i.d. structure (composition with the measurable indicator of `Iic t`), are
`[0,1]`-valued hence integrable, and have common mean `cdf (noiseLaw Λ V γ) t`
(`integral_nullBelow_zero`). The strong law (`strong_law_ae_real`, Etemadi's pairwise form) then
yields `empCDF_tendsto_cdf`: almost surely the empirical CDF `empCDF Λ V γ N t` converges to
`cdf (noiseLaw Λ V γ) t` as `N → ∞` — the pointwise Glivenko–Cantelli theorem. The *uniform*
(sup-norm over `t`) strengthening and the DKW rate are the remaining steps (c)–(d).

**Step (c) — pointwise finite-sample concentration (DKW-at-a-point via Hoeffding)** quantifies the
rate of that convergence at a fixed `t`. Each threshold indicator `nullBelow Λ V γ t i` is bounded in
`[0,1]`, so — once centered at its mean `cdf (noiseLaw Λ V γ) t` — it has a sub-Gaussian moment
generating function with variance proxy `1/4` (Hoeffding's lemma, `hasSubgaussianMGF_of_mem_Icc`).
Mathlib's Hoeffding inequality for sums of independent sub-Gaussians
(`HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun`) then gives, for every `N ≥ 1` and `ε ≥ 0`, the
one-sided tails `empCDF_upper_tail` / `empCDF_lower_tail`
`ℙ(±(empCDF Λ V γ N t − cdf (noiseLaw Λ V γ) t) ≥ ε) ≤ exp(−2·N·ε²)`, and their union the two-sided
`empCDF_concentration` `ℙ(|empCDF Λ V γ N t − cdf (noiseLaw Λ V γ) t| ≥ ε) ≤ 2·exp(−2·N·ε²)` — the
DKW inequality *at a single point* `t`, with the sharp Hoeffding exponent. This is the finite-sample
companion of step (b)'s almost-sure limit. The *uniform-over-`t`* DKW–Massart bound with the global
constant `2` (the genuine Dvoretzky–Kiefer–Wolfowitz theorem) is the research-grade strengthening
still flagged out of scope, and the quantile-transfer step (d) remains.

**Step (d) — quantile transfer (consistency of the empirical percentiles)** inverts steps (b)–(c):
it carries CDF convergence over to convergence of the empirical *quantiles* — the 5%/95% percentiles
the `Z_test` chooser thresholds against. Under the honest hypothesis that the true CDF is continuous
and strictly increasing through the target level `p` at the quantile `q` (`StraddlesQuantile`), the
classical sandwich — pointwise consistency (step (b)) at the two straddle points `q ∓ ε`
(`empCDF_eventually_straddle`) pinning any lower empirical `p`-quantile (`IsLowerQuantile`) into
`[q − ε, q + ε]`, intersected over `ε = 1/(m+1)` via `ae_all_iff` — gives `empQuantile_tendsto`:
almost surely `empQ N → q` as `N → ∞`. This is stated for a generic lower empirical `p`-quantile; the
concrete `zLowFn`/`zHighFn` order statistics instantiate it through the order-statistic count lemmas
with the moving level `p_N = (⌊N/20⌋ + 1)/N → 1/20`, the remaining concrete (triangular-array) bridge.
The *uniform* DKW–Massart sharp constant and the exchangeability rank rate stay research-grade.
-/

@[expose] public section

namespace Spec.Factorization

open MeasureTheory ProbabilityTheory

open scoped NNReal

variable {n : Nat}

noncomputable section

/-- The i.i.d. null-draw sequence: countably many independent standard-Gaussian draws, one per
`Z_test` null sample. The product of probability measures, hence itself a probability measure. -/
noncomputable def nullSeqGaussian (n : Nat) : Measure (ℕ → Fin n → ℝ) :=
  Measure.infinitePi (fun _ : ℕ => nullGaussian n)

instance instIsProbabilityMeasureNullSeqGaussian (n : Nat) :
    IsProbabilityMeasure (nullSeqGaussian n) := by
  unfold nullSeqGaussian; infer_instance

/-- The `i`-th null draw's `noise` statistic: `noiseMap` applied to the `i`-th coordinate of the
i.i.d. sequence. As `i` ranges over `ℕ` this is the i.i.d. real sequence the asymptotic calibration
runs on. -/
noncomputable def nullNoise (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) :
    ℕ → (ℕ → Fin n → ℝ) → ℝ :=
  fun i ω => noiseMap Λ V γ (ω i)

/-- Each draw's `noise` is measurable: the measurable `noiseMap` composed with a coordinate
projection. -/
theorem measurable_nullNoise (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (i : ℕ) :
    Measurable (nullNoise Λ V γ i) :=
  (measurable_noiseMap Λ V γ).comp (measurable_pi_apply i)

/-- **The null-noise sequence is independent.** The coordinate evaluations of the product measure
are independent (`iIndepFun_infinitePi`), and composing each with the measurable `noiseMap`
preserves independence. -/
theorem nullNoise_iIndepFun (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) :
    iIndepFun (nullNoise Λ V γ) (nullSeqGaussian n) :=
  iIndepFun_infinitePi (fun _ => measurable_noiseMap Λ V γ)

/-- The pairwise-independence corollary, in the exact `Pairwise (· ⟂ᵢ[μ] ·) on X` shape the strong
law of large numbers (`strong_law_ae_real`) consumes for its `hindep` hypothesis. -/
theorem nullNoise_pairwise_indepFun (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) :
    Pairwise (Function.onFun (· ⟂ᵢ[nullSeqGaussian n] ·) (nullNoise Λ V γ)) :=
  fun _ _ hij => (nullNoise_iIndepFun Λ V γ).indepFun hij

/-- **Each draw has the same law, `noiseLaw`.** The `i`-th coordinate projection is measure-
preserving from the product measure onto a single `nullGaussian n` draw, and composing with the
measurable `noiseMap` pushes that law forward to `noiseLaw` — independently of `i`. -/
theorem nullNoise_hasLaw (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (i : ℕ) :
    HasLaw (nullNoise Λ V γ i) (noiseLaw Λ V γ) (nullSeqGaussian n) := by
  have hEval := (measurePreserving_eval_infinitePi (fun _ : ℕ => nullGaussian n) i).hasLaw
  have hNoise : HasLaw (noiseMap Λ V γ) (noiseLaw Λ V γ) (nullGaussian n) :=
    { aemeasurable := (measurable_noiseMap Λ V γ).aemeasurable
      map_eq := rfl }
  exact hNoise.fun_comp hEval

/-- **The null-noise sequence is identically distributed.** Every draw has the common law
`noiseLaw`, so any two are identically distributed — the `hident` hypothesis of the strong law,
stated against the `0`-th draw. -/
theorem nullNoise_identDistrib (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (i : ℕ) :
    IdentDistrib (nullNoise Λ V γ i) (nullNoise Λ V γ 0) (nullSeqGaussian n) (nullSeqGaussian n) where
  aemeasurable_fst := (measurable_nullNoise Λ V γ i).aemeasurable
  aemeasurable_snd := (measurable_nullNoise Λ V γ 0).aemeasurable
  map_eq := by rw [(nullNoise_hasLaw Λ V γ i).map_eq, (nullNoise_hasLaw Λ V γ 0).map_eq]

/-- **Every draw's noise lies in `[0,1]`**, pointwise — the verified `varNoiseFn` bound applied to
each coordinate. -/
theorem nullNoise_mem_Icc {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ)
    (V : Fin n → Fin n → ℝ) (i : ℕ) (ω : ℕ → Fin n → ℝ) :
    nullNoise Λ V γ i ω ∈ Set.Icc (0 : ℝ) 1 :=
  Set.mem_Icc.mpr ⟨varNoiseFn_nonneg hΛ hγ _, varNoiseFn_le_one hΛ hγ _⟩

/-- **Each draw's noise is integrable** (bounded in `[0,1]` on the probability space) — the `hint`
hypothesis of the strong law. -/
theorem integrable_nullNoise {Λ : Fin n → ℝ} (hΛ : ∀ i, 0 ≤ Λ i) {γ : ℝ} (hγ : 0 < γ)
    (V : Fin n → Fin n → ℝ) (i : ℕ) :
    Integrable (nullNoise Λ V γ i) (nullSeqGaussian n) :=
  Integrable.of_bound (measurable_nullNoise Λ V γ i).aestronglyMeasurable 1
    (ae_of_all _ fun ω => by
      have h := Set.mem_Icc.mp (nullNoise_mem_Icc hΛ hγ V i ω)
      rw [Real.norm_eq_abs, abs_le]
      exact ⟨by linarith [h.1], h.2⟩)

/-! ## Step (b): pointwise consistency of the empirical CDF (Glivenko–Cantelli via the SLLN)

Fix a threshold `t`. The *threshold indicators* `nullBelow Λ V γ t i ω = 𝟙[nullNoise i ω ≤ t]` are,
like `nullNoise` itself, i.i.d. — composing each independent, identically-distributed draw with the
measurable indicator of `Iic t` preserves both — and `[0,1]`-valued, hence integrable. Their common
mean is exactly the CDF of the null law at `t`,
`∫ ω, nullBelow Λ V γ t 0 ω = (noiseLaw Λ V γ).real (Iic t) = cdf (noiseLaw Λ V γ) t`. The strong law
of large numbers (`strong_law_ae_real`, Etemadi's pairwise-independent form) then gives, almost
surely, `empCDF Λ V γ N t ω → cdf (noiseLaw Λ V γ) t` as `N → ∞`: pointwise consistency of the
empirical distribution function. -/

/-- The threshold indicator of the `i`-th null draw at level `t`: `1` if that draw's `noise` is
`≤ t`, else `0`. Normalized sums of these are the empirical CDF, and as an i.i.d. bounded sequence
they are the random variables the strong law runs on. -/
noncomputable def nullBelow (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) :
    ℕ → (ℕ → Fin n → ℝ) → ℝ :=
  fun i ω => (Set.Iic t).indicator (1 : ℝ → ℝ) (nullNoise Λ V γ i ω)

/-- The **empirical CDF** of the first `N` null draws at threshold `t`:
`F̂_N(t)(ω) = #{i < N : nullNoise i ω ≤ t} / N`, written as the normalized sum of threshold
indicators so it plugs directly into the strong law. -/
noncomputable def empCDF (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (N : ℕ) (t : ℝ)
    (ω : ℕ → Fin n → ℝ) : ℝ :=
  (∑ i ∈ Finset.range N, nullBelow Λ V γ t i ω) / (N : ℝ)

/-- Each threshold indicator is measurable: the measurable indicator of `Iic t` composed with the
measurable `nullNoise`. -/
theorem measurable_nullBelow (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) (i : ℕ) :
    Measurable (nullBelow Λ V γ t i) :=
  (measurable_const.indicator measurableSet_Iic).comp (measurable_nullNoise Λ V γ i)

/-- **The threshold-indicator sequence is pairwise independent** — composing each independent
`nullNoise` draw with the measurable indicator of `Iic t` preserves independence. The exact
`hindep` shape `strong_law_ae_real` consumes. -/
theorem nullBelow_pairwise_indepFun (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) :
    Pairwise (Function.onFun (· ⟂ᵢ[nullSeqGaussian n] ·) (nullBelow Λ V γ t)) := by
  intro i j hij
  exact ((nullNoise_iIndepFun Λ V γ).indepFun hij).comp
    (measurable_const.indicator measurableSet_Iic) (measurable_const.indicator measurableSet_Iic)

/-- **The threshold-indicator sequence is identically distributed** — each is the common `nullNoise`
law pushed through the same indicator. The `hident` hypothesis of the strong law. -/
theorem nullBelow_identDistrib (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) (i : ℕ) :
    IdentDistrib (nullBelow Λ V γ t i) (nullBelow Λ V γ t 0)
      (nullSeqGaussian n) (nullSeqGaussian n) :=
  (nullNoise_identDistrib Λ V γ i).comp (measurable_const.indicator measurableSet_Iic)

/-- **Each threshold indicator is integrable** — it is `[0,1]`-valued on a probability space. The
`hint` hypothesis of the strong law. -/
theorem integrable_nullBelow (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) (i : ℕ) :
    Integrable (nullBelow Λ V γ t i) (nullSeqGaussian n) :=
  Integrable.of_bound (measurable_nullBelow Λ V γ t i).aestronglyMeasurable 1
    (ae_of_all _ fun ω => by
      show ‖(Set.Iic t).indicator (1 : ℝ → ℝ) (nullNoise Λ V γ i ω)‖ ≤ 1
      refine le_trans (norm_indicator_le_norm_self _ _) ?_
      simp)

/-- **The common mean of the threshold indicators is the null CDF at `t`.** Pushing the indicator of
`Iic t` through the `0`-th draw's law `noiseLaw` (via `HasLaw.integral_comp`) turns the expectation
into `(noiseLaw Λ V γ).real (Iic t) = cdf (noiseLaw Λ V γ) t`. -/
theorem integral_nullBelow_zero (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) :
    (nullSeqGaussian n)[nullBelow Λ V γ t 0] = cdf (noiseLaw Λ V γ) t := by
  have hf : AEStronglyMeasurable ((Set.Iic t).indicator (1 : ℝ → ℝ)) (noiseLaw Λ V γ) :=
    (measurable_const.indicator measurableSet_Iic).aestronglyMeasurable
  have key := (nullNoise_hasLaw Λ V γ 0).integral_comp hf
  rw [integral_indicator_one measurableSet_Iic, ← cdf_eq_real] at key
  exact key

/-- **Pointwise consistency of the empirical CDF (pointwise Glivenko–Cantelli via the SLLN).** For
each fixed threshold `t`, almost surely the empirical CDF `empCDF` of the i.i.d. null draws converges
to the true CDF of the null law `noiseLaw` as the number of draws `N → ∞`. This is step (b) of the
asymptotic-calibration plan — the foundation under the 5%/95% percentile convergence, whose uniform
and concentration refinements are steps (c)–(d). -/
theorem empCDF_tendsto_cdf (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) :
    ∀ᵐ ω ∂(nullSeqGaussian n),
      Filter.Tendsto (fun N : ℕ => empCDF Λ V γ N t ω) Filter.atTop
        (nhds (cdf (noiseLaw Λ V γ) t)) := by
  have hlaw := strong_law_ae_real (nullBelow Λ V γ t)
    (integrable_nullBelow Λ V γ t 0)
    (nullBelow_pairwise_indepFun Λ V γ t)
    (fun i => nullBelow_identDistrib Λ V γ t i)
  rw [integral_nullBelow_zero] at hlaw
  exact hlaw

/-! ## Step (c): pointwise finite-sample concentration (DKW-at-a-point via Hoeffding)

Step (b) is an almost-sure *limit*; step (c) is its quantitative, finite-`N` companion. The threshold
indicators `nullBelow Λ V γ t i` are bounded in `[0,1]`, hence — centered at their common mean
`cdf (noiseLaw Λ V γ) t` — sub-Gaussian with variance proxy `(1/2)² = 1/4` (Hoeffding's lemma). Being
i.i.d., their normalized sum (the empirical CDF) concentrates exponentially: Mathlib's sub-Gaussian
Hoeffding inequality gives both one-sided tails with rate `exp(−2·N·ε²)` and, by a union bound, the
two-sided `2·exp(−2·N·ε²)` — the Dvoretzky–Kiefer–Wolfowitz bound *at the single point* `t`. -/

/-- The empirical CDF lies in `[0,1]` pointwise, so each threshold indicator does too. -/
theorem nullBelow_mem_Icc (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) (i : ℕ)
    (ω : ℕ → Fin n → ℝ) : nullBelow Λ V γ t i ω ∈ Set.Icc (0 : ℝ) 1 := by
  unfold nullBelow
  rw [Set.indicator_apply]
  split <;> simp [Set.mem_Icc]

/-- **The threshold indicators are jointly independent** (not just pairwise): composing the i.i.d.
`nullNoise` sequence with the measurable indicator of `Iic t` preserves joint independence. The shape
the Hoeffding sum bound consumes. -/
theorem nullBelow_iIndepFun (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) :
    iIndepFun (nullBelow Λ V γ t) (nullSeqGaussian n) :=
  (nullNoise_iIndepFun Λ V γ).comp (fun _ => (Set.Iic t).indicator (1 : ℝ → ℝ))
    (fun _ => measurable_const.indicator measurableSet_Iic)

/-- The common mean of the threshold indicators is the null CDF at `t`, for *every* draw `i` (not just
the `0`-th): identically distributed draws share their integral. -/
theorem integral_nullBelow_eq (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) (i : ℕ) :
    (nullSeqGaussian n)[nullBelow Λ V γ t i] = cdf (noiseLaw Λ V γ) t := by
  rw [(nullBelow_identDistrib Λ V γ t i).integral_eq, integral_nullBelow_zero]

/-- **Hoeffding's lemma for one threshold indicator.** Centered at its mean `cdf (noiseLaw Λ V γ) t`,
the `[0,1]`-valued indicator has a sub-Gaussian MGF with variance proxy `((1-0)/2)² = 1/4`. -/
theorem nullBelow_subgaussian (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) (i : ℕ) :
    HasSubgaussianMGF (fun ω => nullBelow Λ V γ t i ω - cdf (noiseLaw Λ V γ) t)
      (1 / 4 : ℝ≥0) (nullSeqGaussian n) := by
  have hb : ∀ᵐ ω ∂(nullSeqGaussian n), nullBelow Λ V γ t i ω ∈ Set.Icc (0 : ℝ) 1 :=
    ae_of_all _ (nullBelow_mem_Icc Λ V γ t i)
  have h := hasSubgaussianMGF_of_mem_Icc (measurable_nullBelow Λ V γ t i).aemeasurable hb
  rw [integral_nullBelow_eq] at h
  rwa [show ((‖(1 : ℝ) - 0‖₊) / 2) ^ 2 = (1 / 4 : ℝ≥0) from by
        rw [sub_zero, nnnorm_one]; norm_num] at h

/-- The mean of the *negated*, recentred indicator `cdf (noiseLaw Λ V γ) t − nullBelow` is `0`. -/
theorem integral_negBelow_eq (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) (i : ℕ) :
    (nullSeqGaussian n)[fun ω => cdf (noiseLaw Λ V γ) t - nullBelow Λ V γ t i ω] = 0 := by
  rw [integral_sub (integrable_const _) (integrable_nullBelow Λ V γ t i), integral_const,
    integral_nullBelow_eq]
  simp

/-- The negated indicator `cdf (noiseLaw Λ V γ) t − nullBelow` lies in `[cdf − 1, cdf]`. -/
theorem nullBelow_neg_mem_Icc (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) (i : ℕ)
    (ω : ℕ → Fin n → ℝ) :
    cdf (noiseLaw Λ V γ) t - nullBelow Λ V γ t i ω
      ∈ Set.Icc (cdf (noiseLaw Λ V γ) t - 1) (cdf (noiseLaw Λ V γ) t) := by
  have h := Set.mem_Icc.mp (nullBelow_mem_Icc Λ V γ t i ω)
  rw [Set.mem_Icc]
  constructor <;> linarith [h.1, h.2]

/-- **Hoeffding's lemma for the negated indicator.** `cdf (noiseLaw Λ V γ) t − nullBelow` is
`[cdf − 1, cdf]`-valued (length-`1` interval) and already mean-zero, so it is sub-Gaussian with the
same variance proxy `1/4` — the lower-tail companion of `nullBelow_subgaussian`. -/
theorem nullBelow_neg_subgaussian (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) (i : ℕ) :
    HasSubgaussianMGF (fun ω => cdf (noiseLaw Λ V γ) t - nullBelow Λ V γ t i ω)
      (1 / 4 : ℝ≥0) (nullSeqGaussian n) := by
  have hb : ∀ᵐ ω ∂(nullSeqGaussian n),
      (fun ω => cdf (noiseLaw Λ V γ) t - nullBelow Λ V γ t i ω) ω
        ∈ Set.Icc (cdf (noiseLaw Λ V γ) t - 1) (cdf (noiseLaw Λ V γ) t) :=
    ae_of_all _ (nullBelow_neg_mem_Icc Λ V γ t i)
  have hmeas : Measurable (fun ω => cdf (noiseLaw Λ V γ) t - nullBelow Λ V γ t i ω) :=
    measurable_const.sub (measurable_nullBelow Λ V γ t i)
  have h := hasSubgaussianMGF_of_mem_Icc hmeas.aemeasurable hb
  rw [integral_negBelow_eq] at h
  simp only [sub_zero] at h
  rwa [show ((‖cdf (noiseLaw Λ V γ) t - (cdf (noiseLaw Λ V γ) t - 1)‖₊) / 2) ^ 2 = (1 / 4 : ℝ≥0)
        from by rw [sub_sub_cancel, nnnorm_one]; norm_num] at h

/-- **Hoeffding's inequality for a normalized i.i.d. proxy-`1/4` sub-Gaussian sum.** If `X` is a
jointly independent sequence on the null-draw space, each centered draw sub-Gaussian with variance
proxy `1/4`, then for `N ≥ 1` and `ε ≥ 0` the empirical average `(∑_{i<N} X i)/N` exceeds `ε` with
probability at most `exp(−2·N·ε²)`. The engine under both `empCDF` tails — `ε ↦ N·ε` in Mathlib's
sum bound turns the proxy sum `N/4` into the sharp exponent `−2Nε²`. -/
theorem hoeffding_avg_ge {X : ℕ → (ℕ → Fin n → ℝ) → ℝ}
    (hindep : iIndepFun X (nullSeqGaussian n))
    (hsub : ∀ i, HasSubgaussianMGF (X i) (1 / 4 : ℝ≥0) (nullSeqGaussian n))
    {N : ℕ} (hN : 1 ≤ N) {ε : ℝ} (hε : 0 ≤ ε) :
    (nullSeqGaussian n).real {ω | ε ≤ (∑ i ∈ Finset.range N, X i ω) / (N : ℝ)}
      ≤ Real.exp (-2 * (N : ℝ) * ε ^ 2) := by
  have hNR : (0 : ℝ) < N := by exact_mod_cast hN
  have hbase := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hindep
    (c := fun _ => (1 / 4 : ℝ≥0)) (s := Finset.range N) (fun i _ => hsub i)
    (ε := (N : ℝ) * ε) (by positivity)
  have hset : {ω | (N : ℝ) * ε ≤ ∑ i ∈ Finset.range N, X i ω}
      = {ω | ε ≤ (∑ i ∈ Finset.range N, X i ω) / (N : ℝ)} := by
    ext ω
    simp only [Set.mem_setOf_eq]
    rw [le_div_iff₀ hNR, mul_comm]
  rw [hset] at hbase
  refine hbase.trans (le_of_eq ?_)
  congr 1
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  push_cast
  field_simp
  ring

/-- **Upper-tail concentration of the empirical CDF (Hoeffding).** For `N ≥ 1`, `ε ≥ 0`, the empirical
CDF overshoots the true null CDF at `t` by `ε` with probability `≤ exp(−2·N·ε²)`. -/
theorem empCDF_upper_tail (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) {N : ℕ}
    (hN : 1 ≤ N) {ε : ℝ} (hε : 0 ≤ ε) :
    (nullSeqGaussian n).real {ω | ε ≤ empCDF Λ V γ N t ω - cdf (noiseLaw Λ V γ) t}
      ≤ Real.exp (-2 * (N : ℝ) * ε ^ 2) := by
  have hNR : (0 : ℝ) < N := by exact_mod_cast hN
  have hind : iIndepFun (fun i ω => nullBelow Λ V γ t i ω - cdf (noiseLaw Λ V γ) t)
      (nullSeqGaussian n) :=
    (nullBelow_iIndepFun Λ V γ t).comp (fun _ x => x - cdf (noiseLaw Λ V γ) t)
      (fun _ => measurable_id.sub_const _)
  have htail := hoeffding_avg_ge hind (fun i => nullBelow_subgaussian Λ V γ t i) hN hε
  have hrw : ∀ ω,
      (∑ i ∈ Finset.range N, (nullBelow Λ V γ t i ω - cdf (noiseLaw Λ V γ) t)) / (N : ℝ)
        = empCDF Λ V γ N t ω - cdf (noiseLaw Λ V γ) t := by
    intro ω
    unfold empCDF
    rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, sub_div,
      mul_div_cancel_left₀ (cdf (noiseLaw Λ V γ) t) (ne_of_gt hNR)]
  simp only [hrw] at htail
  exact htail

/-- **Lower-tail concentration of the empirical CDF (Hoeffding).** Symmetrically, the empirical CDF
undershoots the true null CDF at `t` by `ε` with probability `≤ exp(−2·N·ε²)`. -/
theorem empCDF_lower_tail (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) {N : ℕ}
    (hN : 1 ≤ N) {ε : ℝ} (hε : 0 ≤ ε) :
    (nullSeqGaussian n).real {ω | ε ≤ cdf (noiseLaw Λ V γ) t - empCDF Λ V γ N t ω}
      ≤ Real.exp (-2 * (N : ℝ) * ε ^ 2) := by
  have hNR : (0 : ℝ) < N := by exact_mod_cast hN
  have hind : iIndepFun (fun i ω => cdf (noiseLaw Λ V γ) t - nullBelow Λ V γ t i ω)
      (nullSeqGaussian n) :=
    (nullBelow_iIndepFun Λ V γ t).comp (fun _ x => cdf (noiseLaw Λ V γ) t - x)
      (fun _ => measurable_const.sub measurable_id)
  have htail := hoeffding_avg_ge hind (fun i => nullBelow_neg_subgaussian Λ V γ t i) hN hε
  have hrw : ∀ ω,
      (∑ i ∈ Finset.range N, (cdf (noiseLaw Λ V γ) t - nullBelow Λ V γ t i ω)) / (N : ℝ)
        = cdf (noiseLaw Λ V γ) t - empCDF Λ V γ N t ω := by
    intro ω
    unfold empCDF
    rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul, sub_div,
      mul_div_cancel_left₀ (cdf (noiseLaw Λ V γ) t) (ne_of_gt hNR)]
  simp only [hrw] at htail
  exact htail

/-- **Pointwise finite-sample concentration of the empirical CDF (DKW-at-a-point, step (c)).** For
each fixed threshold `t`, every `N ≥ 1` and tolerance `ε ≥ 0`, the empirical CDF of the i.i.d. null
draws deviates from the true null CDF by more than `ε` with probability at most `2·exp(−2·N·ε²)`.
This is the Dvoretzky–Kiefer–Wolfowitz inequality evaluated at a single point, with the sharp
Hoeffding exponent — the finite-sample rate underneath step (b)'s almost-sure limit. The
*uniform-over-`t`* DKW–Massart bound (global constant `2`) is the research-grade strengthening still
flagged out of scope. -/
theorem empCDF_concentration (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (t : ℝ) {N : ℕ}
    (hN : 1 ≤ N) {ε : ℝ} (hε : 0 ≤ ε) :
    (nullSeqGaussian n).real {ω | ε ≤ |empCDF Λ V γ N t ω - cdf (noiseLaw Λ V γ) t|}
      ≤ 2 * Real.exp (-2 * (N : ℝ) * ε ^ 2) := by
  have hsplit : {ω | ε ≤ |empCDF Λ V γ N t ω - cdf (noiseLaw Λ V γ) t|}
      = {ω | ε ≤ empCDF Λ V γ N t ω - cdf (noiseLaw Λ V γ) t}
        ∪ {ω | ε ≤ cdf (noiseLaw Λ V γ) t - empCDF Λ V γ N t ω} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_union, le_abs, neg_sub]
  rw [hsplit]
  refine (measureReal_union_le _ _).trans ?_
  have h1 := empCDF_upper_tail Λ V γ t hN hε
  have h2 := empCDF_lower_tail Λ V γ t hN hε
  linarith

/-! ## Step (d): quantile transfer — consistency of the empirical percentiles

Steps (b)–(c) control the empirical CDF at a *fixed* threshold. Step (d) *inverts* that: it transfers
the convergence of `empCDF` to the convergence of the empirical *quantiles* — the 5%/95% percentiles
`Z_low`/`Z_high` the `Z_test` chooser actually thresholds against. The honest hypothesis under which
this works is that the true CDF is continuous and strictly increasing through the target level `p` at
the quantile `q`, captured by `StraddlesQuantile`: the CDF sits strictly below `p` to the left of `q`
and strictly above `p` to the right.

The argument is the classical sandwich. For any tolerance `ε > 0`, the straddle gives
`cdf (q − ε) < p < cdf (q + ε)`. Pointwise consistency (step (b)) at the two points `q ∓ ε` then says
that almost surely, eventually `empCDF (q − ε) < p < empCDF (q + ε)`. Any *lower empirical
`p`-quantile* `empQ` (CDF strictly below `p` to its left, at least `p` to its right —
`IsLowerQuantile`) is therefore pinned into `[q − ε, q + ε]` once the sandwich holds. Letting `ε` run
over `1/(m+1)` and intersecting the countably many almost-sure events (`ae_all_iff`) yields, almost
surely, `empQ N → q` as `N → ∞`: **consistency of the empirical quantile**.

This is stated for a *generic* lower empirical `p`-quantile `empQ`; the concrete percentile order
statistics `zLowFn`/`zHighFn` instantiate it through the order-statistic count lemmas
(`kthSmallestFn_strictBelow_count_le` / `kthSmallestFn_strictAbove_count_le`), with the index-driven
level `p_N = (⌊N/20⌋ + 1)/N → 1/20` — a triangular-array (moving-level) refinement that is the
remaining concrete bridge, while the *uniform* DKW–Massart sharp constant and the exchangeability rank
rate stay research-grade and out of scope (flagged, never `sorry`'d). -/

/-- **The empirical CDF is monotone in the threshold.** Raising `t` only enlarges `Iic t`, so each
threshold indicator (hence their normalized sum) is nondecreasing — the empirical CDF behaves like a
genuine distribution function in its argument. -/
theorem empCDF_mono (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) (N : ℕ) (ω : ℕ → Fin n → ℝ) :
    Monotone (fun t => empCDF Λ V γ N t ω) := by
  intro t t' htt'
  have hsum : ∑ i ∈ Finset.range N, nullBelow Λ V γ t i ω
      ≤ ∑ i ∈ Finset.range N, nullBelow Λ V γ t' i ω :=
    Finset.sum_le_sum fun i _ =>
      Set.indicator_le_indicator_of_subset (Set.Iic_subset_Iic.mpr htt')
        (fun _ => zero_le_one) (nullNoise Λ V γ i ω)
  simp only [empCDF]
  exact div_le_div_of_nonneg_right hsum (by positivity)

/-- **Population `p`-quantile (continuous, strictly-increasing-through-`p` sense).** `q` straddles
level `p` for the CDF `F` when `F` sits strictly below `p` just left of `q` and strictly above just
right. This holds whenever `F` is continuous and strictly monotone at `q` with `F q = p` — the honest
hypothesis the empirical quantile is consistent under. -/
def StraddlesQuantile (F : ℝ → ℝ) (p q : ℝ) : Prop :=
  ∀ ε : ℝ, 0 < ε → F (q - ε) < p ∧ p < F (q + ε)

/-- **Lower empirical `p`-quantile.** `q` is a lower `p`-quantile of the distribution function `F`
when `F` is strictly below `p` to the left of `q` and at least `p` to the right — exactly
`inf {t | p ≤ F t}` for a right-continuous step CDF. The defining property the order-statistic
percentiles satisfy. -/
def IsLowerQuantile (F : ℝ → ℝ) (p q : ℝ) : Prop :=
  (∀ t, t < q → F t < p) ∧ (∀ t, q < t → p ≤ F t)

/-- **Quantile sandwich (the transfer engine).** If the true null CDF straddles level `p` strictly
across `t₁ < t₂` (`cdf t₁ < p < cdf t₂`), then — by pointwise consistency (step (b)) at the two
points — almost surely the empirical CDF eventually straddles `p` the same way:
`empCDF N t₁ < p < empCDF N t₂` for all large `N`. -/
theorem empCDF_eventually_straddle (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) {p t₁ t₂ : ℝ}
    (h1 : cdf (noiseLaw Λ V γ) t₁ < p) (h2 : p < cdf (noiseLaw Λ V γ) t₂) :
    ∀ᵐ ω ∂(nullSeqGaussian n), ∀ᶠ N in Filter.atTop,
      empCDF Λ V γ N t₁ ω < p ∧ p < empCDF Λ V γ N t₂ ω := by
  filter_upwards [empCDF_tendsto_cdf Λ V γ t₁, empCDF_tendsto_cdf Λ V γ t₂] with ω hω1 hω2
  filter_upwards [hω1.eventually_lt_const h1, hω2.eventually_const_lt h2] with N hN1 hN2
  exact ⟨hN1, hN2⟩

/-- **Consistency of the empirical quantile (quantile transfer, step (d)).** Fix a target level `p`
and a population quantile `q` straddled by the true null CDF. Then for *any* lower empirical
`p`-quantile `empQ` of the empirical CDF (e.g. the percentile order statistics), almost surely
`empQ N → q` as the number of null draws `N → ∞`. This is the honest consistency statement for the
5%/95% thresholds the `Z_test` chooser uses, inverting steps (b)–(c)'s CDF convergence into quantile
convergence wherever the CDF is continuous and strictly monotone at the quantile. -/
theorem empQuantile_tendsto (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (γ : ℝ) {p q : ℝ}
    {empQ : ℕ → (ℕ → Fin n → ℝ) → ℝ}
    (hstr : StraddlesQuantile (cdf (noiseLaw Λ V γ)) p q)
    (hq : ∀ N ω, IsLowerQuantile (fun t => empCDF Λ V γ N t ω) p (empQ N ω)) :
    ∀ᵐ ω ∂(nullSeqGaussian n),
      Filter.Tendsto (fun N => empQ N ω) Filter.atTop (nhds q) := by
  have key : ∀ m : ℕ, ∀ᵐ ω ∂(nullSeqGaussian n),
      ∀ᶠ N in Filter.atTop, |empQ N ω - q| ≤ 1 / (m + 1 : ℝ) := by
    intro m
    have hε : (0 : ℝ) < 1 / (m + 1 : ℝ) := by positivity
    obtain ⟨hlt, hgt⟩ := hstr _ hε
    filter_upwards [empCDF_eventually_straddle Λ V γ hlt hgt] with ω hω
    filter_upwards [hω] with N hN
    obtain ⟨hN1, hN2⟩ := hN
    obtain ⟨hqL, hqR⟩ := hq N ω
    have hub : empQ N ω ≤ q + 1 / (m + 1 : ℝ) := by
      by_contra hc
      exact absurd (hqL _ (not_le.mp hc)) (not_lt.mpr hN2.le)
    have hlb : q - 1 / (m + 1 : ℝ) ≤ empQ N ω := by
      by_contra hc
      exact absurd (hqR _ (not_le.mp hc)) (not_le.mpr hN1)
    rw [abs_le]
    constructor <;> linarith
  filter_upwards [ae_all_iff.mpr key] with ω hω
  rw [Metric.tendsto_atTop]
  intro δ hδ
  obtain ⟨m, hm⟩ := exists_nat_one_div_lt hδ
  obtain ⟨N₀, hN₀⟩ := Filter.eventually_atTop.mp (hω m)
  refine ⟨N₀, fun N hN => ?_⟩
  rw [Real.dist_eq]
  exact lt_of_le_of_lt (hN₀ N hN) hm

end

end Spec.Factorization
