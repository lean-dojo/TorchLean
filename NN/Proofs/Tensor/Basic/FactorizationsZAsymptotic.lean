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

/-!
# CHD `Z_test`: the asymptotic-calibration scaffold and empirical-CDF consistency (steps a–b)

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
-/

@[expose] public section

namespace Spec.Factorization

open MeasureTheory ProbabilityTheory

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

end

end Spec.Factorization
