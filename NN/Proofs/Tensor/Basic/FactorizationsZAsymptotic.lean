/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.FactorizationsZTest
public import Mathlib.Probability.Independence.InfinitePi
public import Mathlib.MeasureTheory.Integral.IntegrableOn

/-!
# CHD `Z_test`: the asymptotic-calibration scaffold (step a)

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

end

end Spec.Factorization
