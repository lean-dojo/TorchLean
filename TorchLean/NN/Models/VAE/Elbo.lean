/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.VAE
public import Mathlib.Probability.Distributions.Gaussian.Real
public import TorchLean.Tensor.Reduce.Real
public import TorchLean.Tensor.Numeric.Real
public import TorchLean.Tensor.Context.Real

/-!
# VAE theory

This file proves facts about TorchLean's VAE spec over `ℝ`.

The stabilized small-variance branch and the final clamp of the spec's KL term
`TorchLean.VAE.meanKLToStandard` agree with the closed form
$\exp v+\mu^2-1-v$ at every coordinate. The KL term is therefore the coordinate sum of the closed
form divided by `2 * Tensor.meanDenominator`. It is nonnegative, it vanishes exactly at zero mean
and zero log-variance, and the β-VAE objective is monotone in β. The scalar reparameterization law
states that $\mu+\sigma\varepsilon\sim\mathcal{N}(\mu,\sigma^2)$ when
$\varepsilon\sim\mathcal{N}(0,1)$.

The β-VAE loss decomposition is a `rfl` fact about the spec, so it is proved with the spec in
`TorchLean.NN.Models.VAE`. Nothing here proves an evidence lower bound or identifies the
tensor loss with a probabilistic expectation.

The closed form is Appendix B of Kingma and Welling. Each summand of their
$-D_{\mathrm{KL}}=\tfrac12\sum_j(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2)$ is minus our
$\exp v+\mu^2-1-v$ at $v=\log\sigma_j^2$. `scalar_reparameterization_law` is the univariate
Gaussian example of their Section 2.4.

References:
- Diederik P. Kingma and Max Welling (2014), "Auto-Encoding Variational Bayes", ICLR 2014,
  arXiv:1312.6114, Section 2.4 and Appendix B.
- Danilo J. Rezende, Shakir Mohamed, and Daan Wierstra (2014), "Stochastic Backpropagation and
  Approximate Inference in Deep Generative Models", ICML 2014, arXiv:1401.4082.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.VAE

open BigOperators
open MeasureTheory ProbabilityTheory

variable {obs latent : Shape}

/-! ## The diagonal-Gaussian KL term -/

/--
Over `ℝ`, the spec's KL term is the coordinate sum of `exp (logvar c) + mu c ^ 2 - 1 - logvar c`
divided by `2 * Tensor.meanDenominator latent`.
-/
theorem meanKLToStandard_eq_sum (mu logvar : Tensor ℝ latent) :
    meanKLToStandard mu logvar =
      (∑ c : latent.Coord, (Real.exp (logvar c) + mu c ^ 2 - 1 - logvar c)) /
        (2 * Tensor.meanDenominator latent) := by
  simp only [meanKLToStandard]
  rw [Tensor.sum_eq_coord_sum,
    show ∀ S : ℝ, 1 / 2 * (S / (Tensor.meanDenominator latent : ℝ)) =
      S / (2 * Tensor.meanDenominator latent) from fun S => by ring]
  congr 1
  refine Finset.sum_congr rfl fun c _ => ?_
  simp only [Tensor.zipWith_apply]
  generalize mu c = m
  generalize logvar c = v
  have hsinh : 2 * Real.sinh (v / 2) * Real.sinh (v / 2) + (Real.sinh v - v) =
      Real.exp v - 1 - v := by
    have hcosh := Real.cosh_two_mul (v / 2)
    rw [show 2 * (v / 2) = v by ring] at hcosh
    linear_combination Real.sinh_add_cosh v - hcosh - Real.cosh_sq (v / 2)
  have hnonneg : 0 ≤ m * m + (Real.exp v - 1 - v) := by
    linarith [Real.add_one_le_exp v, mul_self_nonneg m]
  simp only [MathFunctions.exp_real, MathFunctions.sinh_real]
  split_ifs
  · rw [hsinh, max_eq_right hnonneg]
    ring
  · exact absurd ((LawfulContext.beq_iff _ _).mpr rfl) ‹_›
  · rw [max_eq_right hnonneg]
    ring
  · exact absurd ((LawfulContext.beq_iff _ _).mpr rfl) ‹_›
  · ring

/-- Over `ℝ`, the diagonal-Gaussian KL term is nonnegative. -/
theorem meanKLToStandard_nonneg (mu logvar : Tensor ℝ latent) :
    0 ≤ meanKLToStandard mu logvar := by
  rw [meanKLToStandard_eq_sum]
  refine div_nonneg (Finset.sum_nonneg fun c _ => ?_) (by positivity)
  linarith [Real.add_one_le_exp (logvar c), sq_nonneg (mu c)]

/--
Over `ℝ`, the diagonal-Gaussian KL term vanishes exactly at zero mean and zero log-variance.
-/
theorem meanKLToStandard_eq_zero_iff (mu logvar : Tensor ℝ latent) :
    meanKLToStandard mu logvar = 0 ↔ mu = 0 ∧ logvar = 0 := by
  have hterm : ∀ c : latent.Coord, 0 ≤ Real.exp (logvar c) + mu c ^ 2 - 1 - logvar c :=
    fun c => by linarith [Real.add_one_le_exp (logvar c), sq_nonneg (mu c)]
  have hden : (2 * Tensor.meanDenominator latent : ℝ) ≠ 0 :=
    mul_ne_zero two_ne_zero (Nat.cast_ne_zero.mpr (Tensor.meanDenominator_pos latent).ne')
  rw [meanKLToStandard_eq_sum, div_eq_zero_iff, or_iff_left hden,
    Finset.sum_eq_zero_iff_of_nonneg fun c _ => hterm c]
  constructor
  · intro h
    have hc : ∀ c, mu c = 0 ∧ logvar c = 0 := fun c => by
      have hzero := h c (Finset.mem_univ c)
      by_cases hv : logvar c = 0
      · rw [hv, Real.exp_zero] at hzero
        exact ⟨(pow_eq_zero_iff two_ne_zero).mp (by linarith), hv⟩
      · exfalso
        linarith [Real.add_one_lt_exp hv, sq_nonneg (mu c)]
    exact ⟨Tensor.Rep.ext fun c => by simp [(hc c).1],
      Tensor.Rep.ext fun c => by simp [(hc c).2]⟩
  · rintro ⟨rfl, rfl⟩ c _
    simp

/-! ## Loss identities -/

/-- At $\beta=0$, the VAE objective reduces to reconstruction loss. -/
@[simp] theorem loss_zero_beta
    (model : VAE ℝ obs latent) (x : Tensor ℝ obs) (eps : Tensor ℝ latent) :
    loss model 0 x eps = reconstructionLoss model x eps := by
  simp [loss]

/-- If the KL term is zero, the VAE objective reduces to reconstruction loss for any β. -/
theorem loss_eq_of_zero_kl
    (model : VAE ℝ obs latent) (beta : ℝ) (x : Tensor ℝ obs) (eps : Tensor ℝ latent)
    (hkl : klLoss model x = 0) :
    loss model beta x eps = reconstructionLoss model x eps := by
  simp [loss, hkl]

/-- Over `ℝ`, the β-VAE objective is monotone in β. -/
theorem loss_mono_beta
    (model : VAE ℝ obs latent) (x : Tensor ℝ obs) (eps : Tensor ℝ latent)
    {beta₁ beta₂ : ℝ} (hbeta : beta₁ ≤ beta₂) :
    loss model beta₁ x eps ≤ loss model beta₂ x eps := by
  unfold loss
  exact add_le_add le_rfl (mul_le_mul_of_nonneg_right hbeta
    (meanKLToStandard_nonneg (model.encoder.mean x) (model.encoder.logvar x)))

/-! ## Reparameterization law -/

/--
If `eps` has law $\mathcal{N}(0,1)$, then `mu + sigma * eps` has law
$\mathcal{N}(\mu,\sigma^2)$.
-/
theorem scalar_reparameterization_law
    {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {eps : Ω → ℝ}
    (hε : HasLaw eps (gaussianReal 0 1) P) (mu sigma : ℝ) :
    HasLaw (fun ω => mu + sigma * eps ω)
      (gaussianReal mu (NNReal.mk (sigma ^ 2) (sq_nonneg sigma))) P := by
  simpa using gaussianReal_const_add (gaussianReal_const_mul hε sigma) mu

end TorchLean.VAE
