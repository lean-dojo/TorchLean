/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Vae
public import NN.MLTheory.Generative.Latent.Objective
public import Mathlib.Analysis.SpecialFunctions.Exp
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Probability.Distributions.Gaussian.Real

/-!
# VAE theory

This file records the executable-theory facts for TorchLean's VAE spec.

The full VAE theory involves an evidence lower bound (ELBO), expectations over posterior samples,
and measure-theoretic assumptions. We keep those assumptions explicit while proving the deterministic
and real-valued facts that are stable across implementations: encoder/reparameterization/decoder
factorization, β-VAE loss decomposition, diagonal-Gaussian KL nonnegativity, and the scalar
Gaussian reparameterization law.

The extra real-valued lemmas below formalize the mathematical spine behind the implementation:
the diagonal-Gaussian KL term is nonnegative and vanishes exactly at the standard-normal posterior
parameters; scalar VAE reparameterization preserves Gaussian laws; and the TorchLean β-VAE loss is
the negative-ELBO objective once reconstruction negative log-likelihood and KL terms are identified.

References:
- Diederik P. Kingma and Max Welling, "Auto-Encoding Variational Bayes", ICLR 2014.
- Danilo J. Rezende, Shakir Mohamed, and Daan Wierstra, "Stochastic Backpropagation and
  Approximate Inference in Deep Generative Models", ICML 2014.
-/

@[expose] public section

namespace NN.MLTheory.Generative.Latent.VAE

open _root_.Spec
open _root_.Generative.VAE
open _root_.Generative.Latent
open NN.MLTheory.Generative.Latent.Objective
open BigOperators
open MeasureTheory ProbabilityTheory
open scoped NNReal

variable {α : Type} [Context α]
variable {obs latent : Shape}

/-- VAE latent sampling is exactly diagonal-Gaussian reparameterization of encoder outputs. -/
@[simp] theorem sampleLatent_eq_reparameterize
    (model : Model α obs latent) (x : Tensor α obs) (eps : Tensor α latent) :
    sampleLatent model x eps =
      reparameterizeDiag (model.encoder.mean x) (model.encoder.logvar x) eps := by
  rfl

/-- The forward pass factors through the sampled latent. -/
@[simp] theorem forward_eq_decoder_sampleLatent
    (model : Model α obs latent) (x : Tensor α obs) (eps : Tensor α latent) :
    forward model x eps = model.decoder.forward (sampleLatent model x eps) := by
  rfl

/-- The β-VAE objective is a reconstruction term plus β-weighted KL regularization. -/
@[simp] theorem betaVae_loss_decomposition
    [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α obs latent) (beta : α) (x : Tensor α obs) (eps : Tensor α latent) :
    loss model beta x eps = reconstructionLoss model x eps + beta * klLoss model x := by
  rfl

/-! ## Connection to the shared latent-objective algebra -/

/-- Package the VAE reconstruction and KL terms as a shared weighted two-term objective. -/
noncomputable def vaeObjectiveTerms
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ obs latent) (x : Tensor ℝ obs) (eps : Tensor ℝ latent) :
    WeightedTwoTerm :=
  { base := reconstructionLoss model x eps
    regularizer := klLoss model x }

/-- β-VAE loss is exactly the shared `base + β * regularizer` objective. -/
theorem betaVae_loss_eq_weightedTwoTerm
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ obs latent) (beta : ℝ) (x : Tensor ℝ obs) (eps : Tensor ℝ latent) :
    loss model beta x eps = weightedTwoTerm beta (vaeObjectiveTerms model x eps) := by
  rfl

/-- At `β = 0`, the VAE objective reduces to reconstruction loss. -/
@[simp] theorem betaVae_loss_zero_beta
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ obs latent) (x : Tensor ℝ obs) (eps : Tensor ℝ latent) :
    loss model 0 x eps = reconstructionLoss model x eps := by
  simp [loss]

/-- If the KL term is zero, the VAE objective reduces to reconstruction loss for any β. -/
theorem betaVae_loss_eq_reconstruction_of_zero_kl
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ obs latent) (beta : ℝ) (x : Tensor ℝ obs) (eps : Tensor ℝ latent)
    (hkl : klLoss model x = 0) :
    loss model beta x eps = reconstructionLoss model x eps := by
  simp [loss, hkl]

/--
β monotonicity for the executable VAE loss.

Once a verifier or model-specific theorem establishes that the KL term is nonnegative, increasing
β cannot decrease the objective.
-/
theorem betaVae_loss_mono_beta_of_kl_nonneg
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ obs latent) (x : Tensor ℝ obs) (eps : Tensor ℝ latent)
    {beta₁ beta₂ : ℝ} (hbeta : beta₁ ≤ beta₂) (hkl : 0 ≤ klLoss model x) :
    loss model beta₁ x eps ≤ loss model beta₂ x eps := by
  simp [loss]
  exact mul_le_mul_of_nonneg_right hbeta hkl

/-! ## Real-valued KL facts for diagonal Gaussian posteriors -/

/--
One-coordinate KL contribution for `N(μ, exp(logvar)) || N(0, 1)`.

This is the usual VAE closed form
`0.5 * (exp(logσ²) + μ² - 1 - logσ²)`, stated over `ℝ` so the proof can use
mathlib's real exponential inequalities directly.
-/
noncomputable def coordinateKlToStandard (mu logvar : ℝ) : ℝ :=
  (Real.exp logvar + mu ^ 2 - 1 - logvar) / 2

/--
Diagonal-Gaussian KL against a standard normal prior, represented as finite coordinates.

For tensors, TorchLean's executable spec uses `Spec.meanOver`.  This theorem layer uses a
coordinate sum because it is the cleanest interface for mathlib big-operator reasoning and for
stating "zero iff every coordinate is zero".
-/
noncomputable def diagonalGaussianKlToStandardReal
    {n : Nat} (mu logvar : Fin n → ℝ) : ℝ :=
  ∑ i, coordinateKlToStandard (mu i) (logvar i)

/-- The elementary inequality behind VAE KL nonnegativity: `exp x ≥ 1 + x`. -/
theorem exp_minus_one_minus_nonneg (x : ℝ) : 0 ≤ Real.exp x - 1 - x := by
  have h := Real.add_one_le_exp x
  linarith

/-- Strict form of `exp x ≥ 1 + x`; equality occurs only at `x = 0`. -/
theorem exp_minus_one_minus_pos {x : ℝ} (hx : x ≠ 0) :
    0 < Real.exp x - 1 - x := by
  have h := Real.add_one_lt_exp hx
  linarith

/-- A single diagonal-Gaussian KL coordinate is nonnegative. -/
theorem coordinateKlToStandard_nonneg (mu logvar : ℝ) :
    0 ≤ coordinateKlToStandard mu logvar := by
  unfold coordinateKlToStandard
  have hvar : 0 ≤ Real.exp logvar - 1 - logvar :=
    exp_minus_one_minus_nonneg logvar
  have hmu : 0 ≤ mu ^ 2 := sq_nonneg mu
  have hsum : 0 ≤ Real.exp logvar + mu ^ 2 - 1 - logvar := by
    linarith
  positivity

/--
The one-coordinate KL vanishes exactly when the approximate posterior coordinate is already
standard normal: zero mean and zero log-variance.
-/
theorem coordinateKlToStandard_eq_zero_iff (mu logvar : ℝ) :
    coordinateKlToStandard mu logvar = 0 ↔ mu = 0 ∧ logvar = 0 := by
  constructor
  · intro h
    unfold coordinateKlToStandard at h
    have hnum : Real.exp logvar + mu ^ 2 - 1 - logvar = 0 := by
      nlinarith
    by_cases hl : logvar = 0
    · subst hl
      simp at hnum
      exact ⟨hnum, rfl⟩
    · have hpos_l : 0 < Real.exp logvar - 1 - logvar :=
        exp_minus_one_minus_pos hl
      have hmu : 0 ≤ mu ^ 2 := sq_nonneg mu
      have : 0 < Real.exp logvar + mu ^ 2 - 1 - logvar := by
        linarith
      linarith
  · rintro ⟨rfl, rfl⟩
    unfold coordinateKlToStandard
    norm_num

/-- The finite-dimensional diagonal-Gaussian KL is nonnegative. -/
theorem diagonalGaussianKlToStandardReal_nonneg
    {n : Nat} (mu logvar : Fin n → ℝ) :
    0 ≤ diagonalGaussianKlToStandardReal mu logvar := by
  unfold diagonalGaussianKlToStandardReal
  exact Finset.sum_nonneg
    (fun i _ => coordinateKlToStandard_nonneg (mu i) (logvar i))

/--
The finite-dimensional diagonal-Gaussian KL is zero iff every coordinate has zero mean and
zero log-variance.  This is the exact mathematical sanity check behind the VAE regularizer.
-/
theorem diagonalGaussianKlToStandardReal_eq_zero_iff
    {n : Nat} (mu logvar : Fin n → ℝ) :
    diagonalGaussianKlToStandardReal mu logvar = 0 ↔
      (∀ i, mu i = 0) ∧ (∀ i, logvar i = 0) := by
  constructor
  · intro h
    have hall :
        ∀ i ∈ (Finset.univ : Finset (Fin n)),
          coordinateKlToStandard (mu i) (logvar i) = 0 := by
      exact
        (Finset.sum_eq_zero_iff_of_nonneg
          (fun i _ => coordinateKlToStandard_nonneg (mu i) (logvar i))).mp
          (by simpa [diagonalGaussianKlToStandardReal] using h)
    constructor
    · intro i
      have hz := (coordinateKlToStandard_eq_zero_iff (mu i) (logvar i)).mp
        (hall i (by simp))
      exact hz.1
    · intro i
      have hz := (coordinateKlToStandard_eq_zero_iff (mu i) (logvar i)).mp
        (hall i (by simp))
      exact hz.2
  · rintro ⟨hmu, hlogvar⟩
    unfold diagonalGaussianKlToStandardReal
    apply Finset.sum_eq_zero
    intro i _hi
    exact (coordinateKlToStandard_eq_zero_iff (mu i) (logvar i)).mpr
      ⟨hmu i, hlogvar i⟩

/-! ## Reparameterization law -/

/-- Variance induced by multiplying a standard normal by a scalar `σ`. -/
noncomputable def varianceOfScale (sigma : ℝ) : ℝ≥0 :=
  ⟨sigma ^ 2, sq_nonneg sigma⟩

/--
Scalar VAE reparameterization law.

If `ε ~ N(0, 1)`, then `μ + σ ε ~ N(μ, σ²)`.  The diagonal multivariate statement is
obtained by applying this coordinatewise together with the usual independence/product-measure
assumptions; TorchLean keeps this scalar theorem as the reusable primitive because it already
matches each coordinate in the diagonal-Gaussian reparameterization trick.
-/
theorem scalar_reparameterization_law
    {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {eps : Ω → ℝ}
    (hε : HasLaw eps (gaussianReal 0 1) P) (mu sigma : ℝ) :
    HasLaw (fun ω => mu + sigma * eps ω)
      (gaussianReal mu (varianceOfScale sigma)) P := by
  have hmul := gaussianReal_const_mul (μ := 0) (v := 1) (P := P) hε sigma
  have hadd := gaussianReal_const_add (P := P) hmul mu
  simpa [varianceOfScale, add_comm, add_left_comm, add_assoc] using hadd

/--
Coordinatewise diagonal VAE reparameterization law.

If every coordinate `εᵢ` has standard-normal law, then every coordinate of
`μ + σ ⊙ ε` has Gaussian law `N(μᵢ, σᵢ²)`.  A joint diagonal-Gaussian law additionally requires
the usual independence/product-measure hypothesis; this theorem packages the part that follows
directly from mathlib's one-dimensional Gaussian pushforward lemmas.
-/
theorem diagonal_reparameterization_coordinate_law
    {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {n : Nat}
    {eps : Ω → Fin n → ℝ}
    (hε : ∀ i, HasLaw (fun ω => eps ω i) (gaussianReal 0 1) P)
    (mu sigma : Fin n → ℝ) :
    ∀ i,
      HasLaw (fun ω => mu i + sigma i * eps ω i)
        (gaussianReal (mu i) (varianceOfScale (sigma i))) P := by
  intro i
  exact scalar_reparameterization_law (hε i) (mu i) (sigma i)

/-! ## ELBO bookkeeping -/

/--
Named real-valued terms for the negative ELBO used by a VAE.

`reconstructionNll` is the expected reconstruction negative log-likelihood term and
`klPosteriorPrior` is `KL(qφ(z|x) || p(z))`.  The spec-level `loss` uses concrete tensor
losses; this record is the math-facing bridge that says what those scalars mean.
-/
structure ElboTerms where
  reconstructionNll : ℝ
  klPosteriorPrior : ℝ

/-- Negative ELBO as "reconstruction NLL plus KL". -/
def negativeElbo (terms : ElboTerms) : ℝ :=
  terms.reconstructionNll + terms.klPosteriorPrior

/-- β-negative-ELBO objective, used by β-VAE variants. -/
def betaNegativeElbo (beta : ℝ) (terms : ElboTerms) : ℝ :=
  terms.reconstructionNll + beta * terms.klPosteriorPrior

/--
Formal ELBO decomposition.

Under the explicit identifications that a concrete reconstruction scalar is the reconstruction
negative log-likelihood and a concrete KL scalar is the posterior-prior KL, the β-VAE scalar loss
is exactly the β-weighted negative ELBO. We state those identifications as hypotheses so the
boundary between probabilistic modeling assumptions and executable TorchLean losses stays visible.
-/
theorem betaVae_loss_matches_named_elbo_terms
    (beta reconstructionLossScalar klScalar : ℝ) (terms : ElboTerms)
    (hrecon : reconstructionLossScalar = terms.reconstructionNll)
    (hkl : klScalar = terms.klPosteriorPrior) :
    reconstructionLossScalar + beta * klScalar =
      betaNegativeElbo beta terms := by
  subst hrecon
  subst hkl
  rfl

end NN.MLTheory.Generative.Latent.VAE
