/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic

/-!
# Shared latent-objective algebra

We use this file to connect the latent generative models without forcing them into one artificial
architecture. VAEs, VQ-VAEs, and GANs make different modeling choices, but their training objectives
share a small amount of algebra:

- a base term, such as reconstruction loss or real-sample score regression;
- one or more regularizing/critic terms; and
- a scalar weight that controls the tradeoff.

Keeping that algebra in one place gives the model-specific files a common language. The VAE file can
say "β-VAE is a weighted two-term objective"; the VQ-VAE file can say "VQ-VAE is a weighted
three-term objective"; the GAN file can say "LSGAN is score-regression algebra over the same scalar
loss vocabulary."

References:
- Kingma and Welling, "Auto-Encoding Variational Bayes", ICLR 2014.
- van den Oord, Vinyals, and Kavukcuoglu, "Neural Discrete Representation Learning", NeurIPS 2017.
- Mao et al., "Least Squares Generative Adversarial Networks", ICCV 2017.
-/

@[expose] public section

namespace NN.MLTheory.Generative.Latent.Objective

/-- A two-term latent objective: `base + weight * regularizer`. -/
structure WeightedTwoTerm where
  /-- Main data-fitting term, typically reconstruction or score regression. -/
  base : ℝ
  /-- Latent regularizer, KL term, commitment loss, or critic penalty. -/
  regularizer : ℝ

/-- Evaluate a weighted two-term objective. -/
def weightedTwoTerm (weight : ℝ) (terms : WeightedTwoTerm) : ℝ :=
  terms.base + weight * terms.regularizer

/-- A three-term latent objective: `base + middle + weight * regularizer`. -/
structure WeightedThreeTerm where
  /-- Main data-fitting term. -/
  base : ℝ
  /-- Unweighted auxiliary term, such as the VQ-VAE codebook loss. -/
  middle : ℝ
  /-- Weighted regularizer, such as the VQ-VAE commitment loss. -/
  regularizer : ℝ

/-- Evaluate a weighted three-term objective. -/
def weightedThreeTerm (weight : ℝ) (terms : WeightedThreeTerm) : ℝ :=
  terms.base + terms.middle + weight * terms.regularizer

/-- A weighted two-term objective collapses to its base term when the regularizer is zero. -/
@[simp] theorem weightedTwoTerm_zero_regularizer (weight base : ℝ) :
    weightedTwoTerm weight { base := base, regularizer := 0 } = base := by
  simp [weightedTwoTerm]

/-- At weight zero, a weighted two-term objective ignores the regularizer. -/
@[simp] theorem weightedTwoTerm_zero_weight (terms : WeightedTwoTerm) :
    weightedTwoTerm 0 terms = terms.base := by
  simp [weightedTwoTerm]

/--
If the regularizer is nonnegative, increasing the weight can only increase a two-term objective.

This is the algebraic core behind β-VAE monotonicity in the KL weight.
-/
theorem weightedTwoTerm_mono_weight
    (terms : WeightedTwoTerm) {w₁ w₂ : ℝ}
    (hw : w₁ ≤ w₂) (hreg : 0 ≤ terms.regularizer) :
    weightedTwoTerm w₁ terms ≤ weightedTwoTerm w₂ terms := by
  unfold weightedTwoTerm
  simp
  exact mul_le_mul_of_nonneg_right hw hreg

/-- A weighted three-term objective collapses to `base + middle` when the regularizer is zero. -/
@[simp] theorem weightedThreeTerm_zero_regularizer (weight base middle : ℝ) :
    weightedThreeTerm weight { base := base, middle := middle, regularizer := 0 } =
      base + middle := by
  simp [weightedThreeTerm]

/-- A weighted three-term objective collapses to `base` when both auxiliary terms are zero. -/
@[simp] theorem weightedThreeTerm_zero_middle_zero_regularizer (weight base : ℝ) :
    weightedThreeTerm weight { base := base, middle := 0, regularizer := 0 } = base := by
  simp [weightedThreeTerm]

/-- At weight zero, a weighted three-term objective keeps only its base and middle terms. -/
@[simp] theorem weightedThreeTerm_zero_weight (terms : WeightedThreeTerm) :
    weightedThreeTerm 0 terms = terms.base + terms.middle := by
  simp [weightedThreeTerm]

/--
If the weighted regularizer is nonnegative, increasing the weight can only increase a three-term
objective.

For VQ-VAE this is the commitment-weight monotonicity statement.
-/
theorem weightedThreeTerm_mono_weight
    (terms : WeightedThreeTerm) {w₁ w₂ : ℝ}
    (hw : w₁ ≤ w₂) (hreg : 0 ≤ terms.regularizer) :
    weightedThreeTerm w₁ terms ≤ weightedThreeTerm w₂ terms := by
  unfold weightedThreeTerm
  simp
  exact mul_le_mul_of_nonneg_right hw hreg

end NN.MLTheory.Generative.Latent.Objective
