/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Gan
public import NN.MLTheory.Generative.Latent.Objective

/-!
# GAN theory

TorchLean's baseline GAN spec uses least-squares GAN losses.  This avoids partial logarithms in the
public spec surface while still capturing the generator/discriminator game:

- the discriminator regresses real samples toward score `1` and generated samples toward `0`;
- the generator tries to make generated samples score as real.

These lemmas expose the exact composition and loss decomposition used by examples and downstream
verification work.

The mathematical reason this file is more compact than the VAE/VQ-VAE theory files is that the
public spec deliberately chooses the LSGAN square-loss objective rather than the original minimax
`log D + log(1-D)` objective.  That keeps the executable Lean surface total over every supported
scalar backend; logarithmic GAN objectives belong in a domain-restricted layer with explicit
`0 < D(x) < 1` assumptions.  The facts below are therefore the safe core: they make the
generator/discriminator composition and the score-regression targets transparent, without smuggling
analytic side conditions into a total spec.

- Ian Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014.
- Xudong Mao et al., "Least Squares Generative Adversarial Networks", ICCV 2017.
- Martin Arjovsky, Soumith Chintala, and Léon Bottou, "Wasserstein GAN", ICML 2017, for the broader
  critic-vs-discriminator viewpoint that motivates keeping the score head abstract.
-/

@[expose] public section

namespace NN.MLTheory.Generative.Latent.GAN

open _root_.Spec
open _root_.Generative.GAN
open NN.MLTheory.Generative.Latent.Objective

variable {α : Type} [Context α]
variable {latent obs : Shape}

/-- Generated samples are obtained by applying the generator to latent noise. -/
@[simp] theorem generate_eq_generator
    (model : Model α latent obs) (z : Tensor α latent) :
    generate model z = model.generator.forward z := by
  rfl

/-- Fake scores are discriminator scores on generated samples. -/
@[simp] theorem fakeScore_is_discriminator_after_generator
    (model : Model α latent obs) (z : Tensor α latent) :
    fakeScore model z = model.discriminator.forward (model.generator.forward z) := by
  rfl

/--
LSGAN generator loss regresses fake scores toward the real target.

This is the generator side of Mao et al.'s least-squares objective: the generator does not receive
the discriminator's "fake" target; it tries to move generated samples onto the real-score target.
-/
@[simp] theorem generatorLoss_eq_mse_fake_real
    [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α latent obs) (z : Tensor α latent) :
    generatorLoss model z =
      Spec.mseSpec (s := .scalar) (fakeScore model z) (realTarget (α := α)) := by
  rfl

/--
LSGAN discriminator loss splits into real-score and fake-score regression terms.

The first term pushes `D(x_real)` toward `1`; the second pushes `D(G(z))` toward `0`.  Keeping this
as a named theorem gives downstream examples and proof files a stable reference point for the game
semantics, instead of requiring them to unfold the spec directly.
-/
@[simp] theorem discriminatorLoss_eq_real_fake_terms
    [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α latent obs) (xReal : Tensor α obs) (z : Tensor α latent) :
    discriminatorLoss model xReal z =
      Spec.mseSpec (s := .scalar) (realScore model xReal) (realTarget (α := α)) +
        Spec.mseSpec (s := .scalar) (fakeScore model z) (fakeTarget (α := α)) := by
  rfl

/-! ## Connection to shared objective algebra and equilibrium checks -/

/-- Score-regression MSE is zero when the scalar prediction equals the scalar target. -/
private theorem mse_scalar_self_zero (x : Tensor ℝ .scalar) :
    Spec.mseSpec (s := .scalar) x x = 0 := by
  cases x with
  | scalar a =>
      simp [Spec.mseSpec, Tensor.subSpec, Tensor.mulSpec, Tensor.map2Spec,
        Tensor.sumSpec, Tensor.tensorFoldlSpec, Spec.meanOver, Spec.toScalarSpec,
        Shape.size]

/-- Package the LSGAN generator objective as a two-term objective with no regularizer. -/
noncomputable def generatorObjectiveTerms
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ latent obs) (z : Tensor ℝ latent) : WeightedTwoTerm :=
  { base := generatorLoss model z
    regularizer := 0 }

/-- Package the LSGAN discriminator objective as real-score plus fake-score regression. -/
noncomputable def discriminatorObjectiveTerms
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ latent obs) (xReal : Tensor ℝ obs) (z : Tensor ℝ latent) :
    WeightedThreeTerm :=
  { base := Spec.mseSpec (s := .scalar) (realScore model xReal) (realTarget (α := ℝ))
    middle := Spec.mseSpec (s := .scalar) (fakeScore model z) (fakeTarget (α := ℝ))
    regularizer := 0 }

/-- LSGAN generator loss is the shared two-term objective with zero regularizer. -/
theorem generatorLoss_eq_weightedTwoTerm
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ latent obs) (z : Tensor ℝ latent) (weight : ℝ) :
    generatorLoss model z = weightedTwoTerm weight (generatorObjectiveTerms model z) := by
  simp [generatorObjectiveTerms, weightedTwoTerm]

/-- LSGAN discriminator loss is the shared three-term objective with zero regularizer. -/
theorem discriminatorLoss_eq_weightedThreeTerm
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ latent obs) (xReal : Tensor ℝ obs) (z : Tensor ℝ latent) (weight : ℝ) :
    discriminatorLoss model xReal z =
      weightedThreeTerm weight (discriminatorObjectiveTerms model xReal z) := by
  simp [discriminatorObjectiveTerms, weightedThreeTerm, discriminatorLoss]

/-- If generated samples receive the real target score, the LSGAN generator loss is zero. -/
theorem generatorLoss_zero_of_fake_score_real
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ latent obs) (z : Tensor ℝ latent)
    (hfake : fakeScore model z = realTarget (α := ℝ)) :
    generatorLoss model z = 0 := by
  rw [generatorLoss, ← hfake]
  exact mse_scalar_self_zero (fakeScore model z)

/--
If real samples receive target `1` and generated samples receive target `0`, the LSGAN
discriminator loss is zero.
-/
theorem discriminatorLoss_zero_of_perfect_scores
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ latent obs) (xReal : Tensor ℝ obs) (z : Tensor ℝ latent)
    (hreal : realScore model xReal = realTarget (α := ℝ))
    (hfake : fakeScore model z = fakeTarget (α := ℝ)) :
    discriminatorLoss model xReal z = 0 := by
  rw [discriminatorLoss, hreal, ← hfake]
  simp [mse_scalar_self_zero]

end NN.MLTheory.Generative.Latent.GAN
