/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Loss

/-!
# Generative adversarial network (GAN) spec

This module gives TorchLean a small, total GAN interface:

- `Generator` maps latent noise to synthetic observations;
- `Discriminator` maps observations to scalar "realness" scores; and
- the default objective is least-squares GAN (LSGAN), written with existing MSE specs.

We choose LSGAN as the baseline spec because it is total, compact, and verifier-friendly.  Classical
logistic GAN losses can be built on top of the same `Generator`/`Discriminator` records, but they
need additional domain discipline around `log`.

References:
- Goodfellow et al. (2014), "Generative Adversarial Nets".
- Mao et al. (2017), "Least Squares Generative Adversarial Networks".
-/

@[expose] public section

namespace Generative.GAN

open Spec
open Tensor

variable {α : Type} [Context α]
variable {latent obs : Shape}

/-- Generator `G_θ : z ↦ x_fake`. -/
structure Generator (α : Type) (latent obs : Shape) [Context α] where
  /-- Produce a synthetic observation from latent noise. -/
  forward : Tensor α latent → Tensor α obs

/-- Discriminator/critic `D_φ : x ↦ score`, represented as a scalar tensor. -/
structure Discriminator (α : Type) (obs : Shape) [Context α] where
  /-- Score an observation.  For LSGAN, scores are regressed toward `0` or `1`. -/
  forward : Tensor α obs → Tensor α .scalar

/-- GAN model pair. -/
structure Model (α : Type) (latent obs : Shape) [Context α] where
  /-- Latent-to-observation generator. -/
  generator : Generator α latent obs
  /-- Observation-to-score discriminator. -/
  discriminator : Discriminator α obs

/-- Generate a fake sample. -/
def generate (model : Model α latent obs) (z : Tensor α latent) : Tensor α obs :=
  model.generator.forward z

/-- Discriminator score on a fake sample `G(z)`. -/
def fakeScore (model : Model α latent obs) (z : Tensor α latent) : Tensor α .scalar :=
  model.discriminator.forward (generate model z)

/-- Discriminator score on a real sample. -/
def realScore (model : Model α latent obs) (x : Tensor α obs) : Tensor α .scalar :=
  model.discriminator.forward x

/-- Scalar tensor filled with `1`, the LSGAN "real" target. -/
def realTarget : Tensor α .scalar :=
  fill (α := α) (1 : α) .scalar

/-- Scalar tensor filled with `0`, the LSGAN "fake" target. -/
def fakeTarget : Tensor α .scalar :=
  fill (α := α) (0 : α) .scalar

/--
Least-squares discriminator loss:

`MSE(D(x_real), 1) + MSE(D(G(z)), 0)`.
-/
def discriminatorLoss [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α latent obs) (xReal : Tensor α obs)
    (z : Tensor α latent) : α :=
  Spec.mseSpec (s := .scalar) (realScore model xReal) (realTarget (α := α)) +
    Spec.mseSpec (s := .scalar) (fakeScore model z) (fakeTarget (α := α))

/--
Least-squares generator loss:

`MSE(D(G(z)), 1)`.
-/
def generatorLoss [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α latent obs) (z : Tensor α latent) : α :=
  Spec.mseSpec (s := .scalar) (fakeScore model z) (realTarget (α := α))

/-- Fake scoring expands to discriminator-after-generator. -/
@[simp] theorem fakeScore_eq_discriminator_generate
    (model : Model α latent obs) (z : Tensor α latent) :
    fakeScore model z = model.discriminator.forward (model.generator.forward z) := by
  rfl

/-- The LSGAN discriminator objective is the sum of real and fake score-regression terms. -/
@[simp] theorem discriminatorLoss_eq
    [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α latent obs) (xReal : Tensor α obs) (z : Tensor α latent) :
    discriminatorLoss model xReal z =
      Spec.mseSpec (s := .scalar) (realScore model xReal) (realTarget (α := α)) +
        Spec.mseSpec (s := .scalar) (fakeScore model z) (fakeTarget (α := α)) := by
  rfl

end Generative.GAN
