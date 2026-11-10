/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Loss

/-!
# Generative adversarial network (GAN) spec

A small, total GAN interface:

- the generator maps latent noise to synthetic observations,
- the discriminator maps observations to scalar "realness" scores, and
- the default objective is least-squares GAN (LSGAN), written with the MSE loss specs.

We choose LSGAN as the baseline spec because it is total, compact, and verifier-friendly.
Classical logistic GAN losses can be built on the same `GAN` record, but their `log` terms need
scores strictly between 0 and 1.

The two losses are the 0-1 coding of Mao et al., eq. (9), written by hand: fake label `a = 0`,
real label `b = 1` and generator target `c = 1`. Two choices differ from the paper. Each term
compares one scalar score with its label, so the expectations in eq. (9) are left to the caller.
And we drop the factor `1/2` in front of each term. For a fake score of `0.5` the generator loss
is `0.25` here and `0.125` in eq. (9). A constant factor moves no minimizer. It doubles the
gradients.

References:
- Goodfellow et al. (2014), "Generative Adversarial Nets", NIPS 2014, arXiv:1406.2661, eq. (1).
- Mao et al. (2017), "Least Squares Generative Adversarial Networks", ICCV 2017, arXiv:1611.04076
  (v3), eq. (2) for the a-b-c objective and eq. (9) for the 0-1 coding.

## Implementation status

`nn.Generative.generator` and `nn.Generative.discriminator`
(`TorchLean/NN/Models/Generative.lean`) are independent executable builders. They aren't
constructed from the `GAN` record and no theorem relates them to this file. The theorems that
exist are about these definitions alone. Over `ℝ`,
`TorchLean/NN/Models/GAN/Loss.lean` proves that `generatorLoss model z` is zero when
the fake score is `realTarget`, and that `discriminatorLoss model xReal z` is zero when the real
score is `realTarget` and the fake score is `fakeTarget`.
-/

@[expose] public section

namespace TorchLean.GAN

open TorchLean.Tensor

variable {α : Type} [Storage α]
variable {latent obs : Shape}

/-- Pair of generator and discriminator components forming a GAN-style model. -/
structure _root_.TorchLean.GAN (α : Type) (latent obs : Shape) [Storage α] where
  /-- Generator `G_θ : z ↦ x_fake`, producing a synthetic observation from latent noise. -/
  generator : Tensor α latent → Tensor α obs
  /-- Discriminator/critic `D_φ : x ↦ score`, represented as a scalar tensor. For LSGAN, scores
  are regressed toward `0` or `1`. -/
  discriminator : Tensor α obs → Tensor α .scalar

/-- Discriminator score on a fake sample `G(z)`. -/
def fakeScore (model : GAN α latent obs) (z : Tensor α latent) : Tensor α .scalar :=
  model.discriminator (model.generator z)

/-- Scalar tensor filled with `1`, the LSGAN "real" target. -/
def realTarget [One α] : Tensor α .scalar :=
  Tensor.full (α := α) .scalar (1 : α)

/-- Scalar tensor filled with `0`, the LSGAN "fake" target. -/
def fakeTarget [Zero α] : Tensor α .scalar :=
  Tensor.full (α := α) .scalar (0 : α)

/-- Fake scoring expands to discriminator-after-generator. -/
@[simp] theorem fakeScore_eq
    (model : GAN α latent obs) (z : Tensor α latent) :
    fakeScore model z = model.discriminator (model.generator z) := by
  rfl

variable [Context α]

/--
Least-squares discriminator loss:

`MSE(D(x_real), 1) + MSE(D(G(z)), 0)`.
-/
def discriminatorLoss
    (model : GAN α latent obs) (xReal : Tensor α obs)
    (z : Tensor α latent) : α :=
  Tensor.meanSquaredError (shape := .scalar) (model.discriminator xReal) (realTarget (α := α)) +
    Tensor.meanSquaredError (shape := .scalar) (fakeScore model z) (fakeTarget (α := α))

/--
Least-squares generator loss:

`MSE(D(G(z)), 1)`.
-/
def generatorLoss
    (model : GAN α latent obs) (z : Tensor α latent) : α :=
  Tensor.meanSquaredError (shape := .scalar) (fakeScore model z) (realTarget (α := α))

/--
The LSGAN discriminator objective is the sum of real and fake score-regression terms.

The first term pushes $D(x_{\mathrm{real}})$ toward $1$ and the second pushes $D(G(z))$ toward
$0$. As a simp lemma it lets examples and proof files rewrite the loss into the two terms without
unfolding `discriminatorLoss` by hand.
-/
@[simp] theorem discriminatorLoss_eq
    (model : GAN α latent obs) (xReal : Tensor α obs) (z : Tensor α latent) :
    discriminatorLoss model xReal z =
      Tensor.meanSquaredError (shape := .scalar) (model.discriminator xReal) (realTarget (α := α)) +
        Tensor.meanSquaredError (shape := .scalar) (fakeScore model z) (fakeTarget (α := α)) := by
  rfl

end TorchLean.GAN
