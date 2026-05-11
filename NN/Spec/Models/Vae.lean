/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Generative.Latent

/-!
# Variational autoencoder (VAE) spec

This file gives TorchLean a small, backbone-independent VAE interface:

- an encoder maps an observation `x` to diagonal-Gaussian parameters `(μ, logσ²)`;
- a decoder maps a latent sample `z` back to observation space; and
- the loss combines reconstruction MSE with the diagonal-Gaussian KL term.

The design mirrors the original VAE formulation of Kingma and Welling (2014), while staying
compatible with TorchLean's deterministic spec layer by making the reparameterization noise
explicit.

References:
- Kingma and Welling (2014), "Auto-Encoding Variational Bayes".
- Rezende, Mohamed, and Wierstra (2014), "Stochastic Backpropagation and Approximate Inference".
-/

@[expose] public section

namespace Generative.VAE

open Spec
open Tensor
open Generative.Latent

variable {α : Type} [Context α]
variable {obs latent : Shape}

/-- Diagonal-Gaussian encoder `q_φ(z|x)`, returning `(μ, logσ²)`. -/
structure Encoder (α : Type) (obs latent : Shape) [Context α] where
  /-- Posterior mean. -/
  mean : Tensor α obs → Tensor α latent
  /-- Posterior log-variance. -/
  logvar : Tensor α obs → Tensor α latent

/-- Decoder/generator `p_θ(x|z)` represented by its reconstruction mean. -/
structure Decoder (α : Type) (latent obs : Shape) [Context α] where
  /-- Decode a latent sample into observation space. -/
  forward : Tensor α latent → Tensor α obs

/-- A VAE is an encoder plus a decoder. -/
structure Model (α : Type) (obs latent : Shape) [Context α] where
  /-- Approximate posterior network. -/
  encoder : Encoder α obs latent
  /-- Generative decoder network. -/
  decoder : Decoder α latent obs

/-- Encode once and return `(μ, logσ²)`. -/
def encode (model : Model α obs latent) (x : Tensor α obs) :
    Tensor α latent × Tensor α latent :=
  (model.encoder.mean x, model.encoder.logvar x)

/-- Sample the latent using explicit noise. -/
def sampleLatent (model : Model α obs latent) (x : Tensor α obs)
    (eps : Tensor α latent) : Tensor α latent :=
  let (mu, logvar) := encode model x
  reparameterizeDiag mu logvar eps

/-- Full VAE forward pass: encode, reparameterize with explicit noise, then decode. -/
def forward (model : Model α obs latent) (x : Tensor α obs)
    (eps : Tensor α latent) : Tensor α obs :=
  model.decoder.forward (sampleLatent model x eps)

/-- Reconstruction term, using mean-squared error in observation space. -/
def reconstructionLoss [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α obs latent) (x : Tensor α obs)
    (eps : Tensor α latent) : α :=
  Spec.mseSpec (s := obs) (forward model x eps) x

/-- KL term `KL(q_φ(z|x) || N(0,I))`, averaged across the latent shape. -/
def klLoss (model : Model α obs latent) (x : Tensor α obs) : α :=
  let (mu, logvar) := encode model x
  diagonalGaussianKlToStandard mu logvar

/-- β-VAE objective: reconstruction loss plus `β` times the diagonal-Gaussian KL term. -/
def loss [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α obs latent) (beta : α) (x : Tensor α obs)
    (eps : Tensor α latent) : α :=
  reconstructionLoss model x eps + beta * klLoss model x

/-- Forward expansion lemma used by examples and downstream proof files. -/
@[simp] theorem forward_eq_decode_reparameterize
    (model : Model α obs latent) (x : Tensor α obs) (eps : Tensor α latent) :
    forward model x eps =
      model.decoder.forward
        (reparameterizeDiag (model.encoder.mean x) (model.encoder.logvar x) eps) := by
  rfl

/-- The VAE objective is exactly reconstruction plus a weighted KL term. -/
@[simp] theorem loss_eq_reconstruction_add_kl
    [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α obs latent) (beta : α) (x : Tensor α obs) (eps : Tensor α latent) :
    loss model beta x eps =
      reconstructionLoss model x eps + beta * klLoss model x := by
  rfl

end Generative.VAE
