/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Latent
public import TorchLean.Tensor.Layers.Loss

/-!
# Variational autoencoder (VAE) spec

The spec VAE is a small interface that fixes no backbone. An encoder maps an observation `x` to
diagonal-Gaussian parameters `(μ, logσ²)`, a decoder maps a latent sample `z` back to observation
space, and the loss adds `β` times the diagonal-Gaussian KL term to the reconstruction MSE.

The design follows the VAE of Kingma and Welling (2014). The reparameterization noise is an
argument, which keeps the definitions deterministic like the rest of TorchLean's spec layer.

The scale of `loss` is our choice. Their estimator, eq. (10), sums the KL over the `J` latent
coordinates, and for a unit-variance Gaussian decoder its reconstruction term is `−½ ‖x − x̂‖²`
plus a constant, with `x̂` the decoder output. We average both terms over their coordinates, so
with `D` observation coordinates `loss` is `2 / D` times their negative bound, up to that
constant, when `β = 2J / D`. For `D = 784` and `J = 20` that's `β ≈ 0.051`, and `β = 1` weighs
the KL against reconstruction about 20 times more heavily than eq. (10) does.

References:
- Diederik P. Kingma and Max Welling (2014), "Auto-Encoding Variational Bayes", ICLR 2014,
  arXiv:1312.6114, eq. (10).
- Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra (2014), "Stochastic Backpropagation
  and Approximate Inference in Deep Generative Models", ICML 2014, arXiv:1401.4082.

## Implementation status

No API builder implements this model. `loss_def` below states the β-VAE loss
decomposition. Over `ℝ`, `TorchLean/NN/Models/VAE/Elbo.lean` proves that the KL
term `meanKLToStandard` of `klLoss` is the coordinate sum of the closed form divided by
`2 * Tensor.meanDenominator latent`, nonnegative and zero exactly at zero mean and log-variance,
and that `loss` equals `reconstructionLoss` when `beta` or `klLoss` is zero and is monotone in
`beta`. Its reparameterization law is about real numbers: `mu + sigma * eps` has law `gaussianReal`
with mean `mu` and variance `sigma ^ 2` when `eps` has law `gaussianReal 0 1`. No theorem states
the law of `reparameterize`, and no TorchLeanTest test runs the model.
-/

@[expose] public section

namespace TorchLean.VAE

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]
variable {obs latent : Shape}

/-- Diagonal-Gaussian encoder `q_φ(z|x)`, returning `(μ, logσ²)`. -/
structure GaussianEncoder (α : Type) (obs latent : Shape) [Storage α] where
  /-- Posterior mean. -/
  mean : Tensor α obs → Tensor α latent
  /-- Posterior log-variance. -/
  logvar : Tensor α obs → Tensor α latent

/-- A VAE is an encoder plus a decoder. -/
structure _root_.TorchLean.VAE (α : Type) (obs latent : Shape) [Storage α] where
  /-- Approximate posterior network. -/
  encoder : GaussianEncoder α obs latent
  /-- Decoder/generator `p_θ(x|z)` represented by its reconstruction mean: it decodes a latent
  sample into observation space. -/
  decoder : Tensor α latent → Tensor α obs

/-- Encode once and return `(μ, logσ²)`. -/
def encode (model : VAE α obs latent) (x : Tensor α obs) :
    Tensor α latent × Tensor α latent :=
  (model.encoder.mean x, model.encoder.logvar x)

/-- Sample the latent by reparameterization, `z = μ + exp(logσ² / 2) ⊙ eps`, with caller-supplied
noise `eps`. -/
def sampleLatent (model : VAE α obs latent) (x : Tensor α obs)
    (eps : Tensor α latent) : Tensor α latent :=
  let (mu, logvar) := encode model x
  reparameterize mu logvar eps

/-- Full VAE forward pass: encode, reparameterize with the noise `eps`, then decode. -/
def forward (model : VAE α obs latent) (x : Tensor α obs)
    (eps : Tensor α latent) : Tensor α obs :=
  model.decoder (sampleLatent model x eps)

/-- Reconstruction term, using mean-squared error in observation space. -/
def reconstructionLoss
    (model : VAE α obs latent) (x : Tensor α obs)
    (eps : Tensor α latent) : α :=
  Tensor.meanSquaredError (shape := obs) (forward model x eps) x

/-- KL term `KL(q_φ(z|x) || N(0,I))`, averaged across the latent shape. -/
def klLoss (model : VAE α obs latent) (x : Tensor α obs) : α :=
  let (mu, logvar) := encode model x
  meanKLToStandard mu logvar

/-- β-VAE objective: reconstruction loss plus `β` times the diagonal-Gaussian KL term. -/
def loss
    (model : VAE α obs latent) (beta : α) (x : Tensor α obs)
    (eps : Tensor α latent) : α :=
  reconstructionLoss model x eps + beta * klLoss model x

/-- The forward pass decodes the reparameterized encoder outputs. -/
@[simp] theorem forward_eq_decoder_reparameterize
    (model : VAE α obs latent) (x : Tensor α obs) (eps : Tensor α latent) :
    forward model x eps =
      model.decoder
        (reparameterize (model.encoder.mean x) (model.encoder.logvar x) eps) := by
  rfl

/-- The VAE objective is exactly reconstruction plus a weighted KL term. -/
@[simp] theorem loss_def
    (model : VAE α obs latent) (beta : α) (x : Tensor α obs) (eps : Tensor α latent) :
    loss model beta x eps =
      reconstructionLoss model x eps + beta * klLoss model x := by
  rfl

end TorchLean.VAE
