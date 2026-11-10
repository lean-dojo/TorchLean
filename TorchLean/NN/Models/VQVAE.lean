/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Latent
public import TorchLean.Tensor.Layers.Loss

/-!
# Vector-quantized VAE (VQ-VAE) spec

VQ-VAE replaces a continuous latent sample with a discrete codebook lookup. An encoder produces a
continuous latent `z_e(x)`, a code index selects a codebook vector `z_q`, a decoder reconstructs
from `z_q`, and the loss combines reconstruction, codebook and commitment terms.

The nearest-neighbor assignment is a `Fin numCodes` argument, which keeps the spec total and
leaves the tie-breaking policy out of the mathematical layer. Runtime code can compute the index
by any method and pass it in.

We wrote `loss` by hand from eq. (3) of van den Oord et al. (2017). `codebookLoss` is their
`‖sg[z_e(x)] − e‖²` and `commitmentLoss` is `‖z_e(x) − sg[e]‖²`, with `detach` as `sg` in the
training program. Both are means over the latent coordinates where eq. (3) has squared norms, so
with 64 latent coordinates our codebook term is 1/64 of theirs. In place of eq. (3)'s
`log p(x|z_q(x))`, `reconstructionLoss` is the mean squared error of the decoder output against
`x`. They use `β = 0.25` in every experiment (Section 3.2), and `loss` takes `beta` from the
caller. One index quantizes the whole latent here, while they quantize each of `N` latent
vectors, a 32 × 32 field for ImageNet in Section 4.2.

Reference:
- Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu (2017), "Neural Discrete
  Representation Learning", NIPS 2017, arXiv:1711.00937, eqs. (1) to (3), Sections 3.2 and 4.2.

The loss definitions give scalar values, and training also needs a rule for which parameters
receive each gradient. `straightThroughGrads` states the latent gradient estimator: reconstruction
passes through the encoder, codebook loss updates the selected embedding, and commitment loss
updates the encoder, with the index held fixed during the backward pass. The estimator is a
prescribed rule, separate from the classical derivative of a nearest-neighbor lookup.

`TorchLean.NN.Models.VQVAE.Autograd` implements these boundaries with the runtime's `detach`
operation and a straight-through decoder input. Callers supply the encoder, decoder and code
assignment, and no builder fixes a VQ-VAE backbone. Over `ℝ`, the theorems in
`VQVAE` simplify the loss when `beta = 0` or both quantization terms
vanish, and prove it monotone in `beta` when the commitment term is nonnegative. They also give a
nearest code for every encoder output in a nonempty codebook of real coordinate vectors. No
TorchLeanTest test runs the model.
-/

@[expose] public section

namespace TorchLean.VQVAE

open TorchLean.Tensor
open Codebook

variable {α : Type} [Storage α] [Context α]
variable {obs latent : Shape} {numCodes : Nat}

/-- VQ-VAE model: encoder, codebook, and decoder. -/
structure _root_.TorchLean.VQVAE (α : Type) (obs latent : Shape) (numCodes : Nat) [Storage α] where
  /-- Continuous encoder producing the pre-quantized latent vector `z_e(x)`. -/
  encoder : Tensor α obs → Tensor α latent
  /-- Finite codebook. -/
  codebook : Codebook α numCodes latent
  /-- Decoder mapping a quantized latent vector back to observation space. -/
  decoder : Tensor α latent → Tensor α obs

/-- Pre-quantized latent `z_e(x)`. -/
def encode (model : VQVAE α obs latent numCodes) (x : Tensor α obs) : Tensor α latent :=
  model.encoder x

/-- Quantized latent `z_q`: the codebook embedding at the code index `idx`. -/
def quantized (model : VQVAE α obs latent numCodes) (idx : Fin numCodes) : Tensor α latent :=
  quantizeAt model.codebook idx

/-- VQ-VAE reconstruction: decode the embedding at the code index `idx`. The observation argument
is unused, since the index already fixes the latent. -/
def forward (model : VQVAE α obs latent numCodes) (_x : Tensor α obs)
    (idx : Fin numCodes) : Tensor α obs :=
  model.decoder (quantized model idx)

/-- Mean squared reconstruction error between `dec(z_q)` and the observation. -/
def reconstructionLoss
    (model : VQVAE α obs latent numCodes) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  Tensor.meanSquaredError (shape := obs) (forward model x idx) x

/--
Mean squared distance from the selected embedding to the encoder output.

This definition records the loss value. The training program treats the encoder output as a
fixed target for this term, so only the selected codebook embedding receives its gradient.
-/
def codebookLoss
    (model : VQVAE α obs latent numCodes) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  Tensor.meanSquaredError (shape := latent) (quantized model idx) (encode model x)

/--
Mean squared distance from the encoder output to the selected embedding.

The training program treats the embedding as a fixed target for this term. Its gradient therefore
updates the encoder alone, with weight `β` in the total objective.
-/
def commitmentLoss
    (model : VQVAE α obs latent numCodes) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  Tensor.meanSquaredError (shape := latent) (encode model x) (quantized model idx)

/-- VQ-VAE objective: reconstruction + codebook + β commitment. -/
def loss
    (model : VQVAE α obs latent numCodes) (beta : α) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  reconstructionLoss model x idx + codebookLoss model x idx + beta * commitmentLoss model x idx

/-- Cotangents at the encoder output and the selected codebook embedding for one assignment. -/
structure Grads (α : Type) (latent : Shape) [Storage α] where
  /-- Reconstruction cotangent plus the weighted commitment contribution. -/
  encoder : Tensor α latent
  /-- Codebook-loss contribution to the selected embedding. The other embeddings receive zero for
  this assignment. -/
  codebook : Tensor α latent

/--
The VQ-VAE latent gradient estimator for a fixed code assignment.

Let `g` be the reconstruction cotangent at the decoder input and `n` the number of latent
coordinates. The straight-through rule sends `g` to the encoder and zero to the embedding.
Adding the two auxiliary terms gives
`encoder = g + β * (2 / n) * (encoded - selected)` and
`codebook = (2 / n) * (selected - encoded)`.

`meanSquaredErrorBackward` supplies the same mean reduction as the value losses, including their
empty-shape convention. These are prescribed training signals. A hard nearest-neighbor assignment
does not have this classical derivative, so the runtime builds the estimator from stop-gradient
boundaries. Decoder parameters receive their usual reconstruction gradients outside this pair.
-/
def straightThroughGrads (encoded selected reconstructionCotangent : Tensor α latent)
    (beta : α) : Grads α latent :=
  { encoder := reconstructionCotangent +
      Tensor.scale (Tensor.meanSquaredErrorBackward encoded selected) beta
    codebook := Tensor.meanSquaredErrorBackward selected encoded }

omit [Context α] in
/-- Quantization at a code index is codebook lookup. -/
@[simp] theorem quantized_eq_embedding
    (model : VQVAE α obs latent numCodes) (idx : Fin numCodes) :
    quantized model idx = model.codebook.embedding idx := by
  rfl

/-- The VQ-VAE objective decomposes into the three standard terms. -/
@[simp] theorem loss_def
    (model : VQVAE α obs latent numCodes) (beta : α) (x : Tensor α obs) (idx : Fin numCodes) :
    loss model beta x idx =
      reconstructionLoss model x idx + codebookLoss model x idx +
        beta * commitmentLoss model x idx := by
  rfl

end TorchLean.VQVAE
