/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Generative.Latent

/-!
# Vector-quantized VAE (VQ-VAE) spec

VQ-VAE replaces a continuous latent sample with a discrete codebook lookup.  This file exposes the
core mechanism in a theorem-friendly way:

1. an encoder produces a continuous latent `z_e(x)`;
2. a code index selects a codebook vector `z_q`;
3. a decoder reconstructs from `z_q`;
4. the loss combines reconstruction, codebook, and commitment terms.

The nearest-neighbor assignment is deliberately an explicit `Fin numCodes` argument.  That keeps
the spec total and avoids hiding tie-breaking policy in the mathematical layer; runtime code can
compute the index however it likes and then pass the verified index into this spec.

Reference:
- van den Oord, Vinyals, and Kavukcuoglu (2017), "Neural Discrete Representation Learning".

The loss definitions below describe scalar values. Training also needs a rule for which
parameters receive each gradient. `trainingGradients` states the latent gradient estimator:
reconstruction passes through the encoder, codebook loss updates the selected embedding, and
commitment loss updates the encoder. The index remains fixed during that backward pass. This
estimator is specified separately from the classical derivative of a nearest-neighbor lookup.

`NN.Runtime.Autograd.Model.VqVae` implements these boundaries with the runtime's `detach`
operation and a straight-through decoder input. Callers supply the encoder, decoder, and code
assignment; there is no fixed-backbone VQ-VAE builder. The theorems in
`NN.MLTheory.Generative.Latent.VQVAE` establish loss decomposition, monotonicity in the commitment
weight, and nearest-code minimization for the value definitions.
-/

@[expose] public section

namespace Generative.VQVAE

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Generative.Latent

variable {α : Type} [TorchLean.Storage α] [Context α]
variable {obs latent : Shape} {numCodes : Nat}

/-- Encoder producing the pre-quantized latent vector `z_e(x)`. -/
structure Encoder (α : Type) (obs latent : Shape) [TorchLean.Storage α] [Context α] where
  /-- Continuous encoder output before codebook lookup. -/
  forward : Tensor α obs → Tensor α latent

/-- Decoder mapping a codebook vector back to observation space. -/
structure Decoder (α : Type) (latent obs : Shape) [TorchLean.Storage α] [Context α] where
  /-- Decode a quantized latent vector. -/
  forward : Tensor α latent → Tensor α obs

/-- VQ-VAE model: encoder, codebook, and decoder. -/
structure Model (α : Type) (obs latent : Shape) (numCodes : Nat) [TorchLean.Storage α]
    [Context α] where
  /-- Continuous encoder. -/
  encoder : Encoder α obs latent
  /-- Finite codebook. -/
  codebook : Codebook α numCodes latent
  /-- Decoder from quantized latent vectors. -/
  decoder : Decoder α latent obs

/-- Pre-quantized latent `z_e(x)`. -/
def encode (model : Model α obs latent numCodes) (x : Tensor α obs) : Tensor α latent :=
  model.encoder.forward x

/-- Quantized latent `z_q`, using an explicit code index. -/
def quantized (model : Model α obs latent numCodes) (idx : Fin numCodes) : Tensor α latent :=
  quantizeAt model.codebook idx

/-- VQ-VAE reconstruction from an explicit code assignment. -/
def forward (model : Model α obs latent numCodes) (_x : Tensor α obs)
    (idx : Fin numCodes) : Tensor α obs :=
  model.decoder.forward (quantized model idx)

/-- Mean squared reconstruction error between `dec(z_q)` and the observation. -/
def reconstructionLoss
    (model : Model α obs latent numCodes) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  Spec.mseSpec (s := obs) (forward model x idx) x

/--
Mean squared distance from the selected embedding to the encoder output.

This definition records the loss value. The training program treats the encoder output as a
fixed target for this term, so only the selected codebook embedding receives its gradient.
-/
def codebookLoss
    (model : Model α obs latent numCodes) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  Spec.mseSpec (s := latent) (quantized model idx) (encode model x)

/--
Mean squared distance from the encoder output to the selected embedding.

The training program treats the embedding as a fixed target for this term. Its gradient therefore
updates the encoder alone, with weight `β` in the total objective.
-/
def commitmentLoss
    (model : Model α obs latent numCodes) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  Spec.mseSpec (s := latent) (encode model x) (quantized model idx)

/-- VQ-VAE objective: reconstruction + codebook + β commitment. -/
def loss
    (model : Model α obs latent numCodes) (beta : α) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  reconstructionLoss model x idx + codebookLoss model x idx + beta * commitmentLoss model x idx

/-- Cotangents at the encoder output and the selected codebook embedding for one assignment. -/
structure TrainingGradients (α : Type) (latent : Shape) [TorchLean.Storage α] where
  /-- Reconstruction cotangent plus the weighted commitment contribution. -/
  encoder : Tensor α latent
  /-- Codebook-loss contribution; other embeddings receive zero for this assignment. -/
  codebook : Tensor α latent

/--
The VQ-VAE latent gradient estimator for a fixed code assignment.

Let `g` be the reconstruction cotangent at the decoder input and `n` the number of latent
coordinates. The straight-through rule sends `g` to the encoder and zero to the embedding.
Adding the two auxiliary terms gives
`encoder = g + β * (2 / n) * (encoded - selected)` and
`codebook = (2 / n) * (selected - encoded)`.

`mseDerivSpec` supplies the same mean reduction as the value losses, including their empty-shape
convention. These are prescribed training signals. A hard nearest-neighbor assignment does not
have this classical derivative; the runtime constructs the estimator with explicit stop-gradient
boundaries. Decoder parameters receive their usual reconstruction gradients outside this pair.
-/
def trainingGradients (encoded selected reconstructionCotangent : Tensor α latent)
    (beta : α) : TrainingGradients α latent :=
  { encoder := reconstructionCotangent +
      Tensor.scaleSpec (Spec.mseDerivSpec encoded selected) beta
    codebook := Spec.mseDerivSpec selected encoded }

/-- Quantization by explicit index is exactly codebook lookup. -/
@[simp] theorem quantized_eq_embedding
    (model : Model α obs latent numCodes) (idx : Fin numCodes) :
    quantized model idx = model.codebook.embedding idx := by
  rfl

/-- The VQ-VAE objective decomposes into the three standard terms. -/
@[simp] theorem loss_eq_reconstruction_add_codebook_add_commitment
    (model : Model α obs latent numCodes) (beta : α) (x : Tensor α obs) (idx : Fin numCodes) :
    loss model beta x idx =
      reconstructionLoss model x idx + codebookLoss model x idx +
        beta * commitmentLoss model x idx := by
  rfl

end Generative.VQVAE
