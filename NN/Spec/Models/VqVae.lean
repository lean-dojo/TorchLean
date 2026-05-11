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
-/

@[expose] public section

namespace Generative.VQVAE

open Spec
open Tensor
open Generative.Latent

variable {α : Type} [Context α]
variable {obs latent : Shape} {numCodes : Nat}

/-- Encoder producing the pre-quantized latent vector `z_e(x)`. -/
structure Encoder (α : Type) (obs latent : Shape) [Context α] where
  /-- Continuous encoder output before codebook lookup. -/
  forward : Tensor α obs → Tensor α latent

/-- Decoder mapping a codebook vector back to observation space. -/
structure Decoder (α : Type) (latent obs : Shape) [Context α] where
  /-- Decode a quantized latent vector. -/
  forward : Tensor α latent → Tensor α obs

/-- VQ-VAE model: encoder, codebook, and decoder. -/
structure Model (α : Type) (obs latent : Shape) (numCodes : Nat) [Context α] where
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

/-- Reconstruction term `||dec(z_q)-x||²`. -/
def reconstructionLoss [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α obs latent numCodes) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  Spec.mseSpec (s := obs) (forward model x idx) x

/-- Codebook term `||z_q-z_e||²`, written symmetrically at spec level. -/
def codebookLoss [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α obs latent numCodes) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  Spec.mseSpec (s := latent) (quantized model idx) (encode model x)

/-- Commitment term `||z_e-z_q||²`, weighted by `β` in the total objective. -/
def commitmentLoss [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α obs latent numCodes) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  Spec.mseSpec (s := latent) (encode model x) (quantized model idx)

/-- VQ-VAE objective: reconstruction + codebook + β commitment. -/
def loss [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α obs latent numCodes) (beta : α) (x : Tensor α obs)
    (idx : Fin numCodes) : α :=
  reconstructionLoss model x idx + codebookLoss model x idx + beta * commitmentLoss model x idx

/-- Quantization by explicit index is exactly codebook lookup. -/
@[simp] theorem quantized_eq_embedding
    (model : Model α obs latent numCodes) (idx : Fin numCodes) :
    quantized model idx = model.codebook.embedding idx := by
  rfl

/-- The VQ-VAE objective decomposes into the three standard terms. -/
@[simp] theorem loss_eq_reconstruction_add_codebook_add_commitment
    [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α obs latent numCodes) (beta : α) (x : Tensor α obs) (idx : Fin numCodes) :
    loss model beta x idx =
      reconstructionLoss model x idx + codebookLoss model x idx +
        beta * commitmentLoss model x idx := by
  rfl

end Generative.VQVAE
