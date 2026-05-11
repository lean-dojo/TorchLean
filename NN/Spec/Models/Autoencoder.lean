/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation

/-!
# Autoencoder (spec model)

This file defines a small **fully-connected autoencoder**:

- encoder: `h = act(W_enc x + b_enc)`
- decoder: `x̂ = W_dec h + b_dec`

PyTorch mental model: `nn.Sequential(nn.Linear(inputDim, hiddenDim), act, nn.Linear(hiddenDim,
  inputDim))`
applied to a single vector (no batch dimension).

This is spec-level/reference code. It is written for auditability and differentiation, and it is
intended to be instantiated over multiple scalar backends (`Float`, intervals, proof-level reals,
...).

Note on `activation_type`:

We keep a small string switch for demos and exporters. Most TorchLean code prefers choosing the
activation by composition (at the module level), but having the switch here makes the "one-file
model" convenient for examples.
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-!
## Parameters

We store the encoder and decoder weights explicitly.

Shapes:

- `encoder_weights : (hiddenDim × inputDim)`
- `decoder_weights : (inputDim × hiddenDim)`
- `encoder_bias    : (hiddenDim)`
- `decoder_bias    : (inputDim)`
-/
/-- Parameters for a 1-hidden-layer fully-connected autoencoder. -/
structure AutoencoderSpec (α : Type) (inputDim hiddenDim : Nat) where
  /-- Encoder weights with shape `(hiddenDim × inputDim)`. -/
  encoder_weights : Tensor α (.dim hiddenDim (.dim inputDim .scalar))
  /-- Encoder bias with shape `(hiddenDim)`. -/
  encoder_bias : Tensor α (.dim hiddenDim .scalar)
  /-- Decoder weights with shape `(inputDim × hiddenDim)`. -/
  decoder_weights : Tensor α (.dim inputDim (.dim hiddenDim .scalar))
  /-- Decoder bias with shape `(inputDim)`. -/
  decoder_bias : Tensor α (.dim inputDim .scalar)
  /-- Activation choice used by `autoencoder_activation_spec` (defaults to `"relu"`). -/
  activation_type : String := "relu"

/-!
## Forward
-/

/-- Apply the chosen activation (a small convenience wrapper around `Activation.*_spec`). -/
def autoencoderActivationSpec {n : Nat}
  (activation_type : String)
  (t : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  match activation_type with
  | "relu" => Activation.reluSpec t
  | "sigmoid" => Activation.sigmoidSpec t
  | "tanh" => Activation.tanhSpec t
  | _ => t

/-- Pointwise derivative of the chosen activation (used for the manual backward below). -/
def autoencoderActivationDerivSpec {n : Nat}
  (activation_type : String)
  (t : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  match activation_type with
  | "relu" => Activation.reluDerivSpec t
  | "sigmoid" => Activation.sigmoidDerivSpec t
  | "tanh" => Activation.tanhDerivSpec t
  | _ => fill 1 (.dim n .scalar)

/-- Encode a vector into a hidden representation:

`h = act(W_enc x + b_enc)`.

PyTorch analogy: `act(linear(x))` for a single `nn.Linear`.
-/
def autoencoderEncodeSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α (.dim inputDim .scalar)) :
  Tensor α (.dim hiddenDim .scalar) :=
  let linear_out := addSpec (matVecMulSpec m.encoder_weights input) m.encoder_bias
  autoencoderActivationSpec (α := α) (n := hiddenDim) m.activation_type linear_out

/-- Decode a hidden representation back to input space:

`x̂ = W_dec h + b_dec`.

PyTorch analogy: a second `nn.Linear(hiddenDim, inputDim)` without an activation.
-/
def autoencoderDecodeSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (hidden : Tensor α (.dim hiddenDim .scalar)) :
  Tensor α (.dim inputDim .scalar) :=
  addSpec (matVecMulSpec m.decoder_weights hidden) m.decoder_bias

/-- Full autoencoder forward pass: `decode(encode(x))`. -/
def autoencoderForwardSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α (.dim inputDim .scalar)) :
  Tensor α (.dim inputDim .scalar) :=
  let hidden := autoencoderEncodeSpec m input
  autoencoderDecodeSpec m hidden

/-- Batched forward pass (maps the single-example forward over the outer batch axis). -/
def autoencoderBatchedForwardSpec {batch inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α (.dim batch (.dim inputDim .scalar))) :
  Tensor α (.dim batch (.dim inputDim .scalar)) :=
  match input with
  | Tensor.dim batch_fn =>
    Tensor.dim (fun i => autoencoderForwardSpec m (batch_fn i))

/-!
## Backward (manual VJP)

This file includes a small, explicit backward pass for the autoencoder.

PyTorch analogy: this is what autograd computes, but spelled out as pure functions.
The key linear-algebra identities used are:

- If `y = W x + b`, then `dW = dY ⊗ x`, `db = dY`, and `dX = Wᵀ dY`.
- If `h = act(z)`, then `dZ = dH ⊙ act'(z)`.
-/

/-- Gradient w.r.t. encoder weights: `dW_enc = dZ ⊗ x`. -/
def autoencoderEncoderWeightsDerivSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α (.dim inputDim .scalar))
  (grad_output : Tensor α (.dim inputDim .scalar)) :
  Tensor α (.dim hiddenDim (.dim inputDim .scalar)) :=
  -- `dH = W_decᵀ dOut`.
  let grad_hidden :=
    matVecMulSpec (matrixTransposeSpec m.decoder_weights) grad_output
  let linear_out := addSpec (matVecMulSpec m.encoder_weights input) m.encoder_bias
  let grad_linear := mulSpec grad_hidden (autoencoderActivationDerivSpec (α := α) (n :=
    hiddenDim) m.activation_type linear_out)
  outerProductSpec grad_linear input

/-- Gradient w.r.t. encoder bias: `db_enc = dZ`. -/
def autoencoderEncoderBiasDerivSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α (.dim inputDim .scalar))
  (grad_output : Tensor α (.dim inputDim .scalar)) :
  Tensor α (.dim hiddenDim .scalar) :=
  let grad_hidden :=
    matVecMulSpec (matrixTransposeSpec m.decoder_weights) grad_output
  let linear_out := addSpec (matVecMulSpec m.encoder_weights input) m.encoder_bias
  mulSpec grad_hidden (autoencoderActivationDerivSpec (α := α) (n := hiddenDim)
    m.activation_type linear_out)

/-- Gradient w.r.t. decoder weights: `dW_dec = dOut ⊗ h`. -/
def autoencoderDecoderWeightsDerivSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α (.dim inputDim .scalar))
  (grad_output : Tensor α (.dim inputDim .scalar)) :
  Tensor α (.dim inputDim (.dim hiddenDim .scalar)) :=
  let hidden := autoencoderEncodeSpec m input
  outerProductSpec grad_output hidden

/-- Gradient w.r.t. decoder bias: `db_dec = dOut`. -/
def autoencoderDecoderBiasDerivSpec {inputDim hiddenDim : Nat}
  (_m : AutoencoderSpec α inputDim hiddenDim)
  (grad_output : Tensor α (.dim inputDim .scalar)) :
  Tensor α (.dim inputDim .scalar) :=
  grad_output

/-- Gradient w.r.t. input: `dX = W_encᵀ dZ`. -/
def autoencoderInputDerivSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α (.dim inputDim .scalar))
  (grad_output : Tensor α (.dim inputDim .scalar)) :
  Tensor α (.dim inputDim .scalar) :=
  let grad_hidden :=
    matVecMulSpec (matrixTransposeSpec m.decoder_weights) grad_output
  let linear_out := addSpec (matVecMulSpec m.encoder_weights input) m.encoder_bias
  let grad_linear := mulSpec grad_hidden (autoencoderActivationDerivSpec (α := α) (n :=
    hiddenDim) m.activation_type linear_out)
  matVecMulSpec (matrixTransposeSpec m.encoder_weights) grad_linear

/-- Complete backward pass: returns

`(dW_enc, db_enc, dW_dec, db_dec, dX)`.
-/
def autoencoderBackwardSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α (.dim inputDim .scalar))
  (grad_output : Tensor α (.dim inputDim .scalar)) :
  (Tensor α (.dim hiddenDim (.dim inputDim .scalar)) ×
   Tensor α (.dim hiddenDim .scalar) ×
   Tensor α (.dim inputDim (.dim hiddenDim .scalar)) ×
   Tensor α (.dim inputDim .scalar) ×
   Tensor α (.dim inputDim .scalar)) :=
  let d_encoder_weights := autoencoderEncoderWeightsDerivSpec m input grad_output
  let d_encoder_bias := autoencoderEncoderBiasDerivSpec m input grad_output
  let d_decoder_weights := autoencoderDecoderWeightsDerivSpec m input grad_output
  let d_decoder_bias := autoencoderDecoderBiasDerivSpec m grad_output
  let d_input := autoencoderInputDerivSpec m input grad_output
  (d_encoder_weights, d_encoder_bias, d_decoder_weights, d_decoder_bias, d_input)

/-- Mean-squared reconstruction error (single example).

PyTorch analogy: `F.mse_loss(x_hat, x, reduction="mean")`.
-/
def autoencoderReconstructionErrorSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α (.dim inputDim .scalar)) (h : inputDim ≠ 0) :
  α :=
  let reconstructed := autoencoderForwardSpec m input
  let error := subSpec input reconstructed
  let squared_error := squareSpec error
  have inst : Shape.valid_axis_inst 0 (Shape.dim inputDim Shape.scalar) := by
    apply Shape.validAxisInstZeroAlt h
  toScalar (reduceSumAuto 0 squared_error) / inputDim

/-- A compact helper used by examples: compression ratio as a `Float`.

Note: if `hiddenDim = 0`, this produces `∞`/`NaN` depending on the `Float` backend.
The rest of the spec never needs this number; it is purely for display.
-/
def autoencoderCompressionRatioSpec {inputDim hiddenDim : Nat} :
  Float :=
  inputDim.toFloat / hiddenDim.toFloat

end Spec
