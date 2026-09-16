/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
-- The encoder and decoder halves are linear layers, so their gradients reuse
-- `Spec.LinearParameterGradients` rather than repeating the weight/bias pair here.
public import NN.Spec.Layers.Linear

/-!
# Autoencoder (spec model)

This file defines a small **fully-connected autoencoder**:

- encoder: `h = act(W_enc x + b_enc)`
- decoder: `x̂ = W_dec h + b_dec`

PyTorch analogue: `nn.Sequential(nn.Linear(inputDim, hiddenDim), act, nn.Linear(hiddenDim,
  inputDim))`
applied to a single vector (no batch dimension).

This is spec-level/reference code. It is written for auditability and differentiation, and it is
intended to be instantiated over multiple scalar backends (`Float`, intervals, proof-level reals,
...).

The activation is represented by `Activation.Kind`, so a misspelled configuration cannot silently
change the model into an identity activation.

## Implementation status

`nn.models.Generative.autoencoder` (`NN/API/Models/Generative.lean`) builds a different
architecture, `x -> hidden -> latent -> hidden -> reconstruction`, and is not derived from this
record. `NN/Spec/Module/Autoencoder.lean` wraps this file as a `Spec.Module`. No theorem relates
either to the other.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-!
## Parameters

We store the encoder and decoder weights explicitly.

Shapes:

- `encoderWeight : (hiddenDim × inputDim)`
- `decoderWeight : (inputDim × hiddenDim)`
- `encoderBias   : (hiddenDim)`
- `decoderBias   : (inputDim)`
-/
/-- Parameters for a 1-hidden-layer fully-connected autoencoder. -/
structure AutoencoderSpec (α : Type) [TorchLean.Storage α] (inputDim hiddenDim : Nat) where
  /-- Encoder weights with shape `(hiddenDim × inputDim)`. -/
  encoderWeight : Tensor α [hiddenDim, inputDim]
  /-- Encoder bias with shape `(hiddenDim)`. -/
  encoderBias : Tensor α [hiddenDim]
  /-- Decoder weights with shape `(inputDim × hiddenDim)`. -/
  decoderWeight : Tensor α [inputDim, hiddenDim]
  /-- Decoder bias with shape `(inputDim)`. -/
  decoderBias : Tensor α [inputDim]
  /-- Pointwise activation between the encoder and decoder. -/
  activation : Activation.Kind := .relu

/-!
## Forward
-/

/-- Encode a vector into a hidden representation:

`h = act(W_enc x + b_enc)`.

PyTorch analogy: `act(linear(x))` for a single `nn.linear`.
-/
def autoencoderEncodeSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim]) :
  Tensor α [hiddenDim] :=
  let linearOut := addSpec (matVecMulSpec m.encoderWeight input) m.encoderBias
  m.activation.applySpec linearOut

/-- Decode a hidden representation back to input space:

`x̂ = W_dec h + b_dec`.

PyTorch analogy: a second `nn.Linear(hiddenDim, inputDim)` without an activation.
-/
def autoencoderDecodeSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (hidden : Tensor α [hiddenDim]) :
  Tensor α [inputDim] :=
  addSpec (matVecMulSpec m.decoderWeight hidden) m.decoderBias

/-- Full autoencoder forward pass: `decode(encode(x))`. -/
def autoencoderForwardSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim]) :
  Tensor α [inputDim] :=
  let hidden := autoencoderEncodeSpec m input
  autoencoderDecodeSpec m hidden

/-- Apply an autoencoder independently at every index of a leading shape. -/
def autoencoderForwardLeadingSpec (leading : Shape) {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α (leading.concat [inputDim])) :
  Tensor α (leading.concat [inputDim]) :=
  Tensor.mapLeading leading (autoencoderForwardSpec m) input

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
  (input : Tensor α [inputDim])
  (gradOutput : Tensor α [inputDim]) :
  Tensor α [hiddenDim, inputDim] :=
  -- `dH = W_decᵀ dOut`.
  let gradHidden :=
    matVecMulSpec (swapAdjacentAxes m.decoderWeight 0) gradOutput
  let linearOut := addSpec (matVecMulSpec m.encoderWeight input) m.encoderBias
  let gradLinear := mulSpec gradHidden (m.activation.derivSpec linearOut)
  outerProductSpec gradLinear input

/-- Gradient w.r.t. encoder bias: `db_enc = dZ`. -/
def autoencoderEncoderBiasDerivSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim])
  (gradOutput : Tensor α [inputDim]) :
  Tensor α [hiddenDim] :=
  let gradHidden :=
    matVecMulSpec (swapAdjacentAxes m.decoderWeight 0) gradOutput
  let linearOut := addSpec (matVecMulSpec m.encoderWeight input) m.encoderBias
  mulSpec gradHidden (m.activation.derivSpec linearOut)

/-- Gradient w.r.t. decoder weights: `dW_dec = dOut ⊗ h`. -/
def autoencoderDecoderWeightsDerivSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim])
  (gradOutput : Tensor α [inputDim]) :
  Tensor α [inputDim, hiddenDim] :=
  let hidden := autoencoderEncodeSpec m input
  outerProductSpec gradOutput hidden

/-- Gradient w.r.t. decoder bias: `db_dec = dOut`. -/
def autoencoderDecoderBiasDerivSpec {inputDim hiddenDim : Nat}
  (_m : AutoencoderSpec α inputDim hiddenDim)
  (gradOutput : Tensor α [inputDim]) :
  Tensor α [inputDim] :=
  gradOutput

/-- Gradient w.r.t. input: `dX = W_encᵀ dZ`. -/
def autoencoderInputDerivSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim])
  (gradOutput : Tensor α [inputDim]) :
  Tensor α [inputDim] :=
  let gradHidden :=
    matVecMulSpec (swapAdjacentAxes m.decoderWeight 0) gradOutput
  let linearOut := addSpec (matVecMulSpec m.encoderWeight input) m.encoderBias
  let gradLinear := mulSpec gradHidden (m.activation.derivSpec linearOut)
  matVecMulSpec (swapAdjacentAxes m.encoderWeight 0) gradLinear

/-- Gradients for a linear autoencoder: one bundle per half, plus the input gradient. -/
structure AutoencoderGradients (α : Type) [TorchLean.Storage α] (inputDim hiddenDim : Nat) where
  /-- Gradients for the encoder `inputDim -> hiddenDim`. -/
  encoder : LinearParameterGradients α inputDim hiddenDim
  /-- Gradients for the decoder `hiddenDim -> inputDim`. -/
  decoder : LinearParameterGradients α hiddenDim inputDim
  /-- Gradient with respect to the reconstructed input. -/
  inputGradient : Tensor α [inputDim]

/-- Complete backward pass for an autoencoder. -/
def autoencoderBackwardSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim])
  (gradOutput : Tensor α [inputDim]) :
  AutoencoderGradients α inputDim hiddenDim :=
  { encoder :=
      { weightGradient := autoencoderEncoderWeightsDerivSpec m input gradOutput
        biasGradient := autoencoderEncoderBiasDerivSpec m input gradOutput }
    decoder :=
      { weightGradient := autoencoderDecoderWeightsDerivSpec m input gradOutput
        biasGradient := autoencoderDecoderBiasDerivSpec m gradOutput }
    inputGradient := autoencoderInputDerivSpec m input gradOutput }

/-- Mean-squared reconstruction error (single example).

PyTorch analogy: `F.mse_loss(x_hat, x, reduction="mean")`.
-/
def autoencoderReconstructionErrorSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim]) (h : inputDim ≠ 0) :
  α :=
  let reconstructed := autoencoderForwardSpec m input
  let error := subSpec input reconstructed
  let squaredError := squareSpec error
  have inst : Shape.HasNonemptyAxis 0 (Shape.dim inputDim Shape.scalar) := by
    apply Shape.hasNonemptyAxisZeroOfNe h
  item (reduceSum 0 squaredError inst.proof) / inputDim

/-- A compact helper used by examples: compression ratio as a `Float`.

Note: if `hiddenDim = 0`, this produces `∞`/`NaN` depending on the `Float` backend.
The rest of the spec never needs this number; it is purely for display.
-/
def autoencoderCompressionRatioSpec {inputDim hiddenDim : Nat} :
  Float :=
  inputDim.toFloat / hiddenDim.toFloat

end Spec
