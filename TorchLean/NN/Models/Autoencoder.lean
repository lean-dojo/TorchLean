/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Activation
-- The encoder and decoder halves are linear layers, so their gradients are
-- `TorchLean.Linear` records.
public import TorchLean.Tensor.Layers.Linear

/-!
# Autoencoder (spec model)

A small **fully-connected autoencoder**:

- encoder: `h = act(W_enc x + b_enc)`
- decoder: `x̂ = W_dec h + b_dec`

PyTorch analogue: `nn.Sequential(nn.Linear(inputDim, hiddenDim), act, nn.Linear(hiddenDim,
  inputDim))`
applied to a single vector (no batch dimension).

The model is spec-level reference code, written for auditability and differentiation and meant
to be instantiated over several scalar backends (`Float`, the configured binary formats,
proof-level reals, ...).

The activation is an `Activation.Kind`, so a misspelled configuration cannot silently turn the
model into one with an identity activation.

## Implementation status

`nn.Generative.autoencoder` (`TorchLean/NN/Models/Generative.lean`) builds a different
architecture, `x -> hidden -> latent -> hidden -> reconstruction`, and is not derived from this
record. No theorem relates either to the other.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-!
## Parameters

The record keeps the weight and bias of each half as separate fields, with these shapes:

- `encoderWeight : (hiddenDim × inputDim)`
- `decoderWeight : (inputDim × hiddenDim)`
- `encoderBias   : (hiddenDim)`
- `decoderBias   : (inputDim)`
-/
/-- Parameters for a 1-hidden-layer fully-connected autoencoder. -/
structure Autoencoder (α : Type) [Storage α] (inputDim hiddenDim : Nat) where
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

PyTorch analogy: `act(linear(x))` for a single `nn.Linear(inputDim, hiddenDim)`.
-/
def Autoencoder.encode {inputDim hiddenDim : Nat}
  (m : Autoencoder α inputDim hiddenDim)
  (input : Tensor α [inputDim]) :
  Tensor α [hiddenDim] :=
  let linearOut := Tensor.contract m.encoderWeight input _ _ [hiddenDim] .matVec + m.encoderBias
  m.activation.apply linearOut

/-- Decode a hidden representation back to input space:

`x̂ = W_dec h + b_dec`.

PyTorch analogy: a second `nn.Linear(hiddenDim, inputDim)` without an activation.
-/
def Autoencoder.decode {inputDim hiddenDim : Nat}
  (m : Autoencoder α inputDim hiddenDim)
  (hidden : Tensor α [hiddenDim]) :
  Tensor α [inputDim] :=
  Tensor.contract m.decoderWeight hidden _ _ [inputDim] .matVec + m.decoderBias

/-- Full autoencoder forward pass: `decode(encode(x))`. -/
def Autoencoder.forward {inputDim hiddenDim : Nat}
  (m : Autoencoder α inputDim hiddenDim)
  (input : Tensor α [inputDim]) :
  Tensor α [inputDim] :=
  let hidden := encode m input
  decode m hidden

/-!
## Backward (manual VJP)

The autoencoder's backward pass, written by hand: what PyTorch autograd computes, spelled out as
pure functions. It rests on two identities:

- If `y = W x + b`, then `dW = dY ⊗ x`, `db = dY`, and `dX = Wᵀ dY`.
- If `h = act(z)`, then `dZ = dH ⊙ act'(z)`.

With the default ReLU the slope at `z = 0` is `0`, since `reluDeriv` tests `z > 0`. No theorem
ties `Autoencoder.backward` to a derivative, and no test runs it against torch.
-/

/-- Autoencoder gradients: one linear-parameter bundle per half, plus the input gradient. -/
structure Autoencoder.Grads (α : Type) [Storage α] (inputDim hiddenDim : Nat) where
  /-- Gradients for the encoder `inputDim -> hiddenDim`. -/
  encoder : Linear α inputDim hiddenDim
  /-- Gradients for the decoder `hiddenDim -> inputDim`. -/
  decoder : Linear α hiddenDim inputDim
  /-- Gradient with respect to the encoder input. -/
  inputGradient : Tensor α [inputDim]

/-- Complete backward pass, sharing the hidden gradient across encoder and input derivatives. -/
def Autoencoder.backward {inputDim hiddenDim : Nat}
  (m : Autoencoder α inputDim hiddenDim)
  (input : Tensor α [inputDim])
  (gradOutput : Tensor α [inputDim]) :
  Grads α inputDim hiddenDim :=
  let gradHidden :=
      Tensor.contract
        (Tensor.permute m.decoderWeight (Shape.adjacentSwap [inputDim, hiddenDim] 0)
          [hiddenDim, inputDim]) gradOutput _ _ [hiddenDim] .matVec
  let linearOut := Tensor.contract m.encoderWeight input _ _ [hiddenDim] .matVec + m.encoderBias
  let gradLinear := gradHidden * m.activation.deriv linearOut
  { encoder :=
      { weights := Linear.weightsDeriv input gradLinear
        bias := gradLinear }
    decoder :=
      { weights := Linear.weightsDeriv (encode m input) gradOutput
        bias := gradOutput }
    inputGradient :=
        Tensor.contract
          (Tensor.permute m.encoderWeight (Shape.adjacentSwap [hiddenDim, inputDim] 0)
            [inputDim, hiddenDim]) gradLinear _ _ [inputDim] .matVec }

/-- The mean-squared reconstruction error of one example, `Tensor.meanSquaredError` of the
reconstruction against the input (PyTorch's `F.mse_loss(x_hat, x)`). -/
def Autoencoder.reconstructionError {inputDim hiddenDim : Nat}
  (m : Autoencoder α inputDim hiddenDim)
  (input : Tensor α [inputDim]) :
  α :=
  Tensor.meanSquaredError (forward m input) input

end TorchLean
