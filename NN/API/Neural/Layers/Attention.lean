/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Initialization
public import NN.API.Neural.Builders -- shake: keep

/-!
# Attention

Multi-head self-attention configuration.
-/

@[expose] public section

namespace TorchLean
namespace nn

/--
Multi-head self-attention configuration.

PyTorch analogue: `torch.nn.MultiheadAttention` (conceptually).
See `https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html`.

`modelWidth` belongs to the input and output shape, while this configuration controls the internal
attention width `headCount * headWidth`. Query, key, and value projections map from `modelWidth`
into that internal width; the output projection maps back to `modelWidth`. The two widths therefore
do not need to be equal.
-/
structure MultiHeadAttention.Config where
  /-- Number of attention heads. Must be positive. -/
  headCount : Nat
  /-- Per-head embedding dimension. Must be positive. -/
  headWidth : Nat
  /-- Projection-weight initialization. `none` retains Xavier-uniform initialization. -/
  weightInitialization? : Option Init.Scheme := none
  /--
  Optional initializer for the output projection.

  This is separate because deep residual stacks commonly scale the projection that writes back to
  the residual stream. When omitted, `weightInitialization?` is used.
  -/
  outputWeightInitialization? : Option Init.Scheme := none
  /-- Add a trainable bias after the output projection. -/
  outputBias : Bool := false
  /--
  Add independent trainable biases to the query, key, and value projections.

  Each bias has width `headCount * headWidth`. The default preserves the original bias-free
  projection layout; setting both this field and `outputBias` represents four affine projections.
  -/
  inputBias : Bool := false
  /--
  Drop attention probabilities after softmax and before multiplication by values.

  This is separate from dropout on the projected attention output in a Transformer block.
  Evaluation leaves the probabilities unchanged; `none` adds no dropout state or seed draw.
  -/
  dropout? : Option Float := none

namespace MultiHeadAttention.Config

/-- Validate attention dimensions and projection initializers. -/
def validate (config : MultiHeadAttention.Config)
    (sequenceLength modelWidth : Nat) : Except String Unit := do
  if sequenceLength = 0 then
    throw "MultiHeadAttention: sequence length must be positive"
  if modelWidth = 0 then
    throw "MultiHeadAttention: model width must be positive"
  if config.headCount = 0 then
    throw "MultiHeadAttention: head count must be positive"
  if config.headWidth = 0 then
    throw "MultiHeadAttention: head width must be positive"
  match config.dropout? with
  | none => pure ()
  | some probability =>
      unless probability.isFinite && 0.0 <= probability && probability <= 1.0 do
        throw <| "MultiHeadAttention: dropout probability must be finite and in [0, 1], " ++
          s!"got {probability}"
  match config.weightInitialization? with
  | none => pure ()
  | some initialization => initialization.validate
  match config.outputWeightInitialization? with
  | none => pure ()
  | some initialization => initialization.validate

end MultiHeadAttention.Config

end nn
end TorchLean
