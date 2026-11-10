/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder.Layer -- shake: keep

/-!
# Attention

The configuration record `Attention.Options` of multi-head self-attention and its `validate`
check. The seeded constructor `nn.attention` that reads the record lives in
`TorchLean.NN.Builder`.
-/

@[expose] public section

namespace TorchLean
namespace nn

/--
Configuration of multi-head self-attention.

PyTorch analogue: `torch.nn.MultiheadAttention` (conceptually).
See `https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html`.

`modelWidth` belongs to the input and output shape, and the record fixes the internal attention
width `headCount * headWidth`. The query, key and value projections map `modelWidth` to that
internal width and the output projection maps it back, so the two widths may differ.
-/
structure Attention.Options where
  /-- Number of attention heads. Must be positive. -/
  headCount : Nat
  /-- Per-head embedding dimension. Must be positive. -/
  headWidth : Nat
  /-- Projection-weight initialization. `none` selects Xavier-uniform initialization. -/
  weightInit? : Option Init.Scheme := none
  /--
  Optional initializer for the output projection, falling back to `weightInit?` when
  omitted. Deep residual stacks often scale the projection that writes back to the residual
  stream, and this field lets them initialize it on its own.
  -/
  outputWeightInit? : Option Init.Scheme := none
  /-- Add a trainable bias after the output projection. -/
  outputBias : Bool := false
  /--
  Add independent trainable biases to the query, key, and value projections.

  Each bias has width `headCount * headWidth`. The default keeps the projections bias-free, and
  setting both this field and `outputBias` makes all four projections affine.
  -/
  inputBias : Bool := false
  /--
  Drop attention probabilities after softmax and before the multiplication by the values.

  Dropout on the projected attention output is a separate site of a Transformer block, set by the
  `dropout?` field of `TransformerBlock.Config`. Evaluation leaves the probabilities
  unchanged, and `none` adds no dropout state or seed draw.
  -/
  dropout? : Option Float := none

namespace Attention.Options

/--
Check that `sequenceLength`, `modelWidth`, `headCount` and `headWidth` are positive, that the
dropout probability, if set, is finite and in $[0, 1]$, and that each initializer that is set
passes `Init.Scheme.validate`.
-/
def validate (config : Attention.Options)
    (sequenceLength modelWidth : Nat) : Except String Unit := do
  if sequenceLength = 0 then
    throw "Attention: sequence length must be positive"
  if modelWidth = 0 then
    throw "Attention: model width must be positive"
  if config.headCount = 0 then
    throw "Attention: head count must be positive"
  if config.headWidth = 0 then
    throw "Attention: head width must be positive"
  match config.dropout? with
  | none => pure ()
  | some probability =>
      unless functional.inUnitInterval probability do
        throw <| "Attention: dropout probability must be finite and in [0, 1], " ++
          s!"got {probability}"
  match config.weightInit? with
  | none => pure ()
  | some initialization => initialization.validate
  match config.outputWeightInit? with
  | none => pure ()
  | some initialization => initialization.validate

end Attention.Options

end nn
end TorchLean
