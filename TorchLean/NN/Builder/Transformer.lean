/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import TorchLean.NN.Builder.Blocks -- shake: keep

/-!
# Transformer Blocks

Configuration records for Transformer encoder blocks (`nn.TransformerBlock.Config`, with its
`validate` check) and stacks (`nn.TransformerStack.Config`). The seeded constructors
`nn.transformerBlock` and `nn.transformerStack` live in `TorchLean.NN.Builder`.
-/

@[expose] public section

namespace TorchLean
namespace nn

/--
Configuration of one Transformer encoder block, read by `nn.transformerBlock`.

The record carries no seeds, so one configuration can serve many blocks while the builder draws
each block's seeds from its deterministic stream.
-/
structure TransformerBlock.Config where
  /-- Number of attention heads. Must be positive. -/
  headCount : Nat
  /-- Per-head embedding dimension. Must be positive. -/
  headWidth : Nat
  /-- Hidden dimension of the feed-forward network. -/
  feedForwardWidth : Nat
  /-- Activation used in the feed-forward network. -/
  activation : Activation.Kind := .geluTanh
  /--
  Dropout on the attention and feed-forward outputs before their residual additions, the two
  residual-output sites. Set `attentionDropout?` and `feedForwardDropout?` as well to use all four
  dropout sites of a Transformer block.
  -/
  dropout? : Option Float := none
  /-- Normalize before the attention and feed-forward sublayers (pre-norm), or after each residual
  addition when `false` (post-norm, the default). -/
  normalizeFirst : Bool := false
  /-- Add a trainable bias after the attention output projection. -/
  attentionOutputBias : Bool := false
  /-- Add a separate bias to each query, key, and value projection. -/
  attentionInputBias : Bool := false
  /-- Drop softmax attention probabilities before they weight the value vectors. -/
  attentionDropout? : Option Float := none
  /-- Drop activated feed-forward hidden units before the second affine map. -/
  feedForwardDropout? : Option Float := none
  /-- Attention and feed-forward weight initialization. `none` keeps each layer's default. -/
  weightInit? : Option Init.Scheme := none
  /--
  Initializer for the attention and feed-forward projections that write to residual streams.

  When omitted, `weightInit?` is used. The separate field supports depth-scaled residual
  initialization without imposing that convention on every Transformer.
  -/
  residualInit? : Option Init.Scheme := none

/--
Configuration of a stack of Transformer encoder blocks.

`nn.build` allocates the initialization seeds, which are deliberately kept out of the
configuration.
-/
structure TransformerStack.Config where
  /-- Number of encoder blocks. -/
  layerCount : Nat
  /-- Shared configuration for each block. -/
  block : TransformerBlock.Config

namespace TransformerBlock.Config

/--
Check that the head count, head width and feed-forward width are positive, that every dropout
probability that is set is finite and in $[0, 1]$, and that both initializers, if set, pass
`Init.Scheme.validate`.

The check belongs to the configuration so that an empty stack rejects settings that would fail
as soon as its depth grows.
-/
def validate (config : TransformerBlock.Config)
    (kind : String := "TransformerEncoder") : Except String Unit := do
  if config.headCount = 0 then
    throw s!"{kind}: head count must be positive"
  if config.headWidth = 0 then
    throw s!"{kind}: head width must be positive"
  if config.feedForwardWidth = 0 then
    throw s!"{kind}: feed-forward width must be positive"
  match config.dropout? with
  | none => pure ()
  | some probability =>
      unless functional.inUnitInterval probability do
        throw s!"{kind}: dropout probability must be finite and in [0, 1], got {probability}"
  for (site, probability?) in
      [("attention", config.attentionDropout?), ("feed-forward", config.feedForwardDropout?)] do
    match probability? with
    | none => pure ()
    | some probability =>
        unless functional.inUnitInterval probability do
          throw
            s!"{kind}: {site} dropout probability must be finite and in [0, 1], got {probability}"
  match config.weightInit? with
  | none => pure ()
  | some initialization => initialization.validate
  match config.residualInit? with
  | none => pure ()
  | some initialization => initialization.validate

end TransformerBlock.Config

end nn
end TorchLean
