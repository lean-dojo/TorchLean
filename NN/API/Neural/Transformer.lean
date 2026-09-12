/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Torch.Initialization
public import NN.Spec.Layers.Activation
public import NN.API.Macros -- shake: keep
public import NN.API.Neural.Blocks -- shake: keep

/-!
# Transformer Blocks

This module defines the configuration records for Transformer encoder blocks and stacks. The seeded
constructors `nn.transformerEncoderBlock` and `nn.transformerEncoderStack` live in `NN.API.Seeded`.
-/

@[expose] public section

namespace TorchLean
namespace nn

/--
Config record for `transformerEncoderBlock`.

Separating the config as a structure makes it easier to write readable examples and keep seed
management deterministic.
-/
structure TransformerEncoder.Block.Config where
  /-- Number of attention heads. Must be positive. -/
  headCount : Nat
  /-- Per-head embedding dimension. Must be positive. -/
  headWidth : Nat
  /-- Hidden dimension of the feed-forward network. -/
  feedForwardWidth : Nat
  /-- Activation used in the feed-forward network. -/
  activation : Activation.Kind := .gelu
  /--
  Dropout on the attention and feed-forward outputs before their residual additions.

  This retains the original two-site behavior. Set `attentionDropout?` and
  `feedForwardDropout?` as well when all four Transformer dropout sites are wanted.
  -/
  dropout? : Option Float := none
  /-- Normalize before attention and feed-forward sublayers instead of after each residual. -/
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
  weightInitialization? : Option Init.Scheme := none
  /--
  Initializer for the attention and feed-forward projections that write to residual streams.

  When omitted, `weightInitialization?` is used. The separate field supports depth-scaled residual
  initialization without imposing that convention on every Transformer.
  -/
  residualOutputInitialization? : Option Init.Scheme := none

/--
Configuration for a stack of Transformer encoder blocks.

Initialization seeds are allocated by `nn.build`; they are deliberately not part of model
configuration.
-/
structure TransformerEncoder.Stack.Config where
  /-- Number of encoder blocks. -/
  layerCount : Nat
  /-- Shared configuration for each block. -/
  block : TransformerEncoder.Block.Config

namespace TransformerEncoder.Block.Config

/--
Validate the shared Transformer block template.

Validation belongs to the configuration itself rather than to an instantiated block so an empty
stack cannot silently accept settings that would fail as soon as its depth changes.
-/
def validate (config : TransformerEncoder.Block.Config)
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
      unless probability.isFinite && 0.0 <= probability && probability <= 1.0 do
        throw s!"{kind}: dropout probability must be finite and in [0, 1], got {probability}"
  for (site, probability?) in
      [("attention", config.attentionDropout?), ("feed-forward", config.feedForwardDropout?)] do
    match probability? with
    | none => pure ()
    | some probability =>
        unless probability.isFinite && 0.0 <= probability && probability <= 1.0 do
          throw
            s!"{kind}: {site} dropout probability must be finite and in [0, 1], got {probability}"
  match config.weightInitialization? with
  | none => pure ()
  | some initialization => initialization.validate
  match config.residualOutputInitialization? with
  | none => pure ()
  | some initialization => initialization.validate

end TransformerEncoder.Block.Config

end nn
end TorchLean
