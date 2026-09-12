/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Initialization
public import NN.API.Neural.Builders -- shake: keep
public import NN.Spec.Layers.PositionalEncoding -- shake: keep

@[expose] public section

namespace TorchLean

/-!
# Positional Encodings

Configuration records for learned positions, fixed sinusoidal encodings, and rotary positional
embeddings. The seeded constructors live in `NN.API.Seeded`.
-/

namespace nn

/--
Learned positional embedding configuration.

This is a trainable parameter tensor of shape `(sequenceLength × embeddingWidth)` that is broadcast
across `batchShape` and added to the input.
-/
structure LearnedPositionalEmbedding.Config where
  /-- Initialization scheme for the positional embedding table. -/
  initialization : Init.Scheme := .uniform (-0.02) 0.02

namespace LearnedPositionalEmbedding.Config

/-- Validate table dimensions and initialization. -/
def validate (config : LearnedPositionalEmbedding.Config)
    (sequenceLength embeddingWidth : Nat) : Except String Unit := do
  if sequenceLength = 0 then
    throw "LearnedPositionalEmbedding: sequence length must be positive"
  if embeddingWidth = 0 then
    throw "LearnedPositionalEmbedding: embedding width must be positive"
  config.initialization.validate

end LearnedPositionalEmbedding.Config

/--
Sinusoidal positional encoding configuration.

Classic non-trainable Transformer sinusoidal encoding, added to token embeddings.
`startPosition` is an absolute-position offset for KV-cache decoding.
-/
structure SinusoidalPositionalEncoding.Config where
  /-- Absolute position offset for the first row of the encoding table. -/
  startPosition : Nat := 0

namespace SinusoidalPositionalEncoding.Config

/-- Validate the tensor dimensions before materializing the fixed encoding buffer. -/
def validate (_config : SinusoidalPositionalEncoding.Config)
    (sequenceLength embeddingWidth : Nat) : Except String Unit := do
  if sequenceLength = 0 then
    throw "SinusoidalPositionalEncoding: sequence length must be positive"
  if embeddingWidth = 0 then
    throw "SinusoidalPositionalEncoding: embedding width must be positive"

end SinusoidalPositionalEncoding.Config

/--
Rotary positional embedding (RoPE) configuration.

`startPosition` is an absolute-position offset for KV-cache decoding.
-/
structure RotaryEmbedding.Config where
  /-- Absolute position offset for the first row of RoPE angles. -/
  startPosition : Nat := 0

namespace RotaryEmbedding.Config

/-- Validate the tensor dimensions before materializing the fixed rotation buffers. -/
def validate (_config : RotaryEmbedding.Config)
    (sequenceLength headWidth : Nat) : Except String Unit := do
  if sequenceLength = 0 then
    throw "RoPE: sequence length must be positive"
  if headWidth = 0 then
    throw "RoPE: head width must be positive"

end RotaryEmbedding.Config

end nn
end TorchLean
