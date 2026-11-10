/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder.Layer -- shake: keep
public import TorchLean.Tensor.Layers.PositionalEncoding -- shake: keep

@[expose] public section

namespace TorchLean

/-!
# Positional Encodings

Configuration records for learned positions (`LearnedPositionalEmbedding.Config`), fixed
sinusoidal encodings (`SinusoidalPositionalEncoding.Config`) and rotary embeddings
(`RotaryEmbedding.Config`), each with a `validate` check. The seeded constructors that read them
live in `TorchLean.NN.Builder`.
-/

namespace nn

/--
Configuration of a learned positional embedding: a trainable table of shape
`(sequenceLength × embeddingWidth)`, broadcast across `batchShape` and added to the input.
-/
structure LearnedPositionalEmbedding.Config where
  /-- Initialization scheme for the positional embedding table. -/
  init : Init.Scheme := .uniform (-0.02) 0.02

namespace LearnedPositionalEmbedding.Config

/-- Check that both table dimensions are positive and that `init` passes validation. -/
def validate (config : LearnedPositionalEmbedding.Config)
    (sequenceLength embeddingWidth : Nat) : Except String Unit := do
  if sequenceLength = 0 then
    throw "LearnedPositionalEmbedding: sequence length must be positive"
  if embeddingWidth = 0 then
    throw "LearnedPositionalEmbedding: embedding width must be positive"
  config.init.validate

end LearnedPositionalEmbedding.Config

/--
Configuration of the fixed sinusoidal encoding of the original Transformer, a non-trainable table
added to the token embeddings. `offset` is an absolute-position offset for KV-cache
decoding.
-/
structure SinusoidalPositionalEncoding.Config where
  /-- Absolute position offset for the first row of the encoding table. -/
  offset : Nat := 0

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

`offset` is an absolute-position offset for KV-cache decoding.
-/
structure RotaryEmbedding.Config where
  /-- Absolute position offset for the first row of RoPE angles. -/
  offset : Nat := 0

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
