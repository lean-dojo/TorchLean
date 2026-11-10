/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder

/-!
# Causal Transformer Architecture

`Config` describes a GPT-style causal Transformer, and `Config.tokenShape`,
`Config.vocabularyShape` and `Config.embeddingShape` give its tensor shapes for any batch shape.
`hidden` builds the stack from token embeddings to the final LayerNorm, `fromEmbeddings` adds an
affine vocabulary head and `oneHot` puts a one-hot token embedding in front.

We took the layout from GPT-2 (Radford et al., 2019, Section 2.3), and the tanh GELU default
from GPT-2's released code. Some defaults differ from that code. Its `c_attn` projection gives
queries, keys and values a bias, and ours has none unless `attentionInputBias` is set. Its token
and position tables start from normals with standard deviations `0.02` and `0.01`, while ours
start uniform on `[-0.02, 0.02]`. Section 2.3 also scales the residual projections by `1/√N` at
initialization, for `N` residual layers. We leave that to `residualInit?`, which falls back to
`parameterInit?` when it's `none`. No theorem or torch test covers these builders.

References:
- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever (2019),
  "Language Models are Unsupervised Multitask Learners", OpenAI report, Section 2.3.
- `openai/gpt-2` on GitHub, `src/model.py`: `gelu`, `conv1d`, `block` and the initializers of
  `wte` and `wpe`.
-/

@[expose] public section

open scoped TorchLean.nn.Sequential

namespace TorchLean


open TorchLean.Tensor

namespace nn
namespace CausalTransformer

/--
Configuration shared by TorchLean's GPT-style causal language models, all with the GPT-2 layout:

`embedding → learned positional embedding → (masked self-attention + FFN)×layerCount → LayerNorm`
`→ linear`

The configuration does not fix how token ids enter the model, so `oneHot`, `indexed`, `tied` and
`fromEmbeddings` share one Transformer width, depth and output vocabulary.
-/
structure Config where
  /-- Number of token positions processed in one model invocation. Must be positive. -/
  sequenceLength : Nat
  /-- Number of token categories accepted by the input and vocabulary head. Must be positive. -/
  vocabularySize : Nat
  /-- Number of parallel self-attention heads in each Transformer block. Must be positive. -/
  headCount : Nat
  /-- Width of each attention head. Must be positive. -/
  headWidth : Nat
  /-- Hidden width of each block's position-wise feed-forward network. -/
  feedForwardWidth : Nat
  /-- Number of stacked causal Transformer blocks. -/
  layerCount : Nat
  /-- Feed-forward activation used in every Transformer block. -/
  activation : Activation.Kind := .geluTanh
  /-- Dropout probability for attention and feed-forward outputs. -/
  dropout? : Option Float := none
  /-- Drop attention probabilities before value aggregation, independently of residual dropout. -/
  attentionDropout? : Option Float := none
  /-- Drop activated FFN hidden units before their output projection. -/
  feedForwardDropout? : Option Float := none
  /-- Add a trainable bias to each query, key, and value projection. -/
  attentionInputBias : Bool := false
  /-- Use pre-normalized Transformer blocks, as in GPT-2. -/
  normalizeFirst : Bool := true
  /-- Add a trainable bias after each attention output projection, as in GPT-2. -/
  attentionOutputBias : Bool := true
  /--
  Shared initialization for embedding and projection weights. With `none`, projections keep their
  layer defaults and the embedding tables use `.uniform (-0.02) 0.02`.
  -/
  parameterInit? : Option Init.Scheme := none
  /--
  Initialization for projections whose outputs are added to residual streams. With `none` they use
  `parameterInit?`.

  GPT-2 scales these weights by network depth. A separate field lets each causal Transformer choose
  its own residual initialization without changing the block implementation.
  -/
  residualInit? : Option Init.Scheme := none
deriving Repr

/-- Transformer width implied by `headCount * headWidth`. -/
def Config.modelWidth (config : Config) : Nat :=
  config.headCount * config.headWidth

namespace Config

/-- Token and positional embedding initialization, including the fallback for `none`. -/
def embeddingInit (config : Config) : Init.Scheme :=
  config.parameterInit?.getD (.uniform (-0.02) 0.02)

/-- Block settings shared by validation and construction of the hidden stack. -/
def block (config : Config) : nn.TransformerBlock.Config :=
    { headCount := config.headCount
      headWidth := config.headWidth
      feedForwardWidth := config.feedForwardWidth
      activation := config.activation
      dropout? := config.dropout?
      attentionDropout? := config.attentionDropout?
      feedForwardDropout? := config.feedForwardDropout?
      attentionInputBias := config.attentionInputBias
      normalizeFirst := config.normalizeFirst
      attentionOutputBias := config.attentionOutputBias
      weightInit? := config.parameterInit?
      residualInit? :=
        config.residualInit? }

/-- Validate the hidden Transformer independently of its token-vocabulary boundary. -/
def validateBody (config : Config) : Except String Unit := do
  if config.sequenceLength = 0 then
    throw "CausalTransformer: sequence length must be positive"
  config.block.validate (kind := "CausalTransformer")

/-- Validate the complete language-model configuration before allocating any parameters. -/
def validate (config : Config) : Except String Unit := do
  config.validateBody
  if config.vocabularySize = 0 then
    throw "CausalTransformer: vocabulary size must be positive"

end Config

/-- Bounded token-id tensor shape. -/
abbrev Config.tokenShape (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.sequenceLength]

/-- Per-token vocabulary tensor shape used by one-hot inputs and output logits. -/
abbrev Config.vocabularyShape (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.sequenceLength, config.vocabularySize]

/-- Embedded-token tensor shape. -/
abbrev Config.embeddingShape (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.sequenceLength, config.modelWidth]

/--
Causal Transformer hidden-state stack after token embeddings have been computed.

The stack adds learned positions, applies causally masked Transformer blocks and finishes with
LayerNorm. It has no vocabulary projection, so a language model can attach an independent output
head or reuse its token-embedding matrix.
-/
def hidden (config : Config) (batchShape : Shape := [])
    : nn.Builder
      (nn.Sequential
        (config.embeddingShape batchShape)
        (config.embeddingShape batchShape)) :=
  match config.validateBody with
  | .error message =>
    pure <| nn.Sequential.invalid
        (config.embeddingShape batchShape) (config.embeddingShape batchShape)
        "CausalTransformer" message
  | .ok () =>
      if hWidth : config.modelWidth = 0 then
        pure <| nn.Sequential.invalid
          (config.embeddingShape batchShape)
          (config.embeddingShape batchShape)
          "CausalTransformer"
          "CausalTransformer: model width must be positive"
      else
        letI : NeZero config.modelWidth := ⟨hWidth⟩
        let modelWidth := config.modelWidth
        let encoderConfig : nn.TransformerStack.Config :=
          { layerCount := config.layerCount
            block := config.block }
        do
          let builtPositionalEmbedding ← nn.learnedPositionalEmbedding batchShape
            (sequenceLength := config.sequenceLength) (embeddingWidth := modelWidth)
            { init := config.embeddingInit }
          let positionalEmbedding :
              nn.Sequential
                (config.embeddingShape batchShape)
                (config.embeddingShape batchShape) := by
            simpa only [modelWidth, Config.embeddingShape, Config.modelWidth] using
              builtPositionalEmbedding
          let builtTransformerBlocks ← nn.transformerStack encoderConfig
            (batchShape := batchShape)
            (sequenceLength := config.sequenceLength) (modelWidth := modelWidth)
            (mask := some (Attention.causalMask config.sequenceLength))
          let transformerBlocks :
              nn.Sequential
                (config.embeddingShape batchShape)
                (config.embeddingShape batchShape) := by
            simpa only [modelWidth, Config.embeddingShape, Config.modelWidth] using
              builtTransformerBlocks
          let builtNormalization ←
            nn.layerNorm (batchShape ++ [config.sequenceLength])
              (width := modelWidth)
          let normalization :
              nn.Sequential
                (config.embeddingShape batchShape)
                (config.embeddingShape batchShape) := by
            simpa [modelWidth, Config.embeddingShape, Config.modelWidth] using builtNormalization
          pure (positionalEmbedding >>> transformerBlocks >>> normalization)

/--
GPT-style causal Transformer from token embeddings to vocabulary logits: `hidden` followed by an
independent affine vocabulary head.

Call `hidden` for hidden states or to tie the token-embedding and output matrices.
-/
def fromEmbeddings (config : Config) (batchShape : Shape := [])
    : nn.Builder
      (nn.Sequential
        (config.embeddingShape batchShape)
        (config.vocabularyShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.embeddingShape batchShape) (config.vocabularyShape batchShape)
        "CausalTransformer" message
  | .ok () => do
      let hidden ← hidden config batchShape
      let builtOutputProjection ← nn.linear config.modelWidth config.vocabularySize
        (batchShape := batchShape ++ [config.sequenceLength])
        (config := { weightInit? := config.parameterInit? })
      let outputProjection :
          nn.Sequential
            (config.embeddingShape batchShape)
            (config.vocabularyShape batchShape) := by
        simpa [Config.embeddingShape, Config.vocabularyShape] using builtOutputProjection
      pure (hidden >>> outputProjection)

/--
Build a GPT-2-style causal language model over one-hot tokens.

The runnable GPT-2 examples share this constructor, and as an `nn.Builder` it composes with the
other API-layer model builders.
-/
def oneHot (config : Config) (batchShape : Shape := [])
    : nn.Builder
      (nn.Sequential
        (config.vocabularyShape batchShape)
        (config.vocabularyShape batchShape)) :=
  match config.validate with
  | .error message =>
    pure <| nn.Sequential.invalid
        (config.vocabularyShape batchShape) (config.vocabularyShape batchShape)
        "CausalTransformer" message
  | .ok () =>
      let modelWidth := config.modelWidth
      do
        let builtTokenEmbedding ← nn.oneHotEmbedding config.vocabularySize modelWidth
          { weightInit := config.embeddingInit }
          (batchShape := batchShape ++ [config.sequenceLength])
        let tokenEmbedding :
            nn.Sequential
              (config.vocabularyShape batchShape)
              (config.embeddingShape batchShape) := by
          simpa [modelWidth, Config.vocabularyShape, Config.embeddingShape, Config.modelWidth] using
            builtTokenEmbedding
        let body ← fromEmbeddings config batchShape
        pure (tokenEmbedding >>> body)

end CausalTransformer
end nn

end TorchLean
