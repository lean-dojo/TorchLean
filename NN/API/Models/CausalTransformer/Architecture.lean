/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Causal Transformer Architecture

Configuration, shape families, and graph constructors for GPT-style causal Transformers.
-/

@[expose] public section

namespace TorchLean


open Spec TorchLean TorchLean.Tensor

namespace nn
namespace models
namespace CausalTransformer

/--
Configuration shared by TorchLean's GPT-style causal language models.

The model has the common GPT-2 “shape”:

`embedding → learned positional embedding → (masked self-attention + FFN)×layerCount → LayerNorm`
`→ linear`

The configuration is independent of how token ids enter the model. One-hot, bounded-token, and
pre-embedded constructors reuse the same Transformer width, depth, and output vocabulary.
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
  activation : Activation.Kind := .gelu
  /-- Dropout probability for attention and feed-forward outputs. -/
  dropout? : Option Float := none
  /-- Drop attention probabilities before value aggregation, independently of residual dropout. -/
  attentionDropout? : Option Float := none
  /-- Drop activated FFN hidden units before their output projection. -/
  feedForwardDropout? : Option Float := none
  /-- Enable separate query, key, and value biases; `false` keeps the existing parameter layout. -/
  attentionInputBias : Bool := false
  /-- Use pre-normalized Transformer blocks, as in GPT-2. -/
  normalizeFirst : Bool := true
  /-- Add a trainable bias after each attention output projection, as in GPT-2. -/
  attentionOutputBias : Bool := true
  /-- Shared initialization for embedding and projection weights. `none` keeps layer defaults. -/
  parameterInitialization? : Option Init.Scheme := none
  /--
  Initialization for projections whose outputs are added to residual streams.

  GPT-2 scales these weights by network depth. Keeping the setting explicit lets other causal
  Transformers use their own residual initialization without changing the block implementation.
  -/
  residualProjectionInitialization? : Option Init.Scheme := none
deriving Repr

/-- Transformer width implied by `headCount * headWidth`. -/
def Config.modelWidth (config : Config) : Nat :=
  config.headCount * config.headWidth

namespace Config

/-- Validate the hidden Transformer independently of its token-vocabulary boundary. -/
def validateBody (config : Config) : Except String Unit := do
  if config.sequenceLength = 0 then
    throw "CausalTransformer: sequence length must be positive"
  let block : nn.TransformerEncoder.Block.Config :=
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
      weightInitialization? := config.parameterInitialization?
      residualOutputInitialization? :=
        config.residualProjectionInitialization? }
  block.validate (kind := "CausalTransformer")

/-- Validate the complete language-model configuration before allocating any parameters. -/
def validate (config : Config) : Except String Unit := do
  config.validateBody
  if config.vocabularySize = 0 then
    throw "CausalTransformer: vocabulary size must be positive"

end Config

/-- Bounded token-id tensor shape. -/
abbrev Config.tokenShape (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.sequenceLength

/-- Per-token vocabulary tensor shape used by one-hot inputs and output logits. -/
abbrev Config.vocabularyShape (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape.concat [config.sequenceLength, config.vocabularySize]

/-- Embedded-token tensor shape. -/
abbrev Config.embeddingShape (config : Config) (batchShape : Shape := []) : Shape :=
  batchShape.concat [config.sequenceLength, config.modelWidth]

/--
Causal Transformer hidden-state stack after token embeddings have been computed.

The stack adds learned positions, applies causally masked Transformer blocks, and finishes with
LayerNorm. It deliberately has no vocabulary projection. Language models can therefore choose an
independent output head or reuse their token-embedding matrix.
-/
def hidden (config : Config) (batchShape : Shape := [])
    : nn.Builder
      (nn.Sequential
        (config.embeddingShape batchShape)
        (config.embeddingShape batchShape)) :=
  match config.validateBody with
  | .error message =>
    pure <| nn.Internal.invalidConfiguration
        (config.embeddingShape batchShape) (config.embeddingShape batchShape)
        "CausalTransformer" message
  | .ok () =>
      if hWidth : config.modelWidth = 0 then
        pure <| nn.Internal.invalidConfiguration
          (config.embeddingShape batchShape)
          (config.embeddingShape batchShape)
          "CausalTransformer"
          "CausalTransformer: model width must be positive"
      else
        letI : NeZero config.modelWidth := ⟨hWidth⟩
        let modelWidth := config.modelWidth
        let encoderConfig : nn.TransformerEncoder.Stack.Config :=
          { layerCount := config.layerCount
            block :=
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
                weightInitialization? := config.parameterInitialization?
                residualOutputInitialization? :=
                  config.residualProjectionInitialization? } }
        let positionInitialization :=
          config.parameterInitialization?.getD (.uniform (-0.02) 0.02)
        do
          let builtPositionalEmbedding ← nn.learnedPositionalEmbedding batchShape
            (sequenceLength := config.sequenceLength) (embeddingWidth := modelWidth)
            { initialization := positionInitialization }
          let positionalEmbedding :
              nn.Sequential
                (config.embeddingShape batchShape)
                (config.embeddingShape batchShape) := by
            simpa only [modelWidth, Config.embeddingShape, Config.modelWidth] using
              builtPositionalEmbedding
          let builtTransformerBlocks ← nn.transformerEncoderStack encoderConfig
            (batchShape := batchShape)
            (sequenceLength := config.sequenceLength) (modelWidth := modelWidth)
            (mask := some (Spec.causalMask config.sequenceLength))
          let transformerBlocks :
              nn.Sequential
                (config.embeddingShape batchShape)
                (config.embeddingShape batchShape) := by
            simpa only [modelWidth, Config.embeddingShape, Config.modelWidth] using
              builtTransformerBlocks
          let builtNormalization ←
            nn.layerNorm (batchShape.appendDim config.sequenceLength)
              (width := modelWidth)
          let normalization :
              nn.Sequential
                (config.embeddingShape batchShape)
                (config.embeddingShape batchShape) := by
            simpa [modelWidth, Config.embeddingShape, Config.modelWidth,
              Shape.appendDim_appendDim_eq_concat] using builtNormalization
          pure (positionalEmbedding >>> transformerBlocks >>> normalization)

/--
GPT-style causal Transformer body with an independent affine vocabulary head.

Use `hidden` when the caller needs hidden states or a tied
token-embedding/output matrix.
-/
def fromEmbeddings (config : Config) (batchShape : Shape := [])
    : nn.Builder
      (nn.Sequential
        (config.embeddingShape batchShape)
        (config.vocabularyShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.embeddingShape batchShape) (config.vocabularyShape batchShape)
        "CausalTransformer" message
  | .ok () => do
      let hidden ← hidden config batchShape
      let builtOutputProjection ← nn.linear config.modelWidth config.vocabularySize
        (batchShape := batchShape.appendDim config.sequenceLength)
        (config := { weightInitialization? := config.parameterInitialization? })
      let outputProjection :
          nn.Sequential
            (config.embeddingShape batchShape)
            (config.vocabularyShape batchShape) := by
        simpa [Config.embeddingShape, Config.vocabularyShape,
          Shape.appendDim_appendDim_eq_concat] using builtOutputProjection
      pure (hidden >>> outputProjection)

/--
Build a GPT-2-style causal language model over one-hot tokens.

This is the shared constructor used by the runnable GPT-2 examples. It stays in `nn.Builder` so it
composes with the rest of the API-layer model-building interface.
-/
def oneHot (config : Config) (batchShape : Shape := [])
    : nn.Builder
      (nn.Sequential
        (config.vocabularyShape batchShape)
        (config.vocabularyShape batchShape)) :=
  match config.validate with
  | .error message =>
    pure <| nn.Internal.invalidConfiguration
        (config.vocabularyShape batchShape) (config.vocabularyShape batchShape)
        "CausalTransformer" message
  | .ok () =>
      let modelWidth := config.modelWidth
      let embeddingInitialization :=
        config.parameterInitialization?.getD (.uniform (-0.02) 0.02)
      do
        let builtTokenEmbedding ← nn.oneHotEmbedding config.vocabularySize modelWidth
          { weightInitialization := embeddingInitialization }
          (batchShape := batchShape.appendDim config.sequenceLength)
        let tokenEmbedding :
            nn.Sequential
              (config.vocabularyShape batchShape)
              (config.embeddingShape batchShape) := by
          simpa [modelWidth, Config.vocabularyShape, Config.embeddingShape, Config.modelWidth,
            Shape.appendDim_appendDim_eq_concat] using builtTokenEmbedding
        let body ← fromEmbeddings config batchShape
        pure (tokenEmbedding >>> body)

end CausalTransformer
end models
end nn

end TorchLean
