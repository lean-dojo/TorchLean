/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Vision Transformer

Patch embedding is an arbitrary-dimensional convolution. The spatial output is flattened into a
token axis before the Transformer block, so the construction applies equally to one-dimensional
signals, images, volumes, and higher-dimensional grids.
-/

@[expose] public section

namespace TorchLean

namespace nn
namespace models

/-- How a vision Transformer turns encoded tokens into one vector per sample. -/
inductive ViT.Pooling where
  /-- Average every encoded patch token. -/
  | mean
  /-- Prepend a learned class slot and select it after the encoder. -/
  | cls
deriving DecidableEq, Repr

/-- Configuration for a Transformer encoder over patches from a `d`-dimensional spatial domain. -/
structure ViT.EncoderConfig (d : Nat) where
  /-- Number of channels in each input sample. -/
  inputChannels : Nat
  /-- Extent of each spatial axis. -/
  spatial : Tensor Nat [d]
  /-- Convolution that extracts and embeds patches. -/
  patchEmbedding : Convolution.Config d
  /-- Number of attention heads. Must be positive. -/
  headCount : Nat
  /-- Width of each attention head. Must be positive. -/
  headWidth : Nat
  /-- Width of the feed-forward sublayer. -/
  feedForwardWidth : Nat
  /-- Number of Transformer encoder blocks. -/
  layerCount : Nat := 1
  /-- Dropout probability for attention and feed-forward outputs. -/
  dropout? : Option Float := none
  /-- Drop attention probabilities before value aggregation; independent of residual dropout. -/
  attentionDropout? : Option Float := none
  /-- Drop activated FFN hidden units before their output projection. -/
  feedForwardDropout? : Option Float := none
  /-- Enable trainable query, key, and value biases while preserving the legacy default layout. -/
  attentionInputBias : Bool := false
  /-- Classifier readout; `.cls` also prepends a learned class token. -/
  pooling : ViT.Pooling := .mean

/-- Configuration for a vision Transformer classifier. -/
structure ViT.Config (d : Nat) extends ViT.EncoderConfig d where
  /-- Number of classifier outputs per sample. -/
  classCount : Nat

/-- Attach a classifier output width to reusable encoder settings. -/
def ViT.EncoderConfig.classifier {d : Nat}
    (config : ViT.EncoderConfig d) (classCount : Nat) : ViT.Config d :=
  { toEncoderConfig := config, classCount }

/-- Encoder settings embedded in a classifier configuration. -/
def ViT.Config.encoder {d : Nat} (config : ViT.Config d) : ViT.EncoderConfig d :=
  config.toEncoderConfig

namespace ViT.EncoderConfig

/-- Validate patch extraction and the complete Transformer template before allocating parameters. -/
def validate {d : Nat} (config : ViT.EncoderConfig d)
    (kind : String := "ViT") : Except String Unit := do
  if config.inputChannels = 0 then
    throw s!"{kind}: input channel count must be positive"
  if config.spatial.prod = 0 then
    throw s!"{kind}: input spatial dimensions must be positive"
  config.patchEmbedding.validate config.inputChannels config.spatial (kind := kind)
  if (config.patchEmbedding.output config.spatial).prod = 0 then
    throw s!"{kind}: patch embedding must produce at least one patch"
  let block : TransformerEncoder.Block.Config :=
    { headCount := config.headCount
      headWidth := config.headWidth
      feedForwardWidth := config.feedForwardWidth
      activation := .gelu
      dropout? := config.dropout?
      attentionDropout? := config.attentionDropout?
      feedForwardDropout? := config.feedForwardDropout?
      attentionInputBias := config.attentionInputBias
      normalizeFirst := true
      attentionOutputBias := true }
  block.validate (kind := kind)

end ViT.EncoderConfig

namespace ViT.Config

/-- Validate both the reusable encoder and classifier head before construction. -/
def validate {d : Nat} (config : ViT.Config d) : Except String Unit := do
  config.encoder.validate
  if config.classCount = 0 then
    throw "ViT: class count must be positive"

end ViT.Config

/-- Grid produced by patch embedding. -/
def ViT.EncoderConfig.grid {d : Nat} (config : ViT.EncoderConfig d) : Tensor Nat [d] :=
  config.patchEmbedding.output config.spatial

/-- Number of patch tokens. -/
def ViT.EncoderConfig.patchCount {d : Nat} (config : ViT.EncoderConfig d) : Nat :=
  config.grid.prod

/-- Number of slots inserted before the patch sequence. -/
def ViT.EncoderConfig.prefixLength {d : Nat} (config : ViT.EncoderConfig d) : Nat :=
  match config.pooling with
  | .mean => 0
  | .cls => 1

/-- Number of tokens passed through the Transformer encoder. -/
def ViT.EncoderConfig.sequenceLength {d : Nat} (config : ViT.EncoderConfig d) : Nat :=
  config.prefixLength + config.patchCount

/-- Number of scalar features in the complete encoded token sequence. -/
def ViT.EncoderConfig.flattenedWidth {d : Nat} (config : ViT.EncoderConfig d) : Nat :=
  config.sequenceLength * config.patchEmbedding.outChannels

/-- Input shape for any caller-supplied batch shape. -/
abbrev ViT.EncoderConfig.input {d : Nat} (config : ViT.EncoderConfig d)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat ((config.spatial.to Shape).prependDim config.inputChannels)

/-- Patch tensor produced by the embedding convolution. -/
abbrev ViT.EncoderConfig.patches {d : Nat} (config : ViT.EncoderConfig d)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat ((config.grid.to Shape).prependDim config.patchEmbedding.outChannels)

/-- Patch tokens after moving the embedding width to the final axis. -/
abbrev ViT.EncoderConfig.tokens {d : Nat} (config : ViT.EncoderConfig d)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat [config.patchCount, config.patchEmbedding.outChannels]

/-- Tokens consumed and produced by the Transformer stack. -/
abbrev ViT.EncoderConfig.encoded {d : Nat} (config : ViT.EncoderConfig d)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat [config.sequenceLength, config.patchEmbedding.outChannels]

/-- Classifier input shape for any caller-supplied batch shape. -/
abbrev ViT.Config.input {d : Nat} (config : ViT.Config d)
    (batchShape : Shape := []) : Shape :=
  config.encoder.input batchShape

/-- Classifier output shape for the same batch shape as the input. -/
abbrev ViT.Config.output {d : Nat} (config : ViT.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.classCount

namespace Internal

/-- Implementation layer that turns a patch grid into a token sequence. -/
def patchesToTokens {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Layer (config.patches batchShape) (config.tokens batchShape) :=
  { kind := "ViT.PatchesToTokens"
    stateShapes := []
    initState := .nil
    requiresGrad := #[]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => (show m (TorchLean.Runtime.ValueRef
            (m := m) (α := α) (config.tokens batchShape)) from do
          let middle : Shape :=
            batchShape.concat [config.patchEmbedding.outChannels, config.patchCount]
          let flattened ←
            Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := config.patches batchShape) (s₂ := middle) x (by
                -- Both sides are `batchShape` followed by the channel axis and the grid axes. Fire
                -- the grid readout bridge first: a bare `simp` unfolds the readout into a list
                -- product and then the bridge no longer matches.
                simp only [ViT.EncoderConfig.patches, middle, Shape.size_concat,
                  Shape.size_prependDim, Tensor.size_to_shape]
                simp [ViT.EncoderConfig.patchCount, Shape.size])
          let tokens ← Runtime.Autograd.Torch.swapAdjacentAtDepth
            (m := m) (α := α) (s := middle) batchShape.rank flattened
          return (by
            simpa [middle, ViT.EncoderConfig.tokens] using tokens)) }

/-- Prepend one shared, trainable class token to every independently mapped sequence. -/
def prependClassToken {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Sequential
      (config.tokens batchShape)
      (batchShape.concat [1 + config.patchCount, config.patchEmbedding.outChannels]) :=
  let tokenShape : Shape := [1, config.patchEmbedding.outChannels]
  let initialToken : Tensor Float tokenShape :=
    Tensor.zeros (α := Float) tokenShape
  let core : Sequential
      [config.patchCount, config.patchEmbedding.outChannels]
      [1 + config.patchCount, config.patchEmbedding.outChannels] :=
    nn.Sequential.fromLayer
      { kind := "ViT.ClassToken"
        stateShapes := [tokenShape]
        initState := TorchLean.TensorPack.singleton initialToken
        runtimeInit := some (.cons .zeros .nil)
        requiresGrad := #[true]
        forward := fun _ {α} _ _ =>
          fun {m} _ _ =>
            fun classToken x =>
              (show m (TorchLean.Runtime.ValueRef
                  (m := m) (α := α)
                  [1 + config.patchCount, config.patchEmbedding.outChannels]) from do
              let result ← Runtime.Autograd.Torch.concatLeadingAxis
                (m := m) (α := α) classToken x
              return result) }
  by
      simpa only [ViT.EncoderConfig.tokens] using
      nn.mapLeading batchShape core

/-- Implementation layer that moves the embedding width before the token axis. -/
def tokensToChannels {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Sequential
      (config.encoded batchShape)
      (batchShape.concat [config.patchEmbedding.outChannels, config.sequenceLength]) :=
  nn.Sequential.fromLayer
    { kind := "ViT.TokensToChannels"
      stateShapes := []
      initState := .nil
      requiresGrad := #[]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            (show m (TorchLean.Runtime.ValueRef
                (m := m) (α := α)
                (batchShape.concat
                  [config.patchEmbedding.outChannels, config.sequenceLength])) from by
              simpa [ViT.EncoderConfig.encoded] using
                (Runtime.Autograd.Torch.swapAdjacentAtDepth
                  (m := m) (α := α) (s := config.encoded batchShape)
                    batchShape.rank x)) }

/-- Implementation layer for mean-pool classification. -/
def meanVitTokens {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Builder (Sequential (config.encoded batchShape)
      (batchShape.appendDim config.patchEmbedding.outChannels)) := do
  let spatial : Tensor Nat [1] :=
    [config.sequenceLength]
  let builtPool ← globalAvgPool spatial
    (channels := config.patchEmbedding.outChannels) (batchShape := batchShape)
  let pool : Sequential
      (batchShape.concat [config.patchEmbedding.outChannels, config.sequenceLength])
      (batchShape.appendDim config.patchEmbedding.outChannels) := by
    have hSpatialShape : spatial.to Shape = [config.sequenceLength] := by
      simp [spatial]
    simpa only [hSpatialShape] using builtPool
  pure (nn.compose![tokensToChannels config batchShape, pool])

/-- Implementation layer for class-token classification. -/
def firstVitToken {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Sequential (config.encoded batchShape)
      (batchShape.appendDim config.patchEmbedding.outChannels) :=
  if hSequence : config.sequenceLength = 0 then
    nn.Internal.invalidConfiguration
      (config.encoded batchShape)
      (batchShape.appendDim config.patchEmbedding.outChannels)
      "ViT.FirstToken"
      "ViT.FirstToken: sequence length must be positive"
  else
    let core : Sequential
        [config.sequenceLength, config.patchEmbedding.outChannels]
        [config.patchEmbedding.outChannels] :=
      nn.Sequential.fromLayer
        { kind := "ViT.FirstToken"
          stateShapes := []
          initState := .nil
          requiresGrad := #[]
          forward := fun _ {α} _ _ =>
            fun {m} _ _ =>
            fun x =>
              (show m (TorchLean.Runtime.ValueRef
                  (m := m) (α := α) [config.patchEmbedding.outChannels]) from do
                have hOne : 0 + 1 ≤ config.sequenceLength := by
                  simpa using Nat.one_le_iff_ne_zero.mpr hSequence
                let row ← Runtime.Autograd.Torch.sliceLeadingAxisRange
                  (m := m) (α := α) 0 1 hOne x
                Runtime.Autograd.Torch.reshape
                  (m := m) (α := α)
                  (s₁ := [1, config.patchEmbedding.outChannels])
                  (s₂ := [config.patchEmbedding.outChannels])
                  row (by simp [Shape.size])) }
    by
      simpa only [ViT.EncoderConfig.encoded, Shape.appendDim_eq_concat] using
        nn.mapLeading batchShape core

end Internal

/--
Build the patch and Transformer portion of a vision transformer.

Mean readout leaves the patch sequence unchanged. Class-token readout prepends one learned token
before the Transformer. Classification, reconstruction, and other tasks can attach their own heads
without rebuilding the patch pipeline.
-/
def vitEncoder {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Builder (Sequential (config.input batchShape) (config.encoded batchShape)) :=
  match config.validate with
  | .error message =>
    pure <| nn.Internal.invalidConfiguration
        (config.input batchShape) (config.encoded batchShape) "ViT" message
  | .ok () => do
      let patchEmbedding ←
        conv config.spatial config.patchEmbedding
          (batchShape := batchShape) (inputChannels := config.inputChannels)
      let tokenPrefix :
          Sequential (config.tokens batchShape) (config.encoded batchShape) :=
        match hPooling : config.pooling with
        | .mean => by
            simpa [ViT.EncoderConfig.tokens, ViT.EncoderConfig.encoded,
              ViT.EncoderConfig.sequenceLength, ViT.EncoderConfig.prefixLength, hPooling] using
              (nn.Sequential.identity (config.tokens batchShape))
        | .cls => by
            simpa [ViT.EncoderConfig.encoded, ViT.EncoderConfig.sequenceLength,
              ViT.EncoderConfig.prefixLength, hPooling] using
              Internal.prependClassToken config batchShape
      let positions ← learnedPositionalEmbedding batchShape
        (sequenceLength := config.sequenceLength)
        (embeddingWidth := config.patchEmbedding.outChannels)
      let builtEncoder ← transformerEncoderStack
        (sequenceLength := config.sequenceLength)
        (modelWidth := config.patchEmbedding.outChannels)
        { layerCount := config.layerCount
          block :=
            { headCount := config.headCount
              headWidth := config.headWidth
              feedForwardWidth := config.feedForwardWidth
              activation := .gelu
              dropout? := config.dropout?
              attentionDropout? := config.attentionDropout?
              feedForwardDropout? := config.feedForwardDropout?
              attentionInputBias := config.attentionInputBias
              normalizeFirst := true
              attentionOutputBias := true } }
        (batchShape := batchShape)
      let encoder : Sequential
          (config.encoded batchShape) (config.encoded batchShape) := by
        simpa only [ViT.EncoderConfig.encoded] using builtEncoder
      let builtNormalization ←
        layerNorm (batchShape.appendDim config.sequenceLength)
          (width := config.patchEmbedding.outChannels)
      let normalization : Sequential
          (config.encoded batchShape) (config.encoded batchShape) := by
        simpa [ViT.EncoderConfig.encoded, Shape.appendDim_appendDim_eq_concat] using
          builtNormalization
      pure <| patchEmbedding >>>
        nn.Sequential.fromLayer (Internal.patchesToTokens config batchShape) >>>
        tokenPrefix >>> positions >>> encoder >>> normalization

/-- Build a vision Transformer encoder followed by a linear classifier. -/
def vit {d : Nat} (config : ViT.Config d) (batchShape : Shape := []) :
    Builder (Sequential (config.input batchShape) (config.output batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.input batchShape) (config.output batchShape) "ViT" message
  | .ok () => do
      let encoderConfig := config.encoder
      let encoder ← vitEncoder encoderConfig batchShape
      let pool : Sequential
          (encoderConfig.encoded batchShape)
          (batchShape.appendDim encoderConfig.patchEmbedding.outChannels) ←
        match encoderConfig.pooling with
        | .mean => Internal.meanVitTokens encoderConfig batchShape
        | .cls => pure (Internal.firstVitToken encoderConfig batchShape)
      let classifier ←
        linear encoderConfig.patchEmbedding.outChannels config.classCount
          (batchShape := batchShape)
      pure <| encoder >>> pool >>> classifier

end models
end nn
end TorchLean
