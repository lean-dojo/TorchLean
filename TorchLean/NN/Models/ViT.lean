/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder

/-!
# Vision Transformer

`vit` builds a vision Transformer classifier from a `ViT.Config`, and `vitEncoder` builds the
encoder alone from a `ViT.EncoderConfig`. Patch embedding is a convolution over any number of
spatial axes. Its output grid is flattened into one token axis before the Transformer stack, so
the same construction covers one-dimensional signals, images, volumes and higher-dimensional
grids.

We built it from Dosovitskiy et al., Section 3.1. The embedding convolution is eq. (1)'s linear
map `E` on flattened patches when its kernel and stride both equal the patch size, and learned 1D
positions are added to every token. Each block normalizes first, as in eqs. (2) and (3). After
the stack one more `layerNorm` runs over every token, which is eq. (4)'s `y = LN(z⁰_L)` once the
class token is read out. The head is a single linear layer, the paper's fine-tuning head.

A few things we chose ourselves. The block MLP uses the tanh GELU, PyTorch's
`torch.nn.GELU(approximate='tanh')`, where the paper just says GELU. The default readout averages
the patch tokens: Appendix D.3 finds that this and the class token work about as well, each at its
own learning rate. With `.cls` the class token starts at zero. Where Appendix B.1 uses dropout, it
also applies it right after the positions are added, and we have none there. The query, key and
value projections have no bias unless `attentionInputBias` is set, while the output projection
always has one.

References:
- Alexey Dosovitskiy et al. (2021), "An Image is Worth 16x16 Words: Transformers for Image
  Recognition at Scale", ICLR 2021, arXiv:2010.11929, Section 3.1, eqs. (1) to (4), Appendices
  B.1 and D.3.
- `torch.nn.GELU` (PyTorch 2.8 docs), `approximate='tanh'`.
-/

@[expose] public section

open scoped TorchLean.nn.Sequential
open TorchLean.Shape (size_append size_cons)

namespace TorchLean

namespace nn

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
  patchEmbedding : Conv.Config d
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
  /-- Drop attention probabilities before value aggregation, independently of residual dropout. -/
  attentionDropout? : Option Float := none
  /-- Drop activated FFN hidden units before their output projection. -/
  feedForwardDropout? : Option Float := none
  /-- Enable trainable query, key, and value biases. -/
  attentionInputBias : Bool := false
  /-- Classifier readout, where `cls` also makes the encoder prepend a learned class token. -/
  pooling : ViT.Pooling := .mean

/-- Configuration for a vision Transformer classifier. -/
structure ViT.Config (d : Nat) extends ViT.EncoderConfig d where
  /-- Number of classifier outputs per sample. -/
  classCount : Nat

namespace ViT.EncoderConfig

/-- Transformer block settings shared by encoder validation and construction. -/
def block {d : Nat} (config : ViT.EncoderConfig d) : TransformerBlock.Config :=
  { headCount := config.headCount
    headWidth := config.headWidth
    feedForwardWidth := config.feedForwardWidth
    activation := .geluTanh
    dropout? := config.dropout?
    attentionDropout? := config.attentionDropout?
    feedForwardDropout? := config.feedForwardDropout?
    attentionInputBias := config.attentionInputBias
    normalizeFirst := true
    attentionOutputBias := true }

/-- Validate patch extraction and the complete Transformer template before allocating parameters. -/
def validate {d : Nat} (config : ViT.EncoderConfig d)
    (kind : String := "ViT") : Except String Unit := do
  if config.inputChannels = 0 then
    throw s!"{kind}: input channel count must be positive"
  if config.spatial.prod = 0 then
    throw s!"{kind}: input spatial dimensions must be positive"
  config.patchEmbedding.validate config.inputChannels config.spatial (kind := kind)
  if (config.patchEmbedding.convOutputSpatial config.spatial).prod = 0 then
    throw s!"{kind}: patch embedding must produce at least one patch"
  config.block.validate (kind := kind)

end ViT.EncoderConfig

namespace ViT.Config

/-- Validate both the reusable encoder and classifier head before construction. -/
def validate {d : Nat} (config : ViT.Config d) : Except String Unit := do
  config.toEncoderConfig.validate
  if config.classCount = 0 then
    throw "ViT: class count must be positive"

end ViT.Config

/-- Grid produced by patch embedding. -/
def ViT.EncoderConfig.patchGrid {d : Nat} (config : ViT.EncoderConfig d) : Tensor Nat [d] :=
  config.patchEmbedding.convOutputSpatial config.spatial

/-- Number of patch tokens. -/
def ViT.EncoderConfig.patchCount {d : Nat} (config : ViT.EncoderConfig d) : Nat :=
  config.patchGrid.prod

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
abbrev ViT.EncoderConfig.inputShape {d : Nat} (config : ViT.EncoderConfig d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ (Shape.dim config.inputChannels (config.spatial.to Shape))

/-- Patch tensor produced by the embedding convolution. -/
abbrev ViT.EncoderConfig.patchShape {d : Nat} (config : ViT.EncoderConfig d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ (Shape.dim config.patchEmbedding.outChannels (config.patchGrid.to Shape))

/-- Patch tokens after moving the embedding width to the final axis. -/
abbrev ViT.EncoderConfig.patchTokenShape {d : Nat} (config : ViT.EncoderConfig d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.patchCount, config.patchEmbedding.outChannels]

/-- Tokens consumed and produced by the Transformer stack. -/
abbrev ViT.EncoderConfig.outputShape {d : Nat} (config : ViT.EncoderConfig d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.sequenceLength, config.patchEmbedding.outChannels]

/-- Classifier input shape for any caller-supplied batch shape. -/
abbrev ViT.Config.inputShape {d : Nat} (config : ViT.Config d)
    (batchShape : Shape := []) : Shape :=
  config.toEncoderConfig.inputShape batchShape

/-- Classifier output shape for the same batch shape as the input. -/
abbrev ViT.Config.outputShape {d : Nat} (config : ViT.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.classCount]

namespace ViT

/-- Implementation layer that turns a patch grid into a token sequence. -/
def patchesToTokens {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Layer (config.patchShape batchShape) (config.patchTokenShape batchShape) :=
  { kind := "ViT.PatchesToTokens"
    stateShapes := []
    initState := .nil
    requiresGrad := #[]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => (show m (Ops.Ref
            (m := m) (α := α) (config.patchTokenShape batchShape)) from do
          let middle : Shape :=
            batchShape ++ [config.patchEmbedding.outChannels, config.patchCount]
          let flattened ←
            Ops.apply (m := m) (α := α)
                (.reshape (s := config.patchShape batchShape) (t := middle) (by
                -- Both sides start with `batchShape` and the channel axis, and the target flattens
                -- the grid axes into `patchCount`. Fire the grid readout bridge first, because a
                -- bare `simp` unfolds the readout into a list product the bridge cannot match.
                simp only [ViT.EncoderConfig.patchShape, middle, Shape.size_append,
                  Shape.size_cons, Tensor.size_to_shape]
                simp [ViT.EncoderConfig.patchCount])) (x, ())
          let tokens ← Ops.apply (m := m) (α := α)
              (.permute (s := middle) (Shape.adjacentSwap middle batchShape.rank)
                (Shape.swapAdjacentAtDepth middle batchShape.rank) (by tensor_shape))
              (flattened, ())
          return (by
            simpa [middle, ViT.EncoderConfig.patchTokenShape] using tokens)) }

/-- Prepend one trainable class token, initialized to zeros and shared across `batchShape`. -/
def prependClassToken {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Sequential
      (config.patchTokenShape batchShape)
      (batchShape ++ [1 + config.patchCount, config.patchEmbedding.outChannels]) :=
  let tokenShape : Shape := [1, config.patchEmbedding.outChannels]
  let initialToken : Tensor Float tokenShape :=
    Tensor.zeros (α := Float) tokenShape
  let core : Sequential
      [config.patchCount, config.patchEmbedding.outChannels]
      [1 + config.patchCount, config.patchEmbedding.outChannels] :=
    nn.Sequential.fromLayer
      { kind := "ViT.ClassToken"
        stateShapes := [tokenShape]
        initState := TorchLean.Pack.singleton initialToken
        runtimeInit := some (.cons (.scheme .zeros) .nil)
        requiresGrad := #[true]
        forward := fun _ {α} _ _ =>
          fun {m} _ _ =>
            fun classToken x =>
              (show m (Ops.Ref
                  (m := m) (α := α)
                  [1 + config.patchCount, config.patchEmbedding.outChannels]) from do
              let result ← Ops.apply (m := m) (α := α)
                  (.cat (s := .dim 0 _) ⟨0, by simp⟩ [_, _]) (classToken, x, ())
              return result) }
  by
      simpa only [ViT.EncoderConfig.patchTokenShape] using
      nn.mapLeading batchShape core

/-- Implementation layer that moves the embedding width before the token axis. -/
def tokensToChannels {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Sequential
      (config.outputShape batchShape)
      (batchShape ++ [config.patchEmbedding.outChannels, config.sequenceLength]) :=
  nn.Sequential.fromLayer
    { kind := "ViT.TokensToChannels"
      stateShapes := []
      initState := .nil
      requiresGrad := #[]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            (show m (Ops.Ref
                (m := m) (α := α)
                (batchShape ++
                  [config.patchEmbedding.outChannels, config.sequenceLength])) from by
              simpa [ViT.EncoderConfig.outputShape] using
                (Ops.apply (m := m) (α := α)
                      (.permute (s := config.outputShape batchShape)
                        (Shape.adjacentSwap (config.outputShape batchShape) batchShape.rank)
                        (Shape.swapAdjacentAtDepth (config.outputShape batchShape)
                              batchShape.rank)
                        (by tensor_shape)) (x, ()))) }

/-- Implementation layer for mean-pool classification. -/
def meanTokens {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Builder (Sequential (config.outputShape batchShape)
      (batchShape ++ [config.patchEmbedding.outChannels])) := do
  let spatial : Tensor Nat [1] :=
    [config.sequenceLength]
  let builtPool ← globalAvgPool spatial
    (channels := config.patchEmbedding.outChannels) (batchShape := batchShape)
  let pool : Sequential
      (batchShape ++ [config.patchEmbedding.outChannels, config.sequenceLength])
      (batchShape ++ [config.patchEmbedding.outChannels]) := by
    have hSpatialShape : spatial.to Shape = [config.sequenceLength] := by
      simp [spatial]
    simpa only [hSpatialShape] using builtPool
  pure (nn.compose![tokensToChannels config batchShape, pool])

/-- Implementation layer for class-token classification. -/
def firstToken {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Sequential (config.outputShape batchShape)
      (batchShape ++ [config.patchEmbedding.outChannels]) :=
  if hSequence : config.sequenceLength = 0 then
    nn.Sequential.invalid
      (config.outputShape batchShape)
      (batchShape ++ [config.patchEmbedding.outChannels])
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
              (show m (Ops.Ref
                  (m := m) (α := α) [config.patchEmbedding.outChannels]) from do
                have hOne : 0 + 1 ≤ config.sequenceLength := by
                  simpa using Nat.one_le_iff_ne_zero.mpr hSequence
                let row ←
                    (Ops.Ref.cast
                      (Shape.slice_singleton (.dim _ _) ⟨0, by simp⟩ 0 1) <$>
                      Ops.apply (m := m) (α := α)
                        (.slice (s := .dim _ _) [⟨0, by simp⟩] [0] [0 + 1] [1]
                          (Shape.sliceValid_singleton _ ⟨0, by simp⟩ 0 1 hOne)) (x, ()))
                Ops.apply (m := m) (α := α)
                    (.reshape (s := [1, config.patchEmbedding.outChannels])
                      (t := [config.patchEmbedding.outChannels]) (by simp))
                    (row, ())) }
    by
      simpa only [ViT.EncoderConfig.outputShape] using
        nn.mapLeading batchShape core

end ViT

/--
Build the patch embedding, learned positions, Transformer stack and final LayerNorm of a vision
Transformer.

Mean readout adds no token to the patch sequence, while class-token readout prepends one learned
token before the positions are added. Classification, reconstruction and other tasks attach their
own heads without rebuilding the patch pipeline.
-/
def vitEncoder {d : Nat} (config : ViT.EncoderConfig d) (batchShape : Shape := []) :
    Builder (Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
    pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.outputShape batchShape) "ViT" message
  | .ok () => do
      let patchEmbedding ←
        conv config.spatial config.patchEmbedding
          (batchShape := batchShape) (inputChannels := config.inputChannels)
      let tokenPrefix :
          Sequential (config.patchTokenShape batchShape) (config.outputShape batchShape) :=
        match hPooling : config.pooling with
        | .mean => by
            simpa [ViT.EncoderConfig.patchTokenShape, ViT.EncoderConfig.outputShape,
              ViT.EncoderConfig.sequenceLength, ViT.EncoderConfig.prefixLength, hPooling] using
              (Sequential.id (config.patchTokenShape batchShape))
        | .cls => by
            simpa [ViT.EncoderConfig.outputShape, ViT.EncoderConfig.sequenceLength,
              ViT.EncoderConfig.prefixLength, hPooling] using
              ViT.prependClassToken config batchShape
      let positions ← learnedPositionalEmbedding batchShape
        (sequenceLength := config.sequenceLength)
        (embeddingWidth := config.patchEmbedding.outChannels)
      let builtEncoder ← transformerStack
        (sequenceLength := config.sequenceLength)
        (modelWidth := config.patchEmbedding.outChannels)
        { layerCount := config.layerCount
          block := config.block }
        (batchShape := batchShape)
      let encoder : Sequential
          (config.outputShape batchShape) (config.outputShape batchShape) := by
        simpa only [ViT.EncoderConfig.outputShape] using builtEncoder
      let builtNormalization ←
        layerNorm (batchShape ++ [config.sequenceLength])
          (width := config.patchEmbedding.outChannels)
      let normalization : Sequential
          (config.outputShape batchShape) (config.outputShape batchShape) := by
        simpa [ViT.EncoderConfig.outputShape] using
          builtNormalization
      pure <| patchEmbedding >>>
        nn.Sequential.fromLayer (ViT.patchesToTokens config batchShape) >>>
        tokenPrefix >>> positions >>> encoder >>> normalization

/-- Build `vitEncoder` followed by the configured token readout and a linear classifier. -/
def vit {d : Nat} (config : ViT.Config d) (batchShape : Shape := []) :
    Builder (Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.outputShape batchShape) "ViT" message
  | .ok () => do
      let encoderConfig := config.toEncoderConfig
      let encoder ← vitEncoder encoderConfig batchShape
      let pool : Sequential
          (encoderConfig.outputShape batchShape)
          (batchShape ++ [encoderConfig.patchEmbedding.outChannels]) ←
        match encoderConfig.pooling with
        | .mean => ViT.meanTokens encoderConfig batchShape
        | .cls => pure (ViT.firstToken encoderConfig batchShape)
      let classifier ←
        linear encoderConfig.patchEmbedding.outChannels config.classCount
          (batchShape := batchShape)
      pure <| encoder >>> pool >>> classifier

end nn

end TorchLean
