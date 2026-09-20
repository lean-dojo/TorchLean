/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.Vit
public import NN.API.Models.Generative -- shake: keep

/-!
# Self-Supervised Model Constructors

Most SSL machinery belongs in `TorchLean.ssl`: masks, tensor-to-training-sample transforms, and
objective-facing helpers should work with any compatible model.

This file keeps architecture-level conveniences. The compact masked patch reconstructor below
encodes every patch and reconstructs through a dense head; the SSL objective can also train other
architectures.
-/

@[expose] public section

namespace TorchLean


open Spec TorchLean TorchLean.Tensor

namespace nn
namespace models

/-! ## Masked patch reconstruction -/

/--
Configuration for a compact masked patch-transformer reconstructor.

The input/output contract is MAE-style: a masked channel/spatial tensor is mapped to one flattened
reconstruction vector per batch position.

`reconstructionWidth` can be the full image size (`C*H*W`) or a prefix for faster experiments.
-/
structure ViT.MaskedPatchReconstructor.Config (d : Nat) where
  /-- Patch-transformer encoder configuration. -/
  encoder : ViT.EncoderConfig d
  /-- Number of reconstructed output coordinates. -/
  reconstructionWidth : Nat

namespace ViT.MaskedPatchReconstructor.Config

/-- Validate both the encoder and decoder width before allocating either component. -/
def validate {d : Nat} (config : ViT.MaskedPatchReconstructor.Config d) :
    Except String Unit := do
  config.encoder.validate (kind := "ViT.MaskedPatchReconstructor")
  if config.reconstructionWidth = 0 then
    throw "ViT.MaskedPatchReconstructor: reconstruction width must be positive"

end ViT.MaskedPatchReconstructor.Config

/-- Reconstruction output shape for the same batch shape as the input. -/
abbrev ViT.MaskedPatchReconstructor.Config.output {d : Nat}
    (config : ViT.MaskedPatchReconstructor.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.reconstructionWidth

/--
Compact masked-patch image reconstructor.

Strided convolution embeds the patches, and the Transformer encoder processes every patch token.
The encoded tokens are flattened together and passed to one dense reconstruction projection.
Masked positions remain in the encoder sequence, so increasing the mask ratio does not reduce
its token count. A separate Transformer decoder and token removal/restoration are not part of
this compact architecture.

The masking objective is provided by `TorchLean.ssl.BlockMAE.sample`. Its axis policy is
independent of the model architecture and spatial rank, so this constructor uses the same checked
operation as signal, volume, and higher-dimensional masked-prediction models.
-/
def ViT.maskedPatchReconstructor {d : Nat} (config : ViT.MaskedPatchReconstructor.Config d)
    (batchShape : Shape := []) :
    nn.Builder
      (nn.Sequential
        (config.encoder.input batchShape)
        (config.output batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.encoder.input batchShape) (config.output batchShape)
        "ViT.MaskedPatchReconstructor" message
  | .ok () => do
      let encoder ← vitEncoder config.encoder batchShape
      let builtFlattening ← flattenAfter batchShape
        (shape := [config.encoder.sequenceLength,
          config.encoder.patchEmbedding.outChannels])
      let flattenTokens : Sequential
          (batchShape.concat [config.encoder.sequenceLength,
            config.encoder.patchEmbedding.outChannels])
          (batchShape.appendDim config.encoder.flattenedWidth) := by
        simpa [ViT.EncoderConfig.flattenedWidth, Shape.size] using builtFlattening
      let decoder ←
        linear config.encoder.flattenedWidth config.reconstructionWidth
          (batchShape := batchShape)
      pure <| encoder >>> flattenTokens >>> decoder

end models
end nn

end TorchLean
