/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.ViT

/-!
# Self-Supervised Model Constructors

Most SSL machinery lives in `TorchLean.SelfSupervised`, whose masks, tensor-to-training-sample
transforms and objective-facing helpers work with any compatible model. Architecture-level
conveniences sit here. `nn.maskedPatchReconstructor` encodes every patch and reconstructs through a
dense head, and the SSL objective can train other architectures as well.

The contract comes from MAE (He et al., 2022, Section 3), and we kept the model small. MAE's
encoder sees only the visible patches, a quarter of them at its 75% mask ratio, and a stack of
Transformer blocks decodes. Ours encodes all of them and decodes with one linear layer.

Reference: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick
(2022), "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022, pp. 15979-15988,
doi:10.1109/CVPR52688.2022.01553, arXiv:2111.06377, Section 3.
-/

@[expose] public section

open scoped TorchLean.nn.Sequential

namespace TorchLean


open TorchLean.Tensor

namespace nn

/-! ## Masked patch reconstruction -/

/--
Configuration for a compact masked patch-transformer reconstructor.

The input/output contract is MAE-style: a masked channel/spatial tensor is mapped to one flattened
reconstruction vector per batch position.

`reconstructionWidth` can be the full image size (`C*H*W`) or a prefix for faster experiments.
-/
structure MaskedPatchReconstructor.Config (d : Nat) where
  /-- Patch-transformer encoder configuration. -/
  encoder : ViT.EncoderConfig d
  /-- Number of reconstructed output coordinates. -/
  outputWidth : Nat

namespace MaskedPatchReconstructor.Config

/-- Validate both the encoder and decoder width before allocating either component. -/
def validate {d : Nat} (config : MaskedPatchReconstructor.Config d) :
    Except String Unit := do
  config.encoder.validate (kind := "ViT.MaskedPatchReconstructor")
  if config.outputWidth = 0 then
    throw "ViT.MaskedPatchReconstructor: reconstruction width must be positive"

end MaskedPatchReconstructor.Config

/-- Reconstruction output shape for the same batch shape as the input. -/
abbrev MaskedPatchReconstructor.Config.outputShape {d : Nat}
    (config : MaskedPatchReconstructor.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.outputWidth]

/--
Compact masked-patch image reconstructor.

Strided convolution embeds the patches, and the Transformer encoder processes every patch token.
The encoded tokens are flattened together and passed to one dense reconstruction projection.
Masked positions remain in the encoder sequence, so increasing the mask ratio does not reduce
its token count. A separate Transformer decoder and token removal/restoration are not part of
this compact architecture.

The masking objective is provided by `TorchLean.SelfSupervised.BlockMAE.sample`. Its axis policy is
independent of the model architecture and spatial rank, so this constructor uses the same checked
operation as signal, volume, and higher-dimensional masked-prediction models.
-/
def maskedPatchReconstructor {d : Nat} (config : MaskedPatchReconstructor.Config d)
    (batchShape : Shape := []) :
    nn.Builder
      (nn.Sequential
        (config.encoder.inputShape batchShape)
        (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.encoder.inputShape batchShape) (config.outputShape batchShape)
        "ViT.MaskedPatchReconstructor" message
  | .ok () => do
      let encoder ← vitEncoder config.encoder batchShape
      let flattenTokens : Sequential
          (batchShape ++ [config.encoder.sequenceLength,
            config.encoder.patchEmbedding.outChannels])
          (batchShape ++ [config.encoder.flattenedWidth]) :=
        Sequential.fromLayer <| Layer.op <| .reshape <| by
          rw [Shape.size_append_eq_size_append_singleton]
          simp [ViT.EncoderConfig.flattenedWidth]
      let decoder ←
        linear config.encoder.flattenedWidth config.outputWidth
          (batchShape := batchShape)
      pure <| encoder >>> flattenTokens >>> decoder

end nn

end TorchLean
