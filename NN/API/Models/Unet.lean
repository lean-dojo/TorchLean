/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# U-Net

A typed encoder-decoder with one downsampling stage, one upsampling stage, and a channel-wise skip
connection. The spatial rank is a parameter: the same constructor applies to sequences, images,
volumes, and higher-dimensional grids.

The reusable operations remain visible in the type. Pooling determines the bottleneck extent,
transpose convolution returns it to the input extent, and branch concatenation changes the channel
count from `baseChannels` to `baseChannels + baseChannels`.

Reference: O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical
Image Segmentation," MICCAI 2015.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace models

/-- Configuration for a one-level U-Net over `d` spatial axes. -/
structure UNet.Config (d : Nat) where
  /-- Number of channels in each input sample. -/
  inputChannels : Nat
  /-- Width of the full-resolution feature map. -/
  baseChannels : Nat
  /-- Number of channels produced by the output projection. -/
  outputChannels : Nat
  /-- Size of each input axis. -/
  spatial : Tensor Nat [d]
  /-- Downsampling operation. -/
  pooling : Pooling.Config d
  /--
  Radius of the same-padding kernels used at both resolutions.

  A radius of `1` gives kernel size `3`; these convolutions preserve their grids automatically.
  -/
  kernelRadius : Tensor Nat [d] := Tensor.ones [d]
  /-- Geometry used by the transpose convolution that restores full resolution. -/
  upsampling : Convolution.Geometry d

namespace UNet.Config

/-- Grid size after the downsampling stage. -/
def pooledSpatial {d : Nat} (config : UNet.Config d) : Tensor Nat [d] :=
  config.pooling.outputSpatial config.spatial

/-- Validate every channel width and downsample/upsample geometry before construction. -/
def validate {d : Nat} (config : UNet.Config d) : Except String Unit := do
  if config.inputChannels = 0 then
    throw "UNet: input channel count must be positive"
  if config.baseChannels = 0 then
    throw "UNet: base channel count must be positive"
  if config.outputChannels = 0 then
    throw "UNet: output channel count must be positive"
  config.pooling.validate config.baseChannels config.spatial (kind := "UNet")
  let doubledChannels := config.baseChannels + config.baseChannels
  let upsampling := config.upsampling.transposedConvolution config.baseChannels
  upsampling.validate doubledChannels config.pooledSpatial (kind := "UNet")
  if upsampling.outputSpatial config.pooledSpatial != config.spatial then
    throw s!"UNet: transpose convolution outputs {upsampling.outputSpatial config.pooledSpatial} \
      after pooling, but the input spatial shape is {config.spatial}"

/-- Input shape with arbitrary batch axes. -/
abbrev inputShape {d : Nat} (config : UNet.Config d)
    (batchShape : Spec.Shape := []) : Spec.Shape :=
  batchShape.concat ((config.spatial.to Spec.Shape).prependDim config.inputChannels)

/-- Output shape with the same batch axes as the input. -/
abbrev outputShape {d : Nat} (config : UNet.Config d)
    (batchShape : Spec.Shape := []) : Spec.Shape :=
  batchShape.concat ((config.spatial.to Spec.Shape).prependDim config.outputChannels)

end UNet.Config

/--
Build a one-level U-Net over an arbitrary spatial rank.

Same-padding convolutions preserve both resolutions automatically. Model validation rejects an
empty input or pooled grid, or a pooling/upsampling pair that does not restore the input grid.
-/
def unet {d : Nat} (config : UNet.Config d) (batchShape : Spec.Shape := []) :
    Builder (Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
    pure <| nn.Internal.invalidConfiguration
        (config.inputShape batchShape) (config.outputShape batchShape) "UNet" message
  | .ok () =>
    if hRestores :
      config.upsampling.transposedOutputSpatial config.pooledSpatial = config.spatial then
      let doubledChannels := config.baseChannels + config.baseChannels
      let convolution := Convolution.Geometry.samePadding config.kernelRadius
      have preservesSize : convolution.outputSpatial config.spatial = config.spatial :=
        Convolution.Geometry.outputSpatial_samePadding config.spatial config.kernelRadius
      have preservesPooled :
          convolution.outputSpatial config.pooledSpatial = config.pooledSpatial :=
        Convolution.Geometry.outputSpatial_samePadding config.pooledSpatial config.kernelRadius
      do
        let builtFirstStemConvolution ←
          conv config.spatial (convolution.convolution config.baseChannels)
            (inputChannels := config.inputChannels)
        let firstStemConvolution : Sequential (config.inputShape [])
            ((config.spatial.to Spec.Shape).prependDim config.baseChannels) := by
          simpa [UNet.Config.inputShape, preservesSize] using builtFirstStemConvolution
        let builtSecondStemConvolution ←
          conv config.spatial (convolution.convolution config.baseChannels)
            (inputChannels := config.baseChannels)
        let secondStemConvolution : Sequential
            ((config.spatial.to Spec.Shape).prependDim config.baseChannels)
            ((config.spatial.to Spec.Shape).prependDim config.baseChannels) := by
          simpa [preservesSize] using builtSecondStemConvolution
        let fullResolutionActivation : Sequential
            ((config.spatial.to Spec.Shape).prependDim config.baseChannels)
            ((config.spatial.to Spec.Shape).prependDim config.baseChannels) ← relu
        let stem :=
          nn.compose![firstStemConvolution, fullResolutionActivation,
            secondStemConvolution, fullResolutionActivation]

        let downsample ←
          maxPool config.spatial config.pooling (channels := config.baseChannels)
        let builtFirstBottleneckConvolution ←
          conv config.pooledSpatial (convolution.convolution doubledChannels)
            (inputChannels := config.baseChannels)
        let firstBottleneckConvolution : Sequential
            ((config.pooledSpatial.to Spec.Shape).prependDim config.baseChannels)
            ((config.pooledSpatial.to Spec.Shape).prependDim doubledChannels) := by
          simpa [preservesPooled] using builtFirstBottleneckConvolution
        let builtSecondBottleneckConvolution ←
          conv config.pooledSpatial (convolution.convolution doubledChannels)
            (inputChannels := doubledChannels)
        let secondBottleneckConvolution : Sequential
            ((config.pooledSpatial.to Spec.Shape).prependDim doubledChannels)
            ((config.pooledSpatial.to Spec.Shape).prependDim doubledChannels) := by
          simpa [preservesPooled] using builtSecondBottleneckConvolution
        let bottleneckActivation : Sequential
            ((config.pooledSpatial.to Spec.Shape).prependDim doubledChannels)
            ((config.pooledSpatial.to Spec.Shape).prependDim doubledChannels) ← relu
        let bottleneck :=
          nn.compose![firstBottleneckConvolution, bottleneckActivation,
            secondBottleneckConvolution, bottleneckActivation]
        let builtUpsampling ←
          convTranspose config.pooledSpatial
            (config.upsampling.transposedConvolution config.baseChannels)
            (inputChannels := doubledChannels)
        let upsample : Sequential
            ((config.pooledSpatial.to Spec.Shape).prependDim doubledChannels)
            ((config.spatial.to Spec.Shape).prependDim config.baseChannels) := by
          simpa [hRestores] using builtUpsampling

        let deep := nn.compose![downsample, bottleneck, upsample]
        let skip :=
          Sequential.identity ((config.spatial.to Spec.Shape).prependDim config.baseChannels)
        let merge := concatBranches skip deep

        let builtFirstDecoderConvolution ←
          conv config.spatial (convolution.convolution config.baseChannels)
            (inputChannels := doubledChannels)
        let firstDecoderConvolution : Sequential
            ((config.spatial.to Spec.Shape).prependDim doubledChannels)
            ((config.spatial.to Spec.Shape).prependDim config.baseChannels) := by
          simpa [preservesSize] using builtFirstDecoderConvolution
        let builtSecondDecoderConvolution ←
          conv config.spatial (convolution.convolution config.baseChannels)
            (inputChannels := config.baseChannels)
        let secondDecoderConvolution : Sequential
            ((config.spatial.to Spec.Shape).prependDim config.baseChannels)
            ((config.spatial.to Spec.Shape).prependDim config.baseChannels) := by
          simpa [preservesSize] using builtSecondDecoderConvolution
        let outputProjection ←
          pointwiseConv config.spatial config.outputChannels
            (inputChannels := config.baseChannels)
        let decoder :=
          nn.compose![firstDecoderConvolution, fullResolutionActivation,
            secondDecoderConvolution, fullResolutionActivation, outputProjection]

        let core : Sequential (config.inputShape []) (config.outputShape []) :=
          nn.compose![stem, merge, decoder]
        pure (mapLeading batchShape core)
    else
      pure <| nn.Internal.invalidConfiguration
        (config.inputShape batchShape) (config.outputShape batchShape) "UNet"
        s!"UNet: transpose convolution outputs \
          {config.upsampling.transposedOutputSpatial config.pooledSpatial} after pooling, \
          but the input spatial shape is {config.spatial}"

end models
end nn
end TorchLean
