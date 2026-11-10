/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder

/-!
# U-Net

`unet` builds a typed encoder-decoder from a `UNet.Config`, with one downsampling stage, one
upsampling stage and a channel-wise skip connection. The spatial rank is a parameter, so the same
constructor applies to sequences, images, volumes and higher-dimensional grids.

The reusable operations remain visible in the type. Pooling determines the bottleneck extent,
transpose convolution returns it to the input extent, and branch concatenation changes the channel
count from `baseChannels` to `baseChannels + baseChannels`.

We followed Section 2 of the paper for one level of its ladder: two convolutions with ReLU, a max
pool and two convolutions at twice the channels. An up-convolution returns to `baseChannels`, the
result is concatenated with the skip, and two more convolutions and a final 1 × 1 convolution
follow. The paper stacks more levels (23 convolutions in all), and its 3 × 3 convolutions are
unpadded, so the skip is cropped before the concatenation. Ours are same-padded
(`Window.samePadding`), which keeps the grid and leaves nothing to crop. The skip comes first in
the concatenated channels, and the pool and up-convolution windows are configurable where the
paper fixes 2 × 2.

Reference: O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical
Image Segmentation," MICCAI 2015, LNCS, pp. 234-241, doi:10.1007/978-3-319-24574-4_28,
arXiv:1505.04597, Section 2.
-/

@[expose] public section

namespace TorchLean
namespace nn

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
  pooling : Window d
  /--
  Radius of the same-padding kernels used at both resolutions.

  A radius of `1` gives kernel size `3`, and each of these convolutions preserves its grid.
  -/
  kernelRadius : Tensor Nat [d] := Tensor.ones [d]
  /-- Window of the transpose convolution that restores full resolution. -/
  upsampling : Window d

namespace UNet.Config

/-- Grid size after the downsampling stage. -/
def pooledSpatial {d : Nat} (config : UNet.Config d) : Tensor Nat [d] :=
  config.pooling.poolOutputSpatial config.spatial

/-- Validate every channel width and downsample/upsample geometry before construction. -/
def validate {d : Nat} (config : UNet.Config d) : Except String Unit := do
  if config.inputChannels = 0 then
    throw "UNet: input channel count must be positive"
  if config.baseChannels = 0 then
    throw "UNet: base channel count must be positive"
  if config.outputChannels = 0 then
    throw "UNet: output channel count must be positive"
  config.pooling.validatePool config.baseChannels config.spatial (kind := "UNet")
  let doubledChannels := config.baseChannels + config.baseChannels
  let upsampling := config.upsampling.convolution config.baseChannels
  -- The checks `nn.convTranspose` runs, on the transpose convolution's own output grid.
  nn.validateConvolution doubledChannels upsampling.outChannels config.pooledSpatial
    upsampling.kernelSize upsampling.stride
    (upsampling.transposedOutputSpatial config.pooledSpatial)
    upsampling.weightInit "UNet"
  if upsampling.transposedOutputSpatial config.pooledSpatial != config.spatial then
    throw s!"UNet: transpose convolution outputs \
      {upsampling.transposedOutputSpatial config.pooledSpatial} after pooling, but the input \
      spatial shape is {config.spatial}"

/-- Input shape with arbitrary batch axes. -/
abbrev inputShape {d : Nat} (config : UNet.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ (Shape.dim config.inputChannels (config.spatial.to Shape))

/-- Output shape with the same batch axes as the input. -/
abbrev outputShape {d : Nat} (config : UNet.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ (Shape.dim config.outputChannels (config.spatial.to Shape))

end UNet.Config

/--
Build a one-level U-Net over an arbitrary spatial rank.

Same-padding convolutions preserve both resolutions automatically. Model validation rejects an
empty input or pooled grid, or a pooling/upsampling pair that does not restore the input grid.
-/
def unet {d : Nat} (config : UNet.Config d) (batchShape : Shape := []) :
    Builder (Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
    pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.outputShape batchShape) "UNet" message
  | .ok () =>
    if hRestores :
      config.upsampling.transposedOutputSpatial config.pooledSpatial = config.spatial then
      let doubledChannels := config.baseChannels + config.baseChannels
      let convolution := Window.samePadding config.kernelRadius
      have preservesSize : convolution.convOutputSpatial config.spatial = config.spatial :=
        Window.convOutputSpatial_samePadding config.spatial config.kernelRadius
      have preservesPooled :
          convolution.convOutputSpatial config.pooledSpatial = config.pooledSpatial :=
        Window.convOutputSpatial_samePadding config.pooledSpatial config.kernelRadius
      do
        let builtFirstStemConvolution ←
          conv config.spatial (convolution.convolution config.baseChannels)
            (inputChannels := config.inputChannels)
        let firstStemConvolution : Sequential (config.inputShape [])
            (Shape.dim config.baseChannels (config.spatial.to Shape)) := by
          simpa [UNet.Config.inputShape, preservesSize] using builtFirstStemConvolution
        let builtSecondStemConvolution ←
          conv config.spatial (convolution.convolution config.baseChannels)
            (inputChannels := config.baseChannels)
        let secondStemConvolution : Sequential
            (Shape.dim config.baseChannels (config.spatial.to Shape))
            (Shape.dim config.baseChannels (config.spatial.to Shape)) := by
          simpa [preservesSize] using builtSecondStemConvolution
        let fullResolutionActivation : Sequential
            (Shape.dim config.baseChannels (config.spatial.to Shape))
            (Shape.dim config.baseChannels (config.spatial.to Shape)) ← nn.op .relu
        let stem :=
          nn.compose![firstStemConvolution, fullResolutionActivation,
            secondStemConvolution, fullResolutionActivation]

        let downsample ←
          maxPool config.spatial config.pooling (channels := config.baseChannels)
        let builtFirstBottleneckConvolution ←
          conv config.pooledSpatial (convolution.convolution doubledChannels)
            (inputChannels := config.baseChannels)
        let firstBottleneckConvolution : Sequential
            (Shape.dim config.baseChannels (config.pooledSpatial.to Shape))
            (Shape.dim doubledChannels (config.pooledSpatial.to Shape)) := by
          simpa [preservesPooled] using builtFirstBottleneckConvolution
        let builtSecondBottleneckConvolution ←
          conv config.pooledSpatial (convolution.convolution doubledChannels)
            (inputChannels := doubledChannels)
        let secondBottleneckConvolution : Sequential
            (Shape.dim doubledChannels (config.pooledSpatial.to Shape))
            (Shape.dim doubledChannels (config.pooledSpatial.to Shape)) := by
          simpa [preservesPooled] using builtSecondBottleneckConvolution
        let bottleneckActivation : Sequential
            (Shape.dim doubledChannels (config.pooledSpatial.to Shape))
            (Shape.dim doubledChannels (config.pooledSpatial.to Shape)) ← nn.op .relu
        let bottleneck :=
          nn.compose![firstBottleneckConvolution, bottleneckActivation,
            secondBottleneckConvolution, bottleneckActivation]
        let builtUpsampling ←
          convTranspose config.pooledSpatial
            (config.upsampling.convolution config.baseChannels)
            (inputChannels := doubledChannels)
        let upsample : Sequential
            (Shape.dim doubledChannels (config.pooledSpatial.to Shape))
            (Shape.dim config.baseChannels (config.spatial.to Shape)) := by
          simpa [hRestores] using builtUpsampling

        let deep := nn.compose![downsample, bottleneck, upsample]
        let skip :=
          Sequential.id (Shape.dim config.baseChannels (config.spatial.to Shape))
        let merge := concatBranches skip deep

        let builtFirstDecoderConvolution ←
          conv config.spatial (convolution.convolution config.baseChannels)
            (inputChannels := doubledChannels)
        let firstDecoderConvolution : Sequential
            (Shape.dim doubledChannels (config.spatial.to Shape))
            (Shape.dim config.baseChannels (config.spatial.to Shape)) := by
          simpa [preservesSize] using builtFirstDecoderConvolution
        let builtSecondDecoderConvolution ←
          conv config.spatial (convolution.convolution config.baseChannels)
            (inputChannels := config.baseChannels)
        let secondDecoderConvolution : Sequential
            (Shape.dim config.baseChannels (config.spatial.to Shape))
            (Shape.dim config.baseChannels (config.spatial.to Shape)) := by
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
      pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.outputShape batchShape) "UNet"
        s!"UNet: transpose convolution outputs \
          {config.upsampling.transposedOutputSpatial config.pooledSpatial} after pooling, \
          but the input spatial shape is {config.spatial}"

end nn
end TorchLean
