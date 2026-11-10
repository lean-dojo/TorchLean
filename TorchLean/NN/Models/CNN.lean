/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder

/-!
# Convolutional Classifier

`cnn` builds a classifier over any `batchShape` and any number of spatial axes from a
`CNN.Config`. Each stage configures its own convolution, activation, optional dropout and pooling.
The linear head reads the flattened final feature map, and with no stages it reads the flattened
input.

Inside a stage the order is convolution, activation, the optional dropout, then max pooling, as
`convPoolBlock` builds it. The test `TorchLean.Test.PyTorchRoundtrip.run` checks a model of this
layout against PyTorch in float64. Its two stages of 3 × 3 convolution with padding 1, ReLU and
2 × 2 max pooling run on an 8 × 8 single-channel input, and a linear head follows. That test model
is put together from `nn.conv` and `nn.maxPool`, so it doesn't go through `cnn`.
-/

@[expose] public section

open scoped TorchLean.nn.Sequential

namespace TorchLean

namespace nn

/-- Configuration for a convolutional classifier over `d` spatial axes. -/
structure CNN.Config (d : Nat) where
  /-- Number of channels in each input sample. -/
  inputChannels : Nat
  /-- Extent of each spatial axis. -/
  spatial : Tensor Nat [d]
  /-- Ordered convolution/activation/pooling stages. -/
  stages : List (ConvPoolBlock.Config d) := []
  /-- Number of classifier logits per sample. -/
  classCount : Nat

namespace CNN.Config

/-- Validate the complete classifier geometry before allocating convolution or head parameters. -/
def validate {d : Nat} (config : CNN.Config d) : Except String Unit := do
  let rec
    /-- Validate each stage for the channel count and spatial extents it receives, then require the
    channel count and the spatial extents after the last stage to be positive. -/
    stages (channels : Nat) (spatial : Tensor Nat [d]) :
        List (ConvPoolBlock.Config d) → Except String Unit
      | [] => do
          if channels = 0 then
            throw "CNN: input channel count must be positive"
          if spatial.prod = 0 then
            throw "CNN: input spatial dimensions must be positive"
      | stage :: remaining => do
          stage.validate channels spatial (kind := "CNN")
          stages stage.block.convolution.outChannels
            (stage.pooling.poolOutputSpatial (stage.block.convolution.convOutputSpatial spatial))
            remaining
  stages config.inputChannels config.spatial config.stages
  if config.classCount = 0 then
    throw "CNN: class count must be positive"

end CNN.Config

/-- Input tensor shape after prepending an arbitrary batch shape. -/
abbrev CNN.Config.inputShape {d : Nat} (config : CNN.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ (Shape.dim config.inputChannels (config.spatial.to Shape))

/-- Classifier output shape with the same batch shape as the input. -/
abbrev CNN.Config.outputShape {d : Nat} (config : CNN.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.classCount]

/-- Build the configured feature stages, flatten their output, and apply the linear head. -/
def cnn {d : Nat} (config : CNN.Config d) (batchShape : Shape := []) :
    Builder (Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  let rec
    /-- The `stages` on `channels` channels of spatial extents `spatial`, followed by the
    flattening and the linear head. -/
    buildStages (channels : Nat) (spatial : Tensor Nat [d])
        (stages : List (ConvPoolBlock.Config d)) :
        Builder
          (Sequential (batchShape ++ (Shape.dim channels (spatial.to Shape)))
            (config.outputShape batchShape)) :=
      match stages with
      | [] =>
          let featureShape := Shape.dim channels (spatial.to Shape)
          nn.Sequential![
            nn.op (.reshape (Shape.size_append_eq_size_append_singleton batchShape featureShape)),
            linear featureShape.size config.classCount (batchShape := batchShape)
          ]
      | stage :: remaining => do
          let current ← convPoolBlock spatial stage (batchShape := batchShape)
            (inputChannels := channels)
          let rest ← buildStages stage.block.convolution.outChannels
            (stage.pooling.poolOutputSpatial (stage.block.convolution.convOutputSpatial spatial))
            remaining
          pure (current >>> rest)
  match config.validate with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.outputShape batchShape) "CNN" message
  | .ok () => buildStages config.inputChannels config.spatial config.stages

end nn
end TorchLean
