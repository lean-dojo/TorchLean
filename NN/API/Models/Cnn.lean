/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Convolutional Classifier

The classifier accepts any `batchShape` and number of spatial axes. Each stage independently
configures its convolution, activation, optional dropout, and pooling. The linear head consumes
the final feature map; an empty stage list applies the head directly to the flattened input.
-/

@[expose] public section

namespace TorchLean

open Spec

namespace nn
namespace models

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
  let rec validateStages (channels : Nat) (spatial : Tensor Nat [d]) :
      List (ConvPoolBlock.Config d) → Except String Unit
    | [] => do
        if channels = 0 then
          throw "CNN: input channel count must be positive"
        if spatial.prod = 0 then
          throw "CNN: input spatial dimensions must be positive"
    | stage :: remaining => do
        stage.validate channels spatial (kind := "CNN")
        validateStages stage.block.convolution.outChannels
          (stage.pooling.outputSpatial (stage.block.convolution.outputSpatial spatial)) remaining
  validateStages config.inputChannels config.spatial config.stages
  if config.classCount = 0 then
    throw "CNN: class count must be positive"

end CNN.Config

/-- Input tensor shape after prepending an arbitrary batch shape. -/
abbrev CNN.Config.inputShape {d : Nat} (config : CNN.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat ((config.spatial.to Shape).prependDim config.inputChannels)

/-- Classifier output shape with the same batch shape as the input. -/
abbrev CNN.Config.outputShape {d : Nat} (config : CNN.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.classCount

/-- Build the configured feature stages, flatten their output, and apply the linear head. -/
def cnn {d : Nat} (config : CNN.Config d) (batchShape : Shape := []) :
    Builder (Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  let rec buildStages (channels : Nat) (spatial : Tensor Nat [d])
      (stages : List (ConvPoolBlock.Config d)) :
      Builder
        (Sequential (batchShape.concat ((spatial.to Shape).prependDim channels))
          (config.outputShape batchShape)) :=
    match stages with
    | [] =>
        let featureShape := (spatial.to Shape).prependDim channels
        nn.Sequential![
          flattenAfter batchShape (shape := featureShape),
          linear featureShape.size config.classCount (batchShape := batchShape)
        ]
    | stage :: remaining => do
        let current ← convPoolBlock spatial stage (batchShape := batchShape)
          (inputChannels := channels)
        let rest ← buildStages stage.block.convolution.outChannels
          (stage.pooling.outputSpatial (stage.block.convolution.outputSpatial spatial)) remaining
        pure (current >>> rest)
  match config.validate with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.inputShape batchShape) (config.outputShape batchShape) "CNN" message
  | .ok () => buildStages config.inputChannels config.spatial config.stages

end models
end nn
end TorchLean
