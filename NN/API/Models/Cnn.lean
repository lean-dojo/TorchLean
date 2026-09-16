/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Convolutional Classifier

The classifier accepts any `batchShape` and number of spatial axes. Convolution and pooling use the
same vector-valued configuration for signals, images, volumes, and higher-dimensional data.
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
  /-- Convolution applied before activation and pooling. -/
  convolution : Convolution.Config d
  /-- Pooling applied after the convolutional activation. -/
  pooling : Pooling.Config d
  /-- Number of classifier logits per sample. -/
  classCount : Nat

namespace CNN.Config

/-- Validate the complete classifier geometry before allocating convolution or head parameters. -/
def validate {d : Nat} (config : CNN.Config d) : Except String Unit := do
  config.convolution.validate config.inputChannels config.spatial (kind := "CNN")
  let afterConv := config.convolution.output config.spatial
  config.pooling.validate config.convolution.outChannels afterConv (kind := "CNN")
  if config.classCount = 0 then
    throw "CNN: class count must be positive"

end CNN.Config

/-- Input tensor shape after prepending an arbitrary batch shape. -/
abbrev CNN.Config.input {d : Nat} (config : CNN.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat ((config.spatial.to Shape).prependDim config.inputChannels)

/-- Classifier output shape with the same batch shape as the input. -/
abbrev CNN.Config.output {d : Nat} (config : CNN.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.classCount

/-- Build `convolution -> activation -> max pool -> flatten -> linear`. -/
def cnn {d : Nat} (config : CNN.Config d) (batchShape : Shape := []) :
    Builder (Sequential (config.input batchShape) (config.output batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.input batchShape) (config.output batchShape) "CNN" message
  | .ok () =>
      let afterConv := config.convolution.output config.spatial
      let afterPool := config.pooling.output afterConv
      let featureShape := (afterPool.to Shape).prependDim config.convolution.outChannels
      let featureCount := featureShape.size
      let convolution := conv config.spatial config.convolution (batchShape := batchShape)
        (inputChannels := config.inputChannels)
      let pooling := maxPool afterConv config.pooling (batchShape := batchShape)
      nn.Sequential![
        convolution,
        relu,
        pooling,
        flattenAfter batchShape (shape := featureShape),
        linear featureCount config.classCount (batchShape := batchShape)
      ]

end models
end nn
end TorchLean
