/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.API.Macros -- shake: keep

/-!
# Residual Convolutional Classifier

The model accepts any `batchShape` and spatial rank. Residual branches operate on a common typed
shape, and global average pooling reduces every spatial axis before the classifier head.
-/

@[expose] public section

namespace TorchLean

open Spec

namespace nn
namespace models

/-- Configuration for a residual classifier over `d` spatial axes. -/
structure ResNet.Config (d : Nat) where
  /-- Number of channels in each input sample. -/
  inputChannels : Nat
  /-- Size of each input axis. Values such as `[32, 32]` work directly. -/
  spatial : Tensor Nat [d]
  /-- Channel width used by the residual trunk. -/
  hiddenChannels : Nat
  /--
  Radius of the same-padding convolution kernel on each axis.

  A radius of `1` gives the familiar kernel size `3`; all convolutions therefore preserve the
  input grid without an additional shape proof.
  -/
  kernelRadius : Tensor Nat [d] := Tensor.ones [d]
  /-- Number of classifier logits per sample. -/
  classCount : Nat

namespace ResNet.Config

/-- Validate the complete residual classifier before allocating any branch parameters. -/
def validate {d : Nat} (config : ResNet.Config d) : Except String Unit := do
  if config.inputChannels = 0 then
    throw "ResNet: input channel count must be positive"
  if config.spatial.prod = 0 then
    throw "ResNet: input spatial dimensions must be positive"
  if config.hiddenChannels = 0 then
    throw "ResNet: hidden channel count must be positive"
  if config.classCount = 0 then
    throw "ResNet: class count must be positive"

/-- Input tensor shape with an arbitrary batch shape. -/
abbrev input {d : Nat} (config : ResNet.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat ((config.spatial.to Shape).prependDim config.inputChannels)

/-- Hidden activation shape shared by the residual branches. -/
abbrev hidden {d : Nat} (config : ResNet.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat ((config.spatial.to Shape).prependDim config.hiddenChannels)

/-- Classifier output shape with the same batch shape as the input. -/
abbrev output {d : Nat} (config : ResNet.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.classCount

end ResNet.Config

/-- Build a convolutional stem, two residual blocks, global pooling, and a linear classifier. -/
def resnet {d : Nat} (config : ResNet.Config d) (batchShape : Shape := []) :
    Builder (Sequential (config.input batchShape) (config.output batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.input batchShape) (config.output batchShape) "ResNet" message
  | .ok () =>
      let geometry := Convolution.Geometry.samePadding config.kernelRadius
      have preservesSize : geometry.output config.spatial = config.spatial :=
        Convolution.Geometry.output_samePadding config.spatial config.kernelRadius
      let builtStem := conv config.spatial (geometry.convolution config.hiddenChannels)
        (batchShape := batchShape) (inputChannels := config.inputChannels)
      let stem :
          Builder (Sequential (config.input batchShape) (config.hidden batchShape)) := by
        simpa [ResNet.Config.input, ResNet.Config.hidden,
          preservesSize] using builtStem
      let builtHiddenConvolution :=
        conv config.spatial (geometry.convolution config.hiddenChannels)
          (batchShape := batchShape) (inputChannels := config.hiddenChannels)
      let hiddenConvolution :
          Builder (Sequential (config.hidden batchShape) (config.hidden batchShape)) := by
        simpa [ResNet.Config.hidden, preservesSize] using
          builtHiddenConvolution
      let residualBranch := do
        let branch ← nn.Sequential![hiddenConvolution, relu, hiddenConvolution]
        return residual branch
      let pooling := globalAvgPool config.spatial
        (batchShape := batchShape) (channels := config.hiddenChannels)
      nn.Sequential![
        stem,
        relu,
        residualBranch,
        relu,
        residualBranch,
        relu,
        pooling,
        linear config.hiddenChannels config.classCount (batchShape := batchShape)
      ]

end models
end nn
end TorchLean
