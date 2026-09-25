/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Residual Convolutional Classifier

The model accepts any `batchShape` and spatial rank. Each residual stage chooses its own channel
widths, convolution geometry, and shortcut. Global average pooling reduces every spatial axis
before the classifier head.
-/

@[expose] public section

namespace TorchLean

open Spec

namespace nn
namespace models

/-- The shortcut added to a residual stage's two-convolution branch. -/
inductive ResNet.Shortcut (d : Nat) where
  /-- Reuse the input unchanged; the branch must preserve its complete feature shape. -/
  | identity
  /-- Project the input with a separately configured convolution. -/
  | projection (convolution : Convolution.Config d)

/-- Two convolutions with a ReLU between them, an explicit shortcut, and a final ReLU. -/
structure ResNet.Stage.Config (d : Nat) where
  /-- First branch convolution; its stride may downsample the input grid. -/
  first : Convolution.Config d
  /-- Second branch convolution. -/
  second : Convolution.Config d
  /-- Shortcut whose output must match the branch's channels and spatial grid. -/
  shortcut : ResNet.Shortcut d := .identity

namespace ResNet.Stage.Config

/-- Grid after both branch convolutions. -/
def outputSpatial {d : Nat} (stage : ResNet.Stage.Config d)
    (spatial : Tensor Nat [d]) : Tensor Nat [d] :=
  stage.second.outputSpatial (stage.first.outputSpatial spatial)

/-- Feature shape after adding the branch and shortcut. -/
abbrev outputShape {d : Nat} (stage : ResNet.Stage.Config d)
    (spatial : Tensor Nat [d]) (batchShape : Shape := []) : Shape :=
  batchShape.concat (((stage.outputSpatial spatial).to Shape).prependDim stage.second.outChannels)

/-- Check both branch convolutions and the explicit shortcut before allocating parameters. -/
def validate {d : Nat} (stage : ResNet.Stage.Config d)
    (inputChannels : Nat) (spatial : Tensor Nat [d]) : Except String Unit := do
  stage.first.validate inputChannels spatial (kind := "ResNet")
  stage.second.validate stage.first.outChannels (stage.first.outputSpatial spatial)
    (kind := "ResNet")
  match stage.shortcut with
  | .identity =>
      if stage.outputShape spatial ≠ (spatial.to Shape).prependDim inputChannels then
        throw "ResNet: identity shortcut must preserve the branch input shape"
  | .projection convolution =>
      convolution.validate inputChannels spatial (kind := "ResNet")
      if ((convolution.outputSpatial spatial).to Shape).prependDim convolution.outChannels ≠
          stage.outputShape spatial then
        throw "ResNet: projection shortcut must match the branch output shape"

end ResNet.Stage.Config

/-- Configuration for a residual classifier over `d` spatial axes. -/
structure ResNet.Config (d : Nat) where
  /-- Number of channels in each input sample. -/
  inputChannels : Nat
  /-- Size of each input axis. Values such as `[32, 32]` work directly. -/
  spatial : Tensor Nat [d]
  /-- Convolutional stem, followed by a ReLU. -/
  stem : Convolution.Config d
  /-- Residual stages in execution order. An empty list pools the stem output directly. -/
  stages : List (ResNet.Stage.Config d) := []
  /-- Number of classifier logits per sample. -/
  classCount : Nat

namespace ResNet.Config

/-- Validate the complete residual classifier before allocating any branch parameters. -/
def validate {d : Nat} (config : ResNet.Config d) : Except String Unit := do
  let rec validateStages (channels : Nat) (spatial : Tensor Nat [d])
      (stages : List (ResNet.Stage.Config d)) : Except String Unit := do
    match stages with
    | [] => pure ()
    | stage :: rest =>
        stage.validate channels spatial
        validateStages stage.second.outChannels (stage.outputSpatial spatial) rest
  config.stem.validate config.inputChannels config.spatial (kind := "ResNet")
  validateStages config.stem.outChannels (config.stem.outputSpatial config.spatial) config.stages
  if config.classCount = 0 then
    throw "ResNet: class count must be positive"

/-- Input tensor shape with an arbitrary batch shape. -/
abbrev inputShape {d : Nat} (config : ResNet.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat ((config.spatial.to Shape).prependDim config.inputChannels)

/-- Classifier output shape with the same batch shape as the input. -/
abbrev outputShape {d : Nat} (config : ResNet.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.appendDim config.classCount

end ResNet.Config

namespace ResNet.Internal

/-- Build one stage, retaining the branch-first parameter and addition order. -/
def buildStage {d : Nat} (channels : Nat) (spatial : Tensor Nat [d])
    (stage : ResNet.Stage.Config d) (batchShape : Shape) :
    Builder (Sequential (batchShape.concat ((spatial.to Shape).prependDim channels))
      (stage.outputShape spatial batchShape)) := do
  let branch : Sequential (batchShape.concat ((spatial.to Shape).prependDim channels))
      (stage.outputShape spatial batchShape) ← nn.Sequential![
    conv spatial stage.first (batchShape := batchShape) (inputChannels := channels),
    relu,
    conv (stage.first.outputSpatial spatial) stage.second
      (batchShape := batchShape) (inputChannels := stage.first.outChannels)
  ]
  let combined : Sequential (batchShape.concat ((spatial.to Shape).prependDim channels))
      (stage.outputShape spatial batchShape) ← match stage.shortcut with
    | .identity =>
        if h : stage.outputShape spatial batchShape =
            batchShape.concat ((spatial.to Shape).prependDim channels) then
          let sameShape : Sequential
              (batchShape.concat ((spatial.to Shape).prependDim channels))
              (batchShape.concat ((spatial.to Shape).prependDim channels)) := by
            simpa only [h] using branch
          pure <| by simpa only [h] using residual sameShape
        else
          pure <| nn.Internal.invalidConfiguration _ _ "ResNet"
            "ResNet: identity shortcut must preserve the branch input shape"
    | .projection convolution =>
        if h : batchShape.concat
            (((convolution.outputSpatial spatial).to Shape).prependDim convolution.outChannels) =
            stage.outputShape spatial batchShape then
          let shortcut ← conv spatial convolution
            (batchShape := batchShape) (inputChannels := channels)
          let matchingShape : Sequential
              (batchShape.concat ((spatial.to Shape).prependDim channels))
              (stage.outputShape spatial batchShape) := by
            simpa only [h] using shortcut
          pure (addBranches branch matchingShape)
        else
          pure <| nn.Internal.invalidConfiguration _ _ "ResNet"
            "ResNet: projection shortcut must match the branch output shape"
  pure (combined >>> (← relu))

end ResNet.Internal

/-- Build the configured stem and residual stages, then global pooling and a linear classifier. -/
def resnet {d : Nat} (config : ResNet.Config d) (batchShape : Shape := []) :
    Builder (Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.inputShape batchShape) (config.outputShape batchShape) "ResNet" message
  | .ok () => do
      let stem ← nn.Sequential![
        conv config.spatial config.stem
          (batchShape := batchShape) (inputChannels := config.inputChannels),
        relu
      ]
      let rec buildStages (channels : Nat) (spatial : Tensor Nat [d])
          (stages : List (ResNet.Stage.Config d)) :
          Builder (Sequential (batchShape.concat ((spatial.to Shape).prependDim channels))
            (config.outputShape batchShape)) :=
        match stages with
        | [] => nn.Sequential![
            globalAvgPool spatial (batchShape := batchShape) (channels := channels),
            linear channels config.classCount (batchShape := batchShape)
          ]
        | stage :: rest => do
            let current ← ResNet.Internal.buildStage channels spatial stage batchShape
            let remaining ← buildStages stage.second.outChannels (stage.outputSpatial spatial) rest
            pure (current >>> remaining)
      let stages ← buildStages config.stem.outChannels
        (config.stem.outputSpatial config.spatial) config.stages
      pure (stem >>> stages)

end models
end nn
end TorchLean
