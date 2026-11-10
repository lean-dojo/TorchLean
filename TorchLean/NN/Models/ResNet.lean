/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder

/-!
# Residual Convolutional Classifier

`resnet` builds a residual classifier from a `ResNet.Config` for any `batchShape` and spatial rank.
Each residual stage chooses its own channel widths, convolution geometry and shortcut, and global
average pooling reduces every spatial axis before the classifier head.

A stage follows He et al., Section 3.2: the branch `conv, ReLU, conv` plus the identity shortcut of
eq. (1) or the projection of eq. (2), with the second ReLU after the addition as in their Fig. 2.
The head is Section 3.3's global average pooling and one linear layer, returning logits. We leave
out batch normalization, which Section 3.4 puts after every convolution. The stem is one convolution
and a ReLU with no max pool, and a projection may be any configured convolution where the paper uses
1 × 1 ones.

References:
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2016), "Deep Residual Learning for Image
  Recognition", CVPR 2016, pp. 770-778, doi:10.1109/CVPR.2016.90, arXiv:1512.03385.
-/

@[expose] public section

open scoped TorchLean.nn.Sequential

namespace TorchLean

namespace nn

/-- The shortcut added to a residual stage's two-convolution branch. -/
inductive ResNet.Shortcut (d : Nat) where
  /-- Reuse the input unchanged, which needs a branch that preserves the complete feature shape. -/
  | identity
  /-- Project the input with a separately configured convolution. -/
  | projection (convolution : Conv.Config d)

/-- Two convolutions with a ReLU between them, a configured shortcut and a final ReLU. -/
structure ResNetStage.Config (d : Nat) where
  /-- First branch convolution, whose stride may downsample the input grid. -/
  first : Conv.Config d
  /-- Second branch convolution. -/
  second : Conv.Config d
  /-- Shortcut whose output must match the branch's channels and spatial grid. -/
  shortcut : ResNet.Shortcut d := .identity

namespace ResNetStage.Config

/-- Grid after both branch convolutions. -/
def outputSpatial {d : Nat} (stage : ResNetStage.Config d)
    (spatial : Tensor Nat [d]) : Tensor Nat [d] :=
  stage.second.convOutputSpatial (stage.first.convOutputSpatial spatial)

/-- Feature shape after adding the branch and shortcut. -/
abbrev outputShape {d : Nat} (stage : ResNetStage.Config d)
    (spatial : Tensor Nat [d]) (batchShape : Shape := []) : Shape :=
  batchShape ++ (Shape.dim stage.second.outChannels ((stage.outputSpatial spatial).to Shape))

/-- Check both branch convolutions and the shortcut before allocating parameters. -/
def validate {d : Nat} (stage : ResNetStage.Config d)
    (inputChannels : Nat) (spatial : Tensor Nat [d]) : Except String Unit := do
  stage.first.validate inputChannels spatial (kind := "ResNet")
  stage.second.validate stage.first.outChannels (stage.first.convOutputSpatial spatial)
    (kind := "ResNet")
  match stage.shortcut with
  | .identity =>
      if stage.outputShape spatial ≠ Shape.dim inputChannels (spatial.to Shape) then
        throw "ResNet: identity shortcut must preserve the branch input shape"
  | .projection convolution =>
      convolution.validate inputChannels spatial (kind := "ResNet")
      if Shape.dim convolution.outChannels ((convolution.convOutputSpatial spatial).to Shape) ≠
          stage.outputShape spatial then
        throw "ResNet: projection shortcut must match the branch output shape"

end ResNetStage.Config

/-- Configuration for a residual classifier over `d` spatial axes. -/
structure ResNet.Config (d : Nat) where
  /-- Number of channels in each input sample. -/
  inputChannels : Nat
  /-- Size of each input axis, written as a literal such as `[32, 32]`. -/
  spatial : Tensor Nat [d]
  /-- Convolutional stem, followed by a ReLU. -/
  stem : Conv.Config d
  /-- Residual stages in execution order. With none, global pooling reads the stem output. -/
  stages : List (ResNetStage.Config d) := []
  /-- Number of classifier logits per sample. -/
  classCount : Nat

namespace ResNet.Config

/-- Validate the complete residual classifier before allocating any branch parameters. -/
def validate {d : Nat} (config : ResNet.Config d) : Except String Unit := do
  let rec
    /-- Validate each stage for the channel count and spatial extents it receives. -/
    stages (channels : Nat) (spatial : Tensor Nat [d])
        (stages : List (ResNetStage.Config d)) : Except String Unit := do
      match stages with
      | [] => pure ()
      | stage :: rest =>
          stage.validate channels spatial
          validate.stages stage.second.outChannels (stage.outputSpatial spatial) rest
  config.stem.validate config.inputChannels config.spatial (kind := "ResNet")
  validate.stages config.stem.outChannels (config.stem.convOutputSpatial config.spatial)
    config.stages
  if config.classCount = 0 then
    throw "ResNet: class count must be positive"

/-- Input tensor shape with an arbitrary batch shape. -/
abbrev inputShape {d : Nat} (config : ResNet.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ (Shape.dim config.inputChannels (config.spatial.to Shape))

/-- Classifier output shape with the same batch shape as the input. -/
abbrev outputShape {d : Nat} (config : ResNet.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.classCount]

end ResNet.Config

namespace ResNet

/-- Build one stage, retaining the branch-first parameter and addition order. -/
def buildStage {d : Nat} (channels : Nat) (spatial : Tensor Nat [d])
    (stage : ResNetStage.Config d) (batchShape : Shape) :
    Builder (Sequential (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (stage.outputShape spatial batchShape)) := do
  let branch : Sequential (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (stage.outputShape spatial batchShape) ← nn.Sequential![
    conv spatial stage.first (batchShape := batchShape) (inputChannels := channels),
    nn.op .relu,
    conv (stage.first.convOutputSpatial spatial) stage.second
      (batchShape := batchShape) (inputChannels := stage.first.outChannels)
  ]
  let combined : Sequential (batchShape ++ (Shape.dim channels (spatial.to Shape)))
      (stage.outputShape spatial batchShape) ← match stage.shortcut with
    | .identity =>
        if h : stage.outputShape spatial batchShape =
            batchShape ++ (Shape.dim channels (spatial.to Shape)) then
          let sameShape : Sequential
              (batchShape ++ (Shape.dim channels (spatial.to Shape)))
              (batchShape ++ (Shape.dim channels (spatial.to Shape))) := by
            simpa only [h] using branch
          pure <| by simpa only [h] using residual sameShape
        else
          pure <| nn.Sequential.invalid _ _ "ResNet"
            "ResNet: identity shortcut must preserve the branch input shape"
    | .projection convolution =>
        if h : batchShape ++
            (Shape.dim convolution.outChannels
              ((convolution.convOutputSpatial spatial).to Shape)) =
            stage.outputShape spatial batchShape then
          let shortcut ← conv spatial convolution
            (batchShape := batchShape) (inputChannels := channels)
          let matchingShape : Sequential
              (batchShape ++ (Shape.dim channels (spatial.to Shape)))
              (stage.outputShape spatial batchShape) := by
            simpa only [h] using shortcut
          pure (addBranches branch matchingShape)
        else
          pure <| nn.Sequential.invalid _ _ "ResNet"
            "ResNet: projection shortcut must match the branch output shape"
  let activation : Sequential (stage.outputShape spatial batchShape)
      (stage.outputShape spatial batchShape) ← nn.op .relu
  pure (combined >>> activation)

end ResNet

/-- Build the configured stem and residual stages, then global pooling and a linear classifier. -/
def resnet {d : Nat} (config : ResNet.Config d) (batchShape : Shape := []) :
    Builder (Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.outputShape batchShape) "ResNet" message
  | .ok () => do
      let stem ← nn.Sequential![
        conv config.spatial config.stem
          (batchShape := batchShape) (inputChannels := config.inputChannels),
        nn.op .relu
      ]
      let rec
        /-- The `stages` on `channels` channels of spatial extents `spatial`, followed by global
        average pooling and the linear classifier. -/
        buildStages (channels : Nat) (spatial : Tensor Nat [d])
            (stages : List (ResNetStage.Config d)) :
            Builder (Sequential (batchShape ++ (Shape.dim channels (spatial.to Shape)))
              (config.outputShape batchShape)) :=
          match stages with
          | [] => nn.Sequential![
              globalAvgPool spatial (batchShape := batchShape) (channels := channels),
              linear channels config.classCount (batchShape := batchShape)
            ]
          | stage :: rest => do
              let current ← ResNet.buildStage channels spatial stage batchShape
              let remaining ← buildStages stage.second.outChannels
                (stage.outputSpatial spatial) rest
              pure (current >>> remaining)
      let stages ← buildStages config.stem.outChannels
        (config.stem.convOutputSpatial config.spatial) config.stages
      pure (stem >>> stages)

end nn

end TorchLean
