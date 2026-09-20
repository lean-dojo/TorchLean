/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Module.Activation
public import NN.Spec.Module.Conv
public import NN.Spec.Module.Flatten
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Pooling

/-!
# Convolutional Network Specifications

This module defines a two-block convolutional network over an arbitrary number of spatial axes.
Its spatial parameters are vectors, so the same model definition applies to sequence,
image, volume, and higher-rank data. Both the compositional module description and the explicit
reverse-mode specification use the generic convolution and pooling operations.

## Implementation status

`nn.models.cnn` (`NN/API/Models/Cnn.lean`) builds a one-block classifier
`convolution -> activation -> max pool -> flatten -> linear`, whereas this file specifies a
two-block network; no theorem relates them. This specification is imported by
`NN/Runtime/PyTorch/Export/CNN.lean`.
-/

@[expose] public section

namespace Models

open Spec.Module
open Spec TorchLean
open TorchLean TorchLean.Tensor
open Activation

namespace Cnn

/-- Spatial shape after one convolution followed by one pooling operation. -/
def blockOutSpatial {d : Nat} (spatial kernel convStride convPadding poolKernel poolStride
    poolPadding : TorchLean.Tensor Nat [d]) : TorchLean.Tensor Nat [d] :=
  poolOutSpatialPad (convOutSpatial spatial kernel convStride convPadding)
    poolKernel poolStride poolPadding

/-- Spatial shape after two convolution-pooling blocks. -/
def outputSpatial {d : Nat} (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂
    poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂ : TorchLean.Tensor Nat [d]) :
    TorchLean.Tensor Nat [d] :=
  blockOutSpatial
    (blockOutSpatial spatial kernel convStride₁ convPadding₁ poolKernel poolStride₁ poolPadding₁)
    kernel convStride₂ convPadding₂ poolKernel poolStride₂ poolPadding₂

/-- Feature-map shape after the second pooling operation. -/
def featureShape {d : Nat} (channels : Nat)
    (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : TorchLean.Tensor Nat [d]) : Shape :=
  Shape.ofList (channels ::
    Tensor.to
      (outputSpatial spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel
        poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)
      (List Nat))

/-- Number of scalar features presented to the linear head. -/
def featureSize {d : Nat} (channels : Nat)
    (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : TorchLean.Tensor Nat [d]) : Nat :=
  Shape.size (featureShape channels spatial kernel convStride₁ convPadding₁ convStride₂
    convPadding₂ poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)

/-- Two convolution-pooling blocks followed by a linear head. -/
def spec {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {d inChannels hiddenChannels outputSize : Nat}
    {spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : TorchLean.Tensor Nat [d]}
    {hPoolKernel : ∀ i : Fin d, poolKernel.getScalar i ≠ 0}
    {hPoolStride₁ : ∀ i : Fin d, poolStride₁.getScalar i ≠ 0}
    {hPoolStride₂ : ∀ i : Fin d, poolStride₂.getScalar i ≠ 0}
    (conv₁ : ConvSpec d inChannels hiddenChannels kernel convStride₁ convPadding₁ α)
    (conv₂ : ConvSpec d hiddenChannels hiddenChannels kernel convStride₂ convPadding₂ α)
    (pool₁ : MaxPoolSpec d poolKernel poolStride₁ poolPadding₁ hPoolKernel hPoolStride₁)
    (pool₂ : MaxPoolSpec d poolKernel poolStride₂ poolPadding₂ hPoolKernel hPoolStride₂)
    (head : LinearSpec α
      (featureSize hiddenChannels spatial kernel convStride₁ convPadding₁ convStride₂
        convPadding₂ poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)
      outputSize) :
    Spec.Module.Chain α (Shape.ofList (inChannels :: (Tensor.to spatial (List Nat))))
      (.dim outputSize .scalar) :=
  let convSpatial₁ := convOutSpatial spatial kernel convStride₁ convPadding₁
  let pooledSpatial₁ := poolOutSpatialPad convSpatial₁ poolKernel poolStride₁ poolPadding₁
  let convSpatial₂ := convOutSpatial pooledSpatial₁ kernel convStride₂ convPadding₂
  let pooledSpatial₂ := poolOutSpatialPad convSpatial₂ poolKernel poolStride₂ poolPadding₂
  let convModule₁ :
      Spec.Module α (Shape.ofList (inChannels :: (Tensor.to spatial (List Nat))))
        (Shape.ofList (hiddenChannels :: (Tensor.to convSpatial₁ (List Nat)))) :=
    Spec.Module.conv conv₁
  let poolModule₁ :
      Spec.Module α (Shape.ofList (hiddenChannels :: (Tensor.to convSpatial₁ (List Nat))))
        (Shape.ofList (hiddenChannels :: (Tensor.to pooledSpatial₁ (List Nat)))) :=
    Spec.Module.maxPool pool₁
  let convModule₂ :
      Spec.Module α (Shape.ofList (hiddenChannels :: (Tensor.to pooledSpatial₁ (List Nat))))
        (Shape.ofList (hiddenChannels :: (Tensor.to convSpatial₂ (List Nat)))) :=
    Spec.Module.conv conv₂
  let poolModule₂ :
      Spec.Module α (Shape.ofList (hiddenChannels :: (Tensor.to convSpatial₂ (List Nat))))
        (Shape.ofList (hiddenChannels :: (Tensor.to pooledSpatial₂ (List Nat)))) :=
    Spec.Module.maxPool pool₂
  let flattenModule :=
    Spec.Module.flatten α (Shape.ofList (hiddenChannels :: (Tensor.to pooledSpatial₂ (List Nat))))
  let headModule := Spec.Module.linear head
  Spec.Module.Chain.single convModule₁
    |>.append poolModule₁
    |>.append convModule₂
    |>.append poolModule₂
    |>.append flattenModule
    |>.append headModule

/-- The same network with ReLU after each convolution. -/
def withReluSpec {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {d inChannels hiddenChannels outputSize : Nat}
    {spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : TorchLean.Tensor Nat [d]}
    {hPoolKernel : ∀ i : Fin d, poolKernel.getScalar i ≠ 0}
    {hPoolStride₁ : ∀ i : Fin d, poolStride₁.getScalar i ≠ 0}
    {hPoolStride₂ : ∀ i : Fin d, poolStride₂.getScalar i ≠ 0}
    (conv₁ : ConvSpec d inChannels hiddenChannels kernel convStride₁ convPadding₁ α)
    (conv₂ : ConvSpec d hiddenChannels hiddenChannels kernel convStride₂ convPadding₂ α)
    (pool₁ : MaxPoolSpec d poolKernel poolStride₁ poolPadding₁ hPoolKernel hPoolStride₁)
    (pool₂ : MaxPoolSpec d poolKernel poolStride₂ poolPadding₂ hPoolKernel hPoolStride₂)
    (head : LinearSpec α
      (featureSize hiddenChannels spatial kernel convStride₁ convPadding₁ convStride₂
        convPadding₂ poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)
      outputSize) :
    Spec.Module.Chain α (Shape.ofList (inChannels :: (Tensor.to spatial (List Nat))))
      (.dim outputSize .scalar) :=
  let convSpatial₁ := convOutSpatial spatial kernel convStride₁ convPadding₁
  let pooledSpatial₁ := poolOutSpatialPad convSpatial₁ poolKernel poolStride₁ poolPadding₁
  let convSpatial₂ := convOutSpatial pooledSpatial₁ kernel convStride₂ convPadding₂
  let pooledSpatial₂ := poolOutSpatialPad convSpatial₂ poolKernel poolStride₂ poolPadding₂
  let convModule₁ :
      Spec.Module α (Shape.ofList (inChannels :: (Tensor.to spatial (List Nat))))
        (Shape.ofList (hiddenChannels :: (Tensor.to convSpatial₁ (List Nat)))) :=
    Spec.Module.conv conv₁
  let reluModule₁ := Spec.Module.relu (α := α)
    (Shape.ofList (hiddenChannels :: (Tensor.to convSpatial₁ (List Nat))))
  let poolModule₁ :
      Spec.Module α (Shape.ofList (hiddenChannels :: (Tensor.to convSpatial₁ (List Nat))))
        (Shape.ofList (hiddenChannels :: (Tensor.to pooledSpatial₁ (List Nat)))) :=
    Spec.Module.maxPool pool₁
  let convModule₂ :
      Spec.Module α (Shape.ofList (hiddenChannels :: (Tensor.to pooledSpatial₁ (List Nat))))
        (Shape.ofList (hiddenChannels :: (Tensor.to convSpatial₂ (List Nat)))) :=
    Spec.Module.conv conv₂
  let reluModule₂ := Spec.Module.relu (α := α)
    (Shape.ofList (hiddenChannels :: (Tensor.to convSpatial₂ (List Nat))))
  let poolModule₂ :
      Spec.Module α (Shape.ofList (hiddenChannels :: (Tensor.to convSpatial₂ (List Nat))))
        (Shape.ofList (hiddenChannels :: (Tensor.to pooledSpatial₂ (List Nat)))) :=
    Spec.Module.maxPool pool₂
  let flattenModule :=
    Spec.Module.flatten α (Shape.ofList (hiddenChannels :: (Tensor.to pooledSpatial₂ (List Nat))))
  let headModule := Spec.Module.linear head
  Spec.Module.Chain.single convModule₁
    |>.append reluModule₁
    |>.append poolModule₁
    |>.append convModule₂
    |>.append reluModule₂
    |>.append poolModule₂
    |>.append flattenModule
    |>.append headModule

/-- Evaluate a convolutional chain on one input tensor. -/
def forward {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {d inChannels hiddenChannels outputSize : Nat}
    {spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : TorchLean.Tensor Nat [d]}
    {hPoolKernel : ∀ i : Fin d, poolKernel.getScalar i ≠ 0}
    {hPoolStride₁ : ∀ i : Fin d, poolStride₁.getScalar i ≠ 0}
    {hPoolStride₂ : ∀ i : Fin d, poolStride₂.getScalar i ≠ 0}
    (conv₁ : ConvSpec d inChannels hiddenChannels kernel convStride₁ convPadding₁ α)
    (conv₂ : ConvSpec d hiddenChannels hiddenChannels kernel convStride₂ convPadding₂ α)
    (pool₁ : MaxPoolSpec d poolKernel poolStride₁ poolPadding₁ hPoolKernel hPoolStride₁)
    (pool₂ : MaxPoolSpec d poolKernel poolStride₂ poolPadding₂ hPoolKernel hPoolStride₂)
    (head : LinearSpec α
      (featureSize hiddenChannels spatial kernel convStride₁ convPadding₁ convStride₂
        convPadding₂ poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)
      outputSize)
    (x : Tensor α (Shape.ofList (inChannels :: (Tensor.to spatial (List Nat))))) :
    Tensor α [outputSize] :=
  (spec conv₁ conv₂ pool₁ pool₂ head).forward x

end Cnn

namespace TwoBlockCnn

variable {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Hyperparameters for a two-block convolutional network of spatial rank `d`. -/
structure Config (d : Nat) where
  conv1Channels : Nat := 32
  conv2Channels : Nat := 64
  outputSize : Nat := 10
  kernel : TorchLean.Tensor Nat [d] := Tensor.full [d] 3
  conv1Stride : TorchLean.Tensor Nat [d] := Tensor.full [d] 1
  conv1Padding : TorchLean.Tensor Nat [d] := Tensor.full [d] 1
  conv2Stride : TorchLean.Tensor Nat [d] := Tensor.full [d] 1
  conv2Padding : TorchLean.Tensor Nat [d] := Tensor.full [d] 1
  poolKernel : TorchLean.Tensor Nat [d] := Tensor.full [d] 2
  poolStride1 : TorchLean.Tensor Nat [d] := Tensor.full [d] 2
  poolPadding1 : TorchLean.Tensor Nat [d] := Tensor.full [d] 0
  poolStride2 : TorchLean.Tensor Nat [d] := Tensor.full [d] 2
  poolPadding2 : TorchLean.Tensor Nat [d] := Tensor.full [d] 0

/-- Conditions needed by convolutional and pooling implementations. -/
structure Config.WF {d : Nat} (config : Config d) : Prop where
  conv1Channels_ne_zero : config.conv1Channels ≠ 0
  conv2Channels_ne_zero : config.conv2Channels ≠ 0
  outputSize_ne_zero : config.outputSize ≠ 0
  kernel_ne_zero : ∀ i : Fin d, config.kernel.getScalar i ≠ 0
  conv1Stride_ne_zero : ∀ i : Fin d, config.conv1Stride.getScalar i ≠ 0
  conv2Stride_ne_zero : ∀ i : Fin d, config.conv2Stride.getScalar i ≠ 0
  poolKernel_ne_zero : ∀ i : Fin d, config.poolKernel.getScalar i ≠ 0
  poolStride1_ne_zero : ∀ i : Fin d, config.poolStride1.getScalar i ≠ 0
  poolStride2_ne_zero : ∀ i : Fin d, config.poolStride2.getScalar i ≠ 0

/-- The default configuration at any spatial rank. -/
def defaultConfig (d : Nat) : Config d := {}

/-- The default configuration is well formed. -/
theorem defaultConfig_wf (d : Nat) : (defaultConfig d).WF := by
  constructor
  · simp [defaultConfig]
  · simp [defaultConfig]
  · simp [defaultConfig]
  · intro i
    simp [defaultConfig]
  · intro i
    simp [defaultConfig]
  · intro i
    simp [defaultConfig]
  · intro i
    simp [defaultConfig]
  · intro i
    simp [defaultConfig]
  · intro i
    simp [defaultConfig]

/-- A generic two-block convolutional network with an explicit linear head. -/
structure Model {d : Nat} (config : Config d) (inChannels : Nat)
    (spatial : TorchLean.Tensor Nat [d])
    (α : Type) [TorchLean.Storage α] (hCfg : config.WF) where
  conv1 : ConvSpec d inChannels config.conv1Channels config.kernel config.conv1Stride
    config.conv1Padding α
  conv2 : ConvSpec d config.conv1Channels config.conv2Channels config.kernel config.conv2Stride
    config.conv2Padding α
  pool1 : MaxPoolSpec d config.poolKernel config.poolStride1 config.poolPadding1
    hCfg.poolKernel_ne_zero hCfg.poolStride1_ne_zero
  pool2 : MaxPoolSpec d config.poolKernel config.poolStride2 config.poolPadding2
    hCfg.poolKernel_ne_zero hCfg.poolStride2_ne_zero
  head : LinearSpec α
    (Cnn.featureSize config.conv2Channels spatial config.kernel config.conv1Stride
      config.conv1Padding config.conv2Stride config.conv2Padding config.poolKernel
      config.poolStride1 config.poolPadding1 config.poolStride2 config.poolPadding2)
    config.outputSize

/-- Parameter gradients for `Model`. -/
structure Grads {d : Nat} (config : Config d) (inChannels : Nat)
    (spatial : TorchLean.Tensor Nat [d])
    (α : Type) [TorchLean.Storage α] where
  conv1Kernel : Tensor α
    (Shape.ofList (config.conv1Channels :: inChannels :: Tensor.to config.kernel (List Nat)))
  conv1Bias : Tensor α [config.conv1Channels]
  conv2Kernel : Tensor α
    (Shape.ofList
      (config.conv2Channels :: config.conv1Channels :: Tensor.to config.kernel (List Nat)))
  conv2Bias : Tensor α [config.conv2Channels]
  headWeight : Tensor α [config.outputSize,
    Cnn.featureSize config.conv2Channels spatial config.kernel config.conv1Stride
      config.conv1Padding config.conv2Stride config.conv2Padding config.poolKernel
      config.poolStride1 config.poolPadding1 config.poolStride2 config.poolPadding2]
  headBias : Tensor α [config.outputSize]

/-- Forward pass for `Model`. -/
def Model.forward {d : Nat} {config : Config d} {inChannels : Nat}
    {spatial : TorchLean.Tensor Nat [d]}
    {hCfg : config.WF} (m : Model config inChannels spatial α hCfg)
    (x : Tensor α (Shape.ofList (inChannels :: (Tensor.to spatial (List Nat))))) :
    Tensor α [config.outputSize] :=
  let y₁ := convSpec m.conv1 x
  let r₁ := reluSpec y₁
  let p₁ := maxPoolSpec m.pool1 r₁
  let y₂ := convSpec m.conv2 p₁
  let r₂ := reluSpec y₂
  let p₂ := maxPoolSpec m.pool2 r₂
  linearSpec m.head (Tensor.flattenSpec p₂)

/-- Reverse-mode parameter and input derivatives for `Model`. -/
def Model.backward {d : Nat} {config : Config d} {inChannels : Nat}
    {spatial : TorchLean.Tensor Nat [d]}
    {hCfg : config.WF} (m : Model config inChannels spatial α hCfg)
    (x : Tensor α (Shape.ofList (inChannels :: (Tensor.to spatial (List Nat)))))
    (gradOutput : Tensor α [config.outputSize]) :
    Grads config inChannels spatial α ×
      Tensor α (Shape.ofList (inChannels :: (Tensor.to spatial (List Nat)))) :=
  let convSpatial₁ := convOutSpatial spatial config.kernel config.conv1Stride config.conv1Padding
  let pooledSpatial₁ :=
    poolOutSpatialPad convSpatial₁ config.poolKernel config.poolStride1 config.poolPadding1
  let convSpatial₂ :=
    convOutSpatial pooledSpatial₁ config.kernel config.conv2Stride config.conv2Padding
  let pooledSpatial₂ :=
    poolOutSpatialPad convSpatial₂ config.poolKernel config.poolStride2 config.poolPadding2
  let y₁ := convSpec m.conv1 x
  let r₁ := reluSpec y₁
  let p₁ := maxPoolSpec m.pool1 r₁
  let y₂ := convSpec m.conv2 p₁
  let r₂ := reluSpec y₂
  let p₂ := maxPoolSpec m.pool2 r₂
  let flat := Tensor.flattenSpec p₂
  let headGradients := linearBackwardSpec m.head flat gradOutput
  let featureShape := Shape.ofList (config.conv2Channels :: (Tensor.to pooledSpatial₂ (List Nat)))
  let dP₂ : Tensor α featureShape :=
    Tensor.unflattenSpec featureShape headGradients.inputGradient
  let dR₂ := maxPoolBackwardSpec m.pool2 r₂ dP₂
  let dY₂ := mulSpec dR₂ (reluDerivSpec y₂)
  let conv2Gradients := convBackwardSpec m.conv2 p₁ dY₂
  let dR₁ := maxPoolBackwardSpec m.pool1 r₁ conv2Gradients.inputGradient
  let dY₁ := mulSpec dR₁ (reluDerivSpec y₁)
  let conv1Gradients := convBackwardSpec m.conv1 x dY₁
  ({ conv1Kernel := conv1Gradients.kernelGradient
     conv1Bias := conv1Gradients.biasGradient
     conv2Kernel := conv2Gradients.kernelGradient
     conv2Bias := conv2Gradients.biasGradient
     headWeight := headGradients.weightGradient
     headBias := headGradients.biasGradient }, conv1Gradients.inputGradient)

end TwoBlockCnn

end Models
