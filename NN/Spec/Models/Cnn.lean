/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Pooling
public import NN.Tensor.Conversion

/-!
# Convolutional Network Specifications

This module defines a two-block convolutional network over an arbitrary number of spatial axes.
Its spatial parameters are vectors, so the same model definition applies to sequence,
image, volume, and higher-rank data. The forward and reverse-mode specifications use the
generic convolution and pooling operations.

This is a standalone reference model. The executable `nn.models.cnn` accepts independently
configured stages, and `NN/Runtime/PyTorch/Export/CNN.lean` renders its own configuration.
Neither is constructed from this specification, and no equivalence theorem connects them here.
-/

@[expose] public section

namespace Models

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
  constructor <;> simp [defaultConfig]

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
