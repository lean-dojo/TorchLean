/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Layers.Conv
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Conv.Window
public import TorchLean.Tensor.Conversion

/-!
# Convolutional Network Specifications

A two-block convolutional network over an arbitrary number of spatial axes. Its spatial
parameters are vectors, so the same model definition applies to sequence, image, volume, and
higher-rank data. The forward and reverse-mode specifications use the generic convolution and
pooling operations.

The network is a standalone reference model. The executable `nn.cnn` accepts independently
configured stages. It is not constructed from this specification, and no equivalence theorem
connects the two here.

With the defaults on a 28 × 28 image, each 3 × 3 convolution with padding 1 keeps the size. Each
2 × 2 pool with stride 2 halves it, so the head reads 64 · 7 · 7 = 3136 features. We wrote
`TwoBlock.backward` by hand, layer by layer, from `Linear.backward`, `poolFeaturesBackward`, the
ReLU slope (`0` at `0`) and `Conv.backward`. The max pool sends each output gradient to the input
coordinate its forward pass selected. No theorem or torch test covers `TwoBlock`.
-/

@[expose] public section

open TorchLean.Shape (rank size)

namespace TorchLean

open Conv Pool Linear Shape
open TorchLean.Tensor

namespace CNN

/-- Max pooling over the spatial axes of a channels-first feature map, with unit dilation,
symmetric padding and floor mode: `Tensor.pool .max` cast to the shape of `outSpatial`. -/
def poolFeatures {α : Type} [Storage α] [Context α] {d C : Nat}
    {spatial : Tensor Nat [d]} (kernel stride padding : Tensor Nat [d])
    (x : Tensor α (C :: spatial.data.toList)) :
    Tensor α (C :: (outSpatial spatial kernel stride padding).data.toList) :=
  Tensor.castShape
    (Tensor.pool .max x (List.finRange (Shape.rank (C :: spatial.data.toList))).tail
      kernel.data.toList stride.data.toList padding.data.toList padding.data.toList
      (List.replicate d 1) false)
    (pool_tail_channelsFirst C spatial kernel stride padding)

/-- The gradient of `poolFeatures` for the cotangent `dy`: `Tensor.poolBackward .max`. -/
def poolFeaturesBackward {α : Type} [Storage α] [Context α] {d C : Nat}
    {spatial : Tensor Nat [d]} (kernel stride padding : Tensor Nat [d])
    (x : Tensor α (C :: spatial.data.toList))
    (dy : Tensor α
      (C :: (outSpatial spatial kernel stride padding).data.toList)) :
    Tensor α (C :: spatial.data.toList) :=
  Tensor.poolBackward .max x (List.finRange (Shape.rank (C :: spatial.data.toList))).tail
    kernel.data.toList stride.data.toList padding.data.toList padding.data.toList
    (List.replicate d 1) false
    (Tensor.castShape dy (pool_tail_channelsFirst C spatial kernel stride padding).symm)

/-- Spatial shape after one convolution followed by one pooling operation. -/
def blockOutSpatial {d : Nat} (spatial kernel convStride convPadding poolKernel poolStride
    poolPadding : Tensor Nat [d]) : Tensor Nat [d] :=
  outSpatial (outSpatialDilated spatial kernel convStride
      (Tensor.ones [d]) convPadding convPadding)
    poolKernel poolStride poolPadding

/-- Spatial shape after two convolution-pooling blocks. -/
def outputSpatial {d : Nat} (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂
    poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂ : Tensor Nat [d]) :
    Tensor Nat [d] :=
  blockOutSpatial
    (blockOutSpatial spatial kernel convStride₁ convPadding₁ poolKernel poolStride₁ poolPadding₁)
    kernel convStride₂ convPadding₂ poolKernel poolStride₂ poolPadding₂

/-- Feature-map shape after the second pooling operation. -/
def featureShape {d : Nat} (channels : Nat)
    (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : Tensor Nat [d]) : Shape :=
  (channels ::
    Tensor.to
      (outputSpatial spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel
        poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)
      Shape)

/-- Number of scalar features presented to the linear head. -/
def featureSize {d : Nat} (channels : Nat)
    (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : Tensor Nat [d]) : Nat :=
  Shape.size (featureShape channels spatial kernel convStride₁ convPadding₁ convStride₂
    convPadding₂ poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)


end CNN

namespace CNN.TwoBlock

variable {α : Type} [Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Hyperparameters for a two-block convolutional network of spatial rank `d`. -/
structure Config (d : Nat) where
  /-- Output channels of the first convolution. -/
  conv1Channels : Nat := 32
  /-- Output channels of the second convolution. -/
  conv2Channels : Nat := 64
  /-- Number of outputs of the linear head. -/
  outputSize : Nat := 10
  /-- Kernel size of both convolutions along each spatial axis. -/
  kernel : Tensor Nat [d] := Tensor.full [d] 3
  /-- Stride of the first convolution along each spatial axis. -/
  conv1Stride : Tensor Nat [d] := Tensor.full [d] 1
  /-- Padding on both sides of each spatial axis for the first convolution. -/
  conv1Padding : Tensor Nat [d] := Tensor.full [d] 1
  /-- Stride of the second convolution along each spatial axis. -/
  conv2Stride : Tensor Nat [d] := Tensor.full [d] 1
  /-- Padding on both sides of each spatial axis for the second convolution. -/
  conv2Padding : Tensor Nat [d] := Tensor.full [d] 1
  /-- Window size of both max pools along each spatial axis. -/
  poolKernel : Tensor Nat [d] := Tensor.full [d] 2
  /-- Stride of the first max pool along each spatial axis. -/
  poolStride1 : Tensor Nat [d] := Tensor.full [d] 2
  /-- Padding on both sides of each spatial axis for the first max pool. -/
  poolPadding1 : Tensor Nat [d] := Tensor.full [d] 0
  /-- Stride of the second max pool along each spatial axis. -/
  poolStride2 : Tensor Nat [d] := Tensor.full [d] 2
  /-- Padding on both sides of each spatial axis for the second max pool. -/
  poolPadding2 : Tensor Nat [d] := Tensor.full [d] 0

/-- A generic two-block convolutional network: each block applies a convolution, ReLU and max
pooling, and a linear head reads the flattened features. -/
structure _root_.TorchLean.CNN.TwoBlock {d : Nat} (config : Config d) (inChannels : Nat)
    (spatial : Tensor Nat [d])
    (α : Type) [Storage α] where
  /-- The first convolution, from `inChannels` to `config.conv1Channels` channels. -/
  conv1 : Conv d inChannels config.conv1Channels config.kernel config.conv1Stride
    config.conv1Padding α
  /-- The second convolution, from `config.conv1Channels` to `config.conv2Channels` channels. -/
  conv2 : Conv d config.conv1Channels config.conv2Channels config.kernel config.conv2Stride
    config.conv2Padding α
  /-- The linear head on the flattened output of the second block. -/
  head : Linear α
    (featureSize config.conv2Channels spatial config.kernel config.conv1Stride
      config.conv1Padding config.conv2Stride config.conv2Padding config.poolKernel
      config.poolStride1 config.poolPadding1 config.poolStride2 config.poolPadding2)
    config.outputSize

/-- Parameter gradients for `TwoBlock`. -/
structure Grads {d : Nat} (config : Config d) (inChannels : Nat)
    (spatial : Tensor Nat [d])
    (α : Type) [Storage α] where
  /-- Gradient of the first convolution's kernel. -/
  conv1Kernel : Tensor α
    (config.conv1Channels :: inChannels :: Tensor.to config.kernel Shape)
  /-- Gradient of the first convolution's bias. -/
  conv1Bias : Tensor α [config.conv1Channels]
  /-- Gradient of the second convolution's kernel. -/
  conv2Kernel : Tensor α
    (config.conv2Channels :: config.conv1Channels :: Tensor.to config.kernel Shape)
  /-- Gradient of the second convolution's bias. -/
  conv2Bias : Tensor α [config.conv2Channels]
  /-- Gradient of the head's weight matrix. -/
  headWeight : Tensor α [config.outputSize,
    featureSize config.conv2Channels spatial config.kernel config.conv1Stride
      config.conv1Padding config.conv2Stride config.conv2Padding config.poolKernel
      config.poolStride1 config.poolPadding1 config.poolStride2 config.poolPadding2]
  /-- Gradient of the head's bias. -/
  headBias : Tensor α [config.outputSize]

/-- Forward pass for `TwoBlock`. -/
def forward {d : Nat} {config : Config d} {inChannels : Nat}
    {spatial : Tensor Nat [d]}
    (m : TwoBlock config inChannels spatial α)
    (x : Tensor α (inChannels :: (Tensor.to spatial Shape))) :
    Tensor α [config.outputSize] :=
  let y₁ := Conv.forward m.conv1 x
  let r₁ := Tensor.relu y₁
  let p₁ := poolFeatures config.poolKernel config.poolStride1 config.poolPadding1 r₁
  let y₂ := Conv.forward m.conv2 p₁
  let r₂ := Tensor.relu y₂
  let p₂ := poolFeatures config.poolKernel config.poolStride2 config.poolPadding2 r₂
  Linear.forward m.head (Tensor.reshape p₂ [_] (Shape.size_vector _).symm)

/-- Reverse-mode parameter and input derivatives for `TwoBlock`. -/
def backward {d : Nat} {config : Config d} {inChannels : Nat}
    {spatial : Tensor Nat [d]}
    (m : TwoBlock config inChannels spatial α)
    (x : Tensor α (inChannels :: (Tensor.to spatial Shape)))
    (gradOutput : Tensor α [config.outputSize]) :
    Grads config inChannels spatial α ×
      Tensor α (inChannels :: (Tensor.to spatial Shape)) :=
  let convSpatial₁ := outSpatialDilated spatial config.kernel config.conv1Stride
      (Tensor.ones [d]) config.conv1Padding config.conv1Padding
  let pooledSpatial₁ :=
    outSpatial convSpatial₁ config.poolKernel config.poolStride1 config.poolPadding1
  let convSpatial₂ :=
    outSpatialDilated pooledSpatial₁ config.kernel config.conv2Stride
        (Tensor.ones [d]) config.conv2Padding config.conv2Padding
  let pooledSpatial₂ :=
    outSpatial convSpatial₂ config.poolKernel config.poolStride2 config.poolPadding2
  let y₁ := Conv.forward m.conv1 x
  let r₁ := Tensor.relu y₁
  let p₁ := poolFeatures config.poolKernel config.poolStride1 config.poolPadding1 r₁
  let y₂ := Conv.forward m.conv2 p₁
  let r₂ := Tensor.relu y₂
  let p₂ := poolFeatures config.poolKernel config.poolStride2 config.poolPadding2 r₂
  let flat := Tensor.reshape p₂ [_] (Shape.size_vector _).symm
  let headGradients := Linear.backward m.head flat gradOutput
  let featureShape : Shape := (config.conv2Channels :: (Tensor.to pooledSpatial₂ Shape))
  let dP₂ : Tensor α featureShape :=
    Tensor.reshape headGradients.inputGradient featureShape (Shape.size_vector _)
  let dR₂ := poolFeaturesBackward config.poolKernel config.poolStride2 config.poolPadding2 r₂
    dP₂
  let dY₂ := zipWith (· * ·) dR₂ (Tensor.map Activation.reluDeriv y₂)
  let conv2Gradients := Conv.backward m.conv2 p₁ dY₂
  let dR₁ := poolFeaturesBackward config.poolKernel config.poolStride1 config.poolPadding1 r₁
    conv2Gradients.inputGradient
  let dY₁ := zipWith (· * ·) dR₁ (Tensor.map Activation.reluDeriv y₁)
  let conv1Gradients := Conv.backward m.conv1 x dY₁
  ({ conv1Kernel := conv1Gradients.kernelGradient
     conv1Bias := conv1Gradients.biasGradient
     conv2Kernel := conv2Gradients.kernelGradient
     conv2Bias := conv2Gradients.biasGradient
     headWeight := headGradients.parameters.weights
     headBias := headGradients.parameters.bias }, conv1Gradients.inputGradient)

end CNN.TwoBlock

end TorchLean
