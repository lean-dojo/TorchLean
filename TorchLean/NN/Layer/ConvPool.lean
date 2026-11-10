/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Layer

/-!
# TorchLean NN: Convolution and Pooling Layers

Convolution, transpose convolution, max pooling and average pooling layers for channels-first
inputs with any number of spatial axes. The configuration check of each layer refuses a zero
channel count, a zero input, kernel or stride extent, and a geometry with an empty output grid.
-/

@[expose] public section

namespace TorchLean
namespace nn.Layer

open TorchLean.Tensor

/-- Reject zero spatial, kernel, or stride extents, identifying the layer in the error message. -/
def validateSpatial {rank : Nat} (kind : String)
    (inputSize kernelSize stride : Tensor Nat [rank]) : Except String Unit := do
  if inputSize.prod = 0 then
    throw s!"{kind}: input spatial dimensions must be positive"
  if !decide (∀ axis : Fin rank, kernelSize.getScalar axis ≠ 0) then
    throw s!"{kind}: kernel size entries must be positive"
  if !decide (∀ axis : Fin rank, stride.getScalar axis ≠ 0) then
    throw s!"{kind}: stride entries must be positive"

/--
N-D convolution layer for a channels-first tensor `(batch, inputChannels, spatial...)`.

Parameters:
- weight: `(outputChannels × inputChannels × kernelSize[0] × ... × kernelSize[rank-1])`,
- bias: `(outputChannels)`.

The weight starts from `weightInit` under `weightSeed`, and the bias starts at zero. Along axis
`a` the output extent is the floor-mode sliding-window length
`(input[a] + 2 * padding[a] - kernelSize[a]) / stride[a] + 1` of `Conv.outSpatialDilated`,
with unit dilation and `padding[a]` on each side.

PyTorch analogy: `torch.nn.Conv{d}d` / `torch.nn.functional.conv{d}d` with `groups=1` and
`dilation=1`.
-/
def conv
    (batchSize rank inputChannels outputChannels : Nat)
    (kernelSize stride padding : Tensor Nat [rank])
    (inputSize : Tensor Nat [rank])
    (weightSeed : Nat := 0)
    (weightInit : Init.Scheme := .uniform (-0.1) 0.1) :
    Layer
      (Shape.dim batchSize (Shape.dim inputChannels (inputSize.to Shape)))
      (Shape.dim batchSize (Shape.dim outputChannels
        ((Conv.outSpatialDilated inputSize kernelSize stride
          (Tensor.ones [rank]) padding padding).to Shape))) :=
  let weightShape := Shape.dim outputChannels (Shape.dim inputChannels (kernelSize.to Shape))
  let biasShape : Shape := [outputChannels]
  let initialWeight : Tensor Float weightShape :=
    Init.tensor (shape := weightShape) (scheme := weightInit) (seed := weightSeed)
  let initialBias : Tensor Float biasShape := Tensor.zeros (α := Float) biasShape
  { kind := s!"Conv(rank={rank}, in={inputChannels}, out={outputChannels})"
    stateShapes := [weightShape, biasShape]
    initState := .cons initialWeight (.cons initialBias .nil)
    runtimeInit := some (.cons
      (Init.Source.scheme weightInit weightSeed)
      (.cons (.scheme .zeros) .nil))
    requiresGrad := #[true, true]
    validateConfig := do
      if inputChannels = 0 then
        throw "Conv: input channel count must be positive"
      if outputChannels = 0 then
        throw "Conv: output channel count must be positive"
      validateSpatial "Conv" inputSize kernelSize stride
      if (Conv.outSpatialDilated inputSize kernelSize stride
          (Tensor.ones [rank]) padding padding).prod = 0 then
        throw "Conv: geometry produced an empty spatial grid"
      weightInit.validate
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun weight bias input =>
          functional.conv (m := m) (α := α)
            (leadingShape := [batchSize]) (d := rank)
            (inC := inputChannels) (outC := outputChannels)
            (kernel := kernelSize) (stride := stride) (padding := padding)
            (inSpatial := inputSize)
            weight bias input
  }

/--
N-D transpose convolution layer for a channels-first tensor
`(batch, inputChannels, spatial...)`.

Parameters:
- weight: `(inputChannels × outputChannels × kernelSize[0] × ... × kernelSize[rank-1])`,
- bias: `(outputChannels)`.

The output spatial shape uses:
`output[a] = (input[a] - 1) * stride[a] - 2 * padding[a] + kernelSize[a]`
(with `output_padding = 0`).

PyTorch analogy: `torch.nn.ConvTranspose{d}d` / `torch.nn.functional.conv_transpose{d}d` with
`groups=1`, `dilation=1`, and `output_padding=0`.
-/
def convTranspose
    (batchSize rank inputChannels outputChannels : Nat)
    (kernelSize stride padding : Tensor Nat [rank])
    (inputSize : Tensor Nat [rank])
    (weightSeed : Nat := 0)
    (weightInit : Init.Scheme := .uniform (-0.1) 0.1) :
    Layer
      (Shape.dim batchSize (Shape.dim inputChannels (inputSize.to Shape)))
      (Shape.dim batchSize (Shape.dim outputChannels
        ((ConvTranspose.outSpatial inputSize kernelSize stride padding).to Shape))) :=
  let weightShape := Shape.dim inputChannels (Shape.dim outputChannels (kernelSize.to Shape))
  let biasShape : Shape := [outputChannels]
  let initialWeight : Tensor Float weightShape :=
    Init.tensor (shape := weightShape) (scheme := weightInit) (seed := weightSeed)
  let initialBias : Tensor Float biasShape := Tensor.zeros (α := Float) biasShape
  { kind := s!"ConvTranspose(rank={rank}, in={inputChannels}, out={outputChannels})"
    stateShapes := [weightShape, biasShape]
    initState := .cons initialWeight (.cons initialBias .nil)
    runtimeInit := some (.cons
      (Init.Source.scheme weightInit weightSeed)
      (.cons (.scheme .zeros) .nil))
    requiresGrad := #[true, true]
    validateConfig := do
      if inputChannels = 0 then
        throw "ConvTranspose: input channel count must be positive"
      if outputChannels = 0 then
        throw "ConvTranspose: output channel count must be positive"
      validateSpatial "ConvTranspose" inputSize kernelSize stride
      if (ConvTranspose.outSpatial inputSize kernelSize stride padding).prod = 0 then
        throw "ConvTranspose: geometry produced an empty spatial grid"
      weightInit.validate
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun weight bias input =>
          functional.convTranspose (m := m) (α := α)
            (leadingShape := [batchSize]) (d := rank)
            (inC := inputChannels) (outC := outputChannels)
            (kernel := kernelSize) (stride := stride) (padding := padding)
            (inSpatial := inputSize)
            weight bias input
  }

/--
N-D max pooling layer for a channels-first tensor `(batch, channels, spatial...)`.

Output spatial dimensions follow `Pool.outSpatial` in floor mode, and the maximum of each
window skips its padding cells.

PyTorch analogy: `torch.nn.functional.max_pool{d}d` on an `N×C×...` tensor.
-/
def maxPool
    (batchSize rank channels : Nat)
    (kernelSize stride padding : Tensor Nat [rank])
    (inputSize : Tensor Nat [rank]) :
    Layer
      (Shape.dim batchSize (Shape.dim channels (inputSize.to Shape)))
      (Shape.dim batchSize (Shape.dim channels
        ((Pool.outSpatial inputSize kernelSize stride padding).to Shape))) :=
  { kind := s!"MaxPool(rank={rank})"
    stateShapes := []
    initState := .nil
    validateConfig := do
      if channels = 0 then
        throw "MaxPool: channel count must be positive"
      validateSpatial "MaxPool" inputSize kernelSize stride
      if (Pool.outSpatial inputSize kernelSize stride padding).prod = 0 then
        throw "MaxPool: geometry produced an empty spatial grid"
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun input =>
          functional.maxPool (m := m) (α := α)
            (leadingShape := [batchSize]) (d := rank) (channels := channels)
            (inSpatial := inputSize) (kernel := kernelSize)
            (stride := stride) (padding := padding)
            input
  }

/--
N-D average pooling layer for a channels-first tensor `(batch, channels, spatial...)`.

Output spatial dimensions follow `Pool.outSpatial` in floor mode, and each average counts
the padding cells of its window, as PyTorch's default `count_include_pad=True` does.

PyTorch analogy: `torch.nn.functional.avg_pool{d}d` on an `N×C×...` tensor.
-/
def avgPool
    (batchSize rank channels : Nat)
    (kernelSize stride padding : Tensor Nat [rank])
    (inputSize : Tensor Nat [rank]) :
    Layer
      (Shape.dim batchSize (Shape.dim channels (inputSize.to Shape)))
      (Shape.dim batchSize (Shape.dim channels
        ((Pool.outSpatial inputSize kernelSize stride padding).to Shape))) :=
  { kind := s!"AvgPool(rank={rank})"
    stateShapes := []
    initState := .nil
    validateConfig := do
      if channels = 0 then
        throw "AvgPool: channel count must be positive"
      validateSpatial "AvgPool" inputSize kernelSize stride
      if (Pool.outSpatial inputSize kernelSize stride padding).prod = 0 then
        throw "AvgPool: geometry produced an empty spatial grid"
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun input =>
          functional.avgPool (m := m) (α := α)
            (leadingShape := [batchSize]) (d := rank) (channels := channels)
            (inSpatial := inputSize) (kernel := kernelSize)
            (stride := stride) (padding := padding)
            input
  }

end nn.Layer
end TorchLean
