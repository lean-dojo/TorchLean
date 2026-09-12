/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Core

/-!
# TorchLean NN: Convolution and Pooling Layers
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

namespace Layers

/--
N-D convolution layer for a channels-first tensor `(batch, inputChannels, spatial...)`.

Parameters:
- weight: `(outputChannels × inputChannels × kernelSize[0] × ... × kernelSize[rank-1])`,
- bias: `(outputChannels)`.

The output spatial shape is computed from `(stride, padding, kernelSize)`.

PyTorch analogy: `torch.nn.Conv{d}d` / `torch.nn.functional.conv{d}d` with `groups=1` and
`dilation=1`.
-/
def conv
    (batchSize rank inputChannels outputChannels : Nat)
    (kernelSize stride padding : TorchLean.Tensor Nat [rank])
    (inputSize : TorchLean.Tensor Nat [rank])
    (weightSeed : Nat := 0)
    (weightInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Layer
      ((inputSize.to Shape).prependDim inputChannels |>.prependDim batchSize)
      (((Spec.convOutSpatial inputSize kernelSize stride padding).to Shape)
        |>.prependDim outputChannels |>.prependDim batchSize) :=
  let weightShape := (kernelSize.to Shape).prependDim inputChannels |>.prependDim outputChannels
  let biasShape : Shape := [outputChannels]
  let initialWeight : Tensor Float weightShape :=
    Torch.Init.tensor (s := weightShape) (sch := weightInit) (seed := weightSeed)
  let initialBias : Tensor Float biasShape := Tensor.zeros (α := Float) biasShape
  { kind := s!"Conv(rank={rank}, in={inputChannels}, out={outputChannels})"
    stateShapes := [weightShape, biasShape]
    initState := .cons initialWeight (.cons initialBias .nil)
    runtimeInit := some (.cons
      (Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.ofScheme weightInit weightSeed)
      (.cons .zeros .nil))
    requiresGrad := #[true, true]
    validateConfig := do
      if inputChannels = 0 then
        throw "Conv: input channel count must be positive"
      if outputChannels = 0 then
        throw "Conv: output channel count must be positive"
      if inputSize.prod = 0 then
        throw "Conv: input spatial dimensions must be positive"
      if !decide (∀ axis : Fin rank, kernelSize.getScalar axis ≠ 0) then
        throw "Conv: kernel size entries must be positive"
      if !decide (∀ axis : Fin rank, stride.getScalar axis ≠ 0) then
        throw "Conv: stride entries must be positive"
      if (Spec.convOutSpatial inputSize kernelSize stride padding).prod = 0 then
        throw "Conv: geometry produced an empty spatial grid"
      weightInit.validate
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun weight bias input =>
          Runtime.Autograd.Model.conv (m := m) (α := α)
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
    (kernelSize stride padding : TorchLean.Tensor Nat [rank])
    (inputSize : TorchLean.Tensor Nat [rank])
    (weightSeed : Nat := 0)
    (weightInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Layer
      ((inputSize.to Shape).prependDim inputChannels |>.prependDim batchSize)
      (((Spec.convTransposeOutSpatial inputSize kernelSize stride padding).to Shape)
        |>.prependDim outputChannels |>.prependDim batchSize) :=
  let weightShape := (kernelSize.to Shape).prependDim outputChannels |>.prependDim inputChannels
  let biasShape : Shape := [outputChannels]
  let initialWeight : Tensor Float weightShape :=
    Torch.Init.tensor (s := weightShape) (sch := weightInit) (seed := weightSeed)
  let initialBias : Tensor Float biasShape := Tensor.zeros (α := Float) biasShape
  { kind := s!"ConvTranspose(rank={rank}, in={inputChannels}, out={outputChannels})"
    stateShapes := [weightShape, biasShape]
    initState := .cons initialWeight (.cons initialBias .nil)
    runtimeInit := some (.cons
      (Runtime.Autograd.Model.Module.RuntimeInit.FloatInit.ofScheme weightInit weightSeed)
      (.cons .zeros .nil))
    requiresGrad := #[true, true]
    validateConfig := do
      if inputChannels = 0 then
        throw "ConvTranspose: input channel count must be positive"
      if outputChannels = 0 then
        throw "ConvTranspose: output channel count must be positive"
      if inputSize.prod = 0 then
        throw "ConvTranspose: input spatial dimensions must be positive"
      if !decide (∀ axis : Fin rank, kernelSize.getScalar axis ≠ 0) then
        throw "ConvTranspose: kernel size entries must be positive"
      if !decide (∀ axis : Fin rank, stride.getScalar axis ≠ 0) then
        throw "ConvTranspose: stride entries must be positive"
      if (Spec.convTransposeOutSpatial inputSize kernelSize stride padding).prod = 0 then
        throw "ConvTranspose: geometry produced an empty spatial grid"
      weightInit.validate
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun weight bias input =>
          Runtime.Autograd.Model.convTranspose (m := m) (α := α)
            (leadingShape := [batchSize]) (d := rank)
            (inC := inputChannels) (outC := outputChannels)
            (kernel := kernelSize) (stride := stride) (padding := padding)
            (inSpatial := inputSize)
            weight bias input
  }

/--
N-D max pooling layer for a channels-first tensor `(batch, channels, spatial...)`.

Output spatial dimensions follow `Spec.poolOutSpatialPad`.

PyTorch analogy: `torch.nn.functional.max_pool{d}d` on an `N×C×...` tensor.
-/
def maxPool
    (batchSize rank channels : Nat)
    (kernelSize stride padding : TorchLean.Tensor Nat [rank])
    (inputSize : TorchLean.Tensor Nat [rank]) :
    Layer
      ((inputSize.to Shape).prependDim channels |>.prependDim batchSize)
      (((Spec.poolOutSpatialPad inputSize kernelSize stride padding).to Shape)
        |>.prependDim channels |>.prependDim batchSize) :=
  { kind := s!"MaxPool(rank={rank})"
    stateShapes := []
    initState := .nil
    validateConfig := do
      if channels = 0 then
        throw "MaxPool: channel count must be positive"
      if inputSize.prod = 0 then
        throw "MaxPool: input spatial dimensions must be positive"
      if !decide (∀ axis : Fin rank, kernelSize.getScalar axis ≠ 0) then
        throw "MaxPool: kernel size entries must be positive"
      if !decide (∀ axis : Fin rank, stride.getScalar axis ≠ 0) then
        throw "MaxPool: stride entries must be positive"
      if (Spec.poolOutSpatialPad inputSize kernelSize stride padding).prod = 0 then
        throw "MaxPool: geometry produced an empty spatial grid"
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun input =>
          Runtime.Autograd.Model.maxPool (m := m) (α := α)
            (leadingShape := [batchSize]) (d := rank) (channels := channels)
            (inSpatial := inputSize) (kernel := kernelSize)
            (stride := stride) (padding := padding)
            input
  }

/--
N-D average pooling layer for a channels-first tensor `(batch, channels, spatial...)`.

PyTorch analogy: `torch.nn.functional.avg_pool{d}d` on an `N×C×...` tensor.
-/
def avgPool
    (batchSize rank channels : Nat)
    (kernelSize stride padding : TorchLean.Tensor Nat [rank])
    (inputSize : TorchLean.Tensor Nat [rank]) :
    Layer
      ((inputSize.to Shape).prependDim channels |>.prependDim batchSize)
      (((Spec.poolOutSpatialPad inputSize kernelSize stride padding).to Shape)
        |>.prependDim channels |>.prependDim batchSize) :=
  { kind := s!"AvgPool(rank={rank})"
    stateShapes := []
    initState := .nil
    validateConfig := do
      if channels = 0 then
        throw "AvgPool: channel count must be positive"
      if inputSize.prod = 0 then
        throw "AvgPool: input spatial dimensions must be positive"
      if !decide (∀ axis : Fin rank, kernelSize.getScalar axis ≠ 0) then
        throw "AvgPool: kernel size entries must be positive"
      if !decide (∀ axis : Fin rank, stride.getScalar axis ≠ 0) then
        throw "AvgPool: stride entries must be positive"
      if (Spec.poolOutSpatialPad inputSize kernelSize stride padding).prod = 0 then
        throw "AvgPool: geometry produced an empty spatial grid"
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun input =>
          Runtime.Autograd.Model.avgPool (m := m) (α := α)
            (leadingShape := [batchSize]) (d := rank) (channels := channels)
            (inSpatial := inputSize) (kernel := kernelSize)
            (stride := stride) (padding := padding)
            input
  }

end Layers

end Model
end Autograd
end Runtime
