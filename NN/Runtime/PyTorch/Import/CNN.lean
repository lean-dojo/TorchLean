/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Tensor

/-!
# CNN PyTorch Reference Import

CNN reference weight import from a PyTorch-style `state_dict`.

We mirror the common PyTorch naming convention for modules:

- `conv1.weight`, `conv1.bias`
- `conv2.weight`, `conv2.bias`
- `fc.weight`, `fc.bias`

Each tensor is expected to be encoded as nested JSON arrays whose shape matches the expected
TorchLean `Shape`.
-/

@[expose] public section

open Import.PyTorch
open Spec
open TorchLean
open Lean

namespace Import.PyTorch.CNN

/-- Parameters for the example two-block CNN imported from a PyTorch `state_dict`.

This matches the keys used by the exporter (`conv1.*`, `conv2.*`, `fc.*`) and pins down the exact
shapes expected by TorchLean.
-/
structure Parameters
    (inputChannels outputChannels kernelHeight kernelWidth flattenedWidth : Nat) where
  /-- First convolution kernel, in PyTorch `(output, input, height, width)` layout. -/
  firstConvolutionWeight :
    Tensor Float [outputChannels, inputChannels, kernelHeight, kernelWidth]
  /-- First convolution bias. -/
  firstConvolutionBias : Tensor Float [outputChannels]
  /-- Second convolution kernel. -/
  secondConvolutionWeight :
    Tensor Float [outputChannels, outputChannels, kernelHeight, kernelWidth]
  /-- Second convolution bias. -/
  secondConvolutionBias : Tensor Float [outputChannels]
  /-- Classifier weight, in PyTorch `(output, input)` layout. -/
  classifierWeight : Tensor Float [outputChannels, flattenedWidth]
  /-- Classifier bias. -/
  classifierBias : Tensor Float [outputChannels]

/-- Load CNN parameters from JSON using PyTorch `state_dict` keys. -/
def load (inputChannels outputChannels kernelHeight kernelWidth flattenedWidth : Nat)
    (json : Json) :
    Option (Parameters inputChannels outputChannels kernelHeight kernelWidth flattenedWidth) :=
  let firstConvolutionWeightShape : Shape :=
    [outputChannels, inputChannels, kernelHeight, kernelWidth]
  let firstConvolutionBiasShape : Shape := [outputChannels]
  let secondConvolutionWeightShape : Shape :=
    [outputChannels, outputChannels, kernelHeight, kernelWidth]
  let secondConvolutionBiasShape : Shape := [outputChannels]
  let classifierWeightShape : Shape := [outputChannels, flattenedWidth]
  let classifierBiasShape : Shape := [outputChannels]
  do
    -- Accepts both `{...}` and `{ "params": {...} }`.
    let weights ← loadWeights? json
    let firstConvolutionWeight ←
      getTensor? weights "conv1.weight" firstConvolutionWeightShape
    let firstConvolutionBias ← getTensor? weights "conv1.bias" firstConvolutionBiasShape
    let secondConvolutionWeight ←
      getTensor? weights "conv2.weight" secondConvolutionWeightShape
    let secondConvolutionBias ← getTensor? weights "conv2.bias" secondConvolutionBiasShape
    let classifierWeight ← getTensor? weights "fc.weight" classifierWeightShape
    let classifierBias ← getTensor? weights "fc.bias" classifierBiasShape
    pure {
      firstConvolutionWeight, firstConvolutionBias
      secondConvolutionWeight, secondConvolutionBias
      classifierWeight, classifierBias
    }

end Import.PyTorch.CNN
