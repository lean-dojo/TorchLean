/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Spec.Models.Cnn

/-!
# Convolutional PyTorch Reference Export

PyTorch exporter for the two-block convolutional round-trip reference model.

The Lean configuration is rank-parametric. PyTorch itself exposes separate `Conv1d`, `Conv2d`, and
`Conv3d` classes, so that distinction is introduced only while rendering the external Python code.

The generated model has two convolution, ReLU, and max-pool blocks followed by `Flatten` and one
`Linear` head.
-/

@[expose] public section

open Spec TorchLean
open TorchLean.Tensor
open Spec.Module
open Models
open Export.PyTorch

namespace Export.PyTorch.CNN

/-- Rank-parametric configuration for a PyTorch convolution layer. -/
structure ConvolutionConfig (spatialRank : Nat) where
  /-- Input channels (`in_channels`). -/
  inputChannels : Nat
  /-- Output channels (`out_channels`). -/
  outputChannels : Nat
  /-- Kernel extent along each spatial axis. -/
  kernel : TorchLean.Tensor Nat [spatialRank]
  /-- Stride along each spatial axis. -/
  stride : TorchLean.Tensor Nat [spatialRank]
  /-- Zero-padding along each spatial axis. -/
  padding : TorchLean.Tensor Nat [spatialRank]

/-- Rank-parametric configuration for a PyTorch max-pooling layer. -/
structure PoolingConfig (spatialRank : Nat) where
  /-- Pooling-window extent along each spatial axis. -/
  kernel : TorchLean.Tensor Nat [spatialRank]
  /-- Stride along each spatial axis. -/
  stride : TorchLean.Tensor Nat [spatialRank]
  /-- Zero-padding along each spatial axis. -/
  padding : TorchLean.Tensor Nat [spatialRank]

/-- Configuration for the 2-block CNN exporter. -/
structure Config (spatialRank : Nat) where
  /-- Class name to use in the generated Python. -/
  className : String := "CNN"
  /-- Input channels. -/
  inputChannels : Nat
  /-- Input extent along each spatial axis. -/
  inputSpatial : TorchLean.Tensor Nat [spatialRank]
  /-- First convolution. -/
  firstConvolution : ConvolutionConfig spatialRank
  /-- First pooling layer. -/
  firstPooling : PoolingConfig spatialRank
  /-- Second convolution. -/
  secondConvolution : ConvolutionConfig spatialRank
  /-- Second pooling layer. -/
  secondPooling : PoolingConfig spatialRank
  /-- Flattened feature count consumed by the linear head. -/
  flattenedWidth : Nat
  /-- Output width of the linear head. -/
  outputWidth : Nat

/-- Render a tensor shape as a Python tuple. -/
def dimensionsToPythonTuple (shape : Shape) : String :=
  let dimensions := shape.toList
  "(" ++ ", ".intercalate (dimensions.map toString) ++
    (if dimensions.length = 1 then "," else "") ++ ")"

/-- Select the rank-specific class name required by PyTorch's public API. -/
def spatialClassName (base : String) (spatialRank : Nat) : Except String String :=
  match spatialRank with
  | 1 => .ok s!"{base}1d"
  | 2 => .ok s!"{base}2d"
  | 3 => .ok s!"{base}3d"
  | rank => .error s!"PyTorch provides {base} only for spatial ranks 1, 2, and 3; got {rank}"

/-- Render the two-block CNN as a Python `nn.Module` class definition. -/
def classSource {spatialRank : Nat}
    (config : Config spatialRank) : Except String String := do
  let convClass ← spatialClassName "Conv" spatialRank
  let poolClass ← spatialClassName "MaxPool" spatialRank
  let className := config.className
  let tuple := fun (dimensions : TorchLean.Tensor Nat [spatialRank]) =>
    dimensionsToPythonTuple (dimensions.to Shape)
  let inputShape :=
    dimensionsToPythonTuple <|
      (config.inputSpatial.to Shape).prependDim config.inputChannels
  pure <| joinLines <|
    #[generatePyTorchImports, ""] ++
    #[
      s!"class {className}(nn.Module):",
      indentTwo s!"\"\"\"Two {convClass} / ReLU / {poolClass} blocks, then Flatten / Linear.\"\"\"",
      indentTwo "",
      indentTwo "def __init__(self):",
      indentFour "super().__init__()",
      indentFour (s!"self.conv1 = nn.{convClass}({config.firstConvolution.inputChannels}, " ++
        s!"{config.firstConvolution.outputChannels}, " ++
        s!"kernel_size={tuple config.firstConvolution.kernel}, " ++
        s!"stride={tuple config.firstConvolution.stride}, " ++
        s!"padding={tuple config.firstConvolution.padding})"),
      indentFour "self.relu1 = nn.ReLU()",
      indentFour (s!"self.pool1 = nn.{poolClass}(" ++
        s!"kernel_size={tuple config.firstPooling.kernel}, " ++
        s!"stride={tuple config.firstPooling.stride}, " ++
        s!"padding={tuple config.firstPooling.padding})"),
      indentFour (s!"self.conv2 = nn.{convClass}({config.secondConvolution.inputChannels}, " ++
        s!"{config.secondConvolution.outputChannels}, " ++
        s!"kernel_size={tuple config.secondConvolution.kernel}, " ++
        s!"stride={tuple config.secondConvolution.stride}, " ++
        s!"padding={tuple config.secondConvolution.padding})"),
      indentFour "self.relu2 = nn.ReLU()",
      indentFour (s!"self.pool2 = nn.{poolClass}(" ++
        s!"kernel_size={tuple config.secondPooling.kernel}, " ++
        s!"stride={tuple config.secondPooling.stride}, " ++
        s!"padding={tuple config.secondPooling.padding})"),
      indentFour "self.flatten = nn.Flatten()",
      indentFour s!"self.fc = nn.Linear({config.flattenedWidth}, {config.outputWidth})",
      indentTwo "",
      indentTwo "def forward(self, x):",
      indentFour "x = self.conv1(x)",
      indentFour "x = self.relu1(x)",
      indentFour "x = self.pool1(x)",
      indentFour "x = self.conv2(x)",
      indentFour "x = self.relu2(x)",
      indentFour "x = self.pool2(x)",
      indentFour "x = self.flatten(x)",
      indentFour "x = self.fc(x)",
      indentFour "return x",
      indentTwo "",
      indentTwo "@property",
      indentTwo "def input_shape(self):",
      indentFour s!"return {inputShape}",
      indentTwo "",
      indentTwo "@property",
      indentTwo "def output_shape(self):",
      indentFour s!"return ({config.outputWidth},)",
      indentTwo "",
      indentTwo "@property",
      indentTwo "def layer_count(self):",
      indentFour "return 8",
      indentTwo "",
      indentTwo "@property",
      indentTwo "def operation_types(self):",
      indentFour (s!"return [\"{convClass}\", \"ReLU\", \"{poolClass}\", \"{convClass}\", " ++
        s!"\"ReLU\", \"{poolClass}\", \"Flatten\", \"Linear\"]"),
      indentTwo ""
    ]
    ++ generateGetModelInfoMethodLines className

/-- Generate a Python CNN module plus a helper that loads explicit weights from string literals.

This is mainly used for examples: you can paste JSON/Lean-rendered weight arrays into Python and run
the model without writing an extra serializer.
-/
def withParameters {spatialRank : Nat} (config : Config spatialRank)
    (firstConvolutionWeight firstConvolutionBias : String)
    (secondConvolutionWeight secondConvolutionBias : String)
    (classifierWeight classifierBias : String) : Except String String := do
  let classCode ← classSource config
  let batchInputShape :=
    dimensionsToPythonTuple <|
      ((config.inputSpatial.to Shape).prependDim config.inputChannels).prependDim 1
  pure <| joinLines #[
    classCode,
    "",
    "# Weight initialization functions",
    "def get_cnn_state_dict():",
    indentTwo "state_dict = {}",
    indentTwo s!"state_dict['conv1.weight'] = torch.tensor({firstConvolutionWeight})",
    indentTwo s!"state_dict['conv1.bias'] = torch.tensor({firstConvolutionBias})",
    indentTwo s!"state_dict['conv2.weight'] = torch.tensor({secondConvolutionWeight})",
    indentTwo s!"state_dict['conv2.bias'] = torch.tensor({secondConvolutionBias})",
    indentTwo s!"state_dict['fc.weight'] = torch.tensor({classifierWeight})",
    indentTwo s!"state_dict['fc.bias'] = torch.tensor({classifierBias})",
    indentTwo "return state_dict",
    indentTwo "",
    "def load_cnn_weights(model):",
    indentTwo "state_dict = get_cnn_state_dict()",
    indentTwo "model.load_state_dict(state_dict)",
    indentTwo "return model",
    indentTwo "",
    "# Usage example",
    "if __name__ == \"__main__\":",
    indentTwo s!"model = {config.className}()",
    indentTwo "model = load_cnn_weights(model)",
    indentTwo s!"x = torch.randn{batchInputShape}",
    indentTwo "y = model(x)",
    indentTwo "print(f\"Input shape: {x.shape}\")",
    indentTwo "print(f\"Output shape: {y.shape}\")",
    indentTwo "print(f\"Output: {y}\")",
    indentTwo "print(f\"Model info: {model.get_model_info()}\")"
  ]

end Export.PyTorch.CNN
