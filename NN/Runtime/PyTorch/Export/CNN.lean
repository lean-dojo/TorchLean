/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core

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
open Export.PyTorch

namespace Export.PyTorch.CNN

/-- Shared spatial geometry for convolution and pooling layers. -/
structure Window (spatialRank : Nat) where
  /-- Kernel extent along each spatial axis. -/
  kernel : TorchLean.Tensor Nat [spatialRank]
  /-- Stride along each spatial axis. -/
  stride : TorchLean.Tensor Nat [spatialRank]
  /-- Symmetric padding extent along each spatial axis. -/
  padding : TorchLean.Tensor Nat [spatialRank]

/-- A convolution window together with its channel counts. -/
structure Convolution (spatialRank : Nat) extends Window spatialRank where
  /-- Input channels (`in_channels`). -/
  inputChannels : Nat
  /-- Output channels (`out_channels`). -/
  outputChannels : Nat

/-- Configuration for the 2-block CNN exporter. -/
structure Config (spatialRank : Nat) where
  /-- Class name to use in the generated Python. -/
  className : String := "CNN"
  /-- Input channels. -/
  inputChannels : Nat
  /-- Input extent along each spatial axis. -/
  inputSpatial : TorchLean.Tensor Nat [spatialRank]
  /-- First convolution. -/
  firstConvolution : Convolution spatialRank
  /-- First pooling layer. -/
  firstPooling : Window spatialRank
  /-- Second convolution. -/
  secondConvolution : Convolution spatialRank
  /-- Second pooling layer. -/
  secondPooling : Window spatialRank
  /-- Flattened feature count consumed by the linear head. -/
  flattenedWidth : Nat
  /-- Output width of the linear head. -/
  outputWidth : Nat

/-- Select the rank-specific class name required by PyTorch's public API. -/
def operator (base : String) (spatialRank : Nat) : Except String String :=
  match spatialRank with
  | 1 => .ok s!"{base}1d"
  | 2 => .ok s!"{base}2d"
  | 3 => .ok s!"{base}3d"
  | rank => .error s!"PyTorch provides {base} only for spatial ranks 1, 2, and 3; got {rank}"

/-- Render the two-block CNN as a Python `nn.Module` class definition. -/
def source {spatialRank : Nat}
    (config : Config spatialRank) : Except String String := do
  let convClass ← operator "Conv" spatialRank
  let poolClass ← operator "MaxPool" spatialRank
  let className := config.className
  let inputShape :=
    shapeLiteral <|
      (config.inputSpatial.to Shape).prependDim config.inputChannels
  pure <| joinLines <|
    #[imports, ""] ++
    #[
      s!"class {className}(nn.Module):",
      indent 2 s!"\"\"\"Two {convClass} / ReLU / {poolClass} blocks, then Flatten / Linear.\"\"\"",
      indent 2 "",
      indent 2 "def __init__(self):",
      indent 4 "super().__init__()",
      indent 4 (s!"self.conv1 = nn.{convClass}({config.firstConvolution.inputChannels}, " ++
        s!"{config.firstConvolution.outputChannels}, " ++
        s!"kernel_size={tupleLiteral config.firstConvolution.kernel}, " ++
        s!"stride={tupleLiteral config.firstConvolution.stride}, " ++
        s!"padding={tupleLiteral config.firstConvolution.padding})"),
      indent 4 "self.relu1 = nn.ReLU()",
      indent 4 (s!"self.pool1 = nn.{poolClass}(" ++
        s!"kernel_size={tupleLiteral config.firstPooling.kernel}, " ++
        s!"stride={tupleLiteral config.firstPooling.stride}, " ++
        s!"padding={tupleLiteral config.firstPooling.padding})"),
      indent 4 (s!"self.conv2 = nn.{convClass}({config.secondConvolution.inputChannels}, " ++
        s!"{config.secondConvolution.outputChannels}, " ++
        s!"kernel_size={tupleLiteral config.secondConvolution.kernel}, " ++
        s!"stride={tupleLiteral config.secondConvolution.stride}, " ++
        s!"padding={tupleLiteral config.secondConvolution.padding})"),
      indent 4 "self.relu2 = nn.ReLU()",
      indent 4 (s!"self.pool2 = nn.{poolClass}(" ++
        s!"kernel_size={tupleLiteral config.secondPooling.kernel}, " ++
        s!"stride={tupleLiteral config.secondPooling.stride}, " ++
        s!"padding={tupleLiteral config.secondPooling.padding})"),
      indent 4 "self.flatten = nn.Flatten()",
      indent 4 s!"self.fc = nn.Linear({config.flattenedWidth}, {config.outputWidth})",
      indent 2 "",
      indent 2 "def forward(self, x):",
      indent 4 "x = self.conv1(x)",
      indent 4 "x = self.relu1(x)",
      indent 4 "x = self.pool1(x)",
      indent 4 "x = self.conv2(x)",
      indent 4 "x = self.relu2(x)",
      indent 4 "x = self.pool2(x)",
      indent 4 "x = self.flatten(x)",
      indent 4 "x = self.fc(x)",
      indent 4 "return x",
      indent 2 "",
      indent 2 "@property",
      indent 2 "def input_shape(self):",
      indent 4 s!"return {inputShape}",
      indent 2 "",
      indent 2 "@property",
      indent 2 "def output_shape(self):",
      indent 4 s!"return ({config.outputWidth},)",
      indent 2 "",
      indent 2 "@property",
      indent 2 "def layer_count(self):",
      indent 4 "return 8",
      indent 2 "",
      indent 2 "@property",
      indent 2 "def operation_types(self):",
      indent 4 (s!"return [\"{convClass}\", \"ReLU\", \"{poolClass}\", \"{convClass}\", " ++
        s!"\"ReLU\", \"{poolClass}\", \"Flatten\", \"Linear\"]"),
      indent 2 ""
    ]
    ++ metadata className

/-- Generate a Python CNN module plus a helper that loads explicit weights from string literals.

This is mainly used for examples: you can paste JSON/Lean-rendered weight arrays into Python and run
the model without writing an extra serializer.
-/
def weights {spatialRank : Nat} (config : Config spatialRank)
    (firstConvolutionWeight firstConvolutionBias : String)
    (secondConvolutionWeight secondConvolutionBias : String)
    (classifierWeight classifierBias : String) : Except String String := do
  let classCode ← source config
  let batchInputShape :=
    shapeLiteral <|
      ((config.inputSpatial.to Shape).prependDim config.inputChannels).prependDim 1
  pure <| joinLines #[
    classCode,
    "",
    "# Weight initialization functions",
    "def get_cnn_state_dict():",
    indent 2 "state_dict = {}",
    indent 2 s!"state_dict['conv1.weight'] = torch.tensor({firstConvolutionWeight})",
    indent 2 s!"state_dict['conv1.bias'] = torch.tensor({firstConvolutionBias})",
    indent 2 s!"state_dict['conv2.weight'] = torch.tensor({secondConvolutionWeight})",
    indent 2 s!"state_dict['conv2.bias'] = torch.tensor({secondConvolutionBias})",
    indent 2 s!"state_dict['fc.weight'] = torch.tensor({classifierWeight})",
    indent 2 s!"state_dict['fc.bias'] = torch.tensor({classifierBias})",
    indent 2 "return state_dict",
    indent 2 "",
    "def load_cnn_weights(model):",
    indent 2 "state_dict = get_cnn_state_dict()",
    indent 2 "model.load_state_dict(state_dict)",
    indent 2 "return model",
    indent 2 "",
    "# Usage example",
    "if __name__ == \"__main__\":",
    indent 2 s!"model = {config.className}()",
    indent 2 "model = load_cnn_weights(model)",
    indent 2 s!"x = torch.randn{batchInputShape}",
    indent 2 "y = model(x)",
    indent 2 "print(f\"Input shape: {x.shape}\")",
    indent 2 "print(f\"Output shape: {y.shape}\")",
    indent 2 "print(f\"Output: {y}\")",
    indent 2 "print(f\"Model info: {model.get_model_info()}\")"
  ]

end Export.PyTorch.CNN
