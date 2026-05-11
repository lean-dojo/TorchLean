/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Spec.Models.Cnn

/-!
# CNN PyTorch Fixture Export

PyTorch exporter for the small 2-block ConvNet round-trip fixture.

This exporter is meant to mirror the "classic demo CNN" shape that shows up in many TorchLean
examples: two `Conv2d + ReLU + MaxPool2d` blocks, then `Flatten`, then a single `Linear` head.

Instead of taking a long positional list of naturals, we use small configuration records so call
sites stay readable and it's easy to extend the shape later.
-/

@[expose] public section


namespace Export
namespace CNNPyTorch

open Spec
open Tensor
open ModSpec
open Models
open Export.PyTorch

/-- Configuration for a PyTorch `nn.Conv2d` layer. -/
structure Conv2dCfg where
  /-- Input channels (`in_channels`). -/
  inChannels : Nat
  /-- Output channels (`out_channels`). -/
  outChannels : Nat
  /-- Kernel height. -/
  kernelH : Nat
  /-- Kernel width. -/
  kernelW : Nat
  /-- Stride (applied to both spatial dims). -/
  stride : Nat
  /-- Zero-padding (applied to both spatial dims). -/
  padding : Nat

/-- Configuration for a PyTorch `nn.MaxPool2d` layer. -/
structure MaxPool2dCfg where
  /-- Pool kernel height. -/
  kernelH : Nat
  /-- Pool kernel width. -/
  kernelW : Nat
  /-- Pool stride (applied to both spatial dims). -/
  stride : Nat

/-- Configuration for the 2-block CNN exporter. -/
structure Cnn2Cfg where
  /-- Class name to use in the generated Python. -/
  className : String := "CNN"
  /-- Input image channels. -/
  inputC : Nat
  /-- Input image height. -/
  inputH : Nat
  /-- Input image width. -/
  inputW : Nat
  /-- conv 1. -/
  conv1 : Conv2dCfg
  /-- pool 1. -/
  pool1 : MaxPool2dCfg
  /-- conv 2. -/
  conv2 : Conv2dCfg
  /-- pool 2. -/
  pool2 : MaxPool2dCfg
  /-- flat Size. -/
  flatSize : Nat
  /-- fc Out. -/
  fcOut : Nat

/-- Render the 2-block CNN as a Python `nn.Module` class definition. -/
def generateCnn2PyTorchClass (cfg : Cnn2Cfg) : String :=
  let className := cfg.className
  joinLines <|
    [generatePyTorchImports, ""] ++
    [
      s!"class {className}(nn.Module):",
      indent2 "\"\"\"2-block ConvNet (Conv2d → ReLU → MaxPool2d) × 2, then Flatten → Linear.\"\"\"",
      indent2 "",
      indent2 "def __init__(self):",
      indent4 "super().__init__()",
      (s!"self.conv1 = nn.Conv2d({cfg.conv1.inChannels}, " ++
        s!"{cfg.conv1.outChannels}, kernel_size=({cfg.conv1.kernelH}, " ++
        s!"{cfg.conv1.kernelW}), stride={cfg.conv1.stride}, " ++
        s!"padding={cfg.conv1.padding})"),
      indent4 "self.relu1 = nn.ReLU()",
      (s!"self.pool1 = nn.MaxPool2d(kernel_size=({cfg.pool1.kernelH}, " ++
        s!"{cfg.pool1.kernelW}), stride={cfg.pool1.stride})"),
      (s!"self.conv2 = nn.Conv2d({cfg.conv2.inChannels}, " ++
        s!"{cfg.conv2.outChannels}, kernel_size=({cfg.conv2.kernelH}, " ++
        s!"{cfg.conv2.kernelW}), stride={cfg.conv2.stride}, " ++
        s!"padding={cfg.conv2.padding})"),
      indent4 "self.relu2 = nn.ReLU()",
      (s!"self.pool2 = nn.MaxPool2d(kernel_size=({cfg.pool2.kernelH}, " ++
        s!"{cfg.pool2.kernelW}), stride={cfg.pool2.stride})"),
      indent4 "self.flatten = nn.Flatten()",
      indent4 s!"self.fc = nn.Linear({cfg.flatSize}, {cfg.fcOut})",
      indent2 "",
      indent2 "def forward(self, x):",
      indent4 "x = self.conv1(x)",
      indent4 "x = self.relu1(x)",
      indent4 "x = self.pool1(x)",
      indent4 "x = self.conv2(x)",
      indent4 "x = self.relu2(x)",
      indent4 "x = self.pool2(x)",
      indent4 "x = self.flatten(x)",
      indent4 "x = self.fc(x)",
      indent4 "return x",
      indent2 "",
      indent2 "@property",
      indent2 "def input_shape(self):",
      indent4 s!"return ({cfg.inputC}, {cfg.inputH}, {cfg.inputW})",
      indent2 "",
      indent2 "@property",
      indent2 "def output_shape(self):",
      indent4 s!"return ({cfg.fcOut},)",
      indent2 "",
      indent2 "@property",
      indent2 "def layer_count(self):",
      indent4 "return 8",
      indent2 "",
      indent2 "@property",
      indent2 "def operation_types(self):",
      ("return [\"Conv2D\", \"ReLU\", \"MaxPool2D\", \"Conv2D\", \"ReLU\", " ++
        "\"MaxPool2D\", \"Flatten\", \"Linear\"]"),
      indent2 ""
    ]
    ++ generateGetModelInfoMethodLines className

/-- Backwards-compatible wrapper for `generateCnn2PyTorchClass`.

Call sites that pass a long positional list of naturals can use this adapter.
-/
def generateCNNPyTorchClass
    (inC outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1 poolStride2
      flatSize : Nat)
    (className : String := "CNN") : String :=
  let cfg : Cnn2Cfg :=
    { className := className
      inputC := inC, inputH := inH, inputW := inW
      conv1 := { inChannels := inC, outChannels := outC, kernelH := kH, kernelW := kW, stride :=
        stride1, padding := padding1 }
      pool1 := { kernelH := poolKH, kernelW := poolKW, stride := poolStride1 }
      conv2 := { inChannels := outC, outChannels := outC, kernelH := kH, kernelW := kW, stride :=
        stride2, padding := padding2 }
      pool2 := { kernelH := poolKH, kernelW := poolKW, stride := poolStride2 }
      flatSize := flatSize
      fcOut := outC }
  generateCnn2PyTorchClass cfg

/-- Generate a Python CNN module plus a helper that loads explicit weights from string literals.

This is mainly used for demos: you can paste JSON/Lean-rendered weight arrays into Python and run
the model without writing an extra serializer.
-/
def generateCNNWithWeights (convW1 convB1 convW2 convB2 linearW linearB : String)
    (inC outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolstride1 poolstride2
      flatSize : Nat)
    (className : String := "CNN") : String :=
  joinLines [
    generateCNNPyTorchClass inC outC inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW
      poolstride1 poolstride2 flatSize className,
    "",
    "# Weight initialization functions",
    "def get_cnn_state_dict():",
    indent2 "state_dict = {}",
    indent2 s!"state_dict['conv1.weight'] = torch.tensor({convW1})",
    indent2 s!"state_dict['conv1.bias'] = torch.tensor({convB1})",
    indent2 s!"state_dict['conv2.weight'] = torch.tensor({convW2})",
    indent2 s!"state_dict['conv2.bias'] = torch.tensor({convB2})",
    indent2 s!"state_dict['fc.weight'] = torch.tensor({linearW})",
    indent2 s!"state_dict['fc.bias'] = torch.tensor({linearB})",
    indent2 "return state_dict",
    indent2 "",
    "def load_cnn_weights(model):",
    indent2 "state_dict = get_cnn_state_dict()",
    indent2 "model.load_state_dict(state_dict)",
    indent2 "return model",
    indent2 "",
    "# Usage example",
    "if __name__ == \"__main__\":",
    indent2 s!"model = {className}()",
    indent2 "model = load_cnn_weights(model)",
    indent2 s!"x = torch.randn(1, {inC}, {inH}, {inW})  # batch_size=1, channels, height, width",
    indent2 "y = model(x)",
    indent2 "print(f\"Input shape: {x.shape}\")",
    indent2 "print(f\"Output shape: {y.shape}\")",
    indent2 "print(f\"Output: {y}\")",
    indent2 "print(f\"Model info: {model.get_model_info()}\")"
  ]

end CNNPyTorch
end Export
