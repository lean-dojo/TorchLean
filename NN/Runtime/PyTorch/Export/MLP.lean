/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core

/-!
# MLP PyTorch Reference Export

PyTorch code generator for the MLP round-trip reference model.

The generated Python mirrors the common `nn.Linear → ReLU → nn.Linear` pattern. We also support
embedding explicit weights into a `state_dict`-shaped dictionary for round-trip and regression
checks.
-/

@[expose] public section

open TorchLean
open Export.PyTorch

namespace Export.PyTorch.MLP

/-- How to name `state_dict` keys when exporting weights. -/
inductive KeyStyle where
  /-- Keys like `fc1.weight` / `fc2.bias` (matches PyTorch `nn.Linear` modules). -/
  | linear
  /-- Keys like `layers.0.weight` / `layers.2.bias` (common when exporting `nn.Sequential`). -/
  | sequential
  deriving DecidableEq, Repr

/-- Key name for the first layer's weight tensor in a PyTorch `state_dict`. -/
def firstWeightKey : KeyStyle → String
  | .linear => "fc1.weight"
  | .sequential => "layers.0.weight"

/-- Key name for the first layer's bias tensor in a PyTorch `state_dict`. -/
def firstBiasKey : KeyStyle → String
  | .linear => "fc1.bias"
  | .sequential => "layers.0.bias"

/-- Key name for the second layer's weight tensor in a PyTorch `state_dict`. -/
def secondWeightKey : KeyStyle → String
  | .linear => "fc2.weight"
  | .sequential => "layers.2.weight"

/-- Key name for the second layer's bias tensor in a PyTorch `state_dict`. -/
def secondBiasKey : KeyStyle → String
  | .linear => "fc2.bias"
  | .sequential => "layers.2.bias"

/--
Emit the Python class body for a basic `Linear → ReLU → Linear` MLP.

This returns *lines* (not a single string) so callers can splice it into larger scripts.
-/
def classLines
    (inputWidth hiddenWidth outputWidth : Nat) (className : String) : Array String :=
  #[
    s!"class {className}(nn.Module):",
    indentTwo (s!"\"\"\"Multi-Layer Perceptron with {inputWidth} input, " ++
      s!"{hiddenWidth} hidden, {outputWidth} output dimensions\"\"\""),
    indentTwo "",
    indentTwo (s!"def __init__(self, input_dim: int = {inputWidth}, hidden_dim: int = " ++
      s!"{hiddenWidth}, output_dim: int = {outputWidth}):"),
    indentFour "super().__init__()",
    indentFour "self.input_dim = input_dim",
    indentFour "self.hidden_dim = hidden_dim",
    indentFour "self.output_dim = output_dim",
    indentFour "",
    indentFour "# Define layers",
    indentFour "self.fc1 = nn.Linear(input_dim, hidden_dim)",
    indentFour "self.relu = nn.ReLU()",
    indentFour "self.fc2 = nn.Linear(hidden_dim, output_dim)",
    indentFour "",
    indentTwo "def forward(self, x):",
    indentFour "x = self.fc1(x)",
    indentFour "x = self.relu(x)",
    indentFour "x = self.fc2(x)",
    indentFour "return x",
    indentFour "",
    indentTwo "@property",
    indentTwo "def input_shape(self):",
    indentFour "return (self.input_dim,)",
    indentFour "",
    indentTwo "@property",
    indentTwo "def output_shape(self):",
    indentFour "return (self.output_dim,)",
    indentFour "",
    indentTwo "@property",
    indentTwo "def layer_count(self):",
    indentFour "return 3",  -- fc1, relu, fc2
    indentFour "",
    indentTwo "@property",
    indentTwo "def operation_types(self):",
    indentFour "return [\"Linear\", \"ReLU\", \"Linear\"]",
    indentFour ""
  ] ++
    generateGetModelInfoMethodLines className
      #[ ("input_dim", "self.input_dim")
      , ("hidden_dim", "self.hidden_dim")
      , ("output_dim", "self.output_dim")
      ]

/-- Render a standalone Python file containing an `nn.Module` MLP class. -/
def classSource (inputWidth hiddenWidth outputWidth : Nat)
    (className : String := "MLP") : String :=
  joinLines <|
    #[generatePyTorchImports, ""] ++ classLines inputWidth hiddenWidth outputWidth className

/--
Generate Python code for an MLP plus helper functions that embed concrete weights.

The output contains a `get_mlp_state_dict` function that returns a PyTorch-shaped dictionary
(`state_dict`). Its `load_mlp_weights` helper normalizes either key convention to the generated
class's `fc1`/`fc2` layers before calling `model.load_state_dict(...)`.
-/
def withParameters {inputWidth hiddenWidth outputWidth : Nat}
    (inputWeight : Tensor Float [hiddenWidth, inputWidth])
    (inputBias : Tensor Float [hiddenWidth])
    (outputWeight : Tensor Float [outputWidth, hiddenWidth])
    (outputBias : Tensor Float [outputWidth])
    (className : String := "MLP")
    (keyStyle : KeyStyle := .linear) : String :=
  joinLines #[
    classSource inputWidth hiddenWidth outputWidth className,
    "",
    "# Weight initialization functions",
    "def get_mlp_state_dict():",
    indentTwo "state_dict = {}",
    indentTwo
      s!"state_dict['{firstWeightKey keyStyle}'] = torch.tensor({tensorToPyString inputWeight})",
    indentTwo
      s!"state_dict['{firstBiasKey keyStyle}'] = torch.tensor({tensorToPyString inputBias})",
    indentTwo
      s!"state_dict['{secondWeightKey keyStyle}'] = torch.tensor({tensorToPyString outputWeight})",
    indentTwo
      s!"state_dict['{secondBiasKey keyStyle}'] = torch.tensor({tensorToPyString outputBias})",
    indentTwo "return state_dict",
    indentTwo "",
    "def load_mlp_weights(model):",
    indentTwo "state_dict = get_mlp_state_dict()",
    indentTwo "# Normalize either export key convention to this class's named layers.",
    indentTwo "state_dict = {",
    indentFour s!"'fc1.weight': state_dict['{firstWeightKey keyStyle}'],",
    indentFour s!"'fc1.bias': state_dict['{firstBiasKey keyStyle}'],",
    indentFour s!"'fc2.weight': state_dict['{secondWeightKey keyStyle}'],",
    indentFour s!"'fc2.bias': state_dict['{secondBiasKey keyStyle}'],",
    indentTwo "}",
    indentTwo "model.load_state_dict(state_dict)",
    indentTwo "return model",
    indentTwo "",
    "# Usage example",
    "if __name__ == \"__main__\":",
    indentTwo s!"model = {className}()",
    indentTwo "model = load_mlp_weights(model)",
    indentTwo
      s!"x = torch.randn(1, {inputWidth})  # batch_size=1, features={inputWidth}",
    indentTwo "y = model(x)",
    indentTwo "print(f\"Input shape: {x.shape}\")",
    indentTwo "print(f\"Output shape: {y.shape}\")",
    indentTwo "print(f\"Output: {y}\")",
    indentTwo "print(f\"Model info: {model.get_model_info()}\")"
  ]

/-- Render a line-based MLP class with a terminal softmax. -/
def softmaxClassLines
    {inputWidth hiddenWidth outputWidth : Nat} (className : String) : Array String :=
  #[
    s!"class {className}(nn.Module):",
    indentTwo s!"\"\"\"Multi-Layer Perceptron with softmax output for classification\"\"\"",
    indentTwo "",
    indentTwo (s!"def __init__(self, input_dim: int = {inputWidth}, hidden_dim: int = " ++
      s!"{hiddenWidth}, output_dim: int = {outputWidth}):"),
    indentFour "super().__init__()",
    indentFour "self.input_dim = input_dim",
    indentFour "self.hidden_dim = hidden_dim",
    indentFour "self.output_dim = output_dim",
    indentFour "",
    indentFour "# Define layers",
    indentFour "self.fc1 = nn.Linear(input_dim, hidden_dim)",
    indentFour "self.relu = nn.ReLU()",
    indentFour "self.fc2 = nn.Linear(hidden_dim, output_dim)",
    indentFour "self.softmax = nn.Softmax(dim=-1)",
    indentFour "",
    indentTwo "def forward(self, x):",
    indentFour "x = self.fc1(x)",
    indentFour "x = self.relu(x)",
    indentFour "x = self.fc2(x)",
    indentFour "x = self.softmax(x)",
    indentFour "return x",
    indentFour "",
    indentTwo "@property",
    indentTwo "def input_shape(self):",
    indentFour "return (self.input_dim,)",
    indentFour "",
    indentTwo "@property",
    indentTwo "def output_shape(self):",
    indentFour "return (self.output_dim,)",
    indentFour "",
    indentTwo "@property",
    indentTwo "def layer_count(self):",
    indentFour "return 4",  -- fc1, relu, fc2, softmax
    indentFour "",
    indentTwo "@property",
    indentTwo "def operation_types(self):",
    indentFour "return [\"Linear\", \"ReLU\", \"Linear\", \"Softmax\"]"
  ]

/--
Generate a complete Python script for MLP examples.

This includes:
- a base MLP class,
- a Softmax variant,
- shared helper modules from `NN/Runtime/PyTorch/Export/Core.lean`,
- and convenience helpers for construction and parameter counting.
-/
def completeSource {inputWidth hiddenWidth outputWidth : Nat}
    (className : String := "MLP") : String :=
  joinLines #[
    generatePyTorchImports,
    "",
    joinLines (classLines inputWidth hiddenWidth outputWidth className),
    "",
    joinLines (softmaxClassLines
      (inputWidth := inputWidth) (hiddenWidth := hiddenWidth) (outputWidth := outputWidth)
      s!"{className}WithSoftmax"),
    "",
    generateWeightLoadingUtils,
    "",
    generateTestingUtils,
    "",
    "# MLP-specific utilities",
    ("def create_mlp_from_spec(input_dim: int, hidden_dim: int, output_dim: " ++
      "int, use_softmax: bool = False):"),
    indentTwo "\"\"\"Create an MLP model from specifications.\"\"\"",
    indentTwo "if use_softmax:",
      indentFour s!"return {className}WithSoftmax(input_dim, hidden_dim, output_dim)",
    indentTwo "else:",
      indentFour s!"return {className}(input_dim, hidden_dim, output_dim)",
    indentTwo "",
    "def mlp_parameter_count(input_dim: int, hidden_dim: int, output_dim: int) -> int:",
    indentTwo "\"\"\"Calculate the number of parameters in an MLP.\"\"\"",
    indentTwo "return input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim"
  ]

end Export.PyTorch.MLP
