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

/-- State-dictionary key for a zero-based linear-layer index and parameter field.
Sequential models place a ReLU between linear layers, so their indices are doubled. -/
def key (style : KeyStyle) (layer : Nat) (field : String) : String :=
  match style with
  | .linear => s!"fc{layer + 1}.{field}"
  | .sequential => s!"layers.{2 * layer}.{field}"

/--
Emit the Python class body for a `Linear → ReLU → Linear` MLP, optionally ending in softmax.

This returns *lines* (not a single string) so callers can splice it into larger scripts.
-/
def body
    (inputWidth hiddenWidth outputWidth : Nat) (className : String)
    (softmax : Bool := false) : Array String :=
  #[
    s!"class {className}(nn.Module):",
    indent 2 (if softmax then
      "\"\"\"Multi-Layer Perceptron with softmax output for classification\"\"\""
      else s!"\"\"\"Multi-Layer Perceptron with {inputWidth} input, " ++
        s!"{hiddenWidth} hidden, {outputWidth} output dimensions\"\"\""),
    indent 2 "",
    indent 2 (s!"def __init__(self, input_dim: int = {inputWidth}, hidden_dim: int = " ++
      s!"{hiddenWidth}, output_dim: int = {outputWidth}):"),
    indent 4 "super().__init__()",
    indent 4 "self.input_dim = input_dim",
    indent 4 "self.hidden_dim = hidden_dim",
    indent 4 "self.output_dim = output_dim",
    indent 4 "",
    indent 4 "# Define layers",
    indent 4 "self.fc1 = nn.Linear(input_dim, hidden_dim)",
    indent 4 "self.relu = nn.ReLU()",
    indent 4 "self.fc2 = nn.Linear(hidden_dim, output_dim)"
  ] ++
    (if softmax then #[indent 4 "self.softmax = nn.Softmax(dim=-1)"] else #[]) ++ #[
    indent 4 "",
    indent 2 "def forward(self, x):",
    indent 4 "x = self.fc1(x)",
    indent 4 "x = self.relu(x)",
    indent 4 "x = self.fc2(x)"
  ] ++
    (if softmax then #[indent 4 "x = self.softmax(x)"] else #[]) ++ #[
    indent 4 "return x",
    indent 4 "",
    indent 2 "@property",
    indent 2 "def input_shape(self):",
    indent 4 "return (self.input_dim,)",
    indent 4 "",
    indent 2 "@property",
    indent 2 "def output_shape(self):",
    indent 4 "return (self.output_dim,)",
    indent 4 "",
    indent 2 "@property",
    indent 2 "def layer_count(self):",
    indent 4 (if softmax then "return 4" else "return 3"),
    indent 4 "",
    indent 2 "@property",
    indent 2 "def operation_types(self):",
    indent 4 (if softmax then "return [\"Linear\", \"ReLU\", \"Linear\", \"Softmax\"]"
      else "return [\"Linear\", \"ReLU\", \"Linear\"]")
  ] ++
    (if softmax then #[] else #[indent 4 ""]) ++
    metadata className
      #[ ("input_dim", "self.input_dim")
      , ("hidden_dim", "self.hidden_dim")
      , ("output_dim", "self.output_dim")
      ]

/-- Render a standalone MLP class, optionally applying softmax after its output layer. -/
def source (inputWidth hiddenWidth outputWidth : Nat)
    (className : String := "MLP") (softmax : Bool := false) : String :=
  joinLines <|
    #[imports, ""] ++ body inputWidth hiddenWidth outputWidth className softmax

/--
Generate Python code for an MLP plus helper functions that embed concrete weights.

The output contains a `get_mlp_state_dict` function that returns a PyTorch-shaped dictionary
(`state_dict`). Its `load_mlp_weights` helper normalizes either key convention to the generated
class's `fc1`/`fc2` layers before calling `model.load_state_dict(...)`.
Set `softmax := true` to apply softmax after the final linear layer; parameter keys are unchanged.
-/
def weights {inputWidth hiddenWidth outputWidth : Nat}
    (inputWeight : Tensor Float [hiddenWidth, inputWidth])
    (inputBias : Tensor Float [hiddenWidth])
    (outputWeight : Tensor Float [outputWidth, hiddenWidth])
    (outputBias : Tensor Float [outputWidth])
    (className : String := "MLP")
    (keyStyle : KeyStyle := .linear) (softmax : Bool := false) : String :=
  joinLines #[
    source inputWidth hiddenWidth outputWidth className softmax,
    "",
    "# Weight initialization functions",
    "def get_mlp_state_dict():",
    indent 2 "state_dict = {}",
    indent 2
      s!"state_dict['{key keyStyle 0 "weight"}'] = torch.tensor({tensorLiteral inputWeight})",
    indent 2
      s!"state_dict['{key keyStyle 0 "bias"}'] = torch.tensor({tensorLiteral inputBias})",
    indent 2
      s!"state_dict['{key keyStyle 1 "weight"}'] = torch.tensor({tensorLiteral outputWeight})",
    indent 2
      s!"state_dict['{key keyStyle 1 "bias"}'] = torch.tensor({tensorLiteral outputBias})",
    indent 2 "return state_dict",
    indent 2 "",
    "def load_mlp_weights(model):",
    indent 2 "state_dict = get_mlp_state_dict()",
    indent 2 "# Normalize either export key convention to this class's named layers.",
    indent 2 "state_dict = {",
    indent 4 s!"'fc1.weight': state_dict['{key keyStyle 0 "weight"}'],",
    indent 4 s!"'fc1.bias': state_dict['{key keyStyle 0 "bias"}'],",
    indent 4 s!"'fc2.weight': state_dict['{key keyStyle 1 "weight"}'],",
    indent 4 s!"'fc2.bias': state_dict['{key keyStyle 1 "bias"}'],",
    indent 2 "}",
    indent 2 "model.load_state_dict(state_dict)",
    indent 2 "return model",
    indent 2 "",
    "# Usage example",
    "if __name__ == \"__main__\":",
    indent 2 s!"model = {className}()",
    indent 2 "model = load_mlp_weights(model)",
    indent 2
      s!"x = torch.randn(1, {inputWidth})  # batch_size=1, features={inputWidth}",
    indent 2 "y = model(x)",
    indent 2 "print(f\"Input shape: {x.shape}\")",
    indent 2 "print(f\"Output shape: {y.shape}\")",
    indent 2 "print(f\"Output: {y}\")",
    indent 2 "print(f\"Model info: {model.get_model_info()}\")"
  ]

/--
Generate a complete Python script for MLP examples.

This includes:
- a base MLP class,
- a Softmax variant,
- shared checkpoint and test utilities,
- and convenience helpers for construction and parameter counting.
-/
def script {inputWidth hiddenWidth outputWidth : Nat}
    (className : String := "MLP") : String :=
  joinLines #[
    imports,
    "",
    joinLines (body inputWidth hiddenWidth outputWidth className),
    "",
    joinLines (body inputWidth hiddenWidth outputWidth
      s!"{className}WithSoftmax" (softmax := true)),
    "",
    checkpoints,
    "",
    checks,
    "",
    "# MLP-specific utilities",
    ("def create_mlp_from_spec(input_dim: int, hidden_dim: int, output_dim: " ++
      "int, use_softmax: bool = False):"),
    indent 2 "\"\"\"Create an MLP model from specifications.\"\"\"",
    indent 2 "if use_softmax:",
      indent 4 s!"return {className}WithSoftmax(input_dim, hidden_dim, output_dim)",
    indent 2 "else:",
      indent 4 s!"return {className}(input_dim, hidden_dim, output_dim)",
    indent 2 "",
    "def mlp_parameter_count(input_dim: int, hidden_dim: int, output_dim: int) -> int:",
    indent 2 "\"\"\"Calculate the number of parameters in an MLP.\"\"\"",
    indent 2 "return input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim"
  ]

end Export.PyTorch.MLP
