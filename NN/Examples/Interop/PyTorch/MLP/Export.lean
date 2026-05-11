/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Spec.Models.Mlp

/-!
# MLP PyTorch Fixture Export

PyTorch code generator for the small MLP round-trip fixture.

The generated Python mirrors the common `nn.Linear → ReLU → nn.Linear` pattern. We also support
embedding explicit weights into a `state_dict`-shaped dictionary, which is handy for round-trips
and small regression tests.
-/

@[expose] public section


namespace Export
namespace MLPPyTorch

open Spec
open Tensor
open ModSpec
open SpecChain
open Examples
open Export.PyTorch

/-- How to name `state_dict` keys when exporting weights. -/
inductive WeightKeyStyle where
  /-- Keys like `fc1.weight` / `fc2.bias` (matches PyTorch `nn.Linear` modules). -/
  | linear
  /-- Keys like `layers.0.weight` / `layers.2.bias` (common when exporting `nn.Sequential`). -/
  | sequential
  deriving DecidableEq, Repr

/-- Key name for the first layer's weight tensor in a PyTorch `state_dict`. -/
def w1Key : WeightKeyStyle → String
  | .linear => "fc1.weight"
  | .sequential => "layers.0.weight"

/-- Key name for the first layer's bias tensor in a PyTorch `state_dict`. -/
def b1Key : WeightKeyStyle → String
  | .linear => "fc1.bias"
  | .sequential => "layers.0.bias"

/-- Key name for the second layer's weight tensor in a PyTorch `state_dict`. -/
def w2Key : WeightKeyStyle → String
  | .linear => "fc2.weight"
  | .sequential => "layers.2.weight"

/-- Key name for the second layer's bias tensor in a PyTorch `state_dict`. -/
def b2Key : WeightKeyStyle → String
  | .linear => "fc2.bias"
  | .sequential => "layers.2.bias"

/--
Metadata for an exported MLP.

This is a small “record of facts” about the generated Python: shapes, dimensions, and optionally
embedded weights for round-trip tests.
-/
structure MLPExportMetadata (α : Type) (inDim hidDim outDim : Nat) where
  /-- model Name. -/
  modelName : String
  /-- input Dim. -/
  inputDim : Nat
  /-- hidden Dim. -/
  hiddenDim : Nat
  /-- output Dim. -/
  outputDim : Nat
  /-- has Weights. -/
  hasWeights : Bool
  /-- weights. -/
  weights : Option (Tensor Float (.dim hidDim (.dim inDim .scalar)) ×
                   Tensor Float (.dim hidDim .scalar) ×
                   Tensor Float (.dim outDim (.dim hidDim .scalar)) ×
                   Tensor Float (.dim outDim .scalar))

/--
Emit the Python class body for a basic `Linear → ReLU → Linear` MLP.

This returns *lines* (not a single string) so callers can splice it into larger scripts.
-/
def generateMLPPyTorchClassLines (inDim hidDim outDim : Nat) (className : String) : List String :=
  [
    s!"class {className}(nn.Module):",
    (s!"\"\"\"Multi-Layer Perceptron with {inDim} input, {hidDim} hidden, " ++
      s!"{outDim} output dimensions\"\"\""),
    indent2 "",
    (s!"def __init__(self, input_dim: int = {inDim}, hidden_dim: int = " ++
      s!"{hidDim}, output_dim: int = {outDim}):"),
    indent4 "super().__init__()",
    indent4 "self.input_dim = input_dim",
    indent4 "self.hidden_dim = hidden_dim",
    indent4 "self.output_dim = output_dim",
    indent4 "",
    indent4 "# Define layers",
    indent4 "self.fc1 = nn.Linear(input_dim, hidden_dim)",
    indent4 "self.relu = nn.ReLU()",
    indent4 "self.fc2 = nn.Linear(hidden_dim, output_dim)",
    indent4 "",
    indent2 "def forward(self, x):",
    indent4 "x = self.fc1(x)",
    indent4 "x = self.relu(x)",
    indent4 "x = self.fc2(x)",
    indent4 "return x",
    indent4 "",
    indent2 "@property",
    indent2 "def input_shape(self):",
    indent4 s!"return ({inDim},)",
    indent4 "",
    indent2 "@property",
    indent2 "def output_shape(self):",
    indent4 s!"return ({outDim},)",
    indent4 "",
    indent2 "@property",
    indent2 "def layer_count(self):",
    indent4 "return 3",  -- fc1, relu, fc2
    indent4 "",
    indent2 "@property",
    indent2 "def operation_types(self):",
    indent4 "return [\"Linear\", \"ReLU\", \"Linear\"]",
    indent4 ""
  ] ++
    generateGetModelInfoMethodLines className
      [ ("input_dim", "self.input_dim")
      , ("hidden_dim", "self.hidden_dim")
      , ("output_dim", "self.output_dim")
      ]

/-- Generate a standalone Python file containing an `nn.Module` MLP class. -/
def generateMLPPyTorchClass (inDim hidDim outDim : Nat) (className : String := "MLP") : String :=
  joinLines <|
    [generatePyTorchImports, ""] ++ generateMLPPyTorchClassLines inDim hidDim outDim className

/--
Generate Python code for an MLP plus helper functions that embed concrete weights.

The output contains a `get_mlp_state_dict` function that returns a PyTorch-shaped dictionary
(`state_dict`) and a `load_mlp_weights` helper that calls `model.load_state_dict(...)`.
-/
def generateMLPWithWeights {inDim hidDim outDim : Nat}
  (w1 : Tensor Float (.dim hidDim (.dim inDim .scalar)))
  (b1 : Tensor Float (.dim hidDim .scalar))
  (w2 : Tensor Float (.dim outDim (.dim hidDim .scalar)))
  (b2 : Tensor Float (.dim outDim .scalar))
  (className : String := "MLP")
  (keyStyle : WeightKeyStyle := .linear) : String :=
  joinLines [
    generateMLPPyTorchClass inDim hidDim outDim className,
    "",
    "# Weight initialization functions",
    "def get_mlp_state_dict():",
    indent2 "state_dict = {}",
    indent2 s!"state_dict['{w1Key keyStyle}'] = torch.tensor({tensor2DToPy w1})",
    indent2 s!"state_dict['{b1Key keyStyle}'] = torch.tensor({tensor1DToPy b1})",
    indent2 s!"state_dict['{w2Key keyStyle}'] = torch.tensor({tensor2DToPy w2})",
    indent2 s!"state_dict['{b2Key keyStyle}'] = torch.tensor({tensor1DToPy b2})",
    indent2 "return state_dict",
    indent2 "",
    "def load_mlp_weights(model):",
    indent2 "state_dict = get_mlp_state_dict()",
    indent2 "model.load_state_dict(state_dict)",
    indent2 "return model",
    indent2 "",
    "# Usage example",
    "if __name__ == \"__main__\":",
    indent2 s!"model = {className}()",
    indent2 "model = load_mlp_weights(model)",
    indent2 s!"x = torch.randn(1, {inDim})  # batch_size=1, features={inDim}",
    indent2 "y = model(x)",
    indent2 "print(f\"Input shape: {x.shape}\")",
    indent2 "print(f\"Output shape: {y.shape}\")",
    indent2 "print(f\"Output: {y}\")",
    indent2 "print(f\"Model info: {model.get_model_info()}\")"
  ]

/--
Generate a Python MLP class that ends with a Softmax (classification convenience).

This mirrors common compact PyTorch model code; full training pipelines usually use logits + a combined
loss (e.g. `CrossEntropyLoss`) instead of an explicit softmax.
-/
def generateMLPWithSoftmax {inDim hidDim outDim : Nat} (className : String := "MLPWithSoftmax") :
  String :=
  joinLines <|
  [generatePyTorchImports, ""] ++ [
    s!"class {className}(nn.Module):",
    indent2 s!"\"\"\"Multi-Layer Perceptron with softmax output for classification\"\"\"",
    indent2 "",
    (s!"def __init__(self, input_dim: int = {inDim}, hidden_dim: int = " ++
      s!"{hidDim}, output_dim: int = {outDim}):"),
    indent4 "super().__init__()",
    indent4 "self.input_dim = input_dim",
    indent4 "self.hidden_dim = hidden_dim",
    indent4 "self.output_dim = output_dim",
    indent4 "",
    indent4 "# Define layers",
    indent4 "self.fc1 = nn.Linear(input_dim, hidden_dim)",
    indent4 "self.relu = nn.ReLU()",
    indent4 "self.fc2 = nn.Linear(hidden_dim, output_dim)",
    indent4 "self.softmax = nn.Softmax(dim=1)",
    indent4 "",
    indent2 "def forward(self, x):",
    indent4 "x = self.fc1(x)",
    indent4 "x = self.relu(x)",
    indent4 "x = self.fc2(x)",
    indent4 "x = self.softmax(x)",
    indent4 "return x",
    indent4 "",
    indent2 "@property",
    indent2 "def input_shape(self):",
    indent4 s!"return ({inDim},)",
    indent4 "",
    indent2 "@property",
    indent2 "def output_shape(self):",
    indent4 s!"return ({outDim},)",
    indent4 "",
    indent2 "@property",
    indent2 "def layer_count(self):",
    indent4 "return 4",  -- fc1, relu, fc2, softmax
    indent4 "",
    indent2 "@property",
    indent2 "def operation_types(self):",
    indent4 "return [\"Linear\", \"ReLU\", \"Linear\", \"Softmax\"]"
  ]

/-- Line-based version of `generateMLPWithSoftmax` for script composition. -/
def generateMLPWithSoftmaxLines {inDim hidDim outDim : Nat} (className : String) : List String :=
  [
    s!"class {className}(nn.Module):",
    indent2 s!"\"\"\"Multi-Layer Perceptron with softmax output for classification\"\"\"",
    indent2 "",
    (s!"def __init__(self, input_dim: int = {inDim}, hidden_dim: int = " ++
      s!"{hidDim}, output_dim: int = {outDim}):"),
    indent4 "super().__init__()",
    indent4 "self.input_dim = input_dim",
    indent4 "self.hidden_dim = hidden_dim",
    indent4 "self.output_dim = output_dim",
    indent4 "",
    indent4 "# Define layers",
    indent4 "self.fc1 = nn.Linear(input_dim, hidden_dim)",
    indent4 "self.relu = nn.ReLU()",
    indent4 "self.fc2 = nn.Linear(hidden_dim, output_dim)",
    indent4 "self.softmax = nn.Softmax(dim=1)",
    indent4 "",
    indent2 "def forward(self, x):",
    indent4 "x = self.fc1(x)",
    indent4 "x = self.relu(x)",
    indent4 "x = self.fc2(x)",
    indent4 "x = self.softmax(x)",
    indent4 "return x",
    indent4 "",
    indent2 "@property",
    indent2 "def input_shape(self):",
    indent4 s!"return ({inDim},)",
    indent4 "",
    indent2 "@property",
    indent2 "def output_shape(self):",
    indent4 s!"return ({outDim},)",
    indent4 "",
    indent2 "@property",
    indent2 "def layer_count(self):",
    indent4 "return 4",  -- fc1, relu, fc2, softmax
    indent4 "",
    indent2 "@property",
    indent2 "def operation_types(self):",
    indent4 "return [\"Linear\", \"ReLU\", \"Linear\", \"Softmax\"]"
  ]

-- Export MLP from SpecChain
/--
Export metadata for an MLP described as a `SpecChain`.

The chain is accepted to keep the API aligned with `SpecChain`, while this exporter produces
metadata from the explicit dimensions supplied by the type parameters.
-/
def exportMLPFromSpecChain {α : Type} {inDim hidDim outDim : Nat}
  (_chain : SpecChain α (.dim inDim .scalar) (.dim outDim .scalar))
  (className : String := "ExportedMLP") : MLPExportMetadata α inDim hidDim outDim :=
  {
    modelName := className,
    inputDim := inDim,
    hiddenDim := hidDim,
    outputDim := outDim,
    hasWeights := false,
    weights := none
  }

-- Export MLP with weights
/-- Like `exportMLPFromSpecChain`, but include explicit weights in the metadata record. -/
def exportMLPWithWeights {α : Type} {inDim hidDim outDim : Nat}
  (_chain : SpecChain α (.dim inDim .scalar) (.dim outDim .scalar))
  (w1 : Tensor Float (.dim hidDim (.dim inDim .scalar)))
  (b1 : Tensor Float (.dim hidDim .scalar))
  (w2 : Tensor Float (.dim outDim (.dim hidDim .scalar)))
  (b2 : Tensor Float (.dim outDim .scalar))
  (className : String := "ExportedMLP") : MLPExportMetadata α inDim hidDim outDim :=
  {
    modelName := className,
    inputDim := inDim,
    hiddenDim := hidDim,
    outputDim := outDim,
    hasWeights := true,
    weights := some (w1, b1, w2, b2)
  }

-- Generate complete MLP export
/--
Generate a complete Python script for MLP demos.

This includes:
- a base MLP class,
- a Softmax variant,
- shared helper modules from `NN/Runtime/PyTorch/Export/Core.lean`,
- and a couple compact convenience helpers (constructor + parameter count).
-/
def generateCompleteMLPExport {inDim hidDim outDim : Nat}
  (className : String := "MLP") : String :=
  joinLines [
    generatePyTorchImports,
    "",
    joinLines (generateMLPPyTorchClassLines inDim hidDim outDim className),
    "",
    joinLines (generateMLPWithSoftmaxLines (inDim := inDim) (hidDim := hidDim) (outDim := outDim)
      s!"{className}WithSoftmax"),
    "",
    generateWeightLoadingUtils,
    "",
    generateTestingUtils,
    "",
    "# MLP-specific utilities",
    ("def create_mlp_from_spec(input_dim: int, hidden_dim: int, output_dim: " ++
      "int, use_softmax: bool = False):"),
    indent2 "\"\"\"Create an MLP model from specifications.\"\"\"",
    indent2 "if use_softmax:",
      indent4 s!"return {className}WithSoftmax(input_dim, hidden_dim, output_dim)",
    indent2 "else:",
      indent4 s!"return {className}(input_dim, hidden_dim, output_dim)",
    indent2 "",
    "def mlp_parameter_count(input_dim: int, hidden_dim: int, output_dim: int) -> int:",
    indent2 "\"\"\"Calculate the number of parameters in an MLP.\"\"\"",
    indent2 "return input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim"
  ]

end MLPPyTorch
end Export
