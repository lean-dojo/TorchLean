/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import NN.Spec.Module.Core
public import NN.Spec.Module.Activation -- shake: keep
public import NN.Spec.Module.Linear -- shake: keep
public import NN.Spec.Core.Tensor.Core -- shake: keep

/-!
# Export Core

PyTorch code generation helpers.

This module defines shared string-building utilities used by the PyTorch bridge and round-trip
examples. It emits readable Python `nn.Module` code (optionally with weights embedded) and
centralizes the common prelude used by `NN.Runtime.PyTorch.Export.{MLP,CNN,Transformer}`.

Design note (PyTorch export APIs, for context only):

PyTorch also has *graph capture* / *serialization* mechanisms such as ONNX export and
  `torch.export`.
Those APIs produce IR-like artifacts intended for execution in other runtimes. TorchLean's exporter
in this folder emits auditable Python source for parity checks and round-trip tests.

The public helpers are organized as follows:

- `generatePyTorchImports` / `generatePyTorchSupportDefinitions` provide the shared Python prelude.
- `generateBasePyTorchModule` is the reusable class skeleton for the example exporters.
- `generatePyTorchModule` is the simplest end-to-end exporter for a `Spec.Module.Chain`.
- `NN.Runtime.PyTorch.Export.StateDict` is the general checkpoint-to-JSON adapter for users who
  already have PyTorch weights.

## References

- PyTorch ONNX export: https://pytorch.org/docs/stable/onnx.html
- PyTorch `torch.export`: https://pytorch.org/docs/stable/export.html
-/

@[expose] public section


namespace Export
namespace PyTorch

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Spec.Module
open Spec.Module.Chain

/-- Join an array of lines with newline separators. -/
def joinLines (xs : Array String) : String := String.intercalate "\n" xs.toList
/-- Render a Lean `Bool` as the corresponding Python literal. -/
def pyBool (b : Bool) : String :=
  if b then "True" else "False"
/-- Indent a line by `n` spaces. -/
def indent (n : Nat) (s : String) : String :=
  -- Use a computable definition (Lean's `String.replicate` is `meta` in some imports).
  String.ofList (List.replicate n ' ') ++ s
/-- Indent a line by 2 spaces (common for Python). -/
def indentTwo (s : String) : String := indent 2 s
/-- Indent a line by 4 spaces (common for Python block bodies). -/
def indentFour (s : String) : String := indent 4 s
/-- Indent a line by 6 spaces (used in nested Python blocks). -/
def indentSix (s : String) : String := indent 6 s
/-- Indent a line by 8 spaces (used for nested `nn.Sequential` strings). -/
def indentEight (s : String) : String := indent 8 s

/-!
## Common boilerplate fragments

Many exporters emit the same small pieces of Python: `@property` metadata and a `get_model_info`
dictionary. Shared boilerplate keeps the hand-written example exporters
and the more general IR exporter.
-/

/--
Emit a standard `get_model_info` method used by most TorchLean PyTorch example modules.

`extraFields` are inserted after the `"model_name"` entry. Each element is `(key, valueExpr)` where
`valueExpr` is emitted verbatim as Python code (e.g. `"self.input_shape"` or `"self.hidden_dim"`).
This is meant as a *formatting helper* only; it does not validate Python syntax.
-/
def generateGetModelInfoMethodLines (modelName : String)
    (extraFields : Array (String × String) := #[]) : Array String :=
  let items :=
    #[("model_name", s!"\"{modelName}\"")] ++ extraFields ++
      #[ ("input_shape", "self.input_shape")
       , ("output_shape", "self.output_shape")
       , ("layer_count", "self.layer_count")
       , ("operation_types", "self.operation_types")
       ]
  let rendered := items.mapIdx fun i (k, v) =>
    let comma := if i + 1 < items.size then "," else ""
    indentSix s!"\"{k}\": {v}{comma}"
  #[ indentTwo "def get_model_info(self) -> dict:"
  , indentFour "return {"
  ] ++ rendered ++
  #[indentFour "}"]

/--
Render a `Shape` as a Python tuple literal.

Examples:
- `.scalar` becomes `"()"`,
- a 1D shape becomes `"(n,)"` (note the trailing comma),
- higher-rank shapes become `"(d0, d1, ...)"`.
-/
def shapeToPyTupleString (s : Shape) : String :=
  let dims := Shape.toList s
  match dims with
  | [] => "()"
  | [n] => s!"({n},)"
  | _ => "(" ++ String.intercalate ", " (dims.map (fun n => toString n)) ++ ")"

/-- Count the number of primitive layers in a `Spec.Module.Chain`. -/
def countLayers {α : Type} {s t : Shape} : Spec.Module.Chain α s t → Nat
| .single _ => 1
| .comp a b => countLayers a + countLayers b

/-- Render a Python float expression preserving every finite binary64 value and signed zero.

Short decimal strings are retained only when they parse back to the original bits. Otherwise the
expression uses Python's built-in `float.fromhex` with the exact integer significand and binary
exponent. Infinities and NaN use explicit Python constructors; NaN payload bits are not serialized.
-/
def floatToPyString (value : Float) : String :=
  let bits := value.toBits
  let negative := bits >>> 63 != 0
  if value.isNaN then
    "float('nan')"
  else if value.isInf then
    if negative then "float('-inf')" else "float('inf')"
  else if value == 0 then
    if negative then "-0.0" else "0.0"
  else
    let decimal := toString value
    let parsed := (Lean.Json.parse decimal).bind Lean.Json.getNum?
    if decimal.length ≤ 24 &&
        (match parsed with
        | .ok number => number.toFloat.toBits == bits
        | .error _ => false) then
      decimal
    else
      let fraction := (bits &&& 0x000fffffffffffff).toNat
      let biasedExponent := ((bits >>> 52) &&& 0x7ff).toNat
      let significand := if biasedExponent = 0 then fraction else 2^52 + fraction
      let exponent : Int := if biasedExponent = 0 then -1074 else (biasedExponent : Int) - 1075
      let hex := String.ofList (Nat.toDigits 16 significand)
      let sign := if negative then "-" else ""
      s!"float.fromhex('{sign}0x{hex}p{exponent}')"

/--
Convert a float tensor to a Python list literal without rounding its binary64 elements.

This is a simple recursive printer used for examples and small regression tests; it is not intended
to be fast.
-/
def tensorToPyString {s : Shape} (t : Tensor Float s) : String :=
  match s with
  | .scalar => floatToPyString (item t)
  | .dim n _ =>
      let elems := (List.finRange n).map (fun i =>
        tensorToPyString (Tensor.unstack t i))
      s!"[" ++ String.intercalate ", " elems ++ "]"

/--
Render the transpose of a 2D float tensor as a Python nested-list literal.

TorchLean's matrix-valued specs often follow the mathematical convention where a feature matrix
`W` has shape `(in, out)` and is applied as `X * W`. PyTorch stores `nn.Linear` weights as
`(out, in)` and applies them as `X @ W.T + b`. This helper prints a TorchLean matrix in the
transposed orientation expected by PyTorch.
-/
def transposedMatrixTensorToPy {rows cols : Nat} (t : Tensor Float [rows, cols]) : String :=
  let colToStr (j : Fin cols) : String :=
    let elems := (List.finRange rows).map (fun i =>
      floatToPyString (TorchLean.Tensor.get2 t i j))
    s!"[" ++ String.intercalate ", " elems ++ "]"
  let colsStr := (List.finRange cols).map colToStr
  s!"[" ++ String.intercalate ", " colsStr ++ "]"

/-- Standard imports used by the generated Python snippets. -/
def generatePyTorchImports : String :=
  joinLines #[
    "import torch",
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "import numpy as np",
    "from typing import Optional, Tuple, List"
  ]

/--
Small helper modules used by some `pythonExpr` strings in the Lean specs.

These are small, dependency-free Python utilities (selectors, wrappers, a compact attention helper)
used so the generated model classes stay short and readable.
-/
def generatePyTorchSupportDefinitions : String :=
  joinLines #[
    "",
    "class SelectLast(nn.Module):",
    indentTwo "\"\"\"Select the last timestep from a (batch, seq, hidden) tensor.\"\"\"",
    indentTwo "def forward(self, x):",
    indentFour "return x[:, -1, :]",
    "",
    "class SelectLeading(nn.Module):",
    indentTwo ("\"\"\"Select one position from the leading model dimension after the batch axis."
      ++ "\"\"\""),
    indentTwo "def __init__(self, index: int):",
    indentFour "super().__init__()",
    indentFour "self.index = index",
    indentTwo "def forward(self, x):",
    indentFour "return x[:, self.index, ...]",
    "",
    "class RNNOnlyOutput(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, **kwargs):",
    indentFour "super().__init__()",
    indentFour "self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, **kwargs)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.rnn(x)",
    indentFour "return y",
    "",
    "class GRUOnlyOutput(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, **kwargs):",
    indentFour "super().__init__()",
    indentFour "self.gru = nn.GRU(input_size, hidden_size, batch_first=True, **kwargs)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.gru(x)",
    indentFour "return y",
    "",
    "class LSTMOnlyOutput(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, **kwargs):",
    indentFour "super().__init__()",
    indentFour "self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, **kwargs)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.lstm(x)",
    indentFour "return y",
    "",
    "class RNNClassifier(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, num_classes: int):",
    indentFour "super().__init__()",
    indentFour "self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)",
    indentFour "self.select = SelectLast()",
    indentFour "self.fc = nn.Linear(hidden_size, num_classes)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.rnn(x)",
    indentFour "y = self.select(y)",
    indentFour "return self.fc(y)",
    "",
    "class GRUClassifier(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, num_classes: int):",
    indentFour "super().__init__()",
    indentFour "self.gru = nn.GRU(input_size, hidden_size, batch_first=True)",
    indentFour "self.select = SelectLast()",
    indentFour "self.fc = nn.Linear(hidden_size, num_classes)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.gru(x)",
    indentFour "y = self.select(y)",
    indentFour "return self.fc(y)",
    "",
    "class LSTMClassifier(nn.Module):",
    indentTwo "def __init__(self, input_size: int, hidden_size: int, num_classes: int):",
    indentFour "super().__init__()",
    indentFour "self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)",
    indentFour "self.select = SelectLast()",
    indentFour "self.fc = nn.Linear(hidden_size, num_classes)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.lstm(x)",
    indentFour "y = self.select(y)",
    indentFour "return self.fc(y)",
    "",
    "class UnsupportedLayer(nn.Module):",
    indentTwo "def __init__(self, kind: str, detail: str = \"\"):",
    indentFour "super().__init__()",
    indentFour "self.kind = kind",
    indentFour "self.detail = detail",
    indentTwo "def forward(self, x):",
    indentFour "raise NotImplementedError(f\"Unsupported layer: {self.kind} ({self.detail})\")",
    "",
    "class ScaledDotProductSelfAttention(nn.Module):",
    indentTwo "def __init__(self, d_model: int):",
    indentFour "super().__init__()",
    indentFour "self.d_model = d_model",
    indentTwo "def forward(self, x):",
    indentFour "# x: (batch, seq, d_model)",
    indentFour "scores = torch.matmul(x, x.transpose(-2, -1)) / (self.d_model ** 0.5)",
    indentFour "attn = torch.softmax(scores, dim=-1)",
    indentFour "return torch.matmul(attn, x)",
    "",
    "class SimpleRNN(nn.Module):",
    indentTwo ("def __init__(self, input_size: int, hidden_size: int, output_size: " ++
      "int, bidirectional: bool = False):"),
    indentFour "super().__init__()",
    indentFour
      "self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)",
    indentFour "out_dim = hidden_size * (2 if bidirectional else 1)",
    indentFour "self.fc = nn.Linear(out_dim, output_size)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.rnn(x)",
    indentFour "return self.fc(y)",
    "",
    "class SimpleGRU(nn.Module):",
    indentTwo ("def __init__(self, input_size: int, hidden_size: int, output_size: " ++
      "int, bidirectional: bool = False):"),
    indentFour "super().__init__()",
    indentFour
      "self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)",
    indentFour "out_dim = hidden_size * (2 if bidirectional else 1)",
    indentFour "self.fc = nn.Linear(out_dim, output_size)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.gru(x)",
    indentFour "return self.fc(y)",
    "",
    "class SimpleLSTM(nn.Module):",
    indentTwo ("def __init__(self, input_size: int, hidden_size: int, output_size: " ++
      "int, bidirectional: bool = False):"),
    indentFour "super().__init__()",
    indentFour
      "self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)",
    indentFour "out_dim = hidden_size * (2 if bidirectional else 1)",
    indentFour "self.fc = nn.Linear(out_dim, output_size)",
    indentTwo "def forward(self, x):",
    indentFour "y, _ = self.lstm(x)",
    indentFour "return self.fc(y)",
    "",
    "class GRULanguageModel(nn.Module):",
    indentTwo "def __init__(self, vocab_size: int, hidden_size: int):",
    indentFour "super().__init__()",
    indentFour "self.embed = nn.Linear(vocab_size, hidden_size)",
    indentFour "self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)",
    indentFour "self.proj = nn.Linear(hidden_size, vocab_size)",
    indentTwo "def forward(self, x):",
    indentFour "x = self.embed(x)",
    indentFour "y, _ = self.gru(x)",
    indentFour "return self.proj(y)",
    "",
    "class Seq2SeqInference(nn.Module):",
    indentTwo ("def __init__(self, src_vocab_size: int, tgt_vocab_size: int, " ++
      "embed_dim: int, hidden_dim: int, max_tgt_len: int, start_token: int = " ++
      "0):"),
    indentFour "super().__init__()",
    indentFour "self.src_embed = nn.Linear(src_vocab_size, embed_dim)",
    indentFour "self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim)",
    indentFour "self.encoder = nn.RNN(embed_dim, hidden_dim, batch_first=True)",
    indentFour "self.decoder = nn.RNN(embed_dim, hidden_dim, batch_first=True)",
    indentFour "self.proj = nn.Linear(hidden_dim, tgt_vocab_size)",
    indentFour "self.max_tgt_len = max_tgt_len",
    indentFour "self.start_token = start_token",
    indentTwo "def forward(self, x):",
    indentFour "# x: (batch, src_len, src_vocab_size) one-hot/dists",
    indentFour "x = self.src_embed(x)",
    indentFour "_enc_out, h = self.encoder(x)",
    indentFour "batch = x.shape[0]",
    indentFour "token = torch.full((batch,), self.start_token, dtype=torch.long, device=x.device)",
    indentFour "inp = self.tgt_embed(token).unsqueeze(1)",
    indentFour "logits = []",
    indentFour "h_dec = h",
    indentFour "for _ in range(self.max_tgt_len):",
    indentSix "y, h_dec = self.decoder(inp, h_dec)",
    indentSix "step = self.proj(y.squeeze(1))",
    indentSix "logits.append(step)",
    indentSix "token = torch.argmax(step, dim=-1)",
    indentSix "inp = self.tgt_embed(token).unsqueeze(1)",
    indentFour "return torch.stack(logits, dim=1)",
  ]

/--
Generate a generic base `nn.Module` class skeleton.

This is used by exporters that want a "real" class with an explicit `_initialize_layers` hook,
instead of the simpler `nn.Sequential` emitter.
-/
def generateBasePyTorchModule (className : String) (docstring : String) : String :=
  joinLines <|
    #[ s!"class {className}(nn.Module):"
    , indentTwo s!"\"\"\"{docstring}\"\"\""
    , indentTwo ""
    , indentTwo "def __init__(self):"
    , indentFour "super().__init__()"
    , indentFour "self._initialize_layers()"
    , indentFour ""
    , indentFour "def _initialize_layers(self):"
    , indentSix "raise NotImplementedError(\"Subclasses must implement _initialize_layers\")"
    , indentFour ""
    , indentTwo "def forward(self, x):"
    , indentFour "raise NotImplementedError(\"Subclasses must implement forward\")"
    , indentFour ""
    ] ++
      generateGetModelInfoMethodLines className ++
      #[ indentFour ""
      , indentTwo "@property"
      , indentTwo "def input_shape(self):"
      , indentFour "raise NotImplementedError(\"Subclasses must implement input_shape\")"
      , indentFour ""
      , indentTwo "@property"
      , indentTwo "def output_shape(self):"
      , indentFour "raise NotImplementedError(\"Subclasses must implement output_shape\")"
      , indentFour ""
      , indentTwo "@property"
      , indentTwo "def layer_count(self):"
      , indentFour "raise NotImplementedError(\"Subclasses must implement layer_count\")"
      , indentFour ""
      , indentTwo "@property"
      , indentTwo "def operation_types(self):"
      , indentFour "raise NotImplementedError(\"Subclasses must implement operation_types\")"
      ]

/-- Emit Python helpers for saving/loading state dictionaries and JSON checkpoints. -/
def generateWeightLoadingUtils : String :=
  joinLines #[
    "def load_weights_from_dict(model: nn.Module, state_dict: dict):",
    indentTwo "\"\"\"Load weights from a state dictionary into the model.\"\"\"",
    indentTwo "model.load_state_dict(state_dict)",
    indentTwo "return model",
    "",
    "def save_weights_to_dict(model: nn.Module) -> dict:",
    indentTwo "\"\"\"Save model weights to a state dictionary.\"\"\"",
    indentTwo "return model.state_dict()",
    "",
    "def save_model_to_file(model: nn.Module, filepath: str):",
    indentTwo "\"\"\"Save model weights to a file as a state_dict checkpoint.\"\"\"",
    indentTwo "torch.save(model.state_dict(), filepath)",
    "",
    "def load_model_from_file(filepath: str, model: Optional[nn.Module] = None):",
    indentTwo "\"\"\"Load a state_dict checkpoint; optionally materialize it into `model`.\"\"\"",
    indentTwo "state_dict = torch.load(filepath, weights_only=True)",
    indentTwo "if model is None:",
    indentTwo "    return state_dict",
    indentTwo "model.load_state_dict(state_dict)",
    indentTwo "return model"
  ]

/-- Emit Python helpers for validating exported models. -/
def generateTestingUtils : String :=
  joinLines #[
    "def test_model_forward(model: nn.Module, input_shape: Tuple[int, ...], num_tests: int = 5):",
    indentTwo "\"\"\"Test model forward pass with random inputs.\"\"\"",
    indentTwo "model.eval()",
    indentTwo "with torch.no_grad():",
    indentFour "for i in range(num_tests):",
    indentSix "x = torch.randn(1, *input_shape)",
    indentSix "y = model(x)",
    indentSix "print(f\"Test {i+1}: Input shape: {x.shape}, Output shape: {y.shape}\")",
    indentSix "print(f\"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]\")",
    "",
    "def count_parameters(model: nn.Module) -> int:",
    indentTwo "\"\"\"Count the number of trainable parameters in the model.\"\"\"",
    indentTwo "return sum(p.numel() for p in model.parameters() if p.requires_grad)",
    "",
    "def print_model_summary(model: nn.Module):",
    indentTwo "\"\"\"Print a summary of the model architecture.\"\"\"",
    indentTwo "print(f\"Model: {model.__class__.__name__}\")",
    indentTwo "print(f\"Total parameters: {count_parameters(model):,}\")",
    indentTwo "print(f\"Model info: {model.get_model_info()}\")"
  ]

/--
Generate a complete `nn.Sequential`-based Python module for a `Spec.Module.Chain`.

This is the simplest exporter: we extract an array of `(opName, pythonLayerString)` pairs and drop
them into an `nn.Sequential(...)` in a new class.
-/
def generatePyTorchModule {α : Type} {s t : Shape}
  (chain : Spec.Module.Chain α s t) (className : String := "ExportedModel") : String :=
  let inputShape := shapeToPyTupleString s
  let outputShape := shapeToPyTupleString t
  let layerCount := countLayers chain
  let layers := Spec.Module.Chain.layerInfo chain
  let layerStrings := layers.map (fun (_, pytorch) => indentEight pytorch)
  let opList :=
    "[" ++ String.intercalate ", " (layers.map (fun (op, _) => s!"\"{op}\"")).toList ++ "]"

  joinLines <|
    #[ generatePyTorchImports
    , generatePyTorchSupportDefinitions
    , ""
    , s!"class {className}(nn.Module):"
    , indentTwo "def __init__(self):"
    , indentFour "super().__init__()"
    , indentFour s!"# Input shape: {inputShape}"
    , indentFour s!"# Output shape: {outputShape}"
    , indentFour s!"# Layer count: {layerCount}"
    , indentFour s!"# Operations: {String.intercalate ", " (layers.map (fun (op, _) => op)).toList}"
    , indentFour ""
    , indentFour "self.layers = nn.Sequential("
    , String.intercalate ",\n" layerStrings.toList
    , indentFour ")"
    , ""
    , indentTwo "def forward(self, x):"
    , indentFour "return self.layers(x)"
    , indentFour ""
    , indentTwo "@property"
    , indentTwo "def input_shape(self):"
    , indentFour s!"return {inputShape}"
    , indentFour ""
    , indentTwo "@property"
    , indentTwo "def output_shape(self):"
    , indentFour s!"return {outputShape}"
    , indentFour ""
    , indentTwo "@property"
    , indentTwo "def layer_count(self):"
    , indentFour s!"return {layerCount}"
    , indentFour ""
    , indentTwo "@property"
    , indentTwo "def operation_types(self):"
    , indentFour s!"return {opList}"
    , indentFour ""
    ] ++
      generateGetModelInfoMethodLines className

end PyTorch
end Export
