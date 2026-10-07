/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor
public import NN.Spec.Module.Core
public import NN.Tensor.Conversion

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

- `imports` / `support` provide the shared Python prelude.
- `base` is the reusable class skeleton for the example exporters.
- `emit` is the simplest end-to-end exporter for a `Spec.Module.Chain`.
- `checkpoints` emits Python state-dictionary save/load helpers. Callers supply
  any checkpoint-to-JSON conversion needed by the import path.

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
def boolLiteral (b : Bool) : String :=
  if b then "True" else "False"
/-- Indent a line by `n` spaces. -/
def indent (n : Nat) (s : String) : String :=
  -- Use a computable definition (Lean's `String.replicate` is `meta` in some imports).
  String.ofList (List.replicate n ' ') ++ s

/-!
## Common boilerplate fragments

Many exporters emit the same small pieces of Python: `@property` metadata and a `get_model_info`
dictionary. These fragments are shared by the example exporters and the general IR exporter.
-/

/--
Emit a standard `get_model_info` method used by most TorchLean PyTorch example modules.

`extraFields` are inserted after the `"model_name"` entry. Each element is `(key, valueExpr)` where
`valueExpr` is emitted verbatim as Python code (e.g. `"self.input_shape"` or `"self.hidden_dim"`).
This is meant as a *formatting helper* only; it does not validate Python syntax.
-/
def metadata (modelName : String)
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
    indent 6 s!"\"{k}\": {v}{comma}"
  #[ indent 2 "def get_model_info(self) -> dict:"
  , indent 4 "return {"
  ] ++ rendered ++
  #[indent 4 "}"]

/--
Render a `Shape` as a Python tuple literal.

Examples:
- `.scalar` becomes `"()"`,
- a 1D shape becomes `"(n,)"` (note the trailing comma),
- higher-rank shapes become `"(d0, d1, ...)"`.
-/
def shapeLiteral (s : Shape) : String :=
  let dims := Shape.toList s
  match dims with
  | [] => "()"
  | [n] => s!"({n},)"
  | _ => "(" ++ String.intercalate ", " (dims.map (fun n => toString n)) ++ ")"

/-- Render tensor-valued dimensions as a Python tuple, including the empty and singleton cases. -/
def tupleLiteral {n : Nat} (dimensions : Tensor Nat [n]) : String :=
  shapeLiteral (dimensions.to Shape)

/-- Render a Python float expression preserving every finite binary64 value and signed zero.

Short decimal strings are retained only when they parse back to the original bits. Otherwise the
expression uses Python's built-in `float.fromhex` with the exact integer significand and binary
exponent. Infinities and NaN use explicit Python constructors; NaN payload bits are not serialized.
-/
def floatLiteral (value : Float) : String :=
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
def tensorLiteral {s : Shape} (t : Tensor Float s) : String :=
  match s with
  | .scalar => floatLiteral (item t)
  | .dim n _ =>
      let elems := (List.finRange n).map (fun i =>
        tensorLiteral (Tensor.unstack t i))
      s!"[" ++ String.intercalate ", " elems ++ "]"

/-- Standard imports used by the generated Python snippets. -/
def imports : String :=
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
def support : String :=
  joinLines #[
    "",
    "class SelectLast(nn.Module):",
    indent 2 "\"\"\"Select the last timestep from a (batch, seq, hidden) tensor.\"\"\"",
    indent 2 "def forward(self, x):",
    indent 4 "return x[:, -1, :]",
    "",
    "class SelectLeading(nn.Module):",
    indent 2 ("\"\"\"Select one position from the leading model dimension after the batch axis."
      ++ "\"\"\""),
    indent 2 "def __init__(self, index: int):",
    indent 4 "super().__init__()",
    indent 4 "self.index = index",
    indent 2 "def forward(self, x):",
    indent 4 "return x[:, self.index, ...]",
    "",
    "class RNNOnlyOutput(nn.Module):",
    indent 2 "def __init__(self, input_size: int, hidden_size: int, **kwargs):",
    indent 4 "super().__init__()",
    indent 4 "self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, **kwargs)",
    indent 2 "def forward(self, x):",
    indent 4 "y, _ = self.rnn(x)",
    indent 4 "return y",
    "",
    "class GRUOnlyOutput(nn.Module):",
    indent 2 "def __init__(self, input_size: int, hidden_size: int, **kwargs):",
    indent 4 "super().__init__()",
    indent 4 "self.gru = nn.GRU(input_size, hidden_size, batch_first=True, **kwargs)",
    indent 2 "def forward(self, x):",
    indent 4 "y, _ = self.gru(x)",
    indent 4 "return y",
    "",
    "class LSTMOnlyOutput(nn.Module):",
    indent 2 "def __init__(self, input_size: int, hidden_size: int, **kwargs):",
    indent 4 "super().__init__()",
    indent 4 "self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, **kwargs)",
    indent 2 "def forward(self, x):",
    indent 4 "y, _ = self.lstm(x)",
    indent 4 "return y",
    "",
    "class RNNClassifier(nn.Module):",
    indent 2 "def __init__(self, input_size: int, hidden_size: int, num_classes: int):",
    indent 4 "super().__init__()",
    indent 4 "self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)",
    indent 4 "self.select = SelectLast()",
    indent 4 "self.fc = nn.Linear(hidden_size, num_classes)",
    indent 2 "def forward(self, x):",
    indent 4 "y, _ = self.rnn(x)",
    indent 4 "y = self.select(y)",
    indent 4 "return self.fc(y)",
    "",
    "class GRUClassifier(nn.Module):",
    indent 2 "def __init__(self, input_size: int, hidden_size: int, num_classes: int):",
    indent 4 "super().__init__()",
    indent 4 "self.gru = nn.GRU(input_size, hidden_size, batch_first=True)",
    indent 4 "self.select = SelectLast()",
    indent 4 "self.fc = nn.Linear(hidden_size, num_classes)",
    indent 2 "def forward(self, x):",
    indent 4 "y, _ = self.gru(x)",
    indent 4 "y = self.select(y)",
    indent 4 "return self.fc(y)",
    "",
    "class LSTMClassifier(nn.Module):",
    indent 2 "def __init__(self, input_size: int, hidden_size: int, num_classes: int):",
    indent 4 "super().__init__()",
    indent 4 "self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)",
    indent 4 "self.select = SelectLast()",
    indent 4 "self.fc = nn.Linear(hidden_size, num_classes)",
    indent 2 "def forward(self, x):",
    indent 4 "y, _ = self.lstm(x)",
    indent 4 "y = self.select(y)",
    indent 4 "return self.fc(y)",
    "",
    "class UnsupportedLayer(nn.Module):",
    indent 2 "def __init__(self, kind: str, detail: str = \"\"):",
    indent 4 "super().__init__()",
    indent 4 "self.kind = kind",
    indent 4 "self.detail = detail",
    indent 2 "def forward(self, x):",
    indent 4 "raise NotImplementedError(f\"Unsupported layer: {self.kind} ({self.detail})\")",
    "",
    "class ScaledDotProductSelfAttention(nn.Module):",
    indent 2 "def __init__(self, d_model: int):",
    indent 4 "super().__init__()",
    indent 4 "self.d_model = d_model",
    indent 2 "def forward(self, x):",
    indent 4 "# x: (batch, seq, d_model)",
    indent 4 "scores = torch.matmul(x, x.transpose(-2, -1)) / (self.d_model ** 0.5)",
    indent 4 "attn = torch.softmax(scores, dim=-1)",
    indent 4 "return torch.matmul(attn, x)",
    "",
    "class SimpleRNN(nn.Module):",
    indent 2 ("def __init__(self, input_size: int, hidden_size: int, output_size: " ++
      "int, bidirectional: bool = False):"),
    indent 4 "super().__init__()",
    indent 4
      "self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)",
    indent 4 "out_dim = hidden_size * (2 if bidirectional else 1)",
    indent 4 "self.fc = nn.Linear(out_dim, output_size)",
    indent 2 "def forward(self, x):",
    indent 4 "y, _ = self.rnn(x)",
    indent 4 "return self.fc(y)",
    "",
    "class SimpleGRU(nn.Module):",
    indent 2 ("def __init__(self, input_size: int, hidden_size: int, output_size: " ++
      "int, bidirectional: bool = False):"),
    indent 4 "super().__init__()",
    indent 4
      "self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)",
    indent 4 "out_dim = hidden_size * (2 if bidirectional else 1)",
    indent 4 "self.fc = nn.Linear(out_dim, output_size)",
    indent 2 "def forward(self, x):",
    indent 4 "y, _ = self.gru(x)",
    indent 4 "return self.fc(y)",
    "",
    "class SimpleLSTM(nn.Module):",
    indent 2 ("def __init__(self, input_size: int, hidden_size: int, output_size: " ++
      "int, bidirectional: bool = False):"),
    indent 4 "super().__init__()",
    indent 4
      "self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)",
    indent 4 "out_dim = hidden_size * (2 if bidirectional else 1)",
    indent 4 "self.fc = nn.Linear(out_dim, output_size)",
    indent 2 "def forward(self, x):",
    indent 4 "y, _ = self.lstm(x)",
    indent 4 "return self.fc(y)",
    "",
    "class GRULanguageModel(nn.Module):",
    indent 2 "def __init__(self, vocab_size: int, hidden_size: int):",
    indent 4 "super().__init__()",
    indent 4 "self.embed = nn.Linear(vocab_size, hidden_size)",
    indent 4 "self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)",
    indent 4 "self.proj = nn.Linear(hidden_size, vocab_size)",
    indent 2 "def forward(self, x):",
    indent 4 "x = self.embed(x)",
    indent 4 "y, _ = self.gru(x)",
    indent 4 "return self.proj(y)",
    "",
    "class Seq2SeqInference(nn.Module):",
    indent 2 ("def __init__(self, src_vocab_size: int, tgt_vocab_size: int, " ++
      "embed_dim: int, hidden_dim: int, max_tgt_len: int, start_token: int = " ++
      "0):"),
    indent 4 "super().__init__()",
    indent 4 "self.src_embed = nn.Linear(src_vocab_size, embed_dim)",
    indent 4 "self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim)",
    indent 4 "self.encoder = nn.RNN(embed_dim, hidden_dim, batch_first=True)",
    indent 4 "self.decoder = nn.RNN(embed_dim, hidden_dim, batch_first=True)",
    indent 4 "self.proj = nn.Linear(hidden_dim, tgt_vocab_size)",
    indent 4 "self.max_tgt_len = max_tgt_len",
    indent 4 "self.start_token = start_token",
    indent 2 "def forward(self, x):",
    indent 4 "# x: (batch, src_len, src_vocab_size) one-hot/dists",
    indent 4 "x = self.src_embed(x)",
    indent 4 "_enc_out, h = self.encoder(x)",
    indent 4 "batch = x.shape[0]",
    indent 4 "token = torch.full((batch,), self.start_token, dtype=torch.long, device=x.device)",
    indent 4 "inp = self.tgt_embed(token).unsqueeze(1)",
    indent 4 "logits = []",
    indent 4 "h_dec = h",
    indent 4 "for _ in range(self.max_tgt_len):",
    indent 6 "y, h_dec = self.decoder(inp, h_dec)",
    indent 6 "step = self.proj(y.squeeze(1))",
    indent 6 "logits.append(step)",
    indent 6 "token = torch.argmax(step, dim=-1)",
    indent 6 "inp = self.tgt_embed(token).unsqueeze(1)",
    indent 4 "return torch.stack(logits, dim=1)",
  ]

/--
Generate a generic base `nn.Module` class skeleton.

This is used by exporters that want a "real" class with an explicit `_initialize_layers` hook,
instead of the simpler `nn.Sequential` emitter.
-/
def base (className : String) (docstring : String) : String :=
  joinLines <|
    #[ s!"class {className}(nn.Module):"
    , indent 2 s!"\"\"\"{docstring}\"\"\""
    , indent 2 ""
    , indent 2 "def __init__(self):"
    , indent 4 "super().__init__()"
    , indent 4 "self._initialize_layers()"
    , indent 4 ""
    , indent 2 "def _initialize_layers(self):"
    , indent 4 "raise NotImplementedError(\"Subclasses must implement _initialize_layers\")"
    , indent 4 ""
    , indent 2 "def forward(self, x):"
    , indent 4 "raise NotImplementedError(\"Subclasses must implement forward\")"
    , indent 4 ""
    ] ++
      metadata className ++
      #[ indent 4 ""
      , indent 2 "@property"
      , indent 2 "def input_shape(self):"
      , indent 4 "raise NotImplementedError(\"Subclasses must implement input_shape\")"
      , indent 4 ""
      , indent 2 "@property"
      , indent 2 "def output_shape(self):"
      , indent 4 "raise NotImplementedError(\"Subclasses must implement output_shape\")"
      , indent 4 ""
      , indent 2 "@property"
      , indent 2 "def layer_count(self):"
      , indent 4 "raise NotImplementedError(\"Subclasses must implement layer_count\")"
      , indent 4 ""
      , indent 2 "@property"
      , indent 2 "def operation_types(self):"
      , indent 4 "raise NotImplementedError(\"Subclasses must implement operation_types\")"
      ]

/-- Emit Python helpers for saving/loading state-dict checkpoints. -/
def checkpoints : String :=
  joinLines #[
    "def load_weights_from_dict(model: nn.Module, state_dict: dict):",
    indent 2 "\"\"\"Load weights from a state dictionary into the model.\"\"\"",
    indent 2 "model.load_state_dict(state_dict)",
    indent 2 "return model",
    "",
    "def save_weights_to_dict(model: nn.Module) -> dict:",
    indent 2 "\"\"\"Save model weights to a state dictionary.\"\"\"",
    indent 2 "return model.state_dict()",
    "",
    "def save_model_to_file(model: nn.Module, filepath: str):",
    indent 2 "\"\"\"Save model weights to a file as a state_dict checkpoint.\"\"\"",
    indent 2 "torch.save(model.state_dict(), filepath)",
    "",
    "def load_model_from_file(filepath: str, model: Optional[nn.Module] = None):",
    indent 2 "\"\"\"Load a state_dict checkpoint; optionally materialize it into `model`.\"\"\"",
    indent 2 "state_dict = torch.load(filepath, weights_only=True)",
    indent 2 "if model is None:",
    indent 2 "    return state_dict",
    indent 2 "model.load_state_dict(state_dict)",
    indent 2 "return model"
  ]

/-- Emit Python helpers for validating exported models. -/
def checks : String :=
  joinLines #[
    "def test_model_forward(model: nn.Module, input_shape: Tuple[int, ...], num_tests: int = 5):",
    indent 2 "\"\"\"Test model forward pass with random inputs.\"\"\"",
    indent 2 "model.eval()",
    indent 2 "with torch.no_grad():",
    indent 4 "for i in range(num_tests):",
    indent 6 "x = torch.randn(1, *input_shape)",
    indent 6 "y = model(x)",
    indent 6 "print(f\"Test {i+1}: Input shape: {x.shape}, Output shape: {y.shape}\")",
    indent 6 "print(f\"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]\")",
    "",
    "def count_parameters(model: nn.Module) -> int:",
    indent 2 "\"\"\"Count the number of trainable parameters in the model.\"\"\"",
    indent 2 "return sum(p.numel() for p in model.parameters() if p.requires_grad)",
    "",
    "def print_model_summary(model: nn.Module):",
    indent 2 "\"\"\"Print a summary of the model architecture.\"\"\"",
    indent 2 "print(f\"Model: {model.__class__.__name__}\")",
    indent 2 "print(f\"Total parameters: {count_parameters(model):,}\")",
    indent 2 "print(f\"Model info: {model.get_model_info()}\")"
  ]

/--
Generate a complete `nn.Sequential`-based Python module for a `Spec.Module.Chain`.

This is the simplest exporter: we extract an array of `(opName, pythonLayerString)` pairs and drop
them into an `nn.Sequential(...)` in a new class.
The scalar storage is inherited from the supplied chain; emission only reads its layer metadata.
-/
def emit {α : Type} [TorchLean.Storage α] {s t : Shape}
  (chain : Spec.Module.Chain α s t) (className : String := "ExportedModel") : String :=
  let inputShape := shapeLiteral s
  let outputShape := shapeLiteral t
  let layers := Spec.Module.Chain.layerInfo chain
  let layerCount := layers.size
  let layerStrings := layers.map (fun (_, pytorch) => indent 8 pytorch)
  let opList :=
    "[" ++ String.intercalate ", " (layers.map (fun (op, _) => s!"\"{op}\"")).toList ++ "]"

  joinLines <|
    #[ imports
    , support
    , ""
    , s!"class {className}(nn.Module):"
    , indent 2 "def __init__(self):"
    , indent 4 "super().__init__()"
    , indent 4 s!"# Input shape: {inputShape}"
    , indent 4 s!"# Output shape: {outputShape}"
    , indent 4 s!"# Layer count: {layerCount}"
    , indent 4 s!"# Operations: {String.intercalate ", " (layers.map (fun (op, _) => op)).toList}"
    , indent 4 ""
    , indent 4 "self.layers = nn.Sequential("
    , String.intercalate ",\n" layerStrings.toList
    , indent 4 ")"
    , ""
    , indent 2 "def forward(self, x):"
    , indent 4 "return self.layers(x)"
    , indent 4 ""
    , indent 2 "@property"
    , indent 2 "def input_shape(self):"
    , indent 4 s!"return {inputShape}"
    , indent 4 ""
    , indent 2 "@property"
    , indent 2 "def output_shape(self):"
    , indent 4 s!"return {outputShape}"
    , indent 4 ""
    , indent 2 "@property"
    , indent 2 "def layer_count(self):"
    , indent 4 s!"return {layerCount}"
    , indent 4 ""
    , indent 2 "@property"
    , indent 2 "def operation_types(self):"
    , indent 4 s!"return {opList}"
    , indent 4 ""
    ] ++
      metadata className

end PyTorch
end Export
