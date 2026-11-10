/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Conversion
import Mathlib.Data.String.Defs

/-!
# Export Core

PyTorch code generation helpers.

The PyTorch exporters share the string-building helpers of this module, which write readable
Python `nn.Module` code, optionally with weights embedded.

Design note (PyTorch export APIs, for context only):

PyTorch also has *graph capture* / *serialization* mechanisms such as ONNX export and
`torch.export`, which produce IR-like artifacts intended for execution in other runtimes. The
TorchLean exporters built on these helpers emit auditable Python source for parity checks and
round-trip tests.

The public helpers are organized as follows:

- `imports` is the shared Python prelude.
- `base` is a class skeleton whose `__init__` calls the `_initialize_layers` hook.
- `checkpoints` emits Python state-dictionary save/load helpers. `python/torchlean/state.py`
  writes a state dictionary as the document that `TorchLean.PyTorch.state` reads.

## References

- PyTorch ONNX export: https://pytorch.org/docs/stable/onnx.html
- PyTorch `torch.export`: https://pytorch.org/docs/stable/export.html
-/

@[expose] public section


namespace TorchLean
namespace PyTorch

open TorchLean.Tensor

/-- Join an array of lines with newline separators. -/
def joinLines (xs : Array String) : String := String.intercalate "\n" xs.toList
/-- Render a Lean `Bool` as the corresponding Python literal. -/
def boolLiteral (b : Bool) : String :=
  if b then "True" else "False"
/-- Indent a line by `n` spaces. -/
def indent (n : Nat) (s : String) : String :=
  String.replicate n ' ' ++ s

/-!
## Common boilerplate fragments

Many exporters emit the same small pieces of Python. The example exporters share the
`get_model_info` method of `metadata`, and they and the general IR exporter share the shape, float
and tensor literals.
-/

/--
Emit a standard `get_model_info` method used by most TorchLean PyTorch example modules.

`extraFields` are inserted after the `"model_name"` entry. Each element is `(key, valueExpr)` where
`valueExpr` is emitted verbatim as Python code (e.g. `"self.input_shape"` or `"self.hidden_dim"`).
The helper only formats the entries and does not check Python syntax.
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
  let dims : List Nat := s
  match dims with
  | [] => "()"
  | [n] => s!"({n},)"
  | _ => "(" ++ String.intercalate ", " (dims.map (fun n => toString n)) ++ ")"

/-- Render a Python float expression preserving every finite binary64 value and signed zero.

A decimal string of at most 24 characters is kept when it parses back to the original bits.
Otherwise the expression uses Python's built-in `float.fromhex` with the exact integer significand
and binary exponent. Infinities and NaN use the constructors `float('inf')`, `float('-inf')` and
`float('nan')`, so a NaN loses its sign and payload bits.
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

The recursive printer serves examples and small regression tests and is not tuned for speed.
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
Generate a generic base `nn.Module` class skeleton.

Its `__init__` calls the `_initialize_layers` hook, and the hook, `forward` and the properties
`input_shape`, `output_shape`, `layer_count` and `operation_types` raise `NotImplementedError`
until a subclass overrides them. The class also gets the `get_model_info` method of `metadata`.
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

end PyTorch
end TorchLean
