/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core

/-!
# PyTorch `state_dict` Bridge

This module is the general weight-interchange path for PyTorch users.

The important split is:

- **Weights** move through PyTorch `state_dict`s. PyTorch’s own documentation recommends saving a
  module’s learned parameters with `torch.save(model.state_dict(), path)` because that is the most
  flexible restoration format.
- **Graphs** move through graph capture (`torch.export`, FX, ONNX, or TorchLean `NN.IR.Graph`).
  A `state_dict` alone does not describe the model architecture; it only names tensors.

Lean should not try to parse PyTorch pickle/zip checkpoints directly. Instead, we emit a small Python
adapter that loads a checkpoint with PyTorch, normalizes common wrappers such as
`{"state_dict": ...}`, and writes shape-checkable JSON:

```json
{
  "params": { "layer.weight": [[...]], "layer.bias": [...] },
  "meta": { "layer.weight": { "shape": [out, in], "dtype": "torch.float32" } }
}
```

`NN.Runtime.PyTorch.Import.Core` then parses the `"params"` object into typed TorchLean tensors.
Architecture-specific loaders are still useful, but only for mapping names and shapes. The transport
format itself is model-agnostic.

References:
- PyTorch tutorial, "Saving and Loading Models":
  `https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html`
- PyTorch `torch.export` user guide:
  `https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html`
- PyTorch FX overview:
  `https://docs.pytorch.org/docs/stable/fx.html`
-/

@[expose] public section

namespace Export
namespace PyTorch
namespace StateDict

open Export.PyTorch

/--
Options for the generated Python checkpoint-to-JSON adapter.

This adapter is intentionally conservative: it accepts tensor-valued entries only, drops common
DataParallel prefixes when requested, and writes plain JSON rather than a PyTorch-specific binary
format. That makes the output easy to inspect, diff, and parse in Lean.
-/
structure JsonBridgeOptions where
  /-- Name of the Python helper function emitted into the generated script. -/
  functionName : String := "export_state_dict_json"
  /-- If true, strip a leading `"module."` from keys produced by `torch.nn.DataParallel`. -/
  stripDataParallelPrefix : Bool := true
  /-- If true, include a `"meta"` object with per-key shape and dtype strings. -/
  includeMeta : Bool := true
  /-- Python expression passed as `weights_only` to `torch.load`. -/
  weightsOnlyExpr : String := "True"
  deriving Repr

/-- Render a Python boolean literal. -/
def pyBool (b : Bool) : String :=
  if b then "True" else "False"

/--
Emit a standalone Python script that converts a PyTorch checkpoint into TorchLean JSON.

The script handles three common checkpoint layouts:

- a raw `state_dict`;
- `{ "state_dict": state_dict }`;
- `{ "model_state_dict": state_dict }`.

Usage of the generated script:

```bash
python export_state_dict_json.py model.pt model.json
```

The resulting `model.json` is accepted by `Import.PyTorch.loadWeights?`.
-/
def generateJsonBridgeScript (opts : JsonBridgeOptions := {}) : String :=
  joinLines
    [ "import argparse"
    , "import json"
    , "import torch"
    , ""
    , "def _unwrap_state_dict(obj):"
    , indent4 "if isinstance(obj, dict):"
    , indent8 "if \"state_dict\" in obj and isinstance(obj[\"state_dict\"], dict):"
    , indent8 "    return obj[\"state_dict\"]"
    , indent8 "if \"model_state_dict\" in obj and isinstance(obj[\"model_state_dict\"], dict):"
    , indent8 "    return obj[\"model_state_dict\"]"
    , indent4 "return obj"
    , ""
    , "def _normalize_key(key: str, strip_data_parallel_prefix: bool) -> str:"
    , indent4 "if strip_data_parallel_prefix and key.startswith(\"module.\"):"
    , indent8 "return key[len(\"module.\"):]"
    , indent4 "return key"
    , ""
    , "def _tensor_payload(t: torch.Tensor):"
    , indent4 "return t.detach().cpu().tolist()"
    , ""
    , s!"def {opts.functionName}(checkpoint_path: str, json_path: str):"
    , indent4 s!"obj = torch.load(checkpoint_path, map_location=\"cpu\", weights_only={opts.weightsOnlyExpr})"
    , indent4 "state_dict = _unwrap_state_dict(obj)"
    , indent4 "if not isinstance(state_dict, dict):"
    , indent8 "raise TypeError(\"expected a state_dict or a checkpoint containing one\")"
    , indent4 "params = {}"
    , indent4 "meta = {}"
    , indent4 "for key, value in state_dict.items():"
    , indent8 "if not torch.is_tensor(value):"
    , indent8 "    continue"
    , indent8 s!"k = _normalize_key(str(key), {pyBool opts.stripDataParallelPrefix})"
    , indent8 "v = value.detach().cpu()"
    , indent8 "params[k] = _tensor_payload(v)"
    , indent8 "meta[k] = {\"shape\": list(v.shape), \"dtype\": str(v.dtype)}"
    , indent4 "payload = {\"params\": params}"
    , indent4 s!"if {pyBool opts.includeMeta}:"
    , indent8 "payload[\"meta\"] = meta"
    , indent4 "with open(json_path, \"w\", encoding=\"utf-8\") as f:"
    , indent8 "json.dump(payload, f, indent=2, sort_keys=True)"
    , indent4 "return payload"
    , ""
    , "def main():"
    , indent4 "parser = argparse.ArgumentParser(description=\"Export a PyTorch state_dict to TorchLean JSON\")"
    , indent4 "parser.add_argument(\"checkpoint\", help=\"Path to a .pt/.pth checkpoint\")"
    , indent4 "parser.add_argument(\"json\", help=\"Output JSON path\")"
    , indent4 "args = parser.parse_args()"
    , indent4 s!"payload = {opts.functionName}(args.checkpoint, args.json)"
    , indent4 "print(f\"wrote {len(payload['params'])} tensor entries to {args.json}\")"
    , ""
    , "if __name__ == \"__main__\":"
    , indent4 "main()"
    ]

end StateDict
end PyTorch
end Export
