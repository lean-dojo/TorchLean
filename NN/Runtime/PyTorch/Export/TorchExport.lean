/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core

/-!
# PyTorch `nn.Module` Graph Capture Adapter

This module emits the Python side of the model-import bridge:

```text
PyTorch nn.Module
  --torch.export / FX capture-->
TorchLean graph JSON (`torchlean.ir.v1`)
  --NN.Runtime.PyTorch.Import.TorchExport.parseGraph-->
NN.IR.Graph
```

Why generate a Python adapter instead of parsing PyTorch objects in Lean?

- PyTorch checkpoints are Python/Pickle/zip artifacts; PyTorch is the external loader that knows
  how to read them.
- `torch.export` and FX already know how to inspect PyTorch programs.
- Lean should receive a small, explicit artifact and then validate it with TorchLean's own IR
  checkers.

The generated adapter is intentionally conservative. It lowers only ops that already exist in
`NN.IR.OpKind`; unsupported operators fail with a clear message. This is exactly the behavior we
want for verification: better to reject a model than silently erase semantics.

References:
- `torch.export`: `https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html`
- `torch.fx`: `https://docs.pytorch.org/docs/stable/fx.html`
-/

@[expose] public section

namespace Export
namespace PyTorch
namespace TorchExport

open Export.PyTorch

/-- Options for the generated PyTorch graph-capture script. -/
structure GraphBridgeOptions where
  /-- Name of the Python helper function emitted into the script. -/
  functionName : String := "export_torchlean_graph_json"
  /-- If true, use `torch.export.export` first and fall back to FX symbolic tracing. -/
  preferTorchExport : Bool := true
  /-- If true, include raw PyTorch target strings in each node for debugging. -/
  includeDebugTargets : Bool := true
  deriving Repr

def pyBool (b : Bool) : String :=
  if b then "True" else "False"

/--
Emit a Python script that captures a PyTorch module and writes TorchLean graph JSON.

The generated script expects a Python file containing a zero-argument model constructor or class:

```bash
python export_torchlean_graph.py my_model.py MyModel out_graph.json --example-shape 1,4
```

The first implementation target is the shared IR subset: elementwise ops, matmul, reductions,
reshape/permute/flatten/concat, softmax, layernorm, and simple pooling/conv metadata when PyTorch
exposes enough static arguments. Linear layers can appear either as `aten.linear` or as lower-level
`matmul/add` depending on PyTorch's graph capture.
-/
def generateGraphBridgeScript (opts : GraphBridgeOptions := {}) : String :=
  joinLines
    [ "import argparse"
    , "import importlib.util"
    , "import json"
    , "import operator"
    , "from pathlib import Path"
    , "import torch"
    , "import torch.nn as nn"
    , "import torch.nn.functional as F"
    , ""
    , "FORMAT = \"torchlean.ir.v1\""
    , ""
    , "def _shape_of(value):"
    , indent4 "if hasattr(value, \"shape\"):"
    , indent8 "return [int(x) for x in tuple(value.shape)]"
    , indent4 "if isinstance(value, (tuple, list)) and value and hasattr(value[0], \"shape\"):"
    , indent8 "return [int(x) for x in tuple(value[0].shape)]"
    , indent4 "return []"
    , ""
    , "def _node_shape(node):"
    , indent4 "return _shape_of(node.meta.get(\"val\", node.meta.get(\"tensor_meta\")))"
    , ""
    , "def _shape_from_arg(arg):"
    , indent4 "return _node_shape(arg) if hasattr(arg, \"meta\") else []"
    , ""
    , "def _tuple_shapes_of_node(node):"
    , indent4 "val = node.meta.get(\"val\", None)"
    , indent4 "if isinstance(val, (tuple, list)) and not hasattr(val, \"shape\"):"
    , indent8 "return [_shape_of(x) for x in val]"
    , indent4 "tm = node.meta.get(\"tensor_meta\", None)"
    , indent4 "if hasattr(tm, \"shape\"):"
    , indent8 "return []"
    , indent4 "if isinstance(tm, (tuple, list)):"
    , indent8 "return [_shape_of(x) for x in tm]"
    , indent4 "return []"
    , ""
    , "def _is_getitem(node):"
    , indent4 "return node.target is operator.getitem or _target_name(node.target) == \"<built-in function getitem>\""
    , ""
    , "def _is_getattr(node):"
    , indent4 "return node.target is getattr or _target_name(node.target) == \"<built-in function getattr>\""
    , ""
    , "def _tuple_kind(node, model=None):"
    , indent4 "if node.op == \"call_module\" and model is not None:"
    , indent8 "mod = model.get_submodule(str(node.target))"
    , indent8 "if isinstance(mod, nn.MultiheadAttention):"
    , indent8 "    return {"
    , indent8 "        \"kind\": \"multihead_attention\","
    , indent8 "        \"embed_dim\": int(mod.embed_dim),"
    , indent8 "        \"num_heads\": int(mod.num_heads),"
    , indent8 "        \"batch_first\": bool(mod.batch_first),"
    , indent8 "        \"dropout_zero\": bool(float(mod.dropout) == 0.0),"
    , indent8 "        \"bias\": bool(mod.in_proj_bias is not None),"
    , indent8 "    }"
    , indent4 "return {\"kind\": \"py_tuple\"}"
    , ""
    , "def _load_model(module_path: str, ctor_name: str):"
    , indent4 "spec = importlib.util.spec_from_file_location(\"torchlean_user_model\", module_path)"
    , indent4 "if spec is None or spec.loader is None:"
    , indent8 "raise RuntimeError(f\"could not load Python module from {module_path}\")"
    , indent4 "mod = importlib.util.module_from_spec(spec)"
    , indent4 "spec.loader.exec_module(mod)"
    , indent4 "ctor = getattr(mod, ctor_name)"
    , indent4 "model = ctor()"
    , indent4 "model.eval()"
    , indent4 "return model"
    , ""
    , "def _target_name(target):"
    , indent4 "return str(target).replace(\"torch.ops.\", \"\")"
    , ""
    , "def _node_refs(obj, node_to_id):"
    , indent4 "refs = []"
    , indent4 "def visit(x):"
    , indent8 "if x in node_to_id:"
    , indent8 "    if node_to_id[x] is not None:"
    , indent8 "        refs.append(node_to_id[x])"
    , indent8 "elif isinstance(x, (tuple, list)):"
    , indent8 "    for y in x: visit(y)"
    , indent8 "elif isinstance(x, dict):"
    , indent8 "    for y in x.values(): visit(y)"
    , indent4 "visit(obj)"
    , indent4 "return refs"
    , ""
    , "def _first_int(x, default=0):"
    , indent4 "if isinstance(x, int): return int(x)"
    , indent4 "if isinstance(x, (tuple, list)) and x: return int(x[0])"
    , indent4 "return default"
    , ""
    , "def _pair(x, default):"
    , indent4 "if isinstance(x, int): return int(x), int(x)"
    , indent4 "if isinstance(x, (tuple, list)) and len(x) >= 2: return int(x[0]), int(x[1])"
    , indent4 "return default, default"
    , ""
    , "def _normalize_axis(axis, rank):"
    , indent4 "axis = int(axis)"
    , indent4 "return axis + rank if axis < 0 else axis"
    , ""
    , "def _lower_kind(node, model=None):"
    , indent4 "name = _target_name(node.target)"
    , indent4 "args = node.args"
    , indent4 "kwargs = dict(node.kwargs)"
    , indent4 "axis = kwargs.get(\"dim\", kwargs.get(\"axis\", None))"
    , indent4 "if axis is None and len(args) > 1 and isinstance(args[1], int): axis = args[1]"
    , indent4 "if node.op == \"call_module\" and model is not None:"
    , indent8 "mod = model.get_submodule(str(node.target))"
    , indent8 "if isinstance(mod, nn.Linear): return {\"kind\": \"linear\"}"
    , indent8 "if isinstance(mod, nn.ReLU): return {\"kind\": \"relu\"}"
    , indent8 "if isinstance(mod, nn.Tanh): return {\"kind\": \"tanh\"}"
    , indent8 "if isinstance(mod, nn.Sigmoid): return {\"kind\": \"sigmoid\"}"
    , indent8 "if isinstance(mod, nn.Softmax): return {\"kind\": \"softmax\", \"axis\": int(mod.dim)}"
    , indent8 "if isinstance(mod, nn.Flatten): return {\"kind\": \"flatten\", \"value_shape\": _shape_from_arg(args[0]) if args else []}"
    , indent8 "if isinstance(mod, nn.LayerNorm):"
    , indent8 "    rank = len(_node_shape(node))"
    , indent8 "    norm_rank = len(tuple(mod.normalized_shape))"
    , indent8 "    return {\"kind\": \"layernorm\", \"axis\": max(0, rank - norm_rank)}"
    , indent8 "if isinstance(mod, nn.Conv2d):"
    , indent8 "    kH, kW = _pair(mod.kernel_size, 1)"
    , indent8 "    stride = _first_int(mod.stride, 1)"
    , indent8 "    padding = _first_int(mod.padding, 0)"
    , indent8 "    return {\"kind\": \"conv2d\", \"inC\": int(mod.in_channels), \"outC\": int(mod.out_channels), \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding}"
    , indent8 "if isinstance(mod, nn.MaxPool2d):"
    , indent8 "    kH, kW = _pair(mod.kernel_size, 1)"
    , indent8 "    stride = _first_int(mod.stride if mod.stride is not None else mod.kernel_size, kH)"
    , indent8 "    padding = _first_int(mod.padding, 0)"
    , indent8 "    return {\"kind\": \"max_pool2d_pad\" if padding else \"max_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding} if padding else {\"kind\": \"max_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride}"
    , indent8 "if isinstance(mod, nn.AvgPool2d):"
    , indent8 "    kH, kW = _pair(mod.kernel_size, 1)"
    , indent8 "    stride = _first_int(mod.stride if mod.stride is not None else mod.kernel_size, kH)"
    , indent8 "    padding = _first_int(mod.padding, 0)"
    , indent8 "    return {\"kind\": \"avg_pool2d_pad\" if padding else \"avg_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding} if padding else {\"kind\": \"avg_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride}"
    , indent4 "if name in (\"aten.add.Tensor\", \"aten.add.Scalar\") or node.target is operator.add: return {\"kind\": \"add\"}"
    , indent4 "if name in (\"aten.sub.Tensor\", \"aten.sub.Scalar\") or node.target is operator.sub: return {\"kind\": \"sub\"}"
    , indent4 "if name in (\"aten.mul.Tensor\", \"aten.mul.Scalar\") or node.target is operator.mul: return {\"kind\": \"mul_elem\"}"
    , indent4 "if name.startswith(\"aten.relu\") or node.target is torch.relu or node.target is F.relu: return {\"kind\": \"relu\"}"
    , indent4 "if name.startswith(\"aten.tanh\") or node.target is torch.tanh: return {\"kind\": \"tanh\"}"
    , indent4 "if name.startswith(\"aten.sigmoid\") or node.target is torch.sigmoid: return {\"kind\": \"sigmoid\"}"
    , indent4 "if name.startswith(\"aten.exp\") or node.target is torch.exp: return {\"kind\": \"exp\"}"
    , indent4 "if name.startswith(\"aten.log\") or node.target is torch.log: return {\"kind\": \"log\"}"
    , indent4 "if name.startswith(\"aten.sin\") or node.target is torch.sin: return {\"kind\": \"sin\"}"
    , indent4 "if name.startswith(\"aten.cos\") or node.target is torch.cos: return {\"kind\": \"cos\"}"
    , indent4 "if name.startswith(\"aten.abs\") or node.target is torch.abs: return {\"kind\": \"abs\"}"
    , indent4 "if name.startswith(\"aten.sqrt\") or node.target is torch.sqrt: return {\"kind\": \"sqrt\"}"
    , indent4 "if name.startswith(\"aten.reciprocal\"): return {\"kind\": \"inv\"}"
    , indent4 "if name.startswith(\"aten.maximum\") or node.target is torch.maximum: return {\"kind\": \"max_elem\"}"
    , indent4 "if name.startswith(\"aten.minimum\") or node.target is torch.minimum: return {\"kind\": \"min_elem\"}"
    , indent4 "if name.startswith(\"aten.matmul\") or name.startswith(\"aten.mm\") or node.target is torch.matmul: return {\"kind\": \"matmul\"}"
    , indent4 "if name.startswith(\"aten.sum\") and axis is None: return {\"kind\": \"sum\"}"
    , indent4 "if name.startswith(\"aten.sum\"): return {\"kind\": \"reduce_sum\", \"axis\": int(axis)}"
    , indent4 "if name.startswith(\"aten.mean\"): return {\"kind\": \"reduce_mean\", \"axis\": int(axis)}"
    , indent4 "if name.startswith(\"aten.softmax\") or \"softmax\" in name or node.target is F.softmax:"
    , indent8 "rank = len(_node_shape(node))"
    , indent8 "return {\"kind\": \"softmax\", \"axis\": _normalize_axis(axis if axis is not None else -1, rank)}"
    , indent4 "if name.startswith(\"aten.reshape\") or name.startswith(\"aten.view\") or name in (\"reshape\", \"view\"):"
    , indent8 "return {\"kind\": \"reshape\", \"in_shape\": _shape_from_arg(args[0]) if args else [], \"out_shape\": _node_shape(node)}"
    , indent4 "if name == \"contiguous\":"
    , indent8 "s = _shape_from_arg(args[0]) if args else _node_shape(node)"
    , indent8 "return {\"kind\": \"reshape\", \"in_shape\": s, \"out_shape\": _node_shape(node)}"
    , indent4 "if name.startswith(\"aten.flatten\") or name == \"flatten\" or node.target is torch.flatten: return {\"kind\": \"flatten\", \"value_shape\": _shape_from_arg(args[0]) if args else []}"
    , indent4 "if name.startswith(\"aten.permute\") or name == \"permute\":"
    , indent8 "perm = list(args[1]) if len(args) > 1 and isinstance(args[1], (tuple, list)) else [int(x) for x in args[1:]] if name == \"permute\" else list(kwargs.get(\"dims\", []))"
    , indent8 "return {\"kind\": \"permute\", \"perm\": [int(x) for x in perm]}"
    , indent4 "if name.startswith(\"aten.cat\") or node.target is torch.cat:"
    , indent8 "dim = int(kwargs.get(\"dim\", args[1] if len(args) > 1 and isinstance(args[1], int) else 0))"
    , indent8 "return {\"kind\": \"concat\", \"axis\": dim}"
    , indent4 "if name.startswith(\"aten.transpose\") or name == \"transpose\":"
    , indent8 "d0 = int(args[1]); d1 = int(args[2])"
    , indent8 "if d0 == 0 and d1 == 1: return {\"kind\": \"swap_first_two\"}"
    , indent8 "if d0 == -1 and d1 == -2: return {\"kind\": \"transpose3d_last_two\"}"
    , indent8 "if d0 == -2 and d1 == -1: return {\"kind\": \"transpose3d_last_two\"}"
    , indent4 "if name.startswith(\"aten.layer_norm\") or node.target is F.layer_norm:"
    , indent8 "return {\"kind\": \"layernorm\", \"axis\": 1}"
    , indent4 "if name.startswith(\"aten.linear\") or node.target is F.linear: return {\"kind\": \"linear\"}"
    , indent4 "if name.startswith(\"aten.conv2d\") or node.target is F.conv2d:"
    , indent8 "stride = _first_int(kwargs.get(\"stride\", args[3] if len(args) > 3 else 1), 1)"
    , indent8 "padding = _first_int(kwargs.get(\"padding\", args[4] if len(args) > 4 else 0), 0)"
    , indent8 "weight = args[1] if len(args) > 1 else None"
    , indent8 "wshape = _shape_of(weight.meta.get(\"val\") if hasattr(weight, \"meta\") else weight)"
    , indent8 "outC, inC, kH, kW = (wshape + [0, 0, 0, 0])[:4]"
    , indent8 "return {\"kind\": \"conv2d\", \"inC\": inC, \"outC\": outC, \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding}"
    , indent4 "if name.startswith(\"aten.max_pool2d\") or node.target is F.max_pool2d:"
    , indent8 "kH, kW = _pair(args[1] if len(args) > 1 else kwargs.get(\"kernel_size\", 1), 1)"
    , indent8 "stride = _first_int(kwargs.get(\"stride\", args[2] if len(args) > 2 else kH), kH)"
    , indent8 "padding = _first_int(kwargs.get(\"padding\", args[3] if len(args) > 3 else 0), 0)"
    , indent8 "return {\"kind\": \"max_pool2d_pad\" if padding else \"max_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding} if padding else {\"kind\": \"max_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride}"
    , indent4 "if name.startswith(\"aten.avg_pool2d\") or node.target is F.avg_pool2d:"
    , indent8 "kH, kW = _pair(args[1] if len(args) > 1 else kwargs.get(\"kernel_size\", 1), 1)"
    , indent8 "stride = _first_int(kwargs.get(\"stride\", args[2] if len(args) > 2 else kH), kH)"
    , indent8 "padding = _first_int(kwargs.get(\"padding\", args[3] if len(args) > 3 else 0), 0)"
    , indent8 "return {\"kind\": \"avg_pool2d_pad\" if padding else \"avg_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride, \"padding\": padding} if padding else {\"kind\": \"avg_pool2d\", \"kH\": kH, \"kW\": kW, \"stride\": stride}"
    , indent4 "if _is_getitem(node):"
    , indent8 "idx = int(args[1]) if len(args) > 1 and isinstance(args[1], int) else 0"
    , indent8 "return {\"kind\": \"tuple_getitem\", \"index\": idx}"
    , indent4 "if _is_getattr(node):"
    , indent8 "attr = args[1] if len(args) > 1 else \"<unknown>\""
    , indent8 "raise NotImplementedError(f\"unsupported PyTorch attribute projection: {attr}. If this came from a tuple-returning op such as torch.sort(...).values, add a value-graph/lowering rule for that producer instead of treating the attribute as an ordinary tensor op.\")"
    , indent4 "raise NotImplementedError(f\"unsupported PyTorch op: {name}\")"
    , ""
    , "def _capture(model, example):"
    , indent4 "if " ++ pyBool opts.preferTorchExport ++ ":"
    , indent8 "try:"
    , indent8 "    ep = torch.export.export(model, (example,))"
    , indent8 "    return ep.graph"
    , indent8 "except Exception:"
    , indent8 "    pass"
    , indent4 "from torch.fx import symbolic_trace"
    , indent4 "from torch.fx.passes.shape_prop import ShapeProp"
    , indent4 "gm = symbolic_trace(model)"
    , indent4 "ShapeProp(gm).propagate(example)"
    , indent4 "return gm.graph"
    , ""
    , s!"def {opts.functionName}(model, example, json_path: str):"
    , indent4 "graph = _capture(model, example)"
    , indent4 "nodes = []"
    , indent4 "node_to_id = {}"
    , indent4 "output_id = None"
    , indent4 "input_id = None"
    , indent4 "for node in graph.nodes:"
    , indent8 "if node.op == \"output\":"
    , indent8 "    refs = _node_refs(node.args, node_to_id)"
    , indent8 "    output_id = refs[0] if refs else None"
    , indent8 "    continue"
    , indent8 "if len(getattr(node, \"users\", {})) == 0:"
    , indent8 "    # FX often leaves dead tuple projections behind, e.g. attention weights from"
    , indent8 "    # `y, _ = mha(...)`. They are not part of the exported value, so we omit them"
    , indent8 "    # instead of forcing every unused container projection to have a tensor lowering."
    , indent8 "    node_to_id[node] = None"
    , indent8 "    continue"
    , indent8 "if node.op in (\"get_attr\",):"
    , indent8 "    continue"
    , indent8 "node_id = len(nodes)"
    , indent8 "node_to_id[node] = node_id"
    , indent8 "shape = _node_shape(node)"
    , indent8 "tuple_shapes = _tuple_shapes_of_node(node)"
    , indent8 "value_meta = {\"value_kind\": \"tuple\", \"tuple_shapes\": tuple_shapes} if tuple_shapes else {\"value_kind\": \"tensor\", \"shape\": shape}"
    , indent8 "if node.op == \"placeholder\":"
    , indent8 "    if input_id is not None:"
    , indent8 "        # torch.export lifts parameters/buffers into placeholders. TorchLean IR keeps those"
    , indent8 "        # tensors in an external parameter store, so they are not dataflow parents."
    , indent8 "        node_to_id[node] = None"
    , indent8 "        continue"
    , indent8 "    kind = {\"kind\": \"input\"}"
    , indent8 "    parents = []"
    , indent8 "    if input_id is None: input_id = node_id"
    , indent8 "elif tuple_shapes:"
    , indent8 "    # Preserve tuple/list-valued FX nodes in the import artifact. They are lowered to the"
    , indent8 "    # tensor-only TorchLean IR only when a later semantic lowering rule exists."
    , indent8 "    kind = _tuple_kind(node, model)"
    , indent8 "    parents = _node_refs((node.args, node.kwargs), node_to_id)"
    , indent8 "elif node.op in (\"call_function\", \"call_method\", \"call_module\"):"
    , indent8 "    kind = _lower_kind(node, model)"
    , indent8 "    parents = _node_refs((node.args, node.kwargs), node_to_id)"
    , indent8 "else:"
    , indent8 "    raise NotImplementedError(f\"unsupported FX node op: {node.op}\")"
    , indent8 "entry = {\"id\": node_id, \"parents\": parents, **value_meta, **kind}"
    , indent8 "if " ++ pyBool opts.includeDebugTargets ++ ":"
    , indent8 "    entry[\"debug_target\"] = _target_name(node.target)"
    , indent8 "nodes.append(entry)"
    , indent4 "if input_id is None or output_id is None:"
    , indent8 "raise RuntimeError(\"could not identify graph input/output\")"
    , indent4 "payload = {\"format\": FORMAT, \"input_id\": input_id, \"output_id\": output_id, \"nodes\": nodes}"
    , indent4 "with open(json_path, \"w\", encoding=\"utf-8\") as f:"
    , indent8 "json.dump(payload, f, indent=2, sort_keys=True)"
    , indent4 "return payload"
    , ""
    , "def main():"
    , indent4 "parser = argparse.ArgumentParser(description=\"Export a PyTorch nn.Module to TorchLean IR JSON\")"
    , indent4 "parser.add_argument(\"module\", help=\"Python file containing the model class/constructor\")"
    , indent4 "parser.add_argument(\"ctor\", help=\"Zero-argument model class or constructor name\")"
    , indent4 "parser.add_argument(\"json\", help=\"Output graph JSON path\")"
    , indent4 "parser.add_argument(\"--example-shape\", required=True, help=\"Comma-separated example input shape, e.g. 1,4\")"
    , indent4 "args = parser.parse_args()"
    , indent4 "shape = tuple(int(x) for x in args.example_shape.split(',') if x)"
    , indent4 "model = _load_model(args.module, args.ctor)"
    , indent4 "example = torch.randn(*shape)"
    , indent4 s!"payload = {opts.functionName}(model, example, args.json)"
    , indent4 "print(f\"wrote {len(payload['nodes'])} TorchLean IR nodes to {args.json}\")"
    , ""
    , "if __name__ == \"__main__\":"
    , indent4 "main()"
    ]

end TorchExport
end PyTorch
end Export
