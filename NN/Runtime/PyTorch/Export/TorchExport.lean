/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Runtime.PyTorch.Wire

/-!
# PyTorch graph capture

This module generates a Python script that captures an `nn.Module` and writes its graph as
`torchlean.ir.v1` JSON. The Lean importer reads that artifact and checks it before evaluation:

```text
PyTorch nn.Module
  --torch.export / FX capture-->
TorchLean graph JSON (`torchlean.ir.v1`)
  --Import.PyTorch.TorchExport.parseGraph-->
NN.IR.Graph
```

PyTorch handles capture and shape propagation. The generated script translates supported
operations into `NN.IR.OpKind` and reports an error for unsupported operations or configurations.
The importer then checks the graph's edges, shapes, and operation attributes.

Operator matching uses exact callable or ATen identities. For example, `torch.log` can become an
IR `.log` node, while `torch.log1p` needs a different formula and is rejected. Similar names and
matching output shapes are insufficient to identify the computation.

The script is assembled from one definition per Python section so each piece can be read and
changed on its own. Every `"kind"` string it writes comes from `NN.Runtime.PyTorch.Wire`, the
same table the Lean importer parses with.

References:
- `torch.export`: `https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html`
- `torch.fx`: `https://docs.pytorch.org/docs/stable/fx.html`
-/

@[expose] public section

namespace Export
namespace PyTorch
namespace TorchExport

open Export.PyTorch
open Interop.PyTorch

/-- Options for the generated PyTorch graph-capture script. -/
structure Options where
  /-- Name of the Python helper function emitted into the script. -/
  functionName : String := "export_torchlean_graph_json"
  /-- If true, use `torch.export.export` first and fall back to FX symbolic tracing. -/
  preferTorchExport : Bool := true
  /-- If true, include raw PyTorch target strings in each node for debugging. -/
  includeDebugTargets : Bool := true
deriving Repr

namespace Internal

/-! ## Python sections

Each section is an array of Python lines ending in one blank separator line. Indentation is
applied here so the sections concatenate into a valid module.
-/

/-- Imports and the artifact format marker. -/
def imports : Array String :=
  #[ "import argparse"
   , "import importlib.util"
   , "import json"
   , "import operator"
   , "from pathlib import Path"
   , "import torch"
   , "import torch.nn as nn"
   , "import torch.nn.functional as F"
   , ""
   , "FORMAT = \"" ++ Wire.format ++ "\""
   , "" ]

/-- Shape readers for FX node metadata plus the `getitem`/`getattr` target tests. -/
def shapes : Array String :=
  #[ "def _shape_of(value):"
   , indent 4 "if hasattr(value, \"shape\"):"
   , indent 8 "return [int(x) for x in tuple(value.shape)]"
   , indent 4 "if isinstance(value, (tuple, list)) and value and hasattr(value[0], \"shape\"):"
   , indent 8 "return [int(x) for x in tuple(value[0].shape)]"
   , indent 4 "return []"
   , ""
   , "def _node_shape(node):"
   , indent 4 "return _shape_of(node.meta.get(\"val\", node.meta.get(\"tensor_meta\")))"
   , ""
   , "def _shape_from_arg(arg):"
   , indent 4 "return _node_shape(arg) if hasattr(arg, \"meta\") else []"
   , ""
   , "def _tuple_shapes_of_node(node):"
   , indent 4 "val = node.meta.get(\"val\", None)"
   , indent 4 "if isinstance(val, (tuple, list)) and not hasattr(val, \"shape\"):"
   , indent 8 "return [_shape_of(x) for x in val]"
   , indent 4 "tm = node.meta.get(\"tensor_meta\", None)"
   , indent 4 "if hasattr(tm, \"shape\"):"
   , indent 8 "return []"
   , indent 4 "if isinstance(tm, (tuple, list)):"
   , indent 8 "return [_shape_of(x) for x in tm]"
   , indent 4 "return []"
   , ""
   , "def _is_getitem(node):"
   , indent 4 "return node.op == \"call_function\" and node.target is operator.getitem"
   , ""
   , "def _is_getattr(node):"
   , indent 4 "return node.op == \"call_function\" and node.target is getattr"
   , "" ]

/-- Kind entries for tuple-valued FX nodes; `nn.MultiheadAttention` carries its payload. -/
def tupleKind : Array String :=
  #[ "def _tuple_kind(node, model=None):"
   , indent 4 "if node.op == \"call_module\" and model is not None:"
   , indent 8 "mod = model.get_submodule(str(node.target))"
   , indent 8 "if isinstance(mod, nn.MultiheadAttention):"
   , indent 8 "    if mod.add_zero_attn or mod.bias_k is not None or mod.bias_v is not None:"
   , indent 8 ("        raise NotImplementedError(\"nn.MultiheadAttention with added" ++
       " bias/zero sequence entries is outside the current TorchLean IR import subset\")")
   , indent 8 "    if mod.kdim != mod.embed_dim or mod.vdim != mod.embed_dim:"
   , indent 8 ("        raise NotImplementedError(\"nn.MultiheadAttention with kdim/vdim " ++
       "different from embed_dim is outside the current TorchLean IR import subset\")")
   , indent 8 ("    if node.kwargs.get(\"attn_mask\", None) is not None or " ++
       "node.kwargs.get(\"key_padding_mask\", None) is not None:")
   , indent 8 ("        raise NotImplementedError(\"masked nn.MultiheadAttention needs an " ++
       "explicit TorchLean hard-mask lowering\")")
   , indent 8 "    if bool(node.kwargs.get(\"is_causal\", False)):"
   , indent 8 ("        raise NotImplementedError(\"causal nn.MultiheadAttention needs an " ++
       "explicit TorchLean hard-mask lowering\")")
   , indent 8 "    return {"
   , indent 8 ("        \"kind\": \"" ++ Wire.attention ++ "\",")
   , indent 8 "        \"embed_dim\": int(mod.embed_dim),"
   , indent 8 "        \"num_heads\": int(mod.num_heads),"
   , indent 8 "        \"batch_first\": bool(mod.batch_first),"
   , indent 8 "        \"dropout_zero\": bool(float(mod.dropout) == 0.0),"
   , indent 8 "        \"bias\": bool(mod.in_proj_bias is not None),"
   , indent 8 "        **_multihead_attention_payload(mod),"
   , indent 8 "    }"
   , indent 4 "raise NotImplementedError(f\"unsupported tuple-valued PyTorch op: {node.target}\")"
   , "" ]

/-- Model loading, target naming, FX node reference collection, and spatial tuple parsing. -/
def model : Array String :=
  #[ "def _load_model(module_path: str, ctor_name: str):"
   , indent 4 ("spec = importlib.util.spec_from_file_location(\"torchlean_user_model\", " ++
       "module_path)")
   , indent 4 "if spec is None or spec.loader is None:"
   , indent 8 "raise RuntimeError(f\"could not load Python module from {module_path}\")"
   , indent 4 "mod = importlib.util.module_from_spec(spec)"
   , indent 4 "spec.loader.exec_module(mod)"
   , indent 4 "ctor = getattr(mod, ctor_name)"
   , indent 4 "model = ctor()"
   , indent 4 "model.eval()"
   , indent 4 "return model"
   , ""
   , "def _target_name(target):"
   , indent 4 "return str(target).replace(\"torch.ops.\", \"\")"
   , ""
   , "def _is_op(node, aten=(), functions=(), methods=()):"
   , indent 4 "# ATen overloads have distinct contracts: exp.out writes to an output tensor."
   , indent 4 "# FX keeps Python callables or method names, so match those identities explicitly."
   , indent 4 "if node.op == \"call_function\":"
   , indent 8 "if isinstance(node.target, torch._ops.OpOverload):"
   , indent 8 "    return str(node.target) in aten"
   , indent 8 "return any(node.target is function for function in functions)"
   , indent 4 "return node.op == \"call_method\" and node.target in methods"
   , ""
   , "def _node_refs(obj, node_to_id):"
   , indent 4 "refs = []"
   , indent 4 "def visit(x):"
   , indent 8 "if isinstance(x, torch.fx.Node):"
   , indent 8 "    ref = node_to_id.get(x, None)"
   , indent 8 "    if ref is not None:"
   , indent 8 "        refs.append(ref)"
   , indent 8 "elif isinstance(x, (tuple, list)):"
   , indent 8 "    for y in x: visit(y)"
   , indent 8 "elif isinstance(x, dict):"
   , indent 8 "    for y in x.values(): visit(y)"
   , indent 4 "visit(obj)"
   , indent 4 "return refs"
   , ""
   , "def _spatial_tuple(x, rank, default):"
   , indent 4 "if x is None: return [int(default)] * rank"
   , indent 4 "if isinstance(x, int): return [int(x)] * rank"
   , indent 4 "if isinstance(x, (tuple, list)) and len(x) == rank: return [int(v) for v in x]"
   , indent 4 ("raise NotImplementedError(f\"expected an integer or length-{rank} spatial " ++
       "tuple, got {x!r}\")")
   , "" ]

/-- Parameter payload serialization for linear, attention, and convolution nodes. -/
def payload : Array String :=
  #[ "def _tensor_values(value):"
   , indent 4 "return value.detach().cpu().tolist()"
   , ""
   , "_TENSOR_VALUES = {}"
   , ""
   , "def _resolve_tensor(value):"
   , indent 4 "if isinstance(value, torch.Tensor): return value"
   , indent 4 ("if isinstance(value, torch.fx.Node) and value.name in _TENSOR_VALUES: " ++
       "return _TENSOR_VALUES[value.name]")
   , indent 4 ("raise NotImplementedError(f\"could not resolve tensor argument {value!r} into " ++
       "the exported payload\")")
   , ""
   , "def _linear_payload(weight, bias):"
   , indent 4 "weight = _resolve_tensor(weight)"
   , indent 4 "if weight.ndim != 2:"
   , indent 8 ("raise NotImplementedError(f\"linear weight must have rank 2, got shape " ++
       "{tuple(weight.shape)}\")")
   , indent 4 "out_dim, in_dim = (int(value) for value in weight.shape)"
   , indent 4 "bias = _resolve_tensor(bias) if bias is not None else weight.new_zeros(out_dim)"
   , indent 4 "if tuple(bias.shape) != (out_dim,):"
   , indent 8 ("raise NotImplementedError(f\"linear bias must have shape {(out_dim,)}, got " ++
       "{tuple(bias.shape)}\")")
   , indent 4 ("return {\"out_dim\": out_dim, \"in_dim\": in_dim, \"weight\": " ++
       "_tensor_values(weight), \"bias\": _tensor_values(bias)}")
   , ""
   , "def _multihead_attention_payload(mod):"
   , indent 4 "if mod.in_proj_weight is None:"
   , indent 8 ("raise NotImplementedError(\"nn.MultiheadAttention with separate q/k/v " ++
       "projection weights is outside the current TorchLean IR import subset\")")
   , indent 4 "embed_dim = int(mod.embed_dim)"
   , indent 4 "if tuple(mod.in_proj_weight.shape) != (3 * embed_dim, embed_dim):"
   , indent 8 ("raise NotImplementedError(f\"unexpected nn.MultiheadAttention in_proj_weight " ++
       "shape {tuple(mod.in_proj_weight.shape)}\")")
   , indent 4 "q_weight, k_weight, v_weight = mod.in_proj_weight.chunk(3, dim=0)"
   , indent 4 "if mod.in_proj_bias is None:"
   , indent 8 "q_bias = k_bias = v_bias = mod.in_proj_weight.new_zeros(embed_dim)"
   , indent 4 "else:"
   , indent 8 "if tuple(mod.in_proj_bias.shape) != (3 * embed_dim,):"
   , indent 8 ("    raise NotImplementedError(f\"unexpected nn.MultiheadAttention " ++
       "in_proj_bias shape {tuple(mod.in_proj_bias.shape)}\")")
   , indent 8 "q_bias, k_bias, v_bias = mod.in_proj_bias.chunk(3, dim=0)"
   , indent 4 "if tuple(mod.out_proj.weight.shape) != (embed_dim, embed_dim):"
   , indent 8 ("raise NotImplementedError(f\"unexpected nn.MultiheadAttention out_proj " ++
       "weight shape {tuple(mod.out_proj.weight.shape)}\")")
   , indent 4 ("out_bias = mod.out_proj.bias if mod.out_proj.bias is not None else " ++
       "mod.out_proj.weight.new_zeros(embed_dim)")
   , indent 4 "return {"
   , indent 8 "\"q_weight\": _tensor_values(q_weight), \"q_bias\": _tensor_values(q_bias),"
   , indent 8 "\"k_weight\": _tensor_values(k_weight), \"k_bias\": _tensor_values(k_bias),"
   , indent 8 "\"v_weight\": _tensor_values(v_weight), \"v_bias\": _tensor_values(v_bias),"
   , indent 8 ("\"out_weight\": _tensor_values(mod.out_proj.weight), \"out_bias\": " ++
       "_tensor_values(out_bias),")
   , indent 8 "\"scale\": float((embed_dim // int(mod.num_heads)) ** -0.5),"
   , indent 4 "}"
   , ""
   , "def _convolution_payload(weight, bias, groups):"
   , indent 4 "weight = _resolve_tensor(weight)"
   , indent 4 "groups = int(groups)"
   , indent 4 "if weight.ndim < 3:"
   , indent 8 ("raise NotImplementedError(f\"convolution weight must have rank at least 3, " ++
       "got shape {tuple(weight.shape)}\")")
   , indent 4 "if groups <= 0:"
   , indent 8 ("raise NotImplementedError(f\"convolution groups must be positive, got " ++
       "{groups}\")")
   , indent 4 "out_channels = int(weight.shape[0])"
   , indent 4 "in_channels_per_group = int(weight.shape[1])"
   , indent 4 "if out_channels % groups != 0:"
   , indent 8 ("raise NotImplementedError(f\"convolution out_channels={out_channels} is not " ++
       "divisible by groups={groups}\")")
   , indent 4 "in_channels = in_channels_per_group * groups"
   , indent 4 "if groups == 1:"
   , indent 8 "dense_weight = weight"
   , indent 4 "else:"
   , indent 8 "out_channels_per_group = out_channels // groups"
   , indent 8 "dense_weight = weight.new_zeros((out_channels, in_channels, *weight.shape[2:]))"
   , indent 8 "for group in range(groups):"
   , indent 8 "    out_start = group * out_channels_per_group"
   , indent 8 "    out_end = out_start + out_channels_per_group"
   , indent 8 "    in_start = group * in_channels_per_group"
   , indent 8 "    in_end = in_start + in_channels_per_group"
   , indent 8 "    dense_weight[out_start:out_end, in_start:in_end] = weight[out_start:out_end]"
   , indent 4 ("bias = _resolve_tensor(bias) if bias is not None else " ++
       "weight.new_zeros(out_channels)")
   , indent 4 "if tuple(bias.shape) != (out_channels,):"
   , indent 8 ("raise NotImplementedError(f\"convolution bias must have shape " ++
       "{(out_channels,)}, got {tuple(bias.shape)}\")")
   , indent 4 ("return {\"in_channels\": in_channels, \"out_channels\": out_channels, " ++
       "\"weight\": _tensor_values(dense_weight), \"bias\": _tensor_values(bias)}")
   , "" ]

/-- Flatten-versus-reshape selection and axis normalization. -/
def axes : Array String :=
  #[ "def _flatten_kind(input_shape, output_shape):"
   , indent 4 "if len(output_shape) == 1:"
   , indent 8 ("return {" ++ Wire.field .flatten ++ ", \"value_shape\": input_shape}")
   , indent 4 ("return {" ++ Wire.field .reshape ++ ", \"in_shape\": input_shape, " ++
       "\"out_shape\": output_shape}")
   , ""
   , "def _normalize_axis(axis, rank):"
   , indent 4 "axis = int(axis)"
   , indent 4 "axis = axis + rank if axis < 0 else axis"
   , indent 4 "if axis < 0 or axis >= rank:"
   , indent 8 "raise NotImplementedError(f\"axis {axis} is outside rank {rank}\")"
   , indent 4 "return axis"
   , ""
   , "def _reduction_axes(axis, rank, op_name):"
   , indent 4 ("raw = range(rank) if axis is None else axis if isinstance(axis, (tuple, list))" ++
       " else (axis,)")
   , indent 4 "axes = [_normalize_axis(value, rank) for value in raw]"
   , indent 4 "if len(set(axes)) != len(axes):"
   , indent 8 "raise NotImplementedError(f\"{op_name} has duplicate reduction axes {axes}\")"
   , indent 4 "return axes"
   , "" ]

/-- Opening of `_lower_kind`: target name, argument, and axis extraction shared by all rules. -/
def header : Array String :=
  #[ "def _lower_kind(node, model=None):"
   , indent 4 "name = _target_name(node.target)"
   , indent 4 "args = node.args"
   , indent 4 "kwargs = dict(node.kwargs)"
   , indent 4 "if kwargs.get(\"out\", None) is not None:"
   , indent 8 "raise NotImplementedError(\"out= mutation is outside the pure graph contract\")"
   , indent 4 "axis = kwargs.get(\"dim\", kwargs.get(\"axis\", None))"
   , indent 4 "if axis is None and len(args) > 1: axis = args[1]" ]

/-- `_lower_kind` rules for `call_module` nodes, keyed on the `nn.Module` subclass. -/
def modules : Array String :=
  #[ indent 4 "if node.op == \"call_module\" and model is not None:"
   , indent 8 "mod = model.get_submodule(str(node.target))"
   , indent 8 "if isinstance(mod, nn.Linear):"
   , indent 8 ("    return {" ++ Wire.field .linear ++
       ", **_linear_payload(mod.weight, mod.bias)}")
   , indent 8 "if isinstance(mod, nn.ReLU):"
   , indent 8 ("    if mod.inplace: raise NotImplementedError(\"in-place nn.ReLU mutation is " ++
       "outside the pure TorchLean graph contract\")")
   , indent 8 ("    return " ++ Wire.object .relu)
   , indent 8 ("if isinstance(mod, nn.Tanh): return " ++ Wire.object .tanh)
   , indent 8 ("if isinstance(mod, nn.Sigmoid): return " ++ Wire.object .sigmoid)
   , indent 8 "if isinstance(mod, nn.Softmax):"
   , indent 8 ("    return {" ++ Wire.field .softmax ++
       ", \"axis\": _normalize_axis(mod.dim, len(_node_shape(node)))}")
   , indent 8 "if isinstance(mod, nn.Flatten):"
   , indent 8 ("    return _flatten_kind(_shape_from_arg(args[0]) if args else [], " ++
       "_node_shape(node))")
   , indent 8 "if isinstance(mod, nn.LayerNorm):"
   , indent 8 "    rank = len(_node_shape(node))"
   , indent 8 "    norm_rank = len(tuple(mod.normalized_shape))"
   , indent 8 ("    gamma = mod.weight if mod.weight is not None else " ++
       "torch.ones(mod.normalized_shape)")
   , indent 8 ("    beta = mod.bias if mod.bias is not None else " ++
       "torch.zeros(mod.normalized_shape)")
   , indent 8 ("    return {" ++ Wire.field .layernorm ++
       ", \"axis\": max(0, rank - norm_rank), \"eps\": float(mod.eps), " ++
       "\"gamma\": _tensor_values(gamma), \"beta\": _tensor_values(beta)}")
   , indent 8 "if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):"
   , indent 8 "    rank = len(tuple(mod.kernel_size))"
   , indent 8 "    input_rank = len(_shape_from_arg(args[0])) if args else 0"
   , indent 8 ("    if input_rank not in (rank + 1, rank + 2): raise NotImplementedError(" ++
       "f\"convolution expected rank {rank + 1} or {rank + 2}, got {input_rank}\")")
   , indent 8 "    channel_axis = input_rank - rank - 1"
   , indent 8 ("    if mod.padding_mode != \"zeros\": raise NotImplementedError(" ++
       "f\"convolution padding_mode={mod.padding_mode!r} is outside the current TorchLean IR " ++
       "import subset\")")
   , indent 8 "    padding = _spatial_tuple(mod.padding, rank, 0)"
   , indent 8 "    input_shape = _shape_from_arg(args[0])"
   , indent 8 ("    return {" ++ Wire.field .conv ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": _spatial_tuple(mod.kernel_size, rank, 1), " ++
       "\"stride\": _spatial_tuple(mod.stride, rank, 1), \"padding\": padding, " ++
       "\"padding_after\": padding, \"dilation\": _spatial_tuple(mod.dilation, rank, 1), " ++
       "\"groups\": int(mod.groups), \"channel_axis\": channel_axis, " ++
       "\"input_spatial\": input_shape[channel_axis + 1:], " ++
       "**_convolution_payload(mod.weight, mod.bias, mod.groups)}")
   , indent 8 "if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):"
   , indent 8 "    if mod.training:"
   , indent 8 ("        raise NotImplementedError(\"training-mode BatchNorm needs batch " ++
       "statistics and running-statistic updates\")")
   , indent 8 ("    if not mod.track_running_stats or mod.running_mean is None or " ++
       "mod.running_var is None:")
   , indent 8 ("        raise NotImplementedError(\"nn.BatchNorm2d without running statistics" ++
       " has batch-dependent semantics, not TorchLean eval-mode BatchNorm semantics\")")
   , indent 8 ("    gamma = mod.weight if mod.weight is not None else " ++
       "torch.ones(mod.num_features)")
   , indent 8 ("    beta = mod.bias if mod.bias is not None else " ++
       "torch.zeros(mod.num_features)")
   , indent 8 ("    return {" ++ Wire.field .batchNormEval ++ ", \"channel_axis\": 1, " ++
       "\"channels\": int(mod.num_features), \"eps\": float(mod.eps), " ++
       "\"gamma\": _tensor_values(gamma), \"beta\": _tensor_values(beta), " ++
       "\"mean\": _tensor_values(mod.running_mean), \"var\": _tensor_values(mod.running_var)}")
   , indent 8 "if isinstance(mod, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):"
   , indent 8 ("    rank = 1 if isinstance(mod, nn.MaxPool1d) else 2 if isinstance(mod, " ++
       "nn.MaxPool2d) else 3")
   , indent 8 ("    if _spatial_tuple(mod.dilation, rank, 1) != [1] * rank: raise " ++
       "NotImplementedError(\"dilated max pooling is outside the current TorchLean IR import " ++
       "subset\")")
   , indent 8 ("    if mod.ceil_mode: raise NotImplementedError(\"max pooling with " ++
       "ceil_mode=True is outside the current TorchLean IR import subset\")")
   , indent 8 ("    if mod.return_indices: raise NotImplementedError(\"max pooling with " ++
       "return_indices=True produces a tuple outside the current pooling lowering\")")
   , indent 8 ("    return {" ++ Wire.field .maxPool ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": _spatial_tuple(mod.kernel_size, rank, 1), \"stride\": _spatial_tuple(" ++
       "mod.stride if mod.stride is not None else mod.kernel_size, rank, 1), " ++
       "\"padding\": _spatial_tuple(mod.padding, rank, 0)}")
   , indent 8 "if isinstance(mod, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):"
   , indent 8 ("    rank = 1 if isinstance(mod, nn.AvgPool1d) else 2 if isinstance(mod, " ++
       "nn.AvgPool2d) else 3")
   , indent 8 ("    if mod.ceil_mode: raise NotImplementedError(\"average pooling with " ++
       "ceil_mode=True is outside the current TorchLean IR import subset\")")
   , indent 8 ("    if getattr(mod, 'divisor_override', None) is not None: raise " ++
       "NotImplementedError(\"average pooling with divisor_override is outside the current " ++
       "TorchLean IR import subset\")")
   , indent 8 "    padding = _spatial_tuple(mod.padding, rank, 0)"
   , indent 8 ("    if any(padding) and not mod.count_include_pad: raise " ++
       "NotImplementedError(\"padded average pooling with count_include_pad=False is outside " ++
       "the current TorchLean IR import subset\")")
   , indent 8 ("    return {" ++ Wire.field .avgPool ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": _spatial_tuple(mod.kernel_size, rank, 1), \"stride\": _spatial_tuple(" ++
       "mod.stride if mod.stride is not None else mod.kernel_size, rank, 1), " ++
       "\"padding\": padding}") ]

/-- Reject scalar operands until they have explicit constant/broadcast lowering. -/
def operands : Array String :=
  #[ "        operands = (args[0] if args else kwargs.get(\"input\"),"
   , "                    args[1] if len(args) > 1 else kwargs.get(\"other\"))"
   , "        if not all(isinstance(x, torch.fx.Node) for x in operands):"
   , "            raise NotImplementedError(\"scalar operands need constant/broadcast lowering\")" ]

/-- `_lower_kind` rules for elementwise `aten`/functional targets that carry no payload. -/
def elementwise : Array String :=
  #[ "    if _is_op(node, (\"aten.add.Tensor\",), (operator.add, torch.add), (\"add\",)):" ] ++
    operands ++
  #[ "        alpha = kwargs.get(\"alpha\", args[2] if len(args) > 2 else 1)"
   , "        if alpha != 1:"
   , "            raise NotImplementedError(\"torch.add with alpha != 1 is unsupported\")"
   , "        return {\"kind\": " ++ Wire.quoted .add ++ "}"
   , "    if _is_op(node, (\"aten.sub.Tensor\",), " ++
       "(operator.sub, torch.sub, torch.subtract), (\"sub\", \"subtract\")):" ] ++
    operands ++
  #[ "        alpha = kwargs.get(\"alpha\", args[2] if len(args) > 2 else 1)"
   , "        if alpha != 1:"
   , "            raise NotImplementedError(\"torch.sub with alpha != 1 is unsupported\")"
   , "        return {\"kind\": " ++ Wire.quoted .sub ++ "}"
   , "    if _is_op(node, (\"aten.mul.Tensor\",), " ++
       "(operator.mul, torch.mul, torch.multiply), (\"mul\", \"multiply\")):" ] ++
    operands ++
  #[ "        return {\"kind\": " ++ Wire.quoted .mulElem ++ "}"
   , "    if _is_op(node, (\"aten.relu.default\",), (torch.relu, F.relu), (\"relu\",)):"
   , "        inplace = kwargs.get(\"inplace\", args[1] if len(args) > 1 else False)"
   , "        if inplace: raise " ++ "NotImplementedError(\"in-place relu is outside the " ++
       "pure graph contract\")"
   , "        return {\"kind\": " ++ Wire.quoted .relu ++ "}"
   , "    if _is_op(node, (\"aten.tanh.default\",), (torch.tanh,), (\"tanh\",)):"
   , "        return {\"kind\": " ++ Wire.quoted .tanh ++ "}"
   , "    if _is_op(node, (\"aten.sigmoid.default\",), (torch.sigmoid,), (\"sigmoid\",)):"
   , "        return {\"kind\": " ++ Wire.quoted .sigmoid ++ "}"
   , "    if _is_op(node, (\"aten.exp.default\",), (torch.exp,), (\"exp\",)):"
   , "        return {\"kind\": " ++ Wire.quoted .exp ++ "}"
   , "    if _is_op(node, (\"aten.log.default\",), (torch.log,), (\"log\",)):"
   , "        return {\"kind\": " ++ Wire.quoted .log ++ "}"
   , "    if _is_op(node, (\"aten.sin.default\",), (torch.sin,), (\"sin\",)):"
   , "        return {\"kind\": " ++ Wire.quoted .sin ++ "}"
   , "    if _is_op(node, (\"aten.cos.default\",), (torch.cos,), (\"cos\",)):"
   , "        return {\"kind\": " ++ Wire.quoted .cos ++ "}"
   , "    if _is_op(node, (\"aten.abs.default\",), " ++
       "(torch.abs, torch.absolute), (\"abs\", \"absolute\")):"
   , "        return {\"kind\": " ++ Wire.quoted .abs ++ "}"
   , "    if _is_op(node, (\"aten.sqrt.default\",), (torch.sqrt,), (\"sqrt\",)):"
   , "        return {\"kind\": " ++ Wire.quoted .sqrt ++ "}"
   , "    if _is_op(node, (\"aten.reciprocal.default\",), (torch.reciprocal,), (\"reciprocal\",)):"
   , "        return {\"kind\": " ++ Wire.quoted .inv ++ "}"
   , "    if _is_op(node, (\"aten.maximum.default\",), (torch.maximum,), (\"maximum\",)):"
   , "        return {\"kind\": " ++ Wire.quoted .maxElem ++ "}"
   , "    if _is_op(node, (\"aten.minimum.default\",), (torch.minimum,), (\"minimum\",)):"
   , "        return {\"kind\": " ++ Wire.quoted .minElem ++ "}"
   , "    if _is_op(node, (\"aten.matmul.default\", " ++
       "\"aten.mm.default\"), (torch.matmul, torch.mm, " ++
       "operator.matmul), (\"matmul\", \"mm\")):"
   , "        return {\"kind\": " ++ Wire.quoted .matmul ++ "}" ]

/-- `_lower_kind` rules for reductions, softmax, and shape operations. -/
def tensorOps : Array String :=
  #[ "    is_sum = _is_op(node, (\"aten.sum.default\", " ++
       "\"aten.sum.dim_IntList\"), (torch.sum,), (\"sum\",))"
   , "    if is_sum or _is_op(node, " ++ "(\"aten.mean.default\", \"aten.mean.dim\"), " ++
       "(torch.mean,), (\"mean\",)):"
   , "        input_rank = len(_shape_from_arg(args[0])) if args else len(_node_shape(node))"
   , "        reduction_axes = _reduction_axes(axis, input_rank, name)"
   , "        keepdim = bool(kwargs.get(\"keepdim\", args[2] if len(args) > 2 else False))"
   , "        dtype = kwargs.get(\"dtype\", args[3] if len(args) > 3 else None)"
   , "        if dtype is not None:"
   , "            raise NotImplementedError(\"reduction " ++
       "dtype casts are outside the scalar contract\")"
   , "        return {\"kind\": " ++ Wire.quoted .reduceSum ++ " if is_sum else " ++
       Wire.quoted .reduceMean ++ ","
   , "                \"axes\": reduction_axes, \"keepdim\": keepdim}"
   , "    if _is_op(node, (\"aten.softmax.int\", " ++
       "\"aten._softmax.default\"), (torch.softmax, " ++ "F.softmax), (\"softmax\",)):"
   , "        if _is_op(node, (\"aten._softmax.default\",)):"
   , "            if kwargs.get(\"half_to_float\", args[2] if len(args) > 2 else False):"
   , "                raise NotImplementedError(\"softmax half_to_float changes scalar semantics\")"
   , "            dtype = None"
   , "        else:"
   , "            dtype_pos = 3 if node.target is F.softmax else 2"
   , "            dtype = kwargs.get(\"dtype\", args[dtype_pos] if len(args) > dtype_pos else None)"
   , "        if dtype is not None:"
   , "            raise NotImplementedError(\"softmax " ++
       "dtype casts are outside the scalar contract\")"
   , "        rank = len(_node_shape(node))"
   , "        if axis is None and node.target is F.softmax:"
   , "            axis = 0 if rank in (0, 1, 3) else 1"
   , "        if axis is None: raise NotImplementedError(\"softmax requires a static dimension\")"
   , "        return {\"kind\": " ++ Wire.quoted .softmax ++
       ", \"axis\": _normalize_axis(axis, rank)}"
   , "    if _is_op(node, (\"aten.reshape.default\", " ++
       "\"aten.view.default\"), (torch.reshape,), (\"reshape\", \"view\")):"
   , "        if node.op == \"call_method\" and node.target == \"view\":"
   , "            if isinstance(kwargs.get(\"dtype\", " ++
       "args[1] if len(args) > 1 else None), torch.dtype):"
   , "                raise NotImplementedError(\"view(dtype) changes scalar semantics\")"
   , "        return {\"kind\": " ++ Wire.quoted .reshape ++
       ", \"in_shape\": _shape_from_arg(args[0]),"
   , "                \"out_shape\": _node_shape(node)}"
   , "    if _is_op(node, (\"aten.contiguous.default\",), (), (\"contiguous\",)):"
   , "        return {\"kind\": " ++ Wire.quoted .reshape ++
       ", \"in_shape\": _shape_from_arg(args[0]),"
   , "                \"out_shape\": _node_shape(node)}"
   , "    if _is_op(node, (\"aten.flatten.using_ints\",), (torch.flatten,), (\"flatten\",)):"
   , "        return _flatten_kind(_shape_from_arg(args[0]), _node_shape(node))"
   , "    if _is_op(node, (\"aten.permute.default\",), (torch.permute,), (\"permute\",)):"
   , "        dims = kwargs.get(\"dims\", args[1] if len(args) > 1 else ())"
   , "        perm = list(dims) if isinstance(dims, (tuple, list)) else list(args[1:])"
   , "        rank = len(_node_shape(node))"
   , "        return {\"kind\": " ++ Wire.quoted .permute ++
       ", \"perm\": [_normalize_axis(x, rank) for x in perm]}"
   , "    if _is_op(node, (\"aten.cat.default\",), " ++
       "(torch.cat, torch.concat, torch.concatenate), ()):"
   , "        rank = len(_node_shape(node))"
   , "        dim = _normalize_axis(kwargs.get(\"dim\", args[1] if len(args) > 1 else 0), rank)"
   , "        return {\"kind\": " ++ Wire.quoted .concat ++ ", \"axis\": dim}"
   , "    if _is_op(node, (\"aten.transpose.int\",), (torch.transpose,), (\"transpose\",)):"
   , "        rank = len(_shape_from_arg(args[0]))"
   , "        d0 = _normalize_axis(kwargs.get(\"dim0\", args[1] if len(args) > 1 else None), rank)"
   , "        d1 = _normalize_axis(kwargs.get(\"dim1\", args[2] if len(args) > 2 else None), rank)"
   , "        return {\"kind\": " ++ Wire.quoted .transpose ++
       ", \"axis1\": d0, \"axis2\": d1}" ]

/-- `_lower_kind` rules for functional layers with payloads: layer norm, linear, batch norm. -/
def layers : Array String :=
  #[ indent 4 ("if _is_op(node, (\"aten.layer_norm.default\",), " ++
       "(F.layer_norm, torch.layer_norm), ()):")
   , indent 8 "weight = kwargs.get(\"weight\", args[2] if len(args) > 2 else None)"
   , indent 8 "bias = kwargs.get(\"bias\", args[3] if len(args) > 3 else None)"
   , indent 8 "eps = kwargs.get(\"eps\", args[4] if len(args) > 4 else 1e-5)"
   , indent 8 "input_rank = len(_shape_from_arg(args[0])) if args else len(_node_shape(node))"
   , indent 8 ("normalized_shape = args[1] if len(args) > 1 else " ++
       "kwargs.get(\"normalized_shape\", ())")
   , indent 8 ("normalized_rank = len(normalized_shape) if isinstance(normalized_shape, " ++
       "(tuple, list, torch.Size)) else 1")
   , indent 8 ("weight = _resolve_tensor(weight) if weight is not None else " ++
       "torch.ones(normalized_shape)")
   , indent 8 ("bias = _resolve_tensor(bias) if bias is not None else " ++
       "torch.zeros(normalized_shape)")
   , indent 8 ("return {" ++ Wire.field .layernorm ++
       ", \"axis\": max(0, input_rank - normalized_rank), \"eps\": float(eps), " ++
       "\"gamma\": _tensor_values(weight), \"beta\": _tensor_values(bias)}")
   , indent 4 "if _is_op(node, (\"aten.linear.default\",), (F.linear,), ()):"
   , indent 8 "weight = kwargs.get(\"weight\", args[1] if len(args) > 1 else None)"
   , indent 8 "bias = kwargs.get(\"bias\", args[2] if len(args) > 2 else None)"
   , indent 8 ("return {" ++ Wire.field .linear ++ ", **_linear_payload(weight, bias)}")
   , indent 4 ("if _is_op(node, (\"aten.batch_norm.default\",), " ++
       "(F.batch_norm, torch.batch_norm), ()):")
   , indent 8 "training = bool(kwargs.get(\"training\", args[5] if len(args) > 5 else False))"
   , indent 8 ("if training: raise NotImplementedError(\"training-mode batch_norm is outside " ++
       "the eval-mode TorchLean IR operation\")")
   , indent 8 "is_aten = node.target is not F.batch_norm"
   , indent 8 "mean_pos, var_pos = (3, 4) if is_aten else (1, 2)"
   , indent 8 "weight_pos, bias_pos = (1, 2) if is_aten else (3, 4)"
   , indent 8 ("running_mean = kwargs.get(\"running_mean\", args[mean_pos] if len(args) > " ++
       "mean_pos else None)")
   , indent 8 ("running_var = kwargs.get(\"running_var\", args[var_pos] if len(args) > " ++
       "var_pos else None)")
   , indent 8 ("if running_mean is None or running_var is None: raise NotImplementedError(" ++
       "\"batch_norm without running statistics is outside the eval-mode TorchLean IR operation\")")
   , indent 8 "shape = _node_shape(node)"
   , indent 8 "channels = int(shape[1]) if len(shape) >= 2 else 0"
   , indent 8 "eps = float(kwargs.get(\"eps\", args[7] if len(args) > 7 else 1e-5))"
   , indent 8 "running_mean = _resolve_tensor(running_mean)"
   , indent 8 "running_var = _resolve_tensor(running_var)"
   , indent 8 ("weight = kwargs.get(\"weight\", args[weight_pos] if len(args) > weight_pos " ++
       "else None)")
   , indent 8 "bias = kwargs.get(\"bias\", args[bias_pos] if len(args) > bias_pos else None)"
   , indent 8 "weight = _resolve_tensor(weight) if weight is not None else torch.ones(channels)"
   , indent 8 "bias = _resolve_tensor(bias) if bias is not None else torch.zeros(channels)"
   , indent 8 ("return {" ++ Wire.field .batchNormEval ++ ", \"channel_axis\": 1, " ++
       "\"channels\": channels, \"eps\": eps, \"gamma\": _tensor_values(weight), " ++
       "\"beta\": _tensor_values(bias), \"mean\": _tensor_values(running_mean), " ++
       "\"var\": _tensor_values(running_var)}") ]

/-- `_lower_kind` rules for functional convolution and pooling, then the tuple/fallback cases. -/
def spatial : Array String :=
  #[ indent 4 ("if _is_op(node, (\"aten.conv1d.default\", " ++
       "\"aten.conv2d.default\", \"aten.conv3d.default\"), " ++
       "(F.conv1d, F.conv2d, F.conv3d), ()):")
   , indent 8 "weight_arg = kwargs.get(\"weight\", args[1] if len(args) > 1 else None)"
   , indent 8 "bias_arg = kwargs.get(\"bias\", args[2] if len(args) > 2 else None)"
   , indent 8 "weight = _resolve_tensor(weight_arg)"
   , indent 8 "wshape = _shape_of(weight)"
   , indent 8 "rank = max(0, len(wshape) - 2)"
   , indent 8 "input_rank = len(_shape_from_arg(args[0])) if args else 0"
   , indent 8 ("if input_rank not in (rank + 1, rank + 2): raise NotImplementedError(" ++
       "f\"convolution expected rank {rank + 1} or {rank + 2}, got {input_rank}\")")
   , indent 8 "channel_axis = input_rank - rank - 1"
   , indent 8 ("stride = _spatial_tuple(kwargs.get(\"stride\", args[3] if len(args) > 3 else " ++
       "1), rank, 1)")
   , indent 8 ("padding = _spatial_tuple(kwargs.get(\"padding\", args[4] if len(args) > 4 " ++
       "else 0), rank, 0)")
   , indent 8 ("dilation = _spatial_tuple(kwargs.get(\"dilation\", args[5] if len(args) > 5 " ++
       "else 1), rank, 1)")
   , indent 8 "groups = int(kwargs.get(\"groups\", args[6] if len(args) > 6 else 1))"
   , indent 8 "input_shape = _shape_from_arg(args[0])"
   , indent 8 ("return {" ++ Wire.field .conv ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": [int(v) for v in wshape[2:]], \"stride\": stride, \"padding\": padding, " ++
       "\"padding_after\": padding, \"dilation\": dilation, \"groups\": groups, " ++
       "\"channel_axis\": channel_axis, \"input_spatial\": input_shape[channel_axis + 1:], " ++
       "**_convolution_payload(weight, bias_arg, groups)}")
   , indent 4 ("if _is_op(node, (\"aten.max_pool1d.default\", " ++
       "\"aten.max_pool2d.default\", " ++ "\"aten.max_pool3d.default\"), (F.max_pool1d, " ++
       "F.max_pool2d, F.max_pool3d), ()):")
   , indent 8 ("rank = next(d for d in (1, 2, 3) if _is_op(node, " ++
       "(f\"aten.max_pool{d}d.default\",), (getattr(F, " ++ "f\"max_pool{d}d\"),)))")
   , indent 8 ("kernel = _spatial_tuple(args[1] if len(args) > 1 else " ++
       "kwargs.get(\"kernel_size\", 1), rank, 1)")
   , indent 8 "stride_arg = kwargs.get(\"stride\", args[2] if len(args) > 2 else None)"
   , indent 8 ("stride = _spatial_tuple(stride_arg if stride_arg not in (None, []) else " ++
       "kernel, rank, 1)")
   , indent 8 ("padding = _spatial_tuple(kwargs.get(\"padding\", args[3] if len(args) > 3 " ++
       "else 0), rank, 0)")
   , indent 8 ("dilation = _spatial_tuple(kwargs.get(\"dilation\", args[4] if len(args) > 4 " ++
       "else 1), rank, 1)")
   , indent 8 "ceil_mode = bool(kwargs.get(\"ceil_mode\", args[5] if len(args) > 5 else False))"
   , indent 8 "if kwargs.get(\"return_indices\", args[6] if len(args) > 6 else False):"
   , indent 8 "    raise NotImplementedError(\"max pooling indices need tuple lowering\")"
   , indent 8 ("if dilation != [1] * rank: raise NotImplementedError(\"dilated max pooling is" ++
       " outside the current TorchLean IR import subset\")")
   , indent 8 ("if ceil_mode: raise NotImplementedError(\"max pooling with ceil_mode=True is " ++
       "outside the current TorchLean IR import subset\")")
   , indent 8 ("return {" ++ Wire.field .maxPool ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": kernel, \"stride\": stride, \"padding\": padding}")
   , indent 4 ("if _is_op(node, (\"aten.avg_pool1d.default\", " ++
       "\"aten.avg_pool2d.default\", " ++ "\"aten.avg_pool3d.default\"), (F.avg_pool1d, " ++
       "F.avg_pool2d, F.avg_pool3d), ()):")
   , indent 8 ("rank = next(d for d in (1, 2, 3) if _is_op(node, " ++
       "(f\"aten.avg_pool{d}d.default\",), (getattr(F, " ++ "f\"avg_pool{d}d\"),)))")
   , indent 8 ("kernel = _spatial_tuple(args[1] if len(args) > 1 else " ++
       "kwargs.get(\"kernel_size\", 1), rank, 1)")
   , indent 8 "stride_arg = kwargs.get(\"stride\", args[2] if len(args) > 2 else None)"
   , indent 8 ("stride = _spatial_tuple(stride_arg if stride_arg not in (None, []) else " ++
       "kernel, rank, 1)")
   , indent 8 ("padding = _spatial_tuple(kwargs.get(\"padding\", args[3] if len(args) > 3 " ++
       "else 0), rank, 0)")
   , indent 8 "ceil_mode = bool(kwargs.get(\"ceil_mode\", args[4] if len(args) > 4 else False))"
   , indent 8 ("count_include_pad = bool(kwargs.get(\"count_include_pad\", args[5] if " ++
       "len(args) > 5 else True))")
   , indent 8 ("divisor_override = kwargs.get(\"divisor_override\", args[6] if len(args) > 6 " ++
       "else None)")
   , indent 8 ("if ceil_mode: raise NotImplementedError(\"average pooling with ceil_mode=True" ++
       " is outside the current TorchLean IR import subset\")")
   , indent 8 ("if any(padding) and not count_include_pad: raise NotImplementedError(\"padded" ++
       " average pooling with count_include_pad=False is outside the current TorchLean IR " ++
       "import subset\")")
   , indent 8 ("if divisor_override is not None: raise NotImplementedError(\"average pooling " ++
       "with divisor_override is outside the current TorchLean IR import subset\")")
   , indent 8 ("return {" ++ Wire.field .avgPool ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": kernel, \"stride\": stride, \"padding\": padding}")
   , indent 4 "if _is_getitem(node):"
   , indent 8 "shapes = _tuple_shapes_of_node(args[0]) if args else []"
   , indent 8 "if not shapes or len(args) < 2 or type(args[1]) is not int:"
   , indent 8 "    raise NotImplementedError(\"getitem requires a tuple and a static integer\")"
   , indent 8 "idx = args[1] if args[1] >= 0 else len(shapes) + args[1]"
   , indent 8 "if not 0 <= idx < len(shapes):"
   , indent 8 "    raise NotImplementedError(\"tuple index is out of range\")"
   , indent 8 ("return {\"kind\": \"" ++ Wire.projection ++ "\", \"index\": idx}")
   , indent 4 "if _is_getattr(node):"
   , indent 8 "attr = args[1] if len(args) > 1 else \"<unknown>\""
   , indent 8 ("raise NotImplementedError(f\"unsupported PyTorch attribute projection:" ++
       " {attr}. If this came from a tuple-returning op such as torch.sort(...).values, add a " ++
       "value-graph/lowering rule for that producer instead of treating the attribute as an " ++
       "ordinary tensor op.\")")
   , indent 4 "raise NotImplementedError(f\"unsupported PyTorch op: {name}\")"
   , "" ]

/-- The complete `_lower_kind` function. -/
def lower : Array String :=
  header ++ modules ++ elementwise ++ tensorOps ++
    layers ++ spatial

/-- Check captured operations before dropping unused values.

FX records `x.relu_(); return x` with an unused result for the ReLU call. Skipping that call before
checking its contract would erase the update to `x`. Unsupported calls therefore fail even when
their results have no users. -/
def coverage : Array String :=
  #[ "def _check_graph_operators(graph, model):"
   , indent 4 "for node in graph.nodes:"
   , indent 8 "if node.op in (\"placeholder\", \"get_attr\", \"output\"):"
   , indent 8 "    continue"
   , indent 8 "if _tuple_shapes_of_node(node):"
   , indent 8 "    _tuple_kind(node, model)"
   , indent 8 "elif node.op in (\"call_function\", \"call_method\", \"call_module\"):"
   , indent 8 "    _lower_kind(node, model)"
   , indent 8 "else:"
   , indent 8 "    raise NotImplementedError(f\"unsupported FX node op: {node.op}\")"
   , ""
   , "def _graph_is_lowerable(graph, model):"
   , indent 4 "try:"
   , indent 8 "_check_graph_operators(graph, model)"
   , indent 4 "except NotImplementedError:"
   , indent 8 "return False"
   , indent 4 "return True"
   , "" ]

/-- Graph capture: `torch.export` when preferred and lowerable, otherwise FX symbolic tracing. -/
def capture (options : Options) : Array String :=
  #[ "def _capture(model, example, require_torch_export=False):"
   , indent 4 "global _TENSOR_VALUES"
   , indent 4 "_TENSOR_VALUES = {}"
   , indent 4 "if " ++ boolLiteral options.preferTorchExport ++ ":"
   , indent 8 "try:"
   , indent 8 "    ep = torch.export.export(model, (example,))"
   , indent 8 "    # Functionalization can turn mutation into ordinary tensor operations."
   , indent 8 "    # Extra outputs describe state updates that need their own replay rules."
   , indent 8 ("    if any(spec.kind.name != \"USER_OUTPUT\" for " ++
       "spec in ep.graph_signature.output_specs):")
   , indent 8 ("        raise NotImplementedError(\"exported " ++
       "mutation/effects are outside the pure graph contract\")")
   , indent 8 "    # ExportedProgram lifts parameters, buffers, and constants into graph"
   , indent 8 "    # placeholders. Their order is not a user-input contract, so classify"
   , indent 8 "    # placeholders through graph_signature rather than taking the first one."
   , indent 8 "    user_input_names = {"
   , indent 8 "        spec.arg.name"
   , indent 8 "        for spec in ep.graph_signature.input_specs"
   , indent 8 "        if getattr(spec.kind, \"name\", \"\") == \"USER_INPUT\""
   , indent 8 "    }"
   , indent 8 "    state = dict(ep.state_dict)"
   , indent 8 "    constants = dict(getattr(ep, 'constants', {}))"
   , indent 8 "    for spec in ep.graph_signature.input_specs:"
   , indent 8 "        if getattr(spec.kind, 'name', '') != 'USER_INPUT':"
   , indent 8 "            target = getattr(spec, 'target', None)"
   , indent 8 "            value = state.get(target, constants.get(target, None))"
   , indent 8 ("            if isinstance(value, torch.Tensor): _TENSOR_VALUES[spec.arg.name]" ++
       " = value")
   , indent 8 "    if _graph_is_lowerable(ep.graph, model):"
   , indent 8 "        return ep.graph, user_input_names"
   , indent 8 "except Exception:"
   , indent 8 "    if require_torch_export:"
   , indent 8 "        raise"
   , indent 4 "if require_torch_export:"
   , indent 8 ("raise NotImplementedError(\"torch.export produced operators outside the " ++
       "current TorchLean import subset\")")
   , indent 4 "from torch.fx import symbolic_trace"
   , indent 4 "from torch.fx.passes.shape_prop import ShapeProp"
   , indent 4 "gm = symbolic_trace(model)"
   , indent 4 "ShapeProp(gm).propagate(example)"
   , indent 4 "for node in gm.graph.nodes:"
   , indent 8 "if node.op == 'get_attr':"
   , indent 8 "    value = gm"
   , indent 8 "    for part in str(node.target).split('.'): value = getattr(value, part)"
   , indent 8 "    if isinstance(value, torch.Tensor): _TENSOR_VALUES[node.name] = value"
   , indent 4 "_check_graph_operators(gm.graph, model)"
   , indent 4 "return gm.graph, None"
   , "" ]

/-- The exported entry point: walk the captured graph and write the artifact. -/
def entrypoint (options : Options) : Array String :=
  #[ s!"def {options.functionName}(model, example, json_path: str, require_torch_export=False):"
   , indent 4 "graph, user_input_names = _capture(model, example, require_torch_export)"
   , indent 4 "nodes = []"
   , indent 4 "node_to_id = {}"
   , indent 4 "output_ids = None"
   , indent 4 "input_id = None"
   , indent 4 "for node in graph.nodes:"
   , indent 8 "if node.op == \"output\":"
   , indent 8 "    refs = _node_refs(node.args, node_to_id)"
   , indent 8 "    if not refs:"
   , indent 8 ("        raise NotImplementedError(\"TorchLean graph import requires at least " ++
       "one tensor output\")")
   , indent 8 "    output_ids = refs"
   , indent 8 "    continue"
   , indent 8 "if len(getattr(node, \"users\", {})) == 0:"
   , indent 8 "    # FX often leaves dead tuple projections behind, e.g. attention weights from"
   , indent 8 "    # `y, _ = mha(...)`. They are not part of the exported value, so we omit them"
   , indent 8 ("    # instead of forcing every unused container projection to have a tensor " ++
       "lowering.")
   , indent 8 "    node_to_id[node] = None"
   , indent 8 "    continue"
   , indent 8 "if node.op in (\"get_attr\",):"
   , indent 8 "    continue"
   , indent 8 "if node.op == \"placeholder\" and user_input_names is not None:"
   , indent 8 "    if node.name not in user_input_names:"
   , indent 8 "        # Parameters, buffers, constants, and effect tokens live outside"
   , indent 8 "        # TorchLean's single user-input tensor dataflow graph."
   , indent 8 "        node_to_id[node] = None"
   , indent 8 "        continue"
   , indent 8 "node_id = len(nodes)"
   , indent 8 "node_to_id[node] = node_id"
   , indent 8 "shape = _node_shape(node)"
   , indent 8 "tuple_shapes = _tuple_shapes_of_node(node)"
   , indent 8 ("value_meta = {\"value_kind\": \"tuple\", \"tuple_shapes\": tuple_shapes} if " ++
       "tuple_shapes else {\"value_kind\": \"tensor\", \"shape\": shape}")
   , indent 8 "if node.op == \"placeholder\":"
   , indent 8 "    if input_id is not None:"
   , indent 8 ("        raise NotImplementedError(\"TorchLean graph import currently supports" ++
       " one user input\")")
   , indent 8 ("    kind = " ++ Wire.object .input)
   , indent 8 "    parents = []"
   , indent 8 "    input_id = node_id"
   , indent 8 "elif tuple_shapes:"
   , indent 8 "    # The importer expands known tuple producers into tensor dataflow."
   , indent 8 "    kind = _tuple_kind(node, model)"
   , indent 8 "    parents = _node_refs((node.args, node.kwargs), node_to_id)"
   , indent 8 "elif node.op in (\"call_function\", \"call_method\", \"call_module\"):"
   , indent 8 "    kind = _lower_kind(node, model)"
   , indent 8 "    parents = _node_refs((node.args, node.kwargs), node_to_id)"
   , indent 8 "else:"
   , indent 8 "    raise NotImplementedError(f\"unsupported FX node op: {node.op}\")"
   , indent 8 "entry = {\"id\": node_id, \"parents\": parents, **value_meta, **kind}"
   , indent 8 "if " ++ boolLiteral options.includeDebugTargets ++ ":"
   , indent 8 "    entry[\"debug_target\"] = _target_name(node.target)"
   , indent 8 "nodes.append(entry)"
   , indent 4 "if input_id is None or output_ids is None:"
   , indent 8 "raise RuntimeError(\"could not identify graph input/output\")"
   , indent 4 ("payload = {\"format\": FORMAT, \"input_id\": input_id, \"output_ids\": " ++
       "output_ids, \"nodes\": nodes}")
   , indent 4 "try:"
   , indent 8 "text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)"
   , indent 4 "except ValueError as exc:"
   , indent 8 ("raise ValueError(\"TorchLean IR JSON cannot carry NaN or infinite values;" ++
       " check the model parameters and constants\") from exc")
   , indent 4 "with open(json_path, \"w\", encoding=\"utf-8\") as f:"
   , indent 8 "f.write(text)"
   , indent 4 "return payload"
   , "" ]

/-- Command-line `main` for the generated script. -/
def cli (options : Options) : Array String :=
  #[ "def main():"
   , indent 4 ("parser = argparse.ArgumentParser(description=\"Export a PyTorch nn.Module to " ++
       "TorchLean IR JSON\")")
   , indent 4 ("parser.add_argument(\"module\", help=\"Python file containing the model " ++
       "class/constructor\")")
   , indent 4 ("parser.add_argument(\"ctor\", help=\"Zero-argument model class or constructor" ++
       " name\")")
   , indent 4 "parser.add_argument(\"json\", help=\"Output graph JSON path\")"
   , indent 4 ("parser.add_argument(\"--example-shape\", required=True, help=\"Comma-separated" ++
       " example input shape, e.g. 1,4\")")
   , indent 4 ("parser.add_argument(\"--require-torch-export\", action=\"store_true\", " ++
       "help=\"Fail instead of falling back to FX\")")
   , indent 4 "args = parser.parse_args()"
   , indent 4 "shape = tuple(int(x) for x in args.example_shape.split(',') if x)"
   , indent 4 "model = _load_model(args.module, args.ctor)"
   , indent 4 "example = torch.randn(*shape)"
   , indent 4
       s!"payload = {options.functionName}(model, example, args.json, args.require_torch_export)"
   , indent 4 "print(f\"wrote {len(payload['nodes'])} TorchLean IR nodes to {args.json}\")"
   , ""
   , "if __name__ == \"__main__\":"
   , indent 4 "main()" ]

end Internal

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
def script (options : Options := {}) : String :=
  joinLines <|
    Internal.imports ++ Internal.shapes ++ Internal.tupleKind ++ Internal.model ++
      Internal.payload ++ Internal.axes ++ Internal.lower ++ Internal.coverage ++
      Internal.capture options ++ Internal.entrypoint options ++ Internal.cli options

end TorchExport
end PyTorch
end Export
