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
  --NN.Runtime.PyTorch.Import.TorchExport.parseGraph-->
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
structure GraphBridgeOptions where
  /-- Name of the Python helper function emitted into the script. -/
  functionName : String := "export_torchlean_graph_json"
  /-- If true, use `torch.export.export` first and fall back to FX symbolic tracing. -/
  preferTorchExport : Bool := true
  /-- If true, include raw PyTorch target strings in each node for debugging. -/
  includeDebugTargets : Bool := true
deriving Repr

/-! ## Python sections

Each section is an array of Python lines ending in one blank separator line. Indentation is
applied here so the sections concatenate into a valid module.
-/

/-- Imports and the artifact format marker. -/
def bridgeImports : Array String :=
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
def shapeHelpers : Array String :=
  #[ "def _shape_of(value):"
   , indentFour "if hasattr(value, \"shape\"):"
   , indentEight "return [int(x) for x in tuple(value.shape)]"
   , indentFour "if isinstance(value, (tuple, list)) and value and hasattr(value[0], \"shape\"):"
   , indentEight "return [int(x) for x in tuple(value[0].shape)]"
   , indentFour "return []"
   , ""
   , "def _node_shape(node):"
   , indentFour "return _shape_of(node.meta.get(\"val\", node.meta.get(\"tensor_meta\")))"
   , ""
   , "def _shape_from_arg(arg):"
   , indentFour "return _node_shape(arg) if hasattr(arg, \"meta\") else []"
   , ""
   , "def _tuple_shapes_of_node(node):"
   , indentFour "val = node.meta.get(\"val\", None)"
   , indentFour "if isinstance(val, (tuple, list)) and not hasattr(val, \"shape\"):"
   , indentEight "return [_shape_of(x) for x in val]"
   , indentFour "tm = node.meta.get(\"tensor_meta\", None)"
   , indentFour "if hasattr(tm, \"shape\"):"
   , indentEight "return []"
   , indentFour "if isinstance(tm, (tuple, list)):"
   , indentEight "return [_shape_of(x) for x in tm]"
   , indentFour "return []"
   , ""
   , "def _is_getitem(node):"
   , indentFour "return node.op == \"call_function\" and node.target is operator.getitem"
   , ""
   , "def _is_getattr(node):"
   , indentFour "return node.op == \"call_function\" and node.target is getattr"
   , "" ]

/-- Kind entries for tuple-valued FX nodes; `nn.MultiheadAttention` carries its payload. -/
def tupleKindSection : Array String :=
  #[ "def _tuple_kind(node, model=None):"
   , indentFour "if node.op == \"call_module\" and model is not None:"
   , indentEight "mod = model.get_submodule(str(node.target))"
   , indentEight "if isinstance(mod, nn.MultiheadAttention):"
   , indentEight "    if mod.add_zero_attn or mod.bias_k is not None or mod.bias_v is not None:"
   , indentEight ("        raise NotImplementedError(\"nn.MultiheadAttention with added" ++
       " bias/zero sequence entries is outside the current TorchLean IR import subset\")")
   , indentEight "    if mod.kdim != mod.embed_dim or mod.vdim != mod.embed_dim:"
   , indentEight ("        raise NotImplementedError(\"nn.MultiheadAttention with kdim/vdim " ++
       "different from embed_dim is outside the current TorchLean IR import subset\")")
   , indentEight ("    if node.kwargs.get(\"attn_mask\", None) is not None or " ++
       "node.kwargs.get(\"key_padding_mask\", None) is not None:")
   , indentEight ("        raise NotImplementedError(\"masked nn.MultiheadAttention needs an " ++
       "explicit TorchLean hard-mask lowering\")")
   , indentEight "    if bool(node.kwargs.get(\"is_causal\", False)):"
   , indentEight ("        raise NotImplementedError(\"causal nn.MultiheadAttention needs an " ++
       "explicit TorchLean hard-mask lowering\")")
   , indentEight "    return {"
   , indentEight ("        \"kind\": \"" ++ Wire.mhaTuple ++ "\",")
   , indentEight "        \"embed_dim\": int(mod.embed_dim),"
   , indentEight "        \"num_heads\": int(mod.num_heads),"
   , indentEight "        \"batch_first\": bool(mod.batch_first),"
   , indentEight "        \"dropout_zero\": bool(float(mod.dropout) == 0.0),"
   , indentEight "        \"bias\": bool(mod.in_proj_bias is not None),"
   , indentEight "        **_multihead_attention_payload(mod),"
   , indentEight "    }"
   , indentFour "raise NotImplementedError(f\"unsupported tuple-valued PyTorch op: {node.target}\")"
   , "" ]

/-- Model loading, target naming, FX node reference collection, and spatial tuple parsing. -/
def modelHelpers : Array String :=
  #[ "def _load_model(module_path: str, ctor_name: str):"
   , indentFour ("spec = importlib.util.spec_from_file_location(\"torchlean_user_model\", " ++
       "module_path)")
   , indentFour "if spec is None or spec.loader is None:"
   , indentEight "raise RuntimeError(f\"could not load Python module from {module_path}\")"
   , indentFour "mod = importlib.util.module_from_spec(spec)"
   , indentFour "spec.loader.exec_module(mod)"
   , indentFour "ctor = getattr(mod, ctor_name)"
   , indentFour "model = ctor()"
   , indentFour "model.eval()"
   , indentFour "return model"
   , ""
   , "def _target_name(target):"
   , indentFour "return str(target).replace(\"torch.ops.\", \"\")"
   , ""
   , "def _is_op(node, aten=(), functions=(), methods=()):"
   , indentFour "# ATen overloads have distinct contracts: exp.out writes to an output tensor."
   , indentFour "# FX keeps Python callables or method names, so match those identities explicitly."
   , indentFour "if node.op == \"call_function\":"
   , indentEight "if isinstance(node.target, torch._ops.OpOverload):"
   , indentEight "    return str(node.target) in aten"
   , indentEight "return any(node.target is function for function in functions)"
   , indentFour "return node.op == \"call_method\" and node.target in methods"
   , ""
   , "def _node_refs(obj, node_to_id):"
   , indentFour "refs = []"
   , indentFour "def visit(x):"
   , indentEight "if isinstance(x, torch.fx.Node):"
   , indentEight "    ref = node_to_id.get(x, None)"
   , indentEight "    if ref is not None:"
   , indentEight "        refs.append(ref)"
   , indentEight "elif isinstance(x, (tuple, list)):"
   , indentEight "    for y in x: visit(y)"
   , indentEight "elif isinstance(x, dict):"
   , indentEight "    for y in x.values(): visit(y)"
   , indentFour "visit(obj)"
   , indentFour "return refs"
   , ""
   , "def _spatial_tuple(x, rank, default):"
   , indentFour "if x is None: return [int(default)] * rank"
   , indentFour "if isinstance(x, int): return [int(x)] * rank"
   , indentFour "if isinstance(x, (tuple, list)) and len(x) == rank: return [int(v) for v in x]"
   , indentFour ("raise NotImplementedError(f\"expected an integer or length-{rank} spatial " ++
       "tuple, got {x!r}\")")
   , "" ]

/-- Parameter payload serialization for linear, attention, and convolution nodes. -/
def payloadHelpers : Array String :=
  #[ "def _tensor_values(value):"
   , indentFour "return value.detach().cpu().tolist()"
   , ""
   , "_TENSOR_VALUES = {}"
   , ""
   , "def _resolve_tensor(value):"
   , indentFour "if isinstance(value, torch.Tensor): return value"
   , indentFour ("if isinstance(value, torch.fx.Node) and value.name in _TENSOR_VALUES: " ++
       "return _TENSOR_VALUES[value.name]")
   , indentFour ("raise NotImplementedError(f\"could not resolve tensor argument {value!r} into " ++
       "the exported payload\")")
   , ""
   , "def _linear_payload(weight, bias):"
   , indentFour "weight = _resolve_tensor(weight)"
   , indentFour "if weight.ndim != 2:"
   , indentEight ("raise NotImplementedError(f\"linear weight must have rank 2, got shape " ++
       "{tuple(weight.shape)}\")")
   , indentFour "out_dim, in_dim = (int(value) for value in weight.shape)"
   , indentFour "bias = _resolve_tensor(bias) if bias is not None else weight.new_zeros(out_dim)"
   , indentFour "if tuple(bias.shape) != (out_dim,):"
   , indentEight ("raise NotImplementedError(f\"linear bias must have shape {(out_dim,)}, got " ++
       "{tuple(bias.shape)}\")")
   , indentFour ("return {\"out_dim\": out_dim, \"in_dim\": in_dim, \"weight\": " ++
       "_tensor_values(weight), \"bias\": _tensor_values(bias)}")
   , ""
   , "def _multihead_attention_payload(mod):"
   , indentFour "if mod.in_proj_weight is None:"
   , indentEight ("raise NotImplementedError(\"nn.MultiheadAttention with separate q/k/v " ++
       "projection weights is outside the current TorchLean IR import subset\")")
   , indentFour "embed_dim = int(mod.embed_dim)"
   , indentFour "if tuple(mod.in_proj_weight.shape) != (3 * embed_dim, embed_dim):"
   , indentEight ("raise NotImplementedError(f\"unexpected nn.MultiheadAttention in_proj_weight " ++
       "shape {tuple(mod.in_proj_weight.shape)}\")")
   , indentFour "q_weight, k_weight, v_weight = mod.in_proj_weight.chunk(3, dim=0)"
   , indentFour "if mod.in_proj_bias is None:"
   , indentEight "q_bias = k_bias = v_bias = mod.in_proj_weight.new_zeros(embed_dim)"
   , indentFour "else:"
   , indentEight "if tuple(mod.in_proj_bias.shape) != (3 * embed_dim,):"
   , indentEight ("    raise NotImplementedError(f\"unexpected nn.MultiheadAttention " ++
       "in_proj_bias shape {tuple(mod.in_proj_bias.shape)}\")")
   , indentEight "q_bias, k_bias, v_bias = mod.in_proj_bias.chunk(3, dim=0)"
   , indentFour "if tuple(mod.out_proj.weight.shape) != (embed_dim, embed_dim):"
   , indentEight ("raise NotImplementedError(f\"unexpected nn.MultiheadAttention out_proj " ++
       "weight shape {tuple(mod.out_proj.weight.shape)}\")")
   , indentFour ("out_bias = mod.out_proj.bias if mod.out_proj.bias is not None else " ++
       "mod.out_proj.weight.new_zeros(embed_dim)")
   , indentFour "return {"
   , indentEight "\"q_weight\": _tensor_values(q_weight), \"q_bias\": _tensor_values(q_bias),"
   , indentEight "\"k_weight\": _tensor_values(k_weight), \"k_bias\": _tensor_values(k_bias),"
   , indentEight "\"v_weight\": _tensor_values(v_weight), \"v_bias\": _tensor_values(v_bias),"
   , indentEight ("\"out_weight\": _tensor_values(mod.out_proj.weight), \"out_bias\": " ++
       "_tensor_values(out_bias),")
   , indentEight "\"scale\": float((embed_dim // int(mod.num_heads)) ** -0.5),"
   , indentFour "}"
   , ""
   , "def _convolution_payload(weight, bias, groups):"
   , indentFour "weight = _resolve_tensor(weight)"
   , indentFour "groups = int(groups)"
   , indentFour "if weight.ndim < 3:"
   , indentEight ("raise NotImplementedError(f\"convolution weight must have rank at least 3, " ++
       "got shape {tuple(weight.shape)}\")")
   , indentFour "if groups <= 0:"
   , indentEight ("raise NotImplementedError(f\"convolution groups must be positive, got " ++
       "{groups}\")")
   , indentFour "out_channels = int(weight.shape[0])"
   , indentFour "in_channels_per_group = int(weight.shape[1])"
   , indentFour "if out_channels % groups != 0:"
   , indentEight ("raise NotImplementedError(f\"convolution out_channels={out_channels} is not " ++
       "divisible by groups={groups}\")")
   , indentFour "in_channels = in_channels_per_group * groups"
   , indentFour "if groups == 1:"
   , indentEight "dense_weight = weight"
   , indentFour "else:"
   , indentEight "out_channels_per_group = out_channels // groups"
   , indentEight "dense_weight = weight.new_zeros((out_channels, in_channels, *weight.shape[2:]))"
   , indentEight "for group in range(groups):"
   , indentEight "    out_start = group * out_channels_per_group"
   , indentEight "    out_end = out_start + out_channels_per_group"
   , indentEight "    in_start = group * in_channels_per_group"
   , indentEight "    in_end = in_start + in_channels_per_group"
   , indentEight "    dense_weight[out_start:out_end, in_start:in_end] = weight[out_start:out_end]"
   , indentFour ("bias = _resolve_tensor(bias) if bias is not None else " ++
       "weight.new_zeros(out_channels)")
   , indentFour "if tuple(bias.shape) != (out_channels,):"
   , indentEight ("raise NotImplementedError(f\"convolution bias must have shape " ++
       "{(out_channels,)}, got {tuple(bias.shape)}\")")
   , indentFour ("return {\"in_channels\": in_channels, \"out_channels\": out_channels, " ++
       "\"weight\": _tensor_values(dense_weight), \"bias\": _tensor_values(bias)}")
   , "" ]

/-- Flatten-versus-reshape selection and axis normalization. -/
def axisHelpers : Array String :=
  #[ "def _flatten_kind(input_shape, output_shape):"
   , indentFour "if len(output_shape) == 1:"
   , indentEight ("return {" ++ Wire.kindField .flatten ++ ", \"value_shape\": input_shape}")
   , indentFour ("return {" ++ Wire.kindField .reshape ++ ", \"in_shape\": input_shape, " ++
       "\"out_shape\": output_shape}")
   , ""
   , "def _normalize_axis(axis, rank):"
   , indentFour "axis = int(axis)"
   , indentFour "axis = axis + rank if axis < 0 else axis"
   , indentFour "if axis < 0 or axis >= rank:"
   , indentEight "raise NotImplementedError(f\"axis {axis} is outside rank {rank}\")"
   , indentFour "return axis"
   , ""
   , "def _reduction_axes(axis, rank, op_name):"
   , indentFour ("raw = range(rank) if axis is None else axis if isinstance(axis, (tuple, list))" ++
       " else (axis,)")
   , indentFour "axes = [_normalize_axis(value, rank) for value in raw]"
   , indentFour "if len(set(axes)) != len(axes):"
   , indentEight "raise NotImplementedError(f\"{op_name} has duplicate reduction axes {axes}\")"
   , indentFour "return axes"
   , "" ]

/-- Opening of `_lower_kind`: target name, argument, and axis extraction shared by all rules. -/
def lowerKindHeader : Array String :=
  #[ "def _lower_kind(node, model=None):"
   , indentFour "name = _target_name(node.target)"
   , indentFour "args = node.args"
   , indentFour "kwargs = dict(node.kwargs)"
   , indentFour "if kwargs.get(\"out\", None) is not None:"
   , indentEight "raise NotImplementedError(\"out= mutation is outside the pure graph contract\")"
   , indentFour "axis = kwargs.get(\"dim\", kwargs.get(\"axis\", None))"
   , indentFour "if axis is None and len(args) > 1: axis = args[1]" ]

/-- `_lower_kind` rules for `call_module` nodes, keyed on the `nn.Module` subclass. -/
def lowerModuleRules : Array String :=
  #[ indentFour "if node.op == \"call_module\" and model is not None:"
   , indentEight "mod = model.get_submodule(str(node.target))"
   , indentEight "if isinstance(mod, nn.Linear):"
   , indentEight ("    return {" ++ Wire.kindField .linear ++
       ", **_linear_payload(mod.weight, mod.bias)}")
   , indentEight "if isinstance(mod, nn.ReLU):"
   , indentEight ("    if mod.inplace: raise NotImplementedError(\"in-place nn.ReLU mutation is " ++
       "outside the pure TorchLean graph contract\")")
   , indentEight ("    return " ++ Wire.kindObject .relu)
   , indentEight ("if isinstance(mod, nn.Tanh): return " ++ Wire.kindObject .tanh)
   , indentEight ("if isinstance(mod, nn.Sigmoid): return " ++ Wire.kindObject .sigmoid)
   , indentEight "if isinstance(mod, nn.Softmax):"
   , indentEight ("    return {" ++ Wire.kindField .softmax ++
       ", \"axis\": _normalize_axis(mod.dim, len(_node_shape(node)))}")
   , indentEight "if isinstance(mod, nn.Flatten):"
   , indentEight ("    return _flatten_kind(_shape_from_arg(args[0]) if args else [], " ++
       "_node_shape(node))")
   , indentEight "if isinstance(mod, nn.LayerNorm):"
   , indentEight "    rank = len(_node_shape(node))"
   , indentEight "    norm_rank = len(tuple(mod.normalized_shape))"
   , indentEight ("    gamma = mod.weight if mod.weight is not None else " ++
       "torch.ones(mod.normalized_shape)")
   , indentEight ("    beta = mod.bias if mod.bias is not None else " ++
       "torch.zeros(mod.normalized_shape)")
   , indentEight ("    return {" ++ Wire.kindField .layernorm ++
       ", \"axis\": max(0, rank - norm_rank), \"eps\": float(mod.eps), " ++
       "\"gamma\": _tensor_values(gamma), \"beta\": _tensor_values(beta)}")
   , indentEight "if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):"
   , indentEight "    rank = len(tuple(mod.kernel_size))"
   , indentEight "    input_rank = len(_shape_from_arg(args[0])) if args else 0"
   , indentEight ("    if input_rank not in (rank + 1, rank + 2): raise NotImplementedError(" ++
       "f\"convolution expected rank {rank + 1} or {rank + 2}, got {input_rank}\")")
   , indentEight "    channel_axis = input_rank - rank - 1"
   , indentEight ("    if mod.padding_mode != \"zeros\": raise NotImplementedError(" ++
       "f\"convolution padding_mode={mod.padding_mode!r} is outside the current TorchLean IR " ++
       "import subset\")")
   , indentEight "    padding = _spatial_tuple(mod.padding, rank, 0)"
   , indentEight "    input_shape = _shape_from_arg(args[0])"
   , indentEight ("    return {" ++ Wire.kindField .conv ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": _spatial_tuple(mod.kernel_size, rank, 1), " ++
       "\"stride\": _spatial_tuple(mod.stride, rank, 1), \"padding\": padding, " ++
       "\"padding_after\": padding, \"dilation\": _spatial_tuple(mod.dilation, rank, 1), " ++
       "\"groups\": int(mod.groups), \"channel_axis\": channel_axis, " ++
       "\"input_spatial\": input_shape[channel_axis + 1:], " ++
       "**_convolution_payload(mod.weight, mod.bias, mod.groups)}")
   , indentEight "if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):"
   , indentEight "    if mod.training:"
   , indentEight ("        raise NotImplementedError(\"training-mode BatchNorm needs batch " ++
       "statistics and running-statistic updates\")")
   , indentEight ("    if not mod.track_running_stats or mod.running_mean is None or " ++
       "mod.running_var is None:")
   , indentEight ("        raise NotImplementedError(\"nn.BatchNorm2d without running statistics" ++
       " has batch-dependent semantics, not TorchLean eval-mode BatchNorm semantics\")")
   , indentEight ("    gamma = mod.weight if mod.weight is not None else " ++
       "torch.ones(mod.num_features)")
   , indentEight ("    beta = mod.bias if mod.bias is not None else " ++
       "torch.zeros(mod.num_features)")
   , indentEight ("    return {" ++ Wire.kindField .batchNormEval ++ ", \"channel_axis\": 1, " ++
       "\"channels\": int(mod.num_features), \"eps\": float(mod.eps), " ++
       "\"gamma\": _tensor_values(gamma), \"beta\": _tensor_values(beta), " ++
       "\"mean\": _tensor_values(mod.running_mean), \"var\": _tensor_values(mod.running_var)}")
   , indentEight "if isinstance(mod, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):"
   , indentEight ("    rank = 1 if isinstance(mod, nn.MaxPool1d) else 2 if isinstance(mod, " ++
       "nn.MaxPool2d) else 3")
   , indentEight ("    if _spatial_tuple(mod.dilation, rank, 1) != [1] * rank: raise " ++
       "NotImplementedError(\"dilated max pooling is outside the current TorchLean IR import " ++
       "subset\")")
   , indentEight ("    if mod.ceil_mode: raise NotImplementedError(\"max pooling with " ++
       "ceil_mode=True is outside the current TorchLean IR import subset\")")
   , indentEight ("    if mod.return_indices: raise NotImplementedError(\"max pooling with " ++
       "return_indices=True produces a tuple outside the current pooling lowering\")")
   , indentEight ("    return {" ++ Wire.kindField .maxPool ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": _spatial_tuple(mod.kernel_size, rank, 1), \"stride\": _spatial_tuple(" ++
       "mod.stride if mod.stride is not None else mod.kernel_size, rank, 1), " ++
       "\"padding\": _spatial_tuple(mod.padding, rank, 0)}")
   , indentEight "if isinstance(mod, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):"
   , indentEight ("    rank = 1 if isinstance(mod, nn.AvgPool1d) else 2 if isinstance(mod, " ++
       "nn.AvgPool2d) else 3")
   , indentEight ("    if mod.ceil_mode: raise NotImplementedError(\"average pooling with " ++
       "ceil_mode=True is outside the current TorchLean IR import subset\")")
   , indentEight ("    if getattr(mod, 'divisor_override', None) is not None: raise " ++
       "NotImplementedError(\"average pooling with divisor_override is outside the current " ++
       "TorchLean IR import subset\")")
   , indentEight "    padding = _spatial_tuple(mod.padding, rank, 0)"
   , indentEight ("    if any(padding) and not mod.count_include_pad: raise " ++
       "NotImplementedError(\"padded average pooling with count_include_pad=False is outside " ++
       "the current TorchLean IR import subset\")")
   , indentEight ("    return {" ++ Wire.kindField .avgPool ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": _spatial_tuple(mod.kernel_size, rank, 1), \"stride\": _spatial_tuple(" ++
       "mod.stride if mod.stride is not None else mod.kernel_size, rank, 1), " ++
       "\"padding\": padding}") ]

/-- `_lower_kind` rules for elementwise `aten`/functional targets that carry no payload. -/
def lowerElementwiseRules : Array String :=
  #[ "    if _is_op(node, (\"aten.add.Tensor\",), (operator.add, torch.add), (\"add\",)):"
   , "        operands = (args[0] if args else kwargs.get(\"input\"),"
   , "                    args[1] if len(args) > 1 else kwargs.get(\"other\"))"
   , "        if not all(isinstance(x, torch.fx.Node) for x in operands):"
   , "            raise NotImplementedError(\"scalar operands need constant/broadcast lowering\")"
   , "        alpha = kwargs.get(\"alpha\", args[2] if len(args) > 2 else 1)"
   , "        if alpha != 1:"
   , "            raise NotImplementedError(\"torch.add with alpha != 1 is unsupported\")"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .add ++ "}"
   , "    if _is_op(node, (\"aten.sub.Tensor\",), " ++
       "(operator.sub, torch.sub, torch.subtract), (\"sub\", \"subtract\")):"
   , "        operands = (args[0] if args else kwargs.get(\"input\"),"
   , "                    args[1] if len(args) > 1 else kwargs.get(\"other\"))"
   , "        if not all(isinstance(x, torch.fx.Node) for x in operands):"
   , "            raise NotImplementedError(\"scalar operands need constant/broadcast lowering\")"
   , "        alpha = kwargs.get(\"alpha\", args[2] if len(args) > 2 else 1)"
   , "        if alpha != 1:"
   , "            raise NotImplementedError(\"torch.sub with alpha != 1 is unsupported\")"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .sub ++ "}"
   , "    if _is_op(node, (\"aten.mul.Tensor\",), " ++
       "(operator.mul, torch.mul, torch.multiply), (\"mul\", \"multiply\")):"
   , "        operands = (args[0] if args else kwargs.get(\"input\"),"
   , "                    args[1] if len(args) > 1 else kwargs.get(\"other\"))"
   , "        if not all(isinstance(x, torch.fx.Node) for x in operands):"
   , "            raise NotImplementedError(\"scalar operands need constant/broadcast lowering\")"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .mulElem ++ "}"
   , "    if _is_op(node, (\"aten.relu.default\",), (torch.relu, F.relu), (\"relu\",)):"
   , "        inplace = kwargs.get(\"inplace\", args[1] if len(args) > 1 else False)"
   , "        if inplace: raise " ++ "NotImplementedError(\"in-place relu is outside the " ++
       "pure graph contract\")"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .relu ++ "}"
   , "    if _is_op(node, (\"aten.tanh.default\",), (torch.tanh,), (\"tanh\",)):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .tanh ++ "}"
   , "    if _is_op(node, (\"aten.sigmoid.default\",), (torch.sigmoid,), (\"sigmoid\",)):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .sigmoid ++ "}"
   , "    if _is_op(node, (\"aten.exp.default\",), (torch.exp,), (\"exp\",)):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .exp ++ "}"
   , "    if _is_op(node, (\"aten.log.default\",), (torch.log,), (\"log\",)):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .log ++ "}"
   , "    if _is_op(node, (\"aten.sin.default\",), (torch.sin,), (\"sin\",)):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .sin ++ "}"
   , "    if _is_op(node, (\"aten.cos.default\",), (torch.cos,), (\"cos\",)):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .cos ++ "}"
   , "    if _is_op(node, (\"aten.abs.default\",), " ++
       "(torch.abs, torch.absolute), (\"abs\", \"absolute\")):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .abs ++ "}"
   , "    if _is_op(node, (\"aten.sqrt.default\",), (torch.sqrt,), (\"sqrt\",)):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .sqrt ++ "}"
   , "    if _is_op(node, (\"aten.reciprocal.default\",), (torch.reciprocal,), (\"reciprocal\",)):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .inv ++ "}"
   , "    if _is_op(node, (\"aten.maximum.default\",), (torch.maximum,), (\"maximum\",)):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .maxElem ++ "}"
   , "    if _is_op(node, (\"aten.minimum.default\",), (torch.minimum,), (\"minimum\",)):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .minElem ++ "}"
   , "    if _is_op(node, (\"aten.matmul.default\", " ++
       "\"aten.mm.default\"), (torch.matmul, torch.mm, " ++
       "operator.matmul), (\"matmul\", \"mm\")):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .matmul ++ "}" ]

/-- `_lower_kind` rules for reductions, softmax, and shape operations. -/
def lowerShapeRules : Array String :=
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
   , "        return {\"kind\": " ++ Wire.quotedOpTag .reduceSum ++ " if is_sum else " ++
       Wire.quotedOpTag .reduceMean ++ ","
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
   , "        return {\"kind\": " ++ Wire.quotedOpTag .softmax ++
       ", \"axis\": _normalize_axis(axis, rank)}"
   , "    if _is_op(node, (\"aten.reshape.default\", " ++
       "\"aten.view.default\"), (torch.reshape,), (\"reshape\", \"view\")):"
   , "        if node.op == \"call_method\" and node.target == \"view\":"
   , "            if isinstance(kwargs.get(\"dtype\", " ++
       "args[1] if len(args) > 1 else None), torch.dtype):"
   , "                raise NotImplementedError(\"view(dtype) changes scalar semantics\")"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .reshape ++
       ", \"in_shape\": _shape_from_arg(args[0]),"
   , "                \"out_shape\": _node_shape(node)}"
   , "    if _is_op(node, (\"aten.contiguous.default\",), (), (\"contiguous\",)):"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .reshape ++
       ", \"in_shape\": _shape_from_arg(args[0]),"
   , "                \"out_shape\": _node_shape(node)}"
   , "    if _is_op(node, (\"aten.flatten.using_ints\",), (torch.flatten,), (\"flatten\",)):"
   , "        return _flatten_kind(_shape_from_arg(args[0]), _node_shape(node))"
   , "    if _is_op(node, (\"aten.permute.default\",), (torch.permute,), (\"permute\",)):"
   , "        dims = kwargs.get(\"dims\", args[1] if len(args) > 1 else ())"
   , "        perm = list(dims) if isinstance(dims, (tuple, list)) else list(args[1:])"
   , "        rank = len(_node_shape(node))"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .permute ++
       ", \"perm\": [_normalize_axis(x, rank) for x in perm]}"
   , "    if _is_op(node, (\"aten.cat.default\",), " ++
       "(torch.cat, torch.concat, torch.concatenate), ()):"
   , "        rank = len(_node_shape(node))"
   , "        dim = _normalize_axis(kwargs.get(\"dim\", args[1] if len(args) > 1 else 0), rank)"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .concat ++ ", \"axis\": dim}"
   , "    if _is_op(node, (\"aten.transpose.int\",), (torch.transpose,), (\"transpose\",)):"
   , "        rank = len(_shape_from_arg(args[0]))"
   , "        d0 = _normalize_axis(kwargs.get(\"dim0\", args[1] if len(args) > 1 else None), rank)"
   , "        d1 = _normalize_axis(kwargs.get(\"dim1\", args[2] if len(args) > 2 else None), rank)"
   , "        return {\"kind\": " ++ Wire.quotedOpTag .transpose ++
       ", \"axis1\": d0, \"axis2\": d1}" ]

/-- `_lower_kind` rules for functional layers with payloads: layer norm, linear, batch norm. -/
def lowerFunctionalLayerRules : Array String :=
  #[ indentFour ("if _is_op(node, (\"aten.layer_norm.default\",), " ++
       "(F.layer_norm, torch.layer_norm), ()):")
   , indentEight "weight = kwargs.get(\"weight\", args[2] if len(args) > 2 else None)"
   , indentEight "bias = kwargs.get(\"bias\", args[3] if len(args) > 3 else None)"
   , indentEight "eps = kwargs.get(\"eps\", args[4] if len(args) > 4 else 1e-5)"
   , indentEight "input_rank = len(_shape_from_arg(args[0])) if args else len(_node_shape(node))"
   , indentEight ("normalized_shape = args[1] if len(args) > 1 else " ++
       "kwargs.get(\"normalized_shape\", ())")
   , indentEight ("normalized_rank = len(normalized_shape) if isinstance(normalized_shape, " ++
       "(tuple, list, torch.Size)) else 1")
   , indentEight ("weight = _resolve_tensor(weight) if weight is not None else " ++
       "torch.ones(normalized_shape)")
   , indentEight ("bias = _resolve_tensor(bias) if bias is not None else " ++
       "torch.zeros(normalized_shape)")
   , indentEight ("return {" ++ Wire.kindField .layernorm ++
       ", \"axis\": max(0, input_rank - normalized_rank), \"eps\": float(eps), " ++
       "\"gamma\": _tensor_values(weight), \"beta\": _tensor_values(bias)}")
   , indentFour "if _is_op(node, (\"aten.linear.default\",), (F.linear,), ()):"
   , indentEight "weight = kwargs.get(\"weight\", args[1] if len(args) > 1 else None)"
   , indentEight "bias = kwargs.get(\"bias\", args[2] if len(args) > 2 else None)"
   , indentEight ("return {" ++ Wire.kindField .linear ++ ", **_linear_payload(weight, bias)}")
   , indentFour ("if _is_op(node, (\"aten.batch_norm.default\",), " ++
       "(F.batch_norm, torch.batch_norm), ()):")
   , indentEight "training = bool(kwargs.get(\"training\", args[5] if len(args) > 5 else False))"
   , indentEight ("if training: raise NotImplementedError(\"training-mode batch_norm is outside " ++
       "the eval-mode TorchLean IR operation\")")
   , indentEight "is_aten = node.target is not F.batch_norm"
   , indentEight "mean_pos, var_pos = (3, 4) if is_aten else (1, 2)"
   , indentEight "weight_pos, bias_pos = (1, 2) if is_aten else (3, 4)"
   , indentEight ("running_mean = kwargs.get(\"running_mean\", args[mean_pos] if len(args) > " ++
       "mean_pos else None)")
   , indentEight ("running_var = kwargs.get(\"running_var\", args[var_pos] if len(args) > " ++
       "var_pos else None)")
   , indentEight ("if running_mean is None or running_var is None: raise NotImplementedError(" ++
       "\"batch_norm without running statistics is outside the eval-mode TorchLean IR operation\")")
   , indentEight "shape = _node_shape(node)"
   , indentEight "channels = int(shape[1]) if len(shape) >= 2 else 0"
   , indentEight "eps = float(kwargs.get(\"eps\", args[7] if len(args) > 7 else 1e-5))"
   , indentEight "running_mean = _resolve_tensor(running_mean)"
   , indentEight "running_var = _resolve_tensor(running_var)"
   , indentEight ("weight = kwargs.get(\"weight\", args[weight_pos] if len(args) > weight_pos " ++
       "else None)")
   , indentEight "bias = kwargs.get(\"bias\", args[bias_pos] if len(args) > bias_pos else None)"
   , indentEight "weight = _resolve_tensor(weight) if weight is not None else torch.ones(channels)"
   , indentEight "bias = _resolve_tensor(bias) if bias is not None else torch.zeros(channels)"
   , indentEight ("return {" ++ Wire.kindField .batchNormEval ++ ", \"channel_axis\": 1, " ++
       "\"channels\": channels, \"eps\": eps, \"gamma\": _tensor_values(weight), " ++
       "\"beta\": _tensor_values(bias), \"mean\": _tensor_values(running_mean), " ++
       "\"var\": _tensor_values(running_var)}") ]

/-- `_lower_kind` rules for functional convolution and pooling, then the tuple/fallback cases. -/
def lowerFunctionalSpatialRules : Array String :=
  #[ indentFour ("if _is_op(node, (\"aten.conv1d.default\", " ++
       "\"aten.conv2d.default\", \"aten.conv3d.default\"), " ++
       "(F.conv1d, F.conv2d, F.conv3d), ()):")
   , indentEight "weight_arg = kwargs.get(\"weight\", args[1] if len(args) > 1 else None)"
   , indentEight "bias_arg = kwargs.get(\"bias\", args[2] if len(args) > 2 else None)"
   , indentEight "weight = _resolve_tensor(weight_arg)"
   , indentEight "wshape = _shape_of(weight)"
   , indentEight "rank = max(0, len(wshape) - 2)"
   , indentEight "input_rank = len(_shape_from_arg(args[0])) if args else 0"
   , indentEight ("if input_rank not in (rank + 1, rank + 2): raise NotImplementedError(" ++
       "f\"convolution expected rank {rank + 1} or {rank + 2}, got {input_rank}\")")
   , indentEight "channel_axis = input_rank - rank - 1"
   , indentEight ("stride = _spatial_tuple(kwargs.get(\"stride\", args[3] if len(args) > 3 else " ++
       "1), rank, 1)")
   , indentEight ("padding = _spatial_tuple(kwargs.get(\"padding\", args[4] if len(args) > 4 " ++
       "else 0), rank, 0)")
   , indentEight ("dilation = _spatial_tuple(kwargs.get(\"dilation\", args[5] if len(args) > 5 " ++
       "else 1), rank, 1)")
   , indentEight "groups = int(kwargs.get(\"groups\", args[6] if len(args) > 6 else 1))"
   , indentEight "out_channels = int(wshape[0]) if len(wshape) >= 2 else 0"
   , indentEight "in_channels = int(wshape[1]) * groups if len(wshape) >= 2 else 0"
   , indentEight "input_shape = _shape_from_arg(args[0])"
   , indentEight ("return {" ++ Wire.kindField .conv ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": [int(v) for v in wshape[2:]], \"stride\": stride, \"padding\": padding, " ++
       "\"padding_after\": padding, \"dilation\": dilation, \"groups\": groups, " ++
       "\"channel_axis\": channel_axis, \"input_spatial\": input_shape[channel_axis + 1:], " ++
       "**_convolution_payload(weight, bias_arg, groups)}")
   , indentFour ("if _is_op(node, (\"aten.max_pool1d.default\", " ++
       "\"aten.max_pool2d.default\", " ++ "\"aten.max_pool3d.default\"), (F.max_pool1d, " ++
       "F.max_pool2d, F.max_pool3d), ()):")
   , indentEight ("rank = next(d for d in (1, 2, 3) if _is_op(node, " ++
       "(f\"aten.max_pool{d}d.default\",), (getattr(F, " ++ "f\"max_pool{d}d\"),)))")
   , indentEight ("kernel = _spatial_tuple(args[1] if len(args) > 1 else " ++
       "kwargs.get(\"kernel_size\", 1), rank, 1)")
   , indentEight "stride_arg = kwargs.get(\"stride\", args[2] if len(args) > 2 else None)"
   , indentEight ("stride = _spatial_tuple(stride_arg if stride_arg not in (None, []) else " ++
       "kernel, rank, 1)")
   , indentEight ("padding = _spatial_tuple(kwargs.get(\"padding\", args[3] if len(args) > 3 " ++
       "else 0), rank, 0)")
   , indentEight ("dilation = _spatial_tuple(kwargs.get(\"dilation\", args[4] if len(args) > 4 " ++
       "else 1), rank, 1)")
   , indentEight "ceil_mode = bool(kwargs.get(\"ceil_mode\", args[5] if len(args) > 5 else False))"
   , indentEight "if kwargs.get(\"return_indices\", args[6] if len(args) > 6 else False):"
   , indentEight "    raise NotImplementedError(\"max pooling indices need tuple lowering\")"
   , indentEight ("if dilation != [1] * rank: raise NotImplementedError(\"dilated max pooling is" ++
       " outside the current TorchLean IR import subset\")")
   , indentEight ("if ceil_mode: raise NotImplementedError(\"max pooling with ceil_mode=True is " ++
       "outside the current TorchLean IR import subset\")")
   , indentEight ("return {" ++ Wire.kindField .maxPool ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": kernel, \"stride\": stride, \"padding\": padding}")
   , indentFour ("if _is_op(node, (\"aten.avg_pool1d.default\", " ++
       "\"aten.avg_pool2d.default\", " ++ "\"aten.avg_pool3d.default\"), (F.avg_pool1d, " ++
       "F.avg_pool2d, F.avg_pool3d), ()):")
   , indentEight ("rank = next(d for d in (1, 2, 3) if _is_op(node, " ++
       "(f\"aten.avg_pool{d}d.default\",), (getattr(F, " ++ "f\"avg_pool{d}d\"),)))")
   , indentEight ("kernel = _spatial_tuple(args[1] if len(args) > 1 else " ++
       "kwargs.get(\"kernel_size\", 1), rank, 1)")
   , indentEight "stride_arg = kwargs.get(\"stride\", args[2] if len(args) > 2 else None)"
   , indentEight ("stride = _spatial_tuple(stride_arg if stride_arg not in (None, []) else " ++
       "kernel, rank, 1)")
   , indentEight ("padding = _spatial_tuple(kwargs.get(\"padding\", args[3] if len(args) > 3 " ++
       "else 0), rank, 0)")
   , indentEight "ceil_mode = bool(kwargs.get(\"ceil_mode\", args[4] if len(args) > 4 else False))"
   , indentEight ("count_include_pad = bool(kwargs.get(\"count_include_pad\", args[5] if " ++
       "len(args) > 5 else True))")
   , indentEight ("divisor_override = kwargs.get(\"divisor_override\", args[6] if len(args) > 6 " ++
       "else None)")
   , indentEight ("if ceil_mode: raise NotImplementedError(\"average pooling with ceil_mode=True" ++
       " is outside the current TorchLean IR import subset\")")
   , indentEight ("if any(padding) and not count_include_pad: raise NotImplementedError(\"padded" ++
       " average pooling with count_include_pad=False is outside the current TorchLean IR " ++
       "import subset\")")
   , indentEight ("if divisor_override is not None: raise NotImplementedError(\"average pooling " ++
       "with divisor_override is outside the current TorchLean IR import subset\")")
   , indentEight ("return {" ++ Wire.kindField .avgPool ++ ", \"spatial_rank\": rank, " ++
       "\"kernel\": kernel, \"stride\": stride, \"padding\": padding}")
   , indentFour "if _is_getitem(node):"
   , indentEight "shapes = _tuple_shapes_of_node(args[0]) if args else []"
   , indentEight "if not shapes or len(args) < 2 or type(args[1]) is not int:"
   , indentEight "    raise NotImplementedError(\"getitem requires a tuple and a static integer\")"
   , indentEight "idx = args[1] if args[1] >= 0 else len(shapes) + args[1]"
   , indentEight "if not 0 <= idx < len(shapes):"
   , indentEight "    raise NotImplementedError(\"tuple index is out of range\")"
   , indentEight ("return {\"kind\": \"" ++ Wire.tupleGetItem ++ "\", \"index\": idx}")
   , indentFour "if _is_getattr(node):"
   , indentEight "attr = args[1] if len(args) > 1 else \"<unknown>\""
   , indentEight ("raise NotImplementedError(f\"unsupported PyTorch attribute projection:" ++
       " {attr}. If this came from a tuple-returning op such as torch.sort(...).values, add a " ++
       "value-graph/lowering rule for that producer instead of treating the attribute as an " ++
       "ordinary tensor op.\")")
   , indentFour "raise NotImplementedError(f\"unsupported PyTorch op: {name}\")"
   , "" ]

/-- The complete `_lower_kind` function. -/
def lowerKindSection : Array String :=
  lowerKindHeader ++ lowerModuleRules ++ lowerElementwiseRules ++ lowerShapeRules ++
    lowerFunctionalLayerRules ++ lowerFunctionalSpatialRules

/-- Check captured operations before dropping unused values.

FX records `x.relu_(); return x` with an unused result for the ReLU call. Skipping that call before
checking its contract would erase the update to `x`. Unsupported calls therefore fail even when
their results have no users. -/
def lowerabilitySection : Array String :=
  #[ "def _check_graph_operators(graph, model):"
   , indentFour "for node in graph.nodes:"
   , indentEight "if node.op in (\"placeholder\", \"get_attr\", \"output\"):"
   , indentEight "    continue"
   , indentEight "if _tuple_shapes_of_node(node):"
   , indentEight "    _tuple_kind(node, model)"
   , indentEight "elif node.op in (\"call_function\", \"call_method\", \"call_module\"):"
   , indentEight "    _lower_kind(node, model)"
   , indentEight "else:"
   , indentEight "    raise NotImplementedError(f\"unsupported FX node op: {node.op}\")"
   , ""
   , "def _graph_is_lowerable(graph, model):"
   , indentFour "try:"
   , indentEight "_check_graph_operators(graph, model)"
   , indentFour "except NotImplementedError:"
   , indentEight "return False"
   , indentFour "return True"
   , "" ]

/-- Graph capture: `torch.export` when preferred and lowerable, otherwise FX symbolic tracing. -/
def captureSection (options : GraphBridgeOptions) : Array String :=
  #[ "def _capture(model, example, require_torch_export=False):"
   , indentFour "global _TENSOR_VALUES"
   , indentFour "_TENSOR_VALUES = {}"
   , indentFour "if " ++ pyBool options.preferTorchExport ++ ":"
   , indentEight "try:"
   , indentEight "    ep = torch.export.export(model, (example,))"
   , indentEight "    # Functionalization can turn mutation into ordinary tensor operations."
   , indentEight "    # Extra outputs describe state updates that need their own replay rules."
   , indentEight ("    if any(spec.kind.name != \"USER_OUTPUT\" for " ++
       "spec in ep.graph_signature.output_specs):")
   , indentEight ("        raise NotImplementedError(\"exported " ++
       "mutation/effects are outside the pure graph contract\")")
   , indentEight "    # ExportedProgram lifts parameters, buffers, and constants into graph"
   , indentEight "    # placeholders. Their order is not a user-input contract, so classify"
   , indentEight "    # placeholders through graph_signature rather than taking the first one."
   , indentEight "    user_input_names = {"
   , indentEight "        spec.arg.name"
   , indentEight "        for spec in ep.graph_signature.input_specs"
   , indentEight "        if getattr(spec.kind, \"name\", \"\") == \"USER_INPUT\""
   , indentEight "    }"
   , indentEight "    state = dict(ep.state_dict)"
   , indentEight "    constants = dict(getattr(ep, 'constants', {}))"
   , indentEight "    for spec in ep.graph_signature.input_specs:"
   , indentEight "        if getattr(spec.kind, 'name', '') != 'USER_INPUT':"
   , indentEight "            target = getattr(spec, 'target', None)"
   , indentEight "            value = state.get(target, constants.get(target, None))"
   , indentEight ("            if isinstance(value, torch.Tensor): _TENSOR_VALUES[spec.arg.name]" ++
       " = value")
   , indentEight "    if _graph_is_lowerable(ep.graph, model):"
   , indentEight "        return ep.graph, user_input_names"
   , indentEight "except Exception:"
   , indentEight "    if require_torch_export:"
   , indentEight "        raise"
   , indentFour "if require_torch_export:"
   , indentEight ("raise NotImplementedError(\"torch.export produced operators outside the " ++
       "current TorchLean import subset\")")
   , indentFour "from torch.fx import symbolic_trace"
   , indentFour "from torch.fx.passes.shape_prop import ShapeProp"
   , indentFour "gm = symbolic_trace(model)"
   , indentFour "ShapeProp(gm).propagate(example)"
   , indentFour "for node in gm.graph.nodes:"
   , indentEight "if node.op == 'get_attr':"
   , indentEight "    value = gm"
   , indentEight "    for part in str(node.target).split('.'): value = getattr(value, part)"
   , indentEight "    if isinstance(value, torch.Tensor): _TENSOR_VALUES[node.name] = value"
   , indentFour "_check_graph_operators(gm.graph, model)"
   , indentFour "return gm.graph, None"
   , "" ]

/-- The exported entry point: walk the captured graph and write the artifact. -/
def exportFunctionSection (options : GraphBridgeOptions) : Array String :=
  #[ s!"def {options.functionName}(model, example, json_path: str, require_torch_export=False):"
   , indentFour "graph, user_input_names = _capture(model, example, require_torch_export)"
   , indentFour "nodes = []"
   , indentFour "node_to_id = {}"
   , indentFour "output_ids = None"
   , indentFour "input_id = None"
   , indentFour "for node in graph.nodes:"
   , indentEight "if node.op == \"output\":"
   , indentEight "    refs = _node_refs(node.args, node_to_id)"
   , indentEight "    if not refs:"
   , indentEight ("        raise NotImplementedError(\"TorchLean graph import requires at least " ++
       "one tensor output\")")
   , indentEight "    output_ids = refs"
   , indentEight "    continue"
   , indentEight "if len(getattr(node, \"users\", {})) == 0:"
   , indentEight "    # FX often leaves dead tuple projections behind, e.g. attention weights from"
   , indentEight "    # `y, _ = mha(...)`. They are not part of the exported value, so we omit them"
   , indentEight ("    # instead of forcing every unused container projection to have a tensor " ++
       "lowering.")
   , indentEight "    node_to_id[node] = None"
   , indentEight "    continue"
   , indentEight "if node.op in (\"get_attr\",):"
   , indentEight "    continue"
   , indentEight "if node.op == \"placeholder\" and user_input_names is not None:"
   , indentEight "    if node.name not in user_input_names:"
   , indentEight "        # Parameters, buffers, constants, and effect tokens live outside"
   , indentEight "        # TorchLean's single user-input tensor dataflow graph."
   , indentEight "        node_to_id[node] = None"
   , indentEight "        continue"
   , indentEight "node_id = len(nodes)"
   , indentEight "node_to_id[node] = node_id"
   , indentEight "shape = _node_shape(node)"
   , indentEight "tuple_shapes = _tuple_shapes_of_node(node)"
   , indentEight ("value_meta = {\"value_kind\": \"tuple\", \"tuple_shapes\": tuple_shapes} if " ++
       "tuple_shapes else {\"value_kind\": \"tensor\", \"shape\": shape}")
   , indentEight "if node.op == \"placeholder\":"
   , indentEight "    if input_id is not None:"
   , indentEight ("        raise NotImplementedError(\"TorchLean graph import currently supports" ++
       " one user input\")")
   , indentEight ("    kind = " ++ Wire.kindObject .input)
   , indentEight "    parents = []"
   , indentEight "    input_id = node_id"
   , indentEight "elif tuple_shapes:"
   , indentEight "    # The importer expands known tuple producers into tensor dataflow."
   , indentEight "    kind = _tuple_kind(node, model)"
   , indentEight "    parents = _node_refs((node.args, node.kwargs), node_to_id)"
   , indentEight "elif node.op in (\"call_function\", \"call_method\", \"call_module\"):"
   , indentEight "    kind = _lower_kind(node, model)"
   , indentEight "    parents = _node_refs((node.args, node.kwargs), node_to_id)"
   , indentEight "else:"
   , indentEight "    raise NotImplementedError(f\"unsupported FX node op: {node.op}\")"
   , indentEight "entry = {\"id\": node_id, \"parents\": parents, **value_meta, **kind}"
   , indentEight "if " ++ pyBool options.includeDebugTargets ++ ":"
   , indentEight "    entry[\"debug_target\"] = _target_name(node.target)"
   , indentEight "nodes.append(entry)"
   , indentFour "if input_id is None or output_ids is None:"
   , indentEight "raise RuntimeError(\"could not identify graph input/output\")"
   , indentFour ("payload = {\"format\": FORMAT, \"input_id\": input_id, \"output_ids\": " ++
       "output_ids, \"nodes\": nodes}")
   , indentFour "try:"
   , indentEight "text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)"
   , indentFour "except ValueError as exc:"
   , indentEight ("raise ValueError(\"TorchLean IR JSON cannot carry NaN or infinite values;" ++
       " check the model parameters and constants\") from exc")
   , indentFour "with open(json_path, \"w\", encoding=\"utf-8\") as f:"
   , indentEight "f.write(text)"
   , indentFour "return payload"
   , "" ]

/-- Command-line `main` for the generated script. -/
def mainSection (options : GraphBridgeOptions) : Array String :=
  #[ "def main():"
   , indentFour ("parser = argparse.ArgumentParser(description=\"Export a PyTorch nn.Module to " ++
       "TorchLean IR JSON\")")
   , indentFour ("parser.add_argument(\"module\", help=\"Python file containing the model " ++
       "class/constructor\")")
   , indentFour ("parser.add_argument(\"ctor\", help=\"Zero-argument model class or constructor" ++
       " name\")")
   , indentFour "parser.add_argument(\"json\", help=\"Output graph JSON path\")"
   , indentFour ("parser.add_argument(\"--example-shape\", required=True, help=\"Comma-separated" ++
       " example input shape, e.g. 1,4\")")
   , indentFour ("parser.add_argument(\"--require-torch-export\", action=\"store_true\", " ++
       "help=\"Fail instead of falling back to FX\")")
   , indentFour "args = parser.parse_args()"
   , indentFour "shape = tuple(int(x) for x in args.example_shape.split(',') if x)"
   , indentFour "model = _load_model(args.module, args.ctor)"
   , indentFour "example = torch.randn(*shape)"
   , indentFour
       s!"payload = {options.functionName}(model, example, args.json, args.require_torch_export)"
   , indentFour "print(f\"wrote {len(payload['nodes'])} TorchLean IR nodes to {args.json}\")"
   , ""
   , "if __name__ == \"__main__\":"
   , indentFour "main()" ]

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
def generateGraphBridgeScript (options : GraphBridgeOptions := {}) : String :=
  joinLines <|
    bridgeImports ++ shapeHelpers ++ tupleKindSection ++ modelHelpers ++ payloadHelpers ++
      axisHelpers ++ lowerKindSection ++ lowerabilitySection ++ captureSection options ++
      exportFunctionSection options ++ mainSection options

end TorchExport
end PyTorch
end Export
