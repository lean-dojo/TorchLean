/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Runtime.PyTorch.Wire

/-!
# ONNX Graph Adapter

This module emits a Python-side ONNX adapter for TorchLean graph import.

The adapter does not give ONNX a separate Lean semantics. It reads an ONNX model, lowers the
supported static graph fragment to the same `torchlean.ir.v1` JSON used by the `torch.export`/FX
bridge, and then the Lean side should call `Import.PyTorch.TorchExport.parseGraph`.

That boundary is intentional: ONNX parsing, protobuf handling, and shape inference stay outside
Lean; TorchLean accepts only the small checked graph artifact.

Every `"kind"` string written by the adapter comes from `NN.Runtime.PyTorch.Wire`, the table the
Lean importer parses with.
-/

@[expose] public section

namespace Export
namespace PyTorch
namespace ONNX

open Export.PyTorch
open NN.IR
open Interop.PyTorch

/-- Options for the generated ONNX-to-TorchLean-IR adapter script. -/
structure Options where
  /-- Name of the Python helper function emitted into the script. -/
  functionName : String := "export_onnx_torchlean_graph_json"
  /-- Include ONNX node names/op types in each node for debugging. -/
  includeDebugTargets : Bool := true
deriving Repr

namespace Internal

/-! ## Python sections

Each section is an array of Python lines ending in one blank separator line.
-/

/-- Imports and the artifact format marker. -/
def imports : Array String :=
  #[ "import argparse"
   , "import json"
   , "from pathlib import Path"
   , "import numpy as np"
   , "import onnx"
   , "from onnx import numpy_helper, shape_inference"
   , ""
   , "FORMAT = \"" ++ Wire.format ++ "\""
   , "" ]

/-- Static shape collection and attribute readers. -/
def shapes : Array String :=
  #[ "def _shape_size(shape):"
   , indent 4 "n = 1"
   , indent 4 "for d in shape:"
   , indent 8 "n *= int(d)"
   , indent 4 "return n"
   , ""
   , "def _flat_float_values(arr):"
   , indent 4 "return [float(x) for x in arr.reshape(-1).tolist()]"
   , ""
   , "def _dim_value(dim):"
   , indent 4 "if dim.HasField(\"dim_value\"):"
   , indent 8 "return int(dim.dim_value)"
   , indent 4 "raise RuntimeError(\"TorchLean ONNX import requires static tensor shapes\")"
   , ""
   , "def _shape_from_value_info(value_info):"
   , indent 4 "tt = value_info.type.tensor_type"
   , indent 4 "if not tt.HasField(\"shape\"):"
   , indent 8 "raise RuntimeError(f\"missing shape for value {value_info.name}\")"
   , indent 4 "return [_dim_value(d) for d in tt.shape.dim]"
   , ""
   , "def _collect_shapes(model):"
   , indent 4 "shapes = {}"
   , indent 4 ("for vi in list(model.graph.input) + list(model.graph.value_info) + " ++
       "list(model.graph.output):")
   , indent 8 "if vi.type.HasField(\"tensor_type\"):"
   , indent 8 "    shapes[vi.name] = _shape_from_value_info(vi)"
   , indent 4 "for init in model.graph.initializer:"
   , indent 8 "shapes[init.name] = [int(x) for x in init.dims]"
   , indent 4 "return shapes"
   , ""
   , "def _attr(node, name, default=None):"
   , indent 4 "for a in node.attribute:"
   , indent 8 "if a.name == name:"
   , indent 8 "    return onnx.helper.get_attribute_value(a)"
   , indent 4 "return default"
   , ""
   , "def _opset(model):"
   , indent 4 "for entry in model.opset_import:"
   , indent 8 "if entry.domain in (\"\", \"ai.onnx\"):"
   , indent 8 "    return int(entry.version)"
   , indent 4 "raise RuntimeError(\"ONNX model does not import the default operator set\")"
   , ""
   , "def _norm_axis(op, axis, rank, allow_rank=False):"
   , indent 4 "axis = int(axis)"
   , indent 4 "axis = axis + rank if axis < 0 else axis"
   , indent 4 "if axis < 0 or axis > rank or (axis == rank and not allow_rank):"
   , indent 8 "raise RuntimeError(f\"{op}: axis {axis} is outside rank {rank}\")"
   , indent 4 "return axis"
   , ""
   , "def _axis(node, rank, default):"
   , indent 4 "return _norm_axis(node.op_type, _attr(node, \"axis\", default), rank)"
   , ""
   , "def _transpose_perm(node, rank):"
   , indent 4 "perm = _attr(node, \"perm\", None)"
   , indent 4 "if perm is None:"
   , indent 8 "perm = list(reversed(range(rank)))"
   , indent 4 "return [int(x) for x in perm]"
   , "" ]

/-- One-line `_lower_node` rule mapping an ONNX op type to a payload-free kind. -/
def rule (opType : String) (tag : OpTag) : String :=
  indent 4 ("if op == \"" ++ opType ++ "\": return " ++ Wire.object tag)

/-- `_lower_node`: single-node lowerings keyed on the ONNX op type. -/
def nodes : Array String :=
  #[ "def _lower_node(node, out_shape, input_shapes):"
   , indent 4 "op = node.op_type"
   , indent 4 "rank = len(out_shape)"
   , rule "Add" .add
   , rule "Sub" .sub
   , rule "Mul" .mulElem
   , rule "Relu" .relu
   , rule "Tanh" .tanh
   , rule "Sigmoid" .sigmoid
   , rule "Exp" .exp
   , rule "Log" .log
   , rule "Sin" .sin
   , rule "Cos" .cos
   , rule "Abs" .abs
   , rule "Sqrt" .sqrt
   , rule "Reciprocal" .inv
   , rule "Max" .maxElem
   , rule "Min" .minElem
   , rule "MatMul" .matmul
   , indent 4 ("if op == \"Softmax\": return {" ++ Wire.field .softmax ++
       ", \"axis\": _axis(node, rank, -1)}")
   , indent 4 ("if op == \"Reshape\": return {" ++ Wire.field .reshape ++
       ", \"in_shape\": input_shapes[0], \"out_shape\": out_shape}")
   , indent 4 "if op == \"Flatten\":"
   , indent 8 "in_shape = input_shapes[0]"
   , indent 8 "axis = _norm_axis(op, _attr(node, \"axis\", 1), len(in_shape), allow_rank=True)"
   , indent 8 "flat = [_shape_size(in_shape[:axis]), _shape_size(in_shape[axis:])]"
   , indent 8 "if list(out_shape) != flat:"
   , indent 8 ("    raise RuntimeError(f\"Flatten: inferred shape {out_shape} differs from" ++
       " {flat}\")")
   , indent 8 ("return {" ++ Wire.field .reshape ++
       ", \"in_shape\": in_shape, \"out_shape\": flat}")
   , indent 4 ("if op == \"Concat\": return {" ++ Wire.field .concat ++
       ", \"axis\": _axis(node, rank, default=0)}")
   , indent 4 "if op == \"Transpose\":"
   , indent 8 "perm = _transpose_perm(node, len(input_shapes[0]))"
   , indent 8 ("return {" ++ Wire.field .permute ++ ", \"perm\": perm}")
   , indent 4 "raise RuntimeError(f\"Unsupported ONNX op for TorchLean IR import: {op}\")"
   , "" ]

/--
`_reduce_kind`: `ReduceSum`/`ReduceMean` with ONNX defaults. Axes come from the `axes` attribute
(before opset 13 for `ReduceSum`, before opset 18 for `ReduceMean`) or from a constant second input.
Missing or empty axes reduce every dimension unless `noop_with_empty_axes` is set, and `keepdims`
defaults to 1.
-/
def reductions : Array String :=
  #[ "def _reduce_kind(node, input_names, input_shapes, initializers):"
   , indent 4 "rank = len(input_shapes[0])"
   , indent 4 "axes = _attr(node, \"axes\", None)"
   , indent 4 "if len(input_names) > 1:"
   , indent 8 "if axes is not None:"
   , indent 8 ("    raise RuntimeError(f\"{node.op_type}: axes given both as attribute and" ++
       " input\")")
   , indent 8 "if input_names[1] not in initializers:"
   , indent 8 ("    raise RuntimeError(f\"{node.op_type}: axes input must be a graph" ++
       " initializer\")")
   , indent 8 "axes = numpy_helper.to_array(initializers[input_names[1]]).reshape(-1).tolist()"
   , indent 4 "axes = [] if axes is None else [int(a) for a in axes]"
   , indent 4 "if not axes:"
   , indent 8 "if int(_attr(node, \"noop_with_empty_axes\", 0)) != 0:"
   , indent 8 ("    return {" ++ Wire.field .reshape ++
       ", \"in_shape\": input_shapes[0], \"out_shape\": input_shapes[0]}")
   , indent 8 "axes = list(range(rank))"
   , indent 4 "axes = [_norm_axis(node.op_type, a, rank) for a in axes]"
   , indent 4 "if len(set(axes)) != len(axes):"
   , indent 8 "raise RuntimeError(f\"{node.op_type}: duplicate reduction axes {axes}\")"
   , indent 4 ("tag = " ++ Wire.quoted .reduceSum ++ " if node.op_type == \"ReduceSum\"" ++
       " else " ++ Wire.quoted .reduceMean)
   , indent 4 ("return {\"kind\": tag, \"axes\": sorted(axes), " ++
       "\"keepdim\": bool(int(_attr(node, \"keepdims\", 1)))}")
   , "" ]

/--
`_lower_softmax_coerced`: `Softmax` before opset 13, whose default axis is 1 and which normalizes
over the input coerced to 2-D `[prod(d < axis), prod(d >= axis)]`.
-/
def softmax : Array String :=
  #[ ("def _lower_softmax_coerced(node, out_name, out_shape, input_names, input_shapes, " ++
       "name_to_id, add_node, include_debug):")
   , indent 4 "x_shape = input_shapes[0]"
   , indent 4 "rank = len(x_shape)"
   , indent 4 "axis = _axis(node, rank, 1)"
   , indent 4 "extra = _debug_extra(node) if include_debug else {}"
   , indent 4 "if axis == rank - 1:"
   , indent 8 ("add_node(" ++ Wire.quoted .softmax ++
       ", out_name, [name_to_id[input_names[0]]], out_shape, {\"axis\": axis, **extra})")
   , indent 8 "return True"
   , indent 4 "flat = [_shape_size(x_shape[:axis]), _shape_size(x_shape[axis:])]"
   , indent 4 "flat_in = _shape_name(out_name, \"softmax_coerced_input\")"
   , indent 4 "flat_out = _shape_name(out_name, \"softmax_coerced\")"
   , indent 4 ("add_node(" ++ Wire.quoted .reshape ++
       ", flat_in, [name_to_id[input_names[0]]], flat, " ++
       "{\"in_shape\": x_shape, \"out_shape\": flat})")
   , indent 4 ("add_node(" ++ Wire.quoted .softmax ++
       ", flat_out, [name_to_id[flat_in]], flat, {\"axis\": 1, **extra})")
   , indent 4 ("add_node(" ++ Wire.quoted .reshape ++
       ", out_name, [name_to_id[flat_out]], out_shape, " ++
       "{\"in_shape\": flat, \"out_shape\": out_shape})")
   , indent 4 "return True"
   , "" ]

/-- Kinds whose single parent is the first ONNX input. -/
def unary : List OpTag :=
  [ .abs, .sqrt, .inv, .relu, .tanh, .sigmoid, .exp, .log, .sin, .cos, .softmax, .reduceSum
  , .reduceMean, .reshape, .flatten, .permute, .transpose ]

/-- Kinds that take exactly two ONNX inputs as parents. -/
def binary : List OpTag :=
  [ .add, .sub, .mulElem, .maxElem, .minElem, .matmul ]

/-- `_ir_parent_names`: select the ONNX inputs that become IR parents for a lowered kind. -/
def parents : Array String :=
  #[ "def _ir_parent_names(op_type, tag, input_names):"
   , indent 4 ("unary = " ++ Wire.set unary)
   , indent 4 ("binary = " ++ Wire.set binary)
   , indent 4 "if tag in unary:"
   , indent 8 "if not input_names:"
   , indent 8 "    raise RuntimeError(f\"{op_type}: expected at least one ONNX input\")"
   , indent 8 "return [input_names[0]]"
   , indent 4 "if tag in binary:"
   , indent 8 "if len(input_names) != 2:"
   , indent 8 ("    raise RuntimeError(f\"{op_type}: TorchLean IR lowering expected two" ++
       " tensor parents, got {len(input_names)}\")")
   , indent 8 "return input_names"
   , indent 4 ("if tag == " ++ Wire.quoted .concat ++ ":")
   , indent 8 "if len(input_names) < 2:"
   , indent 8 "    raise RuntimeError(\"Concat: expected at least two inputs\")"
   , indent 8 "return input_names"
   , indent 4 "return input_names"
   , ""
   , "def _debug_extra(node):"
   , indent 4 "extra = {\"onnx_op_type\": node.op_type}"
   , indent 4 "if node.name: extra[\"onnx_name\"] = node.name"
   , indent 4 "return extra"
   , ""
   , "def _shape_name(base, suffix):"
   , indent 4 "return f\"{base}__torchlean_{suffix}\""
   , "" ]

/-- `_conv_payload`: the dense weight and the bias of a `Conv` whose parameters are initializers.

Grouped kernels `[O, I/group, k...]` are expanded to the dense `[O, I, k...]` layout the IR conv
payload uses, with zeros outside each group's block, as the `torch.export` adapter does. A missing
bias becomes zeros.
-/
def payload : Array String :=
  #[ "def _conv_payload(input_names, initializers, out_c, in_c, group):"
   , indent 4 "def initializer(label, name):"
   , indent 8 "if name not in initializers:"
   , indent 8 ("    raise RuntimeError(f\"Conv: {label} {name} must be a graph " ++
       "initializer to be carried in the TorchLean payload\")")
   , indent 8 "return numpy_helper.to_array(initializers[name]).astype(np.float64)"
   , indent 4 "weight = initializer(\"weight\", input_names[1])"
   , indent 4 "if out_c % group != 0:"
   , indent 8 ("raise RuntimeError(f\"Conv: out_channels={out_c} is not divisible by" ++
       " group={group}\")")
   , indent 4 "out_per_group, in_per_group = out_c // group, in_c // group"
   , indent 4 "dense = np.zeros((out_c, in_c) + tuple(weight.shape[2:]), dtype=np.float64)"
   , indent 4 "for g in range(group):"
   , indent 8 "rows = slice(g * out_per_group, (g + 1) * out_per_group)"
   , indent 8 "dense[rows, g * in_per_group:(g + 1) * in_per_group] = weight[rows]"
   , indent 4 "if len(input_names) > 2:"
   , indent 8 "bias = initializer(\"bias\", input_names[2])"
   , indent 8 "if list(bias.shape) != [out_c]:"
   , indent 8 ("    raise RuntimeError(f\"Conv: bias must have shape [{out_c}], got" ++
       " {list(bias.shape)}\")")
   , indent 4 "else:"
   , indent 8 "bias = np.zeros(out_c, dtype=np.float64)"
   , indent 4 "return {\"weight\": dense.tolist(), \"bias\": bias.tolist()}"
   , "" ]

/-- `_lower_conv`: channels-first `Conv` with explicit padding, optionally unwrapping a batch. -/
def convolution : Array String :=
  #[ ("def _lower_conv(node, out_name, out_shape, input_names, input_shapes, name_to_id, " ++
       "add_node, include_debug, initializers):")
   , indent 4 "if len(input_names) < 2:"
   , indent 8 "raise RuntimeError(\"Conv: expected data and weight inputs\")"
   , indent 4 "x_name = input_names[0]"
   , indent 4 "x_shape, w_shape = input_shapes[0], input_shapes[1]"
   , indent 4 "if len(w_shape) < 3:"
   , indent 8 ("raise RuntimeError(f\"Conv: expected OI plus at least one spatial kernel " ++
       "dimension, got {w_shape}\")")
   , indent 4 "spatial_rank = len(w_shape) - 2"
   , indent 4 "group = int(_attr(node, \"group\", 1))"
   , indent 4 "if group <= 0:"
   , indent 8 "raise RuntimeError(f\"Conv: group must be positive, got {group}\")"
   , indent 4 "dilations = [int(x) for x in _attr(node, \"dilations\", [1] * spatial_rank)]"
   , indent 4 "strides = [int(x) for x in _attr(node, \"strides\", [1] * spatial_rank)]"
   , indent 4 "pads = [int(x) for x in _attr(node, \"pads\", [0] * (2 * spatial_rank))]"
   , indent 4 "auto_pad = _attr(node, \"auto_pad\", \"NOTSET\")"
   , indent 4 "if isinstance(auto_pad, bytes): auto_pad = auto_pad.decode('utf-8')"
   , indent 4 "if auto_pad not in (\"\", \"NOTSET\"):"
   , indent 8 ("raise RuntimeError(f\"Conv: auto_pad={auto_pad!r} is unsupported; export " ++
       "explicit zero-padding extents\")")
   , indent 4 "if len(strides) != spatial_rank:"
   , indent 8 ("raise RuntimeError(f\"Conv: expected {spatial_rank} stride entries, got" ++
       " {strides}\")")
   , indent 4 "if len(dilations) != spatial_rank or any(value <= 0 for value in dilations):"
   , indent 8 ("raise RuntimeError(f\"Conv: expected {spatial_rank} positive dilation" ++
       " entries, got {dilations}\")")
   , indent 4 "if len(pads) != 2 * spatial_rank:"
   , indent 8 ("raise RuntimeError(f\"Conv: expected {2 * spatial_rank} explicit pad entries," ++
       " got {pads}\")")
   , indent 4 "out_c, in_c = int(w_shape[0]), int(w_shape[1]) * group"
   , indent 4 "kernel = [int(x) for x in w_shape[2:]]"
   , indent 4 "padding = pads[:spatial_rank]"
   , indent 4 "padding_after = pads[spatial_rank:]"
   , indent 4 ("extra = {\"spatial_rank\": spatial_rank, \"kernel\": kernel, \"stride\":" ++
       " strides, " ++
       "\"padding\": padding, \"padding_after\": padding_after, \"dilation\": dilations, " ++
       "\"groups\": group, \"channel_axis\": 0, \"in_channels\": in_c, \"out_channels\": out_c, " ++
       "\"input_spatial\": [int(x) for x in x_shape[-spatial_rank:]]}")
   , indent 4 "extra.update(_conv_payload(input_names, initializers, out_c, in_c, group))"
   , indent 4 "if include_debug: extra.update(_debug_extra(node))"
   , indent 4 "if len(x_shape) == spatial_rank + 1:"
   , indent 8 ("add_node(" ++ Wire.quoted .conv ++
       ", out_name, [name_to_id[x_name]], out_shape, extra)")
   , indent 8 "return True"
   , indent 4 ("if len(x_shape) == spatial_rank + 2 and x_shape[0] == 1 and len(out_shape) == " ++
       "spatial_rank + 2 and out_shape[0] == 1:")
   , indent 8 "sample_in = x_shape[1:]"
   , indent 8 "sample_out = out_shape[1:]"
   , indent 8 "reshape_in = _shape_name(out_name, \"conv_input_sample\")"
   , indent 8 "conv_out = _shape_name(out_name, \"conv_sample\")"
   , indent 8 ("add_node(" ++ Wire.quoted .reshape ++
       ", reshape_in, [name_to_id[x_name]], " ++
       "sample_in, {\"in_shape\": x_shape, \"out_shape\": sample_in})")
   , indent 8 ("add_node(" ++ Wire.quoted .conv ++
       ", conv_out, [name_to_id[reshape_in]], sample_out, extra)")
   , indent 8 ("add_node(" ++ Wire.quoted .reshape ++
       ", out_name, [name_to_id[conv_out]], " ++
       "out_shape, {\"in_shape\": sample_out, \"out_shape\": out_shape})")
   , indent 8 "return True"
   , indent 4 ("raise RuntimeError(f\"Conv: expected channels-first sample or" ++
       " singleton-batched input for spatial rank {spatial_rank}, got {x_shape}\")")
   , "" ]

/-- `_lower_gemm`: `Gemm` as matmul plus an optional broadcast bias add. -/
def gemm : Array String :=
  #[ ("def _lower_gemm(node, out_name, out_shape, input_names, input_shapes, name_to_id, " ++
       "add_node, include_debug):")
   , indent 4 "if len(input_names) < 2:"
   , indent 8 "raise RuntimeError(\"Gemm: expected at least A and B inputs\")"
   , indent 4 "alpha = float(_attr(node, \"alpha\", 1.0))"
   , indent 4 "beta = float(_attr(node, \"beta\", 1.0))"
   , indent 4 "if alpha != 1.0 or beta != 1.0:"
   , indent 8 ("raise RuntimeError(\"Gemm: alpha/beta scaling is outside the current" ++
       " TorchLean graph adapter\")")
   , indent 4 "if int(_attr(node, \"transA\", 0)) != 0:"
   , indent 8 ("raise RuntimeError(\"Gemm: transA is outside the current TorchLean graph" ++
       " adapter\")")
   , indent 4 "a_name, b_name = input_names[0], input_names[1]"
   , indent 4 "b_parent = name_to_id[b_name]"
   , indent 4 "b_shape = input_shapes[1]"
   , indent 4 "if int(_attr(node, \"transB\", 0)) != 0:"
   , indent 8 "if len(b_shape) != 2:"
   , indent 8 "    raise RuntimeError(f\"Gemm: transB expects rank-2 B, got {b_shape}\")"
   , indent 8 "b_t = _shape_name(out_name, \"gemm_B_t\")"
   , indent 8 ("add_node(" ++ Wire.quoted .transpose ++
       ", b_t, [b_parent], [b_shape[1], " ++
       "b_shape[0]], {\"axis1\": 0, \"axis2\": 1})")
   , indent 8 "b_parent = name_to_id[b_t]"
   , indent 4 ("mat_name = out_name if len(input_names) == 2 else _shape_name(out_name, " ++
       "\"gemm_matmul\")")
   , indent 4 ("add_node(" ++ Wire.quoted .matmul ++ ", mat_name, [name_to_id[a_name], " ++
       "b_parent], out_shape, _debug_extra(node) if include_debug else {})")
   , indent 4 "if len(input_names) > 2:"
   , indent 8 "c_name = input_names[2]"
   , indent 8 "c_shape = input_shapes[2]"
   , indent 8 "c_b = _shape_name(out_name, \"gemm_bias_broadcast\")"
   , indent 8 ("add_node(" ++ Wire.quoted .broadcastTo ++ ", c_b, [name_to_id[c_name]], " ++
       "out_shape, {\"from_shape\": c_shape, \"to_shape\": out_shape})")
   , indent 8 ("add_node(" ++ Wire.quoted .add ++ ", out_name, [name_to_id[mat_name], " ++
       "name_to_id[c_b]], out_shape, {})")
   , indent 4 "return True"
   , "" ]

/--
`_lower_batchnorm`: inference-mode `BatchNormalization` expanded into elementwise nodes.

Each per-channel `[C]` tensor is reshaped to `[C, 1, ..., 1]` before broadcasting, because IR
`broadcastTo` aligns shapes from the right and the channel axis of `[N, C, ...]` is axis 1.
-/
def batchNorm : Array String :=
  #[ ("def _lower_batchnorm(node, out_name, out_shape, input_names, input_shapes, name_to_id, " ++
       "add_node, include_debug):")
   , indent 4 "if len(input_names) < 5:"
   , indent 8 ("raise RuntimeError(\"BatchNormalization: expected x, scale, bias, mean," ++
       " variance\")")
   , indent 4 "if int(_attr(node, \"training_mode\", 0)) != 0:"
   , indent 8 "raise RuntimeError(\"BatchNormalization: training_mode is not supported\")"
   , indent 4 "x, scale, bias, mean, var = input_names[:5]"
   , indent 4 "x_shape = input_shapes[0]"
   , indent 4 "if len(x_shape) < 2:"
   , indent 8 ("raise RuntimeError(f\"BatchNormalization: expected input rank at least 2," ++
       " got {x_shape}\")")
   , indent 4 "channels = int(x_shape[1])"
   , indent 4 "param_labels = (\"scale\", \"bias\", \"mean\", \"var\")"
   , indent 4 "for label, shape in zip(param_labels, input_shapes[1:5]):"
   , indent 8 "if list(shape) != [channels]:"
   , indent 8 ("    raise RuntimeError(f\"BatchNormalization: {label} must have shape" ++
       " [{channels}], got {shape}\")")
   , indent 4 "channel_shape = [channels]"
   , indent 4 "channel_view = [channels] + [1] * (len(x_shape) - 2)"
   , indent 4 "def channel_broadcast(target, source):"
   , indent 8 "parent = name_to_id[source]"
   , indent 8 "if channel_view != channel_shape:"
   , indent 8 "    view = _shape_name(target, \"channel_view\")"
   , indent 8 ("    add_node(" ++ Wire.quoted .reshape ++
       ", view, [parent], channel_view, " ++
       "{\"in_shape\": channel_shape, \"out_shape\": channel_view})")
   , indent 8 "    parent = name_to_id[view]"
   , indent 8 ("add_node(" ++ Wire.quoted .broadcastTo ++ ", target, [parent], x_shape, " ++
       "{\"from_shape\": channel_view, \"to_shape\": x_shape})")
   , indent 4 "eps = _shape_name(out_name, \"bn_eps\")"
   , indent 4 "var_eps = _shape_name(out_name, \"bn_var_eps\")"
   , indent 4 "std = _shape_name(out_name, \"bn_std\")"
   , indent 4 "inv_std = _shape_name(out_name, \"bn_inv_std\")"
   , indent 4 "mean_b = _shape_name(out_name, \"bn_mean_broadcast\")"
   , indent 4 "scale_b = _shape_name(out_name, \"bn_scale_broadcast\")"
   , indent 4 "bias_b = _shape_name(out_name, \"bn_bias_broadcast\")"
   , indent 4 "inv_b = _shape_name(out_name, \"bn_inv_broadcast\")"
   , indent 4 "centered = _shape_name(out_name, \"bn_centered\")"
   , indent 4 "normalized = _shape_name(out_name, \"bn_normalized\")"
   , indent 4 "scaled = _shape_name(out_name, \"bn_scaled\")"
   , indent 4 "eps_value = float(_attr(node, \"epsilon\", 1e-5))"
   , indent 4 ("add_node(" ++ Wire.quoted .const ++
       ", eps, [], channel_shape, {\"value_shape\": " ++
       "channel_shape, \"values\": [eps_value for _ in range(channels)]})")
   , indent 4 ("add_node(" ++ Wire.quoted .add ++
       ", var_eps, [name_to_id[var], name_to_id[eps]], channel_shape, {})")
   , indent 4 ("add_node(" ++ Wire.quoted .sqrt ++
       ", std, [name_to_id[var_eps]], channel_shape, {})")
   , indent 4 ("add_node(" ++ Wire.quoted .inv ++
       ", inv_std, [name_to_id[std]], channel_shape, {})")
   , indent 4 "channel_broadcast(mean_b, mean)"
   , indent 4 ("add_node(" ++ Wire.quoted .sub ++
       ", centered, [name_to_id[x], name_to_id[mean_b]], x_shape, {})")
   , indent 4 "channel_broadcast(scale_b, scale)"
   , indent 4 "channel_broadcast(bias_b, bias)"
   , indent 4 "channel_broadcast(inv_b, inv_std)"
   , indent 4 ("add_node(" ++ Wire.quoted .mulElem ++
       ", normalized, [name_to_id[centered], name_to_id[inv_b]], x_shape, {})")
   , indent 4 ("add_node(" ++ Wire.quoted .mulElem ++
       ", scaled, [name_to_id[normalized], " ++
       "name_to_id[scale_b]], x_shape, _debug_extra(node) if include_debug else {})")
   , indent 4 ("add_node(" ++ Wire.quoted .add ++
       ", out_name, [name_to_id[scaled], name_to_id[bias_b]], out_shape, {})")
   , indent 4 "return True"
   , "" ]

/-- The exported entry point: walk the ONNX graph and write the artifact. -/
def entrypoint (options : Options) : Array String :=
  let debug := boolLiteral options.includeDebugTargets
  let lowering (helper : String) : String :=
    indent 8 ("    " ++ helper ++ "(node, out_name, out_shape, input_names, input_shapes, " ++
      "name_to_id, add_node, " ++ debug ++ ")")
  #[ "def " ++ options.functionName ++ "(onnx_path, out_json_path):"
   , indent 4 "model = onnx.load(str(onnx_path))"
   , indent 4 "model = shape_inference.infer_shapes(model)"
   , indent 4 "shapes = _collect_shapes(model)"
   , indent 4 "opset = _opset(model)"
   , indent 4 "initializers = {init.name: init for init in model.graph.initializer}"
   , indent 4 "initializer_names = set(initializers.keys())"
   , indent 4 "nodes = []"
   , indent 4 "name_to_id = {}"
   , indent 4 "graph_inputs = [x for x in model.graph.input if x.name not in initializer_names]"
   , indent 4 "if len(graph_inputs) != 1:"
   , indent 8 ("raise RuntimeError(\"TorchLean ONNX import currently expects exactly one" ++
       " tensor graph input\")")
   , indent 4 "def add_node(kind, name, parents, shape, extra=None):"
   , indent 8 ("node = {\"id\": len(nodes), \"kind\": kind, \"parents\": parents, \"shape\":" ++
       " shape}")
   , indent 8 "if extra: node.update(extra)"
   , indent 8 "nodes.append(node)"
   , indent 8 "name_to_id[name] = node[\"id\"]"
   , indent 8 "return node[\"id\"]"
   , indent 4 "input_name = graph_inputs[0].name"
   , indent 4 ("add_node(" ++ Wire.quoted .input ++ ", input_name, [], shapes[input_name])")
   , indent 4 "for name, init in initializers.items():"
   , indent 8 "arr = numpy_helper.to_array(init)"
   , indent 8 "shape = [int(x) for x in arr.shape]"
   , indent 8 ("add_node(" ++ Wire.quoted .const ++
       ", name, [], shape, {\"value_shape\": " ++
       "shape, \"values\": _flat_float_values(arr)})")
   , indent 4 "for node in model.graph.node:"
   , indent 8 "if len(node.output) != 1:"
   , indent 8 ("    raise RuntimeError(f\"ONNX node {node.name or node.op_type} has multiple " ++
       "outputs; tuple lowering is not implemented\")")
   , indent 8 "out_name = node.output[0]"
   , indent 8 "if out_name not in shapes:"
   , indent 8 "    raise RuntimeError(f\"missing inferred shape for ONNX value {out_name}\")"
   , indent 8 "input_names = []"
   , indent 8 "input_shapes = []"
   , indent 8 "for x in node.input:"
   , indent 8 "    if not x: continue"
   , indent 8 "    if x not in name_to_id:"
   , indent 8 ("        raise RuntimeError(f\"ONNX input {x} for node {node.name or " ++
       "node.op_type} was not produced earlier\")")
   , indent 8 "    input_names.append(x)"
   , indent 8 "    input_shapes.append(shapes[x])"
   , indent 8 "out_shape = shapes[out_name]"
   , indent 8 "if node.op_type == \"Conv\":"
   , indent 8 ("    _lower_conv(node, out_name, out_shape, input_names, input_shapes, " ++
       "name_to_id, add_node, " ++ debug ++ ", initializers)")
   , indent 8 "    continue"
   , indent 8 "if node.op_type == \"Gemm\":"
   , lowering "_lower_gemm"
   , indent 8 "    continue"
   , indent 8 "if node.op_type == \"BatchNormalization\":"
   , lowering "_lower_batchnorm"
   , indent 8 "    continue"
   , indent 8 "if node.op_type == \"Softmax\" and opset < 13:"
   , lowering "_lower_softmax_coerced"
   , indent 8 "    continue"
   , indent 8 "if node.op_type in (\"ReduceSum\", \"ReduceMean\"):"
   , indent 8 "    kind = _reduce_kind(node, input_names, input_shapes, initializers)"
   , indent 8 "else:"
   , indent 8 "    kind = _lower_node(node, out_shape, input_shapes)"
   , indent 8 "extra = dict(kind)"
   , indent 8 "tag = extra.pop(\"kind\")"
   , indent 8 ("parents = [name_to_id[x] for x in _ir_parent_names(node.op_type, tag," ++
       " input_names)]")
   , indent 8 "if " ++ debug ++ ":"
   , indent 8 "    extra[\"onnx_op_type\"] = node.op_type"
   , indent 8 "    if node.name: extra[\"onnx_name\"] = node.name"
   , indent 8 "add_node(tag, out_name, parents, out_shape, extra)"
   , indent 4 "if len(model.graph.output) != 1:"
   , indent 8 ("raise RuntimeError(\"TorchLean ONNX import currently expects exactly one" ++
       " graph output\")")
   , indent 4 "output_name = model.graph.output[0].name"
   , indent 4 "if output_name not in name_to_id:"
   , indent 8 "raise RuntimeError(f\"ONNX graph output {output_name} was not produced\")"
   , indent 4 ("artifact = {\"format\": FORMAT, \"input_id\": name_to_id[input_name], " ++
       "\"output_ids\": [name_to_id[output_name]], \"nodes\": nodes}")
   , indent 4 "try:"
   , indent 8 "text = json.dumps(artifact, indent=2, allow_nan=False)"
   , indent 4 "except ValueError as exc:"
   , indent 8 ("raise RuntimeError(\"TorchLean IR JSON cannot carry NaN or infinite values;" ++
       " check the ONNX initializers and attributes\") from exc")
   , indent 4 "Path(out_json_path).write_text(text)"
   , indent 4 "return artifact"
   , "" ]

/-- Command-line `main` for the generated script. -/
def cli (options : Options) : Array String :=
  #[ "def main():"
   , indent 4 ("ap = argparse.ArgumentParser(description=\"Lower a conservative ONNX fragment" ++
       " to TorchLean IR JSON\")")
   , indent 4 "ap.add_argument(\"onnx_path\")"
   , indent 4 "ap.add_argument(\"out_json_path\")"
   , indent 4 "args = ap.parse_args()"
   , indent 4 options.functionName ++ "(args.onnx_path, args.out_json_path)"
   , ""
   , "if __name__ == \"__main__\":"
   , indent 4 "main()" ]

end Internal

/--
Emit a Python script that lowers a conservative ONNX fragment to `torchlean.ir.v1`.

Supported first-pass ops are tensor ops that map to current `NN.IR.OpKind`: elementwise
arithmetic/activations, `MatMul`, `ReduceSum`, `ReduceMean`, `Softmax`, `Reshape`, `Flatten`,
`Concat`, `Transpose`, `Gemm`, inference-style `BatchNormalization`, and arbitrary-rank
channels-first `Conv`. `Flatten` becomes the 2-D reshape ONNX specifies, and `Softmax` before
opset 13 is lowered through the same 2-D coercion. Graph initializers become constant nodes whose
values `Import.PyTorch.TorchExport.parsePayload` reads back. A `Conv` node carries its dense weight
and bias in the payload, so its weight and optional bias must be initializers.
NaN or infinite values are rejected because the JSON artifact cannot represent them.
-/
def script (options : Options := {}) : String :=
  joinLines <|
    Internal.imports ++ Internal.shapes ++ Internal.nodes ++ Internal.reductions ++
      Internal.softmax ++ Internal.parents ++ Internal.payload ++ Internal.convolution ++
      Internal.gemm ++ Internal.batchNorm ++ Internal.entrypoint options ++ Internal.cli options

end ONNX
end PyTorch
end Export
