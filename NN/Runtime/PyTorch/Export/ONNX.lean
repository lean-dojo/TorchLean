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
bridge, and then the Lean side should call `NN.Runtime.PyTorch.Import.TorchExport.parseGraph`.

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
structure BridgeOptions where
  /-- Name of the Python helper function emitted into the script. -/
  functionName : String := "export_onnx_torchlean_graph_json"
  /-- Include ONNX node names/op types in each node for debugging. -/
  includeDebugTargets : Bool := true
deriving Repr

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
def shapeHelpers : Array String :=
  #[ "def _shape_size(shape):"
   , indentFour "n = 1"
   , indentFour "for d in shape:"
   , indentEight "n *= int(d)"
   , indentFour "return n"
   , ""
   , "def _flat_float_values(arr):"
   , indentFour "return [float(x) for x in arr.reshape(-1).tolist()]"
   , ""
   , "def _dim_value(dim):"
   , indentFour "if dim.HasField(\"dim_value\"):"
   , indentEight "return int(dim.dim_value)"
   , indentFour "raise RuntimeError(\"TorchLean ONNX import requires static tensor shapes\")"
   , ""
   , "def _shape_from_value_info(value_info):"
   , indentFour "tt = value_info.type.tensor_type"
   , indentFour "if not tt.HasField(\"shape\"):"
   , indentEight "raise RuntimeError(f\"missing shape for value {value_info.name}\")"
   , indentFour "return [_dim_value(d) for d in tt.shape.dim]"
   , ""
   , "def _collect_shapes(model):"
   , indentFour "shapes = {}"
   , indentFour ("for vi in list(model.graph.input) + list(model.graph.value_info) + " ++
       "list(model.graph.output):")
   , indentEight "if vi.type.HasField(\"tensor_type\"):"
   , indentEight "    shapes[vi.name] = _shape_from_value_info(vi)"
   , indentFour "for init in model.graph.initializer:"
   , indentEight "shapes[init.name] = [int(x) for x in init.dims]"
   , indentFour "return shapes"
   , ""
   , "def _attr(node, name, default=None):"
   , indentFour "for a in node.attribute:"
   , indentEight "if a.name == name:"
   , indentEight "    return onnx.helper.get_attribute_value(a)"
   , indentFour "return default"
   , ""
   , "def _opset(model):"
   , indentFour "for entry in model.opset_import:"
   , indentEight "if entry.domain in (\"\", \"ai.onnx\"):"
   , indentEight "    return int(entry.version)"
   , indentFour "raise RuntimeError(\"ONNX model does not import the default operator set\")"
   , ""
   , "def _norm_axis(op, axis, rank, allow_rank=False):"
   , indentFour "axis = int(axis)"
   , indentFour "axis = axis + rank if axis < 0 else axis"
   , indentFour "if axis < 0 or axis > rank or (axis == rank and not allow_rank):"
   , indentEight "raise RuntimeError(f\"{op}: axis {axis} is outside rank {rank}\")"
   , indentFour "return axis"
   , ""
   , "def _axis(node, rank, default):"
   , indentFour "return _norm_axis(node.op_type, _attr(node, \"axis\", default), rank)"
   , ""
   , "def _transpose_perm(node, rank):"
   , indentFour "perm = _attr(node, \"perm\", None)"
   , indentFour "if perm is None:"
   , indentEight "perm = list(reversed(range(rank)))"
   , indentFour "return [int(x) for x in perm]"
   , "" ]

/-- One-line `_lower_node` rule mapping an ONNX op type to a payload-free kind. -/
def rule (opType : String) (tag : OpTag) : String :=
  indentFour ("if op == \"" ++ opType ++ "\": return " ++ Wire.kindObject tag)

/-- `_lower_node`: single-node lowerings keyed on the ONNX op type. -/
def nodeLowering : Array String :=
  #[ "def _lower_node(node, out_shape, input_shapes):"
   , indentFour "op = node.op_type"
   , indentFour "rank = len(out_shape)"
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
   , indentFour ("if op == \"Softmax\": return {" ++ Wire.kindField .softmax ++
       ", \"axis\": _axis(node, rank, -1)}")
   , indentFour ("if op == \"Reshape\": return {" ++ Wire.kindField .reshape ++
       ", \"in_shape\": input_shapes[0], \"out_shape\": out_shape}")
   , indentFour "if op == \"Flatten\":"
   , indentEight "in_shape = input_shapes[0]"
   , indentEight "axis = _norm_axis(op, _attr(node, \"axis\", 1), len(in_shape), allow_rank=True)"
   , indentEight "flat = [_shape_size(in_shape[:axis]), _shape_size(in_shape[axis:])]"
   , indentEight "if list(out_shape) != flat:"
   , indentEight ("    raise RuntimeError(f\"Flatten: inferred shape {out_shape} differs from" ++
       " {flat}\")")
   , indentEight ("return {" ++ Wire.kindField .reshape ++
       ", \"in_shape\": in_shape, \"out_shape\": flat}")
   , indentFour ("if op == \"Concat\": return {" ++ Wire.kindField .concat ++
       ", \"axis\": _axis(node, rank, default=0)}")
   , indentFour "if op == \"Transpose\":"
   , indentEight "perm = _transpose_perm(node, len(input_shapes[0]))"
   , indentEight ("return {" ++ Wire.kindField .permute ++ ", \"perm\": perm}")
   , indentFour "raise RuntimeError(f\"Unsupported ONNX op for TorchLean IR import: {op}\")"
   , "" ]

/--
`_reduce_kind`: `ReduceSum`/`ReduceMean` with ONNX defaults. Axes come from the `axes` attribute
(before opset 13 for `ReduceSum`, before opset 18 for `ReduceMean`) or from a constant second input.
Missing or empty axes reduce every dimension unless `noop_with_empty_axes` is set, and `keepdims`
defaults to 1.
-/
def reduceLowering : Array String :=
  #[ "def _reduce_kind(node, input_names, input_shapes, initializers):"
   , indentFour "rank = len(input_shapes[0])"
   , indentFour "axes = _attr(node, \"axes\", None)"
   , indentFour "if len(input_names) > 1:"
   , indentEight "if axes is not None:"
   , indentEight ("    raise RuntimeError(f\"{node.op_type}: axes given both as attribute and" ++
       " input\")")
   , indentEight "if input_names[1] not in initializers:"
   , indentEight ("    raise RuntimeError(f\"{node.op_type}: axes input must be a graph" ++
       " initializer\")")
   , indentEight "axes = numpy_helper.to_array(initializers[input_names[1]]).reshape(-1).tolist()"
   , indentFour "axes = [] if axes is None else [int(a) for a in axes]"
   , indentFour "if not axes:"
   , indentEight "if int(_attr(node, \"noop_with_empty_axes\", 0)) != 0:"
   , indentEight ("    return {" ++ Wire.kindField .reshape ++
       ", \"in_shape\": input_shapes[0], \"out_shape\": input_shapes[0]}")
   , indentEight "axes = list(range(rank))"
   , indentFour "axes = [_norm_axis(node.op_type, a, rank) for a in axes]"
   , indentFour "if len(set(axes)) != len(axes):"
   , indentEight "raise RuntimeError(f\"{node.op_type}: duplicate reduction axes {axes}\")"
   , indentFour ("tag = " ++ Wire.quotedOpTag .reduceSum ++ " if node.op_type == \"ReduceSum\"" ++
       " else " ++ Wire.quotedOpTag .reduceMean)
   , indentFour ("return {\"kind\": tag, \"axes\": sorted(axes), " ++
       "\"keepdim\": bool(int(_attr(node, \"keepdims\", 1)))}")
   , "" ]

/--
`_lower_softmax_coerced`: `Softmax` before opset 13, whose default axis is 1 and which normalizes
over the input coerced to 2-D `[prod(d < axis), prod(d >= axis)]`.
-/
def softmaxCoercedLowering : Array String :=
  #[ ("def _lower_softmax_coerced(node, out_name, out_shape, input_names, input_shapes, " ++
       "name_to_id, add_node, include_debug):")
   , indentFour "x_shape = input_shapes[0]"
   , indentFour "rank = len(x_shape)"
   , indentFour "axis = _axis(node, rank, 1)"
   , indentFour "extra = _debug_extra(node) if include_debug else {}"
   , indentFour "if axis == rank - 1:"
   , indentEight ("add_node(" ++ Wire.quotedOpTag .softmax ++
       ", out_name, [name_to_id[input_names[0]]], out_shape, {\"axis\": axis, **extra})")
   , indentEight "return True"
   , indentFour "flat = [_shape_size(x_shape[:axis]), _shape_size(x_shape[axis:])]"
   , indentFour "flat_in = _shape_name(out_name, \"softmax_coerced_input\")"
   , indentFour "flat_out = _shape_name(out_name, \"softmax_coerced\")"
   , indentFour ("add_node(" ++ Wire.quotedOpTag .reshape ++
       ", flat_in, [name_to_id[input_names[0]]], flat, " ++
       "{\"in_shape\": x_shape, \"out_shape\": flat})")
   , indentFour ("add_node(" ++ Wire.quotedOpTag .softmax ++
       ", flat_out, [name_to_id[flat_in]], flat, {\"axis\": 1, **extra})")
   , indentFour ("add_node(" ++ Wire.quotedOpTag .reshape ++
       ", out_name, [name_to_id[flat_out]], out_shape, " ++
       "{\"in_shape\": flat, \"out_shape\": out_shape})")
   , indentFour "return True"
   , "" ]

/-- Kinds whose single parent is the first ONNX input. -/
def unaryTags : List OpTag :=
  [ .abs, .sqrt, .inv, .relu, .tanh, .sigmoid, .exp, .log, .sin, .cos, .softmax, .reduceSum
  , .reduceMean, .reshape, .flatten, .permute, .transpose ]

/-- Kinds that take exactly two ONNX inputs as parents. -/
def binaryTags : List OpTag :=
  [ .add, .sub, .mulElem, .maxElem, .minElem, .matmul ]

/-- `_ir_parent_names`: select the ONNX inputs that become IR parents for a lowered kind. -/
def parentNames : Array String :=
  #[ "def _ir_parent_names(op_type, tag, input_names):"
   , indentFour ("unary = " ++ Wire.quotedOpTags unaryTags)
   , indentFour ("binary = " ++ Wire.quotedOpTags binaryTags)
   , indentFour "if tag in unary:"
   , indentEight "if not input_names:"
   , indentEight "    raise RuntimeError(f\"{op_type}: expected at least one ONNX input\")"
   , indentEight "return [input_names[0]]"
   , indentFour "if tag in binary:"
   , indentEight "if len(input_names) != 2:"
   , indentEight ("    raise RuntimeError(f\"{op_type}: TorchLean IR lowering expected two" ++
       " tensor parents, got {len(input_names)}\")")
   , indentEight "return input_names"
   , indentFour ("if tag == " ++ Wire.quotedOpTag .concat ++ ":")
   , indentEight "if len(input_names) < 2:"
   , indentEight "    raise RuntimeError(\"Concat: expected at least two inputs\")"
   , indentEight "return input_names"
   , indentFour "return input_names"
   , ""
   , "def _debug_extra(node):"
   , indentFour "extra = {\"onnx_op_type\": node.op_type}"
   , indentFour "if node.name: extra[\"onnx_name\"] = node.name"
   , indentFour "return extra"
   , ""
   , "def _shape_name(base, suffix):"
   , indentFour "return f\"{base}__torchlean_{suffix}\""
   , "" ]

/-- `_conv_payload`: the dense weight and the bias of a `Conv` whose parameters are initializers.

Grouped kernels `[O, I/group, k...]` are expanded to the dense `[O, I, k...]` layout the IR conv
payload uses, with zeros outside each group's block, as the `torch.export` adapter does. A missing
bias becomes zeros.
-/
def convPayload : Array String :=
  #[ "def _conv_payload(input_names, initializers, out_c, in_c, group):"
   , indentFour "def initializer(label, name):"
   , indentEight "if name not in initializers:"
   , indentEight ("    raise RuntimeError(f\"Conv: {label} {name} must be a graph " ++
       "initializer to be carried in the TorchLean payload\")")
   , indentEight "return numpy_helper.to_array(initializers[name]).astype(np.float64)"
   , indentFour "weight = initializer(\"weight\", input_names[1])"
   , indentFour "if out_c % group != 0:"
   , indentEight ("raise RuntimeError(f\"Conv: out_channels={out_c} is not divisible by" ++
       " group={group}\")")
   , indentFour "out_per_group, in_per_group = out_c // group, in_c // group"
   , indentFour "dense = np.zeros((out_c, in_c) + tuple(weight.shape[2:]), dtype=np.float64)"
   , indentFour "for g in range(group):"
   , indentEight "rows = slice(g * out_per_group, (g + 1) * out_per_group)"
   , indentEight "dense[rows, g * in_per_group:(g + 1) * in_per_group] = weight[rows]"
   , indentFour "if len(input_names) > 2:"
   , indentEight "bias = initializer(\"bias\", input_names[2])"
   , indentEight "if list(bias.shape) != [out_c]:"
   , indentEight ("    raise RuntimeError(f\"Conv: bias must have shape [{out_c}], got" ++
       " {list(bias.shape)}\")")
   , indentFour "else:"
   , indentEight "bias = np.zeros(out_c, dtype=np.float64)"
   , indentFour "return {\"weight\": dense.tolist(), \"bias\": bias.tolist()}"
   , "" ]

/-- `_lower_conv`: channels-first `Conv` with explicit padding, optionally unwrapping a batch. -/
def convLowering : Array String :=
  #[ ("def _lower_conv(node, out_name, out_shape, input_names, input_shapes, name_to_id, " ++
       "add_node, include_debug, initializers):")
   , indentFour "if len(input_names) < 2:"
   , indentEight "raise RuntimeError(\"Conv: expected data and weight inputs\")"
   , indentFour "x_name, w_name = input_names[0], input_names[1]"
   , indentFour "x_shape, w_shape = input_shapes[0], input_shapes[1]"
   , indentFour "if len(w_shape) < 3:"
   , indentEight ("raise RuntimeError(f\"Conv: expected OI plus at least one spatial kernel " ++
       "dimension, got {w_shape}\")")
   , indentFour "spatial_rank = len(w_shape) - 2"
   , indentFour "group = int(_attr(node, \"group\", 1))"
   , indentFour "if group <= 0:"
   , indentEight "raise RuntimeError(f\"Conv: group must be positive, got {group}\")"
   , indentFour "dilations = [int(x) for x in _attr(node, \"dilations\", [1] * spatial_rank)]"
   , indentFour "strides = [int(x) for x in _attr(node, \"strides\", [1] * spatial_rank)]"
   , indentFour "pads = [int(x) for x in _attr(node, \"pads\", [0] * (2 * spatial_rank))]"
   , indentFour "auto_pad = _attr(node, \"auto_pad\", \"NOTSET\")"
   , indentFour "if isinstance(auto_pad, bytes): auto_pad = auto_pad.decode('utf-8')"
   , indentFour "if auto_pad not in (\"\", \"NOTSET\"):"
   , indentEight ("raise RuntimeError(f\"Conv: auto_pad={auto_pad!r} is unsupported; export " ++
       "explicit zero-padding extents\")")
   , indentFour "if len(strides) != spatial_rank:"
   , indentEight ("raise RuntimeError(f\"Conv: expected {spatial_rank} stride entries, got" ++
       " {strides}\")")
   , indentFour "if len(dilations) != spatial_rank or any(value <= 0 for value in dilations):"
   , indentEight ("raise RuntimeError(f\"Conv: expected {spatial_rank} positive dilation" ++
       " entries, got {dilations}\")")
   , indentFour "if len(pads) != 2 * spatial_rank:"
   , indentEight ("raise RuntimeError(f\"Conv: expected {2 * spatial_rank} explicit pad entries," ++
       " got {pads}\")")
   , indentFour "out_c, in_c = int(w_shape[0]), int(w_shape[1]) * group"
   , indentFour "kernel = [int(x) for x in w_shape[2:]]"
   , indentFour "padding = pads[:spatial_rank]"
   , indentFour "padding_after = pads[spatial_rank:]"
   , indentFour ("extra = {\"spatial_rank\": spatial_rank, \"kernel\": kernel, \"stride\":" ++
       " strides, " ++
       "\"padding\": padding, \"padding_after\": padding_after, \"dilation\": dilations, " ++
       "\"groups\": group, \"channel_axis\": 0, \"in_channels\": in_c, \"out_channels\": out_c, " ++
       "\"input_spatial\": [int(x) for x in x_shape[-spatial_rank:]]}")
   , indentFour "extra.update(_conv_payload(input_names, initializers, out_c, in_c, group))"
   , indentFour "if include_debug: extra.update(_debug_extra(node))"
   , indentFour "if len(x_shape) == spatial_rank + 1:"
   , indentEight ("add_node(" ++ Wire.quotedOpTag .conv ++
       ", out_name, [name_to_id[x_name]], out_shape, extra)")
   , indentEight "return True"
   , indentFour ("if len(x_shape) == spatial_rank + 2 and x_shape[0] == 1 and len(out_shape) == " ++
       "spatial_rank + 2 and out_shape[0] == 1:")
   , indentEight "sample_in = x_shape[1:]"
   , indentEight "sample_out = out_shape[1:]"
   , indentEight "reshape_in = _shape_name(out_name, \"conv_input_sample\")"
   , indentEight "conv_out = _shape_name(out_name, \"conv_sample\")"
   , indentEight ("add_node(" ++ Wire.quotedOpTag .reshape ++
       ", reshape_in, [name_to_id[x_name]], " ++
       "sample_in, {\"in_shape\": x_shape, \"out_shape\": sample_in})")
   , indentEight ("add_node(" ++ Wire.quotedOpTag .conv ++
       ", conv_out, [name_to_id[reshape_in]], sample_out, extra)")
   , indentEight ("add_node(" ++ Wire.quotedOpTag .reshape ++
       ", out_name, [name_to_id[conv_out]], " ++
       "out_shape, {\"in_shape\": sample_out, \"out_shape\": out_shape})")
   , indentEight "return True"
   , indentFour ("raise RuntimeError(f\"Conv: expected channels-first sample or" ++
       " singleton-batched input for spatial rank {spatial_rank}, got {x_shape}\")")
   , "" ]

/-- `_lower_gemm`: `Gemm` as matmul plus an optional broadcast bias add. -/
def gemmLowering : Array String :=
  #[ ("def _lower_gemm(node, out_name, out_shape, input_names, input_shapes, name_to_id, " ++
       "add_node, include_debug):")
   , indentFour "if len(input_names) < 2:"
   , indentEight "raise RuntimeError(\"Gemm: expected at least A and B inputs\")"
   , indentFour "alpha = float(_attr(node, \"alpha\", 1.0))"
   , indentFour "beta = float(_attr(node, \"beta\", 1.0))"
   , indentFour "if alpha != 1.0 or beta != 1.0:"
   , indentEight ("raise RuntimeError(\"Gemm: alpha/beta scaling is outside the current" ++
       " TorchLean graph adapter\")")
   , indentFour "if int(_attr(node, \"transA\", 0)) != 0:"
   , indentEight ("raise RuntimeError(\"Gemm: transA is outside the current TorchLean graph" ++
       " adapter\")")
   , indentFour "a_name, b_name = input_names[0], input_names[1]"
   , indentFour "b_parent = name_to_id[b_name]"
   , indentFour "b_shape = input_shapes[1]"
   , indentFour "if int(_attr(node, \"transB\", 0)) != 0:"
   , indentEight "if len(b_shape) != 2:"
   , indentEight "    raise RuntimeError(f\"Gemm: transB expects rank-2 B, got {b_shape}\")"
   , indentEight "b_t = _shape_name(out_name, \"gemm_B_t\")"
   , indentEight ("add_node(" ++ Wire.quotedOpTag .transpose ++
       ", b_t, [b_parent], [b_shape[1], " ++
       "b_shape[0]], {\"axis1\": 0, \"axis2\": 1})")
   , indentEight "b_parent = name_to_id[b_t]"
   , indentFour ("mat_name = out_name if len(input_names) == 2 else _shape_name(out_name, " ++
       "\"gemm_matmul\")")
   , indentFour ("add_node(" ++ Wire.quotedOpTag .matmul ++ ", mat_name, [name_to_id[a_name], " ++
       "b_parent], out_shape, _debug_extra(node) if include_debug else {})")
   , indentFour "if len(input_names) > 2:"
   , indentEight "c_name = input_names[2]"
   , indentEight "c_shape = input_shapes[2]"
   , indentEight "c_b = _shape_name(out_name, \"gemm_bias_broadcast\")"
   , indentEight ("add_node(" ++ Wire.quotedOpTag .broadcastTo ++ ", c_b, [name_to_id[c_name]], " ++
       "out_shape, {\"from_shape\": c_shape, \"to_shape\": out_shape})")
   , indentEight ("add_node(" ++ Wire.quotedOpTag .add ++ ", out_name, [name_to_id[mat_name], " ++
       "name_to_id[c_b]], out_shape, {})")
   , indentFour "return True"
   , "" ]

/--
`_lower_batchnorm`: inference-mode `BatchNormalization` expanded into elementwise nodes.

Each per-channel `[C]` tensor is reshaped to `[C, 1, ..., 1]` before broadcasting, because IR
`broadcastTo` aligns shapes from the right and the channel axis of `[N, C, ...]` is axis 1.
-/
def batchNormLowering : Array String :=
  #[ ("def _lower_batchnorm(node, out_name, out_shape, input_names, input_shapes, name_to_id, " ++
       "add_node, include_debug):")
   , indentFour "if len(input_names) < 5:"
   , indentEight ("raise RuntimeError(\"BatchNormalization: expected x, scale, bias, mean," ++
       " variance\")")
   , indentFour "if int(_attr(node, \"training_mode\", 0)) != 0:"
   , indentEight "raise RuntimeError(\"BatchNormalization: training_mode is not supported\")"
   , indentFour "x, scale, bias, mean, var = input_names[:5]"
   , indentFour "x_shape = input_shapes[0]"
   , indentFour "if len(x_shape) < 2:"
   , indentEight ("raise RuntimeError(f\"BatchNormalization: expected input rank at least 2," ++
       " got {x_shape}\")")
   , indentFour "channels = int(x_shape[1])"
   , indentFour "param_labels = (\"scale\", \"bias\", \"mean\", \"var\")"
   , indentFour "for label, shape in zip(param_labels, input_shapes[1:5]):"
   , indentEight "if list(shape) != [channels]:"
   , indentEight ("    raise RuntimeError(f\"BatchNormalization: {label} must have shape" ++
       " [{channels}], got {shape}\")")
   , indentFour "channel_shape = [channels]"
   , indentFour "channel_view = [channels] + [1] * (len(x_shape) - 2)"
   , indentFour "def channel_broadcast(target, source):"
   , indentEight "parent = name_to_id[source]"
   , indentEight "if channel_view != channel_shape:"
   , indentEight "    view = _shape_name(target, \"channel_view\")"
   , indentEight ("    add_node(" ++ Wire.quotedOpTag .reshape ++
       ", view, [parent], channel_view, " ++
       "{\"in_shape\": channel_shape, \"out_shape\": channel_view})")
   , indentEight "    parent = name_to_id[view]"
   , indentEight ("add_node(" ++ Wire.quotedOpTag .broadcastTo ++ ", target, [parent], x_shape, " ++
       "{\"from_shape\": channel_view, \"to_shape\": x_shape})")
   , indentFour "eps = _shape_name(out_name, \"bn_eps\")"
   , indentFour "var_eps = _shape_name(out_name, \"bn_var_eps\")"
   , indentFour "std = _shape_name(out_name, \"bn_std\")"
   , indentFour "inv_std = _shape_name(out_name, \"bn_inv_std\")"
   , indentFour "mean_b = _shape_name(out_name, \"bn_mean_broadcast\")"
   , indentFour "scale_b = _shape_name(out_name, \"bn_scale_broadcast\")"
   , indentFour "bias_b = _shape_name(out_name, \"bn_bias_broadcast\")"
   , indentFour "inv_b = _shape_name(out_name, \"bn_inv_broadcast\")"
   , indentFour "centered = _shape_name(out_name, \"bn_centered\")"
   , indentFour "normalized = _shape_name(out_name, \"bn_normalized\")"
   , indentFour "scaled = _shape_name(out_name, \"bn_scaled\")"
   , indentFour "eps_value = float(_attr(node, \"epsilon\", 1e-5))"
   , indentFour ("add_node(" ++ Wire.quotedOpTag .const ++
       ", eps, [], channel_shape, {\"value_shape\": " ++
       "channel_shape, \"values\": [eps_value for _ in range(channels)]})")
   , indentFour ("add_node(" ++ Wire.quotedOpTag .add ++
       ", var_eps, [name_to_id[var], name_to_id[eps]], channel_shape, {})")
   , indentFour ("add_node(" ++ Wire.quotedOpTag .sqrt ++
       ", std, [name_to_id[var_eps]], channel_shape, {})")
   , indentFour ("add_node(" ++ Wire.quotedOpTag .inv ++
       ", inv_std, [name_to_id[std]], channel_shape, {})")
   , indentFour "channel_broadcast(mean_b, mean)"
   , indentFour ("add_node(" ++ Wire.quotedOpTag .sub ++
       ", centered, [name_to_id[x], name_to_id[mean_b]], x_shape, {})")
   , indentFour "channel_broadcast(scale_b, scale)"
   , indentFour "channel_broadcast(bias_b, bias)"
   , indentFour "channel_broadcast(inv_b, inv_std)"
   , indentFour ("add_node(" ++ Wire.quotedOpTag .mulElem ++
       ", normalized, [name_to_id[centered], name_to_id[inv_b]], x_shape, {})")
   , indentFour ("add_node(" ++ Wire.quotedOpTag .mulElem ++
       ", scaled, [name_to_id[normalized], " ++
       "name_to_id[scale_b]], x_shape, _debug_extra(node) if include_debug else {})")
   , indentFour ("add_node(" ++ Wire.quotedOpTag .add ++
       ", out_name, [name_to_id[scaled], name_to_id[bias_b]], out_shape, {})")
   , indentFour "return True"
   , "" ]

/-- The exported entry point: walk the ONNX graph and write the artifact. -/
def entrypoint (options : BridgeOptions) : Array String :=
  let debug := pyBool options.includeDebugTargets
  let lowering (helper : String) : String :=
    indentEight ("    " ++ helper ++ "(node, out_name, out_shape, input_names, input_shapes, " ++
      "name_to_id, add_node, " ++ debug ++ ")")
  #[ "def " ++ options.functionName ++ "(onnx_path, out_json_path):"
   , indentFour "model = onnx.load(str(onnx_path))"
   , indentFour "model = shape_inference.infer_shapes(model)"
   , indentFour "shapes = _collect_shapes(model)"
   , indentFour "opset = _opset(model)"
   , indentFour "initializers = {init.name: init for init in model.graph.initializer}"
   , indentFour "initializer_names = set(initializers.keys())"
   , indentFour "nodes = []"
   , indentFour "name_to_id = {}"
   , indentFour "graph_inputs = [x for x in model.graph.input if x.name not in initializer_names]"
   , indentFour "if len(graph_inputs) != 1:"
   , indentEight ("raise RuntimeError(\"TorchLean ONNX import currently expects exactly one" ++
       " tensor graph input\")")
   , indentFour "def add_node(kind, name, parents, shape, extra=None):"
   , indentEight ("node = {\"id\": len(nodes), \"kind\": kind, \"parents\": parents, \"shape\":" ++
       " shape}")
   , indentEight "if extra: node.update(extra)"
   , indentEight "nodes.append(node)"
   , indentEight "name_to_id[name] = node[\"id\"]"
   , indentEight "return node[\"id\"]"
   , indentFour "input_name = graph_inputs[0].name"
   , indentFour ("add_node(" ++ Wire.quotedOpTag .input ++ ", input_name, [], shapes[input_name])")
   , indentFour "for name, init in initializers.items():"
   , indentEight "arr = numpy_helper.to_array(init)"
   , indentEight "shape = [int(x) for x in arr.shape]"
   , indentEight ("add_node(" ++ Wire.quotedOpTag .const ++
       ", name, [], shape, {\"value_shape\": " ++
       "shape, \"values\": _flat_float_values(arr)})")
   , indentFour "for node in model.graph.node:"
   , indentEight "if len(node.output) != 1:"
   , indentEight ("    raise RuntimeError(f\"ONNX node {node.name or node.op_type} has multiple " ++
       "outputs; tuple lowering is not implemented\")")
   , indentEight "out_name = node.output[0]"
   , indentEight "if out_name not in shapes:"
   , indentEight "    raise RuntimeError(f\"missing inferred shape for ONNX value {out_name}\")"
   , indentEight "input_names = []"
   , indentEight "input_shapes = []"
   , indentEight "for x in node.input:"
   , indentEight "    if not x: continue"
   , indentEight "    if x not in name_to_id:"
   , indentEight ("        raise RuntimeError(f\"ONNX input {x} for node {node.name or " ++
       "node.op_type} was not produced earlier\")")
   , indentEight "    input_names.append(x)"
   , indentEight "    input_shapes.append(shapes[x])"
   , indentEight "out_shape = shapes[out_name]"
   , indentEight "if node.op_type == \"Conv\":"
   , indentEight ("    _lower_conv(node, out_name, out_shape, input_names, input_shapes, " ++
       "name_to_id, add_node, " ++ debug ++ ", initializers)")
   , indentEight "    continue"
   , indentEight "if node.op_type == \"Gemm\":"
   , lowering "_lower_gemm"
   , indentEight "    continue"
   , indentEight "if node.op_type == \"BatchNormalization\":"
   , lowering "_lower_batchnorm"
   , indentEight "    continue"
   , indentEight "if node.op_type == \"Softmax\" and opset < 13:"
   , lowering "_lower_softmax_coerced"
   , indentEight "    continue"
   , indentEight "if node.op_type in (\"ReduceSum\", \"ReduceMean\"):"
   , indentEight "    kind = _reduce_kind(node, input_names, input_shapes, initializers)"
   , indentEight "else:"
   , indentEight "    kind = _lower_node(node, out_shape, input_shapes)"
   , indentEight "extra = dict(kind)"
   , indentEight "tag = extra.pop(\"kind\")"
   , indentEight ("parents = [name_to_id[x] for x in _ir_parent_names(node.op_type, tag," ++
       " input_names)]")
   , indentEight "if " ++ debug ++ ":"
   , indentEight "    extra[\"onnx_op_type\"] = node.op_type"
   , indentEight "    if node.name: extra[\"onnx_name\"] = node.name"
   , indentEight "add_node(tag, out_name, parents, out_shape, extra)"
   , indentFour "if len(model.graph.output) != 1:"
   , indentEight ("raise RuntimeError(\"TorchLean ONNX import currently expects exactly one" ++
       " graph output\")")
   , indentFour "output_name = model.graph.output[0].name"
   , indentFour "if output_name not in name_to_id:"
   , indentEight "raise RuntimeError(f\"ONNX graph output {output_name} was not produced\")"
   , indentFour ("artifact = {\"format\": FORMAT, \"input_id\": name_to_id[input_name], " ++
       "\"output_ids\": [name_to_id[output_name]], \"nodes\": nodes}")
   , indentFour "try:"
   , indentEight "text = json.dumps(artifact, indent=2, allow_nan=False)"
   , indentFour "except ValueError as exc:"
   , indentEight ("raise RuntimeError(\"TorchLean IR JSON cannot carry NaN or infinite values;" ++
       " check the ONNX initializers and attributes\") from exc")
   , indentFour "Path(out_json_path).write_text(text)"
   , indentFour "return artifact"
   , "" ]

/-- Command-line `main` for the generated script. -/
def mainFunction (options : BridgeOptions) : Array String :=
  #[ "def main():"
   , indentFour ("ap = argparse.ArgumentParser(description=\"Lower a conservative ONNX fragment" ++
       " to TorchLean IR JSON\")")
   , indentFour "ap.add_argument(\"onnx_path\")"
   , indentFour "ap.add_argument(\"out_json_path\")"
   , indentFour "args = ap.parse_args()"
   , indentFour options.functionName ++ "(args.onnx_path, args.out_json_path)"
   , ""
   , "if __name__ == \"__main__\":"
   , indentFour "main()" ]

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
def generateBridgeScript (options : BridgeOptions := {}) : String :=
  joinLines <|
    imports ++ shapeHelpers ++ nodeLowering ++ reduceLowering ++ softmaxCoercedLowering ++
      parentNames ++ convPayload ++ convLowering ++ gemmLowering ++ batchNormLowering ++
      entrypoint options ++ mainFunction options

end ONNX
end PyTorch
end Export
