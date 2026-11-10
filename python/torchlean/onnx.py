"""Write the static fragment of an ONNX model that TorchLean supports as a typed IR artifact.

Run `python3 -m torchlean.onnx MODEL.onnx OUT.json [--sources SOURCES.json]`, or call
`export_onnx_torchlean_graph_json`. Run it as a module: as a script path, its directory would come
first on `sys.path` and `import onnx` would import this file. The artifact (`IR.Artifact`)
has one input step per graph input that is not an initializer and, for each initializer a node
reads as data, a constant step holding the initializer's own words. An operator step holds the
operator's codec record, and every step has the floating element type of the ONNX value it
computes. Lean decodes the document with `IR.Artifact.ofJson`, checks it and builds its
payload with `Artifact.load`, and runs it with `IR.Program.eval`. ONNX parsing, protobuf
handling and shape inference stay in Python.

The format name, the record version, the codes of a sum, of a true division and of the IEEE square
root, and the JSON of each element type come from Lean's op table (`op_table()`). The record of an
operator generalized over its element type ends with the element field of the scalar,
`{"element": "scalar"}`. The lowering follows each supported operator's ONNX version: NumPy
broadcasting by `Op.broadcast` steps (the `broadcast` and `axis` attributes below opset 7) and
NumPy's `MatMul`. It computes `Conv` with its groups and every leading axis, and inference
`BatchNormalization` as `(x - mean) / sqrt(var + epsilon) * scale + bias`. It follows the coercion
of `Softmax` below opset 13 and `Concat`'s default axis below opset 4. A `ReduceSum` or `ReduceMean`
becomes one reduction over all the axes it reduces. The sum is a row-major fold in the element
type, and the mean rounds the exact quotient of that sum by the entry count once.

Supported operators: `Add`, `Sub`, `Mul`, `Max`, `Min` (`Max` and `Min` of any number of operands),
`Relu`, `Tanh`, `Sigmoid`, `Exp`, `Log`, `Sin`, `Cos`, `Abs`, `Sqrt`, `Reciprocal`, `MatMul`, `Gemm`
(no scaling, no transposed `A`), `Softmax`, `Reshape`, `Flatten`, `Transpose`, `Concat`,
`ReduceSum`, `ReduceMean` (not over an empty fiber), inference `BatchNormalization` and `Conv` with
explicit padding, on any number of tensor inputs and outputs. Each node's data and result have one
floating element type (binary16, bfloat16, binary32 or binary64). Integer and Boolean data and other
operator domains are refused, and integer shape and axis initializers are read as metadata. `Sqrt`
and `BatchNormalization`'s root are IEEE's square root at every operand.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

from ._artifact import _SCALAR, _matmul, _matmul_out
from ._table import op_table

# The floating ONNX element types: the name of each one's element type in the op table, and the
# unsigned integer type of its words.
ELEMENT_TYPES = {
    onnx.TensorProto.FLOAT16: ("float16", np.uint16),
    onnx.TensorProto.BFLOAT16: ("bfloat16", np.uint16),
    onnx.TensorProto.FLOAT: ("float32", np.uint32),
    onnx.TensorProto.DOUBLE: ("float64", np.uint64),
}


def _shape_size(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n

def _dim_value(dim):
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    raise RuntimeError("TorchLean ONNX import requires static tensor shapes")

def _shape_from_value_info(value_info):
    tt = value_info.type.tensor_type
    if not tt.HasField("shape"):
        raise RuntimeError(f"missing shape for value {value_info.name}")
    return [_dim_value(d) for d in tt.shape.dim]

def _collect_metadata(model):
    shapes = {}
    element_types = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.type.HasField("tensor_type"):
            shapes[vi.name] = _shape_from_value_info(vi)
            element_types[vi.name] = vi.type.tensor_type.elem_type
    for init in model.graph.initializer:
        shapes[init.name] = [int(x) for x in init.dims]
        element_types[init.name] = init.data_type
    return shapes, element_types

def _check_element_types(names, output, element_types):
    target = element_types.get(output)
    for name in [*names, output]:
        dtype = element_types.get(name)
        if dtype not in ELEMENT_TYPES:
            raise RuntimeError(f"ONNX tensor {name}: only floating-point data is supported; "
                               "integer and Boolean tensors have no lowering in this adapter")
        if dtype != target:
            raise RuntimeError(f"ONNX tensor {name}: mixed element types need explicit "
                               "operand/accumulator rules")

def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default

def _opset(model):
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return int(entry.version)
    raise RuntimeError("ONNX model does not import the default operator set")

def _norm_axis(op, axis, rank, allow_rank=False):
    axis = int(axis)
    axis = axis + rank if axis < 0 else axis
    if axis < 0 or axis > rank or (axis == rank and not allow_rank):
        raise RuntimeError(f"{op}: axis {axis} is outside rank {rank}")
    return axis

def _axis(node, rank, default):
    return _norm_axis(node.op_type, _attr(node, "axis", default), rank)

def _broadcast_shape(op, s, t):
    # The NumPy broadcast of two shapes, axes aligned from the right.
    rank = max(len(s), len(t))
    s, t = [1] * (rank - len(s)) + list(s), [1] * (rank - len(t)) + list(t)
    if any(a != b and a != 1 and b != 1 for a, b in zip(s, t)):
        raise RuntimeError(f"{op}: shapes {s} and {t} do not broadcast")
    return [b if a == 1 else a for a, b in zip(s, t)]

def _shape(shape):
    return {"shape": [int(d) for d in shape]}

def _nat(n):
    return {"nat": int(n)}

def _nats(ns):
    return {"nats": [int(n) for n in ns]}

def _bool(b):
    return {"bool": bool(b)}

def _dtype_json(element_type):
    # The JSON of the element type of the ONNX element type `element_type`.
    return copy.deepcopy(op_table()["dtypes"][ELEMENT_TYPES[element_type][0]])

def _sqrt_raw():
    # The policy field of `Op.sqrt` at `.raw`, IEEE's root (`TorchLean.Op.Wire.sqrtPolicyCode`).
    return _nat(op_table()["codes"]["sqrt"]["raw"])

class _Program:
    # The steps of a typed artifact: each step's JSON, scalar (the JSON of the element type of the
    # ONNX value it computes), words (constant steps only), output shape and source (the ONNX
    # input, initializer or node it belongs to, as `export_onnx_torchlean_graph_json` describes).
    # `element_type` and `source` describe the steps being added.

    def __init__(self):
        self.nodes, self.scalars, self.constants, self.shapes = [], [], [], []
        self.sources = []
        self.element_type, self.source = None, None

    def _step(self, node, shape, words=None):
        self.nodes.append(node)
        self.scalars.append(_dtype_json(self.element_type))
        self.constants.append(words)
        self.shapes.append([int(d) for d in shape])
        self.sources.append(self.source)
        return len(self.nodes) - 1

    def input(self, index, shape):
        return self._step({"input": {"index": index, "port": {"value": [int(d) for d in shape]}}},
                          shape)

    def constant(self, values):
        # A constant step: `values` rounded once to the element type, its words row-major.
        array = np.array(values, dtype=helper.tensor_dtype_to_np_dtype(self.element_type),
                         order="C")
        words = [int(w) for w in array.reshape(-1).view(ELEMENT_TYPES[self.element_type][1])]
        return self._step({"const": {"value": list(array.shape)}}, array.shape, words)

    def op(self, tag, fields, inputs, shape):
        # The record holds a copy of `fields`, so no field of it is `_SCALAR` or another record's.
        record = {"version": op_table()["version"], "tag": tag, "fields": copy.deepcopy(fields)}
        return self._step({"op": {"record": record, "inputs": [[int(i), 0] for i in inputs]}},
                          shape)

    def broadcast(self, source, shape, offset=None):
        # `source` broadcast to `shape`, its axes placed from `offset` (aligned from the right by
        # default), each of extent one or the target's. A source of that shape is returned as is.
        s, t = self.shapes[source], [int(d) for d in shape]
        if s == t:
            return source
        offset = len(t) - len(s) if offset is None else offset
        if offset < 0 or offset + len(s) > len(t) or \
                any(a != 1 and a != t[offset + i] for i, a in enumerate(s)):
            raise RuntimeError(f"cannot broadcast shape {s} to {t} from axis {offset}")
        dims = [offset + i for i in range(len(s))]
        return self.op("broadcast", [_shape(s), _shape(t), _nats(dims), _SCALAR], [source], t)

    def reshape(self, source, shape):
        s = self.shapes[source]
        if _shape_size(s) != _shape_size(shape):
            raise RuntimeError(f"Reshape: {s} has {_shape_size(s)} entries, {list(shape)} has "
                               f"{_shape_size(shape)}")
        return self.op("reshape", [_shape(s), _shape(shape), _SCALAR], [source], shape)

    def permute(self, source, perm):
        s = self.shapes[source]
        if sorted(perm) != list(range(len(s))):
            raise RuntimeError(f"Transpose: {perm} is not a permutation of the axes of {s}")
        out = [s[p] for p in perm]
        return self.op("permute", [_shape(s), _nats(perm), _shape(out), _SCALAR], [source], out)

    def matmul(self, a, b):
        # NumPy's matmul: an operand of rank one is a vector, the batch prefixes broadcast.
        dims = _matmul(self.shapes[a], self.shapes[b])
        left, right, lead, rows, inner, cols, lv, rv = dims
        fields = [_shape(left), _shape(right), _shape(lead), _nat(rows), _nat(inner), _nat(cols),
                  _bool(lv), _bool(rv)]
        return self.op("matmul", fields, [a, b], _matmul_out(dims))

# Operators of one operand, of two operands and of any number of operands, by ONNX type.
_UNARY = {"Relu": "relu", "Tanh": "tanh", "Sigmoid": "sigmoid", "Exp": "exp", "Log": "log",
          "Sin": "sin", "Cos": "cos", "Abs": "abs", "Sqrt": "sqrt", "Reciprocal": "reciprocal"}
_BINARY = {"Add": "add", "Sub": "sub", "Mul": "mul"}
_VARIADIC = {"Max": "max", "Min": "min"}

def _reduce_axes(node, input_names, rank, initializers):
    # The sorted reduction axes of ReduceSum or ReduceMean with ONNX's defaults: from the `axes`
    # attribute or a constant second input. None or empty reduce every axis unless
    # `noop_with_empty_axes` is set, in which case the node is the identity (None).
    axes = _attr(node, "axes", None)
    if len(input_names) > 1:
        if axes is not None:
            raise RuntimeError(f"{node.op_type}: axes given both as attribute and input")
        if input_names[1] not in initializers:
            raise RuntimeError(f"{node.op_type}: axes input must be a graph initializer")
        axes = numpy_helper.to_array(initializers[input_names[1]]).reshape(-1).tolist()
    axes = [] if axes is None else [int(a) for a in axes]
    if not axes:
        if int(_attr(node, "noop_with_empty_axes", 0)) != 0:
            return None
        axes = list(range(rank))
    axes = [_norm_axis(node.op_type, a, rank) for a in axes]
    if len(set(axes)) != len(axes):
        raise RuntimeError(f"{node.op_type}: duplicate reduction axes {axes}")
    return sorted(axes)

def _lower_conv(program, node, inputs):
    # One registered convolution: weight, bias (zeros when absent) and data as its inputs, the
    # axes preceding the channel axis as its leading axes, the groups and padding as attributes.
    if len(inputs) < 2:
        raise RuntimeError("Conv: expected data and weight inputs")
    x, weight = inputs[0], inputs[1]
    x_shape, w_shape = program.shapes[x], program.shapes[weight]
    if len(w_shape) < 3:
        raise RuntimeError(f"Conv: expected OI plus at least one spatial kernel dimension, got "
                           f"{w_shape}")
    rank = len(w_shape) - 2
    group = int(_attr(node, "group", 1))
    if group <= 0:
        raise RuntimeError(f"Conv: group must be positive, got {group}")
    dilations = [int(v) for v in _attr(node, "dilations", [1] * rank)]
    strides = [int(v) for v in _attr(node, "strides", [1] * rank)]
    pads = [int(v) for v in _attr(node, "pads", [0] * (2 * rank))]
    auto_pad = _attr(node, "auto_pad", "NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("utf-8")
    if auto_pad not in ("", "NOTSET"):
        raise RuntimeError(f"Conv: auto_pad={auto_pad!r} is unsupported; export explicit "
                           "zero-padding extents")
    if len(strides) != rank or any(v <= 0 for v in strides):
        raise RuntimeError(f"Conv: expected {rank} positive stride entries, got {strides}")
    if len(dilations) != rank or any(v <= 0 for v in dilations):
        raise RuntimeError(f"Conv: expected {rank} positive dilation entries, got {dilations}")
    if len(pads) != 2 * rank or any(v < 0 for v in pads):
        raise RuntimeError(f"Conv: expected {2 * rank} explicit pad entries, got {pads}")
    out_c, in_per_group, kernel = w_shape[0], w_shape[1], w_shape[2:]
    if out_c % group != 0:
        raise RuntimeError(f"Conv: out_channels={out_c} is not divisible by group={group}")
    if len(x_shape) < rank + 1:
        raise RuntimeError(f"Conv: expected a channel axis and {rank} spatial axes, got {x_shape}")
    lead = x_shape[:len(x_shape) - rank - 1]
    in_c, spatial = x_shape[len(lead)], x_shape[len(lead) + 1:]
    if in_c != in_per_group * group:
        raise RuntimeError(f"Conv: {in_c} input channels, the weight expects "
                           f"{in_per_group * group}")
    windows = [(spatial[j] + pads[j] + pads[rank + j] - dilations[j] * (kernel[j] - 1) - 1)
               // strides[j] + 1 for j in range(rank)]
    if len(inputs) > 2:
        bias = inputs[2]
        if program.shapes[bias] != [out_c]:
            raise RuntimeError(f"Conv: bias must have shape [{out_c}], got {program.shapes[bias]}")
    else:
        bias = program.constant(np.zeros(out_c))
    axes = list(range(rank + 2))
    fields = [_shape([_shape_size(lead), in_c] + spatial), _shape(w_shape), _nat(rank),
              _nats(axes), _nats(axes), _nats(strides), _nats(pads[:rank]), _nats(pads[rank:]),
              _nats(dilations), _nat(group), _shape(lead)]
    return program.op("conv", fields, [weight, bias, x], lead + [out_c] + windows)

def _lower_gemm(program, node, inputs, opset):
    # Gemm as an optional transpose of B, a matrix product and an optional broadcast bias add.
    if len(inputs) < 2:
        raise RuntimeError("Gemm: expected at least A and B inputs")
    if float(_attr(node, "alpha", 1.0)) != 1.0 or float(_attr(node, "beta", 1.0)) != 1.0:
        raise RuntimeError("Gemm: alpha/beta scaling is outside the TorchLean graph adapter")
    if int(_attr(node, "transA", 0)) != 0:
        raise RuntimeError("Gemm: transA is outside the TorchLean graph adapter")
    a, b = inputs[0], inputs[1]
    if int(_attr(node, "transB", 0)) != 0:
        if len(program.shapes[b]) != 2:
            raise RuntimeError(f"Gemm: transB expects rank-2 B, got {program.shapes[b]}")
        b = program.permute(b, [1, 0])
    result = program.matmul(a, b)
    if len(inputs) > 2:
        shape = program.shapes[result]
        # Below opset 7, C has the result's shape unless `broadcast` is set.
        strict = opset < 7 and not int(_attr(node, "broadcast", 0))
        if strict and program.shapes[inputs[2]] != shape:
            raise RuntimeError(f"Gemm: C has shape {program.shapes[inputs[2]]}, not {shape}, and "
                               "broadcast is 0")
        bias = program.broadcast(inputs[2], shape)
        result = program.op("add", [_shape(shape), _SCALAR], [result, bias], shape)
    return result

def _lower_batchnorm(program, node, inputs, opset):
    # Inference batch normalization in the order of ONNX's formula, per-channel tensors at axis 1.
    if len(inputs) < 5:
        raise RuntimeError("BatchNormalization: expected x, scale, bias, mean, variance")
    if int(_attr(node, "training_mode", 0)) != 0:
        raise RuntimeError("BatchNormalization: training_mode is not supported")
    if opset < 7 and not int(_attr(node, "is_test", 0)):
        raise RuntimeError("BatchNormalization: is_test=0 is not supported")
    x, scale, bias, mean, var = inputs[:5]
    x_shape = program.shapes[x]
    if len(x_shape) < 2:
        raise RuntimeError(f"BatchNormalization: expected input rank at least 2, got {x_shape}")
    channels = x_shape[1]
    for label, value in zip(("scale", "bias", "mean", "var"), (scale, bias, mean, var)):
        if program.shapes[value] != [channels]:
            raise RuntimeError(f"BatchNormalization: {label} must have shape [{channels}], got "
                               f"{program.shapes[value]}")
    # The schema's default epsilon is the binary32 attribute 1e-5f.
    epsilon = program.constant(np.full(channels, float(_attr(node, "epsilon", np.float32(1e-5)))))
    shifted = program.op("add", [_shape([channels]), _SCALAR], [var, epsilon], [channels])
    root = program.op("sqrt", [_shape([channels]), _sqrt_raw()], [shifted], [channels])
    def spread(value):
        return program.broadcast(value, x_shape, 1)
    centered = program.op("sub", [_shape(x_shape), _SCALAR], [x, spread(mean)], x_shape)
    true_division = op_table()["codes"]["div"]["none"]
    normalized = program.op("div", [_shape(x_shape), _nat(true_division), _SCALAR],
                            [centered, spread(root)], x_shape)
    scaled = program.op("mul", [_shape(x_shape), _SCALAR], [normalized, spread(scale)], x_shape)
    return program.op("add", [_shape(x_shape), _SCALAR], [scaled, spread(bias)], x_shape)

def _lower(program, node, inputs, input_names, out_shape, opset, initializers):
    # The step holding the value of one ONNX node.
    op = node.op_type
    if op == "Conv":
        return _lower_conv(program, node, inputs)
    if op == "Gemm":
        return _lower_gemm(program, node, inputs, opset)
    if op == "BatchNormalization":
        return _lower_batchnorm(program, node, inputs, opset)
    if not inputs:
        raise RuntimeError(f"{op}: expected at least one ONNX input")
    if op in _BINARY:
        if len(inputs) != 2:
            raise RuntimeError(f"{op}: TorchLean IR lowering expected two tensor parents, got "
                               f"{len(inputs)}")
        sa, sb = program.shapes[inputs[0]], program.shapes[inputs[1]]
        if opset >= 7:
            shape = _broadcast_shape(op, sa, sb)
            a, b = (program.broadcast(i, shape) for i in inputs)
        elif int(_attr(node, "broadcast", 0)):
            # Below opset 7, B is broadcast to A when it has one element or its shape is the
            # run of A's axes from `axis` (the trailing axes of A by default).
            axis = _attr(node, "axis", None)
            offset = len(sa) - len(sb) if axis is None else _norm_axis(op, axis, len(sa))
            if len(sb) > len(sa) or (_shape_size(sb) != 1 and sa[offset:offset + len(sb)] != sb):
                raise RuntimeError(f"{op}: B of shape {sb} has neither one element nor the axes "
                                   f"of A {sa} from axis {offset}")
            shape, a, b = sa, inputs[0], program.broadcast(inputs[1], sa, len(sa) - len(sb)
                                                           if _shape_size(sb) == 1 else offset)
        elif sa == sb:
            shape, (a, b) = sa, inputs
        else:
            raise RuntimeError(f"{op}: operand shapes {sa} and {sb} differ and broadcast is 0")
        tag = _BINARY[op]
        return program.op(tag, [_shape(shape), _SCALAR], [a, b], shape)
    if op in _VARIADIC:
        # A left fold over the operands, broadcast to their common shape from opset 8 on.
        shapes = [program.shapes[i] for i in inputs]
        if opset < 8 and any(s != shapes[0] for s in shapes):
            raise RuntimeError(f"{op}: operand shapes {shapes} differ; {op} broadcasts from "
                               "opset 8")
        shape = shapes[0]
        for s in shapes[1:]:
            shape = _broadcast_shape(op, shape, s)
        result = program.broadcast(inputs[0], shape)
        for i in inputs[1:]:
            result = program.op(_VARIADIC[op], [_shape(shape), _SCALAR],
                                [result, program.broadcast(i, shape)], shape)
        return result
    if op == "MatMul":
        if len(inputs) != 2:
            raise RuntimeError(f"{op}: TorchLean IR lowering expected two tensor parents, got "
                               f"{len(inputs)}")
        return program.matmul(inputs[0], inputs[1])
    x = inputs[0]
    x_shape = program.shapes[x]
    rank = len(x_shape)
    if op in _UNARY:
        tag = _UNARY[op]
        return program.op(tag, [_shape(x_shape)] + ([_sqrt_raw()] if tag == "sqrt" else []), [x],
                          x_shape)
    if op == "Softmax":
        if opset >= 13:
            return program.op("softmax", [_shape(x_shape), _nat(_axis(node, rank, -1))], [x],
                              x_shape)
        # Below opset 13 Softmax normalizes the input coerced to
        # [prod(d < axis), prod(d >= axis)], its default axis 1.
        axis = _axis(node, rank, 1)
        if axis == rank - 1:
            return program.op("softmax", [_shape(x_shape), _nat(axis)], [x], x_shape)
        flat = [_shape_size(x_shape[:axis]), _shape_size(x_shape[axis:])]
        rows = program.reshape(x, flat)
        normalized = program.op("softmax", [_shape(flat), _nat(1)], [rows], flat)
        return program.reshape(normalized, x_shape)
    if op == "Reshape":
        return program.reshape(x, out_shape)
    if op == "Flatten":
        axis = _norm_axis(op, _attr(node, "axis", 1), rank, allow_rank=True)
        return program.reshape(x, [_shape_size(x_shape[:axis]), _shape_size(x_shape[axis:])])
    if op == "Transpose":
        perm = _attr(node, "perm", None)
        perm = list(reversed(range(rank))) if perm is None else [int(p) for p in perm]
        return program.permute(x, perm)
    if op == "Concat":
        axis = _axis(node, rank, 1 if opset < 4 else 0)
        sizes = [program.shapes[i][axis] for i in inputs]
        result = x_shape[:axis] + [sum(sizes)] + x_shape[axis + 1:]
        return program.op("cat", [_shape(x_shape), _nat(axis), _nats(sizes), _SCALAR], inputs,
                          result)
    if op in ("ReduceSum", "ReduceMean"):
        axes = _reduce_axes(node, input_names, rank, initializers)
        if axes is None:
            return x
        keepdim = bool(int(_attr(node, "keepdims", 1)))
        result = [1 if a in axes else d for a, d in enumerate(x_shape) if keepdim or a not in axes]
        if op == "ReduceMean" and _shape_size([x_shape[a] for a in axes]) == 0 and \
                _shape_size(result) != 0:
            raise RuntimeError("ReduceMean: a mean over an empty fiber is refused: ONNX leaves "
                               "it undefined, the IR mean divides the empty sum by one")
        if op == "ReduceSum":
            reduce_sum = op_table()["codes"]["reduce"]["sum"]
            return program.op("reduce", [_shape(x_shape), _nat(reduce_sum), _nats(axes),
                                         _bool(keepdim)], [x], result)
        return program.op("mean", [_shape(x_shape), _nats(axes), _bool(keepdim)], [x], result)
    raise RuntimeError(f"Unsupported ONNX op for TorchLean IR import: {op}")


def export_onnx_torchlean_graph_json(onnx_path, out_json_path, sources=None):
    """Write the artifact of the ONNX model at `onnx_path` to `out_json_path` and return it.

    A list `sources` receives the source of each step: `("input", None, name)`,
    `("initializer", None, name)`, or the ONNX node's op type, node name and output name.
    """
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model)
    shapes, element_types = _collect_metadata(model)
    opset = _opset(model)
    initializers = {init.name: init for init in model.graph.initializer}
    program = _Program()
    steps = {}
    for index, value in enumerate(x for x in model.graph.input if x.name not in initializers):
        if value.name not in shapes:
            raise RuntimeError(f"ONNX input {value.name}: only tensor inputs are supported")
        _check_element_types([], value.name, element_types)
        program.element_type = element_types[value.name]
        program.source = ("input", None, value.name)
        steps[value.name] = program.input(index, shapes[value.name])
    def step(name):
        # The step of an ONNX value. An initializer becomes a constant step when it is first read.
        if name not in steps:
            saved = (program.element_type, program.source)
            program.element_type = element_types[name]
            program.source = ("initializer", None, name)
            steps[name] = program.constant(numpy_helper.to_array(initializers[name]))
            program.element_type, program.source = saved
        return steps[name]
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx"):
            raise RuntimeError(f"ONNX domain {node.domain!r}: operator {node.op_type} needs its "
                               "own registered semantics")
        if len(node.output) != 1:
            raise RuntimeError(f"ONNX node {node.name or node.op_type} has multiple outputs; "
                               "tuple lowering is not implemented")
        out_name = node.output[0]
        if out_name not in shapes:
            raise RuntimeError(f"missing inferred shape for ONNX value {out_name}")
        input_names = [x for x in node.input if x]
        for x in input_names:
            if x not in steps and x not in initializers:
                raise RuntimeError(f"ONNX input {x} for node {node.name or node.op_type} is not "
                                   "produced by an earlier node")
        # Reshape's shape and the reductions' axes are metadata, not data.
        data = input_names[:1] if node.op_type in ("Reshape", "ReduceSum", "ReduceMean") else \
            input_names
        _check_element_types(data, out_name, element_types)
        program.element_type = element_types[out_name]
        program.source = (node.op_type, node.name, out_name)
        steps[out_name] = _lower(program, node, [step(x) for x in data], input_names,
                                 shapes[out_name], opset, initializers)
        if program.shapes[steps[out_name]] != shapes[out_name]:
            raise RuntimeError(f"{node.op_type}: inferred shape {shapes[out_name]} differs from "
                               f"{program.shapes[steps[out_name]]}")
    outputs = []
    for value in model.graph.output:
        _check_element_types([], value.name, element_types)
        if value.name not in steps and value.name not in initializers:
            raise RuntimeError(f"ONNX graph output {value.name} is not produced")
        outputs.append([step(value.name), 0])
    if sources is not None:
        sources.extend(program.sources)
    artifact = {"format": op_table()["format"], "nodes": program.nodes, "outputs": outputs,
                "scalars": program.scalars, "constants": program.constants}
    Path(out_json_path).write_text(json.dumps(artifact, allow_nan=False))
    return artifact


def main():
    ap = argparse.ArgumentParser(description="Write a conservative ONNX fragment as a "
                                             "TorchLean IR artifact")
    ap.add_argument("onnx_path")
    ap.add_argument("out_json_path")
    ap.add_argument("--sources", help="write the ONNX source of each step here")
    args = ap.parse_args()
    sources = []
    export_onnx_torchlean_graph_json(args.onnx_path, args.out_json_path, sources)
    if args.sources:
        Path(args.sources).write_text(json.dumps(sources))

if __name__ == "__main__":
    main()
