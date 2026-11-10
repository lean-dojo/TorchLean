"""FX node readers of the PyTorch capture: dtypes, shapes, operator matching, payloads and axes.

`_TENSOR_VALUES` maps the name of a placeholder or `get_attr` node to the tensor the capture lifted
for it (a parameter, buffer or constant), and `_graph._reset` empties it at each capture. `_reset`
also records the caller's checker in `_CHECKER`, and `_op_table` reads Lean's op table from it.
The main declarations are `_dtype_of`, `_node_shape`, `_is_op`, `_node_refs`, `_tensor_value`, the
payloads of linear, convolution and normalization nodes, and `_normalize_axis`.
"""

from __future__ import annotations

import operator

import torch

from ._table import op_table
from .state import _words



def _dtype_of(value):
    dtype = getattr(value, 'dtype', None)
    name = str(dtype).removeprefix('torch.')
    if name in ('float16', 'bfloat16', 'float32', 'float64', 'bool'): return name
    if name.startswith(('int', 'uint')) and name.lstrip('uint').isdigit():
        return {'kind': 'integer', 'signed': not name.startswith('uint'),
                'width': int(name.lstrip('uint'))}
    raise NotImplementedError(f'unsupported PyTorch element type: {dtype}')


def _node_dtype(node):
    return _dtype_of(node.meta.get('val', node.meta.get('tensor_meta')))


def _shape_of(value):
    if hasattr(value, "shape"):
        return [int(x) for x in tuple(value.shape)]
    if isinstance(value, (tuple, list)) and value and hasattr(value[0], "shape"):
        return [int(x) for x in tuple(value[0].shape)]
    return []


def _node_shape(node):
    return _shape_of(node.meta.get("val", node.meta.get("tensor_meta")))


def _shape_from_arg(arg):
    return _node_shape(arg) if hasattr(arg, "meta") else []


def _tuple_shapes_of_node(node):
    val = node.meta.get("val", None)
    if isinstance(val, (tuple, list)) and not hasattr(val, "shape"):
        return [_shape_of(x) for x in val]
    tm = node.meta.get("tensor_meta", None)
    if hasattr(tm, "shape"):
        return []
    if isinstance(tm, (tuple, list)):
        return [_shape_of(x) for x in tm]
    return []


def _is_getitem(node):
    return node.op == "call_function" and node.target is operator.getitem


def _is_getattr(node):
    return node.op == "call_function" and node.target is getattr


def _target_name(target):
    return str(target).replace("torch.ops.", "")


def _is_op(node, aten=(), functions=(), methods=()):
    # ATen overloads have distinct contracts: exp.out writes to an output tensor.
    # FX keeps Python callables or method names, which match by identity or by name.
    if node.op == "call_function":
        if isinstance(node.target, torch._ops.OpOverload):
            return str(node.target) in aten
        return any(node.target is function for function in functions)
    return node.op == "call_method" and node.target in methods


def _node_refs(obj, node_to_id, strict=False):
    refs = []
    def visit(x):
        if isinstance(x, torch.fx.Node):
            ref = node_to_id.get(x, None)
            if ref is not None:
                refs.append(ref)
            elif strict:
                raise NotImplementedError('graph result has no tensor lowering')
        elif isinstance(x, (tuple, list)):
            for y in x: visit(y)
        elif isinstance(x, dict):
            for y in x.values(): visit(y)
        elif strict:
            raise NotImplementedError('graph return leaves must be tensors')
    visit(obj)
    return refs


def _spatial_tuple(x, rank, default):
    if x is None: return [int(default)] * rank
    if isinstance(x, int): return [int(x)] * rank
    if isinstance(x, (tuple, list)) and len(x) == rank: return [int(v) for v in x]
    raise NotImplementedError(f"expected an integer or length-{rank} spatial tuple, got {x!r}")


def _top_k_mask(node):
    if not _is_op(node, ('aten.scatter.value',)): return None
    base, dim, index, value = (list(node.args) + [None] * 4)[:4]
    if value is None: value = node.kwargs.get('value')
    if not isinstance(value, (int, float)) or float(value) != 1.0: return None
    if not _is_getitem(index) or index.args[1] != 1: return None
    topk = index.args[0]
    if not _is_op(topk, ('aten.topk.default',)): return None
    source, k = topk.args[0], topk.args[1]
    rank = len(_shape_from_arg(source))
    topk_dim = topk.args[2] if len(topk.args) > 2 else topk.kwargs.get('dim', -1)
    largest = topk.args[3] if len(topk.args) > 3 else topk.kwargs.get('largest', True)
    if not largest or _normalize_axis(topk_dim, rank) != rank - 1 or _normalize_axis(dim, rank) != rank - 1: return None
    if not _zeros_like(base) or _shape_from_arg(base) != _shape_from_arg(source): return None
    if _node_dtype(base) != _node_dtype(source) or _node_dtype(node) != _node_dtype(source): return None
    return source, int(k)


def _tensor_bits(value):
    name = str(value.dtype).removeprefix('torch.')
    if not (name in ('bool', 'float16', 'bfloat16', 'float32', 'float64')
            or name.startswith(('int', 'uint'))):
        raise NotImplementedError(f'no binary word encoding for {name}')
    return _words(value)


def _binary_payload(name, value):
    return {name + '_bits': _tensor_bits(value), name + '_dtype': _dtype_of(value)}


def _tensor_value(value):
    # A tensor as `IR.loadValue` reads it: its dtype, shape and row-major words.
    return {'dtype': _dtype_of(value), 'shape': list(value.shape), 'bits': _tensor_bits(value)}


def _constant(value):
    return {'kind': 'const', **_tensor_value(value)}


_TENSOR_VALUES = {}

# The caller's checker for the running capture, which `_graph._reset` records. The capture's
# records follow its op table.
_CHECKER = [None]


def _op_table():
    return op_table(_CHECKER[0])


def _resolve_tensor(value):
    if isinstance(value, torch.Tensor): return value
    if isinstance(value, torch.fx.Node) and value.name in _TENSOR_VALUES: return _TENSOR_VALUES[value.name]
    raise NotImplementedError(f"could not resolve tensor argument {value!r} into the exported payload")


def _linear_payload(weight, bias):
    weight = _resolve_tensor(weight)
    if weight.ndim != 2:
        raise NotImplementedError(f"linear weight must have rank 2, got shape {tuple(weight.shape)}")
    out_dim, in_dim = (int(value) for value in weight.shape)
    bias = _resolve_tensor(bias) if bias is not None else weight.new_zeros(out_dim)
    if tuple(bias.shape) != (out_dim,):
        raise NotImplementedError(f"linear bias must have shape {(out_dim,)}, got {tuple(bias.shape)}")
    return {"out_dim": out_dim, "in_dim": in_dim, **_binary_payload('weight', weight), **_binary_payload('bias', bias)}


def _convolution_payload(weight, bias, groups):
    weight = _resolve_tensor(weight)
    groups = int(groups)
    if weight.ndim < 3:
        raise NotImplementedError(f"convolution weight must have rank at least 3, got shape {tuple(weight.shape)}")
    if groups <= 0:
        raise NotImplementedError(f"convolution groups must be positive, got {groups}")
    out_channels = int(weight.shape[0])
    in_channels_per_group = int(weight.shape[1])
    if out_channels % groups != 0:
        raise NotImplementedError(f"convolution out_channels={out_channels} is not divisible by groups={groups}")
    in_channels = in_channels_per_group * groups
    if groups == 1:
        dense_weight = weight
    else:
        out_channels_per_group = out_channels // groups
        dense_weight = weight.new_zeros((out_channels, in_channels, *weight.shape[2:]))
        for group in range(groups):
            out_start = group * out_channels_per_group
            out_end = out_start + out_channels_per_group
            in_start = group * in_channels_per_group
            in_end = in_start + in_channels_per_group
            dense_weight[out_start:out_end, in_start:in_end] = weight[out_start:out_end]
    bias = _resolve_tensor(bias) if bias is not None else weight.new_zeros(out_channels)
    if tuple(bias.shape) != (out_channels,):
        raise NotImplementedError(f"convolution bias must have shape {(out_channels,)}, got {tuple(bias.shape)}")
    return {"in_channels": in_channels, "out_channels": out_channels, **_binary_payload('weight', dense_weight), **_binary_payload('bias', bias)}


def _normalization(kind, source, eps, attributes, shape, **parameters):
    # A layer_norm or batch_norm node, emitted with its parameters by _emit_structure.
    # An absent weight or bias is one or zero of the computing dtype over `shape`.
    eps = float(eps)
    if not eps >= 0:
        raise NotImplementedError('a negative or NaN eps: PyTorch takes the square root of var + eps, which the IR clamps at zero')
    parameters = {name: None if value is None else _resolve_tensor(value) for name, value in parameters.items()}
    var = parameters.get('var')
    if var is not None and bool((var < 0).any()):
        raise NotImplementedError('a negative running variance: PyTorch takes the square root of var + eps, which the IR clamps at zero')
    return {'kind': kind, '_plan': {'op': 'normalization', 'operands': [source],
            'attributes': {'kind': kind, **attributes}, 'eps': eps,
            'shape': [int(size) for size in shape], 'parameters': parameters}}


def _reshape_kind():
    # The node's declared shape is the target of a reshape.
    return {"kind": "reshape"}


def _broadcast_kind(source_shape, target_shape):
    # PyTorch broadcasting aligns the source with the trailing target axes.
    start = len(target_shape) - len(source_shape)
    return {"kind": "broadcast", "dims": list(range(start, len(target_shape)))}


def _normalize_axis(axis, rank):
    axis = int(axis)
    axis = axis + rank if axis < 0 else axis
    if axis < 0 or axis >= rank:
        raise NotImplementedError(f"axis {axis} is outside rank {rank}")
    return axis


def _reduction_axes(axis, rank, op_name):
    # PyTorch reduces every axis for dim=None and for an empty dim list.
    raw = tuple(axis) if isinstance(axis, (tuple, list)) else () if axis is None else (axis,)
    raw = raw or range(rank)
    axes = [_normalize_axis(value, rank) for value in raw]
    if len(set(axes)) != len(axes):
        raise NotImplementedError(f"{op_name} has duplicate reduction axes {axes}")
    return axes


_REDUCED_FLOATS = ('bfloat16', 'float16')


def _torch_dtype(dtype):
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return getattr(torch, ('int' if dtype['signed'] else 'uint') + str(dtype['width']))


_FLOATING = ('float64', 'float32', 'bfloat16', 'float16')
_INT64 = {'kind': 'integer', 'signed': True, 'width': 64}
_INDEX_DTYPES = (_INT64, {'kind': 'integer', 'signed': True, 'width': 32})


def _zeros_like(node):
    return _is_op(node, ('aten.zeros_like.default',)) and not node.kwargs.get('dtype')
