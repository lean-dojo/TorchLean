"""Capture a PyTorch model's graph: torch.export's graph when it lowers, else FX's symbolic trace.

The capture's state is module-level (`_fx._TENSOR_VALUES`, the caller's checker `_fx._CHECKER`
and the tables of `_structure`), so a capture runs under `_LOCK` and `_reset` sets the state
first. The main declarations are `_capture`, `_check_graph_operators` and `_reset`.
"""

from __future__ import annotations

import threading

import torch

from ._emit import _lower_kind
from ._fx import (_CHECKER, _TENSOR_VALUES, _is_getitem, _is_op, _target_name, _top_k_mask,
                  _tuple_shapes_of_node, _zeros_like)
from ._pointwise import _reflected_division
from ._structure import (_DROPOUT_MASKS, _FOLDED, _complex_consumer, _complex_producer,
                         _component_producer, _projection, _read_signature)


def _routing_helpers(graph):
    masks = [node for node in graph.nodes if _top_k_mask(node) is not None]
    helpers = set()
    for mask in masks:
        base, index = mask.args[0], mask.args[2]
        topk = index.args[0]
        helpers |= {base, index, topk} | {user for user in topk.users if _is_getitem(user) and not user.users}
    allowed = set(masks) | helpers
    for helper in helpers:
        if any(user not in allowed for user in helper.users):
            raise NotImplementedError('torch.topk values, or zeros shared with other operations, are outside the TorchLean routing idiom')
    # The reciprocal of torch.export's s / x belongs to the division its product lowers.
    helpers |= {node.args[0] for node in graph.nodes if _reflected_division(node) is not None}
    return helpers


def _metadata_constraint(node):
    if not _is_op(node, ('aten._assert_tensor_metadata.default',)): return None
    # The core form passes size, stride and dtype positionally, size and stride None.
    extra = node.args[1:]
    if not node.args or len(extra) > 3 or any(item is not None for item in extra[:2]) or set(node.kwargs) - {'dtype', 'device', 'layout'}:
        raise NotImplementedError('unsupported tensor metadata assertion')
    kwargs = dict(node.kwargs)
    if len(extra) > 2 and extra[2] is not None: kwargs['dtype'] = extra[2]
    dtype = kwargs.get('dtype')
    value = node.args[0].meta.get('val', node.args[0].meta.get('tensor_meta'))
    if dtype is not None and dtype != getattr(value, 'dtype', None):
        raise NotImplementedError('metadata assertion conflicts with the captured dtype')
    return {'value': node.args[0].name,
            **{key: str(value) for key, value in kwargs.items()}}


def _check_graph_operators(graph, model):
    helpers = _routing_helpers(graph)
    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        if node in helpers or _zeros_like(node): continue
        if _metadata_constraint(node) is not None: continue
        if _complex_producer(node):
            if any(not _complex_consumer(user, node) for user in node.users):
                raise NotImplementedError(f'{_target_name(node.target)}: a complex value is read only in paired-real form')
            continue
        if _component_producer(node):
            if any(_projection(user) is None for user in node.users):
                raise NotImplementedError(f'{_target_name(node.target)} is used other than through its components')
            continue
        if _projection(node) is not None and not node.users: continue
        if _tuple_shapes_of_node(node):
            raise NotImplementedError(f"unsupported tuple-valued PyTorch op: {node.target}")
        elif node.op in ("call_function", "call_method", "call_module"):
            _lower_kind(node, model)
        else:
            raise NotImplementedError(f"unsupported FX node op: {node.op}")


def _graph_is_lowerable(graph, model):
    try:
        _check_graph_operators(graph, model)
    except NotImplementedError:
        return False
    return True


# One capture at a time, since the capture's state is module-level.
_LOCK = threading.RLock()


def _reset(signature=None, checker=None):
    # Empty the capture's module state (the lifted tensors, the folded index constants and the
    # dropout masks), read the buffers of the exported signature `signature` (None for FX) and
    # record `checker`. The capture's records follow its op table.
    _CHECKER[0] = checker
    _TENSOR_VALUES.clear()
    _FOLDED.clear()
    _DROPOUT_MASKS.clear()
    _read_signature(signature)


def _capture(model, example, require_torch_export=False, prefer_torch_export=True):
    # The graph of `model` at `example` and its user input names: torch.export's graph when
    # `prefer_torch_export` holds and the graph lowers, else FX's symbolic trace, which
    # `require_torch_export` refuses. The caller holds _LOCK until it has read the state.
    if require_torch_export and not prefer_torch_export:
        raise ValueError('require_torch_export needs prefer_torch_export')
    _reset()
    examples = (example,) if isinstance(example, torch.Tensor) else example
    if not isinstance(examples, tuple) or not examples or not all(
        isinstance(value, torch.Tensor) for value in examples):
        raise NotImplementedError('capture requires positional tensor arguments')
    if prefer_torch_export:
        try:
            ep = torch.export.export(model, examples)
            # Functionalization turns a buffer update into a graph output, which the
            # program returns after its own outputs. Other mutations are refused.
            _read_signature(ep.graph_signature)
            # ExportedProgram lifts parameters, buffers, and constants into graph
            # placeholders. Their order is not a user-input contract, so classify
            # placeholders through graph_signature rather than taking the first one.
            user_input_names = [
                spec.arg.name
                for spec in ep.graph_signature.input_specs
                if getattr(spec.kind, "name", "") == "USER_INPUT"
            ]
            state = dict(ep.state_dict)
            constants = dict(getattr(ep, 'constants', {}))
            for spec in ep.graph_signature.input_specs:
                if getattr(spec.kind, 'name', '') != 'USER_INPUT':
                    target = getattr(spec, 'target', None)
                    value = state.get(target, constants.get(target, None))
                    if isinstance(value, torch.Tensor): _TENSOR_VALUES[spec.arg.name] = value
            if _graph_is_lowerable(ep.graph, model):
                return ep.graph, user_input_names
            if require_torch_export: _check_graph_operators(ep.graph, model)
        except Exception:
            if require_torch_export:
                raise
    if require_torch_export:
        raise NotImplementedError("torch.export produced operators outside the TorchLean import "
                                  "subset")
    _reset()
    from torch.fx import symbolic_trace
    from torch.fx.passes.shape_prop import ShapeProp
    gm = symbolic_trace(model)
    ShapeProp(gm).propagate(*examples)
    for node in gm.graph.nodes:
        if node.op == 'get_attr':
            value = gm
            for part in str(node.target).split('.'): value = getattr(value, part)
            if isinstance(value, torch.Tensor): _TENSOR_VALUES[node.name] = value
    _check_graph_operators(gm.graph, model)
    return gm.graph, [node.name for node in gm.graph.nodes if node.op == 'placeholder']
