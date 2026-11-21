"""Capture a PyTorch model's graph: torch.export's graph when it lowers, else FX's symbolic trace.

The capture's state is module-level (`_fx._TENSOR_VALUES`, the caller's checker `_fx._CHECKER`
and the tables of `_structure`). `_capture_state` locks those tables during lowering and
releases their tensors as soon as the capture finishes.
"""

from __future__ import annotations

from contextlib import contextmanager
import threading

import torch

from . import _hop
from ._bind import bind_arguments
from ._emit import _UnsupportedOperator, _lower_kind
from ._fx import (_CHECKER, _CONVERSIONS, _TENSOR_VALUES, _is_getitem, _is_op, _node_dtype,
                  _target_name, _top_k_mask, _tuple_shapes_of_node, _zeros_like)
from ._pointwise import _reflected_division
from ._structure import (_DROPOUT_MASKS, _FOLDED, _UninitializedRead, _complex_dtype, _complex_reader,
                         _complex_value, _component_producer, _project_results, _projection,
                         _read_signature)


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


def _composite(node):
    # Whether eager PyTorch runs `node` as a composite of other ATen operators on the CPU: an
    # operator with a CompositeImplicitAutograd kernel and no CPU or CompositeExplicitAutograd
    # kernel, as the CPU then runs that kernel (rules 1 to 2.3 of resolve_key in PyTorch 2.13's
    # torch/_ops.py, the dispatcher's computeDispatchTableEntry), or torch.functional.einsum
    # under FX. torch.einsum is one (aten/src/ATen/native/Linear.cpp): it contracts in pairs with
    # bmm in the order of its `path`, opt_einsum's from three operands on when opt_einsum is
    # available, else left to right (the PyTorch 2.13 docstring of torch.einsum). Only a
    # composite of a tensor result is read: `item` has one too, of a number, and stays refused.
    # FX can keep these composites under their public Python names.
    if _is_op(node, (), (torch.functional.einsum, torch.take_along_dim, torch.square),
              ('take_along_dim', 'square')):
        return True
    if node.op != 'call_function':
        return False
    if not isinstance(node.target, torch._ops.OpOverload):
        return False
    value = node.meta.get('val', node.meta.get('tensor_meta'))
    if not (hasattr(value, 'shape') and hasattr(value, 'dtype')):
        return False
    has = lambda key: torch._C._dispatch_has_kernel_for_dispatch_key(node.target.name(), key)
    return has('CompositeImplicitAutograd') and not any(
        has(key) for key in ('CPU', 'CompositeExplicitAutograd',
                             'CompositeExplicitAutogradNonFunctional'))


def _inline_composite(graph, node):
    # Replace the composite `node` (_composite) by the ATen operators eager PyTorch runs for it,
    # traced at fake CPU tensors of its operands' shapes and dtypes with its other arguments, which
    # the capture then reads one by one. A composite that reads a tensor constant or returns other
    # than one tensor is refused.
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.utils import _pytree

    name = _target_name(node.target)
    target = getattr(torch.Tensor, node.target) if node.op == 'call_method' else node.target
    leaves, spec = _pytree.tree_flatten((node.args, node.kwargs))
    positions = [index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.fx.Node)]
    values = [leaves[index].meta.get('val', leaves[index].meta.get('tensor_meta'))
              for index in positions]
    if not all(hasattr(value, 'shape') and hasattr(value, 'dtype') for value in values):
        raise NotImplementedError(f'{name}: a composite needs tensor operands')

    def composite(*tensors):
        filled = list(leaves)
        for index, tensor in zip(positions, tensors):
            filled[index] = tensor
        args, kwargs = _pytree.tree_unflatten(filled, spec)
        return target(*args, **kwargs)

    with FakeTensorMode():
        operands = [torch.empty([int(n) for n in value.shape], dtype=value.dtype)
                    for value in values]
    traced = make_fx(composite)(*operands).graph
    steps = list(traced.nodes)
    if any(step.op == 'get_attr' for step in steps) or \
            not isinstance(steps[-1].args[0], torch.fx.Node):
        raise NotImplementedError(f'{name}: its composite reads a tensor constant or returns '
                                  'other than one tensor')
    placeholders = [step for step in steps if step.op == 'placeholder']
    with graph.inserting_before(node):
        result = graph.graph_copy(traced, dict(zip(placeholders,
                                                   (leaves[index] for index in positions))))
    node.replace_all_uses_with(result)
    graph.erase_node(node)


def _check_graph_operators(graph, model):
    # The one binding of the graph's keywords (`torch.cumsum(x, axis=0)` is `dim=0`): every
    # capture runs this check before it lowers, and the lowerings read canonical keywords only.
    # A composite operator without a lowering of its own becomes the operators of its composite,
    # and the check runs again on the graph that holds them. The graphs of a control-flow
    # operator that may run (`_hop.selected`) are checked in turn, each with its operands.
    mode_refusal = _mode_refusal(graph)
    if mode_refusal is not None:
        raise NotImplementedError(mode_refusal)
    bind_arguments(graph)
    helpers = _routing_helpers(graph)
    composites = []
    for node in graph.nodes:
        _check_float8(node)
        if node.op in ("placeholder", "get_attr") and _complex_value(node):
            raise NotImplementedError(f'unsupported PyTorch element type: '
                                      f'torch.{_complex_dtype(node)} of an input or constant')
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        if _hop.hop(node):
            for module, operands in _hop.selected(graph, node):
                with _hop.bound(module, operands):
                    _check_graph_operators(module.graph, model)
            continue
        if _is_getitem(node) and isinstance(node.args[0], torch.fx.Node) and \
                _hop.hop(node.args[0]):
            continue
        if node in helpers or _zeros_like(node): continue
        if _metadata_constraint(node) is not None: continue
        if _complex_value(node):
            for user in node.users:
                if not _complex_reader(user, node):
                    reader = 'an output' if user.op == 'output' else _target_name(user.target)
                    raise NotImplementedError(
                        f'{reader} of a complex value: a complex value is held in paired-real '
                        'form and read only by the complex operators')
        if _component_producer(node):
            if any(_projection(user) is None for user in node.users):
                raise NotImplementedError(f'{_target_name(node.target)} is used other than through its components')
            continue
        if _projection(node) is not None and not node.users: continue
        if _tuple_shapes_of_node(node):
            raise NotImplementedError(f"unsupported tuple-valued PyTorch op: {node.target}")
        elif node.op in ("call_function", "call_method", "call_module"):
            try:
                _lower_kind(node, model)
            except _UnsupportedOperator:
                if not _composite(node):
                    raise
                composites.append(node)
        else:
            raise NotImplementedError(f"unsupported FX node op: {node.op}")
    for node in composites:
        _inline_composite(graph, node)
    if composites:
        _check_graph_operators(graph, model)


# The operators that move the entries of a float8 value without reading them (`_check_float8`).
_FLOAT8_MOVEMENT = (('aten.view.default', 'aten.reshape.default', 'aten._unsafe_view.default',
                     'aten.flatten.using_ints', 'aten.contiguous.default', 'aten.permute.default',
                     'aten.transpose.int', 'aten.t.default', 'aten.unsqueeze.default',
                     'aten.squeeze.dim', 'aten.squeeze.dims', 'aten.squeeze.default',
                     'aten.expand.default', 'aten.slice.Tensor', 'aten.clone.default'),
                    (torch.reshape, torch.flatten, torch.permute, torch.transpose, torch.t,
                     torch.unsqueeze, torch.squeeze, torch.clone),
                    ('view', 'reshape', 'flatten', 'contiguous', 'permute', 'transpose', 't',
                     'unsqueeze', 'squeeze', 'expand', 'clone'))


def _check_float8(node):
    # A float8 value (`_artifact._FLOAT8`) is an input, a constant, a conversion's result or a
    # movement of a float8 value (_FLOAT8_MOVEMENT), and only these, the outputs and metadata
    # assertions read it. The capture lowers no other operator at the four formats, and every
    # word of each converts exactly to binary16, bfloat16 and wider.
    if node.op == 'output':
        return
    try:
        dtype = _node_dtype(node)
    except NotImplementedError:
        return
    if not (isinstance(dtype, dict) and dtype['kind'] == 'binary'):
        return
    moves = lambda step: _is_op(step, *_CONVERSIONS) or _is_op(step, *_FLOAT8_MOVEMENT)
    if node.op not in ('placeholder', 'get_attr') and not moves(node):
        raise NotImplementedError(f'{_target_name(node.target)} of a float8 result: a float8 '
                                  'value is made only by a conversion or a movement')
    for user in node.users:
        if user.op != 'output' and not moves(user) and _metadata_constraint(user) is None:
            raise NotImplementedError(f'{_target_name(user.target)} of a float8 value: a float8 '
                                      'value is read only by conversions and movements')


_GRAD_MODE_REFUSAL = ('torch.export holds a grad-mode region (no_grad or enable_grad), which the '
                      'IR does not import and a symbolic trace would drop')
_AUTOCAST_REFUSAL = ('torch.export holds an autocast region, which the IR does not import and '
                     'a symbolic trace would drop, changing its precision choices')


def _from_grad_mode(node):
    # Whether the provenance of `node` (from_node: the nodes that PyTorch's passes replaced by it,
    # and theirs) holds a grad-mode block.
    sources = list(node.meta.get('from_node') or ())
    while sources:
        source = sources.pop()
        if getattr(source, 'target', None) == 'wrap_with_set_grad_enabled':
            return True
        sources.extend(getattr(source, 'from_node', None) or ())
    return False


def _mode_refusal(graph):
    # Refuse execution modes that an FX fallback would erase, including those inside an invoked
    # subgraph or a selected control-flow branch. Region bindings keep known predicates known.
    modes = (((torch._C._set_grad_enabled,
               getattr(torch.ops.higher_order, 'wrap_with_set_grad_enabled', None)),
              _GRAD_MODE_REFUSAL),
             ((torch.amp.autocast_mode._enter_autocast,
               getattr(torch.ops.higher_order, 'wrap_with_autocast', None)), _AUTOCAST_REFUSAL))
    for node in graph.nodes:
        if node.op == 'call_function':
            for marks, reason in modes:
                if any(node.target is mark for mark in marks):
                    return reason
            if _from_grad_mode(node):
                return _GRAD_MODE_REFUSAL
        if _hop.hop(node):
            for module, operands in _hop.selected(graph, node):
                with _hop.bound(module, operands):
                    reason = _mode_refusal(module.graph)
                    if reason is not None:
                        return reason
    return None


def _export_inputs(program):
    """Separate an exported program's tensor inputs from its parameters, buffers and constants."""
    from torch.export.graph_signature import ConstantArgument, TensorArgument

    names, lifted = [], {}
    for spec in program.graph_signature.input_specs:
        if spec.kind.name == 'USER_INPUT':
            if isinstance(spec.arg, TensorArgument):
                names.append(spec.arg.name)
            elif not isinstance(spec.arg, ConstantArgument):
                raise NotImplementedError("exported non-tensor dynamic inputs are unsupported")
        else:
            value = program.state_dict.get(spec.target, program.constants.get(spec.target))
            if isinstance(value, torch.Tensor):
                lifted[spec.arg.name] = value
    return names, lifted


# One capture at a time, since the capture's state is module-level.
_LOCK = threading.RLock()

# The dialect of the captured torch.export program, None for a symbolic trace. torch.export gives
# the TRAINING dialect, which keeps detach_, detach and grad-mode regions. A decomposed program
# (`run_decompositions()`, the ATEN dialect) drops detach_ with no trace, writes detach as alias
# (_structure._decomposed_detach) and inlines grad-mode regions (_from_grad_mode).
_DIALECT = [None]


def _reset(signature=None, checker=None):
    # Empty the capture's module state (the lifted tensors, the folded index constants and the
    # dropout masks), read the buffers of the exported signature `signature` (None for FX) and
    # record `checker`. The capture's records follow its op table.
    _CHECKER[0] = checker
    _DIALECT[0] = None
    _TENSOR_VALUES.clear()
    _FOLDED.clear()
    _DROPOUT_MASKS.clear()
    _read_signature(signature)


@contextmanager
def _capture_state(signature=None, checker=None):
    """Scope lifted tensors and folded nodes to one lowering, including failed captures."""
    with _LOCK:
        try:
            _reset(signature, checker)
            yield
        finally:
            _reset()


def _capture(model, example, require_torch_export=False, prefer_torch_export=True):
    # The graph of `model` at `example` and its user input names: torch.export's graph when
    # `prefer_torch_export` holds and the graph lowers, else FX's symbolic trace, which
    # `require_torch_export` refuses, as does a graph with an unsupported execution mode. The
    # caller holds _LOCK until it has read the state.
    if require_torch_export and not prefer_torch_export:
        raise ValueError('require_torch_export needs prefer_torch_export')
    examples = (example,) if isinstance(example, torch.Tensor) else example
    if not isinstance(examples, tuple) or not examples or not all(
        isinstance(value, torch.Tensor) for value in examples):
        raise NotImplementedError('capture requires positional tensor arguments')
    if prefer_torch_export:
        mode_refusal = None
        try:
            ep = torch.export.export(model, examples)
            _read_signature(ep.graph_signature)
            names, lifted = _export_inputs(ep)
            _TENSOR_VALUES.update(lifted)
            mode_refusal = _mode_refusal(ep.graph)
            if mode_refusal is not None:
                raise NotImplementedError(mode_refusal)
            _check_graph_operators(ep.graph, model)
            _DIALECT[0] = getattr(ep, 'dialect', None)
            return ep.graph, names
        except _UninitializedRead:
            raise
        except Exception:
            if require_torch_export or mode_refusal is not None:
                raise
    _reset()
    from torch.fx import symbolic_trace
    from torch.fx.passes.shape_prop import ShapeProp
    gm = symbolic_trace(model)
    _project_results(gm.graph)
    gm.recompile()
    class CaptureShapeProp(ShapeProp):
        def run_node(self, node):
            value = super().run_node(node)
            if isinstance(value, torch.Tensor):
                node.meta['torchlean_device'] = value.device
            return value
    CaptureShapeProp(gm).propagate(*examples)
    _TENSOR_VALUES.update(_hop.tensor_attributes(gm.graph, gm))
    _check_graph_operators(gm.graph, model)
    return gm.graph, [node.name for node in gm.graph.nodes if node.op == 'placeholder']
