"""Structural operators of a PyTorch graph: indexing, scatter, statistics, RNG and buffers.

`_structure_kind` plans one call, and the components of tuple-valued operators are read through
`_projection`. Beside `_fx._TENSOR_VALUES`, the capture's state lives here: the folded index
constants (`_FOLDED`), the exported signature's buffers (`_BUFFERS`, `_OUTPUT_TARGETS`,
`_BUFFER_UPDATES`) and the dropout masks (`_DROPOUT_MASKS`), all emptied by `_graph._reset`.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

from ._artifact import _splitmix64
from ._fx import (_FLOATING, _INDEX_DTYPES, _INT64, _REDUCED_FLOATS, _TENSOR_VALUES,
                  _broadcast_kind, _constant, _convolution_payload, _is_getitem, _is_op,
                  _node_dtype, _node_shape, _normalization, _normalize_axis, _op_table,
                  _reshape_kind, _resolve_tensor, _shape_from_arg, _shape_of, _spatial_tuple,
                  _target_name, _top_k_mask, _torch_dtype)
from ._pointwise import (_append_node, _argument, _broadcast_to, _cast_to, _round_once,
                         _scalar_constant, _scalar_tensor)


_COMPONENTS = {
    'aten.native_layer_norm.default': (0,),
    'aten._native_batch_norm_legit_no_training.default': (0,),
    'aten.max_pool2d_with_indices.default': (0, 1),
    'aten.max_pool3d_with_indices.default': (0, 1),
    'aten.cummax.default': (0, 1),
    'aten.cummin.default': (0, 1),
    'aten.sort.stable': (0, 1),
    'aten.sort.default': (0,),
    'aten.max.dim': (0, 1),
    'aten.min.dim': (0, 1),
    'aten.topk.default': (0, 1),
    'aten.native_group_norm.default': (0, 1, 2),
    'aten._native_batch_norm_legit.no_stats': (0, 1, 2),
    'aten._native_batch_norm_legit_functional.default': (0, 1, 2, 3, 4),
    'aten.native_dropout.default': (0, 1),
    'aten._embedding_bag.default': (0,),
    'aten.embedding_bag.padding_idx': (0,),
    'aten.split_with_sizes.default': None,
    'aten.split.Tensor': None,
    'aten.chunk.default': None,
}
_COMPONENT_OWNERS = {
    'aten.native_layer_norm.default': 'the mean and reciprocal standard deviation outputs',
    'aten._native_batch_norm_legit_no_training.default': 'the saved statistics outputs',
    'aten._embedding_bag.default': 'the offset2bag, bag size and max index outputs',
    'aten.embedding_bag.padding_idx': 'the offset2bag, bag size and max index outputs',
    'aten.sort.default': ('the index outputs (without stable=True ATen orders equal entries, -0 '
                          'and +0 among them, either way)'),
}

# The values of these tuple-valued operators are one IR node (or the extremum composition).
_ORDER_COMPONENTS = ('aten.cummax.default', 'aten.cummin.default', 'aten.sort.stable',
                     'aten.sort.default', 'aten.max.dim', 'aten.min.dim')

# at::acc_type on the CPU, in which cpu_cum_base_kernel accumulates cumsum and cumprod.
_SCAN_ACCUMULATORS = {'float64': 'float64', 'float32': 'float64', 'bfloat16': 'float32',
                      'float16': 'float32'}
_CREATION = ('aten.full.default', 'aten.ones.default', 'aten.zeros.default',
             'aten.scalar_tensor.default', 'aten.arange.default', 'aten.arange.start',
             'aten.arange.start_step')
_SHAPED_CREATION = ('aten.full_like.default', 'aten.ones_like.default',
                    'aten.fill.Scalar', 'aten.new_full.default', 'aten.new_ones.default',
                    'aten.new_zeros.default')
_ALIASES = ('aten.clone.default', 'aten.alias.default', 'aten.lift_fresh_copy.default')

# The default graph's tensor literal is lift_fresh_copy then detach_. As detach_ writes autograd
# state, not data, a folded detach_ is its operand's value.
_DETACH = ('aten.detach_.default',)
_STRUCTURE_PLANS = ('select', 'addmm', 'repeat', 'pad', 'reflect', 'replicate', 'reduce', 'var',
                    'scan', 'extremum', 'leaky_relu', 'diagonal', 'scatter', 'normalization',
                    'registered', 'gather_at', 'scatter_at', 'embedding', 'index', 'index_put',
                    'boolean_reduce', 'col2im', 'scatter_mean', 'take_along_dim',
                    'adaptive_pool', 'upsample_nearest', 'rfft', 'irfft', 'moments',
                    'distance', 'upsample_linear', 'grid_sample', 'random', 'dropout',
                    'masked_scatter', 'nonzero_static', 'embedding_bag', 'topk',
                    'pool_indices', 'scatter_fold', 'buffer_update', 'attention')
_FOLDED = {}


def _folds(node):
    # Whether `node` is a deterministic ATen operator with an integer or Boolean tensor result,
    # which the capture computes as PyTorch does when its operands are constants. An in-place
    # operator is not folded: running it would change a captured tensor, such as a buffer.
    # detach_ is folded without being run (_DETACH).
    meta = node.meta.get('val', node.meta.get('tensor_meta'))
    dtype = getattr(meta, 'dtype', None)
    return (node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload) and
            torch.Tag.nondeterministic_seeded not in node.target.tags and
            (not _mutates(node) or _is_op(node, _DETACH)) and
            dtype is not None and not dtype.is_floating_point and not dtype.is_complex)


def _index_constant(value):
    # The value of an input-independent index or mask (a buffer, a lifted constant, a created
    # tensor, or an integer or Boolean operator on such values), computed by PyTorch at capture,
    # or None for a value computed from the inputs.
    if not isinstance(value, torch.fx.Node):
        return None
    if value in _FOLDED:
        return _FOLDED[value]
    result = None
    if _is_op(value, _ALIASES + _DETACH):
        result = _index_constant(value.args[0])
    elif _is_op(value, _CREATION + _SHAPED_CREATION):
        result = _creation_value(value)
    elif value.name in _TENSOR_VALUES:
        result = _TENSOR_VALUES[value.name]
    elif _folds(value):
        missing = []
        def resolve(operand):
            folded = _index_constant(operand)
            if folded is None: missing.append(operand)
            return folded
        args = torch.fx.node.map_arg(value.args, resolve)
        kwargs = torch.fx.node.map_arg(value.kwargs, resolve)
        if not missing:
            result = value.target(*args, **kwargs)
    _FOLDED[value] = result
    return result


def _index_value(value):
    # The value of a constant index operand, which must exist.
    value = _index_constant(value)
    if value is None:
        raise NotImplementedError('a computed index has no constant value')
    if value.dtype not in (torch.int64, torch.int32) or value.dim() == 0:
        raise NotImplementedError('a gather or scatter index must be an int64 or int32 tensor')
    return value


def _index_operand(value, name):
    # A runtime index operand: an int64 or int32 tensor of the graph.
    if not isinstance(value, torch.fx.Node) or _node_dtype(value) not in _INDEX_DTYPES:
        dtype = _node_dtype(value) if isinstance(value, torch.fx.Node) else type(value).__name__
        raise NotImplementedError(f'{name}: an index operand must be an int64 or int32 tensor, '
                                  f'not {dtype}')
    return value


def _op_record(tag, fields):
    # The codec record of the registered operator `tag` of TorchLean.Op, fields in binder order.
    return {'version': _op_table()['version'], 'tag': tag, 'fields': fields}


def _registered_node(nodes, tag, fields, parents, shape, dtype, scalar=None, output=0):
    # The one constructor of an IR node holding a registered operator of TorchLean.Op: output
    # `output` of the operator whose codec record is (tag, fields), its value ports at `scalar`
    # (the node's dtype unless given).
    scalar = dtype if scalar is None else scalar
    return _append_node(nodes, {'kind': "custom", 'primitive': 'op',
                                'op': _op_record(tag, fields), 'scalar': scalar,
                                'output': output, 'parents': parents, 'shape': shape,
                                'dtype': dtype})


def _iota(nodes, shape, axis):
    # The int64 constant of `shape` whose entry at c is c[axis].
    view = [-1 if a == axis else 1 for a in range(len(shape))]
    values = torch.arange(shape[axis], dtype=torch.int64).reshape(view).expand(shape)
    return _append_node(nodes, {'parents': [], **_constant(values.contiguous())})


def _index_tuples(nodes, components, shape):
    # Index tensors of `shape` stacked along an extra trailing axis (the IR layout of an index port
    # with several bounds). A single component is the port itself.
    if len(components) == 1:
        return components[0]
    pieces = [_append_node(nodes, {**_reshape_kind(), 'parents': [c], 'shape': shape + [1],
                                   'dtype': nodes[c]['dtype']}) for c in components]
    return _append_node(nodes, {'kind': "cat", 'dim': len(shape), 'parents': pieces,
                                'shape': shape + [len(components)], 'dtype': _INT64})


def _wrapped_index(nodes, index, extent):
    # PyTorch's advanced indexing reads a negative index i as i + extent, computed here as
    # where(i < 0, i + extent, i) in int64. That is exact for every int64 or int32 index when
    # extent < 2 ** 63: a sum that wraps belongs to a nonnegative i and is not selected. An index
    # outside [-extent, extent) stays outside, and the registered operator's range check fails at
    # execution, where PyTorch raises an IndexError.
    index = _cast_to(nodes, index, _INT64)
    shape = nodes[index]['shape']
    zero = _broadcast_to(nodes, _scalar_constant(nodes, 0, _INT64), shape)
    size = _broadcast_to(nodes, _scalar_constant(nodes, extent, _INT64), shape)
    negative = _append_node(nodes, {'kind': 'lt', 'parents': [index, zero],
                                    'shape': shape, 'dtype': 'bool'})
    shifted = _append_node(nodes, {'kind': "add", 'parents': [index, size], 'shape': shape,
                                   'dtype': _INT64})
    return _append_node(nodes, {'kind': 'where',
                                'parents': [negative, shifted, index], 'shape': shape,
                                'dtype': _INT64})


def _advanced_indices(nodes, node_to_id, values, axes, extents, positions):
    # The index tuples of PyTorch's advanced indexing along `axes`: each index broadcast to
    # `positions`, a negative entry i read as i + extent, stacked when there are several. A
    # constant index is wrapped at capture, and refused outside [-extent, extent).
    components = []
    for value, axis in zip(values, axes):
        extent = extents[axis]
        constant = _index_constant(value)
        if constant is not None:
            constant = constant.long()
            wrapped = torch.where(constant < 0, constant + extent, constant)
            if ((wrapped < 0) | (wrapped >= extent)).any():
                raise NotImplementedError(f'an index is outside an axis of extent {extent}')
            index = _append_node(nodes, {'parents': [], **_constant(
                wrapped.broadcast_to(positions).contiguous())})
        else:
            index = node_to_id.get(value)
            if index is None: raise NotImplementedError('an index operand has no tensor lowering')
            index = _broadcast_to(nodes, _wrapped_index(nodes, index, extent), positions)
        components.append(index)
    return _index_tuples(nodes, components, positions)


def _advanced_axes(node, source, indices):
    # The indexed axes and the broadcast index shape of index.Tensor and index_put. A Boolean or
    # uint8 mask index has a data-dependent result shape and is refused.
    name = _target_name(node.target)
    extents = _shape_from_arg(source)
    if _node_dtype(source) not in _FLOATING:
        raise NotImplementedError(f'{name} of a {_node_dtype(source)} tensor is not imported')
    if len(indices) > len(extents):
        raise NotImplementedError(f'{name}: more indices than axes')
    axes = [axis for axis, index in enumerate(indices) if index is not None]
    if not axes:
        raise NotImplementedError(f'{name} without an index tensor')
    for axis in axes:
        index = indices[axis]
        dtype = _node_dtype(index) if isinstance(index, torch.fx.Node) else None
        if dtype in ('bool', {'kind': 'integer', 'signed': False, 'width': 8}):
            raise NotImplementedError(f'{name} with a Boolean mask index: the result shape '
                                      'depends on the mask')
        _index_operand(index, name)
    positions = list(torch.broadcast_shapes(*[tuple(_shape_from_arg(indices[a])) for a in axes]))
    return axes, positions


def _slice_positions(extent, start, end, step):
    # The indices start, start + step, ... below end along an axis, as a slice reads them.
    if step is None: step = 1
    if int(step) <= 0: raise NotImplementedError('a slice step must be positive')
    return list(range(*slice(start, end, int(step)).indices(extent)))


def _coordinates(index, dim, extents):
    # The complete coordinate of every entry of `index`, row-major with the coordinate axis last:
    # the entry's own coordinate with `dim` replaced by the entry. PyTorch rejects a negative index.
    grids = torch.meshgrid(*[torch.arange(n) for n in index.shape], indexing='ij')
    coords = torch.stack([index.long() if a == dim else grids[a]
                          for a in range(index.dim())], -1)
    if len(extents) != index.dim() or (coords < 0).any() or \
            (coords >= torch.tensor(extents)).any():
        raise NotImplementedError('a gather or scatter index is outside its axis')
    return coords


# Producers of complex values. The IR has no complex dtype: a real Fourier transform is read only
# through view_as_real or an inverse transform, and view_as_complex of a real tensor only through an
# inverse transform, possibly after slices.
_COMPLEX_PRODUCERS = ('aten._fft_r2c.default', 'aten.fft_rfft.default',
                      'aten.view_as_complex.default')
_COMPLEX_SLICES = ('aten.slice.Tensor',)
_INVERSE_TRANSFORMS = ('aten._fft_c2r.default', 'aten.fft_irfft.default')


def _complex_producer(node):
    if node.op != 'call_function': return False
    if _is_op(node, _COMPLEX_SLICES):
        meta = node.meta.get('val', node.meta.get('tensor_meta'))
        return getattr(getattr(meta, 'dtype', None), 'is_complex', False)
    return _is_op(node, _COMPLEX_PRODUCERS)


def _complex_consumer(user, producer):
    # Whether `user` reads the complex value `producer` in its paired-real form: view_as_real
    # or an inverse transform of a forward transform, an inverse transform or a slice of
    # view_as_complex or of a complex slice.
    if _is_op(producer, ('aten.view_as_complex.default',) + _COMPLEX_SLICES):
        return (_is_op(user, _INVERSE_TRANSFORMS + _COMPLEX_SLICES) and
                user.args[0] is producer)
    return _is_op(user, ('aten.view_as_real.default',)) or (
        _is_op(user, _INVERSE_TRANSFORMS) and user.args[0] is producer)


def _forward_transform(producer):
    # The real operand, dim, length and normalization of _fft_r2c or fft_rfft.
    source = producer.args[0]
    extents = _shape_from_arg(source)
    if _is_op(producer, ('aten._fft_r2c.default',)):
        dims, code, onesided = producer.args[1], producer.args[2], producer.args[3]
        if len(dims) != 1 or not onesided:
            raise NotImplementedError('a real Fourier transform along several axes or two-sided '
                                      'is not imported')
        dim = _normalize_axis(dims[0], len(extents))
        n, scale = extents[dim], {0: None, 1: 'sqrt', 2: 'n'}[int(code)]
    else:
        n, dim, norm = (_argument(producer, 1, 'n'), _argument(producer, 2, 'dim'),
                        _argument(producer, 3, 'norm'))
        dim = _normalize_axis(-1 if dim is None else dim, len(extents))
        n = extents[dim] if n is None else int(n)
        scale = {None: None, 'backward': None, 'ortho': 'sqrt', 'forward': 'n'}[norm]
    if _node_dtype(source) not in ('float32', 'float64') or n <= 0:
        raise NotImplementedError('rfft needs a binary32 or binary64 operand and n > 0')
    return {'source': source, 'dim': dim, 'n': n, 'scale': scale}


def _fourier_scale(code, n, dtype, inverse):
    # The factor applied after TorchLean's transform, computed in the dtype, or None where there
    # is none. The forward transform is unnormalized, so for `code` 'sqrt' or 'n' the factor is
    # 1 / sqrt(n) or 1 / n, and the inverse divides by n, so it is sqrt(n) or n.
    one = torch.tensor(1.0, dtype=_torch_dtype(dtype))
    size = torch.tensor(float(n), dtype=_torch_dtype(dtype))
    if code == 'sqrt':
        return (size.sqrt() if inverse else one / size.sqrt()).item()
    if code == 'n':
        return (size if inverse else one / size).item()
    return None


def _component_producer(node):
    return node.op == 'call_function' and _is_op(node, tuple(_COMPONENTS))


def _projection(node):
    # (producer, index) when `node` projects one component of a tuple-valued operator the IR
    # imports by component.
    if not _is_getitem(node) or len(node.args) != 2: return None
    producer, index = node.args
    if not isinstance(producer, torch.fx.Node) or type(index) is not int: return None
    if not _component_producer(producer): return None
    return producer, index


def _split_windows(producer):
    # The dimension of a split and the start and length of each component along it.
    source = _shape_from_arg(producer.args[0])
    dim = _normalize_axis(_argument(producer, 2, 'dim') or 0, len(source))
    extent = source[dim]
    if _is_op(producer, ('aten.split_with_sizes.default',)):
        sizes = [int(size) for size in producer.args[1]]
    else:
        count = int(producer.args[1])
        if count <= 0: raise NotImplementedError('a split needs a positive size or chunk count')
        size = count if _is_op(producer, ('aten.split.Tensor',)) else -(-extent // count)
        sizes = [min(size, extent - start) for start in range(0, extent, max(size, 1))] or [0]
    if sum(sizes) != extent or any(size < 0 for size in sizes):
        raise NotImplementedError(f'split sizes {sizes} do not cover extent {extent}')
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    return dim, list(zip(starts, sizes))


def _moments_kind(source, mode, weight, bias, eps, output, groups=1, running=None,
                  record=False):
    # A group norm (statistics over each group of channels and the spatial axes) or a training
    # batch norm (over every axis but channel axis 1), output 0 the normalized values, 1 the
    # means and 2 the reciprocal standard deviations. With running = (running_mean,
    # running_var, momentum) outputs 3 and 4 are the updated running statistics, and `record`
    # keeps them as the updates of those buffers (_BUFFER_UPDATES) when output 0 is emitted.
    extents = _shape_from_arg(source)
    if len(extents) < 2 or _node_dtype(source) not in _FLOATING:
        raise NotImplementedError(f'{mode} norm needs a floating operand with a channel axis 1')
    if extents[1] % groups:
        raise NotImplementedError('group norm needs the groups to divide the channels')
    eps = float(eps)
    if not eps >= 0:
        raise NotImplementedError('a negative or NaN eps: PyTorch takes the square root of '
                                  'var + eps, which the IR clamps at zero')
    plan = {'op': 'moments', 'operands': [source, weight, bias], 'mode': mode,
            'groups': groups, 'eps': eps, 'output': output}
    if running is not None:
        mean, var, momentum = running
        dtype, channels = _node_dtype(source), int(extents[1])
        count = 1
        for extent in extents: count *= int(extent)
        if dtype not in ('float32', 'float64'):
            raise NotImplementedError('running-statistic updates of a binary16 or bfloat16 '
                                      'batch norm are not imported')
        if any(not isinstance(value, torch.fx.Node) or _node_dtype(value) != dtype or
               list(_shape_from_arg(value)) != [channels] for value in (mean, var)):
            raise NotImplementedError('batch norm running statistics need the shape [C] and '
                                      'the type of the input')
        if count < 2 * channels:
            # With C > 0 channels, fewer than two values per channel: F.batch_norm refuses one,
            # and an empty batch has no unbiased variance.
            raise NotImplementedError('a training batch norm with running statistics needs at '
                                      'least two values per channel')
        momentum = float(momentum)
        if momentum != momentum or abs(momentum) == float('inf'):
            raise NotImplementedError('a batch norm momentum must be finite')
        plan['operands'] += [mean, var]
        plan.update(momentum=momentum, record=record)
    return {'kind': "layer_norm", '_plan': plan}


_BUFFERS = []
_OUTPUT_TARGETS = []
_BUFFER_UPDATES = {}


def _read_signature(signature):
    # The buffer placeholders of an exported program in input order and the target of each
    # graph output: None for a user output, the placeholder of the buffer a BUFFER_MUTATION
    # output updates. Every other mutation or effect (of a user input or a parameter, a token)
    # is refused. Without a signature (FX graphs) there are no buffers.
    _BUFFERS.clear()
    _OUTPUT_TARGETS.clear()
    _BUFFER_UPDATES.clear()
    if signature is None:
        return
    placeholders = {}
    for spec in signature.input_specs:
        if getattr(spec.kind, 'name', '') == 'BUFFER':
            _BUFFERS.append(spec.arg.name)
            placeholders[spec.target] = spec.arg.name
    for spec in signature.output_specs:
        kind = getattr(spec.kind, 'name', '')
        if kind == 'USER_OUTPUT':
            _OUTPUT_TARGETS.append(None)
        elif kind == 'BUFFER_MUTATION' and spec.target in placeholders:
            _OUTPUT_TARGETS.append(placeholders[spec.target])
        else:
            raise NotImplementedError('exported mutation/effects other than buffer updates are '
                                      'outside the pure graph contract')


def _buffer_increment(node):
    # The default graph's num_batches_tracked update: add_ of the integer 1 to an int64 scalar
    # buffer placeholder that no other node reads.
    if not isinstance(node, torch.fx.Node) or not _is_op(node, ('aten.add_.Tensor',)):
        return False
    target, other = node.args[0], _argument(node, 1, 'other')
    return isinstance(target, torch.fx.Node) and target.op == 'placeholder' and \
        target.name in _BUFFERS and set(target.users) == {node} and \
        _node_dtype(target) == _INT64 and list(_shape_from_arg(target)) == [] and \
        type(other) is int and other == 1 and _argument(node, 2, 'alpha') in (None, 1)


def _updated_buffers():
    # The placeholders of the buffers the program returns after its own outputs, in order.
    return [name for name in _BUFFERS
            if name in _BUFFER_UPDATES or name in _OUTPUT_TARGETS]


def _program_outputs(refs):
    # The program's outputs: the graph's user outputs, then the value of each updated buffer in
    # the order of the buffer placeholders, from a core graph's BUFFER_MUTATION outputs or a
    # default graph's batch_norm and add_ rules. The IR holds no mutation.
    if all(target is None for target in _OUTPUT_TARGETS) and not _BUFFER_UPDATES:
        return refs
    if _OUTPUT_TARGETS and len(_OUTPUT_TARGETS) != len(refs):
        raise NotImplementedError('the graph outputs do not match the exported signature')
    # An updated buffer must not share storage with another captured tensor (another updated
    # buffer, or a buffer, parameter or constant read as captured): eager execution would see
    # the update there. Disjoint views of one storage are refused too.
    def storage(value):
        if isinstance(value, torch.Tensor) and value.numel():
            return value.untyped_storage().data_ptr()
        return None
    for name in _updated_buffers():
        mine = storage(_TENSOR_VALUES.get(name))
        if mine is not None and any(other != name and storage(value) == mine
                                    for other, value in _TENSOR_VALUES.items()):
            raise NotImplementedError(f'the updated buffer {name} shares storage with another '
                                      'captured tensor')
    updates, user = dict(_BUFFER_UPDATES), []
    for ref, target in zip(refs, _OUTPUT_TARGETS or [None] * len(refs)):
        if target is None:
            user.append(ref)
        elif target in updates:
            raise NotImplementedError(f'buffer {target} is updated twice')
        else:
            updates[target] = ref
    return user + [updates[name] for name in _BUFFERS if name in updates]


def _random_key(position, counter_count):
    # The key of the RNG node at capture position j: splitmix64(0) + j * 2 ** 32 modulo 2 ** 64.
    # The node reads the counters key + t for t < counter_count, one per entry for
    # Op.randUniform and Op.bernoulliMask and two per entry for Op.randn. Nodes reading at most
    # 2 ** 32 counters at positions below 2 ** 32 therefore read disjoint counters, though not
    # always distinct values: the reduction modulo 2 ** 32 and the rounding can repeat one.
    # Running the artifact again repeats its draws, and editing the graph can move a node and
    # change them.
    if position < 0 or counter_count < 0:
        raise ValueError('an RNG node position and counter count are nonnegative')
    if position >= 2 ** 32 or counter_count > 2 ** 32:
        raise NotImplementedError('an RNG node at or beyond position 2 ** 32 or reading more '
                                  'than 2 ** 32 counters')
    return (_splitmix64(0) + position * 2 ** 32) % 2 ** 64


_DROPOUT_MASKS = {}


def _dropout_kind(source, p, train, output, producer=None):
    # native_dropout(input, p, train): the input and an all-true mask in evaluation, otherwise a
    # Bernoulli keep mask of probability 1 - p from TorchLean's generator.
    if _node_dtype(source) not in _FLOATING:
        raise NotImplementedError(f'dropout of a {_node_dtype(source)} tensor is not imported')
    if train is None:
        raise NotImplementedError('native_dropout without a train flag is not imported')
    p = float(p)
    if not 0 <= p <= 1:
        raise NotImplementedError('dropout needs 0 <= p <= 1')
    return {'kind': "mul_elem", '_plan': {'op': 'dropout', 'operands': [source], 'p': p,
                                     'train': bool(train), 'output': output,
                                     'producer': producer}}


def _embedding_bag_kind(producer):
    # _embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse,
    # per_sample_weights, include_last_offset, padding_idx), output 0, with constant offsets.
    weight, indices, offsets = producer.args[:3]
    mode = int(_argument(producer, 4, 'mode') or 0)
    weights = _argument(producer, 6, 'per_sample_weights')
    last = bool(_argument(producer, 7, 'include_last_offset'))
    padding = _argument(producer, 8, 'padding_idx')
    if padding is not None and int(padding) >= 0:
        raise NotImplementedError('embedding_bag with a padding_idx is not imported')
    if weights is not None and (mode != 0 or _shape_from_arg(weights) != _shape_from_arg(indices)):
        raise NotImplementedError('per_sample_weights need mode sum and the shape of the indices')
    _index_operand(indices, 'embedding_bag')
    if len(_shape_from_arg(indices)) != 1 or _node_dtype(weight) not in _FLOATING:
        raise NotImplementedError('embedding_bag needs one-dimensional indices and a floating '
                                  'weight')
    offsets = _index_constant(offsets)
    if offsets is None:
        raise NotImplementedError('embedding_bag with computed offsets: the bags depend on the '
                                  'inputs')
    starts = [int(value) for value in offsets.reshape(-1).tolist()]
    count = _shape_from_arg(indices)[0]
    if last and (not starts or starts[-1] != count):
        raise NotImplementedError('embedding_bag with include_last_offset needs the last offset '
                                  'to be the number of indices')
    ends = starts[1:] + ([] if last else [count])
    starts = starts[:len(ends)]
    if not ends:
        raise NotImplementedError('embedding_bag without bags is not imported')
    if starts[0] != 0 or any(not 0 <= a <= b <= count
                                         for a, b in zip(starts, ends)):
        raise NotImplementedError('embedding_bag offsets must start at 0 and be nondecreasing '
                                  'within the indices')
    return {'kind': "gather", '_plan': {'op': 'embedding_bag',
                                        'operands': [weight, indices, weights],
                                        'mode': mode, 'bags': list(zip(starts, ends))}}


def _component_kind(node, producer, index):
    name = _target_name(producer.target)
    allowed = _COMPONENTS[name]
    if allowed is not None and index not in allowed:
        raise NotImplementedError(f'{name}: {_COMPONENT_OWNERS[name]} are not imported')
    source = producer.args[0]
    if allowed is None:
        dim, windows = _split_windows(producer)
        if not 0 <= index < len(windows):
            raise NotImplementedError(f'{name}: component {index} is out of range')
        start, length = windows[index]
        return {'kind': "slice", 'dims': [dim], 'starts': [start], 'stops': [start + length],
                'steps': [1], '_operands': [source]}
    if name in _ORDER_COMPONENTS:
        return _order_kind(name, producer, source, index)
    if name == 'aten.native_layer_norm.default':
        normalized = [int(size) for size in producer.args[1]]
        weight, bias = _argument(producer, 2, 'weight'), _argument(producer, 3, 'bias')
        rank = len(_shape_from_arg(source))
        return _normalization("layer_norm", source, _argument(producer, 4, 'eps'),
                              {'dims': list(range(rank - len(normalized), rank))}, normalized,
                              gamma=weight, beta=bias)
    if name == 'aten.native_dropout.default':
        return _dropout_kind(source, producer.args[1], producer.args[2], index, producer)
    if name in ('aten._embedding_bag.default', 'aten.embedding_bag.padding_idx'):
        return _embedding_bag_kind(producer)
    if name == 'aten.native_group_norm.default':
        # native_group_norm(input, weight, bias, N, C, HxW, group, eps).
        return _moments_kind(source, 'group', producer.args[1], producer.args[2],
                             producer.args[7], index, groups=int(producer.args[6]))
    if name == 'aten._native_batch_norm_legit.no_stats':
        # Training-mode batch norm without running statistics (weight, bias, training, momentum,
        # eps): the statistics of the batch.
        return _moments_kind(source, 'batch', producer.args[1], producer.args[2],
                             producer.args[5], index)
    if name == 'aten._native_batch_norm_legit_functional.default':
        # The core graph's training batch norm (weight, bias, running_mean, running_var,
        # training, momentum, eps): outputs 3 and 4 are the updated running statistics.
        if producer.args[5] is not True:
            raise NotImplementedError('_native_batch_norm_legit_functional outside training')
        if producer.args[3] is producer.args[4]:
            raise NotImplementedError('_native_batch_norm_legit_functional: the running '
                                      'statistics must be two distinct buffers')
        return _moments_kind(source, 'batch', producer.args[1], producer.args[2],
                             producer.args[7], index,
                             running=(producer.args[3], producer.args[4], producer.args[6]))
    if name == 'aten._native_batch_norm_legit_no_training.default':
        shape = _shape_from_arg(source)
        if len(shape) < 2: raise NotImplementedError('batch norm needs a channel axis 1')
        channels = int(shape[1])
        weight, bias = _argument(producer, 1, 'weight'), _argument(producer, 2, 'bias')
        return _normalization("batch_norm", source, _argument(producer, 6, 'eps'), {'dim': 1},
                              [channels], gamma=weight, beta=bias,
                              mean=_argument(producer, 3, 'running_mean'),
                              var=_argument(producer, 4, 'running_var'))
    if name == 'aten.topk.default':
        return _topk_kind(producer, source, index)
    rank = 2 if name == 'aten.max_pool2d_with_indices.default' else 3
    kernel = _spatial_tuple(producer.args[1], rank, 1)
    stride = _argument(producer, 2, 'stride')
    stride = _spatial_tuple(stride if stride not in (None, []) else kernel, rank, 1)
    padding = _spatial_tuple(_argument(producer, 3, 'padding') or 0, rank, 0)
    if _spatial_tuple(_argument(producer, 4, 'dilation') or 1, rank, 1) != [1] * rank:
        raise NotImplementedError('dilated max pooling waits for the general pool')
    if _argument(producer, 5, 'ceil_mode'):
        raise NotImplementedError('max pooling with ceil_mode=True waits for the general pool')
    if index == 1:
        if _node_dtype(source) not in _FLOATING:
            raise NotImplementedError(f'max pooling indices of a {_node_dtype(source)} tensor')
        return {'kind': "gather", '_plan': {'op': 'pool_indices', 'operands': [source],
                                            'kernel': kernel, 'stride': stride,
                                            'padding': padding}}
    return {'kind': "max_pool", 'spatial_rank': rank, 'kernel': kernel, 'stride': stride,
            'padding': padding, '_operands': [source]}


def _registered_kind(tag, fields, source, input_shape, result_shape, output=0):
    # Output `output` of the registered operator `tag` of TorchLean.Op, its codec record's fields
    # in binder order and its value ports at the operand's dtype. The operand is reshaped to
    # `input_shape` first when that differs from its shape.
    return {'kind': tag, '_plan': {'op': 'registered', 'operands': [source], 'tag': tag,
                                   'fields': fields, 'scalar': _node_dtype(source),
                                   'output': output, 'input_shape': input_shape,
                                   'result_shape': result_shape}}


def _topk_kind(producer, source, index):
    # topk(self, k, dim=-1, largest=True, sorted=True): output `index` (values, int64 indices)
    # of the IR's stable topk, whose ties the plan checks at execution.
    shape, dtype = _shape_from_arg(source), _node_dtype(source)
    if not shape:
        raise NotImplementedError('topk of a rank-0 tensor is not imported')
    if dtype not in _FLOATING:
        raise NotImplementedError(f'topk of a {dtype} tensor is not imported')
    k, dim = int(producer.args[1]), _argument(producer, 2, 'dim')
    dim = _normalize_axis(-1 if dim is None else dim, len(shape))
    largest, ordered = _argument(producer, 3, 'largest'), _argument(producer, 4, 'sorted')
    if not 0 <= k <= shape[dim]:
        raise NotImplementedError(f'topk: k = {k} is outside [0, {shape[dim]}]')
    if ordered is False and k > 1:
        raise NotImplementedError('topk with sorted=False: ATen leaves the order of the k '
                                  'entries unspecified')
    return {'kind': "topk", '_plan': {'op': 'topk', 'operands': [source], 'dim': dim, 'k': k,
                                      'largest': largest is not False, 'output': index}}


def _mutates(node):
    # Whether the node's ATen schema writes into one of its arguments (an in-place operator).
    schema = getattr(node.target, '_schema', None)
    return schema is not None and any(
        argument.alias_info is not None and argument.alias_info.is_write
        for argument in schema.arguments)


_ALLOCATING = ('aten.zeros.default', 'aten.ones.default', 'aten.zeros_like.default',
               'aten.ones_like.default', 'aten.full_like.default', 'aten.new_zeros.default',
               'aten.new_ones.default', 'aten.new_full.default', 'aten._to_copy.default',
               'aten.lift_fresh_copy.default')


def _allocates(node):
    # Whether the node's result is a fresh tensor: an audited creation operator, or a functional
    # core or pointwise ATen operator without an alias annotation and not tagged
    # maybe_aliasing_or_mutating (conj_physical of a real tensor returns its input). Composite
    # operators such as type_as and _unsafe_view are neither, since their annotations can omit
    # an alias.
    if not isinstance(node, torch.fx.Node) or node.op != 'call_function':
        return False
    if _is_op(node, _ALLOCATING):
        return True
    schema, tags = getattr(node.target, '_schema', None), getattr(node.target, 'tags', ())
    return schema is not None and not _mutates(node) and \
        any(tag in tags for tag in (torch.Tag.core, torch.Tag.pointwise)) and \
        torch.Tag.maybe_aliasing_or_mutating not in tags and \
        all(value.alias_info is None for value in schema.returns)


def _fresh_destination(node):
    # torch.export's default graph keeps index_put_ and reads the written tensor afterwards
    # through the index_put_ node. The functional index_put is exact when the destination is
    # fresh (an allocating operator's result or an accepted index_put_), the index_put_ is its
    # only reader from then on, and its earlier readers allocate their results.
    target = node.args[0]
    if not isinstance(target, torch.fx.Node) or target.op != 'call_function':
        raise NotImplementedError('index_put_ into an input, parameter or buffer is outside the '
                                  'pure graph contract')
    if not _allocates(target) and not _is_op(target, ('aten.index_put_.default',)):
        raise NotImplementedError('index_put_ into a view or a tensor that may alias another '
                                  'is not imported')
    if any(value is target for value in list(node.args[1]) + [node.args[2]]):
        raise NotImplementedError('index_put_ whose indices or values are its destination')
    order = {other: position for position, other in enumerate(node.graph.nodes)}
    for user in target.users:
        if user is not node and (order[user] > order[node] or not _allocates(user)):
            raise NotImplementedError('index_put_ into a tensor that a view or a later node '
                                      'reads')


def _order_kind(name, producer, source, index=0):
    # The values of a running extremum, of a stable sort, or of max.dim and min.dim, or their
    # indices (component 1), the registered operator's index output.
    shape, dtype = _shape_from_arg(source), _node_dtype(source)
    if not shape:
        raise NotImplementedError(f'{name} of a rank-0 tensor is not imported')
    if dtype not in _FLOATING:
        raise NotImplementedError(f'{name} of a {dtype} tensor is not imported')
    if name in ('aten.sort.stable', 'aten.sort.default'):
        if name == 'aten.sort.default' or producer.kwargs.get('stable') is not True:
            raise NotImplementedError('sort without stable=True: ATen orders equal entries, -0 '
                                      'and +0 among them, either way; the IR sort is stable')
        dim = _normalize_axis(producer.kwargs.get('dim', -1), len(shape))
        descending = bool(producer.kwargs.get('descending', False))
        if index == 1:
            return _registered_kind('sort', [{'shape': shape}, {'nat': dim},
                                             {'bool': descending}], source, shape, shape, 1)
        return {'kind': "sort", 'dim': dim, 'descending': descending, '_operands': [source]}
    dim = _normalize_axis(_argument(producer, 1, 'dim'), len(shape))
    if name in ('aten.cummax.default', 'aten.cummin.default'):
        # cummax_cummin_helper takes a later equal entry and a later NaN, as the IR does.
        tag = 'cummax' if name == 'aten.cummax.default' else 'cummin'
        if index == 1:
            return _registered_kind(tag, [{'shape': shape}, {'nat': dim}], source, shape, shape,
                                    1)
        return {'kind': "cummax" if name == 'aten.cummax.default' else "cummin", 'dim': dim,
                '_operands': [source]}
    if shape[dim] == 0:
        raise NotImplementedError(f'{name} of an empty axis has no value')
    if index == 1:
        keepdim = bool(_argument(producer, 2, 'keepdim'))
        return _extremum_index_kind('argmax' if name == 'aten.max.dim' else 'argmin', source,
                                    dim, keepdim)
    return {'kind': "topk", '_plan': {'op': 'extremum', 'operands': [source], 'dim': dim,
                                      'largest': name == 'aten.max.dim'}}


def _extremum_index_kind(tag, source, dim, keepdim):
    # Op.argmax or Op.argmin along `dim`: the first largest or smallest entry, a NaN first, as
    # ATen's comparators choose. `dim=None` reduces the row-major flattening, and a scalar is
    # read as one entry. The result is then reshaped to PyTorch's shape.
    shape = _shape_from_arg(source)
    if _node_dtype(source) not in _FLOATING:
        raise NotImplementedError(f'{tag} of a {_node_dtype(source)} tensor is not imported: the '
                                  'CPU reference has arithmetic for binary formats only')
    if dim is None or not shape:
        if shape and dim is not None: raise NotImplementedError(f'{tag}: bad dim {dim}')
        if not shape and dim not in (None, 0, -1):
            raise NotImplementedError(f'{tag}: dim {dim} of a scalar')
        size = 1
        for extent in shape: size *= extent
        input_shape, dim, keepdim = [size], 0, False
    else:
        input_shape, dim = shape, _normalize_axis(dim, len(shape))
    if input_shape[dim] == 0:
        raise NotImplementedError(f'{tag} of an empty axis has no index')
    reduced = [1 if axis == dim else extent for axis, extent in enumerate(input_shape)
               if keepdim or axis != dim]
    return _registered_kind(tag, [{'shape': input_shape}, {'nat': dim}, {'bool': keepdim}],
                            source, input_shape, reduced)


def _filled(shape, fill, dtype):
    # torch.full, with a Python fill value rounded once to a binary16 or bfloat16 dtype, which
    # PyTorch rounds through binary32.
    if dtype in (torch.float16, torch.bfloat16) and isinstance(fill, (int, float)):
        fill = _round_once(fill, str(dtype).removeprefix('torch.'))
    return torch.full(shape, fill, dtype=dtype)


def _creation_value(node):
    # The tensor PyTorch creates, computed with the same arguments on the CPU, except that a
    # fill value becomes binary16 or bfloat16 in one rounding (_filled).
    meta = node.meta.get('val', node.meta.get('tensor_meta'))
    shape, dtype = _node_shape(node), getattr(meta, 'dtype', None)
    if _is_op(node, _SHAPED_CREATION):
        if _is_op(node, ('aten.ones_like.default', 'aten.new_ones.default')):
            fill = 1
        elif _is_op(node, ('aten.new_zeros.default',)):
            fill = 0
        elif _is_op(node, ('aten.full_like.default', 'aten.fill.Scalar')):
            fill = node.args[1]
        else:
            fill = node.args[2]
        if isinstance(fill, torch.fx.Node):
            raise NotImplementedError('a fill value computed by the graph is not a constant')
        return _filled(shape, fill, dtype)
    if any(isinstance(value, torch.fx.Node) for value in (*node.args, *node.kwargs.values())):
        raise NotImplementedError('creation operator with a tensor operand')
    if dtype in (torch.float16, torch.bfloat16) and _is_op(
            node, ('aten.full.default', 'aten.scalar_tensor.default')):
        fill = node.args[1] if _is_op(node, ('aten.full.default',)) else node.args[0]
        return _filled(shape, fill, dtype)
    keywords = {key: value for key, value in node.kwargs.items()
                if key not in ('device', 'pin_memory', 'layout')}
    value = node.target(*node.args, **keywords)
    if list(value.shape) != shape or value.dtype != dtype:
        raise NotImplementedError('a created constant differs from its captured metadata')
    return value


def _structure_kind(node, model=None):
    projection = _projection(node)
    if projection is not None:
        return _component_kind(node, *projection)
    if _buffer_increment(node):
        # The buffer is a constant of the program, so its update is too (computed out of place).
        buffer = node.args[0]
        if buffer.name not in _TENSOR_VALUES:
            raise NotImplementedError(f'buffer {buffer.name} has no captured value')
        return {'kind': "add", '_plan': {'op': 'buffer_update', 'operands': [],
                                         'buffer': buffer.name,
                                         'value': torch.add(_TENSOR_VALUES[buffer.name],
                                                            _argument(node, 1, 'other'))}}
    if _folds(node) and not _is_op(node, _CREATION + _SHAPED_CREATION):
        value = _index_constant(node)
        if value is not None:
            return {'_constant': value}
    if _is_op(node, ('aten.take_along_dim.default',), (torch.take_along_dim,),
              ('take_along_dim',)):
        # ATen's take_along_dim broadcasts the operands except along dim and gathers at the
        # index modulo the extent (a negative or too large index wraps). Without dim it gathers
        # from the flattening, where such an index is an error.
        source, index = _argument(node, 0, 'input', 'self'), _argument(node, 1, 'indices')
        dim = _argument(node, 2, 'dim')
        _index_operand(index, 'take_along_dim')
        if _node_dtype(source) not in _FLOATING:
            raise NotImplementedError(f'take_along_dim of a {_node_dtype(source)} tensor')
        extents, positions = _shape_from_arg(source), _shape_from_arg(index)
        if dim is not None:
            dim = _normalize_axis(dim, len(extents))
            if len(positions) != len(extents):
                raise NotImplementedError('take_along_dim needs an index of the input rank')
        return {'kind': "gather", '_plan': {'op': 'take_along_dim', 'operands': [source, index],
                                            'dim': dim}}
    if _is_op(node, ('aten.argmax.default', 'aten.argmin.default'), (torch.argmax, torch.argmin),
              ('argmax', 'argmin')):
        source = _argument(node, 0, 'input', 'self')
        dim = _argument(node, 1, 'dim')
        tag = 'argmin' if _is_op(node, ('aten.argmin.default',), (torch.argmin,),
                                 ('argmin',)) else 'argmax'
        return _extremum_index_kind(tag, source, None if dim is None else int(dim),
                                    bool(_argument(node, 2, 'keepdim')))
    if _is_op(node, _CREATION + _SHAPED_CREATION):
        return {'_constant': _creation_value(node)}
    if _is_op(node, _ALIASES, (torch.clone,), ('clone',)):
        source = _argument(node, 0, 'self', 'input')
        if _node_dtype(source) != _node_dtype(node) or _node_shape(source) != _node_shape(node):
            raise NotImplementedError('an alias must keep its operand shape and dtype')
        return {'_alias': source}
    if _is_op(node, ('aten.copy.default',)):
        # Functional copy_: the source broadcast to the destination and converted to its dtype.
        return {'kind': "custom", '_plan': {'op': 'convert', 'operands': [node.args[1]]}}
    if _is_op(node, ('aten.t.default',), (torch.t,), ('t',)):
        source = node.args[0]
        if len(_shape_from_arg(source)) < 2: return {'_alias': source}
        return {'kind': "permute", 'dims': [1, 0]}
    if _is_op(node, ('aten.expand.default',), (), ('expand',)):
        return _broadcast_kind(_shape_from_arg(node.args[0]), _node_shape(node))
    if _is_op(node, ('aten.unsqueeze.default', 'aten.squeeze.dim', 'aten.squeeze.dims',
                     'aten.squeeze.default', 'aten._unsafe_view.default'),
              (torch.unsqueeze, torch.squeeze), ('unsqueeze', 'squeeze')):
        return _reshape_kind()
    if _is_op(node, ('aten.max.other', 'aten.min.other')):
        return {'kind': "max_elem" if _is_op(node, ('aten.max.other',)) else "min_elem"}
    if _is_op(node, ('aten.cumsum.default', 'aten.cumprod.default'),
              (torch.cumsum, torch.cumprod), ('cumsum', 'cumprod')):
        source, dtype = _argument(node, 0, 'input', 'self'), _node_dtype(node)
        if _argument(node, 2, 'dtype') is not None or _node_dtype(source) != dtype:
            raise NotImplementedError('a cumulative sum or product with a dtype conversion')
        if not isinstance(dtype, str) or dtype not in _SCAN_ACCUMULATORS:
            raise NotImplementedError(f'a cumulative sum or product of a {dtype} tensor')
        dim = _normalize_axis(_argument(node, 1, 'dim'), len(_shape_from_arg(source)))
        scan = "cumsum" if _is_op(node, ('aten.cumsum.default',), (torch.cumsum,),
                                  ('cumsum',)) else "cumprod"
        return {'kind': scan, '_plan': {'op': 'scan', 'operands': [source], 'dim': dim,
                                        'scan': scan, 'accumulator': _SCAN_ACCUMULATORS[dtype]}}
    if _is_op(node, ('aten.leaky_relu.default',), (F.leaky_relu,), ()):
        if node.target is F.leaky_relu and _argument(node, 2, 'inplace'):
            raise NotImplementedError('in-place leaky_relu mutation is outside the pure TorchLean '
                                      'graph contract')
        source, dtype = _argument(node, 0, 'input', 'self'), _node_dtype(node)
        slope = _argument(node, 1, 'negative_slope')
        slope = 0.01 if slope is None else slope
        if dtype not in _FLOATING or _node_dtype(source) != dtype:
            raise NotImplementedError(f'leaky_relu of a {dtype} tensor is not imported')
        if isinstance(slope, bool) or not isinstance(slope, (int, float)):
            raise NotImplementedError('leaky_relu needs a Python number as its slope')
        opmath = 'float32' if dtype in _REDUCED_FLOATS else dtype
        rounded = _scalar_tensor(slope, opmath).item()
        if not rounded > 0:
            raise NotImplementedError('leaky_relu with a slope that is not positive is not a '
                                      'maximum or minimum of x and x * slope')
        return {'kind': "maximum", '_plan': {'op': 'leaky_relu', 'operands': [source],
                                             'slope': slope, 'largest': rounded <= 1}}
    if _is_op(node, ('aten.embedding.default',)):
        # padding_idx, scale_grad_by_freq and sparse change only the gradient.
        weight, index = node.args[0], node.args[1]
        if len(_shape_from_arg(weight)) != 2 or _node_dtype(weight) not in _FLOATING:
            raise NotImplementedError('embedding needs a two-dimensional floating weight')
        _index_operand(index, 'embedding')
        return {'kind': "gather", '_plan': {'op': 'embedding', 'operands': [weight, index]}}
    if _is_op(node, ('aten.index.Tensor',)):
        source, indices = node.args[0], list(node.args[1])
        axes, positions = _advanced_axes(node, source, indices)
        return {'kind': "gather", '_plan': {'op': 'index', 'axes': axes, 'positions': positions,
                                            'operands': [source] + [indices[a] for a in axes]}}
    if _is_op(node, ('aten.index_put.default', 'aten.index_put_.default'), (torch.index_put,),
              ('index_put',)):
        if _is_op(node, ('aten.index_put_.default',)):
            _fresh_destination(node)
        source, indices, values = node.args[0], list(node.args[1]), node.args[2]
        axes, positions = _advanced_axes(node, source, indices)
        if not isinstance(values, torch.fx.Node) or _node_dtype(values) != _node_dtype(source):
            raise NotImplementedError('index_put needs values of the destination dtype')
        return {'kind': "scatter", '_plan': {
            'op': 'index_put', 'axes': axes, 'positions': positions,
            'accumulate': bool(_argument(node, 3, 'accumulate')),
            'operands': [source, values] + [indices[a] for a in axes]}}
    if _is_op(node, tuple(f'aten.{name}.{form}' for name in ('any', 'all')
                          for form in ('default', 'dim', 'dims')), (torch.any, torch.all),
              ('any', 'all')):
        source = _argument(node, 0, 'input', 'self')
        rank, dtype = len(_shape_from_arg(source)), _node_dtype(source)
        if dtype != 'bool' and dtype not in _FLOATING:
            raise NotImplementedError(f'any and all of a {dtype} tensor are not imported')
        dim = _argument(node, 1, 'dim')
        if dim is None:
            dims = list(range(rank))
        elif isinstance(dim, (list, tuple)):
            dims = sorted({_normalize_axis(axis, max(rank, 1)) for axis in dim})
        else:
            dims = [_normalize_axis(dim, max(rank, 1))]
        dims = [axis for axis in dims if axis < rank]
        name = _target_name(node.target) if node.op == 'call_function' and isinstance(
            node.target, torch._ops.OpOverload) else str(getattr(node.target, '__name__',
                                                                  node.target))
        return {'kind': "reduce", '_plan': {'op': 'boolean_reduce', 'operands': [source],
                                            'dims': dims, 'keepdim': bool(_argument(node, 2,
                                                                                    'keepdim')),
                                            'all': 'all' in name}}
    if _is_op(node, ('aten.slice_scatter.default', 'aten.select_scatter.default'),
              (torch.slice_scatter, torch.select_scatter), ('slice_scatter', 'select_scatter')):
        base, update = node.args[0], node.args[1]
        extents = _shape_from_arg(base)
        dim = _normalize_axis(_argument(node, 2, 'dim') or 0, len(extents))
        if _is_op(node, ('aten.select_scatter.default',), (torch.select_scatter,),
                  ('select_scatter',)):
            index = int(_argument(node, 3, 'index'))
            index = index + extents[dim] if index < 0 else index
            if not 0 <= index < extents[dim]:
                raise NotImplementedError(f'select_scatter index {index} is outside its axis')
            positions = [index]
        else:
            positions = _slice_positions(extents[dim], _argument(node, 3, 'start'),
                                         _argument(node, 4, 'end'), _argument(node, 5, 'step'))
        window = [len(positions) if axis == dim else extent
                  for axis, extent in enumerate(extents)]
        view = [-1 if axis == dim else 1 for axis in range(len(extents))]
        index = torch.tensor(positions, dtype=torch.int64).reshape(view).expand(window)
        coords = _coordinates(index, dim, extents)
        return {'kind': "scatter", '_plan': {
            'op': 'scatter', 'operands': [base, update], 'reduce': 'replace',
            'dims': list(range(len(extents))), 'update_shape': window,
            'indices': {'shape': window, 'values': coords.reshape(-1).tolist()}}}
    adaptive = ('aten.adaptive_avg_pool1d.default', 'aten.adaptive_avg_pool2d.default',
                'aten.adaptive_avg_pool3d.default', 'aten._adaptive_avg_pool2d.default',
                'aten._adaptive_avg_pool3d.default')
    if _is_op(node, adaptive, (F.adaptive_avg_pool1d, F.adaptive_avg_pool2d,
                               F.adaptive_avg_pool3d)):
        # The composite adaptive_avg_pool1d/2d/3d is the mean over the spatial axes when every
        # output extent is one, and otherwise, like _adaptive_avg_pool2d/3d, the CPU kernel.
        source = node.args[0]
        name = _target_name(node.target) if isinstance(node.target, torch._ops.OpOverload) \
            else node.target.__name__
        rank = 1 if '1d' in name else 2 if '2d' in name else 3
        extents = _shape_from_arg(source)
        if len(extents) <= rank or _node_dtype(source) not in _FLOATING:
            raise NotImplementedError(f'{name} needs a floating operand with a leading axis')
        sizes = node.args[1] if len(node.args) > 1 else node.kwargs.get('output_size')
        sizes = [sizes] * rank if isinstance(sizes, int) else list(sizes)
        spatial = extents[-rank:]
        sizes = [extent if size is None else int(size) for size, extent in zip(sizes, spatial)]
        if len(sizes) != rank or 0 in spatial or min(sizes) <= 0:
            raise NotImplementedError(f'{name} needs nonempty input and output extents')
        if not name.startswith('aten._') and all(size == 1 for size in sizes):
            dims = list(range(len(extents) - rank, len(extents)))
            attributes = {'kind': "mean", 'dims': dims, 'keepdim': True}
            return {**attributes, '_plan': {'op': 'reduce', 'operands': [source],
                                            'attributes': attributes}}
        return {'kind': "reduce", '_plan': {'op': 'adaptive_pool', 'operands': [source],
                                            'sizes': sizes}}
    nearest = {f'aten.upsample_nearest{rank}d.{form}': rank for rank in (1, 2, 3)
               for form in ('vec', 'default')}
    if _is_op(node, tuple(nearest)):
        # The source index of each output index along each spatial axis is the one PyTorch's
        # nearest_idx computes, read by running the operator on an iota of that axis.
        source = node.args[0]
        name = _target_name(node.target)
        rank = nearest[name]
        extents = _shape_from_arg(source)
        out = _node_shape(node)
        if _node_dtype(source) not in _FLOATING or len(extents) != rank + 2:
            raise NotImplementedError(f'{name} needs a floating operand of rank {rank + 2}')
        if name.endswith('.vec'):
            scales = _argument(node, 2, 'scale_factors')
        else:
            keywords = ['scales'] if rank == 1 else [f'scales_{axis}' for axis in 'dhw'[3 - rank:]]
            scales = [_argument(node, 2 + a, keyword) for a, keyword in enumerate(keywords)]
        indices = []
        for axis in range(rank):
            # upsample_nearest1d.default with the output extent and the axis's own optional
            # scale, which nearest_idx reads in binary32 when it is given.
            extent = extents[2 + axis]
            if extent >= 2 ** 53:
                raise NotImplementedError(f'{name}: an axis of 2 ** 53 or more entries')
            iota = torch.arange(extent, dtype=torch.float64).reshape(1, 1, extent)
            scale = None if scales is None or scales[axis] is None else float(scales[axis])
            picked = torch.ops.aten.upsample_nearest1d.default(iota, [out[2 + axis]], scale)
            indices.append(picked.reshape(-1).long().tolist())
        if [len(index) for index in indices] != out[2:]:
            raise NotImplementedError(f'{name}: the per-axis indices do not match the output')
        return {'kind': "gather", '_plan': {'op': 'upsample_nearest', 'operands': [source],
                                            'indices': indices}}
    if _is_op(node, ('aten.view_as_real.default',)):
        producer = node.args[0]
        if not (isinstance(producer, torch.fx.Node) and
                _is_op(producer, ('aten._fft_r2c.default', 'aten.fft_rfft.default'))):
            raise NotImplementedError('view_as_real of a complex value other than a real '
                                      'Fourier transform')
        forward = _forward_transform(producer)
        return {'kind': "custom", '_plan': {'op': 'rfft', 'operands': [forward['source']],
                                          'forward': forward}}
    if _is_op(node, _INVERSE_TRANSFORMS):
        # The complex operand is view_as_complex of a real [..., 2] tensor or a forward
        # transform, and dim is an axis of the complex value. The raw _fft_c2r reads exactly
        # n // 2 + 1 bins, and fft_irfft crops or zero-pads to them.
        producer = node.args[0]
        slices = []
        while isinstance(producer, torch.fx.Node) and _is_op(producer, _COMPLEX_SLICES):
            # A slice of the complex value is the same slice of its paired-real form.
            slices.insert(0, [_argument(producer, a, key) for a, key in
                              enumerate(('self', 'dim', 'start', 'end', 'step'))][1:])
            producer = producer.args[0]
        if slices and not _is_op(producer, ('aten.view_as_complex.default',)):
            raise NotImplementedError('a slice of a Fourier transform is not imported')
        if not isinstance(producer, torch.fx.Node) or not _is_op(
                producer, ('aten.view_as_complex.default', 'aten._fft_r2c.default',
                           'aten.fft_rfft.default')):
            raise NotImplementedError('an inverse real Fourier transform of a complex value other '
                                      'than view_as_complex or a real Fourier transform')
        if _is_op(producer, ('aten.view_as_complex.default',)):
            source, forward = producer.args[0], None
            paired = _shape_from_arg(source)
            if not paired or paired[-1] != 2:
                raise NotImplementedError('view_as_complex needs a trailing axis of 2')
            bins = paired[:-1]
            for index, (dim, start, end, step) in enumerate(slices):
                dim = _normalize_axis(dim or 0, len(bins))
                kept = _slice_positions(bins[dim], start, end, step)
                slices[index] = (dim, kept[0] if kept else 0, kept[-1] + 1 if kept else 0,
                                 1 if step is None else int(step))
                bins[dim] = len(kept)
        else:
            forward = _forward_transform(producer)
            source = forward['source']
            bins = list(_shape_from_arg(source))
            bins[forward['dim']] = forward['n'] // 2 + 1
        if _is_op(node, ('aten._fft_c2r.default',)):
            dims, code, n = node.args[1], node.args[2], int(node.args[3])
            if len(dims) != 1:
                raise NotImplementedError('an inverse real Fourier transform along several axes '
                                          'is not imported')
            dim = _normalize_axis(dims[0], len(bins))
            scale = {2: None, 1: 'sqrt', 0: 'n'}[int(code)]
            if bins[dim] != n // 2 + 1:
                raise NotImplementedError('_fft_c2r of other than n // 2 + 1 bins')
        else:
            n, dim, norm = (_argument(node, 1, 'n'), _argument(node, 2, 'dim'),
                            _argument(node, 3, 'norm'))
            dim = _normalize_axis(-1 if dim is None else dim, len(bins))
            n = 2 * (bins[dim] - 1) if n is None else int(n)
            scale = {None: None, 'backward': None, 'ortho': 'sqrt', 'forward': 'n'}[norm]
        if _node_dtype(source) not in ('float32', 'float64') or n <= 0:
            raise NotImplementedError('irfft needs a binary32 or binary64 operand and n > 0')
        return {'kind': "custom", '_plan': {'op': 'irfft', 'operands': [source], 'dim': dim, 'n': n,
                                          'scale': scale, 'forward': forward,
                                          'slices': slices}}
    if _is_op(node, ('aten.group_norm.default',), (F.group_norm,)):
        # group_norm(input, num_groups, weight, bias, eps), whose core form is native_group_norm.
        eps = _argument(node, 4, 'eps')
        return _moments_kind(node.args[0], 'group', _argument(node, 2, 'weight'),
                             _argument(node, 3, 'bias'), 1e-5 if eps is None else eps, 0,
                             groups=int(node.args[1]))
    if _is_op(node, ('aten.scaled_dot_product_attention.default',),
              (F.scaled_dot_product_attention,)):
        # scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        # is_causal=False, *, scale=None, enable_gqa=False) as its docstring's equivalent code.
        query, key, value = (_argument(node, position, name)
                             for position, name in enumerate(('query', 'key', 'value')))
        mask, dropout = _argument(node, 3, 'attn_mask'), _argument(node, 4, 'dropout_p')
        causal = bool(_argument(node, 5, 'is_causal'))
        scale, gqa = node.kwargs.get('scale'), bool(node.kwargs.get('enable_gqa', False))
        if dropout not in (None, 0, 0.0):
            # The docstring applies torch.dropout(train=True) at every call, in evaluation too.
            raise NotImplementedError('scaled_dot_product_attention with dropout_p > 0 applies '
                                      'torch.dropout(train=True), a fresh random mask, at every '
                                      'call; only dropout_p = 0 is imported')
        if causal and mask is not None:
            raise NotImplementedError('scaled_dot_product_attention: attn_mask and is_causal '
                                      'together are an error in PyTorch')
        dtype = _node_dtype(query)
        if dtype not in _FLOATING or _node_dtype(key) != dtype or _node_dtype(value) != dtype or \
                mask is not None and _node_dtype(mask) not in ('bool', dtype):
            raise NotImplementedError('scaled_dot_product_attention needs floating query, key '
                                      'and value of one dtype and a Boolean mask or one of '
                                      'their dtype')
        q, k, v = (list(_shape_from_arg(x)) for x in (query, key, value))
        if min(len(q), len(k), len(v)) < 2 or k[-1] != q[-1] or k[-2] != v[-2] or q[-1] == 0:
            raise NotImplementedError('scaled_dot_product_attention needs (..., L, E), '
                                      '(..., S, E) and (..., S, Ev) operands with E > 0')
        group = 1
        if gqa:
            if min(len(q), len(k), len(v)) < 3 or k[-3] != v[-3] or k[-3] == 0 or \
                    q[-3] == 0 or q[-3] % k[-3]:
                raise NotImplementedError('enable_gqa needs a positive head count of the query '
                                          'that the key and value head count divides')
            group = q[-3] // k[-3]
            k, v = (x[:-3] + [x[-3] * group] + x[-2:] for x in (k, v))
        batch = list(torch.broadcast_shapes(tuple(q[:-2]), tuple(k[:-2]), tuple(v[:-2])))
        scale = 1 / math.sqrt(q[-1]) if scale is None else float(scale)
        return {'kind': "matmul", '_plan': {'op': 'attention',
                                            'operands': [query, key, value, mask],
                                            'scale': scale, 'causal': causal, 'group': group,
                                            'batch': batch, 'query': q, 'value': v}}
    if _is_op(node, ('aten.cdist.default', 'aten._cdist_forward.default'), (torch.cdist,)):
        # cdist_impl: for p = 2 the matmul expansion where compute_mode asks for it (mode 1, or
        # the default mode 0 with more than 25 rows on a side), else the distance of each pair.
        x1, x2 = node.args[0], node.args[1]
        p = _argument(node, 2, 'p')
        p = 2.0 if p is None else float(p)
        mode = _argument(node, 3, 'compute_mode')
        mode = {None: 0, 'use_mm_for_euclid_dist_if_necessary': 0, 'use_mm_for_euclid_dist': 1,
                'donot_use_mm_for_euclid_dist': 2}.get(mode, mode)
        a, b = _shape_from_arg(x1), _shape_from_arg(x2)
        if len(a) < 2 or len(a) != len(b) or a[-1] != b[-1] or a[:-2] != b[:-2] or \
                _node_dtype(x1) not in ('float32', 'float64') or a[-1] == 0 or \
                _node_dtype(x2) != _node_dtype(x1) or not p >= 0 or mode not in (0, 1, 2):
            raise NotImplementedError('cdist needs binary32 or binary64 operands of one batch '
                                      'shape and a positive width, and p >= 0')
        euclid = p == 2 and (mode == 1 or (mode == 0 and (a[-2] > 25 or b[-2] > 25)))
        return {'kind': "matmul", '_plan': {'op': 'distance', 'operands': [x1, x2], 'p': p,
                                            'euclid': euclid, 'pairs': None}}
    if _is_op(node, ('aten.pdist.default', 'aten._pdist_forward.default'),
              (torch.nn.functional.pdist,)):
        # The distances of the rows i < j, in row-major order of (i, j).
        x = node.args[0]
        p = _argument(node, 1, 'p')
        p = 2.0 if p is None else float(p)
        extents = _shape_from_arg(x)
        if len(extents) != 2 or _node_dtype(x) not in ('float32', 'float64') or not p >= 0 or \
                extents[1] == 0:
            raise NotImplementedError('pdist needs a binary32 or binary64 matrix and p >= 0')
        pairs = [(i, j) for i in range(extents[0]) for j in range(i + 1, extents[0])]
        return {'kind': "matmul", '_plan': {'op': 'distance', 'operands': [x, x], 'p': p,
                                            'euclid': False, 'pairs': pairs}}
    if _is_op(node, ('aten.upsample_bilinear2d.vec', 'aten.upsample_linear1d.vec',
                     'aten.upsample_bilinear2d.default', 'aten.upsample_linear1d.default')):
        # Per resized axis the two source indices and weights compute_indices_weights_linear
        # makes, read by running upsample_linear1d on one-hot rows in the operand's dtype.
        source = node.args[0]
        name = _target_name(node.target)
        rank = 2 if 'bilinear2d' in name else 1
        extents = _shape_from_arg(source)
        out = _node_shape(node)
        if _node_dtype(source) not in _FLOATING or len(extents) != rank + 2:
            raise NotImplementedError(f'{name} needs a floating operand of rank {rank + 2}')
        align = bool(_argument(node, 2, 'align_corners'))
        if name.endswith('.vec'):
            scales = _argument(node, 3, 'scale_factors')
        else:
            keywords = ['scales'] if rank == 1 else ['scales_h', 'scales_w']
            scales = [_argument(node, 3 + a, keyword) for a, keyword in enumerate(keywords)]
        axes = []
        for axis in range(rank):
            # upsample_linear1d.default with the output extent and the axis's own scale. A
            # one-hot row shows each output's weights. A single nonzero weight at i is
            # compute_source_index_and_lambda's pair (i, min(i + 1, n - 1)) with the second
            # weight zero, which is still read (a NaN there reaches the output).
            extent = extents[2 + axis]
            rows = torch.eye(extent, dtype=_torch_dtype(_node_dtype(source))).reshape(
                extent, 1, extent)
            scale = None if scales is None or scales[axis] is None else float(scales[axis])
            weights = torch.ops.aten.upsample_linear1d.default(rows, [out[2 + axis]], align,
                                                               scale)
            weights = weights.reshape(extent, -1)
            picks = []
            for column in range(weights.shape[1]):
                (nonzero,) = torch.nonzero(weights[:, column], as_tuple=True)
                found = nonzero.tolist()
                if len(found) == 1:
                    i0 = found[0]
                    picks.append((i0, min(i0 + 1, extent - 1), weights[i0, column].item(), 0.0))
                elif len(found) == 2:
                    i0, i1 = found
                    picks.append((i0, i1, weights[i0, column].item(),
                                  weights[i1, column].item()))
                else:
                    raise NotImplementedError(f'{name}: an output reads {len(found)} inputs')
            axes.append(picks)
        if [len(picks) for picks in axes] != out[2:]:
            raise NotImplementedError(f'{name}: the per-axis weights do not match the output')
        return {'kind': "gather", '_plan': {'op': 'upsample_linear', 'operands': [source],
                                            'axes': axes}}
    if _is_op(node, ('aten.grid_sampler_2d.default', 'aten.grid_sampler.default')):
        source, grid = node.args[0], node.args[1]
        mode, padding, align = int(node.args[2]), int(node.args[3]), bool(node.args[4])
        extents, points = _shape_from_arg(source), _shape_from_arg(grid)
        if len(extents) != 4 or len(points) != 4 or points[-1] != 2 or \
                points[0] != extents[0] or _node_dtype(source) not in ('float32', 'float64') or \
                _node_dtype(grid) != _node_dtype(source):
            raise NotImplementedError('grid_sampler_2d needs [N, C, H, W] and [N, Ho, Wo, 2] '
                                      'binary32 or binary64 operands')
        if mode not in (0, 1) or padding not in (0, 1):
            raise NotImplementedError('grid_sampler_2d: bilinear and nearest modes with zeros '
                                      'or border padding are imported')
        return {'kind': "gather", '_plan': {'op': 'grid_sample', 'operands': [source, grid],
                                            'mode': mode, 'padding': padding, 'align': align}}
    if _is_op(node, ('aten.rand.default', 'aten.randn.default', 'aten.randperm.default')):
        # TorchLean's generator under a key fixed at capture (Op.randUniform, Op.randn, Op.sort
        # of a uniform draw for a permutation), not PyTorch's stream. The overloads that take a
        # generator are other targets, refused.
        dtype = _node_dtype(node)
        sampler = 'uniform' if _is_op(node, ('aten.rand.default',)) else \
            'normal' if _is_op(node, ('aten.randn.default',)) else 'permutation'
        if sampler == 'uniform' and dtype not in _FLOATING:
            raise NotImplementedError(f'rand of {dtype} is not imported')
        if sampler == 'normal' and dtype not in _FLOATING:
            raise NotImplementedError(f'randn of {dtype} is not imported')
        if sampler == 'permutation' and dtype not in _INDEX_DTYPES:
            raise NotImplementedError(f'randperm of {dtype} is not imported')
        return {'kind': "custom", '_plan': {'op': 'random', 'operands': [], 'sampler': sampler}}
    if _is_op(node, ('aten.dropout.default',), (F.dropout,)):
        p, train = _argument(node, 1, 'p'), _argument(node, 2, 'train', 'training')
        if node.target is F.dropout:
            if _argument(node, 3, 'inplace'):
                raise NotImplementedError('in-place dropout mutation is outside the pure '
                                          'TorchLean graph contract')
            p = 0.5 if p is None else p
            train = True if train is None else train
        return _dropout_kind(node.args[0], p, train, 0)
    if _is_op(node, ('aten.masked_scatter.default',), (), ('masked_scatter',)):
        base, mask, source = node.args[:3]
        if _node_dtype(base) not in _FLOATING or _node_dtype(source) != _node_dtype(base) or \
                _node_dtype(mask) != 'bool':
            raise NotImplementedError('masked_scatter needs floating values of one dtype and a '
                                      'Boolean mask')
        return {'kind': "scatter", '_plan': {'op': 'masked_scatter',
                                             'operands': [base, mask, source]}}
    if _is_op(node, ('aten.nonzero_static.default',), (torch.nonzero_static,)):
        source = node.args[0]
        size = int(_argument(node, 1, 'size'))
        fill = _argument(node, 2, 'fill_value')
        fill = -1 if fill is None else int(fill)
        if not _shape_from_arg(source) or (_node_dtype(source) != 'bool' and
                                            _node_dtype(source) not in _FLOATING) or size < 0:
            raise NotImplementedError('nonzero_static needs a Boolean or floating operand of '
                                      'positive rank and a nonnegative size')
        if size >= 2 ** 53:
            raise NotImplementedError('nonzero_static with a size of 2 ** 53 or more')
        return {'kind': "scatter", '_plan': {'op': 'nonzero_static', 'operands': [source],
                                             'size': size, 'fill': fill}}
    if _is_op(node, ('aten.col2im.default',)):
        source = node.args[0]
        if _node_dtype(source) not in _FLOATING:
            raise NotImplementedError(f'col2im of a {_node_dtype(source)} tensor is not imported')
        pairs = [_spatial_tuple(value, 2, 1) for value in node.args[1:6]]
        return {'kind': "scatter", '_plan': {'op': 'col2im', 'operands': [source],
                                             'output_size': pairs[0], 'kernel': pairs[1],
                                             'dilation': pairs[2], 'padding': pairs[3],
                                             'stride': pairs[4]}}
    if _is_op(node, ('aten.index_select.default', 'aten.gather.default'),
              (torch.index_select, torch.gather), ('index_select', 'gather')) and \
            _index_constant(_argument(node, 2, 'index')) is None:
        # Op.gather on a runtime index. For index_select the selected axis comes first and is
        # then moved to dim. For gather (take_along_dim) the other axes are batch axes, the input
        # cropped to the index's extents there.
        source = _argument(node, 0, 'input', 'self')
        extents = _shape_from_arg(source)
        dim = _normalize_axis(_argument(node, 1, 'dim'), len(extents))
        index = _index_operand(_argument(node, 2, 'index'), _target_name(node.target))
        positions = _shape_from_arg(index)
        selecting = _is_op(node, ('aten.index_select.default',), (torch.index_select,),
                           ('index_select',))
        if _node_dtype(source) not in _FLOATING:
            raise NotImplementedError(f'gather of a {_node_dtype(source)} tensor is not imported')
        if selecting and len(positions) != 1:
            raise NotImplementedError('index_select needs a one-dimensional index')
        if not selecting and (len(positions) != len(extents) or any(
                positions[a] > extents[a] for a in range(len(extents)) if a != dim)):
            raise NotImplementedError('gather needs an index of the input rank, no longer than '
                                      'the input along the other axes')
        return {'kind': "gather", '_plan': {'op': 'gather_at', 'operands': [source, index],
                                            'dim': dim, 'select': selecting}}
    if _is_op(node, ('aten.index_select.default', 'aten.gather.default'),
              (torch.index_select, torch.gather), ('index_select', 'gather')):
        source = _argument(node, 0, 'input', 'self')
        extents = _shape_from_arg(source)
        dim = _normalize_axis(_argument(node, 1, 'dim'), len(extents))
        index = _index_value(_argument(node, 2, 'index'))
        if _is_op(node, ('aten.index_select.default',), (torch.index_select,),
                  ('index_select',)):
            if index.dim() != 1:
                raise NotImplementedError('index_select needs a one-dimensional index')
            view = [-1 if axis == dim else 1 for axis in range(len(extents))]
            index = index.reshape(view).expand(_node_shape(node))
        coords = _coordinates(index, dim, extents)
        return {'kind': "gather", 'dims': list(range(len(extents))), 'batch_dims': [],
                'indices': {'shape': list(index.shape),
                            'values': coords.reshape(-1).tolist()},
                '_operands': [source]}
    if _is_op(node, ('aten.scatter_reduce.two',), (torch.scatter_reduce,), ('scatter_reduce',)):
        # Without include_self the base entry at a written target takes no part (a sum or a mean
        # starts from zero there). The mean divides by the number of values folded into each
        # entry, one where there are none.
        base, update = _argument(node, 0, 'input', 'self'), _argument(node, 3, 'src')
        extents = _shape_from_arg(base)
        dim = _normalize_axis(_argument(node, 1, 'dim'), len(extents))
        index = _argument(node, 2, 'index')
        reduce = _argument(node, 4, 'reduce')
        include_self = _argument(node, 5, 'include_self')
        include_self = True if include_self is None else bool(include_self)
        if reduce not in ('sum', 'mean', 'prod', 'amax', 'amin'):
            raise NotImplementedError(f'scatter_reduce with reduce={reduce!r}')
        if _node_dtype(base) not in _FLOATING:
            raise NotImplementedError(f'scatter_reduce of a {_node_dtype(base)} tensor')
        if reduce in ('prod', 'amax', 'amin'):
            if reduce == 'prod' and dim == 0 and isinstance(index, torch.fx.Node) and \
                    _is_op(index, ('aten.expand.default',)):
                raise NotImplementedError('scatter_reduce prod along dim 0 of an expanded index: '
                                          'ATen takes its expanded-index path, whose product '
                                          'order is outside the imported schedule')
            return {'kind': "scatter", '_plan': {
                'op': 'scatter_fold', 'operands': [base, index, update], 'dim': dim,
                'reduce': reduce, 'include_self': include_self}}
        return {'kind': "scatter", '_plan': {
            'op': 'scatter_mean', 'operands': [base, index, update], 'dim': dim,
            'mean': reduce == 'mean', 'include_self': include_self}}
    if _is_op(node, ('aten.scatter.src', 'aten.scatter.value', 'aten.scatter_add.default'),
              (torch.scatter, torch.scatter_add), ('scatter', 'scatter_add')) and \
            _top_k_mask(node) is None and _index_constant(_argument(node, 2, 'index')) is None:
        # Op.scatter on a runtime index, at complete coordinates built in the IR.
        base = _argument(node, 0, 'input', 'self')
        extents = _shape_from_arg(base)
        dim = _normalize_axis(_argument(node, 1, 'dim'), len(extents))
        index = _index_operand(_argument(node, 2, 'index'), _target_name(node.target))
        update = _argument(node, 3, 'src', 'value')
        if node.kwargs.get('reduce') is not None:
            raise NotImplementedError('scatter with a reduce argument is not imported')
        if _node_dtype(base) not in _FLOATING:
            raise NotImplementedError(f'scatter into a {_node_dtype(base)} tensor is not imported')
        positions = _shape_from_arg(index)
        if len(positions) != len(extents) or any(
                positions[a] > extents[a] for a in range(len(extents)) if a != dim):
            raise NotImplementedError('scatter needs an index of the input rank, no longer than '
                                      'the input along the other axes')
        if isinstance(update, torch.fx.Node) and (
                _node_dtype(update) != _node_dtype(base) or
                any(size < extent for size, extent in zip(_shape_from_arg(update), positions))):
            raise NotImplementedError('scatter needs a source of the destination dtype, no '
                                      'shorter than the index')
        reduce = 'sum' if _is_op(node, ('aten.scatter_add.default',), (torch.scatter_add,),
                                 ('scatter_add',)) else 'replace'
        return {'kind': "scatter", '_plan': {'op': 'scatter_at', 'operands': [base, index, update],
                                             'dim': dim, 'reduce': reduce}}
    if _is_op(node, ('aten.scatter.src', 'aten.scatter.value', 'aten.scatter_add.default'),
              (torch.scatter, torch.scatter_add), ('scatter', 'scatter_add')) and \
            _top_k_mask(node) is None:
        base = _argument(node, 0, 'input', 'self')
        extents = _shape_from_arg(base)
        dim = _normalize_axis(_argument(node, 1, 'dim'), len(extents))
        index = _index_value(_argument(node, 2, 'index'))
        update = _argument(node, 3, 'src', 'value')
        if node.kwargs.get('reduce') is not None:
            raise NotImplementedError('scatter with a reduce argument is not imported')
        if isinstance(update, torch.fx.Node) and _shape_from_arg(update) != list(index.shape):
            raise NotImplementedError('scatter reads only part of a larger source')
        coords = _coordinates(index, dim, extents)
        if len({tuple(c) for c in coords.reshape(-1, len(extents)).tolist()}) != index.numel():
            # PyTorch leaves the winner of a replacement, and the order of a sum, unspecified.
            raise NotImplementedError('scatter with a repeated target is not imported')
        reduce = 'sum' if _is_op(node, ('aten.scatter_add.default',),
                                         (torch.scatter_add,), ('scatter_add',)) \
            else 'replace'
        return {'kind': "scatter", '_plan': {
            'op': 'scatter', 'operands': [base, update], 'reduce': reduce,
            'dims': list(range(len(extents))),
            'indices': {'shape': list(index.shape), 'values': coords.reshape(-1).tolist()}}}
    if _is_op(node, ('aten.diagonal.default', 'aten.diagonal_copy.default'), (torch.diagonal,),
              ('diagonal',)):
        source = _argument(node, 0, 'input', 'self')
        rank = len(_shape_from_arg(source))
        offset = int(_argument(node, 1, 'offset') or 0)
        dim1 = _normalize_axis(_argument(node, 2, 'dim1') or 0, rank)
        dim2 = _argument(node, 3, 'dim2')
        dim2 = _normalize_axis(1 if dim2 is None else dim2, rank)
        if dim1 == dim2:
            raise NotImplementedError('diagonal needs two distinct axes')
        return {'kind': "slice", '_plan': {'op': 'diagonal', 'operands': [source],
                                           'offset': offset, 'dim1': dim1, 'dim2': dim2}}
    if _is_op(node, ('aten.select.int',), (torch.select,), ('select',)):
        source = _shape_from_arg(node.args[0])
        dim = _normalize_axis(_argument(node, 1, 'dim'), len(source))
        index = int(_argument(node, 2, 'index'))
        index = index + source[dim] if index < 0 else index
        if not 0 <= index < source[dim]:
            raise NotImplementedError(f'select index {index} is outside extent {source[dim]}')
        return {'kind': "slice", '_plan': {'op': 'select', 'operands': [node.args[0]],
                                           'dim': dim, 'index': index}}
    if _is_op(node, ('aten.addmm.default',), (torch.addmm,), ('addmm',)):
        bias, left, right = node.args[:3]
        if _argument(node, 3, 'beta') not in (None, 1) or \
                _argument(node, 4, 'alpha') not in (None, 1):
            raise NotImplementedError('addmm with beta or alpha other than 1 is not imported')
        if len({json.dumps(_node_dtype(value)) for value in (bias, left, right, node)}) != 1:
            raise NotImplementedError('addmm operands must share one dtype')
        return {'kind': "matmul", '_plan': {'op': 'addmm', 'operands': [bias, left, right]}}
    if _is_op(node, ('aten.repeat.default',), (), ('repeat',)):
        repeats = list(node.args[1:]) if len(node.args) > 2 else list(node.args[1])
        return {'kind': "reshape", '_plan': {'op': 'repeat', 'operands': [node.args[0]],
                                             'repeats': [int(r) for r in repeats]}}
    pads = {f'aten.{mode}_pad{d}d.default': mode for mode in ('reflection', 'replication')
            for d in (1, 2, 3)}
    if _is_op(node, tuple(pads)):
        mode = pads[_target_name(node.target)]
        return {'kind': "cat", '_plan': {'op': 'reflect' if mode == 'reflection' else 'replicate',
                                         'operands': [node.args[0]],
                                         'padding': [int(p) for p in node.args[1]]}}
    if _is_op(node, ('aten.convolution.default',)):
        source, weight_arg, bias_arg, stride, padding, dilation, transposed, output_padding, \
            groups = node.args[:9]
        if transposed:
            raise NotImplementedError('transposed convolution waits for the general convTranspose')
        weight = _resolve_tensor(weight_arg)
        rank = len(_shape_of(weight)) - 2
        input_shape = _shape_from_arg(source)
        if len(input_shape) != rank + 2:
            raise NotImplementedError(
                f'convolution expected rank {rank + 2}, got {len(input_shape)}')
        padding = _spatial_tuple(padding, rank, 0)
        return {'kind': "conv", 'spatial_rank': rank,
                'kernel': [int(v) for v in _shape_of(weight)[2:]],
                'stride': _spatial_tuple(stride, rank, 1), 'padding': padding,
                'padding_after': padding, 'dilation': _spatial_tuple(dilation, rank, 1),
                'groups': int(groups), 'channel_axis': 1, 'input_spatial': input_shape[2:],
                **_convolution_payload(weight, bias_arg, groups), '_operands': [source]}
    return None
