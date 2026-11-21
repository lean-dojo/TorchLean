"""Structural operators of a PyTorch graph: indexing, scatter, statistics, RNG and buffers.

`_structure_kind` plans one call, and the components of tuple-valued operators are read through
`_projection`. Beside `_fx._TENSOR_VALUES`, the capture's state lives here: the folded index
constants (`_FOLDED`), the exported signature's buffers (`_BUFFERS`, `_OUTPUT_TARGETS`,
`_BUFFER_UPDATES`) and the dropout masks (`_DROPOUT_MASKS`), all emptied by `_graph._reset`.
"""

from __future__ import annotations

import json
import math
import operator

import torch
import torch.nn.functional as F

from ._fx import (_FLOATING, _INDEX_DTYPES, _INT64, _TENSOR_VALUES, _broadcast_kind, _constant,
                  _convolution_payload, _held_shape, _is_getattr, _is_getitem, _is_op,
                  _node_dtype, _node_shape, _normalization, _normalize_axis, _op_table,
                  _reshape_kind, _shape_from_arg, _spatial_tuple, _target_name, _top_k_mask,
                  _torch_dtype, _zeros_like)
from ._artifact import _LEFT_FOLD, _element, _integer
from ._pointwise import (_append_node, _argument, _broadcast_to, _cast_to, _round_once,
                         _scalar_constant)


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
    'aten.frexp.Tensor': (0, 1),
    'aten.native_group_norm.default': (0, 1, 2),
    'aten._native_batch_norm_legit.no_stats': (0, 1, 2),
    'aten._native_batch_norm_legit.default': (0, 1, 2),
    'aten._native_batch_norm_legit_functional.default': (0, 1, 2, 3, 4),
    'aten.native_dropout.default': (0, 1),
    'aten._embedding_bag.default': (0,),
    'aten.embedding_bag.padding_idx': (0,),
    'aten.split_with_sizes.default': None,
    'aten.split.Tensor': None,
    'aten.chunk.default': None,
    'aten.unbind.int': None,
}
# The tuple-valued operators FX records as a Python callable or a Tensor method, with the field
# names of PyTorch's result, which FX projects by `getattr`: (callables, methods, fields).
_FX_COMPONENTS = {
    'aten.frexp.Tensor': ((torch.frexp,), ('frexp',), ('mantissa', 'exponent')),
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
class _UninitializedRead(NotImplementedError):
    """The captured program reads an allocation before it has a defined value."""


# Uninitialized allocations accepted only as destinations for full writes.
_EMPTY = ('aten.empty.memory_format', 'aten.empty_strided.default', 'aten.empty_permuted.default',
          'aten.empty_like.default', 'aten.new_empty.default', 'aten.new_empty_strided.default')
_CREATION = ('aten.full.default', 'aten.ones.default', 'aten.zeros.default',
             'aten.scalar_tensor.default', 'aten.arange.default', 'aten.arange.start',
             'aten.arange.start_step', 'aten.linspace.default', 'aten.logspace.default')
# zero_ and fill_ write one value over a fresh destination (_fresh_destination).
_IN_PLACE_FILLS = ('aten.zero_.default', 'aten.fill_.Scalar')
_SHAPED_CREATION = ('aten.full_like.default', 'aten.ones_like.default',
                    'aten.fill.Scalar', 'aten.new_full.default', 'aten.new_ones.default',
                    'aten.new_zeros.default', *_IN_PLACE_FILLS)
_ALIASES = ('aten.clone.default', 'aten.alias.default', 'aten.lift_fresh_copy.default')

# The default graph's tensor literal is lift_fresh_copy then detach_. As detach_ writes autograd
# state, not data, a detach_ of such a literal, folded or not, and of an integer or Boolean value
# is its operand's value; of another floating value it is Op.detach (_structure_kind), as is the
# out-of-place detach of a floating value, which a decomposed program writes as alias
# (_decomposed_detach).
_DETACH = ('aten.detach_.default',)
_DETACH_COPY = ('aten.detach.default',)


def _decomposed_detach(node):
    # Whether `node` is the detach of a decomposed program: run_decompositions() writes it as
    # alias and records detach as the node's original_aten.
    return _is_op(node, ('aten.alias.default',)) and \
        str(node.meta.get('original_aten')) == 'aten.detach.default'


_STRUCTURE_PLANS = ('select', 'addmm', 'repeat', 'pad', 'reflect', 'replicate', 'reduce', 'var',
                    'scan', 'extremum', 'leaky_relu', 'diagonal', 'scatter', 'normalization',
                    'registered', 'gather_at', 'scatter_at', 'embedding', 'index', 'index_put',
                    'boolean_reduce', 'col2im', 'scatter_mean',
                    'adaptive_pool', 'upsample_nearest', 'rfft', 'irfft', 'complex', 'moments',
                    'distance', 'upsample_linear', 'grid_sample', 'random', 'dropout',
                    'masked_scatter', 'nonzero_static', 'embedding_bag', 'topk', 'filled',
                    'vector_norm', 'pool_indices', 'scatter_fold', 'buffer_update', 'attention',
                    'householder_product', 'ormqr', 'stack', 'interleave', 'roll', 'unfold',
                    'triangle', 'conv_transpose')
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
            not _is_op(node, _EMPTY) and
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


def _advanced_indices(nodes, node_to_id, values, axes, extents, positions, wrap=True):
    # The index tuples of PyTorch's advanced indexing along `axes`: each index broadcast to
    # `positions`, a negative entry i read as i + extent, stacked when there are several. A
    # constant index is wrapped at capture, and refused outside [-extent, extent). Without
    # `wrap` (index_add and index_copy, which reject a negative index) a negative entry is
    # outside its axis.
    components = []
    for value, axis in zip(values, axes):
        extent = extents[axis]
        constant = _index_constant(value)
        if constant is not None:
            constant = constant.long()
            wrapped = torch.where(constant < 0, constant + extent, constant) if wrap else constant
            if ((wrapped < 0) | (wrapped >= extent)).any():
                raise NotImplementedError(f'an index is outside an axis of extent {extent}')
            index = _append_node(nodes, {'parents': [], **_constant(
                wrapped.broadcast_to(positions).contiguous())})
        else:
            index = node_to_id.get(value)
            if index is None: raise NotImplementedError('an index operand has no tensor lowering')
            index = _wrapped_index(nodes, index, extent) if wrap else \
                _cast_to(nodes, index, _INT64)
            index = _broadcast_to(nodes, index, positions)
        components.append(index)
    return _index_tuples(nodes, components, positions)


def _check_moved(name, dtype, accumulate=False):
    # A runtime gather and a write that replaces move their entries without reading them, at the
    # element of Op.gather and Op.scatter: a floating, integer or Boolean type (a float8 value is
    # `_graph._check_float8`'s). A write that sums reads them and is imported at a floating type.
    if dtype in _FLOATING or not accumulate and (dtype == 'bool' or _integer(dtype) is not None):
        return
    raise NotImplementedError(f'{name} of a {dtype} tensor is not imported')


def _advanced_axes(node, source, indices, accumulate=False):
    # The indexed axes and the broadcast index shape of index.Tensor and index_put. A Boolean or
    # uint8 mask index has a data-dependent result shape and is refused.
    name = _target_name(node.target)
    extents = _shape_from_arg(source)
    _check_moved(name, _node_dtype(source), accumulate)
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


def _storage_words(shape, stride, offset):
    # The storage word of every entry of a tensor of this shape, stride and storage offset.
    grids = torch.meshgrid(*[torch.arange(n) for n in shape], indexing='ij') if shape else ()
    return sum((grid * step for grid, step in zip(grids, stride)),
               torch.full(list(shape), offset, dtype=torch.long))


# Complex values. The IR has no complex dtype: a complex64 or complex128 value is held in its
# paired-real form, its real and imaginary parts on a last axis of 2 in binary32 or binary64, as
# Op.rfft gives a transform and view_as_real reads one. `_complex_kind` lowers the operators below
# on that form, and any other reader of a complex value is refused (`_complex_reader`).
_INVERSE_TRANSFORMS = ('aten._fft_c2r.default', 'aten.fft_irfft.default')
_COMPLEX_RESHAPES = ('aten.unsqueeze.default', 'aten.view.default', 'aten.reshape.default')
_COMPLEX_OPERATORS = ('aten._fft_r2c.default', 'aten.fft_rfft.default',
                      'aten.view_as_complex.default', 'aten.slice.Tensor', 'aten.select.int',
                      'aten.div.Tensor', 'aten.mul.Tensor') + _COMPLEX_RESHAPES
_COMPLEX_READERS = ('aten.view_as_real.default', 'aten.real.default',
                    'aten.imag.default') + _INVERSE_TRANSFORMS


def _complex_value(value):
    meta = value.meta.get('val', value.meta.get('tensor_meta')) if hasattr(value, 'meta') else None
    return getattr(getattr(meta, 'dtype', None), 'is_complex', False)


def _paired_dtype(value):
    # The format of the parts of a complex64 or complex128 value, else None.
    meta = value.meta.get('val', value.meta.get('tensor_meta')) if hasattr(value, 'meta') else None
    return {torch.complex64: 'float32', torch.complex128: 'float64'}.get(
        getattr(meta, 'dtype', None))


def _complex_reader(user, value):
    # Whether the capture lowers `user`'s reading of the complex `value`: an operator of
    # `_COMPLEX_OPERATORS` with a complex result, or view_as_real, real, imag or an inverse
    # transform of `value`.
    if _complex_value(user):
        return user.op == 'call_function' and _is_op(user, _COMPLEX_OPERATORS)
    return _is_op(user, _COMPLEX_READERS) and user.args[0] is value


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


def _selected(node):
    # The axis and the nonnegative position that the select `node` keeps.
    source = _shape_from_arg(node.args[0])
    dim = _normalize_axis(_argument(node, 1, 'dim'), len(source))
    index = int(_argument(node, 2, 'index'))
    index = index + source[dim] if index < 0 else index
    if not 0 <= index < source[dim]:
        raise NotImplementedError(f'select index {index} is outside extent {source[dim]}')
    return dim, index


def _complex_kind(node):
    # The lowering on the paired-real form (above) of a node with a complex result or operand, or
    # None for another node:
    # - rfft and _fft_r2c are Op.rfft, scaled for the normalization, and view_as_complex of a real
    #   [..., 2] tensor and view_as_real are that tensor;
    # - slices and selects along a complex axis, unsqueeze, view and reshape are the same of the
    #   parts, the pair axis kept last;
    # - real and imag read one part, and irfft and _fft_c2r are Op.irfft of the parts;
    # - a quotient by a nonzero finite number d is c10::complex's operator/= at (d, 0)
    #   (torch/headeronly/util/complex.h:246-282 of PyTorch 2.13): rat = 0 / d and
    #   scl = 1 / (d + 0 * rat) in the parts' format, then ((a + b*rat) * scl, (b - a*rat) * scl);
    # - a product of two complex tensors is (ac - bd, ad + bc), rounded per operation. PyTorch's
    #   AVX512 kernel fuses the multiply-adds outside whole vectors, so the IR's
    #   product agrees with it within a tolerance, as the transforms do.
    name = _target_name(node.target) if node.op == 'call_function' else str(node.target)
    operand = node.args[0] if node.args else None
    if not _complex_value(node):
        if not _complex_value(operand):
            return None
        if _paired_dtype(operand) is None:
            raise NotImplementedError(f'{name} of a {_complex_dtype(operand)} value: only '
                                      'complex64 and complex128 are imported')
        if _is_op(node, ('aten.view_as_real.default',)):
            return {'_alias': operand}
        if _is_op(node, ('aten.real.default', 'aten.imag.default')):
            part = int(_is_op(node, ('aten.imag.default',)))
            return {'kind': "custom", '_plan': {'op': 'complex', 'form': 'part', 'part': part,
                                                'operands': [operand]}}
        if not _is_op(node, _INVERSE_TRANSFORMS):
            raise NotImplementedError(f'{name} of a complex value is not imported')
        # dim is an axis of the complex value. The raw _fft_c2r reads exactly n // 2 + 1 bins,
        # and fft_irfft crops or zero-pads to them.
        bins = _shape_from_arg(operand)
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
        if n <= 0:
            raise NotImplementedError('irfft needs n > 0')
        return {'kind': "custom", '_plan': {'op': 'irfft', 'operands': [operand], 'dim': dim,
                                          'n': n, 'scale': scale}}
    part = _paired_dtype(node)
    if part is None:
        raise NotImplementedError(f'{name}: a {_complex_dtype(node)} value is not imported, only '
                                  'complex64 and complex128')
    if node.op != 'call_function' or not _is_op(node, _COMPLEX_OPERATORS):
        raise NotImplementedError(f'{name} with a complex result is not imported')
    if _is_op(node, ('aten._fft_r2c.default', 'aten.fft_rfft.default')):
        forward = _forward_transform(node)
        return {'kind': "custom", '_plan': {'op': 'rfft', 'operands': [forward['source']],
                                          'forward': forward}}
    if _is_op(node, ('aten.view_as_complex.default',)):
        return {'_alias': operand}
    if any(_complex_value(arg) and _paired_dtype(arg) != part for arg in node.args):
        raise NotImplementedError(f'{name} of complex operands of different types is not '
                                  'imported')
    if _is_op(node, _COMPLEX_RESHAPES):
        return {'kind': "custom", '_plan': {'op': 'complex', 'form': 'reshape',
                                            'operands': [operand]}}
    if _is_op(node, ('aten.slice.Tensor',)):
        extents = _shape_from_arg(operand)
        dim, start, end, step = [_argument(node, a, key) for a, key in
                                 enumerate(('self', 'dim', 'start', 'end', 'step'))][1:]
        dim = _normalize_axis(dim or 0, len(extents))
        kept = _slice_positions(extents[dim], start, end, step)
        return {'kind': "custom", '_plan': {'op': 'complex', 'form': 'slice', 'dim': dim,
                                            'start': kept[0] if kept else 0,
                                            'stop': kept[-1] + 1 if kept else 0,
                                            'step': 1 if step is None else int(step),
                                            'operands': [operand]}}
    if _is_op(node, ('aten.select.int',)):
        dim, index = _selected(node)
        return {'kind': "custom", '_plan': {'op': 'complex', 'form': 'select', 'dim': dim,
                                            'index': index, 'operands': [operand]}}
    other = _argument(node, 1, 'other')
    if _is_op(node, ('aten.mul.Tensor',)):
        if not (_complex_value(operand) and _complex_value(other)):
            raise NotImplementedError('mul of a complex and a real value is not imported')
        return {'kind': "custom", '_plan': {'op': 'complex', 'form': 'mul',
                                            'operands': [operand, other]}}
    if (not _complex_value(operand) or isinstance(other, bool) or
            not isinstance(other, (int, float)) or node.kwargs.get('rounding_mode') is not None):
        raise NotImplementedError('div of a complex value is imported only by a real number')
    divisor = torch.tensor(other, dtype=_torch_dtype(part))
    if divisor == 0 or not torch.isfinite(divisor):
        raise NotImplementedError('a complex quotient by zero or a non-finite number is not '
                                  'imported')
    zero, one = torch.zeros((), dtype=divisor.dtype), torch.ones((), dtype=divisor.dtype)
    rat = zero / divisor
    scale = one / (divisor + zero * rat)
    return {'kind': "custom", '_plan': {'op': 'complex', 'form': 'div', 'rat': rat.item(),
                                        'scale': scale.item(), 'operands': [operand]}}


def _complex_dtype(value):
    meta = value.meta.get('val', value.meta.get('tensor_meta'))
    return str(getattr(meta, 'dtype', None)).removeprefix('torch.')


def _component_name(node):
    # The ATen name of the tuple-valued operator that `node` calls and the IR imports by
    # component, or None: its overload under torch.export, its callable or method under FX.
    if node.op == 'call_function' and _is_op(node, tuple(_COMPONENTS)):
        return _target_name(node.target)
    for name, (functions, methods, _) in _FX_COMPONENTS.items():
        if _is_op(node, functions=functions, methods=methods): return name
    return None


def _component_producer(node):
    return _component_name(node) is not None


def _arity(producer):
    # The number of components of the producer's result: its field names under FX, else the
    # length of the value torch.export records, or None where the graph records neither.
    name = _component_name(producer)
    if name in _FX_COMPONENTS: return len(_FX_COMPONENTS[name][2])
    value = producer.meta.get('val')
    return len(value) if isinstance(value, (tuple, list)) else None


def _projection(node):
    # (producer, index) when `node` projects one component of a tuple-valued operator the IR
    # imports by component: by position, or under FX by the field name of PyTorch's result. A
    # negative position counts from the end of the result, as in Python.
    if not (_is_getitem(node) or _is_getattr(node)) or len(node.args) != 2: return None
    producer, index = node.args
    if not isinstance(producer, torch.fx.Node) or not _component_producer(producer): return None
    if _is_getattr(node):
        fields = _FX_COMPONENTS.get(_component_name(producer), ((), (), ()))[2]
        if index not in fields: return None
        index = fields.index(index)
    if type(index) is not int: return None
    arity = _arity(producer)
    return producer, (index + arity if arity is not None and -arity <= index < 0 else index)


def _project_results(graph):
    # FX keeps a tuple-valued result that the program returns whole as one node. The output reads
    # its components instead, as torch.export writes it, so every use of the producer projects.
    output = next(node for node in graph.nodes if node.op == 'output')
    for node in list(graph.nodes):
        name = _component_name(node)
        if name not in _FX_COMPONENTS or output not in node.users: continue
        with graph.inserting_before(output):
            parts = tuple(graph.call_function(operator.getitem, (node, index))
                          for index in range(len(_FX_COMPONENTS[name][2])))
        output.args = torch.fx.node.map_arg(output.args, lambda x: parts if x is node else x)


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


def _running_kind(producer, source, eps):
    # Batch norm over channel axis 1 with the running statistics of `producer`, whose arguments
    # start (input, weight, bias, running_mean, running_var): held, or graph inputs.
    shape = _shape_from_arg(source)
    if len(shape) < 2: raise NotImplementedError('batch norm needs a channel axis 1')
    return _normalization("batch_norm", source, eps, {'dim': 1}, [int(shape[1])],
                          gamma=_argument(producer, 1, 'weight'),
                          beta=_argument(producer, 2, 'bias'),
                          mean=_argument(producer, 3, 'running_mean'),
                          var=_argument(producer, 4, 'running_var'))


def _moments_kind(source, mode, weight, bias, eps, output, groups=1, running=None,
                  record=False):
    # A group norm (statistics over each group of channels and the spatial axes) or a training
    # batch norm (over every axis but channel axis 1), output 0 the normalized values, 1 the
    # means and 2 the reciprocal standard deviations. With running = (running_mean,
    # running_var, momentum) outputs 3 and 4 are the updated running statistics, and `record`
    # keeps them as the updates of those buffers (_BUFFER_UPDATES) when output 0 is emitted.
    # The root of var + eps is taken as it stands, as in PyTorch: NaN where the sum is negative.
    extents = _shape_from_arg(source)
    if len(extents) < 2 or _node_dtype(source) not in _FLOATING:
        raise NotImplementedError(f'{mode} norm needs a floating operand with a channel axis 1')
    if extents[1] % groups:
        raise NotImplementedError('group norm needs the groups to divide the channels')
    eps = float(eps)
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


def _program_outputs(refs, convert, example_tensors):
    # The program's outputs: the graph's user outputs, then the value of each updated buffer in
    # the order of the buffer placeholders, from a core graph's BUFFER_MUTATION outputs or a
    # default graph's batch_norm and add_ rules. The IR holds no mutation. PyTorch copies a
    # BUFFER_MUTATION output into its buffer, so `convert(ref, name)` converts it to the dtype of
    # the buffer `name`.
    if all(target is None for target in _OUTPUT_TARGETS) and not _BUFFER_UPDATES:
        return refs
    if _OUTPUT_TARGETS and len(_OUTPUT_TARGETS) != len(refs):
        raise NotImplementedError('the graph outputs do not match the exported signature')
    # An updated buffer must not share storage with another captured tensor (another updated
    # buffer, or a buffer, parameter or constant read as captured): eager execution would see
    # the update there. Disjoint views of one storage are refused too.
    def storage(value):
        if isinstance(value, torch.Tensor) and value.numel():
            return value.device, value.untyped_storage().data_ptr()
        return None
    input_storage = {storage(tensor) for tensor in example_tensors}
    for name in _updated_buffers():
        mine = storage(_TENSOR_VALUES.get(name))
        if mine is not None and mine in input_storage:
            raise NotImplementedError(f'the updated buffer {name} shares storage with an input')
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
            updates[target] = convert(ref, target)
    return user + [updates[name] for name in _BUFFERS if name in updates]


_DROPOUT_MASKS = {}
# The ATen operators whose rules draw from TorchLean's generator under their node's key: rand,
# randn and randperm (`_artifact._random_key`), and dropout and native_dropout where their train
# flag (argument 2) is set, by Op.bernoulliMask.
_SAMPLERS = ('aten.rand.default', 'aten.randn.default', 'aten.randperm.default')
_DROPOUTS = ('aten.dropout.default', 'aten.native_dropout.default')


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
    name = _component_name(producer)
    allowed = _COMPONENTS[name]
    if allowed is not None and index not in allowed:
        # An operator without an owner entry imports all its components, so there the index is
        # outside the result.
        owners = _COMPONENT_OWNERS.get(name)
        raise NotImplementedError(f'{name}: {owners} are not imported' if owners else
                                  f'{name}: component {index} is out of range')
    source = _argument(producer, 0, 'self', 'input')
    if name == 'aten.unbind.int':
        # unbind returns every slice along dim without that axis (the PyTorch 2.13 docstring of
        # torch.unbind): component i is select(dim, i).
        extents = _shape_from_arg(source)
        dim = _normalize_axis(_argument(producer, 1, 'dim') or 0, len(extents))
        if not 0 <= index < extents[dim]:
            raise NotImplementedError(f'{name}: component {index} is out of range')
        return {'kind': "slice", '_plan': {'op': 'select', 'operands': [source], 'dim': dim,
                                           'index': index}}
    if allowed is None:
        dim, windows = _split_windows(producer)
        if not 0 <= index < len(windows):
            raise NotImplementedError(f'{name}: component {index} is out of range')
        start, length = windows[index]
        return {'kind': "slice", 'dims': [dim], 'starts': [start], 'stops': [start + length],
                'steps': [1], '_operands': [source]}
    if name in _ORDER_COMPONENTS:
        return _order_kind(name, producer, source, index)
    if name == 'aten.frexp.Tensor':
        # C's frexp, Op.frexp: the mantissa (component 0) and the int32 exponent (component 1).
        shape, dtype = _shape_from_arg(source), _node_dtype(source)
        if dtype not in _FLOATING:
            raise NotImplementedError(f'frexp of a {dtype} tensor is not imported')
        return _registered_kind('frexp', [{'shape': shape}], [source], shape, shape, index)
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
    if name == 'aten._native_batch_norm_legit.default':
        # (input, weight, bias, running_mean, running_var, training, momentum, eps). Outside
        # training it is batch norm with the running statistics, and outputs 1 and 2 are empty.
        # In training the batch statistics normalize, and the running statistics, updated in
        # place, must be buffers that only this node reads, as for aten.batch_norm
        # (_emit_structure): their updates are kept when output 0 is emitted.
        if _argument(producer, 5, 'training') is not True:
            if index != 0:
                raise NotImplementedError(f'{name}: the empty saved statistics outputs outside '
                                          'training are not imported')
            return _running_kind(producer, source, _argument(producer, 7, 'eps'))
        running = (_argument(producer, 3, 'running_mean'), _argument(producer, 4, 'running_var'))
        if not all(isinstance(value, torch.fx.Node) and value.op == 'placeholder' and
                   value.name in _BUFFERS and set(value.users) == {producer}
                   for value in running) or running[0] is running[1]:
            raise NotImplementedError(f'{name} in training updates its running statistics in '
                                      'place: they must be two buffers that only it reads')
        if not any(_projection(user) == (producer, 0) for user in producer.users):
            raise NotImplementedError(f'{name} in training whose normalized output is unused')
        return _moments_kind(source, 'batch', producer.args[1], producer.args[2],
                             _argument(producer, 7, 'eps'), index,
                             running=(*running, _argument(producer, 6, 'momentum')),
                             record=index == 0)
    if name == 'aten._native_batch_norm_legit_functional.default':
        # The core graph's batch norm (weight, bias, running_mean, running_var, training,
        # momentum, eps). In training outputs 3 and 4 are the updated running statistics.
        # Outside training output 0 is batch norm with the running statistics, outputs 1 and 2
        # are empty, and outputs 3 and 4 are the running statistics unchanged, converted to the
        # computing dtype (binary32 for binary16 and bfloat16).
        if producer.args[5] is not True:
            if index in (3, 4):
                statistic = producer.args[index]
                if _node_dtype(statistic) == _node_dtype(node):
                    return {'_alias': statistic}
                return {'kind': "custom", '_plan': {'op': 'convert', 'operands': [statistic]}}
            if index != 0:
                raise NotImplementedError(f'{name}: the empty saved statistics outputs outside '
                                          'training are not imported')
            return _running_kind(producer, source, producer.args[7])
        if producer.args[3] is producer.args[4]:
            raise NotImplementedError('_native_batch_norm_legit_functional: the running '
                                      'statistics must be two distinct buffers')
        return _moments_kind(source, 'batch', producer.args[1], producer.args[2],
                             producer.args[7], index,
                             running=(producer.args[3], producer.args[4], producer.args[6]))
    if name == 'aten._native_batch_norm_legit_no_training.default':
        return _running_kind(producer, source, _argument(producer, 6, 'eps'))
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


def _registered_kind(tag, fields, operands, input_shape, result_shape, output=0,
                     broadcast=False):
    # Output `output` of the registered operator `tag` of TorchLean.Op, its codec record's fields
    # in binder order and its value ports at the first operand's dtype. Each operand is reshaped
    # to `input_shape` first when that differs from its shape, or broadcast to it with
    # `broadcast`.
    return {'kind': tag, '_plan': {'op': 'registered', 'operands': operands, 'tag': tag,
                                   'fields': fields, 'scalar': _node_dtype(operands[0]),
                                   'output': output, 'input_shape': input_shape,
                                   'result_shape': result_shape, 'broadcast': broadcast}}


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
               'aten.lift_fresh_copy.default', *_EMPTY)


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


# The in-place writes that the default graph keeps, read as their functional operators.
_SCATTER_WRITES = ('aten.scatter_.src', 'aten.scatter_.value', 'aten.scatter_add_.default')
_IN_PLACE_WRITES = ('aten.index_put_.default', 'aten.index_add_.default',
                    'aten.index_copy_.default', 'aten.copy_.default', *_SCATTER_WRITES,
                    *_IN_PLACE_FILLS)


def _fresh_destination(node):
    # torch.export's default graph keeps index_put_, index_add_, index_copy_, copy_, scatter_,
    # scatter_add_, zero_ and fill_ and reads the written tensor afterwards through their node.
    # The functional operator is exact when the destination is fresh (an allocating operator's
    # result or an accepted in-place write), the in-place node is its only reader from then on,
    # and its earlier readers allocate their results.
    name = _target_name(node.target).split('.')[1]
    target = node.args[0]
    if not isinstance(target, torch.fx.Node) or target.op != 'call_function':
        raise NotImplementedError(f'{name} into an input, parameter or buffer is outside the '
                                  'pure graph contract')
    if not _allocates(target) and not _is_op(target, _IN_PLACE_WRITES):
        raise NotImplementedError(f'{name} into a view or a tensor that may alias another is not '
                                  'imported')
    if any(value is target for argument in node.args[1:]
           for value in (argument if isinstance(argument, (list, tuple)) else [argument])):
        raise NotImplementedError(f'{name} whose other operands include its destination')
    order = {other: position for position, other in enumerate(node.graph.nodes)}
    for user in target.users:
        if user is not node and (order[user] > order[node] or not _allocates(user)):
            raise NotImplementedError(f'{name} into a tensor that a view or a later node reads')


def _order_kind(name, producer, source, index=0):
    # The values of a running extremum, of a stable sort, or of max.dim and min.dim, or their
    # indices (component 1), the registered operator's index output.
    shape, dtype = _shape_from_arg(source), _node_dtype(source)
    if not shape:
        raise NotImplementedError(f'{name} of a rank-0 tensor is not imported')
    sort = name in ('aten.sort.stable', 'aten.sort.default')
    # The IR sort also runs at Booleans, false first, and at integer types by their values.
    if dtype not in _FLOATING and not (sort and (dtype == 'bool' or isinstance(dtype, dict))):
        raise NotImplementedError(f'{name} of a {dtype} tensor is not imported')
    if sort:
        if name == 'aten.sort.default' or producer.kwargs.get('stable') is not True:
            raise NotImplementedError('sort without stable=True: ATen orders equal entries, -0 '
                                      'and +0 among them, either way; the IR sort is stable')
        dim = _normalize_axis(producer.kwargs.get('dim', -1), len(shape))
        descending = bool(producer.kwargs.get('descending', False))
        if index == 1:
            return _registered_kind('sort', [{'shape': shape}, {'nat': dim}, {'bool': descending},
                                             {'element': _element(dtype, _op_table())}],
                                    [source], shape, shape, 1)
        return {'kind': "sort", 'dim': dim, 'descending': descending, '_operands': [source]}
    dim = _normalize_axis(_argument(producer, 1, 'dim'), len(shape))
    if name in ('aten.cummax.default', 'aten.cummin.default'):
        # cummax_cummin_helper takes a later equal entry and a later NaN, as the IR does.
        tag = 'cummax' if name == 'aten.cummax.default' else 'cummin'
        if index == 1:
            return _registered_kind(tag, [{'shape': shape}, {'nat': dim}, _LEFT_FOLD], [source],
                                    shape, shape, 1)
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
                            [source], input_shape, reduced)


def _filled(shape, fill, dtype):
    # torch.full, with a Python fill value rounded once to a binary16 or bfloat16 dtype, which
    # PyTorch rounds through binary32.
    if dtype in (torch.float16, torch.bfloat16) and isinstance(fill, (int, float)):
        fill = _round_once(fill, str(dtype).removeprefix('torch.'))
    return torch.full(shape, fill, dtype=dtype)


def _fill(node):
    # The one value of a creation operator's tensor, as a 0-d tensor of its dtype, or None for
    # arange, linspace and logspace. It is the value PyTorch creates with the same arguments at
    # shape (), except that a Python fill value becomes binary16 or bfloat16 in one rounding
    # (_filled).
    meta = node.meta.get('val', node.meta.get('tensor_meta'))
    dtype = getattr(meta, 'dtype', None)
    if _is_op(node, _SHAPED_CREATION):
        if _is_op(node, ('aten.ones_like.default', 'aten.new_ones.default')):
            fill = 1
        elif _is_op(node, ('aten.new_zeros.default', 'aten.zero_.default')):
            fill = 0
        elif _is_op(node, ('aten.full_like.default', 'aten.fill.Scalar', 'aten.fill_.Scalar')):
            fill = node.args[1]
        else:
            fill = node.args[2]
        if isinstance(fill, torch.fx.Node):
            raise NotImplementedError('a fill value computed by the graph is not a constant')
        return _filled([], fill, dtype)
    if any(isinstance(value, torch.fx.Node) for value in (*node.args, *node.kwargs.values())):
        raise NotImplementedError('creation operator with a tensor operand')
    if not _is_op(node, ('aten.full.default', 'aten.ones.default', 'aten.zeros.default',
                         'aten.scalar_tensor.default')):
        return None
    if dtype in (torch.float16, torch.bfloat16) and _is_op(
            node, ('aten.full.default', 'aten.scalar_tensor.default')):
        fill = node.args[1] if _is_op(node, ('aten.full.default',)) else node.args[0]
        return _filled([], fill, dtype)
    keywords = {key: value for key, value in node.kwargs.items()
                if key not in ('device', 'pin_memory', 'layout')}
    args = node.args if _is_op(node, ('aten.scalar_tensor.default',)) else ([], *node.args[1:])
    value = node.target(*args, **keywords)
    if value.dtype != dtype:
        raise NotImplementedError('a created constant differs from its captured metadata')
    return value


def _creation_value(node):
    # The tensor PyTorch creates, computed with the same arguments on the CPU, except that a
    # fill value becomes binary16 or bfloat16 in one rounding (_fill).
    meta = node.meta.get('val', node.meta.get('tensor_meta'))
    shape, dtype = _node_shape(node), getattr(meta, 'dtype', None)
    fill = _fill(node)
    if fill is not None:
        return fill.expand(shape).contiguous()
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
    complex_kind = _complex_kind(node)
    if complex_kind is not None:
        return complex_kind
    if _folds(node) and not _is_op(node, _CREATION + _SHAPED_CREATION):
        value = _index_constant(node)
        if value is not None:
            return {'_constant': value}
    if _is_op(node, ('aten.argmax.default', 'aten.argmin.default'), (torch.argmax, torch.argmin),
              ('argmax', 'argmin')):
        source = _argument(node, 0, 'input', 'self')
        dim = _argument(node, 1, 'dim')
        tag = 'argmin' if _is_op(node, ('aten.argmin.default',), (torch.argmin,),
                                 ('argmin',)) else 'argmax'
        return _extremum_index_kind(tag, source, None if dim is None else int(dim),
                                    bool(_argument(node, 2, 'keepdim')))
    if _is_op(node, _EMPTY):
        # The complete overwrite uses the destination's shape and dtype only. Outside this
        # pattern a fixed IR value would depend on PyTorch's deterministic-fill settings.
        overwrites = (*_SHAPED_CREATION, 'aten.copy_.default', 'aten.copy.default')
        for user in node.users:
            writes_all = _is_op(user, overwrites) or _zeros_like(user)
            if not writes_all or user.args[0] is not node or node in user.args[1:] or \
                    node in user.kwargs.values():
                raise _UninitializedRead('a read of an uninitialized PyTorch allocation')
        return {'_overwritten_allocation': True}
    if _is_op(node, _CREATION + _SHAPED_CREATION):
        # A tensor of one value is that value broadcast to the shape, which keeps the artifact
        # one word long however large the tensor.
        if _is_op(node, _IN_PLACE_FILLS):
            _fresh_destination(node)
        fill = _fill(node)
        if fill is None:
            return {'_constant': _creation_value(node)}
        return {'kind': "broadcast", '_plan': {'op': 'filled', 'value': fill, 'operands': []}}
    if (_is_op(node, _DETACH, (), ('detach_',)) and _node_dtype(node) in _FLOATING and
            not _is_op(_argument(node, 0, 'self', 'input'), ('aten.lift_fresh_copy.default',))):
        # Tensor.detach_ stops the gradient of its operand where the program reads it next:
        # Op.detach, whose tangent is zero. torch.export writes those reads as uses of detach_'s
        # result. A symbolic trace keeps them on the operand, where they would miss Op.detach, and
        # is refused.
        source = _argument(node, 0, 'self', 'input')
        order = {each: index for index, each in enumerate(node.graph.nodes)}
        if any(order[user] > order[node] for user in source.users if user is not node):
            raise NotImplementedError('detach_ of a value the graph reads after it from the '
                                      'operand')
        return {'kind': "detach"}
    if (_is_op(node, _DETACH_COPY, (torch.detach,), ('detach',)) or _decomposed_detach(node)) \
            and _node_dtype(node) in _FLOATING:
        # Tensor.detach: the operand's value, read through the result without its gradient.
        source = _argument(node, 0, 'self', 'input')
        if _node_dtype(source) != _node_dtype(node) or _node_shape(source) != _node_shape(node):
            raise NotImplementedError('detach must keep its operand shape and dtype')
        return {'kind': "detach"}
    if _is_op(node, _ALIASES + _DETACH + _DETACH_COPY, (torch.clone, torch.detach),
              ('clone', 'detach_', 'detach')):
        source = _argument(node, 0, 'self', 'input')
        if _node_dtype(source) != _node_dtype(node) or _node_shape(source) != _node_shape(node):
            raise NotImplementedError('an alias must keep its operand shape and dtype')
        return {'_alias': source}
    if _is_op(node, ('aten.copy.default', 'aten.copy_.default')):
        # Functional copy_, or copy_ into a fresh destination: the source broadcast to the
        # destination and converted to its dtype.
        if _is_op(node, ('aten.copy_.default',)):
            _fresh_destination(node)
        return {'kind': "custom", '_plan': {'op': 'convert', 'operands': [node.args[1]]}}
    if _is_op(node, ('aten.t.default',), (torch.t,), ('t',)):
        source = node.args[0]
        if len(_shape_from_arg(source)) < 2: return {'_alias': source}
        return {'kind': "permute", 'dims': [1, 0]}
    if _is_op(node, ('aten.numpy_T.default', 'aten.mT.default', 'aten.mH.default',
                     'aten.adjoint.default', 'aten.matrix_H.default'), (torch.adjoint,),
              ('adjoint',)):
        # Tensor.T reverses the axes and is its operand below rank 2. Tensor.mT swaps the last two
        # axes, and Tensor.mH, Tensor.H and torch.adjoint swap them and conjugate, which leaves a
        # real tensor as it is (the PyTorch 2.13 docstrings of these five). Complex tensors are
        # refused with their element type.
        source = _argument(node, 0, 'self', 'input')
        rank = len(_shape_from_arg(source))
        if _is_op(node, ('aten.numpy_T.default',)):
            if rank < 2: return {'_alias': source}
            return {'kind': "permute", 'dims': list(range(rank))[::-1]}
        return {'kind': "permute", 'dims': [*range(rank - 2), rank - 1, rank - 2]}
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
        return {'kind': "leaky_relu", '_plan': {'op': 'leaky_relu', 'operands': [source],
                                                'slope': slope}}
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
        axes, positions = _advanced_axes(node, source, indices,
                                         bool(_argument(node, 3, 'accumulate')))
        if not isinstance(values, torch.fx.Node) or _node_dtype(values) != _node_dtype(source):
            raise NotImplementedError('index_put needs values of the destination dtype')
        return {'kind': "scatter", '_plan': {
            'op': 'index_put', 'axes': axes, 'positions': positions,
            'accumulate': bool(_argument(node, 3, 'accumulate')),
            'operands': [source, values] + [indices[a] for a in axes]}}
    if _is_op(node, tuple(f'aten.{name}.default' for name in ('index_add', 'index_add_',
                                                              'index_copy', 'index_copy_')),
              (torch.index_add, torch.index_copy), ('index_add', 'index_copy')):
        # index_add(self, dim, index, source) accumulates source[..., i, ...] into
        # self[..., index[i], ...] along dim, in the order given in index (the PyTorch 2.13
        # docstring of Tensor.index_add_): index_put with accumulate at the one axis dim, each
        # step rounded in the dtype, as PyTorch's decomposition (torch._refs.index_add) and its
        # CPU kernel of a vector do. Its CPU kernel of a matrix or higher sums binary16 and
        # bfloat16 in binary32 (measured). ATen fuses alpha into the sum, as in add, so alpha
        # other than 1 is refused as there. index_copy copies there instead: index_put without
        # accumulate, as torch._refs.index_copy decomposes it. Its docstring leaves a repeated
        # target nondeterministic, and the CPU kernel keeps the last copy (measured), as the
        # scatter does. Both reject a negative index, which index_put wraps.
        accumulate = 'index_add' in _target_name(node.target)
        name = 'index_add' if accumulate else 'index_copy'
        if _is_op(node, ('aten.index_add_.default', 'aten.index_copy_.default')):
            _fresh_destination(node)
        source, index, values = _argument(node, 0, 'input', 'self'), \
            _argument(node, 2, 'index'), _argument(node, 3, 'source')
        if accumulate and _argument(node, 4, 'alpha') not in (None, 1):
            raise NotImplementedError('index_add with alpha other than 1 is not imported')
        extents = _shape_from_arg(source)
        dim = _normalize_axis(_argument(node, 1, 'dim'), max(len(extents), 1))
        axes, positions = _advanced_axes(node, source, [None] * dim + [index], accumulate)
        if len(_shape_from_arg(index)) != 1:
            raise NotImplementedError(f'{name} needs an index vector')
        if not isinstance(values, torch.fx.Node) or _node_dtype(values) != _node_dtype(source) \
                or _shape_from_arg(values) != extents[:dim] + positions + extents[dim + 1:]:
            raise NotImplementedError(f'{name} needs a source of the destination dtype, as '
                                      'long as the index along dim and equal elsewhere')
        return {'kind': "scatter", '_plan': {
            'op': 'index_put', 'axes': axes, 'positions': positions, 'accumulate': accumulate,
            'wrap': False, 'operands': [source, values, index]}}
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
    if _is_op(node, ('aten.linalg_householder_product.default', 'aten.orgqr.default'),
              (torch.linalg.householder_product, torch.orgqr), ('orgqr',)):
        # householder_product(input, tau) and its alias orgqr(input, input2) (the bindings'
        # names; the docstrings say A and tau): the first n columns of H_1 ... H_k for an input
        # of shape (*, m, n), as LinalgImport.householder_product builds it.
        return {'kind': "matmul", '_plan': {'op': 'householder_product', 'operands': [
            _argument(node, 0, 'input', 'self'), _argument(node, 1, 'tau', 'input2')]}}
    if _is_op(node, ('aten.ormqr.default',), (torch.ormqr,), ('ormqr',)):
        # ormqr(input, input2, input3, left=True, transpose=False) (the docstring's input, tau
        # and other): op(Q) other, or other op(Q), as LinalgImport.reflected builds it.
        left, transpose = _argument(node, 3, 'left'), _argument(node, 4, 'transpose')
        return {'kind': "matmul", '_plan': {'op': 'ormqr', 'operands': [
            _argument(node, 0, 'input', 'self'), _argument(node, 1, 'input2'),
            _argument(node, 2, 'input3')], 'left': True if left is None else bool(left),
            'transpose': bool(transpose)}}
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
    if _is_op(node, _SAMPLERS):
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
        _check_moved(_target_name(node.target), _node_dtype(source))
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
    if _is_op(node, ('aten.as_strided.default',)):
        # Entry i of as_strided(x, size, stride, offset) is the storage word at offset +
        # sum_k i_k * stride_k (offset defaults to x's own). When each word read holds an entry
        # of x in the layout the capture's metadata gives x, that is a gather of those entries.
        # PyTorch's value is that one when the inputs are laid out as the captured ones, which
        # torch.export does not check (export.py records a constraint).
        source = _argument(node, 0, 'input', 'self')
        size, stride = _argument(node, 1, 'size'), _argument(node, 2, 'stride')
        offset = _argument(node, 3, 'storage_offset')
        meta = source.meta.get('val') if isinstance(source, torch.fx.Node) else None
        _check_moved('as_strided', _node_dtype(source))
        if not isinstance(meta, torch.Tensor) or not all(
                isinstance(n, int) for n in (*size, *stride, *meta.stride(), meta.storage_offset(),
                                             0 if offset is None else offset)):
            raise NotImplementedError('as_strided needs a size, stride, offset and operand '
                                      'layout known at capture')
        extents = list(meta.shape)
        held = _storage_words(extents, meta.stride(), meta.storage_offset())
        read = _storage_words(size, stride, meta.storage_offset() if offset is None else offset)
        # A stable sort keeps the first tensor entry at each storage word. Search those
        # words without allocating space for unrelated storage or gaps between strides.
        words, entries = torch.sort(held.reshape(-1), stable=True)
        positions = torch.searchsorted(words, read)
        if (positions == words.numel()).any() or (words[positions] != read).any():
            raise NotImplementedError('as_strided reading memory outside its operand')
        coords = torch.stack(torch.unravel_index(entries[positions], extents), -1) if extents else \
            torch.zeros([*size, 0], dtype=torch.long)
        return {'kind': "gather", 'dims': list(range(len(extents))), 'batch_dims': [],
                'indices': {'shape': list(size), 'values': coords.reshape(-1).tolist()},
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
    scatters = ('aten.scatter.src', 'aten.scatter.value', 'aten.scatter_add.default',
                *_SCATTER_WRITES)
    sums = ('aten.scatter_add.default', 'aten.scatter_add_.default')
    if _is_op(node, _SCATTER_WRITES):
        _fresh_destination(node)
    if _is_op(node, scatters, (torch.scatter, torch.scatter_add), ('scatter', 'scatter_add')) and \
            _top_k_mask(node) is None and _index_constant(_argument(node, 2, 'index')) is None:
        # Op.scatter on a runtime index, at complete coordinates built in the IR.
        base = _argument(node, 0, 'input', 'self')
        extents = _shape_from_arg(base)
        dim = _normalize_axis(_argument(node, 1, 'dim'), len(extents))
        index = _index_operand(_argument(node, 2, 'index'), _target_name(node.target))
        update = _argument(node, 3, 'src', 'value')
        if node.kwargs.get('reduce') is not None:
            raise NotImplementedError('scatter with a reduce argument is not imported')
        summing = _is_op(node, sums, (torch.scatter_add,), ('scatter_add',))
        # A Python value is imported into a floating destination only.
        _check_moved(_target_name(node.target), _node_dtype(base),
                     summing or not isinstance(update, torch.fx.Node))
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
        reduce = 'sum' if summing else 'replace'
        return {'kind': "scatter", '_plan': {'op': 'scatter_at', 'operands': [base, index, update],
                                             'dim': dim, 'reduce': reduce}}
    if _is_op(node, scatters, (torch.scatter, torch.scatter_add), ('scatter', 'scatter_add')) and \
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
        reduce = 'sum' if _is_op(node, sums, (torch.scatter_add,), ('scatter_add',)) else \
            'replace'
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
        dim, index = _selected(node)
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
    if _is_op(node, ('aten.stack.default',), (torch.stack,), ()):
        # torch.stack concatenates its tensors along a new axis `dim` (the PyTorch 2.13
        # docstring).
        tensors = list(_argument(node, 0, 'tensors'))
        if not tensors or any(_node_dtype(t) != _node_dtype(tensors[0]) for t in tensors):
            raise NotImplementedError('stack operands must be tensors of one dtype')
        dim = _argument(node, 1, 'dim')
        return {'kind': "cat", '_plan': {'op': 'stack', 'operands': tensors, 'dim':
                                         _normalize_axis(dim or 0, len(_node_shape(node)))}}
    if _is_op(node, ('aten.repeat_interleave.self_int',), (torch.repeat_interleave,),
              ('repeat_interleave',)):
        # repeat_interleave with an int count repeats each entry `repeats` times in a row along
        # `dim`, or along the flattened tensor without one (the PyTorch 2.13 docstring).
        source = _argument(node, 0, 'input', 'self')
        repeats, dim = _argument(node, 1, 'repeats'), _argument(node, 2, 'dim')
        if not isinstance(repeats, int) or repeats < 0:
            raise NotImplementedError('repeat_interleave needs a nonnegative int count')
        rank = len(_shape_from_arg(source))
        return {'kind': "reshape", '_plan': {'op': 'interleave', 'operands': [source],
                                             'repeats': repeats, 'dim':
                                             None if dim is None else _normalize_axis(dim, rank)}}
    if _is_op(node, ('aten.roll.default',), (torch.roll,), ('roll',)):
        # torch.roll moves the entries `shift` places along each axis in `dims`, wrapping around,
        # and rolls the flattened tensor when `dims` is empty (the PyTorch 2.13 docstring).
        source = _argument(node, 0, 'input', 'self')
        rank = len(_shape_from_arg(source))
        shifts, dims = _argument(node, 1, 'shifts'), _argument(node, 2, 'dims')
        shifts = [shifts] if isinstance(shifts, int) else list(shifts)
        dims = [] if dims is None else [dims] if isinstance(dims, int) else list(dims)
        if len(shifts) != max(len(dims), 1):
            raise NotImplementedError(f'roll shifts {shifts} do not pair with axes {dims}')
        return {'kind': "cat", '_plan': {'op': 'roll', 'operands': [source],
                                         'shifts': [int(shift) for shift in shifts],
                                         'dims': [_normalize_axis(d, rank) for d in dims]}}
    if _is_op(node, ('aten.unfold.default',), (), ('unfold',)):
        # Tensor.unfold(dimension, size, step) holds every slice of `size` entries along
        # `dimension` whose starts are `step` apart, as a new last axis of extent `size` (the
        # PyTorch 2.13 docstring). torch.nn.functional.unfold is im2col, another operator.
        source = _argument(node, 0, 'self', 'input')
        extents = _shape_from_arg(source)
        dim, size, step = (_argument(node, 1, 'dimension'), _argument(node, 2, 'size'),
                           _argument(node, 3, 'step'))
        if not extents or size == 0:
            raise NotImplementedError('unfold of a rank-0 tensor or into windows of size 0 is '
                                      'not imported')
        return {'kind': "cat", '_plan': {'op': 'unfold', 'operands': [source],
                                         'dim': _normalize_axis(dim, len(extents)),
                                         'step': int(step)}}
    if _is_op(node, ('aten.tril.default', 'aten.triu.default'), (torch.tril, torch.triu),
              ('tril', 'triu')):
        # torch.tril keeps the entries (i, j) of the last two axes with j - i <= diagonal, and
        # torch.triu those with j - i >= diagonal. Both write zero elsewhere (the PyTorch 2.13
        # docstrings), and PyTorch refuses an operand of rank below 2 when it traces them.
        source = _argument(node, 0, 'input', 'self')
        lower = _is_op(node, ('aten.tril.default',), (torch.tril,), ('tril',))
        return {'kind': "where", '_plan': {'op': 'triangle', 'operands': [source], 'lower': lower,
                                           'diagonal': int(_argument(node, 1, 'diagonal') or 0)}}
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
        shape = _held_shape(weight_arg)
        rank = len(shape) - 2
        input_shape = _shape_from_arg(source)
        if len(input_shape) != rank + 2:
            raise NotImplementedError(
                f'convolution expected rank {rank + 2}, got {len(input_shape)}')
        padding = _spatial_tuple(padding, rank, 0)
        if transposed:
            # The weight is (in, out / groups, kernel...), and the bias, if any, one entry per
            # output channel; `_emit_structure` writes the convolution.
            groups = int(groups)
            if rank < 1 or groups <= 0 or shape[0] % groups:
                raise NotImplementedError(
                    f'transposed convolution of weight {tuple(shape)} in {groups} groups')
            return {'kind': "conv", '_plan': {
                'op': 'conv_transpose', 'operands': [source, weight_arg, bias_arg],
                'kernel': [int(v) for v in shape[2:]], 'stride': _spatial_tuple(stride, rank, 1),
                'padding': padding, 'dilation': _spatial_tuple(dilation, rank, 1),
                'output_padding': _spatial_tuple(output_padding, rank, 0), 'groups': groups}}
        # A weight or bias not known at capture is an operand after the source too.
        return {'kind': "conv", 'spatial_rank': rank, 'kernel': [int(v) for v in shape[2:]],
                'stride': _spatial_tuple(stride, rank, 1), 'padding': padding,
                'padding_after': padding, 'dilation': _spatial_tuple(dilation, rank, 1),
                'groups': int(groups), 'channel_axis': 1, 'input_spatial': input_shape[2:],
                '_operands': [source], **_convolution_payload(weight_arg, bias_arg, groups, source)}
    return None
