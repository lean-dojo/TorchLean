"""The node kind of every PyTorch call and the capture nodes of the structural plans.

`_lower_kind(node, model)` gives a call's node kind, or a plan that `_emit_structure` (or
`_pointwise._emit_pointwise`) expands into several nodes. `_normalization_payload` writes the
parameters of a layer or batch normalization node as words of its computing dtype.
"""

from __future__ import annotations

import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._artifact import _SCALAR
from ._fx import (_FLOATING, _INT64, _REDUCED_FLOATS, _binary_payload, _broadcast_kind, _constant,
                  _convolution_payload, _dtype_of, _is_getattr, _is_op, _linear_payload,
                  _node_dtype, _node_shape, _normalization, _normalize_axis, _op_table,
                  _reduction_axes, _reshape_kind, _resolve_tensor, _shape_from_arg, _shape_of,
                  _spatial_tuple, _target_name, _top_k_mask, _torch_dtype)
from ._pointwise import (_append_node, _argument, _arithmetic_kind, _broadcast_to, _cast_to,
                         _pointwise_kind, _scalar_constant, _scalar_tensor)
from ._structure import (_BUFFERS, _BUFFER_UPDATES, _DROPOUT_MASKS, _advanced_indices, _coordinates,
                         _fourier_scale, _index_constant, _index_operand, _index_tuples, _iota,
                         _moments_kind, _random_key, _registered_kind, _registered_node,
                         _structure_kind)


def _normalization_payload(plan, dtype):
    # Parameters and eps as interchange words of the computing dtype.
    target = _torch_dtype(dtype)
    eps = _scalar_tensor(plan['eps'], dtype)
    entry = _binary_payload('eps', eps)
    for name, value in plan['parameters'].items():
        if value is None:
            value = (torch.ones if name == 'gamma' else torch.zeros)(plan['shape'], dtype=target)
        value = value.detach()
        if value.dtype != target and not (dtype == 'float32' and
                                          _dtype_of(value) in _REDUCED_FLOATS):
            raise NotImplementedError(f'a {_dtype_of(value)} {name} of a {dtype} normalization')
        value = value.to(target)
        entry.update(_binary_payload(name, value))
    return entry


def _emit_structure(plan, nodes, node_to_id, shape, dtype):
    # Append the IR nodes of a structural operator that takes several of them, and return the last.
    op = plan['op']
    if op == 'buffer_update':
        result = _append_node(nodes, {'parents': [], **_constant(plan['value'])})
        _BUFFER_UPDATES[plan['buffer']] = result
        return result
    def parent(value):
        index = node_to_id.get(value)
        if index is None: raise NotImplementedError(f'{op} operand has no tensor lowering')
        return index
    def emit(attributes, parents, out_shape):
        return _append_node(nodes, {**attributes, 'parents': parents, 'shape': out_shape,
                                    'dtype': dtype})
    def reshape(index, out_shape):
        if nodes[index]['shape'] == out_shape: return index
        return emit(_reshape_kind(), [index], out_shape)
    def broadcast(index, out_shape):
        source = nodes[index]['shape']
        if source == out_shape: return index
        return emit(_broadcast_kind(source, out_shape), [index], out_shape)
    def crop(index, dims, starts, stops):
        # One unit-step slice keeping [start, stop) along each listed axis.
        out = list(nodes[index]['shape'])
        for dim, start, stop in zip(dims, starts, stops):
            out[dim] = stop - start
        return emit({'kind': "slice", 'dims': dims, 'starts': starts, 'stops': stops,
                     'steps': [1] * len(dims)}, [index], out)
    def window(index, dim, start, stop):
        return crop(index, [dim], [start], [stop])
    def concat(pieces, dim):
        if len(pieces) == 1: return pieces[0]
        out = list(nodes[pieces[0]]['shape'])
        out[dim] = sum(nodes[piece]['shape'][dim] for piece in pieces)
        return emit({'kind': "cat", 'dim': dim}, pieces, out)
    def padded_axes(padding, rank):
        # PyTorch lists (low, high) pairs from the last axis backwards.
        if len(padding) % 2 or len(padding) // 2 > rank:
            raise NotImplementedError(f'padding {padding} does not pair with {rank} axes')
        return [(rank - 1 - k, padding[2 * k], padding[2 * k + 1])
                for k in range(len(padding) // 2)]
    if op == 'select':
        index = parent(plan['operands'][0])
        return reshape(window(index, plan['dim'], plan['index'], plan['index'] + 1), shape)
    if op == 'normalization':
        # PyTorch's CPU kernels normalize binary16 and bfloat16 in binary32 and round once.
        compute = 'float32' if dtype in _REDUCED_FLOATS else dtype
        if compute not in ('float32', 'float64'):
            raise NotImplementedError(f"{plan['attributes']['kind']} of a {dtype} tensor")
        index = _cast_to(nodes, parent(plan['operands'][0]), compute)
        result = _append_node(nodes, {**plan['attributes'], **_normalization_payload(plan, compute),
                                      'parents': [index], 'shape': shape, 'dtype': compute})
        return _cast_to(nodes, result, dtype)
    if op == 'attention':
        # The docstring's equivalent code: softmax(q @ k^T * scale + bias, dim=-1) @ v, with
        # k and v repeated head by head for enable_gqa. The bias is built as that code builds it:
        # zeros (+0), -inf where a Boolean mask or the causal mask tril(ones(L, S)) is false, or
        # the floating mask plus those zeros. It is added (a blocked +inf or NaN score makes its
        # row NaN, and so does a row without an allowed key). The scale, a Python float (the
        # default 1 / math.sqrt(E) in binary64), is rounded once to the computing type. Binary16
        # and bfloat16 operands compute in binary32 (the docstring's note: "all intermediates are
        # kept in torch.float") and the result is rounded once.
        compute = 'float32' if dtype in _REDUCED_FLOATS else dtype
        def at(kind, parents, at_shape):
            return _append_node(nodes, {**kind, 'parents': parents, 'shape': at_shape,
                                        'dtype': compute})
        query, key, value = (_cast_to(nodes, parent(x), compute) for x in plan['operands'][:3])
        def repeated(index):
            # repeat_interleave(group, dim=-3): every head `group` times in a row.
            source = nodes[index]['shape']
            split = at(_reshape_kind(), [index], source[:-2] + [1] + source[-2:])
            spread = _broadcast_to(nodes, split, source[:-2] + [plan['group']] + source[-2:])
            return at(_reshape_kind(), [spread],
                      source[:-3] + [source[-3] * plan['group']] + source[-2:])
        if plan['group'] > 1:
            key, value = repeated(key), repeated(value)
        batch, (rows, depth), (keys, width) = plan['batch'], plan['query'][-2:], plan['value'][-2:]
        rank = len(batch) + 2
        query = _broadcast_to(nodes, query, batch + [rows, depth])
        key = _broadcast_to(nodes, key, batch + [keys, depth])
        value = _broadcast_to(nodes, value, batch + [keys, width])
        scores_shape = batch + [rows, keys]
        swap = list(range(rank - 2)) + [rank - 1, rank - 2]
        transposed = at({'kind': "permute", 'dims': swap}, [key], batch + [depth, keys])
        scores = at({'kind': "mul_elem"},
                    [at({'kind': "matmul"}, [query, transposed], scores_shape),
                     _broadcast_to(nodes, _scalar_constant(nodes, plan['scale'], compute),
                                   scores_shape)], scores_shape)
        mask = plan['operands'][3]
        zeros = _broadcast_to(nodes, _scalar_constant(nodes, 0.0, compute), [rows, keys])
        bias = zeros
        if plan['causal']:
            allowed = torch.ones(rows, keys, dtype=torch.bool).tril()
            bias = _append_node(nodes, {'parents': [], **_constant(torch.zeros(
                rows, keys, dtype=_torch_dtype(compute)).masked_fill(~allowed, float('-inf')))})
        elif mask is not None:
            bias = parent(mask)
            if nodes[bias]['dtype'] == 'bool':
                extents = nodes[bias]['shape']
                zero, blocked = (_broadcast_to(nodes, _scalar_constant(nodes, number, compute),
                                               extents) for number in (0.0, float('-inf')))
                bias = _append_node(nodes, {'kind': 'where', 'shape': extents,
                                            'parents': [bias, zero, blocked], 'dtype': compute})
            else:
                bias = _cast_to(nodes, bias, compute)
                extents = list(torch.broadcast_shapes(tuple(nodes[bias]['shape']), (rows, keys)))
                bias = at({'kind': "add"}, [_broadcast_to(nodes, bias, extents),
                                            _broadcast_to(nodes, zeros, extents)], extents)
        scores = at({'kind': "add"}, [scores, _broadcast_to(nodes, bias, scores_shape)],
                    scores_shape)
        weights = at({'kind': "softmax", 'dim': rank - 1}, [scores], scores_shape)
        return _cast_to(nodes, at({'kind': "matmul"}, [weights, value], batch + [rows, width]),
                        dtype)
    if op == 'addmm':
        bias, left, right = (parent(value) for value in plan['operands'])
        product = emit({'kind': "matmul"}, [left, right], shape)
        return emit({'kind': "add"}, [product, broadcast(bias, shape)], shape)
    if op == 'repeat':
        index = parent(plan['operands'][0])
        repeats = plan['repeats']
        source = [1] * (len(repeats) - len(nodes[index]['shape'])) + nodes[index]['shape']
        if len(source) != len(repeats) or any(r < 0 for r in repeats):
            raise NotImplementedError('repeat needs one nonnegative count per axis')
        interleaved = [extent for pair in zip([1] * len(source), source) for extent in pair]
        tiled = [extent for pair in zip(repeats, source) for extent in pair]
        return reshape(broadcast(reshape(index, interleaved), tiled), shape)
    if op == 'reduce':
        index = parent(plan['operands'][0])
        if nodes[index]['dtype'] != dtype:
            # PyTorch accumulates sums and products of Boolean and integer tensors in int64.
            index = _cast_to(nodes, index, dtype)
        return emit(plan['attributes'], [index], shape)
    if op == 'var':
        index = parent(plan['operands'][0])
        source, dims = nodes[index]['shape'], plan['dims']
        if not dims:
            raise NotImplementedError('the variance of a scalar lists no axis')
        count = 1
        for axis in dims: count *= source[axis]
        squares = index
        if count > 0:
            # Over an empty fiber the sum of no squares is 0 and PyTorch's variance 0 / 0. The IR
            # mean of no entries would be 0, so it is not formed.
            kept = [1 if axis in dims else extent for axis, extent in enumerate(source)]
            mean = emit({'kind': "mean", 'dims': dims, 'keepdim': True}, [index], kept)
            centred = emit({'kind': "sub"}, [index, broadcast(mean, source)], source)
            squares = emit({'kind': "mul_elem"}, [centred, centred], source)
        total = emit({'kind': "reduce", 'op': 'sum', 'dims': dims,
                      'keepdim': plan['keepdim']}, [squares], shape)
        divisor = _scalar_constant(nodes, max(0, count - plan['correction']), dtype)
        return emit({'kind': "div"}, [total, broadcast(divisor, shape)], shape)
    if op == 'pad':
        # The nonnegative amounts pad first, and negative amounts then crop the padded axis.
        index = parent(plan['operands'][0])
        source = nodes[index]['shape']
        low, high = [0] * len(source), [0] * len(source)
        for dim, before, after in padded_axes(plan['padding'], len(source)):
            low[dim], high[dim] = before, after
        if any(p > 0 for p in low + high):
            value = _scalar_constant(nodes, plan['value'], dtype)
            padded = [n + max(0, a) + max(0, b) for n, a, b in zip(source, low, high)]
            index = emit({'kind': "pad", 'low': [max(0, p) for p in low],
                          'high': [max(0, p) for p in high], 'interior': [0] * len(source)},
                         [index, value], padded)
        cropped = [dim for dim in range(len(source)) if low[dim] < 0 or high[dim] < 0]
        if cropped:
            extents = nodes[index]['shape']
            starts = [max(0, -low[dim]) for dim in cropped]
            stops = [extents[dim] - max(0, -high[dim]) for dim in cropped]
            if any(stop < start for start, stop in zip(starts, stops)):
                raise NotImplementedError('padding crops more than the axis holds')
            index = crop(index, cropped, starts, stops)
        if nodes[index]['shape'] != shape:
            raise NotImplementedError('constant padding produced an unexpected shape')
        return index
    if op in ('reflect', 'replicate'):
        index = parent(plan['operands'][0])
        for dim, low, high in padded_axes(plan['padding'], len(nodes[index]['shape'])):
            extent = nodes[index]['shape'][dim]
            if low < 0 or high < 0:
                raise NotImplementedError(f'negative {op} padding is not imported')
            if op == 'reflect' and (low >= extent or high >= extent):
                raise NotImplementedError('reflection padding must be smaller than the axis')
            if op == 'replicate' and extent == 0 and (low or high):
                raise NotImplementedError('replication padding of an empty axis')
            def edge(count, first):
                # Reflection mirrors the entries next to the edge. Replication repeats the edge.
                if op == 'reflect':
                    start = 1 if first else extent - 1 - count
                    piece = window(index, dim, start, start + count)
                    return emit({'kind': "flip", 'dims': [dim]}, [piece], nodes[piece]['shape'])
                start = 0 if first else extent - 1
                piece = window(index, dim, start, start + 1)
                target = list(nodes[piece]['shape'])
                target[dim] = count
                return broadcast(piece, target)
            pieces = ([edge(low, True)] if low else []) + [index] + \
                ([edge(high, False)] if high else [])
            index = concat(pieces, dim)
        if nodes[index]['shape'] != shape:
            raise NotImplementedError(f'{op} padding produced an unexpected shape')
        return index
    if op == 'scan':
        # cpu_cum_base_kernel accumulates from 0 or 1 in at::acc_type (binary64 for binary32 and
        # binary64, binary32 for binary16 and bfloat16) and rounds every prefix to the result.
        index = _cast_to(nodes, parent(plan['operands'][0]), plan['accumulator'])
        scanned = _append_node(nodes, {'kind': plan['scan'], 'dim': plan['dim'],
                                       'parents': [index], 'shape': shape,
                                       'dtype': plan['accumulator']})
        return _cast_to(nodes, scanned, dtype)
    if op == 'embedding':
        # Op.embedding: the rows of the table at a runtime index tensor.
        weight, index = (parent(value) for value in plan['operands'])
        num, width = nodes[weight]['shape']
        positions = list(nodes[index]['shape'])
        return _registered_node(nodes, 'embedding',
                                [{'nat': num}, {'nat': width}, {'shape': positions}],
                                [weight, index], positions + [width], dtype)
    if op == 'gather_at':
        source, index = (parent(value) for value in plan['operands'])
        extents, dim = list(nodes[source]['shape']), plan['dim']
        positions = list(nodes[index]['shape'])
        if plan['select']:
            rest = [extent for axis, extent in enumerate(extents) if axis != dim]
            index = _registered_node(nodes, 'gather', [{'shape': extents}, {'shape': positions},
                                                       {'nats': [dim]}, {'pairs': []}, _SCALAR],
                                     [source, index], positions + rest, dtype)
            order = list(range(1, dim + 1)) + [0] + list(range(dim + 1, len(extents)))
            if order != list(range(len(extents))):
                index = emit({'kind': "permute", 'dims': order}, [index], shape)
            return index
        axes = [axis for axis in range(len(extents)) if axis != dim and
                positions[axis] != extents[axis]]
        if axes:
            source = crop(source, axes, [0] * len(axes), [positions[axis] for axis in axes])
        kept = list(nodes[source]['shape'])
        pairs = [[axis, axis] for axis in range(len(extents)) if axis != dim]
        return _registered_node(nodes, 'gather', [{'shape': kept}, {'shape': positions},
                                                  {'nats': [dim]}, {'pairs': pairs}, _SCALAR],
                                [source, index], positions, dtype)
    def scatter_rule(base, tuples, updates, axes, positions, rule):
        # Op.scatter's record fields, the node from the one constructor.
        extents = list(nodes[base]['shape'])
        code = _op_table()['codes']['scatter'][rule]
        return _registered_node(nodes, 'scatter', [{'shape': extents}, {'shape': positions},
                                                   {'nat': code}, {'nats': axes}, _SCALAR],
                                [base, tuples, updates], extents, nodes[base]['dtype'])
    def wide(index):
        # The CPU scatter kernels accumulate binary16 and bfloat16 sums and products in
        # binary32 and round once (measured against scatter_add and scatter_reduce).
        return _cast_to(nodes, index, 'float32') if dtype in _REDUCED_FLOATS else index
    def complete_tuples(index, rank, dim):
        # The complete target coordinate of each index entry: its own coordinate, the entry
        # along dim.
        positions = list(nodes[index]['shape'])
        index = _cast_to(nodes, index, _INT64)
        return _index_tuples(nodes, [index if axis == dim else _iota(nodes, positions, axis)
                                     for axis in range(rank)], positions)
    def updates_of(update, positions):
        if isinstance(update, torch.fx.Node):
            updates = parent(update)
            if nodes[updates]['shape'] != positions:
                updates = crop(updates, list(range(len(positions))), [0] * len(positions),
                               positions)
            return updates
        return _broadcast_to(nodes, _scalar_constant(nodes, update, dtype), positions)
    if op == 'scatter_at':
        # Op.scatter at complete coordinates. A repeated target folds its updates in row-major
        # order of the index tensor, the IR's convention. PyTorch leaves that order, and the
        # winner of a replacement, unspecified. A sum of binary16 or bfloat16 runs in binary32.
        base_value, index_value, update = plan['operands']
        base, index = parent(base_value), parent(index_value)
        rank, positions = len(nodes[base]['shape']), list(nodes[index]['shape'])
        tuples = complete_tuples(index, rank, plan['dim'])
        updates = updates_of(update, positions)
        if plan['reduce'] == 'sum':
            return _cast_to(nodes, scatter_rule(wide(base), tuples, wide(updates),
                                                list(range(rank)), positions, 'sum'), dtype)
        return scatter_rule(base, tuples, updates, list(range(rank)), positions, 'replace')
    if op == 'scatter_mean':
        # scatter_reduce with reduce 'sum' or 'mean'. Without include_self the scattered entries
        # are first zero. The mean divides the sum by the count of folded values (one more with
        # include_self), a count of zero read as one, as scatter_reduce_two does. The sum and
        # the count of binary16 or bfloat16 accumulate in binary32, each rounded once, and
        # their quotient is one division in the dtype.
        base_value, index_value, update = plan['operands']
        base = parent(base_value)
        extents = list(nodes[base]['shape'])
        rank, dim = len(extents), plan['dim']
        constant = _index_constant(index_value)
        if constant is not None:
            index = _append_node(nodes, {'parents': [], **_constant(constant.long())})
        else:
            index = parent(_index_operand(index_value, 'scatter_reduce'))
        positions = list(nodes[index]['shape'])
        if len(positions) != rank or any(
                positions[a] > extents[a] for a in range(rank) if a != dim):
            raise NotImplementedError('scatter_reduce needs an index of the input rank, no '
                                      'longer than the input along the other axes')
        tuples = complete_tuples(index, rank, dim)
        axes = list(range(rank))
        def zeros(at_shape):
            return _broadcast_to(nodes, _scalar_constant(nodes, 0, dtype), at_shape)
        def ones(at_shape):
            return _broadcast_to(nodes, _scalar_constant(nodes, 1, dtype), at_shape)
        seeded = wide(base)
        if not plan['include_self']:
            seeded = scatter_rule(seeded, tuples, wide(zeros(positions)), axes, positions,
                                  'replace')
        total = _cast_to(nodes, scatter_rule(seeded, tuples, wide(updates_of(update, positions)),
                                             axes, positions, 'sum'), dtype)
        if not plan['mean']:
            return total
        start = ones(extents) if plan['include_self'] else zeros(extents)
        count = _cast_to(nodes, scatter_rule(wide(start), tuples, wide(ones(positions)), axes,
                                             positions, 'sum'), dtype)
        empty = _append_node(nodes, {'kind': 'eq',
                                     'parents': [count, zeros(extents)], 'shape': extents,
                                     'dtype': 'bool'})
        count = _append_node(nodes, {'kind': 'where',
                                     'parents': [empty, ones(extents), count], 'shape': extents,
                                     'dtype': dtype})
        return emit({'kind': "div"}, [total, count], extents)
    if op == 'scatter_fold':
        # scatter_reduce with prod, amax or amin: one Op.scatter at complete coordinates with the
        # rule prod, max or min, a repeated target folding its updates in index order along dim,
        # as scatter_reduce_two does. The rules max and min are IEEE maximum and minimum, whose
        # value, though not its NaN word, does not depend on the order of the participants.
        # Without include_self the base entry at a written target takes no part. In that case a
        # product starts from 1, written there by a replacing scatter, and a maximum or minimum
        # from the target's first update, so that an infinite result is not tied by an infinite
        # seed. A product of binary16 or bfloat16 runs in binary32 and rounds once.
        base_value, index_value, update = plan['operands']
        base = parent(base_value)
        extents = list(nodes[base]['shape'])
        rank, dim, reduce = len(extents), plan['dim'], plan['reduce']
        constant = _index_constant(index_value)
        if constant is not None:
            index = _append_node(nodes, {'parents': [], **_constant(constant.long())})
        else:
            index = parent(_index_operand(index_value, 'scatter_reduce'))
        positions = list(nodes[index]['shape'])
        if len(positions) != rank or any(
                positions[a] > extents[a] for a in range(rank) if a != dim):
            raise NotImplementedError('scatter_reduce needs an index of the input rank, no '
                                      'longer than the input along the other axes')
        if 0 in positions:
            # No update lands anywhere: every rule returns the base.
            return base
        rule = {'prod': 'prod', 'amax': 'max', 'amin': 'min'}[reduce]
        tuples, axes = complete_tuples(index, rank, dim), list(range(rank))
        updates = updates_of(update, positions)
        if rule == 'prod':
            base, updates = wide(base), wide(updates)
        if plan['include_self']:
            return _cast_to(nodes, scatter_rule(base, tuples, updates, axes, positions, rule),
                            dtype)
        if rule == 'prod':
            base = scatter_rule(base, tuples, _broadcast_to(nodes, _scalar_constant(
                nodes, 1.0, nodes[base]['dtype']), positions), axes, positions, 'replace')
            return _cast_to(nodes, scatter_rule(base, tuples, updates, axes, positions, rule),
                            dtype)
        # An update is later when an earlier position along dim holds the same index: a pairwise
        # comparison along dim. The replacing scatter writes the first updates and sends the later
        # ones to a slice padded along dim, and the reducing scatter sends the first ones there.
        # The slice is cropped afterwards, so its entries have zero gradient. An index outside
        # [0, extent) becomes extent + 1, outside the padded base too, so the scatters fail on it
        # as the unpadded one would.
        n = positions[dim]
        def flag(name, parents, at_shape):
            return _append_node(nodes, {'kind': name, 'parents': parents,
                                        'shape': at_shape, 'dtype': 'bool'})
        def unsqueezed(axis):
            at_shape = positions[:axis] + [1] + positions[axis:]
            return _broadcast_to(nodes, _append_node(nodes, {
                **_reshape_kind(), 'parents': [index64], 'shape': at_shape, 'dtype': _INT64}),
                pairs)
        def bound(value):
            return _broadcast_to(nodes, _scalar_constant(nodes, value, _INT64), positions)
        raw = _cast_to(nodes, index, _INT64)
        inside = flag('logical_and', [flag('ge', [raw, bound(0)], positions),
                                      flag('lt', [raw, bound(extents[dim])], positions)],
                      positions)
        index64 = _append_node(nodes, {'kind': 'where',
                                       'parents': [inside, raw, bound(extents[dim] + 1)],
                                       'shape': positions, 'dtype': _INT64})
        pairs = positions[:dim + 1] + [n] + positions[dim + 1:]
        earlier = torch.tensor([[j < i for j in range(n)] for i in range(n)],
                               dtype=torch.bool).reshape(
            [1] * dim + [n, n] + [1] * (rank - dim - 1)).expand(pairs).contiguous()
        repeated = flag('logical_and', [flag('eq', [unsqueezed(dim + 1), unsqueezed(dim)], pairs),
                                        _append_node(nodes, {'parents': [], **_constant(earlier)})],
                        pairs)
        counted = _append_node(nodes, {'kind': "reduce", 'op': 'sum', 'dims': [dim + 1],
                                       'keepdim': False,
                                       'parents': [_cast_to(nodes, repeated, 'float64')],
                                       'shape': positions, 'dtype': 'float64'})
        later = flag('ne', [counted, _broadcast_to(nodes, _scalar_constant(nodes, 0, 'float64'),
                                                   positions)], positions)
        beyond = bound(extents[dim])
        def sent(away):
            return complete_tuples(_append_node(nodes, {
                'kind': 'where', 'parents': [away, beyond, index64],
                'shape': positions, 'dtype': _INT64}), rank, dim)
        padded = list(extents)
        padded[dim] += 1
        base = _append_node(nodes, {'kind': "pad", 'low': [0] * rank,
                                    'high': [1 if axis == dim else 0 for axis in range(rank)],
                                    'interior': [0] * rank,
                                    'parents': [base, _scalar_constant(nodes, 0, dtype)],
                                    'shape': padded, 'dtype': dtype})
        seeded = scatter_rule(base, sent(later), updates, axes, positions, 'replace')
        folded = scatter_rule(seeded, sent(flag('logical_not', [later], positions)), updates, axes,
                              positions, rule)
        return crop(folded, [dim], [0], [extents[dim]])
    if op == 'index' or op == 'index_put':
        # Op.gather or Op.scatter along the indexed axes at the broadcast index tuples. The
        # gathered axes lead. When the indexed axes are adjacent, PyTorch puts them at the
        # position of the first indexed axis, so a permutation moves them there (and moves
        # index_put's values back).
        axes, positions = plan['axes'], plan['positions']
        source = parent(plan['operands'][0])
        extents = list(nodes[source]['shape'])
        values = plan['operands'][1:] if op == 'index' else plan['operands'][2:]
        tuples = _advanced_indices(nodes, node_to_id, values, axes, extents, positions)
        rest = [extent for axis, extent in enumerate(extents) if axis not in axes]
        first, count = axes[0], len(positions)
        moved = axes == list(range(first, first + len(axes))) and first > 0
        if op == 'index':
            index = _registered_node(nodes, 'gather', [{'shape': extents}, {'shape': positions},
                                                       {'nats': axes}, {'pairs': []}, _SCALAR],
                                     [source, tuples], positions + rest, dtype)
            if moved:
                order = (list(range(count, count + first)) + list(range(count)) +
                         list(range(count + first, count + len(rest))))
                index = emit({'kind': "permute", 'dims': order}, [index], shape)
            return index
        update = parent(plan['operands'][1])
        selected = rest[:first] + positions + rest[first:] if moved else positions + rest
        update = broadcast(update, selected)
        if moved:
            order = (list(range(first, first + count)) + list(range(first)) +
                     list(range(first + count, count + len(rest))))
            update = emit({'kind': "permute", 'dims': order}, [update], positions + rest)
        return scatter_rule(source, tuples, update, axes, positions,
                            'sum' if plan['accumulate'] else 'replace')
    if op == 'adaptive_pool':
        # The window [floor(o n / m), ceil((o + 1) n / m)) along each spatial axis
        # (AdaptivePooling.h) is summed in row-major order from zero in the opmath type (the
        # dtype itself in 3d). The sum is divided by each window extent in axis order and rounded
        # once to the dtype, a schedule we measured against the CPU kernel on contiguous inputs.
        # When every axis divides evenly the pool is one reshape and one sum, and otherwise one
        # slice and one sum per window, concatenated.
        index = parent(plan['operands'][0])
        source, sizes = list(nodes[index]['shape']), plan['sizes']
        rank = len(sizes)
        lead = len(source) - rank
        # The 3d kernel accumulates binary16 and bfloat16 in their own format (measured).
        opmath = 'float32' if dtype in _REDUCED_FLOATS and rank < 3 else dtype
        index = _cast_to(nodes, index, opmath)
        def wide(attributes, parents, out_shape):
            return _append_node(nodes, {**attributes, 'parents': parents, 'shape': out_shape,
                                        'dtype': opmath})
        divisors = {}
        def divide(value, extent):
            at_shape = list(nodes[value]['shape'])
            key = (extent, tuple(at_shape))
            if key not in divisors:
                divisors[key] = _broadcast_to(nodes, _scalar_constant(nodes, extent, opmath),
                                              at_shape)
            return wide({'kind': "div"}, [value, divisors[key]], at_shape)
        spatial = source[lead:]
        if all(n % m == 0 for n, m in zip(spatial, sizes)):
            kernel = [n // m for n, m in zip(spatial, sizes)]
            split = source[:lead] + [v for m, k in zip(sizes, kernel) for v in (m, k)]
            pooled = wide(_reshape_kind(), [index], split)
            pooled = wide({'kind': "reduce", 'op': 'sum',
                           'dims': [lead + 2 * a + 1 for a in range(rank)], 'keepdim': False},
                          [pooled], source[:lead] + sizes)
            for k in kernel:
                pooled = divide(pooled, k)
            return _cast_to(nodes, pooled, dtype)
        spans = [[((o * n) // m, -((-(o + 1) * n) // m)) for o in range(m)]
                 for n, m in zip(spatial, sizes)]
        dims = list(range(lead, len(source)))
        def window(chosen):
            starts, stops = [a for a, _ in chosen], [b for _, b in chosen]
            piece = wide({'kind': "slice", 'dims': dims, 'starts': starts, 'stops': stops,
                          'steps': [1] * rank}, [index],
                         source[:lead] + [b - a for a, b in chosen])
            piece = wide({'kind': "reduce", 'op': 'sum', 'dims': dims, 'keepdim': True},
                         [piece], source[:lead] + [1] * rank)
            for a, b in chosen:
                piece = divide(piece, b - a)
            return piece
        def build(axis, chosen):
            if axis == rank: return window(chosen)
            pieces = [build(axis + 1, chosen + [span]) for span in spans[axis]]
            if len(pieces) == 1: return pieces[0]
            joined = list(nodes[pieces[0]]['shape'])
            joined[lead + axis] = len(pieces)
            return wide({'kind': "cat", 'dim': lead + axis}, pieces, joined)
        return _cast_to(nodes, build(0, []), dtype)
    if op == 'upsample_nearest':
        # One constant gather per resized axis, at the source indices PyTorch computes.
        index = parent(plan['operands'][0])
        for axis, picked in enumerate(plan['indices']):
            dim = 2 + axis
            extents = list(nodes[index]['shape'])
            if picked == list(range(extents[dim])): continue
            target = list(extents)
            target[dim] = len(picked)
            view = [-1 if a == dim else 1 for a in range(len(extents))]
            positions = torch.tensor(picked, dtype=torch.int64).reshape(view).expand(target)
            coords = _coordinates(positions, dim, extents)
            index = emit({'kind': "gather", 'dims': list(range(len(extents))), 'batch_dims': [],
                          'indices': {'shape': target, 'values': coords.reshape(-1).tolist()}},
                         [index], target)
        return reshape(index, shape)
    def transform(index, inverse, dim, n, code):
        # Op.rfft (unnormalized) or Op.irfft (its coefficients divided by n) along dim, the
        # complex values in paired-real form on a last axis of 2. The operand is first zero-padded
        # or cropped to n samples or n // 2 + 1 bins along dim, the result then multiplied by the
        # normalization's factor, which cannot undo the rounding inside Op.irfft.
        source = list(nodes[index]['shape'])
        want = n // 2 + 1 if inverse else n
        if source[dim] > want:
            index = crop(index, [dim], [0], [want])
        elif source[dim] < want:
            zero = _scalar_constant(nodes, 0, dtype)
            high = [want - source[dim] if axis == dim else 0 for axis in range(len(source))]
            padded = list(source)
            padded[dim] = want
            index = emit({'kind': "pad", 'low': [0] * len(source), 'high': high,
                          'interior': [0] * len(source)}, [index, zero], padded)
        operand = list(nodes[index]['shape'])
        if inverse:
            out = operand[:-1]
            out[dim] = n
            index = _registered_node(nodes, 'irfft', [{'shape': out}, {'nat': dim}, {'nat': n}],
                                     [index], out, dtype)
        else:
            out = list(operand)
            out[dim] = n // 2 + 1
            out = out + [2]
            index = _registered_node(nodes, 'rfft', [{'shape': operand}, {'nat': dim},
                                                     {'nat': n}], [index], out, dtype)
        factor = _fourier_scale(code, n, dtype, inverse)
        if factor is not None:
            scale = _broadcast_to(nodes, _scalar_constant(nodes, factor, dtype), out)
            index = emit({'kind': "mul_elem"}, [index, scale], out)
        return index
    if op == 'rfft' or op == 'irfft':
        index = parent(plan['operands'][0])
        forward = plan['forward']
        for dim, start, stop, step in plan.get('slices', []):
            extents = list(nodes[index]['shape'])
            extents[dim] = len(range(start, stop, step))
            index = emit({'kind': "slice", 'dims': [dim], 'starts': [start], 'stops': [stop],
                          'steps': [step]}, [index], extents)
        if forward is not None:
            index = transform(index, False, forward['dim'], forward['n'], forward['scale'])
        if op == 'irfft':
            index = transform(index, True, plan['dim'], plan['n'], plan['scale'])
        return reshape(index, shape)
    if op == 'moments':
        # GroupNormKernel.cpp and batch_norm_cpu: per group (or channel) the mean and the biased
        # variance in the opmath type, rstd = 1 / sqrt(var + eps). Without weight and bias the
        # group norm is (x - mean) * rstd, otherwise scale = rstd * weight, bias = -scale * mean +
        # bias and y = scale * x + bias. PyTorch accumulates the moments with a vectorized
        # recurrence and may fuse the multiply-add, so the IR's two-pass mean and variance and its
        # rounded products agree within a tolerance, not word for word.
        source_value, weight_value, bias_value = plan['operands'][:3]
        index = parent(source_value)
        extents = list(nodes[index]['shape'])
        opmath = 'float32' if dtype in _REDUCED_FLOATS else dtype
        def wide(attributes, parents, out_shape):
            return _append_node(nodes, {**attributes, 'parents': parents, 'shape': out_shape,
                                        'dtype': opmath})
        def constant(value, at_shape):
            return _broadcast_to(nodes, _scalar_constant(nodes, value, opmath), at_shape)
        index = _cast_to(nodes, index, opmath)
        batch, channels = extents[0], extents[1]
        spatial = 1
        for extent in extents[2:]: spatial *= extent
        if plan['mode'] == 'group':
            groups = plan['groups']
            grouped = [batch, groups, channels // groups * spatial]
            values = wide(_reshape_kind(), [index], grouped)
            dims, count = [2], grouped[2]
        else:
            values = index
            dims = [axis for axis in range(len(extents)) if axis != 1]
            count = batch * spatial
        if count == 0:
            raise NotImplementedError('a normalization over no entries')
        full = list(nodes[values]['shape'])
        kept = [1 if axis in dims else extent for axis, extent in enumerate(full)]
        mean = wide({'kind': "mean", 'dims': dims, 'keepdim': True}, [values], kept)
        centred = wide({'kind': "sub"}, [values, _broadcast_to(nodes, mean, full)], full)
        squares = wide({'kind': "mul_elem"}, [centred, centred], full)
        total = wide({'kind': "reduce", 'op': 'sum', 'dims': dims, 'keepdim': True}, [squares],
                     kept)
        var = wide({'kind': "div"}, [total, constant(count, kept)], kept)
        root = wide({'kind': "sqrt"}, [wide({'kind': "add"}, [var, constant(plan['eps'], kept)],
                                            kept)], kept)
        rstd = wide({'kind': "div"}, [constant(1, kept), root], kept)
        if 'momentum' in plan and (plan['output'] in (3, 4) or plan['record']):
            # momentum * mean + (1 - momentum) * running, and the same of the unbiased variance
            # var_sum / (n - 1), in the statistic's type (binary32 or binary64, that of the
            # input). The momentum is rounded to it, then 1 - momentum, each product and the sum.
            def running_update(statistic, value):
                running = parent(value)
                at = nodes[running]['dtype']
                def apply(kind, parents):
                    return _append_node(nodes, {'kind': kind, 'parents': parents,
                                                'shape': [channels], 'dtype': at})
                flat = _cast_to(nodes, wide(_reshape_kind(), [statistic], [channels]), at)
                factor = _broadcast_to(nodes, _scalar_constant(nodes, plan['momentum'], at),
                                       [channels])
                one = _broadcast_to(nodes, _scalar_constant(nodes, 1, at), [channels])
                keep = apply("sub", [one, factor])
                return apply("add", [apply("mul_elem", [factor, flat]),
                                     apply("mul_elem", [keep, running])])
            unbiased = wide({'kind': "div"}, [total, constant(count - 1, kept)], kept)
            updates = (running_update(mean, plan['operands'][3]),
                       running_update(unbiased, plan['operands'][4]))
            if plan['output'] in (3, 4):
                return updates[plan['output'] - 3]
            _BUFFER_UPDATES[plan['operands'][3].name] = updates[0]
            _BUFFER_UPDATES[plan['operands'][4].name] = updates[1]
        if plan['output'] in (1, 2):
            chosen = mean if plan['output'] == 1 else rstd
            return _cast_to(nodes, wide(_reshape_kind(), [chosen], shape), dtype)
        def per_channel(value):
            # The statistic of each (batch, channel) pair, or of each channel, broadcast over x.
            if plan['mode'] == 'group':
                value = wide(_broadcast_kind(kept, [batch, groups, channels // groups]), [value],
                             [batch, groups, channels // groups])
                value = wide(_reshape_kind(), [value], [batch, channels] + [1] * (len(extents) - 2))
            return _broadcast_to(nodes, value, extents)
        def parameter(value):
            if value is None: return None
            node = parent(value)
            node = _cast_to(nodes, node, opmath)
            node = wide(_reshape_kind(), [node], [1, channels] + [1] * (len(extents) - 2))
            return _broadcast_to(nodes, node, extents)
        gamma, beta = parameter(weight_value), parameter(bias_value)
        if plan['mode'] == 'group' and gamma is None and beta is None:
            centred = wide({'kind': "sub"}, [index, per_channel(mean)], extents)
            return _cast_to(nodes, wide({'kind': "mul_elem"}, [centred, per_channel(rstd)],
                                        extents), dtype)
        scale = per_channel(rstd)
        if gamma is not None:
            scale = wide({'kind': "mul_elem"}, [scale, gamma], extents)
        shift = wide({'kind': "neg"}, [wide({'kind': "mul_elem"}, [scale, per_channel(mean)],
                                            extents)], extents)
        if beta is not None:
            shift = wide({'kind': "add"}, [shift, beta], extents)
        product = wide({'kind': "mul_elem"}, [scale, index], extents)
        return _cast_to(nodes, wide({'kind': "add"}, [product, shift], extents), dtype)
    def constant_at(value, at_shape, at_dtype=None):
        at_dtype = dtype if at_dtype is None else at_dtype
        return _broadcast_to(nodes, _scalar_constant(nodes, value, at_dtype), at_shape)
    def apply_at(kind, parents, at_shape, at_dtype=None):
        return _append_node(nodes, {**kind, 'parents': parents, 'shape': at_shape,
                                    'dtype': dtype if at_dtype is None else at_dtype})
    if op == 'distance':
        # The p-norm distance of row pairs: p = 0 counts nonzero differences, 1 sums their
        # absolute values and 2 is the root of the sum of squares. For p = inf it is the largest
        # absolute value, and for other p the 1/p power of the sum of |d|^p. The euclid flag
        # takes cdist's matmul expansion clamp_min(-2 a.b + |a|^2 + |b|^2, 0).sqrt(), clamp_min
        # the IEEE maximum. PyTorch vectorizes the sums, so the values agree within a tolerance.
        left, right = (parent(value) for value in plan['operands'])
        a, b = list(nodes[left]['shape']), list(nodes[right]['shape'])
        p = plan['p']
        if plan['euclid']:
            lead, width = a[:-2], a[-1]
            def norms(x, rows):
                squared = apply_at({'kind': "mul_elem"}, [x, x], nodes[x]['shape'])
                return apply_at({'kind': "reduce", 'op': 'sum', 'dims': [len(lead) + 1],
                                 'keepdim': True}, [squared], lead + [rows, 1])
            ones_a, ones_b = constant_at(1, lead + [a[-2], 1]), constant_at(1, lead + [b[-2], 1])
            scaled = apply_at({'kind': "mul_elem"}, [left, constant_at(-2, a)], a)
            first = apply_at({'kind': "cat", 'dim': len(lead) + 1},
                             [scaled, norms(left, a[-2]), ones_a], lead + [a[-2], width + 2])
            second = apply_at({'kind': "cat", 'dim': len(lead) + 1},
                              [right, ones_b, norms(right, b[-2])], lead + [b[-2], width + 2])
            order = list(range(len(lead))) + [len(lead) + 1, len(lead)]
            second = apply_at({'kind': "permute", 'dims': order}, [second],
                              lead + [width + 2, b[-2]])
            product = apply_at({'kind': "matmul"}, [first, second], lead + [a[-2], b[-2]])
            clamped = apply_at({'kind': "max_elem"},
                               [product, constant_at(0, lead + [a[-2], b[-2]])],
                               lead + [a[-2], b[-2]])
            return reshape(apply_at({'kind': "sqrt"}, [clamped], lead + [a[-2], b[-2]]), shape)
        if plan['pairs'] is not None:
            rows, width = a
            def rows_at(picked):
                target = [len(picked), width]
                positions = torch.tensor(picked, dtype=torch.int64).reshape(-1, 1).expand(target)
                coords = _coordinates(positions, 0, a)
                return emit({'kind': "gather", 'dims': [0, 1], 'batch_dims': [],
                             'indices': {'shape': target,
                                         'values': coords.reshape(-1).tolist()}},
                            [left], target)
            if not plan['pairs']:
                return _append_node(nodes, {'parents': [], **_constant(
                    torch.zeros(0, dtype=_torch_dtype(dtype)))})
            first = rows_at([i for i, _ in plan['pairs']])
            second = rows_at([j for _, j in plan['pairs']])
            full, axis = [len(plan['pairs']), width], 1
        else:
            lead = a[:-2]
            full = lead + [a[-2], b[-2], a[-1]]
            first = _broadcast_to(nodes, apply_at(_reshape_kind(), [left],
                                                  lead + [a[-2], 1, a[-1]]), full)
            second = _broadcast_to(nodes, apply_at(_reshape_kind(), [right],
                                                   lead + [1, b[-2], b[-1]]), full)
            axis = len(full) - 1
        out = full[:axis]
        difference = apply_at({'kind': "sub"}, [first, second], full)
        def total(values):
            return apply_at({'kind': "reduce", 'op': 'sum', 'dims': [axis], 'keepdim': False},
                            [values], out)
        if p == 0:
            nonzero = apply_at({'kind': 'ne'},
                               [difference, constant_at(0, full)], full, 'bool')
            result = total(_cast_to(nodes, nonzero, dtype))
        else:
            magnitude = apply_at({'kind': "abs"}, [difference], full)
            if p == 1:
                result = total(magnitude)
            elif p == 2:
                result = apply_at({'kind': "sqrt"}, [total(apply_at(
                    {'kind': "mul_elem"}, [difference, difference], full))], out)
            elif p == float('inf'):
                result = apply_at({'kind': "reduce", 'op': 'max', 'dims': [axis],
                                   'keepdim': False}, [magnitude], out)
            else:
                powered = apply_at({'kind': 'pow'},
                                   [magnitude, constant_at(p, full)], full)
                result = apply_at({'kind': 'pow'},
                                  [total(powered), constant_at(1 / p, out)], out)
        return reshape(result, shape)
    if op == 'upsample_linear':
        # upsample_linear1d and upsample_bilinear2d: along each resized axis the entries at the two
        # source indices (constant gathers), weighted and added in the opmath type, the inner
        # axis first. The weights are PyTorch's as one-hot rows show them: at a boundary where both
        # neighbours are one index their sum. A neighbour of weight zero is read. The CPU kernel's
        # evaluation order is not reproduced, so values agree within a tolerance.
        index = parent(plan['operands'][0])
        opmath = 'float32' if dtype in _REDUCED_FLOATS else dtype
        def along(value, axis, picked):
            dim = 2 + axis
            extents = list(nodes[value]['shape'])
            target = list(extents)
            target[dim] = len(picked)
            view = [-1 if a == dim else 1 for a in range(len(extents))]
            positions = torch.tensor(picked, dtype=torch.int64).reshape(view).expand(target)
            coords = _coordinates(positions, dim, extents)
            return apply_at({'kind': "gather", 'dims': list(range(len(extents))), 'batch_dims': [],
                             'indices': {'shape': target, 'values': coords.reshape(-1).tolist()}},
                            [value], target, nodes[value]['dtype'])
        def weight(values, axis, at_shape):
            view = [len(values) if a == 2 + axis else 1 for a in range(len(at_shape))]
            node = _append_node(nodes, {'parents': [], **_constant(
                torch.tensor(values, dtype=_torch_dtype(dtype)).reshape(view))})
            return _broadcast_to(nodes, _cast_to(nodes, node, opmath), at_shape)
        def blend(value, axis):
            picks = plan['axes'][axis]
            low = along(value, axis, [i0 for i0, _, _, _ in picks])
            high = along(value, axis, [i1 for _, i1, _, _ in picks])
            at_shape = list(nodes[low]['shape'])
            low = apply_at({'kind': "mul_elem"}, [low, weight([w0 for _, _, w0, _ in picks], axis,
                                                         at_shape)], at_shape, opmath)
            high = apply_at({'kind': "mul_elem"}, [high, weight([w1 for _, _, _, w1 in picks], axis,
                                                           at_shape)], at_shape, opmath)
            return apply_at({'kind': "add"}, [low, high], at_shape, opmath)
        index = _cast_to(nodes, index, opmath)
        for axis in reversed(range(len(plan['axes']))):
            index = blend(index, axis)
        return _cast_to(nodes, index, dtype)
    if op == 'grid_sample':
        # grid_sampler_2d (GridSampler.h): grid x, y unnormalized to pixel coordinates
        # (align_corners: (g + 1) / 2 (size - 1), else ((g + 1) size - 1) / 2), clipped to the
        # image for border padding. Nearest rounds half to even, and bilinear weights the four
        # corners floor(x) + {0, 1} by the fractional parts. A corner outside the image has its
        # whole weighted contribution selected away to zero (zeros padding), its index clamped
        # into the image in binary before the int64 conversion. The corners index a runtime
        # Op.gather with the batch as a batch axis. Values agree with the CPU kernel within a
        # tolerance, and nearest picks the same pixel unless a coordinate lies within rounding of a
        # tie.
        source, grid = (parent(value) for value in plan['operands'])
        batch, channels, height, width = nodes[source]['shape']
        _, rows, cols, _ = nodes[grid]['shape']
        points = [batch, rows, cols]
        image = apply_at({'kind': "permute", 'dims': [0, 2, 3, 1]}, [source],
                         [batch, height, width, channels])
        def coordinate(component, size):
            # A NaN grid entry reads as -1, as PyTorch documents.
            value = apply_at({'kind': "slice", 'dims': [3], 'starts': [component],
                              'stops': [component + 1], 'steps': [1]}, [grid], points + [1])
            value = apply_at(_reshape_kind(), [value], points)
            missing = apply_at({'kind': 'isnan'}, [value], points, 'bool')
            value = apply_at({'kind': 'where'},
                             [missing, constant_at(-1, points), value], points)
            shifted = apply_at({'kind': "add"}, [value, constant_at(1, points)], points)
            if plan['align']:
                value = apply_at({'kind': "mul_elem"}, [apply_at({'kind': "div"}, [
                    shifted, constant_at(2, points)], points), constant_at(size - 1, points)],
                    points)
            else:
                value = apply_at({'kind': "div"}, [apply_at({'kind': "sub"}, [
                    apply_at({'kind': "mul_elem"}, [shifted, constant_at(size, points)], points),
                    constant_at(1, points)], points), constant_at(2, points)], points)
            if plan['padding'] == 1:
                value = apply_at({'kind': 'clamp'},
                                 [value, constant_at(0, points), constant_at(size - 1, points)],
                                 points)
            return value
        x, y = coordinate(0, width), coordinate(1, height)
        def corner(row, col, weight=None):
            # The pixel at (row, col), binary coordinates with integral values, times `weight`,
            # the whole contribution selected away to zero outside the image.
            inside = None
            for value, size in ((row, height), (col, width)):
                for kind, bound in (('ge', 0), ('le', size - 1)):
                    test = apply_at({'kind': kind},
                                    [value, constant_at(bound, points)], points, 'bool')
                    inside = test if inside is None else apply_at(
                        {'kind': 'logical_and'}, [inside, test], points, 'bool')
            def index_of(value, size):
                kept = apply_at({'kind': 'clamp'},
                                [value, constant_at(0, points), constant_at(size - 1, points)],
                                points)
                return _cast_to(nodes, kept, _INT64)
            tuples = _index_tuples(nodes, [index_of(row, height), index_of(col, width)], points)
            read = _registered_node(nodes, 'gather', [
                {'shape': [batch, height, width, channels]}, {'shape': points}, {'nats': [1, 2]},
                {'pairs': [[0, 0]]}, _SCALAR], [image, tuples], points + [channels], dtype)
            mask = _broadcast_to(nodes, apply_at(_reshape_kind(), [inside], points + [1], 'bool'),
                                 points + [channels])
            if weight is not None:
                read = apply_at({'kind': "mul_elem"}, [read, weight], points + [channels])
            return apply_at({'kind': 'where'},
                            [mask, read, constant_at(0, points + [channels])], points + [channels])
        def spread(value):
            return _broadcast_to(nodes, apply_at(_reshape_kind(), [value], points + [1]),
                                 points + [channels])
        if plan['mode'] == 1:
            result = corner(apply_at({'kind': 'round'}, [y], points),
                            apply_at({'kind': 'round'}, [x], points))
        else:
            west = apply_at({'kind': 'floor'}, [x], points)
            north = apply_at({'kind': 'floor'}, [y], points)
            east = apply_at({'kind': "add"}, [west, constant_at(1, points)], points)
            south = apply_at({'kind': "add"}, [north, constant_at(1, points)], points)
            right = apply_at({'kind': "sub"}, [x, west], points)
            left = apply_at({'kind': "sub"}, [east, x], points)
            down = apply_at({'kind': "sub"}, [y, north], points)
            up = apply_at({'kind': "sub"}, [south, y], points)
            result = None
            for row, col, a, b in ((north, west, up, left), (north, east, up, right),
                                   (south, west, down, left), (south, east, down, right)):
                term = corner(row, col, spread(apply_at({'kind': "mul_elem"}, [a, b], points)))
                result = term if result is None else apply_at({'kind': "add"}, [result, term],
                                                              points + [channels])
        return apply_at({'kind': "permute", 'dims': [0, 3, 1, 2]}, [result],
                        [batch, channels, rows, cols])
    if op == 'random':
        # rand is Op.randUniform under the node's key. randn is Op.randn, Random.normal in the
        # dtype's arithmetic, entry i reading the counters 2 i and 2 i + 1. randperm is the stable
        # ascending sort order of a binary64 uniform draw, equal draws in index order.
        entries = 1
        for extent in shape: entries *= extent
        sampler = plan['sampler']
        key = _random_key(len(nodes), 2 * entries if sampler == 'normal' else entries)
        if sampler != 'permutation':
            return _registered_node(nodes, 'randn' if sampler == 'normal' else 'randUniform',
                                    [{'shape': shape}, {'nat': key}], [], shape, dtype)
        draw = _registered_node(nodes, 'randUniform', [{'shape': shape}, {'nat': key}], [], shape,
                                'float64')
        return _registered_node(nodes, 'sort', [{'shape': shape}, {'nat': 0}, {'bool': False}],
                                [draw], shape, dtype, 'float64', 1)
    if op == 'dropout':
        # native_dropout_cpu: mask = bernoulli(1 - p), output = input.mul(mask).mul_(scale) with
        # scale = 1 / (1 - p), or 0 at p = 1, the scalar product in binary32 for binary16 and
        # bfloat16. The mask output is the Boolean mask.
        source = parent(plan['operands'][0])
        extents = list(nodes[source]['shape'])
        value_dtype = nodes[source]['dtype']
        if not plan['train']:
            if plan['output'] == 0: return source
            return _append_node(nodes, {'parents': [], **_constant(
                torch.ones(extents, dtype=torch.bool))})
        # Both outputs of one native_dropout read one mask node.
        mask = _DROPOUT_MASKS.get(plan['producer'])
        if mask is None:
            keep = _scalar_constant(nodes, 1.0 - plan['p'], value_dtype)
            entries = 1
            for extent in extents: entries *= extent
            mask = _registered_node(nodes, 'bernoulliMask', [
                {'shape': extents}, {'nat': _random_key(len(nodes), entries)}], [keep], extents,
                value_dtype)
            if plan['producer'] is not None:
                _DROPOUT_MASKS[plan['producer']] = mask
        if plan['output'] == 1:
            return _append_node(nodes, {'kind': 'ne', 'parents': [
                mask, _broadcast_to(nodes, _scalar_constant(nodes, 0, value_dtype), extents)],
                'shape': extents, 'dtype': 'bool'})
        kept = _append_node(nodes, {'kind': "mul_elem", 'parents': [source, mask], 'shape': extents,
                                    'dtype': value_dtype})
        opmath = 'float32' if value_dtype in _REDUCED_FLOATS else value_dtype
        scale = 0.0 if plan['p'] == 1 else 1.0 / (1.0 - plan['p'])
        wide = _cast_to(nodes, kept, opmath)
        scaled = _append_node(nodes, {'kind': "mul_elem", 'parents': [
            wide, _broadcast_to(nodes, _scalar_constant(nodes, scale, opmath), extents)],
            'shape': extents, 'dtype': opmath})
        return _cast_to(nodes, scaled, value_dtype)
    def running_count(flags, at_shape):
        # The row-major prefix count of a Boolean tensor in binary64, exact below 2 ** 53.
        total = 1
        for extent in at_shape: total *= extent
        if total >= 2 ** 53: raise NotImplementedError('a prefix count of 2 ** 53 or more entries')
        counted = _cast_to(nodes, flags, 'float64')
        counted = _append_node(nodes, {**_reshape_kind(), 'parents': [counted], 'shape': [total],
                                       'dtype': 'float64'})
        return _append_node(nodes, {'kind': "cumsum", 'dim': 0, 'parents': [counted],
                                    'shape': [total], 'dtype': 'float64'}), total
    def wide_at(kind, parents, at_shape, at_dtype='float64'):
        return _append_node(nodes, {**kind, 'parents': parents, 'shape': at_shape,
                                    'dtype': at_dtype})
    def wide_constant(value, at_shape):
        return _broadcast_to(nodes, _scalar_constant(nodes, value, 'float64'), at_shape)
    if op == 'masked_scatter':
        # self and mask broadcast to the result, and the k-th true entry in row-major order reads
        # source entry k - 1. With one dummy entry prepended to the flattened source, entry i
        # reads index where(mask, c_i, 0) for the prefix count c_i. False entries read the dummy,
        # and a count beyond the source fails the gather's range check, as PyTorch raises.
        base_value, mask_value, source_value = plan['operands']
        base, mask, source = parent(base_value), parent(mask_value), parent(source_value)
        base, mask = broadcast(base, shape), _broadcast_to(nodes, mask, shape)
        counts, total = running_count(mask, shape)
        flat_mask = _append_node(nodes, {**_reshape_kind(), 'parents': [mask], 'shape': [total],
                                         'dtype': 'bool'})
        index = wide_at({'kind': 'where'},
                        [flat_mask, counts, wide_constant(0, [total])], [total])
        index = _cast_to(nodes, index, _INT64)
        size = 1
        for extent in nodes[source]['shape']: size *= extent
        padded = emit({'kind': "cat", 'dim': 0}, [
            _broadcast_to(nodes, _scalar_constant(nodes, 0, dtype), [1]),
            reshape(source, [size])], [size + 1])
        read = _registered_node(nodes, 'gather', [{'shape': [size + 1]}, {'shape': [total]},
                                                  {'nats': [0]}, {'pairs': []}, _SCALAR],
                                [padded, index], [total], dtype)
        return emit({'kind': 'where'},
                    [mask, reshape(read, shape), base], shape)
    if op == 'nonzero_static':
        # torch.nonzero_static: the coordinates of the nonzero entries in row-major order in the
        # first `size` rows, the others fill_value. The k-th nonzero entry (k <= size) writes its
        # coordinates to row k - 1 of a binary64 zero base by Op.scatter, every other entry to a
        # dump row that is cropped. The retained rows have one update each, and coordinates and
        # counts are exact below 2 ** 53. The rows past the count take the int64 fill_value.
        source = parent(plan['operands'][0])
        extents = list(nodes[source]['shape'])
        rank, size = len(extents), plan['size']
        fill = _append_node(nodes, {'parents': [], **_constant(torch.full(
            [size, rank], plan['fill'], dtype=_torch_dtype(dtype)))})
        total = 1
        for extent in extents: total *= extent
        if total == 0 or size == 0:
            return fill
        flags = source
        if nodes[source]['dtype'] != 'bool':
            flags = _append_node(nodes, {'kind': 'ne', 'parents': [
                source, _broadcast_to(nodes, _scalar_constant(nodes, 0, nodes[source]['dtype']),
                                      extents)], 'shape': extents, 'dtype': 'bool'})
        counts, total = running_count(flags, extents)
        flat = _append_node(nodes, {**_reshape_kind(), 'parents': [flags], 'shape': [total],
                                    'dtype': 'bool'})
        within = wide_at({'kind': 'le'}, [counts, wide_constant(size, [total])],
                         [total], 'bool')
        chosen = wide_at({'kind': 'logical_and'}, [flat, within], [total],
                         'bool')
        row = wide_at({'kind': "sub"}, [counts, wide_constant(1, [total])], [total])
        row = wide_at({'kind': 'where'}, [chosen, row,
                                                           wide_constant(size, [total])], [total])
        row = _cast_to(nodes, row, _INT64)
        grids = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in extents],
                               indexing='ij')
        coordinates = torch.stack([grid.reshape(-1) for grid in grids], -1)
        coordinates = _append_node(nodes, {'parents': [], **_constant(coordinates.contiguous())})
        replace = _op_table()['codes']['scatter']['replace']
        written = _registered_node(nodes, 'scatter', [
            {'shape': [size + 1, rank]}, {'shape': [total]}, {'nat': replace},
            {'nats': [0]}, _SCALAR], [wide_constant(0, [size + 1, rank]), row, coordinates],
            [size + 1, rank], 'float64')
        kept = wide_at({'kind': "slice", 'dims': [0], 'starts': [0], 'stops': [size],
                        'steps': [1]}, [written], [size, rank])
        kept = _cast_to(nodes, kept, dtype)
        # Row r is occupied when r is below the number of nonzero entries, the last count.
        last = wide_at({'kind': "slice", 'dims': [0], 'starts': [total - 1], 'stops': [total],
                        'steps': [1]}, [counts], [1])
        rows = _append_node(nodes, {'parents': [], **_constant(
            torch.arange(size, dtype=torch.float64))})
        occupied = wide_at({'kind': 'lt'},
                           [rows, _broadcast_to(nodes, last, [size])], [size], 'bool')
        occupied = _broadcast_to(nodes, _append_node(nodes, {
            **_reshape_kind(), 'parents': [occupied], 'shape': [size, 1], 'dtype': 'bool'}),
            [size, rank])
        return _append_node(nodes, {'kind': 'where',
                                    'parents': [occupied, kept, fill], 'shape': [size, rank],
                                    'dtype': dtype})
    if op == 'embedding_bag':
        # Each bag's rows (a runtime Op.embedding, then constant slices) reduced by sum, mean
        # (the sum over the bag's size) or max, an empty bag zero. Per-sample weights multiply
        # each row before the sum. PyTorch's CPU kernel vectorizes the sums, so values agree
        # within a tolerance.
        weight_value, index_value, weights_value = plan['operands']
        weight, indices = parent(weight_value), parent(index_value)
        num, width = nodes[weight]['shape']
        count = nodes[indices]['shape'][0]
        rows = _registered_node(nodes, 'embedding', [{'nat': num}, {'nat': width},
                                                     {'shape': [count]}],
                                [weight, indices], [count, width], dtype)
        if weights_value is not None:
            weights = reshape(parent(weights_value), [count, 1])
            rows = emit({'kind': "mul_elem"}, [rows, broadcast(weights, [count, width])],
                        [count, width])
        pieces = []
        for start, stop in plan['bags']:
            if start == stop:
                pieces.append(_broadcast_to(nodes, _scalar_constant(nodes, 0, dtype), [1, width]))
                continue
            bag = crop(rows, [0], [start], [stop])
            reduced = emit({'kind': "reduce", 'op': 'max' if plan['mode'] == 2 else 'sum',
                            'dims': [0], 'keepdim': True}, [bag], [1, width])
            if plan['mode'] == 1:
                reduced = emit({'kind': "div"}, [reduced, _broadcast_to(
                    nodes, _scalar_constant(nodes, stop - start, dtype), [1, width])], [1, width])
            pieces.append(reduced)
        if len(pieces) == 1: return reshape(pieces[0], shape)
        return emit({'kind': "cat", 'dim': 0}, pieces, [len(pieces), width])
    if op == 'take_along_dim':
        source, index = (parent(value) for value in plan['operands'])
        dim = plan['dim']
        if dim is None:
            total = 1
            for extent in nodes[source]['shape']: total *= extent
            count = 1
            for extent in nodes[index]['shape']: count *= extent
            source = reshape(source, [total])
            index = _append_node(nodes, {**_reshape_kind(), 'parents': [index], 'shape': [count],
                                         'dtype': nodes[index]['dtype']})
            gathered = _registered_node(nodes, 'gather', [{'shape': [total]}, {'shape': [count]},
                                                          {'nats': [0]}, {'pairs': []}, _SCALAR],
                                        [source, index], [count], dtype)
            return reshape(gathered, shape)
        extents, positions = list(nodes[source]['shape']), list(nodes[index]['shape'])
        common = list(torch.broadcast_shapes(
            tuple(e for a, e in enumerate(extents) if a != dim),
            tuple(e for a, e in enumerate(positions) if a != dim)))
        source = broadcast(source, common[:dim] + [extents[dim]] + common[dim:])
        target = common[:dim] + [positions[dim]] + common[dim:]
        index = _broadcast_to(nodes, index, target)
        extent = _broadcast_to(nodes, _scalar_constant(nodes, extents[dim], nodes[index]['dtype']),
                               target)
        index = _append_node(nodes, {'kind': 'remainder',
                                     'parents': [index, extent], 'shape': target,
                                     'dtype': nodes[index]['dtype']})
        pairs = [[axis, axis] for axis in range(len(target)) if axis != dim]
        return _registered_node(nodes, 'gather', [{'shape': nodes[source]['shape']},
                                                  {'shape': target}, {'nats': [dim]},
                                                  {'pairs': pairs}, _SCALAR], [source, index],
                                target, dtype)
    if op == 'boolean_reduce':
        # any is ne(sum(x as binary32), 0) and all is eq(sum(not x as binary32), 0). A sum of
        # terms 0 and 1 rounded to nearest is zero exactly when every term is, and an empty fiber
        # sums to zero, so any([]) is false and all([]) is true, as in PyTorch. A floating
        # operand is first ne(x, 0), NaN true.
        index = parent(plan['operands'][0])
        source = nodes[index]['shape']
        if nodes[index]['dtype'] != 'bool':
            zero = _broadcast_to(nodes, _scalar_constant(nodes, 0, nodes[index]['dtype']), source)
            index = _append_node(nodes, {'kind': 'ne', 'parents': [index, zero],
                                         'shape': source, 'dtype': 'bool'})
        if not plan['dims']:
            # A reduction over no axis is the truth value itself.
            return reshape(index, shape)
        if plan['all']:
            index = _append_node(nodes, {'kind': 'logical_not',
                                         'parents': [index], 'shape': source, 'dtype': 'bool'})
        counted = _cast_to(nodes, index, 'float32')
        total = _append_node(nodes, {'kind': "reduce", 'op': 'sum', 'dims': plan['dims'],
                                     'keepdim': plan['keepdim'], 'parents': [counted],
                                     'shape': shape, 'dtype': 'float32'})
        zero = _broadcast_to(nodes, _scalar_constant(nodes, 0, 'float32'), shape)
        return _append_node(nodes, {'kind': ('eq' if plan['all'] else 'ne'),
                                    'parents': [total, zero], 'shape': shape, 'dtype': 'bool'})
    if op == 'col2im':
        # im2col.h's col2im starts from zero and adds the column entry (c_col, h_col, w_col) at
        # (c_col / (kh kw), h_col stride_h - pad_h + h_off dil_h, ...) in that loop order,
        # skipping positions outside the image. The scatter adds the same updates in the same
        # order into a zero canvas shifted by the padding, whose margin takes the skipped ones,
        # and a slice keeps the image.
        source = parent(plan['operands'][0])
        columns = list(nodes[source]['shape'])
        batched = len(columns) == 3
        if not batched:
            columns = [1] + columns
            source = reshape(source, columns)
        (height, width), (kh, kw) = plan['output_size'], plan['kernel']
        (dh, dw), (ph, pw), (sh, sw) = plan['dilation'], plan['padding'], plan['stride']
        if min(kh, kw, dh, dw, sh, sw) <= 0 or min(ph, pw) < 0:
            raise NotImplementedError('col2im needs positive kernel, dilation and stride and a '
                                      'nonnegative padding')
        rows = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        cols = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        batch, depth, length = columns
        if depth % (kh * kw) or rows * cols != length or rows <= 0 or cols <= 0:
            raise NotImplementedError('col2im: the columns do not match the output size')
        channels = depth // (kh * kw)
        # Every shifted position h_col stride_h + h_off dil_h lies below H + 2 pad_h.
        canvas = [batch, channels, height + 2 * ph, width + 2 * pw]
        n, c, l = torch.meshgrid(torch.arange(batch), torch.arange(depth), torch.arange(length),
                                 indexing='ij')
        coords = torch.stack([n, c // (kh * kw), (l // cols) * sh + (c // kw % kh) * dh,
                              (l % cols) * sw + (c % kw) * dw], -1)
        zero = _broadcast_to(nodes, _scalar_constant(nodes, 0, dtype), canvas)
        index = emit({'kind': "scatter", 'reduce': 'sum', 'dims': [0, 1, 2, 3],
                      'indices': {'shape': columns, 'values': coords.reshape(-1).tolist()}},
                     [zero, source], canvas)
        image = [batch, channels, height, width]
        index = crop(index, [2, 3], [ph, pw], [ph + height, pw + width])
        return reshape(index, image if batched else image[1:])
    if op == 'registered':
        # One custom node whose primitive is the registered operator of plan['record'].
        index = parent(plan['operands'][0])
        if nodes[index]['shape'] != plan['input_shape']:
            index = _append_node(nodes, {**_reshape_kind(), 'parents': [index],
                                         'shape': plan['input_shape'],
                                         'dtype': nodes[index]['dtype']})
        index = _registered_node(nodes, plan['tag'], plan['fields'], [index],
                                 plan['result_shape'], dtype, plan['scalar'], plan['output'])
        return reshape(index, shape)
    if op == 'topk':
        # ATen's partial_sort and nth_element order tied entries (equal values, -0 and +0, two
        # NaNs) either way, so the stable IR topk is ATen's result unless two of the first
        # min(k + 1, n) sorted entries tie. A tie there replaces the operand by a read at index 1
        # of a one-entry tensor, whose range check fails at execution. Otherwise `where` passes
        # the operand through.
        source = parent(plan['operands'][0])
        extents, value_dtype = list(nodes[source]['shape']), nodes[source]['dtype']
        dim, k, largest = plan['dim'], plan['k'], plan['largest']
        rank, count = len(extents), min(k + 1, extents[dim])
        def at(kind, parents, at_shape, at_dtype):
            return _append_node(nodes, {**kind, 'parents': parents, 'shape': at_shape,
                                        'dtype': at_dtype})
        if count > 1:
            head, pair, unit = list(extents), list(extents), [1] * rank
            head[dim], pair[dim] = count, count - 1
            first = at({'kind': "topk", 'dim': dim, 'k': count, 'largest': largest}, [source],
                       head, value_dtype)
            def part(start):
                return at({'kind': "slice", 'dims': [dim], 'starts': [start],
                           'stops': [start + count - 1], 'steps': [1]}, [first], pair,
                          value_dtype)
            def flag(name, parents):
                return at({'kind': name}, parents, pair, 'bool')
            left, right = part(0), part(1)
            tie = flag('logical_or', [flag('eq', [left, right]), flag('logical_and', [
                flag('isnan', [left]), flag('isnan', [right])])])
            ties = at({'kind': "reduce", 'op': 'sum', 'dims': list(range(rank)), 'keepdim': True},
                      [_cast_to(nodes, tie, 'float32')], unit, 'float32')
            tied = at({'kind': 'ne'}, [ties, _broadcast_to(
                nodes, _scalar_constant(nodes, 0, 'float32'), unit)], unit, 'bool')
            def column(value, at_dtype):
                return _broadcast_to(nodes, _scalar_constant(nodes, value, at_dtype), [1])
            position = at({'kind': 'where'}, [
                tied if rank == 1 else at(_reshape_kind(), [tied], [1], 'bool'),
                column(1, _INT64), column(0, _INT64)], [1], _INT64)
            read = _registered_node(nodes, 'gather', [{'shape': [1]}, {'shape': [1]},
                                                      {'nats': [0]}, {'pairs': []}, _SCALAR],
                                    [column(0, value_dtype), position], [1], value_dtype)
            if rank > 1:
                read = at(_reshape_kind(), [read], unit, value_dtype)
            source = at({'kind': 'where'}, [
                _broadcast_to(nodes, tied, extents), _broadcast_to(nodes, read, extents), source],
                extents, value_dtype)
        out = list(extents)
        out[dim] = k
        if plan['output'] == 0:
            return at({'kind': "topk", 'dim': dim, 'k': k, 'largest': largest}, [source], out,
                      value_dtype)
        return _registered_node(nodes, 'topk', [{'shape': extents}, {'nat': dim}, {'nat': k},
                                                {'bool': largest}], [source], out, dtype,
                                value_dtype, 1)
    if op == 'pool_indices':
        # cpu_max_pool starts each window at its first valid position with maxval = -inf and
        # takes an entry when `val > maxval || isnan(val)`: the first maximum, or the last NaN.
        # A constant gather reads each window of the flattened planes at its valid positions in
        # scan order, its padding slots repeating the last valid one. Op.argmax (the first
        # largest entry) gives the first maximum, Op.argmax of the reversed window's NaN flags
        # the last NaN, and a binary64 table, exact below 2 ** 53, the position in the plane.
        source = parent(plan['operands'][0])
        extents, value_dtype = list(nodes[source]['shape']), nodes[source]['dtype']
        kernel, stride, padding = plan['kernel'], plan['stride'], plan['padding']
        rank = len(kernel)
        lead, spatial = extents[:-rank], extents[-rank:]
        out = [(n + 2 * p - k) // s + 1 for n, k, s, p in zip(spatial, kernel, stride, padding)]
        def grid(sizes):
            points = [[]]
            for size in sizes:
                points = [point + [step] for point in points for step in range(size)]
            return points
        planes, size, volume = 1, 1, 1
        for extent in lead: planes *= extent
        for extent in spatial: size *= extent
        for extent in kernel: volume *= extent
        if planes == 0 or size == 0 or size >= 2 ** 53:
            raise NotImplementedError('max pooling indices need nonempty planes below 2 ** 53')
        windows = []
        for corner in grid(out):
            valid = []
            for offset in grid(kernel):
                point = [o * s - p + q for o, s, p, q in zip(corner, stride, padding, offset)]
                if all(0 <= c < n for c, n in zip(point, spatial)):
                    flat = 0
                    for c, n in zip(point, spatial): flat = flat * n + c
                    valid.append(flat)
            if not valid:
                raise NotImplementedError('max pooling with a window of padding only')
            windows.append(valid + [valid[-1]] * (volume - len(valid)))
        count = len(windows)
        table = torch.tensor(windows, dtype=torch.int64)
        def at(kind, parents, at_shape, at_dtype):
            return _append_node(nodes, {**kind, 'parents': parents, 'shape': at_shape,
                                        'dtype': at_dtype})
        flat = at(_reshape_kind(), [source], [planes, size], value_dtype)
        def read(columns):
            index = _append_node(nodes, {'parents': [], **_constant(columns.contiguous())})
            return _registered_node(nodes, 'gather', [{'shape': [planes, size]},
                                                      {'shape': [count, volume]},
                                                      {'nats': [1]}, {'pairs': []}, _SCALAR],
                                    [flat, index], [count, volume, planes], value_dtype)
        def first_largest(values):
            return _registered_node(nodes, 'argmax', [{'shape': [count, volume, planes]},
                                                      {'nat': 1}, {'bool': False}],
                                    [values], [count, planes], _INT64,
                                    nodes[values]['dtype'])
        reach = [count, planes]
        first = first_largest(read(table))
        flags = _cast_to(nodes, at({'kind': 'isnan'}, [read(table.flip(1))],
                                   [count, volume, planes], 'bool'), 'float32')
        last = at({'kind': "sub"}, [_broadcast_to(nodes, _scalar_constant(
            nodes, volume - 1, _INT64), reach), first_largest(flags)], reach, _INT64)
        nans = at({'kind': "reduce", 'op': 'sum', 'dims': [1], 'keepdim': False}, [flags],
                  reach, 'float32')
        has_nan = at({'kind': 'ne'}, [nans, _broadcast_to(
            nodes, _scalar_constant(nodes, 0, 'float32'), reach)], reach, 'bool')
        choice = at({'kind': 'where'}, [has_nan, last, first], reach, _INT64)
        positions = _append_node(nodes, {'parents': [], **_constant(table.double())})
        index = _registered_node(nodes, 'gather', [{'shape': [count, volume]},
                                                   {'shape': reach}, {'nats': [1]},
                                                   {'pairs': [[0, 0]]}, _SCALAR],
                                 [positions, choice], reach, 'float64')
        index = _cast_to(nodes, index, _INT64)
        index = at({'kind': "permute", 'dims': [1, 0]}, [index], [planes, count], _INT64)
        return at(_reshape_kind(), [index], lead + out, _INT64)
    if op == 'extremum':
        # max.dim keeps the first largest entry, or the first NaN, as topk with k = 1 does.
        # min.dim is the negation of max.dim of the negation, and negation is exact.
        index = parent(plan['operands'][0])
        kept = list(nodes[index]['shape'])
        kept[plan['dim']] = 1
        if not plan['largest']:
            index = emit({'kind': "neg"}, [index], nodes[index]['shape'])
        index = emit({'kind': "topk", 'dim': plan['dim'], 'k': 1, 'largest': True}, [index], kept)
        if not plan['largest']:
            index = emit({'kind': "neg"}, [index], kept)
        return reshape(index, shape)
    if op == 'leaky_relu':
        # leaky_relu_kernel gives x where x > 0 and x * slope elsewhere, with the slope and, for
        # binary16 and bfloat16, x in binary32. Rounding is monotone, so for 0 < slope <= 1 that
        # is maximum(x, x * slope) and for slope > 1 minimum(x, x * slope), at every signed zero,
        # infinity and NaN.
        opmath = 'float32' if dtype in _REDUCED_FLOATS else dtype
        index = _cast_to(nodes, parent(plan['operands'][0]), opmath)
        slope = _broadcast_to(nodes, _scalar_constant(nodes, plan['slope'], opmath), shape)
        def wide(kind, parents):
            return _append_node(nodes, {'kind': kind, 'parents': parents, 'shape': shape,
                                        'dtype': opmath})
        product = wide("mul_elem", [index, slope])
        chosen = wide("maximum" if plan['largest'] else "minimum", [index, product])
        return _cast_to(nodes, chosen, dtype)
    if op == 'scatter':
        # Each update is combined with the entry at its complete target coordinate. The targets
        # are distinct, so the IR's row-major order does not matter.
        base, update = plan['operands']
        shape_of_index = plan['indices']['shape']
        if isinstance(update, torch.fx.Node):
            updates = reshape(parent(update), plan.get('update_shape', shape_of_index))
        else:
            updates = _broadcast_to(nodes, _scalar_constant(nodes, update, dtype), shape_of_index)
        return emit({'kind': "scatter", 'reduce': plan['reduce'], 'dims': plan['dims'],
                     'indices': plan['indices']}, [parent(base), updates], shape)
    if op == 'diagonal':
        # The two axes last, merged into one of n1 * n2 entries, and the entries (i, i + offset)
        # read with step n2 + 1 from the first one.
        index = parent(plan['operands'][0])
        source, dim1, dim2 = nodes[index]['shape'], plan['dim1'], plan['dim2']
        order = [axis for axis in range(len(source)) if axis not in (dim1, dim2)] + [dim1, dim2]
        if order != list(range(len(source))):
            index = emit({'kind': "permute", 'dims': order}, [index], [source[a] for a in order])
        rows, columns, offset = source[dim1], source[dim2], plan['offset']
        rest = [source[axis] for axis in order[:-2]]
        index = reshape(index, rest + [rows * columns])
        if offset >= 0:
            length = max(0, min(rows, columns - offset))
        else:
            length = max(0, min(rows + offset, columns))
        start = (offset if offset >= 0 else -offset * columns) if length else 0
        stop = start + (length - 1) * (columns + 1) + 1 if length else 0
        index = emit({'kind': "slice", 'dims': [len(rest)], 'starts': [start], 'stops': [stop],
                      'steps': [columns + 1]}, [index], rest + [length])
        if nodes[index]['shape'] != shape:
            raise NotImplementedError('diagonal produced an unexpected shape')
        return index
    raise NotImplementedError(f'no structural lowering for {op}')


def _lower_kind(node, model=None):
    name = _target_name(node.target)
    args = node.args
    kwargs = dict(node.kwargs)
    if kwargs.get("out", None) is not None:
        raise NotImplementedError("out= mutation is outside the pure graph contract")
    axis = kwargs.get("dim", kwargs.get("axis", None))
    if axis is None and len(args) > 1: axis = args[1]
    structure = _structure_kind(node, model)
    if structure is not None: return structure
    pointwise = _pointwise_kind(node, model)
    if pointwise is not None: return pointwise
    if node.op == "call_module" and model is not None:
        mod = model.get_submodule(str(node.target))
        if isinstance(mod, nn.Linear):
            return {"kind": "linear", **_linear_payload(mod.weight, mod.bias)}
        if isinstance(mod, nn.ReLU):
            if mod.inplace: raise NotImplementedError("in-place nn.ReLU mutation is outside the pure TorchLean graph contract")
            return {"kind": "relu"}
        if isinstance(mod, nn.Tanh): return {"kind": "tanh"}
        if isinstance(mod, nn.Sigmoid): return {"kind": "sigmoid"}
        if isinstance(mod, nn.Softmax):
            return {"kind": "softmax", "dim": _normalize_axis(mod.dim, len(_node_shape(node)))}
        if isinstance(mod, nn.Flatten): return _reshape_kind()
        if isinstance(mod, nn.LayerNorm):
            rank = len(_node_shape(node))
            norm_rank = len(tuple(mod.normalized_shape))
            return _normalization("layer_norm", _argument(node, 0, 'input'), mod.eps, {'dims': list(range(max(0, rank - norm_rank), rank))},
                mod.normalized_shape, gamma=mod.weight, beta=mod.bias)
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            rank = len(tuple(mod.kernel_size))
            input_rank = len(_shape_from_arg(args[0])) if args else 0
            if input_rank not in (rank + 1, rank + 2): raise NotImplementedError(f"convolution expected rank {rank + 1} or {rank + 2}, got {input_rank}")
            channel_axis = input_rank - rank - 1
            if mod.padding_mode != "zeros":
                raise NotImplementedError(f"convolution padding_mode={mod.padding_mode!r} is "
                                          "outside the TorchLean IR import subset")
            padding = _spatial_tuple(mod.padding, rank, 0)
            input_shape = _shape_from_arg(args[0])
            return {"kind": "conv", "spatial_rank": rank, "kernel": _spatial_tuple(mod.kernel_size, rank, 1), "stride": _spatial_tuple(mod.stride, rank, 1), "padding": padding, "padding_after": padding, "dilation": _spatial_tuple(mod.dilation, rank, 1), "groups": int(mod.groups), "channel_axis": channel_axis, "input_spatial": input_shape[channel_axis + 1:], **_convolution_payload(mod.weight, mod.bias, mod.groups)}
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if mod.training:
                raise NotImplementedError("training-mode BatchNorm needs batch statistics and running-statistic updates")
            if not mod.track_running_stats or mod.running_mean is None or mod.running_var is None:
                raise NotImplementedError("nn.BatchNorm2d without running statistics has batch-dependent semantics, not TorchLean eval-mode BatchNorm semantics")
            return _normalization("batch_norm", _argument(node, 0, 'input'), mod.eps, {'dim': 1},
                [mod.num_features], gamma=mod.weight, beta=mod.bias, mean=mod.running_mean, var=mod.running_var)
        if isinstance(mod, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            rank = 1 if isinstance(mod, nn.MaxPool1d) else 2 if isinstance(mod, nn.MaxPool2d) else 3
            if _spatial_tuple(mod.dilation, rank, 1) != [1] * rank:
                raise NotImplementedError("dilated max pooling is outside the TorchLean IR "
                                          "import subset")
            if mod.ceil_mode:
                raise NotImplementedError("max pooling with ceil_mode=True is outside the "
                                          "TorchLean IR import subset")
            if mod.return_indices:
                raise NotImplementedError("max pooling with return_indices=True produces a tuple "
                                          "outside the pooling lowering")
            return {"kind": "max_pool", "spatial_rank": rank, "kernel": _spatial_tuple(mod.kernel_size, rank, 1), "stride": _spatial_tuple(mod.stride if mod.stride is not None else mod.kernel_size, rank, 1), "padding": _spatial_tuple(mod.padding, rank, 0)}
        if isinstance(mod, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            rank = 1 if isinstance(mod, nn.AvgPool1d) else 2 if isinstance(mod, nn.AvgPool2d) else 3
            if mod.ceil_mode:
                raise NotImplementedError("average pooling with ceil_mode=True is outside the "
                                          "TorchLean IR import subset")
            if getattr(mod, 'divisor_override', None) is not None:
                raise NotImplementedError("average pooling with divisor_override is outside the "
                                          "TorchLean IR import subset")
            padding = _spatial_tuple(mod.padding, rank, 0)
            if any(padding) and not mod.count_include_pad:
                raise NotImplementedError("padded average pooling with count_include_pad=False "
                                          "is outside the TorchLean IR import subset")
            return {"kind": "avg_pool", "spatial_rank": rank, "kernel": _spatial_tuple(mod.kernel_size, rank, 1), "stride": _spatial_tuple(mod.stride if mod.stride is not None else mod.kernel_size, rank, 1), "padding": padding}
    if _is_op(node, ('aten._to_copy.default', 'aten.to.dtype'), (),
              ('to', 'float', 'double', 'half', 'bfloat16')):
        return {"kind": "custom", '_plan': {'op': 'convert', 'operands': [args[0]]}}
    if _is_op(node, ("aten.add.Tensor", "aten.add.Scalar"), (operator.add, torch.add), ("add",)):
        alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
        if alpha != 1:
            raise NotImplementedError("torch.add with alpha != 1 is unsupported")
        return _arithmetic_kind(node, 'add')
    if _is_op(node, ("aten.sub.Tensor", "aten.sub.Scalar"), (operator.sub, torch.sub, torch.subtract), ("sub", "subtract")):
        alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
        if alpha != 1:
            raise NotImplementedError("torch.sub with alpha != 1 is unsupported")
        return _arithmetic_kind(node, 'sub')
    if _is_op(node, ("aten.mul.Tensor", "aten.mul.Scalar"), (operator.mul, torch.mul, torch.multiply), ("mul", "multiply")):
        return _arithmetic_kind(node, 'mul')
    if _top_k_mask(node) is not None:
        return {"kind": "top_k_mask", "k": _top_k_mask(node)[1], "dim": len(_shape_from_arg(_top_k_mask(node)[0])) - 1}
    if _is_op(node, ("aten.relu.default",), (torch.relu, F.relu), ("relu",)):
        inplace = kwargs.get("inplace", args[1] if len(args) > 1 else False)
        if inplace: raise NotImplementedError("in-place relu is outside the pure graph contract")
        return {"kind": "relu"}
    if _is_op(node, ("aten.tanh.default",), (torch.tanh,), ("tanh",)):
        return {"kind": "tanh"}
    if _is_op(node, ("aten.sigmoid.default",), (torch.sigmoid,), ("sigmoid",)):
        return {"kind": "sigmoid"}
    if _is_op(node, ("aten.exp.default",), (torch.exp,), ("exp",)):
        return {"kind": "exp"}
    if _is_op(node, ("aten.log.default",), (torch.log,), ("log",)):
        return {"kind": "log"}
    if _is_op(node, ("aten.sin.default",), (torch.sin,), ("sin",)):
        return {"kind": "sin"}
    if _is_op(node, ("aten.cos.default",), (torch.cos,), ("cos",)):
        return {"kind": "cos"}
    if _is_op(node, ("aten.abs.default",), (torch.abs, torch.absolute), ("abs", "absolute")):
        return {"kind": "abs"}
    if _is_op(node, ("aten.sqrt.default",), (torch.sqrt,), ("sqrt",)):
        source = args[0] if args else kwargs.get('input', kwargs.get('self'))
        # IEEE squareRoot: Op.sqrt at `.raw`, its policy code read from the op table.
        root_shape, root_dtype = _shape_from_arg(source), _node_dtype(source)
        if root_dtype not in _FLOATING:
            raise NotImplementedError(f'sqrt of a {root_dtype} tensor is not imported')
        fields = [{'shape': root_shape}, {'nat': _op_table()['codes']['sqrt']['raw']}]
        return _registered_kind('sqrt', fields, source, root_shape, root_shape)
    if _is_op(node, ("aten.reciprocal.default",), (torch.reciprocal,), ("reciprocal",)):
        return {"kind": "inv"}
    if _is_op(node, ("aten.maximum.default",), (torch.maximum,), ("maximum",)):
        return {"kind": "max_elem"}
    if _is_op(node, ("aten.minimum.default",), (torch.minimum,), ("minimum",)):
        return {"kind": "min_elem"}
    if _is_op(node, ("aten.matmul.default", "aten.mm.default", "aten.bmm.default"), (torch.matmul, torch.mm, torch.bmm, operator.matmul), ("matmul", "mm", "bmm")):
        return {"kind": "matmul"}
    reduction = None
    if _is_op(node, ('aten.sum.default', 'aten.sum.dim_IntList'), (torch.sum,), ('sum',)):
        reduction = 'sum'
    elif _is_op(node, ('aten.prod.default', 'aten.prod.dim_int'), (torch.prod,), ('prod',)):
        reduction = 'prod'
    elif _is_op(node, ('aten.amax.default', 'aten.max.default'), (torch.amax,), ('amax',)):
        reduction = 'max'
    elif _is_op(node, ('aten.amin.default', 'aten.min.default'), (torch.amin,), ('amin',)):
        reduction = 'min'
    elif _is_op(node, ('aten.mean.default', 'aten.mean.dim'), (torch.mean,), ('mean',)):
        reduction = 'mean'
    if reduction is not None:
        operand = _argument(node, 0, 'input', 'self')
        input_rank = len(_shape_from_arg(operand))
        total = _is_op(node, ('aten.sum.default', 'aten.prod.default', 'aten.max.default',
                              'aten.min.default', 'aten.mean.default'))
        dims = _reduction_axes(None if total else axis, input_rank, name)
        keepdim = not total and bool(_argument(node, 2, 'keepdim'))
        if kwargs.get('dtype', args[3] if len(args) > 3 and not total else None) is not None:
            raise NotImplementedError('reduction dtype casts are outside the scalar contract')
        if reduction == 'mean':
            source = _shape_from_arg(operand)
            if 0 in [source[axis] for axis in dims] and 0 not in _node_shape(node):
                raise NotImplementedError('a mean over an empty fiber is refused: PyTorch '
                                          'returns NaN, the IR mean divides the empty sum by one')
            attributes = {'kind': "mean", 'dims': dims, 'keepdim': keepdim}
        else:
            attributes = {'kind': "reduce", 'op': reduction, 'dims': dims, 'keepdim': keepdim}
        return {**attributes, '_plan': {'op': 'reduce', 'operands': [operand],
                                        'attributes': attributes}}
    if _is_op(node, ('aten.var.correction', 'aten.var.dim', 'aten.var.default'), (torch.var,),
              ('var',)):
        # var.default and var.dim take an unbiased flag, var.correction a correction count.
        if _is_op(node, ('aten.var.default',)):
            dims, keepdim, unbiased = None, False, _argument(node, 1, 'unbiased')
            correction = 0 if unbiased is False else 1
        elif _is_op(node, ('aten.var.dim',)):
            dims, keepdim = _argument(node, 1, 'dim'), _argument(node, 3, 'keepdim')
            correction = 0 if _argument(node, 2, 'unbiased') is False else 1
        elif len(args) == 2 and isinstance(args[1], bool):
            # FX and Dynamo record torch.var(x, unbiased) with var.default's positional flag.
            dims, keepdim, correction = None, False, int(args[1])
        elif len(args) > 2:
            # FX and Dynamo record torch.var(x, dim, unbiased[, keepdim]) positionally, as var.dim.
            if (len(args) > 4 or isinstance(args[1], bool)
                    or not all(isinstance(flag, bool) for flag in args[2:])):
                raise NotImplementedError(f'var reads dim, unbiased and keepdim, got {args[1:]}')
            dims, keepdim, correction = args[1], _argument(node, 3, 'keepdim'), int(args[2])
        else:
            dims, keepdim = _argument(node, 1, 'dim'), kwargs.get('keepdim')
            correction = kwargs.get('correction')
            if 'unbiased' in kwargs: correction = 1 if kwargs['unbiased'] else 0
        correction = 1 if correction is None else correction
        if correction != int(correction) or correction < 0:
            raise NotImplementedError(f'var needs a natural correction, got {correction}')
        operand = _argument(node, 0, 'input', 'self')
        dims = _reduction_axes(dims, len(_shape_from_arg(operand)), name)
        return {'kind': "mean", '_plan': {'op': 'var', 'operands': [operand], 'dims': dims,
                                          'correction': int(correction),
                                          'keepdim': bool(keepdim)}}
    if _is_op(node, ("aten.softmax.int", "aten._softmax.default", "aten.log_softmax.int",
                     "aten._log_softmax.default"),
              (torch.softmax, F.softmax, torch.log_softmax, F.log_softmax),
              ("softmax", "log_softmax")):
        logarithm = _is_op(node, ("aten.log_softmax.int", "aten._log_softmax.default"),
                           (torch.log_softmax, F.log_softmax), ("log_softmax",))
        if _is_op(node, ("aten._softmax.default", "aten._log_softmax.default")):
            if kwargs.get("half_to_float", args[2] if len(args) > 2 else False):
                raise NotImplementedError("softmax half_to_float changes scalar semantics")
            dtype = None
        else:
            dtype_pos = 3 if node.target in (F.softmax, F.log_softmax) else 2
            dtype = kwargs.get("dtype", args[dtype_pos] if len(args) > dtype_pos else None)
        if dtype is not None:
            raise NotImplementedError("softmax dtype casts are outside the scalar contract")
        rank = len(_node_shape(node))
        if axis is None and node.target in (F.softmax, F.log_softmax):
            axis = 0 if rank in (0, 1, 3) else 1
        if axis is None: raise NotImplementedError("softmax requires a static dimension")
        return {"kind": "log_softmax" if logarithm else "softmax",
                "dim": _normalize_axis(axis, rank)}
    if _is_op(node, ("aten.reshape.default", "aten.view.default"), (torch.reshape,),
              ("reshape", "view")):
        if node.op == "call_method" and node.target == "view":
            if isinstance(kwargs.get("dtype", args[1] if len(args) > 1 else None), torch.dtype):
                raise NotImplementedError("view(dtype) changes scalar semantics")
        return _reshape_kind()
    if _is_op(node, ("aten.contiguous.default",), (), ("contiguous",)):
        return _reshape_kind()
    if _is_op(node, ("aten.flatten.using_ints",), (torch.flatten,), ("flatten",)):
        return _reshape_kind()
    if _is_op(node, ("aten.permute.default",), (torch.permute,), ("permute",)):
        dims = kwargs.get("dims", args[1] if len(args) > 1 else ())
        perm = list(dims) if isinstance(dims, (tuple, list)) else list(args[1:])
        rank = len(_node_shape(node))
        return {"kind": "permute", "dims": [_normalize_axis(x, rank) for x in perm]}
    if _is_op(node, ("aten.cat.default",), (torch.cat, torch.concat, torch.concatenate), ()):
        rank = len(_node_shape(node))
        dim = _normalize_axis(kwargs.get("dim", args[1] if len(args) > 1 else 0), rank)
        return {"kind": "cat", "dim": dim}
    if _is_op(node, ("aten.slice.Tensor",)):
        source = _shape_from_arg(args[0])
        dim = args[1] if len(args) > 1 else kwargs.get('dim', 0)
        dim = _normalize_axis(dim, len(source))
        start = args[2] if len(args) > 2 else kwargs.get('start', None)
        end = args[3] if len(args) > 3 else kwargs.get('end', None)
        step = args[4] if len(args) > 4 else kwargs.get('step', 1)
        if step is None: step = 1
        if step < 1:
            raise NotImplementedError('a slice needs a positive step')
        start, end, _ = slice(start, end).indices(source[dim])
        return {"kind": "slice", "dims": [dim], "starts": [start], "stops": [max(start, end)],
                "steps": [step]}
    if _is_op(node, ("aten.transpose.int",), (torch.transpose,), ("transpose",)):
        rank = len(_shape_from_arg(args[0]))
        d0 = _normalize_axis(kwargs.get("dim0", args[1] if len(args) > 1 else None), rank)
        d1 = _normalize_axis(kwargs.get("dim1", args[2] if len(args) > 2 else None), rank)
        dims = list(range(rank))
        dims[d0], dims[d1] = dims[d1], dims[d0]
        return {"kind": "permute", "dims": dims}
    if _is_op(node, ("aten.flip.default",), (torch.flip,), ("flip",)):
        rank = len(_shape_from_arg(args[0]))
        dims = kwargs.get("dims", args[1] if len(args) > 1 else ())
        # Tensor.flip(*dims): FX and Dynamo record x.flip(0, 1) with one argument per dim.
        dims = list(args[1:]) if len(args) > 2 else dims
        dims = dims if isinstance(dims, (tuple, list)) else [dims]
        if any(isinstance(d, (tuple, list)) for d in dims):
            raise NotImplementedError(f'flip reads one int per dim or one list, got {args[1:]}')
        dims = [_normalize_axis(d, rank) for d in dims]
        if len(set(dims)) != len(dims):
            raise NotImplementedError(f'flip lists dims {dims} more than once')
        return {"kind": "flip", "dims": dims}
    if _is_op(node, ("aten.constant_pad_nd.default", "aten.pad.default"), (F.pad,), ()):
        padding = [int(p) for p in (args[1] if len(args) > 1 else kwargs.get('pad'))]
        if _is_op(node, ("aten.constant_pad_nd.default",)):
            mode, value = 'constant', _argument(node, 2, 'value')
        else:
            mode, value = _argument(node, 2, 'mode') or 'constant', _argument(node, 3, 'value')
        value = 0 if value is None else value
        if mode == 'constant':
            if len(padding) % 2 or len(padding) // 2 > len(_shape_from_arg(args[0])):
                raise NotImplementedError(f'padding {padding} does not pair with the axes')
            return {"kind": "pad", '_plan': {'op': 'pad', 'operands': [args[0]],
                                            'padding': padding, 'value': value}}
        if mode == 'reflect':
            return {"kind": "cat", '_plan': {'op': 'reflect', 'operands': [args[0]],
                                            'padding': padding}}
        if mode == 'replicate':
            return {"kind": "cat", '_plan': {'op': 'replicate', 'operands': [args[0]],
                                            'padding': padding}}
        raise NotImplementedError(f'pad mode {mode!r} is not imported')

    if _is_op(node, ("aten.layer_norm.default",), (F.layer_norm, torch.layer_norm),
              ()):
        weight = kwargs.get("weight", args[2] if len(args) > 2 else None)
        bias = kwargs.get("bias", args[3] if len(args) > 3 else None)
        eps = kwargs.get("eps", args[4] if len(args) > 4 else 1e-5)
        input_rank = len(_shape_from_arg(args[0])) if args else len(_node_shape(node))
        normalized_shape = args[1] if len(args) > 1 else kwargs.get("normalized_shape", ())
        normalized_rank = len(normalized_shape) if isinstance(normalized_shape,
                                                              (tuple, list, torch.Size)) else 1
        return _normalization(
            "layer_norm", _argument(node, 0, 'input'), eps,
            {"dims": list(range(max(0, input_rank - normalized_rank), input_rank))},
            [normalized_shape] if isinstance(normalized_shape, int) else normalized_shape,
            gamma=weight, beta=bias)
    if _is_op(node, ("aten.linear.default",), (F.linear,), ()):
        weight = kwargs.get("weight", args[1] if len(args) > 1 else None)
        bias = kwargs.get("bias", args[2] if len(args) > 2 else None)
        return {"kind": "linear", **_linear_payload(weight, bias)}
    if _is_op(node, ("aten.batch_norm.default",), (F.batch_norm, torch.batch_norm), ()):
        training = bool(kwargs.get("training", args[5] if len(args) > 5 else False))
        is_aten = node.target is not F.batch_norm
        mean_pos, var_pos = (3, 4) if is_aten else (1, 2)
        weight_pos, bias_pos = (1, 2) if is_aten else (3, 4)
        if training:
            # Batch statistics. The running statistics, updated in place, must be buffers that
            # only this node reads: their updates become outputs of the program
            # (_program_outputs), not a mutation.
            running = [kwargs.get(name, args[pos] if len(args) > pos else None)
                       for name, pos in (("running_mean", mean_pos), ("running_var", var_pos))]
            eps = kwargs.get("eps", args[7] if len(args) > 7 else 1e-5)
            weight = kwargs.get("weight", args[weight_pos] if len(args) > weight_pos else None)
            bias = kwargs.get("bias", args[bias_pos] if len(args) > bias_pos else None)
            if all(value is None for value in running):
                return _moments_kind(args[0], 'batch', weight, bias, eps, 0)
            if not all(isinstance(value, torch.fx.Node) and value.op == 'placeholder' and
                       value.name in _BUFFERS and set(value.users) == {node}
                       for value in running) or running[0] is running[1]:
                raise NotImplementedError("training-mode batch_norm updates its running "
                                          "statistics in place: they must be two buffers that "
                                          "only this batch_norm reads")
            if not node.users:
                raise NotImplementedError("training-mode batch_norm whose result is unused")
            momentum = kwargs.get("momentum", args[6] if len(args) > 6 else 0.1)
            return _moments_kind(args[0], 'batch', weight, bias, eps, 0,
                                 running=(running[0], running[1], momentum), record=True)
        running_mean = kwargs.get("running_mean", args[mean_pos] if len(args) > mean_pos else None)
        running_var = kwargs.get("running_var", args[var_pos] if len(args) > var_pos else None)
        if running_mean is None or running_var is None:
            raise NotImplementedError("batch_norm without running statistics is outside the "
                                      "eval-mode TorchLean IR operation")
        shape = _node_shape(node)
        channels = int(shape[1]) if len(shape) >= 2 else 0
        eps = kwargs.get("eps", args[7] if len(args) > 7 else 1e-5)
        weight = kwargs.get("weight", args[weight_pos] if len(args) > weight_pos else None)
        bias = kwargs.get("bias", args[bias_pos] if len(args) > bias_pos else None)
        return _normalization("batch_norm", _argument(node, 0, 'input'), eps, {"dim": 1},
                              [channels], gamma=weight, beta=bias, mean=running_mean,
                              var=running_var)

    if _is_op(node, ("aten.conv1d.default", "aten.conv2d.default", "aten.conv3d.default"), (F.conv1d, F.conv2d, F.conv3d), ()):
        weight_arg = kwargs.get("weight", args[1] if len(args) > 1 else None)
        bias_arg = kwargs.get("bias", args[2] if len(args) > 2 else None)
        weight = _resolve_tensor(weight_arg)
        wshape = _shape_of(weight)
        rank = max(0, len(wshape) - 2)
        input_rank = len(_shape_from_arg(args[0])) if args else 0
        if input_rank not in (rank + 1, rank + 2): raise NotImplementedError(f"convolution expected rank {rank + 1} or {rank + 2}, got {input_rank}")
        channel_axis = input_rank - rank - 1
        stride = _spatial_tuple(kwargs.get("stride", args[3] if len(args) > 3 else 1), rank, 1)
        padding = _spatial_tuple(kwargs.get("padding", args[4] if len(args) > 4 else 0), rank, 0)
        dilation = _spatial_tuple(kwargs.get("dilation", args[5] if len(args) > 5 else 1), rank, 1)
        groups = int(kwargs.get("groups", args[6] if len(args) > 6 else 1))
        input_shape = _shape_from_arg(args[0])
        return {"kind": "conv", "spatial_rank": rank, "kernel": [int(v) for v in wshape[2:]], "stride": stride, "padding": padding, "padding_after": padding, "dilation": dilation, "groups": groups, "channel_axis": channel_axis, "input_spatial": input_shape[channel_axis + 1:], **_convolution_payload(weight, bias_arg, groups)}
    if _is_op(node, ("aten.max_pool1d.default", "aten.max_pool2d.default", "aten.max_pool3d.default"), (F.max_pool1d, F.max_pool2d, F.max_pool3d), ()):
        rank = next(d for d in (1, 2, 3) if _is_op(node, (f"aten.max_pool{d}d.default",), (getattr(F, f"max_pool{d}d"),)))
        kernel = _spatial_tuple(args[1] if len(args) > 1 else kwargs.get("kernel_size", 1), rank, 1)
        stride_arg = kwargs.get("stride", args[2] if len(args) > 2 else None)
        stride = _spatial_tuple(stride_arg if stride_arg not in (None, []) else kernel, rank, 1)
        padding = _spatial_tuple(kwargs.get("padding", args[3] if len(args) > 3 else 0), rank, 0)
        dilation = _spatial_tuple(kwargs.get("dilation", args[4] if len(args) > 4 else 1), rank, 1)
        ceil_mode = bool(kwargs.get("ceil_mode", args[5] if len(args) > 5 else False))
        if kwargs.get("return_indices", args[6] if len(args) > 6 else False):
            raise NotImplementedError("max pooling indices need tuple lowering")
        if dilation != [1] * rank:
            raise NotImplementedError("dilated max pooling is outside the TorchLean IR import "
                                      "subset")
        if ceil_mode:
            raise NotImplementedError("max pooling with ceil_mode=True is outside the TorchLean "
                                      "IR import subset")
        return {"kind": "max_pool", "spatial_rank": rank, "kernel": kernel, "stride": stride, "padding": padding}
    if _is_op(node, ("aten.avg_pool1d.default", "aten.avg_pool2d.default", "aten.avg_pool3d.default"), (F.avg_pool1d, F.avg_pool2d, F.avg_pool3d), ()):
        rank = next(d for d in (1, 2, 3) if _is_op(node, (f"aten.avg_pool{d}d.default",), (getattr(F, f"avg_pool{d}d"),)))
        kernel = _spatial_tuple(args[1] if len(args) > 1 else kwargs.get("kernel_size", 1), rank, 1)
        stride_arg = kwargs.get("stride", args[2] if len(args) > 2 else None)
        stride = _spatial_tuple(stride_arg if stride_arg not in (None, []) else kernel, rank, 1)
        padding = _spatial_tuple(kwargs.get("padding", args[3] if len(args) > 3 else 0), rank, 0)
        ceil_mode = bool(kwargs.get("ceil_mode", args[4] if len(args) > 4 else False))
        count_include_pad = bool(kwargs.get("count_include_pad", args[5] if len(args) > 5 else True))
        divisor_override = kwargs.get("divisor_override", args[6] if len(args) > 6 else None)
        if ceil_mode:
            raise NotImplementedError("average pooling with ceil_mode=True is outside the "
                                      "TorchLean IR import subset")
        if any(padding) and not count_include_pad:
            raise NotImplementedError("padded average pooling with count_include_pad=False is "
                                      "outside the TorchLean IR import subset")
        if divisor_override is not None:
            raise NotImplementedError("average pooling with divisor_override is outside the "
                                      "TorchLean IR import subset")
        return {"kind": "avg_pool", "spatial_rank": rank, "kernel": kernel, "stride": stride, "padding": padding}
    if _is_getattr(node):
        attr = args[1] if len(args) > 1 else "<unknown>"
        raise NotImplementedError(f"unsupported PyTorch attribute projection: {attr}. If this came from a tuple-returning op such as torch.sort(...).values, add a value-graph/lowering rule for that producer instead of treating the attribute as an ordinary tensor op.")
    raise NotImplementedError(f"unsupported PyTorch op: {name}")
