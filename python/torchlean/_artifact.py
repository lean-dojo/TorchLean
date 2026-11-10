"""The typed artifact of a capture's internal nodes: one lowering for the PyTorch and JAX captures.

`lower(nodes, input_ids, output_ids, table)` returns the artifact, the `[step, output]` reference of
each node's value and the node each step comes from. `table` is Lean's op table, as
`op_table(checker)` returns it for the capture's checker. It gives the format name, the record
version, the JSON of each dtype, the codec codes (those of the bitcast words and the cast sources
included) and the typed form of every pointwise operator.

A node is an input, a constant, a custom node (a conversion, a bitcast, a call or a registered
operator record), an `indices` node, a pointwise operator of the table, or an operator of the
capture vocabulary that `_project` names. A step's scalar is the DType JSON of the binary format it
runs at when one of its ports carries values (`Program.Node.ports`), and null otherwise. The record
of an operator generalized over its element type ends with an `element` field, "scalar" at a binary
format. A record at Booleans or an integer type carries that element in such a field, and Lean
accepts or refuses the operator there when `Artifact.load` checks every step again.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable

_MASK64 = (1 << 64) - 1

Refuse = Callable[[str], str]


def _integer(dtype: Any) -> tuple[bool, int] | None:
    """The signedness and width of an integer element type of positive width, else None."""
    if isinstance(dtype, dict) and dtype.get('kind') == 'integer':
        width = dtype.get('width')
        if isinstance(width, int) and width > 0:
            return bool(dtype.get('signed')), width
    return None


def _binary(dtype: Any, table: dict) -> dict | None:
    """The DType JSON of a binary format, named in `table` or given as its own JSON, else None."""
    if isinstance(dtype, str):
        value = table['dtypes'].get(dtype)
        return dict(value) if isinstance(value, dict) and value.get('kind') == 'binary' else None
    if isinstance(dtype, dict) and dtype.get('kind') == 'binary':
        return dict(dtype)
    return None


def _name(dtype: Any) -> str | None:
    """The op table's name of an element type: a name as given, an integer type by its width."""
    if isinstance(dtype, str):
        return dtype
    integer = _integer(dtype)
    if integer is not None:
        return f"{'int' if integer[0] else 'uint'}{integer[1]}"
    return None


def _describe(dtype: Any) -> str:
    if isinstance(dtype, dict) and dtype.get('kind') == 'integer':
        width = dtype.get('width')
        if width is None:
            return 'Int' if dtype.get('signed') else 'Nat'
        return f"{'int' if dtype.get('signed') else 'uint'}{width}"
    return str(dtype)


# The element field of a record at a binary format, `Op.Element.scalar`.
_SCALAR = {'element': 'scalar'}


def _element(dtype: Any, table: dict) -> Any:
    """The element of a record at `dtype`: Booleans, an integer type, the scalar, or None."""
    if dtype == 'bool':
        return 'bool'
    integer = _integer(dtype)
    if integer is not None:
        return {'integer': {'signed': integer[0], 'width': integer[1]}}
    if _binary(dtype, table) is not None:
        return 'scalar'
    return None


def _port(dtype: Any, shape: list[int], table: dict) -> dict | None:
    """The port of a value of element type `dtype` at `shape` (`Program.Typed.ofDType?`)."""
    if dtype == 'bool':
        return {'bool': list(shape)}
    integer = _integer(dtype)
    if integer is not None:
        return {'integer': {'shape': list(shape), 'signed': integer[0], 'width': integer[1]}}
    if _binary(dtype, table) is not None:
        return {'value': list(shape)}
    return None


def _carries(port: dict) -> bool:
    """Whether a port carries its step's scalar: a value, frozen or detached port."""
    return any(key in port for key in ('value', 'frozen', 'detached'))


def _op(tag: str, fields: list, inputs: list, table: dict) -> dict:
    return {'op': {'inputs': [list(r) for r in inputs],
                   'record': {'version': table['version'], 'tag': tag, 'fields': fields}}}


def _size(shape: list[int]) -> int:
    total = 1
    for extent in shape:
        total *= extent
    return total


def _axis(a: Any, rank: int) -> int:
    """Axis `a` of a shape of rank `rank`, a negative axis counted from the last one."""
    a = int(a)
    return a + rank if -rank <= a < 0 else a


def _dims(name: str, shape: list[int], dims: list) -> list[int]:
    """`IR.dimsOf`: distinct dimensions of `shape`, negative ones counted from the end."""
    given = [int(a) for a in dims]
    dims = [_axis(a, len(shape)) for a in given]
    if any(a < 0 or a >= len(shape) for a in dims) or len(set(dims)) != len(dims):
        raise NotImplementedError(f'{name}: dimensions {given} are not distinct dimensions of '
                                  f'{list(shape)}')
    return dims


def _dim(name: str, shape: list[int], dim: Any) -> int:
    return _dims(name, shape, [dim])[0]


def _matmul(sa: list[int], sb: list[int]) -> tuple:
    """`IR.inferMatmulDims`: (leftLeading, rightLeading, leading, rows, inner, cols, lv, rv)."""
    a, b = list(sa), list(sb)
    if len(a) >= 2 and len(b) >= 2:
        if a[-1] != b[-2]:
            raise NotImplementedError(f'matmul: inner dims mismatch: {a[-1]} vs {b[-2]}')
        if a[:-2] == b[:-2]:
            lead = a[:-2]
            return lead, lead, lead, a[-2], a[-1], b[-1], False, False
        ra, rb, batch = a[:-2][::-1], b[:-2][::-1], []
        for j in range(max(len(ra), len(rb))):
            if j >= len(ra):
                batch.append(rb[j])
            elif j >= len(rb):
                batch.append(ra[j])
            elif ra[j] == rb[j] or rb[j] == 1:
                batch.append(ra[j])
            elif ra[j] == 1:
                batch.append(rb[j])
            else:
                raise NotImplementedError(f'matmul: incompatible batch dimensions {ra[j]} and '
                                          f'{rb[j]}')
        return a[:-2], b[:-2], batch[::-1], a[-2], a[-1], b[-1], False, False
    if len(a) == 1 and len(b) >= 2:
        if a[0] != b[-2]:
            raise NotImplementedError(f'matmul: inner dims mismatch: {a[0]} vs {b[-2]}')
        return [], b[:-2], b[:-2], 1, a[0], b[-1], True, False
    if len(a) >= 2 and len(b) == 1:
        if a[-1] != b[0]:
            raise NotImplementedError(f'matmul: inner dims mismatch: {a[-1]} vs {b[0]}')
        return a[:-2], [], a[:-2], a[-2], a[-1], 1, False, True
    if len(a) == 1 and len(b) == 1:
        if a[0] != b[0]:
            raise NotImplementedError(f'matmul: inner dims mismatch: {a[0]} vs {b[0]}')
        return [], [], [], 1, a[0], 1, True, True
    raise NotImplementedError(f'matmul: expected non-scalar inputs, got {a} and {b}')


def _matmul_out(dims: tuple) -> list[int]:
    _, _, lead, rows, _, cols, lv, rv = dims
    return lead + ([] if lv else [rows]) + ([] if rv else [cols])


def _splitmix64(x: int) -> int:
    """`Random.splitmix64` on 64-bit words."""
    z = (x + 0x9e3779b97f4a7c15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & _MASK64
    return z ^ (z >> 31)


# The capture vocabulary's operators of one shape: the kind and its tag.
_UNARY = {'abs': 'abs', 'inv': 'reciprocal', 'relu': 'relu', 'tanh': 'tanh',
          'sigmoid': 'sigmoid', 'exp': 'exp', 'log': 'log', 'sin': 'sin', 'cos': 'cos',
          'detach': 'detach'}
_BINARY = {'add': 'add', 'sub': 'sub', 'mul_elem': 'mul', 'max_elem': 'max', 'min_elem': 'min'}
_ALONG = {'log_softmax': 'logSoftmax', 'cumsum': 'cumsum', 'cumprod': 'cumprod',
          'cummax': 'cummax', 'cummin': 'cummin'}

# The kinds `_project` names. They win over a pointwise operator of the same name.
_PROJECTED = frozenset(_UNARY) | frozenset(_BINARY) | frozenset(_ALONG) | frozenset({
    'sqrt', 'contract', 'reshape', 'broadcast', 'permute', 'slice', 'pad', 'cat', 'flip', 'gather',
    'scatter', 'reduce', 'mean', 'matmul', 'linear', 'softmax', 'sort', 'layer_norm', 'batch_norm',
    'top_k_mask', 'topk', 'conv', 'max_pool', 'avg_pool'})


def _words(node: dict, name: str, refuse: Refuse) -> list[int]:
    words = node.get(f'{name}_bits')
    if not isinstance(words, list):
        raise NotImplementedError(refuse(f'no {name} payload'))
    return [int(w) for w in words]


def _contract(node: dict, s: list[int], t: list[int], out: list[int], refuse: Refuse,
              located: Refuse) -> tuple:
    """The batch axes, the other axes of `s`, then the other axes of `t`, as `Shape.contract`
    computes them."""
    dims = [[_axis(a, len(s)), _axis(b, len(t))] for a, b in node.get('dims', [])]
    batch = [[_axis(a, len(s)), _axis(b, len(t))] for a, b in node.get('batch_dims', [])]
    pairs = batch + dims
    lefts, rights = [a for a, _ in pairs], [b for _, b in pairs]
    if (any(not 0 <= a < len(s) for a in lefts) or any(not 0 <= b < len(t) for b in rights)
            or len(set(lefts)) != len(lefts) or len(set(rights)) != len(rights)
            or any(s[a] != t[b] for a, b in pairs)):
        raise NotImplementedError(located(f'contract: pairs {dims} and batch pairs {batch} are '
                                          f'not valid for {s} and {t}'))
    shape = ([s[a] for a, _ in batch] + [e for a, e in enumerate(s) if a not in lefts]
             + [e for b, e in enumerate(t) if b not in rights])
    if shape != out:
        raise NotImplementedError(refuse(f'the contracted shape is not the declared {out}'))
    return ('contract', [{'shape': s}, {'shape': t}, {'pairs': dims}, {'pairs': batch}],
            [('parent', 0, s), ('parent', 1, t)], 0, shape)


def _project(node: dict, hints: list, refuse: Refuse, located: Refuse, table: dict) -> tuple:
    """The registered operator of a node of the capture vocabulary with parent shapes `hints`, as
    (tag, fields, feeds, keep, out). A feed is ('parent', j, shape) or ('const', port, words)."""
    kind, out = node['kind'], list(node['shape'])
    ps = [list(h) for h in hints]
    codes = table['codes']
    # The last field of a generalized operator: the node's element, "scalar" at a binary format.
    element = {'element': _element(node.get('dtype'), table)}

    def parents(count: int) -> None:
        if len(ps) != count:
            raise NotImplementedError(refuse('unsupported node'))

    if kind in _UNARY:
        parents(1)
        return _UNARY[kind], [{'shape': out}], [('parent', 0, out)], 0, out
    if kind == 'sqrt':
        # IEEE squareRoot, the codec's `Op.sqrt .raw`.
        parents(1)
        return ('sqrt', [{'shape': out}, {'nat': codes['sqrt']['raw']}], [('parent', 0, out)], 0,
                out)
    if kind in _BINARY:
        parents(2)
        return (_BINARY[kind], [{'shape': out}, element], [('parent', 0, out), ('parent', 1, out)],
                0, out)
    if kind == 'contract':
        parents(2)
        return _contract(node, ps[0], ps[1], out, refuse, located)
    if kind == 'reshape':
        parents(1)
        if _size(ps[0]) != _size(out):
            raise NotImplementedError(refuse(f'reshape: {ps[0]} has {_size(ps[0])} entries, '
                                             f'{out} has {_size(out)}'))
        return ('reshape', [{'shape': ps[0]}, {'shape': out}, element], [('parent', 0, ps[0])], 0,
                out)
    if kind == 'broadcast':
        parents(1)
        dims = [_axis(a, len(out)) for a in node['dims']]
        return ('broadcast', [{'shape': ps[0]}, {'shape': out}, {'nats': dims}, element],
                [('parent', 0, ps[0])], 0, out)
    if kind == 'permute':
        parents(1)
        dims = [_axis(a, len(ps[0])) for a in node['dims']]
        if sorted(dims) != list(range(len(ps[0]))):
            raise NotImplementedError(refuse(f'permute: invalid permutation {dims} for shape '
                                             f'{ps[0]}'))
        permuted = [ps[0][a] for a in dims]
        return ('permute', [{'shape': ps[0]}, {'nats': dims}, {'shape': permuted}, element],
                [('parent', 0, ps[0])], 0, permuted)
    if kind == 'slice':
        parents(1)
        dims = _dims('slice', ps[0], node['dims'])
        starts, stops, steps = ([int(a) for a in node[key]] for key in ('starts', 'stops', 'steps'))
        return ('slice', [{'shape': ps[0]}, {'nats': dims}, {'nats': starts}, {'nats': stops},
                          {'nats': steps}, element], [('parent', 0, ps[0])], 0, out)
    if kind == 'pad':
        # The second parent is the scalar padding value.
        parents(2)
        s = ps[0]
        low, high, interior = ([int(a) for a in node[key]] for key in ('low', 'high', 'interior'))
        for values in (low, high, interior):
            if len(values) != len(s):
                raise NotImplementedError(located(f'pad: expected {len(s)} values, one per '
                                                  f'dimension of {s}, got {values}'))
        padded = [lo + hi + n + max(n - 1, 0) * k for lo, hi, n, k in zip(low, high, s, interior)]
        if padded != out:
            raise NotImplementedError(refuse(f'the padded shape is not the declared {out}'))
        return ('pad', [{'shape': s}, {'nats': low}, {'nats': high}, {'nats': interior}, element],
                [('parent', 0, s), ('parent', 1, [])], 0, out)
    if kind == 'cat':
        if not ps:
            raise NotImplementedError(refuse('unsupported node'))
        dim = _dim('cat', ps[0], node['dim'])
        sizes = [p[dim] if dim < len(p) else 0 for p in ps]
        feeds = [('parent', j, ps[0][:dim] + [sizes[j]] + ps[0][dim + 1:]) for j in range(len(ps))]
        return 'cat', [{'shape': ps[0]}, {'nat': dim}, {'nats': sizes}, element], feeds, 0, out
    if kind == 'flip':
        parents(1)
        dims = _dims('flip', ps[0], node['dims'])
        return 'flip', [{'shape': ps[0]}, {'nats': dims}, element], [('parent', 0, ps[0])], 0, out
    if kind == 'gather':
        parents(1)
        s = ps[0]
        dims = _dims('gather', s, node['dims'])
        index = node['indices']
        t = [int(a) for a in index['shape']]
        batch = [[_axis(a, len(s)), _axis(b, len(t))] for a, b in node.get('batch_dims', [])]
        port = {'index': {'bounds': [s[a] for a in dims], 'shape': t}}
        return ('gather', [{'shape': s}, {'shape': t}, {'nats': dims}, {'pairs': batch}, element],
                [('parent', 0, s), ('const', port, [int(w) for w in index['values']])], 0, out)
    if kind == 'scatter':
        parents(2)
        s = ps[0]
        rule = node.get('reduce')
        if rule not in codes['scatter']:
            raise NotImplementedError(refuse('unsupported node'))
        dims = _dims('scatter', s, node['dims'])
        index = node['indices']
        t = [int(a) for a in index['shape']]
        port = {'index': {'bounds': [s[a] for a in dims], 'shape': t}}
        return ('scatter', [{'shape': s}, {'shape': t}, {'nat': codes['scatter'][rule]},
                            {'nats': dims}, element],
                [('parent', 0, s), ('const', port, [int(w) for w in index['values']]),
                 ('parent', 1, ps[1])], 0, out)
    if kind == 'reduce':
        parents(1)
        s = ps[0]
        op = node.get('op')
        if op not in codes['reduce']:
            raise NotImplementedError(refuse('unsupported node'))
        dims = _dims('reduce', s, node['dims'])
        if op in ('max', 'min') and any(s[a] == 0 for a in dims):
            raise NotImplementedError(refuse(f'reduce: {op} over an empty dimension'))
        return ('reduce', [{'shape': s}, {'nat': codes['reduce'][op]}, {'nats': dims},
                           {'bool': bool(node['keepdim'])}], [('parent', 0, s)], 0, out)
    if kind == 'mean':
        parents(1)
        s = ps[0]
        dims = _dims('mean', s, node['dims'])
        if _size([s[a] for a in dims]) == 0 and _size(out) != 0:
            # The IR mean divides an empty sum by one, where 0 / 0 is NaN.
            raise NotImplementedError(refuse('a mean over an empty fiber is refused: PyTorch and '
                                             'JAX return NaN, the IR mean divides the empty sum '
                                             'by one'))
        return ('mean', [{'shape': s}, {'nats': dims}, {'bool': bool(node['keepdim'])}],
                [('parent', 0, s)], 0, out)
    if kind == 'matmul':
        parents(2)
        dims = _matmul(ps[0], ps[1])
        left, right, lead, rows, inner, cols, lv, rv = dims
        return ('matmul', [{'shape': left}, {'shape': right}, {'shape': lead}, {'nat': rows},
                           {'nat': inner}, {'nat': cols}, {'bool': lv}, {'bool': rv}],
                [('parent', 0, ps[0]), ('parent', 1, ps[1])], 0, _matmul_out(dims))
    if kind == 'linear':
        parents(1)
        s = ps[0]
        if not s:
            raise NotImplementedError(refuse('unsupported lowering (a linear node of a scalar '
                                             'input)'))
        lead, in_dim = s[:-1], s[-1]
        out_dim = out[-1] if out else 0
        dtype = node['dtype']
        return ('linear', [{'shape': lead}, {'nat': in_dim}, {'nat': out_dim}],
                [('const', _port(dtype, [out_dim, in_dim], table), _words(node, 'weight', refuse)),
                 ('const', _port(dtype, [out_dim], table), _words(node, 'bias', refuse)),
                 ('parent', 0, s)], 0, lead + [out_dim])
    if kind == 'softmax':
        parents(1)
        s = ps[0]
        dim = _dim('softmax', s, node['dim'])
        mask = node.get('mask')
        if mask is None:
            return 'softmax', [{'shape': s}, {'nat': dim}], [('parent', 0, s)], 0, s
        shape, allowed = [int(e) for e in mask['shape']], list(mask['allowed'])
        if shape != s:
            raise NotImplementedError(located(f'hard mask: expected shape {s}, but the payload '
                                              f'has shape {shape}'))
        if len(allowed) != _size(s):
            raise NotImplementedError(located(f'hard mask: shape {s} requires {_size(s)} entries, '
                                              f'but the payload contains {len(allowed)}'))
        return ('maskedSoftmax', [{'shape': s}, {'nat': dim}],
                [('parent', 0, s), ('const', {'bool': s}, [1 if a else 0 for a in allowed])], 0, s)
    if kind in _ALONG or kind == 'sort':
        parents(1)
        s = ps[0]
        dim = _dim(kind, s, node['dim'])
        fields = [{'shape': s}, {'nat': dim}]
        if kind == 'sort':
            fields.append({'bool': bool(node['descending'])})
        return _ALONG.get(kind, 'sort'), fields, [('parent', 0, s)], 0, s
    if kind == 'layer_norm':
        parents(1)
        s = ps[0]
        dims = _dims('layer_norm', s, node['dims'])
        if dims != [len(s) - 1] or _size(s[:-1]) <= 0 or s[-1] <= 0:
            raise NotImplementedError(refuse('unsupported lowering (a layer norm other than over '
                                             'the last axis with positive extents)'))
        last, dtype = s[-1], node['dtype']
        frozen = _port(dtype, [], table)
        frozen = {'frozen': frozen['value']} if frozen and 'value' in frozen else frozen
        return ('layerNorm', [{'nat': _size(s[:-1])}, {'nat': last}, {'shape': s[:-1]}],
                [('parent', 0, s),
                 ('const', _port(dtype, [last], table), _words(node, 'gamma', refuse)),
                 ('const', _port(dtype, [last], table), _words(node, 'beta', refuse)),
                 ('const', frozen, _words(node, 'eps', refuse))], 0, s)
    if kind == 'batch_norm':
        parents(1)
        s = ps[0]
        dim = _dim('batch_norm', s, node['dim'])
        channels, dtype = s[dim], node['dtype']
        value = _port(dtype, [channels], table)
        frozen = {'frozen': [channels]} if value and 'value' in value else value
        eps = {'frozen': []} if value and 'value' in value else _port(dtype, [], table)
        return ('batchNormRunning', [{'shape': s}, {'nat': dim}],
                [('parent', 0, s), ('const', value, _words(node, 'gamma', refuse)),
                 ('const', value, _words(node, 'beta', refuse)),
                 ('const', frozen, _words(node, 'mean', refuse)),
                 ('const', frozen, _words(node, 'var', refuse)),
                 ('const', eps, _words(node, 'eps', refuse))], 0, s)
    if kind == 'top_k_mask':
        parents(1)
        s = ps[0]
        dim = _dim('top_k_mask', s, node['dim'])
        return ('topKMask', [{'shape': s}, {'nat': dim}, {'nat': int(node['k'])}],
                [('parent', 0, s)], 0, s)
    if kind == 'topk':
        parents(1)
        s = ps[0]
        dim, k = _dim('topk', s, node['dim']), int(node['k'])
        if k > s[dim]:
            raise NotImplementedError(refuse(f"topk: k = {k} exceeds dimension {dim} of {s}"))
        kept = s[:dim] + [k] + s[dim + 1:]
        return ('topk', [{'shape': s}, {'nat': dim}, {'nat': k}, {'bool': bool(node['largest'])}],
                [('parent', 0, s)], 0, kept)
    if kind == 'conv':
        parents(1)
        s = ps[0]
        r, axis = int(node['spatial_rank']), int(node['channel_axis'])
        spatial = s[axis + 1:]
        if len(spatial) != r:
            raise NotImplementedError(refuse('unsupported lowering (a conv whose spatial axes do '
                                             'not match its kernel)'))
        lead, groups = s[:axis], int(node['groups'])
        in_c, out_c = int(node['in_channels']), int(node['out_channels'])
        kernel = [int(a) for a in node['kernel']]
        if groups <= 0 or out_c % groups != 0:
            raise NotImplementedError(refuse('unsupported lowering (a conv whose groups do not '
                                             'divide its output channels)'))
        packed = [out_c, in_c // groups] + kernel
        dtype = node['dtype']
        # `Conv.packedKernel`: output channel o reads the input channels of its group.
        full, cells, width = _words(node, 'weight', refuse), _size(kernel), in_c // groups
        weights = [full[(o * in_c + o // (out_c // groups) * width + c) * cells + k]
                   for o in range(out_c) for c in range(width) for k in range(cells)]
        return ('conv', [{'shape': [_size(lead), in_c] + spatial}, {'shape': packed}, {'nat': r},
                         {'nats': list(range(r + 2))}, {'nats': list(range(r + 2))},
                         {'nats': [int(a) for a in node['stride']]},
                         {'nats': [int(a) for a in node['padding']]},
                         {'nats': [int(a) for a in node['padding_after']]},
                         {'nats': [int(a) for a in node['dilation']]}, {'nat': groups},
                         {'shape': lead}],
                [('const', _port(dtype, packed, table), weights),
                 ('const', _port(dtype, [out_c], table), _words(node, 'bias', refuse)),
                 ('parent', 0, s)], 0, out)
    if kind in ('max_pool', 'avg_pool'):
        parents(1)
        s = ps[0]
        r = int(node['spatial_rank'])
        if r <= 0 or r > len(s):
            raise NotImplementedError(refuse(f'{kind}: kernel rank {r} exceeds input rank '
                                             f'{len(s)}'))
        padding = [int(a) for a in node['padding']]
        code = codes['pool']['max' if kind == 'max_pool' else 'avg']
        return ('pool', [{'shape': s}, {'nat': code},
                         {'nats': list(range(len(s) - r, len(s)))},
                         {'nats': [int(a) for a in node['kernel']]},
                         {'nats': [int(a) for a in node['stride']]}, {'nats': padding},
                         {'nats': padding}, {'nats': [1] * r}, {'bool': False}],
                [('parent', 0, s)], 0, out)
    raise NotImplementedError(refuse('unsupported node'))


def _index_ports(record: dict) -> tuple[dict, dict]:
    """The index ports of a registered operator the captures record: ({input: (bounds, shape)},
    {output: (bounds, shape)})."""
    tag, fields = record['tag'], [list(f.values())[0] for f in record['fields']]
    if tag == 'embedding':
        return {1: ([fields[0]], fields[2])}, {}
    if tag in ('gather', 'scatter'):
        s, t, dims = fields[0], fields[1], fields[3] if tag == 'scatter' else fields[2]
        return {1: ([s[a] for a in dims], t)}, {}
    if tag in ('argmax', 'argmin'):
        s, dim, keep = fields[0], fields[1], fields[2]
        out = [1 if a == dim else e for a, e in enumerate(s)] if keep else s[:dim] + s[dim + 1:]
        return {}, {0: ([s[dim]], out)}
    if tag in ('sort', 'cummax', 'cummin'):
        s, dim = fields[0], fields[1]
        return {}, {1: ([s[dim]], s)}
    if tag == 'topk':
        s, dim, k = fields[0], fields[1], fields[2]
        return {}, {1: ([s[dim]], s[:dim] + [k] + s[dim + 1:])}
    return {}, {}


def _bit_width(dtype: Any, table: dict) -> int | None:
    integer = _integer(dtype)
    if integer is not None:
        return integer[1]
    binary = _binary(dtype, table)
    if binary is not None:
        return 1 + int(binary['exponent_bits']) + int(binary['fraction_bits'])
    return None


def _bitcast(source: Any, target: Any, s: list[int], table: dict) -> list | None:
    """The bitcast records from `source` to `target` at `s`, each with the type it runs at, or
    None where the widths differ or a side has no words. The codes of the sides (`Op.Word`: the
    scalar, an unsigned and a signed integer) are the table's `codes.word`."""
    width, other = _bit_width(source, table), _bit_width(target, table)
    if width is None or width != other:
        return None
    codes = table['codes']['word']

    def word(dtype: Any) -> int:
        integer = _integer(dtype)
        if integer is None:
            return codes['scalar']
        return codes['signed'] if integer[0] else codes['unsigned']

    def record(a: int, b: int) -> list:
        return [{'shape': list(s)}, {'nat': width}, {'nat': a}, {'nat': b}]

    if _integer(source) is not None and _integer(target) is not None:
        return [(record(word(source), word(target)), source)]
    if _binary(source, table) is not None and _integer(target) is not None:
        return [(record(codes['scalar'], word(target)), source)]
    if _integer(source) is not None and _binary(target, table) is not None:
        return [(record(word(source), codes['scalar']), target)]
    if _binary(source, table) is not None and _binary(target, table) is not None:
        # Two binary formats of one width meet at their unsigned words.
        return [(record(codes['scalar'], codes['unsigned']), source),
                (record(codes['unsigned'], codes['scalar']), target)]
    return None


def _entry(entries: list, node: dict) -> dict | None:
    """The table entry whose keyword the node carries: [key, value] matches node[key] == value,
    and a null keyword matches where the other entries' keys are absent, None or "none"."""
    keys = {entry['keyword'][0] for entry in entries if entry['keyword'] is not None}
    for entry in entries:
        keyword = entry['keyword']
        if keyword is None:
            if all(node.get(key) in (None, 'none') for key in keys):
                return entry
        elif node.get(keyword[0]) == keyword[1]:
            return entry
    return None


def _shaped(record: dict, s: list[int]) -> dict:
    """A table record with its shape placeholders set to the node's shape `s`."""
    record = copy.deepcopy(record)
    record['fields'] = [{'shape': list(s)} if field == {'shape': []} else field
                        for field in record['fields']]
    return record


def _sentinel(record: dict) -> tuple[str, str]:
    """A record's tag and fields with every shape cleared. Two records that agree on it have
    ports of the same sorts."""
    fields = [{'shape': []} if 'shape' in field else field for field in record['fields']]
    return record['tag'], json.dumps(fields, sort_keys=True)


def _at_scalar(record: dict) -> bool:
    """Whether a record is at the scalar: each of its `element` fields is "scalar"."""
    return all(field['element'] == 'scalar' for field in record['fields'] if 'element' in field)


def _scalarless(table: dict) -> frozenset:
    """The sentinels of the records without a value, frozen or detached port, read off the forms
    of `table` that run at Booleans or an integer type. Such a form books each step at that type,
    so a step of it at the scalar carries no scalar."""
    found = set()
    for entries in table['pointwise'].values():
        for entry in entries:
            for results in entry['forms'].values():
                for form in results.values():
                    if _binary(form['at'], table) is not None:
                        continue
                    for item in form['steps']:
                        record = item['record']
                        if _at_scalar(record):
                            found.add(_sentinel(record))
    return frozenset(found)


def _booked(record: dict, at: Any, table: dict, scalarless: frozenset) -> dict | None:
    """The scalar of an operator step that runs at `at`: its DType JSON where the record has a
    value, frozen or detached port, else None."""
    if not _at_scalar(record) or _sentinel(record) in scalarless:
        return None
    return _binary(at, table)


def _block(nodes: list, input_ids: list[int], refs: list, base: int, i: int, node: dict,
           table: dict, scalarless: frozenset) -> tuple[list, list[int]]:
    """The steps of node `i`, the first at `base`, as (step, scalar, words), and the reference of
    its value."""
    d, s, kind = node.get('dtype'), list(node['shape']), node['kind']

    def refuse(what: str) -> str:
        return f'IR import: node {i}: {what} ({kind})'

    def located(message: str) -> str:
        return f'IR eval: node {i}: {message} ({kind})'

    for name in ('weight', 'bias', 'gamma', 'beta', 'mean', 'var', 'eps'):
        if f'{name}_dtype' in node and node[f'{name}_dtype'] != d:
            raise NotImplementedError(f"IR types: node {i}: {name}_dtype has "
                                      f"{_describe(node[f'{name}_dtype'])}, expected "
                                      f"{_describe(d)}")
    port = _port(d, s, table)
    if port is None:
        raise NotImplementedError(refuse(f'no port carries {_describe(d)}'))
    # `_port` gives a value port exactly at a binary format, so an input, a constant, a
    # conversion or a call of this node carries `scalar` exactly when it is not None.
    scalar = _binary(d, table)

    def ref(j: int) -> list[int]:
        if not (0 <= j < len(refs)):
            raise NotImplementedError(f'IR import: node {i}: parent {j} is not earlier')
        return refs[j]

    parents = [int(j) for j in node.get('parents', [])]
    if kind == 'input':
        if i not in input_ids:
            raise NotImplementedError(refuse('an input not in the arguments'))
        return [({'input': {'index': input_ids.index(i), 'port': port}}, scalar, None)], [base, 0]
    if kind == 'const':
        return [({'const': port}, scalar, [int(w) for w in node['bits']])], [base, 0]
    if kind == 'custom':
        primitive = node.get('primitive')
        if primitive == 'cast':
            if not parents:
                raise NotImplementedError(refuse('a cast without an operand'))
            step = {'cast': {'rounding': node['rounding'], 'source': list(ref(parents[0])),
                             'port': port}}
            return [(step, scalar, None)], [base, 0]
        if primitive == 'bitcast':
            if not parents:
                raise NotImplementedError(refuse('a bitcast without an operand'))
            previous = list(ref(parents[0]))
            ops = _bitcast(nodes[parents[0]].get('dtype'), d, s, table)
            if ops is None:
                raise NotImplementedError(refuse('a bitcast between types of other widths'))
            steps = []
            for fields, at in ops:
                steps.append((_op('bitcast', fields, [previous], table), _binary(at, table), None))
                previous = [base + len(steps) - 1, 0]
            return steps, previous
        if primitive == 'call':
            args = [list(ref(j)) for j in parents]
            signature = []
            for j in parents:
                typed = _port(nodes[j].get('dtype'), nodes[j]['shape'], table)
                if typed is None:
                    raise NotImplementedError(refuse(f'no port carries the argument {j}'))
                signature.append({'port': typed, 'scalar': _binary(nodes[j].get('dtype'), table)})
            name = node.get('name')
            if isinstance(name, str):
                name = name.split('.')
            if not isinstance(name, list) or not name:
                raise NotImplementedError(refuse('a call without a name'))
            step = {'call': {'name': [str(c) for c in name], 'inputs': args,
                             'signature': signature, 'port': port}}
            return [(step, scalar, None)], [base, 0]
        if primitive != 'op':
            raise NotImplementedError(refuse('a custom node without a primitive'))
        record, k = node['op'], int(node.get('output', 0))
        value = _binary(node.get('scalar'), table)
        if value is None:
            raise NotImplementedError(refuse(f"an operator record at "
                                             f"{_describe(node.get('scalar'))}, which is not a "
                                             'binary format'))
        index_in, index_out = _index_ports(record)
        steps, operands = [], []
        for position, j in enumerate(parents):
            operand = list(ref(j))
            if position in index_in:
                bounds, shape = index_in[position]
                integer = _integer(nodes[j].get('dtype'))
                if integer is None:
                    raise NotImplementedError(
                        f'IR import: node {i}: an index operand of element type '
                        f'{_describe(nodes[j].get("dtype"))}')
                steps.append((_op('decodeIndex', [{'shape': shape}, {'nats': bounds},
                                                  {'bool': integer[0]}, {'nat': integer[1]}],
                                  [operand], table), None, None))
                operands.append([base + len(steps) - 1, 0])
            else:
                operands.append(operand)
        if record.get('tag') == 'custom':
            raise NotImplementedError(refuse('an operator record of Op.custom'))
        at = base + len(steps)
        steps.append(({'op': {'inputs': operands, 'record': copy.deepcopy(record)}},
                      _booked(record, value, table, scalarless), None))
        if k in index_out:
            bounds, shape = index_out[k]
            integer = _integer(d)
            if integer is None:
                raise NotImplementedError(refuse(f'an index result of element type '
                                                 f'{_describe(d)}'))
            steps.append((_op('encodeIndex', [{'shape': shape}, {'nats': bounds},
                                              {'bool': integer[0]}, {'nat': integer[1]}],
                              [[at, k]], table), None, None))
            return steps, [at + 1, 0]
        return steps, [at, k]
    if kind == 'indices':
        # The coordinate along `dim` of each entry, cast from its index to the scalar.
        if parents:
            raise NotImplementedError(refuse('indices with parents'))
        a = _axis(node['dim'], len(s))
        if not 0 <= a < len(s):
            raise NotImplementedError(located(f"indices: dimension {node['dim']} is out of range "
                                              f'for shape {s}'))
        if scalar is None:
            raise NotImplementedError(refuse(f'indices of element type {_describe(d)}, which is '
                                             'not a binary format'))
        # The table's `codes.cast.index` is the cast source code of `.index 0`. Indices below
        # `bound` have that code plus `bound`.
        code = table['codes']['cast']['index'] + s[a]
        return [(_op('indices', [{'shape': s}, {'nat': a}], [], table), None, None),
                (_op('cast', [{'shape': s}, {'nat': code}], [[base, 0]], table), scalar,
                 None)], [base + 1, 0]
    entries = None if kind in _PROJECTED else table['pointwise'].get(kind)
    if entries is not None:
        args = [list(ref(j)) for j in parents]
        operand = nodes[parents[0]].get('dtype') if parents else d
        missing = refuse(f'{kind} has no implementation for {_describe(d)}')
        entry = _entry(entries, node)
        if entry is None or len(parents) != entry['arity']:
            raise NotImplementedError(missing)
        form = entry['forms'].get(_name(operand) or '', {}).get(_name(d) or '')
        if form is None:
            raise NotImplementedError(missing)
        steps = []
        if form['zero']:
            zero = _port(operand, s, table)
            steps.append(({'const': zero}, _binary(operand, table) if _carries(zero) else None,
                          [0] * _size(s)))
        first = base + len(steps)
        for item in form['steps']:
            inputs = []
            for arg in item['args']:
                if arg[0] == 'operand':
                    inputs.append(list(args[arg[1]]))
                elif arg[0] == 'zero':
                    inputs.append([base, 0])
                else:
                    inputs.append([first + arg[1], 0])
            record = _shaped(item['record'], s)
            steps.append(({'op': {'inputs': inputs, 'record': record}},
                          _booked(record, form['at'], table, scalarless), None))
        return steps, [base + len(steps) - 1, 0]
    for j in parents:
        ref(j)
    hints = [list(nodes[j]['shape']) for j in parents]
    tag, fields, feeds, keep, out = _project(node, hints, refuse, located, table)
    for f in feeds:
        if f[0] == 'parent' and hints[f[1]] != list(f[2]):
            raise NotImplementedError(f'IR lower: node {i}: parent {parents[f[1]]} has shape '
                                      f'{hints[f[1]]}, its port has shape {f[2]} ({kind})')
    if list(out) != s:
        raise NotImplementedError(f'IR lower: node {i}: the kept output has shape {list(out)}, '
                                  f'not the declared {s} ({kind})')
    for j in parents:
        if nodes[j].get('dtype') != d:
            raise NotImplementedError(refuse(f'parent {j} has another element type'))
    element = _element(d, table)
    if element != 'scalar':
        # Lean accepts or refuses the operator at this element when it loads the artifact. A
        # record without an `element` field gets one, which Lean's decoder refuses. Its value
        # constants hold that element, and index and mask constants keep their ports.
        if not any('element' in field for field in fields):
            fields = fields + [{'element': element}]
        feeds = [('const', _port(d, list(f[1].values())[0], table), f[2])
                 if f[0] == 'const' and _carries(f[1]) else f for f in feeds]
    steps, inputs = [], []
    for f in feeds:
        if f[0] == 'parent':
            inputs.append(list(ref(parents[f[1]])))
        else:
            inputs.append([base + len(steps), 0])
            steps.append(({'const': f[1]}, scalar if _carries(f[1]) else None, f[2]))
    at = base + len(steps)
    steps.append((_op(tag, fields, inputs, table), scalar, None))
    return steps, [at, keep]


def lower(nodes: list, input_ids: list[int], output_ids: list[int],
          table: dict) -> tuple[dict, list, list[int]]:
    """The typed artifact of the internal nodes under the op table `table`, the `[step, output]`
    of each node's value, and the internal node whose block wrote each step."""
    scalarless = _scalarless(table)
    steps, scalars, constants, refs, origin = [], [], [], [], []
    input_ids = [int(j) for j in input_ids]
    for i, node in enumerate(nodes):
        if node['id'] != i:
            raise NotImplementedError(f"IR import: node {i} has id {node['id']}")
        base = len(steps)
        block, ref = _block(nodes, input_ids, refs, base, i, node, table, scalarless)
        if not (base <= ref[0] < base + len(block)):
            raise NotImplementedError(f'IR import: node {i}: its value is not read from its steps')
        for step, scalar, words in block:
            steps.append(step)
            scalars.append(scalar)
            constants.append(words)
            origin.append(i)
        refs.append(list(ref))
    outputs = []
    for j in output_ids:
        if not (0 <= j < len(refs)):
            raise NotImplementedError(f'IR import: output {j} is no node')
        outputs.append(refs[j])
    artifact = {'format': table['format'], 'nodes': steps, 'outputs': outputs, 'scalars': scalars,
                'constants': constants}
    return artifact, refs, origin
