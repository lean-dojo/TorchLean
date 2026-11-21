"""torch.export's control-flow operators, lowered to Lean's flows (`_flow`).

torch.export records `torch.cond(pred, true_fn, false_fn, operands)` as
`higher_order.cond(pred, true_graph, false_graph, operands)`, `while_loop(cond_fn, body_fn,
carried, additional)` (torch._higher_order_ops.while_loop) as `higher_order.while_loop` of the
same arguments with the functions' graphs, `map(f, xs, *args)` (torch._higher_order_ops.map) as
`higher_order.map_impl(body_graph, xs, args)`, and a call of a function marked by
`mark_compile_region` (torch._higher_order_ops.invoke_subgraph) as
`higher_order.invoke_subgraph(graph, identifier, *operands)`. Each graph is a `get_attr` of a
GraphModule, the placeholders of a loop's graphs are its carried values, or its mapped slices, and
then its other operands, and each result is read through `operator.getitem` (PyTorch 2.11 and 2.13).

`lower` runs the graphs of one such node through the capture's walk (`export._document`) in
regions of their own (`Regions`), and `bound` scopes the capture's tables, which name values by
their nodes' names, to one graph. The main declarations are `hop`, `regions`, `selected`,
`bound`, `draws`, `Regions` and `lower`.
"""

from __future__ import annotations

import contextlib
from operator import attrgetter

import torch

from . import _flow
from ._artifact import _element
from ._fx import _INT64, _TENSOR_VALUES, _dtype_of, _is_op, _op_table, _shape_of
from ._pointwise import _append_node, _argument, _broadcast_to, _scalar_constant
from ._structure import (_DROPOUT_MASKS, _DROPOUTS, _FOLDED, _SAMPLERS, _index_constant,
                         _registered_node)

_HOPS = ('cond', 'while_loop', 'map_impl', 'invoke_subgraph')


def hop(node):
    """The name of the control-flow operator of `_HOPS` that `node` calls, else None."""
    if node.op != 'call_function':
        return None
    return next((name for name in _HOPS
                 if node.target is getattr(torch.ops.higher_order, name, None)), None)


def subgraph(graph, node):
    """The GraphModule that the `get_attr` node `node` of `graph` reads, else None."""
    if node.op != 'get_attr' or graph.owning_module is None:
        return None
    value = graph.owning_module
    for part in str(node.target).split('.'):
        value = getattr(value, part, None)
    return value if isinstance(value, torch.fx.GraphModule) else None


def tensor_attributes(graph, module):
    """Tensor constants read by this graph's get_attr nodes."""
    for node in graph.nodes:
        if node.op == 'get_attr' and subgraph(graph, node) is None:
            value = attrgetter(str(node.target))(module)
            if isinstance(value, torch.Tensor):
                yield node.name, value


def regions(graph, node):
    """The graphs of the control-flow node `node` of `graph`, each with the outer nodes its
    placeholders read in order, None for a loop's carried value or mapped slice: a cond's false
    graph, then its true graph, a while loop's test, then its body. An operand that is not a node
    of `graph`, such as a Python number that a while loop carries, is refused."""
    name, args = hop(node), node.args
    module = lambda value: subgraph(graph, value)
    looped = []
    if name == 'cond':
        _, true, false, operands = args
        pairs = [(module(false), list(operands)), (module(true), list(operands))]
    elif name == 'while_loop':
        test, body, looped, additional = args
        operands = [None] * len(looped) + list(additional)
        pairs = [(module(test), operands), (module(body), operands)]
    elif name == 'map_impl':
        body, looped, extra = args
        pairs = [(module(body), [None] * len(looped) + list(extra))]
    else:
        pairs = [(module(args[0]), list(args[2:]))]
    for graph_module, operands in pairs:
        if graph_module is None:
            raise NotImplementedError(f'{name}: a graph that is not a GraphModule')
        if not all(value is None or isinstance(value, torch.fx.Node)
                   for value in [*operands, *looped]):
            raise NotImplementedError(f'{name}: an operand that is not a tensor of the graph')
    return pairs


def selected(graph, node):
    """The graphs of `regions(graph, node)` that the capture checks and lowers: for a cond whose
    predicate is known at capture (`_structure._index_constant`), the one graph that the
    predicate selects, so the other, which never runs, may hold what the capture refuses, as an
    unselected branch of a JAX cond with a known index may; for any other node, all of them."""
    pairs = regions(graph, node)
    if hop(node) == 'cond':
        known = _index_constant(node.args[0])
        if known is not None:
            return [pairs[int(bool(known))]]
    return pairs


@contextlib.contextmanager
def bound(graph_module, operands):
    """Scope the capture's tables to the graph of `graph_module`, whose placeholders read
    `operands` (`regions`): `_fx._TENSOR_VALUES` holds the tensor of each placeholder whose
    operand is a lifted tensor and of each tensor `get_attr`, and no other name of the graph,
    and `_structure._FOLDED` the folded value of each placeholder's operand, None for a carried
    value or a slice. Both are restored afterwards, and the dropout masks of the graph's nodes
    (`_structure._DROPOUT_MASKS`) are dropped, so a graph read twice is read afresh and its
    dropout draws a mask of its own each time."""
    graph = graph_module.graph
    placeholders = [node for node in graph.nodes if node.op == 'placeholder']
    if len(placeholders) != len(operands):
        raise NotImplementedError('a control-flow graph whose placeholders are not its operands')
    saved = dict(_TENSOR_VALUES)
    held = {}
    for placeholder, operand in zip(placeholders, operands):
        if operand is not None and operand.name in saved:
            held[placeholder.name] = saved[operand.name]
    held.update(tensor_attributes(graph, graph_module))
    folded = [_index_constant(operand) for operand in operands]
    for node in graph.nodes:
        _TENSOR_VALUES.pop(node.name, None)
    _TENSOR_VALUES.update(held)
    for placeholder, value in zip(placeholders, folded):
        _FOLDED[placeholder] = value
    try:
        yield
    finally:
        _TENSOR_VALUES.clear()
        _TENSOR_VALUES.update(saved)
        for node in graph.nodes:
            _FOLDED.pop(node, None)
            _DROPOUT_MASKS.pop(node, None)


def draws(graph_module):
    """Whether a node of the graph of `graph_module`, or of a graph that one of its nodes reads,
    draws from TorchLean's generator: rand, randn or randperm (`_structure._SAMPLERS`), or dropout
    in training (`_structure._DROPOUTS`; in evaluation it draws nothing)."""
    graph = graph_module.graph
    return any(_is_op(node, _SAMPLERS) or
               _is_op(node, _DROPOUTS) and bool(_argument(node, 2, 'train')) or
               (module := subgraph(graph, node)) is not None and draws(module)
               for node in graph.nodes)


class Regions:
    """The region (`_flow`) that the capture's walk appends its internal nodes to. `flush` adds
    the nodes appended since the last one to the region, except inputs, and `region` gives the
    nodes and flows of a build in a region of its own. The caches of `caches`, which reuse a
    node, are restored after a region, whose nodes run only where it runs."""

    def __init__(self, nodes, caches):
        self.nodes, self.caches = nodes, caches
        self.items, self.cursor = [], 0

    def flush(self):
        self.items += [index for index in range(self.cursor, len(self.nodes))
                       if self.nodes[index]['kind'] != 'input']
        self.cursor = len(self.nodes)

    def region(self, build):
        self.flush()
        outer, self.items = self.items, []
        saved = [dict(cache) for cache in self.caches]
        try:
            results = build()
            self.flush()
            return self.items, results
        finally:
            self.items = outer
            for cache, copy in zip(self.caches, saved):
                cache.clear()
                cache.update(copy)

    def add(self, flow):
        self.flush()
        self.items.append(flow)

    def slot(self, like):
        # A node of node `like`'s type that no block computes: a carried value of a loop inside
        # it, or a result of a flow after it.
        source = self.nodes[like]
        return _append_node(self.nodes, {'kind': 'input', 'parents': [],
                                         'shape': list(source['shape']), 'dtype': source['dtype']})


def lower(graph, node, node_to_id, regions_, walk):
    """The internal nodes of the results of the control-flow node `node` of `graph`, in order.
    `walk(graph_module, operands, ids)` lowers a graph whose placeholders read `operands`
    (`regions`) at the internal nodes `ids` and gives its results, appending to `regions_`.

    `cond` is Lean's `Flow.cond` on the predicate's truth, a Boolean as it is and another value
    compared with zero, as Python's `if pred` reads a one-element tensor: branch 0 is the false
    graph and branch 1, or an index outside [0, 2), the true graph, so only the selected graph
    runs. A predicate known at capture selects its graph there (`selected`), and only that
    graph is checked and lowered. `while_loop` is `Flow.whileLoop`, which runs the body while
    the test's Boolean scalar holds, as `while_loop`'s docstring loops, and fails where it still
    holds after the artifact's fuel of iterations. `map_impl` is `Flow.scan` over the leading
    axis of its mapped operands: the state carries a counter `i` from 0 and the stacked outputs,
    which start at zeros, and iteration `i` reads entry `i` of each mapped operand by
    `Op.gather` and writes its outputs there by `Op.scatter` (`_entry`). A map whose graph draws
    (`draws`) is unrolled instead, iteration `i` reading and writing entry `i` at the constant
    `i`, so that each iteration's draws have keys of their own, as the JAX importer unrolls a scan
    whose body draws; a flow's region is one set of nodes, so a while loop's body draws the same
    entries in each iteration. `invoke_subgraph` is its graph, inlined."""
    nodes, name = regions_.nodes, hop(node)
    pairs = selected(graph, node)
    ids = lambda operands: [node_to_id[value] for value in operands]
    if name == 'invoke_subgraph' or name == 'cond' and len(pairs) == 1:
        # One graph that runs once, inlined: a subgraph's, or the one that a cond's predicate
        # known at capture selects.
        (module, operands), = pairs
        return walk(module, operands, ids(operands))
    if name == 'cond':
        index = node_to_id[node.args[0]]
        if nodes[index]['shape']:
            index = _append_node(nodes, {'kind': 'reshape', 'parents': [index], 'shape': [],
                                         'dtype': nodes[index]['dtype']})
        if nodes[index]['dtype'] != 'bool':
            zero = _scalar_constant(nodes, 0, nodes[index]['dtype'])
            index = _append_node(nodes, {'kind': 'ne', 'parents': [index, zero], 'shape': [],
                                         'dtype': 'bool'})
        built = [regions_.region(lambda module=module, operands=operands:
                                 walk(module, operands, ids(operands)))
                 for module, operands in pairs]
        branches, results = zip(*built)
        outs = [regions_.slot(j) for j in results[-1]]
        regions_.add(_flow.Cond(index, list(branches), list(results), outs))
        return outs
    if name == 'while_loop':
        carried = list(node.args[2])
        init = ids(carried)
        slots = [regions_.slot(j) for j in init]
        (test_module, operands), (body_module, _) = pairs
        inputs = slots + ids(operands[len(carried):])
        test, results = regions_.region(lambda: walk(test_module, operands, inputs))
        if len(results) != 1 or nodes[results[0]]['dtype'] != 'bool' or \
                nodes[results[0]]['shape']:
            raise NotImplementedError('while_loop: a test other than a Boolean scalar')
        body, new = regions_.region(lambda: walk(body_module, operands, inputs))
        outs = [regions_.slot(j) for j in init]
        regions_.add(_flow.Loop(slots, init, body, new, outs, test, results[0]))
        return outs
    (module, operands), = pairs
    xs = node.args[1]
    length = _shape_of(xs[0].meta.get('val'))[0]  # at least 1: PyTorch refuses an empty map
    stacked = []
    for value in node.meta['val']:
        dtype = _dtype_of(value)
        stacked.append(_broadcast_to(nodes, _scalar_constant(nodes, 0, dtype),
                                     _shape_of(value)))
    if draws(module):
        for i in range(length):
            at = _scalar_constant(nodes, i, _INT64)
            ys = walk(module, operands, [_entry(nodes, j, at) for j in ids(xs)] +
                      ids(operands[len(xs):]))
            stacked = [_entry(nodes, buffer, at, y) for buffer, y in zip(stacked, ys, strict=True)]
        return stacked
    init = [_scalar_constant(nodes, 0, _INT64), *stacked]
    slots = [regions_.slot(j) for j in init]

    def iteration():
        counter = slots[0]
        entries = [_entry(nodes, j, counter) for j in ids(xs)]
        ys = walk(module, operands, entries + ids(operands[len(xs):]))
        written = [_entry(nodes, buffer, counter, y)
                   for buffer, y in zip(slots[1:], ys, strict=True)]
        one = _scalar_constant(nodes, 1, _INT64)
        step = _append_node(nodes, {'kind': 'add', 'parents': [counter, one], 'shape': [],
                                    'dtype': _INT64})
        return [step, *written]

    body, results = regions_.region(iteration)
    outs = [regions_.slot(j) for j in init]
    regions_.add(_flow.Loop(slots, init, body, results, outs, length=length))
    return outs[1:]


def _entry(nodes, x, i, y=None):
    # Entry i, an int64 scalar node, of node x along its leading axis by Op.gather, or x with y
    # written there by Op.scatter, which replaces (_flow.entry).
    shape, dtype = list(nodes[x]['shape']), nodes[x]['dtype']
    element = _element(dtype, _op_table())
    if element is None or not shape:
        raise NotImplementedError('map: only floating, integer and Boolean tensors with a leading '
                                  'axis are mapped')
    if y is None:
        return _registered_node(nodes, 'gather', _flow.entry(shape, element), [x, i], shape[1:],
                                dtype)
    code = _op_table()['codes']['scatter']['replace']
    return _registered_node(nodes, 'scatter', _flow.entry(shape, element, code), [x, i, y],
                            shape, dtype)
