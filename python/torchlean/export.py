"""Export a PyTorch model to a TorchLean capture document.

`export_torchlean_graph_json(model, example, json_path)` captures the model at the example, lowers
its nodes with `_artifact.lower`, or with `_flow.lower` where torch.export records control flow
(`_hop`), and writes `{"artifact", "frontend", "capture_constraints", "declared_types"}` as JSON.
`declared_types` gives, for each lowered tensor value, its step output, in its block for a flow
artifact, and the element type and shape that the capture recorded for it. From a shell:
`python3 -m torchlean.export MODULE CTOR JSON --example-shape 1,4`, where `--require-torch-export`
refuses FX's symbolic trace.
"""

from __future__ import annotations

import argparse
import importlib.util
import json

import torch

from . import _artifact, _flow, _hop
from ._emit import _emit_structure, _lower_kind
from ._fx import (_TENSOR_VALUES, _constant, _dtype_of, _is_getitem,
                  _node_dtype, _node_refs, _node_shape, _op_table, _shape_from_arg, _top_k_mask,
                  _zeros_like)
from ._graph import _DIALECT, _capture, _capture_state, _metadata_constraint, _routing_helpers
from ._pointwise import _append_node, _argument, _broadcast_to, _cast_to, _emit_pointwise
from ._structure import (_STRUCTURE_PLANS, _buffer_increment, _component_producer, _mutates,
                         _paired_dtype, _program_outputs)


def _load_model(module_path: str, ctor_name: str):
    spec = importlib.util.spec_from_file_location("torchlean_user_model", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Python module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctor = getattr(mod, ctor_name)
    model = ctor()
    model.eval()
    return model


def _document(model, graph, user_input_names, fuel=_flow.FUEL, *, example_tensors=()):
    # The capture document of a captured graph: its artifact, its frontend, the constraints of
    # the capture and the type the capture recorded for each lowered value. The caller holds
    # _graph._LOCK from the reset on and has run `_check_graph_operators`, which binds the
    # graph's keywords (_bind), so the lowerings read canonical keywords. A control-flow operator
    # (`_hop`) makes the artifact a flow artifact, whose loops run at most `fuel` iterations each
    # time they are entered. A `fuel` that is not a nonnegative integer is refused, also where no
    # flow reads it (`_flow.natural`).
    fuel = _flow.natural(fuel)
    input_names = set(user_input_names)
    nodes = []
    promotion_cache = {}
    broadcast_cache = {}
    regions = _hop.Regions(nodes, (promotion_cache, broadcast_cache))
    node_to_id = {}
    input_nodes = {}
    capture_constraints = []
    if _DIALECT[0] not in (None, 'TRAINING'):
        # PyTorch 2.13's decomposition drops detach_ with no trace, so this graph can miss a
        # stop-gradient: its derivatives can differ from PyTorch's.
        capture_constraints.append({
            'kind': 'decomposed_program', 'dialect': _DIALECT[0], 'status': 'not_checked',
            'assumptions': ['no_in_place_detach'],
            'message': 'The program is a decomposed torch.export program, from which PyTorch '
                       'drops detach_ with no trace. The values are PyTorch\'s; the derivatives '
                       'hold only if the model calls no detach_.'})

    def walk(graph, ids=None):
        # Lower the nodes of `graph` and give the internal nodes of its results: the captured
        # graph's outputs, or those of a control-flow graph (`_hop`), whose placeholders are the
        # internal nodes `ids`.
        placeholders = iter(ids or ())
        flows = {}
        helpers = _routing_helpers(graph)
        for node in graph.nodes:
            if ids is not None and node.op == 'placeholder':
                node_to_id[node] = next(placeholders)
                continue
            if node in helpers or _component_producer(node) or _hop.subgraph(graph, node):
                node_to_id[node] = None
                continue
            constraint = _metadata_constraint(node)
            if constraint is not None:
                capture_constraints.append(constraint)
                continue
            if node.op == "output":
                refs = _node_refs(node.args, node_to_id, strict=True)
                if ids is not None:
                    return refs
                if not refs:
                    raise NotImplementedError("TorchLean graph import requires at least one tensor output")
                return _program_outputs(refs, lambda ref, name: _cast_to(
                    nodes, ref, _dtype_of(_TENSOR_VALUES[name])), example_tensors)
            if _hop.hop(node):
                flows[node] = _hop.lower(graph, node, node_to_id, regions, lowered)
                node_to_id[node] = None
                continue
            if _is_getitem(node) and node.args[0] in flows:
                node_to_id[node] = flows[node.args[0]][node.args[1]]
                continue
            is_input = node.op == 'placeholder' and node.name in input_names
            increment = _buffer_increment(node)
            if len(getattr(node, "users", {})) == 0 and _mutates(node) and not increment:
                raise NotImplementedError(f'in-place {node.target} whose result is unused: its write may reach a view')
            if len(getattr(node, "users", {})) == 0 and not is_input and not increment:
                # FX often leaves dead tuple projections behind, e.g. the indices of
                # `y, _ = x.max(0)`. They are not part of the exported value, so we omit them
                # instead of forcing every unused container projection to have a tensor lowering.
                node_to_id[node] = None
                continue
            if _zeros_like(node):
                # One zero word broadcast to the shape, as other created tensors of one value are.
                value = node.meta.get('val', node.meta.get('tensor_meta'))
                zero = _append_node(nodes, {'parents': [],
                                            **_constant(torch.zeros((), dtype=value.dtype))})
                node_to_id[node] = _broadcast_to(nodes, zero, _node_shape(node))
                continue
            if node.op == 'get_attr' or (node.op == 'placeholder' and
                not is_input):
                value = _TENSOR_VALUES.get(node.name)
                if value is None: raise NotImplementedError('non-tensor lifted input')
                node_to_id[node] = len(nodes)
                nodes.append({'id': len(nodes), 'parents': [], **_constant(value)})
                continue
            node_id = len(nodes)
            node_to_id[node] = node_id
            shape = _node_shape(node)
            if node.op == "placeholder":
                kind = {"kind": "input"}
                parents = []
                input_nodes[node.name] = node_id
            elif node.op in ("call_function", "call_method", "call_module"):
                kind = _lower_kind(node, model)
                if node.target is torch.ops.aten.as_strided.default:
                    # as_strided reads memory, laid out as the inputs are, and torch.export
                    # records their strides and storage offsets without checking them.
                    capture_constraints.append({
                        'kind': 'strided_read', 'node': node_id, 'value': node.name,
                        'status': 'not_checked', 'assumptions': ['inputs_laid_out_as_captured'],
                        'message': 'as_strided reads memory. Its value is PyTorch\'s when every '
                                   'input has the strides and storage offset of the captured '
                                   'one.'})
                if '_overwritten_allocation' in kind:
                    node_to_id[node] = None
                    continue
                if '_alias' in kind:
                    node_to_id[node] = node_to_id.get(kind['_alias'])
                    if node_to_id[node] is None:
                        raise NotImplementedError('alias of a value without a tensor lowering')
                    continue
                if '_constant' in kind:
                    nodes.append({'id': node_id, 'parents': [], **_constant(kind['_constant'])})
                    continue
                if '_plan' in kind:
                    emit = _emit_structure if kind['_plan']['op'] in _STRUCTURE_PLANS else _emit_pointwise
                    # A complex value is lowered in its paired-real form, at its parts' format.
                    result = emit(kind['_plan'], nodes, node_to_id, shape,
                                  _paired_dtype(node) or _node_dtype(node))
                    node_to_id[node] = result
                    continue
                parents = _node_refs((node.args, node.kwargs), node_to_id)
                # Some core exports repeat a stop-gradient. Its result already stops gradients.
                if kind['kind'] == 'detach' and len(parents) == 1 and \
                        nodes[parents[0]]['kind'] == 'detach':
                    node_to_id[node] = parents[0]
                    continue
                if kind['kind'] == "top_k_mask":
                    source, k = _top_k_mask(node)
                    parents = _node_refs((source,), node_to_id)
                    source_shape = _shape_from_arg(source)
                    width = source_shape[-1] if source_shape else 1
                    if 0 < k < width:
                        capture_constraints.append({
                            'kind': 'top_k_mask_agreement', 'node': node_id,
                            'value': source.name, 'value_id': parents[0], 'k': k,
                            'axis': len(source_shape) - 1, 'status': 'not_checked',
                            'assumptions': ['finite_scores', 'strict_cutoff_gap'],
                            'message': 'Each score row is finite, with its k-th largest score '
                                       'strictly above its (k+1)-th. PyTorch tie selection may '
                                       'otherwise differ from the IR lower-index rule.'})
                elif kind['kind'] in ('add', 'sub', 'mul_elem', 'max_elem', 'min_elem', 'matmul'):
                    left = _argument(node, 0, 'input', 'self')
                    right = _argument(node, 1, 'other', 'mat2')
                    parents = _node_refs((left, right), node_to_id)
                if kind['kind'] in ('linear', 'conv', "layer_norm", "batch_norm"):
                    # These operators read their parameters from their payload: their one
                    # parent is the input.
                    first = _argument(node, 0, 'input', 'self')
                    parents = _node_refs((first,), node_to_id)
                if '_operands' in kind:
                    # A component of a tuple-valued operator reads that operator's operands.
                    parents = _node_refs(kind.pop('_operands'), node_to_id, strict=True)
            else:
                raise NotImplementedError(f"unsupported FX node op: {node.op}")
            dtype = _node_dtype(node)
            if kind['kind'] in ('add', 'sub', 'mul_elem', 'div', 'max_elem', 'min_elem'):
                promoted = []
                for parent in parents:
                    source = nodes[parent]
                    if source['dtype'] != dtype:
                        key = (parent, json.dumps(dtype, sort_keys=True))
                        if key not in promotion_cache:
                            promotion_cache[key] = _cast_to(nodes, parent, dtype)
                        parent = promotion_cache[key]
                    if source['shape'] != shape:
                        key = (parent, tuple(shape))
                        if key not in broadcast_cache:
                            broadcast_cache[key] = _broadcast_to(nodes, parent, shape)
                        parent = broadcast_cache[key]
                    promoted.append(parent)
                parents = promoted
            node_id = len(nodes)
            node_to_id[node] = node_id
            entry = {"id": node_id, "parents": parents, "shape": shape, **kind, "dtype": dtype}
            nodes.append(entry)

    def lowered(module, operands, ids):
        with _hop.bound(module, operands):
            return walk(module.graph, ids)

    output_ids = walk(graph)
    if len(input_nodes) != len(user_input_names) or output_ids is None:
        raise RuntimeError("could not identify graph input/output")
    input_ids = [input_nodes[name] for name in user_input_names]
    regions.flush()
    flat = _flow.flat(nodes, regions.items, input_ids, output_ids)
    if flat is not None:
        kept_nodes, number = flat
        artifact, refs, _ = _artifact.lower(kept_nodes, [number[j] for j in input_ids],
                                            [number[j] for j in output_ids], _op_table())
        place = lambda index: {'ref': refs[number[index]]} if index in number else None
    else:
        artifact, where = _flow.lower(nodes, input_ids, output_ids, regions.items, _op_table(),
                                      fuel)
        place = lambda index: {'block': where[index][0], 'ref': where[index][1]} \
            if index in where else None
    # A constraint names the values it constrains by their references [step, output], in their
    # blocks for a flow artifact. One on a value that the artifact does not compute, which no
    # value it returns reads (`_flow.prune`), constrains nothing and is dropped.
    kept = []
    for constraint in capture_constraints:
        if constraint.get('kind') == 'top_k_mask_agreement':
            node, value = place(constraint.pop('node')), place(constraint.pop('value_id'))
            if node is None or value is None:
                continue
            constraint['ref'], constraint['value_ref'] = node['ref'], value['ref']
            if 'block' in node:
                constraint['block'], constraint['value_block'] = node['block'], value['block']
        elif constraint.get('kind') == 'strided_read':
            node = place(constraint.pop('node'))
            if node is None:
                continue
            constraint.update(node)
        kept.append(constraint)
    capture_constraints = kept
    return {"artifact": artifact, "frontend": "pytorch",
            "capture_constraints": capture_constraints,
            "declared_types": _declared_types(node_to_id, place)}


def _declared_types(node_to_id, place):
    # The element type and shape that the capture's metadata gives each lowered tensor value, at
    # the step output holding it (`place`), in its block for a flow artifact. Lean's IR.load and
    # IR.loadFlow require the program to give it that type.
    declared = []
    for node, node_id in node_to_id.items():
        value = node.meta.get('val', node.meta.get('tensor_meta'))
        if node_id is None or not (isinstance(value, torch.Tensor) or
                                   type(value).__name__ == 'TensorMetadata'):
            continue
        held = place(node_id)
        if held is None:
            continue
        # A complex value is its paired-real form (`_structure._complex_kind`).
        paired = _paired_dtype(node)
        declared.append({'node': node_id, **held, 'origin': node.name,
                         'source': 'capture', 'dtype': paired or _node_dtype(node),
                         'shape': _node_shape(node) + ([2] if paired else [])})
    return declared


def export_torchlean_graph_json(model, example, json_path, require_torch_export=False,
                                prefer_torch_export=True, fuel=_flow.FUEL):
    """Capture `model` at `example`, write its capture document to `json_path` and return it.

    The graph is torch.export's when `prefer_torch_export` holds and it lowers, else FX's symbolic
    trace, which `require_torch_export` refuses. Each loop of the artifact runs at most `fuel`
    iterations each time it is entered, and a `fuel` that `_flow.natural` refuses is refused before
    the capture.
    """
    fuel = _flow.natural(fuel)
    with _capture_state():
        graph, user_input_names = _capture(model, example, require_torch_export,
                                           prefer_torch_export)
        inputs = (example,) if isinstance(example, torch.Tensor) else example
        payload = _document(model, graph, user_input_names, fuel, example_tensors=inputs)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(text)
    return payload


def main():
    """The command line `python3 -m torchlean.export MODULE CTOR JSON --example-shape ...`."""
    parser = argparse.ArgumentParser(
        description="Export a PyTorch nn.Module to a TorchLean capture document")
    parser.add_argument("module", help="Python file containing the model class/constructor")
    parser.add_argument("ctor", help="Zero-argument model class or constructor name")
    parser.add_argument("json", help="Output JSON path")
    parser.add_argument("--example-shape", required=True,
                        help="Comma-separated example input shape, e.g. 1,4")
    parser.add_argument("--require-torch-export", action="store_true",
                        help="Fail instead of falling back to FX")
    args = parser.parse_args()
    shape = tuple(int(x) for x in args.example_shape.split(',') if x)
    model = _load_model(args.module, args.ctor)
    example = torch.randn(*shape)
    payload = export_torchlean_graph_json(model, example, args.json, args.require_torch_export)
    artifact = payload['artifact']
    if 'flow' in artifact:
        print(f"wrote a {artifact['format']} flow to {args.json}")
    else:
        print(f"wrote {len(artifact['nodes'])} {artifact['format']} steps to {args.json}")


if __name__ == "__main__":
    main()
