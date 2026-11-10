"""Export a PyTorch model to a TorchLean capture document.

`export_torchlean_graph_json(model, example, json_path)` captures the model at the example, lowers
its nodes with `_artifact.lower` and writes `{"artifact", "frontend", "capture_constraints"}` as
JSON. From a shell: `python3 -m torchlean.export MODULE CTOR JSON --example-shape 1,4`, where
`--require-torch-export` refuses FX's symbolic trace.
"""

from __future__ import annotations

import argparse
import importlib.util
import json

import torch

from . import _artifact
from ._emit import _emit_structure, _lower_kind
from ._fx import (_TENSOR_VALUES, _broadcast_kind, _constant, _node_dtype, _node_refs, _node_shape,
                  _op_table, _shape_from_arg, _top_k_mask, _zeros_like)
from ._graph import _LOCK, _capture, _metadata_constraint, _routing_helpers
from ._pointwise import _cast_to, _emit_pointwise
from ._structure import (_STRUCTURE_PLANS, _buffer_increment, _complex_producer,
                         _component_producer, _mutates, _program_outputs)


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


def _document(model, graph, user_input_names):
    # The capture document of a captured graph: its artifact, its frontend and the
    # constraints of the capture. The caller holds _graph._LOCK from the reset on.
    input_names = set(user_input_names)
    nodes = []
    promotion_cache = {}
    broadcast_cache = {}
    node_to_id = {}
    output_ids = None
    input_nodes = {}
    capture_constraints = []
    helpers = _routing_helpers(graph)
    for node in graph.nodes:
        if node in helpers or _component_producer(node) or _complex_producer(node):
            node_to_id[node] = None
            continue
        constraint = _metadata_constraint(node)
        if constraint is not None:
            capture_constraints.append(constraint)
            continue
        if node.op == "output":
            refs = _node_refs(node.args, node_to_id, strict=True)
            if not refs:
                raise NotImplementedError("TorchLean graph import requires at least one tensor output")
            output_ids = _program_outputs(refs)
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
            value = node.meta.get('val', node.meta.get('tensor_meta'))
            node_to_id[node] = len(nodes)
            nodes.append({'id': len(nodes), 'parents': [], **_constant(torch.zeros(_node_shape(node), dtype=value.dtype))})
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
                result = emit(kind['_plan'], nodes, node_to_id, shape, _node_dtype(node))
                node_to_id[node] = result
                continue
            parents = _node_refs((node.args, node.kwargs), node_to_id)
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
                left = node.args[0] if node.args else node.kwargs.get('input', node.kwargs.get('self'))
                right = node.args[1] if len(node.args) > 1 else node.kwargs.get('other', node.kwargs.get('mat2'))
                parents = _node_refs((left, right), node_to_id)
            if kind['kind'] in ('linear', 'conv', "layer_norm", "batch_norm"):
                # These operators read their parameters from their payload: their one
                # parent is the input.
                first = node.args[0] if node.args else node.kwargs.get('input')
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
                        nodes.append({'id': len(nodes), 'parents': [parent],
                                      **_broadcast_kind(source['shape'], shape),
                                      'shape': shape, 'dtype': dtype})
                        broadcast_cache[key] = len(nodes) - 1
                    parent = broadcast_cache[key]
                promoted.append(parent)
            parents = promoted
        node_id = len(nodes)
        node_to_id[node] = node_id
        entry = {"id": node_id, "parents": parents, "shape": shape, **kind, "dtype": dtype}
        nodes.append(entry)
    if len(input_nodes) != len(user_input_names) or output_ids is None:
        raise RuntimeError("could not identify graph input/output")
    input_ids = [input_nodes[name] for name in user_input_names]
    artifact, refs, _ = _artifact.lower(nodes, input_ids, output_ids, _op_table())
    # A constraint names the values it constrains by their references [step, output].
    for constraint in capture_constraints:
        if constraint.get('kind') == 'top_k_mask_agreement':
            constraint['ref'] = refs[constraint.pop('node')]
            constraint['value_ref'] = refs[constraint.pop('value_id')]
    return {"artifact": artifact, "frontend": "pytorch",
            "capture_constraints": capture_constraints}


def export_torchlean_graph_json(model, example, json_path, require_torch_export=False,
                                prefer_torch_export=True):
    """Capture `model` at `example`, write its capture document to `json_path` and return it.

    The graph is torch.export's when `prefer_torch_export` holds and it lowers, else FX's symbolic
    trace, which `require_torch_export` refuses.
    """
    with _LOCK:
        graph, user_input_names = _capture(model, example, require_torch_export,
                                           prefer_torch_export)
        payload = _document(model, graph, user_input_names)
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
    print(f"wrote {len(artifact['nodes'])} {artifact['format']} steps to {args.json}")


if __name__ == "__main__":
    main()
