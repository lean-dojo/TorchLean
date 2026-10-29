"""Lower Dynamo graphs while retaining their live tensor parameter inputs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._checker import Checker


def lower(graph_module: Any, example_inputs: Sequence[Any], checker: Checker,
          directory: Path) -> dict[str, Any]:
    """Translate a copy of a native compiler graph, without executing user code.

    Linear parameters remain ordinary graph inputs. The translation expresses a
    linear layer with the existing transpose, matrix product and addition nodes;
    it does not certify that their rounding matches a fused frontend kernel.
    """
    import torch
    import torch.nn.functional as functional
    from torch.fx import Graph, GraphModule, Node, map_arg
    from torch.fx.passes.shape_prop import TensorMetadata

    from . import _load_pytorch_adapter, _lower_pytorch_graph

    def metadata(node):
        value = node.meta.get("val", node.meta.get("tensor_meta"))
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise NotImplementedError("compiler graph values require tensor metadata")
        if any(not isinstance(size, int) for size in value.shape):
            raise NotImplementedError("compiler graph lowering requires static tensor shapes")
        return value

    def result_metadata(shape, dtype, source):
        # Only the shape, dtype and strides are needed by the existing adapter.
        # Build metadata directly so neither real tensors nor fake kernels run.
        strides = []
        step = 1
        for size in reversed(shape):
            strides.append(step)
            step *= max(1, size)
        return TensorMetadata(torch.Size(shape), dtype, getattr(source, "requires_grad", False),
                              tuple(reversed(strides)), torch.contiguous_format, False, {})

    def cast(node, dtype):
        value = metadata(node)
        if value.dtype == dtype:
            return node
        result = graph.call_method("to", (node,), {"dtype": dtype})
        result.meta["val"] = result_metadata(value.shape, dtype, value)
        return result

    original_inputs = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    if len(original_inputs) != len(example_inputs):
        raise NotImplementedError("compiler graph input count does not match its examples")
    examples = dict(zip(original_inputs, example_inputs))
    graph = Graph()
    environment = {}
    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            module = graph_module.get_submodule(str(node.target))
            if any(True for _ in module.parameters()) or any(True for _ in module.buffers()):
                raise NotImplementedError("compiler module state must be lifted to tensor inputs")
        if node.op == "get_attr":
            value = graph_module
            for part in str(node.target).split("."):
                value = getattr(value, part)
            if isinstance(value, torch.Tensor):
                raise NotImplementedError("compiler tensor state must be lifted to live inputs")
        is_linear = (node.op == "call_function"
                     and node.target in (functional.linear, torch.ops.aten.linear,
                                         torch.ops.aten.linear.default))
        if not is_linear:
            result = graph.node_copy(node, lambda parent: environment[parent])
            result.meta = dict(node.meta)
            if "val" not in result.meta and "example_value" in result.meta:
                result.meta["val"] = result.meta["example_value"]
            if node.op == "placeholder":
                if not isinstance(examples[node], torch.Tensor):
                    raise NotImplementedError("compiler graph requires tensor-only dynamic inputs")
                if "val" not in result.meta and "tensor_meta" not in result.meta:
                    result.meta["val"] = examples[node]
                metadata(result)
            environment[node] = result
            continue

        args = map_arg(node.args, lambda parent: environment[parent])
        kwargs = map_arg(node.kwargs, lambda parent: environment[parent])
        source = kwargs.get("input", args[0] if args else None)
        weight = kwargs.get("weight", args[1] if len(args) > 1 else None)
        bias = kwargs.get("bias", args[2] if len(args) > 2 else None)
        if kwargs.keys() - {"input", "weight", "bias"} or len(args) > 3:
            raise NotImplementedError("unsupported linear arguments in compiler graph")
        if not isinstance(source, Node) or not isinstance(weight, Node):
            raise NotImplementedError("compiler linear inputs and weights must be tensor nodes")
        source_meta, weight_meta = metadata(source), metadata(weight)
        output_meta = node.meta.get("val", node.meta.get("example_value",
                                                        node.meta.get("tensor_meta")))
        if not hasattr(output_meta, "shape") or not hasattr(output_meta, "dtype"):
            raise NotImplementedError("compiler linear output requires tensor metadata")
        if any(not isinstance(size, int) for size in output_meta.shape):
            raise NotImplementedError("compiler graph lowering requires static tensor shapes")
        if len(weight_meta.shape) != 2 or not source_meta.shape:
            raise NotImplementedError("compiler linear requires rank-2 weights and rank>=1 inputs")
        expected_shape = (*source_meta.shape[:-1], weight_meta.shape[0])
        if (source_meta.shape[-1] != weight_meta.shape[1]
                or tuple(output_meta.shape) != expected_shape):
            raise NotImplementedError("compiler linear input and output shapes are inconsistent")
        if bias is not None and (not isinstance(bias, Node)
                                 or tuple(metadata(bias).shape) != (weight_meta.shape[0],)):
            raise NotImplementedError("compiler linear bias requires one output-feature dimension")
        dtype = output_meta.dtype
        source, weight = cast(source, dtype), cast(weight, dtype)
        transposed = graph.call_function(torch.transpose, (weight, -2, -1))
        transposed.meta["val"] = result_metadata(
            tuple(reversed(weight_meta.shape)), dtype, weight_meta)
        product = graph.call_function(torch.matmul, (source, transposed))
        product.meta["val"] = result_metadata(expected_shape, dtype, output_meta)
        result = product
        if bias is not None:
            result = graph.call_function(torch.add, (product, cast(bias, dtype)))
        result.meta = dict(node.meta)
        result.meta["val"] = output_meta
        environment[node] = result

    graph.lint()
    translated = GraphModule(graph_module, graph)
    adapter = _load_pytorch_adapter(checker, directory)
    input_names = [node.name for node in graph.nodes if node.op == "placeholder"]
    return _lower_pytorch_graph(adapter, translated, graph, input_names, directory)
