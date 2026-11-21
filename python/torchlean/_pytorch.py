"""Snapshot PyTorch examples for verification and lower Dynamo graphs with live parameters."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def copy_input(value, memo):
    """Copy a tensor and its storage aliases without retaining its backward graph."""
    import torch

    if id(value) in memo:
        return memo[id(value)]
    with torch.inference_mode(value.is_inference()), torch.set_grad_enabled(not value.is_leaf):
        if value._base is not None:
            # Replay the original view on the copied base, including dtype-changing views.
            copied = value._view_func(copy_input(value._base, memo))
            if copied is None:
                raise NotImplementedError("PyTorch could not replay an input view on its copy")
            copied.requires_grad_(value.requires_grad)
        elif (value.is_leaf and type(value) is not torch.nn.Parameter
              and not (value.is_neg() or value.is_conj())):
            copied = copy.deepcopy(value, memo)
        else:
            # Tensor.deepcopy materializes a lazy sign or conjugation and loses the layout.
            # Copy the physical storage first, then restore its interpretation on the copy.
            raw = value.detach()
            torch._C._set_neg(raw, False)
            torch._C._set_conj(raw, False)
            copied = copy.deepcopy(raw, memo)
            torch._C._set_neg(copied, value.is_neg())
            torch._C._set_conj(copied, value.is_conj())
            if type(value) is torch.nn.Parameter:
                copied = torch.nn.Parameter(copied, requires_grad=value.requires_grad)
            if value.is_leaf:
                copied.requires_grad_(value.requires_grad)
            else:
                # Mark the copy non-leaf without attaching it to the caller's tape.
                torch._mirror_autograd_meta_to(value, copied)
        # A storage copy starts its own counter. Only adjust the isolated snapshot.
        if not value.is_inference() and copied._version != value._version:
            torch._C._autograd._unsafe_set_version_counter([copied], [value._version])
        memo[id(value)] = copied
        if value.retains_grad:
            copied.retain_grad()
        if (value.is_leaf or value.retains_grad) and value.grad is not None:
            copied.grad = copy_input(value.grad, memo)
    return copied


def reference_inputs(tensors):
    """Refuse reads of backward history that a forward capture did not copy."""
    import torch
    from torch.overrides import TorchFunctionMode

    inputs = {id(t) for t in tensors if not t.is_leaf}
    inputs.update(id(t._base) for t in tensors if t._base is not None and not t._base.is_leaf)
    if not inputs:
        return nullcontext()

    class ForwardInputs(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func == torch.Tensor.grad_fn.__get__ and id(args[0]) in inputs:
                raise NotImplementedError("sample comparison cannot inspect an input's backward graph")
            return func(*args, **(kwargs or {}))

    return ForwardInputs()


def lower(graph_module: Any, example_inputs: Sequence[Any],
          checker: str | Path | Sequence[str] | None = None) -> dict[str, Any]:
    """Translate a copy of a native compiler graph, at the op table of ``checker``.

    No user code runs, linear parameters remain graph inputs, and a linear layer becomes the IR's
    transpose, matrix product and addition nodes. The translation does not certify that their
    rounding matches a fused frontend kernel.
    """
    import torch
    import torch.nn.functional as functional
    from torch.fx import Graph, GraphModule, Node, map_arg
    from torch.fx.passes.shape_prop import TensorMetadata

    from . import _lower_pytorch_graph
    from ._bind import canonical

    def metadata(node):
        value = node.meta.get("val", node.meta.get("tensor_meta"))
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise NotImplementedError("compiler graph values require tensor metadata")
        if any(not isinstance(size, int) for size in value.shape):
            raise NotImplementedError("compiler graph lowering requires static tensor shapes")
        return value

    def result_metadata(shape, dtype, source):
        # The capture reads only the shape, dtype and strides. Building the metadata
        # by hand runs neither real tensors nor fake kernels.
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
        kwargs = canonical(node.op, node.target, args,
                           map_arg(node.kwargs, lambda parent: environment[parent]))
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
    input_names = [node.name for node in graph.nodes if node.op == "placeholder"]
    return _lower_pytorch_graph(translated, graph, input_names, checker=checker)[0]
