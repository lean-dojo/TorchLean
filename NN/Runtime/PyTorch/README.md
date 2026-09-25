# PyTorch Interop (Runtime)

This folder contains TorchLean's bridge layer to and from PyTorch.

We use PyTorch for two practical reasons:

1. Execution and training: PyTorch is a practical runtime for trying models, training on real data,
   and checking that shapes and numerics behave as expected.
2. Moving artifacts: many workflows already produce PyTorch checkpoints or `state_dict`s. This
   bridge gives those parameters a path back into Lean for verification and proofs.

This is an interop layer, not a second semantics for TorchLean. PyTorch may produce code, weights,
graphs, or JSON artifacts, but TorchLean still parses the artifact, checks the shapes and supported
operators, and connects the result to TorchLean IR/spec objects. PyTorch autograd is not part of the
trusted proof boundary.

These files move artifacts between PyTorch and Lean; they are separate from the optional LibTorch
execution provider described in the
[backend chapter](https://lean-dojo.github.io/TorchLean/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/).

## Export

`Export/` contains reusable exporters and adapters.

- `Export/Core.lean` holds shared string utilities and import/header rendering.
- `Export/IRPyTorch.lean` is the general model code path: it exports an `NN.IR.Graph` plus a
  parameter store into a standalone PyTorch module.
- `Export/ONNX.lean` emits a conservative Python adapter that reads an ONNX graph and writes the
  same `torchlean.ir.v1` JSON artifact used by the graph importer. It includes static-shape
  lowerings for common tensor ops plus Conv/Gemm/BatchNorm graph structure where the current IR can
  represent it.
- `Export/StateDict.lean` emits a Python adapter that converts a PyTorch checkpoint
  (`torch.save(model.state_dict(), ...)`, or common checkpoint wrappers) into TorchLean's
  shape checkable JSON format.
- `Export/TorchExport.lean` emits a Python adapter that captures a PyTorch `nn.Module` with
  `torch.export`/FX and writes TorchLean IR JSON for the supported op subset. The script is
  assembled from one Lean definition per Python section (imports, shape helpers, payload
  helpers, `_lower_kind` rule groups, capture, entry point, `main`).

## Semantic operators and v1 wire strings

`NN/IR/Operator.lean` defines `OpKind`, its static attributes, and the constructor identities in
`NN.IR.OpTag`. For example, `(.softmax 1).opTag` is `.softmax`; its axis belongs to `OpKind`.
`OpTag.metadata` gives the parent count and diagnostic name without constructing a dummy operator.
`OpTag.toKind?` reconstructs operations whose tags supply all their static information.

`Wire.lean` assigns each tag its fixed spelling in `torchlean.ir.v1`. Both Python-emitting adapters
write through `Wire.opTag`, and `Import/TorchExport.lean` reads through `Wire.parseOpTag?`.
The v1 strings have their own table, so changing a diagnostic name does not change existing files.
The round trips are proved in Lean:

```lean
theorem Wire.parse_op_tag (tag : OpTag) :
    Wire.parseOpTag? (Wire.opTag tag) = some tag
theorem Wire.parse_op_kind (kind : OpKind) (h : kind.opTag.hasAttributes = false) :
    Wire.parseOpKind? (Wire.opTag kind.opTag) = some kind
```

Operators with axes, shapes, or convolution geometry parse their tag first and then read those
attributes. Tensor parameters, including the weights of `.linear`, come from the payload store.
Value-graph markers (`tuple_getitem`, `multihead_attention`) and the format
marker also live in `Wire`.

The PyTorch adapter matches complete ATen overload names and explicit FX callables or methods.
For example, `aten.log.default` lowers to `log`; `log1p`, `log2`, and `log_softmax` need their own
lowerings and are rejected. Mutating overloads, dtype changes, and unsupported tuple producers are
rejected before the adapter writes an artifact.

## Import

`Import/` parses JSON encoded weights and graphs into TorchLean artifacts.

The bridge uses JSON because Python can write it directly and Lean can parse it without depending on
Python pickle formats.

- `Import/Core.lean` defines `parseTensor`, which turns nested JSON arrays into a `Tensor Float s` when the JSON shape matches `s`.
  Lookups return `Option`, so a missing key or a shape that does not match is a `none` the caller has to
  handle rather than a panic.
- `Import/CrownParamstore.lean` bridges loaded tensors into the graph backend's `ParamStore` when a workflow needs node id keyed parameters.
- `Import/TorchExport.lean` parses TorchLean IR JSON from the generated graph capture adapter and
  accepts only graphs that pass the shared IR validators. All array access is bounds checked and
  reports an `Except` error naming the node; nothing panics on a malformed artifact.

## Model-family adapters and examples

`Import/{MLP,CNN,Transformer}.lean` loads the supported state-dict conventions into typed parameter
records. The corresponding `Export/` modules generate Python classes and embedded parameters.
These reusable adapters are available as `Import.PyTorch.MLP`, `Export.PyTorch.MLP`, and the
matching CNN/Transformer namespaces.

Runnable examples and small reference artifacts live under `NN/Examples/Interop/PyTorch`.
`NN/Tests/Interop/PyTorch.lean` contains graph-capture and numerical parity regressions, run with
`lake exe pytorch_export_check`.

## What Users Can Do Today

- Export PyTorch weights to TorchLean readable JSON through the generated state dict adapter.
- Parse those JSON tensors in Lean with exact shape checks.
- Capture supported PyTorch `nn.Module` graphs as TorchLean IR JSON and validate them in Lean.
- Lower a conservative ONNX static graph fragment into the same TorchLean IR JSON path. Graph
  validation and payload loading stay separate: imported Conv/Gemm/BatchNorm structure can be
  checked as IR, while execution still needs the corresponding payload store.
- Reuse the Lean IR semantics after import. Elementwise ops, reshape/flatten/broadcast/sum, direct
  leading-axis concat plus generic concat through the shared evaluator, axis reductions, axis
  permutation, supported transpose forms, matrix and batched matrix multiplication, arbitrary-axis
  softmax, and eval-mode channel-first BatchNorm have theorem-level IR evaluator bridge facts.
  Payload-backed `linear`, arbitrary-rank no-dilation convolution, payload-backed constants, and
  eval-mode channel-first BatchNorm are also covered at the actual one-step `Graph.evalAt` path as
  well as at their helper evaluators. Arbitrary-rank pooling has the same local bridge to its spec
  operations. LayerNorm is covered through the IR evaluator's rank-independent matrix view, and graph-structural nodes
  such as input, detach, and scalar MSE are covered as well. The
  `NN.Verification.Builtin.Proved.Correctness.Eval` import collects the corresponding concrete
  theorem modules; those imported theorems are the current record of evaluator support.
- Load verification-owned PINN/FNO checkpoints through their `NN/Verification` adapters.
- Emit readable PyTorch code from a TorchLean `NN.IR.Graph` and `ParamStore`.

## Reference (PyTorch)

- PyTorch "Saving and Loading Models" / `state_dict`: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch `torch.export`: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html
- PyTorch FX: https://docs.pytorch.org/docs/stable/fx.html
