# PyTorch Interop (Runtime)

This folder contains TorchLean's bridge layer to and from PyTorch.

We use PyTorch for two practical reasons:

1. Execution and training: PyTorch is a practical runtime for trying models, training on real data,
   and checking that shapes and numerics behave as expected.
2. Moving artifacts: many workflows already produce PyTorch checkpoints or `state_dict`s. This
   bridge gives those parameters a path back into Lean for verification and proofs.

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
  `torch.export`/FX and writes TorchLean IR JSON for the supported op subset.

Reading map:

- Use `Export/Core.lean` for shared formatting helpers.
- Use `Export/StateDict.lean` when you already have PyTorch weights and need a model agnostic
  bridge into Lean readable JSON.
- Use `Export/TorchExport.lean` when you already have a PyTorch `nn.Module` and want to capture
  its tensor program into a TorchLean graph artifact.
- Use `Export/ONNX.lean` when you already have an ONNX graph and want a first-pass static graph
  lowering into the checked TorchLean IR artifact.
- Use `Export/IRPyTorch.lean` when you want a general `NN.IR.Graph` lowering path.

## Import

`Import/` parses JSON encoded weights and graphs into TorchLean artifacts.

Why JSON? Python can write it directly, and Lean can parse it without depending on Python pickle
formats.

- `Import/Core.lean` defines `parseTensor`, which turns nested JSON arrays into a `Tensor Float s` when the JSON shape matches `s`.
  It also provides small error reporting wrappers (`loadWeightsE`, `getTensorE`) for debugging missing keys and shape mismatches.
- `Import/CrownParamstore.lean` bridges loaded tensors into the graph backend's `ParamStore` when a workflow needs node id keyed parameters.
- `Import/TorchExport.lean` parses TorchLean IR JSON from the generated graph capture adapter and
  accepts only graphs that pass the shared IR validators.

Reading map:

- Use `Import/Core.lean` for the generic JSON parsing path.
- Use `Import/TorchExport.lean` when you have a captured PyTorch graph JSON artifact.
- Use `Import/CrownParamstore.lean` when you already have typed tensors and need to assemble a `ParamStore`.

## Examples live elsewhere

Architecture specific loaders and example round trips live under
`NN/Examples/Interop/PyTorch/{Export,Import}`. Runtime should not own model-family modules; it
should own the reusable bridge that examples, verification tools, and downstream projects can
share.

## What Users Can Do Today

- Export PyTorch weights to TorchLean readable JSON through the generated state dict adapter.
- Parse those JSON tensors in Lean with exact shape checks.
- Capture supported PyTorch `nn.Module` graphs as TorchLean IR JSON and validate them in Lean.
- Lower a conservative ONNX static graph fragment into the same TorchLean IR JSON path. Graph
  validation and payload loading stay separate: imported Conv/Gemm/BatchNorm structure can be
  checked as IR, while execution still needs the corresponding payload store.
- Reuse the Lean IR semantics after import. Elementwise ops, reshape/flatten/broadcast/sum, direct
  leading-axis concat plus generic concat through the shared evaluator, axis reductions, axis
  permutation, supported transpose forms, rank-2/3 matmul, softmax through the evaluator's
  axis-permutation path, and eval-mode NCHW BatchNorm now have theorem-level IR evaluator bridge
  facts. Payload-backed `linear`, no-dilation `conv2d`, payload-backed constants, and eval-mode
  NCHW BatchNorm are also covered at the actual one-step `Graph.evalAt` path, not only at their
  helper evaluators. CHW pooling has the same local bridge to its spec operations. LayerNorm is
  covered through the IR evaluator's reshape-to-2D `Spec.layerNorm` path, and graph-structural nodes
  such as input, detach, and scalar MSE are covered as well. The theorem import surface also
  includes an `Eval.Coverage` checkpoint listing the covered IR constructor families.
- Load supported verification model families such as PINNs/FNOs through the example interop loaders.
- Emit readable PyTorch code from a TorchLean `NN.IR.Graph` and `ParamStore`.

What is not claimed: Lean does not prove PyTorch or CUDA kernels correct, and it does
not parse `.pt`/`.pth` pickle/zip checkpoints directly. PyTorch is the external loader for those
files; TorchLean checks the JSON artifact it receives.

## Reference (PyTorch)

- PyTorch "Saving and Loading Models" / `state_dict`: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch `torch.export`: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html
- PyTorch FX: https://docs.pytorch.org/docs/stable/fx.html
