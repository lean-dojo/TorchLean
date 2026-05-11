# PyTorch Interop (Runtime)

This folder contains TorchLean's bridge layer to and from PyTorch.

We use PyTorch for two practical reasons:

1. Execution and training: PyTorch is a practical runtime for trying models, training on real data,
   and checking that shapes and numerics behave as expected.
2. Moving artifacts: many workflows already produce PyTorch checkpoints or `state_dict`s. This
   bridge gives those parameters a path back into Lean for verification and proofs.

## Export

`Export/` contains reusable exporters and adapters.

- `Export/Core.lean` holds shared string utilities and common boilerplate such as imports.
- `Export/IRPyTorch.lean` is the general model code path: it exports an `NN.IR.Graph` plus a
  parameter store into a standalone PyTorch module.
- `Export/StateDict.lean` emits a Python adapter that converts a PyTorch checkpoint
  (`torch.save(model.state_dict(), ...)`, or common checkpoint wrappers) into TorchLean's
  shape checkable JSON format.
- `Export/TorchExport.lean` emits a Python adapter that captures a PyTorch `nn.Module` with
  `torch.export`/FX and writes TorchLean IR JSON for the supported op subset.

Reading map:

- Start with `Export/Core.lean` if you want the shared formatting helpers.
- Use `Export/StateDict.lean` when you already have PyTorch weights and need a model agnostic
  bridge into Lean readable JSON.
- Use `Export/TorchExport.lean` when you already have a PyTorch `nn.Module` and want to capture
  its tensor program into a TorchLean graph artifact.
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

- Start with `Import/Core.lean` if you want the generic JSON parsing path.
- Use `Import/TorchExport.lean` when you have a captured PyTorch graph JSON artifact.
- Use `Import/CrownParamstore.lean` when you already have typed tensors and need to assemble a `ParamStore`.

## Examples live elsewhere

Architecture specific loaders and demo round trips live under
`NN/Examples/Interop/PyTorch/{Export,Import}`. Runtime should not own model-family modules; it
should own the reusable bridge that examples, verification tools, and downstream projects can
share.

## What Users Can Do Today

- Export PyTorch weights to TorchLean readable JSON through the generated state dict adapter.
- Parse those JSON tensors in Lean with exact shape checks.
- Capture supported PyTorch `nn.Module` graphs as TorchLean IR JSON and validate them in Lean.
- Load supported verification model families such as PINNs/FNOs through the example interop loaders.
- Emit readable PyTorch code from a TorchLean `NN.IR.Graph` and `ParamStore`.

What is deliberately not claimed: Lean does not prove PyTorch or CUDA kernels correct, and it does
not parse `.pt`/`.pth` pickle/zip checkpoints directly. PyTorch is the external loader for those
files; TorchLean checks the JSON artifact it receives.

## Reference (PyTorch)

- PyTorch "Saving and Loading Models" / `state_dict`: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch `torch.export`: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html
- PyTorch FX: https://docs.pytorch.org/docs/stable/fx.html
