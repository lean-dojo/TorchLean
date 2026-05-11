# PyTorch Interop Examples

This folder contains PyTorch interop examples. The reusable bridge lives in
`NN/Runtime/PyTorch`; the files here are small interop examples, smoke tests, and tutorial fixtures.

## Two Separate Paths

### 1. Model Agnostic Graph Capture

- Lean entrypoint: `TorchExportSmoke.lean`
- Runtime bridge: `NN.Runtime.PyTorch.Export.TorchExport` and `NN.Runtime.PyTorch.Import.TorchExport`
- Command: `lake exe torchlean pytorch_export_smoke`

This path writes a generated Python adapter, captures small `nn.Module`s through `torch.export`/FX,
emits `torchlean.ir.v1` JSON, and then asks Lean to parse and validate the graph. The importer has a
value graph layer: tensor valued FX nodes lower into `NN.IR.Graph`; tuple valued FX nodes are kept
explicit, and the deterministic `batch_first=True`, `num_heads=1` self attention output of
`nn.MultiheadAttention(...)[0]` is decomposed into ordinary tensor IR primitives. Broader MHA cases
still fail loudly with the unsupported feature named in the error.

### 2. State Dict Round Trips

- Lean entrypoint: `Roundtrip.lean`
- Example exporters: `MLP/Export.lean`, `CNN/Export.lean`, `Transformer/Export.lean`
- Example importers: `MLP/Import.lean`, `CNN/Import.lean`, `Transformer/Import.lean`

Run:

- `lake exe torchlean pytorch_roundtrip --model mlp --action export`
- `lake exe torchlean pytorch_roundtrip --model mlp --action import`
- `lake exe torchlean pytorch_roundtrip --model cnn --action import`
- `lake exe torchlean pytorch_roundtrip --model transformer --action import`

These examples are intentionally model specific: they show how a known MLP/CNN/Transformer JSON
`state_dict` becomes typed Lean tensors and how generated PyTorch code can be produced for the same
small architecture. For arbitrary checkpoints, prefer the general state dict adapter in
`NN.Runtime.PyTorch.Export.StateDict`.

## Fixture folders

- `MLP/`, `CNN/`, `Transformer/` contain the small model specific round trip fixtures: JSON
  weights, compact Python producer scripts, and the Lean import/export helpers for that model.
- Regenerate JSON weights with:
  - `python3 NN/Examples/Interop/PyTorch/MLP/train_mlp.py`
  - `python3 NN/Examples/Interop/PyTorch/CNN/train_cnn.py`
  - `python3 NN/Examples/Interop/PyTorch/Transformer/train_transformer.py`

## What not to put here

- General PyTorch checkpoint adapters belong in `NN/Runtime/PyTorch`.
- Verification checkpoint bridges belong under `NN/Verification/*`, not under examples.
- Architecture specific verification loaders that are used outside examples should move to their
  owning verification package once they stop being example fixtures.
- Generated temporary files from graph capture belong under `.lake/build`, not in this folder.
