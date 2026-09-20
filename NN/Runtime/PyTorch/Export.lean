/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Runtime.PyTorch.Export.CNN
public import NN.Runtime.PyTorch.Export.MLP
public import NN.Runtime.PyTorch.Export.Transformer
public import NN.Runtime.PyTorch.Export.IRPyTorch
public import NN.Runtime.PyTorch.Export.ONNX
public import NN.Runtime.PyTorch.Export.StateDict
public import NN.Runtime.PyTorch.Export.TorchExport

/-!
# `NN.Runtime.PyTorch.Export`

Reusable PyTorch export/adaptation surface.

This umbrella provides graph adapters and model-family exporters:

- `Export.Core` provides shared Python string-generation utilities.
- `Export.IRPyTorch` lowers a TorchLean `NN.IR.Graph` plus parameters into readable PyTorch
  `nn.Module` source.
- `Export.ONNX` emits a conservative ONNX-to-`torchlean.ir.v1` adapter for static graph
  fragments, including expanded graph lowerings for common Conv/Gemm/BatchNorm patterns.
- `Export.StateDict` emits the general checkpoint-to-JSON adapter for PyTorch `state_dict`
  artifacts.
- `Export.TorchExport` emits the Python graph-capture adapter for PyTorch `nn.Module` →
  TorchLean IR JSON.
- `NN.Runtime.PyTorch.Wire` defines the fixed v1 constructor spellings shared by the
  `TorchExport` and `ONNX` adapters and by the importer.

For ONNX workflows, the architecture is the same: the Python-side adapter reads ONNX and emits
`torchlean.ir.v1`; Lean then accepts or rejects the result through
`NN.Runtime.PyTorch.Import.TorchExport`. That keeps graph import tied to the same checked IR
contracts used by `torch.export`/FX capture. The adapter validates graph structure and shapes;
runtime execution of imported parameterized nodes still needs the matching payload store.

`Export.MLP`, `Export.CNN`, and `Export.Transformer` provide model-family adapters.
Their runnable examples and reference artifacts live under `NN.Examples.Interop.PyTorch`.
-/

@[expose] public section
