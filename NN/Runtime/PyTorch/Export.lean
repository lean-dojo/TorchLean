/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.Runtime.PyTorch.Export.IRPyTorch
public import NN.Runtime.PyTorch.Export.StateDict
public import NN.Runtime.PyTorch.Export.TorchExport

/-!
# `NN.Runtime.PyTorch.Export`

Reusable PyTorch export/adaptation surface.

Use this umbrella when you want the runtime bridge, not the demo models:

- `Export.Core` provides shared Python string-generation utilities.
- `Export.IRPyTorch` lowers a TorchLean `NN.IR.Graph` plus parameters into readable PyTorch
  `nn.Module` source.
- `Export.StateDict` emits the general checkpoint-to-JSON adapter for PyTorch `state_dict`
  artifacts.
- `Export.TorchExport` emits the Python graph-capture adapter for PyTorch `nn.Module` →
  TorchLean IR JSON.

Demo-specific MLP/CNN/Transformer code lives beside its fixtures under
`NN.Examples.Interop.PyTorch.{MLP,CNN,Transformer}.*`.
-/

@[expose] public section
