/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Runtime.PyTorch.Import.CrownParamstore
public import NN.Runtime.PyTorch.Import.TorchExport

/-!
# `NN.Runtime.PyTorch.Import`

Reusable PyTorch weight-import surface.

The general import path is JSON-first:

1. PyTorch loads the original checkpoint / `state_dict`.
2. The adapter emitted by `NN.Runtime.PyTorch.Export.StateDict` writes nested-list JSON.
3. `Import.Core` parses that JSON into shape-checked TorchLean tensors.

For graphs, the matching path is:

1. PyTorch captures an `nn.Module` with the adapter emitted by
   `NN.Runtime.PyTorch.Export.TorchExport`.
2. `Import.TorchExport` parses the resulting `torchlean.ir.v1` graph JSON into `NN.IR.Graph`.
3. The parser runs the shared IR well-formedness and shape checkers before accepting the graph.

Architecture-specific example loaders live beside their fixtures under
`NN.Examples.Interop.PyTorch.{MLP,CNN,Transformer}.Import`. They may support
serious verification workflows, but they still bake in model-family key conventions and therefore
do not belong in the model-agnostic runtime bridge.
-/

@[expose] public section
