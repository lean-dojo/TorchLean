/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Runtime.PyTorch.Import.CNN
public import NN.Runtime.PyTorch.Import.MLP
public import NN.Runtime.PyTorch.Import.Transformer
public import NN.Runtime.PyTorch.Import.CrownParamstore
public import NN.Runtime.PyTorch.Import.TorchExport

/-!
# `NN.Runtime.PyTorch.Import`

Reusable PyTorch weight-import surface.

The general import path is JSON-first:

1. PyTorch loads the original checkpoint / `state_dict`.
2. A caller-supplied Python script writes the tensors as nested-list JSON.
3. `Import.PyTorch` parses that JSON into shape-checked TorchLean tensors.

For graphs, the matching path is:

1. PyTorch captures an `nn.Module` with the adapter emitted by
   `Export.PyTorch.TorchExport.script`.
2. `Import.PyTorch.TorchExport.parseGraph` reads the `torchlean.ir.v2` JSON into `NN.IR.Graph`.
3. The parser runs the shared IR well-formedness and shape checkers before accepting the graph.

For per-node element types and explicit binary casts, use `Import.IR.parseProgram` from
`NN.Runtime.IR`. Its exact-word payload loader supplies typed constants and linear parameters.
The Python `torchlean.verify` entry point uses this path; other parameterized operators still need
exact typed payload loaders before binary execution can run them.

Op kind strings in that artifact are resolved through `Interop.PyTorch.Wire`, the same table
the exporters write with; parsing a serialized operator tag recovers its `NN.IR.OpTag` identity.

This is the supported graph-import path today. The ONNX adapter emitted by
`Export.PyTorch.ONNX.script` follows the same rule: lower ONNX nodes into the
`torchlean.ir.v2` artifact, then reuse this parser and the shared IR validators. Keeping the
artifact boundary explicit avoids a second, looser graph semantics.

Graph import and payload import remain separate phases. A Conv/Gemm/BatchNorm graph can be
validated as TorchLean IR once the adapter expands it into supported nodes; executing the graph
then requires the matching constants/linear/conv payload store.

`Import.PyTorch.MLP`, `Import.PyTorch.CNN`, and `Import.PyTorch.Transformer` provide model-family
adapters.
Their runnable examples and reference artifacts live under `NN.Examples.Interop.PyTorch`.
-/

@[expose] public section
