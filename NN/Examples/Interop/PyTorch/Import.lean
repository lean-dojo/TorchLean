/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Interop.PyTorch.CNN.Import
public import NN.Examples.Interop.PyTorch.MLP.Import
public import NN.Examples.Interop.PyTorch.Transformer.Import

/-!
# PyTorch Example Importers

Architecture-specific JSON `state_dict` loaders for examples and verification workflows.

These modules are small example consumers of the bridge, not the general PyTorch import pipeline. The
actual files live beside their JSON/Python fixtures under `MLP/`, `CNN/`, and `Transformer/`.
The general path is:

`torch.save(model.state_dict())` → generated JSON adapter → `NN.Runtime.PyTorch.Import.Core`.

Verification-owned checkpoint bridges, such as trained PINN loading, live under
`NN.Verification.*` rather than in this examples umbrella.
-/

@[expose] public section
