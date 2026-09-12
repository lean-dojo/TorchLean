/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Interop.PyTorch.Roundtrip

/-!
# PyTorch Interop Examples

`Roundtrip` exports MLP, CNN, and transformer models or reads their JSON weights for a Lean
forward pass. Companion Python scripts produce the reference weights.

The reusable importers and exporters live under `NN.Runtime.PyTorch`. Graph-capture regression
checks run separately with `lake exe pytorch_export_check`.
-/

@[expose] public section
