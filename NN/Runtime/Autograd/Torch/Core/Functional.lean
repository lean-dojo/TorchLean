/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Functional.Activation
public import NN.Runtime.Autograd.Torch.Core.Functional.Curried
public import NN.Runtime.Autograd.Torch.Core.Functional.GraphInputs
public import NN.Runtime.Autograd.Torch.Core.Functional.Layers
public import NN.Runtime.Autograd.Torch.Core.Functional.Tensor

/-!
# Functional Tensor Programs

The shared authoring API for eager and typed graph execution. Import `Functional.Ops` for the
backend contract, `Functional.Curried` for argument packs, or the operation module you need.
`Functional.GraphInputs` contains the graph-specific binding helpers.
-/
