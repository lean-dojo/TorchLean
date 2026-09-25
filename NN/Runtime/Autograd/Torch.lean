/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core
public import NN.Runtime.Autograd.Torch.Initialization
public import NN.Runtime.Autograd.Torch.ScalarTrainer
public import NN.Runtime.Autograd.Torch.TypedGraphSession

/-!
# Torch-style runtime front-end

This is the public umbrella for the low-level PyTorch-style runtime layer.

The split is intentional:

- `Torch.Core` defines imperative tensor references, parameters, eager sessions, operation wrappers,
  typed executable graphs, and simple scalar trainers.
- `Torch.TypedGraphSession` is the recorder used to implement imperative typed graph
  execution. Its backpropagation agrees with the runtime tape obtained from the recorded graph.
- `Torch.Initialization` defines deterministic parameter initializers.
- `Torch.ScalarTrainer` provides packed loss, gradient, and SGD update operations.

`NN.Runtime.Autograd.Model` builds the model runtime on top of this layer. Application entrypoints
live under `NN.API`.
-/
