/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core
public import NN.Runtime.Autograd.Torch.LinkedSession
public import NN.Runtime.Autograd.Torch.Utils

/-!
# Torch-style runtime front-end

This is the public umbrella for the low-level PyTorch-style runtime layer.

The split is intentional:

- `Torch.Core` defines imperative tensor references, parameters, eager sessions, operation wrappers,
  compiled scalar/output wrappers, and simple scalar trainers.
- `Torch.LinkedSession` records the same style of imperative computation into the proved
  `GraphData` IR and exposes the theorem connecting compiled runtime backprop to proved graph
  backprop.
- `Torch.Utils` contains small demo/training conveniences such as deterministic initializers,
  small sample builders, and trainer loops.

`TorchLean/*` builds the higher-level model/program API on top of this layer. So `Torch` is the
low-level session/ref bridge; `TorchLean` is the nicer user-facing model stack.
-/

@[expose] public section
