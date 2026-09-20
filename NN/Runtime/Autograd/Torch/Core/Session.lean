/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session.Backend
public import NN.Runtime.Autograd.Torch.Core.Session.Lifecycle
public import NN.Runtime.Autograd.Torch.Core.Session.Parameters
public import NN.Runtime.Autograd.Torch.Core.Session.Random
public import NN.Runtime.Autograd.Torch.Core.Session.Recording
public import NN.Runtime.Autograd.Torch.Core.Session.References
public import NN.Runtime.Autograd.Torch.Core.Session.State

/-!
# Eager Session

The eager session API is split by responsibility:

- `State` holds the tapes, side tables, and reference generation.
- `Parameters` synchronizes host values and persistent CUDA mirrors.
- `References` validates handles and stores non-differentiable inputs.
- `Lifecycle` creates sessions and releases tape-owned buffers.
- `Recording` records leaves and reads tensor values.
- `Backend` binds accepted capsules to executable handlers.
- `Random` shares a deterministic key schedule across CPU and CUDA.

Import a leaf module when only one of these is needed. This facade preserves the full eager API.
-/
