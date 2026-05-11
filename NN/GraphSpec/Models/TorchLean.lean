/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Models.TorchLean.Autoencoder
public import NN.GraphSpec.Models.TorchLean.Cnn
public import NN.GraphSpec.Models.TorchLean.Fno1d
public import NN.GraphSpec.Models.TorchLean.Mlp
public import NN.GraphSpec.Models.TorchLean.Resnet18
public import NN.GraphSpec.Models.TorchLean.TransformerBlock

/-!
# TorchLean-Executable GraphSpec Models

This module is the architecture-facing home for reusable model constructors that still execute via
the TorchLean autograd runtime.

The split is intentional:
- `NN.GraphSpec.Models.*` contains graph-authored architectures and architecture-facing wrappers.
- `NN.GraphSpec.Models.TorchLean.*` contains executable `TorchLean.NN.Seq` / `TorchLean.Program`
  constructors for common models.
- `NN.Runtime.Autograd.TorchLean.*` contains runtime machinery: tensors, ops, backends, sessions,
  losses, optimizers, and training loops.

So users looking for models start from GraphSpec, while runtime internals stay focused on execution.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace Models
namespace TorchLean
end TorchLean
end Models
end GraphSpec
end NN
