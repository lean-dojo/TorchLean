/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.BackwardOptim
public import NN.Runtime.Autograd.Torch.Core.CheckpointIO
public import NN.Runtime.Autograd.Torch.Core.CudaBridge
public import NN.Runtime.Autograd.Torch.Core.Functional
public import NN.Runtime.Autograd.Torch.Core.Ops
public import NN.Runtime.Autograd.Torch.Core.OptimizerCheckpoint
public import NN.Runtime.Autograd.Torch.Core.Session
public import NN.Runtime.Autograd.Torch.Core.TensorTransfer
public import NN.Runtime.Autograd.Torch.Core.Trainer
public import NN.Runtime.Autograd.Torch.Core.Trainer.EagerOps
public import NN.Runtime.Autograd.Torch.Core.Trainer.GraphOps
public import NN.Runtime.Autograd.Torch.Core.Trainer.Recording
public import NN.Runtime.Autograd.Torch.Core.TypedGraph
public import NN.Runtime.Autograd.Torch.Core.Types

/-!
# Torch Core

Torch-style runtime front-end for eager execution, typed graphs, and training helpers.

- `Core.Types`: public handles, options, and parameter wrappers.
- `Core.Session`: eager session state, CUDA bridge, and tape lifecycle helpers.
- `Core.Ops`: eager tensor operations.
- `Core.BackwardOptim`: eager backward passes and optimizers.
- `Core.TypedGraph`: reusable shape-indexed graph wrappers.
- `Core.Functional`: operation-generic `Ops` interface and curried syntax.
- `Core.Trainer.Types`: scalar trainer contracts without backend construction.
- `Core.Trainer`: construct eager or graph trainers.
- `Core.Trainer.EagerOps` / `GraphOps`: backend instances for `Ops`.

This facade re-exports the complete runtime. Library code should import the part it uses:
`Functional.Ops` for generic operators, `Trainer.Types` for trainer consumers, and the relevant
`Session` submodule for eager internals.
-/
