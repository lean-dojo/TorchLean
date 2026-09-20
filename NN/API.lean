/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Adapters
public import NN.API.Arguments
public import NN.API.Autograd
public import NN.API.Checkpoint
public import NN.API.CLI.Trainer
public import NN.API.CLI.Training.Command
public import NN.API.Arithmetic
public import NN.API.Data
public import NN.API.Loss
public import NN.API.Macros
public import NN.API.Models
public import NN.API.Module
public import NN.API.Neural
public import NN.API.Optim
public import NN.API.Precision
public import NN.API.RL
public import NN.API.Runtime
public import NN.API.SelfSupervised
-- `NN.Tensor` re-exports the tensor constructors and operations under their short names, and
-- readers of this umbrella expect `Tensor.zeros`-style spellings to be in scope. Nothing in
-- this file needs it, so shake removes it: keep it for the consumers.
public import NN.API.Sample
public import NN.API.Text
public import NN.API.Trainer

public import NN.API.Data.Image
public import NN.API.Models.Diffusion.Sampling
/-!
# TorchLean

Neural-network construction, training, runtime execution, datasets, reinforcement learning,
self-supervised learning, and automatic differentiation.

Import `NN.API` for model code. Import `NN` when a file also uses specification or proof internals.

Focused application surfaces are also available as `NN.API.Precision`, `NN.API.RL`,
`NN.API.SelfSupervised`, and `NN.API.Verification`.
-/

@[expose] public section
