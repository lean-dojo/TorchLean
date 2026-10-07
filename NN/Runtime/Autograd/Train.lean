/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Core
public import NN.Runtime.Autograd.Train.Trainer
public import NN.Runtime.Autograd.Train.Eval
public import NN.Runtime.Autograd.Train.TapeM
public import NN.Runtime.Autograd.Train.Optim

/-!
# Autograd Train

`NN.Runtime.Autograd.Train` is the curated umbrella for TorchLean's dynamic-tape training helpers.

This layer is about training-loop infrastructure, not model definitions:

- `Core` gives tagged errors plus typed value/gradient extraction from shape-erased tape data.
- `Trainer` collects structured step reports and invokes a caller-supplied logger.
- `Eval` averages reports over samples or batches while checking metric names.
- `TapeM` contains ergonomic tape-building helpers for params, constants, and mean losses.
- `Optim` connects parameter tables, schedulers, and canonical optimizer equations.

This umbrella collects the low-level tape-training helpers used by examples and tests. The public
model and trainer interfaces live under `NN.API`.
-/

@[expose] public section
