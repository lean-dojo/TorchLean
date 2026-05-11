/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Overview
public import NN.Runtime.Autograd.Train.Core
public import NN.Runtime.Autograd.Train.Dataset
public import NN.Runtime.Autograd.Train.Trainer
public import NN.Runtime.Autograd.Train.Logging
public import NN.Runtime.Autograd.Train.Eval
public import NN.Runtime.Autograd.Train.TapeM
public import NN.Runtime.Autograd.Train.TensorLoader
public import NN.Runtime.Autograd.Train.IoLoader
public import NN.Runtime.Autograd.Train.Optim

/-!
# Autograd Train

`NN.Runtime.Autograd.Train` is the curated umbrella for TorchLean's dynamic-tape training helpers.

This layer is intentionally about training-loop infrastructure, not model definitions:

- `Core` gives tagged errors plus typed value/gradient extraction from shape-erased tape data.
- `Dataset` gives pure, deterministic datasets/loaders with seeded shuffling.
- `Trainer` and `Logging` give small report/logging abstractions for example loops and tests.
- `Eval` averages reports over samples or batches while checking metric names.
- `TapeM` contains ergonomic tape-building helpers for params, constants, and mean losses.
- `TensorLoader` and `IoLoader` convert small in-memory/CSV/NPY datasets into typed tensors.
- `Optim` connects parameter tables, schedulers, and canonical optimizer equations.

The public model/training API in `NN.API.*` builds on these pieces. This umbrella exists so tests,
examples, and downstream users can import one stable training helper surface instead of memorizing
the internal file split.
-/

@[expose] public section
