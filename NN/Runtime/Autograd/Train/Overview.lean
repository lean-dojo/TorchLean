/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# `NN.Runtime.Autograd.Train`: Training Utilities Overview

This directory contains lightweight, mostly-pure utilities for writing training loops on top of
TorchLean's runtime autograd tape.

The goal is familiarity (a recognizable workflow) while keeping the code:
- deterministic (pure shuffles; explicit seeds),
- small (easy to audit),
- and compatible with both eager and compiled backends (via the shared tape/`TorchLean.Session`).

The modules here are narrow in scope: they provide the reusable training infrastructure, while
architecture-specific math and constructors stay in `NN.GraphSpec.Models.*`, `NN.API.Models.*`, or
user code.

## Module Map

- `NN.Runtime.Autograd.Train.Core`: tagged errors, typed extraction of values/grads, and one-step
  SGD glue.
- `NN.Runtime.Autograd.Train.Dataset`: deterministic `Dataset` / `DataLoader` analogues with
  seeded shuffle.
- `NN.Runtime.Autograd.Train.Eval`: evaluation helpers that aggregate step reports.
- `NN.Runtime.Autograd.Train.Logging`: a small pluggable logger interface.
- `NN.Runtime.Autograd.Train.Optim`: optimizer and scheduler integration for training loops.
- `NN.Runtime.Autograd.Train.TapeM`: tape-building helpers for params, constants, and batch mean
  losses.
- `NN.Runtime.Autograd.Train.TensorLoader`: in-memory tensor/dataset constructors with shape checks.
- `NN.Runtime.Autograd.Train.IoLoader`: CSV/NPY loaders for reproducible examples and regression
  tests.
- `NN.Runtime.Autograd.Train.Trainer`: a structured trainer loop with reports and optional logging.

The training utilities are intentionally not a second model API. Layers and model constructors live
in the architecture/API modules; this directory only supplies the loop machinery that feeds them.

## References

- PyTorch data loading (`torch.utils.data`):
  https://pytorch.org/docs/stable/data.html
- PyTorch optimizers (`torch.optim`):
  https://pytorch.org/docs/stable/optim.html
- A canonical training/eval loop tutorial:
  https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
-/

@[expose] public section
