/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Data.Loaders.Csv
public import NN.Examples.Data.Loaders.Npy
public import NN.Examples.Data.Loaders.Cifar10Images

/-!
# Data loader tutorials

This umbrella collects the Lean-side data tutorials:

- `Csv`: numeric CSV rows, transforms, minibatching, and a step LR scheduler;
- `Npy`: NumPy/PyTorch `.npy` arrays, metadata inspection, transforms, and minibatching;
- `Cifar10Images`: image-shaped NPY arrays, one-hot labels, train/test split, and CNN training.

The reusable library code lives under `NN.API.Data`; these files are intentionally concrete and
meant to be read and run.
-/

@[expose] public section
