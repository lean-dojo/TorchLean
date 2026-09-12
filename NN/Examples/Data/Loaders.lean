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

This umbrella collects the Lean side data tutorials:

- `Csv`: numeric CSV rows and minibatched regression training;
- `Npy`: NumPy/PyTorch `.npy` arrays, metadata inspection, and minibatched regression training;
- `Cifar10Images`: image-shaped NPY arrays, one-hot labels, train/test split, and CNN training.

The reusable library code is available through `TorchLean.Data`; these files are concrete tutorial
entrypoints that can be read, edited, and run.
-/

@[expose] public section
