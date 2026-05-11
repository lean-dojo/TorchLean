/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Utils
public import NN.Runtime.Autograd.Train

/-!
# Autograd training helpers (umbrella import)

This module is the one-stop umbrella for training scripts and runtime tests.
It gathers the small helpers that sit on top of the dynamic autograd tape and keeps the import
surface stable:

* `Runtime.Autograd.Train.*` - general helpers (error tagging, gradient/value access, SGD)
* `Runtime.Autograd.Train.Dataset` - dataset wrapper and deterministic dataloader
* `Runtime.Autograd.Train.Eval` - evaluation helpers over datasets/batches
* `Runtime.Autograd.Train` IO - CSV/NPY loaders for small self-contained datasets
* `Runtime.Autograd.Train.Logger` - pluggable logging helpers
* `Runtime.Autograd.Train.Optim` - optimizer and scheduler integration
* `Runtime.Autograd.Train.Trainer` - higher-level trainer API with logging and metrics
* `Runtime.Autograd.Train.TapeM.*` - TapeM helpers for params, constants, and batch losses
* tensor loader helpers in `Runtime.Autograd.Train` - `vectorOfList`, `matrixOfLists`,
  `datasetOfPairs`

`NN.Runtime.Autograd.Train` is the focused training-helper umbrella. This file is a broader
autograd utility umbrella because it also re-exports `NN.Runtime.Autograd.Torch.Utils`.
-/

@[expose] public section
