/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.IoLoader.Csv
public import NN.Runtime.Autograd.Train.IoLoader.Npy

/-!
# IO loaders for training datasets

`NN.Runtime.Autograd.Train.IoLoader` is the public umbrella for file-backed training loaders.

The implementation is split by responsibility:

- `IoLoader.Common` contains small shared parser utilities and safety limits.
- `IoLoader.Csv` contains CSV-to-tensor dataset readers.
- `IoLoader.Npy` contains the supported NumPy `.npy` subset for vectors and matrices.

Keeping this file as an umbrella preserves the public import path while avoiding a single large module
that mixes CSV tokenization, NumPy header parsing, byte decoding, and tensor conversion.
-/

@[expose] public section
