/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Public.NN
public import NN.API.Public.TensorPack
public import NN.API.Public.Seeded
public import NN.API.Public.Autograd

/-!
# API Public

Clean public API for TorchLean model code.

User code should use the umbrella import:

```lean
import NN
open TorchLean
```

This module sits one level below the short public names. Those names come through
`NN.Entrypoint.API` and the `NN` umbrella:

- `TorchLean.nn` (model/layer builders)
- `TorchLean.optim` (optimizer configs)
- `TorchLean.Trainer` (train/evaluate APIs)
- `TorchLean.Data` (datasets/loaders + CSV/NPY readers)
- `TorchLean.Loss` and `TorchLean.Metrics`

Import `NN.API.Public` directly when extending TorchLean or working below the `NN` umbrella. Use
`NN.Entrypoint.API` when you only want the focused `TorchLean.*` names.

The callback-heavy training namespace lives in `NN.API.Public.Training`. It is deliberately not
re-exported from this umbrella module: ordinary code should get training through `TorchLean.Trainer`,
while files that truly need callback runners should import the advanced training
module explicitly.
-/
