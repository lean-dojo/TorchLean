/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Verification.Core
public import NN.API.Trainer.Train.Loop
public import NN.API.Verification.Execution

/-!
# Verification

Public API for checking a trained model over an input region:

```lean
let trained ← trainer.train dataset { steps := 100 }
let result ← trained.verify input (radius := 0.05) (norm := .inf)
```

Use `(property := .topLabel label)` to certify one output label. Explicit graph construction and
inspection live in `NN.API.Verification.Lowering`.
-/
