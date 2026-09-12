/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Train.Loop

/-!
# Training

Import aggregator for the training entry points. The implementation lives in
`NN.API.Trainer.Train.Loop`: `Trainer.train`, `predict`, `predictMany`, `load`, `trainStream`,
and `trainAlternating`, all written over `Trainer.Session`.
-/

@[expose] public section
