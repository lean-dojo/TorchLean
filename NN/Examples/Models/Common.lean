/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# Shared Model-Example Helpers

Shared data loaders and command runners for runnable model examples. `RealData` prepares typed
samples and missing-file hints; `Train` parses command options and calls the public trainer.
Model architectures remain in the family-specific modules.
-/

@[expose] public section
