/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# Shared Model-Example Helpers

Shared utilities for runnable model examples. This layer is intentionally small: it should hold
data-path and loading helpers, not model architectures or training loops.
-/

@[expose] public section
