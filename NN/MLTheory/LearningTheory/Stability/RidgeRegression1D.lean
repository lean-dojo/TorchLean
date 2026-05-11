/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.Real

/-!
# 1D ridge regression (stability): umbrella import

The core real-analysis theorem lives in:

- `NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.Real`

We keep this short umbrella module so downstream code can simply:

`import NN.MLTheory.LearningTheory.Stability.RidgeRegression1D`

without needing to remember the internal file split.

The executable float32 development is separate (to avoid pulling in float semantics unless needed):

- `NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec`.
-/

@[expose] public section


