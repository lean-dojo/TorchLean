/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import NN.Tests.Runtime.TypedGraphScalingRegression

/-- Standalone runner for the regressions also included in the maintained Float autograd suite. -/
def main : IO Unit := TypedGraphScalingRegression.run
