/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import NN.Tests.Runtime.TypedGraphScalingRegression

/-!
Standalone executable for `TypedGraphScalingRegression.run`. Run this module directly for
focused TypedGraph regressions. Test suites should import `TypedGraphScalingRegression`;
the executable entry point `main` is defined here.
-/

/-- Standalone runner for the regressions also included in the maintained Float autograd suite. -/
def main : IO Unit := TypedGraphScalingRegression.run
