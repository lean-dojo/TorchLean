/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# FloatApprox

Finite absolute-tolerance comparisons for PINN artifacts and leaf witness-margin bookkeeping.
IBP interval containment and CROWN binary32 replay use separate, tolerance-free comparisons.
An approximate match here is not a proof that an interval encloses every real execution.
-/

@[expose] public section

namespace NN.Verification.Util

/-- Absolute-difference comparison on `Float`. -/
def approxEq (x y : Float) (tol : Float := 1e-6) : Bool :=
  x.isFinite && y.isFinite && tol.isFinite && decide (0.0 ≤ tol) &&
    decide (Float.abs (x - y) ≤ tol)

end NN.Verification.Util
