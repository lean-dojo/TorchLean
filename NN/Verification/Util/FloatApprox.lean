/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# FloatApprox

Small numeric comparison helpers used by verification checkers.

Motivation:
- Many checkers compare a Lean-recomputed bound to a Python-exported certificate.
- Those certificates are typically serialized as decimal floats, so equality is unrealistic.

These helpers stay local to the verification layer.  They model the practical comparison we need
for exported certificates, not a general-purpose analysis library.
-/

@[expose] public section

namespace NN.Verification.Util

/-- Absolute-difference comparison on `Float`. -/
def approxEq (x y : Float) (tol : Float := 1e-6) : Bool :=
  let d := (if x > y then x - y else y - x)
  decide (d ≤ tol)

end NN.Verification.Util
