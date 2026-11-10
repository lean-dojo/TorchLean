/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Tolerance
public import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Core
public import FloatLib.Numerics.ErrorBound
public import Mathlib.Analysis.Calculus.Deriv.Basic
public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Tactic.Linarith

/-!
# Scalar error propagation

The rounded-real backend rounds each operation with `Flocq.round`, which moves a real by at most
half an ulp (`Flocq.error_bound_ulp`). This module defines the scalar absolute-error predicate
`ScalarApprox` and proves `ScalarApprox.roundedMul`, the bound for one rounded multiplication,
from FloatLib's `FloatLib.Numerics.mul_error`. For rounded addition we apply
`FloatLib.Numerics.add_error` directly. Tensor and graph proofs of the backend reuse these scalar
facts. Agreement with an external numerical backend is a separate claim.
-/

@[expose] public section

namespace TorchLean

open scoped Real

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

noncomputable section

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

/-! ## Scalar Approximation Predicate -/

/-- `ScalarApprox x xhat eps` says that the computed value `xhat` is within absolute error `eps`
of the exact real value `x`. -/
def ScalarApprox (x xhat eps : ℝ) : Prop :=
  abs (xhat - x) ≤ eps

/-- `ScalarApprox` implies the shared predicate `ApproxR` at the absolute-only tolerance
`ApproxTol.absOnly eps`. -/
theorem ScalarApprox.approxR_absOnly {x xhat eps : ℝ} (h : ScalarApprox x xhat eps) :
    ApproxR x xhat (ApproxTol.absOnly eps) :=
  (approxR_absOnly_iff ((abs_nonneg _).trans h)).2 h

/-! ## Rounded Multiplication -/

/--
Compositional absolute-error bound for rounded multiplication.

Besides the fresh rounding term for `xhat * yhat`, the budget includes magnitude-weighted input
errors and the conservative cross term `2 * epsx * epsy`, so it is not a first-order-only estimate.
Our derivation starts from the propagated error `|yhat| * epsx + |x| * epsy` that `mul_error`
gives. Bounding `abs x` by `abs xhat + epsx` removes the exact `x` at the cost of one
`epsx * epsy`, and the cross term's other `epsx * epsy` gives both operands one shape, a bound on
`|x|` or `|y|` times the other operand's error.
-/
theorem ScalarApprox.roundedMul {x y xhat yhat epsx epsy : ℝ}
    (hx : ScalarApprox x xhat epsx) (hy : ScalarApprox y yhat epsy) :
    ScalarApprox (x * y) (Flocq.round (β := β) (fexp := fexp) rnd (xhat * yhat))
      ((abs xhat + epsx) * epsy + (abs yhat + epsy) * epsx +
        ulp β fexp (xhat * yhat) / 2) := by
  have hepsx : 0 ≤ epsx := (abs_nonneg _).trans hx
  have hepsy : 0 ≤ epsy := (abs_nonneg _).trans hy
  have hx_abs : abs x ≤ abs xhat + epsx := by
    have htriangle : abs x ≤ abs (xhat - x) + abs xhat := by
      simpa only [sub_zero, abs_sub_comm] using abs_sub_le x xhat 0
    exact (htriangle.trans (add_le_add hx le_rfl)).trans_eq (add_comm _ _)
  have hbudget : epsx * abs yhat + abs x * epsy ≤
      (abs xhat + epsx) * epsy + (abs yhat + epsy) * epsx := by
    nlinarith [mul_le_mul_of_nonneg_right hx_abs hepsy, mul_nonneg hepsx hepsy]
  have hbound := FloatLib.Numerics.mul_error hx hy
    (error_bound_ulp (β := β) (fexp := fexp) rnd (xhat * yhat))
  exact hbound.trans (by nlinarith)

end

end TorchLean
