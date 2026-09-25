/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Core.Tolerance
public import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Core

/-!
# Scalar Rounding Approximation

Rounding-level approximation lemmas.

This module begins the runtime-to-spec bridge. It gives compositional error bounds for expressions
evaluated under a declared `Flocq.round` rounded-real model such as `NF`.

These lemmas are scalar-level and can be lifted to tensors/graphs once a concrete set of ops is
fixed (MLP first, then larger models).

## PyTorch correspondence / citations
In ordinary PyTorch execution, floating-point ops are performed in a chosen dtype (e.g. `float32`)
with hardware/IEEE-754 rounding. In TorchLean, `Flocq.round`/`NF` is a proof-relevant rounding model
where each operation exposes an explicit `ulp`-style error bound suitable for composition.
https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
-/

@[expose] public section


namespace Proofs
namespace RuntimeRoundingApprox

open scoped Real

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

noncomputable section

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

/-! ## Scalar Approximation Predicate -/

/--
Absolute-error approximation for scalar real values.

`scalarApprox x xhat eps` means the rounded/interpreted value `xhat` is within absolute error `eps`
of the ideal real value `x`.
-/
def scalarApprox (x xhat eps : ℝ) : Prop :=
  abs (xhat - x) ≤ eps

/-- Convert the scalar rounding predicate to the shared `ApproxTol.absOnly` predicate. -/
theorem scalarApprox_to_approxR_absOnly {x xhat eps : ℝ} (h : scalarApprox x xhat eps) :
    Proofs.RuntimeApprox.approxR x xhat (Proofs.RuntimeApprox.ApproxTol.absOnly eps) :=
  (RuntimeApprox.approxR_absOnly_iff ((abs_nonneg _).trans h)).2 h

/-- Exact equality is zero-error scalar approximation. -/
theorem scalarApprox_refl_zero (x : ℝ) : scalarApprox x x 0 := by
  simp [scalarApprox]

/-- Enlarging the error budget preserves scalar approximation. -/
theorem scalarApprox_mono {x xhat eps₁ eps₂ : ℝ} (h : scalarApprox x xhat eps₁) (hε : eps₁ ≤ eps₂) :
    scalarApprox x xhat eps₂ :=
  le_trans h hε

/-! ## Single-Step Rounding Bounds -/

/-- Interpret one rounded real operation as `Flocq.round` applied to a real input. -/
def roundR (x : ℝ) : ℝ :=
  Flocq.round (β := β) (fexp := fexp) rnd x

/-- One `Flocq.round` step is within half an ulp of the exact real input. -/
theorem roundR_abs_error (x : ℝ) :
    abs (roundR (β := β) (fexp := fexp) (rnd := rnd) x - x) ≤
      ulp β fexp x / 2 :=
  error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) x

/-! ## Compositional Bounds For `+` And `*` -/

/-- Rounded addition: compute `roundR (x + y)`. -/
def roundedAdd (x y : ℝ) : ℝ :=
  roundR (β := β) (fexp := fexp) (rnd := rnd) (x + y)

/-- Rounded multiplication: compute `roundR (x * y)`. -/
def roundedMul (x y : ℝ) : ℝ :=
  roundR (β := β) (fexp := fexp) (rnd := rnd) (x * y)

/--
Compositional absolute-error bound for rounded addition.

The output budget is the input budgets plus one fresh rounding term for the addition result.
-/
theorem scalarApprox_roundedAdd {x y xhat yhat epsx epsy : ℝ}
    (hx : scalarApprox x xhat epsx) (hy : scalarApprox y yhat epsy) :
    scalarApprox (x + y) (roundedAdd (β := β) (fexp := fexp) (rnd := rnd) xhat yhat)
      (epsx + epsy + ulp β fexp (xhat + yhat) / 2) := by
  have hsum : abs ((xhat + yhat) - (x + y)) ≤ epsx + epsy := by
    rw [add_sub_add_comm]
    exact (abs_add_le _ _).trans (add_le_add hx hy)
  exact ((abs_sub_le _ (xhat + yhat) _).trans
    (add_le_add (roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) _) hsum)).trans_eq
      (add_comm _ _)

/--
Compositional absolute-error bound for rounded multiplication.

Besides the fresh rounding term for `xhat * yhat`, the budget includes the usual first-order
product perturbation terms using the available magnitude/error bounds.
-/
theorem scalarApprox_roundedMul {x y xhat yhat epsx epsy : ℝ}
    (hx : scalarApprox x xhat epsx) (hy : scalarApprox y yhat epsy) :
    scalarApprox (x * y) (roundedMul (β := β) (fexp := fexp) (rnd := rnd) xhat yhat)
      ((abs xhat + epsx) * epsy + (abs yhat + epsy) * epsx +
        ulp β fexp (xhat * yhat) / 2) := by
  have hepsx : 0 ≤ epsx := (abs_nonneg _).trans hx
  have hepsy : 0 ≤ epsy := (abs_nonneg _).trans hy
  have hx_abs : abs x ≤ abs xhat + epsx := by
    have htriangle : abs x ≤ abs (xhat - x) + abs xhat := by
      simpa only [sub_zero, abs_sub_comm] using abs_sub_le x xhat 0
    exact (htriangle.trans (add_le_add hx le_rfl)).trans_eq (add_comm _ _)
  -- Split the perturbation before adding FloatLib's local rounding bound.
  have hpert : abs (xhat * yhat - x * y) ≤ epsx * abs yhat + abs x * epsy := by
    calc
      abs (xhat * yhat - x * y)
          = abs ((xhat - x) * yhat + x * (yhat - y)) := by congr 1; ring
      _ ≤ abs ((xhat - x) * yhat) + abs (x * (yhat - y)) := abs_add_le _ _
      _ = abs (xhat - x) * abs yhat + abs x * abs (yhat - y) := by rw [abs_mul, abs_mul]
      _ ≤ epsx * abs yhat + abs x * epsy :=
        add_le_add (mul_le_mul_of_nonneg_right hx (abs_nonneg _))
          (mul_le_mul_of_nonneg_left hy (abs_nonneg _))
  have hbudget : epsx * abs yhat + abs x * epsy ≤
      (abs xhat + epsx) * epsy + (abs yhat + epsy) * epsx := by
    nlinarith [mul_le_mul_of_nonneg_right hx_abs hepsy, mul_nonneg hepsx hepsy]
  exact ((abs_sub_le _ (xhat * yhat) _).trans
    (add_le_add (roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) _)
      (hpert.trans hbudget))).trans_eq (add_comm _ _)

end

end RuntimeRoundingApprox
end Proofs
