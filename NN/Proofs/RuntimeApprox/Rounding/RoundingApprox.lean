/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.ErrorBounds
public import NN.Proofs.RuntimeApprox.Core.Tolerance

public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Scalar Rounding Approximation

Rounding-level approximation lemmas.

This module is the **start** of the runtime→spec bridge: it gives compositional error bounds for
expressions evaluated under a `neural_round` model (e.g. NF / mixed-precision models).

These lemmas are scalar-level and can be lifted to tensors/graphs once a concrete set of ops is
fixed (MLP first, then larger models).

## PyTorch correspondence / citations
In ordinary PyTorch execution, floating-point ops are performed in a chosen dtype (e.g. `float32`)
with hardware/IEEE-754 rounding. In TorchLean, `neuralRound`/`NF` is a proof-relevant rounding model
where each operation exposes an explicit `ulp`-style error bound suitable for composition.
https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
-/

@[expose] public section


namespace Proofs
namespace RuntimeRoundingApprox

open scoped Real

open TorchLean.Floats

noncomputable section

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

/-! ## Scalar Approximation Predicate -/

/--
Absolute-error approximation for scalar real values.

`scalarApprox x xhat eps` means the rounded/interpreted value `xhat` is within absolute error `eps`
of the ideal real value `x`.
-/
def scalarApprox (x xhat eps : ℝ) : Prop :=
  abs (xhat - x) ≤ eps

/-- Convert the scalar rounding predicate to the shared `ApproxTol.absOnly` predicate. -/
lemma scalarApprox_to_approxR_absOnly {x xhat eps : ℝ} (h : scalarApprox x xhat eps) :
    Proofs.RuntimeApprox.approxR x xhat (Proofs.RuntimeApprox.ApproxTol.absOnly eps) := by
  -- `ApproxTol.absOnly eps` uses `Real.toNNReal eps` (i.e. `max eps 0`) as the budget.
  have h' : abs (xhat - x) ≤ max eps 0 := le_trans h (le_max_left _ _)
  have h'' : abs (xhat - x) ≤ (Real.toNNReal eps : ℝ) := by
    simpa [Real.coe_toNNReal'] using h'
  simpa [Proofs.RuntimeApprox.approxR, Proofs.RuntimeApprox.approxBound_absOnly, abs_sub_comm] using
    h''

/-- Exact equality is zero-error scalar approximation. -/
lemma scalarApprox_refl_zero (x : ℝ) : scalarApprox x x 0 := by
  simp [scalarApprox]

/-- Enlarging the error budget preserves scalar approximation. -/
lemma scalarApprox_mono {x xhat eps₁ eps₂ : ℝ} (h : scalarApprox x xhat eps₁) (hε : eps₁ ≤ eps₂) :
    scalarApprox x xhat eps₂ :=
  le_trans h hε

/-! ## Single-Step Rounding Bounds -/

/-- Interpret one rounded real operation as `neuralRound` applied to a real input. -/
def roundR (x : ℝ) : ℝ :=
  neuralRound (β := β) (fexp := fexp) rnd x

/-- One `neuralRound` step is within half an ulp of the exact real input. -/
lemma roundR_abs_error (x : ℝ) :
    abs (roundR (β := β) (fexp := fexp) (rnd := rnd) x - x) ≤
      neuralUlp β fexp x TrainingPhase.forward / 2 := by
  simpa [roundR] using neural_error_bound_ulp (β := β) (fexp := fexp) (rnd := rnd) x

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
lemma scalarApprox_roundedAdd {x y xhat yhat epsx epsy : ℝ}
    (hx : scalarApprox x xhat epsx) (hy : scalarApprox y yhat epsy) :
    scalarApprox (x + y) (roundedAdd (β := β) (fexp := fexp) (rnd := rnd) xhat yhat)
      (epsx + epsy + neuralUlp β fexp (xhat + yhat) TrainingPhase.forward / 2) := by
  -- Triangle inequality: (rounded(x̂+ŷ) - (x+y)) =
  --   (rounded(x̂+ŷ) - (x̂+ŷ)) + ((x̂+ŷ) - (x+y)).
  have hround :
      abs (roundedAdd (β := β) (fexp := fexp) (rnd := rnd) xhat yhat - (xhat + yhat)) ≤
        neuralUlp β fexp (xhat + yhat) TrainingPhase.forward / 2 := by
    simpa [roundedAdd, roundR] using
      roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (xhat + yhat)

  have hsum :
      abs ((xhat + yhat) - (x + y)) ≤ epsx + epsy := by
    -- |(xhat-x) + (yhat-y)| ≤ |xhat-x| + |yhat-y|
    have hx' : abs (xhat - x) ≤ epsx := hx
    have hy' : abs (yhat - y) ≤ epsy := hy
    simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using
      (abs_add_le (xhat - x) (yhat - y) |>.trans (add_le_add hx' hy'))

  have :=
    calc
      abs (roundedAdd (β := β) (fexp := fexp) (rnd := rnd) xhat yhat - (x + y))
          ≤ abs (roundedAdd (β := β) (fexp := fexp) (rnd := rnd) xhat yhat - (xhat + yhat))
              + abs ((xhat + yhat) - (x + y)) := by
                simpa [sub_eq_add_neg, add_assoc] using
                  abs_sub_le (roundedAdd (β := β) (fexp := fexp) (rnd := rnd) xhat yhat)
                    (xhat + yhat) (x + y)
      _ ≤ neuralUlp β fexp (xhat + yhat) TrainingPhase.forward / 2 + (epsx + epsy) := by
            exact add_le_add hround hsum
      _ = epsx + epsy + neuralUlp β fexp (xhat + yhat) TrainingPhase.forward / 2 := by ring
  simpa [scalarApprox, roundedAdd, sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using this

/--
Compositional absolute-error bound for rounded multiplication.

Besides the fresh rounding term for `xhat * yhat`, the budget includes the usual first-order
product perturbation terms using the available magnitude/error bounds.
-/
lemma scalarApprox_roundedMul {x y xhat yhat epsx epsy : ℝ}
    (hx : scalarApprox x xhat epsx) (hy : scalarApprox y yhat epsy) :
    scalarApprox (x * y) (roundedMul (β := β) (fexp := fexp) (rnd := rnd) xhat yhat)
      ((abs xhat + epsx) * epsy + (abs yhat + epsy) * epsx +
        neuralUlp β fexp (xhat * yhat) TrainingPhase.forward / 2) := by
  have hx' : abs (xhat - x) ≤ epsx := hx
  have hy' : abs (yhat - y) ≤ epsy := hy
  have hepsx : 0 ≤ epsx := le_trans (abs_nonneg (xhat - x)) hx'
  have hepsy : 0 ≤ epsy := le_trans (abs_nonneg (yhat - y)) hy'

  have hround :
      abs (roundedMul (β := β) (fexp := fexp) (rnd := rnd) xhat yhat - (xhat * yhat)) ≤
        neuralUlp β fexp (xhat * yhat) TrainingPhase.forward / 2 := by
    simpa [roundedMul, roundR] using
      roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (xhat * yhat)

  -- Bound |x| and |y| by rounded magnitudes + error.
  have hx_abs : abs x ≤ abs xhat + epsx := by
    -- |x| = |xhat - (xhat - x)| ≤ |xhat| + |xhat - x|
    have h : abs x ≤ abs xhat + abs (xhat - x) := by
      have h' : abs x ≤ abs (x - xhat) + abs xhat := by
        simpa using (abs_sub_le x xhat 0)
      simpa [abs_sub_comm, add_comm, add_left_comm, add_assoc] using h'
    -- Add `abs xhat` to the inequality `|xhat - x| ≤ epsx`.
    have h' : abs xhat + abs (xhat - x) ≤ abs xhat + epsx := by
      linarith [hx']
    exact le_trans h h'

  have hy_abs : abs y ≤ abs yhat + epsy := by
    have h : abs y ≤ abs yhat + abs (yhat - y) := by
      have h' : abs y ≤ abs (y - yhat) + abs yhat := by
        simpa using (abs_sub_le y yhat 0)
      simpa [abs_sub_comm, add_comm, add_left_comm, add_assoc] using h'
    have h' : abs yhat + abs (yhat - y) ≤ abs yhat + epsy := by
      linarith [hy']
    exact le_trans h h'

  -- Product perturbation:
  -- x̂*ŷ - x*y = (x̂-x)*ŷ + x*(ŷ-y)
  have hpert :
      abs (xhat * yhat - x * y) ≤ (abs yhat) * epsx + (abs x) * epsy := by
    -- Rewrite difference and apply `abs_add`.
    have :
        xhat * yhat - x * y = (xhat - x) * yhat + x * (yhat - y) := by
      ring
    calc
      abs (xhat * yhat - x * y)
          = abs ((xhat - x) * yhat + x * (yhat - y)) := by
              simp [this]
      _ ≤ abs ((xhat - x) * yhat) + abs (x * (yhat - y)) := abs_add_le _ _
      _ = abs (xhat - x) * abs yhat + abs x * abs (yhat - y) := by
            simp [abs_mul]
      _ ≤ (epsx * abs yhat) + (abs x * epsy) := by
            exact add_le_add (mul_le_mul_of_nonneg_right hx' (abs_nonneg yhat))
              (mul_le_mul_of_nonneg_left hy' (abs_nonneg x))
      _ = abs yhat * epsx + abs x * epsy := by ring

  -- Combine rounding error + perturbation.
  have :=
    calc
      abs (roundedMul (β := β) (fexp := fexp) (rnd := rnd) xhat yhat - x * y)
          ≤ abs (roundedMul (β := β) (fexp := fexp) (rnd := rnd) xhat yhat - (xhat * yhat))
              + abs (xhat * yhat - x * y) := by
                simpa [sub_eq_add_neg, add_assoc] using
                  abs_sub_le (roundedMul (β := β) (fexp := fexp) (rnd := rnd) xhat yhat)
                    (xhat * yhat) (x * y)
      _ ≤ neuralUlp β fexp (xhat * yhat) TrainingPhase.forward / 2 + ((abs yhat) * epsx + (abs x) *
        epsy) := by
            exact add_le_add hround hpert
      _ ≤ neuralUlp β fexp (xhat * yhat) TrainingPhase.forward / 2 +
            (abs yhat * epsx + (abs xhat + epsx) * epsy) := by
            have hx_mul : abs x * epsy ≤ (abs xhat + epsx) * epsy :=
              mul_le_mul_of_nonneg_right hx_abs hepsy
            have hpert' :
                abs yhat * epsx + abs x * epsy ≤ abs yhat * epsx + (abs xhat + epsx) * epsy := by
              linarith [hx_mul]
            linarith [hpert']
      _ ≤ neuralUlp β fexp (xhat * yhat) TrainingPhase.forward / 2 +
            ((abs yhat + epsy) * epsx + (abs xhat + epsx) * epsy) := by
            have hy_le : abs yhat ≤ abs yhat + epsy := by linarith
            have hy_mul : abs yhat * epsx ≤ (abs yhat + epsy) * epsx :=
              mul_le_mul_of_nonneg_right hy_le hepsx
            have hsum :
                abs yhat * epsx + (abs xhat + epsx) * epsy ≤
                  (abs yhat + epsy) * epsx + (abs xhat + epsx) * epsy := by
              linarith [hy_mul]
            linarith [hsum]
      _ = (abs xhat + epsx) * epsy + (abs yhat + epsy) * epsx +
            neuralUlp β fexp (xhat * yhat) TrainingPhase.forward / 2 := by ring

  simpa [scalarApprox, roundedMul, sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using this

end

end RuntimeRoundingApprox
end Proofs
