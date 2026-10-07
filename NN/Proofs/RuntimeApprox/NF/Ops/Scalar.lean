/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import NN.Proofs.RuntimeApprox.Rounding.RoundingApprox
public import NN.Spec.Core.FloatInstances.NF
public import NN.Proofs.RuntimeApprox.NF.Ops.Plumbing -- shake: keep

/-!
# NF Scalar Primitive Bounds

Scalar bridge lemmas and forward-error bounds for rounded `NF` primitives, lifted pointwise
by the tensor proofs. This is a noncomputable rounded-real model, not a native-kernel proof.
An `NF` value need not be representable in its format, so even negation and absolute value
include a rounding term.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open Proofs.RuntimeRoundingApprox

section Interpretation

variable {β : Radix} {fexp : ℤ → ℤ} {rnd : ℝ → ℤ}

local notation "R" => NF β fexp rnd

/-- Interpret a runtime `NF` scalar as a spec scalar (`ℝ`) by forgetting rounding metadata. -/
@[inline] abbrev toSpec (x : R) : SpecScalar := NF.toReal x

/-- `max` on `NF` is a pure selection, so forgetting the format commutes with maximum exactly. -/
theorem toSpec_max (x y : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (max x y) =
      max (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) y) := by
  by_cases h : x ≤ y
  · have hReal : toSpec (β := β) (fexp := fexp) (rnd := rnd) x ≤
        toSpec (β := β) (fexp := fexp) (rnd := rnd) y := h
    rw [max_eq_right h, max_eq_right hReal]
  · have hyx : y ≤ x := le_of_not_ge h
    have hReal : toSpec (β := β) (fexp := fexp) (rnd := rnd) y ≤
        toSpec (β := β) (fexp := fexp) (rnd := rnd) x := hyx
    rw [max_eq_left hyx, max_eq_left hReal]

end Interpretation

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-- Constructing an `NF` value from a real incurs at most one half ULP.

This is the canonical bridge for rounded constants and casts. Keeping it next to `toSpec` avoids
repeating the implementation-level `NF.ofReal` unfolding in attention scales, optimizer
hyperparameters, and quantization parameters.
-/
theorem approx_ofReal_nf (x : ℝ) :
    abs
      (toSpec (β := β) (fexp := fexp) (rnd := rnd)
          (NF.ofReal (β := β) (fexp := fexp) (rnd := rnd) x) - x) ≤
      ulp β fexp x / 2 := by
  simpa [toSpec, NF.toReal, NF.ofReal,
    NF.roundR, Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error
        (β := β) (fexp := fexp) (rnd := rnd) x)

/-- Casting an exact rational constant rounds its real value once.

This supplies the epsilon-constant hypothesis for normalization without first rounding a
potentially unrepresentable natural denominator. Positivity still needs a separate argument.
-/
theorem approx_ratCast_nf (q : Rat) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (q : R) - (q : ℝ)) ≤
      ulp β fexp (q : ℝ) / 2 :=
  approx_ofReal_nf (β := β) (fexp := fexp) (rnd := rnd) (q : ℝ)

/-!
## Bridge lemmas from `NF` to $\mathbb R$

The bridge lemmas express each `NF` operation as a real calculation followed by rounding.
-/

/-- `toSpec` of runtime `0` is the spec scalar `0`: every valid rounder fixes the integer `0`. -/
@[simp] theorem toSpec_zero : toSpec (β := β) (fexp := fexp) (rnd := rnd) (0 : R) = (0 : ℝ) := by
  have hrnd0 : rnd (0 : ℝ) = 0 := by
    simpa using (ValidRnd.id (rnd := rnd) (n := (0 : ℤ)))
  -- `0 : R` is `NF.ofReal 0`, so `toSpec 0` is the rounding of the real `0`.
  change (NF.ofReal (β := β) (fexp := fexp) (rnd := rnd) (0 : ℝ)).val = (0 : ℝ)
  simp [NF.ofReal, NF.roundR, Flocq.round, Flocq.scaledMantissa, Flocq.toReal, hrnd0]

omit [ValidRndToNearest rnd] in
/--
`toSpec` respects runtime addition, up to an explicit rounding step.

This is the defining `NF` semantics: compute in `ℝ` and then apply `roundR`.
-/
private theorem toSpec_add (x y : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (x + y) =
      roundedAdd (β := β) (fexp := fexp) (rnd := rnd)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) y) := by
  rfl

omit [ValidRndToNearest rnd] in
/-- `toSpec` respects runtime multiplication, up to an explicit rounding step. -/
private theorem toSpec_mul (x y : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (x * y) =
      roundedMul (β := β) (fexp := fexp) (rnd := rnd)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) y) := by
  rfl

omit [ValidRndToNearest rnd] in
/-- `toSpec` respects runtime subtraction, up to an explicit rounding step. -/
private theorem toSpec_sub (x y : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (x - y) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) x -
          toSpec (β := β) (fexp := fexp) (rnd := rnd) y) := by
  rfl

omit [ValidRndToNearest rnd] in
/-- `toSpec` respects runtime negation, up to an explicit rounding step. -/
private theorem toSpec_neg (x : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (-x) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (-toSpec (β := β) (fexp := fexp) (rnd := rnd) x) := by
  simp [toSpec, NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
    NF.roundR, NF.ofReal, Neg.neg]

omit [ValidRndToNearest rnd] in
/-- `toSpec` respects runtime `exp`, up to an explicit rounding step. -/
theorem toSpec_exp (x : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.exp x) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (Real.exp (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)) := by
  simp [toSpec, NF.toReal, Proofs.RuntimeRoundingApprox.roundR, NF.roundR, NF.ofReal,
    Numerics.MathFunctions.exp]

omit [ValidRndToNearest rnd] in
/-- `toSpec` respects runtime `tanh`, up to an explicit rounding step. -/
private theorem toSpec_tanh (x : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.tanh x) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (Real.tanh (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)) := by
  simp [toSpec, NF.toReal, Proofs.RuntimeRoundingApprox.roundR, NF.roundR, NF.ofReal,
    Numerics.MathFunctions.tanh]

omit [ValidRndToNearest rnd] in
/-- `toSpec` respects runtime `sqrt`, up to an explicit rounding step. -/
private theorem toSpec_sqrt (x : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.sqrt x) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (Real.sqrt (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)) := by
  simp [toSpec, NF.toReal, Proofs.RuntimeRoundingApprox.roundR, NF.roundR, NF.ofReal,
    Numerics.MathFunctions.sqrt]

-- ---------------------------------------------------------------------------
-- Sqrt (clamped) approximation
-- ---------------------------------------------------------------------------

private theorem abs_sqrt_sub_sqrt_le_div_sqrt_of_le {a b η : ℝ} (ha : 0 ≤ a) (hη : 0 < η)
    (hb : η ≤ b) :
    abs (Real.sqrt a - Real.sqrt b) ≤ abs (a - b) / Real.sqrt η := by
  have hb0 : 0 < b := lt_of_lt_of_le hη hb
  have hsa_nonneg : 0 ≤ Real.sqrt a := Real.sqrt_nonneg a
  have hden_pos : 0 < Real.sqrt a + Real.sqrt b :=
    add_pos_of_nonneg_of_pos hsa_nonneg (Real.sqrt_pos.2 hb0)
  -- Rationalize: `√a - √b = (a - b) / (√a + √b)` because `(√a)² - (√b)² = a - b`.
  have hdiv : Real.sqrt a - Real.sqrt b = (a - b) / (Real.sqrt a + Real.sqrt b) := by
    rw [eq_div_iff hden_pos.ne', mul_comm, ← sq_sub_sq, Real.sq_sqrt ha, Real.sq_sqrt hb0.le]
  have hden_ge : Real.sqrt η ≤ Real.sqrt a + Real.sqrt b :=
    (Real.sqrt_le_sqrt hb).trans (le_add_of_nonneg_left hsa_nonneg)
  rw [hdiv, abs_div, abs_of_pos hden_pos]
  exact div_le_div_of_nonneg_left (abs_nonneg (a - b)) (Real.sqrt_pos.2 hη) hden_ge

/--
Forward approximation bound for `sqrt (max · 0)` under a positive lower bound.

This is a *clamped* sqrt bound: we work with `sqrt (max x 0)` to avoid the `sqrt` domain issue, but
still require a *strict* lower bound `η > 0` on `max x 0` to control conditioning via
`|√a - √b| ≤ |a-b| / √η`.
-/
theorem approx_sqrt_clamp_nf_of_lb {x : ℝ} {xR : R} {eps η : ℝ}
    (hη : 0 < η) (hdom : η ≤ max x 0)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.sqrt (max xR 0)) -
          Real.sqrt (max x 0)) ≤
      eps / Real.sqrt η +
        ulp β fexp
          (Real.sqrt (max (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) 0)) / 2 := by
  set xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  have hxhat : abs (xhat - x) ≤ eps := by
    simpa [xhat, abs_sub_comm] using hx
  have hmax : abs (max xhat 0 - max x 0) ≤ eps :=
    (abs_max_sub_max_le_abs xhat x 0).trans hxhat
  have hround :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.sqrt (max xR 0)) -
            Real.sqrt (max xhat 0)) ≤
        ulp β fexp (Real.sqrt (max xhat 0)) / 2 := by
    -- `sqrt` on NF is a single rounding of the real `sqrt`.
    have :
        toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.sqrt (max xR 0)) =
          Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
            (Real.sqrt (max xhat 0)) := by
      -- `max xR 0` is either `xR` or `0`; `toSpec` commutes with `max`.
      have hxmax :
          toSpec (β := β) (fexp := fexp) (rnd := rnd) (max xR (0 : R)) = max xhat 0 := by
        simpa [xhat] using
          (toSpec_max (β := β) (fexp := fexp) (rnd := rnd) xR (0 : R))
      simpa [xhat, hxmax] using
        (toSpec_sqrt (β := β) (fexp := fexp) (rnd := rnd) (max xR 0))
    simpa [this, Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (Real.sqrt
        (max xhat 0)))
  -- Conditioning of `sqrt` on `[η, ∞)`, then monotonicity of the numerator bound.
  have hdiff : abs (Real.sqrt (max xhat 0) - Real.sqrt (max x 0)) ≤ eps / Real.sqrt η :=
    (abs_sqrt_sub_sqrt_le_div_sqrt_of_le (le_max_right xhat 0) hη hdom).trans
      (div_le_div_of_nonneg_right hmax (Real.sqrt_nonneg η))
  have :=
    calc
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.sqrt (max xR 0)) -
            Real.sqrt (max x 0))
          ≤ abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (Numerics.MathFunctions.sqrt (max xR 0)) -
                Real.sqrt (max xhat 0)) +
              abs (Real.sqrt (max xhat 0) - Real.sqrt (max x 0)) := abs_sub_le _ _ _
      _ ≤ ulp β fexp (Real.sqrt (max xhat 0)) / 2 + eps / Real.sqrt η := add_le_add hround hdiff
      _ = eps / Real.sqrt η + ulp β fexp (Real.sqrt (max xhat 0)) / 2 := add_comm _ _
  simpa [xhat, add_comm, add_left_comm, add_assoc] using this

/--
Forward approximation bound for addition in `NF`.

In words: if `xR` approximates `x` within `epsx` and `yR` approximates `y` within `epsy`,
then `xR + yR` approximates `x + y` within `epsx + epsy + ulp(toSpec xR + toSpec yR)/2`.
-/
theorem approx_add_nf {x y : ℝ} {xR yR : R} {epsx epsy : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR + yR) - (x + y)) ≤
      epsx + epsy +
        ulp β fexp
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR +
              toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) / 2 := by
  have hx' :
      Proofs.RuntimeRoundingApprox.scalarApprox x
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) epsx := by
    simpa [Proofs.RuntimeRoundingApprox.scalarApprox] using hx
  have hy' :
      Proofs.RuntimeRoundingApprox.scalarApprox y
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) epsy := by
    simpa [Proofs.RuntimeRoundingApprox.scalarApprox] using hy
  have h := scalarApprox_roundedAdd (β := β) (fexp := fexp) (rnd := rnd) hx' hy'
  simpa [Proofs.RuntimeRoundingApprox.scalarApprox,
    toSpec_add (β := β) (fexp := fexp) (rnd := rnd) xR yR] using h

/--
Forward approximation bound for subtraction in `NF`.

The argument is the one behind `scalarApprox_roundedAdd`: the exact difference of the runtime
operands is within `epsx + epsy` of `x - y`, and the runtime result is one rounding away from that
exact difference.
-/
theorem approx_sub_nf {x y : ℝ} {xR yR : R} {epsx epsy : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR - yR) - (x - y)) ≤
      epsx + epsy +
        ulp β fexp
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR -
              toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) / 2 := by
  have hdiff :
      abs ((toSpec (β := β) (fexp := fexp) (rnd := rnd) xR -
          toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) - (x - y)) ≤ epsx + epsy := by
    rw [sub_sub_sub_comm]
    exact (abs_sub _ _).trans (add_le_add hx hy)
  rw [toSpec_sub (β := β) (fexp := fexp) (rnd := rnd) xR yR]
  exact ((abs_sub_le _
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR -
        toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) _).trans
    (add_le_add
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) _)
      hdiff)).trans_eq (by ring)

/-- Forward approximation bound for negation in `NF` (rounding error on `-toSpec xR`). -/
theorem approx_neg_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR) - (-x)) ≤
      eps +
        ulp β fexp
            (-toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) / 2 := by
  have hdiff : abs (-toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - (-x)) ≤ eps := by
    rw [neg_sub_neg, abs_sub_comm]
    exact hx
  rw [toSpec_neg (β := β) (fexp := fexp) (rnd := rnd) xR]
  exact ((abs_sub_le _ (-toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) _).trans
    (add_le_add
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) _)
      hdiff)).trans_eq (add_comm _ _)

/-- Forward approximation bound for absolute value in `NF` (`abs` is pure + a final rounding). -/
theorem approx_abs_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.abs xR) - abs x) ≤
      eps +
        ulp β fexp
            (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)) / 2 := by
  let xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  have hround :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.abs xR) - abs xhat) ≤
        ulp β fexp (abs xhat) / 2 := by
    -- `toSpec (abs xR)` is a single rounding of `|xhat|`.
    simpa [xhat, toSpec, Numerics.MathFunctions.abs, NF.instMathFunctions,
      NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
      NF.roundR, NF.ofReal] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (abs
        xhat))
  have hxhat : abs (xhat - x) ≤ eps := by
    simpa [xhat] using hx
  -- `abs` is `1`-Lipschitz, so the input error passes through unchanged.
  have hdiff : abs (abs xhat - abs x) ≤ eps :=
    (abs_abs_sub_abs_le_abs_sub xhat x).trans hxhat
  have :=
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.abs xR) - abs x)
          ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (Numerics.MathFunctions.abs xR) - abs xhat) +
              abs (abs xhat - abs x) := abs_sub_le _ _ _
      _ ≤ ulp β fexp (abs xhat) / 2 + eps := add_le_add hround hdiff
      _ = eps + ulp β fexp (abs xhat) / 2 := add_comm _ _
  simpa [xhat, add_assoc, add_left_comm, add_comm] using this

/-- Forward-error budget for one rounded exponential.

The first term propagates an input error `eps` through `exp`; the second pays for the final
rounding. In particular, exact input contributes no propagation error. This definition is shared
by scalar activations, tensor lifting, and stable axis softmax so that all three use the same
numerical contract.
-/
def expErrorBound (a eps : ℝ) : ℝ :=
  Real.exp (a + eps) * eps + ulp β fexp (Real.exp a) / 2

/--
Forward approximation bound for `exp` in `NF`.

Uses the mean value theorem for `Real.exp` to bound the propagation of input error, then adds one
rounding-ULP term for the final `NF` rounding.
-/
theorem approx_exp_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.exp xR) - Real.exp x) ≤
      expErrorBound (β := β) (fexp := fexp)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps := by
  let xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  have hinput : abs (xhat - x) ≤ eps := by
    simpa [xhat] using hx
  have heps : 0 ≤ eps := (abs_nonneg _).trans hinput

  -- Both `x` and `xhat` lie in the half-line `Iic (xhat + eps)` used for the mean value bound.
  have hx_le : x ≤ xhat + eps := by
    have h : x - xhat ≤ eps :=
      calc
        x - xhat ≤ abs (x - xhat) := le_abs_self _
        _ = abs (xhat - x) := abs_sub_comm _ _
        _ ≤ eps := hinput
    linarith

  have hxhat_le : xhat ≤ xhat + eps := le_add_of_nonneg_right heps

  have hderiv : ∀ z ∈ Set.Iic (xhat + eps),
      HasDerivWithinAt Real.exp (Real.exp z) (Set.Iic (xhat + eps)) z := by
    intro z _
    exact (Real.hasDerivAt_exp z).hasDerivWithinAt

  have hderivBound : ∀ z ∈ Set.Iic (xhat + eps),
      ‖Real.exp z‖ ≤ Real.exp (xhat + eps) := by
    intro z hz
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos z)]
    exact Real.exp_monotone hz

  have hmean :=
    Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := Real.exp) (f' := Real.exp) (s := Set.Iic (xhat + eps))
      (x := x) (y := xhat) (C := Real.exp (xhat + eps))
      hderiv hderivBound (convex_Iic (xhat + eps)) hx_le hxhat_le

  have hpropagation :
      abs (Real.exp xhat - Real.exp x) ≤ Real.exp (xhat + eps) * eps := by
    calc
      abs (Real.exp xhat - Real.exp x)
          ≤ Real.exp (xhat + eps) * abs (xhat - x) := by
              simpa [Real.norm_eq_abs] using hmean
      _ ≤ Real.exp (xhat + eps) * eps :=
        mul_le_mul_of_nonneg_left hinput (Real.exp_nonneg _)

  have hround :
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp xhat)
            -
            Real.exp xhat) ≤
        ulp β fexp (Real.exp xhat) / 2 := by
    simpa [Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (Real.exp
        xhat))

  have htotal :
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp xhat)
            -
            Real.exp x) ≤
        Real.exp (xhat + eps) * eps + ulp β fexp (Real.exp xhat) / 2 :=
    calc
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp
            xhat) -
            Real.exp x)
          ≤ abs
              (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp
                xhat) -
                Real.exp xhat) +
              abs (Real.exp xhat - Real.exp x) := abs_sub_le _ _ _
      _ ≤ ulp β fexp (Real.exp xhat) / 2 + Real.exp (xhat + eps) * eps :=
          add_le_add hround hpropagation
      _ = Real.exp (xhat + eps) * eps + ulp β fexp (Real.exp xhat) / 2 := add_comm _ _

  simpa [expErrorBound, xhat, toSpec_exp (β := β) (fexp := fexp) (rnd := rnd) xR, add_assoc,
    add_left_comm, add_comm]
    using htotal

/--
Forward approximation bound for `tanh` in `NF` (coarse but unconditional).

Because `tanh` is bounded in `[-1, 1]`, we always have `|tanh(toSpec xR) - tanh(x)| ≤ 2`, and then
we add one rounding-ULP term for the final `NF` rounding step.
-/
theorem approx_tanh_nf {x : ℝ} {xR : R} {eps : ℝ}
    (_hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (Numerics.MathFunctions.tanh xR) - Real.tanh x) ≤
      2 +
        ulp β fexp
            (Real.tanh (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)) / 2 := by
  let xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR

  have hround :
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh xhat)
            -
            Real.tanh xhat) ≤
        ulp β fexp (Real.tanh xhat) / 2 := by
    simpa [Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh
        xhat))

  -- `|tanh| < 1`, so two values of `tanh` are never more than `2` apart.
  have hdiff : abs (Real.tanh xhat - Real.tanh x) ≤ 2 :=
    calc
      abs (Real.tanh xhat - Real.tanh x) ≤ abs (Real.tanh xhat) + abs (Real.tanh x) := abs_sub _ _
      _ ≤ 1 + 1 := add_le_add (Real.abs_tanh_lt_one xhat).le (Real.abs_tanh_lt_one x).le
      _ = 2 := by norm_num

  have htotal :
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh xhat)
            -
            Real.tanh x) ≤
        2 + ulp β fexp (Real.tanh xhat) / 2 :=
    calc
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh
            xhat) -
            Real.tanh x)
          ≤ abs
              (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh
                xhat) -
                Real.tanh xhat) +
              abs (Real.tanh xhat - Real.tanh x) := abs_sub_le _ _ _
      _ ≤ ulp β fexp (Real.tanh xhat) / 2 + 2 := add_le_add hround hdiff
      _ = 2 + ulp β fexp (Real.tanh xhat) / 2 := add_comm _ _

  -- `hx` is not needed for the range-based bound; the statement mirrors the other unary lemmas.
  simpa [xhat, toSpec_tanh (β := β) (fexp := fexp) (rnd := rnd) xR, add_assoc, add_left_comm,
    add_comm]
    using htotal

-- ---------------------------------------------------------------------------
-- Safe log: `log (max x ε)` (needed for unconditional forward bounds)
-- ---------------------------------------------------------------------------

/--
Clamped log on spec scalars: `log (max x ε)`.

This is used to obtain unconditional forward bounds for `log` by avoiding the singularity at `0`.
-/
def safeLog (ε : ℝ) (x : ℝ) : ℝ :=
  Real.log (max x ε)

/--
Clamped log in the rounded-real `NF` model: evaluate `safeLog` and round once.
This definition is noncomputable; it does not implement a native logarithm.
-/
def safeLogR (ε : ℝ) (xR : R) : R :=
  NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (safeLog (ε := ε) (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR))

private theorem abs_log_sub_log_le_one_div_mul_abs_sub {ε u v : ℝ}
    (hε : 0 < ε) (hu : ε ≤ u) (hv : ε ≤ v) :
    abs (Real.log u - Real.log v) ≤ (1 / ε) * abs (u - v) := by
  -- Mean value theorem on `s = Ici ε` (derivative bounded by `1/ε`).
  have hf : ∀ x ∈ Set.Ici ε, HasDerivWithinAt Real.log (x⁻¹) (Set.Ici ε) x := by
    intro x hx
    have hx0 : x ≠ 0 := ne_of_gt (lt_of_lt_of_le hε hx)
    simpa using (Real.hasDerivAt_log (x := x) hx0).hasDerivWithinAt

  have hbound : ∀ x ∈ Set.Ici ε, ‖x⁻¹‖ ≤ (1 / ε) := by
    intro x hx
    have hxpos : 0 < x := lt_of_lt_of_le hε hx
    rw [Real.norm_eq_abs, abs_of_pos (inv_pos.2 hxpos), ← one_div]
    exact one_div_le_one_div_of_le hε hx

  have hmv :=
    Convex.norm_image_sub_le_of_norm_hasDerivWithin_le (f := Real.log) (f' := fun x : ℝ => x⁻¹)
      (s := Set.Ici ε) (x := u) (y := v) (C := (1 / ε))
      hf hbound (convex_Ici ε) hu hv

  simpa [Real.norm_eq_abs, abs_sub_comm] using hmv

/--
Forward approximation bound for `safeLog` in `NF`.

On the clamped domain `u,v ≥ ε > 0`, `log` is `(1/ε)`-Lipschitz. We use that to propagate the input
error and then add one rounding-ULP term for the final `NF` rounding.
-/
theorem approx_safeLog_nf {x : ℝ} {xR : R} {eps ε : ℝ}
    (hε : 0 < ε)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (safeLogR (β := β) (fexp := fexp) (rnd := rnd)
      ε xR) -
          safeLog (ε := ε) x) ≤
      (1 / ε) * eps +
        ulp β fexp
          (safeLog (ε := ε) (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)) / 2 := by
  set xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  set yhat : ℝ := max xhat ε
  set y : ℝ := max x ε

  have hyhat : ε ≤ yhat := le_max_right _ _
  have hy : ε ≤ y := le_max_right _ _

  have hmax : abs (yhat - y) ≤ eps := by
    have hx' : abs (xhat - x) ≤ eps := by
      simpa [xhat] using hx
    simpa [yhat, y] using (abs_max_sub_max_le_abs xhat x ε).trans hx'

  -- `log` is `(1/ε)`-Lipschitz on `[ε, ∞)`.
  have hdiff : abs (Real.log yhat - Real.log y) ≤ (1 / ε) * eps :=
    (abs_log_sub_log_le_one_div_mul_abs_sub hε hyhat hy).trans
      (mul_le_mul_of_nonneg_left hmax (one_div_nonneg.2 hε.le))

  have hround :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            Real.log yhat) ≤
        ulp β fexp (Real.log yhat) / 2 := by
    -- `safeLogR` rounds the real `log (max x̂ ε)`.
    have :
        toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) =
          Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.log yhat)
            := by
      simp [safeLogR, safeLog, toSpec, xhat, yhat, Proofs.RuntimeRoundingApprox.roundR,
        NF.toReal, NF.roundR, NF.ofReal]
    simpa [this] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (Real.log
        yhat))

  have :=
    calc
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            safeLog (ε := ε) x)
          ≤ abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                  (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
                Real.log yhat) +
              abs (Real.log yhat - safeLog (ε := ε) x) := abs_sub_le _ _ _
      _ ≤ ulp β fexp (Real.log yhat) / 2 + (1 / ε) * eps := by
            -- The second term is the `log` perturbation on the clamped inputs.
            have hlogy :
                abs (Real.log yhat - safeLog (ε := ε) x) = abs (Real.log yhat - Real.log y) := by
              simp [safeLog, y]
            rw [hlogy]
            exact add_le_add hround hdiff
      _ = (1 / ε) * eps + ulp β fexp (safeLog (ε := ε) xhat) / 2 := by
            simp [safeLog, xhat, yhat, add_comm]
  simpa [xhat] using this

/-- Forward approximation bound for ordinary square root on a certified positive domain.

Nonnegativity of the represented input lets the clamped bound reduce to ordinary square root.
The underlying `Real.sqrt` is total, returning zero for negative arguments; this hypothesis
describes the theorem's domain, not an IEEE invalid-operation check.
-/
theorem approx_sqrt_nf_of_pos_lb {x : ℝ} {xR : R} {eps η : ℝ}
    (hη : 0 < η) (hdom : η ≤ x)
    (hxR : 0 ≤ toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.sqrt xR) -
          Real.sqrt x) ≤
      eps / Real.sqrt η +
        ulp β fexp
          (Real.sqrt (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)) / 2 := by
  have hx0 : 0 ≤ x := hη.le.trans hdom
  -- The order on `NF` is the order of the underlying reals, and runtime `0` reads as real `0`.
  have hxR0 : (0 : R) ≤ xR := by
    change toSpec (β := β) (fexp := fexp) (rnd := rnd) (0 : R) ≤
      toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
    rw [toSpec_zero]
    exact hxR
  have hmaxR : max xR (0 : R) = xR := max_eq_left hxR0
  have hmaxS : max x 0 = x := max_eq_left hx0
  have h := approx_sqrt_clamp_nf_of_lb (β := β) (fexp := fexp) (rnd := rnd)
    (x := x) (xR := xR) (eps := eps) (η := η) hη (by simpa [hmaxS] using hdom) hx
  simpa [hmaxR, hmaxS, hxR] using h

/-- Square-root approximation when the input error budget itself certifies runtime positivity.

From `η ≤ x`, `|x̂-x| ≤ eps`, and `eps < η`, the rounded input satisfies `0 < x̂`; callers do not
need a separate runtime-domain hypothesis. This is the form used by normalization certificates.
-/
theorem approx_sqrt_nf_of_pos_lb_of_error {x : ℝ} {xR : R} {eps η : ℝ}
    (hη : 0 < η) (hdom : η ≤ x) (hbudget : eps < η)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.sqrt xR) -
          Real.sqrt x) ≤
      eps / Real.sqrt η +
        ulp β fexp
          (Real.sqrt (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)) / 2 := by
  have hxR : 0 ≤ toSpec (β := β) (fexp := fexp) (rnd := rnd) xR := by
    have hdiff : x - toSpec (β := β) (fexp := fexp) (rnd := rnd) xR ≤ eps := by
      calc
        x - toSpec (β := β) (fexp := fexp) (rnd := rnd) xR ≤
            abs (x - toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) := le_abs_self _
        _ = abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) := abs_sub_comm _ _
        _ ≤ eps := hx
    linarith
  exact approx_sqrt_nf_of_pos_lb (β := β) (fexp := fexp) (rnd := rnd)
    hη hdom hxR hx

/--
Forward approximation bound for multiplication in `NF`.

Includes both input perturbations, their cross terms, and one half-ULP rounding term.
-/
theorem approx_mul_nf {x y : ℝ} {xR yR : R} {epsx epsy : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR * yR) - (x * y)) ≤
      ((abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) + epsx) * epsy +
        (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) + epsy) * epsx +
        ulp β fexp
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR *
              toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) / 2) := by
  have hx' :
      Proofs.RuntimeRoundingApprox.scalarApprox x
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) epsx := by
    simpa [Proofs.RuntimeRoundingApprox.scalarApprox] using hx
  have hy' :
      Proofs.RuntimeRoundingApprox.scalarApprox y
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) epsy := by
    simpa [Proofs.RuntimeRoundingApprox.scalarApprox] using hy
  have h := scalarApprox_roundedMul (β := β) (fexp := fexp) (rnd := rnd) hx' hy'
  simpa [Proofs.RuntimeRoundingApprox.scalarApprox,
    toSpec_mul (β := β) (fexp := fexp) (rnd := rnd) xR yR] using h

/-- Forward approximation bound for scaling (elementwise multiply by a runtime constant `c`). -/
theorem approx_scale_nf {x : ℝ} {xR : R} {eps : ℝ} (c : R)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR * c) - (x * toSpec (β := β) (fexp := fexp)
      (rnd := rnd) c)) ≤
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) c) * eps +
        ulp β fexp
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR *
              toSpec (β := β) (fexp := fexp) (rnd := rnd) c) / 2 := by
  have hc : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) c -
        toSpec (β := β) (fexp := fexp) (rnd := rnd) c) ≤ (0 : ℝ) := by
    simp
  have h :=
    approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd)
      (x := x) (y := toSpec (β := β) (fexp := fexp) (rnd := rnd) c)
      (xR := xR) (yR := c) (epsx := eps) (epsy := (0 : ℝ)) hx hc
  simpa [mul_assoc, add_assoc, add_left_comm, add_comm] using h

end NFBackend

end

end RuntimeApprox
end Proofs
