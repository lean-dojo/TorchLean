/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import TorchLean.Verify.RuntimeApprox.Rounding
public import TorchLean.Tensor.NF
public import TorchLean.Verify.RuntimeApprox.ApproxTensor -- shake: keep
import FloatLib.Numerics.MixedError.Elementary

/-!
# NF Scalar Primitive Bounds

Bridge lemmas and forward-error bounds for the scalar primitives of `NF`, which the tensor proofs
lift pointwise. `NF` is FloatLib's noncomputable rounded-real model, and native kernels are outside
these bounds. An `NF` value need not lie on its format's grid, so even negation and absolute value
pay a rounding term.
-/

@[expose] public section

namespace TorchLean
namespace NF

open TorchLean.Tensor

noncomputable section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open ScalarApprox

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-- Constructing an `NF` value from a real incurs at most half an ULP.

Rounded constants and casts, such as the dimension cast of the attention scale, the constant `1`
and the rounded quotient of a mean, take their error from this bound instead of unfolding
`NF.ofReal`.
-/
theorem approx_ofReal (x : ℝ) :
    abs
      (NF.toReal
          (NF.ofReal (β := β) (fexp := fexp) (rnd := rnd) x) - x) ≤
      ulp β fexp x / 2 := by
  simpa [NF.toReal, NF.ofReal, NF.roundR] using
    Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd x

/-!
## Bridge lemmas from `NF` to $\mathbb R$

The bridge lemmas express each `NF` operation as a real calculation followed by rounding.
-/

/-- `NF.toReal` of runtime `0` is the real `0`: every valid rounder fixes the integer `0`. -/
@[simp] theorem toReal_zero : NF.toReal (0 : R) = (0 : ℝ) := by
  have hrnd0 : rnd (0 : ℝ) = 0 := by
    simpa using (ValidRnd.id (rnd := rnd) (n := (0 : ℤ)))
  -- `0 : R` is `NF.ofReal 0`, so `NF.toReal 0` is the rounding of the real `0`.
  change (NF.ofReal (β := β) (fexp := fexp) (rnd := rnd) (0 : ℝ)).val = (0 : ℝ)
  simp [NF.ofReal, NF.roundR, Flocq.round, Flocq.scaledMantissa, Flocq.toReal, hrnd0]

omit [ValidRndToNearest rnd] in
/--
`NF.toReal (x + y)` is `Flocq.round` of `NF.toReal x + NF.toReal y`. That is the defining `NF`
semantics: add in `ℝ`, then apply `Flocq.round`.
-/
private theorem toReal_add (x y : R) :
    NF.toReal (x + y) =
      Flocq.round (β := β) (fexp := fexp) rnd
        (NF.toReal x +
          NF.toReal y) := by
  rfl

omit [ValidRndToNearest rnd] in
/-- `NF.toReal` of a runtime product is `Flocq.round` of the real product. -/
private theorem toReal_mul (x y : R) :
    NF.toReal (x * y) =
      Flocq.round (β := β) (fexp := fexp) rnd
        (NF.toReal x *
          NF.toReal y) := by
  rfl

omit [ValidRndToNearest rnd] in
/-- `NF.toReal` of a runtime difference is `Flocq.round` of the real difference. -/
private theorem toReal_sub (x y : R) :
    NF.toReal (x - y) =
      Flocq.round (β := β) (fexp := fexp) rnd
        (NF.toReal x -
          NF.toReal y) := by
  rfl

omit [ValidRndToNearest rnd] in
/-- `NF.toReal` of a runtime negation is `Flocq.round` of the negated real value. -/
private theorem toReal_neg (x : R) :
    NF.toReal (-x) =
      Flocq.round (β := β) (fexp := fexp) rnd
        (-NF.toReal x) := by
  simp [NF.toReal, NF.roundR, NF.ofReal, Neg.neg]

omit [ValidRndToNearest rnd] in
/-- Runtime `exp` rounds `Real.exp` of the stored real value once. -/
theorem toReal_exp (x : R) :
    NF.toReal (Numerics.MathFunctions.exp x) =
      Flocq.round (β := β) (fexp := fexp) rnd
        (Real.exp (NF.toReal x)) := by
  simp [NF.toReal, NF.roundR, NF.ofReal, Numerics.MathFunctions.exp]

omit [ValidRndToNearest rnd] in
/-- Runtime `tanh` rounds `Real.tanh` of the stored real value once. -/
private theorem toReal_tanh (x : R) :
    NF.toReal (Numerics.MathFunctions.tanh x) =
      Flocq.round (β := β) (fexp := fexp) rnd
        (Real.tanh (NF.toReal x)) := by
  simp [NF.toReal, NF.roundR, NF.ofReal, Numerics.MathFunctions.tanh]

omit [ValidRndToNearest rnd] in
/-- Runtime `sqrt` rounds `Real.sqrt` of the stored real value once. -/
private theorem toReal_sqrt (x : R) :
    NF.toReal (Numerics.MathFunctions.sqrt x) =
      Flocq.round (β := β) (fexp := fexp) rnd
        (Real.sqrt (NF.toReal x)) := by
  simp [NF.toReal, NF.roundR, NF.ofReal, Numerics.MathFunctions.sqrt]

-- ---------------------------------------------------------------------------
-- Sqrt (clamped) approximation
-- ---------------------------------------------------------------------------

/-- The clamped square root `sqrt (max xR 0)` is within `sqrt eps` plus half an ULP of
`Real.sqrt (max x 0)`. No positive lower bound on the input is needed, so the bound holds at zero.
The `sqrt eps` term is the Hölder bound `|√a - √b| ≤ √|a - b|` for `a, b ≥ 0`.
-/
theorem approx_sqrt_clamp {x : ℝ} {xR : R} {eps : ℝ}
    (hx : |NF.toReal xR - x| ≤ eps) :
    |NF.toReal (Numerics.MathFunctions.sqrt (max xR 0)) -
        Real.sqrt (max x 0)| ≤
      Real.sqrt eps + ulp β fexp
        (Real.sqrt (max (NF.toReal xR) 0)) / 2 := by
  let xhat := NF.toReal xR
  have hmax : |max xhat 0 - max x 0| ≤ eps :=
    (abs_max_sub_max_le_abs xhat x 0).trans hx
  have hdiff : |Real.sqrt (max xhat 0) - Real.sqrt (max x 0)| ≤ Real.sqrt eps := by
    simpa only [zero_mul, zero_add] using
      ((MixedError.of_abs_sub_le hmax).sqrt (le_max_right x 0)).bound
  have hsqrt :
      NF.toReal (Numerics.MathFunctions.sqrt (max xR 0)) =
        Flocq.round (β := β) (fexp := fexp) rnd (Real.sqrt (max xhat 0)) := by
    have hxmax :
        NF.toReal (max xR (0 : R)) = max xhat 0 := by
      simpa [xhat] using Monotone.map_max (a := xR) (b := (0 : R))
        (show Monotone (NF.toReal) from fun _ _ h => h)
    simpa [hxmax] using toReal_sqrt (β := β) (fexp := fexp) (rnd := rnd) (max xR 0)
  have hround := Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd (Real.sqrt (max xhat 0))
  rw [hsqrt]
  exact (abs_sub_le _ (Real.sqrt (max xhat 0)) _).trans
    ((add_le_add hround hdiff).trans_eq (add_comm _ _))

/--
Forward approximation bound for `sqrt (max · 0)` under a positive lower bound `η ≤ max x 0`.

The clamp keeps the argument in the domain of `sqrt`. The strict bound `0 < η` controls the
conditioning through `|√a - √b| ≤ |a - b| / √η`, so the error is `eps / √η` plus half an ULP.
-/
theorem approx_sqrt_clamp_of_lb {x : ℝ} {xR : R} {eps η : ℝ}
    (hη : 0 < η) (hdom : η ≤ max x 0)
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs
        (NF.toReal (Numerics.MathFunctions.sqrt (max xR 0)) -
          Real.sqrt (max x 0)) ≤
      eps / Real.sqrt η +
        ulp β fexp
          (Real.sqrt (max (NF.toReal xR) 0)) / 2 := by
  set xhat : ℝ := NF.toReal xR
  have hxhat : abs (xhat - x) ≤ eps := by
    simpa [xhat, abs_sub_comm] using hx
  have hmax : abs (max xhat 0 - max x 0) ≤ eps :=
    (abs_max_sub_max_le_abs xhat x 0).trans hxhat
  have hround :
      abs
          (NF.toReal (Numerics.MathFunctions.sqrt (max xR 0)) -
            Real.sqrt (max xhat 0)) ≤
        ulp β fexp (Real.sqrt (max xhat 0)) / 2 := by
    -- `sqrt` on NF is a single rounding of the real `sqrt`.
    have :
        NF.toReal (Numerics.MathFunctions.sqrt (max xR 0)) =
          Flocq.round (β := β) (fexp := fexp) rnd (Real.sqrt (max xhat 0)) := by
      -- `NF.toReal` is monotone, so it commutes with `max`.
      have hxmax :
          NF.toReal (max xR (0 : R)) = max xhat 0 := by
        simpa [xhat] using Monotone.map_max (a := xR) (b := (0 : R))
          (show Monotone (NF.toReal) from fun _ _ h => h)
      simpa [xhat, hxmax] using
        (toReal_sqrt (β := β) (fexp := fexp) (rnd := rnd) (max xR 0))
    simpa [this] using
      Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd (Real.sqrt (max xhat 0))
  -- Conditioning of `sqrt` on `[η, ∞)`.
  have hdiff : abs (Real.sqrt (max xhat 0) - Real.sqrt (max x 0)) ≤ eps / Real.sqrt η := by
    simpa only [zero_mul, zero_add] using
      ((MixedError.of_abs_sub_le hmax).sqrt_of_le hη hdom).bound
  have :=
    calc
      abs
          (NF.toReal (Numerics.MathFunctions.sqrt (max xR 0)) -
            Real.sqrt (max x 0))
          ≤ abs
              (NF.toReal
                (Numerics.MathFunctions.sqrt (max xR 0)) -
                Real.sqrt (max xhat 0)) +
              abs (Real.sqrt (max xhat 0) - Real.sqrt (max x 0)) := abs_sub_le _ _ _
      _ ≤ ulp β fexp (Real.sqrt (max xhat 0)) / 2 + eps / Real.sqrt η := add_le_add hround hdiff
      _ = eps / Real.sqrt η + ulp β fexp (Real.sqrt (max xhat 0)) / 2 := add_comm _ _
  simpa [xhat, add_comm, add_left_comm, add_assoc] using this

/--
Forward approximation bound for addition in `NF`: if `xR` is within `epsx` of `x` and `yR` within
`epsy` of `y`, then `xR + yR` is within `epsx + epsy + ulp(NF.toReal xR + NF.toReal yR)/2` of
`x + y`.
-/
theorem approx_add {x y : ℝ} {xR yR : R} {epsx epsy : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ epsx)
    (hy : abs (NF.toReal yR - y) ≤ epsy) :
    abs (NF.toReal (xR + yR) - (x + y)) ≤
      epsx + epsy +
        ulp β fexp
            (NF.toReal xR +
              NF.toReal yR) / 2 := by
  rw [toReal_add (β := β) (fexp := fexp) (rnd := rnd) xR yR]
  exact FloatLib.Numerics.add_error hx hy
    (Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd _)

/--
Forward approximation bound for subtraction in `NF`.

The argument is the one behind `approx_add`: the exact difference of the runtime
operands is within `epsx + epsy` of `x - y`, and the runtime result is one rounding away from that
exact difference.
-/
theorem approx_sub {x y : ℝ} {xR yR : R} {epsx epsy : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ epsx)
    (hy : abs (NF.toReal yR - y) ≤ epsy) :
    abs (NF.toReal (xR - yR) - (x - y)) ≤
      epsx + epsy +
        ulp β fexp
            (NF.toReal xR -
              NF.toReal yR) / 2 := by
  have hdiff :
      abs ((NF.toReal xR -
          NF.toReal yR) - (x - y)) ≤ epsx + epsy := by
    rw [sub_sub_sub_comm]
    exact (abs_sub _ _).trans (add_le_add hx hy)
  rw [toReal_sub (β := β) (fexp := fexp) (rnd := rnd) xR yR]
  exact ((abs_sub_le _
      (NF.toReal xR -
        NF.toReal yR) _).trans
    (add_le_add
      (Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd _)
      hdiff)).trans_eq (by ring)

/-- Negation in `NF` keeps the input error `eps` and adds half an ULP of `-NF.toReal xR`. -/
theorem approx_neg {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs (NF.toReal (-xR) - (-x)) ≤
      eps +
        ulp β fexp
            (-NF.toReal xR) / 2 := by
  have hdiff : abs (-NF.toReal xR - (-x)) ≤ eps := by
    rw [neg_sub_neg, abs_sub_comm]
    exact hx
  rw [toReal_neg (β := β) (fexp := fexp) (rnd := rnd) xR]
  exact ((abs_sub_le _ (-NF.toReal xR) _).trans
    (add_le_add
      (Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd _)
      hdiff)).trans_eq (add_comm _ _)

/-- Absolute value in `NF` keeps the input error `eps`, because `abs` is `1`-Lipschitz, and the
final rounding adds half an ULP of `|NF.toReal xR|`. -/
theorem approx_abs {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs (NF.toReal (Numerics.MathFunctions.abs xR) - abs x) ≤
      eps +
        ulp β fexp
            (abs (NF.toReal xR)) / 2 := by
  let xhat : ℝ := NF.toReal xR
  have hround :
      abs (NF.toReal (Numerics.MathFunctions.abs xR) - abs xhat) ≤
        ulp β fexp (abs xhat) / 2 := by
    -- `NF.toReal (abs xR)` is a single rounding of `|xhat|`.
    simpa [xhat, Numerics.MathFunctions.abs, NF.instMathFunctions,
      NF.toReal, NF.roundR, NF.ofReal] using
      Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd (abs xhat)
  have hxhat : abs (xhat - x) ≤ eps := by
    simpa [xhat] using hx
  -- `abs` is `1`-Lipschitz, so the input error passes through unchanged.
  have hdiff : abs (abs xhat - abs x) ≤ eps :=
    (abs_abs_sub_abs_le_abs_sub xhat x).trans hxhat
  have :=
    calc
      abs (NF.toReal (Numerics.MathFunctions.abs xR) - abs x)
          ≤ abs (NF.toReal
              (Numerics.MathFunctions.abs xR) - abs xhat) +
              abs (abs xhat - abs x) := abs_sub_le _ _ _
      _ ≤ ulp β fexp (abs xhat) / 2 + eps := add_le_add hround hdiff
      _ = eps + ulp β fexp (abs xhat) / 2 := add_comm _ _
  simpa [xhat, add_assoc, add_left_comm, add_comm] using this

/-- Forward-error budget `exp (a + eps) * eps + ulp (exp a) / 2` for one rounded exponential.

The first term carries an input error `eps` through `exp` and vanishes for exact input, and the
second pays for the final rounding. Scalar activations, tensor lifting and stable axis softmax all
use this one budget.
-/
def expErrorBound (a eps : ℝ) : ℝ :=
  Real.exp (a + eps) * eps + ulp β fexp (Real.exp a) / 2

/--
Forward approximation bound for `exp` in `NF`, with budget `expErrorBound (NF.toReal xR) eps`.

The mean value inequality on `(-∞, NF.toReal xR + eps]`, where `exp` is at most
`exp (NF.toReal xR + eps)`, bounds the propagated input error. The final rounding adds half an ULP.
-/
theorem approx_exp {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs (NF.toReal (Numerics.MathFunctions.exp xR) - Real.exp x) ≤
      expErrorBound (β := β) (fexp := fexp)
        (NF.toReal xR) eps := by
  let xhat : ℝ := NF.toReal xR
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
      abs (Flocq.round (β := β) (fexp := fexp) rnd (Real.exp xhat) - Real.exp xhat) ≤
        ulp β fexp (Real.exp xhat) / 2 :=
    Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd (Real.exp xhat)

  have htotal :
      abs (Flocq.round (β := β) (fexp := fexp) rnd (Real.exp xhat) - Real.exp x) ≤
        Real.exp (xhat + eps) * eps + ulp β fexp (Real.exp xhat) / 2 :=
    calc
      abs (Flocq.round (β := β) (fexp := fexp) rnd (Real.exp xhat) - Real.exp x)
          ≤ abs (Flocq.round (β := β) (fexp := fexp) rnd (Real.exp xhat) - Real.exp xhat) +
              abs (Real.exp xhat - Real.exp x) := abs_sub_le _ _ _
      _ ≤ ulp β fexp (Real.exp xhat) / 2 + Real.exp (xhat + eps) * eps :=
          add_le_add hround hpropagation
      _ = Real.exp (xhat + eps) * eps + ulp β fexp (Real.exp xhat) / 2 := add_comm _ _

  simpa [expErrorBound, xhat, toReal_exp (β := β) (fexp := fexp) (rnd := rnd) xR, add_assoc,
    add_left_comm, add_comm]
    using htotal

/--
Forward bound for `tanh` in `NF`: if `|NF.toReal xR - x| ≤ eps`, then `NF.toReal (tanh xR)` is
within `min eps 2 + ulp β fexp (Real.tanh (NF.toReal xR)) / 2` of `Real.tanh x`.

The two values of `tanh` differ by at most `eps` because `tanh` is 1-Lipschitz, and by at most `2`
because both lie in `(-1, 1)`. The final rounding adds half an ULP of `tanh (NF.toReal xR)`.
-/
theorem approx_tanh {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs (NF.toReal
      (Numerics.MathFunctions.tanh xR) - Real.tanh x) ≤
      min eps 2 +
        ulp β fexp
            (Real.tanh (NF.toReal xR)) / 2 := by
  have hround := Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd
    (Real.tanh (NF.toReal xR))
  -- Two values of `tanh` lie in `(-1, 1)`, so they are never more than `2` apart.
  have hrange :
      abs (Real.tanh (NF.toReal xR) - Real.tanh x) ≤ 2 :=
    (abs_sub _ _).trans ((add_le_add (Real.abs_tanh_lt_one _).le
      (Real.abs_tanh_lt_one x).le).trans_eq one_add_one_eq_two)
  rw [toReal_tanh (β := β) (fexp := fexp) (rnd := rnd) xR, ← min_add_add_right]
  exact le_min (FloatLib.Numerics.tanh_error hx hround)
    (FloatLib.Numerics.compose_error hround hrange)

-- ---------------------------------------------------------------------------
-- Safe log: `log (max x ε)` (needed for unconditional forward bounds)
-- ---------------------------------------------------------------------------

/--
Clamped log on spec scalars: `log (max x ε)`.

For `0 < ε` the clamp keeps the argument away from the singularity at `0`, which gives forward
bounds for `log` with no domain hypothesis on `x`.
-/
def safeLog (ε : ℝ) (x : ℝ) : ℝ :=
  Real.log (max x ε)

/--
Clamped log in the rounded-real `NF` model: evaluate `safeLog` on the real value and round once.
The definition is noncomputable and does not implement a native logarithm.
-/
def safeLogR (ε : ℝ) (xR : R) : R :=
  NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (safeLog (ε := ε) (NF.toReal xR))

/--
Forward approximation bound for `safeLog` in `NF`, for `0 < ε`.

On `[ε, ∞)` the logarithm is `(1/ε)`-Lipschitz, which carries the input error over as
`(1/ε) * eps`, and the final rounding adds half an ULP.
-/
theorem approx_safeLog {x : ℝ} {xR : R} {eps ε : ℝ}
    (hε : 0 < ε)
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs (NF.toReal (safeLogR (β := β) (fexp := fexp) (rnd := rnd)
      ε xR) -
          safeLog (ε := ε) x) ≤
      (1 / ε) * eps +
        ulp β fexp
          (safeLog (ε := ε) (NF.toReal xR)) / 2 := by
  set xhat : ℝ := NF.toReal xR
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
    ((FloatLib.Numerics.abs_log_sub_log_le hε hyhat hy).trans
      (div_le_div_of_nonneg_right hmax hε.le)).trans_eq (one_div_mul_eq_div ε eps).symm

  have hround :
      abs
          (NF.toReal
              (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            Real.log yhat) ≤
        ulp β fexp (Real.log yhat) / 2 := by
    -- `safeLogR` rounds the real `log (max x̂ ε)`.
    have :
        NF.toReal
            (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) =
          Flocq.round (β := β) (fexp := fexp) rnd (Real.log yhat) := by
      simp [safeLogR, safeLog, xhat, yhat, NF.toReal, NF.roundR, NF.ofReal]
    simpa [this] using Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd (Real.log yhat)

  have :=
    calc
      abs
          (NF.toReal
              (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            safeLog (ε := ε) x)
          ≤ abs
              (NF.toReal
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

/-- Forward approximation bound for the ordinary square root when `0 < η ≤ x` and the runtime
input is nonnegative.

With `0 ≤ NF.toReal xR` the clamped bound reduces to the ordinary square root. The hypothesis `hxR`
states the domain of the theorem: `Real.sqrt` is total (zero on negative arguments), and no IEEE
invalid-operation check is involved.
-/
theorem approx_sqrt_of_pos_lb {x : ℝ} {xR : R} {eps η : ℝ}
    (hη : 0 < η) (hdom : η ≤ x)
    (hxR : 0 ≤ NF.toReal xR)
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs
        (NF.toReal (Numerics.MathFunctions.sqrt xR) -
          Real.sqrt x) ≤
      eps / Real.sqrt η +
        ulp β fexp
          (Real.sqrt (NF.toReal xR)) / 2 := by
  have hx0 : 0 ≤ x := hη.le.trans hdom
  -- The order on `NF` is the order of the underlying reals, and runtime `0` reads as real `0`.
  have hxR0 : (0 : R) ≤ xR := by
    change NF.toReal (0 : R) ≤
      NF.toReal xR
    rw [toReal_zero]
    exact hxR
  have hmaxR : max xR (0 : R) = xR := max_eq_left hxR0
  have hmaxS : max x 0 = x := max_eq_left hx0
  have h := approx_sqrt_clamp_of_lb (β := β) (fexp := fexp) (rnd := rnd)
    (x := x) (xR := xR) (eps := eps) (η := η) hη (by simpa [hmaxS] using hdom) hx
  simpa [hmaxR, hmaxS, hxR] using h

/-- Square-root approximation when the input error budget itself certifies runtime positivity.

From `η ≤ x`, `|x̂ - x| ≤ eps` and `eps < η`, the runtime input satisfies `0 < x̂`, so callers need
no separate runtime-domain hypothesis.
-/
theorem approx_sqrt_of_pos_lb_of_error {x : ℝ} {xR : R} {eps η : ℝ}
    (hη : 0 < η) (hdom : η ≤ x) (hbudget : eps < η)
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs
        (NF.toReal (Numerics.MathFunctions.sqrt xR) -
          Real.sqrt x) ≤
      eps / Real.sqrt η +
        ulp β fexp
          (Real.sqrt (NF.toReal xR)) / 2 := by
  have hxR : 0 ≤ NF.toReal xR := by
    have hdiff : x - NF.toReal xR ≤ eps := by
      calc
        x - NF.toReal xR ≤
            abs (x - NF.toReal xR) := le_abs_self _
        _ = abs (NF.toReal xR - x) := abs_sub_comm _ _
        _ ≤ eps := hx
    linarith
  exact approx_sqrt_of_pos_lb (β := β) (fexp := fexp) (rnd := rnd)
    hη hdom hxR hx

/--
Forward approximation bound for multiplication in `NF`.

For `x̂ = NF.toReal xR` and `ŷ = NF.toReal yR`, the budget is
`(|x̂| + epsx) * epsy + (|ŷ| + epsy) * epsx` for the input perturbations and their cross term, plus
half an ULP of `x̂ * ŷ` for the rounding.
-/
theorem approx_mul {x y : ℝ} {xR yR : R} {epsx epsy : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ epsx)
    (hy : abs (NF.toReal yR - y) ≤ epsy) :
    abs (NF.toReal (xR * yR) - (x * y)) ≤
      ((abs (NF.toReal xR) + epsx) * epsy +
        (abs (NF.toReal yR) + epsy) * epsx +
        ulp β fexp
            (NF.toReal xR *
              NF.toReal yR) / 2) := by
  have hx' :
      ScalarApprox x
        (NF.toReal xR) epsx := by
    simpa [ScalarApprox] using hx
  have hy' :
      ScalarApprox y
        (NF.toReal yR) epsy := by
    simpa [ScalarApprox] using hy
  have h := roundedMul (β := β) (fexp := fexp) (rnd := rnd) hx' hy'
  simpa [ScalarApprox,
    toReal_mul (β := β) (fexp := fexp) (rnd := rnd) xR yR] using h

/-- Multiplying by a runtime constant `c` scales the input error by `|NF.toReal c|` and adds half
an ULP of the product `NF.toReal xR * NF.toReal c`. -/
theorem approx_scale {x : ℝ} {xR : R} {eps : ℝ} (c : R)
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs (NF.toReal (xR * c) - (x * NF.toReal c)) ≤
      abs (NF.toReal c) * eps +
        ulp β fexp
            (NF.toReal xR *
              NF.toReal c) / 2 := by
  have hc : abs (NF.toReal c -
        NF.toReal c) ≤ (0 : ℝ) := by
    simp
  have h :=
    approx_mul (β := β) (fexp := fexp) (rnd := rnd)
      (x := x) (y := NF.toReal c)
      (xR := xR) (yR := c) (epsx := eps) (epsy := (0 : ℝ)) hx hc
  simpa [mul_assoc, add_assoc, add_left_comm, add_comm] using h

end

end NF
end TorchLean
