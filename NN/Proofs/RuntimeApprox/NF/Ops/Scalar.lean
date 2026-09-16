/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
public import FloatLib.Floats.Formats.Flocq.Theory.Scalar.NF
public import NN.Proofs.RuntimeApprox.Rounding.RoundingApprox
public import NN.Spec.Core.Context.Real
public import NN.Spec.Core.FloatInstances.NF
public import NN.Spec.Core.Scalar
public import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Order -- shake: keep
public import NN.Proofs.RuntimeApprox.NF.Ops.Plumbing -- shake: keep

/-!
# NF Scalar Primitive Bounds

Scalar bridge lemmas and forward-error bounds for rounded `NF` primitives.  These are the facts
that later tensor proofs lift pointwise across shapes.
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
      ulp β fexp (q : ℝ) / 2 := by
  exact approx_ofReal_nf (β := β) (fexp := fexp) (rnd := rnd) (q : ℝ)

/-!
## Bridge lemmas from `NF` to $\mathbb R$

Most approximation statements in this file are phrased over the spec scalar `ℝ`, but the runtime
backend is `NF β fexp rnd`. The following lemmas are small bridge facts that let us rewrite
runtime expressions into:

- an exact real expression in terms of `toSpec`, plus
- an explicit rounding operator `roundR` applied at the outermost step.

Keeping these as named lemmas (instead of repeating huge `simp [...]` lists) makes the later
forward-approx proofs much easier to read.
-/

/-- Rounding `0` is `0` for any valid rounding mode. -/
private theorem roundR_zero :
    Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (0 : ℝ) = 0 := by
  -- For `x = 0`, the scaled mantissa is `0`, so `rnd` returns `0` by `ValidRnd.id`,
  -- and `Flocq.toReal` is `0` regardless of exponent.
  have hrnd0 : rnd (0 : ℝ) = 0 := by
    -- `rnd` is exact on integers.
    simpa using (ValidRnd.id (rnd := rnd) (n := (0 : ℤ)))
  simp [Proofs.RuntimeRoundingApprox.roundR, Flocq.round,
    Flocq.scaledMantissa, Flocq.toReal, hrnd0]

/-- The `NF.roundR` wrapper also rounds `0` to `0`. -/
private theorem NF_roundR_zero :
    NF.roundR (β := β) (fexp := fexp) (rnd := rnd) (0 : ℝ) = 0 := by
  have hrnd0 : rnd (0 : ℝ) = 0 := by
    simpa using (ValidRnd.id (rnd := rnd) (n := (0 : ℤ)))
  simp [NF.roundR, Flocq.round,
    Flocq.scaledMantissa, Flocq.toReal, hrnd0]

/-- `toSpec` of runtime `0` is the spec scalar `0`. -/
@[simp] theorem toSpec_zero : toSpec (β := β) (fexp := fexp) (rnd := rnd) (0 : R) = (0 : ℝ) := by
  -- `0 : R` is `NF.ofReal 0`, so `toSpec 0` is `NF.roundR 0`.
  change (NF.ofReal (β := β) (fexp := fexp) (rnd := rnd) (0 : ℝ)).val = (0 : ℝ)
  simpa [NF.ofReal] using
    (NF_roundR_zero (β := β) (fexp := fexp) (rnd := rnd))

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
/-- `toSpec` respects runtime division, up to an explicit rounding step. -/
private theorem toSpec_div (x y : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (x / y) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) x /
          toSpec (β := β) (fexp := fexp) (rnd := rnd) y) := by
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
  simp [toSpec, NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
    NF.roundR, NF.ofReal, Numerics.MathFunctions.exp,
    ]

omit [ValidRndToNearest rnd] in
/-- `toSpec` respects runtime `tanh`, up to an explicit rounding step. -/
private theorem toSpec_tanh (x : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.tanh x) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (Real.tanh (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)) := by
  simp [toSpec, NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
    NF.roundR, NF.ofReal, Numerics.MathFunctions.tanh,
    ]

omit [ValidRndToNearest rnd] in
/-- `toSpec` respects runtime `sqrt`, up to an explicit rounding step. -/
private theorem toSpec_sqrt (x : R) :
    toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.sqrt x) =
      Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
        (Real.sqrt (toSpec (β := β) (fexp := fexp) (rnd := rnd) x)) := by
  simp [toSpec, NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
    NF.roundR, NF.ofReal, Numerics.MathFunctions.sqrt,
    ]

-- ---------------------------------------------------------------------------
-- Sqrt (clamped) approximation
-- ---------------------------------------------------------------------------

private theorem abs_sqrt_sub_sqrt_le_div_sqrt_of_le {a b η : ℝ} (ha : 0 ≤ a) (hη : 0 < η)
    (hb : η ≤ b) :
    abs (Real.sqrt a - Real.sqrt b) ≤ abs (a - b) / Real.sqrt η := by
  have hb0 : 0 < b := lt_of_lt_of_le hη hb
  have hsb_pos : 0 < Real.sqrt b := Real.sqrt_pos.2 hb0
  have hsa_nonneg : 0 ≤ Real.sqrt a := Real.sqrt_nonneg a
  have hden_pos : 0 < Real.sqrt a + Real.sqrt b := add_pos_of_nonneg_of_pos hsa_nonneg hsb_pos
  have hden_ne : Real.sqrt a + Real.sqrt b ≠ 0 := ne_of_gt hden_pos
  have hprod :
      (Real.sqrt a - Real.sqrt b) * (Real.sqrt a + Real.sqrt b) = a - b := by
    -- `(√a - √b) * (√a + √b) = (√a)^2 - (√b)^2 = a - b`
    have ha' : Real.sqrt a ^ 2 = a := Real.sq_sqrt ha
    have hb' : Real.sqrt b ^ 2 = b := Real.sq_sqrt (le_of_lt hb0)
    ring_nf
    simp [ha', hb']
  have hdiv : Real.sqrt a - Real.sqrt b = (a - b) / (Real.sqrt a + Real.sqrt b) :=
    (eq_div_iff hden_ne).2 hprod
  have hden_ge : Real.sqrt η ≤ Real.sqrt a + Real.sqrt b := by
    have hη0 : 0 ≤ η := le_of_lt hη
    have hsqrt : Real.sqrt η ≤ Real.sqrt b := Real.sqrt_le_sqrt hb
    have : Real.sqrt b ≤ Real.sqrt a + Real.sqrt b := by
      simp
    exact le_trans hsqrt this
  have hquot :=
    div_le_div_of_nonneg_left (abs_nonneg (a - b)) (Real.sqrt_pos.2 hη) hden_ge
  calc
    abs (Real.sqrt a - Real.sqrt b)
        = abs ((a - b) / (Real.sqrt a + Real.sqrt b)) := by simp [hdiv]
    _ = abs (a - b) / abs (Real.sqrt a + Real.sqrt b) := by simp [abs_div]
    _ = abs (a - b) / (Real.sqrt a + Real.sqrt b) := by
          simp [abs_of_pos hden_pos]
    _ ≤ abs (a - b) / Real.sqrt η := hquot

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
  have hmax : abs (max xhat 0 - max x 0) ≤ eps := by
    have h1 : abs (max xhat 0 - max x 0) ≤ abs (xhat - x) := by
      simpa using (abs_max_sub_max_le_abs xhat x (0 : ℝ))
    exact le_trans h1 hxhat
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
    -- Now apply the generic rounding error lemma.
    simpa [this, Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (Real.sqrt
        (max xhat 0)))

  have hdiff : abs (Real.sqrt (max xhat 0) - Real.sqrt (max x 0)) ≤ eps / Real.sqrt η := by
    have ha : 0 ≤ max xhat 0 := le_max_right _ _
    exact le_trans
      (abs_sqrt_sub_sqrt_le_div_sqrt_of_le (a := max xhat 0) (b := max x 0) (η := η) ha hη hdom)
      (by
        -- monotonicity in the numerator
        have : abs (max xhat 0 - max x 0) / Real.sqrt η ≤ eps / Real.sqrt η := by
          exact div_le_div_of_nonneg_right hmax (Real.sqrt_nonneg η)
        simpa [abs_sub_comm] using this)

  have :=
    calc
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.sqrt (max xR 0)) -
            Real.sqrt (max x 0))
          ≤ abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (Numerics.MathFunctions.sqrt (max xR 0)) -
                Real.sqrt (max xhat 0)) +
              abs (Real.sqrt (max xhat 0) - Real.sqrt (max x 0)) := by
                simpa [sub_eq_add_neg, add_assoc] using
                  abs_sub_le
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                      (Numerics.MathFunctions.sqrt (max xR 0)))
                    (Real.sqrt (max xhat 0))
                    (Real.sqrt (max x 0))
      _ ≤ ulp β fexp (Real.sqrt (max xhat 0)) / 2 + eps / Real.sqrt η
        := by
            exact add_le_add hround hdiff
      _ = eps / Real.sqrt η +
            ulp β fexp (Real.sqrt (max xhat 0)) / 2 := by
            ring
  simpa [xhat, add_comm, add_left_comm, add_assoc] using this

private theorem abs_tanh_le_one (x : ℝ) : abs (Real.tanh x) ≤ 1 := by
  -- `tanh x = (exp x - exp (-x)) / (exp x + exp (-x))`, so `|tanh x| ≤ 1` by `|a-b| ≤ a+b`.
  have htanh :
      Real.tanh x =
        (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x)) := by
    -- Reduce to `sinh/cosh` and then to the `exp` definitions.
    rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
    -- Cancel the common `/2`.
    field_simp [two_ne_zero]

  have hden_pos : 0 < Real.exp x + Real.exp (-x) :=
    add_pos (Real.exp_pos x) (Real.exp_pos (-x))
  have hden_ne : Real.exp x + Real.exp (-x) ≠ 0 := ne_of_gt hden_pos

  have hnum :
      abs (Real.exp x - Real.exp (-x)) ≤ Real.exp x + Real.exp (-x) := by
    -- `|a-b| ≤ |a| + |b|` and `exp` is nonnegative.
    have := abs_add_le (Real.exp x) (-Real.exp (-x))
    -- `abs (a + (-b)) ≤ abs a + abs (-b)`.
    simpa [sub_eq_add_neg, abs_neg, abs_of_nonneg (Real.exp_nonneg _),
      abs_of_nonneg (Real.exp_nonneg _)] using this

  -- Divide by the positive denominator.
  have hdiv :=
    div_le_div_of_nonneg_right hnum (le_of_lt hden_pos)
  -- `|(a-b)/(a+b)| ≤ (a+b)/(a+b) = 1`.
  simpa [htanh, abs_div, abs_of_pos hden_pos, div_self hden_ne] using hdiv

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
  -- Rewrite the runtime result as `toSpec (xR + yR)`.
  simpa [Proofs.RuntimeRoundingApprox.scalarApprox,
    toSpec_add (β := β) (fexp := fexp) (rnd := rnd) xR yR] using h

/--
Forward approximation bound for subtraction in `NF`.

This is proved by reducing subtraction to addition with a negation and applying `approx_add_nf`.
-/
theorem approx_sub_nf {x y : ℝ} {xR yR : R} {epsx epsy : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR - yR) - (x - y)) ≤
      epsx + epsy +
        ulp β fexp
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR -
              toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) / 2 := by
  let xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  let yhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) yR
  have hround :
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (xhat - yhat) -
            (xhat - yhat)) ≤
        ulp β fexp (xhat - yhat) / 2 := by
    simpa [Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (xhat -
        yhat))
  have hdiff :
      abs ((xhat - yhat) - (x - y)) ≤ epsx + epsy := by
    have hrewrite : (xhat - yhat) - (x - y) = (xhat - x) - (yhat - y) := by ring
    have htri : abs ((xhat - x) - (yhat - y)) ≤ abs (xhat - x) + abs (yhat - y) := by
      simpa [abs_sub_comm] using (abs_sub_le (xhat - x) 0 (yhat - y))
    have hsum : abs (xhat - x) + abs (yhat - y) ≤ epsx + epsy := add_le_add hx hy
    exact le_trans (by simpa [hrewrite] using htri) hsum
  have :=
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR - yR) - (x - y))
          =
          abs
            (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (xhat - yhat)
              -
              (x - y)) := by
              simp [xhat, yhat, toSpec_sub (β := β) (fexp := fexp) (rnd := rnd) xR yR]
      _ ≤
          abs
              (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (xhat -
                yhat) -
                (xhat - yhat)) +
            abs ((xhat - yhat) - (x - y)) := by
              simpa [sub_eq_add_neg, add_assoc] using
                abs_sub_le
                  (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (xhat -
                    yhat))
                  (xhat - yhat) (x - y)
      _ ≤ ulp β fexp (xhat - yhat) / 2 + (epsx + epsy) := by
            exact add_le_add hround hdiff
      _ = epsx + epsy + ulp β fexp (xhat - yhat) / 2 := by ring
  simpa [xhat, yhat, sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using this

/-- Forward approximation bound for negation in `NF` (rounding error on `-toSpec xR`). -/
theorem approx_neg_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR) - (-x)) ≤
      eps +
        ulp β fexp
            (-toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) / 2 := by
  let xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  have hround :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR) - (-xhat)) ≤
        ulp β fexp (-xhat) / 2 := by
    -- `toSpec (-xR)` is a single rounding of `-xhat`.
    simpa [xhat, toSpec_neg (β := β) (fexp := fexp) (rnd := rnd) xR,
      Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (-xhat))
  have hdiff : abs (-xhat - (-x)) ≤ eps := by
    have hxhat : abs (xhat - x) ≤ eps := by
      simpa [xhat] using hx
    have hx' : abs (-xhat + x) ≤ eps := by
      have habs : abs (-xhat + x) = abs (xhat - x) := by
        calc
          abs (-xhat + x) = abs (x - xhat) := by
            simp [sub_eq_add_neg, add_comm]
          _ = abs (xhat - x) := by
            simp [abs_sub_comm]
      simpa [habs] using hxhat
    -- `-xhat - (-x)` is definitional `-xhat + x`.
    simpa [sub_eq_add_neg] using hx'
  have :=
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR) - (-x))
          ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR) - (-xhat)) + abs (-xhat - (-x))
            := by
                simpa [sub_eq_add_neg, add_assoc] using
                  abs_sub_le
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR))
                    (-xhat) (-x)
      _ ≤ ulp β fexp (-xhat) / 2 + eps := by
            exact add_le_add hround hdiff
      _ = eps + ulp β fexp (-xhat) / 2 := by ring
  simpa [xhat, add_assoc, add_left_comm, add_comm] using this

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
  have habs : abs (abs xhat - abs x) ≤ abs (xhat - x) := by
    simpa [abs_sub_comm] using (abs_abs_sub_abs_le_abs_sub xhat x)
  have hxhat : abs (xhat - x) ≤ eps := by
    simpa [xhat, abs_sub_comm] using hx
  have :=
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.abs xR) - abs x)
          ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (Numerics.MathFunctions.abs xR) - abs xhat) +
              abs (abs xhat - abs x) := by
                simpa [sub_eq_add_neg, add_assoc] using
                  abs_sub_le
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.abs xR))
                    (abs xhat) (abs x)
      _ ≤ ulp β fexp (abs xhat) / 2 + abs (xhat - x) := by
            exact add_le_add hround habs
      _ ≤ ulp β fexp (abs xhat) / 2 + eps := by
            linarith [hxhat]
      _ = eps + ulp β fexp (abs xhat) / 2 := by ring
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

  have heps : 0 ≤ eps :=
    le_trans (abs_nonneg (xhat - x)) (by simpa [xhat] using hx)

  have hx_le : x ≤ xhat + eps := by
    have h : x - xhat ≤ eps := by
      calc
        x - xhat ≤ abs (x - xhat) := le_abs_self _
        _ = abs (xhat - x) := abs_sub_comm _ _
        _ ≤ eps := by simpa [xhat] using hx
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

  have hinput : abs (xhat - x) ≤ eps := by
    simpa [xhat] using hx

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
        Real.exp (xhat + eps) * eps + ulp β fexp (Real.exp xhat) / 2 := by
    have :=
      calc
        abs
            (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp
              xhat) -
              Real.exp x)
            ≤ abs
                (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp
                  xhat) -
                  Real.exp xhat) +
                abs (Real.exp xhat - Real.exp x) := by
                  simpa [sub_eq_add_neg, add_assoc] using
                    abs_sub_le
                      (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
                        (Real.exp xhat))
                      (Real.exp xhat) (Real.exp x)
        _ ≤ ulp β fexp (Real.exp xhat) / 2 +
              Real.exp (xhat + eps) * eps :=
              add_le_add hround hpropagation
        _ = Real.exp (xhat + eps) * eps +
              ulp β fexp (Real.exp xhat) / 2 := by ring
    exact this

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

  have hdiff : abs (Real.tanh xhat - Real.tanh x) ≤ 2 := by
    have h' : abs (Real.tanh xhat - Real.tanh x) ≤ abs (Real.tanh xhat) + abs (Real.tanh x) := by
      have := abs_add_le (Real.tanh xhat) (-Real.tanh x)
      simpa [sub_eq_add_neg, abs_neg] using this
    have hxhat_le : abs (Real.tanh xhat) ≤ 1 := abs_tanh_le_one xhat
    have hx_le : abs (Real.tanh x) ≤ 1 := abs_tanh_le_one x
    have : abs (Real.tanh xhat) + abs (Real.tanh x) ≤ (1 : ℝ) + 1 := add_le_add hxhat_le hx_le
    have := le_trans h' this
    simpa [one_add_one_eq_two] using this

  have htotal :
      abs
          (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh xhat)
            -
            Real.tanh x) ≤
        2 + ulp β fexp (Real.tanh xhat) / 2 := by
    have :=
      calc
        abs
            (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh
              xhat) -
              Real.tanh x)
            ≤ abs
                (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh
                  xhat) -
                  Real.tanh xhat) +
                abs (Real.tanh xhat - Real.tanh x) := by
                  simpa [sub_eq_add_neg, add_assoc] using
                    abs_sub_le
                      (Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd)
                        (Real.tanh xhat))
                      (Real.tanh xhat) (Real.tanh x)
        _ ≤ ulp β fexp (Real.tanh xhat) / 2 + 2 := by
              exact add_le_add hround hdiff
        _ = 2 + ulp β fexp (Real.tanh xhat) / 2 := by ring
    exact this

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
Clamped log on runtime `NF` scalars (implemented as `NF.ofReal (safeLog (toSpec xR))`).

This definition keeps the semantic spec function explicit (so proofs can reason about it) while
still producing an executable runtime scalar.
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
    have hxinv : ‖x⁻¹‖ = (1 : ℝ) / x := by
      have hxinvpos : 0 < x⁻¹ := inv_pos.2 hxpos
      calc
        ‖x⁻¹‖ = |x⁻¹| := Real.norm_eq_abs (r := x⁻¹)
        _ = x⁻¹ := abs_of_pos hxinvpos
        _ = (1 : ℝ) / x := (one_div x).symm
    have hle : (1 : ℝ) / x ≤ (1 : ℝ) / ε := by
      simpa using (one_div_le_one_div_of_le hε hx)
    simpa [hxinv] using hle

  have hmv :=
    Convex.norm_image_sub_le_of_norm_hasDerivWithin_le (f := Real.log) (f' := fun x : ℝ => x⁻¹)
      (s := Set.Ici ε) (x := u) (y := v) (C := (1 / ε))
      hf hbound (convex_Ici ε) hu hv

  -- Unwrap norms on `ℝ`.
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
    have hmax' : abs (max xhat ε - max x ε) ≤ abs (xhat - x) := by
      simpa using (abs_max_sub_max_le_abs xhat x ε)
    have hx' : abs (xhat - x) ≤ eps := by
      simpa [xhat, abs_sub_comm] using hx
    -- Rewrite `yhat,y` and chain.
    simpa [yhat, y] using le_trans hmax' hx'

  have hdiff :
      abs (Real.log yhat - Real.log y) ≤ (1 / ε) * eps := by
    have hlog :
        abs (Real.log yhat - Real.log y) ≤ (1 / ε) * abs (yhat - y) := by
      simpa [one_div] using (abs_log_sub_log_le_one_div_mul_abs_sub (ε := ε) (u := yhat) (v := y) hε
        hyhat hy)
    have hεinv_nonneg : 0 ≤ (1 / ε) := by
      exact one_div_nonneg.2 (le_of_lt hε)
    exact le_trans hlog (mul_le_mul_of_nonneg_left hmax hεinv_nonneg)

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
              abs (Real.log yhat - safeLog (ε := ε) x) := by
                simpa [safeLog, y, sub_eq_add_neg, add_assoc] using
                  abs_sub_le
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                      (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR))
                    (Real.log yhat)
                    (safeLog (ε := ε) x)
      _ ≤ ulp β fexp (Real.log yhat) / 2 + (1 / ε) * eps := by
            -- second term is the `log` perturbation
            have : abs (Real.log yhat - safeLog (ε := ε) x) = abs (Real.log yhat - Real.log y) := by
              simp [safeLog, y]
            simpa [this, add_comm, add_left_comm, add_assoc] using add_le_add hround hdiff
      _ = (1 / ε) * eps + ulp β fexp (safeLog (ε := ε) xhat) / 2 := by
            simp [safeLog, xhat, yhat, add_comm]
  simpa [xhat] using this

/-- Forward approximation bound for ordinary square root on a certified positive domain.

The extra runtime hypothesis is not cosmetic: the executable `NF.sqrt` receives the rounded input,
so a real lower bound alone does not rule out an invalid rounded argument. Once both exact and
runtime inputs are nonnegative, the clamped theorem above reduces definitionally to ordinary
square root.
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
  have hx0 : 0 ≤ x := le_trans (le_of_lt hη) hdom
  have hzeroR : toSpec (β := β) (fexp := fexp) (rnd := rnd) (0 : R) = 0 :=
    toSpec_zero (β := β) (fexp := fexp) (rnd := rnd)
  have hxR0 : (0 : R) ≤ xR := by
    change (0 : R).val ≤ xR.val
    have hzeroVal : (0 : R).val = 0 := by
      simpa [toSpec, NF.toReal] using hzeroR
    rw [hzeroVal]
    exact hxR
  have hmaxR : max xR (0 : R) = xR := by
    exact max_eq_left hxR0
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

This has the standard "first-order" shape:
terms proportional to `|toSpec xR| * epsy` and `|toSpec yR| * epsx`, plus an `ulp` term for the
  final
rounding. (For classical background, see Higham, *Accuracy and Stability of Numerical Algorithms*.)
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
  -- Simplify away the `* 0` and `+ 0` terms.
  simpa [mul_assoc, add_assoc, add_left_comm, add_comm] using h


end NFBackend

end

end RuntimeApprox
end Proofs
