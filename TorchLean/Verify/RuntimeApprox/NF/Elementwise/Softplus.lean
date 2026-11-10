/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Gradients.Activation
public import TorchLean.Verify.RuntimeApprox.NF.Scalar

/-!
# NF Elementwise Bounds: Softplus and Safe Log

`softplusR` rounds the constant `1`, the exponential, the addition and the clamped logarithm
separately, while `safeLogSoftplusR` rounds the exact smooth activation `log (softplus x + ε)` once.
The two are rounded-real models with different rounding schedules, and neither executes a native
kernel. Both bounds hold at every input, the second for `0 < ε`, because the clamped logarithm and
the smooth safe log are Lipschitz on all of `ℝ`.
-/

@[expose] public section

namespace TorchLean
namespace NF

open TorchLean.Tensor
open Robustness

noncomputable section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-- Half-ULP rounding budget at the scalar value `1`, covering the rounded constant `1` in the
softplus, sigmoid and logistic bounds. -/
def oneEps : ℝ :=
  ulp β fexp (1 : ℝ) / 2

/-- `oneEps` is a half ulp, hence nonnegative. -/
theorem oneEps_nonneg : 0 ≤ oneEps (β := β) (fexp := fexp) := by
  unfold oneEps
  have := ulp.nonneg β fexp (1 : ℝ)
  linarith

/-- The rounded constant `1 : NF` is within `oneEps` of the real `1`. -/
theorem abs_toReal_one_sub_one_le :
    abs (NF.toReal (1 : R) - (1 : ℝ)) ≤
      oneEps (β := β) (fexp := fexp) := by
  change
    abs ((NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
      (1 : ℝ)).val - (1 : ℝ)) ≤ oneEps (β := β) (fexp := fexp)
  simpa [oneEps, NF.toReal, NF.roundR, NF.ofReal] using
    (Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd (1 : ℝ))

/-- Rounded representation of the scalar constant `1` at the `NF` backend. -/
def oneHat : ℝ :=
  Flocq.round (β := β) (fexp := fexp) rnd (1 : ℝ)

/-- Rounded representation of `exp a` at the `NF` backend. -/
def expHat (a : ℝ) : ℝ :=
  Flocq.round (β := β) (fexp := fexp) rnd (Real.exp a)

/-- Rounded surrogate for the inner `1 + exp a` term appearing in `softplus`. -/
def addHatSoftplus (a : ℝ) : ℝ :=
  Flocq.round (β := β) (fexp := fexp) rnd
    (oneHat (β := β) (fexp := fexp) (rnd := rnd) + expHat (β := β) (fexp := fexp) (rnd := rnd) a)

/-- Forward-error envelope for the rounded `1 + exp a` subexpression used by `softplus`. -/
def addBoundSoftplus (a eps : ℝ) : ℝ :=
  oneEps (β := β) (fexp := fexp) + expErrorBound (β := β) (fexp := fexp) a eps +
    ulp β fexp
        (oneHat (β := β) (fexp := fexp) (rnd := rnd) + expHat (β := β) (fexp := fexp) (rnd := rnd)
          a) / 2

/-- Scalar forward-error budget for `softplus` at every input: the budget `addBoundSoftplus` of
`1 + exp a`, which `safeLog 1` passes through unscaled, plus half an ULP for the rounded logarithm.
-/
def softplusBoundScalar (a eps : ℝ) : ℝ :=
  addBoundSoftplus (β := β) (fexp := fexp) (rnd := rnd) a eps +
    ulp β fexp
        (safeLog (ε := (1 : ℝ)) (addHatSoftplus (β := β) (fexp := fexp) (rnd := rnd) a)) / 2

/-- `softplus` implemented by `safeLog 1 (1 + exp x)` at the `NF` backend. -/
def softplusR (xR : R) : R :=
  let yR : R := (1 : R) + Numerics.MathFunctions.exp xR
  safeLogR (β := β) (fexp := fexp) (rnd := rnd) (ε := (1 : ℝ)) yR

/--
Forward approximation bound for `softplus` in `NF`.

Since `1 + exp x ≥ 1`, `softplus(x) = log(1 + exp x)` equals `safeLog 1 (1 + exp x)`, and the bound
composes the scalar bounds for `exp`, `+` and `safeLog`.
-/
theorem approx_softplus {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs (NF.toReal
          (softplusR (β := β) (fexp := fexp) (rnd := rnd) xR) -
        Activation.softplus (α := ℝ) x) ≤
      softplusBoundScalar (β := β) (fexp := fexp) (rnd := rnd)
        (NF.toReal xR) eps := by
  have hexp :
      abs (NF.toReal
        (Numerics.MathFunctions.exp xR) - Real.exp x) ≤
        expErrorBound (β := β) (fexp := fexp) (NF.toReal xR)
          eps :=
    approx_exp (β := β) (fexp := fexp) (rnd := rnd) hx

  let oneR : R := (1 : R)
  have hOneVal :
      oneR.val = oneHat (β := β) (fexp := fexp) (rnd := rnd) := by
    change
      (NF.ofReal (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ)).val =
        oneHat (β := β) (fexp := fexp) (rnd := rnd)
    rfl
  have hone :
      abs (NF.toReal oneR - (1 : ℝ)) ≤
        oneEps (β := β) (fexp := fexp) :=
    abs_toReal_one_sub_one_le (β := β) (fexp := fexp) (rnd := rnd)

  let yR : R := oneR + Numerics.MathFunctions.exp xR
  let y : ℝ := (1 : ℝ) + Real.exp x
  have hExpVal :
      (Numerics.MathFunctions.exp xR).val =
        expHat (β := β) (fexp := fexp) (rnd := rnd)
          (NF.toReal xR) := by
    simp [expHat, NF.toReal, NF.roundR,
      NF.ofReal, Numerics.MathFunctions.exp]
  have hYVal :
      yR.val =
        addHatSoftplus (β := β) (fexp := fexp) (rnd := rnd)
          (NF.toReal xR) := by
    change
      (NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
        (oneR.val + (Numerics.MathFunctions.exp xR).val)).val =
        addHatSoftplus (β := β) (fexp := fexp) (rnd := rnd)
          (NF.toReal xR)
    rw [hOneVal, hExpVal]
    rfl

  have hy :
      abs (NF.toReal yR - y) ≤
        addBoundSoftplus (β := β) (fexp := fexp) (rnd := rnd)
          (NF.toReal xR) eps := by
    have hadd :=
      approx_add (β := β) (fexp := fexp) (rnd := rnd)
        (x := (1 : ℝ)) (y := Real.exp x)
        (xR := oneR) (yR := Numerics.MathFunctions.exp xR)
        (epsx := oneEps (β := β) (fexp := fexp))
        (epsy := expErrorBound (β := β) (fexp := fexp)
          (NF.toReal xR) eps)
        hone hexp
    simpa [oneR, yR, y, addBoundSoftplus, expErrorBound, oneHat, expHat,
      hOneVal, hExpVal,
      NF.toReal,
      NF.roundR, NF.ofReal, NF.instOne,
      NF.instAdd, One.one, Numerics.MathFunctions.exp,
      toReal_exp (β := β) (fexp := fexp) (rnd := rnd) xR] using hadd

  -- The `safeLog 1` bound carries the error of `1 + exp x` through the logarithm.
  have hlog :=
    approx_safeLog (β := β) (fexp := fexp) (rnd := rnd)
      (x := y) (xR := yR)
      (eps := addBoundSoftplus (β := β) (fexp := fexp) (rnd := rnd)
        (NF.toReal xR) eps)
      (ε := (1 : ℝ)) (hε := by norm_num) hy
  have hlog' :
      abs (NF.toReal
        (safeLogR (β := β) (fexp := fexp) (rnd := rnd) (ε := (1 : ℝ)) yR) -
          safeLog (ε := (1 : ℝ)) y) ≤
        addBoundSoftplus (β := β) (fexp := fexp) (rnd := rnd)
          (NF.toReal xR) eps +
          ulp β fexp
            (safeLog (ε := (1 : ℝ))
              (addHatSoftplus (β := β) (fexp := fexp) (rnd := rnd)
                (NF.toReal xR))) / 2 := by
    simpa [NF.toReal, hYVal] using hlog

  have hy_ge : (1 : ℝ) ≤ y := by
    have : 0 ≤ Real.exp x := by simpa using (Real.exp_nonneg x)
    linarith

  have hsimp : safeLog (ε := (1 : ℝ)) y = Activation.softplus (α := ℝ) x := by
    calc
      safeLog (ε := (1 : ℝ)) y = Real.log y := by
        simp [NF.safeLog, hy_ge]
      _ = Real.log ((1 : ℝ) + Real.exp x) := by
        simp [y]
      _ = Activation.softplus (α := ℝ) x :=
        (Activation.softplus_eq_log_one_add_exp x).symm

  simpa [oneR, softplusR, yR, hsimp, softplusBoundScalar, addHatSoftplus,
    hYVal,
    NF.toReal,
    NF.roundR, NF.ofReal, NF.instOne,
    NF.instAdd, One.one, Numerics.MathFunctions.exp,
    toReal_exp (β := β) (fexp := fexp) (rnd := rnd) xR] using hlog'

/--
Per-entry bound tensor for `softplusR`.

Each entry is `softplusBoundScalar` at the entry's real value, and the `linfNorm` of the
tensor is the output epsilon of `ApproxTensor.softplus`.
-/
def softplusBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map (fun a => softplusBoundScalar (β := β) (fexp := fexp) (rnd := rnd) a eps)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

/--
`ApproxTensor` bound for `softplus` lifted to arbitrary tensor shapes.

The scalar bound `approx_softplus` lifts through `ApproxTensor.map`.
-/
theorem _root_.TorchLean.ApproxTensor.softplus {s : Shape} :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.map (shape := s) (Activation.softplus (α := ℝ)) xS)
          (Tensor.map (shape := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) xR)
          (linfNorm
            (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    ApproxTensor.map (α := R)
      (toSpec := NF.toReal)
      (s := s)
      (fS := Activation.softplus (α := ℝ))
      (fR := softplusR (β := β) (fexp := fexp) (rnd := rnd))
      (bnd := fun a eps => softplusBoundScalar (β := β) (fexp := fexp) (rnd := rnd) a eps)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using (approx_softplus (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR)
          (eps := eps) hx))
  simpa [softplusBoundTensor] using h

-- ---------------------------------------------------------------------------
-- safe_log (smooth activation): `log(softplus(x) + ε)`
-- ---------------------------------------------------------------------------

/-- Rounded-real model of `log (softplus x + ε)` with a single rounding of the exact value, where a
staged composition of `softplusR`, addition and `safeLogR` rounds at every step. -/
def safeLogSoftplusR (ε : ℝ) (xR : R) : R :=
  NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (Activation.safeLog (α := ℝ)
      (NF.toReal xR) ε)

private theorem softplus_nonneg (x : ℝ) :
    0 ≤ Activation.softplus (α := ℝ) x := by
  -- `softplus(x) = log(1 + exp(x)) ≥ 0`.
  have h1 : (1 : ℝ) ≤ (1 : ℝ) + Real.exp x := by
    linarith [Real.exp_pos x]
  rw [Activation.softplus_eq_log_one_add_exp]
  exact Real.log_nonneg h1

/--
`Activation.safeLog` is `(1/ε)`-Lipschitz for `ε > 0`.

The derivative `sigmoid x / (softplus x + ε)` lies in `[0, 1/ε]`, and the mean value theorem turns
that bound into the propagation term of `approx_safeLogSoftplus`.
-/
private theorem abs_safeLog_sub_safeLog_le {ε u v : ℝ}
    (hε : 0 < ε) :
    abs (Activation.safeLog (α := ℝ) u ε - Activation.safeLog (α := ℝ) v ε) ≤
      (1 / ε) * abs (u - v) := by
  -- Mean value theorem on `ℝ` (derivative bounded by `1/ε` everywhere).
  have hf :
      ∀ x ∈ (Set.univ : Set ℝ),
        HasDerivWithinAt (fun y : ℝ => Activation.safeLog (α := ℝ) y ε)
          (Activation.safeLogDeriv (α := ℝ) x ε) (Set.univ : Set ℝ) x := by
    intro x _hx
    exact (Activation.hasDerivAt_safeLog (x := x) (ε := ε) hε).hasDerivWithinAt (s := (Set.univ :
      Set ℝ))

  have hbound :
      ∀ x ∈ (Set.univ : Set ℝ),
        ‖Activation.safeLogDeriv (α := ℝ) x ε‖ ≤ (1 / ε) := by
    intro x _hx
    have hsig0 : 0 ≤ Context.sigmoid (α := ℝ) x := Context.sigmoid_nonneg x
    have hsig1 : Context.sigmoid (α := ℝ) x ≤ 1 := Context.sigmoid_le_one x
    have hsoft0 : 0 ≤ Activation.softplus (α := ℝ) x := softplus_nonneg x
    have hden_ge : ε ≤ Activation.softplus (α := ℝ) x + ε := by linarith
    have hden_pos : 0 < Activation.softplus (α := ℝ) x + ε :=
      lt_of_lt_of_le hε hden_ge

    have hderiv_le :
        abs (Activation.safeLogDeriv (α := ℝ) x ε) ≤ (1 / ε) := by
      -- `safe_log'(x) = sigmoid(x) / (softplus(x) + ε)` and `0 ≤ sigmoid ≤ 1`, `softplus+ε ≥ ε`.
      have hone_div :
          (1 : ℝ) / (Activation.softplus (α := ℝ) x + ε) ≤ (1 : ℝ) / ε := by
        simpa [one_div] using (one_div_le_one_div_of_le hε hden_ge)
      have hsig_div :
          Context.sigmoid (α := ℝ) x / (Activation.softplus (α := ℝ) x + ε) ≤
            (1 : ℝ) / (Activation.softplus (α := ℝ) x + ε) := by
        exact div_le_div_of_nonneg_right hsig1 (le_of_lt hden_pos)
      have hpos :
          0 ≤
            Context.sigmoid (α := ℝ) x /
              (Activation.softplus (α := ℝ) x + ε) := by
        exact div_nonneg hsig0 (le_of_lt hden_pos)
      have habs :
          abs
              (Context.sigmoid (α := ℝ) x /
                (Activation.softplus (α := ℝ) x + ε))
            =
          Context.sigmoid (α := ℝ) x /
                (Activation.softplus (α := ℝ) x + ε) :=
        abs_of_nonneg hpos
      have hsimp :
          Activation.safeLogDeriv (α := ℝ) x ε =
            Context.sigmoid (α := ℝ) x /
              (Activation.softplus (α := ℝ) x + ε) := by
        simp [Activation.safeLogDeriv]
      calc
        abs (Activation.safeLogDeriv (α := ℝ) x ε)
            =
          abs
              (Context.sigmoid (α := ℝ) x /
                (Activation.softplus (α := ℝ) x + ε)) := by
              simp [hsimp]
        _ = Context.sigmoid (α := ℝ) x / (Activation.softplus (α := ℝ) x + ε)
          := habs
        _ ≤ (1 : ℝ) / (Activation.softplus (α := ℝ) x + ε) := hsig_div
        _ ≤ (1 : ℝ) / ε := hone_div
        _ = (1 / ε) := by ring

    simpa [Real.norm_eq_abs] using hderiv_le

  have hmv :=
    Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun y : ℝ => Activation.safeLog (α := ℝ) y ε)
      (f' := fun y : ℝ => Activation.safeLogDeriv (α := ℝ) y ε)
      (s := (Set.univ : Set ℝ))
      (x := u) (y := v) (C := (1 / ε))
      hf hbound (convex_univ : Convex ℝ (Set.univ : Set ℝ))
      (by trivial) (by trivial)

  simpa [Real.norm_eq_abs, abs_sub_comm] using hmv

/--
Forward approximation bound in `NF` for the smooth safe log
`Activation.safeLog x ε = log(softplus(x) + ε)`, for `0 < ε`.

The argument of the logarithm stays above `ε`, so the function is smooth on all of `ℝ`. The bound
adds the `(1/ε) * eps` Lipschitz term of the spec function to half an ULP for the single rounding
in `safeLogSoftplusR`.
-/
theorem approx_safeLogSoftplus {x : ℝ} {xR : R} {eps ε : ℝ}
    (hε : 0 < ε)
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs
        (NF.toReal
            (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
          Activation.safeLog (α := ℝ) x ε) ≤
      (1 / ε) * eps +
        ulp β fexp
            (Activation.safeLog (α := ℝ)
              (NF.toReal xR) ε) / 2 := by
  set xhat : ℝ := NF.toReal xR
  have hx' : abs (xhat - x) ≤ eps := hx

  have hround :
      abs
          (NF.toReal
              (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            Activation.safeLog (α := ℝ) xhat ε) ≤
        ulp β fexp (Activation.safeLog (α := ℝ) xhat ε) / 2 := by
    -- `safeLogSoftplusR` rounds the real `Activation.safeLog`.
    simpa [safeLogSoftplusR, xhat, NF.toReal, NF.ofReal,
      NF.roundR] using
        (Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd
          (Activation.safeLog (α := ℝ) xhat ε))

  have hdiff :
      abs (Activation.safeLog (α := ℝ) xhat ε - Activation.safeLog (α := ℝ) x
        ε) ≤
        (1 / ε) * eps := by
    have hL :=
      abs_safeLog_sub_safeLog_le (ε := ε) (u := xhat) (v := x) hε
    have hcoef : 0 ≤ (1 / ε) := by
      exact le_of_lt (one_div_pos.2 hε)
    have hscale : (1 / ε) * abs (xhat - x) ≤ (1 / ε) * eps :=
      mul_le_mul_of_nonneg_left hx' hcoef
    exact le_trans hL hscale

  have :=
    calc
      abs
          (NF.toReal
              (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            Activation.safeLog (α := ℝ) x ε)
          ≤
        abs
            (NF.toReal
                (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
              Activation.safeLog (α := ℝ) xhat ε) +
          abs
              (Activation.safeLog (α := ℝ) xhat ε -
                Activation.safeLog (α := ℝ) x ε) := abs_sub_le _ _ _
      _ ≤
        ulp β fexp (Activation.safeLog (α := ℝ) xhat ε) / 2 +
          (1 / ε) * eps := add_le_add hround hdiff
      _ = (1 / ε) * eps +
        ulp β fexp (Activation.safeLog (α := ℝ) xhat ε) / 2 := by
            ring
  simpa [xhat] using this

/--
Per-entry bound tensor for `safeLogSoftplusR`.

Each entry is `(1/ε) * eps + ulp (safeLog a ε) / 2` at the real value `a` of the entry, the
bound of `approx_safeLogSoftplus`.
-/
def safeLogSoftplusBoundTensor {s : Shape} (ε eps : ℝ) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map
    (fun a =>
      (1 / ε) * eps +
        ulp β fexp (Activation.safeLog (α := ℝ) a ε) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

/--
For `0 < ε`, `ApproxTensor` bound for the smooth safe log `safeLog · ε` at every tensor shape.

The scalar bound `approx_safeLogSoftplus` lifts through `ApproxTensor.map`.
-/
theorem _root_.TorchLean.ApproxTensor.safeLogSoftplus {s : Shape} (ε : ℝ) (hε : 0 < ε) :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.map (shape := s) (fun x => Activation.safeLog (α := ℝ) x ε) xS)
          (Tensor.map (shape := s) (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε) xR)
          (linfNorm
            (safeLogSoftplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s)
              ε eps xR)) := by
  intro xS xR eps hx
  have h :=
    ApproxTensor.map (α := R)
      (toSpec := NF.toReal)
      (s := s)
      (fS := fun x => Activation.safeLog (α := ℝ) x ε)
      (fR := safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε)
      (bnd := fun a eps =>
        (1 / ε) * eps +
          ulp β fexp (Activation.safeLog (α := ℝ) a ε) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using
          (approx_safeLogSoftplus (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (xR := xR) (eps := eps) (ε := ε) hε hx))
  simpa [safeLogSoftplusBoundTensor] using h

end

end NF
end TorchLean
