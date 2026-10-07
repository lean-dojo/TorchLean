/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Gradients.Activation
public import NN.Proofs.RuntimeApprox.NF.Ops.Scalar

/-!
# NF Elementwise Bounds: Softplus and Safe Log

`softplusR` rounds the constant, exponential, addition, and clamped logarithm separately.
In contrast, `safeLogSoftplusR` rounds the entire exact smooth activation once. These are
different numerical schedules; neither definition executes a native kernel.
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

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-- Half-ULP rounding budget at the scalar value `1`, used in the softplus helper bounds. -/
def oneEps : ℝ :=
  ulp β fexp (1 : ℝ) / 2

/-- `oneEps` is a half ulp, hence nonnegative. -/
theorem oneEps_nonneg : 0 ≤ oneEps (β := β) (fexp := fexp) := by
  unfold oneEps
  have := ulp.nonneg β fexp (1 : ℝ)
  linarith

/-- The rounded constant `1 : NF` is within `oneEps` of the real `1`. -/
theorem abs_toSpec_one_sub_one_le :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) - (1 : ℝ)) ≤
      oneEps (β := β) (fexp := fexp) := by
  change
    abs ((NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
      (1 : ℝ)).val - (1 : ℝ)) ≤ oneEps (β := β) (fexp := fexp)
  simpa [oneEps, NFBackend.toSpec, NF.toReal,
    Proofs.RuntimeRoundingApprox.roundR, NF.roundR,
    NF.ofReal] using
    (Proofs.RuntimeRoundingApprox.roundR_abs_error
      (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ))

/-- Rounded representation of the scalar constant `1` at the `NF` backend. -/
def oneHat : ℝ :=
  Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ)

/-- Rounded representation of `exp a` at the `NF` backend. -/
def expHat (a : ℝ) : ℝ :=
  Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (Real.exp a)

/-- Rounded surrogate for the inner `1 + exp a` term appearing in `softplus`. -/
def addHatSoftplus (a : ℝ) : ℝ :=
  Proofs.RuntimeRoundingApprox.roundR (β := β) (fexp := fexp) (rnd := rnd) (oneHat (β := β) (fexp
    := fexp) (rnd := rnd) + expHat (β := β) (fexp := fexp) (rnd := rnd) a)

/-- Forward-error envelope for the rounded `1 + exp a` subexpression used by `softplus`. -/
def addBoundSoftplus (a eps : ℝ) : ℝ :=
  oneEps (β := β) (fexp := fexp) + expErrorBound (β := β) (fexp := fexp) a eps +
    ulp β fexp
        (oneHat (β := β) (fexp := fexp) (rnd := rnd) + expHat (β := β) (fexp := fexp) (rnd := rnd)
          a) / 2

/-- Unconditional scalar forward-error bound for `softplus`, via the `safeLog` factorization. -/
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

We treat `softplus(x) = log(1 + exp x)` as `safeLog 1 (1 + exp x)` (since `1 + exp x ≥ 1`) and then
compose the scalar bounds for `exp`, `+`, and `safeLog`.
-/
theorem approx_softplus_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
          (softplusR (β := β) (fexp := fexp) (rnd := rnd) xR) -
        Activation.Math.softplusSpec (α := ℝ) x) ≤
      softplusBoundScalar (β := β) (fexp := fexp) (rnd := rnd)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps := by
  have hexp :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
        (Numerics.MathFunctions.exp xR) - Real.exp x) ≤
        expErrorBound (β := β) (fexp := fexp) (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
          eps :=
    approx_exp_nf (β := β) (fexp := fexp) (rnd := rnd) hx

  let oneR : R := (1 : R)
  have hOneVal :
      oneR.val = oneHat (β := β) (fexp := fexp) (rnd := rnd) := by
    change
      (NF.ofReal (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ)).val =
        oneHat (β := β) (fexp := fexp) (rnd := rnd)
    rfl
  have hone :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) oneR - (1 : ℝ)) ≤
        oneEps (β := β) (fexp := fexp) :=
    abs_toSpec_one_sub_one_le (β := β) (fexp := fexp) (rnd := rnd)

  let yR : R := oneR + Numerics.MathFunctions.exp xR
  let y : ℝ := (1 : ℝ) + Real.exp x
  have hExpVal :
      (Numerics.MathFunctions.exp xR).val =
        expHat (β := β) (fexp := fexp) (rnd := rnd)
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) := by
    simp [expHat, NFBackend.toSpec, NF.toReal,
      Proofs.RuntimeRoundingApprox.roundR, NF.roundR,
      NF.ofReal, Numerics.MathFunctions.exp]
  have hYVal :
      yR.val =
        addHatSoftplus (β := β) (fexp := fexp) (rnd := rnd)
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) := by
    change
      (NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
        (oneR.val + (Numerics.MathFunctions.exp xR).val)).val =
        addHatSoftplus (β := β) (fexp := fexp) (rnd := rnd)
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
    rw [hOneVal, hExpVal]
    rfl

  have hy :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤
        addBoundSoftplus (β := β) (fexp := fexp) (rnd := rnd)
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps := by
    have hadd :=
      approx_add_nf (β := β) (fexp := fexp) (rnd := rnd)
        (x := (1 : ℝ)) (y := Real.exp x)
        (xR := oneR) (yR := Numerics.MathFunctions.exp xR)
        (epsx := oneEps (β := β) (fexp := fexp))
        (epsy := expErrorBound (β := β) (fexp := fexp)
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps)
        hone hexp
    simpa [oneR, yR, y, addBoundSoftplus, expErrorBound, oneHat, expHat,
      hOneVal, hExpVal,
      NFBackend.toSpec, NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
      NF.roundR, NF.ofReal, NF.instOne,
      NF.instAdd, One.one, Numerics.MathFunctions.exp,
      toSpec_exp (β := β) (fexp := fexp) (rnd := rnd) xR] using hadd

  -- Step 3: `safeLog 1` turns this into `softplus`.
  have hlog :=
    approx_safeLog_nf (β := β) (fexp := fexp) (rnd := rnd)
      (x := y) (xR := yR)
      (eps := addBoundSoftplus (β := β) (fexp := fexp) (rnd := rnd)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps)
      (ε := (1 : ℝ)) (hε := by norm_num) hy
  have hlog' :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
        (safeLogR (β := β) (fexp := fexp) (rnd := rnd) (ε := (1 : ℝ)) yR) -
          safeLog (ε := (1 : ℝ)) y) ≤
        addBoundSoftplus (β := β) (fexp := fexp) (rnd := rnd)
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps +
          ulp β fexp
            (safeLog (ε := (1 : ℝ))
              (addHatSoftplus (β := β) (fexp := fexp) (rnd := rnd)
                (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR))) / 2 := by
    simpa [NFBackend.toSpec, NF.toReal, hYVal] using hlog

  have hy_ge : (1 : ℝ) ≤ y := by
    have : 0 ≤ Real.exp x := by simpa using (Real.exp_nonneg x)
    linarith

  have hsimp : safeLog (ε := (1 : ℝ)) y = Activation.Math.softplusSpec (α := ℝ) x := by
    calc
      safeLog (ε := (1 : ℝ)) y = Real.log y := by
        simp [NFBackend.safeLog, hy_ge]
      _ = Real.log ((1 : ℝ) + Real.exp x) := by
        simp [y]
      _ = Activation.Math.softplusSpec (α := ℝ) x :=
        (softplus_spec_eq_log_one_add_exp x).symm

  simpa [oneR, softplusR, yR, hsimp, softplusBoundScalar, addHatSoftplus,
    hYVal,
    NFBackend.toSpec, NF.toReal, Proofs.RuntimeRoundingApprox.roundR,
    NF.roundR, NF.ofReal, NF.instOne,
    NF.instAdd, One.one, Numerics.MathFunctions.exp,
    toSpec_exp (β := β) (fexp := fexp) (rnd := rnd) xR] using hlog'

/--
Per-entry bound tensor for `softplusR`.

This is the elementwise lifting of `softplusBoundScalar`, used with `linfNorm` in
`approxTensor_softplus_spec`.
-/
def softplusBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  Tensor.map (fun a => softplusBoundScalar (β := β) (fexp := fexp) (rnd := rnd) a eps)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
`approxTensor` bound for `softplus` lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `approx_softplus_nf`, built via
`approxTensor_map_spec_of_scalar_bound`.
-/
theorem approxTensor_softplus_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (Tensor.map (shape := s) (Activation.Math.softplusSpec (α := ℝ)) xS)
          (Tensor.map (shape := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) xR)
          (linfNorm
            (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_scalar_bound (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s)
      (fS := Activation.Math.softplusSpec (α := ℝ))
      (fR := softplusR (β := β) (fexp := fexp) (rnd := rnd))
      (bnd := fun a eps => softplusBoundScalar (β := β) (fexp := fexp) (rnd := rnd) a eps)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using (approx_softplus_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR)
          (eps := eps) hx))
  simpa [softplusBoundTensor] using h

-- ---------------------------------------------------------------------------
-- safe_log (smooth activation): `log(softplus(x) + ε)`
-- ---------------------------------------------------------------------------

/-- Rounded-real model of `log (softplus x + ε)`, with one rounding of the entire expression.
This is not the staged composition of `softplusR`, addition, and `safeLogR`. -/
def safeLogSoftplusR (ε : ℝ) (xR : R) : R :=
  NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (Activation.Math.safeLogSpec (α := ℝ)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) ε)

private theorem softplus_spec_nonneg (x : ℝ) :
    0 ≤ Activation.Math.softplusSpec (α := ℝ) x := by
  -- `softplus(x) = log(1 + exp(x)) ≥ 0`.
  have h1 : (1 : ℝ) ≤ (1 : ℝ) + Real.exp x := by
    linarith [Real.exp_pos x]
  rw [softplus_spec_eq_log_one_add_exp]
  exact Real.log_nonneg h1

/--
`safeLogSpec` is `(1/ε)`-Lipschitz for `ε > 0`.

This is the analytic heart of `approx_safeLogSoftplus_nf`: it bounds how much the spec output
can change when the input changes by `|u - v|`.
-/
private theorem abs_safeLogSpec_sub_safeLogSpec_le_one_div_mul_abs_sub {ε u v : ℝ}
    (hε : 0 < ε) :
    abs (Activation.Math.safeLogSpec (α := ℝ) u ε - Activation.Math.safeLogSpec (α := ℝ) v ε) ≤
      (1 / ε) * abs (u - v) := by
  -- Mean value theorem on `ℝ` (derivative bounded by `1/ε` everywhere).
  have hf :
      ∀ x ∈ (Set.univ : Set ℝ),
        HasDerivWithinAt (fun y : ℝ => Activation.Math.safeLogSpec (α := ℝ) y ε)
          (Activation.Math.safeLogDerivSpec (α := ℝ) x ε) (Set.univ : Set ℝ) x := by
    intro x _hx
    exact (Proofs.safe_log_deriv_correct (x := x) (ε := ε) hε).hasDerivWithinAt (s := (Set.univ :
      Set ℝ))

  have hbound :
      ∀ x ∈ (Set.univ : Set ℝ),
        ‖Activation.Math.safeLogDerivSpec (α := ℝ) x ε‖ ≤ (1 / ε) := by
    intro x _hx
    have hsig0 : 0 ≤ Activation.Math.sigmoidSpec (α := ℝ) x := sigmoid_spec_nonneg x
    have hsig1 : Activation.Math.sigmoidSpec (α := ℝ) x ≤ 1 := sigmoid_spec_le_one x
    have hsoft0 : 0 ≤ Activation.Math.softplusSpec (α := ℝ) x := softplus_spec_nonneg x
    have hden_ge : ε ≤ Activation.Math.softplusSpec (α := ℝ) x + ε := by linarith
    have hden_pos : 0 < Activation.Math.softplusSpec (α := ℝ) x + ε :=
      lt_of_lt_of_le hε hden_ge

    have hderiv_le :
        abs (Activation.Math.safeLogDerivSpec (α := ℝ) x ε) ≤ (1 / ε) := by
      -- `safe_log'(x) = sigmoid(x) / (softplus(x) + ε)` and `0 ≤ sigmoid ≤ 1`, `softplus+ε ≥ ε`.
      have hone_div :
          (1 : ℝ) / (Activation.Math.softplusSpec (α := ℝ) x + ε) ≤ (1 : ℝ) / ε := by
        simpa [one_div] using (one_div_le_one_div_of_le hε hden_ge)
      have hsig_div :
          Activation.Math.sigmoidSpec (α := ℝ) x / (Activation.Math.softplusSpec (α := ℝ) x + ε) ≤
            (1 : ℝ) / (Activation.Math.softplusSpec (α := ℝ) x + ε) := by
        exact div_le_div_of_nonneg_right hsig1 (le_of_lt hden_pos)
      have hpos :
          0 ≤
            Activation.Math.sigmoidSpec (α := ℝ) x /
              (Activation.Math.softplusSpec (α := ℝ) x + ε) := by
        exact div_nonneg hsig0 (le_of_lt hden_pos)
      have habs :
          abs
              (Activation.Math.sigmoidSpec (α := ℝ) x /
                (Activation.Math.softplusSpec (α := ℝ) x + ε))
            =
          Activation.Math.sigmoidSpec (α := ℝ) x /
                (Activation.Math.softplusSpec (α := ℝ) x + ε) :=
        abs_of_nonneg hpos
      have hsimp :
          Activation.Math.safeLogDerivSpec (α := ℝ) x ε =
            Activation.Math.sigmoidSpec (α := ℝ) x /
              (Activation.Math.softplusSpec (α := ℝ) x + ε) := by
        simp [Activation.Math.safeLogDerivSpec, Activation.Math.softplusDerivSpec]
      calc
        abs (Activation.Math.safeLogDerivSpec (α := ℝ) x ε)
            =
          abs
              (Activation.Math.sigmoidSpec (α := ℝ) x /
                (Activation.Math.softplusSpec (α := ℝ) x + ε)) := by
              simp [hsimp]
        _ = Activation.Math.sigmoidSpec (α := ℝ) x / (Activation.Math.softplusSpec (α := ℝ) x + ε)
          := habs
        _ ≤ (1 : ℝ) / (Activation.Math.softplusSpec (α := ℝ) x + ε) := hsig_div
        _ ≤ (1 : ℝ) / ε := hone_div
        _ = (1 / ε) := by ring

    simpa [Real.norm_eq_abs] using hderiv_le

  have hmv :=
    Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun y : ℝ => Activation.Math.safeLogSpec (α := ℝ) y ε)
      (f' := fun y : ℝ => Activation.Math.safeLogDerivSpec (α := ℝ) y ε)
      (s := (Set.univ : Set ℝ))
      (x := u) (y := v) (C := (1 / ε))
      hf hbound (convex_univ : Convex ℝ (Set.univ : Set ℝ))
      (by trivial) (by trivial)

  simpa [Real.norm_eq_abs, abs_sub_comm] using hmv

/--
Forward approximation bound for the smooth `safeLog` activation in `NF`.

`safeLog` is defined as `log(softplus(x) + ε)`, which is globally well-defined for `ε > 0`. The
proof combines:
- one rounding step for `safeLogSoftplusR` (defined as `NF.ofReal (safeLogSpec ...)`);
- a `(1/ε)` Lipschitz bound for the spec function (via mean value theorem + derivative bound).
-/
theorem approx_safeLogSoftplus_nf {x : ℝ} {xR : R} {eps ε : ℝ}
    (hε : 0 < ε)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
          Activation.Math.safeLogSpec (α := ℝ) x ε) ≤
      (1 / ε) * eps +
        ulp β fexp
            (Activation.Math.safeLogSpec (α := ℝ)
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) ε) / 2 := by
  set xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  have hx' : abs (xhat - x) ≤ eps := hx

  have hround :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            Activation.Math.safeLogSpec (α := ℝ) xhat ε) ≤
        ulp β fexp (Activation.Math.safeLogSpec (α := ℝ) xhat ε) / 2 := by
    -- `safeLogSoftplusR` rounds the real `safeLogSpec`.
    simpa [safeLogSoftplusR, xhat, toSpec, NF.toReal, NF.ofReal,
      NF.roundR, Proofs.RuntimeRoundingApprox.roundR] using
        (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd)
          (Activation.Math.safeLogSpec (α := ℝ) xhat ε))

  have hdiff :
      abs (Activation.Math.safeLogSpec (α := ℝ) xhat ε - Activation.Math.safeLogSpec (α := ℝ) x
        ε) ≤
        (1 / ε) * eps := by
    have hL :=
      abs_safeLogSpec_sub_safeLogSpec_le_one_div_mul_abs_sub (ε := ε) (u := xhat) (v := x) hε
    have hcoef : 0 ≤ (1 / ε) := by
      exact le_of_lt (one_div_pos.2 hε)
    have hscale : (1 / ε) * abs (xhat - x) ≤ (1 / ε) * eps :=
      mul_le_mul_of_nonneg_left hx' hcoef
    exact le_trans hL hscale

  have :=
    calc
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
            Activation.Math.safeLogSpec (α := ℝ) x ε)
          ≤
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
              Activation.Math.safeLogSpec (α := ℝ) xhat ε) +
          abs
              (Activation.Math.safeLogSpec (α := ℝ) xhat ε -
                Activation.Math.safeLogSpec (α := ℝ) x ε) := abs_sub_le _ _ _
      _ ≤
        ulp β fexp (Activation.Math.safeLogSpec (α := ℝ) xhat ε) / 2 +
          (1 / ε) * eps := add_le_add hround hdiff
      _ = (1 / ε) * eps +
        ulp β fexp (Activation.Math.safeLogSpec (α := ℝ) xhat ε) / 2 := by
            ring
  simpa [xhat] using this

/--
Per-entry bound tensor for `safeLogSoftplusR`.

This is the elementwise lifting of `approx_safeLogSoftplus_nf`'s bound.
-/
def safeLogSoftplusBoundTensor {s : Shape} (ε eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  Tensor.map
    (fun a =>
      (1 / ε) * eps +
        ulp β fexp (Activation.Math.safeLogSpec (α := ℝ) a ε) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
`approxTensor` bound for the smooth `safeLogSpec` lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `approx_safeLogSoftplus_nf`, built via
`approxTensor_map_spec_of_scalar_bound`.
-/
theorem approxTensor_safeLogSoftplus_spec {s : Shape} (ε : ℝ) (hε : 0 < ε) :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (Tensor.map (shape := s) (fun x => Activation.Math.safeLogSpec (α := ℝ) x ε) xS)
          (Tensor.map (shape := s) (safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε) xR)
          (linfNorm
            (safeLogSoftplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s)
              ε eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_scalar_bound (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s)
      (fS := fun x => Activation.Math.safeLogSpec (α := ℝ) x ε)
      (fR := safeLogSoftplusR (β := β) (fexp := fexp) (rnd := rnd) ε)
      (bnd := fun a eps =>
        (1 / ε) * eps +
          ulp β fexp (Activation.Math.safeLogSpec (α := ℝ) a ε) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using
          (approx_safeLogSoftplus_nf (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (xR := xR) (eps := eps) (ε := ε) hε hx))
  simpa [safeLogSoftplusBoundTensor] using h

end NFBackend

end

end RuntimeApprox
end Proofs
