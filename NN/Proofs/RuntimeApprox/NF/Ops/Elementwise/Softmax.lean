/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.SafeDivSigmoid

/-!
# NF Elementwise Bounds: Scalar Logistic Node

This file bounds the elementwise NF graph node that is registered under the public name
`softmax`. Despite the name, the node computes the scalar logistic sigmoid in its
`exp(x) / (exp(x) + 1)` form, `Activation.Math.logisticSpec`, entrywise. It is not axis softmax;
that operation lives in `NN.Proofs.RuntimeApprox.NF.SoftmaxAxis`. The public names
`softmaxBoundScalar`, `softmaxBoundTensor`, and `approxTensor_softmax_spec` are kept because the
NF forward and backward graph nodes refer to them; the internal names use `logistic`.

The bound is derived from the conditioned division budget `divPosErrorBound`. The exact denominator
`exp(x) + 1` is at least `1`; the rounded denominator error `logisticDenomError` is the numerator
exponential budget plus the rounded constant `1` plus one addition rounding. When that error is
below `1` the bound is the conditioned division budget; otherwise it falls back to `|ℓ̂| + 1`,
which is always valid because the exact logistic lies in `(0, 1)`. The regression theorems
`softmax_bound_scalar_le_of_denom_le_half` and `softmax_bound_scalar_le_one` show the certified
branch is linear in the rounding budgets and stays below `1` for modest formats on the
nonpositive half line.
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

/-- Runtime numerator `exp x` of the rounded logistic node. -/
def logisticNumR (xR : R) : R :=
  Numerics.MathFunctions.exp xR

/-- Runtime denominator `exp x + 1` of the rounded logistic node, exactly as it appears inside
`Activation.Math.logisticSpec` at the `NF` backend. -/
def logisticDenomR (xR : R) : R :=
  Numerics.MathFunctions.exp xR + (1 : R)

/-- Numerator error budget of the logistic node: one rounded exponential at input error `eps`. -/
def logisticNumError (eps : ℝ) (xR : R) : ℝ :=
  expErrorBound (β := β) (fexp := fexp) (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps

/-- Denominator error budget of the logistic node: the numerator budget, the rounded constant `1`,
and one addition rounding. -/
def logisticDenomError (eps : ℝ) (xR : R) : ℝ :=
  logisticNumError (β := β) (fexp := fexp) (rnd := rnd) eps xR +
    oneEps (β := β) (fexp := fexp) +
    ulp β fexp
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.exp xR) +
        toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R)) / 2

omit [ValidRndToNearest rnd] in
/-- `logisticNumError` is nonnegative for nonnegative input error. -/
theorem logisticNumError_nonneg {eps : ℝ} (xR : R) (heps : 0 ≤ eps) :
    0 ≤ logisticNumError (β := β) (fexp := fexp) (rnd := rnd) eps xR :=
  expErrorBound_nonneg (β := β) (fexp := fexp) _ heps

omit [ValidRndToNearest rnd] in
/-- `logisticDenomError` is nonnegative for nonnegative input error. -/
theorem logisticDenomError_nonneg {eps : ℝ} (xR : R) (heps : 0 ≤ eps) :
    0 ≤ logisticDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  unfold logisticDenomError
  have h1 := logisticNumError_nonneg (β := β) (fexp := fexp) (rnd := rnd) xR heps
  have h2 := oneEps_nonneg (β := β) (fexp := fexp)
  have h3 := ulp.nonneg β fexp
    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (Numerics.MathFunctions.exp xR) +
      toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R))
  linarith

/--
Scalar forward bound for the scalar logistic NF node (public name `softmax`) at input error `eps`.

The node computes `exp(x) / (exp(x) + 1)`, whose exact denominator is at least `1`. When the
rounded denominator budget `logisticDenomError eps xR` is below `1`, the bound is the conditioned
division budget `divPosErrorBound` with margin `1 - logisticDenomError`. Otherwise the certificate
fails and the bound falls back to `|ℓ̂| + 1`, valid because the exact logistic lies in `(0, 1)`.
-/
def softmaxBoundScalar (eps : ℝ) (xR : R) : ℝ :=
  if logisticDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1 then
    divPosErrorBound (β := β) (fexp := fexp) 1
      (logisticNumError (β := β) (fexp := fexp) (rnd := rnd) eps xR)
      (logisticDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticNumR xR))
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticDenomR xR))
  else
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (Activation.Math.logisticSpec (α := R) xR)) + 1

/-- Per-entry bound tensor for the scalar logistic NF node; `eps` is the per-entry input error. -/
def softmaxBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  TorchLean.Tensor.map (softmaxBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps) xR

/-- The rounded logistic denominator approximates `exp x + 1` within `logisticDenomError`. -/
theorem approx_logistic_denom_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticDenomR xR) -
        (Real.exp x + 1)) ≤
      logisticDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  have hnum := approx_exp_nf (β := β) (fexp := fexp) (rnd := rnd) hx
  have hone := abs_toSpec_one_sub_one_le (β := β) (fexp := fexp) (rnd := rnd)
  have hadd := approx_add_nf (β := β) (fexp := fexp) (rnd := rnd) hnum hone
  simpa only [logisticDenomR, logisticDenomError, logisticNumError] using hadd

/-- Scalar certificate used by the shape-generic logistic lifting theorem. -/
private theorem approx_logistic_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
          (Activation.Math.logisticSpec (α := R) xR) -
        Activation.Math.logisticSpec (α := ℝ) x) ≤
      softmaxBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  have hnum := approx_exp_nf (β := β) (fexp := fexp) (rnd := rnd) hx
  have hden := approx_logistic_denom_nf (β := β) (fexp := fexp) (rnd := rnd) hx
  have hexp := Real.exp_pos x
  have hy : (1 : ℝ) ≤ Real.exp x + 1 := by linarith
  have hspecR :
      Activation.Math.logisticSpec (α := R) xR = logisticNumR xR / logisticDenomR xR := rfl
  have hspec :
      Activation.Math.logisticSpec (α := ℝ) x = Real.exp x / (Real.exp x + 1) := by
    simp [Activation.Math.logisticSpec, Numerics.MathFunctions.exp]
  unfold softmaxBoundScalar
  split_ifs with hcert
  · rw [hspecR, hspec]
    exact approx_div_nf_of_pos_lb (β := β) (fexp := fexp) (rnd := rnd) (η := 1)
      hy hcert hnum hden
  · have hpos : 0 < Activation.Math.logisticSpec (α := ℝ) x := by
      rw [hspec]
      positivity
    have hle : Activation.Math.logisticSpec (α := ℝ) x ≤ 1 := by
      rw [hspec]
      exact div_le_one_of_le₀ (by linarith) (by positivity)
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (Activation.Math.logisticSpec (α := R) xR) -
          Activation.Math.logisticSpec (α := ℝ) x)
          ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (Activation.Math.logisticSpec (α := R) xR)) +
            abs (Activation.Math.logisticSpec (α := ℝ) x) := abs_sub _ _
      _ ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (Activation.Math.logisticSpec (α := R) xR)) + 1 := by
            rw [abs_of_pos hpos]
            linarith

omit [ValidRndToNearest rnd] in
/-- Regression: under the half-margin certificate `logisticDenomError eps xR ≤ 1 / 2`, the logistic
bound is linear in the numerator budget, the denominator budget, and one output rounding. -/
theorem softmax_bound_scalar_le_of_denom_le_half {eps : ℝ} (xR : R) (heps : 0 ≤ eps)
    (hden : logisticDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 / 2) :
    softmaxBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤
      2 * logisticNumError (β := β) (fexp := fexp) (rnd := rnd) eps xR +
        (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticNumR xR)) +
            logisticNumError (β := β) (fexp := fexp) (rnd := rnd) eps xR) *
          (4 * logisticDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR) +
        ulp β fexp
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticNumR xR) /
            toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticDenomR xR)) / 2 := by
  have hnum0 := logisticNumError_nonneg (β := β) (fexp := fexp) (rnd := rnd) xR heps
  have hden0 := logisticDenomError_nonneg (β := β) (fexp := fexp) (rnd := rnd) xR heps
  have hlt : logisticDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1 := by linarith
  unfold softmaxBoundScalar
  rw [ite_eq_left hlt]
  have h := divPosErrorBound_le_of_epsy_le_half (β := β) (fexp := fexp) (η := 1)
    (xhat := toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticNumR xR))
    (yhat := toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticDenomR xR))
    one_pos hnum0 hden0 (by linarith)
  exact h.trans (le_of_eq (by ring))

omit [ValidRndToNearest rnd] in
/-- Regression: when the rounded numerator `exp(x̂)` is at most `1` (the nonpositive half line),
the numerator and denominator budgets are at most `1/16`, and the output half ulp is at most `1/4`,
the logistic bound is at most `1`. -/
theorem softmax_bound_scalar_le_one {eps : ℝ} (xR : R) (heps : 0 ≤ eps)
    (hnumHat : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticNumR xR)) ≤ 1)
    (hnum : logisticNumError (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 / 16)
    (hden : logisticDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 / 16)
    (hulp : ulp β fexp
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticNumR xR) /
        toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticDenomR xR)) ≤ 1 / 2) :
    softmaxBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 := by
  have h := softmax_bound_scalar_le_of_denom_le_half (β := β) (fexp := fexp) (rnd := rnd)
    xR heps (by linarith)
  have hnum0 := logisticNumError_nonneg (β := β) (fexp := fexp) (rnd := rnd) xR heps
  have hden0 := logisticDenomError_nonneg (β := β) (fexp := fexp) (rnd := rnd) xR heps
  have hprod :
      (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (logisticNumR xR)) +
          logisticNumError (β := β) (fexp := fexp) (rnd := rnd) eps xR) *
        (4 * logisticDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR) ≤
      (1 + 1 / 16) * (4 * (1 / 16)) :=
    mul_le_mul (by linarith) (by linarith) (by positivity) (by norm_num)
  linarith

/--
`approxTensor` bound for the scalar logistic NF node lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around the scalar certificate for `softmaxBoundScalar`, lifted
componentwise through `linfNorm`. It has no side condition: the per-entry bound already switches to
its fallback branch when the denominator certificate fails.
-/
theorem approxTensor_softmax_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) xS)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) xR)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) :=
    by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_runtime_scalar_bound
      (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s)
      (fS := Activation.Math.logisticSpec (α := ℝ))
      (fR := Activation.Math.logisticSpec (α := R))
      (bnd := fun xR inputError =>
        softmaxBoundScalar (β := β) (fexp := fexp) (rnd := rnd) inputError xR)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hxScalar
        exact approx_logistic_nf (β := β) (fexp := fexp) (rnd := rnd) hxScalar)
  simpa [softmaxBoundTensor] using h
end NFBackend

end

end RuntimeApprox
end Proofs
