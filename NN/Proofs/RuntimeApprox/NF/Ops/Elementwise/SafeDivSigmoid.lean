/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.SoftplusSafeLog
public import NN.Spec.Core.FloatInstances.NF

/-!
# NF Elementwise Bounds: Safe Division and Sigmoid

Forward error bounds for clamped division `safeDiv`, ordinary division on a certified positive
denominator domain (`divPosErrorBound`), and the elementwise sigmoid.

The division budget `divPosErrorBound η epsx epsy xhat yhat` requires `|ŷ - y| ≤ epsy` and a
positive lower bound `η ≤ y`. Its conditioning factor is `1 / (η - epsy)`.

Sigmoid chooses between `1 / (1 + exp (-x))` and `exp x / (1 + exp x)`. Each sequence has its own
numerator and denominator budget, assembled from the operations it actually evaluates. The branch
is selected by the rounded input, while both real expressions equal the same logistic function.
The input approximation may therefore cross zero without requiring a separate sign hypothesis.
When a denominator budget is below `1`, the division certificate applies; otherwise the bound is
`|σ̂| + 1`, using the range of the exact sigmoid.

The `reciprocalSigmoid` declarations retain the first sequence and its half-margin regressions.
They describe that sequence even on negative inputs. `sigmoidBoundScalar` and
`approxTensor_sigmoid_spec` describe the branch-selected public operation.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open TorchLean.Floats
open Proofs.RuntimeRoundingApprox

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

-- ---------------------------------------------------------------------------
-- Safe division (clamped): `x / max y ε`
-- ---------------------------------------------------------------------------

/-- Spec-side safe division with a clamped denominator. -/
def safeDiv (ε : ℝ) (x y : ℝ) : ℝ :=
  x / max y ε

/-- Runtime implementation of `safeDiv` as a single rounded primitive. -/
def safeDivR (ε : ℝ) (xR yR : R) : R :=
  TorchLean.Floats.NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (safeDiv (ε := ε)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR))

/--
Forward approximation bound for `safeDiv` in `NF`.

`safeDiv ε x y = x / max y ε` clamps the denominator away from 0. For `ε > 0`, this yields an
unconditional bound with explicit `(1/ε)` and `(1/ε^2)` sensitivity terms plus one rounding-ULP
  term.
-/
theorem approx_safeDiv_nf {x y : ℝ} {xR yR : R} {epsx epsy ε : ℝ}
    (hε : 0 < ε)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
          safeDiv (ε := ε) x y) ≤
      (1 / ε) * epsx +
        (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) + epsx) * (epsy / (ε * ε)) +
        neuralUlp β fexp
            (safeDiv (ε := ε)
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR)) / 2 := by
  set xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  set yhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) yR
  set uhat : ℝ := max yhat ε
  set u : ℝ := max y ε

  have uhat_ge : ε ≤ uhat := le_max_right _ _
  have u_ge : ε ≤ u := le_max_right _ _
  have uhat_pos : 0 < uhat := lt_of_lt_of_le hε uhat_ge
  have u_pos : 0 < u := lt_of_lt_of_le hε u_ge

  have hx' : abs (xhat - x) ≤ epsx := by
    simpa [xhat, abs_sub_comm] using hx
  have hy' : abs (yhat - y) ≤ epsy := by
    simpa [yhat, abs_sub_comm] using hy

  have hx_abs : abs x ≤ abs xhat + epsx := by
    have h0 : abs x ≤ abs (x - xhat) + abs xhat := by
      simpa using (abs_sub_le x xhat 0)
    have h1 : abs (x - xhat) = abs (xhat - x) := by simp [abs_sub_comm]
    have h2 : abs (x - xhat) ≤ epsx := by simpa [h1] using hx'
    have := le_trans h0 (by
      simpa [add_assoc, add_left_comm, add_comm] using add_le_add_right h2 (abs xhat))
    simpa [add_assoc, add_left_comm, add_comm] using this

  have hmax : abs (uhat - u) ≤ epsy := by
    have hLip : abs (max yhat ε - max y ε) ≤ abs (yhat - y) := by
      simpa [abs_sub_comm] using (abs_max_sub_max_le_abs yhat y ε)
    exact le_trans (by simpa [uhat, u, abs_sub_comm] using hLip) hy'

  have hround :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
            safeDiv (ε := ε) xhat yhat) ≤
        neuralUlp β fexp (safeDiv (ε := ε) xhat yhat) / 2 := by
    simpa [safeDivR, safeDiv, xhat, yhat, toSpec, TorchLean.Floats.NF.toReal,
      TorchLean.Floats.NF.ofReal,
      TorchLean.Floats.NF.roundR, Proofs.RuntimeRoundingApprox.roundR] using
        (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd)
          (safeDiv (ε := ε) xhat yhat))

  have hdiff :
      abs (safeDiv (ε := ε) xhat yhat - safeDiv (ε := ε) x y) ≤
        (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) := by
    -- Split numerator and denominator effects.
    have hsplit :
        abs (xhat / uhat - x / u) ≤ abs (xhat / uhat - x / uhat) + abs (x / uhat - x / u) := by
      -- `|a-c| ≤ |a-b| + |b-c|` with `b = x/uhat`.
      simpa [sub_eq_add_neg, add_assoc] using
        abs_sub_le (xhat / uhat) (x / uhat) (x / u)

    have hnum :
        abs (xhat / uhat - x / uhat) ≤ (1 / ε) * epsx := by
      have hsub : xhat / uhat - x / uhat = (xhat - x) / uhat := by
        simpa using (sub_div xhat x uhat).symm
      have hinv : (1 : ℝ) / uhat ≤ (1 : ℝ) / ε := by
        simpa [one_div] using (one_div_le_one_div_of_le hε uhat_ge)
      have hcoef : 0 ≤ (1 : ℝ) / ε := by exact le_of_lt (one_div_pos.2 hε)
      calc
        abs (xhat / uhat - x / uhat)
            = abs ((xhat - x) / uhat) := by simp [hsub]
        _ = abs (xhat - x) * ((1 : ℝ) / uhat) := by
                simp [div_eq_mul_inv, abs_mul, abs_inv, abs_of_pos uhat_pos]
        _ ≤ abs (xhat - x) * ((1 : ℝ) / ε) := by
              exact mul_le_mul_of_nonneg_left hinv (abs_nonneg _)
        _ ≤ epsx * ((1 : ℝ) / ε) := by
              exact mul_le_mul_of_nonneg_right hx' hcoef
        _ = (1 / ε) * epsx := by ring

    have hden :
        abs (x / uhat - x / u) ≤ (abs xhat + epsx) * (epsy / (ε * ε)) := by
      have hsub : x / uhat - x / u = x * ((1 : ℝ) / uhat - (1 : ℝ) / u) := by
        simp [div_eq_mul_inv, sub_eq_add_neg, mul_add, mul_comm]
      -- Bound `|1/uhat - 1/u|` using algebra and the `max` Lipschitz bound.
      have h_inv :
          abs ((1 : ℝ) / uhat - (1 : ℝ) / u) ≤ epsy / (ε * ε) := by
        have hu0 : uhat ≠ 0 := ne_of_gt uhat_pos
        have hv0 : u ≠ 0 := ne_of_gt u_pos
        have hiden :
            (1 : ℝ) / uhat - (1 : ℝ) / u = (u - uhat) / (uhat * u) := by
          field_simp [hu0, hv0]
        have hprod_ge : (ε * ε) ≤ uhat * u := by
          have : ε ≤ uhat := uhat_ge
          have : ε ≤ u := u_ge
          nlinarith
        have hprod_pos : 0 < uhat * u := mul_pos uhat_pos u_pos
        have hprod_inv :
            (1 : ℝ) / (uhat * u) ≤ (1 : ℝ) / (ε * ε) := by
          simpa [one_div] using (one_div_le_one_div_of_le (mul_pos hε hε) hprod_ge)
        have hprod_inv_nonneg : 0 ≤ (1 : ℝ) / (uhat * u) := by
          exact le_of_lt (one_div_pos.2 hprod_pos)
        calc
          abs ((1 : ℝ) / uhat - (1 : ℝ) / u)
              = abs ((u - uhat) / (uhat * u)) := by
                  simpa using congrArg abs hiden
          _ = abs (u - uhat) / (uhat * u) := by
                  simpa [abs_of_pos hprod_pos] using (abs_div (u - uhat) (uhat * u))
          _ = abs (u - uhat) * ((1 : ℝ) / (uhat * u)) := by
                  simp [div_eq_mul_inv]
          _ ≤ abs (u - uhat) * ((1 : ℝ) / (ε * ε)) := by
                exact mul_le_mul_of_nonneg_left hprod_inv (abs_nonneg _)
          _ ≤ epsy * ((1 : ℝ) / (ε * ε)) := by
                have : abs (u - uhat) ≤ epsy := by simpa [abs_sub_comm] using hmax
                exact mul_le_mul_of_nonneg_right this (by
                  have : 0 < (1 : ℝ) / (ε * ε) := by
                    exact one_div_pos.2 (mul_pos hε hε)
                  exact le_of_lt this)
          _ = epsy / (ε * ε) := by
                simp [div_eq_mul_inv, mul_comm]

      calc
        abs (x / uhat - x / u)
            = abs (x * ((1 : ℝ) / uhat - (1 : ℝ) / u)) := by simp [hsub]
        _ = abs x * abs ((1 : ℝ) / uhat - (1 : ℝ) / u) := by
              simp [abs_mul]
        _ ≤ (abs xhat + epsx) * abs ((1 : ℝ) / uhat - (1 : ℝ) / u) := by
              exact mul_le_mul_of_nonneg_right hx_abs (abs_nonneg _)
        _ ≤ (abs xhat + epsx) * (epsy / (ε * ε)) := by
              have epsx_nonneg : 0 ≤ epsx := le_trans (abs_nonneg _) hx'
              have hsum_nonneg : 0 ≤ abs xhat + epsx := add_nonneg (abs_nonneg _) epsx_nonneg
              exact mul_le_mul_of_nonneg_left h_inv hsum_nonneg

    -- Combine.
    have hadd :=
      calc
        abs (xhat / uhat - x / u)
            ≤ abs (xhat / uhat - x / uhat) + abs (x / uhat - x / u) := hsplit
        _ ≤ (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) := by
              exact add_le_add hnum hden
        _ = (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) := by rfl
    -- Rewrite `safeDiv`.
    simpa [safeDiv, uhat, u] using hadd

  -- Final triangle inequality: rounding + input sensitivity.
  have :=
    calc
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
            safeDiv (ε := ε) x y)
          ≤
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
              safeDiv (ε := ε) xhat yhat) +
          abs (safeDiv (ε := ε) xhat yhat - safeDiv (ε := ε) x y) := by
            simpa [sub_eq_add_neg, add_assoc] using
              abs_sub_le
                (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                  (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR))
                (safeDiv (ε := ε) xhat yhat)
                (safeDiv (ε := ε) x y)
      _ ≤
        neuralUlp β fexp (safeDiv (ε := ε) xhat yhat) / 2 +
          ((1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε))) := by
            exact add_le_add hround hdiff
      _ = (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) +
        neuralUlp β fexp (safeDiv (ε := ε) xhat yhat) / 2 := by
            ring

  simpa [xhat, yhat] using this

/-- Error budget for division with exact denominator lower bound `η` and denominator approximation
error `epsy`. The caller must separately establish `epsy < η`; otherwise the rounded denominator
may cross zero and no finite perturbation bound follows.
-/
def divPosErrorBound (η epsx epsy xhat yhat : ℝ) : ℝ :=
  (1 / (η - epsy)) * epsx +
    (abs xhat + epsx) * (epsy / ((η - epsy) * (η - epsy))) +
    neuralUlp β fexp (xhat / yhat) / 2

/-- Forward error for ordinary division when the exact denominator stays positively separated
from zero and its approximation budget is smaller than that separation.

The effective runtime margin is `η - epsy`: from `η ≤ y` and `|ŷ - y| ≤ epsy` we obtain
`η - epsy ≤ ŷ`. The result is proved through the shared clamped-division analysis, after showing
that neither the exact nor rounded denominator activates the clamp. This is the form needed by
stable softmax and normalization, where a mathematical lower bound on a reduction must survive
rounding before division is allowed.
-/
theorem approx_div_nf_of_pos_lb {x y : ℝ} {xR yR : R} {epsx epsy η : ℝ}
    (hyLower : η ≤ y) (hbudget : epsy < η)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) - x / y) ≤
      divPosErrorBound (β := β) (fexp := fexp) η epsx epsy
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) := by
  let margin : ℝ := η - epsy
  let yhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) yR
  have hepsy : 0 ≤ epsy :=
    le_trans (abs_nonneg (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y)) hy
  have hmargin : 0 < margin := by
    dsimp [margin]
    linarith
  have hyhatLower : margin ≤ yhat := by
    have hdiff : y - yhat ≤ epsy := by
      calc
        y - yhat ≤ abs (y - yhat) := le_abs_self _
        _ = abs (yhat - y) := abs_sub_comm _ _
        _ ≤ epsy := by simpa [yhat] using hy
    dsimp [margin]
    linarith
  have hyMargin : margin ≤ y := by
    dsimp [margin]
    linarith
  have hmaxHat : max (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) margin =
      toSpec (β := β) (fexp := fexp) (rnd := rnd) yR :=
    max_eq_left (by simpa [yhat] using hyhatLower)
  have hmaxReal : max y margin = y := max_eq_left hyMargin
  have hruntime :
      safeDivR (β := β) (fexp := fexp) (rnd := rnd) margin xR yR = xR / yR := by
    simp only [safeDivR, safeDiv, hmaxHat]
    rfl
  have hspec : safeDiv (ε := margin) x y = x / y := by
    simp [safeDiv, hmaxReal]
  have hsafe :=
    approx_safeDiv_nf (β := β) (fexp := fexp) (rnd := rnd)
      (x := x) (y := y) (xR := xR) (yR := yR)
      (epsx := epsx) (epsy := epsy) (ε := margin) hmargin hx hy
  rw [hruntime, hspec] at hsafe
  simpa [safeDiv, hmaxHat, margin, divPosErrorBound] using hsafe

/-- Under the half-margin certificate `epsy ≤ η / 2`, the division budget is controlled by the
numerator error, the denominator error, and one output rounding, with constants that do not depend
on the format. This is the shared regression lemma for sigmoid, logistic, and mean bounds. -/
theorem divPosErrorBound_le_of_epsy_le_half {η epsx epsy xhat yhat : ℝ}
    (hη : 0 < η) (hepsx : 0 ≤ epsx) (hepsy : 0 ≤ epsy) (hhalf : epsy ≤ η / 2) :
    divPosErrorBound (β := β) (fexp := fexp) η epsx epsy xhat yhat ≤
      (2 / η) * epsx + (abs xhat + epsx) * (4 * epsy / (η * η)) +
        neuralUlp β fexp (xhat / yhat) / 2 := by
  have hmargin : η / 2 ≤ η - epsy := by linarith
  have hhalfpos : 0 < η / 2 := by positivity
  have hinv : 1 / (η - epsy) ≤ 2 / η := by
    calc
      1 / (η - epsy) ≤ 1 / (η / 2) := one_div_le_one_div_of_le hhalfpos hmargin
      _ = 2 / η := by field_simp
  have hsq : epsy / ((η - epsy) * (η - epsy)) ≤ 4 * epsy / (η * η) := by
    have hmm : (η / 2) * (η / 2) ≤ (η - epsy) * (η - epsy) :=
      mul_le_mul hmargin hmargin hhalfpos.le (by linarith)
    calc
      epsy / ((η - epsy) * (η - epsy)) ≤ epsy / ((η / 2) * (η / 2)) :=
        div_le_div_of_nonneg_left hepsy (by positivity) hmm
      _ = 4 * epsy / (η * η) := by field_simp; ring
  have h1 : (1 / (η - epsy)) * epsx ≤ (2 / η) * epsx :=
    mul_le_mul_of_nonneg_right hinv hepsx
  have h2 :
      (abs xhat + epsx) * (epsy / ((η - epsy) * (η - epsy))) ≤
        (abs xhat + epsx) * (4 * epsy / (η * η)) :=
    mul_le_mul_of_nonneg_left hsq (add_nonneg (abs_nonneg _) hepsx)
  unfold divPosErrorBound
  linarith

/--
Per-entry bound tensor for `safeDiv`.

This is the elementwise lifting of `approx_safeDiv_nf`'s bound (with a max-clamped denominator).
-/
def safeDivBoundTensor {s : Shape} (ε epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      (1 / ε) * epsx +
        (abs a + epsx) * (epsy / (ε * ε)) +
        neuralUlp β fexp (safeDiv (ε := ε) a b) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/-- Per-entry budget for ordinary division on a certified positive denominator domain.

Unlike `safeDivBoundTensor`, this definition does not change the operation by clamping its
denominator. The accompanying theorem therefore requires `epsy < η`, ensuring that an exact lower
bound `η ≤ y` remains positive after the denominator is rounded.
-/
def divPosBoundTensor {s : Shape} (η epsx epsy : ℝ)
    (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (divPosErrorBound (β := β) (fexp := fexp) η epsx epsy)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/-- Shape-generic forward error for ordinary elementwise division by positive denominators.

The domain condition is stated over the exact tensor, while `epsy < η` certifies that every
runtime denominator remains separated from zero. This is the reusable division rule for softmax,
normalization, and positive quantization scales; callers do not need a rank-specific theorem.
-/
theorem approxTensor_div_spec_of_pos_lb {s : Shape} (η : ℝ) :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
      Tensor.Forall (fun y : ℝ => η ≤ y) yS →
      epsy < η →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (divSpec xS yS) (divSpec xR yR)
          (linfNorm (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) η epsx epsy xR yR)) := by
  induction s with
  | scalar =>
      intro xS yS xR yR epsx epsy hx hy hdom hmargin
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR] at hx
      rw [← Tensor.scalar_item yS, ← Tensor.scalar_item yR] at hy
      rw [← Tensor.scalar_item yS] at hdom
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item yS,
        ← Tensor.scalar_item xR, ← Tensor.scalar_item yR]
      have hx' := (approxTensor_scalar_iff (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).mp hx
      have hy' := (approxTensor_scalar_iff (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).mp hy
      have hdiv := approx_div_nf_of_pos_lb
        (β := β) (fexp := fexp) (rnd := rnd)
        (by simpa [Tensor.Forall] using hdom) hmargin hx' hy'
      apply (approxTensor_scalar_iff (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).mpr
      change
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR.item / yR.item) -
              xS.item / yS.item) ≤
          abs
            (divPosErrorBound (β := β) (fexp := fexp) η epsx epsy
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR.item)
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR.item))
      exact le_trans hdiv (le_abs_self _)
  | dim n inner ih =>
      intro xS yS xR yR epsx epsy hx hy hdom hmargin
      let bound := linfNorm
        (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := .dim n inner) η epsx epsy xR yR)
      have hbound : 0 ≤ bound := linf_norm_nonneg _
      refine approxTensor_dim_of_forall
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (xS := divSpec xS yS) (xR := divSpec xR yR)
        (eps := bound) hbound ?_
      intro i
      have hxI := approxTensor_dim_get (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i
      have hyI := approxTensor_dim_get (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hy i
      have hlocal := ih hxI hyI (by simpa using hdom i) hmargin
      have hle :
          linfNorm
              (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := inner) η epsx epsy (xR.unstack i) (yR.unstack i)) ≤ bound := by
        have h := linf_norm_le_get_dim
          (t := divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := .dim n inner) η epsx epsy xR yR) i
        simpa [bound, divPosBoundTensor, tensorToSpec, map2Spec, mapSpec,
          TorchLean.Tensor.map, TorchLean.Tensor.unstack,
          TorchLean.Tensor.Internal.Rep.map_unstack,
          TorchLean.Tensor.Internal.Rep.zipWith_unstack] using h
      have hmono := approxTensor_mono hlocal hle
      simpa [divSpec, map2Spec, TorchLean.Tensor.unstack,
        TorchLean.Tensor.Internal.Rep.zipWith_unstack] using hmono

/--
`approxTensor` bound for `safeDiv` lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `approx_safeDiv_nf`, built via
  `approxTensor_map2_spec_of_scalar_bound`.
-/
theorem approxTensor_safeDiv_spec {s : Shape} (ε : ℝ) (hε : 0 < ε) :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (map2Spec (s := s) (safeDiv (ε := ε)) xS yS)
          (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε) xR yR)
          (linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε epsx epsy
            xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    approxTensor_map2_spec_of_scalar_bound (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s)
      (fS := safeDiv (ε := ε))
      (fR := safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
      (bnd := fun a b epsx epsy =>
        (1 / ε) * epsx +
          (abs a + epsx) * (epsy / (ε * ε)) +
          neuralUlp β fexp (safeDiv (ε := ε) a b) / 2)
      (xS := xS) (yS := yS) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy) hx hy (by
        intro x y xR yR hx hy
        simpa using
          (approx_safeDiv_nf (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (y := y) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy) (ε := ε) hε hx
              hy))
  simpa [safeDivBoundTensor] using h

-- Sigmoid (elementwise logistic) bounds.

/-- The rounded constant `1 : NF` is within `oneEps` of the real `1`. -/
theorem abs_toSpec_one_sub_one_le :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) - (1 : ℝ)) ≤
      oneEps (β := β) (fexp := fexp) := by
  change
    abs ((TorchLean.Floats.NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
      (1 : ℝ)).val - (1 : ℝ)) ≤ oneEps (β := β) (fexp := fexp)
  simpa [oneEps, NFBackend.toSpec, TorchLean.Floats.NF.toReal,
    Proofs.RuntimeRoundingApprox.roundR, TorchLean.Floats.NF.roundR,
    TorchLean.Floats.NF.ofReal] using
    (Proofs.RuntimeRoundingApprox.roundR_abs_error
      (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ))

/-- `oneEps` is a half ulp, hence nonnegative. -/
theorem oneEps_nonneg : 0 ≤ oneEps (β := β) (fexp := fexp) := by
  unfold oneEps
  have := neuralUlp.nonneg β fexp (1 : ℝ)
  linarith

/-- `expErrorBound` is nonnegative whenever the propagated input error is. -/
theorem expErrorBound_nonneg (a : ℝ) {eps : ℝ} (heps : 0 ≤ eps) :
    0 ≤ expErrorBound (β := β) (fexp := fexp) a eps := by
  unfold expErrorBound
  have h1 : 0 ≤ Real.exp (a + eps) * eps := mul_nonneg (Real.exp_pos _).le heps
  have h2 := neuralUlp.nonneg β fexp (Real.exp a)
  linarith

/-- Sigmoid evaluated as `1 / (1 + exp (-x))` for every input.

This sequence is retained for its NF rounding certificate. The public sigmoid chooses another
sequence on nonpositive inputs, so the two rounded results need not agree. Both approximate the
same real logistic function. -/
def reciprocalSigmoidR (xR : R) : R :=
  (1 : R) / (1 + MathFunctions.exp (-xR))

/-- Rounded denominator of `reciprocalSigmoidR`. -/
def reciprocalSigmoidDenomR (xR : R) : R :=
  (1 : R) + MathFunctions.exp (-xR)

/-- Error budget of the rounded sigmoid denominator `1 + exp(-x)` given input error `eps`.

The summands pay for the rounded constant `1`, the rounded exponential of the rounded negation
(whose input error is `eps` plus one negation rounding), and the final addition rounding. -/
def reciprocalSigmoidDenomError (eps : ℝ) (xR : R) : ℝ :=
  oneEps (β := β) (fexp := fexp) +
    expErrorBound (β := β) (fexp := fexp)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR))
      (eps + neuralUlp β fexp (-toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) / 2) +
    neuralUlp β fexp
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) +
        toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.exp (-xR))) / 2

omit [NeuralValidRndToNearest rnd] in
/-- `reciprocalSigmoidDenomError` is nonnegative for nonnegative input error. -/
theorem reciprocalSigmoidDenomError_nonneg {eps : ℝ} (xR : R) (heps : 0 ≤ eps) :
    0 ≤ reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  unfold reciprocalSigmoidDenomError
  have h1 := oneEps_nonneg (β := β) (fexp := fexp)
  have hulp0 : 0 ≤ neuralUlp β fexp (-toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) / 2 := by
    have := neuralUlp.nonneg β fexp (-toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
    linarith
  have h2 := expErrorBound_nonneg (β := β) (fexp := fexp)
    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (-xR)) (add_nonneg heps hulp0)
  have h3 := neuralUlp.nonneg β fexp
    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) +
      toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.exp (-xR)))
  linarith

/--
Scalar forward bound for `reciprocalSigmoidR` at input error `eps`.

`sigmoid(x) = 1 / (1 + exp(-x))` has exact denominator at least `1`. When the rounded denominator
budget `reciprocalSigmoidDenomError eps xR` is below `1`, the rounded denominator stays positive.
The bound then uses `divPosErrorBound` with margin `1 - reciprocalSigmoidDenomError`.
Otherwise the certificate fails and the bound falls back to `|σ̂| + 1`, which is always valid
because the exact sigmoid lies in `(0, 1]`.
-/
def reciprocalSigmoidBoundScalar (eps : ℝ) (xR : R) : ℝ :=
  if reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1 then
    divPosErrorBound (β := β) (fexp := fexp) 1 (oneEps (β := β) (fexp := fexp))
      (reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R))
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (reciprocalSigmoidDenomR xR))
  else
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR)) + 1

/-- Per-entry bound tensor for `reciprocalSigmoidR`; `eps` is the per-entry input error. -/
def reciprocalSigmoidBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  TorchLean.Tensor.map (reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps) xR

/-- The reciprocal sequence's denominator approximates `1 + exp(-x)` within its composed budget. -/
theorem approx_reciprocal_sigmoid_denom_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (reciprocalSigmoidDenomR xR) -
        (1 + Real.exp (-x))) ≤
      reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  have hneg := approx_neg_nf (β := β) (fexp := fexp) (rnd := rnd) hx
  have hexp := approx_exp_nf (β := β) (fexp := fexp) (rnd := rnd) hneg
  have hone := abs_toSpec_one_sub_one_le (β := β) (fexp := fexp) (rnd := rnd)
  have hadd := approx_add_nf (β := β) (fexp := fexp) (rnd := rnd) hone hexp
  simpa only [reciprocalSigmoidDenomR, reciprocalSigmoidDenomError] using hadd

/-- Scalar certificate for the reciprocal sequence and its shape-generic lifting theorem. -/
private theorem approx_reciprocal_sigmoid_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
          (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR) -
        Activation.Math.sigmoidSpec (α := ℝ) x) ≤
      reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  have hden := approx_reciprocal_sigmoid_denom_nf (β := β) (fexp := fexp) (rnd := rnd) hx
  have hone := abs_toSpec_one_sub_one_le (β := β) (fexp := fexp) (rnd := rnd)
  have hy : (1 : ℝ) ≤ 1 + Real.exp (-x) := by linarith [Real.exp_pos (-x)]
  have hspecR :
      reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR =
        (1 : R) / reciprocalSigmoidDenomR xR := rfl
  have hspec : Activation.Math.sigmoidSpec (α := ℝ) x = 1 / (1 + Real.exp (-x)) := by
    rw [Proofs.sigmoid_eq_inv_exp, one_div]
  unfold reciprocalSigmoidBoundScalar
  split_ifs with hcert
  · rw [hspecR, hspec]
    exact approx_div_nf_of_pos_lb (β := β) (fexp := fexp) (rnd := rnd) (η := 1) hy hcert hone hden
  · have hpos : 0 < Activation.Math.sigmoidSpec (α := ℝ) x := by
      rw [hspec]
      positivity
    have hle : Activation.Math.sigmoidSpec (α := ℝ) x ≤ 1 := by
      rw [hspec]
      exact div_le_one_of_le₀ hy (by positivity)
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR) -
          Activation.Math.sigmoidSpec (α := ℝ) x)
          ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR)) +
            abs (Activation.Math.sigmoidSpec (α := ℝ) x) := abs_sub _ _
      _ ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR)) + 1 := by
            rw [abs_of_pos hpos]
            linarith

omit [NeuralValidRndToNearest rnd] in
/-- With denominator error at most `1/2`, the reciprocal sequence's bound is linear in the
rounding budget of the constant `1`, the denominator error, and one output rounding.
In particular it tends to the output half ulp as the format is refined. -/
theorem reciprocal_sigmoid_bound_scalar_le_of_denom_le_half {eps : ℝ} (xR : R) (heps : 0 ≤ eps)
    (hden : reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 / 2) :
    reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤
      2 * oneEps (β := β) (fexp := fexp) +
        (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R)) +
            oneEps (β := β) (fexp := fexp)) *
          (4 * reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR) +
        neuralUlp β fexp
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) /
            toSpec (β := β) (fexp := fexp) (rnd := rnd) (reciprocalSigmoidDenomR xR)) / 2 := by
  have hden0 := reciprocalSigmoidDenomError_nonneg (β := β) (fexp := fexp) (rnd := rnd) xR heps
  have hlt : reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1 := by
    linarith
  unfold reciprocalSigmoidBoundScalar
  rw [if_pos hlt]
  have h := divPosErrorBound_le_of_epsy_le_half (β := β) (fexp := fexp) (η := 1)
    (xhat := toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R))
    (yhat := toSpec (β := β) (fexp := fexp) (rnd := rnd) (reciprocalSigmoidDenomR xR))
    one_pos (oneEps_nonneg (β := β) (fexp := fexp)) hden0 (by linarith)
  exact h.trans (le_of_eq (by ring))

/-- For a format whose half ulp at `1` is at most `1/16`, whose output half ulp is at most `1/4`,
and whose rounded denominator error is at most `1/16`, the reciprocal sequence's bound is at
most `1`. -/
theorem reciprocal_sigmoid_bound_scalar_le_one {eps : ℝ} (xR : R) (heps : 0 ≤ eps)
    (hone : oneEps (β := β) (fexp := fexp) ≤ 1 / 16)
    (hden : reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 / 16)
    (hulp : neuralUlp β fexp
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) /
        toSpec (β := β) (fexp := fexp) (rnd := rnd) (reciprocalSigmoidDenomR xR)) ≤ 1 / 2) :
    reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 := by
  have h := reciprocal_sigmoid_bound_scalar_le_of_denom_le_half (β := β) (fexp := fexp) (rnd := rnd)
    xR heps (by linarith)
  have hden0 := reciprocalSigmoidDenomError_nonneg (β := β) (fexp := fexp) (rnd := rnd) xR heps
  have hone0 := oneEps_nonneg (β := β) (fexp := fexp)
  have habs : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R)) ≤
      1 + oneEps (β := β) (fexp := fexp) := by
    have h1 := abs_toSpec_one_sub_one_le (β := β) (fexp := fexp) (rnd := rnd)
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R))
          = abs ((toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) - 1) + 1) := by ring_nf
      _ ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) - 1) + abs (1 : ℝ) :=
          abs_add_le _ _
      _ ≤ 1 + oneEps (β := β) (fexp := fexp) := by
          rw [abs_one]
          linarith
  have hprod :
      (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R)) +
          oneEps (β := β) (fexp := fexp)) *
        (4 * reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR) ≤
      (1 + 1 / 16 + 1 / 16) * (4 * (1 / 16)) :=
    mul_le_mul (by linarith) (by linarith) (by positivity) (by norm_num)
  linarith

/--
`approxTensor` bound for the reciprocal sigmoid sequence at arbitrary tensor shapes.

The scalar certificate for `reciprocalSigmoidBoundScalar` lifts componentwise through `linfNorm`.
There is no side condition: the per-entry bound already switches to its fallback branch when the
denominator certificate fails.
-/
theorem approxTensor_reciprocal_sigmoid_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) xS)
          (mapSpec (s := s) (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd)) xR)
          (linfNorm (reciprocalSigmoidBoundTensor
            (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) :=
    by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_runtime_scalar_bound
      (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s)
      (fS := Activation.Math.sigmoidSpec (α := ℝ))
      (fR := reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd))
      (bnd := fun xR inputError =>
        reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) inputError xR)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hxScalar
        exact approx_reciprocal_sigmoid_nf (β := β) (fexp := fexp) (rnd := rnd) hxScalar)
  simpa [reciprocalSigmoidBoundTensor] using h

/-- The negative-input sigmoid sequence, with a shared rounded exponential in the numerator
and denominator. It is defined for every NF input so its certificate can be stated independently
of the comparison that selects the public sigmoid branch. -/
def expRatioSigmoidR (xR : R) : R :=
  let z := MathFunctions.exp xR
  z / (1 + z)

/-- Rounded denominator of `expRatioSigmoidR`. -/
def expRatioSigmoidDenomR (xR : R) : R :=
  (1 : R) + MathFunctions.exp xR

/-- Error in the negative-input sequence's denominator. The exponential has input error `eps`;
the other two terms account for representing `1` and adding it to that exponential. -/
def expRatioSigmoidDenomError (eps : ℝ) (xR : R) : ℝ :=
  oneEps (β := β) (fexp := fexp) +
    expErrorBound (β := β) (fexp := fexp)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps +
    neuralUlp β fexp
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) +
        toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.exp xR)) / 2

/-- Forward-error budget for the exponential-ratio sequence. Both occurrences of `exp x` share
the same rounded value, but the division estimate only needs separate numerator and denominator
error bounds. The exact denominator is at least `1`. -/
def expRatioSigmoidBoundScalar (eps : ℝ) (xR : R) : ℝ :=
  if expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1 then
    divPosErrorBound (β := β) (fexp := fexp) 1
      (expErrorBound (β := β) (fexp := fexp)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps)
      (expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (MathFunctions.exp xR))
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (expRatioSigmoidDenomR xR))
  else
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
      (expRatioSigmoidR xR)) + 1

/-- The exponential-ratio denominator approximates `1 + exp x` within its composed budget. -/
theorem approx_exp_ratio_sigmoid_denom_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (expRatioSigmoidDenomR xR) -
        (1 + Real.exp x)) ≤
      expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  have hexp := approx_exp_nf (β := β) (fexp := fexp) (rnd := rnd) hx
  have hone := abs_toSpec_one_sub_one_le (β := β) (fexp := fexp) (rnd := rnd)
  have hadd := approx_add_nf (β := β) (fexp := fexp) (rnd := rnd) hone hexp
  simpa only [expRatioSigmoidDenomR, expRatioSigmoidDenomError] using hadd

/-- The exponential-ratio sequence approximates the real sigmoid, with no sign or format
restriction beyond the NF rounding model. -/
theorem approx_exp_ratio_sigmoid_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (expRatioSigmoidR xR) -
        Activation.Math.sigmoidSpec (α := ℝ) x) ≤
      expRatioSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  have hnum := approx_exp_nf (β := β) (fexp := fexp) (rnd := rnd) hx
  have hden := approx_exp_ratio_sigmoid_denom_nf (β := β) (fexp := fexp) (rnd := rnd) hx
  have hy : (1 : ℝ) ≤ 1 + Real.exp x := by linarith [Real.exp_pos x]
  unfold expRatioSigmoidBoundScalar
  split_ifs with hcert
  · rw [Proofs.sigmoid_eq_exp_div]
    exact approx_div_nf_of_pos_lb (β := β) (fexp := fexp) (rnd := rnd)
      (η := 1) hy hcert hnum hden
  · have hpos : 0 < Activation.Math.sigmoidSpec (α := ℝ) x := by
      rw [Proofs.sigmoid_eq_exp_div]
      positivity
    have hle : Activation.Math.sigmoidSpec (α := ℝ) x ≤ 1 := by
      rw [Proofs.sigmoid_eq_exp_div]
      exact div_le_one_of_le₀ (by linarith) (by positivity)
    calc
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (expRatioSigmoidR xR) -
          Activation.Math.sigmoidSpec (α := ℝ) x)
          ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (expRatioSigmoidR xR)) +
            abs (Activation.Math.sigmoidSpec (α := ℝ) x) := abs_sub _ _
      _ ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (expRatioSigmoidR xR)) + 1 := by
          rw [abs_of_pos hpos]
          exact add_le_add le_rfl hle

/-- NF error budget for the public sigmoid, selecting the certificate for its actual evaluation
branch. The comparison is made on `xR`, exactly as it is in `Activation.Math.sigmoidSpec`. -/
def sigmoidBoundScalar (eps : ℝ) (xR : R) : ℝ :=
  if xR > 0 then
    reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR
  else
    expRatioSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR

/-- Per-entry forward-error budgets for the public sigmoid. -/
def sigmoidBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  TorchLean.Tensor.map (sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps) xR

/-- Scalar approximation certificate for the branch-stable sigmoid.

Either rounded branch approximates the same real function, so the proof does not assume that
the approximate input and the exact input have the same sign. -/
theorem approx_sigmoid_nf {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ eps) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
          (Activation.Math.sigmoidSpec (α := R) xR) -
        Activation.Math.sigmoidSpec (α := ℝ) x) ≤
      sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  by_cases hpos : xR > 0
  · simpa only [Activation.Math.sigmoidSpec, if_pos hpos, sigmoidBoundScalar,
      reciprocalSigmoidR] using
      (approx_reciprocal_sigmoid_nf (β := β) (fexp := fexp) (rnd := rnd) hx)
  · simpa only [Activation.Math.sigmoidSpec, if_neg hpos, sigmoidBoundScalar,
      expRatioSigmoidR] using
      (approx_exp_ratio_sigmoid_nf (β := β) (fexp := fexp) (rnd := rnd) hx)

/-- Shape-generic approximation certificate for the public sigmoid. Its per-entry budgets cover
both selected evaluation branches and use the range fallback only if the corresponding
denominator certificate fails. -/
theorem approxTensor_sigmoid_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) xS)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) xR)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_runtime_scalar_bound
      (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s)
      (fS := Activation.Math.sigmoidSpec (α := ℝ))
      (fR := Activation.Math.sigmoidSpec (α := R))
      (bnd := fun xR inputError =>
        sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) inputError xR)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hxScalar
        exact approx_sigmoid_nf (β := β) (fexp := fexp) (rnd := rnd) hxScalar)
  simpa [sigmoidBoundTensor] using h

end NFBackend

end

end RuntimeApprox
end Proofs
