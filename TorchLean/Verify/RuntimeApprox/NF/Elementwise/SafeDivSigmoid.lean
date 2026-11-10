/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Elementwise.Softplus

/-!
# NF Elementwise Bounds: Safe Division and Sigmoid

Forward error bounds for clamped division `safeDiv`, ordinary division on a certified positive
denominator domain (`divPosErrorBound`), and the elementwise sigmoid.

The division budget `divPosErrorBound η epsx epsy xhat yhat` applies when `η ≤ y`, `|ŷ - y| ≤ epsy`
and `epsy < η`, and its conditioning factor is `1 / (η - epsy)`.

Sigmoid evaluates `1 / (1 + exp (-x))` on positive inputs and `exp x / (1 + exp x)` on the others.
Each sequence has its own numerator and denominator budget, assembled from the operations it
actually evaluates. The rounded input selects the branch and both real expressions equal the same
logistic function, so the input approximation may cross zero without a separate sign hypothesis.
When a denominator budget is below `1` the division certificate applies, and otherwise the bound is
`|σ̂| + 1`, from the range of the exact sigmoid.

The declarations prefixed `reciprocalSigmoid` and `expRatioSigmoid` describe the two sequences on
every input, and `reciprocalSigmoid` is the tensor bound for the first. Under
small-error hypotheses, `reciprocalSigmoidBoundScalar_le_one` bounds the first budget by `1`, and
`expRatioSigmoidBoundScalar_le_one` bounds the second by `1` when the rounded exponential is at
most `1`. `sigmoidBoundScalar` and `ApproxTensor.sigmoid` describe the branch-selected public
operation.
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

-- ---------------------------------------------------------------------------
-- Safe division (clamped): `x / max y ε`
-- ---------------------------------------------------------------------------

/-- Spec-side safe division with a clamped denominator. -/
def safeDiv (ε : ℝ) (x y : ℝ) : ℝ :=
  x / max y ε

/-- Rounded-real model of clamped division, with one rounding after the exact quotient. The clamp
threshold `ε` enters as an exact real parameter. -/
def safeDivR (ε : ℝ) (xR yR : R) : R :=
  NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (safeDiv (ε := ε)
      (NF.toReal xR)
      (NF.toReal yR))

/--
Forward approximation bound for `safeDiv` in `NF`.

For `0 < ε`, `safeDiv ε x y = x / max y ε` keeps the denominator at least `ε`, and the bound needs
no further hypothesis. It has a `(1/ε)` term for the numerator error, a `(1/ε^2)` term for the
denominator error and half an ULP for the rounding.
-/
theorem approx_safeDiv {x y : ℝ} {xR yR : R} {epsx epsy ε : ℝ}
    (hε : 0 < ε)
    (hx : abs (NF.toReal xR - x) ≤ epsx)
    (hy : abs (NF.toReal yR - y) ≤ epsy) :
    abs
        (NF.toReal
            (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
          safeDiv (ε := ε) x y) ≤
      (1 / ε) * epsx +
        (abs (NF.toReal xR) + epsx) * (epsy / (ε * ε)) +
        ulp β fexp
            (safeDiv (ε := ε)
              (NF.toReal xR)
              (NF.toReal yR)) / 2 := by
  set xhat : ℝ := NF.toReal xR
  set yhat : ℝ := NF.toReal yR
  set uhat : ℝ := max yhat ε
  set u : ℝ := max y ε

  have uhat_ge : ε ≤ uhat := le_max_right _ _
  have u_ge : ε ≤ u := le_max_right _ _
  have uhat_pos : 0 < uhat := lt_of_lt_of_le hε uhat_ge
  have u_pos : 0 < u := lt_of_lt_of_le hε u_ge

  have hx' : abs (xhat - x) ≤ epsx := hx
  have hy' : abs (yhat - y) ≤ epsy := hy

  -- Reverse triangle inequality: `|x| - |x̂| ≤ |x̂ - x| ≤ epsx`.
  have hx_abs : abs x ≤ abs xhat + epsx := by
    have h := abs_sub_abs_le_abs_sub x xhat
    rw [abs_sub_comm] at h
    linarith

  -- `max · ε` is 1-Lipschitz.
  have hmax : abs (uhat - u) ≤ epsy :=
    le_trans (abs_max_sub_max_le_abs yhat y ε) hy'

  have hround :
      abs
          (NF.toReal
              (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
            safeDiv (ε := ε) xhat yhat) ≤
        ulp β fexp (safeDiv (ε := ε) xhat yhat) / 2 := by
    simpa [safeDivR, safeDiv, xhat, yhat, NF.toReal,
      NF.ofReal,
      NF.roundR] using
        (Flocq.error_bound_ulp (β := β) (fexp := fexp) rnd
          (safeDiv (ε := ε) xhat yhat))

  have hdiff :
      abs (safeDiv (ε := ε) xhat yhat - safeDiv (ε := ε) x y) ≤
        (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) := by
    -- Split numerator and denominator effects.
    have hsplit :
        abs (xhat / uhat - x / u) ≤ abs (xhat / uhat - x / uhat) + abs (x / uhat - x / u) :=
      abs_sub_le _ _ _

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
        have hprod_ge : (ε * ε) ≤ uhat * u :=
          mul_le_mul uhat_ge u_ge hε.le uhat_pos.le
        have hprod_pos : 0 < uhat * u := mul_pos uhat_pos u_pos
        have hprod_inv :
            (1 : ℝ) / (uhat * u) ≤ (1 : ℝ) / (ε * ε) := by
          simpa [one_div] using (one_div_le_one_div_of_le (mul_pos hε hε) hprod_ge)
        have hprod_inv_nonneg : 0 ≤ (1 : ℝ) / (uhat * u) := by
          exact le_of_lt (one_div_pos.2 hprod_pos)
        calc
          abs ((1 : ℝ) / uhat - (1 : ℝ) / u)
              = abs ((u - uhat) / (uhat * u)) := by
                  simpa using congrArg _root_.abs hiden
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

    simpa [safeDiv, uhat, u] using le_trans hsplit (add_le_add hnum hden)

  have :=
    calc
      abs
          (NF.toReal
              (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
            safeDiv (ε := ε) x y)
          ≤
        abs
            (NF.toReal
                (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
              safeDiv (ε := ε) xhat yhat) +
          abs (safeDiv (ε := ε) xhat yhat - safeDiv (ε := ε) x y) := abs_sub_le _ _ _
      _ ≤
        ulp β fexp (safeDiv (ε := ε) xhat yhat) / 2 +
          ((1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε))) := add_le_add hround hdiff
      _ = (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) +
        ulp β fexp (safeDiv (ε := ε) xhat yhat) / 2 := by
            ring

  simpa [xhat, yhat] using this

/-- Error budget for division with exact denominator lower bound `η` and denominator approximation
error `epsy`. Callers establish `epsy < η` separately, since without it the rounded denominator may
cross zero and no finite perturbation bound follows.
-/
def divPosErrorBound (η epsx epsy xhat yhat : ℝ) : ℝ :=
  (1 / (η - epsy)) * epsx +
    (abs xhat + epsx) * (epsy / ((η - epsy) * (η - epsy))) +
    ulp β fexp (xhat / yhat) / 2

/-- Forward error for ordinary division when the exact denominator stays positively separated from
zero and its approximation budget is smaller than that separation.

The effective runtime margin is `η - epsy`, since `η ≤ y` and `|ŷ - y| ≤ epsy` give `η - epsy ≤ ŷ`.
The proof goes through the clamped-division bound `approx_safeDiv` at clamp level `η - epsy`,
after showing that neither the exact nor the rounded denominator activates the clamp. Stable softmax
and normalization need this form, because a mathematical lower bound on a reduction has to survive
rounding before the division bound applies.
-/
theorem approx_div_of_pos_lb {x y : ℝ} {xR yR : R} {epsx epsy η : ℝ}
    (hyLower : η ≤ y) (hbudget : epsy < η)
    (hx : abs (NF.toReal xR - x) ≤ epsx)
    (hy : abs (NF.toReal yR - y) ≤ epsy) :
    abs (NF.toReal (xR / yR) - x / y) ≤
      divPosErrorBound (β := β) (fexp := fexp) η epsx epsy
        (NF.toReal xR)
        (NF.toReal yR) := by
  let margin : ℝ := η - epsy
  let yhat : ℝ := NF.toReal yR
  have hepsy : 0 ≤ epsy :=
    le_trans (abs_nonneg (NF.toReal yR - y)) hy
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
  have hmaxHat : max (NF.toReal yR) margin =
      NF.toReal yR :=
    max_eq_left (by simpa [yhat] using hyhatLower)
  have hmaxReal : max y margin = y := max_eq_left hyMargin
  have hruntime :
      safeDivR (β := β) (fexp := fexp) (rnd := rnd) margin xR yR = xR / yR := by
    simp only [safeDivR, safeDiv, hmaxHat]
    rfl
  have hspec : safeDiv (ε := margin) x y = x / y := by
    simp [safeDiv, hmaxReal]
  have hsafe :=
    approx_safeDiv (β := β) (fexp := fexp) (rnd := rnd)
      (x := x) (y := y) (xR := xR) (yR := yR)
      (epsx := epsx) (epsy := epsy) (ε := margin) hmargin hx hy
  rw [hruntime, hspec] at hsafe
  simpa [safeDiv, hmaxHat, margin, divPosErrorBound] using hsafe

/-- Under the half-margin certificate `epsy ≤ η / 2`, the division budget is controlled by the
numerator error, the denominator error, and one output rounding, with constants that do not depend
on the format. The sigmoid bounds use it. -/
theorem divPosErrorBound_le {η epsx epsy xhat yhat : ℝ}
    (hη : 0 < η) (hepsx : 0 ≤ epsx) (hepsy : 0 ≤ epsy) (hhalf : epsy ≤ η / 2) :
    divPosErrorBound (β := β) (fexp := fexp) η epsx epsy xhat yhat ≤
      (2 / η) * epsx + (abs xhat + epsx) * (4 * epsy / (η * η)) +
        ulp β fexp (xhat / yhat) / 2 := by
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

Each entry is the bound of `approx_safeDiv` at the real values of corresponding entries, with the
denominator clamped below at `ε`.
-/
def safeDivBoundTensor {s : Shape} (ε epsx epsy : ℝ) (xR yR : Tensor R s) : Tensor ℝ s :=
  zipWith
    (fun a b =>
      (1 / ε) * epsx +
        (|a| + epsx) * (epsy / (ε * ε)) +
        ulp β fexp (safeDiv (ε := ε) a b) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) yR)

/-- Per-entry budget for ordinary division on a certified positive denominator domain.

Each entry is `divPosErrorBound η epsx epsy` at the real values of corresponding entries. The
division itself is unclamped, so `ApproxTensor.div_of_pos_lb` requires `epsy < η`: every exact
denominator is at least `η`, and every runtime denominator is then at least `η - epsy > 0`.
-/
def divPosBoundTensor {s : Shape} (η epsx epsy : ℝ)
    (xR yR : Tensor R s) : Tensor ℝ s :=
  zipWith
    (divPosErrorBound (β := β) (fexp := fexp) η epsx epsy)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) yR)

/-- Shape-generic forward error for ordinary elementwise division by positive denominators.

The domain condition is stated over the exact tensor, and `epsy < η` keeps every runtime denominator
separated from zero. Axis softmax, normalization and the AdamW update divide through this rule,
which holds at every shape, so none of them needs a rank-specific theorem.
-/
theorem _root_.TorchLean.ApproxTensor.div_of_pos_lb {s : Shape} (η : ℝ) :
    ∀ {xS yS : Tensor ℝ s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR epsx →
      ApproxTensor (α := R) (toSpec := NF.toReal) yS yR epsy →
      Tensor.Forall (fun y : ℝ => η ≤ y) yS →
      epsy < η →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (xS / yS) (xR / yR)
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
        (toSpec := NF.toReal)).mp hx
      have hy' := (approxTensor_scalar_iff (α := R)
        (toSpec := NF.toReal)).mp hy
      have hdiv := approx_div_of_pos_lb
        (β := β) (fexp := fexp) (rnd := rnd)
        (by simpa [Tensor.Forall] using hdom) hmargin hx' hy'
      apply (approxTensor_scalar_iff (α := R)
        (toSpec := NF.toReal)).mpr
      change
        abs
            (NF.toReal (xR.item / yR.item) -
              xS.item / yS.item) ≤
          abs
            (divPosErrorBound (β := β) (fexp := fexp) η epsx epsy
              (NF.toReal xR.item)
              (NF.toReal yR.item))
      exact le_trans hdiv (le_abs_self _)
  | dim n inner ih =>
      intro xS yS xR yR epsx epsy hx hy hdom hmargin
      let bound := linfNorm
        (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := .dim n inner) η epsx epsy xR yR)
      have hbound : 0 ≤ bound := linfNorm_nonneg _
      refine approxTensor_dim_of_forall
        (toSpec := NF.toReal)
        (xS := xS / yS) (xR := xR / yR)
        (eps := bound) hbound ?_
      intro i
      have hxI := ApproxTensor.unstack (α := R)
        (toSpec := NF.toReal) hx i
      have hyI := ApproxTensor.unstack (α := R)
        (toSpec := NF.toReal) hy i
      have hlocal := ih hxI hyI (by simpa using hdom i) hmargin
      have hle :
          linfNorm
              (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := inner) η epsx epsy (xR.unstack i) (yR.unstack i)) ≤ bound := by
        have h := linfNorm_unstack_le
          (t := divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := .dim n inner) η epsx epsy xR yR) i
        simpa [bound, divPosBoundTensor, Tensor.toReal,
          ← Tensor.unstack_map,
          ← Tensor.unstack_zipWith] using h
      have hmono := ApproxTensor.mono hlocal hle
      simpa [div_def,
        ← Tensor.unstack_zipWith] using hmono

/--
`ApproxTensor` bound for `safeDiv ε` at arbitrary tensor shapes, where `0 < ε`.

The scalar bound `approx_safeDiv` lifts through `ApproxTensor.zipWith`, and the
output epsilon is the `linfNorm` of `safeDivBoundTensor`.
-/
theorem _root_.TorchLean.ApproxTensor.safeDiv {s : Shape} (ε : ℝ) (hε : 0 < ε) :
    ∀ {xS yS : Tensor ℝ s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR epsx →
      ApproxTensor (α := R) (toSpec := NF.toReal) yS yR epsy →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (zipWith (s := s) (safeDiv (ε := ε)) xS yS)
          (zipWith (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε) xR yR)
          (linfNorm
            (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε epsx epsy
            xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    ApproxTensor.zipWith (α := R)
      (toSpec := NF.toReal)
      (s := s)
      (fS := NF.safeDiv (ε := ε))
      (fR := safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
      (bnd := fun a b epsx epsy =>
        (1 / ε) * epsx +
          (abs a + epsx) * (epsy / (ε * ε)) +
          ulp β fexp (NF.safeDiv (ε := ε) a b) / 2)
      (xS := xS) (yS := yS) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy) hx hy (by
        intro x y xR yR hx hy
        simpa using
          (approx_safeDiv (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (y := y) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy) (ε := ε) hε hx
              hy))
  simpa [safeDivBoundTensor] using h

-- Sigmoid (elementwise logistic) bounds.

/-- `expErrorBound` is nonnegative whenever the propagated input error is. -/
theorem expErrorBound_nonneg (a : ℝ) {eps : ℝ} (heps : 0 ≤ eps) :
    0 ≤ expErrorBound (β := β) (fexp := fexp) a eps := by
  unfold expErrorBound
  have h1 : 0 ≤ Real.exp (a + eps) * eps := mul_nonneg (Real.exp_pos _).le heps
  have h2 := ulp.nonneg β fexp (Real.exp a)
  linarith

/-- Sigmoid evaluated as `1 / (1 + exp (-x))` for every input. On `NF`,
`Context.sigmoid xR` is `reciprocalSigmoidR xR` when `0 < xR` and
`expRatioSigmoidR xR` otherwise. -/
def reciprocalSigmoidR (xR : R) : R :=
  (1 : R) / (1 + Numerics.MathFunctions.exp (-xR))

/-- Rounded denominator of `reciprocalSigmoidR`. -/
def reciprocalSigmoidDenomR (xR : R) : R :=
  (1 : R) + Numerics.MathFunctions.exp (-xR)

/-- Error budget of the rounded sigmoid denominator `1 + exp(-x)` given input error `eps`.

The summands pay for the rounded constant `1`, the rounded exponential of the rounded negation
(whose input error is `eps` plus one negation rounding), and the final addition rounding. -/
def reciprocalSigmoidDenomError (eps : ℝ) (xR : R) : ℝ :=
  oneEps (β := β) (fexp := fexp) +
    expErrorBound (β := β) (fexp := fexp)
      (NF.toReal (-xR))
      (eps + ulp β fexp (-NF.toReal xR) / 2) +
    ulp β fexp
      (NF.toReal (1 : R) +
        NF.toReal (Numerics.MathFunctions.exp (-xR))) / 2

omit [ValidRndToNearest rnd] in
/-- `reciprocalSigmoidDenomError` is nonnegative for nonnegative input error. -/
theorem reciprocalSigmoidDenomError_nonneg {eps : ℝ} (xR : R) (heps : 0 ≤ eps) :
    0 ≤ reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  unfold reciprocalSigmoidDenomError
  have h1 := oneEps_nonneg (β := β) (fexp := fexp)
  have hulp0 : 0 ≤ ulp β fexp (-NF.toReal xR) / 2 := by
    have := ulp.nonneg β fexp (-NF.toReal xR)
    linarith
  have h2 := expErrorBound_nonneg (β := β) (fexp := fexp)
    (NF.toReal (-xR)) (add_nonneg heps hulp0)
  have h3 := ulp.nonneg β fexp
    (NF.toReal (1 : R) +
      NF.toReal (Numerics.MathFunctions.exp (-xR)))
  linarith

/--
Scalar forward bound for `reciprocalSigmoidR` at input error `eps`.

`sigmoid(x) = 1 / (1 + exp(-x))` has exact denominator at least `1`. When the rounded denominator
budget `reciprocalSigmoidDenomError eps xR` is below `1`, the rounded denominator stays positive and
the bound is `divPosErrorBound` with margin `1 - reciprocalSigmoidDenomError`. Otherwise the bound
falls back to `|σ̂| + 1`, which holds because the exact sigmoid lies in `(0, 1)`.
-/
def reciprocalSigmoidBoundScalar (eps : ℝ) (xR : R) : ℝ :=
  if reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1 then
    divPosErrorBound (β := β) (fexp := fexp) 1 (oneEps (β := β) (fexp := fexp))
      (reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR)
      (NF.toReal (1 : R))
      (NF.toReal (reciprocalSigmoidDenomR xR))
  else
    abs (NF.toReal
      (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR)) + 1

/-- The reciprocal sequence's denominator approximates `1 + exp(-x)` within its composed budget. -/
theorem approx_reciprocalSigmoidDenomR {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    _root_.abs (NF.toReal (reciprocalSigmoidDenomR xR) -
        (1 + Real.exp (-x))) ≤
      reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  have hneg := approx_neg (β := β) (fexp := fexp) (rnd := rnd) hx
  have hexp := approx_exp (β := β) (fexp := fexp) (rnd := rnd) hneg
  have hone := abs_toReal_one_sub_one_le (β := β) (fexp := fexp) (rnd := rnd)
  have hadd := approx_add (β := β) (fexp := fexp) (rnd := rnd) hone hexp
  simpa only [reciprocalSigmoidDenomR, reciprocalSigmoidDenomError] using hadd

/-- Scalar certificate for the reciprocal sequence, the `xR > 0` branch of `approx_sigmoid`. -/
private theorem approx_reciprocalSigmoidR {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs (NF.toReal
          (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR) -
        Context.sigmoid (α := ℝ) x) ≤
      reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  have hden := approx_reciprocalSigmoidDenomR (β := β) (fexp := fexp) (rnd := rnd) hx
  have hone := abs_toReal_one_sub_one_le (β := β) (fexp := fexp) (rnd := rnd)
  have hy : (1 : ℝ) ≤ 1 + Real.exp (-x) := by linarith [Real.exp_pos (-x)]
  have hspecR :
      reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR =
        (1 : R) / reciprocalSigmoidDenomR xR := rfl
  have hspec : Context.sigmoid (α := ℝ) x = 1 / (1 + Real.exp (-x)) := by
    rw [Context.sigmoid_eq_inv_exp, one_div]
  unfold reciprocalSigmoidBoundScalar
  split_ifs with hcert
  · rw [hspecR, hspec]
    exact approx_div_of_pos_lb (β := β) (fexp := fexp) (rnd := rnd) (η := 1) hy hcert hone hden
  · have hnonneg := Context.sigmoid_nonneg x
    have hle := Context.sigmoid_le_one x
    calc
      abs (NF.toReal
            (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR) -
          Context.sigmoid (α := ℝ) x)
          ≤ abs (NF.toReal
              (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR)) +
            abs (Context.sigmoid (α := ℝ) x) := abs_sub _ _
      _ ≤ abs (NF.toReal
              (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) xR)) + 1 := by
            rw [abs_of_nonneg hnonneg]
            linarith

/--
`ApproxTensor` bound for the reciprocal sigmoid sequence at arbitrary tensor shapes, with error at
most the `linfNorm` of the per-entry budgets `reciprocalSigmoidBoundScalar eps`. There is no
side condition, because each budget switches to its fallback when the denominator certificate fails.
-/
theorem _root_.TorchLean.ApproxTensor.reciprocalSigmoid {s : Shape} :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.map (shape := s) (Context.sigmoid (α := ℝ)) xS)
          (Tensor.map (shape := s) (reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd)) xR)
          (linfNorm (Tensor.map
            (reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps) xR)) := by
  intro xS xR eps hx
  have h :=
    ApproxTensor.map_of_runtime_bound
      (α := R) (toSpec := NF.toReal)
      (s := s)
      (fS := Context.sigmoid (α := ℝ))
      (fR := reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd))
      (bnd := fun xR inputError =>
        reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) inputError xR)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hxScalar
        exact approx_reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) hxScalar)
  simpa using h

omit [ValidRndToNearest rnd] in
/-- With denominator error at most `1/2`, the reciprocal sequence's bound is controlled by the
constant-one budget, the denominator error, and one output rounding. The denominator error includes
the input error `eps`, which a finer format does not reduce. -/
theorem reciprocalSigmoidBoundScalar_le {eps : ℝ} (xR : R) (heps : 0 ≤ eps)
    (hden : reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 / 2) :
    reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤
      2 * oneEps (β := β) (fexp := fexp) +
        (abs (NF.toReal (1 : R)) +
            oneEps (β := β) (fexp := fexp)) *
          (4 * reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR) +
        ulp β fexp
          (NF.toReal (1 : R) /
            NF.toReal (reciprocalSigmoidDenomR xR)) / 2 := by
  have hden0 := reciprocalSigmoidDenomError_nonneg (β := β) (fexp := fexp) (rnd := rnd) xR heps
  have hlt : reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1 := by
    linarith
  unfold reciprocalSigmoidBoundScalar
  rw [ite_eq_left hlt]
  have h := divPosErrorBound_le (β := β) (fexp := fexp) (η := 1)
    (xhat := NF.toReal (1 : R))
    (yhat := NF.toReal (reciprocalSigmoidDenomR xR))
    one_pos (oneEps_nonneg (β := β) (fexp := fexp)) hden0 (by linarith)
  exact h.trans (le_of_eq (by ring))

/-- When the half ULP `oneEps` at `1` and the denominator error are at most `1/16` and the output
half ULP is at most `1/4`, the reciprocal sequence's bound is at most `1`. -/
theorem reciprocalSigmoidBoundScalar_le_one {eps : ℝ} (xR : R) (heps : 0 ≤ eps)
    (hone : oneEps (β := β) (fexp := fexp) ≤ 1 / 16)
    (hden : reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 / 16)
    (hulp : ulp β fexp
      (NF.toReal (1 : R) /
        NF.toReal (reciprocalSigmoidDenomR xR)) ≤ 1 / 2) :
    reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 := by
  have h := reciprocalSigmoidBoundScalar_le (β := β) (fexp := fexp) (rnd := rnd)
    xR heps (by linarith)
  have hden0 := reciprocalSigmoidDenomError_nonneg (β := β) (fexp := fexp) (rnd := rnd) xR heps
  have hone0 := oneEps_nonneg (β := β) (fexp := fexp)
  have habs : abs (NF.toReal (1 : R)) ≤
      1 + oneEps (β := β) (fexp := fexp) := by
    have h1 := abs_toReal_one_sub_one_le (β := β) (fexp := fexp) (rnd := rnd)
    calc
      abs (NF.toReal (1 : R))
          = |(NF.toReal (1 : R) - 1) + 1| := by ring_nf
      _ ≤ |NF.toReal (1 : R) - 1| + abs (1 : ℝ) :=
          abs_add_le _ _
      _ ≤ 1 + oneEps (β := β) (fexp := fexp) := by
          rw [abs_one]
          linarith
  have hprod :
      (abs (NF.toReal (1 : R)) +
          oneEps (β := β) (fexp := fexp)) *
        (4 * reciprocalSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR) ≤
      (1 + 1 / 16 + 1 / 16) * (4 * (1 / 16)) :=
    mul_le_mul (by linarith) (by linarith) (by positivity) (by norm_num)
  linarith

/-- The sequence `exp x / (1 + exp x)` that the public sigmoid evaluates on nonpositive inputs, with
one rounded exponential shared by numerator and denominator. It is defined on every NF input, so its
certificate does not depend on the comparison that selects the public branch. -/
def expRatioSigmoidR (xR : R) : R :=
  let z := Numerics.MathFunctions.exp xR
  z / (1 + z)

/-- Rounded denominator of `expRatioSigmoidR`. -/
def expRatioSigmoidDenomR (xR : R) : R :=
  (1 : R) + Numerics.MathFunctions.exp xR

/-- Error budget of the exponential-ratio denominator `1 + exp x`: the exponential at input error
`eps`, the rounded constant `1` and the rounding of their sum. -/
def expRatioSigmoidDenomError (eps : ℝ) (xR : R) : ℝ :=
  oneEps (β := β) (fexp := fexp) +
    expErrorBound (β := β) (fexp := fexp)
      (NF.toReal xR) eps +
    ulp β fexp
      (NF.toReal (1 : R) +
        NF.toReal (Numerics.MathFunctions.exp xR)) / 2

/-- Forward-error budget for the exponential-ratio sequence. Both occurrences of `exp x` share one
rounded value, although the division estimate treats the numerator and denominator errors
separately. The exact denominator is at least `1`, and when the denominator error reaches `1` the
budget falls back to `|σ̂| + 1`. -/
def expRatioSigmoidBoundScalar (eps : ℝ) (xR : R) : ℝ :=
  if expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1 then
    divPosErrorBound (β := β) (fexp := fexp) 1
      (expErrorBound (β := β) (fexp := fexp)
        (NF.toReal xR) eps)
      (expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR)
      (NF.toReal (Numerics.MathFunctions.exp xR))
      (NF.toReal (expRatioSigmoidDenomR xR))
  else
    abs (NF.toReal
      (expRatioSigmoidR xR)) + 1

/-- The exponential-ratio denominator approximates `1 + exp x` within its composed budget. -/
theorem approx_expRatioSigmoidDenomR {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    _root_.abs (NF.toReal (expRatioSigmoidDenomR xR) -
        (1 + Real.exp x)) ≤
      expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  have hexp := approx_exp (β := β) (fexp := fexp) (rnd := rnd) hx
  have hone := abs_toReal_one_sub_one_le (β := β) (fexp := fexp) (rnd := rnd)
  have hadd := approx_add (β := β) (fexp := fexp) (rnd := rnd) hone hexp
  simpa only [expRatioSigmoidDenomR, expRatioSigmoidDenomError] using hadd

/-- The exponential-ratio sequence approximates the real sigmoid, with no sign or format
restriction beyond the NF rounding model. -/
theorem approx_expRatioSigmoidR {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs (NF.toReal (expRatioSigmoidR xR) -
        Context.sigmoid (α := ℝ) x) ≤
      expRatioSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  have hnum := approx_exp (β := β) (fexp := fexp) (rnd := rnd) hx
  have hden := approx_expRatioSigmoidDenomR (β := β) (fexp := fexp) (rnd := rnd) hx
  have hy : (1 : ℝ) ≤ 1 + Real.exp x := by linarith [Real.exp_pos x]
  unfold expRatioSigmoidBoundScalar
  split_ifs with hcert
  · rw [Context.sigmoid_eq_exp_div]
    exact approx_div_of_pos_lb (β := β) (fexp := fexp) (rnd := rnd)
      (η := 1) hy hcert hnum hden
  · have hnonneg := Context.sigmoid_nonneg x
    have hle := Context.sigmoid_le_one x
    calc
      abs (NF.toReal (expRatioSigmoidR xR) -
          Context.sigmoid (α := ℝ) x)
          ≤ abs (NF.toReal (expRatioSigmoidR xR)) +
            abs (Context.sigmoid (α := ℝ) x) := abs_sub _ _
      _ ≤ abs (NF.toReal (expRatioSigmoidR xR)) + 1 := by
          rw [abs_of_nonneg hnonneg]
          exact add_le_add le_rfl hle

omit [ValidRndToNearest rnd] in
/-- `expRatioSigmoidDenomError` is nonnegative for nonnegative input error. -/
theorem expRatioSigmoidDenomError_nonneg {eps : ℝ} (xR : R) (heps : 0 ≤ eps) :
    0 ≤ expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  unfold expRatioSigmoidDenomError
  have h1 := oneEps_nonneg (β := β) (fexp := fexp)
  have h2 := expErrorBound_nonneg (β := β) (fexp := fexp)
    (NF.toReal xR) heps
  have h3 := ulp.nonneg β fexp
    (NF.toReal (1 : R) +
      NF.toReal (Numerics.MathFunctions.exp xR))
  linarith

omit [ValidRndToNearest rnd] in
/-- With denominator error at most `1/2`, the exponential-ratio sequence's bound is linear in the
exponential's budget, the denominator error and one output rounding. -/
theorem expRatioSigmoidBoundScalar_le {eps : ℝ} (xR : R) (heps : 0 ≤ eps)
    (hden : expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 / 2) :
    expRatioSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤
      2 * expErrorBound (β := β) (fexp := fexp)
          (NF.toReal xR) eps +
        (abs (NF.toReal (Numerics.MathFunctions.exp xR)) +
            expErrorBound (β := β) (fexp := fexp)
              (NF.toReal xR) eps) *
          (4 * expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR) +
        ulp β fexp
          (NF.toReal (Numerics.MathFunctions.exp xR) /
            NF.toReal (expRatioSigmoidDenomR xR)) / 2 := by
  have hnum0 := expErrorBound_nonneg (β := β) (fexp := fexp)
    (NF.toReal xR) heps
  have hden0 := expRatioSigmoidDenomError_nonneg (β := β) (fexp := fexp) (rnd := rnd) xR heps
  have hlt : expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1 := by
    linarith
  unfold expRatioSigmoidBoundScalar
  rw [ite_eq_left hlt]
  have h := divPosErrorBound_le (β := β) (fexp := fexp) (η := 1)
    (xhat := NF.toReal (Numerics.MathFunctions.exp xR))
    (yhat := NF.toReal (expRatioSigmoidDenomR xR))
    one_pos hnum0 hden0 (by linarith)
  exact h.trans (le_of_eq (by ring))

omit [ValidRndToNearest rnd] in
/-- When the rounded exponential `exp(x̂)` is at most `1` (the nonpositive half line), its budget
and the denominator error are at most `1/16`, and the output half ulp is at most `1/4`, the
exponential-ratio sequence's bound is at most `1`. -/
theorem expRatioSigmoidBoundScalar_le_one {eps : ℝ} (xR : R) (heps : 0 ≤ eps)
    (hnumHat :
      abs (NF.toReal (Numerics.MathFunctions.exp xR)) ≤ 1)
    (hnum : expErrorBound (β := β) (fexp := fexp)
      (NF.toReal xR) eps ≤ 1 / 16)
    (hden : expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 / 16)
    (hulp : ulp β fexp
      (NF.toReal (Numerics.MathFunctions.exp xR) /
        NF.toReal (expRatioSigmoidDenomR xR)) ≤ 1 / 2) :
    expRatioSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR ≤ 1 := by
  have h := expRatioSigmoidBoundScalar_le (β := β) (fexp := fexp)
    (rnd := rnd) xR heps (by linarith)
  have hnum0 := expErrorBound_nonneg (β := β) (fexp := fexp)
    (NF.toReal xR) heps
  have hden0 := expRatioSigmoidDenomError_nonneg (β := β) (fexp := fexp) (rnd := rnd) xR heps
  have hprod :
      (abs (NF.toReal (Numerics.MathFunctions.exp xR)) +
          expErrorBound (β := β) (fexp := fexp)
            (NF.toReal xR) eps) *
        (4 * expRatioSigmoidDenomError (β := β) (fexp := fexp) (rnd := rnd) eps xR) ≤
      (1 + 1 / 16) * (4 * (1 / 16)) :=
    mul_le_mul (by linarith) (by linarith) (by positivity) (by norm_num)
  linarith

/-- NF error budget for the public sigmoid, taking the certificate of the branch it evaluates. It
branches on `xR > 0`, the test `Context.sigmoid` makes at the `NF` backend. -/
def sigmoidBoundScalar (eps : ℝ) (xR : R) : ℝ :=
  if xR > 0 then
    reciprocalSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR
  else
    expRatioSigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR

/-- Per-entry forward-error budgets for the public sigmoid. -/
def sigmoidBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map (sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps) xR

/-- Scalar approximation certificate for the branch-stable sigmoid.

Either rounded branch approximates the same real function, so the proof does not assume that
the approximate input and the exact input have the same sign. -/
theorem approx_sigmoid {x : ℝ} {xR : R} {eps : ℝ}
    (hx : abs (NF.toReal xR - x) ≤ eps) :
    abs (NF.toReal
          (Context.sigmoid (α := R) xR) -
        Context.sigmoid (α := ℝ) x) ≤
      sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
  have unsupported : Numeric.QuotientArithmetic.supported (α := R) = false := rfl
  have hvalue : Context.sigmoid (α := R) xR =
      if xR > 0 then reciprocalSigmoidR xR else expRatioSigmoidR xR := by
    by_cases hpos : xR > 0
    · simp [Context.sigmoid, Numeric.sigmoid,
        unsupported, hpos, reciprocalSigmoidR]
    · simp [Context.sigmoid, Numeric.sigmoid,
        unsupported, hpos, expRatioSigmoidR]
  rw [hvalue]
  by_cases hpos : xR > 0
  · simpa only [sigmoidBoundScalar, ite_eq_left hpos] using
      (approx_reciprocalSigmoidR (β := β) (fexp := fexp) (rnd := rnd) hx)
  · simpa only [sigmoidBoundScalar, ite_eq_right hpos] using
      (approx_expRatioSigmoidR (β := β) (fexp := fexp) (rnd := rnd) hx)

/-- Shape-generic approximation certificate for the public sigmoid. Its per-entry budgets cover
both selected evaluation branches and use the range fallback only if the corresponding
denominator certificate fails. -/
theorem _root_.TorchLean.ApproxTensor.sigmoid {s : Shape} :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.map (shape := s) (Context.sigmoid (α := ℝ)) xS)
          (Tensor.map (shape := s) (Context.sigmoid (α := R)) xR)
          (linfNorm
            (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    ApproxTensor.map_of_runtime_bound
      (α := R) (toSpec := NF.toReal)
      (s := s)
      (fS := Context.sigmoid (α := ℝ))
      (fR := Context.sigmoid (α := R))
      (bnd := fun xR inputError =>
        sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) inputError xR)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hxScalar
        exact approx_sigmoid (β := β) (fexp := fexp) (rnd := rnd) hxScalar)
  simpa [sigmoidBoundTensor] using h

end

end NF
end TorchLean
