/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Core.Tolerance
public import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Core
public import Mathlib.Analysis.Calculus.Deriv.Basic
public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Algebra.Order.Group.MinMax

/-!
# Scalar error propagation

`Roundoff` composes local absolute errors without choosing a scalar representation. Both the
source-level numerical automation and the rounded-real backend use these same perturbation lemmas.

The rounded-real specialization below uses `Flocq.round` and its half-ULP bound. Tensor and graph
proofs reuse these scalar facts; agreement with an external numerical backend is a separate claim.
-/

@[expose] public section

namespace Proofs.RuntimeApprox.Roundoff

noncomputable section

/-- An exact input introduces no arithmetic error. -/
theorem exact_error (x : ℝ) : |x - x| ≤ 0 := by simp

/-- Numerical interpretation and propagation of input errors are independent proof steps. -/
theorem compose_error {actual intermediate reference localError inputError : ℝ}
    (hlocal : |actual - intermediate| ≤ localError)
    (hinput : |intermediate - reference| ≤ inputError) :
    |actual - reference| ≤ inputError + localError :=
  ((abs_sub_le actual intermediate reference).trans (add_le_add hlocal hinput)).trans_eq
    (add_comm localError inputError)

/-- A Lipschitz function scales the input error before the implementation's local error is added. -/
theorem lipschitz_error {f : ℝ → ℝ} {constant : NNReal}
    {a x result inputError localError : ℝ} (hf : LipschitzWith constant f)
    (hinput : |a - x| ≤ inputError) (hlocal : |result - f a| ≤ localError) :
    |result - f x| ≤ (constant : ℝ) * inputError + localError := by
  have difference : |f a - f x| ≤ (constant : ℝ) * inputError := by
    simpa only [Real.dist_eq] using hf.dist_le_mul_of_le
      (show dist a x ≤ inputError by simpa only [Real.dist_eq] using hinput)
  exact compose_error hlocal difference

/-- A derivative bound is needed only on the interval between the two inputs. -/
theorem derivative_error {f : ℝ → ℝ}
    {a x result inputError localError derivativeBound : ℝ} (hbound : 0 ≤ derivativeBound)
    (hdifferentiable : ∀ y ∈ Set.Icc (min a x) (max a x), DifferentiableAt ℝ f y)
    (hderivative : ∀ y ∈ Set.Icc (min a x) (max a x), |deriv f y| ≤ derivativeBound)
    (hinput : |a - x| ≤ inputError) (hlocal : |result - f a| ≤ localError) :
    |result - f x| ≤ derivativeBound * inputError + localError := by
  have difference : |f a - f x| ≤ derivativeBound * |a - x| := by
    simpa only [Real.norm_eq_abs] using Convex.norm_image_sub_le_of_norm_deriv_le
      (𝕜 := ℝ) (s := Set.Icc (min a x) (max a x)) (f := f) (C := derivativeBound)
      hdifferentiable
      (fun y hy => by simpa only [Real.norm_eq_abs] using hderivative y hy)
      (convex_Icc _ _) (Set.mem_Icc.mpr ⟨min_le_right _ _, le_max_right _ _⟩)
      (Set.mem_Icc.mpr ⟨min_le_left _ _, le_max_left _ _⟩)
  exact compose_error hlocal (difference.trans (mul_le_mul_of_nonneg_left hinput hbound))

/-- Negation introduces no additional rounding. -/
theorem neg_error {actual reference error : ℝ} (h : |actual - reference| ≤ error) :
    |-actual - -reference| ≤ error := by simpa only [neg_sub_neg, abs_sub_comm] using h

/-- Absolute value does not amplify its input error, including when the sign changes. -/
theorem abs_error {actual reference error : ℝ} (h : |actual - reference| ≤ error) :
    abs (|actual| - |reference|) ≤ error :=
  (abs_abs_sub_abs_le_abs_sub actual reference).trans h

/-- A maximum propagates both input errors without requiring agreement on the winning input. -/
theorem max_error {a b x y firstError secondError : ℝ}
    (ha : |a - x| ≤ firstError) (hb : |b - y| ≤ secondError) :
    |max a b - max x y| ≤ max firstError secondError :=
  (abs_max_sub_max_le_max a b x y).trans (max_le_max ha hb)

/-- A minimum has the same perturbation bound as a maximum. -/
theorem min_error {a b x y firstError secondError : ℝ}
    (ha : |a - x| ≤ firstError) (hb : |b - y| ≤ secondError) :
    |min a b - min x y| ≤ max firstError secondError :=
  (abs_min_sub_min_le_max a b x y).trans (max_le_max ha hb)

/-- Square root propagates input error without requiring a positive lower margin. The bound
also covers a real reference below zero, where `Real.sqrt` is totalized to zero. -/
theorem sqrt_error {a x result inputError localError : ℝ}
    (hinput : |a - x| ≤ inputError)
    (hlocal : |result - Real.sqrt a| ≤ localError) :
    |result - Real.sqrt x| ≤ Real.sqrt inputError + localError := by
  have he : 0 ≤ inputError := (abs_nonneg _).trans hinput
  have root_le (u v : ℝ) (hu : u ≤ v + inputError) :
      Real.sqrt u ≤ Real.sqrt v + Real.sqrt inputError := by
    apply Real.sqrt_le_iff.mpr
    refine ⟨add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _), ?_⟩
    have hv : v ≤ Real.sqrt v ^ 2 := by
      rw [Real.sq_sqrt']
      exact le_max_left _ _
    nlinarith [Real.sq_sqrt he,
      mul_nonneg (Real.sqrt_nonneg v) (Real.sqrt_nonneg inputError)]
  have lower := root_le x a (by linarith [(abs_le.mp hinput).1])
  have upper := root_le a x (by linarith [(abs_le.mp hinput).2])
  exact compose_error hlocal (abs_le.mpr ⟨by linarith, by linarith⟩)

/-- The exponential's derivative on the interval between the inputs bounds its input error. -/
theorem exp_error {a x result inputError localError : ℝ}
    (hinput : |a - x| ≤ inputError)
    (hlocal : |result - Real.exp a| ≤ localError) :
    |result - Real.exp x| ≤ Real.exp (max a x) * inputError + localError := by
  apply derivative_error (Real.exp_nonneg _)
    (fun _ _ => Real.differentiableAt_exp) _ hinput hlocal
  intro value hvalue
  simpa only [Real.deriv_exp, abs_of_pos (Real.exp_pos value)] using
    Real.exp_le_exp.mpr hvalue.2

/-- Logarithmic input error needs a positive margin for both the computed and reference values. -/
theorem log_error {a x result inputError localError : ℝ} (ha : 0 < a) (hx : 0 < x)
    (hinput : |a - x| ≤ inputError)
    (hlocal : |result - Real.log a| ≤ localError) :
    |result - Real.log x| ≤ inputError / min a x + localError := by
  have hmin : 0 < min a x := lt_min ha hx
  have estimate : |result - Real.log x| ≤ (1 / min a x) * inputError + localError := by
    apply derivative_error (le_of_lt (one_div_pos.mpr hmin))
      (fun value hvalue =>
        Real.differentiableAt_log (lt_of_lt_of_le hmin hvalue.1).ne') _ hinput hlocal
    intro value hvalue
    have hpositive := lt_of_lt_of_le hmin hvalue.1
    simpa [Real.deriv_log, abs_of_pos hpositive, one_div] using
      one_div_le_one_div_of_le hmin hvalue.1
  simpa only [one_div, div_eq_mul_inv, mul_one, mul_comm] using
    estimate

/-- Sine does not amplify the input error. -/
theorem sin_error {a x result inputError localError : ℝ}
    (hinput : |a - x| ≤ inputError) (hlocal : |result - Real.sin a| ≤ localError) :
    |result - Real.sin x| ≤ inputError + localError := by
  simpa only [NNReal.coe_one, one_mul] using
    lipschitz_error Real.lipschitzWith_sin hinput hlocal

/-- Cosine has the same global input-error bound as sine. -/
theorem cos_error {a x result inputError localError : ℝ}
    (hinput : |a - x| ≤ inputError) (hlocal : |result - Real.cos a| ≤ localError) :
    |result - Real.cos x| ≤ inputError + localError := by
  simpa only [NNReal.coe_one, one_mul] using
    lipschitz_error Real.lipschitzWith_cos hinput hlocal

/-- Hyperbolic tangent has slope at most one on the whole real line. -/
theorem tanh_error {a x result inputError localError : ℝ}
    (hinput : |a - x| ≤ inputError) (hlocal : |result - Real.tanh a| ≤ localError) :
    |result - Real.tanh x| ≤ inputError + localError := by
  have derivative (value : ℝ) : HasDerivAt Real.tanh (1 / Real.cosh value ^ 2) value := by
    convert (Real.hasDerivAt_sinh value).div (Real.hasDerivAt_cosh value)
      (Real.cosh_pos value).ne' using 1
    · funext point
      exact Real.tanh_eq_sinh_div_cosh point
    · simp only [← pow_two, Real.cosh_sq_sub_sinh_sq]
  have bound (value : ℝ) : |deriv Real.tanh value| ≤ (1 : ℝ) := by
    rw [(derivative value).deriv,
      abs_of_nonneg (one_div_nonneg.mpr (sq_nonneg (Real.cosh value)))]
    apply (div_le_one (sq_pos_of_pos (Real.cosh_pos value))).mpr
    nlinarith [Real.one_le_cosh value, sq_nonneg (Real.cosh value - 1)]
  simpa only [one_mul] using derivative_error (f := Real.tanh) (derivativeBound := 1)
    zero_le_one (fun value _ => (derivative value).differentiableAt)
    (fun value _ => bound value) hinput hlocal

private theorem abs_le_input_radius {a x value : ℝ}
    (hvalue : value ∈ Set.Icc (min a x) (max a x)) : |value| ≤ max |a| |x| := by
  rcases le_total a x with h | h
  · simp only [min_eq_left h, max_eq_right h] at hvalue
    exact abs_le_max_abs_abs hvalue.1 hvalue.2
  · simp only [min_eq_right h, max_eq_left h] at hvalue
    exact (abs_le_max_abs_abs hvalue.1 hvalue.2).trans_eq (max_comm _ _)

private theorem cosh_le_input_radius {a x value : ℝ}
    (hvalue : value ∈ Set.Icc (min a x) (max a x)) :
    Real.cosh value ≤ Real.cosh (max |a| |x|) := by
  apply Real.cosh_le_cosh.mpr
  rw [abs_of_nonneg ((abs_nonneg a).trans (le_max_left _ _))]
  exact abs_le_input_radius hvalue

/-- Hyperbolic sine uses a derivative bound over the inputs' absolute radius. -/
theorem sinh_error {a x result inputError localError : ℝ}
    (hinput : |a - x| ≤ inputError) (hlocal : |result - Real.sinh a| ≤ localError) :
    |result - Real.sinh x| ≤ Real.cosh (max |a| |x|) * inputError + localError := by
  apply derivative_error (Real.cosh_pos _).le
    (fun _ _ => Real.differentiableAt_sinh) _ hinput hlocal
  intro value hvalue
  simpa only [Real.deriv_sinh, abs_of_pos (Real.cosh_pos value)] using
    cosh_le_input_radius hvalue

/-- Hyperbolic cosine uses the same local derivative bound as hyperbolic sine. -/
theorem cosh_error {a x result inputError localError : ℝ}
    (hinput : |a - x| ≤ inputError) (hlocal : |result - Real.cosh a| ≤ localError) :
    |result - Real.cosh x| ≤ Real.cosh (max |a| |x|) * inputError + localError := by
  apply derivative_error (Real.cosh_pos _).le
    (fun _ _ => Real.differentiableAt_cosh) _ hinput hlocal
  intro value hvalue
  have hsinh : |Real.sinh value| ≤ Real.cosh value := by
    simpa only [Real.abs_sinh, Real.cosh_abs] using (Real.sinh_lt_cosh (x := |value|)).le
  simpa only [Real.deriv_cosh] using hsinh.trans (cosh_le_input_radius hvalue)

/-- Analyze the branch the program actually selects. Conditions on an inactive branch are
irrelevant; agreement with a different, real-valued branch test must be proved separately. -/
theorem dite_error {α : Type*} (observe : α → ℝ) (condition : Prop) [Decidable condition]
    (yes : condition → α) (no : ¬condition → α)
    {yesReference yesError : condition → ℝ} {noReference noError : ¬condition → ℝ}
    (hyes : ∀ h, |observe (yes h) - yesReference h| ≤ yesError h)
    (hno : ∀ h, |observe (no h) - noReference h| ≤ noError h) :
    |observe (if h : condition then yes h else no h) -
        (if h : condition then yesReference h else noReference h)| ≤
      if h : condition then yesError h else noError h := by
  split_ifs with h
  · exact hyes h
  · exact hno h

/-- An error smaller than the reference magnitude preserves a nonzero result.
This is stronger than accepting a small fixed absolute tolerance near zero. -/
theorem nonzero_of_error_lt {actual reference error : ℝ}
    (herror : |actual - reference| ≤ error) (hsmall : error < |reference|) : actual ≠ 0 := by
  intro hzero
  subst actual
  simp only [zero_sub, abs_neg] at herror
  exact (not_lt_of_ge herror) hsmall

/-- A positive reference larger than the error budget gives a strictly positive result. -/
theorem positive_of_error_lt {actual reference error : ℝ}
    (herror : |actual - reference| ≤ error) (hsmall : error < reference) : 0 < actual := by
  have lower := (abs_le.mp herror).1
  linarith

/-- Compose input perturbations with the error of the operation actually executed. -/
theorem add_error {a b x y r ea eb er : ℝ}
    (ha : |a - x| ≤ ea) (hb : |b - y| ≤ eb)
    (hr : |r - (a + b)| ≤ er) :
    |r - (x + y)| ≤ ea + eb + er := by
  have hp : |(a + b) - (x + y)| ≤ ea + eb := by
    rw [add_sub_add_comm]
    exact (abs_add_le _ _).trans (add_le_add ha hb)
  exact ((abs_sub_le r (a + b) (x + y)).trans (add_le_add hr hp)).trans_eq
    (add_comm er (ea + eb))

/-- Subtraction retains both input errors even when its exact result is very small. -/
theorem sub_error {a b x y r ea eb er : ℝ}
    (ha : |a - x| ≤ ea) (hb : |b - y| ≤ eb)
    (hr : |r - (a - b)| ≤ er) :
    |r - (x - y)| ≤ ea + eb + er := by
  have hp : |(a - b) - (x - y)| ≤ ea + eb := by
    have he : (a - b) - (x - y) = (a - x) - (b - y) := by ring
    rw [he]
    exact (abs_sub _ _).trans (add_le_add ha hb)
  exact ((abs_sub_le r (a - b) (x - y)).trans (add_le_add hr hp)).trans_eq
    (add_comm er (ea + eb))

/-- An asymmetric product bound avoids discarding the higher-order input error.
One factor is the computed value and the other is the exact reference value. -/
theorem mul_error {a b x y r ea eb er : ℝ}
    (ha : |a - x| ≤ ea) (hb : |b - y| ≤ eb)
    (hr : |r - a * b| ≤ er) :
    |r - x * y| ≤ |b| * ea + |x| * eb + er := by
  have hp : |a * b - x * y| ≤ |b| * ea + |x| * eb := by
    calc
      |a * b - x * y| = |(a - x) * b + x * (b - y)| := by congr 1; ring
      _ ≤ |(a - x) * b| + |x * (b - y)| := abs_add_le _ _
      _ = |b| * |a - x| + |x| * |b - y| := by rw [abs_mul, abs_mul]; ring
      _ ≤ |b| * ea + |x| * eb :=
        add_le_add (mul_le_mul_of_nonneg_left ha (abs_nonneg b))
          (mul_le_mul_of_nonneg_left hb (abs_nonneg x))
  exact ((abs_sub_le r (a * b) (x * y)).trans (add_le_add hr hp)).trans_eq
    (add_comm er (|b| * ea + |x| * eb))

/-- Division exposes both the computed and reference denominator.
The reference denominator must be nonzero independently of the floating-point zero test. -/
theorem div_error {a b x y r ea eb er : ℝ} (hb0 : b ≠ 0) (hy0 : y ≠ 0)
    (ha : |a - x| ≤ ea) (hb : |b - y| ≤ eb)
    (hr : |r - a / b| ≤ er) :
    |r - x / y| ≤ ea / |b| + |x| * eb / (|b| * |y|) + er := by
  have hp : |a / b - x / y| ≤ ea / |b| + |x| * eb / (|b| * |y|) := by
    calc
      |a / b - x / y| = |(a - x) / b + x * (y - b) / (b * y)| := by
        congr 1
        field_simp
        ring
      _ ≤ |(a - x) / b| + |x * (y - b) / (b * y)| := abs_add_le _ _
      _ = |a - x| / |b| + |x| * |b - y| / (|b| * |y|) := by
        rw [abs_div, abs_div, abs_mul, abs_mul, abs_sub_comm y b]
      _ ≤ ea / |b| + |x| * eb / (|b| * |y|) :=
        add_le_add (div_le_div_of_nonneg_right ha (abs_nonneg b))
          (div_le_div_of_nonneg_right
            (mul_le_mul_of_nonneg_left hb (abs_nonneg x))
            (mul_nonneg (abs_nonneg b) (abs_nonneg y)))
  exact ((abs_sub_le r (a / b) (x / y)).trans (add_le_add hr hp)).trans_eq
    (add_comm er (ea / |b| + |x| * eb / (|b| * |y|)))

end

end Proofs.RuntimeApprox.Roundoff

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
  exact RuntimeApprox.Roundoff.add_error hx hy
    (roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) _)

/--
Compositional absolute-error bound for rounded multiplication.

Besides the fresh rounding term for `xhat * yhat`, the budget includes magnitude-weighted input
errors and the conservative cross term `2 * epsx * epsy`; it is not a first-order-only estimate.
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
  have hbudget : epsx * abs yhat + abs x * epsy ≤
      (abs xhat + epsx) * epsy + (abs yhat + epsy) * epsx := by
    nlinarith [mul_le_mul_of_nonneg_right hx_abs hepsy, mul_nonneg hepsx hepsy]
  have hbound := RuntimeApprox.Roundoff.mul_error hx hy
    (roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (xhat * yhat))
  exact hbound.trans (by nlinarith)

end

end RuntimeRoundingApprox
end Proofs
