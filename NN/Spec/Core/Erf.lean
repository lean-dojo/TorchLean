/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Calculus.Deriv.MeanValue
public import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
public import Mathlib.MeasureTheory.Integral.IntegralEqImproper
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
public import Mathlib.Topology.Order.MonotoneConvergence

/-!
# The Gauss error function

Mathlib has no error function, so this module defines `erf x = 2 / √π ∫₀ˣ exp (-t²) dt` on the
reals. It proves the facts that `torch.erf` and the exact GELU rely on: `erf 0 = 0`, oddness, the
derivative `2 / √π exp (-x²)`, strict monotonicity, and the limits `1` at `+∞` and `-1` at `-∞`,
hence `-1 < erf x < 1`.

## References

- NIST Digital Library of Mathematical Functions, section 7.2, error functions:
  `https://dlmf.nist.gov/7.2`
-/

@[expose] public section

noncomputable section

namespace Spec

open Real MeasureTheory Filter Topology Set

/-- The Gauss error function, `2 / √π ∫₀ˣ exp (-t²) dt`. -/
def erf (x : ℝ) : ℝ :=
  2 / √π * ∫ t in (0 : ℝ)..x, exp (-t ^ 2)

/-- The Gaussian integrand is continuous. -/
theorem continuous_exp_neg_sq : Continuous fun t : ℝ => exp (-t ^ 2) := by
  fun_prop

/-- The error function vanishes at zero. -/
@[simp] theorem erf_zero : erf 0 = 0 := by
  simp [erf]

/-- The error function is odd. -/
theorem erf_neg (x : ℝ) : erf (-x) = -erf x := by
  have h := intervalIntegral.integral_comp_neg (a := 0) (b := x) fun t : ℝ => exp (-t ^ 2)
  simp only [neg_sq, neg_zero] at h
  have hNeg : (∫ t in (0 : ℝ)..-x, exp (-t ^ 2)) = -∫ t in (0 : ℝ)..x, exp (-t ^ 2) := by
    rw [intervalIntegral.integral_symm, ← h]
  simp only [erf, hNeg]
  ring

/-- The derivative of the error function is `2 / √π exp (-x²)`. -/
theorem hasDerivAt_erf (x : ℝ) : HasDerivAt erf (2 / √π * exp (-x ^ 2)) x :=
  ((continuous_exp_neg_sq.integral_hasStrictDerivAt 0 x).hasDerivAt).const_mul (2 / √π)

/-- `deriv erf` is the scaled Gaussian. -/
theorem deriv_erf (x : ℝ) : deriv erf x = 2 / √π * exp (-x ^ 2) :=
  (hasDerivAt_erf x).deriv

/-- The error function is differentiable everywhere. -/
theorem differentiable_erf : Differentiable ℝ erf :=
  fun x => (hasDerivAt_erf x).differentiableAt

/-- The error function is continuous. -/
theorem continuous_erf : Continuous erf :=
  differentiable_erf.continuous

/-- The error function is strictly increasing. -/
theorem strictMono_erf : StrictMono erf :=
  strictMono_of_deriv_pos fun x => by
    rw [deriv_erf]
    positivity

/-- The error function tends to `1` at `+∞`. -/
theorem tendsto_erf_atTop : Tendsto erf atTop (𝓝 1) := by
  have hIntegrable : IntegrableOn (fun t : ℝ => exp (-1 * t ^ 2)) (Ioi 0) :=
    (integrable_exp_neg_mul_sq one_pos).integrableOn
  have hLimit := intervalIntegral_tendsto_integral_Ioi 0 hIntegrable tendsto_id
  rw [integral_gaussian_Ioi, div_one] at hLimit
  simp only [neg_mul, one_mul, id] at hLimit
  have hPi : √π ≠ 0 := (Real.sqrt_pos.mpr Real.pi_pos).ne'
  have hOne : 2 / √π * (√π / 2) = 1 := by field_simp
  rw [← hOne]
  exact hLimit.const_mul (2 / √π)

/-- The error function tends to `-1` at `-∞`. -/
theorem tendsto_erf_atBot : Tendsto erf atBot (𝓝 (-1)) := by
  have h := (tendsto_erf_atTop.comp tendsto_neg_atBot_atTop).neg
  refine h.congr fun x => ?_
  simp [erf_neg]

/-- The error function stays below `1`. -/
theorem erf_lt_one (x : ℝ) : erf x < 1 :=
  (strictMono_erf (lt_add_one x)).trans_le
    (strictMono_erf.monotone.ge_of_tendsto tendsto_erf_atTop (x + 1))

/-- The error function stays above `-1`. -/
theorem neg_one_lt_erf (x : ℝ) : -1 < erf x := by
  have h := erf_lt_one (-x)
  rw [erf_neg] at h
  linarith

end Spec
