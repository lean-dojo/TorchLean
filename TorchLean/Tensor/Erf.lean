/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Calculus.Deriv.Inverse
public import Mathlib.Analysis.Calculus.Deriv.MeanValue
public import Mathlib.Analysis.Calculus.InverseFunctionTheorem.ContDiff
public import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
public import Mathlib.MeasureTheory.Integral.IntegralEqImproper
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
public import Mathlib.Topology.Order.IntermediateValue
public import Mathlib.Topology.Order.MonotoneConvergence

/-!
# The Gauss error function

Mathlib has no error function, so this module defines `erf x = 2 / √π ∫₀ˣ exp (-t²) dt` on the
reals. It proves the facts that `torch.erf` and the exact GELU rely on: `erf 0 = 0`, oddness, the
derivative `2 / √π exp (-x²)`, strict monotonicity, and the limits `1` at `+∞` and `-1` at `-∞`,
hence `-1 < erf x < 1`.

From `erf` it defines the complementary function `erfc`, the scaled `erfcx`, the inverse `erfinv`,
the standard normal distribution function `ndtr` and its inverse `ndtri`. It proves that `erf`,
`erfc`, `erfcx` and `ndtr` are smooth, the derivative of `erfc`, that `erf` maps onto `(-1, 1)`
(`erf_surjOn`), that `erfinv` inverts `erf` there and is its left inverse everywhere
(`erf_erfinv`, `erfinv_erf`), and the same two facts for `ndtri` and `ndtr` on `(0, 1)`. By the
inverse function theorem `erfinv` is smooth on `(-1, 1)` and `ndtri` on `(0, 1)`.

Our `erf` is DLMF equation 7.2.1, written by hand. We proved the facts above with Mathlib's
calculus, the limit at `+∞` from the Gaussian integral `integral_gaussian_Ioi`. No binary format in
the library computes it. `Erf` has instances for `ℝ` (this `erf`) and for dual numbers over a type
that has one, and none for `Float`, `Float32` or a configured binary format
(`TorchLean.Tensor.ConfiguredBinary`). A binary32 run refuses `torch.erf` and the exact GELU, and
`lax.erf` (`test_error_function_needs_an_implementation` in `scripts/checks/test_ir_capture.py` and
in `scripts/checks/ir_capture_jax.py`). The one approximation is the `Float` erf of the parity test
`TorchLeanTest/Interop/IRPyTorch.lean`, the series of DLMF equation 7.6.2 summed to `n = 128`, with
`2 / √π` rounded to `1.1283791670955126`, and `±1` where `|x| ≥ 6`. The test compares the exact GELU
through it with `torch.nn.functional.gelu` in binary64, to a relative `1e-12` at `±1` and `±30`.

## References

- NIST Digital Library of Mathematical Functions, section 7.2(i), error functions, equation 7.2.1:
  `https://dlmf.nist.gov/7.2`
- NIST Digital Library of Mathematical Functions, section 7.6(i), power series, equation 7.6.2:
  `https://dlmf.nist.gov/7.6`
- NIST Digital Library of Mathematical Functions, section 7.17(i), the inverse error functions:
  `https://dlmf.nist.gov/7.17`
- PyTorch 2.13 documentation, `torch.special`, the entries `erfc`, `erfcx`, `ndtr` and `ndtri`:
  `https://docs.pytorch.org/docs/2.13/special.html`
-/

@[expose] public section

noncomputable section

namespace TorchLean

open Real MeasureTheory Filter Topology Set
open scoped ContDiff

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

/-- The error function is smooth: its derivative is the smooth scaled Gaussian. -/
theorem contDiff_erf : ContDiff ℝ ∞ erf := by
  refine contDiff_infty_iff_deriv.2 ⟨differentiable_erf, ?_⟩
  rw [show deriv erf = fun x => 2 / √π * exp (-x ^ 2) from funext deriv_erf]
  fun_prop

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

/-! ## The complementary, scaled and inverse error functions and the normal distribution -/

/-- The complementary error function `1 - erf x`, `torch.special.erfc` and `lax.erfc`. We wrote it
by hand as DLMF equation 7.2.2. -/
def erfc (x : ℝ) : ℝ := 1 - erf x

/-- The scaled complementary error function `exp (x²) erfc x`, `torch.special.erfcx`. We took this
definition by hand from the `torch.special.erfcx` page of the PyTorch 2.13 documentation. -/
def erfcx (x : ℝ) : ℝ := exp (x ^ 2) * erfc x

/-- The inverse error function, `torch.special.erfinv` and `lax.erf_inv`: on `(-1, 1)` the unique
`x` with `erf x = y` (`erf_erfinv`, `erfinv_erf`). Elsewhere it is `Function.invFun`'s junk value,
where PyTorch 2.13 returns `±∞` at `±1` and a NaN outside `[-1, 1]`, which its docstring doesn't
state; we wrote it by hand from DLMF section 7.17(i). -/
def erfinv (y : ℝ) : ℝ := Function.invFun erf y

/-- The standard normal distribution function `(1 + erf (x / √2)) / 2`, `torch.special.ndtr`. The
`torch.special.ndtr` entry of the PyTorch 2.13 documentation specifies the area under the standard
Gaussian density from `-∞` to `x`; we wrote the closed form by hand, and
`TorchLean.cdf_gaussianReal` (`TorchLean/NN/NormalCDF.lean`) proves that it is that area, Mathlib's
`cdf (gaussianReal 0 1) x`. -/
def ndtr (x : ℝ) : ℝ := (1 + erf (x / √2)) / 2

/-- The inverse of `ndtr` on `(0, 1)`, `torch.special.ndtri`: `√2 erfinv (2 p - 1)`, written by hand
from the `torch.special.ndtri` page of the PyTorch 2.13 documentation and DLMF section 7.17(i). -/
def ndtri (p : ℝ) : ℝ := √2 * erfinv (2 * p - 1)

/-- The derivative of `erfc` is `-2 / √π exp (-x²)`. -/
theorem hasDerivAt_erfc (x : ℝ) : HasDerivAt erfc (-(2 / √π * exp (-x ^ 2))) x :=
  (hasDerivAt_erf x).const_sub 1

/-- `deriv erfc` is the negated scaled Gaussian. -/
theorem deriv_erfc : deriv erfc = fun x => -(2 / √π * exp (-x ^ 2)) :=
  funext fun x => (hasDerivAt_erfc x).deriv

/-- The complementary error function is smooth. -/
theorem contDiff_erfc : ContDiff ℝ ∞ erfc :=
  contDiff_const.sub contDiff_erf

/-- The scaled complementary error function is smooth. -/
theorem contDiff_erfcx : ContDiff ℝ ∞ erfcx :=
  (contDiff_exp.comp (contDiff_id.pow 2)).mul contDiff_erfc

/-- The standard normal distribution function is smooth. -/
theorem contDiff_ndtr : ContDiff ℝ ∞ ndtr :=
  (contDiff_const.add (contDiff_erf.comp (contDiff_id.div_const √2))).div_const 2

/-- The error function takes every value of `(-1, 1)`. -/
theorem erf_surjOn (y : ℝ) (hy : y ∈ Ioo (-1 : ℝ) 1) : y ∈ range erf := by
  obtain ⟨b, hb⟩ := (tendsto_erf_atTop.eventually (lt_mem_nhds hy.2)).exists
  obtain ⟨a, ha⟩ := (tendsto_erf_atBot.eventually (gt_mem_nhds hy.1)).exists
  exact intermediate_value_univ a b continuous_erf ⟨ha.le, hb.le⟩

/-- `erfinv` inverts `erf` on `(-1, 1)`. -/
theorem erf_erfinv {y : ℝ} (hy : y ∈ Ioo (-1 : ℝ) 1) : erf (erfinv y) = y :=
  Function.invFun_eq (erf_surjOn y hy)

/-- `erfinv` is a left inverse of `erf` everywhere. -/
theorem erfinv_erf (x : ℝ) : erfinv (erf x) = x :=
  Function.leftInverse_invFun strictMono_erf.injective x

/-- `erfinv` is smooth on `(-1, 1)`. Near `erf a` it agrees with the local inverse that Mathlib's
inverse function theorem gives for `erf` at `a`, where the slope `2 / √π exp (-a²)` is positive,
and that local inverse is as smooth as `erf`. -/
theorem contDiffAt_erfinv {y : ℝ} (hy : y ∈ Ioo (-1 : ℝ) 1) : ContDiffAt ℝ ∞ erfinv y := by
  obtain ⟨a, rfl⟩ := erf_surjOn y hy
  have hd : 2 / √π * exp (-a ^ 2) ≠ 0 := by positivity
  have hf : ContDiffAt ℝ ∞ erf a := contDiff_erf.contDiffAt
  have hf' := (hasDerivAt_erf a).hasFDerivAt_equiv hd
  have hn : (∞ : WithTop ℕ∞) ≠ 0 := by simp
  refine (hf.to_localInverse hf' hn).congr_of_eventuallyEq ?_
  filter_upwards [(hf.hasStrictFDerivAt hf' hn).eventually_right_inverse] with z hz
  exact (congrArg erfinv hz).symm.trans (erfinv_erf _)

/-- `ndtri` inverts `ndtr` on `(0, 1)`. -/
theorem ndtr_ndtri {p : ℝ} (hp : p ∈ Ioo (0 : ℝ) 1) : ndtr (ndtri p) = p := by
  have h2 : (√2 : ℝ) ≠ 0 := by positivity
  have hy : 2 * p - 1 ∈ Ioo (-1 : ℝ) 1 := ⟨by linarith [hp.1], by linarith [hp.2]⟩
  rw [ndtr, ndtri, mul_div_cancel_left₀ _ h2, erf_erfinv hy]
  ring

/-- `ndtri` is a left inverse of `ndtr` everywhere. -/
theorem ndtri_ndtr (x : ℝ) : ndtri (ndtr x) = x := by
  have h2 : (√2 : ℝ) ≠ 0 := by positivity
  rw [ndtri, ndtr, show 2 * ((1 + erf (x / √2)) / 2) - 1 = erf (x / √2) by ring, erfinv_erf,
    mul_div_cancel₀ _ h2]

/-- `ndtri` is smooth on `(0, 1)`, where `2 p - 1` stays in `(-1, 1)`. -/
theorem contDiffAt_ndtri {p : ℝ} (hp : p ∈ Ioo (0 : ℝ) 1) : ContDiffAt ℝ ∞ ndtri p := by
  have hy : 2 * p - 1 ∈ Ioo (-1 : ℝ) 1 := ⟨by linarith [hp.1], by linarith [hp.2]⟩
  exact contDiffAt_const.mul ((contDiffAt_erfinv hy).comp p (f := fun q => 2 * q - 1) (by fun_prop))

end TorchLean
