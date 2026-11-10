/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Probability.CDF
public import Mathlib.Probability.Distributions.Gaussian.Real
public import TorchLean.Tensor.Op.Pointwise.Real

/-!
# The standard normal distribution function

PyTorch defines the exact GELU as `x Φ(x)`, where `Φ` is the distribution function of the standard
normal law, and computes `Φ(x)` as `(1 + erf (x / √2)) / 2`. That formula holds for Mathlib's
`ProbabilityTheory.cdf (gaussianReal 0 1)` and `erf`: the derivative of the right side is the
standard normal density, and both sides vanish at `-∞`. Over the reals the exact GELU is therefore
`x Φ(x)` with the measure-theoretic `Φ` (`Context.geluErf_eq_mul_cdf`), and `Tensor.gelu`, the
forward of `Op.gelu .none`, maps every entry `x` to `x Φ(x)`
(`TorchLean.Tensor.gelu_eq_map_mul_cdf`).

## References

- Dan Hendrycks and Kevin Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415
- `torch.nn.functional.gelu`:
  `https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html`
-/

@[expose] public section

open ProbabilityTheory MeasureTheory Filter Topology Set Real

namespace TorchLean

/-- The derivative of `(1 + erf (x / √2)) / 2` is the standard normal density. -/
theorem hasDerivAt_normalCDF (x : ℝ) :
    HasDerivAt (fun t => (1 + erf (t / √2)) / 2) (gaussianPDFReal 0 1 x) x := by
  have hInner : HasDerivAt (fun t : ℝ => t / √2) (1 / √2) x := (hasDerivAt_id x).div_const √2
  have h := ((hasDerivAt_erf (x / √2)).comp x hInner).const_add 1 |>.div_const 2
  convert h using 1
  · rfl
  have hSquare : (x / √2) ^ 2 = x ^ 2 / 2 := by
    rw [div_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
  have hRoot : √(2 * π) = √2 * √π := Real.sqrt_mul (by norm_num) π
  have hTwo : (0 : ℝ) < √2 := Real.sqrt_pos.mpr (by norm_num)
  have hPi : (0 : ℝ) < √π := Real.sqrt_pos.mpr Real.pi_pos
  rw [gaussianPDFReal_def]
  simp only [hSquare, sub_zero, NNReal.coe_one, mul_one, hRoot]
  field_simp

/-- The distribution function of the standard normal law is `(1 + erf (x / √2)) / 2`. -/
theorem cdf_gaussianReal (x : ℝ) : cdf (gaussianReal 0 1) x = (1 + erf (x / √2)) / 2 := by
  set F : ℝ → ℝ := fun t => (1 + erf (t / √2)) / 2
  have hIntegral : (gaussianReal 0 1).real (Iic x) = ∫ t in Iic x, gaussianPDFReal 0 1 t := by
    rw [measureReal_def, gaussianReal_apply_eq_integral 0 one_ne_zero,
      ENNReal.toReal_ofReal (integral_nonneg fun t => gaussianPDFReal_nonneg 0 1 t)]
  rw [cdf_eq_real, hIntegral]
  -- The integral over `Iic x` is the limit as `a → -∞` of `∫ t in a..x`, and that integral is
  -- `F x - F a`.
  have hLimit := intervalIntegral_tendsto_integral_Iic (μ := volume) x
    (integrable_gaussianPDFReal 0 1).integrableOn tendsto_id
  have hInterval (a : ℝ) : ∫ t in id a..x, gaussianPDFReal 0 1 t = F x - F a :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => hasDerivAt_normalCDF t)
      (integrable_gaussianPDFReal 0 1).intervalIntegrable
  have hBottom : Tendsto F atBot (𝓝 0) := by
    have h := ((tendsto_erf_atBot.comp (tendsto_id.atBot_div_const
      (Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 2)))).const_add 1).div_const 2
    simpa [F] using h
  have hDifference : Tendsto (fun a => F x - F a) atBot (𝓝 (F x - 0)) :=
    tendsto_const_nhds.sub hBottom
  rw [sub_zero] at hDifference
  exact tendsto_nhds_unique hLimit (hDifference.congr fun a => (hInterval a).symm)

end TorchLean

namespace TorchLean.Context

/-- Over the reals, the exact GELU is `x Φ(x)` for the standard normal distribution function `Φ`. -/
theorem geluErf_eq_mul_cdf (x : ℝ) :
    (Context.geluErf (m := Id) Erf.erf x).run = x * cdf (gaussianReal 0 1) x := by
  rw [geluErf_real, cdf_gaussianReal]
  ring

end TorchLean.Context

namespace TorchLean.Tensor

/-- Over the reals, `Tensor.gelu`, the forward of `Op.gelu .none`, maps every entry `x` to `x Φ(x)`
for the standard normal distribution function `Φ`. -/
theorem gelu_eq_map_mul_cdf {s : Shape} (x : Tensor ℝ s) :
    Tensor.gelu x = Tensor.map (fun y => y * cdf (gaussianReal 0 1) y) x := by
  unfold Tensor.gelu
  exact congrArg (Tensor.map · x) (funext Context.geluErf_eq_mul_cdf)

end TorchLean.Tensor
