/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Probability.Distributions.Gamma
public import TorchLean.Tensor.Special.Gamma

/-!
# The regularized incomplete gamma and beta functions as distribution functions

For `a > 0`, the integrand of DLMF equation 8.2.1 divided by `Γ(a)` is the density of the gamma
law with shape `a` and rate `1`, Mathlib's `ProbabilityTheory.gammaPDFReal a 1`. So on `x ≥ 0`
the regularized lower incomplete gamma function `P(a, x)`, `TorchLean.igamma a x`, is the
distribution function of Mathlib's `ProbabilityTheory.gammaMeasure a 1`
(`igamma_eq_cdf_gammaMeasure`). In the same way, for positive `a` and `b` the integrand of DLMF
equation 8.17.1 divided by `B(a, b)`, the normalization of equation 8.17.2, is the density of
Mathlib's beta law, so on `[0, 1]` the
regularized incomplete beta function `I_x(a, b)`, `TorchLean.betainc a b x`, is the distribution
function of `ProbabilityTheory.betaMeasure a b` (`betainc_eq_cdf_betaMeasure`). The facts Mathlib
proves for every distribution function then hold for both: their values lie in `[0, 1]`
(`igamma_mem_Icc`, `betainc_mem_Icc`), `igamma a` is monotone on `x ≥ 0` (`monotoneOn_igamma`) and
`betainc a b` is monotone on `[0, 1]` (`monotoneOn_betainc`).

This is a module of its own because Mathlib's distribution functions bring in the conditional
distribution machinery of `Mathlib.Probability.CDF`, which the operator table does not need.

## References

- NIST Digital Library of Mathematical Functions, sections 8.2(i) (equations 8.2.1 and 8.2.4) and
  8.17(i) (equations 8.17.1 and 8.17.2): `https://dlmf.nist.gov/8.2`,
  `https://dlmf.nist.gov/8.17`
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Set

namespace TorchLean

/-- For `a > 0` and `x ≥ 0`, `P(a, x)` is the distribution function of the gamma law with shape
`a` and rate `1`, evaluated at `x`. -/
theorem igamma_eq_cdf_gammaMeasure {a x : ℝ} (ha : 0 < a) (hx : 0 ≤ x) :
    igamma a x = cdf (gammaMeasure a 1) x := by
  rw [cdf_gammaMeasure_eq_integral ha one_pos, igamma, ite_eq_right fun h => ha.ne' h.1,
    intervalIntegral.integral_of_le hx, ← integral_div,
    setIntegral_eq_of_subset_of_forall_sdiff_eq_zero (s := Icc 0 x) measurableSet_Iic
      Icc_subset_Iic_self, integral_Icc_eq_integral_Ioc]
  · refine setIntegral_congr_fun measurableSet_Ioc fun t ht => ?_
    rw [gammaPDFReal, ite_eq_left ht.1.le, one_rpow, one_mul, gammaIntegrand]
    ring
  · intro t ht
    have h : t < 0 := lt_of_not_ge fun h => ht.2 ⟨h, ht.1⟩
    rw [gammaPDFReal, ite_eq_right h.not_ge]

/-- For `a > 0`, `P(a, x)` lies in `[0, 1]` at every `x ≥ 0`. -/
theorem igamma_mem_Icc {a x : ℝ} (ha : 0 < a) (hx : 0 ≤ x) : igamma a x ∈ Icc (0 : ℝ) 1 := by
  rw [igamma_eq_cdf_gammaMeasure ha hx]
  exact ⟨cdf_nonneg _ x, cdf_le_one _ x⟩

/-- For `a > 0`, `P(a, ·)` is monotone on `x ≥ 0`. -/
theorem monotoneOn_igamma {a : ℝ} (ha : 0 < a) : MonotoneOn (igamma a) (Ici 0) :=
  fun x hx y hy hxy => by
    rw [igamma_eq_cdf_gammaMeasure ha hx, igamma_eq_cdf_gammaMeasure ha hy]
    exact monotone_cdf _ hxy

/-- The density of the beta law is nonnegative everywhere. -/
private theorem betaPDFReal_nonneg {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (t : ℝ) :
    0 ≤ betaPDFReal a b t := by
  by_cases h : 0 < t ∧ t < 1
  · exact (betaPDFReal_pos h.1 h.2 ha hb).le
  · rw [betaPDFReal, ite_eq_right h]

/-- For positive `a` and `b` and `x ∈ [0, 1]`, `I_x(a, b)` is the distribution function of the
beta law with parameters `a` and `b`, evaluated at `x`. -/
theorem betainc_eq_cdf_betaMeasure {a b x : ℝ} (ha : 0 < a) (hb : 0 < b)
    (hx : x ∈ Icc (0 : ℝ) 1) : betainc a b x = cdf (betaMeasure a b) x := by
  have hCdf : cdf (betaMeasure a b) x = ∫ t in Iic x, betaPDFReal a b t := by
    have := isProbabilityMeasureBeta ha hb
    rw [cdf_eq_real, betaMeasure, measureReal_def, withDensity_apply _ measurableSet_Iic]
    refine (integral_eq_lintegral_of_nonneg_ae ?_ ?_).symm
    · exact ae_of_all _ (betaPDFReal_nonneg ha hb)
    · fun_prop
  rw [hCdf, setIntegral_eq_of_subset_of_forall_sdiff_eq_zero (s := Ioc 0 x) measurableSet_Iic
      Ioc_subset_Iic_self, integral_Ioc_eq_integral_Ioo, betainc,
    intervalIntegral.integral_of_le hx.1, integral_Ioc_eq_integral_Ioo, ← integral_div]
  · refine setIntegral_congr_fun measurableSet_Ioo fun t ht => ?_
    rw [betaPDFReal, ite_eq_left ⟨ht.1, ht.2.trans_le hx.2⟩]
    ring
  · intro t ht
    have h : ¬(0 < t ∧ t < 1) := fun h => ht.2 ⟨h.1, ht.1⟩
    rw [betaPDFReal, ite_eq_right h]

/-- For positive `a` and `b`, `I_x(a, b)` lies in `[0, 1]` at every `x ∈ [0, 1]`. -/
theorem betainc_mem_Icc {a b x : ℝ} (ha : 0 < a) (hb : 0 < b) (hx : x ∈ Icc (0 : ℝ) 1) :
    betainc a b x ∈ Icc (0 : ℝ) 1 := by
  rw [betainc_eq_cdf_betaMeasure ha hb hx]
  exact ⟨cdf_nonneg _ x, cdf_le_one _ x⟩

/-- For positive `a` and `b`, `I_x(a, b)` is monotone in `x` on `[0, 1]`. -/
theorem monotoneOn_betainc {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    MonotoneOn (betainc a b) (Icc 0 1) :=
  fun x hx y hy hxy => by
    rw [betainc_eq_cdf_betaMeasure ha hb hx, betainc_eq_cdf_betaMeasure ha hb hy]
    exact monotone_cdf _ hxy

end TorchLean
