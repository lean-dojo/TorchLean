/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Analytic.IsolatedZeros
public import Mathlib.Analysis.Calculus.Deriv.Abs
public import Mathlib.Analysis.Complex.RealDeriv
public import Mathlib.Analysis.SpecialFunctions.Bessel
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Sinc

/-!
# Bessel functions of orders zero and one

Real Bessel functions for `torch.special` and `jax.lax`, all from Mathlib's `Complex.besselJ`:
`besselJ0` and `besselJ1` (`torch.special.bessel_j0`, `bessel_j1`) are the real parts of `J₀` and
`J₁` at a real argument, and the modified functions `besselI0` and `besselI1`
(`torch.special.i0`, `i1`) are `I_ν(x) = i^(-ν) J_ν(i x)` (DLMF equation 10.27.6). `besselI0e` and
`besselI1e` (`torch.special.i0e`, `i1e`, `lax.bessel_i0e`, `lax.bessel_i1e`) scale these by
`exp (-|x|)`. We wrote each by hand from the DLMF section cited at it. Each definition takes the
real part, and drops nothing: the complex value is real (`ofReal_besselJ0` and its three
siblings), because conjugation commutes with Mathlib's regularized series `₀F₁`, whose
coefficients `1 / (n! Γ(b + n))` it maps to those at `conj b` (`Complex.Gamma_conj`).
`torch.special.spherical_bessel_j0` is `sin x / x` with `1` at `0` (DLMF equation 10.49.3), which is
Mathlib's `Real.sinc`, so it has no definition here: `SpecialFn.sphericalBesselJ0` names
`Real.sinc` directly.

Proved here: realness, the values at zero, `J₀(0) = I₀(0) = 1` and `J₁(0) = I₁(0) = 0`, and
smoothness.
`J₀`, `J₁`, `I₀` and `I₁` are smooth on `ℝ` because Mathlib proves `J_n` entire for every integer
`n` (`Complex.analyticAt_besselJ_int`). The scaled `i0e` and `i1e` are smooth away from `0`, where
the factor `exp (-|x|)` is not differentiable. `Real.sinc` is smooth on `ℝ`: away from `0` it is
`sin x / x`, and at `0` it is the slope of `sin` at `0` (`Real.sinc_eq_dslope`), which the power
series of `sin` makes analytic.

## References

- NIST Digital Library of Mathematical Functions, section 10.2(ii), equation 10.2.2, section
  10.27, equation 10.27.6, and section 10.49(i), equation 10.49.3:
  `https://dlmf.nist.gov/10.2`, `https://dlmf.nist.gov/10.27`, `https://dlmf.nist.gov/10.49`
- PyTorch 2.13 documentation, `torch.special`:
  `https://docs.pytorch.org/docs/2.13/special.html`
-/

@[expose] public section

noncomputable section

namespace TorchLean

open Complex

/-- The Bessel function of the first kind of order zero, `torch.special.bessel_j0`: the real part
of Mathlib's `J₀` (DLMF equation 10.2.2) at a real argument. -/
def besselJ0 (x : ℝ) : ℝ := (besselJ 0 x).re

/-- The Bessel function of the first kind of order one, `torch.special.bessel_j1`: the real part of
Mathlib's `J₁` (DLMF equation 10.2.2) at a real argument. -/
def besselJ1 (x : ℝ) : ℝ := (besselJ 1 x).re

/-- The modified Bessel function of the first kind of order zero, `torch.special.i0`:
`I₀(x) = J₀(i x)`, DLMF equation 10.27.6. -/
def besselI0 (x : ℝ) : ℝ := (besselJ 0 (I * x)).re

/-- The modified Bessel function of the first kind of order one, `torch.special.i1`:
`I₁(x) = -i J₁(i x)`, DLMF equation 10.27.6. -/
def besselI1 (x : ℝ) : ℝ := (-I * besselJ 1 (I * x)).re

/-- `exp (-|x|) I₀(x)`, `torch.special.i0e` and `lax.bessel_i0e`, as the PyTorch 2.13
documentation of `torch.special.i0e` defines it. -/
def besselI0e (x : ℝ) : ℝ := Real.exp (-|x|) * besselI0 x

/-- `exp (-|x|) I₁(x)`, `torch.special.i1e` and `lax.bessel_i1e`, as the PyTorch 2.13
documentation of `torch.special.i1e` defines it. -/
def besselI1e (x : ℝ) : ℝ := Real.exp (-|x|) * besselI1 x

/-- `J₀(0) = 1`. -/
@[simp] theorem besselJ0_zero : besselJ0 0 = 1 := by
  simp [besselJ0, besselJ_zero]

/-- `J₁(0) = 0`. -/
@[simp] theorem besselJ1_zero : besselJ1 0 = 0 := by
  simp [besselJ1, besselJ_zero]

/-- `I₀(0) = 1`. -/
@[simp] theorem besselI0_zero : besselI0 0 = 1 := by
  simp [besselI0, besselJ_zero]

/-- `I₁(0) = 0`. -/
@[simp] theorem besselI1_zero : besselI1 0 = 0 := by
  simp [besselI1, besselJ_zero]

open scoped ComplexConjugate

/-- Conjugating Mathlib's regularized `₀F₁(b; w)` conjugates its parameter and its argument. -/
private theorem conj_regularizedHGFun (b w : ℂ) :
    conj (regularizedHGFun 0 {b} w) = regularizedHGFun 0 {conj b} (conj w) := by
  change conj (FormalMultilinearSeries.ofScalarsSum _ w) =
    FormalMultilinearSeries.ofScalarsSum _ (conj w)
  rw [FormalMultilinearSeries.ofScalars_sum_eq, FormalMultilinearSeries.ofScalars_sum_eq,
    conj_tsum]
  refine tsum_congr fun n => ?_
  simp only [regularizedHGFunCoeff, Multiset.map_zero, Multiset.prod_zero, Multiset.map_singleton,
    Multiset.prod_singleton, smul_eq_mul, map_mul, map_div₀, map_one, map_pow, map_natCast,
    ← Gamma_conj, map_add]

/-- Mathlib's `J₀` is real on the real line, where it is `besselJ0`. -/
theorem ofReal_besselJ0 (x : ℝ) : (besselJ0 x : ℂ) = besselJ 0 x :=
  conj_eq_iff_re.1 <| by
    rw [besselJ, map_mul, conj_regularizedHGFun]
    simp [map_ofNat]

/-- Mathlib's `J₁` is real on the real line, where it is `besselJ1`. -/
theorem ofReal_besselJ1 (x : ℝ) : (besselJ1 x : ℂ) = besselJ 1 x :=
  conj_eq_iff_re.1 <| by
    rw [besselJ, map_mul, conj_regularizedHGFun]
    simp [map_ofNat]

/-- `J₀(i x)` is real for real `x`, where it is `besselI0`. -/
theorem ofReal_besselI0 (x : ℝ) : (besselI0 x : ℂ) = besselJ 0 (I * x) :=
  conj_eq_iff_re.1 <| by
    rw [besselJ, map_mul, conj_regularizedHGFun]
    simp [map_ofNat, neg_div]

/-- `-i J₁(i x)` is real for real `x`, where it is `besselI1`. -/
theorem ofReal_besselI1 (x : ℝ) : (besselI1 x : ℂ) = -I * besselJ 1 (I * x) :=
  conj_eq_iff_re.1 <| by
    rw [besselJ, map_mul, map_mul, conj_regularizedHGFun]
    simp [map_ofNat, neg_div]

open scoped ContDiff

/-- `J₀` is smooth on `ℝ`. -/
theorem contDiff_besselJ0 : ContDiff ℝ ∞ besselJ0 :=
  ContDiff.real_of_complex (e := besselJ 0) <| AnalyticOnNhd.contDiff fun z _ => by
    simpa using analyticAt_besselJ_int 0 z

/-- `J₁` is smooth on `ℝ`. -/
theorem contDiff_besselJ1 : ContDiff ℝ ∞ besselJ1 :=
  ContDiff.real_of_complex (e := besselJ 1) <| AnalyticOnNhd.contDiff fun z _ => by
    simpa using analyticAt_besselJ_int 1 z

/-- `I₀` is smooth on `ℝ`. -/
theorem contDiff_besselI0 : ContDiff ℝ ∞ besselI0 :=
  ContDiff.real_of_complex (e := fun z => besselJ 0 (I * z)) <| AnalyticOnNhd.contDiff fun z _ => by
    have h : AnalyticAt ℂ (fun z => besselJ ((0 : ℤ) : ℂ) (I * z)) z := by fun_prop
    simpa using h

/-- `I₁` is smooth on `ℝ`. -/
theorem contDiff_besselI1 : ContDiff ℝ ∞ besselI1 :=
  ContDiff.real_of_complex (e := fun z => -I * besselJ 1 (I * z)) <|
    AnalyticOnNhd.contDiff fun z _ => by
      have h : AnalyticAt ℂ (fun z => -I * besselJ ((1 : ℤ) : ℂ) (I * z)) z := by fun_prop
      simpa using h

/-- `i0e` is smooth away from `0`. -/
theorem contDiffAt_besselI0e {x : ℝ} (hx : x ≠ 0) : ContDiffAt ℝ ∞ besselI0e x :=
  (Real.contDiff_exp.contDiffAt.comp x (contDiffAt_abs hx).neg).mul
    contDiff_besselI0.contDiffAt

/-- `i1e` is smooth away from `0`. -/
theorem contDiffAt_besselI1e {x : ℝ} (hx : x ≠ 0) : ContDiffAt ℝ ∞ besselI1e x :=
  (Real.contDiff_exp.contDiffAt.comp x (contDiffAt_abs hx).neg).mul
    contDiff_besselI1.contDiffAt

/-- `Real.sinc`, the spherical Bessel function `j₀` (DLMF equation 10.49.3), is smooth on `ℝ`. -/
theorem contDiff_sinc : ContDiff ℝ ∞ Real.sinc := by
  refine contDiff_iff_contDiffAt.2 fun x => ?_
  rcases eq_or_ne x 0 with rfl | hx
  · obtain ⟨p, hp⟩ := Real.analyticAt_sin (x := 0)
    rw [Real.sinc_eq_dslope]
    exact hp.dslope_fslope.analyticAt.contDiffAt
  · exact (Real.contDiff_sin.contDiffAt.div contDiffAt_id hx).congr_of_eventuallyEq
      ((eventually_ne_nhds hx).mono fun y hy => Real.sinc_of_ne_zero hy)

end TorchLean
