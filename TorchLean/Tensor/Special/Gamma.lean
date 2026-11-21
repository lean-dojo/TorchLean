/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Calculus.FDeriv.Analytic
public import Mathlib.Analysis.Calculus.ParametricIntervalIntegral
public import Mathlib.Analysis.MellinTransform
public import Mathlib.Analysis.Normed.Operator.Prod
public import Mathlib.Analysis.Complex.CauchyIntegral
public import Mathlib.Analysis.Complex.RealDeriv
public import Mathlib.Analysis.SpecialFunctions.Gamma.Beta
public import Mathlib.Analysis.SpecialFunctions.Gamma.Deriv
public import Mathlib.Analysis.SpecialFunctions.Gamma.Digamma
public import Mathlib.Analysis.SpecialFunctions.Integrability.Basic
public import Mathlib.Analysis.SpecialFunctions.Pow.Deriv
public import Mathlib.Analysis.Calculus.IteratedDeriv.Defs
public import Mathlib.Analysis.Calculus.LogDeriv
public import Mathlib.MeasureTheory.Integral.IntegralEqImproper
public import Mathlib.Probability.Distributions.Beta

/-!
# The gamma family of special functions

Real functions for `torch.special` and `jax.lax` built on Mathlib's `Real.Gamma`: the log-gamma
function `lgamma`, the digamma and polygamma functions, the regularized incomplete gamma functions
`igamma` and `igammac` with `igammaGradA`, the derivative of `igamma` in its first argument, and
the regularized incomplete beta function `betainc`, normalized by Mathlib's beta function
`ProbabilityTheory.beta`. We wrote each by hand from the DLMF section cited at it. Lean's
functions are total, so a pole, where PyTorch returns an infinity or a NaN, has Mathlib's junk
value (`Real.Gamma` is `0` at the nonpositive integers and `Real.log 0 = 0`). PyTorch 2.13
defines `igamma` and `igammac` over finite reals where both arguments are nonnegative and not both
zero, and returns a NaN elsewhere (the `torch.special.gammainc` page). There `igamma` is the
quotient of the integral by `Real.Gamma a` and `igammac` is one minus that quotient, real numbers
that no theorem relates to PyTorch's NaN; at `(0, 0)`, for instance, `igamma` is `0` and `igammac`
is `1`.

`digamma` is the restriction of Mathlib's `Complex.digamma` to the reals, so Mathlib's digamma
facts apply to it. Proved here: `digamma` is the logarithmic derivative of `Real.Gamma` at every
real (`digamma_eq_logDeriv`) and satisfies `ψ(x + 1) = ψ(x) + 1 / x` off the poles
(`digamma_add_one`); `lgamma` has derivative `digamma` off the poles (`hasDerivAt_lgamma`);
`polygamma 0` is `digamma`; `igamma a + igammac a = 1`; and for `a > 0`, `igamma a x` tends to `1`
as `x → ∞` (`tendsto_igamma_atTop`), from Mathlib's Euler integral `Real.Gamma_eq_integral`. Off
the poles `Complex.Gamma` is analytic, as the reciprocal of the entire function `1 / Γ`
(`analyticAt_complex_Gamma`), so `Γ`, `lgamma` and `digamma` are smooth there. `igamma` and
`igammac` are jointly smooth in `(a, x)` where `a` and `x` are positive (`contDiffOn_igamma`):
`∫₀¹ exp (-t) t ^ (a - 1) dt` is the real part of a Mellin transform, holomorphic in `a`, and the
integral from `1` to `x` has a smooth integrand. For positive `a` and `b`, `I_1(a, b) = 1`
(`betainc_one`), from Mathlib's Euler beta integral `ProbabilityTheory.beta_eq_betaIntegralReal`.
`betainc` is jointly smooth in `(a, b, x)` where `a` and `b` are positive and `x ∈ (0, 1)`
(`contDiffOn_betainc`): the integral splits at `1 / 2`. The integral over `[0, 1 / 2]` is smooth in
`(a, b)` for `a > 0`, as differentiating `t ^ (a - 1) (1 - t) ^ (b - 1)` times powers of `log t`
and `log (1 - t)` under the integral sign gives integrands of the same form; the integral from
`1 / 2` to `x` has a smooth integrand.
`TorchLean.Tensor.Special.GammaCDF` shows that for `a > 0` and `x ≥ 0`, `igamma a x` is the
distribution function at `x` of Mathlib's gamma law with shape `a` and rate `1`, and that for
positive `a` and `b` and `x ∈ [0, 1]`, `betainc a b x` is that of its beta law.

## References

- NIST Digital Library of Mathematical Functions, sections 5.2(i) (equations 5.2.1 and 5.2.2),
  5.5(i) (equation 5.5.2), 5.15, 5.12 (equation 5.12.1), 8.2(i) (equations 8.2.1 and 8.2.4) and
  8.17(i) (equations 8.17.1 and 8.17.2): `https://dlmf.nist.gov/5.2`, `https://dlmf.nist.gov/5.5`,
  `https://dlmf.nist.gov/8.2`, `https://dlmf.nist.gov/8.17`
- PyTorch 2.13 documentation, `torch.special`:
  `https://docs.pytorch.org/docs/2.13/special.html`
-/

@[expose] public section

noncomputable section

namespace TorchLean

open Real MeasureTheory Filter Topology Set
open scoped ContDiff

/-- `log |Γ(x)|`, `torch.lgamma` and `lax.lgamma`, which we wrote by hand from DLMF section 5.2(i).
Mathlib's `Real.log` already takes the absolute value. -/
def lgamma (x : ℝ) : ℝ := Real.log (Real.Gamma x)

/-- The digamma function `Γ'(x) / Γ(x)`, `torch.special.digamma` and `lax.digamma`. DLMF equation
5.2.2 defines it; we wrote it by hand as the real part of Mathlib's `Complex.digamma`, the
logarithmic derivative of `Complex.Gamma`, at the real point `x`. At a pole it is `0`, Mathlib's
junk value, where PyTorch returns an infinity or a NaN. -/
def digamma (x : ℝ) : ℝ := (Complex.digamma x).re

/-- The `n`-th derivative of `digamma`, `torch.special.polygamma` and `lax.polygamma`, as DLMF
section 5.15 defines it; we wrote it by hand. -/
def polygamma (n : ℕ) (x : ℝ) : ℝ := iteratedDeriv n digamma x

/-- The integrand `exp (-t) t ^ (a - 1)` of Euler's integral, in Mathlib's order. -/
def gammaIntegrand (a t : ℝ) : ℝ := exp (-t) * t ^ (a - 1)

/-- The regularized lower incomplete gamma function `P(a, x) = γ(a, x) / Γ(a)`,
`torch.special.gammainc` and `lax.igamma`, from DLMF equations 8.2.1 and 8.2.4, written by hand.
At `a = 0 < x` it is `1`, the limit, which PyTorch 2.13 returns there though its docstring doesn't
say so. -/
def igamma (a x : ℝ) : ℝ :=
  if a = 0 ∧ 0 < x then 1 else (∫ t in (0 : ℝ)..x, gammaIntegrand a t) / Real.Gamma a

/-- The regularized upper incomplete gamma function `Q(a, x) = 1 - P(a, x)`,
`torch.special.gammaincc` and `lax.igammac`, DLMF equation 8.2.5, written by hand. -/
def igammac (a x : ℝ) : ℝ := 1 - igamma a x

/-- The derivative of `igamma` in its first argument, `lax.igamma_grad_a`, which we wrote by hand
from the JAX 0.11.1 documentation of `jax.lax.igamma_grad_a`. -/
def igammaGradA (a x : ℝ) : ℝ := deriv (fun b => igamma b x) a

/-- The regularized incomplete beta function
`I_x(a, b) = ∫₀ˣ t^(a-1) (1-t)^(b-1) dt / B(a, b)`, `lax.betainc`, written by
hand from DLMF equation 8.17.1, the incomplete beta integral, and equation 8.17.2, its quotient by
`B(a, b)`. The beta function `B(a, b)` is Mathlib's
`ProbabilityTheory.beta`, the quotient `Γ(a) Γ(b) / Γ(a + b)` of DLMF equation 5.12.1. -/
def betainc (a b x : ℝ) : ℝ :=
  (∫ t in (0 : ℝ)..x, t ^ (a - 1) * (1 - t) ^ (b - 1)) / ProbabilityTheory.beta a b

/-- A real point off the poles of `Γ` is off them as a complex point. -/
private theorem ofReal_ne_neg_nat {x : ℝ} (hx : ∀ m : ℕ, x ≠ -m) : ∀ m : ℕ, (x : ℂ) ≠ -m :=
  fun m h => hx m (by exact_mod_cast h)

/-- `Real.Gamma` is the real part of `Complex.Gamma` along the real line. -/
private theorem re_Gamma_ofReal : (fun y : ℝ => (Complex.Gamma y).re) = Real.Gamma :=
  funext fun y => by rw [Complex.Gamma_ofReal, Complex.ofReal_re]

/-- On the reals, `digamma` is the logarithmic derivative of `Real.Gamma`, poles included, where
both sides are `0`. -/
theorem digamma_eq_logDeriv (x : ℝ) : digamma x = logDeriv Real.Gamma x := by
  by_cases hx : ∀ m : ℕ, x ≠ -m
  · have hc :=
      ((Complex.differentiableAt_Gamma _ (ofReal_ne_neg_nat hx)).hasDerivAt).real_of_complex
    rw [re_Gamma_ofReal] at hc
    rw [digamma, Complex.digamma_def, logDeriv_apply, logDeriv_apply, hc.deriv,
      Complex.Gamma_ofReal, Complex.div_ofReal_re]
  · push Not at hx
    obtain ⟨m, rfl⟩ := hx
    simp [digamma, Complex.digamma_def, logDeriv_apply, Complex.Gamma_neg_nat_eq_zero,
      Real.Gamma_neg_nat_eq_zero]

/-- The digamma recurrence `ψ(x + 1) = ψ(x) + 1 / x` off the poles, from Mathlib's
`Complex.digamma_apply_add_one`; DLMF equation 5.5.2. -/
theorem digamma_add_one {x : ℝ} (hx : ∀ m : ℕ, x ≠ -m) : digamma (x + 1) = digamma x + x⁻¹ := by
  rw [digamma, Complex.ofReal_add, Complex.ofReal_one,
    Complex.digamma_apply_add_one _ (ofReal_ne_neg_nat hx), Complex.add_re, ← Complex.ofReal_inv,
    Complex.ofReal_re, digamma]

/-- Off the poles, the derivative of `lgamma` is `digamma`. -/
theorem hasDerivAt_lgamma {x : ℝ} (hx : ∀ m : ℕ, x ≠ -m) : HasDerivAt lgamma (digamma x) x := by
  rw [digamma_eq_logDeriv]
  exact (Real.differentiableAt_Gamma hx).hasDerivAt.log (Real.Gamma_ne_zero hx)

/-- `Complex.Gamma` is analytic off its poles: it is the reciprocal of `1 / Γ`, which Mathlib
proves entire (`Complex.differentiable_one_div_Gamma`), at a point where `Γ` does not vanish. -/
theorem analyticAt_complex_Gamma {s : ℂ} (hs : ∀ m : ℕ, s ≠ -m) :
    AnalyticAt ℂ Complex.Gamma s := by
  convert (Complex.differentiable_one_div_Gamma.analyticAt s).inv
    (inv_ne_zero (Complex.Gamma_ne_zero hs)) using 1
  exact (inv_inv Complex.Gamma).symm

/-- `Real.Gamma` is smooth off its poles. -/
theorem contDiffAt_Gamma {x : ℝ} (hx : ∀ m : ℕ, x ≠ -m) : ContDiffAt ℝ ∞ Real.Gamma x := by
  have h := ((analyticAt_complex_Gamma (ofReal_ne_neg_nat hx)).contDiffAt (n := ∞)).real_of_complex
  rwa [re_Gamma_ofReal] at h

/-- `Real.Gamma` is smooth at a positive real. -/
private theorem contDiffAt_Gamma_of_pos {x : ℝ} (hx : 0 < x) : ContDiffAt ℝ ∞ Real.Gamma x :=
  contDiffAt_Gamma fun m h => (neg_nonpos.2 (Nat.cast_nonneg (α := ℝ) m)).not_gt (h ▸ hx)

/-- `lgamma` is smooth off the poles of `Γ`. -/
theorem contDiffAt_lgamma {x : ℝ} (hx : ∀ m : ℕ, x ≠ -m) : ContDiffAt ℝ ∞ lgamma x :=
  (contDiffAt_Gamma hx).log (Real.Gamma_ne_zero hx)

/-- `digamma` is smooth off the poles of `Γ`, where `Complex.digamma = Γ' / Γ` is analytic. -/
theorem contDiffAt_digamma {x : ℝ} (hx : ∀ m : ℕ, x ≠ -m) : ContDiffAt ℝ ∞ digamma x := by
  have hs := ofReal_ne_neg_nat hx
  have h : AnalyticAt ℂ (fun s => deriv Complex.Gamma s / Complex.Gamma s) x :=
    (analyticAt_complex_Gamma hs).deriv.div (analyticAt_complex_Gamma hs) (Complex.Gamma_ne_zero hs)
  exact h.contDiffAt.real_of_complex

/-- The zeroth polygamma function is `digamma`. -/
@[simp] theorem polygamma_zero (x : ℝ) : polygamma 0 x = digamma x := by
  simp [polygamma]

/-- The two regularized incomplete gamma functions add to one. -/
theorem igamma_add_igammac (a x : ℝ) : igamma a x + igammac a x = 1 := by
  simp [igammac]

/-- For `a > 0`, `P(a, x)` tends to `1` as `x → ∞`. -/
theorem tendsto_igamma_atTop {a : ℝ} (ha : 0 < a) : Tendsto (igamma a) atTop (𝓝 1) := by
  have hLimit := intervalIntegral_tendsto_integral_Ioi 0 (Real.GammaIntegral_convergent ha)
    tendsto_id
  rw [← Real.Gamma_eq_integral ha] at hLimit
  have hDiv := hLimit.div_const (Real.Gamma a)
  rw [div_self (Real.Gamma_pos_of_pos ha).ne'] at hDiv
  refine hDiv.congr fun x => ?_
  simp [igamma, ha.ne', gammaIntegrand, id]

/-- `∫₀¹ exp (-t) t ^ (a - 1) dt` is smooth in `a > 0`: it is the real part of the Mellin transform
of `exp (-t)` on `(0, 1]`, which is holomorphic on `re s > 0`. -/
private theorem contDiffOn_gammaHead :
    ContDiffOn ℝ ∞ (fun a => ∫ t in (0 : ℝ)..1, gammaIntegrand a t) (Ioi 0) := by
  set f : ℝ → ℂ := (Ioc (0 : ℝ) 1).indicator fun t => ((Real.exp (-t) : ℝ) : ℂ) with hf
  have heq : (fun a : ℝ => (mellin f a).re) =
      fun a => ∫ t in (0 : ℝ)..1, gammaIntegrand a t := by
    funext a
    have h1 : EqOn (fun t : ℝ => (t : ℂ) ^ ((a : ℂ) - 1) • f t)
        ((Ioc (0 : ℝ) 1).indicator fun t => ((gammaIntegrand a t : ℝ) : ℂ)) (Ioi 0) := by
      intro t ht
      by_cases h : t ∈ Ioc (0 : ℝ) 1
      · simp only [hf, indicator_of_mem h, gammaIntegrand, smul_eq_mul]
        rw [Complex.ofReal_mul, Complex.ofReal_cpow (le_of_lt ht), Complex.ofReal_sub,
          Complex.ofReal_one, mul_comm]
      · simp [hf, indicator_of_notMem h]
    rw [mellin, setIntegral_congr_fun measurableSet_Ioi h1,
      setIntegral_indicator measurableSet_Ioc, inter_eq_right.2 Ioc_subset_Ioi_self,
      integral_complex_ofReal, Complex.ofReal_re, intervalIntegral.integral_of_le zero_le_one]
  have hloc : LocallyIntegrableOn f (Ioi 0) := by
    rw [hf]
    exact ((Continuous.locallyIntegrable (f := fun t : ℝ => ((Real.exp (-t) : ℝ) : ℂ))
      (by fun_prop)).indicator measurableSet_Ioc).locallyIntegrableOn _
  have hbot : f =O[𝓝[>] (0 : ℝ)] fun t : ℝ => t ^ (-(0 : ℝ)) := by
    refine Asymptotics.IsBigO.of_bound 1
      (eventually_mem_nhdsWithin.mono fun t (ht : 0 < t) => ?_)
    rw [neg_zero, Real.rpow_zero, norm_one, one_mul, hf]
    refine (norm_indicator_le_norm_self _ _).trans ?_
    rw [Complex.norm_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    exact Real.exp_le_one_iff.2 (neg_nonpos.2 ht.le)
  have htop : ∀ c : ℝ, f =O[atTop] fun t : ℝ => t ^ (-c) := fun c =>
    EventuallyEq.trans_isBigO ((eventually_gt_atTop 1).mono fun t ht => by
      rw [hf]; exact indicator_of_notMem (fun h : t ∈ Ioc (0 : ℝ) 1 => (not_le.2 ht) h.2) _)
      (Asymptotics.isBigO_zero _ _)
  have hd : DifferentiableOn ℂ (mellin f) {s : ℂ | 0 < s.re} := fun s hs =>
    (mellin_differentiableAt_of_isBigO_rpow hloc (htop (s.re + 1)) (lt_add_one _) hbot
      hs).differentiableWithinAt
  intro a ha
  have han : AnalyticAt ℂ (mellin f) a := hd.analyticAt
    ((isOpen_lt continuous_const Complex.continuous_re).mem_nhds (by simpa using ha))
  have h := (han.contDiffAt (n := ∞)).real_of_complex
  rw [heq] at h
  exact h.contDiffWithinAt

/-- `∫₀¹ exp (-t) t ^ (a - 1)` at `t = (x - 1) w + 1`, the integral from `1` to `x` rescaled to
`[0, 1]`, is smooth in `(a, x)` for `x > 0`, where the integrand is smooth. -/
private theorem contDiffOn_gammaTail : ContDiffOn ℝ ∞
    (fun p : ℝ × ℝ => ∫ w in (0 : ℝ)..1, gammaIntegrand p.1 ((p.2 - 1) * w + 1)) {p | 0 < p.2} := by
  refine ContDiffOn.parametric_intervalIntegral (isOpen_lt continuous_const continuous_snd) ?_
  rintro ⟨⟨a, x⟩, w⟩ ⟨hx, hw⟩
  rw [uIcc_of_le zero_le_one] at hw
  have hx : 0 < x := hx
  have hpos : 0 < (x - 1) * w + 1 := by nlinarith [hw.1, hw.2, mul_nonneg hx.le hw.1]
  refine ContDiffAt.contDiffWithinAt ?_
  simp only [gammaIntegrand]
  exact (by fun_prop : ContDiffAt ℝ ∞
    (fun q : (ℝ × ℝ) × ℝ => Real.exp (-((q.1.2 - 1) * q.2 + 1))) _).mul
    (ContDiffAt.rpow (by fun_prop) (by fun_prop) hpos.ne')

/-- `igamma` is jointly smooth in `(a, x)` where `a` and `x` are positive. -/
theorem contDiffOn_igamma : ContDiffOn ℝ ∞ (Function.uncurry igamma) (Ioi 0 ×ˢ Ioi 0) := by
  have hA := contDiffOn_gammaHead.comp contDiffOn_fst
    fun p (hp : p ∈ Ioi (0 : ℝ) ×ˢ Ioi (0 : ℝ)) => hp.1
  have hB := contDiffOn_gammaTail.mono fun p (hp : p ∈ Ioi (0 : ℝ) ×ˢ Ioi (0 : ℝ)) => hp.2
  have hG : ContDiffOn ℝ ∞ (fun p : ℝ × ℝ => Real.Gamma p.1) (Ioi 0 ×ˢ Ioi 0) := fun p hp =>
    ((contDiffAt_Gamma_of_pos hp.1).comp p contDiffAt_fst).contDiffWithinAt
  refine (((hA.add ((contDiffOn_snd.sub (contDiffOn_const (c := (1 : ℝ)))).mul hB)).div hG
    fun p hp => (Real.Gamma_pos_of_pos hp.1).ne').congr ?_)
  rintro ⟨a, x⟩ ⟨ha, hx⟩
  have ha : 0 < a := ha
  have hx : 0 < x := hx
  have hint : ∀ y, 0 ≤ y → IntervalIntegrable (gammaIntegrand a) volume 0 y := fun y hy =>
    (intervalIntegrable_iff_integrableOn_Ioc_of_le hy).2
      ((Real.GammaIntegral_convergent ha).mono_set Ioc_subset_Ioi_self)
  simp only [Function.uncurry, igamma, ha.ne', false_and, ↓reduceIte]
  rw [← intervalIntegral.integral_add_adjacent_intervals (b := 1) (hint 1 zero_le_one)
    ((hint 1 zero_le_one).symm.trans (hint x hx.le))]
  have := intervalIntegral.smul_integral_comp_mul_add (f := gammaIntegrand a) (a := 0) (b := 1)
    (x - 1) 1
  simp only [mul_zero, zero_add, mul_one, sub_add_cancel, smul_eq_mul] at this
  rw [← this]
  rfl

/-- `igammac` is jointly smooth in `(a, x)` where `a` and `x` are positive. -/
theorem contDiffOn_igammac : ContDiffOn ℝ ∞ (Function.uncurry igammac) (Ioi 0 ×ˢ Ioi 0) :=
  contDiffOn_const.sub contDiffOn_igamma

/-- Euler's beta integral over `[0, 1]`, as a real interval integral. -/
private theorem integral_beta {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    ∫ t in (0 : ℝ)..1, t ^ (a - 1) * (1 - t) ^ (b - 1) = ProbabilityTheory.beta a b := by
  have h : Complex.betaIntegral a b =
      ((∫ t in (0 : ℝ)..1, t ^ (a - 1) * (1 - t) ^ (b - 1) : ℝ) : ℂ) := by
    rw [Complex.betaIntegral, ← intervalIntegral.integral_ofReal]
    refine intervalIntegral.integral_congr fun t ht => ?_
    rw [uIcc_of_le zero_le_one] at ht
    simp only [Complex.ofReal_mul, Complex.ofReal_cpow ht.1,
      Complex.ofReal_cpow (sub_nonneg.2 ht.2), Complex.ofReal_sub, Complex.ofReal_one]
  rw [ProbabilityTheory.beta_eq_betaIntegralReal a b ha hb, h, Complex.ofReal_re]

/-- For positive `a` and `b`, the regularized incomplete beta function is `1` at `x = 1`. -/
theorem betainc_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b) : betainc a b 1 = 1 := by
  rw [betainc, integral_beta ha hb, div_self (ProbabilityTheory.beta_pos ha hb).ne']

/-- A family `g i` of functions on an open set `U`, each with derivative `g' i` there, is smooth on
`U` if every `g' i` is `Cⁿ` on `U` whenever every `g i` is. -/
theorem contDiffOn_of_hasFDerivAt {E F ι : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] {U : Set E} (hU : IsOpen U)
    {g : ι → E → F} {g' : ι → E → E →L[ℝ] F}
    (hd : ∀ i, ∀ x ∈ U, HasFDerivAt (g i) (g' i x) x)
    (hc : ∀ n : ℕ, (∀ i, ContDiffOn ℝ n (g i) U) → ∀ i, ContDiffOn ℝ n (g' i) U) (i : ι) :
    ContDiffOn ℝ ∞ (g i) U := by
  have key : ∀ n : ℕ, ∀ i, ContDiffOn ℝ n (g i) U := by
    intro n
    induction n with
    | zero => exact fun i => contDiffOn_zero.2 fun x hx =>
        (hd i x hx).continuousAt.continuousWithinAt
    | succ n ih =>
      intro i
      rw [Nat.cast_succ, contDiffOn_succ_iff_fderiv_of_isOpen hU]
      exact ⟨fun x hx => (hd i x hx).differentiableAt.differentiableWithinAt,
        fun h => absurd h (by simp), (hc n ih i).congr fun x hx => (hd i x hx).fderiv⟩
  exact contDiffOn_infty.2 fun n => key n i

/-- `t ^ (a - 1) (1 - t) ^ (b - 1) (log t) ^ i (log (1 - t)) ^ j` at `k = (i, j)` and `p = (a, b)`:
the beta integrand and its partial derivatives in `(a, b)`. -/
private def betaTerm (k : ℕ × ℕ) (p : ℝ × ℝ) (t : ℝ) : ℝ :=
  t ^ (p.1 - 1) * (1 - t) ^ (p.2 - 1) * log t ^ k.1 * log (1 - t) ^ k.2

/-- For `c > 0`, `t ^ (c - 1) (1 + |log t|) ^ N` is integrable on `(0, 1 / 2]`: it is at most a
constant times `t ^ (c / 2 - 1)`. -/
private theorem integrableOn_rpow_mul_log {c : ℝ} (hc : 0 < c) (N : ℕ) :
    IntegrableOn (fun t => t ^ (c - 1) * (1 + |log t|) ^ N) (Ioc 0 (1 / 2)) := by
  have hK : ∀ t ∈ Ioc (0 : ℝ) (1 / 2), ‖t ^ (c - 1) * (1 + |log t|) ^ N‖ ≤
      (N.factorial * (2 / c) ^ N * exp (c / 2)) * t ^ (c / 2 - 1) := by
    rintro t ⟨ht0, ht1⟩
    have hlog : |log t| = -log t := abs_of_nonpos (log_nonpos ht0.le (by linarith))
    have h1 := pow_div_factorial_le_exp (c / 2 * (1 + |log t|)) (by positivity) N
    have h2 : exp (c / 2 * (1 + |log t|)) = exp (c / 2) * t ^ (-(c / 2)) := by
      rw [rpow_def_of_pos ht0, ← exp_add, hlog]; ring_nf
    have h3 : t ^ (c - 1) * t ^ (-(c / 2)) = t ^ (c / 2 - 1) := by
      rw [← rpow_add ht0]; ring_nf
    rw [mul_pow, h2, div_le_iff₀ (by positivity)] at h1
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    have hcN : (2 / c) ^ N * (c / 2) ^ N = 1 := by
      rw [← mul_pow, div_mul_div_comm, mul_comm 2 c, div_self (by positivity), one_pow]
    calc t ^ (c - 1) * (1 + |log t|) ^ N
        = t ^ (c - 1) * ((2 / c) ^ N * ((c / 2) ^ N * (1 + |log t|) ^ N)) := by
          rw [← mul_assoc ((2 / c) ^ N), hcN, one_mul]
      _ ≤ t ^ (c - 1) * ((2 / c) ^ N * (exp (c / 2) * t ^ (-(c / 2)) * N.factorial)) := by
          gcongr
      _ = _ := by rw [← h3]; ring
  refine Integrable.mono' ((intervalIntegral.intervalIntegrable_rpow' (a := 0) (b := 1 / 2)
    (r := c / 2 - 1) (by linarith)).1.const_mul _) ?_
    ((ae_restrict_iff' measurableSet_Ioc).2 (Eventually.of_forall hK))
  exact (by fun_prop : Measurable fun t : ℝ => t ^ (c - 1) * (1 + |log t|) ^ N).aestronglyMeasurable

/-- On `(0, 1 / 2]`, a bound on `|betaTerm k q|` that is uniform in `q` with `q.1 > p.1 / 2` and
`|q.2 - p.2| < 1`. -/
private theorem abs_betaTerm_le {p q : ℝ × ℝ} (hq1 : p.1 / 2 < q.1) (hq2 : |q.2 - p.2| < 1)
    (k : ℕ × ℕ) {t : ℝ} (ht : t ∈ Ioc (0 : ℝ) (1 / 2)) :
    |betaTerm k q t| ≤ exp (|p.2| + 2) * (t ^ (p.1 / 2 - 1) * (1 + |log t|) ^ k.1) := by
  obtain ⟨ht0, ht1⟩ := ht
  have hu0 : 0 < 1 - t := by linarith
  have hlogu : |log (1 - t)| ≤ 1 := by
    rw [abs_le]
    constructor
    · have h := one_sub_inv_le_log_of_pos hu0
      have : (1 - t)⁻¹ ≤ 2 := by rw [inv_le_comm₀ hu0 two_pos]; linarith
      linarith
    · have := log_le_sub_one_of_pos hu0; linarith
  have hA : t ^ (q.1 - 1) ≤ t ^ (p.1 / 2 - 1) :=
    rpow_le_rpow_of_exponent_ge ht0 (by linarith) (by linarith)
  have hq : |q.2 - 1| ≤ |p.2| + 2 := abs_le.2
    ⟨by linarith [(abs_le.1 hq2.le).1, neg_abs_le p.2],
      by linarith [(abs_le.1 hq2.le).2, le_abs_self p.2]⟩
  have hB : (1 - t) ^ (q.2 - 1) ≤ exp (|p.2| + 2) := by
    rw [rpow_def_of_pos hu0, exp_le_exp]
    calc log (1 - t) * (q.2 - 1) ≤ |log (1 - t) * (q.2 - 1)| := le_abs_self _
      _ = |log (1 - t)| * |q.2 - 1| := abs_mul _ _
      _ ≤ 1 * (|p.2| + 2) := by gcongr
      _ = _ := one_mul _
  have hC : |log t| ^ k.1 ≤ (1 + |log t|) ^ k.1 :=
    pow_le_pow_left₀ (abs_nonneg _) (by linarith [abs_nonneg (log t)]) _
  have hD : |log (1 - t)| ^ k.2 ≤ 1 := pow_le_one₀ (abs_nonneg _) hlogu
  rw [betaTerm, abs_mul, abs_mul, abs_mul, abs_pow, abs_pow, abs_of_pos (rpow_pos_of_pos ht0 _),
    abs_of_pos (rpow_pos_of_pos hu0 _)]
  calc t ^ (q.1 - 1) * (1 - t) ^ (q.2 - 1) * |log t| ^ k.1 * |log (1 - t)| ^ k.2
      ≤ t ^ (p.1 / 2 - 1) * exp (|p.2| + 2) * (1 + |log t|) ^ k.1 * 1 := by gcongr
    _ = _ := by ring

/-- For `t ∈ (0, 1)`, the derivative of `betaTerm k q t` in `q` has the components
`betaTerm (k.1 + 1, k.2) q t` and `betaTerm (k.1, k.2 + 1) q t`. -/
private theorem hasFDerivAt_betaTerm (k : ℕ × ℕ) (q : ℝ × ℝ) {t : ℝ} (ht0 : 0 < t) (ht1 : t < 1) :
    HasFDerivAt (fun q => betaTerm k q t)
      (betaTerm (k.1 + 1, k.2) q t • ContinuousLinearMap.fst ℝ ℝ ℝ +
        betaTerm (k.1, k.2 + 1) q t • ContinuousLinearMap.snd ℝ ℝ ℝ) q := by
  have h := (((((hasFDerivAt_fst (𝕜 := ℝ) (p := q)).sub_const 1).const_rpow ht0).mul
    (((hasFDerivAt_snd (𝕜 := ℝ) (p := q)).sub_const 1).const_rpow (sub_pos.2 ht1))).mul_const
      (log t ^ k.1)).mul_const (log (1 - t) ^ k.2)
  refine h.congr_fderiv (ContinuousLinearMap.ext fun v => ?_)
  simp [betaTerm, pow_succ]
  ring

/-- For `p.1 > 0`, `betaTerm k p` is integrable on `(0, 1 / 2]`. -/
private theorem integrableOn_betaTerm (k : ℕ × ℕ) {p : ℝ × ℝ} (hp : 0 < p.1) :
    IntegrableOn (betaTerm k p) (Ioc 0 (1 / 2)) :=
  ((integrableOn_rpow_mul_log (half_pos hp) k.1).const_mul (exp (|p.2| + 2))).mono'
    (by unfold betaTerm; fun_prop : Measurable (betaTerm k p)).aestronglyMeasurable
    ((ae_restrict_iff' measurableSet_Ioc).2 (Eventually.of_forall fun t ht =>
      (Real.norm_eq_abs _).trans_le
        (abs_betaTerm_le (by linarith) (by simp) k ht)))

/-- For `p.1 > 0`, the integral of `betaTerm k q` over `[0, 1 / 2]` is differentiable in `q` at
`p`, under the integral sign. -/
private theorem hasFDerivAt_betaHead (k : ℕ × ℕ) {p : ℝ × ℝ} (hp : 0 < p.1) :
    HasFDerivAt (fun q => ∫ t in (0 : ℝ)..1 / 2, betaTerm k q t)
      ((∫ t in (0 : ℝ)..1 / 2, betaTerm (k.1 + 1, k.2) p t) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (∫ t in (0 : ℝ)..1 / 2, betaTerm (k.1, k.2 + 1) p t) •
          ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
  simp only [intervalIntegral.integral_of_le (by norm_num : (0 : ℝ) ≤ 1 / 2)]
  have hs : {q : ℝ × ℝ | p.1 / 2 < q.1 ∧ |q.2 - p.2| < 1} ∈ 𝓝 p := by
    refine IsOpen.mem_nhds ?_ ⟨by linarith, by simp⟩
    exact (isOpen_lt continuous_const continuous_fst).inter
      (isOpen_lt (by fun_prop : Continuous fun q : ℝ × ℝ => |q.2 - p.2|) continuous_const)
  have hm : ∀ k' q, AEStronglyMeasurable (betaTerm k' q) (volume.restrict (Ioc (0 : ℝ) (1 / 2))) :=
    fun k' q => (by unfold betaTerm; fun_prop : Measurable (betaTerm k' q)).aestronglyMeasurable
  have key := hasFDerivAt_integral_of_dominated_of_fderiv_le
    (μ := volume.restrict (Ioc (0 : ℝ) (1 / 2))) (F := fun q t => betaTerm k q t)
    (F' := fun q t => betaTerm (k.1 + 1, k.2) q t • ContinuousLinearMap.fst ℝ ℝ ℝ +
        betaTerm (k.1, k.2 + 1) q t • ContinuousLinearMap.snd ℝ ℝ ℝ)
    (bound := fun t => 2 * exp (|p.2| + 2) * (t ^ (p.1 / 2 - 1) * (1 + |log t|) ^ (k.1 + 1))) hs
    (Eventually.of_forall fun q => hm k q) (integrableOn_betaTerm k hp)
    (((hm _ p).smul_const _).add ((hm _ p).smul_const _))
    ((ae_restrict_iff' measurableSet_Ioc).2 (Eventually.of_forall fun t ht q hq => by
      have hL : 1 ≤ 1 + |log t| := by linarith [abs_nonneg (log t)]
      calc ‖betaTerm (k.1 + 1, k.2) q t • ContinuousLinearMap.fst ℝ ℝ ℝ +
            betaTerm (k.1, k.2 + 1) q t • ContinuousLinearMap.snd ℝ ℝ ℝ‖
          ≤ ‖betaTerm (k.1 + 1, k.2) q t‖ * ‖ContinuousLinearMap.fst ℝ ℝ ℝ‖ +
            ‖betaTerm (k.1, k.2 + 1) q t‖ * ‖ContinuousLinearMap.snd ℝ ℝ ℝ‖ :=
            (norm_add_le _ _).trans (add_le_add (ContinuousLinearMap.opNorm_smul_le _ _)
              (ContinuousLinearMap.opNorm_smul_le _ _))
        _ ≤ ‖betaTerm (k.1 + 1, k.2) q t‖ * 1 + ‖betaTerm (k.1, k.2 + 1) q t‖ * 1 := by
            gcongr
            · exact ContinuousLinearMap.norm_fst_le ℝ ℝ ℝ
            · exact ContinuousLinearMap.norm_snd_le ℝ ℝ ℝ
        _ ≤ exp (|p.2| + 2) * (t ^ (p.1 / 2 - 1) * (1 + |log t|) ^ (k.1 + 1)) +
            exp (|p.2| + 2) * (t ^ (p.1 / 2 - 1) * (1 + |log t|) ^ (k.1 + 1)) := by
            simp only [mul_one, Real.norm_eq_abs]
            refine add_le_add (abs_betaTerm_le hq.1 hq.2 _ ht)
              ((abs_betaTerm_le hq.1 hq.2 _ ht).trans ?_)
            exact mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
              (pow_le_pow_right₀ hL (Nat.le_succ k.1)) (rpow_nonneg ht.1.le _)) (exp_pos _).le
        _ = _ := by ring))
    ((integrableOn_rpow_mul_log (half_pos hp) (k.1 + 1)).const_mul _)
    ((ae_restrict_iff' measurableSet_Ioc).2 (Eventually.of_forall fun t ht q _ =>
      hasFDerivAt_betaTerm k q ht.1 (by linarith [ht.2])))
  rwa [integral_add ((integrableOn_betaTerm _ hp).smul_const _)
    ((integrableOn_betaTerm _ hp).smul_const _), integral_smul_const, integral_smul_const] at key

/-- The integral of `betaTerm k q` over `[0, 1 / 2]` is smooth in `q` where `q.1 > 0`. -/
private theorem contDiffOn_betaHead (k : ℕ × ℕ) :
    ContDiffOn ℝ ∞ (fun q : ℝ × ℝ => ∫ t in (0 : ℝ)..1 / 2, betaTerm k q t) {q | 0 < q.1} :=
  contDiffOn_of_hasFDerivAt (isOpen_lt continuous_const continuous_fst)
    (g := fun k q => ∫ t in (0 : ℝ)..1 / 2, betaTerm k q t)
    (fun k _ hp => hasFDerivAt_betaHead k hp)
    (fun _ h _ => ((h _).smul contDiffOn_const).add ((h _).smul contDiffOn_const)) k

/-- The beta integrand at `t = (x - 1 / 2) w + 1 / 2`, the integral from `1 / 2` to `x` rescaled to
`[0, 1]`, is smooth in `(a, b, x)` for `x ∈ (0, 1)`, where the integrand is smooth. -/
private theorem contDiffOn_betaTail : ContDiffOn ℝ ∞
    (fun p : ℝ × ℝ × ℝ =>
      ∫ w in (0 : ℝ)..1, betaTerm (0, 0) (p.1, p.2.1) ((p.2.2 - 1 / 2) * w + 1 / 2))
    {p | p.2.2 ∈ Ioo 0 1} := by
  refine ContDiffOn.parametric_intervalIntegral (H := ℝ × ℝ × ℝ) (E := ℝ)
    (isOpen_Ioo.preimage (by fun_prop)) ?_
  rintro ⟨⟨a, b, x⟩, w⟩ ⟨⟨hx0, hx1⟩, hw⟩
  rw [uIcc_of_le zero_le_one] at hw
  have h0 : 0 < (x - 1 / 2) * w + 1 / 2 := by
    nlinarith [hw.1, hw.2, mul_nonneg hx0.le hw.1, mul_nonneg hx0.le (sub_nonneg.2 hw.2)]
  have h1 : (x - 1 / 2) * w + 1 / 2 < 1 := by
    nlinarith [hw.1, hw.2, mul_nonneg (sub_nonneg.2 hx1.le) hw.1]
  have hu : ContDiffAt ℝ ∞ (fun q : (ℝ × ℝ × ℝ) × ℝ => (q.1.2.2 - 1 / 2) * q.2 + 1 / 2)
      ((a, b, x), w) := by fun_prop
  refine ContDiffAt.contDiffWithinAt ?_
  simp only [betaTerm, pow_zero, mul_one]
  exact (hu.rpow (by fun_prop) h0.ne').mul
    ((contDiffAt_const.sub hu).rpow (by fun_prop) (sub_pos.2 h1).ne')

/-- Mathlib's beta function `Γ(a) Γ(b) / Γ(a + b)` is smooth where `a` and `b` are positive. -/
private theorem contDiffOn_beta :
    ContDiffOn ℝ ∞ (fun p : ℝ × ℝ => ProbabilityTheory.beta p.1 p.2) (Ioi 0 ×ˢ Ioi 0) :=
  fun p hp => by
    have hG : ∀ f : ℝ × ℝ → ℝ, ContDiffAt ℝ ∞ f p → 0 < f p →
        ContDiffAt ℝ ∞ (fun q => Real.Gamma (f q)) p := fun f hf h =>
      ContDiffAt.comp (g := Real.Gamma) p (contDiffAt_Gamma_of_pos h) hf
    exact ContDiffAt.contDiffWithinAt <|
      ((hG _ contDiffAt_fst hp.1).mul (hG _ contDiffAt_snd hp.2)).div
        (hG _ (contDiffAt_fst.add contDiffAt_snd) (add_pos hp.1 hp.2))
        (Real.Gamma_pos_of_pos (add_pos hp.1 hp.2)).ne'

/-- `betainc` is jointly smooth in `(a, b, x)` where `a` and `b` are positive and `x ∈ (0, 1)`. -/
theorem contDiffOn_betainc : ContDiffOn ℝ ∞ (fun p : ℝ × ℝ × ℝ => betainc p.1 p.2.1 p.2.2)
    (Ioi 0 ×ˢ Ioi 0 ×ˢ Ioo 0 1) := by
  have hK := (contDiffOn_betaHead (0, 0)).comp (contDiffOn_fst.prodMk contDiffOn_snd.fst)
    fun p (hp : p ∈ Ioi (0 : ℝ) ×ˢ Ioi (0 : ℝ) ×ˢ Ioo (0 : ℝ) 1) => hp.1
  have hT := contDiffOn_betaTail.mono
    fun p (hp : p ∈ Ioi (0 : ℝ) ×ˢ Ioi (0 : ℝ) ×ˢ Ioo (0 : ℝ) 1) => hp.2.2
  have hB := contDiffOn_beta.comp (contDiffOn_fst.prodMk contDiffOn_snd.fst)
    fun p (hp : p ∈ Ioi (0 : ℝ) ×ˢ Ioi (0 : ℝ) ×ˢ Ioo (0 : ℝ) 1) => ⟨hp.1, hp.2.1⟩
  refine ((hK.add ((contDiffOn_snd.snd.sub (contDiffOn_const (c := (1 / 2 : ℝ)))).mul hT)).div hB
    fun p hp => (ProbabilityTheory.beta_pos hp.1 hp.2.1).ne').congr ?_
  rintro ⟨a, b, x⟩ ⟨ha, hb, hx0, hx1⟩
  have ha : 0 < a := ha
  have hf : (fun t => t ^ (a - 1) * (1 - t) ^ (b - 1)) = betaTerm (0, 0) (a, b) := by
    funext t; simp [betaTerm]
  have hI1 : IntervalIntegrable (betaTerm (0, 0) (a, b)) volume 0 (1 / 2) :=
    (intervalIntegrable_iff_integrableOn_Ioc_of_le (by norm_num)).2 (integrableOn_betaTerm _ ha)
  have hI2 : IntervalIntegrable (betaTerm (0, 0) (a, b)) volume (1 / 2) x := by
    refine ContinuousOn.intervalIntegrable fun t ht => ContinuousAt.continuousWithinAt ?_
    have ht' : 0 < t ∧ t < 1 := by
      rcases le_total (1 / 2 : ℝ) x with h | h
      · rw [uIcc_of_le h] at ht; constructor <;> linarith [ht.1, ht.2]
      · rw [uIcc_of_ge h] at ht; constructor <;> linarith [ht.1, ht.2]
    rw [← hf]
    exact (continuousAt_id.rpow_const (Or.inl ht'.1.ne')).mul
      ((continuousAt_const.sub continuousAt_id).rpow_const (Or.inl (sub_pos.2 ht'.2).ne'))
  simp only [betainc]
  rw [hf, ← intervalIntegral.integral_add_adjacent_intervals hI1 hI2]
  have := intervalIntegral.smul_integral_comp_mul_add (f := betaTerm (0, 0) (a, b)) (a := 0)
    (b := 1) (x - 1 / 2) (1 / 2)
  simp only [mul_zero, zero_add, mul_one, sub_add_cancel, smul_eq_mul] at this
  rw [← this]
  rfl

end TorchLean
