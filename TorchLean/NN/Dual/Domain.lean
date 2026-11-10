/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Dual.Jet
import Mathlib.Analysis.SpecialFunctions.Complex.LogDeriv
import Mathlib.Analysis.SpecialFunctions.Pow.Deriv

/-!
# Derivative rules on local domains

Logarithms and square roots need not be smooth everywhere, so their jet laws use smoothness near
the evaluation point and a nonzero argument or denominator there. Continuity gives the same domain
condition nearby, where mathlib's derivative rules identify the remaining jet coefficients.

The quotient law in `TorchLean.NN.Dual.Jet` supplies their division prerequisite and
follows the runtime's arithmetic safeguards. All of these theorems are about exact-real
coefficients, so native floating-point execution needs its own rounding argument.

The principal angle is the imaginary part of the complex logarithm. Its dual tangent agrees with
the classical derivative on the slit plane, away from the origin and the negative real axis.
Constant powers skip the logarithm term and use the polynomial rule, which also holds at a zero
base when the exponent permits a derivative there.
-/

public section

open scoped ContDiff Topology
open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Dual

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- If `f` is `C^n` at `x` and `f x ≠ 0`, the runtime logarithm maps the jet of `f` to the jet of
`Real.log ∘ f`. Mathlib's real logarithm is smooth at negative as well as positive inputs, so
`f x` may have either sign. -/
theorem jet_log_comp {n : Nat} (directions : Fin n → E) {f : E → ℝ} {x : E}
    (hf : ContDiffAt ℝ n f x) (hzero : f x ≠ 0) :
    jet directions (fun y => Real.log (f y)) x = MathFunctions.log (jet directions f x) := by
  induction n generalizing f with
  | zero => rfl
  | succ n ih =>
    let v := directions (Fin.last n)
    have hfn : ContDiffAt ℝ n f x := hf.of_le (by simp)
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    have heq : (fun y => fderiv ℝ (fun z => Real.log (f z)) y v) =ᶠ[𝓝 x]
        fun y => fderiv ℝ f y v / f y := by
      filter_upwards [hf.eventually (by simp), hf.continuousAt.eventually_ne hzero]
        with y hfy hy
      rw [fderiv.log (hfy.differentiableAt (by simp)) hy]
      simp only [smul_apply, smul_eq_mul, div_eq_mul_inv, mul_comm]
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ hfn hzero
    · rw [jet_congr _ heq, jet_div _ hf' hfn hzero]
      rfl

/-- For `x ≠ 0`, the runtime logarithm of a seeded scalar input is the jet of `Real.log`. -/
theorem jet_log {n : Nat} (directions : Fin n → ℝ) {x : ℝ} (hx : x ≠ 0) :
    jet directions Real.log x = MathFunctions.log (Nested.seed directions x) := by
  simpa only [jet_id] using jet_log_comp directions (f := fun y => y) contDiffAt_id hx

/-- For `x ≠ 0`, dividing one by a seeded scalar input gives the jet of `fun y => 1 / y`. -/
theorem jet_one_div {n : Nat} (directions : Fin n → ℝ) {x : ℝ} (hx : x ≠ 0) :
    jet directions (fun y => 1 / y) x = 1 / Nested.seed directions x := by
  simpa only [jet_one, jet_id] using jet_div directions
    (f := fun _ : ℝ => 1) (g := fun y => y) contDiffAt_const contDiffAt_id hx

private theorem ofPrimal_natCast (n k : Nat) :
    Nested.ofPrimal n (k : ℝ) = (k : Nested ℝ n) := by
  induction n with
  | zero => rfl
  | succ n ih => exact congrArg (fun a => Dual.mk a 0) ih

/-- If `f` is `C^n` at `x` and `f x ≠ 0`, the runtime square root maps the jet of `f` to the jet
of `Real.sqrt ∘ f`. For a negative exact-real `f x`, the runtime and mathlib both use the locally
constant zero branch. -/
theorem jet_sqrt_comp {n : Nat} (directions : Fin n → E) {f : E → ℝ} {x : E}
    (hf : ContDiffAt ℝ n f x) (hzero : f x ≠ 0) :
    jet directions (fun y => Real.sqrt (f y)) x = MathFunctions.sqrt (jet directions f x) := by
  induction n generalizing f with
  | zero => rfl
  | succ n ih =>
    let ds := Fin.init directions
    let v := directions (Fin.last n)
    have hfn : ContDiffAt ℝ n f x := hf.of_le (by simp)
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    change Dual.mk _ _ = Dual.mk _ (if 0 < jet ds f x then _ else 0)
    congr 1
    · exact ih _ hfn hzero
    · simp only [jet_pos]
      by_cases hpos : 0 < f x
      · simp only [hpos, ↓reduceIte]
        have hUnsupported (depth : Nat) :
            Numeric.QuotientArithmetic.supported (α := Nested ℝ depth) = false := by
          induction depth with
          | zero => rfl
          | succ depth ih => exact ih
        simp only [Numeric.sqrtBackward, hUnsupported, Bool.not_false,
          ↓reduceIte]
        have heq : (fun y => fderiv ℝ (fun z => Real.sqrt (f z)) y v) =ᶠ[𝓝 x]
            fun y => fderiv ℝ f y v / (2 * Real.sqrt (f y)) := by
          filter_upwards [hf.eventually (by simp), hf.continuousAt.eventually_ne hzero]
            with y hfy hy
          rw [fderiv_sqrt (hfy.differentiableAt (by simp)) hy]
          simp only [smul_apply, smul_eq_mul, div_eq_mul_inv, one_mul, mul_comm]
        rw [jet_congr _ heq, jet_div _ hf' (contDiffAt_const.mul (hfn.sqrt hzero))
          (mul_ne_zero two_ne_zero (Real.sqrt_pos.mpr hpos).ne'),
          jet_mul _ contDiffAt_const (hfn.sqrt hzero), jet_const, ih _ hfn hzero]
        rw [show (2 : ℝ) = ((2 : Nat) : ℝ) from rfl, ofPrimal_natCast]
        rfl
      · simp only [hpos, ↓reduceIte]
        have hneg : f x < 0 := lt_of_le_of_ne (le_of_not_gt hpos) hzero
        have heq : (fun y => fderiv ℝ (fun z => Real.sqrt (f z)) y v) =ᶠ[𝓝 x]
            fun _ => 0 := by
          filter_upwards [hf.eventually (by simp), hf.continuousAt.eventually_lt_const hneg]
            with y hfy hy
          rw [fderiv_sqrt (hfy.differentiableAt (by simp)) hy.ne,
            Real.sqrt_eq_zero_of_nonpos hy.le]
          simp
        rw [jet_congr _ heq, jet_const, Nested.ofPrimal_zero]

/-- For `x ≠ 0`, the runtime square root of a seeded scalar input is the jet of `Real.sqrt`. -/
theorem jet_sqrt {n : Nat} (directions : Fin n → ℝ) {x : ℝ} (hx : x ≠ 0) :
    jet directions Real.sqrt x = MathFunctions.sqrt (Nested.seed directions x) := by
  simpa only [jet_id] using jet_sqrt_comp directions (f := fun y => y) contDiffAt_id hx

/-- With a constant exponent, the dual power rule gives the derivative of
`fun t => f t ^ exponent` at `point` from the derivative `df` of `f`.

Where `f point = 0`, the exponent must be zero or at least one. Positive integer exponents
therefore cover zero and negative bases without passing through a logarithm. -/
theorem hasDerivAt_pow_const {f : ℝ → ℝ} {df point exponent : ℝ}
    (hf : HasDerivAt f df point) (h : exponent = 0 ∨ f point ≠ 0 ∨ 1 ≤ exponent) :
    HasDerivAt (fun t => (f t) ^ exponent)
      (((⟨f point, df⟩ : Dual ℝ) ^ (⟨exponent, 0⟩ : Dual ℝ)).du) point := by
  change HasDerivAt _ ((if Context.isZero (0 : ℝ) then
    (if Context.isZero exponent then 0 else exponent * f point ^ (exponent - 1)) * df
    else _)) point
  by_cases he : exponent = 0
  · subst exponent
    simpa [Context.isZero] using (hasDerivAt_const point (1 : ℝ))
  · simpa [Context.isZero, he, mul_comm, mul_left_comm, mul_assoc] using
      hf.rpow_const (h.resolve_left he)

/-- For a positive base `f point`, the dual power rule gives the derivative of
`fun t => f t ^ g t` with both the base and the exponent varying.

Positivity is the domain of the logarithmic formula. When the exponent tangent `dg` is zero, the
rule selects the constant-exponent branch, which gives the same derivative.
-/
theorem hasDerivAt_pow {f g : ℝ → ℝ} {df dg point : ℝ}
    (hf : HasDerivAt f df point) (hg : HasDerivAt g dg point) (h : 0 < f point) :
    HasDerivAt (fun t => (f t) ^ (g t))
      (((⟨f point, df⟩ : Dual ℝ) ^ (⟨g point, dg⟩ : Dual ℝ)).du) point := by
  have hmath := hf.rpow hg h
  change HasDerivAt _ (if Context.isZero dg then
    (if Context.isZero (g point) then 0 else g point * f point ^ (g point - 1)) * df
    else f point ^ g point * (dg * Real.log (f point) + g point * (df / f point))) point
  by_cases hdy : dg = 0
  · subst dg
    by_cases he : g point = 0
    · simpa [Context.isZero, he] using hmath
    · simpa [Context.isZero, he, mul_comm, mul_left_comm, mul_assoc] using hmath
  · have hformula :
        f point ^ g point * (dg * Real.log (f point) + g point * (df / f point)) =
          df * g point * f point ^ (g point - 1) +
            dg * f point ^ g point * Real.log (f point) := by
      rw [Real.rpow_sub_one h.ne']
      ring
    rw [← hformula] at hmath
    simpa [Context.isZero, hdy] using hmath

private theorem angleTangent_eq_of_ne_zero (x y dx dy : ℝ) (h : (⟨x, y⟩ : ℂ) ≠ 0) :
    Numeric.angleTangent x y dx dy =
      (x * dy - y * dx) / (x * x + y * y) := by
  have hs : max |x| |y| ≠ 0 := by
    intro hs
    have hx : x = 0 := abs_nonpos_iff.mp (by simpa [hs] using le_max_left |x| |y|)
    have hy : y = 0 := abs_nonpos_iff.mp (by simpa [hs] using le_max_right |x| |y|)
    exact h (_root_.Complex.ext hx hy)
  have horigin : ¬(x = 0 ∧ y = 0) := fun ⟨hx, hy⟩ => h (_root_.Complex.ext hx hy)
  simp only [Numeric.angleTangent,
    Numeric.QuotientArithmetic.supported_real, Bool.not_false, ↓reduceIte]
  rw [ite_eq_right (by simp [horigin])]
  simp only [div_mul_div_comm, ← sub_div, ← add_div]
  exact div_div_div_cancel_right₀ (mul_ne_zero hs hs) _ _

/-- Away from the branch cut, the dual angle rule gives the derivative of
`fun t => InverseTrig.atan2 (f t) (g t)` for two differentiable real curves.

The slit-plane hypothesis on `⟨g point, f point⟩` excludes the origin and the negative real axis. -/
theorem hasDerivAt_atan2 {f g : ℝ → ℝ} {df dg point : ℝ}
    (hf : HasDerivAt f df point) (hg : HasDerivAt g dg point)
    (h : (⟨g point, f point⟩ : ℂ) ∈ _root_.Complex.slitPlane) :
    HasDerivAt (fun t => InverseTrig.atan2 (f t) (g t))
      (InverseTrig.atan2 (⟨f point, df⟩ : Dual ℝ) ⟨g point, dg⟩).du point := by
  have hcurve : HasDerivAt (fun t => (⟨g t, f t⟩ : ℂ)) ⟨dg, df⟩ point := by
    simpa only [_root_.Complex.mk_eq_add_mul_I] using!
      hg.ofReal_comp.add (hf.ofReal_comp.mul_const _root_.Complex.I)
  have hlog := hcurve.clog_real h
  have him := _root_.Complex.imCLM.hasFDerivAt.comp_hasDerivAt point hlog
  have hcoefficient :
      (InverseTrig.atan2 (⟨f point, df⟩ : Dual ℝ) ⟨g point, dg⟩).du =
        ((⟨dg, df⟩ : ℂ) / ⟨g point, f point⟩).im := by
    change Numeric.angleTangent (g point) (f point) dg df = _
    rw [angleTangent_eq_of_ne_zero _ _ _ _ (_root_.Complex.slitPlane_ne_zero h)]
    simp [_root_.Complex.div_im, _root_.Complex.normSq_apply, mul_comm, sub_div]
  rw [hcoefficient]
  change HasDerivAt (fun t => _root_.Complex.arg ⟨g t, f t⟩) _ point
  change HasDerivAt (fun t => (_root_.Complex.log ⟨g t, f t⟩).im) _ point at him
  simpa only [_root_.Complex.imCLM_apply, _root_.Complex.log_im] using him

end TorchLean.Dual
