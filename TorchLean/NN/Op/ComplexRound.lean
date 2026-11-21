/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Op.Table.Elementwise
public import FloatLib.Floats.Formats.BinaryInterchange.Complex.Elementary.PowerProof
public import FloatLib.Floats.Formats.BinaryInterchange.Complex.Elementary.ValueProof

/-!
# The complex functions and the complex power rounded once

In an IEEE configured format, at each entry whose complex number `z` is finite and none of the
points `complexRegular` leaves out (`0` for `expm1`, `log` and `rsqrt`, `0` and `-1` for `log1p`,
`±1` for `atanh` and `±i` for `atan`), each part of the result of `Op.complex f` is the part of
FloatLib's exact result `complexExact f z` rounded once to nearest even (`complex_forward_round`,
from `ExecComplex.exp_eq` to `ExecComplex.atanh_eq`). The exact result is `ComplexFn.value f z`
where `complexPrincipal` holds (`complexExact_value`): at every `z` for `exp`, `expm1`, `sin`,
`cos`, `tan`, `sinh`, `cosh` and `tanh`, and for the others off the branch cut. On the cut the
principal expression is the limit from one side, and FloatLib's result is that limit where the
sign bits select that side: where the imaginary sign bit is clear for `sqrt`, `rsqrt`, `log`,
`log1p`, `atanh` and `acosh`, where the real one is clear for `atan`, where the two agree for
`asinh`, and where they differ for `asin` and `acos`.

FloatLib extends `asinh` from the closed first quadrant by `asinh (-z) = -asinh z` and
conjugation, and `acosh` and `acos` from the closed upper half-plane by conjugation. The principal
expressions are odd (`ComplexFn.value_asinh_neg`) and commute with conjugation off their cuts
(`ComplexFn.value_asinh_conj`, `ComplexFn.value_acosh_conj` and `ComplexFn.value_acos_conj`),
and `asin z` is `-i asinh (i z)` (`ComplexFn.value_asin`). Where the imaginary sign bit is clear,
FloatLib's `acos` is `-i` times the expression of `acosh`, which is `π / 2 - asin z` off `(1, ∞)`
(`ComplexFn.value_acos`); where it is set, it is the conjugate of that at `conj z`, which is
`π / 2 - asin z` off `(-∞, -1)` (`ComplexFn.value_acos_lower`). Where both premises hold, each
part of the result is the part of `ComplexFn.value f z` rounded once (`complex_forward_value`).

At each entry with a finite base `z` other than zero and a finite exponent `w` where FloatLib
certifies the power (`ExecComplex.powCertified`), each part of the result of `Op.complexPow` is the
part of the exact power `ExecComplex.powExact z w` rounded once (`complexPow_forward_round`, from
`ExecComplex.pow_eq`). That is Mathlib's `z ^ w` where the imaginary sign bit of the base is clear
or the base is off the negative real axis (`powExact_value`, `complexPow_forward_value`).

These hold at every shape, every input and every entry that meets the premises.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

namespace ComplexFn

open Complex (I)

/-- Off `(-∞, 1)`, the principal expression of `acosh` at `conj w` is the conjugate of its value
at `w`. -/
theorem value_acosh_conj {w : ℂ} (hcut : ¬(w.im = 0 ∧ w.re < 1)) :
    value .acosh (starRingEnd ℂ w) = starRingEnd ℂ (value .acosh w) := by
  have hroot : ∀ x : ℂ, x.arg ≠ Real.pi →
      starRingEnd ℂ x ^ (2⁻¹ : ℂ) = starRingEnd ℂ (x ^ (2⁻¹ : ℂ)) := fun x hx => by
    rw [Complex.conj_cpow _ _ hx, map_inv₀, map_ofNat]
  have h1 : ((w + 1) / 2).arg ≠ Real.pi := by
    rw [Ne, Complex.arg_eq_pi_iff, Complex.div_ofNat_re, Complex.div_ofNat_im, Complex.add_re,
      Complex.add_im, Complex.one_re, Complex.one_im]
    rintro ⟨hre, him⟩
    exact hcut ⟨by linarith, by linarith⟩
  have h2 : ((w - 1) / 2).arg ≠ Real.pi := by
    rw [Ne, Complex.arg_eq_pi_iff, Complex.div_ofNat_re, Complex.div_ofNat_im, Complex.sub_re,
      Complex.sub_im, Complex.one_re, Complex.one_im]
    rintro ⟨hre, him⟩
    exact hcut ⟨by linarith, by linarith⟩
  have hsum : (((w + 1) / 2) ^ (2⁻¹ : ℂ) + ((w - 1) / 2) ^ (2⁻¹ : ℂ)).arg ≠ Real.pi := by
    rw [Ne, Complex.arg_eq_pi_iff, Complex.add_re, Complex.cpow_inv_two_re,
      Complex.cpow_inv_two_re]
    rintro ⟨hre, -⟩
    linarith [Real.sqrt_nonneg ((‖(w + 1) / 2‖ + ((w + 1) / 2).re) / 2),
      Real.sqrt_nonneg ((‖(w - 1) / 2‖ + ((w - 1) / 2).re) / 2)]
  have hc1 : (starRingEnd ℂ w + 1) / 2 = starRingEnd ℂ ((w + 1) / 2) := by
    rw [map_div₀, map_add, map_one, map_ofNat]
  have hc2 : (starRingEnd ℂ w - 1) / 2 = starRingEnd ℂ ((w - 1) / 2) := by
    rw [map_div₀, map_sub, map_one, map_ofNat]
  simp only [value]
  rw [hc1, hc2, hroot _ h1, hroot _ h2, ← map_add, Complex.log_conj _ hsum, map_mul, map_ofNat]

/-- In the closed upper half-plane off `(1, ∞)`, `π / 2 - asin w` is `-i` times the principal
expression `2 log (√((w + 1) / 2) + √((w - 1) / 2))` of `acosh w`. -/
theorem value_acos {w : ℂ} (him : 0 ≤ w.im) (hcut : ¬(w.im = 0 ∧ 1 < w.re)) :
    value .acos w = -I * (2 * Complex.log
      (((w + 1) / 2) ^ (2⁻¹ : ℂ) + ((w - 1) / 2) ^ (2⁻¹ : ℂ))) := by
  show (Real.pi : ℂ) / 2 + I * Complex.log (I * w + (1 - w ^ 2) ^ (2⁻¹ : ℂ)) = _
  have hu2 : (((w + 1) / 2) ^ (2⁻¹ : ℂ)) ^ 2 = (w + 1) / 2 := by
    have h := Complex.cpow_nat_inv_pow ((w + 1) / 2) (n := 2) two_ne_zero
    rwa [Nat.cast_ofNat] at h
  have hv2 : (((w - 1) / 2) ^ (2⁻¹ : ℂ)) ^ 2 = (w - 1) / 2 := by
    have h := Complex.cpow_nat_inv_pow ((w - 1) / 2) (n := 2) two_ne_zero
    rwa [Nat.cast_ofNat] at h
  have hq1 : 0 ≤ ((w + 1) / 2).im := by
    rw [show ((w + 1) / 2).im = w.im / 2 by simp]
    linarith
  have hq2 : 0 ≤ ((w - 1) / 2).im := by
    rw [show ((w - 1) / 2).im = w.im / 2 by simp]
    linarith
  have ha : 0 ≤ (((w + 1) / 2) ^ (2⁻¹ : ℂ)).re := by
    rw [Complex.cpow_inv_two_re]
    exact Real.sqrt_nonneg _
  have hb : 0 ≤ (((w + 1) / 2) ^ (2⁻¹ : ℂ)).im := by
    rw [Complex.cpow_inv_two_im_eq_sqrt hq1]
    exact Real.sqrt_nonneg _
  have hc : 0 ≤ (((w - 1) / 2) ^ (2⁻¹ : ℂ)).re := by
    rw [Complex.cpow_inv_two_re]
    exact Real.sqrt_nonneg _
  have hd : 0 ≤ (((w - 1) / 2) ^ (2⁻¹ : ℂ)).im := by
    rw [Complex.cpow_inv_two_im_eq_sqrt hq2]
    exact Real.sqrt_nonneg _
  set u := ((w + 1) / 2) ^ (2⁻¹ : ℂ)
  set v := ((w - 1) / 2) ^ (2⁻¹ : ℂ)
  have hprod : (u + v) * (u - v) = 1 := by
    linear_combination hu2 - hv2
  have hW : u + v ≠ 0 := left_ne_zero_of_mul_eq_one hprod
  -- The principal root of `1 - w ^ 2 = (-2 i u v) ^ 2` is `-2 i u v`, as `u v` is in the closed
  -- upper half-plane and not a positive real off `(1, ∞)`.
  have hsq : (1 - w ^ 2) ^ (2⁻¹ : ℂ) = -2 * I * (u * v) := by
    have h : 1 - w ^ 2 = (-2 * I * (u * v)) ^ 2 := by
      linear_combination (-4 * u ^ 2 * v ^ 2) * Complex.I_sq + (4 * v ^ 2) * hu2 +
        (2 * (w + 1)) * hv2
    have hu2i := congrArg Complex.im hu2
    have hv2r := congrArg Complex.re hv2
    simp only [sq, Complex.mul_re, Complex.mul_im, Complex.div_ofNat_re, Complex.div_ofNat_im,
      Complex.sub_re, Complex.one_re, Complex.add_im, Complex.one_im] at hu2i hv2r
    rw [h]
    apply ExecComplex.sq_cpow_inv_two
    · simp only [Complex.mul_re, Complex.mul_im, Complex.neg_re, Complex.neg_im, Complex.I_re,
        Complex.I_im, Complex.re_ofNat, Complex.im_ofNat]
      nlinarith [mul_nonneg ha hd, mul_nonneg hb hc]
    · simp only [Complex.mul_re, Complex.mul_im, Complex.neg_re, Complex.neg_im, Complex.I_re,
        Complex.I_im, Complex.re_ofNat, Complex.im_ofNat]
      by_cases hp : 0 < u.re * v.im + u.im * v.re
      · left
        nlinarith
      · right
        by_contra hneg
        have had : u.re * v.im = 0 := by nlinarith [mul_nonneg ha hd, mul_nonneg hb hc]
        have hbc : u.im * v.re = 0 := by nlinarith [mul_nonneg ha hd, mul_nonneg hb hc]
        have hac : 0 < u.re * v.re := by nlinarith [mul_nonneg hb hd]
        have ha0 : 0 < u.re := lt_of_le_of_ne ha fun h => by
          rw [← h, zero_mul] at hac
          exact hac.false
        have hc0 : 0 < v.re := lt_of_le_of_ne hc fun h => by
          rw [← h, mul_zero] at hac
          exact hac.false
        have hd0 : v.im = 0 := (mul_eq_zero.mp had).resolve_left ha0.ne'
        have hb0 : u.im = 0 := (mul_eq_zero.mp hbc).resolve_right hc0.ne'
        rw [hb0] at hu2i
        rw [hd0] at hv2r
        exact hcut ⟨by linarith, by nlinarith⟩
  -- `i w - 2 i u v = i (u - v) ^ 2 = exp (π i / 2 - 2 log (u + v))`, whose imaginary part is in
  -- `[-π / 2, π / 2]` as `u + v` is in the closed first quadrant.
  have hlog : Complex.log (I * w + -2 * I * (u * v)) =
      (Real.pi : ℂ) / 2 * I - 2 * Complex.log (u + v) := by
    have hx : Complex.exp ((Real.pi : ℂ) / 2 * I - 2 * Complex.log (u + v)) =
        I * w + -2 * I * (u * v) := by
      rw [Complex.exp_sub, Complex.exp_pi_div_two_mul_I, two_mul, Complex.exp_add,
        Complex.exp_log hW, div_eq_iff (mul_ne_zero hW hW)]
      have hw : u ^ 2 + v ^ 2 = w := by
        rw [hu2, hv2]
        ring
      linear_combination (I * (u + v) ^ 2) * hw + (-I * ((u + v) * (u - v) + 1)) * hprod
    have harg0 : 0 ≤ Complex.arg (u + v) := Complex.arg_nonneg_iff.mpr (by simp; linarith)
    have harg1 : Complex.arg (u + v) ≤ Real.pi / 2 :=
      Complex.arg_le_pi_div_two_iff.mpr (Or.inl (by simp; linarith))
    have him_eq : ((Real.pi : ℂ) / 2 * I - 2 * Complex.log (u + v)).im =
        Real.pi / 2 - 2 * Complex.arg (u + v) := by
      simp [Complex.log_im]
    rw [← hx, Complex.log_exp]
    · rw [him_eq]
      linarith [Real.pi_pos]
    · rw [him_eq]
      linarith [Real.pi_pos]
  rw [hsq, hlog]
  linear_combination ((Real.pi : ℂ) / 2) * Complex.I_sq

/-- `w + √(1 + w²)` times `-w + √(1 + w²)` is `1`. -/
theorem asinh_mul (w : ℂ) :
    (w + (1 + w ^ 2) ^ (2⁻¹ : ℂ)) * (-w + (1 + w ^ 2) ^ (2⁻¹ : ℂ)) = 1 := by
  have h := Complex.cpow_nat_inv_pow (1 + w ^ 2) (n := 2) two_ne_zero
  rw [Nat.cast_ofNat] at h
  linear_combination h

/-- `w + √(1 + w²)` is never a negative real: its product with `-w + √(1 + w²)` is `1` and their
sum `2 √(1 + w²)` has a real part at least `0`. -/
theorem arg_asinh_ne_pi (w : ℂ) : (w + (1 + w ^ 2) ^ (2⁻¹ : ℂ)).arg ≠ Real.pi := by
  have hprod := congrArg Complex.re (asinh_mul w)
  have hr : 0 ≤ ((1 + w ^ 2) ^ (2⁻¹ : ℂ)).re := by
    rw [Complex.cpow_inv_two_re]
    exact Real.sqrt_nonneg _
  rw [Ne, Complex.arg_eq_pi_iff]
  rintro ⟨hre, him⟩
  rw [Complex.mul_re, him, zero_mul, sub_zero, Complex.one_re] at hprod
  have h' : (-w + (1 + w ^ 2) ^ (2⁻¹ : ℂ)).re < 0 := by nlinarith
  rw [Complex.add_re] at hre h'
  rw [Complex.neg_re] at h'
  linarith

/-- The principal expression of `asinh` is odd. -/
theorem value_asinh_neg (w : ℂ) : value .asinh (-w) = -value .asinh w := by
  simp only [value]
  rw [neg_sq, ← inv_eq_of_mul_eq_one_right (asinh_mul w), Complex.log_inv _ (arg_asinh_ne_pi w)]

/-- Off `{i y : 1 < |y|}`, the principal expression of `asinh` at `conj w` is the conjugate of its
value at `w`. -/
theorem value_asinh_conj {w : ℂ} (hcut : ¬(w.re = 0 ∧ 1 < |w.im|)) :
    value .asinh (starRingEnd ℂ w) = starRingEnd ℂ (value .asinh w) := by
  have harg : (1 + w ^ 2).arg ≠ Real.pi := by
    rw [Ne, Complex.arg_eq_pi_iff]
    rintro ⟨hre, him⟩
    simp only [Complex.add_re, Complex.add_im, Complex.one_re, Complex.one_im, sq,
      Complex.mul_re, Complex.mul_im] at hre him
    have h0 : w.re = 0 := by
      rcases mul_eq_zero.mp (show w.re * w.im = 0 by linarith) with h | h
      · exact h
      · rw [h] at hre
        nlinarith
    rw [h0] at hre
    exact hcut ⟨h0, (one_lt_sq_iff_one_lt_abs _).mp (by nlinarith)⟩
  have hc : 1 + starRingEnd ℂ w ^ 2 = starRingEnd ℂ (1 + w ^ 2) := by
    rw [map_add, map_one, map_pow]
  simp only [value]
  rw [hc, Complex.conj_cpow _ _ harg, map_inv₀, map_ofNat, ← map_add,
    Complex.log_conj _ (arg_asinh_ne_pi w)]

/-- `asin w` is `-i asinh (i w)`. -/
theorem value_asin (w : ℂ) : value .asin w = -I * value .asinh (I * w) := by
  simp only [value]
  rw [show 1 + (I * w) ^ 2 = 1 - w ^ 2 by
    rw [mul_pow, Complex.I_sq]
    ring]

/-- Off `{x : 1 < |x|}`, `π / 2 - asin` at `conj w` is the conjugate of its value at `w`. -/
theorem value_acos_conj {w : ℂ} (hcut : ¬(w.im = 0 ∧ 1 < |w.re|)) :
    value .acos (starRingEnd ℂ w) = starRingEnd ℂ (value .acos w) := by
  have hacos : ∀ v, value .acos v = Real.pi / 2 - value .asin v := fun v => by
    simp only [value]
    ring
  have hcut' : ¬((I * w).re = 0 ∧ 1 < |(I * w).im|) := by
    rwa [Complex.I_mul_re, Complex.I_mul_im, neg_eq_zero]
  rw [hacos, hacos, value_asin, value_asin, show I * starRingEnd ℂ w = -starRingEnd ℂ (I * w) by
      rw [map_mul, Complex.conj_I]
      ring,
    value_asinh_neg, value_asinh_conj hcut', map_sub, map_div₀, Complex.conj_ofReal, map_ofNat,
    map_mul, map_neg, Complex.conj_I]
  ring

/-- In the closed lower half-plane off `(-∞, -1)`, `π / 2 - asin w` is the conjugate of
`-i acosh (conj w)`. On `(1, ∞)` this is the limit from below the cut, `i acosh w`. -/
theorem value_acos_lower {w : ℂ} (him : w.im ≤ 0) (hcut : ¬(w.im = 0 ∧ w.re < -1)) :
    value .acos w = starRingEnd ℂ (-I * value .acosh (starRingEnd ℂ w)) := by
  by_cases hw : w.im = 0 ∧ 1 < |w.re|
  · have h1 : 1 < w.re := (lt_abs.mp hw.2).resolve_right fun h => hcut ⟨hw.1, by linarith⟩
    have hc : starRingEnd ℂ w = w := Complex.conj_eq_iff_im.mpr hw.1
    rw [hc, map_mul, map_neg, Complex.conj_I, neg_neg,
      ← value_acosh_conj fun h => by linarith [h.2], hc]
    -- On `(1, ∞)`: `√(1 - x²) = 2 i a b` with `a = √((x + 1) / 2)` and `b = √((x - 1) / 2)`, and
    -- `i x + 2 i a b = i (a + b) ^ 2`.
    obtain ⟨x, rfl⟩ : ∃ x : ℝ, w = x := ⟨w.re, Complex.ext rfl (by simpa using hw.1)⟩
    rw [Complex.ofReal_re] at h1
    obtain ⟨a, ha, ha2⟩ : ∃ a : ℝ, 0 < a ∧ a ^ 2 = (x + 1) / 2 :=
      ⟨_, Real.sqrt_pos.mpr (by linarith), Real.sq_sqrt (by linarith)⟩
    obtain ⟨b, hb, hb2⟩ : ∃ b : ℝ, 0 < b ∧ b ^ 2 = (x - 1) / 2 :=
      ⟨_, Real.sqrt_pos.mpr (by linarith), Real.sq_sqrt (by linarith)⟩
    have ha2' : ((x : ℂ) + 1) / 2 = (a : ℂ) ^ 2 := by exact_mod_cast ha2.symm
    have hb2' : ((x : ℂ) - 1) / 2 = (b : ℂ) ^ 2 := by exact_mod_cast hb2.symm
    have hsq : 1 - (x : ℂ) ^ 2 = (2 * I * (a * b)) ^ 2 := by
      linear_combination (-4 * (a : ℂ) ^ 2 * (b : ℂ) ^ 2) * Complex.I_sq -
        (4 * (a : ℂ) ^ 2) * hb2' - (2 * ((x : ℂ) - 1)) * ha2'
    have hsum : I * x + 2 * I * (a * b) = ((a + b) ^ 2 : ℝ) * I := by
      push_cast
      linear_combination I * ha2' + I * hb2'
    simp only [value]
    rw [ha2', hb2', hsq, ExecComplex.sq_cpow_inv_two (by simp) (Or.inr (by simp; positivity)),
      ExecComplex.sq_cpow_inv_two (by simpa using ha.le) (Or.inr (by simp)),
      ExecComplex.sq_cpow_inv_two (by simpa using hb.le) (Or.inr (by simp)), hsum,
      Complex.log_ofReal_mul (by positivity) Complex.I_ne_zero, Complex.log_I, Real.log_pow,
      ← Complex.ofReal_add, ← Complex.ofReal_log (by positivity)]
    push_cast
    linear_combination ((Real.pi : ℂ) / 2) * Complex.I_sq
  · have him' : 0 ≤ (starRingEnd ℂ w).im := by
      rw [Complex.conj_im]
      linarith
    have hcut' : ¬((starRingEnd ℂ w).im = 0 ∧ 1 < (starRingEnd ℂ w).re) := by
      rw [Complex.conj_im, Complex.conj_re, neg_eq_zero]
      exact fun h => hw ⟨h.1, h.2.trans_le (le_abs_self _)⟩
    rw [show -I * value .acosh (starRingEnd ℂ w) = value .acos (starRingEnd ℂ w) from
      (value_acos him' hcut').symm, value_acos_conj hw, Complex.conj_conj]

end ComplexFn

namespace ConfiguredBinary

/-- Reading back the pair of parts of a complex number gives the number. -/
@[simp] theorem toExecComplex_ofExecComplex (w : ExecComplex format) :
    toExecComplex (ofExecComplex (plan := plan) (code := code) w) = w := by
  simp [toExecComplex, ofExecComplex]

/-- FloatLib's exact result of each function, the result of `complexApply` before rounding. -/
noncomputable def complexExact : ComplexFn → ExecComplex format → ExecComplex.Exact
  | .exp => ExecComplex.expExact
  | .expm1 => ExecComplex.expm1Exact
  | .log => ExecComplex.logExact
  | .log1p => ExecComplex.log1pExact
  | .sqrt => ExecComplex.sqrtExact
  | .rsqrt => ExecComplex.rsqrtExact
  | .sin => ExecComplex.sinExact
  | .cos => ExecComplex.cosExact
  | .tan => ExecComplex.tanExact
  | .sinh => ExecComplex.sinhExact
  | .cosh => ExecComplex.coshExact
  | .tanh => ExecComplex.tanhExact
  | .asin => ExecComplex.asinExact
  | .acos => ExecComplex.acosExact
  | .atan => ExecComplex.atanExact
  | .asinh => ExecComplex.asinhExact
  | .acosh => ExecComplex.acoshExact
  | .atanh => ExecComplex.atanhExact

/-- The argument is none of the points that FloatLib's proof of rounding once leaves out: `0` for
`expm1`, `log` and `rsqrt`, `0` and `-1` for `log1p`, `±1` for `atanh` and `±i` for `atan`. -/
def complexRegular (f : ComplexFn) (z : ExecComplex format) : Prop :=
  match f with
  | .expm1 | .log | .rsqrt => ExecComplex.toComplex z ≠ 0
  | .log1p => ExecComplex.toComplex z ≠ 0 ∧ 1 + ExecComplex.toComplex z ≠ 0
  | .atanh => ExecComplex.toComplex z ≠ 1 ∧ ExecComplex.toComplex z ≠ -1
  | .atan => ExecComplex.toComplex z ≠ Complex.I ∧ ExecComplex.toComplex z ≠ -Complex.I
  | _ => True

/-- Where FloatLib proves that the exact result of `f` is `ComplexFn.value f`: everywhere for
`exp`, `expm1`, `sin`, `cos`, `tan`, `sinh`, `cosh` and `tanh`, and off the branch cut for the
others. On the cut it is where the imaginary sign bit is clear for `sqrt`, `rsqrt`, `log`,
`log1p`, `atanh` and `acosh`, where the real one is clear for `atan`, where the two agree for
`asinh`, and where they differ for `asin` and `acos`. -/
def complexPrincipal (f : ComplexFn) (z : ExecComplex format) : Prop :=
  match f with
  | .sqrt | .rsqrt | .log => Model.signBit z.im = false ∨ (ExecComplex.toComplex z).arg ≠ Real.pi
  | .log1p => Model.signBit z.im = false ∨ (1 + ExecComplex.toComplex z).arg ≠ Real.pi
  | .atanh => Model.signBit z.im = false ∨
      ¬((ExecComplex.toComplex z).im = 0 ∧ 1 < |(ExecComplex.toComplex z).re|)
  | .atan => Model.signBit z.re = false ∨
      ¬((ExecComplex.toComplex z).re = 0 ∧ 1 < |(ExecComplex.toComplex z).im|)
  | .asinh => Model.signBit z.re = Model.signBit z.im ∨
      ¬((ExecComplex.toComplex z).re = 0 ∧ 1 < |(ExecComplex.toComplex z).im|)
  | .asin | .acos => Model.signBit z.re ≠ Model.signBit z.im ∨
      ¬((ExecComplex.toComplex z).im = 0 ∧ 1 < |(ExecComplex.toComplex z).re|)
  | .acosh => Model.signBit z.im = false ∨
      ¬((ExecComplex.toComplex z).im = 0 ∧ (ExecComplex.toComplex z).re < 1)
  | _ => True

/-- In an IEEE format, at a finite argument whose sign bits agree or which is off
`{i y : 1 < |y|}`, FloatLib's exact `asinh` is the principal expression. -/
theorem complexExact_asinh (hfmt : format.isIEEE = true) {z : ExecComplex format}
    (hz : ExecComplex.isFinite z = true)
    (hv : complexPrincipal .asinh z) :
    (ExecComplex.asinhExact z).value = ComplexFn.value .asinh (ExecComplex.toComplex z) := by
  have hcore : ∀ ζ : ExecComplex format, Model.signBit ζ.re = false →
      Model.signBit ζ.im = false →
      ExecComplex.asinhValue (ExecComplex.toComplex ζ) =
        ComplexFn.value .asinh (ExecComplex.toComplex ζ) := fun ζ hr hi => by
    have hx : 0 ≤ (ExecComplex.toComplex ζ).re := Model.toReal_nonneg_of_signBit_eq_false hr
    have hy : 0 ≤ (ExecComplex.toComplex ζ).im := Model.toReal_nonneg_of_signBit_eq_false hi
    exact ExecComplex.asinhValue_eq_log hx fun h => by linarith [h.2]
  have hneg : ∀ x : Model format, Model.signBit x = true → Model.signBit (Model.neg x) = false :=
    fun x h => by simp [ExecComplex.signBit_neg hfmt, h]
  have hci : ∀ hi : Model.signBit z.im = true, Model.signBit (ExecComplex.conj z).im = false :=
    fun hi => hneg _ hi
  cases hr : Model.signBit z.re <;> cases hi : Model.signBit z.im
  · exact ExecComplex.asinhExact_value hr hi
  · have hcr : Model.signBit (ExecComplex.conj z).re = false := hr
    simp only [ExecComplex.asinhExact, ExecComplex.Exact.ofQuadrant,
      ExecComplex.Exact.ofUpperHalf, hi, hcr, Bool.false_eq_true, ↓reduceIte,
      ExecComplex.Exact.conj]
    rw [hcore _ hcr (hci hi), ExecComplex.toComplex_conj z hz,
      ComplexFn.value_asinh_conj (hv.resolve_left (by simp [hr, hi])), Complex.conj_conj]
  · simp only [ExecComplex.asinhExact, ExecComplex.Exact.ofQuadrant,
      ExecComplex.Exact.ofUpperHalf, hi, hr, Bool.false_eq_true, ↓reduceIte]
    rw [hcore ⟨Model.neg z.re, z.im⟩ (hneg _ hr) hi, ExecComplex.toComplex_neg_re hz,
      show (⟨-(ExecComplex.toComplex z).re, (ExecComplex.toComplex z).im⟩ : ℂ) =
        -starRingEnd ℂ (ExecComplex.toComplex z) from Complex.ext (by simp) (by simp),
      ComplexFn.value_asinh_neg, ComplexFn.value_asinh_conj (hv.resolve_left (by simp [hr, hi]))]
    apply Complex.ext <;> simp
  · have hcr : Model.signBit (ExecComplex.conj z).re = true := hr
    simp only [ExecComplex.asinhExact, ExecComplex.Exact.ofQuadrant,
      ExecComplex.Exact.ofUpperHalf, hi, hcr, ↓reduceIte,
      ExecComplex.Exact.conj]
    rw [hcore ⟨Model.neg (ExecComplex.conj z).re, (ExecComplex.conj z).im⟩ (hneg _ hcr) (hci hi),
      ExecComplex.toComplex_neg_re (ExecComplex.isFinite_conj hz), ExecComplex.toComplex_conj z hz,
      show (⟨-(starRingEnd ℂ (ExecComplex.toComplex z)).re,
        (starRingEnd ℂ (ExecComplex.toComplex z)).im⟩ : ℂ) = -ExecComplex.toComplex z from
        Complex.ext (by simp) (by simp),
      ComplexFn.value_asinh_neg]
    apply Complex.ext <;> simp

/-- In an IEEE format, at a finite argument that `complexRegular f` allows, each part of
`complexApply f z` is the part of the exact result `complexExact f z` rounded once to nearest
even. -/
theorem complexApply_eq_round (hfmt : format.isIEEE = true) {f : ComplexFn}
    {z : ExecComplex format} (hz : ExecComplex.isFinite z = true) (h : complexRegular f z) :
    complexApply f z = (complexExact f z).round format .nearest := by
  cases f
  case exp => exact ExecComplex.exp_eq hfmt hz
  case expm1 => exact ExecComplex.expm1_eq hfmt hz h
  case log => exact ExecComplex.log_eq hfmt hz h
  case log1p => exact ExecComplex.log1p_eq hfmt hz h.1 h.2
  case sqrt => exact ExecComplex.sqrt_eq hfmt hz
  case rsqrt => exact ExecComplex.rsqrt_eq hfmt hz h
  case sin => exact ExecComplex.sin_eq hfmt hz
  case cos => exact ExecComplex.cos_eq hfmt hz
  case tan => exact ExecComplex.tan_eq hfmt hz
  case sinh => exact ExecComplex.sinh_eq hfmt hz
  case cosh => exact ExecComplex.cosh_eq hfmt hz
  case tanh => exact ExecComplex.tanh_eq hfmt hz
  case asin => exact ExecComplex.asin_eq hfmt hz
  case acos => exact ExecComplex.acos_eq hfmt hz
  case atan => exact ExecComplex.atan_eq hfmt hz h.1 h.2
  case asinh => exact ExecComplex.asinh_eq hfmt hz
  case acosh => exact ExecComplex.acosh_eq hfmt hz
  case atanh => exact ExecComplex.atanh_eq hfmt hz h.1 h.2

/-- In an IEEE format, at a finite argument that `complexRegular f` and `complexPrincipal f`
allow, the exact result of `f` is `ComplexFn.value f` of the argument. -/
theorem complexExact_value (hfmt : format.isIEEE = true) {f : ComplexFn}
    {z : ExecComplex format} (hz : ExecComplex.isFinite z = true) (h : complexRegular f z)
    (hv : complexPrincipal f z) : (complexExact f z).value = f.value (ExecComplex.toComplex z) := by
  cases f
  case exp => exact ExecComplex.expExact_value hz
  case expm1 => exact ExecComplex.expm1Exact_value hz
  case log => exact ExecComplex.logExact_value hz hv
  case log1p => exact ExecComplex.log1pExact_value hz hv
  case sqrt => exact ExecComplex.sqrtExact_value hz hv
  case rsqrt => exact ExecComplex.rsqrtExact_value hz hv
  case sin => exact ExecComplex.sinExact_value hz
  case cos => exact ExecComplex.cosExact_value hz
  case tan => exact ExecComplex.tanExact_value hz
  case sinh => exact ExecComplex.sinhExact_value hz
  case cosh => exact ExecComplex.coshExact_value hz
  case tanh => exact ExecComplex.tanhExact_value hz
  case asin =>
    change (ExecComplex.asinExact z).value = _
    have h := complexExact_asinh hfmt (ExecComplex.isFinite_mulI hz) (z := ExecComplex.mulI z) (by
      show Model.signBit (Model.neg z.im) = Model.signBit z.re ∨
        ¬((ExecComplex.toComplex (ExecComplex.mulI z)).re = 0 ∧
          1 < |(ExecComplex.toComplex (ExecComplex.mulI z)).im|)
      rw [ExecComplex.signBit_neg hfmt, ExecComplex.toComplex_mulI hz, Complex.I_mul_re,
        Complex.I_mul_im, neg_eq_zero]
      refine hv.imp (fun h => ?_) id
      revert h
      cases Model.signBit z.re <;> cases Model.signBit z.im <;> decide)
    rw [ExecComplex.toComplex_mulI hz] at h
    simp only [ExecComplex.asinExact, ExecComplex.Exact.mulNegI]
    rw [h, ComplexFn.value_asin]
  case acos =>
    change (ExecComplex.acosExact z).value = _
    cases hi : Model.signBit z.im
    · refine (ExecComplex.acosExact_value hi).trans
        (ComplexFn.value_acos (Model.toReal_nonneg_of_signBit_eq_false hi) ?_).symm
      rintro ⟨h0, h1⟩
      rcases hv with hr | hcut
      · have hr' : Model.signBit z.re = true := by simpa [hi] using hr
        have h := Model.toReal_nonneg_of_signBit_eq_false (x := Model.neg z.re)
          (by simp [ExecComplex.signBit_neg hfmt, hr'])
        rw [Model.toReal_neg _ ((ExecComplex.isFinite_eq_true_iff z).1 hz).1] at h
        have : (ExecComplex.toComplex z).re = Model.toReal z.re := rfl
        linarith
      · exact hcut ⟨h0, h1.trans_le (le_abs_self _)⟩
    · have hc : Model.signBit (ExecComplex.conj z).im = false := by
        simp [ExecComplex.conj, ExecComplex.signBit_neg hfmt, hi]
      have him : (ExecComplex.toComplex z).im ≤ 0 := by
        have h : 0 ≤ (ExecComplex.toComplex (ExecComplex.conj z)).im :=
          Model.toReal_nonneg_of_signBit_eq_false hc
        rw [ExecComplex.toComplex_conj z hz, Complex.conj_im] at h
        linarith
      have hcut : ¬((ExecComplex.toComplex z).im = 0 ∧ (ExecComplex.toComplex z).re < -1) := by
        rintro ⟨h0, h1⟩
        rcases hv with hr | hcut
        · have h : 0 ≤ (ExecComplex.toComplex z).re :=
            Model.toReal_nonneg_of_signBit_eq_false (by simpa [hi] using hr)
          linarith
        · exact hcut ⟨h0, by rw [abs_of_neg (by linarith)]; linarith⟩
      have hconj := ExecComplex.acosExact_value hc
      simp only [ExecComplex.acosExact, ExecComplex.Exact.ofUpperHalf, hc, hi, Bool.false_eq_true,
        ↓reduceIte, ExecComplex.Exact.conj] at hconj ⊢
      rw [hconj, ExecComplex.toComplex_conj z hz]
      exact (ComplexFn.value_acos_lower him hcut).symm
  case atan => exact ExecComplex.atanExact_value hz hv h.2
  case asinh => exact complexExact_asinh hfmt hz hv
  case acosh =>
    cases hs : Model.signBit z.im
    · exact ExecComplex.acoshExact_value hs
    · have hc : Model.signBit (ExecComplex.conj z).im = false := by
        simp [ExecComplex.conj, ExecComplex.signBit_neg hfmt, hs]
      have hconj := ExecComplex.acoshExact_value hc
      change (ExecComplex.acoshExact z).value = _
      simp only [ExecComplex.acoshExact, ExecComplex.Exact.ofUpperHalf, hc, hs,
        Bool.false_eq_true, ↓reduceIte, ExecComplex.Exact.conj] at hconj ⊢
      rw [hconj, ExecComplex.toComplex_conj z hz]
      change starRingEnd ℂ (ComplexFn.value .acosh (starRingEnd ℂ (ExecComplex.toComplex z))) = _
      rw [ComplexFn.value_acosh_conj (hv.resolve_left (by simp [hs])), Complex.conj_conj]
  case atanh => exact ExecComplex.atanhExact_value hz hv h.1

/-- At a finite base whose imaginary sign bit is clear or which is off the negative real axis,
and a finite exponent, the exact power is Mathlib's `z ^ w`. -/
theorem powExact_value {z w : ExecComplex format} (hz : ExecComplex.isFinite z = true)
    (hw : ExecComplex.isFinite w = true)
    (hside : Model.signBit z.im = false ∨ (ExecComplex.toComplex z).arg ≠ Real.pi) :
    (ExecComplex.powExact z w).value = ExecComplex.toComplex z ^ ExecComplex.toComplex w := by
  unfold ExecComplex.powExact
  cases hs : Model.signBit z.im
  · rfl
  · have harg := hside.resolve_left (by simp [hs])
    simp only [↓reduceIte, ExecComplex.Exact.conj]
    rw [ExecComplex.toComplex_conj z hz, ExecComplex.toComplex_conj w hw,
      Complex.conj_cpow _ _ harg, Complex.conj_conj, Complex.conj_conj]

end ConfiguredBinary

namespace Op

open ConfiguredBinary

/-- In an IEEE configured format, at each entry `c` whose complex number is finite and allowed by
`complexRegular f`, each part of the result of `Op.complex f` is the part of FloatLib's exact
result `complexExact f` rounded once to nearest even. -/
theorem complex_forward_round (hfmt : format.isIEEE = true) (f : ComplexFn) {s : Shape}
    (x : Tensor Value (s ++ [2])) (c : s.Coord)
    (hz : ExecComplex.isFinite (toExecComplex (x.pairAt c)) = true)
    (h : complexRegular f (toExecComplex (x.pairAt c))) :
    ∃ rule, (Op.complex f (s := s)).rule? (α := Value) = some rule ∧
      toExecComplex ((rule.forward (x, ())).1.pairAt c) =
        (complexExact f (toExecComplex (x.pairAt c))).round format .nearest := by
  refine ⟨_, rfl, ?_⟩
  change toExecComplex ((Tensor.ofPairs fun d =>
    ComplexFunctions.apply f (x.pairAt d)).pairAt c) = _
  rw [Tensor.pairAt_ofPairs]
  change toExecComplex (ofExecComplex (complexApply f (toExecComplex (x.pairAt c)))) = _
  rw [toExecComplex_ofExecComplex]
  exact complexApply_eq_round hfmt hz h

/-- In an IEEE configured format, at each entry `c` whose complex number `z` is finite and allowed
by `complexRegular f` and `complexPrincipal f`, each part of the result of `Op.complex f` is the
part of `ComplexFn.value f z` rounded once to nearest even, with the signs of zero parts of
FloatLib's exact result. -/
theorem complex_forward_value (hfmt : format.isIEEE = true) (f : ComplexFn) {s : Shape}
    (x : Tensor Value (s ++ [2])) (c : s.Coord)
    (hz : ExecComplex.isFinite (toExecComplex (x.pairAt c)) = true)
    (h : complexRegular f (toExecComplex (x.pairAt c)))
    (hv : complexPrincipal f (toExecComplex (x.pairAt c))) :
    ∃ rule, (Op.complex f (s := s)).rule? (α := Value) = some rule ∧ ∃ e : ExecComplex.Exact,
      e.value = f.value (ExecComplex.toComplex (toExecComplex (x.pairAt c))) ∧
        toExecComplex ((rule.forward (x, ())).1.pairAt c) = e.round format .nearest := by
  obtain ⟨rule, hrule, hround⟩ := complex_forward_round hfmt f x c hz h
  exact ⟨rule, hrule, _, complexExact_value hfmt hz h hv, hround⟩

/-- In an IEEE configured format, at each entry `c` with a finite base other than zero and a finite
exponent where FloatLib certifies the power, each part of the result of `Op.complexPow` is the
part of the exact power `ExecComplex.powExact` rounded once to nearest even. -/
theorem complexPow_forward_round (hfmt : format.isIEEE = true) {s : Shape}
    (x y : Tensor Value (s ++ [2])) (c : s.Coord)
    (hz : ExecComplex.isFinite (toExecComplex (x.pairAt c)) = true)
    (hw : ExecComplex.isFinite (toExecComplex (y.pairAt c)) = true)
    (hne : ExecComplex.toComplex (toExecComplex (x.pairAt c)) ≠ 0)
    (hcert : ExecComplex.powCertified (toExecComplex (x.pairAt c))
      (toExecComplex (y.pairAt c)) = true) :
    ∃ rule, (Op.complexPow (s := s)).rule? (α := Value) = some rule ∧
      toExecComplex ((rule.forward (x, y, ())).1.pairAt c) =
        (ExecComplex.powExact (toExecComplex (x.pairAt c)) (toExecComplex (y.pairAt c))).round
          format .nearest := by
  refine ⟨_, rfl, ?_⟩
  change toExecComplex ((Tensor.ofPairs fun d =>
    ComplexFunctions.pow (x.pairAt d) (y.pairAt d)).pairAt c) = _
  rw [Tensor.pairAt_ofPairs]
  change toExecComplex (ofExecComplex
    (ExecComplex.pow (toExecComplex (x.pairAt c)) (toExecComplex (y.pairAt c)))) = _
  rw [toExecComplex_ofExecComplex]
  exact ExecComplex.pow_eq hfmt hz hw hne (h := hcert)

/-- In an IEEE configured format, at each entry `c` with a finite base `z` other than zero and a
finite exponent `w` where FloatLib certifies the power, and where the imaginary sign bit of `z` is
clear or `z` is off the negative real axis, each part of the result of `Op.complexPow` is the part
of `z ^ w` rounded once to nearest even, with the signs of zero parts of FloatLib's exact power. -/
theorem complexPow_forward_value (hfmt : format.isIEEE = true) {s : Shape}
    (x y : Tensor Value (s ++ [2])) (c : s.Coord)
    (hz : ExecComplex.isFinite (toExecComplex (x.pairAt c)) = true)
    (hw : ExecComplex.isFinite (toExecComplex (y.pairAt c)) = true)
    (hne : ExecComplex.toComplex (toExecComplex (x.pairAt c)) ≠ 0)
    (hcert : ExecComplex.powCertified (toExecComplex (x.pairAt c))
      (toExecComplex (y.pairAt c)) = true)
    (hside : Model.signBit (toExecComplex (x.pairAt c)).im = false ∨
      (ExecComplex.toComplex (toExecComplex (x.pairAt c))).arg ≠ Real.pi) :
    ∃ rule, (Op.complexPow (s := s)).rule? (α := Value) = some rule ∧ ∃ e : ExecComplex.Exact,
      e.value = ExecComplex.toComplex (toExecComplex (x.pairAt c)) ^
          ExecComplex.toComplex (toExecComplex (y.pairAt c)) ∧
        toExecComplex ((rule.forward (x, y, ())).1.pairAt c) = e.round format .nearest := by
  obtain ⟨rule, hrule, hround⟩ := complexPow_forward_round hfmt x y c hz hw hne hcert
  exact ⟨rule, hrule, _, powExact_value hz hw hside, hround⟩

end Op

end TorchLean
