/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ConfiguredBinary.Elementary
public import TorchLean.Verify.RuntimeApprox.InverseTrigonometric
public import FloatLib.Numerics.Exact.Hyperbolic.Proof
public import FloatLib.Numerics.Exact.RationalPower.Proof
public import FloatLib.Floats.Formats.BinaryInterchange.Transcendentals.CorrectlyRounded.Proof

/-!
# Rounding of the elementary functions of binary values

For an IEEE descriptor and a finite input in the domain, a finite result of `Elementary.tan`,
`exp2`, `logb`, `asinh`, `acosh` and `atanh` is the nearest-even rounding of the real function
(`toReal_tan`, `toReal_exp2`, `toReal_logb`, `toReal_asinh`, `toReal_acosh`, `toReal_atanh`).
`asinh` and `acosh` of a finite input are always finite (`isFinite_asinh`, `isFinite_acosh`):
`arsinh |a| ≤ |a|` and `arcosh a ≤ a`. In every format `exp2` overflows at large enough finite
inputs (at `128` and above in binary32), and `logb` at a positive input other than one once the
base is close enough to one. `tan` and `atanh` can overflow where the format's range is small for
its precision. In the domain, `tan`, `exp2`, `logb` and `atanh` are finite when the real function
is below the overflow threshold in magnitude (`isFinite_tan`, `isFinite_exp2`, `isFinite_logb`,
`isFinite_atanh`). A finite `logb`, `acosh` or `atanh` of a finite input has its input in the
domain (`toReal_pos_of_isFinite_logb`, `one_le_toReal_of_isFinite_acosh`,
`abs_toReal_lt_one_of_isFinite_atanh`): outside it the result is an infinity or the invalid result.
The special values: a NaN is quieted, and `tan`, `asinh` and `atanh` keep a zero with its sign,
as equations of values. The `ConfiguredBinary` theorems state the rounding on configured binary
values.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Model
open FloatLib.Numerics

namespace Elementary

variable {fmt : FloatFormat}

open InverseTrigonometric (levels roundAt_signed_odd round_cmp_zero)

/-- The inverse hyperbolic tangent is odd. -/
private theorem artanh_neg (x : ℝ) : Real.artanh (-x) = -Real.artanh x := by
  rw [Real.artanh, Real.artanh, show (1 + -x) / (1 - -x) = ((1 + x) / (1 - x))⁻¹ by
    rw [inv_div]; ring, Real.sqrt_inv, Real.log_inv]

/-- Comparing `-t` with `b` is comparing `-b` with `t`. -/
private theorem cmp_neg_left (t b : ℝ) : cmp (-t) b = cmp (-b) t := by
  simp only [cmp, cmpUsing, neg_lt, lt_neg]

/-- The inverse hyperbolic tangent of `|x|` is the magnitude of that of `x`. -/
private theorem artanh_abs (x : ℝ) : Real.artanh |x| = |Real.artanh x| := by
  rcases le_total 0 x with h | h
  · rw [abs_of_nonneg h, abs_of_nonneg (Real.artanh_nonneg h)]
  · rw [abs_of_nonpos h, abs_of_nonpos (Real.artanh_nonpos h), artanh_neg]

/-- At a finite nonzero input of value `a`, `tan` is comparator rounding of `|tan a|` with the
sign of `tan a`. -/
theorem tan_eq_round {x : Model fmt} {a : ℚ} (hx : isFinite x = true)
    (hzero : ¬isZero x = true) (hdecode : toRat? x = some a) :
    tan x = if (TrigonometricComparison.prepareTan a levels).compare 0 = .lt then
        AlgebraicRounding.round fmt true
          fun b => ((TrigonometricComparison.prepareTan a levels).compare (-b)).swap
      else AlgebraicRounding.round fmt false
        (TrigonometricComparison.prepareTan a levels).compare := by
  simp [tan, roundSigned, isNaN_eq_false_of_isFinite_eq_true x hx, hzero, hdecode]

/-- At a finite input, `tan` is finite when the tangent is below the overflow threshold in
magnitude. -/
theorem isFinite_tan (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hrange : |Real.tan (toReal x)| < Transcendentals.Certified.overflowThreshold fmt) :
    isFinite (tan x) = true := by
  by_cases hzero : isZero x = true
  · rw [show tan x = x by simp [tan, isNaN_eq_false_of_isFinite_eq_true x hx, hzero]]
    exact hx
  · obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
    have hcompare := TrigonometricComparison.prepareTan_eq_real a levels
    rw [hreal, Transcendentals.CorrectlyRounded.cast_overflowThreshold] at hrange
    rw [tan_eq_round hx hzero hdecode]
    split
    · exact AlgebraicRounding.isFinite_round_of_lt_overflow fmt hfmt _ _ (-Real.tan (a : ℝ))
        (fun b => by rw [hcompare, cmp_swap, Rat.cast_neg, cmp_neg_left])
        ((neg_le_abs _).trans_lt hrange)
    · exact AlgebraicRounding.isFinite_round_of_lt_overflow fmt hfmt _ _ (Real.tan (a : ℝ))
        hcompare ((le_abs_self _).trans_lt hrange)

/-- At a finite input with a finite result, `tan` is the tangent rounded once. -/
theorem toReal_tan (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hfinite : isFinite (tan x) = true) :
    toReal (tan x) = roundAt fmt (Real.tan (toReal x)) := by
  by_cases hzero : isZero x = true
  · have heq : tan x = x := by simp [tan, isNaN_eq_false_of_isFinite_eq_true x hx, hzero]
    rw [heq, toReal_eq_zero_of_isZero x hzero, Real.tan_zero, roundAt_zero]
  · obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
    have hcompare := TrigonometricComparison.prepareTan_eq_real a levels
    rw [tan_eq_round hx hzero hdecode] at hfinite ⊢
    rw [hreal]
    by_cases hnegative : (TrigonometricComparison.prepareTan a levels).compare 0 = .lt
    · rw [ite_eq_left hnegative] at hfinite ⊢
      rw [hcompare, cmp_eq_lt_iff, Rat.cast_zero] at hnegative
      rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ (-Real.tan (a : ℝ))
        (by linarith) (fun b => by rw [hcompare, cmp_swap, Rat.cast_neg, cmp_neg_left])
        hfinite]
      simp
    · rw [ite_eq_right hnegative] at hfinite ⊢
      rw [hcompare, cmp_eq_lt_iff, Rat.cast_zero, not_lt] at hnegative
      rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ (Real.tan (a : ℝ))
        hnegative hcompare hfinite]
      simp

/-- At a finite input of value `a`, `exp2` is comparator rounding of `2 ^ a`. -/
theorem exp2_eq_round {x : Model fmt} {a : ℚ} (hx : isFinite x = true)
    (hdecode : toRat? x = some a) :
    exp2 x = AlgebraicRounding.round fmt false (RationalPower.compare 2 a) := by
  simp [exp2, isNaN_eq_false_of_isFinite_eq_true x hx, hdecode]

/-- At a finite input, `exp2` is finite when `2 ^ x` is below the overflow threshold. -/
theorem isFinite_exp2 (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hrange : (2 : ℝ) ^ toReal x < Transcendentals.Certified.overflowThreshold fmt) :
    isFinite (exp2 x) = true := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  rw [exp2_eq_round hx hdecode]
  refine AlgebraicRounding.isFinite_round_of_lt_overflow fmt hfmt _ _ ((2 : ℝ) ^ (a : ℝ))
    (fun b => by rw [RationalPower.compare_eq_real 2 a b (by norm_num)]; norm_num) ?_
  rw [← Transcendentals.CorrectlyRounded.cast_overflowThreshold, ← hreal]
  exact hrange

/-- At a finite input with a finite result, `exp2` is `2 ^ x` rounded once. -/
theorem toReal_exp2 (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hfinite : isFinite (exp2 x) = true) :
    toReal (exp2 x) = roundAt fmt ((2 : ℝ) ^ toReal x) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  rw [exp2_eq_round hx hdecode] at hfinite ⊢
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ ((2 : ℝ) ^ (a : ℝ))
    (Real.rpow_nonneg (by norm_num) _)
    (fun b => by rw [RationalPower.compare_eq_real 2 a b (by norm_num)]; norm_num) hfinite,
    hreal]
  simp

/-- At a positive finite input, `logb base` for a base above one is comparator rounding of
`|logb base x|`, negative below one. -/
private theorem logb_eq_round {base : ℚ} (hbase : 1 < base) {x : Model fmt}
    (hx : isFinite x = true) (hpositive : 0 < toReal x) :
    ∃ compare : ℚ → Ordering,
      logb base x = AlgebraicRounding.round fmt (decide (toReal x < 1)) compare ∧
        ∀ b : ℚ, compare b = cmp |Real.logb base (toReal x)| (b : ℝ) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha : 0 < a := by exact_mod_cast (hreal ▸ hpositive : (0 : ℝ) < a)
  have ha' : (0 : ℝ) < a := by exact_mod_cast ha
  have hzero : isZero x = false := by
    rw [Bool.eq_false_iff]
    intro h
    rw [toReal_eq_zero_of_isZero x h] at hpositive
    exact lt_irrefl _ hpositive
  have hbaseReal : (1 : ℝ) < base := by exact_mod_cast hbase
  have hmax : (1 : ℝ) ≤ ((max a a⁻¹ : ℚ) : ℝ) := by
    have : (1 : ℚ) ≤ max a a⁻¹ := by
      rcases le_or_gt 1 a with h | h
      · exact le_max_of_le_left h
      · exact le_max_of_le_right (one_le_inv₀ ha |>.mpr h.le)
    exact_mod_cast this
  have hm : (0 : ℝ) < ((max a a⁻¹ : ℚ) : ℝ) := by linarith
  have habs : Real.logb base ((max a a⁻¹ : ℚ) : ℝ) = |Real.logb base (a : ℝ)| := by
    rcases lt_or_ge a 1 with h | h
    · have h' : (a : ℝ) < 1 := by exact_mod_cast h
      rw [max_eq_right (le_trans h.le (one_le_inv₀ ha |>.mpr h.le)), Rat.cast_inv,
        Real.logb_inv, abs_of_neg (Real.logb_neg hbaseReal ha' h')]
    · have h' : (1 : ℝ) ≤ a := by exact_mod_cast h
      rw [max_eq_left (le_trans (inv_le_one_of_one_le₀ h) h),
        abs_of_nonneg (Real.logb_nonneg hbaseReal h')]
  have hsign : (a : ℝ) < 1 ↔ a < 1 := by norm_cast
  refine ⟨fun b => (RationalPower.compare base b (max a a⁻¹)).swap, ?_, fun b => ?_⟩
  · rw [hreal]
    simp [logb, isNaN_eq_false_of_isFinite_eq_true x hx, hzero, hdecode, ha, hsign]
  · show (RationalPower.compare base b (max a a⁻¹)).swap = _
    rw [RationalPower.compare_eq_real base b _ (by linarith), cmp_swap, hreal, ← habs]
    simp only [cmp, cmpUsing, Real.logb_lt_iff_lt_rpow hbaseReal hm,
      Real.lt_logb_iff_rpow_lt hbaseReal hm]

/-- At a positive finite input with a finite result, `logb base` for a base above one is the
logarithm to that base rounded once. -/
theorem toReal_logb (hfmt : fmt.isIEEE = true) {base : ℚ} (hbase : 1 < base) {x : Model fmt}
    (hx : isFinite x = true) (hpositive : 0 < toReal x)
    (hfinite : isFinite (logb base x) = true) :
    toReal (logb base x) = roundAt fmt (Real.logb base (toReal x)) := by
  obtain ⟨compare, heq, hcompare⟩ := logb_eq_round hbase hx hpositive
  rw [heq] at hfinite ⊢
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ _ (abs_nonneg _) hcompare hfinite]
  have hbaseReal : (1 : ℝ) < base := by exact_mod_cast hbase
  rcases lt_or_ge (toReal x) 1 with h | h
  · simp [h, abs_of_neg (Real.logb_neg hbaseReal hpositive h)]
  · simp [not_lt.mpr h, abs_of_nonneg (Real.logb_nonneg hbaseReal h)]

/-- At a positive finite input, `logb base` for a base above one is finite when the logarithm is
below the overflow threshold in magnitude. -/
theorem isFinite_logb (hfmt : fmt.isIEEE = true) {base : ℚ} (hbase : 1 < base) {x : Model fmt}
    (hx : isFinite x = true) (hpositive : 0 < toReal x)
    (hrange : |Real.logb base (toReal x)| < Transcendentals.Certified.overflowThreshold fmt) :
    isFinite (logb base x) = true := by
  obtain ⟨compare, heq, hcompare⟩ := logb_eq_round hbase hx hpositive
  rw [heq]
  exact AlgebraicRounding.isFinite_round_of_lt_overflow fmt hfmt _ _ _ hcompare
    (Transcendentals.CorrectlyRounded.cast_overflowThreshold fmt ▸ hrange)

/-- A finite value is below the overflow threshold in magnitude. -/
private theorem abs_toReal_lt_overflow (hfmt : fmt.isIEEE = true) {x : Model fmt}
    (hx : isFinite x = true) :
    |toReal x| < FloatLib.Floats.Formats.Flocq.bpow binaryRadix (fmt.maxNormalExponent + 1) -
      FloatLib.Floats.Formats.Flocq.bpow binaryRadix
        (fmt.maxNormalExponent - fmt.fracWidth - 1) := by
  rw [← Transcendentals.CorrectlyRounded.cast_overflowThreshold]
  exact Transcendentals.CorrectlyRounded.abs_lt_overflowThreshold_of_toReal_eq_roundAt hfmt hx
    (roundAt_toReal_eq x hx).symm

/-- At a finite input that decodes to `a`, `asinh` rounds `arsinh |a|` through its comparator,
with the sign of `x`. -/
theorem asinh_eq_round {x : Model fmt} {a : ℚ} (hx : isFinite x = true)
    (hdecode : toRat? x = some a) :
    asinh x = AlgebraicRounding.round fmt (signBit x)
      (HyperbolicComparison.prepareArsinh |a| levels).compare := by
  simp [asinh, isNaN_eq_false_of_isFinite_eq_true x hx, hdecode]

/-- `asinh` of a finite input is finite, since `arsinh |a| ≤ |a|`. -/
theorem isFinite_asinh (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true) :
    isFinite (asinh x) = true := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  rw [asinh_eq_round hx hdecode]
  have hle : Real.arsinh |(a : ℝ)| ≤ |(a : ℝ)| := by
    conv_rhs => rw [← Real.arsinh_sinh |(a : ℝ)|]
    exact Real.arsinh_le_arsinh.mpr (Real.self_le_sinh_iff.mpr (abs_nonneg _))
  exact AlgebraicRounding.isFinite_round_of_lt_overflow fmt hfmt _ _ (Real.arsinh |(a : ℝ)|)
    (fun b => by rw [HyperbolicComparison.prepareArsinh_eq_real]; simp)
    (lt_of_le_of_lt hle (hreal ▸ abs_toReal_lt_overflow hfmt hx))

/-- At a finite input, `asinh` is the inverse hyperbolic sine rounded once. -/
theorem toReal_asinh (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true) :
    toReal (asinh x) = roundAt fmt (Real.arsinh (toReal x)) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have hfinite := isFinite_asinh hfmt hx
  rw [asinh_eq_round hx hdecode] at hfinite ⊢
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ (Real.arsinh |(a : ℝ)|)
    (Real.arsinh_nonneg_iff.mpr (abs_nonneg _))
    (fun b => by rw [HyperbolicComparison.prepareArsinh_eq_real]; simp) hfinite]
  exact roundAt_signed_odd Real.arsinh Real.arsinh_neg hreal

/-- At a finite input that decodes to `a ≥ 1`, `acosh` rounds `arcosh a` through its comparator. -/
theorem acosh_eq_round {x : Model fmt} {a : ℚ} (hx : isFinite x = true)
    (hdecode : toRat? x = some a) (ha : 1 ≤ a) :
    acosh x = AlgebraicRounding.round fmt false
      (HyperbolicComparison.prepareArcosh a levels).compare := by
  simp [acosh, isNaN_eq_false_of_isFinite_eq_true x hx, hdecode, not_lt.mpr ha]

/-- `acosh` of a finite input at least one is finite, since `arcosh a ≤ a`. -/
theorem isFinite_acosh (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hdomain : 1 ≤ toReal x) : isFinite (acosh x) = true := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha : 1 ≤ a := by exact_mod_cast (hreal ▸ hdomain : (1 : ℝ) ≤ a)
  have ha' : (1 : ℝ) ≤ a := by exact_mod_cast ha
  rw [acosh_eq_round hx hdecode ha]
  have hcosh : (a : ℝ) ≤ Real.cosh a :=
    (Real.self_le_sinh_iff.mpr (by linarith)).trans (Real.sinh_lt_cosh _).le
  have hle : Real.arcosh (a : ℝ) ≤ a :=
    calc Real.arcosh (a : ℝ) ≤ Real.arcosh (Real.cosh a) :=
          (Real.arcosh_le_arcosh (by linarith) (Real.cosh_pos _)).mpr hcosh
      _ = a := Real.arcosh_cosh (by linarith)
  exact AlgebraicRounding.isFinite_round_of_lt_overflow fmt hfmt _ _ (Real.arcosh (a : ℝ))
    (fun b => HyperbolicComparison.prepareArcosh_eq_real a levels b ha)
    (lt_of_le_of_lt (hle.trans (le_abs_self _)) (hreal ▸ abs_toReal_lt_overflow hfmt hx))

/-- At a finite input at least one, `acosh` is the inverse hyperbolic cosine rounded once. -/
theorem toReal_acosh (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hdomain : 1 ≤ toReal x) :
    toReal (acosh x) = roundAt fmt (Real.arcosh (toReal x)) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha : 1 ≤ a := by exact_mod_cast (hreal ▸ hdomain : (1 : ℝ) ≤ a)
  have hfinite := isFinite_acosh hfmt hx hdomain
  rw [acosh_eq_round hx hdecode ha] at hfinite ⊢
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ (Real.arcosh (a : ℝ))
    (Real.arcosh_nonneg (by exact_mod_cast ha))
    (fun b => HyperbolicComparison.prepareArcosh_eq_real a levels b ha) hfinite, hreal]
  simp

/-- At a finite input in `(-1, 1)`, `atanh` is comparator rounding of `artanh |a|` with the
sign of `x`. -/
theorem atanh_eq_round {x : Model fmt} {a : ℚ} (hx : isFinite x = true)
    (hdecode : toRat? x = some a) (ha : |a| < 1) :
    atanh x = AlgebraicRounding.round fmt (signBit x)
      (HyperbolicComparison.prepareArtanh |a| levels (by linarith [abs_nonneg a]) ha).compare := by
  simp [atanh, isNaN_eq_false_of_isFinite_eq_true x hx, hdecode, ha]

/-- At a finite input in `(-1, 1)`, `atanh` is finite when the inverse hyperbolic tangent is
below the overflow threshold in magnitude. -/
theorem isFinite_atanh (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hdomain : |toReal x| < 1)
    (hrange : |Real.artanh (toReal x)| < Transcendentals.Certified.overflowThreshold fmt) :
    isFinite (atanh x) = true := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha : |a| < 1 := by exact_mod_cast (hreal ▸ hdomain : |(a : ℝ)| < 1)
  rw [atanh_eq_round hx hdecode ha]
  refine AlgebraicRounding.isFinite_round_of_lt_overflow fmt hfmt _ _ (Real.artanh |(a : ℝ)|)
    (fun b => by rw [HyperbolicComparison.prepareArtanh_eq_real]; simp) ?_
  rw [← Transcendentals.CorrectlyRounded.cast_overflowThreshold, artanh_abs, ← hreal]
  exact hrange

/-- At a finite input in `(-1, 1)`, a finite `atanh` is the inverse hyperbolic tangent rounded
once. -/
theorem toReal_atanh (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hdomain : |toReal x| < 1) (hfinite : isFinite (atanh x) = true) :
    toReal (atanh x) = roundAt fmt (Real.artanh (toReal x)) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha : |a| < 1 := by exact_mod_cast (hreal ▸ hdomain : |(a : ℝ)| < 1)
  rw [atanh_eq_round hx hdecode ha] at hfinite ⊢
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ (Real.artanh |(a : ℝ)|)
    (Real.artanh_nonneg (abs_nonneg _))
    (fun b => by rw [HyperbolicComparison.prepareArtanh_eq_real]; simp) hfinite]
  exact roundAt_signed_odd Real.artanh artanh_neg hreal

/-- A finite `logb base` of a finite input has a positive input: at a zero it is an infinity and
below zero the invalid result. -/
theorem toReal_pos_of_isFinite_logb (hfmt : fmt.isIEEE = true) {base : ℚ} {x : Model fmt}
    (hx : isFinite x = true) (hfinite : isFinite (logb base x) = true) : 0 < toReal x := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  rw [hreal]
  by_cases hzero : isZero x = true
  · simp [logb, isNaN_eq_false_of_isFinite_eq_true x hx, hzero,
      isFinite_nativeOverflow hfmt] at hfinite
  · by_contra ha
    have ha' : ¬ 0 < a := fun h => ha (by exact_mod_cast h)
    simp [logb, isNaN_eq_false_of_isFinite_eq_true x hx, hzero, hdecode, ha',
      isFinite_invalidResult hfmt] at hfinite

/-- A finite `acosh` of a finite input has an input at least one: below one it is the invalid
result. -/
theorem one_le_toReal_of_isFinite_acosh (hfmt : fmt.isIEEE = true) {x : Model fmt}
    (hx : isFinite x = true) (hfinite : isFinite (acosh x) = true) : 1 ≤ toReal x := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  rw [hreal]
  by_contra ha
  have ha' : a < 1 := by exact_mod_cast (lt_of_not_ge ha)
  simp [acosh, isNaN_eq_false_of_isFinite_eq_true x hx, hdecode, ha',
    isFinite_invalidResult hfmt] at hfinite

/-- A finite `atanh` of a finite input has an input in `(-1, 1)`: at `±1` it is an infinity and
outside `[-1, 1]` the invalid result. -/
theorem abs_toReal_lt_one_of_isFinite_atanh (hfmt : fmt.isIEEE = true) {x : Model fmt}
    (hx : isFinite x = true) (hfinite : isFinite (atanh x) = true) : |toReal x| < 1 := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  rw [hreal]
  by_contra ha
  have ha' : ¬ |a| < 1 := fun h => ha (by exact_mod_cast h)
  by_cases hone : |a| = 1
  · simp [atanh, isNaN_eq_false_of_isFinite_eq_true x hx, hdecode, hone,
      isFinite_nativeOverflow hfmt] at hfinite
  · simp [atanh, isNaN_eq_false_of_isFinite_eq_true x hx, hdecode, ha', hone,
      isFinite_invalidResult hfmt] at hfinite

/-! ## Special values -/

/-- `tan` quiets a NaN. -/
theorem tan_of_isNaN {x : Model fmt} (h : isNaN x = true) : tan x = quietNaN x := by
  simp [tan, h]

/-- `exp2` quiets a NaN. -/
theorem exp2_of_isNaN {x : Model fmt} (h : isNaN x = true) : exp2 x = quietNaN x := by
  simp [exp2, h]

/-- `logb` quiets a NaN. -/
theorem logb_of_isNaN (base : ℚ) {x : Model fmt} (h : isNaN x = true) :
    logb base x = quietNaN x := by
  simp [logb, h]

/-- `asinh` quiets a NaN. -/
theorem asinh_of_isNaN {x : Model fmt} (h : isNaN x = true) : asinh x = quietNaN x := by
  simp [asinh, h]

/-- `acosh` quiets a NaN. -/
theorem acosh_of_isNaN {x : Model fmt} (h : isNaN x = true) : acosh x = quietNaN x := by
  simp [acosh, h]

/-- `atanh` quiets a NaN. -/
theorem atanh_of_isNaN {x : Model fmt} (h : isNaN x = true) : atanh x = quietNaN x := by
  simp [atanh, h]

/-- `tan` returns a zero unchanged, with its sign. -/
theorem tan_of_isZero {x : Model fmt} (hx : isFinite x = true) (hzero : isZero x = true) :
    tan x = x := by
  simp [tan, isNaN_eq_false_of_isFinite_eq_true x hx, hzero]

/-- `asinh` keeps a zero, with its sign. -/
theorem asinh_of_toReal_eq_zero {x : Model fmt} (hx : isFinite x = true)
    (hzero : toReal x = 0) : asinh x = zero fmt (signBit x) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha : a = 0 := by exact_mod_cast hreal.symm.trans hzero
  subst ha
  have hcompare : (HyperbolicComparison.prepareArsinh |(0 : ℚ)| levels).compare =
      fun boundary : ℚ => cmp (0 : ℝ) (boundary : ℝ) := by
    funext boundary
    simp [HyperbolicComparison.prepareArsinh_eq_real]
  simp only [asinh, isNaN_eq_false_of_isFinite_eq_true x hx, hdecode, Bool.false_eq_true,
    ↓reduceIte, hcompare]
  exact round_cmp_zero (signBit x)

/-- `atanh` keeps a zero, with its sign. -/
theorem atanh_of_toReal_eq_zero {x : Model fmt} (hx : isFinite x = true)
    (hzero : toReal x = 0) : atanh x = zero fmt (signBit x) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha : a = 0 := by exact_mod_cast hreal.symm.trans hzero
  subst ha
  have hcompare : ∀ h₁ h₂, (HyperbolicComparison.prepareArtanh 0 levels h₁ h₂).compare =
      fun boundary : ℚ => cmp (0 : ℝ) (boundary : ℝ) := by
    intro h₁ h₂
    funext boundary
    simp [HyperbolicComparison.prepareArtanh_eq_real]
  simp only [atanh, isNaN_eq_false_of_isFinite_eq_true x hx, hdecode, Bool.false_eq_true,
    ↓reduceIte, abs_zero, zero_lt_one, ↓reduceDIte]
  rw [hcompare]
  exact round_cmp_zero (signBit x)

end Elementary

namespace ConfiguredBinary

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- At a finite input, a finite `tan` is the tangent rounded once. -/
theorem toReal_tan {x : Value} (hx : ExecFloat.Binary.isFinite x = true)
    (hfinite : ExecFloat.Binary.isFinite (tan x) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    ExecFloat.Binary.toReal (tan x) = roundAt format (Real.tan (ExecFloat.Binary.toReal x)) := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [tan, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact Elementary.toReal_tan hformat hx hfinite

/-- At a finite input, a finite `exp2` is `2 ^ x` rounded once. -/
theorem toReal_exp2 {x : Value} (hx : ExecFloat.Binary.isFinite x = true)
    (hfinite : ExecFloat.Binary.isFinite (exp2 x) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    ExecFloat.Binary.toReal (exp2 x) = roundAt format ((2 : ℝ) ^ ExecFloat.Binary.toReal x) := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [exp2, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact Elementary.toReal_exp2 hformat hx hfinite

/-- At a positive finite input, a finite `logb base` for a base above one is the logarithm to
that base rounded once. -/
theorem toReal_logb {base : ℚ} (hbase : 1 < base) {x : Value}
    (hx : ExecFloat.Binary.isFinite x = true) (hpositive : 0 < ExecFloat.Binary.toReal x)
    (hfinite : ExecFloat.Binary.isFinite (logb base x) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    ExecFloat.Binary.toReal (logb base x) =
      roundAt format (Real.logb base (ExecFloat.Binary.toReal x)) := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [logb, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact Elementary.toReal_logb hformat hbase hx hpositive hfinite

/-- At a finite input, `asinh` is the inverse hyperbolic sine rounded once. -/
theorem toReal_asinh {x : Value} (hx : ExecFloat.Binary.isFinite x = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    ExecFloat.Binary.toReal (asinh x) =
      roundAt format (Real.arsinh (ExecFloat.Binary.toReal x)) := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [asinh, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact Elementary.toReal_asinh hformat hx

/-- At a finite input at least one, `acosh` is the inverse hyperbolic cosine rounded once. -/
theorem toReal_acosh {x : Value} (hx : ExecFloat.Binary.isFinite x = true)
    (hdomain : 1 ≤ ExecFloat.Binary.toReal x)
    (hformat : format.isIEEE = true := by ieee_format) :
    ExecFloat.Binary.toReal (acosh x) =
      roundAt format (Real.arcosh (ExecFloat.Binary.toReal x)) := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [acosh, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact Elementary.toReal_acosh hformat hx hdomain

/-- At a finite input in `(-1, 1)`, a finite `atanh` is the inverse hyperbolic tangent rounded
once. -/
theorem toReal_atanh {x : Value} (hx : ExecFloat.Binary.isFinite x = true)
    (hdomain : |ExecFloat.Binary.toReal x| < 1)
    (hfinite : ExecFloat.Binary.isFinite (atanh x) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    ExecFloat.Binary.toReal (atanh x) =
      roundAt format (Real.artanh (ExecFloat.Binary.toReal x)) := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [atanh, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact Elementary.toReal_atanh hformat hx hdomain hfinite

/-- A finite `logb base` of a finite input has a positive input. -/
theorem toReal_pos_of_isFinite_logb {base : ℚ} {x : Value}
    (hx : ExecFloat.Binary.isFinite x = true)
    (hfinite : ExecFloat.Binary.isFinite (logb base x) = true)
    (hformat : format.isIEEE = true := by ieee_format) : 0 < ExecFloat.Binary.toReal x := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [logb, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact Elementary.toReal_pos_of_isFinite_logb hformat hx hfinite

/-- A finite `acosh` of a finite input has an input at least one. -/
theorem one_le_toReal_of_isFinite_acosh {x : Value} (hx : ExecFloat.Binary.isFinite x = true)
    (hfinite : ExecFloat.Binary.isFinite (acosh x) = true)
    (hformat : format.isIEEE = true := by ieee_format) : 1 ≤ ExecFloat.Binary.toReal x := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [acosh, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact Elementary.one_le_toReal_of_isFinite_acosh hformat hx hfinite

/-- A finite `atanh` of a finite input has an input in `(-1, 1)`. -/
theorem abs_toReal_lt_one_of_isFinite_atanh {x : Value} (hx : ExecFloat.Binary.isFinite x = true)
    (hfinite : ExecFloat.Binary.isFinite (atanh x) = true)
    (hformat : format.isIEEE = true := by ieee_format) : |ExecFloat.Binary.toReal x| < 1 := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [atanh, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact Elementary.abs_toReal_lt_one_of_isFinite_atanh hformat hx hfinite

end ConfiguredBinary

end TorchLean
