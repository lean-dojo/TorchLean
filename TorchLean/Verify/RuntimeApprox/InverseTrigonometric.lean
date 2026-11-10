/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ConfiguredBinary.InverseTrigonometric
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real
public import FloatLib.Floats.Formats.BinaryInterchange.Algebraic.Rounding.Proof
public import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Bounds
public import FloatLib.Floats.Formats.BinaryInterchange.Transcendentals.Pi.Proof
public import FloatLib.Numerics.Exact.Trigonometric.Proof
public import FloatLib.Numerics.Exact.Trigonometric.Atan2.Proof

/-!
# Rounding of the inverse trigonometric functions of binary values

For an IEEE descriptor, `InverseTrigonometric.atan`, `asin`, `acos` and `atan2` are finite at
finite inputs in their domains, and there each is the real function rounded once to nearest even
(`toReal_atan`, `toReal_asin`, `toReal_acos`, and `toReal_atan2` away from the origin). Each real
result is at most `π` in magnitude, which is below the overflow threshold of every IEEE
descriptor, so its rounding is finite.

The special values come in two forms. As equations of values, a NaN is quieted, `atan` and `asin`
keep a signed zero, and `asin` and `acos` give the invalid result outside `[-1, 1]` and at an
infinity. `atan2` of a zero `y` against `+0` or a positive `x`, and of a finite `y` against `+∞`,
is the zero of the sign of `y`. As equations of real values, `atan (±∞)` is the rounding of
`±π/2`. `atan2` takes the sign of `y` and is the rounding of `±π` at a zero `y` against `-0` or a
finite `y` against `-∞`, of `±π/2` at an infinite `y` against a finite `x`, and of `±π/4` or
`±3π/4` at two infinities. For configured binary values, the `ConfiguredBinary` theorems restate
the rounding theorems.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Model
open FloatLib.Numerics.TrigonometricComparison

namespace InverseTrigonometric

variable {fmt : FloatFormat}

/-- A value with its sign bit set is not positive. -/
theorem toReal_nonpos_of_signBit {x : Model fmt} (hsign : signBit x = true) : toReal x ≤ 0 := by
  cases hdecode : toDyadic? x with
  | none => simp [toReal_eq, hdecode]
  | some d =>
      have hnegative : d.negative = true :=
        (sign_eq_signBit_of_toDyadic?_some hdecode).trans hsign
      have hreal : toReal x = d.toReal := by simp [toReal_eq, hdecode]
      rw [hreal]
      exact Dyadic.toReal_nonpos_of_negative_eq_true hnegative

/-- For an odd `f` and `toReal x = a`, rounding `f |a|` with the sign of `x` attached is rounding
`f (toReal x)`. -/
theorem roundAt_signed_odd (f : ℝ → ℝ) (hodd : ∀ r, f (-r) = -f r) {x : Model fmt} {a : ℚ}
    (hreal : toReal x = (a : ℝ)) :
    roundAt fmt (if signBit x then -f |(a : ℝ)| else f |(a : ℝ)|) =
      roundAt fmt (f (toReal x)) := by
  rw [hreal]
  cases hsign : signBit x
  · have hnonneg : 0 ≤ (a : ℝ) := hreal ▸ toReal_nonneg_of_signBit_eq_false hsign
    simp [abs_of_nonneg hnonneg]
  · have hnonpos : (a : ℝ) ≤ 0 := hreal ▸ toReal_nonpos_of_signBit hsign
    simp [abs_of_nonpos hnonpos, hodd]

/-- Comparator rounding of a target at most `π` is finite in every IEEE descriptor. -/
theorem isFinite_round_of_le_pi (hfmt : fmt.isIEEE = true) (sign : Bool)
    (compare : Rat → Ordering) (target : ℝ)
    (hcompare : ∀ candidate : Rat, compare candidate = cmp target (candidate : ℝ))
    (hle : target ≤ Real.pi) : isFinite (AlgebraicRounding.round fmt sign compare) = true :=
  AlgebraicRounding.isFinite_round_of_lt_overflow fmt hfmt sign compare target hcompare
    (lt_of_le_of_lt hle (pi_lt_overflowThreshold fmt hfmt))

/-- Comparator rounding of zero is the zero of the requested sign. -/
theorem round_cmp_zero (sign : Bool) :
    AlgebraicRounding.round fmt sign (fun boundary : ℚ => cmp (0 : ℝ) (boundary : ℝ)) =
      zero fmt sign := by
  have hpos : (0 : ℝ) < FloatLib.Floats.Formats.Flocq.bpow FloatLib.Numerics.binaryRadix
      (AlgebraicRounding.lowerExponent fmt) := by
    rw [FloatLib.Floats.Formats.Flocq.bpow_binaryRadix]
    positivity
  unfold AlgebraicRounding.round
  simp [cmp, cmpUsing, hpos]

/-- At a finite input that decodes to `a`, `atan` rounds `arctan |a|` through its comparator, with
the sign of `x`. -/
theorem atan_eq_round {x : Model fmt} {a : ℚ} (hx : isFinite x = true)
    (hdecode : toRat? x = some a) :
    atan x = AlgebraicRounding.round fmt (signBit x) (prepareArctan |a| levels).compare := by
  simp [atan, isNaN_eq_false_of_isFinite_eq_true x hx, isInf_eq_false_of_isFinite_eq_true x hx,
    hdecode]

/-- At a finite input that decodes to `a` with `|a| ≤ 1`, `asin` rounds `arcsin |a|` through its
comparator, with the sign of `x`. -/
theorem asin_eq_round {x : Model fmt} {a : ℚ} (hx : isFinite x = true)
    (hdecode : toRat? x = some a) (ha : |a| ≤ 1) :
    asin x = AlgebraicRounding.round fmt (signBit x) (prepareArcsin |a| levels).compare := by
  simp [asin, isNaN_eq_false_of_isFinite_eq_true x hx, hdecode, not_lt.mpr ha]

/-- At a finite input that decodes to `a` with `|a| ≤ 1`, `acos` rounds `arccos a` through its
comparator. -/
theorem acos_eq_round {x : Model fmt} {a : ℚ} (hx : isFinite x = true)
    (hdecode : toRat? x = some a) (ha : |a| ≤ 1) :
    acos x = AlgebraicRounding.round fmt false (prepareArccos a levels).compare := by
  simp [acos, isNaN_eq_false_of_isFinite_eq_true x hx, hdecode, not_lt.mpr ha]

/-- The rational value of a finite input in `[-1, 1]` lies in `[-1, 1]`. -/
theorem abs_le_one_of_toReal {x : Model fmt} {a : ℚ} (hreal : toReal x = (a : ℝ))
    (hdomain : |toReal x| ≤ 1) : |a| ≤ 1 := by
  exact_mod_cast (hreal ▸ hdomain : |(a : ℝ)| ≤ 1)

/-- `atan` of a finite input is finite. -/
theorem isFinite_atan (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true) :
    isFinite (atan x) = true := by
  obtain ⟨a, hdecode, -⟩ := Algebraic.exists_toRat?_of_isFinite hx
  rw [atan_eq_round hx hdecode]
  exact isFinite_round_of_le_pi hfmt _ _ (Real.arctan |(a : ℝ)|)
    (fun b => by simp [prepareArctan_eq_real])
    ((Real.arctan_lt_pi_div_two _).le.trans (by linarith [Real.pi_pos]))

/-- `asin` of a finite input in `[-1, 1]` is finite. -/
theorem isFinite_asin (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hdomain : |toReal x| ≤ 1) : isFinite (asin x) = true := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha := abs_le_one_of_toReal hreal hdomain
  rw [asin_eq_round hx hdecode ha]
  exact isFinite_round_of_le_pi hfmt _ _ (Real.arcsin |(a : ℝ)|)
    (fun b => by rw [prepareArcsin_eq_real _ _ _ (by linarith [abs_nonneg a]) ha]; simp)
    ((Real.arcsin_le_pi_div_two _).trans (by linarith [Real.pi_pos]))

/-- `acos` of a finite input in `[-1, 1]` is finite. -/
theorem isFinite_acos (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hdomain : |toReal x| ≤ 1) : isFinite (acos x) = true := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha := abs_le_one_of_toReal hreal hdomain
  rw [acos_eq_round hx hdecode ha]
  exact isFinite_round_of_le_pi hfmt _ _ (Real.arccos (a : ℝ))
    (fun b => prepareArccos_eq_real _ _ _ (abs_le.mp ha).1 (abs_le.mp ha).2)
    (Real.arccos_le_pi _)

/-- At finite operands that decode to `x = a` and `y = b`, `atan2 y x` rounds the principal
argument of `(a, |b|)` through its comparator, with the sign of `y`. A zero `y` against `-0` uses
the point `(-1, 0)`, whose argument is `π`. -/
theorem atan2_eq_round {y x : Model fmt} {a b : ℚ} (hy : isFinite y = true)
    (hx : isFinite x = true) (hda : toRat? x = some a) (hdb : toRat? y = some b) :
    atan2 y x = AlgebraicRounding.round fmt (signBit y)
      (prepareAtan2 (if |b| = 0 ∧ a = 0 ∧ signBit x then -1 else a) |b| levels).compare := by
  have hpoint : atan2Point y x = some (a, |b|) := by
    simp [atan2Point, isInf_eq_false_of_isFinite_eq_true y hy,
      isInf_eq_false_of_isFinite_eq_true x hx, hda, hdb]
  simp only [atan2, isNaN_eq_false_of_isFinite_eq_true y hy,
    isNaN_eq_false_of_isFinite_eq_true x hx, hpoint]
  rfl

/-- `atan2` of finite operands is finite. -/
theorem isFinite_atan2 (hfmt : fmt.isIEEE = true) {y x : Model fmt} (hy : isFinite y = true)
    (hx : isFinite x = true) : isFinite (atan2 y x) = true := by
  obtain ⟨a, hda, -⟩ := Algebraic.exists_toRat?_of_isFinite hx
  obtain ⟨b, hdb, -⟩ := Algebraic.exists_toRat?_of_isFinite hy
  rw [atan2_eq_round hy hx hda hdb]
  exact isFinite_round_of_le_pi hfmt _ _ _ (fun t => prepareAtan2_eq_real _ _ _ t)
    (Complex.arg_le_pi _)

/-- At a finite input, `atan` is the arctangent rounded once. -/
theorem toReal_atan (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true) :
    toReal (atan x) = roundAt fmt (Real.arctan (toReal x)) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have hfinite := isFinite_atan hfmt hx
  rw [atan_eq_round hx hdecode] at hfinite ⊢
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ (Real.arctan |(a : ℝ)|)
    (Real.arctan_nonneg.mpr (abs_nonneg _)) (fun b => by simp [prepareArctan_eq_real]) hfinite]
  exact roundAt_signed_odd Real.arctan Real.arctan_neg hreal

/-- At a finite input in `[-1, 1]`, `asin` is the inverse sine rounded once. -/
theorem toReal_asin (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hdomain : |toReal x| ≤ 1) : toReal (asin x) = roundAt fmt (Real.arcsin (toReal x)) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha := abs_le_one_of_toReal hreal hdomain
  have hfinite := isFinite_asin hfmt hx hdomain
  rw [asin_eq_round hx hdecode ha] at hfinite ⊢
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ (Real.arcsin |(a : ℝ)|)
    (Real.arcsin_nonneg.mpr (abs_nonneg _)) (fun b => by
      rw [prepareArcsin_eq_real _ _ _ (by linarith [abs_nonneg a]) ha]; simp) hfinite]
  exact roundAt_signed_odd Real.arcsin Real.arcsin_neg hreal

/-- At a finite input in `[-1, 1]`, `acos` is the inverse cosine rounded once. -/
theorem toReal_acos (hfmt : fmt.isIEEE = true) {x : Model fmt} (hx : isFinite x = true)
    (hdomain : |toReal x| ≤ 1) : toReal (acos x) = roundAt fmt (Real.arccos (toReal x)) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha := abs_le_one_of_toReal hreal hdomain
  have hfinite := isFinite_acos hfmt hx hdomain
  rw [acos_eq_round hx hdecode ha] at hfinite ⊢
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ (Real.arccos (a : ℝ))
    (Real.arccos_nonneg _) (fun b => prepareArccos_eq_real _ _ _ (abs_le.mp ha).1 (abs_le.mp ha).2)
    hfinite, hreal]
  simp

/-- At finite operands other than the origin, `atan2 y x` is the principal argument `θ` of
`(x, |y|)` with the sign of `y`, rounded once: `Complex.arg ⟨x, y⟩` when `y` is nonzero, and `±π`
for a zero `y` against a negative `x`. The sign of a zero `y` is part of the statement, since
`atan2 (+0, -1) = π` and `atan2 (-0, -1) = -π` have operands of the same real values. -/
theorem toReal_atan2 (hfmt : fmt.isIEEE = true) {y x : Model fmt} (hy : isFinite y = true)
    (hx : isFinite x = true) (horigin : toReal x ≠ 0 ∨ toReal y ≠ 0) :
    toReal (atan2 y x) = roundAt fmt
      (if signBit y then -Complex.arg ⟨toReal x, |toReal y|⟩
        else Complex.arg ⟨toReal x, |toReal y|⟩) := by
  obtain ⟨a, hda, hra⟩ := Algebraic.exists_toRat?_of_isFinite hx
  obtain ⟨b, hdb, hrb⟩ := Algebraic.exists_toRat?_of_isFinite hy
  have hnotOrigin : ¬(|b| = 0 ∧ a = 0 ∧ signBit x = true) := by
    rintro ⟨hb, ha, -⟩
    rcases horigin with h | h
    · exact h (by rw [hra, ha]; simp)
    · exact h (by rw [hrb, abs_eq_zero.mp hb]; simp)
  have hfinite := isFinite_atan2 hfmt hy hx
  rw [atan2_eq_round hy hx hda hdb, ite_eq_right hnotOrigin] at hfinite ⊢
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _
    (Complex.arg ⟨(a : ℝ), |(b : ℝ)|⟩) (Complex.arg_nonneg_iff.mpr (abs_nonneg _))
    (fun t => by rw [prepareAtan2_eq_real]; simp) hfinite, hra, hrb]

/-! ## Special values -/

/-- `atan` quiets a NaN. -/
theorem atan_of_isNaN {x : Model fmt} (h : isNaN x = true) : atan x = quietNaN x := by
  simp [atan, h]

/-- `asin` quiets a NaN. -/
theorem asin_of_isNaN {x : Model fmt} (h : isNaN x = true) : asin x = quietNaN x := by
  simp [asin, h]

/-- `acos` quiets a NaN. -/
theorem acos_of_isNaN {x : Model fmt} (h : isNaN x = true) : acos x = quietNaN x := by
  simp [acos, h]

/-- `atan2` quiets a NaN `y`. -/
theorem atan2_of_isNaN_left {y x : Model fmt} (h : isNaN y = true) :
    atan2 y x = quietNaN y := by
  simp [atan2, h]

/-- `atan2` quiets a NaN `x` against a `y` that is not a NaN. -/
theorem atan2_of_isNaN_right {y x : Model fmt} (hy : isNaN y = false) (h : isNaN x = true) :
    atan2 y x = quietNaN x := by
  simp [atan2, hy, h]

/-- `atan` keeps a zero, with its sign. -/
theorem atan_of_toReal_eq_zero {x : Model fmt} (hx : isFinite x = true) (hzero : toReal x = 0) :
    atan x = zero fmt (signBit x) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha : a = 0 := by exact_mod_cast hreal.symm.trans hzero
  subst ha
  rw [atan_eq_round hx hdecode]
  have hcompare : (prepareArctan |(0 : ℚ)| levels).compare =
      fun boundary : ℚ => cmp (0 : ℝ) (boundary : ℝ) := by
    funext boundary
    simp [prepareArctan_eq_real]
  rw [hcompare]
  exact round_cmp_zero (signBit x)

/-- `asin` keeps a zero, with its sign. -/
theorem asin_of_toReal_eq_zero {x : Model fmt} (hx : isFinite x = true) (hzero : toReal x = 0) :
    asin x = zero fmt (signBit x) := by
  obtain ⟨a, hdecode, hreal⟩ := Algebraic.exists_toRat?_of_isFinite hx
  have ha : a = 0 := by exact_mod_cast hreal.symm.trans hzero
  subst ha
  rw [asin_eq_round hx hdecode (by simp)]
  have hcompare : (prepareArcsin |(0 : ℚ)| levels).compare =
      fun boundary : ℚ => cmp (0 : ℝ) (boundary : ℝ) := by
    funext boundary
    rw [prepareArcsin_eq_real _ _ _ (by norm_num) (by norm_num)]
    simp
  rw [hcompare]
  exact round_cmp_zero (signBit x)

/-- `asin` outside `[-1, 1]`, an infinity included, gives the invalid result. -/
theorem asin_of_not_mem {x : Model fmt} (hnan : isNaN x = false)
    (h : isFinite x = false ∨ 1 < |toReal x|) : asin x = invalidResult fmt := by
  cases hdecode : toRat? x with
  | none => simp [asin, hnan, hdecode]
  | some a =>
      have hfinite := Algebraic.isFinite_of_toRat?_eq_some hdecode
      have hreal := Algebraic.toReal_of_toRat?_eq_some hdecode
      rcases h with h | h
      · rw [hfinite] at h; cases h
      · have : 1 < |a| := by exact_mod_cast (hreal ▸ h : 1 < |(a : ℝ)|)
        simp [asin, hnan, hdecode, this]

/-- `acos` outside `[-1, 1]`, an infinity included, gives the invalid result. -/
theorem acos_of_not_mem {x : Model fmt} (hnan : isNaN x = false)
    (h : isFinite x = false ∨ 1 < |toReal x|) : acos x = invalidResult fmt := by
  cases hdecode : toRat? x with
  | none => simp [acos, hnan, hdecode]
  | some a =>
      have hfinite := Algebraic.isFinite_of_toRat?_eq_some hdecode
      have hreal := Algebraic.toReal_of_toRat?_eq_some hdecode
      rcases h with h | h
      · rw [hfinite] at h; cases h
      · have : 1 < |a| := by exact_mod_cast (hreal ▸ h : 1 < |(a : ℝ)|)
        simp [acos, hnan, hdecode, this]

/-- `atan (±∞)` is the rounding of `±π/2`. -/
theorem toReal_atan_of_isInf (hfmt : fmt.isIEEE = true) {x : Model fmt} (hnan : isNaN x = false)
    (hinf : isInf x = true) :
    toReal (atan x) = roundAt fmt (if signBit x then -(Real.pi / 2) else Real.pi / 2) := by
  have heq : atan x = AlgebraicRounding.round fmt (signBit x) (prepareAtan2 0 1 levels).compare :=
    by simp [atan, hnan, hinf]
  have hcompare : ∀ t : ℚ, (prepareAtan2 0 1 levels).compare t = cmp (Real.pi / 2) (t : ℝ) := by
    intro t
    rw [prepareAtan2_eq_real]
    congr 1
    simp only [Rat.cast_zero, Rat.cast_one]
    exact Complex.arg_I
  have hfinite := isFinite_round_of_le_pi hfmt (signBit x) _ _ hcompare
    (by linarith [Real.pi_pos])
  rw [heq]
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ (Real.pi / 2)
    (by linarith [Real.pi_pos]) hcompare hfinite]

/-- `atan2` of a zero `y` against `+0` or a positive `x` is the zero of the sign of `y`. -/
theorem atan2_of_toReal_eq_zero {y x : Model fmt} (hy : isFinite y = true)
    (hx : isFinite x = true) (hzero : toReal y = 0)
    (hpositive : 0 < toReal x ∨ (toReal x = 0 ∧ signBit x = false)) :
    atan2 y x = zero fmt (signBit y) := by
  obtain ⟨a, hda, hra⟩ := Algebraic.exists_toRat?_of_isFinite hx
  obtain ⟨b, hdb, hrb⟩ := Algebraic.exists_toRat?_of_isFinite hy
  have hb : b = 0 := by exact_mod_cast hrb.symm.trans hzero
  subst hb
  have hnot : ¬(|(0 : ℚ)| = 0 ∧ a = 0 ∧ signBit x = true) := by
    rintro ⟨-, ha, hs⟩
    rcases hpositive with h | ⟨-, h⟩
    · rw [hra, ha] at h; simp at h
    · rw [hs] at h; cases h
  rw [atan2_eq_round hy hx hda hdb, ite_eq_right hnot]
  have ha : 0 ≤ (a : ℝ) := by
    rw [← hra]
    rcases hpositive with h | ⟨h, -⟩
    · exact h.le
    · exact h.ge
  have hcompare : (prepareAtan2 a |(0 : ℚ)| levels).compare =
      fun boundary : ℚ => cmp (0 : ℝ) (boundary : ℝ) := by
    funext t
    rw [prepareAtan2_eq_real]
    congr 1
    exact Complex.arg_eq_zero_iff.mpr ⟨by simpa using ha, by simp⟩
  rw [hcompare]
  exact round_cmp_zero (signBit y)

/-- `atan2` of a zero `y` against `-0` is the rounding of `±π`, with the sign of `y`. -/
theorem toReal_atan2_of_negZero (hfmt : fmt.isIEEE = true) {y x : Model fmt}
    (hy : isFinite y = true) (hx : isFinite x = true) (hzero : toReal y = 0)
    (hxzero : toReal x = 0) (hsign : signBit x = true) :
    toReal (atan2 y x) = roundAt fmt (if signBit y then -Real.pi else Real.pi) := by
  obtain ⟨a, hda, hra⟩ := Algebraic.exists_toRat?_of_isFinite hx
  obtain ⟨b, hdb, hrb⟩ := Algebraic.exists_toRat?_of_isFinite hy
  have hb : b = 0 := by exact_mod_cast hrb.symm.trans hzero
  have ha : a = 0 := by exact_mod_cast hra.symm.trans hxzero
  subst hb ha
  have hfinite := isFinite_atan2 hfmt hy hx
  rw [atan2_eq_round hy hx hda hdb, ite_eq_left ⟨by simp, rfl, hsign⟩] at hfinite ⊢
  have hcompare : ∀ t : ℚ,
      (prepareAtan2 (-1) |(0 : ℚ)| levels).compare t = cmp Real.pi (t : ℝ) := by
    intro t
    rw [prepareAtan2_eq_real]
    congr 1
    have hpoint : (⟨((-1 : ℚ) : ℝ), ((|(0 : ℚ)| : ℚ) : ℝ)⟩ : ℂ) = -1 :=
      Complex.ext (by simp) (by simp)
    rw [hpoint]
    exact Complex.arg_neg_one
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ Real.pi Real.pi_pos.le hcompare
    hfinite]

/-- Comparator rounding of the principal argument of a point `(a, b)` of the closed upper
half-plane, with a sign, is the rounding of that signed argument. -/
theorem toReal_round_arg (hfmt : fmt.isIEEE = true) (sign : Bool) (a b : ℚ) (hb : 0 ≤ b) :
    toReal (AlgebraicRounding.round fmt sign (prepareAtan2 a b levels).compare) =
      roundAt fmt (if sign then -Complex.arg ⟨(a : ℝ), (b : ℝ)⟩
        else Complex.arg ⟨(a : ℝ), (b : ℝ)⟩) := by
  have hcompare : ∀ t : ℚ, (prepareAtan2 a b levels).compare t =
      cmp (Complex.arg ⟨(a : ℝ), (b : ℝ)⟩) (t : ℝ) := fun t => prepareAtan2_eq_real _ _ _ t
  exact AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ _
    (Complex.arg_nonneg_iff.mpr (show (0 : ℝ) ≤ (b : ℝ) by exact_mod_cast hb)) hcompare
    (isFinite_round_of_le_pi hfmt _ _ _ hcompare (Complex.arg_le_pi _))

/-- `atan2` of an infinite `y` against a finite `x` is the rounding of `±π/2`, with the sign of
`y`. -/
theorem toReal_atan2_of_isInf_left (hfmt : fmt.isIEEE = true) {y x : Model fmt}
    (hnan : isNaN y = false) (hinf : isInf y = true) (hx : isFinite x = true) :
    toReal (atan2 y x) = roundAt fmt (if signBit y then -(Real.pi / 2) else Real.pi / 2) := by
  have heq : atan2 y x =
      AlgebraicRounding.round fmt (signBit y) (prepareAtan2 0 1 levels).compare := by
    simp [atan2, atan2Point, hnan, hinf, isNaN_eq_false_of_isFinite_eq_true x hx,
      isInf_eq_false_of_isFinite_eq_true x hx]
  have harg : Complex.arg ⟨((0 : ℚ) : ℝ), ((1 : ℚ) : ℝ)⟩ = Real.pi / 2 := by
    simp only [Rat.cast_zero, Rat.cast_one]
    exact Complex.arg_I
  rw [heq, toReal_round_arg hfmt _ 0 1 (by norm_num), harg]

/-- `atan2` of a finite `y` against `+∞` is the zero of the sign of `y`. -/
theorem atan2_of_posInf_right {y x : Model fmt} (hy : isFinite y = true)
    (hnan : isNaN x = false) (hinf : isInf x = true) (hsign : signBit x = false) :
    atan2 y x = zero fmt (signBit y) := by
  have heq : atan2 y x =
      AlgebraicRounding.round fmt (signBit y) (prepareAtan2 1 0 levels).compare := by
    simp [atan2, atan2Point, hnan, hinf, hsign, isNaN_eq_false_of_isFinite_eq_true y hy,
      isInf_eq_false_of_isFinite_eq_true y hy]
  have hcompare : (prepareAtan2 1 0 levels).compare =
      fun boundary : ℚ => cmp (0 : ℝ) (boundary : ℝ) := by
    funext t
    rw [prepareAtan2_eq_real]
    congr 1
    exact Complex.arg_eq_zero_iff.mpr ⟨by simp, by simp⟩
  rw [heq, hcompare]
  exact round_cmp_zero (signBit y)

/-- `atan2` of a finite `y` against `-∞` is the rounding of `±π`, with the sign of `y`. -/
theorem toReal_atan2_of_negInf_right (hfmt : fmt.isIEEE = true) {y x : Model fmt}
    (hy : isFinite y = true) (hnan : isNaN x = false) (hinf : isInf x = true)
    (hsign : signBit x = true) :
    toReal (atan2 y x) = roundAt fmt (if signBit y then -Real.pi else Real.pi) := by
  have heq : atan2 y x =
      AlgebraicRounding.round fmt (signBit y) (prepareAtan2 (-1) 0 levels).compare := by
    simp [atan2, atan2Point, hnan, hinf, hsign, isNaN_eq_false_of_isFinite_eq_true y hy,
      isInf_eq_false_of_isFinite_eq_true y hy]
  have harg : Complex.arg ⟨((-1 : ℚ) : ℝ), ((0 : ℚ) : ℝ)⟩ = Real.pi := by
    have hpoint : (⟨((-1 : ℚ) : ℝ), ((0 : ℚ) : ℝ)⟩ : ℂ) = -1 := Complex.ext (by simp) (by simp)
    rw [hpoint]
    exact Complex.arg_neg_one
  rw [heq, toReal_round_arg hfmt _ (-1) 0 le_rfl, harg]

/-- `atan2` of two infinities is the rounding of the argument of `(±1, 1)`, the direction of
`x` with a positive `y` (`π/4` against `+∞`, `3π/4` against `-∞`), with the sign of `y`. -/
theorem toReal_atan2_of_isInf (hfmt : fmt.isIEEE = true) {y x : Model fmt}
    (hny : isNaN y = false) (hy : isInf y = true) (hnx : isNaN x = false) (hx : isInf x = true) :
    toReal (atan2 y x) = roundAt fmt
      (if signBit y then -Complex.arg ⟨if signBit x then -1 else 1, 1⟩
        else Complex.arg ⟨if signBit x then -1 else 1, 1⟩) := by
  have heq : atan2 y x = AlgebraicRounding.round fmt (signBit y)
      (prepareAtan2 (if signBit x then -1 else 1) 1 levels).compare := by
    simp [atan2, atan2Point, hny, hy, hnx, hx]
  rw [heq, toReal_round_arg hfmt _ _ 1 (by norm_num)]
  cases signBit x <;> simp

end InverseTrigonometric

namespace ConfiguredBinary

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- At a finite input, `atan` is the arctangent rounded once. -/
theorem toReal_atan {x : Value} (hx : ExecFloat.Binary.isFinite x = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    ExecFloat.Binary.toReal (atan x) =
      roundAt format (Real.arctan (ExecFloat.Binary.toReal x)) := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [atan, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact InverseTrigonometric.toReal_atan hformat hx

/-- At a finite input in `[-1, 1]`, `asin` is the inverse sine rounded once. -/
theorem toReal_asin {x : Value} (hx : ExecFloat.Binary.isFinite x = true)
    (hdomain : |ExecFloat.Binary.toReal x| ≤ 1)
    (hformat : format.isIEEE = true := by ieee_format) :
    ExecFloat.Binary.toReal (asin x) =
      roundAt format (Real.arcsin (ExecFloat.Binary.toReal x)) := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [asin, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact InverseTrigonometric.toReal_asin hformat hx hdomain

/-- At a finite input in `[-1, 1]`, `acos` is the inverse cosine rounded once. -/
theorem toReal_acos {x : Value} (hx : ExecFloat.Binary.isFinite x = true)
    (hdomain : |ExecFloat.Binary.toReal x| ≤ 1)
    (hformat : format.isIEEE = true := by ieee_format) :
    ExecFloat.Binary.toReal (acos x) =
      roundAt format (Real.arccos (ExecFloat.Binary.toReal x)) := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [acos, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact InverseTrigonometric.toReal_acos hformat hx hdomain

/-- At finite operands other than the origin, `atan2 y x` is the principal argument of `(x, |y|)`
with the sign of `y`, rounded once (`InverseTrigonometric.toReal_atan2`). -/
theorem toReal_atan2 {y x : Value} (hy : ExecFloat.Binary.isFinite y = true)
    (hx : ExecFloat.Binary.isFinite x = true)
    (horigin : ExecFloat.Binary.toReal x ≠ 0 ∨ ExecFloat.Binary.toReal y ≠ 0)
    (hformat : format.isIEEE = true := by ieee_format) :
    ExecFloat.Binary.toReal (atan2 y x) = roundAt format
      (if ExecFloat.Binary.signBit y then
        -Complex.arg ⟨ExecFloat.Binary.toReal x, |ExecFloat.Binary.toReal y|⟩
      else Complex.arg ⟨ExecFloat.Binary.toReal x, |ExecFloat.Binary.toReal y|⟩) := by
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal ExecFloat.Binary.signBit at *
  simp only [atan2, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftBinary] at *
  exact InverseTrigonometric.toReal_atan2 hformat hy hx horigin

end ConfiguredBinary

end TorchLean
