/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Numerics.MixedError.Basic
public import TorchLean.Verify.RuntimeApprox.RoundSigned
public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.StandardModel
public import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Exact
public import FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.SignedSemantics.Core

/-!
# Informative mixed bounds

A mixed bound `(r, a)` around a real `y` is informative when its radius `r * |y| + a` is below
`|y|` (`Informative`). For nonnegative `r` and `a` that holds iff no value within the radius is
zero (`informative_iff_excludes_zero`), and iff every such value has the sign of `y`
(`informative_iff_excludes_sign`). `Tight κ y r a` bounds the radius by `κ * |y|`. With `κ < 1`
and `y ≠ 0` a tight bound is informative (`Tight.informative`), and a value within it has relative
error at most `κ` (`Tight.abs_sub_le`) and logarithmic error at most `-log (1 - κ)` (`Tight.rp`).

In a binary format `fmt` with unit roundoff `u = unitRoundoffAt fmt`, underflow error
`η = underflowErrorAt fmt` (half the least subnormal), largest finite value `Ω` and overflow
threshold `Θ`:
* No finite value is closer to `y` than `|y|` when `|y| ≤ η` (`no_informative_below`).
* Every finite value is at least `|y| - Ω` away from `y` (`overflow_floor`).
* The nearest-even rounding `roundAt fmt x` of `x` to the format's grid, which has no largest
  value, has the mixed bound `(u, max 0 (η - u * |x|))` (`once_mixedError`), whose radius is
  `max (u * |x|) η`, and that bound is informative iff `η < |x|` (`once_informative_iff`). In an
  IEEE format the executable rounding `Elementary.roundSigned` agrees with `roundAt` when
  `|x| < Θ` (`Elementary.roundSigned_eq_roundAt`) and is infinite when `Θ ≤ |x|`
  (`Elementary.once_above`).
* In an IEEE format the largest finite value `Ω` is below `Θ`
  (`posMaxFinite_lt_overflowThreshold`), and some finite value is closer to `y` than `|y|` iff
  `η < |y|` (`exists_informative_iff`).
* In an IEEE format and for `|y| < Θ`, `roundSigned` of a comparator of `y` is closer to `y` than
  `|y|` whenever some finite value is (`optimal`).

The format statements hold for every real `y` and every descriptor, IEEE where stated.
-/

@[expose] public section

namespace TorchLean.Stable

open FloatLib.Numerics
open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Model

/-- The mixed bound `(r, a)` around `y` is informative: its radius `r * |y| + a` is below `|y|`. -/
def Informative (y r a : ℝ) : Prop := r * |y| + a < |y|

/-- The radius `r * |y| + a` of the mixed bound `(r, a)` around `y` is at most `κ * |y|`. -/
def Tight (κ y r a : ℝ) : Prop := r * |y| + a ≤ κ * |y|

variable {y r a κ v : ℝ}

/-- For `0 ≤ a`, the bound is informative iff `r < 1` and `a < (1 - r) * |y|`. -/
theorem informative_iff (ha : 0 ≤ a) : Informative y r a ↔ r < 1 ∧ a < (1 - r) * |y| := by
  unfold Informative
  constructor
  · intro h
    refine ⟨?_, by rw [sub_mul, one_mul]; linarith⟩
    by_contra hr
    have := mul_le_mul_of_nonneg_right (not_lt.mp hr) (abs_nonneg y)
    rw [one_mul] at this
    linarith
  · rintro ⟨-, h⟩
    rw [sub_mul, one_mul] at h
    linarith

/-- A bound with `0 ≤ a` around zero is not informative. -/
theorem zero_not_informative (ha : 0 ≤ a) : ¬Informative 0 r a := by
  unfold Informative
  simpa using ha

/-- For nonnegative `r` and `a`, the bound is informative iff no value within it is zero. -/
theorem informative_iff_excludes_zero (hr : 0 ≤ r) (ha : 0 ≤ a) :
    Informative y r a ↔ ∀ v, MixedError v y r a → v ≠ 0 := by
  constructor
  · exact fun h v hv => hv.ne_zero h
  · intro h
    by_contra hni
    exact h 0 ⟨hr, ha, by simpa [Informative] using hni⟩ rfl

/-- For nonnegative `r` and `a`, the bound is informative iff every value within it has the sign
of `y`. -/
theorem informative_iff_excludes_sign (hr : 0 ≤ r) (ha : 0 ≤ a) :
    Informative y r a ↔ ∀ v, MixedError v y r a → 0 < v * y := by
  constructor
  · intro h v hv
    have h' : r * |y| + a < |y| := h
    rcases lt_trichotomy y 0 with hy | hy | hy
    · exact mul_pos_of_neg_of_neg (hv.lt_zero (lt_of_lt_of_eq h' (abs_of_neg hy))) hy
    · exact absurd h (by rw [hy]; exact zero_not_informative ha)
    · exact mul_pos (hv.pos (lt_of_lt_of_eq h' (abs_of_pos hy))) hy
  · intro h
    by_contra hni
    simpa using h 0 ⟨hr, ha, by simpa [Informative] using hni⟩

/-- A bound tight at a grade below one around a nonzero `y` is informative. -/
theorem Tight.informative (h : Tight κ y r a) (hκ : κ < 1) (hy : y ≠ 0) : Informative y r a :=
  lt_of_le_of_lt h (lt_of_lt_of_eq (mul_lt_mul_of_pos_right hκ (abs_pos.mpr hy)) (one_mul _))

/-- A value within a tight bound has relative error at most the grade. -/
theorem Tight.abs_sub_le (h : Tight κ y r a) (hv : MixedError v y r a) : |v - y| ≤ κ * |y| :=
  hv.bound.trans h

/-- A value within a bound tight at a grade `κ < 1` around a nonzero `y` has logarithmic error at
most `-log (1 - κ)`. -/
theorem Tight.rp (h : Tight κ y r a) (hv : MixedError v y r a) (hκ : κ < 1) (hy : y ≠ 0) :
    |Real.log (v / y)| ≤ -Real.log (1 - κ) := by
  have hd := h.abs_sub_le hv
  have hya : 0 < |y| := abs_pos.mpr hy
  have ht : |v / y - 1| ≤ κ := by
    rw [show v / y - 1 = (v - y) / y by rw [sub_div, div_self hy], abs_div, div_le_iff₀ hya]
    exact hd
  obtain ⟨ht1, ht2⟩ := abs_le.mp ht
  have h1 : 0 < 1 - κ := by linarith
  have hpos : 0 < v / y := by linarith
  have hκ0 : 0 ≤ κ := by linarith
  have hinv : 1 + κ ≤ (1 - κ)⁻¹ := by
    rw [inv_eq_one_div, le_div_iff₀ h1]
    nlinarith
  rw [abs_le, _root_.neg_neg, ← Real.log_inv]
  exact ⟨Real.log_le_log h1 (by linarith), Real.log_le_log hpos (by linarith)⟩

variable {fmt : FloatFormat}

/-- When `|y| ≤ η`, every finite value of the format is at least `|y|` away from `y`. -/
theorem no_informative_below (hη : |y| ≤ underflowErrorAt fmt) (x : Model fmt)
    (hx : isFinite x = true) : |y| ≤ |toReal x - y| := by
  by_cases hzero : toReal x = 0
  · rw [hzero, zero_sub, abs_neg]
  · have hmin : FloatLib.Floats.Formats.Flocq.bpow binaryRadix fmt.minSubnormalExponent ≤
        |toReal x| := bpow_minSubnormalExponent_le_abs_toReal hx hzero
    rw [← two_mul_underflowErrorAt] at hmin
    have := abs_sub_abs_le_abs_sub (toReal x) y
    linarith

/-- Every finite value of the format is at least `|y| - Ω` away from `y`. -/
theorem overflow_floor (x : Model fmt) (hx : isFinite x = true) :
    |y| - toReal (posMaxFinite fmt) ≤ |toReal x - y| := by
  have h1 := abs_toReal_le_posMaxFinite_of_isFinite x hx
  have h2 := abs_sub_abs_le_abs_sub y (toReal x)
  rw [abs_sub_comm] at h2
  linarith

/-- The nearest-even rounding `roundAt fmt x` to the format's grid, which has no largest value,
has the mixed bound `(u, max 0 (η - u * |x|))`, whose radius is `max (u * |x|) η`. -/
theorem once_mixedError (fmt : FloatFormat) (x : ℝ) :
    MixedError (roundAt fmt x) x (unitRoundoffAt fmt)
      (max 0 (underflowErrorAt fmt - unitRoundoffAt fmt * |x|)) := by
  obtain ⟨δ, e, hround, hδ, he, hprod⟩ := roundAt_standardModel fmt x
  refine ⟨(unitRoundoffAt_pos fmt).le, le_max_left _ _, ?_⟩
  have hmax := le_max_right 0 (underflowErrorAt fmt - unitRoundoffAt fmt * |x|)
  have hmax0 := le_max_left 0 (underflowErrorAt fmt - unitRoundoffAt fmt * |x|)
  rw [hround]
  rcases mul_eq_zero.mp hprod with h | h
  · rw [h, show x * (1 + 0) + e - x = e by ring]
    linarith
  · rw [h, show x * (1 + δ) + 0 - x = δ * x by ring, abs_mul]
    have := mul_le_mul_of_nonneg_right hδ (abs_nonneg x)
    linarith

/-- The bound of `once_mixedError` is informative iff `|x|` is above `η`. -/
theorem once_informative_iff (fmt : FloatFormat) (x : ℝ) :
    Informative x (unitRoundoffAt fmt)
        (max 0 (underflowErrorAt fmt - unitRoundoffAt fmt * |x|)) ↔
      underflowErrorAt fmt < |x| := by
  unfold Informative
  have hu := unitRoundoffAt_lt_one fmt
  have hη := underflowErrorAt_pos fmt
  constructor
  · intro h
    have := le_max_right 0 (underflowErrorAt fmt - unitRoundoffAt fmt * |x|)
    linarith
  · intro h
    rcases le_total 0 (underflowErrorAt fmt - unitRoundoffAt fmt * |x|) with h0 | h0
    · rw [max_eq_right h0]
      linarith
    · rw [max_eq_left h0]
      have := mul_lt_mul_of_pos_right hu (hη.trans h)
      linarith

/-- In an IEEE format the largest finite value is below the overflow threshold. -/
theorem posMaxFinite_lt_overflowThreshold (hfmt : fmt.isIEEE = true) :
    toReal (posMaxFinite fmt) < Transcendentals.Certified.overflowThreshold fmt := by
  have h := Transcendentals.CorrectlyRounded.abs_lt_overflowThreshold_of_toReal_eq_roundAt
    hfmt (isFinite_posMaxFinite fmt) (roundAt_toReal_eq _ (isFinite_posMaxFinite fmt)).symm
  rwa [abs_of_pos (toReal_posMaxFinite_pos fmt)] at h

/-- In an IEEE format some finite value is closer to `y` than `|y|` iff `|y|` is above `η`. -/
theorem exists_informative_iff (hfmt : fmt.isIEEE = true) :
    (∃ x : Model fmt, isFinite x = true ∧ |toReal x - y| < |y|) ↔
      underflowErrorAt fmt < |y| := by
  constructor
  · rintro ⟨x, hx, hlt⟩
    by_contra hle
    exact absurd (no_informative_below (not_lt.mp hle) x hx) (not_le.mpr hlt)
  · intro hy
    by_cases hΘ : |y| < Transcendentals.Certified.overflowThreshold fmt
    · obtain ⟨hfin, hval⟩ := Elementary.roundSigned_eq_roundAt hfmt
        (c := fun q : ℚ => cmp y (q : ℝ)) (fun _ => rfl) hΘ
      refine ⟨_, hfin, ?_⟩
      rw [hval]
      exact lt_of_le_of_lt (once_mixedError fmt y).bound ((once_informative_iff fmt y).mpr hy)
    · have hΩ := toReal_posMaxFinite_pos fmt
      have hΩy : toReal (posMaxFinite fmt) < |y| :=
        lt_of_lt_of_le (posMaxFinite_lt_overflowThreshold hfmt) (not_lt.mp hΘ)
      rcases le_or_gt 0 y with h0 | h0
      · refine ⟨posMaxFinite fmt, isFinite_posMaxFinite fmt, ?_⟩
        rw [abs_of_nonneg h0] at hΩy ⊢
        rw [abs_of_neg (show toReal (posMaxFinite fmt) - y < 0 by linarith)]
        linarith
      · refine ⟨neg (posMaxFinite fmt), (isFinite_neg _).trans (isFinite_posMaxFinite fmt), ?_⟩
        rw [toReal_neg _ (isFinite_posMaxFinite fmt)]
        rw [abs_of_neg h0] at hΩy ⊢
        rw [abs_of_pos (show 0 < -toReal (posMaxFinite fmt) - y by linarith)]
        linarith

/-- In an IEEE format and for `|y|` below the overflow threshold, `roundSigned` of a comparator of
`y` is finite, and it is closer to `y` than `|y|` iff some finite value is. -/
theorem optimal (hfmt : fmt.isIEEE = true) {c : ℚ → Ordering} (hc : ∀ q : ℚ, c q = cmp y q)
    (hy : |y| < Transcendentals.Certified.overflowThreshold fmt) :
    isFinite (Elementary.roundSigned fmt c) = true ∧
      (|toReal (Elementary.roundSigned fmt c) - y| < |y| ↔
        ∃ x : Model fmt, isFinite x = true ∧ |toReal x - y| < |y|) := by
  obtain ⟨hfin, hval⟩ := Elementary.roundSigned_eq_roundAt hfmt hc hy
  refine ⟨hfin, fun h => ⟨_, hfin, h⟩, fun h => ?_⟩
  rw [hval]
  exact lt_of_le_of_lt (once_mixedError fmt y).bound
    ((once_informative_iff fmt y).mpr ((exists_informative_iff hfmt).mp h))

end TorchLean.Stable
