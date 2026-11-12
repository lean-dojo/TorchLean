/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.StandardModel
public import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Positive
public import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Proof
public import FloatLib.Floats.Formats.Flocq.Theory.Analysis.Neighbors
public import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Order
public import Mathlib.Analysis.Complex.Exponential
import FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.Constants

/-!
# Nearest rounding on the binary grid

`roundAt fmt` is nearest-even rounding to the grid of the binary format `fmt`, a grid with no
largest exponent, and `u` is its unit roundoff. The results hold in every format unless a premise
is stated.

* The successor of a real `x` on the grid lies at least `u * |x|` above it (`gap_ge`), and a
  representable `x` with `|x| < 2^e` has `|x| ≤ (1 - u) * 2^e` (`abs_le_of_lt`).
* A real strictly less than half of the adjacent gap on its side away from a representable `x`
  rounds to `x` (`roundAt_eq_of_near`).
* A representable `x` with `|x| < u` is the rounding of `exp x - 1` (`roundAt_expm1`).
* The grid steps by `2u` above `1` (`succ_one`), and the reals in `[1, 1 + u)` round to `1`
  (`roundAt_eq_one`), as does the exponential of every `d` in `[0, u - u²]`
  (`roundAt_exp_eq_one`).
* With exponent bias at least `2`, the grid steps by `u` below `1` (`pred_one`), `u` is
  representable (`format_unitRoundoff`) and the reals in `(1 - u, 1 - u/2)` round to `1 - u`
  (`roundAt_eq_one_sub`).
* `u ≤ 1/4` (`unitRoundoff_le`), and the underflow error is at most `u` (`underflowError_le`).
* On `|x| ≤ 1`, `|exp x - 1 - x - x^2/2| ≤ 2/9 * |x|^3` (`abs_exp_sub_quadratic_le`).
-/

@[expose] public section

open FloatLib FloatLib.Floats FloatLib.Floats.Formats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.Flocq FloatLib.Floats.Formats.BinaryInterchange.Model

namespace TorchLean.Grid

variable {fmt : FloatFormat}

/-- The unit roundoff of every binary format is at most `1/4`. -/
theorem unitRoundoff_le : unitRoundoffAt fmt ≤ 1 / 4 := by
  have := fmt.fracWidth_pos
  rw [unitRoundoffAt_eq]
  calc (2 : ℝ) ^ (-((fmt.fracWidth : ℤ) + 1)) ≤ 2 ^ (-2 : ℤ) :=
        zpow_le_zpow_right₀ (by norm_num) (by omega)
    _ = 1 / 4 := by norm_num

/-- At a nonzero `x` the spacing of the grid is at least `u` times the least power of two above
`|x|`. -/
theorem ulp_ge {x : ℝ} (hx : x ≠ 0) :
    unitRoundoffAt fmt * bpow Numerics.binaryRadix (magnitude Numerics.binaryRadix x) ≤
      ulp Numerics.binaryRadix (fexpOf fmt) x := by
  rw [ulp.of_ne_zero _ _ _ hx, cexp, unitRoundoffAt, ← bpow.add_exp, bpow_le_bpow_iff]
  simp only [fexpOf, fltExp, Int.ofNat_eq_natCast]
  omega

/-- The successor of every real `x` on the grid lies at least `u * |x|` above `x`. -/
theorem gap_ge (x : ℝ) :
    unitRoundoffAt fmt * |x| ≤ succ (β := Numerics.binaryRadix) (fexp := fexpOf fmt) x - x := by
  have hu := unitRoundoffAt_pos fmt
  rcases lt_trichotomy x 0 with hx | rfl | hx
  · have hy : 0 < -x := neg_pos.mpr hx
    have hm := abs_lt_bpow_magnitude Numerics.binaryRadix (-x)
    rw [abs_of_pos hy] at hm
    rw [abs_of_neg hx, succ, ite_eq_right (not_le.mpr hx), predPos]
    split_ifs with hb
    · have h := (bpow_le_bpow_iff Numerics.binaryRadix
        (-Int.ofNat (fmt.fracWidth + 1) + (magnitude Numerics.binaryRadix (-x) - 1))
        (fexpOf fmt (magnitude Numerics.binaryRadix (-x) - 1))).mpr (by
          simp only [fexpOf, fltExp, Int.ofNat_eq_natCast]
          omega)
      rw [bpow.add_exp, ← hb] at h
      change unitRoundoffAt fmt * -x ≤ _ at h
      linarith
    · have := ulp_ge (fmt := fmt) hy.ne'
      linarith [mul_lt_mul_of_pos_left hm hu]
  · rw [succ_zero, abs_zero, mul_zero, sub_zero]
    exact ulp.nonneg _ _ _
  · rw [abs_of_pos hx, succ_eq_of_nonneg hx.le]
    have := ulp_ge (fmt := fmt) hx.ne'
    have hm := abs_lt_bpow_magnitude Numerics.binaryRadix x
    rw [abs_of_pos hx] at hm
    linarith [mul_lt_mul_of_pos_left hm hu]

/-- A representable `x` with `|x| < 2^e` has `|x| ≤ (1 - u) * 2^e`. -/
theorem abs_le_of_lt {x : ℝ} {e : ℤ} (hx : genericFormat Numerics.binaryRadix (fexpOf fmt) x)
    (h : |x| < bpow Numerics.binaryRadix e) :
    |x| ≤ (1 - unitRoundoffAt fmt) * bpow Numerics.binaryRadix e := by
  have hu := unitRoundoffAt_lt_one fmt
  rcases eq_or_ne x 0 with rfl | h0
  · rw [abs_zero]
    exact mul_nonneg (by linarith) (bpow.nonneg _ _)
  have hpos : 0 < |x| := abs_pos.mpr h0
  have hs := succ_le_magnitude_bpow_of_pos hpos ((generic_format_abs_iff x).mpr hx)
  rw [succ_eq_of_nonneg hpos.le] at hs
  have hg := ulp_ge (fmt := fmt) hpos.ne'
  have hm := magnitude_le_of_abs_lt_bpow Numerics.binaryRadix |x| e hpos.ne' (by rwa [abs_abs])
  have hb := (bpow_le_bpow_iff Numerics.binaryRadix _ _).mpr hm
  linarith [mul_le_mul_of_nonneg_left hb (sub_nonneg.mpr hu.le)]

/-- On `|x| ≤ 1`, `|exp x - 1 - x - x^2/2| ≤ 2/9 * |x|^3`. -/
theorem abs_exp_sub_quadratic_le {x : ℝ} (hx : |x| ≤ 1) :
    |Real.exp x - 1 - x - x ^ 2 / 2| ≤ 2 / 9 * |x| ^ 3 := by
  have h := Real.exp_bound hx (n := 3) (by norm_num)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial] at h
  norm_num at h
  rw [abs_le] at h ⊢
  constructor <;> linarith [h.1, h.2]

/-- A real strictly less than half of the adjacent gap on its side away from a representable `x`
rounds to `x`. -/
theorem roundAt_eq_of_near {x y : ℝ}
    (hx : genericFormat Numerics.binaryRadix (fexpOf fmt) x)
    (hlo : x - y < (x - pred (β := Numerics.binaryRadix) (fexp := fexpOf fmt) x) / 2)
    (hhi : y - x < (succ (β := Numerics.binaryRadix) (fexp := fexpOf fmt) x - x) / 2) :
    roundAt fmt y = x := by
  obtain ⟨hr, hnear⟩ :=
    round_nearestEven_point (β := Numerics.binaryRadix) (fexp := fexpOf fmt) y
  change Flocq.round (β := Numerics.binaryRadix) (fexp := fexpOf fmt) nearestEven y = x
  set r := Flocq.round (β := Numerics.binaryRadix) (fexp := fexpOf fmt) nearestEven y
  have hle := hnear x hx
  have h1 := le_abs_self (r - y)
  have h2 := neg_le_abs (r - y)
  by_contra hne
  rcases lt_or_gt_of_ne hne with hrx | hxr
  · have hs := succ_le_of_lt (β := Numerics.binaryRadix) (fexp := fexpOf fmt)
      (generic_format_neg x hx) (generic_format_neg r hr) (neg_lt_neg hrx)
    have hp : r ≤ pred (β := Numerics.binaryRadix) (fexp := fexpOf fmt) x := by
      simp only [pred]
      linarith
    rcases le_or_gt y x with hyx | hyx
    · rw [abs_of_nonneg (sub_nonneg.mpr hyx)] at hle
      linarith
    · rw [abs_of_neg (sub_neg.mpr hyx)] at hle
      linarith
  · have hs := succ_le_of_lt (β := Numerics.binaryRadix) (fexp := fexpOf fmt) hx hr hxr
    rcases le_or_gt x y with hxy | hxy
    · rw [abs_of_nonpos (sub_nonpos.mpr hxy)] at hle
      linarith
    · rw [abs_of_pos (sub_pos.mpr hxy)] at hle
      linarith

/-- A representable `x` with `|x| < u` is the rounding of `exp x - 1`. -/
theorem roundAt_expm1 {x : ℝ} (hx : genericFormat Numerics.binaryRadix (fexpOf fmt) x)
    (h : |x| < unitRoundoffAt fmt) : roundAt fmt (Real.exp x - 1) = x := by
  rcases eq_or_ne x 0 with rfl | h0
  · simp
  have hu := unitRoundoff_le (fmt := fmt)
  have hu0 := unitRoundoffAt_pos fmt
  have ha : 0 < |x| := abs_pos.mpr h0
  have hb : |x| ≤ (1 - unitRoundoffAt fmt) * unitRoundoffAt fmt :=
    abs_le_of_lt (e := -Int.ofNat (fmt.fracWidth + 1)) hx h
  have hq := abs_exp_sub_quadratic_le (x := x) (by linarith)
  have hg := gap_ge (fmt := fmt) x
  have hl := Real.add_one_le_exp x
  have hp := pred_lt (β := Numerics.binaryRadix) (fexp := fexpOf fmt) h0
  refine roundAt_eq_of_near hx (by linarith) ?_
  have hsq : |x| ^ 2 < unitRoundoffAt fmt ^ 2 := pow_lt_pow_left₀ h (abs_nonneg x) two_ne_zero
  have key : |x| + 4 / 9 * |x| ^ 2 < unitRoundoffAt fmt := by nlinarith
  have h2 : x ^ 2 = |x| ^ 2 := (sq_abs x).symm
  have h3 : |x| * (|x| + 4 / 9 * |x| ^ 2) = |x| ^ 2 + 4 / 9 * |x| ^ 3 := by ring
  have h4 := mul_lt_mul_of_pos_left key ha
  have := le_abs_self (Real.exp x - 1 - x - x ^ 2 / 2)
  nlinarith

/-- The successor of `1` on the grid is `1 + 2u`. -/
theorem succ_one :
    succ (β := Numerics.binaryRadix) (fexp := fexpOf fmt) 1 = 1 + 2 * unitRoundoffAt fmt := by
  have := fmt.exponentBias_pos
  have h1 : (1 : ℝ) = bpow Numerics.binaryRadix 0 := by simp [bpow_binaryRadix]
  have he : fexpOf fmt (0 + 1) = -(fmt.fracWidth : ℤ) := by
    simp only [fexpOf, fltExp, FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
      Int.ofNat_eq_natCast]
    omega
  conv_lhs => rw [h1]
  rw [succ_eq_of_nonneg (by rw [bpow_binaryRadix]; norm_num), ulp_bpow, he, bpow_binaryRadix,
    bpow_binaryRadix, unitRoundoffAt_eq, zpow_zero]
  congr 1
  rw [show -(fmt.fracWidth : ℤ) = 1 + -((fmt.fracWidth : ℤ) + 1) by ring,
    zpow_add₀ two_ne_zero, zpow_one]

/-- With exponent bias at least `2`, the predecessor of `1` on the grid is `1 - u`. -/
theorem pred_one (h : 2 ≤ fmt.exponentBias) :
    pred (β := Numerics.binaryRadix) (fexp := fexpOf fmt) 1 = 1 - unitRoundoffAt fmt := by
  have h1 : (1 : ℝ) = bpow Numerics.binaryRadix 0 := by simp [bpow_binaryRadix]
  have he : fexpOf fmt 0 = -((fmt.fracWidth : ℤ) + 1) := by
    simp only [fexpOf, fltExp, FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
      Int.ofNat_eq_natCast]
    omega
  conv_lhs => rw [h1]
  rw [pred_bpow, he, bpow_binaryRadix, bpow_binaryRadix, unitRoundoffAt_eq, zpow_zero]

/-- `1` is representable. -/
theorem format_one : genericFormat Numerics.binaryRadix (fexpOf fmt) 1 := by
  simpa using toReal_genericFormat_of_isFinite (posOne fmt) (isFinite_posOne fmt)

/-- With exponent bias at least `2`, `u` is representable. -/
theorem format_unitRoundoff (h : 2 ≤ fmt.exponentBias) :
    genericFormat Numerics.binaryRadix (fexpOf fmt) (unitRoundoffAt fmt) :=
  generic_format_bpow (β := Numerics.binaryRadix) (fexp := fexpOf fmt)
    (-Int.ofNat (fmt.fracWidth + 1)) (by
      simp only [fexpOf, fltExp, FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
        Int.ofNat_eq_natCast]
      omega)

/-- With exponent bias at least `2`, the successor of `1 - u` on the grid is `1`. -/
theorem succ_one_sub (h : 2 ≤ fmt.exponentBias) :
    succ (β := Numerics.binaryRadix) (fexp := fexpOf fmt) (1 - unitRoundoffAt fmt) = 1 := by
  have hu := unitRoundoff_le (fmt := fmt)
  have hpos : 0 < 1 - unitRoundoffAt fmt := by linarith
  have hlo : bpow Numerics.binaryRadix (0 - 1) = 1 / 2 := by
    rw [bpow_binaryRadix]
    norm_num
  have hhi : bpow Numerics.binaryRadix 0 = 1 := by
    rw [bpow_binaryRadix, zpow_zero]
  have hm : magnitude Numerics.binaryRadix (1 - unitRoundoffAt fmt) = 0 :=
    magnitude_eq_of_bpow_bounds _ _ _ (by rw [hlo, abs_of_pos hpos]; linarith)
      (by rw [hhi, abs_of_pos hpos]; linarith [unitRoundoffAt_pos fmt])
  have he : fexpOf fmt 0 = -((fmt.fracWidth : ℤ) + 1) := by
    simp only [fexpOf, fltExp, FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
      Int.ofNat_eq_natCast]
    omega
  rw [succ_eq_of_nonneg hpos.le, ulp.of_ne_zero _ _ _ hpos.ne', cexp, hm, he, bpow_binaryRadix,
    ← unitRoundoffAt_eq]
  ring

/-- Every real in `[1, 1 + u)` rounds to `1`. -/
theorem roundAt_eq_one {y : ℝ} (h1 : 1 ≤ y) (h2 : y < 1 + unitRoundoffAt fmt) :
    roundAt fmt y = 1 := by
  refine roundAt_eq_of_near format_one ?_ ?_
  · have := pred_lt (β := Numerics.binaryRadix) (fexp := fexpOf fmt) (one_ne_zero (α := ℝ))
    linarith
  · rw [succ_one]
    linarith

/-- With exponent bias at least `2`, every real in `(1 - u, 1 - u/2)` rounds to `1 - u`. -/
theorem roundAt_eq_one_sub (h : 2 ≤ fmt.exponentBias) {y : ℝ} (h1 : 1 - unitRoundoffAt fmt < y)
    (h2 : y < 1 - unitRoundoffAt fmt / 2) : roundAt fmt y = 1 - unitRoundoffAt fmt := by
  have hf : genericFormat Numerics.binaryRadix (fexpOf fmt) (1 - unitRoundoffAt fmt) := by
    rw [← pred_one h]
    exact generic_format_pred format_one
  refine roundAt_eq_of_near hf ?_ ?_
  · have := pred_le (β := Numerics.binaryRadix) (fexp := fexpOf fmt) (1 - unitRoundoffAt fmt)
    linarith
  · rw [succ_one_sub h]
    linarith

/-- The exponential of every `d` in `[0, u - u²]` rounds to `1`. -/
theorem roundAt_exp_eq_one {d : ℝ} (hd0 : 0 ≤ d)
    (hd : d ≤ unitRoundoffAt fmt - unitRoundoffAt fmt ^ 2) : roundAt fmt (Real.exp d) = 1 := by
  have hu0 := unitRoundoffAt_pos fmt
  have hu := unitRoundoff_le (fmt := fmt)
  have hdu : d < unitRoundoffAt fmt := by nlinarith
  have h1 := Real.add_one_le_exp d
  have h2 := Real.abs_exp_sub_one_sub_id_le (x := d) (by rw [abs_of_nonneg hd0]; linarith)
  have h3 := le_abs_self (Real.exp d - 1 - d)
  have hdd : d ^ 2 < unitRoundoffAt fmt ^ 2 := pow_lt_pow_left₀ hdu hd0 two_ne_zero
  exact roundAt_eq_one (by linarith) (by linarith)

/-- The underflow error of every binary format is at most its unit roundoff. -/
theorem underflowError_le : underflowErrorAt fmt ≤ unitRoundoffAt fmt := by
  have := fmt.exponentBias_pos
  refine (underflowErrorAt_le_zpow _ (e := -((fmt.fracWidth : ℤ) + 1)) ?_).trans_eq
    (unitRoundoffAt_eq fmt).symm
  simp only [FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
    Int.ofNat_eq_natCast]
  omega

end TorchLean.Grid
