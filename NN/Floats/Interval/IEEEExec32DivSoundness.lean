/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Data.Bool.Basic
public import Mathlib.Data.EReal.Basic
public import Mathlib.Data.Real.Basic
public import NN.Floats.IEEEExec.DirectedRoundingSoundness
public import NN.Floats.IEEEExec.DivDirectedRoundingSoundness
public import NN.Floats.IEEEExec.ERealSemantics
public import NN.Floats.Interval.ERealCoercions
public import NN.Floats.Interval.IEEEExec32
public import NN.Floats.Interval.IEEEExec32NoNaN
public import NN.Floats.Interval.RealBounds

/-!
# Soundness of `IEEE32Exec.Interval32.div` / `inv` (4-corner rule + zero-straddle fallback)

`NN/Floats/Interval/IEEEExec32.lean` defines executable endpoint intervals with IEEE32Exec
endpoints and outward-rounded arithmetic. Interval division is implemented as:

* if the denominator interval contains `0`, return the conservative `whole = [-∞,+∞]`,
* otherwise use the classical “4-corner” rule:

```
[a,b] / [c,d] ⊆ [ min(a/c, a/d, b/c, b/d),  max(a/c, a/d, b/c, b/d) ]
```

where each corner is computed using directed rounding (`divDown` / `divUp`) and we take the IEEE
`minimum` / `maximum` of the 4 rounded corners.

This file proves a finite-input soundness theorem for that construction, stated in `EReal` to keep
overflow sound (corner computations may overflow to `±∞`).

References / background (informal pointers):
- Moore–Kearfott–Cloud, *Introduction to Interval Analysis* (2009), Ch. 2 (basic interval ops).
- IEEE 1788-2015 (interval arithmetic standard) for the “whole” fallback behavior when an interval
  straddles `0` in division (true quotient set is typically disconnected).
- IEEE 754-2019 for the executable binary32 kernel we model in `IEEE32Exec`.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

namespace Interval32

noncomputable section

open TorchLean.Floats.Interval

/-! ## `EReal` coercion helpers -/

/--
Reciprocal bounds for positive intervals.

If `0 < c` and `y ∈ [c,d]`, then `1/y ∈ [1/d, 1/c]`.
-/
private lemma inv_mem_Icc_of_mem_Icc_of_pos (c d y : ℝ)
    (hc : 0 < c) (hy : y ∈ Set.Icc c d) :
    (1 / y) ∈ Set.Icc (1 / d) (1 / c) := by
  rcases hy with ⟨hcy, hyd⟩
  have hypos : 0 < y := lt_of_lt_of_le hc hcy
  have hlo : 1 / d ≤ 1 / y := by
    -- `y ≤ d` and `0 < y` implies `1/d ≤ 1/y`
    exact one_div_le_one_div_of_le hypos hyd
  have hhi : 1 / y ≤ 1 / c := by
    exact one_div_le_one_div_of_le hc hcy
  exact ⟨hlo, hhi⟩

/--
Reciprocal bounds for negative intervals.

If `d < 0` and `y ∈ [c,d]`, then `1/y ∈ [1/d, 1/c]`.
-/
private lemma inv_mem_Icc_of_mem_Icc_of_neg (c d y : ℝ)
    (hd : d < 0) (hy : y ∈ Set.Icc c d) :
    (1 / y) ∈ Set.Icc (1 / d) (1 / c) := by
  rcases hy with ⟨hcy, hyd⟩
  have hyneg : y < 0 := lt_of_le_of_lt hyd hd
  have hlo : 1 / d ≤ 1 / y := by
    -- `y ≤ d < 0` implies `1/d ≤ 1/y`.
    exact one_div_le_one_div_of_neg_of_le hd hyd
  have hhi : 1 / y ≤ 1 / c := by
    exact one_div_le_one_div_of_neg_of_le hyneg hcy
  exact ⟨hlo, hhi⟩

/--
4-corner enclosure for division on a sign-stable denominator interval.

Assuming the denominator interval is either entirely negative (`d < 0`) or entirely positive
(`0 < c`), we can bound `x/y` by applying the multiplication 4-corner enclosure to `x * (1/y)`.
-/
private theorem div_bounds_Icc (a b c d x y : ℝ)
    (hx : x ∈ Set.Icc a b) (hy : y ∈ Set.Icc c d) (h0 : d < 0 ∨ 0 < c) :
    min4R (a / c) (a / d) (b / c) (b / d) ≤ x / y ∧
      x / y ≤ max4R (a / c) (a / d) (b / c) (b / d) := by
  have hinv1 : (1 / y) ∈ Set.Icc (1 / d) (1 / c) := by
    cases h0 with
    | inl hd => exact inv_mem_Icc_of_mem_Icc_of_neg (c := c) (d := d) (y := y) hd hy
    | inr hc => exact inv_mem_Icc_of_mem_Icc_of_pos (c := c) (d := d) (y := y) hc hy
  have hinv : y⁻¹ ∈ Set.Icc d⁻¹ c⁻¹ := by
    simpa [one_div] using hinv1
  have hmul :=
    mul_bounds_Icc (a := a) (b := b) (c := d⁻¹) (d := c⁻¹) (x := x) (y := y⁻¹) hx hinv
  -- Rewrite corner products into division corners.
  have hmul_lo :
      min4R (a * d⁻¹) (a * c⁻¹) (b * d⁻¹) (b * c⁻¹) ≤ x * y⁻¹ := hmul.1
  have hmul_hi :
      x * y⁻¹ ≤ max4R (a * d⁻¹) (a * c⁻¹) (b * d⁻¹) (b * c⁻¹) := hmul.2
  -- Reorder corners: our implementation uses `(a/c,a/d,b/c,b/d)`.
  have hmin_reorder :
      min4R (a * d⁻¹) (a * c⁻¹) (b * d⁻¹) (b * c⁻¹) =
        min4R (a / c) (a / d) (b / c) (b / d) := by
    simp [min4R, div_eq_mul_inv, min_comm]
  have hmax_reorder :
      max4R (a * d⁻¹) (a * c⁻¹) (b * d⁻¹) (b * c⁻¹) =
        max4R (a / c) (a / d) (b / c) (b / d) := by
    simp [max4R, div_eq_mul_inv, max_comm]
  constructor
  · -- Lower
    have : min4R (a / c) (a / d) (b / c) (b / d) ≤ x / y := by
      -- `hmul_lo` bounds `x*(1/y)`, and we rewrite to `x/y`.
      simpa [div_eq_mul_inv, hmin_reorder] using hmul_lo
    exact this
  · -- Upper
    have : x / y ≤ max4R (a / c) (a / d) (b / c) (b / d) := by
      simpa [div_eq_mul_inv, hmax_reorder] using hmul_hi
    exact this

/-! ## Non-NaN facts for `divDown` / `divUp` (finite, nonzero denom) -/

/--
`roundRatDown` never produces NaN.

This is used in the dyadic/rational implementation of `divDown` to show NaNs do not appear on the
finite, nonzero-denominator path.
-/
private lemma isNaN_roundRatDown_eq_false (sign : Bool) (num den : Nat) :
    isNaN (roundRatDown sign num den) = false := by
  by_cases hnum0 : num = 0
  · have hbeq : (num == 0) = true := (beq_iff_eq).2 hnum0
    cases hs : sign <;> simp [roundRatDown, hbeq] <;> decide
  · have hbeq : (num == 0) = false := (beq_eq_false_iff_ne).2 hnum0
    cases hs : sign <;> simp (config := { zeta := true }) [roundRatDown, hbeq]
    · exact isNaN_roundDyadicDown_eq_false _
    · exact isNaN_roundDyadicDown_eq_false _

/--
`roundRatUp` never produces NaN.

This is the “upper” analogue of `isNaN_roundRatDown_eq_false`.
-/
private lemma isNaN_roundRatUp_eq_false (sign : Bool) (num den : Nat) :
    isNaN (roundRatUp sign num den) = false := by
  by_cases hnum0 : num = 0
  · have hbeq : (num == 0) = true := (beq_iff_eq).2 hnum0
    cases hs : sign <;> simp [roundRatUp, hbeq] <;> decide
  · have hbeq : (num == 0) = false := (beq_eq_false_iff_ne).2 hnum0
    cases hs : sign <;> simp (config := { zeta := true }) [roundRatUp, hbeq]
    · exact isNaN_roundDyadicUp_eq_false _
    · exact isNaN_roundDyadicUp_eq_false _

/--
`divDown x y` is non-NaN on the finite, nonzero-denominator path.

We need this to justify rewriting `toEReal (min4 ...)` into nested `min` of `toEReal` values.
-/
private lemma isNaN_divDown_eq_false_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) (hy0 : isZero y = false) :
    isNaN (divDown x y) = false := by
  have hxNaN : isNaN x = false := isNaN_eq_false_of_isFinite_eq_true (x := x) hx
  have hyNaN : isNaN y = false := isNaN_eq_false_of_isFinite_eq_true (x := y) hy
  have hxInf : isInf x = false := isInf_eq_false_of_isFinite_eq_true (x := x) hx
  have hyInf : isInf y = false := isInf_eq_false_of_isFinite_eq_true (x := y) hy
  have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
  -- Reduce `divDown` to the dyadic branch; finiteness and `hy0` rule out the special cases.
  cases hdx : toDyadic? x with
  | none =>
      have hfin : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      cases (hx.symm.trans hfin)
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfin : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          cases (hy.symm.trans hfin)
      | some dy =>
          -- Now `divDown` returns either a signed zero or `roundRatDown`.
          simp (config := { zeta := true }) [divDown, hchoose, hxInf, hyInf, hy0, hdx, hdy]
          by_cases h0 : dx.mant = 0
          · -- `0 / y` is a signed zero; signed zeros are never NaN.
            have hz : isNaN (if dx.sign = dy.sign then posZero else negZero) = false := by
              by_cases hs : dx.sign = dy.sign <;> simp [hs] <;> decide
            simp [h0, hz]
          · -- Nonzero numerator: `roundRatDown` never returns NaN.
            simp [h0, isNaN_roundRatDown_eq_false]

/--
`divUp x y` is non-NaN on the finite, nonzero-denominator path.

This is the “upper” analogue of `isNaN_divDown_eq_false_of_isFinite`.
-/
private lemma isNaN_divUp_eq_false_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) (hy0 : isZero y = false) :
    isNaN (divUp x y) = false := by
  have hxNaN : isNaN x = false := isNaN_eq_false_of_isFinite_eq_true (x := x) hx
  have hyNaN : isNaN y = false := isNaN_eq_false_of_isFinite_eq_true (x := y) hy
  have hxInf : isInf x = false := isInf_eq_false_of_isFinite_eq_true (x := x) hx
  have hyInf : isInf y = false := isInf_eq_false_of_isFinite_eq_true (x := y) hy
  have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
  cases hdx : toDyadic? x with
  | none =>
      have hfin : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      cases (hx.symm.trans hfin)
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfin : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          cases (hy.symm.trans hfin)
      | some dy =>
          simp (config := { zeta := true }) [divUp, hchoose, hxInf, hyInf, hy0, hdx, hdy]
          by_cases h0 : dx.mant = 0
          · -- `0 / y` is a signed zero; signed zeros are never NaN.
            have hz : isNaN (if dx.sign = dy.sign then posZero else negZero) = false := by
              by_cases hs : dx.sign = dy.sign <;> simp [hs] <;> decide
            simp [h0, hz]
          · -- Nonzero numerator: `roundRatUp` never returns NaN.
            simp [h0, isNaN_roundRatUp_eq_false]

/-! ## Interval division soundness -/

/--
Decoding of IEEE signed zeros: if `x` is one of `±0`, then `toReal x = 0`.

This is used to rule out the case where an endpoint is IEEE-zero when we have already proved the
real endpoint is strictly positive/negative.
-/
private lemma toReal_eq_zero_of_isZero (x : IEEE32Exec) (hz : isZero x = true) : toReal x = 0 := by
  -- `toDyadic?` explicitly returns the dyadic `0` in the `isZero` case.
  have hxNaN : isNaN x = false := by
    have : (expField x == expAllOnes) = false := by
      -- `isZero` implies `expField x == 0`, hence not all-ones.
      have hz' : (expField x == 0) = true := (by
        have : (expField x == 0) = true ∧ (fracField x == 0) = true := by
          simpa [isZero, Bool.and_eq_true] using hz
        exact this.1)
      have : expField x = 0 := (beq_iff_eq).1 hz'
      exact (beq_eq_false_iff_ne).2 (by simpa [this] using (show (0 : UInt32) ≠ expAllOnes by
        decide))
    simp [isNaN, this]
  have hxInf : isInf x = false := by
    have : (expField x == expAllOnes) = false := by
      have hz' : (expField x == 0) = true := (by
        have : (expField x == 0) = true ∧ (fracField x == 0) = true := by
          simpa [isZero, Bool.and_eq_true] using hz
        exact this.1)
      have : expField x = 0 := (beq_iff_eq).1 hz'
      exact (beq_eq_false_iff_ne).2 (by simpa [this] using (show (0 : UInt32) ≠ expAllOnes by
        decide))
    simp [isInf, this]
  have hdy : toDyadic? x = some { sign := signBit x, mant := 0, exp := (0 : Int) } := by
    unfold toDyadic?
    -- Use `hz` to select the zero branch.
    have hz' : (expField x == 0) = true ∧ (fracField x == 0) = true := by
      simpa [isZero, Bool.and_eq_true] using hz
    simp (config := { zeta := true }) [hxNaN, hxInf, hz'.1, hz'.2]
  -- Now `toReal` is `dyadicToReal 0 = 0`.
  simp [toReal_eq, hdy, dyadicToReal]

/--
If the executable check says `0 ∉ B` (`containsZero B = false`), then the real interval
`[toReal B.lo, toReal B.hi]` is sign-stable: either entirely negative or entirely positive.
-/
private lemma denom_sign_case (B : Interval32)
    (hBlo : isFinite B.lo = true) (hBhi : isFinite B.hi = true) (hcz : containsZero B = false) :
    (toReal B.hi < 0) ∨ (0 < toReal B.lo) := by
  unfold containsZero at hcz
  -- Either the left or the right conjunct in `containsZero` is false.
  have hdisj : leB B.lo posZero = false ∨ leB negZero B.hi = false := by
    -- `containsZero B` is a boolean conjunction.
    exact
      Eq.mp
        (Bool.and_eq_false_eq_eq_false_or_eq_false (leB B.lo posZero) (leB negZero B.hi))
        hcz
  have hp0 : toReal (posZero : IEEE32Exec) = 0 := by
    simp
  have hz0 : toReal (negZero : IEEE32Exec) = 0 := by
    simp
  cases hdisj with
  | inl hloFalse =>
      -- `B.lo` is strictly positive (as a real): `posZero < B.lo`.
      have hxNaN : isNaN B.lo = false := isNaN_eq_false_of_isFinite_eq_true (x := B.lo) hBlo
      have hyFin : isFinite (posZero : IEEE32Exec) = true := by decide
      have hyNaN : isNaN (posZero : IEEE32Exec) = false :=
        isNaN_eq_false_of_isFinite_eq_true (x := (posZero : IEEE32Exec)) hyFin
      have hcmp_ne : compare B.lo posZero ≠ none :=
        compare_ne_none_of_isNaN_eq_false (x := B.lo) (y := posZero) hxNaN hyNaN
      have hcmp : compare B.lo posZero = some .gt := by
        unfold leB at hloFalse
        cases hco : compare B.lo posZero with
        | none => exact (False.elim (hcmp_ne hco))
        | some o =>
            cases o with
            | lt => simp [hco] at hloFalse
            | eq => simp [hco] at hloFalse
            | gt => rfl
      have hgt : toReal (posZero : IEEE32Exec) < toReal B.lo :=
        (compare_eq_some_gt_iff_toReal_gt_of_isFinite (x := B.lo) (y := posZero) hBlo hyFin).1 hcmp
      -- Rewrite `toReal posZero = 0`.
      have hgt0 : 0 < toReal B.lo := by
        have hgt' : toReal (posZero : IEEE32Exec) < toReal B.lo := hgt
        rw [hp0] at hgt'
        exact hgt'
      exact Or.inr hgt0
  | inr hhiFalse =>
      -- `B.hi` is strictly negative (as a real): `B.hi < negZero`.
      have hxFin : isFinite (negZero : IEEE32Exec) = true := by decide
      have hxNaN : isNaN (negZero : IEEE32Exec) = false :=
        isNaN_eq_false_of_isFinite_eq_true (x := (negZero : IEEE32Exec)) hxFin
      have hyNaN : isNaN B.hi = false := isNaN_eq_false_of_isFinite_eq_true (x := B.hi) hBhi
      have hcmp_ne : compare negZero B.hi ≠ none :=
        compare_ne_none_of_isNaN_eq_false (x := negZero) (y := B.hi) hxNaN hyNaN
      have hcmp : compare negZero B.hi = some .gt := by
        unfold leB at hhiFalse
        cases hco : compare negZero B.hi with
        | none => exact (False.elim (hcmp_ne hco))
        | some o =>
            cases o with
            | lt => simp [hco] at hhiFalse
            | eq => simp [hco] at hhiFalse
            | gt => rfl
      have hgt : toReal B.hi < toReal (negZero : IEEE32Exec) :=
        (compare_eq_some_gt_iff_toReal_gt_of_isFinite (x := negZero) (y := B.hi) hxFin hBhi).1 hcmp
      have hgt0 : toReal B.hi < 0 := by
        have hgt' : toReal B.hi < toReal (negZero : IEEE32Exec) := hgt
        rw [hz0] at hgt'
        exact hgt'
      exact Or.inl hgt0

/--
Soundness of `Interval32.div` w.r.t. real division.

This theorem is phrased over real concretizations `Set.Icc (toReal lo) (toReal hi)`.

If the executable implementation detects `0 ∈ B` (via `containsZero`), it returns `whole`, and
the enclosure is trivial. Otherwise we use the 4-corner rule and the directed-rounding soundness
lemmas for `divDown`/`divUp`.
-/
theorem div_sound (A B : Interval32) (hA : Valid A) (hB : Valid B) :
    ∀ {x y : ℝ},
      x ∈ Set.Icc (toReal A.lo) (toReal A.hi) →
      y ∈ Set.Icc (toReal B.lo) (toReal B.hi) →
        toEReal (Interval32.div A B).lo ≤ (x / y : EReal) ∧
          (x / y : EReal) ≤ toEReal (Interval32.div A B).hi := by
  intro x y hx hy
  have hAlo : isFinite A.lo = true := hA.1
  have hAhi : isFinite A.hi = true := hA.2.1
  have hBlo : isFinite B.lo = true := hB.1
  have hBhi : isFinite B.hi = true := hB.2.1
  by_cases hcz : containsZero B = true
  · -- Zero-straddle: `div` returns `whole = [-∞,+∞]`.
    have hlo : toEReal (Interval32.div A B).lo = (⊥ : EReal) := by
      simp [Interval32.div, hcz, Interval32.whole, toEReal_negInf]
    have hhi : toEReal (Interval32.div A B).hi = (⊤ : EReal) := by
      simp [Interval32.div, hcz, Interval32.whole, toEReal_posInf]
    constructor
    · simp [hlo]
    · simp [hhi]
  · -- Safe division: `0 ∉ B` in real semantics.
    have hsign : (toReal B.hi < 0) ∨ (0 < toReal B.lo) :=
      denom_sign_case (B := B) hBlo hBhi (by simpa using hcz)
    have hloLeHi : toReal B.lo ≤ toReal B.hi := le_trans hy.1 hy.2
    -- Real corner bounds.
    have hxy := div_bounds_Icc (a := toReal A.lo) (b := toReal A.hi)
      (c := toReal B.lo) (d := toReal B.hi) (x := x) (y := y) hx hy hsign
    have hxy_lo :
        (min4R (toReal A.lo / toReal B.lo) (toReal A.lo / toReal B.hi)
          (toReal A.hi / toReal B.lo) (toReal A.hi / toReal B.hi) : ℝ) ≤ x / y := hxy.1
    have hxy_hi :
        x / y ≤ (max4R (toReal A.lo / toReal B.lo) (toReal A.lo / toReal B.hi)
          (toReal A.hi / toReal B.lo) (toReal A.hi / toReal B.hi) : ℝ) := hxy.2

    -- Denominator endpoints are nonzero in IEEE sense (otherwise `toReal = 0`, contradicting the
    -- sign case).
    have hyLo0 : isZero B.lo = false := by
      by_cases hz : isZero B.lo = true
      · have h0 : toReal B.lo = 0 := toReal_eq_zero_of_isZero (x := B.lo) hz
        have : False := by
          cases hsign with
          | inr hloPos =>
              have hloPos0 : (0 : ℝ) < 0 := by
                have hloPos' : (0 : ℝ) < toReal B.lo := hloPos
                rw [h0] at hloPos'
                exact hloPos'
              exact (lt_irrefl (0 : ℝ)) hloPos0
          | inl hhiNeg =>
              have hloNeg : toReal B.lo < 0 := lt_of_le_of_lt hloLeHi hhiNeg
              have hloNeg0 : (0 : ℝ) < 0 := by
                have hloNeg' : toReal B.lo < (0 : ℝ) := hloNeg
                rw [h0] at hloNeg'
                exact hloNeg'
              exact (lt_irrefl (0 : ℝ)) hloNeg0
        exact False.elim this
      · -- `Bool`-valued predicates are either `true` or `false`.
        cases hzb : isZero B.lo with
        | true => cases (hz hzb)
        | false => rfl
    have hyHi0 : isZero B.hi = false := by
      by_cases hz : isZero B.hi = true
      · have h0 : toReal B.hi = 0 := toReal_eq_zero_of_isZero (x := B.hi) hz
        have : False := by
          cases hsign with
          | inl hhiNeg =>
              have hhiNeg0 : (0 : ℝ) < 0 := by
                have hhiNeg' : toReal B.hi < (0 : ℝ) := hhiNeg
                rw [h0] at hhiNeg'
                exact hhiNeg'
              exact (lt_irrefl (0 : ℝ)) hhiNeg0
          | inr hloPos =>
              have hhiPos : 0 < toReal B.hi := lt_of_lt_of_le hloPos hloLeHi
              have hhiPos0 : (0 : ℝ) < 0 := by
                have hhiPos' : (0 : ℝ) < toReal B.hi := hhiPos
                rw [h0] at hhiPos'
                exact hhiPos'
              exact (lt_irrefl (0 : ℝ)) hhiPos0
        exact False.elim this
      · cases hzb : isZero B.hi with
        | true => cases (hz hzb)
        | false => rfl

    -- Directed rounding corner quotients.
    let p00 := divDown A.lo B.lo
    let p01 := divDown A.lo B.hi
    let p10 := divDown A.hi B.lo
    let p11 := divDown A.hi B.hi
    let q00 := divUp A.lo B.lo
    let q01 := divUp A.lo B.hi
    let q10 := divUp A.hi B.lo
    let q11 := divUp A.hi B.hi

    have hp00NaN : isNaN p00 = false := isNaN_divDown_eq_false_of_isFinite (x := A.lo) (y := B.lo)
      hAlo hBlo hyLo0
    have hp01NaN : isNaN p01 = false := isNaN_divDown_eq_false_of_isFinite (x := A.lo) (y := B.hi)
      hAlo hBhi hyHi0
    have hp10NaN : isNaN p10 = false := isNaN_divDown_eq_false_of_isFinite (x := A.hi) (y := B.lo)
      hAhi hBlo hyLo0
    have hp11NaN : isNaN p11 = false := isNaN_divDown_eq_false_of_isFinite (x := A.hi) (y := B.hi)
      hAhi hBhi hyHi0

    have hq00NaN : isNaN q00 = false := isNaN_divUp_eq_false_of_isFinite (x := A.lo) (y := B.lo)
      hAlo hBlo hyLo0
    have hq01NaN : isNaN q01 = false := isNaN_divUp_eq_false_of_isFinite (x := A.lo) (y := B.hi)
      hAlo hBhi hyHi0
    have hq10NaN : isNaN q10 = false := isNaN_divUp_eq_false_of_isFinite (x := A.hi) (y := B.lo)
      hAhi hBlo hyLo0
    have hq11NaN : isNaN q11 = false := isNaN_divUp_eq_false_of_isFinite (x := A.hi) (y := B.hi)
      hAhi hBhi hyHi0

    have hp00_le : toEReal p00 ≤ ((toReal A.lo / toReal B.lo : ℝ) : EReal) :=
      toEReal_divDown_le (x := A.lo) (y := B.lo) hAlo hBlo hyLo0
    have hp01_le : toEReal p01 ≤ ((toReal A.lo / toReal B.hi : ℝ) : EReal) :=
      toEReal_divDown_le (x := A.lo) (y := B.hi) hAlo hBhi hyHi0
    have hp10_le : toEReal p10 ≤ ((toReal A.hi / toReal B.lo : ℝ) : EReal) :=
      toEReal_divDown_le (x := A.hi) (y := B.lo) hAhi hBlo hyLo0
    have hp11_le : toEReal p11 ≤ ((toReal A.hi / toReal B.hi : ℝ) : EReal) :=
      toEReal_divDown_le (x := A.hi) (y := B.hi) hAhi hBhi hyHi0

    have hq00_ge : ((toReal A.lo / toReal B.lo : ℝ) : EReal) ≤ toEReal q00 :=
      toEReal_divUp_ge (x := A.lo) (y := B.lo) hAlo hBlo hyLo0
    have hq01_ge : ((toReal A.lo / toReal B.hi : ℝ) : EReal) ≤ toEReal q01 :=
      toEReal_divUp_ge (x := A.lo) (y := B.hi) hAlo hBhi hyHi0
    have hq10_ge : ((toReal A.hi / toReal B.lo : ℝ) : EReal) ≤ toEReal q10 :=
      toEReal_divUp_ge (x := A.hi) (y := B.lo) hAhi hBlo hyLo0
    have hq11_ge : ((toReal A.hi / toReal B.hi : ℝ) : EReal) ≤ toEReal q11 :=
      toEReal_divUp_ge (x := A.hi) (y := B.hi) hAhi hBhi hyHi0

    -- Evaluate endpoints of the executable interval.
    have hlo_eval :
        toEReal (Interval32.div A B).lo =
          min (min (toEReal p00) (toEReal p01)) (min (toEReal p10) (toEReal p11)) := by
      simp [Interval32.div, hcz, p00, p01, p10, p11,
        toEReal_min4_eq (a := p00) (b := p01) (c := p10) (d := p11) hp00NaN hp01NaN hp10NaN hp11NaN]

    have hhi_eval :
        toEReal (Interval32.div A B).hi =
          max (max (toEReal q00) (toEReal q01)) (max (toEReal q10) (toEReal q11)) := by
      simp [Interval32.div, hcz, q00, q01, q10, q11,
        toEReal_max4_eq (a := q00) (b := q01) (c := q10) (d := q11) hq00NaN hq01NaN hq10NaN hq11NaN]

    -- Lower endpoint ≤ min exact corners.
    have hlo_le_minCorners :
        toEReal (Interval32.div A B).lo ≤
          ((min4R (toReal A.lo / toReal B.lo) (toReal A.lo / toReal B.hi)
            (toReal A.hi / toReal B.lo) (toReal A.hi / toReal B.hi) : ℝ) : EReal) := by
      rw [hlo_eval]
      have h01 :
          min (toEReal p00) (toEReal p01) ≤
            min ((toReal A.lo / toReal B.lo : ℝ) : EReal) ((toReal A.lo / toReal B.hi : ℝ) : EReal)
              :=
        min_le_min hp00_le hp01_le
      have h23 :
          min (toEReal p10) (toEReal p11) ≤
            min ((toReal A.hi / toReal B.lo : ℝ) : EReal) ((toReal A.hi / toReal B.hi : ℝ) : EReal)
              :=
        min_le_min hp10_le hp11_le
      have houter :
          min (min (toEReal p00) (toEReal p01)) (min (toEReal p10) (toEReal p11)) ≤
            min (min ((toReal A.lo / toReal B.lo : ℝ) : EReal) ((toReal A.lo / toReal B.hi : ℝ) :
              EReal))
              (min ((toReal A.hi / toReal B.lo : ℝ) : EReal) ((toReal A.hi / toReal B.hi : ℝ) :
                EReal)) :=
        min_le_min h01 h23
      -- Convert the nested `min` of exact corners into `(min4R ...) : EReal`.
      have hnest :
          min (min ((toReal A.lo / toReal B.lo : ℝ) : EReal) ((toReal A.lo / toReal B.hi : ℝ) :
            EReal))
              (min ((toReal A.hi / toReal B.lo : ℝ) : EReal) ((toReal A.hi / toReal B.hi : ℝ) :
                EReal)) =
            ((min4R (toReal A.lo / toReal B.lo) (toReal A.lo / toReal B.hi)
              (toReal A.hi / toReal B.lo) (toReal A.hi / toReal B.hi) : ℝ) : EReal) := by
        simp [min4R, coe_min]
      exact le_trans houter (le_of_eq hnest)

    have hhi_ge_maxCorners :
        ((max4R (toReal A.lo / toReal B.lo) (toReal A.lo / toReal B.hi)
          (toReal A.hi / toReal B.lo) (toReal A.hi / toReal B.hi) : ℝ) : EReal) ≤
          toEReal (Interval32.div A B).hi := by
      rw [hhi_eval]
      have h01 :
          max ((toReal A.lo / toReal B.lo : ℝ) : EReal) ((toReal A.lo / toReal B.hi : ℝ) : EReal) ≤
            max (toEReal q00) (toEReal q01) :=
        max_le_max hq00_ge hq01_ge
      have h23 :
          max ((toReal A.hi / toReal B.lo : ℝ) : EReal) ((toReal A.hi / toReal B.hi : ℝ) : EReal) ≤
            max (toEReal q10) (toEReal q11) :=
        max_le_max hq10_ge hq11_ge
      have houter :
          max (max ((toReal A.lo / toReal B.lo : ℝ) : EReal) ((toReal A.lo / toReal B.hi : ℝ) :
            EReal))
              (max ((toReal A.hi / toReal B.lo : ℝ) : EReal) ((toReal A.hi / toReal B.hi : ℝ) :
                EReal)) ≤
            max (max (toEReal q00) (toEReal q01)) (max (toEReal q10) (toEReal q11)) :=
        max_le_max h01 h23
      have hnest :
          ((max4R (toReal A.lo / toReal B.lo) (toReal A.lo / toReal B.hi)
              (toReal A.hi / toReal B.lo) (toReal A.hi / toReal B.hi) : ℝ) : EReal) =
            max (max ((toReal A.lo / toReal B.lo : ℝ) : EReal) ((toReal A.lo / toReal B.hi : ℝ) :
              EReal))
                (max ((toReal A.hi / toReal B.lo : ℝ) : EReal) ((toReal A.hi / toReal B.hi : ℝ) :
                  EReal)) := by
        simp [max4R, coe_max]
      -- Rewrite `max4R` then apply `houter`.
      exact le_trans (le_of_eq hnest) houter

    -- Combine with the real bound `minCorners ≤ x/y ≤ maxCorners`.
    have hmin_le : ((min4R (toReal A.lo / toReal B.lo) (toReal A.lo / toReal B.hi)
          (toReal A.hi / toReal B.lo) (toReal A.hi / toReal B.hi) : ℝ) : EReal) ≤ (x / y : EReal) :=
            by
      exact (EReal.coe_le_coe_iff).2 hxy_lo
    have hmax_ge : (x / y : EReal) ≤ ((max4R (toReal A.lo / toReal B.lo) (toReal A.lo / toReal B.hi)
          (toReal A.hi / toReal B.lo) (toReal A.hi / toReal B.hi) : ℝ) : EReal) := by
      exact (EReal.coe_le_coe_iff).2 hxy_hi

    constructor
    · exact le_trans hlo_le_minCorners hmin_le
    · exact le_trans hmax_ge hhi_ge_maxCorners

/-!
## Reciprocal as a special case of division

`Interval32.inv B` is defined as `Interval32.div (point posOne) B`. To state its soundness in the
expected mathematical form (`1 / y`), we need a small sanity lemma about the float constant
`posOne`.
-/

/-- `pow2 k` is definitionaly `2^k`. (A small lemma for decoding float constants.) -/
private lemma pow2_eq_two_pow (k : Nat) : (pow2 k) = 2 ^ k := by
  -- `pow2 k` is `1 <<< k`.
  simp [IEEE32Exec.pow2, Nat.shiftLeft_eq]

/-- The IEEE32 constant `posOne` decodes to the real number `1`. -/
private lemma toReal_posOne : toReal (posOne : IEEE32Exec) = 1 := by
  -- `posOne` is the binary32 constant `0x3F800000`, i.e. `mkBits false 127 0`.
  have hbits : (0x3F800000 : UInt32) = mkBits false 127 0 := by decide
  have hdy_mk :
      toDyadic? (ofBits (mkBits false 127 0)) =
        some { sign := false, mant := pow2 23, exp := ((Int.ofNat 127) - 150) } := by
    -- Use the general bitfield decoding theorem for finite `mkBits`.
    simpa using
      (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 127) (frac := 0) (by decide) (by decide))
  have hexp : (Int.ofNat 127) - 150 = (-23 : Int) := by decide
  have hdy :
      toDyadic? (posOne : IEEE32Exec) =
        some { sign := false, mant := pow2 23, exp := (-23 : Int) } := by
    simpa [IEEE32Exec.posOne, hbits, hexp] using hdy_mk
  -- Evaluate `toReal` and simplify the dyadic interpretation.
  have hpow2 : ((pow2 23 : Nat) : ℝ) = (2 : ℝ) ^ (23 : Nat) := by
    -- (We only need the concrete case `k = 23`; proving it by computation is simplest here.)
    have hpow2Nat : pow2 23 = 2 ^ 23 := by
      simpa using (pow2_eq_two_pow (k := 23))
    calc
      ((pow2 23 : Nat) : ℝ) = ((2 ^ 23 : Nat) : ℝ) := by
        exact congrArg (fun n : Nat => (n : ℝ)) hpow2Nat
      _ = (2 : ℝ) ^ (23 : Nat) := by
        simpa using (Nat.cast_pow (α := ℝ) 2 23)
  have hneg : (-23 : Int) = - (Int.ofNat 23) := by decide
  have hpow_ne0 : (2 : ℝ) ^ (23 : Nat) ≠ 0 := by
    exact pow_ne_zero 23 (two_ne_zero : (2 : ℝ) ≠ 0)
  -- `2^23 * 2^(-23) = 1`.
  calc
    toReal (posOne : IEEE32Exec)
        = dyadicToReal { sign := false, mant := pow2 23, exp := (-23 : Int) } := by
            simp [toReal_eq, hdy]
    _ = (pow2 23 : ℝ) * TorchLean.Floats.neuralBpow binaryRadix (-23 : Int) := by
          simp [dyadicToReal]
    _ = ((2 : ℝ) ^ (23 : Nat)) * (((2 : ℝ) ^ (23 : Nat))⁻¹) := by
          -- Rewrite `pow2 23` and `neural_bpow` to powers of `2`.
          simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, hpow2, hneg,
            zpow_neg,
            zpow_ofNat]
    _ = 1 := by
          simp [hpow_ne0]

/--
Soundness of `Interval32.inv` w.r.t. real reciprocal.

This is derived from `div_sound` using the definition `inv B = div (point posOne) B` plus the
fact that `posOne` decodes to the real constant `1`.
-/
theorem inv_sound (B : Interval32) (hB : Valid B) :
    ∀ {y : ℝ},
      y ∈ Set.Icc (toReal B.lo) (toReal B.hi) →
        toEReal (Interval32.inv B).lo ≤ (1 / y : EReal) ∧
          (1 / y : EReal) ≤ toEReal (Interval32.inv B).hi := by
  intro y hy
  -- `inv B = div (point 1) B`.
  have hA : Valid (Interval32.point (posOne : IEEE32Exec)) := by
    -- `posOne` is finite and `point` is trivially ordered.
    refine ⟨?_, ?_, ?_⟩
    · decide
    · decide
    · -- `posOne ≤ posOne` because `compare posOne posOne = some .eq`.
      have hfin : isFinite (posOne : IEEE32Exec) = true := by decide
      have hcmp : compare (posOne : IEEE32Exec) (posOne : IEEE32Exec) = some .eq :=
        (compare_eq_some_eq_iff_toReal_eq_of_isFinite (x := (posOne : IEEE32Exec))
          (y := (posOne : IEEE32Exec)) hfin hfin).2 rfl
      -- Unfold `≤` and discharge the `compare` case split.
      simp [Interval32.point, IEEE32Exec.le, hcmp]
  -- Specialize `x = 1`.
  have hx : (1 : ℝ) ∈ Set.Icc (toReal (posOne : IEEE32Exec)) (toReal (posOne : IEEE32Exec)) := by
    rw [toReal_posOne]
    simp
  have hdiv :=
    div_sound (A := Interval32.point (posOne : IEEE32Exec)) (B := B) hA hB (x := 1) (y := y) hx hy
  simpa [Interval32.inv] using hdiv

end

end Interval32

end IEEE32Exec

end TorchLean.Floats.IEEE754
