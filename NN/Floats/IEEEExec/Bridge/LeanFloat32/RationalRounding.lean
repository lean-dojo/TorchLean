/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Floats.IEEEExec.Bridge.FP32.RatBounds
import NN.Floats.IEEEExec.Rounding.RoundQuotEvenBounds
public import NN.Floats.IEEEExec.Bridge.LeanFloat32.Rounding

/-!
# Rational rounding in the Lean Float32 bridge

Lean's floating-point model represents an inexact quotient by its integer quotient together with
an `Accuracy` value computed from the remainder. `IEEE32Exec` instead rounds the original rational
number after scaling its denominator by a power of two. This file proves that these two views remain
equivalent after any number of right shifts.

The results are stated for arbitrary natural numerators, nonzero natural denominators, and arbitrary
binary shifts. They are therefore independent of binary32's exponent range and of the division
operation that eventually supplies the quotient.

## References

- IEEE Standard for Floating-Point Arithmetic, IEEE 754-2019, Section 4.3.1.
- Lean, `Init.Data.Float.Model.Unpacked.Operations.Div`.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754.Float32Bridge

open Float.Model.UnpackedFloat
open IEEE32Exec

private theorem bpow_interval_exponent_unique
    (x : ℝ) (k l : Int)
    (hkLower : neuralBpow binaryRadix k ≤ x)
    (hkUpper : x < neuralBpow binaryRadix (k + 1))
    (hlLower : neuralBpow binaryRadix l ≤ x)
    (hlUpper : x < neuralBpow binaryRadix (l + 1)) :
    k = l := by
  by_contra hne
  rcases lt_or_gt_of_ne hne with hkl | hlk
  · have hsucc : k + 1 ≤ l := by grind
    have hpow : neuralBpow binaryRadix (k + 1) ≤ neuralBpow binaryRadix l := by
      simpa [neuralBpow, binaryRadix, NeuralRadix.toReal] using
        (zpow_le_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)).2 hsucc
    exact (not_lt_of_ge (hpow.trans hlLower)) hkUpper
  · have hsucc : l + 1 ≤ k := by grind
    have hpow : neuralBpow binaryRadix (l + 1) ≤ neuralBpow binaryRadix k := by
      simpa [neuralBpow, binaryRadix, NeuralRadix.toReal] using
        (zpow_le_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)).2 hsucc
    exact (not_lt_of_ge (hpow.trans hkLower)) hlUpper

/-- Characterize `floorLog2Rat` by the unique binary interval containing a positive rational. -/
theorem floorLog2Rat_eq_of_bounds
    (num den : Nat) (k : Int) (hnum : num ≠ 0) (hden : den ≠ 0)
    (hlower : neuralBpow binaryRadix k ≤ (num : ℝ) / (den : ℝ))
    (hupper : (num : ℝ) / (den : ℝ) < neuralBpow binaryRadix (k + 1)) :
    floorLog2Rat num den = k := by
  have hfloor := floorLog2Rat_bounds num den hnum hden
  exact bpow_interval_exponent_unique ((num : ℝ) / (den : ℝ))
    (floorLog2Rat num den) k hfloor.1 hfloor.2 hlower hupper

/-- A positive rational below one has a negative binary floor logarithm. -/
theorem floorLog2Rat_lt_zero_of_lt
    (num den : Nat) (hnum : num ≠ 0) (hden : den ≠ 0) (hlt : num < den) :
    floorLog2Rat num den < 0 := by
  let k := floorLog2Rat num den
  have hbounds := floorLog2Rat_bounds num den hnum hden
  have hratioLt : (num : ℝ) / (den : ℝ) < 1 := by
    rw [div_lt_one]
    · exact_mod_cast hlt
    · exact_mod_cast Nat.pos_of_ne_zero hden
  by_contra hk
  have hkNonneg : (0 : Int) ≤ k := by grind
  have hbpowMono : neuralBpow binaryRadix 0 ≤ neuralBpow binaryRadix k := by
    rw [neuralBpow_le_neuralBpow_iff]
    exact hkNonneg
  have hone : (1 : ℝ) ≤ (num : ℝ) / (den : ℝ) := by
    have := hbpowMono.trans hbounds.1
    simpa [k, neuralBpow, binaryRadix, NeuralRadix.toReal] using this
  linarith

/--
When the integer quotient is positive, its most-significant bit also locates the exact rational
between consecutive powers of two.

The positivity condition is essential: if `num / den = 0`, integer division has discarded the
leading fractional bits and cannot determine the rational's binary exponent.
-/
theorem floorLog2Rat_eq_log2_div
    (num den : Nat) (hden : den ≠ 0) (hquot : 0 < num / den) :
    floorLog2Rat num den = Int.ofNat (Nat.log2 (num / den)) := by
  have hnum : num ≠ 0 := by
    intro h
    simp [h] at hquot
  let q := num / den
  have hq : q ≠ 0 := Nat.ne_of_gt hquot
  have hdenPos : 0 < den := Nat.pos_of_ne_zero hden
  have hqLowerNat : 2 ^ q.log2 ≤ q := by
    simpa [Nat.log2_eq_log_two] using Nat.pow_log_le_self 2 hq
  have hqUpperNat : q < 2 ^ (q.log2 + 1) := by
    simpa [Nat.log2_eq_log_two] using Nat.lt_pow_succ_log_self (by decide : 1 < 2) q
  have hratioLower :
      neuralBpow binaryRadix (Int.ofNat q.log2) ≤ (num : ℝ) / (den : ℝ) := by
    have hqLeNum : q * den ≤ num := Nat.div_mul_le_self num den
    have hpowLeNum : 2 ^ q.log2 * den ≤ num :=
      (Nat.mul_le_mul_right den hqLowerNat).trans hqLeNum
    have hdenReal : (0 : ℝ) < den := by exact_mod_cast hdenPos
    rw [show neuralBpow binaryRadix (Int.ofNat q.log2) = (2 : ℝ) ^ q.log2 by
      simp [neuralBpow, binaryRadix, NeuralRadix.toReal]]
    rw [le_div_iff₀ hdenReal]
    exact_mod_cast hpowLeNum
  have hratioUpper :
      (num : ℝ) / (den : ℝ) < neuralBpow binaryRadix (Int.ofNat q.log2 + 1) := by
    have hnumLt : num < (q + 1) * den := by
      have h := Nat.lt_div_mul_add (a := num) hdenPos
      simpa [q, Nat.add_mul] using h
    have hqSuccLe : q + 1 ≤ 2 ^ (q.log2 + 1) := by grind
    have hnumLtPow : num < 2 ^ (q.log2 + 1) * den :=
      hnumLt.trans_le (Nat.mul_le_mul_right den hqSuccLe)
    have hdenReal : (0 : ℝ) < den := by exact_mod_cast hdenPos
    rw [show neuralBpow binaryRadix (Int.ofNat q.log2 + 1) =
        (2 : ℝ) ^ (q.log2 + 1) by
      simp only [neuralBpow, binaryRadix, NeuralRadix.toReal]
      calc
        (2 : ℝ) ^ (Int.ofNat q.log2 + 1) = (2 : ℝ) ^ q.log2 * 2 := by
          rw [zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
          simp
        _ = (2 : ℝ) ^ (q.log2 + 1) := by rw [pow_succ]]
    rw [div_lt_iff₀ hdenReal]
    exact_mod_cast hnumLtPow
  exact floorLog2Rat_eq_of_bounds num den (Int.ofNat q.log2) hnum hden
    hratioLower hratioUpper

/-- Scaling an exact rational by `2^exponent` shifts its binary floor logarithm by `exponent`. -/
theorem floorLog2Rat_scaleRatByPow2
    (num den : Nat) (exponent : Int) (hnum : num ≠ 0) (hden : den ≠ 0) :
    floorLog2Rat (scaleRatByPow2 num den exponent).1
        (scaleRatByPow2 num den exponent).2 =
      floorLog2Rat num den + exponent := by
  let scaled := scaleRatByPow2 num den exponent
  have hscaledNum : scaled.1 ≠ 0 :=
    scaleRatByPow2_fst_ne_zero num den exponent hnum
  have hscaledDen : scaled.2 ≠ 0 :=
    scaleRatByPow2_snd_ne_zero num den exponent hden
  let k := floorLog2Rat num den
  have hbase := floorLog2Rat_bounds num den hnum hden
  have hpowPos : 0 < neuralBpow binaryRadix exponent := neuralBpow.pos _ _
  have hreal : (scaled.1 : ℝ) / (scaled.2 : ℝ) =
      ((num : ℝ) / (den : ℝ)) * neuralBpow binaryRadix exponent := by
    simpa [scaled] using scaleRatByPow2_real num den exponent
  have hlower : neuralBpow binaryRadix (k + exponent) ≤
      (scaled.1 : ℝ) / (scaled.2 : ℝ) := by
    rw [hreal, neuralBpow.add_exp]
    exact mul_le_mul_of_nonneg_right hbase.1 hpowPos.le
  have hupper : (scaled.1 : ℝ) / (scaled.2 : ℝ) <
      neuralBpow binaryRadix (k + exponent + 1) := by
    rw [hreal]
    have hmul := mul_lt_mul_of_pos_right hbase.2 hpowPos
    calc
      (num : ℝ) / (den : ℝ) * neuralBpow binaryRadix exponent <
          neuralBpow binaryRadix (k + 1) * neuralBpow binaryRadix exponent := hmul
      _ = neuralBpow binaryRadix (k + 1 + exponent) :=
        (neuralBpow.add_exp binaryRadix (k + 1) exponent).symm
      _ = neuralBpow binaryRadix (k + exponent + 1) := by congr 1; grind
  exact floorLog2Rat_eq_of_bounds scaled.1 scaled.2 (k + exponent)
    hscaledNum hscaledDen hlower hupper

/-- A power-of-two lower bound on a positive rational survives nearest-even integer rounding. -/
theorem pow2_le_roundQuotEven_of_le_div
    (num den exponent : Nat) (hden : den ≠ 0)
    (h : ((pow2 exponent : Nat) : ℝ) ≤ (num : ℝ) / (den : ℝ)) :
    pow2 exponent ≤ roundQuotEven num den := by
  have hdenPos : (0 : ℝ) < (den : Nat) := by
    exact_mod_cast Nat.pos_of_ne_zero hden
  have hmulReal : ((pow2 exponent : Nat) : ℝ) * (den : ℝ) ≤ (num : ℝ) :=
    (le_div_iff₀ hdenPos).mp h
  have hmulNat : pow2 exponent * den ≤ num := by
    exact_mod_cast hmulReal
  exact le_roundQuotEven_of_mul_le num den (pow2 exponent) hden hmulNat

/-- A strict power-of-two upper bound on a rational becomes a weak bound after nearest-even
integer rounding. -/
theorem roundQuotEven_le_pow2_of_div_lt
    (num den exponent : Nat) (hden : den ≠ 0)
    (h : (num : ℝ) / (den : ℝ) < ((pow2 exponent : Nat) : ℝ)) :
    roundQuotEven num den ≤ pow2 exponent := by
  have hdenPos : (0 : ℝ) < (den : Nat) := by
    exact_mod_cast Nat.pos_of_ne_zero hden
  have hmulReal : (num : ℝ) < ((pow2 exponent : Nat) : ℝ) * (den : ℝ) :=
    (div_lt_iff₀ hdenPos).mp h
  have hmulNat : num < pow2 exponent * den := by
    exact_mod_cast hmulReal
  exact roundQuotEven_le_of_lt_mul num den (pow2 exponent) hden hmulNat

/-- Every nonnegative rational strictly below one half rounds to zero. -/
theorem roundQuotEven_eq_zero_of_div_lt_half
    (num den : Nat) (hden : den ≠ 0)
    (h : (num : ℝ) / (den : ℝ) < (1 / 2 : ℝ)) :
    roundQuotEven num den = 0 := by
  have hdenPos : (0 : ℝ) < (den : Nat) := by
    exact_mod_cast Nat.pos_of_ne_zero hden
  have hmulReal : (num : ℝ) < (1 / 2 : ℝ) * (den : ℝ) :=
    (div_lt_iff₀ hdenPos).mp h
  have htwoMulReal : (2 : ℝ) * (num : ℝ) < (den : ℝ) := by
    linarith
  have htwoMulNat : 2 * num < den := by
    exact_mod_cast htwoMulReal
  exact roundQuotEven_eq_zero_of_two_mul_lt num den htwoMulNat

/-- The extended-mantissa representation of the exact rational number `num / den`. -/
def quotientExtendedMantissa (num den : Nat) : ExtendedMantissa :=
  ExtendedMantissa.ofMantissaAndAccuracy (num / den)
    (accuracyOfFraction (num % den) den)

/-- Scaling a nonzero fractional numerator and denominator preserves its rounding direction. -/
theorem accuracyOfFraction_mul_right (remainder denominator factor : Nat)
    (hfactor : factor ≠ 0) :
    accuracyOfFraction (remainder * factor) (denominator * factor) =
      accuracyOfFraction remainder denominator := by
  have hfactorPos : 0 < factor := Nat.pos_of_ne_zero hfactor
  unfold accuracyOfFraction
  simp only [Nat.mul_eq_zero, hfactor, or_false]
  split <;> rename_i hrem
  · rfl
  · congr 1
    cases hcompare : Ord.compare (2 * remainder) denominator with
    | lt =>
        rw [Nat.compare_eq_lt] at hcompare ⊢
        rw [show 2 * (remainder * factor) = (2 * remainder) * factor by ac_rfl]
        exact (Nat.mul_lt_mul_right hfactorPos).2 hcompare
    | eq =>
        rw [Nat.compare_eq_eq] at hcompare ⊢
        rw [show 2 * (remainder * factor) = (2 * remainder) * factor by ac_rfl]
        exact congrArg (· * factor) hcompare
    | gt =>
        rw [Nat.compare_eq_gt] at hcompare ⊢
        rw [show 2 * (remainder * factor) = (2 * remainder) * factor by ac_rfl]
        exact (Nat.mul_lt_mul_right hfactorPos).2 hcompare

/-- A common nonzero scale factor does not change a quotient's extended mantissa. -/
theorem quotientExtendedMantissa_mul_right (num den factor : Nat)
    (hfactor : factor ≠ 0) :
    quotientExtendedMantissa (num * factor) (den * factor) =
      quotientExtendedMantissa num den := by
  have hfactorPos : 0 < factor := Nat.pos_of_ne_zero hfactor
  unfold quotientExtendedMantissa
  rw [Nat.mul_div_mul_right num den hfactorPos]
  rw [Nat.mul_mod_mul_right]
  rw [accuracyOfFraction_mul_right (num % den) den factor hfactor]

/-- Nearest-even quotient rounding depends only on the represented nonnegative rational. -/
theorem roundQuotEven_eq_of_rat_eq
    (num den num' den' : Nat) (hden : den ≠ 0) (hden' : den' ≠ 0)
    (hvalue : (num : ℝ) / (den : ℝ) = (num' : ℝ) / (den' : ℝ)) :
    roundQuotEven num den = roundQuotEven num' den' := by
  have hleft := neural_nearest_even_div_eq_roundQuotEven num den hden
  have hright := neural_nearest_even_div_eq_roundQuotEven num' den' hden'
  apply Int.ofNat_inj.mp
  calc
    Int.ofNat (roundQuotEven num den) =
        TorchLean.Floats.neuralNearestEven ((num : ℝ) / (den : ℝ)) := hleft.symm
    _ = TorchLean.Floats.neuralNearestEven ((num' : ℝ) / (den' : ℝ)) :=
      congrArg TorchLean.Floats.neuralNearestEven hvalue
    _ = Int.ofNat (roundQuotEven num' den') := hright

/-- Successive exact binary scalings may be combined before nearest-even rounding. -/
theorem roundQuotEven_scaleRatByPow2_add
    (num den : Nat) (firstExponent secondExponent : Int) (hden : den ≠ 0) :
    let first := scaleRatByPow2 num den firstExponent
    let second := scaleRatByPow2 first.1 first.2 secondExponent
    let combined := scaleRatByPow2 num den (firstExponent + secondExponent)
    roundQuotEven second.1 second.2 = roundQuotEven combined.1 combined.2 := by
  dsimp only
  apply roundQuotEven_eq_of_rat_eq
  · exact scaleRatByPow2_snd_ne_zero _ _ _
      (scaleRatByPow2_snd_ne_zero num den firstExponent hden)
  · exact scaleRatByPow2_snd_ne_zero num den (firstExponent + secondExponent) hden
  · rw [scaleRatByPow2_real, scaleRatByPow2_real, scaleRatByPow2_real]
    rw [neuralBpow.add_exp]
    ring

/-- Scaling a denominator by `2 ^ shift` is exact binary scaling by `2 ^ (-shift)`. -/
theorem roundQuotEven_mul_pow2_den
    (num den shift : Nat) :
    let scaled := scaleRatByPow2 num den (-Int.ofNat shift)
    roundQuotEven num (den * 2 ^ shift) = roundQuotEven scaled.1 scaled.2 := by
  cases shift with
  | zero => simp [scaleRatByPow2]
  | succ shift =>
      rw [show -Int.ofNat (Nat.succ shift) = Int.negSucc shift by rfl]
      simp [scaleRatByPow2, Nat.shiftLeft_eq]

/--
Round a quotient at a chosen binary exponent either directly or by first restoring the quotient's
exact external exponent. This is the normalization identity shared by division and conversions
from exact rational values.
-/
theorem roundQuotEven_shift_to_exponent
    (num den : Nat) (exponent targetExponent : Int) (hden : den ≠ 0)
    (hexponent : exponent ≤ targetExponent) :
    let shift := (targetExponent - exponent).toNat
    let exact := scaleRatByPow2 num den exponent
    let normalized := scaleRatByPow2 exact.1 exact.2 (-targetExponent)
    roundQuotEven num (den * 2 ^ shift) =
      roundQuotEven normalized.1 normalized.2 := by
  dsimp only
  let shift := (targetExponent - exponent).toNat
  have hshift : (shift : Int) = targetExponent - exponent := by
    exact Int.toNat_of_nonneg (by grind)
  calc
    roundQuotEven num (den * 2 ^ shift) =
        roundQuotEven (scaleRatByPow2 num den (-Int.ofNat shift)).1
          (scaleRatByPow2 num den (-Int.ofNat shift)).2 :=
      roundQuotEven_mul_pow2_den num den shift
    _ = roundQuotEven (scaleRatByPow2 num den (exponent + -targetExponent)).1
          (scaleRatByPow2 num den (exponent + -targetExponent)).2 := by
      congr 2 <;> congr 1 <;> grind
    _ = roundQuotEven
          (scaleRatByPow2 (scaleRatByPow2 num den exponent).1
            (scaleRatByPow2 num den exponent).2 (-targetExponent)).1
          (scaleRatByPow2 (scaleRatByPow2 num den exponent).1
            (scaleRatByPow2 num den exponent).2 (-targetExponent)).2 :=
      (roundQuotEven_scaleRatByPow2_add num den exponent (-targetExponent) hden).symm

@[simp] private theorem ofMantissaAndAccuracy_mantissa
    (mantissa : Nat) (accuracy : Accuracy) :
    (ExtendedMantissa.ofMantissaAndAccuracy mantissa accuracy).mantissa = mantissa := by
  cases accuracy with
  | exact => rfl
  | inexact ordering => cases ordering <;> rfl

@[simp] private theorem ofMantissaAndAccuracy_accuracy
    (mantissa : Nat) (accuracy : Accuracy) :
    (ExtendedMantissa.ofMantissaAndAccuracy mantissa accuracy).accuracy = accuracy := by
  cases accuracy with
  | exact => rfl
  | inexact ordering => cases ordering <;> rfl

/--
Shifting a quotient's extended mantissa right once is equivalent to doubling its denominator.

The proof handles the even and odd low-bit cases separately. When the quotient is odd, that bit
becomes the new round bit; any pre-existing nonzero remainder becomes sticky information.
-/
private theorem quotientExtendedMantissa_shiftRightOne
    (num den : Nat) (hden : den ≠ 0) :
    ExtendedMantissa.shiftRightOne (quotientExtendedMantissa num den) =
      quotientExtendedMantissa num (den * 2) := by
  have hdenPos : 0 < den := Nat.pos_of_ne_zero hden
  have hremLt : num % den < den := Nat.mod_lt num hdenPos
  have hmantissa : num / den / 2 = num / (den * 2) := Nat.div_div_eq_div_mul num den 2
  rcases Nat.mod_two_eq_zero_or_one (num / den) with hbit | hbit
  · have hrem : num % (den * 2) = num % den := by
      have hdecomp := Nat.div_add_mod (num % (den * 2)) den
      rw [Nat.mod_mul_right_div_self, Nat.mod_mul_right_mod, hbit] at hdecomp
      simpa using hdecomp.symm
    by_cases hz : num % den = 0
    · simp [quotientExtendedMantissa, ExtendedMantissa.shiftRightOne,
        ExtendedMantissa.ofMantissaAndAccuracy, accuracyOfFraction, hz, hrem, hbit,
        hmantissa]
    · have hcompare : compare (2 * (num % (den * 2))) (den * 2) = .lt := by
        apply Nat.compare_eq_lt.mpr
        rw [hrem]
        grind
      have hcompare' : compare (2 * (num % den)) (den * 2) = .lt := by
        simpa [hrem] using hcompare
      cases hacc : compare (2 * (num % den)) den <;>
        simp [quotientExtendedMantissa, ExtendedMantissa.shiftRightOne,
          ExtendedMantissa.ofMantissaAndAccuracy, accuracyOfFraction, hz, hrem, hbit,
          hcompare', hacc, hmantissa]
  · have hrem : num % (den * 2) = den + num % den := by
      have hdecomp := Nat.div_add_mod (num % (den * 2)) den
      rw [Nat.mod_mul_right_div_self, Nat.mod_mul_right_mod, hbit] at hdecomp
      simpa [Nat.add_comm] using hdecomp.symm
    by_cases hz : num % den = 0
    · have hcompare' : Ord.compare (2 * den) (den * 2) = Ordering.eq :=
        Nat.compare_eq_eq.mpr (Nat.mul_comm 2 den)
      simp [quotientExtendedMantissa, ExtendedMantissa.shiftRightOne,
        ExtendedMantissa.ofMantissaAndAccuracy, accuracyOfFraction, hz, hrem, hbit,
        hcompare', hmantissa, hden]
    · have hcompare : compare (2 * (num % (den * 2))) (den * 2) = .gt := by
        apply Nat.compare_eq_gt.mpr
        rw [hrem]
        grind
      have hcompare' : compare (2 * (den + num % den)) (den * 2) = .gt := by
        simpa [hrem] using hcompare
      cases hacc : compare (2 * (num % den)) den <;>
        simp [quotientExtendedMantissa, ExtendedMantissa.shiftRightOne,
          ExtendedMantissa.ofMantissaAndAccuracy, accuracyOfFraction, hz, hrem, hbit,
          hcompare', hacc, hmantissa]

/-- Shifting a quotient right by `shift` bits scales its denominator by `2 ^ shift`. -/
theorem quotientExtendedMantissa_shift (num den shift : Nat) (hden : den ≠ 0) :
    quotientExtendedMantissa num den >>> shift =
      quotientExtendedMantissa num (den * 2 ^ shift) := by
  induction shift with
  | zero => simp [HShiftRight.hShiftRight, Nat.repeat]
  | succ shift ih =>
      rw [show quotientExtendedMantissa num den >>> (shift + 1) =
          ExtendedMantissa.shiftRightOne
            (quotientExtendedMantissa num den >>> shift) by rfl]
      rw [ih]
      have hpow : den * 2 ^ (shift + 1) = (den * 2 ^ shift) * 2 := by
        rw [pow_succ]
        ac_rfl
      rw [hpow]
      apply quotientExtendedMantissa_shiftRightOne
      exact mul_ne_zero hden (pow_ne_zero _ (by decide))

/-- Adding quotient bits by shifting the numerator left is undone by the matching right shift. -/
theorem quotientExtendedMantissa_shiftLeft_shiftRight
    (num den shift : Nat) (hden : den ≠ 0) :
    quotientExtendedMantissa (num <<< shift) den >>> shift =
      quotientExtendedMantissa num den := by
  rw [quotientExtendedMantissa_shift (num <<< shift) den shift hden]
  rw [Nat.shiftLeft_eq]
  exact quotientExtendedMantissa_mul_right num den (2 ^ shift)
    (pow_ne_zero _ (by decide))

/--
Nearest-even rounding after shifting a quotient agrees with rounding the rational number whose
denominator has been scaled by the same power of two.
-/
theorem roundedMantissa_shift_quotient (num den shift : Nat) (hden : den ≠ 0) :
    (ExtendedMantissa.ofMantissaAndAccuracy (num / den)
        (accuracyOfFraction (num % den) den) >>> shift).roundedMantissa =
      roundQuotEven num (den * 2 ^ shift) := by
  change (quotientExtendedMantissa num den >>> shift).roundedMantissa = _
  rw [quotientExtendedMantissa_shift num den shift hden]
  unfold quotientExtendedMantissa ExtendedMantissa.roundedMantissa
  simpa using roundToNearestEven_accuracyOfFraction_mod num (den * 2 ^ shift)
    (mul_ne_zero hden (pow_ne_zero _ (by decide)))

/--
Normal form for Lean's quotient-rounding operation: after choosing the format's target exponent,
its rounded mantissa is a single `roundQuotEven` call on the correspondingly scaled denominator.
-/
theorem roundWithAccuracy_quotient_eq_finishRoundedMantissa
    (spec : Float.Model.Format) (sign : Sign) (num den : Nat) (exponent : Int)
    (hden : den ≠ 0) :
    Float.Model.UnpackedFloat.roundWithAccuracy spec sign (num / den) exponent
        (accuracyOfFraction (num % den) den) =
      let shift :=
        (spec.targetExponent (Float.Model.totalExponent (num / den) exponent) - exponent).toNat
      finishRoundedMantissa spec sign
        (roundQuotEven num (den * 2 ^ shift), exponent + shift) := by
  unfold Float.Model.UnpackedFloat.roundWithAccuracy
    Float.Model.UnpackedFloat.shiftToTargetExponent
    Float.Model.UnpackedFloat.shiftToExponent
  simp only
  rw [roundedMantissa_shift_quotient num den _ hden]
  rfl

/--
The binary32 target exponent for a positive quotient is the larger of the normal rounding
position `k - 23` and the subnormal floor `-149`, where `k` is the binary exponent of the exact
rational value after applying its external exponent.
-/
theorem binary32_targetExponent_quotient_eq_max
    (num den : Nat) (exponent : Int)
    (hnum : num ≠ 0) (hden : den ≠ 0) (hquot : 0 < num / den) :
    let exact := scaleRatByPow2 num den exponent
    let k := floorLog2Rat exact.1 exact.2
    Float.Model.Format.binary32.targetExponent
        (Float.Model.totalExponent (num / den) exponent) =
      max (k - 23) (-149) := by
  dsimp only
  have hk : floorLog2Rat (scaleRatByPow2 num den exponent).1
        (scaleRatByPow2 num den exponent).2 =
      Int.ofNat (Nat.log2 (num / den)) + exponent := by
    rw [floorLog2Rat_scaleRatByPow2 num den exponent hnum hden]
    rw [floorLog2Rat_eq_log2_div num den hden hquot]
  rw [hk]
  simp [Float.Model.Format.targetExponent, Float.Model.totalExponent,
    Float.Model.Format.mantissaBits, Float.Model.Format.minExponent]
  congr 1
  grind

/--
The target-exponent formula remains valid when the integer quotient is zero, provided the external
exponent is already no larger than the format's rounding target. This is the extreme-underflow case
produced by `divCore`: the exact value is still carried by the remainder and external exponent.
-/
theorem binary32_targetExponent_quotient_eq_max_of_le
    (num den : Nat) (exponent : Int)
    (hnum : num ≠ 0) (hden : den ≠ 0)
    (hexponent : exponent ≤ Float.Model.Format.binary32.targetExponent
      (Float.Model.totalExponent (num / den) exponent)) :
    let exact := scaleRatByPow2 num den exponent
    let k := floorLog2Rat exact.1 exact.2
    Float.Model.Format.binary32.targetExponent
        (Float.Model.totalExponent (num / den) exponent) =
      max (k - 23) (-149) := by
  dsimp only
  by_cases hquot : 0 < num / den
  · exact binary32_targetExponent_quotient_eq_max num den exponent
      hnum hden hquot
  · have hquotZero : num / den = 0 := Nat.eq_zero_of_not_pos hquot
    have hlt : num < den :=
      (Nat.div_eq_zero_iff_lt (Nat.pos_of_ne_zero hden)).1 hquotZero
    have hkBase : floorLog2Rat num den < 0 :=
      floorLog2Rat_lt_zero_of_lt num den hnum hden hlt
    have hkScale :
        floorLog2Rat (scaleRatByPow2 num den exponent).1
            (scaleRatByPow2 num den exponent).2 =
          floorLog2Rat num den + exponent :=
      floorLog2Rat_scaleRatByPow2 num den exponent hnum hden
    have hexponentFloor : exponent ≤ -149 := by
      unfold Float.Model.Format.targetExponent Float.Model.totalExponent at hexponent
      simp only [hquotZero, Nat.log2_zero, Float.Model.Format.mantissaBits,
        Float.Model.Format.minExponent, Nat.reduceSub] at hexponent
      norm_num at hexponent
      grind
    have htarget : Float.Model.Format.binary32.targetExponent
        (Float.Model.totalExponent (num / den) exponent) = -149 := by
      unfold Float.Model.Format.targetExponent Float.Model.totalExponent
      simp only [hquotZero, Nat.log2_zero, Float.Model.Format.mantissaBits,
        Float.Model.Format.minExponent, Nat.reduceSub]
      rw [max_eq_right]
      · norm_num
      · norm_num
        grind
    rw [htarget, max_eq_right]
    rw [hkScale]
    grind

/--
For a quotient whose external exponent is no larger than binary32's target exponent, Lean's
`roundWithAccuracy` path is exactly nearest-even rounding of the correspondingly normalized
rational value followed by the shared mantissa-finishing stage.
-/
theorem binary32_roundWithAccuracy_quotient_eq_finishRoundedMantissa
    (sign : Sign) (num den : Nat) (exponent : Int) (hden : den ≠ 0)
    (hexponent : exponent ≤ Float.Model.Format.binary32.targetExponent
      (Float.Model.totalExponent (num / den) exponent)) :
    let target := Float.Model.Format.binary32.targetExponent
      (Float.Model.totalExponent (num / den) exponent)
    let exact := scaleRatByPow2 num den exponent
    let normalized := scaleRatByPow2 exact.1 exact.2 (-target)
    Float.Model.UnpackedFloat.roundWithAccuracy Float.Model.Format.binary32 sign
        (num / den) exponent (accuracyOfFraction (num % den) den) =
      finishRoundedMantissa Float.Model.Format.binary32 sign
        (roundQuotEven normalized.1 normalized.2, target) := by
  dsimp only
  let target := Float.Model.Format.binary32.targetExponent
    (Float.Model.totalExponent (num / den) exponent)
  let shift := (target - exponent).toNat
  have hshift : (shift : Int) = target - exponent := by
    exact Int.toNat_of_nonneg (by simpa [target] using sub_nonneg.mpr hexponent)
  rw [roundWithAccuracy_quotient_eq_finishRoundedMantissa
    Float.Model.Format.binary32 sign num den exponent hden]
  dsimp only
  rw [roundQuotEven_shift_to_exponent num den exponent target hden
    (by simpa [target] using hexponent)]
  congr 2
  simpa [shift] using (show exponent + (shift : Int) = target by grind)

/--
Lean's binary32 mantissa-finishing stage agrees with `IEEE32Exec`'s direct exact-rational
rounder. The statement covers strict underflow, subnormals, the smallest normal, ordinary normal
values, a carry into the next exponent, and overflow to infinity.
-/
theorem model_finishRoundedMantissa_quotient_eq_roundRatToIEEE32
    (sign : Sign) (num den : Nat) (hnum : num ≠ 0) (hden : den ≠ 0) :
    let k := floorLog2Rat num den
    let target := max (k - 23) (-149)
    let normalized := scaleRatByPow2 num den (-target)
    modelToIEEE32Exec
        (Float32.Model.pack
          (finishRoundedMantissa Float.Model.Format.binary32 sign
            (roundQuotEven normalized.1 normalized.2, target))) =
      roundRatToIEEE32 (signToBool sign) num den := by
  dsimp only
  let k := floorLog2Rat num den
  let target := max (k - 23) (-149)
  let normalized := scaleRatByPow2 num den (-target)
  have hnormalizedNum : normalized.1 ≠ 0 :=
    scaleRatByPow2_fst_ne_zero num den (-target) hnum
  have hnormalizedDen : normalized.2 ≠ 0 :=
    scaleRatByPow2_snd_ne_zero num den (-target) hden
  have hnormalizedFloor : floorLog2Rat normalized.1 normalized.2 = k - target := by
    simpa [normalized, k, sub_eq_add_neg] using
      floorLog2Rat_scaleRatByPow2 num den (-target) hnum hden
  have hnormalizedBounds := floorLog2Rat_bounds normalized.1 normalized.2
    hnormalizedNum hnormalizedDen
  have hnormalizedLower :
      neuralBpow binaryRadix (k - target) ≤
        (normalized.1 : ℝ) / (normalized.2 : ℝ) := by
    simpa [hnormalizedFloor] using hnormalizedBounds.1
  have hnormalizedUpper :
      (normalized.1 : ℝ) / (normalized.2 : ℝ) <
        neuralBpow binaryRadix (k - target + 1) := by
    simpa [hnormalizedFloor] using hnormalizedBounds.2
  let rounded := roundQuotEven normalized.1 normalized.2
  change modelToIEEE32Exec
      (Float32.Model.pack
        (finishRoundedMantissa Float.Model.Format.binary32 sign (rounded, target))) =
    roundRatToIEEE32 (signToBool sign) num den
  by_cases hkOverflow : 127 < k
  · have htarget : target = k - 23 := by
      simp only [target]
      grind
    have hlowerReal' : (2 : ℝ) ^ 23 ≤
        (normalized.1 : ℝ) / (normalized.2 : ℝ) := by
      simpa [htarget, neuralBpow, binaryRadix, NeuralRadix.toReal,
        pow2_eq_two_pow] using hnormalizedLower
    have hupperReal' : (normalized.1 : ℝ) / (normalized.2 : ℝ) <
        (2 : ℝ) ^ 24 := by
      simpa [htarget, neuralBpow, binaryRadix, NeuralRadix.toReal,
        pow2_eq_two_pow] using hnormalizedUpper
    have hlowerReal : ((pow2 23 : Nat) : ℝ) ≤
        (normalized.1 : ℝ) / (normalized.2 : ℝ) := by
      norm_num [pow2, Nat.shiftLeft_eq] at hlowerReal' ⊢
      exact hlowerReal'
    have hupperReal : (normalized.1 : ℝ) / (normalized.2 : ℝ) <
        ((pow2 24 : Nat) : ℝ) := by
      norm_num [pow2, Nat.shiftLeft_eq] at hupperReal' ⊢
      exact hupperReal'
    have hroundedLower : pow2 23 ≤ rounded :=
      pow2_le_roundQuotEven_of_le_div normalized.1 normalized.2 23
        hnormalizedDen hlowerReal
    have hroundedUpper : rounded ≤ pow2 24 :=
      roundQuotEven_le_pow2_of_div_lt normalized.1 normalized.2 24
        hnormalizedDen hupperReal
    rw [htarget]
    by_cases hcarry : rounded = pow2 24
    · rw [hcarry]
      rw [model_finishRoundedMantissa_normal_carry_overflow sign k (by grind) (by grind)]
      simp [roundRatToIEEE32, hnum, k, hkOverflow]
    · have hroundedHigh : rounded < pow2 24 := by grind
      rw [model_finishRoundedMantissa_normal_overflow sign rounded k hroundedLower
        hroundedHigh (by grind) hkOverflow]
      simp [roundRatToIEEE32, hnum, k, hkOverflow]
  · by_cases hkUnderflow : k < -150
    · have htarget : target = -149 := by
        simp only [target]
        grind
      have hpowLe : neuralBpow binaryRadix (k - target + 1) ≤
          neuralBpow binaryRadix (-1) := by
        rw [neuralBpow_le_neuralBpow_iff]
        grind
      have hratioHalf : (normalized.1 : ℝ) / (normalized.2 : ℝ) <
          (1 / 2 : ℝ) := by
        have hlt := hnormalizedUpper.trans_le hpowLe
        simpa [neuralBpow, binaryRadix, NeuralRadix.toReal] using hlt
      have hroundedZero : rounded = 0 :=
        roundQuotEven_eq_zero_of_div_lt_half normalized.1 normalized.2
          hnormalizedDen hratioHalf
      rw [htarget, hroundedZero, finishRoundedMantissa_binary32_zero,
        modelToIEEE32Exec_pack_zero]
      simp [roundRatToIEEE32, hnum, k, hkOverflow, hkUnderflow]
    · by_cases hkSubnormal : k < -126
      · have htarget : target = -149 := by
          simp only [target]
          grind
        have hpowLe : neuralBpow binaryRadix (k - target + 1) ≤
            neuralBpow binaryRadix 23 := by
          rw [neuralBpow_le_neuralBpow_iff]
          grind
        have hupperReal' : (normalized.1 : ℝ) / (normalized.2 : ℝ) <
            (2 : ℝ) ^ 23 := by
          have hlt := hnormalizedUpper.trans_le hpowLe
          simpa [neuralBpow, binaryRadix, NeuralRadix.toReal] using hlt
        have hupperReal : (normalized.1 : ℝ) / (normalized.2 : ℝ) <
            ((pow2 23 : Nat) : ℝ) := by
          norm_num [pow2, Nat.shiftLeft_eq] at hupperReal' ⊢
          exact hupperReal'
        have hroundedUpper : rounded ≤ pow2 23 :=
          roundQuotEven_le_pow2_of_div_lt normalized.1 normalized.2 23
            hnormalizedDen hupperReal
        have hroundedExec : roundQuotEven (Nat.shiftLeft num 149) den = rounded := by
          simp [rounded, normalized, htarget, scaleRatByPow2]
        have hroundRat :
            roundRatToIEEE32 (signToBool sign) num den =
              if rounded == 0 then
                if signToBool sign then IEEE32Exec.negZero else IEEE32Exec.posZero
              else
                match Nat.decLe (pow2 23) rounded with
                | isTrue _ => IEEE32Exec.ofBits
                    (IEEE32Exec.mkBits (signToBool sign) 1 0)
                | isFalse _ => IEEE32Exec.ofBits
                    (IEEE32Exec.mkBits (signToBool sign) 0 rounded) := by
          unfold roundRatToIEEE32
          simp only [beq_iff_eq, hnum, if_false]
          rw [show floorLog2Rat num den = k by rfl]
          simp only [gt_iff_lt, hkOverflow, if_false, hkUnderflow,
            hkSubnormal, if_true]
          rw [hroundedExec]
          split <;> rfl
        rw [htarget]
        by_cases hzero : rounded = 0
        · rw [hroundRat]
          simp only [beq_iff_eq, if_pos hzero]
          rw [hzero, finishRoundedMantissa_binary32_zero,
            modelToIEEE32Exec_pack_zero]
        · by_cases hcarry : rounded = pow2 23
          · rw [hroundRat]
            simp only [beq_iff_eq, if_neg hzero]
            have hle : pow2 23 ≤ rounded := by grind
            simp [Nat.decLe, hle]
            rw [hcarry, model_finishRoundedMantissa_smallestNormal]
          · have hroundedHigh : rounded < pow2 23 := by grind
            rw [hroundRat]
            simp only [beq_iff_eq, if_neg hzero]
            have hnle : ¬pow2 23 ≤ rounded := by grind
            simp [Nat.decLe, hnle]
            rw [model_finishRoundedMantissa_subnormal sign rounded hzero hroundedHigh]
      · have htarget : target = k - 23 := by
          simp only [target]
          grind
        have hlowerReal' : (2 : ℝ) ^ 23 ≤
            (normalized.1 : ℝ) / (normalized.2 : ℝ) := by
          simpa [htarget, neuralBpow, binaryRadix, NeuralRadix.toReal,
            pow2_eq_two_pow] using hnormalizedLower
        have hupperReal' : (normalized.1 : ℝ) / (normalized.2 : ℝ) <
            (2 : ℝ) ^ 24 := by
          simpa [htarget, neuralBpow, binaryRadix, NeuralRadix.toReal,
            pow2_eq_two_pow] using hnormalizedUpper
        have hlowerReal : ((pow2 23 : Nat) : ℝ) ≤
            (normalized.1 : ℝ) / (normalized.2 : ℝ) := by
          norm_num [pow2, Nat.shiftLeft_eq] at hlowerReal' ⊢
          exact hlowerReal'
        have hupperReal : (normalized.1 : ℝ) / (normalized.2 : ℝ) <
            ((pow2 24 : Nat) : ℝ) := by
          norm_num [pow2, Nat.shiftLeft_eq] at hupperReal' ⊢
          exact hupperReal'
        have hroundedLower : pow2 23 ≤ rounded :=
          pow2_le_roundQuotEven_of_le_div normalized.1 normalized.2 23
            hnormalizedDen hlowerReal
        have hroundedUpper : rounded ≤ pow2 24 :=
          roundQuotEven_le_pow2_of_div_lt normalized.1 normalized.2 24
            hnormalizedDen hupperReal
        have hroundedExec :
            roundQuotEven (scaleRatByPow2 num den (23 - k)).1
                (scaleRatByPow2 num den (23 - k)).2 = rounded := by
          simp [rounded, normalized, htarget]
        have hroundRat :
            roundRatToIEEE32 (signToBool sign) num den =
              let k' : Int := if rounded == pow2 24 then k + 1 else k
              let m' : Nat := if rounded == pow2 24 then pow2 23 else rounded
              if k' > 127 then
                if signToBool sign then IEEE32Exec.negInf else IEEE32Exec.posInf
              else
                IEEE32Exec.ofBits
                  (IEEE32Exec.mkBits (signToBool sign) (k' + 127).toNat
                    (m' - pow2 23)) := by
          unfold roundRatToIEEE32
          simp only [beq_iff_eq, hnum, if_false]
          rw [show floorLog2Rat num den = k by rfl]
          simp only [gt_iff_lt, hkOverflow, if_false, hkUnderflow,
            hkSubnormal, if_false]
          rw [hroundedExec]
        rw [hroundRat, htarget]
        by_cases hcarry : rounded = pow2 24
        · rw [hcarry]
          simp only [beq_iff_eq, if_pos]
          by_cases hkTop : k = 127
          · rw [hkTop]
            rw [model_finishRoundedMantissa_normal_carry_overflow sign 127]
            · simp
            · grind
            · grind
          · have hkHigh : k ≤ 126 := by grind
            rw [model_finishRoundedMantissa_normal_carry sign k (by grind) hkHigh]
            have hkNoCarryOverflow : ¬k + 1 > 127 := by grind
            rw [if_neg hkNoCarryOverflow]
            simp only [Nat.sub_self]
            congr 2
            grind
        · have hroundedHigh : rounded < pow2 24 := by grind
          simp only [beq_iff_eq, if_neg hcarry]
          rw [model_finishRoundedMantissa_normal sign rounded k hroundedLower
            hroundedHigh (by grind) (by grind)]
          simp [hkOverflow]

/--
Lean's quotient-and-remainder rounding agrees with direct exact-rational binary32 rounding whenever
the supplied exponent is already at or below the format's target. Unlike the positive-quotient
normal form, this theorem also covers an intermediate integer quotient of zero during underflow.
-/
theorem model_roundWithAccuracy_quotient_eq_roundRatToIEEE32
    (sign : Sign) (num den : Nat) (exponent : Int)
    (hnum : num ≠ 0) (hden : den ≠ 0)
    (hexponent : exponent ≤ Float.Model.Format.binary32.targetExponent
      (Float.Model.totalExponent (num / den) exponent)) :
    let exact := scaleRatByPow2 num den exponent
    modelToIEEE32Exec
        (Float32.Model.pack
          (Float.Model.UnpackedFloat.roundWithAccuracy
            Float.Model.Format.binary32 sign (num / den) exponent
            (accuracyOfFraction (num % den) den))) =
      roundRatToIEEE32 (signToBool sign) exact.1 exact.2 := by
  dsimp only
  let exact := scaleRatByPow2 num den exponent
  have htarget : Float.Model.Format.binary32.targetExponent
      (Float.Model.totalExponent (num / den) exponent) =
      max (floorLog2Rat exact.1 exact.2 - 23) (-149) := by
    simpa [exact] using
      binary32_targetExponent_quotient_eq_max_of_le
        num den exponent hnum hden hexponent
  rw [binary32_roundWithAccuracy_quotient_eq_finishRoundedMantissa
    sign num den exponent hden hexponent]
  rw [htarget]
  exact model_finishRoundedMantissa_quotient_eq_roundRatToIEEE32
    sign exact.1 exact.2
      (scaleRatByPow2_fst_ne_zero num den exponent hnum)
      (scaleRatByPow2_snd_ne_zero num den exponent hden)

private theorem roundRatToIEEE32_eq_of_rat_eq_sign
    (sign : Sign) (num den num' den' : Nat)
    (hnum : num ≠ 0) (hden : den ≠ 0)
    (hnum' : num' ≠ 0) (hden' : den' ≠ 0)
    (hvalue : (num : ℝ) / (den : ℝ) = (num' : ℝ) / (den' : ℝ)) :
    roundRatToIEEE32 (signToBool sign) num den =
      roundRatToIEEE32 (signToBool sign) num' den' := by
  have hbounds' := floorLog2Rat_bounds num' den' hnum' hden'
  have hk : floorLog2Rat num den = floorLog2Rat num' den' :=
    floorLog2Rat_eq_of_bounds num den (floorLog2Rat num' den') hnum hden
      (by rw [hvalue]; exact hbounds'.1)
      (by rw [hvalue]; exact hbounds'.2)
  let k := floorLog2Rat num den
  let target := max (k - 23) (-149)
  let normalized := scaleRatByPow2 num den (-target)
  let normalized' := scaleRatByPow2 num' den' (-target)
  have hnormalizedDen : normalized.2 ≠ 0 :=
    scaleRatByPow2_snd_ne_zero num den (-target) hden
  have hnormalizedDen' : normalized'.2 ≠ 0 :=
    scaleRatByPow2_snd_ne_zero num' den' (-target) hden'
  have hnormalizedValue :
      (normalized.1 : ℝ) / (normalized.2 : ℝ) =
        (normalized'.1 : ℝ) / (normalized'.2 : ℝ) := by
    simp only [normalized, normalized']
    rw [scaleRatByPow2_real, scaleRatByPow2_real, hvalue]
  have hrounded :
      roundQuotEven normalized.1 normalized.2 =
        roundQuotEven normalized'.1 normalized'.2 :=
    roundQuotEven_eq_of_rat_eq normalized.1 normalized.2
      normalized'.1 normalized'.2 hnormalizedDen hnormalizedDen' hnormalizedValue
  rw [← model_finishRoundedMantissa_quotient_eq_roundRatToIEEE32
    sign num den hnum hden]
  rw [← model_finishRoundedMantissa_quotient_eq_roundRatToIEEE32
    sign num' den' hnum' hden']
  rw [← hk]
  exact congrArg
    (fun mantissa =>
      modelToIEEE32Exec
        (Float32.Model.pack
          (finishRoundedMantissa Float.Model.Format.binary32 sign
            (mantissa, target))))
    hrounded

/--
Exact-rational binary32 rounding is independent of the chosen nonzero numerator and denominator.
In particular, introducing or cancelling a common power of two does not change the result.
-/
theorem roundRatToIEEE32_eq_of_rat_eq
    (sign : Bool) (num den num' den' : Nat)
    (hnum : num ≠ 0) (hden : den ≠ 0)
    (hnum' : num' ≠ 0) (hden' : den' ≠ 0)
    (hvalue : (num : ℝ) / (den : ℝ) = (num' : ℝ) / (den' : ℝ)) :
    roundRatToIEEE32 sign num den = roundRatToIEEE32 sign num' den' := by
  cases sign
  · simpa [signToBool] using
      roundRatToIEEE32_eq_of_rat_eq_sign .positive num den num' den'
        hnum hden hnum' hden' hvalue
  · simpa [signToBool] using
      roundRatToIEEE32_eq_of_rat_eq_sign .negative num den num' den'
        hnum hden hnum' hden' hvalue

end TorchLean.Floats.IEEE754.Float32Bridge
