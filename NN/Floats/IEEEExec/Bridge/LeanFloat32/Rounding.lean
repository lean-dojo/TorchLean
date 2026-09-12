/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32.NearestEven
public import NN.Floats.IEEEExec.Bridge.LeanFloat32.Representation
public import NN.Floats.IEEEExec.Rounding.RoundShiftRightEven

/-!
# Lean Float32 and IEEE32Exec: nearest-even rounding

Lean's `Float32.Model` and TorchLean's `IEEE32Exec` encode discarded mantissa bits differently.
Lean records a round bit and a sticky bit in `ExtendedMantissa`; `IEEE32Exec` rounds an integer
quotient directly. This file proves that the two encodings make the same nearest-even decision.

The result is independent of any particular arithmetic operation. Addition and multiplication use
it after producing an exact integer mantissa, while division and square root can reuse the quotient
lemma once their remainder information has been related to Lean's `Accuracy` type.

## References

- IEEE Standard for Floating-Point Arithmetic, IEEE 754-2019, Section 4.3.1.
- Lean, `Init.Data.Float.Model.Unpacked.Round` and
  `Init.Data.Float.Model.Unpacked.Operations.Div`.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754.Float32Bridge

open Float.Model.UnpackedFloat
open IEEE32Exec

/--
Lean's quotient-plus-accuracy representation makes the same nearest-even choice as
`IEEE32Exec.roundQuotEven`.
-/
theorem roundToNearestEven_accuracyOfFraction_mod
    (num den : Nat) (hden : den ≠ 0) :
    Accuracy.roundToNearestEven (num / den)
        (accuracyOfFraction (num % den) den) =
      roundQuotEven num den := by
  unfold accuracyOfFraction roundQuotEven
  have hdenPos : 0 < den := Nat.pos_of_ne_zero hden
  have hmodLt : num % den < den := Nat.mod_lt num hdenPos
  by_cases hrem : num % den = 0
  · simp [hrem, hdenPos, Accuracy.roundToNearestEven]
  · simp only [hrem, if_false]
    split <;> rename_i hcmp
    · have hcompare : compare (2 * (num % den)) den = .lt :=
        Nat.compare_eq_lt.mpr hcmp
      simp [Accuracy.roundToNearestEven, hcompare]
    · split <;> rename_i hcmp'
      · have hcompare : compare (2 * (num % den)) den = .gt :=
          Nat.compare_eq_gt.mpr hcmp'
        simp [Accuracy.roundToNearestEven, hcompare]
      · have heq : 2 * (num % den) = den := by grind
        have hcompare : compare (2 * (num % den)) den = .eq :=
          Nat.compare_eq_eq.mpr heq
        simp only [Accuracy.roundToNearestEven, hcompare]
        by_cases heven : num / den % 2 = 0
        · simp [heven]
        · have hodd : num / den % 2 = 1 := by grind
          simp [hodd]

/-- An extended mantissa carrying `n` with no accumulated inexactness, used as the starting point
of a shift-and-round chain. -/
private abbrev exactExtendedMantissa (n : Nat) : ExtendedMantissa :=
  ExtendedMantissa.ofMantissaAndAccuracy n .exact

private theorem exactExtendedMantissa_shift_mantissa (n shift : Nat) :
    (exactExtendedMantissa n >>> shift).mantissa = n / 2 ^ shift := by
  induction shift with
  | zero =>
      simp [HShiftRight.hShiftRight, Nat.repeat, exactExtendedMantissa,
        ExtendedMantissa.ofMantissaAndAccuracy]
  | succ shift ih =>
      rw [show exactExtendedMantissa n >>> (shift + 1) =
          ExtendedMantissa.shiftRightOne (exactExtendedMantissa n >>> shift) by rfl]
      simp only [ExtendedMantissa.shiftRightOne]
      rw [ih, Nat.div_div_eq_div_mul]
      simp [pow_succ]

private theorem exactExtendedMantissa_shift_succ_roundBit (n shift : Nat) :
    (exactExtendedMantissa n >>> (shift + 1)).roundBit =
      (n / 2 ^ shift % 2 != 0) := by
  rw [show exactExtendedMantissa n >>> (shift + 1) =
      ExtendedMantissa.shiftRightOne (exactExtendedMantissa n >>> shift) by rfl]
  simp only [ExtendedMantissa.shiftRightOne]
  rw [exactExtendedMantissa_shift_mantissa]

private theorem exactExtendedMantissa_shift_succ_stickyBit (n shift : Nat) :
    (exactExtendedMantissa n >>> (shift + 1)).stickyBit =
      (n % 2 ^ shift != 0) := by
  induction shift with
  | zero =>
      simp [HShiftRight.hShiftRight, Nat.repeat, exactExtendedMantissa,
        ExtendedMantissa.ofMantissaAndAccuracy, ExtendedMantissa.shiftRightOne, Nat.mod_one]
  | succ shift ih =>
      rw [show exactExtendedMantissa n >>> (shift + 1 + 1) =
          ExtendedMantissa.shiftRightOne
            (exactExtendedMantissa n >>> (shift + 1)) by rfl]
      simp only [ExtendedMantissa.shiftRightOne]
      rw [exactExtendedMantissa_shift_succ_roundBit, ih]
      apply Bool.eq_iff_iff.mpr
      simp only [Bool.or_eq_true, bne_iff_ne]
      rw [Nat.mod_pow_succ]
      rcases Nat.mod_two_eq_zero_or_one (n / 2 ^ shift) with hbit | hbit <;>
        simp [hbit]

private theorem exactExtendedMantissa_shift_succ (n shift : Nat) :
    exactExtendedMantissa n >>> (shift + 1) =
      { mantissa := n / 2 ^ (shift + 1)
        roundBit := n / 2 ^ shift % 2 != 0
        stickyBit := n % 2 ^ shift != 0 } := by
  generalize hem : exactExtendedMantissa n >>> (shift + 1) = em
  cases em with
  | mk mantissa roundBit stickyBit =>
      have hm := exactExtendedMantissa_shift_mantissa n (shift + 1)
      have hr := exactExtendedMantissa_shift_succ_roundBit n shift
      have hs := exactExtendedMantissa_shift_succ_stickyBit n shift
      simp only [hem] at hm hr hs
      simp_all

private theorem exactExtendedMantissa_shift_accuracy (n shift : Nat) :
    (exactExtendedMantissa n >>> shift).accuracy =
      accuracyOfFraction (n % 2 ^ shift) (2 ^ shift) := by
  cases shift with
  | zero =>
      simp [HShiftRight.hShiftRight, Nat.repeat, exactExtendedMantissa,
        ExtendedMantissa.ofMantissaAndAccuracy, ExtendedMantissa.accuracy,
        accuracyOfFraction, Nat.mod_one]
  | succ shift =>
      rw [exactExtendedMantissa_shift_succ]
      have hp : 0 < 2 ^ shift := Nat.pow_pos (by decide)
      have hlow : n % 2 ^ shift < 2 ^ shift := Nat.mod_lt n hp
      have hmod := Nat.mod_pow_succ (x := n) (b := 2) (k := shift)
      rcases Nat.mod_two_eq_zero_or_one (n / 2 ^ shift) with hbit | hbit
      · by_cases hz : n % 2 ^ shift = 0
        · have htotal : n % 2 ^ (shift + 1) = 0 := by
            rw [hmod]
            simp [hbit, hz]
          simp [ExtendedMantissa.accuracy, accuracyOfFraction, hbit, hz, htotal]
        · have htotal : n % 2 ^ (shift + 1) = n % 2 ^ shift := by
            rw [hmod]
            simp [hbit]
          have hsticky : (n % 2 ^ shift != 0) = true := (bne_iff_ne).2 hz
          have hcompare : compare (2 * (n % 2 ^ shift)) (2 ^ (shift + 1)) = .lt := by
            apply Nat.compare_eq_lt.mpr
            rw [pow_succ]
            grind
          simp [ExtendedMantissa.accuracy, accuracyOfFraction, hbit, hz, hsticky, htotal,
            hcompare]
      · by_cases hz : n % 2 ^ shift = 0
        · have htotal : n % 2 ^ (shift + 1) = 2 ^ shift := by
            rw [hmod]
            simp [hbit, hz]
          have hcompare : compare (2 * 2 ^ shift) (2 ^ (shift + 1)) = .eq := by
            apply Nat.compare_eq_eq.mpr
            rw [pow_succ]
            grind
          simp [ExtendedMantissa.accuracy, accuracyOfFraction, hbit, hz, htotal,
            hcompare]
        · have htotal : n % 2 ^ (shift + 1) = n % 2 ^ shift + 2 ^ shift := by
            rw [hmod]
            simp [hbit]
          have hsticky : (n % 2 ^ shift != 0) = true := (bne_iff_ne).2 hz
          have hcompare :
              compare (2 * (n % 2 ^ shift + 2 ^ shift)) (2 ^ (shift + 1)) = .gt := by
            apply Nat.compare_eq_gt.mpr
            rw [pow_succ]
            grind
          simp [ExtendedMantissa.accuracy, accuracyOfFraction, hbit, hz, hsticky, htotal,
            hcompare]

/--
Shifting an exact mantissa and rounding the discarded bits to nearest-even agrees with
`IEEE32Exec.roundShiftRightEven` for every shift, including shifts larger than the mantissa.
-/
theorem roundedMantissa_shift_exact (n shift : Nat) :
    (ExtendedMantissa.ofMantissaAndAccuracy n .exact >>> shift).roundedMantissa =
      roundShiftRightEven n shift := by
  unfold ExtendedMantissa.roundedMantissa
  rw [exactExtendedMantissa_shift_accuracy, exactExtendedMantissa_shift_mantissa]
  have hden : 2 ^ shift ≠ 0 := Nat.ne_of_gt (Nat.pow_pos (by decide))
  rw [roundToNearestEven_accuracyOfFraction_mod n (2 ^ shift) hden]
  simpa [pow2, Nat.shiftLeft_eq] using
    (roundShiftRightEven_eq_roundQuotEven_pow2 n shift).symm

/-- A positive right shift rounds to zero when the input lies below the half-way point. -/
theorem roundShiftRightEven_eq_zero_of_lt_half
    (n shift : Nat) (hshift : 0 < shift) (hlt : n < pow2 (shift - 1)) :
    roundShiftRightEven n shift = 0 := by
  have hpowLe : pow2 (shift - 1) ≤ pow2 shift := by
    simp [pow2, Nat.shiftLeft_eq, Nat.pow_le_pow_right]
  have hnPow : n < 2 ^ shift := by
    simpa [pow2, Nat.shiftLeft_eq] using hlt.trans_le hpowLe
  have hq : Nat.shiftRight n shift = 0 := Nat.shiftRight_eq_zero n shift hnPow
  have hshiftNe : shift ≠ 0 := Nat.ne_of_gt hshift
  unfold roundShiftRightEven
  simp only [beq_iff_eq, hshiftNe, if_false]
  rw [hq]
  simp [pow2] at hlt ⊢
  grind

/--
Round an exact integer mantissa after expressing it at `targetExponent`.

If the source exponent is smaller, bits are discarded with nearest-even rounding. If it is larger,
the exact value is preserved by shifting the mantissa left.
-/
def roundMantissaAtExponentEven
    (mantissa : Nat) (exponent targetExponent : Int) : Nat :=
  if exponent ≤ targetExponent then
    roundShiftRightEven mantissa (targetExponent - exponent).toNat
  else
    mantissa <<< (exponent - targetExponent).toNat

/-- Round a positive integer mantissa so that its leading bit occupies `leadingBit`. -/
def roundMantissaToLeadingBitEven (mantissa leadingBit : Nat) : Nat :=
  if leadingBit ≤ mantissa.log2 then
    roundShiftRightEven mantissa (mantissa.log2 - leadingBit)
  else
    mantissa <<< (leadingBit - mantissa.log2)

/--
Rounding at the exponent that places the leading bit at `leadingBit` is the direct
shift-and-round formulation.
-/
theorem roundMantissaAtExponentEven_eq_roundMantissaToLeadingBitEven
    (mantissa leadingBit : Nat) (exponent : Int) :
    roundMantissaAtExponentEven mantissa exponent
        ((mantissa.log2 : Int) + exponent - (leadingBit : Int)) =
      roundMantissaToLeadingBitEven mantissa leadingBit := by
  by_cases hle : leadingBit ≤ mantissa.log2
  · have hexponentLe :
        exponent ≤ (mantissa.log2 : Int) + exponent - (leadingBit : Int) := by
      grind
    have hdistance :
        (((mantissa.log2 : Int) + exponent - (leadingBit : Int)) - exponent).toNat =
          mantissa.log2 - leadingBit := by
      rw [show (mantissa.log2 : Int) + exponent - (leadingBit : Int) - exponent =
          ((mantissa.log2 - leadingBit : Nat) : Int) by grind]
      rfl
    simp [roundMantissaAtExponentEven, roundMantissaToLeadingBitEven,
      hexponentLe, hle, hdistance]
  · have hnotExponentLe :
        ¬exponent ≤ (mantissa.log2 : Int) + exponent - (leadingBit : Int) := by
      grind
    have hdistance :
        (exponent - ((mantissa.log2 : Int) + exponent - (leadingBit : Int))).toNat =
          leadingBit - mantissa.log2 := by
      rw [show exponent - ((mantissa.log2 : Int) + exponent - (leadingBit : Int)) =
          ((leadingBit - mantissa.log2 : Nat) : Int) by grind]
      rfl
    simp [roundMantissaAtExponentEven, roundMantissaToLeadingBitEven,
      hnotExponentLe, hle, hdistance]

/--
Rounding at the binary32 subnormal exponent is the branch expression used by
`IEEE32Exec.roundDyadicToIEEE32`.

Writing this equality once keeps the later subnormal proof independent of the representation that
Lean chooses for positive and negative `Int` values.
-/
theorem roundMantissaAtExponentEven_neg149 (mantissa : Nat) (exponent : Int) :
    roundMantissaAtExponentEven mantissa exponent (-149) =
      match exponent + 149 with
      | .ofNat shift => Nat.shiftLeft mantissa shift
      | .negSucc shift => roundShiftRightEven mantissa (shift + 1) := by
  generalize hsum : exponent + 149 = displacement
  cases displacement with
  | ofNat shift =>
      cases shift with
      | zero =>
          norm_num at hsum
          have hexponent : exponent = -149 := by grind
          subst exponent
          simp [roundMantissaAtExponentEven, roundShiftRightEven]
      | succ shift =>
          simp only [Int.ofNat_eq_natCast, Nat.cast_add, Nat.cast_one] at hsum
          have hnotLe : ¬exponent ≤ -149 := by grind
          simp [roundMantissaAtExponentEven, hnotLe, hsum]
  | negSucc shift =>
      simp only [Int.negSucc_eq] at hsum
      have hle : exponent ≤ -149 := by grind
      have hdistance : ((-149 : Int) - exponent).toNat = shift + 1 := by
        have hdiff : (-149 : Int) - exponent = ((shift + 1 : Nat) : Int) := by grind
        rw [hdiff]
        rfl
      simp [roundMantissaAtExponentEven, hle, hdistance]

/--
Lean's decrease-then-shift normalization of an exact dyadic agrees with direct nearest-even
rounding at the requested exponent.
-/
theorem roundedMantissa_decrease_shift_exact
    (mantissa : Nat) (exponent targetExponent : Int) :
    let decreased :=
      Float.Model.UnpackedFloat.decreaseExponent mantissa exponent targetExponent
    let shifted := Float.Model.UnpackedFloat.shiftToExponent
      decreased.1 decreased.2 .exact targetExponent
    (shifted.1.roundedMantissa, shifted.2) =
      (roundMantissaAtExponentEven mantissa exponent targetExponent, targetExponent) := by
  by_cases hle : exponent ≤ targetExponent
  · have hleft : (exponent - targetExponent).toNat = 0 :=
      Int.toNat_eq_zero.mpr (by grind)
    have hdecreased :
        Float.Model.UnpackedFloat.decreaseExponent mantissa exponent targetExponent =
          (mantissa, exponent) := by
      unfold Float.Model.UnpackedFloat.decreaseExponent
      simp [hleft]
    have hrightInt : ((targetExponent - exponent).toNat : Int) =
        targetExponent - exponent := by
      rw [Int.toNat_of_nonneg (by grind)]
    have hshifted :
        Float.Model.UnpackedFloat.shiftToExponent
            mantissa exponent .exact targetExponent =
          (ExtendedMantissa.ofMantissaAndAccuracy mantissa .exact >>>
            (targetExponent - exponent).toNat, targetExponent) := by
      unfold Float.Model.UnpackedFloat.shiftToExponent
      apply Prod.ext
      · rfl
      · dsimp only [Prod.snd]
        rw [hrightInt]
        grind
    simp only [hdecreased, hshifted]
    apply Prod.ext
    · dsimp only [Prod.fst]
      rw [roundedMantissa_shift_exact]
      simp [roundMantissaAtExponentEven, hle]
    · rfl
  · have hleftInt : ((exponent - targetExponent).toNat : Int) =
        exponent - targetExponent := by
      rw [Int.toNat_of_nonneg (by grind)]
    have hdecreased :
        Float.Model.UnpackedFloat.decreaseExponent mantissa exponent targetExponent =
          (mantissa <<< (exponent - targetExponent).toNat, targetExponent) := by
      unfold Float.Model.UnpackedFloat.decreaseExponent
      apply Prod.ext
      · rfl
      · dsimp only [Prod.snd]
        rw [hleftInt]
        grind
    have hshifted :
        Float.Model.UnpackedFloat.shiftToExponent
            (mantissa <<< (exponent - targetExponent).toNat)
            targetExponent .exact targetExponent =
          (ExtendedMantissa.ofMantissaAndAccuracy
            (mantissa <<< (exponent - targetExponent).toNat) .exact, targetExponent) := by
      unfold Float.Model.UnpackedFloat.shiftToExponent
      simp [HShiftRight.hShiftRight, Nat.repeat,
        ExtendedMantissa.ofMantissaAndAccuracy]
    simp only [hdecreased, hshifted]
    apply Prod.ext
    · dsimp only [Prod.fst]
      simp [ExtendedMantissa.roundedMantissa, ExtendedMantissa.accuracy,
        ExtendedMantissa.ofMantissaAndAccuracy, Accuracy.roundToNearestEven,
        roundMantissaAtExponentEven, hle]
    · rfl

private theorem log2_shiftLeft (mantissa shift : Nat) (hm : mantissa ≠ 0) :
    (mantissa <<< shift).log2 = mantissa.log2 + shift := by
  induction shift with
  | zero => simp
  | succ shift ih =>
      simp only [Nat.shiftLeft_eq] at ih ⊢
      rw [pow_succ]
      rw [← Nat.mul_assoc, Nat.mul_comm (mantissa * 2 ^ shift) 2]
      rw [Nat.log2_two_mul]
      · rw [ih]
        grind
      · exact Nat.mul_ne_zero hm (pow_ne_zero _ (by decide))

/-- Decreasing an exact dyadic exponent preserves the position of its leading binary digit. -/
theorem totalExponent_decreaseExponent
    (mantissa : Nat) (exponent targetExponent : Int) (hm : mantissa ≠ 0) :
    let decreased := Float.Model.UnpackedFloat.decreaseExponent
      mantissa exponent targetExponent
    Float.Model.totalExponent decreased.1 decreased.2 =
      Float.Model.totalExponent mantissa exponent := by
  by_cases hle : exponent ≤ targetExponent
  · have hshift : (exponent - targetExponent).toNat = 0 :=
      Int.toNat_eq_zero.mpr (by grind)
    unfold Float.Model.UnpackedFloat.decreaseExponent
    simp [hshift]
  · have hshiftInt : ((exponent - targetExponent).toNat : Int) =
        exponent - targetExponent := by
      rw [Int.toNat_of_nonneg (by grind)]
    unfold Float.Model.UnpackedFloat.decreaseExponent Float.Model.totalExponent
    dsimp only [Prod.fst, Prod.snd]
    rw [log2_shiftLeft mantissa (exponent - targetExponent).toNat hm]
    rw [Nat.cast_add, hshiftInt]
    grind

/--
The first exact rounding stage uses the target exponent computed from the original dyadic value.

This packages the invariant behind `UnpackedFloat.round`: its preliminary left shift changes the
mantissa representation but not the leading binary exponent.
-/
theorem roundedMantissa_decrease_shiftToTarget_exact
    (spec : Float.Model.Format) (mantissa : Nat) (exponent : Int) (hm : mantissa ≠ 0) :
    let targetExponent := spec.targetExponent (Float.Model.totalExponent mantissa exponent)
    let decreased :=
      Float.Model.UnpackedFloat.decreaseExponent mantissa exponent targetExponent
    let shifted := Float.Model.UnpackedFloat.shiftToTargetExponent
      spec decreased.1 decreased.2 .exact
    (shifted.1.roundedMantissa, shifted.2) =
      (roundMantissaAtExponentEven mantissa exponent targetExponent, targetExponent) := by
  let targetExponent := spec.targetExponent (Float.Model.totalExponent mantissa exponent)
  let decreased :=
    Float.Model.UnpackedFloat.decreaseExponent mantissa exponent targetExponent
  have htotal :
      Float.Model.totalExponent decreased.1 decreased.2 =
        Float.Model.totalExponent mantissa exponent := by
    exact totalExponent_decreaseExponent mantissa exponent targetExponent hm
  have hshift :
      Float.Model.UnpackedFloat.shiftToTargetExponent
          spec decreased.1 decreased.2 .exact =
        Float.Model.UnpackedFloat.shiftToExponent
          decreased.1 decreased.2 .exact targetExponent := by
    unfold Float.Model.UnpackedFloat.shiftToTargetExponent
    rw [htotal]
  change
    ((Float.Model.UnpackedFloat.shiftToTargetExponent
        spec decreased.1 decreased.2 .exact).1.roundedMantissa,
      (Float.Model.UnpackedFloat.shiftToTargetExponent
        spec decreased.1 decreased.2 .exact).2) =
      (roundMantissaAtExponentEven mantissa exponent targetExponent, targetExponent)
  rw [hshift]
  exact roundedMantissa_decrease_shift_exact mantissa exponent targetExponent

/-- Complete rounding after the first rounded mantissa and exponent have been computed. -/
def finishRoundedMantissa
    (spec : Float.Model.Format) (sign : Sign) (rounded : Nat × Int) :
    Float.Model.UnpackedFloat :=
  let final := Float.Model.UnpackedFloat.shiftToTargetExponent
    spec rounded.1 rounded.2 .exact
  if h : final.1.mantissa = 0 then
    .zero sign
  else
    .finite sign final.1.mantissa final.2 (Nat.pos_of_ne_zero h)

/-- The final stage of rounding is `finishRoundedMantissa`. -/
theorem roundWithAccuracy_eq_finishRoundedMantissa
    (spec : Float.Model.Format) (sign : Sign) (mantissa : Nat) (exponent : Int)
    (accuracy : Accuracy) :
    Float.Model.UnpackedFloat.roundWithAccuracy spec sign mantissa exponent accuracy =
      finishRoundedMantissa spec sign
        ((Float.Model.UnpackedFloat.shiftToTargetExponent
            spec mantissa exponent accuracy).1.roundedMantissa,
          (Float.Model.UnpackedFloat.shiftToTargetExponent
            spec mantissa exponent accuracy).2) := by
  rfl

/--
If an exact mantissa is already represented at or below its target exponent, the lightweight
`roundWithAccuracy` path agrees with the unrestricted rounder.

This is the side condition promised by `UnpackedFloat.roundWithAccuracy`: no preliminary left
shift is needed to expose enough mantissa bits. Multiplication, division, and square root establish
this condition from their exact intermediate before using the lemma.
-/
theorem roundWithAccuracy_exact_eq_round_of_le_targetExponent
    (spec : Float.Model.Format) (sign : Sign) (mantissa : Nat) (exponent : Int)
    (h : exponent ≤ spec.targetExponent (Float.Model.totalExponent mantissa exponent)) :
    Float.Model.UnpackedFloat.roundWithAccuracy spec sign mantissa exponent .exact =
      Float.Model.UnpackedFloat.round spec sign mantissa exponent := by
  unfold Float.Model.UnpackedFloat.round Float.Model.UnpackedFloat.decreaseExponent
  have hshift : (exponent - spec.targetExponent
      (Float.Model.totalExponent mantissa exponent)).toNat = 0 :=
    Int.toNat_eq_zero.mpr (by grind)
  simp [hshift]

/-- Exact dyadic rounding factors through the direct target-exponent rounder. -/
theorem round_exact_eq_finishRoundedMantissa
    (spec : Float.Model.Format) (sign : Sign) (mantissa : Nat) (exponent : Int)
    (hm : mantissa ≠ 0) :
    Float.Model.UnpackedFloat.round spec sign mantissa exponent =
      finishRoundedMantissa spec sign
        (roundMantissaAtExponentEven mantissa exponent
          (spec.targetExponent (Float.Model.totalExponent mantissa exponent)),
        spec.targetExponent (Float.Model.totalExponent mantissa exponent)) := by
  let targetExponent := spec.targetExponent (Float.Model.totalExponent mantissa exponent)
  let decreased :=
    Float.Model.UnpackedFloat.decreaseExponent mantissa exponent targetExponent
  have hfirst :=
    roundedMantissa_decrease_shiftToTarget_exact spec mantissa exponent hm
  change finishRoundedMantissa spec sign
      ((Float.Model.UnpackedFloat.shiftToTargetExponent
          spec decreased.1 decreased.2 .exact).1.roundedMantissa,
        (Float.Model.UnpackedFloat.shiftToTargetExponent
          spec decreased.1 decreased.2 .exact).2) =
    finishRoundedMantissa spec sign
      (roundMantissaAtExponentEven mantissa exponent targetExponent, targetExponent)
  exact congrArg (finishRoundedMantissa spec sign) hfirst

private theorem exactExtendedMantissa_zero_shift (shift : Nat) :
    (exactExtendedMantissa 0 >>> shift) = exactExtendedMantissa 0 := by
  induction shift with
  | zero => rfl
  | succ shift ih =>
      change Nat.repeat ExtendedMantissa.shiftRightOne shift (exactExtendedMantissa 0) = _ at ih
      change Nat.repeat ExtendedMantissa.shiftRightOne (Nat.succ shift)
          (exactExtendedMantissa 0) = _
      rw [Nat.repeat, ih]
      rfl

private theorem exactExtendedMantissa_zero_bits_shift (shift : Nat) :
    (({ mantissa := 0, roundBit := false, stickyBit := false } : ExtendedMantissa) >>> shift) =
      { mantissa := 0, roundBit := false, stickyBit := false } := by
  exact exactExtendedMantissa_zero_shift shift

private theorem roundWithAccuracy_exact_zero (sign : Sign) (exponent : Int) :
    Float.Model.UnpackedFloat.roundWithAccuracy
        Float.Model.Format.binary32 sign 0 exponent .exact =
      Float.Model.UnpackedFloat.zero sign := by
  unfold Float.Model.UnpackedFloat.roundWithAccuracy
    Float.Model.UnpackedFloat.shiftToTargetExponent
    Float.Model.UnpackedFloat.shiftToExponent
  simp (config := { zeta := true }) only [exactExtendedMantissa_zero_bits_shift,
    ExtendedMantissa.roundedMantissa, ExtendedMantissa.accuracy,
    ExtendedMantissa.ofMantissaAndAccuracy, Accuracy.roundToNearestEven]
  split
  · rfl
  · contradiction

/-- The finishing stage preserves a zero mantissa as signed zero. -/
theorem finishRoundedMantissa_binary32_zero (sign : Sign) (exponent : Int) :
    finishRoundedMantissa Float.Model.Format.binary32 sign (0, exponent) = .zero sign := by
  unfold finishRoundedMantissa Float.Model.UnpackedFloat.shiftToTargetExponent
    Float.Model.UnpackedFloat.shiftToExponent
  simp only [ExtendedMantissa.ofMantissaAndAccuracy,
    exactExtendedMantissa_zero_bits_shift]
  simp

/-- Rounding an exact zero preserves its sign, independently of the supplied exponent. -/
theorem round_exact_zero (sign : Sign) (exponent : Int) :
    Float.Model.UnpackedFloat.round Float.Model.Format.binary32 sign 0 exponent =
      Float.Model.UnpackedFloat.zero sign := by
  unfold Float.Model.UnpackedFloat.round Float.Model.UnpackedFloat.decreaseExponent
  simp only [Nat.zero_shiftLeft]
  unfold Float.Model.UnpackedFloat.roundWithAccuracy
    Float.Model.UnpackedFloat.shiftToTargetExponent Float.Model.UnpackedFloat.shiftToExponent
  simp (config := { zeta := true }) only [exactExtendedMantissa_zero_bits_shift,
    ExtendedMantissa.roundedMantissa, ExtendedMantissa.accuracy,
    ExtendedMantissa.ofMantissaAndAccuracy, Accuracy.roundToNearestEven]
  split
  · rfl
  · contradiction

/-- Lean's exact dyadic rounder and `IEEE32Exec` agree on both signed zeros. -/
theorem model_round_exact_zero_eq_roundDyadic (sign : Sign) (exponent : Int) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (Float.Model.UnpackedFloat.round Float.Model.Format.binary32 sign 0 exponent)) =
      roundDyadicToIEEE32 { sign := signToBool sign, mant := 0, exp := exponent } := by
  rw [round_exact_zero]
  change IEEE32Exec.ofBits
      (UInt32.ofBitVec (Float.Model.UnpackedFloat.packedZero
        Float.Model.Format.binary32 sign)) = _
  rw [show UInt32.ofBitVec (Float.Model.UnpackedFloat.packedZero
      Float.Model.Format.binary32 sign) =
      IEEE32Exec.mkBits (signToBool sign) 0 0 by
    simpa [Float.Model.UnpackedFloat.packedZero] using
      packComponents_eq_mkBits sign (0#8) (0#23)]
  cases sign <;>
    simp [roundDyadicToIEEE32, signToBool, IEEE32Exec.mkBits,
      IEEE32Exec.negZero, IEEE32Exec.posZero, IEEE32Exec.signMask]
  congr 1

private theorem mantissa_lt_pow_log2_succ (mantissa : Nat) :
    mantissa < 2 ^ (mantissa.log2 + 1) := by
  have hlog : Nat.log2 mantissa = Nat.log 2 mantissa :=
    Nat.log2_eq_log_two (n := mantissa)
  simpa [hlog, Nat.succ_eq_add_one] using
    Nat.lt_pow_succ_log_self (b := 2) (hb := Nat.one_lt_two) mantissa

/-- A nonzero mantissa rounded to leading position `p` is at least `2^p`. -/
theorem pow2_le_roundMantissaToLeadingBitEven
    (mantissa leadingBit : Nat) (hm : mantissa ≠ 0) :
    pow2 leadingBit ≤ roundMantissaToLeadingBitEven mantissa leadingBit := by
  unfold roundMantissaToLeadingBitEven
  split <;> rename_i hle
  · have hdiv :
        2 ^ leadingBit ≤ mantissa / 2 ^ (mantissa.log2 - leadingBit) := by
      rw [Nat.le_div_iff_mul_le (Nat.pow_pos (by decide))]
      rw [show 2 ^ leadingBit * 2 ^ (mantissa.log2 - leadingBit) =
          2 ^ mantissa.log2 by
        rw [Nat.mul_comm, Nat.pow_sub_mul_pow 2 hle]]
      exact Nat.log2_self_le hm
    rw [← Nat.shiftRight_eq_div_pow] at hdiv
    simpa [pow2, Nat.shiftLeft_eq] using
      hdiv.trans (shiftRight_le_roundShiftRightEven mantissa
        (mantissa.log2 - leadingBit))
  · have hlogLe : 2 ^ mantissa.log2 ≤ mantissa := Nat.log2_self_le hm
    have hmul := Nat.mul_le_mul_right (2 ^ (leadingBit - mantissa.log2)) hlogLe
    rw [show 2 ^ mantissa.log2 * 2 ^ (leadingBit - mantissa.log2) =
        2 ^ leadingBit by
      rw [Nat.mul_comm, Nat.pow_sub_mul_pow 2 (Nat.le_of_not_ge hle)]] at hmul
    simpa [pow2, Nat.shiftLeft_eq] using hmul

/-- A mantissa rounded to leading position `p` is at most the one-bit carry `2^(p+1)`. -/
theorem roundMantissaToLeadingBitEven_le_pow2_succ
    (mantissa leadingBit : Nat) :
    roundMantissaToLeadingBitEven mantissa leadingBit ≤ pow2 (leadingBit + 1) := by
  unfold roundMantissaToLeadingBitEven
  split <;> rename_i hle
  · have hdiv :
        mantissa / 2 ^ (mantissa.log2 - leadingBit) < 2 ^ (leadingBit + 1) := by
      have hexponent :
          leadingBit + 1 + (mantissa.log2 - leadingBit) = mantissa.log2 + 1 := by
        grind
      rw [Nat.div_lt_iff_lt_mul (Nat.pow_pos (by decide))]
      rw [show 2 ^ (leadingBit + 1) * 2 ^ (mantissa.log2 - leadingBit) =
          2 ^ (mantissa.log2 + 1) by
        rw [← Nat.pow_add, hexponent]]
      exact mantissa_lt_pow_log2_succ mantissa
    rw [← Nat.shiftRight_eq_div_pow] at hdiv
    have hround := roundShiftRightEven_le_shiftRight_add1 mantissa
      (mantissa.log2 - leadingBit)
    simpa [pow2, Nat.shiftLeft_eq] using hround.trans (Nat.succ_le_iff.mpr hdiv)
  · have hlt : mantissa < 2 ^ (mantissa.log2 + 1) :=
      mantissa_lt_pow_log2_succ mantissa
    have hpowPos : 0 < 2 ^ (leadingBit - mantissa.log2) := Nat.pow_pos (by decide)
    have hmul := Nat.mul_lt_mul_of_pos_right hlt hpowPos
    have hlogLe : mantissa.log2 ≤ leadingBit := Nat.le_of_lt (Nat.lt_of_not_ge hle)
    have hexponent :
        mantissa.log2 + 1 + (leadingBit - mantissa.log2) = leadingBit + 1 := by
      grind
    rw [show 2 ^ (mantissa.log2 + 1) * 2 ^ (leadingBit - mantissa.log2) =
        2 ^ (leadingBit + 1) by
      rw [← Nat.pow_add, hexponent]] at hmul
    simpa [pow2, Nat.shiftLeft_eq] using hmul.le

/--
A dyadic below the normal exponent range rounds to at most `2^23` at exponent `-149`.

Equality is possible: a value immediately below the normal range may round up to the smallest
normal binary32 value. This is why the subnormal packing proof has a separate carry case rather
than assuming that the rounded fraction always fits in 23 bits.
-/
theorem roundMantissaAtExponentEven_neg149_le_pow2_23
    (mantissa : Nat) (exponent : Int)
    (hk : (mantissa.log2 : Int) + exponent < -126) :
    roundMantissaAtExponentEven mantissa exponent (-149) ≤ pow2 23 := by
  rw [roundMantissaAtExponentEven_neg149]
  generalize hsum : exponent + 149 = displacement
  cases displacement with
  | ofNat shift =>
      have hlog : mantissa.log2 + shift < 23 := by
        simp only [Int.ofNat_eq_natCast] at hsum
        grind
      have hraw := Nat.shiftLeft_lt (m := shift) (mantissa_lt_pow_log2_succ mantissa)
      have hexponent : mantissa.log2 + 1 + shift ≤ 23 := by grind
      have hpow : 2 ^ (mantissa.log2 + 1 + shift) ≤ 2 ^ 23 :=
        Nat.pow_le_pow_right (by decide) hexponent
      simpa [pow2, Nat.shiftLeft_eq] using (hraw.trans_le hpow).le
  | negSucc shift =>
      simp only [Int.negSucc_eq] at hsum
      have hlog : mantissa.log2 + 1 ≤ 24 + shift := by grind
      have hmLt : mantissa < 2 ^ (24 + shift) :=
        (mantissa_lt_pow_log2_succ mantissa).trans_le
          (Nat.pow_le_pow_right (by decide) hlog)
      have hpowEq :
          2 ^ (24 + shift) = pow2 23 * 2 ^ (shift + 1) := by
        simp [pow2, Nat.pow_add]
        ring
      have hq : mantissa >>> (shift + 1) < pow2 23 := by
        rw [Nat.shiftRight_eq_div_pow]
        rw [Nat.div_lt_iff_lt_mul (Nat.pow_pos (by decide))]
        rw [← hpowEq]
        exact hmLt
      have hround := roundShiftRightEven_le_shiftRight_add1 mantissa (shift + 1)
      have hshift :
          mantissa >>> (shift + 1) = mantissa / 2 ^ (shift + 1) :=
        Nat.shiftRight_eq_div_pow mantissa (shift + 1)
      change roundShiftRightEven mantissa (shift + 1) ≤ pow2 23
      change roundShiftRightEven mantissa (shift + 1) ≤
        (mantissa >>> (shift + 1)) + 1 at hround
      rw [hshift] at hround
      grind

/--
An exact dyadic whose leading exponent is below `-150` rounds to signed zero in Lean's
binary32 model.

The strict inequality is the binary32 underflow region below half the least positive subnormal.
At the boundary itself, nearest-even also chooses zero, but that tie is handled by the subnormal
regime so that this lemma matches the branch structure of `IEEE32Exec.roundDyadicToIEEE32`.
-/
theorem round_exact_underflow
    (sign : Sign) (mantissa : Nat) (exponent : Int)
    (hk : (mantissa.log2 : Int) + exponent < -150) :
    Float.Model.UnpackedFloat.round
        Float.Model.Format.binary32 sign mantissa exponent =
      Float.Model.UnpackedFloat.zero sign := by
  have htarget :
      Float.Model.Format.binary32.targetExponent
          (Float.Model.totalExponent mantissa exponent) = -149 := by
    rw [Float.Model.Format.targetExponent]
    apply max_eq_right
    simp only [Float.Model.totalExponent,
      Float.Model.Format.mantissaBits, Float.Model.Format.minExponent]
    norm_num
    grind
  have hexp : exponent ≤ -149 := by
    have hlogNonneg : (0 : Int) ≤ (mantissa.log2 : Int) := Int.natCast_nonneg _
    grind
  have hdecrease :
      Float.Model.UnpackedFloat.decreaseExponent mantissa exponent (-149) =
        (mantissa, exponent) := by
    unfold Float.Model.UnpackedFloat.decreaseExponent
    simp only [sub_neg_eq_add]
    have hz : (exponent + 149).toNat = 0 := Int.toNat_eq_zero.mpr (by grind)
    simp [hz]
  have hround :
      Float.Model.UnpackedFloat.round
          Float.Model.Format.binary32 sign mantissa exponent =
        Float.Model.UnpackedFloat.roundWithAccuracy
          Float.Model.Format.binary32 sign mantissa exponent .exact := by
    unfold Float.Model.UnpackedFloat.round
    simp only [htarget]
    rw [hdecrease]
  rw [hround]
  let shift : Nat := ((-149 : Int) - exponent).toNat
  have hshiftInt : (shift : Int) = -149 - exponent := by
    simp only [shift]
    rw [Int.toNat_of_nonneg (by grind)]
  have hshiftPos : 0 < shift := by grind
  have hlogLe : mantissa.log2 + 1 ≤ shift - 1 := by grind
  have hmLt : mantissa < 2 ^ (shift - 1) :=
    (mantissa_lt_pow_log2_succ mantissa).trans_le
      (Nat.pow_le_pow_right (by decide) hlogLe)
  have hrounded : roundShiftRightEven mantissa shift = 0 :=
    roundShiftRightEven_eq_zero_of_lt_half mantissa shift hshiftPos <| by
      simpa [pow2, Nat.shiftLeft_eq] using hmLt
  have hfirst :
      Float.Model.UnpackedFloat.shiftToTargetExponent
          Float.Model.Format.binary32 mantissa exponent .exact =
        (ExtendedMantissa.ofMantissaAndAccuracy mantissa .exact >>> shift, -149) := by
    unfold Float.Model.UnpackedFloat.shiftToTargetExponent
      Float.Model.UnpackedFloat.shiftToExponent
    rw [htarget]
    apply Prod.ext
    · rfl
    · dsimp only [Prod.snd]
      rw [hshiftInt]
      grind
  have hem :
      (ExtendedMantissa.ofMantissaAndAccuracy mantissa .exact >>> shift).roundedMantissa = 0 := by
    rw [roundedMantissa_shift_exact, hrounded]
  unfold Float.Model.UnpackedFloat.roundWithAccuracy
  rw [hfirst]
  dsimp only [Prod.fst, Prod.snd]
  rw [hem]
  exact roundWithAccuracy_exact_zero sign (-149)

/-- Lean's exact rounder and `IEEE32Exec` agree throughout the strict underflow region. -/
theorem model_round_exact_underflow_eq_roundDyadic
    (sign : Sign) (mantissa : Nat) (exponent : Int)
    (hk : (mantissa.log2 : Int) + exponent < -150) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (Float.Model.UnpackedFloat.round
            Float.Model.Format.binary32 sign mantissa exponent)) =
      roundDyadicToIEEE32
        { sign := signToBool sign, mant := mantissa, exp := exponent } := by
  by_cases hm : mantissa = 0
  · subst mantissa
    exact model_round_exact_zero_eq_roundDyadic sign exponent
  have hover : ¬127 < (mantissa.log2 : Int) + exponent := by grind
  rw [round_exact_underflow sign mantissa exponent hk]
  change IEEE32Exec.ofBits
      (UInt32.ofBitVec
        (Float.Model.UnpackedFloat.packedZero Float.Model.Format.binary32 sign)) = _
  rw [show UInt32.ofBitVec
      (Float.Model.UnpackedFloat.packedZero Float.Model.Format.binary32 sign) =
      IEEE32Exec.mkBits (signToBool sign) 0 0 by
    simpa [Float.Model.UnpackedFloat.packedZero] using
      packComponents_eq_mkBits sign (0#8) (0#23)]
  cases sign <;>
    simp [roundDyadicToIEEE32, signToBool, hm, hover, hk, IEEE32Exec.mkBits,
      IEEE32Exec.negZero, IEEE32Exec.posZero, IEEE32Exec.signMask]
  all_goals congr 1

private theorem targetExponent_subnormal_mantissa
    (mantissa : Nat) (hm : mantissa ≠ 0) (hlt : mantissa < pow2 23) :
    Float.Model.Format.binary32.targetExponent
        (Float.Model.totalExponent mantissa (-149)) = -149 := by
  have hlog : mantissa.log2 < 23 :=
    (Nat.log2_lt hm).2 (by simpa [pow2, Nat.shiftLeft_eq] using hlt)
  unfold Float.Model.Format.targetExponent Float.Model.totalExponent
  simp only [Float.Model.Format.mantissaBits, Float.Model.Format.minExponent]
  norm_num
  grind

private theorem shiftToTargetExponent_subnormal_mantissa
    (mantissa : Nat) (hm : mantissa ≠ 0) (hlt : mantissa < pow2 23) :
    Float.Model.UnpackedFloat.shiftToTargetExponent
        Float.Model.Format.binary32 mantissa (-149) .exact =
      (ExtendedMantissa.ofMantissaAndAccuracy mantissa .exact, -149) := by
  unfold Float.Model.UnpackedFloat.shiftToTargetExponent
    Float.Model.UnpackedFloat.shiftToExponent
  rw [targetExponent_subnormal_mantissa mantissa hm hlt]
  simp [HShiftRight.hShiftRight, Nat.repeat, ExtendedMantissa.ofMantissaAndAccuracy]

/--
The second rounding stage packs every positive 23-bit mantissa at exponent `-149` as a subnormal.

The sign is retained and the natural mantissa becomes the raw fraction field. The hypotheses are
exactly the nonzero and field-width conditions needed by binary32 packing.
-/
theorem model_finishRoundedMantissa_subnormal
    (sign : Sign) (mantissa : Nat) (hm : mantissa ≠ 0) (hlt : mantissa < pow2 23) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (finishRoundedMantissa Float.Model.Format.binary32 sign (mantissa, -149))) =
      IEEE32Exec.ofBits (IEEE32Exec.mkBits (signToBool sign) 0 mantissa) := by
  have hfinish :
      finishRoundedMantissa Float.Model.Format.binary32 sign (mantissa, -149) =
        .finite sign mantissa (-149) (Nat.pos_of_ne_zero hm) := by
    unfold finishRoundedMantissa
    simp only [shiftToTargetExponent_subnormal_mantissa mantissa hm hlt,
      ExtendedMantissa.ofMantissaAndAccuracy, hm, ↓reduceDIte]
  rw [hfinish]
  unfold modelToIEEE32Exec Float32.Model.pack
  change IEEE32Exec.ofBits
    (UInt32.ofBitVec
      (Float.Model.UnpackedFloat.pack Float.Model.Format.binary32
        (.finite sign mantissa (-149) (Nat.pos_of_ne_zero hm)))) = _
  unfold Float.Model.UnpackedFloat.pack
  have hlog : mantissa.log2 < 23 :=
    (Nat.log2_lt hm).2 (by simpa [pow2, Nat.shiftLeft_eq] using hlt)
  simp only [Float.Model.Format.exponentBias, Float.Model.Format.mantissaBits]
  norm_num
  simp only [show ¬mantissa.log2 + 1 = 24 by grind, if_false]
  rw [packComponents_eq_mkBits]
  simp only [BitVec.toNat_ofNat, Nat.zero_mod]
  rw [Nat.mod_eq_of_lt]
  simpa [pow2, Nat.shiftLeft_eq] using hlt

private theorem shiftToTargetExponent_smallestNormal :
    Float.Model.UnpackedFloat.shiftToTargetExponent Float.Model.Format.binary32
        (pow2 23) (-149) .exact =
      (ExtendedMantissa.ofMantissaAndAccuracy (pow2 23) .exact, -149) := by
  have htarget :
      Float.Model.Format.binary32.targetExponent
          (Float.Model.totalExponent (pow2 23) (-149)) = -149 := by
    unfold Float.Model.Format.targetExponent Float.Model.totalExponent
    simp only [Float.Model.Format.mantissaBits, Float.Model.Format.minExponent]
    norm_num [pow2, Nat.shiftLeft_eq]
    decide
  unfold Float.Model.UnpackedFloat.shiftToTargetExponent
    Float.Model.UnpackedFloat.shiftToExponent
  rw [htarget]
  simp [HShiftRight.hShiftRight, Nat.repeat, ExtendedMantissa.ofMantissaAndAccuracy]

/-- A subnormal carry of exactly `2^23` packs as the smallest normal binary32 value. -/
theorem model_finishRoundedMantissa_smallestNormal (sign : Sign) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (finishRoundedMantissa Float.Model.Format.binary32 sign (pow2 23, -149))) =
      IEEE32Exec.ofBits (IEEE32Exec.mkBits (signToBool sign) 1 0) := by
  have hm : pow2 23 ≠ 0 := by decide
  have hfinish :
      finishRoundedMantissa Float.Model.Format.binary32 sign (pow2 23, -149) =
        .finite sign (pow2 23) (-149) (Nat.pos_of_ne_zero hm) := by
    unfold finishRoundedMantissa
    simp only [shiftToTargetExponent_smallestNormal,
      ExtendedMantissa.ofMantissaAndAccuracy, hm, ↓reduceDIte]
  rw [hfinish]
  unfold modelToIEEE32Exec Float32.Model.pack Float.Model.UnpackedFloat.pack
  simp only [Float.Model.Format.exponentBias, Float.Model.Format.mantissaBits]
  norm_num [pow2, Nat.shiftLeft_eq]
  simp only [show Nat.log2 8388608 + 1 = 24 by decide, if_true]
  rw [packComponents_eq_mkBits]
  cases sign <;>
    norm_num [signToBool, IEEE32Exec.mkBits, IEEE32Exec.expAllOnes,
      IEEE32Exec.fracMask, UInt32.ofNat]

/--
Lean's exact binary32 rounder and `IEEE32Exec` agree when the exact dyadic value lies in the
subnormal exponent range.

The result covers all three possible rounded outcomes in this range: signed zero, a genuine
subnormal, and a carry into the smallest normal value.
-/
theorem model_round_exact_subnormal_eq_roundDyadic
    (sign : Sign) (mantissa : Nat) (exponent : Int) (hm : mantissa ≠ 0)
    (hlow : -150 ≤ (mantissa.log2 : Int) + exponent)
    (hhigh : (mantissa.log2 : Int) + exponent < -126) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (Float.Model.UnpackedFloat.round
            Float.Model.Format.binary32 sign mantissa exponent)) =
      roundDyadicToIEEE32
        { sign := signToBool sign, mant := mantissa, exp := exponent } := by
  let frac :=
    match exponent + 149 with
    | .ofNat shift => Nat.shiftLeft mantissa shift
    | .negSucc shift => roundShiftRightEven mantissa (shift + 1)
  have hfracRound : frac = roundMantissaAtExponentEven mantissa exponent (-149) := by
    exact (roundMantissaAtExponentEven_neg149 mantissa exponent).symm
  have htarget :
      Float.Model.Format.binary32.targetExponent
          (Float.Model.totalExponent mantissa exponent) = -149 := by
    unfold Float.Model.Format.targetExponent Float.Model.totalExponent
    simp only [Float.Model.Format.mantissaBits, Float.Model.Format.minExponent]
    norm_num
    grind
  have hround :
      Float.Model.UnpackedFloat.round
          Float.Model.Format.binary32 sign mantissa exponent =
        finishRoundedMantissa Float.Model.Format.binary32 sign (frac, -149) := by
    rw [round_exact_eq_finishRoundedMantissa
      Float.Model.Format.binary32 sign mantissa exponent hm]
    simp only [htarget]
    rw [← hfracRound]
  have hfracLe : frac ≤ pow2 23 := by
    rw [hfracRound]
    exact roundMantissaAtExponentEven_neg149_le_pow2_23 mantissa exponent hhigh
  have hover : ¬Int.ofNat mantissa.log2 + exponent > 127 := by
    simpa only [Int.ofNat_eq_natCast] using
      (show ¬127 < (mantissa.log2 : Int) + exponent by grind)
  have hnotUnder : ¬Int.ofNat mantissa.log2 + exponent < -150 := by
    simpa only [Int.ofNat_eq_natCast] using
      (show ¬(mantissa.log2 : Int) + exponent < -150 by grind)
  have hhigh' : Int.ofNat mantissa.log2 + exponent < -126 := by
    simpa only [Int.ofNat_eq_natCast] using hhigh
  have hp : pow2 23 ≠ 0 := by decide
  have hexec :
      roundDyadicToIEEE32
          { sign := signToBool sign, mant := mantissa, exp := exponent } =
        if frac == 0 then
          if signToBool sign then IEEE32Exec.negZero else IEEE32Exec.posZero
        else
          match Nat.decLe (pow2 23) frac with
          | isTrue _ => IEEE32Exec.ofBits (IEEE32Exec.mkBits (signToBool sign) 1 0)
          | isFalse _ => IEEE32Exec.ofBits (IEEE32Exec.mkBits (signToBool sign) 0 frac) := by
    unfold roundDyadicToIEEE32
    simp only [beq_iff_eq, hm, if_false]
    rw [if_neg hover, if_neg hnotUnder, if_pos hhigh']
    change (if frac = 0 then
        if signToBool sign then IEEE32Exec.negZero else IEEE32Exec.posZero
      else
        match Nat.decLe (pow2 23) frac with
        | isTrue _ => IEEE32Exec.ofBits (IEEE32Exec.mkBits (signToBool sign) 1 0)
        | isFalse _ => IEEE32Exec.ofBits (IEEE32Exec.mkBits (signToBool sign) 0 frac)) = _
    rfl
  rw [hround, hexec]
  by_cases hzero : frac = 0
  · rw [hzero, finishRoundedMantissa_binary32_zero]
    change IEEE32Exec.ofBits
        (UInt32.ofBitVec
          (Float.Model.UnpackedFloat.packedZero Float.Model.Format.binary32 sign)) = _
    rw [show UInt32.ofBitVec
        (Float.Model.UnpackedFloat.packedZero Float.Model.Format.binary32 sign) =
          IEEE32Exec.mkBits (signToBool sign) 0 0 by
      simpa [Float.Model.UnpackedFloat.packedZero] using
        packComponents_eq_mkBits sign (0#8) (0#23)]
    cases sign <;>
      simp [signToBool, IEEE32Exec.mkBits, IEEE32Exec.negZero,
        IEEE32Exec.posZero, IEEE32Exec.signMask]
    all_goals congr 1
  · by_cases hcarry : frac = pow2 23
    · rw [hcarry, model_finishRoundedMantissa_smallestNormal]
      rw [if_neg (by simpa [hcarry] using hp)]
      cases hdec : Nat.decLe (pow2 23) (pow2 23) with
      | isTrue _ => rfl
      | isFalse hnot => exact (hnot (by simp)).elim
    · have hfracLt : frac < pow2 23 := lt_of_le_of_ne hfracLe hcarry
      rw [model_finishRoundedMantissa_subnormal sign frac hzero hfracLt]
      rw [if_neg (by simpa using hzero)]
      cases hdec : Nat.decLe (pow2 23) frac with
      | isTrue hle => exact (Nat.not_le.mpr hfracLt hle).elim
      | isFalse _ => rfl

private theorem log2_eq_23_of_normalized_mantissa
    (mantissa : Nat) (hlow : pow2 23 ≤ mantissa) (hhigh : mantissa < pow2 24) :
    mantissa.log2 = 23 := by
  have hm : mantissa ≠ 0 :=
    Nat.ne_of_gt ((pow2_pos 23).trans_le hlow)
  apply (Nat.log2_eq_iff hm).2
  simpa [pow2, Nat.shiftLeft_eq] using And.intro hlow hhigh

private theorem shiftToTargetExponent_normal_mantissa
    (mantissa : Nat) (k : Int)
    (hlow : pow2 23 ≤ mantissa) (hhigh : mantissa < pow2 24)
    (hk : -126 ≤ k) (accuracy : Accuracy) :
    Float.Model.UnpackedFloat.shiftToTargetExponent
        Float.Model.Format.binary32 mantissa (k - 23) accuracy =
      (ExtendedMantissa.ofMantissaAndAccuracy mantissa accuracy, k - 23) := by
  have hlog := log2_eq_23_of_normalized_mantissa mantissa hlow hhigh
  have htarget :
      Float.Model.Format.binary32.targetExponent
          (Float.Model.totalExponent mantissa (k - 23)) = k - 23 := by
    unfold Float.Model.Format.targetExponent Float.Model.totalExponent
    simp only [Float.Model.Format.mantissaBits, Float.Model.Format.minExponent, hlog]
    norm_num
    grind
  unfold Float.Model.UnpackedFloat.shiftToTargetExponent
    Float.Model.UnpackedFloat.shiftToExponent
  rw [htarget]
  simp [HShiftRight.hShiftRight, Nat.repeat, ExtendedMantissa.ofMantissaAndAccuracy]

/--
Rounding a normalized 24-bit mantissa only applies the supplied nearest-even accuracy decision.

The exponent is already the binary32 target exponent, so no residual bit is shifted before the
decision. This is the operation-independent form used by multiplication, division, and square
root once they have produced a normalized intermediate.
-/
theorem roundWithAccuracy_normalized_eq_finishRoundedMantissa
    (sign : Sign) (mantissa : Nat) (k : Int) (accuracy : Accuracy)
    (hlow : pow2 23 ≤ mantissa) (hhigh : mantissa < pow2 24) (hk : -126 ≤ k) :
    Float.Model.UnpackedFloat.roundWithAccuracy
        Float.Model.Format.binary32 sign mantissa (k - 23) accuracy =
      finishRoundedMantissa Float.Model.Format.binary32 sign
        (accuracy.roundToNearestEven mantissa, k - 23) := by
  rw [roundWithAccuracy_eq_finishRoundedMantissa]
  rw [shiftToTargetExponent_normal_mantissa mantissa k hlow hhigh hk accuracy]
  cases accuracy with
  | exact => rfl
  | inexact ordering => cases ordering <;> rfl

/-- Pack a normalized 24-bit mantissa with leading exponent `k`. -/
theorem model_finishRoundedMantissa_normal
    (sign : Sign) (mantissa : Nat) (k : Int)
    (hlow : pow2 23 ≤ mantissa) (hhigh : mantissa < pow2 24)
    (hkLow : -126 ≤ k) (hkHigh : k ≤ 127) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (finishRoundedMantissa Float.Model.Format.binary32 sign (mantissa, k - 23))) =
      IEEE32Exec.ofBits
        (IEEE32Exec.mkBits (signToBool sign) (k + 127).toNat
          (mantissa - pow2 23)) := by
  have hm : mantissa ≠ 0 :=
    Nat.ne_of_gt ((pow2_pos 23).trans_le hlow)
  have hlog := log2_eq_23_of_normalized_mantissa mantissa hlow hhigh
  have hfinish :
      finishRoundedMantissa Float.Model.Format.binary32 sign (mantissa, k - 23) =
        .finite sign mantissa (k - 23) (Nat.pos_of_ne_zero hm) := by
    unfold finishRoundedMantissa
    simp only [shiftToTargetExponent_normal_mantissa mantissa k hlow hhigh hkLow .exact,
      ExtendedMantissa.ofMantissaAndAccuracy, hm, ↓reduceDIte]
  rw [hfinish]
  unfold modelToIEEE32Exec Float32.Model.pack Float.Model.UnpackedFloat.pack
  simp only [Float.Model.Format.exponentBias, Float.Model.Format.mantissaBits, hlog]
  norm_num
  have hbiased : (k - 23 + 127 + 23).toNat = (k + 127).toNat := by
    congr 1
    grind
  rw [hbiased]
  have hbiasedNonneg : 0 ≤ k + 127 := by grind
  have hbiasedValue : ((k + 127).toNat : Int) = k + 127 :=
    Int.toNat_of_nonneg hbiasedNonneg
  have hnotInf : ¬256 ≤ (k + 127).toNat + 1 := by
    grind
  simp only [hnotInf, if_false]
  rw [packComponents_eq_mkBits]
  have hexponent :
      (BitVec.ofNat 8 (k + 127).toNat).toNat = (k + 127).toNat := by
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt]
    grind
  have hdouble : pow2 24 = pow2 23 + pow2 23 := by
    norm_num [pow2, Nat.shiftLeft_eq]
  have hsubLt : mantissa - pow2 23 < pow2 23 := by grind
  have hfraction :
      (BitVec.ofNat 23 mantissa).toNat = mantissa - pow2 23 := by
    rw [BitVec.toNat_ofNat]
    change mantissa % pow2 23 = mantissa - pow2 23
    rw [Nat.mod_eq_sub_mod hlow, Nat.mod_eq_of_lt hsubLt]
  rw [hexponent, hfraction]

private theorem shiftToTargetExponent_normal_carry (k : Int) (hk : -126 ≤ k) :
    Float.Model.UnpackedFloat.shiftToTargetExponent Float.Model.Format.binary32
        (pow2 24) (k - 23) .exact =
      (ExtendedMantissa.ofMantissaAndAccuracy (pow2 23) .exact, k - 22) := by
  have hlog : (pow2 24).log2 = 24 := by
    norm_num [pow2, Nat.shiftLeft_eq]
    decide
  have htarget :
      Float.Model.Format.binary32.targetExponent
          (Float.Model.totalExponent (pow2 24) (k - 23)) = k - 22 := by
    unfold Float.Model.Format.targetExponent Float.Model.totalExponent
    simp only [Float.Model.Format.mantissaBits, Float.Model.Format.minExponent, hlog]
    norm_num
    grind
  unfold Float.Model.UnpackedFloat.shiftToTargetExponent
    Float.Model.UnpackedFloat.shiftToExponent
  rw [htarget]
  have hdistance : (k - 22 - (k - 23)).toNat = 1 := by grind
  have hexponent : k - 23 + (1 : Nat) = k - 22 := by grind
  simp (config := { zeta := true }) only [hdistance]
  apply Prod.ext
  · norm_num [HShiftRight.hShiftRight, Nat.repeat,
      ExtendedMantissa.ofMantissaAndAccuracy, ExtendedMantissa.shiftRightOne,
      ExtendedMantissa.accuracy, Accuracy.roundToNearestEven, pow2, Nat.shiftLeft_eq]
  · exact hexponent

private theorem finishRoundedMantissa_normal_carry (sign : Sign) (k : Int) (hk : -126 ≤ k) :
    finishRoundedMantissa Float.Model.Format.binary32 sign (pow2 24, k - 23) =
      finishRoundedMantissa Float.Model.Format.binary32 sign (pow2 23, k - 22) := by
  have hleft :
      finishRoundedMantissa Float.Model.Format.binary32 sign (pow2 24, k - 23) =
        .finite sign (pow2 23) (k - 22) (by decide) := by
    unfold finishRoundedMantissa
    simp only [shiftToTargetExponent_normal_carry k hk,
      ExtendedMantissa.ofMantissaAndAccuracy, show pow2 23 ≠ 0 by decide,
      ↓reduceDIte]
  have hright :
      finishRoundedMantissa Float.Model.Format.binary32 sign (pow2 23, k - 22) =
        .finite sign (pow2 23) (k - 22) (by decide) := by
    have hlow : pow2 23 ≤ pow2 23 := Nat.le_refl _
    have hhigh : pow2 23 < pow2 24 := by decide
    have hshift := shiftToTargetExponent_normal_mantissa
      (pow2 23) (k + 1) hlow hhigh (by grind) .exact
    rw [show k - 22 = k + 1 - 23 by grind]
    unfold finishRoundedMantissa
    simp only [hshift, ExtendedMantissa.ofMantissaAndAccuracy,
      show pow2 23 ≠ 0 by decide, ↓reduceDIte]
  rw [hleft, hright]

/-- A carry from `2^24` is packed as the next normal exponent below overflow. -/
theorem model_finishRoundedMantissa_normal_carry
    (sign : Sign) (k : Int) (hkLow : -126 ≤ k) (hkHigh : k ≤ 126) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (finishRoundedMantissa Float.Model.Format.binary32 sign (pow2 24, k - 23))) =
      IEEE32Exec.ofBits
        (IEEE32Exec.mkBits (signToBool sign) (k + 128).toNat 0) := by
  rw [finishRoundedMantissa_normal_carry sign k hkLow]
  have hnormal := model_finishRoundedMantissa_normal sign (pow2 23) (k + 1)
    (Nat.le_refl _) (by decide) (by grind) (by grind)
  simpa only [show k - 22 = k + 1 - 23 by grind,
    show k + 1 + 127 = k + 128 by grind, Nat.sub_self] using hnormal

/-- A normalized mantissa whose leading exponent exceeds `127` packs as signed infinity. -/
theorem model_finishRoundedMantissa_normal_overflow
    (sign : Sign) (mantissa : Nat) (k : Int)
    (hlow : pow2 23 ≤ mantissa) (hhigh : mantissa < pow2 24)
    (hkLow : -126 ≤ k) (hkHigh : 127 < k) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (finishRoundedMantissa Float.Model.Format.binary32 sign (mantissa, k - 23))) =
      if signToBool sign then IEEE32Exec.negInf else IEEE32Exec.posInf := by
  have hm : mantissa ≠ 0 :=
    Nat.ne_of_gt ((pow2_pos 23).trans_le hlow)
  have hlog := log2_eq_23_of_normalized_mantissa mantissa hlow hhigh
  have hfinish :
      finishRoundedMantissa Float.Model.Format.binary32 sign (mantissa, k - 23) =
        .finite sign mantissa (k - 23) (Nat.pos_of_ne_zero hm) := by
    unfold finishRoundedMantissa
    simp only [shiftToTargetExponent_normal_mantissa mantissa k hlow hhigh hkLow .exact,
      ExtendedMantissa.ofMantissaAndAccuracy, hm, ↓reduceDIte]
  rw [hfinish]
  unfold modelToIEEE32Exec Float32.Model.pack Float.Model.UnpackedFloat.pack
  simp only [Float.Model.Format.exponentBias, Float.Model.Format.mantissaBits, hlog]
  norm_num
  have hbiased : (k - 23 + 127 + 23).toNat = (k + 127).toNat := by
    congr 1
    grind
  rw [hbiased]
  have hbiasedNonneg : 0 ≤ k + 127 := by grind
  have hbiasedValue : ((k + 127).toNat : Int) = k + 127 :=
    Int.toNat_of_nonneg hbiasedNonneg
  have hinf : 256 ≤ (k + 127).toNat + 1 := by grind
  simp only [hinf, if_true]
  change IEEE32Exec.ofBits
      (UInt32.ofBitVec
        (Float.Model.UnpackedFloat.packedInfinity Float.Model.Format.binary32 sign)) = _
  rw [show UInt32.ofBitVec
      (Float.Model.UnpackedFloat.packedInfinity Float.Model.Format.binary32 sign) =
        IEEE32Exec.mkBits (signToBool sign) 255 0 by
    simpa [Float.Model.UnpackedFloat.packedInfinity] using
      packComponents_eq_mkBits sign (-1#8) (0#23)]
  cases sign <;>
    rfl

/-- A normalized carry at exponent `127` or above packs as signed infinity. -/
theorem model_finishRoundedMantissa_normal_carry_overflow
    (sign : Sign) (k : Int) (hkLow : -126 ≤ k) (hkHigh : 127 ≤ k) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (finishRoundedMantissa Float.Model.Format.binary32 sign (pow2 24, k - 23))) =
      if signToBool sign then IEEE32Exec.negInf else IEEE32Exec.posInf := by
  rw [finishRoundedMantissa_normal_carry sign k hkLow]
  have hoverflow := model_finishRoundedMantissa_normal_overflow
    sign (pow2 23) (k + 1) (Nat.le_refl _) (by decide) (by grind) (by grind)
  simpa only [show k - 22 = k + 1 - 23 by grind] using hoverflow

/--
Lean's exact binary32 rounder and `IEEE32Exec` agree throughout the normal exponent range.

The statement includes overflow. In particular, it covers both an input whose leading exponent is
already above `127` and a value at exponent `127` whose rounded mantissa carries into infinity.
-/
theorem model_round_exact_normal_eq_roundDyadic
    (sign : Sign) (mantissa : Nat) (exponent : Int) (hm : mantissa ≠ 0)
    (hlow : -126 ≤ (mantissa.log2 : Int) + exponent) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (Float.Model.UnpackedFloat.round
            Float.Model.Format.binary32 sign mantissa exponent)) =
      roundDyadicToIEEE32
        { sign := signToBool sign, mant := mantissa, exp := exponent } := by
  let k : Int := (mantissa.log2 : Int) + exponent
  let m24 : Nat := roundMantissaToLeadingBitEven mantissa 23
  have htarget :
      Float.Model.Format.binary32.targetExponent
          (Float.Model.totalExponent mantissa exponent) = k - 23 := by
    unfold Float.Model.Format.targetExponent Float.Model.totalExponent
    simp only [Float.Model.Format.mantissaBits, Float.Model.Format.minExponent]
    norm_num
    dsimp only [k]
    grind
  have hm24Round :
      roundMantissaAtExponentEven mantissa exponent (k - 23) = m24 := by
    dsimp only [k, m24]
    exact roundMantissaAtExponentEven_eq_roundMantissaToLeadingBitEven
      mantissa 23 exponent
  have hround :
      Float.Model.UnpackedFloat.round
          Float.Model.Format.binary32 sign mantissa exponent =
        finishRoundedMantissa Float.Model.Format.binary32 sign (m24, k - 23) := by
    rw [round_exact_eq_finishRoundedMantissa
      Float.Model.Format.binary32 sign mantissa exponent hm]
    rw [htarget, hm24Round]
  have hm24Low : pow2 23 ≤ m24 :=
    pow2_le_roundMantissaToLeadingBitEven mantissa 23 hm
  have hm24Le : m24 ≤ pow2 24 := by
    simpa only [show 23 + 1 = 24 by grind] using
      roundMantissaToLeadingBitEven_le_pow2_succ mantissa 23
  rw [hround]
  by_cases hkHigh : 127 < k
  · have hover : Int.ofNat mantissa.log2 + exponent > 127 := by
      simpa only [Int.ofNat_eq_natCast, k] using hkHigh
    have hexec :
        roundDyadicToIEEE32
            { sign := signToBool sign, mant := mantissa, exp := exponent } =
          if signToBool sign then IEEE32Exec.negInf else IEEE32Exec.posInf := by
      unfold roundDyadicToIEEE32
      simp only [beq_iff_eq, hm, if_false]
      rw [if_pos hover]
    rw [hexec]
    by_cases hcarry : m24 = pow2 24
    · rw [hcarry]
      exact model_finishRoundedMantissa_normal_carry_overflow sign k hlow (by grind)
    · exact model_finishRoundedMantissa_normal_overflow sign m24 k hm24Low
        (lt_of_le_of_ne hm24Le hcarry) hlow hkHigh
  · have hkLe : k ≤ 127 := by grind
    have hover : ¬Int.ofNat mantissa.log2 + exponent > 127 := by
      simpa only [Int.ofNat_eq_natCast, k] using hkHigh
    have hnotUnder : ¬Int.ofNat mantissa.log2 + exponent < -150 := by
      simpa only [Int.ofNat_eq_natCast, k] using
        (show ¬k < -150 by grind)
    have hnotSubnormal : ¬Int.ofNat mantissa.log2 + exponent < -126 := by
      simpa only [Int.ofNat_eq_natCast, k] using
        (show ¬k < -126 by grind)
    have hexec :
        roundDyadicToIEEE32
            { sign := signToBool sign, mant := mantissa, exp := exponent } =
          let k' : Int := if m24 = pow2 24 then k + 1 else k
          let m24' : Nat := if m24 = pow2 24 then pow2 23 else m24
          if k' > 127 then
            if signToBool sign then IEEE32Exec.negInf else IEEE32Exec.posInf
          else
            IEEE32Exec.ofBits
              (IEEE32Exec.mkBits (signToBool sign) (k' + 127).toNat
                (m24' - pow2 23)) := by
      unfold roundDyadicToIEEE32
      simp only [beq_iff_eq, hm, if_false]
      rw [if_neg hover, if_neg hnotUnder, if_neg hnotSubnormal]
      rfl
    rw [hexec]
    by_cases hcarry : m24 = pow2 24
    · rw [hcarry]
      by_cases hkEdge : k = 127
      · rw [model_finishRoundedMantissa_normal_carry_overflow sign k hlow (by grind)]
        simp [hkEdge]
      · have hk126 : k ≤ 126 := by grind
        rw [model_finishRoundedMantissa_normal_carry sign k hlow hk126]
        rw [if_neg (by grind)]
        simp
        congr 2
        grind
    · have hm24Lt : m24 < pow2 24 := lt_of_le_of_ne hm24Le hcarry
      rw [model_finishRoundedMantissa_normal sign m24 k hm24Low hm24Lt hlow hkLe]
      simp [hcarry, hkLe]

/-- Lean's exact binary32 rounder and `IEEE32Exec` agree for every signed dyadic. -/
theorem model_round_exact_eq_roundDyadic
    (sign : Sign) (mantissa : Nat) (exponent : Int) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (Float.Model.UnpackedFloat.round
            Float.Model.Format.binary32 sign mantissa exponent)) =
      roundDyadicToIEEE32
        { sign := signToBool sign, mant := mantissa, exp := exponent } := by
  by_cases hm : mantissa = 0
  · subst mantissa
    exact model_round_exact_zero_eq_roundDyadic sign exponent
  by_cases hnormal : -126 ≤ (mantissa.log2 : Int) + exponent
  · exact model_round_exact_normal_eq_roundDyadic sign mantissa exponent hm hnormal
  have hsubnormal : (mantissa.log2 : Int) + exponent < -126 := lt_of_not_ge hnormal
  by_cases hunderflow : (mantissa.log2 : Int) + exponent < -150
  · exact model_round_exact_underflow_eq_roundDyadic sign mantissa exponent hunderflow
  exact model_round_exact_subnormal_eq_roundDyadic sign mantissa exponent hm
    (le_of_not_gt hunderflow) hsubnormal

/--
Lean's exact `roundWithAccuracy` path agrees with executable binary32 dyadic rounding whenever its
documented target-exponent precondition holds.
-/
theorem model_roundWithAccuracy_exact_eq_roundDyadic_of_le_targetExponent
    (sign : Sign) (mantissa : Nat) (exponent : Int)
    (h : exponent ≤ Float.Model.Format.binary32.targetExponent
      (Float.Model.totalExponent mantissa exponent)) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (Float.Model.UnpackedFloat.roundWithAccuracy
            Float.Model.Format.binary32 sign mantissa exponent .exact)) =
      roundDyadicToIEEE32
        { sign := signToBool sign, mant := mantissa, exp := exponent } := by
  rw [roundWithAccuracy_exact_eq_round_of_le_targetExponent _ _ _ _ h]
  exact model_round_exact_eq_roundDyadic sign mantissa exponent

end TorchLean.Floats.IEEE754.Float32Bridge
