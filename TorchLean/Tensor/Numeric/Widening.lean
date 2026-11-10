/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Kernels.FixedWord.Core.Runtime
import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Proof
import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Basic
import FloatLib.Floats.Formats.BinaryInterchange.Model.Fields.Optimized
import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Directed.Dyadic
import FloatLib.Kernels.FixedWord.Core.Proof.Word

/-!
# Bit-level widening of configured binary values to `Float`

FloatLib's `ExecFloat.Binary.toFloat` reads a configured value back as Lean's `Float`. A binary64
value crosses as its word, and every other format goes through `Model.cast` into binary64, one call
of the compiled cast status kernel per element. When binary64 holds every value of the source
format, that cast only re-encodes the fields, so we compute the binary64 word with `UInt64` shifts,
masks, and additions and prove it equal to FloatLib's word for every value and rounding mode, NaN
payloads included.

A NaN keeps its sign and the integer value of its payload, as in FloatLib's cast, so binary32
`0x7FC00001` widens to `0x7FF8000000000001`. Hardware does otherwise. Agner Fog's 2018 note on
NaN payloads, a background document of the 754-2019 revision
(https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/nan-propagation.pdf),
reports that on every platform investigated a widening conversion appends zeros to the payload,
which gives `0x7FF8000020000000`. We match FloatLib because the `csimp` lemma needs its word, and
Lean's `Float.ofBits` turns both words into the same quiet NaN at run time.

The bit path covers the formats with `fracWidth ≤ 52`, `exponentBias + fracWidth ≤ 1075`, and
`maxFiniteExpField ≤ exponentBias + 1023`, in all four encodings. In FloatLib's catalog these are
the formats of at most 64 bits: binary16, bfloat16, binary32, tf32, the eight-bit and smaller
formats, and binary64, which crosses as its word. `widensToWord64` decides these conditions with
word operations, because `Nat.pow` in `maxFiniteExpField` costs more than the widening itself.
`toFloat_eq_widenToFloat` is a `csimp` lemma, so it rewrites `toFloat` only in code compiled in
modules that import this one.

## Main declarations

* `widenToWord64`: the binary64 word of a source interchange word, by bit arithmetic.
* `toWord64_eq_widenToWord64`: `widenToWord64` agrees with FloatLib's cast into binary64.
* `widensToWord64`, `widensToWord64_iff`: the conditions of the bit path as a `Bool`.
* `widenToFloat`: `toFloat` with the bit path.
* `toFloat_eq_widenToFloat`: the `csimp` equality of `toFloat` and `widenToFloat`.
-/

@[expose] public section

namespace TorchLean.Numeric

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Numerics.FixedWord (log2Word log2Word_toNat shiftLeft_toNat
  two_pow_le_shiftLeft_sub_log2 shiftLeft_sub_log2_lt_two_pow shiftLeft_lt_two_pow_of_log2_add_le)

/-! ## Exact rounding into binary64 -/

/-- Packing a nonzero significand `r < 2 ^ 52` as a binary64 subnormal gives exponent field zero
and fraction `r`. -/
private theorem packRoundedSubnormal_binary64 (neg : Bool) (zero : Model .binary64) (r : Nat)
    (h0 : r ≠ 0) (h : r < 2 ^ 52) :
    Model.packRoundedSubnormal .binary64 neg zero r = Model.ofFields .binary64 neg 0 r := by
  unfold Model.packRoundedSubnormal
  simp only [beq_iff_eq, h0, ite_false, Model.pow2_eq_two_pow,
    show FloatFormat.binary64.fracWidth = 52 by rfl]
  rw [ite_eq_right (by omega)]

/-- Packing a significand in `[2 ^ 52, 2 ^ 53)` at an exponent `t` in `[-1022, 1023]` gives the
binary64 word with exponent field `t + 1023`. -/
private theorem packRoundedNormal_binary64 (neg : Bool) (overflow : Model .binary64) (t : Int)
    (r : Nat) (ht : -1022 ≤ t) (ht' : t ≤ 1023) (hr : 2 ^ 52 ≤ r) (hr' : r < 2 ^ 53) :
    Model.packRoundedNormal .binary64 neg overflow t r =
      Model.ofFields .binary64 neg (t + 1023).toNat (r - 2 ^ 52) := by
  unfold Model.packRoundedNormal
  simp only [Model.pow2_eq_two_pow, beq_iff_eq, show FloatFormat.binary64.fracWidth = 52 by rfl,
    show FloatFormat.binary64.maxNormalExponent = 1023 by decide,
    show FloatFormat.binary64.maxFiniteExpField = 2046 by decide,
    show FloatFormat.binary64.maxFiniteFracField = 2 ^ 52 - 1 by decide,
    show FloatFormat.binary64.exponentBias = 1023 by rfl]
  have hne : r ≠ 2 ^ (52 + 1) := by omega
  simp only [hne, ite_false]
  rw [ite_eq_right (by omega), ite_eq_right (by simp; omega)]
  rfl

/-- Nearest-even rounding into binary64 gives the exact word of `m * 2 ^ q` when `m ≠ 0`,
`log2 m ≤ 52`, `-1074 ≤ q`, and `log2 m + q ≤ 1023`. -/
private theorem roundDyadicGeneral_binary64 (neg : Bool) (m : Nat) (q : Int) (hm : m ≠ 0)
    (hlog : Nat.log2 m ≤ 52) (hq : -1074 ≤ q) (htop : (Nat.log2 m : Int) + q ≤ 1023) :
    Model.roundDyadicGeneral .binary64 ⟨neg, m, q⟩ =
      if (Nat.log2 m : Int) + q < -1022 then
        Model.ofFields .binary64 neg 0 (m <<< (q + 1074).toNat)
      else
        Model.ofFields .binary64 neg ((Nat.log2 m : Int) + q + 1023).toNat
          (m <<< (52 - Nat.log2 m) - 2 ^ 52) := by
  have hnlo := two_pow_le_shiftLeft_sub_log2 52 m hm
  have hnhi : m <<< (52 - m.log2) < 2 ^ 53 := shiftLeft_sub_log2_lt_two_pow 52 m hlog
  have hk : q - FloatFormat.binary64.minSubnormalExponent = Int.ofNat (q + 1074).toNat := by
    rw [show FloatFormat.binary64.minSubnormalExponent = -1074 by decide, Int.ofNat_eq_natCast]
    omega
  unfold Model.roundDyadicGeneral
  simp only [hk, beq_iff_eq, hm, ite_false,
    show FloatFormat.binary64.maxNormalExponent = 1023 by decide,
    show FloatFormat.binary64.minNormalExponent = -1022 by decide,
    show FloatFormat.binary64.fracWidth = 52 by rfl]
  rw [ite_eq_right (by omega)]
  have hround : (if 52 ≤ m.log2 then FloatLib.Numerics.roundShiftRightEven m (m.log2 - 52)
      else m <<< (52 - m.log2)) = m <<< (52 - m.log2) := by
    split_ifs with h
    · rw [show m.log2 - 52 = 0 by omega, show 52 - m.log2 = 0 by omega,
        FloatLib.Numerics.roundShiftRightEven_zero, Nat.shiftLeft_zero]
    · rfl
  rw [hround]
  split_ifs with hsub
  · exact packRoundedSubnormal_binary64 _ _ _
      (by rw [Nat.shiftLeft_eq]; exact Nat.mul_ne_zero hm (Nat.two_pow_pos _).ne')
      (shiftLeft_lt_two_pow_of_log2_add_le _ _ _ (by omega))
  · exact packRoundedNormal_binary64 _ _ _ _ (by omega) htop hnlo hnhi

/-- Rounding the magnitude up into binary64 gives the exact word of `m * 2 ^ q` under the
conditions of `roundDyadicGeneral_binary64`. -/
private theorem roundDyadicMagnitudeUp_binary64 (neg : Bool) (m : Nat) (q : Int) (hm : m ≠ 0)
    (hlog : Nat.log2 m ≤ 52) (hq : -1074 ≤ q) (htop : (Nat.log2 m : Int) + q ≤ 1023) :
    Model.roundDyadicMagnitudeUp .binary64 neg m q =
      if (Nat.log2 m : Int) + q < -1022 then
        Model.ofFields .binary64 neg 0 (m <<< (q + 1074).toNat)
      else
        Model.ofFields .binary64 neg ((Nat.log2 m : Int) + q + 1023).toNat
          (m <<< (52 - Nat.log2 m) - 2 ^ 52) := by
  have hnlo := two_pow_le_shiftLeft_sub_log2 52 m hm
  have hnhi : m <<< (52 - m.log2) < 2 ^ 53 := shiftLeft_sub_log2_lt_two_pow 52 m hlog
  have hk : q - FloatFormat.binary64.minSubnormalExponent = Int.ofNat (q + 1074).toNat := by
    rw [show FloatFormat.binary64.minSubnormalExponent = -1074 by decide, Int.ofNat_eq_natCast]
    omega
  unfold Model.roundDyadicMagnitudeUp
  simp only [hk, show FloatFormat.binary64.maxNormalExponent = 1023 by decide,
    show FloatFormat.binary64.minNormalExponent = -1022 by decide,
    show FloatFormat.binary64.fracWidth = 52 by rfl]
  simp only [Int.ofNat_eq_natCast, Nat.shiftLeft_eq',
    show FloatFormat.binary64.minSubnormalExponent = -1074 by decide]
  rw [ite_eq_right (by omega), ite_eq_right (by omega)]
  have hround : (if m.log2 ≥ 52 then FloatLib.Numerics.shiftRightCeil m (m.log2 - 52)
      else m <<< (52 - m.log2)) = m <<< (52 - m.log2) := by
    split_ifs with h
    · rw [show m.log2 - 52 = 0 by omega, show 52 - m.log2 = 0 by omega,
        FloatLib.Numerics.shiftRightCeil_zero, Nat.shiftLeft_zero]
    · rfl
  rw [hround]
  split_ifs with hsub
  · exact packRoundedSubnormal_binary64 _ _ _
      (by rw [Nat.shiftLeft_eq]; exact Nat.mul_ne_zero hm (Nat.two_pow_pos _).ne')
      (shiftLeft_lt_two_pow_of_log2_add_le _ _ _ (by omega))
  · exact packRoundedNormal_binary64 _ _ _ _ (by omega) htop hnlo hnhi

/-- Rounding the magnitude down into binary64 gives the exact word of `m * 2 ^ q` under the
conditions of `roundDyadicGeneral_binary64`. -/
private theorem roundDyadicMagnitudeDown_binary64 (neg : Bool) (m : Nat) (q : Int) (hm : m ≠ 0)
    (hlog : Nat.log2 m ≤ 52) (hq : -1074 ≤ q) (htop : (Nat.log2 m : Int) + q ≤ 1023) :
    Model.roundDyadicMagnitudeDown .binary64 neg m q =
      if (Nat.log2 m : Int) + q < -1022 then
        Model.ofFields .binary64 neg 0 (m <<< (q + 1074).toNat)
      else
        Model.ofFields .binary64 neg ((Nat.log2 m : Int) + q + 1023).toNat
          (m <<< (52 - Nat.log2 m) - 2 ^ 52) := by
  have hnlo := two_pow_le_shiftLeft_sub_log2 52 m hm
  have hnhi : m <<< (52 - m.log2) < 2 ^ 53 := shiftLeft_sub_log2_lt_two_pow 52 m hlog
  have hk : q - FloatFormat.binary64.minSubnormalExponent = Int.ofNat (q + 1074).toNat := by
    rw [show FloatFormat.binary64.minSubnormalExponent = -1074 by decide, Int.ofNat_eq_natCast]
    omega
  unfold Model.roundDyadicMagnitudeDown
  simp only [hk, show FloatFormat.binary64.maxNormalExponent = 1023 by decide,
    show FloatFormat.binary64.minNormalExponent = -1022 by decide,
    show FloatFormat.binary64.fracWidth = 52 by rfl]
  simp only [Int.ofNat_eq_natCast, Nat.shiftLeft_eq',
    show FloatFormat.binary64.minSubnormalExponent = -1074 by decide]
  rw [ite_eq_right (by omega), ite_eq_right (by omega)]
  have hround : (if m.log2 ≥ 52 then m.shiftRight (m.log2 - 52) else m <<< (52 - m.log2)) =
      m <<< (52 - m.log2) := by
    split_ifs with h
    · rw [show m.log2 - 52 = 0 by omega, show 52 - m.log2 = 0 by omega, Nat.shiftLeft_zero]
      rfl
    · rfl
  have hne : m <<< (q + 1074).toNat ≠ 0 := by
    rw [Nat.shiftLeft_eq]; exact Nat.mul_ne_zero hm (Nat.two_pow_pos _).ne'
  simp only [hround, beq_iff_eq, hne, ite_false, Model.pow2_eq_two_pow,
    show FloatFormat.binary64.exponentBias = 1023 by rfl,
    show FloatFormat.binary64.maxFiniteExpField = 2046 by decide,
    show FloatFormat.binary64.maxFiniteFracField = 2 ^ 52 - 1 by decide]
  split_ifs with hsub hover
  · rfl
  · exfalso
    simp only [Nat.cast_ofNat, Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true,
      beq_iff_eq] at hover
    omega
  · rfl

/-- A dyadic `m * 2 ^ q` with `log2 m ≤ 52` and `-1074 ≤ q` is never a binary64 grid tie. -/
private theorem dyadicIsGridTie_binary64 (neg : Bool) (m : Nat) (q : Int)
    (hlog : Nat.log2 m ≤ 52) (hq : -1074 ≤ q) :
    Model.dyadicIsGridTie .binary64 ⟨neg, m, q⟩ = false := by
  unfold Model.dyadicIsGridTie
  split
  · rfl
  · rename_i shift h
    have hneg := Int.negSucc_lt_zero shift
    rw [show FloatFormat.binary64.minSubnormalExponent = -1074 by decide] at h
    simp only [show FloatFormat.binary64.fracWidth = 52 by rfl] at h
    omega

/-- Every rounding direction gives the exact binary64 word of `m * 2 ^ q` under the conditions of
`roundDyadicGeneral_binary64`. -/
private theorem roundDyadic_binary64 (mode : Model.IEEERoundingMode) (neg : Bool) (m : Nat)
    (q : Int) (hm : m ≠ 0) (hlog : Nat.log2 m ≤ 52) (hq : -1074 ≤ q)
    (htop : (Nat.log2 m : Int) + q ≤ 1023) :
    Model.Internal.Directed.roundDyadic .binary64 mode ⟨neg, m, q⟩ =
      if (Nat.log2 m : Int) + q < -1022 then
        Model.ofFields .binary64 neg 0 (m <<< (q + 1074).toNat)
      else
        Model.ofFields .binary64 neg ((Nat.log2 m : Int) + q + 1023).toNat
          (m <<< (52 - Nat.log2 m) - 2 ^ 52) := by
  have hgen := roundDyadicGeneral_binary64 neg m q hm hlog hq htop
  rw [Model.roundDyadicGeneral_eq_roundDyadic] at hgen
  have hup := roundDyadicMagnitudeUp_binary64 neg m q hm hlog hq htop
  have hdown := roundDyadicMagnitudeDown_binary64 neg m q hm hlog hq htop
  have htie := dyadicIsGridTie_binary64 neg m q hlog hq
  cases mode <;> cases neg <;>
    simp only [Model.Internal.Directed.roundDyadic, Model.roundDyadicNearestAway,
      Model.roundDyadicTowardZero, Model.roundDyadicUp, Model.roundDyadicDown,
      Model.roundDyadicAway, htie, beq_iff_eq, hm, ite_false, Bool.false_eq_true, ite_true,
      hgen, hup, hdown]

/-- Every rounding direction sends a zero significand to the binary64 zero of the same sign. -/
private theorem roundDyadic_binary64_zero (mode : Model.IEEERoundingMode) (neg : Bool)
    (q : Int) :
    Model.Internal.Directed.roundDyadic .binary64 mode ⟨neg, 0, q⟩ = Model.zero .binary64 neg := by
  have hgen : Model.roundDyadic .binary64 ⟨neg, 0, q⟩ = Model.zero .binary64 neg := by
    rw [← Model.roundDyadicGeneral_eq_roundDyadic]
    simp [Model.roundDyadicGeneral]
  cases mode <;>
    simp [Model.Internal.Directed.roundDyadic, Model.roundDyadicNearestAway,
      Model.roundDyadicTowardZero, Model.roundDyadicUp, Model.roundDyadicDown,
      Model.roundDyadicAway, hgen]

/-! ## Interchange words of the binary64 cast -/

/-- The interchange word of `Model.ofFields` places the sign, the exponent field, and the fraction
field side by side. -/
private theorem toNatBits_ofFields (fmt : FloatFormat) (sign : Bool) (exponent fraction : Nat)
    (he : exponent < 2 ^ fmt.expWidth) (hf : fraction < 2 ^ fmt.fracWidth) :
    (Model.ofFields fmt sign exponent fraction).toNatBits =
      2 ^ (fmt.expWidth + fmt.fracWidth) * sign.toNat +
        (2 ^ fmt.fracWidth * exponent + fraction) := by
  rw [Model.ofFields_eq_internal]
  simp only [Model.Internal.ofFields, Model.Internal.mkBits, Model.ofBits, Model.toNatBits,
    BitVec.toNat_ofNatLT, FloatFormat.expAllOnesNat, FloatFormat.fracMaskNat,
    Nat.and_two_pow_sub_one_of_lt_two_pow he, Nat.and_two_pow_sub_one_of_lt_two_pow hf]
  have hlow : 2 ^ fmt.fracWidth * exponent + fraction < 2 ^ (fmt.expWidth + fmt.fracWidth) := by
    calc 2 ^ fmt.fracWidth * exponent + fraction < 2 ^ fmt.fracWidth * (exponent + 1) := by
          rw [Nat.mul_succ]; omega
      _ ≤ 2 ^ fmt.fracWidth * 2 ^ fmt.expWidth := Nat.mul_le_mul_left _ he
      _ = 2 ^ (fmt.expWidth + fmt.fracWidth) := by rw [← Nat.pow_add, Nat.add_comm]
  rw [Nat.two_pow_add_eq_or_of_lt hf, Nat.two_pow_add_eq_or_of_lt
    (by simpa [Nat.two_pow_add_eq_or_of_lt hf] using hlow)]
  cases sign <;> simp [Nat.shiftLeft_eq, Nat.mul_comm, Nat.or_assoc]

/-- Every interchange word is its sign, exponent field, and fraction field side by side. -/
private theorem toNatBits_eq_fields {fmt : FloatFormat} (x : Model fmt) :
    x.toNatBits = 2 ^ (fmt.expWidth + fmt.fracWidth) * (Model.signBit x).toNat +
      (2 ^ fmt.fracWidth * Model.expField x + Model.fracField x) := by
  conv_lhs => rw [← Model.ofFields_signBit_expField_fracField x]
  exact toNatBits_ofFields fmt _ _ _ (Model.expField_lt_pow2 x) (Model.fracField_lt_pow2 x)

/-- `toNatBits_ofFields` for binary64, with the field widths evaluated. -/
private theorem toNatBits_ofFields_binary64 (sign : Bool) (exponent fraction : Nat)
    (he : exponent < 2 ^ 11) (hf : fraction < 2 ^ 52) :
    (Model.ofFields .binary64 sign exponent fraction).toNatBits =
      2 ^ 63 * sign.toNat + (2 ^ 52 * exponent + fraction) :=
  toNatBits_ofFields .binary64 sign exponent fraction he hf

/-- The binary64 zero of sign `sign` has the word `2 ^ 63 * sign.toNat`. -/
private theorem toNatBits_zero_binary64 (sign : Bool) :
    (Model.zero .binary64 sign).toNatBits = 2 ^ 63 * sign.toNat := by
  cases sign <;> rfl

/-- Under the conditions of `toWord64_eq_widenToWord64`, the binary64 cast of a finite value in
any mode has the word built from the re-biased exponent and the shifted fraction. -/
private theorem toNatBits_cast_binary64_of_isFinite {fmt : FloatFormat}
    (hfrac : fmt.fracWidth ≤ 52) (hmin : fmt.exponentBias + fmt.fracWidth ≤ 1075)
    (hmax : fmt.maxFiniteExpField ≤ fmt.exponentBias + 1023) (x : Model fmt)
    (hx : Model.isFinite x = true) (mode : Model.IEEERoundingMode) :
    (Model.cast fmt .binary64 x mode).toNatBits =
      if Model.expField x = 0 then
        if Model.fracField x = 0 then 2 ^ 63 * (Model.signBit x).toNat
        else if fmt.exponentBias + fmt.fracWidth ≤ (Model.fracField x).log2 + 1023 then
          2 ^ 63 * (Model.signBit x).toNat +
            2 ^ 52 * ((Model.fracField x).log2 + 1023 - fmt.exponentBias - fmt.fracWidth) +
            Model.fracField x <<< (52 - (Model.fracField x).log2)
        else
          2 ^ 63 * (Model.signBit x).toNat +
            Model.fracField x <<< (1075 - fmt.exponentBias - fmt.fracWidth)
      else if fmt.exponentBias ≤ Model.expField x + 1022 then
        2 ^ 63 * (Model.signBit x).toNat + 2 ^ 52 * (Model.expField x + 1023 - fmt.exponentBias) +
          Model.fracField x <<< (52 - fmt.fracWidth)
      else
        2 ^ 63 * (Model.signBit x).toNat +
          (2 ^ fmt.fracWidth + Model.fracField x) <<<
            (Model.expField x + 1074 - fmt.exponentBias - fmt.fracWidth) := by
  have hb := fmt.exponentBias_pos
  have hw := fmt.fracWidth_pos
  have he := Model.expField_lt_pow2 x
  have hf := Model.fracField_lt_pow2 x
  have hemax := Model.expField_le_maxFiniteExpField_of_isFinite x hx
  by_cases hcompat : fmt.expWidth = 11 ∧ fmt.exponentBias = 1023 ∧ fmt.encoding = .ieee
  · obtain ⟨h1, h2, h3⟩ := hcompat
    rw [Model.cast_eq_widenExact_of_compatibleWidening x h1 h2 h3 hfrac hx mode]
    unfold Model.widenExact
    simp only [show FloatFormat.binary64.fracWidth = 52 from rfl, Nat.shiftLeft_eq']
    have hshift : Model.fracField x <<< (52 - fmt.fracWidth) < 2 ^ 52 := by
      rw [Nat.shiftLeft_eq]
      calc Model.fracField x * 2 ^ (52 - fmt.fracWidth)
          < 2 ^ fmt.fracWidth * 2 ^ (52 - fmt.fracWidth) :=
            Nat.mul_lt_mul_of_pos_right hf (Nat.two_pow_pos _)
        _ = 2 ^ 52 := by rw [← Nat.pow_add]; congr 1; omega
    rw [toNatBits_ofFields_binary64 _ _ _ (by rw [h1] at he; exact he) hshift]
    have hk : 1075 - fmt.exponentBias - fmt.fracWidth = 52 - fmt.fracWidth := by omega
    rw [hk]
    by_cases he0 : Model.expField x = 0
    · by_cases hf0 : Model.fracField x = 0
      · simp [he0, hf0]
      · have hL : (Model.fracField x).log2 < fmt.fracWidth := (Nat.log2_lt hf0).2 hf
        rw [ite_eq_left he0, ite_eq_right hf0, ite_eq_right (by omega), he0]
        omega
    · rw [ite_eq_right he0, ite_eq_left (by omega)]
      omega
  · have hnan := Model.isNaN_eq_false_of_isFinite_eq_true x hx
    have hinf := Model.isInf_eq_false_of_isFinite_eq_true x hx
    have hformat : fmt ≠ .binary64 := by
      rintro rfl
      exact hcompat ⟨rfl, rfl, rfl⟩
    have hcast : Model.cast fmt .binary64 x mode =
        Model.Internal.Directed.roundDyadic .binary64 mode (Model.finiteDyadic x hx) := by
      cases mode <;>
        simp only [Model.cast, Model.Internal.Nearest.cast, Model.Internal.Directed.cast,
          Model.Internal.Directed.roundDyadic, hnan, hinf, Bool.false_eq_true, dite_false,
          hformat] <;>
        split_ifs with h1 h2 h3 <;>
        first | rfl | exact (hcompat ⟨h1, h2, h3⟩).elim
    have hdecode := Model.toDyadic?_ofFields_of_isFinite fmt (Model.signBit x) (Model.expField x)
      (Model.fracField x) he hf (by rw [Model.ofFields_signBit_expField_fracField]; exact hx)
    rw [Model.ofFields_signBit_expField_fracField] at hdecode
    rw [hcast]
    by_cases he0 : Model.expField x = 0
    · by_cases hf0 : Model.fracField x = 0
      · rw [ite_eq_left he0, ite_eq_left hf0] at hdecode
        rw [Model.finiteDyadic_eq_of_toDyadic hx hdecode, roundDyadic_binary64_zero,
          toNatBits_zero_binary64, ite_eq_left he0, ite_eq_left hf0]
      · rw [ite_eq_left he0, ite_eq_right hf0] at hdecode
        rw [Model.finiteDyadic_eq_of_toDyadic hx hdecode, ite_eq_left he0, ite_eq_right hf0]
        have hL : (Model.fracField x).log2 < fmt.fracWidth := (Nat.log2_lt hf0).2 hf
        have hsub : fmt.minSubnormalExponent = 1 - (fmt.exponentBias : Int) - fmt.fracWidth := by
          simp only [FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
            Int.ofNat_eq_natCast]
        rw [roundDyadic_binary64 mode _ _ _ hf0 (by omega) (by rw [hsub]; omega)
          (by rw [hsub]; omega)]
        have hnlo := two_pow_le_shiftLeft_sub_log2 52 _ hf0
        have hnhi : Model.fracField x <<< (52 - (Model.fracField x).log2) < 2 ^ 53 :=
          shiftLeft_sub_log2_lt_two_pow 52 _ (by omega)
        by_cases hS : fmt.exponentBias + fmt.fracWidth ≤ (Model.fracField x).log2 + 1023
        · rw [ite_eq_right (by rw [hsub]; omega), ite_eq_left hS,
            toNatBits_ofFields_binary64 _ _ _ (by rw [hsub]; omega) (by omega)]
          rw [hsub]
          omega
        · have hk : (fmt.minSubnormalExponent + 1074).toNat =
              1075 - fmt.exponentBias - fmt.fracWidth := by rw [hsub]; omega
          rw [ite_eq_left (by rw [hsub]; omega), ite_eq_right hS, hk,
            toNatBits_ofFields_binary64 _ _ _ (by omega)
              (shiftLeft_lt_two_pow_of_log2_add_le _ _ _ (by omega))]
          omega
    · rw [ite_eq_right he0] at hdecode
      rw [Model.finiteDyadic_eq_of_toDyadic hx hdecode, ite_eq_right he0, Model.pow2_eq_two_pow]
      have hm0 : 2 ^ fmt.fracWidth + Model.fracField x ≠ 0 := by
        have := Nat.two_pow_pos fmt.fracWidth; omega
      have hlog : (2 ^ fmt.fracWidth + Model.fracField x).log2 = fmt.fracWidth := by
        rw [Nat.log2_eq_iff hm0]
        constructor
        · omega
        · rw [Nat.pow_succ]; omega
      have hq : Int.ofNat (Model.expField x) - Int.ofNat fmt.exponentBias -
          Int.ofNat fmt.fracWidth =
            (Model.expField x : Int) - fmt.exponentBias - fmt.fracWidth := by
        simp only [Int.ofNat_eq_natCast]
      rw [hq, roundDyadic_binary64 mode _ _ _ hm0 (by omega) (by omega) (by omega), hlog]
      by_cases hN : fmt.exponentBias ≤ Model.expField x + 1022
      · have hsplit : (2 ^ fmt.fracWidth + Model.fracField x) <<< (52 - fmt.fracWidth) =
            2 ^ 52 + Model.fracField x <<< (52 - fmt.fracWidth) := by
          rw [Nat.shiftLeft_eq, Nat.shiftLeft_eq, Nat.add_mul, ← Nat.pow_add]
          congr 2
          omega
        have hlt : Model.fracField x <<< (52 - fmt.fracWidth) < 2 ^ 52 := by
          rw [Nat.shiftLeft_eq]
          calc Model.fracField x * 2 ^ (52 - fmt.fracWidth)
              < 2 ^ fmt.fracWidth * 2 ^ (52 - fmt.fracWidth) :=
                Nat.mul_lt_mul_of_pos_right hf (Nat.two_pow_pos _)
            _ = 2 ^ 52 := by rw [← Nat.pow_add]; congr 1; omega
        have hE : ((fmt.fracWidth : Int) + ((Model.expField x : Int) - fmt.exponentBias -
            fmt.fracWidth) + 1023).toNat = Model.expField x + 1023 - fmt.exponentBias := by omega
        rw [ite_eq_right (by omega), ite_eq_left hN, hsplit, Nat.add_sub_cancel_left, hE,
          toNatBits_ofFields_binary64 _ _ _ (by omega) hlt, Nat.add_assoc]
      · have hk : ((Model.expField x : Int) - fmt.exponentBias - fmt.fracWidth + 1074).toNat =
            Model.expField x + 1074 - fmt.exponentBias - fmt.fracWidth := by omega
        rw [ite_eq_left (by omega), ite_eq_right hN, hk,
          toNatBits_ofFields_binary64 _ _ _ (by omega)
            (shiftLeft_lt_two_pow_of_log2_add_le _ _ _ (by rw [hlog]; omega))]
        omega

/-- The binary64 cast of an infinity in any mode is the binary64 infinity of the same sign. -/
private theorem toNatBits_cast_binary64_of_isInf {fmt : FloatFormat} (x : Model fmt)
    (hinf : Model.isInf x = true) (mode : Model.IEEERoundingMode) :
    (Model.cast fmt .binary64 x mode).toNatBits =
      2 ^ 63 * (Model.signBit x).toNat + 0x7FF0000000000000 := by
  have hnan := Model.isNaN_eq_false_of_isInf_eq_true x hinf
  have hcast : Model.cast fmt .binary64 x mode =
      Model.nativeOverflow .binary64 (Model.signBit x) := by
    cases mode <;>
      simp only [Model.cast, Model.Internal.Nearest.cast, Model.Internal.Directed.cast, hnan,
        hinf, Bool.false_eq_true, dite_false, dite_true]
  rw [hcast, Model.nativeOverflow_eq_signedInf_of_isIEEE .binary64 rfl]
  cases Model.signBit x <;> rfl

/-- The NaN payload of `x` is its fraction field without the quiet bit for the IEEE encoding and
zero for the other encodings. -/
private theorem nanPayload_eq {fmt : FloatFormat} (x : Model fmt) (hnan : Model.isNaN x = true) :
    Model.nanPayload x =
      if fmt.encoding = .ieee then Model.fracField x % 2 ^ (fmt.fracWidth - 1) else 0 := by
  have hw := fmt.fracWidth_pos
  have hf := Model.fracField_lt_pow2 x
  unfold Model.nanPayload Model.payloadOfNaNField
  cases henc : fmt.encoding
  · have hsnan : Model.isSNaN x = !(Model.fracField x).testBit (fmt.fracWidth - 1) := by
      have hq := Model.and_quietBit_eq_zero_iff x
      have hn : Model.IEEE.isNaN x = true := by simpa [Model.isNaN, henc] using hnan
      simp only [Model.isSNaN, henc, Model.IEEE.isSNaN, hn, Bool.true_and, hq]
    have hpow : 2 ^ fmt.fracWidth = 2 ^ (fmt.fracWidth - 1) * 2 := by
      rw [← Nat.pow_succ]; congr 1; omega
    have hdiv : Model.fracField x / 2 ^ (fmt.fracWidth - 1) < 2 :=
      Nat.div_lt_of_lt_mul (by rw [← hpow]; exact hf)
    have hsplit := Nat.div_add_mod (Model.fracField x) (2 ^ (fmt.fracWidth - 1))
    have hmodlt := Nat.mod_lt (Model.fracField x) (Nat.two_pow_pos (fmt.fracWidth - 1))
    rw [hsnan, Nat.testBit_eq_decide_div_mod_eq]
    simp only [ite_true]
    by_cases hbit : Model.fracField x / 2 ^ (fmt.fracWidth - 1) % 2 = 1
    · have h1 : Model.fracField x / 2 ^ (fmt.fracWidth - 1) = 1 := by
        generalize Model.fracField x / 2 ^ (fmt.fracWidth - 1) = q at hbit hdiv
        omega
      have hlog : (Model.fracField x).log2 = fmt.fracWidth - 1 := by
        rw [Nat.log2_eq_iff (by intro h; rw [h] at h1; simp at h1)]
        constructor
        · rw [h1] at hsplit; omega
        · rw [show fmt.fracWidth - 1 + 1 = fmt.fracWidth by omega]; exact hf
      simp only [hbit, decide_true, Bool.not_true, Bool.false_eq_true, ite_false, hlog]
      rw [h1] at hsplit
      omega
    · have h0 : Model.fracField x / 2 ^ (fmt.fracWidth - 1) = 0 := by
        generalize Model.fracField x / 2 ^ (fmt.fracWidth - 1) = q at hbit hdiv
        omega
      simp only [hbit, decide_false, Bool.not_false, ite_true]
      rw [h0] at hsplit
      omega
  all_goals rfl

/-- The binary64 cast of a NaN in any mode is the quiet NaN of the same sign carrying the payload
of `nanPayload_eq`. -/
private theorem toNatBits_cast_binary64_of_isNaN {fmt : FloatFormat} (hfrac : fmt.fracWidth ≤ 52)
    (x : Model fmt) (hnan : Model.isNaN x = true) (mode : Model.IEEERoundingMode) :
    (Model.cast fmt .binary64 x mode).toNatBits =
      2 ^ 63 * (Model.signBit x).toNat + 0x7FF8000000000000 +
        if fmt.encoding = .ieee then Model.fracField x % 2 ^ (fmt.fracWidth - 1) else 0 := by
  rw [Model.cast_of_isNaN x hnan mode, ← nanPayload_eq x hnan]
  have hlt : Model.nanPayload x < 2 ^ 51 := by
    rw [nanPayload_eq x hnan]
    split
    · calc Model.fracField x % 2 ^ (fmt.fracWidth - 1) < 2 ^ (fmt.fracWidth - 1) :=
            Nat.mod_lt _ (Nat.two_pow_pos _)
        _ ≤ 2 ^ 51 := Nat.pow_le_pow_right (by decide) (by omega)
    · exact Nat.two_pow_pos _
  unfold Model.propagatedNaN
  simp only [show FloatFormat.binary64.encoding = .ieee from rfl,
    show FloatFormat.binary64.fracWidth - 1 = 51 from rfl,
    show FloatFormat.binary64.expAllOnesNat = 2047 from rfl, ite_eq_left hlt]
  rw [toNatBits_ofFields_binary64 _ _ _ (by decide) (by omega)]
  omega

/-! ## The bit-level widening -/

/--
The binary64 word of the `fmt` value with interchange word `bits`, computed with `UInt64` shifts,
masks, additions, and `log2Word` on subnormal fractions. `toWord64_eq_widenToWord64` proves it
equal to FloatLib's cast into binary64 in every rounding mode when `fracWidth ≤ 52`,
`exponentBias + fracWidth ≤ 1075`, and `maxFiniteExpField ≤ exponentBias + 1023`.
-/
@[inline] def widenToWord64 (fmt : FloatFormat) (bits : UInt64) : UInt64 :=
  let ew := UInt64.ofNat fmt.expWidth
  let w := UInt64.ofNat fmt.fracWidth
  let bias := UInt64.ofNat fmt.exponentBias
  let sign := bits >>> (ew + w) <<< 63
  let e := (bits >>> w) &&& ((1 <<< ew) - 1)
  let f := bits &&& ((1 <<< w) - 1)
  let nan := sign + 0x7FF8000000000000
  let finite :=
    if e = 0 then
      if f = 0 then sign
      -- Shifting the leading one of `f` to bit 52 adds one to the exponent field.
      else if bias + w ≤ log2Word f + 1023 then
        sign + ((log2Word f + 1023 - bias - w) <<< 52) + (f <<< (52 - log2Word f))
      else sign + (f <<< (1075 - bias - w))
    else if bias ≤ e + 1022 then sign + ((e + 1023 - bias) <<< 52) + (f <<< (52 - w))
    else sign + (((1 <<< w) + f) <<< (e + 1074 - bias - w))
  match fmt.encoding with
  | .ieee =>
      if e = (1 <<< ew) - 1 then
        if f = 0 then sign + 0x7FF0000000000000 else nan + (f &&& ((1 <<< (w - 1)) - 1))
      else finite
  | .finiteMaxNaN => if e = (1 <<< ew) - 1 ∧ f = (1 <<< w) - 1 then nan else finite
  | .finiteUnsignedZero => if bits = 1 <<< (ew + w) then nan else finite
  | .finite => finite

/-- The conditions of `toWord64_eq_widenToWord64` give `expWidth ≤ 11`. -/
private theorem expWidth_le_eleven {fmt : FloatFormat}
    (hmin : fmt.exponentBias + fmt.fracWidth ≤ 1075)
    (hmax : fmt.maxFiniteExpField ≤ fmt.exponentBias + 1023) : fmt.expWidth ≤ 11 := by
  by_contra hlt
  have h12 : 2 ^ 12 ≤ 2 ^ fmt.expWidth := Nat.pow_le_pow_right (by decide) (by omega)
  have hmf : 2 ^ fmt.expWidth - 2 ≤ fmt.maxFiniteExpField := by
    unfold FloatFormat.maxFiniteExpField FloatFormat.Encoding.maxFiniteExponent
    split <;> omega
  omega

/--
For a format with `fracWidth ≤ 52`, `exponentBias + fracWidth ≤ 1075`, and
`maxFiniteExpField ≤ exponentBias + 1023`, FloatLib's binary64 word of `x` cast in any `mode` is
`widenToWord64` of the interchange word of `x`.
-/
theorem toWord64_eq_widenToWord64 {fmt : FloatFormat} (hfrac : fmt.fracWidth ≤ 52)
    (hmin : fmt.exponentBias + fmt.fracWidth ≤ 1075)
    (hmax : fmt.maxFiniteExpField ≤ fmt.exponentBias + 1023) (x : Model fmt)
    (mode : Model.IEEERoundingMode) :
    ExecFloat.Binary.Internal.Native.toWord64 x mode =
      widenToWord64 fmt (UInt64.ofNat x.toNatBits) := by
  have hw0 := fmt.fracWidth_pos
  have hew : fmt.expWidth ≤ 11 := expWidth_le_eleven hmin hmax
  have he := Model.expField_lt_pow2 x
  have hf := Model.fracField_lt_pow2 x
  have hs : (Model.signBit x).toNat ≤ 1 := Bool.toNat_le _
  have hlow : 2 ^ fmt.fracWidth * Model.expField x + Model.fracField x <
      2 ^ (fmt.expWidth + fmt.fracWidth) := by
    calc 2 ^ fmt.fracWidth * Model.expField x + Model.fracField x <
          2 ^ fmt.fracWidth * (Model.expField x + 1) := by rw [Nat.mul_succ]; omega
      _ ≤ 2 ^ fmt.fracWidth * 2 ^ fmt.expWidth := Nat.mul_le_mul_left _ he
      _ = 2 ^ (fmt.expWidth + fmt.fracWidth) := by rw [← Nat.pow_add, Nat.add_comm]
  have hsplit : x.toNatBits = 2 ^ fmt.fracWidth *
      (2 ^ fmt.expWidth * (Model.signBit x).toNat + Model.expField x) + Model.fracField x := by
    rw [toNatBits_eq_fields x, Nat.pow_add, Nat.mul_add, ← Nat.mul_assoc,
      Nat.mul_comm (2 ^ fmt.fracWidth) (2 ^ fmt.expWidth), Nat.add_assoc]
  have hn : x.toNatBits < 2 ^ 64 := lt_of_lt_of_le (Model.toNatBits_lt_two_pow x)
    (Nat.pow_le_pow_right (by decide) (by unfold FloatFormat.bitWidth; omega))
  have hbits : (UInt64.ofNat x.toNatBits).toNat = x.toNatBits := UInt64.toNat_ofNat_of_lt' hn
  have hpow : ∀ k : UInt64, k.toNat < 64 → ((1 : UInt64) <<< k).toNat = 2 ^ k.toNat := by
    intro k hk
    rw [UInt64.toNat_shiftLeft, UInt64.toNat_one, Nat.mod_eq_of_lt hk, Nat.one_shiftLeft,
      Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by decide) hk)]
  have hmask : ∀ k : UInt64, k.toNat < 64 → ((1 : UInt64) <<< k - 1).toNat = 2 ^ k.toNat - 1 := by
    intro k hk
    rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr (by
      rw [hpow k hk, UInt64.toNat_one]; exact Nat.one_le_two_pow)), hpow k hk, UInt64.toNat_one]
  have hshl : ∀ a b : UInt64, a.toNat <<< b.toNat < 2 ^ 64 →
      (a <<< b).toNat = a.toNat <<< b.toNat := fun a b h => by
    simpa only [UInt64.ofNat_toNat] using shiftLeft_toNat a b.toNat h
  have hc : (Model.cast fmt .binary64 x mode).toNatBits < UInt64.size :=
    Model.toNatBits_lt_two_pow _
  simp only [ExecFloat.Binary.Internal.Native.toWord64, ExecFloat.Binary.Internal.Native.cast,
    Model.Internal.Status.cast_eq, Model.value_cast]
  unfold widenToWord64
  extract_lets ew w bias sign e f nan finite
  have hewU : ew.toNat = fmt.expWidth := UInt64.toNat_ofNat_of_lt' (by unfold UInt64.size; omega)
  have hwU : w.toNat = fmt.fracWidth := UInt64.toNat_ofNat_of_lt' (by unfold UInt64.size; omega)
  have hbU : bias.toNat = fmt.exponentBias :=
    UInt64.toNat_ofNat_of_lt' (by unfold UInt64.size; omega)
  have hS : sign.toNat = 2 ^ 63 * (Model.signBit x).toNat := by
    show ((UInt64.ofNat x.toNatBits >>> (ew + w)) <<< 63).toNat = _
    have hsh : (ew + w).toNat % 64 = fmt.expWidth + fmt.fracWidth := by
      rw [UInt64.toNat_add, hewU, hwU]; omega
    have hdiv : x.toNatBits / 2 ^ (fmt.expWidth + fmt.fracWidth) = (Model.signBit x).toNat := by
      rw [toNatBits_eq_fields x, Nat.mul_add_div (Nat.two_pow_pos _), Nat.div_eq_of_lt hlow,
        Nat.add_zero]
    rw [UInt64.toNat_shiftLeft, UInt64.toNat_shiftRight, hsh, hbits, Nat.shiftRight_eq_div_pow,
      hdiv]
    cases Model.signBit x <;> decide
  have hE : e.toNat = Model.expField x := by
    show ((UInt64.ofNat x.toNatBits >>> w) &&& ((1 <<< ew) - 1)).toNat = _
    rw [UInt64.toNat_and, hmask ew (by rw [hewU]; omega), hewU, Nat.and_two_pow_sub_one_eq_mod,
      UInt64.toNat_shiftRight, hwU, hbits, Nat.mod_eq_of_lt (show fmt.fracWidth < 64 by omega),
      Nat.shiftRight_eq_div_pow, hsplit, Nat.mul_add_div (Nat.two_pow_pos _),
      Nat.div_eq_of_lt hf, Nat.add_zero, Nat.mul_add_mod, Nat.mod_eq_of_lt he]
  have hF : f.toNat = Model.fracField x := by
    show (UInt64.ofNat x.toNatBits &&& ((1 <<< w) - 1)).toNat = _
    rw [UInt64.toNat_and, hmask w (by rw [hwU]; omega), hwU, Nat.and_two_pow_sub_one_eq_mod,
      hbits, hsplit, Nat.mul_add_mod, Nat.mod_eq_of_lt hf]
  have hnanU : nan.toNat = 2 ^ 63 * (Model.signBit x).toNat + 0x7FF8000000000000 := by
    show (sign + 0x7FF8000000000000).toNat = _
    rw [UInt64.toNat_add, hS]
    simp only [UInt64.reduceToNat]
    omega
  have hfin : Model.isFinite x = true →
      finite.toNat = (Model.cast fmt .binary64 x mode).toNatBits := by
    intro hx
    have hemax := Model.expField_le_maxFiniteExpField_of_isFinite x hx
    rw [toNatBits_cast_binary64_of_isFinite hfrac hmin hmax x hx mode]
    simp only [finite]
    by_cases he0 : Model.expField x = 0
    · rw [ite_eq_left (show e = 0 from UInt64.toNat_inj.mp (by rw [hE, he0]; rfl)),
        ite_eq_left he0]
      by_cases hf0 : Model.fracField x = 0
      · rw [ite_eq_left (show f = 0 from UInt64.toNat_inj.mp (by rw [hF, hf0]; rfl)),
          ite_eq_left hf0, hS]
      · rw [ite_eq_right (show ¬f = 0 from fun h => hf0 (by rw [← hF, h]; rfl)),
          ite_eq_right hf0]
        have hlog : (log2Word f).toNat = (Model.fracField x).log2 := by rw [log2Word_toNat, hF]
        have hL : (Model.fracField x).log2 < fmt.fracWidth := (Nat.log2_lt hf0).mpr hf
        by_cases hnorm : fmt.exponentBias + fmt.fracWidth ≤ (Model.fracField x).log2 + 1023
        · rw [ite_eq_left (show bias + w ≤ log2Word f + 1023 by
              rw [UInt64.le_iff_toNat_le, UInt64.toNat_add, UInt64.toNat_add, hbU, hwU, hlog]
              simp only [UInt64.reduceToNat]
              omega),
            ite_eq_left hnorm]
          have hhi : Model.fracField x <<< (52 - (Model.fracField x).log2) < 2 ^ 53 :=
            shiftLeft_sub_log2_lt_two_pow 52 _ (by omega)
          have hamt : (52 - log2Word f).toNat = 52 - (Model.fracField x).log2 := by
            rw [UInt64.toNat_sub, hlog]
            simp only [UInt64.reduceToNat]
            omega
          have hT : (f <<< (52 - log2Word f)).toNat =
              Model.fracField x <<< (52 - (Model.fracField x).log2) := by
            rw [hshl f (52 - log2Word f) (by rw [hamt, hF]; exact lt_of_lt_of_le hhi (by decide)),
              hamt, hF]
          have hA : (log2Word f + 1023 - bias - w).toNat =
              (Model.fracField x).log2 + 1023 - fmt.exponentBias - fmt.fracWidth := by
            rw [UInt64.toNat_sub, UInt64.toNat_sub, UInt64.toNat_add, hlog, hbU, hwU]
            simp only [UInt64.reduceToNat]
            omega
          have hX : ((log2Word f + 1023 - bias - w) <<< 52).toNat =
              2 ^ 52 * ((Model.fracField x).log2 + 1023 - fmt.exponentBias - fmt.fracWidth) := by
            rw [hshl (log2Word f + 1023 - bias - w) 52 (by
                rw [hA, Nat.shiftLeft_eq]; simp only [UInt64.reduceToNat]; omega),
              hA, Nat.shiftLeft_eq]
            simp only [UInt64.reduceToNat]
            omega
          rw [UInt64.toNat_add, UInt64.toNat_add, hS, hX, hT]
          generalize Model.fracField x <<< (52 - (Model.fracField x).log2) = T at hhi ⊢
          omega
        · rw [ite_eq_right (show ¬bias + w ≤ log2Word f + 1023 by
              rw [UInt64.le_iff_toNat_le, UInt64.toNat_add, UInt64.toNat_add, hbU, hwU, hlog]
              simp only [UInt64.reduceToNat]
              omega),
            ite_eq_right hnorm]
          have hamt : (1075 - bias - w).toNat = 1075 - fmt.exponentBias - fmt.fracWidth := by
            rw [UInt64.toNat_sub, UInt64.toNat_sub, hbU, hwU]
            simp only [UInt64.reduceToNat]
            omega
          have hlt : Model.fracField x <<< (1075 - fmt.exponentBias - fmt.fracWidth) < 2 ^ 52 :=
            shiftLeft_lt_two_pow_of_log2_add_le _ _ _ (by omega)
          have hT : (f <<< (1075 - bias - w)).toNat =
              Model.fracField x <<< (1075 - fmt.exponentBias - fmt.fracWidth) := by
            rw [hshl f (1075 - bias - w) (by rw [hamt, hF]; exact lt_of_lt_of_le hlt (by decide)),
              hamt, hF]
          rw [UInt64.toNat_add, hS, hT]
          generalize Model.fracField x <<< (1075 - fmt.exponentBias - fmt.fracWidth) = T at hlt ⊢
          omega
    · rw [ite_eq_right (show ¬e = 0 from fun h => he0 (by rw [← hE, h]; rfl)), ite_eq_right he0]
      by_cases hnorm : fmt.exponentBias ≤ Model.expField x + 1022
      · rw [ite_eq_left (show bias ≤ e + 1022 by
            rw [UInt64.le_iff_toNat_le, UInt64.toNat_add, hbU, hE]
            simp only [UInt64.reduceToNat]
            omega),
          ite_eq_left hnorm]
        have hA : (e + 1023 - bias).toNat = Model.expField x + 1023 - fmt.exponentBias := by
          rw [UInt64.toNat_sub, UInt64.toNat_add, hE, hbU]
          simp only [UInt64.reduceToNat]
          omega
        have hX : ((e + 1023 - bias) <<< 52).toNat =
            2 ^ 52 * (Model.expField x + 1023 - fmt.exponentBias) := by
          rw [hshl (e + 1023 - bias) 52 (by
              rw [hA, Nat.shiftLeft_eq]; simp only [UInt64.reduceToNat]; omega),
            hA, Nat.shiftLeft_eq]
          simp only [UInt64.reduceToNat]
          omega
        have hamt : (52 - w).toNat = 52 - fmt.fracWidth := by
          rw [UInt64.toNat_sub, hwU]
          simp only [UInt64.reduceToNat]
          omega
        have hlt : Model.fracField x <<< (52 - fmt.fracWidth) < 2 ^ 52 := by
          have := Nat.shiftLeft_lt (m := 52 - fmt.fracWidth) hf
          rwa [show fmt.fracWidth + (52 - fmt.fracWidth) = 52 by omega] at this
        have hT : (f <<< (52 - w)).toNat = Model.fracField x <<< (52 - fmt.fracWidth) := by
          rw [hshl f (52 - w) (by rw [hamt, hF]; exact lt_of_lt_of_le hlt (by decide)), hamt, hF]
        rw [UInt64.toNat_add, UInt64.toNat_add, hS, hX, hT]
        generalize Model.fracField x <<< (52 - fmt.fracWidth) = T at hlt ⊢
        omega
      · rw [ite_eq_right (show ¬bias ≤ e + 1022 by
            rw [UInt64.le_iff_toNat_le, UInt64.toNat_add, hbU, hE]
            simp only [UInt64.reduceToNat]
            omega),
          ite_eq_right hnorm]
        have hamt : (e + 1074 - bias - w).toNat =
            Model.expField x + 1074 - fmt.exponentBias - fmt.fracWidth := by
          rw [UInt64.toNat_sub, UInt64.toNat_sub, UInt64.toNat_add, hE, hbU, hwU]
          simp only [UInt64.reduceToNat]
          omega
        have hM : ((1 <<< w) + f).toNat = 2 ^ fmt.fracWidth + Model.fracField x := by
          rw [UInt64.toNat_add, hpow w (by rw [hwU]; omega), hwU, hF]
          have : 2 ^ fmt.fracWidth ≤ 2 ^ 52 := Nat.pow_le_pow_right (by decide) hfrac
          omega
        have hlt : (2 ^ fmt.fracWidth + Model.fracField x) <<<
            (Model.expField x + 1074 - fmt.exponentBias - fmt.fracWidth) < 2 ^ 52 := by
          have h2 : 2 ^ fmt.fracWidth + Model.fracField x < 2 ^ (fmt.fracWidth + 1) := by
            rw [Nat.pow_succ]; omega
          have := Nat.shiftLeft_lt
            (m := Model.expField x + 1074 - fmt.exponentBias - fmt.fracWidth) h2
          exact lt_of_lt_of_le this (Nat.pow_le_pow_right (by decide) (by omega))
        have hT : (((1 <<< w) + f) <<< (e + 1074 - bias - w)).toNat =
            (2 ^ fmt.fracWidth + Model.fracField x) <<<
              (Model.expField x + 1074 - fmt.exponentBias - fmt.fracWidth) := by
          rw [hshl ((1 <<< w) + f) (e + 1074 - bias - w)
            (by rw [hamt, hM]; exact lt_of_lt_of_le hlt (by decide)), hamt, hM]
        rw [UInt64.toNat_add, hS, hT]
        generalize (2 ^ fmt.fracWidth + Model.fracField x) <<<
          (Model.expField x + 1074 - fmt.exponentBias - fmt.fracWidth) = T at hlt ⊢
        omega
  have hexp : ((1 : UInt64) <<< ew - 1).toNat = fmt.expAllOnesNat := by
    rw [hmask ew (by rw [hewU]; omega), hewU]; rfl
  have hfracMask : ((1 : UInt64) <<< w - 1).toNat = fmt.fracMaskNat := by
    rw [hmask w (by rw [hwU]; omega), hwU]; rfl
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_ofNat_of_lt' hc]
  cases henc : fmt.encoding
  · simp only []
    by_cases hE1 : Model.expField x = fmt.expAllOnesNat
    · rw [ite_eq_left (UInt64.toNat_inj.mp (by rw [hE, hexp, hE1]))]
      by_cases hF0 : Model.fracField x = 0
      · rw [ite_eq_left (show f = 0 from UInt64.toNat_inj.mp (by rw [hF, hF0]; rfl))]
        have hinf : Model.isInf x = true := by
          simp [Model.isInf, henc, Model.IEEE.isInf, hE1, hF0]
        rw [toNatBits_cast_binary64_of_isInf x hinf mode, UInt64.toNat_add, hS]
        simp only [UInt64.reduceToNat]
        omega
      · rw [ite_eq_right (show ¬f = 0 from fun h => hF0 (by rw [← hF, h]; rfl))]
        have hnan : Model.isNaN x = true := by
          simp [Model.isNaN, henc, Model.IEEE.isNaN, hE1, hF0]
        have hw1 : (w - 1).toNat = fmt.fracWidth - 1 := by
          rw [UInt64.toNat_sub_of_le _ _ (UInt64.le_iff_toNat_le.mpr (by
            rw [hwU, UInt64.toNat_one]; omega)), hwU, UInt64.toNat_one]
        have hq : (f &&& ((1 <<< (w - 1)) - 1)).toNat =
            Model.fracField x % 2 ^ (fmt.fracWidth - 1) := by
          rw [UInt64.toNat_and, hmask (w - 1) (by rw [hw1]; omega), hw1, hF,
            Nat.and_two_pow_sub_one_eq_mod]
        have hp : Model.fracField x % 2 ^ (fmt.fracWidth - 1) < 2 ^ 51 :=
          lt_of_lt_of_le (Nat.mod_lt _ (Nat.two_pow_pos _))
            (Nat.pow_le_pow_right (by decide) (by omega))
        rw [toNatBits_cast_binary64_of_isNaN hfrac x hnan mode, ite_eq_left henc,
          UInt64.toNat_add, hnanU, hq]
        generalize Model.fracField x % 2 ^ (fmt.fracWidth - 1) = p at hp ⊢
        omega
    · rw [ite_eq_right (fun h => hE1 (by rw [← hE, h, hexp]))]
      exact (hfin (by simp [Model.isFinite, henc, Model.IEEE.isFinite, hE1])).symm
  · simp only []
    by_cases hN : Model.expField x = fmt.expAllOnesNat ∧ Model.fracField x = fmt.fracMaskNat
    · rw [ite_eq_left ⟨UInt64.toNat_inj.mp (by rw [hE, hexp, hN.1]),
        UInt64.toNat_inj.mp (by rw [hF, hfracMask, hN.2])⟩]
      have hnan : Model.isNaN x = true := by simp [Model.isNaN, henc, hN.1, hN.2]
      rw [toNatBits_cast_binary64_of_isNaN hfrac x hnan mode, ite_eq_right (by simp [henc]),
        hnanU]
    · rw [ite_eq_right (fun h => hN ⟨by rw [← hE, h.1, hexp], by rw [← hF, h.2, hfracMask]⟩)]
      refine (hfin ?_).symm
      simp only [Model.isFinite, Model.isNaN, henc]
      by_contra hc
      exact hN (by simpa using hc)
  · simp only []
    have hiff : x.bits = FloatFormat.signMask fmt ↔
        x.toNatBits = 2 ^ (fmt.expWidth + fmt.fracWidth) := by
      rw [← BitVec.toNat_inj, FloatFormat.signMask_toNat,
        show fmt.bitWidth - 1 = fmt.expWidth + fmt.fracWidth by unfold FloatFormat.bitWidth; omega]
      rfl
    have hsm : ((1 : UInt64) <<< (ew + w)).toNat = 2 ^ (fmt.expWidth + fmt.fracWidth) := by
      rw [hpow (ew + w) (by rw [UInt64.toNat_add, hewU, hwU]; omega), UInt64.toNat_add, hewU,
        hwU, Nat.mod_eq_of_lt (by omega)]
    by_cases hN : x.toNatBits = 2 ^ (fmt.expWidth + fmt.fracWidth)
    · rw [ite_eq_left (UInt64.toNat_inj.mp (by rw [hbits, hsm, hN]))]
      have hnan : Model.isNaN x = true := by simp [Model.isNaN, henc, hiff.mpr hN]
      rw [toNatBits_cast_binary64_of_isNaN hfrac x hnan mode, ite_eq_right (by simp [henc]),
        hnanU]
    · rw [ite_eq_right (fun h => hN (by rw [← hbits, h, hsm]))]
      refine (hfin ?_).symm
      simp only [Model.isFinite, Model.isNaN, henc]
      simpa [hiff] using hN
  · exact (hfin (by simp [Model.isFinite, henc])).symm

/--
`true` exactly when `fmt` meets the conditions of `toWord64_eq_widenToWord64`
(`widensToWord64_iff`). After the test `expWidth ≤ 11` it computes `2 ^ expWidth` as a `UInt64`
shift.
-/
@[inline] def widensToWord64 (fmt : FloatFormat) : Bool :=
  fmt.expWidth ≤ 11 && fmt.fracWidth ≤ 52 && fmt.exponentBias + fmt.fracWidth ≤ 1075 &&
    ((1 : UInt64) <<< UInt64.ofNat fmt.expWidth).toNat - (if fmt.encoding = .ieee then 2 else 1) ≤
      fmt.exponentBias + 1023

/--
`widensToWord64 fmt` holds exactly when `fmt` meets the conditions of
`toWord64_eq_widenToWord64`.
-/
theorem widensToWord64_iff {fmt : FloatFormat} :
    widensToWord64 fmt = true ↔ fmt.fracWidth ≤ 52 ∧ fmt.exponentBias + fmt.fracWidth ≤ 1075 ∧
      fmt.maxFiniteExpField ≤ fmt.exponentBias + 1023 := by
  have hpow : fmt.expWidth ≤ 11 →
      ((1 : UInt64) <<< UInt64.ofNat fmt.expWidth).toNat -
          (if fmt.encoding = .ieee then 2 else 1) = fmt.maxFiniteExpField := by
    intro h
    have h64 : fmt.expWidth < 64 := by omega
    have hk : (UInt64.ofNat fmt.expWidth).toNat = fmt.expWidth :=
      UInt64.toNat_ofNat_of_lt' (lt_of_lt_of_le h64 (by decide))
    rw [UInt64.toNat_shiftLeft, UInt64.toNat_one, hk, Nat.mod_eq_of_lt h64, Nat.one_shiftLeft,
      Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by decide) h64)]
    unfold FloatFormat.maxFiniteExpField FloatFormat.Encoding.maxFiniteExponent
    generalize fmt.encoding = enc
    cases enc <;> rfl
  unfold widensToWord64
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨⟨⟨h11, hf⟩, hb⟩, hm⟩
    exact ⟨hf, hb, hpow h11 ▸ hm⟩
  · rintro ⟨hf, hb, hm⟩
    have h11 := expWidth_le_eleven hb hm
    exact ⟨⟨⟨h11, hf⟩, hb⟩, (hpow h11).symm ▸ hm⟩

/-! ## Readback to `Float` -/

section Native

open ExecFloat.Binary

variable {format : FloatFormat} {plan : Configured.StoragePlan format}

local notation "Value" =>
  ExecFloat.Binary format.expWidth format.fracWidth format.encoding format.exponentBias
    format.expWidth_ge_two format.fracWidth_pos format.exponentBias_pos
    format.exponentBias_le_maxFinite plan

/--
`ExecFloat.Binary.toFloat` with the bit path: a binary64 value in a `word64` plan crosses as its
word, any other value whose format passes `widensToWord64` crosses as `Float.ofBits` of its
`widenToWord64` word, and the remaining values take FloatLib's cast. The `word64` plan tests
`FloatFormat.IsBinary64` first, which compares four fields, so the compiler folds the test away
wherever it knows the format.
-/
@[inline] def widenToFloat (value : Value) (mode : Model.IEEERoundingMode := .nearest) :
    Float :=
  match plan, value with
  | .word64 _, value =>
      if FloatFormat.IsBinary64 format then Float.ofBits value.raw.1
      else if widensToWord64 format then Float.ofBits (widenToWord64 format value.raw.1)
      else Float.ofBits (Internal.Native.toWord64 (toModel value) mode)
  | plan, value =>
      if widensToWord64 format then
        match plan, value with
        | .byte _, value => Float.ofBits (widenToWord64 format value.raw.1.toUInt64)
        | .word16 _, value => Float.ofBits (widenToWord64 format value.raw.1.toUInt64)
        | .word32 _, value => Float.ofBits (widenToWord64 format value.raw.1.toUInt64)
        | _, value => Float.ofBits (widenToWord64 format (UInt64.ofNat (toModel value).toNatBits))
      else Float.ofBits (Internal.Native.toWord64 (toModel value) mode)

/--
`ExecFloat.Binary.toFloat` equals `widenToFloat` for every format, storage plan, value,
and rounding mode. As a `csimp` lemma it makes compiled code in importing modules run the bit path.
-/
@[csimp] theorem toFloat_eq_widenToFloat :
    @ExecFloat.Binary.toFloat = @widenToFloat := by
  funext format plan value mode
  unfold ExecFloat.Binary.toFloat widenToFloat
  cases plan with
  | word64 h =>
      have hbits : (toModel value).toNatBits = value.raw.1.toNat := rfl
      dsimp only
      by_cases hb : format = FloatFormat.binary64
      · have h64 : FloatFormat.IsBinary64 format := hb ▸ FloatFormat.isBinary64_binary64
        rw [ite_eq_left hb, ite_eq_left h64]
      · have h64 : ¬FloatFormat.IsBinary64 format :=
          fun h => hb (FloatFormat.eq_binary64_of_isBinary64 h)
        rw [ite_eq_right hb, ite_eq_right h64]
        split
        next hcov =>
          obtain ⟨hf, hmin, hmax⟩ := widensToWord64_iff.1 hcov
          rw [toWord64_eq_widenToWord64 hf hmin hmax, hbits, UInt64.ofNat_toNat]
        next => rfl
  | byte h =>
      have hbits : (toModel value).toNatBits = value.raw.1.toNat % 2 ^ format.bitWidth := rfl
      dsimp only
      split
      next hcov =>
        obtain ⟨hf, hmin, hmax⟩ := widensToWord64_iff.1 hcov
        rw [toWord64_eq_widenToWord64 hf hmin hmax, hbits, Nat.mod_eq_of_lt value.raw.2,
          UInt64.ofNat_uInt8ToNat]
      next => rfl
  | word16 h =>
      have hbits : (toModel value).toNatBits = value.raw.1.toNat := rfl
      dsimp only
      split
      next hcov =>
        obtain ⟨hf, hmin, hmax⟩ := widensToWord64_iff.1 hcov
        rw [toWord64_eq_widenToWord64 hf hmin hmax, hbits, UInt64.ofNat_uInt16ToNat]
      next => rfl
  | word32 h =>
      have hbits : (toModel value).toNatBits = value.raw.1.toNat := rfl
      dsimp only
      split
      next hcov =>
        obtain ⟨hf, hmin, hmax⟩ := widensToWord64_iff.1 hcov
        rw [toWord64_eq_widenToWord64 hf hmin hmax, hbits, UInt64.ofNat_uInt32ToNat]
      next => rfl
  | pair h₁ h₂ | wide | limbs h =>
      dsimp only
      split
      next hcov =>
        obtain ⟨hf, hmin, hmax⟩ := widensToWord64_iff.1 hcov
        exact congrArg Float.ofBits (toWord64_eq_widenToWord64 hf hmin hmax _ mode)
      next => rfl

end Native

end TorchLean.Numeric
