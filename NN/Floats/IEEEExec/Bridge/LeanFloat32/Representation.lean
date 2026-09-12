/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Floats.IEEEExec.Encoding.Negation
import Mathlib.Tactic.IntervalCases
public import NN.Floats.IEEEExec.Exec32.Dyadic

/-!
# Lean Float32 and IEEE32Exec: representation

This file relates the canonical bit representation used by Lean's `Float32.Model` to TorchLean's
independent `IEEE32Exec` decoder. The central theorem, `toDyadic_modelToIEEE32Exec`, says that both
decoders recover the same sign, integer mantissa, and exponent from every finite model value.

No arithmetic operation is involved here. Keeping representation agreement separate prevents the
addition, multiplication, division, and square-root proofs from repeating bit-field arguments.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open IEEE32Exec

namespace Float32Bridge

/-- Read a canonical `Float32.Model` value as TorchLean's executable binary32 representation. -/
@[inline] def modelToIEEE32Exec (x : Float32.Model) : IEEE32Exec :=
  IEEE32Exec.ofBits x.toBits

/-- Read a Lean `Float32` through its logical model and expose the resulting binary32 bits. -/
@[inline] def toIEEE32Exec (x : _root_.Float32) : IEEE32Exec :=
  modelToIEEE32Exec x.toModel

/-- Construct a Lean `Float32` from raw `IEEE32Exec` bits. NaN payloads are canonicalized. -/
@[inline] def ofIEEE32Exec (x : IEEE32Exec) : _root_.Float32 :=
  Float32.ofBits x.bits

/-- Canonicalize the NaN representation of an executable binary32 value. -/
@[inline] def canonicalize (x : IEEE32Exec) : IEEE32Exec :=
  modelToIEEE32Exec (Float32.Model.ofBits x.bits)

/-- Convert Lean's unpacked sign to the Boolean sign convention used by `IEEE32Exec`. -/
def signToBool : Float.Model.UnpackedFloat.Sign → Bool
  | .negative => true
  | .positive => false

/-- Negating an unpacked sign negates its Boolean sign bit. -/
@[simp] theorem signToBool_neg (sign : Float.Model.UnpackedFloat.Sign) :
    signToBool (-sign) = !signToBool sign := by
  cases sign <;> rfl

/-- Masking a zero-extended bit-vector to its original width leaves it unchanged. -/
private theorem setWidth32_and_lowMask {width : Nat} (hwidth : width ≤ 32)
    (x : BitVec width) :
    x.setWidth 32 &&& BitVec.ofNat 32 (2 ^ width - 1) = x.setWidth 32 := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_and, BitVec.toNat_setWidth, BitVec.toNat_ofNat]
  have hpow : 2 ^ width ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) hwidth
  have hx : x.toNat < 2 ^ 32 := x.isLt.trans_le hpow
  have hmask : 2 ^ width - 1 < 2 ^ 32 := by grind
  rw [Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hmask]
  exact Nat.and_two_pow_sub_one_of_lt_two_pow x.isLt

/-- Lean's packed `notANumber` is TorchLean's canonical binary32 NaN. -/
@[simp] theorem modelToIEEE32Exec_pack_notANumber :
    modelToIEEE32Exec (Float32.Model.pack .notANumber) =
      IEEE32Exec.canonicalNaN := by
  rfl

/-- Packing an unpacked infinity preserves its sign in `IEEE32Exec`. -/
@[simp] theorem modelToIEEE32Exec_pack_infinity
    (sign : Float.Model.UnpackedFloat.Sign) :
    modelToIEEE32Exec (Float32.Model.pack (.infinity sign)) =
      if signToBool sign then IEEE32Exec.negInf else IEEE32Exec.posInf := by
  cases sign <;> rfl

/-- Packing an unpacked zero preserves its sign in `IEEE32Exec`. -/
@[simp] theorem modelToIEEE32Exec_pack_zero
    (sign : Float.Model.UnpackedFloat.Sign) :
    modelToIEEE32Exec (Float32.Model.pack (.zero sign)) =
      if signToBool sign then IEEE32Exec.negZero else IEEE32Exec.posZero := by
  cases sign <;> rfl

/--
Lean's field concatenation and `IEEE32Exec.mkBits` produce the same binary32 bit pattern.

This isolates the representation-level part of arithmetic agreement: operation proofs may reason
about a sign, biased exponent, and fraction as numbers, then use this theorem to compare the packed
results.
-/
theorem packComponents_eq_mkBits
    (sign : Float.Model.UnpackedFloat.Sign)
    (exponent : BitVec 8) (mantissa : BitVec 23) :
    UInt32.ofBitVec
        (Float.Model.UnpackedFloat.packComponents Float.Model.Format.binary32
          sign exponent mantissa) =
      IEEE32Exec.mkBits (signToBool sign) exponent.toNat mantissa.toNat := by
  rw [← UInt32.toBitVec_inj]
  have packed (signBits : BitVec 1) :
      signBits ++ exponent ++ mantissa =
        (signBits.setWidth 32 <<< 31) |||
          (exponent.setWidth 32 <<< 23) ||| mantissa.setWidth 32 := by
    simpa using
      (BitVec.setWidth_append_append_eq_shiftLeft_setWidth_or
        (b := signBits) (b' := exponent) (b'' := mantissa) (w''' := 32))
  cases sign <;>
    simp [Float.Model.UnpackedFloat.packComponents, signToBool,
      Float.Model.UnpackedFloat.Sign.toBitVec, IEEE32Exec.mkBits,
      IEEE32Exec.expAllOnes, IEEE32Exec.fracMask, UInt32.toBitVec_ofNat',
      BitVec.ofNat_toNat, packed]
  all_goals
    rw [show 255#32 = BitVec.ofNat 32 (2 ^ 8 - 1) by decide,
      show 8388607#32 = BitVec.ofNat 32 (2 ^ 23 - 1) by decide]
    rw [show exponent.setWidth 32 &&& BitVec.ofNat 32 (2 ^ 8 - 1) =
      exponent.setWidth 32 by exact setWidth32_and_lowMask (by decide) exponent]
    rw [show mantissa.setWidth 32 &&& BitVec.ofNat 32 (2 ^ 23 - 1) =
      mantissa.setWidth 32 by exact setWidth32_and_lowMask (by decide) mantissa]

/--
Decode an unpacked Lean float into TorchLean's exact dyadic representation. NaNs and infinities
have no finite dyadic value; signed zero retains its sign.
-/
def unpackedToDyadic? : Float.Model.UnpackedFloat → Option IEEE32Exec.Dyadic
  | .notANumber | .infinity _ => none
  | .zero sign => some { sign := signToBool sign, mant := 0, exp := 0 }
  | .finite sign mantissa exponent _ =>
      some { sign := signToBool sign, mant := mantissa, exp := exponent }

private theorem unpackExponent_toNat (bits : UInt32) :
    (Float.Model.UnpackedFloat.unpackExponent
        (spec := Float.Model.Format.binary32) bits.toBitVec).toNat =
      (IEEE32Exec.expField (IEEE32Exec.ofBits bits)).toNat := by
  simp [Float.Model.UnpackedFloat.unpackExponent, IEEE32Exec.expField, IEEE32Exec.ofBits,
    UInt32.toNat_and, UInt32.toNat_shiftRight]
  rw [show IEEE32Exec.expAllOnes.toNat = 2 ^ 8 - 1 by decide]
  rw [Nat.and_two_pow_sub_one_eq_mod]

private theorem unpackMantissa_toNat (bits : UInt32) :
    (Float.Model.UnpackedFloat.unpackMantissa
        (spec := Float.Model.Format.binary32) bits.toBitVec).toNat =
      (IEEE32Exec.fracField (IEEE32Exec.ofBits bits)).toNat := by
  simp [Float.Model.UnpackedFloat.unpackMantissa, IEEE32Exec.fracField, IEEE32Exec.ofBits,
    UInt32.toNat_and]
  rw [show IEEE32Exec.fracMask.toNat = 2 ^ 23 - 1 by decide]
  rw [Nat.and_two_pow_sub_one_eq_mod]

/-- Flipping the packed sign bit leaves Lean's unpacked exponent unchanged. -/
theorem unpackExponent_xor_sign (bits : UInt32) :
    Float.Model.UnpackedFloat.unpackExponent
        (spec := Float.Model.Format.binary32)
        (bits ^^^ IEEE32Exec.signMask).toBitVec =
      Float.Model.UnpackedFloat.unpackExponent
        (spec := Float.Model.Format.binary32) bits.toBitVec := by
  apply BitVec.eq_of_toNat_eq
  rw [unpackExponent_toNat, unpackExponent_toNat]
  exact congrArg UInt32.toNat (IEEE32Exec.expField_ofBits_xor_signMask bits)

/-- Flipping the packed sign bit leaves Lean's unpacked mantissa unchanged. -/
theorem unpackMantissa_xor_sign (bits : UInt32) :
    Float.Model.UnpackedFloat.unpackMantissa
        (spec := Float.Model.Format.binary32)
        (bits ^^^ IEEE32Exec.signMask).toBitVec =
      Float.Model.UnpackedFloat.unpackMantissa
        (spec := Float.Model.Format.binary32) bits.toBitVec := by
  apply BitVec.eq_of_toNat_eq
  rw [unpackMantissa_toNat, unpackMantissa_toNat]
  exact congrArg UInt32.toNat (IEEE32Exec.fracField_ofBits_xor_signMask bits)

/-- Lean's unpacked exponent is all ones exactly when the executable exponent field is all ones. -/
theorem unpackExponent_allOnes_iff (bits : UInt32) :
    Float.Model.UnpackedFloat.unpackExponent
        (spec := Float.Model.Format.binary32) bits.toBitVec = -1#8 ↔
      IEEE32Exec.expField (IEEE32Exec.ofBits bits) = IEEE32Exec.expAllOnes := by
  constructor
  · intro h
    apply UInt32.toNat_inj.mp
    rw [← unpackExponent_toNat, h]
    decide
  · intro h
    apply BitVec.eq_of_toNat_eq
    rw [unpackExponent_toNat, h]
    decide

/-- Lean's unpacked mantissa is zero exactly when the executable fraction field is zero. -/
theorem unpackMantissa_zero_iff (bits : UInt32) :
    Float.Model.UnpackedFloat.unpackMantissa
        (spec := Float.Model.Format.binary32) bits.toBitVec = 0#23 ↔
      IEEE32Exec.fracField (IEEE32Exec.ofBits bits) = 0 := by
  constructor
  · intro h
    apply UInt32.toNat_inj.mp
    rw [← unpackMantissa_toNat, h]
    rfl
  · intro h
    apply BitVec.eq_of_toNat_eq
    rw [unpackMantissa_toNat, h]
    rfl

private theorem unpackSign_zero_iff (bits : UInt32) :
    Float.Model.UnpackedFloat.unpackSign
        (spec := Float.Model.Format.binary32) bits.toBitVec = 0#1 ↔
      bits &&& IEEE32Exec.signMask = 0 := by
  rw [← UInt32.toBitVec_inj]
  simp [Float.Model.UnpackedFloat.unpackSign, IEEE32Exec.signMask]
  constructor
  · intro h
    have h31 := congrArg (fun x : BitVec 1 => x.getLsbD 0) h
    have hbit : bits.toBitVec.getLsbD 31 = false := by
      simp only [BitVec.getLsbD_extractLsb, Nat.reduceSub, Nat.reduceAdd,
        Nat.zero_add, BitVec.getLsbD_zero] at h31
      exact h31
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    interval_cases i <;> simp_all
  · intro h
    have h31 := congrArg (fun x : BitVec 32 => x.getLsbD 31) h
    have hbit : bits.toBitVec.getLsbD 31 = false := by simpa using h31
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    interval_cases i
    simpa [BitVec.getLsbD_extractLsb]

/-- Flipping the packed sign bit complements Lean's one-bit sign field. -/
theorem unpackSignField_xor_sign (bits : UInt32) :
    Float.Model.UnpackedFloat.unpackSign
        (spec := Float.Model.Format.binary32)
        (bits ^^^ IEEE32Exec.signMask).toBitVec =
      Float.Model.UnpackedFloat.unpackSign
        (spec := Float.Model.Format.binary32) bits.toBitVec ^^^ 1#1 := by
  rw [show (bits ^^^ IEEE32Exec.signMask).toBitVec =
      bits.toBitVec ^^^ IEEE32Exec.signMask.toBitVec by rfl]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  interval_cases i
  simp [Float.Model.UnpackedFloat.unpackSign, IEEE32Exec.signMask]

/-- Flipping the packed sign bit negates the corresponding unpacked sign. -/
theorem unpackSign_xor_sign (bits : UInt32) :
    Float.Model.UnpackedFloat.Sign.ofBitVec
        (Float.Model.UnpackedFloat.unpackSign
          (spec := Float.Model.Format.binary32)
          (bits ^^^ IEEE32Exec.signMask).toBitVec) =
      -Float.Model.UnpackedFloat.Sign.ofBitVec
        (Float.Model.UnpackedFloat.unpackSign
          (spec := Float.Model.Format.binary32) bits.toBitVec) := by
  rw [unpackSignField_xor_sign]
  rcases BitVec.eq_zero_or_eq_one
      (Float.Model.UnpackedFloat.unpackSign
        (spec := Float.Model.Format.binary32) bits.toBitVec) with h | h <;>
    rw [h] <;> rfl

private theorem signToBool_ofBitVec_unpackSign (bits : UInt32) :
    signToBool
        (Float.Model.UnpackedFloat.Sign.ofBitVec
          (Float.Model.UnpackedFloat.unpackSign
            (spec := Float.Model.Format.binary32) bits.toBitVec)) =
      IEEE32Exec.signBit (IEEE32Exec.ofBits bits) := by
  by_cases hs : Float.Model.UnpackedFloat.unpackSign
      (spec := Float.Model.Format.binary32) bits.toBitVec = 0#1
  · have hz : bits &&& IEEE32Exec.signMask = 0 := (unpackSign_zero_iff bits).1 hs
    simp [Float.Model.UnpackedFloat.Sign.ofBitVec, hs, signToBool,
      IEEE32Exec.signBit, IEEE32Exec.ofBits, hz]
  · have hz : bits &&& IEEE32Exec.signMask ≠ 0 := by
      intro hz
      exact hs ((unpackSign_zero_iff bits).2 hz)
    simp [Float.Model.UnpackedFloat.Sign.ofBitVec, hs, signToBool,
      IEEE32Exec.signBit, IEEE32Exec.ofBits, hz]

/-- Repacking the three fields extracted from a raw binary32 word recovers that word. -/
theorem packComponents_unpacked_fields (bits : UInt32) :
    UInt32.ofBitVec
        (Float.Model.UnpackedFloat.packComponents Float.Model.Format.binary32
          (Float.Model.UnpackedFloat.Sign.ofBitVec
            (Float.Model.UnpackedFloat.unpackSign
              (spec := Float.Model.Format.binary32) bits.toBitVec))
          (Float.Model.UnpackedFloat.unpackExponent
            (spec := Float.Model.Format.binary32) bits.toBitVec)
          (Float.Model.UnpackedFloat.unpackMantissa
            (spec := Float.Model.Format.binary32) bits.toBitVec)) = bits := by
  rw [packComponents_eq_mkBits, signToBool_ofBitVec_unpackSign,
    unpackExponent_toNat, unpackMantissa_toNat]
  exact congrArg IEEE32Exec.bits
    (IEEE32Exec.ofBits_mkBits_fields (IEEE32Exec.ofBits bits))

/--
Lean's unpacker and `IEEE32Exec.toDyadic?` recover the same exact finite value from every canonical
`Float32.Model` bit pattern.
-/
theorem toDyadic_modelToIEEE32Exec (a : Float32.Model) :
    IEEE32Exec.toDyadic? (modelToIEEE32Exec a) = unpackedToDyadic? a.unpack := by
  cases a with
  | mk bits valid =>
    simp only [modelToIEEE32Exec, Float32.Model.unpack,
      Float.Model.UnpackedFloat.unpack]
    split <;> rename_i hExpOnes
    · have hExpField : IEEE32Exec.expField (IEEE32Exec.ofBits bits) =
          IEEE32Exec.expAllOnes := by
        apply UInt32.toNat_inj.mp
        rw [← unpackExponent_toNat]
        simp [hExpOnes, IEEE32Exec.expAllOnes]
      split <;> rename_i hMantissa
      · have hFracField : IEEE32Exec.fracField (IEEE32Exec.ofBits bits) = 0 := by
          apply UInt32.toNat_inj.mp
          rw [← unpackMantissa_toNat]
          simp [hMantissa]
        simp [unpackedToDyadic?, IEEE32Exec.toDyadic?, IEEE32Exec.isNaN,
          IEEE32Exec.isInf, hExpField, hFracField]
      · have hFracField : IEEE32Exec.fracField (IEEE32Exec.ofBits bits) ≠ 0 := by
          intro h
          apply hMantissa
          apply BitVec.toNat_inj.mp
          rw [unpackMantissa_toNat]
          simp [h]
        simp [unpackedToDyadic?, IEEE32Exec.toDyadic?, IEEE32Exec.isNaN,
          IEEE32Exec.isInf, hExpField, hFracField]
    · have hExpField : IEEE32Exec.expField (IEEE32Exec.ofBits bits) ≠
          IEEE32Exec.expAllOnes := by
        intro h
        apply hExpOnes
        apply BitVec.toNat_inj.mp
        rw [unpackExponent_toNat]
        simp [h, IEEE32Exec.expAllOnes]
      have hNaN : IEEE32Exec.isNaN (IEEE32Exec.ofBits bits) = false := by
        simp [IEEE32Exec.isNaN, hExpField]
      have hInf : IEEE32Exec.isInf (IEEE32Exec.ofBits bits) = false := by
        simp [IEEE32Exec.isInf, hExpField]
      split <;> rename_i hExpZero
      · have hExpFieldZero : IEEE32Exec.expField (IEEE32Exec.ofBits bits) = 0 := by
          apply UInt32.toNat_inj.mp
          rw [← unpackExponent_toNat]
          simp [hExpZero]
        split <;> rename_i hMantissa
        · have hFracField : IEEE32Exec.fracField (IEEE32Exec.ofBits bits) = 0 := by
            apply UInt32.toNat_inj.mp
            rw [← unpackMantissa_toNat]
            simp [hMantissa]
          simp [unpackedToDyadic?, IEEE32Exec.toDyadic?, hNaN, hInf,
            hExpFieldZero, hFracField, signToBool_ofBitVec_unpackSign]
        · have hFracField : IEEE32Exec.fracField (IEEE32Exec.ofBits bits) ≠ 0 := by
            intro h
            apply hMantissa
            apply BitVec.toNat_inj.mp
            rw [unpackMantissa_toNat]
            simp [h]
          have hMantNat := unpackMantissa_toNat bits
          have hExpNatZero :
              (Float.Model.UnpackedFloat.unpackExponent
                  (spec := Float.Model.Format.binary32) bits.toBitVec).toNat = 0 := by
            exact congrArg BitVec.toNat hExpZero
          simp [unpackedToDyadic?, IEEE32Exec.toDyadic?, hNaN, hInf,
            hExpFieldZero, hFracField, signToBool_ofBitVec_unpackSign, hMantNat,
            hExpNatZero, Float.Model.Format.exponentBias]
      · have hExpFieldZero : IEEE32Exec.expField (IEEE32Exec.ofBits bits) ≠ 0 := by
          intro h
          apply hExpZero
          apply BitVec.toNat_inj.mp
          rw [unpackExponent_toNat]
          simp [h]
        have hMantNat := unpackMantissa_toNat bits
        have hExpNat := unpackExponent_toNat bits
        have hAppended :
            (1#1 ++ Float.Model.UnpackedFloat.unpackMantissa
                (spec := Float.Model.Format.binary32) bits.toBitVec).toNat =
              2 ^ 23 + (Float.Model.UnpackedFloat.unpackMantissa
                (spec := Float.Model.Format.binary32) bits.toBitVec).toNat := by
          rw [BitVec.toNat_append]
          rw [← Nat.shiftLeft_add_eq_or_of_lt
            (Float.Model.UnpackedFloat.unpackMantissa
              (spec := Float.Model.Format.binary32) bits.toBitVec).isLt]
          simp [Nat.shiftLeft_eq]
        simp [unpackedToDyadic?, IEEE32Exec.toDyadic?, hNaN, hInf,
          hExpFieldZero, signToBool_ofBitVec_unpackSign, hMantNat, hExpNat,
          Float.Model.Format.exponentBias, IEEE32Exec.pow2, Nat.shiftLeft_eq,
          hAppended]

/--
Every finite value produced by Lean's binary32 unpacker is in canonical finite form.

A subnormal has the fixed exponent `-149` and a mantissa below `2^23`. A normal has a 24-bit
mantissa and an exponent of at least `-149`. The two cases intentionally overlap at exponent
`-149`: the mantissa bound distinguishes subnormals from values in the smallest normal binade.
-/
theorem unpack_finite_bounds (a : Float32.Model)
    (sign : Float.Model.UnpackedFloat.Sign) (mantissa : Nat) (exponent : Int)
    (positive : 0 < mantissa)
    (ha : a.unpack = .finite sign mantissa exponent positive) :
    (exponent = -149 ∧ mantissa < 2 ^ 23) ∨
      (-149 ≤ exponent ∧ 2 ^ 23 ≤ mantissa ∧ mantissa < 2 ^ 24) := by
  cases a with
  | mk bits valid =>
    simp only [Float32.Model.unpack, Float.Model.UnpackedFloat.unpack] at ha
    split at ha <;> rename_i hexpOnes
    · split at ha <;> contradiction
    · split at ha <;> rename_i hexpZero
      · split at ha <;> rename_i hmantZero
        · contradiction
        · simp only [Float.Model.UnpackedFloat.finite.injEq] at ha
          rcases ha with ⟨rfl, rfl, rfl⟩
          left
          constructor
          · simp [hexpZero, Float.Model.Format.exponentBias]
          · exact (Float.Model.UnpackedFloat.unpackMantissa
              (spec := Float.Model.Format.binary32) bits.toBitVec).isLt
      · simp only [Float.Model.UnpackedFloat.finite.injEq] at ha
        rcases ha with ⟨rfl, rfl, rfl⟩
        right
        constructor
        · have hpos : 0 < (Float.Model.UnpackedFloat.unpackExponent
              (spec := Float.Model.Format.binary32) bits.toBitVec).toNat :=
            BitVec.toNat_pos_of_ne_zero hexpZero
          simp [Float.Model.Format.exponentBias]
          grind
        · have happended :
              (1#1 ++ Float.Model.UnpackedFloat.unpackMantissa
                  (spec := Float.Model.Format.binary32) bits.toBitVec).toNat =
                2 ^ 23 + (Float.Model.UnpackedFloat.unpackMantissa
                  (spec := Float.Model.Format.binary32) bits.toBitVec).toNat := by
            rw [BitVec.toNat_append]
            rw [← Nat.shiftLeft_add_eq_or_of_lt
              (Float.Model.UnpackedFloat.unpackMantissa
                (spec := Float.Model.Format.binary32) bits.toBitVec).isLt]
            simp [Nat.shiftLeft_eq]
          constructor
          · grind
          · have hlt := (Float.Model.UnpackedFloat.unpackMantissa
                (spec := Float.Model.Format.binary32) bits.toBitVec).isLt
            grind

/-- Every nonzero finite binary32 unpacking has a dyadic exponent between `-149` and `104`. -/
theorem unpack_finite_exponent_bounds (a : Float32.Model)
    (sign : Float.Model.UnpackedFloat.Sign) (mantissa : Nat) (exponent : Int)
    (positive : 0 < mantissa)
    (ha : a.unpack = .finite sign mantissa exponent positive) :
    -149 ≤ exponent ∧ exponent ≤ 104 := by
  cases a with
  | mk bits valid =>
    simp only [Float32.Model.unpack, Float.Model.UnpackedFloat.unpack] at ha
    split at ha <;> rename_i hexpOnes
    · split at ha <;> contradiction
    · split at ha <;> rename_i hexpZero
      · split at ha <;> rename_i hmantZero
        · contradiction
        · simp only [Float.Model.UnpackedFloat.finite.injEq] at ha
          rcases ha with ⟨rfl, rfl, rfl⟩
          simp [hexpZero, Float.Model.Format.exponentBias]
      · simp only [Float.Model.UnpackedFloat.finite.injEq] at ha
        rcases ha with ⟨rfl, rfl, rfl⟩
        have hpos : 0 < (Float.Model.UnpackedFloat.unpackExponent
            (spec := Float.Model.Format.binary32) bits.toBitVec).toNat :=
          BitVec.toNat_pos_of_ne_zero hexpZero
        have hlt : (Float.Model.UnpackedFloat.unpackExponent
            (spec := Float.Model.Format.binary32) bits.toBitVec).toNat < 255 := by
          have hwidth := (Float.Model.UnpackedFloat.unpackExponent
            (spec := Float.Model.Format.binary32) bits.toBitVec).isLt
          norm_num [Float.Model.Format.binary32] at hwidth
          have hne : (Float.Model.UnpackedFloat.unpackExponent
              (spec := Float.Model.Format.binary32) bits.toBitVec).toNat ≠ 255 := by
            intro heq
            apply hexpOnes
            apply BitVec.toNat_inj.mp
            simpa using heq
          grind
        simp only [Float.Model.Format.exponentBias]
        norm_num
        grind

end Float32Bridge

end TorchLean.Floats.IEEE754
