/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.LeanFloat32.RationalRounding
public import NN.Floats.IEEEExec.Bridge.LeanFloat32.Representation
public import NN.Floats.IEEEExec.Bridge.LeanFloat32.Rounding
import NN.Floats.IEEEExec.Bridge.FP32.Compare
import NN.Floats.IEEEExec.Rules.SpecialRules
public import NN.Floats.IEEEExec.Exec32.Compare

/-!
# Lean `Float32` and `IEEE32Exec`

Lean gives `Float32` a logical semantics through `Float32.Model`. Core arithmetic is defined in
the kernel by converting operands to the model, evaluating the corresponding model operation, and
converting back. The compiler replaces that definition with the platform's native `float`
instruction. This creates two distinct questions:

1. **Logical agreement.** Does TorchLean's independently implemented `IEEE32Exec` operation agree
   with `Float32.Model`?
2. **Native conformance.** Does the compiled instruction executed on a particular platform agree
   with the logical definition?

This file exposes the model behind ordinary Lean `Float32` expressions. It proves representation
and classification agreement, then proves comparisons, addition, subtraction, multiplication,
division, square root, negation, and absolute value against the independent executable model.
Native CPU, CUDA, and external-library agreement remains a backend contract; it is not folded into
the logical bridge.

Lean's language reference deliberately presents `Float32.Model` as a transfer target rather than
the foundation of a general floating-point library. TorchLean follows that design: its generic
rounding theory and `IEEE32Exec` remain independent, and this file is where results are transferred
to ordinary Lean `Float32` programs.

`Float32.Model` canonicalizes NaNs, while `IEEE32Exec` can retain arbitrary NaN payloads. The map
from the model therefore lands in the canonical subset of `IEEE32Exec`. Arithmetic results are
canonicalized before their bits are compared; otherwise even negating a canonical NaN would expose
the different payload policies rather than a difference in numerical semantics.

## References

- `Init.Data.Float.Model.Float32`, the logical model used by `Float32`.
- The Lean Language Reference, “Floating-Point Numbers.”
  https://lean-lang.org/doc/reference/latest/Basic-Types/Floating-Point-Numbers/
- IEEE Standard for Floating-Point Arithmetic, IEEE 754-2019.
  https://doi.org/10.1109/IEEESTD.2019.8766229
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open IEEE32Exec

namespace Float32Bridge

/-! ## Lean's logical `Float32` semantics -/

@[simp] theorem toModel_ofModel (x : Float32.Model) :
    (Float32.ofModel x).toModel = x := by
  rfl

/-- `Float32` and its model are the same data, so the round trip is the identity by `rfl`.

This direction is the one that matters for rewriting: a goal about the opaque `Float32` can be
turned into a goal about the model without any side condition. -/
@[simp] theorem ofModel_toModel (x : _root_.Float32) :
    Float32.ofModel x.toModel = x := by
  cases x
  rfl

/-- Lean addition exposes exactly `Float32.Model.add` to the kernel. -/
@[simp] theorem toModel_add (a b : _root_.Float32) :
    (Float32.add a b).toModel = a.toModel + b.toModel := by
  rfl

/-- Lean subtraction exposes exactly `Float32.Model.sub` to the kernel. -/
@[simp] theorem toModel_sub (a b : _root_.Float32) :
    (Float32.sub a b).toModel = a.toModel - b.toModel := by
  rfl

/-- Lean multiplication exposes exactly `Float32.Model.mul` to the kernel. -/
@[simp] theorem toModel_mul (a b : _root_.Float32) :
    (Float32.mul a b).toModel = a.toModel * b.toModel := by
  rfl

/-- Lean division exposes exactly `Float32.Model.div` to the kernel. -/
@[simp] theorem toModel_div (a b : _root_.Float32) :
    (Float32.div a b).toModel = a.toModel / b.toModel := by
  rfl

/-- Lean negation exposes exactly `Float32.Model.neg` to the kernel. -/
@[simp] theorem toModel_neg (a : _root_.Float32) :
    (Float32.neg a).toModel = -a.toModel := by
  rfl

/-- Lean square root exposes exactly `Float32.Model.sqrt` to the kernel. -/
@[simp] theorem toModel_sqrt (a : _root_.Float32) :
    (Float32.sqrt a).toModel = a.toModel.sqrt := by
  rfl

/-- Lean absolute value exposes exactly `Float32.Model.abs` to the kernel. -/
@[simp] theorem toModel_abs (a : _root_.Float32) :
    (Float32.abs a).toModel = a.toModel.abs := by
  rfl

/-- Reading a raw bit pattern as `Float32` uses Lean's canonical model constructor. -/
@[simp] theorem toModel_ofBits (bits : UInt32) :
    (Float32.ofBits bits).toModel = Float32.Model.ofBits bits := by
  rfl

/-- `Float32.toBits` exposes the bit pattern stored by the logical model. -/
@[simp] theorem toBits_eq_model (a : _root_.Float32) :
    Float32.toBits a = a.toModel.toBits := by
  rfl

/-- Lean strict comparison exposes exactly the model's IEEE comparison. -/
@[simp] theorem lt_eq_model (a b : _root_.Float32) :
    Float32.lt a b = Float32.Model.lt a.toModel b.toModel := by
  change decide (Float32.Model.lt a.toModel b.toModel = true) = _
  cases Float32.Model.lt a.toModel b.toModel <;> rfl

/-- Lean non-strict comparison exposes exactly the model's IEEE comparison. -/
@[simp] theorem le_eq_model (a b : _root_.Float32) :
    Float32.le a b = Float32.Model.le a.toModel b.toModel := by
  change decide (Float32.Model.le a.toModel b.toModel = true) = _
  cases Float32.Model.le a.toModel b.toModel <;> rfl

/-- Lean floating-point equality exposes exactly the model's IEEE equality test. -/
@[simp] theorem beq_eq_model (a b : _root_.Float32) :
    Float32.beq a b = Float32.Model.beq a.toModel b.toModel := by
  rfl

/-- `Float32.isFinite` is the model predicate. -/
@[simp] theorem isFinite_eq_model (a : _root_.Float32) :
    Float32.isFinite a = a.toModel.isFinite := by
  rfl

/-- `Float32.isInf` is the model predicate. -/
@[simp] theorem isInf_eq_model (a : _root_.Float32) :
    Float32.isInf a = a.toModel.isInf := by
  rfl

/-- `Float32.isNaN` is the model predicate. -/
@[simp] theorem isNaN_eq_model (a : _root_.Float32) :
    Float32.isNaN a = a.toModel.isNaN := by
  rfl

/-- Converting raw executable bits through Lean performs precisely model-level canonicalization. -/
@[simp] theorem toIEEE32Exec_ofIEEE32Exec (x : IEEE32Exec) :
    toIEEE32Exec (ofIEEE32Exec x) = canonicalize x := by
  rfl

/-! ## Canonical model round trips -/

private theorem model_ext {a b : Float32.Model} (h : a.toBits = b.toBits) : a = b := by
  cases a
  cases b
  cases h
  rfl

private theorem packComponents_neg_sign
    (sign : Float.Model.UnpackedFloat.Sign) (exponent : BitVec 8) (mantissa : BitVec 23) :
    UInt32.ofBitVec
        (Float.Model.UnpackedFloat.packComponents Float.Model.Format.binary32
          (-sign) exponent mantissa) =
      UInt32.ofBitVec
          (Float.Model.UnpackedFloat.packComponents Float.Model.Format.binary32
            sign exponent mantissa) ^^^ IEEE32Exec.signMask := by
  rw [packComponents_eq_mkBits, packComponents_eq_mkBits, signToBool_neg]
  exact IEEE32Exec.mkBits_not_sign _ _ _

private theorem packComponents_abs_sign
    (sign : Float.Model.UnpackedFloat.Sign) (exponent : BitVec 8) (mantissa : BitVec 23) :
    UInt32.ofBitVec
        (Float.Model.UnpackedFloat.packComponents Float.Model.Format.binary32
          .positive exponent mantissa) =
      UInt32.ofBitVec
          (Float.Model.UnpackedFloat.packComponents Float.Model.Format.binary32
            sign exponent mantissa) &&& (~~~IEEE32Exec.signMask) := by
  rw [packComponents_eq_mkBits, packComponents_eq_mkBits]
  exact IEEE32Exec.mkBits_clear_sign _ _ _

private theorem pack_neg_eq_xor_sign :
    ∀ value : Float.Model.UnpackedFloat,
      value ≠ .notANumber →
        UInt32.ofBitVec
            (Float.Model.UnpackedFloat.pack Float.Model.Format.binary32 value.neg) =
          UInt32.ofBitVec
              (Float.Model.UnpackedFloat.pack Float.Model.Format.binary32 value) ^^^
            IEEE32Exec.signMask
  | .notANumber, h => (h rfl).elim
  | .infinity sign, _ => by
      simp [Float.Model.UnpackedFloat.neg, Float.Model.UnpackedFloat.pack,
        Float.Model.UnpackedFloat.packedInfinity, packComponents_neg_sign]
  | .zero sign, _ => by
      simp [Float.Model.UnpackedFloat.neg, Float.Model.UnpackedFloat.pack,
        Float.Model.UnpackedFloat.packedZero, packComponents_neg_sign]
  | .finite sign mantissa exponent positive, _ => by
      simp only [Float.Model.UnpackedFloat.neg, Float.Model.UnpackedFloat.pack]
      split
      · simp [Float.Model.UnpackedFloat.packedInfinity, packComponents_neg_sign]
      · split <;> simp only [packComponents_neg_sign]

private theorem pack_abs_eq_clear_sign :
    ∀ value : Float.Model.UnpackedFloat,
      value ≠ .notANumber →
        UInt32.ofBitVec
            (Float.Model.UnpackedFloat.pack Float.Model.Format.binary32 value.abs) =
          UInt32.ofBitVec
              (Float.Model.UnpackedFloat.pack Float.Model.Format.binary32 value) &&&
            (~~~IEEE32Exec.signMask)
  | .notANumber, h => (h rfl).elim
  | .infinity sign, _ => by
      simp only [Float.Model.UnpackedFloat.abs, Float.Model.UnpackedFloat.pack,
        Float.Model.UnpackedFloat.packedInfinity]
      change UInt32.ofBitVec (Float.Model.UnpackedFloat.packComponents
          Float.Model.Format.binary32 .positive (-1#8) 0#23) =
        UInt32.ofBitVec (Float.Model.UnpackedFloat.packComponents
          Float.Model.Format.binary32 sign (-1#8) 0#23) &&& (~~~IEEE32Exec.signMask)
      exact packComponents_abs_sign sign (-1#8) 0#23
  | .zero sign, _ => by
      simp only [Float.Model.UnpackedFloat.abs, Float.Model.UnpackedFloat.pack,
        Float.Model.UnpackedFloat.packedZero]
      change UInt32.ofBitVec (Float.Model.UnpackedFloat.packComponents
          Float.Model.Format.binary32 .positive 0#8 0#23) =
        UInt32.ofBitVec (Float.Model.UnpackedFloat.packComponents
          Float.Model.Format.binary32 sign 0#8 0#23) &&& (~~~IEEE32Exec.signMask)
      exact packComponents_abs_sign sign 0#8 0#23
  | .finite sign mantissa exponent positive, _ => by
      simp only [Float.Model.UnpackedFloat.abs, Float.Model.UnpackedFloat.pack]
      split
      · simp only [Float.Model.UnpackedFloat.packedInfinity]
        change UInt32.ofBitVec (Float.Model.UnpackedFloat.packComponents
            Float.Model.Format.binary32 .positive (-1#8) 0#23) =
          UInt32.ofBitVec (Float.Model.UnpackedFloat.packComponents
            Float.Model.Format.binary32 sign (-1#8) 0#23) &&& (~~~IEEE32Exec.signMask)
        exact packComponents_abs_sign sign (-1#8) 0#23
      · split
        · exact packComponents_abs_sign sign _ _
        · exact packComponents_abs_sign sign _ _

private theorem appendedMantissa_log2 (mantissa : BitVec 23) :
    (1#1 ++ mantissa).toNat.log2 = 23 := by
  rw [Nat.log2_eq_iff]
  · constructor
    · rw [BitVec.toNat_append, ← Nat.shiftLeft_add_eq_or_of_lt mantissa.isLt]
      simp [Nat.shiftLeft_eq]
    · rw [BitVec.toNat_append, ← Nat.shiftLeft_add_eq_or_of_lt mantissa.isLt]
      simp only [BitVec.toNat_ofNat, Nat.reducePow, Nat.reduceMod, Nat.shiftLeft_eq]
      have := mantissa.isLt
      grind
  · rw [BitVec.toNat_append, ← Nat.shiftLeft_add_eq_or_of_lt mantissa.isLt]
    simp [Nat.shiftLeft_eq]

private theorem setWidth_appendedMantissa (mantissa : BitVec 23) :
    BitVec.setWidth 23 (1#1 ++ mantissa) = mantissa := by
  rw [BitVec.setWidth_append]
  simp

private theorem pack_unpack_valid (bits : UInt32)
    (valid : Float.Model.Format.binary32.Valid bits.toBitVec) :
    UInt32.ofBitVec
        (Float.Model.UnpackedFloat.pack Float.Model.Format.binary32
          (Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32 bits.toBitVec)) = bits := by
  by_cases hExpOnes : Float.Model.UnpackedFloat.unpackExponent
      (spec := Float.Model.Format.binary32) bits.toBitVec = -1#8
  · by_cases hMantissa : Float.Model.UnpackedFloat.unpackMantissa
        (spec := Float.Model.Format.binary32) bits.toBitVec = 0#23
    · have hUnpack : Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32
          bits.toBitVec = .infinity
            (Float.Model.UnpackedFloat.Sign.ofBitVec
              (Float.Model.UnpackedFloat.unpackSign
                (spec := Float.Model.Format.binary32) bits.toBitVec)) := by
        simp only [Float.Model.UnpackedFloat.unpack]
        rw [if_pos hExpOnes, if_pos hMantissa]
      rw [hUnpack]
      change UInt32.ofBitVec
        (Float.Model.UnpackedFloat.packComponents Float.Model.Format.binary32
          (Float.Model.UnpackedFloat.Sign.ofBitVec
            (Float.Model.UnpackedFloat.unpackSign
              (spec := Float.Model.Format.binary32) bits.toBitVec)) (-1#8) 0#23) = bits
      have h := packComponents_unpacked_fields bits
      rw [hExpOnes, hMantissa] at h
      exact h
    · have hNaN : bits.toBitVec =
          Float.Model.UnpackedFloat.packedNaN Float.Model.Format.binary32 :=
        valid.eq_packedNaN hExpOnes hMantissa
      have hUnpack : Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32
          bits.toBitVec = .notANumber := by
        simp only [Float.Model.UnpackedFloat.unpack]
        rw [if_pos hExpOnes, if_neg hMantissa]
      rw [hUnpack]
      simp only [Float.Model.UnpackedFloat.pack]
      exact UInt32.toBitVec_inj.mp (by simpa using hNaN.symm)
  · by_cases hExpZero : Float.Model.UnpackedFloat.unpackExponent
        (spec := Float.Model.Format.binary32) bits.toBitVec = 0#8
    · by_cases hMantissa : Float.Model.UnpackedFloat.unpackMantissa
          (spec := Float.Model.Format.binary32) bits.toBitVec = 0#23
      · have hUnpack : Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32
            bits.toBitVec = .zero
              (Float.Model.UnpackedFloat.Sign.ofBitVec
                (Float.Model.UnpackedFloat.unpackSign
                  (spec := Float.Model.Format.binary32) bits.toBitVec)) := by
          simp only [Float.Model.UnpackedFloat.unpack]
          rw [if_neg hExpOnes, if_pos hExpZero, dif_pos hMantissa]
        rw [hUnpack]
        change UInt32.ofBitVec
          (Float.Model.UnpackedFloat.packComponents Float.Model.Format.binary32
            (Float.Model.UnpackedFloat.Sign.ofBitVec
              (Float.Model.UnpackedFloat.unpackSign
                (spec := Float.Model.Format.binary32) bits.toBitVec)) 0#8 0#23) = bits
        have h := packComponents_unpacked_fields bits
        rw [hExpZero, hMantissa] at h
        exact h
      · simp [Float.Model.UnpackedFloat.unpack, hExpZero,
          hMantissa, Float.Model.UnpackedFloat.pack]
        have hMantissaNat :
            (Float.Model.UnpackedFloat.unpackMantissa
              (spec := Float.Model.Format.binary32) bits.toBitVec).toNat ≠ 0 := by
          intro h
          apply hMantissa
          exact BitVec.toNat_inj.mp (by simpa using h)
        have hNotNormal :
            (Float.Model.UnpackedFloat.unpackMantissa
                (spec := Float.Model.Format.binary32) bits.toBitVec).toNat.log2 + 1 ≠ 24 := by
          intro h
          have hLog :
              (Float.Model.UnpackedFloat.unpackMantissa
                  (spec := Float.Model.Format.binary32) bits.toBitVec).toNat.log2 = 23 := by
            grind
          have hLower : 2 ^ 23 ≤
              (Float.Model.UnpackedFloat.unpackMantissa
                (spec := Float.Model.Format.binary32) bits.toBitVec).toNat :=
            ((Nat.log2_eq_iff hMantissaNat).mp hLog).1
          exact (Nat.not_le_of_lt (BitVec.isLt _)) hLower
        simp [Float.Model.Format.exponentBias, Float.Model.Format.mantissaBits,
          hNotNormal]
        have h := packComponents_unpacked_fields bits
        rw [hExpZero] at h
        exact h
    · have hExp255 : Float.Model.UnpackedFloat.unpackExponent
          (spec := Float.Model.Format.binary32) bits.toBitVec ≠ 255#8 := by
        simpa using hExpOnes
      let exponent := Float.Model.UnpackedFloat.unpackExponent
        (spec := Float.Model.Format.binary32) bits.toBitVec
      let mantissa := Float.Model.UnpackedFloat.unpackMantissa
        (spec := Float.Model.Format.binary32) bits.toBitVec
      have hBiasedExponent :
          ((exponent.toNat : Int) -
                (Float.Model.Format.binary32.exponentBias + 23) +
              Float.Model.Format.binary32.exponentBias + 23).toNat = exponent.toNat := by
        change ((exponent.toNat : Int) - (127 + 23) + 127 + 23).toNat = exponent.toNat
        grind
      have hExponentNat : exponent.toNat ≠ 255 := by
        intro h
        apply hExp255
        apply BitVec.toNat_inj.mp
        simpa [exponent] using h
      have hNoOverflow : ¬255 ≤ exponent.toNat := by
        have hExponentLt : exponent.toNat < 256 := exponent.isLt
        grind
      have hNormal : (1#1 ++ mantissa).toNat.log2 + 1 = 24 := by
        rw [appendedMantissa_log2]
      simp [Float.Model.UnpackedFloat.unpack, hExp255, hExpZero,
        Float.Model.UnpackedFloat.pack, exponent, mantissa, hBiasedExponent,
        hNoOverflow, hNormal, Float.Model.Format.mantissaBits,
        setWidth_appendedMantissa]
      have h := packComponents_unpacked_fields bits
      simpa [exponent, mantissa] using h

/-- Reconstructing a canonical model value from its bits returns the original model value. -/
@[simp] theorem model_ofBits_toBits (a : Float32.Model) :
    Float32.Model.ofBits a.toBits = a := by
  cases a with
  | mk bits valid =>
    apply model_ext
    simpa [Float32.Model.ofBits, Float32.Model.pack] using pack_unpack_valid bits valid

/-- Packing the canonical unpacked representation of a model value returns that value. -/
@[simp] theorem model_pack_unpack (a : Float32.Model) :
    Float32.Model.pack a.unpack = a := by
  apply model_ext
  cases a with
  | mk bits valid =>
    simpa [Float32.Model.pack, Float32.Model.unpack] using pack_unpack_valid bits valid

/-- Canonicalizing a value already obtained from `Float32.Model` changes no bits. -/
@[simp] theorem canonicalize_modelToIEEE32Exec (a : Float32.Model) :
    canonicalize (modelToIEEE32Exec a) = modelToIEEE32Exec a := by
  change IEEE32Exec.ofBits (Float32.Model.ofBits a.toBits).toBits =
    IEEE32Exec.ofBits a.toBits
  rw [model_ofBits_toBits]

/-- NaN canonicalization is idempotent. -/
@[simp] theorem canonicalize_idempotent (x : IEEE32Exec) :
    canonicalize (canonicalize x) = canonicalize x := by
  change canonicalize (modelToIEEE32Exec (Float32.Model.ofBits x.bits)) = _
  rw [canonicalize_modelToIEEE32Exec]
  rfl

/-- The canonical model embedding is injective. -/
theorem modelToIEEE32Exec_injective : Function.Injective modelToIEEE32Exec := by
  intro a b h
  apply model_ext
  exact congrArg IEEE32Exec.bits h

/-! ## Reading an unpacked model value -/

/-- A model value that unpacks as NaN maps to TorchLean's canonical NaN encoding. -/
theorem modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq
    (a : Float32.Model) (h : a.unpack = .notANumber) :
    modelToIEEE32Exec a = IEEE32Exec.canonicalNaN := by
  rw [← model_pack_unpack a, h]
  exact modelToIEEE32Exec_pack_notANumber

/-- A model infinity maps to the executable infinity with the same sign. -/
theorem modelToIEEE32Exec_eq_inf_of_unpack_eq
    (a : Float32.Model) (sign : Float.Model.UnpackedFloat.Sign)
    (h : a.unpack = .infinity sign) :
    modelToIEEE32Exec a =
      if signToBool sign then IEEE32Exec.negInf else IEEE32Exec.posInf := by
  rw [← model_pack_unpack a, h]
  exact modelToIEEE32Exec_pack_infinity sign

/-- A model zero maps to the executable zero with the same sign. -/
theorem modelToIEEE32Exec_eq_zero_of_unpack_eq
    (a : Float32.Model) (sign : Float.Model.UnpackedFloat.Sign)
    (h : a.unpack = .zero sign) :
    modelToIEEE32Exec a =
      if signToBool sign then IEEE32Exec.negZero else IEEE32Exec.posZero := by
  rw [← model_pack_unpack a, h]
  exact modelToIEEE32Exec_pack_zero sign

/-- A finite model value maps to the exact signed dyadic exposed by its unpacked form. -/
theorem toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
    (a : Float32.Model) (sign : Float.Model.UnpackedFloat.Sign)
    (mantissa : Nat) (exponent : Int) (positive : 0 < mantissa)
    (h : a.unpack = .finite sign mantissa exponent positive) :
    IEEE32Exec.toDyadic? (modelToIEEE32Exec a) = some
      { sign := signToBool sign, mant := mantissa, exp := exponent } := by
  rw [toDyadic_modelToIEEE32Exec, h]
  rfl

/-- Lean's canonical model never maps to an executable signaling NaN. -/
@[simp] theorem model_isSNaN_eq_false (a : Float32.Model) :
    IEEE32Exec.isSNaN (modelToIEEE32Exec a) = false := by
  cases h : a.unpack with
  | notANumber =>
      rw [modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq a h]
      decide
  | infinity sign =>
      rw [modelToIEEE32Exec_eq_inf_of_unpack_eq a sign h]
      cases sign <;> decide
  | zero sign =>
      rw [modelToIEEE32Exec_eq_zero_of_unpack_eq a sign h]
      cases sign <;> decide
  | finite sign mantissa exponent positive =>
      have hdy := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
        a sign mantissa exponent positive h
      have hNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some hdy
      simp [IEEE32Exec.isSNaN, hNaN]

/-- Flipping the packed sign bit negates the corresponding unpacked binary32 value. -/
private theorem unpack_xor_sign (bits : UInt32) :
    Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32
        (bits ^^^ IEEE32Exec.signMask).toBitVec =
      (Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32 bits.toBitVec).neg := by
  simp only [Float.Model.UnpackedFloat.unpack, unpackExponent_xor_sign,
    unpackMantissa_xor_sign, unpackSign_xor_sign]
  split
  · split <;> rfl
  · split
    · split <;> rfl
    · rfl

private theorem model_neg_eq_ofBits_xor_sign (a : Float32.Model) :
    Float32.Model.neg a =
      Float32.Model.ofBits (a.toBits ^^^ IEEE32Exec.signMask) := by
  apply model_ext
  have hPack : (Float32.Model.pack a.unpack).toBits = a.toBits :=
    congrArg Float32.Model.toBits (model_pack_unpack a)
  by_cases hNaN : a.unpack = .notANumber
  · have ha : a = Float32.Model.pack .notANumber := by
      apply model_ext
      rw [hNaN] at hPack
      exact hPack.symm
    rw [ha]
    rfl
  · have hNegBits : (Float32.Model.pack a.unpack.neg).toBits =
        a.toBits ^^^ IEEE32Exec.signMask := by
      calc
        (Float32.Model.pack a.unpack.neg).toBits =
            (Float32.Model.pack a.unpack).toBits ^^^ IEEE32Exec.signMask :=
          pack_neg_eq_xor_sign a.unpack hNaN
        _ = a.toBits ^^^ IEEE32Exec.signMask := by rw [hPack]
    change (Float32.Model.pack a.unpack.neg).toBits =
      (Float32.Model.ofBits (a.toBits ^^^ IEEE32Exec.signMask)).toBits
    rw [← hNegBits, model_ofBits_toBits]

/-- Model-level negation changes only the unpacked sign. -/
@[simp] theorem model_unpack_neg (a : Float32.Model) :
    (Float32.Model.neg a).unpack = a.unpack.neg := by
  unfold Float32.Model.neg
  by_cases hNaN : a.unpack = .notANumber
  · rw [hNaN]
    rfl
  · have hBits : (Float32.Model.pack a.unpack.neg).toBits =
        a.toBits ^^^ IEEE32Exec.signMask := by
      calc
        (Float32.Model.pack a.unpack.neg).toBits =
            (Float32.Model.pack a.unpack).toBits ^^^ IEEE32Exec.signMask :=
          pack_neg_eq_xor_sign a.unpack hNaN
        _ = a.toBits ^^^ IEEE32Exec.signMask := by rw [model_pack_unpack]
    change Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32
        (Float32.Model.pack a.unpack.neg).toBits.toBitVec = a.unpack.neg
    rw [hBits]
    exact unpack_xor_sign a.toBits

private theorem unpacked_sub_eq_add_neg (x y : Float.Model.UnpackedFloat) :
    Float.Model.UnpackedFloat.sub Float.Model.Format.binary32 x y =
      Float.Model.UnpackedFloat.add Float.Model.Format.binary32 x y.neg := by
  cases x <;> cases y <;>
    simp [Float.Model.UnpackedFloat.sub, Float.Model.UnpackedFloat.add,
      Float.Model.UnpackedFloat.neg]
  case finite.finite sign₁ _ _ _ sign₂ _ _ _ =>
    cases sign₂ <;> simp [Float.Model.UnpackedFloat.Sign.apply]
    congr 2

/-- Binary32 subtraction in Lean's model is addition after negating the second operand. -/
theorem model_sub_eq_add_neg (a b : Float32.Model) :
    Float32.Model.sub a b = Float32.Model.add a (Float32.Model.neg b) := by
  unfold Float32.Model.sub Float32.Model.add
  rw [model_unpack_neg, unpacked_sub_eq_add_neg]

private theorem model_abs_eq_ofBits_clear_sign (a : Float32.Model) :
    a.abs = Float32.Model.ofBits (a.toBits &&& (~~~IEEE32Exec.signMask)) := by
  apply model_ext
  have hPack : (Float32.Model.pack a.unpack).toBits = a.toBits :=
    congrArg Float32.Model.toBits (model_pack_unpack a)
  by_cases hNaN : a.unpack = .notANumber
  · have ha : a = Float32.Model.pack .notANumber := by
      apply model_ext
      rw [hNaN] at hPack
      exact hPack.symm
    rw [ha]
    rfl
  · have hAbsBits : (Float32.Model.pack a.unpack.abs).toBits =
        a.toBits &&& (~~~IEEE32Exec.signMask) := by
      calc
        (Float32.Model.pack a.unpack.abs).toBits =
            (Float32.Model.pack a.unpack).toBits &&& (~~~IEEE32Exec.signMask) :=
          pack_abs_eq_clear_sign a.unpack hNaN
        _ = a.toBits &&& (~~~IEEE32Exec.signMask) := by rw [hPack]
    change (Float32.Model.pack a.unpack.abs).toBits =
      (Float32.Model.ofBits (a.toBits &&& (~~~IEEE32Exec.signMask))).toBits
    rw [← hAbsBits, model_ofBits_toBits]

/-! ## Classification agrees without assumptions -/

/-- Canonicalization leaves every non-NaN encoding unchanged. -/
theorem canonicalize_eq_self_of_isNaN_eq_false (x : IEEE32Exec)
    (hNaN : IEEE32Exec.isNaN x = false) : canonicalize x = x := by
  have hValid : Float.Model.Format.binary32.Valid x.bits.toBitVec := by
    constructor
    intro hExponent hMantissa
    have hField : IEEE32Exec.expField x = IEEE32Exec.expAllOnes := by
      simpa [IEEE32Exec.ofBits] using (unpackExponent_allOnes_iff x.bits).1 hExponent
    have hFraction : IEEE32Exec.fracField x ≠ 0 := by
      intro hZero
      exact hMantissa ((unpackMantissa_zero_iff x.bits).2 hZero)
    have : IEEE32Exec.isNaN x = true := by
      simp [IEEE32Exec.isNaN, hField, hFraction]
    simp_all
  apply congrArg IEEE32Exec.ofBits
  simpa [canonicalize, modelToIEEE32Exec, Float32.Model.ofBits, Float32.Model.pack] using
    pack_unpack_valid x.bits hValid

/-- Canonicalization maps every NaN payload to Lean's unique model NaN. -/
theorem canonicalize_eq_canonicalNaN_of_isNaN_eq_true (x : IEEE32Exec)
    (hNaN : IEEE32Exec.isNaN x = true) : canonicalize x = IEEE32Exec.canonicalNaN := by
  have hParts : IEEE32Exec.expField x = IEEE32Exec.expAllOnes ∧
      IEEE32Exec.fracField x ≠ 0 := by
    simpa [IEEE32Exec.isNaN] using hNaN
  have hField : IEEE32Exec.expField x = IEEE32Exec.expAllOnes := by
    exact hParts.1
  have hFraction : IEEE32Exec.fracField x ≠ 0 := by
    exact hParts.2
  have hExponent := (unpackExponent_allOnes_iff x.bits).2 hField
  have hMantissa : Float.Model.UnpackedFloat.unpackMantissa
      (spec := Float.Model.Format.binary32) x.bits.toBitVec ≠ 0#23 := by
    intro hZero
    exact hFraction ((unpackMantissa_zero_iff x.bits).1 hZero)
  have hUnpack : Float.Model.UnpackedFloat.unpack Float.Model.Format.binary32
      x.bits.toBitVec = .notANumber := by
    simp [Float.Model.UnpackedFloat.unpack, hExponent, hMantissa]
  have hPacked : UInt32.ofBitVec
        (Float.Model.UnpackedFloat.packedNaN Float.Model.Format.binary32) =
      IEEE32Exec.expMask ||| IEEE32Exec.quietBit := by
    decide
  simpa [canonicalize, modelToIEEE32Exec, Float32.Model.ofBits, Float32.Model.pack,
    hUnpack, Float.Model.UnpackedFloat.pack, IEEE32Exec.canonicalNaN] using
      congrArg IEEE32Exec.ofBits hPacked

private theorem isNaN_add_eq_true_of_right (x y : IEEE32Exec)
    (hy : IEEE32Exec.isNaN y = true) : IEEE32Exec.isNaN (IEEE32Exec.add x y) = true := by
  cases hxS : IEEE32Exec.isSNaN x with
  | true =>
      have hx : IEEE32Exec.isNaN x = true := by
        have hParts : IEEE32Exec.isNaN x = true ∧
            (x.bits &&& IEEE32Exec.quietBit) = 0 := by
          simpa [IEEE32Exec.isSNaN] using hxS
        exact hParts.1
      rw [IEEE32Exec.add_eq_of_chooseNaN2_some x y (IEEE32Exec.quietNaN x)
        (IEEE32Exec.chooseNaN2_of_isSNaN_left x y hxS)]
      exact IEEE32Exec.isNaN_quietNaN_eq_true x hx
  | false =>
      cases hyS : IEEE32Exec.isSNaN y with
      | true =>
          rw [IEEE32Exec.add_eq_of_chooseNaN2_some x y (IEEE32Exec.quietNaN y)
            (IEEE32Exec.chooseNaN2_of_isSNaN_right x y hxS hyS)]
          exact IEEE32Exec.isNaN_quietNaN_eq_true y hy
      | false =>
          cases hx : IEEE32Exec.isNaN x with
          | true =>
              rw [IEEE32Exec.add_eq_of_chooseNaN2_some x y (IEEE32Exec.quietNaN x)
                (IEEE32Exec.chooseNaN2_of_isNaN_left x y hxS hyS hx)]
              exact IEEE32Exec.isNaN_quietNaN_eq_true x hx
          | false =>
              rw [IEEE32Exec.add_eq_of_chooseNaN2_some x y (IEEE32Exec.quietNaN y)
                (IEEE32Exec.chooseNaN2_of_isNaN_right x y hxS hyS hx hy)]
              exact IEEE32Exec.isNaN_quietNaN_eq_true y hy

/-- Canonicalizing an input NaN payload does not change the canonical result of addition. -/
theorem canonicalize_add_right (x y : IEEE32Exec) :
    canonicalize (IEEE32Exec.add x (canonicalize y)) =
      canonicalize (IEEE32Exec.add x y) := by
  cases hy : IEEE32Exec.isNaN y with
  | false => rw [canonicalize_eq_self_of_isNaN_eq_false y hy]
  | true =>
      rw [canonicalize_eq_canonicalNaN_of_isNaN_eq_true y hy]
      rw [canonicalize_eq_canonicalNaN_of_isNaN_eq_true
        (IEEE32Exec.add x IEEE32Exec.canonicalNaN)
        (isNaN_add_eq_true_of_right x IEEE32Exec.canonicalNaN (by decide))]
      rw [canonicalize_eq_canonicalNaN_of_isNaN_eq_true
        (IEEE32Exec.add x y) (isNaN_add_eq_true_of_right x y hy)]

/-- Lean's model and `IEEE32Exec` classify every canonical binary32 value as finite identically. -/
theorem model_isFinite_eq_ieee32 (a : Float32.Model) :
    a.isFinite = IEEE32Exec.isFinite (modelToIEEE32Exec a) := by
  cases a with
  | mk bits valid =>
    by_cases hExp : Float.Model.UnpackedFloat.unpackExponent
        (spec := Float.Model.Format.binary32) bits.toBitVec = -1#8
    · have hExp255 : Float.Model.UnpackedFloat.unpackExponent
          (spec := Float.Model.Format.binary32) bits.toBitVec = 255#8 := by
        simpa using hExp
      have hField := (unpackExponent_allOnes_iff bits).1 hExp
      simp [Float32.Model.isFinite, Float32.Model.unpack, modelToIEEE32Exec,
        IEEE32Exec.isFinite, Float.Model.UnpackedFloat.unpack, hExp255, hField]
      split_ifs <;> rfl
    · have hExp255 : Float.Model.UnpackedFloat.unpackExponent
          (spec := Float.Model.Format.binary32) bits.toBitVec ≠ 255#8 := by
        simpa using hExp
      have hField : IEEE32Exec.expField (IEEE32Exec.ofBits bits) ≠
          IEEE32Exec.expAllOnes := by
        exact fun h => hExp ((unpackExponent_allOnes_iff bits).2 h)
      simp [Float32.Model.isFinite, Float32.Model.unpack, modelToIEEE32Exec,
        IEEE32Exec.isFinite, Float.Model.UnpackedFloat.unpack, hExp255]
      split_ifs <;> simp [Float.Model.UnpackedFloat.isFinite, hField]

/-- Lean's model and `IEEE32Exec` identify the two signed infinities identically. -/
theorem model_isInf_eq_ieee32 (a : Float32.Model) :
    a.isInf = IEEE32Exec.isInf (modelToIEEE32Exec a) := by
  cases a with
  | mk bits valid =>
    by_cases hExp : Float.Model.UnpackedFloat.unpackExponent
        (spec := Float.Model.Format.binary32) bits.toBitVec = -1#8
    · have hExp255 : Float.Model.UnpackedFloat.unpackExponent
          (spec := Float.Model.Format.binary32) bits.toBitVec = 255#8 := by
        simpa using hExp
      have hField := (unpackExponent_allOnes_iff bits).1 hExp
      by_cases hMantissa : Float.Model.UnpackedFloat.unpackMantissa
          (spec := Float.Model.Format.binary32) bits.toBitVec = 0#23
      · have hFraction := (unpackMantissa_zero_iff bits).1 hMantissa
        simp [Float32.Model.isInf, Float32.Model.unpack, modelToIEEE32Exec,
          IEEE32Exec.isInf, Float.Model.UnpackedFloat.unpack,
          Float.Model.UnpackedFloat.isInf, hExp255, hMantissa, hField, hFraction]
      · have hFraction : IEEE32Exec.fracField (IEEE32Exec.ofBits bits) ≠ 0 := by
          exact fun h => hMantissa ((unpackMantissa_zero_iff bits).2 h)
        simp [Float32.Model.isInf, Float32.Model.unpack, modelToIEEE32Exec,
          IEEE32Exec.isInf, Float.Model.UnpackedFloat.unpack,
          Float.Model.UnpackedFloat.isInf, hExp255, hMantissa, hField, hFraction]
    · have hExp255 : Float.Model.UnpackedFloat.unpackExponent
          (spec := Float.Model.Format.binary32) bits.toBitVec ≠ 255#8 := by
        simpa using hExp
      have hField : IEEE32Exec.expField (IEEE32Exec.ofBits bits) ≠
          IEEE32Exec.expAllOnes := by
        exact fun h => hExp ((unpackExponent_allOnes_iff bits).2 h)
      simp [Float32.Model.isInf, Float32.Model.unpack, modelToIEEE32Exec,
        IEEE32Exec.isInf, Float.Model.UnpackedFloat.unpack, hExp255]
      split_ifs <;> simp [Float.Model.UnpackedFloat.isInf, hField]

/-- Lean's model and `IEEE32Exec` classify canonical NaNs identically. -/
theorem model_isNaN_eq_ieee32 (a : Float32.Model) :
    a.isNaN = IEEE32Exec.isNaN (modelToIEEE32Exec a) := by
  cases a with
  | mk bits valid =>
    by_cases hExp : Float.Model.UnpackedFloat.unpackExponent
        (spec := Float.Model.Format.binary32) bits.toBitVec = -1#8
    · have hExp255 : Float.Model.UnpackedFloat.unpackExponent
          (spec := Float.Model.Format.binary32) bits.toBitVec = 255#8 := by
        simpa using hExp
      have hField := (unpackExponent_allOnes_iff bits).1 hExp
      by_cases hMantissa : Float.Model.UnpackedFloat.unpackMantissa
          (spec := Float.Model.Format.binary32) bits.toBitVec = 0#23
      · have hFraction := (unpackMantissa_zero_iff bits).1 hMantissa
        simp [Float32.Model.isNaN, Float32.Model.unpack, modelToIEEE32Exec,
          IEEE32Exec.isNaN, Float.Model.UnpackedFloat.unpack,
          Float.Model.UnpackedFloat.isNaN, hExp255, hMantissa, hField, hFraction]
      · have hFraction : IEEE32Exec.fracField (IEEE32Exec.ofBits bits) ≠ 0 := by
          exact fun h => hMantissa ((unpackMantissa_zero_iff bits).2 h)
        simp [Float32.Model.isNaN, Float32.Model.unpack, modelToIEEE32Exec,
          IEEE32Exec.isNaN, Float.Model.UnpackedFloat.unpack,
          Float.Model.UnpackedFloat.isNaN, hExp255, hMantissa, hField, hFraction]
    · have hExp255 : Float.Model.UnpackedFloat.unpackExponent
          (spec := Float.Model.Format.binary32) bits.toBitVec ≠ 255#8 := by
        simpa using hExp
      have hField : IEEE32Exec.expField (IEEE32Exec.ofBits bits) ≠
          IEEE32Exec.expAllOnes := by
        exact fun h => hExp ((unpackExponent_allOnes_iff bits).2 h)
      simp [Float32.Model.isNaN, Float32.Model.unpack, modelToIEEE32Exec,
        IEEE32Exec.isNaN, Float.Model.UnpackedFloat.unpack, hExp255]
      split_ifs <;> simp [Float.Model.UnpackedFloat.isNaN, hField]

/-- Finiteness agrees between Lean `Float32` and our `IEEE32Exec` encoding. -/
@[simp] theorem float32_isFinite_eq_ieee32 (a : _root_.Float32) :
    Float32.isFinite a = IEEE32Exec.isFinite (toIEEE32Exec a) := by
  exact model_isFinite_eq_ieee32 a.toModel

/-- Infinity detection agrees between Lean `Float32` and `IEEE32Exec`. -/
@[simp] theorem float32_isInf_eq_ieee32 (a : _root_.Float32) :
    Float32.isInf a = IEEE32Exec.isInf (toIEEE32Exec a) := by
  exact model_isInf_eq_ieee32 a.toModel

/-- NaN detection agrees between Lean `Float32` and `IEEE32Exec`, payload and all. -/
@[simp] theorem float32_isNaN_eq_ieee32 (a : _root_.Float32) :
    Float32.isNaN a = IEEE32Exec.isNaN (toIEEE32Exec a) := by
  exact model_isNaN_eq_ieee32 a.toModel

/-! ## Comparisons -/

/-- The shape of a finite binary32 payload: either a subnormal at the minimal exponent `-149`
with a mantissa below `2^23`, or a normal number with a mantissa in `[2^23, 2^24)`.

The comparison lemmas below all need this same case split, so it is named once here instead of
being restated at every use site. -/
private def finiteBounds (mantissa : Nat) (exponent : Int) : Prop :=
  (exponent = -149 ∧ mantissa < 2 ^ 23) ∨
    (-149 ≤ exponent ∧ 2 ^ 23 ≤ mantissa ∧ mantissa < 2 ^ 24)

private lemma finiteBounds_exponent_lower
    {mantissa : Nat} {exponent : Int} (h : finiteBounds mantissa exponent) :
    -149 ≤ exponent := by
  rcases h with ⟨rfl, _⟩ | ⟨h, _⟩ <;> grind

private lemma finiteBounds_mantissa_upper
    {mantissa : Nat} {exponent : Int} (h : finiteBounds mantissa exponent) :
    mantissa < 2 ^ 24 := by
  rcases h with ⟨_, h⟩ | ⟨_, _, h⟩
  · grind
  · exact h

private lemma finiteBounds_mantissa_lower_of_exponent_gt
    {mantissa : Nat} {exponent : Int} (h : finiteBounds mantissa exponent)
    (hexp : -149 < exponent) :
    2 ^ 23 ≤ mantissa := by
  rcases h with ⟨h, _⟩ | ⟨_, h, _⟩
  · grind
  · exact h

private lemma mantissa_lt_shift_of_exponent_lt
    {m₁ m₂ : Nat} {e₁ e₂ : Int}
    (h₁ : finiteBounds m₁ e₁) (h₂ : finiteBounds m₂ e₂)
    (he : e₁ < e₂) :
    m₁ < Nat.shiftLeft m₂ (Int.toNat (e₂ - e₁)) := by
  have he₁ : -149 ≤ e₁ := finiteBounds_exponent_lower h₁
  have he₂ : -149 < e₂ := lt_of_le_of_lt he₁ he
  have hm₁ : m₁ < 2 ^ 24 := finiteBounds_mantissa_upper h₁
  have hm₂ : 2 ^ 23 ≤ m₂ := finiteBounds_mantissa_lower_of_exponent_gt h₂ he₂
  have hshift : 1 ≤ Int.toNat (e₂ - e₁) := by
    grind
  change m₁ < m₂ <<< Int.toNat (e₂ - e₁)
  rw [Nat.shiftLeft_eq]
  have hpow : 2 ^ 1 ≤ 2 ^ Int.toNat (e₂ - e₁) :=
    Nat.pow_le_pow_right (by decide) hshift
  have : 2 ^ 24 ≤ m₂ * 2 ^ Int.toNat (e₂ - e₁) := by
    calc
      2 ^ 24 = 2 ^ 23 * 2 ^ 1 := by norm_num [pow_succ]
      _ ≤ m₂ * 2 ^ Int.toNat (e₂ - e₁) := Nat.mul_le_mul hm₂ hpow
  grind

private lemma compare_int_ofNat (m n : Nat) :
    Ord.compare (Int.ofNat m) (Int.ofNat n) = Ord.compare m n := by
  cases h : Ord.compare m n with
  | lt =>
      rw [show Ord.compare (Int.ofNat m) (Int.ofNat n) = .lt by
        exact compare_lt_iff_lt.mpr (Int.ofNat_lt.mpr (compare_lt_iff_lt.mp h))]
  | eq =>
      rw [show Ord.compare (Int.ofNat m) (Int.ofNat n) = .eq by
        exact compare_eq_iff_eq.mpr (Int.ofNat_inj.mpr (compare_eq_iff_eq.mp h))]
  | gt =>
      rw [show Ord.compare (Int.ofNat m) (Int.ofNat n) = .gt by
        exact compare_gt_iff_gt.mpr (Int.ofNat_lt.mpr (compare_gt_iff_gt.mp h))]

private lemma compare_neg_neg (a b : Int) :
    Ord.compare (-a) (-b) = (Ord.compare a b).swap := by
  cases h : Ord.compare a b with
  | lt =>
      have hab : a < b := compare_lt_iff_lt.mp h
      simp only [Ordering.swap_lt]
      exact compare_gt_iff_gt.mpr (Int.neg_lt_neg hab)
  | eq =>
      have hab : a = b := compare_eq_iff_eq.mp h
      simp [hab]
  | gt =>
      have hab : b < a := compare_gt_iff_gt.mp h
      simp only [Ordering.swap_gt]
      exact compare_lt_iff_lt.mpr (Int.neg_lt_neg hab)

private lemma cmpDyadic_positive_canonical
    {m₁ m₂ : Nat} {e₁ e₂ : Int}
    (hm₁ : 0 < m₁) (hm₂ : 0 < m₂)
    (h₁ : finiteBounds m₁ e₁) (h₂ : finiteBounds m₂ e₂) :
    cmpDyadic { sign := false, mant := m₁, exp := e₁ }
        { sign := false, mant := m₂, exp := e₂ } =
      (compare e₁ e₂).then (compare m₁ m₂) := by
  by_cases heq : e₁ = e₂
  · subst e₂
    simpa [cmpDyadic, hm₁.ne', hm₂.ne'] using compare_int_ofNat m₁ m₂
  · by_cases hlt : e₁ < e₂
    · have hm := mantissa_lt_shift_of_exponent_lt h₁ h₂ hlt
      rw [show (Ord.compare e₁ e₂).then (Ord.compare m₁ m₂) = .lt by
        rw [compare_lt_iff_lt.mpr hlt]
        rfl]
      simpa [cmpDyadic, hm₁.ne', hm₂.ne', le_of_lt hlt] using
        (compare_lt_iff_lt.mpr (Int.ofNat_lt.mpr hm))
    · have hgt : e₂ < e₁ := lt_of_le_of_ne (le_of_not_gt hlt) (Ne.symm heq)
      have hm := mantissa_lt_shift_of_exponent_lt h₂ h₁ hgt
      rw [show (Ord.compare e₁ e₂).then (Ord.compare m₁ m₂) = .gt by
        rw [compare_gt_iff_gt.mpr hgt]
        rfl]
      simpa [cmpDyadic, hm₁.ne', hm₂.ne', not_le.mpr hgt] using
        (compare_gt_iff_gt.mpr (Int.ofNat_lt.mpr hm))

private lemma cmpDyadic_negative_canonical
    {m₁ m₂ : Nat} {e₁ e₂ : Int}
    (hm₁ : 0 < m₁) (hm₂ : 0 < m₂)
    (h₁ : finiteBounds m₁ e₁) (h₂ : finiteBounds m₂ e₂) :
    cmpDyadic { sign := true, mant := m₁, exp := e₁ }
        { sign := true, mant := m₂, exp := e₂ } =
      ((Ord.compare e₁ e₂).then (Ord.compare m₁ m₂)).swap := by
  have hpositive := cmpDyadic_positive_canonical hm₁ hm₂ h₁ h₂
  rw [← hpositive]
  unfold cmpDyadic
  simp [hm₁.ne', hm₂.ne', compare_neg_neg]

private lemma cmpDyadic_negative_positive
    {m₁ m₂ : Nat} {e₁ e₂ : Int} (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) :
    cmpDyadic { sign := true, mant := m₁, exp := e₁ }
        { sign := false, mant := m₂, exp := e₂ } = .lt := by
  have hzero : (m₁ == 0 && m₂ == 0) = false := by simp [hm₁.ne']
  unfold cmpDyadic
  simp only [hzero, Bool.false_eq_true, if_false, if_true]
  apply compare_lt_iff_lt.mpr
  have h₁ : 0 < Nat.shiftLeft m₁
      (Int.toNat (e₁ - if e₁ ≤ e₂ then e₁ else e₂)) :=
    Nat.shiftLeft_pos_iff.mpr hm₁
  have h₂ : 0 < Nat.shiftLeft m₂
      (Int.toNat (e₂ - if e₁ ≤ e₂ then e₁ else e₂)) :=
    Nat.shiftLeft_pos_iff.mpr hm₂
  have h₁' : 0 < Int.ofNat (Nat.shiftLeft m₁
      (Int.toNat (e₁ - if e₁ ≤ e₂ then e₁ else e₂))) := Int.natCast_pos.mpr h₁
  have h₂' : 0 < Int.ofNat (Nat.shiftLeft m₂
      (Int.toNat (e₂ - if e₁ ≤ e₂ then e₁ else e₂))) := Int.natCast_pos.mpr h₂
  grind

private lemma cmpDyadic_positive_negative
    {m₁ m₂ : Nat} {e₁ e₂ : Int} (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) :
    cmpDyadic { sign := false, mant := m₁, exp := e₁ }
        { sign := true, mant := m₂, exp := e₂ } = .gt := by
  have hzero : (m₁ == 0 && m₂ == 0) = false := by simp [hm₁.ne']
  unfold cmpDyadic
  simp only [hzero, Bool.false_eq_true, if_false, if_true]
  apply compare_gt_iff_gt.mpr
  have h₁ : 0 < Nat.shiftLeft m₁
      (Int.toNat (e₁ - if e₁ ≤ e₂ then e₁ else e₂)) :=
    Nat.shiftLeft_pos_iff.mpr hm₁
  have h₂ : 0 < Nat.shiftLeft m₂
      (Int.toNat (e₂ - if e₁ ≤ e₂ then e₁ else e₂)) :=
    Nat.shiftLeft_pos_iff.mpr hm₂
  have h₁' : 0 < Int.ofNat (Nat.shiftLeft m₁
      (Int.toNat (e₁ - if e₁ ≤ e₂ then e₁ else e₂))) := Int.natCast_pos.mpr h₁
  have h₂' : 0 < Int.ofNat (Nat.shiftLeft m₂
      (Int.toNat (e₂ - if e₁ ≤ e₂ then e₁ else e₂))) := Int.natCast_pos.mpr h₂
  grind

private lemma cmpDyadic_zero_finite
    {signZero sign : Bool} {mantissa : Nat} {exponent : Int} (hpos : 0 < mantissa) :
    cmpDyadic { sign := signZero, mant := 0, exp := 0 }
        { sign := sign, mant := mantissa, exp := exponent } =
      if sign then .gt else .lt := by
  have hzero : (0 == 0 && mantissa == 0) = false := by simp [hpos.ne']
  unfold cmpDyadic
  simp only [hzero, Bool.false_eq_true, if_false]
  have hshiftInt' : 0 < Int.ofNat mantissa <<<
      (Int.toNat (exponent - if 0 ≤ exponent then 0 else exponent)) := by
    rw [Int.shiftLeft_eq']
    exact Int.mul_pos (Int.natCast_pos.mpr hpos)
      (Int.natCast_pos.mpr (Nat.two_pow_pos _))
  have hneg : -(Int.ofNat mantissa <<<
      (Int.toNat (exponent - if 0 ≤ exponent then 0 else exponent))) < 0 :=
    neg_lt_zero.mpr hshiftInt'
  cases signZero <;> cases sign <;> simp
  · exact compare_lt_iff_lt.mpr hshiftInt'
  · exact compare_gt_iff_gt.mpr hneg
  · exact compare_lt_iff_lt.mpr hshiftInt'
  · exact compare_gt_iff_gt.mpr hneg

private lemma cmpDyadic_finite_zero
    {sign signZero : Bool} {mantissa : Nat} {exponent : Int} (hpos : 0 < mantissa) :
    cmpDyadic { sign := sign, mant := mantissa, exp := exponent }
        { sign := signZero, mant := 0, exp := 0 } =
      if sign then .lt else .gt := by
  have hzero : (mantissa == 0 && 0 == 0) = false := by simp [hpos.ne']
  unfold cmpDyadic
  simp only [hzero, Bool.false_eq_true, if_false]
  have hshiftInt' : 0 < Int.ofNat mantissa <<<
      (Int.toNat (exponent - if exponent ≤ 0 then exponent else 0)) := by
    rw [Int.shiftLeft_eq']
    exact Int.mul_pos (Int.natCast_pos.mpr hpos)
      (Int.natCast_pos.mpr (Nat.two_pow_pos _))
  have hneg : -(Int.ofNat mantissa <<<
      (Int.toNat (exponent - if exponent ≤ 0 then exponent else 0))) < 0 :=
    neg_lt_zero.mpr hshiftInt'
  cases signZero <;> cases sign <;> simp
  · exact compare_gt_iff_gt.mpr hshiftInt'
  · exact compare_lt_iff_lt.mpr hneg
  · exact compare_gt_iff_gt.mpr hshiftInt'
  · exact compare_lt_iff_lt.mpr hneg

@[simp] private lemma negInf_isNaN : IEEE32Exec.isNaN IEEE32Exec.negInf = false := by decide
@[simp] private lemma posInf_isNaN : IEEE32Exec.isNaN IEEE32Exec.posInf = false := by decide
@[simp] private lemma negInf_isInf : IEEE32Exec.isInf IEEE32Exec.negInf = true := by decide
@[simp] private lemma posInf_isInf : IEEE32Exec.isInf IEEE32Exec.posInf = true := by decide
@[simp] private lemma negInf_signBit : IEEE32Exec.signBit IEEE32Exec.negInf = true := by decide
@[simp] private lemma posInf_signBit : IEEE32Exec.signBit IEEE32Exec.posInf = false := by decide

/-- Model comparison and `IEEE32Exec.compare` agree, including the unordered NaN case.

Comparison is where the two developments are most easily out of step, since IEEE 754 asks for a
partial order rather than a total one; carrying `Option Ordering` on both sides is what makes the
agreement statable at all. The three user-facing operators below are corollaries. -/
theorem model_compare_eq_ieee32 (a b : Float32.Model) :
    Float32.Model.compare a b =
      IEEE32Exec.compare (modelToIEEE32Exec a) (modelToIEEE32Exec b) := by
  cases ha : a.unpack with
  | notANumber =>
      rw [Float32.Model.compare, ha,
        modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq a ha]
      have hNaN : IEEE32Exec.isNaN IEEE32Exec.canonicalNaN = true := by decide
      simp [Float.Model.UnpackedFloat.compare, IEEE32Exec.compare, hNaN]
  | infinity sa =>
      cases hb : b.unpack with
      | notANumber =>
          rw [Float32.Model.compare, ha, hb,
            modelToIEEE32Exec_eq_inf_of_unpack_eq a sa ha,
            modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb]
          cases sa <;> decide
      | infinity sb =>
          rw [Float32.Model.compare, ha, hb,
            modelToIEEE32Exec_eq_inf_of_unpack_eq a sa ha,
            modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb]
          cases sa <;> cases sb <;> decide
      | zero sb =>
          rw [Float32.Model.compare, ha, hb,
            modelToIEEE32Exec_eq_inf_of_unpack_eq a sa ha,
            modelToIEEE32Exec_eq_zero_of_unpack_eq b sb hb]
          cases sa <;> cases sb <;> decide
      | finite sb mb eb hbpos =>
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          have hbNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some hbExec
          have hbInf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some hbExec
          rw [Float32.Model.compare, ha, hb,
            modelToIEEE32Exec_eq_inf_of_unpack_eq a sa ha]
          cases sa <;>
            simp [Float.Model.UnpackedFloat.compare, signToBool,
              IEEE32Exec.compare, hbNaN, hbInf]
  | zero sa =>
      cases hb : b.unpack with
      | notANumber =>
          rw [Float32.Model.compare, ha, hb,
            modelToIEEE32Exec_eq_zero_of_unpack_eq a sa ha,
            modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb]
          cases sa <;> decide
      | infinity sb =>
          rw [Float32.Model.compare, ha, hb,
            modelToIEEE32Exec_eq_zero_of_unpack_eq a sa ha,
            modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb]
          cases sa <;> cases sb <;> decide
      | zero sb =>
          rw [Float32.Model.compare, ha, hb,
            modelToIEEE32Exec_eq_zero_of_unpack_eq a sa ha,
            modelToIEEE32Exec_eq_zero_of_unpack_eq b sb hb]
          cases sa <;> cases sb <;> decide
      | finite sb mb eb hbpos =>
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          have haExec : IEEE32Exec.toDyadic? (modelToIEEE32Exec a) = some
              { sign := signToBool sa, mant := 0, exp := 0 } := by
            rw [modelToIEEE32Exec_eq_zero_of_unpack_eq a sa ha]
            cases sa <;> decide
          have hcmp := IEEE32Exec.compare_eq_some_cmpDyadic_of_toDyadic?
            (modelToIEEE32Exec a) (modelToIEEE32Exec b) haExec hbExec
          rw [Float32.Model.compare, ha, hb, hcmp]
          cases sa <;> cases sb <;>
            simp [Float.Model.UnpackedFloat.compare, signToBool,
              cmpDyadic_zero_finite hbpos]
  | finite sa ma ea hapos =>
      have haExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
        a sa ma ea hapos ha
      cases hb : b.unpack with
      | notANumber =>
          rw [Float32.Model.compare, ha, hb,
            modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb]
          have haNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some haExec
          have hNaN : IEEE32Exec.isNaN IEEE32Exec.canonicalNaN = true := by decide
          simp [Float.Model.UnpackedFloat.compare, IEEE32Exec.compare, haNaN, hNaN]
      | infinity sb =>
          have haNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some haExec
          have haInf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some haExec
          rw [Float32.Model.compare, ha, hb,
            modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb]
          cases sa <;> cases sb <;>
            simp [Float.Model.UnpackedFloat.compare, signToBool,
              IEEE32Exec.compare, haNaN, haInf]
      | zero sb =>
          have hbExec : IEEE32Exec.toDyadic? (modelToIEEE32Exec b) = some
              { sign := signToBool sb, mant := 0, exp := 0 } := by
            rw [modelToIEEE32Exec_eq_zero_of_unpack_eq b sb hb]
            cases sb <;> decide
          have hcmp := IEEE32Exec.compare_eq_some_cmpDyadic_of_toDyadic?
            (modelToIEEE32Exec a) (modelToIEEE32Exec b) haExec hbExec
          rw [Float32.Model.compare, ha, hb, hcmp]
          cases sa <;> cases sb <;>
            simp [Float.Model.UnpackedFloat.compare, signToBool,
              cmpDyadic_finite_zero hapos]
      | finite sb mb eb hbpos =>
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          have hcmp := IEEE32Exec.compare_eq_some_cmpDyadic_of_toDyadic?
            (modelToIEEE32Exec a) (modelToIEEE32Exec b) haExec hbExec
          have habounds := unpack_finite_bounds a sa ma ea hapos ha
          have hbbounds := unpack_finite_bounds b sb mb eb hbpos hb
          rw [Float32.Model.compare, ha, hb, hcmp]
          cases sa <;> cases sb
          · simpa [Float.Model.UnpackedFloat.compare, signToBool] using congrArg some
              (cmpDyadic_negative_canonical hapos hbpos habounds hbbounds).symm
          · simpa [Float.Model.UnpackedFloat.compare, signToBool] using congrArg some
              (cmpDyadic_negative_positive hapos hbpos).symm
          · simpa [Float.Model.UnpackedFloat.compare, signToBool] using congrArg some
              (cmpDyadic_positive_negative hapos hbpos).symm
          · simpa [Float.Model.UnpackedFloat.compare, signToBool] using congrArg some
              (cmpDyadic_positive_canonical hapos hbpos habounds hbbounds).symm

/-- Strict less-than in terms of `IEEE32Exec.compare`. -/
theorem model_lt_eq_ieee32 (a b : Float32.Model) :
    Float32.Model.lt a b =
      (IEEE32Exec.compare (modelToIEEE32Exec a) (modelToIEEE32Exec b) == some .lt) := by
  simp only [Float32.Model.lt, Float.Model.UnpackedFloat.lt]
  rw [← Float32.Model.compare, model_compare_eq_ieee32]

/-- Less-or-equal in terms of `IEEE32Exec.compare`; `Option.any` makes an unordered pair false. -/
theorem model_le_eq_ieee32 (a b : Float32.Model) :
    Float32.Model.le a b =
      (IEEE32Exec.compare (modelToIEEE32Exec a) (modelToIEEE32Exec b)).any Ordering.isLE := by
  simp only [Float32.Model.le, Float.Model.UnpackedFloat.le]
  rw [← Float32.Model.compare, model_compare_eq_ieee32]

/-- Float equality in terms of `IEEE32Exec.compare`, so `NaN ≠ NaN` and `-0.0 = 0.0` both fall out
of the comparison rather than being special cased. -/
theorem model_beq_eq_ieee32 (a b : Float32.Model) :
    Float32.Model.beq a b =
      (IEEE32Exec.compare (modelToIEEE32Exec a) (modelToIEEE32Exec b) == some .eq) := by
  simp only [Float32.Model.beq, Float.Model.UnpackedFloat.beq]
  rw [← Float32.Model.compare, model_compare_eq_ieee32]

/-- `Float32.lt` in terms of `IEEE32Exec.compare`. -/
theorem float32_lt_eq_ieee32 (a b : _root_.Float32) :
    Float32.lt a b =
      (IEEE32Exec.compare (toIEEE32Exec a) (toIEEE32Exec b) == some .lt) := by
  rw [lt_eq_model, model_lt_eq_ieee32]
  rfl

/-- `Float32.le` in terms of `IEEE32Exec.compare`. -/
theorem float32_le_eq_ieee32 (a b : _root_.Float32) :
    Float32.le a b =
      (IEEE32Exec.compare (toIEEE32Exec a) (toIEEE32Exec b)).any Ordering.isLE := by
  rw [le_eq_model, model_le_eq_ieee32]
  rfl

/-- `Float32.beq` in terms of `IEEE32Exec.compare`. -/
theorem float32_beq_eq_ieee32 (a b : _root_.Float32) :
    Float32.beq a b =
      (IEEE32Exec.compare (toIEEE32Exec a) (toIEEE32Exec b) == some .eq) := by
  rw [beq_eq_model, model_beq_eq_ieee32]
  rfl

/-! ## Logical arithmetic -/

open Float.Model
open Float.Model.UnpackedFloat

private theorem quietNaN_canonicalNaN :
    IEEE32Exec.quietNaN IEEE32Exec.canonicalNaN = IEEE32Exec.canonicalNaN := by
  decide

private theorem isNaN_eq_false_of_isInf_eq_true
    (x : IEEE32Exec) (hx : IEEE32Exec.isInf x = true) :
    IEEE32Exec.isNaN x = false := by
  simp only [IEEE32Exec.isInf, Bool.and_eq_true] at hx
  have hxExp : IEEE32Exec.expField x = IEEE32Exec.expAllOnes := eq_of_beq hx.1
  have hxFrac : IEEE32Exec.fracField x = 0 := eq_of_beq hx.2
  simp [IEEE32Exec.isNaN, hxExp, hxFrac]

/-! ## Addition -/

private theorem add_eq_left_of_isInf_left
    (x y : IEEE32Exec) (hxInf : IEEE32Exec.isInf x = true)
    (hyNaN : IEEE32Exec.isNaN y = false) (hyInf : IEEE32Exec.isInf y = false) :
    IEEE32Exec.add x y = x := by
  have hxNaN := isNaN_eq_false_of_isInf_eq_true x hxInf
  have hchoose := IEEE32Exec.chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
  simp [IEEE32Exec.add, hchoose, hxInf, hyInf]

private theorem add_eq_right_of_isInf_right
    (x y : IEEE32Exec) (hyInf : IEEE32Exec.isInf y = true)
    (hxNaN : IEEE32Exec.isNaN x = false) (hxInf : IEEE32Exec.isInf x = false) :
    IEEE32Exec.add x y = y := by
  have hyNaN := isNaN_eq_false_of_isInf_eq_true y hyInf
  have hchoose := IEEE32Exec.chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
  simp [IEEE32Exec.add, hchoose, hxInf, hyInf]

private theorem sign_apply_eq (sign : Sign) (mantissa : Nat) :
    sign.apply mantissa =
      if signToBool sign then -(Int.ofNat mantissa) else Int.ofNat mantissa := by
  cases sign <;> rfl

private theorem sign_apply_int_eq (sign : Sign) (mantissa : Int) :
    sign.apply mantissa =
      if signToBool sign then -mantissa else mantissa := by
  cases sign <;> rfl

private theorem model_pack_normalize_eq_roundDyadic
    (mantissa : Int) (exponent : Int) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (normalize Format.binary32 mantissa exponent .positive)) =
      roundDyadicToIEEE32
        { sign := mantissa < 0, mant := mantissa.natAbs, exp := exponent } := by
  cases hcmp : compare mantissa 0 with
  | lt =>
      have hneg : mantissa < 0 := Int.compare_eq_lt.mp hcmp
      simp only [Float.Model.UnpackedFloat.normalize, hcmp]
      rw [model_round_exact_eq_roundDyadic]
      congr 2
      · simp [signToBool, hneg]
      · grind
  | eq =>
      have hzero : mantissa = 0 := Int.compare_eq_eq.mp hcmp
      subst mantissa
      simp [Float.Model.UnpackedFloat.normalize, modelToIEEE32Exec_pack_zero,
        roundDyadicToIEEE32, signToBool]
  | gt =>
      have hpos : 0 < mantissa := Int.compare_eq_gt.mp hcmp
      simp only [Float.Model.UnpackedFloat.normalize, hcmp]
      rw [model_round_exact_eq_roundDyadic]
      congr 2
      · have hnot : ¬ mantissa < 0 := by grind
        simp [signToBool, hnot]
      · grind

private theorem signToBool_and_eq_false_of_apply_add_eq_zero
    (sa sb : Sign) (ma mb : Nat) (hma : 0 < ma) (hmb : 0 < mb)
    (hsum : sa.apply (Int.ofNat ma) + sb.apply (Int.ofNat mb) = 0) :
    (signToBool sa && signToBool sb) = false := by
  cases sa <;> cases sb
  · exfalso
    have hmaInt : (0 : Int) < Int.ofNat ma := Int.natCast_pos.mpr hma
    have hmbInt : (0 : Int) < Int.ofNat mb := Int.natCast_pos.mpr hmb
    simp only [Float.Model.UnpackedFloat.Sign.apply] at hsum
    grind
  all_goals rfl

private theorem roundDyadic_normalized_eq_zeroBranch
    (sum : Int) (exponent : Int) (zeroSign : Bool)
    (hzeroSign : sum = 0 → zeroSign = false) :
    roundDyadicToIEEE32
        { sign := decide (sum < 0), mant := sum.natAbs, exp := exponent } =
      roundDyadicToIEEE32
        (if sum == 0 then { sign := zeroSign, mant := 0, exp := 0 }
        else { sign := decide (sum < 0), mant := sum.natAbs, exp := exponent }) := by
  by_cases hsum : sum = 0
  · subst sum
    simp [hzeroSign rfl, roundDyadicToIEEE32]
  · simp [(beq_eq_false_iff_ne).2 hsum]

private theorem model_add_finite_eq_roundDyadic
    (sa sb : Sign) (ma mb : Nat) (ea eb : Int)
    (hma : 0 < ma) (hmb : 0 < mb) :
    modelToIEEE32Exec
        (Float32.Model.pack
          (UnpackedFloat.add Format.binary32
            (.finite sa ma ea hma) (.finite sb mb eb hmb))) =
      roundDyadicToIEEE32
        (addDyadic
          { sign := signToBool sa, mant := ma, exp := ea }
          { sign := signToBool sb, mant := mb, exp := eb }) := by
  simp only [UnpackedFloat.add]
  rw [model_pack_normalize_eq_roundDyadic]
  by_cases hab : ea ≤ eb
  · let shift := (eb - ea).toNat
    let mbAligned := Nat.shiftLeft mb shift
    have hmbAligned : 0 < mbAligned := Nat.shiftLeft_pos_iff.mpr hmb
    have hzero :
        sa.apply (Int.ofNat ma) + sb.apply (Int.ofNat mbAligned) = 0 →
          (signToBool sa && signToBool sb) = false :=
      signToBool_and_eq_false_of_apply_add_eq_zero
        sa sb ma mbAligned hma hmbAligned
    simpa [addDyadic, hma.ne', hmb.ne', hab, decreaseExponent, min_eq_left hab,
      shift, mbAligned, sign_apply_eq, sign_apply_int_eq] using
        roundDyadic_normalized_eq_zeroBranch
          (sa.apply (Int.ofNat ma) + sb.apply (Int.ofNat mbAligned)) ea
          (signToBool sa && signToBool sb) hzero
  · have hba : eb < ea := lt_of_not_ge hab
    let shift := (ea - eb).toNat
    let maAligned := Nat.shiftLeft ma shift
    have hmaAligned : 0 < maAligned := Nat.shiftLeft_pos_iff.mpr hma
    have hzero :
        sa.apply (Int.ofNat maAligned) + sb.apply (Int.ofNat mb) = 0 →
          (signToBool sa && signToBool sb) = false :=
      signToBool_and_eq_false_of_apply_add_eq_zero
        sa sb maAligned mb hmaAligned hmb
    simpa [addDyadic, hma.ne', hmb.ne', hab, decreaseExponent,
      min_eq_right (le_of_lt hba), shift, maAligned, sign_apply_eq, sign_apply_int_eq] using
        roundDyadic_normalized_eq_zeroBranch
          (sa.apply (Int.ofNat maAligned) + sb.apply (Int.ofNat mb)) eb
          (signToBool sa && signToBool sb) hzero

/--
Lean's logical binary32 addition and `IEEE32Exec.add` agree on every canonical operand.

The finite branch identifies both algorithms with exact dyadic addition followed by one
round-to-nearest-even step. The other branches prove the IEEE rules for NaNs, infinities, and
signed zeros.
-/
theorem model_add_eq_ieee32 (a b : Float32.Model) :
    modelToIEEE32Exec (Float32.Model.add a b) =
      IEEE32Exec.add (modelToIEEE32Exec a) (modelToIEEE32Exec b) := by
  unfold Float32.Model.add
  cases ha : a.unpack with
  | notANumber =>
      have haExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq a ha
      rw [haExec]
      have hchoose : IEEE32Exec.chooseNaN2 IEEE32Exec.canonicalNaN
          (modelToIEEE32Exec b) = some IEEE32Exec.canonicalNaN := by
        simpa [quietNaN_canonicalNaN] using
          IEEE32Exec.chooseNaN2_of_isNaN_left
            IEEE32Exec.canonicalNaN (modelToIEEE32Exec b)
            (by decide) (model_isSNaN_eq_false b) (by decide)
      rw [IEEE32Exec.add_eq_of_chooseNaN2_some _ _ _ hchoose]
      simp [UnpackedFloat.add]
  | infinity sa =>
      have haExec := modelToIEEE32Exec_eq_inf_of_unpack_eq a sa ha
      cases hb : b.unpack with
      | notANumber =>
          have hbExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb
          rw [haExec, hbExec]
          cases sa <;> decide
      | infinity sb =>
          have hbExec := modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb
          rw [haExec, hbExec]
          cases sa <;> cases sb <;> decide
      | zero sb =>
          have hbExec := modelToIEEE32Exec_eq_zero_of_unpack_eq b sb hb
          rw [haExec, hbExec]
          cases sa <;> cases sb <;> decide
      | finite sb mb eb hbpos =>
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          have hbNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some hbExec
          have hbInf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some hbExec
          rw [haExec]
          simp only [UnpackedFloat.add, modelToIEEE32Exec_pack_infinity]
          rw [add_eq_left_of_isInf_left _ _ (by cases sa <;> decide) hbNaN hbInf]
  | zero sa =>
      have haExec := modelToIEEE32Exec_eq_zero_of_unpack_eq a sa ha
      have haDyadic : IEEE32Exec.toDyadic? (modelToIEEE32Exec a) = some
          { sign := signToBool sa, mant := 0, exp := 0 } := by
        rw [toDyadic_modelToIEEE32Exec, ha]
        rfl
      cases hb : b.unpack with
      | notANumber =>
          have hbExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb
          rw [haExec, hbExec]
          cases sa <;> decide
      | infinity sb =>
          have hbExec := modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb
          rw [haExec, hbExec]
          cases sa <;> cases sb <;> decide
      | zero sb =>
          have hbExec := modelToIEEE32Exec_eq_zero_of_unpack_eq b sb hb
          have hbDyadic : IEEE32Exec.toDyadic? (modelToIEEE32Exec b) = some
              { sign := signToBool sb, mant := 0, exp := 0 } := by
            rw [toDyadic_modelToIEEE32Exec, hb]
            rfl
          rw [IEEE32Exec.add_eq_roundDyadicToIEEE32_of_toDyadic? haDyadic hbDyadic]
          simp only [UnpackedFloat.add]
          cases sa <;> cases sb <;> decide
      | finite sb mb eb hbpos =>
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          rw [IEEE32Exec.add_eq_roundDyadicToIEEE32_of_toDyadic? haDyadic hbExec]
          simp only [UnpackedFloat.add]
          rw [show Float32.Model.pack (.finite sb mb eb hbpos) = b by
            rw [← hb, model_pack_unpack]]
          have hadd : IEEE32Exec.addDyadic
              { sign := signToBool sa, mant := 0, exp := 0 }
              { sign := signToBool sb, mant := mb, exp := eb } =
              { sign := signToBool sb, mant := mb, exp := eb } := by
            cases sa <;> cases sb <;>
              simp [IEEE32Exec.addDyadic, signToBool, hbpos.ne']
          rw [hadd, IEEE32Exec.roundDyadicToIEEE32_of_toDyadic?_some hbExec]
  | finite sa ma ea hapos =>
      have haExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
        a sa ma ea hapos ha
      cases hb : b.unpack with
      | notANumber =>
          have hbExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb
          rw [hbExec]
          have hchoose : IEEE32Exec.chooseNaN2 (modelToIEEE32Exec a)
              IEEE32Exec.canonicalNaN = some IEEE32Exec.canonicalNaN := by
            simpa [quietNaN_canonicalNaN] using
              IEEE32Exec.chooseNaN2_of_isNaN_right
                (modelToIEEE32Exec a) IEEE32Exec.canonicalNaN
                (model_isSNaN_eq_false a) (by decide)
                (IEEE32Exec.isNaN_eq_false_of_toDyadic?_some haExec) (by decide)
          rw [IEEE32Exec.add_eq_of_chooseNaN2_some _ _ _ hchoose]
          simp [UnpackedFloat.add]
      | infinity sb =>
          have hbExec := modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb
          have haNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some haExec
          have haInf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some haExec
          rw [hbExec]
          simp only [UnpackedFloat.add, modelToIEEE32Exec_pack_infinity]
          rw [add_eq_right_of_isInf_right _ _ (by cases sb <;> decide) haNaN haInf]
      | zero sb =>
          have hbDyadic : IEEE32Exec.toDyadic? (modelToIEEE32Exec b) = some
              { sign := signToBool sb, mant := 0, exp := 0 } := by
            rw [toDyadic_modelToIEEE32Exec, hb]
            rfl
          rw [IEEE32Exec.add_eq_roundDyadicToIEEE32_of_toDyadic? haExec hbDyadic]
          simp only [UnpackedFloat.add]
          rw [show Float32.Model.pack (.finite sa ma ea hapos) = a by
            rw [← ha, model_pack_unpack]]
          have hadd : IEEE32Exec.addDyadic
              { sign := signToBool sa, mant := ma, exp := ea }
              { sign := signToBool sb, mant := 0, exp := 0 } =
              { sign := signToBool sa, mant := ma, exp := ea } := by
            cases sa <;> cases sb <;>
              simp [IEEE32Exec.addDyadic, signToBool, hapos.ne']
          rw [hadd, IEEE32Exec.roundDyadicToIEEE32_of_toDyadic?_some haExec]
      | finite sb mb eb hbpos =>
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          rw [IEEE32Exec.add_eq_roundDyadicToIEEE32_of_toDyadic? haExec hbExec]
          exact model_add_finite_eq_roundDyadic sa sb ma mb ea eb hapos hbpos

/-! ## Multiplication -/

private theorem mul_eq_signedInf_of_isInf_left
    (x y : IEEE32Exec) (hx : IEEE32Exec.isInf x = true)
    (hyNaN : IEEE32Exec.isNaN y = false) (hyZero : IEEE32Exec.isZero y = false) :
    IEEE32Exec.mul x y =
      if IEEE32Exec.signBit x != IEEE32Exec.signBit y then
        IEEE32Exec.negInf
      else
        IEEE32Exec.posInf := by
  have hxNaN := isNaN_eq_false_of_isInf_eq_true x hx
  have hchoose := IEEE32Exec.chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
  simp [IEEE32Exec.mul, hchoose, hx, hyZero]

private theorem mul_eq_signedInf_of_isInf_right
    (x y : IEEE32Exec) (hy : IEEE32Exec.isInf y = true)
    (hxNaN : IEEE32Exec.isNaN x = false) (hxInf : IEEE32Exec.isInf x = false)
    (hxZero : IEEE32Exec.isZero x = false) :
    IEEE32Exec.mul x y =
      if IEEE32Exec.signBit x != IEEE32Exec.signBit y then
        IEEE32Exec.negInf
      else
        IEEE32Exec.posInf := by
  have hyNaN := isNaN_eq_false_of_isInf_eq_true y hy
  have hchoose := IEEE32Exec.chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
  simp [IEEE32Exec.mul, hchoose, hxInf, hy, hxZero]

private theorem mul_exponent_le_targetExponent_of_finite_bounds
    (m₁ m₂ : Nat) (e₁ e₂ : Int) (hm₁ : 0 < m₁) (hm₂ : 0 < m₂)
    (h₁ : (e₁ = -149 ∧ m₁ < 2 ^ 23) ∨
      (-149 ≤ e₁ ∧ 2 ^ 23 ≤ m₁ ∧ m₁ < 2 ^ 24))
    (h₂ : (e₂ = -149 ∧ m₂ < 2 ^ 23) ∨
      (-149 ≤ e₂ ∧ 2 ^ 23 ≤ m₂ ∧ m₂ < 2 ^ 24)) :
    e₁ + e₂ ≤ Format.binary32.targetExponent
      (totalExponent (m₁ * m₂) (e₁ + e₂)) := by
  rcases h₁ with h₁ | h₁ <;> rcases h₂ with h₂ | h₂
  · unfold Format.targetExponent totalExponent
    simp only [Format.mantissaBits, Format.minExponent, Nat.reduceSub]
    grind
  all_goals
    have hprod : 0 < m₁ * m₂ := Nat.mul_pos hm₁ hm₂
    have hpow : 2 ^ 23 ≤ m₁ * m₂ := by
      first
      | exact h₁.2.1.trans (Nat.le_mul_of_pos_right m₁ hm₂)
      | exact h₂.2.1.trans (Nat.le_mul_of_pos_left m₂ hm₁)
    have hlog : 23 ≤ (m₁ * m₂).log2 :=
      (Nat.le_log2 (Nat.ne_of_gt hprod)).2 hpow
    unfold Format.targetExponent totalExponent
    simp only [Format.mantissaBits, Format.minExponent, Nat.reduceSub]
    grind

/--
Lean's logical binary32 multiplication and `IEEE32Exec.mul` agree on every canonical operand.

The proof covers NaNs, both signed infinities, both signed zeros, subnormals, normal values,
underflow, and overflow. In the finite case both algorithms compute the same exact dyadic product;
`model_roundWithAccuracy_exact_eq_roundDyadic_of_le_targetExponent` then identifies their final
nearest-even rounding step.
-/
theorem model_mul_eq_ieee32 (a b : Float32.Model) :
    modelToIEEE32Exec (Float32.Model.mul a b) =
      IEEE32Exec.mul (modelToIEEE32Exec a) (modelToIEEE32Exec b) := by
  unfold Float32.Model.mul
  cases ha : a.unpack with
  | notANumber =>
      have haExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq a ha
      rw [haExec]
      have hchoose : IEEE32Exec.chooseNaN2 IEEE32Exec.canonicalNaN
          (modelToIEEE32Exec b) = some IEEE32Exec.canonicalNaN := by
        simpa [quietNaN_canonicalNaN] using
          IEEE32Exec.chooseNaN2_of_isNaN_left
            IEEE32Exec.canonicalNaN (modelToIEEE32Exec b)
            (by decide) (model_isSNaN_eq_false b) (by decide)
      rw [IEEE32Exec.mul_eq_of_chooseNaN2_some _ _ _ hchoose]
      simp [UnpackedFloat.mul]
  | infinity sa =>
      have haExec := modelToIEEE32Exec_eq_inf_of_unpack_eq a sa ha
      cases hb : b.unpack with
      | notANumber =>
          have hbExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb
          rw [haExec, hbExec]
          cases sa <;> decide
      | infinity sb =>
          have hbExec := modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb
          rw [haExec, hbExec]
          cases sa <;> cases sb <;> decide
      | zero sb =>
          have hbExec := modelToIEEE32Exec_eq_zero_of_unpack_eq b sb hb
          rw [haExec, hbExec]
          cases sa <;> cases sb <;> decide
      | finite sb mb eb hbpos =>
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          have hbNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some hbExec
          have hbZero :=
            IEEE32Exec.isZero_eq_false_of_toDyadic?_some_of_mant_ne_zero hbExec
              (Nat.ne_of_gt hbpos)
          have hbSign := (IEEE32Exec.sign_eq_signBit_of_toDyadic?_some hbExec).symm
          have hmul := mul_eq_signedInf_of_isInf_left
            (modelToIEEE32Exec a) (modelToIEEE32Exec b)
            (by rw [haExec]; cases sa <;> decide) hbNaN hbZero
          simp only [UnpackedFloat.mul]
          rw [modelToIEEE32Exec_pack_infinity, hmul, haExec, hbSign]
          cases sa <;> cases sb <;> rfl
  | zero sa =>
      have haExec := modelToIEEE32Exec_eq_zero_of_unpack_eq a sa ha
      cases hb : b.unpack with
      | notANumber =>
          have hbExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb
          rw [haExec, hbExec]
          cases sa <;> decide
      | infinity sb =>
          have hbExec := modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb
          rw [haExec, hbExec]
          cases sa <;> cases sb <;> decide
      | zero sb =>
          have hbExec := modelToIEEE32Exec_eq_zero_of_unpack_eq b sb hb
          rw [haExec, hbExec]
          cases sa <;> cases sb <;> decide
      | finite sb mb eb hbpos =>
          have haDyadic : IEEE32Exec.toDyadic? (modelToIEEE32Exec a) = some
              { sign := signToBool sa, mant := 0, exp := 0 } := by
            rw [toDyadic_modelToIEEE32Exec, ha]
            rfl
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          rw [IEEE32Exec.mul_eq_roundDyadicToIEEE32_of_toDyadic? haDyadic hbExec]
          simp only [UnpackedFloat.mul]
          rw [modelToIEEE32Exec_pack_zero]
          simp [IEEE32Exec.roundDyadicToIEEE32]
          cases sa <;> cases sb <;> rfl
  | finite sa ma ea hapos =>
      have haExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
        a sa ma ea hapos ha
      cases hb : b.unpack with
      | notANumber =>
          have hbExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb
          rw [hbExec]
          have hchoose : IEEE32Exec.chooseNaN2 (modelToIEEE32Exec a)
              IEEE32Exec.canonicalNaN = some IEEE32Exec.canonicalNaN := by
            simpa [quietNaN_canonicalNaN] using
              IEEE32Exec.chooseNaN2_of_isNaN_right
                (modelToIEEE32Exec a) IEEE32Exec.canonicalNaN
                (model_isSNaN_eq_false a) (by decide)
                (IEEE32Exec.isNaN_eq_false_of_toDyadic?_some haExec) (by decide)
          rw [IEEE32Exec.mul_eq_of_chooseNaN2_some _ _ _ hchoose]
          simp [UnpackedFloat.mul]
      | infinity sb =>
          have hbExec := modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb
          have haNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some haExec
          have haInf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some haExec
          have haZero :=
            IEEE32Exec.isZero_eq_false_of_toDyadic?_some_of_mant_ne_zero haExec
              (Nat.ne_of_gt hapos)
          have haSign := (IEEE32Exec.sign_eq_signBit_of_toDyadic?_some haExec).symm
          have hmul := mul_eq_signedInf_of_isInf_right
            (modelToIEEE32Exec a) (modelToIEEE32Exec b)
            (by rw [hbExec]; cases sb <;> decide) haNaN haInf haZero
          simp only [UnpackedFloat.mul]
          rw [modelToIEEE32Exec_pack_infinity, hmul, haSign, hbExec]
          cases sa <;> cases sb <;> rfl
      | zero sb =>
          have hbDyadic : IEEE32Exec.toDyadic? (modelToIEEE32Exec b) = some
              { sign := signToBool sb, mant := 0, exp := 0 } := by
            rw [toDyadic_modelToIEEE32Exec, hb]
            rfl
          rw [IEEE32Exec.mul_eq_roundDyadicToIEEE32_of_toDyadic? haExec hbDyadic]
          simp only [UnpackedFloat.mul]
          rw [modelToIEEE32Exec_pack_zero]
          simp [IEEE32Exec.roundDyadicToIEEE32]
          cases sa <;> cases sb <;> rfl
      | finite sb mb eb hbpos =>
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          have habounds := unpack_finite_bounds a sa ma ea hapos ha
          have hbbounds := unpack_finite_bounds b sb mb eb hbpos hb
          have hexp := mul_exponent_le_targetExponent_of_finite_bounds
            ma mb ea eb hapos hbpos habounds hbbounds
          simp only [UnpackedFloat.mul]
          rw [model_roundWithAccuracy_exact_eq_roundDyadic_of_le_targetExponent
            (sa * sb) (ma * mb) (ea + eb) hexp]
          rw [IEEE32Exec.mul_eq_roundDyadicToIEEE32_of_toDyadic? haExec hbExec]
          cases sa <;> cases sb <;> rfl

/-! ## Division -/

/--
The exponent selected by Lean's binary32 `divCore` is low enough for its subsequent
`roundWithAccuracy` call. Above the subnormal floor, the numerator shift exposes at least 24
quotient bits; at or below the floor, the format target itself supplies the inequality.
-/
private theorem divCore_target_le_targetExponent
    (m₁ m₂ : Nat) (e₁ e₂ : Int)
    (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) (hm₁High : m₁ < 2 ^ 24) :
    let exponent := e₁ - e₂
    let target := min exponent
      (Format.binary32.targetExponent
        (totalExponent m₁ e₁ - totalExponent m₂ e₂))
    let shift := (exponent - target).toNat
    let numerator := m₁ <<< shift
    target ≤ Format.binary32.targetExponent
      (totalExponent (numerator / m₂) target) := by
  dsimp only
  let exponent := e₁ - e₂
  let roundTarget := Format.binary32.targetExponent
    (totalExponent m₁ e₁ - totalExponent m₂ e₂)
  let target := min exponent roundTarget
  let shift := (exponent - target).toNat
  let numerator := m₁ <<< shift
  have htargetLe : target ≤ exponent := min_le_left _ _
  have hshift : (shift : Int) = exponent - target :=
    Int.toNat_of_nonneg (sub_nonneg.mpr htargetLe)
  by_cases hfloor : target ≤ -149
  · unfold Format.targetExponent
    simp only [Format.mantissaBits, Format.minExponent, Nat.reduceSub]
    exact hfloor.trans (le_max_right _ _)
  · have hm₁LogHigh : m₁.log2 < 24 := (Nat.log2_lt hm₁.ne').2 hm₁High
    have hroundTarget : roundTarget =
        (m₁.log2 : Int) - (m₂.log2 : Int) + exponent - 24 := by
      unfold roundTarget Format.targetExponent totalExponent
      simp only [Format.mantissaBits, Format.minExponent, Nat.reduceSub]
      rw [max_eq_left]
      · congr 1
        grind
      · by_contra h
        have hmax : roundTarget = -149 := by
          unfold roundTarget Format.targetExponent totalExponent
          simp only [Format.mantissaBits, Format.minExponent, Nat.reduceSub]
          rw [max_eq_right (by grind)]
          norm_num
        have : target ≤ -149 :=
          (min_le_right exponent roundTarget).trans_eq hmax
        exact hfloor this
    have htarget : target = roundTarget := by
      have hroundTargetLe : roundTarget ≤ exponent := by
        by_contra h
        have hexpLe : exponent ≤ roundTarget := by grind
        rw [hroundTarget] at hexpLe
        have hm₂LogNonneg : (0 : Int) ≤ m₂.log2 := by grind
        grind
      simpa [target] using min_eq_right hroundTargetLe
    have hshiftValue : (shift : Int) =
        24 + (m₂.log2 : Int) - (m₁.log2 : Int) := by
      rw [hshift, htarget, hroundTarget]
      grind
    have hquot : 2 ^ 23 ≤ numerator / m₂ := by
      have hshiftLog : shift + m₁.log2 = 24 + m₂.log2 := by grind
      have hnumLower : 2 ^ (24 + m₂.log2) ≤ numerator := by
        calc
          2 ^ (24 + m₂.log2) = 2 ^ (m₁.log2 + shift) := by
            congr 1
            grind
          _ = 2 ^ m₁.log2 * 2 ^ shift := Nat.pow_add 2 m₁.log2 shift
          _ ≤ m₁ * 2 ^ shift :=
            Nat.mul_le_mul_right (2 ^ shift) (Nat.log2_self_le hm₁.ne')
          _ = numerator := by simp [numerator, Nat.shiftLeft_eq]
      have hdenUpper : 2 ^ 23 * m₂ < 2 ^ (24 + m₂.log2) := by
        calc
          2 ^ 23 * m₂ < 2 ^ 23 * 2 ^ (m₂.log2 + 1) :=
            (Nat.mul_lt_mul_left (Nat.pow_pos (by decide))).2 Nat.lt_log2_self
          _ = 2 ^ (24 + m₂.log2) := by
            rw [← Nat.pow_add]
            congr 1
            grind
      exact (Nat.le_div_iff_mul_le hm₂).2
        ((Nat.le_of_lt hdenUpper).trans hnumLower)
    have hquotPos : 0 < numerator / m₂ := lt_of_lt_of_le (by decide) hquot
    have hquotLog : 23 ≤ (numerator / m₂).log2 :=
      (Nat.le_log2 (Nat.ne_of_gt hquotPos)).2 hquot
    unfold Format.targetExponent totalExponent
    simp only [Format.mantissaBits, Format.minExponent, Nat.reduceSub]
    apply le_max_of_le_left
    change target ≤
      (Int.ofNat (Nat.log2 (numerator / m₂)) + 1 + target) - 24
    have hquotLogInt : (23 : Int) ≤ Int.ofNat (Nat.log2 (numerator / m₂)) := by
      change Int.ofNat 23 ≤ Int.ofNat (Nat.log2 (numerator / m₂))
      exact Int.ofNat_le.2 hquotLog
    grind

/--
For canonical finite operands, Lean's logical division and `IEEE32Exec` round the same exact
rational quotient. The proof includes extreme underflow, where `divCore` temporarily stores an
integer quotient of zero and keeps the discarded information in its `Accuracy` value.
-/
private theorem model_div_finite_eq_roundRatToIEEE32
    (s₁ s₂ : Sign) (m₁ m₂ : Nat) (e₁ e₂ : Int)
    (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) (hm₁High : m₁ < 2 ^ 24) :
    let exact := scaleRatByPow2 m₁ m₂ (e₁ - e₂)
    modelToIEEE32Exec
        (Float32.Model.pack
          (UnpackedFloat.div Format.binary32
            (.finite s₁ m₁ e₁ hm₁) (.finite s₂ m₂ e₂ hm₂))) =
      roundRatToIEEE32 (signToBool (s₁ / s₂)) exact.1 exact.2 := by
  simp only [UnpackedFloat.div, UnpackedFloat.divCore]
  let exponent := e₁ - e₂
  let target := min exponent
    (Format.binary32.targetExponent
      (totalExponent m₁ e₁ - totalExponent m₂ e₂))
  let shift := (exponent - target).toNat
  let numerator := m₁ <<< shift
  change modelToIEEE32Exec
      (Float32.Model.pack
        (UnpackedFloat.roundWithAccuracy Format.binary32
          (s₁ / s₂) (numerator / m₂) target
          (accuracyOfFraction (numerator % m₂) m₂))) =
    roundRatToIEEE32 (signToBool (s₁ / s₂))
      (scaleRatByPow2 m₁ m₂ exponent).1
      (scaleRatByPow2 m₁ m₂ exponent).2
  have htargetLe : target ≤ exponent := min_le_left _ _
  have hshift : (shift : Int) = exponent - target :=
    Int.toNat_of_nonneg (sub_nonneg.mpr htargetLe)
  have hnumerator : numerator ≠ 0 := by
    simp [numerator, Nat.shiftLeft_eq, hm₁.ne']
  have htargetInvariant : target ≤ Format.binary32.targetExponent
      (totalExponent (numerator / m₂) target) := by
    simpa [exponent, target, shift, numerator] using
      divCore_target_le_targetExponent m₁ m₂ e₁ e₂
        hm₁ hm₂ hm₁High
  rw [model_roundWithAccuracy_quotient_eq_roundRatToIEEE32
    (s₁ / s₂) numerator m₂ target hnumerator hm₂.ne' htargetInvariant]
  apply roundRatToIEEE32_eq_of_rat_eq
  · exact scaleRatByPow2_fst_ne_zero numerator m₂ target hnumerator
  · exact scaleRatByPow2_snd_ne_zero numerator m₂ target hm₂.ne'
  · exact scaleRatByPow2_fst_ne_zero m₁ m₂ exponent hm₁.ne'
  · exact scaleRatByPow2_snd_ne_zero m₁ m₂ exponent hm₂.ne'
  · rw [scaleRatByPow2_real, scaleRatByPow2_real]
    have hnumeratorReal :
        (numerator : ℝ) / (m₂ : ℝ) =
          ((m₁ : ℝ) / (m₂ : ℝ)) *
            neuralBpow binaryRadix (Int.ofNat shift) := by
      simpa [numerator, scaleRatByPow2] using
        scaleRatByPow2_real m₁ m₂ (Int.ofNat shift)
    rw [hnumeratorReal, mul_assoc, ← neuralBpow.add_exp]
    have hexponents : Int.ofNat shift + target = exponent := by
      have hshiftOfNat : Int.ofNat shift = exponent - target := hshift
      rw [hshiftOfNat]
      grind
    rw [hexponents]

/--
Lean's logical binary32 division and `IEEE32Exec.div` agree on every canonical operand.

The finite branch reduces both implementations to one exact rational quotient and one
nearest-even rounding operation. The remaining branches check the IEEE rules for NaNs,
infinities, division by zero, and signed zero.
-/
theorem model_div_eq_ieee32 (a b : Float32.Model) :
    modelToIEEE32Exec (Float32.Model.div a b) =
      IEEE32Exec.div (modelToIEEE32Exec a) (modelToIEEE32Exec b) := by
  unfold Float32.Model.div
  cases ha : a.unpack with
  | notANumber =>
      have haExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq a ha
      rw [haExec]
      have hchoose : IEEE32Exec.chooseNaN2 IEEE32Exec.canonicalNaN
          (modelToIEEE32Exec b) = some IEEE32Exec.canonicalNaN := by
        simpa [quietNaN_canonicalNaN] using
          IEEE32Exec.chooseNaN2_of_isNaN_left
            IEEE32Exec.canonicalNaN (modelToIEEE32Exec b)
            (by decide) (model_isSNaN_eq_false b) (by decide)
      rw [IEEE32Exec.div_eq_of_chooseNaN2_some _ _ _ hchoose]
      simp [UnpackedFloat.div]
  | infinity sa =>
      have haExec := modelToIEEE32Exec_eq_inf_of_unpack_eq a sa ha
      cases hb : b.unpack with
      | notANumber =>
          have hbExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb
          rw [haExec, hbExec]
          cases sa <;> decide
      | infinity sb =>
          have hbExec := modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb
          rw [haExec, hbExec]
          cases sa <;> cases sb <;> decide
      | zero sb =>
          have hbExec := modelToIEEE32Exec_eq_zero_of_unpack_eq b sb hb
          rw [haExec, hbExec]
          cases sa <;> cases sb <;> decide
      | finite sb mb eb hbpos =>
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          have hbNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some hbExec
          have hbInf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some hbExec
          have hbSign := (IEEE32Exec.sign_eq_signBit_of_toDyadic?_some hbExec).symm
          simp only [UnpackedFloat.div]
          rw [modelToIEEE32Exec_pack_infinity, haExec]
          unfold IEEE32Exec.div
          rw [hbExec]
          simp [IEEE32Exec.chooseNaN2, hbNaN, hbInf, model_isSNaN_eq_false b, hbSign]
          cases sa <;> cases sb <;> rfl
  | zero sa =>
      have haExec := modelToIEEE32Exec_eq_zero_of_unpack_eq a sa ha
      have haDyadic : IEEE32Exec.toDyadic? (modelToIEEE32Exec a) = some
          { sign := signToBool sa, mant := 0, exp := 0 } := by
        rw [toDyadic_modelToIEEE32Exec, ha]
        rfl
      cases hb : b.unpack with
      | notANumber =>
          have hbExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb
          rw [haExec, hbExec]
          cases sa <;> decide
      | infinity sb =>
          have hbExec := modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb
          rw [haExec, hbExec]
          cases sa <;> cases sb <;> decide
      | zero sb =>
          have hbExec := modelToIEEE32Exec_eq_zero_of_unpack_eq b sb hb
          rw [haExec, hbExec]
          cases sa <;> cases sb <;> decide
      | finite sb mb eb hbpos =>
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          simp only [UnpackedFloat.div]
          rw [modelToIEEE32Exec_pack_zero]
          unfold IEEE32Exec.div
          rw [haDyadic, hbExec]
          simp [hbpos.ne']
          cases sa <;> cases sb <;> rfl
  | finite sa ma ea hapos =>
      have haExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
        a sa ma ea hapos ha
      cases hb : b.unpack with
      | notANumber =>
          have hbExec := modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq b hb
          rw [hbExec]
          have hchoose : IEEE32Exec.chooseNaN2 (modelToIEEE32Exec a)
              IEEE32Exec.canonicalNaN = some IEEE32Exec.canonicalNaN := by
            simpa [quietNaN_canonicalNaN] using
              IEEE32Exec.chooseNaN2_of_isNaN_right
                (modelToIEEE32Exec a) IEEE32Exec.canonicalNaN
                (model_isSNaN_eq_false a) (by decide)
                (IEEE32Exec.isNaN_eq_false_of_toDyadic?_some haExec) (by decide)
          rw [IEEE32Exec.div_eq_of_chooseNaN2_some _ _ _ hchoose]
          simp [UnpackedFloat.div]
      | infinity sb =>
          have hbExec := modelToIEEE32Exec_eq_inf_of_unpack_eq b sb hb
          have haNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some haExec
          have haInf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some haExec
          have haSign := (IEEE32Exec.sign_eq_signBit_of_toDyadic?_some haExec).symm
          simp only [UnpackedFloat.div]
          rw [modelToIEEE32Exec_pack_zero, hbExec]
          unfold IEEE32Exec.div
          rw [haExec]
          simp [IEEE32Exec.chooseNaN2, haNaN, haInf, model_isSNaN_eq_false a, haSign]
          cases sa <;> cases sb <;> rfl
      | zero sb =>
          have hbExec := modelToIEEE32Exec_eq_zero_of_unpack_eq b sb hb
          have hbDyadic : IEEE32Exec.toDyadic? (modelToIEEE32Exec b) = some
              { sign := signToBool sb, mant := 0, exp := 0 } := by
            rw [toDyadic_modelToIEEE32Exec, hb]
            rfl
          simp only [UnpackedFloat.div]
          rw [modelToIEEE32Exec_pack_infinity]
          unfold IEEE32Exec.div
          rw [haExec, hbDyadic]
          simp [hapos.ne']
          cases sa <;> cases sb <;> rfl
      | finite sb mb eb hbpos =>
          have hbExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            b sb mb eb hbpos hb
          have habounds := unpack_finite_bounds a sa ma ea hapos ha
          have hmaHigh : ma < 2 ^ 24 := by
            rcases habounds with hsubnormal | hnormal
            · exact hsubnormal.2.trans (by norm_num)
            · exact hnormal.2.2
          change modelToIEEE32Exec
              (Float32.Model.pack
                (UnpackedFloat.div Format.binary32
                  (.finite sa ma ea hapos) (.finite sb mb eb hbpos))) =
            IEEE32Exec.div (modelToIEEE32Exec a) (modelToIEEE32Exec b)
          rw [model_div_finite_eq_roundRatToIEEE32
            sa sb ma mb ea eb hapos hbpos hmaHigh]
          rw [IEEE32Exec.div_eq_roundRatToIEEE32_of_toDyadic?
            haExec hbExec hbpos.ne']
          cases sa <;> cases sb <;> rfl

/-! ## Square-root agreement

The two algorithms present their intermediate values differently. Lean's model uses
`sqrtCore` and an accuracy flag, while `IEEE32Exec` computes an integer root and remainder.
The lemmas below show that they choose the same target exponent, radicand, root rounding, and
packed result.
-/

/-- Doubling a positive mantissa increments its binary leading-bit position. -/
private theorem log2_mul_two (m : Nat) (hm : 0 < m) :
    (m * 2).log2 = m.log2 + 1 := by
  simpa [Nat.mul_comm] using Nat.log2_two_mul hm.ne'

/-- Lean's square-root target exponent agrees with parity normalization for normal inputs. -/
private theorem sqrt_target_exponent_normal
    (m : Nat) (e : Int) (heLow : -149 ≤ e)
    (hmLow : 2 ^ 23 ≤ m) (hmHigh : m < 2 ^ 24) :
    let expOdd : Bool := (e % 2) != 0
    let mant' : Nat := if expOdd then m * 2 else m
    let expEven : Int := if expOdd then e - 1 else e
    let expHalf : Int := expEven / 2
    let t : Nat := mant'.log2 / 2
    let k0 : Int := expHalf + Int.ofNat t
    min (e.ediv 2)
        (Format.binary32.targetExponent ((totalExponent m e + 1).ediv 2)) =
      k0 - 23 := by
  dsimp only
  have hm : m ≠ 0 := Nat.ne_of_gt ((by norm_num : 0 < 2 ^ 23).trans_le hmLow)
  have hlog : m.log2 = 23 := by
    apply (Nat.log2_eq_iff hm).2
    exact ⟨hmLow, by simpa using hmHigh⟩
  have htarget :
      Format.binary32.targetExponent ((totalExponent m e + 1).ediv 2) =
        (e + 25).ediv 2 - 24 := by
    unfold Format.targetExponent totalExponent
    simp only [Format.mantissaBits, Format.minExponent, hlog]
    norm_num
    rw [show (24 : Int) + e + 1 = e + 25 by grind]
    rw [max_eq_left]
    have hquot : (-124 : Int) / 2 ≤ (e + 25) / 2 :=
      Int.ediv_le_ediv (by norm_num) (by grind)
    have hedivEq : (e + 25).ediv 2 = (e + 25) / 2 := rfl
    norm_num at hquot
    grind
  rw [htarget]
  have hmod := Int.emod_two_eq_zero_or_one e
  rcases hmod with hmod | hmod
  · have hodd : (e % 2 != 0) = false := by simp [hmod]
    simp [hodd, hlog]
    have heven : e / 2 * 2 = e := by
      exact Int.ediv_mul_cancel (Int.dvd_iff_emod_eq_zero.2 hmod)
    have hediv : e.ediv 2 = e / 2 := rfl
    have hshift : (e + 25).ediv 2 = e.ediv 2 + 12 := by
      rw [hediv]
      calc
        (e + 25) / 2 = (e / 2 * 2 + 25) / 2 := by rw [heven]
        _ = e / 2 + 25 / 2 := by rw [Int.mul_add_ediv_right _ _ (by norm_num)]
        _ = e / 2 + 12 := by norm_num
    have htargetLe : (e + 25).ediv 2 - 24 ≤ e.ediv 2 := by
      rw [hshift]
      grind
    rw [min_eq_right htargetLe]
    rw [hshift]
    grind
  · have hodd : (e % 2 != 0) = true := by simp [hmod]
    have hlog2 : (m * 2).log2 = 24 := by
      rw [log2_mul_two m (Nat.zero_lt_of_lt hmLow), hlog]
    simp [hodd, hlog2]
    have heven : (e - 1) / 2 * 2 = e - 1 := by
      apply Int.ediv_mul_cancel
      rw [Int.dvd_iff_emod_eq_zero, Int.sub_emod, hmod]
      decide
    have hediv : e.ediv 2 = e / 2 := rfl
    have heq : (e - 1) / 2 * 2 + 1 = e := by grind
    have hedivOdd : e / 2 = (e - 1) / 2 := by
      calc
        e / 2 = ((e - 1) / 2 * 2 + 1) / 2 := by rw [heq]
        _ = (e - 1) / 2 + 1 / 2 := by rw [Int.mul_add_ediv_right _ _ (by norm_num)]
        _ = (e - 1) / 2 := by norm_num
    have hshift : (e + 25).ediv 2 = e.ediv 2 + 13 := by
      rw [hediv, hedivOdd]
      calc
        (e + 25) / 2 = ((e - 1) / 2 * 2 + 26) / 2 := by
          congr 1
          grind
        _ = (e - 1) / 2 + 26 / 2 := by
          rw [Int.mul_add_ediv_right _ _ (by norm_num)]
        _ = (e - 1) / 2 + 13 := by norm_num
    have htargetLe : (e + 25).ediv 2 - 24 ≤ e.ediv 2 := by
      rw [hshift]
      grind
    rw [min_eq_right htargetLe]
    rw [hshift]
    grind

/-- Lean's square-root target exponent agrees with parity normalization for subnormal inputs. -/
private theorem sqrt_target_exponent_subnormal
    (m : Nat) (hmPos : 0 < m) (hmHigh : m < 2 ^ 23) :
    let e : Int := -149
    let expOdd : Bool := (e % 2) != 0
    let mant' : Nat := if expOdd then m * 2 else m
    let expEven : Int := if expOdd then e - 1 else e
    let expHalf : Int := expEven / 2
    let t : Nat := mant'.log2 / 2
    let k0 : Int := expHalf + Int.ofNat t
    min (e.ediv 2)
        (Format.binary32.targetExponent ((totalExponent m e + 1).ediv 2)) =
      k0 - 23 := by
  dsimp only
  have hlogHigh : m.log2 < 23 := (Nat.log2_lt hmPos.ne').2 hmHigh
  have hlog2 : (m * 2).log2 = m.log2 + 1 := log2_mul_two m hmPos
  have hodd : (((-149 : Int) % 2) != 0) = true := by decide
  simp only [hodd, if_true, hlog2]
  unfold Format.targetExponent totalExponent
  simp only [Format.mantissaBits, Format.minExponent]
  norm_num
  have htarget :
      max ((((m.log2 : Int) + 1 + -149 + 1).ediv 2) - 24) (-149) =
        (((m.log2 : Int) + 1 + -149 + 1).ediv 2) - 24 := by
    rw [max_eq_left]
    have hdiv :
        ((m.log2 : Int) + 1 + -149 + 1).ediv 2 =
          ((m.log2 : Int) - 147) / 2 := by
      change ((m.log2 : Int) + 1 + -149 + 1) / 2 =
        ((m.log2 : Int) - 147) / 2
      congr 1
      grind
    have hquot : (-147 : Int) / 2 ≤ ((m.log2 : Int) - 147) / 2 :=
      Int.ediv_le_ediv (by norm_num) (by grind)
    norm_num at hquot
    rw [hdiv]
    grind
  rw [htarget]
  have hshift :
      (((m.log2 : Int) + 1 + -149 + 1).ediv 2) =
        (((m.log2 : Int) + 1).ediv 2) - 74 := by
    calc
      ((m.log2 : Int) + 1 + -149 + 1).ediv 2 =
          ((m.log2 : Int) + 1 - 74 * 2).ediv 2 := by congr 1; grind
      _ = ((m.log2 : Int) + 1).ediv 2 - 74 := by
        change ((m.log2 : Int) + 1 - 74 * 2) / 2 =
          ((m.log2 : Int) + 1) / 2 - 74
        rw [Int.sub_mul_ediv_right _ _ (by norm_num)]
  rw [hshift]
  have hlogCast : (m.log2 : Int) ≤ 22 := by grind
  have hquot : ((m.log2 : Int) + 1) / 2 ≤ 23 / 2 :=
    Int.ediv_le_ediv (by norm_num) (by grind)
  have hdiv : ((m.log2 : Int) + 1).ediv 2 = ((m.log2 : Int) + 1) / 2 := rfl
  rw [min_eq_right]
  · rw [hdiv]
    norm_num at hquot ⊢
    grind
  · rw [hdiv]
    norm_num at hquot ⊢
    grind

/-- The model and executable algorithms form the same parity-normalized integer radicand. -/
private theorem sqrt_radicand_eq
    (m : Nat) (e : Int)
    (ht : ((if (e % 2 != 0) then m * 2 else m).log2 / 2) ≤ 23)
    (htarget :
      min (e.ediv 2)
          (Format.binary32.targetExponent ((totalExponent m e + 1).ediv 2)) =
        (if (e % 2 != 0) then e - 1 else e) / 2 +
          Int.ofNat ((if (e % 2 != 0) then m * 2 else m).log2 / 2) - 23) :
    m <<< (e - 2 * min (e.ediv 2)
          (Format.binary32.targetExponent ((totalExponent m e + 1).ediv 2))).toNat =
      (if (e % 2 != 0) then m * 2 else m) <<<
        (2 * (23 - ((if (e % 2 != 0) then m * 2 else m).log2 / 2))) := by
  rw [htarget]
  have hmod := Int.emod_two_eq_zero_or_one e
  rcases hmod with hmod | hmod
  · have hodd : (e % 2 != 0) = false := by simp [hmod]
    simp [hodd] at ht ⊢
    have heven : e / 2 * 2 = e :=
      Int.ediv_mul_cancel (Int.dvd_iff_emod_eq_zero.2 hmod)
    have hcast : (Int.ofNat (23 - m.log2 / 2) : Int) =
        23 - Int.ofNat (m.log2 / 2) := by
      exact Int.ofNat_sub ht
    have hshift :
        (e - 2 * (e / 2 + (m.log2 : Int) / 2 - 23)).toNat =
          2 * (23 - m.log2 / 2) := by
      have hnatDiv : (m.log2 : Int) / 2 = Int.ofNat (m.log2 / 2) := by
        exact (Int.natCast_ediv m.log2 2).symm
      rw [hnatDiv]
      have hcastMul : Int.ofNat (2 * (23 - m.log2 / 2)) =
          2 * Int.ofNat (23 - m.log2 / 2) := by norm_num
      have heq : e - 2 * (e / 2 + Int.ofNat (m.log2 / 2) - 23) =
          2 * Int.ofNat (23 - m.log2 / 2) := by rw [hcast]; grind
      rw [heq, ← hcastMul]
      rfl
    rw [hshift]
  · have hodd : (e % 2 != 0) = true := by simp [hmod]
    simp [hodd] at ht ⊢
    have heven : (e - 1) / 2 * 2 = e - 1 := by
      apply Int.ediv_mul_cancel
      rw [Int.dvd_iff_emod_eq_zero, Int.sub_emod, hmod]
      decide
    have hcast : (Int.ofNat (23 - (m * 2).log2 / 2) : Int) =
        23 - Int.ofNat ((m * 2).log2 / 2) := by
      exact Int.ofNat_sub ht
    have hshift :
        (e - 2 * ((e - 1) / 2 + ((m * 2).log2 : Int) / 2 - 23)).toNat =
          2 * (23 - (m * 2).log2 / 2) + 1 := by
      have hnatDiv : ((m * 2).log2 : Int) / 2 =
          Int.ofNat ((m * 2).log2 / 2) := by
        exact (Int.natCast_ediv (m * 2).log2 2).symm
      rw [hnatDiv]
      have hcastSum : Int.ofNat (2 * (23 - (m * 2).log2 / 2) + 1) =
          2 * Int.ofNat (23 - (m * 2).log2 / 2) + 1 := by norm_num
      have heq : e - 2 * ((e - 1) / 2 + Int.ofNat ((m * 2).log2 / 2) - 23) =
          2 * Int.ofNat (23 - (m * 2).log2 / 2) + 1 := by rw [hcast]; grind
      rw [heq, ← hcastSum]
      rfl
    rw [hshift]
    simp [Nat.shiftLeft_eq, Nat.pow_succ, Nat.mul_assoc, Nat.mul_comm]

/-- Scaling the radicand places its integer square root in the normalized 24-bit range. -/
private theorem sqrt_scaled_root_bounds (mantissa : Nat)
    (hm : mantissa ≠ 0) (hmHigh : mantissa < 2 ^ 25) :
    let t := mantissa.log2 / 2
    let p := 23 - t
    let n := mantissa <<< (2 * p)
    let q := Nat.sqrt n
    t ≤ 12 ∧ 2 ^ 23 ≤ q ∧ q < 2 ^ 24 := by
  dsimp only
  let l := mantissa.log2
  let t := l / 2
  let p := 23 - t
  let n := mantissa <<< (2 * p)
  let q := Nat.sqrt n
  have hlHigh : l ≤ 24 := by
    have : l < 25 := (Nat.log2_lt hm).2 hmHigh
    grind
  have htHigh : t ≤ 12 := by
    have hdiv : l / 2 ≤ 24 / 2 := Nat.div_le_div_right hlHigh
    simpa [t] using hdiv
  have ht23 : t ≤ 23 := htHigh.trans (by decide)
  have hpowLow : 2 ^ l ≤ mantissa := (Nat.le_log2 hm).1 le_rfl
  have hpowHigh : mantissa < 2 ^ (l + 1) := (Nat.log2_lt hm).1 (Nat.lt_succ_self l)
  have htwiceLow : 2 * t ≤ l := by
    simpa [t, Nat.mul_comm] using Nat.mul_div_le l 2
  have hmantissaLow : 2 ^ (2 * t) ≤ mantissa :=
    (Nat.pow_le_pow_right (by decide) htwiceLow).trans hpowLow
  have hlHigh' : l + 1 ≤ 2 * (t + 1) := by
    have hrem : l % 2 < 2 := Nat.mod_lt l (by decide)
    have hdecomp : 2 * (l / 2) + l % 2 = l := by
      simpa [Nat.add_comm, Nat.mul_comm] using Nat.mod_add_div l 2
    have : l < 2 * (l / 2) + 2 := by grind
    simpa [t, Nat.mul_add, Nat.mul_one] using Nat.succ_le_of_lt this
  have hmantissaHigh : mantissa < 2 ^ (2 * (t + 1)) :=
    hpowHigh.trans_le (Nat.pow_le_pow_right (by decide) hlHigh')
  have htp : t + p = 23 := by
    simpa [p] using Nat.add_sub_of_le ht23
  have hsum : 2 * t + 2 * p = 46 := by grind
  have hn : n = mantissa * 2 ^ (2 * p) := by
    simp [n, Nat.shiftLeft_eq]
  have hnLow : 2 ^ 46 ≤ n := by
    have hmul := Nat.mul_le_mul_right (2 ^ (2 * p)) hmantissaLow
    rw [← Nat.pow_add] at hmul
    simpa [hn, hsum] using hmul
  have hnHigh : n < 2 ^ 48 := by
    have hmul : mantissa * 2 ^ (2 * p) <
        2 ^ (2 * (t + 1)) * 2 ^ (2 * p) :=
      Nat.mul_lt_mul_of_pos_right hmantissaHigh (Nat.pow_pos (by decide))
    rw [← Nat.pow_add] at hmul
    have hsum' : 2 * (t + 1) + 2 * p = 48 := by grind
    simpa [hn, hsum'] using hmul
  have hqLow : 2 ^ 23 ≤ q := by
    apply Nat.le_sqrt.2
    have hpow : 2 ^ 23 * 2 ^ 23 = 2 ^ 46 := by norm_num [← Nat.pow_add]
    simpa [q, hpow] using hnLow
  have hqHigh : q < 2 ^ 24 := by
    apply Nat.sqrt_lt.2
    have hpow : 2 ^ 24 * 2 ^ 24 = 2 ^ 48 := by norm_num [← Nat.pow_add]
    simpa [q, hpow] using hnHigh
  exact ⟨htHigh, hqLow, hqHigh⟩

/-- Every finite binary32 input produces a square-root leading exponent in the normal range. -/
private theorem sqrt_leading_exponent_bounds (e : Int) (t : Nat)
    (heLow : -149 ≤ e) (heHigh : e ≤ 104) (ht : t ≤ 12) :
    let expOdd : Bool := (e % 2) != 0
    let expEven : Int := if expOdd then e - 1 else e
    let expHalf : Int := expEven / 2
    let k0 : Int := expHalf + Int.ofNat t
    (-75 : Int) ≤ k0 ∧ k0 ≤ 64 := by
  dsimp only
  let expOdd : Bool := (e % 2) != 0
  let expEven : Int := if expOdd then e - 1 else e
  let expHalf : Int := expEven / 2
  have hevenLow : -150 ≤ expEven := by
    cases hodd : expOdd <;> simp [expEven, hodd] <;> grind
  have hevenHigh : expEven ≤ 104 := by
    cases hodd : expOdd <;> simp [expEven, hodd] <;> grind
  have hhalfLow : -75 ≤ expHalf := by
    have := Int.ediv_le_ediv (by decide : (0 : Int) < 2) hevenLow
    simpa [expHalf] using this
  have hhalfHigh : expHalf ≤ 52 := by
    have := Int.ediv_le_ediv (by decide : (0 : Int) < 2) hevenHigh
    simpa [expHalf] using this
  have htInt : (Int.ofNat t : Int) ≤ 12 := Int.ofNat_le.2 ht
  have htNonneg : (0 : Int) ≤ Int.ofNat t := Int.natCast_nonneg t
  change (-75 : Int) ≤ expHalf + Int.ofNat t ∧ expHalf + Int.ofNat t ≤ 64
  constructor <;> grind

/-- The model accuracy flag and executable remainder test make the same nearest-even decision. -/
private theorem sqrt_accuracy_round (q r : Nat) :
    Accuracy.roundToNearestEven q
        (if r = 0 then .exact else .inexact (if r ≤ q then .lt else .gt)) =
      if r > q then q + 1 else q := by
  by_cases hr : r = 0
  · simp [hr, Accuracy.roundToNearestEven]
  · by_cases hle : r ≤ q
    · simp [hr, hle, Accuracy.roundToNearestEven]
    · simp [hr, hle, Accuracy.roundToNearestEven]

/--
Lean's logical square root and `IEEE32Exec.sqrt` agree for every canonical binary32 value.

The positive finite case aligns their parity-normalized radicands, integer roots, remainder-based
nearest-even decisions, and the possible carry into the next exponent. The other cases cover NaNs,
infinities, signed zeros, and negative finite inputs directly.
-/
theorem model_sqrt_eq_ieee32 (a : Float32.Model) :
    modelToIEEE32Exec (Float32.Model.sqrt a) =
      IEEE32Exec.sqrt (modelToIEEE32Exec a) := by
  unfold Float32.Model.sqrt
  cases ha : a.unpack with
  | notANumber =>
      rw [modelToIEEE32Exec_eq_canonicalNaN_of_unpack_eq a ha]
      simp only [UnpackedFloat.sqrt, modelToIEEE32Exec_pack_notANumber]
      decide
  | infinity sign =>
      cases sign with
      | negative =>
          rw [modelToIEEE32Exec_eq_inf_of_unpack_eq a .negative ha]
          simp only [UnpackedFloat.sqrt, modelToIEEE32Exec_pack_notANumber]
          decide
      | positive =>
          rw [modelToIEEE32Exec_eq_inf_of_unpack_eq a .positive ha]
          simp only [UnpackedFloat.sqrt, modelToIEEE32Exec_pack_infinity]
          decide
  | zero sign =>
      cases sign with
      | negative =>
          rw [modelToIEEE32Exec_eq_zero_of_unpack_eq a .negative ha]
          simp only [UnpackedFloat.sqrt, modelToIEEE32Exec_pack_zero]
          decide
      | positive =>
          rw [modelToIEEE32Exec_eq_zero_of_unpack_eq a .positive ha]
          simp only [UnpackedFloat.sqrt, modelToIEEE32Exec_pack_zero]
          decide
  | finite sign mantissa exponent positive =>
      cases sign with
      | negative =>
          have haExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            a .negative mantissa exponent positive ha
          have haNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some haExec
          have haInf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some haExec
          have haZero := IEEE32Exec.isZero_eq_false_of_toDyadic?_some_of_mant_ne_zero
            haExec positive.ne'
          have haSign := IEEE32Exec.sign_eq_signBit_of_toDyadic?_some haExec
          have haSignBit : IEEE32Exec.signBit (modelToIEEE32Exec a) = true := by
            simpa [signToBool] using haSign.symm
          simp only [UnpackedFloat.sqrt]
          rw [modelToIEEE32Exec_pack_notANumber]
          unfold IEEE32Exec.sqrt
          simp [IEEE32Exec.chooseNaN1, haNaN, haInf, haZero, haSignBit]
      | positive =>
          have haExec := toDyadic_modelToIEEE32Exec_eq_some_of_unpack_eq
            a .positive mantissa exponent positive ha
          have haNaN := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some haExec
          have haInf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some haExec
          have haZero := IEEE32Exec.isZero_eq_false_of_toDyadic?_some_of_mant_ne_zero
            haExec positive.ne'
          have haSign := IEEE32Exec.sign_eq_signBit_of_toDyadic?_some haExec
          have haSignBit : IEEE32Exec.signBit (modelToIEEE32Exec a) = false := by
            simpa [signToBool] using haSign.symm
          have hexponentBounds := unpack_finite_exponent_bounds
            a .positive mantissa exponent positive ha
          have hmantissaBounds := unpack_finite_bounds
            a .positive mantissa exponent positive ha
          let expOdd : Bool := (exponent % 2) != 0
          let mantissa' : Nat := if expOdd then mantissa * 2 else mantissa
          let expEven : Int := if expOdd then exponent - 1 else exponent
          let expHalf : Int := expEven / 2
          let t : Nat := mantissa'.log2 / 2
          let p : Nat := 23 - t
          let radicand : Nat := mantissa' <<< (2 * p)
          let root : Nat := Nat.sqrt radicand
          let remainder : Nat := radicand - root * root
          let accuracy : Accuracy :=
            if remainder = 0 then .exact
            else .inexact (if remainder ≤ root then .lt else .gt)
          let roundedRoot : Nat := if remainder > root then root + 1 else root
          let leadingExponent : Int := expHalf + Int.ofNat t
          have hmantissa'Pos : 0 < mantissa' := by
            dsimp only [mantissa', expOdd]
            split <;> grind
          have hmantissa'High : mantissa' < 2 ^ 25 := by
            dsimp only [mantissa', expOdd]
            split
            · rcases hmantissaBounds with hsubnormal | hnormal
              · have : mantissa * 2 < 2 ^ 24 := by grind
                exact this.trans (by norm_num)
              · norm_num at hnormal ⊢
                grind
            · rcases hmantissaBounds with hsubnormal | hnormal
              · exact hsubnormal.2.trans (by norm_num)
              · exact hnormal.2.2.trans (by norm_num)
          have hrootBounds := sqrt_scaled_root_bounds
            mantissa' hmantissa'Pos.ne' hmantissa'High
          have htHigh : t ≤ 12 := by
            simpa [t, p, radicand, root] using hrootBounds.1
          have ht23 : t ≤ 23 := htHigh.trans (by decide)
          have hrootLow : 2 ^ 23 ≤ root := by
            simpa [t, p, radicand, root] using hrootBounds.2.1
          have hrootHigh : root < 2 ^ 24 := by
            simpa [t, p, radicand, root] using hrootBounds.2.2
          have hleadingBounds := sqrt_leading_exponent_bounds
            exponent t hexponentBounds.1 hexponentBounds.2 htHigh
          have hleadingLow : (-75 : Int) ≤ leadingExponent := by
            simpa [leadingExponent, expHalf, expEven, expOdd] using hleadingBounds.1
          have hleadingHigh : leadingExponent ≤ 64 := by
            simpa [leadingExponent, expHalf, expEven, expOdd] using hleadingBounds.2
          have htarget :
              min (exponent.ediv 2)
                  (Format.binary32.targetExponent
                    ((totalExponent mantissa exponent + 1).ediv 2)) =
                leadingExponent - 23 := by
            rcases hmantissaBounds with hsubnormal | hnormal
            · rcases hsubnormal with ⟨rfl, hmantissaHigh⟩
              simpa [leadingExponent, expHalf, expEven, expOdd, t, mantissa'] using
                sqrt_target_exponent_subnormal mantissa positive hmantissaHigh
            · simpa [leadingExponent, expHalf, expEven, expOdd, t, mantissa'] using
                sqrt_target_exponent_normal mantissa exponent hnormal.1
                  hnormal.2.1 hnormal.2.2
          have hradicand :
              mantissa <<<
                  (exponent - 2 * min (exponent.ediv 2)
                    (Format.binary32.targetExponent
                      ((totalExponent mantissa exponent + 1).ediv 2))).toNat =
                radicand := by
            simpa [radicand, p, t, mantissa', expOdd,
              leadingExponent, expHalf, expEven] using
                sqrt_radicand_eq mantissa exponent ht23 htarget
          have hradicandAtTarget :
              mantissa <<< (exponent - 2 * (leadingExponent - 23)).toNat =
                radicand := by
            rw [← htarget]
            exact hradicand
          have haccuracyRound :
              accuracy.roundToNearestEven root = roundedRoot := by
            simpa [accuracy, roundedRoot] using
              sqrt_accuracy_round root remainder
          have hroundedLow : 2 ^ 23 ≤ roundedRoot := by
            dsimp only [roundedRoot]
            split <;> grind
          have hroundedLe : roundedRoot ≤ 2 ^ 24 := by
            dsimp only [roundedRoot]
            split <;> grind
          have hsqrtCore :
              sqrtCore Format.binary32 mantissa exponent =
                (root, leadingExponent - 23, accuracy) := by
            unfold sqrtCore
            rw [htarget]
            dsimp only
            rw [hradicandAtTarget]
          have hexec :
              IEEE32Exec.sqrt (modelToIEEE32Exec a) =
                IEEE32Exec.ofBits
                  (IEEE32Exec.mkBits false
                    ((if roundedRoot == IEEE32Exec.pow2 24 then leadingExponent + 1
                      else leadingExponent) + 127).toNat
                    ((if roundedRoot == IEEE32Exec.pow2 24 then
                        IEEE32Exec.pow2 23 else roundedRoot) -
                      IEEE32Exec.pow2 23)) := by
            unfold IEEE32Exec.sqrt
            simp [IEEE32Exec.chooseNaN1, haNaN, haInf, haZero, haSignBit, haExec,
              signToBool, expOdd, mantissa', expEven, expHalf, t, p,
              radicand, root, remainder, roundedRoot, leadingExponent]
          simp only [UnpackedFloat.sqrt, hsqrtCore]
          rw [roundWithAccuracy_normalized_eq_finishRoundedMantissa
            .positive root leadingExponent accuracy hrootLow hrootHigh (by grind)]
          rw [haccuracyRound]
          rw [hexec]
          by_cases hcarry : roundedRoot = IEEE32Exec.pow2 24
          · rw [hcarry]
            rw [model_finishRoundedMantissa_normal_carry
              .positive leadingExponent (by grind) (by grind)]
            simp only [signToBool, beq_self_eq_true, if_true, Nat.sub_self]
            congr 2
            grind
          · have hroundedHigh : roundedRoot < IEEE32Exec.pow2 24 := by
              have hle : roundedRoot ≤ IEEE32Exec.pow2 24 := by
                simpa [IEEE32Exec.pow2, Nat.shiftLeft_eq] using hroundedLe
              exact lt_of_le_of_ne hle hcarry
            have hroundedLow' : IEEE32Exec.pow2 23 ≤ roundedRoot := by
              simpa [IEEE32Exec.pow2, Nat.shiftLeft_eq] using hroundedLow
            simpa [hcarry, signToBool] using
              model_finishRoundedMantissa_normal .positive roundedRoot leadingExponent
                hroundedLow' hroundedHigh (by grind) (by grind)


/-! ## Agreement between the two logical algorithms -/

/--
Pointwise agreement between one unary `Float32.Model` operation and an `IEEE32Exec` operation.

The executable result is canonicalized because `Float32.Model` retains only one NaN encoding,
whereas `IEEE32Exec` deliberately preserves payload and sign bits.
-/
def UnaryAgreement
    (modelOp : Float32.Model → Float32.Model)
    (execOp : IEEE32Exec → IEEE32Exec) : Prop :=
  ∀ a, modelToIEEE32Exec (modelOp a) = canonicalize (execOp (modelToIEEE32Exec a))

/-- Pointwise agreement between binary operations in the two logical binary32 models. -/
def BinaryAgreement
    (modelOp : Float32.Model → Float32.Model → Float32.Model)
    (execOp : IEEE32Exec → IEEE32Exec → IEEE32Exec) : Prop :=
  ∀ a b, modelToIEEE32Exec (modelOp a b) =
    canonicalize (execOp (modelToIEEE32Exec a) (modelToIEEE32Exec b))

/-- Model negation and executable sign-bit negation agree for every canonical binary32 value. -/
theorem neg_agreement : UnaryAgreement Float32.Model.neg IEEE32Exec.neg := by
  intro a
  unfold modelToIEEE32Exec canonicalize IEEE32Exec.neg
  change IEEE32Exec.ofBits (Float32.Model.neg a).toBits =
    IEEE32Exec.ofBits
      (Float32.Model.ofBits (a.toBits ^^^ IEEE32Exec.signMask)).toBits
  rw [model_neg_eq_ofBits_xor_sign]

/-- Model absolute value and executable sign clearing agree for every canonical binary32 value. -/
theorem abs_agreement : UnaryAgreement Float32.Model.abs IEEE32Exec.abs := by
  intro a
  unfold modelToIEEE32Exec canonicalize IEEE32Exec.abs
  change IEEE32Exec.ofBits (Float32.Model.abs a).toBits =
    IEEE32Exec.ofBits
      (Float32.Model.ofBits (a.toBits &&& (~~~IEEE32Exec.signMask))).toBits
  rw [model_abs_eq_ofBits_clear_sign]

/-- Model multiplication and executable multiplication agree for all canonical binary32 values. -/
theorem mul_agreement : BinaryAgreement Float32.Model.mul IEEE32Exec.mul := by
  intro a b
  calc
    modelToIEEE32Exec (Float32.Model.mul a b) =
        IEEE32Exec.mul (modelToIEEE32Exec a) (modelToIEEE32Exec b) :=
      model_mul_eq_ieee32 a b
    _ = canonicalize (IEEE32Exec.mul (modelToIEEE32Exec a) (modelToIEEE32Exec b)) := by
      rw [← model_mul_eq_ieee32]
      exact (canonicalize_modelToIEEE32Exec (Float32.Model.mul a b)).symm

/-- Model addition and executable addition agree for all canonical binary32 values. -/
theorem add_agreement : BinaryAgreement Float32.Model.add IEEE32Exec.add := by
  intro a b
  calc
    modelToIEEE32Exec (Float32.Model.add a b) =
        IEEE32Exec.add (modelToIEEE32Exec a) (modelToIEEE32Exec b) :=
      model_add_eq_ieee32 a b
    _ = canonicalize (IEEE32Exec.add (modelToIEEE32Exec a) (modelToIEEE32Exec b)) := by
      rw [← model_add_eq_ieee32]
      exact (canonicalize_modelToIEEE32Exec (Float32.Model.add a b)).symm

/-- Model division and executable division agree for all canonical binary32 values. -/
theorem div_agreement : BinaryAgreement Float32.Model.div IEEE32Exec.div := by
  intro a b
  calc
    modelToIEEE32Exec (Float32.Model.div a b) =
        IEEE32Exec.div (modelToIEEE32Exec a) (modelToIEEE32Exec b) :=
      model_div_eq_ieee32 a b
    _ = canonicalize (IEEE32Exec.div (modelToIEEE32Exec a) (modelToIEEE32Exec b)) := by
      rw [← model_div_eq_ieee32]
      exact (canonicalize_modelToIEEE32Exec (Float32.Model.div a b)).symm

/-- Model square root and executable square root agree for every canonical binary32 value. -/
theorem sqrt_agreement : UnaryAgreement Float32.Model.sqrt IEEE32Exec.sqrt := by
  intro a
  calc
    modelToIEEE32Exec (Float32.Model.sqrt a) =
        IEEE32Exec.sqrt (modelToIEEE32Exec a) :=
      model_sqrt_eq_ieee32 a
    _ = canonicalize (IEEE32Exec.sqrt (modelToIEEE32Exec a)) := by
      rw [← model_sqrt_eq_ieee32]
      exact (canonicalize_modelToIEEE32Exec (Float32.Model.sqrt a)).symm

private theorem sub_agreement_of_add
    (h : BinaryAgreement Float32.Model.add IEEE32Exec.add) :
    BinaryAgreement Float32.Model.sub IEEE32Exec.sub := by
  intro a b
  rw [model_sub_eq_add_neg, h a (Float32.Model.neg b), neg_agreement b]
  unfold IEEE32Exec.sub
  exact canonicalize_add_right (modelToIEEE32Exec a)
    (IEEE32Exec.neg (modelToIEEE32Exec b))

/-- Subtraction agrees because it is addition after exact sign-bit negation in both models. -/
theorem sub_agreement : BinaryAgreement Float32.Model.sub IEEE32Exec.sub :=
  sub_agreement_of_add add_agreement

/-- Transfer Lean `Float32` addition to the independent executable binary32 model. -/
theorem toIEEE32Exec_add (a b : _root_.Float32) :
    toIEEE32Exec (Float32.add a b) =
      canonicalize (IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b)) := by
  exact add_agreement a.toModel b.toModel

/-- Transfer Lean `Float32` subtraction to the independent executable binary32 model. -/
theorem toIEEE32Exec_sub (a b : _root_.Float32) :
    toIEEE32Exec (Float32.sub a b) =
      canonicalize (IEEE32Exec.sub (toIEEE32Exec a) (toIEEE32Exec b)) := by
  exact sub_agreement a.toModel b.toModel

/-- Transfer Lean `Float32` multiplication to the independent executable binary32 model. -/
theorem toIEEE32Exec_mul (a b : _root_.Float32) :
    toIEEE32Exec (Float32.mul a b) =
      canonicalize (IEEE32Exec.mul (toIEEE32Exec a) (toIEEE32Exec b)) := by
  exact mul_agreement a.toModel b.toModel

/-- Transfer Lean `Float32` division to the independent executable binary32 model. -/
theorem toIEEE32Exec_div (a b : _root_.Float32) :
    toIEEE32Exec (Float32.div a b) =
      canonicalize (IEEE32Exec.div (toIEEE32Exec a) (toIEEE32Exec b)) := by
  exact div_agreement a.toModel b.toModel

/-- Transfer Lean `Float32` negation to `IEEE32Exec` without an arithmetic hypothesis. -/
theorem toIEEE32Exec_neg (a : _root_.Float32) :
    toIEEE32Exec (Float32.neg a) = canonicalize (IEEE32Exec.neg (toIEEE32Exec a)) := by
  exact neg_agreement a.toModel

/-- Transfer Lean `Float32.abs` to `IEEE32Exec` without an arithmetic hypothesis. -/
theorem toIEEE32Exec_abs (a : _root_.Float32) :
    toIEEE32Exec (Float32.abs a) = canonicalize (IEEE32Exec.abs (toIEEE32Exec a)) := by
  exact abs_agreement a.toModel

/-- Transfer Lean `Float32.sqrt` to the independent executable binary32 model. -/
theorem toIEEE32Exec_sqrt (a : _root_.Float32) :
    toIEEE32Exec (Float32.sqrt a) = canonicalize (IEEE32Exec.sqrt (toIEEE32Exec a)) := by
  exact sqrt_agreement a.toModel

end Float32Bridge

end TorchLean.Floats.IEEE754
