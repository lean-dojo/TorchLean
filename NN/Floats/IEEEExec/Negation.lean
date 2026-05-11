/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Nat.Bitwise
public import NN.Floats.IEEEExec.Exec32

/-!
# Lemmas about sign-bit flips (`b ^^^ signMask`)

The executable IEEE32 kernel implements negation by flipping the sign bit of the underlying
`UInt32` encoding. For many proofs (finite/special classification, interval soundness, bridge
lemmas) we need the basic fact that this operation does **not** affect the exponent or fraction
fields.

This file centralizes those bit-manipulation lemmas for the large proof modules.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

/--
Flipping the sign bit of a float32 encoding does not affect its exponent field.

We use this to show that `IEEE32Exec.neg` preserves the “finite vs NaN/Inf” classification, because
NaN/Inf are detected from exponent+fraction (not from sign).
-/
theorem expField_ofBits_xor_signMask (b : UInt32) :
    expField (ofBits (b ^^^ signMask)) = expField (ofBits b) := by
  apply UInt32.toNat_inj.1
  have hSignMask : signMask.toNat = 2 ^ 31 := by decide
  have hExpMask : expAllOnes.toNat = 2 ^ 8 - 1 := by decide
  -- Reduce to a nat-level statement and reason by `testBit`.
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < 8
  · have hmask : Nat.testBit (2 ^ 8 - 1) i = true := by
      simpa [hi] using (Nat.testBit_two_pow_sub_one 8 i)
    have hmask' : Nat.testBit expAllOnes.toNat i = true := by
      simpa [hExpMask] using hmask
    -- The sign bit turns into bit 8 after shifting by 23, so it is masked out by `expAllOnes`.
    have hxorBit :
        Nat.testBit (b.toNat ^^^ signMask.toNat) (23 + i) = Nat.testBit b.toNat (23 + i) := by
      have hSign : Nat.testBit signMask.toNat (23 + i) = false := by
        have hlt : 23 + i < 31 := by
          -- `i < 8`, so `23 + i < 23 + 8 = 31`.
          have : 23 + i < 23 + 8 := Nat.add_lt_add_left hi 23
          simpa using this
        have hne : 31 ≠ 23 + i := ne_of_gt hlt
        simpa [hSignMask] using (Nat.testBit_two_pow_of_ne (n := 31) (m := 23 + i) hne)
      simp [Nat.testBit_xor, hSign]
    calc
      Nat.testBit (((((b ^^^ signMask).toNat >>> 23) &&& expAllOnes.toNat))) i
          = (Nat.testBit (((b ^^^ signMask).toNat >>> 23)) i && Nat.testBit expAllOnes.toNat i) := by
              simp
      _ = Nat.testBit (((b ^^^ signMask).toNat >>> 23)) i := by simp [hmask']
      _ = Nat.testBit ((b ^^^ signMask).toNat) (23 + i) := by simp [Nat.testBit_shiftRight]
      _ = Nat.testBit (b.toNat ^^^ signMask.toNat) (23 + i) := by simp [UInt32.toNat_xor]
      _ = Nat.testBit b.toNat (23 + i) := hxorBit
      _ = Nat.testBit (b.toNat >>> 23) i := by simp [Nat.testBit_shiftRight]
      _ = Nat.testBit ((b.toNat >>> 23) &&& expAllOnes.toNat) i := by simp [hmask']
      _ = Nat.testBit (expField (ofBits b)).toNat i := by
            simp [expField, ofBits, UInt32.toNat_and, UInt32.toNat_shiftRight]
  · have hi' : 8 ≤ i := Nat.le_of_not_gt hi
    have hmask : Nat.testBit expAllOnes.toNat i = false := by
      have h := Nat.testBit_two_pow_sub_one 8 i
      simpa [hi, hExpMask] using h
    simp [expField, ofBits, UInt32.toNat_and, UInt32.toNat_shiftRight, hmask]

/--
Flipping the sign bit of a float32 encoding does not affect its fraction field.

This complements `expField_ofBits_xor_signMask`: NaN payload bits are unchanged.
-/
theorem fracField_ofBits_xor_signMask (b : UInt32) :
    fracField (ofBits (b ^^^ signMask)) = fracField (ofBits b) := by
  apply UInt32.toNat_inj.1
  have hSignMask : signMask.toNat = 2 ^ 31 := by decide
  have hFracMask : fracMask.toNat = 2 ^ 23 - 1 := by decide
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < 23
  · have hmask : Nat.testBit fracMask.toNat i = true := by
      simpa [hFracMask, hi] using (Nat.testBit_two_pow_sub_one 23 i)
    have hxorBit : Nat.testBit (b.toNat ^^^ signMask.toNat) i = Nat.testBit b.toNat i := by
      have hSign : Nat.testBit signMask.toNat i = false := by
        have hne : 31 ≠ i := by
          have hlt : i < 31 := lt_trans hi (by decide : 23 < 31)
          exact ne_of_gt hlt
        simpa [hSignMask] using (Nat.testBit_two_pow_of_ne (n := 31) (m := i) hne)
      simp [Nat.testBit_xor, hSign]
    calc
      Nat.testBit ((fracField (ofBits (b ^^^ signMask))).toNat) i
          = Nat.testBit (((b ^^^ signMask).toNat &&& fracMask.toNat)) i := by
              simp [fracField, ofBits, UInt32.toNat_and]
      _ = (Nat.testBit (b.toNat ^^^ signMask.toNat) i && Nat.testBit fracMask.toNat i) := by
            simp [UInt32.toNat_xor]
      _ = Nat.testBit (b.toNat ^^^ signMask.toNat) i := by simp [hmask]
      _ = Nat.testBit b.toNat i := hxorBit
      _ = (Nat.testBit b.toNat i && Nat.testBit fracMask.toNat i) := by simp [hmask]
      _ = Nat.testBit (((b.toNat) &&& fracMask.toNat)) i := by simp
      _ = Nat.testBit ((fracField (ofBits b)).toNat) i := by
            simp [fracField, ofBits, UInt32.toNat_and]
  · have hi' : 23 ≤ i := Nat.le_of_not_gt hi
    have hmask : Nat.testBit fracMask.toNat i = false := by
      have h := Nat.testBit_two_pow_sub_one 23 i
      simpa [hi, hFracMask] using h
    simp [fracField, ofBits, UInt32.toNat_and, hmask]

end IEEE32Exec
end TorchLean.Floats.IEEE754
