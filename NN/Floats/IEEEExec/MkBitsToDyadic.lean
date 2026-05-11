/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Nat.Bitwise
public import NN.Floats.IEEEExec.Exec32

/-!
# MkBitsToDyadic

`Exec32.lean` defines the bit-level encoding/decoding functions for IEEE-754 binary32, including
`mkBits` and `toDyadic?`.

This module provides the finite-path decoding lemma for bit patterns constructed as
`ofBits (mkBits sign exp frac)`. Keeping the lemma in a dedicated module avoids import cycles and
gives the bridge, interval, and runtime-approximation proofs a shared decoding interface.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

/-! ## `toDyadic? (ofBits (mkBits …))` on the finite path -/

private lemma nat_and_two_pow_sub_one_eq_self {n k : Nat} (hn : n < 2 ^ k) :
    n &&& (2 ^ k - 1) = n := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < k
  · simp [hi]
  · have hi' : k ≤ i := Nat.le_of_not_gt hi
    have hn' : n < 2 ^ i :=
      lt_of_lt_of_le hn (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) hi')
    have hnbit : n.testBit i = false := Nat.testBit_eq_false_of_lt hn'
    simp [hi, hnbit]

private lemma mkBits_toNat (sign : Bool) (exp frac : Nat) (hexp : exp < 256) (hfrac : frac < 2 ^ 23)
  :
    (mkBits sign exp frac).toNat = (if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac := by
  have hFracMask : fracMask.toNat = 2 ^ 23 - 1 := by decide
  have hMod32 : (2 ^ 32 : Nat) = 4294967296 := by decide
  have hfrac_lt32 : frac < 4294967296 := by
    simpa [hMod32] using
      (lt_of_lt_of_le hfrac (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) (by decide : (23 :
        Nat) ≤ 32)))
  have hExp_lt32 : exp < 4294967296 := by
    have : exp < 2 ^ 32 :=
      lt_of_lt_of_le hexp (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) (by decide : (8 : Nat) ≤
        32))
    simpa [hMod32] using this
  have hExp_mod : exp % 4294967296 = exp := Nat.mod_eq_of_lt hExp_lt32

  have hFracField : ((UInt32.ofNat frac) &&& fracMask).toNat = frac := by
    calc
      ((UInt32.ofNat frac) &&& fracMask).toNat
          = (UInt32.ofNat frac).toNat &&& fracMask.toNat := by simp [UInt32.toNat_and]
      _ = (frac % 4294967296) &&& (2 ^ 23 - 1) := by
            simp [hFracMask, hMod32]
      _ = frac &&& (2 ^ 23 - 1) := by
            simp [Nat.mod_eq_of_lt hfrac_lt32]
      _ = frac := nat_and_two_pow_sub_one_eq_self hfrac

  have hShift_lt32 : exp <<< 23 < 4294967296 := by
    have hexp' : exp < 2 ^ 8 := by simpa using hexp
    have : exp <<< 23 = exp * 2 ^ 23 := by simp [Nat.shiftLeft_eq]
    rw [this]
    have hmul : exp * 2 ^ 23 < (2 ^ 8) * 2 ^ 23 :=
      Nat.mul_lt_mul_of_pos_right hexp' (Nat.pow_pos (by decide : 0 < (2 : Nat)))
    have h31 : (2 ^ 8) * 2 ^ 23 = 2 ^ 31 := by
      simp []
    have hlt31 : exp * 2 ^ 23 < 2 ^ 31 := by simpa [h31] using hmul
    have hlt32 : 2 ^ 31 ≤ 2 ^ 32 :=
      Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) (by decide : (31 : Nat) ≤ 32)
    have hle : 2 ^ 31 ≤ 4294967296 := by
      have h := hlt32
      rw [hMod32] at h
      exact h
    exact lt_of_lt_of_le hlt31 hle
  have hShift_mod : (exp <<< 23) % 4294967296 = exp <<< 23 := Nat.mod_eq_of_lt hShift_lt32

  by_cases hs : sign
  · have hs' : sign = true := by simpa using hs
    have hSign : ((1 : UInt32) <<< (UInt32.ofNat 31)).toNat = 2 ^ 31 := by
      simp [UInt32.toNat_shiftLeft]
    simp [mkBits, hs', UInt32.toNat_or, hFracField, UInt32.toNat_shiftLeft, UInt32.toNat_ofNat,
      hMod32, hExp_mod, hShift_mod]
  · have hs' : sign = false := by simpa using hs
    simp [mkBits, hs', UInt32.toNat_or, hFracField, UInt32.toNat_shiftLeft, UInt32.toNat_ofNat,
      hMod32, hExp_mod, hShift_mod]

private lemma nat_extract_fracField (sign : Bool) (exp frac : Nat)
    (_hexp : exp < 256) (hfrac : frac < 2 ^ 23) :
    (((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) &&& (2 ^ 23 - 1)) = frac := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < 23
  · have hExp : Nat.testBit (exp <<< 23) i = false := by
      have : ¬ i ≥ 23 := not_le_of_gt hi
      simp [Nat.testBit_shiftLeft, this]
    have hSign : Nat.testBit (if sign then 2 ^ 31 else 0) i = false := by
      cases hs : sign with
      | false => simp []
      | true =>
          have : 31 ≠ i := by
            have hlt : i < 31 := lt_trans hi (by decide : 23 < 31)
            exact ne_of_gt hlt
          simpa [hs] using (Nat.testBit_two_pow_of_ne (n := 31) (m := i) this)
    calc
      Nat.testBit (((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) &&& (2 ^ 23 - 1)) i
          =
          (Nat.testBit ((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) i &&
              Nat.testBit (2 ^ 23 - 1) i) := by
            simp []
      _ = Nat.testBit ((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) i := by
            rw [Nat.testBit_two_pow_sub_one 23 i]
            simp [hi]
      _ = Nat.testBit frac i := by
            rw [Nat.testBit_lor]
            rw [Nat.testBit_lor]
            rw [hSign, hExp]
            simp
  · have hi' : 23 ≤ i := Nat.le_of_not_gt hi
    have hfracbit : Nat.testBit frac i = false := by
      have hlt : frac < 2 ^ i :=
        lt_of_lt_of_le hfrac (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) hi')
      exact Nat.testBit_eq_false_of_lt hlt
    calc
      Nat.testBit (((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) &&& (2 ^ 23 - 1)) i
          =
          (Nat.testBit ((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) i &&
              Nat.testBit (2 ^ 23 - 1) i) := by
            simp []
      _ = false := by
            rw [Nat.testBit_two_pow_sub_one 23 i]
            simp [hi]
      _ = Nat.testBit frac i := by simp [hfracbit]

private lemma nat_extract_expField (sign : Bool) (exp frac : Nat)
    (hexp : exp < 256) (hfrac : frac < 2 ^ 23) :
    ((((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) >>> 23) &&& (2 ^ 8 - 1)) = exp := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < 8
  · have hfrac_hi : Nat.testBit frac (23 + i) = false := by
      have hlt : frac < 2 ^ (23 + i) :=
        lt_of_lt_of_le hfrac (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) (Nat.le_add_right _
          _))
      exact Nat.testBit_eq_false_of_lt hlt
    have hsign_hi : Nat.testBit (if sign then 2 ^ 31 else 0) (23 + i) = false := by
      cases hs : sign with
      | false => simp []
      | true =>
          have hlt : 23 + i < 31 := by
            have : 23 + i < 23 + 8 := Nat.add_lt_add_left hi 23
            simpa using this
          have : 31 ≠ 23 + i := ne_of_gt hlt
          simpa [hs] using (Nat.testBit_two_pow_of_ne (n := 31) (m := 23 + i) this)
    have hExpBit : Nat.testBit (exp <<< 23) (23 + i) = Nat.testBit exp i := by
      simp [Nat.testBit_shiftLeft]
    calc
      Nat.testBit ((((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) >>> 23) &&& (2 ^ 8 -
        1)) i
          = (Nat.testBit (((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) >>> 23) i) := by
              simp []
              rw [Nat.testBit_two_pow_sub_one 8 i]
              simp [hi]
      _ = Nat.testBit ((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) (23 + i) := by
              simp [Nat.testBit_shiftRight]
      _ = Nat.testBit (exp <<< 23) (23 + i) := by
              rw [Nat.testBit_lor]
              rw [Nat.testBit_lor]
              rw [hsign_hi, hfrac_hi]
              simp
      _ = Nat.testBit exp i := by simp [hExpBit]
  · have hi' : 8 ≤ i := Nat.le_of_not_gt hi
    have hExp : Nat.testBit exp i = false := by
      have hlt : exp < 2 ^ i :=
        lt_of_lt_of_le hexp (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) hi')
      exact Nat.testBit_eq_false_of_lt hlt
    calc
      Nat.testBit ((((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) >>> 23) &&& (2 ^ 8 -
        1)) i
          =
          (Nat.testBit (((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) >>> 23) i &&
              Nat.testBit (2 ^ 8 - 1) i) := by
            simp []
      _ = false := by
            rw [Nat.testBit_two_pow_sub_one 8 i]
            simp [hi]
      _ = Nat.testBit exp i := by simp [hExp]

private lemma fracField_ofBits_mkBits (sign : Bool) (exp frac : Nat) (hexp : exp < 256) (hfrac :
  frac < 2 ^ 23) :
    fracField (ofBits (mkBits sign exp frac)) = UInt32.ofNat frac := by
  apply UInt32.toNat_inj.1
  have hBits := mkBits_toNat (sign := sign) (exp := exp) (frac := frac) hexp hfrac
  have hFracMask : fracMask.toNat = 2 ^ 23 - 1 := by decide
  have hMod32 : (2 ^ 32 : Nat) = 4294967296 := by decide
  have hfrac_lt32 : frac < 4294967296 := by
    simpa [hMod32] using
      (lt_of_lt_of_le hfrac (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) (by decide : (23 :
        Nat) ≤ 32)))
  have hfrac_mod : frac % 4294967296 = frac := Nat.mod_eq_of_lt hfrac_lt32
  calc
    (fracField (ofBits (mkBits sign exp frac))).toNat
        = ((mkBits sign exp frac).toNat &&& fracMask.toNat) := by
            simp [fracField, ofBits, UInt32.toNat_and]
    _ = (((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) &&& (2 ^ 23 - 1)) := by
          simp [hBits, hFracMask]
    _ = frac := nat_extract_fracField (sign := sign) (exp := exp) (frac := frac) hexp hfrac
    _ = (UInt32.ofNat frac).toNat := by
          simp [hfrac_mod]

private lemma expField_ofBits_mkBits (sign : Bool) (exp frac : Nat) (hexp : exp < 256) (hfrac : frac
  < 2 ^ 23) :
    expField (ofBits (mkBits sign exp frac)) = UInt32.ofNat exp := by
  apply UInt32.toNat_inj.1
  have hBits := mkBits_toNat (sign := sign) (exp := exp) (frac := frac) hexp hfrac
  have hExpAllOnes : expAllOnes.toNat = 2 ^ 8 - 1 := by decide
  have hMod32 : (2 ^ 32 : Nat) = 4294967296 := by decide
  have hexp_lt32 : exp < 4294967296 := by
    have : exp < 2 ^ 32 :=
      lt_of_lt_of_le hexp (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) (by decide : (8 : Nat) ≤
        32))
    simpa [hMod32] using this
  have hexp_mod : exp % 4294967296 = exp := Nat.mod_eq_of_lt hexp_lt32
  calc
    (expField (ofBits (mkBits sign exp frac))).toNat
        = (((mkBits sign exp frac).toNat >>> 23) &&& expAllOnes.toNat) := by
            simp [expField, ofBits, UInt32.toNat_and, UInt32.toNat_shiftRight]
    _ = ((((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) >>> 23) &&& (2 ^ 8 - 1)) := by
          simp [hBits, hExpAllOnes]
    _ = exp := nat_extract_expField (sign := sign) (exp := exp) (frac := frac) hexp hfrac
    _ = (UInt32.ofNat exp).toNat := by
          simp [hexp_mod]

private lemma nat_extract_signField (sign : Bool) (exp frac : Nat)
    (hexp : exp < 256) (hfrac : frac < 2 ^ 23) :
    (((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) &&& (2 ^ 31)) =
      (if sign then 2 ^ 31 else 0) := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i = 31
  · subst hi
    have hExp31 : Nat.testBit (exp <<< 23) 31 = false := by
      have hbit8 : Nat.testBit exp 8 = false := by
        have hlt : exp < 2 ^ 8 := by simpa using hexp
        exact Nat.testBit_eq_false_of_lt hlt
      simp [Nat.testBit_shiftLeft, hbit8]
    have hFrac31 : Nat.testBit frac 31 = false := by
      have hlt : frac < 2 ^ 31 :=
        lt_of_lt_of_le hfrac (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) (by decide : (23 :
          Nat) ≤ 31))
      exact Nat.testBit_eq_false_of_lt hlt
    calc
      Nat.testBit (((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) &&& (2 ^ 31)) 31
          =
          (Nat.testBit ((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) 31 &&
              Nat.testBit (2 ^ 31) 31) := by
            simp []
      _ = Nat.testBit ((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) 31 := by
            have h31 : Nat.testBit (2 ^ 31) 31 = true := by
              simpa using (Nat.testBit_two_pow_self (n := 31))
            cases h : Nat.testBit ((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) 31 with
            | false => simp []
            | true => simpa [h] using h31
      _ = Nat.testBit (if sign then 2 ^ 31 else 0) 31 := by
            rw [Nat.testBit_lor]
            rw [Nat.testBit_lor]
            rw [hExp31, hFrac31]
            simp
  · have hrhs : Nat.testBit (if sign then 2 ^ 31 else 0) i = false := by
      cases hs : sign with
      | false => simp []
      | true =>
          simpa [hs] using (Nat.testBit_two_pow_of_ne (n := 31) (m := i) (Ne.symm hi))
    calc
      Nat.testBit (((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) &&& (2 ^ 31)) i
          =
          (Nat.testBit ((if sign then 2 ^ 31 else 0) ||| (exp <<< 23) ||| frac) i &&
              Nat.testBit (2 ^ 31) i) := by
            simp []
      _ = false := by
            rw [Nat.testBit_two_pow_of_ne (n := 31) (m := i) (Ne.symm hi)]
            simp
      _ = Nat.testBit (if sign then 2 ^ 31 else 0) i := by
            simpa [hrhs]

private lemma signBit_ofBits_mkBits (sign : Bool) (exp frac : Nat) (hexp : exp < 256) (hfrac : frac
  < 2 ^ 23) :
    signBit (ofBits (mkBits sign exp frac)) = sign := by
  have hBits := mkBits_toNat (sign := sign) (exp := exp) (frac := frac) hexp hfrac
  have hSignMask : signMask.toNat = 2 ^ 31 := by decide
  have hAnd : (mkBits sign exp frac &&& signMask) = (if sign then signMask else 0) := by
    apply UInt32.toNat_inj.1
    have hL : (mkBits sign exp frac &&& signMask).toNat = (if sign then 2 ^ 31 else 0) := by
      simp [UInt32.toNat_and, hBits, hSignMask]
      exact nat_extract_signField (sign := sign) (exp := exp) (frac := frac) hexp hfrac
    have hR : ((if sign then signMask else 0) : UInt32).toNat = (if sign then 2 ^ 31 else 0) := by
      cases hs : sign <;> simp [hSignMask]
    simp [hL, hR]
  cases hs : sign with
  | false =>
      have hAnd0 : mkBits false exp frac &&& signMask = 0 := by
        simpa [hs] using hAnd
      simp [signBit, ofBits, hAnd0]
  | true =>
      have hAnd1 : mkBits true exp frac &&& signMask = signMask := by
        simpa [hs] using hAnd
      have hSignMaskNe0 : (signMask != (0 : UInt32)) = true := by decide
      simp [signBit, ofBits, hAnd1, hSignMaskNe0]

/--
Decode `ofBits (mkBits sign exp frac)` to a dyadic in the finite range (`exp < 255`).

This is the exact decoding rule used throughout IEEEExec proofs that construct floats from their
fields and then want to reason about their real meaning via `toDyadic?`.
-/
theorem toDyadic?_ofBits_mkBits_fin (sign : Bool) (exp frac : Nat)
    (hexp : exp < 255) (hfrac : frac < 2 ^ 23) :
    toDyadic? (ofBits (mkBits sign exp frac)) =
      if exp = 0 then
        if frac = 0 then
          some { sign := sign, mant := 0, exp := 0 }
        else
          some { sign := sign, mant := frac, exp := -149 }
      else
        some { sign := sign, mant := pow2 23 + frac, exp := (Int.ofNat exp) - 150 } := by
  have hexp256 : exp < 256 := lt_trans hexp (by decide : (255 : Nat) < 256)
  have hs : signBit (ofBits (mkBits sign exp frac)) = sign :=
    signBit_ofBits_mkBits (sign := sign) (exp := exp) (frac := frac) hexp256 hfrac
  have he : expField (ofBits (mkBits sign exp frac)) = UInt32.ofNat exp :=
    expField_ofBits_mkBits (sign := sign) (exp := exp) (frac := frac) hexp256 hfrac
  have hf : fracField (ofBits (mkBits sign exp frac)) = UInt32.ofNat frac :=
    fracField_ofBits_mkBits (sign := sign) (exp := exp) (frac := frac) hexp256 hfrac

  have hMod32 : (2 ^ 32 : Nat) = 4294967296 := by decide
  have hexp_lt32 : exp < 4294967296 := by
    have : exp < 2 ^ 32 :=
      lt_of_lt_of_le hexp256 (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) (by decide : (8 :
        Nat) ≤ 32))
    simpa [hMod32] using this
  have hexp_mod : exp % 4294967296 = exp := Nat.mod_eq_of_lt hexp_lt32
  have hfrac_lt32 : frac < 4294967296 := by
    simpa [hMod32] using
      (lt_of_lt_of_le hfrac (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) (by decide : (23 :
        Nat) ≤ 32)))
  have hfrac_mod : frac % 4294967296 = frac := Nat.mod_eq_of_lt hfrac_lt32

  have hExpAll : (UInt32.ofNat exp == expAllOnes) = false := by
    apply (beq_eq_false_iff_ne).2
    intro hEq
    have hEqNat := congrArg UInt32.toNat hEq
    have : exp = expAllOnes.toNat := by
      simpa [UInt32.toNat_ofNat, hexp_mod] using hEqNat
    exact (Nat.ne_of_lt hexp) (by simpa [expAllOnes] using this)

  have hNoSpecial :
      (isNaN (ofBits (mkBits sign exp frac)) || isInf (ofBits (mkBits sign exp frac))) = false := by
    simp [isNaN, isInf, he, hf, hExpAll]

  by_cases hExp0 : exp = 0
  · subst hExp0
    by_cases hFrac0 : frac = 0
    · subst hFrac0
      simp [toDyadic?, hNoSpecial, hs, he, hf]
    · have hF0 : UInt32.ofNat frac ≠ (0 : UInt32) := by
        intro hEq
        have hEqNat := congrArg UInt32.toNat hEq
        have : frac = 0 := by
          simpa [UInt32.toNat_ofNat, hfrac_mod] using hEqNat
        exact hFrac0 this
      simp [toDyadic?, hNoSpecial, hs, he, hf, hFrac0, hF0, hfrac_mod]
  · have hE0 : (UInt32.ofNat exp == (0 : UInt32)) = false := by
      apply (beq_eq_false_iff_ne).2
      intro hEq
      have hEqNat := congrArg UInt32.toNat hEq
      have : exp = 0 := by
        simpa [UInt32.toNat_ofNat, hexp_mod] using hEqNat
      exact hExp0 this
    simp [toDyadic?, hNoSpecial, hs, he, hf, pow2, Nat.shiftLeft_eq, hExp0, hE0, hexp_mod,
      hfrac_mod]

end IEEE32Exec
end TorchLean.Floats.IEEE754
