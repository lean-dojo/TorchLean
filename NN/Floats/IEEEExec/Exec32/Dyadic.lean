/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32.Core
import Mathlib.Tactic.IntervalCases
public import Aesop.BuiltinRules
public import Mathlib.Algebra.Group.Defs
import Mathlib.Tactic.ToDual
import Mathlib.Data.Nat.Bitwise -- shake: keep

/-!
Dyadic helpers for executable IEEE32 arithmetic.

This file contains the integer/rational scaffolding used to describe binary significands,
exponents, and exact dyadic values before rounding to IEEE32.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

/-- Exact dyadic value `(-1)^sign * mant * 2^exp` used as an intermediate for finite ops. -/
structure Dyadic where
  /-- Sign of the value, with `true` meaning negative. Kept separate from `mant` so the
  significand can stay a `Nat` and the rounding code never has to reason about negative division. -/
  sign : Bool
  /-- Unnormalized significand. It is a plain `Nat`, so an exact product of two binary32
  significands (up to 48 bits) fits without any intermediate rounding. -/
  mant : Nat
  /-- Binary exponent, so the represented value is `(-1)^sign * mant * 2^exp`. `Int` rather than a
  bounded field: intermediate exponents during multiplication and FMA leave the binary32 range. -/
  exp : Int
  deriving Repr, DecidableEq

/-- `2^k` as a natural number. -/
@[inline] def pow2 (k : Nat) : Nat :=
  Nat.shiftLeft 1 k

/--
Round `n / 2^shift` to nearest, ties-to-even.

This is the primitive "shift + rounding" operation we use when shrinking a mantissa back down to a
fixed bit width. It is the same tie-breaking policy as IEEE round-to-nearest-even.
-/
@[inline] def roundShiftRightEven (n : Nat) (shift : Nat) : Nat :=
  if shift == 0 then
    n
  else
    let q := Nat.shiftRight n shift
    let rem := n - Nat.shiftLeft q shift
    let half := pow2 (shift - 1)
    if rem < half then q
    else if rem > half then q + 1
    else if q % 2 == 0 then q else q + 1

/--
Construct a raw binary32 bit-pattern from fields.

`mkBits sign exp frac` places the low bits of each supplied field:
- `sign` in bit 31,
- the low 8 bits of `exp` in bits 30..23,
- `frac` in bits 22..0 (masked to 23 bits).
-/
@[inline] def mkBits (sign : Bool) (exp : Nat) (frac : Nat) : UInt32 :=
  let s : UInt32 := if sign then (1 : UInt32) <<< 31 else 0
  let e : UInt32 := ((UInt32.ofNat exp) &&& expAllOnes) <<< 23
  let f : UInt32 := (UInt32.ofNat frac) &&& fracMask
  s ||| e ||| f

/-- Reassembling the decoded sign, exponent, and fraction fields recovers the original value. -/
theorem ofBits_mkBits_fields (x : IEEE32Exec) :
    ofBits (mkBits (signBit x) (expField x).toNat (fracField x).toNat) = x := by
  cases x with
  | mk bits =>
      apply congrArg IEEE32Exec.mk
      simp only [mkBits, expField, fracField, UInt32.ofNat_toNat]
      rw [← UInt32.toBitVec_inj]
      cases hsign : (bits &&& signMask != 0) with
      | false =>
          simp [bne, signMask] at hsign
          have hsignBV := congrArg UInt32.toBitVec hsign
          simp [UInt32.toBitVec_and] at hsignBV
          have hbit : bits.toBitVec.getLsbD 31 = false := by
            have h := congrArg (fun x : BitVec 32 => x.getLsbD 31) hsignBV
            simpa [BitVec.getLsbD_and, BitVec.getLsbD_ofNat] using h
          simp [signBit, signMask, expAllOnes, fracMask,
            UInt32.toBitVec_shiftRight, UInt32.toBitVec_shiftLeft,
            UInt32.toBitVec_and, UInt32.toBitVec_or, hsign]
          apply BitVec.eq_of_getLsbD_eq
          intro i hi
          interval_cases i <;> simp_all
      | true =>
          simp [bne, signMask] at hsign
          have hsignBV : bits.toBitVec &&& 2147483648#32 ≠ 0#32 := by
            intro h
            apply hsign
            rw [← UInt32.toBitVec_inj]
            simpa [UInt32.toBitVec_and]
          have hbit : bits.toBitVec.getLsbD 31 = true := by
            cases hbitEq : bits.toBitVec.getLsbD 31
            · exfalso
              apply hsignBV
              apply BitVec.eq_of_getLsbD_eq
              intro i hi
              interval_cases i <;> simp_all
            · rfl
          simp [signBit, signMask, expAllOnes, fracMask,
            UInt32.toBitVec_shiftRight, UInt32.toBitVec_shiftLeft,
            UInt32.toBitVec_and, UInt32.toBitVec_or, hsign]
          apply BitVec.eq_of_getLsbD_eq
          intro i hi
          interval_cases i <;> simp_all

/--
Decode a finite binary32 into an exact dyadic value.

Returns `none` for NaN/Inf.
-/
@[inline] def toDyadic? (x : IEEE32Exec) : Option Dyadic :=
  if isNaN x || isInf x then
    none
  else
    let s := signBit x
    let e := expField x
    let f := fracField x
    if e == 0 then
      if f == 0 then
        some { sign := s, mant := 0, exp := 0 }
      else
        -- subnormal: value = frac * 2^-149
        some { sign := s, mant := f.toNat, exp := -149 }
    else
      -- normal: value = (2^23 + frac) * 2^(e-bias-23) = (2^23+frac) * 2^(e - 150)
      let mant := (pow2 23) + f.toNat
      let exp : Int := (Int.ofNat e.toNat) - 150
      some { sign := s, mant := mant, exp := exp }

/-- Convert an exact dyadic value to its exact rational value. -/
def Dyadic.toRat (d : Dyadic) : Rat :=
  let s : Int := if d.sign then -(Int.ofNat d.mant) else Int.ofNat d.mant
  if d.exp ≥ 0 then
    let e : Nat := Int.toNat d.exp
    Rat.ofInt (s * Int.ofNat (pow2 e))
  else
    let e : Nat := Int.toNat (-d.exp)
    (Rat.ofInt s) / (Rat.ofInt (Int.ofNat (pow2 e)))

/-- Exact rational value of a finite binary32 value; returns `none` for NaN or infinity. -/
def toRat? (x : IEEE32Exec) : Option Rat :=
  (toDyadic? x).map Dyadic.toRat

/--
If `toDyadic? x = some d` then `x` is not a NaN.

Informal: `toDyadic?` only returns `some _` for finite floats; NaNs map to `none`.
-/
lemma isNaN_eq_false_of_toDyadic?_some {x : IEEE32Exec} {d : Dyadic}
    (hx : toDyadic? x = some d) : isNaN x = false := by
  cases h : isNaN x
  · rfl
  · simp [toDyadic?, h] at hx

/--
If `toDyadic? x = some d` then `x` is not an infinity.

Informal: `toDyadic?` only returns `some _` for finite floats; infinities map to `none`.
-/
lemma isInf_eq_false_of_toDyadic?_some {x : IEEE32Exec} {d : Dyadic}
    (hx : toDyadic? x = some d) : isInf x = false := by
  cases h : isInf x
  · rfl
  · simp [toDyadic?, h] at hx

/-- A successful dyadic decoding certifies that the source bit pattern is finite. -/
theorem isFinite_eq_true_of_toDyadic?_some {x : IEEE32Exec} {d : Dyadic}
    (hx : toDyadic? x = some d) : isFinite x = true := by
  unfold isFinite
  apply (bne_iff_ne).2
  intro hexp
  have hexpB : (expField x == expAllOnes) = true := (beq_iff_eq).2 hexp
  by_cases hfrac : fracField x = 0
  · have hinf : isInf x = true := by simp [isInf, hexpB, hfrac]
    have hnotInf := isInf_eq_false_of_toDyadic?_some hx
    simp [hinf] at hnotInf
  · have hnan : isNaN x = true := by simp [isNaN, hexpB, hfrac]
    have hnotNaN := isNaN_eq_false_of_toDyadic?_some hx
    simp [hnan] at hnotNaN

/-- The extracted binary32 fraction field fits in its 23-bit storage width. -/
theorem fracField_toNat_lt_pow2_23 (x : IEEE32Exec) :
    (fracField x).toNat < 2 ^ 23 := by
  have hfracMask : fracMask.toNat = 2 ^ 23 - 1 := by decide
  have hfracLe : (fracField x).toNat ≤ fracMask.toNat := by
    simp [fracField, UInt32.toNat_and]
    apply Nat.le_of_testBit
    intro i hi
    have hi' :
        Nat.testBit x.bits.toNat i = true ∧ Nat.testBit fracMask.toNat i = true := by
      simpa [Nat.testBit_land, Bool.and_eq_true] using hi
    exact hi'.2
  have hmaskLt : fracMask.toNat < 2 ^ 23 := by
    rw [hfracMask]
    exact Nat.sub_lt (Nat.pow_pos (by decide)) (by decide)
  exact lt_of_le_of_lt hfracLe hmaskLt

/-- A finite binary32 value has a biased exponent field at most `254`. -/
theorem expField_toNat_le_254_of_isFinite (x : IEEE32Exec)
    (hfin : isFinite x = true) :
    (expField x).toNat ≤ 254 := by
  have hexpNe : expField x ≠ expAllOnes := (bne_iff_ne).mp hfin
  have hexpLe : (expField x).toNat ≤ expAllOnes.toNat := by
    simp [expField, UInt32.toNat_and]
    apply Nat.le_of_testBit
    intro i hi
    have hi' :
        Nat.testBit ((x.bits >>> 23).toNat) i = true ∧
          Nat.testBit expAllOnes.toNat i = true := by
      simpa [Nat.testBit_land, Bool.and_eq_true] using hi
    exact hi'.2
  have hexpNe255 : (expField x).toNat ≠ 255 := by
    intro h
    apply hexpNe
    apply UInt32.toNat_inj.mp
    simpa [show expAllOnes.toNat = 255 by decide, UInt32.toNat_ofNat] using h
  have hexpLt255 : (expField x).toNat < 255 := by
    exact lt_of_le_of_ne (by simpa [show expAllOnes.toNat = 255 by decide] using hexpLe) hexpNe255
  grind

/-- Dyadic decoding preserves the sign bit of every finite executable value. -/
theorem sign_eq_signBit_of_toDyadic?_some {x : IEEE32Exec} {d : Dyadic}
    (hx : toDyadic? x = some d) : d.sign = signBit x := by
  have hnan : isNaN x = false := isNaN_eq_false_of_toDyadic?_some hx
  have hinf : isInf x = false := isInf_eq_false_of_toDyadic?_some hx
  unfold toDyadic? at hx
  simp only [hnan, hinf, Bool.false_or, Bool.false_eq_true, if_false] at hx
  split at hx
  · split at hx
    · simpa using congrArg Dyadic.sign (Option.some.inj hx.symm)
    · simpa using congrArg Dyadic.sign (Option.some.inj hx.symm)
  · simpa using congrArg Dyadic.sign (Option.some.inj hx.symm)

/-- NaNs have no finite dyadic decoding. -/
@[simp] lemma toDyadic?_eq_none_of_isNaN {x : IEEE32Exec} (hx : isNaN x = true) :
    toDyadic? x = none := by
  simp [toDyadic?, hx]

/-- Infinities have no finite dyadic decoding. -/
@[simp] lemma toDyadic?_eq_none_of_isInf {x : IEEE32Exec} (hx : isInf x = true) :
    toDyadic? x = none := by
  simp [toDyadic?, hx]

/-- A value that is neither NaN nor infinity has an exact dyadic decoding. -/
lemma exists_toDyadic?_of_not_isNaN_not_isInf {x : IEEE32Exec}
    (hnan : isNaN x = false) (hinf : isInf x = false) :
    ∃ d, toDyadic? x = some d := by
  by_cases he : expField x = 0
  · by_cases hf : fracField x = 0
    · exact ⟨Dyadic.mk (signBit x) 0 0, by simp [toDyadic?, hnan, hinf, he, hf]⟩
    · exact ⟨Dyadic.mk (signBit x) (fracField x).toNat (-149),
        by simp [toDyadic?, hnan, hinf, he, hf]⟩
  · exact ⟨Dyadic.mk (signBit x) (pow2 23 + (fracField x).toNat)
        (Int.ofNat (expField x).toNat - 150), by simp [toDyadic?, hnan, hinf, he]⟩

/-- Every finite binary32 value has an exact dyadic decoding. -/
lemma exists_toDyadic?_of_isFinite {x : IEEE32Exec} (hx : isFinite x = true) :
    ∃ d, toDyadic? x = some d := by
  have hexp : expField x ≠ expAllOnes := (bne_iff_ne).mp hx
  have hnan : isNaN x = false := by simp [isNaN, hexp]
  have hinf : isInf x = false := by simp [isInf, hexp]
  exact exists_toDyadic?_of_not_isNaN_not_isInf hnan hinf

/-- Dyadic decoding succeeds exactly on finite executable binary32 values. -/
theorem toDyadic?_isSome_eq_isFinite (x : IEEE32Exec) :
    (toDyadic? x).isSome = isFinite x := by
  cases hdy : toDyadic? x with
  | some d =>
      have hfin := isFinite_eq_true_of_toDyadic?_some hdy
      simp [hfin]
  | none =>
      cases hfin : isFinite x with
      | false => rfl
      | true =>
          obtain ⟨d, hd⟩ := exists_toDyadic?_of_isFinite hfin
          rw [hdy] at hd
          contradiction

/-- Both signed zeros decode to the zero dyadic while retaining their sign bit. -/
@[simp] lemma toDyadic?_eq_zero_of_isZero {x : IEEE32Exec} (hx : isZero x = true) :
    toDyadic? x = some { sign := signBit x, mant := 0, exp := 0 } := by
  have hfields : (expField x == 0) = true ∧ (fracField x == 0) = true := by
    simpa [isZero, Bool.and_eq_true] using hx
  have hnan : isNaN x = false := by
    simp [isNaN, (beq_iff_eq).1 hfields.1, expAllOnes]
  have hinf : isInf x = false := by
    simp [isInf, (beq_iff_eq).1 hfields.1, expAllOnes]
  simp [toDyadic?, hnan, hinf, hfields.1, hfields.2]

/-- A decoded dyadic has zero mantissa only when the source bit pattern is a signed zero. -/
lemma isZero_eq_true_of_toDyadic?_some_of_mant_eq_zero {x : IEEE32Exec} {d : Dyadic}
    (hx : toDyadic? x = some d) (hm : d.mant = 0) : isZero x = true := by
  unfold toDyadic? at hx
  have hcond : (isNaN x || isInf x) = false := by
    cases h : (isNaN x || isInf x) <;> simp [h] at hx
    exact rfl
  set e : UInt32 := expField x
  set f : UInt32 := fracField x
  cases he : (e == 0) with
  | true =>
      cases hf : (f == 0) with
      | true => simp [isZero, e, f, he, hf]
      | false =>
          have hto : some (Dyadic.mk (signBit x) f.toNat (-149)) = some d := by
            simpa [hcond, e, f, he, hf] using hx
          have hmant : d.mant = f.toNat := by
            simpa using (congrArg Dyadic.mant (Option.some.inj hto)).symm
          have hfne : f ≠ 0 := (beq_eq_false_iff_ne).1 hf
          have htoNat : f.toNat ≠ 0 := by
            intro h0
            exact hfne ((UInt32.toNat_inj).1 (by simp [h0]))
          exact (htoNat (by simpa [hmant] using hm)).elim
  | false =>
      have hto : some (Dyadic.mk (signBit x) (pow2 23 + f.toNat)
          (Int.ofNat e.toNat - 150)) = some d := by
        simpa [hcond, e, f, he] using hx
      have hmant : d.mant = pow2 23 + f.toNat := by
        simpa using (congrArg Dyadic.mant (Option.some.inj hto)).symm
      have hpos : 0 < d.mant := by
        rw [hmant]
        exact lt_of_lt_of_le (by decide : 0 < pow2 23) (Nat.le_add_right (pow2 23) f.toNat)
      exact (Nat.ne_of_gt hpos hm).elim

/-- A decoded dyadic with nonzero mantissa cannot have come from either signed zero. -/
lemma isZero_eq_false_of_toDyadic?_some_of_mant_ne_zero {x : IEEE32Exec} {d : Dyadic}
    (hx : toDyadic? x = some d) (hm : d.mant ≠ 0) : isZero x = false := by
  cases hzero : isZero x with
  | false => rfl
  | true =>
      have hdy := toDyadic?_eq_zero_of_isZero hzero
      rw [hx] at hdy
      have hmant : d.mant = 0 :=
        congrArg Dyadic.mant (Option.some.inj hdy)
      exact (hm hmant).elim

/-!
## Rounding back to binary32

The general pattern for the finite ops in this file is:

1. decode float32(s) to an exact intermediate representation (`Dyadic` for `+ - * fma sqrt`,
   rationals for `/`),
2. compute the exact result in that intermediate representation,
3. round once to float32 using round-to-nearest, ties-to-even.
-/

/--
Round an exact dyadic value to binary32 (ties-to-even).

This function implements:

- overflow to ±Inf,
- gradual underflow into subnormals (down to exponent `-149`),
- underflow-to-zero below `2^-150` (half the minimum subnormal, where ties-to-even chooses 0),
- mantissa rounding to the 24-bit precision of binary32 normal numbers.
-/
@[inline] def roundDyadicToIEEE32 (d : Dyadic) : IEEE32Exec :=
  -- Exact 0 becomes signed 0.
  if d.mant == 0 then
    if d.sign then negZero else posZero
  else
    let log2m : Nat := Nat.log2 d.mant
    let k : Int := (Int.ofNat log2m) + d.exp
    -- IEEE754 binary32 exponent range (unbiased): normal [-126,127], subnormal down to -149.
    if k > 127 then
      if d.sign then negInf else posInf
    -- Underflow-to-zero threshold is at half the smallest subnormal: 2^-150 (ties-to-even pick 0).
    -- In terms of `k = ⌊log₂ |x|⌋`, all values with `k < -150` round to 0.
    else if k < -150 then
      if d.sign then negZero else posZero
    else if k < -126 then
      -- subnormal rounding: frac = round_to_even( mant * 2^(exp+149) )
      let fracNat : Nat :=
        match d.exp + 149 with
        | .ofNat sh => Nat.shiftLeft d.mant sh
        | .negSucc sh => roundShiftRightEven d.mant (sh + 1)
      if fracNat == 0 then
        if d.sign then negZero else posZero
      else
        match Nat.decLe (pow2 23) fracNat with
        | isTrue _ =>
            -- Rounds up to the smallest normal: exp=1, frac=0.
            ofBits (mkBits d.sign 1 0)
        | isFalse _ =>
            ofBits (mkBits d.sign 0 fracNat)
    else
      -- normal rounding
      let m24 : Nat :=
        if log2m >= 23 then
          roundShiftRightEven d.mant (log2m - 23)
        else
          Nat.shiftLeft d.mant (23 - log2m)
      let k' : Int := if m24 == pow2 24 then k + 1 else k
      let m24' : Nat := if m24 == pow2 24 then pow2 23 else m24
      if k' > 127 then
        if d.sign then negInf else posInf
      else
        let expNat : Nat := Int.toNat (k' + 127)
        let fracNat : Nat := m24' - pow2 23
        ofBits (mkBits d.sign expNat fracNat)

/-- Decoding a finite binary32 value and rounding the exact dyadic back recovers its bits. -/
theorem roundDyadicToIEEE32_of_toDyadic?_some
    {x : IEEE32Exec} {d : Dyadic} (hx : toDyadic? x = some d) :
    roundDyadicToIEEE32 d = x := by
  have hnan : isNaN x = false := isNaN_eq_false_of_toDyadic?_some hx
  have hinf : isInf x = false := isInf_eq_false_of_toDyadic?_some hx
  have hfin : isFinite x = true := isFinite_eq_true_of_toDyadic?_some hx
  by_cases he : expField x = 0
  · by_cases hf : fracField x = 0
    · have hd : d = { sign := signBit x, mant := 0, exp := 0 } := by
        have hsome : some { sign := signBit x, mant := 0, exp := 0 } = some d := by
          simpa [toDyadic?, hnan, hinf, he, hf] using hx
        exact (Option.some.inj hsome).symm
      subst d
      calc
        roundDyadicToIEEE32 { sign := signBit x, mant := 0, exp := 0 } =
            ofBits (mkBits (signBit x) 0 0) := by
          cases signBit x <;> decide
        _ = ofBits (mkBits (signBit x) (expField x).toNat (fracField x).toNat) := by
          simp [he, hf]
        _ = x := ofBits_mkBits_fields x
    · have hd : d =
          { sign := signBit x, mant := (fracField x).toNat, exp := -149 } := by
        have hsome :
            some { sign := signBit x, mant := (fracField x).toNat, exp := -149 } =
              some d := by
          simpa [toDyadic?, hnan, hinf, he, hf] using hx
        exact (Option.some.inj hsome).symm
      subst d
      have hmant : (fracField x).toNat ≠ 0 := by
        intro hzero
        apply hf
        exact UInt32.toNat_inj.mp (by simpa using hzero)
      have hmantLt : (fracField x).toNat < 2 ^ 23 := fracField_toNat_lt_pow2_23 x
      have hlogLt : (fracField x).toNat.log2 < 23 :=
        (Nat.log2_lt hmant).2 hmantLt
      have hkNotOver :
          ¬ (127 : Int) < Int.ofNat (fracField x).toNat.log2 + -149 := by
        simp only [Int.ofNat_eq_natCast]
        grind
      have hkNotUnder :
          ¬ Int.ofNat (fracField x).toNat.log2 + -149 < (-150 : Int) := by
        simp only [Int.ofNat_eq_natCast]
        grind
      have hkSubnormal :
          Int.ofNat (fracField x).toNat.log2 + -149 < (-126 : Int) := by
        simp only [Int.ofNat_eq_natCast]
        grind
      have hpowNotLe : ¬ pow2 23 ≤ (fracField x).toNat := by
        simpa [pow2, Nat.shiftLeft_eq] using Nat.not_le_of_lt hmantLt
      have hshiftZero : (fracField x).toNat.shiftLeft 0 = (fracField x).toNat := by
        change (fracField x).toNat <<< 0 = (fracField x).toNat
        exact Nat.shiftLeft_zero
      calc
        roundDyadicToIEEE32
            { sign := signBit x, mant := (fracField x).toNat, exp := -149 } =
            ofBits (mkBits (signBit x) 0 (fracField x).toNat) := by
          simp only [roundDyadicToIEEE32]
          rw [if_neg (by simpa using hmant)]
          rw [if_neg hkNotOver, if_neg hkNotUnder, if_pos hkSubnormal]
          simp only [Int.reduceNeg, Int.reduceAdd]
          rw [hshiftZero]
          rw [if_neg (by simpa using hmant)]
          cases hdec : Nat.decLe (pow2 23) (fracField x).toNat with
          | isTrue h => exact (hpowNotLe h).elim
          | isFalse _ => rfl
        _ = ofBits (mkBits (signBit x) (expField x).toNat (fracField x).toNat) := by
          simp [he]
        _ = x := ofBits_mkBits_fields x
  · have hd : d =
        { sign := signBit x,
          mant := pow2 23 + (fracField x).toNat,
          exp := Int.ofNat (expField x).toNat - 150 } := by
      have hsome :
          some
              { sign := signBit x,
                mant := pow2 23 + (fracField x).toNat,
                exp := Int.ofNat (expField x).toNat - 150 } = some d := by
        simpa [toDyadic?, hnan, hinf, he] using hx
      exact (Option.some.inj hsome).symm
    subst d
    have hfracLt : (fracField x).toNat < 2 ^ 23 := fracField_toNat_lt_pow2_23 x
    have hmantPos : 0 < pow2 23 + (fracField x).toNat := by
      change 0 < 2 ^ 23 + (fracField x).toNat
      grind
    have hmantLt : pow2 23 + (fracField x).toNat < 2 ^ 24 := by
      change 2 ^ 23 + (fracField x).toNat < 2 ^ 24
      grind
    have hlog : (pow2 23 + (fracField x).toNat).log2 = 23 := by
      rw [Nat.log2_eq_iff (Nat.ne_of_gt hmantPos)]
      change 2 ^ 23 ≤ 2 ^ 23 + (fracField x).toNat ∧
        2 ^ 23 + (fracField x).toNat < 2 ^ (23 + 1)
      grind
    have hexpPos : 0 < (expField x).toNat := by
      have hne : (expField x).toNat ≠ 0 := by
        intro hzero
        apply he
        apply UInt32.toNat_inj.mp
        simpa using hzero
      grind
    have hexpLe : (expField x).toNat ≤ 254 :=
      expField_toNat_le_254_of_isFinite x hfin
    have hkLower : (-126 : Int) ≤ Int.ofNat (expField x).toNat - 127 := by
      simp only [Int.ofNat_eq_natCast]
      grind
    have hkUpper : Int.ofNat (expField x).toNat - 127 ≤ 127 := by
      simp only [Int.ofNat_eq_natCast]
      grind
    have hnotCarry : pow2 23 + (fracField x).toNat ≠ pow2 24 :=
      Nat.ne_of_lt hmantLt
    have hexpToNat :
        (Int.ofNat (expField x).toNat - 127 + 127).toNat = (expField x).toNat := by
      simp only [Int.ofNat_eq_natCast]
      grind
    have hkEq : Int.ofNat 23 + (Int.ofNat (expField x).toNat - 150) =
        Int.ofNat (expField x).toNat - 127 := by
      simp only [Int.ofNat_eq_natCast]
      grind
    have hkNotOver : ¬ Int.ofNat (expField x).toNat - 127 > 127 := by grind
    have hkNotUnder : ¬ Int.ofNat (expField x).toNat - 127 < -150 := by grind
    have hkNormal : ¬ Int.ofNat (expField x).toNat - 127 < -126 := by grind
    have hlogGe : 23 ≤ (pow2 23 + (fracField x).toNat).log2 := by grind
    have hshift :
        roundShiftRightEven (pow2 23 + (fracField x).toNat)
            ((pow2 23 + (fracField x).toNat).log2 - 23) =
          pow2 23 + (fracField x).toNat := by
      simp [hlog, roundShiftRightEven]
    have hmantNe : pow2 23 + (fracField x).toNat ≠ 0 := Nat.ne_of_gt hmantPos
    have hcarryFalse :
        (pow2 23 + (fracField x).toNat == pow2 24) = false :=
      beq_eq_false_iff_ne.mpr hnotCarry
    calc
      roundDyadicToIEEE32
          { sign := signBit x,
            mant := pow2 23 + (fracField x).toNat,
            exp := Int.ofNat (expField x).toNat - 150 } =
          ofBits (mkBits (signBit x) (expField x).toNat (fracField x).toNat) := by
        simp only [roundDyadicToIEEE32]
        rw [if_neg (by simpa using hmantNe)]
        rw [hlog, hkEq]
        rw [if_neg hkNotOver, if_neg hkNotUnder, if_neg hkNormal]
        simp only [ge_iff_le, le_refl, ↓reduceIte, Nat.sub_self, roundShiftRightEven,
          beq_self_eq_true]
        simp only [hcarryFalse, Bool.false_eq_true, ↓reduceIte]
        rw [if_neg hkNotOver]
        rw [hexpToNat]
        simp only [Nat.add_sub_cancel_left]
      _ = x := ofBits_mkBits_fields x

/-!
## Exact dyadic arithmetic (finite core)

`Dyadic` is closed under `+` and `*` (at the exact level). We use these helpers before rounding
back to float32.
-/

/--
Exact dyadic addition.

We align exponents by shifting the mantissa of the operand with the larger exponent, add signed
integers, and then return an exact dyadic (no rounding yet).
-/
@[inline] def addDyadic (a b : Dyadic) : Dyadic :=
  if a.mant == 0 then
    if b.mant == 0 then
      { sign := a.sign && b.sign, mant := 0, exp := 0 }
    else
      b
  else if b.mant == 0 then
    a
  else if a.exp ≤ b.exp then
    let sh : Nat := Int.toNat (b.exp - a.exp)
    let m1 : Int := if a.sign then -(Int.ofNat a.mant) else (Int.ofNat a.mant)
    let m2s : Nat := Nat.shiftLeft b.mant sh
    let m2 : Int := if b.sign then -(Int.ofNat m2s) else (Int.ofNat m2s)
    let s : Int := m1 + m2
    if s == 0 then
      { sign := a.sign && b.sign, mant := 0, exp := 0 }
    else
      { sign := s < 0, mant := Int.natAbs s, exp := a.exp }
  else
    let sh : Nat := Int.toNat (a.exp - b.exp)
    let m1s : Nat := Nat.shiftLeft a.mant sh
    let m1 : Int := if a.sign then -(Int.ofNat m1s) else (Int.ofNat m1s)
    let m2 : Int := if b.sign then -(Int.ofNat b.mant) else (Int.ofNat b.mant)
    let s : Int := m1 + m2
    if s == 0 then
      { sign := a.sign && b.sign, mant := 0, exp := 0 }
    else
      { sign := s < 0, mant := Int.natAbs s, exp := b.exp }

/-!
## Exact rationals for division

For division we compute an exact rational `num/den` (with `den > 0`) and then round it to binary32
using round-to-nearest, ties-to-even.
-/

/-- Round `num/den` to nearest, ties-to-even (assumes `den > 0`). -/
@[inline] def roundQuotEven (num den : Nat) : Nat :=
  let q := num / den
  let r := num % den
  let twice := 2 * r
  if twice < den then q
  else if twice > den then q + 1
  else if q % 2 == 0 then q else q + 1

/-- Test whether `num/den < 2^k` without converting to reals. -/
@[inline] def ratLtPow2 (num den : Nat) (k : Int) : Bool :=
  match k with
  | .ofNat kn => num < Nat.shiftLeft den kn
  | .negSucc kn => Nat.shiftLeft num (kn + 1) < den

/-- Test whether `num/den ≥ 2^k` without converting to reals. -/
@[inline] def ratGePow2 (num den : Nat) (k : Int) : Bool :=
  match k with
  | .ofNat kn => num ≥ Nat.shiftLeft den kn
  | .negSucc kn => Nat.shiftLeft num (kn + 1) ≥ den

/--
Compute `⌊log₂(num/den)⌋` as an `Int` (assumes `num > 0` and `den > 0`).

We start from the initial estimate `log2(num) - log2(den)` and then adjust by checking against
powers of two.
-/
@[inline] def floorLog2Rat (num den : Nat) : Int :=
  -- num > 0, den > 0
  let k0 : Int := (Int.ofNat (Nat.log2 num)) - (Int.ofNat (Nat.log2 den))
  let k1 : Int := if ratLtPow2 num den k0 then k0 - 1 else k0
  if ratGePow2 num den (k1 + 1) then k1 + 1 else k1

/-- Represent `(num / den) * 2^exponent` as a ratio of natural numbers.

Nonnegative exponents shift the numerator; negative exponents shift the denominator. The
operation deliberately does not cancel common factors, since the rounding algorithms consume the
quotient and remainder of this exact representation directly.
-/
@[inline] def scaleRatByPow2 (num den : Nat) : Int → Nat × Nat
  | .ofNat shift => (Nat.shiftLeft num shift, den)
  | .negSucc shift => (num, Nat.shiftLeft den (shift + 1))

/--
Round an exact rational `num/den` to binary32 (ties-to-even).

This is the division analogue of `roundDyadicToIEEE32`: it uses the same exponent thresholds and
the same final mantissa rounding policy.
-/
@[inline] def roundRatToIEEE32 (sign : Bool) (num den : Nat) : IEEE32Exec :=
  if num == 0 then
    if sign then negZero else posZero
  else
    let k : Int := floorLog2Rat num den
    if k > 127 then
      if sign then negInf else posInf
    -- Same underflow threshold as `roundDyadicToIEEE32`.
    else if k < -150 then
      if sign then negZero else posZero
    else if k < -126 then
      -- subnormal: frac = round_to_even( (num/den) * 2^149 )
      let num' := Nat.shiftLeft num 149
      let frac := roundQuotEven num' den
      if frac == 0 then
        if sign then negZero else posZero
      else
        match Nat.decLe (pow2 23) frac with
        | isTrue _ => ofBits (mkBits sign 1 0)
        | isFalse _ => ofBits (mkBits sign 0 frac)
    else
      -- normal: m = round_to_even( (num/den) * 2^(23-k) )
      let shift : Int := 23 - k
      let (num', den') := scaleRatByPow2 num den shift
      let m := roundQuotEven num' den'
      let k' : Int := if m == pow2 24 then k + 1 else k
      let m' : Nat := if m == pow2 24 then pow2 23 else m
      if k' > 127 then
        if sign then negInf else posInf
      else
        let expNat : Nat := Int.toNat (k' + 127)
        let fracNat : Nat := m' - pow2 23
        ofBits (mkBits sign expNat fracNat)

/-- Rounding an exact zero rational preserves its requested sign. -/
@[simp] theorem roundRatToIEEE32_zero (sign : Bool) (den : Nat) :
    roundRatToIEEE32 sign 0 den = if sign then negZero else posZero := by
  simp [roundRatToIEEE32]

/-- Scaling a zero rational by a power of two still rounds to signed zero. -/
theorem roundRatToIEEE32_scaleRatByPow2_zero
    (sign : Bool) (den : Nat) (exponent : Int) :
    let scaled := scaleRatByPow2 0 den exponent
    roundRatToIEEE32 sign scaled.1 scaled.2 = if sign then negZero else posZero := by
  cases exponent <;> simp [scaleRatByPow2]


end IEEE32Exec

end TorchLean.Floats.IEEE754
