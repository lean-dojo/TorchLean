/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Core
public import NN.Floats.NeuralFloat.Format.Theorems
public import NN.Floats.IEEEExec.Encoding.MkBitsToDyadic
public import NN.Floats.IEEEExec.Semantics.RealSemantics
public import NN.Floats.IEEEExec.Encoding.Negation
public import NN.Floats.IEEEExec.Rounding.NatLemmas
public import Mathlib.Algebra.Order.Algebra
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset

/-!
# IEEE32Exec and FP32: Real Semantics Core
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/-!
## Reals view of `IEEE32Exec`

Real semantics (`toReal?`/`toReal`) live in `NN.Floats.IEEEExec.Semantics.RealSemantics`. This bridge file
uses them pervasively, but does not define them.
-/

/-- `FP32` rounding viewed as a real function. -/
noncomputable abbrev fp32Round (x : ℝ) : ℝ :=
  neuralRound (β := binaryRadix) (fexp := TorchLean.Floats.fexp32) TorchLean.Floats.rnd32 x

/-! ### Basic checks for `fp32Round` -/

/-- Rounding `0` to float32 yields `0`. -/
theorem fp32Round_zero : fp32Round 0 = 0 := by
  -- This proof proceeds by unfolding: `fp32Round` is defined via `neuralRound`.
  have hne0 : TorchLean.Floats.neuralNearestEven 0 = 0 := by
    simp [TorchLean.Floats.neuralNearestEven]
  have :
      TorchLean.Floats.neuralNearestEven 0 = 0 ∨ neuralBpow binaryRadix (-24) = 0 :=
    Or.inl hne0
  simpa [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal,
    TorchLean.Floats.neuralScaledMantissa, TorchLean.Floats.neuralCexp,
      TorchLean.Floats.neuralMagnitude,
    TorchLean.Floats.fexp32, TorchLean.Floats.FLTExp, TorchLean.Floats.rnd32] using this

/-!
## Helper lemmas (magnitude/rounding)

Most of the file consists of bridge lemmas: once we decide on the refinement statement, we need many
small facts that connect:

- executable *bitfield* manipulations (extracting sign/exponent/fraction, flipping the sign bit),
- exact *dyadic* arithmetic (what the decoded value means as a real),
- and the `FP32` rounding model (which is expressed using `neuralMagnitude` / nearest-even).

These lemmas are local: they exist to keep the later op-level theorems readable.
-/

/-- The real number a sign bit denotes: `1` for a positive sign, `-1` for a negative one. -/
noncomputable def signFactor (s : Bool) : ℝ :=
  if s then (-1 : ℝ) else (1 : ℝ)

/-- The sign bit is multiplicative: xor on bits becomes multiplication of `±1`.

This is what lets the multiplication and division proofs handle signs once, up front, and then argue
only about magnitudes. -/
lemma signFactor_xor (a b : Bool) :
    signFactor (Bool.xor a b) = signFactor a * signFactor b := by
  by_cases ha : a <;> by_cases hb : b <;> simp [signFactor, Bool.xor, ha, hb]

/--
Absolute value of a decoded dyadic.

Informal: `dyadicToReal d = ± (mant * 2^exp)` and since `mant ≥ 0`, the absolute value is always
`mant * 2^exp` regardless of the sign bit.
-/
theorem abs_dyadicToReal (d : Dyadic) :
    _root_.abs (dyadicToReal d) = (d.mant : ℝ) * neuralBpow binaryRadix d.exp := by
  have hnonneg : 0 ≤ (d.mant : ℝ) * neuralBpow binaryRadix d.exp := by
    exact mul_nonneg (Nat.cast_nonneg _) (neuralBpow.nonneg binaryRadix d.exp)
  by_cases hs : d.sign
  · -- sign = negative
    simp [dyadicToReal, hs, hnonneg, _root_.abs_of_nonneg,
      mul_comm]
  · -- sign = positive
    simp [dyadicToReal, hs, hnonneg, _root_.abs_of_nonneg,
      mul_comm]

/--
Exact multiplication of dyadics commutes with decoding to reals.

Informal: decoding the dyadic that multiplies mantissas and adds exponents gives the product of the
decoded real values.
-/
theorem dyadicToReal_mul_exact (a b : Dyadic) :
    dyadicToReal { sign := Bool.xor a.sign b.sign, mant := a.mant * b.mant, exp := a.exp + b.exp } =
      dyadicToReal a * dyadicToReal b := by
  by_cases ha : a.sign <;> by_cases hb : b.sign <;>
    simp [dyadicToReal, Bool.xor, ha, hb, neuralBpow.add_exp, mul_assoc, mul_left_comm, mul_comm]

/--
Negating a dyadic (flipping its sign bit) negates its decoded real value.
-/
theorem dyadicToReal_neg (d : Dyadic) :
    dyadicToReal { sign := (!d.sign), mant := d.mant, exp := d.exp } = -dyadicToReal d := by
  by_cases hs : d.sign <;> simp [dyadicToReal, hs]


/-- `signBit (ofBits b)` is exactly the 31st bit of `b` at the natural-number level. -/
lemma signBit_ofBits_eq_testBit31 (b : UInt32) :
    signBit (ofBits b) = Nat.testBit b.toNat 31 := by
  classical
  have hSignMask : signMask.toNat = 2 ^ 31 := by decide
  by_cases hb : Nat.testBit b.toNat 31
  · -- bit 31 is set, so `b &&& signMask` is nonzero.
    have hnat : b.toNat &&& signMask.toNat = 2 ^ 31 := by
      simpa [hSignMask, Nat.and_two_pow, hb] using (Nat.and_two_pow b.toNat 31)
    have hne : (b &&& signMask) ≠ 0 := by
      intro h0
      have h0' : (b &&& signMask).toNat = 0 := by
        simp [h0]
      have : (b.toNat &&& signMask.toNat) = 0 := by
        simpa [UInt32.toNat_and] using h0'
      have : (2 ^ 31 : Nat) = 0 := by simpa [hnat] using this.symm
      exact (Nat.ne_of_gt (Nat.pow_pos (a := 2) (n := 31) (by decide : 0 < (2 : Nat)))) this
    have hbne : (b &&& signMask != 0) = true := (bne_iff_ne).2 hne
    simp [signBit, ofBits, hb, hbne]
  · -- bit 31 is not set, so `b &&& signMask = 0`.
    have hnat : b.toNat &&& signMask.toNat = 0 := by
      simpa [hSignMask, Nat.and_two_pow, hb] using (Nat.and_two_pow b.toNat 31)
    have heq : (b &&& signMask) = 0 := by
      apply (UInt32.toNat_inj).1
      simp [UInt32.toNat_and, hnat]
    simp [signBit, ofBits, hb, heq]

/-- Flipping `signMask` toggles the sign bit and leaves everything else unchanged. -/
lemma signBit_ofBits_xor_signMask (b : UInt32) :
    signBit (ofBits (b ^^^ signMask)) = (!signBit (ofBits b)) := by
  have hSignMask : signMask.toNat = 2 ^ 31 := by decide
  have hmask : Nat.testBit signMask.toNat 31 = true := by
    simpa [hSignMask] using (Nat.testBit_two_pow_self (n := 31))
  -- Rewrite both sides via `testBit`.
  have h1 := signBit_ofBits_eq_testBit31 (b := b)
  have h2 := signBit_ofBits_eq_testBit31 (b := (b ^^^ signMask))
  -- `testBit` toggles at bit 31 because `signMask` has exactly that bit set.
  have hx : Nat.testBit (b.toNat ^^^ signMask.toNat) 31 = (!Nat.testBit b.toNat 31) := by
    simp [Nat.testBit_xor, hmask]
  -- Finish.
  simp [h1, h2, UInt32.toNat_xor, hx]

/--
If `x` decodes to the dyadic `d`, then `neg x` decodes to the same magnitude with a flipped sign.

This lemma is one of the bitfield bridge steps that lets us transport algebraic facts from the
dyadic semantics to `IEEE32Exec`.
-/
theorem toDyadic?_neg_of_toDyadic?_some (x : IEEE32Exec) {d : Dyadic}
    (hx : toDyadic? x = some d) :
    toDyadic? (neg x) = some { sign := (!d.sign), mant := d.mant, exp := d.exp } := by
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  -- `neg` preserves the exponent/fraction fields, so NaN/Inf status is unchanged.
  have hexp : expField (neg x) = expField x := by
    simpa [IEEE32Exec.neg, hxNaN, ofBits] using (expField_ofBits_xor_signMask (b := x.bits))
  have hfrac : fracField (neg x) = fracField x := by
    simpa [IEEE32Exec.neg, hxNaN, ofBits] using (fracField_ofBits_xor_signMask (b := x.bits))
  have hNoSpecial : (isNaN (neg x) || isInf (neg x)) = false := by
    have hxNaNneg : isNaN (neg x) = false := by
      have hEq : isNaN (neg x) = isNaN x := by
        unfold isNaN
        simp [hexp, hfrac]
      simpa [hEq] using hxNaN
    have hxInfneg : isInf (neg x) = false := by
      have hEq : isInf (neg x) = isInf x := by
        unfold isInf
        simp [hexp, hfrac]
      simpa [hEq] using hxInf
    simp [hxNaNneg, hxInfneg]
  -- Decode `neg x` by reusing the decode of `x`.
  unfold toDyadic? at hx ⊢
  have hnaninf : (isNaN x || isInf x) = false := by simp [hxNaN, hxInf]
  have hs : signBit (neg x) = (!signBit x) := by
    simpa [IEEE32Exec.neg, hxNaN, ofBits] using (signBit_ofBits_xor_signMask (b := x.bits))
  simp (config := { zeta := true }) [hnaninf, hNoSpecial, hs, hexp, hfrac] at hx ⊢
  by_cases hE : x.expField = 0
  · by_cases hF : x.fracField = 0
    · have hx' :
          some { sign := x.signBit, mant := 0, exp := 0 } = some d := by
          simpa [hE, hF] using hx
      have hd : d = { sign := x.signBit, mant := 0, exp := 0 } := (Option.some.inj hx').symm
      simp [hE, hF, hd]
    · have hx' :
          some { sign := x.signBit, mant := x.fracField.toNat, exp := -149 } = some d := by
          simpa [hE, hF] using hx
      have hd : d = { sign := x.signBit, mant := x.fracField.toNat, exp := -149 } :=
        (Option.some.inj hx').symm
      simp [hE, hF, hd]
  · have hx' :
        some { sign := x.signBit, mant := pow2 23 + x.fracField.toNat, exp := ↑x.expField.toNat -
          150 } = some d := by
        simpa [hE] using hx
    have hd :
        d = { sign := x.signBit, mant := pow2 23 + x.fracField.toNat, exp := ↑x.expField.toNat - 150
          } :=
      (Option.some.inj hx').symm
    simp [hE, hd]

/--
On finite values, `toReal` respects `IEEE32Exec.neg`.

We phrase this as an equality on `toReal` for convenience, but the proof uses the
finiteness witness `toDyadic? x = some d`.
-/
theorem toReal_neg_eq_neg (x : IEEE32Exec) {d : Dyadic}
    (hx : toDyadic? x = some d) : toReal (neg x) = -toReal x := by
  have hneg : toDyadic? (neg x) = some { sign := (!d.sign), mant := d.mant, exp := d.exp } :=
    toDyadic?_neg_of_toDyadic?_some (x := x) (d := d) hx
  simp [toReal_eq, hx, hneg, dyadicToReal_neg]

/--
The executable maximum-finite binary32 bit pattern denotes `FP32.ieeeMaxFinite`.

This identifies the concrete endpoint `0x7f7fffff` with the rounded-real guard
`(2 - 2⁻²³) * 2¹²⁷`.  Keeping this fact in the core bridge gives overflow and interval proofs a
single canonical statement about the upper edge of finite binary32.
-/
theorem toReal_posMaxFinite_eq_ieeeMaxFinite :
    toReal (posMaxFinite : IEEE32Exec) = FP32.ieeeMaxFinite := by
  have hexp : (254 : Nat) < 255 := by decide
  have hfrac : pow2 23 - 1 < 2 ^ 23 := by
    norm_num [pow2_eq_two_pow]
  have hbits : mkBits false 254 (pow2 23 - 1) = 0x7f7fffff := by decide
  have hdecode :
      toDyadic? (posMaxFinite : IEEE32Exec) =
        some {
          sign := false
          mant := pow2 23 + (pow2 23 - 1)
          exp := Int.ofNat 254 - 150 } := by
    simpa [posMaxFinite, hbits] using
      (toDyadic?_ofBits_mkBits_fin
        (sign := false) (exp := 254) (frac := pow2 23 - 1) hexp hfrac)
  rw [toReal_eq, hdecode]
  norm_num [dyadicToReal, FP32.ieeeMaxFinite, NeuralPrecision.machineEpsilon,
    NeuralPrecision.mantissaBits, NeuralPrecision.expBits, neuralBpow, binaryRadix,
    NeuralRadix.toReal, pow2_eq_two_pow]

/--
Every finite executable binary32 value lies between the two maximum-finite endpoints.

The proof reads the stored exponent and fraction fields directly.  It covers normal numbers,
subnormals, and both signed zeros; NaNs and infinities are excluded by `hfin`.
-/
theorem abs_toReal_le_ieeeMaxFinite_of_isFinite (x : IEEE32Exec)
    (hfin : isFinite x = true) :
    |toReal x| ≤ FP32.ieeeMaxFinite := by
  have hexp : expField x ≠ expAllOnes := (bne_iff_ne).mp hfin
  have hnan : isNaN x = false := by simp [isNaN, hexp]
  have hinf : isInf x = false := by simp [isInf, hexp]
  by_cases he : expField x = 0
  · by_cases hf : fracField x = 0
    · have hdecode :
          toDyadic? x = some { sign := signBit x, mant := 0, exp := 0 } := by
        simp [toDyadic?, hnan, hinf, he, hf]
      rw [toReal_eq, hdecode]
      simp [dyadicToReal, FP32.ieeeMaxFinite_eq, neuralBpow.nonneg]
    · have hdecode :
          toDyadic? x =
            some { sign := signBit x, mant := (fracField x).toNat, exp := -149 } := by
        simp [toDyadic?, hnan, hinf, he, hf]
      have hfracNat : (fracField x).toNat ≤ 2 ^ 24 - 1 := by
        have hfrac := fracField_toNat_lt_pow2_23 x
        grind
      have hfracReal :
          ((fracField x).toNat : ℝ) ≤ ((2 ^ 24 - 1 : Nat) : ℝ) := by
        exact_mod_cast hfracNat
      have hbpow :
          neuralBpow binaryRadix (-149) ≤ neuralBpow binaryRadix 104 :=
        (neuralBpow_le_neuralBpow_iff binaryRadix _ _).2 (by norm_num)
      rw [toReal_eq, hdecode, abs_dyadicToReal, FP32.ieeeMaxFinite_eq]
      exact mul_le_mul hfracReal hbpow (neuralBpow.nonneg binaryRadix _)
        (Nat.cast_nonneg _)
  · have hdecode :
        toDyadic? x =
          some {
            sign := signBit x
            mant := pow2 23 + (fracField x).toNat
            exp := Int.ofNat (expField x).toNat - 150 } := by
      simp [toDyadic?, hnan, hinf, he]
    have hmantNat : pow2 23 + (fracField x).toNat ≤ 2 ^ 24 - 1 := by
      have hfrac := fracField_toNat_lt_pow2_23 x
      rw [pow2_eq_two_pow]
      grind
    have hmantReal :
        ((pow2 23 + (fracField x).toNat : Nat) : ℝ) ≤
          ((2 ^ 24 - 1 : Nat) : ℝ) := by
      exact_mod_cast hmantNat
    have hexpNat := expField_toNat_le_254_of_isFinite x hfin
    have hexpInt : Int.ofNat (expField x).toNat ≤ 254 :=
      Int.ofNat_le.mpr hexpNat
    have hexpBound : Int.ofNat (expField x).toNat - 150 ≤ 104 := by
      grind
    have hbpow :
        neuralBpow binaryRadix (Int.ofNat (expField x).toNat - 150) ≤
          neuralBpow binaryRadix 104 :=
      (neuralBpow_le_neuralBpow_iff binaryRadix _ _).2 hexpBound
    rw [toReal_eq, hdecode, abs_dyadicToReal, FP32.ieeeMaxFinite_eq]
    exact mul_le_mul hmantReal hbpow (neuralBpow.nonneg binaryRadix _)
      (Nat.cast_nonneg _)

/--
The real denotation of every finite executable binary32 value belongs to the rounded-real
binary32 format.

This is the representability bridge from an IEEE bit pattern to `fexp32`. It is what permits
format-level results, such as Sterbenz's lemma, to be applied directly to executable operands.
-/
theorem toReal_neuralGenericFormat_of_isFinite (x : IEEE32Exec)
    (hfin : isFinite x = true) :
    neuralGenericFormat binaryRadix fexp32 (toReal x) := by
  have hexp : expField x ≠ expAllOnes := (bne_iff_ne).mp hfin
  have hnan : isNaN x = false := by simp [isNaN, hexp]
  have hinf : isInf x = false := by simp [isInf, hexp]
  apply (generic_format_FLT_iff (-149) 24 (by norm_num) (toReal x)).2
  by_cases he : expField x = 0
  · by_cases hf : fracField x = 0
    · have hdecode :
          toDyadic? x = some { sign := signBit x, mant := 0, exp := 0 } := by
        simp [toDyadic?, hnan, hinf, he, hf]
      refine ⟨by norm_num, { mantissa := 0, exponent := 0 }, ?_, by norm_num [binaryRadix], by norm_num⟩
      rw [toReal_eq, hdecode]
      simp [dyadicToReal, neuralToReal]
    · let m : Int :=
        if signBit x then -(Int.ofNat (fracField x).toNat)
        else Int.ofNat (fracField x).toNat
      have hdecode :
          toDyadic? x =
            some { sign := signBit x, mant := (fracField x).toNat, exp := -149 } := by
        simp [toDyadic?, hnan, hinf, he, hf]
      refine ⟨by norm_num, { mantissa := m, exponent := -149 }, ?_, ?_, le_rfl⟩
      · rw [toReal_eq, hdecode]
        by_cases hs : signBit x <;> simp [dyadicToReal, neuralToReal, m, hs]
      · have hfrac := fracField_toNat_lt_pow2_23 x
        have hfrac24 : (fracField x).toNat < 2 ^ 24 :=
          lt_trans hfrac (Nat.pow_lt_pow_right (by decide) (by decide))
        by_cases hs : signBit x <;> simpa [m, hs, binaryRadix] using hfrac24
  · let m : Int :=
      if signBit x then -(Int.ofNat (pow2 23 + (fracField x).toNat))
      else Int.ofNat (pow2 23 + (fracField x).toNat)
    have hdecode :
        toDyadic? x =
          some {
            sign := signBit x
            mant := pow2 23 + (fracField x).toNat
            exp := Int.ofNat (expField x).toNat - 150 } := by
      simp [toDyadic?, hnan, hinf, he]
    refine ⟨by norm_num, {
      mantissa := m
      exponent := Int.ofNat (expField x).toNat - 150 }, ?_, ?_, ?_⟩
    · rw [toReal_eq, hdecode]
      by_cases hs : signBit x <;> simp [dyadicToReal, neuralToReal, m, hs]
    · have hfrac := fracField_toNat_lt_pow2_23 x
      have hmant : pow2 23 + (fracField x).toNat < 2 ^ 24 := by
        calc
          pow2 23 + (fracField x).toNat < pow2 23 + 2 ^ 23 :=
            Nat.add_lt_add_left hfrac _
          _ = 2 ^ 24 := by norm_num [pow2_eq_two_pow]
      have hmabs : m.natAbs = pow2 23 + (fracField x).toNat := by
        cases hs : signBit x with
        | false =>
            have hm : m = Int.ofNat (pow2 23 + (fracField x).toNat) := by simp [m, hs]
            rw [hm]
            exact Int.natAbs_ofNat' _
        | true =>
            have hm : m = -Int.ofNat (pow2 23 + (fracField x).toNat) := by simp [m, hs]
            rw [hm]
            rw [Int.natAbs_neg]
            exact Int.natAbs_ofNat' _
      simpa [hmabs, binaryRadix] using hmant
    · have hePos : 0 < (expField x).toNat := by
        exact Nat.pos_of_ne_zero (fun h0 => he (UInt32.toNat_inj.mp (by simpa using h0)))
      have : (1 : Int) ≤ Int.ofNat (expField x).toNat := by
        exact Int.ofNat_le.mpr (Nat.succ_le_of_lt hePos)
      linarith
end IEEE32Exec

end TorchLean.Floats.IEEE754
