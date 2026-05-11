/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Analysis.SpecialFunctions.Log.Base
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.Data.Nat.Bitwise
public import Mathlib.Data.Nat.Sqrt
public import Mathlib.Data.Rat.Floor
public import NN.Floats.FP32
public import NN.Floats.IEEEExec.Exec32
public import NN.Floats.IEEEExec.MkBitsToDyadic
public import NN.Floats.IEEEExec.RealSemantics
public import NN.Floats.IEEEExec.Negation
public import NN.Floats.IEEEExec.NatLemmas
public import NN.Floats.IEEEExec.RoundShiftRightEven

/-!
# BridgeFP32

Bridge theorems: `IEEE32Exec` (a bit-level, executable IEEE-754 binary32 kernel) ↔ `FP32`
(a Flocq-style “round-to-`ℝ`” model for *finite* float32 computations).

TorchLean keeps two views of float32:

- **Executable view** (`IEEE32Exec`): we can run models inside Lean by implementing IEEE-754-style
  operations on the raw 32-bit encoding.
- **Mathematical view** (`FP32`): we can reuse a large amount of existing floating-point theory
  (rounding, ulps, error bounds) phrased as real arithmetic + a rounding operator.

This file bridges those two views. Informally, the main statements have the shape:

```
toReal (op_exec x y) = fp32Round (op_real (toReal x) (toReal y))
```

under the side-condition that we stay on the **finite / no-overflow** path. That is exactly the
“refinement” we need to justify using `FP32`/`NF` reasoning when our runtime execution is driven by
`IEEE32Exec`.

What we *do not* claim here:
- `FP32` models only finite values, so NaNs/Infs are not bridged (proofs use `toReal?` and carry
  “isFinite = true” hypotheses).
- Some transcendental functions in `IEEE32Exec` exist for executability but are not part of the
  verified IEEE kernel; this file therefore focuses on the core arithmetic and the reductions we
  rely on in practice.

Pointers / background:
- IEEE 754-2019 (rounding modes, signed zeros, specials):
  https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg (1991), a classic practical guide to IEEE-754 reasoning:
  https://doi.org/10.1145/103162.103163
- Higham (2002), error analysis in finite precision (book): ISBN 0-89871-521-0
- Flocq (Boldo–Melquiond, 2011), a widely-used formal floating-point library design:
  https://doi.org/10.1109/ARITH.2011.40
- Boldo et al. (2012), “Floating-point arithmetic in the Coq system”:
  https://doi.org/10.1016/j.ic.2011.09.005
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/-!
## Reals view of `IEEE32Exec`

Real semantics (`toReal?`/`toReal`) live in `NN.Floats.IEEEExec.RealSemantics`. This bridge file
uses them pervasively, but does not define them.
-/

/-- `FP32` rounding viewed as a real function. -/
noncomputable abbrev fp32Round (x : ℝ) : ℝ :=
  neuralRound (β := binaryRadix) (fexp := TorchLean.Floats.fexp32) TorchLean.Floats.rnd32 x

/-! ### Basic sanity checks for `fp32Round` -/

/-- Rounding `0` to float32 yields `0`. -/
theorem fp32Round_zero : fp32Round 0 = 0 := by
  -- This proof proceeds by unfolding: `fp32Round` is defined via `neural_round`.
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
- and the `FP32` rounding model (which is expressed using `neural_magnitude` / nearest-even).

These lemmas are local: they exist to keep the later op-level theorems readable.
-/

private noncomputable def signFactor (s : Bool) : ℝ :=
  if s then (-1 : ℝ) else (1 : ℝ)

private lemma signFactor_xor (a b : Bool) :
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


/-- `signBit (ofBits b)` is literally the 31st bit of `b` (at the nat level). -/
private lemma signBit_ofBits_eq_testBit31 (b : UInt32) :
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
private lemma signBit_ofBits_xor_signMask (b : UInt32) :
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

We phrase this as an equality on `toReal` for convenience, but the proof fundamentally uses the
finiteness witness `toDyadic? x = some d`.
-/
theorem toReal_neg_eq_neg (x : IEEE32Exec) {d : Dyadic}
    (hx : toDyadic? x = some d) : toReal (neg x) = -toReal x := by
  have hneg : toDyadic? (neg x) = some { sign := (!d.sign), mant := d.mant, exp := d.exp } :=
    toDyadic?_neg_of_toDyadic?_some (x := x) (d := d) hx
  simp [toReal_eq, hx, hneg, dyadicToReal_neg]

/-!
## Exact dyadic addition (used by op-level refinement theorems)

IEEE-754 addition is “exact add, then round”. To connect an executable `add` to the `FP32` model, we
factor the refinement into two steps:

1. perform an *exact* addition in an unbounded format (here: dyadics with integer exponents), and
2. apply float32 rounding.

`addDyadic` is our exact step (it aligns exponents, adds signed mantissas, and normalizes), and the
lemmas in this section show that its real interpretation is literally real addition.
-/

private noncomputable def signedMant (sign : Bool) (m : Nat) : Int :=
  if sign then -(Int.ofNat m) else Int.ofNat m

private lemma dyadicToReal_eq_signedMant (d : Dyadic) :
    dyadicToReal d = (signedMant d.sign d.mant : ℝ) * neuralBpow binaryRadix d.exp := by
  by_cases hs : d.sign <;> simp [dyadicToReal, signedMant, hs]

private lemma signedMant_shiftLeft (sign : Bool) (m sh : Nat) :
    ((signedMant sign (Nat.shiftLeft m sh) : Int) : ℝ) =
      ((signedMant sign m : Int) : ℝ) * neuralBpow binaryRadix (Int.ofNat sh) := by
  by_cases hs : sign
  · simp [signedMant, hs, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
    Nat.shiftLeft_eq,
      Nat.cast_mul, Nat.cast_pow]
  · simp [signedMant, hs, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
    Nat.shiftLeft_eq,
      Nat.cast_mul, Nat.cast_pow]

private lemma signFactor_natAbs_int (s : Int) :
    (if decide (s < 0) then (-1 : ℝ) else (1 : ℝ)) * (Int.natAbs s : ℝ) = (s : ℝ) := by
  cases s with
  | ofNat n =>
      simp
  | negSucc n =>
      simp

private lemma dyadicToReal_ofNatAbs (s : Int) (e : Int) :
    dyadicToReal { sign := decide (s < 0), mant := Int.natAbs s, exp := e } =
      (s : ℝ) * neuralBpow binaryRadix e := by
  dsimp [dyadicToReal]
  rw [signFactor_natAbs_int (s := s)]

private lemma dyadicToReal_zero (sign : Bool) :
    dyadicToReal { sign := sign, mant := 0, exp := 0 } = (0 : ℝ) := by
  by_cases hs : sign <;> simp [dyadicToReal, hs, TorchLean.Floats.neuralBpow, binaryRadix,
    NeuralRadix.toReal]

/--
`addDyadic` is exact with respect to `dyadicToReal`.

Informal: `addDyadic` aligns exponents, adds signed mantissas, and normalizes; decoding the result
gives the sum of the decoded inputs.
-/
theorem dyadicToReal_addDyadic_exact (a b : Dyadic) :
    dyadicToReal (addDyadic a b) = dyadicToReal a + dyadicToReal b := by
  classical
  by_cases hab : a.exp ≤ b.exp
  · -- align to `a.exp`
    let sh : Nat := Int.toNat (b.exp - a.exp)
    have hdiff_nonneg : 0 ≤ b.exp - a.exp := sub_nonneg.mpr hab
    have hdiff : (b.exp - a.exp) = (sh : Int) := by
      have := (Int.toNat_of_nonneg (a := b.exp - a.exp) hdiff_nonneg)
      simpa [sh] using this.symm
    have hbexp : b.exp = a.exp + (sh : Int) := by
      have hb : a.exp + (b.exp - a.exp) = b.exp := by
        simp [sub_eq_add_neg]
      have hb' : a.exp + (sh : Int) = b.exp := by
        simpa [hdiff] using hb
      exact hb'.symm

    let m1 : Int := signedMant a.sign a.mant
    let m2 : Int := signedMant b.sign (Nat.shiftLeft b.mant sh)
    let s : Int := m1 + m2

    have hadd :
        addDyadic a b =
          if s == 0 then { sign := a.sign && b.sign, mant := 0, exp := 0 }
          else { sign := decide (s < 0), mant := Int.natAbs s, exp := a.exp } := by
      simp (config := { zeta := true }) [addDyadic, hab, sh, m1, m2, s, signedMant]

    have ha : dyadicToReal a = (m1 : ℝ) * neuralBpow binaryRadix a.exp := by
      simp [m1, dyadicToReal_eq_signedMant, signedMant]

    have hb : dyadicToReal b = (m2 : ℝ) * neuralBpow binaryRadix a.exp := by
      have hb0 := dyadicToReal_eq_signedMant (d := b)
      rw [hb0, hbexp]
      rw [neuralBpow.add_exp binaryRadix a.exp (sh : Int)]
      calc
        (signedMant b.sign b.mant : ℝ) *
            (neuralBpow binaryRadix a.exp * neuralBpow binaryRadix (sh : Int)) =
            ((signedMant b.sign b.mant : ℝ) * neuralBpow binaryRadix (sh : Int)) *
              neuralBpow binaryRadix a.exp := by
              ring
        _ = (m2 : ℝ) * neuralBpow binaryRadix a.exp := by
            have hm2 : (m2 : ℝ) = (signedMant b.sign b.mant : ℝ) * neuralBpow binaryRadix (sh :
              Int) := by
              simpa [m2] using (signedMant_shiftLeft (sign := b.sign) (m := b.mant) (sh := sh))
            simp [hm2]

    have hsum : dyadicToReal a + dyadicToReal b = (s : ℝ) * neuralBpow binaryRadix a.exp := by
      rw [ha, hb]
      have hfactor :
          (m1 : ℝ) * neuralBpow binaryRadix a.exp + (m2 : ℝ) * neuralBpow binaryRadix a.exp =
            ((m1 : ℝ) + (m2 : ℝ)) * neuralBpow binaryRadix a.exp := by
        simpa using (add_mul (m1 : ℝ) (m2 : ℝ) (neuralBpow binaryRadix a.exp)).symm
      have hcast : ((m1 : ℝ) + (m2 : ℝ)) = (s : ℝ) := by
        simp [s, Int.cast_add]
      calc
        (m1 : ℝ) * neuralBpow binaryRadix a.exp + (m2 : ℝ) * neuralBpow binaryRadix a.exp =
            ((m1 : ℝ) + (m2 : ℝ)) * neuralBpow binaryRadix a.exp := hfactor
        _ = (s : ℝ) * neuralBpow binaryRadix a.exp := by simp [hcast]

    by_cases hs0 : s = 0
    · rw [hadd]
      simp [hs0, dyadicToReal_zero, hsum]
    · have hs0b : (s == 0) = false := (beq_eq_false_iff_ne).2 hs0
      have hres :
          dyadicToReal { sign := decide (s < 0), mant := Int.natAbs s, exp := a.exp } =
            (s : ℝ) * neuralBpow binaryRadix a.exp := by
        simpa using (dyadicToReal_ofNatAbs (s := s) (e := a.exp))
      rw [hadd]
      simp [hs0b, hres, hsum]

  · -- align to `b.exp`
    have hba : b.exp ≤ a.exp := le_of_not_ge hab
    let sh : Nat := Int.toNat (a.exp - b.exp)
    have hdiff_nonneg : 0 ≤ a.exp - b.exp := sub_nonneg.mpr hba
    have hdiff : (a.exp - b.exp) = (sh : Int) := by
      have := (Int.toNat_of_nonneg (a := a.exp - b.exp) hdiff_nonneg)
      simpa [sh] using this.symm
    have haexp : a.exp = b.exp + (sh : Int) := by
      have hb : b.exp + (a.exp - b.exp) = a.exp := by
        simp [sub_eq_add_neg]
      have hb' : b.exp + (sh : Int) = a.exp := by
        simpa [hdiff] using hb
      exact hb'.symm

    let m1 : Int := signedMant a.sign (Nat.shiftLeft a.mant sh)
    let m2 : Int := signedMant b.sign b.mant
    let s : Int := m1 + m2

    have hadd :
        addDyadic a b =
          if s == 0 then { sign := a.sign && b.sign, mant := 0, exp := 0 }
          else { sign := decide (s < 0), mant := Int.natAbs s, exp := b.exp } := by
      simp (config := { zeta := true }) [addDyadic, hab, sh, m1, m2, s, signedMant]

    have hb' : dyadicToReal b = (m2 : ℝ) * neuralBpow binaryRadix b.exp := by
      simp [m2, dyadicToReal_eq_signedMant, signedMant]

    have ha' : dyadicToReal a = (m1 : ℝ) * neuralBpow binaryRadix b.exp := by
      have ha0 := dyadicToReal_eq_signedMant (d := a)
      rw [ha0, haexp]
      rw [neuralBpow.add_exp binaryRadix b.exp (sh : Int)]
      calc
        (signedMant a.sign a.mant : ℝ) *
            (neuralBpow binaryRadix b.exp * neuralBpow binaryRadix (sh : Int)) =
            ((signedMant a.sign a.mant : ℝ) * neuralBpow binaryRadix (sh : Int)) *
              neuralBpow binaryRadix b.exp := by
              ring
        _ = (m1 : ℝ) * neuralBpow binaryRadix b.exp := by
            have hm1 : (m1 : ℝ) = (signedMant a.sign a.mant : ℝ) * neuralBpow binaryRadix (sh :
              Int) := by
              simpa [m1] using (signedMant_shiftLeft (sign := a.sign) (m := a.mant) (sh := sh))
            simp [hm1]

    have hsum : dyadicToReal a + dyadicToReal b = (s : ℝ) * neuralBpow binaryRadix b.exp := by
      rw [ha', hb']
      have hfactor :
          (m1 : ℝ) * neuralBpow binaryRadix b.exp + (m2 : ℝ) * neuralBpow binaryRadix b.exp =
            ((m1 : ℝ) + (m2 : ℝ)) * neuralBpow binaryRadix b.exp := by
        simpa using (add_mul (m1 : ℝ) (m2 : ℝ) (neuralBpow binaryRadix b.exp)).symm
      have hcast : ((m1 : ℝ) + (m2 : ℝ)) = (s : ℝ) := by
        simp [s, Int.cast_add]
      calc
        (m1 : ℝ) * neuralBpow binaryRadix b.exp + (m2 : ℝ) * neuralBpow binaryRadix b.exp =
            ((m1 : ℝ) + (m2 : ℝ)) * neuralBpow binaryRadix b.exp := hfactor
        _ = (s : ℝ) * neuralBpow binaryRadix b.exp := by simp [hcast]

    by_cases hs0 : s = 0
    · rw [hadd]
      simp [hs0, dyadicToReal_zero, hsum]
    · have hs0b : (s == 0) = false := (beq_eq_false_iff_ne).2 hs0
      have hres :
          dyadicToReal { sign := decide (s < 0), mant := Int.natAbs s, exp := b.exp } =
            (s : ℝ) * neuralBpow binaryRadix b.exp := by
        simpa using (dyadicToReal_ofNatAbs (s := s) (e := b.exp))
      rw [hadd]
      simp [hs0b, hres, hsum]

/--
For a nonzero dyadic, `neural_magnitude` matches the expected “power-of-two interval”
characterization: `mag = ⌊logb 2 (mant)⌋ + exp + 1`.

We use this as a key link between the `FP32` rounding model (defined using `neural_magnitude`) and
the executable kernel (which naturally computes `Nat.log2 mant + exp` from the decoded dyadic).
-/
theorem neural_magnitude_dyadic (d : Dyadic) (hm : d.mant ≠ 0) :
    neuralMagnitude binaryRadix (dyadicToReal d) =
      (Int.ofNat (Nat.log 2 d.mant)) + d.exp + 1 := by
  have hx : dyadicToReal d ≠ 0 := by
    have hs : (if d.sign then (-1 : ℝ) else (1 : ℝ)) ≠ 0 := by
      by_cases h : d.sign <;> simp [h]
    have hm' : (d.mant : ℝ) ≠ 0 := by
      exact_mod_cast hm
    have hb : neuralBpow binaryRadix d.exp ≠ 0 := neuralBpow.ne_zero binaryRadix d.exp
    -- `s * mant * 2^exp ≠ 0`.
    have : (if d.sign then (-1 : ℝ) else (1 : ℝ)) * (d.mant : ℝ) * neuralBpow binaryRadix d.exp ≠
      0 := by
      exact mul_ne_zero (mul_ne_zero hs hm') hb
    simpa [dyadicToReal, mul_assoc] using this

  -- Expand `neural_magnitude` and rewrite the log ratio as `Real.logb`.
  simp [TorchLean.Floats.neuralMagnitude, hx, Real.log_div_log, abs_dyadicToReal d]

  -- Use `logb_mul` to split `logb 2 (mant * 2^exp)` as `logb 2 mant + logb 2 (2^exp)`.
  have hmpos : (0 : ℝ) < (d.mant : ℝ) := by
    have : 0 < d.mant := Nat.pos_of_ne_zero hm
    exact_mod_cast this
  have hbpos : (0 : ℝ) < neuralBpow binaryRadix d.exp := neuralBpow.pos binaryRadix d.exp

  have hlogb_mul :
      Real.logb (binaryRadix.toReal) ((d.mant : ℝ) * neuralBpow binaryRadix d.exp) =
        Real.logb (binaryRadix.toReal) (d.mant : ℝ) +
        Real.logb (binaryRadix.toReal) (neuralBpow binaryRadix d.exp) := by
    -- `logb_mul` needs nonzero arguments.
    have hm0 : (d.mant : ℝ) ≠ 0 := (ne_of_gt hmpos)
    have hb0 : neuralBpow binaryRadix d.exp ≠ 0 := (ne_of_gt hbpos)
    simpa [binaryRadix, NeuralRadix.toReal] using
      (Real.logb_mul (b := (binaryRadix.toReal)) (x := (d.mant : ℝ)) (y := neuralBpow
        binaryRadix d.exp) hm0 hb0)

  -- `logb 2 (2^e) = e`.
  have hlogb_bpow : Real.logb (binaryRadix.toReal) (neuralBpow binaryRadix d.exp) = (d.exp : ℝ)
    := by
    -- `logb 2 (2^e) = e` using `Real.log_zpow`.
    have hlog2 : Real.log (2 : ℝ) ≠ 0 := by
      have h2 : (2 : ℝ) ≠ 0 := by norm_num
      have h21 : (2 : ℝ) ≠ 1 := by norm_num
      have h2m1 : (2 : ℝ) ≠ -1 := by norm_num
      exact Real.log_ne_zero.mpr ⟨h2, h21, h2m1⟩
    -- Unfold `neural_bpow` and reduce to division by `log 2`.
    simp [Real.logb, neuralBpow, binaryRadix, NeuralRadix.toReal, Real.log_zpow, hlog2]

  -- Now take floors: `floor (a + z) = floor a + z`.
  -- First rewrite the base to `2` (a Nat) so we can use `Real.floor_logb_natCast`.
  have hb2 : (binaryRadix.toReal) = (2 : ℝ) := by rfl
  -- Reduce to `⌊logb 2 mant⌋`.
  have hfloor_mant :
      ⌊Real.logb (binaryRadix.toReal) (d.mant : ℝ)⌋ = Int.ofNat (Nat.log 2 d.mant) := by
    have hr : (0 : ℝ) ≤ (d.mant : ℝ) := Nat.cast_nonneg _
    -- `⌊logb 2 n⌋ = Int.log 2 n = Nat.log 2 n`
    have h :=
      (Real.floor_logb_natCast (b := 2) (r := (d.mant : ℝ)) hr)
    -- Rewrite base `binary_radix.to_real` to `2`, then simplify `Int.log` on a nat cast.
    simpa [hb2, Int.log_natCast] using h

  -- Put it together.
  -- `floor (logb 2 (mant*2^exp)) = floor (logb 2 mant + exp) = floor(logb2 mant) + exp`.
  have hfloor_total :
      ⌊Real.logb (binaryRadix.toReal) ((d.mant : ℝ) * neuralBpow binaryRadix d.exp)⌋ =
        Int.ofNat (Nat.log 2 d.mant) + d.exp := by
    -- rewrite using `hlogb_mul` and `hlogb_bpow`
    rw [hlogb_mul, hlogb_bpow]
    -- `floor (a + z) = floor a + z`
    simp [hfloor_mant]

  -- `simp` reduced the goal to a statement about the floored `logb`; discharge it.
  exact hfloor_total

private lemma neural_magnitude_eq_of_bpow_bounds (x : ℝ) (k : Int)
    (hx0 : x ≠ 0)
    (hlo : neuralBpow binaryRadix k ≤ _root_.abs x)
    (hhi : _root_.abs x < neuralBpow binaryRadix (k + 1)) :
    neuralMagnitude binaryRadix x = k + 1 := by
  have hxabs : 0 < _root_.abs x := abs_pos.mpr hx0
  have hb : (1 : ℝ) < (2 : ℝ) := by norm_num

  have hlo_z : (2 : ℝ) ^ (k : Int) ≤ _root_.abs x := by
    simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using hlo
  have hlo_r : (2 : ℝ) ^ (k : ℝ) ≤ _root_.abs x := by
    calc
      (2 : ℝ) ^ (k : ℝ) = (2 : ℝ) ^ (k : Int) := by
        simp
      _ ≤ _root_.abs x := hlo_z

  have hhi_z : _root_.abs x < (2 : ℝ) ^ (k + 1 : Int) := by
    simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using hhi
  have hhi_r : _root_.abs x < (2 : ℝ) ^ ((k + 1 : Int) : ℝ) := by
    calc
      _root_.abs x < (2 : ℝ) ^ (k + 1 : Int) := hhi_z
      _ = (2 : ℝ) ^ ((k + 1 : Int) : ℝ) := by
        simpa using (Real.rpow_intCast (2 : ℝ) (k + 1)).symm

  have hlo' : (k : ℝ) ≤ Real.logb (2 : ℝ) (_root_.abs x) :=
    (Real.le_logb_iff_rpow_le (b := (2 : ℝ)) (x := (k : ℝ)) (y := _root_.abs x) hb hxabs).2 hlo_r
  have hhi' : Real.logb (2 : ℝ) (_root_.abs x) < ((k + 1 : Int) : ℝ) :=
    (Real.logb_lt_iff_lt_rpow (b := (2 : ℝ)) (x := _root_.abs x) (y := ((k + 1 : Int) : ℝ)) hb
      hxabs).2 hhi_r

  have hfloor : (⌊Real.logb (2 : ℝ) (_root_.abs x)⌋ : Int) = k := by
    have hhi'' : Real.logb (2 : ℝ) (_root_.abs x) < (k : ℝ) + 1 := by
      simpa [Int.cast_add, Int.cast_one] using hhi'
    exact (Int.floor_eq_iff).2 ⟨hlo', hhi''⟩

  unfold TorchLean.Floats.neuralMagnitude
  rw [if_neg hx0]
  have hfloor_ratio_abs :
      (⌊Real.log (_root_.abs x) / Real.log (binaryRadix.toReal)⌋ : Int) = k := by
    simpa [Real.logb, binaryRadix, NeuralRadix.toReal] using hfloor
  have hfloor_ratio :
      (⌊Real.log x / Real.log (binaryRadix.toReal)⌋ : Int) = k := by
    simpa [Real.log_abs] using hfloor_ratio_abs
  simp [hfloor_ratio]

/-
## Nearest-even on rationals (core bridge lemma)

To relate the executable kernel’s integer rounding (`roundQuotEven` / `roundShiftRightEven`) to the
proof-relevant model’s rounding (`neural_nearest_even`), we need a lemma that computes nearest-even
rounding of a nonnegative rational `num/den` using Euclidean division.
-/

private lemma fract_real_div_nat (n den : Nat) (hden : den ≠ 0) :
    (n : ℝ) / (den : ℝ) - ((n / den : Nat) : ℝ) = ((n % den : Nat) : ℝ) / (den : ℝ) := by
  have hdiv : den * (n / den) + n % den = n := Nat.div_add_mod n den
  have hdivR : ((den * (n / den) : Nat) : ℝ) + ((n % den : Nat) : ℝ) = (n : ℝ) := by
    exact_mod_cast hdiv
  have hdenR : (den : ℝ) ≠ 0 := by exact_mod_cast hden
  have hsplit :
      (n : ℝ) / (den : ℝ) =
        ((n / den : Nat) : ℝ) + ((n % den : Nat) : ℝ) / (den : ℝ) := by
    have := congrArg (fun t : ℝ => t / (den : ℝ)) hdivR
    -- `simp` knows `((den*q)/den) = q` given `hdenR`.
    simp [add_div, hdenR] at this
    simpa using this.symm
  calc
    (n : ℝ) / (den : ℝ) - ((n / den : Nat) : ℝ)
        = (((n / den : Nat) : ℝ) + ((n % den : Nat) : ℝ) / (den : ℝ)) - ((n / den : Nat) : ℝ) := by
            simp [hsplit]
    _ = ((n % den : Nat) : ℝ) / (den : ℝ) := by ring

private lemma floor_real_nat_div (n den : Nat) :
    (⌊(n : ℝ) / (den : ℝ)⌋ : Int) = (n / den : Nat) := by
  -- Reduce to `Rat` where the floor/div lemma exists, then cast back to `ℝ`.
  calc
    (⌊(n : ℝ) / (den : ℝ)⌋ : Int)
        = (⌊(((n : ℚ) / (den : ℚ)) : ℝ)⌋ : Int) := by
            simp
    _ = (⌊((n : ℚ) / (den : ℚ))⌋ : Int) := by
            simpa using (Rat.floor_cast (α := ℝ) ((n : ℚ) / (den : ℚ)))
    _ = (n / den : Nat) := by
            simpa using (Rat.floor_natCast_div_natCast n den)

private lemma div_lt_half_iff (r den : Nat) (hden : den ≠ 0) :
    ((r : ℝ) / (den : ℝ) < (2⁻¹ : ℝ)) ↔ (2 * r < den) := by
  have hdenpos : (0 : ℝ) < (den : ℝ) := by
    exact_mod_cast (Nat.pos_of_ne_zero hden)
  have h2pos : (0 : ℝ) < (2 : ℝ) := by norm_num
  have h1 :
      ((r : ℝ) / (den : ℝ) < (2⁻¹ : ℝ)) ↔ (r : ℝ) < (2⁻¹ : ℝ) * (den : ℝ) := by
    simpa using (div_lt_iff₀ hdenpos)
  have hscale : (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) = (den : ℝ) := by
    calc
      (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ))
          = ((2 : ℝ) * (2⁻¹ : ℝ)) * (den : ℝ) := by simp []
      _ = (1 : ℝ) * (den : ℝ) := by
          have h : (2 : ℝ) * (2⁻¹ : ℝ) = (1 : ℝ) := by
            simp
          simp []
      _ = (den : ℝ) := by simp
  constructor
  · intro h
    have hr : (r : ℝ) < (2⁻¹ : ℝ) * (den : ℝ) := (h1.mp h)
    have hmul : (2 : ℝ) * (r : ℝ) < (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) :=
      mul_lt_mul_of_pos_left hr h2pos
    have hmul' : (2 : ℝ) * (r : ℝ) < (den : ℝ) := by simpa [hscale] using hmul
    have : ((2 * r : Nat) : ℝ) < (den : ℝ) := by
      simpa [Nat.cast_mul, Nat.cast_ofNat, mul_assoc, mul_left_comm, mul_comm] using hmul'
    exact (by exact_mod_cast this)
  · intro h
    have hR : ((2 * r : Nat) : ℝ) < (den : ℝ) := by exact_mod_cast h
    have hmul : (2 : ℝ) * (r : ℝ) < (den : ℝ) := by
      simpa [Nat.cast_mul, Nat.cast_ofNat, mul_assoc, mul_left_comm, mul_comm] using hR
    have hmul' : (2 : ℝ) * (r : ℝ) < (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) := by
      simpa [hscale] using hmul
    have hr : (r : ℝ) < (2⁻¹ : ℝ) * (den : ℝ) := (mul_lt_mul_iff_right₀ h2pos).1 hmul'
    exact h1.mpr hr

private lemma half_lt_div_iff (r den : Nat) (hden : den ≠ 0) :
    ((2⁻¹ : ℝ) < (r : ℝ) / (den : ℝ)) ↔ (den < 2 * r) := by
  have hdenpos : (0 : ℝ) < (den : ℝ) := by
    exact_mod_cast (Nat.pos_of_ne_zero hden)
  have h2pos : (0 : ℝ) < (2 : ℝ) := by norm_num
  have h1 :
      ((2⁻¹ : ℝ) < (r : ℝ) / (den : ℝ)) ↔ (2⁻¹ : ℝ) * (den : ℝ) < (r : ℝ) := by
    simpa using (lt_div_iff₀ hdenpos)
  have hscale : (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) = (den : ℝ) := by
    calc
      (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ))
          = ((2 : ℝ) * (2⁻¹ : ℝ)) * (den : ℝ) := by simp []
      _ = (1 : ℝ) * (den : ℝ) := by
          have h : (2 : ℝ) * (2⁻¹ : ℝ) = (1 : ℝ) := by
            simp
          simp []
      _ = (den : ℝ) := by simp
  constructor
  · intro h
    have hr : (2⁻¹ : ℝ) * (den : ℝ) < (r : ℝ) := (h1.mp h)
    have hmul : (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) < (2 : ℝ) * (r : ℝ) :=
      mul_lt_mul_of_pos_left hr h2pos
    have hmul' : (den : ℝ) < (2 : ℝ) * (r : ℝ) := by simpa [hscale] using hmul
    have : (den : ℝ) < ((2 * r : Nat) : ℝ) := by
      simpa [Nat.cast_mul, Nat.cast_ofNat, mul_assoc, mul_left_comm, mul_comm] using hmul'
    exact (by exact_mod_cast this)
  · intro h
    have hR : (den : ℝ) < ((2 * r : Nat) : ℝ) := by exact_mod_cast h
    have hmul : (den : ℝ) < (2 : ℝ) * (r : ℝ) := by
      simpa [Nat.cast_mul, Nat.cast_ofNat, mul_assoc, mul_left_comm, mul_comm] using hR
    have hmul' : (2 : ℝ) * ((2⁻¹ : ℝ) * (den : ℝ)) < (2 : ℝ) * (r : ℝ) := by
      simpa [hscale] using hmul
    have hr : (2⁻¹ : ℝ) * (den : ℝ) < (r : ℝ) := (mul_lt_mul_iff_right₀ h2pos).1 hmul'
    exact h1.mpr hr

private lemma neural_nearest_even_div_eq_roundQuotEven (num den : Nat) (hden : den ≠ 0) :
    TorchLean.Floats.neuralNearestEven ((num : ℝ) / (den : ℝ)) =
      Int.ofNat (roundQuotEven num den) := by
  classical
  set q : Nat := num / den
  set r : Nat := num % den
  have hfloor : (⌊((num : ℝ) / (den : ℝ))⌋ : Int) = q := by
    simpa [q] using (floor_real_nat_div (n := num) (den := den))
  have hfract : ((num : ℝ) / (den : ℝ)) - (q : ℝ) = (r : ℝ) / (den : ℝ) := by
    simpa [q, r] using (fract_real_div_nat (n := num) (den := den) hden)

  unfold TorchLean.Floats.neuralNearestEven
  simp [hfloor]
  rw [hfract]
  simp [roundQuotEven, q, r]

  have hlt :
      (((num % den : Nat) : ℝ) / (den : ℝ) < (2⁻¹ : ℝ)) ↔ (2 * (num % den) < den) := by
    simpa using (div_lt_half_iff (r := num % den) (den := den) hden)
  have hgt :
      ((2⁻¹ : ℝ) < ((num % den : Nat) : ℝ) / (den : ℝ)) ↔ (den < 2 * (num % den)) := by
    simpa using (half_lt_div_iff (r := num % den) (den := den) hden)

  by_cases h2lt : (2 * (num % den) < den)
  · have hrlt : (((num % den : Nat) : ℝ) / (den : ℝ) < (2⁻¹ : ℝ)) := (hlt.mpr h2lt)
    simp [h2lt, hrlt]
  · have hrlt : ¬(((num % den : Nat) : ℝ) / (den : ℝ) < (2⁻¹ : ℝ)) := by
      intro hr; exact h2lt (hlt.mp hr)
    by_cases h2gt : (den < 2 * (num % den))
    · have hrgt : ((2⁻¹ : ℝ) < ((num % den : Nat) : ℝ) / (den : ℝ)) := (hgt.mpr h2gt)
      simp [h2lt, hrlt, h2gt, hrgt]
    · have hrgt : ¬((2⁻¹ : ℝ) < ((num % den : Nat) : ℝ) / (den : ℝ)) := by
        intro hr; exact h2gt (hgt.mp hr)
      simp [h2lt, hrlt, h2gt, hrgt, Nat.even_iff]

/-!
## Signed nearest-even (and power-of-two specialization)

`FP32` uses “round to nearest, ties to even” (IEEE 754's default). The executable kernel implements
the same policy, but at the level of integer arithmetic on mantissas.

In this section we establish basic algebraic properties of nearest-even rounding that we can reuse
later when relating the `IEEE32Exec` rounding code to `fp32Round`.
-/

private lemma neural_nearest_even_neg (x : ℝ) :
    TorchLean.Floats.neuralNearestEven (-x) = -TorchLean.Floats.neuralNearestEven x := by
  classical
  -- Use `r = x - ⌊x⌋` for the case split (integer vs non-integer).
  set r : ℝ := x - (⌊x⌋ : ℝ)
  by_cases hr0 : r = 0
  · -- Integer case: both `x` and `-x` have fractional part `0`, so both round to their floors.
    have hx_eq : x = (⌊x⌋ : ℝ) := by linarith [hr0]
    have hceil : ⌈x⌉ = ⌊x⌋ := by
      have hx_le : x ≤ (⌊x⌋ : ℝ) := by linarith [hx_eq]
      have hceil_le : ⌈x⌉ ≤ ⌊x⌋ := (Int.ceil_le).2 hx_le
      exact le_antisymm hceil_le (Int.floor_le_ceil x)
    have hfloor_neg : ⌊-x⌋ = -⌊x⌋ := by
      calc
        ⌊-x⌋ = -⌈x⌉ := by simpa using (Int.floor_neg (a := x))
        _ = -⌊x⌋ := by simp [hceil]
    have hx_lt_half : x - (⌊x⌋ : ℝ) < (1 / 2 : ℝ) := by
      -- `x - ⌊x⌋ = 0`.
      simpa [r] using (by linarith [hr0] : r < (1 / 2 : ℝ))
    have hneg_lt_half : (-x) - (⌊-x⌋ : ℝ) < (1 / 2 : ℝ) := by
      have hcast : ((↑(-⌊x⌋) : ℝ)) = -((⌊x⌋ : ℤ) : ℝ) := by
        -- `Int.cast_neg` with explicit result type.
        simp
      have : (-x) - (⌊-x⌋ : ℝ) = 0 := by
        rw [hfloor_neg]
        rw [hcast]
        ring_nf
        linarith [hx_eq]
      linarith [this]
    have hx_round :
        TorchLean.Floats.neuralNearestEven x = ⌊x⌋ :=
      TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_lt_half x hx_lt_half
    have hy_round :
        TorchLean.Floats.neuralNearestEven (-x) = ⌊-x⌋ :=
      TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_lt_half (-x) hneg_lt_half
    -- Replace `⌊-x⌋` with `-⌊x⌋` and finish.
    simp [hx_round, hy_round, hfloor_neg]
  · -- Non-integer case: `⌈x⌉ = ⌊x⌋ + 1` and `⌊-x⌋ = -⌊x⌋ - 1`, and `fract(-x) = 1 - r`.
    have hx_ne : (⌊x⌋ : ℝ) ≠ x := by
      intro hx_eq
      apply hr0
      have : x - (⌊x⌋ : ℝ) = 0 := by linarith [hx_eq]
      simpa [r] using this
    have hx_lt : (⌊x⌋ : ℝ) < x :=
      lt_of_le_of_ne (Int.floor_le x) hx_ne
    have hceil : ⌈x⌉ = ⌊x⌋ + 1 := by
      have hx_le' : x ≤ ((⌊x⌋ + 1 : ℤ) : ℝ) := by
        have hx_lt_add : x < (⌊x⌋ : ℝ) + 1 := Int.lt_floor_add_one x
        have : x < ((⌊x⌋ : ℤ) : ℝ) + 1 := by simp
        have : x < ((⌊x⌋ + 1 : ℤ) : ℝ) := by
          simp [Int.cast_add, Int.cast_one]
        exact le_of_lt this
      apply (Int.ceil_eq_iff).2
      constructor
      · -- `((⌊x⌋+1):ℝ) - 1 = ⌊x⌋ < x`
        have : ((⌊x⌋ : ℤ) : ℝ) < x := hx_lt
        simpa [Int.cast_add, Int.cast_one, sub_eq_add_neg, add_assoc] using this
      · simpa using hx_le'
    have hfloor_neg : ⌊-x⌋ = -⌊x⌋ - 1 := by
      calc
        ⌊-x⌋ = -⌈x⌉ := by simpa using (Int.floor_neg (a := x))
        _ = -(⌊x⌋ + 1) := by simp [hceil]
        _ = -⌊x⌋ - 1 := by ring
    have hfract_neg : (-x) - (⌊-x⌋ : ℝ) = 1 - r := by
      -- Expand `⌊-x⌋ = -⌊x⌋ - 1`, then normalize.
      rw [hfloor_neg]
      have hcast : ((-⌊x⌋ - 1 : ℤ) : ℝ) = -((⌊x⌋ : ℤ) : ℝ) - 1 := by
        simp [Int.cast_neg, Int.cast_one]
      rw [hcast]
      dsimp [r]
      -- If `Int.fract` appears, unfold it explicitly (avoid `simp [Int.fract]` loops).
      rw [Int.fract]
      ring

    by_cases hlt : r < (1 / 2 : ℝ)
    · -- `r < 1/2`: `x` rounds down, `-x` rounds up.
      have hx_round :
          TorchLean.Floats.neuralNearestEven x = ⌊x⌋ := by
        have : x - (⌊x⌋ : ℝ) < (1 / 2 : ℝ) := by simpa [r] using hlt
        exact TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_lt_half x this
      have hneg_gt : (-x) - (⌊-x⌋ : ℝ) > (1 / 2 : ℝ) := by
        -- `1 - r > 1/2`.
        have : (1 / 2 : ℝ) < 1 - r := by linarith [hlt]
        -- rewrite using `hfract_neg`
        have : (1 / 2 : ℝ) < (-x) - (⌊-x⌋ : ℝ) := by simpa [hfract_neg] using this
        linarith
      have hy_round :
          TorchLean.Floats.neuralNearestEven (-x) = ⌊-x⌋ + 1 :=
        TorchLean.Floats.neural_nearest_even_eq_ceil_of_frac_gt_half (-x) hneg_gt
      have hfloor_succ : ⌊-x⌋ + 1 = -⌊x⌋ := by linarith [hfloor_neg]
      simp [hx_round, hy_round, hfloor_succ]
    · by_cases hgt : r > (1 / 2 : ℝ)
      · -- `r > 1/2`: `x` rounds up, `-x` rounds down.
        have hx_round :
            TorchLean.Floats.neuralNearestEven x = ⌊x⌋ + 1 := by
          have : x - (⌊x⌋ : ℝ) > (1 / 2 : ℝ) := by simpa [r] using hgt
          exact TorchLean.Floats.neural_nearest_even_eq_ceil_of_frac_gt_half x this
        have hneg_lt : (-x) - (⌊-x⌋ : ℝ) < (1 / 2 : ℝ) := by
          have : 1 - r < (1 / 2 : ℝ) := by linarith [hgt]
          simpa [hfract_neg] using this
        have hy_round :
            TorchLean.Floats.neuralNearestEven (-x) = ⌊-x⌋ :=
          TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_lt_half (-x) hneg_lt
        -- Reduce to an integer identity via `hfloor_neg`.
        rw [hy_round, hx_round, hfloor_neg]
        ring
      · -- Tie: `r = 1/2`; reduce to parity of the floor.
        have hr_eq : r = (1 / 2 : ℝ) := by
          have hge : (1 / 2 : ℝ) ≤ r := le_of_not_gt hlt
          have hle : r ≤ (1 / 2 : ℝ) := le_of_not_gt (by intro h; exact hgt h)
          exact le_antisymm hle hge
        have hneg_eq : (-x) - (⌊-x⌋ : ℝ) = (1 / 2 : ℝ) := by
          have : 1 - r = (1 / 2 : ℝ) := by linarith [hr_eq]
          simp [hfract_neg, this]

        have heven_floor_neg : Even (⌊-x⌋) ↔ ¬Even (⌊x⌋) := by
          -- `⌊-x⌋ = -(⌊x⌋ + 1)` and parity toggles under `+1`.
          have h1 : ⌊-x⌋ = -(⌊x⌋ + 1) := by linarith [hfloor_neg]
          -- `Even (-(a)) ↔ Even a` and `Even (a+1) ↔ ¬Even a`.
          have : Even (-(⌊x⌋ + 1)) ↔ ¬Even (⌊x⌋) := by
            exact (Iff.trans (even_neg (a := (⌊x⌋ + 1 : ℤ))) (Int.even_add_one (n := ⌊x⌋)))
          simpa [h1] using this

        by_cases hf : Even (⌊x⌋)
        · -- `x` rounds to `⌊x⌋`, `-x` rounds to `⌊-x⌋ + 1 = -⌊x⌋`.
          have hx_round :
              TorchLean.Floats.neuralNearestEven x = ⌊x⌋ := by
            have : x - (⌊x⌋ : ℝ) = (1 / 2 : ℝ) := by simpa [r] using hr_eq
            exact TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_half_even x this hf
          have hfloor_odd : ¬Even (⌊-x⌋) := by
            intro hef
            have : ¬Even (⌊x⌋) := (heven_floor_neg.mp hef)
            exact this hf
          have hy_round :
              TorchLean.Floats.neuralNearestEven (-x) = ⌊-x⌋ + 1 :=
            TorchLean.Floats.neural_nearest_even_eq_ceil_of_frac_half_odd (-x) hneg_eq hfloor_odd
          have hfloor_succ : ⌊-x⌋ + 1 = -⌊x⌋ := by linarith [hfloor_neg]
          simp [hx_round, hy_round, hfloor_succ]
        · -- `x` rounds to `⌊x⌋+1`, `-x` rounds to `⌊-x⌋ = -⌊x⌋-1`.
          have hx_round :
              TorchLean.Floats.neuralNearestEven x = ⌊x⌋ + 1 := by
            have : x - (⌊x⌋ : ℝ) = (1 / 2 : ℝ) := by simpa [r] using hr_eq
            exact TorchLean.Floats.neural_nearest_even_eq_ceil_of_frac_half_odd x this hf
          have hfloor_even : Even (⌊-x⌋) := by
            exact (heven_floor_neg.mpr hf)
          have hy_round :
              TorchLean.Floats.neuralNearestEven (-x) = ⌊-x⌋ :=
            TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_half_even (-x) hneg_eq hfloor_even
          -- `-(⌊x⌋+1) = -⌊x⌋-1`.
          rw [hy_round, hx_round, hfloor_neg]
          ring

private lemma neural_nearest_even_neg_div_eq_roundQuotEven (num den : Nat) (hden : den ≠ 0) :
    TorchLean.Floats.neuralNearestEven (-((num : ℝ) / (den : ℝ))) =
      -Int.ofNat (roundQuotEven num den) := by
  have hpos :=
    neural_nearest_even_div_eq_roundQuotEven (num := num) (den := den) hden
  -- Use oddness and then the nonnegative rational lemma.
  have hodd :=
    neural_nearest_even_neg (x := (num : ℝ) / (den : ℝ))
  -- `neural_nearest_even (-x) = -neural_nearest_even x`.
  simpa [hpos] using hodd

private lemma roundShiftRightEven_eq_roundQuotEven_pow2 (n shift : Nat) :
    roundShiftRightEven n shift = roundQuotEven n (pow2 shift) := by
  classical
  cases shift with
  | zero =>
      -- `pow2 0 = 1`, and `roundQuotEven n 1 = n`.
      simp [roundShiftRightEven, pow2, roundQuotEven, Nat.div_one, Nat.mod_one]
  | succ s =>
      -- Abbreviations.
      let den : Nat := pow2 (Nat.succ s)
      let half : Nat := pow2 s
      have hden : den = 2 * half := by
        -- `2^(s+1) = 2 * 2^s`.
        simp [den, half, pow2_eq_two_pow, Nat.pow_succ, Nat.mul_comm]

      -- `Nat.shiftRight`/`Nat.shiftLeft` simp to `>>>`/`<<<`, so work in that notation.
      have hq : n >>> (Nat.succ s) = n / den := by
        -- `n >>> k = n / 2^k` and `den = 2^k`.
        have : n >>> (Nat.succ s) = n / (2 ^ Nat.succ s) := Nat.shiftRight_eq_div_pow n (Nat.succ s)
        simpa [den, pow2_eq_two_pow] using this

      have hrem : n - (n >>> (Nat.succ s) <<< (Nat.succ s)) = n % den := by
        -- Replace shifts with `/` and `*`, then use the division algorithm.
        have hshiftLeft : (n >>> (Nat.succ s) <<< (Nat.succ s)) = (n / den) * den := by
          -- `a <<< k = a * 2^k`.
          simp [Nat.shiftLeft_eq, hq, den, pow2_eq_two_pow, Nat.mul_comm]
        rw [hshiftLeft]
        -- `((n/den)*den) + (n%den) = n`.
        have hdiv : (n / den) * den + n % den = n := by
          simpa [Nat.mul_comm] using (Nat.div_add_mod n den)
        calc
          n - (n / den) * den = ((n / den) * den + n % den) - (n / den) * den := by simp [hdiv]
          _ = n % den := Nat.add_sub_cancel_left _ _

      -- Now both rounders are the same case analysis, just phrased differently.
      have hshift_def :
          roundShiftRightEven n (Nat.succ s) =
            (let q := n / den
             let r := n % den
             if r < half then q
             else if half < r then q + 1
             else if q % 2 == 0 then q else q + 1) := by
        have hrem_div : n - (n / den) <<< (Nat.succ s) = n % den := by
          -- `simp` will rewrite `n >>> _` into `n / den`, so rewrite `hrem` first.
          simpa [hq] using hrem
        -- Unfold and rewrite `q`/`rem`.
        simp [roundShiftRightEven, den, half, hq, hrem_div]

      have hquot_def :
          roundQuotEven n den =
            (let q := n / den
             let r := n % den
             let twice := 2 * r
             if twice < den then q
             else if den < twice then q + 1
             else if q % 2 == 0 then q else q + 1) := by
        simp [roundQuotEven, den]

      -- Finish by splitting on `r` relative to `half`.
      -- `den = 2 * half`, so comparisons against `half` match comparisons of `2*r` against `den`.
      -- Expand both sides and do a 3-way case split.
      rw [hshift_def]
      -- Rewrite RHS goal to use `roundQuotEven n den`.
      -- `pow2 (succ s)` is `den` by definition.
      have : roundQuotEven n (pow2 (Nat.succ s)) = roundQuotEven n den := by rfl
      -- Unfold `roundQuotEven` with `den`.
      rw [this, hquot_def]

      -- Now compare the conditionals.
      -- Let-bindings: keep `q`/`r` as in both sides.
      -- A small local `simp` step exposes the shared structure.
      simp only
      -- Case split on the remainder.
      by_cases hrlt : n % den < half
      · -- `r < half` ⇒ `2*r < den`.
        have htw_lt : 2 * (n % den) < den := by
          -- `2*r < 2*half`.
          have : 2 * (n % den) < 2 * half :=
            (Nat.mul_lt_mul_left (by decide : 0 < (2 : Nat))).2 hrlt
          simpa [hden, Nat.mul_assoc] using this
        simp [hrlt, htw_lt]
      · by_cases hrgt : half < n % den
        · -- `half < r` ⇒ `den < 2*r`.
          have htw_gt : den < 2 * (n % den) := by
            have : 2 * half < 2 * (n % den) :=
              (Nat.mul_lt_mul_left (by decide : 0 < (2 : Nat))).2 hrgt
            -- Rewrite `2*half` as `den`.
            exact lt_of_eq_of_lt hden this
          have htw_lt' : ¬(2 * (n % den) < den) := by
            intro h; exact (not_lt_of_ge (le_of_lt htw_gt)) h
          simp [hrlt, hrgt, htw_lt', htw_gt]
        · -- Tie: `r = half`, so `2*r = den` and both use the parity branch.
          have hre : n % den = half := by
            exact le_antisymm (le_of_not_gt hrgt) (le_of_not_gt hrlt)
          have hre₂ : n % (2 * half) = half := by
            -- Useful because simp will rewrite `den` using `hden`.
            simpa [hden] using hre
          have htw_eq : 2 * (n % den) = den := by
            -- `2*half = den`
            calc
              2 * (n % den) = 2 * half := by simp [hre]
              _ = den := hden.symm
          have htw_lt' : ¬(2 * (n % den) < den) := by
            simp [htw_eq]
          have htw_gt' : ¬(den < 2 * (n % den)) := by
            simp [htw_eq]
          -- Give `simp` the rewritten remainder for `den = 2*half`.
          simp [hrlt, hrgt, htw_lt', htw_gt']

private lemma neural_nearest_even_div_pow2_eq_roundShiftRightEven (num shift : Nat) :
    TorchLean.Floats.neuralNearestEven ((num : ℝ) / (pow2 shift : ℝ)) =
      Int.ofNat (roundShiftRightEven num shift) := by
  have hden : pow2 shift ≠ 0 := by
    have : 0 < pow2 shift := by
      simp [pow2_eq_two_pow]
    exact Nat.ne_of_gt this
  -- Reduce to the generic rational lemma via `roundQuotEven`, then rewrite to
  -- `roundShiftRightEven`.
  have h :=
    neural_nearest_even_div_eq_roundQuotEven (num := num) (den := pow2 shift) hden
  simpa [roundShiftRightEven_eq_roundQuotEven_pow2 (n := num) (shift := shift)] using h

private lemma neural_nearest_even_sqrt_nat (n : Nat) :
    TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) =
      Int.ofNat
        (let q : Nat := Nat.sqrt n
         let r : Nat := n - q * q
         if r > q then q + 1 else q) := by
  classical
  set q : Nat := Nat.sqrt n
  set r : Nat := n - q * q
  have hqle : q * q ≤ n := by
    simpa [q, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using (Nat.sqrt_le n)
  have hn_eq : n = q * q + r := by
    have : (n - q * q) + q * q = n := Nat.sub_add_cancel hqle
    calc
      n = (n - q * q) + q * q := this.symm
      _ = r + q * q := by simp [r]
      _ = q * q + r := by ac_rfl
  have hfloor : (⌊Real.sqrt (n : ℝ)⌋ : Int) = q := by
    simp [q, Real.floor_real_sqrt_eq_nat_sqrt (a := n)]

  by_cases hgt : r > q
  · have hrge : q + 1 ≤ r := Nat.succ_le_of_lt hgt
    have hn_ge : q * q + (q + 1) ≤ n := by
      have : q * q + (q + 1) ≤ q * q + r := Nat.add_le_add_left hrge (q * q)
      simpa [hn_eq] using this
    have hmid_lt : ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 < (n : ℝ) := by
      have h1 : ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 < (q * q + (q + 1) : ℝ) := by
        have hR : (q * q + (q + 1) : ℝ) = (q : ℝ) ^ 2 + (q : ℝ) + 1 := by
          simp [pow_two]
          ring
        have hL : ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 = (q : ℝ) ^ 2 + (q : ℝ) + (1 / 4 : ℝ) := by
          ring
        rw [hL, hR]
        linarith
      have h2 : (q * q + (q + 1) : ℝ) ≤ (n : ℝ) := by
        exact_mod_cast hn_ge
      exact lt_of_lt_of_le h1 h2
    have hx0 : 0 ≤ (q : ℝ) + (1 / 2 : ℝ) := by
      have : 0 ≤ (q : ℝ) := by exact_mod_cast (Nat.zero_le q)
      linarith
    have hgt_mid : (q : ℝ) + (1 / 2 : ℝ) < Real.sqrt (n : ℝ) := by
      exact
        (Real.lt_sqrt (x := (q : ℝ) + (1 / 2 : ℝ)) (y := (n : ℝ)) hx0).2 (by
          simpa using hmid_lt)
    have hfrac_gt : Real.sqrt (n : ℝ) - (⌊Real.sqrt (n : ℝ)⌋ : ℝ) > (1 / 2 : ℝ) := by
      have hfloorR : (⌊Real.sqrt (n : ℝ)⌋ : ℝ) = (q : ℝ) := by
        have := congrArg (fun z : Int => (z : ℝ)) hfloor
        simpa using this
      have : (1 / 2 : ℝ) < Real.sqrt (n : ℝ) - (q : ℝ) := by linarith [hgt_mid]
      simpa [hfloorR] using this
    have hround :
        TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) =
          (⌊Real.sqrt (n : ℝ)⌋ : Int) + 1 :=
      TorchLean.Floats.neural_nearest_even_eq_ceil_of_frac_gt_half (Real.sqrt (n : ℝ)) (by
        simpa using hfrac_gt)
    have hLHS : TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) = Int.ofNat (q + 1) := by
      calc
        TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) = (⌊Real.sqrt (n : ℝ)⌋ : Int) + 1
          := hround
        _ = (q : Int) + 1 := by simp [hfloor]
        _ = Int.ofNat (q + 1) := by simp
    simpa [r, hgt] using hLHS
  · have hle : r ≤ q := Nat.le_of_not_gt hgt
    have hn_le : n ≤ q * q + q := by
      have : q * q + r ≤ q * q + q := Nat.add_le_add_left hle (q * q)
      simpa [hn_eq] using this
    have hmid_gt : (n : ℝ) < ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 := by
      have h1 : (n : ℝ) ≤ (q * q + q : ℝ) := by
        exact_mod_cast hn_le
      have h2 : (q * q + q : ℝ) < ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 := by
        have hR : (q * q + q : ℝ) = (q : ℝ) ^ 2 + (q : ℝ) := by
          simp [pow_two]
        have hL : ((q : ℝ) + (1 / 2 : ℝ)) ^ 2 = (q : ℝ) ^ 2 + (q : ℝ) + (1 / 4 : ℝ) := by
          ring
        rw [hR, hL]
        linarith
      exact lt_of_le_of_lt h1 h2
    have hy0 : 0 < (q : ℝ) + (1 / 2 : ℝ) := by
      have : 0 ≤ (q : ℝ) := by exact_mod_cast (Nat.zero_le q)
      linarith
    have hlt_mid : Real.sqrt (n : ℝ) < (q : ℝ) + (1 / 2 : ℝ) := by
      exact
        (Real.sqrt_lt' (x := (n : ℝ)) (y := (q : ℝ) + (1 / 2 : ℝ)) hy0).2 (by
          simpa using hmid_gt)
    have hfrac_lt : Real.sqrt (n : ℝ) - (⌊Real.sqrt (n : ℝ)⌋ : ℝ) < (1 / 2 : ℝ) := by
      have hfloorR : (⌊Real.sqrt (n : ℝ)⌋ : ℝ) = (q : ℝ) := by
        have := congrArg (fun z : Int => (z : ℝ)) hfloor
        simpa using this
      have : Real.sqrt (n : ℝ) - (q : ℝ) < (1 / 2 : ℝ) := by linarith [hlt_mid]
      simpa [hfloorR] using this
    have hround :
        TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) =
          (⌊Real.sqrt (n : ℝ)⌋ : Int) :=
      TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_lt_half (Real.sqrt (n : ℝ)) (by
        simpa using hfrac_lt)
    have hLHS : TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) = Int.ofNat q := by
      calc
        TorchLean.Floats.neuralNearestEven (Real.sqrt (n : ℝ)) = (⌊Real.sqrt (n : ℝ)⌋ : Int) :=
          hround
        _ = q := hfloor
        _ = Int.ofNat q := by simp
    simpa [r, hgt] using hLHS

/-!
## Real semantics of exponent tests for rationals

Some executable rounding code works by inspecting the magnitude of a rational (represented as
`num / den` with `Nat`s) and branching on exponent ranges. These lemmas justify those branches in
terms of real inequalities, so later refinement proofs can stay “math-first”.
-/

private lemma ratLtPow2_eq_true_iff (num den : Nat) (k : Int) (hden : den ≠ 0) :
    ratLtPow2 num den k = true ↔ (num : ℝ) / (den : ℝ) < neuralBpow binaryRadix k := by
  classical
  cases k with
  | ofNat kn =>
      simp [ratLtPow2]
      have hden_pos : (0 : ℝ) < (den : ℝ) := by
        exact_mod_cast (Nat.pos_of_ne_zero hden)
      have hshift : (Nat.shiftLeft den kn : ℝ) = (den : ℝ) * (2 : ℝ) ^ kn := by
        simp [Nat.shiftLeft_eq, Nat.cast_mul, Nat.cast_pow]
      constructor
      · intro h
        have hR : (num : ℝ) < (Nat.shiftLeft den kn : ℝ) := by
          exact_mod_cast h
        have hmul : (num : ℝ) < (den : ℝ) * (2 : ℝ) ^ kn :=
          lt_of_lt_of_eq hR hshift
        have hgoal :
            (num : ℝ) / (den : ℝ) < (2 : ℝ) ^ kn ↔ (num : ℝ) < (2 : ℝ) ^ kn * (den : ℝ) := by
          simpa using
            (div_lt_iff₀ (a := (2 : ℝ) ^ kn) (b := (num : ℝ)) (c := (den : ℝ)) hden_pos)
        have hdiv : (num : ℝ) / (den : ℝ) < (2 : ℝ) ^ kn := by
          apply hgoal.mpr
          simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
        simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using hdiv
      · intro h
        have h' : (num : ℝ) / (den : ℝ) < (2 : ℝ) ^ kn := by
          simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using h
        have hgoal :
            (num : ℝ) / (den : ℝ) < (2 : ℝ) ^ kn ↔ (num : ℝ) < (2 : ℝ) ^ kn * (den : ℝ) := by
          simpa using
            (div_lt_iff₀ (a := (2 : ℝ) ^ kn) (b := (num : ℝ)) (c := (den : ℝ)) hden_pos)
        have hmul : (num : ℝ) < (den : ℝ) * (2 : ℝ) ^ kn := by
          have : (num : ℝ) < (2 : ℝ) ^ kn * (den : ℝ) := (hgoal.mp h')
          simpa [mul_assoc, mul_left_comm, mul_comm] using this
        have hR : (num : ℝ) < (Nat.shiftLeft den kn : ℝ) :=
          lt_of_lt_of_eq hmul hshift.symm
        exact_mod_cast hR
  | negSucc kn =>
      simp [ratLtPow2]
      have hden_pos : (0 : ℝ) < (den : ℝ) := by
        exact_mod_cast (Nat.pos_of_ne_zero hden)
      have hpow_pos : (0 : ℝ) < (2 : ℝ) ^ (kn + 1) := by
        exact pow_pos (by norm_num : (0 : ℝ) < 2) _
      have hbpow : neuralBpow binaryRadix (Int.negSucc kn) = (1 : ℝ) / (2 : ℝ) ^ (kn + 1) := by
        simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, zpow_negSucc,
          div_eq_mul_inv]
      have hshift : (Nat.shiftLeft num (kn + 1) : ℝ) = (num : ℝ) * (2 : ℝ) ^ (kn + 1) := by
        simp [Nat.shiftLeft_eq, Nat.cast_mul, Nat.cast_pow]
      constructor
      · intro h
        have hR : (Nat.shiftLeft num (kn + 1) : ℝ) < (den : ℝ) := by
          exact_mod_cast h
        have hmul : (num : ℝ) * (2 : ℝ) ^ (kn + 1) < (den : ℝ) :=
          lt_of_eq_of_lt hshift.symm hR
        have hnum_lt : (num : ℝ) < (den : ℝ) / (2 : ℝ) ^ (kn + 1) :=
          (lt_div_iff₀ hpow_pos).2 hmul
        have hgoal :
            (num : ℝ) / (den : ℝ) < (1 : ℝ) / (2 : ℝ) ^ (kn + 1) ↔
              (num : ℝ) < ((1 : ℝ) / (2 : ℝ) ^ (kn + 1)) * (den : ℝ) := by
          simpa using
            (div_lt_iff₀ (a := (1 : ℝ) / (2 : ℝ) ^ (kn + 1)) (b := (num : ℝ)) (c := (den : ℝ))
              hden_pos)
        have hdiv : (num : ℝ) / (den : ℝ) < (1 : ℝ) / (2 : ℝ) ^ (kn + 1) := by
          apply hgoal.mpr
          simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using hnum_lt
        simpa [hbpow] using hdiv
      · intro h
        have h' : (num : ℝ) / (den : ℝ) < (1 : ℝ) / (2 : ℝ) ^ (kn + 1) := by
          simpa [hbpow] using h
        have hgoal :
            (num : ℝ) / (den : ℝ) < (1 : ℝ) / (2 : ℝ) ^ (kn + 1) ↔
              (num : ℝ) < ((1 : ℝ) / (2 : ℝ) ^ (kn + 1)) * (den : ℝ) := by
          simpa using
            (div_lt_iff₀ (a := (1 : ℝ) / (2 : ℝ) ^ (kn + 1)) (b := (num : ℝ)) (c := (den : ℝ))
              hden_pos)
        have hnum_lt : (num : ℝ) < (den : ℝ) / (2 : ℝ) ^ (kn + 1) := by
          have : (num : ℝ) < ((1 : ℝ) / (2 : ℝ) ^ (kn + 1)) * (den : ℝ) := (hgoal.mp h')
          simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using this
        have hmul : (num : ℝ) * (2 : ℝ) ^ (kn + 1) < (den : ℝ) :=
          (lt_div_iff₀ hpow_pos).1 hnum_lt
        have hR : (Nat.shiftLeft num (kn + 1) : ℝ) < (den : ℝ) :=
          lt_of_eq_of_lt hshift hmul
        exact_mod_cast hR

private lemma ratGePow2_eq_true_iff (num den : Nat) (k : Int) (hden : den ≠ 0) :
    ratGePow2 num den k = true ↔ neuralBpow binaryRadix k ≤ (num : ℝ) / (den : ℝ) := by
  classical
  cases k with
  | ofNat kn =>
      simp [ratGePow2]
      have hden_pos : (0 : ℝ) < (den : ℝ) := by
        exact_mod_cast (Nat.pos_of_ne_zero hden)
      have hshift : (Nat.shiftLeft den kn : ℝ) = (den : ℝ) * (2 : ℝ) ^ kn := by
        simp [Nat.shiftLeft_eq, Nat.cast_mul, Nat.cast_pow]
      constructor
      · intro h
        have hR : (Nat.shiftLeft den kn : ℝ) ≤ (num : ℝ) := by
          exact_mod_cast h
        have hmul : (den : ℝ) * (2 : ℝ) ^ kn ≤ (num : ℝ) :=
          le_of_eq_of_le hshift.symm hR
        have hgoal :
            (2 : ℝ) ^ kn ≤ (num : ℝ) / (den : ℝ) ↔ (2 : ℝ) ^ kn * (den : ℝ) ≤ (num : ℝ) := by
          simpa using
            (le_div_iff₀ (a := (2 : ℝ) ^ kn) (b := (num : ℝ)) (c := (den : ℝ)) hden_pos)
        have hdiv : (2 : ℝ) ^ kn ≤ (num : ℝ) / (den : ℝ) := by
          apply hgoal.mpr
          simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
        simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using hdiv
      · intro h
        have h' : (2 : ℝ) ^ kn ≤ (num : ℝ) / (den : ℝ) := by
          simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using h
        have hgoal :
            (2 : ℝ) ^ kn ≤ (num : ℝ) / (den : ℝ) ↔ (2 : ℝ) ^ kn * (den : ℝ) ≤ (num : ℝ) := by
          simpa using
            (le_div_iff₀ (a := (2 : ℝ) ^ kn) (b := (num : ℝ)) (c := (den : ℝ)) hden_pos)
        have hmul : (den : ℝ) * (2 : ℝ) ^ kn ≤ (num : ℝ) := by
          have : (2 : ℝ) ^ kn * (den : ℝ) ≤ (num : ℝ) := (hgoal.mp h')
          simpa [mul_assoc, mul_left_comm, mul_comm] using this
        have hR : (Nat.shiftLeft den kn : ℝ) ≤ (num : ℝ) :=
          le_of_eq_of_le hshift hmul
        exact_mod_cast hR
  | negSucc kn =>
      simp [ratGePow2]
      have hden_pos : (0 : ℝ) < (den : ℝ) := by
        exact_mod_cast (Nat.pos_of_ne_zero hden)
      have hpow_pos : (0 : ℝ) < (2 : ℝ) ^ (kn + 1) := by
        exact pow_pos (by norm_num : (0 : ℝ) < 2) _
      have hbpow : neuralBpow binaryRadix (Int.negSucc kn) = (1 : ℝ) / (2 : ℝ) ^ (kn + 1) := by
        simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, zpow_negSucc,
          div_eq_mul_inv]
      have hshift : (Nat.shiftLeft num (kn + 1) : ℝ) = (num : ℝ) * (2 : ℝ) ^ (kn + 1) := by
        simp [Nat.shiftLeft_eq, Nat.cast_mul, Nat.cast_pow]
      constructor
      · intro h
        have hR : (den : ℝ) ≤ (Nat.shiftLeft num (kn + 1) : ℝ) := by
          exact_mod_cast h
        have hmul : (den : ℝ) ≤ (num : ℝ) * (2 : ℝ) ^ (kn + 1) :=
          le_trans hR (le_of_eq hshift)
        have hnum_le : (den : ℝ) / (2 : ℝ) ^ (kn + 1) ≤ (num : ℝ) :=
          (div_le_iff₀ hpow_pos).2 (by simpa [mul_comm, mul_left_comm, mul_assoc] using hmul)
        have hgoal :
            (1 : ℝ) / (2 : ℝ) ^ (kn + 1) ≤ (num : ℝ) / (den : ℝ) ↔
              ((1 : ℝ) / (2 : ℝ) ^ (kn + 1)) * (den : ℝ) ≤ (num : ℝ) := by
          simpa using
            (le_div_iff₀ (a := (1 : ℝ) / (2 : ℝ) ^ (kn + 1)) (b := (num : ℝ)) (c := (den : ℝ))
              hden_pos)
        have hdiv : (1 : ℝ) / (2 : ℝ) ^ (kn + 1) ≤ (num : ℝ) / (den : ℝ) := by
          apply hgoal.mpr
          simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using hnum_le
        simpa [hbpow] using hdiv
      · intro h
        have h' : (1 : ℝ) / (2 : ℝ) ^ (kn + 1) ≤ (num : ℝ) / (den : ℝ) := by
          simpa [hbpow] using h
        have hgoal :
            (1 : ℝ) / (2 : ℝ) ^ (kn + 1) ≤ (num : ℝ) / (den : ℝ) ↔
              ((1 : ℝ) / (2 : ℝ) ^ (kn + 1)) * (den : ℝ) ≤ (num : ℝ) := by
          simpa using
            (le_div_iff₀ (a := (1 : ℝ) / (2 : ℝ) ^ (kn + 1)) (b := (num : ℝ)) (c := (den : ℝ))
              hden_pos)
        have hnum_le : (den : ℝ) / (2 : ℝ) ^ (kn + 1) ≤ (num : ℝ) := by
          have : ((1 : ℝ) / (2 : ℝ) ^ (kn + 1)) * (den : ℝ) ≤ (num : ℝ) := (hgoal.mp h')
          simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using this
        have hmul : (den : ℝ) ≤ (num : ℝ) * (2 : ℝ) ^ (kn + 1) :=
          (div_le_iff₀ hpow_pos).1 hnum_le
        have hR : (den : ℝ) ≤ (Nat.shiftLeft num (kn + 1) : ℝ) :=
          le_trans hmul (le_of_eq hshift.symm)
        exact_mod_cast hR

/-!
## Coarse log₂ bounds for rationals

The rounding code needs cheap bounds on `log₂` (or “bit-length”) to decide normal vs subnormal
cases. We prove coarse but robust bounds that are easy to compute from `Nat.log2`.
-/

private lemma bpow_k0_sub_one_eq (ln ld : Nat) :
    neuralBpow binaryRadix (Int.ofNat ln - Int.ofNat ld - 1) =
      (2 : ℝ) ^ ln / (2 : ℝ) ^ ld.succ := by
  simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, zpow_sub₀]
  simp [pow_succ, div_eq_mul_inv, mul_left_comm, mul_comm]

private lemma bpow_k0_add_one_eq (ln ld : Nat) :
    neuralBpow binaryRadix (Int.ofNat ln - Int.ofNat ld + 1) =
      (2 : ℝ) ^ ln.succ / (2 : ℝ) ^ ld := by
  have hk : (Int.ofNat ln) - (Int.ofNat ld) + 1 = (Int.ofNat ln.succ) - (Int.ofNat ld) := by
    simp [sub_eq_add_neg, add_assoc, add_left_comm, add_comm]
  rw [hk]
  simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, zpow_sub₀]
  have hnum : (2 : ℝ) ^ (↑ln + 1 : Int) = (2 : ℝ) ^ ln.succ := by
    have hexp : (↑ln + 1 : Int) = (Int.ofNat ln.succ) := by
      simp
    rw [hexp]
    exact zpow_ofNat (2 : ℝ) ln.succ
  rw [hnum]

private lemma rat_bounds_k0 (num den : Nat) (hnum : num ≠ 0) (hden : den ≠ 0) :
    let ln : Nat := Nat.log2 num
    let ld : Nat := Nat.log2 den
    let k0 : Int := (Int.ofNat ln) - (Int.ofNat ld)
    neuralBpow binaryRadix (k0 - 1) ≤ (num : ℝ) / (den : ℝ) ∧
      (num : ℝ) / (den : ℝ) < neuralBpow binaryRadix (k0 + 1) := by
  classical
  set ln : Nat := Nat.log2 num
  set ld : Nat := Nat.log2 den
  set k0 : Int := (Int.ofNat ln) - (Int.ofNat ld)

  have hnum_ge_nat : 2 ^ ln ≤ num := by
    have h := Nat.pow_log_le_self (b := 2) (x := num) hnum
    simpa [ln, Nat.log2_eq_log_two] using h
  have hnum_lt_nat : num < 2 ^ ln.succ := by
    have h := Nat.lt_pow_succ_log_self (b := 2) (hb := Nat.one_lt_two) num
    simpa [ln, Nat.log2_eq_log_two] using h
  have hden_ge_nat : 2 ^ ld ≤ den := by
    have h := Nat.pow_log_le_self (b := 2) (x := den) hden
    simpa [ld, Nat.log2_eq_log_two] using h
  have hden_lt_nat : den < 2 ^ ld.succ := by
    have h := Nat.lt_pow_succ_log_self (b := 2) (hb := Nat.one_lt_two) den
    simpa [ld, Nat.log2_eq_log_two] using h

  have hnum_ge : (2 : ℝ) ^ ln ≤ (num : ℝ) := by
    have : ((2 ^ ln : Nat) : ℝ) ≤ (num : ℝ) := by exact_mod_cast hnum_ge_nat
    simpa [Nat.cast_pow] using this
  have hnum_lt : (num : ℝ) < (2 : ℝ) ^ ln.succ := by
    have : (num : ℝ) < ((2 ^ ln.succ : Nat) : ℝ) := by exact_mod_cast hnum_lt_nat
    simpa [Nat.cast_pow] using this
  have hden_ge : (2 : ℝ) ^ ld ≤ (den : ℝ) := by
    have : ((2 ^ ld : Nat) : ℝ) ≤ (den : ℝ) := by exact_mod_cast hden_ge_nat
    simpa [Nat.cast_pow] using this
  have hden_lt : (den : ℝ) < (2 : ℝ) ^ ld.succ := by
    have : (den : ℝ) < ((2 ^ ld.succ : Nat) : ℝ) := by exact_mod_cast hden_lt_nat
    simpa [Nat.cast_pow] using this

  have hden_pos : (0 : ℝ) < (den : ℝ) := by
    exact_mod_cast (Nat.pos_of_ne_zero hden)

  have hlo_pow : (2 : ℝ) ^ ln / (2 : ℝ) ^ ld.succ ≤ (num : ℝ) / (den : ℝ) := by
    have hpow_pos : (0 : ℝ) < (2 : ℝ) ^ ld.succ := by
      exact pow_pos (by norm_num : (0 : ℝ) < 2) _
    have hden_le : (den : ℝ) ≤ (2 : ℝ) ^ ld.succ := le_of_lt hden_lt
    have h1 : (num : ℝ) / (2 : ℝ) ^ ld.succ ≤ (num : ℝ) / (den : ℝ) :=
      div_le_div_of_nonneg_left (Nat.cast_nonneg num) hden_pos hden_le
    have h2 : (2 : ℝ) ^ ln / (2 : ℝ) ^ ld.succ ≤ (num : ℝ) / (2 : ℝ) ^ ld.succ :=
      div_le_div_of_nonneg_right hnum_ge (le_of_lt hpow_pos)
    exact le_trans h2 h1

  have hhi_pow : (num : ℝ) / (den : ℝ) < (2 : ℝ) ^ ln.succ / (2 : ℝ) ^ ld := by
    have h3 : (num : ℝ) / (den : ℝ) < (2 : ℝ) ^ ln.succ / (den : ℝ) :=
      div_lt_div_of_pos_right hnum_lt hden_pos
    have hpow_pos : (0 : ℝ) < (2 : ℝ) ^ ld := by
      exact pow_pos (by norm_num : (0 : ℝ) < 2) _
    have hpow_le : (2 : ℝ) ^ ld ≤ (den : ℝ) := hden_ge
    have h4 : (2 : ℝ) ^ ln.succ / (den : ℝ) ≤ (2 : ℝ) ^ ln.succ / (2 : ℝ) ^ ld :=
      div_le_div_of_nonneg_left (le_of_lt (pow_pos (by norm_num : (0 : ℝ) < 2) _)) hpow_pos hpow_le
    exact lt_of_lt_of_le h3 h4

  have hbpow_lo : neuralBpow binaryRadix (k0 - 1) = (2 : ℝ) ^ ln / (2 : ℝ) ^ ld.succ := by
    have : neuralBpow binaryRadix (k0 - 1) =
        neuralBpow binaryRadix (Int.ofNat ln - Int.ofNat ld - 1) := by
      simp [k0, sub_eq_add_neg, add_assoc]
    rw [this]
    exact bpow_k0_sub_one_eq (ln := ln) (ld := ld)

  have hbpow_hi : neuralBpow binaryRadix (k0 + 1) = (2 : ℝ) ^ ln.succ / (2 : ℝ) ^ ld := by
    have : neuralBpow binaryRadix (k0 + 1) =
        neuralBpow binaryRadix (Int.ofNat ln - Int.ofNat ld + 1) := by
      simp [k0, sub_eq_add_neg, add_assoc]
    rw [this]
    exact bpow_k0_add_one_eq (ln := ln) (ld := ld)

  refine ⟨?_, ?_⟩
  · have : neuralBpow binaryRadix (k0 - 1) ≤ (num : ℝ) / (den : ℝ) := by
      rw [hbpow_lo]
      exact hlo_pow
    simpa using this
  · have : (num : ℝ) / (den : ℝ) < neuralBpow binaryRadix (k0 + 1) := by
      rw [hbpow_hi]
      exact hhi_pow
    simpa using this

private lemma floorLog2Rat_bounds (num den : Nat) (hnum : num ≠ 0) (hden : den ≠ 0) :
    let k : Int := floorLog2Rat num den
    neuralBpow binaryRadix k ≤ (num : ℝ) / (den : ℝ) ∧
      (num : ℝ) / (den : ℝ) < neuralBpow binaryRadix (k + 1) := by
  classical
  have hden_pos : (0 : ℝ) < (den : ℝ) := by
    exact_mod_cast (Nat.pos_of_ne_zero hden)
  set r : ℝ := (num : ℝ) / (den : ℝ)

  -- Unfold `floorLog2Rat` into the intermediate candidate `k0` and adjusted exponent `k1`.
  set k0 : Int := (Int.ofNat (Nat.log2 num)) - (Int.ofNat (Nat.log2 den))
  set k1 : Int := if ratLtPow2 num den k0 then k0 - 1 else k0

  have hk0_bounds : neuralBpow binaryRadix (k0 - 1) ≤ r ∧ r < neuralBpow binaryRadix (k0 + 1) :=
    by
    simpa [r, k0] using (rat_bounds_k0 (num := num) (den := den) hnum hden)

  have hk1_ge : neuralBpow binaryRadix k1 ≤ r := by
    by_cases hlt : ratLtPow2 num den k0 = true
    · have hk1 : k1 = k0 - 1 := by simp [k1, hlt]
      simpa [hk1] using hk0_bounds.1
    · have hltFalse : ratLtPow2 num den k0 = false := by
        cases hb : ratLtPow2 num den k0 with
        | true =>
            exfalso
            exact hlt (by simpa using hb)
        | false =>
            simp
      have hk1 : k1 = k0 := by simp [k1, hltFalse]
      have hr_not_lt : ¬ r < neuralBpow binaryRadix k0 := by
        intro hr_lt
        have : ratLtPow2 num den k0 = true :=
          (ratLtPow2_eq_true_iff (num := num) (den := den) (k := k0) hden).2 (by simpa [r] using
            hr_lt)
        exact hlt this
      have : neuralBpow binaryRadix k0 ≤ r := le_of_not_gt hr_not_lt
      simpa [hk1] using this

  have hk1_lt : r < neuralBpow binaryRadix (k1 + 1) := by
    by_cases hlt : ratLtPow2 num den k0 = true
    · have hk1 : k1 = k0 - 1 := by simp [k1, hlt]
      have : r < neuralBpow binaryRadix k0 := by
        have := (ratLtPow2_eq_true_iff (num := num) (den := den) (k := k0) hden).1 hlt
        simpa [r] using this
      simpa [hk1, add_assoc] using this
    · have hk1 : k1 = k0 := by simp [k1, hlt]
      simpa [hk1] using hk0_bounds.2

  -- The final `ratGePow2` check is inconsistent with `hk1_lt`, so `floorLog2Rat = k1`.
  have hge_false : ratGePow2 num den (k1 + 1) = false := by
    by_cases hge : ratGePow2 num den (k1 + 1) = true
    · have hr_ge : neuralBpow binaryRadix (k1 + 1) ≤ r :=
        (ratGePow2_eq_true_iff (num := num) (den := den) (k := k1 + 1) hden).1 (by simpa using hge)
      have : False := (not_lt_of_ge hr_ge) hk1_lt
      cases this
    · simpa using hge

  have hk : floorLog2Rat num den = k1 := by
    -- `simp` expands the internal `k1` definition, so first rewrite `hge_false` into the
    -- matching expanded form.
    have hge_false' :
        ratGePow2 num den
            ((if ratLtPow2 num den ((Int.ofNat (Nat.log2 num)) - (Int.ofNat (Nat.log2 den))) = true
              then
                  ((Int.ofNat (Nat.log2 num)) - (Int.ofNat (Nat.log2 den))) - 1
                else
                  (Int.ofNat (Nat.log2 num)) - (Int.ofNat (Nat.log2 den))) +
              1) =
          false := by
      simpa [k0, k1] using hge_false
    -- Unfolding `floorLog2Rat` reduces this to the final `ratGePow2` branch.
    simp [floorLog2Rat, k0, k1]
    simpa using hge_false'

  -- The goal is a `let`; unfold it and substitute `floorLog2Rat num den = k1`.
  simpa [hk, r] using And.intro hk1_ge hk1_lt

/-!
## FP32 refinement for executable rounding

This is the core bridge step: we prove that the executable rounding kernel (which produces an
`IEEE32Exec` value) agrees with `FP32` rounding on reals (`fp32Round`), provided we stay on the
finite/no-overflow path.

Once we have this, most op-level bridge theorems reduce to: “compute an exact dyadic/rational
intermediate, then apply this rounding refinement theorem”.
-/

private lemma neural_nearest_even_eq_zero_of_abs_lt_half (x : ℝ) (hx : _root_.abs x < (1 / 2 : ℝ)) :
    TorchLean.Floats.neuralNearestEven x = 0 := by
  have hx_abs : (- (1 / 2 : ℝ)) < x ∧ x < (1 / 2 : ℝ) := abs_lt.mp hx
  by_cases hx0 : x < 0
  · have hfloor : (⌊x⌋ : ℤ) = -1 := by
      have hx_ge : ((-1 : ℤ) : ℝ) ≤ x := by
        have : (-1 : ℝ) < x := by linarith [hx_abs.1]
        simpa using (le_of_lt this)
      have hx_lt : x < ((-1 : ℤ) : ℝ) + 1 := by
        simpa using hx0
      exact (Int.floor_eq_iff).2 ⟨hx_ge, hx_lt⟩
    have hfrac_gt : x - (⌊x⌋ : ℝ) > (1 / 2 : ℝ) := by
      have : x + 1 > (1 / 2 : ℝ) := by linarith [hx_abs.1]
      simpa [hfloor, sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using this
    have := TorchLean.Floats.neural_nearest_even_eq_ceil_of_frac_gt_half x hfrac_gt
    simpa [hfloor] using this
  · have hx_nonneg : (0 : ℝ) ≤ x := le_of_not_gt hx0
    have hx_lt1 : x < (1 : ℝ) := lt_trans hx_abs.2 (by norm_num)
    have hfloor : (⌊x⌋ : ℤ) = 0 := by
      have : x ∈ Set.Ico (0 : ℝ) 1 := ⟨hx_nonneg, hx_lt1⟩
      exact (Int.floor_eq_zero_iff).2 this
    have hfrac_lt : x - (⌊x⌋ : ℝ) < (1 / 2 : ℝ) := by
      simpa [hfloor] using hx_abs.2
    have := TorchLean.Floats.neural_nearest_even_eq_floor_of_frac_lt_half x hfrac_lt
    simpa [hfloor] using this

/--
A coarse magnitude bound for a decoded dyadic.

Informal: `|mant * 2^exp| < 2^(log2 mant + exp + 1)`. This is convenient when reasoning about
normalization by `log2` and when relating dyadic magnitudes to exponent ranges.
-/
theorem abs_dyadicToReal_lt_bpow_succ_log2 (d : Dyadic) :
    _root_.abs (dyadicToReal d) <
      neuralBpow binaryRadix (Int.ofNat (Nat.log2 d.mant) + d.exp + 1) := by
  -- Bound `mant` by the next power of two above it, then scale by `2^exp`.
  set l : Nat := Nat.log 2 d.mant
  have hl : Nat.log2 d.mant = l := by
    simpa [l] using (Nat.log2_eq_log_two (n := d.mant))
  have hmant_nat : d.mant < 2 ^ l.succ :=
    Nat.lt_pow_succ_log_self (b := 2) (hb := Nat.one_lt_two) d.mant
  have hmant : (d.mant : ℝ) < ((2 ^ l.succ : Nat) : ℝ) := by
    exact_mod_cast hmant_nat
  have hbpos : 0 < neuralBpow binaryRadix d.exp :=
    neuralBpow.pos binaryRadix d.exp
  have hmul : (d.mant : ℝ) * neuralBpow binaryRadix d.exp < ((2 ^ l.succ : Nat) : ℝ) * neuralBpow
    binaryRadix d.exp :=
    (mul_lt_mul_of_pos_right hmant hbpos)
  -- Rewrite into the desired `bpow` bound.
  have hpow : neuralBpow binaryRadix (Int.ofNat l.succ) = ((2 ^ l.succ : Nat) : ℝ) := by
    calc
      neuralBpow binaryRadix (Int.ofNat l.succ)
          = (2 : ℝ) ^ (Int.ofNat l.succ) := by
              simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
      _ = (2 : ℝ) ^ (l.succ : Nat) := by
              simpa using (zpow_ofNat (2 : ℝ) l.succ)
      _ = ((2 ^ l.succ : Nat) : ℝ) := by
              simp
  -- `abs (dyadicToReal d) = mant * 2^exp`.
  have habs : _root_.abs (dyadicToReal d) = (d.mant : ℝ) * neuralBpow binaryRadix d.exp := by
    simpa using (abs_dyadicToReal d)
  -- Finish.
  have : _root_.abs (dyadicToReal d) < neuralBpow binaryRadix (Int.ofNat l.succ + d.exp) := by
    -- Start from the mantissa bound, then combine powers of two.
    have hmul' :
        (d.mant : ℝ) * neuralBpow binaryRadix d.exp <
          neuralBpow binaryRadix (Int.ofNat l.succ) * neuralBpow binaryRadix d.exp := by
      simpa [hpow.symm] using hmul
    have hmul'' :
        (d.mant : ℝ) * neuralBpow binaryRadix d.exp < neuralBpow binaryRadix ((l : Int) + 1 +
          d.exp) := by
      simpa [(neuralBpow.add_exp binaryRadix ((l : Int) + 1) d.exp).symm, add_assoc] using hmul'
    simpa [habs] using hmul''
  -- Rewrite `l` back to `log2 mant` and expand `succ` as `+1`.
  -- `Int.ofNat l.succ = Int.ofNat l + 1`.
  simpa [hl, Int.natCast_succ, add_assoc, add_left_comm, add_comm] using this

/-! ### Signed zeros -/

/--
Both `+0` and `-0` decode to the real number `0`.

IEEE-754 has signed zeros because they matter for some operations (notably division and some
transcendentals). Our finite `FP32` model treats them as equal at the real level, and the bridge
lemmas in this file use this fact repeatedly.
-/
theorem toReal_signedZero (s : Bool) : toReal (if s then negZero else posZero) = 0 := by
  cases s
  · -- +0
    have hbits : (0 : UInt32) = mkBits false 0 0 := by decide
    simp [posZero, hbits, toReal_eq,
      toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := 0)
        (hexp := (by decide : (0 : Nat) < 255)) (hfrac := (by decide : (0 : Nat) < 2 ^ 23)),
      dyadicToReal, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
  · -- -0
    have hbits : signMask = mkBits true 0 0 := by decide
    simp [negZero, hbits, toReal_eq,
      toDyadic?_ofBits_mkBits_fin (sign := true) (exp := 0) (frac := 0)
        (hexp := (by decide : (0 : Nat) < 255)) (hfrac := (by decide : (0 : Nat) < 2 ^ 23)),
      dyadicToReal, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]

/-- `+0` decodes to the real number `0`. -/
@[simp] theorem toReal_posZero : toReal (posZero : IEEE32Exec) = 0 := by
  simpa using (toReal_signedZero (s := false))

/-- `-0` decodes to the real number `0`. -/
@[simp] theorem toReal_negZero : toReal (negZero : IEEE32Exec) = 0 := by
  simpa using (toReal_signedZero (s := true))

/--
Refinement theorem (finite/no-overflow): rounding an exact dyadic with the executable IEEE32 kernel
agrees with the Flocq-style `FP32` rounding-on-`ℝ` model.

The hypothesis `isFinite (roundDyadicToIEEE32 d) = true` rules out the overflow-to-`±Inf` branches.
-/
theorem toReal_roundDyadicToIEEE32_eq_fp32Round (d : Dyadic)
    (hfin : isFinite (roundDyadicToIEEE32 d) = true) :
    toReal (roundDyadicToIEEE32 d) = fp32Round (dyadicToReal d) := by
  classical
  by_cases hm : d.mant = 0
  · -- Both sides are real `0`.
    have hto : toReal (roundDyadicToIEEE32 d) = 0 := by
      have hround0 : roundDyadicToIEEE32 d = (if d.sign then negZero else posZero) := by
        simp [roundDyadicToIEEE32, hm]
      rw [hround0]
      simpa using (toReal_signedZero d.sign)
    have hfp : fp32Round (dyadicToReal d) = 0 := by
      -- `dyadicToReal d = 0` when `mant = 0`.
      simp [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal,
        TorchLean.Floats.neuralScaledMantissa, TorchLean.Floats.neuralCexp,
          TorchLean.Floats.neuralMagnitude,
        TorchLean.Floats.fexp32, TorchLean.Floats.FLTExp, TorchLean.Floats.rnd32, dyadicToReal, hm,
        TorchLean.Floats.neuralNearestEven, TorchLean.Floats.neuralBpow, binaryRadix,
          NeuralRadix.toReal]
    rw [hto, hfp]
  · have hmbeq : (d.mant == 0) = false := (beq_eq_false_iff_ne).2 hm
    set log2m : Nat := Nat.log2 d.mant
    set k : Int := (Int.ofNat log2m) + d.exp
    -- Eliminate the overflow-to-Inf branch.
    by_cases hkHi : k > 127
    · have hlogdef : Nat.log2 d.mant = log2m := by
        simp [log2m]
      have hkdef : (Int.ofNat log2m) + d.exp = k := by
        simp [k]
      have hround : roundDyadicToIEEE32 d = (if d.sign then negInf else posInf) := by
        -- `simp` reduces to the `k ≤ 127` branch; discharge it by contradiction with `hkHi`.
        simp [roundDyadicToIEEE32, hmbeq, hlogdef]
        intro hkLe
        have hkLe' : k ≤ 127 := by
          simpa [hkdef] using hkLe
        exact (False.elim ((not_lt_of_ge hkLe') hkHi))
      have hfalse : isFinite (roundDyadicToIEEE32 d) = false := by
        rw [hround]
        cases d.sign <;> decide
      cases (hfalse.symm.trans hfin)
    · -- Non-overflowing exponent range: `k ≤ 127`.
      by_cases hkUnder : k < -150
      · -- Underflow-to-zero: show `FP32` rounding also yields `0`.
        have hround : roundDyadicToIEEE32 d = (if d.sign then negZero else posZero) := by
          have hkHi0 : ¬(127 < Int.ofNat (Nat.log2 d.mant) + d.exp) := by
            simpa [k, log2m] using hkHi
          have hkUnder0 : Int.ofNat (Nat.log2 d.mant) + d.exp < -150 := by
            simpa [k, log2m] using hkUnder
          -- Coercions print as `↑`; rewriting matches on that form.
          have hkHi' : ¬(127 < (d.mant.log2 : Int) + d.exp) := by
            simpa using hkHi0
          have hkUnder' : (d.mant.log2 : Int) + d.exp < -150 := by
            simpa using hkUnder0
          -- Unfold the executable rounding and take the underflow branch.
          simp (config := { zeta := true }) [roundDyadicToIEEE32, hmbeq]
          rw [if_neg hkHi']
          rw [if_pos hkUnder']
        have hto : toReal (roundDyadicToIEEE32 d) = 0 := by
          simpa [hround] using (toReal_signedZero d.sign)
        -- Compute the FP32 rounding exponent and show the scaled mantissa is within `1/2`.
        have hAbsBpow : _root_.abs (dyadicToReal d) < neuralBpow binaryRadix (k + 1) := by
          simpa [k, log2m, add_assoc] using (abs_dyadicToReal_lt_bpow_succ_log2 d)
        have hk1_le : k + 1 ≤ (-150 : Int) := by
          -- `k` is an int, so `< -150` means `≤ -151`.
          linarith
        have hBpow_le :
            neuralBpow binaryRadix (k + 1) ≤ neuralBpow binaryRadix (-150 : Int) := by
          -- Monotonicity of `zpow` for base `2` (in a `GroupWithZero`).
          simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
          exact zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2) hk1_le
        have hAbs150 : _root_.abs (dyadicToReal d) < neuralBpow binaryRadix (-150 : Int) :=
          lt_of_lt_of_le hAbsBpow hBpow_le
        have hbpos149 : 0 < neuralBpow binaryRadix (149 : Int) := neuralBpow.pos binaryRadix 149
        have hAbsScaled :
            _root_.abs (dyadicToReal d * neuralBpow binaryRadix (149 : Int)) < (1 / 2 : ℝ) := by
          -- `abs (x * 2^149) = abs x * 2^149`.
          have habs_mul :
              _root_.abs (dyadicToReal d * neuralBpow binaryRadix (149 : Int)) =
                _root_.abs (dyadicToReal d) * neuralBpow binaryRadix (149 : Int) := by
            have hnonneg : 0 ≤ neuralBpow binaryRadix (149 : Int) :=
              le_of_lt hbpos149
            simp [abs_mul, abs_of_nonneg hnonneg]
          have hmul :
              _root_.abs (dyadicToReal d) * neuralBpow binaryRadix (149 : Int) <
                neuralBpow binaryRadix (-150 : Int) * neuralBpow binaryRadix (149 : Int) :=
            (mul_lt_mul_of_pos_right hAbs150 hbpos149)
          -- `2^-150 * 2^149 = 2^-1 = 1/2`.
          have hprod :
              neuralBpow binaryRadix (-150 : Int) * neuralBpow binaryRadix (149 : Int) = (1 / 2
                : ℝ) := by
            -- combine exponents and unfold `bpow` at base 2
            have := (neuralBpow.add_exp binaryRadix (-150 : Int) (149 : Int))
            -- `bpow (-1) = 1/2`
            -- `simp` takes care of `2^(-1)`.
            simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using this.symm
          -- finish
          simpa [habs_mul, hprod, mul_assoc] using hmul
        have hRnd0 :
            TorchLean.Floats.neuralNearestEven (dyadicToReal d * neuralBpow binaryRadix (149 :
              Int)) = 0 :=
          neural_nearest_even_eq_zero_of_abs_lt_half _ hAbsScaled
        have hfp : fp32Round (dyadicToReal d) = 0 := by
          -- Unfold FP32 rounding and rewrite by `cexp = -149`.
          -- Under `k < -150`, the format exponent is `-149` and the mantissa rounds to `0`.
          have hcexp : TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32
            (dyadicToReal d) = (-149 : Int) := by
            -- `neural_cexp = fexp32 (neural_magnitude)` and `k` is below normal range.
            have hmag :
                TorchLean.Floats.neuralMagnitude binaryRadix (dyadicToReal d) = k + 1 := by
              have hmag0 := neural_magnitude_dyadic (d := d) hm
              -- rewrite `Nat.log 2` to `Nat.log2`
              have hlog : Nat.log 2 d.mant = Nat.log2 d.mant := (Nat.log2_eq_log_two (n :=
                d.mant)).symm
              simpa [k, log2m, hlog] using hmag0
            have hk_le : k + 1 ≤ (-125 : Int) := by linarith [hkUnder]
            have hk_le' : k + 1 - 24 ≤ (-149 : Int) := by linarith [hk_le]
            -- unfold and simplify the `max`
            simp [TorchLean.Floats.neuralCexp, TorchLean.Floats.fexp32, TorchLean.Floats.FLTExp,
              hmag, max_eq_right hk_le']
          have hscaled :
              TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32
                (dyadicToReal d) =
                dyadicToReal d * neuralBpow binaryRadix (149 : Int) := by
            -- `scaled_mantissa = x * bpow (-cexp)` and `cexp = -149`.
            simp [TorchLean.Floats.neuralScaledMantissa, hcexp]
          have hmant :
              TorchLean.Floats.rnd32
                  (TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32
                    (dyadicToReal d)) =
                0 := by
            simpa [TorchLean.Floats.rnd32, hscaled] using hRnd0
          -- Now compute `fp32Round`: mantissa is `0`, so the result is `0`.
          simp [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal, hmant,
            hcexp]
        rw [hto, hfp]
      · -- Remaining cases (`k ≥ -150`): subnormal or normal rounding.
        -- Remaining cases (`k ≥ -150`): split into the subnormal and normal rounding regimes.
        by_cases hkSub : k < -126
        · -- Subnormal (possibly rounding up to the smallest normal).
          -- Define the exact subnormal mantissa computed by the executable kernel.
          set fracNat : Nat :=
            match d.exp + 149 with
            | .ofNat sh => Nat.shiftLeft d.mant sh
            | .negSucc sh => roundShiftRightEven d.mant (sh + 1)
          have hround :
              roundDyadicToIEEE32 d =
                if fracNat == 0 then
                  (if d.sign then negZero else posZero)
                else
                  match Nat.decLe (pow2 23) fracNat with
                  | isTrue _ => ofBits (mkBits d.sign 1 0)
                  | isFalse _ => ofBits (mkBits d.sign 0 fracNat) := by
            -- Unfold and align with `fracNat`.
            have hlogdef : Nat.log2 d.mant = log2m := by
              simp [log2m]
            have hkdef0 : Int.ofNat log2m + d.exp = k := by
              simp [k]
            have hkdef : (log2m : Int) + d.exp = k := by
              simpa using hkdef0
            simp (config := { zeta := true }) [roundDyadicToIEEE32, hmbeq, hlogdef, hkdef, hkHi,
              hkUnder, hkSub, fracNat]
            rfl
          -- FP32 rounding uses exponent `-149` in the entire subnormal range.
          have hcexp : TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32
            (dyadicToReal d) = (-149 : Int) := by
            have hmag :
                TorchLean.Floats.neuralMagnitude binaryRadix (dyadicToReal d) = k + 1 := by
              have hmag0 := neural_magnitude_dyadic (d := d) hm
              have hlog : Nat.log 2 d.mant = Nat.log2 d.mant := (Nat.log2_eq_log_two (n :=
                d.mant)).symm
              simpa [k, log2m, hlog] using hmag0
            have hk_le : k + 1 ≤ (-125 : Int) := by linarith [hkSub]
            have hk_le' : k + 1 - 24 ≤ (-149 : Int) := by linarith [hk_le]
            simp [TorchLean.Floats.neuralCexp, TorchLean.Floats.fexp32, TorchLean.Floats.FLTExp,
              hmag, max_eq_right hk_le']
          -- Compute the rounded scaled mantissa in FP32 (as an integer), matching `fracNat`.
          have hRndFrac :
              TorchLean.Floats.neuralNearestEven (dyadicToReal d * neuralBpow binaryRadix (149 :
                Int)) =
                if d.sign then -Int.ofNat fracNat else Int.ofNat fracNat := by
            -- Split on the sign bit.
            cases hs : d.sign
            · -- positive
              simp []
              have harg :
                  dyadicToReal d * neuralBpow binaryRadix (149 : Int) =
                    (d.mant : ℝ) * neuralBpow binaryRadix (d.exp + 149) := by
                -- combine the dyadic exponent with `149`
                have hb :
                    neuralBpow binaryRadix d.exp * neuralBpow binaryRadix (149 : Int) =
                      neuralBpow binaryRadix (d.exp + 149) := by
                  simpa using (neuralBpow.add_exp binaryRadix d.exp (149 : Int)).symm
                simp [dyadicToReal, hs, hb, mul_assoc, mul_comm]
              -- Now decide based on `d.exp + 149`.
              cases hshift : d.exp + 149 with
              | ofNat sh =>
                  -- integer case: rounding is exact
                  have hargeq :
                      (d.mant : ℝ) * neuralBpow binaryRadix (sh : Int) =
                        ((Nat.shiftLeft d.mant sh : Nat) : ℝ) := by
                    -- `bpow sh = 2^sh` and `mant * 2^sh = mant <<< sh`
                    simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                      Nat.shiftLeft_eq, Nat.cast_mul,
                      Nat.cast_pow]
                  have hroundInt :
                      TorchLean.Floats.neuralNearestEven ((Nat.shiftLeft d.mant sh : Nat) : ℝ) =
                        Int.ofNat (Nat.shiftLeft d.mant sh) := by
                    simpa using
                      (TorchLean.Floats.NeuralValidRnd.id (rnd :=
                        TorchLean.Floats.neuralNearestEven)
                        (Int.ofNat (Nat.shiftLeft d.mant sh)))
                  -- rewrite `fracNat` and discharge
                  have : TorchLean.Floats.neuralNearestEven (dyadicToReal d * neuralBpow
                    binaryRadix (149 : Int)) =
                      Int.ofNat fracNat := by
                    -- `harg` gives the scaled-mantissa form; `hshift` picks the `shiftLeft` branch.
                    simpa [harg, hshift, fracNat, hargeq] using hroundInt
                  simpa [hs] using this
              | negSucc sh =>
                  -- rational case: connect to `roundShiftRightEven`
                  have hargeq :
                      (d.mant : ℝ) * neuralBpow binaryRadix (Int.negSucc sh) =
                        (d.mant : ℝ) / (pow2 (sh + 1) : ℝ) := by
                    -- `bpow (-(sh+1)) = (2^(sh+1))⁻¹`
                    simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                      pow2_eq_two_pow,
                      div_eq_mul_inv]
                  have hroundRat :=
                    neural_nearest_even_div_pow2_eq_roundShiftRightEven (num := d.mant) (shift := sh
                      + 1)
                  have : TorchLean.Floats.neuralNearestEven (dyadicToReal d * neuralBpow
                    binaryRadix (149 : Int)) =
                      Int.ofNat fracNat := by
                    -- `harg` gives the scaled-mantissa form; `hshift` picks the
                    -- `roundShiftRightEven` branch.
                    simpa [harg, hshift, fracNat, hargeq] using hroundRat
                  simpa [hs] using this
            · -- negative
              -- reduce to the positive case using oddness of nearest-even
              have hodd :=
                neural_nearest_even_neg
                  (x := dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                    neuralBpow binaryRadix (149 : Int))
              have hx :
                  dyadicToReal d * neuralBpow binaryRadix (149 : Int) =
                    -(dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                      neuralBpow binaryRadix (149 : Int)) := by
                simp [dyadicToReal, hs, mul_assoc, mul_comm]
              -- Use the `sign=false` computation (the first branch) and then negate.
              have hpos :
                  TorchLean.Floats.neuralNearestEven
                      (dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                        neuralBpow binaryRadix (149 : Int)) =
                    Int.ofNat fracNat := by
                have harg0 :
                    dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                        neuralBpow binaryRadix (149 : Int) =
                      (d.mant : ℝ) * neuralBpow binaryRadix (d.exp + 149) := by
                  have hb :
                      neuralBpow binaryRadix d.exp * neuralBpow binaryRadix (149 : Int) =
                        neuralBpow binaryRadix (d.exp + 149) := by
                    simpa using (neuralBpow.add_exp binaryRadix d.exp (149 : Int)).symm
                  simp [dyadicToReal, hb, mul_assoc, mul_comm]
                cases hshift : d.exp + 149 with
                | ofNat sh =>
                    have hargeq :
                        (d.mant : ℝ) * neuralBpow binaryRadix (sh : Int) =
                          ((Nat.shiftLeft d.mant sh : Nat) : ℝ) := by
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                        Nat.shiftLeft_eq, Nat.cast_mul,
                        Nat.cast_pow]
                    have hroundInt :
                        TorchLean.Floats.neuralNearestEven ((Nat.shiftLeft d.mant sh : Nat) : ℝ) =
                          Int.ofNat (Nat.shiftLeft d.mant sh) := by
                      simpa using
                        (TorchLean.Floats.NeuralValidRnd.id (rnd :=
                          TorchLean.Floats.neuralNearestEven)
                          (Int.ofNat (Nat.shiftLeft d.mant sh)))
                    simpa [harg0, hshift, fracNat, hargeq] using hroundInt
                | negSucc sh =>
                    have hargeq :
                        (d.mant : ℝ) * neuralBpow binaryRadix (Int.negSucc sh) =
                          (d.mant : ℝ) / (pow2 (sh + 1) : ℝ) := by
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                        pow2_eq_two_pow,
                        div_eq_mul_inv]
                    have hroundRat :=
                      neural_nearest_even_div_pow2_eq_roundShiftRightEven (num := d.mant) (shift :=
                        sh + 1)
                    simpa [harg0, hshift, fracNat, hargeq] using hroundRat
              -- Negate using `neural_nearest_even_neg`.
              have : TorchLean.Floats.neuralNearestEven (dyadicToReal d * neuralBpow binaryRadix
                (149 : Int)) =
                  -Int.ofNat fracNat := by
                simpa [hx, hpos] using hodd
              simpa [hs] using this
          -- Show the executable output has the same real value as FP32 rounding.
          -- First, normalize the executable branch to a single real expression.
          have hto :
              toReal (roundDyadicToIEEE32 d) =
                if fracNat == 0 then 0 else
                  (if d.sign then (-1 : ℝ) else (1 : ℝ)) * (fracNat : ℝ) * neuralBpow binaryRadix
                    (-149 : Int) := by
            -- Expand the executable definition.
            rw [hround]
            by_cases hF0 : fracNat = 0
            · -- `fracNat = 0` ⇒ executable output is signed zero.
              simpa [hF0, toReal_eq] using (toReal_signedZero d.sign)
            · have hF0b : (fracNat == 0) = false := (beq_eq_false_iff_ne).2 hF0
              -- Split on the `pow2 23 ≤ fracNat` decision.
              cases hdec : Nat.decLe (pow2 23) fracNat with
              | isTrue hle =>
                  -- In the subnormal range, `fracNat` cannot exceed `2^23`, so this is the tie
                  -- case.
                  have hle' : fracNat ≤ pow2 23 := by
                    -- Bound the scaled mantissa: `|x| < 2^-126` ⇒ `|x|*2^149 < 2^23`.
                    have hAbsBpow : _root_.abs (dyadicToReal d) < neuralBpow binaryRadix (k + 1)
                      := by
                      simpa [k, log2m, add_assoc] using (abs_dyadicToReal_lt_bpow_succ_log2 d)
                    have hk1_le : k + 1 ≤ (-126 : Int) := by linarith [hkSub]
                    have hBpow_le :
                        neuralBpow binaryRadix (k + 1) ≤ neuralBpow binaryRadix (-126 : Int) :=
                          by
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                      exact zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2) hk1_le
                    have hAbs126 : _root_.abs (dyadicToReal d) < neuralBpow binaryRadix (-126 :
                      Int) :=
                      lt_of_lt_of_le hAbsBpow hBpow_le
                    have hbpos149 : 0 < neuralBpow binaryRadix (149 : Int) := neuralBpow.pos
                      binaryRadix 149
                    have hAbsScaled :
                        _root_.abs (dyadicToReal d * neuralBpow binaryRadix (149 : Int)) < (pow2
                          23 : ℝ) := by
                      have habs_mul :
                          _root_.abs (dyadicToReal d * neuralBpow binaryRadix (149 : Int)) =
                            _root_.abs (dyadicToReal d) * neuralBpow binaryRadix (149 : Int) := by
                        have hnonneg : 0 ≤ neuralBpow binaryRadix (149 : Int) := le_of_lt hbpos149
                        simp [abs_mul, abs_of_nonneg hnonneg]
                      have hmul :
                          _root_.abs (dyadicToReal d) * neuralBpow binaryRadix (149 : Int) <
                            neuralBpow binaryRadix (-126 : Int) * neuralBpow binaryRadix (149 :
                              Int) :=
                        (mul_lt_mul_of_pos_right hAbs126 hbpos149)
                      have hprod :
                          neuralBpow binaryRadix (-126 : Int) * neuralBpow binaryRadix (149 :
                            Int) = (pow2 23 : ℝ) := by
                        have hsum : (-126 : Int) + 149 = 23 := by norm_num
                        have hmul :
                            neuralBpow binaryRadix (-126 : Int) * neuralBpow binaryRadix (149 :
                              Int) =
                              neuralBpow binaryRadix (23 : Int) := by
                          simpa [hsum] using (neuralBpow.add_exp binaryRadix (-126 : Int) (149 :
                            Int)).symm
                        have h23 : neuralBpow binaryRadix (23 : Int) = (pow2 23 : ℝ) := by
                          simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                            pow2_eq_two_pow]
                          norm_num
                        exact hmul.trans h23
                      simpa [habs_mul, hprod, mul_assoc] using hmul
                    -- `neural_nearest_even` of a value `< 2^23` is `≤ 2^23`.
                    have hbound :
                        TorchLean.Floats.neuralNearestEven (_root_.abs (dyadicToReal d *
                          neuralBpow binaryRadix (149 : Int))) ≤
                          Int.ofNat (pow2 23) := by
                      -- Use bounds: `neural_nearest_even t ≤ ⌊t⌋ + 1`, and `t < 2^23`.
                      have hbnds := TorchLean.Floats.neural_nearest_even_bounds (_root_.abs
                        (dyadicToReal d * neuralBpow binaryRadix (149 : Int)))
                      have hfloor_lt : (⌊_root_.abs (dyadicToReal d * neuralBpow binaryRadix (149
                        : Int))⌋ : Int) < Int.ofNat (pow2 23) := by
                        -- `floor t < N` when `t < N` and `N` is an integer.
                        have : _root_.abs (dyadicToReal d * neuralBpow binaryRadix (149 : Int)) <
                          (pow2 23 : ℝ) := hAbsScaled
                        have : (⌊_root_.abs (dyadicToReal d * neuralBpow binaryRadix (149 : Int))⌋
                          : Int) < Int.ofNat (pow2 23) := by
                          exact Int.floor_lt.2 (by simpa using this)
                        exact this
                      have hfloor_le : (⌊_root_.abs (dyadicToReal d * neuralBpow binaryRadix (149
                        : Int))⌋ : Int) + 1 ≤ Int.ofNat (pow2 23) := by
                        exact Int.add_one_le_iff.mpr hfloor_lt
                      exact le_trans hbnds.2 hfloor_le
                    -- Relate `fracNat` to this bound via `hRndFrac` and sign of the dyadic.
                    let scaled : ℝ := dyadicToReal d * neuralBpow binaryRadix (149 : Int)
                    have hroundAbs : TorchLean.Floats.neuralNearestEven (_root_.abs scaled) =
                      Int.ofNat fracNat := by
                      cases hs : d.sign
                      · -- `scaled ≥ 0`, so `abs scaled = scaled`.
                        have hdy_nonneg : 0 ≤ dyadicToReal d := by
                          have hmant_nonneg : 0 ≤ (d.mant : ℝ) := Nat.cast_nonneg _
                          have hbexp_nonneg : 0 ≤ neuralBpow binaryRadix d.exp :=
                            neuralBpow.nonneg binaryRadix d.exp
                          simp [dyadicToReal, hs, mul_nonneg, hmant_nonneg, hbexp_nonneg]
                        have hb149_nonneg : 0 ≤ neuralBpow binaryRadix (149 : Int) := le_of_lt
                          hbpos149
                        have hscaled_nonneg : 0 ≤ scaled := by
                          simpa [scaled] using mul_nonneg hdy_nonneg hb149_nonneg
                        have habs : _root_.abs scaled = scaled := abs_of_nonneg hscaled_nonneg
                        have hscaled_round :
                            TorchLean.Floats.neuralNearestEven scaled = Int.ofNat fracNat := by
                          simpa [scaled, hs] using hRndFrac
                        simpa [habs] using hscaled_round
                      · -- `scaled ≤ 0`, so `abs scaled = -scaled` and oddness flips the sign.
                        have hdy_nonpos : dyadicToReal d ≤ 0 := by
                          have hmant_nonneg : 0 ≤ (d.mant : ℝ) := Nat.cast_nonneg _
                          have hbexp_nonneg : 0 ≤ neuralBpow binaryRadix d.exp :=
                            neuralBpow.nonneg binaryRadix d.exp
                          have hpos : 0 ≤ (d.mant : ℝ) * neuralBpow binaryRadix d.exp :=
                            mul_nonneg hmant_nonneg hbexp_nonneg
                          simpa [dyadicToReal, hs, mul_assoc] using (neg_nonpos.2 hpos)
                        have hb149_nonneg : 0 ≤ neuralBpow binaryRadix (149 : Int) := le_of_lt
                          hbpos149
                        have hscaled_nonpos : scaled ≤ 0 := by
                          simpa [scaled] using mul_nonpos_of_nonpos_of_nonneg hdy_nonpos
                            hb149_nonneg
                        have habs : _root_.abs scaled = -scaled := abs_of_nonpos hscaled_nonpos
                        have hscaled_round :
                            TorchLean.Floats.neuralNearestEven scaled = -Int.ofNat fracNat := by
                          simpa [scaled, hs] using hRndFrac
                        have hneg :
                            TorchLean.Floats.neuralNearestEven (-scaled) =
                              -TorchLean.Floats.neuralNearestEven scaled := by
                          simpa using (neural_nearest_even_neg (x := scaled))
                        calc
                          TorchLean.Floats.neuralNearestEven (_root_.abs scaled)
                              = TorchLean.Floats.neuralNearestEven (-scaled) := by simp [habs]
                          _ = -TorchLean.Floats.neuralNearestEven scaled := hneg
                          _ = Int.ofNat fracNat := by simp [hscaled_round]
                    have hfracNat_le : Int.ofNat fracNat ≤ Int.ofNat (pow2 23) := by
                      -- rewrite `scaled` back into `hbound` and then use `hroundAbs`.
                      have : TorchLean.Floats.neuralNearestEven (_root_.abs scaled) ≤ Int.ofNat
                        (pow2 23) := by
                        simpa [scaled] using hbound
                      simpa [hroundAbs] using this
                    exact (Int.ofNat_le).1 hfracNat_le
                  have hEq : fracNat = pow2 23 := le_antisymm hle' hle
                  -- Output is the smallest normal; its dyadic real value is `2^23 * 2^-149`.
                  have hp23 : pow2 23 ≠ 0 := by
                    simp [pow2_eq_two_pow]
                  have hexp1 : (1 : Nat) < 255 := by decide
                  have hfrac0 : (0 : Nat) < 2 ^ 23 := by decide
                  simp [hEq, toReal_eq,
                    toDyadic?_ofBits_mkBits_fin (sign := d.sign) (exp := 1) (frac := 0) (hexp :=
                      hexp1)
                      (hfrac := hfrac0),
                    dyadicToReal, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                      hp23]
              | isFalse hlt =>
                  -- True subnormal: decode `mkBits sign 0 fracNat`.
                  have hlt' : fracNat < 2 ^ 23 := by
                    -- `fracNat < pow2 23` from the negation.
                    simpa [pow2_eq_two_pow] using (Nat.lt_of_not_ge hlt)
                  simp [toReal_eq, toDyadic?_ofBits_mkBits_fin (sign := d.sign) (exp := 0) (frac :=
                    fracNat)
                    (hexp := (by decide : (0 : Nat) < 255)) (hfrac := hlt'),
                    dyadicToReal, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                      hF0b, hF0]
          -- FP32 result is `fracNat * 2^-149` with the correct sign.
          have hfp :
              fp32Round (dyadicToReal d) =
                (if d.sign then (-1 : ℝ) else (1 : ℝ)) * (fracNat : ℝ) * neuralBpow binaryRadix
                  (-149 : Int) := by
            -- Use the computed mantissa rounding and exponent `-149`.
            cases hs : d.sign
            · -- positive
              -- Unfold `fp32Round`; `simp` reduces the goal to a cancellation fact.
              simp [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal,
                TorchLean.Floats.neuralScaledMantissa, TorchLean.Floats.rnd32, hcexp,
                mul_comm]
              left
              have h0 :
                  TorchLean.Floats.neuralNearestEven
                      (dyadicToReal d * neuralBpow binaryRadix (149 : Int)) =
                    Int.ofNat fracNat := by
                simpa [hs] using hRndFrac
              have h1 :
                  TorchLean.Floats.neuralNearestEven
                      (neuralBpow binaryRadix (149 : Int) * dyadicToReal d) =
                    Int.ofNat fracNat := by
                simpa [mul_comm] using h0
              -- Cast `Int.ofNat` to `ℝ`.
              simpa using congrArg (fun z : Int => (z : ℝ)) h1
            · -- negative
              simp [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal,
                TorchLean.Floats.neuralScaledMantissa, TorchLean.Floats.rnd32, hcexp,
                mul_comm]
              have h0 :
                  TorchLean.Floats.neuralNearestEven
                      (dyadicToReal d * neuralBpow binaryRadix (149 : Int)) =
                    -Int.ofNat fracNat := by
                simpa [hs] using hRndFrac
              have h1 :
                  TorchLean.Floats.neuralNearestEven
                      (neuralBpow binaryRadix (149 : Int) * dyadicToReal d) =
                    -Int.ofNat fracNat := by
                simpa [mul_comm] using h0
              have h1' :
                  ((TorchLean.Floats.neuralNearestEven
                      (neuralBpow binaryRadix (149 : Int) * dyadicToReal d)) : ℝ) =
                    -(fracNat : ℝ) := by
                simpa using congrArg (fun z : Int => (z : ℝ)) h1
              -- Push the minus sign out.
              simp [h1']
          -- Combine.
          by_cases hF0 : fracNat = 0
          · have hF0b : (fracNat == 0) = true := by simp [hF0]
            rw [hto, hfp]
            simp [hF0]
          · have hF0b : (fracNat == 0) = false := (beq_eq_false_iff_ne).2 hF0
            rw [hto, hfp]
            simp [hF0b]
        · -- Normal rounding.
            -- Mirror the executable kernel definitions.
            set m24 : Nat :=
              if log2m >= 23 then
                roundShiftRightEven d.mant (log2m - 23)
              else
                Nat.shiftLeft d.mant (23 - log2m)
            set k' : Int := if m24 == pow2 24 then k + 1 else k
            set m24' : Nat := if m24 == pow2 24 then pow2 23 else m24
            have hround :
                roundDyadicToIEEE32 d =
                  if k' > 127 then
                    (if d.sign then negInf else posInf)
                  else
                    let expNat : Nat := Int.toNat (k' + 127)
                    let fracNat : Nat := m24' - pow2 23
                    ofBits (mkBits d.sign expNat fracNat) := by
              have hlogdef : Nat.log2 d.mant = log2m := by
                simp [log2m]
              have hkdef0 : (Int.ofNat log2m) + d.exp = k := by
                simp [k]
              have hkdef : (log2m : Int) + d.exp = k := by
                simpa using hkdef0
              simp (config := { zeta := true }) [roundDyadicToIEEE32, hmbeq, hlogdef, hkdef, hkHi,
                hkUnder, hkSub, m24,
                k', m24']
            -- If the carry-adjusted exponent overflows, the result is `±Inf`, contradicting `hfin`.
            by_cases hk'Hi : k' > 127
            · have hInf : roundDyadicToIEEE32 d = (if d.sign then negInf else posInf) := by
                simp [hround, hk'Hi]
              have hfalse : isFinite (roundDyadicToIEEE32 d) = false := by
                rw [hInf]
                cases d.sign <;> decide
              cases (hfalse.symm.trans hfin)
            · have hk'le : k' ≤ 127 := le_of_not_gt hk'Hi
              -- Name the fields for decoding.
              set expNat : Nat := Int.toNat (k' + 127)
              set fracNat : Nat := m24' - pow2 23
              have hroundBits :
                  roundDyadicToIEEE32 d = ofBits (mkBits d.sign expNat fracNat) := by
                simp [hround, hk'Hi, expNat, fracNat]

              -- `k'` is still in the normal range on the low end (`k ≥ -126`).
              have hk'ge : (-126 : Int) ≤ k' := by
                have hkge : (-126 : Int) ≤ k := (not_lt).1 hkSub
                by_cases hcarry : m24 == pow2 24
                · simp [k', hcarry]
                  linarith [hkge]
                · simpa [k', hcarry] using hkge
              have hk'exp_nonneg : 0 ≤ k' + 127 := by linarith [hk'ge]
              have hk'exp_lt : k' + 127 < (255 : Int) := by
                -- From `k' ≤ 127`, we have `k' + 127 ≤ 254 < 255`.
                linarith [hk'le]
              have hexp : expNat < 255 := by
                have h :=
                  (Int.toNat_lt_of_ne_zero (m := k' + 127) (n := 255) (by decide)).2 hk'exp_lt
                simpa [expNat] using h

              -- Bound the 24-bit mantissa (needed to show `fracNat < 2^23`).
              have hmant_lt : d.mant < 2 ^ log2m.succ := by
                -- `log2m = log2 d.mant` and `mant ≠ 0`.
                have h0 : d.mant ≠ 0 := hm
                have hl : Nat.log2 d.mant = Nat.log 2 d.mant := Nat.log2_eq_log_two (n := d.mant)
                -- `lt_pow_succ_log_self` is stated using `Nat.log`.
                simpa [log2m, hl] using
                  (Nat.lt_pow_succ_log_self (b := 2) (hb := Nat.one_lt_two) d.mant)

              have hpow_le : 2 ^ log2m ≤ d.mant := by
                have h0 : d.mant ≠ 0 := hm
                have hlog : Nat.log 2 d.mant = Nat.log2 d.mant := by
                  simpa using (Nat.log2_eq_log_two (n := d.mant)).symm
                have hpow' : 2 ^ (Nat.log2 d.mant) ≤ d.mant := by
                  simpa [hlog] using (Nat.pow_log_le_self 2 (x := d.mant) h0)
                have hlogdef : Nat.log2 d.mant = log2m := by
                  simp [log2m]
                simpa [hlogdef] using hpow'

              have hm24_ge : 2 ^ 23 ≤ m24 := by
                by_cases hge : log2m ≥ 23
                · have hle : 23 ≤ log2m := hge
                  set sh : Nat := log2m - 23
                  have hsh : log2m = sh + 23 := by
                    simpa [sh, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
                      (Nat.sub_add_cancel hle).symm
                  have hpow_le_q : 2 ^ 23 ≤ Nat.shiftRight d.mant sh := by
                    have hpos : 0 < 2 ^ sh :=
                      Nat.pow_pos (a := 2) (n := sh) (by decide : 0 < (2 : Nat))
                    have hdiv := Nat.div_le_div_right (c := 2 ^ sh) hpow_le
                    have hpow_div : (2 ^ log2m) / (2 ^ sh) = 2 ^ 23 := by
                      have hpow : 2 ^ log2m = (2 ^ sh) * (2 ^ 23) := by
                        simp [hsh, Nat.pow_add]
                      simp [hpow]
                    simpa [Nat.shiftRight_eq_div_pow, sh, hpow_div] using hdiv
                  have hq_le_m24 : Nat.shiftRight d.mant sh ≤ m24 := by
                    have := shiftRight_le_roundShiftRightEven (n := d.mant) (shift := sh)
                    simpa [m24, hge, sh] using this
                  exact le_trans hpow_le_q hq_le_m24
                · have hlt : log2m < 23 := lt_of_not_ge hge
                  set sh : Nat := 23 - log2m
                  have hpos : 0 < 2 ^ sh :=
                    Nat.pow_pos (a := 2) (n := sh) (by decide : 0 < (2 : Nat))
                  have hmul := Nat.mul_le_mul_right (k := 2 ^ sh) hpow_le
                  have hpow23 : 2 ^ log2m * 2 ^ sh = 2 ^ 23 := by
                    have hsum : log2m + sh = 23 := Nat.add_sub_of_le (Nat.le_of_lt hlt)
                    have : 2 ^ 23 = 2 ^ log2m * 2 ^ sh := by
                      simpa [hsum] using (Nat.pow_add 2 log2m sh)
                    simpa using this.symm
                  simpa [m24, hge, sh, Nat.shiftLeft_eq, hpow23, Nat.mul_assoc] using hmul

              have hm24_le : m24 ≤ 2 ^ 24 := by
                by_cases hge : log2m ≥ 23
                · have hle : 23 ≤ log2m := hge
                  set sh : Nat := log2m - 23
                  have hsh : log2m.succ = sh + 24 := by
                    have hlog : log2m = sh + 23 := by
                      simpa [sh, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
                        (Nat.sub_add_cancel hle).symm
                    rw [hlog]
                  have hq_lt : Nat.shiftRight d.mant sh < 2 ^ 24 := by
                    have hpow : (2 ^ log2m.succ) = 2 ^ (sh + 24) :=
                      congrArg (fun n : Nat => 2 ^ n) hsh
                    have hmant_lt' : d.mant < 2 ^ (sh + 24) :=
                      lt_of_lt_of_eq hmant_lt hpow
                    have hmul : d.mant < (2 ^ sh) * (2 ^ 24) := by
                      -- Arrange as `b * c` so `Nat.div_lt_of_lt_mul` can divide by `2^sh`.
                      simpa [Nat.pow_add, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using
                        hmant_lt'
                    have hdiv_lt : d.mant / 2 ^ sh < 2 ^ 24 :=
                      Nat.div_lt_of_lt_mul hmul
                    simpa [Nat.shiftRight_eq_div_pow, sh] using hdiv_lt
                  have hq_succ_le : Nat.shiftRight d.mant sh + 1 ≤ 2 ^ 24 :=
                    Nat.succ_le_of_lt hq_lt
                  have hm24_le_q : m24 ≤ Nat.shiftRight d.mant sh + 1 := by
                    have := roundShiftRightEven_le_shiftRight_add1 (n := d.mant) (shift := sh)
                    simpa [m24, hge, sh] using this
                  exact le_trans hm24_le_q hq_succ_le
                · have hlt : log2m < 23 := lt_of_not_ge hge
                  set sh : Nat := 23 - log2m
                  have hpos : 0 < 2 ^ sh :=
                    Nat.pow_pos (a := 2) (n := sh) (by decide : 0 < (2 : Nat))
                  have hmul :
                      d.mant * 2 ^ sh < (2 ^ log2m.succ) * 2 ^ sh :=
                    Nat.mul_lt_mul_of_pos_right hmant_lt hpos
                  have hsum : log2m.succ + sh = 24 := by
                    calc
                      log2m.succ + sh = (log2m + sh) + 1 := by
                        simp [Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
                      _ = 23 + 1 := by
                        simp [sh, Nat.add_sub_of_le (Nat.le_of_lt hlt)]
                      _ = 24 := by decide
                  have hprod : (2 ^ log2m.succ) * 2 ^ sh = 2 ^ 24 := by
                    have : (2 ^ log2m.succ) * 2 ^ sh = 2 ^ (log2m.succ + sh) := by
                      simpa using (Nat.pow_add 2 log2m.succ sh).symm
                    simpa [hsum] using this
                  have hmul' : d.mant * 2 ^ sh < 2 ^ 24 := by
                    simpa [hprod] using hmul
                  have hshift : d.mant.shiftLeft sh < 2 ^ 24 := by
                    simpa [Nat.shiftLeft_eq] using hmul'
                  exact le_of_lt (by simpa [m24, hge, sh] using hshift)

              have hm24'_ge : 2 ^ 23 ≤ m24' := by
                cases hcarry : m24 == pow2 24 with
                | true =>
                    have hm24' : m24' = pow2 23 := by
                      simp [m24', hcarry]
                    simp [hm24', pow2_eq_two_pow]
                | false =>
                    simpa [m24', hcarry] using hm24_ge

              have hm24'_lt : m24' < 2 ^ 24 := by
                cases hcarry : m24 == pow2 24 with
                | true =>
                    have : m24' = pow2 23 := by simp [m24', hcarry]
                    simp [this, pow2_eq_two_pow]
                | false =>
                    have hm24_ne : m24 ≠ 2 ^ 24 := by
                      have : m24 ≠ pow2 24 := (beq_eq_false_iff_ne).1 hcarry
                      simpa [pow2_eq_two_pow] using this
                    have hm24_lt : m24 < 2 ^ 24 := lt_of_le_of_ne hm24_le hm24_ne
                    simpa [m24', hcarry] using hm24_lt

              have hfrac : fracNat < 2 ^ 23 := by
                have hm24'_ge' : pow2 23 ≤ m24' := by
                  simpa [pow2_eq_two_pow] using hm24'_ge
                have hm24'_lt' : m24' < pow2 24 := by
                  simpa [pow2_eq_two_pow] using hm24'_lt
                have hsum : fracNat + pow2 23 = m24' := by
                  have : (m24' - pow2 23) + pow2 23 = m24' := Nat.sub_add_cancel hm24'_ge'
                  simpa [fracNat] using this
                have hlt : fracNat + pow2 23 < pow2 24 := by
                  simpa [hsum] using hm24'_lt'
                have hpow : pow2 24 = pow2 23 + pow2 23 := by
                  -- `2^24 = 2^23 + 2^23`.
                  simp [pow2_eq_two_pow]
                have hlt' : fracNat + pow2 23 < pow2 23 + pow2 23 := by
                  simpa [hpow] using hlt
                have : fracNat < pow2 23 :=
                  (Nat.add_lt_add_iff_right (k := pow2 23)).1 hlt'
                simpa [pow2_eq_two_pow] using this

              -- Compute the executable real value via decoding.
              have hto :
                  toReal (roundDyadicToIEEE32 d) =
                    (if d.sign then (-1 : ℝ) else (1 : ℝ)) *
                      (m24' : ℝ) * neuralBpow binaryRadix (k' - 23) := by
                rw [hroundBits]
                have hk'exp_int : (Int.ofNat expNat : Int) = k' + 127 := by
                  simp [expNat, Int.toNat_of_nonneg hk'exp_nonneg]
                have hkpos : (0 : Int) < k' + 127 := by
                  linarith [hk'ge]
                have hExpNat_ne0 : expNat ≠ 0 := by
                  intro h0
                  have h0eq : (0 : Int) = k' + 127 := by
                    simpa [h0] using hk'exp_int
                  have : (k' + 127) = 0 := h0eq.symm
                  exact (ne_of_gt hkpos) this
                have hmantNat : pow2 23 + fracNat = m24' := by
                  have hm24'_ge' : pow2 23 ≤ m24' := by
                    simpa [pow2_eq_two_pow] using hm24'_ge
                  have hsub : (m24' - pow2 23) + pow2 23 = m24' := Nat.sub_add_cancel hm24'_ge'
                  calc
                    pow2 23 + fracNat = pow2 23 + (m24' - pow2 23) := by simp [fracNat]
                    _ = (m24' - pow2 23) + pow2 23 := by
                      simp [Nat.add_comm]
                    _ = m24' := by simpa using hsub
                have hkexp : (Int.ofNat expNat : Int) - 150 = k' - 23 := by
                  rw [hk'exp_int]
                  linarith
                -- Use the decoding lemma.
                rw [toReal_eq]
                have hdy :
                    toDyadic? (ofBits (mkBits d.sign expNat fracNat)) =
                      some { sign := d.sign, mant := pow2 23 + fracNat, exp := (Int.ofNat expNat) -
                        150 } := by
                  have hdec :=
                    toDyadic?_ofBits_mkBits_fin (sign := d.sign) (exp := expNat) (frac := fracNat)
                      (hexp := hexp) (hfrac := by simpa [pow2_eq_two_pow] using hfrac)
                  simpa [hExpNat_ne0] using hdec
                -- Rewrite mantissa and simplify.
                simp [hdy, dyadicToReal, hmantNat]
                -- Rewrite the exponent argument.
                have hkexp' : (expNat : Int) - 150 = k' - 23 := by
                  simpa using hkexp
                rw [hkexp']

              -- Compute `FP32` rounding: exponent is `k - 23` in the normal range.
              have hcexp : TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32
                (dyadicToReal d) = (k - 23 : Int) := by
                have hmag :
                    TorchLean.Floats.neuralMagnitude binaryRadix (dyadicToReal d) = k + 1 := by
                  have hmag0 := neural_magnitude_dyadic (d := d) hm
                  have hlog : Nat.log 2 d.mant = Nat.log2 d.mant := (Nat.log2_eq_log_two (n :=
                    d.mant)).symm
                  simpa [k, log2m, hlog] using hmag0
                have hk_ge : (-149 : Int) ≤ k - 23 := by
                  have hkge : (-126 : Int) ≤ k := (not_lt).1 hkSub
                  linarith [hkge]
                have hk' : k + 1 - 24 = k - 23 := by linarith
                simp [TorchLean.Floats.neuralCexp, TorchLean.Floats.fexp32,
                  TorchLean.Floats.FLTExp, hmag, hk',
                  max_eq_left hk_ge]

              -- Show the mantissa rounding in `FP32` matches the kernel’s `m24`.
              have hRndM24 :
                  TorchLean.Floats.neuralNearestEven (dyadicToReal d * neuralBpow binaryRadix
                    (-(k - 23 : Int))) =
                    if d.sign then -Int.ofNat m24 else Int.ofNat m24 := by
                -- First compute the `sign=false` case (depends only on `mant/exp`), then use
                -- oddness.
                have hpos :
                    TorchLean.Floats.neuralNearestEven
                        (dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                          neuralBpow binaryRadix (-(k - 23 : Int))) =
                      Int.ofNat m24 := by
                  -- Simplify the scaling: exponent cancellation removes `d.exp`.
                  have hscale :
                      dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                          neuralBpow binaryRadix (-(k - 23 : Int)) =
                        (d.mant : ℝ) * neuralBpow binaryRadix (23 - Int.ofNat log2m) := by
                    have hk : (-(k - 23 : Int)) = (23 - k) := by linarith
                    have hexp : d.exp + (23 - k) = 23 - Int.ofNat log2m := by
                      simp [k, sub_eq_add_neg, add_assoc, add_comm]
                    have hb :
                        neuralBpow binaryRadix d.exp * neuralBpow binaryRadix (23 - k) =
                          neuralBpow binaryRadix (23 - Int.ofNat log2m) := by
                      calc
                        neuralBpow binaryRadix d.exp * neuralBpow binaryRadix (23 - k) =
                            neuralBpow binaryRadix (d.exp + (23 - k)) := by
                              simpa using (neuralBpow.add_exp binaryRadix d.exp (23 - k)).symm
                        _ = neuralBpow binaryRadix (23 - Int.ofNat log2m) := by
                              simp [hexp]
                    simp [dyadicToReal, hk, hb, mul_assoc, mul_comm]

                  by_cases hge : log2m ≥ 23
                  · -- shift-right rounding
                    set sh : Nat := log2m - 23
                    have hpow :
                        (d.mant : ℝ) * neuralBpow binaryRadix (23 - Int.ofNat log2m) =
                          (d.mant : ℝ) / (pow2 sh : ℝ) := by
                      have hle : 23 ≤ log2m := hge
                      have hsub : (23 : Int) - (Int.ofNat log2m) = - (Int.ofNat sh) := by
                        have : (Int.ofNat sh : Int) = (Int.ofNat log2m : Int) - 23 := by
                          simp [sh, Int.ofNat_sub hle]
                        simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using (congrArg
                          Neg.neg this).symm
                      calc
                        (d.mant : ℝ) * neuralBpow binaryRadix (23 - Int.ofNat log2m) =
                            (d.mant : ℝ) * neuralBpow binaryRadix (-Int.ofNat sh) := by
                              -- Avoid `simp` cancellation on the common factor `(d.mant : ℝ)`.
                              exact
                                congrArg (fun e : Int => (d.mant : ℝ) * neuralBpow binaryRadix e)
                                  hsub
                        _ = (d.mant : ℝ) * (neuralBpow binaryRadix (Int.ofNat sh))⁻¹ := by
                              simp [neuralBpow.neg_exp]
                        _ = (d.mant : ℝ) / (neuralBpow binaryRadix (Int.ofNat sh)) := by
                              simp [div_eq_mul_inv]
                        _ = (d.mant : ℝ) / (pow2 sh : ℝ) := by
                              have hden : neuralBpow binaryRadix (Int.ofNat sh) = (pow2 sh : ℝ) :=
                                by
                                simp [TorchLean.Floats.neuralBpow, binaryRadix,
                                  NeuralRadix.toReal, pow2_eq_two_pow,
                                  Nat.cast_pow]
                              rw [hden]
                    have hround :=
                      neural_nearest_even_div_pow2_eq_roundShiftRightEven (num := d.mant) (shift :=
                        sh)
                    have hcong :
                        TorchLean.Floats.neuralNearestEven ((d.mant : ℝ) * neuralBpow
                          binaryRadix (23 - Int.ofNat log2m)) =
                          TorchLean.Floats.neuralNearestEven ((d.mant : ℝ) / (pow2 sh : ℝ)) :=
                      congrArg TorchLean.Floats.neuralNearestEven hpow
                    have hne :
                        TorchLean.Floats.neuralNearestEven ((d.mant : ℝ) * neuralBpow
                          binaryRadix (23 - Int.ofNat log2m)) =
                          Int.ofNat (roundShiftRightEven d.mant sh) := by
                      calc
                        TorchLean.Floats.neuralNearestEven
                            ((d.mant : ℝ) * neuralBpow binaryRadix (23 - Int.ofNat log2m)) =
                            TorchLean.Floats.neuralNearestEven ((d.mant : ℝ) / (pow2 sh : ℝ)) :=
                              by
                              simpa using hcong
                        _ = Int.ofNat (roundShiftRightEven d.mant sh) := hround
                    have hcong' :
                        TorchLean.Floats.neuralNearestEven
                            (dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                              neuralBpow binaryRadix (-(k - 23 : Int))) =
                          TorchLean.Floats.neuralNearestEven
                              ((d.mant : ℝ) * neuralBpow binaryRadix (23 - Int.ofNat log2m)) :=
                      congrArg TorchLean.Floats.neuralNearestEven hscale
                    calc
                      TorchLean.Floats.neuralNearestEven
                          (dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                            neuralBpow binaryRadix (-(k - 23 : Int))) =
                          TorchLean.Floats.neuralNearestEven
                              ((d.mant : ℝ) * neuralBpow binaryRadix (23 - Int.ofNat log2m)) := by
                            simpa using hcong'
                      _ = Int.ofNat (roundShiftRightEven d.mant sh) := hne
                      _ = Int.ofNat m24 := by
                            simp [m24, hge, sh]
                  · -- shift-left (exact)
                    have hlt : log2m < 23 := lt_of_not_ge hge
                    set sh : Nat := 23 - log2m
                    have hsub : (23 : Int) - (Int.ofNat log2m) = Int.ofNat sh := by
                      have hle : log2m ≤ 23 := Nat.le_of_lt hlt
                      simp [sh, Int.ofNat_sub hle, sub_eq_add_neg, add_comm]
                    have hpow :
                        (d.mant : ℝ) * neuralBpow binaryRadix (23 - Int.ofNat log2m) =
                          ((Nat.shiftLeft d.mant sh : Nat) : ℝ) := by
                      have hsub' : (23 - Int.ofNat log2m) = Int.ofNat sh := by
                        simpa using hsub
                      rw [hsub']
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                        Nat.shiftLeft_eq, Nat.cast_mul,
                        Nat.cast_pow]
                    have hid :
                        TorchLean.Floats.neuralNearestEven ((Nat.shiftLeft d.mant sh : Nat) : ℝ) =
                          Int.ofNat (Nat.shiftLeft d.mant sh) := by
                      simpa using
                        (TorchLean.Floats.NeuralValidRnd.id (rnd :=
                          TorchLean.Floats.neuralNearestEven)
                          (Int.ofNat (Nat.shiftLeft d.mant sh)))
                    have hcong :
                        TorchLean.Floats.neuralNearestEven ((d.mant : ℝ) * neuralBpow
                          binaryRadix (23 - Int.ofNat log2m)) =
                          TorchLean.Floats.neuralNearestEven ((Nat.shiftLeft d.mant sh : Nat) : ℝ)
                            :=
                      congrArg TorchLean.Floats.neuralNearestEven hpow
                    have hne :
                        TorchLean.Floats.neuralNearestEven ((d.mant : ℝ) * neuralBpow
                          binaryRadix (23 - Int.ofNat log2m)) =
                          Int.ofNat (Nat.shiftLeft d.mant sh) := by
                      calc
                        TorchLean.Floats.neuralNearestEven
                            ((d.mant : ℝ) * neuralBpow binaryRadix (23 - Int.ofNat log2m)) =
                            TorchLean.Floats.neuralNearestEven ((Nat.shiftLeft d.mant sh : Nat) :
                              ℝ) := by
                              simpa using hcong
                        _ = Int.ofNat (Nat.shiftLeft d.mant sh) := hid
                    have hcong' :
                        TorchLean.Floats.neuralNearestEven
                            (dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                              neuralBpow binaryRadix (-(k - 23 : Int))) =
                          TorchLean.Floats.neuralNearestEven
                              ((d.mant : ℝ) * neuralBpow binaryRadix (23 - Int.ofNat log2m)) :=
                      congrArg TorchLean.Floats.neuralNearestEven hscale
                    calc
                      TorchLean.Floats.neuralNearestEven
                          (dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                            neuralBpow binaryRadix (-(k - 23 : Int))) =
                          TorchLean.Floats.neuralNearestEven
                              ((d.mant : ℝ) * neuralBpow binaryRadix (23 - Int.ofNat log2m)) := by
                            simpa using hcong'
                      _ = Int.ofNat (Nat.shiftLeft d.mant sh) := hne
                      _ = Int.ofNat m24 := by
                            simp [m24, hge, sh]

                cases hs : d.sign
                · -- positive
                  simpa [hs, dyadicToReal] using hpos
                · -- negative
                  have hodd :=
                    neural_nearest_even_neg
                      (x := dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                        neuralBpow binaryRadix (-(k - 23 : Int)))
                  have hx :
                      dyadicToReal d * neuralBpow binaryRadix (-(k - 23 : Int)) =
                        -(dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                          neuralBpow binaryRadix (-(k - 23 : Int))) := by
                    simp [dyadicToReal, hs, mul_assoc, mul_comm]
                  have hneg :
                      TorchLean.Floats.neuralNearestEven
                          (dyadicToReal d * neuralBpow binaryRadix (-(k - 23 : Int))) =
                        -Int.ofNat m24 := by
                    calc
                      TorchLean.Floats.neuralNearestEven
                          (dyadicToReal d * neuralBpow binaryRadix (-(k - 23 : Int))) =
                          TorchLean.Floats.neuralNearestEven
                              (-(dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                                neuralBpow binaryRadix (-(k - 23 : Int)))) := by
                            exact congrArg TorchLean.Floats.neuralNearestEven hx
                      _ =
                          -TorchLean.Floats.neuralNearestEven
                              (dyadicToReal { sign := false, mant := d.mant, exp := d.exp } *
                                neuralBpow binaryRadix (-(k - 23 : Int))) := by
                            simpa using hodd
                      _ = -Int.ofNat m24 := by
                            rw [hpos]
                  simpa [hs] using hneg

              have hfp :
                  fp32Round (dyadicToReal d) =
                    (if d.sign then (-1 : ℝ) else (1 : ℝ)) * (m24 : ℝ) * neuralBpow binaryRadix (k
                      - 23) := by
                cases hs : d.sign
                · -- positive
                  simp [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal,
                    TorchLean.Floats.neuralScaledMantissa, TorchLean.Floats.rnd32, hcexp,
                    mul_comm]
                  left
                  have h0 :
                      TorchLean.Floats.neuralNearestEven
                          (dyadicToReal d * neuralBpow binaryRadix (-(k - 23 : Int))) =
                        Int.ofNat m24 := by
                    simpa [hs] using hRndM24
                  have h1 :
                      TorchLean.Floats.neuralNearestEven
                          (neuralBpow binaryRadix (-(k - 23 : Int)) * dyadicToReal d) =
                        Int.ofNat m24 := by
                    simpa [mul_comm] using h0
                  simpa using congrArg (fun z : Int => (z : ℝ)) h1
                · -- negative
                  simp [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal,
                    TorchLean.Floats.neuralScaledMantissa, TorchLean.Floats.rnd32, hcexp,
                    mul_comm]
                  have h0 :
                      TorchLean.Floats.neuralNearestEven
                          (dyadicToReal d * neuralBpow binaryRadix (-(k - 23 : Int))) =
                        -Int.ofNat m24 := by
                    simpa [hs] using hRndM24
                  have h1 :
                      TorchLean.Floats.neuralNearestEven
                          (neuralBpow binaryRadix (-(k - 23 : Int)) * dyadicToReal d) =
                        -Int.ofNat m24 := by
                    simpa [mul_comm] using h0
                  have h1' :
                      ((TorchLean.Floats.neuralNearestEven
                          (neuralBpow binaryRadix (-(k - 23 : Int)) * dyadicToReal d)) : ℝ) =
                        -(m24 : ℝ) := by
                    simpa using congrArg (fun z : Int => (z : ℝ)) h1
                  have hsub : (-(k - 23 : Int)) = (23 - k) := by linarith
                  have h1'' :
                      ((TorchLean.Floats.neuralNearestEven
                          (neuralBpow binaryRadix (23 - k) * dyadicToReal d)) : ℝ) =
                        -(m24 : ℝ) := by
                    simpa [hsub] using h1'
                  simp [h1'']

              -- Final step: relate `(m24', k')` to `(m24, k)` (carry adjustment).
              have hadj :
                  (if d.sign then (-1 : ℝ) else (1 : ℝ)) * (m24' : ℝ) * neuralBpow binaryRadix (k'
                    - 23) =
                    (if d.sign then (-1 : ℝ) else (1 : ℝ)) * (m24 : ℝ) * neuralBpow binaryRadix (k
                      - 23) := by
                cases hcarry : m24 == pow2 24 with
                | true =>
                    have hm24Nat : m24 = pow2 24 := (beq_iff_eq).1 hcarry
                    have hm24R : (m24 : ℝ) = (pow2 24 : ℝ) := by
                      simp [hm24Nat]
                    -- Reduce the `if`/`ite` carry adjustments.
                    simp [k', m24', hcarry, hm24R, mul_comm]
                    -- Now show: `2^23 * 2^(k+1-23) = 2^24 * 2^(k-23)`.
                    have hk : (k + 1 : Int) - 23 = (k - 23) + 1 := by linarith
                    rw [hk, neuralBpow.add_exp]
                    have htwo : neuralBpow binaryRadix (1 : Int) = (2 : ℝ) := by
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                    rw [htwo]
                    -- `pow2 24 = 2 * pow2 23` (as reals).
                    have hp : ((pow2 24 : Nat) : ℝ) = (2 : ℝ) * (pow2 23 : ℝ) := by
                      -- `pow2 k = 2^k` and `2^24 = 2 * 2^23`.
                      simp [pow2_eq_two_pow, Nat.pow_succ]
                      norm_num
                    -- Finish by rewriting the constant factor.
                    simp [hp, mul_assoc, mul_left_comm, mul_comm]
                | false =>
                    simp [k', m24', hcarry]

              -- Combine the executable and FP32 computations.
              rw [hto, hfp]
              exact hadj
        -- end hkSub split
        -- end hkUnder split
    -- end hkHi split

/-!
## Rounding rationals (finite/no-overflow)

Some operations (notably division and parts of transcendental approximations) naturally produce
rationals `num / den`. The executable kernel rounds those rationals to float32 by:

- classifying magnitude (normal/subnormal/underflow/overflow),
- computing a scaled mantissa,
- applying nearest-even,
- and assembling the output bits.

This section connects that algorithm to the `FP32` real rounding model.
-/

private lemma neural_magnitude_signedRat (sign : Bool) (num den : Nat) (hnum : num ≠ 0) (hden : den
  ≠ 0) :
    TorchLean.Floats.neuralMagnitude binaryRadix ((if sign then (-1 : ℝ) else 1) * ((num : ℝ) /
      (den : ℝ))) =
      floorLog2Rat num den + 1 := by
  classical
  set k : Int := floorLog2Rat num den
  set r : ℝ := (num : ℝ) / (den : ℝ)
  set x : ℝ := (if sign then (-1 : ℝ) else 1) * r
  have hdenpos : (0 : ℝ) < (den : ℝ) := by
    exact_mod_cast Nat.pos_of_ne_zero hden
  have hnumpos : (0 : ℝ) < (num : ℝ) := by
    exact_mod_cast Nat.pos_of_ne_zero hnum
  have hrpos : 0 < r := div_pos hnumpos hdenpos
  have hx : x ≠ 0 := by
    have hr0 : r ≠ 0 := ne_of_gt hrpos
    cases sign <;> simp [x, hr0]
  have habs : _root_.abs x = r := by
    cases sign <;> simp [x, r, abs_of_pos hrpos]

  have hbounds := floorLog2Rat_bounds (num := num) (den := den) hnum hden
  have hk_le : neuralBpow binaryRadix k ≤ r := by
    simpa [k, r] using hbounds.1
  have hk_lt : r < neuralBpow binaryRadix (k + 1) := by
    simpa [k, r] using hbounds.2

  have hkpow_le : (2 : ℝ) ^ (k : ℝ) ≤ r := by
    have : (2 : ℝ) ^ k ≤ r := by
      simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, k, r] using hk_le
    have hEq : (2 : ℝ) ^ k = (2 : ℝ) ^ ((k : ℤ) : ℝ) := by
      simp
    have : (2 : ℝ) ^ ((k : ℤ) : ℝ) ≤ r := le_of_eq_of_le hEq.symm this
    simpa using this
  have hkpow_lt : r < (2 : ℝ) ^ ((k + 1) : ℝ) := by
    have : r < (2 : ℝ) ^ (k + 1) := by
      simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, k, r] using hk_lt
    have hEq : (2 : ℝ) ^ (k + 1) = (2 : ℝ) ^ (((k + 1 : ℤ)) : ℝ) := by
      simpa using (Real.rpow_intCast (x := (2 : ℝ)) (n := k + 1)).symm
    have : r < (2 : ℝ) ^ (((k + 1 : ℤ)) : ℝ) := lt_of_lt_of_eq this hEq
    -- Rewrite `((k+1 : ℤ) : ℝ)` as `((k : ℤ) : ℝ) + 1`.
    simpa [Int.cast_add, Int.cast_one] using this

  have hk_logb : (k : ℝ) ≤ Real.logb 2 r := by
    have hb : 1 < (2 : ℝ) := by norm_num
    exact (Real.le_logb_iff_rpow_le (b := (2 : ℝ)) (x := (k : ℝ)) (y := r) hb hrpos).2 hkpow_le
  have hk_logb_lt : Real.logb 2 r < (k : ℝ) + 1 := by
    have hb : 1 < (2 : ℝ) := by norm_num
    have hr_lt' : r < (2 : ℝ) ^ ((k : ℝ) + 1) := by
      -- Rewrite `(k+1 : ℝ)` as `k + 1`.
      simpa [Int.cast_add, Int.cast_one, add_assoc] using hkpow_lt
    exact (Real.logb_lt_iff_lt_rpow (b := (2 : ℝ)) (x := r) (y := (k : ℝ) + 1) hb hrpos).2 hr_lt'

  have hfloor : (⌊Real.logb 2 r⌋ : Int) = k := by
    refine (Int.floor_eq_iff).2 ?_
    constructor
    · exact hk_logb
    · exact hk_logb_lt

  have hb2 : binaryRadix.toReal = (2 : ℝ) := by rfl
  have hfloor' : (⌊Real.logb (binaryRadix.toReal) r⌋ : Int) = k := by
    simpa [hb2] using hfloor
  have : TorchLean.Floats.neuralMagnitude binaryRadix x = k + 1 := by
    -- Unfold and reduce `neural_magnitude` to the floor-log identity already proved as `hfloor`.
    simp [TorchLean.Floats.neuralMagnitude, hx, Real.log_div_log, habs, hb2]
    simpa [k] using hfloor
  simpa [x, k] using this

-- Shared dyadic/rational scaling lemmas live in `NN.Floats.IEEEExec.RatScaling`.

/--
Refinement theorem (finite/no-overflow): rounding an exact rational with the executable IEEE32
  kernel
agrees with the Flocq-style `FP32` rounding-on-`ℝ` model.

The hypothesis `isFinite (roundRatToIEEE32 sign num den) = true` rules out the overflow-to-`±Inf`
branches.
-/
theorem toReal_roundRatToIEEE32_eq_fp32Round (sign : Bool) (num den : Nat) (hden : den ≠ 0)
    (hfin : isFinite (roundRatToIEEE32 sign num den) = true) :
    toReal (roundRatToIEEE32 sign num den) =
      fp32Round ((if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ))) := by
  classical
  by_cases hnum : num = 0
  · -- Both sides are real `0`.
    have hto : toReal (roundRatToIEEE32 sign num den) = 0 := by
      have hround0 : roundRatToIEEE32 sign num den = (if sign then negZero else posZero) := by
        simp [roundRatToIEEE32, hnum]
      rw [hround0]
      simpa using (toReal_signedZero sign)
    have hfp : fp32Round ((if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ))) = 0 := by
      have hx0 :
          (if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ)) = 0 := by
        simp [hnum]
      have hrnd0 : TorchLean.Floats.rnd32
          (TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 0) = 0 := by
        -- scaled mantissa is `0`, and nearest-even rounds `0` to `0`.
        have hAbs : _root_.abs (0 : ℝ) < (1 / 2 : ℝ) := by simp
        simpa [TorchLean.Floats.rnd32, TorchLean.Floats.neuralScaledMantissa] using
          (neural_nearest_even_eq_zero_of_abs_lt_half (x := (0 : ℝ)) hAbs)
      -- Now compute `fp32Round` at `0`.
      -- Keep `rnd32`/`neural_scaled_mantissa` folded so `hrnd0` can rewrite the mantissa to `0`.
      simp [fp32Round, hx0, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal, hrnd0]
    rw [hto, hfp]
  · have hnumbeq : (num == 0) = false := (beq_eq_false_iff_ne).2 hnum
    have hdenpos : (0 : ℝ) < (den : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hden
    have hnumpos : (0 : ℝ) < (num : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hnum
    set r : ℝ := (num : ℝ) / (den : ℝ)
    set x : ℝ := (if sign then (-1 : ℝ) else 1) * r
    have hrpos : 0 < r := div_pos hnumpos hdenpos
    have hx : x ≠ 0 := by
      have : r ≠ 0 := ne_of_gt hrpos
      cases sign <;> simp [x, this]

    set k : Int := floorLog2Rat num den
    -- Eliminate the overflow-to-Inf branch.
    by_cases hkHi : k > 127
    · have hround : roundRatToIEEE32 sign num den = (if sign then negInf else posInf) := by
        simp [roundRatToIEEE32, hnumbeq, k, hkHi]
      have hfalse : isFinite (roundRatToIEEE32 sign num den) = false := by
        rw [hround]
        cases sign <;> decide
      cases (hfalse.symm.trans hfin)
    · -- Non-overflowing exponent range: `k ≤ 127`.
      by_cases hkUnder : k < -150
      · -- Underflow-to-zero: show `FP32` rounding also yields `0`.
        have hround : roundRatToIEEE32 sign num den = (if sign then negZero else posZero) := by
          simp [roundRatToIEEE32, hnumbeq, k, hkHi, hkUnder]
        have hto : toReal (roundRatToIEEE32 sign num den) = 0 := by
          simpa [hround] using (toReal_signedZero sign)

        have hmag :
            TorchLean.Floats.neuralMagnitude binaryRadix x = k + 1 := by
          simpa [x, r, k] using
            (neural_magnitude_signedRat (sign := sign) (num := num) (den := den) hnum hden)
        have hcexp : TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32 x = (-149 :
          Int) := by
          have hk1_le : k + 1 ≤ (-150 : Int) := by linarith
          have hk1_le' : k + 1 - 24 ≤ (-149 : Int) := by linarith
          simp [TorchLean.Floats.neuralCexp, TorchLean.Floats.fexp32, TorchLean.Floats.FLTExp,
            hmag,
            max_eq_right hk1_le']

        have hAbsBpow : _root_.abs x < neuralBpow binaryRadix (k + 1) := by
          have hbounds := floorLog2Rat_bounds (num := num) (den := den) hnum hden
          have habs : _root_.abs x = r := by
            cases sign <;> simp [x, r, abs_of_pos hrpos]
          simpa [habs, r, k] using hbounds.2
        have hk1_le : k + 1 ≤ (-150 : Int) := by linarith
        have hBpow_le :
            neuralBpow binaryRadix (k + 1) ≤ neuralBpow binaryRadix (-150 : Int) := by
          simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
          exact zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2) hk1_le
        have hAbs150 : _root_.abs x < neuralBpow binaryRadix (-150 : Int) :=
          lt_of_lt_of_le hAbsBpow hBpow_le
        have hbpos149 : 0 < neuralBpow binaryRadix (149 : Int) := neuralBpow.pos binaryRadix 149
        have hAbsScaled :
            _root_.abs (x * neuralBpow binaryRadix (149 : Int)) < (1 / 2 : ℝ) := by
          have habs_mul :
              _root_.abs (x * neuralBpow binaryRadix (149 : Int)) =
                _root_.abs x * neuralBpow binaryRadix (149 : Int) := by
            have hnonneg : 0 ≤ neuralBpow binaryRadix (149 : Int) := le_of_lt hbpos149
            simp [abs_mul, abs_of_nonneg hnonneg]
          have hmul :
              _root_.abs x * neuralBpow binaryRadix (149 : Int) <
                neuralBpow binaryRadix (-150 : Int) * neuralBpow binaryRadix (149 : Int) :=
            (mul_lt_mul_of_pos_right hAbs150 hbpos149)
          have hprod :
              neuralBpow binaryRadix (-150 : Int) * neuralBpow binaryRadix (149 : Int) = (1 / 2
                : ℝ) := by
            have := (neuralBpow.add_exp binaryRadix (-150 : Int) (149 : Int))
            simpa [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal] using this.symm
          simpa [habs_mul, hprod, mul_assoc] using hmul
        have hRnd0 :
            TorchLean.Floats.neuralNearestEven (x * neuralBpow binaryRadix (149 : Int)) = 0 :=
          neural_nearest_even_eq_zero_of_abs_lt_half _ hAbsScaled
        have hfp : fp32Round x = 0 := by
          have hscaled :
              TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 x =
                x * neuralBpow binaryRadix (149 : Int) := by
            simp [TorchLean.Floats.neuralScaledMantissa, hcexp]
          have hmant :
              TorchLean.Floats.rnd32
                  (TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 x) =
                0 := by
            simpa [TorchLean.Floats.rnd32, hscaled] using hRnd0
          -- With mantissa rounded to `0`, `fp32Round x` is `0`.
          have hdisj :
              TorchLean.Floats.neuralNearestEven (x * neuralBpow binaryRadix (149 : Int)) = 0 ∨
                neuralBpow binaryRadix (-149 : Int) = 0 :=
            Or.inl hRnd0
          simpa [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal,
            TorchLean.Floats.neuralScaledMantissa, TorchLean.Floats.rnd32, hmant, hcexp, hscaled]
              using hdisj
        rw [hto, hfp]
      · -- Remaining cases (`k ≥ -150`): subnormal or normal rounding.
        by_cases hkSub : k < -126
        · -- Subnormal rounding.
          set frac : Nat := roundQuotEven (Nat.shiftLeft num 149) den
          have hround :
              roundRatToIEEE32 sign num den =
                if frac == 0 then
                  (if sign then negZero else posZero)
                else
                  match Nat.decLe (pow2 23) frac with
                  | isTrue _ => ofBits (mkBits sign 1 0)
                  | isFalse _ => ofBits (mkBits sign 0 frac) := by
            simp (config := { zeta := true }) [roundRatToIEEE32, hnumbeq, k, hkHi, hkUnder, hkSub,
              frac]
            rfl
          have hmag :
              TorchLean.Floats.neuralMagnitude binaryRadix x = k + 1 := by
            simpa [x, r, k] using
              (neural_magnitude_signedRat (sign := sign) (num := num) (den := den) hnum hden)
          have hcexp : TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32 x = (-149 :
            Int) := by
            have hk1_le : k + 1 ≤ (-125 : Int) := by linarith [hkSub]
            have hk1_le' : k + 1 - 24 ≤ (-149 : Int) := by linarith [hk1_le]
            simp [TorchLean.Floats.neuralCexp, TorchLean.Floats.fexp32, TorchLean.Floats.FLTExp,
              hmag,
              max_eq_right hk1_le']

          have hscaled :
              TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 x =
                x * neuralBpow binaryRadix (149 : Int) := by
            simp [TorchLean.Floats.neuralScaledMantissa, hcexp]
          have hScaleRat :
              r * neuralBpow binaryRadix (Int.ofNat 149) =
                ((Nat.shiftLeft num 149 : Nat) : ℝ) / (den : ℝ) := by
            -- `r = num/den`
            dsimp [r]
            exact scaleRat_ofNat (num := num) (den := den) (sh := 149)

          have hRndFracPos :
              TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (Int.ofNat 149)) =
                Int.ofNat frac := by
            have hden' : den ≠ 0 := hden
            have h :=
              neural_nearest_even_div_eq_roundQuotEven (num := Nat.shiftLeft num 149) (den := den)
                hden'
            -- rewrite the argument using `hScaleRat`
            calc
              TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (Int.ofNat 149))
                  =
                  TorchLean.Floats.neuralNearestEven (((Nat.shiftLeft num 149 : Nat) : ℝ) / (den :
                    ℝ)) := by
                    rw [hScaleRat]
              _ = Int.ofNat (roundQuotEven (Nat.shiftLeft num 149) den) := h
              _ = Int.ofNat frac := by simp [frac]

          have hRndFrac :
              TorchLean.Floats.neuralNearestEven (x * neuralBpow binaryRadix (149 : Int)) =
                if sign then -Int.ofNat frac else Int.ofNat frac := by
            cases hs : sign
            · -- positive
              have hx' : x = r := by simp [x, hs]
              have hb149 : neuralBpow binaryRadix (149 : Int) = neuralBpow binaryRadix
                (Int.ofNat 149) := by rfl
              simp [hx', hb149, hRndFracPos]
            · -- negative
              have hx' : x = -r := by simp [x, hs]
              have hb149 : neuralBpow binaryRadix (149 : Int) = neuralBpow binaryRadix
                (Int.ofNat 149) := by rfl
              have hneg :
                  TorchLean.Floats.neuralNearestEven (- (r * neuralBpow binaryRadix (Int.ofNat
                    149))) =
                    -TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (Int.ofNat
                      149)) := by
                simpa using (neural_nearest_even_neg (x := r * neuralBpow binaryRadix (Int.ofNat
                  149)))
              have hscale :
                  x * neuralBpow binaryRadix (149 : Int) = - (r * neuralBpow binaryRadix
                    (Int.ofNat 149)) := by
                simp [hx', hb149]
              -- Reduce to the positive case via oddness.
              calc
                TorchLean.Floats.neuralNearestEven (x * neuralBpow binaryRadix (Int.ofNat 149))
                    = TorchLean.Floats.neuralNearestEven (- (r * neuralBpow binaryRadix
                      (Int.ofNat 149))) := by
                      simpa [hb149] using congrArg TorchLean.Floats.neuralNearestEven hscale
                _ = -TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (Int.ofNat
                  149)) := hneg
                _ = -Int.ofNat frac := by simpa [hRndFracPos]

          have hfp :
              fp32Round x =
                (if sign then (-1 : ℝ) else 1) * (frac : ℝ) * neuralBpow binaryRadix (-149 : Int)
                  := by
            have hround :
                fp32Round x =
                  (TorchLean.Floats.rnd32
                      (TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32
                        x) : ℝ) *
                    neuralBpow binaryRadix (-149 : Int) := by
              simp [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal,
                hcexp]
            have hrndInt :
                TorchLean.Floats.rnd32 (TorchLean.Floats.neuralScaledMantissa binaryRadix
                  TorchLean.Floats.fexp32 x) =
                  (if sign then -Int.ofNat frac else Int.ofNat frac) := by
              -- `rnd32` is nearest-even, and `neural_scaled_mantissa = x * 2^149` in this branch.
              simpa [TorchLean.Floats.rnd32, hscaled] using hRndFrac
            have hrnd :
                (TorchLean.Floats.rnd32
                      (TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32
                        x) : ℝ) =
                  (if sign then (-1 : ℝ) else 1) * (frac : ℝ) := by
              have hrnd' := congrArg (fun z : Int => (z : ℝ)) hrndInt
              cases sign <;> simp [hrnd']
            calc
              fp32Round x =
                  (TorchLean.Floats.rnd32
                      (TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32
                        x) : ℝ) *
                    neuralBpow binaryRadix (-149 : Int) := hround
              _ = ((if sign then (-1 : ℝ) else 1) * (frac : ℝ)) * neuralBpow binaryRadix (-149 :
                Int) := by
                  simp [hrnd]
              _ = (if sign then (-1 : ℝ) else 1) * (frac : ℝ) * neuralBpow binaryRadix (-149 :
                Int) := by
                  simp []

          -- Compute `toReal` for the executable result.
          have hto :
              toReal (roundRatToIEEE32 sign num den) =
                (if sign then (-1 : ℝ) else 1) * (frac : ℝ) * neuralBpow binaryRadix (-149 : Int)
                  := by
            -- split on `frac == 0` and the `pow2 23 ≤ frac` test.
            by_cases hF0 : frac = 0
            · have hF0b : (frac == 0) = true := by simp [hF0]
              have hres : roundRatToIEEE32 sign num den = (if sign then negZero else posZero) := by
                simp [hround, hF0b]
              calc
                toReal (roundRatToIEEE32 sign num den) = 0 := by
                  rw [hres]
                  simpa using (toReal_signedZero sign)
                _ =
                    (if sign then (-1 : ℝ) else 1) * (frac : ℝ) *
                      neuralBpow binaryRadix (-149 : Int) := by
                  simp [hF0]
            · have hF0b : (frac == 0) = false := (beq_eq_false_iff_ne).2 hF0
              have hres : roundRatToIEEE32 sign num den =
                  match Nat.decLe (pow2 23) frac with
                  | isTrue _ => ofBits (mkBits sign 1 0)
                  | isFalse _ => ofBits (mkBits sign 0 frac) := by
                simp [hround, hF0b]
              rw [hres]
              -- `frac` is nonzero, so the output is either smallest normal or a true subnormal.
              cases hlt : Nat.decLe (pow2 23) frac with
              | isTrue hle =>
                  -- Output is the smallest normal, which corresponds to `frac = 2^23`.
                  have hle' : pow2 23 ≤ frac := hle
                  have hfrac_le : frac ≤ pow2 23 := by
                    -- Bound `r * 2^149 < 2^23`, then use `neural_nearest_even_bounds`.
                    have hk_lt : r < neuralBpow binaryRadix (k + 1) := by
                      have hbounds := floorLog2Rat_bounds (num := num) (den := den) hnum hden
                      simpa [r, k] using hbounds.2
                    have hk1_le : k + 1 ≤ (-126 : Int) := by linarith [hkSub]
                    have hbpow_le :
                        neuralBpow binaryRadix (k + 1) ≤ neuralBpow binaryRadix (-126 : Int) :=
                          by
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                      exact zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2) hk1_le
                    have hr_lt126 : r < neuralBpow binaryRadix (-126 : Int) :=
                      lt_of_lt_of_le hk_lt hbpow_le
                    have hbpos149 : 0 < neuralBpow binaryRadix (149 : Int) := neuralBpow.pos
                      binaryRadix 149
                    have hscaled_lt : r * neuralBpow binaryRadix (149 : Int) < (pow2 23 : ℝ) := by
                      have hsum : (-126 : Int) + 149 = 23 := by norm_num
                      have hmul :
                          neuralBpow binaryRadix (-126 : Int) * neuralBpow binaryRadix (149 :
                            Int) =
                            neuralBpow binaryRadix (23 : Int) := by
                        simpa [hsum] using (neuralBpow.add_exp binaryRadix (-126 : Int) (149 :
                          Int)).symm
                      have h23 : neuralBpow binaryRadix (23 : Int) = (pow2 23 : ℝ) := by
                        simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                          pow2_eq_two_pow]
                        norm_num
                      have hprod :
                          neuralBpow binaryRadix (-126 : Int) * neuralBpow binaryRadix (149 :
                            Int) = (pow2 23 : ℝ) :=
                        hmul.trans h23
                      have : r * neuralBpow binaryRadix (149 : Int) <
                          neuralBpow binaryRadix (-126 : Int) * neuralBpow binaryRadix (149 :
                            Int) :=
                        mul_lt_mul_of_pos_right hr_lt126 hbpos149
                      simpa [mul_assoc, hprod] using this
                    have hbound :
                        TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix
                          (Int.ofNat 149)) ≤
                          Int.ofNat (pow2 23) := by
                      have hbnds := TorchLean.Floats.neural_nearest_even_bounds (r * neuralBpow
                        binaryRadix (Int.ofNat 149))
                      have hfloor_lt :
                          (⌊r * neuralBpow binaryRadix (Int.ofNat 149)⌋ : Int) < Int.ofNat (pow2
                            23) :=
                        Int.floor_lt.2 (by simpa using hscaled_lt)
                      have hfloor_le :
                          (⌊r * neuralBpow binaryRadix (Int.ofNat 149)⌋ : Int) + 1 ≤ Int.ofNat
                            (pow2 23) :=
                        Int.add_one_le_iff.mpr hfloor_lt
                      exact le_trans hbnds.2 hfloor_le
                    have hint : Int.ofNat frac ≤ Int.ofNat (pow2 23) := by
                      -- Rewrite the LHS into the bounded nearest-even term.
                      rw [← hRndFracPos]
                      exact hbound
                    exact (Int.ofNat_le).1 hint
                  have hEq : frac = pow2 23 := le_antisymm hfrac_le hle'
                  have hexp1 : (1 : Nat) < 255 := by decide
                  have hfrac0 : (0 : Nat) < 2 ^ 23 := by decide
                  simp [hEq, toReal_eq,
                    toDyadic?_ofBits_mkBits_fin (sign := sign) (exp := 1) (frac := 0)
                      (hexp := hexp1) (hfrac := hfrac0),
                    dyadicToReal, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                    pow2_eq_two_pow, Nat.cast_ofNat, mul_comm]
              | isFalse hlt' =>
                  -- True subnormal: decode `mkBits sign 0 frac`.
                  have hfrac_lt : frac < 2 ^ 23 := by
                    -- `¬ pow2 23 ≤ frac` implies `frac < pow2 23 = 2^23`.
                    have : frac < pow2 23 := Nat.lt_of_not_ge hlt'
                    simpa [pow2_eq_two_pow] using this
                  simp [toReal_eq,
                    toDyadic?_ofBits_mkBits_fin (sign := sign) (exp := 0) (frac := frac)
                      (hexp := (by decide : (0 : Nat) < 255)) (hfrac := hfrac_lt),
                    dyadicToReal, TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                      hF0,
                    ]

          -- Combine the executable and FP32 computations.
          rw [hto, hfp]
        · -- Normal rounding.
          -- Mirror the executable kernel definitions.
          set shift : Int := 23 - k
          set numden :=
            match shift with
            | .ofNat sh => (Nat.shiftLeft num sh, den)
            | .negSucc sh => (num, Nat.shiftLeft den (sh + 1))
          set num' : Nat := numden.1
          set den' : Nat := numden.2
          set m : Nat := roundQuotEven num' den'
          set k' : Int := if m == pow2 24 then k + 1 else k
          set m' : Nat := if m == pow2 24 then pow2 23 else m
          have hround :
              roundRatToIEEE32 sign num den =
                if k' > 127 then
                  (if sign then negInf else posInf)
                else
                  let expNat : Nat := Int.toNat (k' + 127)
                  let fracNat : Nat := m' - pow2 23
                  ofBits (mkBits sign expNat fracNat) := by
            simp (config := { zeta := true }) [roundRatToIEEE32, hnumbeq, k, hkHi, hkUnder, hkSub,
              shift, numden, num',
              den', m, k', m']
            rfl
          -- If the carry-adjusted exponent overflows, the result is `±Inf`, contradicting `hfin`.
          by_cases hk'Hi : k' > 127
          · have hInf : roundRatToIEEE32 sign num den = (if sign then negInf else posInf) := by
              simp [hround, hk'Hi]
            have hfalse : isFinite (roundRatToIEEE32 sign num den) = false := by
              rw [hInf]
              cases sign <;> decide
            cases (hfalse.symm.trans hfin)
          · have hk'le : k' ≤ 127 := le_of_not_gt hk'Hi
            -- Name the fields for decoding.
            set expNat : Nat := Int.toNat (k' + 127)
            set fracNat : Nat := m' - pow2 23
            have hroundBits : roundRatToIEEE32 sign num den = ofBits (mkBits sign expNat fracNat) :=
              by
              simp [hround, hk'Hi, expNat, fracNat]

            have hkge : (-126 : Int) ≤ k := (not_lt).1 hkSub
            have hk'ge : (-126 : Int) ≤ k' := by
              by_cases hcarry : m == pow2 24
              · simp [k', hcarry]
                linarith [hkge]
              · simpa [k', hcarry] using hkge
            have hk'exp_nonneg : 0 ≤ k' + 127 := by linarith [hk'ge]
            have hk'exp_lt : k' + 127 < (255 : Int) := by linarith [hk'le]
            have hexp : expNat < 255 := by
              have h := (Int.toNat_lt_of_ne_zero (m := k' + 127) (n := 255) (by decide)).2 hk'exp_lt
              simpa [expNat] using h

            -- Compute the FP32 canonical exponent `k - 23`.
            have hmag :
                TorchLean.Floats.neuralMagnitude binaryRadix x = k + 1 := by
              simpa [x, r, k] using
                (neural_magnitude_signedRat (sign := sign) (num := num) (den := den) hnum hden)
            have hcexp :
                TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32 x = k - 23 := by
              have hk1_ge : (-149 : Int) ≤ k + 1 - 24 := by linarith [hkge]
              have hk123 : k + 1 - 24 = k - 23 := by linarith
              have h' :
                  TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32 x = k + 1 - 24
                    := by
                simp [TorchLean.Floats.neuralCexp, TorchLean.Floats.fexp32,
                  TorchLean.Floats.FLTExp, hmag,
                  max_eq_left hk1_ge]
              simpa [hk123] using h'
            have hscaled :
                TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 x =
                  x * neuralBpow binaryRadix (23 - k) := by
              simp [TorchLean.Floats.neuralScaledMantissa, hcexp]

            -- The positive scaled mantissa is a nonnegative rational `num'/den'`.
            have hden' : den' ≠ 0 := by
              cases hshift : shift with
              | ofNat sh =>
                  have hden'eq : den' = den := by
                    simp [den', numden, hshift]
                  simpa [hden'eq] using hden
              | negSucc sh =>
                  have hden'eq : den' = Nat.shiftLeft den (sh + 1) := by
                    simp [den', numden, hshift]
                  intro h0
                  have h0' : Nat.shiftLeft den (sh + 1) = 0 := by
                    simpa [hden'eq] using h0
                  have hmul : den * 2 ^ (sh + 1) = 0 := by
                    simpa [Nat.shiftLeft_eq] using h0'
                  have : den = 0 := by
                    have : den = 0 ∨ 2 ^ (sh + 1) = 0 := Nat.mul_eq_zero.mp hmul
                    cases this with
                    | inl h => exact h
                    | inr hpow =>
                        have hpos : 0 < 2 ^ (sh + 1) :=
                          Nat.pow_pos (a := 2) (n := sh + 1) (by decide : 0 < (2 : Nat))
                        have : False := (Nat.ne_of_gt hpos) hpow
                        exact False.elim this
                  exact (hden this).elim

            have hScalePos :
                r * neuralBpow binaryRadix (23 - k) = (num' : ℝ) / (den' : ℝ) := by
              cases hshift : shift with
              | ofNat sh =>
                  have hk : 23 - k = (Int.ofNat sh) := by
                    simpa [shift] using hshift
                  have hnumden : numden = (Nat.shiftLeft num sh, den) := by
                    simp [numden, hshift]
                  -- `r * 2^sh = (num <<< sh) / den`
                  have hbpow : neuralBpow binaryRadix (Int.ofNat sh) = (2 : ℝ) ^ sh := by
                    simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                  calc
                    r * neuralBpow binaryRadix (23 - k)
                        = (num : ℝ) / (den : ℝ) * neuralBpow binaryRadix (Int.ofNat sh) := by
                          simp [r, hk]
                    _ = (num : ℝ) / (den : ℝ) * (2 : ℝ) ^ sh := by
                          rw [hbpow]
                    _ = ((num : ℝ) * (2 : ℝ) ^ sh) / (den : ℝ) := by
                          simp [div_mul_eq_mul_div]
                    _ = (num' : ℝ) / (den' : ℝ) := by
                          have hnum' : (num' : ℝ) = (num : ℝ) * (2 : ℝ) ^ sh := by
                            -- `num' = num <<< sh`.
                            simp [hnumden, num', Nat.shiftLeft_eq, Nat.cast_mul, Nat.cast_pow]
                          have hden'' : (den' : ℝ) = (den : ℝ) := by
                            simp [hnumden, den']
                          simp [hnum', hden'']
              | negSucc sh =>
                  have hk : 23 - k = (Int.negSucc sh) := by
                    simpa [shift] using hshift
                  have hnumden : numden = (num, Nat.shiftLeft den (sh + 1)) := by
                    simp [numden, hshift]
                  have hbpow :
                      neuralBpow binaryRadix (Int.negSucc sh) = ((2 : ℝ) ^ (sh + 1))⁻¹ := by
                    simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                  calc
                    r * neuralBpow binaryRadix (23 - k)
                        = (num : ℝ) / (den : ℝ) * neuralBpow binaryRadix (Int.negSucc sh) := by
                          simp [r, hk]
                    _ = (num : ℝ) / (den : ℝ) * ((2 : ℝ) ^ (sh + 1))⁻¹ := by simp [hbpow]
                    _ = ((num : ℝ) / (den : ℝ)) / ((2 : ℝ) ^ (sh + 1)) := by
                          simp [div_eq_mul_inv]
                    _ = (num : ℝ) / ((den : ℝ) * (2 : ℝ) ^ (sh + 1)) := by
                          simp [div_div]
                    _ = (num' : ℝ) / (den' : ℝ) := by
                          simp [hnumden, num', den', Nat.shiftLeft_eq, Nat.cast_mul, Nat.cast_pow]

            have hRndPos :
                TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (23 - k)) =
                  Int.ofNat m := by
              have h :=
                neural_nearest_even_div_eq_roundQuotEven (num := num') (den := den') hden'
              simpa [m, hScalePos] using h

            have hRnd :
                TorchLean.Floats.neuralNearestEven (x * neuralBpow binaryRadix (23 - k)) =
                  if sign then -Int.ofNat m else Int.ofNat m := by
              cases hs : sign
              · have hx' : x = r := by simp [x, r, hs]
                simp [hx', hRndPos]
              · have hx' : x = -r := by simp [x, r, hs]
                have hscale : x * neuralBpow binaryRadix (23 - k) = -(r * neuralBpow binaryRadix
                  (23 - k)) := by
                  simp [hx']
                have hneg :
                    TorchLean.Floats.neuralNearestEven (-(r * neuralBpow binaryRadix (23 - k)))
                      =
                      -TorchLean.Floats.neuralNearestEven (r * neuralBpow binaryRadix (23 - k))
                        := by
                  simpa using (neural_nearest_even_neg (x := r * neuralBpow binaryRadix (23 - k)))
                simp [hscale, hneg, hRndPos]

            -- Bound the mantissa to show `fracNat < 2^23` in the non-carry branch.
            have hm_ge : pow2 23 ≤ m := by
              -- `2^23 ≤ r * 2^(23-k)` from `2^k ≤ r`.
              have hbounds := floorLog2Rat_bounds (num := num) (den := den) hnum hden
              have hbpos : 0 < neuralBpow binaryRadix (23 - k) := neuralBpow.pos binaryRadix (23
                - k)
              have hk_le_r : neuralBpow binaryRadix k ≤ r := by simpa [k, r] using hbounds.1
              have hmul := mul_le_mul_of_nonneg_right hk_le_r (le_of_lt hbpos)
              have hprod :
                  neuralBpow binaryRadix k * neuralBpow binaryRadix (23 - k) = neuralBpow
                    binaryRadix (23 : Int) := by
                have hadd := (neuralBpow.add_exp binaryRadix k (23 - k))
                have hsum : k + (23 - k) = (23 : Int) := by linarith
                simpa [hsum] using hadd.symm
              have hle : neuralBpow binaryRadix (23 : Int) ≤ r * neuralBpow binaryRadix (23 - k)
                := by
                simpa [hprod] using hmul
              have hmono :=
                (NeuralValidRnd.monotone (rnd := TorchLean.Floats.neuralNearestEven)
                  (x := (neuralBpow binaryRadix (23 : Int))) (y := r * neuralBpow binaryRadix
                    (23 - k)) hle)
              have hid :
                  TorchLean.Floats.neuralNearestEven (neuralBpow binaryRadix (23 : Int)) =
                    Int.ofNat (pow2 23) := by
                -- `bpow 23 = 2^23 = pow2 23`.
                have hbpow : neuralBpow binaryRadix (23 : Int) = (pow2 23 : ℝ) := by
                  have hcast : ((2 ^ 23 : Nat) : ℝ) = (pow2 23 : ℝ) := by
                    simpa using congrArg (fun n : Nat => (n : ℝ)) (pow2_eq_two_pow 23).symm
                  have hb : neuralBpow binaryRadix (23 : Int) = ((2 ^ 23 : Nat) : ℝ) := by
                    have hb' : neuralBpow binaryRadix (23 : Int) = (2 : ℝ) ^ 23 := by
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                        zpow_ofNat]
                    have hcastpow : ((2 ^ 23 : Nat) : ℝ) = (2 : ℝ) ^ 23 := by
                      simpa using (Nat.cast_pow (α := ℝ) 2 23)
                    exact hb'.trans hcastpow.symm
                  exact hb.trans hcast
                have : TorchLean.Floats.neuralNearestEven (pow2 23 : ℝ) = Int.ofNat (pow2 23) :=
                  by
                  simpa using
                    (TorchLean.Floats.NeuralValidRnd.id (rnd :=
                      TorchLean.Floats.neuralNearestEven) (Int.ofNat (pow2 23)))
                simpa [hbpow] using this
              have : Int.ofNat (pow2 23) ≤ Int.ofNat m := by
                simpa [hid, hRndPos] using hmono
              exact (Int.ofNat_le).1 this

            have hm_le : m ≤ pow2 24 := by
              -- `r * 2^(23-k) < 2^24` from `r < 2^(k+1)`.
              have hbounds := floorLog2Rat_bounds (num := num) (den := den) hnum hden
              have hbpos : 0 < neuralBpow binaryRadix (23 - k) := neuralBpow.pos binaryRadix (23
                - k)
              have hr_lt : r < neuralBpow binaryRadix (k + 1) := by simpa [k, r] using hbounds.2
              have hmul := mul_lt_mul_of_pos_right hr_lt hbpos
              have hprod :
                  neuralBpow binaryRadix (k + 1) * neuralBpow binaryRadix (23 - k) = neuralBpow
                    binaryRadix (24 : Int) := by
                have hadd := (neuralBpow.add_exp binaryRadix (k + 1) (23 - k))
                have hsum : (k + 1) + (23 - k) = (24 : Int) := by linarith
                simpa [hsum] using hadd.symm
              have hlt : r * neuralBpow binaryRadix (23 - k) < neuralBpow binaryRadix (24 : Int)
                := by
                simpa [hprod] using hmul
              have hle : r * neuralBpow binaryRadix (23 - k) ≤ neuralBpow binaryRadix (24 : Int)
                := le_of_lt hlt
              have hmono :=
                (NeuralValidRnd.monotone (rnd := TorchLean.Floats.neuralNearestEven)
                  (x := r * neuralBpow binaryRadix (23 - k)) (y := neuralBpow binaryRadix (24 :
                    Int)) hle)
              have hid :
                  TorchLean.Floats.neuralNearestEven (neuralBpow binaryRadix (24 : Int)) =
                    Int.ofNat (pow2 24) := by
                have hbpow : neuralBpow binaryRadix (24 : Int) = (pow2 24 : ℝ) := by
                  have hcast : ((2 ^ 24 : Nat) : ℝ) = (pow2 24 : ℝ) := by
                    simpa using congrArg (fun n : Nat => (n : ℝ)) (pow2_eq_two_pow 24).symm
                  have hb : neuralBpow binaryRadix (24 : Int) = ((2 ^ 24 : Nat) : ℝ) := by
                    have hb' : neuralBpow binaryRadix (24 : Int) = (2 : ℝ) ^ 24 := by
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                        zpow_ofNat]
                    have hcastpow : ((2 ^ 24 : Nat) : ℝ) = (2 : ℝ) ^ 24 := by
                      simpa using (Nat.cast_pow (α := ℝ) 2 24)
                    exact hb'.trans hcastpow.symm
                  exact hb.trans hcast
                have : TorchLean.Floats.neuralNearestEven (pow2 24 : ℝ) = Int.ofNat (pow2 24) :=
                  by
                  simpa using
                    (TorchLean.Floats.NeuralValidRnd.id (rnd :=
                      TorchLean.Floats.neuralNearestEven) (Int.ofNat (pow2 24)))
                simpa [hbpow] using this
              have : Int.ofNat m ≤ Int.ofNat (pow2 24) := by
                simpa [hid, hRndPos] using hmono
              exact (Int.ofNat_le).1 this

            have hfrac_lt : fracNat < 2 ^ 23 := by
              by_cases hcarryEq : m = pow2 24
              · have hb : (m == pow2 24) = true := by simp [hcarryEq]
                simp [fracNat, m', hb]
              · have hb : (m == pow2 24) = false := by
                  apply (Bool.eq_false_iff).2
                  intro hb'
                  have : m = pow2 24 := (beq_iff_eq).1 hb'
                  exact hcarryEq this
                have hm_lt : m < pow2 24 := lt_of_le_of_ne hm_le hcarryEq
                have hm'_eq : m' = m := by simp [m', hb]
                have hdiff : fracNat = m - pow2 23 := by simp [fracNat, hm'_eq]
                have hm_lt' : m - pow2 23 < pow2 23 := by
                  have hm_lt24 : m < 2 * pow2 23 := by
                    simpa [pow2_eq_two_pow, Nat.pow_succ] using hm_lt
                  -- `m < 2*2^23` implies `m - 2^23 < 2^23`.
                  have : m < pow2 23 + pow2 23 := by
                    simpa [two_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hm_lt24
                  exact Nat.sub_lt_left_of_lt_add hm_ge this
                simpa [hdiff, pow2_eq_two_pow] using hm_lt'

            -- Compute `toReal` for the executable result.
            have hto :
                toReal (roundRatToIEEE32 sign num den) =
                  (if sign then (-1 : ℝ) else 1) * (m' : ℝ) * neuralBpow binaryRadix (k' - 23) :=
                    by
              -- Decode the produced bits.
              have hdy :
                  toDyadic? (ofBits (mkBits sign expNat fracNat)) =
                    some { sign := sign, mant := pow2 23 + fracNat, exp := (Int.ofNat expNat) - 150
                      } := by
                have hexp' : expNat < 255 := by simpa using hexp
                have hfrac' : fracNat < 2 ^ 23 := by simpa using hfrac_lt
                have hexpNat0 : expNat ≠ 0 := by
                  intro h0
                  have hk127pos : 0 < k' + 127 := by linarith [hk'ge]
                  have hk127le : k' + 127 ≤ 0 := by
                    have : (k' + 127).toNat = 0 := by simpa [expNat] using h0
                    exact (Int.toNat_eq_zero).1 this
                  linarith
                simp [toDyadic?_ofBits_mkBits_fin (sign := sign) (exp := expNat) (frac := fracNat)
                  (hexp := hexp') (hfrac := hfrac'), hexpNat0]
              have hm'_mant : pow2 23 + fracNat = m' := by
                have hm'_ge : pow2 23 ≤ m' := by
                  by_cases hcarry : m == pow2 24
                  · simp [m', hcarry]
                  · have : m' = m := by simp [m', hcarry]
                    simpa [this] using hm_ge
                simp [fracNat, Nat.add_sub_of_le hm'_ge]
              have hexpInt : (expNat : Int) - 150 = k' - 23 := by
                -- `expNat = (k' + 127).toNat`, and `k' + 127 ≥ 0`.
                have hk'127 : (expNat : Int) = k' + 127 := by
                  simpa [expNat] using (Int.toNat_of_nonneg hk'exp_nonneg)
                linarith [hk'127]
              have htoBits : toReal (ofBits (mkBits sign expNat fracNat)) =
                  (if sign then (-1 : ℝ) else 1) * (m' : ℝ) * neuralBpow binaryRadix (k' - 23) :=
                    by
                -- Avoid `simp` timeouts by rewriting stepwise.
                rw [toReal_eq]
                simp [hdy, dyadicToReal, hm'_mant, hexpInt]
              simpa [hroundBits] using htoBits

            have hfp :
                fp32Round x =
                  (if sign then (-1 : ℝ) else 1) * (m : ℝ) * neuralBpow binaryRadix (k - 23) := by
              -- Unfold `fp32Round` using the computed mantissa and exponent.
              have hrnd :
                  ((TorchLean.Floats.rnd32
                          (TorchLean.Floats.neuralScaledMantissa binaryRadix
                            TorchLean.Floats.fexp32 x) : ℤ) : ℝ) =
                    (if sign then -(m : ℝ) else (m : ℝ)) := by
                have h0 :
                    TorchLean.Floats.neuralNearestEven
                        (TorchLean.Floats.neuralScaledMantissa binaryRadix
                          TorchLean.Floats.fexp32 x) =
                      if sign then -(m : ℤ) else (m : ℤ) := by
                  simpa [hscaled] using hRnd
                have h0' :
                    TorchLean.Floats.rnd32
                        (TorchLean.Floats.neuralScaledMantissa binaryRadix
                          TorchLean.Floats.fexp32 x) =
                      if sign then -(m : ℤ) else (m : ℤ) := by
                  simpa [TorchLean.Floats.rnd32] using h0
                cases hs : sign
                · -- `sign = false`
                  have h0'' :
                      ((TorchLean.Floats.rnd32
                              (TorchLean.Floats.neuralScaledMantissa binaryRadix
                                TorchLean.Floats.fexp32 x) : ℤ) : ℝ) =
                        (m : ℝ) := by
                    have : TorchLean.Floats.rnd32
                            (TorchLean.Floats.neuralScaledMantissa binaryRadix
                              TorchLean.Floats.fexp32 x) =
                          (m : ℤ) := by
                      simpa [hs] using h0'
                    simpa using congrArg (fun z : ℤ => (z : ℝ)) this
                  simpa [hs] using h0''
                · -- `sign = true`
                  have h0'' :
                      ((TorchLean.Floats.rnd32
                              (TorchLean.Floats.neuralScaledMantissa binaryRadix
                                TorchLean.Floats.fexp32 x) : ℤ) : ℝ) =
                        -(m : ℝ) := by
                    have : TorchLean.Floats.rnd32
                            (TorchLean.Floats.neuralScaledMantissa binaryRadix
                              TorchLean.Floats.fexp32 x) =
                          -(m : ℤ) := by
                      simpa [hs] using h0'
                    simpa using congrArg (fun z : ℤ => (z : ℝ)) this
                  simpa [hs] using h0''
              -- Finish without simp-canceling products.
              calc
                fp32Round x =
                    ((TorchLean.Floats.rnd32
                            (TorchLean.Floats.neuralScaledMantissa binaryRadix
                              TorchLean.Floats.fexp32 x) : ℤ) : ℝ) *
                      neuralBpow binaryRadix (k - 23) := by
                      simp [fp32Round, TorchLean.Floats.neuralRound,
                        TorchLean.Floats.neuralToReal, hcexp]
                _ =
                    (if sign then -(m : ℝ) else (m : ℝ)) * neuralBpow binaryRadix (k - 23) := by
                      simp [hrnd]
                _ =
                    (if sign then (-1 : ℝ) else 1) * (m : ℝ) * neuralBpow binaryRadix (k - 23) :=
                      by
                      cases sign <;> simp []

            -- Carry adjustment: `m' * 2^(k'-23) = m * 2^(k-23)` as reals.
            have hadj :
                (m' : ℝ) * neuralBpow binaryRadix (k' - 23) =
                  (m : ℝ) * neuralBpow binaryRadix (k - 23) := by
              by_cases hcarryEq : m = pow2 24
              · have hb : (m == pow2 24) = true := by simp [hcarryEq]
                have hm' : m' = pow2 23 := by simp [m', hb]
                have hk' : k' = k + 1 := by simp [k', hb]
                -- Rewrite everything into the canonical carry form.
                rw [hm', hk', hcarryEq]
                have hk : (k + 1 : Int) - 23 = (k - 23) + 1 := by linarith
                rw [hk, neuralBpow.add_exp]
                have htwo : neuralBpow binaryRadix (1 : Int) = (2 : ℝ) := by
                  simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                rw [htwo]
                have hp : ((pow2 24 : Nat) : ℝ) = (2 : ℝ) * (pow2 23 : ℝ) := by
                  simp [pow2_eq_two_pow, Nat.pow_succ]
                  norm_num
                -- Now it's pure algebra.
                simp [hp, mul_assoc, mul_left_comm, mul_comm]
              · have hb : (m == pow2 24) = false := by
                  apply (Bool.eq_false_iff).2
                  intro hb'
                  have : m = pow2 24 := (beq_iff_eq).1 hb'
                  exact hcarryEq this
                simp [k', m', hb]

            rw [hto, hfp]
            -- Push the carry adjustment through the sign factor.
            cases hs : sign <;> simp [hadj, mul_comm]

/-!
## Op-level refinement theorems (finite/no-overflow)

These are the results that the rest of TorchLean typically consumes: statements that each arithmetic
operation in `IEEE32Exec` refines its `FP32` real-rounded counterpart.

If you are coming from PyTorch: this is the “float32 math model” that underlies many informal
numerical arguments (“the kernel computes the exact real result, then rounds to float32”),
but made explicit and proved for our executable kernel.
-/

/-- Finite refinement for addition: `IEEE32Exec.add` = exact real add + float32 rounding. -/
theorem toReal_add_eq_fp32Round (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy)
    (hfin : isFinite (add x y) = true) :
    toReal (add x y) = fp32Round (toReal x + toReal y) := by
  have hadd : add x y = roundDyadicToIEEE32 (addDyadic dx dy) :=
    add_eq_roundDyadicToIEEE32_of_toDyadic? (hx := hx) (hy := hy)
  have hfin' : isFinite (roundDyadicToIEEE32 (addDyadic dx dy)) = true := by
    simpa [hadd] using hfin
  calc
    toReal (add x y) = toReal (roundDyadicToIEEE32 (addDyadic dx dy)) := by
      simp [hadd]
    _ = fp32Round (dyadicToReal (addDyadic dx dy)) := by
      simpa using (toReal_roundDyadicToIEEE32_eq_fp32Round (d := addDyadic dx dy) hfin')
    _ = fp32Round (dyadicToReal dx + dyadicToReal dy) := by
      rw [dyadicToReal_addDyadic_exact (a := dx) (b := dy)]
    _ = fp32Round (toReal x + toReal y) := by
      simp [toReal_eq, hx, hy]

/-- Finite refinement for subtraction, reduced to addition + negation. -/
theorem toReal_sub_eq_fp32Round (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy)
    (hfin : isFinite (sub x y) = true) :
    toReal (sub x y) = fp32Round (toReal x - toReal y) := by
  classical
  -- `sub x y` is defined as `add x (neg y)`.
  let dyNeg : Dyadic := { sign := (!dy.sign), mant := dy.mant, exp := dy.exp }
  have hyNeg : toDyadic? (neg y) = some dyNeg := by
    simpa [dyNeg] using (toDyadic?_neg_of_toDyadic?_some (x := y) (d := dy) hy)
  have hfin' : isFinite (add x (neg y)) = true := by simpa [sub] using hfin
  have hadd :
      toReal (add x (neg y)) = fp32Round (toReal x + toReal (neg y)) := by
    simpa [dyNeg] using
      (toReal_add_eq_fp32Round (x := x) (y := neg y) (dx := dx) (dy := dyNeg) hx hyNeg hfin')
  have hnegReal : toReal (neg y) = -toReal y := toReal_neg_eq_neg (x := y) (d := dy) hy
  calc
    toReal (sub x y) = fp32Round (toReal x + toReal (neg y)) := by
      simpa [sub] using hadd
    _ = fp32Round (toReal x - toReal y) := by
      simp [hnegReal, sub_eq_add_neg]

/-- Finite refinement for multiplication: `IEEE32Exec.mul` = exact real mul + float32 rounding. -/
theorem toReal_mul_eq_fp32Round (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy)
    (hfin : isFinite (mul x y) = true) :
    toReal (mul x y) = fp32Round (toReal x * toReal y) := by
  let prod : Dyadic :=
    { sign := Bool.xor dx.sign dy.sign, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
  have hmul : mul x y = roundDyadicToIEEE32 prod :=
    mul_eq_roundDyadicToIEEE32_of_toDyadic? (hx := hx) (hy := hy)
  have hfin' : isFinite (roundDyadicToIEEE32 prod) = true := by
    simpa [hmul] using hfin
  calc
    toReal (mul x y) = toReal (roundDyadicToIEEE32 prod) := by
      simp [hmul]
    _ = fp32Round (dyadicToReal prod) := by
      simpa using (toReal_roundDyadicToIEEE32_eq_fp32Round (d := prod) hfin')
    _ = fp32Round (dyadicToReal dx * dyadicToReal dy) := by
      -- exact dyadic product semantics
      simpa [prod] using congrArg fp32Round (dyadicToReal_mul_exact (a := dx) (b := dy))
    _ = fp32Round (toReal x * toReal y) := by
      simp [toReal_eq, hx, hy]

/-- Finite refinement for fused multiply-add: `fma x y z` rounds `x*y + z` once at the end. -/
theorem toReal_fma_eq_fp32Round (x y z : IEEE32Exec) {dx dy dz : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) (hz : toDyadic? z = some dz)
    (hfin : isFinite (fma x y z) = true) :
    toReal (fma x y z) = fp32Round (toReal x * toReal y + toReal z) := by
  let prod : Dyadic :=
    { sign := Bool.xor dx.sign dy.sign, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
  have hfma : fma x y z = roundDyadicToIEEE32 (addDyadic prod dz) :=
    fma_eq_roundDyadicToIEEE32_of_toDyadic? (hx := hx) (hy := hy) (hz := hz)
  have hfin' : isFinite (roundDyadicToIEEE32 (addDyadic prod dz)) = true := by
    simpa [hfma] using hfin
  calc
    toReal (fma x y z) = toReal (roundDyadicToIEEE32 (addDyadic prod dz)) := by
      simp [hfma]
    _ = fp32Round (dyadicToReal (addDyadic prod dz)) := by
      simpa using (toReal_roundDyadicToIEEE32_eq_fp32Round (d := addDyadic prod dz) hfin')
    _ = fp32Round (dyadicToReal prod + dyadicToReal dz) := by
      rw [dyadicToReal_addDyadic_exact (a := prod) (b := dz)]
    _ = fp32Round (dyadicToReal dx * dyadicToReal dy + dyadicToReal dz) := by
      -- dyadic product semantics inside the sum
      simpa [prod] using
        congrArg fp32Round
          (congrArg (fun r : ℝ => r + dyadicToReal dz) (dyadicToReal_mul_exact (a := dx) (b := dy)))
    _ = fp32Round (toReal x * toReal y + toReal z) := by
      simp [toReal_eq, hx, hy, hz]

/--
Finite refinement for division.

At the executable level, division is implemented by forming an exact rational quotient `num/den`
(after aligning dyadic exponents) and then rounding that rational to float32. This theorem states
that the overall real meaning is `FP32` rounding of real division.
-/
theorem toReal_div_eq_fp32Round (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) (hy0 : dy.mant ≠ 0)
    (hfin : isFinite (div x y) = true) :
    toReal (div x y) = fp32Round (toReal x / toReal y) := by
  -- Reduce IEEE32 division to rounding an exact rational quotient.
  have hdiv :
      div x y =
        let sign : Bool := Bool.xor dx.sign dy.sign
        let eDiff : Int := dx.exp - dy.exp
        let (num, den) :=
          match eDiff with
          | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
          | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
        roundRatToIEEE32 sign num den :=
    div_eq_roundRatToIEEE32_of_toDyadic? (hx := hx) (hy := hy) hy0
  have hfin' :
      isFinite
          (let sign : Bool := Bool.xor dx.sign dy.sign
            let eDiff : Int := dx.exp - dy.exp
            let (num, den) :=
              match eDiff with
              | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
              | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
            roundRatToIEEE32 sign num den) = true := by
    simpa [hdiv] using hfin
  -- Real semantics of the exact quotient is `toReal x / toReal y`.
  let sign : Bool := Bool.xor dx.sign dy.sign
  cases hE : (dx.exp - dy.exp) with
  | ofNat sh =>
      let num : Nat := Nat.shiftLeft dx.mant sh
      let den : Nat := dy.mant
      have hden0 : den ≠ 0 := by
        simpa [den] using hy0
      have htoRat :
          toReal x / toReal y = (if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ)) := by
        have hrat := dyadicToReal_div_eq_signedRat_mul (dx := dx) (dy := dy) hy0
        have hrat' :
            dyadicToReal dx / dyadicToReal dy =
              (if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ)) := by
          simpa (config := { zeta := true }) [sign, hE, num, den] using hrat
        simpa [toReal_eq, hx, hy] using hrat'
      have hfinCase : isFinite (roundRatToIEEE32 sign num den) = true := by
        simpa (config := { zeta := true }) [sign, hE, num, den] using hfin'
      -- Apply the rounding refinement theorem.
      calc
        toReal (div x y) = toReal (roundRatToIEEE32 sign num den) := by
          simp (config := { zeta := true }) [hdiv, sign, hE, num, den]
        _ = fp32Round ((if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ))) := by
          simpa using (toReal_roundRatToIEEE32_eq_fp32Round (sign := sign) (num := num) (den := den)
            hden0 hfinCase)
        _ = fp32Round (toReal x / toReal y) := by
          rw [htoRat]
  | negSucc sh =>
      let num : Nat := dx.mant
      let den : Nat := Nat.shiftLeft dy.mant (sh + 1)
      have hden0 : den ≠ 0 := by
        intro h0
        have hmul : dy.mant * 2 ^ (sh + 1) = 0 := by
          simpa [den, Nat.shiftLeft_eq] using h0
        have : dy.mant = 0 := by
          have : dy.mant = 0 ∨ 2 ^ (sh + 1) = 0 := Nat.mul_eq_zero.mp hmul
          cases this with
          | inl h => exact h
          | inr hpow =>
              have hpos : 0 < 2 ^ (sh + 1) :=
                Nat.pow_pos (a := 2) (n := sh + 1) (by decide : 0 < (2 : Nat))
              exact False.elim ((Nat.ne_of_gt hpos) hpow)
        exact (hy0 this).elim
      have htoRat :
          toReal x / toReal y = (if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ)) := by
        have hrat := dyadicToReal_div_eq_signedRat_mul (dx := dx) (dy := dy) hy0
        have hrat' :
            dyadicToReal dx / dyadicToReal dy =
              (if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ)) := by
          simpa (config := { zeta := true }) [sign, hE, num, den] using hrat
        simpa [toReal_eq, hx, hy] using hrat'
      have hfinCase : isFinite (roundRatToIEEE32 sign num den) = true := by
        simpa (config := { zeta := true }) [sign, hE, num, den] using hfin'
      calc
        toReal (div x y) = toReal (roundRatToIEEE32 sign num den) := by
          simp (config := { zeta := true }) [hdiv, sign, hE, num, den]
        _ = fp32Round ((if sign then (-1 : ℝ) else 1) * ((num : ℝ) / (den : ℝ))) := by
          simpa using (toReal_roundRatToIEEE32_eq_fp32Round (sign := sign) (num := num) (den := den)
            hden0 hfinCase)
        _ = fp32Round (toReal x / toReal y) := by
          rw [htoRat]

/--
Finite refinement for square root.

`IEEE32Exec.sqrt` computes an executable approximation and then rounds it to float32. This bridge
theorem states that, on the finite path, the real meaning agrees with `FP32` rounding of
`Real.sqrt`.
-/
theorem toReal_sqrt_eq_fp32Round (x : IEEE32Exec) {dx : Dyadic}
    (hx : toDyadic? x = some dx)
    (hfin : isFinite (sqrt x) = true) :
    toReal (sqrt x) = fp32Round (Real.sqrt (toReal x)) := by
  classical
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hchoose : chooseNaN1 x = none := by simp [chooseNaN1, hxNaN]

  by_cases hz : isZero x = true
  · have hsqrt : sqrt x = x := by simp [IEEE32Exec.sqrt, hchoose, hxInf, hz]
    have hfields : (expField x == 0) = true ∧ (fracField x == 0) = true := by
      simpa [IEEE32Exec.isZero, Bool.and_eq_true] using hz
    have hnaninf : (isNaN x || isInf x) = false := by simp [hxNaN, hxInf]
    have hx0_expected : toDyadic? x = some { sign := signBit x, mant := 0, exp := 0 } := by
      unfold toDyadic?
      simp (config := { zeta := true }) [hnaninf, hfields.1, hfields.2]
    have hdx0 : dx = { sign := signBit x, mant := 0, exp := 0 } := by
      have : (some dx : Option Dyadic) = some { sign := signBit x, mant := 0, exp := 0 } := by
        simpa [hx] using hx0_expected
      exact Option.some.inj this
    have hx0 : toReal x = 0 := by
      simp [toReal_eq, hx, dyadicToReal, hdx0]
    have hfp0 : fp32Round 0 = 0 := by
      have hne0 : TorchLean.Floats.neuralNearestEven 0 = 0 := by
        simp [TorchLean.Floats.neuralNearestEven]
      have : TorchLean.Floats.neuralNearestEven 0 = 0 ∨ TorchLean.Floats.neuralBpow binaryRadix
        (-24) = 0 :=
        Or.inl hne0
      simpa [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal,
        TorchLean.Floats.neuralScaledMantissa, TorchLean.Floats.neuralCexp,
          TorchLean.Floats.neuralMagnitude,
        TorchLean.Floats.fexp32, TorchLean.Floats.FLTExp, TorchLean.Floats.rnd32] using this
    rw [hsqrt, hx0]
    simp [hfp0]
  · have hz' : isZero x = false := by simpa using hz
    cases hs : signBit x with
    | true =>
        have hbad : isFinite (sqrt x) = false := by
          -- `sqrt` of a negative, finite, nonzero input is `canonicalNaN`.
          simp [IEEE32Exec.sqrt, hchoose, hxInf, hz', hs]
          decide
        have : False := by
          simp [hbad] at hfin
        exact False.elim this
    | false =>
        have hnaninf : (isNaN x || isInf x) = false := by simp [hxNaN, hxInf]
        have hx_dec0 := hx
        unfold toDyadic? at hx_dec0
        simp (config := { zeta := true, failIfUnchanged := false }) [hnaninf, hs] at hx_dec0

        -- Decoder bounds: `dx.sign = false`, `dx.mant ≠ 0`, `dx.mant < 2^24`, `-149 ≤ dx.exp ≤
        -- 104`.
        have hdx_sign : dx.sign = false := by
          by_cases hE : expField x = 0
          · by_cases hF : fracField x = 0
            · have hdx : dx = { sign := false, mant := 0, exp := 0 } := by
                simpa [hE, hF] using hx_dec0.symm
              cases hdx
              rfl
            · have hdx : dx = { sign := false, mant := (fracField x).toNat, exp := (-149 : Int) } :=
              by
                simpa [hE, hF] using hx_dec0.symm
              cases hdx
              rfl
          · have hdx :
                dx =
                  { sign := false
                    mant := pow2 23 + (fracField x).toNat
                    exp := (Int.ofNat (expField x).toNat) - 150 } := by
              simpa [hE] using hx_dec0.symm
            cases hdx
            rfl
        have hfracMask : (fracMask.toNat : Nat) = 2 ^ 23 - 1 := by decide
        have hfrac_le : (fracField x).toNat ≤ fracMask.toNat := by
          -- `fracField x = x.bits &&& fracMask`, so its nat value is `x.bits.toNat &&&
          -- fracMask.toNat`,
          -- which is bitwise-≤ `fracMask.toNat`.
          simp [fracField, UInt32.toNat_and]
          apply Nat.le_of_testBit
          intro i hi
          have hi' : Nat.testBit x.bits.toNat i = true ∧ Nat.testBit fracMask.toNat i = true := by
            simpa [Nat.testBit_land, Bool.and_eq_true] using hi
          exact hi'.2
        have hfrac_lt : (fracField x).toNat < 2 ^ 23 := by
          have : fracMask.toNat < 2 ^ 23 := by
            rw [hfracMask]
            have hpos : 0 < (2 ^ 23 : Nat) := Nat.pow_pos (Nat.succ_pos 1)
            exact Nat.sub_lt hpos (Nat.succ_pos 0)
          exact lt_of_le_of_lt hfrac_le this

        have hdx_mant_ne0 : dx.mant ≠ 0 := by
          intro hm0
          by_cases hE : expField x = 0
          · by_cases hF : fracField x = 0
            · have hE' : (expField x == 0) = true := (beq_iff_eq).2 hE
              have hF' : (fracField x == 0) = true := (beq_iff_eq).2 hF
              have hz0 : isZero x = true := by simp [isZero, hE', hF']
              have hzF : False := by
                have hz'' := hz'
                simp [hz0] at hz''
              exact hzF
            · have hdx : dx = { sign := false, mant := (fracField x).toNat, exp := (-149 : Int) } :=
              by
                simpa [hE, hF] using hx_dec0.symm
              have hm : dx.mant = (fracField x).toNat := by
                simpa using congrArg Dyadic.mant hdx
              have hf0 : (fracField x).toNat = 0 := by simpa [hm] using hm0
              have : fracField x = 0 := by
                apply UInt32.toNat_inj.1
                simpa [UInt32.toNat_ofNat] using hf0
              have hFbeq : (fracField x == 0) = true := by simpa using (beq_iff_eq).2 this
              exact (hF ((beq_iff_eq).1 hFbeq)).elim
          · have hdx :
                dx =
                  { sign := false
                    mant := pow2 23 + (fracField x).toNat
                    exp := (Int.ofNat (expField x).toNat) - 150 } := by
              simpa [hE] using hx_dec0.symm
            have hm : dx.mant = pow2 23 + (fracField x).toNat := by
              simpa using congrArg Dyadic.mant hdx
            have hpow23pos : 0 < pow2 23 := by
              rw [pow2_eq_two_pow]
              exact Nat.pow_pos (Nat.succ_pos 1)
            have : 0 < dx.mant := by simpa [hm] using Nat.add_pos_left hpow23pos (fracField x).toNat
            exact (ne_of_gt this) hm0

        have hdx_mant_lt : dx.mant < 2 ^ 24 := by
          by_cases hE : expField x = 0
          · by_cases hF : fracField x = 0
            · have hE' : (expField x == 0) = true := (beq_iff_eq).2 hE
              have hF' : (fracField x == 0) = true := (beq_iff_eq).2 hF
              have hz0 : isZero x = true := by simp [isZero, hE', hF']
              have hzF : False := by
                have hz'' := hz'
                simp [hz0] at hz''
              exact False.elim hzF
            · have hdx : dx = { sign := false, mant := (fracField x).toNat, exp := (-149 : Int) } :=
              by
                simpa [hE, hF] using hx_dec0.symm
              have hm : dx.mant = (fracField x).toNat := by
                simpa using congrArg Dyadic.mant hdx
              have : (fracField x).toNat < 2 ^ 24 :=
                lt_trans hfrac_lt (Nat.pow_lt_pow_right (by decide : 1 < (2 : Nat)) (by decide : (23
                  : Nat) < 24))
              simpa [hm] using this
          · have hdx :
                dx =
                  { sign := false
                    mant := pow2 23 + (fracField x).toNat
                    exp := (Int.ofNat (expField x).toNat) - 150 } := by
              simpa [hE] using hx_dec0.symm
            have hm : dx.mant = pow2 23 + (fracField x).toNat := by
              simpa using congrArg Dyadic.mant hdx
            have hpow23 : pow2 23 = 2 ^ 23 := pow2_eq_two_pow 23
            have : pow2 23 + (fracField x).toNat < pow2 24 := by
              have : (fracField x).toNat ≤ 2 ^ 23 - 1 := by
                rw [← hfracMask]
                exact hfrac_le
              have hsum : pow2 23 + (fracField x).toNat ≤ pow2 23 + (2 ^ 23 - 1) := by
                have h := Nat.add_le_add_right this (pow2 23)
                -- `add_le_add_right` produces `a + c ≤ b + c`; commute to match our normal form.
                simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using h
              have hconst : (2 ^ 23) + (2 ^ 23 - 1) < 2 ^ 24 := by
                decide
              have hpow24 : pow2 24 = 2 ^ 24 := pow2_eq_two_pow 24
              have : pow2 23 + (2 ^ 23 - 1) < pow2 24 := by
                rw [hpow23, hpow24]
                exact hconst
              exact lt_of_le_of_lt hsum this
            simpa [hm, pow2_eq_two_pow 24] using this

        have hdx_exp_lo : (-149 : Int) ≤ dx.exp := by
          by_cases hE : expField x = 0
          · by_cases hF : fracField x = 0
            · have hE' : (expField x == 0) = true := (beq_iff_eq).2 hE
              have hF' : (fracField x == 0) = true := (beq_iff_eq).2 hF
              have hz0 : isZero x = true := by simp [isZero, hE', hF']
              have hzF : False := by
                have hz'' := hz'
                simp [hz0] at hz''
              exact False.elim hzF
            · have hdx : dx = { sign := false, mant := (fracField x).toNat, exp := (-149 : Int) } :=
              by
                simpa [hE, hF] using hx_dec0.symm
              have he : dx.exp = (-149 : Int) := by
                simpa using congrArg Dyadic.exp hdx
              simp [he]
          · have hdx :
                dx =
                  { sign := false
                    mant := pow2 23 + (fracField x).toNat
                    exp := (Int.ofNat (expField x).toNat) - 150 } := by
              simpa [hE] using hx_dec0.symm
            have he : dx.exp = (Int.ofNat (expField x).toNat) - 150 := by
              simpa using congrArg Dyadic.exp hdx
            have he_toNat_pos : 0 < (expField x).toNat := by
              have : (expField x).toNat ≠ 0 := by
                intro h0
                have : expField x = 0 := by
                  apply UInt32.toNat_inj.1
                  simpa using h0
                exact (hE this).elim
              exact Nat.pos_of_ne_zero this
            have : (-149 : Int) ≤ (Int.ofNat (expField x).toNat) - 150 := by
              have hnat : (1 : Nat) ≤ (expField x).toNat := Nat.succ_le_of_lt he_toNat_pos
              have : (1 : Int) ≤ (Int.ofNat (expField x).toNat) := by
                exact (Int.ofNat_le.2 hnat)
              linarith
            simpa [he]

        have hdx_exp_hi : dx.exp ≤ 104 := by
          by_cases hE : expField x = 0
          · by_cases hF : fracField x = 0
            · have hE' : (expField x == 0) = true := (beq_iff_eq).2 hE
              have hF' : (fracField x == 0) = true := (beq_iff_eq).2 hF
              have hz0 : isZero x = true := by simp [isZero, hE', hF']
              have hzF : False := by
                have hz'' := hz'
                simp [hz0] at hz''
              exact False.elim hzF
            · have hdx : dx = { sign := false, mant := (fracField x).toNat, exp := (-149 : Int) } :=
              by
                simpa [hE, hF] using hx_dec0.symm
              have he : dx.exp = (-149 : Int) := by
                simpa using congrArg Dyadic.exp hdx
              linarith [he]
          · -- normal: exponent field is at most 254.
            have hexpAll : expField x ≠ expAllOnes := by
              intro hEq
              have hEqb : (expField x == expAllOnes) = true := (beq_iff_eq).2 hEq
              by_cases hf0 : fracField x == 0
              · have hinf : isInf x = true := by simp [isInf, hEqb, hf0]
                have hxF : False := by
                  have hxInf' := hxInf
                  simp [hinf] at hxInf'
                exact hxF
              · have hne : (fracField x != 0) = true := by
                  cases hneq : (fracField x != 0) <;> try rfl
                  have : fracField x = 0 := (bne_eq_false_iff_eq).1 hneq
                  have : (fracField x == 0) = true := (beq_iff_eq).2 this
                  have hfF : False := by
                    have hf0' := hf0
                    simp [this] at hf0'
                  exact False.elim hfF
                have hnan : isNaN x = true := by simp [isNaN, hEqb, hne]
                have hxF : False := by
                  have hxNaN' := hxNaN
                  simp [hnan] at hxNaN'
                exact hxF
            have hexp_le255 : (expField x).toNat ≤ 255 := by
              have : (expAllOnes.toNat : Nat) = 255 := by decide
              have hle : (expField x).toNat ≤ expAllOnes.toNat := by
                simp [expField, UInt32.toNat_and]
                apply Nat.le_of_testBit
                intro i hi
                have hi' :
                    Nat.testBit ((x.bits >>> 23).toNat) i = true ∧ Nat.testBit expAllOnes.toNat i =
                      true := by
                  simpa [Nat.testBit_land, Bool.and_eq_true] using hi
                exact hi'.2
              simpa [this] using hle
            have hexp_ne255 : (expField x).toNat ≠ 255 := by
              intro h0
              have : expField x = expAllOnes := by
                apply UInt32.toNat_inj.1
                simpa [UInt32.toNat_ofNat] using h0
              exact hexpAll this
            have hexp_lt255 : (expField x).toNat < 255 := lt_of_le_of_ne hexp_le255 hexp_ne255
            have hexp_le254 : (expField x).toNat ≤ 254 := Nat.le_of_lt_succ (by simpa using
              hexp_lt255)
            have hdx :
                dx =
                  { sign := false
                    mant := pow2 23 + (fracField x).toNat
                    exp := (Int.ofNat (expField x).toNat) - 150 } := by
              simpa [hE] using hx_dec0.symm
            have he : dx.exp = (Int.ofNat (expField x).toNat) - 150 := by
              simpa using congrArg Dyadic.exp hdx
            have : (Int.ofNat (expField x).toNat) - 150 ≤ 104 := by
              have : (Int.ofNat (expField x).toNat) ≤ (254 : Int) := by
                exact (Int.ofNat_le.2 hexp_le254)
              linarith
            simpa [he] using this

        -- Unfold the sqrt algorithm into named intermediates (same as `Exec32.sqrt`).
        set expOdd : Bool := (dx.exp % 2) != 0 with hexpOdd_def
        set mant' : Nat := if expOdd then dx.mant * 2 else dx.mant with hmant'_def
        set expEven : Int := if expOdd then dx.exp - 1 else dx.exp with hexpEven_def
        set expHalf : Int := expEven / 2 with hexpHalf_def
        set l : Nat := Nat.log2 mant' with hl_def
        set t : Nat := l / 2 with ht_def
        set p : Nat := 23 - t with hp_def
        set n : Nat := Nat.shiftLeft mant' (2 * p) with hn_def
        set q : Nat := Nat.sqrt n with hq_def
        set r : Nat := n - q * q with hr_def
        set m0 : Nat := if r > q then q + 1 else q with hm0_def
        set k0 : Int := expHalf + Int.ofNat t with hk0_def
        set k : Int := if m0 == pow2 24 then k0 + 1 else k0 with hk_def
        set m24 : Nat := if m0 == pow2 24 then pow2 23 else m0 with hm24_def
        set expNat : Nat := Int.toNat (k + 127) with hexpNat_def
        set fracNat : Nat := m24 - pow2 23 with hfracNat_def

        have hsqrt_bits : sqrt x = ofBits (mkBits false expNat fracNat) := by
          simp (config := { zeta := true }) [IEEE32Exec.sqrt, hchoose, hxInf, hz', hs, hx,
            hexpOdd_def, hmant'_def,
            hexpEven_def, hexpHalf_def, hl_def, ht_def, hp_def, hn_def, hq_def, hr_def, hm0_def,
              hk0_def, hk_def,
            hm24_def, hexpNat_def, hfracNat_def]

        -- Mantissa bounds: `2^23 ≤ m0 ≤ 2^24` and `2^23 ≤ m24 < 2^24`.
        have hmant'_ne0 : mant' ≠ 0 := by
          cases hOdd : expOdd with
          | false =>
              simpa [hmant'_def, hOdd] using hdx_mant_ne0
          | true =>
              simpa [hmant'_def, hOdd] using
                Nat.mul_ne_zero hdx_mant_ne0 (by decide : (2 : Nat) ≠ 0)
        have hmant'_lt25 : mant' < 2 ^ 25 := by
          cases hOdd : expOdd with
          | false =>
              have : dx.mant < 2 ^ 25 :=
                lt_trans hdx_mant_lt
                  (Nat.pow_lt_pow_right (by decide : 1 < (2 : Nat)) (by decide : (24 : Nat) < 25))
              simpa [hmant'_def, hOdd] using this
          | true =>
              have : dx.mant * 2 < (2 ^ 24) * 2 := Nat.mul_lt_mul_of_pos_right hdx_mant_lt (by
                decide)
              have : dx.mant * 2 < 2 ^ 25 := by
                simpa [Nat.pow_succ, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using this
              simpa [hmant'_def, hOdd] using this
        have hl_le24 : l ≤ 24 := by
          have hl_lt25 : l < 25 := (Nat.log2_lt hmant'_ne0).2 hmant'_lt25
          exact Nat.le_of_lt_succ (by simpa using hl_lt25)
        have ht_le12 : t ≤ 12 := by
          have : l / 2 ≤ 24 / 2 := Nat.div_le_div_right hl_le24
          simpa [ht_def] using this
        have ht_le23 : t ≤ 23 := le_trans ht_le12 (by decide : (12 : Nat) ≤ 23)

        have hpow_le : 2 ^ l ≤ mant' := (Nat.le_log2 hmant'_ne0).1 le_rfl
        have hpow_hi : mant' < 2 ^ (l + 1) := (Nat.log2_lt hmant'_ne0).1 (Nat.lt_succ_self l)
        have ht2_le : 2 * t ≤ l := by
          have : 2 * (l / 2) ≤ l := by
            simpa [Nat.mul_comm] using (Nat.mul_div_le l 2)
          simpa [ht_def] using this
        have hmant_low : 2 ^ (2 * t) ≤ mant' := by
          exact le_trans (Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) ht2_le) hpow_le
        have hl1_le : l + 1 ≤ 2 * (t + 1) := by
          have hrem : l % 2 < 2 := Nat.mod_lt l (by decide : 0 < (2 : Nat))
          have hdecomp : 2 * (l / 2) + l % 2 = l := by
            simpa [Nat.add_comm, Nat.mul_comm] using (Nat.mod_add_div l 2)
          have hl_lt : l < 2 * (l / 2) + 2 := by
            have : 2 * (l / 2) + l % 2 < 2 * (l / 2) + 2 := Nat.add_lt_add_left hrem _
            simpa [hdecomp] using this
          have : l + 1 ≤ 2 * (l / 2) + 2 := Nat.succ_le_of_lt hl_lt
          simpa [ht_def, Nat.mul_add, Nat.mul_one, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm]
            using this
        have hmant_hi : mant' < 2 ^ (2 * (t + 1)) := by
          have : 2 ^ (l + 1) ≤ 2 ^ (2 * (t + 1)) :=
            Nat.pow_le_pow_right (by decide : (2 : Nat) > 0) hl1_le
          exact lt_of_lt_of_le hpow_hi this

        have ht_p : t + p = 23 := by
          simpa [hp_def] using (Nat.add_sub_of_le ht_le23)
        have hsum_tp : 2 * t + 2 * p = 46 := by
          calc
            2 * t + 2 * p = 2 * (t + p) := by
              simp [Nat.mul_add]
            _ = 46 := by simp [ht_p]
        have hn_mul : n = mant' * 2 ^ (2 * p) := by
          simp [hn_def, Nat.shiftLeft_eq]
        have hn_lo : 2 ^ 46 ≤ n := by
          have hmul : 2 ^ (2 * t) * 2 ^ (2 * p) ≤ mant' * 2 ^ (2 * p) :=
            Nat.mul_le_mul_right (2 ^ (2 * p)) hmant_low
          have : 2 ^ (2 * t + 2 * p) ≤ mant' * 2 ^ (2 * p) := by
            simpa [Nat.pow_add, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using hmul
          simpa [hn_mul, hsum_tp] using this
        have hn_hi : n < 2 ^ 48 := by
          have hmul : mant' * 2 ^ (2 * p) < 2 ^ (2 * (t + 1)) * 2 ^ (2 * p) :=
            Nat.mul_lt_mul_of_pos_right hmant_hi (Nat.pow_pos (by decide : 0 < (2 : Nat)))
          have : mant' * 2 ^ (2 * p) < 2 ^ (2 * (t + 1) + 2 * p) := by
            simpa [Nat.pow_add, Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using hmul
          have hsum_tp' : 2 * (t + 1) + 2 * p = 48 := by
            calc
              2 * (t + 1) + 2 * p = (2 * t + 2) + 2 * p := by
                simp [Nat.mul_add, Nat.add_assoc, Nat.add_comm]
              _ = (2 * t + 2 * p) + 2 := by
                simp [Nat.add_assoc, Nat.add_comm]
              _ = 48 := by simp [hsum_tp]
          simpa [hn_mul, hsum_tp'] using this
        have hq_ge : pow2 23 ≤ q := by
          have : pow2 23 * pow2 23 ≤ n := by
            have hpow : pow2 23 * pow2 23 = 2 ^ 46 := by
              simp [pow2_eq_two_pow]
            simpa [hpow] using hn_lo
          simpa [hq_def] using (Nat.le_sqrt.2 this)
        have hq_lt : q < pow2 24 := by
          have : n < pow2 24 * pow2 24 := by
            have hpow : pow2 24 * pow2 24 = 2 ^ 48 := by
              simp [pow2_eq_two_pow]
            simpa [hpow] using hn_hi
          simpa [hq_def] using (Nat.sqrt_lt.2 this)
        have hm0_ge : pow2 23 ≤ m0 := by
          by_cases hrgt : r > q <;>
            simp [hm0_def, hrgt, hq_ge, Nat.le_succ_of_le hq_ge]
        have hm0_le : m0 ≤ pow2 24 := by
          by_cases hrgt : r > q
          · have : q + 1 ≤ pow2 24 := Nat.succ_le_of_lt hq_lt
            simpa [hm0_def, hrgt] using this
          · have : q ≤ pow2 24 := Nat.le_of_lt hq_lt
            simpa [hm0_def, hrgt] using this
        have hm24_ge : pow2 23 ≤ m24 := by
          by_cases hround : (m0 == pow2 24) = true
          · simp [hm24_def, hround]
          · have hround' : (m0 == pow2 24) = false := by simpa using hround
            simp [hm24_def, hround', hm0_ge]
        have hm24_lt : m24 < pow2 24 := by
          by_cases hround : (m0 == pow2 24) = true
          · have : m24 = pow2 23 := by simp [hm24_def, hround]
            simp [this, pow2_eq_two_pow]
          · have hround' : (m0 == pow2 24) = false := by simpa using hround
            have hm0_lt : m0 < pow2 24 := lt_of_le_of_ne hm0_le (by
              intro hEq
              have ht : (m0 == pow2 24) = true := by simp [hEq]
              have : (true : Bool) = false := ht.symm.trans hround'
              cases this)
            simpa [hm24_def, hround'] using hm0_lt
        have hfracNat_lt : fracNat < 2 ^ 23 := by
          have hm24_eq : pow2 23 + fracNat = m24 := by
            simpa [hfracNat_def] using (Nat.add_sub_of_le hm24_ge)
          have hpow24 : pow2 24 = pow2 23 + pow2 23 := by
            simp [pow2_eq_two_pow, Nat.pow_succ, Nat.mul_two, Nat.mul_comm]
          have hsum : pow2 23 + fracNat < pow2 23 + pow2 23 := by
            have : pow2 23 + fracNat < pow2 24 := by
              simpa [hm24_eq] using hm24_lt
            simpa [hpow24] using this
          have hfrac : fracNat < pow2 23 := by
            have := (Nat.add_lt_add_iff_left (k := pow2 23) (n := fracNat) (m := pow2 23)).1 hsum
            simpa using this
          simpa [pow2_eq_two_pow] using hfrac

        -- Bound the output exponent field so `toDyadic?` can decode the result.
        have expHalf_lo : (-75 : Int) ≤ expHalf := by
          have expEven_lo : (-150 : Int) ≤ expEven := by
            cases hOdd : expOdd with
            | false =>
                have : (-150 : Int) ≤ dx.exp :=
                  le_trans (by decide : (-150 : Int) ≤ (-149 : Int)) hdx_exp_lo
                simpa [hexpEven_def, hOdd] using this
            | true =>
                have : (-150 : Int) ≤ dx.exp - 1 := by linarith [hdx_exp_lo]
                simpa [hexpEven_def, hOdd] using this
          have : (-150 : Int) / 2 ≤ expEven / 2 :=
            Int.ediv_le_ediv (by decide : (0 : Int) < 2) expEven_lo
          simpa [hexpHalf_def] using this
        have expHalf_hi : expHalf ≤ 52 := by
          have expEven_hi : expEven ≤ 104 := by
            cases hOdd : expOdd with
            | false =>
                simpa [hexpEven_def, hOdd] using hdx_exp_hi
            | true =>
                have : dx.exp - 1 ≤ 104 := by linarith [hdx_exp_hi]
                simpa [hexpEven_def, hOdd] using this
          have : expEven / 2 ≤ 104 / 2 :=
            Int.ediv_le_ediv (by decide : (0 : Int) < 2) expEven_hi
          simpa [hexpHalf_def] using this
        have hk0_lo : (-75 : Int) ≤ k0 := by
          have ht0 : (0 : Int) ≤ Int.ofNat t := by simp
          have hinc : expHalf ≤ expHalf + Int.ofNat t := le_add_of_nonneg_right ht0
          have : (-75 : Int) ≤ expHalf + Int.ofNat t := le_trans expHalf_lo hinc
          simpa [hk0_def] using this
        have hk0_hi : k0 ≤ 64 := by
          have ht_int : (Int.ofNat t : Int) ≤ 12 := by
            exact (Int.ofNat_le.2 ht_le12)
          have : expHalf + Int.ofNat t ≤ 52 + 12 := add_le_add expHalf_hi ht_int
          have : expHalf + Int.ofNat t ≤ 64 := le_trans this (by decide : (52 + 12 : Int) ≤ 64)
          simpa [hk0_def] using this
        have hk_hi : k ≤ 65 := by
          by_cases hround : (m0 == pow2 24) = true
          · simp [hk_def, hround]
            linarith [hk0_hi]
          · have hround' : (m0 == pow2 24) = false := by simpa using hround
            simp [hk_def, hround']
            exact le_trans hk0_hi (by decide : (64 : Int) ≤ 65)
        have hk_lo : (-75 : Int) ≤ k := by
          by_cases hround : (m0 == pow2 24) = true
          · simp [hk_def, hround]
            linarith [hk0_lo]
          · have hround' : (m0 == pow2 24) = false := by simpa using hround
            simp [hk_def, hround']
            exact hk0_lo
        have hk127_nonneg : 0 ≤ k + 127 := by linarith [hk_lo]
        have hk127_lt : k + 127 < Int.ofNat 255 := by
          have : k + 127 ≤ 65 + 127 := add_le_add hk_hi le_rfl
          have : k + 127 ≤ 192 := le_trans this (by decide : (65 + 127 : Int) ≤ 192)
          have h192 : (192 : Int) < Int.ofNat 255 := by decide
          exact lt_of_le_of_lt this h192
        have hexpNat : expNat < 255 := by
          have hkexpNat : (Int.ofNat expNat) = k + 127 := by
            simpa [hexpNat_def] using (Int.toNat_of_nonneg hk127_nonneg)
          have : (Int.ofNat expNat) < (Int.ofNat 255) := by
            simpa [hkexpNat.symm] using hk127_lt
          simpa using (Int.ofNat_lt).1 this
        have hexpNat_ne0 : expNat ≠ 0 := by
          intro h0
          have hkexpNat : (Int.ofNat expNat) = k + 127 := by
            simpa [hexpNat_def] using (Int.toNat_of_nonneg hk127_nonneg)
          have : k + 127 = 0 := by simpa [h0] using hkexpNat.symm
          have hkpos : (0 : Int) < k + 127 := by linarith [hk_lo]
          exact (ne_of_gt hkpos) this

        -- Compute `toReal (sqrt x)` from the bit-level output and rewrite it to the canonical real
        -- value.
        have htoDy : toDyadic? (ofBits (mkBits false expNat fracNat)) =
          some { sign := false, mant := pow2 23 + fracNat, exp := (Int.ofNat expNat) - 150 } := by
          have hx' : expNat < 255 := hexpNat
          have hf' : fracNat < 2 ^ 23 := hfracNat_lt
          simpa [hexpNat_ne0] using
            (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := expNat) (frac := fracNat) (hexp :=
              hx') (hfrac := hf'))
        have htoReal_sqrt :
          toReal (sqrt x) =
            ((pow2 23 + fracNat : Nat) : ℝ) * neuralBpow binaryRadix ((Int.ofNat expNat) - 150) :=
              by
          -- unfold `toReal` and the dyadic interpretation.
          have :
              toReal (sqrt x) =
                dyadicToReal { sign := false, mant := pow2 23 + fracNat, exp := (Int.ofNat expNat) -
                  150 } := by
            simp [toReal_eq, hsqrt_bits, htoDy]
          -- `dyadicToReal` with `sign = false`.
          have hdy :
              dyadicToReal { sign := false, mant := pow2 23 + fracNat, exp := (Int.ofNat expNat) -
                150 } =
                ((pow2 23 + fracNat : Nat) : ℝ) * neuralBpow binaryRadix ((Int.ofNat expNat) -
                  150) := by
            unfold dyadicToReal
            simp (config := { zeta := true }) only [Bool.false_eq_true, ite_cond_eq_false, one_mul]
          -- Avoid `simp` loops by chaining equalities directly.
          calc
            toReal (sqrt x) =
                dyadicToReal { sign := false, mant := pow2 23 + fracNat, exp := (Int.ofNat expNat) -
                  150 } := this
            _ = ((pow2 23 + fracNat : Nat) : ℝ) * neuralBpow binaryRadix ((Int.ofNat expNat) -
              150) := hdy

        -- Compute `fp32Round` on the real square root.
        set y : ℝ := Real.sqrt (toReal x)
        have hy_nonneg : 0 ≤ y := Real.sqrt_nonneg _
        -- Magnitude bounds: `2^k0 ≤ y < 2^(k0+1)`.
        have hxReal : toReal x = (dx.mant : ℝ) * neuralBpow binaryRadix dx.exp := by
          simp [toReal_eq, hx, dyadicToReal, hdx_sign]
        have hx_as_mant :
            toReal x = (mant' : ℝ) * neuralBpow binaryRadix expEven := by
          cases hOdd : expOdd with
          | false =>
              simpa [hmant'_def, hexpEven_def, hOdd] using hxReal
          | true =>
              have hbpow :
                  neuralBpow binaryRadix dx.exp =
                    neuralBpow binaryRadix (dx.exp - 1) * neuralBpow binaryRadix 1 := by
                simpa [Int.sub_add_cancel] using (neuralBpow.add_exp binaryRadix (dx.exp - 1) 1)
              calc
                toReal x
                    = (dx.mant : ℝ) * neuralBpow binaryRadix dx.exp := hxReal
                _ = (dx.mant : ℝ) *
                      (neuralBpow binaryRadix (dx.exp - 1) * neuralBpow binaryRadix 1) := by
                      simp [hbpow]
                _ = (dx.mant : ℝ) * neuralBpow binaryRadix (dx.exp - 1) * neuralBpow binaryRadix
                  1 := by
                      ring_nf
                _ = ((dx.mant : ℝ) * neuralBpow binaryRadix (dx.exp - 1)) * (2 : ℝ) := by
                      simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                _ = (dx.mant : ℝ) * (2 : ℝ) * neuralBpow binaryRadix (dx.exp - 1) := by
                      ring_nf
                _ = (dx.mant * 2 : ℝ) * neuralBpow binaryRadix (dx.exp - 1) := by
                      simp [mul_assoc]
                _ = (mant' : ℝ) * neuralBpow binaryRadix expEven := by
                      simp [hmant'_def, hexpEven_def, hOdd]
        have hy_eq : y = Real.sqrt (mant' : ℝ) * neuralBpow binaryRadix expHalf := by
          -- `expEven` is even, so `sqrt(2^expEven) = 2^(expEven/2)`.
          have hmod : expEven % 2 = 0 := by
            cases hOdd : expOdd with
            | false =>
                have : dx.exp % 2 = 0 := by
                  have : (dx.exp % 2 != 0) = false := by
                    simpa [hexpOdd_def] using hOdd
                  exact (bne_eq_false_iff_eq).1 this
                simp [hexpEven_def, hOdd, this]
            | true =>
                have hne : dx.exp % 2 ≠ 0 := by
                  intro hEq
                  have hExpOddTrue : (dx.exp % 2 != 0) = true := by
                    simpa [hexpOdd_def] using hOdd
                  have hExpOddFalse : (dx.exp % 2 != 0) = false := by
                    simp [hEq]
                  have : (true : Bool) = false := hExpOddTrue.symm.trans hExpOddFalse
                  cases this
                have h01 := Int.emod_two_eq_zero_or_one dx.exp
                have h1 : dx.exp % 2 = 1 := by
                  cases h01 with
                  | inl h0 => exact (hne h0).elim
                  | inr h1 => exact h1
                have : (dx.exp - 1) % 2 = 0 := by
                  simp [Int.sub_emod, h1]
                simpa [hexpEven_def, hOdd] using this
          have hmul : expEven / 2 * 2 = expEven := by
            simpa using (Int.ediv_mul_cancel (Int.dvd_iff_emod_eq_zero.2 hmod))
          have hbpow_sqrt : Real.sqrt (neuralBpow binaryRadix expEven) = neuralBpow binaryRadix
            expHalf := by
            have hexp : expEven = expHalf + expHalf := by
              have : expEven = expHalf * 2 := by
                simpa [hexpHalf_def] using hmul.symm
              simpa [mul_two] using this
            have hb :
                neuralBpow binaryRadix expEven =
                  neuralBpow binaryRadix expHalf * neuralBpow binaryRadix expHalf := by
              simpa [hexp] using (neuralBpow.add_exp binaryRadix expHalf expHalf)
            have hbpos : 0 ≤ neuralBpow binaryRadix expHalf :=
              le_of_lt (neuralBpow.pos binaryRadix expHalf)
            calc
              Real.sqrt (neuralBpow binaryRadix expEven)
                  = Real.sqrt (neuralBpow binaryRadix expHalf * neuralBpow binaryRadix expHalf)
                    := by
                      simp [hb]
              _ = neuralBpow binaryRadix expHalf := by
                      simpa [mul_assoc] using (Real.sqrt_mul_self hbpos)
          have hmnonneg : 0 ≤ (mant' : ℝ) := by exact_mod_cast Nat.zero_le _
          calc
            y = Real.sqrt ((mant' : ℝ) * neuralBpow binaryRadix expEven) := by
                  simpa [y] using congrArg Real.sqrt hx_as_mant
            _ = Real.sqrt (mant' : ℝ) * Real.sqrt (neuralBpow binaryRadix expEven) := by
                  simp
            _ = Real.sqrt (mant' : ℝ) * neuralBpow binaryRadix expHalf := by
                  simp [hbpow_sqrt]

        -- Bound `Real.sqrt mant'` using `t = log2 mant' / 2`.
        have hsqrt_lo : (pow2 t : ℝ) ≤ Real.sqrt (mant' : ℝ) := by
          have hx0 : 0 ≤ (pow2 t : ℝ) := by exact_mod_cast Nat.zero_le _
          have hy0 : 0 ≤ (mant' : ℝ) := by exact_mod_cast Nat.zero_le _
          have : (pow2 t : ℝ) ^ 2 ≤ (mant' : ℝ) := by
            have hnat : 2 ^ (2 * t) ≤ mant' := hmant_low
            have hcast : ((2 ^ (2 * t) : Nat) : ℝ) ≤ (mant' : ℝ) := by
              exact_mod_cast hnat
            calc
              (pow2 t : ℝ) ^ 2 = ((2 : ℝ) ^ t) ^ 2 := by
                simp [pow2_eq_two_pow, Nat.cast_pow]
              _ = (2 : ℝ) ^ (t * 2) := by
                simp [pow_mul]
              _ = (2 : ℝ) ^ (2 * t) := by
                simp [Nat.mul_comm]
              _ = ((2 ^ (2 * t) : Nat) : ℝ) := by
                simp [Nat.cast_pow]
              _ ≤ (mant' : ℝ) := hcast
          exact (Real.le_sqrt hx0 hy0).2 this
        have hsqrt_hi : Real.sqrt (mant' : ℝ) < (pow2 (t + 1) : ℝ) := by
          have hypos : (0 : ℝ) < (pow2 (t + 1) : ℝ) := by
            have : 0 < pow2 (t + 1) := by
              simp [pow2_eq_two_pow]
            exact_mod_cast this
          have : (mant' : ℝ) < (pow2 (t + 1) : ℝ) ^ 2 := by
            have hnat : mant' < 2 ^ (2 * (t + 1)) := hmant_hi
            have hcast : (mant' : ℝ) < ((2 ^ (2 * (t + 1)) : Nat) : ℝ) := by
              exact_mod_cast hnat
            have hpow :
                (pow2 (t + 1) : ℝ) ^ 2 = ((2 ^ (2 * (t + 1)) : Nat) : ℝ) := by
              calc
                (pow2 (t + 1) : ℝ) ^ 2 = ((2 : ℝ) ^ (t + 1)) ^ 2 := by
                  simp [pow2_eq_two_pow, Nat.cast_pow]
                _ = (2 : ℝ) ^ ((t + 1) * 2) := by
                  simp [pow_mul]
                _ = (2 : ℝ) ^ (2 * (t + 1)) := by
                  simp [Nat.mul_comm]
                _ = ((2 ^ (2 * (t + 1)) : Nat) : ℝ) := by
                  simp [Nat.cast_pow]
            simpa [hpow] using hcast
          exact (Real.sqrt_lt' hypos).2 this
        have hy_lo : neuralBpow binaryRadix k0 ≤ _root_.abs y := by
          have hbpowk0 :
              neuralBpow binaryRadix k0 = (pow2 t : ℝ) * neuralBpow binaryRadix expHalf := by
            calc
              neuralBpow binaryRadix k0 = neuralBpow binaryRadix (expHalf + Int.ofNat t) := by
                simp [hk0_def]
              _ = neuralBpow binaryRadix expHalf * neuralBpow binaryRadix (Int.ofNat t) := by
                simpa using (neuralBpow.add_exp binaryRadix expHalf (Int.ofNat t))
              _ = neuralBpow binaryRadix expHalf * (pow2 t : ℝ) := by
                simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal,
                  pow2_eq_two_pow, Nat.cast_pow,
                  zpow_natCast]
              _ = (pow2 t : ℝ) * neuralBpow binaryRadix expHalf := by
                simp [mul_comm]
          have hbpos : 0 ≤ neuralBpow binaryRadix expHalf :=
            le_of_lt (neuralBpow.pos binaryRadix expHalf)
          have hineq :
              (pow2 t : ℝ) * neuralBpow binaryRadix expHalf ≤
                Real.sqrt (mant' : ℝ) * neuralBpow binaryRadix expHalf :=
            mul_le_mul_of_nonneg_right hsqrt_lo hbpos
          have hyabs : _root_.abs y = y := abs_of_nonneg hy_nonneg
          have h' : neuralBpow binaryRadix k0 ≤ y := by
            have : neuralBpow binaryRadix k0 ≤ Real.sqrt (mant' : ℝ) * neuralBpow binaryRadix
              expHalf := by
              simpa [hbpowk0] using hineq
            simpa [hy_eq] using this
          simpa [hyabs] using h'
        have hy_hi : _root_.abs y < neuralBpow binaryRadix (k0 + 1) := by
          have hbpos : 0 < neuralBpow binaryRadix expHalf := neuralBpow.pos binaryRadix expHalf
          have hineq :
              Real.sqrt (mant' : ℝ) * neuralBpow binaryRadix expHalf <
                (pow2 (t + 1) : ℝ) * neuralBpow binaryRadix expHalf :=
            mul_lt_mul_of_pos_right hsqrt_hi hbpos
          have hbpowk1 :
              neuralBpow binaryRadix (k0 + 1) =
                (pow2 (t + 1) : ℝ) * neuralBpow binaryRadix expHalf := by
            calc
              neuralBpow binaryRadix (k0 + 1) =
                  neuralBpow binaryRadix (expHalf + Int.ofNat t + 1) := by
                    simp [hk0_def, add_assoc]
              _ = neuralBpow binaryRadix expHalf * neuralBpow binaryRadix (Int.ofNat t + 1) :=
                by
                    simpa [add_assoc] using (neuralBpow.add_exp binaryRadix expHalf (Int.ofNat t +
                      1))
              _ = neuralBpow binaryRadix expHalf * neuralBpow binaryRadix (Int.ofNat (t + 1)) :=
                by
                    simp
              _ = neuralBpow binaryRadix expHalf * (pow2 (t + 1) : ℝ) := by
                    have hb : neuralBpow binaryRadix (Int.ofNat (t + 1)) = (pow2 (t + 1) : ℝ) :=
                      by
                      calc
                        neuralBpow binaryRadix (Int.ofNat (t + 1))
                            = (2 : ℝ) ^ (Int.ofNat (t + 1)) := by
                                simp [TorchLean.Floats.neuralBpow, binaryRadix,
                                  NeuralRadix.toReal]
                        _ = (2 : ℝ) ^ (t + 1 : Nat) := by
                                simpa using (zpow_ofNat (2 : ℝ) (t + 1))
                        _ = ((2 ^ (t + 1) : Nat) : ℝ) := by
                                simp
                        _ = (pow2 (t + 1) : ℝ) := by
                                simp [pow2_eq_two_pow]
                    exact congrArg (fun z : ℝ => neuralBpow binaryRadix expHalf * z) hb
              _ = (pow2 (t + 1) : ℝ) * neuralBpow binaryRadix expHalf := by
                    simp [mul_comm]
          have hyabs : _root_.abs y = y := abs_of_nonneg hy_nonneg
          have h' : y < neuralBpow binaryRadix (k0 + 1) := by
            have : Real.sqrt (mant' : ℝ) * neuralBpow binaryRadix expHalf < neuralBpow
              binaryRadix (k0 + 1) := by
              simpa [hbpowk1] using hineq
            simpa [hy_eq] using this
          simpa [hyabs] using h'
        have hy0 : y ≠ 0 := by
          have hyabspos : 0 < _root_.abs y :=
            lt_of_lt_of_le (neuralBpow.pos binaryRadix k0) hy_lo
          have hyabs_ne0 : _root_.abs y ≠ 0 := ne_of_gt hyabspos
          intro hy0
          exact hyabs_ne0 (by simp [hy0])
        have hmag : TorchLean.Floats.neuralMagnitude binaryRadix y = k0 + 1 :=
          neural_magnitude_eq_of_bpow_bounds (x := y) (k := k0) hy0 hy_lo hy_hi
        have hcexp : TorchLean.Floats.neuralCexp binaryRadix TorchLean.Floats.fexp32 y = k0 - 23
          := by
          have hshift : k0 + 1 - 24 = k0 - 23 := by ring
          have hle : (-149 : Int) ≤ k0 - 23 := by
            -- `k0 ≥ -75` implies `k0 - 23 ≥ -98 ≥ -149`.
            linarith [hk0_lo]
          simp [TorchLean.Floats.neuralCexp, TorchLean.Floats.fexp32, TorchLean.Floats.FLTExp,
            hmag, hshift, max_eq_left hle]
        have hscaled :
            TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 y =
              Real.sqrt (n : ℝ) := by
          have hscaled' :
              TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 y =
                y * neuralBpow binaryRadix (-(k0 - 23)) := by
            simp [TorchLean.Floats.neuralScaledMantissa, hcexp]
          have hn_cast : (n : ℝ) = (mant' : ℝ) * (2 : ℝ) ^ (2 * p) := by
            simp [hn_def, Nat.shiftLeft_eq, Nat.cast_mul, Nat.cast_pow]
          have hsqrt_n : Real.sqrt (n : ℝ) = Real.sqrt (mant' : ℝ) * (2 : ℝ) ^ p := by
            have hmnonneg : 0 ≤ (mant' : ℝ) := by exact_mod_cast Nat.zero_le _
            calc
              Real.sqrt (n : ℝ) = Real.sqrt ((mant' : ℝ) * (2 : ℝ) ^ (2 * p)) := by simp [hn_cast]
              _ = Real.sqrt (mant' : ℝ) * Real.sqrt ((2 : ℝ) ^ (2 * p)) := by
                    simp
              _ = Real.sqrt (mant' : ℝ) * (2 : ℝ) ^ p := by
                    have : Real.sqrt ((2 : ℝ) ^ (2 * p)) = (2 : ℝ) ^ p := by
                      have hp' : 0 ≤ (2 : ℝ) ^ p := by exact le_of_lt (pow_pos (by norm_num) _)
                      have : (2 : ℝ) ^ (2 * p) = (2 : ℝ) ^ p * (2 : ℝ) ^ p := by
                        simp [two_mul, pow_add]
                      simp [this]
                    simp [this]
          have hbpow :
              y * neuralBpow binaryRadix (-(k0 - 23)) =
                Real.sqrt (mant' : ℝ) * (2 : ℝ) ^ p := by
            have hexp : expHalf + (-(k0 - 23)) = (23 : Int) - Int.ofNat t := by
              linarith [hk0_def]
            have hbpow_cancel :
                neuralBpow binaryRadix expHalf * neuralBpow binaryRadix (-(k0 - 23)) =
                  neuralBpow binaryRadix ((23 : Int) - Int.ofNat t) := by
              have hb :
                  neuralBpow binaryRadix expHalf * neuralBpow binaryRadix (-(k0 - 23)) =
                    neuralBpow binaryRadix (expHalf + (-(k0 - 23))) := by
                simpa using (neuralBpow.add_exp binaryRadix expHalf (-(k0 - 23))).symm
              have hb' :
                  neuralBpow binaryRadix (expHalf + (-(k0 - 23))) =
                    neuralBpow binaryRadix ((23 : Int) - Int.ofNat t) := by
                simpa using congrArg (neuralBpow binaryRadix) hexp
              exact hb.trans hb'
            have hp_int : (Int.ofNat p : Int) = (23 : Int) - Int.ofNat t := by
              simpa [hp_def] using (Int.ofNat_sub ht_le23)
            have hbpow_p : neuralBpow binaryRadix ((23 : Int) - Int.ofNat t) = (2 : ℝ) ^ p := by
              calc
                neuralBpow binaryRadix ((23 : Int) - Int.ofNat t)
                    = neuralBpow binaryRadix (Int.ofNat p) := by
                        have hp_int' : (23 : Int) - Int.ofNat t = Int.ofNat p := by
                          simpa using hp_int.symm
                        exact congrArg (neuralBpow binaryRadix) hp_int'
                _ = (2 : ℝ) ^ (Int.ofNat p) := by
                        simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
                _ = (2 : ℝ) ^ p := by
                        simp
            calc
              y * neuralBpow binaryRadix (-(k0 - 23))
                  = (Real.sqrt (mant' : ℝ) * neuralBpow binaryRadix expHalf) *
                      neuralBpow binaryRadix (-(k0 - 23)) := by
                        simp [hy_eq, mul_assoc]
              _ = Real.sqrt (mant' : ℝ) *
                    (neuralBpow binaryRadix expHalf * neuralBpow binaryRadix (-(k0 - 23))) := by
                        ring_nf
              _ = Real.sqrt (mant' : ℝ) * neuralBpow binaryRadix ((23 : Int) - Int.ofNat t) := by
                        exact congrArg (fun z : ℝ => Real.sqrt (mant' : ℝ) * z) hbpow_cancel
              _ = Real.sqrt (mant' : ℝ) * (2 : ℝ) ^ p := by
                        exact congrArg (fun z : ℝ => Real.sqrt (mant' : ℝ) * z) hbpow_p
          calc
            TorchLean.Floats.neuralScaledMantissa binaryRadix TorchLean.Floats.fexp32 y
                = y * neuralBpow binaryRadix (-(k0 - 23)) := hscaled'
            _ = Real.sqrt (mant' : ℝ) * (2 : ℝ) ^ p := hbpow
            _ = Real.sqrt (n : ℝ) := by simp [hsqrt_n]
        have hrnd :
            TorchLean.Floats.rnd32 (TorchLean.Floats.neuralScaledMantissa binaryRadix
              TorchLean.Floats.fexp32 y) =
              Int.ofNat m0 := by
          simpa [TorchLean.Floats.rnd32, hscaled, hm0_def, hq_def, hr_def] using
            (neural_nearest_even_sqrt_nat (n := n))
        have hfp : fp32Round y = (m0 : ℝ) * neuralBpow binaryRadix (k0 - 23) := by
          simp [fp32Round, TorchLean.Floats.neuralRound, TorchLean.Floats.neuralToReal, hcexp,
            hrnd]

        -- Now compute `toReal (sqrt x)` and match it to `fp32Round y`.
        have hm24_eq : pow2 23 + fracNat = m24 := by
          simpa [hfracNat_def] using (Nat.add_sub_of_le hm24_ge)
        have hkexpNat : (Int.ofNat expNat) = k + 127 := by
          simpa [hexpNat_def] using (Int.toNat_of_nonneg hk127_nonneg)
        have hkexp : (Int.ofNat expNat) - 150 = k - 23 := by linarith [hkexpNat]
        have htoReal_sqrt' :
            toReal (sqrt x) = (m24 : ℝ) * neuralBpow binaryRadix (k - 23) := by
          have hm24_cast : ((pow2 23 + fracNat : Nat) : ℝ) = (m24 : ℝ) := by
            exact_mod_cast hm24_eq
          calc
            toReal (sqrt x) =
                ((pow2 23 + fracNat : Nat) : ℝ) * neuralBpow binaryRadix ((Int.ofNat expNat) -
                  150) := by
                  exact htoReal_sqrt
            _ = (m24 : ℝ) * neuralBpow binaryRadix ((Int.ofNat expNat) - 150) := by
                  exact congrArg (fun z : ℝ => z * neuralBpow binaryRadix ((Int.ofNat expNat) -
                    150)) hm24_cast
            _ = (m24 : ℝ) * neuralBpow binaryRadix (k - 23) := by
                  exact congrArg (fun e : Int => (m24 : ℝ) * neuralBpow binaryRadix e) hkexp
        have htoReal_sqrt'' :
            toReal (sqrt x) = (m0 : ℝ) * neuralBpow binaryRadix (k0 - 23) := by
          by_cases hround : (m0 == pow2 24) = true
          · have hm0eq : m0 = pow2 24 := (beq_iff_eq).1 hround
            have hk : k = k0 + 1 := by simp [hk_def, hround]
            have hm24 : m24 = pow2 23 := by simp [hm24_def, hround]
            have hpow : (pow2 24 : ℝ) = (pow2 23 : ℝ) * (2 : ℝ) := by
              have hNat : (pow2 24 : Nat) = pow2 23 * 2 := by
                have : (2 : Nat) ^ 24 = (2 : Nat) ^ 23 * 2 := by
                  simp
                simp [pow2_eq_two_pow]
              exact_mod_cast hNat
            have hbpow1 : neuralBpow binaryRadix (1 : Int) = (2 : ℝ) := by
              simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]
            have hexp : k0 + 1 - 23 = (k0 - 23) + 1 := by ring
            have hbpow_shift :
                neuralBpow binaryRadix (k0 + 1 - 23) =
                  neuralBpow binaryRadix (k0 - 23) * (2 : ℝ) := by
              calc
                neuralBpow binaryRadix (k0 + 1 - 23)
                    = neuralBpow binaryRadix ((k0 - 23) + 1) := by
                        simp [hexp]
                _ = neuralBpow binaryRadix (k0 - 23) * neuralBpow binaryRadix (1 : Int) := by
                        simpa using (neuralBpow.add_exp binaryRadix (k0 - 23) 1)
                _ = neuralBpow binaryRadix (k0 - 23) * (2 : ℝ) := by
                        simp [hbpow1]
            have hm0_cast : (pow2 24 : ℝ) = (m0 : ℝ) := by
              exact_mod_cast hm0eq.symm
            calc
              toReal (sqrt x) = (m24 : ℝ) * neuralBpow binaryRadix (k - 23) := htoReal_sqrt'
              _ = (pow2 23 : ℝ) * neuralBpow binaryRadix (k0 + 1 - 23) := by
                    -- rewrite `k` and `m24` in the carry case.
                    simp [hk, hm24]
              _ = (pow2 23 : ℝ) * (neuralBpow binaryRadix (k0 - 23) * (2 : ℝ)) := by
                    simp [hbpow_shift]
              _ = ((pow2 23 : ℝ) * (2 : ℝ)) * neuralBpow binaryRadix (k0 - 23) := by
                    ring_nf
              _ = (pow2 24 : ℝ) * neuralBpow binaryRadix (k0 - 23) := by
                    simp [hpow, mul_assoc]
              _ = (m0 : ℝ) * neuralBpow binaryRadix (k0 - 23) := by
                    simp [hm0_cast]
          · have hround' : (m0 == pow2 24) = false := by simpa using hround
            have hk : k = k0 := by simp [hk_def, hround']
            have hm24 : m24 = m0 := by simp [hm24_def, hround']
            calc
              toReal (sqrt x) = (m24 : ℝ) * neuralBpow binaryRadix (k - 23) := htoReal_sqrt'
              _ = (m0 : ℝ) * neuralBpow binaryRadix (k0 - 23) := by
                    simp [hk, hm24]

        -- Final: both sides are the same real value.
        have hfp' : fp32Round (Real.sqrt (toReal x)) = (m0 : ℝ) * neuralBpow binaryRadix (k0 - 23)
          := by
          simpa [y] using hfp
        calc
          toReal (sqrt x) = (m0 : ℝ) * neuralBpow binaryRadix (k0 - 23) := htoReal_sqrt''
          _ = fp32Round (Real.sqrt (toReal x)) := by simpa using hfp'.symm

/-!
## Comparisons and min/max (finite refinement)

Comparisons are subtle in IEEE-754 because of NaNs and signed zeros. Since `FP32` is a finite-only
model, we only bridge the *finite* behavior here: comparisons/min/max agree with the corresponding
real comparisons once NaNs/Infs are ruled out.
-/

private lemma dyadicToReal_eq_signedMant_shiftLeft_toExp (d : Dyadic) (e : Int) (he : e ≤ d.exp) :
    dyadicToReal d =
      (signedMant d.sign (Nat.shiftLeft d.mant (Int.toNat (d.exp - e))) : ℝ) *
        neuralBpow binaryRadix e := by
  have hnonneg : 0 ≤ d.exp - e := sub_nonneg.mpr he
  have htoNat : (Int.ofNat (Int.toNat (d.exp - e))) = d.exp - e := by
    simpa using (Int.toNat_of_nonneg hnonneg)
  have hexp : e + (d.exp - e) = d.exp := by
    simp [sub_eq_add_neg]
  -- Expand the `bpow` product at `d.exp = e + (d.exp - e)`.
  have hbpow : neuralBpow binaryRadix d.exp = neuralBpow binaryRadix e * neuralBpow
    binaryRadix (d.exp - e) := by
    simpa [hexp] using (neuralBpow.add_exp binaryRadix e (d.exp - e))
  -- Turn `bpow (d.exp - e)` into an `ofNat` exponent and absorb it into a `shiftLeft`.
  let sh : Nat := Int.toNat (d.exp - e)
  have hbpow' : neuralBpow binaryRadix (d.exp - e) = neuralBpow binaryRadix (Int.ofNat sh) := by
    simpa [sh] using congrArg (fun t : Int => neuralBpow binaryRadix t) htoNat.symm
  have hshift :
      (signedMant d.sign (Nat.shiftLeft d.mant sh) : ℝ) =
        (signedMant d.sign d.mant : ℝ) * neuralBpow binaryRadix (d.exp - e) := by
    have hshift0 :
        (signedMant d.sign (Nat.shiftLeft d.mant sh) : ℝ) =
          (signedMant d.sign d.mant : ℝ) * neuralBpow binaryRadix (Int.ofNat sh) := by
      simpa [sh] using (signedMant_shiftLeft (sign := d.sign) (m := d.mant) (sh := sh))
    -- Replace `bpow (ofNat sh)` with `bpow (d.exp - e)`.
    simpa [hbpow'] using hshift0
  calc
    dyadicToReal d = (signedMant d.sign d.mant : ℝ) * neuralBpow binaryRadix d.exp := by
      simpa using (dyadicToReal_eq_signedMant (d := d))
    _ =
        (signedMant d.sign d.mant : ℝ) *
          (neuralBpow binaryRadix e * neuralBpow binaryRadix (d.exp - e)) := by
      simp [hbpow]
    _ =
        ((signedMant d.sign d.mant : ℝ) * neuralBpow binaryRadix (d.exp - e)) *
          neuralBpow binaryRadix e := by
      ring_nf
    _ = (signedMant d.sign (Nat.shiftLeft d.mant sh) : ℝ) * neuralBpow binaryRadix e := by
      -- Replace the first factor using `hshift`.
      simpa [mul_assoc] using congrArg (fun z : ℝ => z * neuralBpow binaryRadix e) hshift.symm

/--
Dyadic comparison correctness (lt case).

`cmpDyadic` compares two decoded dyadics by aligning exponents and comparing signed integers.
This theorem states that the `.lt` result agrees with the real-ordering of `dyadicToReal`.
-/
theorem cmpDyadic_lt_iff (a b : Dyadic) :
    cmpDyadic a b = .lt ↔ dyadicToReal a < dyadicToReal b := by
  classical
  unfold cmpDyadic
  cases hzero : (a.mant == 0 && b.mant == 0) with
  | true =>
      -- Both are real zero.
      have hab : (a.mant == 0) = true ∧ (b.mant == 0) = true := by
        simpa [Bool.and_eq_true] using (show (a.mant == 0 && b.mant == 0) = true from hzero)
      have ha0 : a.mant = 0 := (beq_iff_eq).1 hab.1
      have hb0 : b.mant = 0 := (beq_iff_eq).1 hab.2
      simp [dyadicToReal, ha0, hb0]
  | false =>
      -- Reduce to comparing aligned signed integers, then map to ℝ via `dyadicToReal`.
      let e : Int := if a.exp ≤ b.exp then a.exp else b.exp
      have heA : e ≤ a.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · have : b.exp ≤ a.exp := le_of_lt (lt_of_not_ge (show ¬ b.exp ≥ a.exp by simpa using hab))
          simp [e, hab, this]
      have heB : e ≤ b.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · simp [e, hab]
      let shA : Nat := Int.toNat (a.exp - e)
      let shB : Nat := Int.toNat (b.exp - e)
      let aNat : Nat := Nat.shiftLeft a.mant shA
      let bNat : Nat := Nat.shiftLeft b.mant shB
      let aInt : Int := signedMant a.sign aNat
      let bInt : Int := signedMant b.sign bNat
      have hcmp : cmpDyadic a b = Ord.compare aInt bInt := by
        simp (config := { zeta := true }) [cmpDyadic, hzero, e, shA, shB, aNat, bNat, aInt, bInt,
          signedMant]
      have ha : dyadicToReal a = (aInt : ℝ) * neuralBpow binaryRadix e := by
        simp [aInt, aNat, shA, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := a) (e := e) heA]
      have hb : dyadicToReal b = (bInt : ℝ) * neuralBpow binaryRadix e := by
        simp [bInt, bNat, shB, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := b) (e := e) heB]
      have hbpos : 0 < neuralBpow binaryRadix e := neuralBpow.pos binaryRadix e
      -- Cancel the positive scaling factor `2^e`.
      have hlt :
          aInt < bInt ↔ dyadicToReal a < dyadicToReal b := by
        constructor
        · intro hab
          have habR : (aInt : ℝ) < (bInt : ℝ) := by
            exact_mod_cast hab
          have habScaled :
              (aInt : ℝ) * neuralBpow binaryRadix e < (bInt : ℝ) * neuralBpow binaryRadix e :=
            mul_lt_mul_of_pos_right habR hbpos
          simpa [ha, hb] using habScaled
        · intro h
          have habScaled :
              (aInt : ℝ) * neuralBpow binaryRadix e < (bInt : ℝ) * neuralBpow binaryRadix e :=
                by
            simpa [ha, hb] using h
          have habR : (aInt : ℝ) < (bInt : ℝ) :=
            lt_of_mul_lt_mul_right habScaled (le_of_lt hbpos)
          exact (by exact_mod_cast habR)
      -- Finish.
      simpa [hcmp, compare_lt_iff_lt, hlt]

/--
Dyadic comparison correctness (eq case).

This is the equality variant of `cmpDyadic_lt_iff`.
-/
theorem cmpDyadic_eq_iff (a b : Dyadic) :
    cmpDyadic a b = .eq ↔ dyadicToReal a = dyadicToReal b := by
  classical
  unfold cmpDyadic
  cases hzero : (a.mant == 0 && b.mant == 0) with
  | true =>
      have hab : (a.mant == 0) = true ∧ (b.mant == 0) = true := by
        simpa [Bool.and_eq_true] using (show (a.mant == 0 && b.mant == 0) = true from hzero)
      have ha0 : a.mant = 0 := (beq_iff_eq).1 hab.1
      have hb0 : b.mant = 0 := (beq_iff_eq).1 hab.2
      simp [dyadicToReal, ha0, hb0]
  | false =>
      let e : Int := if a.exp ≤ b.exp then a.exp else b.exp
      have heA : e ≤ a.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · have : b.exp ≤ a.exp := le_of_lt (lt_of_not_ge (show ¬ b.exp ≥ a.exp by simpa using hab))
          simp [e, hab, this]
      have heB : e ≤ b.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · simp [e, hab]
      let shA : Nat := Int.toNat (a.exp - e)
      let shB : Nat := Int.toNat (b.exp - e)
      let aNat : Nat := Nat.shiftLeft a.mant shA
      let bNat : Nat := Nat.shiftLeft b.mant shB
      let aInt : Int := signedMant a.sign aNat
      let bInt : Int := signedMant b.sign bNat
      have hcmp : cmpDyadic a b = Ord.compare aInt bInt := by
        simp (config := { zeta := true }) [cmpDyadic, hzero, e, shA, shB, aNat, bNat, aInt, bInt,
          signedMant]
      have ha : dyadicToReal a = (aInt : ℝ) * neuralBpow binaryRadix e := by
        simp [aInt, aNat, shA, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := a) (e := e) heA]
      have hb : dyadicToReal b = (bInt : ℝ) * neuralBpow binaryRadix e := by
        simp [bInt, bNat, shB, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := b) (e := e) heB]
      have hbpos : 0 < neuralBpow binaryRadix e := neuralBpow.pos binaryRadix e
      have heq :
          aInt = bInt ↔ dyadicToReal a = dyadicToReal b := by
        calc
          aInt = bInt ↔ (aInt : ℝ) = (bInt : ℝ) := by
            simp
          _ ↔ (aInt : ℝ) * neuralBpow binaryRadix e = (bInt : ℝ) * neuralBpow binaryRadix e :=
            by
            have hbne : neuralBpow binaryRadix e ≠ 0 := ne_of_gt hbpos
            constructor
            · intro h
              simp [h]
            · intro h
              exact mul_right_cancel₀ hbne h
          _ ↔ dyadicToReal a = dyadicToReal b := by
            simp [ha, hb]
      -- `compare = .eq` ↔ equality.
      simpa [hcmp, compare_eq_iff_eq, heq]

/--
Dyadic comparison correctness (gt case).

This is the greater-than variant of `cmpDyadic_lt_iff`, phrased as `dyadicToReal b < dyadicToReal
  a`.
-/
theorem cmpDyadic_gt_iff (a b : Dyadic) :
    cmpDyadic a b = .gt ↔ dyadicToReal b < dyadicToReal a := by
  classical
  unfold cmpDyadic
  cases hzero : (a.mant == 0 && b.mant == 0) with
  | true =>
      have hab : (a.mant == 0) = true ∧ (b.mant == 0) = true := by
        simpa [Bool.and_eq_true] using (show (a.mant == 0 && b.mant == 0) = true from hzero)
      have ha0 : a.mant = 0 := (beq_iff_eq).1 hab.1
      have hb0 : b.mant = 0 := (beq_iff_eq).1 hab.2
      simp [dyadicToReal, ha0, hb0]
  | false =>
      let e : Int := if a.exp ≤ b.exp then a.exp else b.exp
      have heA : e ≤ a.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · have : b.exp ≤ a.exp := le_of_lt (lt_of_not_ge (show ¬ b.exp ≥ a.exp by simpa using hab))
          simp [e, hab, this]
      have heB : e ≤ b.exp := by
        by_cases hab : a.exp ≤ b.exp
        · simp [e, hab]
        · simp [e, hab]
      let shA : Nat := Int.toNat (a.exp - e)
      let shB : Nat := Int.toNat (b.exp - e)
      let aNat : Nat := Nat.shiftLeft a.mant shA
      let bNat : Nat := Nat.shiftLeft b.mant shB
      let aInt : Int := signedMant a.sign aNat
      let bInt : Int := signedMant b.sign bNat
      have hcmp : cmpDyadic a b = Ord.compare aInt bInt := by
        simp (config := { zeta := true }) [cmpDyadic, hzero, e, shA, shB, aNat, bNat, aInt, bInt,
          signedMant]
      have ha : dyadicToReal a = (aInt : ℝ) * neuralBpow binaryRadix e := by
        simp [aInt, aNat, shA, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := a) (e := e) heA]
      have hb : dyadicToReal b = (bInt : ℝ) * neuralBpow binaryRadix e := by
        simp [bInt, bNat, shB, e, dyadicToReal_eq_signedMant_shiftLeft_toExp (d := b) (e := e) heB]
      have hbpos : 0 < neuralBpow binaryRadix e := neuralBpow.pos binaryRadix e
      have hgt :
          bInt < aInt ↔ dyadicToReal b < dyadicToReal a := by
        constructor
        · intro hab
          have habR : (bInt : ℝ) < (aInt : ℝ) := by
            exact_mod_cast hab
          have habScaled :
              (bInt : ℝ) * neuralBpow binaryRadix e < (aInt : ℝ) * neuralBpow binaryRadix e :=
            mul_lt_mul_of_pos_right habR hbpos
          simpa [ha, hb] using habScaled
        · intro h
          have habScaled :
              (bInt : ℝ) * neuralBpow binaryRadix e < (aInt : ℝ) * neuralBpow binaryRadix e :=
                by
            simpa [ha, hb] using h
          have habR : (bInt : ℝ) < (aInt : ℝ) :=
            lt_of_mul_lt_mul_right habScaled (le_of_lt hbpos)
          exact (by exact_mod_cast habR)
      simpa [hcmp, compare_gt_iff_gt, hgt]

/--
Bridge for `IEEE32Exec.compare` on finite values.

When both operands decode to dyadics (`toDyadic? = some`), `compare` returns a result and it is
exactly `cmpDyadic` of those dyadics.
-/
theorem compare_eq_some_cmpDyadic_of_toDyadic? (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    compare x y = some (cmpDyadic dx dy) := by
  unfold compare
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
  simp [hxNaN, hyNaN, hxInf, hyInf, hx, hy]

/--
`compare x y = .lt` if and only if `toReal x < toReal y` (finite path).

This is the user-facing ordering theorem that lets downstream reasoning switch between
`IEEE32Exec.compare` and `<` on reals.
-/
theorem compare_eq_some_lt_iff_toReal_lt (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    compare x y = some .lt ↔ toReal x < toReal y := by
  have hcmp : compare x y = some (cmpDyadic dx dy) :=
    compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hx) (hy := hy)
  have hto : toReal x = dyadicToReal dx := by simp [toReal_eq, hx]
  have hto' : toReal y = dyadicToReal dy := by simp [toReal_eq, hy]
  -- Reduce to the dyadic comparison.
  simpa [hcmp, hto, hto'] using
    (cmpDyadic_lt_iff (a := dx) (b := dy))

/--
`compare x y = .eq` if and only if `toReal x = toReal y` (finite path).

Note: this equality is on the decoded real values; it ignores NaN payloads and
signed-zero distinctions (those are handled explicitly elsewhere).
-/
theorem compare_eq_some_eq_iff_toReal_eq (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    compare x y = some .eq ↔ toReal x = toReal y := by
  have hcmp : compare x y = some (cmpDyadic dx dy) :=
    compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hx) (hy := hy)
  have hto : toReal x = dyadicToReal dx := by simp [toReal_eq, hx]
  have hto' : toReal y = dyadicToReal dy := by simp [toReal_eq, hy]
  simpa [hcmp, hto, hto'] using
    (cmpDyadic_eq_iff (a := dx) (b := dy))

/--
`compare x y = .gt` if and only if `toReal y < toReal x` (finite path).

This is the greater-than companion to `compare_eq_some_lt_iff_toReal_lt`.
-/
theorem compare_eq_some_gt_iff_toReal_gt (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    compare x y = some .gt ↔ toReal y < toReal x := by
  have hcmp : compare x y = some (cmpDyadic dx dy) :=
    compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hx) (hy := hy)
  have hto : toReal x = dyadicToReal dx := by simp [toReal_eq, hx]
  have hto' : toReal y = dyadicToReal dy := by simp [toReal_eq, hy]
  simpa [hcmp, hto, hto'] using
    (cmpDyadic_gt_iff (a := dx) (b := dy))

private lemma toReal_eq_zero_of_isZero (x : IEEE32Exec) {d : Dyadic}
    (hx : toDyadic? x = some d) (hz : isZero x = true) : toReal x = 0 := by
  -- Extract bitfield facts from `isZero`.
  unfold isZero at hz
  have hfields : (expField x == 0) = true ∧ (fracField x == 0) = true := by
    simpa [Bool.and_eq_true] using hz
  -- `toDyadic?` returns the canonical dyadic `0` in the `isZero` case.
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hnaninf : (isNaN x || isInf x) = false := by simp [hxNaN, hxInf]
  have hdy :
      { sign := signBit x, mant := 0, exp := 0 } = d := by
    unfold toDyadic? at hx
    simp (config := { zeta := true }) [hnaninf, hfields.1, hfields.2] at hx
    simpa using hx
  -- Hence `toReal x = 0`.
  have hd : d = { sign := signBit x, mant := 0, exp := 0 } := by
    simpa using hdy.symm
  simp [toReal_eq, hx, hd, dyadicToReal, TorchLean.Floats.neuralBpow, binaryRadix,
    NeuralRadix.toReal]

/--
Bridge for `IEEE32Exec.minimum` on finite values: its real meaning is `min (toReal x) (toReal y)`.

This proof follows IEEE-754 style rules (including NaN propagation and signed-zero handling), but
the statement is on `toReal`, which erases the sign of zero.
-/
theorem toReal_minimum_eq_min (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    toReal (minimum x y) = min (toReal x) (toReal y) := by
  classical
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hchoose : chooseNaN2 x y = none := by
    simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
  -- Reduce to dyadic `compare`.
  have hcmp : compare x y = some (cmpDyadic dx dy) :=
    compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hx) (hy := hy)
  cases hord : cmpDyadic dx dy with
  | lt =>
      have hcmp' : compare x y = some .lt := by simpa [hord] using hcmp
      have hlt : toReal x < toReal y :=
        (compare_eq_some_lt_iff_toReal_lt (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      -- `minimum` returns `x`.
      have hmin : toReal (minimum x y) = toReal x := by
        simp [minimum, hchoose, hcmp']
      calc
        toReal (minimum x y) = toReal x := hmin
        _ = min (toReal x) (toReal y) := by
          simpa using (min_eq_left (le_of_lt hlt)).symm
  | gt =>
      have hcmp' : compare x y = some .gt := by simpa [hord] using hcmp
      have hlt : toReal y < toReal x :=
        (compare_eq_some_gt_iff_toReal_gt (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      have hmin : toReal (minimum x y) = toReal y := by
        simp [minimum, hchoose, hcmp']
      calc
        toReal (minimum x y) = toReal y := hmin
        _ = min (toReal x) (toReal y) := by
          simpa using (min_eq_right (le_of_lt hlt)).symm
  | eq =>
      have hcmp' : compare x y = some .eq := by simpa [hord] using hcmp
      have heq : toReal x = toReal y :=
        (compare_eq_some_eq_iff_toReal_eq (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      cases hzeros : (isZero x && isZero y) with
      | true =>
          have hxz : isZero x = true := by
            have : (isZero x = true) ∧ (isZero y = true) := by
              simpa [Bool.and_eq_true] using (show (isZero x && isZero y) = true from hzeros)
            exact this.1
          have hyz : isZero y = true := by
            have : (isZero x = true) ∧ (isZero y = true) := by
              simpa [Bool.and_eq_true] using (show (isZero x && isZero y) = true from hzeros)
            exact this.2
          have hx0 : toReal x = 0 := toReal_eq_zero_of_isZero (x := x) (hx := hx) (hz := hxz)
          have hy0 : toReal y = 0 := toReal_eq_zero_of_isZero (x := y) (hx := hy) (hz := hyz)
          have hmin : toReal (minimum x y) = 0 := by
            -- `minimum` returns a signed zero; `toReal` erases the sign.
              cases hs : (signBit x || signBit y) with
              | true =>
                  simp [minimum, hchoose, hcmp', hzeros, hs]
              | false =>
                  simp [minimum, hchoose, hcmp', hzeros, hs]
          calc
            toReal (minimum x y) = 0 := hmin
            _ = min (toReal x) (toReal y) := by
              simp [hx0, hy0]
      | false =>
          have hmin : toReal (minimum x y) = toReal x := by
            simp [minimum, hchoose, hcmp', hzeros]
          calc
            toReal (minimum x y) = toReal x := hmin
            _ = min (toReal x) (toReal y) := by
              simpa using (min_eq_left (le_of_eq heq)).symm

/--
Bridge for `IEEE32Exec.maximum` on finite values: its real meaning is `max (toReal x) (toReal y)`.

This is the companion of `toReal_minimum_eq_min`. As above, the conclusion is phrased in terms of
`toReal`, so signed zeros are identified.
-/
theorem toReal_maximum_eq_max (x y : IEEE32Exec) {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    toReal (maximum x y) = max (toReal x) (toReal y) := by
  classical
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hchoose : chooseNaN2 x y = none := by
    simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
  have hcmp : compare x y = some (cmpDyadic dx dy) :=
    compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hx) (hy := hy)
  cases hord : cmpDyadic dx dy with
  | lt =>
      have hcmp' : compare x y = some .lt := by simpa [hord] using hcmp
      have hlt : toReal x < toReal y :=
        (compare_eq_some_lt_iff_toReal_lt (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      have hmax : toReal (maximum x y) = toReal y := by
        simp [maximum, hchoose, hcmp']
      calc
        toReal (maximum x y) = toReal y := hmax
        _ = max (toReal x) (toReal y) := by
          simpa using (max_eq_right (le_of_lt hlt)).symm
  | gt =>
      have hcmp' : compare x y = some .gt := by simpa [hord] using hcmp
      have hlt : toReal y < toReal x :=
        (compare_eq_some_gt_iff_toReal_gt (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      have hmax : toReal (maximum x y) = toReal x := by
        simp [maximum, hchoose, hcmp']
      calc
        toReal (maximum x y) = toReal x := hmax
        _ = max (toReal x) (toReal y) := by
          simpa using (max_eq_left (le_of_lt hlt)).symm
  | eq =>
      have hcmp' : compare x y = some .eq := by simpa [hord] using hcmp
      have heq : toReal x = toReal y :=
        (compare_eq_some_eq_iff_toReal_eq (x := x) (y := y) (hx := hx) (hy := hy)).1 hcmp'
      cases hzeros : (isZero x && isZero y) with
      | true =>
          have hxz : isZero x = true := by
            have : (isZero x = true) ∧ (isZero y = true) := by
              simpa [Bool.and_eq_true] using (show (isZero x && isZero y) = true from hzeros)
            exact this.1
          have hyz : isZero y = true := by
            have : (isZero x = true) ∧ (isZero y = true) := by
              simpa [Bool.and_eq_true] using (show (isZero x && isZero y) = true from hzeros)
            exact this.2
          have hx0 : toReal x = 0 := toReal_eq_zero_of_isZero (x := x) (hx := hx) (hz := hxz)
          have hy0 : toReal y = 0 := toReal_eq_zero_of_isZero (x := y) (hx := hy) (hz := hyz)
          have hmax : toReal (maximum x y) = 0 := by
            -- `maximum` returns a signed zero; `toReal` erases the sign.
              cases hs : ((!signBit x) || (!signBit y)) with
              | true =>
                  simp [maximum, hchoose, hcmp', hzeros, hs]
              | false =>
                  simp [maximum, hchoose, hcmp', hzeros, hs]
          calc
            toReal (maximum x y) = 0 := hmax
            _ = max (toReal x) (toReal y) := by
              simp [hx0, hy0]
      | false =>
          have hmax : toReal (maximum x y) = toReal x := by
            simp [maximum, hchoose, hcmp', hzeros]
          calc
            toReal (maximum x y) = toReal x := hmax
            _ = max (toReal x) (toReal y) := by
              simpa using (max_eq_left (le_of_eq heq.symm)).symm

end IEEE32Exec

end TorchLean.Floats.IEEE754
