/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.DirectedRoundingSoundness.Positive

/-!
Signed directed-rounding soundness.

The lemmas in this file handle sign-sensitive arithmetic cases for lower and upper IEEE32
rounding, connecting executable operations to real-valued enclosure statements.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/--
`toEReal` respects negation on the finite/dyadic branch.

We phrase this lemma using the dyadic decode witness `toDyadic? x = some d`, which guarantees:
- `x` is finite (hence `toEReal` agrees with `toReal`), and
- `toReal (neg x) = -toReal x` (via `toReal_neg_eq_neg`).
-/
theorem toEReal_neg_of_toDyadic?_some (x : IEEE32Exec) {d : Dyadic} (hx : toDyadic? x = some d) :
    toEReal (neg x) = -toEReal x := by
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hfin : isFinite x = true :=
    isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := x) hxNaN hxInf
  have hdyNeg : toDyadic? (neg x) = some { sign := (!d.sign), mant := d.mant, exp := d.exp } :=
    toDyadic?_neg_of_toDyadic?_some (x := x) (d := d) hx
  have hfinNeg : isFinite (neg x) = true := by
    have hnan : isNaN (neg x) = false := isNaN_eq_false_of_toDyadic?_some (hx := hdyNeg)
    have hinf : isInf (neg x) = false := isInf_eq_false_of_toDyadic?_some (hx := hdyNeg)
    exact isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := neg x) hnan hinf
  have hxE : toEReal? x = some (toReal x : EReal) :=
    toEReal?_eq_some_toReal_of_isFinite_eq_true (x := x) hfin
  have hxNegE : toEReal? (neg x) = some (toReal (neg x) : EReal) :=
    toEReal?_eq_some_toReal_of_isFinite_eq_true (x := neg x) hfinNeg
  have hxReal : toReal (neg x) = -toReal x := toReal_neg_eq_neg (x := x) (d := d) hx
  have hxE0 : toEReal x = (toReal x : EReal) := toEReal_of_toEReal? hxE
  have hxNegE0 : toEReal (neg x) = (toReal (neg x) : EReal) := toEReal_of_toEReal? hxNegE
  have hxRealE : (toReal (neg x) : EReal) = (-toReal x : EReal) :=
    congrArg (fun r : ℝ => (r : EReal)) hxReal
  calc
    toEReal (neg x) = (toReal (neg x) : EReal) := hxNegE0
    _ = (-toReal x : EReal) := hxRealE
    _ = -(toReal x : EReal) := by simp
    _ = -toEReal x := by simp [hxE0]

/-
`IEEE32Exec.neg` flips only the sign bit. For our `EReal` semantics, later proofs often push
negation through `toEReal`. This is automatic in the finite/dyadic case
(`toEReal_neg_of_toDyadic?_some`), but we also need a lemma for `±∞`.

To avoid relying on private bitfield lemmas from other files, we prove the small bitwise facts we
need directly using the `Nat.testBit` API from Lean's core bitwise theory.
-/

/-- `signMask` is the single bit `2^31` (as a `Nat`). -/
@[simp] private lemma signMask_toNat : (signMask : UInt32).toNat = 2 ^ 31 := by decide

/-- `expAllOnes` is the mask `2^8-1` (as a `Nat`). -/
@[simp] private lemma expAllOnes_toNat : (expAllOnes : UInt32).toNat = 2 ^ 8 - 1 := by decide

/-- `fracMask` is the mask `2^23-1` (as a `Nat`). -/
@[simp] private lemma fracMask_toNat : (fracMask : UInt32).toNat = 2 ^ 23 - 1 := by decide

/--
`signBit x` is exactly the `31`-st bit of `x.bits` (viewed as a `Nat`).

This is the bridge we use to reason about sign-bit toggling under `neg` using the core lemma
`Nat.testBit_xor`.
-/
private lemma signBit_eq_testBit31 (x : IEEE32Exec) :
    signBit x = x.bits.toNat.testBit 31 := by
  classical
  have hSignMask : signMask.toNat = 2 ^ 31 := signMask_toNat
  by_cases hb : x.bits.toNat.testBit 31
  · -- bit 31 is set, so `x.bits &&& signMask` is nonzero.
    have hnat : x.bits.toNat &&& signMask.toNat = 2 ^ 31 := by
      simpa [hSignMask, Nat.and_two_pow, hb] using (Nat.and_two_pow x.bits.toNat 31)
    have hne : (x.bits &&& signMask) ≠ 0 := by
      intro h0
      have h0' : (x.bits &&& signMask).toNat = 0 := by simp [h0]
      have hAnd0 : (x.bits.toNat &&& signMask.toNat) = 0 := by
        simpa [UInt32.toNat_and] using h0'
      have hPow0 : (2 ^ 31 : Nat) = 0 := by
        -- `x.bits.toNat &&& signMask.toNat` is simultaneously `2^31` (because bit 31 is set) and
        -- `0`
        -- (because the underlying `UInt32` result is `0`), contradiction.
        exact hnat.symm.trans hAnd0
      exact (Nat.ne_of_gt (Nat.pow_pos (a := 2) (n := 31) (by decide : 0 < (2 : Nat)))) hPow0
    have hbne : (x.bits &&& signMask != 0) = true := (bne_iff_ne).2 hne
    simp [IEEE32Exec.signBit, hb, hbne]
  · -- bit 31 is not set, so `x.bits &&& signMask = 0`.
    have hnat : x.bits.toNat &&& signMask.toNat = 0 := by
      simpa [hSignMask, Nat.and_two_pow, hb] using (Nat.and_two_pow x.bits.toNat 31)
    have heq : (x.bits &&& signMask) = 0 := by
      apply (UInt32.toNat_inj).1
      simpa [UInt32.toNat_and, hnat]
    simp [IEEE32Exec.signBit, hb, heq]

/--
`signBit` toggles under `neg` (on the non-NaN branch).

This is the only fact about `neg` we need in the `±∞` case of `toEReal_neg_of_isNaN_eq_false`.
-/
private lemma signBit_neg_of_isNaN_eq_false (x : IEEE32Exec) (hnan : isNaN x = false) :
    signBit (neg x) = (!signBit x) := by
  have hnegBits : (neg x).bits = x.bits ^^^ signMask := by
    simp [IEEE32Exec.neg, hnan, ofBits]
  have hNeg : signBit (neg x) = (neg x).bits.toNat.testBit 31 :=
    signBit_eq_testBit31 (x := neg x)
  have hOrig : signBit x = x.bits.toNat.testBit 31 :=
    signBit_eq_testBit31 (x := x)
  -- Reduce everything to the nat-level `testBit` statement, then use `Nat.testBit_xor`.
  rw [hNeg, hOrig, hnegBits]
  have hx :
      ((x.bits ^^^ signMask).toNat).testBit 31 =
        Bool.xor (x.bits.toNat.testBit 31) (signMask.toNat.testBit 31) := by
    simp [UInt32.toNat_xor, Nat.testBit_xor]
  rw [hx]
  have hmask : signMask.toNat.testBit 31 = true := by
    simpa [signMask_toNat] using (Nat.testBit_two_pow_self (n := 31))
  rw [hmask]
  cases hb : x.bits.toNat.testBit 31 <;> simp

/-! ## Negation / sign-bit flip helpers -/

-- The core bitfield lemmas about sign-bit flips live in `NN.Floats.IEEEExec.Negation`.

/-! ## Negation preserves “not NaN” -/

/--
If `x` is not a NaN then `neg x` is not a NaN.

We only state the direction we actually need in the directed-rounding development: negative
directed rounding is implemented as `neg` of a positive kernel, so we need to know this cannot
introduce a NaN when starting from a non-NaN value.
-/
theorem isNaN_neg_eq_false_of_isNaN_eq_false (x : IEEE32Exec) (hnan : isNaN x = false) :
    isNaN (neg x) = false := by
  -- In the non-NaN branch, `neg` is `ofBits (x.bits ^^^ signMask)`.
  have hexp : expField (neg x) = expField x := by
    simpa [IEEE32Exec.neg, hnan, ofBits] using (expField_ofBits_xor_signMask (b := x.bits))
  have hfrac : fracField (neg x) = fracField x := by
    simpa [IEEE32Exec.neg, hnan, ofBits] using (fracField_ofBits_xor_signMask (b := x.bits))
  -- `isNaN` depends only on exponent+fraction fields.
  have hEq : isNaN (neg x) = isNaN x := by
    unfold IEEE32Exec.isNaN
    simp [hexp, hfrac]
  -- Rewrite the goal using `hEq`.
  simpa [hEq] using hnan

/--
`toEReal` commutes with `neg` as long as we are not in the NaN case.

This is a small but important glue lemma: it lets us lift the positive rounding kernels to the
signed directed-rounding functions (`roundDyadicDown` / `roundDyadicUp`) without introducing new
floating-point corner cases.
-/
theorem toEReal_neg_of_isNaN_eq_false (x : IEEE32Exec) (hnan : isNaN x = false) :
    toEReal (neg x) = -toEReal x := by
  classical
  cases hx : toDyadic? x with
  | some d =>
      exact toEReal_neg_of_toDyadic?_some (x := x) (d := d) hx
  | none =>
      have hxInf : isInf x = true :=
        isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := x) hx hnan
      -- Compute `toEReal x` and `toEReal (neg x)` in the `±∞` branch.
      have hxE : toEReal? x = some (if signBit x then (⊥ : EReal) else (⊤ : EReal)) := by
        simp [IEEE32Exec.toEReal?, hnan, hxInf]
      -- `neg` xors only the sign bit, so exponent+fraction fields are unchanged.
      have hexp : expField (neg x) = expField x := by
        simpa [IEEE32Exec.neg, hnan, ofBits] using (expField_ofBits_xor_signMask (b := x.bits))
      have hfrac : fracField (neg x) = fracField x := by
        simpa [IEEE32Exec.neg, hnan, ofBits] using (fracField_ofBits_xor_signMask (b := x.bits))
      have hxNegInf : isInf (neg x) = true := by
        have hEq : isInf (neg x) = isInf x := by
          unfold IEEE32Exec.isInf
          simp [hexp, hfrac]
        simpa [hEq] using hxInf
      have hxNegNaN : isNaN (neg x) = false := by
        have hEq : isNaN (neg x) = isNaN x := by
          unfold IEEE32Exec.isNaN
          simp [hexp, hfrac]
        simpa [hEq] using hnan
      have hxNegE : toEReal? (neg x) = some (if signBit (neg x) then (⊥ : EReal) else (⊤ : EReal))
        := by
        simp [IEEE32Exec.toEReal?, hxNegNaN, hxNegInf]
      have hs : signBit (neg x) = (!signBit x) := signBit_neg_of_isNaN_eq_false (x := x) hnan
      -- Reduce to the two `±∞` cases by splitting on the sign bit.
      cases hsb : signBit x <;> simp [toEReal, hxE, hxNegE, hs, hsb]

/--
Soundness of `roundDyadicDown`: the result is ≤ the exact dyadic real value (in `EReal`).

This is the key lemma needed to justify `addDown/mulDown` as interval *lower endpoints*.
-/
theorem isNaN_roundDyadicPosUp_eq_false (mant : Nat) (exp : Int) (hm : mant ≠ 0) :
    isNaN (roundDyadicPosUp mant exp) = false := by
  -- If `roundDyadicPosUp` were a NaN, then `toEReal?` would be `none`, hence `toEReal = 0`.
  -- But `toEReal_roundDyadicPosUp_ge` shows it is ≥ the strictly positive exact value.
  cases hnan' : isNaN (roundDyadicPosUp mant exp) with
  | false =>
      simp
  | true =>
      have hE? : toEReal? (roundDyadicPosUp mant exp) = none := by
        simp [IEEE32Exec.toEReal?, hnan']
      have hE : toEReal (roundDyadicPosUp mant exp) = 0 := by
        simp [toEReal, hE?]
      have hge :
          ((mant : ℝ) * bpow exp : EReal) ≤ toEReal (roundDyadicPosUp mant exp) :=
        toEReal_roundDyadicPosUp_ge (mant := mant) (exp := exp) hm
      have hle0 : ((mant : ℝ) * bpow exp : EReal) ≤ (0 : EReal) := by
        simpa [hE] using hge
      have hle0R : (mant : ℝ) * bpow exp ≤ (0 : ℝ) := by
        have hle0' : ((mant : ℝ) * bpow exp : EReal) ≤ ((0 : ℝ) : EReal) := by
          simpa using hle0
        exact (EReal.coe_le_coe_iff).1 hle0'
      have hmpos : (0 : ℝ) < (mant : ℝ) := Nat.cast_pos.2 (Nat.pos_of_ne_zero hm)
      have hbpos : (0 : ℝ) < bpow exp := bpow_pos exp
      have hpos : (0 : ℝ) < (mant : ℝ) * bpow exp := mul_pos hmpos hbpos
      have : False := (not_le_of_gt hpos) hle0R
      cases this

/--
`roundDyadicPosDown` (directed rounding toward `-∞` for *positive* dyadics) never produces a
NaN/Inf value.

We package this as an existence statement: the output always has a dyadic decode.
This is the bridge we use later to turn `toEReal` into `toReal` (as an `EReal`) without unfolding
all bit-level definitions at call sites.
-/
theorem toDyadic?_roundDyadicPosDown_some (mant : Nat) (exp : Int) :
    ∃ d, toDyadic? (roundDyadicPosDown mant exp) = some d := by
  classical
  set log2m : Nat := Nat.log2 mant with hlog2m
  set k : Int := (Int.ofNat log2m) + exp with hk
  by_cases hkHi : k > 127
  ·
    -- Overflow branch: `roundDyadicPosDown` saturates to the maximum finite float.
    refine ⟨{ sign := false, mant := pow2 23 + (pow2 23 - 1), exp := (Int.ofNat 254) - 150 }, ?_⟩
    have hOut : roundDyadicPosDown mant exp = posMaxFinite := by
      have hkHiExp : (127 : Int) < ((log2m : Int) + exp) := by
        have : (127 : Int) < k := hkHi
        simpa [hk] using this
      simp (config := { zeta := true }) [roundDyadicPosDown, hkHiExp, hlog2m.symm]
    have hexp : (254 : Nat) < 255 := by decide
    have hfrac : (pow2 23 - 1) < 2 ^ 23 := by
      have h : (pow2 23 - 1) < pow2 23 := Nat.sub_lt (pow2_pos 23) (by decide)
      -- Rewrite the goal's RHS to `pow2 23`, then discharge it using `Nat.sub_lt`.
      -- (We avoid `simp` on the whole inequality here, since it can reduce it to a decidable
      -- proposition.)
      rw [← pow2_eq_two_pow 23]
      exact h
    have hdy :
        toDyadic? (posMaxFinite : IEEE32Exec) =
          some { sign := false, mant := pow2 23 + (pow2 23 - 1), exp := (Int.ofNat 254) - 150 } := by
      have hbits : mkBits false 254 (pow2 23 - 1) = 0x7F7FFFFF := by decide
      simpa [posMaxFinite, hbits] using
        (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 254) (frac := pow2 23 - 1) hexp hfrac)
    simpa [hOut] using hdy
  ·
    have hkHi' : ¬ k > 127 := hkHi
    by_cases hkUnder : k < -149
    ·
      -- Underflow: exact value is < 2^-149, so the floor is `+0`.
      refine ⟨{ sign := false, mant := 0, exp := 0 }, ?_⟩
      have hOut : roundDyadicPosDown mant exp = posZero := by
        have hkHiExp : ¬ (127 : Int) < ((log2m : Int) + exp) := by
          simpa [hk] using hkHi'
        have hkUnderExp : ((log2m : Int) + exp) < -149 := by
          simpa [hk] using hkUnder
        simp (config := { zeta := true }) [roundDyadicPosDown, hkHiExp, hkUnderExp, hlog2m.symm]
      have hexp : (0 : Nat) < 255 := by decide
      have hfrac : (0 : Nat) < 2 ^ 23 := by decide
      have hdy : toDyadic? (posZero : IEEE32Exec) = some { sign := false, mant := 0, exp := 0 } := by
        have hbits : mkBits false 0 0 = 0 := by decide
        simpa [posZero, hbits] using
          (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := 0) hexp hfrac)
      simpa [hOut] using hdy
    ·
      have hkUnder' : ¬ k < -149 := hkUnder
      by_cases hkSub : k < -126
      ·
        -- Subnormal branch: exponent field is `0`, so it's always finite.
        set fracNat : Nat :=
            match exp + 149 with
            | .ofNat sh => Nat.shiftLeft mant sh
            | .negSucc sh => Nat.shiftRight mant (sh + 1) with hfracNat
        set frac : Nat := fracNat % pow2 23 with hfrac
        have hfrac_lt : frac < 2 ^ 23 := by
          have : frac < pow2 23 := Nat.mod_lt _ (pow2_pos 23)
          simpa [hfrac, pow2_eq_two_pow] using this
        have hexp0 : (0 : Nat) < 255 := by decide
        -- The output is either `posZero` or `ofBits (mkBits false 0 frac)`. Both have a dyadic
        -- decode.
        cases hZero : (fracNat == 0) with
        | true =>
            -- In this sub-branch `roundDyadicPosDown` returns `posZero`.
            refine ⟨{ sign := false, mant := 0, exp := 0 }, ?_⟩
            have hOut : roundDyadicPosDown mant exp = posZero := by
              have hkHiExp : ¬ (127 : Int) < ((log2m : Int) + exp) := by
                simpa [hk] using hkHi'
              have hkUnderExp : ¬ ((log2m : Int) + exp) < -149 := by
                simpa [hk] using hkUnder'
              have hkSubExp : ((log2m : Int) + exp) < -126 := by
                simpa [hk] using hkSub
              have hZeroEq :
                  (match exp + 149 with
                    | Int.ofNat sh => mant <<< sh
                    | Int.negSucc sh => mant >>> (sh + 1)) =
                    0 := by
                  have : fracNat = 0 := (beq_iff_eq).1 hZero
                  simpa [hfracNat] using this
                -- Reduce to the final subnormal `if` and discharge the remaining obligation by
                -- contradiction.
              simp (config := { zeta := true }) [roundDyadicPosDown, hkHiExp, hkUnderExp, hkSubExp,
                hlog2m.symm, posZero]
              intro hne
              exact False.elim (hne hZeroEq)
            have hdy :
                toDyadic? (posZero : IEEE32Exec) = some { sign := false, mant := 0, exp := 0 } := by
              have hfrac0 : (0 : Nat) < 2 ^ 23 := by decide
              have hbits : mkBits false 0 0 = 0 := by decide
              simpa [posZero, hbits] using
                (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := 0) hexp0 hfrac0)
            simpa [hOut] using hdy
        | false =>
            -- Otherwise `roundDyadicPosDown` returns the subnormal float `mkBits false 0 frac`.
            by_cases hf0 : frac = 0
            ·
              -- Even if `fracNat ≠ 0`, masking can still yield `frac = 0`; then the output is
              -- `posZero`.
              refine ⟨{ sign := false, mant := 0, exp := 0 }, ?_⟩
              have hOut : roundDyadicPosDown mant exp = posZero := by
                have hkHiExp : ¬ (127 : Int) < ((log2m : Int) + exp) := by
                  simpa [hk] using hkHi'
                have hkUnderExp : ¬ ((log2m : Int) + exp) < -149 := by
                  simpa [hk] using hkUnder'
                have hkSubExp : ((log2m : Int) + exp) < -126 := by
                  simpa [hk] using hkSub
                have hZeroNe :
                    (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => mant >>> (sh + 1)) ≠
                      0 := by
                  have : fracNat ≠ 0 := (beq_eq_false_iff_ne (a := fracNat) (b := 0)).1 hZero
                  simpa [hfracNat] using this
                -- Reduce to an implication goal, then show the masked fraction is `0` using `hf0`.
                simp (config := { zeta := true }) [roundDyadicPosDown, hkHiExp, hkUnderExp,
                  hkSubExp, hlog2m.symm, posZero]
                intro _
                have hmod0 : fracNat % pow2 23 = 0 := by
                  simpa [hf0] using hfrac.symm
                -- Keep `mkBits` opaque and discharge the remaining bit-level equality by `decide`.
                have hmod0' :
                    (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => mant >>> (sh + 1)) %
                        pow2 23 =
                      0 := by
                  simpa [hfracNat] using hmod0
                have hbits : mkBits false 0 0 = (0 : UInt32) := by decide
                have hmk :
                    mkBits false 0
                        ((match exp + 149 with
                          | Int.ofNat sh => mant <<< sh
                          | Int.negSucc sh => mant >>> (sh + 1)) %
                          pow2 23) =
                      mkBits false 0 0 := by
                  simpa using congrArg (fun n : Nat => mkBits false 0 n) hmod0'
                calc
                  ofBits
                        (mkBits false 0
                          ((match exp + 149 with
                            | Int.ofNat sh => mant <<< sh
                            | Int.negSucc sh => mant >>> (sh + 1)) %
                            pow2 23)) =
                        ofBits (mkBits false 0 0) := by
                              simpa using congrArg ofBits hmk
                        _ = ofBits 0 := by
                              simp [hbits]
              have hdy :
                  toDyadic? (posZero : IEEE32Exec) =
                    some { sign := false, mant := 0, exp := 0 } := by
                have hfrac0 : (0 : Nat) < 2 ^ 23 := by decide
                have hbits : mkBits false 0 0 = 0 := by decide
                simpa [posZero, hbits] using
                  (toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := 0) hexp0 hfrac0)
              simpa [hOut] using hdy
            ·
              refine ⟨{ sign := false, mant := frac, exp := -149 }, ?_⟩
              have hOut : roundDyadicPosDown mant exp = ofBits (mkBits false 0 frac) := by
                have hkHiExp : ¬ (127 : Int) < ((log2m : Int) + exp) := by
                  simpa [hk] using hkHi'
                have hkUnderExp : ¬ ((log2m : Int) + exp) < -149 := by
                  simpa [hk] using hkUnder'
                have hkSubExp : ((log2m : Int) + exp) < -126 := by
                  simpa [hk] using hkSub
                have hZeroNe :
                    (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => mant >>> (sh + 1)) ≠
                      0 := by
                  have : fracNat ≠ 0 := (beq_eq_false_iff_ne (a := fracNat) (b := 0)).1 hZero
                  simpa [hfracNat] using this
                -- Reduce to the final subnormal `if` and select the `else` branch using `hZeroNe`.
                simp (config := { zeta := true })
                  [roundDyadicPosDown, hkHiExp, hkUnderExp, hkSubExp, hlog2m.symm]
                have hmodEq' :
                    (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => mant >>> (sh + 1)) %
                        pow2 23 =
                      frac := by
                  simpa [hfracNat] using hfrac.symm
                have hmk :
                    mkBits false 0
                        ((match exp + 149 with
                          | Int.ofNat sh => mant <<< sh
                          | Int.negSucc sh => mant >>> (sh + 1)) %
                          pow2 23) =
                      mkBits false 0 frac := by
                  simpa using congrArg (fun n : Nat => mkBits false 0 n) hmodEq'
                -- Discharge the `if` by a case split on whether the (inlined) subnormal `fracNat`
                -- is `0`.
                by_cases hEq :
                    (match exp + 149 with
                      | Int.ofNat sh => mant <<< sh
                      | Int.negSucc sh => mant >>> (sh + 1)) =
                      0
                · exact False.elim (hZeroNe hEq)
                ·
                  -- Select the `else` branch and rewrite the masked fraction using `hmodEq'`.
                  calc
                    (if
                          (match exp + 149 with
                            | Int.ofNat sh => mant <<< sh
                            | Int.negSucc sh => mant >>> (sh + 1)) =
                            0 then
                        posZero
                      else
                        ofBits
                          (mkBits false 0
                            ((match exp + 149 with
                              | Int.ofNat sh => mant <<< sh
                              | Int.negSucc sh => mant >>> (sh + 1)) %
                              pow2 23))) =
                        ofBits
                          (mkBits false 0
                            ((match exp + 149 with
                              | Int.ofNat sh => mant <<< sh
                              | Int.negSucc sh => mant >>> (sh + 1)) %
                              pow2 23)) := by
                        simpa using
                          (if_neg (h := (inferInstance :
                            Decidable ((match exp + 149 with
                              | Int.ofNat sh => mant <<< sh
                              | Int.negSucc sh => mant >>> (sh + 1)) = 0))) hEq)
                    _ = ofBits (mkBits false 0 frac) := congrArg ofBits hmk
              have hdy :
                  toDyadic? (ofBits (mkBits false 0 frac)) =
                    some { sign := false, mant := frac, exp := -149 } := by
                have h := toDyadic?_ofBits_mkBits_fin (sign := false) (exp := 0) (frac := frac)
                  hexp0 hfrac_lt
                simpa [hf0] using h
              simpa [hOut] using hdy
      ·
        -- Normal branch: exponent field is `expNat = toNat (k+127)`, which is in `[1,254]`.
        set expNat : Nat := Int.toNat (k + 127) with hexpNat
        set m24 : Nat :=
            if log2m >= 23 then Nat.shiftRight mant (log2m - 23) else Nat.shiftLeft mant (23 -
              log2m) with hm24
        set fracNat : Nat := (m24 - pow2 23) % pow2 23 with hfracNat
        have hfrac_lt : fracNat < 2 ^ 23 := by
          have : fracNat < pow2 23 := Nat.mod_lt _ (pow2_pos 23)
          simpa [hfracNat, pow2_eq_two_pow] using this
        have hk_ge : (-126 : Int) ≤ k := le_of_not_gt hkSub
        have hk_le : k ≤ 127 := le_of_not_gt hkHi'
        have hk_pos : (0 : Int) < k + 127 := by
          linarith [hk_ge]
        have hk_nonneg : 0 ≤ k + 127 := le_of_lt hk_pos
        have hz : (expNat : Int) = k + 127 := by
          simpa [expNat] using (Int.toNat_of_nonneg hk_nonneg)
        have hexpNat_lt : expNat < 255 := by
          have hk_lt : k + 127 < 255 := by
            linarith [hk_le]
          have : (expNat : Int) < (255 : Int) := by simpa [hz] using hk_lt
          exact (Int.ofNat_lt).1 this
        have expNat_ne0 : expNat ≠ 0 := by
          have : (0 : Int) < (expNat : Int) := by simpa [hz] using hk_pos
          have : 0 < expNat := (Int.natCast_pos).1 this
          exact Nat.ne_of_gt this
        have hOut : roundDyadicPosDown mant exp = ofBits (mkBits false expNat fracNat) := by
          have hkHiExp : ¬ (127 : Int) < ((log2m : Int) + exp) := by
            simpa [hk] using hkHi'
          have hkUnderExp : ¬ ((log2m : Int) + exp) < -149 := by
            simpa [hk] using hkUnder'
          have hkSubExp : ¬ ((log2m : Int) + exp) < -126 := by
            simpa [hk] using hkSub
          -- Reduce the outer branches using the translated hypotheses, then expand local
          -- abbreviations on the RHS.
          simp (config := { zeta := true })
            [roundDyadicPosDown, hkHiExp, hkUnderExp, hkSubExp, hlog2m.symm, expNat, m24, fracNat,
              hk]
        refine ⟨{ sign := false, mant := pow2 23 + fracNat, exp := (Int.ofNat expNat) - 150 }, ?_⟩
        have hdy :
            toDyadic? (ofBits (mkBits false expNat fracNat)) =
              some { sign := false, mant := pow2 23 + fracNat, exp := (Int.ofNat expNat) - 150 } :=
                by
          have h := toDyadic?_ofBits_mkBits_fin (sign := false) (exp := expNat) (frac := fracNat)
            hexpNat_lt hfrac_lt
          simpa [expNat_ne0] using h
        simpa [hOut] using hdy

/--
Soundness of `roundDyadicDown`: it produces an `EReal` lower bound for the exact dyadic value.

Informal: rounding down is monotone towards `-∞`, so the rounded result is always `≤` the exact
real meaning.
-/
theorem toEReal_roundDyadicDown_le (d : Dyadic) :
    toEReal (roundDyadicDown d) ≤ (dyadicToReal d : EReal) := by
  classical
  by_cases h0 : d.mant = 0
  ·
    -- Signed zero: exact value is `0`, and `roundDyadicDown` returns ±0.
    have hrd : roundDyadicDown d = (if d.sign then negZero else posZero) := by
      simp [roundDyadicDown, h0]
    -- Both sides are exactly `0` in `EReal`.
    rw [hrd]
    cases hs : d.sign <;> simp [hs, dyadicToReal, h0, directed_toEReal_posZero,
      directed_toEReal_negZero]
  ·
    have hm : d.mant ≠ 0 := h0
    by_cases hs : d.sign = true
    ·
      -- Negative: rounding down makes it more negative, i.e. negate the positive rounding up.
      have hOut : roundDyadicDown d = neg (roundDyadicPosUp d.mant d.exp) := by
        simp [roundDyadicDown, hm, hs]
      have hpos : ((d.mant : ℝ) * bpow d.exp : EReal) ≤ toEReal (roundDyadicPosUp d.mant d.exp) :=
        toEReal_roundDyadicPosUp_ge (mant := d.mant) (exp := d.exp) hm
      -- Negate the inequality (order-reversing).
      have hneg : toEReal (neg (roundDyadicPosUp d.mant d.exp)) ≤ (-( (d.mant : ℝ) * bpow d.exp) :
        EReal) := by
        have hnan : isNaN (roundDyadicPosUp d.mant d.exp) = false :=
          isNaN_roundDyadicPosUp_eq_false (mant := d.mant) (exp := d.exp) hm
        -- `toEReal (neg x) = -toEReal x`.
        have htoNeg :
            toEReal (neg (roundDyadicPosUp d.mant d.exp)) =
              -toEReal (roundDyadicPosUp d.mant d.exp) :=
          toEReal_neg_of_isNaN_eq_false (x := roundDyadicPosUp d.mant d.exp) hnan
        -- Negation on `EReal` is order-reversing: `b ≤ a` implies `-a ≤ -b`.
        have hneg' :
            (-toEReal (roundDyadicPosUp d.mant d.exp)) ≤ (-( (d.mant : ℝ) * bpow d.exp : EReal)) :=
          (EReal.neg_le_neg_iff).2 hpos
        simpa [htoNeg] using hneg'
      -- Rewrite the exact dyadic semantics for a negative sign.
      have hd : (dyadicToReal d : EReal) = (-( (d.mant : ℝ) * bpow d.exp) : EReal) := by
        simp [dyadicToReal, hs, bpow]
      simpa [hOut, hd] using hneg
    ·
      -- Positive: reduce to `toReal_roundDyadicPosDown_le` and cast to `EReal`.
      have hs' : d.sign = false := by
        cases h : d.sign with
        | false => rfl
        | true => cases (hs h)
      have hOut : roundDyadicDown d = roundDyadicPosDown d.mant d.exp := by
        simp [roundDyadicDown, hm, hs']
      have hleR : toReal (roundDyadicPosDown d.mant d.exp) ≤ (d.mant : ℝ) * bpow d.exp :=
        toReal_roundDyadicPosDown_le (mant := d.mant) (exp := d.exp) hm
      have hleE : (toReal (roundDyadicPosDown d.mant d.exp) : EReal) ≤ ((d.mant : ℝ) * bpow d.exp :
        EReal) := by
        exact_mod_cast hleR
      have hE : toEReal (roundDyadicPosDown d.mant d.exp) = (toReal (roundDyadicPosDown d.mant
        d.exp) : EReal) := by
        -- `roundDyadicPosDown` always decodes to a dyadic, hence is finite.
        have ⟨dd, hdd⟩ := toDyadic?_roundDyadicPosDown_some (mant := d.mant) (exp := d.exp)
        have hnan : isNaN (roundDyadicPosDown d.mant d.exp) = false :=
          isNaN_eq_false_of_toDyadic?_some (hx := hdd)
        have hinf : isInf (roundDyadicPosDown d.mant d.exp) = false :=
          isInf_eq_false_of_toDyadic?_some (hx := hdd)
        have hfin : isFinite (roundDyadicPosDown d.mant d.exp) = true :=
          isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := roundDyadicPosDown d.mant
            d.exp) hnan hinf
        have hE? :
            toEReal? (roundDyadicPosDown d.mant d.exp) =
              some (toReal (roundDyadicPosDown d.mant d.exp) : EReal) :=
          toEReal?_eq_some_toReal_of_isFinite_eq_true (x := roundDyadicPosDown d.mant d.exp) hfin
        exact toEReal_of_toEReal? hE?
      have hd : (dyadicToReal d : EReal) = ((d.mant : ℝ) * bpow d.exp : EReal) := by
        simp [dyadicToReal, hs']
      simpa [hOut, hE, hd] using hleE

/--
Soundness of `roundDyadicUp`: the result is ≥ the exact dyadic real value (in `EReal`).

This is the key lemma needed to justify `addUp/mulUp` as interval *upper endpoints*.
-/
theorem toEReal_roundDyadicUp_ge (d : Dyadic) :
    (dyadicToReal d : EReal) ≤ toEReal (roundDyadicUp d) := by
  classical
  by_cases h0 : d.mant = 0
  ·
    have hru : roundDyadicUp d = (if d.sign then negZero else posZero) := by
      simp [roundDyadicUp, h0]
    -- Both `dyadicToReal d` and `toEReal` of a signed zero are exactly `0`.
    rw [hru]
    cases hs : d.sign <;> simp [dyadicToReal, h0, hs, directed_toEReal_posZero,
      directed_toEReal_negZero]
  ·
    have hm : d.mant ≠ 0 := h0
    by_cases hs : d.sign = true
    ·
      -- Negative: rounding up makes it less negative, i.e. negate the positive rounding down.
      have hOut : roundDyadicUp d = neg (roundDyadicPosDown d.mant d.exp) := by
        simp [roundDyadicUp, hm, hs]
      have hpos : toReal (roundDyadicPosDown d.mant d.exp) ≤ (d.mant : ℝ) * bpow d.exp :=
        toReal_roundDyadicPosDown_le (mant := d.mant) (exp := d.exp) hm
      have hposE : (toReal (roundDyadicPosDown d.mant d.exp) : EReal) ≤ ((d.mant : ℝ) * bpow d.exp :
        EReal) :=
        by exact_mod_cast hpos
      have hnan : isNaN (roundDyadicPosDown d.mant d.exp) = false := by
        have ⟨dd, hdd⟩ := toDyadic?_roundDyadicPosDown_some (mant := d.mant) (exp := d.exp)
        exact isNaN_eq_false_of_toDyadic?_some (hx := hdd)
      have htoNeg :
          toEReal (neg (roundDyadicPosDown d.mant d.exp)) =
            -toEReal (roundDyadicPosDown d.mant d.exp) :=
        toEReal_neg_of_isNaN_eq_false (x := roundDyadicPosDown d.mant d.exp) hnan
      have hE : toEReal (roundDyadicPosDown d.mant d.exp) = (toReal (roundDyadicPosDown d.mant
        d.exp) : EReal) := by
        have ⟨dd, hdd⟩ := toDyadic?_roundDyadicPosDown_some (mant := d.mant) (exp := d.exp)
        have hinf : isInf (roundDyadicPosDown d.mant d.exp) = false :=
          isInf_eq_false_of_toDyadic?_some (hx := hdd)
        have hfin : isFinite (roundDyadicPosDown d.mant d.exp) = true :=
          isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := roundDyadicPosDown d.mant
            d.exp) hnan hinf
        have hE? :
            toEReal? (roundDyadicPosDown d.mant d.exp) =
              some (toReal (roundDyadicPosDown d.mant d.exp) : EReal) :=
          toEReal?_eq_some_toReal_of_isFinite_eq_true (x := roundDyadicPosDown d.mant d.exp) hfin
        exact toEReal_of_toEReal? hE?
      -- Negate the inequality.
      have : (-( (d.mant : ℝ) * bpow d.exp) : EReal) ≤ toEReal (neg (roundDyadicPosDown d.mant
        d.exp)) := by
        -- `toReal ≤ exact` implies `-exact ≤ -toReal`.
        have hneg : (-( (d.mant : ℝ) * bpow d.exp) : EReal) ≤ (-(toReal (roundDyadicPosDown d.mant
          d.exp) : EReal)) :=
          (EReal.neg_le_neg_iff).2 hposE
        -- Rewrite the RHS as `toEReal (neg ...)`.
        simpa [htoNeg, hE] using hneg
      have hd : (dyadicToReal d : EReal) = (-( (d.mant : ℝ) * bpow d.exp) : EReal) := by
        simp [dyadicToReal, hs, bpow]
      simpa [hOut, hd] using this
    ·
      have hs' : d.sign = false := by
        cases h : d.sign with
        | false => rfl
        | true => cases (hs h)
      have hOut : roundDyadicUp d = roundDyadicPosUp d.mant d.exp := by
        simp [roundDyadicUp, hm, hs']
      have hge : ((d.mant : ℝ) * bpow d.exp : EReal) ≤ toEReal (roundDyadicPosUp d.mant d.exp) :=
        toEReal_roundDyadicPosUp_ge (mant := d.mant) (exp := d.exp) hm
      have hd : (dyadicToReal d : EReal) = ((d.mant : ℝ) * bpow d.exp : EReal) := by
        simp [dyadicToReal, hs']
      simpa [hOut, hd] using hge

/--
Lower-endpoint soundness for `addDown` on finite inputs:
the result is ≤ the exact real sum (in `EReal`), even in overflow-to-`-∞` scenarios.
-/
theorem toEReal_addDown_le (x y : IEEE32Exec) (hx : isFinite x = true) (hy : isFinite y = true) :
    toEReal (addDown x y) ≤ ((toReal x + toReal y : ℝ) : EReal) := by
  classical
  -- Decode both operands to dyadics.
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        simp [hx] at h
      exact False.elim this
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfinFalse : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            have h := hfinFalse
            simp [hy] at h
          exact False.elim this
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hdx)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hdy)
          have hadd :
              addDown x y = roundDyadicDown (addDyadic dx dy) := by
            simp [addDown, hchoose, hxInf, hyInf, hdx, hdy]
          have hsum :
              (dyadicToReal (addDyadic dx dy) : EReal) = ((toReal x + toReal y : ℝ) : EReal) := by
            -- exactness of dyadic addition + decode of operands
            simp [dyadicToReal_addDyadic_exact (a := dx) (b := dy), toReal_eq, hdx, hdy]
          have hle :
              toEReal (roundDyadicDown (addDyadic dx dy)) ≤ (dyadicToReal (addDyadic dx dy) : EReal)
                :=
            toEReal_roundDyadicDown_le (d := addDyadic dx dy)
          simpa [hadd, hsum] using hle

/--
Upper-endpoint soundness for `addUp` on finite inputs:
the exact real sum is ≤ the result (in `EReal`), with overflow rounding to `+∞`.
-/
theorem toEReal_addUp_ge (x y : IEEE32Exec) (hx : isFinite x = true) (hy : isFinite y = true) :
    ((toReal x + toReal y : ℝ) : EReal) ≤ toEReal (addUp x y) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        simp [hx] at h
      exact False.elim this
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfinFalse : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            have h := hfinFalse
            simp [hy] at h
          exact False.elim this
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hdx)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hdy)
          have hadd :
              addUp x y = roundDyadicUp (addDyadic dx dy) := by
            simp [addUp, hchoose, hxInf, hyInf, hdx, hdy]
          have hsum :
              (dyadicToReal (addDyadic dx dy) : EReal) = ((toReal x + toReal y : ℝ) : EReal) := by
            simp [dyadicToReal_addDyadic_exact (a := dx) (b := dy), toReal_eq, hdx, hdy]
          have hge :
              (dyadicToReal (addDyadic dx dy) : EReal) ≤ toEReal (roundDyadicUp (addDyadic dx dy))
                :=
            toEReal_roundDyadicUp_ge (d := addDyadic dx dy)
          simpa [hadd, hsum] using hge

/-- Lower-endpoint soundness for `mulDown` on finite inputs. -/
theorem toEReal_mulDown_le (x y : IEEE32Exec) (hx : isFinite x = true) (hy : isFinite y = true) :
    toEReal (mulDown x y) ≤ ((toReal x * toReal y : ℝ) : EReal) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        simp [hx] at h
      exact False.elim this
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfinFalse : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            have h := hfinFalse
            simp [hy] at h
          exact False.elim this
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hdx)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hdy)
          have hxR : (toReal x : ℝ) = dyadicToReal dx := by simp [toReal_eq, hdx]
          have hyR : (toReal y : ℝ) = dyadicToReal dy := by simp [toReal_eq, hdy]
          set s : Bool := Bool.xor dx.sign dy.sign
          set prod : Dyadic := { sign := s, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
          have hmul :
              mulDown x y =
                if dx.mant == 0 || dy.mant == 0 then
                  if s then negZero else posZero
                else
                  roundDyadicDown prod := by
            simp [mulDown, hchoose, hxInf, hyInf, hdx, hdy, s, prod]
          cases hz : (dx.mant == 0 || dy.mant == 0) with
          | true =>
              -- At least one factor is zero, so both the exact product and the endpoint are zero.
              have hz' : dx.mant = 0 ∨ dy.mant = 0 := by
                have hzProp : (dx.mant == 0) = true ∨ (dy.mant == 0) = true :=
                  (Eq.mp (Bool.or_eq_true (dx.mant == 0) (dy.mant == 0)) hz)
                cases hzProp with
                | inl hx0 => exact Or.inl ((beq_iff_eq).1 hx0)
                | inr hy0 => exact Or.inr ((beq_iff_eq).1 hy0)
              have hprod0 : (toReal x * toReal y : ℝ) = 0 := by
                cases hz' with
                | inl hx0 =>
                    have hx0R : (toReal x : ℝ) = 0 := by
                      -- Keep `toReal x` opaque and rewrite via the extracted dyadic.
                      rw [hxR]
                      simp [dyadicToReal, hx0]
                    rw [hx0R]
                    simp
                | inr hy0 =>
                    have hy0R : (toReal y : ℝ) = 0 := by
                      rw [hyR]
                      simp [dyadicToReal, hy0]
                    rw [hy0R]
                    simp
              have hout0 : toEReal (mulDown x y) = (0 : EReal) := by
                -- `mulDown` returns a signed zero in this branch.
                by_cases hs : s <;> simp [hmul, hz, hs, directed_toEReal_posZero,
                  directed_toEReal_negZero]
              -- Both the returned endpoint and the exact product are `0`.
              -- We rewrite explicitly (instead of `simp`) to avoid unfolding `toReal` into matches.
              rw [hout0]
              have hprod0E : ((toReal x * toReal y : ℝ) : EReal) = (0 : EReal) := by
                simpa using congrArg (fun r : ℝ => (r : EReal)) hprod0
              -- Rewrite the exact product to `0`.
              rw [hprod0E]
          | false =>
              -- Nonzero mantissas: reduce to `roundDyadicDown` applied to the exact dyadic product.
              have hmul' : mulDown x y = roundDyadicDown prod := by
                simp [hmul, hz]
              have hle : toEReal (roundDyadicDown prod) ≤ (dyadicToReal prod : EReal) :=
                toEReal_roundDyadicDown_le (d := prod)
              have hprod :
                  (dyadicToReal prod : EReal) = ((toReal x * toReal y : ℝ) : EReal) := by
                have hR : dyadicToReal prod = dyadicToReal dx * dyadicToReal dy := by
                  simpa [prod, s] using (dyadicToReal_mul_exact (a := dx) (b := dy))
                -- Work in `ℝ`, then cast the equality into `EReal`.
                have hReal : dyadicToReal prod = (toReal x : ℝ) * (toReal y : ℝ) := by
                  calc
                    dyadicToReal prod = dyadicToReal dx * dyadicToReal dy := hR
                    _ = (toReal x : ℝ) * (toReal y : ℝ) := by
                      -- Reduce `toReal` by rewriting `toDyadic?` via the pattern matches `hdx/hdy`.
                      simp [toReal_eq, hdx, hdy]
                simpa using congrArg (fun r : ℝ => (r : EReal)) hReal
              simpa [hmul', hprod] using hle

/-- Upper-endpoint soundness for `mulUp` on finite inputs. -/
theorem toEReal_mulUp_ge (x y : IEEE32Exec) (hx : isFinite x = true) (hy : isFinite y = true) :
    ((toReal x * toReal y : ℝ) : EReal) ≤ toEReal (mulUp x y) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        simp [hx] at h
      exact False.elim this
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfinFalse : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            have h := hfinFalse
            simp [hy] at h
          exact False.elim this
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hdx)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hdy)
          have hxR : (toReal x : ℝ) = dyadicToReal dx := by simp [toReal_eq, hdx]
          have hyR : (toReal y : ℝ) = dyadicToReal dy := by simp [toReal_eq, hdy]
          set s : Bool := Bool.xor dx.sign dy.sign
          set prod : Dyadic := { sign := s, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
          have hmul :
              mulUp x y =
                if dx.mant == 0 || dy.mant == 0 then
                  if s then negZero else posZero
                else
                  roundDyadicUp prod := by
            simp [mulUp, hchoose, hxInf, hyInf, hdx, hdy, s, prod]
          cases hz : (dx.mant == 0 || dy.mant == 0) with
          | true =>
              -- At least one factor is zero, so both the exact product and the endpoint are zero.
              have hz' : dx.mant = 0 ∨ dy.mant = 0 := by
                have hzProp : (dx.mant == 0) = true ∨ (dy.mant == 0) = true :=
                  (Eq.mp (Bool.or_eq_true (dx.mant == 0) (dy.mant == 0)) hz)
                cases hzProp with
                | inl hx0 => exact Or.inl ((beq_iff_eq).1 hx0)
                | inr hy0 => exact Or.inr ((beq_iff_eq).1 hy0)
              have hprod0 : (toReal x * toReal y : ℝ) = 0 := by
                cases hz' with
                | inl hx0 =>
                    have hx0R : (toReal x : ℝ) = 0 := by
                      rw [hxR]
                      simp [dyadicToReal, hx0]
                    rw [hx0R]
                    simp
                | inr hy0 =>
                    have hy0R : (toReal y : ℝ) = 0 := by
                      rw [hyR]
                      simp [dyadicToReal, hy0]
                    rw [hy0R]
                    simp
              have hout0 : toEReal (mulUp x y) = (0 : EReal) := by
                by_cases hs : s <;>
                  simp [hmul, hz, hs, directed_toEReal_posZero, directed_toEReal_negZero]
              -- `0 ≤ 0`.
              rw [hout0]
              rw [hprod0]
              simp
          | false =>
              have hmul' : mulUp x y = roundDyadicUp prod := by
                simp [hmul, hz]
              have hge : (dyadicToReal prod : EReal) ≤ toEReal (roundDyadicUp prod) :=
                toEReal_roundDyadicUp_ge (d := prod)
              have hprod :
                  (dyadicToReal prod : EReal) = ((toReal x * toReal y : ℝ) : EReal) := by
                have hR : dyadicToReal prod = dyadicToReal dx * dyadicToReal dy := by
                  simpa [prod, s] using (dyadicToReal_mul_exact (a := dx) (b := dy))
                have hReal : dyadicToReal prod = (toReal x : ℝ) * (toReal y : ℝ) := by
                  calc
                    dyadicToReal prod = dyadicToReal dx * dyadicToReal dy := hR
                    _ = (toReal x : ℝ) * (toReal y : ℝ) := by
                          have hxR' : dyadicToReal dx = (toReal x : ℝ) := by simpa using hxR.symm
                          have hyR' : dyadicToReal dy = (toReal y : ℝ) := by simpa using hyR.symm
                          simp [hxR', hyR']
                simpa using congrArg (fun r : ℝ => (r : EReal)) hReal
              -- Rewrite the exact product into the dyadic product and apply the rounding theorem.
              have : ((toReal x * toReal y : ℝ) : EReal) ≤ toEReal (roundDyadicUp prod) := by
                simpa [hprod] using hge
              rw [hmul']
              exact this

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
