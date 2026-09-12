/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32.Core
public import NN.Floats.IEEEExec.Bridge.FP32.NearestEven
public import NN.Floats.IEEEExec.Exec32.Arithmetic

/-!
# IEEE32Exec and FP32: Square Root Refinement

The square-root bridge lives apart from the other arithmetic bridges in
`NN.Floats.IEEEExec.Bridge.FP32.Ops`. Its proof is by far the longest of the six, and since it needs
nothing from the add/sub/mul/fma/div theorems the two files elaborate side by side instead of one
after the other. It also needs neither of the two rounding bridges the division proof relies on
(`RoundDyadic` and `RoundRat`), which together are the longest link in the float chain, so leaving
them out lets this proof start about twenty seconds earlier in a parallel build. It also needs neither of the two rounding bridges the division proof relies on
(`RoundDyadic` and `RoundRat`), which together are the longest link in the float chain, so leaving
them out lets this proof start about twenty seconds earlier in a parallel build.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

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
    rw [hsqrt, hx0]
    simp
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
                simpa [show expAllOnes.toNat = 255 by decide, UInt32.toNat_ofNat] using h0
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
                                simpa [Int.ofNat_eq_natCast] using
                                  (zpow_natCast (2 : ℝ) (t + 1))
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
end IEEE32Exec

end TorchLean.Floats.IEEE754
