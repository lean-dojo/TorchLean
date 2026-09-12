/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32.Core
public import NN.Floats.IEEEExec.Bridge.FP32.NearestEven
public import NN.Floats.IEEEExec.Bridge.FP32.RatBounds
public import NN.Floats.IEEEExec.Rounding.RoundShiftRightEven

/-!
# IEEE32Exec and FP32: Dyadic Rounder Correctness
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

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
Nearest-even rounding of a dyadic whose magnitude is at most the largest finite binary32 value
cannot overflow.

The initial exponent guard is ruled out by the lower `log2` magnitude bound.  At exponent `127`,
the only remaining overflow path is a carry from a rounded 24-bit significand; that carry implies
that the exact significand was strictly larger than `2^24 - 1`, contradicting `hbound`.
-/
theorem isFinite_roundDyadicToIEEE32_of_abs_le_ieeeMaxFinite (d : Dyadic)
    (hbound : |dyadicToReal d| ≤ FP32.ieeeMaxFinite) :
    isFinite (roundDyadicToIEEE32 d) = true := by
  classical
  by_cases hm : d.mant = 0
  · have hround :
        roundDyadicToIEEE32 d = if d.sign then negZero else posZero := by
      simp [roundDyadicToIEEE32, hm]
    rw [hround]
    cases d.sign <;> decide
  · let log2m := Nat.log2 d.mant
    let k : Int := Int.ofNat log2m + d.exp
    have hmbeq : (d.mant == 0) = false := (beq_eq_false_iff_ne).2 hm
    have hkLe : k ≤ 127 := by
      by_contra hk
      have hk128 : (128 : Int) ≤ k := by grind
      have hbpow128_le :
          neuralBpow binaryRadix 128 ≤ neuralBpow binaryRadix k :=
        (neuralBpow_le_neuralBpow_iff binaryRadix _ _).2 hk128
      have hbpowk_le : neuralBpow binaryRadix k ≤ |dyadicToReal d| := by
        simpa [k, log2m] using bpow_log2_add_exp_le_abs_dyadicToReal d hm
      exact (not_le_of_gt FP32.ieeeMaxFinite_lt_bpow_128)
        ((hbpow128_le.trans hbpowk_le).trans hbound)
    have hkHi : ¬k > 127 := not_lt.mpr hkLe
    by_cases hkUnder : k < -150
    · have hround :
          roundDyadicToIEEE32 d = if d.sign then negZero else posZero := by
        have hlogdef : Nat.log2 d.mant = log2m := by simp [log2m]
        have hkdef : Int.ofNat log2m + d.exp = k := by simp [k]
        have hkHi' : ¬(127 < Int.ofNat log2m + d.exp) := by
          simpa [k, log2m] using hkHi
        have hkUnder' : Int.ofNat log2m + d.exp < -150 := by
          simpa [k, log2m] using hkUnder
        have hkHi'' : ¬(127 < (log2m : Int) + d.exp) := by simpa using hkHi'
        have hkUnder'' : (log2m : Int) + d.exp < -150 := by simpa using hkUnder'
        simp (config := { zeta := true }) [roundDyadicToIEEE32, hmbeq, hlogdef]
        rw [if_neg hkHi'', if_pos hkUnder'']
      rw [hround]
      cases d.sign <;> decide
    · by_cases hkSub : k < -126
      · let fracNat : Nat :=
          match d.exp + 149 with
          | .ofNat sh => Nat.shiftLeft d.mant sh
          | .negSucc sh => roundShiftRightEven d.mant (sh + 1)
        have hlogdef : Nat.log2 d.mant = log2m := by simp [log2m]
        have hkdef : Int.ofNat log2m + d.exp = k := by simp [k]
        have hkHi' : ¬(127 < (log2m : Int) + d.exp) := by
          change ¬(127 < Int.ofNat log2m + d.exp)
          rw [hkdef]
          exact hkHi
        have hkUnder' : ¬((log2m : Int) + d.exp < -150) := by
          change ¬(Int.ofNat log2m + d.exp < -150)
          rw [hkdef]
          exact hkUnder
        have hkSub' : (log2m : Int) + d.exp < -126 := by
          change Int.ofNat log2m + d.exp < -126
          rw [hkdef]
          exact hkSub
        simp (config := { zeta := true }) [roundDyadicToIEEE32, hmbeq, hlogdef]
        rw [if_neg hkHi', if_neg hkUnder', if_pos hkSub']
        change
          isFinite
            (@ite IEEE32Exec (fracNat = (0 : Nat))
              (Nat.decEq fracNat 0)
              (if d.sign then negZero else posZero)
              (match Nat.decLe (pow2 23) fracNat with
              | isTrue _ => ofBits (mkBits d.sign 1 0)
              | isFalse _ => ofBits (mkBits d.sign 0 fracNat))) = true
        cases hzero : Nat.decEq fracNat 0 with
        | isTrue heq =>
          rw [if_pos heq]
          cases d.sign <;> decide
        | isFalse hne =>
          rw [if_neg hne]
          cases hdec : Nat.decLe (pow2 23) fracNat with
          | isTrue hle =>
              simpa [hdec] using
                (isFinite_ofBits_mkBits_fin d.sign 1 0 (by decide) (by decide))
          | isFalse hnle =>
              have hfrac : fracNat < 2 ^ 23 := by
                simpa [pow2_eq_two_pow] using Nat.lt_of_not_ge hnle
              simpa [hdec] using
                (isFinite_ofBits_mkBits_fin d.sign 0 fracNat (by decide) hfrac)
      · let m24 : Nat :=
          if log2m >= 23 then
            roundShiftRightEven d.mant (log2m - 23)
          else
            Nat.shiftLeft d.mant (23 - log2m)
        let k' : Int := if m24 == pow2 24 then k + 1 else k
        let m24' : Nat := if m24 == pow2 24 then pow2 23 else m24
        have hmantLt : d.mant < 2 ^ log2m.succ := by
          have hlog : Nat.log2 d.mant = Nat.log 2 d.mant :=
            Nat.log2_eq_log_two (n := d.mant)
          simpa [log2m, hlog] using
            Nat.lt_pow_succ_log_self (b := 2) (hb := Nat.one_lt_two) d.mant
        have hm24Le : m24 ≤ 2 ^ 24 := by
          by_cases hge : log2m ≥ 23
          · let sh := log2m - 23
            have hlog : log2m = sh + 23 := by
              simpa [sh, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
                (Nat.sub_add_cancel hge).symm
            have hpowEq : 2 ^ log2m.succ = (2 ^ sh) * (2 ^ 24) := by
              rw [hlog]
              simp [Nat.succ_eq_add_one, Nat.pow_add, Nat.mul_assoc]
            have hmantLt' : d.mant < (2 ^ sh) * (2 ^ 24) :=
              hmantLt.trans_eq hpowEq
            have hqLt : Nat.shiftRight d.mant sh < 2 ^ 24 := by
              simpa [Nat.shiftRight_eq_div_pow] using Nat.div_lt_of_lt_mul hmantLt'
            have hm24Le' :
                roundShiftRightEven d.mant sh ≤ Nat.shiftRight d.mant sh + 1 :=
              roundShiftRightEven_le_shiftRight_add1 d.mant sh
            simpa [m24, hge, sh] using hm24Le'.trans (Nat.succ_le_of_lt hqLt)
          · have hlt : log2m < 23 := lt_of_not_ge hge
            let sh := 23 - log2m
            have hpowPos : 0 < 2 ^ sh := Nat.pow_pos (by decide)
            have hmul :
                d.mant * 2 ^ sh < (2 ^ log2m.succ) * 2 ^ sh :=
              Nat.mul_lt_mul_of_pos_right hmantLt hpowPos
            have hsum : log2m.succ + sh = 24 := by
              grind
            have hmul' : d.mant * 2 ^ sh < 2 ^ (log2m.succ + sh) := by
              simpa [Nat.pow_add] using hmul
            have hshift : Nat.shiftLeft d.mant sh < 2 ^ 24 := by
              simpa [Nat.shiftLeft_eq, hsum] using hmul'
            exact le_of_lt (by simpa [m24, hge, sh] using hshift)
        have hk'Le : k' ≤ 127 := by
          by_cases hcarry : m24 == pow2 24
          · have hm24Eq : m24 = pow2 24 := (beq_iff_eq).mp hcarry
            have hge : log2m ≥ 23 := by
              by_contra hnot
              have hlt : log2m < 23 := lt_of_not_ge hnot
              let sh := 23 - log2m
              have hpowPos : 0 < 2 ^ sh := Nat.pow_pos (by decide)
              have hmul :
                  d.mant * 2 ^ sh < (2 ^ log2m.succ) * 2 ^ sh :=
                Nat.mul_lt_mul_of_pos_right hmantLt hpowPos
              have hsum : log2m.succ + sh = 24 := by grind
              have hmul' : d.mant * 2 ^ sh < 2 ^ (log2m.succ + sh) := by
                simpa [Nat.pow_add] using hmul
              have hm24Lt : m24 < pow2 24 := by
                have hshift : Nat.shiftLeft d.mant sh < 2 ^ 24 := by
                  simpa [Nat.shiftLeft_eq, hsum] using hmul'
                simpa [m24, hnot, sh, pow2_eq_two_pow] using hshift
              exact (Nat.ne_of_lt hm24Lt) hm24Eq
            let sh := log2m - 23
            have hlog : log2m = sh + 23 := by
              simpa [sh, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
                (Nat.sub_add_cancel hge).symm
            have hround : roundShiftRightEven d.mant sh = pow2 24 := by
              simpa [m24, hge, sh] using hm24Eq
            have hpredNat : (pow2 24 - 1) * pow2 sh < d.mant :=
              pred_mul_pow2_lt_of_roundShiftRightEven_eq
                d.mant sh (pow2 24) hround (pow2_pos 24)
            have hpredReal :
                (((pow2 24 - 1) * pow2 sh : Nat) : ℝ) < (d.mant : ℝ) := by
              exact_mod_cast hpredNat
            by_contra hk'Hi
            have hkEq : k = 127 := by
              have : 127 < k + 1 := by simpa [k', hcarry] using lt_of_not_ge hk'Hi
              grind
            have hexpSum : Int.ofNat sh + d.exp = 104 := by
              have hlogInt : Int.ofNat log2m = Int.ofNat sh + 23 := by
                simpa using congrArg Int.ofNat hlog
              have hkEq' : Int.ofNat log2m + d.exp = 127 := by
                simpa [k] using hkEq
              rw [hlogInt] at hkEq'
              grind
            have hscale :
                ((pow2 sh : Nat) : ℝ) * neuralBpow binaryRadix d.exp =
                  neuralBpow binaryRadix 104 := by
              calc
                ((pow2 sh : Nat) : ℝ) * neuralBpow binaryRadix d.exp =
                    neuralBpow binaryRadix (Int.ofNat sh) *
                      neuralBpow binaryRadix d.exp := by
                        simp [pow2_eq_two_pow, neuralBpow, binaryRadix, NeuralRadix.toReal]
                _ = neuralBpow binaryRadix (Int.ofNat sh + d.exp) :=
                  (neuralBpow.add_exp binaryRadix _ _).symm
                _ = neuralBpow binaryRadix 104 := by rw [hexpSum]
            have hmaxLt : FP32.ieeeMaxFinite < |dyadicToReal d| := by
              calc
                FP32.ieeeMaxFinite =
                    ((2 ^ 24 - 1 : Nat) : ℝ) * neuralBpow binaryRadix 104 :=
                  FP32.ieeeMaxFinite_eq
                _ = (((pow2 24 - 1) * pow2 sh : Nat) : ℝ) *
                      neuralBpow binaryRadix d.exp := by
                  rw [← hscale]
                  rw [Nat.cast_mul, Nat.cast_sub (by decide : 1 ≤ pow2 24)]
                  simp only [pow2_eq_two_pow]
                  ring
                _ < (d.mant : ℝ) * neuralBpow binaryRadix d.exp :=
                  mul_lt_mul_of_pos_right hpredReal (neuralBpow.pos binaryRadix _)
                _ = |dyadicToReal d| := (abs_dyadicToReal d).symm
            exact (not_lt_of_ge hbound) hmaxLt
          · simpa [k', hcarry] using hkLe
        have hk'Hi : ¬k' > 127 := not_lt.mpr hk'Le
        let expNat := Int.toNat (k' + 127)
        let fracNat := m24' - pow2 23
        have hk'Ge : (-126 : Int) ≤ k' := by
          have hkGe : (-126 : Int) ≤ k := not_lt.mp hkSub
          by_cases hcarry : m24 == pow2 24
          · simp [k', hcarry]
            grind
          · simpa [k', hcarry] using hkGe
        have hexpNat : expNat < 255 := by
          have hnonneg : 0 ≤ k' + 127 := by grind
          have hlt : k' + 127 < (255 : Int) := by grind
          simpa [expNat, Int.toNat_of_nonneg hnonneg] using hlt
        have hfracNat : fracNat < 2 ^ 23 := by
          by_cases hcarry : m24 == pow2 24
          · simp [fracNat, m24', hcarry]
          · have hm24Ne : m24 ≠ pow2 24 := by
              intro heq
              exact hcarry ((beq_iff_eq).2 heq)
            have hm24Lt : m24 < 2 ^ 24 := by
              apply lt_of_le_of_ne hm24Le
              simpa [pow2_eq_two_pow] using hm24Ne
            have hm24'Eq : m24' = m24 := by simp [m24', hm24Ne]
            dsimp [fracNat]
            rw [hm24'Eq]
            simp [pow2_eq_two_pow]
            grind
        have hround :
            roundDyadicToIEEE32 d = ofBits (mkBits d.sign expNat fracNat) := by
          have hlogdef : Nat.log2 d.mant = log2m := by simp [log2m]
          have hkdef : Int.ofNat log2m + d.exp = k := by simp [k]
          have hkHi' : ¬(127 < (log2m : Int) + d.exp) := by
            change ¬(127 < Int.ofNat log2m + d.exp)
            rw [hkdef]
            exact hkHi
          have hkUnder' : ¬((log2m : Int) + d.exp < -150) := by
            change ¬(Int.ofNat log2m + d.exp < -150)
            rw [hkdef]
            exact hkUnder
          have hkSub' : ¬((log2m : Int) + d.exp < -126) := by
            change ¬(Int.ofNat log2m + d.exp < -126)
            rw [hkdef]
            exact hkSub
          have hk'Hi0 :
              ¬(127 <
                (if
                    (if 23 ≤ log2m then
                      roundShiftRightEven d.mant (log2m - 23)
                    else d.mant <<< (23 - log2m)) = pow2 24 then
                  (log2m : Int) + d.exp + 1
                else (log2m : Int) + d.exp)) := by
            simpa [m24, k', k, beq_iff_eq] using hk'Hi
          simp (config := { zeta := true }) [roundDyadicToIEEE32, hmbeq, hlogdef]
          rw [if_neg hkHi', if_neg hkUnder', if_neg hkSub']
          rw [if_neg hk'Hi0]
          simp [expNat, fracNat, m24', k', m24, k, beq_iff_eq]
        rw [hround]
        exact isFinite_ofBits_mkBits_fin d.sign expNat fracNat hexpNat hfracNat

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
          simpa [hkdef.symm] using hkLe
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
                    simpa [Nat.shiftLeft_eq] using
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
                      simpa [Nat.shiftLeft_eq] using
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
                      simpa [Nat.shiftLeft_eq] using
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
end IEEE32Exec

end TorchLean.Floats.IEEE754
