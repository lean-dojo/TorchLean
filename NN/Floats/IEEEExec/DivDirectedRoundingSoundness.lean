/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic
public import NN.Floats.IEEEExec.BridgeFP32Total
public import NN.Floats.IEEEExec.RatScaling
public import NN.Floats.IEEEExec.DirectedRoundingSoundness
public import NN.Floats.IEEEExec.MinMaxERealSoundness
public import NN.Floats.IEEEExec.NatLemmas

/-!
# Directed rounding soundness for division (`IEEE32Exec`)

`NN/Floats/IEEEExec/Exec32.lean` defines executable outward-rounded division endpoints:

- `roundRatDown` / `roundRatUp`: enclose an exact rational `±(num/den)` by a dyadic at scale
  `2^ratApproxShift` and then apply `roundDyadicDown` / `roundDyadicUp`.
- `divDown` / `divUp`: lift this to IEEE32 floats (handling NaN/Inf/±0 cases explicitly).

This file proves the enclosure direction that is needed for interval arithmetic soundness:

- `toEReal (roundRatDown …) ≤ exact`,
- `exact ≤ toEReal (roundRatUp …)`,
- `toEReal (divDown x y) ≤ toReal x / toReal y` and `toReal x / toReal y ≤ toEReal (divUp x y)`
  in the finite, nonzero-divisor regime.

We work in `EReal` so that overflow of the *computed endpoints* to `±∞` remains a sound enclosure.

References (interval arithmetic background):
- IEEE 754-2019 (floating-point arithmetic): doi:10.1109/IEEESTD.2019.8766229
- Goldberg (1991): doi:10.1145/103162.103163
- Moore–Kearfott–Cloud (2009), *Introduction to Interval Analysis*.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

 noncomputable section

 /-! ## Local helpers -/

private lemma shiftLeft_ne_zero (n k : Nat) (hn : n ≠ 0) : Nat.shiftLeft n k ≠ 0 := by
  intro h0
  have hmul : n * (2 ^ k) = 0 := by
    simpa [Nat.shiftLeft_eq] using h0
  have h : n = 0 ∨ 2 ^ k = 0 := Nat.mul_eq_zero.mp hmul
  cases h with
  | inl hn0 => exact hn hn0
  | inr hk0 =>
      have hkpos : 0 < 2 ^ k := Nat.pow_pos (n := k) (by decide : 0 < 2)
      exact (Nat.ne_of_gt hkpos) hk0

private lemma shiftLeft_cast (n k : Nat) :
    ((Nat.shiftLeft n k : Nat) : ℝ) = (n : ℝ) * (2 : ℝ) ^ k := by
  simp [Nat.shiftLeft_eq, Nat.cast_mul, Nat.cast_pow]

private lemma neural_bpow_ofNat_div (k : Nat) :
    neuralBpow binaryRadix (Int.ofNat k) = (2 : ℝ) ^ k := by
  simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal]

private lemma neural_bpow_neg_ofNat_div (k : Nat) :
    neuralBpow binaryRadix (-(Int.ofNat k)) = ((2 : ℝ) ^ k)⁻¹ := by
  -- `bpow (-k) = (bpow k)⁻¹`.
  have hneg := neuralBpow.neg_exp binaryRadix (Int.ofNat k)
  -- Rewrite `bpow (Int.ofNat k)` to `(2:ℝ)^k`.
  simpa [neural_bpow_ofNat_div] using hneg


/-! ## Natural ceil helper used by `roundRatUp` -/

private lemma quotCeil_mul_ge (num den : Nat) (hden : den ≠ 0) :
    num ≤ quotCeil num den * den := by
  classical
  have hden' : (den == 0) = false := (beq_eq_false_iff_ne).2 hden
  -- Work with the Euclidean decomposition `num = q*den + r`.
  set q : Nat := num / den
  set r : Nat := num % den
  have hn : num = q * den + r := by
    simpa [q, r, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using (Nat.div_add_mod num
      den).symm
  have hrlt : r < den := Nat.mod_lt num (Nat.pos_of_ne_zero hden)
  by_cases hr0 : r = 0
  · -- Exact division: `quotCeil = q`.
    have hceil : quotCeil num den = q := by
      simp [quotCeil, hden', q, r, hr0]
    -- Then `num = q*den` and the claim follows.
    have hn' : num = q * den := by
      simp [hn, hr0]
    -- Rewrite without rewriting `num` inside `quotCeil num den`.
    rw [hceil]
    exact le_of_eq hn'
  · -- Inexact division: `quotCeil = q+1`.
    have : quotCeil num den = q + 1 := by
      have : (r == 0) = false := (beq_eq_false_iff_ne).2 hr0
      simp [quotCeil, hden', q, r, this]
    -- Bound `q*den + r ≤ (q+1)*den` using `r < den`.
    have hle : q * den + r ≤ (q + 1) * den := by
      have hlt : q * den + r < q * den + den := Nat.add_lt_add_left hrlt (q * den)
      have hlt' : q * den + r < (q + 1) * den := by
        simpa [Nat.mul_add, Nat.add_mul, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hlt
      exact Nat.le_of_lt hlt'
    -- Rewrite the goal using `hn` on the LHS and `this` on the RHS.
    -- Avoid rewriting `num` inside `quotCeil num den` (which would break the use of `this`).
    have : q * den + r ≤ quotCeil num den * den := by
      simpa [this] using hle
    simpa [hn] using this

/-! ## Dyadic enclosure of `num/den` at scale `2^ratApproxShift` -/

private lemma ratLowerMant_mul_le (num den : Nat) :
    ratLowerMant num den * den ≤ Nat.shiftLeft num ratApproxShift := by
  -- `ratLowerMant = floor( (num*2^K) / den )` so `floor * den ≤ num*2^K`.
  simpa [ratLowerMant] using Nat.div_mul_le_self (Nat.shiftLeft num ratApproxShift) den

private lemma ratUpperMant_mul_ge (num den : Nat) (hden : den ≠ 0) :
    Nat.shiftLeft num ratApproxShift ≤ ratUpperMant num den * den := by
  -- `ratUpperMant` is `ceil( (num*2^K) / den )`.
  simpa [ratUpperMant] using
    (quotCeil_mul_ge (num := Nat.shiftLeft num ratApproxShift) (den := den) hden)

private lemma ratLower_le (num den : Nat) (hden : den ≠ 0) :
    ((ratLowerMant num den : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹) ≤ (num : ℝ) / (den : ℝ) := by
  have hdenpos : (0 : ℝ) < (den : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hden
  have hpowpos : (0 : ℝ) < (2 : ℝ) ^ ratApproxShift := by
    exact pow_pos (by norm_num : (0 : ℝ) < 2) _
  -- Cross-multiply: it suffices to show `ratLowerMant * den ≤ num * 2^K` (in `ℝ`).
  have hNat : ratLowerMant num den * den ≤ Nat.shiftLeft num ratApproxShift :=
    ratLowerMant_mul_le (num := num) (den := den)
  have hShift : (Nat.shiftLeft num ratApproxShift : ℝ) = (num : ℝ) * (2 : ℝ) ^ ratApproxShift :=
    shiftLeft_cast num ratApproxShift
  -- Convert the cross-multiplied inequality into the desired ratio inequality.
  have hcross :
      (ratLowerMant num den : ℝ) * (den : ℝ) ≤ (num : ℝ) * (2 : ℝ) ^ ratApproxShift := by
    have hR : (ratLowerMant num den : ℝ) * (den : ℝ) ≤ (Nat.shiftLeft num ratApproxShift : ℝ) := by
      exact_mod_cast hNat
    calc
      (ratLowerMant num den : ℝ) * (den : ℝ) ≤ (Nat.shiftLeft num ratApproxShift : ℝ) := hR
      _ = (num : ℝ) * (2 : ℝ) ^ ratApproxShift := hShift
  -- Now divide by `den` and by `2^K`.
  have h1 : (ratLowerMant num den : ℝ) ≤ ((num : ℝ) * (2 : ℝ) ^ ratApproxShift) / (den : ℝ) :=
    (le_div_iff₀ hdenpos).2 (by simpa [mul_assoc, mul_left_comm, mul_comm] using hcross)
  have h2 :
      (ratLowerMant num den : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹ ≤
        (((num : ℝ) * (2 : ℝ) ^ ratApproxShift) / (den : ℝ)) * ((2 : ℝ) ^ ratApproxShift)⁻¹ := by
    exact mul_le_mul_of_nonneg_right h1 (by exact inv_nonneg.2 (le_of_lt hpowpos))
  -- Simplify the RHS: `((num*2^K)/den) * (2^K)⁻¹ = num/den`.
  have hpowne : ((2 : ℝ) ^ ratApproxShift) ≠ 0 := (ne_of_gt hpowpos)
  have hsimp :
      (((num : ℝ) * (2 : ℝ) ^ ratApproxShift) / (den : ℝ)) * ((2 : ℝ) ^ ratApproxShift)⁻¹ =
        (num : ℝ) / (den : ℝ) := by
    calc
      (((num : ℝ) * (2 : ℝ) ^ ratApproxShift) / (den : ℝ)) * ((2 : ℝ) ^ ratApproxShift)⁻¹
          = ((num : ℝ) * (2 : ℝ) ^ ratApproxShift) * (den : ℝ)⁻¹ * ((2 : ℝ) ^ ratApproxShift)⁻¹ :=
            by
              simp [div_eq_mul_inv]
      _ = (num : ℝ) * (((2 : ℝ) ^ ratApproxShift) * ((2 : ℝ) ^ ratApproxShift)⁻¹) * (den : ℝ)⁻¹ :=
        by
              ring_nf
      _ = (num : ℝ) * (1 : ℝ) * (den : ℝ)⁻¹ := by
              simp [hpowne]
      _ = (num : ℝ) / (den : ℝ) := by
              simp [div_eq_mul_inv]
  simpa [hsimp] using h2

private lemma le_ratUpper (num den : Nat) (hden : den ≠ 0) :
    (num : ℝ) / (den : ℝ) ≤ ((ratUpperMant num den : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹) := by
  have hdenpos : (0 : ℝ) < (den : ℝ) := by exact_mod_cast Nat.pos_of_ne_zero hden
  have hpowpos : (0 : ℝ) < (2 : ℝ) ^ ratApproxShift := by
    exact pow_pos (by norm_num : (0 : ℝ) < 2) _
  have hNat : Nat.shiftLeft num ratApproxShift ≤ ratUpperMant num den * den :=
    ratUpperMant_mul_ge (num := num) (den := den) hden
  have hShift : (Nat.shiftLeft num ratApproxShift : ℝ) = (num : ℝ) * (2 : ℝ) ^ ratApproxShift :=
    shiftLeft_cast num ratApproxShift
  have hcross :
      (num : ℝ) * (2 : ℝ) ^ ratApproxShift ≤ (ratUpperMant num den : ℝ) * (den : ℝ) := by
    have hR : (Nat.shiftLeft num ratApproxShift : ℝ) ≤ (ratUpperMant num den : ℝ) * (den : ℝ) := by
      exact_mod_cast hNat
    calc
      (num : ℝ) * (2 : ℝ) ^ ratApproxShift = (Nat.shiftLeft num ratApproxShift : ℝ) := by
        symm
        exact hShift
      _ ≤ (ratUpperMant num den : ℝ) * (den : ℝ) := hR
  -- Divide by `den` and by `2^K` in the correct direction.
  have h1 : ((num : ℝ) * (2 : ℝ) ^ ratApproxShift) / (den : ℝ) ≤ (ratUpperMant num den : ℝ) :=
    (div_le_iff₀ hdenpos).2 (by simpa [mul_assoc, mul_left_comm, mul_comm] using hcross)
  have h2 :
      (((num : ℝ) * (2 : ℝ) ^ ratApproxShift) / (den : ℝ)) * ((2 : ℝ) ^ ratApproxShift)⁻¹ ≤
        (ratUpperMant num den : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹ := by
    exact mul_le_mul_of_nonneg_right h1 (by exact inv_nonneg.2 (le_of_lt hpowpos))
  have hpowne : ((2 : ℝ) ^ ratApproxShift) ≠ 0 := (ne_of_gt hpowpos)
  have hsimp :
      (((num : ℝ) * (2 : ℝ) ^ ratApproxShift) / (den : ℝ)) * ((2 : ℝ) ^ ratApproxShift)⁻¹ =
        (num : ℝ) / (den : ℝ) := by
    calc
      (((num : ℝ) * (2 : ℝ) ^ ratApproxShift) / (den : ℝ)) * ((2 : ℝ) ^ ratApproxShift)⁻¹
          = ((num : ℝ) * (2 : ℝ) ^ ratApproxShift) * (den : ℝ)⁻¹ * ((2 : ℝ) ^ ratApproxShift)⁻¹ :=
            by
              simp [div_eq_mul_inv]
      _ = (num : ℝ) * (((2 : ℝ) ^ ratApproxShift) * ((2 : ℝ) ^ ratApproxShift)⁻¹) * (den : ℝ)⁻¹ :=
        by
              ring_nf
      _ = (num : ℝ) * (1 : ℝ) * (den : ℝ)⁻¹ := by
              simp [hpowne]
      _ = (num : ℝ) / (den : ℝ) := by
              simp [div_eq_mul_inv]
  -- Rewriting turns `h2` into the desired inequality.
  simpa [hsimp] using h2

/-! ## Soundness of `roundRatDown` / `roundRatUp` -/

/--
Soundness of `roundRatDown`: the computed lower endpoint is below (or equal to) the exact rational.

This is the enclosure direction interval arithmetic needs:
`roundRatDown sign num den` is a *lower* bound on the exact signed quotient `±(num/den)`.
-/
theorem toEReal_roundRatDown_le (sign : Bool) (num den : Nat) (hden : den ≠ 0) :
    toEReal (roundRatDown sign num den) ≤
      ((if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) : EReal) := by
  classical
  by_cases hnum0 : num = 0
  · -- `roundRatDown` returns signed zero and the exact rational is `0`.
    have hE : toEReal (if sign then negZero else posZero) = (0 : EReal) := toEReal_signedZero sign
    simp [roundRatDown, hnum0, hE]
  ·
    have hnumbeq : (num == 0) = false := (beq_eq_false_iff_ne).2 hnum0
    -- Unfold the definition and split on the sign.
    cases hsign : sign <;>
      simp (config := { zeta := true }) [roundRatDown, hnumbeq]
    · -- positive: `loMant / 2^K ≤ num/den`
      set loMant : Nat := ratLowerMant num den
      set exp : Int := -(Int.ofNat ratApproxShift)
      have hb : neuralBpow binaryRadix exp = ((2 : ℝ) ^ ratApproxShift)⁻¹ := by
        simpa [exp] using (neural_bpow_neg_ofNat_div ratApproxShift)
      have hdy :
          dyadicToReal { sign := false, mant := loMant, exp := exp } =
            (loMant : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹ := by
        by_cases hm : loMant = 0
        · simp [dyadicToReal, hm]
        · simp [dyadicToReal, hb]
      have hlo :
          (dyadicToReal { sign := false, mant := loMant, exp := exp } : EReal) ≤
            ((num : ℝ) / (den : ℝ) : EReal) := by
        have hloR : dyadicToReal { sign := false, mant := loMant, exp := exp } ≤ (num : ℝ) / (den :
          ℝ) := by
          have hloR' : (loMant : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹ ≤ (num : ℝ) / (den : ℝ) :=
            ratLower_le (num := num) (den := den) hden
          simpa [hdy] using hloR'
        exact (EReal.coe_le_coe_iff).2 hloR
      have hrd :
          toEReal (roundDyadicDown { sign := false, mant := loMant, exp := exp }) ≤
            (dyadicToReal { sign := false, mant := loMant, exp := exp } : EReal) :=
        toEReal_roundDyadicDown_le (d := { sign := false, mant := loMant, exp := exp })
      -- Chain the bounds.
      have : toEReal (roundDyadicDown { sign := false, mant := loMant, exp := exp }) ≤
          ((num : ℝ) / (den : ℝ) : EReal) := le_trans hrd hlo
      simpa using this
    · -- negative: `num/den ≤ hiMant / 2^K`, so `-(hiMant/2^K) ≤ -(num/den)`
      set hiMant : Nat := ratUpperMant num den
      set exp : Int := -(Int.ofNat ratApproxShift)
      have hb : neuralBpow binaryRadix exp = ((2 : ℝ) ^ ratApproxShift)⁻¹ := by
        simpa [exp] using (neural_bpow_neg_ofNat_div ratApproxShift)
      have hdy :
          dyadicToReal { sign := true, mant := hiMant, exp := exp } =
            -((hiMant : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹) := by
        by_cases hm : hiMant = 0
        · simp [dyadicToReal, hm]
        · simp [dyadicToReal, hb]
      have hhiR : (num : ℝ) / (den : ℝ) ≤ (hiMant : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹ :=
        le_ratUpper (num := num) (den := den) hden
      have hnegR :
          -((hiMant : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹) ≤ -((num : ℝ) / (den : ℝ)) := by
        exact (neg_le_neg_iff).2 hhiR
      have hhi :
          (dyadicToReal { sign := true, mant := hiMant, exp := exp } : EReal) ≤
            ((-((num : ℝ) / (den : ℝ))) : EReal) := by
        have hhiR' : dyadicToReal { sign := true, mant := hiMant, exp := exp } ≤ -((num : ℝ) / (den
          : ℝ)) := by
          simpa [hdy] using hnegR
        exact (EReal.coe_le_coe_iff).2 hhiR'
      have hrd :
          toEReal (roundDyadicDown { sign := true, mant := hiMant, exp := exp }) ≤
            (dyadicToReal { sign := true, mant := hiMant, exp := exp } : EReal) :=
        toEReal_roundDyadicDown_le (d := { sign := true, mant := hiMant, exp := exp })
      have : toEReal (roundDyadicDown { sign := true, mant := hiMant, exp := exp }) ≤
          ((-((num : ℝ) / (den : ℝ))) : EReal) := le_trans hrd hhi
      simpa using this

/--
Soundness of `roundRatUp`: the exact rational is below (or equal to) the computed upper endpoint.

Together with `toEReal_roundRatDown_le`, this yields an `EReal` enclosure for `±(num/den)`.
-/
theorem toEReal_roundRatUp_ge (sign : Bool) (num den : Nat) (hden : den ≠ 0) :
    ((if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) : EReal) ≤
      toEReal (roundRatUp sign num den) := by
  classical
  by_cases hnum0 : num = 0
  ·
    have hE : toEReal (if sign then negZero else posZero) = (0 : EReal) := toEReal_signedZero sign
    simp [roundRatUp, hnum0, hE]
  ·
    have hnumbeq : (num == 0) = false := (beq_eq_false_iff_ne).2 hnum0
    cases hsign : sign <;>
      simp (config := { zeta := true }) [roundRatUp, hnumbeq]
    · -- positive: `num/den ≤ hiMant / 2^K`
      set hiMant : Nat := ratUpperMant num den
      set exp : Int := -(Int.ofNat ratApproxShift)
      have hb : neuralBpow binaryRadix exp = ((2 : ℝ) ^ ratApproxShift)⁻¹ := by
        simpa [exp] using (neural_bpow_neg_ofNat_div ratApproxShift)
      have hdy :
          dyadicToReal { sign := false, mant := hiMant, exp := exp } =
            (hiMant : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹ := by
        by_cases hm : hiMant = 0
        · simp [dyadicToReal, hm]
        · simp [dyadicToReal, hb]
      have hhiR : (num : ℝ) / (den : ℝ) ≤ (hiMant : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹ :=
        le_ratUpper (num := num) (den := den) hden
      have hhi :
          ((num : ℝ) / (den : ℝ) : EReal) ≤
            (dyadicToReal { sign := false, mant := hiMant, exp := exp } : EReal) := by
        have hhiR' : (num : ℝ) / (den : ℝ) ≤ dyadicToReal { sign := false, mant := hiMant, exp :=
          exp } := by
          simpa [hdy] using hhiR
        exact (EReal.coe_le_coe_iff).2 hhiR'
      have hru :
          (dyadicToReal { sign := false, mant := hiMant, exp := exp } : EReal) ≤
            toEReal (roundDyadicUp { sign := false, mant := hiMant, exp := exp }) :=
        toEReal_roundDyadicUp_ge (d := { sign := false, mant := hiMant, exp := exp })
      have : ((num : ℝ) / (den : ℝ) : EReal) ≤
          toEReal (roundDyadicUp { sign := false, mant := hiMant, exp := exp }) :=
        le_trans hhi hru
      simpa using this
    · -- negative: `loMant / 2^K ≤ num/den`, so `-(num/den) ≤ -(loMant/2^K)`
      set loMant : Nat := ratLowerMant num den
      set exp : Int := -(Int.ofNat ratApproxShift)
      have hb : neuralBpow binaryRadix exp = ((2 : ℝ) ^ ratApproxShift)⁻¹ := by
        simpa [exp] using (neural_bpow_neg_ofNat_div ratApproxShift)
      have hdy :
          dyadicToReal { sign := true, mant := loMant, exp := exp } =
            -((loMant : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹) := by
        by_cases hm : loMant = 0
        · simp [dyadicToReal, hm]
        · simp [dyadicToReal, hb]
      have hloR : (loMant : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹ ≤ (num : ℝ) / (den : ℝ) :=
        ratLower_le (num := num) (den := den) hden
      have hnegR :
          -((num : ℝ) / (den : ℝ)) ≤ -((loMant : ℝ) * ((2 : ℝ) ^ ratApproxShift)⁻¹) := by
        exact (neg_le_neg_iff).2 hloR
      have hlo :
          ((-((num : ℝ) / (den : ℝ))) : EReal) ≤
            (dyadicToReal { sign := true, mant := loMant, exp := exp } : EReal) := by
        have hloR' : -((num : ℝ) / (den : ℝ)) ≤ dyadicToReal { sign := true, mant := loMant, exp :=
          exp } := by
          simpa [hdy] using hnegR
        exact (EReal.coe_le_coe_iff).2 hloR'
      have hru :
          (dyadicToReal { sign := true, mant := loMant, exp := exp } : EReal) ≤
            toEReal (roundDyadicUp { sign := true, mant := loMant, exp := exp }) :=
        toEReal_roundDyadicUp_ge (d := { sign := true, mant := loMant, exp := exp })
      have : ((-((num : ℝ) / (den : ℝ))) : EReal) ≤
          toEReal (roundDyadicUp { sign := true, mant := loMant, exp := exp }) :=
        le_trans hlo hru
      simpa using this

/-! ## Expressing dyadic division as the same signed rational used by `divDown/divUp`

We reuse shared helper lemmas from `RatScaling`:
`scaleRat_ofNat`, `scaleRat_negSucc`, `neural_bpow_div`, `dyadicToReal_div_eq_signedRat`.
-/

/-! ## Soundness of `divDown` / `divUp` in the finite, nonzero-divisor regime -/

private lemma isZero_eq_true_of_toDyadic?_some_of_mant_eq_zero {y : IEEE32Exec} {dy : Dyadic}
    (hdy : toDyadic? y = some dy) (hm : dy.mant = 0) : isZero y = true := by
  -- Unfold `toDyadic?`; the `some` result forces the non-NaN/non-Inf branch.
  unfold toDyadic? at hdy
  have hcond : (isNaN y || isInf y) = false := by
    cases h : (isNaN y || isInf y) <;> simp [h] at hdy
    exact rfl
  -- Name the fields (to keep the `simp` patterns stable).
  set e : UInt32 := expField y
  set f : UInt32 := fracField y
  cases he : (e == 0) with
  | true =>
      cases hf : (f == 0) with
      | true =>
          -- Signed zero.
          simp [isZero, e, f, he, hf]
      | false =>
          -- Subnormal: mantissa is `f.toNat`, so it cannot be `0` if `f ≠ 0`.
          have hto : some { sign := signBit y, mant := f.toNat, exp := (-149 : Int) } = some dy :=
            by
            simpa [hcond, e, f, he, hf] using hdy
          have hmant : dy.mant = f.toNat := by
            simpa using (congrArg Dyadic.mant (Option.some.inj hto)).symm
          have hfne : f ≠ 0 := (beq_eq_false_iff_ne).1 hf
          have : f.toNat ≠ 0 := by
            intro h0
            have : f = 0 := (UInt32.toNat_inj).1 (by simp [h0])
            exact hfne this
          exact False.elim (this (by simpa [hmant] using hm))
  | false =>
      -- Normal: mantissa is `2^23 + f.toNat`, never `0`.
      have hto :
          some { sign := signBit y, mant := pow2 23 + f.toNat, exp := (Int.ofNat e.toNat) - 150 } =
            some dy := by
        simpa [hcond, e, f, he] using hdy
      have hmant : dy.mant = pow2 23 + f.toNat := by
        simpa using (congrArg Dyadic.mant (Option.some.inj hto)).symm
      have hpos : (0 : Nat) < pow2 23 := pow2_pos 23
      have : dy.mant ≠ 0 := by
        have : pow2 23 ≤ dy.mant := by
          rw [hmant]
          exact Nat.le_add_right (pow2 23) f.toNat
        exact Nat.ne_of_gt (lt_of_lt_of_le hpos this)
      exact False.elim (this hm)

/--
Soundness of `divDown`: a lower enclosure for real division in `EReal`.

In the finite, nonzero-divisor regime this shows:
`toEReal (divDown x y) <= (toReal x / toReal y)`.
We use `EReal` so that overflow of the *computed endpoint* to `±∞` remains a sound enclosure.
-/
theorem toEReal_divDown_le (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) (hy0 : isZero y = false) :
    toEReal (divDown x y) ≤ ((toReal x / toReal y : ℝ) : EReal) := by
  classical
  -- Decode `x` and `y` to dyadics; finiteness guarantees decoding succeeds.
  cases hdx : toDyadic? x with
  | none =>
      have hfin : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      cases (hx.symm.trans hfin)
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfin : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          cases (hy.symm.trans hfin)
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hdx)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hdy)
          have hyMant0 : dy.mant ≠ 0 := by
            intro hm
            have hz : isZero y = true := isZero_eq_true_of_toDyadic?_some_of_mant_eq_zero (hdy :=
              hdy) hm
            cases (hy0.symm.trans hz)

          -- Handle the `dx.mant = 0` branch (exact quotient is `0`).
          by_cases hx0 : dx.mant = 0
          ·
            have hxR : (toReal x : ℝ) = 0 := by
              simp [toReal_eq, hdx, dyadicToReal, hx0]
            have hq : ((toReal x / toReal y : ℝ) : EReal) = (0 : EReal) := by
              -- Rewrite `toReal x = 0` first, then use `zero_div`.
              -- We use `rw` (not `simp`) to avoid unfolding `toReal` via the `[simp]` lemma
              -- `IEEE32Exec.toReal_eq`, which can otherwise trigger `div_eq_zero_iff` side-goals.
              rw [hxR]
              simp
            have hendpoint :
                divDown x y = (if Bool.xor dx.sign dy.sign then negZero else posZero) := by
              simp (config := { zeta := true })
                [divDown, hchoose, hxInf, hyInf, hy0, hdx, hdy, hx0]
            have hout : toEReal (divDown x y) = (0 : EReal) := by
              -- Signed zeros always map to `0` in `toEReal`.
              rw [hendpoint]
              simpa using toEReal_signedZero (Bool.xor dx.sign dy.sign)
            -- Close the goal without unfolding `toReal` (which has a `[simp]` lemma).
            simp [hout, hq]
          ·
            have hx0' : (dx.mant == 0) = false := (beq_eq_false_iff_ne).2 hx0
            set sign : Bool := Bool.xor dx.sign dy.sign
            -- Case split on the exponent difference (which determines the rational `num/den`).
            cases hE : (dx.exp - dy.exp) with
            | ofNat sh =>
                let num : Nat := Nat.shiftLeft dx.mant sh
                let den : Nat := dy.mant
                have hden0 : den ≠ 0 := hyMant0
                have hround : divDown x y = roundRatDown sign num den := by
                  simp (config := { zeta := true })
                    [divDown, hchoose, hxInf, hyInf, hy0, hdx, hdy, hx0', sign, hE, num, den]
                have hrat :
                    dyadicToReal dx / dyadicToReal dy =
                      (if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) := by
                  have h := dyadicToReal_div_eq_signedRat (dx := dx) (dy := dy) hyMant0
                  simpa [sign, hE, num, den] using h
                have hexact :
                    (toReal x / toReal y : ℝ) =
                      (if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) := by
                  -- `toReal` agrees with `dyadicToReal` under `toDyadic? = some`.
                  simpa [toReal_eq, hdx, hdy] using hrat
                have hle0 :
                    toEReal (roundRatDown sign num den) ≤
                      ((if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) : EReal)
                        :=
                  toEReal_roundRatDown_le (sign := sign) (num := num) (den := den) hden0
                -- Normalize the RHS into the `EReal` division form used by later rewriting.
                have hle :
                    toEReal (roundRatDown sign num den) ≤
                      (if sign then -(((num : ℝ) : EReal) / ((den : ℝ) : EReal))
                       else ((num : ℝ) : EReal) / ((den : ℝ) : EReal)) := by
                  cases hsign : sign <;> simpa [hsign, EReal.coe_div] using hle0
                -- Rewrite the executable endpoint, then rewrite the real quotient using `hexact`.
                rw [hround]
                have hexactE :
                    ((toReal x / toReal y : ℝ) : EReal) =
                      (if sign then -(((num : ℝ) : EReal) / ((den : ℝ) : EReal))
                       else ((num : ℝ) : EReal) / ((den : ℝ) : EReal)) := by
                  have h := congrArg (fun r : ℝ => (r : EReal)) hexact
                  cases hsign : sign <;> simpa [hsign, EReal.coe_div] using h
                rw [hexactE]
                exact hle
            | negSucc sh =>
                let num : Nat := dx.mant
                let den : Nat := Nat.shiftLeft dy.mant (sh + 1)
                have hden0 : den ≠ 0 := shiftLeft_ne_zero dy.mant (sh + 1) hyMant0
                have hround : divDown x y = roundRatDown sign num den := by
                  simp (config := { zeta := true })
                    [divDown, hchoose, hxInf, hyInf, hy0, hdx, hdy, hx0', sign, hE, num, den]
                have hrat :
                    dyadicToReal dx / dyadicToReal dy =
                      (if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) := by
                  have h := dyadicToReal_div_eq_signedRat (dx := dx) (dy := dy) hyMant0
                  simpa [sign, hE, num, den] using h
                have hexact :
                    (toReal x / toReal y : ℝ) =
                      (if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) := by
                  simpa [toReal_eq, hdx, hdy] using hrat
                have hle0 :
                    toEReal (roundRatDown sign num den) ≤
                      ((if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) : EReal)
                        :=
                  toEReal_roundRatDown_le (sign := sign) (num := num) (den := den) hden0
                have hle :
                    toEReal (roundRatDown sign num den) ≤
                      (if sign then -(((num : ℝ) : EReal) / ((den : ℝ) : EReal))
                       else ((num : ℝ) : EReal) / ((den : ℝ) : EReal)) := by
                  cases hsign : sign <;> simpa [hsign, EReal.coe_div] using hle0
                rw [hround]
                have hexactE :
                    ((toReal x / toReal y : ℝ) : EReal) =
                      (if sign then -(((num : ℝ) : EReal) / ((den : ℝ) : EReal))
                       else ((num : ℝ) : EReal) / ((den : ℝ) : EReal)) := by
                  have h := congrArg (fun r : ℝ => (r : EReal)) hexact
                  cases hsign : sign <;> simpa [hsign, EReal.coe_div] using h
                rw [hexactE]
                exact hle

/--
Soundness of `divUp`: an upper enclosure for real division in `EReal`.

In the finite, nonzero-divisor regime this shows:
`(toReal x / toReal y) <= toEReal (divUp x y)`.
-/
theorem toEReal_divUp_ge (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) (hy0 : isZero y = false) :
    ((toReal x / toReal y : ℝ) : EReal) ≤ toEReal (divUp x y) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have hfin : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      cases (hx.symm.trans hfin)
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfin : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          cases (hy.symm.trans hfin)
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hdx)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hdy)
          have hyMant0 : dy.mant ≠ 0 := by
            intro hm
            have hz : isZero y = true := isZero_eq_true_of_toDyadic?_some_of_mant_eq_zero (hdy :=
              hdy) hm
            cases (hy0.symm.trans hz)

          by_cases hx0 : dx.mant = 0
          ·
            have hxR : (toReal x : ℝ) = 0 := by
              simp [toReal_eq, hdx, dyadicToReal, hx0]
            have hq : ((toReal x / toReal y : ℝ) : EReal) = (0 : EReal) := by
              rw [hxR]
              simp
            have hendpoint :
                divUp x y = (if Bool.xor dx.sign dy.sign then negZero else posZero) := by
              simp (config := { zeta := true })
                [divUp, hchoose, hxInf, hyInf, hy0, hdx, hdy, hx0]
            have hout : toEReal (divUp x y) = (0 : EReal) := by
              rw [hendpoint]
              simpa using toEReal_signedZero (Bool.xor dx.sign dy.sign)
            simp [hq, hout]
          ·
            have hx0' : (dx.mant == 0) = false := (beq_eq_false_iff_ne).2 hx0
            set sign : Bool := Bool.xor dx.sign dy.sign
            cases hE : (dx.exp - dy.exp) with
            | ofNat sh =>
                let num : Nat := Nat.shiftLeft dx.mant sh
                let den : Nat := dy.mant
                have hden0 : den ≠ 0 := hyMant0
                have hround : divUp x y = roundRatUp sign num den := by
                  simp (config := { zeta := true })
                    [divUp, hchoose, hxInf, hyInf, hy0, hdx, hdy, hx0', sign, hE, num, den]
                have hrat :
                    dyadicToReal dx / dyadicToReal dy =
                      (if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) := by
                  have h := dyadicToReal_div_eq_signedRat (dx := dx) (dy := dy) hyMant0
                  simpa [sign, hE, num, den] using h
                have hexact :
                    (toReal x / toReal y : ℝ) =
                      (if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) := by
                  simpa [toReal_eq, hdx, hdy] using hrat
                have hge0 :
                    ((if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) : EReal) ≤
                      toEReal (roundRatUp sign num den) :=
                  toEReal_roundRatUp_ge (sign := sign) (num := num) (den := den) hden0
                have hge :
                    (if sign then -(((num : ℝ) : EReal) / ((den : ℝ) : EReal))
                     else ((num : ℝ) : EReal) / ((den : ℝ) : EReal)) ≤
                      toEReal (roundRatUp sign num den) := by
                  cases hsign : sign <;> simpa [hsign, EReal.coe_div] using hge0
                rw [hround]
                have hexactE :
                    ((toReal x / toReal y : ℝ) : EReal) =
                      (if sign then -(((num : ℝ) : EReal) / ((den : ℝ) : EReal))
                       else ((num : ℝ) : EReal) / ((den : ℝ) : EReal)) := by
                  have h := congrArg (fun r : ℝ => (r : EReal)) hexact
                  cases hsign : sign <;> simpa [hsign, EReal.coe_div] using h
                rw [hexactE]
                exact hge
            | negSucc sh =>
                let num : Nat := dx.mant
                let den : Nat := Nat.shiftLeft dy.mant (sh + 1)
                have hden0 : den ≠ 0 := shiftLeft_ne_zero dy.mant (sh + 1) hyMant0
                have hround : divUp x y = roundRatUp sign num den := by
                  simp (config := { zeta := true })
                    [divUp, hchoose, hxInf, hyInf, hy0, hdx, hdy, hx0', sign, hE, num, den]
                have hrat :
                    dyadicToReal dx / dyadicToReal dy =
                      (if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) := by
                  have h := dyadicToReal_div_eq_signedRat (dx := dx) (dy := dy) hyMant0
                  simpa [sign, hE, num, den] using h
                have hexact :
                    (toReal x / toReal y : ℝ) =
                      (if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) := by
                  simpa [toReal_eq, hdx, hdy] using hrat
                have hge0 :
                    ((if sign then -((num : ℝ) / (den : ℝ)) else (num : ℝ) / (den : ℝ)) : EReal) ≤
                      toEReal (roundRatUp sign num den) :=
                  toEReal_roundRatUp_ge (sign := sign) (num := num) (den := den) hden0
                have hge :
                    (if sign then -(((num : ℝ) : EReal) / ((den : ℝ) : EReal))
                     else ((num : ℝ) : EReal) / ((den : ℝ) : EReal)) ≤
                      toEReal (roundRatUp sign num den) := by
                  cases hsign : sign <;> simpa [hsign, EReal.coe_div] using hge0
                rw [hround]
                have hexactE :
                    ((toReal x / toReal y : ℝ) : EReal) =
                      (if sign then -(((num : ℝ) : EReal) / ((den : ℝ) : EReal))
                       else ((num : ℝ) : EReal) / ((den : ℝ) : EReal)) := by
                  have h := congrArg (fun r : ℝ => (r : EReal)) hexact
                  cases hsign : sign <;> simpa [hsign, EReal.coe_div] using h
                rw [hexactE]
                exact hge

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
