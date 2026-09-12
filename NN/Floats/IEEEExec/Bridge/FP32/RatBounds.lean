/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32.Core

/-!
# IEEE32Exec and FP32: Rational Magnitude Bounds
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/-!
## Real semantics of exponent tests for rationals

Some executable rounding code works by inspecting the magnitude of a rational (represented as
`num / den` with `Nat`s) and branching on exponent ranges. These lemmas justify those branches in
terms of real inequalities, so later refinement proofs can stay “math-first”.
-/

lemma ratLtPow2_eq_true_iff (num den : Nat) (k : Int) (hden : den ≠ 0) :
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

/-- The executable comparison against a power of two is correct on `ℝ`.

The implementation cross-multiplies to stay in `Nat`, so the two exponent signs are separate cases;
nothing here rounds, which is why the statement is an iff rather than a bound. -/
lemma ratGePow2_eq_true_iff (num den : Nat) (k : Int) (hden : den ≠ 0) :
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

lemma bpow_k0_sub_one_eq (ln ld : Nat) :
    neuralBpow binaryRadix (Int.ofNat ln - Int.ofNat ld - 1) =
      (2 : ℝ) ^ ln / (2 : ℝ) ^ ld.succ := by
  simp [TorchLean.Floats.neuralBpow, binaryRadix, NeuralRadix.toReal, zpow_sub₀]
  simp [pow_succ, div_eq_mul_inv, mul_left_comm, mul_comm]

/-- Arithmetic step: shifting the log-difference by one is multiplying by two. -/
lemma bpow_k0_add_one_eq (ln ld : Nat) :
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

/-- The difference of the integer `log2`s brackets the quotient within one power of two.

A difference of logs can be off by one either way, so this is deliberately a two-power window;
`floorLog2Rat_bounds` narrows it to the exact exponent with one comparison. -/
lemma rat_bounds_k0 (num den : Nat) (hnum : num ≠ 0) (hden : den ≠ 0) :
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
    have hk : k0 - 1 = Int.ofNat ln - Int.ofNat ld - 1 := by
      simp [k0, sub_eq_add_neg, add_assoc]
    simpa [hk] using this
  · have : (num : ℝ) / (den : ℝ) < neuralBpow binaryRadix (k0 + 1) := by
      rw [hbpow_hi]
      exact hhi_pow
    have hk : k0 + 1 = Int.ofNat ln - Int.ofNat ld + 1 := by
      simp [k0, sub_eq_add_neg, add_assoc]
    simpa [hk] using this

/-- `floorLog2Rat` really is the floor of the base-two logarithm of a positive rational.

This is the entry point the rounding code uses to find a starting exponent, and the tight bracket is
what makes the subsequent normalization loop terminate immediately. -/
lemma floorLog2Rat_bounds (num den : Nat) (hnum : num ≠ 0) (hden : den ≠ 0) :
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

lemma neural_nearest_even_eq_zero_of_abs_lt_half (x : ℝ) (hx : _root_.abs x < (1 / 2 : ℝ)) :
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
              exact zpow_natCast (2 : ℝ) l.succ
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

/--
Lower companion to `abs_dyadicToReal_lt_bpow_succ_log2`.

For a nonzero mantissa, `2^(log2 mant + exp)` is no larger than the decoded dyadic magnitude.
-/
theorem bpow_log2_add_exp_le_abs_dyadicToReal (d : Dyadic) (hm : d.mant ≠ 0) :
    neuralBpow binaryRadix (Int.ofNat (Nat.log2 d.mant) + d.exp) ≤
      _root_.abs (dyadicToReal d) := by
  have hpowNat : 2 ^ Nat.log2 d.mant ≤ d.mant := by
    have hlog : Nat.log 2 d.mant = Nat.log2 d.mant := by
      simpa using (Nat.log2_eq_log_two (n := d.mant)).symm
    simpa [hlog] using Nat.pow_log_le_self 2 hm
  have hpowReal : ((2 ^ Nat.log2 d.mant : Nat) : ℝ) ≤ (d.mant : ℝ) := by
    exact_mod_cast hpowNat
  calc
    neuralBpow binaryRadix (Int.ofNat (Nat.log2 d.mant) + d.exp) =
        neuralBpow binaryRadix (Int.ofNat (Nat.log2 d.mant)) *
          neuralBpow binaryRadix d.exp := neuralBpow.add_exp binaryRadix _ _
    _ = ((2 ^ Nat.log2 d.mant : Nat) : ℝ) * neuralBpow binaryRadix d.exp := by
      simp [neuralBpow, binaryRadix, NeuralRadix.toReal]
    _ ≤ (d.mant : ℝ) * neuralBpow binaryRadix d.exp :=
      mul_le_mul_of_nonneg_right hpowReal (neuralBpow.nonneg binaryRadix _)
    _ = _root_.abs (dyadicToReal d) := (abs_dyadicToReal d).symm
end IEEE32Exec

end TorchLean.Floats.IEEE754
