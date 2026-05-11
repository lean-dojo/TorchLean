/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32
public import NN.Floats.IEEEExec.NatLemmas

/-!
# Basic order bounds for `roundShiftRightEven`

`IEEE32Exec.roundShiftRightEven` is the core integer-rounding primitive used by
`roundDyadicToIEEE32`:

- We first compute an *exact* mantissa `n` (a `Nat`).
- We then shrink it by `shift` bits to land on the target mantissa width.
- The policy is **round-to-nearest, ties-to-even**.

For interval verification we also implement **directed rounding** (toward `-∞` and `+∞`):

- floor-quotient: `q = n >>> shift`
- ceil-quotient:  `ceil(n / 2^shift)` which we implement as `shiftRightCeilPow2 n shift`

The key structural fact (used later to relate "nearest" to "directed" rounding) is:

> `roundShiftRightEven n shift` always lies between the floor and ceil quotients.

In other words, round-to-nearest can only choose `q` or `q+1`, and those are exactly the two
directed-rounding candidates at this scale.

We prove that here in a way that is robust to later refactors of `roundDyadicToIEEE32`:
these lemmas talk only about the low-level Nat operations, not about floats.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

namespace IEEE32Exec

open Nat

/--
`roundShiftRightEven` returns either the floor quotient `q = n >>> shift` or `q+1`.

This is the core structural fact we use to derive order bounds: nearest-even rounding can only
move the quotient by at most one ulp in the shifted domain.
-/
private lemma roundShiftRightEven_eq_q_or_q_add1 (n shift : Nat) :
    let q := Nat.shiftRight n shift
    roundShiftRightEven n shift = q ∨ roundShiftRightEven n shift = q + 1 := by
  classical
  by_cases h0 : (shift == 0) = true
  · have hs : shift = 0 := Eq.mp (Nat.beq_eq_true_eq shift 0) h0
    subst hs
    simp [roundShiftRightEven]
  · -- `shift > 0` branch: unfold and split on the remainder comparisons.
    have hk0 : (shift == 0) = false := by
      cases hs : (shift == 0) <;> try rfl
      exfalso; exact h0 hs
    set q : Nat := Nat.shiftRight n shift
    set rem : Nat := n - Nat.shiftLeft q shift
    set half : Nat := pow2 (shift - 1)
    have hdef :
        roundShiftRightEven n shift =
          if rem < half then q
          else if rem > half then q + 1
          else if q % 2 = 0 then q else q + 1 := by
      simp (config := { zeta := true }) [roundShiftRightEven, hk0, q, rem, half]
    -- Now do a direct case split on `rem < half` / `rem > half` / tie.
    by_cases hlt : rem < half
    · left
      simp [hdef, hlt]
    · by_cases hgt : rem > half
      · right
        simp [hdef, hlt, hgt]
      · -- tie case (`rem = half`): pick `q` or `q+1` depending on parity.
        by_cases heven : q % 2 = 0
        · left
          simp [hdef, hlt, hgt, heven]
        · right
          simp [hdef, hlt, hgt, heven]

/-!
### Public “two-point” characterization

The nearest-even quotient can only be one of the two directed-division candidates:

- the floor quotient `q = n >>> shift`, or
- the next integer `q + 1`.

This public lemma connects
`roundDyadicToIEEE32` (nearest rounding) to `roundDyadicPosDown`/`roundDyadicPosUp`
(directed rounding).
-/

/--
Nearest-even shift-right rounding yields one of two adjacent candidates.

Informal: `roundShiftRightEven n shift` is either the floor quotient `n >>> shift` or that value
plus one.
-/
theorem roundShiftRightEven_eq_shiftRight_or_shiftRight_add1 (n shift : Nat) :
    roundShiftRightEven n shift = Nat.shiftRight n shift ∨
      roundShiftRightEven n shift = Nat.shiftRight n shift + 1 := by
  -- This is just `roundShiftRightEven_eq_q_or_q_add1` with `q := n >>> shift`.
  simpa using (roundShiftRightEven_eq_q_or_q_add1 (n := n) (shift := shift))

/-- Nearest-even rounding is never below the floor-quotient (`shiftRight`). -/
lemma shiftRight_le_roundShiftRightEven (n shift : Nat) :
    Nat.shiftRight n shift ≤ roundShiftRightEven n shift := by
  classical
  set q : Nat := Nat.shiftRight n shift
  have hor := (roundShiftRightEven_eq_q_or_q_add1 (n := n) (shift := shift))
  have : roundShiftRightEven n shift = q ∨ roundShiftRightEven n shift = q + 1 := by
    simpa [q] using hor
  rcases this with hq | hq
  · simp [hq, q]
  · simp [hq, q]

/-- Nearest-even rounding is never above `shiftRight n shift + 1`. -/
lemma roundShiftRightEven_le_shiftRight_add1 (n shift : Nat) :
    roundShiftRightEven n shift ≤ Nat.shiftRight n shift + 1 := by
  classical
  set q : Nat := Nat.shiftRight n shift
  have hor := (roundShiftRightEven_eq_q_or_q_add1 (n := n) (shift := shift))
  have : roundShiftRightEven n shift = q ∨ roundShiftRightEven n shift = q + 1 := by
    simpa [q] using hor
  rcases this with hq | hq
  · simp [hq, q]
  · simp [hq, q]

/--
Nearest-even rounding is bounded above by the “ceiling quotient” `shiftRightCeilPow2`.

This is useful when connecting nearest-even rounding to **directed rounding up**, since the latter
is often implemented as “divide by 2^k and round the quotient up”.
-/
lemma roundShiftRightEven_le_shiftRightCeilPow2 (n shift : Nat) :
    roundShiftRightEven n shift ≤ shiftRightCeilPow2 n shift := by
  classical
  by_cases h0 : (shift == 0) = true
  · have hs : shift = 0 := Eq.mp (Nat.beq_eq_true_eq shift 0) h0
    subst hs
    simp [roundShiftRightEven, shiftRightCeilPow2]
  · have hk0 : (shift == 0) = false := by
      cases hs : (shift == 0) <;> try rfl
      exfalso; exact h0 hs
    set q : Nat := Nat.shiftRight n shift
    set rem : Nat := n - Nat.shiftLeft q shift
    have hceil :
        shiftRightCeilPow2 n shift = (if rem == 0 then q else q + 1) := by
      simp (config := { zeta := true }) [shiftRightCeilPow2, hk0, q, rem]
    by_cases hrem : rem = 0
    · -- `ceil = q`, and `rem = 0` forces nearest rounding to pick `q` (since `0 < half`).
      have hceilq : shiftRightCeilPow2 n shift = q := by simp [hceil, hrem]
      have hhalfPos : 0 < pow2 (shift - 1) := pow2_pos (shift - 1)
      have hdef :
          roundShiftRightEven n shift =
            if rem < pow2 (shift - 1) then q
            else if rem > pow2 (shift - 1) then q + 1
            else if q % 2 = 0 then q else q + 1 := by
        simp (config := { zeta := true }) [roundShiftRightEven, hk0, q, rem]
      have hlt : rem < pow2 (shift - 1) := by
        -- `rem = 0` and `pow2 (shift-1) > 0`.
        simpa [hrem] using hhalfPos
      have hnearest : roundShiftRightEven n shift = q := by
        simp [hdef, hlt]
      simp [hceilq, hnearest]
    · have hceilq1 : shiftRightCeilPow2 n shift = q + 1 := by simp [hceil, hrem]
      have hle : roundShiftRightEven n shift ≤ q + 1 := by
        simpa [q] using (roundShiftRightEven_le_shiftRight_add1 (n := n) (shift := shift))
      simpa [hceilq1] using hle

/-- Convenience bundling of the lower+upper bounds used most often downstream. -/
lemma shiftRight_le_roundShiftRightEven_le_shiftRightCeilPow2 (n shift : Nat) :
    Nat.shiftRight n shift ≤ roundShiftRightEven n shift ∧
      roundShiftRightEven n shift ≤ shiftRightCeilPow2 n shift := by
  exact ⟨shiftRight_le_roundShiftRightEven (n := n) (shift := shift),
    roundShiftRightEven_le_shiftRightCeilPow2 (n := n) (shift := shift)⟩

end IEEE32Exec

end TorchLean.Floats.IEEE754
