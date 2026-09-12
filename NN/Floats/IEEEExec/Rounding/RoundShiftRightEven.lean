/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Rounding.NatLemmas
public import Mathlib.Algebra.GroupWithZero.Nat
public import Mathlib.Algebra.NeZero
public import Mathlib.Data.Nat.Basic
public import Mathlib.Order.Basic
import Mathlib.Tactic.Attr.Core
import Mathlib.Tactic.Push
import Mathlib.Tactic.Set
public import NN.Floats.IEEEExec.Exec32.Directed

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

/--
If nearest-even shift rounding returns the positive integer `r`, then the unrounded numerator is
strictly larger than `(r - 1) * 2^shift`.

The strict inequality matters at the top of the binary32 range: a rounded significand of `2^24`
must come from a value strictly above the largest 24-bit significand `2^24 - 1`.
-/
theorem pred_mul_pow2_lt_of_roundShiftRightEven_eq (n shift r : Nat)
    (hr : roundShiftRightEven n shift = r) (hrpos : 0 < r) :
    (r - 1) * pow2 shift < n := by
  classical
  by_cases hshift : shift = 0
  · subst shift
    have hr' : n = r := by simpa [roundShiftRightEven] using hr
    subst r
    simp [pow2]
    grind
  · let q := Nat.shiftRight n shift
    let d := pow2 shift
    have hdpos : 0 < d := pow2_pos shift
    have hqdle : q * d ≤ n := by
      simpa [q, d, Nat.shiftRight_eq_div_pow, pow2_eq_two_pow] using
        (Nat.div_mul_le_self n (2 ^ shift))
    rcases roundShiftRightEven_eq_shiftRight_or_shiftRight_add1 n shift with hfloor | hceil
    · have hrq : r = q := hr.symm.trans (by simpa [q] using hfloor)
      rw [hrq] at hrpos ⊢
      have hqpos : 0 < q := hrpos
      exact
        (Nat.mul_lt_mul_of_pos_right (Nat.pred_lt (Nat.ne_of_gt hqpos)) hdpos).trans_le hqdle
    · have hrq : r = q + 1 := hr.symm.trans (by simpa [q] using hceil)
      rw [hrq] at hrpos ⊢
      have hqdlt : q * d < n := by
        apply lt_of_le_of_ne hqdle
        intro heq
        have hshiftb : (shift == 0) = false := by simp [hshift]
        have hshiftLeft : Nat.shiftLeft q shift = n := by
          simpa [d, Nat.shiftLeft_eq, pow2_eq_two_pow] using heq
        have hshiftLeft' : Nat.shiftLeft (Nat.shiftRight n shift) shift = n := by
          simpa [q] using hshiftLeft
        have hrem : n - Nat.shiftLeft (Nat.shiftRight n shift) shift = 0 := by
          rw [hshiftLeft']
          simp
        have hhalfPos : 0 < pow2 (shift - 1) := pow2_pos (shift - 1)
        let rem := n - Nat.shiftLeft q shift
        have hrem0 : rem = 0 := by simpa [rem, q] using hrem
        have hdef :
            roundShiftRightEven n shift =
              if rem < pow2 (shift - 1) then q
              else if rem > pow2 (shift - 1) then q + 1
              else if q % 2 = 0 then q else q + 1 := by
          simp (config := { zeta := true }) [roundShiftRightEven, hshiftb, q, rem]
        have hroundq : roundShiftRightEven n shift = q := by
          rw [hdef]
          simp [hrem0, hhalfPos]
        grind
      simpa using hqdlt

/-- Convenience bundling of the lower+upper bounds used most often downstream. -/
lemma shiftRight_le_roundShiftRightEven_le_shiftRightCeilPow2 (n shift : Nat) :
    Nat.shiftRight n shift ≤ roundShiftRightEven n shift ∧
      roundShiftRightEven n shift ≤ shiftRightCeilPow2 n shift := by
  exact ⟨shiftRight_le_roundShiftRightEven (n := n) (shift := shift),
    roundShiftRightEven_le_shiftRightCeilPow2 (n := n) (shift := shift)⟩

end IEEE32Exec

end TorchLean.Floats.IEEE754
