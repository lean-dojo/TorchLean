/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Nat.Basic
public import Mathlib.Order.Basic
import Mathlib.Tactic.Attr.Core
import Mathlib.Tactic.Push
import Mathlib.Tactic.Set
public import NN.Floats.IEEEExec.Exec32.Dyadic

/-!
# Order bounds for `roundQuotEven`

`IEEE32Exec.roundRatToIEEE32` rounds an *exact rational* `num/den` to binary32 using
round-to-nearest, ties-to-even.

At the integer level it relies on:

```
roundQuotEven num den : Nat
```

which rounds the rational quotient to the nearest integer (ties-to-even), assuming `den > 0`.

For later "nearest vs directed" arguments, we only need the simple fact that
`roundQuotEven num den` is always either the floor quotient `q = num / den` or the next integer
`q+1`. This is the exact analogue of `roundShiftRightEven_eq_q_or_q_add1` for the power-of-two
division case.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

open Nat

private lemma roundQuotEven_eq_q_or_q_add1 (num den : Nat) :
    let q := num / den
    roundQuotEven num den = q ∨ roundQuotEven num den = q + 1 := by
  classical
  by_cases hden : den = 0
  · subst hden
    -- Totalized behavior: `num/0 = 0`, `num%0 = num`.
    -- The comparisons reduce to whether `2*num` is `0` (i.e. `num = 0`).
    by_cases hnum : num = 0
    · subst hnum
      simp [roundQuotEven]
    · -- `num ≠ 0` implies `2*num > 0`, so `twice > den` and we return `q+1 = 1`.
        have hpos : 0 < 2 * num := by
          exact Nat.mul_pos (by decide : 0 < (2 : Nat)) (Nat.pos_of_ne_zero hnum)
        have hgt : 2 * num > 0 := hpos
        simp [roundQuotEven, hgt]
  ·
    -- Main (intended) case: `den > 0`. The result is always `q` or `q+1`.
    set q : Nat := num / den
    set r : Nat := num % den
    set twice : Nat := 2 * r
    by_cases hlt : twice < den
    · left
      simp [roundQuotEven, q, r, twice, hlt]
    · by_cases hgt : twice > den
      · right
        simp [roundQuotEven, q, r, twice, hlt, hgt]
      · -- tie case (`twice = den`): ties-to-even picks `q` or `q+1`.
        by_cases heven : q % 2 = 0
        · left
          simp [roundQuotEven, q, r, twice, hlt, hgt, heven]
        · right
          simp [roundQuotEven, q, r, twice, hlt, hgt, heven]

/-- `roundQuotEven num den` is either the floor quotient `num/den` or that value plus one. -/
theorem roundQuotEven_eq_div_or_div_add1 (num den : Nat) :
    roundQuotEven num den = num / den ∨ roundQuotEven num den = num / den + 1 := by
  -- Wrapper around the `q`-based statement used by downstream bounds.
  simpa using (roundQuotEven_eq_q_or_q_add1 (num := num) (den := den))

/-- Lower bound: the floor quotient is at most `roundQuotEven num den`. -/
lemma div_le_roundQuotEven (num den : Nat) :
    num / den ≤ roundQuotEven num den := by
  classical
  set q : Nat := num / den
  have hor := roundQuotEven_eq_q_or_q_add1 (num := num) (den := den)
  have : roundQuotEven num den = q ∨ roundQuotEven num den = q + 1 := by
    simpa [q] using hor
  rcases this with hq | hq
  · simp [hq, q]
  · simp [hq, q]

/-- Upper bound: `roundQuotEven num den` is at most the floor quotient plus one. -/
lemma roundQuotEven_le_div_add1 (num den : Nat) :
    roundQuotEven num den ≤ num / den + 1 := by
  classical
  set q : Nat := num / den
  have hor := roundQuotEven_eq_q_or_q_add1 (num := num) (den := den)
  have : roundQuotEven num den = q ∨ roundQuotEven num den = q + 1 := by
    simpa [q] using hor
  rcases this with hq | hq
  · simp [hq, q]
  · simp [hq, q]

/-- A nonnegative rational strictly below one half rounds to zero. -/
theorem roundQuotEven_eq_zero_of_two_mul_lt
    (num den : Nat) (h : 2 * num < den) :
    roundQuotEven num den = 0 := by
  have hnum : num < den := by grind
  simp [roundQuotEven, Nat.div_eq_of_lt hnum, Nat.mod_eq_of_lt hnum, h]

/-- A rational lower bound is inherited by its nearest-even integer rounding. -/
theorem le_roundQuotEven_of_mul_le
    (num den lower : Nat) (hden : den ≠ 0) (h : lower * den ≤ num) :
    lower ≤ roundQuotEven num den := by
  have hdenPos : 0 < den := Nat.pos_of_ne_zero hden
  have hfloor : lower ≤ num / den := (Nat.le_div_iff_mul_le hdenPos).2 h
  exact hfloor.trans (div_le_roundQuotEven num den)

/-- A strict rational upper bound is a weak upper bound after nearest-even rounding. -/
theorem roundQuotEven_le_of_lt_mul
    (num den upper : Nat) (hden : den ≠ 0) (h : num < upper * den) :
    roundQuotEven num den ≤ upper := by
  have hdenPos : 0 < den := Nat.pos_of_ne_zero hden
  have hfloor : num / den < upper := (Nat.div_lt_iff_lt_mul hdenPos).2 h
  exact (roundQuotEven_le_div_add1 num den).trans (by grind)

end IEEE32Exec
end TorchLean.Floats.IEEE754
