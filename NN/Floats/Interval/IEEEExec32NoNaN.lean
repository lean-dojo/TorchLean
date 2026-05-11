/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.DirectedRoundingSoundness
public import NN.Floats.IEEEExec.MinMaxERealSoundness
public import NN.Floats.Interval.IEEEExec32

/-!
# Non-NaN helpers for `IEEE32Exec` interval soundness proofs

Several `IEEE32Exec.Interval32` enclosure proofs need to establish that intermediate helper
computations do not produce NaNs:

- directed-rounding kernels such as `roundDyadicDown` / `roundDyadicUp`,
- comparison-based helpers `minimum` / `maximum` (used by `min4` / `max4`).

This file centralizes those facts so the soundness proofs for interval multiplication/division can
share the same non-NaN case analysis.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-!
## Dyadic rounding is never NaN

These facts are used to show that `mulDown`/`mulUp` and `divDown`/`divUp` do not produce NaNs on the
finite, nonzero-denominator path.
-/

/-- `roundDyadicPosDown` never produces a NaN. -/
theorem isNaN_roundDyadicPosDown_eq_false (mant : Nat) (exp : Int) :
    isNaN (roundDyadicPosDown mant exp) = false := by
  obtain ⟨d, hd⟩ := IEEE32Exec.toDyadic?_roundDyadicPosDown_some (mant := mant) (exp := exp)
  exact isNaN_eq_false_of_toDyadic?_some (x := roundDyadicPosDown mant exp) (d := d) hd

/-- `roundDyadicDown` never produces a NaN. -/
theorem isNaN_roundDyadicDown_eq_false (d : Dyadic) :
    isNaN (roundDyadicDown d) = false := by
  by_cases hm0 : d.mant == 0
  · -- signed zero
    cases hs : d.sign <;> simp [roundDyadicDown, hm0, hs] <;> decide
  · have hm0' : d.mant ≠ 0 := by
      intro h0
      have : (d.mant == 0) = true := (beq_iff_eq).2 h0
      simp [this] at hm0
    cases hs : d.sign <;> simp [roundDyadicDown, hm0, hs]
    · -- positive: `roundDyadicPosDown` is never NaN
      simpa using isNaN_roundDyadicPosDown_eq_false (mant := d.mant) (exp := d.exp)
    · -- negative: `neg (roundDyadicPosUp ..)` and `roundDyadicPosUp ..` is never NaN
      have hnanPos : isNaN (roundDyadicPosUp d.mant d.exp) = false :=
        IEEE32Exec.isNaN_roundDyadicPosUp_eq_false (mant := d.mant) (exp := d.exp) hm0'
      simpa using
        (IEEE32Exec.isNaN_neg_eq_false_of_isNaN_eq_false (x := roundDyadicPosUp d.mant d.exp)
          hnanPos)

/-- `roundDyadicUp` never produces a NaN. -/
theorem isNaN_roundDyadicUp_eq_false (d : Dyadic) :
    isNaN (roundDyadicUp d) = false := by
  by_cases hm0 : d.mant == 0
  · cases hs : d.sign <;> simp [roundDyadicUp, hm0, hs] <;> decide
  · have hm0' : d.mant ≠ 0 := by
      intro h0
      have : (d.mant == 0) = true := (beq_iff_eq).2 h0
      simp [this] at hm0
    cases hs : d.sign <;> simp [roundDyadicUp, hm0, hs]
    · -- positive: `roundDyadicPosUp` is never NaN
      simpa using IEEE32Exec.isNaN_roundDyadicPosUp_eq_false (mant := d.mant) (exp := d.exp) hm0'
    · -- negative: `neg (roundDyadicPosDown ..)` and `roundDyadicPosDown ..` is never NaN
      have hnanPos : isNaN (roundDyadicPosDown d.mant d.exp) = false :=
        isNaN_roundDyadicPosDown_eq_false (mant := d.mant) (exp := d.exp)
      simpa using
        (IEEE32Exec.isNaN_neg_eq_false_of_isNaN_eq_false (x := roundDyadicPosDown d.mant d.exp)
          hnanPos)

/-!
## `minimum` / `maximum` are non-NaN on non-NaN inputs

These facts allow us to apply `toEReal_minimum_eq_min` / `toEReal_maximum_eq_max` from
`NN/Floats/IEEEExec/MinMaxERealSoundness.lean` without threading through a large amount of
comparison lemmas.
-/

/--
If neither input is NaN, then IEEE `compare` is never unordered (`none`).

This is a small bridge lemma used to reason about `minimum`/`maximum`, which dispatch on the
`compare` result and treat the unordered case as NaN-propagation.
-/
theorem compare_ne_none_of_isNaN_eq_false (x y : IEEE32Exec)
    (hx : isNaN x = false) (hy : isNaN y = false) :
    compare x y ≠ none := by
  intro hcmp
  unfold compare at hcmp
  -- eliminate the NaN guard first
  have hcmp' :
      (if isInf x then
          if isInf y then
            if signBit x == signBit y then some .eq
            else if signBit x then some .lt else some .gt
          else if signBit x then some .lt else some .gt
        else if isInf y then
          if signBit y then some .gt else some .lt
        else
          match toDyadic? x, toDyadic? y with
          | some dx, some dy => some (cmpDyadic dx dy)
          | _, _ => none) =
        (none : Option Ordering) := by
    simpa [hx, hy] using hcmp
  clear hcmp
  -- show this cannot happen by splitting on the remaining cases
  cases hxInf : isInf x with
  | true =>
      cases hyInf : isInf y with
      | true =>
          cases hsx : signBit x <;> cases hsy : signBit y <;>
            simp [hxInf, hyInf, hsx, hsy] at hcmp'
      | false =>
          cases hsx : signBit x <;>
            simp [hxInf, hyInf, hsx] at hcmp'
  | false =>
      cases hyInf : isInf y with
      | true =>
          cases hsy : signBit y <;>
            simp [hxInf, hyInf, hsy] at hcmp'
      | false =>
          -- finite branch: `toDyadic?` must succeed (it only fails on NaN/Inf)
          have hxFin : isFinite x = true :=
            isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := x) hx (by simpa using hxInf)
          have hyFin : isFinite y = true :=
            isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := y) hy (by simpa using hyInf)
          cases hdx : toDyadic? x with
          | none =>
              have hxFinFalse : isFinite x = false :=
                isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
              have : False := by
                simp [hxFin] at hxFinFalse
              exact this.elim
          | some dx =>
              cases hdy : toDyadic? y with
              | none =>
                  have hyFinFalse : isFinite y = false :=
                    isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
                  have : False := by
                    simp [hyFin] at hyFinFalse
                  exact this.elim
              | some dy =>
                  simp [hxInf, hyInf, hdx, hdy] at hcmp'

/--
`minimum x y` is non-NaN whenever both inputs are non-NaN.

This matches the IEEE-754 intent: `minimum` propagates NaNs if present, and otherwise returns one
of the inputs (with a special tie-break rule for signed zeros).
-/
theorem isNaN_minimum_eq_false_of_isNaN_eq_false (x y : IEEE32Exec)
    (hx : isNaN x = false) (hy : isNaN y = false) :
    isNaN (minimum x y) = false := by
  have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hx hy
  have hcmp_ne : compare x y ≠ none := compare_ne_none_of_isNaN_eq_false (x := x) (y := y) hx hy
  unfold minimum
  simp [hchoose]
  cases hcmp : compare x y with
  | none =>
      exfalso
      exact hcmp_ne hcmp
  | some o =>
      cases o with
      | lt =>
          -- `minimum` returns `x`
          simp [hx]
      | gt =>
          -- `minimum` returns `y`
          simp [hy]
      | eq =>
          -- tie-breaking: if both are ±0, the result is either `-0` or `+0`; otherwise it is `x`.
          by_cases hz : isZero x = true ∧ isZero y = true
          · by_cases hs : signBit x = true ∨ signBit y = true
            · simp [hz, hs]; decide
            · simp [hz, hs]; decide
          · simp [hz, hx]

/--
`maximum x y` is non-NaN whenever both inputs are non-NaN.

As with `minimum`, the only way for `maximum` to produce NaN is to propagate an input NaN or to
encounter an unordered `compare` (which can only happen with NaNs).
-/
theorem isNaN_maximum_eq_false_of_isNaN_eq_false (x y : IEEE32Exec)
    (hx : isNaN x = false) (hy : isNaN y = false) :
    isNaN (maximum x y) = false := by
  have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hx hy
  have hcmp_ne : compare x y ≠ none := compare_ne_none_of_isNaN_eq_false (x := x) (y := y) hx hy
  unfold maximum
  simp [hchoose]
  cases hcmp : compare x y with
  | none =>
      exfalso
      exact hcmp_ne hcmp
  | some o =>
      cases o with
      | lt =>
          -- `maximum` returns `y`
          simp [hy]
      | gt =>
          -- `maximum` returns `x`
          simp [hx]
      | eq =>
          -- tie-breaking: if both are ±0, the result is either `+0` or `-0`; otherwise it is `x`.
          by_cases hz : isZero x = true ∧ isZero y = true
          · by_cases hs : signBit x = false ∨ signBit y = false
            · simp [hz, hs]; decide
            · simp [hz, hs]; decide
          · simp [hz, hx]

namespace Interval32

/-- `toEReal` semantics of `min4` in the non-NaN regime. -/
theorem toEReal_min4_eq (a b c d : IEEE32Exec)
    (ha : isNaN a = false) (hb : isNaN b = false) (hc : isNaN c = false) (hd : isNaN d = false) :
    toEReal (Interval32.min4 a b c d) =
      min (min (toEReal a) (toEReal b)) (min (toEReal c) (toEReal d)) := by
  have habNaN : isNaN (minimum a b) = false :=
    isNaN_minimum_eq_false_of_isNaN_eq_false (x := a) (y := b) ha hb
  have hcdNaN : isNaN (minimum c d) = false :=
    isNaN_minimum_eq_false_of_isNaN_eq_false (x := c) (y := d) hc hd
  unfold Interval32.min4
  -- Outer minimum
  rw [IEEE32Exec.toEReal_minimum_eq_min (x := minimum a b) (y := minimum c d) habNaN hcdNaN]
  -- Inner minima
  simp [IEEE32Exec.toEReal_minimum_eq_min (x := a) (y := b) ha hb,
    IEEE32Exec.toEReal_minimum_eq_min (x := c) (y := d) hc hd]

/-- `toEReal` semantics of `max4` in the non-NaN regime. -/
theorem toEReal_max4_eq (a b c d : IEEE32Exec)
    (ha : isNaN a = false) (hb : isNaN b = false) (hc : isNaN c = false) (hd : isNaN d = false) :
    toEReal (Interval32.max4 a b c d) =
      max (max (toEReal a) (toEReal b)) (max (toEReal c) (toEReal d)) := by
  have habNaN : isNaN (maximum a b) = false :=
    isNaN_maximum_eq_false_of_isNaN_eq_false (x := a) (y := b) ha hb
  have hcdNaN : isNaN (maximum c d) = false :=
    isNaN_maximum_eq_false_of_isNaN_eq_false (x := c) (y := d) hc hd
  unfold Interval32.max4
  rw [IEEE32Exec.toEReal_maximum_eq_max (x := maximum a b) (y := maximum c d) habNaN hcdNaN]
  simp [IEEE32Exec.toEReal_maximum_eq_max (x := a) (y := b) ha hb,
    IEEE32Exec.toEReal_maximum_eq_max (x := c) (y := d) hc hd]

end Interval32

end -- noncomputable section

end IEEE32Exec

end TorchLean.Floats.IEEE754

end -- public section
