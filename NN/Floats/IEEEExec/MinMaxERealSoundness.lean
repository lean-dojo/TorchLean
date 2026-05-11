/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.EReal.Basic
public import NN.Floats.IEEEExec.BridgeFP32Total
public import NN.Floats.IEEEExec.ERealSemantics

/-!
# `minimum`/`maximum` in `EReal` semantics (IEEE32Exec)

`NN/Floats/IEEEExec/Exec32.lean` defines executable IEEE-754 comparison operators:

- `compare` (unordered if either operand is NaN),
- `minimum` / `maximum` (NaNs propagate; tie-break on signed zeros).

For interval endpoint soundness, we interpret float32 values in the extended reals:

- finite values map to `↑(toReal x)`,
- `+∞` / `-∞` map to `⊤` / `⊥`,
- NaN is excluded by assumption (the theorems below assume `isNaN = false`).

This file provides the bridge lemmas used by interval arithmetic proofs:

```
toEReal (minimum x y) = min (toEReal x) (toEReal y)
toEReal (maximum x y) = max (toEReal x) (toEReal y)
```

including the overflow cases where one of `x`/`y` is an infinity.

References:
- IEEE 754-2019: doi:10.1109/IEEESTD.2019.8766229
- Goldberg (1991): doi:10.1145/103162.103163
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

private lemma coe_min (a b : ℝ) : ((min a b : ℝ) : EReal) = min (a : EReal) (b : EReal) := by
  by_cases h : a ≤ b
  · have hE : (a : EReal) ≤ (b : EReal) := by simpa [EReal.coe_le_coe_iff] using h
    simp [min_eq_left h, min_eq_left hE]
  · have h' : b ≤ a := le_of_not_ge h
    have hE : (b : EReal) ≤ (a : EReal) := by simpa [EReal.coe_le_coe_iff] using h'
    simp [min_eq_right h', min_eq_right hE]

private lemma coe_max (a b : ℝ) : ((max a b : ℝ) : EReal) = max (a : EReal) (b : EReal) := by
  by_cases h : a ≤ b
  · have hE : (a : EReal) ≤ (b : EReal) := by simpa [EReal.coe_le_coe_iff] using h
    simp [max_eq_right h, max_eq_right hE]
  · have h' : b ≤ a := le_of_not_ge h
    have hE : (b : EReal) ≤ (a : EReal) := by simpa [EReal.coe_le_coe_iff] using h'
    simp [max_eq_left h', max_eq_left hE]

/-! ## Main bridge lemmas -/

/--
`minimum` agrees with `min` on the `toEReal` interpretation, assuming no NaNs.
-/
theorem toEReal_minimum_eq_min (x y : IEEE32Exec)
    (hxNaN : isNaN x = false) (hyNaN : isNaN y = false) :
    toEReal (minimum x y) = min (toEReal x) (toEReal y) := by
  have hchoose : chooseNaN2 x y = none :=
    chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
  by_cases hxInf : isInf x = true
  · by_cases hyInf : isInf y = true
    · -- Both infinities: case split on sign bits.
      have hx0 : isZero x = false := isZero_eq_false_of_isInf (x := x) hxInf
      have hy0 : isZero y = false := isZero_eq_false_of_isInf (x := y) hyInf
      cases hsx : signBit x <;> cases hsy : signBit y <;>
        simp [minimum, compare, toEReal_eq_ite, *]
    · -- `x` is ±∞, `y` is finite: `minimum` returns `x` iff `x=-∞`, else returns `y`.
      cases hsx : signBit x <;>
        simp [minimum, compare, toEReal_eq_ite, *]
  · have hxInf' : isInf x = false := by simpa using hxInf
    by_cases hyInf : isInf y = true
    · -- `x` finite, `y` is ±∞: `minimum` returns `y` iff `y=-∞`, else returns `x`.
      cases hsy : signBit y <;>
        simp [minimum, compare, toEReal_eq_ite, *]
    · -- Both finite: reduce to the `toReal` bridge theorem.
      have hyInf' : isInf y = false := by simpa using hyInf
      have hxFin : isFinite x = true :=
        isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := x) hxNaN hxInf'
      have hyFin : isFinite y = true :=
        isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := y) hyNaN hyInf'
      have hminFin : isFinite (minimum x y) = true := isFinite_minimum_of_isFinite (x := x) (y := y)
        hxFin hyFin
      have htoReal : toReal (minimum x y) = min (toReal x) (toReal y) :=
        toReal_minimum_eq_min_of_isFinite (x := x) (y := y) hxFin hyFin
      calc
        toEReal (minimum x y) = (toReal (minimum x y) : EReal) :=
          toEReal_eq_coe_toReal_of_isFinite (x := minimum x y) hminFin
        _ = ((min (toReal x) (toReal y) : ℝ) : EReal) := by
          rw [htoReal]
        _ = min (toReal x : EReal) (toReal y : EReal) := coe_min _ _
        _ = min (toEReal x) (toEReal y) := by
          simp [toEReal_eq_coe_toReal_of_isFinite (x := x) hxFin,
            toEReal_eq_coe_toReal_of_isFinite (x := y) hyFin]

/--
`maximum` agrees with `max` on the `toEReal` interpretation, assuming no NaNs.
-/
theorem toEReal_maximum_eq_max (x y : IEEE32Exec)
    (hxNaN : isNaN x = false) (hyNaN : isNaN y = false) :
    toEReal (maximum x y) = max (toEReal x) (toEReal y) := by
  have hchoose : chooseNaN2 x y = none :=
    chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
  by_cases hxInf : isInf x = true
  · by_cases hyInf : isInf y = true
    ·
      have hx0 : isZero x = false := isZero_eq_false_of_isInf (x := x) hxInf
      have hy0 : isZero y = false := isZero_eq_false_of_isInf (x := y) hyInf
      cases hsx : signBit x <;> cases hsy : signBit y <;>
        simp [maximum, compare, toEReal_eq_ite, *]
    · cases hsx : signBit x <;>
        simp [maximum, compare, toEReal_eq_ite, *]
  · have hxInf' : isInf x = false := by simpa using hxInf
    by_cases hyInf : isInf y = true
    · cases hsy : signBit y <;>
        simp [maximum, compare, toEReal_eq_ite, *]
    · have hyInf' : isInf y = false := by simpa using hyInf
      have hxFin : isFinite x = true :=
        isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := x) hxNaN hxInf'
      have hyFin : isFinite y = true :=
        isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := y) hyNaN hyInf'
      have hmaxFin : isFinite (maximum x y) = true := isFinite_maximum_of_isFinite (x := x) (y := y)
        hxFin hyFin
      have htoReal : toReal (maximum x y) = max (toReal x) (toReal y) :=
        toReal_maximum_eq_max_of_isFinite (x := x) (y := y) hxFin hyFin
      calc
        toEReal (maximum x y) = (toReal (maximum x y) : EReal) :=
          toEReal_eq_coe_toReal_of_isFinite (x := maximum x y) hmaxFin
        _ = ((max (toReal x) (toReal y) : ℝ) : EReal) := by
          rw [htoReal]
        _ = max (toReal x : EReal) (toReal y : EReal) := coe_max _ _
        _ = max (toEReal x) (toEReal y) := by
          simp [toEReal_eq_coe_toReal_of_isFinite (x := x) hxFin,
            toEReal_eq_coe_toReal_of_isFinite (x := y) hyFin]

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
