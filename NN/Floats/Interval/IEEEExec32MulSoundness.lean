/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.EReal.Basic
public import Mathlib.Data.Real.Basic
public import NN.Floats.IEEEExec.DirectedRoundingSoundness
public import NN.Floats.IEEEExec.ERealSemantics
public import NN.Floats.Interval.ERealCoercions
public import NN.Floats.Interval.IEEEExec32
public import NN.Floats.Interval.IEEEExec32NoNaN
public import NN.Floats.Interval.RealBounds

/-!
# Soundness of `IEEE32Exec.Interval32.mul` (4-corner rule)

`NN/Floats/Interval/IEEEExec32.lean` defines executable endpoint intervals with IEEE32Exec
endpoints and outward-rounded arithmetic. Interval multiplication is implemented with the classical
“4-corner” rule:

```
[a,b] * [c,d] ⊆ [ min(ac, ad, bc, bd),  max(ac, ad, bc, bd) ].
```

In our executable implementation, each corner product is computed using directed rounding:
- `mulDown` for lower endpoints,
- `mulUp` for upper endpoints,
and then the minimum/maximum of the four corners is taken using IEEE `minimum`/`maximum`.

This file proves the main enclosure theorem connecting that executable construction to real
semantics (in `EReal` to allow overflow to `±∞`).

The theorem is stated in a *finite-input* regime (`isFinite` endpoints), which matches the setting
for interval bound propagation over real-valued networks. Overflow may still occur in intermediate
corner computations, and the proof handles it via `EReal`.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

namespace Interval32

noncomputable section

open TorchLean.Floats.Interval

/-! ## Real helper lemmas (min/max of 4 corners) -/

/--
`mulDown x y` is non-NaN on finite inputs.

This is a small bridge fact: it lets us apply the `toEReal_minimum_eq_min` lemma (which
requires a non-NaN hypothesis) when reasoning about the `min4` corner selection used by
`Interval32.mul`.
-/
private lemma isNaN_mulDown_eq_false_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    isNaN (mulDown x y) = false := by
  have hxNaN : isNaN x = false := isNaN_eq_false_of_isFinite_eq_true (x := x) hx
  have hyNaN : isNaN y = false := isNaN_eq_false_of_isFinite_eq_true (x := y) hy
  have hxInf : isInf x = false := isInf_eq_false_of_isFinite_eq_true (x := x) hx
  have hyInf : isInf y = false := isInf_eq_false_of_isFinite_eq_true (x := y) hy
  have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
  -- Reduce `mulDown` to the dyadic branch; finiteness guarantees decoding succeeds.
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        rw [hx] at h
        cases h
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfinFalse : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            have h := hfinFalse
            rw [hy] at h
            cases h
          exact this.elim
      | some dy =>
          -- Now `mulDown` returns either a signed zero or `roundDyadicDown` of the exact dyadic
          -- product.
          simp (config := { zeta := true }) [mulDown, hchoose, hxInf, hyInf, hdx, hdy]
          by_cases h0 : dx.mant = 0 ∨ dy.mant = 0
          · -- signed zero
            have hz : isNaN (if dx.sign = dy.sign then posZero else negZero) = false := by
              by_cases hs : dx.sign = dy.sign <;> simp [hs] <;> decide
            simp [h0, hz]
          · -- nonzero: `roundDyadicDown` never produces a NaN
            simp [h0, isNaN_roundDyadicDown_eq_false]

/--
`mulUp x y` is non-NaN on finite inputs.

As for `isNaN_mulDown_eq_false_of_isFinite`, this is used to justify that the IEEE `maximum`/`min4`
helpers behave like `max`/`min` on the `EReal` semantics of the rounded corner products.
-/
private lemma isNaN_mulUp_eq_false_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    isNaN (mulUp x y) = false := by
  have hxNaN : isNaN x = false := isNaN_eq_false_of_isFinite_eq_true (x := x) hx
  have hyNaN : isNaN y = false := isNaN_eq_false_of_isFinite_eq_true (x := y) hy
  have hxInf : isInf x = false := isInf_eq_false_of_isFinite_eq_true (x := x) hx
  have hyInf : isInf y = false := isInf_eq_false_of_isFinite_eq_true (x := y) hy
  have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_not_isNaN x y hxNaN hyNaN
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        rw [hx] at h
        cases h
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hfinFalse : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            have h := hfinFalse
            rw [hy] at h
            cases h
          exact this.elim
      | some dy =>
          simp (config := { zeta := true }) [mulUp, hchoose, hxInf, hyInf, hdx, hdy]
          by_cases h0 : dx.mant = 0 ∨ dy.mant = 0
          ·
            have hz : isNaN (if dx.sign = dy.sign then posZero else negZero) = false := by
              by_cases hs : dx.sign = dy.sign <;> simp [hs] <;> decide
            simp [h0, hz]
          ·
            simp [h0, isNaN_roundDyadicUp_eq_false]

/-! ## Interval multiplication soundness -/

/--
Soundness of `Interval32.mul` w.r.t. real multiplication:

If `x ∈ [A.lo, A.hi]` and `y ∈ [B.lo, B.hi]` (interpreted as real intervals), then
`x*y` lies in the real interval concretization of `Interval32.mul A B`.

The endpoints are interpreted in `EReal` so that overflow to `±∞` remains a sound enclosure.
-/
theorem mul_sound (A B : Interval32)
    (hAlo : isFinite A.lo = true) (hAhi : isFinite A.hi = true)
    (hBlo : isFinite B.lo = true) (hBhi : isFinite B.hi = true) :
    ∀ {x y : ℝ},
      x ∈ Set.Icc (toReal A.lo) (toReal A.hi) →
      y ∈ Set.Icc (toReal B.lo) (toReal B.hi) →
        toEReal (Interval32.mul A B).lo ≤ (x * y : EReal) ∧
          (x * y : EReal) ≤ toEReal (Interval32.mul A B).hi := by
  intro x y hx hy

  -- Real corner bounds.
  have hxy := mul_bounds_Icc (a := toReal A.lo) (b := toReal A.hi) (c := toReal B.lo) (d := toReal
    B.hi)
    (x := x) (y := y) hx hy
  have hxy_lo : (min4R (toReal A.lo * toReal B.lo) (toReal A.lo * toReal B.hi)
                    (toReal A.hi * toReal B.lo) (toReal A.hi * toReal B.hi) : ℝ) ≤ x * y := hxy.1
  have hxy_hi : x * y ≤ (max4R (toReal A.lo * toReal B.lo) (toReal A.lo * toReal B.hi)
                    (toReal A.hi * toReal B.lo) (toReal A.hi * toReal B.hi) : ℝ) := hxy.2

  -- Directed rounding corner products.
  let p00 := mulDown A.lo B.lo
  let p01 := mulDown A.lo B.hi
  let p10 := mulDown A.hi B.lo
  let p11 := mulDown A.hi B.hi
  let q00 := mulUp A.lo B.lo
  let q01 := mulUp A.lo B.hi
  let q10 := mulUp A.hi B.lo
  let q11 := mulUp A.hi B.hi

  have hp00NaN : isNaN p00 = false := isNaN_mulDown_eq_false_of_isFinite (x := A.lo) (y := B.lo)
    hAlo hBlo
  have hp01NaN : isNaN p01 = false := isNaN_mulDown_eq_false_of_isFinite (x := A.lo) (y := B.hi)
    hAlo hBhi
  have hp10NaN : isNaN p10 = false := isNaN_mulDown_eq_false_of_isFinite (x := A.hi) (y := B.lo)
    hAhi hBlo
  have hp11NaN : isNaN p11 = false := isNaN_mulDown_eq_false_of_isFinite (x := A.hi) (y := B.hi)
    hAhi hBhi

  have hq00NaN : isNaN q00 = false := isNaN_mulUp_eq_false_of_isFinite (x := A.lo) (y := B.lo) hAlo
    hBlo
  have hq01NaN : isNaN q01 = false := isNaN_mulUp_eq_false_of_isFinite (x := A.lo) (y := B.hi) hAlo
    hBhi
  have hq10NaN : isNaN q10 = false := isNaN_mulUp_eq_false_of_isFinite (x := A.hi) (y := B.lo) hAhi
    hBlo
  have hq11NaN : isNaN q11 = false := isNaN_mulUp_eq_false_of_isFinite (x := A.hi) (y := B.hi) hAhi
    hBhi

  have hp00_le : toEReal p00 ≤ ((toReal A.lo * toReal B.lo : ℝ) : EReal) :=
    toEReal_mulDown_le (x := A.lo) (y := B.lo) hAlo hBlo
  have hp01_le : toEReal p01 ≤ ((toReal A.lo * toReal B.hi : ℝ) : EReal) :=
    toEReal_mulDown_le (x := A.lo) (y := B.hi) hAlo hBhi
  have hp10_le : toEReal p10 ≤ ((toReal A.hi * toReal B.lo : ℝ) : EReal) :=
    toEReal_mulDown_le (x := A.hi) (y := B.lo) hAhi hBlo
  have hp11_le : toEReal p11 ≤ ((toReal A.hi * toReal B.hi : ℝ) : EReal) :=
    toEReal_mulDown_le (x := A.hi) (y := B.hi) hAhi hBhi

  have hq00_ge : ((toReal A.lo * toReal B.lo : ℝ) : EReal) ≤ toEReal q00 :=
    toEReal_mulUp_ge (x := A.lo) (y := B.lo) hAlo hBlo
  have hq01_ge : ((toReal A.lo * toReal B.hi : ℝ) : EReal) ≤ toEReal q01 :=
    toEReal_mulUp_ge (x := A.lo) (y := B.hi) hAlo hBhi
  have hq10_ge : ((toReal A.hi * toReal B.lo : ℝ) : EReal) ≤ toEReal q10 :=
    toEReal_mulUp_ge (x := A.hi) (y := B.lo) hAhi hBlo
  have hq11_ge : ((toReal A.hi * toReal B.hi : ℝ) : EReal) ≤ toEReal q11 :=
    toEReal_mulUp_ge (x := A.hi) (y := B.hi) hAhi hBhi

  -- Lower endpoint: min of rounded-down corners ≤ min of exact corners ≤ x*y.
  have hlo_eval :
      toEReal (Interval32.mul A B).lo =
        min (min (toEReal p00) (toEReal p01)) (min (toEReal p10) (toEReal p11)) := by
    -- unfold `mul` and rewrite `toEReal (min4 ...)`
    simp [Interval32.mul, p00, p01, p10, p11,
      toEReal_min4_eq (a := p00) (b := p01) (c := p10) (d := p11) hp00NaN hp01NaN hp10NaN hp11NaN]

  have hhi_eval :
      toEReal (Interval32.mul A B).hi =
        max (max (toEReal q00) (toEReal q01)) (max (toEReal q10) (toEReal q11)) := by
    simp [Interval32.mul, q00, q01, q10, q11,
      toEReal_max4_eq (a := q00) (b := q01) (c := q10) (d := q11) hq00NaN hq01NaN hq10NaN hq11NaN]

  have hlo_le_minCorners :
      toEReal (Interval32.mul A B).lo ≤
        ((min4R (toReal A.lo * toReal B.lo) (toReal A.lo * toReal B.hi)
          (toReal A.hi * toReal B.lo) (toReal A.hi * toReal B.hi) : ℝ) : EReal) := by
    -- Rewrite `toEReal lo` into nested `min`, then use monotonicity of `min`.
    rw [hlo_eval]
    -- Build the corresponding nested `min` in `EReal` of the *exact* corners.
    have h01 :
        min (toEReal p00) (toEReal p01) ≤ min ((toReal A.lo * toReal B.lo : ℝ) : EReal)
              ((toReal A.lo * toReal B.hi : ℝ) : EReal) :=
      min_le_min hp00_le hp01_le
    have h23 :
        min (toEReal p10) (toEReal p11) ≤ min ((toReal A.hi * toReal B.lo : ℝ) : EReal)
              ((toReal A.hi * toReal B.hi : ℝ) : EReal) :=
      min_le_min hp10_le hp11_le
    have hmin :
        min (min (toEReal p00) (toEReal p01)) (min (toEReal p10) (toEReal p11)) ≤
          min (min ((toReal A.lo * toReal B.lo : ℝ) : EReal) ((toReal A.lo * toReal B.hi : ℝ) :
            EReal))
              (min ((toReal A.hi * toReal B.lo : ℝ) : EReal) ((toReal A.hi * toReal B.hi : ℝ) :
                EReal)) :=
      min_le_min h01 h23
    refine le_trans hmin ?_
    -- Convert the `EReal`-nested min into a coercion of `min4R`.
    have : (min (min ((toReal A.lo * toReal B.lo : ℝ) : EReal) ((toReal A.lo * toReal B.hi : ℝ) :
      EReal))
            (min ((toReal A.hi * toReal B.lo : ℝ) : EReal) ((toReal A.hi * toReal B.hi : ℝ) :
              EReal))) =
          ((min4R (toReal A.lo * toReal B.lo) (toReal A.lo * toReal B.hi)
            (toReal A.hi * toReal B.lo) (toReal A.hi * toReal B.hi) : ℝ) : EReal) := by
      -- Regroup the `min`s and push coercions through `min`.
      simp [min4R, coe_min, min_left_comm, min_comm]
    exact le_of_eq this

  have hmaxCorners_le_hhi :
      ((max4R (toReal A.lo * toReal B.lo) (toReal A.lo * toReal B.hi)
          (toReal A.hi * toReal B.lo) (toReal A.hi * toReal B.hi) : ℝ) : EReal) ≤
        toEReal (Interval32.mul A B).hi := by
    rw [hhi_eval]
    have h01 :
        max ((toReal A.lo * toReal B.lo : ℝ) : EReal) ((toReal A.lo * toReal B.hi : ℝ) : EReal) ≤
          max (toEReal q00) (toEReal q01) :=
      max_le_max hq00_ge hq01_ge
    have h23 :
        max ((toReal A.hi * toReal B.lo : ℝ) : EReal) ((toReal A.hi * toReal B.hi : ℝ) : EReal) ≤
          max (toEReal q10) (toEReal q11) :=
      max_le_max hq10_ge hq11_ge
    have hmax :
        max (max ((toReal A.lo * toReal B.lo : ℝ) : EReal) ((toReal A.lo * toReal B.hi : ℝ) :
          EReal))
            (max ((toReal A.hi * toReal B.lo : ℝ) : EReal) ((toReal A.hi * toReal B.hi : ℝ) :
              EReal)) ≤
          max (max (toEReal q00) (toEReal q01)) (max (toEReal q10) (toEReal q11)) :=
      max_le_max h01 h23
    refine le_trans ?_ hmax
    -- Convert the `max4R` coercion to the nested `max` over the four exact corners.
    have : ((max4R (toReal A.lo * toReal B.lo) (toReal A.lo * toReal B.hi)
        (toReal A.hi * toReal B.lo) (toReal A.hi * toReal B.hi) : ℝ) : EReal) =
          max (max ((toReal A.lo * toReal B.lo : ℝ) : EReal) ((toReal A.lo * toReal B.hi : ℝ) :
            EReal))
              (max ((toReal A.hi * toReal B.lo : ℝ) : EReal) ((toReal A.hi * toReal B.hi : ℝ) :
                EReal)) := by
      simp [max4R, coe_max, max_left_comm, max_comm]
    exact le_of_eq this

  -- Cast real bounds into `EReal`.
  have hxy_loE : ((min4R (toReal A.lo * toReal B.lo) (toReal A.lo * toReal B.hi)
        (toReal A.hi * toReal B.lo) (toReal A.hi * toReal B.hi) : ℝ) : EReal) ≤ (x * y : EReal) :=
          by
    exact (EReal.coe_le_coe_iff).2 hxy_lo
  have hxy_hiE : (x * y : EReal) ≤ ((max4R (toReal A.lo * toReal B.lo) (toReal A.lo * toReal B.hi)
        (toReal A.hi * toReal B.lo) (toReal A.hi * toReal B.hi) : ℝ) : EReal) := by
    exact (EReal.coe_le_coe_iff).2 hxy_hi

  have hlo : toEReal (Interval32.mul A B).lo ≤ (x * y : EReal) :=
    le_trans hlo_le_minCorners hxy_loE
  have hhi : (x * y : EReal) ≤ toEReal (Interval32.mul A B).hi :=
    le_trans hxy_hiE hmaxCorners_le_hhi

  exact ⟨hlo, hhi⟩

end

end Interval32

end IEEE32Exec

end TorchLean.Floats.IEEE754
