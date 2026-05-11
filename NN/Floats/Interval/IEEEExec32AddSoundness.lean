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
public import NN.Floats.Interval.IEEEExec32

/-!
# Soundness of `IEEE32Exec.Interval32.add` / `sub` / `neg`

This file proves the “golden theorem”-style enclosure results for the *monotone* interval
operations:

- addition: `[a,b] + [c,d] ⊆ [a+c, b+d]`,
- negation: `-[a,b] = [-b, -a]`,
- subtraction: `[a,b] - [c,d] ⊆ [a-d, b-c]` (a derived combination of addition + negation).

In `NN/Floats/Interval/IEEEExec32.lean`, these are implemented with IEEE32 executable
  outward-rounded
endpoints:

- `add` uses `addDown` / `addUp`,
- `sub` uses `subDown` / `subUp`, where `subDown x y = addDown x (neg y)` and similarly for `subUp`.

We work in `EReal` so that later pipelines can compose these lemmas with the multiplication/division
soundness lemmas (which are naturally overflow-aware). In the finite-input regime considered here,
addition/subtraction themselves cannot overflow to `±∞`, but phrasing the result in `EReal` keeps
  the
API uniform.

Standards alignment (informal):
- IEEE 754-2019 defines binary32 arithmetic and special values (NaN/Inf/signed zero).
- IEEE 1788-2015 defines a standard API and semantics for interval arithmetic; the enclosures above
  are the basic “set-based” interval laws in the *valid* (outer enclosure) mode.

Note: we do **not** attempt to prove bit-for-bit conformance of `IEEE32Exec` to any specific
  CPU/GPU.
Instead, we prove that our *executable model* (which follows IEEE-style rules) has the stated
enclosure relationship to the real semantics on the finite path.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

namespace Interval32

noncomputable section

/-! ## Small helpers -/

/--
Negation preserves finiteness.

We use this to justify that `subDown a b = addDown a (neg b)` is still in the finite regime when
`b` is finite, so we can reuse directed-rounding soundness lemmas that assume finiteness.
-/
private lemma isFinite_neg_of_isFinite (x : IEEE32Exec) (hx : isFinite x = true) :
    isFinite (IEEE32Exec.neg x) = true := by
  -- Use the dyadic decode witness to avoid bit-twiddling facts about exponent fields.
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        rw [hx] at h
        cases h
      exact this.elim
  | some d =>
      have hdxNeg :
          toDyadic? (IEEE32Exec.neg x) = some { sign := (!d.sign), mant := d.mant, exp := d.exp } :=
        toDyadic?_neg_of_toDyadic?_some (x := x) (d := d) hdx
      have hnan : isNaN (IEEE32Exec.neg x) = false := isNaN_eq_false_of_toDyadic?_some (hx :=
        hdxNeg)
      have hinf : isInf (IEEE32Exec.neg x) = false := isInf_eq_false_of_toDyadic?_some (hx :=
        hdxNeg)
      exact isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := IEEE32Exec.neg x) hnan hinf

/--
Real semantics of `IEEE32Exec.neg` on finite inputs.

This is the expected arithmetic law: for finite `x`, decoding `-x` gives the negation of the
decoded real value.
-/
private lemma toReal_neg_eq_neg_of_isFinite (x : IEEE32Exec) (hx : isFinite x = true) :
    toReal (IEEE32Exec.neg x) = -toReal x := by
  cases hdx : toDyadic? x with
  | none =>
      have hfinFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have h := hfinFalse
        rw [hx] at h
        cases h
      exact this.elim
  | some d =>
      simpa using (toReal_neg_eq_neg (x := x) (d := d) hdx)

/-! ## Interval addition -/

/--
Soundness of `Interval32.add` w.r.t. real addition, in the finite-endpoint regime.

If `x ∈ [A.lo,A.hi]` and `y ∈ [B.lo,B.hi]` in real semantics, then `x+y` lies between the `EReal`
interpretation of the executable endpoints of `Interval32.add A B`.
-/
theorem add_sound (A B : Interval32) (hA : Valid A) (hB : Valid B) :
    ∀ {x y : ℝ},
      x ∈ Set.Icc (toReal A.lo) (toReal A.hi) →
      y ∈ Set.Icc (toReal B.lo) (toReal B.hi) →
        toEReal (Interval32.add A B).lo ≤ (x + y : EReal) ∧
          (x + y : EReal) ≤ toEReal (Interval32.add A B).hi := by
  intro x y hx hy
  have hAlo : isFinite A.lo = true := hA.1
  have hAhi : isFinite A.hi = true := hA.2.1
  have hBlo : isFinite B.lo = true := hB.1
  have hBhi : isFinite B.hi = true := hB.2.1

  have hdown : toEReal (addDown A.lo B.lo) ≤ ((toReal A.lo + toReal B.lo : ℝ) : EReal) :=
    toEReal_addDown_le (x := A.lo) (y := B.lo) hAlo hBlo
  have hup : ((toReal A.hi + toReal B.hi : ℝ) : EReal) ≤ toEReal (addUp A.hi B.hi) :=
    toEReal_addUp_ge (x := A.hi) (y := B.hi) hAhi hBhi

  have hsum_lo : toReal A.lo + toReal B.lo ≤ x + y := add_le_add hx.1 hy.1
  have hsum_hi : x + y ≤ toReal A.hi + toReal B.hi := add_le_add hx.2 hy.2
  have hsum_loE : ((toReal A.lo + toReal B.lo : ℝ) : EReal) ≤ (x + y : EReal) :=
    (EReal.coe_le_coe_iff).2 hsum_lo
  have hsum_hiE : (x + y : EReal) ≤ ((toReal A.hi + toReal B.hi : ℝ) : EReal) :=
    (EReal.coe_le_coe_iff).2 hsum_hi

  constructor
  · simpa [Interval32.add] using le_trans hdown hsum_loE
  · simpa [Interval32.add] using le_trans hsum_hiE hup

/-! ## Interval negation -/

/--
Soundness of `Interval32.neg` w.r.t. real negation, in the finite-endpoint regime.

If `x ∈ [A.lo,A.hi]` in real semantics, then `-x` lies between the `EReal` interpretation of the
executable endpoints of `Interval32.neg A`.
-/
theorem neg_sound (A : Interval32) (hA : Valid A) :
    ∀ {x : ℝ},
      x ∈ Set.Icc (toReal A.lo) (toReal A.hi) →
        toEReal (Interval32.neg A).lo ≤ (-x : EReal) ∧
          (-x : EReal) ≤ toEReal (Interval32.neg A).hi := by
  intro x hx
  have hAlo : isFinite A.lo = true := hA.1
  have hAhi : isFinite A.hi = true := hA.2.1
  have hnanLo : isNaN A.lo = false := isNaN_eq_false_of_isFinite_eq_true (x := A.lo) hAlo
  have hnanHi : isNaN A.hi = false := isNaN_eq_false_of_isFinite_eq_true (x := A.hi) hAhi

  have hnegLo : toEReal (IEEE32Exec.neg A.hi) = -toEReal A.hi :=
    toEReal_neg_of_isNaN_eq_false (x := A.hi) hnanHi
  have hnegHi : toEReal (IEEE32Exec.neg A.lo) = -toEReal A.lo :=
    toEReal_neg_of_isNaN_eq_false (x := A.lo) hnanLo
  have hcoeLo : toEReal A.lo = (toReal A.lo : EReal) := toEReal_eq_coe_toReal_of_isFinite (x :=
    A.lo) hAlo
  have hcoeHi : toEReal A.hi = (toReal A.hi : EReal) := toEReal_eq_coe_toReal_of_isFinite (x :=
    A.hi) hAhi

  have hlo : -toReal A.hi ≤ -x := neg_le_neg hx.2
  have hhi : -x ≤ -toReal A.lo := neg_le_neg hx.1
  have hloE : ((-toReal A.hi : ℝ) : EReal) ≤ (-x : EReal) := (EReal.coe_le_coe_iff).2 hlo
  have hhiE : (-x : EReal) ≤ ((-toReal A.lo : ℝ) : EReal) := (EReal.coe_le_coe_iff).2 hhi

  constructor
  · -- lower endpoint
    have : toEReal (Interval32.neg A).lo = ((-toReal A.hi : ℝ) : EReal) := by
      simp [Interval32.neg, hnegLo, hcoeHi]
    simpa [this] using hloE
  · -- upper endpoint
    have : toEReal (Interval32.neg A).hi = ((-toReal A.lo : ℝ) : EReal) := by
      simp [Interval32.neg, hnegHi, hcoeLo]
    simpa [this] using hhiE

/-! ## Interval subtraction -/

/--
Soundness of `Interval32.sub` w.r.t. real subtraction, in the finite-endpoint regime.

This is a derived enclosure rule:
`[a,b] - [c,d] ⊆ [a-d, b-c]`, implemented with directed rounding via `subDown/subUp`.
-/
theorem sub_sound (A B : Interval32) (hA : Valid A) (hB : Valid B) :
    ∀ {x y : ℝ},
      x ∈ Set.Icc (toReal A.lo) (toReal A.hi) →
      y ∈ Set.Icc (toReal B.lo) (toReal B.hi) →
        toEReal (Interval32.sub A B).lo ≤ (x - y : EReal) ∧
          (x - y : EReal) ≤ toEReal (Interval32.sub A B).hi := by
  intro x y hx hy
  have hAlo : isFinite A.lo = true := hA.1
  have hAhi : isFinite A.hi = true := hA.2.1
  have hBlo : isFinite B.lo = true := hB.1
  have hBhi : isFinite B.hi = true := hB.2.1

  have hBhiNeg : isFinite (IEEE32Exec.neg B.hi) = true := isFinite_neg_of_isFinite (x := B.hi) hBhi
  have hBloNeg : isFinite (IEEE32Exec.neg B.lo) = true := isFinite_neg_of_isFinite (x := B.lo) hBlo

  have hnegHi : toReal (IEEE32Exec.neg B.hi) = -toReal B.hi := toReal_neg_eq_neg_of_isFinite (x :=
    B.hi) hBhi
  have hnegLo : toReal (IEEE32Exec.neg B.lo) = -toReal B.lo := toReal_neg_eq_neg_of_isFinite (x :=
    B.lo) hBlo

  -- Lower endpoint: `A.lo - B.hi`.
  have hdown : toEReal (subDown A.lo B.hi) ≤ ((toReal A.lo - toReal B.hi : ℝ) : EReal) := by
    -- `subDown a b = addDown a (neg b)`
    have h :=
      toEReal_addDown_le (x := A.lo) (y := IEEE32Exec.neg B.hi) hAlo hBhiNeg
    simpa [IEEE32Exec.subDown, hnegHi, sub_eq_add_neg] using h

  -- Upper endpoint: `A.hi - B.lo`.
  have hup : ((toReal A.hi - toReal B.lo : ℝ) : EReal) ≤ toEReal (subUp A.hi B.lo) := by
    have h :=
      toEReal_addUp_ge (x := A.hi) (y := IEEE32Exec.neg B.lo) hAhi hBloNeg
    simpa [IEEE32Exec.subUp, hnegLo, sub_eq_add_neg] using h

  have hsub_lo : toReal A.lo - toReal B.hi ≤ x - y := sub_le_sub hx.1 hy.2
  have hsub_hi : x - y ≤ toReal A.hi - toReal B.lo := sub_le_sub hx.2 hy.1
  have hsub_loE : ((toReal A.lo - toReal B.hi : ℝ) : EReal) ≤ (x - y : EReal) :=
    (EReal.coe_le_coe_iff).2 hsub_lo
  have hsub_hiE : (x - y : EReal) ≤ ((toReal A.hi - toReal B.lo : ℝ) : EReal) :=
    (EReal.coe_le_coe_iff).2 hsub_hi

  constructor
  ·
    have hlo : toEReal (Interval32.sub A B).lo ≤ (x - y : EReal) := by
      simpa [Interval32.sub] using le_trans hdown hsub_loE
    exact hlo
  ·
    have hhi : (x - y : EReal) ≤ toEReal (Interval32.sub A B).hi := by
      simpa [Interval32.sub] using le_trans hsub_hiE hup
    exact hhi

end

end Interval32

end IEEE32Exec

end TorchLean.Floats.IEEE754
