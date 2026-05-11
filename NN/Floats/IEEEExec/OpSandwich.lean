/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.EReal.Basic
public import NN.Floats.IEEEExec.Exec32
public import NN.Floats.IEEEExec.BridgeFP32Total
public import NN.Floats.IEEEExec.ERealSemantics
public import NN.Floats.IEEEExec.RoundDyadicToIEEE32Bounds
public import NN.Floats.IEEEExec.SpecialRules

/-!
# Nearest-even lies between directed roundings (op-level corollaries)

`NN.Floats.IEEEExec.RoundDyadicToIEEE32Bounds` proves the core theorem:

`roundDyadicDown d ≤ roundDyadicToIEEE32 d ≤ roundDyadicUp d` (in `EReal`).

This file packages small **op-level** corollaries for `IEEE32Exec.add` / `mul` / `sub` on the
finite path:

`addDown x y ≤ add x y ≤ addUp x y`, and similarly for multiplication and subtraction.

These are useful for “checked boundary => enclosure” statements in downstream code (e.g. RL shadow
interval diagnostics).
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

open TorchLean.Floats

noncomputable section

/-! ## Small helpers -/

private lemma chooseNaN2_none_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    chooseNaN2 x y = none := by
  have hxNaN : isNaN x = false := isNaN_eq_false_of_isFinite_eq_true (x := x) hx
  have hyNaN : isNaN y = false := isNaN_eq_false_of_isFinite_eq_true (x := y) hy
  simpa using (chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN)

private lemma isFinite_neg_of_isFinite (x : IEEE32Exec) (hx : isFinite x = true) :
    isFinite (neg x) = true := by
  -- Use the dyadic decode witness to avoid bit-level facts about exponent fields.
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by simp [hx] at this
      exact this.elim
  | some d =>
      have hdxNeg :
          toDyadic? (neg x) = some { sign := (!d.sign), mant := d.mant, exp := d.exp } :=
        toDyadic?_neg_of_toDyadic?_some (x := x) (d := d) hdx
      have hnan : isNaN (neg x) = false := isNaN_eq_false_of_toDyadic?_some (hx := hdxNeg)
      have hinf : isInf (neg x) = false := isInf_eq_false_of_toDyadic?_some (hx := hdxNeg)
      exact isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := neg x) hnan hinf

/-!
## Addition sandwich

On the finite path, `add`/`addDown`/`addUp` all reduce to rounding the same exact dyadic sum with
different rounding modes.
-/

theorem toEReal_addDown_le_add_le_addUp_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toEReal (addDown x y) ≤ toEReal (add x y) ∧
      toEReal (add x y) ≤ toEReal (addUp x y) := by
  classical
  have hxInf : isInf x = false := isInf_eq_false_of_isFinite_eq_true (x := x) hx
  have hyInf : isInf y = false := isInf_eq_false_of_isFinite_eq_true (x := y) hy
  have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_isFinite (x := x) (y := y) hx hy
  -- Extract dyadic witnesses from finiteness.
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by simp [hx] at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by simp [hy] at this
          exact this.elim
      | some dy =>
          have hadd :
              add x y = roundDyadicToIEEE32 (addDyadic dx dy) := by
            simp [IEEE32Exec.add, hchoose, hxInf, hyInf, hdx, hdy]
          have haddDown :
              addDown x y = roundDyadicDown (addDyadic dx dy) := by
            simp [IEEE32Exec.addDown, hchoose, hxInf, hyInf, hdx, hdy]
          have haddUp :
              addUp x y = roundDyadicUp (addDyadic dx dy) := by
            simp [IEEE32Exec.addUp, hchoose, hxInf, hyInf, hdx, hdy]
          simpa [hadd, haddDown, haddUp] using
            (toEReal_roundDyadicDown_le_roundDyadicToIEEE32_le_roundDyadicUp (d := addDyadic dx dy))

/-!
## Multiplication sandwich

On finite inputs, `mul`/`mulDown`/`mulUp` reduce to rounding the same exact dyadic product with
different rounding modes.
-/

theorem toEReal_mulDown_le_mul_le_mulUp_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toEReal (mulDown x y) ≤ toEReal (mul x y) ∧
      toEReal (mul x y) ≤ toEReal (mulUp x y) := by
  classical
  have hxInf : isInf x = false := isInf_eq_false_of_isFinite_eq_true (x := x) hx
  have hyInf : isInf y = false := isInf_eq_false_of_isFinite_eq_true (x := y) hy
  have hchoose : chooseNaN2 x y = none := chooseNaN2_none_of_isFinite (x := x) (y := y) hx hy
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by simp [hx] at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by simp [hy] at this
          exact this.elim
      | some dy =>
          let prod : Dyadic :=
            { sign := Bool.xor dx.sign dy.sign, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
          have hmul :
              mul x y = roundDyadicToIEEE32 prod := by
            -- Unfold `mul`; the `mant==0` case is handled by `roundDyadicToIEEE32` on `prod`.
            simp (config := { zeta := true }) [IEEE32Exec.mul, hchoose, hxInf, hyInf, hdx, hdy, prod]
            · intro h0
              have hmant : dx.mant * dy.mant = 0 := by
                rcases h0 with hx0 | hy0
                · simp [hx0]
                · simp [hy0]
              -- Reduce `roundDyadicToIEEE32` in the `mant = 0` branch and discharge the sign case split.
              cases dx.sign <;> cases dy.sign <;> simp [IEEE32Exec.roundDyadicToIEEE32, hmant]
          have hmulDown :
              mulDown x y = roundDyadicDown prod := by
            simp (config := { zeta := true }) [IEEE32Exec.mulDown, hchoose, hxInf, hyInf, hdx, hdy, prod]
            · intro h0
              have hmant : dx.mant * dy.mant = 0 := by
                rcases h0 with hx0 | hy0
                · simp [hx0]
                · simp [hy0]
              cases dx.sign <;> cases dy.sign <;> simp [IEEE32Exec.roundDyadicDown, hmant]
          have hmulUp :
              mulUp x y = roundDyadicUp prod := by
            simp (config := { zeta := true }) [IEEE32Exec.mulUp, hchoose, hxInf, hyInf, hdx, hdy, prod]
            · intro h0
              have hmant : dx.mant * dy.mant = 0 := by
                rcases h0 with hx0 | hy0
                · simp [hx0]
                · simp [hy0]
              cases dx.sign <;> cases dy.sign <;> simp [IEEE32Exec.roundDyadicUp, hmant]
          simpa [hmul, hmulDown, hmulUp] using
            (toEReal_roundDyadicDown_le_roundDyadicToIEEE32_le_roundDyadicUp (d := prod))

/-!
## Subtraction sandwich

This is a direct corollary of the addition sandwich, since:

`sub x y = add x (neg y)` and similarly for directed endpoints.
-/

theorem toEReal_subDown_le_sub_le_subUp_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toEReal (subDown x y) ≤ toEReal (sub x y) ∧
      toEReal (sub x y) ≤ toEReal (subUp x y) := by
  have hyNeg : isFinite (neg y) = true := isFinite_neg_of_isFinite (x := y) hy
  -- Reduce to the addition sandwich.
  simpa [IEEE32Exec.sub, IEEE32Exec.subDown, IEEE32Exec.subUp] using
    (toEReal_addDown_le_add_le_addUp_of_isFinite (x := x) (y := neg y) hx hyNeg)

end

end IEEE32Exec
end TorchLean.Floats.IEEE754
