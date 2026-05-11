/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.EReal.Basic
public import NN.Floats.IEEEExec.BridgeERealTotal

/-!
# Total `EReal` semantics helpers for `IEEE32Exec`

`BridgeERealTotal.lean` defines `toEReal? : IEEE32Exec → Option EReal`, which is the *canonical*
extended-real interpretation we use when we want to distinguish `+∞` from `-∞` and treat NaN as
"unordered" (`none`).

In a few proof scripts, it is convenient to have a *total* function into `EReal`. We therefore
define:

* `toEReal : IEEE32Exec → EReal`, which totalizes `toEReal?` by mapping the NaN case to `0`.

Informal reading rule:

If you see a theorem stated using `toEReal`, it should be read under hypotheses that exclude NaNs
(`isNaN x = false`). Under that assumption, `toEReal x` agrees with the intended `toEReal?` meaning
and behaves exactly as you would expect:

* finite values map to `↑(toReal x)`,
* `+∞`/`-∞` map to `⊤`/`⊥`.

This file also collects small helper lemmas (`toEReal_eq_ite`, signed-zero simp facts, etc.) used by
multiple IEEEExec proof modules.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-!
## `toEReal` (totalized) and unfolding helpers
-/

/--
Total `EReal` interpretation of `IEEE32Exec`.

This is `toEReal?` with the NaN case mapped to `0`.

Intended invariant:
Most theorems about interval endpoints assume `isNaN x = false`. Under that precondition, the NaN
branch is unreachable and `toEReal` agrees with the partial `EReal` semantics.
-/
noncomputable def toEReal (x : IEEE32Exec) : EReal :=
  match toEReal? x with
  | some r => r
  | none => 0

/-- If `toEReal? x = some r`, then `toEReal x = r`. -/
@[simp] lemma toEReal_of_toEReal? {x : IEEE32Exec} {r : EReal} (h : toEReal? x = some r) :
    toEReal x = r := by
  simp [toEReal, h]

/-- If `toEReal? x = none`, then `toEReal x = 0`. -/
@[simp] lemma toEReal_of_toEReal?_none {x : IEEE32Exec} (h : toEReal? x = none) :
    toEReal x = (0 : EReal) := by
  simp [toEReal, h]

/--
Convenient unfolding lemma for `toEReal`.

This is the form that works best with `simp` once you have established/assumed (non-)NaN and
(non-)Inf facts.
-/
theorem toEReal_eq_ite (x : IEEE32Exec) :
    toEReal x =
      if isNaN x then
        (0 : EReal)
      else if isInf x then
        if signBit x then (⊥ : EReal) else (⊤ : EReal)
      else
        (toReal x : EReal) := by
  unfold toEReal IEEE32Exec.toEReal?
  by_cases hnan : isNaN x <;> by_cases hinf : isInf x <;> simp [hnan, hinf]

/-!
## Small finiteness helpers

These are convenience lemmas for switching between `isFinite` and special-value predicates in
  proofs.
-/

/-- If `x` is finite, then `x` is not an infinity. -/
theorem isInf_eq_false_of_isFinite_eq_true (x : IEEE32Exec) (hx : isFinite x = true) :
    isInf x = false := by
  have hne : (expField x != expAllOnes) = true := by simpa [isFinite] using hx
  -- If `expField x == expAllOnes` were true, then `expField x != expAllOnes` would be false.
  have hexp : (expField x == expAllOnes) = false := by
    cases hEq : (expField x == expAllOnes) <;> simp [bne, hEq] at hne
    exact rfl
  simp [isInf, hexp]

/-- If `x` is finite, then `x` is not a NaN. -/
theorem isNaN_eq_false_of_isFinite_eq_true (x : IEEE32Exec) (hx : isFinite x = true) :
    isNaN x = false := by
  have hne : (expField x != expAllOnes) = true := by simpa [isFinite] using hx
  have hexp : (expField x == expAllOnes) = false := by
    cases hEq : (expField x == expAllOnes) <;> simp [bne, hEq] at hne
    exact rfl
  simp [isNaN, hexp]

/--
On finite values, `toEReal` reduces to `↑(toReal x)`.

This is the lemma we use most often to turn an `EReal` endpoint goal into a `ℝ` goal.
-/
theorem toEReal_eq_coe_toReal_of_isFinite (x : IEEE32Exec) (hx : isFinite x = true) :
    toEReal x = (toReal x : EReal) := by
  have hnan : isNaN x = false := isNaN_eq_false_of_isFinite_eq_true (x := x) hx
  have hinf : isInf x = false := isInf_eq_false_of_isFinite_eq_true (x := x) hx
  simp [toEReal_eq_ite, hnan, hinf]

/-!
## Signed-zero simp facts

IEEE-754 has distinct bit patterns for `+0` and `-0`, but both decode to real `0`. In endpoint
  proofs
we frequently want this fact at the `toEReal` level.
-/

/--
Both signed zeros map to `0` under `toEReal`.

This is safe to use as a simp lemma: it is a very specific pattern and does not cause unfolding
loops.
-/
@[simp] theorem toEReal_signedZero (s : Bool) :
    toEReal (if s then negZero else posZero) = (0 : EReal) := by
  have hfin : isFinite (if s then negZero else posZero) = true := by
    cases s <;> decide
  have hz : toReal (if s then negZero else posZero) = 0 := toReal_signedZero s
  calc
    toEReal (if s then negZero else posZero) =
        (toReal (if s then negZero else posZero) : EReal) :=
      toEReal_eq_coe_toReal_of_isFinite (x := if s then negZero else posZero) hfin
    _ = (0 : EReal) := by
      rw [hz]
      simp

/-- `toEReal` interprets `+∞` as `⊤`. -/
@[simp] theorem toEReal_posInf : toEReal (posInf : IEEE32Exec) = (⊤ : EReal) := by
  have hnan : isNaN (posInf : IEEE32Exec) = false := by decide
  have hinf : isInf (posInf : IEEE32Exec) = true := by decide
  have hsign : signBit (posInf : IEEE32Exec) = false := by decide
  simp [toEReal_eq_ite, hnan, hinf, hsign]

/-- `toEReal` interprets `-∞` as `⊥`. -/
@[simp] theorem toEReal_negInf : toEReal (negInf : IEEE32Exec) = (⊥ : EReal) := by
  have hnan : isNaN (negInf : IEEE32Exec) = false := by decide
  have hinf : isInf (negInf : IEEE32Exec) = true := by decide
  have hsign : signBit (negInf : IEEE32Exec) = true := by decide
  simp [toEReal_eq_ite, hnan, hinf, hsign]

end -- noncomputable section

end IEEE32Exec

end TorchLean.Floats.IEEE754

end -- public section
