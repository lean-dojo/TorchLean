/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.EReal.Basic
public import NN.Floats.IEEEExec.BridgeFP32Total

/-!
# BridgeERealTotal

Extended-real (`EReal`) semantics for `IEEE32Exec`.

We use `toReal?` as the main “finite-only” semantic function in TorchLean. For some statements,
though, we really want to distinguish `+∞` from `-∞` instead of collapsing both to `none`.

This file packages a slightly richer view:

- `toEReal? x = none` exactly for NaN (unordered),
- `toEReal? x = some ⊤` / `some ⊥` for `+∞` / `-∞`,
- `toEReal? x = some (↑(toReal x))` for finite values.

We then prove “golden theorem”-style lemmas for core ops, where the finite branch refines to the
`FP32` rounding-on-`ℝ` model (`fp32Round`) and the non-finite branch preserves IEEE-754 special
behavior.

Background:
- IEEE 754-2019: https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg (1991): https://doi.org/10.1145/103162.103163
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-- Extended-real interpretation of `IEEE32Exec` (`none` exactly for NaN). -/
noncomputable def toEReal? (x : IEEE32Exec) : Option EReal :=
  if isNaN x then
    none
  else if isInf x then
    some (if signBit x then (⊥ : EReal) else (⊤ : EReal))
  else
    some (toReal x : EReal)

/-- Quick bridge: `toEReal?` is `none` exactly on NaN. -/
theorem toEReal?_eq_none_iff_isNaN_eq_true (x : IEEE32Exec) :
    toEReal? x = none ↔ isNaN x = true := by
  cases hnan : isNaN x <;> cases hinf : isInf x <;> simp [toEReal?, hnan, hinf]

/-- Quick bridge: NaNs stay `none`. -/
theorem toEReal?_eq_none_of_isNaN_eq_true (x : IEEE32Exec) (hx : isNaN x = true) :
    toEReal? x = none := by
  simp [toEReal?, hx]

/-- Quick bridge: `isInf` rules out NaN. -/
theorem isNaN_eq_false_of_isInf_eq_true (x : IEEE32Exec) (hx : isInf x = true) :
    isNaN x = false := by
  -- `isInf` implies `fracField x == 0`, hence `fracField x != 0` is false.
  have hx' : (expField x == expAllOnes && fracField x == 0) = true := by
    simpa [isInf] using hx
  have hfracEq : (fracField x == 0) = true := by
    have : (expField x == expAllOnes) = true ∧ (fracField x == 0) = true := by
      simpa [Bool.and_eq_true] using hx'
    exact this.2
  have hne : fracField x = 0 := (beq_iff_eq).1 hfracEq
  have hfracNe : (fracField x != 0) = false := by simp [hne]
  -- With `fracField x != 0 = false`, `isNaN` is false regardless of the exponent bit.
  simp [isNaN, hfracNe]

/-- Quick bridge: finite means neither NaN nor Inf. -/
theorem isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x : IEEE32Exec)
    (hnan : isNaN x = false) (hinf : isInf x = false) :
    isFinite x = true := by
  cases hexp : (expField x == expAllOnes) with
  | false =>
      -- Exponent is not all-ones, i.e. finite.
      have hne : expField x ≠ expAllOnes := (beq_eq_false_iff_ne).1 hexp
      have hbne : (expField x != expAllOnes) = true := (bne_iff_ne).2 hne
      simpa [isFinite] using hbne
  | true =>
      -- If exp is all-ones, then either `isInf` or `isNaN` must hold, contradicting assumptions.
      cases hfrac : (fracField x == 0) with
      | true =>
          have : isInf x = true := by simp [isInf, hexp, hfrac]
          have : False := by
            simp [hinf]  at this
          exact this.elim
      | false =>
          have hne : fracField x ≠ 0 := (beq_eq_false_iff_ne).1 hfrac
          have hbne : (fracField x != 0) = true := (bne_iff_ne).2 hne
          have : isNaN x = true := by simp [isNaN, hexp, hbne]
          have : False := by
            simp [hnan]  at this
          exact this.elim

/-- Quick bridge: finite values coerce to `EReal` via `toReal`. -/
theorem toEReal?_eq_some_toReal_of_isFinite_eq_true (x : IEEE32Exec) (hx : isFinite x = true) :
    toEReal? x = some (toReal x : EReal) := by
  have hne : expField x ≠ expAllOnes := (bne_iff_ne).1 (by simpa [isFinite] using hx)
  have hexpFalse : (expField x == expAllOnes) = false := (beq_eq_false_iff_ne).2 hne
  have hnan : isNaN x = false := by simp [isNaN, hexpFalse]
  have hinf : isInf x = false := by simp [isInf, hexpFalse]
  simp [toEReal?, hnan, hinf]

/-- Quick bridge: `add` becomes `fp32Round` in `EReal`. -/
theorem toEReal?_add_eq_ite (x y : IEEE32Exec) :
    toEReal? (add x y) =
      if isNaN (add x y) then
        none
      else if isInf (add x y) then
        some (if signBit (add x y) then (⊥ : EReal) else (⊤ : EReal))
      else
        some (fp32Round (toReal x + toReal y) : EReal) := by
  cases hnan : isNaN (add x y) with
  | true =>
      simp [toEReal?, hnan]
  | false =>
      cases hinf : isInf (add x y) with
      | true =>
          simp [toEReal?, hnan, hinf]
      | false =>
          have hfin : isFinite (add x y) = true :=
            isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := add x y) hnan hinf
          have hreal : toReal (add x y) = fp32Round (toReal x + toReal y) :=
            toReal_add_eq_fp32Round_of_isFinite (x := x) (y := y) hfin
          have hrealE :
              (toReal (add x y) : EReal) = (fp32Round (toReal x + toReal y) : EReal) :=
            congrArg (fun r : ℝ => (r : EReal)) hreal
          unfold toEReal?
          rw [hnan, hinf, hrealE]

/-- Quick bridge: `mul` becomes `fp32Round` in `EReal`. -/
theorem toEReal?_mul_eq_ite (x y : IEEE32Exec) :
    toEReal? (mul x y) =
      if isNaN (mul x y) then
        none
      else if isInf (mul x y) then
        some (if signBit (mul x y) then (⊥ : EReal) else (⊤ : EReal))
      else
        some (fp32Round (toReal x * toReal y) : EReal) := by
  cases hnan : isNaN (mul x y) with
  | true =>
      simp [toEReal?, hnan]
  | false =>
      cases hinf : isInf (mul x y) with
      | true =>
          simp [toEReal?, hnan, hinf]
      | false =>
          have hfin : isFinite (mul x y) = true :=
            isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (x := mul x y) hnan hinf
          have hreal : toReal (mul x y) = fp32Round (toReal x * toReal y) :=
            toReal_mul_eq_fp32Round_of_isFinite (x := x) (y := y) hfin
          have hrealE :
              (toReal (mul x y) : EReal) = (fp32Round (toReal x * toReal y) : EReal) :=
            congrArg (fun r : ℝ => (r : EReal)) hreal
          unfold toEReal?
          rw [hnan, hinf, hrealE]

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
