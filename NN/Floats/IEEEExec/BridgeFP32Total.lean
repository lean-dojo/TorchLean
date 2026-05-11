/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeFP32
public import NN.Floats.IEEEExec.SpecialRules

/-!
# BridgeFP32Total

“Total” bridge theorems combining:

- `IEEE32Exec`'s proved NaN/Inf propagation rules, and
- the `FP32`-on-`ℝ` refinement theorems for the finite/no-overflow branch (`BridgeFP32.lean`).

The key end-user view is `toReal?`:
- `toReal? x = none` for NaN/Inf,
- `toReal? x = some r` for finite values, with `r : ℝ`.

In most of TorchLean, the finite path is treated as real arithmetic + float32 rounding while
special-value behavior is kept explicit. This file packages that split in one place.

The per-op lemmas are phrased in the style:

`toReal? (op …) = if isFinite (op …) then some (fp32Round …) else none`.

That makes the trust boundary readable at the call site: the `if` is exactly where NaN/Inf (or
overflow-to-Inf) can occur.

Background references (for float32 rounding/special values):
- IEEE 754-2019: https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg (1991): https://doi.org/10.1145/103162.103163
- Flocq (Boldo–Melquiond, 2011): https://doi.org/10.1109/ARITH.2011.40
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-! ## Basic facts: `isFinite` ↔ `toDyadic?`/`toReal?` -/

/-- `toDyadic? x = none` implies `x` is not finite. -/
theorem isFinite_eq_false_of_toDyadic?_eq_none (x : IEEE32Exec) (hx : toDyadic? x = none) :
    isFinite x = false := by
  unfold toDyadic? at hx
  cases hcond : (isNaN x || isInf x) with
  | false =>
      -- In the non-special branch, `toDyadic?` always returns `some …`.
      cases hE : (expField x == 0) <;> cases hF : (fracField x == 0) <;>
        simp [hcond, hE, hF] at hx
  | true =>
      cases hnan : isNaN x with
      | true =>
          have hexp : expField x = expAllOnes := expField_eq_expAllOnes_of_isNaN (x := x) hnan
          exact isFinite_eq_false_of_expField_eq_expAllOnes (x := x) hexp
      | false =>
          -- `isNaN x = false` and `isNaN x || isInf x = true` implies `isInf x = true`.
          cases hinf : isInf x with
          | true =>
              have hexp : expField x = expAllOnes := expField_eq_expAllOnes_of_isInf (x := x) hinf
              exact isFinite_eq_false_of_expField_eq_expAllOnes (x := x) hexp
          | false =>
              -- Contradiction: the disjunction cannot be `true`.
              have hcondFalse : (isNaN x || isInf x) = false := by simp [hnan, hinf]
              have hcontra : False := by
                have : (false : Bool) = true := by
                  simp [hcondFalse]  at hcond
                cases this
              exact hcontra.elim

/-- If `x` is not finite, then `toDyadic? x = none`. -/
theorem toDyadic?_eq_none_of_isFinite_eq_false (x : IEEE32Exec) (hx : isFinite x = false) :
    toDyadic? x = none := by
  -- By cases on the fraction field: expField=all-ones gives either Inf or NaN.
  unfold toDyadic?
  unfold isFinite at hx
  have hx' : (expField x != expAllOnes) = false := by simpa using hx
  have hexp : expField x = expAllOnes := by
    by_contra hne
    have htrue : (expField x != expAllOnes) = true := (bne_iff_ne).2 hne
    have : False := by
      simp [htrue] at hx'
    exact this.elim
  have hexpB : (expField x == expAllOnes) = true := (beq_iff_eq).2 hexp
  by_cases hfrac : fracField x = 0
  · have hfracB : (fracField x == 0) = true := (beq_iff_eq).2 hfrac
    have hinf : isInf x = true := by simp [isInf, hexpB, hfracB]
    simp [hinf]
  · have hfracNeB : (fracField x != 0) = true := (bne_iff_ne).2 hfrac
    have hnan : isNaN x = true := by simp [isNaN, hexpB, hfracNeB]
    simp [hnan]

/-- `toReal?` returns `none` on non-finite values. -/
theorem toReal?_eq_none_of_isFinite_eq_false (x : IEEE32Exec) (hx : isFinite x = false) :
    toReal? x = none := by
  have hdy : toDyadic? x = none := toDyadic?_eq_none_of_isFinite_eq_false (x := x) hx
  simp [toReal?, hdy]

/-- On finite values, `toReal? x` is just `some (toReal x)`. -/
theorem toReal?_eq_some_toReal_of_isFinite_eq_true (x : IEEE32Exec) (hx : isFinite x = true) :
    toReal? x = some (toReal x) := by
  -- `toDyadic? x` cannot be `none`, otherwise `isFinite x = false`.
  cases hdy : toDyadic? x with
  | none =>
      have hxFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdy
      have : False := by
        simp [hx]  at hxFalse
      exact this.elim
  | some d =>
      simp [toReal?, toReal, hdy]

/-! ## Helpers: NaN/Inf/zero interactions -/

/-- NaNs are not infinities. -/
theorem isInf_eq_false_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    isInf x = false := by
  -- `isNaN` means `fracField x != 0`, hence `fracField x == 0` is false.
  have hnan : (expField x == expAllOnes && fracField x != 0) = true := by
    simpa [isNaN] using hx
  have hfracNe : (fracField x != 0) = true := by
    have : (expField x == expAllOnes) = true ∧ (fracField x != 0) = true := by
      simpa [Bool.and_eq_true] using hnan
    exact this.2
  have hne : fracField x ≠ 0 := (bne_iff_ne).1 hfracNe
  have hfracEqFalse : (fracField x == 0) = false := (beq_eq_false_iff_ne).2 hne
  simp [isInf, hfracEqFalse]

/-- Infinities are not zeros. -/
theorem isZero_eq_false_of_isInf (x : IEEE32Exec) (hx : isInf x = true) :
    isZero x = false := by
  have hinf : (expField x == expAllOnes && fracField x == 0) = true := by
    simpa [isInf] using hx
  have hexp : expField x = expAllOnes := by
    have : (expField x == expAllOnes) = true ∧ (fracField x == 0) = true := by
      simpa [Bool.and_eq_true] using hinf
    exact (beq_iff_eq).1 this.1
  have hexpNe0 : expField x ≠ 0 := by
    intro h0
    have : expAllOnes = 0 := by simpa [hexp] using h0
    exact (by decide : (expAllOnes : UInt32) ≠ 0) this
  have hexp0 : (expField x == 0) = false := (beq_eq_false_iff_ne).2 hexpNe0
  simp [isZero, hexp0]

/-- If dyadic decoding fails and the value is not NaN, then it must be infinite. -/
theorem isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x : IEEE32Exec)
    (hx : toDyadic? x = none) (hxNaN : isNaN x = false) :
    isInf x = true := by
  -- `toDyadic? x = none` means we took the `isNaN x || isInf x` branch.
  unfold toDyadic? at hx
  cases hcond : (isNaN x || isInf x) with
  | true =>
      -- With `isNaN x = false`, the disjunction being true forces `isInf x = true`.
      cases hinf : isInf x with
      | true => rfl
      | false =>
          have : (isNaN x || isInf x) = false := by simp [hxNaN, hinf]
          have : False := by
            simp [this]  at hcond
          exact this.elim
  | false =>
      -- Contradiction: in the non-special branch `toDyadic?` always returns `some …`.
      cases hE : (expField x == 0) <;> cases hF : (fracField x == 0) <;>
        simp [hcond, hE, hF] at hx

/-! ## Per-op “finite branch” wrappers (hide dyadic witnesses) -/

/-- Addition refinement packaged for total reasoning. -/
theorem toReal_add_eq_fp32Round_of_isFinite (x y : IEEE32Exec)
    (hfin : isFinite (add x y) = true) :
    toReal (add x y) = fp32Round (toReal x + toReal y) := by
  classical
  cases hchoose : chooseNaN2 x y with
  | some nan =>
      have hnfin : isFinite nan = false :=
        isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
      have hadd : add x y = nan := add_eq_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
      have : isFinite (add x y) = false := by simp [hadd, hnfin]
      have : False := by
        simp [hfin]  at this
      exact this.elim
  | none =>
      cases hxInf : isInf x with
      | true =>
          -- Any `Inf` branch is non-finite.
          have hxExp : expField x = expAllOnes := expField_eq_expAllOnes_of_isInf (x := x) hxInf
          have hxFin : isFinite x = false := isFinite_eq_false_of_expField_eq_expAllOnes (x := x)
            hxExp
          cases hyInf : isInf y with
          | true =>
              cases hsign : (signBit x == signBit y) with
              | true =>
                  have hadd : add x y = x := by simp [add, hchoose, hxInf, hyInf, hsign]
                  have : isFinite (add x y) = false := by simp [hadd, hxFin]
                  have : False := by
                    simp [hfin]  at this
                  exact this.elim
              | false =>
                  have hadd : add x y = canonicalNaN := by simp [add, hchoose, hxInf, hyInf, hsign]
                  have hcn : isFinite canonicalNaN = false := by decide
                  have : isFinite (add x y) = false := by simp [hadd, hcn]
                  have : False := by
                    simp [hfin]  at this
                  exact this.elim
          | false =>
              have hadd : add x y = x := by simp [add, hchoose, hxInf, hyInf]
              have : isFinite (add x y) = false := by simp [hadd, hxFin]
              have : False := by
                simp [hfin]  at this
              exact this.elim
      | false =>
          cases hyInf : isInf y with
          | true =>
              have hyExp : expField y = expAllOnes := expField_eq_expAllOnes_of_isInf (x := y) hyInf
              have hyFin : isFinite y = false := isFinite_eq_false_of_expField_eq_expAllOnes (x :=
                y) hyExp
              have hadd : add x y = y := by simp [add, hchoose, hxInf, hyInf]
              have : isFinite (add x y) = false := by simp [hadd, hyFin]
              have : False := by
                simp [hfin]  at this
              exact this.elim
          | false =>
              -- Finite core: must have dyadic decodes for both operands.
              cases hx : toDyadic? x with
              | some dx =>
                  cases hy : toDyadic? y with
                  | some dy =>
                      exact toReal_add_eq_fp32Round (x := x) (y := y) (dx := dx) (dy := dy) hx hy
                        hfin
                  | none =>
                      have hadd : add x y = canonicalNaN := by simp [add, hchoose, hxInf, hyInf, hx,
                        hy]
                      have hcn : isFinite canonicalNaN = false := by decide
                      have : isFinite (add x y) = false := by simp [hadd, hcn]
                      have : False := by
                        simp [hfin]  at this
                      exact this.elim
              | none =>
                  have hadd : add x y = canonicalNaN := by simp [add, hchoose, hxInf, hyInf, hx]
                  have hcn : isFinite canonicalNaN = false := by decide
                  have : isFinite (add x y) = false := by simp [hadd, hcn]
                  have : False := by
                    simp [hfin]  at this
                  exact this.elim

/--
Subtraction refinement packaged for total reasoning (hide dyadic witnesses).

This is the “finite-path” wrapper around `BridgeFP32.toReal_sub_eq_fp32Round`, replacing explicit
`toDyadic?` witnesses with the more user-facing finiteness hypotheses.
-/
theorem toReal_sub_eq_fp32Round_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true)
    (hfin : isFinite (sub x y) = true) :
    toReal (sub x y) = fp32Round (toReal x - toReal y) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have hxfalse : isFinite x = false :=
        isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        have hxfalse' := hxfalse
        rw [hx] at hxfalse'
        cases hxfalse'
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have hyfalse : isFinite y = false :=
            isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            have hyfalse' := hyfalse
            rw [hy] at hyfalse'
            cases hyfalse'
          exact this.elim
      | some dy =>
          exact
            IEEE32Exec.toReal_sub_eq_fp32Round (x := x) (y := y) (dx := dx) (dy := dy)
              hdx hdy hfin

/-- Multiplication refinement packaged for total reasoning. -/
theorem toReal_mul_eq_fp32Round_of_isFinite (x y : IEEE32Exec)
    (hfin : isFinite (mul x y) = true) :
    toReal (mul x y) = fp32Round (toReal x * toReal y) := by
  classical
  cases hchoose : chooseNaN2 x y with
  | some nan =>
      have hnfin : isFinite nan = false :=
        isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
      have hmul : mul x y = nan := mul_eq_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
      have : isFinite (mul x y) = false := by simp [hmul, hnfin]
      have : False := by
        simp [hfin]  at this
      exact this.elim
  | none =>
      cases hxInf : isInf x with
      | true =>
          have hcn : isFinite canonicalNaN = false := by decide
          have hni : isFinite negInf = false := by decide
          have hpi : isFinite posInf = false := by decide
          -- Either `Inf * 0 = NaN` or `Inf * finite = ±Inf`.
          by_cases hy0 : isZero y = true
          · have : isFinite (mul x y) = false := by
              simp [mul, hchoose, hxInf, hy0, hcn]
            have : False := by
              simp [hfin]  at this
            exact this.elim
          · cases hsign : (signBit x != signBit y) with
            | true =>
                have : isFinite (mul x y) = false := by
                  simp [mul, hchoose, hxInf, hy0, hsign, hni]
                have : False := by
                  simp [hfin]  at this
                exact this.elim
            | false =>
                have : isFinite (mul x y) = false := by
                  simp [mul, hchoose, hxInf, hy0, hsign, hpi]
                have : False := by
                  simp [hfin]  at this
                exact this.elim
      | false =>
          cases hyInf : isInf y with
          | true =>
              have hcn : isFinite canonicalNaN = false := by decide
              have hni : isFinite negInf = false := by decide
              have hpi : isFinite posInf = false := by decide
              by_cases hx0 : isZero x = true
              · have : isFinite (mul x y) = false := by
                  simp [mul, hchoose, hxInf, hyInf, hx0, hcn]
                have : False := by
                  simp [hfin]  at this
                exact this.elim
              · cases hsign : (signBit x != signBit y) with
                | true =>
                    have : isFinite (mul x y) = false := by
                      simp [mul, hchoose, hxInf, hyInf, hx0, hsign, hni]
                    have : False := by
                      simp [hfin]  at this
                    exact this.elim
                | false =>
                    have : isFinite (mul x y) = false := by
                      simp [mul, hchoose, hxInf, hyInf, hx0, hsign, hpi]
                    have : False := by
                      simp [hfin]  at this
                    exact this.elim
          | false =>
              cases hx : toDyadic? x with
              | some dx =>
                  cases hy : toDyadic? y with
                  | some dy =>
                      exact toReal_mul_eq_fp32Round (x := x) (y := y) (dx := dx) (dy := dy) hx hy
                        hfin
                  | none =>
                      have hmul : mul x y = canonicalNaN := by simp [mul, hchoose, hxInf, hyInf, hx,
                        hy]
                      have hcn : isFinite canonicalNaN = false := by decide
                      have : isFinite (mul x y) = false := by simp [hmul, hcn]
                      have : False := by
                        simp [hfin]  at this
                      exact this.elim
              | none =>
                  have hmul : mul x y = canonicalNaN := by simp [mul, hchoose, hxInf, hyInf, hx]
                  have hcn : isFinite canonicalNaN = false := by decide
                  have : isFinite (mul x y) = false := by simp [hmul, hcn]
                  have : False := by
                    simp [hfin]  at this
                  exact this.elim

/-- Fused multiply-add refinement packaged for total reasoning. -/
theorem toReal_fma_eq_fp32Round_of_isFinite (x y z : IEEE32Exec)
    (hfin : isFinite (fma x y z) = true) :
    toReal (fma x y z) = fp32Round (toReal x * toReal y + toReal z) := by
  classical
  cases hchoose : chooseNaN3 x y z with
  | some nan =>
      have hnfin : isFinite nan = false :=
        isFinite_eq_false_of_chooseNaN3_some (x := x) (y := y) (z := z) (nan := nan) hchoose
      have hfma : fma x y z = nan := fma_eq_of_chooseNaN3_some (x := x) (y := y) (z := z) (nan :=
        nan) hchoose
      have : isFinite (fma x y z) = false := by simp [hfma, hnfin]
      have : False := by
        simp [hfin]  at this
      exact this.elim
  | none =>
      -- Any Inf involvement forces a non-finite result; so we can reduce to the dyadic branch.
      cases hxInf : (isInf x || isInf y) with
      | true =>
          -- The `Inf*0` and `Inf±Inf` exceptional cases yield NaN/Inf.
          have hcn : isFinite canonicalNaN = false := by decide
          have hni : isFinite negInf = false := by decide
          have hpi : isFinite posInf = false := by decide
          -- Directly evaluate the `Inf` branch.
          have : isFinite (fma x y z) = false := by
            cases hzero : (isZero x || isZero y) with
            | true =>
                simp [fma, hchoose, hxInf, hzero, hcn]
            | false =>
                -- `prodInf` is ±Inf.
                cases hprodSign : Bool.xor (signBit x) (signBit y) with
                | true =>
                    -- `prodInf = negInf`
                    cases hzInf : isInf z with
                    | true =>
                        cases hbad : (signBit z != true) with
                        | true =>
                            simp [fma, hchoose, hxInf, hzero, hprodSign, hzInf, hbad, hcn]
                        | false =>
                            simp [fma, hchoose, hxInf, hzero, hprodSign, hzInf, hbad, hni]
                    | false =>
                        simp [fma, hchoose, hxInf, hzero, hprodSign, hzInf, hni]
                | false =>
                    -- `prodInf = posInf`
                    cases hzInf : isInf z with
                    | true =>
                        cases hbad : (signBit z != false) with
                        | true =>
                            simp [fma, hchoose, hxInf, hzero, hprodSign, hzInf, hbad, hcn]
                        | false =>
                            simp [fma, hchoose, hxInf, hzero, hprodSign, hzInf, hbad, hpi]
                    | false =>
                        simp [fma, hchoose, hxInf, hzero, hprodSign, hzInf, hpi]
          have : False := by
            simp [hfin]  at this
          exact this.elim
      | false =>
          cases hzInf : isInf z with
          | true =>
              have hzExp : expField z = expAllOnes := expField_eq_expAllOnes_of_isInf (x := z) hzInf
              have hzFin : isFinite z = false := isFinite_eq_false_of_expField_eq_expAllOnes (x :=
                z) hzExp
              have hfma : fma x y z = z := by simp [fma, hchoose, hxInf, hzInf]
              have : isFinite (fma x y z) = false := by simp [hfma, hzFin]
              have : False := by
                simp [hfin]  at this
              exact this.elim
          | false =>
              cases hx : toDyadic? x with
              | some dx =>
                  cases hy : toDyadic? y with
                  | some dy =>
                      cases hz : toDyadic? z with
                      | some dz =>
                          exact toReal_fma_eq_fp32Round (x := x) (y := y) (z := z) (dx := dx) (dy :=
                            dy) (dz := dz) hx hy hz hfin
                      | none =>
                          have hfma : fma x y z = canonicalNaN := by simp [fma, hchoose, hxInf,
                            hzInf, hx, hy, hz]
                          have hcn : isFinite canonicalNaN = false := by decide
                          have : isFinite (fma x y z) = false := by simp [hfma, hcn]
                          have : False := by
                            simp [hfin]  at this
                          exact this.elim
                  | none =>
                      have hfma : fma x y z = canonicalNaN := by simp [fma, hchoose, hxInf, hzInf,
                        hx, hy]
                      have hcn : isFinite canonicalNaN = false := by decide
                      have : isFinite (fma x y z) = false := by simp [hfma, hcn]
                      have : False := by
                        simp [hfin]  at this
                      exact this.elim
              | none =>
                  have hfma : fma x y z = canonicalNaN := by simp [fma, hchoose, hxInf, hzInf, hx]
                  have hcn : isFinite canonicalNaN = false := by decide
                  have : isFinite (fma x y z) = false := by simp [hfma, hcn]
                  have : False := by
                    simp [hfin]  at this
                  exact this.elim

/-- Square-root refinement packaged for total reasoning. -/
theorem toReal_sqrt_eq_fp32Round_of_isFinite (x : IEEE32Exec)
    (hfin : isFinite (sqrt x) = true) :
    toReal (sqrt x) = fp32Round (Real.sqrt (toReal x)) := by
  classical
  cases hchoose : chooseNaN1 x with
  | some nan =>
      have hnfin : isFinite nan = false :=
        isFinite_eq_false_of_chooseNaN1_some (x := x) (nan := nan) hchoose
      have hsqrt : sqrt x = nan := sqrt_eq_of_chooseNaN1_some (x := x) (nan := nan) hchoose
      have : isFinite (sqrt x) = false := by simp [hsqrt, hnfin]
      have : False := by
        simp [hfin]  at this
      exact this.elim
  | none =>
      cases hxInf : isInf x with
      | true =>
          have hcn : isFinite canonicalNaN = false := by decide
          have hpi : isFinite posInf = false := by decide
          have : isFinite (sqrt x) = false := by
            cases hs : signBit x <;> simp [sqrt, hchoose, hxInf, hs, hcn, hpi]
          have : False := by
            simp [hfin]  at this
          exact this.elim
      | false =>
          cases hx : toDyadic? x with
          | some dx =>
              exact toReal_sqrt_eq_fp32Round (x := x) (dx := dx) hx hfin
          | none =>
              -- Impossible: `chooseNaN1 x = none` gives `isNaN x = false`, and `hxInf : isInf x =
              -- false`.
              have hxNaN : isNaN x = false := by
                cases hnan : isNaN x with
                | true =>
                    have : (some (quietNaN x) : Option IEEE32Exec) = none := by
                      simp [chooseNaN1, hnan]  at hchoose
                    cases this
                | false =>
                    rfl
              have hcond : (isNaN x || isInf x) = false := by
                simp [hxNaN, hxInf]
              -- With the special-condition false, `toDyadic? x` reduces to a `some` branch.
              unfold toDyadic? at hx
              cases hE : (expField x == 0) <;> cases hF : (fracField x == 0) <;>
                simp [hcond, hE, hF] at hx

/-- Division refinement packaged for total reasoning. -/
theorem toReal_div_eq_fp32Round_of_isFinite (x y : IEEE32Exec)
    (hfin : isFinite (div x y) = true) :
    toReal (div x y) = fp32Round (toReal x / toReal y) := by
  classical
  cases hchoose : chooseNaN2 x y with
  | some nan =>
      have hnfin : isFinite nan = false :=
        isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
      have hdiv : div x y = nan := div_eq_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
      have : isFinite (div x y) = false := by simp [hdiv, hnfin]
      have : False := by
        simp [hfin]  at this
      exact this.elim
  | none =>
      cases hxInf : isInf x with
      | true =>
          -- `Inf / y` is either NaN (if `y` is Inf) or ±Inf.
          have hcn : isFinite canonicalNaN = false := by decide
          have hni : isFinite negInf = false := by decide
          have hpi : isFinite posInf = false := by decide
          by_cases hyInf : isInf y = true
          · have : isFinite (div x y) = false := by
              simp [div, hchoose, hxInf, hyInf, hcn]
            have : False := by
              simp [hfin]  at this
            exact this.elim
          · cases hsign : (signBit x != signBit y) with
            | true =>
                have : isFinite (div x y) = false := by
                  simp [div, hchoose, hxInf, hyInf, hsign, hni]
                have : False := by
                  simp [hfin]  at this
                exact this.elim
            | false =>
                have : isFinite (div x y) = false := by
                  simp [div, hchoose, hxInf, hyInf, hsign, hpi]
                have : False := by
                  simp [hfin]  at this
                exact this.elim
      | false =>
          cases hyInf : isInf y with
          | true =>
              -- `finite / ±Inf` is signed zero; the total `toReal` maps ±Inf to 0, so `x / toReal y
              -- = 0`.
              have hdiv : div x y = (if signBit x != signBit y then negZero else posZero) := by
                simp [div, hchoose, hxInf, hyInf]
              have hyDy : toDyadic? y = none := by
                -- Any `Inf` is a special value for `toDyadic?`.
                simp [toDyadic?, hyInf]
              have hyReal : toReal y = 0 := by simp [toReal_eq, hyDy]
              -- The result is a signed zero, whose real interpretation is 0.
              have hresReal : toReal (div x y) = 0 := by
                cases hs : (signBit x != signBit y) with
                | true =>
                    have hdiv' : div x y = negZero := by simpa [hs] using hdiv
                    simp [hdiv']
                | false =>
                    have hdiv' : div x y = posZero := by simpa [hs] using hdiv
                    simp [hdiv']
              calc
                toReal (div x y) = 0 := hresReal
                _ = fp32Round 0 := by simp [fp32Round_zero]
                _ = fp32Round (toReal x / toReal y) := by
                  rw [hyReal]
                  simp
          | false =>
              cases hy0 : isZero y with
              | true =>
                  -- `x / 0` is NaN (if x=0) or ±Inf.
                  have hcn : isFinite canonicalNaN = false := by decide
                  have hni : isFinite negInf = false := by decide
                  have hpi : isFinite posInf = false := by decide
                  by_cases hx0 : isZero x = true
                  · have : isFinite (div x y) = false := by
                      simp [div, hchoose, hxInf, hyInf, hy0, hx0, hcn]
                    have : False := by
                      simp [hfin]  at this
                    exact this.elim
                  · cases hsign : (signBit x != signBit y) with
                    | true =>
                        have : isFinite (div x y) = false := by
                          simp [div, hchoose, hxInf, hyInf, hy0, hx0, hsign, hni]
                        have : False := by
                          simp [hfin]  at this
                        exact this.elim
                    | false =>
                        have : isFinite (div x y) = false := by
                          simp [div, hchoose, hxInf, hyInf, hy0, hx0, hsign, hpi]
                        have : False := by
                          simp [hfin]  at this
                        exact this.elim
              | false =>
                  cases hx : toDyadic? x with
                  | some dx =>
                      cases hy : toDyadic? y with
                      | some dy =>
                          have hy0' : dy.mant ≠ 0 := by
                            intro hmant0
                            have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx :=
                              hy)
                            have hyInf' : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx :=
                              hy)
                            have hcond : (isNaN y || isInf y) = false := by simp [hyNaN, hyInf']
                            unfold toDyadic? at hy
                            cases hE : (expField y == 0) with
                            | true =>
                                cases hF : (fracField y == 0) with
                                | true =>
                                    have hy' :
                                        some { sign := signBit y, mant := 0, exp := 0 } = some dy :=
                                          by
                                      simpa [hcond, hE, hF] using hy
                                    have hdy : dy = { sign := signBit y, mant := 0, exp := 0 } :=
                                      (Option.some.inj hy').symm
                                    have hzTrue : isZero y = true := by simp [isZero, hE, hF]
                                    have hzFalse : isZero y = false := by simpa using hy0
                                    have : False := by
                                      simp [hzFalse]  at hzTrue
                                    exact this.elim
                                | false =>
                                    have hy' :
                                        some { sign := signBit y, mant := (fracField y).toNat, exp
                                          := -149 } =
                                          some dy := by
                                      simpa [hcond, hE, hF] using hy
                                    have hdy : dy =
                                        { sign := signBit y, mant := (fracField y).toNat, exp :=
                                          -149 } :=
                                      (Option.some.inj hy').symm
                                    have hne : fracField y ≠ 0 := (beq_eq_false_iff_ne).1 hF
                                    have : dy.mant ≠ 0 := by
                                      intro h0
                                      have : fracField y = 0 := by
                                        apply UInt32.toNat_inj.1
                                        simpa [hdy] using h0
                                      exact hne this
                                    exact this (by simpa [hdy] using hmant0)
                            | false =>
                                have hy' :
                                    some
                                        { sign := signBit y
                                          mant := pow2 23 + (fracField y).toNat
                                          exp := Int.ofNat (expField y).toNat - 150 } = some dy :=
                                            by
                                  simpa [hcond, hE] using hy
                                have hdy : dy =
                                    { sign := signBit y
                                      mant := pow2 23 + (fracField y).toNat
                                      exp := Int.ofNat (expField y).toNat - 150 } :=
                                  (Option.some.inj hy').symm
                                have hpow : (pow2 23 : Nat) ≠ 0 := by decide
                                have : dy.mant ≠ 0 := by
                                  intro h0
                                  have : pow2 23 = 0 := (Nat.add_eq_zero_iff.mp (by simpa [hdy]
                                    using h0)).1
                                  exact hpow this
                                exact this (by simpa [hdy] using hmant0)
                          exact
                            toReal_div_eq_fp32Round (x := x) (y := y) (dx := dx) (dy := dy) hx hy
                              hy0' hfin
                      | none =>
                          have hdiv : div x y = canonicalNaN := by simp [div, hchoose, hxInf, hyInf,
                            hy0, hx, hy]
                          have hcn : isFinite canonicalNaN = false := by decide
                          have : isFinite (div x y) = false := by simp [hdiv, hcn]
                          have : False := by
                            simp [hfin]  at this
                          exact this.elim
                  | none =>
                      have hdiv : div x y = canonicalNaN := by simp [div, hchoose, hxInf, hyInf,
                        hy0, hx]
                      have hcn : isFinite canonicalNaN = false := by decide
                      have : isFinite (div x y) = false := by simp [hdiv, hcn]
                      have : False := by
                        simp [hfin]  at this
                      exact this.elim

/-- On finite values, IEEE-754 `minimum` agrees with real `min`. -/
theorem toReal_minimum_eq_min_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal (minimum x y) = min (toReal x) (toReal y) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          exact toReal_minimum_eq_min (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, IEEE-754 `maximum` agrees with real `max`. -/
theorem toReal_maximum_eq_max_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal (maximum x y) = max (toReal x) (toReal y) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          exact toReal_maximum_eq_max (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, `compare x y = some .lt` iff `toReal x < toReal y`. -/
theorem compare_eq_some_lt_iff_toReal_lt_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    compare x y = some .lt ↔ toReal x < toReal y := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          exact compare_eq_some_lt_iff_toReal_lt (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, `compare x y = some .eq` iff `toReal x = toReal y`. -/
theorem compare_eq_some_eq_iff_toReal_eq_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    compare x y = some .eq ↔ toReal x = toReal y := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          exact compare_eq_some_eq_iff_toReal_eq (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, `compare x y = some .gt` iff `toReal y < toReal x`. -/
theorem compare_eq_some_gt_iff_toReal_gt_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    compare x y = some .gt ↔ toReal y < toReal x := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          exact compare_eq_some_gt_iff_toReal_gt (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-! ## “Both” view: `toReal?` semantics as an `ite` -/

/-- `toReal? (add x y)` as an `ite` over finiteness. -/
theorem toReal?_add_eq_ite (x y : IEEE32Exec) :
    toReal? (add x y) =
      if isFinite (add x y) then some (fp32Round (toReal x + toReal y)) else none := by
  cases hfin : isFinite (add x y) with
  | true =>
      have hto : toReal (add x y) = fp32Round (toReal x + toReal y) :=
        toReal_add_eq_fp32Round_of_isFinite (x := x) (y := y) (by simpa using hfin)
      have hSome : toReal? (add x y) = some (toReal (add x y)) :=
        toReal?_eq_some_toReal_of_isFinite_eq_true (x := add x y) (by simpa using hfin)
      rw [hSome]
      rw [hto]
      simp
  | false =>
      have hNone : toReal? (add x y) = none :=
        toReal?_eq_none_of_isFinite_eq_false (x := add x y) hfin
      rw [hNone]
      simp

/-- `toReal? (mul x y)` as an `ite` over finiteness. -/
theorem toReal?_mul_eq_ite (x y : IEEE32Exec) :
    toReal? (mul x y) =
      if isFinite (mul x y) then some (fp32Round (toReal x * toReal y)) else none := by
  cases hfin : isFinite (mul x y) with
  | true =>
      have hto : toReal (mul x y) = fp32Round (toReal x * toReal y) :=
        toReal_mul_eq_fp32Round_of_isFinite (x := x) (y := y) (by simpa using hfin)
      have hSome : toReal? (mul x y) = some (toReal (mul x y)) :=
        toReal?_eq_some_toReal_of_isFinite_eq_true (x := mul x y) (by simpa using hfin)
      rw [hSome]
      rw [hto]
      simp
  | false =>
      have hNone : toReal? (mul x y) = none :=
        toReal?_eq_none_of_isFinite_eq_false (x := mul x y) hfin
      rw [hNone]
      simp

/-- `toReal? (fma x y z)` as an `ite` over finiteness. -/
theorem toReal?_fma_eq_ite (x y z : IEEE32Exec) :
    toReal? (fma x y z) =
      if isFinite (fma x y z) then some (fp32Round (toReal x * toReal y + toReal z)) else none := by
  cases hfin : isFinite (fma x y z) with
  | true =>
      have hto : toReal (fma x y z) = fp32Round (toReal x * toReal y + toReal z) :=
        toReal_fma_eq_fp32Round_of_isFinite (x := x) (y := y) (z := z) (by simpa using hfin)
      have hSome : toReal? (fma x y z) = some (toReal (fma x y z)) :=
        toReal?_eq_some_toReal_of_isFinite_eq_true (x := fma x y z) (by simpa using hfin)
      rw [hSome]
      rw [hto]
      simp
  | false =>
      have hNone : toReal? (fma x y z) = none :=
        toReal?_eq_none_of_isFinite_eq_false (x := fma x y z) hfin
      rw [hNone]
      simp

/-- `toReal? (sqrt x)` as an `ite` over finiteness. -/
theorem toReal?_sqrt_eq_ite (x : IEEE32Exec) :
    toReal? (sqrt x) =
      if isFinite (sqrt x) then some (fp32Round (Real.sqrt (toReal x))) else none := by
  cases hfin : isFinite (sqrt x) with
  | true =>
      have hto : toReal (sqrt x) = fp32Round (Real.sqrt (toReal x)) :=
        toReal_sqrt_eq_fp32Round_of_isFinite (x := x) (by simpa using hfin)
      have hSome : toReal? (sqrt x) = some (toReal (sqrt x)) :=
        toReal?_eq_some_toReal_of_isFinite_eq_true (x := sqrt x) (by simpa using hfin)
      rw [hSome]
      rw [hto]
      simp
  | false =>
      have hNone : toReal? (sqrt x) = none :=
        toReal?_eq_none_of_isFinite_eq_false (x := sqrt x) hfin
      rw [hNone]
      simp

/-- `toReal? (div x y)` as an `ite` over finiteness. -/
theorem toReal?_div_eq_ite (x y : IEEE32Exec) :
    toReal? (div x y) =
      if isFinite (div x y) then some (fp32Round (toReal x / toReal y)) else none := by
  cases hfin : isFinite (div x y) with
  | true =>
      have hto : toReal (div x y) = fp32Round (toReal x / toReal y) :=
        toReal_div_eq_fp32Round_of_isFinite (x := x) (y := y) (by simpa using hfin)
      have hSome : toReal? (div x y) = some (toReal (div x y)) :=
        toReal?_eq_some_toReal_of_isFinite_eq_true (x := div x y) (by simpa using hfin)
      rw [hSome]
      rw [hto]
      simp
  | false =>
      have hNone : toReal? (div x y) = none :=
        toReal?_eq_none_of_isFinite_eq_false (x := div x y) hfin
      rw [hNone]
      simp

/--
`minimum` of two finite values is finite.
-/
theorem isFinite_minimum_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    isFinite (minimum x y) = true := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := by
            simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
          have hcmp : compare x y = some (cmpDyadic dx dy) :=
            compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hdx) (hy := hdy)
          cases hord : cmpDyadic dx dy with
          | lt =>
              have hcmp' : compare x y = some .lt := by simpa [hord] using hcmp
              have hmin : minimum x y = x := by simp [minimum, hchoose, hcmp']
              simpa [hmin] using hx
          | gt =>
              have hcmp' : compare x y = some .gt := by simpa [hord] using hcmp
              have hmin : minimum x y = y := by simp [minimum, hchoose, hcmp']
              simpa [hmin] using hy
          | eq =>
              have hcmp' : compare x y = some .eq := by simpa [hord] using hcmp
              cases hzeros : (isZero x && isZero y) with
              | true =>
                  have hmin : minimum x y = (if signBit x || signBit y then negZero else posZero) :=
                    by
                    simp [minimum, hchoose, hcmp', hzeros]
                  cases hs : (signBit x || signBit y) <;> simp [hmin, hs] <;> decide
              | false =>
                  have hmin : minimum x y = x := by simp [minimum, hchoose, hcmp', hzeros]
                  simpa [hmin] using hx

/--
On finite inputs, `toReal? (minimum x y)` returns `some (min (toReal x) (toReal y))`.
-/
theorem toReal?_minimum_eq_min_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal? (minimum x y) = some (min (toReal x) (toReal y)) := by
  have hfin : isFinite (minimum x y) = true := isFinite_minimum_of_isFinite (x := x) (y := y) hx hy
  have hSome : toReal? (minimum x y) = some (toReal (minimum x y)) :=
    toReal?_eq_some_toReal_of_isFinite_eq_true (x := minimum x y) hfin
  have hto : toReal (minimum x y) = min (toReal x) (toReal y) :=
    toReal_minimum_eq_min_of_isFinite (x := x) (y := y) hx hy
  rw [hSome, hto]

/--
`maximum` of two finite values is finite.
-/
theorem isFinite_maximum_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    isFinite (maximum x y) = true := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := by
            simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
          have hcmp : compare x y = some (cmpDyadic dx dy) :=
            compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hdx) (hy := hdy)
          cases hord : cmpDyadic dx dy with
          | lt =>
              have hcmp' : compare x y = some .lt := by simpa [hord] using hcmp
              have hmax : maximum x y = y := by simp [maximum, hchoose, hcmp']
              simpa [hmax] using hy
          | gt =>
              have hcmp' : compare x y = some .gt := by simpa [hord] using hcmp
              have hmax : maximum x y = x := by simp [maximum, hchoose, hcmp']
              simpa [hmax] using hx
          | eq =>
              have hcmp' : compare x y = some .eq := by simpa [hord] using hcmp
              cases hzeros : (isZero x && isZero y) with
              | true =>
                  have hmax : maximum x y =
                      (if (!signBit x) || (!signBit y) then posZero else negZero) := by
                    simp [maximum, hchoose, hcmp', hzeros]
                  cases hs : ((!signBit x) || (!signBit y)) <;> simp [hmax, hs] <;> decide
              | false =>
                  have hmax : maximum x y = x := by simp [maximum, hchoose, hcmp', hzeros]
                  simpa [hmax] using hx

/--
On finite inputs, `toReal? (maximum x y)` returns `some (max (toReal x) (toReal y))`.
-/
theorem toReal?_maximum_eq_max_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal? (maximum x y) = some (max (toReal x) (toReal y)) := by
  have hfin : isFinite (maximum x y) = true := isFinite_maximum_of_isFinite (x := x) (y := y) hx hy
  have hSome : toReal? (maximum x y) = some (toReal (maximum x y)) :=
    toReal?_eq_some_toReal_of_isFinite_eq_true (x := maximum x y) hfin
  have hto : toReal (maximum x y) = max (toReal x) (toReal y) :=
    toReal_maximum_eq_max_of_isFinite (x := x) (y := y) hx hy
  rw [hSome, hto]

/-! ## Total packaging for `minimum`/`maximum` (covers ±Inf) -/

/--
Total characterization of `toReal? (minimum x y)` via `toReal? x` and `toReal? y`.

This lemma covers the cases where one side is `+∞` (which acts as a neutral element for `min`) and
the cases where `toReal?` is `none` because of NaN.
-/
theorem toReal?_minimum_eq_match_total (x y : IEEE32Exec) :
    toReal? (minimum x y) =
      match toReal? x, toReal? y with
      | some rx, some ry => some (min rx ry)
      | some rx, none => if isInf y && (!signBit y) then some rx else none
      | none, some ry => if isInf x && (!signBit x) then some ry else none
      | none, none => none := by
  classical
  cases hx : toDyadic? x with
  | some dx =>
      cases hy : toDyadic? y with
      | some dy =>
          -- Both finite.
          have hxFin : isFinite x = true := by
            cases hfx : isFinite x with
            | true => rfl
            | false =>
                have hnone : toDyadic? x = none := toDyadic?_eq_none_of_isFinite_eq_false (x := x)
                  hfx
                have : (none : Option Dyadic) = some dx := by
                  simp [hnone]  at hx
                cases this
          have hyFin : isFinite y = true := by
            cases hfy : isFinite y with
            | true => rfl
            | false =>
                have hnone : toDyadic? y = none := toDyadic?_eq_none_of_isFinite_eq_false (x := y)
                  hfy
                have : (none : Option Dyadic) = some dy := by
                  simp [hnone]  at hy
                cases this
          have hmin : toReal? (minimum x y) = some (min (toReal x) (toReal y)) :=
            toReal?_minimum_eq_min_of_isFinite (x := x) (y := y) hxFin hyFin
          simpa [IEEE32Exec.toReal?, IEEE32Exec.toReal_eq, hx, hy] using hmin
      | none =>
          -- `x` finite, `y` special.
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
          cases hyNaN : isNaN y with
          | true =>
              -- NaN propagates.
              have hxS : isSNaN x = false := by simp [isSNaN, hxNaN]
              have hchoose : chooseNaN2 x y = some (quietNaN y) := by
                simp [chooseNaN2, hxS, hxNaN, hyNaN]
              have hminEq : minimum x y = quietNaN y := by simp [minimum, hchoose]
              have hfinFalse : isFinite (quietNaN y) = false :=
                isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := quietNaN y) hchoose
              have hnone : toReal? (quietNaN y) = none :=
                toReal?_eq_none_of_isFinite_eq_false (x := quietNaN y) hfinFalse
              have hyInfFalse : isInf y = false := isInf_eq_false_of_isNaN (x := y) hyNaN
              simpa [hminEq, hnone, IEEE32Exec.toReal?, hx, hy, hyInfFalse, hyNaN]
          | false =>
              -- Inf case.
              have hyInf : isInf y = true :=
                isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := y) hy hyNaN
              have hchoose : chooseNaN2 x y = none :=
                chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
              cases hsy : signBit y with
                | true =>
                    have hcmp : compare x y = some .gt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsy]
                    have hminEq : minimum x y = y := by simp [minimum, hchoose, hcmp]
                    simp [hminEq, IEEE32Exec.toReal?, hx, hy, hyInf]
                | false =>
                    have hcmp : compare x y = some .lt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsy]
                    have hminEq : minimum x y = x := by simp [minimum, hchoose, hcmp]
                    simp [hminEq, IEEE32Exec.toReal?, hx, hy, hyInf]
  | none =>
      cases hy : toDyadic? y with
      | some dy =>
          -- `x` special, `y` finite.
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
          cases hxNaN : isNaN x with
          | true =>
              -- NaN propagates.
              have hyS : isSNaN y = false := by simp [isSNaN, hyNaN]
              -- `chooseNaN2 x y` always selects a NaN when `x` is NaN.
              cases hchoose : chooseNaN2 x y with
              | some nan =>
                  have hminEq : minimum x y = nan := minimum_eq_of_chooseNaN2_some (x := x) (y := y)
                    (nan := nan) hchoose
                  have hfinFalse : isFinite nan = false :=
                    isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                  have hnone : toReal? nan = none :=
                    toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                  have hxInfFalse : isInf x = false := isInf_eq_false_of_isNaN (x := x) hxNaN
                  simpa [hminEq, hnone, IEEE32Exec.toReal?, hx, hy, hxInfFalse, hxNaN]
              | none =>
                  -- Impossible: `x` is NaN, so `chooseNaN2` cannot be `none`.
                  have : False := by
                    cases hxS : isSNaN x <;>
                      simp [chooseNaN2, hxS, hxNaN, hyS] at hchoose
                  exact this.elim
          | false =>
              -- Inf case.
              have hxInf : isInf x = true :=
                isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := x) hx hxNaN
              have hchoose : chooseNaN2 x y = none :=
                chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
              cases hsx : signBit x with
                | true =>
                    -- `x = -Inf`, so `minimum x y = x`.
                    have hcmp : compare x y = some .lt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx]
                    have hminEq : minimum x y = x := by simp [minimum, hchoose, hcmp]
                    simp [hminEq, IEEE32Exec.toReal?, hx, hy, hxInf]
                | false =>
                    -- `x = +Inf`, so `minimum x y = y`.
                    have hcmp : compare x y = some .gt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx]
                    have hminEq : minimum x y = y := by simp [minimum, hchoose, hcmp]
                    simp [hminEq, IEEE32Exec.toReal?, hx, hy, hxInf]
      | none =>
          -- Both special.
          cases hxNaN : isNaN x with
          | true =>
              -- NaN propagates.
              cases hchoose : chooseNaN2 x y with
              | some nan =>
                  have hminEq : minimum x y = nan := minimum_eq_of_chooseNaN2_some (x := x) (y := y)
                    (nan := nan) hchoose
                  have hfinFalse : isFinite nan = false :=
                    isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                  have hnone : toReal? nan = none :=
                    toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                  simpa [hminEq, hnone, IEEE32Exec.toReal?, hx, hy]
              | none =>
                  -- Impossible: `x` is NaN.
                  have : False := by
                    cases hxS : isSNaN x <;> cases hyS : isSNaN y <;>
                      simp [chooseNaN2, hxS, hyS, hxNaN] at hchoose
                  exact this.elim
          | false =>
              cases hyNaN : isNaN y with
              | true =>
                  -- NaN propagates.
                  cases hchoose : chooseNaN2 x y with
                  | some nan =>
                      have hminEq : minimum x y = nan := minimum_eq_of_chooseNaN2_some (x := x) (y
                        := y) (nan := nan) hchoose
                      have hfinFalse : isFinite nan = false :=
                        isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                      have hnone : toReal? nan = none :=
                        toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                      simpa [hminEq, hnone, IEEE32Exec.toReal?, hx, hy]
                  | none =>
                      -- Impossible: `y` is NaN.
                      have : False := by
                        cases hxS : isSNaN x <;> cases hyS : isSNaN y <;>
                          simp [chooseNaN2, hxS, hyS, hyNaN, hxNaN] at hchoose
                      exact this.elim
              | false =>
                  -- Both are Infs; `minimum` returns an Inf.
                  have hxInf : isInf x = true :=
                    isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := x) hx hxNaN
                  have hyInf : isInf y = true :=
                    isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := y) hy hyNaN
                  have hchoose : chooseNaN2 x y = none :=
                    chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
                  have hx0 : isZero x = false := isZero_eq_false_of_isInf (x := x) hxInf
                  have hy0 : isZero y = false := isZero_eq_false_of_isInf (x := y) hyInf
                  cases hs : (signBit x == signBit y) with
                  | true =>
                      have hcmp : compare x y = some .eq := by
                        simp [compare, hxNaN, hyNaN, hxInf, hyInf, hs]
                      have hminEq : minimum x y = x := by
                        simp [minimum, hchoose, hcmp, hx0, hy0]
                      simp [hminEq, IEEE32Exec.toReal?, hx, hy]
                  | false =>
                      cases hsx : signBit x with
                      | true =>
                          -- Then `signBit y = false` (since `==` is false).
                          cases hsy : signBit y with
                          | true =>
                              have : (signBit x == signBit y) = true := by simp [hsx, hsy]
                              have : False := by
                                simp [this]  at hs
                              exact this.elim
                          | false =>
                              have hcmp : compare x y = some .lt := by
                                simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx, hsy]
                              have hminEq : minimum x y = x := by simp [minimum, hchoose, hcmp]
                              simp [hminEq, IEEE32Exec.toReal?, hx, hy]
                      | false =>
                          -- Then `signBit y = true` (since `==` is false).
                          cases hsy : signBit y with
                          | false =>
                              have : (signBit x == signBit y) = true := by simp [hsx, hsy]
                              have : False := by
                                simp [this]  at hs
                              exact this.elim
                          | true =>
                              have hcmp : compare x y = some .gt := by
                                simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx, hsy]
                              have hminEq : minimum x y = y := by simp [minimum, hchoose, hcmp]
                              simp [hminEq, IEEE32Exec.toReal?, hx, hy]

/--
Total characterization of `toReal? (maximum x y)` via `toReal? x` and `toReal? y`.

This lemma covers the cases where one side is `-∞` (which acts as a neutral element for `max`) and
the cases where `toReal?` is `none` because of NaN.
-/
theorem toReal?_maximum_eq_match_total (x y : IEEE32Exec) :
    toReal? (maximum x y) =
      match toReal? x, toReal? y with
      | some rx, some ry => some (max rx ry)
      | some rx, none => if isInf y && (signBit y) then some rx else none
      | none, some ry => if isInf x && (signBit x) then some ry else none
      | none, none => none := by
  classical
  cases hx : toDyadic? x with
  | some dx =>
      cases hy : toDyadic? y with
      | some dy =>
          -- Both finite.
          have hxFin : isFinite x = true := by
            cases hfx : isFinite x with
            | true => rfl
            | false =>
                have hnone : toDyadic? x = none := toDyadic?_eq_none_of_isFinite_eq_false (x := x)
                  hfx
                have : (none : Option Dyadic) = some dx := by
                  simp [hnone]  at hx
                cases this
          have hyFin : isFinite y = true := by
            cases hfy : isFinite y with
            | true => rfl
            | false =>
                have hnone : toDyadic? y = none := toDyadic?_eq_none_of_isFinite_eq_false (x := y)
                  hfy
                have : (none : Option Dyadic) = some dy := by
                  simp [hnone]  at hy
                cases this
          have hmax : toReal? (maximum x y) = some (max (toReal x) (toReal y)) :=
            toReal?_maximum_eq_max_of_isFinite (x := x) (y := y) hxFin hyFin
          simpa [IEEE32Exec.toReal?, IEEE32Exec.toReal_eq, hx, hy] using hmax
      | none =>
          -- `x` finite, `y` special.
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
          cases hyNaN : isNaN y with
          | true =>
              -- NaN propagates.
              have hxS : isSNaN x = false := by simp [isSNaN, hxNaN]
              -- `chooseNaN2` selects a NaN; result is non-finite.
              cases hchoose : chooseNaN2 x y with
              | some nan =>
                  have hmaxEq : maximum x y = nan := maximum_eq_of_chooseNaN2_some (x := x) (y := y)
                    (nan := nan) hchoose
                  have hfinFalse : isFinite nan = false :=
                    isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                  have hnone : toReal? nan = none :=
                    toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                  have hyInfFalse : isInf y = false := isInf_eq_false_of_isNaN (x := y) hyNaN
                  rw [hmaxEq, hnone]
                  simp [IEEE32Exec.toReal?, hx, hy, hyInfFalse]
              | none =>
                  have : False := by
                    cases hyS : isSNaN y <;> simp [chooseNaN2, hxS, hxNaN, hyNaN, hyS] at hchoose
                  exact this.elim
          | false =>
              -- Inf case.
              have hyInf : isInf y = true :=
                isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := y) hy hyNaN
              have hchoose : chooseNaN2 x y = none :=
                chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
              cases hsy : signBit y with
                | true =>
                    -- `y = -Inf`, so `maximum x y = x`.
                    have hcmp : compare x y = some .gt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsy]
                    have hmaxEq : maximum x y = x := by simp [maximum, hchoose, hcmp]
                    simp [hmaxEq, IEEE32Exec.toReal?, hx, hy, hyInf]
                | false =>
                    -- `y = +Inf`, so `maximum x y = y`.
                    have hcmp : compare x y = some .lt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsy]
                    have hmaxEq : maximum x y = y := by simp [maximum, hchoose, hcmp]
                    simp [hmaxEq, IEEE32Exec.toReal?, hx, hy, hyInf]
  | none =>
      cases hy : toDyadic? y with
      | some dy =>
          -- `x` special, `y` finite.
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
          cases hxNaN : isNaN x with
          | true =>
              -- NaN propagates.
              have hyS : isSNaN y = false := by simp [isSNaN, hyNaN]
              cases hchoose : chooseNaN2 x y with
              | some nan =>
                  have hmaxEq : maximum x y = nan := maximum_eq_of_chooseNaN2_some (x := x) (y := y)
                    (nan := nan) hchoose
                  have hfinFalse : isFinite nan = false :=
                    isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                  have hnone : toReal? nan = none :=
                    toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                  have hxInfFalse : isInf x = false := isInf_eq_false_of_isNaN (x := x) hxNaN
                  rw [hmaxEq, hnone]
                  simp [IEEE32Exec.toReal?, hx, hy, hxInfFalse]
              | none =>
                  have : False := by
                    cases hxS : isSNaN x <;> simp [chooseNaN2, hxS, hxNaN, hyS] at hchoose
                  exact this.elim
          | false =>
              -- Inf case.
              have hxInf : isInf x = true :=
                isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := x) hx hxNaN
              have hchoose : chooseNaN2 x y = none :=
                chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
              cases hsx : signBit x with
                | true =>
                    -- `x = -Inf`, so `maximum x y = y`.
                    have hcmp : compare x y = some .lt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx]
                    have hmaxEq : maximum x y = y := by simp [maximum, hchoose, hcmp]
                    simp [hmaxEq, IEEE32Exec.toReal?, hx, hy, hxInf]
                | false =>
                    -- `x = +Inf`, so `maximum x y = x`.
                    have hcmp : compare x y = some .gt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx]
                    have hmaxEq : maximum x y = x := by simp [maximum, hchoose, hcmp]
                    simp [hmaxEq, IEEE32Exec.toReal?, hx, hy, hxInf]
      | none =>
          -- Both special.
          cases hxNaN : isNaN x with
          | true =>
              cases hchoose : chooseNaN2 x y with
              | some nan =>
                  have hmaxEq : maximum x y = nan := maximum_eq_of_chooseNaN2_some (x := x) (y := y)
                    (nan := nan) hchoose
                  have hfinFalse : isFinite nan = false :=
                    isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                  have hnone : toReal? nan = none :=
                    toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                  simpa [hmaxEq, hnone, IEEE32Exec.toReal?, hx, hy]
              | none =>
                  have : False := by
                    cases hxS : isSNaN x <;> cases hyS : isSNaN y <;>
                      simp [chooseNaN2, hxS, hyS, hxNaN] at hchoose
                  exact this.elim
          | false =>
              cases hyNaN : isNaN y with
              | true =>
                  cases hchoose : chooseNaN2 x y with
                  | some nan =>
                      have hmaxEq : maximum x y = nan := maximum_eq_of_chooseNaN2_some (x := x) (y
                        := y) (nan := nan) hchoose
                      have hfinFalse : isFinite nan = false :=
                        isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                      have hnone : toReal? nan = none :=
                        toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                      simpa [hmaxEq, hnone, IEEE32Exec.toReal?, hx, hy]
                  | none =>
                      have : False := by
                        cases hxS : isSNaN x <;> cases hyS : isSNaN y <;>
                          simp [chooseNaN2, hxS, hyS, hyNaN, hxNaN] at hchoose
                      exact this.elim
              | false =>
                  -- Both Infs.
                  have hxInf : isInf x = true :=
                    isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := x) hx hxNaN
                  have hyInf : isInf y = true :=
                    isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := y) hy hyNaN
                  have hchoose : chooseNaN2 x y = none :=
                    chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
                  have hx0 : isZero x = false := isZero_eq_false_of_isInf (x := x) hxInf
                  have hy0 : isZero y = false := isZero_eq_false_of_isInf (x := y) hyInf
                  cases hs : (signBit x == signBit y) with
                  | true =>
                      have hcmp : compare x y = some .eq := by
                        simp [compare, hxNaN, hyNaN, hxInf, hyInf, hs]
                      have hmaxEq : maximum x y = x := by
                        simp [maximum, hchoose, hcmp, hx0, hy0]
                      simp [hmaxEq, IEEE32Exec.toReal?, hx, hy]
                  | false =>
                      cases hsx : signBit x with
                      | true =>
                          cases hsy : signBit y with
                          | true =>
                              have : (signBit x == signBit y) = true := by simp [hsx, hsy]
                              have : False := by
                                simp [this]  at hs
                              exact this.elim
                          | false =>
                              have hcmp : compare x y = some .lt := by
                                simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx, hsy]
                              have hmaxEq : maximum x y = y := by simp [maximum, hchoose, hcmp]
                              simp [hmaxEq, IEEE32Exec.toReal?, hx, hy]
                      | false =>
                          cases hsy : signBit y with
                          | false =>
                              have : (signBit x == signBit y) = true := by simp [hsx, hsy]
                              have : False := by
                                simp [this]  at hs
                              exact this.elim
                          | true =>
                              have hcmp : compare x y = some .gt := by
                                simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx, hsy]
                              have hmaxEq : maximum x y = x := by simp [maximum, hchoose, hcmp]
                              simp [hmaxEq, IEEE32Exec.toReal?, hx, hy]

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
