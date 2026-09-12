/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32.Ops
public import NN.Floats.IEEEExec.Bridge.FP32.Sqrt
public import NN.Floats.IEEEExec.Bridge.FP32Total.Core
public import NN.Floats.IEEEExec.Rules.SpecialRules

/-!
# Total FP32 Bridge: Arithmetic

This module lifts the finite `FP32` refinement results for addition, subtraction, multiplication,
division, and square root to the total `IEEE32Exec` domain. A theorem either recovers the rounded
real equation from a finite executable result or leaves the NaN/infinity branch visible through
`toReal?`.

See `FP32Total.Core` for the finite/special-value split and references.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-! ## Arithmetic Refinement -/

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
Finite executable addition also agrees with the canonical mantissa/exponent produced by the
effective nearest-even calculation layer.
-/
theorem toReal_add_eq_computed_of_isFinite (x y : IEEE32Exec)
    (hfin : isFinite (add x y) = true) :
    toReal (add x y) =
      neuralToReal (β := binaryRadix) {
        mantissa := neuralNearestEvenMantissa
          (neuralScaledMantissa binaryRadix fexp32 (toReal x + toReal y))
        exponent := neuralCexp binaryRadix fexp32 (toReal x + toReal y) } := by
  rw [toReal_add_eq_fp32Round_of_isFinite x y hfin]
  change neuralRound (β := binaryRadix) (fexp := fexp32) rnd32
    (toReal x + toReal y) = _
  simpa [rnd32] using
    (neuralRound_nearestEven_computed
      (β := binaryRadix) (fexp := fexp32) (toReal x + toReal y))

/--
Subtraction refinement packaged for total reasoning (hide dyadic witnesses).

This is the finite-path wrapper around `toReal_sub_eq_fp32Round`, replacing explicit
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
                _ = fp32Round 0 := by simp
                _ = fp32Round (toReal x / toReal y) := by
                  rw [hyReal]
                  simp
          | false =>
              cases hy0 : isZero y with
              | true =>
                  -- `x / 0` is NaN (if x=0) or ±Inf.
                  have hcn : isFinite canonicalNaN = false := by decide
                  by_cases hx0 : isZero x = true
                  · have : isFinite (div x y) = false := by
                      simp [div, hy0, hx0, hcn]
                    have : False := by
                      simp [hfin]  at this
                    exact this.elim
                  · have hxNaN : isNaN x = false :=
                      (not_isNaN_of_chooseNaN2_none x y hchoose).1
                    obtain ⟨dx, hdx⟩ := exists_toDyadic?_of_not_isNaN_not_isInf hxNaN hxInf
                    have hdxMant : dx.mant ≠ 0 := by
                      intro hmant
                      exact hx0 (isZero_eq_true_of_toDyadic?_some_of_mant_eq_zero hdx hmant)
                    have : isFinite (div x y) = false := by
                      simp only [div, hdx, toDyadic?_eq_zero_of_isZero hy0]
                      simp [hdxMant]
                      split <;> decide
                    have : False := by
                      simp [hfin] at this
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
                            hx, hy]
                          have hcn : isFinite canonicalNaN = false := by decide
                          have : isFinite (div x y) = false := by simp [hdiv, hcn]
                          have : False := by
                            simp [hfin]  at this
                          exact this.elim
                  | none =>
                      have hdiv : div x y = canonicalNaN := by simp [div, hchoose, hxInf, hyInf,
                        hx]
                      have hcn : isFinite canonicalNaN = false := by decide
                      have : isFinite (div x y) = false := by simp [hdiv, hcn]
                      have : False := by
                        simp [hfin]  at this
                      exact this.elim

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
