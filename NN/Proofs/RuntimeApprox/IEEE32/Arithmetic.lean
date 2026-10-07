/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.Finite

/-!
# Finite binary32 refinement for proof consumers

These lemmas transfer FloatLib's real rounding theorems through the configured binary32 codec.
The executable operations remain FloatLib operations. The total refinement theorems assume only
a finite result and discharge exceptional inputs internally. In particular, division of a finite
value by infinity produces signed zero, agreeing with the totalized real interpretation.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite ofModel toModel toModel_ofModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Floats.IEEE754.IEEE32Exec

open FloatLib.Floats.Formats.BinaryInterchange

/-- A decoded dyadic witnesses that the IEEE value is finite. -/
theorem isFinite_eq_true_of_toDyadic?_some {x : ExecFloat.Binary 8 23} {d :
  FloatLib.Numerics.Dyadic}
    (h : (toModel x).toDyadic? = some d) : isFinite x = true :=
  Model.isFinite_eq_true_of_toDyadic?_some h

/-- Configured subtraction decodes to FloatLib's model subtraction. -/
@[simp] theorem toModel_sub (x y : ExecFloat.Binary 8 23) :
    toModel (ExecFloat.sub x y) = Model.sub (toModel x) (toModel y) :=
  IEEEExec.toModel_sub x y

/-- Configured division decodes to FloatLib's model division. -/
@[simp] theorem toModel_div (x y : ExecFloat.Binary 8 23) :
    toModel (ExecFloat.div x y) = Model.div (toModel x) (toModel y) := by
  rw [FloatLib.Floats.ExecFloat.Proof.div_eq_spec, Model.Proof.div_eq_spec]
  exact toModel_ofModel (Model.Spec.div (toModel x) (toModel y))

/-- Configured nearest-even FMA decodes to the model's single-rounding FMA. -/
@[simp] theorem toModel_fma (x y z : ExecFloat.Binary 8 23) :
    toModel ((ExecFloat.Binary.fmaWithRounding (rounding := .nearestEven)) x y z) =
      Model.fma (toModel x) (toModel y) (toModel z) :=
  FloatLib.Floats.ExecFloat.Binary.toModel_fmaWithRounding x y z .nearestEven

/-- Configured nearest-even square root decodes to the model square root. -/
@[simp] theorem toModel_sqrt (x : ExecFloat.Binary 8 23) :
    toModel ((ExecFloat.Binary.sqrtWithRounding (rounding := .nearestEven)) x) =
      Model.sqrt (toModel x) :=
  FloatLib.Floats.ExecFloat.Binary.toModel_sqrtWithRounding x .nearestEven

/-- The binary32 zero literal denotes real zero. -/
@[simp] theorem toReal_zero : (toModel (0 : ExecFloat.Binary 8 23)).toReal = 0 :=
  IEEEExec.toReal_zero

/-- IEEE minimum agrees with real minimum on finite operands, including signed zeros. -/
theorem toReal_minimum_eq_min_of_isFinite {x y : ExecFloat.Binary 8 23}
    (hx : isFinite x = true) (hy : isFinite y = true) :
    (toModel (min x y)).toReal = min ((toModel x).toReal) ((toModel y).toReal) := by
  rw [FloatLib.Floats.ExecFloat.Binary.toModel_min]
  exact Model.toReal_minimum_eq_min_of_isFinite (toModel x) (toModel y) hx hy

/-- IEEE maximum agrees with real maximum on finite operands, including signed zeros. -/
theorem toReal_maximum_eq_max_of_isFinite {x y : ExecFloat.Binary 8 23}
    (hx : isFinite x = true) (hy : isFinite y = true) :
    (toModel (max x y)).toReal = max ((toModel x).toReal) ((toModel y).toReal) :=
  IEEEExec.toReal_maximum_eq_max_of_isFinite hx hy

/-- Finite subtraction refines one nearest-even rounding of the real difference. -/
theorem toReal_sub_eq_round_of_isFinite {x y : ExecFloat.Binary 8 23}
    (hx : isFinite x = true) (hy : isFinite y = true)
    (hfin : isFinite (ExecFloat.sub x y) = true) :
    (toModel (ExecFloat.sub x y)).toReal = Model.roundAt FloatFormat.binary32
      ((toModel x).toReal - (toModel y).toReal) :=
  IEEEExec.toReal_sub_eq_round_of_isFinite (by decide) hx hy hfin

/-- Finite dyadic rounding refines binary32 rounding of the exact dyadic real value. -/
theorem toReal_roundDyadic_eq_round {d : FloatLib.Numerics.Dyadic}
    (hfin : isFinite ((ofModel (Model.roundDyadic FloatFormat.binary32 d) : ExecFloat.Binary 8 23))
      = true) :
    (toModel ((ofModel (Model.roundDyadic FloatFormat.binary32 d) : ExecFloat.Binary 8 23))).toReal
      = Model.roundAt FloatFormat.binary32 d.toReal :=
  IEEEExec.toReal_roundDyadic_eq_round (by decide) hfin

/-- Finite square-root evaluation excludes negative nonzero inputs and refines real rounding. -/
theorem toReal_sqrt_eq_round {x : ExecFloat.Binary 8 23} {dx : FloatLib.Numerics.Dyadic}
    (hx : (toModel x).toDyadic? = some dx)
    (hfin : isFinite ((ExecFloat.Binary.sqrtWithRounding (rounding := .nearestEven)) x) = true) :
    (toModel ((ExecFloat.Binary.sqrtWithRounding (rounding := .nearestEven)) x)).toReal =
      Model.roundAt FloatFormat.binary32 (Real.sqrt ((toModel x).toReal)) := by
  have hxfinite := Model.isFinite_eq_true_of_toDyadic?_some hx
  have hmodel : Model.isFinite (Model.sqrt (toModel x)) = true := by
    change Model.isFinite (toModel
      (ExecFloat.Binary.sqrtWithRounding x .nearestEven)) = true at hfin
    simpa only [toModel_sqrt] using hfin
  have hdomain : Model.isZero (toModel x) = true ∨ Model.signBit (toModel x) = false := by
    by_cases hzero : Model.isZero (toModel x) = true
    · exact Or.inl hzero
    · right
      cases hsign : Model.signBit (toModel x) with
      | false => rfl
      | true =>
          rw [Model.Proof.sqrt_eq_spec] at hmodel
          simp only [Model.Spec.sqrt, Model.chooseNaN1_none_of_isFinite (toModel x) hxfinite,
            Model.isInf_eq_false_of_isFinite_eq_true (toModel x) hxfinite, hzero, hsign,
            ↓reduceIte, Bool.false_eq_true] at hmodel
          have hbad : Model.isFinite (Model.invalidResult FloatFormat.binary32) = true := hmodel
          exact hbad.symm.trans model_isFinite_invalidResult
  rw [toModel_sqrt]
  exact Model.toReal_sqrt_eq_roundAt (fmt := FloatFormat.binary32)
    (toModel x) (by decide) hxfinite hdomain

private abbrev Binary32Model := Model FloatFormat.binary32

private theorem model_isFinite_chooseNaN1 (x n : Binary32Model)
    (h : Model.chooseNaN1 x = some n) : Model.isFinite n = false := by
  cases hnan : Model.isNaN x with
  | false => simp [Model.chooseNaN1, hnan] at h
  | true =>
      have hn : Model.quietNaN x = n := by simpa [Model.chooseNaN1, hnan] using h
      rw [← hn, model_isFinite_quietNaN]
      exact Model.isFinite_eq_false_of_isNaN hnan

private theorem model_isFinite_chooseNaN3 (x y z n : Binary32Model)
    (h : Model.chooseNaN3 x y z = some n) : Model.isFinite n = false := by
  have hxS := Model.isSNaN_eq_false_of_isNaN_eq_false x
  have hyS := Model.isSNaN_eq_false_of_isNaN_eq_false y
  have hzS := Model.isSNaN_eq_false_of_isNaN_eq_false z
  cases hxNaN : Model.isNaN x <;> cases hyNaN : Model.isNaN y <;>
    cases hzNaN : Model.isNaN z <;> cases hxSNaN : Model.isSNaN x <;>
    cases hySNaN : Model.isSNaN y <;> cases hzSNaN : Model.isSNaN z <;>
    simp_all [Model.chooseNaN3]
  all_goals
    subst n
    rw [model_isFinite_quietNaN]
    apply Model.isFinite_eq_false_of_isNaN
    assumption

private theorem model_finite_operands_of_fma (x y z : Binary32Model)
    (hfin : Model.isFinite (Model.fma x y z) = true) :
    Model.isFinite x = true ∧ Model.isFinite y = true ∧ Model.isFinite z = true := by
  rw [Model.Proof.fma_eq_spec] at hfin
  cases hchoose : Model.chooseNaN3 x y z with
  | some n =>
      have hn := model_isFinite_chooseNaN3 x y z n hchoose
      simp [Model.Spec.fma, hchoose, hn] at hfin
  | none =>
      obtain ⟨hxNaN, hyNaN, hzNaN⟩ := (Model.chooseNaN3_eq_none_iff x y z).mp hchoose
      have hxyInf : (Model.isInf x || Model.isInf y) = false := by
        cases hxy : Model.isInf x || Model.isInf y with
        | false => rfl
        | true =>
            simp only [Model.Spec.fma, hchoose, hxy, ↓reduceIte] at hfin
            split at hfin
            · simp only [model_isFinite_invalidResult, Bool.false_eq_true] at hfin
            · split at hfin
              · split at hfin
                · simp only [model_isFinite_invalidResult, Bool.false_eq_true] at hfin
                · simp only [model_isFinite_nativeOverflow, Bool.false_eq_true] at hfin
              · simp only [model_isFinite_nativeOverflow, Bool.false_eq_true] at hfin
      have hzInf : Model.isInf z = false := by
        cases hz : Model.isInf z with
        | false => rfl
        | true =>
            have hzfin : Model.isFinite z = true := by
              simpa only [Model.Spec.fma, hchoose, hxyInf, hz, Bool.false_eq_true,
                ↓reduceIte] using hfin
            have := Model.isInf_eq_false_of_isFinite_eq_true z hzfin
            simp [hz] at this
      have hparts : Model.isInf x = false ∧ Model.isInf y = false := by
        simpa only [Bool.or_eq_false_iff] using hxyInf
      exact ⟨Model.isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false x hxNaN hparts.1,
        Model.isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false y hyNaN hparts.2,
        Model.isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false z hzNaN hzInf⟩

/-- Fused multiply-add refinement packaged for total reasoning. -/
theorem toReal_fma_eq_round_of_isFinite (x y z : ExecFloat.Binary 8 23)
    (hfin : isFinite ((ExecFloat.Binary.fmaWithRounding (rounding := .nearestEven)) x y z) = true) :
    (toModel ((ExecFloat.Binary.fmaWithRounding (rounding := .nearestEven)) x y z)).toReal =
      Model.roundAt FloatFormat.binary32
        ((toModel x).toReal * (toModel y).toReal + (toModel z).toReal) := by
  have hmodel :
      Model.isFinite (Model.fma (toModel x) (toModel y) (toModel z)) = true := by
    change Model.isFinite
      (toModel ((ExecFloat.Binary.fmaWithRounding (rounding := .nearestEven)) x y z)) = true at hfin
    simpa only [toModel_fma] using hfin
  obtain ⟨hx, hy, hz⟩ := model_finite_operands_of_fma (toModel x) (toModel y) (toModel z) hmodel
  rw [toModel_fma]
  exact Model.toReal_fma_eq_roundAt (fmt := FloatFormat.binary32)
    (toModel x) (toModel y) (toModel z) (by decide) hx hy hz hmodel

/-- Square-root refinement packaged for total reasoning. -/
theorem toReal_sqrt_eq_round_of_isFinite (x : ExecFloat.Binary 8 23)
    (hfin : isFinite ((ExecFloat.Binary.sqrtWithRounding (rounding := .nearestEven)) x) = true) :
    (toModel ((ExecFloat.Binary.sqrtWithRounding (rounding := .nearestEven)) x)).toReal =
      Model.roundAt FloatFormat.binary32 (Real.sqrt ((toModel x).toReal)) := by
  have hmodel : Model.isFinite (Model.sqrt (toModel x)) = true := by
    change Model.isFinite (toModel
      (ExecFloat.Binary.sqrtWithRounding x .nearestEven)) = true at hfin
    simpa only [toModel_sqrt] using hfin
  rw [Model.Proof.sqrt_eq_spec] at hmodel
  cases hchoose : Model.chooseNaN1 (toModel x) with
  | some n =>
      have hn := model_isFinite_chooseNaN1 (toModel x) n hchoose
      simp only [Model.Spec.sqrt, hchoose] at hmodel
      have hnfinite : Model.isFinite (fmt := FloatFormat.binary32) n = true := hmodel
      have hcontra : false = true := hn.symm.trans hnfinite
      cases hcontra
  | none =>
      have hxInf : Model.isInf (toModel x) = false := by
        cases hi : Model.isInf (toModel x) with
        | false => rfl
        | true =>
            simp only [Model.Spec.sqrt, hchoose, hi, ↓reduceIte] at hmodel
            split at hmodel
            · have hbad : Model.isFinite (Model.invalidResult FloatFormat.binary32) = true :=
                hmodel
              exact hbad.symm.trans model_isFinite_invalidResult
            · have hbad :
                  Model.isFinite (Model.nativeOverflow FloatFormat.binary32 false) = true := hmodel
              exact hbad.symm.trans (model_isFinite_nativeOverflow false)
      have hxfinite := Model.isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false
        (toModel x) ((Model.chooseNaN1_eq_none_iff (toModel x)).mp hchoose) hxInf
      obtain ⟨dx, hx⟩ := Model.exists_toDyadic?_of_isFinite hxfinite
      exact toReal_sqrt_eq_round (dx := dx) hx hfin

private theorem model_div_eq_roundAt_of_isFinite (x y : Binary32Model)
    (hfin : Model.isFinite (Model.div x y) = true) :
    Model.toReal (Model.div x y) =
      Model.roundAt FloatFormat.binary32 (Model.toReal x / Model.toReal y) := by
  cases hx : Model.toDyadic? x <;> cases hy : Model.toDyadic? y
  case some.some dx dy =>
    have hy0 : dy.significand ≠ 0 := by
      intro hzero
      rw [Model.Proof.div_eq_spec] at hfin
      simp only [Model.Spec.div, hx, hy, hzero, beq_self_eq_true, ↓reduceIte] at hfin
      split at hfin
      · simp only [model_isFinite_invalidResult, Bool.false_eq_true] at hfin
      · simp only [model_isFinite_nativeOverflow, Bool.false_eq_true] at hfin
    have hyNonzero : Model.isZero y = false := by
      rw [Model.isZero_eq_beq_zero_of_toDyadic?_some hy]
      exact beq_eq_false_iff_ne.mpr hy0
    exact Model.toReal_div_eq_roundAt x y (by decide)
      (Model.isFinite_eq_true_of_toDyadic?_some hx)
      (Model.isFinite_eq_true_of_toDyadic?_some hy) hyNonzero hfin
  all_goals
    have hdiv : Model.div x y = Model.Spec.divSpecial x y := by
      simp only [Model.Proof.div_eq_spec, Model.Spec.div, hx, hy]
    rw [hdiv] at hfin ⊢
    cases hchoose : Model.chooseNaN2 x y with
    | some n =>
        have hn := model_isFinite_chooseNaN2 x y n hchoose
        simp [Model.Spec.divSpecial, hchoose, hn] at hfin
    | none =>
        have hxInf : Model.isInf x = false := by
          cases hi : Model.isInf x with
          | false => rfl
          | true =>
              simp only [Model.Spec.divSpecial, hchoose, hi, ↓reduceIte] at hfin
              split at hfin
              · simp only [model_isFinite_invalidResult, Bool.false_eq_true] at hfin
              · simp only [model_isFinite_nativeOverflow, Bool.false_eq_true] at hfin
        cases hyInf : Model.isInf y with
        | false =>
            simp only [Model.Spec.divSpecial, hchoose, hxInf, hyInf,
              model_isFinite_invalidResult, Bool.false_eq_true, ↓reduceIte] at hfin
        | true =>
            have hyDecode : Model.toDyadic? y = none := by
              cases hd : Model.toDyadic? y with
              | none => rfl
              | some d =>
                  have := Model.isInf_eq_false_of_toDyadic?_some hd
                  simp [hyInf] at this
            have hyReal : Model.toReal y = 0 := by
              simp only [Model.toReal_eq, hyDecode]
            simp only [Model.Spec.divSpecial, hchoose, hxInf, hyInf, Bool.false_eq_true,
              ↓reduceIte]
            rw [Model.toReal_zero, hyReal, div_zero, Model.roundAt_zero]

/-- Division refinement packaged for total reasoning. -/
theorem toReal_div_eq_round_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hfin : isFinite (ExecFloat.div x y) = true) :
    (toModel (ExecFloat.div x y)).toReal = Model.roundAt FloatFormat.binary32
      ((toModel x).toReal / (toModel y).toReal) := by
  have hmodel : Model.isFinite (Model.div (toModel x) (toModel y)) = true :=
    (congrArg Model.isFinite (toModel_div x y)).symm.trans hfin
  rw [toModel_div]
  exact model_div_eq_roundAt_of_isFinite (toModel x) (toModel y) hmodel

end TorchLean.Floats.IEEE754.IEEE32Exec
