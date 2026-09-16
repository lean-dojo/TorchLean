/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.IEEE32.Contracts

/-!
# Total optional observations of binary32 minimum and maximum

Finite operands use the real minimum and maximum contracts. NaNs propagate without a real
observation, while positive infinity is neutral for minimum and negative infinity is neutral for
maximum. The exceptional cases follow FloatLib's NaN selection and infinity comparison directly.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite isInfinite isNaN signBit toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Floats.IEEE754.IEEE32Exec

open FloatLib.Floats.Formats.BinaryInterchange

private theorem eq_of_model_eq {x y : ExecFloat.Binary 8 23}
    (h : toModel x = toModel y) : x = y :=
  ExecFloat.Binary.toModel_inj.mp h

private theorem nan_observation (x : ExecFloat.Binary 8 23) (hx : isNaN x = true) :
    toReal? x = none ∧ isInfinite x = false := by
  constructor
  · apply toReal?_eq_none_of_isFinite_eq_false
    change Model.isFinite (toModel x) = false
    exact Model.isFinite_eq_false_of_isNaN (x := toModel x) hx
  · cases hi : isInfinite x with
    | false => rfl
    | true =>
        have hn : isNaN x = false := Model.isNaN_eq_false_of_isInf_eq_true (toModel x) hi
        simp [hx] at hn

private theorem infinity_of_nonfinite (x : ExecFloat.Binary 8 23)
    (hx : isFinite x = false) (hnan : isNaN x = false) : isInfinite x = true := by
  cases hi : isInfinite x with
  | true => rfl
  | false =>
      have hf : isFinite x = true :=
        Model.isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false (toModel x) hnan hi
      simp [hx] at hf

private theorem minmax_none_of_nan (x y : ExecFloat.Binary 8 23)
    (hnan : isNaN x = true ∨ isNaN y = true) :
    toReal? (min x y) = none ∧ toReal? (max x y) = none := by
  cases hc : Model.chooseNaN2 (toModel x) (toModel y) with
  | none =>
      have hn := (Model.chooseNaN2_eq_none_iff (toModel x) (toModel y)).1 hc
      rcases hnan with hx | hy
      · have hxFalse : isNaN x = false := hn.1
        simp [hx] at hxFalse
      · have hyFalse : isNaN y = false := hn.2
        simp [hy] at hyFalse
  | some n =>
      have hf := model_isFinite_chooseNaN2 (toModel x) (toModel y) n hc
      constructor
      · apply toReal?_eq_none_of_isFinite_eq_false
        change Model.isFinite (FloatLib.Floats.ExecFloat.Binary.toModel (min x y)) = false
        rw [FloatLib.Floats.ExecFloat.Binary.toModel_min]
        rw [Model.minimum, Model.withNaNSelection_of_some _ _ n hc]
        exact hf
      · apply toReal?_eq_none_of_isFinite_eq_false
        change Model.isFinite (FloatLib.Floats.ExecFloat.Binary.toModel (max x y)) = false
        rw [FloatLib.Floats.ExecFloat.Binary.toModel_max]
        rw [Model.maximum, Model.withNaNSelection_of_some _ _ n hc]
        exact hf

private theorem minmax_infinity_right (x y : ExecFloat.Binary 8 23)
    (hx : isFinite x = true) (hy : isInfinite y = true) :
    min x y = (if signBit y then y else x) ∧
      max x y = (if signBit y then x else y) := by
  have hxNaN := Model.isNaN_eq_false_of_isFinite_eq_true (toModel x) hx
  have hyNaN := Model.isNaN_eq_false_of_isInf_eq_true (toModel y) hy
  have hxInf := Model.isInf_eq_false_of_isFinite_eq_true (toModel x) hx
  have hyInf : Model.isInf (toModel y) = true := hy
  have hc := Model.chooseNaN2_none_of_not_isNaN (toModel x) (toModel y) hxNaN hyNaN
  constructor
  · apply eq_of_model_eq
    rw [FloatLib.Floats.ExecFloat.Binary.toModel_min]
    rw [apply_ite toModel]
    change Model.minimum (toModel x) (toModel y) =
      if Model.signBit (toModel y) then toModel y else toModel x
    simp only [Model.minimum, Model.withNaNSelection_of_none _ _ hc]
    cases hs : Model.signBit (toModel y) <;>
      simp [Model.compareNonNaN, hxInf, hyInf, hs]
  · apply eq_of_model_eq
    rw [FloatLib.Floats.ExecFloat.Binary.toModel_max]
    rw [apply_ite toModel]
    change Model.maximum (toModel x) (toModel y) =
      if Model.signBit (toModel y) then toModel x else toModel y
    simp only [Model.maximum, Model.withNaNSelection_of_none _ _ hc]
    cases hs : Model.signBit (toModel y) <;>
      simp [Model.compareNonNaN, hxInf, hyInf, hs]

private theorem minmax_infinity_left (x y : ExecFloat.Binary 8 23)
    (hx : isInfinite x = true) (hy : isFinite y = true) :
    min x y = (if signBit x then x else y) ∧
      max x y = (if signBit x then y else x) := by
  have hxNaN := Model.isNaN_eq_false_of_isInf_eq_true (toModel x) hx
  have hyNaN := Model.isNaN_eq_false_of_isFinite_eq_true (toModel y) hy
  have hxInf : Model.isInf (toModel x) = true := hx
  have hyInf := Model.isInf_eq_false_of_isFinite_eq_true (toModel y) hy
  have hc := Model.chooseNaN2_none_of_not_isNaN (toModel x) (toModel y) hxNaN hyNaN
  constructor
  · apply eq_of_model_eq
    rw [FloatLib.Floats.ExecFloat.Binary.toModel_min]
    rw [apply_ite toModel]
    change Model.minimum (toModel x) (toModel y) =
      if Model.signBit (toModel x) then toModel x else toModel y
    simp only [Model.minimum, Model.withNaNSelection_of_none _ _ hc]
    cases hs : Model.signBit (toModel x) <;>
      simp [Model.compareNonNaN, hxInf, hyInf, hs]
  · apply eq_of_model_eq
    rw [FloatLib.Floats.ExecFloat.Binary.toModel_max]
    rw [apply_ite toModel]
    change Model.maximum (toModel x) (toModel y) =
      if Model.signBit (toModel x) then toModel y else toModel x
    simp only [Model.maximum, Model.withNaNSelection_of_none _ _ hc]
    cases hs : Model.signBit (toModel x) <;>
      simp [Model.compareNonNaN, hxInf, hyInf, hs]

private theorem minmax_none_of_nonfinite (x y : ExecFloat.Binary 8 23)
    (hx : isFinite x = false) (hy : isFinite y = false)
    (hxNaN : isNaN x = false) (hyNaN : isNaN y = false) :
    toReal? (min x y) = none ∧ toReal? (max x y) = none := by
  have hxInf := infinity_of_nonfinite x hx hxNaN
  have hxZero := Model.isZero_eq_false_of_isInf_eq_true (toModel x) hxInf
  have hxFinite : Model.isFinite (toModel x) = false := hx
  have hyFinite : Model.isFinite (toModel y) = false := hy
  have hc := Model.chooseNaN2_none_of_not_isNaN (toModel x) (toModel y) hxNaN hyNaN
  constructor
  · apply toReal?_eq_none_of_isFinite_eq_false
    change Model.isFinite (FloatLib.Floats.ExecFloat.Binary.toModel (min x y)) = false
    rw [FloatLib.Floats.ExecFloat.Binary.toModel_min]
    simp only [Model.minimum, Model.withNaNSelection_of_none _ _ hc]
    cases Model.compareNonNaN (toModel x) (toModel y) hxNaN hyNaN <;>
      simp [hxZero, hxFinite, hyFinite]
  · apply toReal?_eq_none_of_isFinite_eq_false
    change Model.isFinite (FloatLib.Floats.ExecFloat.Binary.toModel (max x y)) = false
    rw [FloatLib.Floats.ExecFloat.Binary.toModel_max]
    simp only [Model.maximum, Model.withNaNSelection_of_none _ _ hc]
    cases Model.compareNonNaN (toModel x) (toModel y) hxNaN hyNaN <;>
      simp [hxZero, hxFinite, hyFinite]

/--
Total characterization of `toReal? (minimum x y)` via `toReal? x` and `toReal? y`.

This lemma covers the cases where one side is `+∞` (which acts as a neutral element for `min`) and
the cases where `toReal?` is `none` because of NaN.
-/
theorem toReal?_minimum_eq_match_total (x y : ExecFloat.Binary 8 23) :
    toReal? (min x y) =
      match toReal? x, toReal? y with
      | some rx, some ry => some (min rx ry)
      | some rx, none => if isInfinite y && (!signBit y) then some rx else none
      | none, some ry => if isInfinite x && (!signBit x) then some ry else none
      | none, none => none := by
  by_cases hxNaN : isNaN x = true
  · obtain ⟨hxNone, hxInf⟩ := nan_observation x hxNaN
    rw [(minmax_none_of_nan x y (Or.inl hxNaN)).1, hxNone]
    cases toReal? y <;> simp [hxInf]
  have hxNaN : isNaN x = false := Bool.eq_false_of_not_eq_true hxNaN
  by_cases hyNaN : isNaN y = true
  · obtain ⟨hyNone, hyInf⟩ := nan_observation y hyNaN
    rw [(minmax_none_of_nan x y (Or.inr hyNaN)).1, hyNone]
    cases toReal? x <;> simp [hyInf]
  have hyNaN : isNaN y = false := Bool.eq_false_of_not_eq_true hyNaN
  cases hx : isFinite x <;> cases hy : isFinite y
  · rw [(minmax_none_of_nonfinite x y hx hy hxNaN hyNaN).1]
    simp [toReal?_eq_ite, hx, hy]
  · have hxInf := infinity_of_nonfinite x hx hxNaN
    rw [(minmax_infinity_left x y hxInf hy).1]
    cases signBit x <;> simp [toReal?_eq_ite, hx, hy, hxInf]
  · have hyInf := infinity_of_nonfinite y hy hyNaN
    rw [(minmax_infinity_right x y hx hyInf).1]
    cases signBit y <;> simp [toReal?_eq_ite, hx, hy, hyInf]
  · simpa only [toReal?_eq_ite, hx, hy, ↓reduceIte] using
      toReal?_minimum_eq_min_of_isFinite x y hx hy

/--
Total characterization of `toReal? (maximum x y)` via `toReal? x` and `toReal? y`.

This lemma covers the cases where one side is `-∞` (which acts as a neutral element for `max`) and
the cases where `toReal?` is `none` because of NaN.
-/
theorem toReal?_maximum_eq_match_total (x y : ExecFloat.Binary 8 23) :
    toReal? (max x y) =
      match toReal? x, toReal? y with
      | some rx, some ry => some (max rx ry)
      | some rx, none => if isInfinite y && (signBit y) then some rx else none
      | none, some ry => if isInfinite x && (signBit x) then some ry else none
      | none, none => none := by
  by_cases hxNaN : isNaN x = true
  · obtain ⟨hxNone, hxInf⟩ := nan_observation x hxNaN
    rw [(minmax_none_of_nan x y (Or.inl hxNaN)).2, hxNone]
    cases toReal? y <;> simp [hxInf]
  have hxNaN : isNaN x = false := Bool.eq_false_of_not_eq_true hxNaN
  by_cases hyNaN : isNaN y = true
  · obtain ⟨hyNone, hyInf⟩ := nan_observation y hyNaN
    rw [(minmax_none_of_nan x y (Or.inr hyNaN)).2, hyNone]
    cases toReal? x <;> simp [hyInf]
  have hyNaN : isNaN y = false := Bool.eq_false_of_not_eq_true hyNaN
  cases hx : isFinite x <;> cases hy : isFinite y
  · rw [(minmax_none_of_nonfinite x y hx hy hxNaN hyNaN).2]
    simp [toReal?_eq_ite, hx, hy]
  · have hxInf := infinity_of_nonfinite x hx hxNaN
    rw [(minmax_infinity_left x y hxInf hy).2]
    cases signBit x <;> simp [toReal?_eq_ite, hx, hy, hxInf]
  · have hyInf := infinity_of_nonfinite y hy hyNaN
    rw [(minmax_infinity_right x y hx hyInf).2]
    cases signBit y <;> simp [toReal?_eq_ite, hx, hy, hyInf]
  · simpa only [toReal?_eq_ite, hx, hy, ↓reduceIte] using
      toReal?_maximum_eq_max_of_isFinite x y hx hy

end TorchLean.Floats.IEEE754.IEEE32Exec
