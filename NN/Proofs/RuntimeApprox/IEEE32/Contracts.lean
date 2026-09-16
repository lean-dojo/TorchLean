/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.IEEE32.Arithmetic
public import NN.Floats.FP32.Error

/-!
# Binary32 observation and error contracts

These interfaces connect TorchLean's binary32 observations to FloatLib's real semantics.
Optional observations distinguish finite values from infinities and NaNs. ULP queries return
the exponent of the spacing at a finite value, and each arithmetic error bound follows from
FloatLib's rounding theorem with the original finiteness hypotheses.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Floats.IEEE754.IEEE32Exec

open TorchLean.Floats
open FloatLib.Numerics FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.Formats.BinaryInterchange

/-- Real observation of a finite word; infinities and NaNs have no real observation. -/
noncomputable abbrev toReal? (x : ExecFloat.Binary 8 23) : Option ℝ :=
  Model.toReal? (toModel x)

/-- Optional real observation succeeds exactly when the word is finite. -/
theorem toReal?_eq_ite (x : ExecFloat.Binary 8 23) :
    toReal? x = if isFinite x then some ((toModel x).toReal) else none := by
  change Model.toReal? (toModel x) =
    if Model.isFinite (toModel x) then some (Model.toReal (toModel x)) else none
  rw [← Model.toDyadic?_isSome_eq_isFinite]
  cases hd : Model.toDyadic? (toModel x) <;>
    simp [Model.toReal?, Model.toReal, hd]

/-- A finite word has the same optional and total real observations. -/
theorem toReal?_eq_some_toReal_of_isFinite_eq_true {x : ExecFloat.Binary 8 23}
    (hx : isFinite x = true) : toReal? x = some ((toModel x).toReal) := by
  rw [toReal?_eq_ite, hx]
  rfl

/-- A nonfinite word has no optional real observation. -/
theorem toReal?_eq_none_of_isFinite_eq_false {x : ExecFloat.Binary 8 23}
    (hx : isFinite x = false) : toReal? x = none := by
  rw [toReal?_eq_ite, hx]
  rfl

/-- ULP exponent of a finite word, including the subnormal spacing at zero. -/
def ulpExp? (x : ExecFloat.Binary 8 23) : Option Int :=
  match (Model.toDyadic? ∘ toModel) x with
  | some d =>
      some (if d.significand = 0 then -149
        else fexp32 (Int.ofNat d.significand.log2 + d.exponent + 1))
  | none => none

/-- ULP queries succeed precisely on finite words. -/
@[simp] theorem ulpExp?_isSome (x : ExecFloat.Binary 8 23) :
    (ulpExp? x).isSome = isFinite x := by
  change (ulpExp? x).isSome = Model.isFinite (toModel x)
  rw [← Model.toDyadic?_isSome_eq_isFinite]
  cases hd : (toModel x).toDyadic? <;> simp [ulpExp?, hd]

/-- Infinities and NaNs are exactly the words without a finite ULP exponent. -/
theorem ulpExp?_eq_none_iff (x : ExecFloat.Binary 8 23) :
    ulpExp? x = none ↔ isFinite x = false := by
  rw [← ulpExp?_isSome]
  cases ulpExp? x <;> simp

/-- The executable ULP exponent gives the exact real spacing at a decoded value. -/
theorem bpow_ulpExp?_eq_ulp32 (x : ExecFloat.Binary 8 23) {d : FloatLib.Numerics.Dyadic}
    (hx : (toModel x).toDyadic? = some d) :
    (ulpExp? x).map (bpow binaryRadix) = some (ulp32 ((toModel x).toReal)) := by
  have hxreal : (toModel x).toReal = d.toReal := by
    rw [Model.toReal_eq, hx]
  by_cases hm : d.significand = 0
  · have hz : d.toReal = 0 := by
      have h := Model.Dyadic.abs_toReal d
      rw [hm, Nat.cast_zero, zero_mul] at h
      exact abs_eq_zero.mp h
    simp only [ulpExp?, Function.comp_apply, hx, hm, ↓reduceIte, Option.map_some, hxreal, hz,
      ulp32_zero]
  · have hpos : 0 < _root_.abs d.toReal := by
      rw [Model.Dyadic.abs_toReal]
      exact mul_pos (by exact_mod_cast Nat.pos_of_ne_zero hm)
        (bpow.pos binaryRadix d.exponent)
    have hne : d.toReal ≠ 0 := abs_pos.mp hpos
    simp only [ulpExp?, Function.comp_apply, hx, hm, ↓reduceIte, Option.map_some, Option.some.injEq]
    rw [hxreal]
    change bpow binaryRadix (fexp32 (Int.ofNat d.significand.log2 + d.exponent + 1)) =
      ulp binaryRadix fexp32 d.toReal
    rw [ulp.of_ne_zero binaryRadix fexp32 d.toReal hne]
    simp only [cexp]
    rw [Model.Dyadic.magnitude_toReal d hm]

/-- A successful ULP query identifies the real spacing directly. -/
theorem bpow_eq_ulp32_of_ulpExp?_eq_some {x : ExecFloat.Binary 8 23} {k : Int}
    (hx : ulpExp? x = some k) :
    bpow binaryRadix k = ulp32 ((toModel x).toReal) := by
  cases hd : (toModel x).toDyadic? with
  | none => simp [ulpExp?, hd] at hx
  | some d =>
      have h := bpow_ulpExp?_eq_ulp32 x hd
      rw [hx] at h
      exact Option.some.inj h

/-- Whether adding the second word leaves the first word unchanged. -/
def absorbs (a b : ExecFloat.Binary 8 23) : Bool := decide (ExecFloat.add a b = a)

/-- An observed absorption agrees with the rounded-real addition. -/
theorem round32_add_eq_left_of_absorbs {a b : ExecFloat.Binary 8 23} {da db :
  FloatLib.Numerics.Dyadic}
    (ha : (toModel a).toDyadic? = some da) (hb : (toModel b).toDyadic? = some db)
    (hfin : isFinite (ExecFloat.add a b) = true) (habs : absorbs a b = true) :
    round32 ((toModel a).toReal + (toModel b).toReal) = (toModel a).toReal := by
  have _ := ha
  have _ := hb
  have h := toReal_add_eq_fp32Round_of_isFinite hfin
  have heq : ExecFloat.add a b = a := of_decide_eq_true habs
  rw [heq] at h
  exact h.symm

/-- Finiteness checks suffice to establish rounded-real absorption. -/
theorem round32_add_eq_left_of_absorbs_of_isFinite {a b : ExecFloat.Binary 8 23}
    (ha : isFinite a = true) (hb : isFinite b = true)
    (hadd : isFinite (ExecFloat.add a b) = true) (habs : absorbs a b = true) :
    round32 ((toModel a).toReal + (toModel b).toReal) = (toModel a).toReal := by
  obtain ⟨da, hda⟩ := Model.exists_toDyadic?_of_isFinite (x := toModel a) ha
  obtain ⟨db, hdb⟩ := Model.exists_toDyadic?_of_isFinite (x := toModel b) hb
  exact round32_add_eq_left_of_absorbs (da := da) (db := db) hda hdb hadd habs

/-- Effective nearest-even representation of a finite subtraction. -/
theorem toReal_sub_eq_computed_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hx : isFinite x = true) (hy : isFinite y = true)
    (hfin : isFinite (ExecFloat.sub x y) = true) :
    (toModel (ExecFloat.sub x y)).toReal =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix fexp32 ((toModel x).toReal - (toModel y).toReal))
        exponent := cexp binaryRadix fexp32 ((toModel x).toReal - (toModel y).toReal) } := by
  rw [toReal_sub_eq_fp32Round_of_isFinite hx hy hfin]
  exact fp32Round_eq_computed _

/-- Effective nearest-even representation of a finite division result. -/
theorem toReal_div_eq_computed_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hfin : isFinite (ExecFloat.div x y) = true) :
    (toModel (ExecFloat.div x y)).toReal =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix fexp32 ((toModel x).toReal / (toModel y).toReal))
        exponent := cexp binaryRadix fexp32 ((toModel x).toReal / (toModel y).toReal) } := by
  rw [toReal_div_eq_fp32Round_of_isFinite x y hfin]
  exact fp32Round_eq_computed _

/-- Optional addition result characterized by one real rounding. -/
theorem toReal?_add_eq_ite (x y : ExecFloat.Binary 8 23) :
    toReal? (ExecFloat.add x y) =
      if isFinite (ExecFloat.add x y) then some (fp32Round ((toModel x).toReal + (toModel
        y).toReal)) else none := by
  rw [toReal?_eq_ite]
  by_cases h : isFinite (ExecFloat.add x y) = true
  · simp only [h, ↓reduceIte]
    exact congrArg some (toReal_add_eq_fp32Round_of_isFinite h)
  · simp [h]

/-- Optional multiplication result characterized by one real rounding. -/
theorem toReal?_mul_eq_ite (x y : ExecFloat.Binary 8 23) :
    toReal? (ExecFloat.mul x y) =
      if isFinite (ExecFloat.mul x y) then some (fp32Round ((toModel x).toReal * (toModel
        y).toReal)) else none := by
  rw [toReal?_eq_ite]
  by_cases h : isFinite (ExecFloat.mul x y) = true
  · simp only [h, ↓reduceIte]
    exact congrArg some (toReal_mul_eq_fp32Round_of_isFinite h)
  · simp [h]

/-- Optional fused multiply-add result characterized by one real rounding. -/
theorem toReal?_fma_eq_ite (x y z : ExecFloat.Binary 8 23) :
    toReal? ((ExecFloat.Binary.fma (rounding := .nearestEven)) x y z) =
      if isFinite ((ExecFloat.Binary.fma (rounding := .nearestEven)) x y z) then some (fp32Round
        ((toModel x).toReal * (toModel y).toReal + (toModel z).toReal))
      else none := by
  rw [toReal?_eq_ite]
  by_cases h : isFinite ((ExecFloat.Binary.fma (rounding := .nearestEven)) x y z) = true
  · simp only [h, ↓reduceIte]
    exact congrArg some (toReal_fma_eq_fp32Round_of_isFinite x y z h)
  · simp [h]

/-- Optional square-root result characterized by one real rounding. -/
theorem toReal?_sqrt_eq_ite (x : ExecFloat.Binary 8 23) :
    toReal? ((ExecFloat.Binary.sqrt (rounding := .nearestEven)) x) =
      if isFinite ((ExecFloat.Binary.sqrt (rounding := .nearestEven)) x) then some (fp32Round
        (Real.sqrt ((toModel x).toReal))) else none := by
  rw [toReal?_eq_ite]
  by_cases h : isFinite ((ExecFloat.Binary.sqrt (rounding := .nearestEven)) x) = true
  · simp only [h, ↓reduceIte]
    exact congrArg some (toReal_sqrt_eq_fp32Round_of_isFinite x h)
  · simp [h]

/-- Optional division result, including finite division by infinity. -/
theorem toReal?_div_eq_ite (x y : ExecFloat.Binary 8 23) :
    toReal? (ExecFloat.div x y) =
      if isFinite (ExecFloat.div x y) then some (fp32Round ((toModel x).toReal / (toModel
        y).toReal)) else none := by
  rw [toReal?_eq_ite]
  by_cases h : isFinite (ExecFloat.div x y) = true
  · simp only [h, ↓reduceIte]
    exact congrArg some (toReal_div_eq_fp32Round_of_isFinite x y h)
  · simp [h]

/-- Minimum of two finite words remains finite, including either signed zero. -/
theorem isFinite_minimum_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    isFinite (min x y) = true := by
  change Model.isFinite (FloatLib.Floats.ExecFloat.Binary.toModel (min x y)) = true
  rw [FloatLib.Floats.ExecFloat.Binary.toModel_min]
  change Model.isFinite (toModel x) = true at hx
  change Model.isFinite (toModel y) = true at hy
  have hxNaN := Model.isNaN_eq_false_of_isFinite_eq_true (toModel x) hx
  have hyNaN := Model.isNaN_eq_false_of_isFinite_eq_true (toModel y) hy
  have hc := Model.chooseNaN2_none_of_not_isNaN (toModel x) (toModel y) hxNaN hyNaN
  have hz (s : Bool) : Model.isFinite (Model.zero FloatFormat.binary32 s) = true :=
    Model.isFinite_eq_true_of_isZero_eq_true _ (Model.isZero_zero _ s)
  rw [Model.minimum, Model.withNaNSelection_of_none _ _ hc]
  cases Model.compareNonNaN (toModel x) (toModel y) hxNaN hyNaN with
  | lt => exact hx
  | gt => exact hy
  | eq =>
      dsimp only
      split
      · exact hz (Model.signBit (toModel x) || Model.signBit (toModel y))
      · exact hx

/-- Maximum of two finite words remains finite, including either signed zero. -/
theorem isFinite_maximum_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    isFinite (max x y) = true := by
  change Model.isFinite (FloatLib.Floats.ExecFloat.Binary.toModel (max x y)) = true
  rw [FloatLib.Floats.ExecFloat.Binary.toModel_max]
  change Model.isFinite (toModel x) = true at hx
  change Model.isFinite (toModel y) = true at hy
  have hxNaN := Model.isNaN_eq_false_of_isFinite_eq_true (toModel x) hx
  have hyNaN := Model.isNaN_eq_false_of_isFinite_eq_true (toModel y) hy
  have hc := Model.chooseNaN2_none_of_not_isNaN (toModel x) (toModel y) hxNaN hyNaN
  have hz (s : Bool) : Model.isFinite (Model.zero FloatFormat.binary32 s) = true :=
    Model.isFinite_eq_true_of_isZero_eq_true _ (Model.isZero_zero _ s)
  rw [Model.maximum, Model.withNaNSelection_of_none _ _ hc]
  cases Model.compareNonNaN (toModel x) (toModel y) hxNaN hyNaN with
  | lt => exact hy
  | gt => exact hx
  | eq =>
      dsimp only
      split
      · exact hz (Model.signBit (toModel x) && Model.signBit (toModel y))
      · exact hx

/-- Optional minimum agrees with real minimum on finite inputs. -/
theorem toReal?_minimum_eq_min_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal? (min x y) = some (min ((toModel x).toReal) ((toModel y).toReal)) := by
  rw [toReal?_eq_some_toReal_of_isFinite_eq_true (isFinite_minimum_of_isFinite x y hx hy)]
  congr 1
  rw [FloatLib.Floats.ExecFloat.Binary.toModel_min]
  exact Model.toReal_minimum_eq_min_of_isFinite (toModel x) (toModel y) hx hy

/-- Optional maximum agrees with real maximum on finite inputs. -/
theorem toReal?_maximum_eq_max_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal? (max x y) = some (max ((toModel x).toReal) ((toModel y).toReal)) := by
  rw [toReal?_eq_some_toReal_of_isFinite_eq_true (isFinite_maximum_of_isFinite x y hx hy)]
  rw [toReal_maximum_eq_max_of_isFinite hx hy]

/-- Nearest-even binary32 rounding incurs at most half an ULP of absolute error. -/
theorem fp32Round_abs_error (x : ℝ) :
    _root_.abs (fp32Round x - x) ≤ eps32 x :=
  FP32.round_abs_error x

/-- Normal nonzero results have relative error at most binary32 unit roundoff. -/
theorem fp32Round_relative_error_of_normal (x : ℝ) (hx : x ≠ 0)
    (hnormal : FP32.minNormal ≤ _root_.abs x) :
    ErrorBounds.relativeError x (fp32Round x) hx ≤ bpow binaryRadix (-24) :=
  FP32.round_relative_error_of_normal x hx hnormal

/-- Half-ULP absolute error for a finite addition result. -/
theorem toReal_add_abs_error_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hfin : isFinite (ExecFloat.add x y) = true) :
    _root_.abs ((toModel (ExecFloat.add x y)).toReal - ((toModel x).toReal + (toModel y).toReal)) ≤
      eps32 ((toModel x).toReal + (toModel y).toReal) := by
  rw [toReal_add_eq_fp32Round_of_isFinite hfin]
  exact fp32Round_abs_error _

/-- Half-ULP absolute error for subtraction of finite values with a finite result. -/
theorem toReal_sub_abs_error_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hx : isFinite x = true) (hy : isFinite y = true)
    (hfin : isFinite (ExecFloat.sub x y) = true) :
    _root_.abs ((toModel (ExecFloat.sub x y)).toReal - ((toModel x).toReal - (toModel y).toReal)) ≤
      eps32 ((toModel x).toReal - (toModel y).toReal) := by
  rw [toReal_sub_eq_fp32Round_of_isFinite hx hy hfin]
  exact fp32Round_abs_error _

/-- Half-ULP absolute error for a finite multiplication result. -/
theorem toReal_mul_abs_error_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hfin : isFinite (ExecFloat.mul x y) = true) :
    _root_.abs ((toModel (ExecFloat.mul x y)).toReal - ((toModel x).toReal * (toModel y).toReal)) ≤
      eps32 ((toModel x).toReal * (toModel y).toReal) := by
  rw [toReal_mul_eq_fp32Round_of_isFinite hfin]
  exact fp32Round_abs_error _

/-- Half-ULP absolute error for a finite division result. -/
theorem toReal_div_abs_error_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hfin : isFinite (ExecFloat.div x y) = true) :
    _root_.abs ((toModel (ExecFloat.div x y)).toReal - ((toModel x).toReal / (toModel y).toReal)) ≤
      eps32 ((toModel x).toReal / (toModel y).toReal) := by
  rw [toReal_div_eq_fp32Round_of_isFinite x y hfin]
  exact fp32Round_abs_error _

/-- Half-ULP absolute error for a finite single-rounding fused multiply-add result. -/
theorem toReal_fma_abs_error_of_isFinite (x y z : ExecFloat.Binary 8 23)
    (hfin : isFinite ((ExecFloat.Binary.fma (rounding := .nearestEven)) x y z) = true) :
    _root_.abs ((toModel ((ExecFloat.Binary.fma (rounding := .nearestEven)) x y z)).toReal -
      ((toModel x).toReal * (toModel y).toReal + (toModel z).toReal)) ≤
      eps32 ((toModel x).toReal * (toModel y).toReal + (toModel z).toReal) := by
  rw [toReal_fma_eq_fp32Round_of_isFinite x y z hfin]
  exact fp32Round_abs_error _

/-- Half-ULP absolute error for a finite square-root result. -/
theorem toReal_sqrt_abs_error_of_isFinite (x : ExecFloat.Binary 8 23)
    (hfin : isFinite ((ExecFloat.Binary.sqrt (rounding := .nearestEven)) x) = true) :
    _root_.abs ((toModel ((ExecFloat.Binary.sqrt (rounding := .nearestEven)) x)).toReal - Real.sqrt
      ((toModel x).toReal)) ≤ eps32 (Real.sqrt ((toModel x).toReal)) := by
  rw [toReal_sqrt_eq_fp32Round_of_isFinite x hfin]
  exact fp32Round_abs_error _

end TorchLean.Floats.IEEE754.IEEE32Exec
