/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Floats.ExecFloat.Proof.Arithmetic
import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Proof

/-!
# Finite configured arithmetic for reduction proofs

FloatLib supplies the arithmetic kernels and their real rounding semantics. This module transports
those theorems to configured values for any IEEE format and model codec. A finite add or multiply
result implies finite operands, so their refinement statements need only result finiteness. The
binary32 declarations specialize this shared bridge for existing reduction proofs. No global
relative-error assumption is made at subnormal values.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite ofModel toModel toModel_ofModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Floats.IEEEExec

open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- Configured addition decodes to FloatLib's proved model addition. -/
@[simp] theorem toModel_add (x y : Value) :
    toModel (ExecFloat.add x y) = Model.add (toModel x) (toModel y) := by
  rw [FloatLib.Floats.ExecFloat.Proof.add_eq_spec, Model.Proof.add_eq_spec]
  exact ExecFloat.Binary.toModel_ofModel (Model.Spec.add (toModel x) (toModel y))

/-- Configured multiplication decodes to FloatLib's proved model multiplication. -/
@[simp] theorem toModel_mul (x y : Value) :
    toModel (ExecFloat.mul x y) = Model.mul (toModel x) (toModel y) := by
  rw [FloatLib.Floats.ExecFloat.Proof.mul_eq_spec, Model.Proof.mul_eq_spec]
  exact ExecFloat.Binary.toModel_ofModel (Model.Spec.mul (toModel x) (toModel y))

/-- Quieting a NaN preserves its non-finite classification. -/
theorem model_isFinite_quietNaN (hformat : format.isIEEE = true) (x : Model format) :
    Model.isFinite (Model.quietNaN x) = Model.isFinite x := by
  have he := ((FloatFormat.isIEEE_eq_true_iff format).mp hformat).1
  simp only [Model.quietNaN, Model.isFinite, he]
  split
  · exact congrArg (fun e => e != format.expAllOnesNat)
      (Model.expField_or_quietBit x.bits)
  · rfl

/-- A selected IEEE NaN cannot be finite. -/
theorem model_isFinite_chooseNaN2 (hformat : format.isIEEE = true) (x y n : Model format)
    (h : Model.chooseNaN2 x y = some n) : Model.isFinite n = false := by
  have hxS := Model.isSNaN_eq_false_of_isNaN_eq_false x
  have hyS := Model.isSNaN_eq_false_of_isNaN_eq_false y
  cases hxNaN : Model.isNaN x <;> cases hyNaN : Model.isNaN y <;>
    cases hxSNaN : Model.isSNaN x <;> cases hySNaN : Model.isSNaN y <;>
    simp_all [Model.chooseNaN2]
  all_goals
    subst n
    rw [model_isFinite_quietNaN hformat]
    apply Model.isFinite_eq_false_of_isNaN
    assumption

/-- An IEEE invalid operation produces a non-finite value. -/
theorem model_isFinite_invalidResult (hformat : format.isIEEE = true) :
    Model.isFinite (Model.invalidResult format) = false := by
  have he := ((FloatFormat.isIEEE_eq_true_iff format).mp hformat).1
  rw [Model.invalidResult_eq_canonicalNaN_of_isIEEE format hformat]
  simp only [Model.isFinite, he, Model.IEEE.isFinite, Model.canonicalNaN,
    Model.expField_or_quietBit]
  have hexp := Model.expField_ofModel_infinity format .positive
  rw [← Model.posInf_eq_ofModel_infinity] at hexp
  simp only [Model.posInf] at hexp
  rw [hexp]
  simp

/-- Overflow in an IEEE format produces a non-finite value with either sign. -/
theorem model_isFinite_nativeOverflow (hformat : format.isIEEE = true) (sign : Bool) :
    Model.isFinite (Model.nativeOverflow format sign) = false := by
  have hs := FloatFormat.supportsInfinity_eq_true_of_isIEEE format hformat
  cases sign <;> simp [Model.nativeOverflow_eq_signedInf_of_isIEEE format hformat,
    Model.isFinite_posInf format hs, Model.isFinite_negInf format hs]

private theorem model_finite_operands_of_add (hformat : format.isIEEE = true) (x y : Model format)
    (hfin : Model.isFinite (Model.add x y) = true) :
    Model.isFinite x = true ∧ Model.isFinite y = true := by
  have hinvalid := model_isFinite_invalidResult hformat
  have hxInf := Model.isInf_eq_false_of_isFinite_eq_true x
  have hyInf := Model.isInf_eq_false_of_isFinite_eq_true y
  cases hdx : Model.toDyadic? x <;> cases hdy : Model.toDyadic? y
  case some.some dx dy =>
    exact ⟨Model.isFinite_eq_true_of_toDyadic?_some hdx,
      Model.isFinite_eq_true_of_toDyadic?_some hdy⟩
  all_goals
    exfalso
    rw [Model.Proof.add_eq_spec] at hfin
    simp only [Model.Spec.add, hdx, hdy] at hfin
    cases hnan : Model.chooseNaN2 x y with
    | some n =>
        have hn := model_isFinite_chooseNaN2 hformat x y n hnan
        simp [hnan, hn] at hfin
    | none =>
        simp only [hnan] at hfin
        split at hfin
        · split at hfin
          · split at hfin <;> simp_all
          · simp_all
        · split at hfin <;> simp_all

private theorem model_finite_operands_of_mul (hformat : format.isIEEE = true) (x y : Model format)
    (hfin : Model.isFinite (Model.mul x y) = true) :
    Model.isFinite x = true ∧ Model.isFinite y = true := by
  have hinvalid := model_isFinite_invalidResult hformat
  have hoverflow := model_isFinite_nativeOverflow hformat
  cases hdx : Model.toDyadic? x <;> cases hdy : Model.toDyadic? y
  case some.some dx dy =>
    exact ⟨Model.isFinite_eq_true_of_toDyadic?_some hdx,
      Model.isFinite_eq_true_of_toDyadic?_some hdy⟩
  all_goals
    exfalso
    rw [Model.Proof.mul_eq_spec] at hfin
    simp only [Model.Spec.mul, hdx, hdy] at hfin
    cases hnan : Model.chooseNaN2 x y with
    | some n =>
        have hn := model_isFinite_chooseNaN2 hformat x y n hnan
        simp [hnan, hn] at hfin
    | none =>
        simp only [hnan] at hfin
        split at hfin
        · split at hfin
          · simp only [hinvalid, Bool.false_eq_true] at hfin
          · simp only [hoverflow, Bool.false_eq_true] at hfin
        · split at hfin
          · split at hfin
            · simp only [hinvalid, Bool.false_eq_true] at hfin
            · simp only [hoverflow, Bool.false_eq_true] at hfin
          · simp only [hinvalid, Bool.false_eq_true] at hfin

/-- A finite executable sum is one rounding in the selected IEEE format of the exact real sum. -/
theorem toReal_add_eq_round_of_isFinite (hformat : format.isIEEE = true) {x y : Value}
    (hfin : isFinite (ExecFloat.add x y) = true) :
    (toModel (ExecFloat.add x y)).toReal =
      Model.roundAt format ((toModel x).toReal + (toModel y).toReal) := by
  have hmodel : Model.isFinite (Model.add (toModel x) (toModel y)) = true :=
    (congrArg Model.isFinite (toModel_add x y)).symm.trans hfin
  obtain ⟨hx, hy⟩ := model_finite_operands_of_add hformat (toModel x) (toModel y) hmodel
  rw [toModel_add]
  exact Model.toReal_add_eq_roundAt (fmt := format)
    (toModel x) (toModel y) hformat hx hy hmodel

/-- A finite executable product is one rounding in the selected IEEE format of the exact real
product. -/
theorem toReal_mul_eq_round_of_isFinite (hformat : format.isIEEE = true) {x y : Value}
    (hfin : isFinite (ExecFloat.mul x y) = true) :
    (toModel (ExecFloat.mul x y)).toReal =
      Model.roundAt format ((toModel x).toReal * (toModel y).toReal) := by
  have hmodel : Model.isFinite (Model.mul (toModel x) (toModel y)) = true :=
    (congrArg Model.isFinite (toModel_mul x y)).symm.trans hfin
  obtain ⟨hx, hy⟩ := model_finite_operands_of_mul hformat (toModel x) (toModel y) hmodel
  rw [toModel_mul]
  exact Model.toReal_mul_eq_roundAt (fmt := format)
    (toModel x) (toModel y) hformat hx hy hmodel

/-- Configured subtraction decodes to FloatLib’s proved model subtraction. -/
theorem toModel_sub (x y : Value) :
    toModel (ExecFloat.sub x y) = Model.sub (toModel x) (toModel y) := by
  rw [FloatLib.Floats.ExecFloat.Proof.sub_eq_spec, Model.Proof.sub_eq_spec]
  exact ExecFloat.Binary.toModel_ofModel _

/-- Finite subtraction refines one nearest-even rounding of the real difference. -/
theorem toReal_sub_eq_round_of_isFinite (hformat : format.isIEEE = true) {x y : Value}
    (hx : isFinite x = true) (hy : isFinite y = true)
    (hfin : isFinite (ExecFloat.sub x y) = true) :
    (toModel (ExecFloat.sub x y)).toReal =
      Model.roundAt format ((toModel x).toReal - (toModel y).toReal) := by
  have hmodel : Model.isFinite (Model.sub (toModel x) (toModel y)) = true :=
    (congrArg Model.isFinite (toModel_sub x y)).symm.trans hfin
  rw [toModel_sub]
  exact Model.toReal_sub_eq_roundAt (toModel x) (toModel y) hformat hx hy hmodel

/-- The configured zero literal decodes to positive model zero. -/
@[simp] theorem toModel_zero : toModel (0 : Value) = Model.zero format false := by
  change toModel (ofModel (Model.roundRatQ format 0) : Value) = _
  rw [ExecFloat.Binary.toModel_ofModel]
  simp [Model.roundRatQ, Model.roundRatQWithRounding, Model.roundRatWithRounding,
    Model.roundRatWithRoundingScaled]

/-- The configured zero literal denotes real zero. -/
@[simp] theorem toReal_zero : (toModel (0 : Value)).toReal = 0 := by
  rw [toModel_zero, Model.toReal_zero]

/-- The configured zero literal is finite. -/
theorem isFinite_zero : isFinite (0 : Value) = true := by
  change Model.isFinite (toModel (0 : Value)) = true
  rw [toModel_zero]
  exact Model.isFinite_eq_true_of_isZero_eq_true _ (Model.isZero_zero format false)

/-- IEEE maximum agrees with real maximum on finite operands, including signed zeros. -/
theorem toReal_maximum_eq_max_of_isFinite {x y : Value}
    (hx : isFinite x = true) (hy : isFinite y = true) :
    (toModel (max x y)).toReal = max (toModel x).toReal (toModel y).toReal := by
  rw [ExecFloat.Binary.toModel_max]
  exact Model.toReal_maximum_eq_max_of_isFinite (toModel x) (toModel y) hx hy

/-- Finite dyadic conversion refines rounding of the exact dyadic real value. -/
theorem toReal_roundDyadic_eq_round (hformat : format.isIEEE = true)
    {d : FloatLib.Numerics.Dyadic}
    (hfin : isFinite (ofModel (Model.roundDyadic format d) : Value) = true) :
    (toModel (ofModel (Model.roundDyadic format d) : Value)).toReal =
      Model.roundAt format d.toReal := by
  have hdecode : toModel (ofModel (Model.roundDyadic format d) : Value) =
      Model.roundDyadic format d := ExecFloat.Binary.toModel_ofModel _
  change Model.isFinite (toModel (ofModel (Model.roundDyadic format d) : Value)) = true at hfin
  rw [hdecode] at hfin ⊢
  exact Model.toReal_roundDyadic_eq_roundAt format hformat d hfin

end TorchLean.Floats.IEEEExec

namespace TorchLean.Floats.IEEE754.IEEE32Exec

open FloatLib.Numerics FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.Formats.BinaryInterchange

/-- Configured addition decodes to FloatLib's proved model addition. -/
@[simp] theorem toModel_add (x y : ExecFloat.Binary 8 23) :
    toModel (ExecFloat.add x y) = Model.add (toModel x) (toModel y) :=
  IEEEExec.toModel_add x y

/-- Configured multiplication decodes to FloatLib's proved model multiplication. -/
@[simp] theorem toModel_mul (x y : ExecFloat.Binary 8 23) :
    toModel (ExecFloat.mul x y) = Model.mul (toModel x) (toModel y) :=
  IEEEExec.toModel_mul x y

/-- Quieting a binary32 NaN preserves its non-finite classification. -/
theorem model_isFinite_quietNaN (x : Model FloatFormat.binary32) :
    Model.isFinite (Model.quietNaN x) = Model.isFinite x :=
  IEEEExec.model_isFinite_quietNaN (by decide) x

/-- A selected binary32 NaN cannot be finite. -/
theorem model_isFinite_chooseNaN2 (x y n : Model FloatFormat.binary32)
    (h : Model.chooseNaN2 x y = some n) : Model.isFinite n = false :=
  IEEEExec.model_isFinite_chooseNaN2 (by decide) x y n h

/-- The invalid-operation result is a non-finite binary32 value. -/
theorem model_isFinite_invalidResult :
    Model.isFinite (Model.invalidResult FloatFormat.binary32) = false :=
  IEEEExec.model_isFinite_invalidResult (by decide)

/-- Binary32 overflow produces a non-finite value with either sign. -/
theorem model_isFinite_nativeOverflow (sign : Bool) :
    Model.isFinite (Model.nativeOverflow FloatFormat.binary32 sign) = false :=
  IEEEExec.model_isFinite_nativeOverflow (by decide) sign

/-- A finite executable sum is one binary32 rounding of the exact real sum. -/
theorem toReal_add_eq_round_of_isFinite {x y : ExecFloat.Binary 8 23}
    (hfin : isFinite (ExecFloat.add x y) = true) :
    (toModel (ExecFloat.add x y)).toReal =
      Model.roundAt FloatFormat.binary32 ((toModel x).toReal + (toModel y).toReal) :=
  IEEEExec.toReal_add_eq_round_of_isFinite (by decide) hfin

/-- A finite executable product is one binary32 rounding of the exact real product. -/
theorem toReal_mul_eq_round_of_isFinite {x y : ExecFloat.Binary 8 23}
    (hfin : isFinite (ExecFloat.mul x y) = true) :
    (toModel (ExecFloat.mul x y)).toReal =
      Model.roundAt FloatFormat.binary32 ((toModel x).toReal * (toModel y).toReal) :=
  IEEEExec.toReal_mul_eq_round_of_isFinite (by decide) hfin

/-- Effective representation of a finite executable product. -/
theorem toReal_mul_eq_computed_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hfin : isFinite (ExecFloat.mul x y) = true) :
    (toModel (ExecFloat.mul x y)).toReal =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix (Model.fexpOf FloatFormat.binary32)
            ((toModel x).toReal * (toModel y).toReal))
        exponent := cexp binaryRadix (Model.fexpOf FloatFormat.binary32)
          ((toModel x).toReal * (toModel y).toReal) } := by
  rw [toReal_mul_eq_round_of_isFinite hfin]
  exact round_nearestEven_computed _

end TorchLean.Floats.IEEE754.IEEE32Exec
