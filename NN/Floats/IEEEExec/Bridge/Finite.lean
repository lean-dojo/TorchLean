/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Core
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Floats.ExecFloat.Proof.Arithmetic

/-!
# Finite binary32 arithmetic for reduction proofs

FloatLib supplies the arithmetic kernels and their real rounding semantics. This module transports
those theorems to FloatLib's configured binary32 values. A finite add or multiply result implies
finite operands, so the public refinement statements need only the original result-finiteness
hypothesis. No global relative-error assumption is made at subnormal values.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite toModel toModel_ofModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Floats.IEEE754.IEEE32Exec

open FloatLib.Numerics FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.Formats.BinaryInterchange

private abbrev Binary32Model := Model FloatFormat.binary32

/-- Configured addition decodes to FloatLib's proved model addition. -/
@[simp] theorem toModel_add (x y : ExecFloat.Binary 8 23) :
    toModel (ExecFloat.add x y) = Model.add (toModel x) (toModel y) := by
  rw [FloatLib.Floats.ExecFloat.Proof.add_eq_spec, Model.Proof.add_eq_spec]
  exact toModel_ofModel (Model.Spec.add (toModel x) (toModel y))

/-- Configured multiplication decodes to FloatLib's proved model multiplication. -/
@[simp] theorem toModel_mul (x y : ExecFloat.Binary 8 23) :
    toModel (ExecFloat.mul x y) = Model.mul (toModel x) (toModel y) := by
  rw [FloatLib.Floats.ExecFloat.Proof.mul_eq_spec, Model.Proof.mul_eq_spec]
  exact toModel_ofModel (Model.Spec.mul (toModel x) (toModel y))

/-- Quieting a binary32 NaN preserves its non-finite classification. -/
theorem model_isFinite_quietNaN (x : Model FloatFormat.binary32) :
    Model.isFinite (Model.quietNaN x) = Model.isFinite x := by
  have hexp :
      Model.expField (Model.ofBits (x.bits ||| FloatFormat.quietBit FloatFormat.binary32)) =
        Model.expField x := by
    let bits : BitVec 32 := x.bits
    change (((bits ||| (0x00400000 : BitVec 32)) >>> 23) &&& 255).toNat =
      ((bits >>> 23) &&& 255).toNat
    have hquiet : (0x00400000 : BitVec 32) >>> 23 = 0#32 := by decide
    rw [BitVec.ushiftRight_or_distrib, hquiet, BitVec.or_zero]
  change Model.IEEE.isFinite
      (if Model.IEEE.isNaN x then
        Model.ofBits (x.bits ||| FloatFormat.quietBit FloatFormat.binary32) else x) =
    Model.IEEE.isFinite x
  split
  · simp only [Model.IEEE.isFinite, hexp]
  · rfl

/-- A selected binary32 NaN cannot be finite. -/
theorem model_isFinite_chooseNaN2 (x y n : Model FloatFormat.binary32)
    (h : Model.chooseNaN2 x y = some n) : Model.isFinite n = false := by
  have hxS := Model.isSNaN_eq_false_of_isNaN_eq_false x
  have hyS := Model.isSNaN_eq_false_of_isNaN_eq_false y
  cases hxNaN : Model.isNaN x <;> cases hyNaN : Model.isNaN y <;>
    cases hxSNaN : Model.isSNaN x <;> cases hySNaN : Model.isSNaN y <;>
    simp_all [Model.chooseNaN2]
  all_goals
    subst n
    rw [model_isFinite_quietNaN]
    apply Model.isFinite_eq_false_of_isNaN
    assumption

private theorem model_finite_operands_of_add (x y : Binary32Model)
    (hfin : Model.isFinite (Model.add x y) = true) :
    Model.isFinite x = true ∧ Model.isFinite y = true := by
  have hinvalid : Model.isFinite (Model.invalidResult FloatFormat.binary32) = false := by
    decide
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
        have hn := model_isFinite_chooseNaN2 x y n hnan
        simp [hnan, hn] at hfin
    | none =>
        simp only [hnan] at hfin
        split at hfin
        · split at hfin
          · split at hfin <;> simp_all
          · simp_all
        · split at hfin <;> simp_all

private theorem model_finite_operands_of_mul (x y : Binary32Model)
    (hfin : Model.isFinite (Model.mul x y) = true) :
    Model.isFinite x = true ∧ Model.isFinite y = true := by
  have hinvalid : Model.isFinite (Model.invalidResult FloatFormat.binary32) = false := by
    decide
  have hoverflow (sign : Bool) :
      Model.isFinite (Model.nativeOverflow FloatFormat.binary32 sign) = false := by
    cases sign <;> decide
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
        have hn := model_isFinite_chooseNaN2 x y n hnan
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

/-- Nearest-even real rounding with binary32 gradual underflow and no upper exponent bound. -/
noncomputable abbrev fp32Round (x : ℝ) : ℝ :=
  round (β := binaryRadix) (fexp := fexp32) rnd32 x

/-- FloatLib's binary32 real rounding is TorchLean's gradual-underflow rounding grid. -/
theorem roundAt_binary32 (x : ℝ) :
    Model.roundAt FloatFormat.binary32 x = fp32Round x := by
  unfold Model.roundAt Model.fexpOf fp32Round fexp32 rnd32
  rfl

/-- A finite executable sum is one binary32 rounding of the exact real sum. -/
theorem toReal_add_eq_fp32Round_of_isFinite {x y : ExecFloat.Binary 8 23}
    (hfin : isFinite (ExecFloat.add x y) = true) :
    (toModel (ExecFloat.add x y)).toReal = fp32Round ((toModel x).toReal + (toModel y).toReal) := by
  have hmodel : Model.isFinite (Model.add (toModel x) (toModel y)) = true := by
    change Model.isFinite (toModel (ExecFloat.add x y)) = true at hfin
    simpa only [toModel_add] using hfin
  obtain ⟨hx, hy⟩ := model_finite_operands_of_add (toModel x) (toModel y) hmodel
  rw [toModel_add, ← roundAt_binary32]
  exact Model.toReal_add_eq_roundAt (fmt := FloatFormat.binary32)
    (toModel x) (toModel y) (by decide) hx hy hmodel

/-- A finite executable product is one binary32 rounding of the exact real product. -/
theorem toReal_mul_eq_fp32Round_of_isFinite {x y : ExecFloat.Binary 8 23}
    (hfin : isFinite (ExecFloat.mul x y) = true) :
    (toModel (ExecFloat.mul x y)).toReal = fp32Round ((toModel x).toReal * (toModel y).toReal) := by
  have hmodel : Model.isFinite (Model.mul (toModel x) (toModel y)) = true := by
    change Model.isFinite (toModel (ExecFloat.mul x y)) = true at hfin
    simpa only [toModel_mul] using hfin
  obtain ⟨hx, hy⟩ := model_finite_operands_of_mul (toModel x) (toModel y) hmodel
  rw [toModel_mul, ← roundAt_binary32]
  exact Model.toReal_mul_eq_roundAt (fmt := FloatFormat.binary32)
    (toModel x) (toModel y) (by decide) hx hy hmodel

/-- The rounded value has FloatLib's effective mantissa/exponent representation. -/
theorem fp32Round_eq_computed (z : ℝ) :
    fp32Round z =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa (scaledMantissa binaryRadix fexp32 z)
        exponent := cexp binaryRadix fexp32 z } :=
  FP32.round_eq_computed z

/-- Effective representation of a finite executable product. -/
theorem toReal_mul_eq_computed_of_isFinite (x y : ExecFloat.Binary 8 23)
    (hfin : isFinite (ExecFloat.mul x y) = true) :
    (toModel (ExecFloat.mul x y)).toReal =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix fexp32 ((toModel x).toReal * (toModel y).toReal))
        exponent := cexp binaryRadix fexp32 ((toModel x).toReal * (toModel y).toReal) } := by
  rw [toReal_mul_eq_fp32Round_of_isFinite hfin, fp32Round_eq_computed]

end TorchLean.Floats.IEEE754.IEEE32Exec
