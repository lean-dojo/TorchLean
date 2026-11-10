/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Floats.FP32
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError.Rounding
public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Floats.ExecFloat.Proof.Arithmetic

/-!
# Finite configured arithmetic for reduction proofs

FloatLib supplies the arithmetic kernels and their real rounding semantics, including the
finite-result rounding equalities for addition, subtraction and multiplication. This module adds
the configured-value facts that the reduction and expression proofs use beyond those, for any IEEE
format and model codec. Division and square root refine one rounding under result finiteness
alone, with exceptional inputs discharged internally. In particular, division of a finite value
by infinity produces signed zero, agreeing with the totalized real interpretation. No global
relative-error assumption is made at subnormal values.

We also decode the zero literal, finite minimum and maximum, and rounded dyadics, and state when
an addition absorbs its second operand. The half-ULP error of a finite arithmetic result follows
from its rounding equality, such as FloatLib's `ExecFloat.Binary.toReal_sub_eq_roundAt`, and
`Model.abs_roundAt_sub_le`.

`min` and `max` run FloatLib's `minimum` and `maximum`, the operations that IEEE 754-2019
recommends in 9.6. They return a quiet NaN for a NaN operand and put `-0` below `+0`. Both zeros
decode to the real `0`, so `toReal_minimum_eq_min` and `toReal_maximum_eq_max` hold at signed
zeros as well.

Theorems that need an IEEE descriptor take `hformat : format.isIEEE = true` as their last
argument, with FloatLib's `ieee_format` tactic as its default, which decides it for a concrete
IEEE format such as binary32.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite ofModel toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.IEEE754

open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-! ## Model division with exceptional operands -/

private theorem model_div_eq_roundAt (x y : Model format)
    (hfin : Model.isFinite (Model.div x y) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    Model.toReal (Model.div x y) = Model.roundAt format (Model.toReal x / Model.toReal y) := by
  cases hx : Model.toDyadic? x <;> cases hy : Model.toDyadic? y
  case some.some =>
    exact Model.toReal_div_eq_roundAt x y hformat
      (Model.isFinite_eq_true_of_toDyadic?_some hy) hfin
  all_goals
    have hdiv : Model.div x y = Model.Spec.divSpecial x y := by
      simp only [Model.Proof.div_eq_spec, Model.Spec.div, hx, hy]
    rw [hdiv] at hfin ⊢
    cases hchoose : Model.chooseNaN2 x y with
    | some n =>
        have hn := Model.isFinite_eq_false_of_isNaN (Model.isNaN_of_chooseNaN2_eq_some hchoose)
        simp [Model.Spec.divSpecial, hchoose, hn] at hfin
    | none =>
        have hxInf : Model.isInf x = false := by
          cases hi : Model.isInf x with
          | false => rfl
          | true =>
              simp only [Model.Spec.divSpecial, hchoose, hi, ↓reduceIte] at hfin
              split at hfin
              · simp only [Model.isFinite_invalidResult hformat, Bool.false_eq_true] at hfin
              · simp only [Model.isFinite_nativeOverflow hformat, Bool.false_eq_true] at hfin
        cases hyInf : Model.isInf y with
        | false =>
            simp only [Model.Spec.divSpecial, hchoose, hxInf, hyInf,
              Model.isFinite_invalidResult hformat, Bool.false_eq_true, ↓reduceIte] at hfin
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

/-! ## Literals, minimum and maximum -/

/-- The configured zero literal decodes to positive model zero. -/
@[simp] theorem toModel_zero : toModel (0 : Value) = Model.zero format false := by
  change toModel (ofModel (Model.ofRat format 0) : Value) = _
  rw [ExecFloat.Binary.toModel_ofModel]
  simp [Model.ofRat, Model.Internal.Directed.roundRat,
    Model.Internal.Directed.roundRatScaled]

/-- IEEE minimum agrees with real minimum on finite operands, including signed zeros. -/
theorem toReal_minimum_eq_min {x y : Value}
    (hx : isFinite x = true) (hy : isFinite y = true) :
    (toModel (min x y)).toReal = min (toModel x).toReal (toModel y).toReal := by
  rw [ExecFloat.Binary.toModel_min]
  exact Model.toReal_minimum_eq_min_of_isFinite (toModel x) (toModel y) hx hy

/-- IEEE maximum agrees with real maximum on finite operands, including signed zeros. -/
theorem toReal_maximum_eq_max {x y : Value}
    (hx : isFinite x = true) (hy : isFinite y = true) :
    (toModel (max x y)).toReal = max (toModel x).toReal (toModel y).toReal := by
  rw [ExecFloat.Binary.toModel_max]
  exact Model.toReal_maximum_eq_max_of_isFinite (toModel x) (toModel y) hx hy

/-- If the encoded rounding of a dyadic `d` is finite, its real value is the nearest-even rounding
of `d` in the format. -/
theorem toReal_roundDyadic_eq_round {d : FloatLib.Numerics.Dyadic}
    (hfin : isFinite (ofModel (Model.roundDyadic format d) : Value) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    (toModel (ofModel (Model.roundDyadic format d) : Value)).toReal =
      Model.roundAt format d.toReal := by
  have hdecode : toModel (ofModel (Model.roundDyadic format d) : Value) =
      Model.roundDyadic format d := ExecFloat.Binary.toModel_ofModel _
  change Model.isFinite (toModel (ofModel (Model.roundDyadic format d) : Value)) = true at hfin
  rw [hdecode] at hfin ⊢
  exact Model.toReal_roundDyadic_eq_roundAt format hformat d hfin

/-! ## Rounded arithmetic

These theorems hold for any capability whose specification is FloatLib's configured operation, as
FloatLib's own rounding theorems do. Instance search proves that for every capability the library
declares.
-/

section Rounding

variable [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]

/-- A finite executable quotient is one rounding of the exact real quotient, including a finite
value over an infinity, whose signed-zero result agrees with the totalized real quotient. -/
theorem toReal_div_eq_round {x y : Value}
    [cap : ExecFloat.Div (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.div]
    (hfin : isFinite (ExecFloat.div x y) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    (toModel (ExecFloat.div x y)).toReal =
      Model.roundAt format ((toModel x).toReal / (toModel y).toReal) := by
  have hdecode : toModel (ExecFloat.div x y) = Model.div (toModel x) (toModel y) :=
    ExecFloat.Binary.toModel_div x y .nearest
  have hmodel : Model.isFinite (Model.div (toModel x) (toModel y)) = true :=
    (congrArg Model.isFinite hdecode).symm.trans hfin
  rw [hdecode]
  exact model_div_eq_roundAt (toModel x) (toModel y) hmodel hformat

/-- A finite executable square root is one rounding of the exact real square root. Finiteness of
the result excludes negative nonzero operands. -/
theorem toReal_sqrt_eq_round {x : Value}
    [cap : ExecFloat.Sqrt (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.sqrt]
    (hfin : isFinite ((ExecFloat.Binary.sqrt (mode := .nearest)) x) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    (toModel ((ExecFloat.Binary.sqrt (mode := .nearest)) x)).toReal =
      Model.roundAt format (Real.sqrt ((toModel x).toReal)) := by
  have hmodel : Model.isFinite (Model.sqrt (toModel x)) = true :=
    (congrArg Model.isFinite (ExecFloat.Binary.toModel_sqrt x .nearest)).symm.trans hfin
  rw [Model.Proof.sqrt_eq_spec] at hmodel
  cases hchoose : Model.chooseNaN1 (toModel x) with
  | some n =>
      have hn := Model.isFinite_eq_false_of_isNaN (Model.isNaN_of_chooseNaN1_eq_some hchoose)
      simp only [Model.Spec.sqrt, hchoose] at hmodel
      have hnfinite : Model.isFinite (fmt := format) n = true := hmodel
      have hcontra : false = true := hn.symm.trans hnfinite
      cases hcontra
  | none =>
      have hxInf : Model.isInf (toModel x) = false := by
        cases hi : Model.isInf (toModel x) with
        | false => rfl
        | true =>
            simp only [Model.Spec.sqrt, hchoose, hi, ↓reduceIte] at hmodel
            split at hmodel
            · have hbad : Model.isFinite (Model.invalidResult format) = true := hmodel
              exact hbad.symm.trans (Model.isFinite_invalidResult hformat)
            · have hbad : Model.isFinite (Model.nativeOverflow format false) = true := hmodel
              exact hbad.symm.trans (Model.isFinite_nativeOverflow hformat false)
      have hxfinite := Model.isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false
        (toModel x) ((Model.chooseNaN1_eq_none_iff (toModel x)).mp hchoose) hxInf
      have hdomain : Model.isZero (toModel x) = true ∨ Model.signBit (toModel x) = false := by
        by_cases hzero : Model.isZero (toModel x) = true
        · exact Or.inl hzero
        · right
          cases hsign : Model.signBit (toModel x) with
          | false => rfl
          | true =>
              simp only [Model.Spec.sqrt, hchoose, hxInf, hzero, hsign, ↓reduceIte,
                Bool.false_eq_true] at hmodel
              have hbad : Model.isFinite (Model.invalidResult format) = true := hmodel
              exact hbad.symm.trans (Model.isFinite_invalidResult hformat)
      rw [ExecFloat.Binary.toModel_sqrt x .nearest]
      exact Model.toReal_sqrt_eq_roundAt (fmt := format) (toModel x) hformat hxfinite hdomain

/-- Whether adding the second value leaves the first encoded value unchanged, including its zero
sign. `round_add_eq_left_of_absorbs` turns this into a statement about real rounding when the sum
is finite. -/
def absorbs [ExecFloat.Add (Configured.Family format code plan)] (a b : Value) : Bool :=
  decide (ExecFloat.add a b = a)

/-- An observed absorption agrees with the rounded-real addition. Only finiteness of the sum is
required: a finite executable sum forces finite operands. -/
theorem round_add_eq_left_of_absorbs {a b : Value}
    [cap : ExecFloat.Add (Configured.Family format code plan)]
    [ExecFloat.Capability.HasSpec cap Configured.Spec.add]
    (hfin : isFinite (ExecFloat.add a b) = true) (habs : absorbs a b = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    Model.roundAt format ((toModel a).toReal + (toModel b).toReal) = (toModel a).toReal := by
  have h : (toModel (ExecFloat.add a b)).toReal =
      Model.roundAt format ((toModel a).toReal + (toModel b).toReal) :=
    ExecFloat.Binary.toReal_add_eq_roundAt hfin hformat
  have heq : ExecFloat.add a b = a := of_decide_eq_true habs
  rw [heq] at h
  exact h.symm

end Rounding

end TorchLean.IEEE754
