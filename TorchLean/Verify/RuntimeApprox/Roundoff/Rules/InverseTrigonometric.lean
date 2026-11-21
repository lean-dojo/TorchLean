/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError.Rounding
public import TorchLean.Verify.RuntimeApprox.InverseTrigonometric
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Tactic.Roundoff.Extension

/-!
# Inverse trigonometric functions for `roundoff`

In an IEEE format, the configured `atan` of a finite value, and `asin` and `acos` of a finite
value in `[-1, 1]`, round `Real.arctan`, `Real.arcsin` and `Real.arccos` of its value once
(`ConfiguredBinary.toReal_atan`, `toReal_asin`, `toReal_acos`), and the result is finite
(`InverseTrigonometric.isFinite_atan` and the others), as each real result is at most `π` in
magnitude. The rules here state this for an exact finite operand of value `a`: within half an ulp
of the real function at `a`, and the mixed error of one rounding. They are registered on the
`InverseTrig` methods, so that the analysis reads them before it unfolds the configured functions
to their special cases.

As for `atanh` (`TorchLean.Roundoff.mixedError_toReal_atanh`), the absolute bounds of `asin` and
`acos` take the domain `|a| ≤ 1`, and the mixed bounds take a finite result instead, which puts
the operand in the domain (`ConfiguredBinary.abs_toReal_le_one_of_isFinite_asin`), because mixed
analysis keeps a finiteness premise open but not an order premise. The finiteness rules turn that
premise into the condition `|x| ≤ 1` on the operand.

`atan2`'s value at a zero `y` depends on the sign of that zero (`±π` against a negative `x`), which
the real operands do not carry, and at the origin it is `±0` or `±π` where the real polar angle is
zero. So its bounds take a nonzero `y` or a positive `x` (`toReal_atan2_eq_roundAt`), which the
analysis keeps as a condition on the operands.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open ExecFloat.Binary (toModel toReal isFinite)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- In an IEEE format, `atan` of a finite configured value is finite. -/
theorem isFinite_atan {x : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true) :
    isFinite (InverseTrig.atan x) = true := by
  rw [show InverseTrig.atan x = ConfiguredBinary.atan x from rfl]
  unfold ExecFloat.Binary.isFinite at *
  simp only [ConfiguredBinary.atan, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact InverseTrigonometric.isFinite_atan hformat hx

/-- `atan` of an exact finite operand is rounded once: an absolute bound. -/
theorem abs_toReal_atan_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0) (hxf : isFinite x = true) :
    |toReal (InverseTrig.atan x) - Real.arctan a| ≤ Model.halfUlpAt format (Real.arctan a) := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  rw [show InverseTrig.atan x = ConfiguredBinary.atan x from rfl,
    ConfiguredBinary.toReal_atan hxf hformat, ha]
  exact Model.abs_roundAt_sub_le _ _

/-- `atan` of an exact finite operand is rounded once: a mixed bound. -/
theorem mixedError_toReal_atan {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0) (hxf : isFinite x = true) :
    MixedError (toReal (InverseTrig.atan x)) (Real.arctan a) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  rw [show InverseTrig.atan x = ConfiguredBinary.atan x from rfl]
  exact ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| by
    rw [ConfiguredBinary.toReal_atan hxf hformat, ha]

/-- In an IEEE format, `asin` of a finite configured value in `[-1, 1]` is finite. -/
theorem isFinite_asin {x : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true)
    (hdomain : |toReal x| ≤ 1) : isFinite (InverseTrig.asin x) = true := by
  rw [show InverseTrig.asin x = ConfiguredBinary.asin x from rfl]
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [ConfiguredBinary.asin, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact InverseTrigonometric.isFinite_asin hformat hx hdomain

/-- `asin` of an exact finite operand in `[-1, 1]` is rounded once: an absolute bound. -/
theorem abs_toReal_asin_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0) (hxf : isFinite x = true) (hdomain : |a| ≤ 1) :
    |toReal (InverseTrig.asin x) - Real.arcsin a| ≤ Model.halfUlpAt format (Real.arcsin a) := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  rw [show InverseTrig.asin x = ConfiguredBinary.asin x from rfl,
    ConfiguredBinary.toReal_asin hxf (ha ▸ hdomain) hformat, ha]
  exact Model.abs_roundAt_sub_le _ _

/-- A finite `asin` of an exact finite operand is rounded once: a mixed bound. The finite result
puts the operand in `[-1, 1]`. -/
theorem mixedError_toReal_asin {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0) (hxf : isFinite x = true)
    (hfinite : isFinite (InverseTrig.asin x) = true) :
    MixedError (toReal (InverseTrig.asin x)) (Real.arcsin a) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  rw [show InverseTrig.asin x = ConfiguredBinary.asin x from rfl] at hfinite ⊢
  exact ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| by
    rw [ConfiguredBinary.toReal_asin hxf
      (ConfiguredBinary.abs_toReal_le_one_of_isFinite_asin hxf hfinite hformat) hformat, ha]

/-- In an IEEE format, `acos` of a finite configured value in `[-1, 1]` is finite. -/
theorem isFinite_acos {x : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true)
    (hdomain : |toReal x| ≤ 1) : isFinite (InverseTrig.acos x) = true := by
  rw [show InverseTrig.acos x = ConfiguredBinary.acos x from rfl]
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [ConfiguredBinary.acos, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftUnary] at *
  exact InverseTrigonometric.isFinite_acos hformat hx hdomain

/-- `acos` of an exact finite operand in `[-1, 1]` is rounded once: an absolute bound. -/
theorem abs_toReal_acos_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0) (hxf : isFinite x = true) (hdomain : |a| ≤ 1) :
    |toReal (InverseTrig.acos x) - Real.arccos a| ≤ Model.halfUlpAt format (Real.arccos a) := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  rw [show InverseTrig.acos x = ConfiguredBinary.acos x from rfl,
    ConfiguredBinary.toReal_acos hxf (ha ▸ hdomain) hformat, ha]
  exact Model.abs_roundAt_sub_le _ _

/-- A finite `acos` of an exact finite operand is rounded once: a mixed bound. The finite result
puts the operand in `[-1, 1]`. -/
theorem mixedError_toReal_acos {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0) (hxf : isFinite x = true)
    (hfinite : isFinite (InverseTrig.acos x) = true) :
    MixedError (toReal (InverseTrig.acos x)) (Real.arccos a) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  rw [show InverseTrig.acos x = ConfiguredBinary.acos x from rfl] at hfinite ⊢
  exact ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| by
    rw [ConfiguredBinary.toReal_acos hxf
      (ConfiguredBinary.abs_toReal_le_one_of_isFinite_acos hxf hfinite hformat) hformat, ha]

/-- The polar angle of `(a, |b|)`, negated under a sign that only a nonpositive `b` has and kept
under one that only a nonnegative `b` has, is the polar angle of `(a, b)` when `b` is nonzero or
`a` is positive: for a negative `b` by `Complex.arg_conj`, as `(a, b)` is then off the negative
real axis, and for a zero `b` because the angle of a positive `a` is zero. -/
theorem signed_arg_eq_arg {a b : ℝ} {s : Bool} (hset : s = true → b ≤ 0)
    (hclear : s = false → 0 ≤ b) (hdomain : b ≠ 0 ∨ 0 < a) :
    (if s then -Complex.arg ⟨a, |b|⟩ else Complex.arg ⟨a, |b|⟩) = Complex.arg ⟨a, b⟩ := by
  cases s with
  | false => simp [abs_of_nonneg (hclear rfl)]
  | true =>
    simp only [ite_true]
    rcases (hset rfl).lt_or_eq with hb | hb
    · rw [abs_of_neg hb]
      have hconj : (⟨a, -b⟩ : ℂ) = (starRingEnd ℂ) ⟨a, b⟩ := Complex.ext (by simp) (by simp)
      rw [hconj, Complex.arg_conj, ite_eq_right]
      · ring
      · rw [Complex.arg_eq_pi_iff]
        simp [hb.ne]
    · subst hb
      have ha : 0 < a := hdomain.resolve_left (by simp)
      have hreal : (⟨a, 0⟩ : ℂ) = (a : ℂ) := Complex.ext (by simp) (by simp)
      simp [hreal, Complex.arg_ofReal_of_nonneg ha.le]

/-- In an IEEE format, `atan2` of finite configured values is finite. -/
theorem isFinite_atan2 {y x : Value} (hformat : format.isIEEE = true) (hy : isFinite y = true)
    (hx : isFinite x = true) : isFinite (InverseTrig.atan2 y x) = true := by
  rw [show InverseTrig.atan2 y x = ConfiguredBinary.atan2 y x from rfl]
  unfold ExecFloat.Binary.isFinite at *
  simp only [ConfiguredBinary.atan2, ExecFloat.Binary.toModel, Configured.Family.toModel,
    ExecFloat.ModelCodec.decode_liftBinary] at *
  exact InverseTrigonometric.isFinite_atan2 hformat hy hx

/-- In an IEEE format, `atan2` of finite configured values `y` and `x`, with `y` nonzero or `x`
positive, rounds the polar angle of `(x, y)` once. The sign of `y` decides the sign of the result
(`ConfiguredBinary.toReal_atan2`), and off the origin and the negative real axis it is the sign of
the angle (`signed_arg_eq_arg`). -/
theorem toReal_atan2_eq_roundAt {y x : Value} (hformat : format.isIEEE = true)
    (hy : isFinite y = true) (hx : isFinite x = true) (hdomain : toReal y ≠ 0 ∨ 0 < toReal x) :
    toReal (InverseTrig.atan2 y x) = Model.roundAt format (Complex.arg ⟨toReal x, toReal y⟩) := by
  have horigin : toReal x ≠ 0 ∨ toReal y ≠ 0 := by
    rcases hdomain with h | h
    · exact .inr h
    · exact .inl h.ne'
  have hset : ExecFloat.Binary.signBit y = true → toReal y ≤ 0 := fun h => by
    unfold ExecFloat.Binary.signBit at h
    unfold ExecFloat.Binary.toReal
    exact InverseTrigonometric.toReal_nonpos_of_signBit h
  have hclear : ExecFloat.Binary.signBit y = false → 0 ≤ toReal y := fun h => by
    unfold ExecFloat.Binary.signBit at h
    unfold ExecFloat.Binary.toReal
    exact Model.toReal_nonneg_of_signBit_eq_false h
  rw [show InverseTrig.atan2 y x = ConfiguredBinary.atan2 y x from rfl,
    ConfiguredBinary.toReal_atan2 hy hx horigin hformat, signed_arg_eq_arg hset hclear hdomain]

/-- `atan2` of exact finite operands of values `b` and `a`, with `b` nonzero or `a` positive, is
rounded once: an absolute bound. The real polar angle is `Complex.arg ⟨a, b⟩`, the `InverseTrig ℝ`
instance's `atan2 b a`. -/
theorem abs_toReal_atan2_sub_le {y x : Value} {b a : ℝ} (hformat : format.isIEEE = true)
    (hy : |toReal y - b| ≤ 0) (hyf : isFinite y = true) (hx : |toReal x - a| ≤ 0)
    (hxf : isFinite x = true) (hdomain : b ≠ 0 ∨ 0 < a) :
    |toReal (InverseTrig.atan2 y x) - Complex.arg ⟨a, b⟩| ≤
      Model.halfUlpAt format (Complex.arg ⟨a, b⟩) := by
  have hb : toReal y = b := by simpa [sub_eq_zero] using hy
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  rw [toReal_atan2_eq_roundAt hformat hyf hxf (by rw [ha, hb]; exact hdomain), ha, hb]
  exact Model.abs_roundAt_sub_le _ _

/-- `atan2` of exact finite operands of values `b` and `a`, with `b` nonzero or `a` positive, is
rounded once: a mixed bound. -/
theorem mixedError_toReal_atan2 {y x : Value} {b a : ℝ} (hformat : format.isIEEE = true)
    (hy : MixedError (toReal y) b 0 0) (hyf : isFinite y = true)
    (hx : MixedError (toReal x) a 0 0) (hxf : isFinite x = true) (hdomain : b ≠ 0 ∨ 0 < a) :
    MixedError (toReal (InverseTrig.atan2 y x)) (Complex.arg ⟨a, b⟩)
      (Model.unitRoundoffAt format) (Model.underflowErrorAt format) := by
  have hb := mixedError_zero_zero_iff.mp hy
  have ha := mixedError_zero_zero_iff.mp hx
  exact ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| by
    rw [toReal_atan2_eq_roundAt hformat hyf hxf (by rw [ha, hb]; exact hdomain), ha, hb]

attribute [roundoff_rule] isFinite_atan abs_toReal_atan_sub_le mixedError_toReal_atan
  isFinite_asin abs_toReal_asin_sub_le mixedError_toReal_asin
  isFinite_acos abs_toReal_acos_sub_le mixedError_toReal_acos
  isFinite_atan2 abs_toReal_atan2_sub_le mixedError_toReal_atan2

end TorchLean.Roundoff
