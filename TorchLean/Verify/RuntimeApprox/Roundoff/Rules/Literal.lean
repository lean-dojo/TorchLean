/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Algebraic.Rounding.SqrtProof
public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.MixedError
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# Literal constants for `roundoff`

When `1e-6` rounds to zero in a configured binary format, `Context.defaultEpsilon` is the encoding
word `1`, the least positive subnormal. `roundoff` reads its value as one over a natural power of
two, which the analysis treats as exact natural-number data, and takes its finiteness from
FloatLib's `isFinite_ofNatBits_one`. Whether or not `1e-6` rounds to zero, `isFinite_defaultEpsilon`
in `TorchLean.Verify.RuntimeApprox.Roundoff.Mixed` proves that `Context.defaultEpsilon` is finite.

When the cast of a count overflows, the configured `Context.sqrtNat` rounds the rational square
root of the count (`Model.AlgebraicRounding.sqrtRat`) without casting it, and the attention scale
`sqrt headDim` can take this branch. FloatLib proves that the result rounds the real square root
once (`Model.AlgebraicRounding.toReal_sqrtRat_eq_roundAt`). From that, `abs_toReal_sqrtRat_sub_le`
and `sqrtRat_rounding` give the absolute and mixed bounds of one rounding.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

/-- The least positive subnormal of a configured format is one over the natural number
`2 ^ (exponentBias + fracWidth - 1)`. -/
theorem toReal_ofNatBits_one :
    ExecFloat.Binary.toReal
        (ExecFloat.Binary.ofNatBits 1 : ExecFloat (Configured.Family format code plan)) =
      1 / ((2 ^ (format.exponentBias + format.fracWidth - 1) : ℕ) : ℝ) := by
  have exponent : format.minSubnormalExponent =
      -((format.exponentBias + format.fracWidth - 1 : ℕ) : ℤ) := by
    have := format.exponentBias_pos
    simp only [FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
      Int.ofNat_eq_natCast]
    omega
  rw [ExecFloat.Binary.toReal_ofNatBits_one, exponent, zpow_neg, zpow_natCast, Nat.cast_pow,
    Nat.cast_ofNat, one_div]

attribute [roundoff_rule] toReal_ofNatBits_one ExecFloat.Binary.isFinite_ofNatBits_one

/-- In an IEEE format, a finite `sqrtRat` of a nonnegative radicand is within half an ulp of the
real square root, because it rounds that root once. -/
theorem abs_toReal_sqrtRat_sub_le (format : FloatFormat) (hformat : format.isIEEE = true)
    (radicand : Rat) (hradicand : 0 ≤ radicand)
    (hfinite : Model.isFinite (Model.AlgebraicRounding.sqrtRat format false radicand) = true) :
    |Model.toReal (Model.AlgebraicRounding.sqrtRat format false radicand) -
        Real.sqrt (radicand : ℝ)| ≤ Model.halfUlpAt format (Real.sqrt (radicand : ℝ)) := by
  have h := Model.AlgebraicRounding.toReal_sqrtRat_eq_roundAt format hformat false radicand
    hradicand hfinite
  simp only [Bool.false_eq_true, ↓reduceIte] at h
  rw [h]
  exact Model.abs_roundAt_sub_le format _

/-- In an IEEE format, a finite `sqrtRat` of a nonnegative radicand approximates the real square
root with the relative and absolute error of one rounding. -/
theorem sqrtRat_rounding (format : FloatFormat) (hformat : format.isIEEE = true)
    (radicand : Rat) (hradicand : 0 ≤ radicand)
    (hfinite : Model.isFinite (Model.AlgebraicRounding.sqrtRat format false radicand) = true) :
    MixedError (Model.toReal (Model.AlgebraicRounding.sqrtRat format false radicand))
      (Real.sqrt (radicand : ℝ)) (Model.unitRoundoffAt format) (Model.underflowErrorAt format) := by
  have h := Model.AlgebraicRounding.toReal_sqrtRat_eq_roundAt format hformat false radicand
    hradicand hfinite
  simp only [Bool.false_eq_true, ↓reduceIte] at h
  exact Model.mixedError_toReal_of_eq_roundAt h

attribute [roundoff_rule] abs_toReal_sqrtRat_sub_le sqrtRat_rounding

end TorchLean.Roundoff
