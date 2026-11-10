/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals
public import FloatLib.Floats.Formats.BinaryInterchange.Operations.PowerProof
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Verify.RuntimeApprox.Roundoff.Recovery
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# Powers for `roundoff`

A configured scalar's power `x ^ y` is FloatLib's `Model.pow`. In an IEEE format it rounds the real
power once in two cases: finite operands with an integral exponent and a finite result
(`Model.Power.toReal_pow_of_eq_intCast`), and any operands for which the certified enclosure search
settles (`Model.Power.pow_spec_of_certified`). Both are stated as the rounding of the real power
`b ^ e` of operands whose values are exactly `b` and `e`, absolutely and as a mixed error, and
registered as `roundoff` rules. The same rounding at the values of computed operands composes with
FloatLib's propagation of their errors through the real power (`rpow_error`, `rpow_const_error` and
their mixed forms). The pullback of a constant exponent recovers its slope from the exact operands
when an intermediate loses range, and that recovery rounds the slope `e * x ^ (e - 1) * g` once
(`abs_enclosedPowerSlope_sub_le`).
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open ExecFloat.Binary (toModel toReal isFinite)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" =>
  ExecFloat.Binary format.expWidth format.fracWidth format.encoding format.exponentBias
    format.expWidth_ge_two format.fracWidth_pos format.exponentBias_pos
    format.exponentBias_le_maxFinite plan code

/-- In an IEEE format, a finite power of finite operands of values `b` and `e`, with `e` an integer,
rounds the real power `b ^ e` once. -/
theorem toReal_pow_of_integral {base exponent : Value} {b e : ℝ}
    (hformat : format.isIEEE = true) (hbase : toReal base = b) (hexponent : toReal exponent = e)
    (hbaseFinite : isFinite base = true) (hexponentFinite : isFinite exponent = true)
    (hintegral : ∃ n : ℤ, e = n) (hresult : isFinite (base ^ exponent : Value) = true) :
    toReal (base ^ exponent : Value) = Model.roundAt format (b ^ e) := by
  obtain ⟨n, rfl⟩ := hintegral
  unfold toReal isFinite at *
  rw [ConfiguredBinary.toModel_hPow] at hresult ⊢
  rw [← hbase, Real.rpow_intCast]
  exact Model.Power.toReal_pow_of_eq_intCast hformat hbaseFinite hexponentFinite hexponent hresult

/-- In an IEEE format, a finite power of finite operands of values `b` and `e` rounds the real power
`b ^ e` once when `e` is an integer or the certified search settles. -/
theorem toReal_pow_of_settled {base exponent : Value} {b e : ℝ}
    (hformat : format.isIEEE = true) (hbase : toReal base = b) (hexponent : toReal exponent = e)
    (hbaseFinite : isFinite base = true) (hexponentFinite : isFinite exponent = true)
    (hsettled : (∃ n : ℤ, e = n) ∨ (Model.Transcendentals.Certified.pow (toModel base)
      (toModel exponent) Model.Power.certifiedBudget).isOk = true)
    (hresult : isFinite (base ^ exponent : Value) = true) :
    toReal (base ^ exponent : Value) = Model.roundAt format (b ^ e) := by
  subst hbase hexponent
  by_cases hintegral : ∃ n : ℤ, toReal exponent = n
  · exact toReal_pow_of_integral hformat rfl rfl hbaseFinite hexponentFinite hintegral hresult
  · unfold toReal
    rw [ConfiguredBinary.toModel_hPow]
    exact (Model.Power.pow_spec_of_certified (not_exists.mp hintegral)
      (hsettled.resolve_left hintegral)).2

/-- The absolute form of `toReal_pow_of_settled`, for operands of exact values `b` and `e`: the
power is within half an ulp of `b ^ e`. -/
theorem abs_toReal_pow_sub_le {base exponent : Value} {b e : ℝ}
    (hformat : format.isIEEE = true) (hbase : |toReal base - b| ≤ 0)
    (hexponent : |toReal exponent - e| ≤ 0)
    (hbaseFinite : isFinite base = true) (hexponentFinite : isFinite exponent = true)
    (hsettled : (∃ n : ℤ, e = n) ∨ (Model.Transcendentals.Certified.pow (toModel base)
      (toModel exponent) Model.Power.certifiedBudget).isOk = true)
    (hresult : isFinite (base ^ exponent : Value) = true) :
    |toReal (base ^ exponent : Value) - b ^ e| ≤ Model.halfUlpAt format (b ^ e) := by
  rw [toReal_pow_of_settled hformat (by simpa [sub_eq_zero] using hbase)
    (by simpa [sub_eq_zero] using hexponent) hbaseFinite hexponentFinite hsettled hresult]
  exact Model.abs_roundAt_sub_le _ _

/-- The mixed form of `toReal_pow_of_settled`, for operands of exact values `b` and `e`. -/
theorem mixedError_toReal_pow {base exponent : Value} {b e : ℝ}
    (hformat : format.isIEEE = true) (hbase : MixedError (toReal base) b 0 0)
    (hexponent : MixedError (toReal exponent) e 0 0)
    (hbaseFinite : isFinite base = true) (hexponentFinite : isFinite exponent = true)
    (hsettled : (∃ n : ℤ, e = n) ∨ (Model.Transcendentals.Certified.pow (toModel base)
      (toModel exponent) Model.Power.certifiedBudget).isOk = true)
    (hresult : isFinite (base ^ exponent : Value) = true) :
    MixedError (toReal (base ^ exponent : Value)) (b ^ e) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) :=
  ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| toReal_pow_of_settled hformat
    (mixedError_zero_zero_iff.mp hbase) (mixedError_zero_zero_iff.mp hexponent) hbaseFinite
    hexponentFinite hsettled hresult

/-- In an IEEE format, a finite power of finite operands is within half an ulp of the real power of
their values when the exponent is integral or the certified search settles. -/
theorem abs_toReal_pow_sub_rpow_le {base exponent : Value} (hformat : format.isIEEE = true)
    (hbaseFinite : isFinite base = true) (hexponentFinite : isFinite exponent = true)
    (hsettled : (∃ n : ℤ, toReal exponent = n) ∨ (Model.Transcendentals.Certified.pow
      (toModel base) (toModel exponent) Model.Power.certifiedBudget).isOk = true)
    (hresult : isFinite (base ^ exponent : Value) = true) :
    |toReal (base ^ exponent : Value) - toReal base ^ toReal exponent| ≤
      Model.halfUlpAt format (toReal base ^ toReal exponent) := by
  rw [toReal_pow_of_settled hformat rfl rfl hbaseFinite hexponentFinite hsettled hresult]
  exact Model.abs_roundAt_sub_le _ _

/-- The mixed form of `abs_toReal_pow_sub_rpow_le`. -/
theorem mixedError_toReal_pow_rpow {base exponent : Value} (hformat : format.isIEEE = true)
    (hbaseFinite : isFinite base = true) (hexponentFinite : isFinite exponent = true)
    (hsettled : (∃ n : ℤ, toReal exponent = n) ∨ (Model.Transcendentals.Certified.pow
      (toModel base) (toModel exponent) Model.Power.certifiedBudget).isOk = true)
    (hresult : isFinite (base ^ exponent : Value) = true) :
    MixedError (toReal (base ^ exponent : Value)) (toReal base ^ toReal exponent)
      (Model.unitRoundoffAt format) (Model.underflowErrorAt format) :=
  ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| toReal_pow_of_settled hformat rfl rfl
    hbaseFinite hexponentFinite hsettled hresult

attribute [roundoff_rule] abs_toReal_pow_sub_le mixedError_toReal_pow abs_toReal_pow_sub_rpow_le
  mixedError_toReal_pow_rpow abs_enclosedPowerSlope_sub_le
  mixedError_enclosedPowerSlope

end TorchLean.Roundoff
