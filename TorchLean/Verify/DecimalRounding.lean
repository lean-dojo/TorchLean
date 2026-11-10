/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ConfiguredBinary

public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Text.Roundtrip
public import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Rational.Conversion
public import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Directed.Runtime
public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Numerics.Exact.DecimalText.Runtime
public import TorchLean.Tensor.Conversion

/-!
# Rounding exact decimals into binary formats

A JSON number is an exact decimal, FloatLib's `Decimal`. Reading it as a `Float` and then casting
to binary32 rounds twice, and the first rounding is not directed. FloatLib's
`Model.convertDecimalText` rounds the exact decimal once, in the requested direction, into a binary
format, as the convertFromDecimalCharacter conversion of IEEE 754-2019 section 5.12 does, with the
overflow result of each direction and the sign of a zero decimal. `float` is its host binary64
form. C23 binds the conversion to `strtod` (F.3 of the draft WG14 N3096).

The conversion clamps an exponent far outside the range of the format before it forms a power of
ten, so the token `1e-999999999` does not build a power of ten with a billion digits.
`convertDecimalText_value_eq_roundRat` shows through `Model.convertDecimalText_eq_exact` that the
clamp changes no result: the value is the directed rounding of the exact rational. In an IEEE
format, `toEReal_convertDecimalText_down_le` and `le_toEReal_convertDecimalText_up` bound the
decimal from below and from above, and FloatLib's `Model.toReal_convertDecimalText_nearest` gives
the nearest rounding of a finite result.
`TorchLean.Test.CertificatePreconditions.signedDecimals` checks binary32 in all six modes: `1e39`
and `1e999999999` give `0x7F7FFFFF` down and toward zero and `+∞` in the other four, and `1e-50`
gives the smallest subnormal up and away and `+0` in the rest.

Certificate parsers use `.down` for lower endpoints and `.up` for upper endpoints, so the parsed
box contains the decimal box that the artifact states. Transcript entries that must equal a
replayed value use `.nearest`.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open FloatLib.Floats.Formats.BinaryInterchange.Model (IEEERoundingMode)
open FloatLib.Numerics.DecimalText (Decimal)

namespace TorchLean.DecimalRounding

/-- The value of `Model.convertDecimalText` is the directed rounding of the exact rational of `d`
with the sign of `d`, so the exponent clamp of the conversion changes no result. -/
theorem convertDecimalText_value_eq_roundRat (fmt : FloatFormat) (mode : IEEERoundingMode)
    (d : Decimal) :
    (Model.convertDecimalText fmt mode d).value =
      Model.Internal.Directed.roundRat fmt mode d.negative
        ((d.significand : Rat) * (10 : Rat) ^ d.exponent).num.natAbs
        ((d.significand : Rat) * (10 : Rat) ^ d.exponent).den := by
  rw [Model.convertDecimalText_eq_exact, Model.convertDecimalTextExact,
    Model.rationalRoundingOutcomeScaled_value]

/-- Rounding down gives at most the value of `d`, compared as extended reals. -/
theorem toEReal_convertDecimalText_down_le (fmt : FloatFormat) (hfmt : fmt.isIEEE = true)
    (d : Decimal) :
    Model.toEReal (Model.convertDecimalText fmt .down d).value ≤ ((d.toRat : ℝ) : EReal) := by
  have h := Model.signedScaledRatToReal_decimal d
  dsimp only at h
  rw [convertDecimalText_value_eq_roundRat, ← h]
  exact Model.toEReal_roundRatDown_le fmt _ _ _ hfmt (Rat.den_nz _)

/-- Rounding up gives at least the value of `d`, compared as extended reals. -/
theorem le_toEReal_convertDecimalText_up (fmt : FloatFormat) (hfmt : fmt.isIEEE = true)
    (d : Decimal) :
    ((d.toRat : ℝ) : EReal) ≤ Model.toEReal (Model.convertDecimalText fmt .up d).value := by
  have h := Model.signedScaledRatToReal_decimal d
  dsimp only at h
  rw [convertDecimalText_value_eq_roundRat, ← h]
  exact Model.le_toEReal_roundRatUp fmt _ _ _ hfmt (Rat.den_nz _)

/-- Round the exact decimal value of `d` once into binary64 under `mode`, as a host `Float`. -/
def float (mode : IEEERoundingMode) (d : Decimal) : Float :=
  ExecFloat.Binary.toFloat (ExecFloat.Binary.ofModel
    (Model.convertDecimalText FloatFormat.binary64 mode d).value : FloatLib.Floats.Binary64)

end TorchLean.DecimalRounding
