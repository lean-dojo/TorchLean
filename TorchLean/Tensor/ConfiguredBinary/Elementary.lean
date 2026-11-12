/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ConfiguredBinary.InverseTrigonometric
public import TorchLean.Tensor.ConfiguredBinary.RoundSigned
public import FloatLib.Numerics.Exact.Hyperbolic.Runtime
public import FloatLib.Numerics.Exact.RationalPower.Runtime

/-!
# Correctly rounded elementary functions of binary values

`Elementary.tan`, `exp2`, `logb`, `asinh`, `acosh` and `atanh` round the real function once, to
nearest even. FloatLib's exact comparisons of the tangent, of the inverse hyperbolic functions
(`Numerics.HyperbolicComparison`) and of real powers of rationals (`Numerics.RationalPower`)
with rational boundaries decide the rounding through `Model.AlgebraicRounding.round`, as
`InverseTrigonometric` does. No floating-point approximation enters, so an exact result is
returned exactly: `exp2 3 = 8`, `logb 10 1000 = 3`, `acosh 1 = +0`. The tangent's period
reduction uses an enclosure of π as precise as the argument's magnitude requires, so its cost
grows with that magnitude, to minutes for binary64 arguments near `1e300`.

The special values are those of C99 Annex F. A NaN is quieted. `tan`, `asinh` and `atanh` keep the
sign of a zero, `asinh (±∞) = ±∞`, `acosh (+∞) = exp2 (+∞) = logb b (+∞) = +∞` and `exp2 (-∞) = +0`.
`atanh (±1)` is the overflow of the sign of its input and `logb b (±0)` the negative overflow, `±∞`
in an IEEE format. `tan (±∞)`, `acosh` below `1`, `atanh` outside `[-1, 1]` and `logb` below zero
give the invalid result, at the infinities outside each domain too. We took each case by hand from
WG14 N1256, the C99 committee draft: F.9.1.7 for `tan`, F.9.2.1 to F.9.2.3 for `acosh`, `asinh` and
`atanh`, F.9.3.2 for `exp2`, and C's `log10` and `log2` (F.9.3.8 and F.9.3.10) for `logb`. C's own
`logb` (7.12.6.11) extracts the exponent, and its `logb (-∞)` is `+∞` (F.9.3.11). A quiet NaN comes
back unchanged, the recommended practice of F.9, paragraph 13.

`TorchLean.Verify.RuntimeApprox.Elementary` proves the rounding in an IEEE format, and
`TorchLean.Test.Elementary.checkElementary` tests binary16, binary32 and binary64 words against
400-digit mpmath: binary16 `tan 355` is the subnormal `0x1fa`.
`ConfiguredBinary.tan`, `exp2`, `logb`, `asinh`, `acosh` and `atanh` are the same functions on
configured binary values, and the `Context` of every configured format uses them.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Model
open FloatLib.Numerics

namespace Elementary

variable {fmt : FloatFormat}

open InverseTrigonometric (levels)

/-- The tangent rounded once to nearest even, with a zero returned unchanged. -/
def tan (x : Model fmt) : Model fmt :=
  if isNaN x then quietNaN x
  else if isZero x then x
  else match toRat? x with
    | some a => roundSigned fmt (TrigonometricComparison.prepareTan a levels).compare
    | none => invalidResult fmt

/-- `2 ^ x` rounded once to nearest even, with `exp2 (-∞) = +0` and `exp2 (+∞) = +∞`. -/
def exp2 (x : Model fmt) : Model fmt :=
  if isNaN x then quietNaN x
  else match toRat? x with
    | some a => AlgebraicRounding.round fmt false (RationalPower.compare 2 a)
    | none => if signBit x then zero fmt false else x

/-- The logarithm to a base `base` above one, rounded once to nearest even: the magnitude
`logb base (max a a⁻¹)` of the result is compared through `base ^ b`. -/
def logb (base : ℚ) (x : Model fmt) : Model fmt :=
  if isNaN x then quietNaN x
  else if isZero x then nativeOverflow fmt true
  else match toRat? x with
    | some a =>
        if 0 < a then
          AlgebraicRounding.round fmt (decide (a < 1)) fun b =>
            (RationalPower.compare base b (max a a⁻¹)).swap
        else invalidResult fmt
    | none => if signBit x then invalidResult fmt else x

/-- The inverse hyperbolic sine rounded once to nearest even, with `asinh (±∞) = ±∞`. -/
def asinh (x : Model fmt) : Model fmt :=
  if isNaN x then quietNaN x
  else match toRat? x with
    | some a => AlgebraicRounding.round fmt (signBit x)
        (HyperbolicComparison.prepareArsinh |a| levels).compare
    | none => x

/-- The inverse hyperbolic cosine rounded once to nearest even, with `acosh (+∞) = +∞` and the
invalid result below `1`. -/
def acosh (x : Model fmt) : Model fmt :=
  if isNaN x then quietNaN x
  else match toRat? x with
    | some a =>
        if a < 1 then invalidResult fmt
        else AlgebraicRounding.round fmt false (HyperbolicComparison.prepareArcosh a levels).compare
    | none => if signBit x then invalidResult fmt else x

/-- The inverse hyperbolic tangent rounded once to nearest even, with the overflow of the input's
sign at `±1` and the invalid result outside `[-1, 1]`. -/
def atanh (x : Model fmt) : Model fmt :=
  if isNaN x then quietNaN x
  else match toRat? x with
    | some a =>
        if h : |a| < 1 then
          AlgebraicRounding.round fmt (signBit x)
            (HyperbolicComparison.prepareArtanh |a| levels (by linarith [abs_nonneg a]) h).compare
        else if |a| = 1 then nativeOverflow fmt (signBit x)
        else invalidResult fmt
    | none => invalidResult fmt

end Elementary

namespace ConfiguredBinary

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The tangent rounded once to nearest even (`Elementary.tan`). -/
def tan (x : Value) : Value :=
  ExecFloat.ModelCodec.liftUnary (Model := Model format) (plan := plan) Elementary.tan x

/-- `2 ^ x` rounded once to nearest even (`Elementary.exp2`). -/
def exp2 (x : Value) : Value :=
  ExecFloat.ModelCodec.liftUnary (Model := Model format) (plan := plan) Elementary.exp2 x

/-- The logarithm to a base above one, rounded once to nearest even (`Elementary.logb`). -/
def logb (base : ℚ) (x : Value) : Value :=
  ExecFloat.ModelCodec.liftUnary (Model := Model format) (plan := plan) (Elementary.logb base) x

/-- The inverse hyperbolic sine rounded once to nearest even (`Elementary.asinh`). -/
def asinh (x : Value) : Value :=
  ExecFloat.ModelCodec.liftUnary (Model := Model format) (plan := plan) Elementary.asinh x

/-- The inverse hyperbolic cosine rounded once to nearest even (`Elementary.acosh`). -/
def acosh (x : Value) : Value :=
  ExecFloat.ModelCodec.liftUnary (Model := Model format) (plan := plan) Elementary.acosh x

/-- The inverse hyperbolic tangent rounded once to nearest even (`Elementary.atanh`). -/
def atanh (x : Value) : Value :=
  ExecFloat.ModelCodec.liftUnary (Model := Model format) (plan := plan) Elementary.atanh x

end ConfiguredBinary

end TorchLean
