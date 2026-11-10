/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Numeric.InverseTrig
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Algebraic.Rounding.Runtime
public import FloatLib.Numerics.Exact.Trigonometric.Runtime
public import FloatLib.Numerics.Exact.Trigonometric.Atan2.Runtime

/-!
# Correctly rounded inverse trigonometric functions of binary values

`InverseTrigonometric.atan`, `asin`, `acos` and `atan2` round the real function once, to nearest
even, in an IEEE descriptor. FloatLib's exact comparisons of the arctangent, the inverse sine and
cosine and the principal argument with rational boundaries (`Numerics.TrigonometricComparison`)
decide the rounding through `Model.AlgebraicRounding.round`, as FloatLib's `Model.pi` does, and no
floating-point approximation enters. The special values are those of C99 Annex F, which PyTorch's
CPU kernels return. A NaN is quieted, `atan (±∞)` is `±π/2` rounded, `asin` and `acos` outside
`[-1, 1]` give the invalid result, and `asin` and `atan` keep the sign of a zero. `atan2` takes its
sign from `y` and follows the Annex F table for zeros and infinities, as in `atan2 (±0, -0) = ±π`
and `atan2 (±∞, +∞) = ±π/4`. We copied the cases by hand from F.9.1.1 to F.9.1.4 of WG14 N1256.
`TorchLean.Test.Pointwise.runBinary` checks the binary32 words against eager PyTorch on CPU, at 12
inputs each for `atan`, `asin` and `acos` and 22 pairs for `atan2`.

`TorchLean.Verify.RuntimeApprox.InverseTrigonometric` proves the rounding and the finiteness of the
results at finite inputs in the domain of an IEEE format, and the special values.
`ConfiguredBinary.atan`, `asin`, `acos` and `atan2` are the same functions on configured binary
values and form the `InverseTrig` instance of every configured format, so `Op.atan`, `Op.asin`,
`Op.acos` and `Op.atan2` evaluate them.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Model
open FloatLib.Numerics.TrigonometricComparison

namespace InverseTrigonometric

variable {fmt : FloatFormat}

/-- The enclosure depths each comparator caches before refining past them. `Elementary.tan` of a
huge argument refines its period reduction this deep at every comparison. -/
def levels : Nat := 16

/-- The arctangent rounded once to nearest even, with `atan (±∞)` equal to `±π/2` rounded. -/
def atan (x : Model fmt) : Model fmt :=
  if isNaN x then quietNaN x
  else if isInf x then AlgebraicRounding.round fmt (signBit x) (prepareAtan2 0 1 levels).compare
  else match toRat? x with
    | some a => AlgebraicRounding.round fmt (signBit x) (prepareArctan |a| levels).compare
    | none => invalidResult fmt

/-- The inverse sine rounded once to nearest even, with the invalid result outside `[-1, 1]`. -/
def asin (x : Model fmt) : Model fmt :=
  if isNaN x then quietNaN x
  else match toRat? x with
    | some a =>
        if 1 < |a| then invalidResult fmt
        else AlgebraicRounding.round fmt (signBit x) (prepareArcsin |a| levels).compare
    | none => invalidResult fmt

/-- The inverse cosine rounded once to nearest even, with the invalid result outside `[-1, 1]`. -/
def acos (x : Model fmt) : Model fmt :=
  if isNaN x then quietNaN x
  else match toRat? x with
    | some a =>
        if 1 < |a| then invalidResult fmt
        else AlgebraicRounding.round fmt false (prepareArccos a levels).compare
    | none => invalidResult fmt

/-- The point of the closed upper half-plane whose principal argument is the magnitude of
`atan2 y x`: `(x, |y|)` for finite operands, and a point in the direction of the infinite ones
otherwise. -/
def atan2Point (y x : Model fmt) : Option (Rat × Rat) :=
  let direction (v : Model fmt) : Rat := if signBit v then -1 else 1
  if isInf y then some (if isInf x then direction x else 0, 1)
  else if isInf x then some (direction x, 0)
  else (toRat? x).bind fun a => (toRat? y).map fun b => (a, |b|)

/-- `atan2 y x`, the polar angle of `(x, y)` rounded once to nearest even, with the sign of `y`.
A zero `y` gives a zero against `+0` or a positive `x` and `±π` against `-0` or a negative `x`. -/
def atan2 (y x : Model fmt) : Model fmt :=
  if isNaN y then quietNaN y
  else if isNaN x then quietNaN x
  else match atan2Point y x with
    | some (a, b) =>
        let a := if b = 0 ∧ a = 0 ∧ signBit x then -1 else a
        AlgebraicRounding.round fmt (signBit y) (prepareAtan2 a b levels).compare
    | none => invalidResult fmt

end InverseTrigonometric

namespace ConfiguredBinary

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The arctangent rounded once to nearest even (`InverseTrigonometric.atan`). -/
def atan (x : Value) : Value :=
  ExecFloat.ModelCodec.liftUnary (Model := Model format) (plan := plan) InverseTrigonometric.atan x

/-- The inverse sine rounded once to nearest even (`InverseTrigonometric.asin`). -/
def asin (x : Value) : Value :=
  ExecFloat.ModelCodec.liftUnary (Model := Model format) (plan := plan) InverseTrigonometric.asin x

/-- The inverse cosine rounded once to nearest even (`InverseTrigonometric.acos`). -/
def acos (x : Value) : Value :=
  ExecFloat.ModelCodec.liftUnary (Model := Model format) (plan := plan) InverseTrigonometric.acos x

/-- `atan2 y x` rounded once to nearest even (`InverseTrigonometric.atan2`). -/
def atan2 (y x : Value) : Value :=
  ExecFloat.ModelCodec.liftBinary (Model := Model format) (plan := plan)
    InverseTrigonometric.atan2 y x

end ConfiguredBinary

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

/-- The polar angle and the inverse trigonometric functions of a configured format, each the real
function rounded once to nearest even. -/
instance configuredBinaryInverseTrig :
    InverseTrig (ExecFloat (Configured.Family format code plan)) :=
  ⟨ConfiguredBinary.atan2, ConfiguredBinary.atan, ConfiguredBinary.asin, ConfiguredBinary.acos⟩

end TorchLean
