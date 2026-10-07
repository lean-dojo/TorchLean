/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Numerics.Enclosure.Rational.Runtime

/-!
# Comparison helpers for executable interval examples

This module contains small, reusable baselines for numerical-audit examples:

- `Interval Float32`: a deliberately naive runtime-`Float32` interval model;
- FloatLib's `RationalInterval`: exact rational endpoints for small reference checks;
- conversions from finite `Binary 8 23` / runtime `Float32` endpoints into rational
intervals.

FloatLib supplies the interval arithmetic and its proofs. This module provides baselines that
make examples and regression tests easier to read.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Numerics (RationalInterval)


namespace TorchLean.Floats.Interval.Comparison


/-- Display configured endpoints through binary64, retaining their full original bit encodings. -/
def formatConfigured {fmt : FloatFormat} {plan : Configured.StoragePlan fmt}
    {code : Type} [ExecFloat.ModelCodec plan (Model fmt) code]
    (I : Interval (ExecFloat (Configured.Family fmt code plan))) : String :=
  let lo := ExecFloat.Binary.toFloat <| ExecFloat.Binary.ofModel <|
    Model.cast fmt FloatFormat.binary64 (ExecFloat.Binary.toModel I.lo)
  let hi := ExecFloat.Binary.toFloat <| ExecFloat.Binary.ofModel <|
    Model.cast fmt FloatFormat.binary64 (ExecFloat.Binary.toModel I.hi)
  s!"[{lo} (bits={ExecFloat.Binary.toNatBits I.lo}), " ++
    s!"{hi} (bits={ExecFloat.Binary.toNatBits I.hi})]"

/-- Pretty-print a scalar together with its raw encoding. -/
def formatValue {α β : Type} [ToString α] [ToString β] (bits : α → β) (x : α) : String :=
  s!"{x} (bits={bits x})"

/-- Pretty-print endpoints using a scalar renderer. -/
def format {α : Type} (render : α → String) (I : Interval α) : String :=
  s!"[{render I.lo}, {render I.hi}]"

namespace Naive

/-- `+0.0f` by IEEE-754 binary32 bits. -/
@[inline] def posZero : Float32 := Float32.ofBits 0

/-- `-0.0f` by IEEE-754 binary32 bits. -/
@[inline] def negZero : Float32 := Float32.ofBits (0x80000000 : UInt32)

/-- `+∞` by IEEE-754 binary32 bits. -/
@[inline] def posInf : Float32 := Float32.ofBits (0x7f800000 : UInt32)

/-- `-∞` by IEEE-754 binary32 bits. -/
@[inline] def negInf : Float32 := Float32.ofBits (0xff800000 : UInt32)

/-- Naive endpoint addition; no directed rounding. -/
@[inline] def add {α : Type} [Add α] (A B : Interval α) : Interval α :=
  ⟨A.lo + B.lo, A.hi + B.hi⟩

/-- Naive endpoint subtraction; no directed rounding. -/
@[inline] def sub {α : Type} [Sub α] (A B : Interval α) : Interval α :=
  ⟨A.lo - B.hi, A.hi - B.lo⟩

/-- Classical four-corner multiplication in the endpoint carrier; no directed rounding. -/
def mul {α : Type} [Mul α] [Min α] [Max α] (A B : Interval α) : Interval α :=
  let p00 := A.lo * B.lo
  let p01 := A.lo * B.hi
  let p10 := A.hi * B.lo
  let p11 := A.hi * B.hi
  ⟨min (min p00 p01) (min p10 p11), max (max p00 p01) (max p10 p11)⟩

/-- Conservative fallback interval `[-∞, +∞]`. -/
@[inline] def whole : Interval Float32 := ⟨negInf, posInf⟩

/-- Return `true` iff the interval contains zero, including signed-zero endpoints. -/
def containsZero (I : Interval Float32) : Bool :=
  decide (I.lo ≤ posZero) && decide (negZero ≤ I.hi)

/--
Naive four-corner division when the denominator does not contain zero.

If the denominator straddles zero, return `whole`, mirroring the shape of
`Binary.Interval.div` but without directed rounding.
-/
def div (A B : Interval Float32) : Interval Float32 :=
  if containsZero B then
    whole
  else
    let p00 := A.lo / B.lo
    let p01 := A.lo / B.hi
    let p10 := A.hi / B.lo
    let p11 := A.hi / B.hi
    ⟨min (min p00 p01) (min p10 p11), max (max p00 p01) (max p10 p11)⟩

end Naive

namespace Rational

/-- Classical four-corner multiplication over exact rationals. -/
def mul (A B : RationalInterval) : RationalInterval :=
  let product := Naive.mul (⟨A.lo, A.hi⟩ : Interval Rat) ⟨B.lo, B.hi⟩
  ⟨product.lo, product.hi⟩

/-- Boolean check that `outer` contains `inner`. -/
def contains (outer inner : RationalInterval) : Bool :=
  decide (outer.lo ≤ inner.lo ∧ inner.hi ≤ outer.hi)

/-- Pretty-print an exact rational interval. -/
def format (I : RationalInterval) : String :=
  s!"[{I.lo}, {I.hi}]"

end Rational

/-- Decode rational endpoints, preserving the decoder's special-value policy on failure. -/
def toRational? {α : Type} (decode : α → Option Rat) (I : Interval α) :
    Option RationalInterval :=
  (I.decode? decode).map fun J => ⟨J.lo, J.hi⟩

/--
Endpoint-evaluate a unary function, using the carrier's minimum and maximum.

This is not a sound transcendental interval rule in general; it is a comparison
baseline for examples.
-/
def endpoints {α : Type} [Min α] [Max α] (f : α → α) (lo hi : α) :
    Interval α :=
  let a := f lo
  let b := f hi
  ⟨min a b, max a b⟩

end TorchLean.Floats.Interval.Comparison
