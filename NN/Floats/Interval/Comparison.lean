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

- `Float32Interval.IntervalF32`: a deliberately naive runtime-`Float32` interval model;
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
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open FloatLib.Numerics (RationalInterval)


namespace TorchLean.Floats.Interval.Comparison


/-- Pretty-print an executable binary32 interval, including endpoint bits. -/
def showInterval32 (I : Interval (Binary 8 23)) : String :=
  let lo := ExecFloat.Binary.toFloat <| ExecFloat.Binary.ofModel <|
    Model.cast FloatFormat.binary32 FloatFormat.binary64 (ExecFloat.Binary.toModel I.lo)
  let hi := ExecFloat.Binary.toFloat <| ExecFloat.Binary.ofModel <|
    Model.cast FloatFormat.binary32 FloatFormat.binary64 (ExecFloat.Binary.toModel I.hi)
  s!"[{lo} (bits={ExecFloat.Binary.toBits32 I.lo}), " ++
    s!"{hi} (bits={ExecFloat.Binary.toBits32 I.hi})]"

/-- Pretty-print a runtime `Float32`, including its raw IEEE-754 bit pattern. -/
def showFloat32 (x : Float32) : String :=
  s!"{x.toString} (bits={x.toBits})"

namespace Float32Interval

/--
Closed interval with runtime `Float32` endpoints.

This baseline uses ordinary runtime `Float32` arithmetic and provides no outward-rounding
guarantee. Examples compare it with FloatLib's proved directed interval arithmetic.
-/
structure IntervalF32 where
  /-- Lower endpoint. -/
  lo : Float32
  /-- Upper endpoint. -/
  hi : Float32
  deriving Repr

namespace IntervalF32

/-- Degenerate runtime-`Float32` interval `[x, x]`. -/
@[inline] def point (x : Float32) : IntervalF32 := ⟨x, x⟩

/-- `+0.0f` by IEEE-754 binary32 bits. -/
@[inline] def posZero : Float32 := Float32.ofBits 0

/-- `-0.0f` by IEEE-754 binary32 bits. -/
@[inline] def negZero : Float32 := Float32.ofBits (0x80000000 : UInt32)

/-- `+∞` by IEEE-754 binary32 bits. -/
@[inline] def posInf : Float32 := Float32.ofBits (0x7f800000 : UInt32)

/-- `-∞` by IEEE-754 binary32 bits. -/
@[inline] def negInf : Float32 := Float32.ofBits (0xff800000 : UInt32)

/-- Minimum of four runtime `Float32` values using Lean's runtime order. -/
def minOfFour (a b c d : Float32) : Float32 :=
  min (min a b) (min c d)

/-- Maximum of four runtime `Float32` values using Lean's runtime order. -/
def maxOfFour (a b c d : Float32) : Float32 :=
  max (max a b) (max c d)

/-- Naive endpoint addition; no directed rounding. -/
@[inline] def add (A B : IntervalF32) : IntervalF32 :=
  ⟨A.lo + B.lo, A.hi + B.hi⟩

/-- Naive interval negation: `-[lo, hi] = [-hi, -lo]`. -/
@[inline] def neg (A : IntervalF32) : IntervalF32 :=
  ⟨-A.hi, -A.lo⟩

/-- Naive endpoint subtraction; no directed rounding. -/
@[inline] def sub (A B : IntervalF32) : IntervalF32 :=
  ⟨A.lo - B.hi, A.hi - B.lo⟩

/-- Classical four-corner multiplication using runtime `Float32`; no directed rounding. -/
def mul (A B : IntervalF32) : IntervalF32 :=
  let p00 := A.lo * B.lo
  let p01 := A.lo * B.hi
  let p10 := A.hi * B.lo
  let p11 := A.hi * B.hi
  ⟨minOfFour p00 p01 p10 p11, maxOfFour p00 p01 p10 p11⟩

/-- Conservative fallback interval `[-∞, +∞]`. -/
@[inline] def whole : IntervalF32 := ⟨negInf, posInf⟩

/-- Boolean comparison wrapper; NaN comparisons evaluate to `false`. -/
@[inline] def leB (x y : Float32) : Bool :=
  decide (x ≤ y)

/-- Return `true` iff the interval contains zero, including signed-zero endpoints. -/
def containsZero (I : IntervalF32) : Bool :=
  leB I.lo posZero && leB negZero I.hi

/--
Naive four-corner division when the denominator does not contain zero.

If the denominator straddles zero, return `whole`, mirroring the shape of
`Binary.Interval.div` but without directed rounding.
-/
def div (A B : IntervalF32) : IntervalF32 :=
  if containsZero B then
    whole
  else
    let p00 := A.lo / B.lo
    let p01 := A.lo / B.hi
    let p10 := A.hi / B.lo
    let p11 := A.hi / B.hi
    ⟨minOfFour p00 p01 p10 p11, maxOfFour p00 p01 p10 p11⟩

end IntervalF32

end Float32Interval

/-- Pretty-print a naive runtime-`Float32` interval. -/
def showIntervalF32 (I : Float32Interval.IntervalF32) : String :=
  s!"[{showFloat32 I.lo}, {showFloat32 I.hi}]"

namespace Rational

/-- Classical four-corner multiplication over exact rationals. -/
def mul (A B : RationalInterval) : RationalInterval :=
  let p00 := A.lo * B.lo
  let p01 := A.lo * B.hi
  let p10 := A.hi * B.lo
  let p11 := A.hi * B.hi
  ⟨min (min p00 p01) (min p10 p11), max (max p00 p01) (max p10 p11)⟩

/-- Boolean check that `outer` contains `inner`. -/
def contains (outer inner : RationalInterval) : Bool :=
  decide (outer.lo ≤ inner.lo ∧ inner.hi ≤ outer.hi)

/-- Pretty-print an exact rational interval. -/
def format (I : RationalInterval) : String :=
  s!"[{I.lo}, {I.hi}]"

end Rational

/-- Exact rational bounds for a finite binary32 interval; `none` for NaN/Inf. -/
def interval32ToRat? (I : Interval (Binary 8 23)) : Option RationalInterval := do
  let lo ← ExecFloat.Binary.toRat? I.lo
  let hi ← ExecFloat.Binary.toRat? I.hi
  pure ⟨lo, hi⟩

/-- Exact rational value of a finite runtime `Float32`; `none` for NaN/Inf. -/
def float32ToRat? (x : Float32) : Option Rat :=
  ExecFloat.Binary.toRat? (ExecFloat.Binary.ofBits32 x.toBits)

/-- Exact rational endpoint interval for a finite runtime-`Float32` interval. -/
def intervalF32ToRat? (I : Float32Interval.IntervalF32) : Option RationalInterval := do
  let lo ← float32ToRat? I.lo
  let hi ← float32ToRat? I.hi
  pure ⟨lo, hi⟩

/--
Endpoint-evaluate a unary function over an `Binary 8 23` interval.

This is not a sound transcendental interval rule in general; it is a comparison
baseline for examples.
-/
def intervalUnaryEndpoints (f : Binary 8 23 → (Binary 8 23)) (lo hi :
  Binary 8 23) : Interval (Binary 8 23) :=
  let a := f lo
  let b := f hi
  ⟨min a b, max a b⟩

/--
Endpoint-evaluate a unary function over a runtime-`Float32` interval.

This is the naive runtime baseline paired with `intervalUnaryEndpoints`.
-/
def intervalUnaryEndpointsF32 (f : Float32 → Float32) (lo hi : Float32) :
    Float32Interval.IntervalF32 :=
  let a := f lo
  let b := f hi
  ⟨min a b, max a b⟩

end TorchLean.Floats.Interval.Comparison
