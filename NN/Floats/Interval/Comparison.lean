/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Init.Data.Float32
public import Init.Data.Rat
public import NN.Floats.Interval.IEEEExec32ArbTrans

/-!
# Comparison helpers for executable interval demos

This module contains small, reusable baselines for numerical-audit examples:

- `Float32Interval.IntervalF32`: a deliberately naive runtime-`Float32` interval model;
- `RealInterval.IntervalRat`: exact rational interval arithmetic for small reference checks;
- conversions from finite `IEEE32Exec` / runtime `Float32` endpoints into rational intervals.

The important design point is separation: examples should print comparisons, not quietly define a
second interval library. The primary TorchLean interval implementation is
`IEEE32Exec.Interval32`; this module only provides baselines that make examples and regression tests easier
to read.
-/

@[expose] public section

namespace TorchLean.Floats.Interval.Comparison

open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.IEEE32Exec

/-- Pretty-print an executable `IEEE32Exec.Interval32`, including endpoint bits. -/
def showInterval32 (I : Interval32) : String :=
  s!"[{IEEE32Exec.toFloat I.lo} (bits={I.lo.bits}), {IEEE32Exec.toFloat I.hi} (bits={I.hi.bits})]"

/-- Pretty-print a runtime `Float32`, including its raw IEEE-754 bit pattern. -/
def showFloat32 (x : Float32) : String :=
  s!"{x.toString} (bits={x.toBits})"

namespace Float32Interval

/--
Closed interval with runtime `Float32` endpoints.

This is a *baseline*, not the verified interval implementation. Operations use ordinary runtime
`Float32` arithmetic and therefore do not provide outward-rounding guarantees. It is useful in demos
because it shows exactly why `IEEE32Exec.Interval32` exists.
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
def min4 (a b c d : Float32) : Float32 :=
  min (min a b) (min c d)

/-- Maximum of four runtime `Float32` values using Lean's runtime order. -/
def max4 (a b c d : Float32) : Float32 :=
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
  ⟨min4 p00 p01 p10 p11, max4 p00 p01 p10 p11⟩

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
`IEEE32Exec.Interval32.div` but without directed rounding.
-/
def div (A B : IntervalF32) : IntervalF32 :=
  if containsZero B then
    whole
  else
    let p00 := A.lo / B.lo
    let p01 := A.lo / B.hi
    let p10 := A.hi / B.lo
    let p11 := A.hi / B.hi
    ⟨min4 p00 p01 p10 p11, max4 p00 p01 p10 p11⟩

end IntervalF32

end Float32Interval

/-- Pretty-print a naive runtime-`Float32` interval. -/
def showIntervalF32 (I : Float32Interval.IntervalF32) : String :=
  s!"[{showFloat32 I.lo}, {showFloat32 I.hi}]"

namespace RealInterval

/--
Closed interval with exact rational endpoints.

This is a compact reference domain for examples. It is intentionally small: enough for corner-rule
checks and containment comparisons, not a replacement for a full real-analysis interval library.
-/
structure IntervalRat where
  /-- Lower endpoint. -/
  lo : Rat
  /-- Upper endpoint. -/
  hi : Rat
  deriving Repr

namespace IntervalRat

/-- Degenerate rational interval `[x, x]`. -/
@[inline] def point (x : Rat) : IntervalRat := ⟨x, x⟩

/-- Minimum of four rationals. -/
def min4 (a b c d : Rat) : Rat :=
  min (min a b) (min c d)

/-- Maximum of four rationals. -/
def max4 (a b c d : Rat) : Rat :=
  max (max a b) (max c d)

/-- Exact interval addition over rationals. -/
@[inline] def add (A B : IntervalRat) : IntervalRat :=
  ⟨A.lo + B.lo, A.hi + B.hi⟩

/-- Exact interval negation: `-[lo, hi] = [-hi, -lo]`. -/
@[inline] def neg (A : IntervalRat) : IntervalRat :=
  ⟨-A.hi, -A.lo⟩

/-- Exact interval subtraction over rationals. -/
@[inline] def sub (A B : IntervalRat) : IntervalRat :=
  ⟨A.lo - B.hi, A.hi - B.lo⟩

/-- Classical four-corner multiplication over exact rationals. -/
def mul (A B : IntervalRat) : IntervalRat :=
  let p00 := A.lo * B.lo
  let p01 := A.lo * B.hi
  let p10 := A.hi * B.lo
  let p11 := A.hi * B.hi
  ⟨min4 p00 p01 p10 p11, max4 p00 p01 p10 p11⟩

/-- Boolean check that `outer` contains `inner`. -/
def contains (outer inner : IntervalRat) : Bool :=
  decide (outer.lo ≤ inner.lo ∧ inner.hi ≤ outer.hi)

end IntervalRat

/-- Pretty-print an exact rational interval. -/
def showIntervalRat (I : IntervalRat) : String :=
  s!"[{I.lo}, {I.hi}]"

end RealInterval

open RealInterval

/-- Exact rational endpoint interval for a finite `IEEE32Exec.Interval32`; `none` for NaN/Inf. -/
def interval32ToRat? (I : Interval32) : Option IntervalRat := do
  let lo ← IEEE32Exec.toRat? I.lo
  let hi ← IEEE32Exec.toRat? I.hi
  pure ⟨lo, hi⟩

/-- Exact rational value of a finite runtime `Float32`; `none` for NaN/Inf. -/
def float32ToRat? (x : Float32) : Option Rat :=
  IEEE32Exec.toRat? (IEEE32Exec.ofBits x.toBits)

/-- Exact rational endpoint interval for a finite runtime-`Float32` interval. -/
def intervalF32ToRat? (I : Float32Interval.IntervalF32) : Option IntervalRat := do
  let lo ← float32ToRat? I.lo
  let hi ← float32ToRat? I.hi
  pure ⟨lo, hi⟩

/--
Endpoint-evaluate a unary function over an `IEEE32Exec` interval.

This is intentionally not a sound transcendental interval rule in general; it is a comparison
baseline for demos.
-/
def intervalUnaryEndpoints (f : IEEE32Exec → IEEE32Exec) (lo hi : IEEE32Exec) : Interval32 :=
  let a := f lo
  let b := f hi
  ⟨IEEE32Exec.minimum a b, IEEE32Exec.maximum a b⟩

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
