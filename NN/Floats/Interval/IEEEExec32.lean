/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32

/-!
# Executable IEEE32Exec endpoint intervals

This module contains the executable interval definitions for `IEEE32Exec` endpoints.

It keeps the API small: the main goal is to have an **executable** interval type
with endpoints in the same discrete grid as IEEE-754 binary32 (`IEEE32Exec`), using outward-rounded
endpoint arithmetic (`addDown/addUp/mulDown/mulUp`).

Soundness theorems (enclosure proofs) are not bundled here; they are best stated
relative to a chosen real/extended-real interpretation (see
  `NN/Floats/IEEEExec/BridgeERealTotal.lean`).
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

namespace IEEE32Exec

/--
An executable closed interval with IEEE-754 binary32 semantics (`IEEE32Exec`) as endpoints.

This type is intended for *computation* (endpoint arithmetic with outward rounding). Soundness
theorems relating these intervals to real/extended-real interpretations are stated in separate
bridge files (so theorems can choose the right notion of "real meaning" for the application).
-/
structure Interval32 where
  /-- lo. -/
  lo : IEEE32Exec
  /-- hi. -/
  hi : IEEE32Exec
  deriving Repr

namespace Interval32

/-- Membership predicate: `x` lies between `lo` and `hi` using the `IEEE32Exec` order. -/
def mem (I : Interval32) (x : IEEE32Exec) : Prop :=
  le I.lo x ∧ le x I.hi

/-- Enable `x ∈ I` notation for executable `Interval32` intervals. -/
instance : Membership IEEE32Exec Interval32 where
  mem I x := Interval32.mem I x

/-- Unfold membership: `x ∈ I ↔ I.lo ≤ x ∧ x ≤ I.hi`. -/
@[simp] theorem mem_iff (I : Interval32) (x : IEEE32Exec) : x ∈ I ↔ le I.lo x ∧ le x I.hi :=
  Iff.rfl

/--
Validity predicate for executable intervals.

We require both endpoints to be finite (not NaN/Inf) and ordered (`lo ≤ hi`).
-/
def Valid (I : Interval32) : Prop :=
  isFinite I.lo = true ∧ isFinite I.hi = true ∧ le I.lo I.hi

/-- Degenerate interval `[x, x]`. -/
@[inline] def point (x : IEEE32Exec) : Interval32 := ⟨x, x⟩

/--
Interval hull / union enclosure of two executable intervals.

This is the smallest interval (by endpoints) that contains both `A` and `B`, computed by taking the
minimum of lower endpoints and maximum of upper endpoints.

Implementation note: we use IEEE-754 `minimum`/`maximum` so NaNs propagate.
-/
@[inline] def hull (A B : Interval32) : Interval32 :=
  { lo := IEEE32Exec.minimum A.lo B.lo
    hi := IEEE32Exec.maximum A.hi B.hi }

/-- `point x` is valid exactly when `x` is finite. -/
@[simp] theorem valid_point_iff_isFinite (x : IEEE32Exec) :
    Valid (point x) ↔ isFinite x = true := by
  simp [Valid, point]

/-- Outward-rounded interval addition. -/
@[inline] def add (A B : Interval32) : Interval32 :=
  ⟨addDown A.lo B.lo, addUp A.hi B.hi⟩

/-- Interval negation: `-[lo, hi] = [-hi, -lo]`. -/
@[inline] def neg (A : Interval32) : Interval32 :=
  ⟨IEEE32Exec.neg A.hi, IEEE32Exec.neg A.lo⟩

/-- Outward-rounded interval subtraction. -/
@[inline] def sub (A B : Interval32) : Interval32 :=
  ⟨subDown A.lo B.hi, subUp A.hi B.lo⟩

/-!
Helper combinators for interval multiplication.

`mul` needs the minimum/maximum of 4 corner products. We expose these helpers (instead of keeping
them `private`) so downstream soundness proofs can unfold `Interval32.mul` in a stable way.
-/

/-- Minimum of 4 float values, using IEEE `minimum` (NaNs propagate). -/
def min4 (a b c d : IEEE32Exec) : IEEE32Exec :=
  minimum (minimum a b) (minimum c d)

/-- Maximum of 4 float values, using IEEE `maximum` (NaNs propagate). -/
def max4 (a b c d : IEEE32Exec) : IEEE32Exec :=
  maximum (maximum a b) (maximum c d)

/--
Outward-rounded interval multiplication via the classical 4-corner rule.

We compute downward-rounded lower-corner products for the lower bound and upward-rounded products
for the upper bound, then take the min/max across corners.
-/
def mul (A B : Interval32) : Interval32 :=
  let p00 := mulDown A.lo B.lo
  let p01 := mulDown A.lo B.hi
  let p10 := mulDown A.hi B.lo
  let p11 := mulDown A.hi B.hi
  let q00 := mulUp A.lo B.lo
  let q01 := mulUp A.lo B.hi
  let q10 := mulUp A.hi B.lo
  let q11 := mulUp A.hi B.hi
  ⟨min4 p00 p01 p10 p11, max4 q00 q01 q10 q11⟩

/-- The "whole" interval `[-∞, +∞]` (useful as a conservative fallback). -/
@[inline] def whole : Interval32 := ⟨negInf, posInf⟩

/--
Executable Boolean `x ≤ y` using IEEE `compare`.

If `compare` is unordered (`none`, i.e. NaN involved), we return `false`.
-/
def leB (x y : IEEE32Exec) : Bool :=
  match compare x y with
  | some .lt => true
  | some .eq => true
  | _ => false

/--
Returns `true` iff the real interval denoted by `I` contains `0`.

We use this to conservative-handle division by an interval that straddles zero: a single interval
cannot precisely represent the true quotient set (which is typically a union), so we return
`whole` instead.
-/
def containsZero (I : Interval32) : Bool :=
  leB I.lo posZero && leB negZero I.hi

/--
Interval division via the classical 4-corner rule when the denominator interval does not contain
  `0`.

If `0 ∈ B`, we conservatively return `whole = [-∞,+∞]`.
-/
def div (A B : Interval32) : Interval32 :=
  if containsZero B then
    whole
  else
    let p00 := divDown A.lo B.lo
    let p01 := divDown A.lo B.hi
    let p10 := divDown A.hi B.lo
    let p11 := divDown A.hi B.hi
    let q00 := divUp A.lo B.lo
    let q01 := divUp A.lo B.hi
    let q10 := divUp A.hi B.lo
    let q11 := divUp A.hi B.hi
    ⟨min4 p00 p01 p10 p11, max4 q00 q01 q10 q11⟩

/--
Interval reciprocal `1/B`, implemented as a special case of interval division.

If `0 ∈ B`, we return `whole`.
-/
@[inline] def inv (B : Interval32) : Interval32 :=
  div (point posOne) B

end Interval32

end IEEE32Exec

end TorchLean.Floats.IEEE754
