/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Rat.Floor
public import NN.Floats.Arb.Oracle
public import NN.Floats.Interval.IEEEExec32

/-!
# Arb-backed transcendentals for `IEEE32Exec.Interval32`

`NN/Floats/Interval/IEEEExec32.lean` provides an *executable* endpoint-interval type
`IEEE32Exec.Interval32` with outward-rounded endpoint arithmetic for `add/sub/mul`:

- endpoints live on the IEEE-754 binary32 grid (`IEEE32Exec`),
- `addDown/addUp/mulDown/mulUp` are implemented via exact-dyadic arithmetic + directed rounding.

For transcendentals (`exp/log/tanh/sqrt/...`) the situation is different:

- IEEE-754 does **not** specify correctly-rounded transcendentals (libm is out of scope),
- `NN/Floats/IEEEExec/Exec32.lean` contains *deterministic* transcendental approximations, but
  they are not proved outward-rounded w.r.t. real semantics.

This file implements a pragmatic “sound route” for interval endpoints of transcendentals:

1. Call the Arb oracle (`NN/Floats/Arb`) to obtain a **rigorous real enclosure** `[L,U] ⊇ f([a,b])`.
2. Convert `L,U : ℚ` to **float32 endpoints** by rounding outward to the `IEEE32Exec` grid:
   - lower endpoint: round toward `-∞` (using `roundDyadicDown`),
   - upper endpoint: round toward `+∞` (using `roundDyadicUp`).

Trust boundary:
- The enclosure `[L,U]` is an **oracle claim** from Arb/python-flint; this module does not prove the
  Arb computation inside Lean.
- The rounding-to-float32 step is in-Lean and (for dyadic rounding) proved sound in
  `NN/Floats/IEEEExec/DirectedRoundingSoundness.lean`.

The result is useful when you want executable float32 endpoints *and* a clearly delineated source
of transcendental soundness (Arb).
-/

@[expose] public section


namespace Rat

/-- Render a rational in a format that Arb's parser accepts (e.g. `-3/2`, `5`). -/
def toArbString (q : Rat) : String :=
  let n := q.num
  let d := q.den
  if d = 1 then
    toString n
  else
    s!"{n}/{d}"

end Rat

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/-! ## Exact rational view of float32 values -/

/-- Convert an exact dyadic `(-1)^sign * mant * 2^exp` into an exact rational (`ℚ`). -/
def Dyadic.toRat (d : Dyadic) : Rat :=
  let s : Int := if d.sign then -(Int.ofNat d.mant) else Int.ofNat d.mant
  if d.exp ≥ 0 then
    let e : Nat := Int.toNat d.exp
    Rat.ofInt (s * Int.ofNat (pow2 e))
  else
    let e : Nat := Int.toNat (-d.exp)
    (Rat.ofInt s) / (Rat.ofInt (Int.ofNat (pow2 e)))

/--
Exact rational value of a finite `IEEE32Exec` float.

Returns `none` for NaN/Inf.
-/
def toRat? (x : IEEE32Exec) : Option Rat :=
  match toDyadic? x with
  | some d => some d.toRat
  | none => none

/-! ## Outward rounding from `ℚ` to `IEEE32Exec` -/

/--
A rational view of `2^k`.

This is a small helper used to implement `ratToDyadicDown`/`ratToDyadicUp`.
-/
def pow2Rat (k : Nat) : Rat :=
  Rat.ofInt (Int.ofNat (pow2 k))

/--
Approximate a rational `q` from below by a dyadic with denominator `2^k`:

`d = floor(q * 2^k) * 2^{-k}`.
-/
def ratToDyadicDown (q : Rat) (k : Nat) : Dyadic :=
  let m : Int := ⌊q * pow2Rat k⌋
  { sign := m < 0
    mant := Int.natAbs m
    exp := - (Int.ofNat k) }

/--
Approximate a rational `q` from above by a dyadic with denominator `2^k`:

`d = ceil(q * 2^k) * 2^{-k}`.
-/
def ratToDyadicUp (q : Rat) (k : Nat) : Dyadic :=
  let m : Int := ⌈q * pow2Rat k⌉
  { sign := m < 0
    mant := Int.natAbs m
    exp := - (Int.ofNat k) }

/-- Outward rounding down of a rational to the float32 grid (via a dyadic approximation). -/
def roundRatQDown (q : Rat) (k : Nat := 200) : IEEE32Exec :=
  roundDyadicDown (ratToDyadicDown q k)

/-- Outward rounding up of a rational to the float32 grid (via a dyadic approximation). -/
def roundRatQUp (q : Rat) (k : Nat := 200) : IEEE32Exec :=
  roundDyadicUp (ratToDyadicUp q k)

/-! ## Arb-backed interval endpoints for transcendentals -/

namespace Interval32

/--
Decode a float endpoint as an exact rational, failing if the value is NaN/Inf.

This is used to feed exact endpoint strings into the Arb oracle.
-/
def ensureFinite (x : IEEE32Exec) (label : String) : IO Rat := do
  match toRat? x with
  | some q => pure q
  | none => throw <| IO.userError s!"Expected finite IEEE32Exec for {label}, got NaN/Inf."

/--
Call Arb on the real interval `[X.lo, X.hi]` (interpreted exactly as rationals) and return the
oracle-provided rational enclosure bounds `(L,U)`.

This is the only step that crosses the trust boundary.
-/
def arbBounds (func : String) (X : Interval32) (precBits digits : Nat := 200) : IO (Rat × Rat) := do
  let loQ ← ensureFinite X.lo "lo"
  let hiQ ← ensureFinite X.hi "hi"
  let q : TorchLean.Floats.Arb.Query :=
    { func := func
      lo := Rat.toArbString loQ
      hi := Rat.toArbString hiQ
      precBits := precBits
      digits := digits }
  let r ← TorchLean.Floats.Arb.run q
  pure r.outputBall.toRatBounds

/--
Compute an `IEEE32Exec.Interval32` enclosure for a transcendental unary `func` by:

- getting a real enclosure `[L,U]` from Arb,
- rounding endpoints outward to the binary32 grid.

`dyadicBits` controls the internal dyadic approximation used when rounding rationals to float32;
it can be increased if you want the outward rounding to be closer to the Arb bounds.
-/
def arbUnary (func : String) (X : Interval32) (precBits digits : Nat := 200) (dyadicBits : Nat :=
  200) :
    IO Interval32 := do
  let (L, U) ← arbBounds func X (precBits := precBits) (digits := digits)
  let lo32 := roundRatQDown L dyadicBits
  let hi32 := roundRatQUp U dyadicBits
  pure ⟨lo32, hi32⟩

/-- Arb-backed `tanh` enclosure for `Interval32` (oracle + outward rounding to float32 endpoints).
  -/
@[inline] def tanhArb (X : Interval32) (precBits digits : Nat := 200) (dyadicBits : Nat := 200) : IO
  Interval32 :=
  arbUnary "tanh" X (precBits := precBits) (digits := digits) (dyadicBits := dyadicBits)

/-- Arb-backed `exp` enclosure for `Interval32` (oracle + outward rounding to float32 endpoints). -/
@[inline] def expArb (X : Interval32) (precBits digits : Nat := 200) (dyadicBits : Nat := 200) : IO
  Interval32 :=
  arbUnary "exp" X (precBits := precBits) (digits := digits) (dyadicBits := dyadicBits)

/-- Arb-backed `log` enclosure for `Interval32` (oracle + outward rounding to float32 endpoints). -/
@[inline] def logArb (X : Interval32) (precBits digits : Nat := 200) (dyadicBits : Nat := 200) : IO
  Interval32 :=
  arbUnary "log" X (precBits := precBits) (digits := digits) (dyadicBits := dyadicBits)

/-- Arb-backed `sqrt` enclosure for `Interval32` (oracle + outward rounding to float32 endpoints).
  -/
@[inline] def sqrtArb (X : Interval32) (precBits digits : Nat := 200) (dyadicBits : Nat := 200) : IO
  Interval32 :=
  arbUnary "sqrt" X (precBits := precBits) (digits := digits) (dyadicBits := dyadicBits)

end Interval32

end IEEE32Exec

end TorchLean.Floats.IEEE754
