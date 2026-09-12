/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Arb.Oracle
public import NN.Floats.IEEEExec.DirectedRoundingSoundness.Division
public import NN.Floats.Interval.IEEEExec32

/-!
# Arb-backed transcendentals for `IEEE32Exec.Interval32`

`NN/Floats/Interval/IEEEExec32.lean` provides an *executable* endpoint-interval type
`IEEE32Exec.Interval32` with outward-rounded endpoint arithmetic for `add/sub/mul`:

- endpoints live on the IEEE-754 binary32 grid (`IEEE32Exec`),
- `addDown/addUp/mulDown/mulUp` are implemented via exact-dyadic arithmetic + directed rounding.

For transcendentals (`exp/log/tanh/sqrt/...`) the situation is different:

- the executable transcendental wrappers have no proved real-error contract (libm is out of scope),
- `NN/Floats/IEEEExec/Exec32.lean` contains *deterministic* transcendental approximations, but
  they are not proved outward-rounded w.r.t. real semantics.

This file implements a pragmatic “sound route” for interval endpoints of transcendentals:

1. Call the Arb oracle (`NN/Floats/Arb`) to obtain a **rigorous real enclosure** `[L,U] ⊇ f([a,b])`.
2. Convert `L,U : ℚ` directly to **float32 endpoints** with the proved rational rounders:
   - lower endpoint: `roundRatDown`,
   - upper endpoint: `roundRatUp`.

Trust boundary:
- The enclosure `[L,U]` is an **oracle claim** from Arb/python-flint; Arb is the external trusted
  producer for that real enclosure.
- The exact-rational-to-float32 step is in Lean and its enclosure inequalities are proved in
  `NN/Floats/IEEEExec/DirectedRoundingSoundness/Division.lean`.

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

/-! ## Proved outward rounding from `ℚ` to `IEEE32Exec` -/

/-- Proved outward rounding down of an exact rational to a binary32 endpoint. -/
def roundRatQDown (q : Rat) : IEEE32Exec :=
  roundRatDown (q.num < 0) q.num.natAbs q.den

/-- Proved outward rounding up of an exact rational to a binary32 endpoint. -/
def roundRatQUp (q : Rat) : IEEE32Exec :=
  roundRatUp (q.num < 0) q.num.natAbs q.den

/-- Rewrite a rational cast into the signed numerator/positive-denominator form used by the
directed rational rounders. -/
private theorem rat_cast_eq_signed (q : Rat) :
    (q : ℝ) = if q.num < 0 then -((q.num.natAbs : ℝ) / (q.den : ℝ))
      else (q.num.natAbs : ℝ) / (q.den : ℝ) := by
  rw [Rat.cast_def]
  split_ifs with h
  · simp [abs_of_neg h]
    ring
  · have hn : 0 ≤ q.num := le_of_not_gt h
    simp [abs_of_nonneg hn]

/-- The lower rational endpoint conversion is an `EReal` lower bound. -/
theorem toEReal_roundRatQDown_le (q : Rat) :
    toEReal (roundRatQDown q) ≤ ((q : ℝ) : EReal) := by
  rw [rat_cast_eq_signed]
  by_cases h : q.num < 0
  · simpa [roundRatQDown, h, EReal.coe_div, EReal.coe_neg] using
      (toEReal_roundRatDown_le (sign := q.num < 0) (num := q.num.natAbs)
        (den := q.den) q.den_nz)
  · simpa [roundRatQDown, h, EReal.coe_div] using
      (toEReal_roundRatDown_le (sign := q.num < 0) (num := q.num.natAbs)
        (den := q.den) q.den_nz)

/-- The upper rational endpoint conversion is an `EReal` upper bound. -/
theorem toEReal_roundRatQUp_ge (q : Rat) :
    ((q : ℝ) : EReal) ≤ toEReal (roundRatQUp q) := by
  rw [rat_cast_eq_signed]
  by_cases h : q.num < 0
  · simpa [roundRatQUp, h, EReal.coe_div, EReal.coe_neg] using
      (toEReal_roundRatUp_ge (sign := q.num < 0) (num := q.num.natAbs)
        (den := q.den) q.den_nz)
  · simpa [roundRatQUp, h, EReal.coe_div] using
      (toEReal_roundRatUp_ge (sign := q.num < 0) (num := q.num.natAbs)
        (den := q.den) q.den_nz)

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

The exact rational endpoints are passed directly to the proved directed-rational interface. Its
internal fixed-point quotient enclosure may be conservative, but the wrapper inequalities above
cover the complete conversion and there is no caller-selected approximation scale.
-/
def arbUnary (func : String) (X : Interval32) (precBits digits : Nat := 200) : IO Interval32 := do
  let (L, U) ← arbBounds func X (precBits := precBits) (digits := digits)
  let lo32 := roundRatQDown L
  let hi32 := roundRatQUp U
  pure ⟨lo32, hi32⟩

/-- Arb-backed `tanh` enclosure for `Interval32` (oracle + outward rounding to float32 endpoints).
  -/
@[inline] def tanhArb (X : Interval32) (precBits digits : Nat := 200) : IO Interval32 :=
  arbUnary "tanh" X (precBits := precBits) (digits := digits)

/-- Arb-backed `exp` enclosure for `Interval32` (oracle + outward rounding to float32 endpoints). -/
@[inline] def expArb (X : Interval32) (precBits digits : Nat := 200) : IO Interval32 :=
  arbUnary "exp" X (precBits := precBits) (digits := digits)

/-- Arb-backed `log` enclosure for `Interval32` (oracle + outward rounding to float32 endpoints). -/
@[inline] def logArb (X : Interval32) (precBits digits : Nat := 200) : IO Interval32 :=
  arbUnary "log" X (precBits := precBits) (digits := digits)

/-- Arb-backed `sqrt` enclosure for `Interval32` (oracle + outward rounding to float32 endpoints).
  -/
@[inline] def sqrtArb (X : Interval32) (precBits digits : Nat := 200) : IO Interval32 :=
  arbUnary "sqrt" X (precBits := precBits) (digits := digits)

end Interval32

end IEEE32Exec

end TorchLean.Floats.IEEE754
