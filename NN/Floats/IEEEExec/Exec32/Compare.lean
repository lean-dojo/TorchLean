/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32.Arithmetic

/-!
Comparisons for executable IEEE32 values.

The definitions here implement the ordering and classification behavior used by the executable
binary32 arithmetic layer.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

/--
IEEE754 numerical comparison.

Returns `none` (unordered) if either operand is NaN; otherwise returns an `Ordering`.
-/
def compare (x y : IEEE32Exec) : Option Ordering :=
  if isNaN x || isNaN y then
    none
  else if isInf x then
    if isInf y then
      if signBit x == signBit y then some .eq
      else if signBit x then some .lt else some .gt
    else
      if signBit x then some .lt else some .gt
  else if isInf y then
    if signBit y then some .gt else some .lt
  else
    match toDyadic? x, toDyadic? y with
    | some dx, some dy => some (cmpDyadic dx dy)
    | _, _ => none

/--
Strict order induced by IEEE-754 comparison.

`lt x y` is true exactly when `compare x y = some .lt`. In particular, if either side is NaN then
`lt x y` is false (because `compare` returns `none`).
-/
def lt (x y : IEEE32Exec) : Prop :=
  compare x y = some .lt

/--
Non-strict order induced by IEEE-754 comparison.

`le x y` is true when `compare x y` returns `.lt` or `.eq`, and false otherwise (including the NaN
unordered case).
-/
def le (x y : IEEE32Exec) : Prop :=
  match compare x y with
  | some .lt => True
  | some .eq => True
  | _ => False

/-!
## Order lemmas

IEEE-754 comparisons treat NaNs as unordered, so in particular `le x x` is **not** true for NaNs.
For the interval layer, we mainly need the basic fact that `le` is reflexive on finite values.
-/

private lemma cmpDyadic_self (d : Dyadic) : cmpDyadic d d = .eq := by
  by_cases hm : d.mant == 0 <;> simp [cmpDyadic, hm]

/--
`IEEE32Exec.le` is reflexive on finite values.

Informally: if `x` is a finite float32, then `x ≤ x`. (NaNs are excluded by the `isFinite` premise:
for NaN, `isFinite x = false` and `x ≤ x` is false because `compare` returns `none`.)

This lemma is used by the executable interval layer to show that the "point interval" `[x, x]` is
valid whenever `x` is finite.
-/
@[simp] theorem le_self_of_isFinite_eq_true (x : IEEE32Exec) (hx : isFinite x = true) : le x x := by
  have hne : (expField x != expAllOnes) = true := by simpa [isFinite] using hx
  have hexp : (expField x == expAllOnes) = false := by
    cases hEq : (expField x == expAllOnes) with
    | true =>
        have : False := by
          -- Under `hEq`, `expField x != expAllOnes` simplifies to `false`, contradicting `hne`.
          have hne' := hne
          simp [bne, hEq] at hne'
        cases this
    | false =>
        rfl
  have hnan : isNaN x = false := by simp [isNaN, hexp]
  have hinf : isInf x = false := by simp [isInf, hexp]
  -- Split on decoding branches so `toDyadic?` reduces to a constructor and the `compare` match
  -- fires.
  cases hexp0 : (expField x == 0) <;> cases hfrac0 : (fracField x == 0) <;>
    simp [le, compare, toDyadic?, hnan, hinf, hexp0, hfrac0, cmpDyadic_self]

/-- Curried form of `le_self_of_isFinite_eq_true` (useful for `simp`). -/
theorem le_self_of_isFinite_eq_true_imp (x : IEEE32Exec) : isFinite x = true → le x x := by
  intro hx
  exact le_self_of_isFinite_eq_true (x := x) hx

/--
`isFinite x = true → x ≤ x` is always true.

This looks a bit odd, but it's exactly the side-goal produced by `simp [Valid, point]` in the
executable interval module (`NN/Floats/Interval/IEEEExec32.lean`). Registering this as a simp lemma
lets that file stay a one-liner.
-/
@[simp] theorem isFinite_imp_le_self_iff_true (x : IEEE32Exec) :
    (isFinite x = true → le x x) ↔ True := by
  constructor
  · intro _
    trivial
  · intro _
    exact le_self_of_isFinite_eq_true_imp (x := x)

/-- IEEE754 `minimum`: NaNs propagate; `minimum(-0,+0) = -0`. -/
def minimum (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      match compare x y with
      | some .lt => x
      | some .gt => y
      | some .eq =>
          if isZero x && isZero y then
            if signBit x || signBit y then negZero else posZero
          else
            x
      | none => canonicalNaN

/-- IEEE754 `maximum`: NaNs propagate; `maximum(-0,+0) = +0`. -/
def maximum (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      match compare x y with
      | some .lt => y
      | some .gt => x
      | some .eq =>
          if isZero x && isZero y then
            if (!signBit x) || (!signBit y) then posZero else negZero
          else
            x
      | none => canonicalNaN

/--
IEEE754 `minNum`: if exactly one operand is a quiet NaN, return the other operand.

Signaling NaNs still propagate (quieted).
-/
def minNum (x y : IEEE32Exec) : IEEE32Exec :=
  if isSNaN x then quietNaN x
  else if isSNaN y then quietNaN y
  else if isNaN x then
    if isNaN y then quietNaN x else y
  else if isNaN y then
    x
  else
    minimum x y

/--
IEEE754 `maxNum`: if exactly one operand is a quiet NaN, return the other operand.

Signaling NaNs still propagate (quieted).
-/
def maxNum (x y : IEEE32Exec) : IEEE32Exec :=
  if isSNaN x then quietNaN x
  else if isSNaN y then quietNaN y
  else if isNaN x then
    if isNaN y then quietNaN x else y
  else if isNaN y then
    x
  else
    maximum x y

/-- Convert to an exact `Float` (binary64); finite float32 values embed exactly in binary64. -/
def toFloat (x : IEEE32Exec) : Float :=
  if isNaN x then
    Float.ofBits 0x7FF8000000000000
  else if isInf x then
    if signBit x then Float.ofBits 0xFFF0000000000000 else Float.ofBits 0x7FF0000000000000
  else
    match toDyadic? x with
    | none => 0
    | some d =>
        let m : Float := Float.ofNat d.mant
        let m := if d.sign then -m else m
        m.scaleB d.exp

/-- Convert/round an IEEE binary64 `Float` to float32 (ties-to-even). -/
def ofFloat (x : Float) : IEEE32Exec :=
  let b : UInt64 := x.toBits
  let sign : Bool := ((b >>> 63) &&& 0x1) == 0x1
  let e : UInt64 := (b >>> 52) &&& 0x7FF
  let f : UInt64 := b &&& 0x000FFFFFFFFFFFFF
  if e == 0x7FF then
    if f == 0 then (if sign then negInf else posInf) else canonicalNaN
  else if e == 0 then
    if f == 0 then (if sign then negZero else posZero)
    else
      -- subnormal binary64: value = frac * 2^-1074
      roundDyadicToIEEE32 { sign := sign, mant := f.toNat, exp := -1074 }
  else
    -- normal binary64: value = (2^52 + frac) * 2^(e - 1023 - 52) = (2^52+frac) * 2^(e - 1075)
    let mant : Nat := (Nat.shiftLeft 1 52) + f.toNat
    let exp : Int := (Int.ofNat e.toNat) - 1075
    roundDyadicToIEEE32 { sign := sign, mant := mant, exp := exp }

/-
Transcendentals (`exp`, `log`, ...) are not specified by IEEE-754. The executable path uses
Lean definitions and rounds back to float32 for executability.

For better determinism/portability, we provide integer-only approximations for `exp` and `log`
directly in Lean (still no claim of correctly-rounded libm behavior).
-/


end IEEE32Exec

end TorchLean.Floats.IEEE754
