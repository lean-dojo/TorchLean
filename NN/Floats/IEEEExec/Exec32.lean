/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context

/-!
# Executable IEEE-754 binary32 (`IEEE32Exec`)

TorchLean uses two complementary ways to talk about "float32":

- `NN/Floats/NeuralFloat/*` and `NN/Floats/FP32/*` model rounding-on-`ℝ`. This is suited to proofs
  and for compositional "real computation + rounding error" arguments.
- `IEEE32Exec` (in this file) models **bit-level IEEE-754** behavior. This is what you want when you
  care about corner cases like NaN/Inf payload propagation, signed zero, and exact tie-breaking.

We implement `IEEE32Exec` as raw `UInt32` bits and provide:

- decoders/encoders for the binary32 layout,
- `nextUp`/`nextDown` (adjacent representable floats),
- basic arithmetic (`+ - * / fma`) by decoding to an exact dyadic/rational intermediate and then
  rounding once (round-to-nearest, ties-to-even),
- comparisons and `min`/`max` with IEEE-754 NaN rules,
- `sqrt` via integer arithmetic on the exact input value, rounded back to binary32.

We also provide a `Context IEEE32Exec` instance so the spec layer can run modules with an
executable scalar. That is why we import `NN.Spec.Core.Context` here.

## About transcendentals

IEEE-754 does not specify implementations for transcendental functions (`exp`, `tanh`, ...). In
practice those are provided by `libm` (or vendor math libraries) and vary across platforms.

We provide deterministic implementations for a few transcendentals in Lean so examples can
run without delegating to the host runtime. For the remaining ones, we may still delegate to Lean's
`Float` (binary64) and round back to binary32. These functions are executable and stable, but they
are **not** claimed to be correctly rounded or to match any particular hardware/libm.

## References

- IEEE Standard for Floating-Point Arithmetic, IEEE 754-2019.
- David Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic”,
  *ACM Computing Surveys* (1991). DOI: 10.1145/103162.103163
- Jean-Michel Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed. (2018).
- S. Boldo, G. Melquiond, “Flocq: a unified Coq library for proving floating-point algorithms
  correct” (ARITH 2011). DOI: 10.1109/ARITH.2011.40
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

/-- Executable IEEE-754 binary32 value, stored as raw bits. -/
structure IEEE32Exec where
  /-- bits. -/
  bits : UInt32
  deriving DecidableEq, Repr

namespace IEEE32Exec

/-- Wrap raw binary32 bits as an `IEEE32Exec`. -/
@[inline] def ofBits (b : UInt32) : IEEE32Exec := ⟨b⟩

/-- Extract the raw binary32 bits of an `IEEE32Exec`. -/
@[inline] def toBits (x : IEEE32Exec) : UInt32 := x.bits

/-- `toBits (ofBits b) = b`. -/
@[simp] theorem toBits_ofBits (b : UInt32) : toBits (ofBits b) = b := rfl

/-- `ofBits (toBits x) = x`. -/
@[simp] theorem ofBits_toBits (x : IEEE32Exec) : ofBits (toBits x) = x := by
  cases x
  rfl

/-- Default inhabitant: all bits zero, i.e. `+0.0`. -/
instance : Inhabited IEEE32Exec where
  default := ofBits 0

/-!
## Binary32 bit layout

IEEE-754 binary32 is stored as:

- sign bit `s` in bit 31,
- exponent field `e` in bits 30..23 (8 bits, bias 127),
- fraction field `f` in bits 22..0 (23 bits).

For NaNs, the "quiet" bit is the top fraction bit.
-/

-- Masks/constants (binary32 layout: sign[31] exp[30..23] frac[22..0]).
/-- Mask selecting the sign bit (bit 31). -/
def signMask : UInt32 := 0x80000000

/-- Mask selecting the 8-bit exponent field (bits 30..23). -/
def expMask : UInt32 := 0x7F800000

/-- Mask selecting the 23-bit fraction field (bits 22..0). -/
def fracMask : UInt32 := 0x007FFFFF

/-- The IEEE-754 "quiet NaN" indicator bit (top fraction bit). -/
def quietBit : UInt32 := 0x00400000

/-- 8-bit value `0xFF`, used to test the “all ones” exponent field. -/
def expAllOnes : UInt32 := 0xFF

/-- True iff the sign bit (bit 31) is set. -/
@[inline] def signBit (x : IEEE32Exec) : Bool :=
  (x.bits &&& signMask) != 0

/-- Extract the 8-bit exponent field (bits 30..23). -/
@[inline] def expField (x : IEEE32Exec) : UInt32 :=
  (x.bits >>> 23) &&& expAllOnes

/-- Extract the 23-bit fraction field (bits 22..0). -/
@[inline] def fracField (x : IEEE32Exec) : UInt32 :=
  x.bits &&& fracMask

/-- Predicate for NaN: exponent all ones and fraction nonzero. -/
@[inline] def isNaN (x : IEEE32Exec) : Bool :=
  expField x == expAllOnes && fracField x != 0

/-- Predicate for quiet NaN (NaN with the quiet bit set). -/
@[inline] def isQNaN (x : IEEE32Exec) : Bool :=
  isNaN x && (x.bits &&& quietBit) != 0

/-- Predicate for signaling NaN (NaN with the quiet bit clear). -/
@[inline] def isSNaN (x : IEEE32Exec) : Bool :=
  isNaN x && (x.bits &&& quietBit) == 0

/-- Predicate for infinity: exponent all ones and fraction zero. -/
@[inline] def isInf (x : IEEE32Exec) : Bool :=
  expField x == expAllOnes && fracField x == 0

/-- Predicate for signed zero (both `+0` and `-0`). -/
@[inline] def isZero (x : IEEE32Exec) : Bool :=
  expField x == 0 && fracField x == 0

/-- Predicate for finiteness: exponent field is not all ones (excludes NaN/Inf). -/
@[inline] def isFinite (x : IEEE32Exec) : Bool :=
  expField x != expAllOnes

/-- `+0.0` as an executable binary32 constant. -/
@[inline] def posZero : IEEE32Exec := ofBits 0
/-- `-0.0` as an executable binary32 constant. -/
@[inline] def negZero : IEEE32Exec := ofBits signMask

/-- `+1.0` as an IEEE-754 binary32 constant. -/
@[inline] def posOne : IEEE32Exec := ofBits (0x3F800000 : UInt32)
/-- `-1.0` as an IEEE-754 binary32 constant. -/
@[inline] def negOne : IEEE32Exec := ofBits (0xBF800000 : UInt32)

/-- `+∞` as an executable binary32 constant. -/
@[inline] def posInf : IEEE32Exec := ofBits expMask
/-- `-∞` as an executable binary32 constant. -/
@[inline] def negInf : IEEE32Exec := ofBits (signMask ||| expMask)

/-- A canonical quiet NaN payload used by the executable kernel. -/
@[inline] def canonicalNaN : IEEE32Exec := ofBits (expMask ||| quietBit)

/-!
## NaN selection / payload propagation

IEEE-754 leaves some freedom in how NaNs are "chosen" when multiple NaNs appear.
For reproducibility (and nicer debugging), we make the choice deterministic (left-to-right) and we
quiet signaling NaNs by setting the quiet bit.
-/

/-- Quiet a NaN by setting the quiet bit (and leave non-NaNs unchanged). -/
@[inline] def quietNaN (x : IEEE32Exec) : IEEE32Exec :=
  if isNaN x then
    -- IEEE754: quiet NaN has the top fraction bit set.
    ofBits (x.bits ||| quietBit)
  else
    x

/-- If `x` is a NaN, return it (quieted). -/
def chooseNaN1 (x : IEEE32Exec) : Option IEEE32Exec :=
  if isNaN x then some (quietNaN x) else none

/--
Choose a NaN from two operands.

This is the "NaN propagation" policy used by most binary ops in this file:

- if any operand is a signaling NaN, return that operand (quieted), left-to-right,
- otherwise if any operand is a quiet NaN, return that operand, left-to-right,
- otherwise return `none`.
-/
def chooseNaN2 (x y : IEEE32Exec) : Option IEEE32Exec :=
  -- Prefer signaling NaNs (quieted), then quiet NaNs; deterministic left-to-right choice.
  if isSNaN x then some (quietNaN x)
  else if isSNaN y then some (quietNaN y)
  else if isNaN x then some (quietNaN x)
  else if isNaN y then some (quietNaN y)
  else none

/-- Like `chooseNaN2`, but for ternary ops (used for `fma`). -/
def chooseNaN3 (x y z : IEEE32Exec) : Option IEEE32Exec :=
  if isSNaN x then some (quietNaN x)
  else if isSNaN y then some (quietNaN y)
  else if isSNaN z then some (quietNaN z)
  else if isNaN x then some (quietNaN x)
  else if isNaN y then some (quietNaN y)
  else if isNaN z then some (quietNaN z)
  else none

/-! ## Adjacent floats (`nextUp`/`nextDown`) -/

-- Smallest positive subnormal (2^-149) and its negative.
/-- Smallest positive subnormal (bit pattern `0x00000001`, value `2^-149`). -/
@[inline] def posMinSubnormal : IEEE32Exec := ofBits 0x00000001
/-- Smallest negative subnormal (bit pattern `0x80000001`, value `-2^-149`). -/
@[inline] def negMinSubnormal : IEEE32Exec := ofBits (signMask ||| 0x00000001)

-- Largest finite magnitude (just below ±Inf).
/-- Largest finite positive float32 (bit pattern `0x7F7FFFFF`). -/
@[inline] def posMaxFinite : IEEE32Exec := ofBits 0x7F7FFFFF
/-- Largest finite negative float32 (bit pattern `0xFF7FFFFF`). -/
@[inline] def negMaxFinite : IEEE32Exec := ofBits 0xFF7FFFFF

/--
`nextUp x` is the next representable float32 strictly greater than `x`.

IEEE-754 special cases:
- NaN propagates (quieted).
- `nextUp (+∞) = +∞`.
- `nextUp (-0) = +minSubnormal` (since `+0` is not strictly greater than `-0`).
-/
@[inline] def nextUp (x : IEEE32Exec) : IEEE32Exec :=
  if isNaN x then
    quietNaN x
  else if x.bits == posInf.bits then
    posInf
  else if isZero x && signBit x then
    posMinSubnormal
  else if signBit x then
    ofBits (x.bits - 1)
  else
    ofBits (x.bits + 1)

/--
`nextDown x` is the next representable float32 strictly less than `x`.

IEEE-754 special cases:
- NaN propagates (quieted).
- `nextDown (-∞) = -∞`.
- `nextDown (+0) = -minSubnormal` (since `-0` is not strictly less than `+0`).
-/
@[inline] def nextDown (x : IEEE32Exec) : IEEE32Exec :=
  if isNaN x then
    quietNaN x
  else if x.bits == negInf.bits then
    negInf
  else if isZero x && !signBit x then
    negMinSubnormal
  else if signBit x then
    ofBits (x.bits + 1)
  else
    ofBits (x.bits - 1)

/-- Flip the sign bit (works for finite/Inf/NaN, and distinguishes ±0). -/
@[inline] def neg (x : IEEE32Exec) : IEEE32Exec :=
  let b := if isNaN x then (x.bits ||| quietBit) else x.bits
  ofBits (b ^^^ signMask)

/-- Clear the sign bit. -/
@[inline] def abs (x : IEEE32Exec) : IEEE32Exec :=
  let b := x.bits &&& (~~~signMask)
  if isNaN x then ofBits (b ||| quietBit) else ofBits b

/-- Exact dyadic value `(-1)^sign * mant * 2^exp` used as an intermediate for finite ops. -/
structure Dyadic where
  /-- sign. -/
  sign : Bool
  /-- mant. -/
  mant : Nat
  /-- exp. -/
  exp : Int
  deriving Repr, DecidableEq

/-- `2^k` as a natural number. -/
def pow2 (k : Nat) : Nat :=
  Nat.shiftLeft 1 k

/--
Round `n / 2^shift` to nearest, ties-to-even.

This is the primitive "shift + rounding" operation we use when shrinking a mantissa back down to a
fixed bit width. It is the same tie-breaking policy as IEEE round-to-nearest-even.
-/
def roundShiftRightEven (n : Nat) (shift : Nat) : Nat :=
  if shift == 0 then
    n
  else
    let q := Nat.shiftRight n shift
    let rem := n - Nat.shiftLeft q shift
    let half := pow2 (shift - 1)
    if rem < half then q
    else if rem > half then q + 1
    else if q % 2 == 0 then q else q + 1

/--
Construct a raw binary32 bit-pattern from fields.

`mkBits sign exp frac` places:
- `sign` in bit 31,
- `exp` in bits 30..23 (8 bits),
- `frac` in bits 22..0 (masked to 23 bits).
-/
def mkBits (sign : Bool) (exp : Nat) (frac : Nat) : UInt32 :=
  let s : UInt32 := if sign then (1 : UInt32) <<< 31 else 0
  let e : UInt32 := (UInt32.ofNat exp) <<< 23
  let f : UInt32 := (UInt32.ofNat frac) &&& fracMask
  s ||| e ||| f

/--
Decode a finite binary32 into an exact dyadic value.

Returns `none` for NaN/Inf.
-/
def toDyadic? (x : IEEE32Exec) : Option Dyadic :=
  if isNaN x || isInf x then
    none
  else
    let s := signBit x
    let e := expField x
    let f := fracField x
    if e == 0 then
      if f == 0 then
        some { sign := s, mant := 0, exp := 0 }
      else
        -- subnormal: value = frac * 2^-149
        some { sign := s, mant := f.toNat, exp := -149 }
    else
      -- normal: value = (2^23 + frac) * 2^(e-bias-23) = (2^23+frac) * 2^(e - 150)
      let mant := (pow2 23) + f.toNat
      let exp : Int := (Int.ofNat e.toNat) - 150
      some { sign := s, mant := mant, exp := exp }

/--
If `toDyadic? x = some d` then `x` is not a NaN.

Informal: `toDyadic?` only returns `some _` for finite floats; NaNs map to `none`.
-/
lemma isNaN_eq_false_of_toDyadic?_some {x : IEEE32Exec} {d : Dyadic}
    (hx : toDyadic? x = some d) : isNaN x = false := by
  cases hnan : isNaN x
  · rfl
  · -- `isNaN x = true` forces `toDyadic? x = none`.
    unfold toDyadic? at hx
    have hcond : (isNaN x || isInf x) = true := by
      simp [hnan]
    have : (none : Option Dyadic) = some d := by
      simp [hcond] at hx
    cases this

/--
If `toDyadic? x = some d` then `x` is not an infinity.

Informal: `toDyadic?` only returns `some _` for finite floats; infinities map to `none`.
-/
lemma isInf_eq_false_of_toDyadic?_some {x : IEEE32Exec} {d : Dyadic}
    (hx : toDyadic? x = some d) : isInf x = false := by
  cases hinf : isInf x
  · rfl
  · -- `isInf x = true` forces `toDyadic? x = none`.
    unfold toDyadic? at hx
    have hcond : (isNaN x || isInf x) = true := by
      simp [hinf]
    have : (none : Option Dyadic) = some d := by
      simp [hcond] at hx
    cases this

/-!
## Rounding back to binary32

The general pattern for the finite ops in this file is:

1. decode float32(s) to an exact intermediate representation (`Dyadic` for `+ - * fma sqrt`,
   rationals for `/`),
2. compute the exact result in that intermediate representation,
3. round once to float32 using round-to-nearest, ties-to-even.
-/

/--
Round an exact dyadic value to binary32 (ties-to-even).

This function implements:

- overflow to ±Inf,
- gradual underflow into subnormals (down to exponent `-149`),
- underflow-to-zero below `2^-150` (half the minimum subnormal, where ties-to-even chooses 0),
- mantissa rounding to the 24-bit precision of binary32 normal numbers.
-/
def roundDyadicToIEEE32 (d : Dyadic) : IEEE32Exec :=
  -- Exact 0 becomes signed 0.
  if d.mant == 0 then
    if d.sign then negZero else posZero
  else
    let log2m : Nat := Nat.log2 d.mant
    let k : Int := (Int.ofNat log2m) + d.exp
    -- IEEE754 binary32 exponent range (unbiased): normal [-126,127], subnormal down to -149.
    if k > 127 then
      if d.sign then negInf else posInf
    -- Underflow-to-zero threshold is at half the smallest subnormal: 2^-150 (ties-to-even pick 0).
    -- In terms of `k = ⌊log₂ |x|⌋`, all values with `k < -150` round to 0.
    else if k < -150 then
      if d.sign then negZero else posZero
    else if k < -126 then
      -- subnormal rounding: frac = round_to_even( mant * 2^(exp+149) )
      let fracNat : Nat :=
        match d.exp + 149 with
        | .ofNat sh => Nat.shiftLeft d.mant sh
        | .negSucc sh => roundShiftRightEven d.mant (sh + 1)
      if fracNat == 0 then
        if d.sign then negZero else posZero
      else
        match Nat.decLe (pow2 23) fracNat with
        | isTrue _ =>
            -- Rounds up to the smallest normal: exp=1, frac=0.
            ofBits (mkBits d.sign 1 0)
        | isFalse _ =>
            ofBits (mkBits d.sign 0 fracNat)
    else
      -- normal rounding
      let m24 : Nat :=
        if log2m >= 23 then
          roundShiftRightEven d.mant (log2m - 23)
        else
          Nat.shiftLeft d.mant (23 - log2m)
      let k' : Int := if m24 == pow2 24 then k + 1 else k
      let m24' : Nat := if m24 == pow2 24 then pow2 23 else m24
      if k' > 127 then
        if d.sign then negInf else posInf
      else
        let expNat : Nat := Int.toNat (k' + 127)
        let fracNat : Nat := m24' - pow2 23
        ofBits (mkBits d.sign expNat fracNat)

/-!
## Exact dyadic arithmetic (finite core)

`Dyadic` is closed under `+` and `*` (at the exact level). We use these helpers before rounding
back to float32.
-/

/--
Exact dyadic addition.

We align exponents by shifting the mantissa of the operand with the larger exponent, add signed
integers, and then return an exact dyadic (no rounding yet).
-/
def addDyadic (a b : Dyadic) : Dyadic :=
  if a.exp ≤ b.exp then
    let sh : Nat := Int.toNat (b.exp - a.exp)
    let m1 : Int := if a.sign then -(Int.ofNat a.mant) else (Int.ofNat a.mant)
    let m2s : Nat := Nat.shiftLeft b.mant sh
    let m2 : Int := if b.sign then -(Int.ofNat m2s) else (Int.ofNat m2s)
    let s : Int := m1 + m2
    if s == 0 then
      { sign := a.sign && b.sign, mant := 0, exp := 0 }
    else
      { sign := s < 0, mant := Int.natAbs s, exp := a.exp }
  else
    let sh : Nat := Int.toNat (a.exp - b.exp)
    let m1s : Nat := Nat.shiftLeft a.mant sh
    let m1 : Int := if a.sign then -(Int.ofNat m1s) else (Int.ofNat m1s)
    let m2 : Int := if b.sign then -(Int.ofNat b.mant) else (Int.ofNat b.mant)
    let s : Int := m1 + m2
    if s == 0 then
      { sign := a.sign && b.sign, mant := 0, exp := 0 }
    else
      { sign := s < 0, mant := Int.natAbs s, exp := b.exp }

/-!
## Exact rationals for division

For division we compute an exact rational `num/den` (with `den > 0`) and then round it to binary32
using round-to-nearest, ties-to-even.
-/

/-- Round `num/den` to nearest, ties-to-even (assumes `den > 0`). -/
def roundQuotEven (num den : Nat) : Nat :=
  let q := num / den
  let r := num % den
  let twice := 2 * r
  if twice < den then q
  else if twice > den then q + 1
  else if q % 2 == 0 then q else q + 1

/-- Test whether `num/den < 2^k` without converting to reals. -/
def ratLtPow2 (num den : Nat) (k : Int) : Bool :=
  match k with
  | .ofNat kn => num < Nat.shiftLeft den kn
  | .negSucc kn => Nat.shiftLeft num (kn + 1) < den

/-- Test whether `num/den ≥ 2^k` without converting to reals. -/
def ratGePow2 (num den : Nat) (k : Int) : Bool :=
  match k with
  | .ofNat kn => num ≥ Nat.shiftLeft den kn
  | .negSucc kn => Nat.shiftLeft num (kn + 1) ≥ den

/--
Compute `⌊log₂(num/den)⌋` as an `Int` (assumes `num > 0` and `den > 0`).

We start from the rough estimate `log2(num) - log2(den)` and then adjust by checking against
powers of two.
-/
def floorLog2Rat (num den : Nat) : Int :=
  -- num > 0, den > 0
  let k0 : Int := (Int.ofNat (Nat.log2 num)) - (Int.ofNat (Nat.log2 den))
  let k1 : Int := if ratLtPow2 num den k0 then k0 - 1 else k0
  if ratGePow2 num den (k1 + 1) then k1 + 1 else k1

/--
Round an exact rational `num/den` to binary32 (ties-to-even).

This is the division analogue of `roundDyadicToIEEE32`: it uses the same exponent thresholds and
the same final mantissa rounding policy.
-/
def roundRatToIEEE32 (sign : Bool) (num den : Nat) : IEEE32Exec :=
  if num == 0 then
    if sign then negZero else posZero
  else
    let k : Int := floorLog2Rat num den
    if k > 127 then
      if sign then negInf else posInf
    -- Same underflow threshold as `roundDyadicToIEEE32`.
    else if k < -150 then
      if sign then negZero else posZero
    else if k < -126 then
      -- subnormal: frac = round_to_even( (num/den) * 2^149 )
      let num' := Nat.shiftLeft num 149
      let frac := roundQuotEven num' den
      if frac == 0 then
        if sign then negZero else posZero
      else
        match Nat.decLe (pow2 23) frac with
        | isTrue _ => ofBits (mkBits sign 1 0)
        | isFalse _ => ofBits (mkBits sign 0 frac)
    else
      -- normal: m = round_to_even( (num/den) * 2^(23-k) )
      let shift : Int := 23 - k
      let (num', den') :=
        match shift with
        | .ofNat sh => (Nat.shiftLeft num sh, den)
        | .negSucc sh => (num, Nat.shiftLeft den (sh + 1))
      let m := roundQuotEven num' den'
      let k' : Int := if m == pow2 24 then k + 1 else k
      let m' : Nat := if m == pow2 24 then pow2 23 else m
      if k' > 127 then
        if sign then negInf else posInf
      else
        let expNat : Nat := Int.toNat (k' + 127)
        let fracNat : Nat := m' - pow2 23
        ofBits (mkBits sign expNat fracNat)

/-- IEEE754 addition (round-to-nearest, ties-to-even), with NaN/Inf rules. -/
def add (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isInf y then
          if signBit x == signBit y then x else canonicalNaN
        else
          x
      else if isInf y then
        y
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy => roundDyadicToIEEE32 (addDyadic dx dy)
        | _, _ => canonicalNaN

/-- IEEE754 subtraction (defined as addition with sign-flip). -/
@[inline] def sub (x y : IEEE32Exec) : IEEE32Exec :=
  add x (neg y)

/-- IEEE754 multiplication (round-to-nearest, ties-to-even), with NaN/Inf rules. -/
def mul (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isZero y then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else if isInf y then
        if isZero x then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy =>
            let s := Bool.xor dx.sign dy.sign
            if dx.mant == 0 || dy.mant == 0 then
              if s then negZero else posZero
            else
              roundDyadicToIEEE32 { sign := s, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
        | _, _ => canonicalNaN

/-- IEEE754 division (round-to-nearest, ties-to-even), with NaN/Inf rules. -/
def div (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isInf y then canonicalNaN
        else
          -- ±Inf / finite (including ±0) = ±Inf
          if signBit x != signBit y then negInf else posInf
      else if isInf y then
        -- finite / ±Inf = signed zero
        if signBit x != signBit y then negZero else posZero
      else if isZero y then
        if isZero x then canonicalNaN
        else
          -- finite nonzero / ±0 = ±Inf
          if signBit x != signBit y then negInf else posInf
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy =>
            -- Exact quotient: (mx * 2^ex) / (my * 2^ey) = (mx/my) * 2^(ex-ey).
            let sign := Bool.xor dx.sign dy.sign
            if dx.mant == 0 then
              if sign then negZero else posZero
            else
              let eDiff : Int := dx.exp - dy.exp
              let (num, den) :=
                match eDiff with
                | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
                | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
              roundRatToIEEE32 sign num den
        | _, _ => canonicalNaN

/-- IEEE754 fused multiply-add: compute `x*y+z` and round once (ties-to-even). -/
def fma (x y z : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN3 x y z with
  | some nan => nan
  | none =>
      if isInf x || isInf y then
        if isZero x || isZero y then
          canonicalNaN
        else
          let prodSign := Bool.xor (signBit x) (signBit y)
          let prodInf := if prodSign then negInf else posInf
          if isInf z then
            if signBit z != prodSign then canonicalNaN else prodInf
          else
            prodInf
      else if isInf z then
        z
      else
        match toDyadic? x, toDyadic? y, toDyadic? z with
        | some dx, some dy, some dz =>
            let prod : Dyadic :=
              { sign := Bool.xor dx.sign dy.sign
                mant := dx.mant * dy.mant
                exp := dx.exp + dy.exp }
            roundDyadicToIEEE32 (addDyadic prod dz)
        | _, _, _ => canonicalNaN

/--
If both inputs decode to dyadics, `add` is “exact dyadic add, then round once”.

Informal: for finite `x,y`, we compute the exact dyadic value `dx + dy` and apply IEEE
round-to-nearest-even.
-/
theorem add_eq_roundDyadicToIEEE32_of_toDyadic? {x y : IEEE32Exec} {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    add x y = roundDyadicToIEEE32 (addDyadic dx dy) := by
  unfold add
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
  have hchoose : chooseNaN2 x y = none := by
    simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
  simp [hchoose, hxInf, hyInf, hx, hy]

/--
If both inputs decode to dyadics, `mul` is “exact dyadic multiply, then round once”.

Informal: for finite `x,y`, the exact product is a dyadic with mantissa `dx.mant * dy.mant` and
exponent `dx.exp + dy.exp`, and we round that back to binary32.
-/
theorem mul_eq_roundDyadicToIEEE32_of_toDyadic? {x y : IEEE32Exec} {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) :
    mul x y =
      roundDyadicToIEEE32
        { sign := Bool.xor dx.sign dy.sign, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp } :=
          by
  unfold mul
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
  have hchoose : chooseNaN2 x y = none := by
    simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
  simp (config := { zeta := true }) [hchoose, hxInf, hyInf, hx, hy]
  -- Remaining goal: the explicit zero short-circuit agrees with rounding a dyadic with mantissa 0.
  cases h0 : (dx.mant == 0 || dy.mant == 0)
  · -- Nonzero mantissas: the implication premise is impossible.
    intro h
    have hboth : (dx.mant == 0) = false ∧ (dy.mant == 0) = false :=
      (Bool.or_eq_false_iff (x := dx.mant == 0) (y := dy.mant == 0)).1 h0
    cases h with
    | inl hx0 =>
        have hx0b : (dx.mant == 0) = true := (beq_iff_eq).2 hx0
        have : false = true := by
          simp [hboth.1] at hx0b
        cases this
    | inr hy0 =>
        have hy0b : (dy.mant == 0) = true := (beq_iff_eq).2 hy0
        have : false = true := by
          simp [hboth.2] at hy0b
        cases this
  · have h0' : (dx.mant == 0 || dy.mant == 0) = true := by simpa using h0
    have hor : (dx.mant == 0) = true ∨ (dy.mant == 0) = true := by
      have h := h0'
      rw [Bool.or_eq_true (a := dx.mant == 0) (b := dy.mant == 0)] at h
      exact h
    have hprod : ((dx.mant * dy.mant) == 0) = true := by
      cases hor with
      | inl hx0 =>
          have hx0' : dx.mant = 0 := (beq_iff_eq).1 hx0
          simp [hx0']
      | inr hy0 =>
          have hy0' : dy.mant = 0 := (beq_iff_eq).1 hy0
          simp [hy0']
    simp [roundDyadicToIEEE32, hprod]

/--
If all inputs decode to dyadics, `fma x y z` is “exact dyadic `(x*y) + z`, then round once”.

Informal: for finite inputs, we compute the exact dyadic product `dx*dy`, add `dz`, and finally
apply IEEE round-to-nearest-even.
-/
theorem fma_eq_roundDyadicToIEEE32_of_toDyadic? {x y z : IEEE32Exec} {dx dy dz : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) (hz : toDyadic? z = some dz) :
    fma x y z =
      roundDyadicToIEEE32
        (addDyadic
          { sign := Bool.xor dx.sign dy.sign, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
            dz) := by
  unfold fma
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hzNaN : isNaN z = false := isNaN_eq_false_of_toDyadic?_some (hx := hz)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
  have hzInf : isInf z = false := isInf_eq_false_of_toDyadic?_some (hx := hz)
  have hchoose : chooseNaN3 x y z = none := by
    simp [chooseNaN3, isSNaN, hxNaN, hyNaN, hzNaN]
  simp (config := { zeta := true }) [hchoose, hxInf, hyInf, hzInf, hx, hy, hz]

/--
If `x` and `y` decode to dyadics and the denominator mantissa is nonzero, `div x y` is obtained by
forming the exact rational quotient and rounding once.

Informal: for finite nonzero `y`, we compute the exact value `(dx.mant * 2^dx.exp) / (dy.mant *
  2^dy.exp)`
as a rational `num/den` with an exponent adjustment, then apply IEEE round-to-nearest-even.
-/
theorem div_eq_roundRatToIEEE32_of_toDyadic? {x y : IEEE32Exec} {dx dy : Dyadic}
    (hx : toDyadic? x = some dx) (hy : toDyadic? y = some dy) (hy0 : dy.mant ≠ 0) :
    div x y =
      let sign : Bool := Bool.xor dx.sign dy.sign
      let eDiff : Int := dx.exp - dy.exp
      let (num, den) :=
        match eDiff with
        | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
        | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
      roundRatToIEEE32 sign num den := by
  classical
  unfold div
  have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
  have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
  have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
  have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
  have hchoose : chooseNaN2 x y = none := by
    simp [chooseNaN2, isSNaN, hxNaN, hyNaN]

  have hyZero : isZero y = false := by
    cases hzy : isZero y with
    | false => rfl
    | true =>
        unfold isZero at hzy
        have hfields : (expField y == 0) = true ∧ (fracField y == 0) = true := by
          simpa [Bool.and_eq_true] using hzy
        have hdy :
            { sign := signBit y, mant := 0, exp := 0 } = dy := by
          -- In the `isZero` case, `toDyadic?` returns the canonical dyadic `0`.
          unfold toDyadic? at hy
          have hnaninf : (isNaN y || isInf y) = false := by
            simp [hyNaN, hyInf]
          -- Reduce the nested `if` using the extracted bitfield facts.
          simp (config := { zeta := true }) [hnaninf, hfields.1, hfields.2] at hy
          simpa using hy
        have : dy.mant = 0 := by simp [hdy.symm]
        exact (hy0 this).elim

  -- Reduce to the dyadic branch (finite, nonzero divisor).
  simp (config := { zeta := true }) [hchoose, hxInf, hyInf, hyZero, hx, hy]
  -- The explicit `dx.mant == 0` short-circuit agrees with `roundRatToIEEE32`'s `num == 0` branch.
  cases hx0 : (dx.mant == 0) with
  | true =>
      have hx0' : dx.mant = 0 := (beq_iff_eq).1 hx0
      cases hE : dx.exp - dy.exp <;> simp [hx0', roundRatToIEEE32]
  | false =>
      intro h
      have hx0' : dx.mant ≠ 0 := (beq_eq_false_iff_ne (a := dx.mant) (b := 0)).1 hx0
      exact (hx0' h).elim

/-- IEEE754 square root (ties-to-even). -/
def sqrt (x : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN1 x with
  | some nan => nan
  | none =>
      if isInf x then
        if signBit x then canonicalNaN else posInf
      else if isZero x then
        -- sqrt(±0) = ±0
        x
      else if signBit x then
        canonicalNaN
      else
        match toDyadic? x with
        | none => canonicalNaN
        | some d =>
            -- For any finite binary32 input, sqrt is finite and *normal* (no subnormal outputs).
            let expOdd : Bool := (d.exp % 2) != 0
            let mant' : Nat := if expOdd then d.mant * 2 else d.mant
            let expEven : Int := if expOdd then d.exp - 1 else d.exp
            let expHalf : Int := expEven / 2
            let l : Nat := Nat.log2 mant'
            let t : Nat := l / 2
            let p : Nat := 23 - t
            let n : Nat := Nat.shiftLeft mant' (2 * p)
            let q : Nat := Nat.sqrt n
            let r : Nat := n - q * q
            -- Round `sqrt(n)` to the nearest integer.
            --
            -- Write `q = ⌊sqrt(n)⌋` and `r = n - q^2`. The midpoint between `q` and `q+1` is
            -- `q + 1/2`, and for `n : Nat` the comparison reduces to a simple integer test:
            -- we round up iff `r > q`.
            let m0 : Nat :=
              if r > q then q + 1 else q
            let k0 : Int := expHalf + Int.ofNat t
            let k : Int := if m0 == pow2 24 then k0 + 1 else k0
            let m24 : Nat := if m0 == pow2 24 then pow2 23 else m0
            let expNat : Nat := Int.toNat (k + 127)
            let fracNat : Nat := m24 - pow2 23
            ofBits (mkBits false expNat fracNat)

/--
Compare two exact dyadics by exact integer comparison.

This is used internally by executable rounding and special-case logic. The implementation
normalizes the exponents to a common minimum exponent and compares the scaled integer mantissas.
-/
def cmpDyadic (a b : Dyadic) : Ordering :=
  if a.mant == 0 && b.mant == 0 then
    .eq
  else
    let e : Int := if a.exp ≤ b.exp then a.exp else b.exp
    let shA : Nat := Int.toNat (a.exp - e)
    let shB : Nat := Int.toNat (b.exp - e)
    let aNat : Nat := Nat.shiftLeft a.mant shA
    let bNat : Nat := Nat.shiftLeft b.mant shB
    let aInt : Int := if a.sign then -(Int.ofNat aNat) else Int.ofNat aNat
    let bInt : Int := if b.sign then -(Int.ofNat bNat) else Int.ofNat bNat
    compare aInt bInt

/-
Outward-rounded arithmetic (interval-friendly)
=============================================

These ops are meant for *sound enclosures*, not to exactly model hardware rounding-mode flags.
They compute the exact dyadic result and then apply **directed rounding** to float32:

- `roundDyadicDown` rounds toward `-∞` (a lower bound),
- `roundDyadicUp` rounds toward `+∞` (an upper bound).

This matches the way interval arithmetic packages (e.g. INTLAB / IEEE 1788 workflows) implement
outward-rounded endpoints: instead of relying on host rounding modes, we implement the directed
rounding logic explicitly on the binary32 grid.
-/

/-- `ceil(n / 2^shift)` for naturals, implemented via shifts (used for directed rounding). -/
def shiftRightCeilPow2 (n shift : Nat) : Nat :=
  if shift == 0 then
    n
  else
    let q := Nat.shiftRight n shift
    let rem := n - Nat.shiftLeft q shift
    if rem == 0 then q else q + 1

/-- Directed rounding down (toward `-∞`) for a *positive* dyadic `mant * 2^exp`. -/
def roundDyadicPosDown (mant : Nat) (exp : Int) : IEEE32Exec :=
  -- `mant > 0` by construction at call sites.
  let log2m : Nat := Nat.log2 mant
  let k : Int := (Int.ofNat log2m) + exp
  if k > 127 then
    posMaxFinite
  else if k < -149 then
    posZero
  else if k < -126 then
    -- subnormal: value = frac * 2^-149, so frac = floor(mant * 2^(exp+149))
    let fracNat : Nat :=
      match exp + 149 with
      | .ofNat sh => Nat.shiftLeft mant sh
      | .negSucc sh => Nat.shiftRight mant (sh + 1)
    -- `mkBits` masks the fraction to 23 bits; we reduce explicitly to make proofs easier.
    if fracNat == 0 then posZero else ofBits (mkBits false 0 (fracNat % pow2 23))
  else
    -- normal: m24 = floor(mant * 2^(23 - log2m))
    let m24 : Nat :=
      if log2m >= 23 then
        Nat.shiftRight mant (log2m - 23)
      else
        Nat.shiftLeft mant (23 - log2m)
    let expNat : Nat := Int.toNat (k + 127)
    -- `mkBits` masks the fraction to 23 bits; we reduce explicitly to make proofs easier.
    let fracNat : Nat := (m24 - pow2 23) % pow2 23
    ofBits (mkBits false expNat fracNat)

/-- Directed rounding up (toward `+∞`) for a *positive* dyadic `mant * 2^exp`. -/
def roundDyadicPosUp (mant : Nat) (exp : Int) : IEEE32Exec :=
  -- `mant > 0` by construction at call sites.
  let log2m : Nat := Nat.log2 mant
  let k : Int := (Int.ofNat log2m) + exp
  if k > 127 then
    posInf
  else if k < -149 then
    posMinSubnormal
  else if k < -126 then
    -- subnormal: frac = ceil(mant * 2^(exp+149))
    let fracNat : Nat :=
      match exp + 149 with
      | .ofNat sh => Nat.shiftLeft mant sh
      | .negSucc sh => shiftRightCeilPow2 mant (sh + 1)
    if fracNat == 0 then
      posMinSubnormal
    else
      match Nat.decLe (pow2 23) fracNat with
      | isTrue _ =>
          -- rounds up to the smallest normal: exp=1, frac=0
          ofBits (mkBits false 1 0)
      | isFalse _ =>
          ofBits (mkBits false 0 fracNat)
  else
    -- normal: m24 = ceil(mant * 2^(23 - log2m))
    let m24 : Nat :=
      if log2m >= 23 then
        shiftRightCeilPow2 mant (log2m - 23)
      else
        Nat.shiftLeft mant (23 - log2m)
    let k' : Int := if m24 == pow2 24 then k + 1 else k
    let m24' : Nat := if m24 == pow2 24 then pow2 23 else m24
    if k' > 127 then
      posInf
    else
      let expNat : Nat := Int.toNat (k' + 127)
      let fracNat : Nat := m24' - pow2 23
      ofBits (mkBits false expNat fracNat)

/-- Directed rounding down (toward `-∞`) of an exact dyadic to float32. -/
def roundDyadicDown (d : Dyadic) : IEEE32Exec :=
  if d.mant == 0 then
    if d.sign then negZero else posZero
  else if d.sign then
    -- Negative: rounding down makes it *more negative* → round magnitude up.
    neg (roundDyadicPosUp d.mant d.exp)
  else
    roundDyadicPosDown d.mant d.exp

/-- Directed rounding up (toward `+∞`) of an exact dyadic to float32. -/
def roundDyadicUp (d : Dyadic) : IEEE32Exec :=
  if d.mant == 0 then
    if d.sign then negZero else posZero
  else if d.sign then
    -- Negative: rounding up makes it *less negative* → round magnitude down.
    neg (roundDyadicPosDown d.mant d.exp)
  else
    roundDyadicPosUp d.mant d.exp

/-- `addDown x y` is a float32 lower bound for the exact real sum (when finite). -/
def addDown (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isInf y then
          if signBit x == signBit y then x else canonicalNaN
        else
          x
      else if isInf y then
        y
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy => roundDyadicDown (addDyadic dx dy)
        | _, _ => canonicalNaN

/-- `addUp x y` is a float32 upper bound for the exact real sum (when finite). -/
def addUp (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isInf y then
          if signBit x == signBit y then x else canonicalNaN
        else
          x
      else if isInf y then
        y
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy => roundDyadicUp (addDyadic dx dy)
        | _, _ => canonicalNaN

/-- `subDown x y` is a float32 lower bound for the exact real difference (when finite). -/
@[inline] def subDown (x y : IEEE32Exec) : IEEE32Exec :=
  addDown x (neg y)

/-- `subUp x y` is a float32 upper bound for the exact real difference (when finite). -/
@[inline] def subUp (x y : IEEE32Exec) : IEEE32Exec :=
  addUp x (neg y)

/-- `mulDown x y` is a float32 lower bound for the exact real product (when finite). -/
def mulDown (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isZero y then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else if isInf y then
        if isZero x then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy =>
            let s := Bool.xor dx.sign dy.sign
            if dx.mant == 0 || dy.mant == 0 then
              if s then negZero else posZero
            else
              roundDyadicDown { sign := s, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
        | _, _ => canonicalNaN

/-- `mulUp x y` is a float32 upper bound for the exact real product (when finite). -/
def mulUp (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isZero y then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else if isInf y then
        if isZero x then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy =>
            let s := Bool.xor dx.sign dy.sign
            if dx.mant == 0 || dy.mant == 0 then
              if s then negZero else posZero
            else
              roundDyadicUp { sign := s, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
        | _, _ => canonicalNaN

/-!
## Directed rounding for exact rationals (division-friendly)

For `divDown`/`divUp` we need outward rounding of an exact rational `num/den` to the float32 grid.
Our dyadic-directed rounders (`roundDyadicDown`/`roundDyadicUp`) already have a clean soundness
proof,
so we reduce rational rounding to dyadic rounding by building a **dyadic enclosure** of `num/den`:

- lower dyadic: `⌊(num/den) * 2^K⌋ * 2^{-K}`,
- upper dyadic: `⌈(num/den) * 2^K⌉ * 2^{-K}`.

We then apply `roundDyadicDown`/`roundDyadicUp` to these dyadics.

This is sound (it produces outward-rounded endpoints), but it is not necessarily optimally tight; a
larger `ratApproxShift` improves tightness at some computational cost.
-/

/-- Number of extra bits used when turning `num/den` into a dyadic enclosure. -/
def ratApproxShift : Nat := 200

/-- `ceil(num/den)` for naturals, totalized (returns `0` when `den = 0`). -/
def quotCeil (num den : Nat) : Nat :=
  if den == 0 then
    0
  else
    let q := num / den
    let r := num % den
    if r == 0 then q else q + 1

/-- Lower dyadic mantissa for `num/den` at scale `2^ratApproxShift`. -/
def ratLowerMant (num den : Nat) : Nat :=
  (Nat.shiftLeft num ratApproxShift) / den

/-- Upper dyadic mantissa for `num/den` at scale `2^ratApproxShift`. -/
def ratUpperMant (num den : Nat) : Nat :=
  quotCeil (Nat.shiftLeft num ratApproxShift) den

/--
Directed rounding down (toward `-∞`) for a rational `±(num/den)` with `den > 0`.

We do not attempt to be "correctly rounded" in the IEEE-754 sense; we only need a sound lower bound.
-/
def roundRatDown (sign : Bool) (num den : Nat) : IEEE32Exec :=
  if num == 0 then
    if sign then negZero else posZero
  else
    let loMant := ratLowerMant num den
    let hiMant := ratUpperMant num den
    let exp : Int := - (Int.ofNat ratApproxShift)
    if sign then
      -- Negative: rounding down makes it more negative → use an upper bound on the magnitude.
      roundDyadicDown { sign := true, mant := hiMant, exp := exp }
    else
      roundDyadicDown { sign := false, mant := loMant, exp := exp }

/--
Directed rounding up (toward `+∞`) for a rational `±(num/den)` with `den > 0`.
-/
def roundRatUp (sign : Bool) (num den : Nat) : IEEE32Exec :=
  if num == 0 then
    if sign then negZero else posZero
  else
    let loMant := ratLowerMant num den
    let hiMant := ratUpperMant num den
    let exp : Int := - (Int.ofNat ratApproxShift)
    if sign then
      -- Negative: rounding up makes it less negative → use a lower bound on the magnitude.
      roundDyadicUp { sign := true, mant := loMant, exp := exp }
    else
      roundDyadicUp { sign := false, mant := hiMant, exp := exp }

/-- `divDown x y` is a float32 lower bound for the exact real quotient (when finite and `y ≠ 0`). -/
def divDown (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isInf y then canonicalNaN
        else
          -- ±Inf / finite (including ±0) = ±Inf
          if signBit x != signBit y then negInf else posInf
      else if isInf y then
        -- finite / ±Inf = signed zero
        if signBit x != signBit y then negZero else posZero
      else if isZero y then
        if isZero x then canonicalNaN
        else
          -- finite nonzero / ±0 = ±Inf
          if signBit x != signBit y then negInf else posInf
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy =>
            -- Exact quotient: (mx * 2^ex) / (my * 2^ey) = (mx/my) * 2^(ex-ey).
            let sign := Bool.xor dx.sign dy.sign
            if dx.mant == 0 then
              if sign then negZero else posZero
            else
              let eDiff : Int := dx.exp - dy.exp
              let (num, den) :=
                match eDiff with
                | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
                | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
              roundRatDown sign num den
        | _, _ => canonicalNaN

/-- `divUp x y` is a float32 upper bound for the exact real quotient (when finite and `y ≠ 0`). -/
def divUp (x y : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN2 x y with
  | some nan => nan
  | none =>
      if isInf x then
        if isInf y then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else if isInf y then
        if signBit x != signBit y then negZero else posZero
      else if isZero y then
        if isZero x then canonicalNaN
        else
          if signBit x != signBit y then negInf else posInf
      else
        match toDyadic? x, toDyadic? y with
        | some dx, some dy =>
            let sign := Bool.xor dx.sign dy.sign
            if dx.mant == 0 then
              if sign then negZero else posZero
            else
              let eDiff : Int := dx.exp - dy.exp
              let (num, den) :=
                match eDiff with
                | .ofNat sh => (Nat.shiftLeft dx.mant sh, dy.mant)
                | .negSucc sh => (dx.mant, Nat.shiftLeft dy.mant (sh + 1))
              roundRatUp sign num den
        | _, _ => canonicalNaN

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

namespace Transcendentals

/-- Fixed-point scale (in bits) used by the integer-only `exp`/`log` approximations. -/
def fixedScale : Nat := 48

/-- `fixedScale` as an `Int`. -/
def fixedScaleInt : Int := Int.ofNat fixedScale

/-- Fixed-point encoding of `1` at scale `fixedScale` (i.e. `2^fixedScale`). -/
def fixedOne : Int := Int.ofNat (pow2 fixedScale)

/-- Integer power of two: `pow2Int k = 2^k` as an `Int`. -/
def pow2Int (k : Nat) : Int := Int.ofNat (pow2 k)

/--
Round an integer quotient `num/den` to the nearest integer, ties-to-even.

Assumes `den > 0`.
-/
def roundQuotEvenInt (num den : Int) : Int :=
  -- Round `num/den` to nearest, ties-to-even (assumes `den > 0`).
  let q := Int.ediv num den
  let r := Int.emod num den
  let twice := 2 * r
  if twice < den then q
  else if twice > den then q + 1
  else
    if q % 2 == 0 then q else q + 1

/-- Divide by `2^shift`, rounding to nearest with ties-to-even. -/
def roundDivPow2EvenInt (n : Int) (shift : Nat) : Int :=
  if shift == 0 then n else roundQuotEvenInt n (pow2Int shift)

/--
Shift by a power of two: multiply when `k ≥ 0`, divide when `k < 0`.

Division uses ties-to-even rounding.
-/
def shiftPow2EvenInt (n : Int) (k : Int) : Int :=
  match k with
  | .ofNat sh => n * pow2Int sh
  | .negSucc sh => roundDivPow2EvenInt n (sh + 1)

/-- Fixed-point multiplication at scale `fixedScale` (ties-to-even). -/
def fixedMul (a b : Int) : Int :=
  roundDivPow2EvenInt (a * b) fixedScale

/--
Fixed-point division at scale `fixedScale` (ties-to-even).

If `a` and `b` are fixed-point at scale `fixedScale`, the result is at the same scale.
-/
def fixedDiv (a b : Int) : Int :=
  -- `a` and `b` are fixedpoint at scale `fixedScale`; result is fixedpoint at the same scale.
  roundQuotEvenInt (a * fixedOne) b

/-- Divide by a natural number, rounding to nearest with ties-to-even. -/
def fixedDivByNat (a : Int) (n : Nat) : Int :=
  roundQuotEvenInt a (Int.ofNat n)

/-- Convert a dyadic number to a signed fixed-point integer at scale `fixedScale`. -/
def fixedOfDyadic (d : Dyadic) : Int :=
  let signedMant : Int := if d.sign then -(Int.ofNat d.mant) else (Int.ofNat d.mant)
  shiftPow2EvenInt signedMant (d.exp + fixedScaleInt)

/-- Convert a signed fixed-point integer at scale `fixedScale` to a dyadic number. -/
def fixedToDyadic (x : Int) : Dyadic :=
  { sign := x < 0, mant := Int.natAbs x, exp := -fixedScaleInt }

/-- Fixed-point approximation to `ln 2` at scale `fixedScale`. -/
def fixedLn2 : Int := 195103586505167     -- round(ln2 * 2^48)
/-- Fixed-point approximation to `1/ln 2` at scale `fixedScale`. -/
def fixedInvLn2 : Int := 406082553034800  -- round((1/ln2) * 2^48)

-- Coefficients for `2^x` on `[-1/2, 1/2]` using the Taylor series:
--   2^x = Σ (ln 2)^n / n! * x^n
-- Each coefficient is rounded to scale `2^48`.
/-- Fixed-point Taylor coefficients (highest degree first) for `2^x` on `[-1/2, 1/2]`. -/
def exp2PolyCoeffsDesc : List Int :=
  [ 1985781
  , 28648765
  , 371982884
  , 4293262892
  , 43357083587
  , 375306296874
  , 2707262666570
  , 15623017693776
  , 67617750451595
  , 195103586505167
  , 281474976710656
  ]

/-- Evaluate the fixed-point `2^x` polynomial approximation using Horner’s method. -/
def evalExp2Poly (xFixed : Int) : Int :=
  match exp2PolyCoeffsDesc with
  | [] => fixedOne
  | c0 :: cs =>
      -- Horner: p = cN; p = c + x*p
      cs.foldl (fun p c => c + fixedMul xFixed p) c0

end Transcendentals

/-- Deterministic `exp` (no delegation to `Float`): range-reduced `2^(x/ln2)` with a fixedpoint
  polynomial. -/
def exp (x : IEEE32Exec) : IEEE32Exec :=
  if isNaN x then quietNaN x
  else if isInf x then
    if signBit x then posZero else posInf
  else
    match toDyadic? x with
    | none => canonicalNaN
    | some dx =>
        let xFixed := Transcendentals.fixedOfDyadic dx
        let yFixed := Transcendentals.fixedMul xFixed Transcendentals.fixedInvLn2
        -- k = round(y), f = y - k in [-1/2, 1/2].
        let k : Int := Transcendentals.roundDivPow2EvenInt yFixed Transcendentals.fixedScale
        let fFixed : Int := yFixed - k * Transcendentals.pow2Int Transcendentals.fixedScale
        let pFixed : Int := Transcendentals.evalExp2Poly fFixed
        if pFixed ≤ 0 then
          posZero
        else
          roundDyadicToIEEE32
            { sign := false
              , mant := Int.natAbs pFixed
              , exp := k - Transcendentals.fixedScaleInt }

/-- Deterministic `log` (no delegation to `Float`): normalize `x = m*2^k` and use an atanh-series
  for `log m`. -/
def log (x : IEEE32Exec) : IEEE32Exec :=
  if isNaN x then quietNaN x
  else if isInf x then
    if signBit x then canonicalNaN else posInf
  else if isZero x then
    negInf
  else if signBit x then
    -- We follow common libm behavior: log(negative) = NaN (including log(-0) already handled
    -- above).
    canonicalNaN
  else
    match toDyadic? x with
    | none => canonicalNaN
    | some dx =>
        if dx.mant == 0 then
          negInf
        else
          let k : Int := (Int.ofNat (Nat.log2 dx.mant)) + dx.exp
          -- m = x / 2^k ∈ [1,2)
          let mFixed : Int :=
            Transcendentals.fixedOfDyadic { sign := false, mant := dx.mant, exp := dx.exp - k }
          let u : Int := mFixed - Transcendentals.fixedOne
          let v : Int := mFixed + Transcendentals.fixedOne
          let t : Int := Transcendentals.fixedDiv u v
          let t2 : Int := Transcendentals.fixedMul t t
          -- log(m) = 2 * (t + t^3/3 + t^5/5 + ...), convergent for m ∈ [1,2).
          let term3 : Int := Transcendentals.fixedMul t t2
          let term5 : Int := Transcendentals.fixedMul term3 t2
          let term7 : Int := Transcendentals.fixedMul term5 t2
          let term9 : Int := Transcendentals.fixedMul term7 t2
          let term11 : Int := Transcendentals.fixedMul term9 t2
          let term13 : Int := Transcendentals.fixedMul term11 t2
          let term15 : Int := Transcendentals.fixedMul term13 t2
          let sum : Int :=
            t
            + Transcendentals.fixedDivByNat term3 3
            + Transcendentals.fixedDivByNat term5 5
            + Transcendentals.fixedDivByNat term7 7
            + Transcendentals.fixedDivByNat term9 9
            + Transcendentals.fixedDivByNat term11 11
            + Transcendentals.fixedDivByNat term13 13
            + Transcendentals.fixedDivByNat term15 15
          let logmFixed : Int := 2 * sum
          let kLn2Fixed : Int := k * Transcendentals.fixedLn2
          let logxFixed : Int := logmFixed + kLn2Fixed
          roundDyadicToIEEE32 (Transcendentals.fixedToDyadic logxFixed)

/-- Deterministic `sinh` (no delegation to `Float`): defined via `exp`. -/
def sinh (x : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN1 x with
  | some nan => nan
  | none =>
      if isInf x then
        x
      else
        -- For small `|x|`, `0.5*(exp(x)-exp(-x))` suffers cancellation. A short Taylor polynomial
        -- is both deterministic and more accurate near 0.
        let ax := abs x
        let half : IEEE32Exec := ofBits 0x3F000000
        match compare ax half with
        | some .lt =>
            -- `sinh x ≈ x + x^3/3! + x^5/5!` for `|x| < 0.5`.
            let x2 := mul x x
            let x3 := mul x2 x
            let x5 := mul x3 x2
            let six : IEEE32Exec := ofBits 0x40C00000      -- 6.0
            let oneTwenty : IEEE32Exec := ofBits 0x42F00000 -- 120.0
            add (add x (div x3 six)) (div x5 oneTwenty)
        | _ =>
            mul (sub (exp x) (exp (neg x))) half

/-- Deterministic `cosh` (no delegation to `Float`): defined via `exp`. -/
def cosh (x : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN1 x with
  | some nan => nan
  | none =>
      if isInf x then
        posInf
      else
        let half : IEEE32Exec := ofBits 0x3F000000
        mul (add (exp x) (exp (neg x))) half

/-- Deterministic `tanh` (no delegation to `Float`): stable form `tanh x = s*(1 -
  2/(exp(2*|x|)+1))`. -/
def tanh (x : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN1 x with
  | some nan => nan
  | none =>
      if isInf x then
        if signBit x then ofBits 0xBF800000 else ofBits 0x3F800000
  else
        let one : IEEE32Exec := ofBits 0x3F800000
        let two : IEEE32Exec := ofBits 0x40000000
        let s : Bool := signBit x
        let ax : IEEE32Exec := abs x
        let e : IEEE32Exec := exp (add ax ax)
        let tpos : IEEE32Exec := sub one (div two (add e one))
        if s then neg tpos else tpos

namespace Trig

/-!
Deterministic `sin`/`cos`
========================

Unlike `exp`/`log`, `sin` and `cos` are used by the runtime FFT layer (`NN.Runtime.*.Fft`) to build
twiddle factors. Delegating to the host `Float` implementation makes results platform-dependent.

We implement `sin`/`cos` purely inside Lean:

1. scale the input down by a power of two so `|y| < 1/2`,
2. approximate `sin y` and `cos y` by exact Taylor partial sums (degree 13 / 12),
3. scale back up using `m` applications of the double-angle formulas.

This is **deterministic** and uses only the IEEE32Exec kernel ops (`roundRatToIEEE32`, `add/mul/sub`,
etc.). We do not claim correctly-rounded libm behavior; the goal is reproducible execution.
-/

@[inline] def mulDyadic (a b : Dyadic) : Dyadic :=
  { sign := Bool.xor a.sign b.sign, mant := a.mant * b.mant, exp := a.exp + b.exp }

/-- Dyadic `1`. -/
@[inline] def oneDyadic : Dyadic :=
  { sign := false, mant := 1, exp := 0 }

/--
Round the exact rational `d / den` to binary32, where `d` is an exact dyadic `mant * 2^exp`.

We package the dyadic exponent into a rational numerator/denominator and call `roundRatToIEEE32`.
-/
def roundDyadicDivNat (d : Dyadic) (den : Nat) : IEEE32Exec :=
  if d.mant == 0 then
    if d.sign then negZero else posZero
  else
    match d.exp with
    | .ofNat sh =>
        roundRatToIEEE32 d.sign (Nat.shiftLeft d.mant sh) den
    | .negSucc sh =>
        roundRatToIEEE32 d.sign d.mant (den * pow2 (sh + 1))

/-!
### Taylor partial sums on `|y| < 1/2`

We encode the partial sums using a common factorial denominator so the coefficients are *exact*
integers, not approximations.

For `z = y^2`:

* `sin y = ∑_{i=0}^6 (-1)^i y^(2i+1)/(2i+1)! + R₇(y)`
  where the polynomial part can be written as
  `y * (∑_{i=0}^6 (-1)^i (13!/(2i+1)!) z^i) / 13!`.

* `cos y = ∑_{i=0}^6 (-1)^i y^(2i)/(2i)! + R₇'(y)`
  i.e.
  `(∑_{i=0}^6 (-1)^i (12!/(2i)!) z^i) / 12!`.
-/

def sinDen : Nat := 6227020800   -- 13!
def cosDen : Nat := 479001600    -- 12!

def sinCoeff (i : Nat) : Int :=
  match i with
  | 0 => 6227020800
  | 1 => -1037836800
  | 2 => 51891840
  | 3 => -1235520
  | 4 => 17160
  | 5 => -156
  | _ => 1

def cosCoeff (i : Nat) : Int :=
  match i with
  | 0 => 479001600
  | 1 => -239500800
  | 2 => 19958400
  | 3 => -665280
  | 4 => 11880
  | 5 => -132
  | _ => 1

def coeffToDyadic (c : Int) : Dyadic :=
  if c < 0 then
    { sign := true, mant := Int.natAbs c, exp := 0 }
  else
    { sign := false, mant := Int.natAbs c, exp := 0 }

def evalPolyNumerator (coeff : Nat → Int) (z : Dyadic) : Dyadic :=
  -- Degree 6 polynomial: Σ_{i=0..6} coeff(i) * z^i, evaluated by successive multiplication.
  let step (st : Dyadic × Dyadic) (i : Nat) : Dyadic × Dyadic :=
    let pow := st.1
    let acc := st.2
    let term := mulDyadic (coeffToDyadic (coeff i)) pow
    let acc' := addDyadic acc term
    let pow' := mulDyadic pow z
    (pow', acc')
  -- Start with `pow = 1`, `acc = 0`, and fold over `i = 0..6`.
  let acc0 : Dyadic := { sign := false, mant := 0, exp := 0 }
  ((List.range 7).foldl step (oneDyadic, acc0)).2

def sinCosTaylorSmall (y : Dyadic) : IEEE32Exec × IEEE32Exec :=
  -- z = y^2
  let z : Dyadic :=
    { sign := false, mant := y.mant * y.mant, exp := y.exp + y.exp }
  let sinNum : Dyadic := evalPolyNumerator sinCoeff z
  let cosNum : Dyadic := evalPolyNumerator cosCoeff z
  let sinDy : Dyadic := mulDyadic y sinNum
  let s : IEEE32Exec := roundDyadicDivNat sinDy sinDen
  let c : IEEE32Exec := roundDyadicDivNat cosNum cosDen
  (s, c)

@[inline] def doubleAngle (sc : IEEE32Exec × IEEE32Exec) : IEEE32Exec × IEEE32Exec :=
  let s := sc.1
  let c := sc.2
  let two : IEEE32Exec := ofBits 0x40000000
  let ss := mul s s
  let cc := mul c c
  let sc' := mul s c
  let s' := mul two sc'
  let c' := sub cc ss
  (s', c')

def iterDoubleAngle : Nat → (IEEE32Exec × IEEE32Exec) → (IEEE32Exec × IEEE32Exec)
  | 0, sc => sc
  | n + 1, sc => iterDoubleAngle n (doubleAngle sc)

@[inline] def sinCosPow2 (y : Dyadic) (m : Nat) : IEEE32Exec × IEEE32Exec :=
  iterDoubleAngle m (sinCosTaylorSmall y)

def sinCosScaled (dx : Dyadic) : IEEE32Exec × IEEE32Exec :=
  -- Scale down so `|y| < 1/2`, approximate there, then scale back up with double angles.
  let k : Int := (Int.ofNat (Nat.log2 dx.mant)) + dx.exp
  let m : Nat :=
    match k + 2 with
    | .ofNat n => n
    | .negSucc _ => 0
  let y : Dyadic :=
    { sign := dx.sign, mant := dx.mant, exp := dx.exp - (Int.ofNat m) }
  sinCosPow2 y m

/-- Joint deterministic `sin`/`cos` computation for `IEEE32Exec`. -/
def sinCos (x : IEEE32Exec) : IEEE32Exec × IEEE32Exec :=
  if isNaN x then
    let q := quietNaN x
    (q, q)
  else if isInf x then
    (canonicalNaN, canonicalNaN)
  else if isZero x then
    -- Preserve signed zero for `sin` and return `cos(±0) = 1`.
    (x, ofBits 0x3F800000)
  else
    match toDyadic? x with
    | none => (canonicalNaN, canonicalNaN)
    | some dx => sinCosScaled dx

/-- Deterministic `sin` implementation (shared core via `sinCos`). -/
def sin (x : IEEE32Exec) : IEEE32Exec :=
  (sinCos x).1

/-- Deterministic `cos` implementation (shared core via `sinCos`). -/
def cos (x : IEEE32Exec) : IEEE32Exec :=
  (sinCos x).2

end Trig

/-!
### Public `sin` / `cos`

We expose `sin`/`cos` as executable ops on `IEEE32Exec` using the deterministic implementation
above, together with standard IEEE special-case conventions.
-/

/-- Deterministic `sin` for `IEEE32Exec`. -/
def sin (x : IEEE32Exec) : IEEE32Exec :=
  Trig.sin x

/-- Deterministic `cos` for `IEEE32Exec`. -/
def cos (x : IEEE32Exec) : IEEE32Exec :=
  Trig.cos x

/-- Pretty-print using Lean's `Float` printer (via `toFloat`). -/
instance : ToString IEEE32Exec where
  toString x := toString (toFloat x)

/-- Coerce naturals to binary32 by converting through Lean's `Float` and re-encoding. -/
instance : Coe Nat IEEE32Exec where
  coe n := ofFloat (Float.ofNat n)

/--
Numeral literals for `IEEE32Exec`.

This allows writing:
- `(1 : IEEE32Exec)`
- `(42 : IEEE32Exec)`

The interpretation is `Nat → Float (binary64) → IEEE32Exec` via `ofFloat`, i.e. it rounds the exact
integer to the nearest representable float32 (which is exact for small enough integers).
-/
instance (n : Nat) : OfNat IEEE32Exec n :=
  ⟨(n : IEEE32Exec)⟩

/-- `0.0` as an executable binary32 value (chosen as `+0.0`). -/
instance : Zero IEEE32Exec where
  zero := posZero
/-- `1.0` as an executable binary32 value. -/
instance : One IEEE32Exec where
  one := ofBits 0x3F800000

/-- Unary negation (IEEE-754 sign flip, with NaN payload rules). -/
instance : Neg IEEE32Exec where
  neg := neg
/-- IEEE-754 addition (with NaN/Inf rules). -/
instance : Add IEEE32Exec where
  add := add
/-- IEEE-754 subtraction (with NaN/Inf rules). -/
instance : Sub IEEE32Exec where
  sub := sub
/-- IEEE-754 multiplication (with NaN/Inf rules). -/
instance : Mul IEEE32Exec where
  mul := mul
/-- IEEE-754 division (with NaN/Inf rules). -/
instance : Div IEEE32Exec where
  div := div

/--
Exponentiation instance.

This is a *deterministic* executable choice, not a claim about correctly-rounded `pow`:
we implement a small set of IEEE-like special cases and handle integer exponents exactly enough to
avoid the most common footguns (negative bases, `0^0`, `1^∞`, etc.). For general non-integer
exponents we fall back to `exp (b * log a)` on positive bases.
-/
instance : Pow IEEE32Exec IEEE32Exec where
  pow a b :=
    -- `x^0 = 1` even if `x` is NaN/Inf (common `pow` convention; also avoids `∞*0 = NaN`).
    if isZero b then
      (1 : IEEE32Exec)
    else
      match chooseNaN2 a b with
      | some nan => nan
      | none =>
          -- `1^y = 1` for all non-NaN `y` (including `±Inf`).
          if compare a (1 : IEEE32Exec) = some .eq then
            (1 : IEEE32Exec)
          else if compare b (1 : IEEE32Exec) = some .eq then
            a
          else
            let intOfDyadic? (d : Dyadic) : Option Int :=
              if d.mant == 0 then
                some 0
              else
                match d.exp with
                | .ofNat sh =>
                    let n : Nat := Nat.shiftLeft d.mant sh
                    some (if d.sign then -Int.ofNat n else Int.ofNat n)
                | .negSucc sh =>
                    -- integer iff `mant` is divisible by `2^(sh+1)`.
                    let k := sh + 1
                    let denom := pow2 k
                    if d.mant % denom == 0 then
                      let q := d.mant / denom
                      some (if d.sign then -Int.ofNat q else Int.ofNat q)
                    else
                      none

            let intOfIEEE? (x : IEEE32Exec) : Option Int :=
              match toDyadic? x with
              | some dx => intOfDyadic? dx
              | none => none

            let oddInt (n : Int) : Bool :=
              (Int.natAbs n) % 2 == 1

            let rec powNatLinear (a : IEEE32Exec) : Nat → IEEE32Exec
              | 0 => (1 : IEEE32Exec)
              | n + 1 => mul a (powNatLinear a n)

            let powIntLinear (a : IEEE32Exec) (n : Int) : IEEE32Exec :=
              match n with
              | .ofNat k => powNatLinear a k
              | .negSucc k => div (1 : IEEE32Exec) (powNatLinear a (Nat.succ k))

            let smallPowLimit : Nat := 256

            -- Integer exponent fast path: keep this cheap for common small integers, and use
            -- `exp/log` for huge integer exponents (still deterministic, and avoids O(n) loops).
            let powIntDet (a : IEEE32Exec) (n : Int) : IEEE32Exec :=
              if Int.natAbs n ≤ smallPowLimit then
                powIntLinear a n
              else
                exp (b * log (abs a))

            match compare a posZero with
            | some .eq =>
                -- `0^b` for nonzero `b`.
                match intOfIEEE? b with
                | some n =>
                    let odd := oddInt n
                    match compare b posZero with
                    | some .lt =>
                        -- `±0` raised to a negative integer exponent is an infinity; sign depends on
                        -- oddness (mirrors the real-limit behavior).
                        if signBit a && odd then negInf else posInf
                    | some .gt =>
                        if signBit a && odd then negZero else posZero
                    | _ =>
                        posZero
                | none =>
                    match compare b posZero with
                    | some .lt => posInf
                    | _ => posZero
            | some .lt =>
                -- Negative base: only defined for integer exponents over ℝ.
                match intOfIEEE? b with
                | none => canonicalNaN
                | some n =>
                    let mag := powIntDet a n
                    if oddInt n then neg (abs mag) else abs mag
            | _ =>
                -- Positive base.
                match intOfIEEE? b with
                | some n => powIntDet a n
                | none => exp (b * log a)

/-
Equality and order
==================

These instances are kept explicit and simple:

- `BEq` returns `false` if either side is NaN (matching IEEE comparisons being unordered),
- `BEq` treats `+0` and `-0` as equal (use `bits` if you need to distinguish them),
- `<`/`≤` are defined via `compare`; unordered comparisons are `False`.
-/

/--
Boolean equality with IEEE-754 NaN/zero conventions.

- If either side is NaN, we return `false`.
- If both are zeros (either sign), we return `true`.
- Otherwise we compare raw bits.
-/
instance : BEq IEEE32Exec where
  beq a b :=
    if isNaN a || isNaN b then
      false
    else if isZero a && isZero b then
      true
    else
      a.bits == b.bits

/-- Strict order instance, defined via `IEEE32Exec.lt`. -/
instance : LT IEEE32Exec where
  lt := lt
/-- Non-strict order instance, defined via `IEEE32Exec.le`. -/
instance : LE IEEE32Exec where
  le := le

/-- Decidable `<` inherited from the `compare`-based definition. -/
instance : DecidableRel ((· < ·) : IEEE32Exec → IEEE32Exec → Prop) := by
  intro x y
  -- `x < y` is definitionally `IEEE32Exec.lt x y`.
  change Decidable (IEEE32Exec.lt x y)
  dsimp [IEEE32Exec.lt]
  infer_instance

/-- Decidable `≤` inherited from the `compare`-based definition. -/
instance : DecidableRel ((· ≤ ·) : IEEE32Exec → IEEE32Exec → Prop) := by
  intro x y
  -- `x ≤ y` is definitionally `IEEE32Exec.le x y`.
  change Decidable (IEEE32Exec.le x y)
  dsimp [IEEE32Exec.le]
  cases h : IEEE32Exec.compare x y with
  | none =>
      -- `le` returns `False` on unordered (NaN) comparisons.
      exact isFalse (by intro hFalse; cases hFalse)
  | some o =>
      cases o with
      | lt => exact isTrue trivial
      | eq => exact isTrue trivial
      | gt => exact isFalse (by intro hFalse; cases hFalse)

/-- `min` operator, implemented by IEEE-754 `minimum`. -/
instance : Min IEEE32Exec where
  min := minimum
/-- `max` operator, implemented by IEEE-754 `maximum`. -/
instance : Max IEEE32Exec where
  max := maximum

/-- Provide the `MathFunctions` interface using the deterministic implementations in this file. -/
instance : MathFunctions IEEE32Exec where
  exp x := IEEE32Exec.exp x
  tanh x := IEEE32Exec.tanh x
  cosh x := IEEE32Exec.cosh x
  sqrt x := IEEE32Exec.sqrt x
  abs x := IEEE32Exec.abs x
  log x := IEEE32Exec.log x
  pi := ofBits 0x40490FDB
  cos x := IEEE32Exec.cos x
  sin x := IEEE32Exec.sin x
  sinh x := IEEE32Exec.sinh x

/-- Numeric constants used by the spec library, instantiated at binary32. -/
instance : Numbers IEEE32Exec where
  neg_point_five := ofFloat (-0.5)
  neg_one := ofFloat (-1)
  pointone := ofFloat 0.1
  pointfive := ofFloat 0.5
  zero := posZero
  one := ofBits 0x3F800000
  two := ofFloat 2
  three := ofFloat 3
  four := ofFloat 4
  five := ofFloat 5
  ten := ofFloat 10
  log10 := ofFloat (Float.log 10)
  log10000 := ofFloat (Float.log 10000)
  epsilon := ofFloat (1e-6)
  neg_thousand := ofFloat (-1000)

/-- `Context` instance so the spec layer can execute with `IEEE32Exec` scalars. -/
instance : Context IEEE32Exec := {
  decidable_gt := fun x y => inferInstanceAs (Decidable (x > y))
}

end IEEE32Exec

end TorchLean.Floats.IEEE754
