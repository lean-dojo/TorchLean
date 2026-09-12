/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32.Dyadic

/-!
Directed executable IEEE32 operations.

This file provides lower and upper rounding variants for arithmetic operations used to compute
outward interval enclosures.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

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
        | some dx, some dy =>
            let exact := addDyadic dx dy
            if exact.mant == 0 then
              -- IEEE roundTowardNegative gives an exact cancelling sum the sign of -0. Same-sign
              -- zero addition retains that common sign.
              if dx.sign || dy.sign then negZero else posZero
            else
              roundDyadicDown exact
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

/--
`fmaDown x y z` computes the exact value `x * y + z` and rounds once toward `-∞`.

This is a directed fused multiply-add: the product is not rounded before the addition. The
special-value cases agree with `fma`; finite inputs are converted to exact dyadics, and each
nonzero exact product-plus-sum is passed once to `roundDyadicDown`. Exact cancellation bypasses the
generic rounder and returns the signed zero required by round-toward-negative.
-/
def fmaDown (x y z : IEEE32Exec) : IEEE32Exec :=
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
            let exact := addDyadic prod dz
            if exact.mant == 0 then
              if prod.sign || dz.sign then negZero else posZero
            else
              roundDyadicDown exact
        | _, _, _ => canonicalNaN

/--
`fmaUp x y z` computes the exact value `x * y + z` and rounds once toward `+∞`.

As with `fmaDown`, no intermediate product rounding occurs.  Keeping both operations in terms of
the same exact dyadic expression makes their enclosure theorems share one semantic statement.
-/
def fmaUp (x y z : IEEE32Exec) : IEEE32Exec :=
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
            roundDyadicUp (addDyadic prod dz)
        | _, _, _ => canonicalNaN

/-!
## Directed square root

For a positive dyadic `mant * 2^exp`, parity normalization rewrites the radicand as
`mant' * 2^(2 * expHalf)`.  After scaling `mant'` by an even power of two, `Nat.sqrt` gives the
lower integer endpoint and its remainder tells us whether the upper endpoint is distinct.  The
resulting dyadics enclose the exact square root before they are rounded to binary32.
-/

/-- Dyadic lower and upper endpoints for a nonnegative square root. -/
structure SqrtDyadicBracket where
  /-- Lower endpoint. -/
  lower : Dyadic
  /-- Upper endpoint. -/
  upper : Dyadic

/-- Compute both square-root endpoints from one parity normalization and integer square root. -/
def sqrtDyadicBracket (d : Dyadic) : SqrtDyadicBracket :=
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
  let endpointExp := expHalf - Int.ofNat p
  { lower := { sign := false, mant := q, exp := endpointExp }
    upper :=
      { sign := false
        mant := if r == 0 then q else q + 1
        exp := endpointExp } }

/-- Square root rounded toward `-∞`; negative finite inputs produce NaN. -/
def sqrtDown (x : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN1 x with
  | some nan => nan
  | none =>
      if isInf x then
        if signBit x then canonicalNaN else posInf
      else if isZero x then
        x
      else if signBit x then
        canonicalNaN
      else
        match toDyadic? x with
        | some d => roundDyadicDown (sqrtDyadicBracket d).lower
        | none => canonicalNaN

/-- Square root rounded toward `+∞`; negative finite inputs produce NaN. -/
def sqrtUp (x : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN1 x with
  | some nan => nan
  | none =>
      if isInf x then
        if signBit x then canonicalNaN else posInf
      else if isZero x then
        x
      else if signBit x then
        canonicalNaN
      else
        match toDyadic? x with
        | some d => roundDyadicUp (sqrtDyadicBracket d).upper
        | none => canonicalNaN

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
              let (num, den) := scaleRatByPow2 dx.mant dy.mant eDiff
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
              let (num, den) := scaleRatByPow2 dx.mant dy.mant eDiff
              roundRatUp sign num den
        | _, _ => canonicalNaN


end IEEE32Exec

end TorchLean.Floats.IEEE754
