/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32.Dyadic
public import Mathlib.Data.Nat.Basic
import Mathlib.Tactic.Attr.Core
import Mathlib.Tactic.Push

/-!
# IEEE32 Executable Arithmetic

This file defines executable binary32 arithmetic for the core operations: addition, subtraction,
multiplication, division, square root, and fused multiply-add. The implementation follows IEEE-style
case splits for NaN/Inf/zero and delegates finite rounding to the dyadic rounding kernel.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

/-- IEEE754 addition (round-to-nearest, ties-to-even), with NaN/Inf rules. -/
@[inline] def add (x y : IEEE32Exec) : IEEE32Exec :=
  match toDyadic? x, toDyadic? y with
  | some dx, some dy => roundDyadicToIEEE32 (addDyadic dx dy)
  | _, _ =>
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
            canonicalNaN

/-- IEEE754 subtraction (defined as addition with sign-flip). -/
@[inline] def sub (x y : IEEE32Exec) : IEEE32Exec :=
  add x (neg y)

/-- IEEE754 multiplication (round-to-nearest, ties-to-even), with NaN/Inf rules. -/
@[inline] def mul (x y : IEEE32Exec) : IEEE32Exec :=
  match toDyadic? x, toDyadic? y with
  | some dx, some dy =>
      let s := Bool.xor dx.sign dy.sign
      if dx.mant == 0 || dy.mant == 0 then
        if s then negZero else posZero
      else
        roundDyadicToIEEE32 { sign := s, mant := dx.mant * dy.mant, exp := dx.exp + dy.exp }
  | _, _ =>
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
            canonicalNaN

/-- IEEE754 division (round-to-nearest, ties-to-even), with NaN/Inf rules. -/
@[inline] def div (x y : IEEE32Exec) : IEEE32Exec :=
  match toDyadic? x, toDyadic? y with
  | some dx, some dy =>
      let sign := Bool.xor dx.sign dy.sign
      if dy.mant == 0 then
        if dx.mant == 0 then canonicalNaN
        else if sign then negInf else posInf
      else if dx.mant == 0 then
        if sign then negZero else posZero
      else
        -- Exact quotient: (mx * 2^ex) / (my * 2^ey) = (mx/my) * 2^(ex-ey).
        let eDiff : Int := dx.exp - dy.exp
        let (num, den) := scaleRatByPow2 dx.mant dy.mant eDiff
        roundRatToIEEE32 sign num den
  | _, _ =>
      match chooseNaN2 x y with
      | some nan => nan
      | none =>
          if isInf x then
            if isInf y then canonicalNaN
            else if signBit x != signBit y then negInf else posInf
          else if isInf y then
            if signBit x != signBit y then negZero else posZero
          else
            canonicalNaN

/-- IEEE754 fused multiply-add: compute `x*y+z` and round once (ties-to-even). -/
@[inline] def fma (x y z : IEEE32Exec) : IEEE32Exec :=
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
  simp [hx, hy]

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
  simp (config := { zeta := true }) [hx, hy]
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
      let (num, den) := scaleRatByPow2 dx.mant dy.mant eDiff
      roundRatToIEEE32 sign num den := by
  unfold div
  simp only [hx, hy]
  split <;> rename_i hden
  · exact (hy0 ((beq_iff_eq).1 hden)).elim
  · split <;> rename_i hnum
    · have hnum' : dx.mant = 0 := (beq_iff_eq).1 hnum
      rw [hnum']
      exact (roundRatToIEEE32_scaleRatByPow2_zero
        (Bool.xor dx.sign dy.sign) dy.mant (dx.exp - dy.exp)).symm
    · rfl

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

/-!
## IEEE exception status

The value-only operations above remain the basic executable API. The wrappers below return the
same value together with the five exception indicators defined by IEEE 754. Tininess is detected
after rounding, and underflow is reported only when the rounded result is both tiny and inexact.

The status is computed from the exact dyadic or rational intermediate already used by each
operation. No host floating-point operation or numerical tolerance enters this classification.
-/

/-- The five exception indicators associated with an IEEE-754 operation. -/
structure IEEEStatus where
  invalid : Bool := false
  divideByZero : Bool := false
  overflow : Bool := false
  underflow : Bool := false
  inexact : Bool := false
  deriving Repr, DecidableEq, Inhabited

namespace IEEEStatus

/-- No exception was raised. -/
def clear : IEEEStatus := {}

/-- Accumulate exception indicators from two operations. -/
def union (a b : IEEEStatus) : IEEEStatus where
  invalid := a.invalid || b.invalid
  divideByZero := a.divideByZero || b.divideByZero
  overflow := a.overflow || b.overflow
  underflow := a.underflow || b.underflow
  inexact := a.inexact || b.inexact

end IEEEStatus

/-- An executable binary32 value paired with the exception status produced while computing it. -/
structure IEEEOutcome where
  value : IEEE32Exec
  status : IEEEStatus
  deriving Repr, DecidableEq, Inhabited

/-- Whether a finite binary32 result is zero or subnormal. -/
def isTinyAfterRounding (x : IEEE32Exec) : Bool :=
  isFinite x && expField x == 0

/-- Classify rounding an exact dyadic to the supplied binary32 result. -/
def dyadicRoundingStatus (exact : Dyadic) (rounded : IEEE32Exec) : IEEEStatus :=
  if isInf rounded then
    { overflow := true, inexact := true }
  else
    match toDyadic? rounded with
    | none => .clear
    | some actual =>
        let inexact := cmpDyadic exact actual != .eq
        { underflow := isTinyAfterRounding rounded && inexact
          inexact := inexact }

/-- Exact equality between `num / den` and a dyadic magnitude, including the result sign. -/
def rationalEqualsDyadic (sign : Bool) (num den : Nat) (d : Dyadic) : Bool :=
  if num == 0 then
    d.mant == 0
  else if d.mant == 0 || sign != d.sign then
    false
  else
    match d.exp with
    | .ofNat e => num == den * Nat.shiftLeft d.mant e
    | .negSucc e => Nat.shiftLeft num (e + 1) == den * d.mant

/-- Classify rounding an exact signed rational to the supplied binary32 result. -/
def rationalRoundingStatus (sign : Bool) (num den : Nat)
    (rounded : IEEE32Exec) : IEEEStatus :=
  if isInf rounded then
    { overflow := true, inexact := true }
  else
    match toDyadic? rounded with
    | none => .clear
    | some actual =>
        let inexact := !(rationalEqualsDyadic sign num den actual)
        { underflow := isTinyAfterRounding rounded && inexact
          inexact := inexact }

/-- Binary32 addition with IEEE exception status. -/
def addWithStatus (x y : IEEE32Exec) : IEEEOutcome :=
  match toDyadic? x, toDyadic? y with
  | some dx, some dy =>
      let exact := addDyadic dx dy
      let value := roundDyadicToIEEE32 exact
      { value, status := dyadicRoundingStatus exact value }
  | _, _ =>
      let value := add x y
      let hasNaN := isNaN x || isNaN y
      let generatedInvalid := isInf x && isInf y && signBit x != signBit y
      let invalid := isSNaN x || isSNaN y || (!hasNaN && generatedInvalid)
      if invalid then
        { value, status := { invalid := true } }
      else
        { value, status := .clear }

/-- Binary32 subtraction with IEEE exception status. -/
def subWithStatus (x y : IEEE32Exec) : IEEEOutcome :=
  addWithStatus x (neg y)

/-- Binary32 multiplication with IEEE exception status. -/
def mulWithStatus (x y : IEEE32Exec) : IEEEOutcome :=
  match toDyadic? x, toDyadic? y with
  | some dx, some dy =>
      let exact : Dyadic :=
        { sign := Bool.xor dx.sign dy.sign
          mant := dx.mant * dy.mant
          exp := dx.exp + dy.exp }
      let value := roundDyadicToIEEE32 exact
      { value, status := dyadicRoundingStatus exact value }
  | _, _ =>
      let value := mul x y
      let hasNaN := isNaN x || isNaN y
      let generatedInvalid := (isInf x && isZero y) || (isInf y && isZero x)
      let invalid := isSNaN x || isSNaN y || (!hasNaN && generatedInvalid)
      if invalid then
        { value, status := { invalid := true } }
      else
        { value, status := .clear }

/-- Binary32 division with IEEE exception status. -/
def divWithStatus (x y : IEEE32Exec) : IEEEOutcome :=
  match toDyadic? x, toDyadic? y with
  | some dx, some dy =>
      let sign := Bool.xor dx.sign dy.sign
      if dy.mant == 0 then
        if dx.mant == 0 then
          { value := canonicalNaN, status := { invalid := true } }
        else
          { value := if sign then negInf else posInf, status := { divideByZero := true } }
      else if dx.mant == 0 then
        { value := if sign then negZero else posZero, status := .clear }
      else
        let eDiff := dx.exp - dy.exp
        let (num, den) := scaleRatByPow2 dx.mant dy.mant eDiff
        let value := roundRatToIEEE32 sign num den
        { value, status := rationalRoundingStatus sign num den value }
  | _, _ =>
      let value := div x y
      let hasNaN := isNaN x || isNaN y
      let generatedInvalid := isInf x && isInf y
      let invalid := isSNaN x || isSNaN y || (!hasNaN && generatedInvalid)
      if invalid then
        { value, status := { invalid := true } }
      else
        { value, status := .clear }

/-- Fused multiply-add with IEEE exception status. -/
def fmaWithStatus (x y z : IEEE32Exec) : IEEEOutcome :=
  match toDyadic? x, toDyadic? y, toDyadic? z with
  | some dx, some dy, some dz =>
      let product : Dyadic :=
        { sign := Bool.xor dx.sign dy.sign
          mant := dx.mant * dy.mant
          exp := dx.exp + dy.exp }
      let exact := addDyadic product dz
      let value := roundDyadicToIEEE32 exact
      { value, status := dyadicRoundingStatus exact value }
  | _, _, _ =>
      let value := fma x y z
      let hasNaN := isNaN x || isNaN y || isNaN z
      let invalidProduct := (isInf x || isInf y) && (isZero x || isZero y)
      let productIsInf := (isInf x || isInf y) && !(isZero x || isZero y)
      let oppositeInfiniteAddend :=
        productIsInf && isInf z && signBit z != Bool.xor (signBit x) (signBit y)
      let generatedInvalid := invalidProduct || oppositeInfiniteAddend
      let invalid := isSNaN x || isSNaN y || isSNaN z || (!hasNaN && generatedInvalid)
      if invalid then
        { value, status := { invalid := true } }
      else
        { value, status := .clear }

/-- Binary32 square root with IEEE exception status. -/
def sqrtWithStatus (x : IEEE32Exec) : IEEEOutcome :=
  let value := sqrt x
  let generatedInvalid := !isNaN x && signBit x && !isZero x
  let invalid := isSNaN x || generatedInvalid
  if invalid then
    { value, status := { invalid := true } }
  else
    match toDyadic? x, toDyadic? value with
    | some exact, some root =>
        let squared : Dyadic :=
          { sign := false, mant := root.mant * root.mant, exp := root.exp + root.exp }
        { value, status := { inexact := cmpDyadic squared exact != .eq } }
    | _, _ => { value, status := .clear }

/-- After-rounding dyadic underflow is reported only for an inexact result. -/
theorem dyadicRoundingStatus_inexact_of_underflow {exact : Dyadic} {rounded : IEEE32Exec}
    (h : (dyadicRoundingStatus exact rounded).underflow = true) :
    (dyadicRoundingStatus exact rounded).inexact = true := by
  unfold dyadicRoundingStatus at h ⊢
  split
  · simp
  · split <;> simp_all [IEEEStatus.clear]

/-- After-rounding rational underflow is reported only for an inexact result. -/
theorem rationalRoundingStatus_inexact_of_underflow {sign : Bool} {num den : Nat}
    {rounded : IEEE32Exec}
    (h : (rationalRoundingStatus sign num den rounded).underflow = true) :
    (rationalRoundingStatus sign num den rounded).inexact = true := by
  unfold rationalRoundingStatus at h ⊢
  split
  · simp
  · split <;> simp_all [IEEEStatus.clear]

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


end IEEE32Exec

end TorchLean.Floats.IEEE754
