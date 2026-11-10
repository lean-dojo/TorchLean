/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Rounding.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Rational.Conversion
public import FloatLib.Floats.Formats.IEEE754.Native.Representation
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.CoreProof
public import TorchLean.Tensor.Numeric.Widening -- shake: keep

/-!
# Interchange words, host formats and arithmetic for nested quotient rules

`Word α` maps each value of a binary scalar type to one word of `Word.width α` bits, and every value
is the value of its word. The words of `Float` and `Float32` are their native bits and those of a
configured FloatLib format are its complete encodings. Lean's model of a native float keeps no NaN
payload, so a native NaN word need not be the word of its value.

`HostFloat α` reads the words of `Float` and `Float32` as FloatLib models in binary64 and binary32
and adds C's `round`, `floor` and `ceil`, so one definition of each host operation serves both
types. The word operations `Word.ofNatBits`, `Word.succ` and `Word.pred` and the sign operations
`HostFloat.signBit` and `HostFloat.copysign` are class fields that default to their `BitVec` and
FloatLib definitions. The `Float` and `Float32` instances compute them with native integer
operations and prove them equal to the defaults.

A quotient rule such as `seed / (2 * root)` first runs in floating point through checked operations,
which record range loss until the complete quotient has been evaluated. If a flag is raised, the
cold path evaluates the original finite coefficient trees over `Rat` and rounds each coefficient
once to the requested format. The same algebra carries the partial interval enclosures of compound
expressions, so its product and quotient rules are written once. Replay cannot recover information
lost before the quotient's inputs were formed.

`Float` and `Float32` take their `NatCast` from this module and their `RatCast` from
`QuotientArithmetic.round`.

## Main definitions

- `Word` is the interchange word of a binary scalar type.
- `HostFloat` is a host binary type with its interchange format and C's rounding functions.
- `Coefficients α n` holds coefficients in `n` commuting square-zero variables.
- `QuotientArithmetic` is the checked arithmetic of one complete quotient, with byte flags for
  range loss and the coefficient conversions of its cold path. It has instances for `Float`,
  `Float32` and the configured IEEE formats (`QuotientArithmetic.configured`).
- `QuotientArithmetic.round` rounds a rational once to a host format, when `NatCast`, `Neg` and
  `Div` are the format's IEEE 754 conversion, negation and division.

## Main statements

- `Numeric.bits_copySign`: in a format with signed zeros, the word of
  `Model.copySign a b` is the word of `a` with the sign bit of `b`.
-/

@[expose] public section

namespace TorchLean.Numeric

/-- The interchange word of a binary scalar type (IEEE 754-2019 3.4): one word of `width` bits per
value, from which the value is recovered. `ofNatBits`, `succ` and `pred` default to their `BitVec`
formulas, and an instance that computes them otherwise proves its version equal. -/
class Word (α : Type) where
  /-- The width of a word. -/
  width : Nat
  /-- The word of a value. -/
  toBits : α → BitVec width
  /-- The value of a word. -/
  ofBits : BitVec width → α
  /-- A value is the value of its word. -/
  ofBits_toBits : ∀ x, ofBits (toBits x) = x
  /-- The value of the word of the natural number `n`, taken modulo `2 ^ width`. -/
  ofNatBits (n : Nat) : α := ofBits (BitVec.ofNat width n)
  /-- `ofNatBits n` is `ofBits (BitVec.ofNat width n)`. -/
  ofNatBits_eq : ∀ n, ofNatBits n = ofBits (BitVec.ofNat width n) := by intros; rfl
  /-- The value of the next word: the word of `x` plus one, modulo `2 ^ width`. -/
  succ (x : α) : α := ofBits (toBits x + 1)
  /-- `succ x` is `ofBits (toBits x + 1)`. -/
  succ_eq : ∀ x, succ x = ofBits (toBits x + 1) := by intros; rfl
  /-- The value of the previous word: the word of `x` minus one, modulo `2 ^ width`. -/
  pred (x : α) : α := ofBits (toBits x - 1)
  /-- `pred x` is `ofBits (toBits x - 1)`. -/
  pred_eq : ∀ x, pred x = ofBits (toBits x - 1) := by intros; rfl

/-- The word of a `Float` is its binary64 encoding, with `ofNatBits`, `succ` and `pred` in `UInt64`
arithmetic. -/
@[inline] instance : Word Float where
  width := 64
  toBits x := x.toBits.toBitVec
  ofBits b := Float.ofBits ⟨b⟩
  ofBits_toBits := FloatLib.Floats.ExecFloat.Binary.toFloat_ofFloat
  ofNatBits n := Float.ofBits n.toUInt64
  ofNatBits_eq _ := rfl
  succ x := Float.ofBits (x.toBits + 1)
  succ_eq _ := rfl
  pred x := Float.ofBits (x.toBits - 1)
  pred_eq _ := rfl

/-- The word of a `Float32` is its binary32 encoding, with `ofNatBits`, `succ` and `pred` in
`UInt32` arithmetic. -/
@[inline] instance : Word Float32 where
  width := 32
  toBits x := x.toBits.toBitVec
  ofBits b := Float32.ofBits ⟨b⟩
  ofBits_toBits := FloatLib.Floats.ExecFloat.Binary.toFloat32_ofFloat32
  ofNatBits n := Float32.ofBits n.toUInt32
  ofNatBits_eq _ := rfl
  succ x := Float32.ofBits (x.toBits + 1)
  succ_eq _ := rfl
  pred x := Float32.ofBits (x.toBits - 1)
  pred_eq _ := rfl

section ConfiguredWord

open FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats (ExecFloat)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

/-- The word of a configured FloatLib value is its complete encoding of `format.bitWidth` bits. -/
instance Word.configured : Word (ExecFloat (Configured.Family format code plan)) where
  width := format.bitWidth
  toBits x := (ExecFloat.Binary.toModel x).bits
  ofBits b := ExecFloat.Binary.ofModel ⟨b⟩
  ofBits_toBits := ExecFloat.Binary.ofModel_toModel

end ConfiguredWord

section

open FloatLib.Floats.Formats.BinaryInterchange

/-- In a format with signed zeros, the word of `Model.copySign a b` is the word of `a` with the
sign bit of `b`. -/
theorem bits_copySign {fmt : FloatFormat}
    (h : fmt.supportsSignedZero = true) (a b : Model fmt) :
    (Model.copySign a b).bits = (a.bits &&& ~~~BitVec.twoPow _ (fmt.bitWidth - 1)) |||
      (b.bits &&& BitVec.twoPow _ (fmt.bitWidth - 1)) := by
  ext i hi
  by_cases hsign : i = fmt.bitWidth - 1
  · subst hsign
    simpa [Model.signBit_eq_msb, BitVec.msb_eq_getLsbD_last, BitVec.getLsbD_eq_getElem hi] using
      Model.signBit_copySign_of_supportsSignedZero a b h
  · have : i < fmt.fracWidth + fmt.expWidth := by unfold FloatFormat.bitWidth at hi hsign; omega
    have field {x y : Model fmt} (hexp : x.expField = y.expField)
        (hfrac : x.fracField = y.fracField) : x.bits[i] = y.bits[i] := by
      by_cases hlow : i < fmt.fracWidth
      · simpa [Model.fracField, FloatFormat.fracMask, FloatFormat.ofWordNat,
          FloatFormat.fracMaskNat, BitVec.getElem_eq_testBit_toNat, hlow, hi] using
          congrArg (·.testBit i) hfrac
      · simpa [Model.expField, FloatFormat.expAllOnes, FloatFormat.ofWordNat,
          FloatFormat.expAllOnesNat, BitVec.getElem_eq_testBit_toNat,
          Nat.add_sub_of_le (Nat.not_lt.1 hlow), show i - fmt.fracWidth < fmt.expWidth by omega,
          show i - fmt.fracWidth < fmt.bitWidth by omega] using
          congrArg (·.testBit (i - fmt.fracWidth)) hexp
    simpa [hsign] using field (Model.expField_copySign a b) (Model.fracField_copySign a b)

/-- A host binary floating-point type: `Word` encodings of the interchange format `format`, with
C's `round`, `floor` and `ceil`. `signBit` and `copysign` default to FloatLib's `Model.signBit` and
`Model.copySign`, and an instance that computes them otherwise proves its version equal. -/
class HostFloat (α : Type) extends Word α where
  /-- The interchange format of the words. -/
  format : FloatLib.Floats.Formats.BinaryInterchange.FloatFormat
  /-- A word has the width of the format. -/
  width_eq : width = format.bitWidth
  /-- C's `round`: the nearest integral value, a tie away from zero. -/
  roundAway : α → α
  /-- C's `floor`: the largest integral value not above the argument. -/
  floor : α → α
  /-- C's `ceil`: the smallest integral value not below the argument. -/
  ceil : α → α
  /-- The sign bit of `x`. -/
  signBit (x : α) : Bool := (⟨(toBits x).cast width_eq⟩ : Model format).signBit
  /-- `signBit x` is the sign bit of the FloatLib model of `x`. -/
  signBit_eq : ∀ x, signBit x = (⟨(toBits x).cast width_eq⟩ : Model format).signBit := by
    intros; rfl
  /-- C's `copysign`: the magnitude of `x` with the sign bit of `y`. A NaN result is Lean's
  canonical NaN, `0x7FF8000000000000` in `Float` and `0x7FC00000` in `Float32`, whatever the sign
  of `y`. Lean 4.35.0's bit conversions in `object.cpp` (`lean_float_of_bits` and the rest) map
  every NaN to that quiet NaN, and so drop the sign that C23's `copysign` takes from `y`
  (7.12.11.1 of the draft WG14 N3096). -/
  copysign (x y : α) : α := ofBits ((Model.copySign (⟨(toBits x).cast width_eq⟩ : Model format)
    ⟨(toBits y).cast width_eq⟩).bits.cast width_eq.symm)
  /-- `copysign x y` is FloatLib's `Model.copySign` of the models of `x` and `y`. -/
  copysign_eq : ∀ x y, copysign x y = ofBits ((Model.copySign
    (⟨(toBits x).cast width_eq⟩ : Model format) ⟨(toBits y).cast width_eq⟩).bits.cast
      width_eq.symm) := by intros; rfl

/-- `Float` is binary64, with the C library's `round`, `floor` and `ceil`, and `signBit` and
`copysign` by masks of the `UInt64` word. -/
@[inline] instance : HostFloat Float where
  toWord := inferInstance
  format := .binary64
  width_eq := rfl
  roundAway := Float.round
  floor := Float.floor
  ceil := Float.ceil
  signBit x := x.toBits &&& 0x8000000000000000 != 0
  signBit_eq _ := rfl
  copysign x y :=
    Float.ofBits ((x.toBits &&& 0x7FFFFFFFFFFFFFFF) ||| (y.toBits &&& 0x8000000000000000))
  copysign_eq _ _ := by rw [bits_copySign rfl]; rfl

/-- `Float32` is binary32, with the C library's `roundf`, `floorf` and `ceilf`, and `signBit` and
`copysign` by masks of the `UInt32` word. -/
@[inline] instance : HostFloat Float32 where
  toWord := inferInstance
  format := .binary32
  width_eq := rfl
  roundAway := Float32.round
  floor := Float32.floor
  ceil := Float32.ceil
  signBit x := x.toBits &&& 0x80000000 != 0
  signBit_eq _ := rfl
  copysign x y := Float32.ofBits ((x.toBits &&& 0x7FFFFFFF) ||| (y.toBits &&& 0x80000000))
  copysign_eq _ _ := by rw [bits_copySign rfl]; rfl

end

namespace HostFloat

open FloatLib.Floats.Formats.BinaryInterchange

variable {α : Type} [HostFloat α]

/-- The FloatLib model of a host value: its word read in the format `HostFloat.format α`. -/
@[inline] def toModel (x : α) : Model (HostFloat.format α) :=
  ⟨(Word.toBits x).cast (HostFloat.width_eq (α := α))⟩

/-- The host value of a FloatLib model word. -/
@[inline] def ofModel (m : Model (HostFloat.format α)) : α :=
  Word.ofBits (m.bits.cast (HostFloat.width_eq (α := α)).symm)

/-- `+∞` in the host format. -/
@[inline] def posInf : α := ofModel (Model.posInf _)

/-- `-∞` in the host format. -/
@[inline] def negInf : α := ofModel (Model.negInf _)

end HostFloat

/-- Coefficients in `n` commuting square-zero variables, with a chosen scalar carrier. -/
inductive Coefficients (α : Type) : Nat → Type where
  /-- A tree in no variables, one scalar. -/
  | scalar : α → Coefficients α 0
  /-- `dual x dx` is `x + dx ε` for one more square-zero variable `ε`. -/
  | dual {n : Nat} :
      Coefficients α n → Coefficients α n → Coefficients α (n + 1)
  deriving BEq, ReflBEq, LawfulBEq

/-- Coefficient trees with `Rat` leaves by default, as exact replay uses them. Enclosure replay
chooses another leaf carrier. -/
abbrev QuotientCoefficients (n : Nat) (α : Type := Rat) := Coefficients α n

namespace QuotientCoefficients

/-- Constant coefficient, obtained by setting all square-zero variables to zero. -/
def primal {α : Type} : {n : Nat} → QuotientCoefficients n α → α
  | _, .scalar x => x
  | _, .dual x _ => primal x

/-- Add corresponding coefficients using the chosen leaf arithmetic. -/
def add {α : Type} [Add α] : {n : Nat} → QuotientCoefficients n α →
    QuotientCoefficients n α → QuotientCoefficients n α
  | _, .scalar x, .scalar y => .scalar (x + y)
  | _, .dual x dx, .dual y dy => .dual (add x y) (add dx dy)

/-- Subtract corresponding coefficients using the chosen leaf arithmetic. -/
def sub {α : Type} [Sub α] : {n : Nat} → QuotientCoefficients n α →
    QuotientCoefficients n α → QuotientCoefficients n α
  | _, .scalar x, .scalar y => .scalar (x - y)
  | _, .dual x dx, .dual y dy => .dual (sub x y) (sub dx dy)

/-- Multiply coefficient trees using the product rule at each square-zero extension. -/
def mul {α : Type} [Add α] [Mul α] : {n : Nat} → QuotientCoefficients n α →
    QuotientCoefficients n α → QuotientCoefficients n α
  | _, .scalar x, .scalar y => .scalar (x * y)
  | _, .dual x dx, .dual y dy => .dual (mul x y) (add (mul dx y) (mul x dy))

/-- Divide coefficient trees by the quotient rule. Rational callers check the primal denominator
first, and partial enclosure arithmetic propagates a failed or zero-crossing divisor at each leaf.
-/
def div {α : Type} [Add α] [Sub α] [Mul α] [Div α] : {n : Nat} →
    QuotientCoefficients n α → QuotientCoefficients n α → QuotientCoefficients n α
  | _, .scalar x, .scalar y => .scalar (x / y)
  | _, .dual x dx, .dual y dy =>
      let q := div x y
      .dual q (div (sub dx (mul q dy)) y)

/-- Change the leaf representation without changing the differentiation directions. -/
def map {α β : Type} (f : α → β) : {n : Nat} →
    QuotientCoefficients n α → QuotientCoefficients n β
  | _, .scalar x => .scalar (f x)
  | _, .dual x dx => .dual (map f x) (map f dx)

/-- Convert every coefficient, propagating a failed conversion. -/
def mapM {α β : Type} {m : Type → Type} [Monad m] (f : α → m β) : {n : Nat} →
    QuotientCoefficients n α → m (QuotientCoefficients n β)
  | _, .scalar x => return .scalar (← f x)
  | _, .dual x dx => return .dual (← mapM f x) (← mapM f dx)

/-- A constant has zero coefficients in every differentiation direction. -/
def ofScalar {α : Type} [Zero α] : (n : Nat) → α → QuotientCoefficients n α
  | 0, x => .scalar x
  | n + 1, x => .dual (ofScalar n x) (ofScalar n 0)

/-- Lift a scalar exponential through the coefficient product rule, with derivative
`exp x * dx`. -/
def exp {α : Type} [Add α] [Mul α] (evaluate : α → α) : {n : Nat} →
    QuotientCoefficients n α → QuotientCoefficients n α
  | _, .scalar x => .scalar (evaluate x)
  | _, .dual x dx =>
      let value := exp evaluate x
      .dual value (mul value dx)

/-- The primal coefficient of a sum. -/
theorem primal_add {α : Type} [Add α] {n : Nat} (left right : QuotientCoefficients n α) :
    primal (add left right) = primal left + primal right := by
  induction n with
  | zero => cases left; cases right; rfl
  | succ n ih =>
      cases left with | dual left leftDirection =>
        cases right with | dual right rightDirection => exact ih left right

/-- The primal coefficient of a difference. -/
theorem primal_sub {α : Type} [Sub α] {n : Nat} (left right : QuotientCoefficients n α) :
    primal (sub left right) = primal left - primal right := by
  induction n with
  | zero => cases left; cases right; rfl
  | succ n ih =>
      cases left with | dual left leftDirection =>
        cases right with | dual right rightDirection => exact ih left right

/-- The primal coefficient of a product. -/
theorem primal_mul {α : Type} [Add α] [Mul α] {n : Nat}
    (left right : QuotientCoefficients n α) :
    primal (mul left right) = primal left * primal right := by
  induction n with
  | zero => cases left; cases right; rfl
  | succ n ih =>
      cases left with | dual left leftDirection =>
        cases right with | dual right rightDirection => exact ih left right

/-- The primal coefficient of a quotient. -/
theorem primal_div {α : Type} [Add α] [Sub α] [Mul α] [Div α] {n : Nat}
    (left right : QuotientCoefficients n α) :
    primal (div left right) = primal left / primal right := by
  induction n with
  | zero => cases left; cases right; rfl
  | succ n ih =>
      cases left with | dual left leftDirection =>
        cases right with | dual right rightDirection => exact ih left right

/-- Mapping the leaf values commutes with taking the primal coefficient. -/
theorem primal_map {α β : Type} (function : α → β) {n : Nat}
    (coefficients : QuotientCoefficients n α) :
    primal (map function coefficients) = function (primal coefficients) := by
  induction n with
  | zero => cases coefficients; rfl
  | succ n ih =>
      cases coefficients with | dual coefficients direction => exact ih coefficients

end QuotientCoefficients

/--
Checked arithmetic for one complete quotient, together with conversion to and from finite
coefficient trees.

Each operation returns its value and a byte of flags that records a nonfinite or subnormal
result, including a nonzero product or quotient rounded to zero. Zero means no range loss, and
recursive carriers combine flags with bitwise OR. `divChecked` turns the flags into one Boolean
at the end, so internal nodes never convert. Unsupported carriers use the ordinary operations and
supply no conversion.
-/
class QuotientArithmetic (α : Type) where
  /-- Whether the carrier supplies checked operations and a conversion. -/
  supported : Bool := false
  /-- The depth of the coefficient trees that `encode` and `decode` convert. -/
  depth : Nat := 0
  /-- Addition with its range flags. -/
  addWithFlags : α → α → α × UInt8
  /-- Subtraction with its range flags. -/
  subWithFlags : α → α → α × UInt8
  /-- Multiplication with its range flags. -/
  mulWithFlags : α → α → α × UInt8
  /-- Division with its range flags. -/
  divWithFlags : α → α → α × UInt8
  /-- The finite coefficient tree of a value, if it has one. -/
  encode : α → Option (QuotientCoefficients depth) := fun _ => none
  /-- The value of a finite coefficient tree, if it has one. -/
  decode : QuotientCoefficients depth → Option α := fun _ => none
  /-- Copy the scalar primal's native sign of zero after replaying derivative coefficients. -/
  copyPrimal : α → α → α := fun _ y => y

/-- The ordinary operations with no flags and no conversion, for unsupported carriers. -/
instance (priority := low) QuotientArithmetic.unsupported {α : Type} [Add α] [Sub α] [Mul α]
    [Div α] : QuotientArithmetic α where
  addWithFlags x y := (x + y, 0)
  subWithFlags x y := (x - y, 0)
  mulWithFlags x y := (x * y, 0)
  divWithFlags x y := (x / y, 0)

/-- `x / y` with one flag, set when the division raised a range flag. -/
@[inline] def QuotientArithmetic.divChecked {α : Type} [ops : QuotientArithmetic α]
    (x y : α) : α × Bool :=
  let (value, flags) := ops.divWithFlags x y
  (value, 0 < flags)

/--
Exact replay of `x / y` over finite coefficient trees, or `none` when an input has no finite
encoding, the primal denominator is zero or decoding fails.

Only coefficient conversion crosses this cold boundary, which leaves the checked arithmetic
operations available for specialization in the caller.
-/
@[noinline] def QuotientArithmetic.replay? {α : Type} {depth : Nat}
    (encode : α → Option (QuotientCoefficients depth))
    (decode : QuotientCoefficients depth → Option α) (x y : α) : Option α := do
  let a ← encode x
  let b ← encode y
  if b.primal == 0 then none else decode (QuotientCoefficients.div a b)

/-- Native constants, selected by the scalar type. -/
class NativeFormat (α : Type) where
  /-- The native zero. -/
  zero : α
  /-- The smallest positive normal value. -/
  minNormal : α
  /-- The negation of `minNormal`. -/
  negMinNormal : α

-- Initialize the native constants once, outside all checked operations.
/-- The binary64 constants, from their bits. -/
@[noinline] instance : NativeFormat Float where
  zero := Word.ofBits 0
  minNormal := Word.ofBits 0x0010000000000000
  negMinNormal := Word.ofBits 0x8010000000000000

/-- The binary32 constants, from their bits. -/
@[noinline] instance : NativeFormat Float32 where
  zero := Word.ofBits 0
  minNormal := Word.ofBits 0x00800000
  negMinNormal := Word.ofBits 0x80800000

/-- The rational `value` rounded once to the host format, to nearest with ties to even, when
`NatCast`, `Neg` and `Div` are the format's IEEE 754 conversion, negation and division. -/
def QuotientArithmetic.round {α : Type} [HostFloat α] [NatCast α] [Neg α] [Div α]
    (value : Rat) : α :=
  -- A numerator and denominator of magnitude at most `2^p`, for the precision `p`, are exact, so
  -- their native quotient rounds the value once. Larger ones could round twice or overflow to
  -- `∞ / ∞`, so FloatLib rounds the exact rational instead.
  let bound := 2 ^ ((HostFloat.format α).fracWidth + 1)
  if value.num.natAbs ≤ bound && value.den ≤ bound then
    let quotient := (value.num.natAbs : α) / (value.den : α)
    if value.num < 0 then -quotient else quotient
  else
    HostFloat.ofModel
      (FloatLib.Floats.Formats.BinaryInterchange.Model.ofRat (HostFloat.format α) value)

namespace NativeFormat

variable {α : Type} [NativeFormat α] [BEq α] [LT α] [DecidableLT α] [Sub α]

/-- One when `x == NativeFormat.zero`, and zero otherwise. -/
@[inline] def zeroFlag (x : α) : UInt8 := (x == NativeFormat.zero).toUInt8

-- `x - x` is zero for finite `x` and NaN otherwise, and comparing that difference with itself
-- detects NaN without a floating zero constant.
/-- One when `x` is not finite, and zero otherwise. -/
@[inline] def nonfiniteFlag (x : α) : UInt8 :=
  let difference := x - x
  (difference == difference).toUInt8 ^^^ 1

-- Only predicates cross this boundary, and value-producing arithmetic stays inline.
-- The open interval (-minNormal, minNormal) holds only finite values, and excluding zero from it
-- leaves the subnormals, so its branch needs a zero check. Outside it, only nonfiniteness can
-- raise a flag.
/-- The range flags of a native sum or difference `z`: one when `z` is subnormal (nonzero by
`zero` and strictly between `negMinNormal` and `minNormal`) or not finite, and zero otherwise. -/
@[noinline, specialize] def addSubFlags (zero : α → UInt8) (z : α) : UInt8 :=
  if ((decide (NativeFormat.negMinNormal < z)).toUInt8 &&&
      (decide (z < NativeFormat.minNormal)).toUInt8) == 1 then
    zero z ^^^ 1
  else nonfiniteFlag z

/-- Range flags for the native product `z = x * y`, with native zero equality. Inside the open
subnormal interval, which includes zero, the flag is raised when both inputs are nonzero, so a
product rounded to zero is caught. `z` must be the actual product of `x` and `y`. -/
@[noinline, specialize] def productFlags (zero : α → UInt8) (x y z : α) : UInt8 :=
  if ((decide (NativeFormat.negMinNormal < z)).toUInt8 &&&
      (decide (z < NativeFormat.minNormal)).toUInt8) == 1 then
    (zero x ^^^ 1) &&& (zero y ^^^ 1)
  else nonfiniteFlag z

/-- Range flags for the actual native quotient `z = x / y`, using native zero equality.
A zero denominator already makes the result nonfinite. A zero or subnormal result with a
nonzero numerator requires replay. This formula assumes `z` is the actual division result. -/
@[noinline, specialize] def quotientFlags (zero : α → UInt8) (x z : α) : UInt8 :=
  if ((decide (NativeFormat.negMinNormal < z)).toUInt8 &&&
      (decide (z < NativeFormat.minNormal)).toUInt8) == 1 then
    zero x ^^^ 1
  else nonfiniteFlag z

/-- The native sum `x + y` with its range flags. -/
@[inline] def addWithFlags [Add α] (x y : α) : α × UInt8 :=
  let z := x + y
  (z, addSubFlags zeroFlag z)

/-- The native difference `x - y` with its range flags. -/
@[inline] def subWithFlags (x y : α) : α × UInt8 :=
  let z := x - y
  (z, addSubFlags zeroFlag z)

/-- The native product `x * y` with its `productFlags`. -/
@[inline] def mulWithFlags [Mul α] (x y : α) : α × UInt8 :=
  let z := x * y
  (z, productFlags zeroFlag x y z)

/-- The native quotient `x / y` with its `quotientFlags`. -/
@[inline] def divWithFlags [Div α] (x y : α) : α × UInt8 :=
  let z := x / y
  (z, quotientFlags zeroFlag x z)

end NativeFormat

/-- Checked native arithmetic, encoding a finite value as its exact rational. -/
@[inline] instance QuotientArithmetic.native {α : Type} [NativeFormat α] [HostFloat α] [BEq α]
    [LT α] [DecidableLT α] [Add α] [Sub α] [Mul α] [Div α] [NatCast α] [Neg α] :
    QuotientArithmetic α where
  supported := true
  addWithFlags x y := NativeFormat.addWithFlags x y
  subWithFlags x y := NativeFormat.subWithFlags x y
  mulWithFlags x y := NativeFormat.mulWithFlags x y
  divWithFlags x y := NativeFormat.divWithFlags x y
  encode x := (FloatLib.Floats.Formats.BinaryInterchange.Model.toRat?
    (HostFloat.toModel x)).map Coefficients.scalar
  decode | .scalar x => some (QuotientArithmetic.round x)
  copyPrimal x _ := x

section Configured

open FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats (ExecFloat)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/--
FloatLib's IEEE exception status, joined with the stricter range test a nested quotient needs.

The IEEE underflow flag is raised only for an inexact tiny result, but an exact subnormal
intermediate must also trigger replay, because later products and quotients can lose its
coefficients. Ordinary inexact rounding alone does not trigger replay. In binary32,
`2^-70 * 2^-70` is the exact subnormal `2^-140`: IEEE raises no flag there, and this byte is set.
-/
def QuotientArithmetic.configuredFlags (outcome : ExecFloat.Binary.IEEEOutcome
    (format := format) (plan := plan) (code := code)) : UInt8 :=
  let value := outcome.value
  let status := outcome.status
  (status.invalid || status.divideByZero || status.overflow || status.underflow ||
    !ExecFloat.Binary.isFinite value || ExecFloat.Binary.isSubnormal value).toUInt8

/--
Configured values with IEEE exceptional encoding use FloatLib's nearest-even operations and
exact finite decoder, including custom exponent biases.
Status is computed on the descriptor model and repacked through the codec, independently of
the selected arithmetic backend.

The complete derivative tree is replayed by the recursive `Dual` instance. Every coefficient is
rounded independently to the original descriptor and storage plan, including formats wider than
binary64. Finite-only encodings are not enabled: their saturation policy would require a separate
derivative contract. True coefficients outside the IEEE range may still round to zero or infinity.
-/
instance QuotientArithmetic.configured : QuotientArithmetic Value where
  supported := format.encoding == .ieee
  addWithFlags x y :=
    let outcome := ExecFloat.Binary.IEEEOutcome.ofModel <|
      Model.add (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel y) (status := true)
    (outcome.value, configuredFlags outcome)
  subWithFlags x y :=
    let outcome := ExecFloat.Binary.IEEEOutcome.ofModel <|
      Model.sub (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel y) (status := true)
    (outcome.value, configuredFlags outcome)
  mulWithFlags x y :=
    let outcome := ExecFloat.Binary.IEEEOutcome.ofModel <|
      Model.mul (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel y) (status := true)
    let lostZero := ExecFloat.Binary.isZero outcome.value &&
      !ExecFloat.Binary.isZero x && !ExecFloat.Binary.isZero y
    (outcome.value, configuredFlags outcome ||| lostZero.toUInt8)
  divWithFlags x y :=
    let outcome := ExecFloat.Binary.IEEEOutcome.ofModel <|
      Model.div (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel y) (status := true)
    let lostZero := ExecFloat.Binary.isZero outcome.value && !ExecFloat.Binary.isZero x
    (outcome.value, configuredFlags outcome ||| lostZero.toUInt8)
  encode x := (ExecFloat.Binary.toRat? x).map Coefficients.scalar
  decode
    | .scalar x => some <| ExecFloat.Binary.ofModel <| Model.ofRat format x
  copyPrimal x _ := x

end Configured

end TorchLean.Numeric

namespace TorchLean

/-- Round a natural once to binary64. -/
instance instNatCastFloat : NatCast Float where
  natCast := Float.ofNat

/-- Round a natural once to binary32. -/
instance instNatCastFloat32 : NatCast Float32 where
  natCast := Float32.ofNat

/-- Round a rational once to binary64. -/
instance instRatCastFloat : RatCast Float where
  ratCast value := Numeric.QuotientArithmetic.round value

/-- Round a rational once to binary32, without an intermediate binary64 rounding. For
`1 + 2^-24 + 2^-60` that gives `1 + 2^-23`, where a binary64 step would stop at the tie
`1 + 2^-24`, which then rounds to even, `1`. -/
instance instRatCastFloat32 : RatCast Float32 where
  ratCast value := Numeric.QuotientArithmetic.round value

end TorchLean
