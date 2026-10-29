/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Rounding.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Rational.Conversion

/-!
# Arithmetic for nested quotient rules

Checked operations retain range-loss information until the complete quotient has been evaluated.
The cold path evaluates the original finite coefficient trees over `Rat`, then rounds each
coefficient through FloatLib to the requested format. The same algebra can carry partial
enclosures for compound expressions without duplicating its product and quotient rules.
Quotient replay alone cannot recover information lost before its inputs were formed.
-/

@[expose] public section

namespace TorchLean.Numeric

/-- Coefficients in `n` commuting square-zero variables, with a chosen scalar carrier. -/
inductive Coefficients (α : Type) : Nat → Type where
  | scalar : α → Coefficients α 0
  | dual {n : Nat} :
      Coefficients α n → Coefficients α n → Coefficients α (n + 1)
  deriving BEq

/-- Finite rational coefficients by default; enclosure replay can choose another leaf carrier. -/
abbrev QuotientCoefficients (n : Nat) (α : Type := Rat) := Coefficients α n

namespace QuotientCoefficients

export Coefficients (scalar dual)

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

/-- Divide coefficient trees. Rational callers check the primal denominator first; partial
enclosure arithmetic propagates a failed or zero-crossing divisor at each leaf. -/
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

/-- Lift scalar exponential evaluation through the existing coefficient product rule. -/
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
Operations used inside one complete quotient, together with finite coefficient conversion.

Internal byte flags record nonfinite or subnormal results, including a nonzero product or quotient
rounded to zero. Zero means no range loss; recursive carriers combine flags with bitwise OR.
The checked-operation entrypoints expose the same Boolean result without converting at each
internal node. Unsupported carriers use the ordinary operations and do not supply a conversion.
-/
class QuotientArithmetic (α : Type) where
  supported : Bool := false
  depth : Nat := 0
  addWithFlags : α → α → α × UInt8
  subWithFlags : α → α → α × UInt8
  mulWithFlags : α → α → α × UInt8
  divWithFlags : α → α → α × UInt8
  encode : α → Option (QuotientCoefficients depth) := fun _ => none
  decode : QuotientCoefficients depth → Option α := fun _ => none
  /-- Copy the scalar primal's native sign of zero after replaying derivative coefficients. -/
  copyPrimal : α → α → α := fun _ y => y

instance (priority := low) {α : Type} [Add α] [Sub α] [Mul α] [Div α] :
    QuotientArithmetic α where
  addWithFlags x y := (x + y, 0)
  subWithFlags x y := (x - y, 0)
  mulWithFlags x y := (x * y, 0)
  divWithFlags x y := (x / y, 0)

namespace QuotientArithmetic

@[inline] def divChecked {α : Type} [ops : QuotientArithmetic α] (x y : α) : α × Bool :=
  let (value, flags) := ops.divWithFlags x y
  (value, 0 < flags)

/--
Replay only finite input trees with an invertible denominator.

Only coefficient conversion crosses this cold boundary, leaving the checked arithmetic
operations available for specialization in the caller.
-/
@[noinline] def replay? {α : Type} {depth : Nat}
    (encode : α → Option (QuotientCoefficients depth))
    (decode : QuotientCoefficients depth → Option α) (x y : α) : Option α := do
  let a ← encode x
  let b ← encode y
  if b.primal == 0 then none else decode (QuotientCoefficients.div a b)

/-- Native constants and rational rounding, selected by the scalar type. -/
class NativeFormat (α : Type) where
  zero : α
  minNormal : α
  negMinNormal : α
  /-- Round a rational once to the native scalar type. -/
  round : Rat → α

-- Initialize the native constants once, outside all checked operations.
@[noinline] instance : NativeFormat Float where
  zero := Float.ofBits 0
  minNormal := Float.ofBits 0x0010000000000000
  negMinNormal := Float.ofBits 0x8010000000000000
  round value := FloatLib.Floats.ExecFloat.Binary.toFloat <|
    (FloatLib.Floats.ExecFloat.Binary.ofModel
      (FloatLib.Floats.Formats.BinaryInterchange.Model.ofRat
        FloatLib.Floats.Formats.BinaryInterchange.FloatFormat.binary64 value) :
      FloatLib.Floats.Binary64)

@[noinline] instance : NativeFormat Float32 where
  zero := Float32.ofBits 0
  minNormal := Float32.ofBits 0x00800000
  negMinNormal := Float32.ofBits 0x80800000
  round value := FloatLib.Floats.ExecFloat.Binary.toFloat32 <|
    (FloatLib.Floats.ExecFloat.Binary.ofModel
      (FloatLib.Floats.Formats.BinaryInterchange.Model.ofRat
        FloatLib.Floats.Formats.BinaryInterchange.FloatFormat.binary32 value) :
      FloatLib.Floats.Binary32)

export NativeFormat (round)

namespace Native

variable {α : Type} [NativeFormat α] [BEq α] [LT α] [DecidableLT α] [Sub α]

@[inline] def zeroFlag (x : α) : UInt8 := (x == NativeFormat.zero).toUInt8

-- A finite self-difference is zero; a nonfinite self-difference is NaN.
-- Reflexive equality of that difference detects NaN without a floating zero constant.
@[inline] def nonfiniteFlag (x : α) : UInt8 :=
  let difference := x - x
  (difference == difference).toUInt8 ^^^ 1

-- Excluding zero from the interval (-minNormal, minNormal) leaves the subnormals.
@[inline] def rangeFlags (zero : α → UInt8) (x : α) : UInt8 :=
  if ((decide (NativeFormat.negMinNormal < x)).toUInt8 &&&
      (decide (x < NativeFormat.minNormal)).toUInt8) == 1 then
    zero x ^^^ 1
  else nonfiniteFlag x

-- Only predicates cross this boundary. Value-producing arithmetic remains inline.
-- The open interval contains only finite values. Its branch needs zero checks;
-- outside it, only nonfiniteness can raise a flag.
@[noinline, specialize] def addSubFlags (zero : α → UInt8) (z : α) : UInt8 :=
  rangeFlags zero z

/-- Range flags for the actual native product `z = x * y`, using native zero equality.
The open subnormal interval includes zero; nonzero inputs detect a product rounded to zero.
This formula requires the actual product, not an independently supplied third value. -/
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

@[inline] def addWithFlags [Add α] (x y : α) : α × UInt8 :=
  let z := x + y
  (z, addSubFlags zeroFlag z)

@[inline] def subWithFlags (x y : α) : α × UInt8 :=
  let z := x - y
  (z, addSubFlags zeroFlag z)

@[inline] def mulWithFlags [Mul α] (x y : α) : α × UInt8 :=
  let z := x * y
  (z, productFlags zeroFlag x y z)

@[inline] def divWithFlags [Div α] (x y : α) : α × UInt8 :=
  let z := x / y
  (z, quotientFlags zeroFlag x z)

end Native

@[inline] instance : QuotientArithmetic Float where
  supported := true
  addWithFlags x y := Native.addWithFlags x y
  subWithFlags x y := Native.subWithFlags x y
  mulWithFlags x y := Native.mulWithFlags x y
  divWithFlags x y := Native.divWithFlags x y
  encode x := (FloatLib.Floats.ExecFloat.Binary.toRat?
    (FloatLib.Floats.ExecFloat.Binary.ofFloat x :
      FloatLib.Floats.Binary64)).map QuotientCoefficients.scalar
  decode | .scalar x => some (round x)
  copyPrimal x _ := x

@[inline] instance : QuotientArithmetic Float32 where
  supported := true
  addWithFlags x y := Native.addWithFlags x y
  subWithFlags x y := Native.subWithFlags x y
  mulWithFlags x y := Native.mulWithFlags x y
  divWithFlags x y := Native.divWithFlags x y
  encode x := (FloatLib.Floats.ExecFloat.Binary.toRat?
    (FloatLib.Floats.ExecFloat.Binary.ofFloat x.toFloat :
      FloatLib.Floats.Binary64)).map QuotientCoefficients.scalar
  decode | .scalar x => some (round x)
  copyPrimal x _ := x

section Configured

open FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats (ExecFloat)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/--
FloatLib exception status plus the stricter range predicate needed by a nested quotient.

IEEE underflow requires inexactness. An exact subnormal intermediate must also trigger replay,
since subsequent multiplication or division can lose coefficients. Ordinary inexact rounding
alone does not trigger replay.
-/
def configuredFlags (outcome : ExecFloat.Binary.IEEEOutcome
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
instance configured : QuotientArithmetic Value where
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
  encode x := (ExecFloat.Binary.toRat? x).map QuotientCoefficients.scalar
  decode
    | .scalar x => some <| ExecFloat.Binary.ofModel <| Model.ofRat format x
  copyPrimal x _ := x

end Configured

end QuotientArithmetic
end TorchLean.Numeric
