/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Numeric
public import TorchLean.Tensor.Numeric.Quotient
public import TorchLean.Tensor.Numeric.InverseTrig
public import TorchLean.Tensor.Op.Pointwise
public import FloatLib.Floats.Formats.BinaryInterchange.Algebraic.Rounding.SqrtRuntime
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Algebraic.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Operations.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.CorrectlyRounded

/-!
# `Context α`: the scalar interface for models and proofs

TorchLean is designed to be *scalar-polymorphic*: the same model/layer definitions can be
instantiated over many numeric backends:

- `Float` (fast binary64 execution with a logical model and a separate native boundary),
- FloatLib configured binary values, with the exponent and fraction widths selected in their type
  (import `TorchLean.Tensor.ConfiguredBinary` for their adapters),
- `Float32`, FloatLib's rounded-real model `NF β fexp rnd` and the scoped rationals `ℚ`,
- complex numbers `TorchLean.Complex α` and the dual numbers `TorchLean.Dual α`, nested as
  `Dual.Nested α n`,
- `ℝ` (proof-level mathematics).

No interval type has a `Context` instance: IBP and CROWN bound tensors by boxes of endpoint
tensors over one of these scalars (`TorchLean.CROWN.Box`).

The scalar parameter serves several purposes:

- We did not want separate "Float model code", "proof model code", and "verification model code"
  that slowly diverge and become inconsistent.
- In practice, we iterate across phases: execute a compact model, state the proof-level contract,
  then run verification bounds. Rewriting each model for each phase is error-prone.
- A scalar-polymorphic specification keeps the layer and model definitions shared while the scalar
  instance determines their numerical meaning.
- Cross-checking happens at the scalar semantics layer, over one shared copy of each model.
- `Context α` is larger than a minimal arithmetic interface, but avoids duplicating architectures
  across execution, proof, and verification code.

Related work:

- Bezanson et al., "Julia: A Fresh Approach to Numerical Computing" (generic numeric code across
  many scalar types, with performance from specialization): https://arxiv.org/abs/1411.1607
- Spitters and van der Weegen, "Type classes for mathematics in type theory" (typeclass-based
  algebraic interfaces for reusable formalization and instances):
  https://doi.org/10.1017/S0960129511000119
- Elliott, "The Simple Essence of Automatic Differentiation" (one abstract formulation specialized
  to multiple concrete semantics/representations): https://arxiv.org/abs/1804.00746

Our `Context α` is the same engineering pattern in a Lean setting: one model/layer definition, many
scalar interpretations, and stated tradeoffs about semantics.

To make this practical, we collect the numeric operations required by neural networks into the
single typeclass `Context α`. It is broader than a standard algebraic structure: it bundles
arithmetic, ordering, and the common transcendental functions (exp/tanh/log/sqrt) of activations
and losses.

## Notes

- Many spec definitions assume `[Context α]` so they can be reused at multiple dtypes.
- The theorems about exact semantics use the scalar `ℝ`.
- `TorchLean.Context.decidableGT` is included so executable code can decide comparisons (e.g.
  ReLU / argmax). The derived global `DecidableRel` instance has low priority so native decision
  procedures win.
- `LawfulContext α` (below) records when a `Context` agrees with a Mathlib ordered field on `α`.
- The `ℝ` context dictionary lives in `TorchLean.Tensor.Context.Real` and the opt-in
  rational one in `TorchLean.Tensor.Context.Rational`. FloatLib's shared
  elementary-function class already imports real analysis. Selecting a context adds no accuracy
  theorem for finite-precision arithmetic.
- For executable examples, `TorchLean.Context.gtBool` converts `x > y` into a printable `Bool`.
- `TorchLean.Context.isZero` includes differentiation metadata, unlike the primal-only comparisons
  of dual numbers. Composite carriers lift this test through every component.
- `Context` requires neither a linear order nor agreement between its comparisons and those of the
  reals. Binary formats leave a NaN unordered, and `TorchLean.Complex α` compares real parts.
- The elementary functions PyTorch adds to `MathFunctions` (`copysign`, `isNaN`, `isInf`, `tan`,
  `exp2`, `log2`, `log10`, `asinh`, `acosh`, `atanh`, `hypot`) are fields whose defaults are
  formulas in the other fields. `Float` and `Float32` replace every one, the transcendental ones
  with the C library's functions. Configured binary formats replace every one, `tan` to `atanh`
  with the real functions rounded once. `ℝ` replaces the transcendental ones with Mathlib's, and
  dual numbers replace every one by the chain rule over their scalar's functions. FloatLib's rounded
  reals `NF` and the rational fragment keep the defaults, and `Complex α` replaces only `tan`,
  `asinh` and `acosh`. Each step of a default formula rounds in a rounded type: `log2` and `log10`
  can miss the exact result at a power of their base and `exp2` at an integer, `x * x` in `asinh`,
  `acosh` and `hypot` can overflow where the result is representable, and `asinh` loses a small `x`
  in `|x| + √(x² + 1)`. Over `ℚ` the defaults inherit the rational dictionary's placeholder
  transcendentals.
- `quietNaN` makes a signaling NaN quiet and returns every other value, as IEEE 754-2019 requires
  of an operation that returns a NaN operand, such as `Context.maximum`. Its default, the
  identity, is right for a scalar type without signaling NaNs, Lean's `Float` and `Float32`
  included: their model has one NaN, which is quiet. FloatLib binary formats set the quiet bit.
- `extremaTie x y r` is the result of `Context.maximum` or `Context.minimum` of `x` and `y` when
  neither operand wins strictly (equal operands or a NaN) and `r` is the value chosen. Its default
  is `r`. Dual numbers keep `r`'s primal and average the two tangents, the tie rule of their `max`
  and `min`.
- `Rounding` supplies `floor`, `ceil`, `trunc`, `round` with ties to even and C's `fmod`. It is a
  separate class, so a scalar type without these operations, such as `Complex α`, has none of the
  operators that need them. `Float` and `Float32` share `TorchLean.Numeric.HostFloat.rounding`,
  built on C's `floor`, `ceil` and `round` and FloatLib's exact IEEE remainder. We wrote it by
  hand from the C99 committee draft WG14 N1256: C's `round` is its 7.12.9.6, and `fmodSchedule`
  follows the `fmod` that F.9.7.1 builds from `remainder`. `Erf`, the error function, is optional
  in the same way.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Numerics (MathFunctions)

/-- The full scalar interface required by spec-level tensors and models. -/
class Context (α : Type) extends
  Inhabited α, One α, Zero α,
  Add α, Sub α, Mul α, Div α, Neg α, Pow α α, Max α, Min α,
  BEq α, LT α, LE α, -- For ordering
  MathFunctions α,
  NatCast α, RatCast α, Numeric.QuotientArithmetic α where
  /--
  Convert a rational constant using the backend's arithmetic.

  The default casts its numerator and denominator separately. Native `Float`, `Float32`, and
  configured binary backends override this to round the exact fraction, avoiding overflow in
  either intermediate integer cast.
  -/
  ratCast value :=
    let numerator : α := match value.num with
      | .ofNat n => (n : α)
      | .negSucc n => -((n + 1 : Nat) : α)
    numerator / (value.den : α)
  /-- Construct a square-root constant from a natural count.

  The default takes the square root after casting. Binary backends retain that path when the cast
  does not overflow, but use FloatLib's exact rational-root algorithm otherwise. A dimension can
  exceed the scalar range even when its square root is representable. -/
  sqrtNat (value : Nat) : α := MathFunctions.sqrt (value : α)
  /-- `x` divided by the count `n`. The default divides by the cast count, which in a lawful
  context is field division by the cast count (`LawfulContext.divNat_eq`). Binary backends round
  the exact quotient once, so no count is rounded or overflows first, and divide by zero at the
  count zero. The result is a value, without exception status. -/
  divNat (x : α) (n : Nat) : α := x / (n : α)
  /-- Test for zero including differentiation metadata.

  Ordinary scalars use Boolean equality. Dual and compound scalars must inspect every component:
  a zero primal with a nonzero tangent is not a constant zero in a differentiated expression. -/
  isZero (value : α) : Bool := value == 0
  /-- The safeguard constant that guarded formulas use by default, chosen by the backend. It is not
  machine epsilon. -/
  defaultEpsilon : α
  /-- Decision procedure for the scalar type's strict order. -/
  decidableGT : DecidableRel (· > · : α → α → Prop)
  /-- Branch-stable sigmoid. Differentiation carriers apply its complete weighted chain rule. -/
  sigmoid : α → α :=
    letI : DecidableRel ((· > ·) : α → α → Prop) := decidableGT
    Numeric.sigmoid
  /-- `log(1 + x)`, rounded once in configured IEEE formats and by default compensated for
  rounding in the addition. Differentiation carriers supply its analytic chain rule. -/
  log1p : α → α := Numeric.log1p
  /-- `exp x - 1`, rounded once in configured IEEE formats. The default compensates for rounding in
  the exponential when `|x| < 1` and otherwise subtracts `1` from the rounded `exp x`, rounding
  twice (`Float`'s `expm1 1` is one ulp below the correctly rounded value). Differentiation carriers
  supply its analytic chain rule. -/
  expm1 : α → α :=
    letI : DecidableRel ((· > ·) : α → α → Prop) := decidableGT
    Numeric.expm1
  /-- `exp x * w`. Differentiation carriers use it for the tangents of `exp` and `expm1`, and
  `Op.exp` and `Op.expm1` for their pullbacks. The default multiplies the rounded exponential. A
  backend that rounds the product once keeps it finite when `exp x` alone overflows, and zero when
  `w` is. It can also overflow where the default does not: in binary32 with `x = 3 * 2^-26` and the
  largest finite `w`, the exponential rounds to one and the default returns `w`, while the exact
  product is above the overflow threshold, so its correct rounding is infinity. -/
  expMul (x w : α) : α := MathFunctions.exp x * w
  /--
  Remove differentiation metadata from a scalar at a `detach` boundary.

  Ordinary numeric carriers leave this as `none`, so detaching a tensor can reuse its storage.
  A carrier such as `Dual α` supplies a map that keeps the primal value and clears its tangents.
  A reverse pass over dual numbers needs it, since cutting the tape edge alone would leave the
  detached value's forward tangent available to later operations.
  -/
  stopGradient? : Option (α → α) := none
  /-- The magnitude of `x` with the sign of `y`. The default tests `0 > y`, so it gives `|x|` when
  `y` is a zero or a NaN of either sign. IEEE formats copy the sign bit. -/
  copysign (x y : α) : α :=
    letI : DecidableRel ((· > ·) : α → α → Prop) := decidableGT
    if 0 > y then -MathFunctions.abs x else MathFunctions.abs x
  /-- Whether `x` is a NaN, by default whether `x == x` fails. -/
  isNaN (x : α) : Bool := !(x == x)
  /-- Whether `x` is an infinity, by default whether `x == x` holds and `x - x == 0` fails. -/
  isInf (x : α) : Bool := x == x && !(x - x == 0)
  /-- `x` with a signaling NaN made quiet and every other value unchanged. The default is the
  identity, which suits a scalar type without signaling NaNs. -/
  quietNaN (x : α) : α := x
  /-- The tie rule of an extremum: `extremaTie x y r` is the result when neither `x` nor `y` wins
  strictly (equal operands or a NaN) and `r` is the value chosen. The default returns `r`. -/
  extremaTie : α → α → α → α := fun _ _ r => r
  /-- Tangent, by default `sin x / cos x`, three rounded operations in a rounded type. -/
  tan (x : α) : α := MathFunctions.sin x / MathFunctions.cos x
  /-- `2 ^ x`, by default `exp (x * log 2)`. In a rounded type the product rounds first and the
  exponential turns that rounding into a relative error growing with `|x|`, so an exact power of
  two can be missed. -/
  exp2 (x : α) : α := MathFunctions.exp (x * MathFunctions.log (1 + 1))
  /-- Base-two logarithm, by default `log x / log 2`, which in a rounded type can miss the exact
  result at a power of two. -/
  log2 (x : α) : α := MathFunctions.log x / MathFunctions.log (1 + 1)
  /-- Base-ten logarithm, by default `log x / log 10`, which in a rounded type can miss the exact
  result at a power of ten. -/
  log10 (x : α) : α := MathFunctions.log x / MathFunctions.log ((10 : Nat) : α)
  /-- Inverse hyperbolic sine, by default `log (|x| + √(x² + 1))` with the sign of `x` given by
  `copysign`, so a zero keeps its sign where `copysign` reads it. In a rounded type `x²` can
  overflow where the result is representable, and the sum loses a small `x`. -/
  asinh (x : α) : α :=
    copysign (MathFunctions.log (MathFunctions.abs x + MathFunctions.sqrt (x * x + 1))) x
  /-- Inverse hyperbolic cosine, by default `log (x + √(x² - 1))`. In a rounded type `x²` can
  overflow where the result is representable, and `x² - 1` cancels near one. -/
  acosh (x : α) : α := MathFunctions.log (x + MathFunctions.sqrt (x * x - 1))
  /-- Inverse hyperbolic tangent, by default `(log1p x - log1p (-x)) / 2`, three rounded
  operations after the two logarithms. -/
  atanh (x : α) : α := (log1p x - log1p (-x)) / (1 + 1)
  /-- `√(x² + y²)`, by default computed as written, which can overflow or underflow in a binary
  format where the result is representable. The binary formats round the exact value once. -/
  hypot (x y : α) : α := MathFunctions.sqrt (x * x + y * y)

/--
Rounding to an integral value and the truncating remainder, for scalar types that have them.

In IEEE formats the four roundings return an infinity or a zero unchanged and a NaN at a NaN, and
every zero result of the five operations has the sign of `x`. A scalar type without these
operations has no instance, so the operators that need them are unavailable there.

`floor`, `ceil`, `trunc` and `round` model IEEE 754-2019's roundToIntegralTowardNegative,
roundToIntegralTowardPositive, roundToIntegralTowardZero and roundToIntegralTiesToEven, which the
C23 draft WG14 N3096 binds to C's `floor`, `ceil`, `trunc` and `roundeven` (F.3).
-/
class Rounding (α : Type) where
  /-- The largest integer not above `x`. -/
  floor : α → α
  /-- The smallest integer not below `x`. -/
  ceil : α → α
  /-- `x` rounded toward zero. -/
  trunc : α → α
  /-- `x` rounded to the nearest integer, a tie to the even one, as `torch.round` (PyTorch 2.8
  docs) does. `TorchLean.Test.Elementwise.run` tests the halves `-2.5` to `2.5` on `Float`, where
  `-0.5` goes to `-0`. -/
  round : α → α
  /-- C's `fmod`: `x - y * trunc (x / y)` computed exactly, with the sign of `x`. It's C99
  7.12.10.1 (WG14 N1256), the `std::fmod` that `torch.fmod` (PyTorch 2.8 docs) applies. -/
  fmod : α → α → α

/-- The Gauss error function `erf x = 2 / √π ∫₀ˣ exp (-t²) dt`, for scalar types that have it. -/
class Erf (α : Type) where
  /-- The error function. -/
  erf : α → α

/-- The real cube root, IEEE 754-2019 `rootn(x, 3)`, for scalar types that have it: `-0` at `-0`,
`±∞` at `±∞` and a quiet NaN at a NaN in a binary format. -/
class Cbrt (α : Type) where
  /-- The cube root. -/
  cbrt : α → α

/-- C's `nextafter`, for scalar types with neighbouring values: a NaN where either operand is one,
the second operand where they are equal, and otherwise the neighbour of the first toward the
second, IEEE 754-2019 `nextUp` or `nextDown`. -/
class NextAfter (α : Type) where
  /-- The neighbour of the first operand toward the second. -/
  nextafter : α → α → α

/-- The interchange word of a binary floating-point scalar (IEEE 754-2019 3.4) where every word of
`TorchLean.Numeric.Word.width α` bits is the word of one value, NaN payloads and signaling NaNs
included. -/
class Bits (α : Type) extends Numeric.Word α where
  /-- A word is the word of its value. -/
  toBits_ofBits : ∀ b, toBits (ofBits b) = b

namespace Context

/-- `1 / √x`, two roundings in a binary format, so not always the correctly rounded `rSqrt x`: in
binary64, `rsqrt 3` is one ulp above it. -/
def rsqrt {α : Type} [One α] [Div α] [MathFunctions α] (x : α) : α :=
  1 / MathFunctions.sqrt x

/-- The fractional part `x - trunc x`: it has the sign of `x` at a finite non-integer, is zero at
an integer, `-0` included (`+0` when a binary format rounds to nearest, like `1 - 1` in C99 F.8.2,
WG14 N1256), and is a NaN at an infinity or a NaN. -/
def frac {α : Type} [Sub α] [Rounding α] (x : α) : α := x - Rounding.trunc x

/-- Python's `%`, which `torch.remainder` (PyTorch 2.8 docs) documents: `fmod x y`, moved by `y`
when it is nonzero and its sign differs from `y`'s, and a zero result with the sign of `y`. The
addition rounds, so a nonzero result need not be the exact remainder. We keep the documented sign
at a zero, so `-0.0 % 3.0` is `0.0` as in Python, where PyTorch 2.8.0's CPU kernel returns
`fmod`'s `-0.0`. `TorchLean.Test.Elementwise.run` checks it on `pythonRemainders`, 20 values from
CPython 3.12 and a NaN at the zero divisor. -/
def remainder {α : Type} [Zero α] [Neg α] [Add α] [Sub α] [LT α] [DecidableRel (α := α) (· < ·)]
    [BEq α] [Rounding α] (x y : α) : α :=
  remainderSchedule Rounding.fmod x y

/-- Decide `x > y` as a `Bool` using the `Context`'s `decidableGT`. -/
@[inline] def gtBool {α : Type} [Context α] (x y : α) : Bool :=
  let _ : Decidable (x > y) := (Context.decidableGT) x y
  decide (x > y)

/-- Clear a scalar's differentiation metadata, if its carrier has any. -/
def stopGradient {α : Type} [Context α] (x : α) : α :=
  match Context.stopGradient? (α := α) with
  | none => x
  | some clear => clear x

end Context

/--
The decidable `>` of a `Context`, as a `DecidableRel` instance.

The instance has low priority so that a type's own decision procedure (`Real.decidableLT`,
`Float.decLt`, ...) wins whenever one exists.
-/
@[inline] instance (priority := low) Context.decidableRelGt {α : Type} [Context α] :
    DecidableRel ((· > ·) : α → α → Prop) :=
  Context.decidableGT

/-!
## Lawful contexts

`Context α` carries no laws: it only bundles operations. `LawfulContext α` records, for a scalar
type that also carries a Mathlib ordered field structure, that the `Context` dictionary computes
the same operations. Generic proofs about scalar-polymorphic specifications can then
rewrite the dictionary operations into the ordinary ring operations and finish with Mathlib.

Only exact facts are recorded: the field operations, the order, `max`, `min`, `==`, the natural and
rational casts, `divNat`, `abs` and a positive `defaultEpsilon`. The other `MathFunctions` members
(`sqrt`, `exp`, `pi`, ...), the total power operation and the remaining fields of `Context` are
backend specific and stay unconstrained.
-/

/--
A `Context` dictionary that agrees with a linearly ordered field structure on the same type.

Each field equates a `Context` projection, written through its instance path `Context.to*`, with
the corresponding Mathlib operation.
-/
class LawfulContext (α : Type) [Context α] [Field α] [LinearOrder α]
    [IsStrictOrderedRing α] : Prop where
  /-- Dictionary addition is ring addition. -/
  add_eq (x y : α) :
    @HAdd.hAdd α α α (@instHAdd α Context.toAdd) x y =
      @HAdd.hAdd α α α (@instHAdd α Distrib.toAdd) x y
  /-- Dictionary multiplication is ring multiplication. -/
  mul_eq (x y : α) :
    @HMul.hMul α α α (@instHMul α Context.toMul) x y =
      @HMul.hMul α α α (@instHMul α Distrib.toMul) x y
  /-- Dictionary subtraction is ring subtraction. -/
  sub_eq (x y : α) :
    @HSub.hSub α α α (@instHSub α Context.toSub) x y =
      @HSub.hSub α α α (@instHSub α SubNegMonoid.toSub) x y
  /-- Dictionary division is field division. -/
  div_eq (x y : α) :
    @HDiv.hDiv α α α (@instHDiv α Context.toDiv) x y =
      @HDiv.hDiv α α α (@instHDiv α DivisionRing.toDiv) x y
  /-- Dictionary negation is ring negation. -/
  neg_eq (x : α) :
    @Neg.neg α Context.toNeg x = @Neg.neg α NegZeroClass.toNeg x
  /-- The dictionary zero is the ring zero. -/
  zero_eq : @Zero.zero α Context.toZero = @Zero.zero α MulZeroClass.toZero
  /-- The dictionary one is the ring one. -/
  one_eq : @One.one α Context.toOne = @One.one α Monoid.toOne
  /-- The dictionary strict order is the field order. -/
  lt_iff (x y : α) : @LT.lt α Context.toLT x y ↔ @LT.lt α Preorder.toLT x y
  /-- The dictionary order is the field order. -/
  le_iff (x y : α) : @LE.le α Context.toLE x y ↔ @LE.le α Preorder.toLE x y
  /-- Dictionary `max` is the lattice maximum. -/
  max_eq (x y : α) : @Max.max α Context.toMax x y = @Max.max α LinearOrder.toMax x y
  /-- Dictionary `min` is the lattice minimum. -/
  min_eq (x y : α) : @Min.min α Context.toMin x y = @Min.min α LinearOrder.toMin x y
  /-- Boolean equality decides propositional equality. -/
  beq_iff (x y : α) : (x == y) = true ↔ x = y
  /-- The natural-number cast is the semiring cast. -/
  natCast_eq (n : Nat) :
    @Nat.cast α Context.toNatCast n = @Nat.cast α Semiring.toNatCast n
  /-- Dictionary division by a count is field division by the cast count. -/
  divNat_eq (x : α) (n : Nat) :
    Context.divNat x n =
      @HDiv.hDiv α α α (@instHDiv α DivisionRing.toDiv) x (@Nat.cast α Semiring.toNatCast n)
  /-- The rational-number cast is the field cast. -/
  ratCast_eq (value : Rat) :
    @Rat.cast α Context.toRatCast value = @Rat.cast α DivisionRing.toRatCast value
  /-- The dictionary absolute value is the lattice absolute value. -/
  abs_eq (x : α) : MathFunctions.abs x = |x|
  /-- The backend tolerance is strictly positive. -/
  defaultEpsilon_pos : (0 : α) < Context.defaultEpsilon

attribute [simp] LawfulContext.add_eq LawfulContext.mul_eq LawfulContext.sub_eq
  LawfulContext.div_eq LawfulContext.neg_eq LawfulContext.zero_eq LawfulContext.one_eq
  LawfulContext.lt_iff LawfulContext.le_iff LawfulContext.max_eq LawfulContext.min_eq
  LawfulContext.beq_iff LawfulContext.natCast_eq LawfulContext.divNat_eq LawfulContext.abs_eq
  LawfulContext.ratCast_eq

/-- C's `fmod` from the exact IEEE remainder `remainder a b`: that remainder where it is zero or
has the sign bit of `a`, and otherwise the remainder plus the magnitude of `b` with the sign of `a`.
The sum is the exact truncating remainder, so a binary format's rounded addition returns it
unchanged. C99's reference `fmod` (F.9.7.1 in WG14 N1256) takes `remainder (fabs x) (fabs y)` and
adds `fabs y` to a negative one before copying the sign of `x`, which gives our values, NaN signs
and payloads aside. `TorchLean.Roundoff.toReal_fmod` proves the configured result exact at finite
operands and a nonzero divisor. -/
@[inline] def fmodSchedule {α : Type} [Add α] (remainder : α → α → α) (isZero signBit : α → Bool)
    (copysign : α → α → α) (a b : α) : α :=
  let r := remainder a b
  if !isZero r && signBit r != signBit a then r + copysign b a else r

namespace Numeric.HostFloat

open FloatLib.Floats.Formats.BinaryInterchange

variable {α : Type} [HostFloat α]

/-- `x` rounded to the nearest integer with ties to even, from C's `round`, which sends ties away
from zero. `roundAway x - x` is exact, so a tie is detected exactly, and then
`2 * roundAway (x / 2)` is the even neighbour, `x / 2` being exact because a tie is at least one
half in magnitude. -/
@[inline] def roundEven [Context α] (x : α) : α :=
  let two : α := 1 + 1
  let away := roundAway x
  if MathFunctions.abs (away - x) == 1 / two then two * roundAway (x / two) else away

/-- C's `hypot`, FloatLib's `Model.hypot`: the exact norm `√(x² + y²)` rounded once. -/
@[inline] def hypot (x y : α) : α := ofModel (Model.hypot (toModel x) (toModel y))

/-- The IEEE 754-2019 `remainder` of `x` and `y`: `x - n * y` for the integer `n` nearest to
`x / y`, a tie to the even one, computed exactly. -/
@[inline] def remainder (x y : α) : α := ofModel (Model.remainder (toModel x) (toModel y))

/-- The square root of the count `n`: the root of its cast where the cast is finite, and otherwise
the exact root rounded once. -/
@[inline] def sqrtNat [NatCast α] [MathFunctions α] (n : Nat) : α :=
  let input : α := n
  if (toModel input).isFinite then MathFunctions.sqrt input else
    ofModel (Model.AlgebraicRounding.sqrtRat _ false (n : Rat))

/-- `x` divided by the count `n`: one host division where `n ≤ 2 ^ p` for the precision `p`,
which makes the cast of `n` exact. For larger `n` it is the exact quotient rounded once, `x`
itself when `x` is a zero and `x / 1` when it is an infinity or a NaN. -/
@[inline] def divNat [NatCast α] [Div α] [One α] (x : α) (n : Nat) : α :=
  if n ≤ 2 ^ ((HostFloat.format α).fracWidth + 1) then x / (n : α) else
    match Model.toRat? (toModel x) with
    | some q => if q = 0 then x else ofModel (Model.ofRat _ (q / n))
    | none => x / 1

/-- `Rounding` for a host binary type: C's `floor` and `ceil`, `trunc` from them by sign,
`roundEven`, and C's `fmod` from the exact IEEE `remainder`. -/
abbrev rounding [Context α] : Rounding α where
  floor x := HostFloat.floor x
  ceil x := HostFloat.ceil x
  trunc x := if Context.gtBool 0 x then HostFloat.ceil x else HostFloat.floor x
  round x := roundEven x
  fmod x y := fmodSchedule remainder (· == 0) HostFloat.signBit HostFloat.copysign x y

end Numeric.HostFloat

/-- Full `Context` instance for `Float` (runtime backend). -/
@[inline] instance : Context Float where
  toRatCast := inferInstance
  sqrtNat n := Numeric.HostFloat.sqrtNat n
  divNat x n := Numeric.HostFloat.divNat x n
  defaultEpsilon := 1e-6
  decidableGT := inferInstance
  copysign x y := Numeric.HostFloat.copysign x y
  hypot x y := Numeric.HostFloat.hypot x y
  -- The native product is kept where it is finite and either `w` is zero or both it and `exp x`
  -- are normal. Elsewhere `exp x` may overflow, or be subnormal with bits lost before the
  -- product, and FloatLib's `expMul` rounds once.
  expMul x w :=
    let e := MathFunctions.exp x
    let product := e * w
    let minNormal := Numeric.NativeFormat.minNormal (α := Float)
    if product - product == 0 && (w == 0 || minNormal ≤ product.abs && minNormal ≤ e) then product
    else FloatLib.Floats.ExecFloat.Binary.toFloat
      (FloatLib.Floats.ExecFloat.Binary.CorrectlyRounded.expMul
        (FloatLib.Floats.ExecFloat.Binary.ofFloat x : FloatLib.Floats.Binary64)
        (FloatLib.Floats.ExecFloat.Binary.ofFloat w))
  isNaN := Float.isNaN
  isInf := Float.isInf
  tan := Float.tan
  exp2 := Float.exp2
  log2 := Float.log2
  log10 := Float.log10
  asinh := Float.asinh
  acosh := Float.acosh
  atanh := Float.atanh

/-- Full `Context` instance for native binary32 execution. -/
@[inline] instance : Context Float32 where
  toRatCast := inferInstance
  sqrtNat n := Numeric.HostFloat.sqrtNat n
  divNat x n := Numeric.HostFloat.divNat x n
  defaultEpsilon := (1e-6 : Float).toFloat32
  decidableGT := inferInstance
  copysign x y := Numeric.HostFloat.copysign x y
  hypot x y := Numeric.HostFloat.hypot x y
  expMul x w :=
    let e := MathFunctions.exp x
    let product := e * w
    let minNormal := Numeric.NativeFormat.minNormal (α := Float32)
    if product - product == 0 && (w == 0 || minNormal ≤ product.abs && minNormal ≤ e) then product
    else FloatLib.Floats.ExecFloat.Binary.toFloat32
      (FloatLib.Floats.ExecFloat.Binary.CorrectlyRounded.expMul
        (FloatLib.Floats.ExecFloat.Binary.ofFloat32 x : FloatLib.Floats.Binary32)
        (FloatLib.Floats.ExecFloat.Binary.ofFloat32 w))
  isNaN := Float32.isNaN
  isInf := Float32.isInf
  tan := Float32.tan
  exp2 := Float32.exp2
  log2 := Float32.log2
  log10 := Float32.log10
  asinh := Float32.asinh
  acosh := Float32.acosh
  atanh := Float32.atanh

/-- `Rounding` for `Float`: `TorchLean.Numeric.HostFloat.rounding`. -/
instance : Rounding Float := Numeric.HostFloat.rounding

/-- `Rounding` for `Float32`: `TorchLean.Numeric.HostFloat.rounding`. -/
instance : Rounding Float32 := Numeric.HostFloat.rounding

end TorchLean


