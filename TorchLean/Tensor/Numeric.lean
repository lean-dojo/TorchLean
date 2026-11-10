/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Numerics.Capabilities.Elementary
public import TorchLean.Tensor.Numeric.Enclosure

/-!
# Scalar numeric interface and stable scalar formulas

FloatLib owns the `MathFunctions` class of transcendental operations and its `Float` and `ℝ`
instances, and its elementary capability module imports real analysis. Files that name the class
open it with `open FloatLib.Numerics (MathFunctions)`. This module supplies TorchLean's native
binary32 instance of that class, compensated `log1p`/`expm1` formulas, and input-based sigmoid/tanh
slopes and weighted pullbacks. It knows nothing about tensors, model composition or runtimes.

Most formulas run their ordinary arithmetic through the checked operations of
`QuotientArithmetic`. When an intermediate loses range, a rational expression is replayed over
finite coefficients and only its result is rounded. The power rules, `log1p`, and a sigmoid or
tanh slope whose half exponential itself loses range use a bounded enclosure search instead,
which evaluates the complete expression before rounding. Nested floating-point differentiation
uses reciprocal cosh identities to keep small derivative coefficients near zero.

The main definitions are `sigmoid`, the input-based slopes `sigmoidDeriv` and `tanhDeriv`, the
weighted pullbacks `sigmoidBackward` and `tanhBackward`, `log1p`, `log1pBackward`, `expm1` and
`normalizationEpsilon`. The shared chain rules include `Numeric.sqrtBackward`,
`Numeric.hypotBackward`, `Numeric.angleTangent`, `Numeric.powerBackward` and
`Numeric.powerTangent`. Integer and rational constants use the standard numerical interfaces,
and the neural-model interface `Context` lives in `TorchLean.Tensor.Context`.
-/

public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

namespace Numeric

/-- Logistic sigmoid, `1 / (1 + exp (-x))` for `x > 0` and `exp x / (1 + exp x)` otherwise, so a
real input never meets a positive exponential argument.

If the checked sum in the denominator loses range, numerator and denominator are replayed
together over finite rational coefficients and only the quotient is rounded. A valid configured
format need not represent `2`, although it represents the value `1 / 2` at zero, and the replay
still returns that value. Unsupported carriers use the ordinary branch formula.
-/
@[expose] def sigmoid {α : Type} [Zero α] [One α] [Add α] [Div α] [Neg α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [MathFunctions α]
    [ops : QuotientArithmetic α] (x : α) : α :=
  let positive := decide (x > 0)
  let z := MathFunctions.exp (if positive then -x else x)
  let numerator := if positive then 1 else z
  if !ops.supported then numerator / (1 + z)
  else
    let (denominator, flags) := ops.addWithFlags 1 z
    let ordinary := numerator / denominator
    if flags == 0 then ordinary
    else
      match ops.encode (1 : α), ops.encode z with
      | some one, some tail =>
          let denominator := QuotientCoefficients.add one tail
          if denominator.primal == 0 then ordinary
          else
            let coefficients := QuotientCoefficients.div (if positive then one else tail)
              denominator
            (ops.decode coefficients).getD ordinary
      | _, _ => ordinary

/-- Average of two derivative coefficients that keeps a representable result through the halving.

The ordinary path computes `half * x + half * y`, with `half` cast from the rational `1 / 2` so
the scalar type need not represent `2`. If a checked product or the sum loses range, the whole
average is replayed over finite rational coefficients and only its result is rounded. Nonfinite
inputs keep the ordinary result. -/
@[expose] def average {α : Type} [RatCast α] [Add α] [Mul α]
    [ops : QuotientArithmetic α] (x y : α) : α :=
  let half : α := Rat.cast (1 / 2 : Rat)
  if !ops.supported then half * x + half * y
  else
    let (left, leftFlags) := ops.mulWithFlags half x
    let (right, rightFlags) := ops.mulWithFlags half y
    let (ordinary, sumFlags) := ops.addWithFlags left right
    if leftFlags ||| rightFlags ||| sumFlags == 0 then ordinary
    else
      match ops.encode x, ops.encode y with
      | some x, some y =>
          let exact := QuotientCoefficients.mul
            (QuotientCoefficients.ofScalar ops.depth (1 / 2 : Rat))
            (QuotientCoefficients.add x y)
          (ops.decode exact).getD ordinary
      | _, _ => ordinary

/-- Chain rule of the square root, `seed / (2 * root)`, from the root the forward pass computed.

The seed is divided by twice the root in one rounding. If checked arithmetic loses range, the
weighted quotient is replayed over finite rational coefficients, where `2` is exact even when its
scalar cast overflows. The replay starts from the supplied root, so the rounding of the root itself
stays in the result. Nonfinite inputs keep the ordinary result. -/
@[expose] def sqrtBackward {α : Type} [NatCast α] [One α] [Mul α] [Div α]
    [ops : QuotientArithmetic α] (root seed : α) : α :=
  let two : α := (2 : Nat)
  if !ops.supported then
    seed / (two * root)
  else
    let (denominator, denominatorFlags) := ops.mulWithFlags two root
    let (ordinary, flags) := ops.divWithFlags seed denominator
    if denominatorFlags ||| flags == 0 then ordinary
    else
      match ops.encode root, ops.encode seed with
      | some rootCoefficients, some seedCoefficients =>
          let denominator := QuotientCoefficients.mul
            (QuotientCoefficients.ofScalar ops.depth (2 : Rat)) rootCoefficients
          if denominator.primal == 0 then ordinary
          else
            match ops.decode (QuotientCoefficients.div seedCoefficients denominator) with
            | none => ordinary
            | some value =>
                -- Rational zero has no sign, so the native quotient `seed / root` supplies it.
                if seedCoefficients.primal == 0 then ops.copyPrimal (seed / root) value
                else value
      | _, _ => ordinary

/-- The chain rule of `asinh`, `seed / √(x² + 1)`, from the radius `√(x² + 1)` that the caller
computes as `hypot x 1`, which does not square `x`. -/
@[expose] def asinhBackward {α : Type} [Div α] (radius seed : α) : α := seed / radius

/-- Chain rule of `acosh` as `seed / (√(x - 1) * √(x + 1))`. Above one this is
`seed / √(x² - 1)` without squaring `x`, and at a complex `x` it is the derivative of the
principal branch: `x + 1` is written `x - (0 - 1)` so that the sign of a zero imaginary part
selects the side of the cut. -/
@[expose] def acoshBackward {α : Type} [Zero α] [One α] [Sub α] [Mul α] [Div α]
    [MathFunctions α] (x seed : α) : α :=
  seed / (MathFunctions.sqrt (x - 1) * MathFunctions.sqrt (x - (0 - 1)))

/-- The chain rule of `atanh`, `seed / ((1 - x) * (1 + x))`, without the cancellation of
`1 - x * x` near `|x| = 1`. -/
@[expose] def atanhBackward {α : Type} [One α] [Add α] [Sub α] [Mul α] [Div α]
    (x seed : α) : α :=
  seed / ((1 - x) * (1 + x))

/-- The chain rule of `atan`, `seed / (x² + 1)`. Where `x * x + 1` is finite it computes
`seed / (x * x + 1)`, and otherwise `seed / x / x`, which does not square `x` and gives zero at an
infinite `x`. -/
@[expose] def atanBackward {α : Type} [Zero α] [One α] [Add α] [Sub α] [Mul α] [Div α] [BEq α]
    (x seed : α) : α :=
  let d := x * x + 1
  if d - d == 0 then seed / d else seed / x / x

/-- One coordinate's term `seed * coordinate / radius` of the chain rule of `hypot`, from the
radius the forward pass computed. At the origin, where the radius is zero, the term is `0 * seed`.

If checked arithmetic loses range, the quotient of the supplied values is replayed over finite
coefficients and rounded once, so an overflowing product keeps a representable result where the
radius is finite. -/
@[expose] def hypotBackward {α : Type} [Zero α] [Mul α] [Div α] [BEq α]
    [ops : QuotientArithmetic α] (coordinate radius seed : α) : α :=
  if radius == 0 then 0 * seed
  else if !ops.supported then seed * coordinate / radius
  else
    let (product, productFlags) := ops.mulWithFlags seed coordinate
    let (ordinary, quotientFlags) := ops.divWithFlags product radius
    if productFlags ||| quotientFlags == 0 then ordinary
    else
      match ops.encode coordinate, ops.encode radius, ops.encode seed with
      | some x, some h, some g =>
          match ops.decode (QuotientCoefficients.div (QuotientCoefficients.mul g x) h) with
          | none => ordinary
          | some recovered =>
              if ordinary == 0 && recovered == 0 then ops.copyPrimal ordinary recovered
              else recovered
      | _, _, _ => ordinary

/-- Directional derivative `(x * dy - y * dx) / (x² + y²)` of the angle of `(x, y)` in the
direction `(dx, dy)`.

All four inputs are first divided by `max |x| |y|`, which avoids squaring large coordinates. If a
checked intermediate loses range, or the scaled products cancel, the complete quotient is
evaluated over finite rational coefficients from the original inputs and rounded once: a scaled
direction can overflow even when the derivative is representable. At the origin, and at an
infinite coordinate where the derivative tends to zero, the result is the zero slope
`0 * dx + 0 * dy`. A NaN coordinate keeps the ordinary formula. Across the branch cut, where the
principal angle jumps, the formula is the derivative of the continuous local branch. -/
@[expose] def angleTangent {α : Type} [Zero α] [Add α] [Sub α] [Mul α] [Div α]
    [BEq α] [Max α] [MathFunctions α] [ops : QuotientArithmetic α] (x y dx dy : α) : α :=
  let scale := max (MathFunctions.abs x) (MathFunctions.abs y)
  if x == 0 && y == 0 || x == x && y == y && !(scale - scale == 0) then 0 * dx + 0 * dy else
    if !ops.supported then
      let a := x / scale
      let b := y / scale
      (a * (dy / scale) - b * (dx / scale)) / (a * a + b * b)
    else
      let (a, xFlags) := ops.divWithFlags x scale
      let (b, yFlags) := ops.divWithFlags y scale
      let (dxScaled, dxFlags) := ops.divWithFlags dx scale
      let (dyScaled, dyFlags) := ops.divWithFlags dy scale
      let (left, leftFlags) := ops.mulWithFlags a dyScaled
      let (right, rightFlags) := ops.mulWithFlags b dxScaled
      let (numerator, numeratorFlags) := ops.subWithFlags left right
      let (aa, xxFlags) := ops.mulWithFlags a a
      let (bb, yyFlags) := ops.mulWithFlags b b
      let (denominator, denominatorFlags) := ops.addWithFlags aa bb
      let (ordinary, quotientFlags) := ops.divWithFlags numerator denominator
      let flags := xFlags ||| yFlags ||| dxFlags ||| dyFlags ||| leftFlags ||| rightFlags |||
        numeratorFlags ||| xxFlags ||| yyFlags ||| denominatorFlags ||| quotientFlags
      let cancellation := numerator == 0 && (left != 0 || right != 0)
      if flags == 0 && !cancellation then ordinary
      else
        match ops.encode x, ops.encode y, ops.encode dx, ops.encode dy with
        | some x, some y, some dx, some dy =>
            let numerator := QuotientCoefficients.sub
              (QuotientCoefficients.mul x dy) (QuotientCoefficients.mul y dx)
            let denominator := QuotientCoefficients.add
              (QuotientCoefficients.mul x x) (QuotientCoefficients.mul y y)
            if denominator.primal == 0 then ordinary else
              match ops.decode (QuotientCoefficients.div numerator denominator) with
              | none => ordinary
              | some recovered =>
                  if numerator.primal == 0 && ordinary == 0 then
                    ops.copyPrimal ordinary recovered
                  else recovered
        | _, _, _, _ => ordinary

/-- Power rule `exponent * base ^ (exponent - 1) * seed` for a constant exponent, without rounding
an out-of-range local factor first.

The ordinary arithmetic order stays unless a checked intermediate loses range or the computed
`base ^ (exponent - 1)` is zero at a nonzero base. Recovery then encloses the complete weighted
expression, nested derivative coefficients included, from the original inputs with
`enclosedPowerSlope?`. Nonfinite inputs, unsupported carriers and inconclusive bounds keep the
ordinary result. The supplied zero predicate `isZero` must inspect every differentiation
coefficient, and a zero exponent gives `0 * seed`. -/
@[expose] def powerBackward {α : Type} [Zero α] [One α] [Sub α] [Mul α] [Pow α α]
    [BEq α] [ops : QuotientArithmetic α] (isZero : α → Bool) (base exponent seed : α) : α :=
  if !ops.supported then
    (if isZero exponent then 0 else exponent * base ^ (exponent - 1)) * seed
  else if isZero exponent then (0 : α) * seed
  else
    let (shifted, shiftFlags) := ops.subWithFlags exponent 1
    let power := base ^ shifted
    let (_, powerFlags) := ops.addWithFlags power 0
    let (factor, factorFlags) := ops.mulWithFlags exponent power
    let (ordinary, productFlags) := ops.mulWithFlags factor seed
    if shiftFlags ||| powerFlags ||| factorFlags ||| productFlags == 0 &&
        !(power == 0 && !isZero base) then ordinary
    else
      match enclosedPowerSlope? base exponent seed with
      | none => ordinary
      | some recovered =>
          if ordinary == 0 && recovered == 0 then ops.copyPrimal ordinary recovered
          else recovered

/-- Joint tangent of `base ^ exponent` in both arguments,
`power * (exponentDirection * log base + exponent * (baseDirection / base))`, from the `power`
the forward pass computed and in its ordinary arithmetic order.

If an intermediate loses range, the supplied power is zero at a nonzero base, or the two terms
cancel after rounding, `enclosedPowerTangent?` rebuilds the whole positive-base expression from
the original base, exponent and directions. The supplied `power` enters only the ordinary path.
Unsupported domains, nonfinite encodings and inconclusive bounds keep the ordinary result.
-/
@[expose] def powerTangent {α : Type} [Zero α] [Add α] [Mul α] [Div α]
    [BEq α] [MathFunctions α] [ops : QuotientArithmetic α]
    (base exponent power baseDirection exponentDirection : α) : α :=
  if !ops.supported then
    power * (exponentDirection * MathFunctions.log base + exponent * (baseDirection / base))
  else
    let logarithm := MathFunctions.log base
    let (_, logarithmFlags) := ops.addWithFlags logarithm 0
    let (_, powerFlags) := ops.addWithFlags power 0
    let (left, leftFlags) := ops.mulWithFlags exponentDirection logarithm
    let (quotient, quotientFlags) := ops.divWithFlags baseDirection base
    let (right, rightFlags) := ops.mulWithFlags exponent quotient
    let (sum, sumFlags) := ops.addWithFlags left right
    let (ordinary, productFlags) := ops.mulWithFlags power sum
    let flags := logarithmFlags ||| powerFlags ||| leftFlags ||| quotientFlags |||
      rightFlags ||| sumFlags ||| productFlags
    let cancellation := sum == 0 && (left != 0 || right != 0)
    if flags == 0 && !(power == 0 && base != 0) && !cancellation then ordinary
    else
      match enclosedPowerTangent? base exponent baseDirection exponentDirection with
      | none => ordinary
      | some recovered =>
          if ordinary == 0 && recovered == 0 then ops.copyPrimal ordinary recovered
          else recovered

/-- Sigmoid or tanh slope at `x`, multiplied by the incoming gradient when one is given.

Ordinary scalars use the decaying exponential, and nested floating-point derivatives use
reciprocal cosh identities, which avoid cancellation near zero. When a checked step loses range,
the slope is rebuilt over finite rational coefficients from the half exponential, so the local
slope is not rounded to zero before the gradient is applied. If the half exponential itself loses
range, the original input and seed go to the complete coefficient enclosure `enclosedSlope?`
instead. Without a seed the result is the slope itself, with no extra multiplication. Exposed
callers need this helper visible for specialization.
-/
@[expose] def applySlope {α : Type} [Zero α] [One α] [Add α] [Mul α] [Div α] [Neg α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [BEq α] [RatCast α] [MathFunctions α]
    [ops : QuotientArithmetic α] (x : α) (seed : Option α) (isTanh : Bool) : α :=
  let inputFlags := if ops.supported && ops.depth != 0 then
    (ops.addWithFlags x 0).2 else 0
  let (slope, factor) := if ops.supported && ops.depth != 0 && inputFlags == 0 then
    let ch := MathFunctions.cosh x
    let slope := if isTanh then
      let reciprocalCosh := 1 / ch
      reciprocalCosh * reciprocalCosh
    else 1 / ((1 + 1) * (1 + ch))
    (slope, ch)
  else
    let z := MathFunctions.exp (if x > 0 then -x else x)
    let slope := if isTanh then
      let reciprocalCosh := (z + z) / (1 + z * z)
      reciprocalCosh * reciprocalCosh
    else
      let denominator := 1 + z
      z / (denominator * denominator)
    (slope, z)
  if !ops.supported then
    match seed with
    | none => slope
    | some upstream => slope * upstream
  else
    let (ordinary, productFlags) := match seed with
      | none => (slope, 0)
      | some upstream => ops.mulWithFlags slope upstream
    let (_, slopeFlags) := ops.addWithFlags slope 0
    -- A normal primal can conceal a vanished derivative coefficient.
    let sourceFlags := if ops.depth == 0 then 0 else
      let (_, factorFlags) := ops.addWithFlags factor 0
      inputFlags ||| factorFlags
    -- A unit-rounded factor can hide products of small differentiation directions, even at zero.
    let roundedUnit := ops.depth != 0 && factor == 1
    if slopeFlags ||| productFlags ||| sourceFlags == 0 && !(slope == 0) &&
        !roundedUnit then ordinary
    else
      let half : α := Rat.cast (1 / 2 : Rat)
      let (argument, argumentFlags) := ops.mulWithFlags (if x > 0 then -x else x) half
      let q := MathFunctions.exp argument
      let (_, halfFlags) := ops.addWithFlags q 0
      -- A nonzero subnormal exponential can already have lost most of its significant bits.
      let nestedFlags := if ops.depth == 0 then 0 else slopeFlags
      let enclosed := if roundedUnit || q == 0 ||
          argumentFlags ||| halfFlags ||| sourceFlags ||| nestedFlags != 0 then
        enclosedSlope? x seed isTanh else none
      match enclosed with
      | some repaired =>
          if ordinary == 0 && repaired == 0 then ops.copyPrimal ordinary repaired else repaired
      | none =>
          if q == 0 then ordinary else
            match ops.encode q, ops.encode (seed.getD 1), ops.encode (1 : α) with
            | some tail, some upstream, some one =>
                let square := QuotientCoefficients.mul tail tail
                let z := if isTanh then QuotientCoefficients.mul square square else square
                let denominator := QuotientCoefficients.add one z
                if denominator.primal == 0 then ordinary
                else
                  let slope := QuotientCoefficients.div z
                    (QuotientCoefficients.mul denominator denominator)
                  let twice := QuotientCoefficients.add slope slope
                  let slope := if isTanh then QuotientCoefficients.add twice twice else slope
                  let coefficients := QuotientCoefficients.mul slope upstream
                  match ops.decode coefficients with
                  | some repaired =>
                      if coefficients.primal == 0 && ordinary == 0 then
                        ops.copyPrimal ordinary repaired
                      else repaired
                  | none => ordinary
            | _, _, _ => ordinary

/-- Sigmoid slope `z / (1 + z)^2` from the decaying exponential `z`, which keeps a representable
tail after the activation has rounded to one, where `sigmoid x * (1 - sigmoid x)` is zero.

Nested finite-precision carriers use `1 / (2 * (1 + cosh x))` instead, which avoids cancellation
in its derivatives near zero. The standalone slope and the weighted pullback share one range
repair. The argument of `z` is `-x` or `x` by the sign test `x > 0`, never a modulus, so a complex
input keeps the algebraic formula.
-/
@[expose] def sigmoidDeriv {α : Type} [Zero α] [One α] [Add α] [Mul α] [Div α] [Neg α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [BEq α] [RatCast α] [MathFunctions α]
    [QuotientArithmetic α] (x : α) : α :=
  applySlope x none false

/-- Tanh slope as the square of the reciprocal cosh `(z + z) / (1 + z * z)`, from the decaying
exponential `z`, so no rounded square is subtracted from one.

Nested finite-precision carriers use `(1 / cosh x)^2`, whose derivatives retain small inputs
through `sinh`. Detected range loss goes to the weighted pullback's repair, which also covers
formats whose range cannot hold an intermediate `2`.
-/
@[expose] def tanhDeriv {α : Type} [Zero α] [One α] [Add α] [Mul α] [Div α] [Neg α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [BEq α] [RatCast α] [MathFunctions α]
    [QuotientArithmetic α] (x : α) : α :=
  applySlope x none true

/-- Sigmoid derivative at `x` times the incoming gradient `seed`, without first losing the
slope's small tail.

If no checked step loses range, the result is the ordinary slope times `seed`. Differentiated
carriers use reciprocal cosh identities and also check the input, the cosh factor and the slope.
A cosh factor rounded to one forces the complete enclosure (zero inputs at higher orders
included), because its derivative coefficients can already have lost a product of small
directions. Other range loss replays the weighted slope from the rounded half exponential, whose
rounding stays a separate source of error. If the input, the half argument or the half
exponential loses information, a bounded search encloses the complete expression from the
original input and accepts only agreeing finite rounded endpoints in every coefficient, so
primal-only agreement never suffices. A search that does not settle keeps the staged repair.
Unsupported carriers and nonfinite inputs use ordinary arithmetic.
-/
@[expose] def sigmoidBackward {α : Type} [Zero α] [One α] [Add α] [Mul α] [Div α] [Neg α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [BEq α] [RatCast α] [MathFunctions α]
    [QuotientArithmetic α] (x seed : α) : α :=
  applySlope x (some seed) false

/-- Tanh derivative at `x` times the incoming gradient `seed`, with intermediate range loss
repaired.

The cold path evaluates `4 * q^4 * seed / (1 + q^4)^2` from the rounded half exponential
`q = exp (-|x| / 2)`. If `q` or one of its coefficients loses range, an enclosure evaluates
`4 * exp (-2 * |x|) * seed / (1 + exp (-2 * |x|))^2` before output rounding, and an inconclusive
search keeps the staged repair. Unsupported carriers and nonfinite inputs use ordinary
arithmetic. Nested carriers share the reciprocal-cosh calculation and range checks with
`sigmoidBackward`.
-/
@[expose] def tanhBackward {α : Type} [Zero α] [One α] [Add α] [Mul α] [Div α] [Neg α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [BEq α] [RatCast α] [MathFunctions α]
    [QuotientArithmetic α] (x seed : α) : α :=
  applySlope x (some seed) true

/-- `log (1 + x)` with the rounding of `1 + x` compensated: for the rounded sum `s` it computes
`log s * (x / (s - 1))`, and `x` itself when the sum rounds to one.

Equality rather than order detects the sum rounding to one, so complex scalars take the same path.
Accuracy depends on the scalar logarithm, and the result is not correctly rounded in general. If
finite inputs overflow the addition, the logarithm of their exact coefficient sum is enclosed
before rounding. An unsupported codec or an inconclusive search keeps the ordinary arithmetic,
which may be nonfinite.
-/
@[expose] def log1p {α : Type} [One α] [Add α] [Sub α] [Mul α] [Div α] [BEq α]
    [MathFunctions α] [ops : QuotientArithmetic α] (x : α) : α :=
  let shifted := 1 + x
  let ordinary := fun (_ : Unit) => MathFunctions.log shifted * (x / (shifted - 1))
  if shifted == 1 then x
  else if shifted == x then MathFunctions.log shifted
  else if shifted - shifted == (1 : α) - 1 then ordinary ()
  else
    match ops.encode x, ops.encode (1 : α) with
    | some input, some one =>
        match enclosedLog? (QuotientCoefficients.add one input) with
        | some recovered => recovered
        | none => ordinary ()
    | _, _ => ordinary ()

/-- Chain rule of `log1p`, `seed / (1 + x)`, without first overflowing `1 + x`.

The ordinary quotient keeps its arithmetic order. If the sum or the quotient loses range, the
whole quotient is replayed over finite rational coefficients, nested derivative coefficients
included, and rounded once.
-/
@[expose] def log1pBackward {α : Type} [Zero α] [One α] [Add α] [Div α] [BEq α]
    [ops : QuotientArithmetic α] (x seed : α) : α :=
  if !ops.supported then seed / (1 + x)
  else
    let (denominator, sumFlags) := ops.addWithFlags 1 x
    let (ordinary, quotientFlags) := ops.divWithFlags seed denominator
    if sumFlags ||| quotientFlags == 0 then ordinary
    else
      match ops.encode (1 : α), ops.encode x, ops.encode seed with
      | some one, some input, some upstream =>
          let denominator := QuotientCoefficients.add one input
          if denominator.primal == 0 then ordinary
          else
            match ops.decode (QuotientCoefficients.div upstream denominator) with
            | none => ordinary
            | some recovered =>
                if ordinary == 0 && recovered == 0 then ops.copyPrimal ordinary recovered
                else recovered
      | _, _, _ => ordinary

/-- `exp x - 1` without losing small arguments to the subtraction. For `|x| < 1` it computes
`(e - 1) * (x / log e)` with `e = exp x`, and `x` itself when `e` rounds to one.

Restricting the logarithmic correction to the unit neighborhood avoids the branch ambiguity of a
complex logarithm. A zero exponential gives `-1`, which handles underflow. Accuracy depends on the
scalar `exp` and `log`.
-/
@[expose] def expm1 {α : Type} [Zero α] [One α] [Sub α] [Mul α] [Div α] [Neg α] [BEq α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [MathFunctions α] (x : α) : α :=
  let e := MathFunctions.exp x
  if MathFunctions.abs x < 1 then
    if e == 1 then x
    else if e == 0 then -1
    else (e - 1) * (x / MathFunctions.log e)
  else e - 1

end Numeric

/-- Default normalization stabilizer `1e-5`, cast once from the exact rational `1 / 100000`.

Native and configured binary contexts round the fraction without first casting its denominator,
so binary16 gets its representable tolerance even though `100000` overflows there. The cast can
still round to zero in a format with no positive value near `1e-5`, and callers using such a
format must choose their own positive tolerance.
-/
@[expose] def normalizationEpsilon {α : Type} [RatCast α] : α :=
  Rat.cast (1 / 100000 : Rat)

end TorchLean

namespace TorchLean

/-- Native binary32 implementations of the scalar transcendental interface. -/
instance instMathFunctionsFloat32 : MathFunctions Float32 where
  exp := Float32.exp
  tanh := Float32.tanh
  cosh := Float32.cosh
  sqrt := Float32.sqrt
  abs := Float32.abs
  log := Float32.log
  pi := (3.14159265358979323846 : Float).toFloat32
  cos := Float32.cos
  sin := Float32.sin
  sinh := Float32.sinh

end TorchLean
