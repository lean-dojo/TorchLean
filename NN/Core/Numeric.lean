/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Numerics.Capabilities.Elementary
public import NN.Core.Numeric.Enclosure

/-!
# Foundational Numeric Interfaces

This module re-exports FloatLib's elementary-function interface and supplies TorchLean's native
binary32 instance, native scalar casts, compensated `log1p`/`expm1` formulas, and input-based
sigmoid/tanh slopes and weighted pullbacks. Nested floating-point differentiation uses reciprocal
cosh identities to retain small derivative coefficients near zero. The pullbacks reuse finite
coefficient arithmetic when an intermediate slope or product loses range. Weighted square-root
quotients use the same finite coefficient machinery. If the half exponential
itself loses range, a bounded enclosure search evaluates the complete expression before rounding.
This module knows nothing about tensors, model composition, or runtimes.

`MathFunctions` names the transcendental operations used by numerical code. Integer and rational
constants use the standard numerical interfaces. The broader neural-model interface, `Context`,
lives in `NN.Spec.Core.Context`.

FloatLib owns the single `MathFunctions` class and its `Float` and `ℝ` instances. Its elementary
capability module imports real analysis; this facade does not define a second class or duplicate
those instances.
-/

public section

export FloatLib.Numerics (MathFunctions)

namespace MathFunctions

export FloatLib.Numerics.MathFunctions (exp tanh cosh sqrt abs log pi cos sin sinh)

end MathFunctions

namespace TorchLean

namespace Numeric

/-- Logistic sigmoid with a nonpositive exponential argument on either branch.

If a checked sum loses range, finite coefficient arithmetic reconstructs the numerator and
denominator together. A valid configured format need not represent `2`, although it represents
the sigmoid's result at zero. Unsupported carriers retain the ordinary branch formula.
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

namespace Internal

/-- Average two derivative coefficients without losing a representable result during halving.

Ordinary products and addition keep their order. If a checked operation loses range, replay
the complete average over finite coefficients and round only its result. The rational constant
does not require the scalar type to represent `2`. Nonfinite inputs keep the ordinary result. -/
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

/-- Apply the square-root chain rule using the root already computed by the forward pass.

`factorFirst` preserves the reverse rule's local-factor schedule; dual numbers divide the seed
directly. If checked arithmetic loses range, replay the complete weighted quotient over finite
coefficients. The constant `2` is then exact even when its scalar cast overflows. This recovers
the quotient for the supplied, rounded root, not the analytic derivative before root rounding.
Nonfinite inputs retain the ordinary result. -/
@[expose] def sqrtBackward {α : Type} [NatCast α] [One α] [Mul α] [Div α]
    [ops : QuotientArithmetic α] (root seed : α) (factorFirst : Bool := false) : α :=
  let two : α := (2 : Nat)
  if !ops.supported then
    if factorFirst then (1 / (two * root)) * seed else seed / (two * root)
  else
    let (denominator, denominatorFlags) := ops.mulWithFlags two root
    let (ordinary, flags) :=
      if factorFirst then
        let (factor, factorFlags) := ops.divWithFlags 1 denominator
        let (value, productFlags) := ops.mulWithFlags factor seed
        (value, factorFlags ||| productFlags)
      else ops.divWithFlags seed denominator
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
                -- Rational zero has no sign; retain it from a quotient with the finite root.
                if seedCoefficients.primal == 0 then ops.copyPrimal (seed / root) value
                else value
      | _, _ => ordinary

/-- Angular directional derivative at `(x, y)` in direction `(dx, dy)`.

Scaling the coordinates avoids squaring large inputs. If a checked intermediate loses range,
or the scaled products cancel, evaluate the complete finite quotient from the original inputs
before rounding. A scaled seed can overflow even when the combined derivative is representable.
The origin keeps its selected zero tangent; nonfinite inputs retain the ordinary formula.
This does not assert differentiability across the principal angle's branch cut. -/
@[expose] def angleTangent {α : Type} [Zero α] [Add α] [Sub α] [Mul α] [Div α]
    [BEq α] [Max α] [MathFunctions α] [ops : QuotientArithmetic α] (x y dx dy : α) : α :=
  let scale := max (MathFunctions.abs x) (MathFunctions.abs y)
  if scale == 0 then 0 else
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

/-- Apply a constant-exponent power rule without rounding an out-of-range local factor first.

The ordinary arithmetic order is unchanged unless a checked intermediate loses range. Recovery
encloses the complete weighted expression from the original inputs, including nested derivative
coefficients. Nonfinite inputs, unsupported carriers and inconclusive bounds retain the ordinary
result. The supplied zero predicate must inspect every differentiation coefficient. -/
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

/-- Joint power tangent, retaining the forward pass's primal and ordinary arithmetic order.

If an intermediate loses range or the two terms cancel after rounding, bounded coefficient
enclosures reconstruct the entire positive-base expression from the original base, exponent,
and directions. The supplied forward value is used only by the ordinary path. Unsupported
domains, nonfinite encodings and inconclusive bounds retain that ordinary result.
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

/-- Evaluate a sigmoid or tanh slope, optionally applying an incoming gradient.

Reciprocal cosh identities avoid cancellation near zero in nested floating-point derivatives.
Ordinary scalar evaluation retains the decaying-exponential schedule. The half exponential
avoids rounding the local slope to zero before applying its incoming gradient. If that
intermediate loses range, the original input and seed feed a complete coefficient enclosure
instead. An absent seed returns the slope itself, without an extra multiplication.
Exposed callers need this helper visible for specialization.
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

end Internal

/-- Sigmoid slope retaining small activation and derivative tails.

Unlike `sigmoid x * (1 - sigmoid x)`, this retains a representable tail after the activation has
rounded to one. Nested finite-precision carriers instead use `1 / (2 * (1 + cosh x))` to avoid
cancellation in its derivatives near zero. The same range repair serves the standalone slope and
weighted pullback. Selecting the sign of the exponential argument preserves the complex algebraic
formula; taking its modulus would not.
-/
@[expose] def sigmoidDeriv {α : Type} [Zero α] [One α] [Add α] [Mul α] [Div α] [Neg α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [BEq α] [RatCast α] [MathFunctions α]
    [QuotientArithmetic α] (x : α) : α :=
  Internal.applySlope x none false

/-- Tanh slope evaluated as the square of a scaled reciprocal cosh.

The ordinary scalar formula doubles the decaying exponential before squaring, instead of
subtracting a rounded square from one. Nested finite-precision carriers use `(1 / cosh x)^2`,
whose derivatives retain small inputs through `sinh`. Detected range loss uses the weighted
pullback's repair; this also covers formats whose range cannot hold an intermediate `2`.
-/
@[expose] def tanhDeriv {α : Type} [Zero α] [One α] [Add α] [Mul α] [Div α] [Neg α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [BEq α] [RatCast α] [MathFunctions α]
    [QuotientArithmetic α] (x : α) : α :=
  Internal.applySlope x none true

/-- Apply the sigmoid derivative to an incoming gradient without first losing its small tail.

Normal checked products retain the selected slope and multiplication. Differentiated carriers
use reciprocal cosh identities and check the input, hyperbolic factor and slope for range loss.
A factor rounded to one also requires the complete enclosure: its derivative coefficients can
already have lost a product of small directions. This includes zero inputs at higher orders.
Detected range loss otherwise replays the weighted slope from a rounded half exponential.
If the input, half argument or exponential loses information, a bounded search encloses the
complete expression from the original input and accepts only agreeing finite rounded coefficients.
The search may fail to settle, in which case the staged repair is retained. Nested carriers use
every coefficient, not primal-only equality. Unsupported carriers and nonfinite inputs retain
ordinary arithmetic.
The half-exponential approximation remains a separate source of error when that path is used.
-/
@[expose] def sigmoidBackward {α : Type} [Zero α] [One α] [Add α] [Mul α] [Div α] [Neg α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [BEq α] [RatCast α] [MathFunctions α]
    [QuotientArithmetic α] (x seed : α) : α :=
  Internal.applySlope x (some seed) false

/-- Apply the tanh derivative to an incoming gradient, repairing intermediate range loss.

The cold path evaluates `4 * q^4 * seed / (1 + q^4)^2` from the rounded half exponential
`q = exp (-|x| / 2)`. If `q` or one of its coefficients loses range, an enclosure evaluates
`4 * exp (-2 * |x|) * seed / (1 + exp (-2 * |x|))^2` before output rounding. An inconclusive
search retains the staged repair. Unsupported carriers and nonfinite inputs use ordinary arithmetic.
Nested carriers share the reciprocal-cosh calculation and range checks with `sigmoidBackward`.
-/
@[expose] def tanhBackward {α : Type} [Zero α] [One α] [Add α] [Mul α] [Div α] [Neg α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)] [BEq α] [RatCast α] [MathFunctions α]
    [QuotientArithmetic α] (x seed : α) : α :=
  Internal.applySlope x (some seed) true

/-- Compensate rounding in `1 + x` before taking its logarithm.

This retains a small argument when the addition rounds to one. Accuracy still depends on the
scalar logarithm; the formula is not a correct-rounding guarantee. If finite inputs overflow
the addition, enclose the logarithm of their exact coefficient sum before rounding. An
unsupported codec or inconclusive search retains ordinary arithmetic, which may be nonfinite.
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
        match Internal.enclosedLogarithm? (QuotientCoefficients.add one input) with
        | some recovered => recovered
        | none => ordinary ()
    | _, _ => ordinary ()

/-- Apply the logarithm's chain rule without first overflowing `1 + x`.

The ordinary quotient keeps its arithmetic order. Range loss replays the complete denominator
and incoming gradient from finite coefficients, including nested differentiation metadata.
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

/-- Evaluate `exp x - 1` without losing small arguments to subtraction.

The logarithmic correction is restricted to the unit neighborhood, avoiding complex logarithm
branch ambiguity. The zero-exponential branch handles underflow.
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

/-- Default normalization stabilizer, obtained by casting the exact rational `1e-5` once.

Native and configured binary contexts round the fraction without first casting its denominator.
This preserves the representable binary16 tolerance even though `100000` itself overflows there.
The result can still be zero in a format that cannot represent a nearby positive value; callers
using such a format must choose an explicit positive tolerance.
-/
@[expose] def normalizationEpsilon {α : Type} [RatCast α] : α :=
  Rat.cast (1 / 100000 : Rat)

end TorchLean

/-- Native binary32 implementations of the scalar transcendental interface. -/
instance : MathFunctions Float32 where
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

/-- Cast naturals into Lean's host `Float`. -/
instance : NatCast Float where
  natCast := Float.ofNat

/-- Round naturals directly to binary32, without an intermediate binary64 rounding.

Machine-sized inputs use Lean's native integer conversion; larger naturals use its binary32 model.
-/
instance : NatCast Float32 where
  natCast n :=
    if n < UInt64.size then
      (UInt64.ofNat n).toFloat32
    else
      Float32.ofModel (Float32.Model.ofNat n)

/-- Round a rational once to native binary64 through FloatLib's binary model. -/
instance : RatCast Float where
  ratCast value := TorchLean.Numeric.QuotientArithmetic.round value

/-- Round a rational directly to native binary32, without an intermediate binary64 rounding. -/
instance : RatCast Float32 where
  ratCast value := TorchLean.Numeric.QuotientArithmetic.round value
