/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.FromFloat
public import TorchLean.Tensor.Arithmetic
public import TorchLean.Tensor.Numeric.InverseTrig

/-!
# Dual

Dual numbers for forward-mode differentiation.

The runtime scalar `Dual α` computes:
- Jacobian-vector products by running a function once with dual inputs, and
- Hessian-vector products by running reverse-mode over dual scalars (forward-over-reverse).

The rules follow the primal branch. Absolute value has the tangent `Context.sign x * x'` of
`Op.abs`: a signed zero at a zero and a NaN at a NaN. The guarded square root selects zero tangent
at zero. Minimum and maximum, both `Min`/`Max` and the IEEE `Context.minimum`/`Context.maximum`
(through `Context.extremaTie`), average the two tangents where neither operand wins strictly: at a
tie or an unordered pair. These extrema choices agree with the tie and unordered-pair conventions of
`Op.max`/`Op.min` (forward and `jvp?`, which GraphM also runs), whose pullback gives each input
half of the cotangent there.
`Context.fmax` and `Context.fmin` carry the tangent of the operand they return, the one
`Context.fmaxTakesFirst` or `Context.fminTakesFirst` names, as the pullbacks of `Op.fmax` and
`Op.fmin` send the cotangent to it.
ReLU separately selects zero at its kink. `Context.clamp` composes the extrema, so it averages at an
endpoint tie, while tensor clamp at `ClampPolicy.interior` (`Op.clamp`) clears the tangent outside
the open interval, endpoints included.

`detach` removes the tangent and recursively clears differentiation metadata from the primal,
so nested dual computations respect the same boundary. A power with a constant exponent uses
$yx^{y-1}x'$, without evaluating a logarithm. A varying exponent uses
$x^y(y'\log x+yx'/x)$ on its domain. Mathematical derivative theorems state their domain
assumptions in the separate `fderiv` proof layer.

`log1p` and `expm1` follow analytic chain rules and ignore the rounding-dependent branches of
their compensated defaults, which also preserves higher derivatives when dual numbers are nested.
The tangents of `exp` and `expm1` are `TorchLean.Context.expMul x x'`, so a backend that rounds
`exp x * x'` once keeps them finite when `exp x` alone overflows. Nested duals differentiate
`expMul x w` as `exp x * (x' * w + w')`.
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open TorchLean.Tensor

/--
A dual number $x+\varepsilon\,dx$ used for forward-mode automatic differentiation.

`re` is the primal value and `du` is the tangent (directional derivative). Running a computation
once on dual inputs gives a Jacobian-vector product in the resulting `du` component.
-/
structure Dual (α : Type) where
  /-- Primal part: the value the ordinary computation would have produced. -/
  re : α
  /-- Tangent part, the coefficient of `ε`. Truncating at `ε² = 0` is what makes the arithmetic
  rules below exactly the chain rule. -/
  du : α
deriving Repr

namespace Dual

/-- Embed a primal value as a dual number with zero tangent. -/
def ofPrimal {α : Type} [Zero α] (x : α) : Dual α := ⟨x, 0⟩
/-- `Dual` is inhabited by the `default` primal value with zero tangent. -/
instance {α : Type} [Inhabited α] [Zero α] : Inhabited (Dual α) :=
  ⟨⟨default, 0⟩⟩

/--
Inject binary64 literals into the dual-number backend used by the runtime autograd engine.

A literal becomes a primal value with zero tangent/adjoint component.
-/
instance {α : Type} [FromFloat α] [Zero α] :
    FromFloat (Dual α) where
  ofFloat x mode := ofPrimal (ofFloat x mode)
  roundForValidation := FromFloat.roundForValidation (α := α)

/-- Zero dual number: $0+\varepsilon\cdot0$. -/
instance {α : Type} [Zero α] : Zero (Dual α) where
  zero := ⟨0, 0⟩
/-- One dual number: $1+\varepsilon\cdot0$. -/
instance {α : Type} [One α] [Zero α] : One (Dual α) where
  one := ⟨1, 0⟩
/-- Negation is componentwise:
$-(x+\varepsilon\,dx)=-x+\varepsilon(-dx)$. -/
instance {α : Type} [Neg α] : Neg (Dual α) where
  neg x := ⟨-x.re, -x.du⟩

/-- Addition is componentwise, so tangents add linearly. -/
instance {α : Type} [Add α] : Add (Dual α) where
  add x y := ⟨x.re + y.re, x.du + y.du⟩
/-- Subtraction is componentwise, so tangents subtract linearly. -/
instance {α : Type} [Sub α] : Sub (Dual α) where
  sub x y := ⟨x.re - y.re, x.du - y.du⟩

/-- Multiplication uses the product rule: $(xy)'=x'y+xy'$. -/
instance {α : Type} [Mul α] [Add α] : Mul (Dual α) :=
  ⟨fun x y => ⟨x.re * y.re, x.du * y.re + x.re * y.du⟩⟩

/-- Evaluate a complete quotient layer while keeping all enclosed range flags. -/
@[inline] def checkedDiv {α : Type}
    (ops : Numeric.QuotientArithmetic α) (x y : Dual α) : Dual α × UInt8 :=
  let (re, a) := ops.divWithFlags x.re y.re
  let (denom, b) := ops.mulWithFlags y.re y.re
  let (left, c) := ops.mulWithFlags x.du y.re
  let (right, d) := ops.mulWithFlags x.re y.du
  let (numerator, e) := ops.subWithFlags left right
  let (du, f) := ops.divWithFlags numerator denom
  (⟨re, du⟩, a ||| b ||| c ||| d ||| e ||| f)

/-- Lift checked arithmetic through both coefficients without discarding inner range flags. -/
@[inline] instance {α : Type} [lower : Numeric.QuotientArithmetic α] :
    Numeric.QuotientArithmetic (Dual α) where
  supported := lower.supported
  depth := lower.depth + 1
  addWithFlags x y :=
    let (re, a) := lower.addWithFlags x.re y.re
    let (du, b) := lower.addWithFlags x.du y.du
    (⟨re, du⟩, a ||| b)
  subWithFlags x y :=
    let (re, a) := lower.subWithFlags x.re y.re
    let (du, b) := lower.subWithFlags x.du y.du
    (⟨re, du⟩, a ||| b)
  mulWithFlags x y :=
    let (re, a) := lower.mulWithFlags x.re y.re
    let (left, b) := lower.mulWithFlags x.du y.re
    let (right, c) := lower.mulWithFlags x.re y.du
    let (du, d) := lower.addWithFlags left right
    (⟨re, du⟩, a ||| b ||| c ||| d)
  divWithFlags x y :=
    checkedDiv lower x y
  encode x := do
    let re ← lower.encode x.re
    let du ← lower.encode x.du
    return .dual re du
  decode | .dual re du => do
    let re ← lower.decode re
    let du ← lower.decode du
    return ⟨re, du⟩
  copyPrimal native exact := ⟨lower.copyPrimal native.re exact.re, exact.du⟩

/--
Division uses the quotient rule $(x/y)'=(x'y-xy')/y^2$ at every nesting depth.

For native binary32/64 coefficient trees, ordinary operations keep this evaluation order. If any
enclosed operation exposes a nonfinite or subnormal value, including a nonzero product or quotient
rounded to zero, the complete quotient is replayed over exact rational coefficients. Every final
coefficient is then rounded independently to its native format. A representable mixed derivative can
therefore survive an intermediate overflow or underflow at an inner depth. Replay requires all input
coefficients finite and the denominator's scalar primal nonzero. Otherwise the native quotient-rule
result is returned. An unrepresentable exact coefficient rounds to infinity or zero, and
coefficients are never clamped or deleted.

Each replay covers one quotient of the given coefficient trees. Rounding before entry is
irreversible, and ordinary rounding and cancellation outside a detected range event keep their
native semantics. Carriers without checked coefficient arithmetic evaluate $(x'y-xy')/y^2$. If $y^2$
is zero or not finite or that tangent is not finite, while the inputs and the quotient are finite
and the quotient of a nonzero numerator is nonzero, they evaluate $(x'-(x/y)y')/y$.
-/
@[inline] instance instDiv {α : Type} [Div α] [Mul α] [Sub α] [Add α] [BEq α] [Zero α]
    [Numeric.QuotientArithmetic α] : Div (Dual α) :=
  ⟨fun x y =>
    let ops : Numeric.QuotientArithmetic (Dual α) := inferInstance
    if ops.supported then
      let (result, replay) :=
        @Numeric.QuotientArithmetic.divChecked _ ops x y
      if replay then
        match Numeric.QuotientArithmetic.replay? ops.encode ops.decode x y with
        | some exact => ops.copyPrimal result exact
        | none => result
      else result
    else
      let quotient := x.re / y.re
      let denom := y.re * y.re
      let tangent := (x.du * y.re - x.re * y.du) / denom
      let needsFallback := denom == 0 || !((denom - denom) == 0) ||
        !((tangent - tangent) == 0)
      let finiteInputs := (x.re - x.re) == 0 && (x.du - x.du) == 0 &&
        (y.re - y.re) == 0 && (y.du - y.du) == 0
      let quotientRoundedAway := quotient == 0 && !(x.re == 0)
      let derivative :=
        if needsFallback && finiteInputs && (quotient - quotient) == 0 &&
            !quotientRoundedAway then
          (x.du - quotient * y.du) / y.re
        else tangent
      ⟨quotient, derivative⟩⟩

/-- Boolean equality compares primals only (tangents are treated as metadata). -/
instance {α : Type} [BEq α] : BEq (Dual α) where
  beq x y := x.re == y.re
/-- Strict order compares primals only. -/
instance {α : Type} [LT α] : LT (Dual α) where
  lt x y := x.re < y.re
/-- Non-strict order compares primals only. -/
instance {α : Type} [LE α] : LE (Dual α) where
  le x y := x.re ≤ y.re

/--
Take the larger primal and its tangent. Equal primals split the tangent equally between the two
arguments, matching the graph's elementwise maximum rule. The shared average scales before adding
and replays finite coefficients if halving loses range. Neither an overflowing raw sum nor two
rounded-zero halves should erase a representable average. An unordered pair also averages the
tangents, and its primal is the scalar's `Max.max`. For `Float` and `Float32` that is core
`maxOfLe`, which keeps a NaN first primal and drops a NaN second one.
-/
instance {α : Type} [Storage α] [Context α] : Max (Dual α) where
  max x y :=
    if x.re > y.re then x
    else if y.re > x.re then y
    else
      ⟨Max.max x.re y.re, Numeric.average x.du y.du⟩

/-- Elementwise minimum uses the same equal split at a tie as the graph's minimum rule. An
unordered pair also averages the tangents, and its primal is the scalar's `Min.min`. For `Float`
and `Float32` that is core `minOfLe`, which keeps a NaN second primal and drops a NaN first one. -/
instance {α : Type} [Storage α] [Context α] : Min (Dual α) where
  min x y :=
    if x.re < y.re then x
    else if y.re < x.re then y
    else
      ⟨Min.min x.re y.re, Numeric.average x.du y.du⟩

/-- Coerce naturals into dual numbers with zero tangent. -/
instance {α : Type} [Storage α] [Context α] : NatCast (Dual α) where
  natCast n := ⟨(n : α), 0⟩

/-- Forward-mode chain rule implementations for `MathFunctions` over dual numbers. -/
instance instMathFunctions {α : Type} [Storage α] [Context α] : MathFunctions (Dual α) where
  exp x := ⟨MathFunctions.exp x.re, Context.expMul x.re x.du⟩
  tanh x :=
    let th := MathFunctions.tanh x.re
    ⟨th, Numeric.tanhBackward x.re x.du⟩
  cosh x :=
    let ch := MathFunctions.cosh x.re
    ⟨ch, MathFunctions.sinh x.re * x.du⟩
  sinh x :=
    let sh := MathFunctions.sinh x.re
    ⟨sh, MathFunctions.cosh x.re * x.du⟩
  sqrt x :=
    let r := MathFunctions.sqrt x.re
    ⟨r, if x.re > 0 then Numeric.sqrtBackward r x.du else 0⟩
  abs x := ⟨MathFunctions.abs x.re, Context.sign x.re * x.du⟩
  log x :=
    ⟨MathFunctions.log x.re, x.du / x.re⟩
  pi := ⟨MathFunctions.pi, 0⟩
  cos x :=
    ⟨MathFunctions.cos x.re, -(MathFunctions.sin x.re) * x.du⟩
  sin x :=
    ⟨MathFunctions.sin x.re, (MathFunctions.cos x.re) * x.du⟩

/--
Power rule for `x^y` over dual numbers.

If the exponent tangent is identically zero, the tangent is $yx^{y-1}x'$, computed without a
logarithm or a division by the base, which covers polynomial derivatives at zero and at negative
inputs. The zero test inspects all nested coefficients, primals and tangents alike. A varying
exponent uses the logarithmic rule on the scalar backend's domain. When its arithmetic loses
range, a bounded positive-base enclosure combines both directions before rounding.
-/
instance {α : Type} [Storage α] [Context α] : Pow (Dual α) (Dual α) where
  pow x y :=
    let r : α := x.re ^ y.re
    let dr : α :=
      if Context.isZero y.du then
        Numeric.powerBackward Context.isZero x.re y.re x.du
      else
        Numeric.powerTangent x.re y.re r x.du y.du
    ⟨r, dr⟩

/-- Lift a scalar `Context` to dual numbers by deciding comparisons on primals. -/
instance {α : Type} [Storage α] [Context α] : Context (Dual α) where
  sqrtNat n := ⟨Context.sqrtNat (α := α) n, 0⟩
  isZero x := Context.isZero x.re && Context.isZero x.du
  defaultEpsilon := ⟨Context.defaultEpsilon (α := α), 0⟩
  decidableGT := fun x y => (Context.decidableGT) x.re y.re
  stopGradient? := some fun x => ⟨Context.stopGradient x.re, 0⟩
  ratCast value := ⟨(value : α), 0⟩
  -- `x / n` for the constant count `n` has tangent `x' / n`: each part divided once.
  divNat x n := ⟨Context.divNat x.re n, Context.divNat x.du n⟩
  -- Tangents follow the mathematical functions and ignore their rounding-compensation branches.
  sigmoid x := ⟨Context.sigmoid x.re, Numeric.sigmoidBackward x.re x.du⟩
  log1p x := ⟨Context.log1p x.re, Numeric.log1pBackward x.re x.du⟩
  expm1 x := ⟨Context.expm1 x.re, Context.expMul x.re x.du⟩
  -- The product rule for `exp x * w` with `(exp x)' = exp x * x'`.
  expMul x w := ⟨Context.expMul x.re w.re, Context.expMul x.re (x.du * w.re + w.du)⟩
  copysign x y := ⟨Context.copysign x.re y.re, Context.copysignTangent x.re y.re x.du⟩
  isNaN x := Context.isNaN x.re
  isInf x := Context.isInf x.re
  quietNaN x := ⟨Context.quietNaN x.re, x.du⟩
  -- The IEEE primal of the tie with the average tangent of `Max (Dual α)` and `Min (Dual α)`.
  -- The primal goes through `extremaTie` itself, so nested duals average at every level.
  extremaTie x y r :=
    ⟨Context.extremaTie x.re y.re r.re, Numeric.average x.du y.du⟩
  tan x :=
    let t := Context.tan x.re
    ⟨t, (1 + t * t) * x.du⟩
  exp2 x :=
    let y := Context.exp2 x.re
    ⟨y, y * MathFunctions.log (1 + 1 : α) * x.du⟩
  log2 x := ⟨Context.log2 x.re, x.du / (x.re * MathFunctions.log (1 + 1 : α))⟩
  log10 x := ⟨Context.log10 x.re, x.du / (x.re * MathFunctions.log ((10 : Nat) : α))⟩
  -- The chain rules of `Tensor.asinhBackward`, `acoshBackward`, `atanhBackward` and
  -- `hypotBackward`, so that a tangent is the registered operator's JVP.
  asinh x := ⟨Context.asinh x.re,
    Numeric.asinhBackward (Context.hypot x.re 1) x.du⟩
  acosh x := ⟨Context.acosh x.re, Numeric.acoshBackward x.re x.du⟩
  atanh x := ⟨Context.atanh x.re, Numeric.atanhBackward x.re x.du⟩
  -- The tangent of the radius is the radial component of the tangent, zero at the origin, and a
  -- zero where the radius overflows and the products `x x'` and `y y'` do not.
  hypot x y :=
    let h := Context.hypot x.re y.re
    ⟨h, Numeric.hypotBackward x.re h x.du +
      Numeric.hypotBackward y.re h y.du⟩

/-- Integral roundings are locally constant, so their tangents are zero. `fmod x y` is `x - n y`
for the truncated exact quotient `n`, locally constant, so its tangent is `x' - y' n`, with `n`
from `Context.fmodQuotient x y`, exact while `|n| < 2 ^ (p - 2)` at precision `p` (PyTorch's
truncates the rounded quotient). -/
instance {α : Type} [Storage α] [Context α] [Rounding α] : Rounding (Dual α) where
  floor x := ⟨Rounding.floor x.re, 0⟩
  ceil x := ⟨Rounding.ceil x.re, 0⟩
  trunc x := ⟨Rounding.trunc x.re, 0⟩
  round x := ⟨Rounding.round x.re, 0⟩
  fmod x y := ⟨Rounding.fmod x.re y.re, x.du - y.du * Context.fmodQuotient x.re y.re⟩

/-- The error function with tangent `2 / √π * exp (-x²) * x'`. -/
instance {α : Type} [Storage α] [Context α] [Erf α] : Erf (Dual α) where
  erf x :=
    ⟨Erf.erf x.re, (1 + 1) / MathFunctions.sqrt (MathFunctions.pi : α) *
      MathFunctions.exp (-(x.re * x.re)) * x.du⟩

/--
Differentiate the polar angle in real coordinates away from the origin and branch cut, and the
inverse trigonometric functions with the tangent schedules of the registered operators' JVPs.

Scaling both coordinates avoids squaring a large magnitude. At the origin, where the angle has no
classical derivative, the tangent is zero. The primal keeps the component backend's signed-zero
and branch-cut convention. The shared tangent rule combines the original finite coefficients if
scaled seeds lose range or their products cancel. The tangents of `atan`, `asin` and `acos` are
`dx / (x * x + 1)`, or `dx / x / x` where that denominator is not finite, `dx / √((1 - x)(1 + x))`
and `-(dx / √((1 - x)(1 + x)))`, as `Tensor.atanBackward`, `asinBackward` and `acosBackward` compute
them.
-/
instance instInverseTrig {α : Type} [Storage α] [Context α] [InverseTrig α] :
    InverseTrig (Dual α) where
  atan2 y x :=
    ⟨InverseTrig.atan2 y.re x.re,
      Numeric.angleTangent x.re y.re x.du y.du⟩
  atan x := ⟨InverseTrig.atan x.re, Numeric.atanBackward x.re x.du⟩
  asin x := ⟨InverseTrig.asin x.re, x.du / MathFunctions.sqrt ((1 - x.re) * (1 + x.re))⟩
  acos x := ⟨InverseTrig.acos x.re, -(x.du / MathFunctions.sqrt ((1 - x.re) * (1 + x.re)))⟩

end Dual

namespace DualTensor

/--
Combine a primal tensor and a tangent tensor into a dual tensor, the tensor-level analogue of
`Dual.mk`.
-/
def withTangents {α : Type} [Storage α]
    [Storage (Dual α)]
    {s : Shape} (primal tangent : Tensor α s) : Tensor (Dual α) s :=
  Tensor.zipWith Dual.mk primal tangent

/-- Project the tangent part of a dual tensor. -/
def tangent {α : Type} [Storage α]
    [Storage (Dual α)] {s : Shape}
    (tensor : Tensor (Dual α) s) : Tensor α s :=
  Tensor.map Dual.du tensor

end DualTensor

end TorchLean
