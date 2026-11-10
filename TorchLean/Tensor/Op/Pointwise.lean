/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Pointwise operators

`PointwiseOp` lists the pointwise operators of core ATen and `jax.lax`, with the static attributes
PyTorch passes them, such as the rounding mode of `torch.div`. They share one shape rule: every
operand and the result have the same shape, so one part of the op table that the captures read
(`TorchLean.IR.table`), keyed by `PointwiseOp.name` and `PointwiseOp.keyword`, describes them all.
In a graph over one scalar type, a comparison or a predicate returns `1` or `0`, and `where` and the
logical operators read a nonzero operand, NaN included, as true, as PyTorch does with a
floating-point condition.

The op table lists an operator on operands at a scalar type as the registered operators
(`TorchLean.Op`) of its fragment (`TorchLean.IR.Pointwise.fragment?`). Where one of them needs a
capability the scalar type lacks (`Op.Capabilities`), such as the integral roundings or `erf`, the
operator has no implementation at that type. The bitwise operators, the shifts,
`population_count`, `count_leading_zeros` and `mulhi` act on the words of integer types, and
`cbrt` and `nextafter` have registered operators only at binary formats. The op table lists these
as the operators of their element types (`TorchLean.IR.Pointwise.typed?`).

We wrote `remainderSchedule` and `divFloorSchedule` by hand from CPython 3.12's `float_rem` and
`_float_div_mod` (`Objects/floatobject.c`), following their steps. They depart from CPython at a
zero divisor, where Python raises and they return torch's IEEE results. `divFloorSchedule` also
departs at an infinite dividend over a finite divisor: it returns the infinite quotient, where
Python's `//` and torch both give a NaN. The scalar path of torch 2.8's CPU floor division,
`c10::div_floor_floating` (`c10/util/generic_math.h`), takes CPython's steps, and its NOTE [Floor
Division in Python] cites that code. `TorchLeanTest.IR.Pointwise` checks `div` in its three modes
against eager PyTorch's CPU words in binary32 and lists two departures, flooring at `(-∞, 2)` and
truncating at `(1, 0.1)`. `TorchLeanTest.Tensor.Elementwise` checks `Context.remainder` against
CPython 3.12's `%` at 21 operand pairs, with NumPy's NaN at the one zero divisor.

## References

- PyTorch Core ATen IR (PyTorch 2.8 docs):
  `https://docs.pytorch.org/docs/2.8/torch.compiler_ir.html`
- `torch.div` (PyTorch 2.8 docs): `https://docs.pytorch.org/docs/2.8/generated/torch.div.html`
- `torch.nn.functional.gelu` (PyTorch 2.8 docs):
  `https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.gelu.html`
- `torch.remainder` (PyTorch 2.8 docs):
  `https://docs.pytorch.org/docs/2.8/generated/torch.remainder.html`
- Dan Hendrycks and Kevin Gimpel, 2016, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415,
  Section 2
- CPython 3.12, `Objects/floatobject.c`, `float_rem` and `_float_div_mod`:
  `https://github.com/python/cpython/blob/v3.12.0/Objects/floatobject.c`
- PyTorch 2.8.0, `c10/util/generic_math.h`, `c10::div_floor_floating`:
  `https://github.com/pytorch/pytorch/blob/v2.8.0/c10/util/generic_math.h`
- PyTorch 2.8.0, `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp`, `remainder_kernel`:
  `https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp`
-/

@[expose] public section

namespace TorchLean

/-- Rounding applied to a quotient, as the `rounding_mode` argument of `torch.div`. -/
inductive DivRounding where
  /-- True division, PyTorch's `rounding_mode=None`. -/
  | none
  /-- Truncating division, `rounding_mode="trunc"`: the flooring division of `|a|` by `|b|` with
  the sign of `a / b` (`trunc (a / b)` over the reals,
  `TorchLean.Tensor.copysign_floorDivide_abs_real`). In a binary format it is a NaN where `a / b`
  is one, `a / b` where that is infinite, and a zero with the sign of `a / b` where `|a| < |b|`, as
  in PyTorch, and `9` at `(1, 0.1)`, where PyTorch truncates the rounded quotient to `10`. -/
  | trunc
  /-- Flooring division `divFloorSchedule`, PyTorch's `rounding_mode="floor"`: the floor of
  `a / b` over the reals. In a binary format the exact remainder corrects the rounded quotient (`9`
  at `(1, 0.1)`), an infinite dividend over a finite divisor gives the infinite quotient, and a
  nonzero finite `a` over an infinite `b` of the other sign gives `-1`. -/
  | floor
  deriving Repr, DecidableEq, Inhabited

/-- The formula selected by the `approximate` argument of `torch.nn.functional.gelu`. Both formulas
are from Section 2 of Hendrycks and Gimpel (2016). -/
inductive GeluApproximation where
  /-- `x Φ(x)`, with the standard normal distribution function `Φ` written through `erf`. -/
  | none
  /-- The tanh approximation `0.5 x (1 + tanh (√(2 / π) (x + 0.044715 x³)))`. -/
  | tanh
  deriving Repr, DecidableEq, Inhabited

/-- Pointwise operators of core ATen and `jax.lax`, which the op table of the captures
(`TorchLean.IR.table`) lists as one family. -/
inductive PointwiseOp where
  /-- Negation, `torch.neg`. -/
  | neg
  /-- Reciprocal square root, `torch.rsqrt`. -/
  | rsqrt
  /-- Gauss error function, `torch.erf`. -/
  | erf
  /-- Gaussian error linear unit, `torch.nn.functional.gelu`. -/
  | gelu (approximate : GeluApproximation)
  /-- Division with an optional rounding of the quotient, `torch.div`. -/
  | div (rounding : DivRounding)
  /-- Power with a tensor exponent, `torch.pow`. -/
  | pow
  /-- Lower bound with a scalar bound, `torch.clamp_min`: the bound where the operand is below
  it, and a NaN where either is one. -/
  | clampMin
  /-- `1` above zero, `-1` below, and the operand at a zero or a NaN, as `jax.lax.sign`.
  `torch.sign` gives `0` at `-0` and at a NaN. -/
  | sign
  /-- `x * x`, `torch.square`. -/
  | square
  /-- Rounding toward negative infinity, `torch.floor`. -/
  | floor
  /-- Rounding toward positive infinity, `torch.ceil`. -/
  | ceil
  /-- Rounding to the nearest integer, ties to even, `torch.round`. -/
  | round
  /-- Rounding toward zero, `torch.trunc`. -/
  | trunc
  /-- `x - trunc x`, `torch.frac`. -/
  | frac
  /-- `2 ^ x`, `torch.exp2`. -/
  | exp2
  /-- `exp x - 1`, `torch.expm1`. -/
  | expm1
  /-- Base-two logarithm, `torch.log2`. -/
  | log2
  /-- Base-ten logarithm, `torch.log10`. -/
  | log10
  /-- `log (1 + x)`, `torch.log1p`. -/
  | log1p
  /-- Tangent, `torch.tan`. -/
  | tan
  /-- Arcsine, `torch.asin`. -/
  | asin
  /-- Arccosine, `torch.acos`. -/
  | acos
  /-- Arctangent, `torch.atan`. -/
  | atan
  /-- Hyperbolic sine, `torch.sinh`. -/
  | sinh
  /-- Hyperbolic cosine, `torch.cosh`. -/
  | cosh
  /-- Inverse hyperbolic sine, `torch.asinh`. -/
  | asinh
  /-- Inverse hyperbolic cosine, `torch.acosh`. -/
  | acosh
  /-- Inverse hyperbolic tangent, `torch.atanh`. -/
  | atanh
  /-- `1` at a NaN and `0` elsewhere, `torch.isnan`. -/
  | isNaN
  /-- `1` at an infinity and `0` elsewhere, `torch.isinf`. -/
  | isInf
  /-- `1` at a finite value and `0` elsewhere, `torch.isfinite`. -/
  | isFinite
  /-- `1` at zero and `0` elsewhere, `torch.logical_not`. -/
  | logicalNot
  /-- C's truncating remainder, with the sign of the dividend, `torch.fmod`. -/
  | fmod
  /-- Python's flooring remainder, `torch.remainder`: C's `fmod`, plus the divisor where that is
  nonzero with the other sign. A zero result has the divisor's sign. `torch.remainder` (PyTorch 2.8
  docs) documents the divisor's sign, but its CPU `remainder_kernel` keeps `fmod`'s zero, which has
  the dividend's sign, so `torch.remainder(-4.0, 2.0)` is `-0` where Python's `-4.0 % 2.0` is
  `+0`. -/
  | remainder
  /-- Polar angle of `(other, input)`, `torch.atan2`. -/
  | atan2
  /-- `√(x² + y²)`, `torch.hypot`. -/
  | hypot
  /-- Magnitude of the first operand with the sign of the second, `torch.copysign`. -/
  | copysign
  /-- The larger operand, NaN-propagating, `torch.maximum`. -/
  | maximum
  /-- The smaller operand, NaN-propagating, `torch.minimum`. -/
  | minimum
  /-- The larger operand, ignoring a NaN against a number, `torch.fmax`. -/
  | fmax
  /-- The smaller operand, ignoring a NaN against a number, `torch.fmin`. -/
  | fmin
  /-- `1` where the operands are equal, `torch.eq`. -/
  | eq
  /-- `1` where the operands differ, `torch.ne`. -/
  | ne
  /-- `1` where the first operand is smaller, `torch.lt`. -/
  | lt
  /-- `1` where the first operand is smaller or equal, `torch.le`. -/
  | le
  /-- `1` where the first operand is larger, `torch.gt`. -/
  | gt
  /-- `1` where the first operand is larger or equal, `torch.ge`. -/
  | ge
  /-- `1` where both operands are nonzero, `torch.logical_and`. -/
  | logicalAnd
  /-- `1` where either operand is nonzero, `torch.logical_or`. -/
  | logicalOr
  /-- `1` where exactly one operand is nonzero, `torch.logical_xor`. -/
  | logicalXor
  /-- The second operand where the first is nonzero and the third elsewhere, `torch.where`. -/
  | select
  /-- The first operand limited by the second below and the third above, `torch.clamp`. -/
  | clamp
  /-- Linear interpolation from the first to the second operand by the third, `torch.lerp`. -/
  | lerp
  /-- `x + value * a * b` for operands `x, a, b, value`, `torch.addcmul`. -/
  | addcmul
  /-- `x + value * (a / b)` for operands `x, a, b, value`, `torch.addcdiv`. -/
  | addcdiv
  /-- The first operand with a NaN, `+∞` and `-∞` replaced by the other three operands,
  `torch.nan_to_num`. -/
  | nanToNum
  /-- The real cube root, IEEE 754-2019 `rootn(x, 3)` (StableHLO's `cbrt`, `lax.cbrt`): `-0` at
  `-0` and `±∞` at `±∞`. -/
  | cbrt
  /-- The value next to the first operand toward the second, C's `nextafter` (`torch.nextafter`,
  `lax.nextafter`): a NaN where either operand is one, the second operand where they are equal,
  and otherwise IEEE 754-2019 `nextUp` or `nextDown` of the first. -/
  | nextafter
  /-- Bitwise and of integers, `torch.bitwise_and`, `lax.bitwise_and`. -/
  | bitwiseAnd
  /-- Bitwise or of integers, `torch.bitwise_or`, `lax.bitwise_or`. -/
  | bitwiseOr
  /-- Bitwise exclusive or of integers, `torch.bitwise_xor`, `lax.bitwise_xor`. -/
  | bitwiseXor
  /-- Bitwise complement of an integer, `torch.bitwise_not`, `lax.bitwise_not`. -/
  | bitwiseNot
  /-- The first integer shifted left by the second, `torch.bitwise_left_shift`, `lax.shift_left`.
  It is zero once the amount, its word read unsigned, reaches the width. -/
  | bitwiseLeftShift
  /-- The first integer shifted right by the second, filled with its top bit,
  `lax.shift_right_arithmetic` (`torch.bitwise_right_shift` of a signed type). Once the amount,
  its word read unsigned, reaches the width, the top bit fills every position. -/
  | shiftRightArithmetic
  /-- The first integer shifted right by the second, filled with zeros, `lax.shift_right_logical`
  (`torch.bitwise_right_shift` of an unsigned type). It is zero once the amount, its word read
  unsigned, reaches the width. -/
  | shiftRightLogical
  /-- The number of one bits of an integer's word, `lax.population_count`. -/
  | populationCount
  /-- The number of zero bits above the highest one bit of an integer's word, its width at zero,
  `lax.clz`. -/
  | countLeadingZeros
  /-- The upper half of the exact product of two integers of one width, signed or unsigned as
  their type (the floor of the product over `2 ^ width`), `lax.mulhi`. -/
  | mulhi
  deriving Repr, DecidableEq, Inhabited

namespace PointwiseOp

/-- Number of tensor operands. -/
def arity : PointwiseOp → Nat
  | .neg | .rsqrt | .erf | .gelu _ | .sign | .square | .floor | .ceil | .round | .trunc | .frac
    | .exp2 | .expm1 | .log2 | .log10 | .log1p | .tan | .asin | .acos | .atan | .sinh | .cosh
    | .asinh | .acosh | .atanh | .isNaN | .isInf | .isFinite | .logicalNot | .cbrt | .bitwiseNot
    | .populationCount | .countLeadingZeros => 1
  | .div _ | .pow | .clampMin | .fmod | .remainder | .atan2 | .hypot | .copysign | .maximum
    | .minimum | .fmax | .fmin | .eq | .ne | .lt | .le | .gt | .ge | .logicalAnd | .logicalOr
    | .logicalXor | .nextafter | .bitwiseAnd | .bitwiseOr | .bitwiseXor | .bitwiseLeftShift
    | .shiftRightArithmetic | .shiftRightLogical | .mulhi => 2
  | .select | .clamp | .lerp => 3
  | .addcmul | .addcdiv | .nanToNum => 4

/-- Every operator takes at least one operand. -/
theorem arity_pos (op : PointwiseOp) : 0 < op.arity := by
  cases op <;> simp [arity]

/-- The operator's name without attributes: ATen's name where PyTorch has the operator, and the
`jax.lax` or StableHLO name otherwise, such as `population_count`. The op table
(`TorchLean.IR.table`) keys the operator's entries by it, so renaming an operator here changes the
nodes the captures match. Diagnostics group by it too. -/
def name : PointwiseOp → String
  | .neg => "neg"
  | .rsqrt => "rsqrt"
  | .erf => "erf"
  | .gelu _ => "gelu"
  | .div _ => "div"
  | .pow => "pow"
  | .clampMin => "clamp_min"
  | .sign => "sign"
  | .square => "square"
  | .floor => "floor"
  | .ceil => "ceil"
  | .round => "round"
  | .trunc => "trunc"
  | .frac => "frac"
  | .exp2 => "exp2"
  | .expm1 => "expm1"
  | .log2 => "log2"
  | .log10 => "log10"
  | .log1p => "log1p"
  | .tan => "tan"
  | .asin => "asin"
  | .acos => "acos"
  | .atan => "atan"
  | .sinh => "sinh"
  | .cosh => "cosh"
  | .asinh => "asinh"
  | .acosh => "acosh"
  | .atanh => "atanh"
  | .isNaN => "isnan"
  | .isInf => "isinf"
  | .isFinite => "isfinite"
  | .logicalNot => "logical_not"
  | .fmod => "fmod"
  | .remainder => "remainder"
  | .atan2 => "atan2"
  | .hypot => "hypot"
  | .copysign => "copysign"
  | .maximum => "maximum"
  | .minimum => "minimum"
  | .fmax => "fmax"
  | .fmin => "fmin"
  | .eq => "eq"
  | .ne => "ne"
  | .lt => "lt"
  | .le => "le"
  | .gt => "gt"
  | .ge => "ge"
  | .logicalAnd => "logical_and"
  | .logicalOr => "logical_or"
  | .logicalXor => "logical_xor"
  | .select => "where"
  | .clamp => "clamp"
  | .lerp => "lerp"
  | .addcmul => "addcmul"
  | .addcdiv => "addcdiv"
  | .nanToNum => "nan_to_num"
  | .cbrt => "cbrt"
  | .nextafter => "nextafter"
  | .bitwiseAnd => "bitwise_and"
  | .bitwiseOr => "bitwise_or"
  | .bitwiseXor => "bitwise_xor"
  | .bitwiseNot => "bitwise_not"
  | .bitwiseLeftShift => "bitwise_left_shift"
  | .shiftRightArithmetic => "shift_right_arithmetic"
  | .shiftRightLogical => "shift_right_logical"
  | .populationCount => "population_count"
  | .countLeadingZeros => "count_leading_zeros"
  | .mulhi => "mulhi"

/-- The operator's attribute as PyTorch's keyword argument and its value: GELU's `approximate` and
a rounded division's `rounding_mode`. True division has none. The op table (`TorchLean.IR.table`)
writes it as the `keyword` of the operator's entry, which a captured node matches when it has that
field with that value. -/
def keyword : PointwiseOp → Option (String × String)
  | .gelu .none => some ("approximate", "none")
  | .gelu .tanh => some ("approximate", "tanh")
  | .div .trunc => some ("rounding_mode", "trunc")
  | .div .floor => some ("rounding_mode", "floor")
  | _ => none

/-- Operator name with its attribute, for diagnostics: `div(rounding_mode=floor)`. -/
def describe (op : PointwiseOp) : String :=
  match op.keyword with
  | some (key, value) => s!"{op.name}({key}={value})"
  | none => op.name

/-- Every operator with every attribute value, in declaration order. -/
def all : List PointwiseOp :=
  [.neg, .rsqrt, .erf, .gelu .none, .gelu .tanh, .div .none, .div .trunc, .div .floor, .pow,
    .clampMin, .sign, .square, .floor, .ceil, .round, .trunc, .frac, .exp2, .expm1, .log2, .log10,
    .log1p, .tan, .asin, .acos, .atan, .sinh, .cosh, .asinh, .acosh, .atanh, .isNaN, .isInf,
    .isFinite, .logicalNot, .fmod, .remainder, .atan2, .hypot, .copysign, .maximum, .minimum,
    .fmax, .fmin, .eq, .ne, .lt, .le, .gt, .ge, .logicalAnd, .logicalOr, .logicalXor, .select,
    .clamp, .lerp, .addcmul, .addcdiv, .nanToNum, .cbrt, .nextafter, .bitwiseAnd, .bitwiseOr,
    .bitwiseXor, .bitwiseNot, .bitwiseLeftShift, .shiftRightArithmetic, .shiftRightLogical,
    .populationCount, .countLeadingZeros, .mulhi]

/-- The enumeration is complete. -/
theorem mem_all (op : PointwiseOp) : op ∈ all := by
  cases op <;> (try cases ‹GeluApproximation›) <;> (try cases ‹DivRounding›) <;> simp [all]

end PointwiseOp

/--
Python's `%` from C's `fmod`, as CPython computes it for floats: the truncating remainder `r`,
plus `b` when `r` is nonzero and its sign differs from the sign of `b`, and a zero result with the
sign of `b`. `r` is zero only at a nonzero `b` that is not a NaN, so `b < 0` reads that sign. In
round to nearest `0 + r` is `+0` and `-(0 - r)` is `-0` for either zero, and both keep the
differentiation metadata of `r`.
-/
def remainderSchedule {α : Type} [Zero α] [Neg α] [Add α] [Sub α] [LT α]
    [DecidableRel (α := α) (· < ·)] [BEq α] (fmod : α → α → α) (a b : α) : α :=
  let r := fmod a b
  if r == 0 then (if b < 0 then -(0 - r) else 0 + r)
  else if decide (r < 0) != decide (b < 0) then r + b else r

/--
PyTorch's flooring division, `c10::div_floor_floating`, over the operations it uses, with the floor
of the quotient at a zero divisor and at an infinite dividend.

`fmod` is C's truncating remainder, `floor` rounds toward negative infinity, and `signedZero q` is
zero with the sign of `q`. At a zero divisor the result is `floor (a / b)`. It is also
`floor (a / b)` when the remainder is a NaN and `a / b` is not, that is, at an infinite dividend
over a finite divisor. PyTorch returns a NaN there. Otherwise the remainder corrects the quotient,
as Python's `//` does, and the final comparison with `half` repairs a quotient that rounded across
an integer. Over the reals the schedule computes `⌊a / b⌋` (`divFloorSchedule_real`). A binary
format runs it with its rounded operations, which can miss the floor by one at large quotients: in
binary64, -1463292664415432.8 over -0.46591834334263044 gives 3140663348683280, while `⌊a / b⌋` is
3140663348683281.
-/
def divFloorSchedule {α : Type} [Zero α] [One α] [Add α] [Sub α] [Div α] [LT α]
    [DecidableRel (α := α) (· < ·)] [BEq α]
    (fmod : α → α → α) (floor : α → α) (signedZero : α → α) (half : α) (a b : α) : α :=
  -- `floor` gives the quotient a rounding's zero tangent under dual execution.
  if b == 0 then floor (a / b)
  else
    let remainder := fmod a b
    let quotient := (a - remainder) / b
    let quotient :=
      if !(remainder == 0) && (decide (b < 0) != decide (remainder < 0)) then quotient - 1
      else quotient
    if !(remainder == remainder) && a / b == a / b then floor (a / b)
    else if !(quotient == 0) then
      let rounded := floor quotient
      if half < quotient - rounded then rounded + 1 else rounded
    else
      signedZero (a / b)

end TorchLean
