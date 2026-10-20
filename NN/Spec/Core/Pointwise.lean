/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Pointwise operators

PyTorch's core ATen vocabulary contains many pointwise operators, such as `neg`, `div`, `pow`,
`rsqrt`, `erf`, `gelu` and `clamp_min`. They share one shape rule: every operand and the result
have the same shape. Graphs therefore represent them as one operation family whose operator is
data, and `PointwiseOp` lists the operators of that family together with the static attributes
PyTorch passes them, such as the rounding mode of `torch.div`.

What an operator computes depends on the scalar type. A scalar dictionary lists the operators it
implements in `Context.pointwise`, as a `PointwiseTable`. Each entry is a total function that
returns the result or a reason why the scalar type has none; over the reals, for example, a zero
divisor has no quotient. An operator missing from a table is unsupported for that scalar type, and
evaluators report it instead of approximating it.

## References

- PyTorch Core ATen IR: `https://docs.pytorch.org/docs/stable/torch.compiler_ir.html`
- `torch.div`: `https://docs.pytorch.org/docs/stable/generated/torch.div.html`
- `torch.nn.functional.gelu`:
  `https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html`
-/

@[expose] public section

namespace Spec

/-- Rounding applied to a quotient, as the `rounding_mode` argument of `torch.div`. -/
inductive DivRounding where
  /-- True division, PyTorch's `rounding_mode=None`. -/
  | none
  /-- Round the quotient toward zero, `rounding_mode="trunc"`. -/
  | trunc
  /-- Round the quotient toward negative infinity, `rounding_mode="floor"`. -/
  | floor
  deriving Repr, DecidableEq, Inhabited

/-- The formula selected by the `approximate` argument of `torch.nn.functional.gelu`. -/
inductive GeluApproximation where
  /-- `x Φ(x)`, with the standard normal distribution function `Φ` written through `erf`. -/
  | none
  /-- The tanh approximation `0.5 x (1 + tanh (√(2 / π) (x + 0.044715 x³)))`. -/
  | tanh
  deriving Repr, DecidableEq, Inhabited

/-- Pointwise operators represented by the graph family `NN.IR.OpKind.pointwise`. -/
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
  deriving Repr, DecidableEq, Inhabited

namespace PointwiseOp

/-- Number of tensor operands. -/
def arity : PointwiseOp → Nat
  | .neg | .rsqrt | .erf | .gelu _ => 1
  | .div _ | .pow | .clampMin => 2

/-- Every operator takes at least one operand. -/
theorem arity_pos (op : PointwiseOp) : 0 < op.arity := by
  cases op <;> simp [arity]

/-- Short operator name without attributes, for grouping diagnostics. -/
def name : PointwiseOp → String
  | .neg => "neg"
  | .rsqrt => "rsqrt"
  | .erf => "erf"
  | .gelu _ => "gelu"
  | .div _ => "div"
  | .pow => "pow"
  | .clampMin => "clamp_min"

/-- Operator name with its attribute, for diagnostics. Artifact codecs define their own spelling. -/
def describe : PointwiseOp → String
  | .neg => "neg"
  | .rsqrt => "rsqrt"
  | .erf => "erf"
  | .gelu .none => "gelu(approximate=none)"
  | .gelu .tanh => "gelu(approximate=tanh)"
  | .div .none => "div"
  | .div .trunc => "div(rounding_mode=trunc)"
  | .div .floor => "div(rounding_mode=floor)"
  | .pow => "pow"
  | .clampMin => "clamp_min"

/-- Every operator with every attribute value, in declaration order. -/
def all : List PointwiseOp :=
  [.neg, .rsqrt, .erf, .gelu .none, .gelu .tanh, .div .none, .div .trunc, .div .floor, .pow,
    .clampMin]

/-- The enumeration is complete. -/
theorem mem_all (op : PointwiseOp) : op ∈ all := by
  rcases op with _ | _ | _ | approximate | rounding | _ | _
  · simp [all]
  · simp [all]
  · simp [all]
  · cases approximate <;> simp [all]
  · cases rounding <;> simp [all]
  · simp [all]
  · simp [all]

end PointwiseOp

/-- The meaning of an operator with `arity` operands at the scalar type `α`.

The function receives the operands in order and returns the result, or the reason why the scalar
type has no result for them. -/
abbrev PointwiseFn (α : Type) (arity : Nat) := (Fin arity → α) → Except String α

namespace PointwiseFn

/-- A meaning with one operand. -/
def unary {α : Type} (f : α → Except String α) : PointwiseFn α 1 :=
  fun operands => f (operands 0)

/-- A meaning with two operands, in order. -/
def binary {α : Type} (f : α → α → Except String α) : PointwiseFn α 2 :=
  fun operands => f (operands 0) (operands 1)

end PointwiseFn

/-- The pointwise operators a scalar type implements, each with its meaning. -/
abbrev PointwiseTable (α : Type) := (op : PointwiseOp) → Option (PointwiseFn α op.arity)

/--
PyTorch's flooring division, `c10::div_floor_floating`, over the operations it uses.

`fmod` is C's truncating remainder, `floor` rounds toward negative infinity, and `signedZero q`
is zero with the sign of `q`. A zero divisor returns the quotient itself. Otherwise the remainder
corrects the quotient, as Python's `//` does, and the final comparison with `half` repairs a
quotient that rounded across an integer. A binary format runs this schedule with its rounded
operations; over the reals it computes `⌊a / b⌋`.
-/
def divFloorSchedule {α : Type} [Zero α] [One α] [Add α] [Sub α] [Div α] [LT α]
    [DecidableRel (α := α) (· < ·)] [BEq α]
    (fmod : α → α → α) (floor : α → α) (signedZero : α → α) (half : α) (a b : α) : α :=
  if b == 0 then a / b
  else
    let remainder := fmod a b
    let quotient := (a - remainder) / b
    let quotient :=
      if !(remainder == 0) && (decide (b < 0) != decide (remainder < 0)) then quotient - 1
      else quotient
    if !(quotient == 0) then
      let rounded := floor quotient
      if half < quotient - rounded then rounded + 1 else rounded
    else
      signedZero (a / b)

end Spec
