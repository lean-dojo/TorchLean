/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context

/-!
# The opt-in rational algebraic backend

Algebraic tests and rational verification code select these instances with
`open scoped TorchLean.RationalAlgebraic`. Importing the general `Context` does not select them.
-/

@[expose] public section

/-!
## What the rational context computes

`Context` includes transcendental functions and real-valued exponentiation (`Pow α α`), which
softmax, tanh and the other analytic layers need over `Float` or `ℝ`. Most of
them do not map rationals to rationals, so without the scope `ℚ` has no `Context` instance and a
softmax over `ℚ` is a typeclass error.

The instances compute the field operations, the comparisons, `max`, `min`, `abs` and `x ^ y` at an
integer `y` (`y.den = 1`) exactly. They do not compute the rest: `exp`, `log`, `sqrt`, `tanh`,
`cosh`, `sin`, `cos`, `sinh` and the constant `pi` are `0`, and so is `x ^ y` at a non-integer `y`.
The `Context` fields whose defaults call them (`sqrtNat`, `sigmoid`, `log1p`, `expm1`, `expMul`,
`tan`, `exp2`, `log2`, `log10`, `asinh`, `acosh`, `atanh` and `hypot`) are not the mathematical
functions over `ℚ` either.
-/

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.RationalAlgebraic

/-- `x ^ y` over `ℚ`: the integer power `x ^ y.num` when `y` is an integer (`y.den = 1`), and `0`
at every other `y`. -/
scoped instance instPowRatRat : Pow ℚ ℚ where
  pow x y :=
    if y.den = 1 then
      x ^ y.num
    else
      0

/-- The `MathFunctions` of `ℚ`: `abs` is the absolute value, and `exp`, `log`, `sqrt`, `tanh`,
`cosh`, `sin`, `cos`, `sinh` and `pi` are `0`. -/
scoped instance instMathFunctionsRat : MathFunctions ℚ where
  exp := fun _ => 0
  tanh := fun _ => 0
  cosh := fun _ => 0
  sqrt := fun _ => 0
  abs := fun x => |x|
  log := fun _ => 0
  pi := 0
  cos := fun _ => 0
  sin := fun _ => 0
  sinh := fun _ => 0

/-- Full opt-in `Context` dictionary for exact rational algebraic fragments. -/
scoped instance instContextRat : Context ℚ where
  defaultEpsilon := 1 / 1000000
  decidableGT := inferInstance

/-- The rational dictionary uses Mathlib's field operations and has strictly positive tolerance. -/
scoped instance instLawfulContextRat : LawfulContext ℚ where
  add_eq _ _ := rfl
  mul_eq _ _ := rfl
  sub_eq _ _ := rfl
  div_eq _ _ := rfl
  neg_eq _ := rfl
  zero_eq := rfl
  one_eq := rfl
  lt_iff _ _ := Iff.rfl
  le_iff _ _ := Iff.rfl
  max_eq _ _ := rfl
  min_eq _ _ := rfl
  beq_iff _ _ := beq_iff_eq
  natCast_eq _ := rfl
  divNat_eq _ _ := rfl
  ratCast_eq _ := rfl
  abs_eq _ := rfl
  defaultEpsilon_pos := by
    show (0 : ℚ) < 1 / 1000000
    norm_num

end TorchLean.RationalAlgebraic
