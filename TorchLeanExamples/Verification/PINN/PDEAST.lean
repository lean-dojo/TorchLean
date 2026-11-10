/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.ODE.AST

/-!
# PdeAst

A compact PDE mini-language and its interval evaluator. The expressions describe PDE residuals in
terms of `u`, its first and second partial derivatives along the axes, and arithmetic combinators.
The evaluator takes primitive bounds (`u`, `u_x`, `u_y`, `u_xx`, ...) and computes residual
endpoints with the outward-widened host `Float` operations of `BoundOps`: `ODE.Ival.add` and
`ODE.Ival.sub` for sums and differences, and `Graph.intervalMul` for products and scaling. It does
not check that the primitive bounds are finite.

The axes are `x` and `y` for 2D workflows, and 1D workflows ignore `y`.

References:
- PINNs: `https://arxiv.org/abs/1711.10561`
- Interval/CROWN-style primitive bounds used by the surrounding workflows:
  `https://arxiv.org/abs/1811.00866`
-/

@[expose] public section


namespace TorchLean.Examples.PINN

open CROWN.Graph

open TorchLean.Tensor

/-- Coordinate axis used by the PDE residual language. -/
inductive Axis where
/-- The axis `x`. -/
| x
/-- The axis `y`. -/
| y
  deriving DecidableEq, Repr

/--
Expression language for scalar PDE residuals.

The constructors refer to `u`, first partials, second partials, constants, and arithmetic
combinators. Evaluation returns intervals, matching the bounds that the surrounding PINN checker
works with.
-/
inductive Expr where
/-- A real constant known to lie between the binary64 values `lo` and `hi`. -/
| const  (lo hi : Float)
/-- The solution value `u(x, y)`. -/
| u
/-- The first partial derivative of `u` along the axis `a`. -/
| du    (a : Axis)
/-- The second partial derivative of `u` along the axis `a`. -/
| d2u   (a : Axis)
/-- The sum `e1 + e2`. -/
| add   (e1 e2 : Expr)
/-- The difference `e1 - e2`. -/
| sub   (e1 e2 : Expr)
/-- The product `e1 * e2`. -/
| mul   (e1 e2 : Expr)
/-- The product `c * e` of the constant `c` and `e`. -/
| scale (c : Float) (e : Expr)
/-- The negation `-e`. -/
| neg   (e : Expr)
  deriving Repr

/-- Primitive interval bounds supplied to the PDE expression evaluator. -/
structure Prims where
  /-- Enclosure for the solution value `u` itself. -/
  u    : Option (Float × Float)
  /-- Enclosure for `∂u/∂x`. -/
  duX  : Option (Float × Float)
  /-- Enclosure for `∂u/∂y`. -/
  duY  : Option (Float × Float)
  /-- Enclosure for `∂²u/∂x²`. -/
  d2uX : Option (Float × Float)
  /-- Enclosure for `∂²u/∂y²`. Every field is optional because a residual needs only the
  derivatives its PDE mentions, and evaluating a missing one returns `none` with no default
  substituted. -/
  d2uY : Option (Float × Float)


/-- Evaluate a PDE expression to an interval from primitive bounds, returning `none` when a needed
primitive is missing or the expression is deeper than `fuel`. -/
def Expr.evalWithFuel (fuel : Nat) (p : Prims) (e : Expr) : Option (Float × Float) :=
  match fuel with
  | 0 => none
  | fuel+1 =>
    match e with
    | .const lo hi => some (lo, hi)
    | .u => p.u
    | .du .x => p.duX
    | .du .y => p.duY
    | .d2u .x => p.d2uX
    | .d2u .y => p.d2uY
    | .add e1 e2 =>
      match evalWithFuel fuel p e1, evalWithFuel fuel p e2 with
      | some a, some b => some (ODE.Ival.add a b)
      | _, _ => none
    | .sub e1 e2 =>
      match evalWithFuel fuel p e1, evalWithFuel fuel p e2 with
      | some a, some b => some (ODE.Ival.sub a b)
      | _, _ => none
    | .mul e1 e2 =>
      match evalWithFuel fuel p e1, evalWithFuel fuel p e2 with
      | some a, some b => some (intervalMul a.1 a.2 b.1 b.2)
      | _, _ => none
    | .scale c e =>
      match evalWithFuel fuel p e with
      | some a => some (intervalMul c c a.1 a.2)
      | none => none
    | .neg e =>
      match evalWithFuel fuel p e with
      | some a => some (ODE.Ival.neg a)
      | none => none

/-- Evaluate a PDE expression with the default recursion budget of 256. -/
def Expr.eval (p : Prims) (e : Expr) : Option (Float × Float) :=
  evalWithFuel 256 p e

end TorchLean.Examples.PINN
