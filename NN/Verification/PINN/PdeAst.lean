/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.ResidualAffine

/-!
# PdeAst

A compact PDE mini-language (AST) and interval evaluator.

This module defines a small expression language to describe PDE residuals in
terms of u, its first/second partial derivatives along axes, and arithmetic
combinators. An interval evaluator consumes primitive bounds (u, u_x, u_y, u_xx,
...) and assembles a conservative residual bound for the whole expression.

We focus on 2D workflows: axes are X and Y, but the evaluator works for 1D by
ignoring Y.

References:
- PINNs: `https://arxiv.org/abs/1711.10561`
- Interval/CROWN-style primitive bounds used by the surrounding workflows:
  `https://arxiv.org/abs/1811.00866`
-/

@[expose] public section


namespace NN.Verification.PINN.PdeAst

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.PINN.ResidualAffine
open Spec TorchLean
open TorchLean.Tensor

/-- Coordinate axis used by the PDE residual language. -/
inductive Axis where
| X | Y
  deriving DecidableEq, Repr

/--
Expression language for scalar PDE residuals.

The constructors refer to `u`, first partials, second partials, constants, and arithmetic
combinators. Evaluation is interval-valued because the surrounding PINN checker works with bounds
rather than single floating-point samples.
-/
inductive Expr where
| const  (c : Float)
| u                              -- u(x,y)
| du    (a : Axis)               -- first partial derivative along axis
| d2u   (a : Axis)               -- second partial derivative along axis
| add   (e1 e2 : Expr)
| sub   (e1 e2 : Expr)
| mul   (e1 e2 : Expr)
| scale (c : Float) (e : Expr)
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
  /-- Enclosure for `∂²u/∂y²`. Every field is optional because a residual only needs the
  derivatives its own PDE mentions; a missing one makes evaluation fail loudly instead of
  silently substituting a default. -/
  d2uY : Option (Float × Float)

/-- Add two closed Float intervals endpointwise. -/
@[inline] def ivalAdd (a b : Float × Float) : Float × Float :=
  let (al, ah) := a; let (bl, bh) := b; (al + bl, ah + bh)

/-- Subtract two closed Float intervals using outward endpoint selection. -/
@[inline] def ivalSub (a b : Float × Float) : Float × Float :=
  let (al, ah) := a; let (bl, bh) := b; (al - bh, ah - bl)

/-- Scale a closed Float interval, reversing endpoint order for negative coefficients. -/
@[inline] def ivalScale (c : Float) (a : Float × Float) : Float × Float :=
  let (al, ah) := a
  if c ≥ 0.0 then (c * al, c * ah) else (c * ah, c * al)

/-- Negate a closed Float interval by swapping and negating its endpoints. -/
@[inline] def ivalNeg (a : Float × Float) : Float × Float :=
  let (al, ah) := a; (-ah, -al)

/-- Bound the product of two Float intervals through the shared McCormick envelopes. -/
@[inline] def ivalMul (a b : Float × Float) : Float × Float :=
  -- Use McCormick envelopes to bound u*v conservatively
  let (al, ah) := a; let (bl, bh) := b
  let (axU, ayU, cU) := mccormickUpper al ah bl bh
  let (_, uHi) := evalAffine2DOnBox axU ayU cU al ah bl bh
  let (axL, ayL, cL) := mccormickLower al ah bl bh
  let (lLo, _) := evalAffine2DOnBox axL ayL cL al ah bl bh
  (lLo, uHi)

/-- Evaluate a PDE expression to an interval, given primitive bounds. -/
def evalWithFuel (fuel : Nat) (p : Prims) (e : Expr) : Option (Float × Float) :=
  match fuel with
  | 0 => none
  | fuel+1 =>
    match e with
    | .const c => some (c, c)
    | .u => p.u
    | .du .X => p.duX
    | .du .Y => p.duY
    | .d2u .X => p.d2uX
    | .d2u .Y => p.d2uY
    | .add e1 e2 =>
      match evalWithFuel fuel p e1, evalWithFuel fuel p e2 with
      | some a, some b => some (ivalAdd a b)
      | _, _ => none
    | .sub e1 e2 =>
      match evalWithFuel fuel p e1, evalWithFuel fuel p e2 with
      | some a, some b => some (ivalSub a b)
      | _, _ => none
    | .mul e1 e2 =>
      match evalWithFuel fuel p e1, evalWithFuel fuel p e2 with
      | some a, some b => some (ivalMul a b)
      | _, _ => none
    | .scale c e =>
      match evalWithFuel fuel p e with
      | some a => some (ivalScale c a)
      | none => none
    | .neg e =>
      match evalWithFuel fuel p e with
      | some a => some (ivalNeg a)
      | none => none

/-- Evaluate a PDE expression with the default recursion budget. -/
def eval (p : Prims) (e : Expr) : Option (Float × Float) :=
  evalWithFuel 256 p e

end NN.Verification.PINN.PdeAst
