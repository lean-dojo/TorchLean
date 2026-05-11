/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor

/-!
# Ast

ODE RHS expression language + interval evaluator.

This is a small companion to the existing PINN PDE DSL, specialized for ODE IVPs:
  u'(t) = f(t, u(t)).

We use conservative interval arithmetic over `α × α` intervals (for any scalar `α` with a
`Context` instance), with a few common transcendentals (`sin`, `cos`, `exp`, `log`) needed by the
benchmarks in arXiv:2601.19818.

For the trigonometric cases we use a 1-Lipschitz enclosure around the midpoint and then clamp to
`[-1,1]`. Over real-valued semantics this is the intended enclosure argument; executable scalar
backends rely on their `Context` operations matching the assumed real behavior closely enough for
the checker mode being used.
-/

@[expose] public section


namespace NN.Verification.ODE

open Spec

/-!
`Expr` is an AST for ODE right-hand sides `f(t,u)`.

We cover the arithmetic and elementary functions needed by the ODE certificate format used in
TorchLean: constants, the independent variable `t`, the state variable `u`, field arithmetic, and
the common scalar functions `sin`, `cos`, `exp`, and `log`.  Keeping this language explicit makes
the checker easier to inspect while still covering the benchmark equations we want to verify.
-/
inductive Expr where
  | const (c : Float)
  | t
  | u
  | add (a b : Expr)
  | sub (a b : Expr)
  | mul (a b : Expr)
  | div (a b : Expr)
  | neg (a : Expr)
  | sin (a : Expr)
  | cos (a : Expr)
  | exp (a : Expr)
  | log (a : Expr)
  deriving Repr

/-!
An evaluation environment for `Expr`.

We interpret `t` and `u` as intervals `(lo,hi)`; everything evaluates to an interval.
-/
structure Env (α : Type) where
  /-- Interval for the independent time variable `t`. -/
  t : α × α
  /-- Interval for the current state value `u(t)`. -/
  u : α × α

/-!
Interval primitives used by `eval`.

These operations are written against the abstract scalar interface `Context α` so we can evaluate
the same expression under `Float` or `IEEE32Exec` (or other executable scalars).
-/
namespace Ival

/- The interval layer below is the checker kernel for ODE certificates.  The definitions favor
direct, inspectable enclosures over clever rewrites so that each arithmetic case can be reviewed
against the mathematical interval rule it implements. -/

/-- Boolean `x ≤ y` test, implemented via the primitive `Context.gtBool`. -/
@[inline] def leBool {α : Type} [Context α] (x y : α) : Bool :=
  not (Context.gtBool x y)

/-- Binary minimum. -/
@[inline] def min2 {α : Type} [Context α] (a b : α) : α :=
  if leBool a b then a else b

/-- Binary maximum. -/
@[inline] def max2 {α : Type} [Context α] (a b : α) : α :=
  if leBool a b then b else a

/-- Interval addition. -/
@[inline] def add {α : Type} [Context α] (x y : α × α) : α × α :=
  let (xl, xh) := x; let (yl, yh) := y; (xl + yl, xh + yh)

/-- Interval subtraction. -/
@[inline] def sub {α : Type} [Context α] (x y : α × α) : α × α :=
  let (xl, xh) := x; let (yl, yh) := y; (xl - yh, xh - yl)

/-- Interval negation. -/
@[inline] def neg {α : Type} [Context α] (x : α × α) : α × α :=
  let (xl, xh) := x; (-xh, -xl)

/-- Interval multiplication using the standard four-product enclosure. -/
@[inline] def mul {α : Type} [Context α] (x y : α × α) : α × α :=
  -- Standard interval product: compute four products and take min/max.
  let (xl, xh) := x; let (yl, yh) := y
  let p1 := xl * yl
  let p2 := xl * yh
  let p3 := xh * yl
  let p4 := xh * yh
  let lo := min2 (min2 p1 p2) (min2 p3 p4)
  let hi := max2 (max2 p1 p2) (max2 p3 p4)
  (lo, hi)

/--
Interval reciprocal.

Returns `none` if the interval contains `0`, because `1/x` is not interval-safe across a pole.
-/
@[inline] def inv {α : Type} [Context α] (y : α × α) : Option (α × α) :=
  let (yl, yh) := y
  let z := Numbers.zero
  -- If 0 ∈ [yl,yh], reciprocal is not interval-safe.
  if leBool yl z && leBool z yh then
    none
  else if Context.gtBool yl z then
    -- 1/x is decreasing on (0,∞).
    some (Numbers.one / yh, Numbers.one / yl)
  else
    -- yl < 0 and yh < 0, decreasing on (-∞,0).
    some (Numbers.one / yh, Numbers.one / yl)

/-- Interval division, returning `none` if the denominator interval contains `0`. -/
@[inline] def div {α : Type} [Context α] (x y : α × α) : Option (α × α) := do
  let iy ← inv y
  pure (mul x iy)

/-- Interval exponential (monotone, so endpoints map to endpoints). -/
@[inline] def exp {α : Type} [Context α] (x : α × α) : α × α :=
  let (xl, xh) := x
  (MathFunctions.exp xl, MathFunctions.exp xh)

/--
Interval logarithm.

Returns `none` unless the interval is strictly positive.
-/
@[inline] def log {α : Type} [Context α] (x : α × α) : Option (α × α) :=
  let (xl, xh) := x
  if Context.gtBool xl Numbers.zero then
    some (MathFunctions.log xl, MathFunctions.log xh)
  else
    none

/--
Interval sine enclosure.

We use a 1‑Lipschitz enclosure around the midpoint and clamp to `[-1, 1]`.
-/
@[inline] def sin {α : Type} [Context α] (x : α × α) : α × α :=
  -- 1‑Lipschitz enclosure around midpoint, clamped to [-1,1].
  let (l, u) := x
  let m := (l + u) * Numbers.pointfive
  let r := (u - l) * Numbers.pointfive
  let base := MathFunctions.sin m
  let lo0 := base - r
  let hi0 := base + r
  let lo := max2 lo0 Numbers.neg_one
  let hi := min2 hi0 Numbers.one
  (lo, hi)

/--
Interval cosine enclosure.

Same strategy as `sin`: 1‑Lipschitz around the midpoint, clamped to `[-1, 1]`.
-/
@[inline] def cos {α : Type} [Context α] (x : α × α) : α × α :=
  -- Same 1‑Lipschitz enclosure as `sin`, clamped to [-1,1].
  let (l, u) := x
  let m := (l + u) * Numbers.pointfive
  let r := (u - l) * Numbers.pointfive
  let base := MathFunctions.cos m
  let lo0 := base - r
  let hi0 := base + r
  let lo := max2 lo0 Numbers.neg_one
  let hi := min2 hi0 Numbers.one
  (lo, hi)

end Ival

/-!
Interval evaluation for `Expr`.

`evalWithFuel` uses a fuel parameter so the evaluator is total even for malformed/self-referential
expressions (though `Expr` itself has no recursion).
-/
def evalWithFuel {α : Type} [Context α] (ofFloat : Float → α) (fuel : Nat) (env : Env α) (e : Expr)
  :
    Option (α × α) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match e with
    | .const c =>
      let v := ofFloat c
      some (v, v)
    | .t => some env.t
    | .u => some env.u
    | .add a b =>
      match evalWithFuel ofFloat fuel env a, evalWithFuel ofFloat fuel env b with
      | some x, some y => some (Ival.add x y)
      | _, _ => none
    | .sub a b =>
      match evalWithFuel ofFloat fuel env a, evalWithFuel ofFloat fuel env b with
      | some x, some y => some (Ival.sub x y)
      | _, _ => none
    | .mul a b =>
      match evalWithFuel ofFloat fuel env a, evalWithFuel ofFloat fuel env b with
      | some x, some y => some (Ival.mul x y)
      | _, _ => none
    | .div a b =>
      match evalWithFuel ofFloat fuel env a, evalWithFuel ofFloat fuel env b with
      | some x, some y => Ival.div x y
      | _, _ => none
    | .neg a =>
      match evalWithFuel ofFloat fuel env a with
      | some x => some (Ival.neg x)
      | none => none
    | .sin a =>
      match evalWithFuel ofFloat fuel env a with
      | some x => some (Ival.sin x)
      | none => none
    | .cos a =>
      match evalWithFuel ofFloat fuel env a with
      | some x => some (Ival.cos x)
      | none => none
    | .exp a =>
      match evalWithFuel ofFloat fuel env a with
      | some x => some (Ival.exp x)
      | none => none
    | .log a =>
      match evalWithFuel ofFloat fuel env a with
      | some x => Ival.log x
      | none => none

/-!
Convenience wrapper with a generous fixed fuel.
-/
def eval {α : Type} [Context α] (ofFloat : Float → α) (env : Env α) (e : Expr) : Option (α × α) :=
  evalWithFuel ofFloat 512 env e

end NN.Verification.ODE
