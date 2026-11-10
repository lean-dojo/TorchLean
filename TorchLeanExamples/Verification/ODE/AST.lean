/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context
public import TorchLean.Tensor.FromFloat
public import TorchLean.Tensor.OfFn -- shake: keep
public import TorchLean.Tensor.LinearAlgebra.Identity -- shake: keep
public import TorchLean.Verify.CROWN.FlatBox.Transfer

/-!
# Ast

ODE right-hand-side expressions and interval endpoint calculations, a small companion to the PINN
PDE DSL specialized to ODE initial value problems:

$$
u'(t)=f(t,u(t)).
$$

The evaluator works on `α × α` endpoint pairs. Constants are converted outward from their binary64
bounds, sums and differences use the outward `BoundOps` operations, products use
`Graph.intervalMul`, and division, `sin`, `cos`, `exp` and `log` (the functions of the benchmarks
in arXiv:2601.19818) use `NonlinearBoundOps`. The result is `none` when an operation has no
enclosure in the chosen arithmetic, such as `exp` and `log` under host `Float`, or when an endpoint
is infinite or NaN.

No theorem in this module states that the computed pairs enclose the exact-real expression. That
argument goes through the enclosure lemma of each operation, such as `Graph.intervalMul_encloses`.
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Examples.ODE

open CROWN

/--
Syntax of ODE right-hand sides $f(t,u)$: real constants between two `Float` bounds, the time
variable `t`, the state variable `u`, addition, subtraction, multiplication, division, negation and
the scalar functions `sin`, `cos`, `exp` and `log`. We keep the language this small so the checker
is easy to inspect, and it covers the benchmark equations of the ODE certificates.
-/
inductive Expr where
  /-- A real constant known to lie between the binary64 values `lo` and `hi`. -/
  | const (lo hi : Float)
  /-- The time variable `t`. -/
  | t
  /-- The state variable `u`. -/
  | u
  /-- The sum `a + b`. -/
  | add (a b : Expr)
  /-- The difference `a - b`. -/
  | sub (a b : Expr)
  /-- The product `a * b`. -/
  | mul (a b : Expr)
  /-- The quotient `a / b`. -/
  | div (a b : Expr)
  /-- The negation `-a`. -/
  | neg (a : Expr)
  /-- The sine of `a`. -/
  | sin (a : Expr)
  /-- The cosine of `a`. -/
  | cos (a : Expr)
  /-- The exponential of `a`. -/
  | exp (a : Expr)
  /-- The natural logarithm of `a`. -/
  | log (a : Expr)
  deriving Repr

/--
An evaluation environment for `Expr`. Both `t` and `u` are intervals `(lo,hi)`, so every
expression evaluates to an interval.
-/
structure Env (α : Type) where
  /-- Interval for the independent time variable `t`. -/
  t : α × α
  /-- Interval for the current state value `u(t)`. -/
  u : α × α

/-!
Interval primitives used by `eval`. They are written against the scalar interfaces `Context α` and
`BoundOps α`, so the same expression evaluates under `Float`, `Binary32` or another executable
scalar.
-/
namespace Ival

/--
Whether a checker scalar is finite under the supported `Float` and `Binary32`
backends, tested as `x - x == 0`.

Every finite value subtracts from itself to zero, while both backends produce NaN for
$\infty-\infty$ and for $\mathrm{NaN}-\mathrm{NaN}$. Phrasing the guard through `Context` keeps
the interval evaluator generic, with no backend-specific bit decoder.
-/
@[noinline] def isFiniteBool {α : Type} [Storage α] [Context α] (x : α) : Bool :=
  x - x == (0 : α)

/--
Boolean $x\leq y$ test that rejects NaN and infinite endpoints.

Writing it as $\neg(x>y)$ would make an unordered value less than or equal to every value, and some
host `Float` comparisons use a total implementation order, so the finite guards belong to the
certificate check itself.
-/
@[noinline] def leBool {α : Type} [Storage α] [Context α] (x y : α) : Bool :=
  isFiniteBool x && isFiniteBool y && (Context.gtBool y x || x == y)

/-- Add two closed intervals, the lower endpoint by `BoundOps.addDown` and the upper by
`BoundOps.addUp`. -/
@[inline] def add {α : Type} [Storage α] [Context α] [BoundOps α] (x y : α × α) :
    α × α :=
  let (xl, xh) := x; let (yl, yh) := y; (BoundOps.addDown xl yl, BoundOps.addUp xh yh)

/-- Subtract closed intervals as $[x_l-y_h, x_h-y_l]$, the lower endpoint by `BoundOps.subDown`
and the upper by `BoundOps.subUp`. -/
@[inline] def sub {α : Type} [Storage α] [Context α] [BoundOps α] (x y : α × α) :
    α × α :=
  let (xl, xh) := x; let (yl, yh) := y; (BoundOps.subDown xl yh, BoundOps.subUp xh yl)

/-- Negate a closed interval by swapping and negating endpoints. -/
@[inline] def neg {α : Type} [Storage α] [Context α] (x : α × α) : α × α :=
  let (xl, xh) := x; (-xh, -xl)


end Ival

/--
Interval evaluation for `Expr`, `none` when a division, `sin`, `cos`, `exp` or `log` enclosure
fails, a node's pair fails `Graph.finiteBounds?`, or the expression is deeper than `fuel`.

Structural recursion on `Expr` would already be total. The fuel caps the accepted expression depth
(512 via `eval`), and deeper expressions return `none` unevaluated.
-/
def Expr.evalWithFuel {α : Type} [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α]
    [FromFloat α] (fuel : Nat) (env : Env α) (e : Expr) : Option (α × α) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    match e with
    | .const lo hi =>
      Graph.finiteBounds? (ofFloat lo (mode := .down),
        ofFloat hi (mode := .up))
    | .t => Graph.finiteBounds? env.t
    | .u => Graph.finiteBounds? env.u
    | .add a b => do
      let x ← evalWithFuel fuel env a
      let y ← evalWithFuel fuel env b
      Graph.finiteBounds? (Ival.add x y)
    | .sub a b => do
      let x ← evalWithFuel fuel env a
      let y ← evalWithFuel fuel env b
      Graph.finiteBounds? (Ival.sub x y)
    | .mul a b => do
      let x ← evalWithFuel fuel env a
      let y ← evalWithFuel fuel env b
      Graph.finiteBounds? (Graph.intervalMul x.1 x.2 y.1 y.2)
    | .div a b => do
      let x ← evalWithFuel fuel env a
      let y ← evalWithFuel fuel env b
      Graph.finiteBounds? (← NonlinearBoundOps.divBounds x.1 x.2 y.1 y.2)
    | .neg a => do
      let x ← evalWithFuel fuel env a
      Graph.finiteBounds? (Ival.neg x)
    | .sin a => do
      let x ← evalWithFuel fuel env a
      Graph.finiteBounds? (← NonlinearBoundOps.sinBounds x.1 x.2)
    | .cos a => do
      let x ← evalWithFuel fuel env a
      Graph.finiteBounds? (← NonlinearBoundOps.cosBounds x.1 x.2)
    | .exp a => do
      let x ← evalWithFuel fuel env a
      Graph.finiteBounds? (← NonlinearBoundOps.expBounds x.1 x.2)
    | .log a => do
      let x ← evalWithFuel fuel env a
      Graph.finiteBounds? (← NonlinearBoundOps.logBounds x.1 x.2)

/-- Evaluate an ODE expression with the default recursion fuel used by certificate checking. -/
def Expr.eval {α : Type} [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α]
    [FromFloat α] (env : Env α) (e : Expr) : Option (α × α) :=
  evalWithFuel 512 env e

end TorchLean.Examples.ODE
