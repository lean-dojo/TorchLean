/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train
public import NN.Tensor
public import NN.Spec.Core.Context.Rational

/-!
# ElementwiseDivTest

Regression test for the CPU elementwise division node (`Tape.div` / `TapeM.div`) over `ℚ`.

Exact rational arithmetic (no floating-point roundoff) lets us assert both the forward value
$a/b$ *and* the quotient-rule backward by exact equality:

$$
\frac{\partial(a/b)}{\partial a}=\frac1b,\qquad
\frac{\partial(a/b)}{\partial b}=-\frac{a}{b^2}.
$$

With $a=[6,8]$, $b=[2,4]$, and upstream gradient
$\mathrm{dLdy}=[1,1]$:

$$
y=[3,2],\qquad
\frac{\partial L}{\partial a}=[1/2,1/4],\qquad
\frac{\partial L}{\partial b}=[-3/2,-1/2].
$$
-/

open scoped Spec.RationalAlgebraic

@[expose] public section

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tests
namespace Rationals
namespace ElementwiseDiv

open Runtime.Autograd

/-- Tag used for readable error messages. -/
abbrev tag : String := "elementwise_div_test (Rat)"

/-- Two-element vector shape. -/
abbrev s2 : Shape := [2]

/-- Numerator `a`. -/
def a : Tensor ℚ s2 := [6.0, 8.0]
/-- Denominator `b`. -/
def b : Tensor ℚ s2 := [2.0, 4.0]
/-- Upstream gradient $\partial L/\partial y$. -/
def dLdy : Tensor ℚ s2 := [1.0, 1.0]

/-- Expected forward value $a/b=[3,2]$. -/
def yExp : Tensor ℚ s2 := [3.0, 2.0]
/-- Expected gradient $\partial L/\partial a=\mathrm{dLdy}/b=[1/2,1/4]$. -/
def daExp : Tensor ℚ s2 := [0.5, 0.25]
/-- Expected gradient
$\partial L/\partial b=-\mathrm{dLdy}\,a/b^2=[-3/2,-1/2]$
(built by negating $[3/2,1/2]$). -/
def dbExp : Tensor ℚ s2 := Tensor.negSpec [1.5, 0.5]

/-- Build a tape for $y=a/b$, run the backward pass, and check the forward value and both
input gradients against the exact rational references. -/
def checkDiv : Runtime.Autograd.Result Bool := do
  let t0 : Tape ℚ := Tape.empty
  let m : TapeM ℚ _ := do
    let aId ← TapeM.leaf a (name := some "a") (requiresGrad := true)
    let bId ← TapeM.leaf b (name := some "b") (requiresGrad := true)
    let yId ← TapeM.div (s := s2) aId bId
    let t ← TapeM.getTape
    let yVal ← liftM (Tape.requireValue (α := ℚ) (t := t) (s := s2) yId)
    let grads ← liftM (Tape.backward (t := t) yId (Spec.SomeTensor.ofTensor dLdy))
    pure (aId, bId, yVal, grads)
  let ((aId, bId, yVal, grads), _) ← TapeM.run t0 m
  let da ← Train.requireGradTensor (tag := tag) (s := s2) grads aId
  let db ← Train.requireGradTensor (tag := tag) (s := s2) grads bId
  let okY  := decide (pretty yVal = pretty yExp)
  let okDa := decide (pretty da = pretty daExp)
  let okDb := decide (pretty db = pretty dbExp)
  pure (okY && okDa && okDb)

/-- Entrypoint (called by `NN/Tests/Runtime/Rationals/Suite.lean`). -/
def run : IO Unit := do
  match checkDiv with
  | .ok true => IO.println "elementwise_div_test (Rat): OK"
  | .ok false => throw <| IO.userError "elementwise_div_test (Rat): FAILED"
  | .error msg => throw <| IO.userError s!"elementwise_div_test (Rat): {msg}"

end ElementwiseDiv
end Rationals
end Tests
