/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
public import Mathlib.Data.Nat.Factorial.DoubleFactorial
public import TorchLean.NN.Program

/-!
# TwoStage Core

Shared core of the TwoStage neural-controller and neural-Lyapunov workflow in this directory
(`TorchLeanExamples/Verification/Lyapunov/`). The workflow is the all-in-Lean pipeline (iii) of
Figure 7 in the TorchLean paper (`arXiv:2602.22631`): both training stages and the final IBP/CROWN
bounds on the loss run inside TorchLean under `Binary32` semantics.

This file contains the shapes and parameter layout of a small controller and a one-hidden-layer
Lyapunov net, and the scalar loss `lossProgram`. The same TorchLean program plays two roles. It
executes and trains at `Binary32`, and it lowers to a graph of registered operators on which the
in-repo IBP/CROWN bound propagation runs.
-/

@[expose] public section

open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


open TorchLean.Tensor

namespace TorchLean.Examples.Lyapunov

/-- State dimension for the 2D Lyapunov example. -/
abbrev xDim : Nat := 2
/-- Control dimension for the 2D Lyapunov example. -/
abbrev uDim : Nat := 1

/-- Tensor shape for the state vector `x`. -/
def xShape : Shape := [xDim]
/-- Tensor shape for the control vector `u`. -/
def uShape : Shape := [uDim]

/-- Parameter shapes for the two-stage controller and Lyapunov network, as a flat list. -/
def paramShapes (width : Nat) : List Shape :=
  [ [uDim, xDim] -- Wc
  , [uDim]       -- bc
  , [width, xDim] -- W1
  , [width]       -- b1
  , [1, width]    -- W2
  , [1]           -- b2
  ]

/-!
Loss program:

$$
\begin{aligned}
\operatorname{penalty}_{\mathrm{pos}}
  &= \operatorname{ReLU}\!\left(c_V\lVert x\rVert^2-V(x)\right),\\
\operatorname{penalty}_{\mathrm{dec}}
  &= \operatorname{ReLU}\!\left(\dot V(x)+c_DV(x)\right),\\
\operatorname{loss}
  &= \operatorname{penalty}_{\mathrm{pos}}+\operatorname{penalty}_{\mathrm{dec}}.
\end{aligned}
$$

Where:

$$
\begin{aligned}
u(x) &= \mathrm{scale}_U\tanh(W_cx+b_c),\\
s(x) &= w_2\mathbin{\cdot}\tanh(W_1x+b_1)+b_2,\\
V(x) &= s(x)^2,\\
\dot V(x) &= \nabla V(x)\mathbin{\cdot}f(x,u(x)).
\end{aligned}
$$

The dynamics in the last line are van der Pol with the control added to the second coordinate,
$f(x, u) = (x_2,\ -x_1 + \mu(1 - x_1^2)x_2 + u)$. `lossProgram` fixes $\mu = \mathrm{scale}_U = 1$
and $c_V = c_D = 1/10$. We compute $\nabla V$ analytically for the one-hidden-layer tanh network
followed by a square, so training needs only first-order AD.
-/

/-- Evaluate the controller and return its scalar control output. -/
@[noinline, nospecialize]
def controllerOutput
    {β : Type} [Storage β] [Context β]
    {m : Type → Type} [Monad m]
    [Ops (m := m) (α := β)]
    (wC : Ops.Ref (m := m) (α := β) [uDim, xDim])
    (bC : Ops.Ref (m := m) (α := β) [uDim])
    (x : Ops.Ref (m := m) (α := β) xShape)
    (scaleU : β) : m (Ops.Ref (m := m) (α := β) []) := do
  let uPre ← Ops.apply (m := m) (α := β)
      (.linear (lead := []) (inDim := xDim) (outDim := uDim)) (wC, bC, x, ())
  let uT ← Ops.apply (m := m) (α := β) (.tanh (s := uShape)) (uPre, ())
  let uVec ← Ops.apply (m := m) (α := β) (.scale (s := uShape))
      (uT, .data (Ops.dataConst (m := m) (α := β) (Tensor.scalar scaleU)),
        ())
  Ops.apply (m := m) (α := β)
      (.indexAt (s := [uDim]) ⟨0, Shape.AxisInBounds.proof⟩ ⟨0, by decide⟩) (uVec, ())

/-- Evaluate the Lyapunov network together with its analytic gradient in the state coordinates. -/
@[noinline, nospecialize]
def valueAndGradient
    {β : Type} [Storage β] [Context β]
    {m : Type → Type} [Monad m]
    [Ops (m := m) (α := β)]
    (width : Nat)
    (w1 : Ops.Ref (m := m) (α := β) [width, xDim])
    (b1 : Ops.Ref (m := m) (α := β) [width])
    (w2 : Ops.Ref (m := m) (α := β) [1, width])
    (b2 : Ops.Ref (m := m) (α := β) [1])
    (x : Ops.Ref (m := m) (α := β) xShape)
    (oneS : Ops.Ref (m := m) (α := β) [])
    (two : β) :
    m (Ops.Ref (m := m) (α := β) [] ×
      Ops.Ref (m := m) (α := β) xShape) := do
  let z1 ← Ops.apply (m := m) (α := β)
      (.linear (lead := []) (inDim := xDim) (outDim := width)) (w1, b1, x, ())
  let h1 ← Ops.apply (m := m) (α := β) (.tanh (s := [width])) (z1, ())
  let sVec ← Ops.apply (m := m) (α := β)
      (.linear (lead := []) (inDim := width) (outDim := 1)) (w2, b2, h1, ())
  let s0 : Ops.Ref (m := m) (α := β) [] ←
    Ops.apply (m := m) (α := β) (.reshape (s := [1]) (t := []) (by
      simp)) (sVec, ())
  let V ← Ops.apply (m := m) (α := β) (.mul (s := [])) (s0, s0, ())

  let w2Row : Ops.Ref (m := m) (α := β) [width] ←
    Ops.apply (m := m) (α := β)
        (.reshape (s := [1, width]) (t := [width]) (by
      simp)) (w2, ())
  let h1Sq ← Ops.apply (m := m) (α := β) (.mul (s := [width])) (h1, h1, ())
  let oneW ← Ops.apply (m := m) (α := β)
      (.broadcast (s := []) (t := [width]) (Shape.scalar_canBroadcastTo [width]).dims
        (Shape.scalar_canBroadcastTo [width]).broadcastDims) (oneS, ())
  let dh ← Ops.apply (m := m) (α := β) (.sub (s := [width])) (oneW, h1Sq, ())
  let gHidden ← Ops.apply (m := m) (α := β) (.mul (s := [width]))
      (w2Row, dh, ())
  let gHiddenM ← Ops.apply (m := m) (α := β)
      (.reshape (s := [width]) (t := [width, 1]) (by
      simp)) (gHidden, ())
  let w1T ← Ops.apply (m := m) (α := β)
      (.permute (s := [width, xDim]) (Shape.adjacentSwap [width, xDim] 0)
        (Shape.swapAdjacentAtDepth [width, xDim] 0) (by tensor_shape)) (w1, ())
  let dsM ← Ops.apply (m := m) (α := β)
      (.matmul (batchA := []) (batchB := []) (batch := []) (m := xDim) (n := width) (p := 1) false
        false (fun h => nomatch h) (fun h => nomatch h) Shape.BroadcastTo.proof
        Shape.BroadcastTo.proof) (w1T, gHiddenM, ())
  let ds ← Ops.apply (m := m) (α := β)
      (.reshape (s := [xDim, 1]) (t := xShape) (by
      simp [xShape])) (dsM, ())
  let k : Ops.Ref (m := m) (α := β) [] ←
    Ops.apply (m := m) (α := β) (.scale (s := []))
        (s0, .data (Ops.dataConst (m := m) (α := β) (Tensor.scalar two)), ())
  let kV ← Ops.apply (m := m) (α := β)
      (.broadcast (s := []) (t := xShape) (Shape.scalar_canBroadcastTo xShape).dims
        (Shape.scalar_canBroadcastTo xShape).broadcastDims) (k, ())
  let gradV ← Ops.apply (m := m) (α := β) (.mul (s := xShape)) (kV, ds, ())
  pure (V, gradV)

/-- Evaluate the closed-loop van der Pol-like dynamics at `x`. -/
@[noinline, nospecialize]
def closedLoopDynamics
    {β : Type} [Storage β] [Context β]
    {m : Type → Type} [Monad m]
    [Ops (m := m) (α := β)]
    (x : Ops.Ref (m := m) (α := β) xShape)
    (u0 oneS : Ops.Ref (m := m) (α := β) [])
    (mu one : β) : m (Ops.Ref (m := m) (α := β) xShape) := do
  let x1 ← Ops.apply (m := m) (α := β)
      (.indexAt (s := [xDim]) ⟨0, Shape.AxisInBounds.proof⟩ ⟨0, by decide⟩) (x, ())
  let x2 ← Ops.apply (m := m) (α := β)
      (.indexAt (s := [xDim]) ⟨0, Shape.AxisInBounds.proof⟩ ⟨1, by decide⟩) (x, ())
  let x1Sq0 ← Ops.apply (m := m) (α := β) (.mul (s := [])) (x1, x1, ())
  let oneMinus ← Ops.apply (m := m) (α := β) (.sub (s := []))
      (oneS, x1Sq0, ())
  let term0 ← Ops.apply (m := m) (α := β) (.mul (s := [])) (oneMinus, x2, ())
  let term ← Ops.apply (m := m) (α := β) (.scale (s := []))
      (term0, .data (Ops.dataConst (m := m) (α := β) (Tensor.scalar mu)), ())
  let negx1 ← Ops.apply (m := m) (α := β) (.scale (s := []))
      (x1, .data (Ops.dataConst (m := m) (α := β) (Tensor.scalar (-one))),
        ())
  let dx2pre ← Ops.apply (m := m) (α := β) (.add (s := [])) (negx1, term, ())
  let dx2 ← Ops.apply (m := m) (α := β) (.add (s := [])) (dx2pre, u0, ())
  let x2V ← Ops.apply (m := m) (α := β)
      (.reshape (s := []) (t := [1]) (by simp)) (x2, ())
  let dx2V ← Ops.apply (m := m) (α := β)
      (.reshape (s := []) (t := [1]) (by simp)) (dx2, ())
  Ops.apply (m := m) (α := β) (.cat (s := .dim 0 []) ⟨0, by simp⟩ [1, 1])
      (x2V, dx2V, ())

/-- Sum the positivity and decrease penalties formed from `V`, `∇V` and the closed-loop dynamics. -/
@[noinline, nospecialize]
def lossFromDynamics
    {β : Type} [Storage β] [Context β]
    {m : Type → Type} [Monad m]
    [Ops (m := m) (α := β)]
    (x : Ops.Ref (m := m) (α := β) xShape)
    (V : Ops.Ref (m := m) (α := β) [])
    (gradV dynamics : Ops.Ref (m := m) (α := β) xShape)
    (cV cD : β) : m (Ops.Ref (m := m) (α := β) []) := do
  let prod ← Ops.apply (m := m) (α := β) (.mul (s := xShape))
      (gradV, dynamics, ())
  let Vdot ←
      (Ops.Ref.cast (Shape.reduce_finRange xShape) <$>
        Ops.apply (m := m) (α := β)
          (.reduce (s := xShape) .sum (List.finRange _) false
            (fun h => by rcases h with h | h <;> cases h)) (prod, ()))
  let xSqV ← Ops.apply (m := m) (α := β) (.mul (s := xShape)) (x, x, ())
  let xSq ←
      (Ops.Ref.cast (Shape.reduce_finRange xShape) <$>
        Ops.apply (m := m) (α := β)
          (.reduce (s := xShape) .sum (List.finRange _) false
            (fun h => by rcases h with h | h <;> cases h)) (xSqV, ()))
  let posScaled ← Ops.apply (m := m) (α := β) (.scale (s := []))
      (xSq, .data (Ops.dataConst (m := m) (α := β) (Tensor.scalar cV)), ())
  let posExpr ← Ops.apply (m := m) (α := β) (.sub (s := []))
      (posScaled, V, ())
  let posPenalty ← Ops.apply (m := m) (α := β) (.relu (s := []))
      (posExpr, ())
  let decScaled ← Ops.apply (m := m) (α := β) (.scale (s := []))
      (V, .data (Ops.dataConst (m := m) (α := β) (Tensor.scalar cD)), ())
  let decExpr ← Ops.apply (m := m) (α := β) (.add (s := []))
      (Vdot, decScaled, ())
  let decPenalty ← Ops.apply (m := m) (α := β) (.relu (s := []))
      (decExpr, ())
  Ops.apply (m := m) (α := β) (.add (s := [])) (posPenalty, decPenalty, ())

/-- The full training loss as a graph program: the positivity penalty plus the decrease penalty.
Both Lean stages train on it, and IBP and CROWN bound it at the end.

It is marked `noinline` and `nospecialize` because the program is built once and run many times,
so specializing it per storage type would cost compile time and buy nothing at run time. -/
@[noinline, nospecialize]
def lossProgram (width : Nat) :
    ∀ {β : Type}, [Storage β] → [Context β] →
      Program β (paramShapes width ++ [xShape]) [] :=
  fun {β} _ _ =>
    fun {m} _ _ =>
      fun wC bC w1 b1 w2 b2 x =>
        (do
          let mu : β := ((1 : Nat) : β)
          let scaleU : β := ((1 : Nat) : β)
          let cV : β := ((1 : Nat) : β) / ((10 : Nat) : β)
          let cD : β := ((1 : Nat) : β) / ((10 : Nat) : β)
          let one : β := ((1 : Nat) : β)
          let two : β := ((2 : Nat) : β)

          let oneS ← Ops.const (m := m) (α := β) (s := [])
            (Tensor.full [] one)
          let u0 ← controllerOutput (m := m) wC bC x scaleU
          let (V, gradV) ←
            valueAndGradient (m := m) width w1 b1 w2 b2 x oneS two
          let dynamics ← closedLoopDynamics (m := m) x u0 oneS mu one
          lossFromDynamics (m := m) x V gradV dynamics cV cD
          : m (Ops.Ref (m := m) (α := β) []))

end TorchLean.Examples.Lyapunov
