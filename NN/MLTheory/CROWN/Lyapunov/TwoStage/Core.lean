/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec
public import NN.Runtime.Autograd.Model.Functional.Core
public import NN.Runtime.Autograd.Model.Functional.ShapeOps

/-!
# TwoStage Core

Shared core for the **TwoStage neural-controller / neural-Lyapunov workflows**.

This directory (`NN/MLTheory/CROWN/Lyapunov/TwoStage/`) is the Lean counterpart of the “three
pipeline” workflow in the TorchLean paper (`arXiv:2602.22631`, Figure 7):

- (i) **Python-only**: PyTorch + α/β-CROWN produce numeric bounds; a Lean result is conditional on
  a proof that the imported certificate is valid for the stated model and region.
- (ii) **Hybrid**: Stage-1 training in PyTorch, exported as *float32 bit patterns*; Stage-2
  refinement + the final IBP/CROWN check run inside TorchLean under exact `ExecFloat.Binary 8 23`
  semantics.
- (iii) **All-in-Lean**: both stages run inside TorchLean under `ExecFloat.Binary 8 23`; the final
IBP/CROWN
  check is also in Lean.

This file is shared by (ii) and (iii). It contains:
- the shapes / parameter pack layout for a small controller and a 1-hidden-layer Lyapunov net, and
- the scalar TorchLean `lossProgram` used for both training and verification lowering.

Key point: the *same* TorchLean program is used in two roles:
- **execution/training** (with `α = IEEE32Exec`), and
- **lowering** to the op-tagged verifier IR used by in-repo IBP/CROWN bound propagation.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


open Spec TorchLean
open TorchLean.Tensor
open Runtime
open Runtime.Autograd

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.Core

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

The last line uses the van der Pol-like dynamics. We compute $\nabla V$ analytically for the
one-hidden-layer tanh network followed by a square, so training needs only first-order AD.
-/

/-- Evaluate the controller and return its scalar control output. -/
@[noinline, nospecialize]
def Internal.controllerOutput
    {β : Type} [TorchLean.Storage β] [Context β]
    {m : Type → Type} [Monad m]
    [Runtime.Autograd.Torch.Ops (m := m) (α := β)]
    (wC : Runtime.Autograd.Model.RefTy (m := m) (α := β) [uDim, xDim])
    (bC : Runtime.Autograd.Model.RefTy (m := m) (α := β) [uDim])
    (x : Runtime.Autograd.Model.RefTy (m := m) (α := β) xShape)
    (scaleU : β) : m (Runtime.Autograd.Model.RefTy (m := m) (α := β) []) := do
  let uPre ← Runtime.Autograd.Torch.linear
    (m := m) (α := β) (inDim := xDim) (outDim := uDim) wC bC x
  let uT ← Runtime.Autograd.Model.tanh (m := m) (α := β) (s := uShape) uPre
  let uVec ← Runtime.Autograd.Model.scale (m := m) (α := β) (s := uShape) uT (c := scaleU)
  Runtime.Autograd.Model.select (m := m) (α := β) (s := [uDim]) 0 uVec ⟨0, by decide⟩

/-- Evaluate the Lyapunov network together with its analytic gradient in the state coordinates. -/
@[noinline, nospecialize]
def Internal.lyapunovValueAndGradient
    {β : Type} [TorchLean.Storage β] [Context β]
    {m : Type → Type} [Monad m]
    [Runtime.Autograd.Torch.Ops (m := m) (α := β)]
    (width : Nat)
    (w1 : Runtime.Autograd.Model.RefTy (m := m) (α := β) [width, xDim])
    (b1 : Runtime.Autograd.Model.RefTy (m := m) (α := β) [width])
    (w2 : Runtime.Autograd.Model.RefTy (m := m) (α := β) [1, width])
    (b2 : Runtime.Autograd.Model.RefTy (m := m) (α := β) [1])
    (x : Runtime.Autograd.Model.RefTy (m := m) (α := β) xShape)
    (oneS : Runtime.Autograd.Model.RefTy (m := m) (α := β) [])
    (two : β) :
    m (Runtime.Autograd.Model.RefTy (m := m) (α := β) [] ×
      Runtime.Autograd.Model.RefTy (m := m) (α := β) xShape) := do
  let z1 ← Runtime.Autograd.Torch.linear
    (m := m) (α := β) (inDim := xDim) (outDim := width) w1 b1 x
  let h1 ← Runtime.Autograd.Model.tanh (m := m) (α := β) (s := [width]) z1
  let sVec ← Runtime.Autograd.Torch.linear
    (m := m) (α := β) (inDim := width) (outDim := 1) w2 b2 h1
  let s0 : Runtime.Autograd.Model.RefTy (m := m) (α := β) [] ←
    Runtime.Autograd.Model.reshape (m := m) (α := β) (s₁ := [1]) (s₂ := []) sVec (by
      simp [Spec.Shape.size])
  let V ← Runtime.Autograd.Model.mul (m := m) (α := β) (s := []) s0 s0

  let w2Row : Runtime.Autograd.Model.RefTy (m := m) (α := β) [width] ←
    Runtime.Autograd.Model.reshape (m := m) (α := β) (s₁ := [1, width]) (s₂ := [width]) w2 (by
      simp [Spec.Shape.size])
  let h1Sq ← Runtime.Autograd.Model.mul (m := m) (α := β) (s := [width]) h1 h1
  let oneW ← Runtime.Autograd.Model.broadcastTo (m := m) (α := β) (s₁ := [])
    (s₂ := [width]) (Shape.CanBroadcastTo.scalarTo [width]) oneS
  let dh ← Runtime.Autograd.Model.sub (m := m) (α := β) (s := [width]) oneW h1Sq
  let gHidden ← Runtime.Autograd.Model.mul (m := m) (α := β) (s := [width]) w2Row dh
  let gHiddenM ← Runtime.Autograd.Model.reshape (m := m) (α := β)
    (s₁ := [width]) (s₂ := [width, 1]) gHidden (by
      simp [Spec.Shape.size])
  let w1T ← Runtime.Autograd.Model.swapAdjacentAtDepth (m := m) (α := β)
    (s := [width, xDim]) 0 w1
  let dsM ← Runtime.Autograd.Model.matmul (m := m) (α := β)
    (batchA := []) (batchB := []) (batch := [])
    (mDim := xDim) (nDim := width) (pDim := 1) w1T gHiddenM
  let ds ← Runtime.Autograd.Model.reshape (m := m) (α := β)
    (s₁ := [xDim, 1]) (s₂ := xShape) dsM (by
      simp [xShape, Spec.Shape.size])
  let k : Runtime.Autograd.Model.RefTy (m := m) (α := β) [] ←
    Runtime.Autograd.Model.scale (m := m) (α := β) (s := []) s0 (c := two)
  let kV ← Runtime.Autograd.Model.broadcastTo (m := m) (α := β) (s₁ := []) (s₂ := xShape)
    (Shape.CanBroadcastTo.scalarTo xShape) k
  let gradV ← Runtime.Autograd.Model.mul (m := m) (α := β) (s := xShape) kV ds
  pure (V, gradV)

/-- Evaluate the closed-loop van der Pol-like dynamics at `x`. -/
@[noinline, nospecialize]
def Internal.closedLoopDynamics
    {β : Type} [TorchLean.Storage β] [Context β]
    {m : Type → Type} [Monad m]
    [Runtime.Autograd.Torch.Ops (m := m) (α := β)]
    (x : Runtime.Autograd.Model.RefTy (m := m) (α := β) xShape)
    (u0 oneS : Runtime.Autograd.Model.RefTy (m := m) (α := β) [])
    (mu one : β) : m (Runtime.Autograd.Model.RefTy (m := m) (α := β) xShape) := do
  let x1 ← Runtime.Autograd.Model.select (m := m) (α := β) (s := [xDim]) 0 x ⟨0, by decide⟩
  let x2 ← Runtime.Autograd.Model.select (m := m) (α := β) (s := [xDim]) 0 x ⟨1, by decide⟩
  let x1Sq0 ← Runtime.Autograd.Model.mul (m := m) (α := β) (s := []) x1 x1
  let oneMinus ← Runtime.Autograd.Model.sub (m := m) (α := β) (s := []) oneS x1Sq0
  let term0 ← Runtime.Autograd.Model.mul (m := m) (α := β) (s := []) oneMinus x2
  let term ← Runtime.Autograd.Model.scale (m := m) (α := β) (s := []) term0 (c := mu)
  let negx1 ← Runtime.Autograd.Model.scale (m := m) (α := β) (s := []) x1 (c := (-one))
  let dx2pre ← Runtime.Autograd.Model.add (m := m) (α := β) (s := []) negx1 term
  let dx2 ← Runtime.Autograd.Model.add (m := m) (α := β) (s := []) dx2pre u0
  let x2V ← Runtime.Autograd.Model.reshape (m := m) (α := β) (s₁ := [])
    (s₂ := [1]) x2 (by simp [Spec.Shape.size])
  let dx2V ← Runtime.Autograd.Model.reshape (m := m) (α := β) (s₁ := [])
    (s₂ := [1]) dx2 (by simp [Spec.Shape.size])
  Runtime.Autograd.Model.concatLeadingAxis (m := m) (α := β) (s := [])
    (nDim := 1) (mDim := 1) x2V dx2V

/-- Form the positivity and decrease penalties from `V`, `∇V`, and the closed-loop dynamics. -/
@[noinline, nospecialize]
def Internal.lossFromDynamics
    {β : Type} [TorchLean.Storage β] [Context β]
    {m : Type → Type} [Monad m]
    [Runtime.Autograd.Torch.Ops (m := m) (α := β)]
    (x : Runtime.Autograd.Model.RefTy (m := m) (α := β) xShape)
    (V : Runtime.Autograd.Model.RefTy (m := m) (α := β) [])
    (gradV dynamics : Runtime.Autograd.Model.RefTy (m := m) (α := β) xShape)
    (cV cD : β) : m (Runtime.Autograd.Model.RefTy (m := m) (α := β) []) := do
  let prod ← Runtime.Autograd.Model.mul (m := m) (α := β) (s := xShape) gradV dynamics
  let Vdot ← Runtime.Autograd.Model.sum (m := m) (α := β) (s := xShape) prod
  let xSqV ← Runtime.Autograd.Model.mul (m := m) (α := β) (s := xShape) x x
  let xSq ← Runtime.Autograd.Model.sum (m := m) (α := β) (s := xShape) xSqV
  let posScaled ← Runtime.Autograd.Model.scale (m := m) (α := β) (s := []) xSq (c := cV)
  let posExpr ← Runtime.Autograd.Model.sub (m := m) (α := β) (s := []) posScaled V
  let posPenalty ← Runtime.Autograd.Model.relu (m := m) (α := β) (s := []) posExpr
  let decScaled ← Runtime.Autograd.Model.scale (m := m) (α := β) (s := []) V (c := cD)
  let decExpr ← Runtime.Autograd.Model.add (m := m) (α := β) (s := []) Vdot decScaled
  let decPenalty ← Runtime.Autograd.Model.relu (m := m) (α := β) (s := []) decExpr
  Runtime.Autograd.Model.add (m := m) (α := β) (s := []) posPenalty decPenalty

/-- The full stage-1 loss as a graph program: MSE-style fit plus the two Lyapunov penalties.

Marked `noinline`/`nospecialize` on purpose. The program is built once and then run many times, so
specializing it per storage type costs compile time without buying anything at run time. -/
@[noinline, nospecialize]
def lossProgram (width : Nat) :
    ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
      Runtime.Autograd.Model.Program β (paramShapes width ++ [xShape]) [] :=
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

          let oneS ← Runtime.Autograd.Model.const (m := m) (α := β) (s := []) (Tensor.full [] one)
          let u0 ← Internal.controllerOutput (m := m) wC bC x scaleU
          let (V, gradV) ←
            Internal.lyapunovValueAndGradient (m := m) width w1 b1 w2 b2 x oneS two
          let dynamics ← Internal.closedLoopDynamics (m := m) x u0 oneS mu one
          Internal.lossFromDynamics (m := m) x V gradV dynamics cV cD
          : m (Runtime.Autograd.Model.RefTy (m := m) (α := β) []))

end NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
