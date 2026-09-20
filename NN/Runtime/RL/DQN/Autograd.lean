/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Loss

/-!
# Differentiable DQN Objectives

These losses build backend-generic autograd programs for value learning. Bellman targets are
detached inside the objective: a DQN update fits the online prediction to a fixed target, even when
the caller computed that target through a differentiable network.

Huber loss bounds the derivative with respect to each TD residual. It limits the influence of large
Bellman errors without bounding the parameter gradient or guaranteeing that training is stable.
-/

@[expose] public section

namespace Runtime.RL.DQN.Autograd

open Spec TorchLean Runtime.Autograd.Model

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Huber loss between predictions and detached Bellman targets.

For a positive `delta`, the elementwise loss is `d² / 2` when `|d| ≤ delta` and
`delta * (|d| - delta / 2)` otherwise, where `d = prediction - target`. The target receives
zero gradient. `reduction` applies to all entries of the prediction tensor.

This is Huber loss, whose outer derivative has magnitude `delta`. For `delta ≠ 1`, it differs
from Smooth L1 loss by a factor of `delta`.
-/
def huberTDLoss
    {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {s : Shape}
    (prediction target : RefTy (m := m) (α := α) s)
    (delta : α := 1) (reduction : Loss.Reduction := .mean) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let fixedTarget ← detach (m := m) (α := α) target
  let residual ← sub (m := m) (α := α) prediction fixedTarget
  let magnitude ← abs (m := m) (α := α) residual
  let bounded ← clamp (m := m) (α := α) magnitude 0 delta
  -- Factoring avoids subtracting and then adding `delta` in the quadratic-region gradient.
  -- That cancellation can erase small residuals in floating-point reverse mode.
  let halfBounded ← scale (m := m) (α := α) bounded ((1 : α) / 2)
  let remainder ← sub (m := m) (α := α) magnitude halfBounded
  let losses ← mul (m := m) (α := α) bounded remainder
  Loss.reduceLoss (m := m) (α := α) losses reduction

/--
Mean DQN Huber loss for a batch of Q vectors, one-hot actions, and scalar Bellman targets.

`qValues` and `actionOneHot` have shape `(batch, nActions)`; `target` has shape `(batch)`.
Each row of `actionOneHot` must select exactly one action. Actions and targets are detached,
so only the selected online Q values receive gradients. Reduction averages over transitions;
adding unused actions does not rescale the loss.

Pass a positive `delta`. Target construction, termination masking, and target-network updates
remain the caller's responsibility.
-/
def actionHuberLossBatch
    {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {batch nActions : Nat} [NeZero batch] [NeZero nActions]
    (qValues actionOneHot :
      RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (target : RefTy (m := m) (α := α) (.dim batch .scalar))
    (delta : α := 1) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let s : Shape := .dim batch (.dim nActions .scalar)
  let _ : Shape.WellFormed s := by infer_instance
  let _ : Shape.HasNonemptyAxis 1 s :=
    Shape.inferNonemptyAxis (by simp [s, Shape.rank])
  let fixedActions ← detach (m := m) (α := α) actionOneHot
  let masked ← mul (m := m) (α := α) qValues fixedActions
  let selected ← reduceSum (m := m) (α := α) (s := s) (axis := 1) masked
  huberTDLoss (m := m) (α := α) selected target delta

end Runtime.RL.DQN.Autograd
