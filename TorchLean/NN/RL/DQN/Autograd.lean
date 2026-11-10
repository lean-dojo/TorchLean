/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Loss

/-!
# Differentiable DQN Objectives

Backend-generic autograd programs for value-learning losses. The Bellman targets are detached
inside the objective, so a DQN update fits the online prediction to a fixed target even when the
caller computed that target through a differentiable network.

Huber loss bounds the derivative with respect to each TD residual, which limits the influence of
large Bellman errors. The parameter gradient has no such bound, and training can diverge.

We wrote `huber` to match `torch.nn.HuberLoss` (PyTorch 2.8 docs): the same two branches, with
`delta = 1` and the mean as defaults. At `delta = 1` its derivative clips each residual to
$[-1, 1]$, the error clipping in the Methods of Mnih et al. (2015, Nature 518:529-533). The test
`checkDQNHuber` in `TorchLeanTest/CUDA/Trainer.lean` runs `huber` on the CUDA tape in binary32
and binary64, both reductions, and compares the bits of the loss and the gradient with
`Tensor.huberLoss` and `Tensor.huberLossBackward`. That reference is ours, and no test runs
PyTorch on these objectives.
-/

@[expose] public section

namespace TorchLean.RL.DQN.Autograd

open Ops

variable {α : Type} [Storage α] [Context α]

/--
Huber loss between predictions and Bellman targets read without a gradient: one application of
`Op.huberLoss`, the mean or the sum by `reduction`.

For a positive `delta`, the elementwise loss is `d² / 2` when `|d| ≤ delta` and
`delta * (|d| - delta / 2)` otherwise, where `d = prediction - target` (`Huber.term`). The
target receives zero gradient. `reduction` applies to all entries of the prediction tensor.

Outside `[-delta, delta]` the derivative has magnitude `delta`. Huber loss is `delta` times
Smooth L1 loss with `beta = delta`, so the two agree only for `delta = 1`.
-/
def huber
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {s : Shape}
    (prediction target : Ref (m := m) (α := α) s)
    (delta : α := 1) (reduction : Loss.Reduction := .mean) :
    m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.apply (m := m) (α := α) (.huberLoss (reduction == .mean))
    (prediction, .ref target,
      .data (Ops.dataConst (m := m) (α := α) (Tensor.scalar delta)), ())

/--
Mean over the batch of the Huber loss between `qValues[b][actions[b]]`, read by `Op.gather`, and
the Bellman target `target[b]`, read without a gradient, for a positive `delta`. The other entries
of `qValues`, finite or not, change neither the loss nor any gradient.
-/
def loss
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch nActions : Nat}
    (qValues : Ref (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (actions : Ops.DataRef (m := m) (α := α) (Fin nActions)
      (.dim batch .scalar))
    (target : Ref (m := m) (α := α) (.dim batch .scalar))
    (delta : α := 1) :
    m (Ref (m := m) (α := α) Shape.scalar) := do
  -- `Op.gather` along axis 1 takes indices of shape `(batch, 1)`: row `b` holds `actions[b]`.
  let actionColumn := Ops.mapData (m := m) (α := α)
    (fun a : Tensor (Fin nActions) [batch] => Tensor.reshape a [batch, 1] (by simp))
    actions
  let picked : Ops.Ref (m := m) (α := α) [batch, 1] ←
    Ref.cast (m := m) (α := α)
        (by rw [Shape.eraseAxes_selectBatch, List.append_nil]; rfl) <$>
      Ops.apply (m := m) (α := α)
        (.gather (Shape.selectDims [batch, nActions] ⟨1, Shape.AxisInBounds.proof⟩)
          (Shape.selectBatch [batch, nActions] ⟨1, Shape.AxisInBounds.proof⟩ 1)
          (.select [batch, nActions] ⟨1, Shape.AxisInBounds.proof⟩ 1))
        (qValues, actionColumn, ())
  let selected ← Ops.apply (m := m) (α := α)
    (.reshape (s := [batch, 1]) (t := [batch]) (by simp)) (picked, ())
  huber (m := m) (α := α) selected target delta

end TorchLean.RL.DQN.Autograd
