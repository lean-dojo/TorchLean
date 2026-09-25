/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Seq

/-!
# Autograd Policy-Gradient Objectives

This module provides differentiable policy-gradient / actor-critic helpers expressed in terms of
TorchLean's backend-generic `Ops` interface, so they can run under:

- the eager runtime backend (imperative autograd, GPU-capable), and
- typed graph execution (recorded graph and proof tooling).

This file lives with the RL runtime, not the TorchLean runtime internals, because these are RL
objectives that happen to be differentiable through TorchLean. It is the autograd companion to the
pure helpers in `NN.Runtime.RL.Algorithms.PolicyGradient`:

- `NN.Runtime.RL.PolicyGradient` works with concrete spec tensors (`Tensor α s`).
- `NN.Runtime.RL.PolicyGradient.Autograd` works with backend refs
  (`RefTy (m := m) (α := α) s`) so autograd can differentiate the objectives.

## Action Encoding

We assume **categorical** (finite-action) policies parameterized by logits, and we represent the
chosen action as a **one-hot** tensor with the same shape as the logits. The selected
log-probability remains differentiable with respect to the logits without introducing a separate
integer index type into the `Ops` surface.

## Primary References

- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement
  Learning" (1992): https://doi.org/10.1023/A:1022672621406
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), Chapters 12–13:
  http://incompleteideas.net/book/the-book-2nd.html
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
-/

@[expose] public section

namespace Runtime
namespace RL
namespace PolicyGradient
namespace Autograd

open Spec TorchLean
open Runtime.Autograd.Model

variable {α : Type} [TorchLean.Storage α] [Context α]

/-! ## Log-probabilities and entropy (batched, one-hot actions) -/

/--
Per-sample log-probability for one-hot actions under a batched categorical policy.

Input shapes:
- `logits : (N × A)`
- `actionOneHot : (N × A)`

Output shape:
- `logProb : (N)` where `logProb[i] = log π(a_i | s_i)`.

The log-softmax is clamped to `[-10^30, 10^30]` before it is multiplied by the one-hot mask. A
logit of `-inf` has log-probability `-inf`, and without the clamp its unselected entry would
contribute `0 * -inf = NaN` to the row sum. Log-probabilities are at most 0, so the clamp only
changes values below `-10^30`, which a policy reaches only through infinite logits or logit gaps
beyond `10^30`. Inside that range the clamp's derivative is 1, so gradients also agree with the
unclamped formula.
-/
def actionLogProbOneHot
    {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {batch nActions : Nat} [NeZero batch] [NeZero nActions]
    (logits :
      Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (actionOneHot :
      Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar))) :
    m (Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch .scalar)) := do
  let s : Shape := .dim batch (.dim nActions .scalar)
  let _ : Shape.WellFormed s := by infer_instance
  let _ : Shape.HasNonemptyAxis 1 s :=
    Shape.inferNonemptyAxis (by simp [s, Shape.rank])
  let logp ← F.logSoftmax (m := m) (α := α) (s := s) 1 logits
  let bound : α := ((10 ^ 30 : Nat) : α)
  let finiteLogp ← clamp (m := m) (α := α) (s := s) logp (-bound) bound
  let masked ← mul (m := m) (α := α) (s := s) actionOneHot finiteLogp
  reduceSum (m := m) (α := α) (s := s) (axis := 1) masked

/--
Mean entropy of a batched categorical policy.

Input shape:
- `logits : (N × A)`

Output shape:
- scalar entropy mean: `mean_i[ -Σ_a p_i(a) log p_i(a) ]`.
-/
def entropyMean
    {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {batch nActions : Nat} [NeZero batch] [NeZero nActions]
    (logits :
      Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar))) :
    m (Runtime.Autograd.Model.RefTy (m := m) (α := α) Shape.scalar) := do
  let s : Shape := .dim batch (.dim nActions .scalar)
  let _ : Shape.WellFormed s := by infer_instance
  let _ : Shape.HasNonemptyAxis 1 s :=
    Shape.inferNonemptyAxis (by simp [s, Shape.rank])
  let logp ← F.logSoftmax (m := m) (α := α) (s := s) 1 logits
  let probs ← exp (m := m) (α := α) (s := s) logp
  let plogp ← mul (m := m) (α := α) (s := s) probs logp
  let sumActions ← reduceSum (m := m) (α := α) (s := s) (axis := 1) plogp
  let entropyVec ← scale (m := m) (α := α) (s := .dim batch .scalar) sumActions (-1)
  Runtime.Autograd.Model.F.mean (m := m) (α := α) (s := .dim batch .scalar) entropyVec

/-! ## PPO (batched) -/

/--
PPO clipped surrogate objective (the thing to maximize), computed per sample:

`L_clip_i = min(r_i * A_i, clip(r_i, 1-ε, 1+ε) * A_i)`

where `r_i = exp(logπ_new(a_i|s_i) - logπ_old(a_i|s_i))`.
-/
def ppoClippedObjective
    {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {batch nActions : Nat} [NeZero batch] [NeZero nActions]
    (newLogits :
      Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (actionOneHot :
      Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (oldLogProb : Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch .scalar))
    (advantage : Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch .scalar))
    (clipEps : α := (1 : α) / ((5 : Nat) : α)) :
    m (Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch .scalar)) := do
  let sVec : Shape := .dim batch .scalar
  let newLogProb ←
    actionLogProbOneHot (m := m) (α := α) (batch := batch) (nActions := nActions)
      newLogits actionOneHot
  let diff ← sub (m := m) (α := α) (s := sVec) newLogProb oldLogProb
  let ratio ← exp (m := m) (α := α) (s := sVec) diff
  let clippedRatio ←
    clamp (m := m) (α := α) (s := sVec) ratio ((1 : α) - clipEps) ((1 : α) + clipEps)
  let unclipped ← mul (m := m) (α := α) (s := sVec) ratio advantage
  let clipped ← mul (m := m) (α := α) (s := sVec) clippedRatio advantage
  min (m := m) (α := α) (s := sVec) unclipped clipped

/--
PPO scalar loss to *minimize* (mean over batch):

`loss = -mean(L_clip) + c_v * MSE(v, v_target) - c_e * mean(entropy)`

This is the standard discrete-action PPO loss used in many reference implementations.
-/
def ppoLoss
    {m : Type → Type} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {batch nActions : Nat} [NeZero batch] [NeZero nActions]
    (newLogits :
      Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (actionOneHot :
      Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (oldLogProb : Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch .scalar))
    (advantage : Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch .scalar))
    (valuePred valueTarget :
      Runtime.Autograd.Model.RefTy (m := m) (α := α) (.dim batch (.dim 1 .scalar)))
    (clipEps : α := (1 : α) / ((5 : Nat) : α))
    (valueCoef : α := (1 : α) / ((2 : Nat) : α))
    (entropyCoef : α := (1 : α) / ((100 : Nat) : α)) :
    m (Runtime.Autograd.Model.RefTy (m := m) (α := α) Shape.scalar) := do
  let obj ←
    ppoClippedObjective (m := m) (α := α) (batch := batch) (nActions := nActions)
      newLogits actionOneHot oldLogProb advantage (clipEps := clipEps)
  let objMean ← Runtime.Autograd.Model.F.mean (m := m) (α := α) (s := .dim batch .scalar) obj
  let policyLoss ← scale (m := m) (α := α) (s := Shape.scalar) objMean (-1)
  let valueLoss ←
    TorchLean.Loss.mse (m := m) (α := α) (s := .dim batch (.dim 1 .scalar)) valuePred valueTarget
  let valueLossScaled ← scale (m := m) (α := α) (s := Shape.scalar) valueLoss valueCoef
  let entropy ← entropyMean (m := m) (α := α) (batch := batch) (nActions := nActions) newLogits
  let entropyScaled ← scale (m := m) (α := α) (s := Shape.scalar) entropy entropyCoef
  let tmp ← add (m := m) (α := α) (s := Shape.scalar) policyLoss valueLossScaled
  sub (m := m) (α := α) (s := Shape.scalar) tmp entropyScaled

/-! ## PPO module wrapper (two-model actor/critic) -/

/--
Bundle an actor and critic into an `ObjectiveDef` whose inputs are a PPO minibatch:

- `states : (N × stateDim)`
- `actionsOneHot : (N × A)`
- `oldLogProb : (N)`
- `advantages : (N)`
- `valueTarget : (N × 1)`

The model state is `actor.state ++ critic.state`, and one optimizer step updates both models.
-/
def ppoActorCriticObjectiveDef
    {stateShape : Shape} {batch nActions : Nat} [NeZero batch] [NeZero nActions]
    (actor : Runtime.Autograd.Model.Layers.Seq stateShape (.dim batch (.dim nActions .scalar)))
    (critic : Runtime.Autograd.Model.Layers.Seq stateShape (.dim batch (.dim 1 .scalar))) :
    Runtime.Autograd.Model.Module.ObjectiveDef Unit
      (Runtime.Autograd.Model.Layers.Seq.stateShapes actor ++
        Runtime.Autograd.Model.Layers.Seq.stateShapes critic)
      [stateShape, (.dim batch (.dim nActions .scalar)), (.dim batch .scalar), (.dim batch .scalar),
        (.dim batch (.dim 1 .scalar))] :=
  { initState :=
      TorchLean.TensorPack.append (α := Float)
        (ss₁ := Runtime.Autograd.Model.Layers.Seq.stateShapes actor)
        (ss₂ := Runtime.Autograd.Model.Layers.Seq.stateShapes critic)
        (Runtime.Autograd.Model.Layers.Seq.initState actor)
        (Runtime.Autograd.Model.Layers.Seq.initState critic)
    requiresGrad := Runtime.Autograd.Model.Layers.Seq.requiresGrad actor ++
      Runtime.Autograd.Model.Layers.Seq.requiresGrad critic
    validate := do
      Runtime.Autograd.Model.Layers.Seq.validate actor
      Runtime.Autograd.Model.Layers.Seq.validate critic
    loss := fun {α} => by
      intro _ _; exact
        (fun {m} _ _ =>
          Runtime.Autograd.Torch.CurriedRef.curry
            (Ref := fun sh => Runtime.Autograd.Model.RefTy (m := m) (α := α) sh)
            (ss := (Runtime.Autograd.Model.Layers.Seq.stateShapes actor ++
              Runtime.Autograd.Model.Layers.Seq.stateShapes critic) ++
              [stateShape, (.dim batch (.dim nActions .scalar)), (.dim batch .scalar),
                (.dim batch .scalar), (.dim batch (.dim 1 .scalar))])
            (β := m (Runtime.Autograd.Model.RefTy (m := m) (α := α) Shape.scalar))
            (fun args => do
              let (ps, xs) :=
                Runtime.Autograd.Torch.RefList.split
                  (Ref := fun sh => Runtime.Autograd.Model.RefTy (m := m) (α := α) sh)
                  (ss₁ := (Runtime.Autograd.Model.Layers.Seq.stateShapes actor ++
                    Runtime.Autograd.Model.Layers.Seq.stateShapes critic))
                  (ss₂ := [stateShape, (.dim batch (.dim nActions .scalar)), (.dim batch .scalar),
                    (.dim batch .scalar), (.dim batch (.dim 1 .scalar))])
                  args
              let (psActor, psCritic) :=
                Runtime.Autograd.Torch.RefList.split
                  (Ref := fun sh => Runtime.Autograd.Model.RefTy (m := m) (α := α) sh)
                  (ss₁ := Runtime.Autograd.Model.Layers.Seq.stateShapes actor)
                  (ss₂ := Runtime.Autograd.Model.Layers.Seq.stateShapes critic)
                  ps
              let .cons states (.cons actionsOneHot (.cons oldLogProb
                  (.cons advantages (.cons valueTarget .nil)))) := xs
              let logits ←
                Runtime.Autograd.Model.Layers.Seq.forwardState
                  (model := actor) (α := α) (m := m) .train psActor states
              let values ←
                Runtime.Autograd.Model.Layers.Seq.forwardState
                  (model := critic) (α := α) (m := m) .train psCritic states
              ppoLoss (m := m) (α := α) (batch := batch) (nActions := nActions)
                logits actionsOneHot oldLogProb advantages values valueTarget))
  }

end Autograd
end PolicyGradient
end RL
end Runtime
