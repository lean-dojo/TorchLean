/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Functional
public import NN.Runtime.Autograd.TorchLean.Loss
public import NN.Runtime.Autograd.TorchLean.Module
public import NN.Runtime.Autograd.TorchLean.NN

/-!
# Autograd Policy-Gradient Objectives

This module provides differentiable policy-gradient / actor-critic helpers expressed in terms of
TorchLean's backend-generic `Ops` interface, so they can run under:

- the eager runtime backend (imperative autograd, GPU-capable), and
- the compiled backend (graph recording / proof tooling).

This file lives with the RL runtime, not the TorchLean runtime internals, because these are RL
objectives that happen to be differentiable through TorchLean. It is the autograd companion to the
pure helpers in `NN.Runtime.RL.Algorithms.PolicyGradient`:

- `NN.Runtime.RL.PolicyGradient` works with concrete spec tensors (`Tensor α s`).
- `NN.Runtime.RL.PolicyGradient.Autograd` works with backend refs
  (`RefTy (m := m) (α := α) s`) so autograd can differentiate the objectives.

## Action Encoding

We assume **categorical** (finite-action) policies parameterized by logits, and we represent the
chosen action as a **one-hot** tensor with the same shape as the logits. This keeps the selected
log-probability differentiable with respect to the logits without introducing a separate integer
index type into the `Ops` surface.

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

open Spec
open _root_.Runtime.Autograd.TorchLean

variable {α : Type} [Context α] [DecidableEq Shape]

/-! ## Log-probabilities and entropy (batched, one-hot actions) -/

/--
Per-sample log-probability for one-hot actions under a batched categorical policy.

Input shapes:
- `logits : (N × A)`
- `actionOneHot : (N × A)`

Output shape:
- `logProb : (N)` where `logProb[i] = log π(a_i | s_i)`.

Implementation note: this uses `log_softmax` and a reduce-sum over the action axis.
-/
def actionLogProbOneHotBatch
    {m : Type → Type} [Monad m] [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {batch nActions : Nat}
    [hBatch : Fact (0 < batch)] [hAct : Fact (0 < nActions)]
    (logits : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (actionOneHot : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (ε : α := Numbers.epsilon) :
    m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch .scalar)) := do
  let s : Shape := .dim batch (.dim nActions .scalar)
  let _ : Shape.WellFormed s := by infer_instance
  let _ : Shape.valid_axis_inst 1 s :=
    Shape.validAxisInstOne (h₁ := Nat.ne_of_gt hBatch.out) (h₂ := Nat.ne_of_gt hAct.out)
  let logp ← logSoftmax (m := m) (α := α) (s := s) logits (ε := ε)
  let masked ← mul (m := m) (α := α) (s := s) actionOneHot logp
  reduceSum (m := m) (α := α) (s := s) (axis := 1) masked

/--
Mean entropy of a batched categorical policy.

Input shape:
- `logits : (N × A)`

Output shape:
- scalar entropy mean: `mean_i[ -Σ_a p_i(a) log p_i(a) ]`.
-/
def entropyMean
    {m : Type → Type} [Monad m] [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {batch nActions : Nat}
    [hBatch : Fact (0 < batch)] [hAct : Fact (0 < nActions)]
    (logits : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (ε : α := Numbers.epsilon) :
    m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) Shape.scalar) := do
  let s : Shape := .dim batch (.dim nActions .scalar)
  let _ : Shape.WellFormed s := by infer_instance
  let _ : Shape.valid_axis_inst 1 s :=
    Shape.validAxisInstOne (h₁ := Nat.ne_of_gt hBatch.out) (h₂ := Nat.ne_of_gt hAct.out)
  let logp ← logSoftmax (m := m) (α := α) (s := s) logits (ε := ε)
  let probs ← exp (m := m) (α := α) (s := s) logp
  let plogp ← mul (m := m) (α := α) (s := s) probs logp
  let sumActions ← reduceSum (m := m) (α := α) (s := s) (axis := 1) plogp
  let entropyVec ← scale (m := m) (α := α) (s := .dim batch .scalar) sumActions (-1)
  _root_.Runtime.Autograd.TorchLean.F.mean (m := m) (α := α) (s := .dim batch .scalar) entropyVec

/-! ## PPO (batched) -/

/--
PPO clipped surrogate objective (the thing to maximize), computed per sample:

`L_clip_i = min(r_i * A_i, clip(r_i, 1-ε, 1+ε) * A_i)`

where `r_i = exp(logπ_new(a_i|s_i) - logπ_old(a_i|s_i))`.
-/
def ppoClippedObjectiveBatch
    {m : Type → Type} [Monad m] [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {batch nActions : Nat}
    [hBatch : Fact (0 < batch)] [hAct : Fact (0 < nActions)]
    (newLogits : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (actionOneHot : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (oldLogProb : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch .scalar))
    (advantage : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch .scalar))
    (clipEps : α := (1 : α) / ((5 : Nat) : α))
    (ε : α := Numbers.epsilon) :
    m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch .scalar)) := do
  let sVec : Shape := .dim batch .scalar
  let newLogProb ←
    actionLogProbOneHotBatch (m := m) (α := α) (batch := batch) (nActions := nActions)
      newLogits actionOneHot (ε := ε)
  let diff ← sub (m := m) (α := α) (s := sVec) newLogProb oldLogProb
  let ratio ← exp (m := m) (α := α) (s := sVec) diff
  let clippedRatio ← clamp (m := m) (α := α) (s := sVec) ratio ((1 : α) - clipEps) ((1 : α) + clipEps)
  let unclipped ← mul (m := m) (α := α) (s := sVec) ratio advantage
  let clipped ← mul (m := m) (α := α) (s := sVec) clippedRatio advantage
  min (m := m) (α := α) (s := sVec) unclipped clipped

/--
PPO scalar loss to *minimize* (mean over batch):

`loss = -mean(L_clip) + c_v * MSE(v, v_target) - c_e * mean(entropy)`

This is the standard discrete-action PPO loss used in many reference implementations.
-/
def ppoLossBatch
    {m : Type → Type} [Monad m] [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {batch nActions : Nat}
    [hBatch : Fact (0 < batch)] [hAct : Fact (0 < nActions)]
    (newLogits : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (actionOneHot : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (oldLogProb : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch .scalar))
    (advantage : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch .scalar))
    (valuePred valueTarget : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch (.dim 1 .scalar)))
    (clipEps : α := (1 : α) / ((5 : Nat) : α))
    (valueCoef : α := (1 : α) / ((2 : Nat) : α))
    (entropyCoef : α := (1 : α) / ((100 : Nat) : α))
    (ε : α := Numbers.epsilon) :
    m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) Shape.scalar) := do
  let obj ←
    ppoClippedObjectiveBatch (m := m) (α := α) (batch := batch) (nActions := nActions)
      newLogits actionOneHot oldLogProb advantage (clipEps := clipEps) (ε := ε)
  let objMean ← _root_.Runtime.Autograd.TorchLean.F.mean (m := m) (α := α) (s := .dim batch .scalar) obj
  let policyLoss ← scale (m := m) (α := α) (s := Shape.scalar) objMean (-1)
  let valueLoss ← _root_.Runtime.Autograd.TorchLean.Loss.mse (m := m) (α := α) (s := .dim batch (.dim 1 .scalar)) valuePred valueTarget
  let valueLossScaled ← scale (m := m) (α := α) (s := Shape.scalar) valueLoss valueCoef
  let entropy ←
    entropyMean (m := m) (α := α) (batch := batch) (nActions := nActions) newLogits (ε := ε)
  let entropyScaled ← scale (m := m) (α := α) (s := Shape.scalar) entropy entropyCoef
  let tmp ← add (m := m) (α := α) (s := Shape.scalar) policyLoss valueLossScaled
  sub (m := m) (α := α) (s := Shape.scalar) tmp entropyScaled

/-! ## PPO module wrapper (two-model actor/critic) -/

/--
Bundle an actor and critic into a `ScalarModuleDef` whose inputs are a PPO minibatch:

- `states : (N × stateDim)`
- `actionsOneHot : (N × A)`
- `oldLogProb : (N)`
- `advantages : (N)`
- `valueTarget : (N × 1)`

The parameters are `actor.params ++ critic.params`, and one optimizer step updates both.
-/
def ppoActorCriticScalarModuleDef
    {stateShape : Shape} {batch nActions : Nat}
    [hBatch : Fact (0 < batch)] [hAct : Fact (0 < nActions)]
    (actor : _root_.Runtime.Autograd.TorchLean.NN.Seq stateShape (.dim batch (.dim nActions .scalar)))
    (critic : _root_.Runtime.Autograd.TorchLean.NN.Seq stateShape (.dim batch (.dim 1 .scalar))) :
    _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef
      (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic)
      [stateShape, (.dim batch (.dim nActions .scalar)), (.dim batch .scalar), (.dim batch .scalar),
        (.dim batch (.dim 1 .scalar))] :=
  { initParams :=
      _root_.Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.append (α := Float)
        (ss₁ := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor) (ss₂ := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic)
        (_root_.Runtime.Autograd.TorchLean.NN.Seq.initParams actor) (_root_.Runtime.Autograd.TorchLean.NN.Seq.initParams critic)
    initRequiresGrad := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramRequiresGrad actor ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.paramRequiresGrad critic
    loss := fun {α} => by
      intro _ _; exact
        (fun {m} _ _ =>
          _root_.Runtime.Autograd.Torch.CurriedRef.curry (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
            (ss := (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic) ++
              [stateShape, (.dim batch (.dim nActions .scalar)), (.dim batch .scalar), (.dim batch .scalar),
                (.dim batch (.dim 1 .scalar))])
            (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) Shape.scalar))
            (fun args => do
              let (ps, xs) :=
                _root_.Runtime.Autograd.Torch.RefList.split (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
                  (ss₁ := (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic))
                  (ss₂ := [stateShape, (.dim batch (.dim nActions .scalar)), (.dim batch .scalar),
                    (.dim batch .scalar), (.dim batch (.dim 1 .scalar))])
                  args
              let (psActor, psCritic) :=
                _root_.Runtime.Autograd.Torch.RefList.split (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
                  (ss₁ := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor)
                  (ss₂ := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic)
                  ps
              let .cons states (.cons actionsOneHot (.cons oldLogProb (.cons advantages (.cons valueTarget .nil)))) :=
                xs
              let logits ←
                _root_.Runtime.Autograd.TorchLean.NN.Seq.evalParams (model := actor) (α := α) (m := m) .train psActor states
              let values ←
                _root_.Runtime.Autograd.TorchLean.NN.Seq.evalParams (model := critic) (α := α) (m := m) .train psCritic states
              ppoLossBatch (m := m) (α := α) (batch := batch) (nActions := nActions)
                logits actionsOneHot oldLogProb advantages values valueTarget))
  }

end Autograd
end PolicyGradient
end RL
end Runtime
