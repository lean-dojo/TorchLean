/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Core
public import NN.Spec.Core.Tensor.Numerics
public import NN.Tensor.Pack

/-!
# PPO Rollouts (Discrete Actions)

This file defines:

- fixed-horizon PPO rollout records stored as typed tensors / arrays, and
- a conversion to the minibatch format expected by the PPO autograd objective
  (`Runtime.RL.PolicyGradient.Autograd.ppoActorCriticObjectiveDef`).

The single-mask tensor GAE/return definitions live in `NN.Spec.RL.Core` and are re-exported by
`NN.Runtime.RL.Core`. This typed rollout layer separates task termination from episode boundaries
when computing PPO advantages.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
-/

@[expose] public section

namespace Runtime
namespace RL
namespace PPO

open Spec TorchLean
open TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-!
## Shapes

For a fixed horizon `T`, PPO minibatches are typically stored in "PyTorch-shaped" tensors:

- `states : (T × obsShape)`
- `actionsOneHot : (T × nActions)`
- `oldLogProb : (T)`
- `advantages : (T)`
- `valueTarget : (T × 1)`
-/

/-- Batch shape for a fixed-horizon sequence of observations: `horizon × obsShape`. -/
abbrev StateBatchShape (horizon : Nat) (obsShape : Shape) : Shape :=
  obsShape.prependDim horizon

/-- Batch shape for a fixed-horizon sequence of action logits: `horizon × nActions`. -/
abbrev LogitsBatchShape (horizon nActions : Nat) : Shape :=
  [horizon, nActions]

/-- Batch shape for a fixed-horizon sequence of scalars: `horizon`. -/
abbrev ScalarBatchShape (horizon : Nat) : Shape :=
  [horizon]

/-- Batch shape for a fixed-horizon sequence of scalar values stored as a column: `horizon × 1`. -/
abbrev ValueBatchShape (horizon : Nat) : Shape :=
  [horizon, 1]

/-!
## Rollouts
-/

/--
One fixed-horizon PPO step record.

This is the “typed parallel arrays” data layout commonly used in PPO implementations, but kept as
a single record so downstream code cannot accidentally desynchronize fields.
-/
structure Step (α : Type) [TorchLean.Storage α] (obsShape : Shape) (nActions : Nat) where
  /-- Observation `s_t` (already cast into the training scalar backend). -/
  state : Tensor α obsShape
  /-- Sampled action `a_t`. -/
  action : Fin nActions
  /-- Log-probability `log π_old(a_t | s_t)` under the behavior policy. -/
  oldLogProb : α
  /-- Reward `r_t`. -/
  reward : α
  /-- Episode boundary marker (Gym-style `terminated || truncated`). -/
  done : Bool
  /-- Baseline value prediction `V(s_t)`. -/
  value : α
  /-- Bootstrap value prediction `V(s_{t+1})` (before any auto-reset). -/
  nextValue : α
  /-- Task termination suppresses value bootstrapping. An external truncation sets `done`
  but leaves this false. The default preserves the single-mask behavior of older records. -/
  terminated : Bool := done

/--
Fixed-horizon rollout buffer for PPO.

The `steps_size_eq_horizon` field records the invariant that the buffer has exactly `horizon`
steps; this lets downstream tensor conversion be total without runtime bounds checks.
-/
structure Rollout (α : Type) [TorchLean.Storage α] (obsShape : Shape) (nActions horizon : Nat) where
  steps : Array (Step α obsShape nActions)
  /-- Invariant: fixed-horizon rollouts always have exactly `horizon` steps. -/
  steps_size_eq_horizon : steps.size = horizon

/-- Named tensors consumed by one PPO actor-critic update. -/
structure TrainingBatch (α : Type) [TorchLean.Storage α]
    (obsShape : Shape) (nActions horizon : Nat) where
  /-- Observations for each rollout step. -/
  states : Tensor α (StateBatchShape horizon obsShape)
  /-- Sampled actions encoded against the action-logit axis. -/
  actionsOneHot : Tensor α (LogitsBatchShape horizon nActions)
  /-- Behavior-policy log-probability for each sampled action. -/
  oldLogProb : Tensor α (ScalarBatchShape horizon)
  /-- Normalized generalized advantages used by the policy objective. -/
  advantages : Tensor α (ScalarBatchShape horizon)
  /-- Lambda-return targets used by the value objective. -/
  valueTargets : Tensor α (ValueBatchShape horizon)

namespace TrainingBatch.Internal

/-- Pack a named PPO batch for the low-level autograd objective. -/
def arguments {α : Type} [TorchLean.Storage α]
    {obsShape : Shape} {nActions horizon : Nat}
    (batch : TrainingBatch α obsShape nActions horizon) :
    TorchLean.TensorPack α
      [StateBatchShape horizon obsShape,
       LogitsBatchShape horizon nActions,
       ScalarBatchShape horizon,
       ScalarBatchShape horizon,
       ValueBatchShape horizon] :=
  .cons batch.states <|
    .cons batch.actionsOneHot <|
      .cons batch.oldLogProb <|
        .cons batch.advantages <|
          .cons batch.valueTargets .nil

end TrainingBatch.Internal

namespace Rollout

namespace Internal

/-- GAE with separate masks for the next-state value and continuation into the next step. -/
def generalizedAdvantageEstimationWithBoundaries {n : Nat} (gamma lam : α)
    (rewards values nextValues : Tensor α [n])
    (terminated boundaries : Tensor Bool [n]) : Tensor α [n] :=
  let indices : Tensor (Fin n) [n] := Tensor.ofFn id
  Tensor.scanr (fun i nextAdvantage =>
    let bootstrapMask := Core.continueMask (α := α) terminated[i]
    let continuationMask := Core.continueMask (α := α) boundaries[i]
    let delta := rewards[i] + gamma * bootstrapMask * nextValues[i] - values[i]
    delta + gamma * lam * continuationMask * nextAdvantage) 0 indices

/-- Using the same mask retains the existing GAE definition, including arithmetic order. -/
private theorem generalizedAdvantageEstimationWithBoundaries_same {n : Nat} (gamma lam : α)
    (rewards values nextValues : Tensor α [n]) (dones : Tensor Bool [n]) :
    generalizedAdvantageEstimationWithBoundaries gamma lam rewards values nextValues dones dones =
      Core.generalizedAdvantageEstimation gamma lam rewards values nextValues dones := by
  rfl

end Internal

/--
Unnormalized GAE for a PPO rollout. Task termination suppresses the next-state value;
every episode boundary stops advantage continuation. Thus a truncation bootstraps from
`nextValue` before auto-reset without using rewards from the following episode.
-/
def generalizedAdvantages {obsShape : Shape} {nActions horizon : Nat}
    (gamma lam : α) (r : Rollout α obsShape nActions horizon) : Tensor α [horizon] :=
  let stepAt (index : Fin horizon) :=
    r.steps[index.val]'(by simp [r.steps_size_eq_horizon])
  let rewards : Tensor α [horizon] := Tensor.ofFn (fun index => (stepAt index).reward)
  let terminated : Tensor Bool [horizon] := Tensor.ofFn (fun index => (stepAt index).terminated)
  let boundaries : Tensor Bool [horizon] := Tensor.ofFn (fun index => (stepAt index).done)
  let values : Tensor α [horizon] := Tensor.ofFn (fun index => (stepAt index).value)
  let nextValues : Tensor α [horizon] := Tensor.ofFn (fun index => (stepAt index).nextValue)
  Internal.generalizedAdvantageEstimationWithBoundaries gamma lam rewards values nextValues
    terminated boundaries

/--
Convert a fixed-horizon rollout into the PPO minibatch expected by
`Autograd.ppoActorCriticObjectiveDef`.

Notes:

- Advantages are normalized (z-score) for the policy-gradient term, a common PPO
  variance-reduction practice.
  Value targets (lambda-returns) are computed from the *unnormalized* advantages.
- Termination suppresses bootstrapping; truncation retains the pre-reset next-state value.
  Both stop advantage continuation across the episode boundary.
-/
def trainingBatch {obsShape : Shape} {nActions horizon : Nat}
    [NeZero horizon] [NeZero nActions]
    (gamma lam : α)
    (r : Rollout α obsShape nActions horizon) :
    IO (TrainingBatch α obsShape nActions horizon) := do
  let stepAt (index : Fin horizon) :=
    r.steps[index.val]'(by simp [r.steps_size_eq_horizon])
  let states : Tensor α (StateBatchShape horizon obsShape) :=
    Tensor.stackLeading (fun index => (stepAt index).state)
  let actionsOneHot : Tensor α (LogitsBatchShape horizon nActions) :=
    Tensor.stackLeading (fun index => Tensor.oneHot (α := α) nActions (stepAt index).action)
  let oldLogProb : Tensor α (ScalarBatchShape horizon) :=
    Tensor.ofFn (fun index => (stepAt index).oldLogProb)
  let values : Tensor α [horizon] := Tensor.ofFn (fun index => (stepAt index).value)

  let advRaw := generalizedAdvantages gamma lam r
  let returns := Core.returnsFromAdvantages (α := α) (n := horizon) advRaw values
  let normalizedAdvantages := Spec.normalizeZscoreSpec (α := α) (n := horizon) advRaw

  let valueTargets : Tensor α (ValueBatchShape horizon) :=
    Tensor.reshapeSpec returns (by simp [Shape.size])
  let advantages : Tensor α (ScalarBatchShape horizon) := normalizedAdvantages

  pure { states, actionsOneHot, oldLogProb, advantages, valueTargets }

end Rollout

end PPO
end RL
end Runtime
