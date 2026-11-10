/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Transition
public import TorchLean.Tensor.LinearAlgebra.Algorithms
public import TorchLean.Tensor.Pack
public import TorchLean.Tensor.Operations

/-!
# PPO Rollouts (Discrete Actions)

Fixed-horizon PPO rollout records, stored as typed tensors and arrays, and their conversion to the
minibatch format of the PPO autograd objective (`TorchLean.RL.PPO.Autograd.create`).
The GAE and return definitions are in `TorchLean.NN.RL`, and the rollout passes task
termination and episode boundaries to that GAE as two separate masks.

We wrote the rollout code by hand from Section 5 of the PPO paper. Its truncated GAE, eqs. (11)
and (12), is eq. (16) of the GAE paper cut off at the horizon. `Rollout.generalizedAdvantages`
computes it from the right, so the last step's advantage is its residual $\delta_t$, which
bootstraps from that step's `nextValue` unless the step terminated. The value targets of
`Rollout.trainingBatch` add the unnormalized advantages back to the values. That's the
TD($\lambda$) target of footnote 2 in the GAE paper, whose main text fits the discounted return of
its eq. (28). Neither paper normalizes advantages.
We z-score them with `Tensor.normalizeZscore`, which divides by the population standard deviation
with no epsilon, and a batch whose centered advantages are all 0 goes through as zeros.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017), arXiv:1707.06347, Section 5,
  eqs. (11) and (12): https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015), ICLR 2016, eq. (16), Section 5 and its footnote 2: https://arxiv.org/abs/1506.02438
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace PPO

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-!
## Shapes

For a fixed horizon `T`, PPO minibatches are typically stored in "PyTorch-shaped" tensors:

- `states : (T × obsShape)`
- `actions : (T)`, sampled action indices as `Fin nActions` data
- `oldLogProb : (T)`
- `advantages : (T)`
- `valueTargets : (T × 1)`
-/

/-- Batch shape for a fixed-horizon sequence of observations: `horizon × obsShape`. -/
abbrev stateBatchShape (horizon : Nat) (obsShape : Shape) : Shape :=
  Shape.dim horizon obsShape

/-- Batch shape for a fixed-horizon sequence of action logits: `horizon × nActions`. -/
abbrev logitsBatchShape (horizon nActions : Nat) : Shape :=
  [horizon, nActions]

/-- Batch shape for a fixed-horizon sequence of scalars: `horizon`. -/
abbrev scalarBatchShape (horizon : Nat) : Shape :=
  [horizon]

/-- Batch shape for a fixed-horizon sequence of scalar values stored as a column: `horizon × 1`. -/
abbrev valueBatchShape (horizon : Nat) : Shape :=
  [horizon, 1]

/-!
## Rollouts
-/

/--
One fixed-horizon PPO step record.

PPO implementations commonly store these fields as parallel arrays. Here they form one record per
step, so downstream code cannot desynchronize them.
-/
structure Step (α : Type) [Storage α] (obsShape : Shape) (nActions : Nat) where
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
  /-- Whether the task terminated, which suppresses value bootstrapping. An external truncation sets
  `done` but leaves this flag false. -/
  terminated : Bool

/--
Fixed-horizon rollout buffer for PPO.

The field `steps_size_eq_horizon` records that the buffer has exactly `horizon` steps, so the
conversion to tensors is total and needs no runtime bounds checks.
-/
structure Rollout (α : Type) [Storage α] (obsShape : Shape) (nActions horizon : Nat) where
  /-- The steps in collection order. -/
  steps : Array (Step α obsShape nActions)
  /-- Invariant: fixed-horizon rollouts always have exactly `horizon` steps. -/
  steps_size_eq_horizon : steps.size = horizon

/-- Named tensors consumed by one PPO actor-critic update. -/
structure TrainingBatch (α : Type) [Storage α]
    (obsShape : Shape) (nActions horizon : Nat) where
  /-- Observations for each rollout step. -/
  states : Tensor α (stateBatchShape horizon obsShape)
  /-- Sampled actions, one index per step. -/
  actions : Tensor (Fin nActions) (scalarBatchShape horizon)
  /-- Behavior-policy log-probability for each sampled action. -/
  oldLogProb : Tensor α (scalarBatchShape horizon)
  /-- Normalized generalized advantages used by the policy objective. -/
  advantages : Tensor α (scalarBatchShape horizon)
  /-- Lambda-return targets used by the value objective. -/
  valueTargets : Tensor α (valueBatchShape horizon)

namespace TrainingBatch

/-- Pack the differentiable tensors of a named PPO batch (`states`, `oldLogProb`, `advantages` and
`valueTargets`) for the low-level autograd objective, which takes the actions as data. -/
def arguments {α : Type} [Storage α]
    {obsShape : Shape} {nActions horizon : Nat}
    (batch : TrainingBatch α obsShape nActions horizon) :
    TensorPack α
      [stateBatchShape horizon obsShape,
       scalarBatchShape horizon,
       scalarBatchShape horizon,
       valueBatchShape horizon] :=
  .cons batch.states <|
    .cons batch.oldLogProb <|
      .cons batch.advantages <|
        .cons batch.valueTargets .nil

end TrainingBatch

namespace Rollout

/--
Unnormalized GAE for a PPO rollout, with `terminated` as the termination mask and `done` as the
boundary mask. Termination drops the next-state value, and every episode boundary stops the
advantage recursion, so a truncated step bootstraps from its pre-reset `nextValue` without reading
rewards of the following episode.
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
  RL.gae gamma lam rewards values nextValues terminated boundaries

/--
Convert a fixed-horizon rollout into the PPO minibatch expected by `PPO.Autograd.create`.

The policy term receives z-score normalized advantages, a common PPO variance-reduction practice,
while the value targets (lambda-returns) come from the *unnormalized* advantages. Episode
boundaries are handled as in `generalizedAdvantages`.
-/
def trainingBatch {obsShape : Shape} {nActions horizon : Nat}
    [NeZero horizon] [NeZero nActions]
    (gamma lam : α)
    (r : Rollout α obsShape nActions horizon) :
    IO (TrainingBatch α obsShape nActions horizon) := do
  let stepAt (index : Fin horizon) :=
    r.steps[index.val]'(by simp [r.steps_size_eq_horizon])
  let states : Tensor α (stateBatchShape horizon obsShape) :=
    Tensor.dim (fun index => (stepAt index).state)
  let actions : Tensor (Fin nActions) (scalarBatchShape horizon) :=
    Tensor.ofFn (fun index => (stepAt index).action)
  let oldLogProb : Tensor α (scalarBatchShape horizon) :=
    Tensor.ofFn (fun index => (stepAt index).oldLogProb)
  let values : Tensor α [horizon] := Tensor.ofFn (fun index => (stepAt index).value)

  let advRaw := generalizedAdvantages gamma lam r
  let returns := RL.returnsFromAdvantages (α := α) (n := horizon) advRaw values
  let advantages := Tensor.normalizeZscore (α := α) (n := horizon) advRaw

  let valueTargets : Tensor α (valueBatchShape horizon) :=
    Tensor.reshape returns _ (by simp)
  pure { states, actions, oldLogProb, advantages, valueTargets }

end Rollout

end PPO
end RL
end TorchLean
