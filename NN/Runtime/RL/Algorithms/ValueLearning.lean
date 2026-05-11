/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Core
public import NN.Runtime.Autograd.TorchLean.Metrics

/-!
# Deep Value-Learning Objectives

This module packages the core scalar objectives / targets behind common deep RL algorithms:

- DQN and Double DQN,
- DDPG-style actor / critic objectives,
- TD3 clipped double critics,
- SAC entropy-regularized targets and actor objectives.

The functions are intentionally small and typed. They expose the textbook math while leaving
experience replay, target-network sync, and optimizer orchestration to higher-level code.

Primary references:

- Mnih et al., "Human-level control through deep reinforcement learning" (2015):
  https://doi.org/10.1038/nature14236
- van Hasselt, Guez, and Silver, "Deep Reinforcement Learning with Double Q-learning" (2016):
  https://arxiv.org/abs/1509.06461
- Lillicrap et al., "Continuous Control with Deep Reinforcement Learning" (2015):
  https://arxiv.org/abs/1509.02971
- Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (2018):
  https://arxiv.org/abs/1802.09477
- Haarnoja et al., "Soft Actor-Critic" (2018): https://arxiv.org/abs/1801.01290
-/

@[expose] public section

namespace Runtime
namespace RL
namespace ValueLearning

open Spec
open Tensor

variable {α : Type} [Context α]

/-- Extract `Q(s, a)` from a vector of action-values. -/
def chosenActionValue {nActions : Nat} (qValues : Tensor α (.dim nActions .scalar))
    (action : Fin nActions) : α :=
  Tensor.vecGet qValues action

/-- Maximum Q-value in a vector, defaulting to `0` when `nActions = 0`. -/
def maxQValue {nActions : Nat} (qValues : Tensor α (.dim nActions .scalar)) : α :=
  match Runtime.Autograd.TorchLean.Metrics.argmax? (α := α) (n := nActions) qValues with
  | some action => Tensor.vecGet qValues action
  | none => 0

/-- DQN bootstrap target `r + γ max_a Q_target(s', a)`. -/
def dqnTarget {nActions : Nat} (reward gamma : α) (done : Bool)
    (nextQTarget : Tensor α (.dim nActions .scalar)) : α :=
  Core.tdTarget (α := α) reward gamma (maxQValue (α := α) nextQTarget) done

/-- Double DQN target:
select with the online network, evaluate with the target network. -/
def doubleDqnTarget {nActions : Nat} (reward gamma : α) (done : Bool)
    (nextQOnline nextQTarget : Tensor α (.dim nActions .scalar)) : α :=
  match Runtime.Autograd.TorchLean.Metrics.argmax? (α := α) (n := nActions) nextQOnline with
  | some action => Core.tdTarget (α := α) reward gamma (Tensor.vecGet nextQTarget action) done
  | none => reward

/-- DQN temporal-difference residual for one sampled action. -/
def dqnResidual {nActions : Nat} (qPred : Tensor α (.dim nActions .scalar)) (action : Fin nActions)
    (reward gamma : α) (done : Bool) (nextQTarget : Tensor α (.dim nActions .scalar)) : α :=
  let target := dqnTarget (α := α) reward gamma done nextQTarget
  target - chosenActionValue (α := α) qPred action

/-- Squared TD loss for DQN. -/
def dqnMSELoss {nActions : Nat} (qPred : Tensor α (.dim nActions .scalar)) (action : Fin nActions)
    (reward gamma : α) (done : Bool) (nextQTarget : Tensor α (.dim nActions .scalar)) : α :=
  let target := dqnTarget (α := α) reward gamma done nextQTarget
  Core.squaredError (α := α) (chosenActionValue qPred action) target

/-- Huber TD loss for DQN. -/
def dqnHuberLoss {nActions : Nat} (qPred : Tensor α (.dim nActions .scalar)) (action : Fin nActions)
    (reward gamma : α) (done : Bool) (nextQTarget : Tensor α (.dim nActions .scalar))
    (delta : α := 1) : α :=
  let target := dqnTarget (α := α) reward gamma done nextQTarget
  Core.huberLoss (α := α) (chosenActionValue qPred action) target delta

/-- Double-DQN temporal-difference residual. -/
def doubleDqnResidual {nActions : Nat} (qPred : Tensor α (.dim nActions .scalar))
    (action : Fin nActions) (reward gamma : α) (done : Bool)
    (nextQOnline nextQTarget : Tensor α (.dim nActions .scalar)) : α :=
  let target := doubleDqnTarget (α := α) reward gamma done nextQOnline nextQTarget
  target - chosenActionValue (α := α) qPred action

/-- Deterministic-policy-gradient actor objective used by DDPG:
maximize `Q(s, μ(s))`, or equivalently minimize `-Q(s, μ(s))`. -/
def ddpgActorObjective (criticValue : α) : α :=
  -criticValue

/-- DDPG critic target `r + γ Q_target(s', μ_target(s'))`. -/
def ddpgCriticTarget (reward gamma nextCriticValue : α) (done : Bool := false) : α :=
  Core.tdTarget (α := α) reward gamma nextCriticValue done

/-- TD3 clipped-double target using the minimum of the two target critics. -/
def td3Target (reward gamma nextCritic1 nextCritic2 : α) (done : Bool := false) : α :=
  Core.tdTarget (α := α) reward gamma (Min.min nextCritic1 nextCritic2) done

/-- SAC entropy-regularized soft target:
`r + γ (min(Q1', Q2') - α * log π(a'|s'))`. -/
def sacTarget (reward gamma nextCritic1 nextCritic2 logProb temperature : α)
    (done : Bool := false) : α :=
  let softBootstrap := Min.min nextCritic1 nextCritic2 - temperature * logProb
  Core.tdTarget (α := α) reward gamma softBootstrap done

/-- SAC actor objective:
minimize `α * log π(a|s) - min(Q1, Q2)`. -/
def sacActorObjective (critic1 critic2 logProb temperature : α) : α :=
  temperature * logProb - Min.min critic1 critic2

end ValueLearning
end RL
end Runtime
