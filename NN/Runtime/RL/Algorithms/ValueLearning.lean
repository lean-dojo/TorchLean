/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Core
public import NN.Runtime.Autograd.Model.Metrics

/-!
# Deep Value-Learning Objectives

This module packages the core scalar objectives / targets behind common deep RL algorithms:

- DQN and Double DQN,
- DDPG-style actor / critic objectives,
- TD3 clipped double critics,
- SAC entropy-regularized targets and actor objectives.

Compute a bootstrap target, then pass it to `loss` with the selected action. DDPG uses the ordinary
`Core.discountedBackup` for its critic and negates the critic value for its actor. A TD residual
is the target minus the selected tensor entry. Replay, target-network synchronization and optimizer
orchestration live in higher-level modules.

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

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Maximum Q-value in a vector, defaulting to `0` when `nActions = 0`. -/
def maximum {nActions : Nat} (qValues : Tensor α [nActions]) : α :=
  match TorchLean.Metrics.argmax? (α := α) qValues with
  | some action => Tensor.getScalar qValues (Fin.cast (by simp [Shape.size]) action)
  | none => 0

/-- DQN bootstrap target `r + γ max_a Q_target(s', a)`. -/
def dqnTarget {nActions : Nat} (reward gamma : α) (done : Bool)
    (nextQTarget : Tensor α [nActions]) : α :=
  Core.discountedBackup (α := α) reward gamma (maximum (α := α) nextQTarget) done

/-- Double DQN target:
select with the online network, evaluate with the target network. -/
def doubleDqnTarget {nActions : Nat} (reward gamma : α) (done : Bool)
    (nextQOnline nextQTarget : Tensor α [nActions]) : α :=
  match TorchLean.Metrics.argmax? (α := α) nextQOnline with
  | some action =>
      Core.discountedBackup (α := α) reward gamma
        (Tensor.getScalar nextQTarget (Fin.cast (by simp [Shape.size]) action)) done
  | none => reward

/-- Loss for a selected action against a supplied target.

Compute the target separately with `dqnTarget`, `doubleDqnTarget`, or another bootstrap rule.
The default is squared error; pass `error` to use Huber loss or another scalar objective.
-/
def loss {nActions : Nat} (q : Tensor α [nActions]) (action : Fin nActions)
    (target : α) (error : α → α → α := Core.squaredError) : α :=
  error (Tensor.getScalar q action) target

/-- TD3 clipped-double target using the minimum of the two target critics. -/
def td3Target (reward gamma nextCritic1 nextCritic2 : α) (done : Bool := false) : α :=
  Core.discountedBackup (α := α) reward gamma (Min.min nextCritic1 nextCritic2) done

/-- SAC entropy-regularized soft target:
`r + γ (min(Q1', Q2') - α * log π(a'|s'))`. -/
def sacTarget (reward gamma nextCritic1 nextCritic2 logProb temperature : α)
    (done : Bool := false) : α :=
  let softBootstrap := Min.min nextCritic1 nextCritic2 - temperature * logProb
  Core.discountedBackup (α := α) reward gamma softBootstrap done

/-- SAC actor objective:
minimize `α * log π(a|s) - min(Q1, Q2)`. -/
def sacActorObjective (critic1 critic2 logProb temperature : α) : α :=
  temperature * logProb - Min.min critic1 critic2

end ValueLearning
end RL
end Runtime
