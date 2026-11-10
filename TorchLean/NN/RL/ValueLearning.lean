/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Transition
public import TorchLean.Tensor.Layers.Activation

/-!
# Deep Value-Learning Objectives

The scalar targets and objectives behind DQN, Double DQN, DDPG, TD3 and SAC. Compute a bootstrap
target, then pass it to `loss` with the selected action. The targets go through
`TorchLean.RL.discountedBackup`, so a terminal step (`done`) gives the reward alone. DDPG needs no
function of its own: its critic uses the ordinary `TorchLean.RL.discountedBackup` and its actor
minimizes the negated critic value. A TD residual is the target minus the selected tensor entry.
Replay is `TorchLean.NN.RL.Replay`, target-network synchronization is
`RL.DQN.updateTarget`, and optimizer steps belong to the training loops.

Each target's `done` argument is the termination flag, Gymnasium's `terminated`. A time-limit
truncation passes `false`, since the state after it has a value and the target bootstraps from it.

We wrote every target by hand from its paper. `dqnTarget` is $y_j$ of Algorithm 1 in Mnih et al.,
and `doubleDQNTarget` is the target of the Double DQN section of van Hasselt, Guez and Silver.
The DDPG critic target is eq. (5) of Lillicrap et al., evaluated with the target networks of
their Algorithm 1. `td3Target` is eq. (10) of Fujimoto et al. The caller passes both critics at
the next action, so the clipped smoothing noise of their eq. (15) is the caller's choice.

SAC comes from the second Haarnoja et al. paper. Its eq. (6) writes the temperature $\alpha$ into
the target and eq. (7) gives the actor objective. Section 6 takes the smaller of two critics in
eq. (6) and in eq. (10), the gradient of eq. (7) after reparameterization. The first paper folds
$\alpha$ into the reward scale and fits a separate value network.
No theorem or TorchLeanTest test checks these targets.

Primary references:

- Mnih et al., "Human-level control through deep reinforcement learning" (2015), Nature
  518(7540):529-533, Methods, Algorithm 1: https://doi.org/10.1038/nature14236
- van Hasselt, Guez, and Silver, "Deep Reinforcement Learning with Double Q-learning" (2016),
  AAAI 2016, section "Double DQN": https://arxiv.org/abs/1509.06461
- Lillicrap et al., "Continuous Control with Deep Reinforcement Learning" (2015), ICLR 2016,
  eq. (5) and Algorithm 1: https://arxiv.org/abs/1509.02971
- Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (2018),
  ICML 2018, eqs. (10) and (15) and Algorithm 1: https://arxiv.org/abs/1802.09477
- Haarnoja, Zhou, Abbeel and Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
  Reinforcement Learning with a Stochastic Actor" (2018), ICML 2018:
  https://arxiv.org/abs/1801.01290
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2018), arXiv:1812.05905,
  eqs. (6) and (7) and Section 6: https://arxiv.org/abs/1812.05905
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace ValueLearning

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- Maximum Q-value in a vector, `Softmax.maxVec` (a fold of the IEEE 754-2019 `Context.maximum`,
so a NaN entry gives NaN and `+0` is above `-0`), and `0` when `nActions = 0`. -/
def maximum {nActions : Nat} (qValues : Tensor α [nActions]) : α :=
  match nActions, qValues with
  | 0, _ => 0
  | _ + 1, qValues => (Softmax.maxVec qValues).item

/-- DQN bootstrap target `r + γ max_a Q_target(s', a)`. -/
def dqnTarget {nActions : Nat} (reward gamma : α) (done : Bool)
    (nextQTarget : Tensor α [nActions]) : α :=
  RL.discountedBackup (α := α) reward gamma (maximum (α := α) nextQTarget) done

/-- Double DQN target: select with the online network, evaluate with the target network. With
no actions the target is `reward`. At a tie `Tensor.argmax` keeps the first index, as
`torch.argmax` does, and the first NaN online value wins the pick. -/
def doubleDQNTarget {nActions : Nat} (reward gamma : α) (done : Bool)
    (nextQOnline nextQTarget : Tensor α [nActions]) : α :=
  if h : 0 < nActions then
    RL.discountedBackup (α := α) reward gamma
      (Tensor.getScalar nextQTarget (Tensor.argmax nextQOnline ⟨0, Nat.one_pos⟩ false h).item) done
  else reward

/-- Loss of the selected action's value against a supplied target, `error q[action] target`.

Compute the target separately with `dqnTarget`, `doubleDQNTarget` or another bootstrap rule.
`error` defaults to squared error, and Huber loss or another scalar objective can take its place.
-/
def loss {nActions : Nat} (q : Tensor α [nActions]) (action : Fin nActions)
    (target : α) (error : α → α → α := squaredError) : α :=
  error (Tensor.getScalar q action) target

/-- TD3 clipped-double target using `Context.minimum` of the two target critics, so a NaN critic
makes the bootstrap value NaN and `-0` orders below `+0`. -/
def td3Target (reward gamma nextCritic1 nextCritic2 : α) (done : Bool := false) : α :=
  RL.discountedBackup (α := α) reward gamma (Context.minimum nextCritic1 nextCritic2) done

/-- SAC entropy-regularized soft target `r + γ (min(Q1', Q2') - α * log π(a'|s'))`, with the
temperature `α`, `logProb = log π(a'|s')` and the minimum taken by `Context.minimum`. -/
def sacTarget (reward gamma nextCritic1 nextCritic2 logProb temperature : α)
    (done : Bool := false) : α :=
  let softBootstrap := Context.minimum nextCritic1 nextCritic2 - temperature * logProb
  RL.discountedBackup (α := α) reward gamma softBootstrap done

/-- SAC actor objective, minimized: `α * log π(a|s) - min(Q1, Q2)` with temperature `α` and the
minimum taken by `Context.minimum`. -/
def sacActorObjective (critic1 critic2 logProb temperature : α) : α :=
  temperature * logProb - Context.minimum critic1 critic2

end ValueLearning
end RL
end TorchLean
