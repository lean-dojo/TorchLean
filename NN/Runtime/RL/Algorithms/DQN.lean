/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Replay
public import NN.Runtime.RL.Algorithms.ValueLearning

/-!
# DQN Minibatch Helpers

`NN.Runtime.RL.Algorithms.ValueLearning` contains the scalar DQN/Double-DQN targets. This module
adds the missing batch-facing layer used by replay-buffer training loops:

- evaluate one transition with caller-provided online/target Q-functions;
- average DQN or Double-DQN losses over an `Array` minibatch;
- soft-update scalar parameters for target networks.

The functions are intentionally higher-order: TorchLean examples can pass compiled/eager model
closures without this module knowing anything about parameters, optimizers, or autograd sessions.

References:
- Mnih et al., "Human-level control through deep reinforcement learning" (2015):
  https://doi.org/10.1038/nature14236
- van Hasselt, Guez, and Silver, "Deep Reinforcement Learning with Double Q-learning" (2016):
  https://arxiv.org/abs/1509.06461
- Polyak and Juditsky, "Acceleration of Stochastic Approximation by Averaging" (1992), background
  for moving-average target-network updates.
-/

@[expose] public section

namespace Runtime
namespace RL
namespace DQN

open Spec
open Tensor

variable {α : Type} [Context α]

/-- Average an array of scalar losses, returning `0` for an empty minibatch. -/
def meanArray (xs : Array α) : α :=
  if xs.isEmpty then
    0
  else
    xs.foldl (fun acc x => acc + x) 0 / (xs.size : α)

/-- One-transition DQN squared TD loss from online and target Q-functions. -/
def transitionMSELoss {obsShape : Shape} {nActions : Nat}
    (onlineQ targetQ : Tensor α obsShape → Tensor α (.dim nActions .scalar))
    (gamma : α)
    (tr : Core.Transition α obsShape nActions) : α :=
  ValueLearning.dqnMSELoss (α := α)
    (qPred := onlineQ tr.state)
    (action := tr.action)
    (reward := tr.reward)
    (gamma := gamma)
    (done := tr.done)
    (nextQTarget := targetQ tr.nextState)

/-- One-transition DQN Huber TD loss from online and target Q-functions. -/
def transitionHuberLoss {obsShape : Shape} {nActions : Nat}
    (onlineQ targetQ : Tensor α obsShape → Tensor α (.dim nActions .scalar))
    (gamma : α) (delta : α := 1)
    (tr : Core.Transition α obsShape nActions) : α :=
  ValueLearning.dqnHuberLoss (α := α)
    (qPred := onlineQ tr.state)
    (action := tr.action)
    (reward := tr.reward)
    (gamma := gamma)
    (done := tr.done)
    (nextQTarget := targetQ tr.nextState)
    (delta := delta)

/-- One-transition Double-DQN Huber TD loss. -/
def transitionDoubleHuberLoss {obsShape : Shape} {nActions : Nat}
    (onlineQ targetQ : Tensor α obsShape → Tensor α (.dim nActions .scalar))
    (gamma : α) (delta : α := 1)
    (tr : Core.Transition α obsShape nActions) : α :=
  let qPred := onlineQ tr.state
  let nextOnline := onlineQ tr.nextState
  let nextTarget := targetQ tr.nextState
  let target := ValueLearning.doubleDqnTarget (α := α)
    (reward := tr.reward) (gamma := gamma) (done := tr.done)
    (nextQOnline := nextOnline) (nextQTarget := nextTarget)
  Core.huberLoss (α := α) (ValueLearning.chosenActionValue qPred tr.action) target delta

/-- Mean DQN squared TD loss over a replay minibatch. -/
def minibatchMSELoss {obsShape : Shape} {nActions : Nat}
    (onlineQ targetQ : Tensor α obsShape → Tensor α (.dim nActions .scalar))
    (gamma : α)
    (batch : Array (Core.Transition α obsShape nActions)) : α :=
  meanArray (α := α) <| batch.map (transitionMSELoss (α := α) onlineQ targetQ gamma)

/-- Mean DQN Huber TD loss over a replay minibatch. -/
def minibatchHuberLoss {obsShape : Shape} {nActions : Nat}
    (onlineQ targetQ : Tensor α obsShape → Tensor α (.dim nActions .scalar))
    (gamma : α) (delta : α := 1)
    (batch : Array (Core.Transition α obsShape nActions)) : α :=
  meanArray (α := α) <| batch.map (transitionHuberLoss (α := α) onlineQ targetQ gamma delta)

/-- Mean Double-DQN Huber TD loss over a replay minibatch. -/
def minibatchDoubleHuberLoss {obsShape : Shape} {nActions : Nat}
    (onlineQ targetQ : Tensor α obsShape → Tensor α (.dim nActions .scalar))
    (gamma : α) (delta : α := 1)
    (batch : Array (Core.Transition α obsShape nActions)) : α :=
  meanArray (α := α) <| batch.map (transitionDoubleHuberLoss (α := α) onlineQ targetQ gamma delta)

/--
Soft target-network update for a single scalar:

`target ← τ * online + (1 - τ) * target`.

Use this elementwise over parameter tensors/lists when implementing DQN/DDPG/TD3/SAC target sync.
-/
def softUpdateScalar (tau online target : α) : α :=
  tau * online + ((1 : α) - tau) * target

end DQN
end RL
end Runtime
