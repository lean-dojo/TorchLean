/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.BatchInput
public import NN.Runtime.RL.Algorithms.ValueLearning

/-!
# DQN Minibatch Helpers

`NN.Runtime.RL.Algorithms.ValueLearning` contains the scalar DQN/Double-DQN targets. This module
adds the missing batch-facing layer used by replay-buffer training loops:

- evaluate one transition with caller-provided online/target Q-functions;
- average DQN or Double-DQN losses as a tensor over a replay minibatch;
- update target-network parameters by a weighted average.

The functions are intentionally higher-order: TorchLean examples can pass typed-graph/eager model
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

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- DQN loss for one transition, or the mean loss of a replay batch (`batch := true`).

`double := true` selects actions with the online network and evaluates them with the target
network. `error` defaults to squared error; pass a scalar loss such as `Core.huberLoss` to change
it. Batch reduction uses `Tensor.mean`, including its empty-batch convention.
-/
def loss {obsShape : Shape} {nActions : Nat}
    {Input : Type}
    (onlineQ targetQ : Tensor α obsShape → Tensor α [nActions]) (gamma : α)
    (input : Input) (batch : Bool := false)
    (double : Bool := false)
    (error : α → α → α := Core.squaredError)
    [TorchLean.Internal.BatchInput (Core.Transition α obsShape nActions)
      (Array (Core.Transition α obsShape nActions)) batch Input] : α := by
  have inputType := TorchLean.Internal.BatchInput.type_eq
    (single := Core.Transition α obsShape nActions)
    (many := Array (Core.Transition α obsShape nActions)) (batch := batch)
  subst Input
  let evaluate := fun tr : Core.Transition α obsShape nActions =>
    let prediction := Tensor.getScalar (onlineQ tr.state) tr.action
    let target := if double then
        ValueLearning.doubleDqnTarget tr.reward gamma tr.done
          (onlineQ tr.nextState) (targetQ tr.nextState)
      else ValueLearning.dqnTarget tr.reward gamma tr.done (targetQ tr.nextState)
    error prediction target
  cases batch with
  | false => exact evaluate input
  | true =>
      let inputs : Array (Core.Transition α obsShape nActions) := input
      exact (Tensor.ofFn fun index : Fin inputs.size => evaluate inputs[index]).mean

/--
Weighted target-network update:

`target ← τ * online + (1 - τ) * target`.

Use this elementwise over parameter tensors for DQN/DDPG/TD3/SAC target synchronization.
The mixing weight is used as supplied, without validation. The arithmetic order is retained
for floating-point callers; algebraically equivalent rearrangements can round differently.
-/
def updateTarget {α : Type*} [One α] [Add α] [Sub α] [Mul α]
    (tau online target : α) : α :=
  tau * online + ((1 : α) - tau) * target

end DQN
end RL
end Runtime
