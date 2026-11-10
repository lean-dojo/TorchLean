/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.BatchInput
public import TorchLean.NN.RL.ValueLearning

/-!
# DQN Minibatch Helpers

`loss` is the DQN or Double-DQN loss of one transition or the mean loss of a replay minibatch, at
the bootstrap targets of `ValueLearning`. `updateTarget` is the weighted target-network update.
The online and target Q-functions are functions on tensors, so a caller passes any model closure.

We wrote `loss` by hand from Algorithm 1 of Mnih et al.: a transition contributes
`error (Q(s, a)) y`, and a minibatch takes the mean. Their Methods clip the error term to
$[-1, 1]$, which is the gradient of `TorchLean.Huber.term` at its default `delta = 1`, so passing
that as `error` matches the paper. `updateTarget` is the soft update
$\theta' \leftarrow \tau\theta + (1-\tau)\theta'$ from Section 3 of Lillicrap et al., who take
$\tau = 0.001$ in their Section 7. The hard copy of Mnih et al. is `tau = 1` once every $C$
updates. `DQN.updateTarget_one` and `DQN.updateTarget_sub_target` prove the copy and the gap
identity over `ℝ`. No test checks `loss`.

References:
- Mnih et al., "Human-level control through deep reinforcement learning" (2015), Nature
  518(7540):529-533, Methods, Algorithm 1: https://doi.org/10.1038/nature14236
- van Hasselt, Guez, and Silver, "Deep Reinforcement Learning with Double Q-learning" (2016),
  AAAI 2016: https://arxiv.org/abs/1509.06461
- Lillicrap et al., "Continuous Control with Deep Reinforcement Learning" (2015), ICLR 2016,
  Sections 3 and 7: https://arxiv.org/abs/1509.02971
- Polyak and Juditsky, "Acceleration of Stochastic Approximation by Averaging" (1992), SIAM
  Journal on Control and Optimization 30(4):838-855, background for moving-average
  target-network updates: https://doi.org/10.1137/0330046
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace DQN

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- `ValueLearning.loss` of the online Q-values at the DQN target of one transition, or its mean
over a replay batch (`batch := true`): `Tensor.sum` divided by `Tensor.meanDenominator`, which is
1 for an empty batch.

`double := true` takes the Double-DQN target, which selects the next action with the online network
and evaluates it with the target network. `error` is squared error unless given, such as
`TorchLean.Huber.term`.
-/
def loss {obsShape : Shape} {nActions : Nat}
    {Input : Type}
    (onlineQ targetQ : Tensor α obsShape → Tensor α [nActions]) (gamma : α)
    (input : Input) (batch : Bool := false)
    (double : Bool := false)
    (error : α → α → α := squaredError)
    [BatchInput (Transition α obsShape nActions)
      (Array (Transition α obsShape nActions)) batch Input] : α := by
  have inputType := BatchInput.type_eq
    (single := Transition α obsShape nActions)
    (many := Array (Transition α obsShape nActions)) (batch := batch)
  subst Input
  let evaluate := fun tr : Transition α obsShape nActions =>
    let target := if double then
        ValueLearning.doubleDQNTarget tr.reward gamma tr.done
          (onlineQ tr.nextState) (targetQ tr.nextState)
      else ValueLearning.dqnTarget tr.reward gamma tr.done (targetQ tr.nextState)
    ValueLearning.loss (onlineQ tr.state) tr.action target error
  cases batch with
  | false => exact evaluate input
  | true =>
      let inputs : Array (Transition α obsShape nActions) := input
      exact Tensor.sum (Tensor.ofFn fun index : Fin inputs.size => evaluate inputs[index]) /
        (Tensor.meanDenominator [inputs.size] : α)

/--
Weighted target-network update `tau * online + (1 - tau) * target`, for elementwise use on the
parameters of a DQN, DDPG, TD3 or SAC target network. It is evaluated in this order with `tau` as
supplied, without validation: algebraically equal rearrangements can round differently.
-/
def updateTarget {α : Type*} [One α] [Add α] [Sub α] [Mul α]
    (tau online target : α) : α :=
  tau * online + ((1 : α) - tau) * target

end DQN
end RL
end TorchLean
