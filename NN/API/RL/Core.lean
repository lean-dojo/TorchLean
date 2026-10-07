/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.API.Runtime -- shake: keep
public import NN.Spec.RL.FiniteStochasticMDP -- shake: keep
public import NN.Spec.RL.MarkovMDP -- shake: keep
public import NN.Runtime.RL.Algorithms -- shake: keep
public import NN.Runtime.RL.DQN.Autograd -- shake: keep
public import NN.Runtime.RL.Eval -- shake: keep
public import NN.Runtime.RL.PolicyGradient.Autograd -- shake: keep
public import NN.Runtime.RL.Replay -- shake: keep
public import NN.Runtime.Training.Log -- shake: keep

/-!
# Reinforcement Learning

Mathematical definitions and executable algorithms exposed under `TorchLean.rl`.

References (background and terminology):
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.):
  http://incompleteideas.net/book/the-book-2nd.html
- Puterman, *Markov Decision Processes* (finite discounted MDPs):
  https://doi.org/10.1002/9780470316887
- Gymnasium API reference (reset/step, `terminated` vs `truncated`):
  https://gymnasium.farama.org/
-/

@[expose] public section

namespace TorchLean
namespace rl

namespace env
export Spec.RL
  (StepResult ObservedTransition Env SafeEnv
   reset stepGym stateAfter evolve
   states rollout)
export Spec.RL.StepResult (done)
export Spec.RL.SafeEnv (actionPathOk)
end env

namespace core
export Spec.RL
  (continueMask discountedBackup tdResidual)
export Runtime.RL.Core
  (Transition
   discountedReturns discountedReturnsDone
   generalizedAdvantageEstimation returnsFromAdvantages
   squaredError huberLoss)
end core

namespace mdp
export Spec.RL
  (ValueFunction Policy FiniteMDP
   valueAt stateActionValue actionValues
   bellmanPolicy bellmanOptimality)
export Spec.RL.FiniteMDP (toEnv)
end mdp

namespace finiteStochastic
export Spec.RL.FiniteStochastic
  (MDP Valid
   expectedNextValue actionValue actionValues
   bellmanPolicy bellmanOptimality)
end finiteStochastic

/-! The measure-theoretic MDP from `NN.Spec.RL.MarkovMDP`: a general state space whose transitions
are probability kernels, with the expected next value as a Bochner integral. -/
namespace markov
export Spec.RL.Markov
  (ValueFunction Policy MDP Valid
   transitionMeasure
   expectedNextValue actionValue
   bellmanPolicy bellmanOptimality)
end markov

namespace bandits
/-!
Both state types expose `init` and `update`. A value state uses sample averages:

```lean
let state : rl.bandits.ValueState Float 3 := rl.bandits.ValueState.init
let state := state.update action reward
let action := state.greedy?
```

A preference state instead learns a softmax policy: `state.update action reward stepSize`, with
`baseline := false` to omit the running-average baseline. Read its probabilities with
`state.policy`. Exploration draws for `epsilonGreedy?` are supplied by the caller.
-/
export Runtime.RL.Bandits
  (ValueState PreferenceState ucbBonus)

namespace ValueState
export Runtime.RL.Bandits.ValueState (init greedy? epsilonGreedy? update pulls ucbScores ucb?)
end ValueState

namespace PreferenceState
export Runtime.RL.Bandits.PreferenceState (init policy update)
end PreferenceState
end bandits

namespace tabular
/-!
Compute a target, then move one table entry toward it with `update`:

```lean
let target := rl.tabular.qLearningTarget q nextState reward discount
let q := rl.tabular.update q (state, action, PUnit.unit) target stepSize
```

State-value tables use `(state, PUnit.unit)` instead. The coordinate is checked against the
tensor's shape by Lean; `update` does not convert it through a runtime index list.
-/
export Runtime.RL.Tabular
  (row maximum greedy? expectation
   update sarsaTarget expectedSarsaTarget qLearningTarget doubleQTarget)
end tabular

namespace value
/-!
Keep target selection separate from the scalar error function:

```lean
let target := rl.value.dqnTarget reward discount done nextValues
let loss := rl.value.loss values action target
let robustLoss := rl.value.loss values action target
  (error := fun prediction target => rl.core.huberLoss prediction target threshold)
```

For a temporal-difference residual, subtract `Tensor.getScalar values action` from the target.
DDPG's critic uses `rl.core.discountedBackup`; its actor minimizes the negative critic value.
-/
export Runtime.RL.ValueLearning
  (maximum loss
   dqnTarget doubleDqnTarget
   td3Target
   sacTarget sacActorObjective)
end value

namespace replay
export Runtime.RL.Replay (Transition Buffer ofObservedTransition)
export Runtime.RL.Replay.Buffer
  (empty size isEmpty isFull push pushMany getModulo? sampleContiguous sampleRandom)
end replay

namespace dqn
export Runtime.RL.DQN
  (loss updateTarget)

namespace autograd
/-!
Differentiable DQN losses over TorchLean backend references.

These helpers build scalar semi-gradient losses for eager or typed graph autograd. Targets and
action indicators are detached; the selected online Q values receive the loss gradient.

`huber prediction target` detaches Bellman targets before reducing the elementwise loss.
`loss qValues actions target` also detaches the one-hot actions and averages over transitions.
-/
export Runtime.RL.DQN.Autograd (huber loss)
end autograd
end dqn

namespace policy
/-!
`probabilities` applies softmax. `probability` and `logProbability` guard a selected probability
with epsilon; `logSoftmax` instead selects an unclamped log-softmax entry. `entropy` uses guarded
weights without renormalizing, whereas `kl` clamps and normalizes both probability vectors.

`sample seed counter values` consumes probability weights. Set `logits := true` to apply softmax
first. Both modes return the next counter and chosen action; supplied weights are not validated.
For TRPO's unpenalized surrogate, use `ratio * advantage`; the trust-region constraint is separate.
-/
export Runtime.RL.PolicyGradient
  (probabilities probability logProbability logSoftmax entropy
   reinforceLoss criticLoss actorCriticLoss
   ratio kl klFromLogits
   klLoss sacActorLoss)
export Runtime.RL.PolicyGradient
  (sample)

namespace autograd
/-!
Differentiable policy-gradient losses over TorchLean backend references.

The pure exports above are algebra over concrete spec tensors. These helpers are the training-time
counterpart: they build scalar losses from backend refs, so the same formulas can run through eager
or typed graph autograd.
-/
export Runtime.RL.PolicyGradient.Autograd
  (logProbability entropy)
end autograd
end policy

namespace ppo
/-!
`objective ratio advantage clipEps` is the clipped surrogate to maximize. When starting from
policy logits and a cached old log-probability, use `objectiveFromLogits`; it applies the guarded
selected probability from `policy.logProbability`. `loss` negates that objective and adds value
regression and entropy terms. These pure functions do not construct a differentiation tape.
-/
export Runtime.RL.PolicyGradient.PPO (objective objectiveFromLogits loss)

namespace autograd
/-!
`objective` records the per-sample clipped surrogate from logits and one-hot actions; `loss`
records the mean policy, value and entropy loss. Both use `policy.autograd.logProbability`'s
finite log-softmax guard. `create actor critic` bundles the two models into an objective definition
for a PPO batch. The pure `ppo.loss` above instead operates on concrete tensors for one action.
-/
export Runtime.RL.PolicyGradient.Autograd.PPO (objective loss create)
end autograd
end ppo

namespace eval
/-!
`totalReward` evaluates one greedy-policy episode without recording its history. `meanReward`
averages these episode totals, not individual step rewards. `path` instead records the checked
session states, including the initial state; those states can contain nonnumeric environment data.
-/
export Runtime.RL.Eval
  (greedy totalReward path meanReward)
end eval

end rl
end TorchLean
