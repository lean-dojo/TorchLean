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
   reset stepGym stateAfter evolve evolveFrom
   states statesFrom rollout rolloutFrom)
export Spec.RL.StepResult (done)
export Spec.RL.SafeEnv (actionPathOk)
end env

namespace core
export Spec.RL
  (continueMask discountedBackup tdTarget tdResidual)
export Runtime.RL.Core
  (Transition
   discountedReturnsFrom discountedReturns discountedReturnsDone
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
export Runtime.RL.Bandits
  (ValueState PreferenceState
   greedyAction? epsilonGreedyAction?
   sampleAverageStep totalPulls
   ucb1Bonus ucb1Scores ucb1Action?
   gradientPolicy gradientBanditStep)
export Runtime.RL.Bandits.ValueState (init)
export Runtime.RL.Bandits.PreferenceState (init)
end bandits

namespace tabular
export Runtime.RL.Tabular
  (actionRow maxActionValue greedyAction? expectedActionValue
   td0Update
   sarsaTarget expectedSarsaTarget qLearningTarget doubleQTarget
   sarsaUpdate expectedSarsaUpdate qLearningUpdate
   doubleQUpdateLeft doubleQUpdateRight)
end tabular

namespace value
export Runtime.RL.ValueLearning
  (chosenActionValue maxQValue
   dqnTarget doubleDqnTarget
   dqnResidual dqnMSELoss dqnHuberLoss doubleDqnResidual
   ddpgActorObjective ddpgCriticTarget td3Target
   sacTarget sacActorObjective)
end value

namespace replay
export Runtime.RL.Replay (Transition Buffer ofObservedTransition)
export Runtime.RL.Replay.Buffer
  (empty size isEmpty isFull push pushMany getModulo? sampleContiguous sampleRandom)
end replay

namespace dqn
export Runtime.RL.DQN
  (transitionMSELoss transitionHuberLoss transitionDoubleHuberLoss
   minibatchMSELoss minibatchHuberLoss minibatchDoubleHuberLoss
   softUpdateScalar)

namespace autograd
/-!
Differentiable DQN losses over TorchLean backend references.

These helpers build scalar semi-gradient losses for eager or typed graph autograd. Targets and
action indicators are detached; the selected online Q values receive the loss gradient.
-/
export Runtime.RL.DQN.Autograd (huberTDLoss actionHuberLoss)
end autograd
end dqn

namespace policy
export Runtime.RL.PolicyGradient
  (actionPolicy actionProbability actionLogProbability actionLogSoftmax entropyBonus
   reinforceLoss actorLoss criticLoss actorCriticLoss
   a2cLoss
   importanceRatio categoricalKL categoricalKLFromLogits
   trpoSurrogateFromRatio klPenalizedPolicyLoss sacCategoricalActorLoss
   ppoClippedObjectiveFromRatio ppoClippedObjective ppoLoss)
export Runtime.RL.PolicyGradient
  (sampleCategorical sampleActionFromLogits)

namespace autograd
/-!
Differentiable policy-gradient losses over TorchLean backend references.

The pure exports above are algebra over concrete spec tensors. These helpers are the training-time
counterpart: they build scalar losses from backend refs, so the same formulas can run through eager
or typed graph autograd.
-/
export Runtime.RL.PolicyGradient.Autograd
  (actionLogProbOneHot
   entropyMean
   ppoClippedObjective
   ppoLoss)
end autograd
end policy

namespace eval
export Runtime.RL.Eval
  (greedyActionFromLogits episodeTotalReward episodeSessPath averageEpisodeTotalReward)
end eval

end rl
end TorchLean
