/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime
public import NN.Spec.RL.Core
public import NN.Spec.RL.Environment
public import NN.Spec.RL.MDP
public import NN.Spec.RL.MarkovMDP
public import NN.Spec.RL.FiniteStochasticMDP
public import NN.Runtime.Training.Log

/-!
# Public RL Facade

This module exposes TorchLean's reinforcement-learning helper surface under the public
`NN.API.rl.*` namespace.

Design intent:
- keep the public API smaller and easier to browse than the full runtime namespace,
- mirror the existing `NN.API.*` facade pattern,
- expose typed RL math while keeping environment/trainer integration separate.

References (background and terminology):
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.):
  http://incompleteideas.net/book/the-book-2nd.html
- Puterman, *Markov Decision Processes* (finite discounted MDPs):
  https://doi.org/10.1002/9780470316887
- Gymnasium API docs (reset/step, `terminated` vs `truncated`):
  https://gymnasium.farama.org/
-/

@[expose] public section

namespace NN
namespace API

namespace rl

namespace env
export _root_.Spec.RL
  (StepResult ObservedTransition Env SafeEnv
   reset stepGym evolve evolveFrom
   states statesFrom rollout rolloutFrom)
export _root_.Spec.RL.StepResult (done)
export _root_.Spec.RL.SafeEnv (actionPathOk)
end env

namespace core
export _root_.Spec.RL
  (AdvantageStep
   continueMask discountedBackup tdTarget tdResidual
   discountedReturns discountedReturnsFrom discountedReturnsDone
   generalizedAdvantageEstimation returnsFromAdvantages)
export _root_.Runtime.RL.Core
  (Transition IndexedTransition
   oneHotAction
   discountedReturnsVecFrom discountedReturnsVec discountedReturnsVecDone
   generalizedAdvantageEstimationVec returnsFromAdvantagesVec
   squaredError huberLoss)
end core

namespace mdp
export _root_.Spec.RL
  (ValueFunction Policy FiniteMDP
   valueAt stateActionValue actionValues
   bellmanPolicy bellmanOptimality)
export _root_.Spec.RL.FiniteMDP (toEnv)
end mdp

namespace markov
export _root_.Spec.RL.Markov
  (ValueFunction Policy MDP Valid
   transitionMeasure
   expectedNextValue actionValue
   bellmanPolicy bellmanOptimality)
end markov

namespace finiteStochastic
export _root_.Spec.RL.FiniteStochastic
  (MDP Valid
   expectedNextValue actionValue actionValues
   bellmanPolicy bellmanOptimality)
end finiteStochastic

namespace bandits
export _root_.Runtime.RL.Bandits
  (ValueState PreferenceState
   greedyAction? epsilonGreedyAction?
   sampleAverageStep totalPulls
   ucb1Bonus ucb1Scores ucb1Action?
   gradientPolicy gradientBanditStep)
export _root_.Runtime.RL.Bandits.ValueState (init)
export _root_.Runtime.RL.Bandits.PreferenceState (init)
end bandits

namespace tabular
export _root_.Runtime.RL.Tabular
  (actionRow maxActionValue greedyAction? expectedActionValue
   td0Update
   sarsaTarget expectedSarsaTarget qLearningTarget doubleQTarget
   sarsaUpdate expectedSarsaUpdate qLearningUpdate
   doubleQUpdateLeft doubleQUpdateRight)
end tabular

namespace value
export _root_.Runtime.RL.ValueLearning
  (chosenActionValue maxQValue
   dqnTarget doubleDqnTarget
   dqnResidual dqnMSELoss dqnHuberLoss doubleDqnResidual
   ddpgActorObjective ddpgCriticTarget td3Target
   sacTarget sacActorObjective)
end value

namespace replay
export _root_.Runtime.RL.Replay (Transition Buffer)
export _root_.Runtime.RL.Replay.Buffer
  (empty size isEmpty isFull push pushMany getModulo? sampleContiguous sampleRandom)
end replay

namespace dqn
export _root_.Runtime.RL.DQN
  (meanArray
   transitionMSELoss transitionHuberLoss transitionDoubleHuberLoss
   minibatchMSELoss minibatchHuberLoss minibatchDoubleHuberLoss
   softUpdateScalar)
end dqn

namespace policy
export _root_.Runtime.RL.PolicyGradient
  (actionPolicy actionProbability actionLogProbability entropyBonus
   reinforceLoss actorLoss criticLoss actorCriticLoss
   a2cLoss
   importanceRatio categoricalKL categoricalKLFromLogits
   trpoSurrogateFromRatio klPenalizedPolicyLoss sacCategoricalActorLoss
   ppoClippedObjectiveFromRatio ppoClippedObjective ppoLoss)
export _root_.Runtime.RL.PolicyGradient
  (sampleCategorical sampleActionFromLogits)

namespace autograd
/-!
Differentiable policy-gradient losses over TorchLean backend references.

The pure exports above are algebra over concrete spec tensors. These helpers are the training-time
counterpart: they build scalar losses from backend refs, so the same formulas can run through eager
or compiled autograd.
-/
export _root_.Runtime.RL.PolicyGradient.Autograd
  (actionLogProbOneHotBatch
   entropyMean
   ppoClippedObjectiveBatch
   ppoLossBatch
   ppoActorCriticScalarModuleDef)
end autograd
end policy

namespace eval
export _root_.Runtime.RL.Eval
  (greedyActionFromLogits episodeTotalReward episodeSessPath averageEpisodeTotalReward)
end eval

namespace train
/-!
## Training Logs (Widgets and Examples)

TorchLean does not aim to be a full “trainer framework”, but many executable examples want to:

- evaluate a scalar metric every `N` updates,
- append it to a curve, and
- write a small JSON file for widgets (`#train_log_file_view`).

This namespace re-exports the small, stable log types and JSON IO helpers.
-/
export _root_.Runtime.Training
  (Series TrainLog Curve ConfigEntry Artifact RunInfo ExperimentLog LogDestination)
export _root_.Runtime.Training.Curve (push toTrainLog)
export _root_.Runtime.Training.TrainLog (writeJson readJson)
export _root_.Runtime.Training.ExperimentLog (init log logRow addArtifact toTrainLog)
export _root_.Runtime.Training.LogDestination (disabled json isEnabled path? writeTrainLog)
end train

namespace boundary
export _root_.Runtime.RL.Boundary
  (isFiniteFloat tensorAll tensorFinite tensorInClosedInterval
   Contract Transition
   checkAction
   checkObservation checkReward checkDoneFlags
   checkTransitionFin checkTransition
   parseTransitionJson loadRollout)
export _root_.Runtime.RL.Boundary.Transition (done)

/-!
## Casting to Other Scalar Backends

The trust-boundary checker (`Runtime.RL.Boundary`) validates rollouts in terms of host `Float`
because that is what our lightweight JSON interchange format uses.

Most RL math in TorchLean is scalar-polymorphic (`[Context α]`), so it is often convenient to
cast a validated `Float` rollout into the chosen runtime scalar backend:
- `Float` (fast host execution),
- `IEEE32Exec` (executable bit-level float32),
- any other backend that supports `Runtime.ofFloat`.
-/

/-- Cast a `Float` observation tensor into a runtime scalar backend `α`. -/
def castObs {α : Type} [Runtime.Scalar α] {obsShape : _root_.Spec.Shape}
    (t : _root_.Spec.Tensor Float obsShape) : _root_.Spec.Tensor α obsShape :=
  _root_.Spec.mapTensor (Runtime.ofFloat (α := α)) t

/-- Cast a validated `Float` transition into a runtime scalar backend `α`. -/
def castTransition {α : Type} [Runtime.Scalar α]
    {obsShape : _root_.Spec.Shape} {nActions : Nat}
    (tr : _root_.Runtime.RL.Boundary.Transition obsShape nActions) :
    _root_.Spec.RL.ObservedTransition (_root_.Spec.Tensor α obsShape) (Fin nActions) α :=
  { observation := castObs (α := α) tr.observation
    action := tr.action
    reward := Runtime.ofFloat (α := α) tr.reward
    nextObservation := castObs (α := α) tr.nextObservation
    terminated := tr.terminated
    truncated := tr.truncated }

/-- Cast a whole rollout (array of transitions) into a runtime scalar backend `α`. -/
def castRollout {α : Type} [Runtime.Scalar α]
    {obsShape : _root_.Spec.Shape} {nActions : Nat}
    (xs : Array (_root_.Runtime.RL.Boundary.Transition obsShape nActions)) :
    Array (_root_.Spec.RL.ObservedTransition (_root_.Spec.Tensor α obsShape) (Fin nActions) α) :=
  xs.map (castTransition (α := α) (obsShape := obsShape) (nActions := nActions))

/-- Load a rollout JSON file, validate it with the boundary contract, then cast to scalar `α`. -/
def loadRolloutCast {α : Type} [Runtime.Scalar α]
    {obsShape : _root_.Spec.Shape} {nActions : Nat}
    (path : String)
    (c : _root_.Runtime.RL.Boundary.Contract obsShape nActions) :
    IO (Array (_root_.Spec.RL.ObservedTransition (_root_.Spec.Tensor α obsShape) (Fin nActions) α)) := do
  let xs ← _root_.Runtime.RL.Boundary.loadRollout (obsShape := obsShape) (nActions := nActions) path c
  pure (castRollout (α := α) (obsShape := obsShape) (nActions := nActions) xs)

end boundary

namespace numerics
namespace float32
export _root_.Runtime.RL.Numerics.Float32
  (Float32Exec Interval32
   ofFloatIEEE32ExecChecked castTensorIEEE32ExecChecked castTransitionIEEE32ExecChecked
   discountedBackupIEEE32ExecChecked discountedReturnsVecFromIEEE32ExecChecked
   tdResidualIEEE32ExecChecked
   generalizedAdvantageEstimationVecIEEE32ExecChecked
   normalizeZScoreIEEE32ExecChecked
   importanceRatioIEEE32ExecChecked
   ppoClippedObjectiveFromRatioIEEE32ExecChecked
   discountedBackupInterval32 tdResidualInterval32
   ppoClippedObjectiveFromRatioInterval32
   discountedReturnsIntervals32 generalizedAdvantageEstimationIntervals32
   returnsWithinIntervals32)
end float32
end numerics

namespace session
export _root_.Runtime.RL.Session (CheckedSession)
export _root_.Runtime.RL.Session.CheckedSession (gymnasium ofEnv)
end session

namespace gym
export _root_.Runtime.RL.Gymnasium (Client Session)

namespace client
-- Only export the stable high-level entry points. The JSON request/response protocol and raw-step
-- protocol remain behind `NN.Runtime.RL.Gymnasium`.
export _root_.Runtime.RL.Gymnasium.Client (spawn reset close withClient)
end client

namespace session
export _root_.Runtime.RL.Gymnasium.Session (start reset stepChecked close withSession)
end session

end gym

namespace ppo
export _root_.Runtime.RL.PPO
  (StateBatchShape LogitsBatchShape ScalarBatchShape ValueBatchShape
   Step Rollout
   collectRolloutSessionWith collectRolloutCheckedSessionWith collectRolloutWith)
export _root_.Runtime.RL.PPO.Rollout (toActorCriticSample)

/--
Split a concatenated actor-critic parameter pack into `(actorParams, criticParams)`.

PPO examples often bundle actor and critic parameters as `actor.params ++ critic.params` to update
them with a single optimizer step (`ppoActorCriticScalarModuleDef`). When we want to run just the
actor for evaluation or action selection, we need to recover the actor slice.

This helper keeps example code from reaching into the long proved `TList.splitAppend` path.
-/
def splitActorCriticParams
    {σ₁ τ₁ σ₂ τ₂ : _root_.Spec.Shape}
    (actor : _root_.Runtime.Autograd.TorchLean.NN.Seq σ₁ τ₁)
    (critic : _root_.Runtime.Autograd.TorchLean.NN.Seq σ₂ τ₂)
    {α : Type}
    (ps :
      _root_.Runtime.Autograd.Torch.TList α
        (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor ++
          _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic)) :
    _root_.Runtime.Autograd.Torch.TList α (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor) ×
      _root_.Runtime.Autograd.Torch.TList α (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic) :=
  _root_.Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.splitAppend (α := α)
    (ss₁ := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor)
    (ss₂ := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic)
    ps
end ppo

end rl

end API
end NN
