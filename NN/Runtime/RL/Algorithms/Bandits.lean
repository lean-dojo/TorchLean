/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Core
public import NN.Runtime.Autograd.TorchLean.Metrics
public import NN.Spec.Layers.Activation

/-!
# Bandit Algorithms

This module implements a small set of classic discrete-action bandit algorithms:

- greedy / epsilon-greedy action selection,
- UCB1-style confidence bonuses,
- incremental sample-average value estimation,
- gradient bandits with a softmax policy over preferences.

Primary references:

- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., bandit chapter):
  http://incompleteideas.net/book/the-book-2nd.html
- Auer, Cesa-Bianchi, and Fischer, "Finite-time Analysis of the Multiarmed Bandit Problem" (2002):
  https://doi.org/10.1023/A:1013689704352
- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement
  Learning" (1992): https://doi.org/10.1023/A:1022672621406
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Bandits

open Spec
open Tensor

variable {α : Type} [Context α]

/-- Value-estimation state for finite-armed bandits. -/
structure ValueState (α : Type) (nActions : Nat) where
  /-- Per-action pull counts. -/
  counts : Tensor α (.dim nActions .scalar)
  /-- Per-action estimated values. -/
  values : Tensor α (.dim nActions .scalar)

/-- Preference / policy-gradient state for gradient bandits. -/
structure PreferenceState (α : Type) (nActions : Nat) where
  /-- Number of observed rewards so far (tracked as the ambient scalar type). -/
  steps : α
  /-- Preference logits over actions. -/
  preferences : Tensor α (.dim nActions .scalar)
  /-- Running average reward baseline. -/
  averageReward : α

/-- Zero-initialized action-value state. -/
def ValueState.init {nActions : Nat} : ValueState α nActions :=
  { counts := fill 0 (.dim nActions .scalar)
    values := fill 0 (.dim nActions .scalar) }

/-- Zero-initialized preference state. -/
def PreferenceState.init {nActions : Nat} : PreferenceState α nActions :=
  { steps := 0
    preferences := fill 0 (.dim nActions .scalar)
    averageReward := 0 }

/-- Greedy action under the current estimates, if the action space is nonempty. -/
def greedyAction? {nActions : Nat} (state : ValueState α nActions) : Option (Fin nActions) :=
  Runtime.Autograd.TorchLean.Metrics.argmax? (α := α) (n := nActions) state.values

/-- Epsilon-greedy action selection with explicit exploration draw and fallback action.

The caller supplies:
- `epsilon`: exploration probability,
- `draw`: a pre-sampled uniform value in `[0,1)`,
- `exploreAction`: the action to use when the exploration branch is taken.
-/
def epsilonGreedyAction? {nActions : Nat} (state : ValueState α nActions)
    (epsilon draw : α) (exploreAction : Fin nActions) : Option (Fin nActions) :=
  if epsilon > draw then
    some exploreAction
  else
    greedyAction? (α := α) state

/-- Incremental sample-average update for one bandit arm. -/
def sampleAverageStep {nActions : Nat} (state : ValueState α nActions) (action : Fin nActions)
    (reward : α) : ValueState α nActions :=
  let oldCount := Tensor.vecGet state.counts action
  let newCount := oldCount + 1
  let oldValue := Tensor.vecGet state.values action
  let newValue := oldValue + (reward - oldValue) / newCount
  { counts := Tensor.updateSpec state.counts [action.val] newCount
    values := Tensor.updateSpec state.values [action.val] newValue }

/-- Total number of pulls recorded in a `ValueState`. -/
def totalPulls {nActions : Nat} (state : ValueState α nActions) : α :=
  sumSpec state.counts

/-- UCB1-style exploration bonus.

We use `max(pulls, epsilon)` in the denominator so the helper stays total while still giving
very large bonuses to unseen or nearly-unseen actions.
-/
def ucb1Bonus (exploration totalPulls actionPulls : α) : α :=
  let pullsSafe := Max.max actionPulls Numbers.epsilon
  exploration * MathFunctions.sqrt (MathFunctions.log (totalPulls + 1) / pullsSafe)

/-- Per-action UCB1 scores. -/
def ucb1Scores {nActions : Nat} (state : ValueState α nActions) (exploration : α := Numbers.two) :
    Tensor α (.dim nActions .scalar) :=
  let total := totalPulls (α := α) state
  Tensor.dim (fun i =>
    let value := Tensor.vecGet state.values i
    let pulls := Tensor.vecGet state.counts i
    Tensor.scalar (value + ucb1Bonus (α := α) exploration total pulls))

/-- Best action under UCB1 scores, if the action space is nonempty. -/
def ucb1Action? {nActions : Nat} (state : ValueState α nActions) (exploration : α := Numbers.two) :
    Option (Fin nActions) :=
  Runtime.Autograd.TorchLean.Metrics.argmax? (α := α) (n := nActions)
    (ucb1Scores (α := α) state exploration)

/-- Softmax policy used by the gradient-bandit algorithm. -/
def gradientPolicy {nActions : Nat} (state : PreferenceState α nActions) :
    Tensor α (.dim nActions .scalar) :=
  Activation.softmaxVecSpec (α := α) (n := nActions) state.preferences

/-- Gradient-bandit preference update with an optional average-reward baseline. -/
def gradientBanditStep {nActions : Nat} (state : PreferenceState α nActions) (action :
    Fin nActions) (reward stepSize : α) (useBaseline : Bool := true) : PreferenceState α nActions :=
  let newSteps := state.steps + 1
  let probs := gradientPolicy (α := α) state
  let baseline := if useBaseline then state.averageReward else 0
  let advantage := reward - baseline
  let newPreferences :=
    Tensor.dim (fun i =>
      let p := Tensor.vecGet probs i
      let pref := Tensor.vecGet state.preferences i
      let indicator : α := if i = action then 1 else 0
      Tensor.scalar (pref + stepSize * advantage * (indicator - p)))
  let newAverageReward :=
    state.averageReward + (reward - state.averageReward) / newSteps
  { steps := newSteps
    preferences := newPreferences
    averageReward := newAverageReward }

end Bandits
end RL
end Runtime
