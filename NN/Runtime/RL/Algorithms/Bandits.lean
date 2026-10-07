/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Metrics
public import NN.Runtime.RL.Core -- shake: keep

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

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Value-estimation state for finite-armed bandits. -/
structure ValueState (α : Type) [TorchLean.Storage α] (nActions : Nat) where
  /-- Per-action pull counts. -/
  counts : Tensor α [nActions]
  /-- Per-action estimated values. -/
  values : Tensor α [nActions]

/-- Preference / policy-gradient state for gradient bandits. -/
structure PreferenceState (α : Type) [TorchLean.Storage α] (nActions : Nat) where
  /-- Number of observed rewards so far (tracked as the ambient scalar type). -/
  steps : α
  /-- Preference logits over actions. -/
  preferences : Tensor α [nActions]
  /-- Running average reward baseline. -/
  averageReward : α

/-- Zero-initialized action-value state. -/
def ValueState.init {nActions : Nat} : ValueState α nActions :=
  { counts := Tensor.full (.dim nActions .scalar) 0
    values := Tensor.full (.dim nActions .scalar) 0 }

/-- Zero-initialized preference state. -/
def PreferenceState.init {nActions : Nat} : PreferenceState α nActions :=
  { steps := 0
    preferences := Tensor.full (.dim nActions .scalar) 0
    averageReward := 0 }

/-- Greedy action under the current estimates, if the action space is nonempty. -/
def ValueState.greedy? {nActions : Nat} (state : ValueState α nActions) : Option (Fin nActions) :=
  (TorchLean.Metrics.argmax? (α := α) state.values).map
    (Fin.cast (by simp [Shape.size]))

/-- Epsilon-greedy action selection with explicit exploration draw and fallback action.

The caller supplies:
- `epsilon`: exploration probability,
- `draw`: a pre-sampled uniform value in `[0,1)`,
- `exploreAction`: the action to use when the exploration branch is taken.
-/
def ValueState.epsilonGreedy? {nActions : Nat} (state : ValueState α nActions)
    (epsilon draw : α) (exploreAction : Fin nActions) : Option (Fin nActions) :=
  if epsilon > draw then
    some exploreAction
  else
    ValueState.greedy? (α := α) state

/-- Incremental sample-average update for one bandit arm. -/
def ValueState.update {nActions : Nat} (state : ValueState α nActions) (action : Fin nActions)
    (reward : α) : ValueState α nActions :=
  let oldCount := Tensor.getScalar state.counts action
  let newCount := oldCount + 1
  let oldValue := Tensor.getScalar state.values action
  let newValue := oldValue + (reward - oldValue) / newCount
  { counts := Tensor.set state.counts (action, PUnit.unit) newCount
    values := Tensor.set state.values (action, PUnit.unit) newValue }

/-- Total number of pulls recorded in a `ValueState`. -/
def ValueState.pulls {nActions : Nat} (state : ValueState α nActions) : α :=
  sumSpec state.counts

/-- UCB1-style exploration bonus.

The denominator is clamped at `Context.defaultEpsilon`, without a special infinity rule for unseen
arms. `exploration` controls the multiplier. Counts and the scalar epsilon are used as supplied.
-/
def ucbBonus (exploration totalPulls actionPulls : α) : α :=
  let pullsSafe := Max.max actionPulls Context.defaultEpsilon
  exploration * MathFunctions.sqrt (MathFunctions.log (totalPulls + 1) / pullsSafe)

/-- Per-action UCB1 scores. -/
def ValueState.ucbScores {nActions : Nat} (state : ValueState α nActions) (exploration : α := 2) :
    Tensor α [nActions] :=
  let total := ValueState.pulls (α := α) state
  Tensor.dim (fun i =>
    let value := Tensor.getScalar state.values i
    let pulls := Tensor.getScalar state.counts i
    Tensor.scalar (value + ucbBonus (α := α) exploration total pulls))

/-- Best action under UCB1 scores, if the action space is nonempty. -/
def ValueState.ucb? {nActions : Nat} (state : ValueState α nActions) (exploration : α := 2) :
    Option (Fin nActions) :=
  (TorchLean.Metrics.argmax? (α := α)
    (ValueState.ucbScores (α := α) state exploration)).map (Fin.cast (by simp [Shape.size]))

/-- Softmax policy used by the gradient-bandit algorithm. -/
def PreferenceState.policy {nActions : Nat} (state : PreferenceState α nActions) :
    Tensor α [nActions] :=
  Activation.softmaxVecSpec (α := α) (n := nActions) state.preferences

/-- Gradient-bandit preference update with an optional average-reward baseline. -/
def PreferenceState.update {nActions : Nat} (state : PreferenceState α nActions)
    (action : Fin nActions) (reward stepSize : α) (baseline : Bool := true) :
    PreferenceState α nActions :=
  let newSteps := state.steps + 1
  let probs := PreferenceState.policy (α := α) state
  let reference := if baseline then state.averageReward else 0
  let advantage := reward - reference
  let newPreferences :=
    Tensor.dim (fun i =>
      let p := Tensor.getScalar probs i
      let pref := Tensor.getScalar state.preferences i
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
