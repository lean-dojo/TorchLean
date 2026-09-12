/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Metrics
public import NN.Spec.Layers.Activation
public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
public import NN.Tensor.Internal.Elab.TensorLiteral
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
def greedyAction? {nActions : Nat} (state : ValueState α nActions) : Option (Fin nActions) :=
  (TorchLean.Metrics.argmax? (α := α) state.values).map
    (Fin.cast (by simp [Shape.size]))

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
  let oldCount := Tensor.getScalar state.counts action
  let newCount := oldCount + 1
  let oldValue := Tensor.getScalar state.values action
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
  let pullsSafe := Max.max actionPulls Context.defaultEpsilon
  exploration * MathFunctions.sqrt (MathFunctions.log (totalPulls + 1) / pullsSafe)

/-- Per-action UCB1 scores. -/
def ucb1Scores {nActions : Nat} (state : ValueState α nActions) (exploration : α := 2) :
    Tensor α [nActions] :=
  let total := totalPulls (α := α) state
  Tensor.dim (fun i =>
    let value := Tensor.getScalar state.values i
    let pulls := Tensor.getScalar state.counts i
    Tensor.scalar (value + ucb1Bonus (α := α) exploration total pulls))

/-- Best action under UCB1 scores, if the action space is nonempty. -/
def ucb1Action? {nActions : Nat} (state : ValueState α nActions) (exploration : α := 2) :
    Option (Fin nActions) :=
  (TorchLean.Metrics.argmax? (α := α)
    (ucb1Scores (α := α) state exploration)).map (Fin.cast (by simp [Shape.size]))

/-- Softmax policy used by the gradient-bandit algorithm. -/
def gradientPolicy {nActions : Nat} (state : PreferenceState α nActions) :
    Tensor α [nActions] :=
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
