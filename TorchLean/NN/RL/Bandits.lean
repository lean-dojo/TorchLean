/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Softmax
public import TorchLean.Tensor.Operations

/-!
# Bandit Algorithms

Classic finite-armed bandit algorithms: greedy and epsilon-greedy action selection, UCB1-style
confidence bonuses, incremental sample-average value estimates, and gradient bandits with a
softmax policy over preferences. Pull and reward counts stay natural numbers. The updates divide
by a count once with `Context.divNat`, and `logCount` takes the logarithm of a count without
casting the count into the scalar type.

We wrote each rule by hand from Chapter 2 of Sutton and Barto. `greedy?` is their (2.2) and
`ValueState.update` is the incremental average (2.3). `ucbScores` is the bracket of (2.10), whose
$t$ is one more than the pulls so far, so `logCount (totalPulls + 1)` is their $\ln t$ and the
default `exploration = 2` is the $c = 2$ of their Figure 2.4. UCB1, Figure 1 of Auer,
Cesa-Bianchi and Fischer, adds $\sqrt{2 \ln n / n_j}$ with $n$ the plays so far, which is
$c = \sqrt 2$ with $\ln (n + 1)$ in place of $\ln n$. Sutton and Barto break argmax ties
arbitrarily. Ours go to the lowest index, through `Tensor.argmax`.

An arm with no pulls counts as maximizing in Sutton and Barto. We divide by
`Context.defaultEpsilon` there, `1e-6` for `Float` and `ℝ`, so after one pull an untried arm's
bonus at the default is $2\sqrt{10^6 \ln 2} \approx 1665$. Before any pull every bonus is 0, so
from `ValueState.init` all scores tie and arm 0 goes first. With rewards in $[0, 1]$ the run then
tries arms 1, 2 and so on once each before it repeats one, the initialization of UCB1.

`policy` is the softmax (2.11) and `PreferenceState.update` is (2.12), whose baseline averages the
rewards before this one. Sutton and Barto set $\bar R_1 = R_1$, so their first step moves nothing.
Ours starts the mean at 0. With two arms at probability 0.5, a first reward of 5 at
`stepSize = 0.1` raises the pulled arm's preference by 0.25 and lowers the other by 0.25. Their
footnote puts $R_t$ in the baseline for the chapter's experiments, and we follow the equation.
`baseline := false` matches the run without baseline in their Figure 2.5. We found no theorem or
test for these rules.

Primary references:

- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  Sections 2.2, 2.4, 2.7 and 2.8: http://incompleteideas.net/book/the-book-2nd.html
- Auer, Cesa-Bianchi, and Fischer, "Finite-time Analysis of the Multiarmed Bandit Problem" (2002),
  Machine Learning 47(2-3):235-256, Figure 1 (UCB1): https://doi.org/10.1023/A:1013689704352
- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement
  Learning" (1992), Machine Learning 8(3-4):229-256, the family that Sutton and Barto's notes to
  Section 2.8 place gradient bandits in: https://doi.org/10.1023/A:1022672621406
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean
namespace RL
namespace Bandits

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- Value-estimation state for finite-armed bandits. -/
structure ValueState (α : Type) [Storage α] (nActions : Nat) where
  /-- Per-action pull counts. -/
  counts : Tensor Nat [nActions]
  /-- Per-action estimated values. -/
  values : Tensor α [nActions]

/-- Preference / policy-gradient state for gradient bandits. -/
structure PreferenceState (α : Type) [Storage α] (nActions : Nat) where
  /-- Number of observed rewards so far. -/
  steps : Nat
  /-- Preference logits over actions. -/
  preferences : Tensor α [nActions]
  /-- Running mean of the observed rewards, the baseline of `PreferenceState.update`. -/
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
  if h : 0 < nActions then some (Tensor.argmax state.values ⟨0, Nat.one_pos⟩ false h).item
  else none

/-- Epsilon-greedy action selection: `exploreAction` when `draw < epsilon`, otherwise the greedy
action. The caller samples both `draw` and `exploreAction`.

A unit draw such as `Random.sampleUnit` lies in `[0, 1]`, because rounding can return the
endpoint `1`, so with `epsilon = 1` a draw of `1` takes the greedy branch.
-/
def ValueState.epsilonGreedy? {nActions : Nat} (state : ValueState α nActions)
    (epsilon draw : α) (exploreAction : Fin nActions) : Option (Fin nActions) :=
  if epsilon > draw then
    some exploreAction
  else
    ValueState.greedy? (α := α) state

/-- Incremental sample-average update for one bandit arm: the arm's `n`-th reward `r` moves its
value `q` to `q + (r - q) / n`, the difference divided once by the exact count
(`Context.divNat`), so no count is rounded into `α`. -/
def ValueState.update {nActions : Nat} (state : ValueState α nActions) (action : Fin nActions)
    (reward : α) : ValueState α nActions :=
  let newCount := Tensor.getScalar state.counts action + 1
  let oldValue := Tensor.getScalar state.values action
  let newValue := oldValue + Context.divNat (reward - oldValue) newCount
  { counts := Tensor.set state.counts (action, PUnit.unit) newCount
    values := Tensor.set state.values (action, PUnit.unit) newValue }

/-- Total number of pulls recorded in a `ValueState`. -/
def ValueState.pulls {nActions : Nat} (state : ValueState α nActions) : Nat :=
  Tensor.sum state.counts

/-- The natural logarithm of a positive count `n`, as `log (n / 2 ^ k) + k * log 2` with
`k = Nat.log2 n`: the ratio in `[1, 2)` is rounded once from the exact rational, so the count is
never cast into `α`, where it can overflow. -/
def logCount (n : Nat) : α :=
  let k := Nat.log2 n
  MathFunctions.log ((n / 2 ^ k : Rat) : α) + (k : α) * MathFunctions.log ((2 : Nat) : α)

/-- UCB1-style exploration bonus `exploration * √(log (totalPulls + 1) / actionPulls)`.

The logarithm is `logCount` of the exact count, divided once by the exact pull count
(`Context.divNat`). An arm without pulls divides by `Context.defaultEpsilon` in place of a count,
which gives it a large bonus (infinite where the quotient overflows) once any arm has been pulled.
UCB1 itself pulls every arm once before it uses the bonus.
-/
def ucbBonus (exploration : α) (totalPulls actionPulls : Nat) : α :=
  let logTotal := logCount (α := α) (totalPulls + 1)
  let ratio := if actionPulls = 0 then logTotal / Context.defaultEpsilon
    else Context.divNat logTotal actionPulls
  exploration * MathFunctions.sqrt ratio

/-- Per-action UCB1 scores: each estimated value plus its `ucbBonus`. -/
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
  if h : 0 < nActions then
    some (Tensor.argmax (ValueState.ucbScores (α := α) state exploration) ⟨0, Nat.one_pos⟩ false
      h).item
  else none

/-- Softmax policy used by the gradient-bandit algorithm. -/
def PreferenceState.policy {nActions : Nat} (state : PreferenceState α nActions) :
    Tensor α [nActions] :=
  Tensor.softmax (α := α) state.preferences ⟨0, Nat.one_pos⟩

/-- Gradient-bandit preference update
`H(i) ← H(i) + stepSize * (reward - b) * (1[i = action] - π(i))`, where `π` is the softmax of the
preferences before the update and `b` is the mean of the earlier rewards (`0` with
`baseline := false`).

The running mean is updated after every reward, with or without the baseline, as in
`ValueState.update` with the exact reward count.
-/
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
    state.averageReward + Context.divNat (reward - state.averageReward) newSteps
  { steps := newSteps
    preferences := newPreferences
    averageReward := newAverageReward }

end Bandits
end RL
end TorchLean
