/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.FiniteMDP

/-!
# Finite Stochastic Discounted MDPs

Finite stochastic MDPs extend TorchLean's finite deterministic MDP layer with stochastic
transitions. States and actions are finite, rewards and the discount factor are real, and each
state-action pair has a vector of next-state probabilities, row-stochastic under `Valid`. The
Bellman expectation and optimality operators of the standard discounted setting need no more than
this, so the layer uses no measure-theoretic probability.

We wrote it by hand from Sutton and Barto's three-argument form of the dynamics. Row
`transitionProb s a` is their $p(\mathord{\cdot} \mid s, a)$ of (3.4) and `reward s a` is the
expected reward $r(s, a)$ of (3.5). The structure itself takes any real rows. We kept the
probability conditions in `Valid`, which asks each row to be nonnegative and to sum to 1 as (3.3)
does for $p$. `Valid` also asks for a discount in $[0, 1)$, and the monotonicity and contraction
theorems all take it as a hypothesis. A row
`(0.5, 0.6)` builds fine and none of them applies. The terminal flag belongs to the pair `(s, a)`,
and a terminal pair's action value is its reward with no expected next value added.

References:

- Bellman, *Dynamic Programming* (Princeton University Press, 1957)
- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (Wiley, 1994),
  Chapter 6: https://doi.org/10.1002/9780470316887
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  Section 3.1, eqs. (3.3) to (3.5): http://incompleteideas.net/book/the-book-2nd.html
- TorchRL documentation for practical stochastic-environment / rollout APIs:
  https://pytorch.org/rl/

Naming note:

- The structure is `RL.StochasticMDP`, and `Valid`, `actionValue` and the Bellman operators live in
  its namespace. They are the finite stochastic versions, not those of the deterministic tensor MDP
  `RL.FiniteMDP` or of the Markov-kernel MDP `RL.MDP`.
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace StochasticMDP

open TorchLean.Tensor

variable {nStates nActions : Nat}

/-- Finite discounted MDP with stochastic next-state transitions. -/
structure _root_.TorchLean.RL.StochasticMDP (nStates nActions : Nat) where
  /-- Canonical reset state. -/
  initialState : Fin nStates
  /-- Transition probabilities $P(\mathord{\cdot}\mid s,a)$ over the finite next-state space. -/
  transitionProb : Fin nStates → Fin nActions → Tensor ℝ [nStates]
  /-- Immediate reward `r(s, a)`. -/
  reward : Fin nStates → Fin nActions → ℝ
  /-- Task-defined terminal flag for `(s, a)`. -/
  terminated : Fin nStates → Fin nActions → Bool := fun _ _ => false
  /-- Discount factor. -/
  discount : ℝ

/-- Well-formedness assumptions for a finite stochastic MDP. -/
structure Valid {nStates nActions : Nat} (mdp : StochasticMDP nStates nActions) : Prop where
  /-- Transition probabilities are nonnegative. -/
  transition_nonneg :
    ∀ state action nextState, 0 ≤ Tensor.getScalar (mdp.transitionProb state action) nextState
  /-- Each transition row sums to `1`. -/
  transition_sums_to_one :
    ∀ state action,
      (Finset.univ : Finset (Fin nStates)).sum
        (fun nextState => Tensor.getScalar (mdp.transitionProb state action) nextState) = 1
  /-- Discount factor is nonnegative. -/
  discount_nonneg : 0 ≤ mdp.discount
  /-- Discount factor is strictly less than `1`. -/
  discount_lt_one : mdp.discount < 1

/-- Expected next-state value under $P(\mathord{\cdot}\mid s,a)$. -/
def expectedNextValue
    (mdp : StochasticMDP nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) : ℝ :=
  (Finset.univ : Finset (Fin nStates)).sum
    (fun nextState =>
      Tensor.getScalar (mdp.transitionProb state action) nextState *
        Tensor.getScalar values nextState)

/-- Bellman-style state-action value induced by a candidate value function. -/
def actionValue
    (mdp : StochasticMDP nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) : ℝ :=
  discountedBackup
    (reward := mdp.reward state action)
    (gamma := mdp.discount)
    (bootstrap := expectedNextValue mdp values state action)
    (done := mdp.terminated state action)

/-- All state-action values $Q_v(s,\mathord{\cdot})$ for a fixed state and candidate value
function. -/
def actionValues
    (mdp : StochasticMDP nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) : Tensor ℝ [nActions] :=
  Tensor.ofFn (fun action => actionValue mdp values state action)

/-- Bellman expectation operator for a deterministic policy. -/
def bellmanPolicy
    (mdp : StochasticMDP nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates) : ValueFunction ℝ nStates :=
  Tensor.ofFn (fun state => actionValue mdp values state (policy state))

/-- Bellman optimality operator for a finite stochastic MDP. -/
def bellmanMax
    [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions)
    (values : ValueFunction ℝ nStates) : ValueFunction ℝ nStates :=
  let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  Tensor.ofFn (fun state =>
    (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty
      (actionValue mdp values state))

end StochasticMDP
end RL
end TorchLean
