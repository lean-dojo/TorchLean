/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Finset.Lattice.Fold
public import NN.Spec.Core.TensorOps
public import NN.Spec.RL.Core
public import NN.Spec.RL.MDP

/-!
# Finite Stochastic Discounted MDPs

This module extends TorchLean's finite deterministic MDP layer with finite-state stochastic
transitions.

We stay in a deliberately small setting:

- finitely many states and actions,
- real-valued rewards and discount factor,
- row-stochastic transition kernels represented as typed vectors.

This is enough to formalize the Bellman expectation and optimality operators in the standard
discounted setting without immediately introducing full measure-theoretic probability.

References:

- Bellman, *Dynamic Programming* (1957)
- Puterman, *Markov Decision Processes* (1994)
- Sutton and Barto, *Reinforcement Learning: An Introduction*
- TorchRL documentation for practical stochastic-environment / rollout APIs:
  https://pytorch.org/rl/

Naming note:

- The short names in this file live under `Spec.RL.FiniteStochastic`. Thus `MDP`, `Valid`,
  `actionValue`, and the Bellman operators mean the finite stochastic versions, not the
  deterministic tensor MDPs from `Spec.RL.MDP` or the Markov-kernel MDPs from
  `Spec.RL.MarkovMDP`.
- We use the file name `FiniteStochasticMDP.lean` to make the layer clear in imports, but keep the
  structure name `MDP` inside the namespace. The qualified name `Spec.RL.FiniteStochastic.MDP` is
  clearer at call sites than repeating the layer name twice.
-/

@[expose] public section

namespace Spec
namespace RL
namespace FiniteStochastic

open Tensor

variable {nStates nActions : Nat}

/-- Finite discounted MDP with stochastic next-state transitions. -/
structure MDP (nStates nActions : Nat) where
  /-- Canonical reset state. -/
  initialState : Fin nStates
  /-- Transition probabilities `P(. | s, a)` over the finite next-state space. -/
  transitionProb : Fin nStates → Fin nActions → Tensor ℝ (.dim nStates .scalar)
  /-- Immediate reward `r(s, a)`. -/
  reward : Fin nStates → Fin nActions → ℝ
  /-- Task-defined terminal flag for `(s, a)`. -/
  terminated : Fin nStates → Fin nActions → Bool := fun _ _ => false
  /-- Discount factor. -/
  discount : ℝ

/-- Well-formedness assumptions for a finite stochastic MDP. -/
structure Valid {nStates nActions : Nat} (mdp : MDP nStates nActions) : Prop where
  /-- Transition probabilities are nonnegative. -/
  transition_nonneg :
    ∀ state action nextState, 0 ≤ Tensor.vecGet (mdp.transitionProb state action) nextState
  /-- Each transition row sums to `1`. -/
  transition_sums_to_one :
    ∀ state action,
      (Finset.univ : Finset (Fin nStates)).sum
        (fun nextState => Tensor.vecGet (mdp.transitionProb state action) nextState) = 1
  /-- Discount factor is nonnegative. -/
  discount_nonneg : 0 ≤ mdp.discount
  /-- Discount factor is strictly less than `1`. -/
  discount_lt_one : mdp.discount < 1

/-- Expected next-state value under `P(. | s, a)`. -/
def expectedNextValue
    (mdp : MDP nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) : ℝ :=
  (Finset.univ : Finset (Fin nStates)).sum
    (fun nextState =>
      Tensor.vecGet (mdp.transitionProb state action) nextState * valueAt values nextState)

/-- Bellman-style state-action value induced by a candidate value function. -/
def actionValue
    (mdp : MDP nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) : ℝ :=
  discountedBackup
    (reward := mdp.reward state action)
    (gamma := mdp.discount)
    (bootstrap := expectedNextValue mdp values state action)
    (done := mdp.terminated state action)

/-- All state-action values `Q_v(s, ·)` for a fixed state and candidate value function. -/
def actionValues
    (mdp : MDP nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) : Tensor ℝ (.dim nActions .scalar) :=
  Tensor.dim (fun action => Tensor.scalar (actionValue mdp values state action))

/-- Bellman expectation operator for a deterministic policy. -/
def bellmanPolicy
    (mdp : MDP nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates) : ValueFunction ℝ nStates :=
  Tensor.dim (fun state =>
    Tensor.scalar (actionValue mdp values state (policy state)))

/-- Bellman optimality operator for a finite stochastic MDP. -/
def bellmanOptimality
    [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (values : ValueFunction ℝ nStates) : ValueFunction ℝ nStates :=
  let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  Tensor.dim (fun state =>
    Tensor.scalar
      ((Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty
        (actionValue mdp values state)))

end FiniteStochastic
end RL
end Spec
