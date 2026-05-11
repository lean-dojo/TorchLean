/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic
public import NN.Spec.RL.Envs.GridWorld

/-!
# GridWorld proof layer

This module proves a small set of “environment well-formedness” facts for the Lean-native
GridWorld defined in `NN.Spec.RL.Envs.GridWorld`:

- the step function keeps coordinates in bounds,
- rewards are bounded (`reward ∈ [-1, 0]`),
- the induced one-hot `FiniteStochastic.MDP` view satisfies the row-stochastic axioms
  (`transition_nonneg` and `transition_sums_to_one`).

These are deliberately lightweight, but they are the kind of lemmas that let downstream RL
algorithms state and prove properties without re-opening the environment definition each time.

References:

- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018),
  gridworld-style examples and dynamic programming chapters:
  http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace Proofs
namespace RL
namespace Envs
namespace GridWorld

open Spec.RL
open Spec.RL.FiniteStochastic

variable {width height : Nat}

/-- Stepping a GridWorld keeps the successor coordinates in range. -/
theorem step_state_in_bounds
    (gw : Spec.RL.Envs.GridWorld width height)
    (state : Spec.RL.Envs.GridWorld.State width height)
    (action : Spec.RL.Envs.GridWorld.Action) :
    (gw.step state action).state.1.val < height ∧
      (gw.step state action).state.2.val < width := by
  exact ⟨(gw.step state action).state.1.isLt, (gw.step state action).state.2.isLt⟩

/-- GridWorld rewards are bounded below by `-1`. -/
theorem step_reward_ge_neg_one
    (gw : Spec.RL.Envs.GridWorld width height)
    (state : Spec.RL.Envs.GridWorld.State width height)
    (action : Spec.RL.Envs.GridWorld.Action) :
    (-1 : ℝ) ≤ (gw.step state action).reward := by
  by_cases hGoal : state = gw.goal
  · simp [Spec.RL.Envs.GridWorld.step, hGoal]
  · -- In the non-goal case the reward is `0` (if we reach the goal) or `-1` (otherwise).
    by_cases hNextGoal :
        Spec.RL.Envs.GridWorld.nextState (width := width) (height := height) state action = gw.goal
    · simp [Spec.RL.Envs.GridWorld.step, hGoal, hNextGoal]
    · simp [Spec.RL.Envs.GridWorld.step, hGoal, hNextGoal]

/-- GridWorld rewards are bounded above by `0`. -/
theorem step_reward_le_zero
    (gw : Spec.RL.Envs.GridWorld width height)
    (state : Spec.RL.Envs.GridWorld.State width height)
    (action : Spec.RL.Envs.GridWorld.Action) :
    (gw.step state action).reward ≤ (0 : ℝ) := by
  by_cases hGoal : state = gw.goal
  · simp [Spec.RL.Envs.GridWorld.step, hGoal]
  · -- In the non-goal case the reward is `0` (if we reach the goal) or `-1` (otherwise).
    by_cases hNextGoal :
        Spec.RL.Envs.GridWorld.nextState (width := width) (height := height) state action = gw.goal
    · simp [Spec.RL.Envs.GridWorld.step, hGoal, hNextGoal]
    · simp [Spec.RL.Envs.GridWorld.step, hGoal, hNextGoal]

/-- Combined reward bound for convenience (`reward ∈ [-1, 0]`). -/
theorem step_reward_bounds
    (gw : Spec.RL.Envs.GridWorld width height)
    (state : Spec.RL.Envs.GridWorld.State width height)
    (action : Spec.RL.Envs.GridWorld.Action) :
    (-1 : ℝ) ≤ (gw.step state action).reward ∧ (gw.step state action).reward ≤ (0 : ℝ) :=
  ⟨step_reward_ge_neg_one (gw := gw) (state := state) (action := action),
    step_reward_le_zero (gw := gw) (state := state) (action := action)⟩

/-!
## Finite-stochastic MDP validity

The `FiniteStochastic.MDP` view of GridWorld represents deterministic transitions as one-hot rows.
Here we show this satisfies the standard “row-stochastic” assumptions used throughout the
finite-state proof development.
-/

/-- The one-hot `FiniteStochastic.MDP` view of GridWorld is valid assuming the discount is in `[0,1)`. -/
theorem toFiniteStochasticMDP_valid
    (gw : Spec.RL.Envs.GridWorld width height)
    (hγ₀ : 0 ≤ gw.discount)
    (hγ₁ : gw.discount < 1) :
    Valid (Spec.RL.Envs.GridWorld.toFiniteStochasticMDP (width := width) (height := height) gw) := by
  classical
  refine
    { transition_nonneg := ?_
      transition_sums_to_one := ?_
      discount_nonneg := hγ₀
      discount_lt_one := hγ₁ }
  · intro state action nextState
    -- Entries are `0` or `1` by construction.
    by_cases hx : nextState = (gw.toFiniteMDP.step state action).state
    · simp [Spec.RL.Envs.GridWorld.toFiniteStochasticMDP, Spec.RL.Envs.GridWorld.oneHot,
        Spec.Tensor.vecGet, Spec.get, Spec.getAtSpec, Spec.Tensor.toScalar, hx]
    · simp [Spec.RL.Envs.GridWorld.toFiniteStochasticMDP, Spec.RL.Envs.GridWorld.oneHot,
        Spec.Tensor.vecGet, Spec.get, Spec.getAtSpec, Spec.Tensor.toScalar, hx]
  · intro state action
    -- A one-hot row sums to `1`.
    simp [Spec.RL.Envs.GridWorld.toFiniteStochasticMDP, Spec.RL.Envs.GridWorld.oneHot,
      Spec.Tensor.vecGet, Spec.get, Spec.getAtSpec, Spec.Tensor.toScalar,
      Finset.sum_ite_eq', Finset.mem_univ]

/-!
## Deterministic / One-Hot Equivalence

The finite-stochastic GridWorld view is not a different environment. It packages the deterministic
successor as a one-hot transition row. The next theorem makes that bridge explicit at the Bellman
interface: taking an expectation against that one-hot row is the same as reading the value of the
deterministic successor state.
-/

/--
Expected next-state value in the one-hot finite-stochastic GridWorld view equals the value of the
successor produced by the deterministic finite-MDP view.
-/
theorem toFiniteStochasticMDP_expectedNextValue_eq_toFiniteMDP_successor
    (gw : Spec.RL.Envs.GridWorld width height)
    (values : ValueFunction ℝ (height * width))
    (state : Fin (height * width))
    (action : Fin 4) :
    expectedNextValue
        (Spec.RL.Envs.GridWorld.toFiniteStochasticMDP (width := width) (height := height) gw)
        values state action =
      valueAt values
        ((Spec.RL.Envs.GridWorld.toFiniteMDP (width := width) (height := height) gw).step
          state action).state := by
  classical
  simp [Spec.RL.FiniteStochastic.expectedNextValue,
    Spec.RL.Envs.GridWorld.toFiniteStochasticMDP,
    Spec.RL.Envs.GridWorld.oneHot,
    Spec.Tensor.vecGet, Spec.get, Spec.getAtSpec, Spec.Tensor.toScalar,
    Finset.sum_ite_eq', Finset.mem_univ]

/--
The full Bellman state-action value agrees between GridWorld's deterministic finite-MDP view and
its one-hot finite-stochastic view.
-/
theorem toFiniteStochasticMDP_actionValue_eq_toFiniteMDP_stateActionValue
    (gw : Spec.RL.Envs.GridWorld width height)
    (values : ValueFunction ℝ (height * width))
    (state : Fin (height * width))
    (action : Fin 4) :
    Spec.RL.FiniteStochastic.actionValue
        (Spec.RL.Envs.GridWorld.toFiniteStochasticMDP (width := width) (height := height) gw)
        values state action =
      stateActionValue
        (Spec.RL.Envs.GridWorld.toFiniteMDP (width := width) (height := height) gw)
        values state action := by
  simp only [Spec.RL.FiniteStochastic.actionValue, stateActionValue]
  rw [toFiniteStochasticMDP_expectedNextValue_eq_toFiniteMDP_successor]
  simp [Spec.RL.Envs.GridWorld.toFiniteStochasticMDP, Spec.RL.Envs.GridWorld.toFiniteMDP]

end GridWorld
end Envs
end RL
end Proofs
