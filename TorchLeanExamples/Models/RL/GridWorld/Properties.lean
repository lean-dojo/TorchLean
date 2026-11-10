/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Models.RL.GridWorld.Env

/-!
# GridWorld proof layer

Well-formedness facts for the Lean-native GridWorld of `TorchLeanExamples.Models.RL.GridWorld.Env`:

- coordinates stay in bounds by the type `Fin height × Fin width` of a state,
- every reward lies in `[-1, 0]`,
- the induced one-hot `RL.StochasticMDP` view satisfies the row-stochastic axioms
  (`transition_nonneg` and `transition_sums_to_one`).

Later RL proofs can cite these lemmas without unfolding the step function.

References:

- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018),
  gridworld-style examples and dynamic programming chapters:
  http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace TorchLean
namespace Examples.GridWorld

open RL FiniteMDP
open StochasticMDP

variable {width height : Nat}

/-- GridWorld rewards are bounded below by `-1`. -/
theorem step_reward_ge_neg_one
    (gw : GridWorld width height)
    (state : GridWorld.State width height)
    (action : GridWorld.Action) :
    (-1 : ℝ) ≤ (gw.step state action).reward := by
  by_cases hGoal : state = gw.goal
  · simp [GridWorld.step, hGoal]
  · -- Away from the goal the reward is `0` if the step reaches the goal and `-1` otherwise.
    by_cases hNextGoal :
        GridWorld.nextState (width := width) (height := height) state action = gw.goal
    all_goals simp [GridWorld.step, hGoal, hNextGoal]

/-- GridWorld rewards are bounded above by `0`. -/
theorem step_reward_le_zero
    (gw : GridWorld width height)
    (state : GridWorld.State width height)
    (action : GridWorld.Action) :
    (gw.step state action).reward ≤ (0 : ℝ) := by
  by_cases hGoal : state = gw.goal
  · simp [GridWorld.step, hGoal]
  · -- Away from the goal the reward is `0` if the step reaches the goal and `-1` otherwise.
    by_cases hNextGoal :
        GridWorld.nextState (width := width) (height := height) state action = gw.goal
    all_goals simp [GridWorld.step, hGoal, hNextGoal]

/-- Every GridWorld reward lies in `[-1, 0]`. -/
theorem step_reward_bounds
    (gw : GridWorld width height)
    (state : GridWorld.State width height)
    (action : GridWorld.Action) :
    (-1 : ℝ) ≤ (gw.step state action).reward ∧ (gw.step state action).reward ≤ (0 : ℝ) :=
  ⟨step_reward_ge_neg_one (gw := gw) (state := state) (action := action),
    step_reward_le_zero (gw := gw) (state := state) (action := action)⟩

/-!
## Finite-stochastic MDP validity

The `RL.StochasticMDP` view of GridWorld writes each deterministic transition as a one-hot
row, and one-hot rows satisfy the “row-stochastic” assumptions of the finite-state proofs: their
entries are nonnegative and sum to `1`.
-/

/-- If the discount lies in `[0, 1)`, the one-hot `RL.StochasticMDP` view of GridWorld is
`Valid`. -/
theorem toStochasticMDP_valid
    (gw : GridWorld width height)
    (hγ₀ : 0 ≤ gw.discount)
    (hγ₁ : gw.discount < 1) :
    Valid (GridWorld.toStochasticMDP (width := width) (height := height) gw)
      := by
  classical
  refine
    { transition_nonneg := ?_
      transition_sums_to_one := ?_
      discount_nonneg := hγ₀
      discount_lt_one := hγ₁ }
  · intro state action nextState
    -- Entries are `0` or `1` by construction.
    by_cases hx : nextState = (gw.toFiniteMDP.step state action).state <;>
      simp [GridWorld.toStochasticMDP, Tensor.oneHot,
        Tensor.getScalar, Tensor.item, hx]
  · intro state action
    -- A one-hot row sums to `1`.
    simp [GridWorld.toStochasticMDP, Tensor.oneHot,
      Tensor.getScalar, Tensor.item,
      Finset.sum_ite_eq', Finset.mem_univ]

/-!
## Deterministic / One-Hot Equivalence

The finite-stochastic GridWorld view is the same environment, with the deterministic successor
packaged as a one-hot transition row. At the Bellman interface, the expectation against that row
is the value of the deterministic successor state, and the state-action values of the two views
agree.
-/

/--
Expected next-state value in the one-hot finite-stochastic GridWorld view equals the value of the
successor produced by the deterministic finite-MDP view.
-/
theorem expectedNextValue_eq_getScalar
    (gw : GridWorld width height)
    (values : ValueFunction ℝ (height * width))
    (state : Fin (height * width))
    (action : Fin 4) :
    expectedNextValue
        (GridWorld.toStochasticMDP (width := width) (height := height) gw)
        values state action =
      Tensor.getScalar values
        ((GridWorld.toFiniteMDP (width := width) (height := height) gw).step
          state action).state := by
  classical
  simp [RL.StochasticMDP.expectedNextValue,
    GridWorld.toStochasticMDP,
    Tensor.oneHot,
    Tensor.getScalar, Tensor.item,
    Finset.sum_ite_eq', Finset.mem_univ]

/--
The full Bellman state-action value agrees between GridWorld's deterministic finite-MDP view and
its one-hot finite-stochastic view.
-/
theorem actionValue_toStochasticMDP
    (gw : GridWorld width height)
    (values : ValueFunction ℝ (height * width))
    (state : Fin (height * width))
    (action : Fin 4) :
    RL.StochasticMDP.actionValue
        (GridWorld.toStochasticMDP (width := width) (height := height) gw)
        values state action =
      FiniteMDP.actionValue
        (GridWorld.toFiniteMDP (width := width) (height := height) gw)
        values state action := by
  simp only [RL.StochasticMDP.actionValue, FiniteMDP.actionValue]
  rw [expectedNextValue_eq_getScalar]
  simp [GridWorld.toStochasticMDP, GridWorld.toFiniteMDP]

end Examples.GridWorld
end TorchLean
