/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.StochasticMDP

/-!
# GridWorld (Lean-native finite RL environment)

A small deterministic **GridWorld** environment in TorchLean's *spec* layer, with three views: a
`RL.Env` whose observations are the latent states, a deterministic finite-state discounted MDP
`TorchLean.RL.FiniteMDP`, and a `TorchLean.RL.StochasticMDP` whose transition rows are **one-hot**.

The grid is a Lean-native testbed for RL algorithm development and proofs. It is small enough that
proofs about its transitions stay short, and its views have the shape of the objects in standard
RL theory.

References (high-level context only):
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 3
  (gridworld examples, discounted returns, Bellman operators).
- Puterman, *Markov Decision Processes* (1994), Chapters 6–7 (finite discounted MDPs).
- Gymnasium and TorchRL are useful API reference points for the environment/rollout shape:
  https://gymnasium.farama.org/ and https://pytorch.org/rl/
-/

@[expose] public section

namespace TorchLean
namespace Examples

open TorchLean.Tensor

/-!
## Environment Dynamics

Dynamics are deterministic and border-clamped: a move that would leave the grid keeps that
coordinate unchanged.

Reward and termination:
- At the goal, a step stays at the goal with `terminated = true` and reward `0`.
- Elsewhere, a step yields reward `0` if the successor is the goal and `-1` otherwise, and
  `terminated` holds iff the successor is the goal.
- `truncated` is always `false`, since the environment has no time limit.
-/

/-- A small deterministic GridWorld with a start cell, goal cell, and discount factor. -/
structure GridWorld (width height : Nat) where
  /-- Initial state returned by `reset`. -/
  start : Fin height × Fin width
  /-- Terminal goal cell. -/
  goal : Fin height × Fin width
  /-- Discount factor $\gamma$ used by induced MDP views. -/
  discount : ℝ

namespace GridWorld

variable {width height : Nat}

/-- A GridWorld state, represented by its row and column. -/
abbrev State (width height : Nat) : Type := Fin height × Fin width

/-- A four-neighborhood GridWorld action. -/
abbrev Action : Type := Fin 4

namespace Action

/-- Move one row upward. -/
def up : Action := ⟨0, by decide⟩

/-- Move one row downward. -/
def down : Action := ⟨1, by decide⟩

/-- Move one column to the left. -/
def left : Action := ⟨2, by decide⟩

/-- Move one column to the right. -/
def right : Action := ⟨3, by decide⟩

end Action

/-- The next row when moving one step up (saturating at `0`). -/
def rowUp (row : Fin height) : Fin height :=
  ⟨row.val - 1, Nat.lt_of_le_of_lt (Nat.sub_le row.val 1) row.isLt⟩

/-- The next row when moving one step down (clamped at `height-1`). -/
def rowDown (row : Fin height) : Fin height :=
  if h : row.val + 1 < height then
    ⟨row.val + 1, h⟩
  else
    row

/-- The next column when moving one step left (saturating at `0`). -/
def colLeft (col : Fin width) : Fin width :=
  ⟨col.val - 1, Nat.lt_of_le_of_lt (Nat.sub_le col.val 1) col.isLt⟩

/-- The next column when moving one step right (clamped at `width-1`). -/
def colRight (col : Fin width) : Fin width :=
  if h : col.val + 1 < width then
    ⟨col.val + 1, h⟩
  else
    col

/-- Deterministic successor state (border-clamped). -/
def nextState (state : State width height) (action : Action) : State width height :=
  if action = Action.up then
    (rowUp (height := height) state.1, state.2)
  else if action = Action.down then
    (rowDown (height := height) state.1, state.2)
  else if action = Action.left then
    (state.1, colLeft (width := width) state.2)
  else
    (state.1, colRight (width := width) state.2)

/-- One deterministic step, returning reward and termination flags. -/
def step (gw : GridWorld width height) (state : State width height) (action : Action) :
    RL.StepResult (State width height) ℝ :=
  if _hGoal : state = gw.goal then
    { state := state
      reward := 0
      terminated := true
      truncated := false }
  else
    let next := nextState (width := width) (height := height) state action
    if _hNextGoal : next = gw.goal then
      { state := next
        reward := 0
        terminated := true
        truncated := false }
    else
      { state := next
        reward := -1
        terminated := false
        truncated := false }

/-- `RL.Env` view of GridWorld with observations equal to latent states. -/
def toEnv (gw : GridWorld width height) :
    RL.Env (State width height) Action (State width height) ℝ :=
  { initialState := gw.start
    observe := id
    step := gw.step }

/-!
## Finite-state MDP Views

The finite discounted MDP views flatten the coordinate state through the equivalence
`Fin height × Fin width ≃ Fin (height * width)`. It is mathlib's row-major `finProdFinEquiv`, which
sends `(row, col)` to `col + width * row`.
-/

/-- Flatten a `(row,col)` grid coordinate into `Fin (height * width)`. -/
def encode (pos : State width height) : Fin (height * width) :=
  (finProdFinEquiv (m := height) (n := width)) pos

/-- Unflatten a `Fin (height * width)` state index into a `(row,col)` grid coordinate. -/
def decode (i : Fin (height * width)) : State width height :=
  (finProdFinEquiv (m := height) (n := width)).symm i

/-- Deterministic finite-state discounted MDP view of GridWorld. -/
def toFiniteMDP (gw : GridWorld width height) : RL.FiniteMDP ℝ (height * width) 4 :=
  { initialState := encode (width := width) (height := height) gw.start
    step := fun state action =>
      let out := gw.step (decode (width := width) (height := height) state) action
      { state := encode (width := width) (height := height) out.state
        reward := out.reward
        terminated := out.terminated
        truncated := out.truncated }
    discount := gw.discount }

/-- Finite-stochastic MDP view of GridWorld where $P(\mathord{\cdot}\mid s,a)$ is a one-hot row. -/
def toStochasticMDP (gw : GridWorld width height) : RL.StochasticMDP (height * width) 4 :=
  { initialState := (toFiniteMDP (width := width) (height := height) gw).initialState
    transitionProb := fun state action =>
      let out := (toFiniteMDP (width := width) (height := height) gw).step state action
      Tensor.oneHot (height * width) out.state
    reward := fun state action =>
      ((toFiniteMDP (width := width) (height := height) gw).step state action).reward
    terminated := fun state action =>
      ((toFiniteMDP (width := width) (height := height) gw).step state action).terminated
    discount := gw.discount }

end GridWorld
end Examples
end TorchLean
