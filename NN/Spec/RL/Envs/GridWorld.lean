/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.RL.FiniteStochasticMDP

/-!
# GridWorld (Lean-native finite RL environment)

This file defines a small deterministic **GridWorld** environment in TorchLean’s *spec* layer,
along with two induced “MDP views”:

- a `Spec.RL.Env` view with explicit latent state,
- a `Spec.RL.FiniteMDP` view (deterministic finite-state discounted MDP),
- and a `Spec.RL.FiniteStochastic.MDP` view where transitions are represented as **one-hot**
  row-stochastic kernels.

This is intended as a Lean-native testbed for RL algorithm development and proofs: small enough
to keep the transition proof small, but shaped like the objects used in standard RL theory.

References (high-level context only):
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 3
  (gridworld examples, discounted returns, Bellman operators).
- Puterman, *Markov Decision Processes* (1994), Chapters 6–7 (finite discounted MDPs).
- Gymnasium and TorchRL are useful API reference points for the environment/rollout shape:
  https://gymnasium.farama.org/ and https://pytorch.org/rl/
-/

@[expose] public section

open TorchLean

namespace Spec
namespace RL
namespace Envs

open TorchLean TorchLean.Tensor

/-!
## Environment Dynamics

Dynamics are deterministic and border-clamped:
- attempting to move outside the grid keeps the coordinate unchanged.

Reward / termination scheme:
- If the agent is already at the goal, the environment remains terminal (`terminated = true`)
  and yields reward `0`.
- Otherwise, a step yields reward `0` iff the successor state is the goal, and reward `-1`
  otherwise. The `terminated` flag matches “successor is goal”.
- `truncated` is always `false` (no time-limit semantics in this environment).
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
    StepResult (State width height) ℝ :=
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

/-- `Spec.RL.Env` view of GridWorld with observations equal to latent states. -/
def toEnv (gw : GridWorld width height) :
    Env (State width height) Action (State width height) ℝ :=
  { initialState := gw.start
    observe := id
    step := gw.step }

/-!
## Finite-state MDP Views

To connect GridWorld to TorchLean’s finite discounted MDP layer we flatten the coordinate state:

`Fin height × Fin width ≃ Fin (height * width)`.

This is the standard row-major encoding used throughout mathlib.
-/

/-- Flatten a `(row,col)` grid coordinate into `Fin (height * width)`. -/
def encode (pos : State width height) : Fin (height * width) :=
  (finProdFinEquiv (m := height) (n := width)) pos

/-- Unflatten a `Fin (height * width)` state index into a `(row,col)` grid coordinate. -/
def decode (i : Fin (height * width)) : State width height :=
  (finProdFinEquiv (m := height) (n := width)).symm i

/-- Deterministic finite-state discounted MDP view of GridWorld. -/
def toFiniteMDP (gw : GridWorld width height) : FiniteMDP ℝ (height * width) 4 :=
  { initialState := encode (width := width) (height := height) gw.start
    step := fun state action =>
      let out := gw.step (decode (width := width) (height := height) state) action
      { state := encode (width := width) (height := height) out.state
        reward := out.reward
        terminated := out.terminated
        truncated := out.truncated }
    discount := gw.discount }

/-- One-hot transition kernel for a deterministic next state. -/
def oneHot (next : Fin (height * width)) : Tensor ℝ [height * width] :=
  Tensor.dim (fun i => Tensor.scalar (if i = next then (1 : ℝ) else 0))

/-- Finite-stochastic MDP view of GridWorld where $P(\mathord{\cdot}\mid s,a)$ is a one-hot row. -/
def toFiniteStochasticMDP (gw : GridWorld width height) : FiniteStochastic.MDP (height * width) 4 :=
  { initialState := (toFiniteMDP (width := width) (height := height) gw).initialState
    transitionProb := fun state action =>
      let out := (toFiniteMDP (width := width) (height := height) gw).step state action
      oneHot (width := width) (height := height) out.state
    reward := fun state action =>
      ((toFiniteMDP (width := width) (height := height) gw).step state action).reward
    terminated := fun state action =>
      ((toFiniteMDP (width := width) (height := height) gw).step state action).terminated
    discount := gw.discount }

end GridWorld
end Envs
end RL
end Spec
