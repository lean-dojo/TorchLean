/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.List.Basic
public import NN.Spec.RL.Environment

/-!
# RL Environment Proofs

These theorems capture the first "guarantee layer" for TorchLean's Gym-style environment API:

- state traces have predictable lengths,
- rollouts have predictable lengths,
- safe environments preserve invariants along valid action paths.

References:
- Gymnasium API design (reset/step, terminated vs truncated): https://gymnasium.farama.org/
- This module’s `SafeEnv` invariants are a lightweight formal analogue of the “safety wrapper”
  patterns used in practical RL systems.
-/

@[expose] public section

namespace Proofs
namespace RL
namespace Environment

open Spec.RL
universe u v w z
variable {State : Type u} {Action : Type v} {Observation : Type w} {Reward : Type z}

/-- `statesFrom` records the initial state plus one state per action. -/
theorem statesFrom_length
    (env : Env State Action Observation Reward) (state : State) (actions : List Action) :
    (statesFrom env state actions).length = actions.length + 1 := by
  induction actions generalizing state with
  | nil =>
      simp [statesFrom]
  | cons action actions ih =>
      simp [statesFrom, ih, Nat.add_assoc]

/-- `states` records the initial state plus one successor per action. -/
theorem states_length
    (env : Env State Action Observation Reward) (actions : List Action) :
    (states env actions).length = actions.length + 1 := by
  simpa [states] using statesFrom_length (env := env) (state := env.initialState) (actions := actions)

/-- `rolloutFrom` emits exactly one observed transition per action. -/
theorem rolloutFrom_length
    (env : Env State Action Observation Reward) (state : State) (actions : List Action) :
    (rolloutFrom env state actions).length = actions.length := by
  induction actions generalizing state with
  | nil =>
      simp [rolloutFrom]
  | cons action actions ih =>
      simp [rolloutFrom, ih]

/-- `rollout` emits exactly one observed transition per action. -/
theorem rollout_length
    (env : Env State Action Observation Reward) (actions : List Action) :
    (rollout env actions).length = actions.length := by
  simpa [rollout] using rolloutFrom_length (env := env) (state := env.initialState) (actions := actions)

/-- Safe environments preserve the invariant along any valid action path. -/
theorem evolveFrom_safe
    (env : SafeEnv State Action Observation Reward)
    {state : State} {actions : List Action}
    (hInv : env.Invariant state)
    (hOk : env.actionPathOk state actions) :
    env.Invariant (evolveFrom env.toEnv state actions) := by
  induction actions generalizing state with
  | nil =>
      simpa [evolveFrom] using hInv
  | cons action actions ih =>
      have hAction : env.ActionOk state action := hOk.1
      have hNext : env.Invariant (env.toEnv.step state action).state :=
        env.step_safe hInv hAction
      have hTail : env.actionPathOk (env.toEnv.step state action).state actions := hOk.2
      simpa [evolveFrom] using ih hNext hTail

/-- Safe environments preserve the invariant from reset under any valid action path. -/
theorem evolve_safe
    (env : SafeEnv State Action Observation Reward)
    {actions : List Action}
    (hOk : env.actionPathOk env.toEnv.initialState actions) :
    env.Invariant (evolve env.toEnv actions) := by
  simpa [evolve] using
    evolveFrom_safe (env := env) (state := env.toEnv.initialState) (actions := actions) env.init_safe hOk

/-- Every state in `statesFrom` satisfies the invariant along a valid action path. -/
theorem statesFrom_safe
    (env : SafeEnv State Action Observation Reward)
    {state : State} {actions : List Action}
    (hInv : env.Invariant state)
    (hOk : env.actionPathOk state actions) :
    List.Forall env.Invariant (statesFrom env.toEnv state actions) := by
  induction actions generalizing state with
  | nil =>
      simp [statesFrom, hInv]
  | cons action actions ih =>
      have hAction : env.ActionOk state action := hOk.1
      have hNext : env.Invariant (env.toEnv.step state action).state :=
        env.step_safe hInv hAction
      have hTail : env.actionPathOk (env.toEnv.step state action).state actions := hOk.2
      simpa [List.Forall, statesFrom, hInv] using ih hNext hTail

/-- Every state in `states` satisfies the invariant from reset along a valid action path. -/
theorem states_safe
    (env : SafeEnv State Action Observation Reward)
    {actions : List Action}
    (hOk : env.actionPathOk env.toEnv.initialState actions) :
    List.Forall env.Invariant (states env.toEnv actions) := by
  simpa [states] using
    statesFrom_safe (env := env) (state := env.toEnv.initialState) (actions := actions) env.init_safe hOk

end Environment
end RL
end Proofs
