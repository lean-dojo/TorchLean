/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.RL.Environment

/-!
# RL Environment Proofs

These theorems capture the first "guarantee layer" for TorchLean's Gym-style environment API:

- state traces have predictable lengths,
- rollouts have predictable lengths,
- safe environments preserve invariants along valid action paths.

References:
- Gymnasium API design (reset/step, terminated vs truncated): https://gymnasium.farama.org/
- This module’s `SafeEnv` invariants are a finite-state formal analogue of the “safety wrapper”
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
    (env : Env State Action Observation Reward) (state : State) (actions : Array Action) :
    (statesFrom env state actions).size = actions.size + 1 := by
  simp [statesFrom]

/-- `states` records the initial state plus one successor per action. -/
theorem states_length
    (env : Env State Action Observation Reward) (actions : Array Action) :
    (states env actions).size = actions.size + 1 := by
  simpa [states] using
    statesFrom_length (env := env) (state := env.initialState) (actions := actions)

/-- `rolloutFrom` emits exactly one observed transition per action. -/
theorem rolloutFrom_length
    (env : Env State Action Observation Reward) (state : State) (actions : Array Action) :
    (rolloutFrom env state actions).size = actions.size := by
  simp [rolloutFrom]

/-- `rollout` emits exactly one observed transition per action. -/
theorem rollout_length
    (env : Env State Action Observation Reward) (actions : Array Action) :
    (rollout env actions).size = actions.size := by
  simpa [rollout] using
    rolloutFrom_length (env := env) (state := env.initialState) (actions := actions)

/-- An action-valid prefix preserves the environment invariant. -/
private theorem stateAfter_safe
    (env : SafeEnv State Action Observation Reward)
    (state : State) (actions : Array Action)
    (hInv : env.Invariant state)
    (hOk : env.actionPathOk state actions) :
    ∀ n, n ≤ actions.size → env.Invariant (stateAfter env.toEnv state actions n)
  | 0, _ => hInv
  | n + 1, hSize => by
      have hn : n < actions.size := by grind
      rw [stateAfter, Array.getElem?_eq_getElem hn]
      exact env.step_safe
        (stateAfter_safe env state actions hInv hOk n (by grind))
        (hOk ⟨n, hn⟩)

/-- Safe environments preserve the invariant along any valid action path. -/
theorem evolveFrom_safe
    (env : SafeEnv State Action Observation Reward)
    {state : State} {actions : Array Action}
    (hInv : env.Invariant state)
    (hOk : env.actionPathOk state actions) :
    env.Invariant (evolveFrom env.toEnv state actions) := by
  exact stateAfter_safe env state actions hInv hOk actions.size (by simp)

/-- Safe environments preserve the invariant from reset under any valid action path. -/
theorem evolve_safe
    (env : SafeEnv State Action Observation Reward)
    {actions : Array Action}
    (hOk : env.actionPathOk env.toEnv.initialState actions) :
    env.Invariant (evolve env.toEnv actions) := by
  simpa [evolve] using
    evolveFrom_safe (env := env) (state := env.toEnv.initialState) (actions := actions)
      env.init_safe hOk

/-- Every state in `statesFrom` satisfies the invariant along a valid action path. -/
theorem statesFrom_safe
    (env : SafeEnv State Action Observation Reward)
    {state : State} {actions : Array Action}
    (hInv : env.Invariant state)
    (hOk : env.actionPathOk state actions) :
    ∀ state' ∈ statesFrom env.toEnv state actions, env.Invariant state' := by
  intro state' hState
  simp only [statesFrom, Array.mem_ofFn] at hState
  obtain ⟨i, rfl⟩ := hState
  exact stateAfter_safe env state actions hInv hOk i (by grind)

/-- Every state in `states` satisfies the invariant from reset along a valid action path. -/
theorem states_safe
    (env : SafeEnv State Action Observation Reward)
    {actions : Array Action}
    (hOk : env.actionPathOk env.toEnv.initialState actions) :
    ∀ state ∈ states env.toEnv actions, env.Invariant state := by
  simpa [states] using
    statesFrom_safe (env := env) (state := env.toEnv.initialState) (actions := actions)
      env.init_safe hOk

end Environment
end RL
end Proofs
