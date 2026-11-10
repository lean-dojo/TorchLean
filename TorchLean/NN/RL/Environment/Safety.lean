/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Environment

/-!
# RL Environment Proofs

Basic facts about TorchLean's Gym-style environment API: a state trace has `actions.size + 1`
entries, a rollout has one transition per action, and a `SafeEnv` keeps its invariant along every
valid action path.

We proved these by hand. The two size facts unfold the definitions, and the safety facts go by
induction on the length of the action prefix through `stateAfter`. `ActionOk` plays the part of
the set of safe actions that a preemptive shield hands the agent at every step in Alshiekh et al.
Their shield is synthesized from a temporal-logic specification. A `SafeEnv` takes the invariant
and `ActionOk` from its author, and `step_safe` is the obligation that ties the two together.

References:
- Gymnasium API design (reset/step, terminated vs truncated): https://gymnasium.farama.org/
- Alshiekh, Bloem, Ehlers, Könighofer, Niekum and Topcu, "Safe Reinforcement Learning via
  Shielding", AAAI 2018, https://doi.org/10.1609/aaai.v32i1.11797. Preemptive shielding is
  Section 5.1 of arXiv:1708.08611: https://arxiv.org/abs/1708.08611
-/

@[expose] public section

namespace TorchLean
namespace RL

open Env
universe u v w z
variable {State : Type u} {Action : Type v} {Observation : Type w} {Reward : Type z}

/-- `states` records the initial state plus one state per action. -/
theorem Env.size_states
    (env : Env State Action Observation Reward) (actions : Array Action)
    (state : State := env.initialState) :
    (states env actions (state := state)).size = actions.size + 1 := by
  simp [states]

/-- `rollout` emits exactly one observed transition per action. -/
theorem Env.size_rollout
    (env : Env State Action Observation Reward) (actions : Array Action)
    (state : State := env.initialState) :
    (rollout env actions (state := state)).size = actions.size := by
  simp [rollout]

/-- From an invariant state, every prefix of a valid action path ends in an invariant state. -/
private theorem SafeEnv.stateAfter_safe
    (env : SafeEnv State Action Observation Reward)
    (state : State) (actions : Array Action)
    (hInv : env.Invariant state)
    (hOk : env.ActionPathOk state actions) :
    ∀ n, n ≤ actions.size → env.Invariant (stateAfter env.toEnv state actions n)
  | 0, _ => hInv
  | n + 1, hSize => by
      have hn : n < actions.size := by grind
      rw [stateAfter, Array.getElem?_eq_getElem hn]
      exact env.step_safe
        (stateAfter_safe env state actions hInv hOk n (by grind))
        (hOk ⟨n, hn⟩)

/-- Safe environments preserve the invariant along any valid action path. -/
theorem SafeEnv.evolve_safe_of_invariant
    (env : SafeEnv State Action Observation Reward)
    {state : State} {actions : Array Action}
    (hInv : env.Invariant state)
    (hOk : env.ActionPathOk state actions) :
    env.Invariant (evolve env.toEnv actions (state := state)) := by
  exact stateAfter_safe env state actions hInv hOk actions.size (by simp)

/-- Safe environments preserve the invariant from reset under any valid action path. -/
theorem SafeEnv.evolve_safe
    (env : SafeEnv State Action Observation Reward)
    {actions : Array Action}
    (hOk : env.ActionPathOk env.toEnv.initialState actions) :
    env.Invariant (evolve env.toEnv actions) := by
  simpa [evolve] using
    evolve_safe_of_invariant (env := env) (state := env.toEnv.initialState) (actions := actions)
      env.init_safe hOk

/-- Every state in `states` satisfies the invariant along a valid action path. -/
theorem SafeEnv.states_safe_of_invariant
    (env : SafeEnv State Action Observation Reward)
    {state : State} {actions : Array Action}
    (hInv : env.Invariant state)
    (hOk : env.ActionPathOk state actions) :
    ∀ state' ∈ states env.toEnv actions (state := state), env.Invariant state' := by
  intro state' hState
  simp only [states, Array.mem_ofFn] at hState
  obtain ⟨i, rfl⟩ := hState
  exact stateAfter_safe env state actions hInv hOk i (by grind)

/-- Every state in `states` satisfies the invariant from reset along a valid action path. -/
theorem SafeEnv.states_safe
    (env : SafeEnv State Action Observation Reward)
    {actions : Array Action}
    (hOk : env.ActionPathOk env.toEnv.initialState actions) :
    ∀ state ∈ states env.toEnv actions, env.Invariant state := by
  simpa [states] using
    states_safe_of_invariant (env := env) (state := env.toEnv.initialState) (actions := actions)
      env.init_safe hOk

end RL
end TorchLean
