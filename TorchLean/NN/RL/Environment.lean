/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Reinforcement-Learning Environments

A small, proof-friendly environment interface modeled on Gym/Gymnasium. `reset` returns an initial
observation and state, and `stepGym` returns `(observation, reward, terminated, truncated, state)`.
Helper functions build array-backed state traces and transition rollouts from an action sequence,
applying every action in turn without stopping at a terminated or truncated step.

The interface is purely functional: each step takes the latent state as an argument and returns its
successor, which makes safety and invariant properties easy to state and prove.

We wrote it by hand from Gymnasium's `Env.reset` and `Env.step`. Their `reset` takes `seed` and
`options` and returns `(observation, info)`. Ours takes neither and always starts from
`initialState`, Sutton and Barto's reset "to a standard starting state". A seed would have nothing
to act on, since `step` is a function and any randomness has to live in the latent state. That
state also fills the slot where Gymnasium returns `info`, in `reset` and in `stepGym`.
`StepResult.done` is Gymnasium's end-of-episode test, `terminated or truncated`. Gym v0.26
took the single `done` out of the step result in favor of those two flags, and `StepResult` keeps
both, each false by default.

References:
- Gymnasium API reference for `Env.reset` and `Env.step` (`terminated` vs `truncated`):
  https://gymnasium.farama.org/api/env/
- TorchRL documentation (environment transforms, rollout collection, and TensorDict interface):
  https://pytorch.org/rl/
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  Sections 3.1 and 3.3: http://incompleteideas.net/book/the-book-2nd.html
- Farama Foundation, "Deep Dive: Gymnasium Step API" (27 October 2023), its Summary on Gym v0.26:
  https://farama.org/Gymnasium-Terminated-Truncated-Step-API
-/

@[expose] public section

namespace TorchLean
namespace RL

universe u v w z
variable {State : Type u} {Action : Type v} {Observation : Type w} {Reward : Type z}

/-- Result of stepping an environment from one state with one action. -/
structure StepResult (State : Type u) (Reward : Type v) where
  /-- Next latent state. -/
  state : State
  /-- Immediate reward. -/
  reward : Reward
  /-- Task-defined terminal flag. -/
  terminated : Bool := false
  /-- Time-limit / truncation flag. -/
  truncated : Bool := false

/-- Gymnasium-style `done` flag: an episode is done if it is terminated or truncated. -/
def StepResult.done (r : StepResult State Reward) : Bool :=
  r.terminated || r.truncated

/-- Rollout record that stores observations on both sides of a step. -/
structure ObservedTransition (Observation : Type u) (Action : Type v) (Reward : Type w) where
  /-- Observation before the action. -/
  observation : Observation
  /-- Action taken. -/
  action : Action
  /-- Reward returned by the environment. -/
  reward : Reward
  /-- Observation after the step. -/
  nextObservation : Observation
  /-- Task-defined terminal flag. -/
  terminated : Bool
  /-- Time-limit / truncation flag. -/
  truncated : Bool

/-- Pure Gym-style environment over a latent state that each step takes and returns. -/
structure Env (State : Type u) (Action : Type v) (Observation : Type w) (Reward : Type z) where
  /-- Initial latent state used by `reset`. -/
  initialState : State
  /-- Observation function from latent states. -/
  observe : State → Observation
  /-- Single-step transition function. -/
  step : State → Action → StepResult State Reward

/-- Gym-style reset: return the initial observation and latent state. -/
def Env.reset (env : Env State Action Observation Reward) : Observation × State :=
  (env.observe env.initialState, env.initialState)

/-- Gym-style step result:
`(nextObservation, reward, terminated, truncated, nextState)`. -/
def Env.stepGym (env : Env State Action Observation Reward) (state : State) (action : Action) :
    Observation × Reward × Bool × Bool × State :=
  let out := env.step state action
  (env.observe out.state, out.reward, out.terminated, out.truncated, out.state)

/-- State reached after the first `n` actions, or after every available action when `n` is larger.
-/
def Env.stateAfter (env : Env State Action Observation Reward) (initialState : State)
    (actions : Array Action) : Nat → State
  | 0 => initialState
  | n + 1 =>
      let state := stateAfter env initialState actions n
      match actions[n]? with
      | some action => (env.step state action).state
      | none => state

/-- Final state after the actions, starting at `env.initialState` unless `state` is supplied. -/
def Env.evolve (env : Env State Action Observation Reward) (actions : Array Action)
    (state : State := env.initialState) : State :=
  stateAfter env state actions actions.size

/-- The `actions.size + 1` states visited along `actions`, beginning with `state`, which defaults
to `env.initialState`. -/
def Env.states (env : Env State Action Observation Reward) (actions : Array Action)
    (state : State := env.initialState) : Array State :=
  Array.ofFn fun i : Fin (actions.size + 1) => stateAfter env state actions i

/-- One observed transition per action, beginning at `state`, which defaults to
`env.initialState`. -/
def Env.rollout (env : Env State Action Observation Reward) (actions : Array Action)
    (state : State := env.initialState) : Array (ObservedTransition Observation Action Reward) :=
  Array.ofFn fun i : Fin actions.size =>
      let currentState := stateAfter env state actions i
      let action := actions[i]
      let out := env.step currentState action
      { observation := env.observe currentState
        action := action
        reward := out.reward
        nextObservation := env.observe out.state
        terminated := out.terminated
        truncated := out.truncated }

/-- Environment with an invariant and an action-validity predicate. -/
structure SafeEnv (State : Type u) (Action : Type v) (Observation : Type w) (Reward : Type z) where
  /-- Underlying environment dynamics. -/
  toEnv : Env State Action Observation Reward
  /-- State invariant: `init_safe` establishes it and `step_safe` preserves it along valid steps. -/
  Invariant : State → Prop
  /-- Legality / admissibility of actions at a state. -/
  ActionOk : State → Action → Prop := fun _ _ => True
  /-- Reset starts in a safe state. -/
  init_safe : Invariant toEnv.initialState
  /-- One valid step preserves the invariant. -/
  step_safe : ∀ {state action}, Invariant state → ActionOk state action →
    Invariant (toEnv.step state action).state

/-- Every action is valid in the state reached immediately before it is executed. -/
def SafeEnv.ActionPathOk (env : SafeEnv State Action Observation Reward) (state : State)
    (actions : Array Action) : Prop :=
  ∀ i : Fin actions.size,
    env.ActionOk (Env.stateAfter env.toEnv state actions i) actions[i]

end RL
end TorchLean
