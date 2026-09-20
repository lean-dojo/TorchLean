/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Reinforcement-Learning Environments

This module provides a small, proof-friendly environment interface inspired by Gym/Gymnasium:

- `reset` returns an initial observation and state,
- `stepGym` returns `(observation, reward, terminated, truncated, state)`,
- helper functions build array-backed state traces and transition rollouts from an action sequence.

Unlike a typical Python RL environment, this interface is purely functional and keeps the hidden
state explicit. That makes it much easier to state and prove safety / invariant properties.

References:
- Gymnasium API reference (reset/step, `terminated` vs `truncated`):
  https://gymnasium.farama.org/
- TorchRL documentation (environment transforms, rollout collection, and TensorDict interface):
  https://pytorch.org/rl/
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.):
  http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace Spec
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

/-- Pure Gym-style environment with explicit latent state. -/
structure Env (State : Type u) (Action : Type v) (Observation : Type w) (Reward : Type z) where
  /-- Initial latent state used by `reset`. -/
  initialState : State
  /-- Observation function from latent states. -/
  observe : State → Observation
  /-- Single-step transition function. -/
  step : State → Action → StepResult State Reward

/-- Gym-style reset: return the initial observation and latent state. -/
def reset (env : Env State Action Observation Reward) : Observation × State :=
  (env.observe env.initialState, env.initialState)

/-- Gym-style step result:
`(nextObservation, reward, terminated, truncated, nextState)`. -/
def stepGym (env : Env State Action Observation Reward) (state : State) (action : Action) :
    Observation × Reward × Bool × Bool × State :=
  let out := env.step state action
  (env.observe out.state, out.reward, out.terminated, out.truncated, out.state)

/-- State reached after the first `n` actions, or after every available action when `n` is larger.
-/
def stateAfter (env : Env State Action Observation Reward) (initialState : State)
    (actions : Array Action) : Nat → State
  | 0 => initialState
  | n + 1 =>
      let state := stateAfter env initialState actions n
      match actions[n]? with
      | some action => (env.step state action).state
      | none => state

/-- Final latent state reached after executing an array of actions. -/
def evolveFrom (env : Env State Action Observation Reward) (state : State)
    (actions : Array Action) : State :=
  stateAfter env state actions actions.size

/-- Final latent state reached from the environment's initial state. -/
def evolve (env : Env State Action Observation Reward) (actions : Array Action) : State :=
  evolveFrom env env.initialState actions

/-- State trace that records the initial state and every successor state. -/
def statesFrom (env : Env State Action Observation Reward) (state : State)
    (actions : Array Action) : Array State :=
  Array.ofFn fun i : Fin (actions.size + 1) => stateAfter env state actions i

/-- State trace from the environment's initial state. -/
def states (env : Env State Action Observation Reward) (actions : Array Action) : Array State :=
  statesFrom env env.initialState actions

/-- Observed transition rollout from an explicit initial state. -/
def rolloutFrom (env : Env State Action Observation Reward) (state : State)
    (actions : Array Action) : Array (ObservedTransition Observation Action Reward) :=
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

/-- Observed transition rollout from the environment's initial state. -/
def rollout (env : Env State Action Observation Reward) (actions : Array Action) :
    Array (ObservedTransition Observation Action Reward) :=
  rolloutFrom env env.initialState actions

/-- Environment with an invariant and an action-validity predicate. -/
structure SafeEnv (State : Type u) (Action : Type v) (Observation : Type w) (Reward : Type z) where
  /-- Underlying environment dynamics. -/
  toEnv : Env State Action Observation Reward
  /-- State invariant we want preserved along valid executions. -/
  Invariant : State → Prop
  /-- Legality / admissibility of actions at a state. -/
  ActionOk : State → Action → Prop := fun _ _ => True
  /-- Reset starts in a safe state. -/
  init_safe : Invariant toEnv.initialState
  /-- One valid step preserves the invariant. -/
  step_safe : ∀ {state action}, Invariant state → ActionOk state action →
    Invariant (toEnv.step state action).state

/-- Every action is valid in the state reached immediately before it is executed. -/
def SafeEnv.actionPathOk (env : SafeEnv State Action Observation Reward) (state : State)
    (actions : Array Action) : Prop :=
  ∀ i : Fin actions.size,
    env.ActionOk (stateAfter env.toEnv state actions i) actions[i]

end RL
end Spec
