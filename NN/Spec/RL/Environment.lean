/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.List.Basic

/-!
# Reinforcement-Learning Environments

This module provides a small, proof-friendly environment interface inspired by Gym/Gymnasium:

- `reset` returns an initial observation and state,
- `stepGym` returns `(observation, reward, terminated, truncated, state)`,
- helper functions build state traces and transition rollouts from an action sequence.

Unlike a typical Python RL environment, this interface is purely functional and keeps the hidden
state explicit. That makes it much easier to state and prove safety / invariant properties.

References:
- Gymnasium API docs (reset/step, `terminated` vs `truncated`):
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

/-- Final latent state reached after executing a list of actions. -/
def evolveFrom (env : Env State Action Observation Reward) : State → List Action → State
  | state, [] => state
  | state, action :: actions =>
      evolveFrom env (env.step state action).state actions

/-- Final latent state reached from the environment's initial state. -/
def evolve (env : Env State Action Observation Reward) (actions : List Action) : State :=
  evolveFrom env env.initialState actions

/-- State trace that records the initial state and every successor state. -/
def statesFrom (env : Env State Action Observation Reward) : State → List Action → List State
  | state, [] => [state]
  | state, action :: actions =>
      state :: statesFrom env (env.step state action).state actions

/-- State trace from the environment's initial state. -/
def states (env : Env State Action Observation Reward) (actions : List Action) : List State :=
  statesFrom env env.initialState actions

/-- Observed transition rollout from an explicit initial state. -/
def rolloutFrom (env : Env State Action Observation Reward) : State → List Action →
    List (ObservedTransition Observation Action Reward)
  | _state, [] => []
  | state, action :: actions =>
      let out := env.step state action
      { observation := env.observe state
        action := action
        reward := out.reward
        nextObservation := env.observe out.state
        terminated := out.terminated
        truncated := out.truncated } :: rolloutFrom env out.state actions

/-- Observed transition rollout from the environment's initial state. -/
def rollout (env : Env State Action Observation Reward) (actions : List Action) :
    List (ObservedTransition Observation Action Reward) :=
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

/-- Path validity for a list of actions from a given starting state. -/
def SafeEnv.actionPathOk (env : SafeEnv State Action Observation Reward) :
    State → List Action → Prop
  | _state, [] => True
  | state, action :: actions =>
      env.ActionOk state action ∧
        env.actionPathOk (env.toEnv.step state action).state actions

/--
If a path is action-valid and starts from an invariant state, then its final state is invariant.

This is the basic safety theorem users want from `SafeEnv`: once we specify the one-step invariant,
valid rollouts inherit it automatically.
-/
theorem SafeEnv.evolveFrom_safe (env : SafeEnv State Action Observation Reward) :
    ∀ (state : State) (actions : List Action),
      env.Invariant state →
      env.actionPathOk state actions →
      env.Invariant (_root_.Spec.RL.evolveFrom env.toEnv state actions)
  | state, [], hInv, _ => hInv
  | state, action :: actions, hInv, hPath => by
      exact env.evolveFrom_safe
        (env.toEnv.step state action).state
        actions
        (env.step_safe hInv hPath.1)
        hPath.2

/-- Any valid action path from reset remains safe at the final state. -/
theorem SafeEnv.evolve_safe (env : SafeEnv State Action Observation Reward)
    (actions : List Action)
    (hPath : env.actionPathOk env.toEnv.initialState actions) :
    env.Invariant (_root_.Spec.RL.evolve env.toEnv actions) := by
  exact env.evolveFrom_safe env.toEnv.initialState actions env.init_safe hPath

end RL
end Spec
