/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Gymnasium.Session

/-!
# Checked RL Sessions (Unified Runtime Interface)

Experience comes from two kinds of source: external samplers such as Python Gymnasium
(`TorchLean.NN.RL.Gymnasium.Session`), and Lean-native environments (`RL.Env`), which give the
strongest end-to-end guarantees. `Session.Checked` is one interface over both. A session has a state
type `State`, exposes the current observation, and steps with a `Fin nActions` action to a fully
observed, contract-checked `RL.Boundary.Transition`. Rollout and collection code (PPO, DQN-style
collection) written against it runs unchanged on either source.

The runtime layer returns validated values without Prop-level proofs. The proof layer's
`RL.Boundary.checkTransitionFinWithProof` (`TorchLean/NN/RL/Boundary/Soundness.lean`) checks a
transition and returns it with its contract proof, and `Boundary.stepChecked` accepts it as its
checker.

References:
- Gymnasium API reference (reset/step, `terminated` vs `truncated`): https://gymnasium.farama.org/
- Trust-boundary contract: `TorchLean.NN.RL.Boundary`.
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace Session

open TorchLean.Tensor

/-!
## Checked session interface
-/

/--
A stateful environment session whose steps return contract-checked, fully observed transitions.

By convention `stepChecked s a` uses `observe s` as the transition's `observation`, steps to a
`nextObservation`, and validates the whole transition against a contract before returning. The
fields carry no proof of this.
-/
structure Checked (obsShape : Shape) (nActions : Nat) where
  /-- Session state type. -/
  State : Type
  /-- Initialize a fresh session (typically a reset). -/
  start : IO State
  /-- Read the current observation (before taking an action). -/
  observe : State → Tensor Float obsShape
  /-- Take one action and return the checked transition with the next session state. -/
  step : State → Fin nActions → IO (Boundary.Transition obsShape nActions × State)

namespace Checked

/-!
## Constructors
-/

/--
A `Checked` over an external Gymnasium client, backed by `RL.Gymnasium.Session`.

The session stores the previous observation, so each step yields a fully observed transition,
and it validates every transition against the client's contract. `seed?` seeds the reset of every
`start`, and the resets after `done` send no seed. With `resetOnDone` (the default), a step
that ends an episode resets the environment.
-/
def gymnasium {obsShape : Shape} {nActions : Nat}
    (gym : Gymnasium.Client obsShape nActions)
    (seed? : Option Nat := none)
    (resetOnDone : Bool := true) :
    Checked obsShape nActions :=
  { State := Gymnasium.Session obsShape nActions
    start :=
      Gymnasium.Session.start (obsShape := obsShape) (nActions := nActions) gym (seed? := seed?)
    observe := fun s => s.observation
    step := fun s a =>
      Gymnasium.Session.stepChecked (obsShape := obsShape) (nActions := nActions) s a
        (resetOnDone := resetOnDone) }

/--
A `Checked` over a pure Lean-native environment (`RL.Env`).

The dynamics are Lean code, but every step goes through the same contract checker as an
external rollout, so training consumes the same `RL.ObservedTransition` data from both
kinds of environment. A reset on `done` returns to `env.initialState`.
-/
def ofEnv {State : Type} {obsShape : Shape} {nActions : Nat}
    (env : RL.Env State (Fin nActions) (Tensor Float obsShape) Float)
    (contract : Boundary.Contract obsShape nActions)
    (resetOnDone : Bool := true) :
    Checked obsShape nActions :=
  { State := State
    start := pure env.initialState
    observe := env.observe
    step :=
      Boundary.stepChecked env.observe
        (fun st a =>
          let out := env.step st a
          pure (env.observe out.state, out.reward, out.terminated, out.truncated, out.state))
        (fun _ => pure env.initialState) (Boundary.checkTransitionFin contract)
        Boundary.Transition.done resetOnDone }

end Checked

end Session
end RL
end TorchLean
