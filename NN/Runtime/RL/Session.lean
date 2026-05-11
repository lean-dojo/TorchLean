/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Boundary.Core
public import NN.Runtime.RL.Gymnasium.Session
public import NN.Spec.RL.Environment

/-!
# Checked RL Sessions (Unified Runtime Interface)

TorchLean supports two “sources of experience”:

1. **External samplers** like Python Gymnasium (via `NN.Runtime.RL.Gymnasium`), and
2. **Lean-native environments** (`Spec.RL.Env`), useful for strongest end-to-end guarantees.

To avoid duplicating rollout/data-collection infrastructure per example or per algorithm, this module
defines a small **unified session interface**:

- it is stateful (has a session state type `Sess`),
- it exposes the current observation, and
- it steps with a discrete `Fin nActions` action and returns a fully-observed, contract-checked
  `Runtime.RL.Boundary.Transition`.

Algorithms (PPO, DQN-style collection, etc.) can be written against this interface and then reused
unchanged with either Gymnasium or a Lean-native environment.

Notes:
- The runtime layer returns *validated values* but does not carry Prop-level proofs. Proof-layer
  wrappers live in `NN/Proofs/RL/*` (e.g. `NN/Proofs/RL/Gymnasium.lean`).

References:
- Gymnasium API docs (reset/step, `terminated` vs `truncated`): https://gymnasium.farama.org/
- Trust-boundary contract: `NN.Runtime.RL.Boundary`.
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Session

open Spec
open Tensor

/-!
## Checked session interface
-/

/--
A stateful environment session that can produce contract-checked, fully observed transitions.

The intention is that `stepChecked`:
- uses the current observation (`observe s`) as the `observation` field,
- produces a `nextObservation`, and
- validates the whole transition against some (possibly implicit) contract before returning.
-/
structure CheckedSession (obsShape : Shape) (nActions : Nat) where
  /-- Session state type. -/
  Sess : Type
  /-- Initialize a fresh session (typically a reset). -/
  start : IO Sess
  /-- Read the current observation (before taking an action). -/
  observe : Sess → Tensor Float obsShape
  /-- One checked step. -/
  stepChecked : Sess → Fin nActions → IO (Boundary.Transition obsShape nActions × Sess)

namespace CheckedSession

/-!
## Constructors
-/

/--
Build a `CheckedSession` from an external Gymnasium client.

This session is backed by `Runtime.RL.Gymnasium.Session` and therefore:
- stores the previous observation to produce fully observed transitions, and
- validates every transition against the client's trust-boundary contract.
-/
def gymnasium {obsShape : Shape} {nActions : Nat}
    (gym : Gymnasium.Client obsShape nActions)
    (seed? : Option Nat := none)
    (resetOnDone : Bool := true) :
    CheckedSession obsShape nActions :=
  { Sess := Gymnasium.Session obsShape nActions
    start := Gymnasium.Session.start (obsShape := obsShape) (nActions := nActions) gym (seed? := seed?)
    observe := fun s => s.observation
    stepChecked := fun s a =>
      Gymnasium.Session.stepChecked (obsShape := obsShape) (nActions := nActions) s a
        (resetOnDone := resetOnDone) }

/--
Build a `CheckedSession` from a pure Lean-native environment (`Spec.RL.Env`).

Even though the dynamics are defined in Lean, we keep the same trust-boundary contract checker in
the loop. This keeps the downstream training code uniform: it consumes the same
`Spec.RL.ObservedTransition`-shaped data regardless of whether the environment is external or
internal.
-/
def ofEnv {State : Type} {obsShape : Shape} {nActions : Nat}
    (env : Spec.RL.Env State (Fin nActions) (Tensor Float obsShape) Float)
    (contract : Boundary.Contract obsShape nActions)
    (resetOnDone : Bool := true) :
    CheckedSession obsShape nActions :=
  { Sess := State
    start := pure env.initialState
    observe := env.observe
    stepChecked := fun st a => do
      let obs : Tensor Float obsShape := env.observe st
      let out := env.step st a
      let nextObs : Tensor Float obsShape := env.observe out.state
      let tr ←
        match Boundary.checkTransitionFin (obsShape := obsShape) (nActions := nActions) contract
            obs nextObs a out.reward out.terminated out.truncated with
        | .ok t => pure t
        | .error e => throw <| IO.userError e
      let done : Bool := Boundary.Transition.done tr
      let st' :=
        if resetOnDone && done then
          env.initialState
        else
          out.state
      pure (tr, st') }

end CheckedSession

end Session
end RL
end Runtime
