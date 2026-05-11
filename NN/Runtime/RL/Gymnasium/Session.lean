/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Gymnasium.Client

/-!
# Gymnasium Bridge (Session)

Stepping an external Gymnasium environment yields only the *next* observation. To validate a full
Gym-style transition we need both:

- `observation` before the action, and
- `nextObservation` after the action.

`Session` stores the last observation so `stepChecked` can return a fully-observed,
contract-checked transition (`Runtime.RL.Boundary.Transition`).

This is the main entry point used by executable RL workflows: it is small, typed, and keeps
trust-boundary validation in one place.

References:
- Gymnasium API docs (`reset`/`step`, `terminated` vs `truncated`): https://gymnasium.farama.org/
- The original Gym API paper (background on the env interface): https://arxiv.org/abs/1606.01540
- Trust-boundary contract definition: `NN.Runtime.RL.Boundary`.
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Gymnasium

open Spec
open Tensor

/-!
## Stateful session (validated transitions)
-/

/--
Stateful Gymnasium session that stores the most recent observation.

This is the state required to emit a *fully observed* transition on each step.
-/
structure Session (obsShape : Shape) (nActions : Nat) where
  /-- Subprocess client used to communicate with Python Gymnasium. -/
  client : Client obsShape nActions
  /-- Current observation (the one to be used as `observation` on the next step). -/
  observation : Tensor Float obsShape

namespace Session

/-- Create a session by resetting the environment once. -/
def start {obsShape : Shape} {nActions : Nat}
    (client : Client obsShape nActions) (seed? : Option Nat := none) :
    IO (Session obsShape nActions) := do
  let obs ← client.reset (seed? := seed?)
  pure { client := client, observation := obs }

/-- Spawn a client + start a session, ensuring the subprocess is closed after `k` returns. -/
def withSession {α : Type} {obsShape : Shape} {nActions : Nat}
    (serverScript envId : String)
    (contract : Boundary.Contract obsShape nActions)
    (seed? : Option Nat := none)
    (k : Session obsShape nActions → IO α) : IO α := do
  Client.withClient (obsShape := obsShape) (nActions := nActions) serverScript envId contract
    (fun g => do
      let s ← start (obsShape := obsShape) (nActions := nActions) g (seed? := seed?)
      k s)

/-- Reset and replace the stored observation. -/
def reset {obsShape : Shape} {nActions : Nat}
    (s : Session obsShape nActions) (seed? : Option Nat := none) :
    IO (Session obsShape nActions) := do
  let obs ← s.client.reset (seed? := seed?)
  pure { s with observation := obs }

/--
Step once, validate against the trust-boundary contract, and optionally auto-reset on `done`.

`resetOnDone=true` is convenient for fixed-horizon rollouts where we want to keep collecting even
across episode boundaries.
-/
def stepChecked {obsShape : Shape} {nActions : Nat}
    (s : Session obsShape nActions) (action : Fin nActions) (resetOnDone : Bool := true) :
    IO (Boundary.Transition obsShape nActions × Session obsShape nActions) := do
  let obs := s.observation
  let (obs', reward, terminated, truncated) ← s.client.stepRaw (action := action.1)
  let tr ←
    match Boundary.checkTransitionFin (obsShape := obsShape) (nActions := nActions) s.client.contract
        obs obs' action reward terminated truncated with
    | .ok t => pure t
    | .error e => throw <| IO.userError e
  let done : Bool := Boundary.Transition.done tr
  let nextObs ←
    if resetOnDone && done then
      s.client.reset
    else
      pure obs'
  pure (tr, { s with observation := nextObs })

/-- Close the underlying client/subprocess. -/
def close {obsShape : Shape} {nActions : Nat} (s : Session obsShape nActions) : IO Unit :=
  s.client.close

end Session

end Gymnasium
end RL
end Runtime
