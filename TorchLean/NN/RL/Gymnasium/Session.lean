/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Gymnasium.Client

/-!
# Gymnasium Bridge (Session)

Stepping an external Gymnasium environment yields only the *next* observation, while a full
Gym-style transition needs `observation` before the action and `nextObservation` after it.
`Session` stores the last observation, so `stepChecked` can return a fully observed,
contract-checked transition (`TorchLean.RL.Boundary.Transition`).

Executable RL workflows step Gymnasium through this session, by itself or wrapped in
`RL.Session.Checked.gymnasium`, so trust-boundary validation stays in one place.

We wrote the session by hand. With `resetOnDone` set it resets in the step that ends the episode,
like the same-step autoreset of Gymnasium's vector environments. The two differ in where the
episode's last observation goes. Ours keeps it in the transition as `nextObservation` and stores
the reset observation in the session, while Gymnasium's `step` returns the reset observation and
puts the last one in `info["final_obs"]`.

References:
- Gymnasium API reference (`reset`/`step`, `terminated` vs `truncated`):
  https://gymnasium.farama.org/
- Brockman et al., "OpenAI Gym" (2016), arXiv:1606.01540, background on the env interface:
  https://arxiv.org/abs/1606.01540
- Towers, "Deep Dive: VectorEnv Autoreset" (Farama Foundation, 20 February 2025), the same-step
  mode: https://farama.org/Vector-Autoreset-Mode
- Trust-boundary contract definition: `TorchLean.NN.RL.Boundary`.
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace Gymnasium

open TorchLean.Tensor

/-!
## Stateful session (validated transitions)
-/

/--
Stateful Gymnasium session that stores the most recent observation, the state a step needs to emit
a *fully observed* transition.
-/
structure Session (obsShape : Shape) (nActions : Nat) where
  /-- Subprocess client used to communicate with Python Gymnasium. -/
  client : Client obsShape nActions
  /-- Current observation (the one to be used as `observation` on the next step). -/
  observation : Tensor Float obsShape

namespace Session

/-- Create a session from one reset of the environment, seeded by `seed?` when given. -/
def start {obsShape : Shape} {nActions : Nat}
    (client : Client obsShape nActions) (seed? : Option Nat := none) :
    IO (Session obsShape nActions) := do
  let obs ← client.reset (seed? := seed?)
  pure { client := client, observation := obs }

/-- Reset and replace the stored observation. -/
def reset {obsShape : Shape} {nActions : Nat}
    (s : Session obsShape nActions) (seed? : Option Nat := none) :
    IO (Session obsShape nActions) := do
  let obs ← s.client.reset (seed? := seed?)
  pure { s with observation := obs }

/--
Step once, validate the transition against the client's contract, and reset on `done` when
`resetOnDone` is set.

The reset sends no seed. `resetOnDone = true` suits fixed-horizon rollouts that keep collecting
across episode boundaries.
-/
def stepChecked {obsShape : Shape} {nActions : Nat}
    (s : Session obsShape nActions) (action : Fin nActions) (resetOnDone : Bool := true) :
    IO (Boundary.Transition obsShape nActions × Session obsShape nActions) :=
  Boundary.stepChecked (fun s : Session obsShape nActions => s.observation)
    (fun s action => do
      let (obs', reward, terminated, truncated) ← Client.step s.client action.1
      pure (obs', reward, terminated, truncated, { s with observation := obs' }))
    (fun s => reset s) (Boundary.checkTransitionFin s.client.contract) Boundary.Transition.done
    resetOnDone s action

/-- Close the underlying client/subprocess. -/
def close {obsShape : Shape} {nActions : Nat} (s : Session obsShape nActions) : IO Unit :=
  s.client.close

end Session

end Gymnasium
end RL
end TorchLean
