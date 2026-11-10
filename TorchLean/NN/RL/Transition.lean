/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- Downstream RL modules reach `TorchLean.NN.RL` through this module, so keep its import.
module -- shake: keep-downstream

public import TorchLean.NN.RL -- shake: keep

/-!
# Core Reinforcement-Learning Runtime Helpers

The runtime RL layer adds two things to the mathematical core in `TorchLean.NN.RL`: a typed
transition record for tensor-valued states and discrete actions, and the squared error used by
critic and TD objectives (the Huber loss is `TorchLean.Huber.term`). Bellman backups, returns and
GAE stay in the spec layer, so runtime code and proofs share one definition of each.
-/

@[expose] public section

namespace TorchLean
namespace RL

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- A typed one-step transition for discrete-action RL over a tensor-valued state. -/
structure Transition (α : Type) [Storage α]
    (σ : Shape) (nActions : Nat) where
  /-- Current state `s_t`. -/
  state : Tensor α σ
  /-- Discrete action `a_t`. -/
  action : Fin nActions
  /-- Reward `r_t`. -/
  reward : α
  /-- Next state `s_{t+1}`. -/
  nextState : Tensor α σ
  /-- Whether `nextState` is terminal. TD targets drop the bootstrap term when this is set, so
  set it only for true termination and leave it `false` after a time-limit truncation. -/
  done : Bool

/-- Squared error `d * d` with `d = prediction - target`, for critic and TD objectives. -/
def squaredError (prediction target : α) : α :=
  let d := prediction - target
  d * d

end RL
end TorchLean
