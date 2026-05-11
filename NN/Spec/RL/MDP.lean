/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Finset.Lattice.Fold
public import NN.Spec.Core.TensorOps
public import NN.Spec.RL.Core
public import NN.Spec.RL.Environment

/-!
# Finite Discounted MDPs

This module gives TorchLean a small, proof-friendly finite-state discounted MDP layer.

Design choices:

- transitions are deterministic and total,
- the latent state space is `Fin nStates`,
- the action space is `Fin nActions`,
- Bellman operators are defined directly on typed value tensors.

This keeps the first formalization manageable while still supporting the core objects used by RL
theory: policies, value functions, state-action values, and Bellman operators.

References:

- Bellman, *Dynamic Programming* (1957)
- Puterman, *Markov Decision Processes* (1994)
- Sutton and Barto, *Reinforcement Learning: An Introduction*
- Gymnasium and TorchRL are useful runtime reference points, but this file intentionally stays at
  the pure finite-MDP semantics level rather than modeling replay buffers or collectors.

Naming note:

- In this namespace, `FiniteMDP`, `Policy`, `ValueFunction`, and the Bellman operators refer to
  deterministic finite tensor MDPs.
- `Spec.RL.FiniteStochastic` and `Spec.RL.Markov` deliberately reuse standard RL words such as `MDP`
  and `Policy` inside their own namespaces. We keep the short mathematical names there because
  the fully qualified names already say which semantic layer is being used.
-/

@[expose] public section

namespace Spec
namespace RL

open Tensor

variable {α : Type}
variable {nStates nActions : Nat}

/-- Value function over a finite state space. -/
abbrev ValueFunction (α : Type) (nStates : Nat) := Tensor α (.dim nStates .scalar)

/-- Deterministic policy over a finite state / action space. -/
abbrev Policy (nStates nActions : Nat) := Fin nStates → Fin nActions

/-- Finite discounted MDP with deterministic transitions. -/
structure FiniteMDP (α : Type) (nStates nActions : Nat) where
  /-- Canonical reset state. -/
  initialState : Fin nStates
  /-- One-step deterministic transition / reward dynamics. -/
  step : Fin nStates → Fin nActions → StepResult (Fin nStates) α
  /-- Discount factor used by Bellman operators. -/
  discount : α

/-- View a finite MDP as a Gym-style environment with observations equal to latent states. -/
def FiniteMDP.toEnv (mdp : FiniteMDP α nStates nActions) :
    Env (Fin nStates) (Fin nActions) (Fin nStates) α :=
  { initialState := mdp.initialState
    observe := id
    step := mdp.step }

/-- Lookup a state's value. -/
def valueAt {nStates : Nat} (values : ValueFunction α nStates) (state : Fin nStates) : α :=
  Tensor.vecGet values state

/-- One-step state-action value induced by a candidate value function. -/
def stateActionValue [Zero α] [One α] [Add α] [Mul α]
    (mdp : FiniteMDP α nStates nActions)
    (values : ValueFunction α nStates)
    (state : Fin nStates)
    (action : Fin nActions) : α :=
  let out := mdp.step state action
  discountedBackup (α := α) out.reward mdp.discount (valueAt values out.state) out.terminated

/-- All state-action values `Q_v(s, ·)` for a fixed state and candidate value function. -/
def actionValues [Zero α] [One α] [Add α] [Mul α]
    (mdp : FiniteMDP α nStates nActions)
    (values : ValueFunction α nStates)
    (state : Fin nStates) : Tensor α (.dim nActions .scalar) :=
  Tensor.dim (fun action => Tensor.scalar (stateActionValue (α := α) mdp values state action))

/-- Bellman operator for a deterministic policy. -/
def bellmanPolicy [Zero α] [One α] [Add α] [Mul α]
    (mdp : FiniteMDP α nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction α nStates) : ValueFunction α nStates :=
  Tensor.dim (fun state =>
    Tensor.scalar (stateActionValue (α := α) mdp values state (policy state)))

/-- Bellman optimality operator for a finite action space. -/
def bellmanOptimality [Zero α] [One α] [Add α] [Mul α] [LinearOrder α]
    [Fact (0 < nActions)]
    (mdp : FiniteMDP α nStates nActions)
    (values : ValueFunction α nStates) : ValueFunction α nStates :=
  let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  Tensor.dim (fun state =>
    Tensor.scalar
      ((Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty
        (stateActionValue (α := α) mdp values state)))

end RL
end Spec
