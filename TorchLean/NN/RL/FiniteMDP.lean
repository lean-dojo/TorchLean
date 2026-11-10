/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL
public import TorchLean.NN.RL.Environment

/-!
# Finite Discounted MDPs

A small, proof-friendly finite-state discounted MDP layer. Transitions are deterministic and
total, latent states are `Fin nStates` and actions `Fin nActions`, and the Bellman operators act on
typed value tensors. The layer covers the core objects of RL theory: policies, value functions,
state-action values and Bellman operators.

We wrote it by hand from the finite MDPs of Sutton and Barto, with one `StepResult` per state and
action where their dynamics $p(s', r \mid s, a)$ are a distribution. `actionValue` reads
`terminated` and ignores `truncated`, so a step cut off by a time limit still bootstraps from the
next state's value. `bellmanMax` asks for `Fact (0 < nActions)` and a `LinearOrder` on the
scalars, since `Finset.sup'` needs a nonempty set to take its maximum over.

References:

- Bellman, *Dynamic Programming* (Princeton University Press, 1957)
- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (Wiley, 1994),
  Chapter 6: https://doi.org/10.1002/9780470316887
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  Chapter 3, eq. (3.2) for the dynamics: http://incompleteideas.net/book/the-book-2nd.html
- Gymnasium and TorchRL are runtime reference points. The file defines the pure finite-MDP
  semantics and models no replay buffers or collectors.

Naming note:

- In this namespace, `FiniteMDP` and the Bellman operators refer to deterministic finite tensor
  MDPs. `Policy` and `ValueFunction` are the finite policy and value types, which
  `RL.StochasticMDP` shares.
- `RL.StochasticMDP` is the stochastic finite MDP and `RL.MDP` the Markov-kernel MDP. The
  namespace `RL.MDP` has its own `Policy` and `ValueFunction`. They keep the short mathematical
  names because the fully qualified names say which semantic layer is in use.
-/

@[expose] public section

namespace TorchLean
namespace RL

open TorchLean.Tensor

variable {α : Type} [Storage α]
variable {nStates nActions : Nat}

/-- Value function over a finite state space. -/
abbrev ValueFunction (α : Type) [Storage α] (nStates : Nat) :=
  Tensor α [nStates]

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

/-- One-step state-action value induced by a candidate value function. -/
def FiniteMDP.actionValue [Add α] [Mul α]
    (mdp : FiniteMDP α nStates nActions)
    (values : ValueFunction α nStates)
    (state : Fin nStates)
    (action : Fin nActions) : α :=
  let out := mdp.step state action
  discountedBackup (α := α) out.reward mdp.discount (Tensor.getScalar values out.state)
    out.terminated

/-- All state-action values $Q_v(s,\mathord{\cdot})$ for a fixed state and candidate value
function. -/
def FiniteMDP.actionValues [Add α] [Mul α]
    (mdp : FiniteMDP α nStates nActions)
    (values : ValueFunction α nStates)
    (state : Fin nStates) : Tensor α [nActions] :=
  Tensor.ofFn (fun action => actionValue (α := α) mdp values state action)

/-- Bellman operator for a deterministic policy. -/
def FiniteMDP.bellmanPolicy [Add α] [Mul α]
    (mdp : FiniteMDP α nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction α nStates) : ValueFunction α nStates :=
  Tensor.ofFn (fun state =>
    actionValue (α := α) mdp values state (policy state))

/-- Bellman optimality operator for a finite action space. -/
def FiniteMDP.bellmanMax [Add α] [Mul α] [LinearOrder α]
    [Fact (0 < nActions)]
    (mdp : FiniteMDP α nStates nActions)
    (values : ValueFunction α nStates) : ValueFunction α nStates :=
  let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  Tensor.ofFn (fun state =>
    (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty
      (actionValue (α := α) mdp values state))

end RL
end TorchLean
