/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.MeasureTheory.Integral.Bochner.Basic
public import Mathlib.Probability.Kernel.Defs
public import TorchLean.NN.RL

/-!
# Measure-Theoretic Discounted MDPs (Markov Kernels)

A small, proof-friendly discounted MDP interface on **general measurable** state and action
spaces, built on mathlib's Markov kernels.

The finite layers `TorchLean.NN.RL.FiniteMDP` and `TorchLean.NN.RL.StochasticMDP` use
typed tensors. Many RL models are naturally measure-theoretic, with continuous state spaces and
stochastic dynamics, and for them the right abstraction is a Markov kernel
`κ : (S × A) → Measure S`. The interface has deterministic policies $\pi:S\to A$, Markov-kernel
transitions for next states, real-valued rewards, discounting and an optional per-(state,action)
terminal flag.

The **proof layer** (see `TorchLean.NN.RL.MDP.Bellman`) adds the standard discounted Bellman
facts: contraction in the sup metric for bounded measurable value functions, the unique fixed point,
and value-iteration convergence.

We wrote the structure by hand on mathlib's kernels. The kernel is the measure form of Sutton and
Barto's transition probabilities $p(s' \mid s, a)$, and `reward` is their expected reward
$r(s, a)$. Policies are deterministic, so `bellmanPolicy` reads one action per state where their
Bellman equation averages over $\pi(a \mid s)$. `actionValue` goes through `RL.discountedBackup`,
and a terminal `(s, a)` gets its reward with no expected next value added. `expectedNextValue` is
mathlib's Bochner integral, which is 0 for a function that isn't integrable. With the next state
$n$ drawn with probability $2^{-(n+1)}$ and $v(n) = 2^n$ the expectation diverges, and
`expectedNextValue` returns 0. The Bellman theorems take bounded measurable value functions, which
a Markov kernel always integrates.

References:

- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (Wiley, 1994),
  Chapter 6, "Discounted Markov Decision Problems": https://doi.org/10.1002/9780470316887
- Bertsekas, *Dynamic Programming and Optimal Control*, Vol. II (4th ed., Athena Scientific,
  2012), Chapter 1, the theory of discounted problems: https://www.mit.edu/~dimitrib/dpbook.html
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  Chapter 3, eqs. (3.4) and (3.5) for the dynamics and (3.14) for the Bellman equation:
  http://incompleteideas.net/book/the-book-2nd.html
- mathlib: `ProbabilityTheory.Kernel` and `ProbabilityTheory.IsMarkovKernel` in
  `Mathlib/Probability/Kernel/Defs.lean`.
- TorchRL documentation is a helpful engineering analogue for stochastic environment interfaces.
  The file uses mathlib kernels because its goal is formal probability semantics:
  https://pytorch.org/rl/

Naming note:

- The names in this file live under `RL.MDP`, the measurable-space Markov-kernel MDP. The finite
  tensor versions are `RL.FiniteMDP` and `RL.StochasticMDP`.
- `Policy`, `ValueFunction` and `Valid` stay short inside the namespace: they are the standard
  mathematical words, and the namespace carries the context.
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace MDP

open MeasureTheory ProbabilityTheory

universe u v

variable {S : Type u} {A : Type v} [MeasurableSpace S] [MeasurableSpace A]

/-- Value function over an arbitrary measurable state space. -/
abbrev ValueFunction (S : Type u) : Type u := S → ℝ

/-- Deterministic policy. Proofs that need measurability take it as a hypothesis. -/
abbrev Policy (S : Type u) (A : Type v) : Type (max u v) := S → A

/-- Discounted MDP specified by a Markov kernel $P(\mathord{\cdot}\mid s,a)$ plus
reward/termination metadata. -/
structure _root_.TorchLean.RL.MDP (S : Type u) (A : Type v) [MeasurableSpace S]
    [MeasurableSpace A] where
  /-- Canonical reset state. -/
  initialState : S
  /-- Transition kernel: $P(\mathord{\cdot}\mid s,a)$ as a measure on next states. -/
  transition : Kernel (S × A) S
  /-- Immediate reward `r(s, a)`. -/
  reward : S → A → ℝ
  /-- Task-defined terminal flag for `(s, a)`. -/
  terminated : S → A → Bool := fun _ _ => false
  /-- Discount factor. -/
  discount : ℝ

/-- Well-formedness assumptions for a Markov-kernel MDP. -/
structure Valid (mdp : MDP S A) : Prop where
  /-- Transition kernel is Markov: each $P(\mathord{\cdot}\mid s,a)$ is a probability measure. -/
  isMarkov : IsMarkovKernel mdp.transition
  /-- Reward is measurable as a function of `(s,a)`. -/
  measurable_reward : Measurable (fun sa : S × A => mdp.reward sa.1 sa.2)
  /-- Terminal flag is measurable as a function of `(s,a)`. -/
  measurable_terminated : Measurable (fun sa : S × A => mdp.terminated sa.1 sa.2)
  /-- Discount factor is nonnegative. -/
  discount_nonneg : 0 ≤ mdp.discount
  /-- Discount factor is strictly less than `1`. -/
  discount_lt_one : mdp.discount < 1

/-- The transition measure $P(\mathord{\cdot}\mid s,a)$ obtained by applying the Markov kernel. -/
noncomputable def transitionMeasure (mdp : MDP S A) (state : S) (action : A) : Measure S :=
  mdp.transition (state, action)

/-- Expected next-state value
$\mathbb{E}[v(s_{t+1})\mid s_t=s,\ a_t=a]$. -/
noncomputable def expectedNextValue
    (mdp : MDP S A)
    (values : ValueFunction S)
    (state : S)
    (action : A) : ℝ :=
  ∫ nextState, values nextState ∂(transitionMeasure mdp state action)

/-- Bellman-style state-action value induced by a candidate value function. -/
noncomputable def actionValue
    (mdp : MDP S A)
    (values : ValueFunction S)
    (state : S)
    (action : A) : ℝ :=
  discountedBackup (α := ℝ)
    (reward := mdp.reward state action)
    (gamma := mdp.discount)
    (bootstrap := expectedNextValue mdp values state action)
    (done := mdp.terminated state action)

/-- Bellman expectation operator for a deterministic policy. -/
noncomputable def bellmanPolicy
    (mdp : MDP S A)
    (policy : Policy S A)
    (values : ValueFunction S) : ValueFunction S :=
  fun state => actionValue mdp values state (policy state)

/-- Bellman optimality operator for a *finite* action space. -/
noncomputable def bellmanMax
    (mdp : MDP S A)
    [Fintype A] [Nonempty A]
    (values : ValueFunction S) : ValueFunction S :=
  fun state =>
    (Finset.univ : Finset A).sup' Finset.univ_nonempty
      (fun action => actionValue mdp values state action)

end MDP
end RL
end TorchLean
