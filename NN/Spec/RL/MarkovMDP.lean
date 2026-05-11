/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.MeasureTheory.Integral.Bochner.Basic
public import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
public import Mathlib.Probability.Kernel.Defs
public import NN.Spec.RL.Core

/-!
# Measure-Theoretic Discounted MDPs (Markov Kernels)

This module defines a small, proof-friendly discounted MDP interface on **general measurable**
state and action spaces, using mathlib's Markov kernel formalization.

Why this layer exists (and why it is separate from the finite-state tensor MDPs):

- `NN.Spec.RL.MDP` and `NN.Spec.RL.FiniteStochasticMDP` are *finite* and use typed tensors.
- Many RL models are naturally measure-theoretic (continuous state spaces, stochastic dynamics).
  For these, the right abstraction is a Markov kernel `κ : (S × A) → Measure S`.

This file stays intentionally small:

- deterministic policies `π : S → A`,
- Markov-kernel transitions for next states,
- real-valued rewards, discounting, and an optional per-(state,action) terminal flag.

The **proof layer** (see `NN.Proofs.RL.MarkovMDP`) adds the standard discounted Bellman facts,
including monotonicity and contraction in the sup metric for bounded value functions.

References:

- Puterman, *Markov Decision Processes* (1994), Chapters 6–7.
- Bertsekas, *Dynamic Programming and Optimal Control*.
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 3.
- mathlib: `ProbabilityTheory.Kernel` and `ProbabilityTheory.IsMarkovKernel` in
  `Mathlib/Probability/Kernel/Defs.lean`.
- TorchRL documentation is a helpful engineering analogue for stochastic environment interfaces,
  but this file uses mathlib kernels because the goal here is formal probability semantics:
  https://pytorch.org/rl/

Naming note:

- The names in this file live under `Spec.RL.Markov`. A reference to `Markov.MDP` is the
  measurable-space Markov-kernel object, while `Spec.RL.FiniteMDP` and `Spec.RL.FiniteStochastic.MDP`
  are the finite tensor layers.
- We keep `Policy`, `ValueFunction`, and `Valid` short inside the namespace because they are the
  standard mathematical words for this layer, and the namespace carries the disambiguating
  context.
-/

@[expose] public section

namespace Spec
namespace RL
namespace Markov

open MeasureTheory ProbabilityTheory

variable {S A : Type} [MeasurableSpace S] [MeasurableSpace A]

/-- Value function over an arbitrary measurable state space. -/
abbrev ValueFunction (S : Type) : Type := S → ℝ

/-- Deterministic policy (measurability assumptions live in proofs, not in the raw definition). -/
abbrev Policy (S A : Type) : Type := S → A

/-- Discounted MDP specified by a Markov kernel `P(. | s, a)` plus reward/termination metadata. -/
structure MDP (S A : Type) [MeasurableSpace S] [MeasurableSpace A] where
  /-- Canonical reset state. -/
  initialState : S
  /-- Transition kernel: `P(. | s, a)` as a measure on next states. -/
  transition : Kernel (S × A) S
  /-- Immediate reward `r(s, a)`. -/
  reward : S → A → ℝ
  /-- Task-defined terminal flag for `(s, a)`. -/
  terminated : S → A → Bool := fun _ _ => false
  /-- Discount factor. -/
  discount : ℝ

/-- Well-formedness assumptions for a Markov-kernel MDP. -/
structure Valid (mdp : MDP S A) : Prop where
  /-- Transition kernel is Markov: each `P(. | s, a)` is a probability measure. -/
  isMarkov : IsMarkovKernel mdp.transition
  /-- Reward is measurable as a function of `(s,a)`. -/
  measurable_reward : Measurable (fun sa : S × A => mdp.reward sa.1 sa.2)
  /-- Terminal flag is measurable as a function of `(s,a)`. -/
  measurable_terminated : Measurable (fun sa : S × A => mdp.terminated sa.1 sa.2)
  /-- Discount factor is nonnegative. -/
  discount_nonneg : 0 ≤ mdp.discount
  /-- Discount factor is strictly less than `1`. -/
  discount_lt_one : mdp.discount < 1

/-- The transition measure `P(. | s, a)` (convenience wrapper around the kernel application). -/
noncomputable def transitionMeasure (mdp : MDP S A) (state : S) (action : A) : Measure S :=
  mdp.transition (state, action)

/-- Expected next-state value `E[v(s_{t+1}) | s_t = s, a_t = a]`. -/
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
noncomputable def bellmanOptimality
    (mdp : MDP S A)
    [Fintype A] [Nonempty A]
    (values : ValueFunction S) : ValueFunction S :=
  fun state =>
    (Finset.univ : Finset A).sup' Finset.univ_nonempty
      (fun action => actionValue mdp values state action)

end Markov
end RL
end Spec
