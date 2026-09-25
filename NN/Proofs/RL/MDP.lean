/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RL.Core
public import NN.Proofs.RL.FinsetSup
public import NN.Spec.RL.MDP
public import Mathlib.Topology.MetricSpace.Contracting
public import Mathlib.Topology.MetricSpace.Pseudo.Pi

/-!
# Finite-MDP Proofs

This module proves the foundational theorems for TorchLean's finite discounted MDP layer:

- Bellman policy operators are monotone for nonnegative discounts,
- Bellman optimality operators dominate every policy operator,
- Bellman optimality is itself monotone,
- Bellman policy and Bellman optimality are contractions in the finite sup metric,
- for `γ < 1` both have a unique fixed point, and value iteration converges to it.

The fixed-point facts are proved once, for any operator on finite value tables that contracts the
sup distance (`SupContraction`). They come from Mathlib's Banach fixed-point theorem
(`ContractingWith.fixedPoint`) after carrying value tables to `Fin n → ℝ`, whose metric is the sup
metric. `Proofs.RL.FiniteStochastic` reuses the same development for stochastic transitions.

References:
- Puterman, *Markov Decision Processes* (1994), discounted dynamic programming chapter:
  https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887
- Bertsekas, *Dynamic Programming and Optimal Control*, Vol. 1 (monotonicity/contraction proofs):
  http://web.mit.edu/dimitrib/www/dpoc.html
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018),
  Bellman operators in the discounted case:
  http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace Proofs
namespace RL
namespace MDP

open Spec.RL
open Filter Topology
open Proofs.RL.Core (discountedBackup_mono discountedBackup_abs_sub_le)

variable {nStates nActions : Nat}

/-!
## Sup Metric for Finite Value Tables

The metric is the maximum absolute pointwise difference between two value tables. The finite
stochastic development opens these declarations rather than restating them.

`Proofs.RL.Markov.valueSupDist` measures the same thing on a possibly infinite state space. That one
is an `sSup` rather than a `Finset.sup'`, so it needs a boundedness hypothesis before any of these
facts hold.
-/

/-- Value tables as functions `Fin n → ℝ`. Mathlib's metric on that type is the sup metric. -/
abbrev valueEquiv (nStates : Nat) : ValueFunction ℝ nStates ≃ (Fin nStates → ℝ) :=
  TorchLean.Tensor.vectorEquiv nStates

/-- Sup distance on finite value functions, using the maximum absolute pointwise difference. -/
noncomputable def valueSupDist [Fact (0 < nStates)]
    (values₁ values₂ : ValueFunction ℝ nStates) : ℝ :=
  let _ : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  (Finset.univ : Finset (Fin nStates)).sup' Finset.univ_nonempty
    (fun state => |valueAt values₁ state - valueAt values₂ state|)

/-- Every pointwise absolute difference is bounded by the sup distance. -/
theorem abs_sub_valueAt_le_valueSupDist [Fact (0 < nStates)]
    (values₁ values₂ : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    |valueAt values₁ state - valueAt values₂ state| ≤ valueSupDist values₁ values₂ := by
  let _ : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  exact Finset.le_sup' (fun s => |valueAt values₁ s - valueAt values₂ s|) (Finset.mem_univ state)

/-- The sup distance is nonnegative. -/
theorem valueSupDist_nonneg [Fact (0 < nStates)]
    (values₁ values₂ : ValueFunction ℝ nStates) :
    0 ≤ valueSupDist values₁ values₂ :=
  (abs_nonneg _).trans (abs_sub_valueAt_le_valueSupDist values₁ values₂ ⟨0, Fact.out⟩)

/-- A uniform bound on pointwise differences bounds the sup distance. -/
theorem valueSupDist_le [Fact (0 < nStates)]
    {values₁ values₂ : ValueFunction ℝ nStates} {bound : ℝ}
    (h : ∀ state, |valueAt values₁ state - valueAt values₂ state| ≤ bound) :
    valueSupDist values₁ values₂ ≤ bound := by
  unfold valueSupDist
  exact Finset.sup'_le _ _ fun state _ => h state

/-- `valueSupDist` is Mathlib's sup distance on `Fin n → ℝ`. -/
theorem valueSupDist_eq_dist [Fact (0 < nStates)]
    (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist values₁ values₂ = dist (valueEquiv nStates values₁) (valueEquiv nStates values₂) :=
  le_antisymm
    (valueSupDist_le fun state => dist_le_pi_dist (valueEquiv nStates values₁) _ state)
    ((dist_pi_le_iff (valueSupDist_nonneg values₁ values₂)).2
      (abs_sub_valueAt_le_valueSupDist values₁ values₂))

/-- `valueSupDist = 0` iff two finite value functions are equal. -/
theorem valueSupDist_eq_zero_iff [Fact (0 < nStates)]
    (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist values₁ values₂ = 0 ↔ values₁ = values₂ := by
  rw [valueSupDist_eq_dist, dist_eq_zero, (valueEquiv nStates).injective.eq_iff]

/-!
## Fixed Points of Sup-Norm Contractions

Everything here holds for any operator `T` on finite value tables with
`valueSupDist (T v) (T w) ≤ γ * valueSupDist v w` and `0 ≤ γ < 1`. Conjugating `T` by `valueEquiv`
gives a `ContractingWith` map on the complete space `Fin n → ℝ`, and the conclusions are Mathlib's.
-/

/-- `T` contracts the finite sup distance by the factor `γ ∈ [0, 1)`. -/
structure SupContraction [Fact (0 < nStates)] (γ : ℝ)
    (T : ValueFunction ℝ nStates → ValueFunction ℝ nStates) : Prop where
  /-- The factor is nonnegative. -/
  nonneg : 0 ≤ γ
  /-- The factor is strictly below one. -/
  lt_one : γ < 1
  /-- One application of `T` shrinks every sup distance by `γ`. -/
  le : ∀ values₁ values₂, valueSupDist (T values₁) (T values₂) ≤ γ * valueSupDist values₁ values₂

namespace SupContraction

variable [Fact (0 < nStates)] {γ : ℝ} {T : ValueFunction ℝ nStates → ValueFunction ℝ nStates}

/-- The operator `T` read on `Fin n → ℝ` is a Mathlib `ContractingWith` map. -/
theorem contractingWith (h : SupContraction γ T) :
    ContractingWith γ.toNNReal ((valueEquiv nStates).conj T) := by
  refine ⟨Real.toNNReal_lt_one.mpr h.lt_one, LipschitzWith.of_dist_le_mul fun f g => ?_⟩
  simpa [valueSupDist_eq_dist, h.nonneg] using
    h.le ((valueEquiv nStates).symm f) ((valueEquiv nStates).symm g)

omit [Fact (0 < nStates)] in
private theorem iterate_apply (k : Nat) (values : ValueFunction ℝ nStates) :
    ((valueEquiv nStates).conj T)^[k] (valueEquiv nStates values) =
      valueEquiv nStates (T^[k] values) :=
  ((Function.Semiconj.iterate_right (fun v => by simp) k) values).symm

/-- The fixed point of `T` given by Banach's fixed-point theorem. -/
noncomputable def fixedPoint (h : SupContraction γ T) : ValueFunction ℝ nStates :=
  (valueEquiv nStates).symm (ContractingWith.fixedPoint _ h.contractingWith)

/-- `fixedPoint` is a fixed point. -/
theorem fixedPoint_isFixed (h : SupContraction γ T) : T h.fixedPoint = h.fixedPoint := by
  have hp := ContractingWith.fixedPoint_isFixedPt (f := (valueEquiv nStates).conj T)
    h.contractingWith
  rw [Function.IsFixedPt, Equiv.conj_apply] at hp
  exact (valueEquiv nStates).eq_symm_apply.mpr hp

/-- Every fixed point of `T` is `fixedPoint`. -/
theorem eq_fixedPoint (h : SupContraction γ T) {values : ValueFunction ℝ nStates}
    (hfix : T values = values) : values = h.fixedPoint :=
  (valueEquiv nStates).eq_symm_apply.mpr
    (ContractingWith.fixedPoint_unique h.contractingWith (x := valueEquiv nStates values)
      (by simp [Function.IsFixedPt, hfix]))

/-- `T` has exactly one fixed point. -/
theorem existsUnique_fixedPoint (h : SupContraction γ T) : ∃! values, T values = values :=
  ⟨h.fixedPoint, h.fixedPoint_isFixed, fun _ => h.eq_fixedPoint⟩

/-- `k` applications of `T` shrink every sup distance by `γ ^ k`. -/
theorem iterate_le (h : SupContraction γ T) (k : Nat) (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist (T^[k] values₁) (T^[k] values₂) ≤ γ ^ k * valueSupDist values₁ values₂ := by
  simpa [valueSupDist_eq_dist, iterate_apply, h.nonneg] using
    (h.contractingWith.toLipschitzWith.iterate k).dist_le_mul
      (valueEquiv nStates values₁) (valueEquiv nStates values₂)

/-- The distance from the `k`-th iterate to any fixed point decays like `γ ^ k`. -/
theorem iterate_error (h : SupContraction γ T) {values vStar : ValueFunction ℝ nStates}
    (hfix : T vStar = vStar) (k : Nat) :
    valueSupDist (T^[k] values) vStar ≤ γ ^ k * valueSupDist values vStar := by
  simpa [Function.iterate_fixed hfix k] using h.iterate_le k values vStar

/-- A priori error bound for value iteration, computable from the first step alone. -/
theorem apriori_error (h : SupContraction γ T) (values : ValueFunction ℝ nStates) (k : Nat) :
    valueSupDist (T^[k] values) h.fixedPoint ≤
      valueSupDist values (T values) * γ ^ k / (1 - γ) := by
  simpa [valueSupDist_eq_dist, iterate_apply, fixedPoint, h.nonneg] using
    h.contractingWith.apriori_dist_iterate_fixedPoint_le (valueEquiv nStates values) k

/-- Value iteration converges to the fixed point from any starting table. -/
theorem tendsto_iterate (h : SupContraction γ T) (values : ValueFunction ℝ nStates) :
    Tendsto (fun k => valueSupDist (T^[k] values) h.fixedPoint) atTop (𝓝 0) := by
  have := tendsto_iff_dist_tendsto_zero.mp
    (h.contractingWith.tendsto_iterate_fixedPoint (valueEquiv nStates values))
  simpa [valueSupDist_eq_dist, iterate_apply, fixedPoint] using this

end SupContraction

/-!
## Deterministic Bellman Operators
-/

/-- Policy Bellman operators read back exactly the selected state-action value. -/
theorem valueAt_bellmanPolicy
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    valueAt (bellmanPolicy mdp policy values) state =
      stateActionValue mdp values state (policy state) := by
  simp [valueAt, bellmanPolicy]

/-- Bellman optimality reads back the best state-action value. -/
theorem valueAt_bellmanOptimality
    [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
    valueAt (bellmanOptimality mdp values) state =
      (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty
        (stateActionValue mdp values state) := by
  simp [valueAt, bellmanOptimality]

/-- A Bellman state-action value is monotone in the candidate value function when `γ ≥ 0`. -/
theorem stateActionValue_monotone
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ : 0 ≤ mdp.discount)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates)
    (action : Fin nActions) :
    stateActionValue mdp values₁ state action ≤
      stateActionValue mdp values₂ state action :=
  discountedBackup_mono _ hγ (hValues _)

/-- Bellman policy operators are pointwise monotone for nonnegative discounts. -/
theorem bellmanPolicy_monotone
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ : 0 ≤ mdp.discount)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates) :
    valueAt (bellmanPolicy mdp policy values₁) state ≤
      valueAt (bellmanPolicy mdp policy values₂) state := by
  simpa [valueAt_bellmanPolicy] using
    stateActionValue_monotone mdp values₁ values₂ hγ hValues state (policy state)

/-- Bellman optimality dominates every particular action. -/
theorem stateActionValue_le_bellmanOptimality
    [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) :
    stateActionValue mdp values state action ≤
      valueAt (bellmanOptimality mdp values) state := by
  rw [valueAt_bellmanOptimality]
  exact Finset.le_sup' (stateActionValue mdp values state) (Finset.mem_univ action)

/-- Bellman optimality dominates Bellman evaluation under any deterministic policy. -/
theorem bellmanPolicy_le_bellmanOptimality
    [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    valueAt (bellmanPolicy mdp policy values) state ≤
      valueAt (bellmanOptimality mdp values) state := by
  simpa [valueAt_bellmanPolicy] using
    stateActionValue_le_bellmanOptimality mdp values state (policy state)

/-- Bellman optimality is pointwise monotone for nonnegative discounts. -/
theorem bellmanOptimality_monotone
    [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ : 0 ≤ mdp.discount)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates) :
    valueAt (bellmanOptimality mdp values₁) state ≤
      valueAt (bellmanOptimality mdp values₂) state := by
  simp only [valueAt_bellmanOptimality]
  exact Finset.sup'_mono_fun fun action _ =>
    stateActionValue_monotone mdp values₁ values₂ hγ hValues state action

/-- Deterministic state-action Bellman values are Lipschitz with constant `γ`. -/
theorem stateActionValue_abs_sub_le
    [Fact (0 < nStates)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ₀ : 0 ≤ mdp.discount)
    (state : Fin nStates)
    (action : Fin nActions) :
    |stateActionValue mdp values₁ state action - stateActionValue mdp values₂ state action|
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  discountedBackup_abs_sub_le _ hγ₀ (valueSupDist_nonneg values₁ values₂)
    (abs_sub_valueAt_le_valueSupDist values₁ values₂ _)

/-- Bellman evaluation for a deterministic policy is a `γ`-contraction in the finite sup metric. -/
theorem bellmanPolicy_contraction
    [Fact (0 < nStates)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ₀ : 0 ≤ mdp.discount) :
    valueSupDist (bellmanPolicy mdp policy values₁) (bellmanPolicy mdp policy values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  valueSupDist_le fun state => by
    simpa [valueAt_bellmanPolicy] using
      stateActionValue_abs_sub_le mdp values₁ values₂ hγ₀ state (policy state)

/-- At a fixed state, Bellman optimality is Lipschitz with constant `γ`. -/
theorem bellmanOptimality_abs_sub_le
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ₀ : 0 ≤ mdp.discount)
    (state : Fin nStates) :
    |valueAt (bellmanOptimality mdp values₁) state -
        valueAt (bellmanOptimality mdp values₂) state|
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  simp only [valueAt_bellmanOptimality]
  exact abs_sup'_sub_sup'_le _ _ _ _ _ fun action _ =>
    stateActionValue_abs_sub_le mdp values₁ values₂ hγ₀ state action

/-- Bellman optimality is a `γ`-contraction in the finite sup metric. -/
theorem bellmanOptimality_contraction
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ₀ : 0 ≤ mdp.discount) :
    valueSupDist (bellmanOptimality mdp values₁) (bellmanOptimality mdp values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  valueSupDist_le (bellmanOptimality_abs_sub_le mdp values₁ values₂ hγ₀)

/-- For `0 ≤ γ < 1`, policy evaluation is a `SupContraction`: it has a unique fixed point and
value iteration converges to it. -/
theorem bellmanPolicy_supContraction
    [Fact (0 < nStates)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (hγ₀ : 0 ≤ mdp.discount) (hγ₁ : mdp.discount < 1) :
    SupContraction mdp.discount (bellmanPolicy mdp policy) :=
  ⟨hγ₀, hγ₁, fun values₁ values₂ => bellmanPolicy_contraction mdp policy values₁ values₂ hγ₀⟩

/-- For `0 ≤ γ < 1`, Bellman optimality is a `SupContraction`: the optimal value table exists, is
unique, and value iteration converges to it. -/
theorem bellmanOptimality_supContraction
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (hγ₀ : 0 ≤ mdp.discount) (hγ₁ : mdp.discount < 1) :
    SupContraction mdp.discount (bellmanOptimality mdp) :=
  ⟨hγ₀, hγ₁, fun values₁ values₂ => bellmanOptimality_contraction mdp values₁ values₂ hγ₀⟩

end MDP
end RL
end Proofs
