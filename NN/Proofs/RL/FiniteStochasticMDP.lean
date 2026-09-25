/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RL.MDP
public import NN.Spec.RL.FiniteStochasticMDP

/-!
# Finite Stochastic MDP Proofs

This module proves the key discounted Bellman facts for TorchLean's finite stochastic MDP layer:

- monotonicity of Bellman expectation and Bellman optimality,
- Bellman expectation is a contraction in the sup metric,
- Bellman optimality is also a contraction in the sup metric,
- both operators have a unique fixed point, and value iteration converges to it geometrically.

The setting is intentionally finite and concrete: a clean, trustworthy formal base that mirrors the
standard textbook RL theory for discounted MDPs, rather than maximal generality.

References:
- Puterman, *Markov Decision Processes* (1994), discounted case:
  https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887
- Bertsekas, *Dynamic Programming and Optimal Control*, Vol. 1 (contraction mapping argument):
  http://web.mit.edu/dimitrib/www/dpoc.html
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018),
  Bellman expectation/optimality operators:
  http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace Proofs
namespace RL
namespace FiniteStochastic

open Spec.RL
open Spec.RL.FiniteStochastic

variable {nStates nActions : Nat}

/-!
The sup metric and the fixed-point theory come from `Proofs.RL.MDP`, which proves them once for any
`SupContraction` on finite value tables. This file only proves the one-step bounds for stochastic
transitions. The `open` is fully qualified because `MDP` on its own is also the name of the
transition structure used below.
-/
open _root_.Proofs.RL.MDP (valueSupDist valueSupDist_nonneg valueSupDist_le
  abs_sub_valueAt_le_valueSupDist SupContraction)
open Proofs.RL.Core (discountedBackup_mono discountedBackup_abs_sub_le)
open Filter Topology

/-- Policy evaluation reads back the selected action value. -/
theorem valueAt_bellmanPolicy
    (mdp : MDP nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    valueAt (bellmanPolicy mdp policy values) state =
      actionValue mdp values state (policy state) := by
  simp [valueAt, Spec.RL.FiniteStochastic.bellmanPolicy]

/-- Bellman optimality reads back the best action value. -/
theorem valueAt_bellmanOptimality
    [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
    valueAt (bellmanOptimality mdp values) state =
      (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty
        (actionValue mdp values state) := by
  simp [valueAt, Spec.RL.FiniteStochastic.bellmanOptimality]

/-- Expected next-state value is monotone in the candidate value function. -/
theorem expectedNextValue_monotone
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates)
    (action : Fin nActions) :
    expectedNextValue mdp values₁ state action ≤ expectedNextValue mdp values₂ state action :=
  Finset.sum_le_sum fun nextState _ =>
    mul_le_mul_of_nonneg_left (hValues nextState) (valid.transition_nonneg state action nextState)

/-- Bellman state-action values are monotone in the candidate value function. -/
theorem actionValue_monotone
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates)
    (action : Fin nActions) :
    actionValue mdp values₁ state action ≤ actionValue mdp values₂ state action :=
  discountedBackup_mono _ valid.discount_nonneg
    (expectedNextValue_monotone mdp valid values₁ values₂ hValues state action)

/-- Bellman expectation operators are pointwise monotone. -/
theorem bellmanPolicy_monotone
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (policy : Policy nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates) :
    valueAt (bellmanPolicy mdp policy values₁) state ≤
      valueAt (bellmanPolicy mdp policy values₂) state := by
  simpa [valueAt_bellmanPolicy] using
    actionValue_monotone mdp valid values₁ values₂ hValues state (policy state)

/-- Optimal Bellman operators are pointwise monotone. -/
theorem bellmanOptimality_monotone
    [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates) :
    valueAt (bellmanOptimality mdp values₁) state ≤
      valueAt (bellmanOptimality mdp values₂) state := by
  simp only [valueAt_bellmanOptimality]
  exact Finset.sup'_mono_fun fun action _ =>
    actionValue_monotone mdp valid values₁ values₂ hValues state action

/-- Coordinatewise expectation difference is bounded by the sup distance. -/
theorem expectedNextValue_abs_sub_le
    [Fact (0 < nStates)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) :
    |expectedNextValue mdp values₁ state action - expectedNextValue mdp values₂ state action|
      ≤ valueSupDist values₁ values₂ := by
  let p := fun nextState => (mdp.transitionProb state action).getScalar nextState
  have hp := valid.transition_nonneg state action
  calc
    |expectedNextValue mdp values₁ state action - expectedNextValue mdp values₂ state action|
        = |∑ s, p s * (valueAt values₁ s - valueAt values₂ s)| := by
          simp [expectedNextValue, p, mul_sub, Finset.sum_sub_distrib]
    _ ≤ ∑ s, p s * |valueAt values₁ s - valueAt values₂ s| :=
          (Finset.abs_sum_le_sum_abs _ _).trans_eq
            (Finset.sum_congr rfl fun s _ => by rw [abs_mul, abs_of_nonneg (hp s)])
    _ ≤ ∑ s, p s * valueSupDist values₁ values₂ :=
          Finset.sum_le_sum fun s _ =>
            mul_le_mul_of_nonneg_left (abs_sub_valueAt_le_valueSupDist values₁ values₂ s) (hp s)
    _ = valueSupDist values₁ values₂ := by
          rw [← Finset.sum_mul, valid.transition_sums_to_one state action, one_mul]

/-- State-action Bellman values are Lipschitz with constant `γ` in the sup metric. -/
theorem actionValue_abs_sub_le
    [Fact (0 < nStates)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) :
    |actionValue mdp values₁ state action - actionValue mdp values₂ state action|
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  discountedBackup_abs_sub_le _ valid.discount_nonneg (valueSupDist_nonneg values₁ values₂)
    (expectedNextValue_abs_sub_le mdp valid values₁ values₂ state action)

/-- Bellman expectation is a contraction with modulus `γ` in the sup metric:

`valueSupDist (T^π values₁) (T^π values₂) ≤ γ * valueSupDist values₁ values₂`. -/
theorem bellmanPolicy_contraction
    [Fact (0 < nStates)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (policy : Policy nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist (bellmanPolicy mdp policy values₁) (bellmanPolicy mdp policy values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  valueSupDist_le fun state => by
    simpa [valueAt_bellmanPolicy] using
      actionValue_abs_sub_le mdp valid values₁ values₂ state (policy state)

/-- Every particular action-value is bounded by Bellman optimality. -/
theorem actionValue_le_bellmanOptimality
    [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) :
    actionValue mdp values state action ≤ valueAt (bellmanOptimality mdp values) state := by
  rw [valueAt_bellmanOptimality]
  exact Finset.le_sup' (actionValue mdp values state) (Finset.mem_univ action)

/-- Bellman optimality dominates Bellman evaluation under any deterministic policy. -/
theorem bellmanPolicy_le_bellmanOptimality
    [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    valueAt (bellmanPolicy mdp policy values) state ≤
      valueAt (bellmanOptimality mdp values) state := by
  simpa [valueAt_bellmanPolicy] using
    actionValue_le_bellmanOptimality mdp values state (policy state)

/-- At a fixed state, Bellman optimality is a contraction with modulus `γ`. -/
theorem bellmanOptimality_abs_sub_le
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    |valueAt (bellmanOptimality mdp values₁) state -
        valueAt (bellmanOptimality mdp values₂) state|
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  simp only [valueAt_bellmanOptimality]
  exact abs_sup'_sub_sup'_le _ _ _ _ _ fun action _ =>
    actionValue_abs_sub_le mdp valid values₁ values₂ state action

/-- Bellman optimality is a contraction with modulus `γ` in the sup metric:

`valueSupDist (T* values₁) (T* values₂) ≤ γ * valueSupDist values₁ values₂`. -/
theorem bellmanOptimality_contraction
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist (bellmanOptimality mdp values₁)
      (bellmanOptimality mdp values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  valueSupDist_le (bellmanOptimality_abs_sub_le mdp valid values₁ values₂)

/-!
## Fixed Points and Value Iteration

`Valid` puts the discount in `[0, 1)`, so both operators are `SupContraction`s. The fixed point
exists by Banach's theorem, is unique, and value iteration converges to it from any start. The
theorems below name the consequences most often cited.
-/

section FixedPoints

variable [Fact (0 < nStates)]

/-- Policy evaluation is a `SupContraction`. -/
theorem bellmanPolicy_supContraction
    (mdp : MDP nStates nActions) (valid : Valid mdp) (policy : Policy nStates nActions) :
    SupContraction mdp.discount (bellmanPolicy mdp policy) :=
  ⟨valid.discount_nonneg, valid.discount_lt_one, bellmanPolicy_contraction mdp valid policy⟩

/-- Bellman optimality is a `SupContraction`. -/
theorem bellmanOptimality_supContraction [Fact (0 < nActions)]
    (mdp : MDP nStates nActions) (valid : Valid mdp) :
    SupContraction mdp.discount (bellmanOptimality mdp) :=
  ⟨valid.discount_nonneg, valid.discount_lt_one, bellmanOptimality_contraction mdp valid⟩

/-- The value of a policy exists and is unique: `T^π` has exactly one fixed point. -/
theorem bellmanPolicy_existsUnique_fixedPoint
    (mdp : MDP nStates nActions) (valid : Valid mdp) (policy : Policy nStates nActions) :
    ∃! values : ValueFunction ℝ nStates, bellmanPolicy mdp policy values = values :=
  (bellmanPolicy_supContraction mdp valid policy).existsUnique_fixedPoint

/-- The optimal value function exists and is unique: `T*` has exactly one fixed point. -/
theorem bellmanOptimality_existsUnique_fixedPoint [Fact (0 < nActions)]
    (mdp : MDP nStates nActions) (valid : Valid mdp) :
    ∃! values : ValueFunction ℝ nStates, bellmanOptimality mdp values = values :=
  (bellmanOptimality_supContraction mdp valid).existsUnique_fixedPoint

/-- Value iteration converges to the optimal value function from any starting table. -/
theorem bellmanOptimality_valueIteration_tendsto [Fact (0 < nActions)]
    (mdp : MDP nStates nActions) (valid : Valid mdp) (values : ValueFunction ℝ nStates) :
    Tendsto (fun k => valueSupDist ((bellmanOptimality mdp)^[k] values)
      (bellmanOptimality_supContraction mdp valid).fixedPoint) atTop (𝓝 0) :=
  (bellmanOptimality_supContraction mdp valid).tendsto_iterate values

/-- `bellmanPolicy` iterates are geometric contractions in `valueSupDist`. -/
theorem bellmanPolicy_iterate_contraction
    (mdp : MDP nStates nActions) (valid : Valid mdp) (policy : Policy nStates nActions)
    (k : Nat) (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist ((bellmanPolicy mdp policy)^[k] values₁) ((bellmanPolicy mdp policy)^[k] values₂)
      ≤ mdp.discount ^ k * valueSupDist values₁ values₂ :=
  (bellmanPolicy_supContraction mdp valid policy).iterate_le k values₁ values₂

/-- A fixed point of the Bellman policy operator is unique. -/
theorem bellmanPolicy_fixedPoint_unique
    (mdp : MDP nStates nActions) (valid : Valid mdp) (policy : Policy nStates nActions)
    (v w : ValueFunction ℝ nStates)
    (hv : bellmanPolicy mdp policy v = v) (hw : bellmanPolicy mdp policy w = w) :
    v = w :=
  let h := bellmanPolicy_supContraction mdp valid policy
  (h.eq_fixedPoint hv).trans (h.eq_fixedPoint hw).symm

/-- Error bound to a fixed point: iterating the Bellman policy operator reduces sup-distance
geometrically (`γ^k`). -/
theorem bellmanPolicy_iterate_error_to_fixedPoint
    (mdp : MDP nStates nActions) (valid : Valid mdp) (policy : Policy nStates nActions)
    (v vStar : ValueFunction ℝ nStates) (hvStar : bellmanPolicy mdp policy vStar = vStar)
    (k : Nat) :
    valueSupDist ((bellmanPolicy mdp policy)^[k] v) vStar
      ≤ mdp.discount ^ k * valueSupDist v vStar :=
  (bellmanPolicy_supContraction mdp valid policy).iterate_error hvStar k

/-- `bellmanOptimality` iterates are geometric contractions in `valueSupDist`. -/
theorem bellmanOptimality_iterate_contraction [Fact (0 < nActions)]
    (mdp : MDP nStates nActions) (valid : Valid mdp)
    (k : Nat) (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist ((bellmanOptimality mdp)^[k] values₁) ((bellmanOptimality mdp)^[k] values₂)
      ≤ mdp.discount ^ k * valueSupDist values₁ values₂ :=
  (bellmanOptimality_supContraction mdp valid).iterate_le k values₁ values₂

/-- A fixed point of the Bellman optimality operator is unique. -/
theorem bellmanOptimality_fixedPoint_unique [Fact (0 < nActions)]
    (mdp : MDP nStates nActions) (valid : Valid mdp)
    (v w : ValueFunction ℝ nStates)
    (hv : bellmanOptimality mdp v = v) (hw : bellmanOptimality mdp w = w) :
    v = w :=
  let h := bellmanOptimality_supContraction mdp valid
  (h.eq_fixedPoint hv).trans (h.eq_fixedPoint hw).symm

/-- Error bound to a fixed point: iterating Bellman optimality shrinks the sup distance
geometrically. -/
theorem bellmanOptimality_iterate_error_to_fixedPoint [Fact (0 < nActions)]
    (mdp : MDP nStates nActions) (valid : Valid mdp)
    (v vStar : ValueFunction ℝ nStates) (hvStar : bellmanOptimality mdp vStar = vStar)
    (k : Nat) :
    valueSupDist ((bellmanOptimality mdp)^[k] v) vStar
      ≤ mdp.discount ^ k * valueSupDist v vStar :=
  (bellmanOptimality_supContraction mdp valid).iterate_error hvStar k

end FixedPoints

end FiniteStochastic
end RL
end Proofs
