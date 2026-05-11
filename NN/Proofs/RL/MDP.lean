/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic
public import Mathlib.Logic.Function.Iterate
public import NN.Proofs.RL.FinsetSup
public import NN.Proofs.Tensor.Basic
public import NN.Spec.RL.MDP

/-!
# Finite-MDP Proofs

This module proves the first foundational theorems for TorchLean's finite discounted MDP layer:

- Bellman policy operators are monotone for nonnegative discounts,
- Bellman optimality operators dominate every policy operator,
- Bellman optimality is itself monotone,
- Bellman policy and Bellman optimality are contractions in the finite sup metric.

The proofs are intentionally about deterministic finite MDPs first. That gives us a clean,
trustworthy base to build on before introducing stochastic transitions or richer measure-theoretic
machinery.

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

variable {nStates nActions : Nat}

/-!
## Sup Metric for Finite Value Tables

The finite deterministic and finite stochastic developments intentionally use the same metric:
the maximum absolute pointwise difference between two value tables.
-/

/-- Sup distance on finite value functions, using the maximum absolute pointwise difference. -/
noncomputable def valueSupDist [Fact (0 < nStates)]
    (values₁ values₂ : ValueFunction ℝ nStates) : ℝ :=
  let _ : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  (Finset.univ : Finset (Fin nStates)).sup' Finset.univ_nonempty
    (fun state => |valueAt values₁ state - valueAt values₂ state|)

/-- The sup distance is nonnegative. -/
theorem valueSupDist_nonneg [Fact (0 < nStates)]
    (values₁ values₂ : ValueFunction ℝ nStates) :
    0 ≤ valueSupDist values₁ values₂ := by
  have hcoord :
      0 ≤ |valueAt values₁ ⟨0, Fact.out⟩ - valueAt values₂ ⟨0, Fact.out⟩| := abs_nonneg _
  have hle :
      |valueAt values₁ ⟨0, Fact.out⟩ - valueAt values₂ ⟨0, Fact.out⟩| ≤
          valueSupDist values₁ values₂ := by
    unfold valueSupDist
    exact Finset.le_sup' (fun state => |valueAt values₁ state - valueAt values₂ state|)
      (Finset.mem_univ ⟨0, Fact.out⟩)
  exact hcoord.trans hle

/-- Every pointwise absolute difference is bounded by the sup distance. -/
theorem abs_sub_valueAt_le_valueSupDist [Fact (0 < nStates)]
    (values₁ values₂ : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    |valueAt values₁ state - valueAt values₂ state| ≤ valueSupDist values₁ values₂ := by
  unfold valueSupDist
  exact Finset.le_sup' (fun s => |valueAt values₁ s - valueAt values₂ s|) (Finset.mem_univ state)

/-- `stateActionValue` is exactly the Bellman backup on the chosen successor state. -/
theorem stateActionValue_eq
    (mdp : FiniteMDP ℝ nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) :
    stateActionValue mdp values state action =
      let out := mdp.step state action
      out.reward + mdp.discount * continueMask (α := ℝ) out.terminated * valueAt values out.state := by
  rfl

/-- Policy Bellman operators read back exactly the selected state-action value. -/
theorem valueAt_bellmanPolicy
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    valueAt (bellmanPolicy mdp policy values) state =
      stateActionValue mdp values state (policy state) := by
  rfl

/-- `continueMask` is always nonnegative. -/
theorem continueMask_nonneg (done : Bool) :
    0 ≤ (continueMask (α := ℝ) done : ℝ) := by
  cases done <;> norm_num [continueMask]

/-- A Bellman state-action value is monotone in the candidate value function when `γ ≥ 0`. -/
theorem stateActionValue_monotone
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ : 0 ≤ mdp.discount)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates)
    (action : Fin nActions) :
    stateActionValue mdp values₁ state action ≤
      stateActionValue mdp values₂ state action := by
  let out := mdp.step state action
  have hMask : 0 ≤ continueMask (α := ℝ) out.terminated :=
    continueMask_nonneg out.terminated
  have hScale : 0 ≤ mdp.discount * continueMask (α := ℝ) out.terminated :=
    mul_nonneg hγ hMask
  have hNext : valueAt values₁ out.state ≤ valueAt values₂ out.state :=
    hValues out.state
  have hMul :
      mdp.discount * continueMask (α := ℝ) out.terminated * valueAt values₁ out.state ≤
        mdp.discount * continueMask (α := ℝ) out.terminated * valueAt values₂ out.state :=
    mul_le_mul_of_nonneg_left hNext hScale
  simpa [stateActionValue_eq, out]
    using add_le_add_left hMul out.reward

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
  let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  simp [bellmanOptimality, valueAt]
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
  let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  simp [bellmanOptimality, valueAt]
  refine Finset.sup'_le Finset.univ_nonempty (stateActionValue mdp values₁ state) ?_
  intro action _
  exact (stateActionValue_monotone mdp values₁ values₂ hγ hValues state action).trans
    (Finset.le_sup' (stateActionValue mdp values₂ state) (Finset.mem_univ action))

/-!
## Contraction Guarantees

For `0 ≤ γ < 1`, deterministic finite Bellman operators are contractions in `valueSupDist`.
This is the same mathematical guarantee used by value iteration in the stochastic layer; the only
difference is that the next state is a single successor instead of an expectation over a transition
row.
-/

/-- Deterministic state-action Bellman values are Lipschitz with constant `γ`. -/
theorem stateActionValue_abs_sub_le
    [Fact (0 < nStates)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ₀ : 0 ≤ mdp.discount)
    (state : Fin nStates)
    (action : Fin nActions) :
    |stateActionValue mdp values₁ state action - stateActionValue mdp values₂ state action|
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  let out := mdp.step state action
  by_cases hdone : out.terminated
  · have hnonneg :
        0 ≤ mdp.discount * valueSupDist values₁ values₂ :=
      mul_nonneg hγ₀ (valueSupDist_nonneg values₁ values₂)
    simp [stateActionValue, discountedBackup, continueMask, out, hdone, hnonneg]
  · have hpoint :
        |valueAt values₁ out.state - valueAt values₂ out.state| ≤
          valueSupDist values₁ values₂ :=
      abs_sub_valueAt_le_valueSupDist values₁ values₂ out.state
    have hmul :
        |mdp.discount * (valueAt values₁ out.state - valueAt values₂ out.state)|
          ≤ mdp.discount * valueSupDist values₁ values₂ := by
      calc
        |mdp.discount * (valueAt values₁ out.state - valueAt values₂ out.state)|
            = |mdp.discount| * |valueAt values₁ out.state - valueAt values₂ out.state| := by
                rw [abs_mul]
        _ = mdp.discount * |valueAt values₁ out.state - valueAt values₂ out.state| := by
                rw [abs_of_nonneg hγ₀]
        _ ≤ mdp.discount * valueSupDist values₁ values₂ :=
                mul_le_mul_of_nonneg_left hpoint hγ₀
    have hrewrite :
        stateActionValue mdp values₁ state action - stateActionValue mdp values₂ state action =
          mdp.discount * (valueAt values₁ out.state - valueAt values₂ out.state) := by
      simp [stateActionValue, discountedBackup, continueMask, out, hdone]
      ring
    rw [hrewrite]
    exact hmul

/-- Bellman evaluation for a deterministic policy is a `γ`-contraction in the finite sup metric. -/
theorem bellmanPolicy_contraction
    [Fact (0 < nStates)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ₀ : 0 ≤ mdp.discount) :
    valueSupDist (bellmanPolicy mdp policy values₁) (bellmanPolicy mdp policy values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  let _ : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  unfold valueSupDist
  refine Finset.sup'_le (s := (Finset.univ : Finset (Fin nStates))) Finset.univ_nonempty
    (f := fun state =>
      |valueAt (bellmanPolicy mdp policy values₁) state -
          valueAt (bellmanPolicy mdp policy values₂) state|) ?_
  intro state _
  simpa [bellmanPolicy, valueAt] using
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
  let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  let bound := mdp.discount * valueSupDist values₁ values₂
  let f : Fin nActions → ℝ := stateActionValue mdp values₁ state
  let g : Fin nActions → ℝ := stateActionValue mdp values₂ state
  have hfg : ∀ action ∈ (Finset.univ : Finset (Fin nActions)), f action ≤ g action + bound := by
    intro action _
    have habs := stateActionValue_abs_sub_le mdp values₁ values₂ hγ₀ state action
    linarith [abs_sub_le_iff.mp habs]
  have hgf : ∀ action ∈ (Finset.univ : Finset (Fin nActions)), g action ≤ f action + bound := by
    intro action _
    have habs := stateActionValue_abs_sub_le mdp values₁ values₂ hγ₀ state action
    linarith [abs_sub_le_iff.mp habs]
  have hs1 :
      (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty f
        ≤ (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty g + bound :=
    _root_.Proofs.RL.sup'_le_add_const
      (Finset.univ : Finset (Fin nActions)) Finset.univ_nonempty f g bound hfg
  have hs2 :
      (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty g
        ≤ (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty f + bound :=
    _root_.Proofs.RL.sup'_le_add_const
      (Finset.univ : Finset (Fin nActions)) Finset.univ_nonempty g f bound hgf
  have habs :
      |(Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty f -
          (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty g|
        ≤ bound :=
    abs_sub_le_iff.mpr ⟨sub_le_iff_le_add'.mpr hs1, sub_le_iff_le_add'.mpr hs2⟩
  simpa [bellmanOptimality, valueAt, f, g, bound] using habs

/-- Bellman optimality is a `γ`-contraction in the finite sup metric. -/
theorem bellmanOptimality_contraction
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ₀ : 0 ≤ mdp.discount) :
    valueSupDist (bellmanOptimality mdp values₁) (bellmanOptimality mdp values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  let _ : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  unfold valueSupDist
  refine Finset.sup'_le (s := (Finset.univ : Finset (Fin nStates))) Finset.univ_nonempty
    (f := fun state =>
      |valueAt (bellmanOptimality mdp values₁) state -
          valueAt (bellmanOptimality mdp values₂) state|) ?_
  intro state _
  exact bellmanOptimality_abs_sub_le mdp values₁ values₂ hγ₀ state

end MDP
end RL
end Proofs
