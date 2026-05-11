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
public import NN.Spec.RL.FiniteStochasticMDP

/-!
# Finite Stochastic MDP Proofs

This module proves the key discounted Bellman facts for TorchLean's finite stochastic MDP layer:

- monotonicity of Bellman expectation and Bellman optimality,
- Bellman expectation is a contraction in the sup metric,
- Bellman optimality is also a contraction in the sup metric.

The setting is intentionally finite and concrete. The goal is not maximal generality; it is a
clean, trustworthy formal base that mirrors the standard textbook RL theory for discounted MDPs.

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
      |valueAt values₁ ⟨0, Fact.out⟩ - valueAt values₂ ⟨0, Fact.out⟩| ≤ valueSupDist values₁ values₂ := by
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

/-- Expected next-state value is monotone in the candidate value function. -/
theorem expectedNextValue_monotone
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates)
    (action : Fin nActions) :
    expectedNextValue mdp values₁ state action ≤ expectedNextValue mdp values₂ state action := by
  unfold expectedNextValue
  refine Finset.sum_le_sum ?_
  intro nextState _
  exact mul_le_mul_of_nonneg_left (hValues nextState)
    (valid.transition_nonneg state action nextState)

/-- Bellman state-action values are monotone in the candidate value function. -/
theorem actionValue_monotone
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates)
    (action : Fin nActions) :
    Spec.RL.FiniteStochastic.actionValue mdp values₁ state action ≤
      Spec.RL.FiniteStochastic.actionValue mdp values₂ state action := by
  by_cases hdone : mdp.terminated state action
  · simp [Spec.RL.FiniteStochastic.actionValue, discountedBackup, continueMask, hdone]
  · simp [Spec.RL.FiniteStochastic.actionValue, discountedBackup, continueMask, hdone]
    exact mul_le_mul_of_nonneg_left
      (expectedNextValue_monotone mdp valid values₁ values₂ hValues state action)
      valid.discount_nonneg

/-- Bellman expectation operators are pointwise monotone. -/
theorem bellmanPolicy_monotone
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (policy : Policy nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates) :
    valueAt (Spec.RL.FiniteStochastic.bellmanPolicy mdp policy values₁) state ≤
      valueAt (Spec.RL.FiniteStochastic.bellmanPolicy mdp policy values₂) state := by
  simpa [Spec.RL.FiniteStochastic.bellmanPolicy, valueAt] using
    actionValue_monotone mdp valid values₁ values₂ hValues state (policy state)

/-- Optimal Bellman operators are pointwise monotone. -/
theorem bellmanOptimality_monotone
    [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, valueAt values₁ state ≤ valueAt values₂ state)
    (state : Fin nStates) :
    valueAt (Spec.RL.FiniteStochastic.bellmanOptimality mdp values₁) state ≤
      valueAt (Spec.RL.FiniteStochastic.bellmanOptimality mdp values₂) state := by
  let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  simp [Spec.RL.FiniteStochastic.bellmanOptimality, valueAt]
  refine Finset.sup'_le (s := (Finset.univ : Finset (Fin nActions))) Finset.univ_nonempty
    (Spec.RL.FiniteStochastic.actionValue mdp values₁ state) ?_
  intro action _
  exact (actionValue_monotone mdp valid values₁ values₂ hValues state action).trans
    (Finset.le_sup' (Spec.RL.FiniteStochastic.actionValue mdp values₂ state) (Finset.mem_univ action))

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
  let row := mdp.transitionProb state action
  have hrewrite :
      Spec.RL.FiniteStochastic.expectedNextValue mdp values₁ state action -
          Spec.RL.FiniteStochastic.expectedNextValue mdp values₂ state action =
        (Finset.univ : Finset (Fin nStates)).sum
          (fun nextState =>
            row.vecGet nextState *
              (valueAt values₁ nextState - valueAt values₂ nextState)) := by
    change
      (∑ nextState : Fin nStates,
          (mdp.transitionProb state action).vecGet nextState * valueAt values₁ nextState) -
        (∑ nextState : Fin nStates,
          (mdp.transitionProb state action).vecGet nextState * valueAt values₂ nextState) =
      ∑ nextState : Fin nStates,
        row.vecGet nextState * (valueAt values₁ nextState - valueAt values₂ nextState)
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl ?_
    intro nextState _
    simp [row]
    ring
  rw [hrewrite]
  calc
    |∑ nextState : Fin nStates,
        row.vecGet nextState * (valueAt values₁ nextState - valueAt values₂ nextState)|
      ≤ ∑ nextState : Fin nStates,
          |row.vecGet nextState * (valueAt values₁ nextState - valueAt values₂ nextState)| := by
            simpa using (Finset.abs_sum_le_sum_abs
              (s := (Finset.univ : Finset (Fin nStates)))
              (f := fun nextState =>
                row.vecGet nextState *
                  (valueAt values₁ nextState - valueAt values₂ nextState)))
    _ = ∑ nextState : Fin nStates,
          row.vecGet nextState * |valueAt values₁ nextState - valueAt values₂ nextState| := by
            refine Finset.sum_congr rfl ?_
            intro nextState _
            rw [abs_mul, abs_of_nonneg]
            simpa [row] using valid.transition_nonneg state action nextState
    _ ≤ ∑ nextState : Fin nStates,
          row.vecGet nextState * valueSupDist values₁ values₂ := by
            refine Finset.sum_le_sum ?_
            intro nextState _
            exact mul_le_mul_of_nonneg_left
              (abs_sub_valueAt_le_valueSupDist values₁ values₂ nextState)
              (by simpa [row] using valid.transition_nonneg state action nextState)
    _ = ((Finset.univ : Finset (Fin nStates)).sum
          (fun nextState => row.vecGet nextState)) * valueSupDist values₁ values₂ := by
            simpa using
              (Finset.sum_mul (s := (Finset.univ : Finset (Fin nStates)))
                (f := fun nextState => row.vecGet nextState)
                (a := valueSupDist values₁ values₂)).symm
    _ = valueSupDist values₁ values₂ := by
            rw [show (Finset.univ : Finset (Fin nStates)).sum
                (fun nextState => row.vecGet nextState) = 1 by
              simpa [row] using valid.transition_sums_to_one state action, one_mul]

/-- State-action Bellman values are Lipschitz with constant `γ` in the sup metric. -/
theorem actionValue_abs_sub_le
    [Fact (0 < nStates)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) :
    |Spec.RL.FiniteStochastic.actionValue mdp values₁ state action -
        Spec.RL.FiniteStochastic.actionValue mdp values₂ state action|
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  by_cases hdone : mdp.terminated state action
  · have hnonneg :
      0 ≤ mdp.discount * valueSupDist values₁ values₂ := by
        exact mul_nonneg valid.discount_nonneg (valueSupDist_nonneg values₁ values₂)
    simp [Spec.RL.FiniteStochastic.actionValue, discountedBackup, continueMask, hdone, hnonneg]
  · have hexp :=
      expectedNextValue_abs_sub_le mdp valid values₁ values₂ state action
    have hmul :
        |mdp.discount * (expectedNextValue mdp values₁ state action -
            expectedNextValue mdp values₂ state action)|
          ≤ mdp.discount * valueSupDist values₁ values₂ := by
      calc
        |mdp.discount * (expectedNextValue mdp values₁ state action -
            expectedNextValue mdp values₂ state action)|
            = |mdp.discount| *
                |expectedNextValue mdp values₁ state action -
                  expectedNextValue mdp values₂ state action| := by
                    rw [abs_mul]
        _ = mdp.discount *
              |expectedNextValue mdp values₁ state action -
                expectedNextValue mdp values₂ state action| := by
                  rw [abs_of_nonneg valid.discount_nonneg]
        _ ≤ mdp.discount * valueSupDist values₁ values₂ := by
              exact mul_le_mul_of_nonneg_left hexp valid.discount_nonneg
    have hrewrite :
        Spec.RL.FiniteStochastic.actionValue mdp values₁ state action -
            Spec.RL.FiniteStochastic.actionValue mdp values₂ state action =
          mdp.discount *
            (expectedNextValue mdp values₁ state action -
              expectedNextValue mdp values₂ state action) := by
      simp [Spec.RL.FiniteStochastic.actionValue, discountedBackup, continueMask, hdone]
      ring
    rw [hrewrite]
    exact hmul

/-- Bellman expectation is a contraction with modulus `γ` in the sup metric:

`valueSupDist (T^π values₁) (T^π values₂) ≤ γ * valueSupDist values₁ values₂`. -/
theorem bellmanPolicy_contraction
    [Fact (0 < nStates)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (policy : Policy nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist (Spec.RL.FiniteStochastic.bellmanPolicy mdp policy values₁)
      (Spec.RL.FiniteStochastic.bellmanPolicy mdp policy values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  let _ : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  unfold valueSupDist
  refine Finset.sup'_le (s := (Finset.univ : Finset (Fin nStates))) Finset.univ_nonempty
    (f := fun state =>
      |valueAt (Spec.RL.FiniteStochastic.bellmanPolicy mdp policy values₁) state -
          valueAt (Spec.RL.FiniteStochastic.bellmanPolicy mdp policy values₂) state|) ?_
  intro state _
  simpa [Spec.RL.FiniteStochastic.bellmanPolicy, valueAt] using
    actionValue_abs_sub_le mdp valid values₁ values₂ state (policy state)

/-- Every particular action-value is bounded by Bellman optimality. -/
theorem actionValue_le_bellmanOptimality
    [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) :
    Spec.RL.FiniteStochastic.actionValue mdp values state action ≤
      valueAt (Spec.RL.FiniteStochastic.bellmanOptimality mdp values) state := by
  let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  simp [Spec.RL.FiniteStochastic.bellmanOptimality, valueAt]
  exact Finset.le_sup' (Spec.RL.FiniteStochastic.actionValue mdp values state) (Finset.mem_univ action)

/-- Bellman optimality dominates Bellman evaluation under any deterministic policy. -/
theorem bellmanPolicy_le_bellmanOptimality
    [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    valueAt (Spec.RL.FiniteStochastic.bellmanPolicy mdp policy values) state ≤
      valueAt (Spec.RL.FiniteStochastic.bellmanOptimality mdp values) state := by
  simpa [Spec.RL.FiniteStochastic.bellmanPolicy, valueAt] using
    actionValue_le_bellmanOptimality mdp values state (policy state)

/-- At a fixed state, Bellman optimality is a contraction with modulus `γ`. -/
theorem bellmanOptimality_abs_sub_le
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    |valueAt (Spec.RL.FiniteStochastic.bellmanOptimality mdp values₁) state -
        valueAt (Spec.RL.FiniteStochastic.bellmanOptimality mdp values₂) state|
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  let bound := mdp.discount * valueSupDist values₁ values₂
  let f : Fin nActions → ℝ := Spec.RL.FiniteStochastic.actionValue mdp values₁ state
  let g : Fin nActions → ℝ := Spec.RL.FiniteStochastic.actionValue mdp values₂ state
  have hfg : ∀ action ∈ (Finset.univ : Finset (Fin nActions)), f action ≤ g action + bound := by
    intro action _
    have habs := actionValue_abs_sub_le mdp valid values₁ values₂ state action
    linarith [abs_sub_le_iff.mp habs]
  have hgf : ∀ action ∈ (Finset.univ : Finset (Fin nActions)), g action ≤ f action + bound := by
    intro action _
    have habs := actionValue_abs_sub_le mdp valid values₁ values₂ state action
    linarith [abs_sub_le_iff.mp habs]
  have hs1 :
      (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty f
        ≤ (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty g + bound := by
    exact _root_.Proofs.RL.sup'_le_add_const
      (Finset.univ : Finset (Fin nActions)) Finset.univ_nonempty f g bound hfg
  have hs2 :
      (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty g
        ≤ (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty f + bound := by
    exact _root_.Proofs.RL.sup'_le_add_const
      (Finset.univ : Finset (Fin nActions)) Finset.univ_nonempty g f bound hgf
  have habs :
      |(Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty f -
          (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty g|
        ≤ bound := by
    exact abs_sub_le_iff.mpr
      ⟨sub_le_iff_le_add'.mpr hs1, sub_le_iff_le_add'.mpr hs2⟩
  simpa [Spec.RL.FiniteStochastic.bellmanOptimality, valueAt, f, g, bound] using habs

/-- Bellman optimality is a contraction with modulus `γ` in the sup metric:

`valueSupDist (T* values₁) (T* values₂) ≤ γ * valueSupDist values₁ values₂`. -/
theorem bellmanOptimality_contraction
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist (Spec.RL.FiniteStochastic.bellmanOptimality mdp values₁)
      (Spec.RL.FiniteStochastic.bellmanOptimality mdp values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  let _ : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  unfold valueSupDist
  refine Finset.sup'_le (s := (Finset.univ : Finset (Fin nStates))) Finset.univ_nonempty
    (f := fun state =>
      |valueAt (Spec.RL.FiniteStochastic.bellmanOptimality mdp values₁) state -
          valueAt (Spec.RL.FiniteStochastic.bellmanOptimality mdp values₂) state|) ?_
  intro state _
  exact bellmanOptimality_abs_sub_le mdp valid values₁ values₂ state

/-!
## Contraction Iterates and Fixed Points

The earlier theorems show that (under `0 ≤ γ < 1`) the Bellman operators are `γ`-contractions in the
sup metric (`valueSupDist`).

This section packages the standard consequences used throughout discounted-RL theory:

- iterating a contraction shrinks distances geometrically (`γ^k`),
- fixed points are unique,
- the error to a fixed point decays geometrically under iteration.

These statements are the formal backbone behind “value iteration converges” style arguments, and
they are useful even before we prove existence of a fixed point (existence is typically obtained
via a completeness argument, or via an explicit linear-system solution in the finite case).
-/

section FixedPoints

variable {nStates nActions : Nat}

private lemma dimScalarEquiv_apply_eq_valueAt
    (values : ValueFunction ℝ nStates) (state : Fin nStates) :
    (Spec.Tensor.dimScalarEquiv (α := ℝ) nStates values) state = valueAt values state := by
  cases values with
  | dim _ =>
      rfl

/-- `valueSupDist = 0` iff two finite value functions are equal. -/
theorem valueSupDist_eq_zero_iff
    [Fact (0 < nStates)]
    (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist values₁ values₂ = 0 ↔ values₁ = values₂ := by
  constructor
  · intro h
    apply (Spec.Tensor.dimScalarEquiv (α := ℝ) nStates).injective
    funext state
    have habs :
        |valueAt values₁ state - valueAt values₂ state| ≤ valueSupDist values₁ values₂ :=
      abs_sub_valueAt_le_valueSupDist (values₁ := values₁) (values₂ := values₂) state
    have habs0 : |valueAt values₁ state - valueAt values₂ state| ≤ 0 := by
      simpa [h] using habs
    have habseq : |valueAt values₁ state - valueAt values₂ state| = 0 :=
      le_antisymm habs0 (abs_nonneg _)
    have hdiff : valueAt values₁ state - valueAt values₂ state = 0 :=
      abs_eq_zero.mp habseq
    have hcoord : valueAt values₁ state = valueAt values₂ state :=
      sub_eq_zero.mp hdiff
    simpa [dimScalarEquiv_apply_eq_valueAt] using hcoord
  · intro h
    subst h
    simp [valueSupDist, valueAt]

/-- `bellmanPolicy` iterates are geometric contractions in `valueSupDist`. -/
theorem bellmanPolicy_iterate_contraction
    [Fact (0 < nStates)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (policy : Policy nStates nActions)
    (k : Nat)
    (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist ((Spec.RL.FiniteStochastic.bellmanPolicy mdp policy)^[k] values₁)
      ((Spec.RL.FiniteStochastic.bellmanPolicy mdp policy)^[k] values₂)
      ≤ mdp.discount ^ k * valueSupDist values₁ values₂ := by
  induction k generalizing values₁ values₂ with
  | zero =>
      simp
  | succ k ih =>
      let f := Spec.RL.FiniteStochastic.bellmanPolicy mdp policy
      have ih' :
          valueSupDist (f^[k] (f values₁)) (f^[k] (f values₂))
            ≤ mdp.discount ^ k * valueSupDist (f values₁) (f values₂) :=
        ih (values₁ := f values₁) (values₂ := f values₂)
      have hcon :
          valueSupDist (f values₁) (f values₂) ≤ mdp.discount * valueSupDist values₁ values₂ := by
        simpa [f] using
          bellmanPolicy_contraction (nStates := nStates) (nActions := nActions) mdp valid policy values₁ values₂
      have hγk : 0 ≤ mdp.discount ^ k := pow_nonneg valid.discount_nonneg k
      have hmul :
          mdp.discount ^ k * valueSupDist (f values₁) (f values₂)
            ≤ mdp.discount ^ k * (mdp.discount * valueSupDist values₁ values₂) :=
        mul_le_mul_of_nonneg_left hcon hγk
      -- `f^[k+1] = f^[k] ∘ f`, so the succ case is `f^[k] (f values)`.
      simpa [Function.iterate_succ_apply, pow_succ, mul_assoc] using
        (le_trans ih' (le_trans hmul (by
          -- `γ^k * (γ * d) = γ^(k+1) * d`
          simp)))

/--
If a discounted Bellman policy operator has a fixed point, it is unique.

This is the standard “contraction has at most one fixed point” argument.
-/
theorem bellmanPolicy_fixedPoint_unique
    [Fact (0 < nStates)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (policy : Policy nStates nActions)
    (v w : ValueFunction ℝ nStates)
    (hv : Spec.RL.FiniteStochastic.bellmanPolicy mdp policy v = v)
    (hw : Spec.RL.FiniteStochastic.bellmanPolicy mdp policy w = w) :
    v = w := by
  have hcon :=
    bellmanPolicy_contraction (nStates := nStates) (nActions := nActions) mdp valid policy v w
  have hle : valueSupDist v w ≤ mdp.discount * valueSupDist v w := by
    simpa [hv, hw] using hcon
  have hsub : valueSupDist v w - mdp.discount * valueSupDist v w ≤ 0 :=
    sub_nonpos.mpr hle
  have hmul : (1 - mdp.discount) * valueSupDist v w ≤ 0 := by
    have : (1 - mdp.discount) * valueSupDist v w = valueSupDist v w - mdp.discount * valueSupDist v w := by
      ring
    simpa [this] using hsub
  have hmul_nonneg : 0 ≤ (1 - mdp.discount) * valueSupDist v w := by
    have h1 : 0 ≤ (1 - mdp.discount) :=
      sub_nonneg.mpr (le_of_lt valid.discount_lt_one)
    have h2 : 0 ≤ valueSupDist v w := valueSupDist_nonneg (values₁ := v) (values₂ := w)
    exact mul_nonneg h1 h2
  have hmul_eq : (1 - mdp.discount) * valueSupDist v w = 0 :=
    le_antisymm hmul hmul_nonneg
  have hne : (1 - mdp.discount) ≠ (0 : ℝ) := by
    intro h0
    have hEq : mdp.discount = (1 : ℝ) := by
      have : (1 : ℝ) = mdp.discount := sub_eq_zero.mp h0
      simpa using this.symm
    exact (ne_of_lt valid.discount_lt_one) hEq
  have hd0 : valueSupDist v w = 0 :=
    (mul_eq_zero.mp hmul_eq).resolve_left hne
  exact (valueSupDist_eq_zero_iff (nStates := nStates) (values₁ := v) (values₂ := w)).1 hd0

/--
Error bound to a fixed point: iterating the Bellman policy operator reduces sup-distance
geometrically (`γ^k`).
-/
theorem bellmanPolicy_iterate_error_to_fixedPoint
    [Fact (0 < nStates)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (policy : Policy nStates nActions)
    (v vStar : ValueFunction ℝ nStates)
    (hvStar : Spec.RL.FiniteStochastic.bellmanPolicy mdp policy vStar = vStar)
    (k : Nat) :
    valueSupDist ((Spec.RL.FiniteStochastic.bellmanPolicy mdp policy)^[k] v) vStar
      ≤ mdp.discount ^ k * valueSupDist v vStar := by
  -- Contract iterates, and use that a fixed point stays fixed under iteration.
  have h :=
    bellmanPolicy_iterate_contraction (nStates := nStates) (nActions := nActions) mdp valid policy k v vStar
  have hfix : (Spec.RL.FiniteStochastic.bellmanPolicy mdp policy)^[k] vStar = vStar :=
    Function.iterate_fixed (f := Spec.RL.FiniteStochastic.bellmanPolicy mdp policy) hvStar k
  simpa [hfix] using h

/-- `bellmanOptimality` iterates are geometric contractions in `valueSupDist`. -/
theorem bellmanOptimality_iterate_contraction
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (k : Nat)
    (values₁ values₂ : ValueFunction ℝ nStates) :
    valueSupDist ((Spec.RL.FiniteStochastic.bellmanOptimality mdp)^[k] values₁)
      ((Spec.RL.FiniteStochastic.bellmanOptimality mdp)^[k] values₂)
      ≤ mdp.discount ^ k * valueSupDist values₁ values₂ := by
  induction k generalizing values₁ values₂ with
  | zero =>
      simp
  | succ k ih =>
      let f := Spec.RL.FiniteStochastic.bellmanOptimality mdp
      have ih' :
          valueSupDist (f^[k] (f values₁)) (f^[k] (f values₂))
            ≤ mdp.discount ^ k * valueSupDist (f values₁) (f values₂) :=
        ih (values₁ := f values₁) (values₂ := f values₂)
      have hcon :
          valueSupDist (f values₁) (f values₂) ≤ mdp.discount * valueSupDist values₁ values₂ := by
        simpa [f] using
          bellmanOptimality_contraction (nStates := nStates) (nActions := nActions) mdp valid values₁ values₂
      have hγk : 0 ≤ mdp.discount ^ k := pow_nonneg valid.discount_nonneg k
      have hmul :
          mdp.discount ^ k * valueSupDist (f values₁) (f values₂)
            ≤ mdp.discount ^ k * (mdp.discount * valueSupDist values₁ values₂) :=
        mul_le_mul_of_nonneg_left hcon hγk
      simpa [Function.iterate_succ_apply, pow_succ, mul_assoc] using
        (le_trans ih' (le_trans hmul (by simp)))

/--
If a discounted Bellman optimality operator has a fixed point, it is unique.

This is the “contraction has at most one fixed point” argument for `T*`.
-/
theorem bellmanOptimality_fixedPoint_unique
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (v w : ValueFunction ℝ nStates)
    (hv : Spec.RL.FiniteStochastic.bellmanOptimality mdp v = v)
    (hw : Spec.RL.FiniteStochastic.bellmanOptimality mdp w = w) :
    v = w := by
  have hcon :=
    bellmanOptimality_contraction (nStates := nStates) (nActions := nActions) mdp valid v w
  have hle : valueSupDist v w ≤ mdp.discount * valueSupDist v w := by
    simpa [hv, hw] using hcon
  have hsub : valueSupDist v w - mdp.discount * valueSupDist v w ≤ 0 :=
    sub_nonpos.mpr hle
  have hmul : (1 - mdp.discount) * valueSupDist v w ≤ 0 := by
    have : (1 - mdp.discount) * valueSupDist v w = valueSupDist v w - mdp.discount * valueSupDist v w := by
      ring
    simpa [this] using hsub
  have hmul_nonneg : 0 ≤ (1 - mdp.discount) * valueSupDist v w := by
    have h1 : 0 ≤ (1 - mdp.discount) :=
      sub_nonneg.mpr (le_of_lt valid.discount_lt_one)
    have h2 : 0 ≤ valueSupDist v w := valueSupDist_nonneg (values₁ := v) (values₂ := w)
    exact mul_nonneg h1 h2
  have hmul_eq : (1 - mdp.discount) * valueSupDist v w = 0 :=
    le_antisymm hmul hmul_nonneg
  have hne : (1 - mdp.discount) ≠ (0 : ℝ) := by
    intro h0
    have hEq : mdp.discount = (1 : ℝ) := by
      have : (1 : ℝ) = mdp.discount := sub_eq_zero.mp h0
      simpa using this.symm
    exact (ne_of_lt valid.discount_lt_one) hEq
  have hd0 : valueSupDist v w = 0 :=
    (mul_eq_zero.mp hmul_eq).resolve_left hne
  exact (valueSupDist_eq_zero_iff (nStates := nStates) (values₁ := v) (values₂ := w)).1 hd0

/--
Error bound to a fixed point: iterating Bellman optimality reduces sup-distance geometrically.
-/
theorem bellmanOptimality_iterate_error_to_fixedPoint
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : MDP nStates nActions)
    (valid : Valid mdp)
    (v vStar : ValueFunction ℝ nStates)
    (hvStar : Spec.RL.FiniteStochastic.bellmanOptimality mdp vStar = vStar)
    (k : Nat) :
    valueSupDist ((Spec.RL.FiniteStochastic.bellmanOptimality mdp)^[k] v) vStar
      ≤ mdp.discount ^ k * valueSupDist v vStar := by
  have h :=
    bellmanOptimality_iterate_contraction (nStates := nStates) (nActions := nActions) mdp valid k v vStar
  have hfix : (Spec.RL.FiniteStochastic.bellmanOptimality mdp)^[k] vStar = vStar :=
    Function.iterate_fixed (f := Spec.RL.FiniteStochastic.bellmanOptimality mdp) hvStar k
  simpa [hfix] using h

end FixedPoints

end FiniteStochastic
end RL
end Proofs
