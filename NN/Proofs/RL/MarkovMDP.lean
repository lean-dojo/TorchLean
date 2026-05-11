/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic
public import Mathlib.MeasureTheory.Integral.Bochner.Basic
public import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
public import NN.Proofs.RL.FinsetSup
public import NN.Spec.RL.MarkovMDP

/-!
# Markov-Kernel MDP Proofs (Measure Theory)

This module proves the key discounted Bellman facts for TorchLean's measure-theoretic MDP layer
(`NN.Spec.RL.MarkovMDP`), built on mathlib's Markov kernels.

We formalize the standard argument used in dynamic programming:

- if two value functions are uniformly close (bounded sup distance),
  then their Bellman backups are uniformly close,
- in particular, the Bellman expectation operator for a fixed deterministic policy is a
  `γ`-contraction in the sup metric (on bounded value functions),
- for finite action spaces, Bellman optimality is also a `γ`-contraction in the same metric.

References:

- Puterman, *Markov Decision Processes* (1994), Section 6.2 (discounted case):
  https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887
- Bertsekas, *Dynamic Programming and Optimal Control*, Vol. 1 (contraction mapping argument):
  http://web.mit.edu/dimitrib/www/dpoc.html
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018),
  Bellman expectation/optimality operators in the discounted setting:
  http://incompleteideas.net/book/the-book-2nd.html
- mathlib: `ProbabilityTheory.Kernel` and `MeasureTheory` integration lemmas such as
  `abs_integral_le_integral_abs` and `integral_mono`.
  Docs entry point: https://leanprover-community.github.io/mathlib4_docs/Mathlib/Probability/Kernel/Basic.html
-/

@[expose] public section

namespace Proofs
namespace RL
namespace Markov

open MeasureTheory ProbabilityTheory
open Spec.RL
open Spec.RL.Markov

open scoped BigOperators

section SupDist

variable {S : Type}

/-- Sup distance on value functions, using `sSup` over pointwise absolute differences. -/
noncomputable def valueSupDist [Nonempty S] (values₁ values₂ : ValueFunction S) : ℝ :=
  sSup (Set.range fun s => |values₁ s - values₂ s|)

/-- Every pointwise absolute difference is bounded by the sup distance (boundedness assumed). -/
theorem abs_sub_le_valueSupDist [Nonempty S]
    (values₁ values₂ : ValueFunction S)
    (hBdd : BddAbove (Set.range fun s => |values₁ s - values₂ s|))
    (state : S) :
    |values₁ state - values₂ state| ≤ valueSupDist values₁ values₂ := by
  have hmem : |values₁ state - values₂ state| ∈ Set.range fun s => |values₁ s - values₂ s| :=
    ⟨state, rfl⟩
  unfold valueSupDist
  exact le_csSup hBdd hmem

/-- A simple boundedness helper: if both functions are bounded, then their difference is bounded. -/
theorem bddAbove_abs_sub_of_bddAbove_abs
    (values₁ values₂ : ValueFunction S)
    (h₁ : BddAbove (Set.range fun s => |values₁ s|))
    (h₂ : BddAbove (Set.range fun s => |values₂ s|)) :
    BddAbove (Set.range fun s => |values₁ s - values₂ s|) := by
  rcases h₁ with ⟨B₁, hB₁⟩
  rcases h₂ with ⟨B₂, hB₂⟩
  refine ⟨B₁ + B₂, ?_⟩
  rintro _ ⟨s, rfl⟩
  have h1 : |values₁ s| ≤ B₁ := hB₁ (a := |values₁ s|) ⟨s, rfl⟩
  have h2 : |values₂ s| ≤ B₂ := hB₂ (a := |values₂ s|) ⟨s, rfl⟩
  have htriangle : |values₁ s - values₂ s| ≤ |values₁ s| + |values₂ s| := by
    simpa [sub_eq_add_neg] using (abs_add_le (values₁ s) (-values₂ s))
  linarith

/--
`valueSupDist = 0` iff two (bounded) value functions are equal.

Because `valueSupDist` is defined via `sSup` over pointwise absolute differences, we need a
boundedness hypothesis to use `le_csSup` (see `abs_sub_le_valueSupDist`).
-/
theorem valueSupDist_eq_zero_iff [Nonempty S]
    (values₁ values₂ : ValueFunction S)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|)) :
    valueSupDist values₁ values₂ = 0 ↔ values₁ = values₂ := by
  have hBddDiff : BddAbove (Set.range fun s => |values₁ s - values₂ s|) :=
    bddAbove_abs_sub_of_bddAbove_abs (S := S) values₁ values₂ hBdd₁ hBdd₂
  constructor
  · intro h0
    funext state
    have habs : |values₁ state - values₂ state| ≤ valueSupDist values₁ values₂ :=
      abs_sub_le_valueSupDist (S := S) (values₁ := values₁) (values₂ := values₂) hBddDiff state
    have habs0 : |values₁ state - values₂ state| ≤ 0 := by
      simpa [h0] using habs
    have habseq : |values₁ state - values₂ state| = 0 :=
      le_antisymm habs0 (abs_nonneg _)
    have hdiff : values₁ state - values₂ state = 0 :=
      abs_eq_zero.mp habseq
    exact sub_eq_zero.mp hdiff
  · intro hEq
    subst hEq
    simp [valueSupDist]

end SupDist

section MarkovMDP

variable {S A : Type} [MeasurableSpace S] [MeasurableSpace A]

private lemma integrable_of_abs_bdd
    {μ : Measure S} [IsFiniteMeasure μ]
    (values : ValueFunction S)
    (hMeas : Measurable values)
    (hBdd : BddAbove (Set.range fun s => |values s|)) :
    Integrable values μ := by
  rcases hBdd with ⟨B, hB⟩
  have hbound : ∀ s, ‖values s‖ ≤ B := by
    intro s
    have : |values s| ≤ B := hB (a := |values s|) ⟨s, rfl⟩
    simpa [Real.norm_eq_abs] using this
  refine Integrable.mono' (integrable_const (μ := μ) B) (hMeas.aestronglyMeasurable) ?_
  exact ae_of_all _ hbound

/-- Coordinatewise expectation difference is bounded by the sup distance. -/
theorem expectedNextValue_abs_sub_le [Nonempty S]
    (mdp : MDP S A)
    (valid : Valid (S := S) (A := A) mdp)
    (values₁ values₂ : ValueFunction S)
    (hMeas₁ : Measurable values₁)
    (hMeas₂ : Measurable values₂)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|))
    (state : S)
    (action : A) :
    |expectedNextValue mdp values₁ state action - expectedNextValue mdp values₂ state action|
      ≤ valueSupDist values₁ values₂ := by
  haveI : IsMarkovKernel mdp.transition := valid.isMarkov
  let μ : Measure S := mdp.transition (state, action)
  haveI : IsProbabilityMeasure μ := by
    simpa [μ] using (by infer_instance : IsProbabilityMeasure (mdp.transition (state, action)))
  haveI : IsFiniteMeasure μ := by
    simpa [μ] using (by infer_instance : IsFiniteMeasure (mdp.transition (state, action)))
  have hBddDiff : BddAbove (Set.range fun s => |values₁ s - values₂ s|) :=
    bddAbove_abs_sub_of_bddAbove_abs (S := S) values₁ values₂ hBdd₁ hBdd₂
  have hint₁ : Integrable values₁ μ := integrable_of_abs_bdd (μ := μ) values₁ hMeas₁ hBdd₁
  have hint₂ : Integrable values₂ μ := integrable_of_abs_bdd (μ := μ) values₂ hMeas₂ hBdd₂
  have hrewrite :
      expectedNextValue mdp values₁ state action - expectedNextValue mdp values₂ state action =
        ∫ nextState, (values₁ nextState - values₂ nextState) ∂μ := by
    -- Rewrite `∫ values₁ - ∫ values₂` into a single integral over the difference.
    simpa [expectedNextValue, transitionMeasure, μ] using (integral_sub (μ := μ) hint₁ hint₂).symm
  rw [hrewrite]
  have hintAbs : Integrable (fun s => |values₁ s - values₂ s|) μ := by
    simpa [Real.norm_eq_abs] using (hint₁.sub hint₂).abs
  calc
    |∫ nextState, (values₁ nextState - values₂ nextState) ∂μ|
        ≤ ∫ nextState, |values₁ nextState - values₂ nextState| ∂μ := by
          simpa using (abs_integral_le_integral_abs (μ := μ)
            (f := fun nextState => values₁ nextState - values₂ nextState))
    _ ≤ ∫ _ : S, valueSupDist values₁ values₂ ∂μ := by
          have hbound :
              ∀ nextState, |values₁ nextState - values₂ nextState| ≤ valueSupDist values₁ values₂ :=
            fun nextState =>
              abs_sub_le_valueSupDist (S := S) (values₁ := values₁) (values₂ := values₂)
                hBddDiff nextState
          refine integral_mono (μ := μ) hintAbs (integrable_const (μ := μ) (valueSupDist values₁ values₂)) ?_
          intro nextState
          exact hbound nextState
    _ = valueSupDist values₁ values₂ := by
          simp [integral_const, MeasureTheory.probReal_univ, smul_eq_mul]

/-- Bellman state-action values are Lipschitz with constant `γ` in the sup metric. -/
theorem actionValue_abs_sub_le [Nonempty S]
    (mdp : MDP S A)
    (valid : Valid (S := S) (A := A) mdp)
    (values₁ values₂ : ValueFunction S)
    (hMeas₁ : Measurable values₁)
    (hMeas₂ : Measurable values₂)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|))
    (state : S)
    (action : A) :
    |actionValue mdp values₁ state action - actionValue mdp values₂ state action|
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  by_cases hdone : mdp.terminated state action
  · have hnonneg : 0 ≤ mdp.discount * valueSupDist values₁ values₂ :=
      mul_nonneg valid.discount_nonneg (by
        have hBddDiff : BddAbove (Set.range fun s => |values₁ s - values₂ s|) :=
          bddAbove_abs_sub_of_bddAbove_abs (S := S) values₁ values₂ hBdd₁ hBdd₂
        -- `valueSupDist` is a `sSup` of nonnegative quantities, hence nonnegative.
        have hx : 0 ≤ |values₁ (Classical.choice (inferInstance : Nonempty S)) -
            values₂ (Classical.choice (inferInstance : Nonempty S))| := abs_nonneg _
        have hxle : |values₁ (Classical.choice (inferInstance : Nonempty S)) -
            values₂ (Classical.choice (inferInstance : Nonempty S))| ≤ valueSupDist values₁ values₂ := by
          simpa using abs_sub_le_valueSupDist (S := S) (values₁ := values₁) (values₂ := values₂)
            hBddDiff (Classical.choice (inferInstance : Nonempty S))
        exact hx.trans hxle)
    simp [actionValue, discountedBackup, continueMask, hdone, hnonneg]
  · have hexp :=
      expectedNextValue_abs_sub_le (S := S) (A := A) mdp valid values₁ values₂
        hMeas₁ hMeas₂ hBdd₁ hBdd₂ state action
    have hrewrite :
        actionValue mdp values₁ state action - actionValue mdp values₂ state action =
          mdp.discount *
            (expectedNextValue mdp values₁ state action - expectedNextValue mdp values₂ state action) := by
      simp [actionValue, discountedBackup, continueMask, hdone]
      ring
    rw [hrewrite]
    calc
      |mdp.discount * (expectedNextValue mdp values₁ state action -
            expectedNextValue mdp values₂ state action)|
          = mdp.discount *
              |expectedNextValue mdp values₁ state action -
                expectedNextValue mdp values₂ state action| := by
              simp [abs_mul, abs_of_nonneg valid.discount_nonneg]
      _ ≤ mdp.discount * valueSupDist values₁ values₂ := by
            exact mul_le_mul_of_nonneg_left hexp valid.discount_nonneg

/-- Bellman expectation for a deterministic policy is a `γ`-contraction in the sup metric:

`valueSupDist (T^π values₁) (T^π values₂) ≤ γ * valueSupDist values₁ values₂`. -/
theorem bellmanPolicy_contraction [Nonempty S]
    (mdp : MDP S A)
    (valid : Valid (S := S) (A := A) mdp)
    (policy : Policy S A)
    (values₁ values₂ : ValueFunction S)
    (hMeas₁ : Measurable values₁)
    (hMeas₂ : Measurable values₂)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|)) :
    valueSupDist (bellmanPolicy mdp policy values₁) (bellmanPolicy mdp policy values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  unfold valueSupDist
  refine csSup_le (s := Set.range fun s : S =>
      |bellmanPolicy mdp policy values₁ s - bellmanPolicy mdp policy values₂ s|)
    (by
      rcases (inferInstance : Nonempty S) with ⟨s0⟩
      exact ⟨_, ⟨s0, rfl⟩⟩)
    ?_
  rintro _ ⟨state, rfl⟩
  simpa [bellmanPolicy] using
    actionValue_abs_sub_le (S := S) (A := A) mdp valid values₁ values₂
      hMeas₁ hMeas₂ hBdd₁ hBdd₂ state (policy state)

/-- At a fixed state, Bellman optimality is a contraction with modulus `γ` (finite action space). -/
theorem bellmanOptimality_abs_sub_le [Nonempty S]
    (mdp : MDP S A)
    (valid : Valid (S := S) (A := A) mdp)
    [Fintype A] [Nonempty A]
    (values₁ values₂ : ValueFunction S)
    (hMeas₁ : Measurable values₁)
    (hMeas₂ : Measurable values₂)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|))
    (state : S) :
    |bellmanOptimality mdp values₁ state - bellmanOptimality mdp values₂ state|
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  let bound := mdp.discount * valueSupDist values₁ values₂
  let f : A → ℝ := fun action => actionValue mdp values₁ state action
  let g : A → ℝ := fun action => actionValue mdp values₂ state action
  have habsAction :
      ∀ action : A,
        |f action - g action| ≤ bound := by
    intro action
    simpa [f, g, bound] using
      actionValue_abs_sub_le (S := S) (A := A) mdp valid values₁ values₂
        hMeas₁ hMeas₂ hBdd₁ hBdd₂ state action
  have hfg : ∀ action ∈ (Finset.univ : Finset A), f action ≤ g action + bound := by
    intro action _
    have habs := habsAction action
    linarith [abs_sub_le_iff.mp habs]
  have hgf : ∀ action ∈ (Finset.univ : Finset A), g action ≤ f action + bound := by
    intro action _
    have habs := habsAction action
    linarith [abs_sub_le_iff.mp habs]
  have hs1 :
      (Finset.univ : Finset A).sup' Finset.univ_nonempty f
        ≤ (Finset.univ : Finset A).sup' Finset.univ_nonempty g + bound := by
    exact _root_.Proofs.RL.sup'_le_add_const
      (Finset.univ : Finset A) Finset.univ_nonempty f g bound hfg
  have hs2 :
      (Finset.univ : Finset A).sup' Finset.univ_nonempty g
        ≤ (Finset.univ : Finset A).sup' Finset.univ_nonempty f + bound := by
    exact _root_.Proofs.RL.sup'_le_add_const
      (Finset.univ : Finset A) Finset.univ_nonempty g f bound hgf
  have habs :
      |(Finset.univ : Finset A).sup' Finset.univ_nonempty f -
          (Finset.univ : Finset A).sup' Finset.univ_nonempty g|
        ≤ bound := by
    exact abs_sub_le_iff.mpr
      ⟨sub_le_iff_le_add'.mpr hs1, sub_le_iff_le_add'.mpr hs2⟩
  simpa [bellmanOptimality, f, g, bound] using habs

/-- Bellman optimality is a `γ`-contraction in the sup metric (finite action space):

`valueSupDist (T* values₁) (T* values₂) ≤ γ * valueSupDist values₁ values₂`. -/
theorem bellmanOptimality_contraction [Nonempty S]
    (mdp : MDP S A)
    (valid : Valid (S := S) (A := A) mdp)
    [Fintype A] [Nonempty A]
    (values₁ values₂ : ValueFunction S)
    (hMeas₁ : Measurable values₁)
    (hMeas₂ : Measurable values₂)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|)) :
    valueSupDist (bellmanOptimality mdp values₁) (bellmanOptimality mdp values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ := by
  unfold valueSupDist
  refine csSup_le (s := Set.range fun s : S =>
      |bellmanOptimality mdp values₁ s - bellmanOptimality mdp values₂ s|)
    (by
      rcases (inferInstance : Nonempty S) with ⟨s0⟩
      exact ⟨_, ⟨s0, rfl⟩⟩)
    ?_
  rintro _ ⟨state, rfl⟩
  simpa using
    bellmanOptimality_abs_sub_le (S := S) (A := A) mdp valid values₁ values₂
      hMeas₁ hMeas₂ hBdd₁ hBdd₂ state

/-!
## Fixed Point Uniqueness

The contraction theorems imply that (when `0 ≤ γ < 1`) both Bellman operators have **at most one**
fixed point on the class of bounded measurable value functions. This is the standard “contraction
has at most one fixed point” argument from discounted dynamic programming.
-/

section FixedPoints

/--
If the Bellman expectation operator for a fixed deterministic policy has a fixed point, it is
unique (among bounded measurable value functions).
-/
theorem bellmanPolicy_fixedPoint_unique [Nonempty S]
    (mdp : MDP S A)
    (valid : Valid (S := S) (A := A) mdp)
    (policy : Policy S A)
    (v w : ValueFunction S)
    (hv : bellmanPolicy mdp policy v = v)
    (hw : bellmanPolicy mdp policy w = w)
    (hMeasV : Measurable v)
    (hMeasW : Measurable w)
    (hBddV : BddAbove (Set.range fun s => |v s|))
    (hBddW : BddAbove (Set.range fun s => |w s|)) :
    v = w := by
  have hcon :=
    bellmanPolicy_contraction (S := S) (A := A) mdp valid policy v w hMeasV hMeasW hBddV hBddW
  have hle : valueSupDist v w ≤ mdp.discount * valueSupDist v w := by
    simpa [hv, hw] using hcon
  have hsub : valueSupDist v w - mdp.discount * valueSupDist v w ≤ 0 :=
    sub_nonpos.mpr hle
  have hmul : (1 - mdp.discount) * valueSupDist v w ≤ 0 := by
    have : (1 - mdp.discount) * valueSupDist v w = valueSupDist v w - mdp.discount * valueSupDist v w := by
      ring
    simpa [this] using hsub

  have hBddDiff : BddAbove (Set.range fun s => |v s - w s|) :=
    bddAbove_abs_sub_of_bddAbove_abs (S := S) v w hBddV hBddW
  have hdist_nonneg : 0 ≤ valueSupDist v w := by
    rcases (inferInstance : Nonempty S) with ⟨s0⟩
    have hx : 0 ≤ |v s0 - w s0| := abs_nonneg _
    have hxle : |v s0 - w s0| ≤ valueSupDist v w :=
      abs_sub_le_valueSupDist (S := S) (values₁ := v) (values₂ := w) hBddDiff s0
    exact hx.trans hxle

  have hmul_nonneg : 0 ≤ (1 - mdp.discount) * valueSupDist v w := by
    have h1 : 0 ≤ (1 - mdp.discount) :=
      sub_nonneg.mpr (le_of_lt valid.discount_lt_one)
    exact mul_nonneg h1 hdist_nonneg
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

  exact
    (valueSupDist_eq_zero_iff (S := S) (values₁ := v) (values₂ := w) hBddV hBddW).1 hd0

/--
If the Bellman optimality operator has a fixed point, it is unique (finite action space).
-/
theorem bellmanOptimality_fixedPoint_unique [Nonempty S]
    (mdp : MDP S A)
    (valid : Valid (S := S) (A := A) mdp)
    [Fintype A] [Nonempty A]
    (v w : ValueFunction S)
    (hv : bellmanOptimality mdp v = v)
    (hw : bellmanOptimality mdp w = w)
    (hMeasV : Measurable v)
    (hMeasW : Measurable w)
    (hBddV : BddAbove (Set.range fun s => |v s|))
    (hBddW : BddAbove (Set.range fun s => |w s|)) :
    v = w := by
  have hcon :=
    bellmanOptimality_contraction (S := S) (A := A) mdp valid v w hMeasV hMeasW hBddV hBddW
  have hle : valueSupDist v w ≤ mdp.discount * valueSupDist v w := by
    simpa [hv, hw] using hcon
  have hsub : valueSupDist v w - mdp.discount * valueSupDist v w ≤ 0 :=
    sub_nonpos.mpr hle
  have hmul : (1 - mdp.discount) * valueSupDist v w ≤ 0 := by
    have : (1 - mdp.discount) * valueSupDist v w = valueSupDist v w - mdp.discount * valueSupDist v w := by
      ring
    simpa [this] using hsub

  have hBddDiff : BddAbove (Set.range fun s => |v s - w s|) :=
    bddAbove_abs_sub_of_bddAbove_abs (S := S) v w hBddV hBddW
  have hdist_nonneg : 0 ≤ valueSupDist v w := by
    rcases (inferInstance : Nonempty S) with ⟨s0⟩
    have hx : 0 ≤ |v s0 - w s0| := abs_nonneg _
    have hxle : |v s0 - w s0| ≤ valueSupDist v w :=
      abs_sub_le_valueSupDist (S := S) (values₁ := v) (values₂ := w) hBddDiff s0
    exact hx.trans hxle

  have hmul_nonneg : 0 ≤ (1 - mdp.discount) * valueSupDist v w := by
    have h1 : 0 ≤ (1 - mdp.discount) :=
      sub_nonneg.mpr (le_of_lt valid.discount_lt_one)
    exact mul_nonneg h1 hdist_nonneg
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

  exact
    (valueSupDist_eq_zero_iff (S := S) (values₁ := v) (values₂ := w) hBddV hBddW).1 hd0

end FixedPoints

end MarkovMDP

end Markov
end RL
end Proofs
