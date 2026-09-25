/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RL.Core
public import NN.Proofs.RL.FinsetSup
public import NN.Spec.RL.MarkovMDP
public import Mathlib.Analysis.Normed.Lp.lpSpace
public import Mathlib.MeasureTheory.Constructions.BorelSpace.Metrizable
public import Mathlib.MeasureTheory.Order.Lattice
public import Mathlib.Probability.Kernel.MeasurableIntegral
public import Mathlib.Topology.MetricSpace.Contracting

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
  Docs entry point:
  https://leanprover-community.github.io/mathlib4_docs/Mathlib/Probability/Kernel/Basic.html
-/

@[expose] public section

namespace Proofs
namespace RL
namespace Markov

open MeasureTheory ProbabilityTheory
open Filter Topology
open Spec.RL
open Spec.RL.Markov

open Proofs.RL.Core (discountedBackup_abs_sub_le eq_zero_of_le_mul_self)

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
    |values₁ state - values₂ state| ≤ valueSupDist values₁ values₂ :=
  le_csSup hBdd ⟨state, rfl⟩

/-- The sup distance is nonnegative. -/
theorem valueSupDist_nonneg [Nonempty S] (values₁ values₂ : ValueFunction S) :
    0 ≤ valueSupDist values₁ values₂ :=
  Real.sSup_nonneg (Set.forall_mem_range.2 fun _ => abs_nonneg _)

/-- A uniform bound on pointwise differences bounds the sup distance. -/
theorem valueSupDist_le [Nonempty S] {values₁ values₂ : ValueFunction S} {bound : ℝ}
    (h : ∀ state, |values₁ state - values₂ state| ≤ bound) :
    valueSupDist values₁ values₂ ≤ bound :=
  csSup_le (Set.range_nonempty _) (Set.forall_mem_range.2 h)

/-- A simple boundedness helper: if both functions are bounded, then their difference is bounded. -/
theorem bddAbove_abs_sub_of_bddAbove_abs
    (values₁ values₂ : ValueFunction S)
    (h₁ : BddAbove (Set.range fun s => |values₁ s|))
    (h₂ : BddAbove (Set.range fun s => |values₂ s|)) :
    BddAbove (Set.range fun s => |values₁ s - values₂ s|) := by
  rcases h₁ with ⟨B₁, hB₁⟩
  rcases h₂ with ⟨B₂, hB₂⟩
  refine ⟨B₁ + B₂, Set.forall_mem_range.2 fun s => (abs_sub _ _).trans ?_⟩
  exact add_le_add (hB₁ ⟨s, rfl⟩) (hB₂ ⟨s, rfl⟩)

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
  refine ⟨fun h0 => funext fun state => sub_eq_zero.mp (abs_nonpos_iff.mp ?_), fun h => ?_⟩
  · simpa [h0] using abs_sub_le_valueSupDist values₁ values₂
      (bddAbove_abs_sub_of_bddAbove_abs values₁ values₂ hBdd₁ hBdd₂) state
  · subst h
    simp [valueSupDist]

/-- A bounded fixed point of an operator that contracts `valueSupDist` by `γ < 1` is unique. -/
private theorem eq_of_isFixed [Nonempty S] {T : ValueFunction S → ValueFunction S} {γ : ℝ}
    (hγ : γ < 1) {v w : ValueFunction S} (hv : T v = v) (hw : T w = w)
    (hBddV : BddAbove (Set.range fun s => |v s|))
    (hBddW : BddAbove (Set.range fun s => |w s|))
    (hT : valueSupDist (T v) (T w) ≤ γ * valueSupDist v w) :
    v = w := by
  rw [hv, hw] at hT
  exact (valueSupDist_eq_zero_iff v w hBddV hBddW).1
    (eq_zero_of_le_mul_self hγ (valueSupDist_nonneg v w) hT)

end SupDist

section MarkovMDP

variable {S A : Type} [MeasurableSpace S] [MeasurableSpace A]

private theorem integrable_of_abs_bdd
    {μ : Measure S} [IsFiniteMeasure μ]
    (values : ValueFunction S)
    (hMeas : Measurable values)
    (hBdd : BddAbove (Set.range fun s => |values s|)) :
    Integrable values μ := by
  rcases hBdd with ⟨B, hB⟩
  exact Integrable.mono' (integrable_const B) hMeas.aestronglyMeasurable
    (ae_of_all _ fun s => (Real.norm_eq_abs _).trans_le (hB ⟨s, rfl⟩))

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
  have : IsMarkovKernel mdp.transition := valid.isMarkov
  let μ : Measure S := mdp.transition (state, action)
  have hint₁ : Integrable values₁ μ := integrable_of_abs_bdd values₁ hMeas₁ hBdd₁
  have hint₂ : Integrable values₂ μ := integrable_of_abs_bdd values₂ hMeas₂ hBdd₂
  have hBddDiff := bddAbove_abs_sub_of_bddAbove_abs values₁ values₂ hBdd₁ hBdd₂
  calc
    |expectedNextValue mdp values₁ state action - expectedNextValue mdp values₂ state action|
        = |∫ nextState, (values₁ nextState - values₂ nextState) ∂μ| := by
          rw [integral_sub hint₁ hint₂]
          rfl
    _ ≤ ∫ nextState, |values₁ nextState - values₂ nextState| ∂μ :=
          abs_integral_le_integral_abs
    _ ≤ ∫ _ : S, valueSupDist values₁ values₂ ∂μ :=
          integral_mono (hint₁.sub hint₂).abs (integrable_const _)
            (abs_sub_le_valueSupDist values₁ values₂ hBddDiff)
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
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  discountedBackup_abs_sub_le _ valid.discount_nonneg (valueSupDist_nonneg values₁ values₂)
    (expectedNextValue_abs_sub_le mdp valid values₁ values₂ hMeas₁ hMeas₂ hBdd₁ hBdd₂ state
      action)

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
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  valueSupDist_le fun state =>
    actionValue_abs_sub_le mdp valid values₁ values₂ hMeas₁ hMeas₂ hBdd₁ hBdd₂ state
      (policy state)

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
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  abs_sup'_sub_sup'_le _ _ _ _ _ fun action _ =>
    actionValue_abs_sub_le mdp valid values₁ values₂ hMeas₁ hMeas₂ hBdd₁ hBdd₂ state action

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
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  valueSupDist_le
    (bellmanOptimality_abs_sub_le mdp valid values₁ values₂ hMeas₁ hMeas₂ hBdd₁ hBdd₂)

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
    v = w :=
  eq_of_isFixed valid.discount_lt_one hv hw hBddV hBddW
    (bellmanPolicy_contraction mdp valid policy v w hMeasV hMeasW hBddV hBddW)

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
    v = w :=
  eq_of_isFixed valid.discount_lt_one hv hw hBddV hBddW
    (bellmanOptimality_contraction mdp valid v w hMeasV hMeasW hBddV hBddW)

end FixedPoints

/-!
## Existence and Value Iteration

Bounded measurable value functions, with the sup distance, form a complete metric space: they are
the measurable elements of Mathlib's `lp (fun _ : S => ℝ) ∞`, and a sup-norm limit of measurable
functions is measurable. Banach's fixed-point theorem on that space gives the fixed point.

Existence needs two assumptions that `Valid` does not carry: rewards must be bounded, otherwise a
backup of a bounded function need not be bounded, and the policy must be measurable, otherwise a
backup of a measurable function need not be measurable.
-/

section Existence

/-- Bounded measurable value functions, as a subtype of `ℓ^∞(S)`. -/
private abbrev BddMeas (S : Type) [MeasurableSpace S] :=
  {f : lp (fun _ : S => ℝ) ⊤ // Measurable (f : S → ℝ)}

private theorem isClosed_measurable :
    IsClosed {f : lp (fun _ : S => ℝ) ⊤ | Measurable (f : S → ℝ)} :=
  IsSeqClosed.isClosed fun _ _ hu hlim =>
    measurable_of_tendsto_metrizable hu <| tendsto_pi_nhds.2 fun s =>
      ((lp.lipschitzWith_one_eval ⊤ s).continuous.tendsto _).comp hlim

private instance : CompleteSpace (BddMeas S) := isClosed_measurable.isComplete.completeSpace_coe

private instance : Nonempty (BddMeas S) := ⟨⟨0, by rw [lp.coeFn_zero]; exact measurable_const⟩⟩

private theorem dist_eq_valueSupDist [Nonempty S] (f g : BddMeas S) :
    dist f g = valueSupDist (f.1 : S → ℝ) g.1 := by
  simp [Subtype.dist_eq, dist_eq_norm, lp.norm_eq_ciSup, valueSupDist, iSup, Real.norm_eq_abs]

private theorem bddAbove_abs (f : BddMeas S) : BddAbove (Set.range fun s => |(f.1 : S → ℝ) s|) := by
  simpa [Real.norm_eq_abs] using memℓp_infty_iff.1 f.1.2

/-- A bounded measurable function as an element of `BddMeas S`. -/
private def BddMeas.mk (v : ValueFunction S) (hMeas : Measurable v)
    (hBdd : BddAbove (Set.range fun s => |v s|)) : BddMeas S :=
  ⟨⟨v, memℓp_infty_iff.2 (by simpa [Real.norm_eq_abs] using hBdd)⟩, hMeas⟩

/-- An operator that preserves bounded measurable functions and contracts `valueSupDist` on them by
`γ < 1` has a bounded measurable fixed point, and its iterates converge to it. -/
private theorem exists_fixedPoint_of_contraction [Nonempty S]
    {T : ValueFunction S → ValueFunction S} {γ : ℝ}
    (hγ₀ : 0 ≤ γ) (hγ₁ : γ < 1)
    (hMeas : ∀ v, Measurable v → Measurable (T v))
    (hBdd : ∀ v, BddAbove (Set.range fun s => |v s|) → BddAbove (Set.range fun s => |T v s|))
    (hT : ∀ v w, Measurable v → Measurable w → BddAbove (Set.range fun s => |v s|) →
      BddAbove (Set.range fun s => |w s|) → valueSupDist (T v) (T w) ≤ γ * valueSupDist v w) :
    ∃ vStar : ValueFunction S, Measurable vStar ∧ BddAbove (Set.range fun s => |vStar s|) ∧
      T vStar = vStar ∧ ∀ v, Measurable v → BddAbove (Set.range fun s => |v s|) →
        Tendsto (fun k => valueSupDist (T^[k] v) vStar) atTop (𝓝 0) := by
  let F : BddMeas S → BddMeas S := fun f =>
    BddMeas.mk (T f.1) (hMeas _ f.2) (hBdd _ (bddAbove_abs f))
  have hF : ContractingWith γ.toNNReal F := by
    refine ⟨Real.toNNReal_lt_one.mpr hγ₁, LipschitzWith.of_dist_le_mul fun f g => ?_⟩
    simpa [dist_eq_valueSupDist, hγ₀, F, BddMeas.mk] using
      hT _ _ f.2 g.2 (bddAbove_abs f) (bddAbove_abs g)
  have hiter : ∀ (k : Nat) (f : BddMeas S), ((F^[k] f).1 : S → ℝ) = T^[k] f.1 := by
    intro k
    induction k with
    | zero => intro f; rfl
    | succ k ih => intro f; rw [Function.iterate_succ_apply, Function.iterate_succ_apply, ih]; rfl
  let p := ContractingWith.fixedPoint F hF
  refine ⟨p.1, p.2, bddAbove_abs p, ?_, fun v hv hb => ?_⟩
  · exact congrArg (fun f : BddMeas S => (f.1 : S → ℝ)) (ContractingWith.fixedPoint_isFixedPt hF)
  · have := tendsto_iff_dist_tendsto_zero.mp
      (ContractingWith.tendsto_iterate_fixedPoint hF (BddMeas.mk v hv hb))
    simpa [dist_eq_valueSupDist, hiter, BddMeas.mk] using this

private theorem measurable_actionValue (mdp : MDP S A) (valid : Valid (S := S) (A := A) mdp)
    {v : ValueFunction S} (hv : Measurable v) :
    Measurable fun sa : S × A => actionValue mdp v sa.1 sa.2 := by
  have hE : Measurable fun sa : S × A => ∫ y, v y ∂mdp.transition sa :=
    (hv.stronglyMeasurable.integral_kernel (κ := mdp.transition)).measurable
  have hMask : Measurable fun sa : S × A => (continueMask (mdp.terminated sa.1 sa.2) : ℝ) :=
    (measurable_of_countable fun b : Bool => (continueMask b : ℝ)).comp
      valid.measurable_terminated
  exact valid.measurable_reward.add ((measurable_const.mul hMask).mul hE)

private theorem abs_actionValue_le [Nonempty S] (mdp : MDP S A)
    (valid : Valid (S := S) (A := A) mdp)
    {v : ValueFunction S} {R B : ℝ} (hR : ∀ s a, |mdp.reward s a| ≤ R) (hB : ∀ s, |v s| ≤ B)
    (s : S) (a : A) :
    |actionValue mdp v s a| ≤ R + mdp.discount * B := by
  have : IsMarkovKernel mdp.transition := valid.isMarkov
  have hB0 : 0 ≤ B := (abs_nonneg _).trans (hB (Classical.arbitrary S))
  have hE : |expectedNextValue mdp v s a| ≤ B := by
    simpa [expectedNextValue, transitionMeasure] using
      norm_integral_le_of_norm_le_const (μ := mdp.transition (s, a))
        (ae_of_all _ fun x => (Real.norm_eq_abs _).trans_le (hB x))
  have hγB : 0 ≤ mdp.discount * B := mul_nonneg valid.discount_nonneg hB0
  cases hdone : mdp.terminated s a
  · simp only [actionValue, discountedBackup, continueMask, hdone, Bool.false_eq_true, ite_false,
      mul_one]
    refine (abs_add_le _ _).trans (add_le_add (hR s a) ?_)
    rw [abs_mul, abs_of_nonneg valid.discount_nonneg]
    exact mul_le_mul_of_nonneg_left hE valid.discount_nonneg
  · simpa [actionValue, discountedBackup, continueMask, hdone] using
      (hR s a).trans (le_add_of_nonneg_right hγB)

omit [MeasurableSpace S] in
private theorem bddAbove_of_abs_le {v : ValueFunction S} {C : ℝ} (h : ∀ s, |v s| ≤ C) :
    BddAbove (Set.range fun s => |v s|) :=
  ⟨C, Set.forall_mem_range.2 h⟩

/-- For bounded rewards and a measurable policy, policy evaluation has exactly one bounded
measurable fixed point. -/
theorem bellmanPolicy_existsUnique_fixedPoint [Nonempty S]
    (mdp : MDP S A) (valid : Valid (S := S) (A := A) mdp)
    (policy : Policy S A) (hPolicy : Measurable policy)
    {R : ℝ} (hR : ∀ s a, |mdp.reward s a| ≤ R) :
    ∃! v : ValueFunction S, (Measurable v ∧ BddAbove (Set.range fun s => |v s|)) ∧
      bellmanPolicy mdp policy v = v := by
  obtain ⟨vStar, hMeas, hBdd, hfix, -⟩ :=
    exists_fixedPoint_of_contraction (T := bellmanPolicy mdp policy)
      valid.discount_nonneg valid.discount_lt_one
      (fun v hv => (measurable_actionValue mdp valid hv).comp (measurable_id.prodMk hPolicy))
      (fun v ⟨B, hB⟩ => bddAbove_of_abs_le fun s =>
        abs_actionValue_le mdp valid hR (fun s => hB ⟨s, rfl⟩) s (policy s))
      (fun v w hv hw hbv hbw => bellmanPolicy_contraction mdp valid policy v w hv hw hbv hbw)
  exact ⟨vStar, ⟨⟨hMeas, hBdd⟩, hfix⟩, fun w ⟨⟨hw, hbw⟩, hwfix⟩ =>
    bellmanPolicy_fixedPoint_unique mdp valid policy w vStar hwfix hfix hw hMeas hbw hBdd⟩

/-- For bounded rewards and a finite action space, Bellman optimality has exactly one bounded
measurable fixed point, and value iteration converges to it from every bounded measurable start. -/
theorem bellmanOptimality_existsUnique_fixedPoint [Nonempty S]
    (mdp : MDP S A) (valid : Valid (S := S) (A := A) mdp)
    [Fintype A] [Nonempty A] [MeasurableSingletonClass A]
    {R : ℝ} (hR : ∀ s a, |mdp.reward s a| ≤ R) :
    ∃ vStar : ValueFunction S, Measurable vStar ∧ BddAbove (Set.range fun s => |vStar s|) ∧
      bellmanOptimality mdp vStar = vStar ∧
      (∀ w, Measurable w → BddAbove (Set.range fun s => |w s|) →
        bellmanOptimality mdp w = w → w = vStar) ∧
      ∀ v, Measurable v → BddAbove (Set.range fun s => |v s|) →
        Tendsto (fun k => valueSupDist ((bellmanOptimality mdp)^[k] v) vStar) atTop (𝓝 0) := by
  obtain ⟨vStar, hMeas, hBdd, hfix, hlim⟩ :=
    exists_fixedPoint_of_contraction (T := bellmanOptimality mdp)
      valid.discount_nonneg valid.discount_lt_one
      (fun v hv => by
        have h := measurable_actionValue mdp valid hv
        convert Finset.measurable_sup' (f := fun a s => actionValue mdp v s a)
          Finset.univ_nonempty fun a _ => h.comp (measurable_id.prodMk measurable_const) using 1
        funext s
        rw [Finset.sup'_apply]
        rfl)
      (fun v ⟨B, hB⟩ => bddAbove_of_abs_le fun s => abs_le.2
        ⟨(neg_le_of_abs_le (abs_actionValue_le mdp valid hR (fun s => hB ⟨s, rfl⟩) s
            (Classical.arbitrary A))).trans (Finset.le_sup' _ (Finset.mem_univ _)),
          Finset.sup'_le _ _ fun a _ =>
            le_of_abs_le (abs_actionValue_le mdp valid hR (fun s => hB ⟨s, rfl⟩) s a)⟩)
      (fun v w hv hw hbv hbw => bellmanOptimality_contraction mdp valid v w hv hw hbv hbw)
  exact ⟨vStar, hMeas, hBdd, hfix, fun w hw hbw hwfix =>
    bellmanOptimality_fixedPoint_unique mdp valid w vStar hwfix hfix hw hMeas hbw hBdd, hlim⟩

end Existence

end MarkovMDP

end Markov
end RL
end Proofs
