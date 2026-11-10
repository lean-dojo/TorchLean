/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Backup
public import TorchLean.NN.RL.MDP.Bellman
public import TorchLean.NN.RL.FiniteMDP
public import Mathlib.Topology.MetricSpace.Contracting
public import Mathlib.Topology.MetricSpace.Pseudo.Pi

/-!
# Finite-MDP Proofs

TorchLean's finite discounted MDPs have deterministic transitions. Bellman optimality dominates
every Bellman policy operator, and for `γ ≥ 0` both operators are monotone and are `γ`-contractions
in the finite sup metric. For `0 ≤ γ < 1` each has a unique fixed point, and value iteration
converges to it.

The contraction and fixed-point theorems are proved once, in `TorchLean.RL.MDP`, and derived here.
`toMarkov` reads a finite MDP at `ℝ` as a Markov MDP on `Fin n` whose transitions are Dirac
measures, and the contraction theorems follow from the Markov ones through
`MDP.valueSupDist_eq_dist`, `actionValue_toMarkov`, `vectorEquiv_bellmanPolicy` and
`vectorEquiv_bellmanMax`. The fixed-point theorems hold for any operator on finite value
tables that contracts the sup distance (`SupContraction`): read on `Fin n → ℝ`, such an operator is
a `MDP.BddContraction` (`SupContraction.toBdd`). `TorchLean.RL.StochasticMDP` reuses the same
development for stochastic transitions.

These proofs are by hand. Take one state and one action with reward 1, a self-loop and γ = 0.5.
The fixed point is 2, since $v = 1 + 0.5v$, and value iteration from the zero table reads 1, 1.5
and 1.75 after three steps. The error left after `k` steps is $2 \cdot 0.5^k$, the bound of
`SupContraction.iterate_error` with equality.

References:
- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (Wiley, 1994),
  Chapter 6, "Discounted Markov Decision Problems": https://doi.org/10.1002/9780470316887
- Bertsekas, *Dynamic Programming and Optimal Control*, Vol. II (4th ed., Athena Scientific,
  2012), Chapter 1, sections "Shorthand Notation and Monotonicity" and "The Role of Contraction
  Mappings": https://www.mit.edu/~dimitrib/dpbook.html
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  eqs. (3.14) and (3.19) for the Bellman equations and (4.10) for value iteration:
  http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace TorchLean
namespace RL

open FiniteMDP
open Filter Topology
open RL (discountedBackup_mono)
open TorchLean.Tensor (vectorEquiv)

variable {nStates nActions : Nat}

/-!
## Sup Metric for Finite Value Tables

The distance between two value tables is Mathlib's sup distance on `Fin n → ℝ`, the maximum
absolute pointwise difference, between the tables read through `Tensor.vectorEquiv`.

`RL.MDP.valueSupDist` measures the same thing on a possibly infinite state space. It is an
`sSup`, so its facts need a boundedness hypothesis. On a finite state space it is Mathlib's sup
distance (`MDP.valueSupDist_eq_dist`).
-/

/-!
## Fixed Points of Sup-Norm Contractions

On a finite state space every value function is bounded and measurable, so a `SupContraction`
read on `Fin n → ℝ` is a `MDP.BddContraction` (`toBdd`), and the fixed-point theorems below come
from there.
-/

/-- `T` contracts the sup distance of value tables, read on `Fin n → ℝ`, by the factor
`γ ∈ [0, 1)`. -/
structure SupContraction [Fact (0 < nStates)] (γ : ℝ)
    (T : ValueFunction ℝ nStates → ValueFunction ℝ nStates) : Prop where
  /-- The factor is nonnegative. -/
  nonneg : 0 ≤ γ
  /-- The factor is strictly below one. -/
  lt_one : γ < 1
  /-- One application of `T` shrinks every sup distance by `γ`. -/
  dist_le : ∀ values₁ values₂,
    dist (vectorEquiv nStates (T values₁)) (vectorEquiv nStates (T values₂)) ≤
      γ * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂)

namespace SupContraction

variable [Fact (0 < nStates)] {γ : ℝ} {T : ValueFunction ℝ nStates → ValueFunction ℝ nStates}

/-- The operator `T` read on `Fin n → ℝ` is a Mathlib `ContractingWith` map. -/
theorem contractingWith (h : SupContraction γ T) :
    ContractingWith γ.toNNReal ((vectorEquiv nStates).conj T) := by
  refine ⟨Real.toNNReal_lt_one.mpr h.lt_one, LipschitzWith.of_dist_le_mul fun f g => ?_⟩
  simpa [h.nonneg] using
    h.dist_le ((vectorEquiv nStates).symm f) ((vectorEquiv nStates).symm g)

omit [Fact (0 < nStates)] in
private theorem iterate_apply (k : Nat) (values : ValueFunction ℝ nStates) :
    ((vectorEquiv nStates).conj T)^[k] (vectorEquiv nStates values) =
      vectorEquiv nStates (T^[k] values) :=
  ((Function.Semiconj.iterate_right (fun v => by simp) k) values).symm

/-- `T` read on `Fin n → ℝ` is a `MDP.BddContraction`. -/
theorem toBdd (h : SupContraction γ T) :
    haveI : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
    MDP.BddContraction γ ((vectorEquiv nStates).conj T) := by
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  refine ⟨h.nonneg, h.lt_one, fun _ _ => MDP.isBddMeas_of_finite _, fun f g _ _ => ?_⟩
  have := h.dist_le ((vectorEquiv nStates).symm f) ((vectorEquiv nStates).symm g)
  rw [Equiv.apply_symm_apply, Equiv.apply_symm_apply] at this
  rwa [MDP.valueSupDist_eq_dist, MDP.valueSupDist_eq_dist, Equiv.conj_apply,
    Equiv.conj_apply]

/-- The fixed point of `T` given by Banach's fixed-point theorem, through `toBdd`. -/
noncomputable def fixedPoint (h : SupContraction γ T) : ValueFunction ℝ nStates :=
  haveI : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  (vectorEquiv nStates).symm h.toBdd.fixedPoint

/-- `fixedPoint` is a fixed point. -/
theorem map_fixedPoint (h : SupContraction γ T) : T h.fixedPoint = h.fixedPoint := by
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  have hp := h.toBdd.map_fixedPoint
  rw [Equiv.conj_apply] at hp
  exact (vectorEquiv nStates).eq_symm_apply.mpr hp

/-- Every fixed point of `T` is `fixedPoint`. -/
theorem eq_fixedPoint (h : SupContraction γ T) {values : ValueFunction ℝ nStates}
    (hfix : T values = values) : values = h.fixedPoint := by
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  exact (vectorEquiv nStates).eq_symm_apply.mpr
    (h.toBdd.eq_fixedPoint (MDP.isBddMeas_of_finite _) (by simp [hfix]))

/-- `T` has exactly one fixed point. -/
theorem existsUnique_fixedPoint (h : SupContraction γ T) : ∃! values, T values = values :=
  ⟨h.fixedPoint, h.map_fixedPoint, fun _ => h.eq_fixedPoint⟩

/-- `k` applications of `T` shrink every sup distance by `γ ^ k`. -/
theorem iterate_le (h : SupContraction γ T) (k : Nat) (values₁ values₂ : ValueFunction ℝ nStates) :
    dist (vectorEquiv nStates (T^[k] values₁)) (vectorEquiv nStates (T^[k] values₂)) ≤
      γ ^ k * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) := by
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  rw [← MDP.valueSupDist_eq_dist, ← MDP.valueSupDist_eq_dist, ← iterate_apply,
    ← iterate_apply]
  exact h.toBdd.iterate_le k (MDP.isBddMeas_of_finite _) (MDP.isBddMeas_of_finite _)

/-- The distance from the `k`-th iterate to any fixed point decays like `γ ^ k`. -/
theorem iterate_error (h : SupContraction γ T) {values vStar : ValueFunction ℝ nStates}
    (hfix : T vStar = vStar) (k : Nat) :
    dist (vectorEquiv nStates (T^[k] values)) (vectorEquiv nStates vStar) ≤
      γ ^ k * dist (vectorEquiv nStates values) (vectorEquiv nStates vStar) := by
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  rw [← MDP.valueSupDist_eq_dist, ← MDP.valueSupDist_eq_dist, ← iterate_apply]
  exact h.toBdd.iterate_error (MDP.isBddMeas_of_finite _) (MDP.isBddMeas_of_finite _)
    (by simp [hfix]) k

/-- A priori error bound for value iteration, computable from the first step alone. -/
theorem apriori_error (h : SupContraction γ T) (values : ValueFunction ℝ nStates) (k : Nat) :
    dist (vectorEquiv nStates (T^[k] values)) (vectorEquiv nStates h.fixedPoint) ≤
      dist (vectorEquiv nStates values) (vectorEquiv nStates (T values)) * γ ^ k / (1 - γ) := by
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  have := h.toBdd.apriori_error (MDP.isBddMeas_of_finite (vectorEquiv nStates values)) k
  rw [iterate_apply, Equiv.conj_apply, Equiv.symm_apply_apply] at this
  rwa [← MDP.valueSupDist_eq_dist, ← MDP.valueSupDist_eq_dist, fixedPoint,
    Equiv.apply_symm_apply]

/-- Value iteration converges to the fixed point from any starting table. -/
theorem tendsto_iterate (h : SupContraction γ T) (values : ValueFunction ℝ nStates) :
    Tendsto (fun k => dist (vectorEquiv nStates (T^[k] values)) (vectorEquiv nStates h.fixedPoint))
      atTop (𝓝 0) := by
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  simpa [MDP.valueSupDist_eq_dist, iterate_apply, fixedPoint] using
    h.toBdd.tendsto_iterate (MDP.isBddMeas_of_finite (vectorEquiv nStates values))

end SupContraction

/-!
## Deterministic Bellman Operators
-/

/-- Policy Bellman operators read back exactly the selected state-action value. -/
theorem FiniteMDP.getScalar_bellmanPolicy
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    Tensor.getScalar (bellmanPolicy mdp policy values) state =
      actionValue mdp values state (policy state) := by
  simp [bellmanPolicy]

/-- Bellman optimality reads back the best state-action value. -/
theorem FiniteMDP.getScalar_bellmanMax
    [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
    Tensor.getScalar (bellmanMax mdp values) state =
      (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty
        (actionValue mdp values state) := by
  simp [bellmanMax]

/-- A Bellman state-action value is monotone in the candidate value function when `γ ≥ 0`. -/
theorem FiniteMDP.actionValue_mono
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ : 0 ≤ mdp.discount)
    (hValues : ∀ state, Tensor.getScalar values₁ state ≤ Tensor.getScalar values₂ state)
    (state : Fin nStates)
    (action : Fin nActions) :
    actionValue mdp values₁ state action ≤
      actionValue mdp values₂ state action :=
  discountedBackup_mono _ hγ (hValues _)

/-- Bellman policy operators are pointwise monotone for nonnegative discounts. -/
theorem FiniteMDP.bellmanPolicy_mono
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ : 0 ≤ mdp.discount)
    (hValues : ∀ state, Tensor.getScalar values₁ state ≤ Tensor.getScalar values₂ state)
    (state : Fin nStates) :
    Tensor.getScalar (bellmanPolicy mdp policy values₁) state ≤
      Tensor.getScalar (bellmanPolicy mdp policy values₂) state := by
  simpa [getScalar_bellmanPolicy] using
    actionValue_mono mdp values₁ values₂ hγ hValues state (policy state)

/-- Bellman optimality dominates the state-action value of every action. -/
theorem FiniteMDP.actionValue_le_bellmanMax
    [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) :
    actionValue mdp values state action ≤
      Tensor.getScalar (bellmanMax mdp values) state := by
  rw [getScalar_bellmanMax]
  exact Finset.le_sup' (actionValue mdp values state) (Finset.mem_univ action)

/-- Bellman optimality dominates Bellman evaluation under any deterministic policy. -/
theorem FiniteMDP.bellmanPolicy_le_bellmanMax
    [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    Tensor.getScalar (bellmanPolicy mdp policy values) state ≤
      Tensor.getScalar (bellmanMax mdp values) state := by
  simpa [getScalar_bellmanPolicy] using
    actionValue_le_bellmanMax mdp values state (policy state)

/-- Bellman optimality is pointwise monotone for nonnegative discounts. -/
theorem FiniteMDP.bellmanMax_mono
    [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ : 0 ≤ mdp.discount)
    (hValues : ∀ state, Tensor.getScalar values₁ state ≤ Tensor.getScalar values₂ state)
    (state : Fin nStates) :
    Tensor.getScalar (bellmanMax mdp values₁) state ≤
      Tensor.getScalar (bellmanMax mdp values₂) state := by
  simp only [getScalar_bellmanMax]
  exact Finset.sup'_mono_fun fun action _ =>
    actionValue_mono mdp values₁ values₂ hγ hValues state action

/-!
## Deterministic MDPs as Markov MDPs

The Bellman operators of `toMarkov` are those of the finite MDP, read on `Fin n → ℝ`, so the
contraction theorems below are instances of the Markov ones.
-/

section MarkovTransfer

open MeasureTheory ProbabilityTheory

/-- `mdp` as a Markov MDP on `Fin nStates`: the transition from `(state, action)` is the Dirac
measure at `(mdp.step state action).state`, and the reward, termination flag and discount are
those of `mdp`. -/
noncomputable def FiniteMDP.toMarkov (mdp : FiniteMDP ℝ nStates nActions) :
    RL.MDP (Fin nStates) (Fin nActions) where
  initialState := mdp.initialState
  transition := Kernel.deterministic (fun sa => (mdp.step sa.1 sa.2).state) (measurable_of_finite _)
  reward state action := (mdp.step state action).reward
  terminated state action := (mdp.step state action).terminated
  discount := mdp.discount

/-- The transition of `toMarkov mdp` from `(state, action)` is the Dirac measure at the next
state. -/
theorem FiniteMDP.toMarkov_transition (mdp : FiniteMDP ℝ nStates nActions) (state : Fin nStates)
    (action : Fin nActions) :
    (toMarkov mdp).transition (state, action) = Measure.dirac (mdp.step state action).state :=
  rfl

/-- The transition of `toMarkov mdp` is a Markov kernel. -/
instance FiniteMDP.isMarkovKernel_toMarkov (mdp : FiniteMDP ℝ nStates nActions) :
    IsMarkovKernel (toMarkov mdp).transition :=
  Kernel.isMarkovKernel_deterministic (measurable_of_finite _)

/-- For `0 ≤ γ < 1`, `toMarkov mdp` is a valid Markov MDP. -/
theorem FiniteMDP.toMarkov_valid (mdp : FiniteMDP ℝ nStates nActions) (hγ₀ : 0 ≤ mdp.discount)
    (hγ₁ : mdp.discount < 1) : MDP.Valid (toMarkov mdp) :=
  ⟨inferInstance, measurable_of_finite _, measurable_of_finite _, hγ₀, hγ₁⟩

/-- The expected next value of `toMarkov mdp` is the value of the next state. -/
theorem FiniteMDP.expectedNextValue_toMarkov (mdp : FiniteMDP ℝ nStates nActions)
    (values : ValueFunction ℝ nStates) (state : Fin nStates) (action : Fin nActions) :
    MDP.expectedNextValue (toMarkov mdp) (vectorEquiv nStates values) state action =
      Tensor.getScalar values (mdp.step state action).state := by
  rw [MDP.expectedNextValue, MDP.transitionMeasure, toMarkov_transition, integral_dirac]
  rfl

/-- The action values of `toMarkov mdp` are the state-action values of `mdp`. -/
theorem FiniteMDP.actionValue_toMarkov (mdp : FiniteMDP ℝ nStates nActions)
    (values : ValueFunction ℝ nStates) (state : Fin nStates) (action : Fin nActions) :
    MDP.actionValue (toMarkov mdp) (vectorEquiv nStates values) state action =
      actionValue mdp values state action := by
  rw [MDP.actionValue, expectedNextValue_toMarkov]
  rfl

/-- Policy evaluation of `mdp`, read on `Fin n → ℝ`, is policy evaluation of `toMarkov mdp`. -/
theorem FiniteMDP.vectorEquiv_bellmanPolicy (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions) (values : ValueFunction ℝ nStates) :
    vectorEquiv nStates (bellmanPolicy mdp policy values) =
      MDP.bellmanPolicy (toMarkov mdp) policy (vectorEquiv nStates values) := by
  funext state
  exact (getScalar_bellmanPolicy mdp policy values state).trans
    (actionValue_toMarkov mdp values state (policy state)).symm

/-- Bellman optimality of `mdp`, read on `Fin n → ℝ`, is Bellman optimality of `toMarkov mdp`. -/
theorem FiniteMDP.vectorEquiv_bellmanMax [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions) (values : ValueFunction ℝ nStates) :
    haveI : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
    vectorEquiv nStates (bellmanMax mdp values) =
      MDP.bellmanMax (toMarkov mdp) (vectorEquiv nStates values) := by
  funext state
  change Tensor.getScalar (bellmanMax mdp values) state = _
  rw [getScalar_bellmanMax]
  simp only [MDP.bellmanMax, actionValue_toMarkov]

end MarkovTransfer

/-- For `γ ≥ 0`, deterministic state-action Bellman values are Lipschitz with constant `γ`. -/
theorem FiniteMDP.actionValue_abs_sub_le
    [Fact (0 < nStates)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ₀ : 0 ≤ mdp.discount)
    (state : Fin nStates)
    (action : Fin nActions) :
    |actionValue mdp values₁ state action - actionValue mdp values₂ state action|
      ≤ mdp.discount * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) := by
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  rw [← MDP.valueSupDist_eq_dist, ← actionValue_toMarkov, ← actionValue_toMarkov]
  exact MDP.actionValue_abs_sub_le (toMarkov mdp) hγ₀ _ _ (measurable_of_finite _)
    (measurable_of_finite _) (Set.finite_range _).bddAbove (Set.finite_range _).bddAbove state
    action

/-- For `γ ≥ 0`, Bellman evaluation for a deterministic policy is a `γ`-contraction in the sup
metric of `Fin n → ℝ`, the tables read through `Tensor.vectorEquiv`. -/
theorem FiniteMDP.bellmanPolicy_contraction
    [Fact (0 < nStates)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ₀ : 0 ≤ mdp.discount) :
    dist (vectorEquiv nStates (bellmanPolicy mdp policy values₁))
        (vectorEquiv nStates (bellmanPolicy mdp policy values₂))
      ≤ mdp.discount * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) := by
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  rw [← MDP.valueSupDist_eq_dist, ← MDP.valueSupDist_eq_dist, vectorEquiv_bellmanPolicy,
    vectorEquiv_bellmanPolicy]
  exact MDP.bellmanPolicy_contraction (toMarkov mdp) hγ₀ policy _ _ (measurable_of_finite _)
    (measurable_of_finite _) (Set.finite_range _).bddAbove (Set.finite_range _).bddAbove

/-- For `γ ≥ 0` and at a fixed state, Bellman optimality is Lipschitz with constant `γ`. -/
theorem FiniteMDP.bellmanMax_abs_sub_le
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ₀ : 0 ≤ mdp.discount)
    (state : Fin nStates) :
    |Tensor.getScalar (bellmanMax mdp values₁) state -
        Tensor.getScalar (bellmanMax mdp values₂) state|
      ≤ mdp.discount * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) := by
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  have : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  have h := MDP.bellmanMax_abs_sub_le (toMarkov mdp) hγ₀ (vectorEquiv nStates values₁)
    (vectorEquiv nStates values₂) (measurable_of_finite _) (measurable_of_finite _)
    (Set.finite_range _).bddAbove (Set.finite_range _).bddAbove state
  rwa [← vectorEquiv_bellmanMax, ← vectorEquiv_bellmanMax,
    MDP.valueSupDist_eq_dist] at h

/-- For `γ ≥ 0`, Bellman optimality is a `γ`-contraction in the sup metric of `Fin n → ℝ`, the
tables read through `Tensor.vectorEquiv`. -/
theorem FiniteMDP.bellmanMax_contraction
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hγ₀ : 0 ≤ mdp.discount) :
    dist (vectorEquiv nStates (bellmanMax mdp values₁))
        (vectorEquiv nStates (bellmanMax mdp values₂))
      ≤ mdp.discount * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) := by
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  have : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  rw [← MDP.valueSupDist_eq_dist, ← MDP.valueSupDist_eq_dist, vectorEquiv_bellmanMax,
    vectorEquiv_bellmanMax]
  exact MDP.bellmanMax_contraction (toMarkov mdp) hγ₀ _ _ (measurable_of_finite _)
    (measurable_of_finite _) (Set.finite_range _).bddAbove (Set.finite_range _).bddAbove

/-- For `0 ≤ γ < 1`, policy evaluation is a `SupContraction`: it has a unique fixed point and
value iteration converges to it. -/
theorem FiniteMDP.bellmanPolicy_supContraction
    [Fact (0 < nStates)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (policy : Policy nStates nActions)
    (hγ₀ : 0 ≤ mdp.discount) (hγ₁ : mdp.discount < 1) :
    SupContraction mdp.discount (bellmanPolicy mdp policy) :=
  ⟨hγ₀, hγ₁, fun values₁ values₂ => bellmanPolicy_contraction mdp policy values₁ values₂ hγ₀⟩

/-- For `0 ≤ γ < 1`, Bellman optimality is a `SupContraction`: the optimal value table exists, is
unique, and value iteration converges to it. -/
theorem FiniteMDP.bellmanMax_supContraction
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : FiniteMDP ℝ nStates nActions)
    (hγ₀ : 0 ≤ mdp.discount) (hγ₁ : mdp.discount < 1) :
    SupContraction mdp.discount (bellmanMax mdp) :=
  ⟨hγ₀, hγ₁, fun values₁ values₂ => bellmanMax_contraction mdp values₁ values₂ hγ₀⟩

end RL
end TorchLean
