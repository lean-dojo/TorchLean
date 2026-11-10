/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.FiniteMDP.Bellman
public import TorchLean.NN.RL.StochasticMDP

/-!
# Finite Stochastic MDP Proofs

For a valid finite stochastic MDP (nonnegative transition rows that sum to `1`, discount in
`[0, 1)`), the Bellman expectation and optimality operators are monotone and are
`γ`-contractions in the sup metric. Each has a unique fixed point, and value iteration converges
to it geometrically.

The contraction bounds are those of `TorchLean.RL.MDP`: `toMarkov` reads a finite stochastic MDP
as a Markov MDP on `Fin n` whose transition from `(s, a)` is the sum of the Dirac measures at the
next states, weighted by the row `transitionProb s a`. The fixed-point theorems come from
`TorchLean.RL.SupContraction`, which derives them from `RL.MDP.BddContraction`.

We proved these by hand. Monotonicity is a plain sum argument that needs every row entry
nonnegative: with the row `(-1, 2)`, raising the first state's value by 1 lowers the expected next
value by 1. The contraction bounds also need each row to sum to 1, because they go through the
Markov kernel of `isMarkovKernel_toMarkov`.

References:
- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (Wiley, 1994),
  Chapter 6, "Discounted Markov Decision Problems": https://doi.org/10.1002/9780470316887
- Bertsekas, *Dynamic Programming and Optimal Control*, Vol. II (4th ed., Athena Scientific,
  2012), Chapter 1, section "The Role of Contraction Mappings":
  https://www.mit.edu/~dimitrib/dpbook.html
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  eqs. (3.14) and (3.19) for the Bellman equations, (4.5) for iterative policy evaluation and
  (4.10) for value iteration: http://incompleteideas.net/book/the-book-2nd.html
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace StochasticMDP


variable {nStates nActions : Nat}

/-!
Distances between value tables are Mathlib's sup distance on `Fin n → ℝ`, through
`Tensor.vectorEquiv`. The `open` is fully qualified because `RL.StochasticMDP` on its own is also
the name of the transition structure used below.
-/
open RL (SupContraction)
open RL (discountedBackup_mono)
open TorchLean.Tensor (vectorEquiv)
open Filter Topology

/-- Policy evaluation reads back the selected action value. -/
theorem getScalar_bellmanPolicy
    (mdp : StochasticMDP nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    Tensor.getScalar (bellmanPolicy mdp policy values) state =
      actionValue mdp values state (policy state) := by
  simp [RL.StochasticMDP.bellmanPolicy]

/-- Bellman optimality reads back the best action value. -/
theorem getScalar_bellmanMax
    [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    let _ : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
    Tensor.getScalar (bellmanMax mdp values) state =
      (Finset.univ : Finset (Fin nActions)).sup' Finset.univ_nonempty
        (actionValue mdp values state) := by
  simp [RL.StochasticMDP.bellmanMax]

/-- Expected next-state value is monotone in the candidate value function. -/
theorem expectedNextValue_mono
    (mdp : StochasticMDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, Tensor.getScalar values₁ state ≤ Tensor.getScalar values₂ state)
    (state : Fin nStates)
    (action : Fin nActions) :
    expectedNextValue mdp values₁ state action ≤ expectedNextValue mdp values₂ state action :=
  Finset.sum_le_sum fun nextState _ =>
    mul_le_mul_of_nonneg_left (hValues nextState) (valid.transition_nonneg state action nextState)

/-- Bellman state-action values are monotone in the candidate value function. -/
theorem actionValue_mono
    (mdp : StochasticMDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, Tensor.getScalar values₁ state ≤ Tensor.getScalar values₂ state)
    (state : Fin nStates)
    (action : Fin nActions) :
    actionValue mdp values₁ state action ≤ actionValue mdp values₂ state action :=
  discountedBackup_mono _ valid.discount_nonneg
    (expectedNextValue_mono mdp valid values₁ values₂ hValues state action)

/-- Bellman expectation operators are pointwise monotone. -/
theorem bellmanPolicy_mono
    (mdp : StochasticMDP nStates nActions)
    (valid : Valid mdp)
    (policy : Policy nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, Tensor.getScalar values₁ state ≤ Tensor.getScalar values₂ state)
    (state : Fin nStates) :
    Tensor.getScalar (bellmanPolicy mdp policy values₁) state ≤
      Tensor.getScalar (bellmanPolicy mdp policy values₂) state := by
  simpa [getScalar_bellmanPolicy] using
    actionValue_mono mdp valid values₁ values₂ hValues state (policy state)

/-- Optimal Bellman operators are pointwise monotone. -/
theorem bellmanMax_mono
    [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (hValues : ∀ state, Tensor.getScalar values₁ state ≤ Tensor.getScalar values₂ state)
    (state : Fin nStates) :
    Tensor.getScalar (bellmanMax mdp values₁) state ≤
      Tensor.getScalar (bellmanMax mdp values₂) state := by
  simp only [getScalar_bellmanMax]
  exact Finset.sup'_mono_fun fun action _ =>
    actionValue_mono mdp valid values₁ values₂ hValues state action

/-!
## Finite Stochastic MDPs as Markov MDPs

For a valid MDP the transition of `toMarkov` is a Markov kernel and its Bellman operators are those
of the finite MDP, read on `Fin n → ℝ`, so the contraction theorems below are instances of the
Markov ones.
-/

section MarkovTransfer

open MeasureTheory ProbabilityTheory

/-- `mdp` as a Markov MDP on `Fin nStates`: the transition from `(state, action)` is the sum over
`nextState` of the Dirac measure at `nextState` weighted by the transition probability (clipped
below at `0`), and the reward, termination flag and discount are those of `mdp`. -/
noncomputable def toMarkov (mdp : StochasticMDP nStates nActions) :
    RL.MDP (Fin nStates) (Fin nActions) where
  initialState := mdp.initialState
  transition := Kernel.ofFunOfCountable fun sa =>
    ∑ nextState, ((mdp.transitionProb sa.1 sa.2).getScalar nextState).toNNReal •
      Measure.dirac nextState
  reward := mdp.reward
  terminated := mdp.terminated
  discount := mdp.discount

/-- The transition of `toMarkov mdp` from `(state, action)` as a weighted sum of Dirac
measures. -/
theorem toMarkov_transition (mdp : StochasticMDP nStates nActions) (state : Fin nStates)
    (action : Fin nActions) :
    (toMarkov mdp).transition (state, action) =
      ∑ nextState, ((mdp.transitionProb state action).getScalar nextState).toNNReal •
        Measure.dirac nextState :=
  rfl

/-- For a valid `mdp`, the transition of `toMarkov mdp` is a Markov kernel. -/
theorem isMarkovKernel_toMarkov (mdp : StochasticMDP nStates nActions) (valid : Valid mdp) :
    IsMarkovKernel (toMarkov mdp).transition := by
  refine ⟨fun ⟨state, action⟩ => ⟨?_⟩⟩
  rw [toMarkov_transition, Measure.finsetSum_apply]
  simp only [Measure.coe_nnreal_smul_apply, measure_univ, mul_one]
  rw [← ENNReal.ofNNReal_finsetSum, ← Real.toNNReal_sum_of_nonneg fun nextState _ =>
    valid.transition_nonneg state action nextState, valid.transition_sums_to_one state action]
  simp

/-- For a valid `mdp`, `toMarkov mdp` is a valid Markov MDP. -/
theorem toMarkov_valid (mdp : StochasticMDP nStates nActions) (valid : Valid mdp) :
    MDP.Valid (toMarkov mdp) :=
  ⟨isMarkovKernel_toMarkov mdp valid, measurable_of_finite _, measurable_of_finite _,
    valid.discount_nonneg, valid.discount_lt_one⟩

/-- For a valid `mdp`, the expected next value of `toMarkov mdp` is that of `mdp`. -/
theorem expectedNextValue_toMarkov (mdp : StochasticMDP nStates nActions) (valid : Valid mdp)
    (values : ValueFunction ℝ nStates) (state : Fin nStates) (action : Fin nActions) :
    MDP.expectedNextValue (toMarkov mdp) (vectorEquiv nStates values) state action =
      expectedNextValue mdp values state action := by
  rw [MDP.expectedNextValue, MDP.transitionMeasure, toMarkov_transition,
    integral_finsetSum_measure fun _ _ => Integrable.of_finite]
  simp only [integral_smul_nnreal_measure, integral_dirac]
  refine Finset.sum_congr rfl fun nextState _ => ?_
  rw [NNReal.smul_def, smul_eq_mul,
    Real.coe_toNNReal _ (valid.transition_nonneg state action nextState)]
  rfl

/-- For a valid `mdp`, the action values of `toMarkov mdp` are those of `mdp`. -/
theorem actionValue_toMarkov (mdp : StochasticMDP nStates nActions) (valid : Valid mdp)
    (values : ValueFunction ℝ nStates) (state : Fin nStates) (action : Fin nActions) :
    MDP.actionValue (toMarkov mdp) (vectorEquiv nStates values) state action =
      actionValue mdp values state action := by
  rw [MDP.actionValue, expectedNextValue_toMarkov mdp valid]
  rfl

/-- For a valid `mdp`, policy evaluation read on `Fin n → ℝ` is policy evaluation of
`toMarkov mdp`. -/
theorem vectorEquiv_bellmanPolicy (mdp : StochasticMDP nStates nActions) (valid : Valid mdp)
    (policy : Policy nStates nActions) (values : ValueFunction ℝ nStates) :
    vectorEquiv nStates (bellmanPolicy mdp policy values) =
      MDP.bellmanPolicy (toMarkov mdp) policy (vectorEquiv nStates values) := by
  funext state
  exact (getScalar_bellmanPolicy mdp policy values state).trans
    (actionValue_toMarkov mdp valid values state (policy state)).symm

/-- For a valid `mdp`, Bellman optimality read on `Fin n → ℝ` is Bellman optimality of
`toMarkov mdp`. -/
theorem vectorEquiv_bellmanMax [Fact (0 < nActions)] (mdp : StochasticMDP nStates nActions)
    (valid : Valid mdp) (values : ValueFunction ℝ nStates) :
    haveI : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
    vectorEquiv nStates (bellmanMax mdp values) =
      MDP.bellmanMax (toMarkov mdp) (vectorEquiv nStates values) := by
  funext state
  change Tensor.getScalar (bellmanMax mdp values) state = _
  rw [getScalar_bellmanMax]
  simp only [MDP.bellmanMax, actionValue_toMarkov mdp valid]

end MarkovTransfer

/-- For a valid `mdp`, at each state and action the expected next values of two tables differ by at
most their sup distance. -/
theorem expectedNextValue_abs_sub_le
    [Fact (0 < nStates)]
    (mdp : StochasticMDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) :
    |expectedNextValue mdp values₁ state action - expectedNextValue mdp values₂ state action|
      ≤ dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) := by
  have := isMarkovKernel_toMarkov mdp valid
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  rw [← MDP.valueSupDist_eq_dist, ← expectedNextValue_toMarkov mdp valid,
    ← expectedNextValue_toMarkov mdp valid]
  exact MDP.expectedNextValue_abs_sub_le (toMarkov mdp) _ _ (measurable_of_finite _)
    (measurable_of_finite _) (Set.finite_range _).bddAbove (Set.finite_range _).bddAbove state
    action

/-- State-action Bellman values are Lipschitz with constant `γ` in the sup metric. -/
theorem actionValue_abs_sub_le
    [Fact (0 < nStates)]
    (mdp : StochasticMDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) :
    |actionValue mdp values₁ state action - actionValue mdp values₂ state action|
      ≤ mdp.discount * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) := by
  have := isMarkovKernel_toMarkov mdp valid
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  rw [← MDP.valueSupDist_eq_dist, ← actionValue_toMarkov mdp valid,
    ← actionValue_toMarkov mdp valid]
  exact MDP.actionValue_abs_sub_le (toMarkov mdp) valid.discount_nonneg _ _
    (measurable_of_finite _) (measurable_of_finite _) (Set.finite_range _).bddAbove
    (Set.finite_range _).bddAbove state action

/-- Bellman expectation is a contraction with modulus `γ` in the sup metric of `Fin n → ℝ`:

`dist (T^π values₁) (T^π values₂) ≤ γ * dist values₁ values₂`, the tables read through
`Tensor.vectorEquiv`. -/
theorem bellmanPolicy_contraction
    [Fact (0 < nStates)]
    (mdp : StochasticMDP nStates nActions)
    (valid : Valid mdp)
    (policy : Policy nStates nActions)
    (values₁ values₂ : ValueFunction ℝ nStates) :
    dist (vectorEquiv nStates (bellmanPolicy mdp policy values₁))
        (vectorEquiv nStates (bellmanPolicy mdp policy values₂))
      ≤ mdp.discount * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) := by
  have := isMarkovKernel_toMarkov mdp valid
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  rw [← MDP.valueSupDist_eq_dist, ← MDP.valueSupDist_eq_dist,
    vectorEquiv_bellmanPolicy mdp valid, vectorEquiv_bellmanPolicy mdp valid]
  exact MDP.bellmanPolicy_contraction (toMarkov mdp) valid.discount_nonneg policy _ _
    (measurable_of_finite _) (measurable_of_finite _) (Set.finite_range _).bddAbove
    (Set.finite_range _).bddAbove

/-- Every action value is at most the Bellman optimality value at its state. -/
theorem actionValue_le_bellmanMax
    [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates)
    (action : Fin nActions) :
    actionValue mdp values state action ≤ Tensor.getScalar (bellmanMax mdp values) state := by
  rw [getScalar_bellmanMax]
  exact Finset.le_sup' (actionValue mdp values state) (Finset.mem_univ action)

/-- Bellman optimality dominates Bellman evaluation under any deterministic policy. -/
theorem bellmanPolicy_le_bellmanMax
    [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions)
    (policy : Policy nStates nActions)
    (values : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    Tensor.getScalar (bellmanPolicy mdp policy values) state ≤
      Tensor.getScalar (bellmanMax mdp values) state := by
  simpa [getScalar_bellmanPolicy] using
    actionValue_le_bellmanMax mdp values state (policy state)

/-- At a fixed state, Bellman optimality is a contraction with modulus `γ`. -/
theorem bellmanMax_abs_sub_le
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates)
    (state : Fin nStates) :
    |Tensor.getScalar (bellmanMax mdp values₁) state -
        Tensor.getScalar (bellmanMax mdp values₂) state|
      ≤ mdp.discount * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) := by
  have := isMarkovKernel_toMarkov mdp valid
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  have : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  have h := MDP.bellmanMax_abs_sub_le (toMarkov mdp) valid.discount_nonneg
    (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) (measurable_of_finite _)
    (measurable_of_finite _) (Set.finite_range _).bddAbove (Set.finite_range _).bddAbove state
  rwa [← vectorEquiv_bellmanMax mdp valid, ← vectorEquiv_bellmanMax mdp valid,
    MDP.valueSupDist_eq_dist] at h

/-- Bellman optimality is a contraction with modulus `γ` in the sup metric of `Fin n → ℝ`:

`dist (T* values₁) (T* values₂) ≤ γ * dist values₁ values₂`, the tables read through
`Tensor.vectorEquiv`. -/
theorem bellmanMax_contraction
    [Fact (0 < nStates)] [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions)
    (valid : Valid mdp)
    (values₁ values₂ : ValueFunction ℝ nStates) :
    dist (vectorEquiv nStates (bellmanMax mdp values₁))
        (vectorEquiv nStates (bellmanMax mdp values₂))
      ≤ mdp.discount * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) := by
  have := isMarkovKernel_toMarkov mdp valid
  have : Nonempty (Fin nStates) := ⟨⟨0, Fact.out⟩⟩
  have : Nonempty (Fin nActions) := ⟨⟨0, Fact.out⟩⟩
  rw [← MDP.valueSupDist_eq_dist, ← MDP.valueSupDist_eq_dist,
    vectorEquiv_bellmanMax mdp valid, vectorEquiv_bellmanMax mdp valid]
  exact MDP.bellmanMax_contraction (toMarkov mdp) valid.discount_nonneg _ _
    (measurable_of_finite _) (measurable_of_finite _) (Set.finite_range _).bddAbove
    (Set.finite_range _).bddAbove

/-!
## Fixed Points and Value Iteration

`Valid` puts the discount in `[0, 1)`, so both operators are `SupContraction`s. The fixed point
exists by Banach's fixed-point theorem, is unique, and value iteration converges to it from any
start. The theorems below name the consequences most often cited.
-/

section FixedPoints

variable [Fact (0 < nStates)]

/-- Policy evaluation is a `SupContraction`. -/
theorem bellmanPolicy_supContraction
    (mdp : StochasticMDP nStates nActions) (valid : Valid mdp) (policy : Policy nStates nActions) :
    SupContraction mdp.discount (bellmanPolicy mdp policy) :=
  ⟨valid.discount_nonneg, valid.discount_lt_one, bellmanPolicy_contraction mdp valid policy⟩

/-- Bellman optimality is a `SupContraction`. -/
theorem bellmanMax_supContraction [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions) (valid : Valid mdp) :
    SupContraction mdp.discount (bellmanMax mdp) :=
  ⟨valid.discount_nonneg, valid.discount_lt_one, bellmanMax_contraction mdp valid⟩

/-- The value of a policy exists and is unique: `T^π` has exactly one fixed point. -/
theorem bellmanPolicy_existsUnique_fixedPoint
    (mdp : StochasticMDP nStates nActions) (valid : Valid mdp) (policy : Policy nStates nActions) :
    ∃! values : ValueFunction ℝ nStates, bellmanPolicy mdp policy values = values :=
  (bellmanPolicy_supContraction mdp valid policy).existsUnique_fixedPoint

/-- The optimal value function exists and is unique: `T*` has exactly one fixed point. -/
theorem bellmanMax_existsUnique_fixedPoint [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions) (valid : Valid mdp) :
    ∃! values : ValueFunction ℝ nStates, bellmanMax mdp values = values :=
  (bellmanMax_supContraction mdp valid).existsUnique_fixedPoint

/-- `bellmanPolicy` iterates are geometric contractions in the sup metric. -/
theorem bellmanPolicy_iterate_le
    (mdp : StochasticMDP nStates nActions) (valid : Valid mdp) (policy : Policy nStates nActions)
    (k : Nat) (values₁ values₂ : ValueFunction ℝ nStates) :
    dist (vectorEquiv nStates ((bellmanPolicy mdp policy)^[k] values₁))
        (vectorEquiv nStates ((bellmanPolicy mdp policy)^[k] values₂))
      ≤ mdp.discount ^ k * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) :=
  (bellmanPolicy_supContraction mdp valid policy).iterate_le k values₁ values₂

/-- A fixed point of the Bellman policy operator is unique. -/
theorem bellmanPolicy_fixedPoint_unique
    (mdp : StochasticMDP nStates nActions) (valid : Valid mdp) (policy : Policy nStates nActions)
    (v w : ValueFunction ℝ nStates)
    (hv : bellmanPolicy mdp policy v = v) (hw : bellmanPolicy mdp policy w = w) :
    v = w :=
  let h := bellmanPolicy_supContraction mdp valid policy
  (h.eq_fixedPoint hv).trans (h.eq_fixedPoint hw).symm

/-- After `k` iterations of the Bellman policy operator, the sup distance to a fixed point is at
most `γ ^ k` times the starting distance. -/
theorem bellmanPolicy_iterate_error
    (mdp : StochasticMDP nStates nActions) (valid : Valid mdp) (policy : Policy nStates nActions)
    (v vStar : ValueFunction ℝ nStates) (hvStar : bellmanPolicy mdp policy vStar = vStar)
    (k : Nat) :
    dist (vectorEquiv nStates ((bellmanPolicy mdp policy)^[k] v)) (vectorEquiv nStates vStar)
      ≤ mdp.discount ^ k * dist (vectorEquiv nStates v) (vectorEquiv nStates vStar) :=
  (bellmanPolicy_supContraction mdp valid policy).iterate_error hvStar k

/-- `bellmanMax` iterates are geometric contractions in the sup metric. -/
theorem bellmanMax_iterate_le [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions) (valid : Valid mdp)
    (k : Nat) (values₁ values₂ : ValueFunction ℝ nStates) :
    dist (vectorEquiv nStates ((bellmanMax mdp)^[k] values₁))
        (vectorEquiv nStates ((bellmanMax mdp)^[k] values₂))
      ≤ mdp.discount ^ k * dist (vectorEquiv nStates values₁) (vectorEquiv nStates values₂) :=
  (bellmanMax_supContraction mdp valid).iterate_le k values₁ values₂

/-- A fixed point of the Bellman optimality operator is unique. -/
theorem bellmanMax_fixedPoint_unique [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions) (valid : Valid mdp)
    (v w : ValueFunction ℝ nStates)
    (hv : bellmanMax mdp v = v) (hw : bellmanMax mdp w = w) :
    v = w :=
  let h := bellmanMax_supContraction mdp valid
  (h.eq_fixedPoint hv).trans (h.eq_fixedPoint hw).symm

/-- After `k` iterations of Bellman optimality, the sup distance to a fixed point is at most
`γ ^ k` times the starting distance. -/
theorem bellmanMax_iterate_error [Fact (0 < nActions)]
    (mdp : StochasticMDP nStates nActions) (valid : Valid mdp)
    (v vStar : ValueFunction ℝ nStates) (hvStar : bellmanMax mdp vStar = vStar)
    (k : Nat) :
    dist (vectorEquiv nStates ((bellmanMax mdp)^[k] v)) (vectorEquiv nStates vStar)
      ≤ mdp.discount ^ k * dist (vectorEquiv nStates v) (vectorEquiv nStates vStar) :=
  (bellmanMax_supContraction mdp valid).iterate_error hvStar k

end FixedPoints

end StochasticMDP
end RL
end TorchLean
