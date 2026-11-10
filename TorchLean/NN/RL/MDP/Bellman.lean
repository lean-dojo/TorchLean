/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Backup
public import TorchLean.NN.FinsetSup
public import TorchLean.NN.RL.MDP
public import Mathlib.Analysis.Normed.Lp.lpSpace
public import Mathlib.MeasureTheory.Constructions.BorelSpace.Metrizable
public import Mathlib.MeasureTheory.Order.Lattice
public import Mathlib.Probability.Kernel.MeasurableIntegral
public import Mathlib.Topology.MetricSpace.Contracting

/-!
# Markov-Kernel MDP Proofs (Measure Theory)

The theorems here are the discounted Bellman facts for TorchLean's measure-theoretic MDP layer
(`TorchLean.NN.RL.MDP`), which is built on Mathlib's Markov kernels. They follow the
standard argument of dynamic programming:

- if two bounded measurable value functions are uniformly close, so are their Bellman backups,
- in particular, the Bellman expectation operator for a fixed deterministic policy is a
  `γ`-contraction in the sup metric on bounded measurable value functions,
- for finite action spaces, Bellman optimality is also a `γ`-contraction in the same metric,
- with bounded rewards, both operators have exactly one bounded measurable fixed point (policy
  evaluation also needs a measurable policy), and value iteration for Bellman optimality
  converges to it.

The one-step bounds need only that the transition is a Markov kernel and that the discount is
nonnegative. The fixed-point theory is proved once, by Banach's fixed-point theorem, for any
operator that keeps bounded measurable value functions bounded and measurable and contracts their
sup distance by a factor in `[0, 1)` (`BddContraction`). `TorchLean.RL.FiniteMDP` and
`TorchLean.RL.StochasticMDP` read their finite MDPs as Markov MDPs (`toMarkov`) and derive their
contraction bounds from the ones here. Their fixed-point theorems come from `BddContraction`
through `TorchLean.RL.SupContraction.toBdd`.

We proved all of it by hand. Banach's theorem is mathlib's `ContractingWith.fixedPoint`, and
`apriori_error` reads back `ContractingWith.apriori_dist_iterate_fixedPoint_le`. At γ = 0.9 that
bound puts a start whose first step moves it by 1 in sup distance within $10 \cdot 0.9^k$ of the
fixed point after `k` steps, under 0.01 from `k = 66` on.

References:

- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming* (Wiley, 1994),
  Chapter 6, "Discounted Markov Decision Problems": https://doi.org/10.1002/9780470316887
- Bertsekas, *Dynamic Programming and Optimal Control*, Vol. II (4th ed., Athena Scientific,
  2012), Chapter 1, section "The Role of Contraction Mappings":
  https://www.mit.edu/~dimitrib/dpbook.html
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  eqs. (3.14) and (3.19) for the Bellman equations, (4.5) for iterative policy evaluation and
  (4.10) for value iteration: http://incompleteideas.net/book/the-book-2nd.html
- mathlib: `ProbabilityTheory.Kernel` and `MeasureTheory` integration lemmas such as
  `abs_integral_le_integral_abs` and `integral_mono`.
  Docs entry point:
  https://leanprover-community.github.io/mathlib4_docs/Mathlib/Probability/Kernel/Basic.html
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace MDP

open MeasureTheory ProbabilityTheory
open Filter Topology

open RL (discountedBackup_abs_sub_le eq_zero_of_le_mul)

section SupDist

variable {S : Type}

/-- Sup distance on value functions, using `sSup` over pointwise absolute differences. -/
noncomputable def valueSupDist [Nonempty S] (values₁ values₂ : ValueFunction S) : ℝ :=
  sSup (Set.range fun s => |values₁ s - values₂ s|)

/-- If the pointwise absolute differences are bounded above, each is at most the sup distance. -/
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

/-- On a finite state space `valueSupDist` is Mathlib's sup distance on `S → ℝ`. -/
theorem valueSupDist_eq_dist [Fintype S] [Nonempty S] (values₁ values₂ : ValueFunction S) :
    valueSupDist values₁ values₂ = dist values₁ values₂ :=
  le_antisymm (valueSupDist_le fun state => dist_le_pi_dist values₁ values₂ state)
    ((dist_pi_le_iff (valueSupDist_nonneg values₁ values₂)).2 fun state =>
      abs_sub_le_valueSupDist values₁ values₂ (Set.finite_range _).bddAbove state)

/-- If both functions are bounded in absolute value, so is their difference. -/
theorem bddAbove_abs_sub
    (values₁ values₂ : ValueFunction S)
    (h₁ : BddAbove (Set.range fun s => |values₁ s|))
    (h₂ : BddAbove (Set.range fun s => |values₂ s|)) :
    BddAbove (Set.range fun s => |values₁ s - values₂ s|) := by
  rcases h₁ with ⟨B₁, hB₁⟩
  rcases h₂ with ⟨B₂, hB₂⟩
  refine ⟨B₁ + B₂, Set.forall_mem_range.2 fun s => (abs_sub _ _).trans ?_⟩
  exact add_le_add (hB₁ ⟨s, rfl⟩) (hB₂ ⟨s, rfl⟩)

/--
For bounded value functions, `valueSupDist = 0` iff they are equal.

The boundedness hypotheses matter because `valueSupDist` is an `sSup`: in Mathlib the `sSup` of a
set of reals that is not bounded above is `0`, so without them the forward direction fails.
-/
theorem valueSupDist_eq_zero_iff [Nonempty S]
    (values₁ values₂ : ValueFunction S)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|)) :
    valueSupDist values₁ values₂ = 0 ↔ values₁ = values₂ := by
  refine ⟨fun h0 => funext fun state => sub_eq_zero.mp (abs_nonpos_iff.mp ?_), fun h => ?_⟩
  · simpa [h0] using abs_sub_le_valueSupDist values₁ values₂
      (bddAbove_abs_sub values₁ values₂ hBdd₁ hBdd₂) state
  · subst h
    simp [valueSupDist]

/-- Two bounded fixed points of `T` are equal when `T` shrinks their `valueSupDist` by a factor
`γ < 1`. -/
private theorem fixedPoint_unique [Nonempty S] {T : ValueFunction S → ValueFunction S} {γ : ℝ}
    (hγ : γ < 1) {v w : ValueFunction S} (hv : T v = v) (hw : T w = w)
    (hBddV : BddAbove (Set.range fun s => |v s|))
    (hBddW : BddAbove (Set.range fun s => |w s|))
    (hT : valueSupDist (T v) (T w) ≤ γ * valueSupDist v w) :
    v = w := by
  rw [hv, hw] at hT
  exact (valueSupDist_eq_zero_iff v w hBddV hBddW).1
    (eq_zero_of_le_mul hγ (valueSupDist_nonneg v w) hT)

end SupDist

/-!
## Contractions on Bounded Measurable Value Functions

Bounded measurable value functions, with the sup distance, form a complete metric space: they are
the measurable elements of Mathlib's `lp (fun _ : S => ℝ) ∞`, and a sup-norm limit of measurable
functions is measurable. Banach's fixed-point theorem on that space gives every `BddContraction`
exactly one bounded measurable fixed point, which its iterates reach from every bounded measurable
start.
-/

section BoundedContraction

variable {S : Type} [MeasurableSpace S]

/-- `v` is measurable and `|v|` is bounded above. -/
def IsBddMeas (v : ValueFunction S) : Prop :=
  Measurable v ∧ BddAbove (Set.range fun s => |v s|)

/-- On a finite state space with measurable singletons, every value function is bounded and
measurable. -/
theorem isBddMeas_of_finite [Finite S] [MeasurableSingletonClass S] (v : ValueFunction S) :
    IsBddMeas v :=
  ⟨measurable_of_finite v, (Set.finite_range _).bddAbove⟩

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

private theorem isBddMeas_val (f : BddMeas S) : IsBddMeas (f.1 : S → ℝ) :=
  ⟨f.2, by simpa [Real.norm_eq_abs] using memℓp_infty_iff.1 f.1.2⟩

/-- A bounded measurable function as an element of `BddMeas S`. -/
private def BddMeas.mk (v : ValueFunction S) (hv : IsBddMeas v) : BddMeas S :=
  ⟨⟨v, memℓp_infty_iff.2 (by simpa [Real.norm_eq_abs] using hv.2)⟩, hv.1⟩

/-- `T` keeps bounded measurable value functions bounded and measurable and, on them, contracts
`valueSupDist` by the factor `γ ∈ [0, 1)`. -/
structure BddContraction [Nonempty S] (γ : ℝ) (T : ValueFunction S → ValueFunction S) :
    Prop where
  /-- The factor is nonnegative. -/
  nonneg : 0 ≤ γ
  /-- The factor is strictly below one. -/
  lt_one : γ < 1
  /-- `T` maps bounded measurable value functions to bounded measurable value functions. -/
  maps : ∀ v, IsBddMeas v → IsBddMeas (T v)
  /-- On bounded measurable value functions, one application of `T` shrinks `valueSupDist` by
  `γ`. -/
  dist_le : ∀ v w, IsBddMeas v → IsBddMeas w → valueSupDist (T v) (T w) ≤ γ * valueSupDist v w

namespace BddContraction

variable [Nonempty S] {γ : ℝ} {T : ValueFunction S → ValueFunction S}

/-- `T` on `BddMeas S`. -/
private def onBddMeas (h : BddContraction γ T) (f : BddMeas S) : BddMeas S :=
  BddMeas.mk (T f.1) (h.maps _ (isBddMeas_val f))

private theorem contractingWith (h : BddContraction γ T) :
    ContractingWith γ.toNNReal (onBddMeas h) := by
  refine ⟨Real.toNNReal_lt_one.mpr h.lt_one, LipschitzWith.of_dist_le_mul fun f g => ?_⟩
  simpa [dist_eq_valueSupDist, h.nonneg, onBddMeas, BddMeas.mk] using
    h.dist_le _ _ (isBddMeas_val f) (isBddMeas_val g)

private theorem iterate_val (h : BddContraction γ T) (k : Nat) (f : BddMeas S) :
    (((onBddMeas h)^[k] f).1 : S → ℝ) = T^[k] f.1 :=
  (Function.Semiconj.iterate_right (f := fun f : BddMeas S => (f.1 : S → ℝ))
    (ga := onBddMeas h) (gb := T) (fun _ => rfl) k) f

/-- `T` has a bounded measurable fixed point, and from every bounded measurable start its iterates
converge to it in `valueSupDist`. -/
theorem exists_fixedPoint (h : BddContraction γ T) :
    ∃ vStar : ValueFunction S, IsBddMeas vStar ∧ T vStar = vStar ∧
      ∀ v, IsBddMeas v → Tendsto (fun k => valueSupDist (T^[k] v) vStar) atTop (𝓝 0) := by
  have hF := contractingWith h
  let p := ContractingWith.fixedPoint (onBddMeas h) hF
  refine ⟨p.1, isBddMeas_val p, ?_, fun v hv => ?_⟩
  · exact congrArg (fun f : BddMeas S => (f.1 : S → ℝ)) (ContractingWith.fixedPoint_isFixedPt hF)
  · have := tendsto_iff_dist_tendsto_zero.mp
      (ContractingWith.tendsto_iterate_fixedPoint hF (BddMeas.mk v hv))
    simpa [dist_eq_valueSupDist, iterate_val, BddMeas.mk] using this

/-- A bounded measurable fixed point of `T`, chosen by `exists_fixedPoint`. By `eq_fixedPoint` it
is the only one. -/
noncomputable def fixedPoint (h : BddContraction γ T) : ValueFunction S :=
  Classical.choose h.exists_fixedPoint

/-- `fixedPoint` is bounded and measurable. -/
theorem fixedPoint_isBddMeas (h : BddContraction γ T) : IsBddMeas h.fixedPoint :=
  (Classical.choose_spec h.exists_fixedPoint).1

/-- `fixedPoint` is a fixed point of `T`. -/
theorem map_fixedPoint (h : BddContraction γ T) : T h.fixedPoint = h.fixedPoint :=
  (Classical.choose_spec h.exists_fixedPoint).2.1

/-- From every bounded measurable start, the iterates of `T` converge to `fixedPoint` in
`valueSupDist`. -/
theorem tendsto_iterate (h : BddContraction γ T) {v : ValueFunction S} (hv : IsBddMeas v) :
    Tendsto (fun k => valueSupDist (T^[k] v) h.fixedPoint) atTop (𝓝 0) :=
  (Classical.choose_spec h.exists_fixedPoint).2.2 v hv

/-- Every bounded measurable fixed point of `T` is `fixedPoint`. -/
theorem eq_fixedPoint (h : BddContraction γ T) {v : ValueFunction S} (hv : IsBddMeas v)
    (hfix : T v = v) : v = h.fixedPoint :=
  fixedPoint_unique h.lt_one hfix h.map_fixedPoint hv.2 h.fixedPoint_isBddMeas.2
    (h.dist_le _ _ hv h.fixedPoint_isBddMeas)

/-- `T` has exactly one bounded measurable fixed point. -/
theorem existsUnique_fixedPoint (h : BddContraction γ T) :
    ∃! v : ValueFunction S, IsBddMeas v ∧ T v = v :=
  ⟨h.fixedPoint, ⟨h.fixedPoint_isBddMeas, h.map_fixedPoint⟩,
    fun _ hv => h.eq_fixedPoint hv.1 hv.2⟩

/-- The iterates of `T` keep a value function bounded and measurable. -/
theorem iterate_isBddMeas (h : BddContraction γ T) {v : ValueFunction S} (hv : IsBddMeas v)
    (k : Nat) : IsBddMeas (T^[k] v) := by
  induction k with
  | zero => exact hv
  | succ k ih =>
    rw [Function.iterate_succ_apply']
    exact h.maps _ ih

/-- On bounded measurable value functions, `k` applications of `T` shrink `valueSupDist` by
`γ ^ k`. -/
theorem iterate_le (h : BddContraction γ T) (k : Nat) {v w : ValueFunction S}
    (hv : IsBddMeas v) (hw : IsBddMeas w) :
    valueSupDist (T^[k] v) (T^[k] w) ≤ γ ^ k * valueSupDist v w := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', pow_succ', mul_assoc]
    exact (h.dist_le _ _ (h.iterate_isBddMeas hv k) (h.iterate_isBddMeas hw k)).trans
      (mul_le_mul_of_nonneg_left ih h.nonneg)

/-- From a bounded measurable start, the distance from the `k`-th iterate of `T` to a bounded
measurable fixed point is at most `γ ^ k` times the starting distance. -/
theorem iterate_error (h : BddContraction γ T) {v vStar : ValueFunction S} (hv : IsBddMeas v)
    (hvStar : IsBddMeas vStar) (hfix : T vStar = vStar) (k : Nat) :
    valueSupDist (T^[k] v) vStar ≤ γ ^ k * valueSupDist v vStar := by
  simpa [Function.iterate_fixed hfix k] using h.iterate_le k hv hvStar

/-- A priori error bound for iterating `T` from a bounded measurable start, computable from the
first step alone. -/
theorem apriori_error (h : BddContraction γ T) {v : ValueFunction S} (hv : IsBddMeas v)
    (k : Nat) :
    valueSupDist (T^[k] v) h.fixedPoint ≤ valueSupDist v (T v) * γ ^ k / (1 - γ) := by
  have hF := contractingWith h
  have hp : ((ContractingWith.fixedPoint (onBddMeas h) hF).1 : S → ℝ) = h.fixedPoint :=
    h.eq_fixedPoint (isBddMeas_val _)
      (congrArg (fun f : BddMeas S => (f.1 : S → ℝ)) (ContractingWith.fixedPoint_isFixedPt hF))
  have := hF.apriori_dist_iterate_fixedPoint_le (BddMeas.mk v hv) k
  rw [dist_eq_valueSupDist, dist_eq_valueSupDist, iterate_val, hp] at this
  simpa [h.nonneg, onBddMeas, BddMeas.mk] using this

end BddContraction

end BoundedContraction

section MarkovMDP

variable {S A : Type} [MeasurableSpace S] [MeasurableSpace A]

/-- For a Markov kernel, the expected next values of two bounded measurable value functions differ
by at most their sup distance. -/
theorem expectedNextValue_abs_sub_le [Nonempty S]
    (mdp : MDP S A)
    [IsMarkovKernel mdp.transition]
    (values₁ values₂ : ValueFunction S)
    (hMeas₁ : Measurable values₁)
    (hMeas₂ : Measurable values₂)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|))
    (state : S)
    (action : A) :
    |expectedNextValue mdp values₁ state action - expectedNextValue mdp values₂ state action|
      ≤ valueSupDist values₁ values₂ := by
  let μ : Measure S := mdp.transition (state, action)
  have hint₁ : Integrable values₁ μ := by
    obtain ⟨B, hB⟩ := hBdd₁
    exact Integrable.of_bound hMeas₁.aestronglyMeasurable B
      (ae_of_all _ fun s => (Real.norm_eq_abs _).trans_le (hB ⟨s, rfl⟩))
  have hint₂ : Integrable values₂ μ := by
    obtain ⟨B, hB⟩ := hBdd₂
    exact Integrable.of_bound hMeas₂.aestronglyMeasurable B
      (ae_of_all _ fun s => (Real.norm_eq_abs _).trans_le (hB ⟨s, rfl⟩))
  have hBddDiff := bddAbove_abs_sub values₁ values₂ hBdd₁ hBdd₂
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

/-- For a Markov kernel and a nonnegative discount, Bellman state-action values are Lipschitz with
constant `γ` in the sup metric. -/
theorem actionValue_abs_sub_le [Nonempty S]
    (mdp : MDP S A)
    [IsMarkovKernel mdp.transition]
    (hγ₀ : 0 ≤ mdp.discount)
    (values₁ values₂ : ValueFunction S)
    (hMeas₁ : Measurable values₁)
    (hMeas₂ : Measurable values₂)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|))
    (state : S)
    (action : A) :
    |actionValue mdp values₁ state action - actionValue mdp values₂ state action|
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  discountedBackup_abs_sub_le _ hγ₀
    (expectedNextValue_abs_sub_le mdp values₁ values₂ hMeas₁ hMeas₂ hBdd₁ hBdd₂ state action)

/-- For a Markov kernel and a nonnegative discount, Bellman expectation for a deterministic policy
is a `γ`-contraction in the sup metric:

`valueSupDist (T^π values₁) (T^π values₂) ≤ γ * valueSupDist values₁ values₂`. -/
theorem bellmanPolicy_contraction [Nonempty S]
    (mdp : MDP S A)
    [IsMarkovKernel mdp.transition]
    (hγ₀ : 0 ≤ mdp.discount)
    (policy : Policy S A)
    (values₁ values₂ : ValueFunction S)
    (hMeas₁ : Measurable values₁)
    (hMeas₂ : Measurable values₂)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|)) :
    valueSupDist (bellmanPolicy mdp policy values₁) (bellmanPolicy mdp policy values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  valueSupDist_le fun state =>
    actionValue_abs_sub_le mdp hγ₀ values₁ values₂ hMeas₁ hMeas₂ hBdd₁ hBdd₂ state
      (policy state)

/-- For a Markov kernel and a nonnegative discount, at a fixed state, Bellman optimality is a
contraction with modulus `γ` (finite action space). -/
theorem bellmanMax_abs_sub_le [Nonempty S]
    (mdp : MDP S A)
    [IsMarkovKernel mdp.transition]
    (hγ₀ : 0 ≤ mdp.discount)
    [Fintype A] [Nonempty A]
    (values₁ values₂ : ValueFunction S)
    (hMeas₁ : Measurable values₁)
    (hMeas₂ : Measurable values₂)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|))
    (state : S) :
    |bellmanMax mdp values₁ state - bellmanMax mdp values₂ state|
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  abs_sup'_sub_sup'_le _ _ _ _ _ fun action _ =>
    actionValue_abs_sub_le mdp hγ₀ values₁ values₂ hMeas₁ hMeas₂ hBdd₁ hBdd₂ state action

/-- For a Markov kernel and a nonnegative discount, Bellman optimality is a `γ`-contraction in the
sup metric (finite action space):

`valueSupDist (T* values₁) (T* values₂) ≤ γ * valueSupDist values₁ values₂`. -/
theorem bellmanMax_contraction [Nonempty S]
    (mdp : MDP S A)
    [IsMarkovKernel mdp.transition]
    (hγ₀ : 0 ≤ mdp.discount)
    [Fintype A] [Nonempty A]
    (values₁ values₂ : ValueFunction S)
    (hMeas₁ : Measurable values₁)
    (hMeas₂ : Measurable values₂)
    (hBdd₁ : BddAbove (Set.range fun s => |values₁ s|))
    (hBdd₂ : BddAbove (Set.range fun s => |values₂ s|)) :
    valueSupDist (bellmanMax mdp values₁) (bellmanMax mdp values₂)
      ≤ mdp.discount * valueSupDist values₁ values₂ :=
  valueSupDist_le
    (bellmanMax_abs_sub_le mdp hγ₀ values₁ values₂ hMeas₁ hMeas₂ hBdd₁ hBdd₂)

/-!
## Fixed Point Uniqueness

For `0 ≤ γ < 1`, each Bellman operator has **at most one** fixed point among bounded measurable
value functions. The proof is the standard one of discounted dynamic programming: a contraction has
at most one fixed point.
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
  haveI := valid.isMarkov
  fixedPoint_unique valid.discount_lt_one hv hw hBddV hBddW
    (bellmanPolicy_contraction mdp valid.discount_nonneg policy v w hMeasV hMeasW hBddV hBddW)

/--
If the Bellman optimality operator has a fixed point, it is unique among bounded measurable value
functions (finite action space).
-/
theorem bellmanMax_fixedPoint_unique [Nonempty S]
    (mdp : MDP S A)
    (valid : Valid (S := S) (A := A) mdp)
    [Fintype A] [Nonempty A]
    (v w : ValueFunction S)
    (hv : bellmanMax mdp v = v)
    (hw : bellmanMax mdp w = w)
    (hMeasV : Measurable v)
    (hMeasW : Measurable w)
    (hBddV : BddAbove (Set.range fun s => |v s|))
    (hBddW : BddAbove (Set.range fun s => |w s|)) :
    v = w :=
  haveI := valid.isMarkov
  fixedPoint_unique valid.discount_lt_one hv hw hBddV hBddW
    (bellmanMax_contraction mdp valid.discount_nonneg v w hMeasV hMeasW hBddV hBddW)

end FixedPoints

/-!
## Existence and Value Iteration

Both Bellman operators are `BddContraction`s under assumptions that `Valid` does not carry.
Rewards must be bounded, otherwise a backup of a bounded function need not be bounded. Policy
evaluation also needs a measurable policy, otherwise a backup of a measurable function need not be
measurable, and Bellman optimality needs a finite nonempty action space.
-/

section Existence

private theorem measurable_actionValue (mdp : MDP S A) (valid : Valid (S := S) (A := A) mdp)
    {v : ValueFunction S} (hv : Measurable v) :
    Measurable fun sa : S × A => actionValue mdp v sa.1 sa.2 := by
  have hE : Measurable fun sa : S × A => ∫ y, v y ∂mdp.transition sa :=
    (hv.stronglyMeasurable.integral_kernel (κ := mdp.transition)).measurable
  have hTerminal : MeasurableSet {sa : S × A | mdp.terminated sa.1 sa.2 = true} :=
    valid.measurable_terminated (measurableSet_singleton true)
  exact Measurable.ite hTerminal valid.measurable_reward
    (valid.measurable_reward.add (measurable_const.mul hE))

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
  · simp only [actionValue, discountedBackup, hdone, Bool.false_eq_true, ite_false]
    refine (abs_add_le _ _).trans (add_le_add (hR s a) ?_)
    rw [abs_mul, abs_of_nonneg valid.discount_nonneg]
    exact mul_le_mul_of_nonneg_left hE valid.discount_nonneg
  · simpa [actionValue, discountedBackup, hdone] using
      (hR s a).trans (le_add_of_nonneg_right hγB)

omit [MeasurableSpace S] in
private theorem bddAbove_of_abs_le {v : ValueFunction S} {C : ℝ} (h : ∀ s, |v s| ≤ C) :
    BddAbove (Set.range fun s => |v s|) :=
  ⟨C, Set.forall_mem_range.2 h⟩

/-- For a valid MDP with bounded rewards and a measurable policy, policy evaluation is a
`BddContraction` with factor `γ`. -/
theorem bellmanPolicy_bddContraction [Nonempty S]
    (mdp : MDP S A) (valid : Valid (S := S) (A := A) mdp)
    (policy : Policy S A) (hPolicy : Measurable policy)
    {R : ℝ} (hR : ∀ s a, |mdp.reward s a| ≤ R) :
    BddContraction mdp.discount (bellmanPolicy mdp policy) := by
  have : IsMarkovKernel mdp.transition := valid.isMarkov
  refine ⟨valid.discount_nonneg, valid.discount_lt_one, fun v hv => ⟨?_, ?_⟩,
    fun v w hv hw => bellmanPolicy_contraction mdp valid.discount_nonneg policy v w hv.1 hw.1
      hv.2 hw.2⟩
  · exact (measurable_actionValue mdp valid hv.1).comp (measurable_id.prodMk hPolicy)
  · obtain ⟨B, hB⟩ := hv.2
    exact bddAbove_of_abs_le fun s =>
      abs_actionValue_le mdp valid hR (fun s => hB ⟨s, rfl⟩) s (policy s)

/-- For a valid MDP with bounded rewards and a finite nonempty action space, Bellman optimality is
a `BddContraction` with factor `γ`. -/
theorem bellmanMax_bddContraction [Nonempty S]
    (mdp : MDP S A) (valid : Valid (S := S) (A := A) mdp)
    [Fintype A] [Nonempty A]
    {R : ℝ} (hR : ∀ s a, |mdp.reward s a| ≤ R) :
    BddContraction mdp.discount (bellmanMax mdp) := by
  have : IsMarkovKernel mdp.transition := valid.isMarkov
  refine ⟨valid.discount_nonneg, valid.discount_lt_one, fun v hv => ⟨?_, ?_⟩,
    fun v w hv hw => bellmanMax_contraction mdp valid.discount_nonneg v w hv.1 hw.1
      hv.2 hw.2⟩
  · have h := measurable_actionValue mdp valid hv.1
    convert Finset.measurable_sup' (f := fun a s => actionValue mdp v s a)
      Finset.univ_nonempty fun a _ => h.comp (measurable_id.prodMk measurable_const) using 1
    funext s
    rw [Finset.sup'_apply]
    rfl
  · obtain ⟨B, hB⟩ := hv.2
    exact bddAbove_of_abs_le fun s => abs_le.2
      ⟨(neg_le_of_abs_le (abs_actionValue_le mdp valid hR (fun s => hB ⟨s, rfl⟩) s
          (Classical.arbitrary A))).trans (Finset.le_sup' _ (Finset.mem_univ _)),
        Finset.sup'_le _ _ fun a _ =>
          le_of_abs_le (abs_actionValue_le mdp valid hR (fun s => hB ⟨s, rfl⟩) s a)⟩

/-- For bounded rewards and a measurable policy, policy evaluation has exactly one bounded
measurable fixed point. -/
theorem bellmanPolicy_existsUnique_fixedPoint [Nonempty S]
    (mdp : MDP S A) (valid : Valid (S := S) (A := A) mdp)
    (policy : Policy S A) (hPolicy : Measurable policy)
    {R : ℝ} (hR : ∀ s a, |mdp.reward s a| ≤ R) :
    ∃! v : ValueFunction S, (Measurable v ∧ BddAbove (Set.range fun s => |v s|)) ∧
      bellmanPolicy mdp policy v = v :=
  (bellmanPolicy_bddContraction mdp valid policy hPolicy hR).existsUnique_fixedPoint

/-- For bounded rewards and a finite action space, Bellman optimality has exactly one bounded
measurable fixed point. -/
theorem bellmanMax_existsUnique_fixedPoint [Nonempty S]
    (mdp : MDP S A) (valid : Valid (S := S) (A := A) mdp)
    [Fintype A] [Nonempty A]
    {R : ℝ} (hR : ∀ s a, |mdp.reward s a| ≤ R) :
    ∃! v : ValueFunction S, (Measurable v ∧ BddAbove (Set.range fun s => |v s|)) ∧
      bellmanMax mdp v = v :=
  (bellmanMax_bddContraction mdp valid hR).existsUnique_fixedPoint

/-- For bounded rewards and a finite action space, value iteration converges to the bounded
measurable fixed point of Bellman optimality from every bounded measurable start. -/
theorem tendsto_iterate_bellmanMax [Nonempty S]
    (mdp : MDP S A) (valid : Valid (S := S) (A := A) mdp)
    [Fintype A] [Nonempty A]
    {R : ℝ} (hR : ∀ s a, |mdp.reward s a| ≤ R)
    {vStar : ValueFunction S} (hMeas : Measurable vStar)
    (hBdd : BddAbove (Set.range fun s => |vStar s|))
    (hfix : bellmanMax mdp vStar = vStar)
    (v : ValueFunction S) (hv : Measurable v) (hb : BddAbove (Set.range fun s => |v s|)) :
    Tendsto (fun k => valueSupDist ((bellmanMax mdp)^[k] v) vStar) atTop (𝓝 0) := by
  have h := bellmanMax_bddContraction mdp valid hR
  rw [h.eq_fixedPoint ⟨hMeas, hBdd⟩ hfix]
  exact h.tendsto_iterate ⟨hv, hb⟩

end Existence

end MarkovMDP

end MDP
end RL
end TorchLean
