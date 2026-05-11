/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.MvPolynomial.Eval
public import Mathlib.Algebra.Ring.GeomSum
public import Mathlib.Data.Finsupp.Multiset
public import Mathlib.Data.Fintype.Perm
public import Mathlib.Data.Multiset.Fintype
public import Mathlib.Topology.Compactness.Compact
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationND
public import NN.MLTheory.Proofs.ReLU.Approx.ReLUMulApprox

/-!
# ReLU approximation on compact sets (nD)

Key theorems proved in this file:
- `approxOnC_of_mem_coordSubalg`: every coordinate-polynomial (`coordSubalg`) on a compact set `K`
  is uniformly approximable by a 2-layer ReLU MLP (in the sense `ApproxOnC`).
- `relu_universal_approximation_compact`: for compact `K` and any `f : C(K,ℝ)`, `f` is uniformly
  approximable on `K` by a 2-layer ReLU MLP.

Dependencies:
- `NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximation` (constructive 1D ReLU
  approximation).
- `NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationND` (Stone–Weierstrass density
  of coordinate polynomials on compact sets of tensor vectors).
- `NN.MLTheory.Proofs.ReLU.Bridge.ReLUMlpBridge` (lifting 1D MLP constructions to `TensorVec n`).
-/

@[expose] public section


namespace NN.MLTheory.Proofs.ReLU.Approximation.CompactSet
open _root_.Spec
open Examples

open NN.MLTheory.Proofs.UniversalApproximation
open NN.MLTheory.Proofs.ReLUMlpBridge
open NN.MLTheory.Proofs.ReLUMulApprox
open NN.MLTheory.Proofs.UniversalApproximationND

/-- `ApproxOn D f` means: on the domain `D`, the scalar function `f` can be uniformly approximated
by a single-hidden-layer ReLU MLP (`mlp_eval_nd`). -/
def ApproxOn {n : Nat} (D : Set (ReLUMlpBridge.TensorVec n)) (f : ReLUMlpBridge.TensorVec n → ℝ) :
  Prop :=
  ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : LinearSpec ℝ n hidDim) (l2 : LinearSpec ℝ hidDim 1),
    ∀ x ∈ D, |f x - mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x| < ε

namespace ApproxOn

/-- The zero function is uniformly approximable on any domain `D`. -/
theorem zero {n : Nat} (D : Set (ReLUMlpBridge.TensorVec n)) :
    ApproxOn (n := n) D (fun _ => (0 : ℝ)) := by
  intro ε hε
  -- Use exact representability of affine maps with zero weight and zero bias.
  refine ⟨2, affineIdLayer1 (n := n) (w := fun _ => (0 : ℝ)) (b := 0), affineIdLayer2, ?_⟩
  intro x hx
  have : mlpEvalNd (n := n) (hidDim := 2)
        (affineIdLayer1 (n := n) (w := fun _ => (0 : ℝ)) (b := 0)) affineIdLayer2 x = 0 := by
    simp [mlp_eval_affine_id, dot]
  simpa [this] using hε

/-- If `f` and `g` are uniformly approximable on `D`, then so is `f + g`. -/
theorem add {n : Nat} {D : Set (ReLUMlpBridge.TensorVec n)}
    {f g : ReLUMlpBridge.TensorVec n → ℝ}
    (hf : ApproxOn (n := n) D f) (hg : ApproxOn (n := n) D g) :
    ApproxOn (n := n) D (fun x => f x + g x) := by
  intro ε hε
  have hε2 : 0 < ε / 2 := by nlinarith
  rcases hf (ε / 2) hε2 with ⟨m, l1f, l2f, hf'⟩
  rcases hg (ε / 2) hε2 with ⟨k, l1g, l2g, hg'⟩
  -- Combine the two networks by appending hidden units and taking α=β=1, γ=0 at the output.
  refine ⟨m + k, appendLinearSpec (inDim := n) l1f l1g,
    combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g, ?_⟩
  intro x hx
  have hcomb :=
    mlp_eval_append_linear (inDim := n) (m := m) (n := k)
      (l1a := l1f) (l1b := l1g) (l2a := l2f) (l2b := l2g)
      (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) (x := x)
  -- Turn the combined evaluation into a triangle-inequality bound.
  have hf'' : |f x - mlpEvalNd (n := n) (hidDim := m) l1f l2f x| < ε / 2 := hf' x hx
  have hg'' : |g x - mlpEvalNd (n := n) (hidDim := k) l1g l2g x| < ε / 2 := hg' x hx
  -- Rewrite the combined network output into the two independent approximation errors.
  have hre :
      (f x + g x) - mlpEvalNd (n := n) (hidDim := m + k)
          (appendLinearSpec (inDim := n) l1f l1g)
          (combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g) x
        =
      (f x - mlpEvalNd (n := n) (hidDim := m) l1f l2f x)
        + (g x - mlpEvalNd (n := n) (hidDim := k) l1g l2g x) := by
    have hcomb' :
        mlpEvalNd (n := n) (hidDim := m + k)
            (appendLinearSpec (inDim := n) l1f l1g)
            (combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g) x
          =
        mlpEvalNd (n := n) (hidDim := m) l1f l2f x
          + mlpEvalNd (n := n) (hidDim := k) l1g l2g x := by
      -- Specialize the output-combination lemma at α=β=1 and γ=0.
      simp [hcomb, one_mul, zero_add]
    -- `a + b - (â + b̂) = (a - â) + (b - b̂)`.
    simp [hcomb', sub_eq_add_neg, add_assoc, add_left_comm, add_comm]
  -- Finish with triangle inequality.
  have htri :
      |(f x - mlpEvalNd (n := n) (hidDim := m) l1f l2f x)
          + (g x - mlpEvalNd (n := n) (hidDim := k) l1g l2g x)|
        ≤ |f x - mlpEvalNd (n := n) (hidDim := m) l1f l2f x|
          + |g x - mlpEvalNd (n := n) (hidDim := k) l1g l2g x| := by
    simpa using (abs_add_le _ _)
  have hsum : |f x - mlpEvalNd (n := n) (hidDim := m) l1f l2f x|
        + |g x - mlpEvalNd (n := n) (hidDim := k) l1g l2g x| < ε := by
    linarith [hf'', hg'']
  have : |(f x + g x) - mlpEvalNd (n := n) (hidDim := m + k)
          (appendLinearSpec (inDim := n) l1f l1g)
          (combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g) x| < ε :=
            by
    -- Rewrite with `hre`, then apply the two half-ε bounds.
    have hle : |(f x + g x) -
          mlpEvalNd (n := n) (hidDim := m + k)
            (appendLinearSpec (inDim := n) l1f l1g)
            (combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g) x|
        ≤ |f x - mlpEvalNd (n := n) (hidDim := m) l1f l2f x|
          + |g x - mlpEvalNd (n := n) (hidDim := k) l1g l2g x| := by
      simpa [hre] using htri
    exact lt_of_le_of_lt hle hsum
  exact this

/-- If `f` is uniformly approximable on `D`, then so is the scalar multiple `c • f`. -/
theorem smul {n : Nat} {D : Set (ReLUMlpBridge.TensorVec n)}
    {f : ReLUMlpBridge.TensorVec n → ℝ} (c : ℝ)
    (hf : ApproxOn (n := n) D f) :
    ApproxOn (n := n) D (fun x => c * f x) := by
  intro ε hε
  by_cases hc : c = 0
  · subst hc
    simpa [zero_mul] using (zero (n := n) D) ε hε
  have hcabs : 0 < |c| := abs_pos.2 hc
  have hε' : 0 < ε / |c| := by
    exact div_pos hε hcabs
  rcases hf (ε / |c|) hε' with ⟨m, l1, l2, hf'⟩
  -- Scale only the output layer weights/bias by `c`.
  let l2' : LinearSpec ℝ m 1 :=
    { weights := matrixMN 1 m (fun _ j => c * mat1Get l2.weights j)
      bias := vectorN 1 (fun _ => c * extractScalarOutput l2.bias) }
  refine ⟨m, l1, l2', ?_⟩
  intro x hx
  have hf'' : |f x - mlpEvalNd (n := n) (hidDim := m) l1 l2 x| < ε / |c| := hf' x hx
  -- `mlp_eval_nd` is affine in the output layer, so scaling the output layer scales the output.
  -- Use `mlp_eval_nd_eq_bias_sum` from the multiplication file to reduce to algebra.
  classical
  have hscale :
      mlpEvalNd (n := n) (hidDim := m) l1 l2' x = c * mlpEvalNd (n := n) (hidDim := m) l1 l2 x
        := by
    -- Unfold both sides using the explicit bias-plus-sum form.
    rw [mlp_eval_nd_eq_bias_sum (l1 := l1) (l2 := l2') (x := x)]
    rw [mlp_eval_nd_eq_bias_sum (l1 := l1) (l2 := l2) (x := x)]
    -- Compute the scaled bias and weights.
    simp [l2', mat1_get_matrixMN, extractScalarOutput, vectorN, Tensor.toScalar,
      mul_add, Finset.mul_sum, mul_left_comm, mul_comm]
  -- Bound `|c*f - c*mlp|` by factoring out the output-layer scale.
  have :
      |c * f x - mlpEvalNd (n := n) (hidDim := m) l1 l2' x| < ε := by
    -- Reduce to `|c| * |f-mlp| < ε`.
    have habs : |c * f x - mlpEvalNd (n := n) (hidDim := m) l1 l2' x|
        = |c| * |f x - mlpEvalNd (n := n) (hidDim := m) l1 l2 x| := by
      -- `c*f - c*mlp = c*(f-mlp)`.
      have : c * f x - mlpEvalNd (n := n) (hidDim := m) l1 l2' x
          = c * (f x - mlpEvalNd (n := n) (hidDim := m) l1 l2 x) := by
        simp [hscale, sub_eq_add_neg, mul_add]
      -- Take absolute values.
      simp [this, abs_mul]
    -- Multiply the base error by `|c|`.
    have hmul : |c| * |f x - mlpEvalNd (n := n) (hidDim := m) l1 l2 x| < |c| * (ε / |c|) := by
      exact (mul_lt_mul_of_pos_left hf'' hcabs)
    have hcancel : |c| * (ε / |c|) = ε := by
      field_simp [hc, abs_ne_zero.2 hc]
    -- Cancel the positive scale factor.
    simpa [habs, hcancel] using lt_of_lt_of_eq hmul hcancel
  exact this

end ApproxOn

-- ---------------------------------------------------------------------------
-- Same approximation predicate, but for `C(K,ℝ)` (so we can use Stone–Weierstrass directly)
-- ---------------------------------------------------------------------------

/-- `ApproxOnC K f` means: the continuous map `f : C(K,ℝ)` can be uniformly approximated (on `K`)
by a single-hidden-layer ReLU MLP (`mlp_eval_nd`, evaluated on the underlying point `x.1`). -/
def ApproxOnC {n : Nat} (K : Set (ReLUMlpBridge.TensorVec n)) (f : C(K, ℝ)) : Prop :=
  ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : LinearSpec ℝ n hidDim) (l2 : LinearSpec ℝ hidDim 1),
    ∀ x : K, |f x - mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1| < ε

namespace ApproxOnC

/-! ## Closure properties -/

/-- The zero continuous function is uniformly approximable on `K`. -/
theorem zero {n : Nat} (K : Set (ReLUMlpBridge.TensorVec n)) :
    ApproxOnC (n := n) K (0 : C(K, ℝ)) := by
  intro ε hε
  refine ⟨2, affineIdLayer1 (n := n) (w := fun _ => (0 : ℝ)) (b := 0), affineIdLayer2, ?_⟩
  intro x
  have : mlpEvalNd (n := n) (hidDim := 2)
        (affineIdLayer1 (n := n) (w := fun _ => (0 : ℝ)) (b := 0)) affineIdLayer2 x.1 = 0 := by
    simp [mlp_eval_affine_id, dot]
  simpa [this] using hε

/-- If `f` and `g` are uniformly approximable on `K`, then so is `f + g`. -/
theorem add {n : Nat} {K : Set (ReLUMlpBridge.TensorVec n)}
    {f g : C(K, ℝ)}
    (hf : ApproxOnC (n := n) K f) (hg : ApproxOnC (n := n) K g) :
    ApproxOnC (n := n) K (f + g) := by
  intro ε hε
  have hε2 : 0 < ε / 2 := by nlinarith
  rcases hf (ε / 2) hε2 with ⟨m, l1f, l2f, hf'⟩
  rcases hg (ε / 2) hε2 with ⟨k, l1g, l2g, hg'⟩
  refine ⟨m + k, appendLinearSpec (inDim := n) l1f l1g,
    combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g, ?_⟩
  intro x
  have hcomb :=
    mlp_eval_append_linear (inDim := n) (m := m) (n := k)
      (l1a := l1f) (l1b := l1g) (l2a := l2f) (l2b := l2g)
      (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) (x := x.1)
  have hf'' : |f x - mlpEvalNd (n := n) (hidDim := m) l1f l2f x.1| < ε / 2 := hf' x
  have hg'' : |g x - mlpEvalNd (n := n) (hidDim := k) l1g l2g x.1| < ε / 2 := hg' x
  have hre :
      (f x + g x) - mlpEvalNd (n := n) (hidDim := m + k)
          (appendLinearSpec (inDim := n) l1f l1g)
          (combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g) x.1
        =
      (f x - mlpEvalNd (n := n) (hidDim := m) l1f l2f x.1)
        + (g x - mlpEvalNd (n := n) (hidDim := k) l1g l2g x.1) := by
    have hcomb' :
        mlpEvalNd (n := n) (hidDim := m + k)
            (appendLinearSpec (inDim := n) l1f l1g)
            (combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g) x.1
          =
        mlpEvalNd (n := n) (hidDim := m) l1f l2f x.1
          + mlpEvalNd (n := n) (hidDim := k) l1g l2g x.1 := by
      simpa [add_assoc, add_left_comm, add_comm] using hcomb
    -- Rearrange the two approximation errors.
    simp [hcomb', sub_eq_add_neg, add_assoc, add_left_comm, add_comm]
  have htri : |(f x + g x) - mlpEvalNd (n := n) (hidDim := m + k)
          (appendLinearSpec (inDim := n) l1f l1g)
          (combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g) x.1|
        ≤ |f x - mlpEvalNd (n := n) (hidDim := m) l1f l2f x.1|
          + |g x - mlpEvalNd (n := n) (hidDim := k) l1g l2g x.1| := by
    -- Apply the triangle inequality to the two approximation errors.
    simpa [hre] using
      (abs_add_le (f x - mlpEvalNd (n := n) (hidDim := m) l1f l2f x.1)
        (g x - mlpEvalNd (n := n) (hidDim := k) l1g l2g x.1))
  have : |(f x + g x) - mlpEvalNd (n := n) (hidDim := m + k)
          (appendLinearSpec (inDim := n) l1f l1g)
          (combineOutput (m := m) (n := k) (α := (1 : ℝ)) (β := (1 : ℝ)) (γ := 0) l2f l2g) x.1|
        < ε := by
    have hsum : |f x - mlpEvalNd (n := n) (hidDim := m) l1f l2f x.1|
          + |g x - mlpEvalNd (n := n) (hidDim := k) l1g l2g x.1| < ε := by
      nlinarith [hf'', hg'']
    exact lt_of_le_of_lt htri hsum
  simpa using this

/-- If `f` is uniformly approximable on `K`, then so is the scalar multiple `c • f`. -/
theorem smul {n : Nat} {K : Set (ReLUMlpBridge.TensorVec n)}
    (c : ℝ) {f : C(K, ℝ)} (hf : ApproxOnC (n := n) K f) :
    ApproxOnC (n := n) K (c • f) := by
  by_cases hc : c = 0
  · subst hc
    simpa using (zero (n := n) K)
  · intro ε hε
    have hcabs : 0 < |c| := abs_pos.2 hc
    have hε' : 0 < ε / |c| := by exact div_pos hε hcabs
    rcases hf (ε / |c|) hε' with ⟨m, l1, l2, hf'⟩
    -- Scale only the output layer weights/bias by `c`.
    let l2' : LinearSpec ℝ m 1 :=
      { weights := matrixMN 1 m (fun _ j => c * mat1Get l2.weights j)
        bias := vectorN 1 (fun _ => c * extractScalarOutput l2.bias) }
    refine ⟨m, l1, l2', ?_⟩
    intro x
    have hscale :
        mlpEvalNd (n := n) (hidDim := m) l1 l2' x.1
          =
        c * mlpEvalNd (n := n) (hidDim := m) l1 l2 x.1 := by
      -- We prove it by unfolding the bias+sum form.
      classical
      rw [mlp_eval_nd_eq_bias_sum (l1 := l1) (l2 := l2') (x := x.1)]
      rw [mlp_eval_nd_eq_bias_sum (l1 := l1) (l2 := l2) (x := x.1)]
      simp [l2', mat1_get_matrixMN, extractScalarOutput, vectorN, Tensor.toScalar,
        mul_add, Finset.mul_sum, mul_left_comm, mul_comm]
    have habs :
        |c * f x - mlpEvalNd (n := n) (hidDim := m) l1 l2' x.1|
          =
        |c| * |f x - mlpEvalNd (n := n) (hidDim := m) l1 l2 x.1| := by
      -- `c*f - c*net = c*(f-net)`
      have : c * f x - mlpEvalNd (n := n) (hidDim := m) l1 l2' x.1
          = c * (f x - mlpEvalNd (n := n) (hidDim := m) l1 l2 x.1) := by
        simp [hscale, sub_eq_add_neg, mul_add, add_comm]
      simp [this, abs_mul]
    have hmul : |c| * |f x - mlpEvalNd (n := n) (hidDim := m) l1 l2 x.1| < |c| * (ε / |c|) := by
      exact mul_lt_mul_of_pos_left (hf' x) hcabs
    have hcancel : |c| * (ε / |c|) = ε := by
      field_simp [hc, abs_ne_zero.2 hc]
    have : |c • f x - mlpEvalNd (n := n) (hidDim := m) l1 l2' x.1| < ε := by
      -- Rewrite `hmul` using the absolute-value identity.
      have : |c| * |f x - mlpEvalNd (n := n) (hidDim := m) l1 l2 x.1| < ε := by
        exact lt_of_lt_of_eq hmul hcancel
      -- `c • f x = c * f x` in `ℝ`, then use `habs`.
      simpa [habs] using this
    simpa using this

/-- Finite sums preserve `ApproxOnC` (Finset-indexed). -/
theorem sum_finset {n : Nat} {K : Set (ReLUMlpBridge.TensorVec n)}
    {ι : Type} (s : Finset ι) (f : ι → C(K, ℝ))
    (hf : ∀ i ∈ s, ApproxOnC (n := n) K (f i)) :
    ApproxOnC (n := n) K (∑ i ∈ s, f i) := by
  classical
  -- Induct on the finset while threading the hypothesis `hf`.
  revert hf
  induction s using Finset.induction_on with
  | empty =>
      intro _hf
      simpa using (zero (n := n) K)
  | @insert a s ha ih =>
      intro hf'
      have ha' : ApproxOnC (n := n) K (f a) := hf' a (by simp [ha])
      have hs' : ∀ i ∈ s, ApproxOnC (n := n) K (f i) := by
        intro i hi
        exact hf' i (by simp [hi])
      have ih' : ApproxOnC (n := n) K (∑ i ∈ s, f i) := ih hs'
      simpa [Finset.sum_insert ha, add_comm, add_left_comm, add_assoc] using
        (add (n := n) (K := K) (f := f a) (g := ∑ i ∈ s, f i) ha' ih')

/-- Finite sums preserve `ApproxOnC` (Fintype-indexed). -/
theorem sum_fintype {n : Nat} {K : Set (ReLUMlpBridge.TensorVec n)}
    {ι : Type} [Fintype ι] (f : ι → C(K, ℝ)) (hf : ∀ i : ι, ApproxOnC (n := n) K (f i)) :
    ApproxOnC (n := n) K (∑ i : ι, f i) := by
  classical
  -- `∑ i, f i` is a `Finset.univ` sum.
  simpa using
    (sum_finset (n := n) (K := K) (s := (Finset.univ : Finset ι)) (f := f)
      (by intro i hi; simpa using hf i))

end ApproxOnC

-- ---------------------------------------------------------------------------
-- Stone–Weierstrass coordinate algebra = multivariate-polynomial evaluation range
-- ---------------------------------------------------------------------------

/--
Identify the Stone–Weierstrass coordinate subalgebra with the range of multivariate-polynomial
evaluation.

This is a small algebraic normalization lemma used to connect coordinate polynomials to
`MvPolynomial` syntax (`aeval`).
-/
theorem coordSubalg_eq_range_aeval {n : Nat} (K : Set
  (NN.MLTheory.Proofs.UniversalApproximationND.TensorVec n)) :
    NN.MLTheory.Proofs.UniversalApproximationND.StoneWeierstrass.coordSubalg (K := K) =
      (MvPolynomial.aeval (NN.MLTheory.Proofs.UniversalApproximationND.StoneWeierstrass.coord (K :=
        K))).range := by
  simpa [NN.MLTheory.Proofs.UniversalApproximationND.StoneWeierstrass.coordSubalg] using
    (Algebra.adjoin_range_eq_range_aeval (R := ℝ)
      (f := NN.MLTheory.Proofs.UniversalApproximationND.StoneWeierstrass.coord (K := K)))

-- ---------------------------------------------------------------------------
-- Polarization identity for products (sum over {±1}^d picks out the full product)
-- ---------------------------------------------------------------------------

section Polarization

open scoped BigOperators

/-- Sign associated to a Boolean: `true ↦ +1`, `false ↦ -1`. -/
noncomputable def sgn (b : Bool) : ℝ := if b then (1 : ℝ) else (-1 : ℝ)

/-- Product of signs for an assignment `ε : Fin d → Bool`. -/
noncomputable def signedProd {d : Nat} (ε : Fin d → Bool) : ℝ :=
  ∏ i : Fin d, sgn (ε i)

/-- Signed linear form `∑ i, sgn (ε i) * u i`. -/
noncomputable def signedSum {d : Nat} (ε : Fin d → Bool) (u : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, sgn (ε i) * u i

/-- Closed form for `∑ b : Bool, (sgn b)^k`. -/
lemma sum_bool_sgn_pow (k : ℕ) : (∑ b : Bool, (sgn b) ^ k) = (1 : ℝ) + (-1 : ℝ) ^ k := by
  classical
  -- `Fintype.sum_bool` expands the sum over the two Bool values.
  simp [sgn]

/-- For even exponents, `∑ b : Bool, (sgn b)^k = 2`. -/
lemma sum_bool_sgn_pow_even (k : ℕ) (hk : Even k) : (∑ b : Bool, (sgn b) ^ k) = (2 : ℝ) := by
  have h : (∑ b : Bool, (sgn b) ^ k) = (1 : ℝ) + (1 : ℝ) := by
    simpa [sum_bool_sgn_pow, hk.neg_one_pow] using (sum_bool_sgn_pow (k := k))
  -- `1 + 1 = 2`
  nlinarith

/-- For odd exponents, `∑ b : Bool, (sgn b)^k = 0`. -/
lemma sum_bool_sgn_pow_odd (k : ℕ) (hk : Odd k) : (∑ b : Bool, (sgn b) ^ k) = (0 : ℝ) := by
  simpa [sum_bool_sgn_pow, hk.neg_one_pow] using (sum_bool_sgn_pow (k := k))

-- Fiber cardinality as a finite `Nat` count.
/-- The cardinality of the fiber `{ i | p i = j }` as a natural number. -/
noncomputable def fiberCount {d : Nat} (p : Fin d → Fin d) (j : Fin d) : ℕ :=
  (Finset.univ.filter (fun i : Fin d => p i = j)).card

/--
Rewrite `∏ i, sgn (ε (p i))` as a product over fibers of `p`, i.e. as powers of `sgn (ε j)`.
-/
lemma prod_sgn_comp_eq_prod_pow_fiberCount {d : Nat} (p : Fin d → Fin d) (ε : Fin d → Bool) :
    (Finset.univ.prod fun i : Fin d => sgn (ε (p i)))
      =
    Finset.univ.prod fun j : Fin d => (sgn (ε j)) ^ (fiberCount (d := d) p j) := by
  classical
  -- Use the generic fiberwise product lemma, then simplify each fiber product as a power.
  -- `prod_fiberwise'` gives:
  --   `∏ j, ∏ i ∈ univ with p i = j, (sgn (ε j)) = ∏ i, (sgn (ε (p i)))`.
  -- We rewrite each inner product as a `pow` using `Finset.prod_const`.
  have hfib :
      (Finset.univ.prod fun j : Fin d =>
          ∏ i ∈ (Finset.univ : Finset (Fin d)) with p i = j, sgn (ε j))
        =
      (Finset.univ.prod fun i : Fin d => sgn (ε (p i))) := by
    simpa using
      (Finset.prod_fiberwise' (s := (Finset.univ : Finset (Fin d))) (g := p)
        (f := fun j : Fin d => sgn (ε j)))
  -- Rewrite the LHS to the desired `∏ j, (sgn (ε j)) ^ fiberCount p j`.
  -- Each fiber product is a constant product over a filtered finset.
  have hpow :
      (Finset.univ.prod fun j : Fin d =>
          ∏ i ∈ (Finset.univ : Finset (Fin d)) with p i = j, sgn (ε j))
        =
      (Finset.univ.prod fun j : Fin d => (sgn (ε j)) ^ (fiberCount (d := d) p j)) := by
    -- Each inner product is a constant product, hence a power by the fiber card.
    simp [fiberCount, Finset.prod_const]
  -- Combine.
  simpa [hpow] using hfib.symm

/--
The “sign cancellation coefficient” associated to a map `p : Fin d → Fin d`.

This is the coefficient that appears when expanding the polarization sum and swapping the order
of summation: it measures how many sign assignments `ε` survive after cancellations.
-/
noncomputable def signCoeff {d : Nat} (p : Fin d → Fin d) : ℝ :=
  ∑ ε : (Fin d → Bool),
    (Finset.univ.prod fun i : Fin d => sgn (ε i)) *
      (Finset.univ.prod fun i : Fin d => sgn (ε (p i)))

/-- Product-of-sums form for `signCoeff`, expressed in terms of fiber cardinalities of `p`. -/
lemma signCoeff_eq_prod_sum_pow {d : Nat} (p : Fin d → Fin d) :
    signCoeff (d := d) p
      =
    ∏ j : Fin d, (∑ b : Bool, (sgn b) ^ (fiberCount (d := d) p j + 1)) := by
  classical
  -- Rewrite the second product using fiber counts, then factor the sum over `ε` as a product over
  -- coordinates.
  have hrewrite :
      signCoeff (d := d) p
        =
      ∑ ε : (Fin d → Bool),
        (∏ j : Fin d, (sgn (ε j)) ^ (fiberCount (d := d) p j + 1)) := by
    -- expand `signCoeff`, rewrite the composed product, then combine powers.
    classical
    unfold signCoeff
    refine Finset.sum_congr rfl ?_
    intro ε hε
    have hcomp :
        (Finset.univ.prod fun i : Fin d => sgn (ε (p i)))
          =
        (Finset.univ.prod fun j : Fin d => (sgn (ε j)) ^ (fiberCount (d := d) p j)) := by
      simpa using (prod_sgn_comp_eq_prod_pow_fiberCount (d := d) p ε)
    -- Multiply by `∏ j, sgn(ε j)` and absorb into the exponent `+1`.
    -- `a * a^k = a^(k+1)` in a commutative monoid.
    -- Convert the `Finset.univ.prod` to `∏ j, ...` and use commutativity to combine the powers.
    simp [hcomp, fiberCount, Finset.prod_mul_distrib, pow_succ, mul_comm]
  -- Now factor the sum over all assignments `ε : Fin d → Bool`.
  -- This is the standard “sum over product type = product of sums” lemma (`Fintype.prod_sum`) in
  -- reverse.
  -- `Fintype.prod_sum` is stated as `∏ i, ∑ j, f i j = ∑ x, ∏ i, f i (x i)`.
  -- We use it symmetrically with `ι = Fin d` and `κ i = Bool`.
  have hfactor :
      (∑ ε : (Fin d → Bool),
          (∏ j : Fin d, (sgn (ε j)) ^ (fiberCount (d := d) p j + 1)))
        =
      (∏ j : Fin d, (∑ b : Bool, (sgn b) ^ (fiberCount (d := d) p j + 1))) := by
    simpa using
      (Fintype.prod_sum (ι := Fin d) (κ := fun _ : Fin d => Bool)
        (f := fun j b => (sgn b) ^ (fiberCount (d := d) p j + 1))).symm
  -- Put it together.
  simp [hrewrite, hfactor]

/--
Evaluate `signCoeff`: it is `2^d` iff all fibers of `p` have odd cardinality, and `0` otherwise.
-/
theorem signCoeff_eq_two_pow_iff_allOdd {d : Nat} (p : Fin d → Fin d) :
    signCoeff (d := d) p =
      if (∀ j : Fin d, Odd (fiberCount (d := d) p j)) then (2 : ℝ) ^ d else 0 := by
  classical
  -- Use the product-of-sums form.
  rw [signCoeff_eq_prod_sum_pow (d := d) p]
  by_cases hall : ∀ j : Fin d, Odd (fiberCount (d := d) p j)
  · -- Each factor is `2` since `fiberCount j + 1` is even.
    -- Reduce the goal `... = if ... then ... else ...` using `hall`.
    simp [hall]
    have hfac : ∀ j : Fin d,
        (∑ b : Bool, (sgn b) ^ (fiberCount (d := d) p j + 1)) = (2 : ℝ) := by
      intro j
      have hj : Odd (fiberCount (d := d) p j) := hall j
      -- `Odd n` means `¬Even n`, hence `Even (n+1)`.
      have hev : Even (fiberCount (d := d) p j + 1) := by
        -- `Even (n+1) ↔ ¬Even n`
        have : ¬ Even (fiberCount (d := d) p j) := by
          simpa [Nat.not_even_iff_odd] using hj
        exact (Nat.even_add_one).2 this
      -- apply the even case
      simpa using sum_bool_sgn_pow_even (k := fiberCount (d := d) p j + 1) hev
    -- Rewrite the product using `hfac`, then compute the product of the constant `2`.
    have hprod :
        (Finset.univ.prod fun j : Fin d =>
            (sgn true ^ (fiberCount (d := d) p j + 1) + sgn false ^ (fiberCount (d := d) p j + 1)))
          =
        (Finset.univ.prod fun _j : Fin d => (2 : ℝ)) := by
      classical
      refine Finset.prod_congr rfl ?_
      intro j hj
      simpa [Fintype.sum_bool, add_comm, add_left_comm, add_assoc] using hfac j
    -- `∏ j, 2 = 2^d`.
    have hconst : (Finset.univ.prod fun _j : Fin d => (2 : ℝ)) = (2 : ℝ) ^ d := by
      simp [Finset.prod_const]
    -- Unfold the `Fintype` product to a `Finset.univ.prod` and finish.
    change (Finset.univ.prod fun j : Fin d =>
        (sgn true ^ (fiberCount (d := d) p j + 1) + sgn false ^ (fiberCount (d := d) p j + 1))) = (2
          : ℝ) ^ d
    simp [hprod, hconst]
  · -- Some fiber count is even, hence one factor is `0`, so the whole product is `0`.
    -- Reduce the goal `... = if ... then ... else ...` using `hall`.
    simp [hall]
    have hex : ∃ j : Fin d, Even (fiberCount (d := d) p j) := by
      -- `¬(∀ j, Odd ...)` gives a witness with `¬Odd`, i.e. `Even`.
      have : ∃ j : Fin d, ¬ Odd (fiberCount (d := d) p j) := by
        exact not_forall.mp hall
      rcases this with ⟨j, hj⟩
      refine ⟨j, ?_⟩
      -- `¬Odd n` implies `Even n` for naturals.
      simpa [Nat.not_odd_iff_even] using hj
    rcases hex with ⟨j0, hj0⟩
    have hfactor0 :
        (∑ b : Bool, (sgn b) ^ (fiberCount (d := d) p j0 + 1)) = (0 : ℝ) := by
      -- If `fiberCount` is even, then `fiberCount+1` is odd.
      have hodd : Odd (fiberCount (d := d) p j0 + 1) := by
        -- `Even n` ↔ `¬Even (n+1)`, hence `¬Even (n+1)` which is `Odd (n+1)`.
        have : ¬ Even (fiberCount (d := d) p j0 + 1) := by
          simpa [Nat.even_add_one] using hj0
        simpa [Nat.not_even_iff_odd] using this
      simpa using sum_bool_sgn_pow_odd (k := fiberCount (d := d) p j0 + 1) hodd
    -- The product over `univ` is zero if any factor is zero.
    have : (Finset.univ.prod fun j : Fin d =>
          (sgn true ^ (fiberCount (d := d) p j + 1) + sgn false ^ (fiberCount (d := d) p j + 1))) =
            (0 : ℝ) := by
      classical
      apply Finset.prod_eq_zero (Finset.mem_univ j0)
      simpa [Fintype.sum_bool, add_comm, add_left_comm, add_assoc] using hfactor0
    -- unfold the `Fintype` product to a `Finset` product.
    change (Finset.univ.prod fun j : Fin d =>
        (sgn true ^ (fiberCount (d := d) p j + 1) + sgn false ^ (fiberCount (d := d) p j + 1))) = 0
    simpa using this

/--
For a function `p : Fin d → Fin d`, all fiber cardinalities are odd iff `p` is bijective.

Since `Fin d` is finite of size `d`, odd fibers force every fiber to have size `1`.
-/
lemma allOdd_fiberCount_iff_bijective {d : Nat} (p : Fin d → Fin d) :
    (∀ j : Fin d, Odd (fiberCount (d := d) p j)) ↔ Function.Bijective p := by
  classical
  cases d with
  | zero =>
    simp [fiberCount]
  | succ d =>
    constructor
    · intro hall
      have hsum :
          (Finset.univ.sum fun j : Fin (Nat.succ d) => fiberCount (d := Nat.succ d) p j) = Nat.succ
            d := by
        have h :=
          (Finset.card_eq_sum_card_fiberwise (f := p)
            (s := (Finset.univ : Finset (Fin (Nat.succ d))))
            (t := (Finset.univ : Finset (Fin (Nat.succ d))))
            (H := by
              intro x hx
              simp))
        simpa [fiberCount] using h.symm
      have hpos : ∀ j : Fin (Nat.succ d), 1 ≤ fiberCount (d := Nat.succ d) p j := by
        intro j
        exact Nat.succ_le_of_lt (hall j).pos
      have hone : ∀ j : Fin (Nat.succ d), fiberCount (d := Nat.succ d) p j = 1 := by
        intro j
        by_contra hj
        have hj2 : 2 ≤ fiberCount (d := Nat.succ d) p j := by
          have hjge : 1 ≤ fiberCount (d := Nat.succ d) p j := hpos j
          exact (Nat.succ_le_iff).2 (lt_of_le_of_ne hjge (Ne.symm hj))
        have hrest :
            d ≤
              ∑ k ∈ (Finset.univ : Finset (Fin (Nat.succ d))).erase j,
                fiberCount (d := Nat.succ d) p k := by
          have hle :
              (∑ k ∈ (Finset.univ : Finset (Fin (Nat.succ d))).erase j, (1 : ℕ))
                ≤
              ∑ k ∈ (Finset.univ : Finset (Fin (Nat.succ d))).erase j,
                fiberCount (d := Nat.succ d) p k := by
            refine Finset.sum_le_sum ?_
            intro k hk
            exact hpos k
          simpa using hle
        have hdecomp :
            fiberCount (d := Nat.succ d) p j
              + ∑ k ∈ (Finset.univ : Finset (Fin (Nat.succ d))).erase j,
                  fiberCount (d := Nat.succ d) p k
              =
            (Finset.univ.sum fun k : Fin (Nat.succ d) => fiberCount (d := Nat.succ d) p k) := by
          simpa using
            (Finset.add_sum_erase (s := (Finset.univ : Finset (Fin (Nat.succ d))))
              (f := fun k : Fin (Nat.succ d) => fiberCount (d := Nat.succ d) p k)
              (h := Finset.mem_univ j))
        have hbig :
            Nat.succ (Nat.succ d) ≤
              (Finset.univ.sum fun k : Fin (Nat.succ d) => fiberCount (d := Nat.succ d) p k) := by
          have hle' :
              2 + d ≤
                fiberCount (d := Nat.succ d) p j
                  + ∑ k ∈ (Finset.univ : Finset (Fin (Nat.succ d))).erase j,
                      fiberCount (d := Nat.succ d) p k := by
            exact Nat.add_le_add hj2 hrest
          have hle'' :
              Nat.succ (Nat.succ d) ≤
                fiberCount (d := Nat.succ d) p j
                  + ∑ k ∈ (Finset.univ : Finset (Fin (Nat.succ d))).erase j,
                      fiberCount (d := Nat.succ d) p k := by
            simpa [Nat.succ_eq_add_one, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hle'
          simpa [hdecomp] using hle''
        have : Nat.succ (Nat.succ d) ≤ Nat.succ d := by
          simp [hsum] at hbig
        exact Nat.not_succ_le_self (Nat.succ d) this
      refine ⟨?_, ?_⟩
      · intro a b hab
        by_contra hne
        have ha : a ∈ Finset.univ.filter (fun i : Fin (Nat.succ d) => p i = p a) := by
          simp
        have hb : b ∈ Finset.univ.filter (fun i : Fin (Nat.succ d) => p i = p a) := by
          simp [hab]
        have hlt :
            1 < (Finset.univ.filter (fun i : Fin (Nat.succ d) => p i = p a)).card := by
          exact (Finset.one_lt_card_iff).2 ⟨a, b, ha, hb, hne⟩
        have htwo :
            2 ≤ fiberCount (d := Nat.succ d) p (p a) := by
          have : 2 ≤ (Finset.univ.filter (fun i : Fin (Nat.succ d) => p i = p a)).card := by
            exact (Nat.succ_le_iff).2 hlt
          simpa [fiberCount] using this
        have h1 : fiberCount (d := Nat.succ d) p (p a) = 1 := by
          simpa [fiberCount] using hone (p a)
        exact Nat.not_succ_le_self 1 (by
          simp [h1] at htwo)
      · intro j
        have hcard : (Finset.univ.filter (fun i : Fin (Nat.succ d) => p i = j)).card = 1 := by
          simpa [fiberCount] using hone j
        have : (Finset.univ.filter (fun i : Fin (Nat.succ d) => p i = j)).Nonempty := by
          exact Finset.card_pos.1 (by simp [hcard])
        rcases this with ⟨i, hi⟩
        refine ⟨i, ?_⟩
        simpa using (Finset.mem_filter.1 hi).2
    · intro hb j
      rcases hb.2 j with ⟨i, rfl⟩
      have hset :
          (Finset.univ.filter fun k : Fin (Nat.succ d) => p k = p i) =
            ({i} : Finset (Fin (Nat.succ d))) := by
        classical
        ext k
        simp [hb.1.eq_iff]
      have h1 : fiberCount (d := Nat.succ d) p (p i) = 1 := by
        simp [fiberCount, hset]
      simp [h1]

/-- Evaluate `signCoeff`: it is `2^d` iff `p` is bijective, and `0` otherwise. -/
theorem signCoeff_eq_two_pow_iff_bijective {d : Nat} (p : Fin d → Fin d) :
    signCoeff (d := d) p = if Function.Bijective p then (2 : ℝ) ^ d else 0 := by
  classical
  have h := signCoeff_eq_two_pow_iff_allOdd (d := d) p
  by_cases hb : Function.Bijective p
  · -- reduce the goal with `hb`, then use the all-odd characterization.
    rw [if_pos hb]
    have hall : ∀ j : Fin d, Odd (fiberCount (d := d) p j) :=
      (allOdd_fiberCount_iff_bijective (d := d) p).2 hb
    simpa [hall] using h
  · rw [if_neg hb]
    have hall : ¬ (∀ j : Fin d, Odd (fiberCount (d := d) p j)) := by
      intro hall
      exact hb ((allOdd_fiberCount_iff_bijective (d := d) p).1 hall)
    simpa [hall] using h

/-! ## Polarization identity -/

set_option maxHeartbeats 1000000 in
/--
Polarization identity for products (algebraic form).

The signed sum of `d`-th powers isolates the full product `∏ i, u i`, up to the constant
`2^d * d!`.
-/
theorem polarization_prod {d : Nat} (u : Fin d → ℝ) :
    (∑ ε : (Fin d → Bool), (signedProd (d := d) ε) * (signedSum (d := d) ε u) ^ d)
      =
    (2 : ℝ) ^ d * (Nat.factorial d) * (∏ i : Fin d, u i) := by
  classical
  -- New proof: expand powers, swap sums, use `signCoeff` evaluation, and count bijections.
  have hpow :
      ∀ ε : (Fin d → Bool),
        (signedSum (d := d) ε u) ^ d
          =
        ∑ p : (Fin d → Fin d), ∏ i : Fin d, (sgn (ε (p i)) * u (p i)) := by
    intro ε
    simpa [signedSum, mul_assoc, mul_left_comm, mul_comm] using
      (Fintype.sum_pow (ι := Fin d) (f := fun i : Fin d => sgn (ε i) * u i) d)

  have hswap :
      (∑ ε : (Fin d → Bool), (signedProd (d := d) ε) * (signedSum (d := d) ε u) ^ d)
        =
      ∑ p : (Fin d → Fin d),
        (∏ i : Fin d, u (p i)) * signCoeff (d := d) p := by
    -- Pure algebraic rearrangement; keep `signedSum` folded until `hpow` fires.
    simp [hpow, signedProd, signCoeff, Finset.mul_sum,
      Finset.prod_mul_distrib, mul_left_comm, mul_comm]
    exact Finset.sum_comm

  rw [hswap]

  have hrewrite :
      (∑ p : (Fin d → Fin d), (∏ i : Fin d, u (p i)) * signCoeff (d := d) p)
        =
      ∑ p : (Fin d → Fin d),
        if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0 := by
    classical
    refine Fintype.sum_congr
      (fun p : (Fin d → Fin d) => (∏ i : Fin d, u (p i)) * signCoeff (d := d) p)
      (fun p : (Fin d → Fin d) =>
        if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0) (fun p => ?_)
    dsimp
    by_cases hb : Function.Bijective p
    · -- bijective case: both sides collapse to the same constant.
      have hprod : (∏ i : Fin d, u (p i)) = ∏ i : Fin d, u i := by
        simpa using (Function.Bijective.prod_comp (e := p) hb (g := u))
      -- left side
      rw [signCoeff_eq_two_pow_iff_bijective (d := d) p, if_pos hb]
      -- right side
      rw [if_pos hb]
      -- rewrite the `u`-product and commute.
      simp [hprod, mul_comm]
    · -- non-bijective: `signCoeff = 0` and the RHS is `0`.
      rw [signCoeff_eq_two_pow_iff_bijective (d := d) p, if_neg hb]
      rw [if_neg hb]
      simp

  rw [hrewrite]

  -- Count bijections: `|{p : Fin d → Fin d // Bijective p}| = d!`.
  have hcard_bij :
      Fintype.card {p : (Fin d → Fin d) // Function.Bijective p} = Nat.factorial d := by
    classical
    let e : ({p : (Fin d → Fin d) // Function.Bijective p} ≃ Equiv.Perm (Fin d)) :=
      { toFun := fun p => Equiv.ofBijective p.1 p.2
        invFun := fun σ => ⟨σ, σ.bijective⟩
        left_inv := by
          intro p
          ext i
          rfl
        right_inv := by
          intro σ
          ext i
          rfl }
    simpa using (Fintype.card_congr e).trans (by simpa using (Fintype.card_perm (α := Fin d)))

  have hind :
      (∑ p : (Fin d → Fin d), if Function.Bijective p then (1 : ℝ) else 0)
        = (Nat.factorial d : ℝ) := by
    classical
    have hsum :
        (∑ p : (Fin d → Fin d), if Function.Bijective p then (1 : ℝ) else 0)
          =
        ((Finset.univ.filter fun p : (Fin d → Fin d) => Function.Bijective p).card : ℝ) := by
      simp
    have hfilter :
        (Finset.univ.filter fun p : (Fin d → Fin d) => Function.Bijective p).card = Nat.factorial d
          := by
      -- Relate the filter-card to the subtype cardinality without simp-rewriting `Bijective`.
      have hmem :
          ∀ p : (Fin d → Fin d),
            p ∈ (Finset.univ.filter fun p : (Fin d → Fin d) => Function.Bijective p) ↔
              Function.Bijective p := by
        intro p
        constructor
        · intro hp
          exact (Finset.mem_filter.1 hp).2
        · intro hp
          exact Finset.mem_filter.2 ⟨Finset.mem_univ p, hp⟩
      have hcard' :
          Fintype.card {p : (Fin d → Fin d) // Function.Bijective p} =
            (Finset.univ.filter fun p : (Fin d → Fin d) => Function.Bijective p).card :=
        Fintype.card_of_subtype _ hmem
      exact (hcard'.symm.trans hcard_bij)
    -- conclude by casting the card equality
    calc
      (∑ p : (Fin d → Fin d), if Function.Bijective p then (1 : ℝ) else 0)
          = ((Finset.univ.filter fun p : (Fin d → Fin d) => Function.Bijective p).card : ℝ) := hsum
      _ = (Nat.factorial d : ℝ) := by
          exact_mod_cast hfilter

  -- Factor out the constant and finish.
  have hfact :
      (∑ p : (Fin d → Fin d),
        if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0)
        =
      ((2 : ℝ) ^ d * (∏ i : Fin d, u i)) *
        (∑ p : (Fin d → Fin d), if Function.Bijective p then (1 : ℝ) else 0) := by
    classical
    have :
        (∑ p : (Fin d → Fin d),
          if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0)
          =
        ∑ p : (Fin d → Fin d),
          ((2 : ℝ) ^ d * (∏ i : Fin d, u i)) * (if Function.Bijective p then (1 : ℝ) else 0) := by
      refine Fintype.sum_congr
        (fun p : (Fin d → Fin d) =>
          if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0)
        (fun p : (Fin d → Fin d) =>
          ((2 : ℝ) ^ d * (∏ i : Fin d, u i)) * (if Function.Bijective p then (1 : ℝ) else 0))
        (fun p => ?_)
      by_cases hb : Function.Bijective p <;> simp
    rw [this]
    simpa using
      (Finset.mul_sum (a := ((2 : ℝ) ^ d * (∏ i : Fin d, u i)))
        (f := fun p : (Fin d → Fin d) => (if Function.Bijective p then (1 : ℝ) else 0))
        (s := (Finset.univ : Finset (Fin d → Fin d)))).symm

  calc
      (∑ p : (Fin d → Fin d),
          if Function.Bijective p then (2 : ℝ) ^ d * (∏ i : Fin d, u i) else 0)
          =
        ((2 : ℝ) ^ d * (∏ i : Fin d, u i)) *
          (∑ p : (Fin d → Fin d), if Function.Bijective p then (1 : ℝ) else 0) := hfact
    _ = (2 : ℝ) ^ d * (Nat.factorial d) * (∏ i : Fin d, u i) := by
      -- Normalize by rewriting first, then use commutativity and associativity.
      rw [hind]
      ring_nf

-- The permanent proof path is the compact polarization argument above.

end Polarization

/-! ## Compact domains: boxes and linear forms -/

/-- The box `[-M,M]^n` as a subset of `TensorVec n`. -/
noncomputable def boxN (n : Nat) (M : ℝ) : Set (ReLUMlpBridge.TensorVec n) :=
  fun x => ∀ i : Fin n, toVec x i ∈ Set.Icc (-M) M

/-- A coordinate of `x ∈ boxN n M` lies in the interval `[-M, M]`. -/
lemma coord_mem_Icc {n : Nat} {M : ℝ} {x : ReLUMlpBridge.TensorVec n} (hx : x ∈ boxN n M) (i : Fin
  n) :
    toVec x i ∈ Set.Icc (-M) M :=
  hx i

/-- The weight vector `e_i + e_j` (sum of two standard basis vectors). -/
noncomputable def wPlus {n : Nat} (i j : Fin n) : Fin n → ℝ :=
  fun k => stdBasis (n := n) i k + stdBasis (n := n) j k

/-- The weight vector `e_i - e_j` (difference of two standard basis vectors). -/
noncomputable def wMinus {n : Nat} (i j : Fin n) : Fin n → ℝ :=
  fun k => stdBasis (n := n) i k - stdBasis (n := n) j k

/-- Linearity of `dot` in the weight argument: `dot (w1+w2) = dot w1 + dot w2`. -/
lemma dot_add {n : Nat} (w1 w2 : Fin n → ℝ) (x : ReLUMlpBridge.TensorVec n) :
    dot (fun k => w1 k + w2 k) x = dot w1 x + dot w2 x := by
  classical
  simp [dot, add_mul, Finset.sum_add_distrib]

/-- Negation compatibility for `dot`: `dot (-w) = - dot w`. -/
lemma dot_neg {n : Nat} (w : Fin n → ℝ) (x : ReLUMlpBridge.TensorVec n) :
    dot (fun k => -w k) x = - dot w x := by
  classical
  simp [dot, Finset.sum_neg_distrib]

/-- `dot (e_i + e_j) x = x_i + x_j` for `TensorVec` coordinates. -/
lemma dot_wPlus {n : Nat} (i j : Fin n) (x : ReLUMlpBridge.TensorVec n) :
    dot (wPlus (n := n) i j) x = toVec x i + toVec x j := by
  classical
  have hadd :
      dot (wPlus (n := n) i j) x =
        dot (stdBasis (n := n) i) x + dot (stdBasis (n := n) j) x := by
    simpa [wPlus] using dot_add (n := n) (w1 := stdBasis (n := n) i) (w2 := stdBasis (n := n) j) x
  simp [hadd, dot_stdBasis]

/-- `dot (e_i - e_j) x = x_i - x_j` for `TensorVec` coordinates. -/
lemma dot_wMinus {n : Nat} (i j : Fin n) (x : ReLUMlpBridge.TensorVec n) :
    dot (wMinus (n := n) i j) x = toVec x i - toVec x j := by
  classical
  have hadd :
      dot (wMinus (n := n) i j) x =
        dot (stdBasis (n := n) i) x + dot (fun k => - stdBasis (n := n) j k) x := by
    -- rewrite `wMinus` as `w1 + (-w2)` then apply linearity
    simpa [wMinus, sub_eq_add_neg, add_assoc] using
      dot_add (n := n) (w1 := stdBasis (n := n) i) (w2 := fun k => - stdBasis (n := n) j k) x
  -- finish with `dot_stdBasis` and `dot_neg`
  have hneg : dot (fun k => - stdBasis (n := n) j k) x = - dot (stdBasis (n := n) j) x := by
    simpa using dot_neg (n := n) (w := stdBasis (n := n) j) x
  -- rewrite the RHS as `a + (-b)`
  simp [hadd, hneg, dot_stdBasis, sub_eq_add_neg]

/-- If `x ∈ [-M,M]^n`, then `x_i + x_j ∈ [-2M, 2M]`. -/
lemma sum_mem_Icc {n : Nat} {M : ℝ} (_hM : 0 ≤ M) {x : ReLUMlpBridge.TensorVec n} (hx : x ∈ boxN n
  M) (i j : Fin n) :
    dot (wPlus (n := n) i j) x ∈ Set.Icc (-2*M) (2*M) := by
  have hxi := coord_mem_Icc (n := n) (M := M) hx i
  have hxj := coord_mem_Icc (n := n) (M := M) hx j
  have hxi_l : -M ≤ toVec x i := hxi.1
  have hxi_u : toVec x i ≤ M := hxi.2
  have hxj_l : -M ≤ toVec x j := hxj.1
  have hxj_u : toVec x j ≤ M := hxj.2
  have hl : -(2*M) ≤ toVec x i + toVec x j := by linarith
  have hu : toVec x i + toVec x j ≤ 2*M := by linarith
  simpa [dot_wPlus] using And.intro hl hu

/-- If `x ∈ [-M,M]^n`, then `x_i - x_j ∈ [-2M, 2M]`. -/
lemma diff_mem_Icc {n : Nat} {M : ℝ} (_hM : 0 ≤ M) {x : ReLUMlpBridge.TensorVec n} (hx : x ∈ boxN n
  M) (i j : Fin n) :
    dot (wMinus (n := n) i j) x ∈ Set.Icc (-2*M) (2*M) := by
  have hxi := coord_mem_Icc (n := n) (M := M) hx i
  have hxj := coord_mem_Icc (n := n) (M := M) hx j
  have hxi_l : -M ≤ toVec x i := hxi.1
  have hxi_u : toVec x i ≤ M := hxi.2
  have hxj_l : -M ≤ toVec x j := hxj.1
  have hxj_u : toVec x j ≤ M := hxj.2
  have hl : -(2*M) ≤ toVec x i - toVec x j := by linarith
  have hu : toVec x i - toVec x j ≤ 2*M := by linarith
  simpa [dot_wMinus] using And.intro hl hu

/--
Coordinate multiplication is uniformly approximable on the box `[-M,M]^n`.

More precisely: for fixed indices `i,j : Fin n`, the function `x ↦ x_i * x_j` can be uniformly
approximated on `boxN n M` by a single-hidden-layer ReLU MLP.
-/
theorem relu_mul_coord_universal_approximation_box
    {n : Nat} {M : ℝ} (hM : 0 < M) (i j : Fin n) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : LinearSpec ℝ n hidDim) (l2 : LinearSpec ℝ hidDim 1),
      ∀ x ∈ boxN n M, |(toVec x i * toVec x j) - mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x| <
        ε := by
  classical
  intro ε hε
  have hM0 : 0 ≤ M := le_of_lt hM
  -- Approximate `square` on `[-2M,2M]` with error `δ = 2ε`.
  let δ : ℝ := 2*ε
  have hδ : 0 < δ := by nlinarith
  have h_ab : (-2*M) < (2*M) := by nlinarith
  have hL : 0 < (4*M) := by nlinarith
  have h_lip :
      ∀ x ∈ Set.Icc (-2*M) (2*M), ∀ y ∈ Set.Icc (-2*M) (2*M),
        |(x*x) - (y*y)| ≤ (4*M) * |x - y| := by
    intro x hx y hy
    have h :=
      square_lipschitz_Icc (R := 2*M) (by nlinarith [hM0]) x (by simpa using hx) y (by simpa using
        hy)
    convert h using 1
    ring
  rcases relu_universal_approximation_Icc (f := fun u => u*u) (a := -2*M) (b := 2*M) (L := 4*M)
      h_ab hL h_lip δ hδ with ⟨hidSq, l1Sq, l2Sq, hSq⟩
  -- Lift to `u = x_i + x_j` and `u = x_i - x_j`.
  let l1Plus : LinearSpec ℝ n hidSq := liftLayer1From1d (n := n) l1Sq (wPlus (n := n) i j) 0
  let l1Minus : LinearSpec ℝ n hidSq := liftLayer1From1d (n := n) l1Sq (wMinus (n := n) i j) 0
  let l1Prod : LinearSpec ℝ n (hidSq + hidSq) := appendLinearSpec (inDim := n) l1Plus l1Minus
  let l2Prod : LinearSpec ℝ (hidSq + hidSq) 1 :=
    combineOutput (m := hidSq) (n := hidSq) (α := (1/4 : ℝ)) (β := (-1/4 : ℝ)) (γ := 0) l2Sq l2Sq
  refine ⟨hidSq + hidSq, l1Prod, l2Prod, ?_⟩
  intro x hx
  have hx_plus : dot (wPlus (n := n) i j) x ∈ Set.Icc (-2*M) (2*M) := sum_mem_Icc (n := n) (M := M)
    hM0 hx i j
  have hx_minus : dot (wMinus (n := n) i j) x ∈ Set.Icc (-2*M) (2*M) := diff_mem_Icc (n := n) (M :=
    M) hM0 hx i j
  have hplus_eval :
      mlpEvalNd (n := n) (hidDim := hidSq) l1Plus l2Sq x =
        mlpEval1d hidSq l1Sq l2Sq (dot (wPlus (n := n) i j) x) := by
    simpa [l1Plus] using
      (mlp_eval_lift_from_1d (n := n) (hidDim := hidSq) l1Sq l2Sq (wPlus (n := n) i j) 0 x)
  have hminus_eval :
      mlpEvalNd (n := n) (hidDim := hidSq) l1Minus l2Sq x =
        mlpEval1d hidSq l1Sq l2Sq (dot (wMinus (n := n) i j) x) := by
    simpa [l1Minus] using
      (mlp_eval_lift_from_1d (n := n) (hidDim := hidSq) l1Sq l2Sq (wMinus (n := n) i j) 0 x)
  have hcomb :
      mlpEvalNd (n := n) (hidDim := hidSq + hidSq) l1Prod l2Prod x
        =
      (1/4 : ℝ) * mlpEvalNd (n := n) (hidDim := hidSq) l1Plus l2Sq x
        + (-1/4 : ℝ) * mlpEvalNd (n := n) (hidDim := hidSq) l1Minus l2Sq x := by
    have :=
      mlp_eval_append_linear (inDim := n) (m := hidSq) (n := hidSq)
        (l1a := l1Plus) (l1b := l1Minus) (l2a := l2Sq) (l2b := l2Sq)
        (α := (1/4 : ℝ)) (β := (-1/4 : ℝ)) (γ := 0) (x := x)
    simpa [l1Prod, l2Prod, add_assoc, add_left_comm, add_comm] using this
  have hsq_plus :
      |(dot (wPlus (n := n) i j) x) * (dot (wPlus (n := n) i j) x)
        - mlpEval1d hidSq l1Sq l2Sq (dot (wPlus (n := n) i j) x)| < δ :=
    hSq (dot (wPlus (n := n) i j) x) hx_plus
  have hsq_minus :
      |(dot (wMinus (n := n) i j) x) * (dot (wMinus (n := n) i j) x)
        - mlpEval1d hidSq l1Sq l2Sq (dot (wMinus (n := n) i j) x)| < δ :=
    hSq (dot (wMinus (n := n) i j) x) hx_minus
  -- Finish with `uv = ((u+v)^2 - (u-v)^2)/4` and the same triangle bound as the 2D proof.
  have hmul :
      (toVec x i * toVec x j)
        = ((dot (wPlus (n := n) i j) x) * (dot (wPlus (n := n) i j) x)
            - (dot (wMinus (n := n) i j) x) * (dot (wMinus (n := n) i j) x)) / 4 := by
    have := mul_identity (toVec x i) (toVec x j)
    simpa [dot_wPlus, dot_wMinus, sub_eq_add_neg, add_assoc, add_comm, add_left_comm] using this
  -- Main error bound
  have : |(toVec x i * toVec x j) - mlpEvalNd (n := n) (hidDim := hidSq + hidSq) l1Prod l2Prod x|
    < ε := by
    rw [hmul, hcomb, hplus_eval, hminus_eval]
    set e1 := (dot (wPlus (n := n) i j) x) * (dot (wPlus (n := n) i j) x)
        - mlpEval1d hidSq l1Sq l2Sq (dot (wPlus (n := n) i j) x) with he1
    set e2 := (dot (wMinus (n := n) i j) x) * (dot (wMinus (n := n) i j) x)
        - mlpEval1d hidSq l1Sq l2Sq (dot (wMinus (n := n) i j) x) with he2
    have hrew :
        ((dot (wPlus (n := n) i j) x) * (dot (wPlus (n := n) i j) x)
              - (dot (wMinus (n := n) i j) x) * (dot (wMinus (n := n) i j) x)) / 4
            - ((1 / 4 : ℝ) * mlpEval1d hidSq l1Sq l2Sq (dot (wPlus (n := n) i j) x) +
                (-1 / 4 : ℝ) * mlpEval1d hidSq l1Sq l2Sq (dot (wMinus (n := n) i j) x))
          =
        (e1 - e2) / 4 := by
      subst e1 e2
      ring
    have htri : |e1 - e2| ≤ |e1| + |e2| := by
      simpa [sub_eq_add_neg] using (abs_add_le e1 (-e2))
    have habs : |(e1 - e2) / 4| = |e1 - e2| / 4 := by
      simp [abs_div]
    have he1lt : |e1| < δ := by simpa [he1] using hsq_plus
    have he2lt : |e2| < δ := by simpa [he2] using hsq_minus
    have hsumlt : |e1| + |e2| < 2*δ := by linarith
    have hmain : |(e1 - e2) / 4| < ε := by
      have hle : |e1 - e2| / 4 ≤ (|e1| + |e2|) / 4 := by
        have := div_le_div_of_nonneg_right htri (by norm_num : (0:ℝ) ≤ 4)
        simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using this
      have hlt : (|e1| + |e2|) / 4 < ε := by
        have h' : (|e1| + |e2|) / 4 < (2*δ) / 4 :=
          div_lt_div_of_pos_right hsumlt (by norm_num : (0:ℝ) < 4)
        have hEq : (2*δ) / 4 = ε := by
          simp [δ]
          ring
        exact lt_of_lt_of_eq h' hEq
      exact lt_of_le_of_lt (by simpa [habs] using hle) hlt
    have : |((dot (wPlus (n := n) i j) x) * (dot (wPlus (n := n) i j) x)
              - (dot (wMinus (n := n) i j) x) * (dot (wMinus (n := n) i j) x)) / 4
            - ((1 / 4 : ℝ) * mlpEval1d hidSq l1Sq l2Sq (dot (wPlus (n := n) i j) x) +
                (-1 / 4 : ℝ) * mlpEval1d hidSq l1Sq l2Sq (dot (wMinus (n := n) i j) x))| < ε := by
      have hrew' :
          ((dot (wPlus (n := n) i j) x) * (dot (wPlus (n := n) i j) x)
                - (dot (wMinus (n := n) i j) x) * (dot (wMinus (n := n) i j) x)) / 4
              - ((4 : ℝ)⁻¹ * mlpEval1d hidSq l1Sq l2Sq (dot (wPlus (n := n) i j) x) +
                  (-1 / 4 : ℝ) * mlpEval1d hidSq l1Sq l2Sq (dot (wMinus (n := n) i j) x))
            =
          (e1 - e2) / 4 := by
        simpa [one_div] using hrew
      simpa [hrew'] using hmain
    simpa [sub_eq_add_neg, add_assoc, add_comm, add_left_comm] using this
  exact this

-- ---------------------------------------------------------------------------
-- Next building block: approximating `u ↦ u^d` on bounded intervals
-- ---------------------------------------------------------------------------

/--
Lipschitz bound for the power function on a bounded interval.

For `x,y ∈ [-R,R]`, the map `u ↦ u^d` is Lipschitz with constant `d * R^(d-1)` (with the
convention that the `d=0` case is constant).
-/
lemma pow_lipschitz_Icc {R : ℝ} (hR : 0 ≤ R) :
    ∀ d : ℕ, ∀ x ∈ Set.Icc (-R) R, ∀ y ∈ Set.Icc (-R) R,
      |x ^ d - y ^ d| ≤ (d * R ^ (d - 1)) * |x - y| := by
  intro d
  cases d with
  | zero =>
    intro x hx y hy
    simp
  | succ d =>
    intro x hx y hy
    have hxabs : |x| ≤ R := by
      have hx' : -R ≤ x ∧ x ≤ R := by simpa [Set.Icc] using hx
      exact (abs_le).2 hx'
    have hyabs : |y| ≤ R := by
      have hy' : -R ≤ y ∧ y ≤ R := by simpa [Set.Icc] using hy
      exact (abs_le).2 hy'
    -- Use `x^n - y^n = (∑ x^i*y^(n-1-i)) * (x-y)` and bound the geometric sum by `n * R^(n-1)`.
    have hfactor :
        x ^ (d + 1) - y ^ (d + 1) =
          (∑ i ∈ Finset.range (d + 1), x ^ i * y ^ (d - i)) * (x - y) := by
      -- `geom_sum₂_mul` gives `(...)*(x-y) = x^n - y^n`.
      simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
        (by
          have := geom_sum₂_mul x y (d + 1)
          -- unfold `(d+1)-1-i` to `d-i`
          simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc, Nat.succ_eq_add_one] using
            this.symm)
    have hsum_bound :
        |∑ i ∈ Finset.range (d + 1), x ^ i * y ^ (d - i)|
          ≤ (d + 1) * R ^ d := by
      -- Bound each term by `R^d` using `|x|,|y| ≤ R`.
      have hterm :
          ∀ i ∈ Finset.range (d + 1), |x ^ i * y ^ (d - i)| ≤ R ^ d := by
        intro i hi
        have hxpow : |x ^ i| ≤ R ^ i := by
          simpa [abs_pow] using pow_le_pow_left₀ (abs_nonneg x) hxabs i
        have hypow : |y ^ (d - i)| ≤ R ^ (d - i) := by
          simpa [abs_pow] using pow_le_pow_left₀ (abs_nonneg y) hyabs (d - i)
        calc
          |x ^ i * y ^ (d - i)| = |x ^ i| * |y ^ (d - i)| := by simp [abs_mul]
          _ ≤ (R ^ i) * |y ^ (d - i)| := by
            exact mul_le_mul_of_nonneg_right hxpow (abs_nonneg _)
          _ ≤ (R ^ i) * (R ^ (d - i)) := by
            exact mul_le_mul_of_nonneg_left hypow (pow_nonneg hR _)
          _ = R ^ d := by
            -- `i + (d-i) = d` for `i ≤ d`.
            have hid : i ≤ d := by
              -- from `i < d+1`
              exact Nat.le_of_lt_succ (Finset.mem_range.1 hi)
            calc
              (R ^ i) * (R ^ (d - i)) = R ^ (i + (d - i)) := by
                simp [pow_add]
              _ = R ^ d := by
                simp [Nat.add_sub_of_le hid]
      -- Sum the bound.
      calc
        |∑ i ∈ Finset.range (d + 1), x ^ i * y ^ (d - i)|
            ≤ ∑ i ∈ Finset.range (d + 1), |x ^ i * y ^ (d - i)| := by
              simpa using (Finset.abs_sum_le_sum_abs (s := Finset.range (d + 1))
                (f := fun i => x ^ i * y ^ (d - i)))
        _ ≤ ∑ _i ∈ Finset.range (d + 1), R ^ d := by
              exact Finset.sum_le_sum (fun i hi => hterm i hi)
        _ = (d + 1) * R ^ d := by
              simp []
    calc
      |x ^ (d + 1) - y ^ (d + 1)| = |(∑ i ∈ Finset.range (d + 1), x ^ i * y ^ (d - i)) * (x - y)| :=
        by
        simp [hfactor]
      _ = |∑ i ∈ Finset.range (d + 1), x ^ i * y ^ (d - i)| * |x - y| := by
        simp [abs_mul]
      _ ≤ ((d + 1) * R ^ d) * |x - y| := by
        exact mul_le_mul_of_nonneg_right hsum_bound (abs_nonneg (x - y))
      _ = ((Nat.succ d) * R ^ (Nat.succ d - 1)) * |x - y| := by
        simp

/--
Uniform approximation of the power function on a bounded interval by a 1D ReLU MLP.

This packages the 1D Lipschitz ReLU approximation theorem for the specific function
`x ↦ x^d` on `[-R,R]`.
-/
theorem relu_universal_approximation_pow_Icc {R : ℝ} (hR : 0 < R) (d : ℕ) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : LinearSpec ℝ 1 hidDim) (l2 : LinearSpec ℝ hidDim 1),
      ∀ x ∈ Set.Icc (-R) R, |x ^ d - mlpEval1d hidDim l1 l2 x| < ε := by
  intro ε hε
  have hR0 : 0 ≤ R := le_of_lt hR
  have hab : (-R) < R := by nlinarith
  let L : ℝ := d * R ^ (d - 1)
  have hLip :
      ∀ x ∈ Set.Icc (-R) R, ∀ y ∈ Set.Icc (-R) R, |x ^ d - y ^ d| ≤ L * |x - y| := by
    intro x hx y hy
    simpa [L, mul_assoc, mul_left_comm, mul_comm] using
      (pow_lipschitz_Icc (R := R) hR0 d x hx y hy)
  -- Use the existing 1D ReLU approximation theorem, which is stated for Lipschitz functions.
  rcases relu_universal_approximation_Icc (f := fun x : ℝ => x ^ d) (a := -R) (b := R)
        (L := max L 1) hab (lt_of_lt_of_le (show (0 : ℝ) < 1 from zero_lt_one) (le_max_right L 1))
          (by
          intro x hx y hy
          have := hLip x hx y hy
          -- `L ≤ max L 1`
          exact le_trans this (by
            have hmax : L * |x - y| ≤ max L 1 * |x - y| :=
              mul_le_mul_of_nonneg_right (le_max_left L 1) (abs_nonneg (x - y))
            simpa [mul_assoc] using hmax))
        ε hε with ⟨hidDim, l1, l2, h⟩
  exact ⟨hidDim, l1, l2, h⟩

-- ---------------------------------------------------------------------------
-- Stone–Weierstrass → ReLU bridge (compact-set approximation)
-- ---------------------------------------------------------------------------

section ReLUStoneWeierstrassBridge

open NN.MLTheory.Proofs.UniversalApproximationND.StoneWeierstrass
open scoped BigOperators
open ContinuousMap

variable {n : Nat} (K : Set (ReLUMlpBridge.TensorVec n)) [CompactSpace K]

/-- The linear form `x ↦ w ⋅ x` as a continuous map on the compact set `K`. -/
noncomputable def linFormC (K : Set (ReLUMlpBridge.TensorVec n)) (w : Fin n → ℝ) : C(K, ℝ) :=
  ∑ i : Fin n, w i • StoneWeierstrass.coord (K := K) i

omit [CompactSpace K] in
/-- Evaluate `linFormC` as the dot product `w ⋅ x` on the underlying tensor vector. -/
lemma linFormC_apply (w : Fin n → ℝ) (x : K) :
    linFormC K w x = dot w x.1 := by
  classical
  simp [linFormC, StoneWeierstrass.coord, dot, ReLUMlpBridge.toVec]

-- Approximate `x ↦ (w⋅x)^d` on a compact set, using the 1D power approximation and ridge lifting.
/-- Uniform approximation of the continuous function `x ↦ (w ⋅ x)^d` on `K` by a 2-layer ReLU MLP.
  -/
theorem approx_pow_linFormC (w : Fin n → ℝ) (d : ℕ) :
    ApproxOnC (n := n) K ((linFormC K w) ^ d) := by
  intro ε hε
  let R : ℝ := max 1 ‖linFormC K w‖
  have hR : 0 < R := lt_of_lt_of_le zero_lt_one (le_max_left 1 ‖linFormC K w‖)
  rcases relu_universal_approximation_pow_Icc (R := R) hR d ε hε with ⟨hidDim, l1, l2, hpow⟩
  let l1' : LinearSpec ℝ n hidDim := liftLayer1From1d (n := n) l1 w 0
  refine ⟨hidDim, l1', l2, ?_⟩
  intro x
  have hxR : linFormC K w x ∈ Set.Icc (-R) R := by
    have habs : |linFormC K w x| ≤ ‖linFormC K w‖ := by
      simpa using (ContinuousMap.norm_coe_le_norm (linFormC K w) x)
    have habsR : |linFormC K w x| ≤ R := le_trans habs (le_max_right 1 ‖linFormC K w‖)
    exact (abs_le).1 habsR
  have hpowx :
      |(linFormC K w x) ^ d - mlpEval1d hidDim l1 l2 (linFormC K w x)| < ε :=
    hpow (linFormC K w x) hxR
  have hlift :
      mlpEvalNd (n := n) (hidDim := hidDim) l1' l2 x.1
        =
      mlpEval1d hidDim l1 l2 (dot w x.1) := by
    simpa [l1'] using
      (mlp_eval_lift_from_1d (n := n) (hidDim := hidDim) l1 l2 w 0 x.1)
  have hdot : dot w x.1 = linFormC K w x :=
    (linFormC_apply (K := K) (w := w) (x := x)).symm
  -- rewrite the approximator in terms of the lifted network.
  simpa [hlift, hdot] using hpowx

end ReLUStoneWeierstrassBridge

section ReLUStoneWeierstrassBridgeProducts

open NN.MLTheory.Proofs.UniversalApproximationND.StoneWeierstrass
open scoped BigOperators
open ContinuousMap

variable {n : Nat} (K : Set (ReLUMlpBridge.TensorVec n)) [CompactSpace K]

-- Weight vector for the signed sum `∑ i, sgn(ε i) * x_{idx i}`.
/-- Weight vector encoding a signed sum of selected coordinates `∑ i, sgn(ε i) * x_{idx i}`. -/
noncomputable def wSigned {d : Nat} (idx : Fin d → Fin n) (ε : Fin d → Bool) : Fin n → ℝ :=
  fun j : Fin n => ∑ i : Fin d, sgn (ε i) * stdBasis (n := n) (idx i) j

/-- `dot (wSigned idx ε) x` computes the signed sum of the selected coordinates of `x`. -/
lemma dot_wSigned_eq_signedSum {d : Nat} (idx : Fin d → Fin n) (ε : Fin d → Bool)
    (x : ReLUMlpBridge.TensorVec n) :
    dot (wSigned (n := n) idx ε) x =
      signedSum (d := d) ε (fun i : Fin d => toVec x (idx i)) := by
  classical
  -- Expand `dot` and rearrange into a sum of basis-vector dots.
  unfold dot wSigned signedSum
  -- First distribute the `toVec x j` multiplier across the inner sum.
  have hdist :
      (∑ j : Fin n, (∑ i : Fin d, sgn (ε i) * stdBasis (n := n) (idx i) j) * toVec x j)
        =
      ∑ j : Fin n, ∑ i : Fin d, (sgn (ε i) * stdBasis (n := n) (idx i) j) * toVec x j := by
    refine Fintype.sum_congr _ _ (fun j => ?_)
    -- `(∑ i, a i) * b = ∑ i, a i * b` on `Finset.univ`.
    simpa using
      (Finset.sum_mul (s := (Finset.univ : Finset (Fin d)))
        (f := fun i : Fin d => sgn (ε i) * stdBasis (n := n) (idx i) j)
        (a := toVec x j))
  -- Swap the two sums.
  have hswap :
      (∑ j : Fin n, ∑ i : Fin d, (sgn (ε i) * stdBasis (n := n) (idx i) j) * toVec x j)
        =
      ∑ i : Fin d, ∑ j : Fin n, (sgn (ε i) * stdBasis (n := n) (idx i) j) * toVec x j := by
    -- This is the standard `Finset.sum_comm` over `Finset.univ`.
    exact Finset.sum_comm
  -- Simplify the inner sum using `dot_stdBasis`.
  have hinner :
      ∀ i : Fin d,
        (∑ j : Fin n, (sgn (ε i) * stdBasis (n := n) (idx i) j) * toVec x j)
          =
        sgn (ε i) * toVec x (idx i) := by
    intro i
    -- Factor out the constant `sgn (ε i)` and recognize `dot (stdBasis (idx i)) x`.
    have :
        (∑ j : Fin n, (sgn (ε i) * stdBasis (n := n) (idx i) j) * toVec x j)
          =
        sgn (ε i) * (∑ j : Fin n, stdBasis (n := n) (idx i) j * toVec x j) := by
      -- `∑ j, (c * a j) * b j = c * ∑ j, a j * b j`
      simp [mul_assoc, Finset.mul_sum]
    have hdotbasis :
        (∑ j : Fin n, stdBasis (n := n) (idx i) j * toVec x j) = toVec x (idx i) := by
      simpa [dot] using (dot_stdBasis (n := n) (i := idx i) (x := x))
    calc
      (∑ j : Fin n, (sgn (ε i) * stdBasis (n := n) (idx i) j) * toVec x j)
          = sgn (ε i) * (∑ j : Fin n, stdBasis (n := n) (idx i) j * toVec x j) := this
      _ = sgn (ε i) * toVec x (idx i) := by simp [hdotbasis]
  -- Put everything together.
  calc
    (∑ j : Fin n, (∑ i : Fin d, sgn (ε i) * stdBasis (n := n) (idx i) j) * toVec x j)
        = ∑ i : Fin d, ∑ j : Fin n, (sgn (ε i) * stdBasis (n := n) (idx i) j) * toVec x j := by
          simpa [hdist] using hswap
    _ = ∑ i : Fin d, sgn (ε i) * toVec x (idx i) := by
          refine Fintype.sum_congr _ _ (fun i => ?_)
          simpa using hinner i

-- The coordinate-product function is approximable on any compact set:
-- it is a linear combination of ridge-lifted 1D power approximations via polarization.
/--
Uniform approximation of a coordinate-product monomial on a compact set.

For a fixed index map `idx : Fin d → Fin n`, the function
`x ↦ ∏ i, x_{idx i}` (expressed as a product of coordinate maps on `K`) is uniformly approximable
by a 2-layer ReLU MLP.
-/
theorem approx_coordProd_fin {d : Nat} (idx : Fin d → Fin n) :
    ApproxOnC (n := n) K (∏ i : Fin d, StoneWeierstrass.coord (K := K) (idx i)) := by
  classical
  -- Rewrite the product using the polarization identity, then approximate that RHS.
  let C : ℝ := (2 : ℝ) ^ d * (Nat.factorial d)
  have hCpos : 0 < C := by
    have hpow : 0 < (2 : ℝ) ^ d := by
      exact pow_pos (by norm_num : (0 : ℝ) < 2) d
    have hfac : 0 < (Nat.factorial d : ℝ) := by
      exact_mod_cast Nat.factorial_pos d
    simpa [C, mul_assoc] using mul_pos hpow hfac
  have hCne : C ≠ 0 := ne_of_gt hCpos

  -- Define the polarization RHS as a continuous map.
  let term : (Fin d → Bool) → C(K, ℝ) :=
    fun ε =>
      (signedProd (d := d) ε) • ((linFormC K (wSigned (n := n) idx ε)) ^ d)
  let rhs : C(K, ℝ) := (1 / C) • (∑ ε : (Fin d → Bool), term ε)

  have hrhs_eq :
      rhs = (∏ i : Fin d, StoneWeierstrass.coord (K := K) (idx i)) := by
    ext x
    -- reduce to a pointwise real identity and apply `polarization_prod`.
    have hpol := polarization_prod (d := d) (u := fun i : Fin d => toVec x.1 (idx i))
    -- rewrite the RHS evaluation into the polarization sum
    have hterm_eval :
        (∑ ε : (Fin d → Bool), term ε) x
          =
        ∑ ε : (Fin d → Bool),
          (signedProd (d := d) ε) *
            (signedSum (d := d) ε (fun i : Fin d => toVec x.1 (idx i))) ^ d := by
      -- evaluate `term` and rewrite the lifted linear form as the signed sum.
      simp [term, linFormC_apply (K := K), dot_wSigned_eq_signedSum (n := n) (idx := idx),
        ]
    -- now cancel the constant `C` using `hpol`
    have : (rhs x) = (∏ i : Fin d, StoneWeierstrass.coord (K := K) (idx i) x) := by
      -- unfold the scalings, use `hpol`, and simplify.
      have hpolC :
          (∑ ε : (Fin d → Bool),
              (signedProd (d := d) ε) *
                (signedSum (d := d) ε (fun i : Fin d => toVec x.1 (idx i))) ^ d)
            = C * (∏ i : Fin d, toVec x.1 (idx i)) := by
        simpa [C, mul_assoc, mul_left_comm, mul_comm] using hpol
      -- compute the coordinate product pointwise
      have hprod :
          (∏ i : Fin d, StoneWeierstrass.coord (K := K) (idx i) x) = ∏ i : Fin d, toVec x.1 (idx i)
            := by
        simp [StoneWeierstrass.coord, ReLUMlpBridge.toVec]
      -- simplify `rhs x`
      -- Keep `C` folded so `simp` can cancel using `hCne`.
      simp [rhs, ContinuousMap.smul_apply, hterm_eval, hprod, hpolC, hCne]
    simpa using this

  -- `rhs` is approximable by closure under sum/scalar-mul, and equality transfers it to the
  -- product.
  have happ_rhs : ApproxOnC (n := n) K rhs := by
    -- First approximate each `term ε`.
    have happ_term : ∀ ε : (Fin d → Bool), ApproxOnC (n := n) K (term ε) := by
      intro ε
      -- approximate the `d`-th power of the corresponding linear form, then scale by `signedProd
      -- ε`.
      have hp : ApproxOnC (n := n) K ((linFormC K (wSigned (n := n) idx ε)) ^ d) :=
        approx_pow_linFormC (K := K) (w := wSigned (n := n) idx ε) d
      simpa [term] using (ApproxOnC.smul (n := n) (K := K) (c := signedProd (d := d) ε) (f := _) hp)
    -- sum the terms, then scale by `1/C`.
    have happ_sum : ApproxOnC (n := n) K (∑ ε : (Fin d → Bool), term ε) :=
      ApproxOnC.sum_fintype (n := n) (K := K) (f := term) happ_term
    simpa [rhs] using (ApproxOnC.smul (n := n) (K := K) (c := (1 / C)) (f := _) happ_sum)

  -- finish
  simpa [hrhs_eq] using happ_rhs

-- Generalize `approx_coordProd_fin` to any finite index type by reindexing along an equivalence to
-- `Fin (Fintype.card ι)`. This avoids the missing `Fintype (ι → Bool)` instance noted elsewhere.
/--
Uniform approximation of a coordinate-product over an arbitrary finite index type.

This is a reindexed form of `approx_coordProd_fin`, using an equivalence `ι ≃ Fin d`.
-/
theorem approx_coordProd {ι : Type} [Fintype ι] (idx : ι → Fin n) :
    ApproxOnC (n := n) K (∏ i : ι, StoneWeierstrass.coord (K := K) (idx i)) := by
  classical
  let d : Nat := Fintype.card ι
  let e : ι ≃ Fin d := Fintype.equivFin ι
  have happ :
      ApproxOnC (n := n) K (∏ j : Fin d, StoneWeierstrass.coord (K := K) (idx (e.symm j))) := by
    simpa using (approx_coordProd_fin (K := K) (n := n) (d := d) (idx := fun j => idx (e.symm j)))
  have hprod :
      (∏ i : ι, StoneWeierstrass.coord (K := K) (idx i))
        =
      (∏ j : Fin d, StoneWeierstrass.coord (K := K) (idx (e.symm j))) := by
    -- `Fintype.prod_equiv` is the cleanest way to reindex the product.
    refine Fintype.prod_equiv e
      (fun i : ι => StoneWeierstrass.coord (K := K) (idx i))
      (fun j : Fin d => StoneWeierstrass.coord (K := K) (idx (e.symm j))) ?_
    intro i
    simp
  simpa [hprod] using happ

end ReLUStoneWeierstrassBridgeProducts

-- ---------------------------------------------------------------------------
-- Polynomials in coordinates are approximable by 2-layer ReLU MLPs (compact-set, nD)
-- ---------------------------------------------------------------------------

section ReLUStoneWeierstrassBridgePolynomials

open NN.MLTheory.Proofs.UniversalApproximationND.StoneWeierstrass
open scoped BigOperators
open ContinuousMap

variable {n : Nat} (K : Set (ReLUMlpBridge.TensorVec n)) [CompactSpace K]

-- `∏ x : m, f (x : α)` (as a fintype product over the multiset coerced to a type) is exactly the
-- multiset product of `m.map f`.
/-- Re-express a fintype product over a multiset as the corresponding multiset product. -/
lemma prod_over_multiset_eq_multiset_prod {α β : Type} [DecidableEq α] [CommMonoid β]
    (m : Multiset α) (f : α → β) :
    (∏ x : m, f (x : α)) = (m.map f).prod := by
  classical
  -- Expand the `Fintype` product as a `Finset.univ` product, then rewrite as a multiset product.
  -- Finally, use the `Multiset.map_univ` lemma that characterizes the underlying multiset of
  -- `univ`.
  simp [Finset.prod_eq_multiset_prod]

-- A `Finsupp` exponent-vector product can be seen as a product over the corresponding multiset of
-- indices (with repetition).
/--
Re-express a `Finsupp` exponent-vector product as a product over `toMultiset`.

This is a small bookkeeping lemma: `d.prod (fun a n => (g a)^n)` is the same as multiplying `g a`
once for each occurrence of `a` in the multiset `d.toMultiset`.
-/
lemma finsupp_prod_pow_eq_prod_toMultiset {α β : Type} [DecidableEq α] [CommMonoid β]
    (d : α →₀ ℕ) (g : α → β) :
    (d.prod fun a n => (g a) ^ n) = ∏ x : d.toMultiset, g (x : α) := by
  classical
  -- Rewrite the RHS as a multiset product.
  have hRHS :
      (d.toMultiset.map g).prod = ∏ x : d.toMultiset, g (x : α) := by
    simpa using (prod_over_multiset_eq_multiset_prod (m := d.toMultiset) (f := g)).symm
  -- Compute the multiset product by `Finsupp` induction.
  have hmapProd : (d.toMultiset.map g).prod = d.prod fun a n => (g a) ^ n := by
    refine d.induction ?_ ?_
    · simp
    · intro a n d ha hn ih
      -- LHS: adding a `single a n` adds `n` copies of `a` to the multiset.
      have hL :
          ((Finsupp.toMultiset (Finsupp.single a n + d)).map g).prod
            =
          (g a) ^ n * ((Finsupp.toMultiset d).map g).prod := by
        simp [Finsupp.toMultiset_add, Finsupp.toMultiset_single, Multiset.map_add,
          Multiset.map_nsmul,
          Multiset.prod_add, Multiset.prod_nsmul]
      -- RHS: `Finsupp.prod` turns `+` into `*` for `pow` (via `pow_add`).
      have hR :
          (Finsupp.single a n + d).prod (fun a n => (g a) ^ n)
            =
          (g a) ^ n * d.prod (fun a n => (g a) ^ n) := by
        classical
        -- split the product over `single a n + d` into `single` and `d`
        calc
          (Finsupp.single a n + d).prod (fun a n => (g a) ^ n)
              =
            (Finsupp.single a n).prod (fun a n => (g a) ^ n) * d.prod (fun a n => (g a) ^ n) := by
              simpa using
                (Finsupp.prod_add_index'
                  (f := Finsupp.single a n) (g := d)
                  (h := fun a n => (g a) ^ n)
                  (by intro a; simp)
                  (by intro a b₁ b₂; simp [pow_add]))
          _ = (g a) ^ n * d.prod (fun a n => (g a) ^ n) := by
              simp [Finsupp.prod_single_index]
      -- Combine and rewrite with the induction hypothesis.
      calc
        ((Finsupp.toMultiset (Finsupp.single a n + d)).map g).prod
            = (g a) ^ n * ((Finsupp.toMultiset d).map g).prod := hL
        _ = (g a) ^ n * d.prod (fun a n => (g a) ^ n) := by simp [ih]
        _ = (Finsupp.single a n + d).prod (fun a n => (g a) ^ n) := by simp [hR]
  -- Finish by rewriting with `hRHS`.
  calc
    d.prod (fun a n => (g a) ^ n) = (d.toMultiset.map g).prod := by simpa using hmapProd.symm
    _ = ∏ x : d.toMultiset, g (x : α) := hRHS

-- Each monomial in the coordinates is approximable (in the `C(K,ℝ)` sense).
/-- Uniform approximation for an evaluated coordinate monomial `aeval (monomial d r)` on `K`. -/
theorem approx_aeval_coord_monomial (d : (Fin n) →₀ ℕ) (r : ℝ) :
    ApproxOnC (n := n) K
      (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) (MvPolynomial.monomial d r)) := by
  classical
  -- Approximate the repeated-coordinate product coming from `d.toMultiset`.
  have hprod :
      ApproxOnC (n := n) K
        (∏ x : d.toMultiset, StoneWeierstrass.coord (K := K) (x : Fin n)) :=
    approx_coordProd (K := K) (n := n) (idx := fun x : d.toMultiset => (x : Fin n))
  -- Rewrite the monomial evaluation into a scalar multiple of the repeated-coordinate product.
  have hrewrite :
      (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) (MvPolynomial.monomial d r))
        =
      r • (∏ x : d.toMultiset, StoneWeierstrass.coord (K := K) (x : Fin n)) := by
    -- `aeval_monomial` gives `algebraMap r * d.prod (...)`; turn that into a scalar action,
    -- then rewrite the `Finsupp.prod` as a product over `d.toMultiset`.
    have hpow :
        d.prod (fun i k => (StoneWeierstrass.coord (K := K) i) ^ k)
          =
        ∏ x : d.toMultiset, StoneWeierstrass.coord (K := K) (x : Fin n) := by
      simpa using
        (finsupp_prod_pow_eq_prod_toMultiset (d := d)
          (g := fun i : Fin n => StoneWeierstrass.coord (K := K) i))
    calc
      MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) (MvPolynomial.monomial d r)
          =
        (algebraMap ℝ C(K, ℝ)) r *
          d.prod (fun i k => (StoneWeierstrass.coord (K := K) i) ^ k) := by
            simpa using (MvPolynomial.aeval_monomial (g := StoneWeierstrass.coord (K := K)) (d := d)
              (r := r))
      _ =
        (algebraMap ℝ C(K, ℝ)) r *
          (∏ x : d.toMultiset, StoneWeierstrass.coord (K := K) (x : Fin n)) := by
            simp [hpow]
      _ = r • (∏ x : d.toMultiset, StoneWeierstrass.coord (K := K) (x : Fin n)) := by
            simp [Algebra.smul_def]
  -- Finish via closure under scalar multiplication.
  simpa [hrewrite] using (ApproxOnC.smul (n := n) (K := K) (c := r) (f := _) hprod)

-- Polynomials in the coordinates are approximable (compact set, nD).
/-- Uniform approximation of a coordinate polynomial `aeval coord p` on a compact set `K`. -/
theorem approx_aeval_coord (p : MvPolynomial (Fin n) ℝ) :
    ApproxOnC (n := n) K (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) := by
  classical
  -- Expand `p` as a finite sum of monomials over its support.
  -- Then use closure under finite sums and scalar multiplication.
  -- `p.as_sum : p = ∑ d ∈ p.support, monomial d (p.coeff d)`.
  -- `aeval_sum` pushes `aeval` through this finite sum.
  have hdecomp :
      MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p
        =
      ∑ d ∈ p.support,
        MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) (MvPolynomial.monomial d (p.coeff d))
          := by
    -- Rewrite `p` using `p.as_sum`, then push `aeval` through the finite sum.
    calc
      MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p
          =
        MvPolynomial.aeval (StoneWeierstrass.coord (K := K))
          (∑ d ∈ p.support, MvPolynomial.monomial d (p.coeff d)) := by
            simp
      _ =
        ∑ d ∈ p.support,
          MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) (MvPolynomial.monomial d (p.coeff d))
            := by
            simpa using (MvPolynomial.aeval_sum (f := StoneWeierstrass.coord (K := K))
              (s := p.support) (φ := fun d => MvPolynomial.monomial d (p.coeff d)))
  -- Approximate each monomial term, then sum.
  have hterm :
      ∀ d ∈ p.support,
        ApproxOnC (n := n) K
          (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) (MvPolynomial.monomial d (p.coeff
            d))) := by
    intro d hd
    simpa using approx_aeval_coord_monomial (K := K) (n := n) d (p.coeff d)
  -- Use `ApproxOnC.sum_finset` on the finite support.
  have hsum :
      ApproxOnC (n := n) K
        (∑ d ∈ p.support,
          MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) (MvPolynomial.monomial d (p.coeff
            d))) :=
    ApproxOnC.sum_finset (n := n) (K := K) (s := p.support)
      (f := fun d => MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) (MvPolynomial.monomial d
        (p.coeff d))) hterm
  simpa [hdecomp] using hsum

end ReLUStoneWeierstrassBridgePolynomials

-- ---------------------------------------------------------------------------
-- Full compact-set nD approximation for 2-layer ReLU MLPs (via Stone–Weierstrass)
-- ---------------------------------------------------------------------------

section ReLUStoneWeierstrassBridgeFull

open NN.MLTheory.Proofs.UniversalApproximationND.StoneWeierstrass
open scoped BigOperators
open ContinuousMap

variable {n : Nat} (K : Set (ReLUMlpBridge.TensorVec n)) [CompactSpace K]

-- ---------------------------------------------------------------------------
-- Bridge theorem: coordinate Stone–Weierstrass subalgebra ⊆ ReLU uniform closure (as `ApproxOnC`)
-- ---------------------------------------------------------------------------

/--
Bridge lemma: elements of the Stone–Weierstrass coordinate subalgebra are `ApproxOnC`-approximable.

This packages the facts that:
- `coordSubalg` is the range of `MvPolynomial.aeval coord`, and
- coordinate polynomials are approximable by ReLU MLPs (previous section).
-/
theorem approxOnC_of_mem_coordSubalg {g : C(K, ℝ)}
    (hg : g ∈ StoneWeierstrass.coordSubalg (K := K)) :
    ApproxOnC (n := n) K g := by
  classical
  have hgmem :
      (g : C(K, ℝ)) ∈ (MvPolynomial.aeval (R := ℝ) (StoneWeierstrass.coord (K := K))).range := by
    simpa [coordSubalg_eq_range_aeval (K := K)] using hg
  rcases hgmem with ⟨p, hp⟩
  have hp' : (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) = g := by
    simpa using hp
  simpa [hp'] using (approx_aeval_coord (K := K) (n := n) p)

/--
ReLU universal approximation on compact sets (nD).

For compact `K` and any continuous `f : C(K,ℝ)`, `f` is uniformly approximable on `K` by a
single-hidden-layer ReLU MLP, in the `ApproxOnC` sense.
-/
theorem relu_universal_approximation_compact (f : C(K, ℝ)) :
    ApproxOnC (n := n) K f := by
  classical
  intro ε hε
  have hε2 : 0 < ε / 2 := by nlinarith
  -- Step 1: Stone–Weierstrass gives a coordinate-subalgebra element `g` close to `f` in sup norm.
  rcases StoneWeierstrass.exists_coordSubalg_near_continuousMap (K := K) f (ε / 2) hε2 with ⟨g, hg⟩
  -- Step 2: `coordSubalg` is the range of `MvPolynomial.aeval coord`, so `g` is a coordinate
  -- polynomial.
  have hgmem :
      (g : C(K, ℝ)) ∈ (MvPolynomial.aeval (R := ℝ) (StoneWeierstrass.coord (K := K))).range := by
    -- rewrite membership along `coordSubalg_eq_range_aeval`
    have : (g : C(K, ℝ)) ∈ StoneWeierstrass.coordSubalg (K := K) :=
      g.property
    -- `coordSubalg_eq_range_aeval` lives earlier in this file.
    simpa [coordSubalg_eq_range_aeval (K := K)] using this
  rcases hgmem with ⟨p, hp⟩
  -- Step 3: approximate that polynomial by a 2-layer ReLU MLP.
  have happ_poly :
      ApproxOnC (n := n) K (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) :=
    approx_aeval_coord (K := K) (n := n) p
  rcases happ_poly (ε / 2) hε2 with ⟨hidDim, l1, l2, hnet⟩
  refine ⟨hidDim, l1, l2, ?_⟩
  intro x
  -- Use triangle inequality: `f - net = (f - g) + (g - net)`.
  -- The first term is controlled by the sup norm bound `hg`; the second by `hnet`.
  have hgf : |(MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x - f x| < ε / 2 := by
    -- pointwise bound from the sup norm
    have hle : |((MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p - f) x)| ≤
        ‖MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p - f‖ := by
      simpa using (ContinuousMap.norm_coe_le_norm (MvPolynomial.aeval (StoneWeierstrass.coord (K :=
        K)) p - f) x)
    -- rewrite `g` as `aeval p` and use `hg`
    have hg' : ‖(MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) - f‖ < ε / 2 := by
      -- `hp : aeval coord p = g`, so use it to rewrite `hg` in the right direction.
      have hp' : (g : C(K, ℝ)) = MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p := by
        simpa using hp.symm
      simpa [hp'] using hg
    -- simplify the pointwise expression
    have : |(MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x - f x| < ε / 2 := by
      -- `((aeval p - f) x) = aeval p x - f x`
      have hle' : |(MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x - f x|
          ≤ ‖MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p - f‖ := by
        simpa using hle
      exact lt_of_le_of_lt hle' hg'
    exact this
  have hnet' : |(MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x -
        mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1| < ε / 2 := hnet x
  -- Combine the two `< ε/2` bounds.
  have htri :
      |f x - mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1|
        ≤
      |f x - (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x|
        +
      |(MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x -
          mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1| := by
    -- `f - net = (f - poly) + (poly - net)`
    have hdecomp : f x - mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1
        =
      (f x - (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x)
        +
      ((MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x -
        mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1) := by
      ring
    have habs :
        |f x - mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1|
          =
        |(f x - (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x)
            +
          ((MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x -
            mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1)| := by
      simp
    -- Now apply `abs_add`.
    calc
      |f x - mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1|
          =
        |(f x - (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x)
            +
          ((MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x -
            mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1)| := habs
      _ ≤
        |f x - (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x|
          +
        |(MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x -
            mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1| := by
          simpa using
            (abs_add_le
              (f x - (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x)
              ((MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x -
                mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1))
  have hsum : |f x - (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x|
        + |(MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x -
            mlpEvalNd (n := n) (hidDim := hidDim) l1 l2 x.1| < ε := by
    -- `|f - poly| = |poly - f|` and both pieces are `< ε/2`.
    have hfg : |f x - (MvPolynomial.aeval (StoneWeierstrass.coord (K := K)) p) x| < ε / 2 := by
      simpa [abs_sub_comm] using hgf
    linarith [hfg, hnet']
  exact lt_of_le_of_lt htri hsum

end ReLUStoneWeierstrassBridgeFull

/-! ## Two compatibility forms for two-dimensional multiplication -/

/--
Compatibility theorem for the standalone 2D multiplication construction from `ReLUMulApprox`.

The n-dimensional development below subsumes this result, but keeping this theorem name gives
downstream files a stable import point for the classical two-coordinate multiplication statement.
-/
theorem relu_mul_universal_approximation_box2d
    {M : ℝ} (hM : 0 < M) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : LinearSpec ℝ 2 hidDim) (l2 : LinearSpec ℝ hidDim 1),
      ∀ x ∈ ReLUMulApprox.box M,
        |ReLUMulApprox.mulFun x - mlpEvalNd (n := 2) (hidDim := hidDim) l1 l2 x| < ε := by
  simpa using (ReLUMulApprox.relu_mul_universal_approximation_box (M := M) hM)

lemma boxN_two_iff_box (M : ℝ) (x : ReLUMulApprox.TensorVec2) :
    x ∈ boxN 2 M ↔ x ∈ ReLUMulApprox.box M := by
  constructor
  · intro hx
    refine And.intro ?_ ?_
    · have := hx (0 : Fin 2)
      simpa [ReLUMulApprox.box, ReLUMulApprox.x0] using this
    · have := hx (1 : Fin 2)
      simpa [ReLUMulApprox.box, ReLUMulApprox.x1] using this
  · intro hx
    -- Convert a two-coordinate box proof into the corresponding pair of interval facts.
    dsimp [boxN]
    refine (Fin.forall_fin_two).2 ?_
    refine And.intro ?_ ?_
    · simpa [ReLUMulApprox.box, ReLUMulApprox.x0] using hx.1
    · simpa [ReLUMulApprox.box, ReLUMulApprox.x1] using hx.2

/--
The same 2D multiplication guarantee derived from the nD coordinate-product theorem.

This theorem is a cross-check between the specialized two-dimensional construction and the general
coordinate-product approximation pipeline used by the compact-set theorem.
-/
theorem relu_mul_universal_approximation_box2d_via_nd
    {M : ℝ} (hM : 0 < M) :
    ∀ ε > 0, ∃ (hidDim : ℕ) (l1 : LinearSpec ℝ 2 hidDim) (l2 : LinearSpec ℝ hidDim 1),
      ∀ x ∈ ReLUMulApprox.box M,
        |ReLUMulApprox.mulFun x - mlpEvalNd (n := 2) (hidDim := hidDim) l1 l2 x| < ε := by
  intro ε hε
  rcases relu_mul_coord_universal_approximation_box (n := 2) (M := M) hM (0 : Fin 2) (1 : Fin 2) ε
    hε with
    ⟨hidDim, l1, l2, h⟩
  refine ⟨hidDim, l1, l2, ?_⟩
  intro x hx
  have hxN : x ∈ boxN 2 M := (boxN_two_iff_box (M := M) (x := x)).2 hx
  simpa [ReLUMulApprox.mulFun, ReLUMulApprox.x0, ReLUMulApprox.x1] using h x hxN

end NN.MLTheory.Proofs.ReLU.Approximation.CompactSet
