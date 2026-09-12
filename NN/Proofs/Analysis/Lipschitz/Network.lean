/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Lipschitz.Norm
public import NN.Spec.Layers.Activation
import Mathlib.Tactic.Positivity.Finset
public import NN.Proofs.Tensor.Basic.Algebra

/-!
# Lipschitz bounds for neural-network operations

This module owns Lipschitz estimates for ReLU, matrix-vector multiplication, linear layers, and
composition. The underlying real-valued tensor norm API lives in
`NN.Proofs.Analysis.Lipschitz.Norm`.

Import `NN.Proofs.Analysis.Lipschitz` for the complete norm and network-bound API.
-/

@[expose] public section

namespace Proofs

open Spec _root_.TorchLean
open _root_.TorchLean _root_.TorchLean.Tensor
open Activation
open scoped BigOperators

open Spec (dot tensorNormSquared tensor_norm_squared_nonneg
           tensor_norm_squared_zero_iff mul_spec_comm add_spec_comm dot_comm
           sum_spec_add_distrib mul_spec_add_left mul_spec_add_right
           add_spec_assoc)

/-! ## Bridges between `tensorL2Dist` bounds and Mathlib's `LipschitzWith` -/

/-- A `tensorL2Dist` bound with a nonnegative constant is a Mathlib Lipschitz bound. -/
theorem lipschitzWith_of_tensorL2Dist_le {s t : Shape} {f : Tensor ℝ s → Tensor ℝ t} {L : ℝ}
    (hL : 0 ≤ L) (h : ∀ x y, tensorL2Dist (f x) (f y) ≤ L * tensorL2Dist x y) :
    LipschitzWith ⟨L, hL⟩ f :=
  LipschitzWith.of_dist_le_mul fun x y => by
    rw [← tensorL2Dist_eq_dist, ← tensorL2Dist_eq_dist]
    exact h x y

/-- A Mathlib Lipschitz bound on real tensors is a `tensorL2Dist` bound. -/
theorem _root_.LipschitzWith.tensorL2Dist_le {s t : Shape} {f : Tensor ℝ s → Tensor ℝ t}
    {K : NNReal} (hf : LipschitzWith K f) (x y : Tensor ℝ s) :
    tensorL2Dist (f x) (f y) ≤ K * tensorL2Dist x y := by
  simpa only [tensorL2Dist_eq_dist] using hf.dist_le_mul x y

-- ====================================================================
-- RELU LIPSCHITZ CONTINUITY PROOFS
-- ===================================================================

/--
Pointwise ReLU is 1-Lipschitz for scalars.
Foundation for tensor-level Lipschitz bounds.
-/
theorem relu_scalar_lipschitz (x y : ℝ) :
  |max (0 : ℝ) x - max (0 : ℝ) y| ≤ |x - y| := by
  -- ReLU is 1-Lipschitz: |max(0,x) - max(0,y)| ≤ |x - y|
  -- This follows from case analysis on the signs of x and y
  -- We'll consider four cases based on the signs of x and y
  by_cases hx : (0 : ℝ) ≤ x
  · by_cases hy : (0 : ℝ) ≤ y
    · -- Case 1: x ≥ 0 and y ≥ 0, so max 0 x = x and max 0 y = y
      simp [max_eq_right hx, max_eq_right hy]
    · -- Case 2: x ≥ 0 and y < 0, so max 0 x = x and max 0 y = 0
      push Not at hy
      simp [max_eq_right hx, max_eq_left (le_of_lt hy)]
      -- Need to show |x - 0| ≤ |x - y|
      -- Since x ≥ 0 and y < 0, we have x - y > x
      have h : (0 : ℝ) ≤ x - y := by linarith
      have hx_pos : (0 : ℝ) ≤ x := hx
      rw [abs_of_nonneg hx_pos, abs_of_nonneg h]
      simp
      linarith
  · push Not at hx
    by_cases hy : (0 : ℝ) ≤ y
    · -- Case 3: x < 0 and y ≥ 0, so max 0 x = 0 and max 0 y = y
      simp [max_eq_left (le_of_lt hx), max_eq_right hy]
      -- Need to show |0 - y| ≤ |x - y|
      -- Since x < 0 and y ≥ 0, we have |x - y| ≥ y
      have h : x - y ≤ (0 : ℝ) := by linarith
      rw [abs_of_nonneg hy, abs_of_nonpos h]
      simp
      linarith
    · -- Case 4: x < 0 and y < 0, so max 0 x = 0 and max 0 y = 0
      push Not at hy
      simp [max_eq_left (le_of_lt hx), max_eq_left (le_of_lt hy)]


private theorem relu_squared_difference_le (x y : ℝ) :
    (Math.reluSpec x - Math.reluSpec y) * (Math.reluSpec x - Math.reluSpec y) ≤
      (x - y) * (x - y) := by
  have hAbs : |Math.reluSpec x - Math.reluSpec y| ≤ |x - y| := by
    simpa [Math.reluSpec_eq_max, max_comm] using relu_scalar_lipschitz x y
  have hSquared :=
    mul_self_le_mul_self (abs_nonneg (Math.reluSpec x - Math.reluSpec y)) hAbs
  simpa [sq_abs, pow_two] using hSquared

private theorem relu_lipschitz_packed {shape : Shape}
    (x y : Tensor ℝ shape) :
    tensorL2Dist (reluSpec x) (reluSpec y) ≤ tensorL2Dist x y := by
  unfold tensorL2Dist tensorL2Norm tensorNormSquared dot
  apply Real.sqrt_le_sqrt
  rw [sum_spec_eq_coord_sum, sum_spec_eq_coord_sum]
  apply Finset.sum_le_sum
  intro coordinate _
  simpa [reluSpec, mapSpec, subSpec, mulSpec, Tensor.map] using
    relu_squared_difference_le (x coordinate) (y coordinate)

/-- ReLU is 1-Lipschitz on scalar tensors. -/
theorem relu_scalar_tensor_lipschitz (x y : Tensor ℝ .scalar) :
    tensorL2Dist (reluSpec x) (reluSpec y) ≤ tensorL2Dist x y :=
  relu_lipschitz_packed x y

/-- ReLU is 1-Lipschitz in the L2 norm for tensors of every shape. -/
theorem relu_lipschitz_general {s : Shape} (x y : Tensor ℝ s) :
    tensorL2Dist (reluSpec x) (reluSpec y) ≤ tensorL2Dist x y :=
  relu_lipschitz_packed x y

/--
Rank-one ReLU is 1-Lipschitz in $\ell_2$.

This theorem is just the vector specialization of `relu_lipschitz_general`, but it is convenient
for callers working with ordinary `.dim n .scalar` activations.
-/
theorem relu_vector_lipschitz {n : Nat} (x y : Tensor ℝ [n]) :
  tensorL2Dist (reluSpec x) (reluSpec y) ≤ tensorL2Dist x y := by
  simpa using (relu_lipschitz_general (s := .dim n .scalar) x y)

/-- ReLU is `1`-Lipschitz for the Euclidean metric on real tensors. -/
theorem reluSpec_lipschitzWith {s : Shape} :
    LipschitzWith 1 (reluSpec : Tensor ℝ s → Tensor ℝ s) :=
  LipschitzWith.of_dist_le_mul fun x y => by
    simpa only [tensorL2Dist_eq_dist, NNReal.coe_one, one_mul] using relu_lipschitz_general x y

-- Linear-operator norm bounds for affine layers and matrix products.

/--
Tensor subtraction can be rewritten as addition of a `-1` scale.

This is a small algebraic normal form used by linear-operator proofs, where it is often easier to
reuse additive and scaling lemmas than reason about `subSpec` directly.
-/
theorem sub_spec_eq_add_scale_neg_one {s : Shape} (a b : Tensor ℝ s) :
  subSpec a b = addSpec a (scaleSpec b (-1 : ℝ)) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [subSpec, addSpec, scaleSpec, map2Spec, mapSpec, Tensor.map]
  ring

/-- Subtracting the zero tensor on the right leaves the tensor unchanged. -/
theorem sub_spec_zero_right {s : Shape} (t : Tensor ℝ s) :
  subSpec t (Tensor.full s (0 : ℝ)) = t := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [subSpec, map2Spec, Tensor.full]

/--
Matrix-vector multiplication sends the zero vector to the zero vector.

The proof follows the spec definition: each output coordinate is a fold over scalar products, and
every scalar product contains a zero input coordinate.
-/
theorem mat_vec_mul_spec_zero {m n : Nat} (W : Tensor ℝ [m, n]) :
    matVecMulSpec W (Tensor.full (.dim n .scalar) (0 : ℝ)) =
      Tensor.full (.dim m .scalar) (0 : ℝ) := by
  classical
  apply Tensor.ext_vector
  intro i
  simp [getScalar_mat_vec_mul_spec]

/--
Frobenius norm of a matrix tensor.

We use the Frobenius-norm-style bound:

$$
\lVert W\rVert_F
=\sqrt{\sum_i\lVert\operatorname{row}_i(W)\rVert_2^2},
$$

This is an upper bound for the induced Euclidean operator norm; it is not the spectral norm.
-/
noncomputable def matrixFrobeniusNorm {m n : Nat} (W : Tensor ℝ [m, n]) : ℝ :=
  Real.sqrt (∑ i : Fin m, tensorNormSquared (get W i))

/--
Compatibility between two row/column access views:

`getScalar (get W i) j` and `get2 W i j` name the same scalar entry of a matrix tensor.
-/
private theorem getScalar_get_eq_get2 {m n : Nat}
    (W : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    getScalar (get W i) j = get2 W i j := by
  rfl

/--
Each coordinate of `matVecMulSpec W x` is the dot product of the corresponding matrix row with
`x`.

This is the coordinate bridge used by the Frobenius/operator-norm bound below.
-/
private theorem mat_vec_coord_eq_dot_row {m n : Nat}
    (W : Tensor ℝ [m, n])
    (x : Tensor ℝ [n]) (i : Fin m) :
    getScalar (matVecMulSpec W x) i = dot (get W i) x := by
  classical
  -- Expand both sides as `Finset.univ` sums and match terms.
  rw [getScalar_mat_vec_mul_spec (A := W) (v := x) (i := i)]
  rw [dot_vec_eq_sum (a := get W i) (b := x)]
  refine Finset.sum_congr rfl ?_
  intro j _
  simp [getScalar_get_eq_get2 (W := W) (i := i) (j := j)]

/-- The Frobenius norm bounds matrix-vector multiplication in the Euclidean norm. -/
theorem matVec_norm_le_frobenius {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x : Tensor ℝ [n]) :
  tensorL2Norm (matVecMulSpec W x) ≤ matrixFrobeniusNorm W * tensorL2Norm x := by
  classical
  -- Work with the squared form and then apply `Real.sqrt`.
  have hsum_nonneg : 0 ≤ ∑ i : Fin m, tensorNormSquared (get W i) := by
    have : 0 ≤ ∑ i ∈ (Finset.univ : Finset (Fin m)), tensorNormSquared (get W i) := by
      refine Finset.sum_nonneg ?_
      intro i _
      exact tensor_norm_squared_nonneg (tensor := get W i)
    simpa using this

  have hsquared :
      tensorNormSquared (matVecMulSpec W x) ≤
        (∑ i : Fin m, tensorNormSquared (get W i)) * tensorNormSquared x := by
    -- Expand `‖W x‖²` as a sum of squared coordinates.
    have hnormsq :
        tensorNormSquared (matVecMulSpec W x) =
          ∑ i : Fin m, (getScalar (matVecMulSpec W x) i) * (getScalar (matVecMulSpec W x) i) := by
      simpa [tensorNormSquared] using
        (dot_vec_eq_sum (a := matVecMulSpec W x) (b := matVecMulSpec W x))

    -- Bound each coordinate via Cauchy–Schwarz on the corresponding row.
    have hterm :
        ∀ i : Fin m,
          (getScalar (matVecMulSpec W x) i) * (getScalar (matVecMulSpec W x) i) ≤
            tensorNormSquared (get W i) * tensorNormSquared x := by
      intro i
      have hcoord : getScalar (matVecMulSpec W x) i = dot (get W i) x :=
        mat_vec_coord_eq_dot_row (W := W) (x := x) (i := i)
      have cs :
          |dot (get W i) x| ≤ tensorL2Norm (get W i) * tensorL2Norm x :=
        tensor_cauchy_schwarz (x := get W i) (y := x)
      have cs2 :
          (dot (get W i) x) ^ 2 ≤ (tensorL2Norm (get W i) * tensorL2Norm x) ^ 2 := by
        -- Square both sides of `cs` via `mul_le_mul`.
        have hmul :
            |dot (get W i) x| * |dot (get W i) x| ≤
              (tensorL2Norm (get W i) * tensorL2Norm x) *
                (tensorL2Norm (get W i) * tensorL2Norm x) := by
          refine mul_le_mul cs cs (abs_nonneg (dot (get W i) x)) ?_
          exact mul_nonneg (tensor_l2_norm_nonneg (get W i)) (tensor_l2_norm_nonneg x)
        have hsq :
            (|dot (get W i) x|) ^ 2 ≤ (tensorL2Norm (get W i) * tensorL2Norm x) ^ 2 := by
          simpa [pow_two] using hmul
        simpa [sq_abs] using hsq

      have row_sq : (tensorL2Norm (get W i)) ^ 2 = tensorNormSquared (get W i) := by
        unfold tensorL2Norm
        simp [Real.sq_sqrt (tensor_norm_squared_nonneg (tensor := get W i))]
      have x_sq : (tensorL2Norm x) ^ 2 = tensorNormSquared x := by
        unfold tensorL2Norm
        simp [Real.sq_sqrt (tensor_norm_squared_nonneg (tensor := x))]
      have rhs_sq :
          (tensorL2Norm (get W i) * tensorL2Norm x) ^ 2 =
            tensorNormSquared (get W i) * tensorNormSquared x := by
        -- `(a*b)^2 = a^2 * b^2`, then unfold the squares of the norms.
        simp [mul_pow, row_sq, x_sq]

      have hsq :
          (getScalar (matVecMulSpec W x) i) ^ 2 ≤
            tensorNormSquared (get W i) * tensorNormSquared x := by
        -- Replace the coordinate by the row dot-product and use `cs2`.
        simpa [hcoord, rhs_sq] using cs2

      -- Convert `a^2` back into `a*a`.
      simpa [pow_two] using hsq

    -- Sum the coordinate-wise bounds and factor out `‖x‖²`.
    have hsum_le :
        (∑ i : Fin m,
              (getScalar (matVecMulSpec W x) i) * (getScalar (matVecMulSpec W x) i)) ≤
          ∑ i : Fin m, tensorNormSquared (get W i) * tensorNormSquared x := by
      have :
          (∑ i ∈ (Finset.univ : Finset (Fin m)),
                (getScalar (matVecMulSpec W x) i) * (getScalar (matVecMulSpec W x) i)) ≤
            ∑ i ∈ (Finset.univ : Finset (Fin m)), tensorNormSquared (get W i) *
              tensorNormSquared x := by
        refine Finset.sum_le_sum ?_
        intro i _
        exact hterm i
      simpa using this

    have hfactor :
        (∑ i : Fin m, tensorNormSquared (get W i) * tensorNormSquared x) =
          (∑ i : Fin m, tensorNormSquared (get W i)) * tensorNormSquared x := by
      have h :=
        (Finset.sum_mul (s := (Finset.univ : Finset (Fin m)))
          (f := fun i : Fin m => tensorNormSquared (get W i)) (a := tensorNormSquared x))
      simpa using h.symm

    -- Put everything together.
    calc
      tensorNormSquared (matVecMulSpec W x)
          = ∑ i : Fin m,
              (getScalar (matVecMulSpec W x) i) * (getScalar (matVecMulSpec W x) i) := hnormsq
      _ ≤ ∑ i : Fin m, tensorNormSquared (get W i) * tensorNormSquared x := hsum_le
      _ = (∑ i : Fin m, tensorNormSquared (get W i)) * tensorNormSquared x := hfactor

  -- Take square roots and rewrite the RHS using `Real.sqrt_mul`.
  unfold matrixFrobeniusNorm tensorL2Norm
  have hsqrt := Real.sqrt_le_sqrt hsquared
  -- Rewrite `√(A * B)` as `√A * √B` with `A = ∑ i, ‖row_i‖² ≥ 0`.
  rw [Real.sqrt_mul hsum_nonneg (tensorNormSquared x)] at hsqrt
  simpa using hsqrt

/--
Linear transformations preserve $\ell_2$-norm bounds.
Fundamental theorem for neural network stability analysis.
-/
theorem linear_op_norm_bound {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x y : Tensor ℝ [n]) :
  tensorL2Dist (matVecMulSpec W x) (matVecMulSpec W y) ≤
  matrixFrobeniusNorm W * tensorL2Dist x y := by
  have h_linear : matVecMulSpec W (subSpec x y) =
    subSpec (matVecMulSpec W x) (matVecMulSpec W y) := by
    -- Express subtraction as addition with scaling, then use `mat_vec_add`/`mat_vec_scale`.
    rw [sub_spec_eq_add_scale_neg_one (a := x) (b := y)]
    rw [Spec.mat_vec_add]
    rw [Spec.mat_vec_scale]
    -- Rewrite the RHS subtraction similarly.
    simp [sub_spec_eq_add_scale_neg_one]

  unfold tensorL2Dist
  rw [← h_linear]
  exact matVec_norm_le_frobenius W (subSpec x y)

-- Composition theorems for building network-level Lipschitz bounds.

/--
Composition of Lipschitz functions preserves Lipschitz property.
Essential for analyzing deep neural networks.

This is `LipschitzWith.comp` read through `tensorL2Dist_eq_dist`. A negative `Lf` is degenerate:
the hypothesis on `f` then forces `x = y`, and both sides vanish.
-/
theorem lipschitz_composition {s t u : Shape}
  (f : Tensor ℝ s → Tensor ℝ t) (g : Tensor ℝ t → Tensor ℝ u)
  (Lf Lg : ℝ)
  (hf : ∀ x y, tensorL2Dist (f x) (f y) ≤ Lf * tensorL2Dist x y)
  (hg : ∀ x y, tensorL2Dist (g x) (g y) ≤ Lg * tensorL2Dist x y)
  (hLg : 0 ≤ Lg)
  (x y : Tensor ℝ s) :
  tensorL2Dist (g (f x)) (g (f y)) ≤ (Lg * Lf) * tensorL2Dist x y := by
  rcases le_or_gt 0 Lf with hLf | hLf
  · have h :=
      ((lipschitzWith_of_tensorL2Dist_le hLg hg).comp
        (lipschitzWith_of_tensorL2Dist_le hLf hf)).tensorL2Dist_le x y
    exact h
  · have hxy : x = y := by
      have h0 : 0 ≤ Lf * tensorL2Dist x y :=
        le_trans (by rw [tensorL2Dist_eq_dist]; exact dist_nonneg) (hf x y)
      have hle : tensorL2Dist x y ≤ 0 := by
        by_contra hpos
        have hpos' : 0 < tensorL2Dist x y := lt_of_not_ge hpos
        exact absurd h0 (not_le.mpr (mul_neg_of_neg_of_pos hLf hpos'))
      rw [tensorL2Dist_eq_dist] at hle
      exact dist_le_zero.mp hle
    subst hxy
    simp only [tensorL2Dist_eq_dist, dist_self, mul_zero, le_refl]

/--
ReLU + Linear composition Lipschitz bound.
Practical theorem for single neural network layer analysis.
-/
theorem relu_linear_lipschitz {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x y : Tensor ℝ [n]) :
  tensorL2Dist (reluSpec (matVecMulSpec W x)) (reluSpec (matVecMulSpec W y)) ≤
  matrixFrobeniusNorm W * tensorL2Dist x y := by
  calc tensorL2Dist (reluSpec (matVecMulSpec W x)) (reluSpec (matVecMulSpec W y))
    ≤ tensorL2Dist (matVecMulSpec W x) (matVecMulSpec W y)  :=
      relu_lipschitz_general (matVecMulSpec W x) (matVecMulSpec W y)
    _ ≤ matrixFrobeniusNorm W * tensorL2Dist x y                        :=
      linear_op_norm_bound W x y

-- ====================================================================
-- SPECIALIZED ACTIVATION FUNCTION ANALYSIS
-- ====================================================================

end Proofs
