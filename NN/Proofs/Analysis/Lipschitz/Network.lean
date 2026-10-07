/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Lipschitz.Norm
public import NN.Spec.Layers.Activation
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

open Spec (dot tensorNormSquared tensor_norm_squared_nonneg)

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

/--
Pointwise ReLU is 1-Lipschitz for scalars.
Foundation for tensor-level Lipschitz bounds.
-/
theorem relu_scalar_lipschitz (x y : ℝ) :
  |max (0 : ℝ) x - max (0 : ℝ) y| ≤ |x - y| := by
  simpa only [max_comm (0 : ℝ)] using abs_max_sub_max_le_abs x y 0

private theorem relu_squared_difference_le (x y : ℝ) :
    (Math.reluSpec x - Math.reluSpec y) * (Math.reluSpec x - Math.reluSpec y) ≤
      (x - y) * (x - y) := by
  have hAbs : |Math.reluSpec x - Math.reluSpec y| ≤ |x - y| := by
    simpa [Math.reluSpec_eq_max, max_comm] using relu_scalar_lipschitz x y
  have hSquared :=
    mul_self_le_mul_self (abs_nonneg (Math.reluSpec x - Math.reluSpec y)) hAbs
  simpa [sq_abs, pow_two] using hSquared

/-- ReLU is 1-Lipschitz in the L2 norm for tensors of every shape. -/
theorem relu_lipschitz_general {s : Shape} (x y : Tensor ℝ s) :
    tensorL2Dist (reluSpec x) (reluSpec y) ≤ tensorL2Dist x y := by
  unfold tensorL2Dist tensorL2Norm tensorNormSquared dot
  apply Real.sqrt_le_sqrt
  rw [sum_spec_eq_coord_sum, sum_spec_eq_coord_sum]
  apply Finset.sum_le_sum
  intro coordinate _
  simpa [reluSpec, Tensor.map, subSpec, mulSpec] using
    relu_squared_difference_le (x coordinate) (y coordinate)

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
  rw [subSpec_eq_sub, addSpec_eq_add, scaleSpec_eq_smul, neg_one_smul, sub_eq_add_neg]

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
Each coordinate of `matVecMulSpec W x` is the dot product of the corresponding matrix row with
`x`.

This is the coordinate bridge used by the Frobenius/operator-norm bound below.
-/
private theorem mat_vec_coord_eq_dot_row {m n : Nat}
    (W : Tensor ℝ [m, n])
    (x : Tensor ℝ [n]) (i : Fin m) :
    getScalar (matVecMulSpec W x) i = dot (get W i) x := by
  rw [getScalar_mat_vec_mul_spec, dot_vec_eq_sum]
  rfl

/-- The Frobenius norm bounds matrix-vector multiplication in the Euclidean norm. -/
theorem matVec_norm_le_frobenius {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x : Tensor ℝ [n]) :
  tensorL2Norm (matVecMulSpec W x) ≤ matrixFrobeniusNorm W * tensorL2Norm x := by
  classical
  have hsum_nonneg : 0 ≤ ∑ i : Fin m, tensorNormSquared (get W i) :=
    Finset.sum_nonneg fun i _ => tensor_norm_squared_nonneg (get W i)
  -- Each coordinate of `W x` is a row inner product, so squared Cauchy–Schwarz bounds its square
  -- by `‖rowᵢ‖² ‖x‖²`; summing over the rows gives the squared Frobenius bound.
  have hterm : ∀ i : Fin m,
      getScalar (matVecMulSpec W x) i * getScalar (matVecMulSpec W x) i ≤
        tensorNormSquared (get W i) * tensorNormSquared x := by
    intro i
    rw [mat_vec_coord_eq_dot_row]
    simp only [tensorNormSquared, dot_eq_inner]
    exact real_inner_mul_inner_self_le _ _
  have hsquared :
      tensorNormSquared (matVecMulSpec W x) ≤
        (∑ i : Fin m, tensorNormSquared (get W i)) * tensorNormSquared x := by
    calc
      tensorNormSquared (matVecMulSpec W x)
          = ∑ i : Fin m, getScalar (matVecMulSpec W x) i * getScalar (matVecMulSpec W x) i := by
            rw [tensorNormSquared, dot_vec_eq_sum]
      _ ≤ ∑ i : Fin m, tensorNormSquared (get W i) * tensorNormSquared x :=
            Finset.sum_le_sum fun i _ => hterm i
      _ = (∑ i : Fin m, tensorNormSquared (get W i)) * tensorNormSquared x :=
            (Finset.sum_mul _ _ _).symm
  unfold matrixFrobeniusNorm tensorL2Norm
  rw [← Real.sqrt_mul hsum_nonneg]
  exact Real.sqrt_le_sqrt hsquared

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

The nonnegative `Lg` preserves the inequality when multiplying the bound for `f`.
No sign assumption on `Lf` is needed.
-/
theorem lipschitz_composition {s t u : Shape}
  (f : Tensor ℝ s → Tensor ℝ t) (g : Tensor ℝ t → Tensor ℝ u)
  (Lf Lg : ℝ)
  (hf : ∀ x y, tensorL2Dist (f x) (f y) ≤ Lf * tensorL2Dist x y)
  (hg : ∀ x y, tensorL2Dist (g x) (g y) ≤ Lg * tensorL2Dist x y)
  (hLg : 0 ≤ Lg)
  (x y : Tensor ℝ s) :
  tensorL2Dist (g (f x)) (g (f y)) ≤ (Lg * Lf) * tensorL2Dist x y := by
  calc
    tensorL2Dist (g (f x)) (g (f y)) ≤ Lg * tensorL2Dist (f x) (f y) := hg (f x) (f y)
    _ ≤ Lg * (Lf * tensorL2Dist x y) := mul_le_mul_of_nonneg_left (hf x y) hLg
    _ = (Lg * Lf) * tensorL2Dist x y := (mul_assoc _ _ _).symm

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

end Proofs
