/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Lipschitz.TensorNorm
public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Algebra.Real
public import TorchLean.Tensor.Contract

/-!
# Lipschitz bounds for neural-network operations

We prove Lipschitz bounds for ReLU, matrix-vector multiplication, linear layers and their
composition. Distances and norms are Mathlib's `dist` and `‖·‖` on real tensors
(`TorchLean.Tensor.Euclidean`), and the bound for a matrix `W` is its Frobenius norm `‖W‖`.
It bounds the operator norm from above and can exceed it: the `n × n` identity has `‖W‖ = √n` and
operator norm `1`.

The main results are `l2Dist_relu_le`, `l2Norm_matVec_le`, `l2Dist_matVec_le`,
`l2Dist_comp_le` and `l2Dist_relu_matVec_le`.
-/

@[expose] public section

namespace TorchLean

open Tensor
open scoped BigOperators

open Rep (dot)

private theorem Activation.relu_squared_difference_le (x y : ℝ) :
    (relu x - relu y) * (relu x - relu y) ≤
      (x - y) * (x - y) := by
  have hAbs : |relu x - relu y| ≤ |x - y| := by
    simpa [relu_eq_max, max_comm] using abs_max_sub_max_le_abs x y 0
  have hSquared :=
    mul_self_le_mul_self (abs_nonneg (relu x - relu y)) hAbs
  simpa [sq_abs, pow_two] using hSquared

/-- ReLU is 1-Lipschitz in the L2 norm for tensors of every shape. -/
theorem Tensor.l2Dist_relu_le {s : Shape} (x y : Tensor ℝ s) :
    dist (Tensor.relu x) (Tensor.relu y) ≤ dist x y := by
  rw [Tensor.dist_eq_sqrt_sum, Tensor.dist_eq_sqrt_sum]
  apply Real.sqrt_le_sqrt
  apply Finset.sum_le_sum
  intro coordinate _
  simpa [Tensor.relu, sq] using
    Activation.relu_squared_difference_le (x coordinate) (y coordinate)

/-- ReLU is `1`-Lipschitz for the Euclidean metric on real tensors. -/
theorem Tensor.lipschitzWith_relu {s : Shape} :
    LipschitzWith 1 (Tensor.relu : Tensor ℝ s → Tensor ℝ s) :=
  LipschitzWith.of_dist_le_mul fun x y => by
    simpa only [NNReal.coe_one, one_mul] using l2Dist_relu_le x y

-- Operator-norm bounds for matrix-vector multiplication.

/--
Tensor subtraction is addition of the subtrahend scaled by `-1`.

Linear-operator proofs rewrite with it so that the lemmas for addition and scaling cover
subtraction too.
-/
theorem Tensor.sub_eq_add_scale_neg_one {s : Shape} (a b : Tensor ℝ s) :
  a - b = a + Tensor.scale b (-1 : ℝ) := by
  rw [scale_eq_smul, neg_one_smul, sub_eq_add_neg]

/--
Matrix-vector multiplication sends the zero vector to the zero vector, because each output
coordinate is a sum of products that each contain a zero input coordinate.
-/
theorem Tensor.contract_matVec_zero {m n : Nat} (W : Tensor ℝ [m, n]) :
    Tensor.contract W (Tensor.full (.dim n .scalar) (0 : ℝ)) _ _ [m] .matVec =
      Tensor.full (.dim m .scalar) (0 : ℝ) := by
  classical
  apply Tensor.ext_vector
  intro i
  simp [getScalar_contract_matVec_eq_sum]

/--
Coordinate `i` of the matrix-vector contraction of `W` and `x` is the dot product of row `i` of
`W` with `x`. Cauchy-Schwarz then bounds each output coordinate in `Tensor.l2Norm_matVec_le`.
-/
private theorem Tensor.getScalar_matVec_eq_dot {m n : Nat}
    (W : Tensor ℝ [m, n])
    (x : Tensor ℝ [n]) (i : Fin m) :
    getScalar (Tensor.contract W x _ _ [m] .matVec) i = Rep.dot (Tensor.unstack W i) x := by
  rw [getScalar_contract_matVec_eq_sum, dot_vec_eq_sum]
  rfl

/-- The Frobenius norm bounds matrix-vector multiplication in the Euclidean norm. -/
theorem Tensor.l2Norm_matVec_le {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x : Tensor ℝ [n]) :
  ‖Tensor.contract W x _ _ [m] .matVec‖ ≤ ‖W‖ * ‖x‖ := by
  classical
  -- Each coordinate of `W x` is a row inner product, so squared Cauchy-Schwarz bounds its square
  -- by `‖rowᵢ‖² ‖x‖²`. Summing over the rows gives `‖W‖² ‖x‖²`.
  have hterm : ∀ i : Fin m,
      getScalar (Tensor.contract W x _ _ [m] .matVec) i * getScalar
          (Tensor.contract W x _ _ [m] .matVec) i ≤
        ‖Tensor.unstack W i‖ ^ 2 * ‖x‖ ^ 2 := by
    intro i
    rw [getScalar_matVec_eq_dot, dot_eq_inner, ← real_inner_self_eq_norm_sq,
      ← real_inner_self_eq_norm_sq]
    exact real_inner_mul_inner_self_le _ _
  have hsquared : ‖Tensor.contract W x _ _ [m] .matVec‖ ^ 2 ≤ (‖W‖ * ‖x‖) ^ 2 := by
    calc
      ‖Tensor.contract W x _ _ [m] .matVec‖ ^ 2
          = ∑ i : Fin m, getScalar (Tensor.contract W x _ _ [m] .matVec) i * getScalar
              (Tensor.contract W x _ _ [m] .matVec) i := by
            rw [← real_inner_self_eq_norm_sq, ← dot_eq_inner, dot_vec_eq_sum]
      _ ≤ ∑ i : Fin m, ‖Tensor.unstack W i‖ ^ 2 * ‖x‖ ^ 2 :=
            Finset.sum_le_sum fun i _ => hterm i
      _ = (‖W‖ * ‖x‖) ^ 2 := by
            rw [mul_pow, ← Finset.sum_mul, Tensor.norm_sq_eq_sum_unstack W]
  exact le_of_sq_le_sq hsquared (mul_nonneg (norm_nonneg _) (norm_nonneg _))

/-- Multiplication by `W` is Lipschitz with constant `‖W‖`, the Frobenius norm of `W`. -/
theorem Tensor.l2Dist_matVec_le {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x y : Tensor ℝ [n]) :
  dist (Tensor.contract W x _ _ [m] .matVec) (Tensor.contract W y _ _ [m] .matVec) ≤
  ‖W‖ * dist x y := by
  have h_linear : Tensor.contract W (x - y) _ _ [m] .matVec =
    Tensor.contract W x _ _ [m] .matVec - Tensor.contract W y _ _ [m] .matVec := by
    -- With `x - y` written as `x + scale y (-1)`, linearity follows from `contract_matVec_add`
    -- and `contract_matVec_scale`.
    rw [sub_eq_add_scale_neg_one (a := x) (b := y)]
    rw [contract_matVec_add]
    rw [contract_matVec_scale]
    simp [sub_eq_add_scale_neg_one]

  rw [dist_eq_norm, dist_eq_norm, ← h_linear]
  exact l2Norm_matVec_le W (x - y)

-- Composition theorems for building network-level Lipschitz bounds.

/-- Composing two distance bounds multiplies their constants, given `0 ≤ Lg` for the outer map `g`.
No sign assumption on `Lf` is needed. -/
theorem Tensor.l2Dist_comp_le {s t u : Shape}
  (f : Tensor ℝ s → Tensor ℝ t) (g : Tensor ℝ t → Tensor ℝ u)
  (Lf Lg : ℝ)
  (hf : ∀ x y, dist (f x) (f y) ≤ Lf * dist x y)
  (hg : ∀ x y, dist (g x) (g y) ≤ Lg * dist x y)
  (hLg : 0 ≤ Lg)
  (x y : Tensor ℝ s) :
  dist (g (f x)) (g (f y)) ≤ (Lg * Lf) * dist x y := by
  calc
    dist (g (f x)) (g (f y)) ≤ Lg * dist (f x) (f y) := hg (f x) (f y)
    _ ≤ Lg * (Lf * dist x y) := mul_le_mul_of_nonneg_left (hf x y) hLg
    _ = (Lg * Lf) * dist x y := (mul_assoc _ _ _).symm

/-- ReLU after multiplication by `W` keeps the Lipschitz constant `‖W‖`, since ReLU itself is
1-Lipschitz. -/
theorem Tensor.l2Dist_relu_matVec_le {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x y : Tensor ℝ [n]) :
  dist (Tensor.relu (Tensor.contract W x _ _ [m] .matVec)) (Tensor.relu
      (Tensor.contract W y _ _ [m] .matVec)) ≤
  ‖W‖ * dist x y :=
  (l2Dist_relu_le _ _).trans (l2Dist_matVec_le W x y)

end TorchLean
