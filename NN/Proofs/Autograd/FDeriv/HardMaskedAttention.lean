/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.HardMaskedSoftmax
public import NN.Proofs.Autograd.FDeriv.Reindex
public import NN.Proofs.Tensor.Basic.BoundsNorms

/-!
# Rectangular attention with a Boolean mask

Query and key lengths are independent. The mask is fixed, and every row may have its own allowed
keys, including no allowed keys at all. The statements below concern the actual attention JVP and
backward specifications, with their existing scale and zero-row conventions.

The pairing theorem identifies the transpose relation between these two implementations. The
analytic derivative of the row normalization comes from `HardMaskedSoftmax`; identifying the
whole attention JVP with the derivative of the forward map additionally requires composing the
matrix and row maps.
-/

@[expose] public section

namespace Proofs.Autograd.HardMaskedAttention

open Spec TorchLean
open TorchLean.Tensor
open scoped BigOperators

noncomputable section

/-- Each query row is normalized against its own row of the Boolean mask. -/
theorem unstack_hardMaskedSoftmaxSpec {nQ nK : Nat}
    (scores : Tensor ℝ [nQ, nK]) (mask : Tensor Bool [nQ, nK]) (i : Fin nQ) :
    (Spec.hardMaskedSoftmaxSpec scores mask).unstack i =
      Spec.hardMaskedSoftmaxVecSpec (scores.unstack i) (mask.unstack i) := by
  exact Tensor.unstack_dim _ i

/-- The weighted-centering helper acts independently on each query row. -/
theorem unstack_softmaxBackwardFromWeightsSpec {nQ nK : Nat}
    (weights gradient : Tensor ℝ [nQ, nK]) (i : Fin nQ) :
    (Spec.softmaxBackwardFromWeightsSpec weights gradient).unstack i =
      Spec.softmaxBackwardFromWeightsSpec (weights.unstack i) (gradient.unstack i) := by
  exact Tensor.unstack_dim _ i

/-- The concrete masked row helper is symmetric for rectangular score matrices.

There is no requirement that a row contain an allowed key. The all-false case is already part of
the row derivative theorem, so summing its inner-product identity over query rows preserves it. -/
theorem dot_backward_comm {nQ nK : Nat} (scores dx gradient : Tensor ℝ [nQ, nK])
    (mask : Tensor Bool [nQ, nK]) :
    Spec.dot (Spec.softmaxBackwardFromWeightsSpec
      (Spec.hardMaskedSoftmaxSpec scores mask) dx) gradient =
        Spec.dot dx (Spec.softmaxBackwardFromWeightsSpec
          (Spec.hardMaskedSoftmaxSpec scores mask) gradient) := by
  rw [Reindex.dot_eq_inner, Reindex.dot_eq_inner,
    Reindex.inner_eq_sum_unstack, Reindex.inner_eq_sum_unstack]
  apply Finset.sum_congr rfl
  intro i _
  simp only [unstack_softmaxBackwardFromWeightsSpec,
    unstack_hardMaskedSoftmaxSpec]
  rw [← Reindex.dot_eq_inner, ← Reindex.dot_eq_inner,
    dot_eq_inner_vec, dot_eq_inner_vec,
    HardMaskedSoftmax.backward_eq_derivative, HardMaskedSoftmax.backward_eq_derivative]
  exact HardMaskedSoftmax.inner_derivative
    (mask.unstack i) (getScalarE (scores.unstack i))
    (getScalarE (dx.unstack i)) (getScalarE (gradient.unstack i))

/-- A fixed real scale moves across the Frobenius pairing without changing it. -/
private theorem dot_scale_comm {shape : Shape} (x y : Tensor ℝ shape) (c : ℝ) :
    Spec.dot (scaleSpec x c) y = Spec.dot x (scaleSpec y c) := by
  calc
    Spec.dot (scaleSpec x c) y = c * Spec.dot x y := Spec.dot_scale_left x y c
    _ = c * Spec.dot y x := by rw [Spec.dot_comm x y]
    _ = Spec.dot (scaleSpec y c) x := (Spec.dot_scale_left y x c).symm
    _ = Spec.dot x (scaleSpec y c) := Spec.dot_comm _ _

/-- The key tangent enters scores through its transpose, so its cotangent is transposed back. -/
private theorem dot_key_product {nQ nK d : Nat}
    (query : Tensor ℝ [nQ, d]) (keyTangent : Tensor ℝ [nK, d])
    (scoreGradient : Tensor ℝ [nQ, nK]) :
    Spec.dot (matMulSpec query (swapAdjacentAxes keyTangent 0)) scoreGradient =
      Spec.dot keyTangent (matMulSpec (swapAdjacentAxes scoreGradient 0) query) := by
  rw [← Spec.dot_mat_transpose (matMulSpec query (swapAdjacentAxes keyTangent 0))
    scoreGradient]
  rw [Spec.matrix_transpose_mul, Spec.matrix_transpose_involution,
    Spec.dot_mat_mul_right_adjoint, Spec.matrix_transpose_involution]

/-- The spec SDPA JVP and backward formulas have the exact Frobenius transpose relation.

The three terms keep query, key, and value gradients separate. In particular, the key term uses
the rectangular transposition dictated by `Q Kᵀ`; no equality of query and key lengths is assumed.
This algebraic identity is stated separately from differentiability of the forward map. -/
theorem jvp_backward_pairing {nQ nK d : Nat} {hQ : nQ ≠ 0} {hK : nK ≠ 0}
    (ctx : Spec.AttentionContext ℝ nQ nK d hQ hK)
    (mask : Tensor Bool [nQ, nK]) (hmask : ctx.mask = some mask)
    (dQ : Tensor ℝ [nQ, d]) (dK dV : Tensor ℝ [nK, d])
    (gradient : Tensor ℝ [nQ, d]) :
    Spec.dot (Spec.scaledDotProductAttentionJvp ctx dQ dK dV) gradient =
      Spec.dot dQ (Spec.scaledDotProductAttentionBackward ctx gradient).1 +
        (Spec.dot dK (Spec.scaledDotProductAttentionBackward ctx gradient).2.1 +
          Spec.dot dV (Spec.scaledDotProductAttentionBackward ctx gradient).2.2) := by
  let c : ℝ := 1 / Spec.attentionScaleDenom (α := ℝ) d
  let scores := scaleSpec (matMulSpec ctx.Q (swapAdjacentAxes ctx.K 0)) c
  let weights := Spec.hardMaskedSoftmaxSpec scores mask
  let dScores := addSpec (matMulSpec dQ (swapAdjacentAxes ctx.K 0))
    (matMulSpec ctx.Q (swapAdjacentAxes dK 0))
  let dWeights := matMulSpec gradient (swapAdjacentAxes ctx.V 0)
  let scoreGradient := scaleSpec (Spec.softmaxBackwardFromWeightsSpec weights dWeights) c
  simp only [Spec.scaledDotProductAttentionJvp, Spec.scaledDotProductAttentionBackward, hmask]
  change
    Spec.dot (addSpec
      (matMulSpec (Spec.softmaxBackwardFromWeightsSpec weights (scaleSpec dScores c)) ctx.V)
      (matMulSpec weights dV)) gradient =
        Spec.dot dQ (matMulSpec scoreGradient ctx.K) +
          (Spec.dot dK (matMulSpec (swapAdjacentAxes scoreGradient 0) ctx.Q) +
            Spec.dot dV (matMulSpec (swapAdjacentAxes weights 0) gradient))
  rw [Spec.dot_add_left, Spec.dot_mat_mul_right_adjoint,
    Spec.dot_mat_mul_left_adjoint]
  rw [dot_backward_comm scores (scaleSpec dScores c) dWeights mask, dot_scale_comm]
  change
    Spec.dot dScores scoreGradient +
        Spec.dot dV (matMulSpec (swapAdjacentAxes weights 0) gradient) = _
  rw [Spec.dot_add_left, Spec.dot_mat_mul_right_adjoint,
    Spec.matrix_transpose_involution, dot_key_product, add_assoc]

end

end Proofs.Autograd.HardMaskedAttention
