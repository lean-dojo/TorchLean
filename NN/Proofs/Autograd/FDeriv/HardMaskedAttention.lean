/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.HardMaskedSoftmax
public import NN.Proofs.Autograd.FDeriv.Reindex
import NN.Proofs.Models.Attention.HardMask

/-!
# Rectangular attention with a Boolean mask

Query and key lengths are independent. The mask is fixed, and every row may have its own allowed
keys, including no allowed keys at all. The statements below concern the actual attention JVP and
backward specifications, with their existing scale and zero-row conventions.

The pairing theorem identifies the transpose relation between these two implementations. The
analytic derivative of the row normalization comes from `HardMaskedSoftmax`; identifying the
whole attention JVP with the derivative of the forward map additionally requires composing the
matrix and row maps.

The compound softmax–multiplier–value operation also has a joint pairing theorem for arbitrary
batch prefixes and an analytic output-coordinate derivative. Its multiplier may vary alongside
scores and values; the Boolean mask remains fixed.
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
      Spec.hardMaskedSoftmaxVecSpec (scores.unstack i) (mask.unstack i) :=
  Tensor.unstack_dim _ i

/-- The weighted-centering helper acts independently on each query row. -/
theorem unstack_softmaxBackwardFromWeightsSpec {nQ nK : Nat}
    (weights gradient : Tensor ℝ [nQ, nK]) (i : Fin nQ) (center : Bool := false) :
    (Spec.softmaxBackwardFromWeightsSpec weights gradient (center := center)).unstack i =
      Spec.softmaxBackwardFromWeightsSpec (weights.unstack i) (gradient.unstack i)
        (center := center) := by
  cases nQ with
  | zero => exact Fin.elim0 i
  | succ nQ =>
      cases nK <;> exact Tensor.unstack_dim _ i

/-- The concrete masked row helper is symmetric for rectangular score matrices.

There is no requirement that a row contain an allowed key. The all-false case is already part of
the row derivative theorem, so summing its inner-product identity over query rows preserves it. -/
theorem dot_backward_comm {nQ nK : Nat} (scores dx gradient : Tensor ℝ [nQ, nK])
    (mask : Tensor Bool [nQ, nK]) (center : Bool := false) :
    Spec.dot (Spec.softmaxBackwardFromWeightsSpec
      (Spec.hardMaskedSoftmaxSpec scores mask) dx (center := center)) gradient =
        Spec.dot dx (Spec.softmaxBackwardFromWeightsSpec
          (Spec.hardMaskedSoftmaxSpec scores mask) gradient (center := center)) := by
  rw [Spec.dot_eq_inner, Spec.dot_eq_inner,
    Reindex.inner_eq_sum_unstack, Reindex.inner_eq_sum_unstack]
  apply Finset.sum_congr rfl
  intro i _
  simp only [unstack_softmaxBackwardFromWeightsSpec (center := center),
    unstack_hardMaskedSoftmaxSpec]
  rw [← Spec.dot_eq_inner, ← Spec.dot_eq_inner,
    dot_eq_inner_vec, dot_eq_inner_vec,
    HardMaskedSoftmax.backward_eq_derivative (center := center),
    HardMaskedSoftmax.backward_eq_derivative (center := center)]
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

/-- Real contractions retain the ordinary matrix expression, with either scaling schedule. -/
private theorem scaled_product_reference {m n p : Nat}
    (a : Tensor ℝ [m, n]) (b : Tensor ℝ [n, p]) (c : ℝ) (scaleFirst : Bool) :
    Spec.Internal.scaledProducts (fun _ : Fin 1 => a) (fun _ => b) c scaleFirst =
      if scaleFirst then matMulSpec (scaleSpec a c) b else scaleSpec (matMulSpec a b) c := by
  have unsupported : Numeric.QuotientArithmetic.supported (α := ℝ) = false := rfl
  cases scaleFirst <;> simp [Spec.Internal.scaledProducts, unsupported]

private theorem scaled_pair_reference {m n p : Nat}
    (a da : Tensor ℝ [m, n]) (b db : Tensor ℝ [n, p]) (c : ℝ) :
    Spec.Internal.scaledProducts (fun i : Fin 2 => if i.val == 0 then da else a)
      (fun i => if i.val == 0 then b else db) c =
        scaleSpec (addSpec (matMulSpec da b) (matMulSpec a db)) c := by
  have unsupported : Numeric.QuotientArithmetic.supported (α := ℝ) = false := rfl
  simp [Spec.Internal.scaledProducts, unsupported, List.finRange_succ]

private theorem scale_transpose {m n : Nat} (a : Tensor ℝ [m, n]) (c : ℝ) :
    scaleSpec (swapAdjacentAxes a 0) c = swapAdjacentAxes (scaleSpec a c) 0 := by
  apply Spec.matrix_ext
  intro i j
  simp only [Spec.get2_scaleSpec, Spec.get2_matrix_transpose_spec]

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
  let scoreGradient :=
    scaleSpec (Spec.softmaxBackwardFromWeightsSpec weights dWeights (center := true)) c
  have unsupported : Numeric.QuotientArithmetic.supported (α := ℝ) = false := rfl
  simp only [Spec.scaledDotProductAttentionJvp, Spec.scaledDotProductAttentionBackward, hmask,
    Spec.Internal.pullbackProducts, unsupported, Bool.not_false,
    scaled_product_reference, scaled_pair_reference, ↓reduceIte, scale_transpose]
  change
    Spec.dot (addSpec
      (matMulSpec (Spec.softmaxBackwardFromWeightsSpec weights (scaleSpec dScores c)
        (center := true)) ctx.V)
      (matMulSpec weights dV)) gradient =
        Spec.dot dQ (matMulSpec scoreGradient ctx.K) +
          (Spec.dot dK (matMulSpec (swapAdjacentAxes scoreGradient 0) ctx.Q) +
            Spec.dot dV (matMulSpec (swapAdjacentAxes weights 0) gradient))
  rw [Spec.dot_add_left, Spec.dot_mat_mul_right_adjoint,
    Spec.dot_mat_mul_left_adjoint]
  rw [dot_backward_comm scores (scaleSpec dScores c) dWeights mask (center := true),
    dot_scale_comm]
  change
    Spec.dot dScores scoreGradient +
        Spec.dot dV (matMulSpec (swapAdjacentAxes weights 0) gradient) = _
  rw [Spec.dot_add_left, Spec.dot_mat_mul_right_adjoint,
    Spec.matrix_transpose_involution, dot_key_product, add_assoc]

private theorem dot_mul_assoc {shape : Shape} (x y z : Tensor ℝ shape) :
    Spec.dot (mulSpec x y) z = Spec.dot x (mulSpec y z) := by
  unfold Spec.dot
  simp only [Spec.mul_spec_assoc]

/-- An absent mask is the fixed all-true mask over real scores. -/
private theorem softmaxMatmulWeights_real {rows keys : Nat}
    (scores : Tensor ℝ [rows, keys]) (mask : Option (Tensor Bool [rows, keys])) :
    Spec.softmaxMatmulWeights (batch := .scalar) scores mask =
      Spec.hardMaskedSoftmaxSpec scores (mask.getD (Spec.allTrueMask rows keys)) := by
  cases mask with
  | some mask => rfl
  | none =>
    change Activation.Internal.softmaxInnermostSpec scores =
      Spec.hardMaskedSoftmaxSpec scores (Spec.allTrueMask rows keys)
    rw [NN.Proofs.Models.Attention.hardMaskedSoftmaxSpec_allTrueMask]
    exact (Proofs.softmaxSpec_one_matrix scores).symm

/-- The compound normalized-product rules are adjoint, including a differentiable multiplier.

The optional Boolean mask is fixed; an absent mask allows every key. No row needs an allowed key:
the zero-row derivative is included in `HardMaskedSoftmax`, and batch prefixes simply sum
independent Frobenius pairings.
-/
theorem softmaxMatmul_jvp_backward_pairing {batch : Shape} {rows keys width : Nat}
    (scores deltaScores multiplier deltaMultiplier : Tensor ℝ (batch.concat [rows, keys]))
    (value deltaValue : Tensor ℝ (batch.concat [keys, width]))
    (gradient : Tensor ℝ (batch.concat [rows, width]))
    (mask : Option (Tensor Bool [rows, keys])) :
    let weights := Spec.softmaxMatmulWeights (batch := batch) scores mask
    let backward := Spec.softmaxMatmulBackward (batch := batch) weights value multiplier gradient
    Spec.dot (Spec.softmaxMatmulJvp (batch := batch) weights value multiplier deltaScores
        deltaValue deltaMultiplier)
        gradient =
      Spec.dot deltaScores backward.1 +
        (Spec.dot deltaValue backward.2.1 + Spec.dot deltaMultiplier backward.2.2) := by
  induction batch with
  | scalar =>
    let fixedMask := mask.getD (Spec.allTrueMask rows keys)
    let weights := Spec.hardMaskedSoftmaxSpec scores fixedMask
    let upstream := matMulSpec gradient (swapAdjacentAxes value 0)
    have unsupported : Numeric.QuotientArithmetic.supported (α := ℝ) = false := rfl
    simp only [softmaxMatmulWeights_real, Spec.softmaxMatmulBackward, Spec.softmaxMatmulJvp,
      unsupported, Bool.not_false, ↓reduceIte, scaled_product_reference,
      scaled_pair_reference, Spec.scaleSpec_one]
    change Spec.dot (addSpec
        (matMulSpec (addSpec
          (mulSpec (Spec.softmaxBackwardFromWeightsSpec weights deltaScores (center := true))
            multiplier)
          (mulSpec weights deltaMultiplier)) value)
        (matMulSpec (mulSpec weights multiplier) deltaValue)) gradient =
      Spec.dot deltaScores (Spec.softmaxBackwardFromWeightsSpec weights
          (mulSpec upstream multiplier) (center := true)) +
        (Spec.dot deltaValue (matMulSpec (swapAdjacentAxes (mulSpec weights multiplier) 0)
          gradient) + Spec.dot deltaMultiplier (mulSpec weights upstream))
    rw [Spec.dot_add_left, Spec.dot_mat_mul_right_adjoint, Spec.dot_mat_mul_left_adjoint,
      Spec.dot_add_left, dot_mul_assoc, Spec.mul_spec_comm multiplier upstream,
      dot_backward_comm scores deltaScores (mulSpec upstream multiplier) fixedMask (center := true),
      Spec.mul_spec_comm weights deltaMultiplier, dot_mul_assoc]
    ac_rfl
  | dim n batch ih =>
    simp only [Spec.softmaxMatmulWeights, Spec.softmaxMatmulBackward, Spec.softmaxMatmulJvp]
    simp only [Spec.dot_eq_inner, Reindex.inner_eq_sum_unstack, Tensor.unstack_dim]
    simp_rw [← Spec.dot_eq_inner]
    rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => ih (scores.unstack i) (deltaScores.unstack i)
      (multiplier.unstack i) (deltaMultiplier.unstack i) (value.unstack i)
      (deltaValue.unstack i) (gradient.unstack i)

/-- Every output coordinate differentiates by the actual compound JVP.

Scores, values and multipliers may vary simultaneously along an arbitrary real parameter. The
optional mask stays fixed, including rows with no allowed key. Row normalization uses the analytic
theorem for the stable masked-softmax implementation; only scalar products and finite sums remain.
-/
theorem hasDerivAt_softmaxMatmul_get2 {rows keys width : Nat}
    (scores multiplier : ℝ → Tensor ℝ [rows, keys])
    (value : ℝ → Tensor ℝ [keys, width])
    (deltaScores deltaMultiplier : Tensor ℝ [rows, keys])
    (deltaValue : Tensor ℝ [keys, width])
    (mask : Option (Tensor Bool [rows, keys])) (point : ℝ)
    (hs : ∀ i : Fin rows, HasDerivAt (fun t => getScalarE ((scores t).unstack i))
      (getScalarE (deltaScores.unstack i)) point)
    (hm : ∀ (i : Fin rows) (k : Fin keys), HasDerivAt (fun t => get2 (multiplier t) i k)
      (get2 deltaMultiplier i k) point)
    (hv : ∀ (k : Fin keys) (j : Fin width), HasDerivAt (fun t => get2 (value t) k j)
      (get2 deltaValue k j) point) (row : Fin rows) (column : Fin width) :
    HasDerivAt
      (fun t => get2 (Spec.softmaxMatmul (batch := .scalar)
        (scores t) (value t) (multiplier t) mask)
        row column)
      (get2 (Spec.softmaxMatmulJvp (batch := .scalar)
        (Spec.softmaxMatmulWeights (batch := .scalar) (scores point) mask) (value point)
        (multiplier point) deltaScores deltaValue deltaMultiplier) row column) point := by
  let fixedMask := mask.getD (Spec.allTrueMask rows keys)
  let weights := fun t => Spec.hardMaskedSoftmaxSpec (scores t) fixedMask
  let deltaWeights := Spec.softmaxBackwardFromWeightsSpec (weights point) deltaScores
    (center := true)
  have hw (key : Fin keys) : HasDerivAt (fun t => get2 (weights t) row key)
      (get2 deltaWeights row key) point := by
    have hrow := (HardMaskedSoftmax.hasFDerivAt_spec (fixedMask.unstack row)
      (getScalarE ((scores point).unstack row))).comp_hasDerivAt point (hs row)
    have hcoordinate := (evalCLM key).hasFDerivAt.comp_hasDerivAt point hrow
    have hresult := congrArg (fun v : Vec keys => v key)
      (HardMaskedSoftmax.backward_eq_derivative ((scores point).unstack row)
        (deltaScores.unstack row) (fixedMask.unstack row) (center := true))
    simpa only [weights, deltaWeights, Function.comp_def, evalCLM_apply, ofFnE_getScalarE,
      unstack_hardMaskedSoftmaxSpec, unstack_softmaxBackwardFromWeightsSpec,
      Spec.get2_eq_getScalar_get, Spec.get, getScalarE_ofLp] using
        hcoordinate.congr_deriv hresult.symm
  have hsum := HasDerivAt.fun_sum (u := (Finset.univ : Finset (Fin keys)))
    (fun key _ => ((hw key).mul (hm row key)).mul (hv key column))
  have unsupported : Numeric.QuotientArithmetic.supported (α := ℝ) = false := rfl
  simp only [Spec.softmaxMatmul, Spec.Internal.softmaxMatmulFromWeights,
    softmaxMatmulWeights_real, Spec.softmaxMatmulJvp,
    unsupported, Bool.not_false, ↓reduceIte, scaled_pair_reference, Spec.scaleSpec_one]
  simp only [Spec.get2_mat_mul_spec, Spec.get2_addSpec]
  simpa only [Spec.get2_eq_apply, mulSpec, TorchLean.Tensor.map2Spec_apply,
    Pi.mul_apply, Finset.sum_add_distrib,
    weights, deltaWeights, fixedMask] using hsum

end

end Proofs.Autograd.HardMaskedAttention
