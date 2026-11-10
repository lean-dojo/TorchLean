/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Softmax.HardMask
public import TorchLean.NN.FDeriv.Reindex
import TorchLean.NN.Attention.HardMask

/-!
# Rectangular attention with a Boolean mask

Query and key lengths are independent. The mask is fixed, and each row may have its own allowed
keys, including none at all. All statements concern the attention JVP and backward specifications
as defined, with their scale and zero-row conventions.

`dot_scaledDotProductJVP` gives the transpose relation between these two specifications. The
analytic derivative of the row normalization comes from `HardMaskedSoftmax`, and identifying the
whole attention JVP with the derivative of the forward map requires composing the matrix and row
maps as well.

Attention after its score product, the softmax, multiplier and value operation of
`TorchLean.Attention.values`, also has a joint pairing theorem for arbitrary batch prefixes and
an analytic derivative for each output coordinate. Its multiplier may vary alongside the scores
and values, while the Boolean mask stays fixed.
-/

@[expose] public section

namespace TorchLean

open Tensor
open scoped BigOperators
open Rep (dot)

noncomputable section

/-- Each query row is normalized against its own row of the Boolean mask. -/
theorem Softmax.unstack_hardMasked {nQ nK : Nat}
    (scores : Tensor ℝ [nQ, nK]) (mask : Tensor Bool [nQ, nK]) (i : Fin nQ) :
    (hardMasked scores mask).unstack i =
      hardMaskedVec (scores.unstack i) (mask.unstack i) :=
  Tensor.unstack_dim _ i

/-- The weighted-centering helper acts independently on each query row. -/
theorem Softmax.unstack_backwardFromWeights {nQ nK : Nat}
    (weights gradient : Tensor ℝ [nQ, nK]) (i : Fin nQ) (center : Bool := false) :
    (backwardFromWeights weights gradient (center := center)).unstack i =
      backwardFromWeights (weights.unstack i) (gradient.unstack i)
        (center := center) := by
  cases nQ with
  | zero => exact Fin.elim0 i
  | succ nQ =>
      cases nK <;> exact Tensor.unstack_dim _ i

/-- With masked-softmax weights, the row helper `Softmax.backwardFromWeights` is symmetric
under the Frobenius pairing for rectangular score matrices.

A row need not contain an allowed key. Since the row derivative theorem covers the all-false case,
summing its inner-product identity over the query rows gives the matrix statement. -/
theorem Softmax.dot_backwardFromWeights_comm {nQ nK : Nat}
    (scores dx gradient : Tensor ℝ [nQ, nK])
    (mask : Tensor Bool [nQ, nK]) (center : Bool := false) :
    Rep.dot (backwardFromWeights
      (hardMasked scores mask) dx (center := center)) gradient =
        Rep.dot dx (backwardFromWeights
          (hardMasked scores mask) gradient (center := center)) := by
  rw [Tensor.dot_eq_inner, Tensor.dot_eq_inner,
    Tensor.inner_eq_sum_unstack, Tensor.inner_eq_sum_unstack]
  apply Finset.sum_congr rfl
  intro i _
  simp only [unstack_backwardFromWeights (center := center),
    unstack_hardMasked]
  rw [← Tensor.dot_eq_inner, ← Tensor.dot_eq_inner,
    dot_eq_inner_getScalarE, dot_eq_inner_getScalarE,
    HardMaskedSoftmax.backward_eq_derivative (center := center),
    HardMaskedSoftmax.backward_eq_derivative (center := center)]
  exact HardMaskedSoftmax.inner_derivative
    (mask.unstack i) (getScalarE (scores.unstack i))
    (getScalarE (dx.unstack i)) (getScalarE (gradient.unstack i))

/-- The key tangent enters scores through its transpose, so its cotangent is transposed back. -/
private theorem Attention.dot_key_product {nQ nK d : Nat}
    (query : Tensor ℝ [nQ, d]) (keyTangent : Tensor ℝ [nK, d])
    (scoreGradient : Tensor ℝ [nQ, nK]) :
    Rep.dot (Tensor.matmul query (Tensor.permute keyTangent
        (Shape.adjacentSwap [nK, d] 0) [d, nK])) scoreGradient =
      Rep.dot keyTangent (Tensor.matmul (Tensor.permute scoreGradient
          (Shape.adjacentSwap [nQ, nK] 0) [nK, nQ]) query) := by
  rw [← Tensor.dot_transpose (Tensor.matmul query (Tensor.permute keyTangent
      (Shape.adjacentSwap [nK, d] 0) [d, nK]))
    scoreGradient]
  rw [Tensor.transpose_matmul, Tensor.transpose_transpose,
    Tensor.dot_matmul_right_adjoint, Tensor.transpose_transpose]

private theorem Attention.scale_transpose {m n : Nat} (a : Tensor ℝ [m, n]) (c : ℝ) :
    Tensor.scale (Tensor.permute a (Shape.adjacentSwap [m, n] 0) [n, m]) c =
      Tensor.permute (Tensor.scale a c) (Shape.adjacentSwap [m, n] 0) [n, m] := by
  apply Tensor.ext_get2
  intro i j
  simp only [Tensor.get2_scale, Tensor.get2_transpose]

/-- For an attention context with mask `some mask`, the spec SDPA JVP and backward formulas are
Frobenius transposes: pairing the JVP with `gradient` gives the sum of separate query, key and
value pairings with the backward outputs.

The key term uses the rectangular transposition dictated by `Q Kᵀ`, so query and key lengths may
differ. Differentiability of the forward map is a separate statement from this algebraic
identity. -/
theorem Attention.dot_scaledDotProductJVP {nQ nK d : Nat} {hQ : nQ ≠ 0} {hK : nK ≠ 0}
    (ctx : Inputs ℝ nQ nK d hQ hK)
    (mask : Tensor Bool [nQ, nK]) (hmask : ctx.mask = some mask)
    (dQ : Tensor ℝ [nQ, d]) (dK dV : Tensor ℝ [nK, d])
    (gradient : Tensor ℝ [nQ, d]) :
    Rep.dot (scaledDotProductJVP ctx dQ dK dV) gradient =
      Rep.dot dQ (scaledDotProductBackward ctx gradient).1 +
        (Rep.dot dK (scaledDotProductBackward ctx gradient).2.1 +
          Rep.dot dV (scaledDotProductBackward ctx gradient).2.2) := by
  let c : ℝ := 1 / scaleDenom (α := ℝ) d
  let scores := Tensor.scale (Tensor.matmul ctx.query (Tensor.permute ctx.key
      (Shape.adjacentSwap [nK, d] 0) [d, nK])) c
  let weights := Softmax.hardMasked scores mask
  let dScores := (Tensor.matmul dQ (Tensor.permute ctx.key
      (Shape.adjacentSwap [nK, d] 0) [d, nK])) +
    Tensor.matmul ctx.query (Tensor.permute dK (Shape.adjacentSwap [nK, d] 0) [d, nK])
  let dWeights := Tensor.matmul gradient (Tensor.permute ctx.value
      (Shape.adjacentSwap [nK, d] 0) [d, nK])
  let scoreGradient :=
    Tensor.scale (Softmax.backwardFromWeights weights dWeights (center := true)) c
  simp only [scaledDotProductJVP, scaledDotProductBackward, hmask,
    pullbackProducts, Numeric.QuotientArithmetic.supported_real, Bool.not_false,
    scaledProductsSlice_one Numeric.QuotientArithmetic.supported_real,
    scaledProductsSlice_two Numeric.QuotientArithmetic.supported_real,
    Bool.false_eq_true, ↓reduceIte, scale_transpose]
  change
    Rep.dot ((Tensor.matmul (Softmax.backwardFromWeights weights (Tensor.scale dScores c)
        (center := true)) ctx.value) +
      Tensor.matmul weights dV) gradient =
        Rep.dot dQ (Tensor.matmul scoreGradient ctx.key) +
          (Rep.dot dK (Tensor.matmul (Tensor.permute scoreGradient
              (Shape.adjacentSwap [nQ, nK] 0) [nK, nQ]) ctx.query) +
            Rep.dot dV (Tensor.matmul (Tensor.permute weights
                (Shape.adjacentSwap [nQ, nK] 0) [nK, nQ]) gradient))
  rw [Tensor.dot_add_left, Tensor.dot_matmul_right_adjoint,
    Tensor.dot_matmul_left_adjoint]
  rw [Softmax.dot_backwardFromWeights_comm scores (Tensor.scale dScores c) dWeights mask
    (center := true),
    Tensor.dot_scale_left, ← Tensor.dot_scale_right]
  change
    Rep.dot dScores scoreGradient +
        Rep.dot dV (Tensor.matmul (Tensor.permute weights
            (Shape.adjacentSwap [nQ, nK] 0) [nK, nQ]) gradient) = _
  rw [Tensor.dot_add_left, Tensor.dot_matmul_right_adjoint,
    Tensor.transpose_transpose, dot_key_product, add_assoc]

/-- Over real scores, `Attention.weightsSlice` is the hard-masked softmax, with an
absent mask read as the all-true mask. -/
theorem Attention.weightsSlice_eq_hardMasked {rows keys : Nat}
    (scores : Tensor ℝ [rows, keys]) (mask : Option (Tensor Bool [rows, keys])) :
    weightsSlice scores mask =
      Softmax.hardMasked scores (mask.getD (allTrueMask rows keys)) := by
  cases mask with
  | some mask => rfl
  | none =>
    change Softmax.innermost scores =
      Softmax.hardMasked scores (allTrueMask rows keys)
    rw [Softmax.hardMasked_allTrueMask]
    exact (Tensor.softmax_one_matrix scores).symm

/-- The batch slice of `Attention.weights` is `Attention.weightsSlice` of the batch slice of the
scores. -/
theorem Attention.weights_batchSlice {α : Type} [Storage α] [Context α] {batch : Shape}
    {rows keys : Nat} (scores : Tensor α (batch ++ [rows, keys]))
    (mask : Option (Tensor Bool [rows, keys])) (b : batch.Coord) :
    (Attention.weights scores mask).batchSlice b =
      Attention.weightsSlice (scores.batchSlice b) mask := by
  refine Tensor.Rep.ext fun q => ?_
  rw [Tensor.batchSlice_apply, Attention.weights_apply]
  simp only [Equiv.apply_symm_apply]

/-- `Attention.valuesJVP` and `Attention.valuesBackward` are adjoint over
any batch prefix, with tangents for the scores, the values and the multiplier.

An absent mask allows every key, and a present one stays fixed. No row needs an allowed key,
because `HardMaskedSoftmax` includes the zero-row derivative. Over a batch prefix the pairing is a
sum of independent Frobenius pairings.
-/
theorem Attention.dot_valuesJVP {batch : Shape} {rows keys width : Nat}
    (scores deltaScores multiplier deltaMultiplier : Tensor ℝ (batch ++ [rows, keys]))
    (value deltaValue : Tensor ℝ (batch ++ [keys, width]))
    (gradient : Tensor ℝ (batch ++ [rows, width]))
    (mask : Option (Tensor Bool [rows, keys])) :
    let weights := Attention.weights (batch := batch) scores mask
    let backward := valuesBackward (batch := batch) weights value multiplier
      gradient
    Rep.dot (valuesJVP (batch := batch) weights value multiplier
        deltaScores deltaValue deltaMultiplier)
        gradient =
      Rep.dot deltaScores backward.1 +
        (Rep.dot deltaValue backward.2.1 + Rep.dot deltaMultiplier backward.2.2) := by
  rw [weightsFast_eq,
    valuesBackwardFast_eq,
    valuesJVPFast_eq]
  induction batch with
  | scalar =>
    change Tensor ℝ [rows, keys] at scores deltaScores multiplier deltaMultiplier
    change Tensor ℝ [keys, width] at value deltaValue
    change Tensor ℝ [rows, width] at gradient
    let fixedMask := mask.getD (allTrueMask rows keys)
    let weights := Softmax.hardMasked scores fixedMask
    let upstream := Tensor.matmul gradient (Tensor.permute value
        (Shape.adjacentSwap [keys, width] 0) (Shape.swapAdjacentAtDepth [keys, width] 0))
    simp only [weightsFast, weightsSlice_eq_hardMasked,
      valuesBackwardFast, valuesBackwardSlice,
      valuesJVPFast, valuesJVPSlice,
      Numeric.QuotientArithmetic.supported_real, Bool.not_false, Bool.false_eq_true, ↓reduceIte,
      scaledProductsSlice_one Numeric.QuotientArithmetic.supported_real,
      scaledProductsSlice_two Numeric.QuotientArithmetic.supported_real,
      Tensor.scale_one]
    change Rep.dot ((Tensor.matmul ((Softmax.backwardFromWeights weights deltaScores
        (center := true) *
            multiplier) +
          (weights * deltaMultiplier)) value) +
        Tensor.matmul (weights * multiplier) deltaValue) gradient =
      Rep.dot deltaScores (Softmax.backwardFromWeights weights
          (upstream * multiplier) (center := true)) +
        (Rep.dot deltaValue (Tensor.matmul (Tensor.permute (weights * multiplier)
            (Shape.adjacentSwap [rows, keys] 0) (Shape.swapAdjacentAtDepth [rows, keys] 0))
          gradient) + Rep.dot deltaMultiplier (weights * upstream))
    -- `ac_rfl` matches atoms syntactically, so the rewrite must keep the real instances.
    have hmul : ∀ {shape : Shape} (x y z : Tensor ℝ shape), Rep.dot (x * y) z = Rep.dot x (y * z) :=
      fun x y z => Tensor.dot_elemwise_adjoint x y z
    rw [Tensor.dot_add_left, Tensor.dot_matmul_right_adjoint,
      Tensor.dot_matmul_left_adjoint,
      Tensor.dot_add_left, hmul, mul_comm multiplier upstream,
      Softmax.dot_backwardFromWeights_comm scores deltaScores (upstream * multiplier) fixedMask
        (center := true),
      mul_comm weights deltaMultiplier, hmul]
    ac_rfl
  | dim n batch ih =>
    simp only [weightsFast, valuesBackwardFast,
      valuesJVPFast]
    simp only [List.cons_append, Tensor.dot_eq_inner, Tensor.inner_eq_sum_unstack,
      Tensor.unstack_dim]
    simp_rw [← Tensor.dot_eq_inner]
    rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun i _ => ih (scores.unstack i) (deltaScores.unstack i)
      (multiplier.unstack i) (deltaMultiplier.unstack i) (value.unstack i)
      (deltaValue.unstack i) (gradient.unstack i)

/-- Each output coordinate of attention after its score product has derivative equal to the matching
coordinate of `Attention.valuesJVP`.

Scores, values and multipliers may vary together along a real parameter, and the optional mask
stays fixed, including rows with no allowed key. Row normalization uses the analytic theorem for
the stable masked-softmax implementation, and the rest is scalar products and finite sums.
-/
theorem Attention.hasDerivAt_values_get2 {rows keys width : Nat}
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
      (fun t => get2 (values (batch := .scalar)
        (Attention.weights (batch := .scalar) (scores t) mask) (value t)
        (some (multiplier t)))
        row column)
      (get2 (valuesJVP (batch := .scalar)
        (Attention.weights (batch := .scalar) (scores point) mask) (value point)
        (multiplier point) deltaScores deltaValue deltaMultiplier) row column) point := by
  let fixedMask := mask.getD (allTrueMask rows keys)
  let weights := fun t => Softmax.hardMasked (scores t) fixedMask
  let deltaWeights := Softmax.backwardFromWeights (weights point) deltaScores
    (center := true)
  have hw (key : Fin keys) : HasDerivAt (fun t => get2 (weights t) row key)
      (get2 deltaWeights row key) point := by
    have hrow := (Softmax.hasFDerivAt_hardMaskedVec (fixedMask.unstack row)
      (getScalarE ((scores point).unstack row))).comp_hasDerivAt point (hs row)
    have hcoordinate := (EuclideanSpace.proj (𝕜 := ℝ) key).hasFDerivAt.comp_hasDerivAt point hrow
    have hresult := congrArg (fun v : Vec keys => v key)
      (HardMaskedSoftmax.backward_eq_derivative ((scores point).unstack row)
        (deltaScores.unstack row) (fixedMask.unstack row) (center := true))
    simpa only [weights, deltaWeights, Function.comp_def, EuclideanSpace.coe_proj,
      ofFnE_getScalarE, Softmax.unstack_hardMasked, Softmax.unstack_backwardFromWeights,
      Tensor.get2_eq_getScalar_unstack, getScalarE_ofLp] using
        hcoordinate.congr_deriv hresult.symm
  have hsum := HasDerivAt.fun_sum (u := (Finset.univ : Finset (Fin keys)))
    (fun key _ => ((hw key).mul (hm row key)).mul (hv key column))
  rw [valuesFast_eq,
    weightsFast_eq,
    valuesJVPFast_eq]
  simp only [valuesFast, weightsFast,
    weightsSlice_eq_hardMasked, valuesJVPFast,
    valuesJVPSlice, Numeric.QuotientArithmetic.supported_real,
    Bool.not_false, Bool.false_eq_true, ↓reduceIte,
    scaledProductsSlice_two Numeric.QuotientArithmetic.supported_real,
    Tensor.scale_one]
  simp only [Tensor.get2_matmul_eq_sum, Tensor.get2_add]
  simpa only [Tensor.get2_eq_apply, mul_def, Tensor.zipWith_apply,
    Pi.mul_apply, Finset.sum_add_distrib,
    weights, deltaWeights, fixedMask] using hsum

end

end TorchLean
