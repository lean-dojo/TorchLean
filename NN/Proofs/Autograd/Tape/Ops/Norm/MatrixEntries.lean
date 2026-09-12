/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization.Core
public import NN.Proofs.Autograd.Tape.Nodes.Matrix

/-!
# Entrywise formulas for row-normalization tensors

The normalization specs are written with `reduceMean`, `reduceVar`, `broadcastAfterSum`,
`broadcastTo`, and pointwise tensor arithmetic on `[m, n]` matrices. This file records what each
of those operations does to a single entry `get2 t i j`, and how `tensorToVec` indexes a matrix,
so that the LayerNorm and BatchNorm proofs can work with plain real sums.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Norm

open Spec TorchLean
open TorchLean.Tensor

open scoped BigOperators

noncomputable section

variable {m n : Nat}

/-! ## Matrix vectorization -/

/-- `tensorToVec` of a matrix at the flattened index `(i, j)` is the matrix entry. -/
theorem tensorToVec_idxMN (A : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    tensorToVec (t := A) (TapeNodes.Matmul.idxMN (m := m) (n := n) i j) = Spec.get2 A i j := by
  change
    TorchLean.Tensor.getScalar (TorchLean.Tensor.flattenSpec A)
        (TapeNodes.Matmul.idxMN (m := m) (n := n) i j) =
      Spec.get2 A i j
  rw [TorchLean.Tensor.getScalar_eq_apply]
  unfold TorchLean.Tensor.flattenSpec Spec.get2 TorchLean.Tensor.getScalar Spec.get
    TorchLean.Tensor.unstack TorchLean.Tensor.item
  rw [TorchLean.Tensor.Internal.Rep.reshape_apply_coordEquiv]
  rw [TorchLean.Tensor.Internal.Rep.unstack_apply,
    TorchLean.Tensor.Internal.Rep.unstack_apply]
  apply congrArg A
  apply TorchLean.Tensor.Internal.Coord.linearize_injective
  apply Fin.ext
  rw [TorchLean.Tensor.reshapeCoordEquiv_linearize_val]
  rw [TorchLean.Tensor.vectorCoordinate_linearize_val]
  have hlinearize :
      (TorchLean.Tensor.Internal.Coord.linearize (s := [m, n])
        (i, j, PUnit.unit)).val =
        j.val + n * i.val := by
    calc
      _ =
          (TorchLean.Tensor.Internal.Coord.linearize (s := [n])
            (j, PUnit.unit)).val +
            TorchLean.Tensor.Internal.Shape.size [n] * i.val :=
        TorchLean.Tensor.Internal.Coord.linearize_cons_val (s := [n]) i (j, PUnit.unit)
      _ = j.val + n * i.val := by
        rw [TorchLean.Tensor.vectorCoordinate_linearize_val]
        simp
  rw [hlinearize]
  simp [TapeNodes.Matmul.idxMN, finProdFinEquiv_apply_val]

/-- Every flattened matrix index is `idxMN` of its row and column. -/
theorem idxMN_divNat_modNat (ip : Fin (TapeNodes.Matmul.matSize m n)) :
    TapeNodes.Matmul.idxMN (m := m) (n := n)
      (ip.divNat (m := m) (n := TapeNodes.Matmul.vecSize n))
      (Fin.cast (TapeNodes.Matmul.vecSize_eq n)
        (ip.modNat (m := m) (n := TapeNodes.Matmul.vecSize n))) = ip := by
  apply Fin.ext
  change (ip.modNat (m := m) (n := TapeNodes.Matmul.vecSize n)).val +
      TapeNodes.Matmul.vecSize n * (ip.divNat (m := m) (n := TapeNodes.Matmul.vecSize n)).val =
        ip.val
  exact Nat.mod_add_div _ _

/-- Two flattened matrices agree once they agree at every `idxMN i j`. -/
theorem vec_ext_idxMN {u v : Vec (TapeNodes.Matmul.matSize m n)}
    (h : ∀ (i : Fin m) (j : Fin n),
      u (TapeNodes.Matmul.idxMN (m := m) (n := n) i j) =
        v (TapeNodes.Matmul.idxMN (m := m) (n := n) i j)) :
    u = v := by
  apply PiLp.ext
  intro ip
  rw [← idxMN_divNat_modNat ip]
  exact h _ _

/-! ## Vector and matrix entries of pointwise operations -/

/-- Entries of a filled vector. -/
theorem getScalar_full {k : Nat} (v : ℝ) (i : Fin k) :
    getScalar (Tensor.full (.dim k .scalar) v) i = v := by
  simp [Tensor.full]

/-- Entries of a filled matrix. -/
theorem get2_full (v : ℝ) (i : Fin m) (j : Fin n) :
    Spec.get2 (Tensor.full (.dim m (.dim n .scalar)) v) i j = v := by
  simp [Tensor.full, Spec.get2, Spec.get, getScalar_eq_apply, TorchLean.Tensor.unstack]

/-- Entries of a row broadcast: every column of row `i` reads the vector entry `i`. -/
theorem get2_broadcastAfterSum_one (v : Tensor ℝ [m]) (i : Fin m) (j : Fin n) :
    Spec.get2 (broadcastAfterSum (.dim m (.dim n .scalar)) 1 v) i j = getScalar v i := by
  rw [← Tensor.dim_unstack v]
  simp [broadcastAfterSum, Spec.get2, TorchLean.Tensor.getScalar, Spec.get]

/-- Entries of a column broadcast: every row reads the vector entry `j`. -/
theorem get2_broadcastTo_col (h : Shape.CanBroadcastTo (.dim n .scalar) (.dim m (.dim n .scalar)))
    (v : Tensor ℝ [n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (broadcastTo h v) i j = getScalar v j := by
  simp [Spec.get2, Spec.get, TorchLean.Tensor.getScalar]

/-- Row sums along the last axis. -/
theorem getScalar_reduceSum_one (x : Tensor ℝ [m, n])
    (h : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (i : Fin m) :
    getScalar (reduceSum 1 x h) i = ∑ j : Fin n, Spec.get2 x i j := by
  rw [← Tensor.dim_unstack x]
  simp only [reduceSum, reduceDim, TorchLean.Tensor.Reduction.Internal.reduceDimCore_dim_succ,
    TorchLean.Tensor.Reduction.Internal.reduceDimCore_dim_zero,
    TorchLean.Tensor.Reduction.Internal.reduceOuterAxis_vector, shapeAfterSum,
    TorchLean.Tensor.getScalar_dim]
  rw [sum_spec_vec]
  apply Finset.sum_congr rfl
  intro j _
  simp [Spec.get2, Spec.get]

/-- Row means along the last axis. -/
theorem getScalar_reduceMean_one (x : Tensor ℝ [m, n])
    (h : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (i : Fin m) :
    getScalar (reduceMean 1 x h) i = (∑ j : Fin n, Spec.get2 x i j) / n := by
  simp only [reduceMean, getScalar_mapSpec, Shape.axisSize_succ, Shape.axisSize_zero]
  rw [getScalar_reduceSum_one]

/-- Scalar entry of a pointwise binary operation on rank-zero tensors. -/
theorem item_map2Spec (f : ℝ → ℝ → ℝ) (a b : Tensor ℝ .scalar) :
    (map2Spec f a b).item = f a.item b.item := by
  simp [map2Spec, Tensor.item]

/-- Population variance of a vector, as the scalar entry of `reduceVar 0`. -/
theorem item_reduceVar_zero (row : Tensor ℝ [n]) (h : Shape.NonemptyAxis 0 (.dim n .scalar)) :
    (reduceVar 0 row h).item =
      (∑ j : Fin n, (getScalar row j - (∑ k : Fin n, getScalar row k) / n) *
        (getScalar row j - (∑ k : Fin n, getScalar row k) / n)) / n := by
  simp only [reduceVar, reduceMean, reduceSum, reduceDim,
    TorchLean.Tensor.Reduction.Internal.reduceDimCore_dim_zero,
    TorchLean.Tensor.Reduction.Internal.reduceOuterAxis_vector, shapeAfterSum, toScalar_mapSpec,
    Tensor.item_scalar, Shape.axisSize_zero, mapSpec_dim]
  rw [sum_spec_vec, sum_spec_vec]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  simp only [TorchLean.Tensor.getScalar_dim_entry, toScalar_mapSpec, subSpec, toScalar_map2Spec]
  rfl

/-- Row variances along the last axis (population variance of each row). -/
theorem getScalar_reduceVar_one (x : Tensor ℝ [m, n])
    (h : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (i : Fin m) :
    getScalar (reduceVar 1 x h) i =
      (∑ j : Fin n,
        (Spec.get2 x i j - (∑ k : Fin n, Spec.get2 x i k) / n) *
          (Spec.get2 x i j - (∑ k : Fin n, Spec.get2 x i k) / n)) / n := by
  have hrow := item_reduceVar_zero (Tensor.unstack x i)
    (by cases h with | succ h' => exact h')
  simp only [reduceVar] at hrow
  rw [← Tensor.dim_unstack x]
  simp only [reduceVar, TorchLean.Tensor.getScalar_dim_entry, Tensor.unstack_dim]
  rw [hrow]
  simp [Spec.get2, Spec.get]

/-! ## Vectors -/

/-- `tensorToVec` of a rank-one tensor reads the corresponding scalar entry. -/
theorem tensorToVec_vec {k : Nat} (v : Tensor ℝ [k]) (p : Fin (Spec.Shape.size (.dim k .scalar))) :
    tensorToVec (t := v) p = getScalar v (Fin.cast (TapeNodes.Matmul.vecSize_eq k) p) := by
  have hp : p = finProdFinEquiv (Fin.cast (TapeNodes.Matmul.vecSize_eq k) p,
      (⟨0, by simp [Spec.Shape.size]⟩ : Fin (Spec.Shape.size Shape.scalar))) := by
    apply Fin.ext
    rw [finProdFinEquiv_apply_val]
    simp [Spec.Shape.size]
  conv_lhs => rw [← Tensor.dim_unstack v, hp]
  rw [tensorToVec_dim_apply (by simp [Spec.Shape.size])]
  rw [← Tensor.scalar_item (Tensor.unstack v _), tensorToVec_scalar]
  rfl

/-! ## Pointwise operations on matrices and vectors -/

/-- Entries of a matrix sum. -/
theorem get2_addSpec (a b : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (addSpec a b) i j = Spec.get2 a i j + Spec.get2 b i j := by
  simp [addSpec]

/-- Entries of a matrix difference. -/
theorem get2_subSpec (a b : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (subSpec a b) i j = Spec.get2 a i j - Spec.get2 b i j := by
  simp [subSpec]

/-- Entries of a pointwise matrix product. -/
theorem get2_mulSpec (a b : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (mulSpec a b) i j = Spec.get2 a i j * Spec.get2 b i j := by
  simp [mulSpec]

/-- Entries of a pointwise matrix quotient. -/
theorem get2_divSpec (a b : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (divSpec a b) i j = Spec.get2 a i j / Spec.get2 b i j := by
  simp [divSpec]

/-- Entries of a vector sum. -/
theorem getScalar_addSpec' {k : Nat} (a b : Tensor ℝ [k]) (i : Fin k) :
    getScalar (addSpec a b) i = getScalar a i + getScalar b i := by
  simp [addSpec]

/-- Entries of a pointwise vector quotient. -/
theorem getScalar_divSpec {k : Nat} (a b : Tensor ℝ [k]) (i : Fin k) :
    getScalar (divSpec a b) i = getScalar a i / getScalar b i := by
  simp [divSpec]

/-- Entries of a pointwise vector maximum. -/
theorem getScalar_maxSpec {k : Nat} (a b : Tensor ℝ [k]) (i : Fin k) :
    getScalar (maxSpec a b) i = max (getScalar a i) (getScalar b i) := by
  simp [maxSpec]

/-- Entries of the clamped square root. -/
theorem getScalar_sqrtSpec {k : Nat} (a : Tensor ℝ [k]) (i : Fin k) :
    getScalar (sqrtSpec a) i = Real.sqrt (max (getScalar a i) 0) := by
  simp [sqrtSpec]
  rfl

/-! ## Row statistics of a spec matrix -/

/-- Mean of row `i`. -/
def rowMeanE (x : Tensor ℝ [m, n]) (i : Fin m) : ℝ :=
  (∑ j : Fin n, Spec.get2 x i j) / n

/-- Population variance of row `i`. -/
def rowVarE (x : Tensor ℝ [m, n]) (i : Fin m) : ℝ :=
  (∑ j : Fin n, (Spec.get2 x i j - rowMeanE x i) * (Spec.get2 x i j - rowMeanE x i)) / n

/-- The row variance is a mean of squares. -/
theorem rowVarE_nonneg (x : Tensor ℝ [m, n]) (i : Fin m) : 0 ≤ rowVarE x i :=
  div_nonneg (Finset.sum_nonneg fun _ _ => mul_self_nonneg _) (Nat.cast_nonneg n)

/-- Centered row entries sum to zero. -/
theorem sum_sub_rowMeanE (hn : 0 < n) (x : Tensor ℝ [m, n]) (i : Fin m) :
    ∑ j : Fin n, (Spec.get2 x i j - rowMeanE x i) = 0 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  simp only [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, rowMeanE]
  field_simp
  ring

/-! ## `Spec.layerNorm` entrywise -/

/-- The feature axis of an `[m, n]` matrix is nonempty when `0 < n`. -/
theorem nonemptyAxis_one (hn : 0 < n) : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar)) :=
  Shape.NonemptyAxis.succ (Shape.hasNonemptyAxisZeroOfPos hn).proof

/-- Column-broadcast evidence used by `Spec.layerNorm` for `gamma` and `beta`. -/
theorem colBroadcast : Shape.CanBroadcastTo (.dim n .scalar) (.dim m (.dim n .scalar)) :=
  Shape.CanBroadcastTo.expand_dims (Shape.CanBroadcastTo.dim_eq Shape.CanBroadcastTo.scalar)

/-- Row means of the LayerNorm input. -/
def lnMean (x : Tensor ℝ [m, n]) (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) :
    Tensor ℝ [m] :=
  reduceMean 1 x h1

/-- Centered LayerNorm input. -/
def lnCentered (x : Tensor ℝ [m, n]) (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) :
    Tensor ℝ [m, n] :=
  subSpec x (broadcastAfterSum (.dim m (.dim n .scalar)) 1 (lnMean x h1))

/-- Clamped LayerNorm standard deviation `sqrt (max (max var 0 + ε) 0)`. -/
def lnStd (x : Tensor ℝ [m, n]) (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (ε : ℝ) :
    Tensor ℝ [m] :=
  sqrtSpec (addSpec (maxSpec (reduceVar 1 (lnCentered x h1) h1) (Tensor.full (.dim m .scalar) 0))
    (Tensor.full (.dim m .scalar) ε))

/-- `Spec.layerNorm` with its axis and broadcast evidence spelled out. -/
def layerNormMat (x : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n])
    (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (ε : ℝ) : Tensor ℝ [m, n] :=
  addSpec
    (mulSpec
      (divSpec (lnCentered x h1) (broadcastAfterSum (.dim m (.dim n .scalar)) 1 (lnStd x h1 ε)))
      (broadcastTo (colBroadcast (m := m) (n := n)) gamma))
    (broadcastTo (colBroadcast (m := m) (n := n)) beta)

/-- `Spec.layerNorm` is `layerNormMat` by unfolding. -/
theorem layerNorm_eq_layerNormMat (hm : 0 < m) (hn : 0 < n) (ε : ℝ) (x : Tensor ℝ [m, n])
    (gamma beta : Tensor ℝ [n]) :
    Spec.layerNorm x gamma beta hm hn ε = layerNormMat x gamma beta (nonemptyAxis_one hn) ε :=
  rfl

/-- Entries of the LayerNorm row means. -/
theorem getScalar_lnMean (x : Tensor ℝ [m, n])
    (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (i : Fin m) :
    getScalar (lnMean x h1) i = rowMeanE x i := by
  rw [lnMean, getScalar_reduceMean_one]
  rfl

/-- Entries of the centered LayerNorm input. -/
theorem get2_lnCentered (x : Tensor ℝ [m, n])
    (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) :
    Spec.get2 (lnCentered x h1) i j = Spec.get2 x i j - rowMeanE x i := by
  rw [lnCentered, get2_subSpec, get2_broadcastAfterSum_one, getScalar_lnMean]

/-- Entries of the LayerNorm standard deviation: the clamp on the variance is inactive. -/
theorem getScalar_lnStd (hn : 0 < n) (x : Tensor ℝ [m, n])
    (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (ε : ℝ) (i : Fin m) :
    getScalar (lnStd x h1 ε) i = Real.sqrt (max (rowVarE x i + ε) 0) := by
  rw [lnStd, getScalar_sqrtSpec, getScalar_addSpec', getScalar_maxSpec, getScalar_full,
    getScalar_full, getScalar_reduceVar_one]
  simp only [get2_lnCentered]
  have h0 : (∑ k : Fin n, (Spec.get2 x i k - rowMeanE x i)) / n = 0 := by
    rw [sum_sub_rowMeanE hn]
    simp
  rw [h0]
  simp only [sub_zero]
  change Real.sqrt (max (max (rowVarE x i) 0 + ε) 0) = _
  rw [max_eq_left (rowVarE_nonneg x i)]

/-- Entries of `layerNormMat`. -/
theorem get2_layerNormMat (hn : 0 < n) (x : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n])
    (h1 : Shape.NonemptyAxis 1 (.dim m (.dim n .scalar))) (ε : ℝ) (i : Fin m) (j : Fin n) :
    Spec.get2 (layerNormMat x gamma beta h1 ε) i j =
      (Spec.get2 x i j - rowMeanE x i) / Real.sqrt (max (rowVarE x i + ε) 0) *
          getScalar gamma j + getScalar beta j := by
  rw [layerNormMat, get2_addSpec, get2_mulSpec, get2_divSpec, get2_broadcastAfterSum_one,
    get2_lnCentered, getScalar_lnStd hn, get2_broadcastTo_col, get2_broadcastTo_col]

/-- Entries of `Spec.layerNorm`: each row is centered, divided by the clamped standard deviation
of that row, then scaled and shifted by the per-column parameters. -/
theorem get2_layerNorm (hm : 0 < m) (hn : 0 < n) (ε : ℝ) (x : Tensor ℝ [m, n])
    (gamma beta : Tensor ℝ [n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (Spec.layerNorm x gamma beta hm hn ε) i j =
      (Spec.get2 x i j - rowMeanE x i) / Real.sqrt (max (rowVarE x i + ε) 0) *
          getScalar gamma j + getScalar beta j := by
  rw [layerNorm_eq_layerNormMat, get2_layerNormMat hn]

end

end Norm
end Autograd
end Proofs
