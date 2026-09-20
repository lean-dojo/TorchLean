/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.SoftmaxSpec
public import NN.Proofs.Autograd.Tape.Ops.Attention.ScaledDotProduct
public import NN.Proofs.Autograd.Tape.Ops.Norm.CtxVecEval
public import NN.Proofs.Models.Attention.HardMask

/-!
# Scaled dot-product attention: tape graph versus specification

`NN.Proofs.Autograd.Tape.Ops.Attention.ScaledDotProduct` proves `backprop = (fderiv eval)†` for a
tape graph built from proven nodes. `NN.Spec.Layers.Attention` defines
`Spec.scaledDotProductAttention` directly on tensors. This file connects the two:

- `get_sdpaOutIdx_evalVec`: the output block of the graph evaluation is exactly the vectorized
  specification forward pass `softmax(Q Kᵀ / √d) V`;
- `hasFDerivAt_sdpaSpecForwardVec`: the specification forward pass is Fréchet-differentiable, with
  derivative read off the graph;
- `backpropVec_eq_adjoint_fderiv_scaledDotProductAttention`: the tape reverse pass seeded on the
  output block is the vector-Jacobian product of `Spec.scaledDotProductAttention`;
- `backpropVec_eq_adjoint_fderiv_scaledDotProductAttention_allTrueMask`: the same statement for
  the masked code path with the all-true Boolean mask.

Blocks of the evaluated graph are read with the `Graph.evalVec` lemmas of
`NN.Proofs.Autograd.Tape.Ops.Norm.CtxVecEval`. The first half of this file relates the flattened
node functions (`matmulVec`, `transposeVec`, `forwardMN`) to `Spec.matMulSpec`,
`swapAdjacentAxes`, `scaleSpec`, and `Activation.softmaxSpec 1`; the second half walks the
attention graph node by node.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor
open TapeNodes DGraph
open TapeNodes.Matmul

noncomputable section

/-! ## Flattened matrices -/

theorem val_finProdFinEquiv {m k : Nat} (i : Fin m) (j : Fin k) :
    (finProdFinEquiv (i, j)).val = j.val + k * i.val := rfl

/-- The flattened index of entry `(i, j)` is `j + n * i`: row-major layout, as in PyTorch. -/
theorem val_idxMN {m n : Nat} (i : Fin m) (j : Fin n) :
    (idxMN (m := m) (n := n) i j).val = j.val + n * i.val := by
  change j.val + Spec.Shape.size (Shape.dim n Shape.scalar) * i.val = _
  simp [Spec.Shape.size]

/-- Every flattened matrix index is the image of a row/column pair. -/
theorem exists_idxMN {m n : Nat} (ip : Fin (matSize m n)) :
    ∃ i : Fin m, ∃ j : Fin n, idxMN (m := m) (n := n) i j = ip := by
  have hsz : matSize m n = m * n := by simp [matSize, Spec.Shape.size]
  refine ⟨(Fin.cast hsz ip).divNat, (Fin.cast hsz ip).modNat, ?_⟩
  apply Fin.ext
  rw [val_idxMN]
  simp only [Fin.coe_divNat, Fin.coe_modNat, Fin.val_cast]
  exact Nat.mod_add_div ip.val n

/-- Coordinate `idxMN i j` of a vectorized matrix is the matrix entry `A i j`.

This is the bridge the whole file is named for. Attention is proved in the flat `Vec` world, where
Mathlib's calculus lives, and stated about `Tensor ℝ [m, n]`; this lemma is what connects the two
without ever unfolding the flattening. -/
theorem tensorToVec_idxMN {m n : Nat} (A : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    tensorToVec (t := A) (idxMN (m := m) (n := n) i j) = Spec.get2 A i j := by
  have hn : 0 < Spec.Shape.size (Shape.dim n Shape.scalar) := by
    simp only [Spec.Shape.size, Nat.mul_one]
    exact j.pos
  have h1 : 0 < Spec.Shape.size Shape.scalar := by simp [Spec.Shape.size]
  have hrow := tensorToVec_dim_apply hn (Tensor.unstack A)
    (i, finProdFinEquiv (j, (⟨0, h1⟩ : Fin (Spec.Shape.size Shape.scalar))))
  have hcol := tensorToVec_dim_apply h1 (Tensor.unstack (Tensor.unstack A i))
    (j, (⟨0, h1⟩ : Fin (Spec.Shape.size Shape.scalar)))
  rw [Tensor.dim_unstack] at hrow hcol
  have hidx : (finProdFinEquiv (i, finProdFinEquiv (j, (⟨0, h1⟩ : Fin (Spec.Shape.size
      Shape.scalar)))) : Fin (matSize m n)) = idxMN (m := m) (n := n) i j := by
    apply Fin.ext
    rw [val_idxMN]
    change (0 + Spec.Shape.size Shape.scalar * j.val) +
      Spec.Shape.size (Shape.dim n Shape.scalar) * i.val = j.val + n * i.val
    simp [Spec.Shape.size]
  rw [hidx] at hrow
  rw [hrow, hcol, ← Tensor.scalar_item (Tensor.unstack (Tensor.unstack A i) j), tensorToVec_scalar]
  rfl

/-- Vectorization is homogeneous: scaling a matrix scales its vector. This is what lets the `1/√d`
factor in scaled dot-product attention be handled as ordinary scalar multiplication. -/
theorem tensorToVec_scaleSpec {m n : Nat} (A : Tensor ℝ [m, n]) (c : ℝ) :
    tensorToVec (t := scaleSpec A c) = c • tensorToVec (t := A) := by
  ext ip
  obtain ⟨i, j, rfl⟩ := exists_idxMN ip
  rw [PiLp.smul_apply, tensorToVec_idxMN, tensorToVec_idxMN, smul_eq_mul]
  simp [scaleSpec, mul_comm]

/-- Transposition on flat vectors swaps the row and column index, as expected. -/
theorem transposeVec_idxMN {m n : Nat} (a : Vec (matSize m n)) (i : Fin n) (j : Fin m) :
    MatTranspose.transposeVec (m := m) (n := n) a (idxMN (m := n) (n := m) i j) =
      a (idxMN (m := m) (n := n) j i) := by
  have hidx : Fin.cast (MatTranspose.matSize_eq_mul m n).symm
      ((MatTranspose.transposeEquiv m n).symm
        (Fin.cast (MatTranspose.matSize_eq_mul n m) (idxMN (m := n) (n := m) i j))) =
      idxMN (m := m) (n := n) j i := by
    apply Fin.ext
    change (idxMN (m := n) (n := m) i j).val / m + n * ((idxMN (m := n) (n := m) i j).val % m) =
      (idxMN (m := m) (n := n) j i).val
    rw [val_idxMN, val_idxMN, Nat.add_mul_div_left _ _ j.pos, Nat.add_mul_mod_self_left,
      Nat.div_eq_of_lt j.isLt, Nat.mod_eq_of_lt j.isLt, Nat.zero_add]
  simp only [MatTranspose.transposeVec, castVec_apply, vecOfFun_apply]
  rw [hidx]

/-- Swapping the two axes of a matrix tensor corresponds to the flat transpose.

Attention transposes the key matrix, so without this the `Qᵀ` in `Q Kᵀ` would have to be reasoned
about at the tensor level and at the vector level separately. -/
theorem tensorToVec_swapAdjacentAxes {m n : Nat} (A : Tensor ℝ [m, n]) :
    tensorToVec (t := swapAdjacentAxes A 0) =
      MatTranspose.transposeVec (m := m) (n := n) (tensorToVec (t := A)) := by
  ext ip
  obtain ⟨i, j, rfl⟩ := exists_idxMN ip
  rw [tensorToVec_idxMN, Spec.get2_matrix_transpose_spec, transposeVec_idxMN, tensorToVec_idxMN]

/-- Reading a vectorized matrix through the `Fin m × Fin n` product equivalence, after the size
cast,
gives the matrix entry. This is `tensorToVec_idxMN` in the indexing the softmax development uses. -/
theorem castVec_tensorToVec_finProdFinEquiv {m n : Nat} (A : Tensor ℝ [m, n])
    (hsz : Spec.Shape.size (Shape.dim m (Shape.dim n Shape.scalar)) = m * n)
    (i : Fin m) (j : Fin n) :
    castVec hsz (tensorToVec (t := A)) (finProdFinEquiv (i, j)) = Spec.get2 A i j := by
  have hidx : Fin.cast hsz.symm (finProdFinEquiv (i, j)) = idxMN (m := m) (n := n) i j := by
    apply Fin.ext
    rw [val_idxMN]
    rfl
  rw [castVec_apply, hidx, tensorToVec_idxMN]

/-- Row `i` of a vectorized matrix is the vectorization of row `i` of the tensor. -/
theorem rows_castVec_tensorToVec {m n : Nat} (A : Tensor ℝ [m, n])
    (hsz : Spec.Shape.size (Shape.dim m (Shape.dim n Shape.scalar)) = m * n) (i : Fin m) :
    SoftmaxLastAxis.rows (m := m) (n := n) (castVec hsz (tensorToVec (t := A))) i =
      getScalarE (Spec.get A i) := by
  ext j
  rw [SoftmaxLastAxis.rows, vecOfFun_apply, castVec_tensorToVec_finProdFinEquiv,
    getScalarE_ofLp, Spec.get2_eq_getScalar_get]

/-- Row-wise softmax on tensors agrees with the flat `forwardMN` softmax on vectors.

The temperature is fixed to `1` because that is the only case attention needs; the scaling is
already
folded into the scores by `tensorToVec_scaleSpec`. -/
theorem castVec_tensorToVec_softmaxSpec_one {m n : Nat} (S : Tensor ℝ [m, n])
    (hsz : Spec.Shape.size (Shape.dim m (Shape.dim n Shape.scalar)) = m * n) :
    castVec hsz (tensorToVec (t := Activation.softmaxSpec (α := ℝ) (s := [m, n]) 1 S)) =
      SoftmaxLastAxis.forwardMN (m := m) (n := n) (castVec hsz (tensorToVec (t := S))) := by
  ext ip
  obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective ip
  rw [castVec_tensorToVec_finProdFinEquiv, Spec.get2_eq_getScalar_get, Proofs.get_softmaxSpec_one,
    ← getScalarE_ofLp, getScalarE_softmaxVecSpec]
  simp only [SoftmaxLastAxis.forwardMN, SoftmaxLastAxis.unrows, vecOfFun_apply,
    Equiv.symm_apply_apply, rows_castVec_tensorToVec]

/-! ## Attention graph evaluation -/

namespace Attention

/-- Index of the attention output block in the saved context. -/
def sdpaOutIdx (m d : Nat) :
    Idx (ΓQKV m d ++ ssScaledDotProduct m d) (.dim m (.dim d .scalar)) :=
  Idx.last (Γ := ΓQKV m d)
    (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
      .dim m (.dim m .scalar)])
    (τ := .dim m (.dim d .scalar))

/-- The `Q` tensor stored in a vectorized attention context. -/
abbrev ctxQ {m d : Nat} (xV : CtxVec (ΓQKV m d)) : Tensor ℝ [m, d] :=
  vecToTensor (s := QKVShape m d) (CtxVec.get (idxQ (m := m) (d := d) (ss := [])) xV)

/-- The `K` tensor stored in a vectorized attention context. -/
abbrev ctxK {m d : Nat} (xV : CtxVec (ΓQKV m d)) : Tensor ℝ [m, d] :=
  vecToTensor (s := QKVShape m d) (CtxVec.get (idxK (m := m) (d := d) (ss := [])) xV)

/-- The `V` tensor stored in a vectorized attention context. -/
abbrev ctxV {m d : Nat} (xV : CtxVec (ΓQKV m d)) : Tensor ℝ [m, d] :=
  vecToTensor (s := QKVShape m d) (CtxVec.get (idxV (m := m) (d := d) (ss := [])) xV)

/-- Tensor-level attention forward pass with an explicit scale factor `c`. -/
def attentionForwardSpec {m d : Nat} (c : ℝ) (Q K V : Tensor ℝ [m, d]) : Tensor ℝ [m, d] :=
  matMulSpec
    (Activation.softmaxSpec (α := ℝ) (s := [m, m]) 1
      (scaleSpec (matMulSpec Q (swapAdjacentAxes K 0)) c))
    V

/-- Unmasked `Spec.scaledDotProductAttention` is `attentionForwardSpec` at the scale `1 / √d`. -/
theorem scaledDotProductAttention_eq_attentionForwardSpec {m d : Nat} (hm : m ≠ 0)
    (Q K V : Tensor ℝ [m, d]) :
    Spec.scaledDotProductAttention (h1 := hm) (h2 := hm) { Q := Q, K := K, V := V, mask := none } =
      attentionForwardSpec (1 / Spec.attentionScaleDenom (α := ℝ) d) Q K V := rfl

/-- The output block of the attention graph is the vectorized tensor forward pass. -/
theorem get_sdpaOutIdx_evalVec {m d : Nat} (c : ℝ) (xV : CtxVec (ΓQKV m d)) :
    CtxVec.get (sdpaOutIdx m d) (Graph.evalVec (scaledDotProductGraph c) xV) =
      tensorToVec (attentionForwardSpec c (ctxQ xV) (ctxK xV) (ctxV xV)) := by
  have hOut := Graph.get_evalVec_snoc_last (graphProbs m d c) (nodeOut m d) xV (sdpaOutIdx m d)
    rfl
  have hP := Graph.get_evalVec_snoc_last (graphScaled m d c) (nodeProbs m d) xV
    (Idx.last (Γ := ΓQKV m d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))) rfl
  have hS := Graph.get_evalVec_snoc_last (graphLogits m d) (nodeScaled m d c) xV
    (Idx.last (Γ := ΓQKV m d) (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))) rfl
  have hL := Graph.get_evalVec_snoc_last (graphKt m d) (nodeLogits m d) xV
    (Idx.last (Γ := ΓQKV m d) (ss := [.dim d (.dim m .scalar)]) (τ := .dim m (.dim m .scalar)))
    rfl
  have hKt := Graph.get_evalVec_snoc_last Graph.nil (nodeKt m d) xV
    (Idx.last (Γ := ΓQKV m d) (ss := []) (τ := .dim d (.dim m .scalar))) rfl
  have hV : CtxVec.get (idxV (m := m) (d := d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
        .dim m (.dim m .scalar)])) (Graph.evalVec (graphProbs m d c) xV) =
      CtxVec.get (idxV (m := m) (d := d) (ss := [])) xV :=
    Graph.get_evalVec_input _ xV _ _ rfl
  have hQ : CtxVec.get (idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
      (Graph.evalVec (graphKt m d) xV) = CtxVec.get (idxQ (m := m) (d := d) (ss := [])) xV :=
    Graph.get_evalVec_input _ xV _ _ rfl
  have hK : CtxVec.get (idxK (m := m) (d := d) (ss := [])) (Graph.evalVec Graph.nil xV) =
      CtxVec.get (idxK (m := m) (d := d) (ss := [])) xV :=
    Graph.get_evalVec_input _ xV _ _ rfl
  have hKmat : MatTranspose.transposeVec (m := m) (n := d)
      (CtxVec.get (idxK (m := m) (d := d) (ss := [])) xV) =
      tensorToVec (swapAdjacentAxes (ctxK xV) 0) := by
    rw [tensorToVec_swapAdjacentAxes, tensorToVec_vecToTensor]
  rw [scaledDotProductGraph, hOut]
  simp only [nodeOut, TapeNodes.matmul, Node.forwardVec_ofFn]
  rw [hP, hV]
  simp only [nodeProbs, TapeNodes.softmaxLast, Node.forwardVec_ofFn]
  rw [hS]
  simp only [nodeScaled, TapeNodes.scale, Node.forwardVec_ofFn]
  rw [hL]
  simp only [nodeLogits, TapeNodes.matmul, Node.forwardVec_ofFn]
  rw [hQ, hKt]
  simp only [nodeKt, TapeNodes.matrixTranspose, Node.forwardVec_ofFn]
  rw [hK, hKmat, vecToTensor_tensorToVec, ← tensorToVec_scaleSpec,
    ← castVec_tensorToVec_softmaxSpec_one, castVec_castVec, castVec_rfl,
    vecToTensor_tensorToVec]
  rfl

/-- The specification forward pass as a function on vectorized contexts. -/
def sdpaSpecForwardVec {m d : Nat} (c : ℝ) (xV : CtxVec (ΓQKV m d)) :
    Vec (Spec.Shape.size (Shape.dim m (Shape.dim d Shape.scalar))) :=
  tensorToVec (attentionForwardSpec c (ctxQ xV) (ctxK xV) (ctxV xV))

/-- The specification forward pass is the output projection of the graph evaluation. -/
theorem sdpaSpecForwardVec_eq_getCLM_evalVec {m d : Nat} (c : ℝ) :
    sdpaSpecForwardVec (m := m) (d := d) c =
      fun xV => CtxVec.getCLM (sdpaOutIdx m d) (Graph.evalVec (scaledDotProductGraph c) xV) := by
  funext xV
  rw [CtxVec.getCLM_apply, get_sdpaOutIdx_evalVec]
  rfl

/-- The specification attention forward pass is Fréchet-differentiable; its derivative is the
output projection of the graph derivative. -/
theorem hasFDerivAt_sdpaSpecForwardVec {m d : Nat} (c : ℝ) (xV : CtxVec (ΓQKV m d)) :
    HasFDerivAt (sdpaSpecForwardVec (m := m) (d := d) c)
      ((CtxVec.getCLM (sdpaOutIdx m d)).comp
        (fderiv ℝ (Graph.evalVec (scaledDotProductGraph (m := m) (d := d) c)) xV)) xV := by
  obtain ⟨D, hD, -⟩ := Graph.hasFDerivAt_evalVec_and_jvp (scaledDotProductGraph c)
    (scaledDotProductDGraph c).hg xV
  rw [sdpaSpecForwardVec_eq_getCLM_evalVec, hD.fderiv]
  exact (CtxVec.getCLM (sdpaOutIdx m d)).hasFDerivAt.comp xV hD

/-- The adjoint of block projection is block injection. -/
theorem adjoint_getCLM {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (v : Vec (Spec.Shape.size s)) :
    (CtxVec.getCLM idx).adjoint v = CtxVec.single idx v := by
  apply ext_inner_left ℝ
  intro x
  rw [ContinuousLinearMap.adjoint_inner_right, CtxVec.getCLM_apply, CtxVec.inner_get_single]

/-- Tape backprop seeded on the output block is the VJP of the specification forward pass. -/
theorem backpropVec_eq_adjoint_fderiv_sdpaSpec {m d : Nat} (c : ℝ) (xV : CtxVec (ΓQKV m d))
    (δ : Vec (Spec.Shape.size (Shape.dim m (Shape.dim d Shape.scalar)))) :
    Graph.backpropVec (scaledDotProductGraph c) xV (CtxVec.single (sdpaOutIdx m d) δ) =
      (fderiv ℝ (sdpaSpecForwardVec (m := m) (d := d) c) xV).adjoint δ := by
  rw [(hasFDerivAt_sdpaSpecForwardVec c xV).fderiv, ContinuousLinearMap.adjoint_comp,
    ContinuousLinearMap.comp_apply, adjoint_getCLM]
  exact Graph.backpropVec_eq_adjoint_fderiv _ (scaledDotProductDGraph c).hg xV _

/-- The output block of the attention graph at scale `1 / √d` is the vectorized
`Spec.scaledDotProductAttention` forward pass. -/
theorem get_sdpaOutIdx_evalVec_scaledDotProductAttention {m d : Nat} (hm : m ≠ 0)
    (xV : CtxVec (ΓQKV m d)) :
    CtxVec.get (sdpaOutIdx m d)
        (Graph.evalVec (scaledDotProductGraph (1 / Spec.attentionScaleDenom (α := ℝ) d)) xV) =
      tensorToVec (Spec.scaledDotProductAttention (h1 := hm) (h2 := hm)
        { Q := ctxQ xV, K := ctxK xV, V := ctxV xV, mask := none }) :=
  get_sdpaOutIdx_evalVec _ xV

/-- Tape backprop is the vector-Jacobian product of unmasked `Spec.scaledDotProductAttention`. -/
theorem backpropVec_eq_adjoint_fderiv_scaledDotProductAttention {m d : Nat} (hm : m ≠ 0)
    (xV : CtxVec (ΓQKV m d))
    (δ : Vec (Spec.Shape.size (Shape.dim m (Shape.dim d Shape.scalar)))) :
    Graph.backpropVec (scaledDotProductGraph (1 / Spec.attentionScaleDenom (α := ℝ) d)) xV
        (CtxVec.single (sdpaOutIdx m d) δ) =
      (fderiv ℝ (fun xV : CtxVec (ΓQKV m d) =>
        tensorToVec (Spec.scaledDotProductAttention (h1 := hm) (h2 := hm)
          { Q := ctxQ xV, K := ctxK xV, V := ctxV xV, mask := none })) xV).adjoint δ :=
  backpropVec_eq_adjoint_fderiv_sdpaSpec _ xV δ

/-- With the all-true Boolean mask, the masked code path of `Spec.scaledDotProductAttention` has
the same vector-Jacobian product, computed by the same tape backprop. -/
theorem backpropVec_eq_adjoint_fderiv_scaledDotProductAttention_allTrueMask {m d : Nat}
    (hm : m ≠ 0) (xV : CtxVec (ΓQKV m d))
    (δ : Vec (Spec.Shape.size (Shape.dim m (Shape.dim d Shape.scalar)))) :
    Graph.backpropVec (scaledDotProductGraph (1 / Spec.attentionScaleDenom (α := ℝ) d)) xV
        (CtxVec.single (sdpaOutIdx m d) δ) =
      (fderiv ℝ (fun xV : CtxVec (ΓQKV m d) =>
        tensorToVec (Spec.scaledDotProductAttention (h1 := hm) (h2 := hm)
          { Q := ctxQ xV, K := ctxK xV, V := ctxV xV, mask := some (Spec.allTrueMask m m) }))
        xV).adjoint δ := by
  have hfun : (fun xV : CtxVec (ΓQKV m d) =>
      tensorToVec (Spec.scaledDotProductAttention (h1 := hm) (h2 := hm)
        { Q := ctxQ xV, K := ctxK xV, V := ctxV xV, mask := some (Spec.allTrueMask m m) })) =
      fun xV : CtxVec (ΓQKV m d) =>
        tensorToVec (Spec.scaledDotProductAttention (h1 := hm) (h2 := hm)
          { Q := ctxQ xV, K := ctxK xV, V := ctxV xV, mask := none }) := by
    funext xV
    exact congrArg tensorToVec
      (NN.Proofs.Models.Attention.scaledDotProductAttention_allTrueMask
        (ctx := { Q := ctxQ xV, K := ctxK xV, V := ctxV xV, mask := none }))
  rw [hfun]
  exact backpropVec_eq_adjoint_fderiv_scaledDotProductAttention hm xV δ

end Attention

end
end Autograd
end Proofs
