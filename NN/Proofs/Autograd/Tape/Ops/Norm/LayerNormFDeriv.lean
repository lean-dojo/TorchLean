/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNorm

/-!
# LayerNorm derivatives

LayerNorm subtracts each row's mean, divides by `sqrt (variance + ε)`, and applies the
per-column scale and bias. The input, scale, and bias all vary in these theorems; epsilon is a
fixed positive real number. In particular, neither the scale nor its perturbation is replaced
by an all-ones vector.

The graph calculus already proves that `Spec.layerNorm` is differentiable. Here we identify its
derivative with the actual `Spec.layerNormJvp` formula. We differentiate a normalized row entry,
apply the product rule for the scale, and add the bias perturbation. The existing JVP/VJP pairing
then identifies `Spec.layerNormBackward` with the adjoint derivative, including both parameter
cotangents. This also identifies the three tensors returned by the primitive with the backward
result of the detailed proof graph.

Positive epsilon keeps every row's standard deviation nonzero, including constant rows and
rows with a single feature. All statements here concern exact real arithmetic.
-/

@[expose] public section

namespace Proofs.Autograd.LayerNorm

open Spec TorchLean
open TorchLean.Tensor
open TapeNodes TapeNodes.Matmul
open scoped BigOperators

noncomputable section

variable {m n : Nat}

/-- The tensor and vector descriptions use the same row mean. -/
theorem rowMeanE_specX (q : CtxVec (ΓLN m n)) (i : Fin m) :
    Norm.rowMeanE (specX q) i = RowNorm.rowMean (valX q) i := by
  simp only [Norm.rowMeanE, RowNorm.rowMean, valX_idxMN]

/-- The tensor and vector descriptions use the same population variance. -/
theorem rowVarE_specX (q : CtxVec (ΓLN m n)) (i : Fin m) :
    Norm.rowVarE (specX q) i = RowNorm.rowVar (valX q) i := by
  simp only [Norm.rowVarE, RowNorm.rowVar, RowNorm.centered, valX_idxMN,
    rowMeanE_specX]

/-- With positive epsilon the standard-deviation clamp is inactive in every row. -/
theorem lnInvStd_specX (hn : 0 < n) {ε : ℝ} (hε : 0 < ε)
    (q : CtxVec (ΓLN m n)) (i : Fin m) :
    getScalar (lnInvStd (specX q) (Norm.nonemptyAxis_one hn) ε) i =
      RowNorm.invStd (valX q) ε i := by
  rw [lnInvStd, Norm.getScalar_divSpec, Norm.getScalar_full, Norm.getScalar_lnStd hn,
    rowVarE_specX, max_eq_left (RowNorm.rowVar_add_pos hε (valX q) i).le, one_div]
  rfl

/-- One output entry is the normalized input times its own scale, plus its own bias. -/
theorem specLayerNormVec_idxMN (hm : 0 < m) (hn : 0 < n) {ε : ℝ} (hε : 0 < ε)
    (q : CtxVec (ΓLN m n)) (i : Fin m) (j : Fin n) :
    specLayerNormVec hm hn ε q (idxMN (m := m) (n := n) i j) =
      RowNorm.nrm (valX q) ε i j * valGamma q j + valBeta q j := by
  rw [specLayerNormVec_eq_valY, valY_idxMN, rowVarE_specX,
    max_eq_left (RowNorm.rowVar_add_pos hε (valX q) i).le,
    ← valX_idxMN, rowMeanE_specX, ← valGamma_apply, ← valBeta_apply]
  rfl

/-- Entrywise differentiation includes the input tangent and both affine parameter tangents.

The normalized row differential subtracts the tangent's row mean and its component along the
normalized input. The product rule supplies the additional `xhat * dgamma` term. -/
theorem fderiv_specLayerNormVec_idxMN (hm : 0 < m) (hn : 0 < n) {ε : ℝ} (hε : 0 < ε)
    (q dq : CtxVec (ΓLN m n)) (i : Fin m) (j : Fin n) :
    fderiv ℝ (specLayerNormVec hm hn ε) q dq (idxMN (m := m) (n := n) i j) =
      RowNorm.nrmJvp (valX q) ε (valX dq) i j * valGamma q j +
        RowNorm.nrm (valX q) ε i j * valGamma dq j + valBeta dq j := by
  let input := CtxVec.getCLM (Γ := ΓLN m n) (s := MatShape m n) idxX0
  let scale := (EuclideanSpace.proj (𝕜 := ℝ) j).comp
    (getVecCLM (Γ := ΓLN m n) idxGamma0)
  let bias := (EuclideanSpace.proj (𝕜 := ℝ) j).comp
    (getVecCLM (Γ := ΓLN m n) idxBeta0)
  -- Each context projection carries a size cast. Its evaluation lemma identifies the block
  -- without expanding the recursive context representation during the derivative calculation.
  have hinput (p : CtxVec (ΓLN m n)) : input p = valX p :=
    CtxVec.getCLM_apply idxX0 p
  have hscale (p : CtxVec (ΓLN m n)) : scale p = valGamma p j := by
    change (getVecCLM idxGamma0 p) j = valGamma p j
    exact congrArg (fun v : Vec n => v j) (getVecCLM_apply idxGamma0 p)
  have hbias (p : CtxVec (ΓLN m n)) : bias p = valBeta p j := by
    change (getVecCLM idxBeta0 p) j = valBeta p j
    exact congrArg (fun v : Vec n => v j) (getVecCLM_apply idxBeta0 p)
  have hX : HasFDerivAt (fun p : CtxVec (ΓLN m n) => valX p) input q :=
    input.hasFDerivAt.congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun p => (hinput p).symm)
  have hg : HasFDerivAt (fun p : CtxVec (ΓLN m n) => valGamma p j) scale q :=
    scale.hasFDerivAt.congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun p => (hscale p).symm)
  have hb : HasFDerivAt (fun p : CtxVec (ΓLN m n) => valBeta p j) bias q :=
    bias.hasFDerivAt.congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun p => (hbias p).symm)
  have hentry := (((RowNorm.hasFDerivAt_nrm hε (valX q) i j).comp q hX).mul hg).add hb
  let read := EuclideanSpace.proj (𝕜 := ℝ) (idxMN (m := m) (n := n) i j)
  have hcoord := read.hasFDerivAt.comp q
    (differentiableAt_specLayerNormVec hm hn hε q).hasFDerivAt
  have hfun :
      (read ∘ specLayerNormVec hm hn ε) =
        fun p : CtxVec (ΓLN m n) =>
          RowNorm.nrm (valX p) ε i j * valGamma p j + valBeta p j := by
    funext p
    exact specLayerNormVec_idxMN hm hn hε p i j
  rw [hfun] at hcoord
  have h := congrArg (fun D : CtxVec (ΓLN m n) →L[ℝ] ℝ => D dq)
    (hcoord.unique hentry)
  change
    fderiv ℝ (specLayerNormVec hm hn ε) q dq (idxMN (m := m) (n := n) i j) =
      (RowNorm.nrm (valX q) ε i j * scale dq +
        valGamma q j * RowNorm.nrmD (valX q) ε i j (input dq)) + bias dq at h
  rw [hinput, hscale, hbias, RowNorm.nrmD_apply hn hε] at h
  simpa only [mul_comm, add_comm] using h

/-- The primitive JVP has exactly the row differential obtained by differentiating the forward
formula, with the same epsilon, scale, and bias perturbations. -/
theorem get2_layerNormJvp_eq_row_differential (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (q dq : CtxVec (ΓLN m n)) (i : Fin m) (j : Fin n) :
    Spec.get2
        (Spec.layerNormJvp hm hn (specX q) (specX dq)
          (specGamma q) (specGamma dq) (specBeta q) (specBeta dq) ε) i j =
      RowNorm.nrmJvp (valX q) ε (valX dq) i j * valGamma q j +
        RowNorm.nrm (valX q) ε i j * valGamma dq j + valBeta dq j := by
  rw [layerNormJvp_eq_mat]
  simp only [get2_layerNormJvpMat, Norm.get2_lnCentered, lnInvStd_specX hn hε,
    rowMeanE_specX, ← valX_idxMN, ← valGamma_apply, ← valBeta_apply,
    RowNorm.nrmJvp, RowNorm.nrm, RowNorm.centered, RowNorm.rowMean]

/-- The Fréchet derivative of the actual LayerNorm specification is its supplied JVP.

Both arguments use the graph's `[input, scale, bias]` packing, so the identity can be used
directly in a graph proof without changing the parameter layout. -/
theorem fderiv_specLayerNormVec_eq_layerNormJvp (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (q dq : CtxVec (ΓLN m n)) :
    fderiv ℝ (specLayerNormVec hm hn ε) q dq =
      tensorToVec
        (Spec.layerNormJvp hm hn (specX q) (specX dq)
          (specGamma q) (specGamma dq) (specBeta q) (specBeta dq) ε) := by
  apply Norm.vec_ext_idxMN
  intro i j
  rw [fderiv_specLayerNormVec_idxMN hm hn hε, Norm.tensorToVec_idxMN,
    get2_layerNormJvp_eq_row_differential hm hn hε]

/-- Tensor form of the derivative bridge, with arbitrary input, scale, and bias tangents. -/
theorem fderiv_layerNorm_eq_layerNormJvp (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (x dx : Tensor ℝ [m, n])
    (gamma dgamma beta dbeta : Tensor ℝ [n]) :
    fderiv ℝ (specLayerNormVec hm hn ε) (packLN x gamma beta) (packLN dx dgamma dbeta) =
      tensorToVec (Spec.layerNormJvp hm hn x dx gamma dgamma beta dbeta ε) := by
  rw [fderiv_specLayerNormVec_eq_layerNormJvp hm hn hε,
    specX_packLN, specX_packLN, specGamma_packLN, specGamma_packLN,
    specBeta_packLN, specBeta_packLN]

/-- Every vector in the canonical LayerNorm context packs one input matrix and two parameter
vectors. This also applies to arbitrary test directions in the adjoint proof. -/
theorem exists_packLN (q : CtxVec (ΓLN m n)) :
    ∃ (x : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]), packLN x gamma beta = q := by
  cases h : unflattenCtx q with
  | cons x tail =>
    cases tail with
    | cons gamma tail =>
      cases tail with
      | cons beta tail =>
        cases tail
        exact ⟨x, gamma, beta, by simpa only [h, packLN] using flattenCtx_unflattenCtx q⟩

/-- The packed inner product is the sum of the input, scale, and bias tensor dot products. -/
theorem inner_packLN (x dx : Tensor ℝ [m, n]) (gamma dgamma beta dbeta : Tensor ℝ [n]) :
    inner ℝ (packLN x gamma beta) (packLN dx dgamma dbeta) =
      dot x dx + dot gamma dgamma + dot beta dbeta := by
  rw [packLN, packLN, ← dotList_eq_inner_flattenCtx]
  simp only [TensorPack.dotList, add_zero, add_assoc]

/-- The primitive backward rule is the adjoint of the actual LayerNorm derivative.

The three output tensors occupy the original input, scale, and bias slots. In particular, the
scale and bias cotangents sum over rows because those parameters are shared by every row. -/
theorem adjoint_fderiv_layerNorm_eq_layerNormBackward (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (x gradOutput : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    (fderiv ℝ (specLayerNormVec hm hn ε) (packLN x gamma beta)).adjoint
        (tensorToVec gradOutput) =
      let gradients := Spec.layerNormBackward hm hn x gamma gradOutput ε
      packLN gradients.inputGradient gradients.scaleGradient gradients.biasGradient := by
  apply ext_inner_left ℝ
  intro dq
  obtain ⟨dx, dgamma, dbeta, rfl⟩ := exists_packLN dq
  rw [ContinuousLinearMap.adjoint_inner_right, fderiv_layerNorm_eq_layerNormJvp hm hn hε,
    ← dot_eq_inner_tensorToVec, inner_packLN]
  exact layerNormJvp_layerNormBackward_adjoint hm hn x dx gradOutput gamma dgamma beta dbeta ε

/-- Backpropagating through the detailed LayerNorm graph returns exactly the same three
cotangents as the primitive backward rule. -/
theorem backpropVec_single_eq_layerNormBackward (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (x gradOutput : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    Graph.backpropVec (Γ := ΓLN m n) (ss := ssLayerNorm m n)
        (layerNormGraph (m := m) (n := n) ε) (packLN x gamma beta)
        (CtxVec.single (Γ := ΓLN m n ++ ssLayerNorm m n) (s := MatShape m n)
          (idxY (m := m) (n := n)) (tensorToVec gradOutput)) =
      let gradients := Spec.layerNormBackward hm hn x gamma gradOutput ε
      packLN gradients.inputGradient gradients.scaleGradient gradients.biasGradient := by
  rw [backpropVec_single_eq_adjoint_specLayerNorm hm hn hε,
    adjoint_fderiv_layerNorm_eq_layerNormBackward hm hn hε]

end

end Proofs.Autograd.LayerNorm
