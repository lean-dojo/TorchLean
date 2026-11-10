/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Norm.LayerNorm

/-!
# LayerNorm derivatives

LayerNorm subtracts each row's mean, divides by `sqrt (variance + ε)`, and applies the
per-column scale and bias. The input, scale, and bias all vary in these theorems, and epsilon is
a fixed positive real number. Neither the scale nor its perturbation is replaced by an all-ones
vector.

The graph calculus proves that `Tensor.layerNorm` is differentiable, and the theorems here identify
its derivative with the `LayerNorm.jvp` formula. The proof differentiates a normalized row
entry, applies the product rule for the scale and adds the bias perturbation. The JVP/VJP pairing of
`TorchLean.NN.Norm.LayerNorm.Adjoint` then identifies `LayerNorm.backward` with the adjoint of the
derivative, including both parameter cotangents. Composing that identification with the
reverse-mode bridge of the graph shows that the three tensors returned by the primitive are the
backward result of the detailed proof graph.

Positive epsilon keeps every row's standard deviation nonzero, including constant rows and
rows with a single feature. All statements here concern exact real arithmetic.
-/

@[expose] public section

namespace TorchLean.LayerNorm

open TorchLean.Tensor
open CtxVec Matmul
open Rep (dot)
open scoped BigOperators

noncomputable section

variable {m n : Nat}

/-- The tensor and vector descriptions use the same row mean. -/
theorem rowMeanE_inputX (q : CtxVec (ctx m n)) (i : Fin m) :
    𝔼 j, Tensor.get2 (inputX q) i j = 𝔼 j, valX q (idxMN (m := m) (n := n) i j) := by
  simp only [valX_idxMN]

/-- The tensor and vector descriptions use the same population variance. -/
theorem rowVarE_inputX (q : CtxVec (ctx m n)) (i : Fin m) :
    TorchLean.var (Tensor.get2 (inputX q) i) =
      TorchLean.var fun j => valX q (idxMN (m := m) (n := n) i j) := by
  simp only [valX_idxMN]

/-- For positive `ε`, the spec's inverse standard deviation `invStd` agrees with
`RowNorm.invStd`, because the standard-deviation clamp is inactive in every row. -/
theorem invStd_inputX {ε : ℝ} (hε : 0 < ε) (q : CtxVec (ctx m n)) (i : Fin m) :
    getScalar (invStd (inputX q) ε) i =
      RowNorm.invStd (valX q) ε i := by
  rw [invStd, Tensor.getScalar_div, getScalar_full, Normalization.getScalar_lnStd,
    rowVarE_inputX, max_eq_left (RowNorm.rowVar_add_pos hε (valX q) i).le, one_div]
  rfl

/-- For positive `n` and `ε`, one output entry is the normalized input times its own scale, plus
its own bias. -/
theorem forwardVec_idxMN (hn : 0 < n) {ε : ℝ} (hε : 0 < ε)
    (q : CtxVec (ctx m n)) (i : Fin m) (j : Fin n) :
    forwardVec ε q (idxMN (m := m) (n := n) i j) =
      RowNorm.nrm (valX q) ε i j * valGamma q j + valBeta q j := by
  rw [forwardVec_eq_valY hn, valY_idxMN, rowVarE_inputX,
    max_eq_left (RowNorm.rowVar_add_pos hε (valX q) i).le,
    ← valX_idxMN, rowMeanE_inputX, ← valGamma_apply, ← valBeta_apply]
  rfl

/-- For positive `n` and `ε`, each entry of the LayerNorm derivative combines the input tangent
and both affine parameter tangents.

The normalized row differential subtracts the tangent's row mean and its component along the
normalized input, and the product rule supplies the extra `xhat * dgamma` term. -/
theorem fderiv_forwardVec_idxMN (hn : 0 < n) {ε : ℝ} (hε : 0 < ε)
    (q dq : CtxVec (ctx m n)) (i : Fin m) (j : Fin n) :
    fderiv ℝ (forwardVec ε) q dq (idxMN (m := m) (n := n) i j) =
      RowNorm.nrmJVP (valX q) ε (valX dq) i j * valGamma q j +
        RowNorm.nrm (valX q) ε i j * valGamma dq j + valBeta dq j := by
  let input := CtxVec.getCLM (Γ := ctx m n) (s := [m, n])
    (idxX (m := m) (n := n) (ss := []))
  let scale := (EuclideanSpace.proj (𝕜 := ℝ) j).comp
    (getVecCLM (Γ := ctx m n) (idxGamma (m := m) (n := n) (ss := [])))
  let bias := (EuclideanSpace.proj (𝕜 := ℝ) j).comp
    (getVecCLM (Γ := ctx m n) (idxBeta (m := m) (n := n) (ss := [])))
  -- Each context projection carries a size cast. Its evaluation lemma identifies the block
  -- without expanding the recursive context representation during the derivative calculation.
  have hinput (p : CtxVec (ctx m n)) : input p = valX p :=
    CtxVec.getCLM_apply (Γ := ctx m n) (idxX (m := m) (n := n) (ss := [])) p
  have hscale (p : CtxVec (ctx m n)) : scale p = valGamma p j := by
    change (getVecCLM (Γ := ctx m n) (idxGamma (m := m) (n := n) (ss := [])) p) j =
      valGamma p j
    exact congrArg (fun v : Vec n => v j)
      (getVecCLM_apply (Γ := ctx m n) (idxGamma (m := m) (n := n) (ss := [])) p)
  have hbias (p : CtxVec (ctx m n)) : bias p = valBeta p j := by
    change (getVecCLM (Γ := ctx m n) (idxBeta (m := m) (n := n) (ss := [])) p) j =
      valBeta p j
    exact congrArg (fun v : Vec n => v j)
      (getVecCLM_apply (Γ := ctx m n) (idxBeta (m := m) (n := n) (ss := [])) p)
  have hX : HasFDerivAt (fun p : CtxVec (ctx m n) => valX p) input q :=
    input.hasFDerivAt.congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun p => (hinput p).symm)
  have hg : HasFDerivAt (fun p : CtxVec (ctx m n) => valGamma p j) scale q :=
    scale.hasFDerivAt.congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun p => (hscale p).symm)
  have hb : HasFDerivAt (fun p : CtxVec (ctx m n) => valBeta p j) bias q :=
    bias.hasFDerivAt.congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun p => (hbias p).symm)
  have hentry := (((RowNorm.hasFDerivAt_nrm hε (valX q) i j).comp q hX).mul hg).add hb
  let read := EuclideanSpace.proj (𝕜 := ℝ) (idxMN (m := m) (n := n) i j)
  have hcoord := read.hasFDerivAt.comp q
    (differentiableAt_forwardVec hn hε q).hasFDerivAt
  have hfun :
      (read ∘ forwardVec ε) =
        fun p : CtxVec (ctx m n) =>
          RowNorm.nrm (valX p) ε i j * valGamma p j + valBeta p j := by
    funext p
    exact forwardVec_idxMN hn hε p i j
  rw [hfun] at hcoord
  have h := congrArg (fun D : CtxVec (ctx m n) →L[ℝ] ℝ => D dq)
    (hcoord.unique hentry)
  change
    fderiv ℝ (forwardVec ε) q dq (idxMN (m := m) (n := n) i j) =
      (RowNorm.nrm (valX q) ε i j * scale dq +
        valGamma q j * RowNorm.nrmD (valX q) ε i j (input dq)) + bias dq at h
  rw [hinput, hscale, hbias, RowNorm.nrmD_apply hn hε] at h
  simpa only [mul_comm, add_comm] using h

/-- For positive `m`, `n` and `ε`, each entry of `LayerNorm.jvp` is the row differential
that `fderiv_forwardVec_idxMN` obtains from the forward formula, with the same epsilon and
the same scale and bias perturbations. -/
theorem get2_jvp_eq_row_differential (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (q dq : CtxVec (ctx m n)) (i : Fin m) (j : Fin n) :
    Tensor.get2
        (jvp hm hn (inputX q) (inputX dq)
          (inputGamma q) (inputGamma dq) (inputBeta q) (inputBeta dq) ε) i j =
      RowNorm.nrmJVP (valX q) ε (valX dq) i j * valGamma q j +
        RowNorm.nrm (valX q) ε i j * valGamma dq j + valBeta dq j := by
  rw [jvp_eq_jvpMat]
  simp only [get2_jvpMat, Normalization.get2_lnCentered, invStd_inputX hε, ← valX_idxMN,
    ← valGamma_apply, ← valBeta_apply, RowNorm.nrmJVP, RowNorm.nrm, RowNorm.centered,
    Fintype.expect_eq_sum_div_card, Fintype.card_fin]

/-- For positive `m`, `n` and `ε`, the Fréchet derivative of the LayerNorm specification is its
JVP `LayerNorm.jvp`.

Both arguments use the graph's `[input, scale, bias]` packing, so a graph proof can use the
identity without changing the parameter layout. -/
theorem fderiv_forwardVec_eq_jvp (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (q dq : CtxVec (ctx m n)) :
    fderiv ℝ (forwardVec ε) q dq =
      toVec
        (jvp hm hn (inputX q) (inputX dq)
          (inputGamma q) (inputGamma dq) (inputBeta q) (inputBeta dq) ε) := by
  apply Normalization.vec_ext_idxMN
  intro i j
  rw [fderiv_forwardVec_idxMN hn hε, Matmul.toVec_idxMN,
    get2_jvp_eq_row_differential hm hn hε]

/-- Tensor form of the derivative bridge, with arbitrary input, scale, and bias tangents. -/
theorem fderiv_eq_jvp (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (x dx : Tensor ℝ [m, n])
    (gamma dgamma beta dbeta : Tensor ℝ [n]) :
    fderiv ℝ (forwardVec ε) (pack x gamma beta) (pack dx dgamma dbeta) =
      toVec (jvp hm hn x dx gamma dgamma beta dbeta ε) := by
  rw [fderiv_forwardVec_eq_jvp hm hn hε,
    inputX_pack, inputX_pack, inputGamma_pack, inputGamma_pack,
    inputBeta_pack, inputBeta_pack]

/-- Every vector in the canonical LayerNorm context packs one input matrix and two parameter
vectors, so the adjoint proof can unpack an arbitrary test direction. -/
theorem exists_pack (q : CtxVec (ctx m n)) :
    ∃ (x : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]), pack x gamma beta = q := by
  cases h : TensorPack.ofVec q with
  | cons x tail =>
    cases tail with
    | cons gamma tail =>
      cases tail with
      | cons beta tail =>
        cases tail
        exact ⟨x, gamma, beta, by simpa only [h, pack] using TensorPack.toVec_ofVec q⟩

/-- The packed inner product is the sum of the input, scale, and bias tensor dot products. -/
theorem inner_pack (x dx : Tensor ℝ [m, n]) (gamma dgamma beta dbeta : Tensor ℝ [n]) :
    inner ℝ (pack x gamma beta) (pack dx dgamma dbeta) =
      Rep.dot x dx + Rep.dot gamma dgamma + Rep.dot beta dbeta := by
  rw [pack, pack, ← TensorPack.realDotList_eq_inner_toVec]
  simp only [TensorPack.dotList, add_zero, add_assoc]

/-- For positive `m`, `n` and `ε`, the primitive backward rule is the adjoint of the LayerNorm
derivative.

The three output tensors fill the input, scale, and bias slots of the packed context, and the
scale and bias cotangents sum over rows because every row shares those parameters. -/
theorem adjoint_fderiv_eq_backward (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (x gradOutput : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    (fderiv ℝ (forwardVec ε) (pack x gamma beta)).adjoint
        (toVec gradOutput) =
      let gradients := backward hm hn x gamma gradOutput ε
      pack gradients.inputGradient gradients.scaleGradient gradients.biasGradient := by
  apply ext_inner_left ℝ
  intro dq
  obtain ⟨dx, dgamma, dbeta, rfl⟩ := exists_pack dq
  rw [ContinuousLinearMap.adjoint_inner_right, fderiv_eq_jvp hm hn hε,
    ← dot_eq_inner_toVec, inner_pack]
  exact jvp_backward_adjoint hm hn x dx gradOutput gamma dgamma beta dbeta ε

/-- For positive `m`, `n` and `ε`, backpropagating a cotangent seeded on the output block
through the detailed LayerNorm graph returns the three cotangents of the primitive backward
rule. -/
theorem backprop_eq_backward (hm : 0 < m) (hn : 0 < n)
    {ε : ℝ} (hε : 0 < ε) (x gradOutput : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    Tape.GraphData.backpropVec (Γ := ctx m n) (ss := LayerNorm.ss m n)
        (graph (m := m) (n := n) ε).toData () (pack x gamma beta)
        (CtxVec.single (Γ := ctx m n ++ LayerNorm.ss m n) (s := [m, n])
          (idxY (m := m) (n := n)) (toVec gradOutput)) =
      let gradients := backward hm hn x gamma gradOutput ε
      pack gradients.inputGradient gradients.scaleGradient gradients.biasGradient := by
  rw [backprop_eq_fderiv_forwardVec hn hε,
    adjoint_fderiv_eq_backward hm hn hε]

end

end TorchLean.LayerNorm
