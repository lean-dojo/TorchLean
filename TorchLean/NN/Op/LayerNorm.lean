/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Norm.LayerNorm.FDeriv
public import TorchLean.NN.FDeriv.TensorVectorization
public import TorchLean.Tensor.Op.Cert
public import TorchLean.Tensor.Op.Table
public import TorchLean.Tensor.Euclidean
public import Mathlib.Analysis.InnerProductSpace.Adjoint
public import Mathlib.Analysis.InnerProductSpace.ProdL2

/-!
# The derivative of layer normalization in its input, scale and bias

`Op.layerNorm` reshapes its input to a `(seqLen, embedDim)` matrix, normalizes each row, and
reshapes the result back. Over the reals a reshape is a linear isometry equivalence for the
Euclidean inner product (`TorchLean.Tensor.reshapeₗᵢ`), and so are the tape's flattening of a
tensor (`TorchLean.Tensor.toVecₗᵢ`) and the packing of a matrix, a scale and a bias into the
LayerNorm context vector (`TorchLean.LayerNorm.packₗᵢ`). Each has its inverse as adjoint,
so the matrix statement `LayerNorm.adjoint_fderiv_eq_backward` carries over to the arm at
every leading shape (`TorchLean.Op.layerNorm_prod_backward_eq_adjoint`). Epsilon is fixed and
positive throughout.
-/

@[expose] public section

open scoped RealInnerProductSpace
open TorchLean.Shape (size)

namespace TorchLean.Tensor

variable {s : Shape}

/-- Reshaping a real tensor to a shape with as many entries, as a linear isometry equivalence for
the Euclidean inner product. -/
noncomputable def reshapeₗᵢ (t : Shape) (h : s.size = t.size) : Tensor ℝ s ≃ₗᵢ[ℝ] Tensor ℝ t where
  toFun x := reshape x t h
  invFun y := reshape y s h.symm
  map_add' x y := Rep.ext fun c => by
    simp only [reshape_apply, Tensor.add_apply]
  map_smul' r x := Rep.ext fun c => by
    simp only [reshape_apply, Rep.smul_apply, RingHom.id_apply]
  left_inv x := reshape_reshape_self x h h.symm
  right_inv y := reshape_reshape_self y h.symm h
  norm_map' x := by
    show ‖reshape x t h‖ = ‖x‖
    have happly (c : t.Coord) :
        reshape x t h c = x (Shape.Coord.reshapeEquiv h c) :=
      Rep.reshape_apply_reshapeEquiv _ x c
    rw [norm_eq_sqrt_sum, norm_eq_sqrt_sum]
    simp only [happly]
    rw [(Shape.Coord.reshapeEquiv h).sum_comp fun c => x c ^ 2]

/-- The reshape isometry reshapes. -/
@[simp] theorem reshapeₗᵢ_apply (t : Shape) (h : s.size = t.size) (x : Tensor ℝ s) :
    reshapeₗᵢ t h x = reshape x t h :=
  rfl

/-- The inverse of the reshape isometry reshapes back. -/
@[simp] theorem reshapeₗᵢ_symm_apply (t : Shape) (h : s.size = t.size) (y : Tensor ℝ t) :
    (reshapeₗᵢ t h).symm y = reshape y s h.symm :=
  rfl

end TorchLean.Tensor

namespace TorchLean

namespace LayerNorm

variable {m n : Nat}

/-- Packing is additive in the input matrix, the scale and the bias together. -/
theorem pack_add (x x' : Tensor ℝ [m, n]) (gamma gamma' beta beta' : Tensor ℝ [n]) :
    pack (x + x') (gamma + gamma') (beta + beta') =
      pack x gamma beta + pack x' gamma' beta' := by
  apply ext_inner_right ℝ
  intro q
  obtain ⟨dx, dgamma, dbeta, rfl⟩ := exists_pack q
  simp only [inner_add_left, inner_pack, Tensor.dot_eq_inner]
  ring

/-- Packing commutes with scaling the input matrix, the scale and the bias together. -/
theorem pack_smul (r : ℝ) (x : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    pack (r • x) (r • gamma) (r • beta) = r • pack x gamma beta := by
  apply ext_inner_right ℝ
  intro q
  obtain ⟨dx, dgamma, dbeta, rfl⟩ := exists_pack q
  simp only [real_inner_smul_left, inner_pack, Tensor.dot_eq_inner]
  ring

/-- Packing an input matrix, a scale and a bias into the LayerNorm context vector, as a linear
isometry equivalence from their L2 product. -/
noncomputable def packₗᵢ :
    WithLp 2 (Tensor ℝ [m, n] × WithLp 2 (Tensor ℝ [n] × Tensor ℝ [n])) ≃ₗᵢ[ℝ]
      CtxVec (ctx m n) where
  toFun p := pack p.fst p.snd.fst p.snd.snd
  invFun q := WithLp.toLp 2 (inputX q, WithLp.toLp 2 (inputGamma q, inputBeta q))
  map_add' p p' := pack_add _ _ _ _ _ _
  map_smul' r p := pack_smul r _ _ _
  left_inv p := by
    simp only [inputX_pack, inputGamma_pack, inputBeta_pack]
    rfl
  right_inv q := by
    obtain ⟨x, gamma, beta, rfl⟩ := exists_pack q
    show pack (inputX (pack x gamma beta)) (inputGamma (pack x gamma beta))
      (inputBeta (pack x gamma beta)) = pack x gamma beta
    rw [inputX_pack, inputGamma_pack, inputBeta_pack]
  norm_map' p := by
    show ‖pack p.fst p.snd.fst p.snd.snd‖ = ‖p‖
    rw [norm_eq_sqrt_real_inner, norm_eq_sqrt_real_inner, inner_pack, WithLp.prod_inner_apply,
      WithLp.prod_inner_apply, Tensor.dot_eq_inner, Tensor.dot_eq_inner,
      Tensor.dot_eq_inner, add_assoc]
    rfl

/-- The packing isometry is `pack` of the three components. -/
@[simp] theorem packₗᵢ_apply
    (p : WithLp 2 (Tensor ℝ [m, n] × WithLp 2 (Tensor ℝ [n] × Tensor ℝ [n]))) :
    packₗᵢ p = pack p.fst p.snd.fst p.snd.snd :=
  rfl

/-- On a packed context, `forwardVec` is the flattened `Tensor.layerNorm` of the input matrix
over its last axis. -/
theorem forwardVec_pack (ε : ℝ) (x : Tensor ℝ [m, n]) (gamma beta : Tensor ℝ [n]) :
    forwardVec ε (pack x gamma beta) =
      Tensor.toVec (Tensor.layerNorm x [⟨1, Nat.one_lt_two⟩] gamma beta ε) := by
  rw [forwardVec, inputX_pack]
  erw [inputGamma_pack, inputBeta_pack]

end LayerNorm

end TorchLean

namespace TorchLean.Op

open Tensor LayerNorm

/-- With a fixed positive epsilon, the forward of `Op.layerNorm` is differentiable in its input,
scale and bias on their L2 product, and its backward is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.layerNorm derivative 0 1 2]
theorem layerNorm_prod_backward_eq_adjoint {seqLen embedDim : Nat} (hSeq : seqLen > 0)
    (hEmbed : embedDim > 0) {lead x : Shape} (hlead : seqLen = lead.size)
    (hx : x = lead ++ [embedDim]) (input : Tensor ℝ x) (gamma beta : Tensor ℝ [embedDim])
    (epsilon : Tensor ℝ []) (hε : 0 < epsilon.item) (δ : Tensor ℝ x) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ x ×
          WithLp 2 (Tensor ℝ [embedDim] × Tensor ℝ [embedDim])) =>
        (((Op.layerNorm hSeq hEmbed hlead hx).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, epsilon, ())).1)
      (WithLp.toLp 2 (input, WithLp.toLp 2 (gamma, beta))) ∧
    WithLp.toLp 2
        ((((Op.layerNorm hSeq hEmbed hlead hx).rule? (α := ℝ) |>.get rfl).backward
          (input, gamma, beta, epsilon, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.layerNorm hSeq hEmbed hlead hx).rule? (α := ℝ) |>.get rfl).backward
            (input, gamma, beta, epsilon, ()) (δ, ())).2.1,
          (((Op.layerNorm hSeq hEmbed hlead hx).rule? (α := ℝ) |>.get rfl).backward
            (input, gamma, beta, epsilon, ()) (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ x ×
          WithLp 2 (Tensor ℝ [embedDim] × Tensor ℝ [embedDim])) =>
        (((Op.layerNorm hSeq hEmbed hlead hx).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd, epsilon, ())).1)
        (WithLp.toLp 2 (input, WithLp.toLp 2 (gamma, beta)))).adjoint δ := by
  have hs : x.size = Shape.size [seqLen, embedDim] := Shape.size_eq_of_append hlead hx
  let pack := (LinearIsometryEquiv.withLpProdCongr 2 (reshapeₗᵢ [seqLen, embedDim] hs)
    (LinearIsometryEquiv.refl ℝ (WithLp 2 (Tensor ℝ [embedDim] × Tensor ℝ [embedDim])))).trans
      packₗᵢ
  let out := (toVecₗᵢ [seqLen, embedDim]).symm.trans (reshapeₗᵢ x hs.symm)
  have hF : (fun p : WithLp 2 (Tensor ℝ x ×
        WithLp 2 (Tensor ℝ [embedDim] × Tensor ℝ [embedDim])) =>
      (((Op.layerNorm hSeq hEmbed hlead hx).rule? (α := ℝ) |>.get rfl).forward
        (p.fst, p.snd.fst, p.snd.snd, epsilon, ())).1) =
      fun p => out (forwardVec epsilon.item (pack p)) := by
    funext p
    show Tensor.reshape (Tensor.layerNorm (Tensor.reshape p.fst [seqLen, embedDim] hs)
        [⟨1, Nat.one_lt_two⟩] p.snd.fst p.snd.snd epsilon.item) x hs.symm =
      Tensor.reshape (ofVec (forwardVec epsilon.item
        (LayerNorm.pack (Tensor.reshape p.fst [seqLen, embedDim] hs) p.snd.fst p.snd.snd))) x
          hs.symm
    rw [forwardVec_pack, ofVec_toVec]
  set point := WithLp.toLp 2 (input, WithLp.toLp 2 (gamma, beta))
  have hpack : HasFDerivAt pack (pack : WithLp 2 (Tensor ℝ x ×
      WithLp 2 (Tensor ℝ [embedDim] × Tensor ℝ [embedDim])) →L[ℝ] CtxVec (ctx seqLen embedDim))
      point :=
    (pack : WithLp 2 (Tensor ℝ x × WithLp 2 (Tensor ℝ [embedDim] × Tensor ℝ [embedDim]))
      →L[ℝ] CtxVec (ctx seqLen embedDim)).hasFDerivAt
  have hout (v : Vec (Shape.size [seqLen, embedDim])) :
      HasFDerivAt out (out : Vec (Shape.size [seqLen, embedDim]) →L[ℝ] Tensor ℝ x) v :=
    (out : Vec (Shape.size [seqLen, embedDim]) →L[ℝ] Tensor ℝ x).hasFDerivAt
  have hD := (differentiableAt_forwardVec hEmbed hε (pack point)).hasFDerivAt
  have hG : HasFDerivAt (fun p => out (forwardVec epsilon.item (pack p)))
      ((out : Vec (Shape.size [seqLen, embedDim]) →L[ℝ] Tensor ℝ x).comp
        ((fderiv ℝ (forwardVec epsilon.item) (pack point)).comp
          (pack : WithLp 2 (Tensor ℝ x × WithLp 2 (Tensor ℝ [embedDim] × Tensor ℝ [embedDim]))
            →L[ℝ] CtxVec (ctx seqLen embedDim)))) point :=
    (hout _).comp point (hD.comp point hpack)
  rw [hF]
  refine ⟨hG.differentiableAt, ?_⟩
  rw [hG.fderiv, ContinuousLinearMap.adjoint_comp, ContinuousLinearMap.adjoint_comp,
    LinearIsometryEquiv.adjoint_eq_symm, LinearIsometryEquiv.adjoint_eq_symm]
  have hpoint : pack point =
    LayerNorm.pack (Tensor.reshape input [seqLen, embedDim] hs) gamma beta := rfl
  have hδ : out.symm δ = toVec (Tensor.reshape δ [seqLen, embedDim] hs) := rfl
  simp only [ContinuousLinearMap.comp_apply, LinearIsometryEquiv.coe_coe'', hδ]
  rw [hpoint, adjoint_fderiv_eq_backward hSeq hEmbed hε,
    LinearIsometryEquiv.eq_symm_apply]
  show LayerNorm.pack (Tensor.reshape (Tensor.reshape _ x hs.symm) [seqLen, embedDim] hs) _ _ = _
  rw [reshape_reshape_self]
  rfl

end TorchLean.Op
