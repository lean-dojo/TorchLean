/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.Reindex
public import TorchLean.NN.Tape.RealNode.Context

/-!
# Continuous tensor vectorization

The tape's flattened vectors and native real tensors use the same inner product, so the tape's
flattening `toVec` is a linear isometry equivalence (`toVecₗᵢ`). Derivatives and
adjoints pass through it without unfolding storage or choosing a second coordinate order. Context
projections then return native tensors, including tensors with empty shapes.

`toVec_apply` lists the entries of the vectorization in row-major order.
`CtxVec.toVec_getTensorCLM` and `CtxVec.getIdx_unflattenCtx` pass between a context block and
the tensor read from it, and `CtxVec.forall_getTensorCLM_iff` carries an entrywise condition across.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open Tensor

noncomputable section

open TensorPack (getIdx)

/-- The tape's flattening of a real tensor, as a linear isometry equivalence onto Euclidean
space. -/
def Tensor.toVecₗᵢ (s : Shape) : Tensor ℝ s ≃ₗᵢ[ℝ] Vec (Shape.size s) where
  toFun := toVec
  invFun := ofVec
  map_add' := toVec_add
  map_smul' r a := by
    rw [RingHom.id_apply]
    apply ext_inner_right ℝ
    intro v
    obtain ⟨c, rfl⟩ := Function.RightInverse.surjective (toVec_ofVec (s := s)) v
    rw [real_inner_smul_left, ← dot_eq_inner_toVec, ← dot_eq_inner_toVec,
      dot_eq_inner, dot_eq_inner, real_inner_smul_left]
  left_inv := ofVec_toVec
  right_inv := toVec_ofVec
  norm_map' a := by
    show ‖toVec a‖ = ‖a‖
    rw [norm_eq_sqrt_real_inner, norm_eq_sqrt_real_inner, ← dot_eq_inner_toVec,
      dot_eq_inner]

/-- The flattening isometry is `toVec`. -/
@[simp] theorem Tensor.toVecₗᵢ_apply {s : Shape} (a : Tensor ℝ s) :
    toVecₗᵢ s a = toVec a :=
  rfl

/-- The inverse of the flattening isometry is `ofVec`. -/
@[simp] theorem Tensor.toVecₗᵢ_symm_apply {s : Shape} (v : Vec (Shape.size s)) :
    (toVecₗᵢ s).symm v = ofVec v :=
  rfl

/-- Read a typed tensor block from a flattened context, as a continuous linear map. -/
def CtxVec.getTensorCLM {Γ : List Shape} {shape : Shape} (input : Idx Γ shape) :
    CtxVec Γ →L[ℝ] Tensor ℝ shape :=
  (toVecₗᵢ shape).symm.toContinuousLinearEquiv.toContinuousLinearMap.comp (getCLM input)

/-- The continuous projection agrees with tensor-pack indexing after reconstruction. -/
@[simp] theorem CtxVec.getTensorCLM_apply {Γ : List Shape} {shape : Shape}
    (input : Idx Γ shape) (x : CtxVec Γ) :
    getTensorCLM input x = getIdx (TensorPack.ofVec x) input := by
  apply (toVecₗᵢ shape).injective
  change toVecₗᵢ shape ((toVecₗᵢ shape).symm (getCLM input x)) = _
  rw [LinearIsometryEquiv.apply_symm_apply, getCLM_apply]
  exact (congrArg (get input) (TensorPack.toVec_ofVec x)).symm.trans
    (get_toVec input (TensorPack.ofVec x))

/-- The entries of the vectorization of a tensor are its entries in row-major order. -/
theorem toVec_apply {s : Shape} (t : Tensor ℝ s) (i : Fin (Shape.size s)) :
    toVec t i = t ((Shape.Coord.equivFin s).symm i) :=
  (getScalarE_ofLp _ i).trans (getScalar_reshape_equivFin t i)

/-- Vectorizing the tensor read from a flattened context gives the block read from it. -/
theorem CtxVec.toVec_getTensorCLM {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (x : CtxVec Γ) : toVec (CtxVec.getTensorCLM idx x) = CtxVec.get idx x :=
  ((toVecₗᵢ s).apply_symm_apply (CtxVec.getCLM idx x)).trans (CtxVec.getCLM_apply idx x)

/-- A condition holds at every entry of the tensor read from a flattened context exactly when it
holds at every coordinate of the block read from it. -/
theorem CtxVec.forall_getTensorCLM_iff {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    (x : CtxVec Γ) (P : ℝ → Prop) :
    (∀ c, P (CtxVec.getTensorCLM idx x c)) ↔ ∀ i, P (CtxVec.get idx x i) := by
  rw [(Shape.Coord.equivFin s).forall_congr_left]
  simp only [← toVec_apply, CtxVec.toVec_getTensorCLM]

/-- Reading a tensor from an unflattened context is reading its block and unvectorizing it. -/
theorem CtxVec.getIdx_unflattenCtx {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (x : CtxVec Γ) :
    getIdx (TensorPack.ofVec x) idx = ofVec (CtxVec.get idx x) := by
  rw [← CtxVec.getTensorCLM_apply, ← CtxVec.toVec_getTensorCLM, ofVec_toVec]

end

end TorchLean
