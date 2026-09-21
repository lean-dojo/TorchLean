/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Reindex
public import NN.Proofs.Autograd.Tape.Nodes.Context

/-!
# Continuous tensor vectorization

The tape's flattened vectors and native real tensors use the same inner product. Bundling the
existing conversion as a continuous linear equivalence lets derivatives pass through it without
unfolding storage or choosing a second coordinate order. Context projections then return native
tensors, including tensors with empty shapes.
-/

@[expose] public section

namespace Proofs.Autograd

open Spec TorchLean

noncomputable section

/-- The tape's existing tensor conversion as a continuous linear equivalence. -/
def tensorVecEquiv (shape : Shape) : Tensor ℝ shape ≃L[ℝ] Vec (Shape.size shape) := by
  let linear : Tensor ℝ shape →ₗ[ℝ] Vec (Shape.size shape) :=
    { toFun := tensorToVec
      map_add' := by
        intro a b
        apply ext_inner_left ℝ
        intro v
        obtain ⟨c, rfl⟩ := Function.RightInverse.surjective tensorToVec_vecToTensor v
        simp only [inner_add_right, ← dot_eq_inner_tensorToVec, Reindex.dot_eq_inner]
      map_smul' := by
        intro r a
        apply ext_inner_left ℝ
        intro v
        obtain ⟨c, rfl⟩ := Function.RightInverse.surjective tensorToVec_vecToTensor v
        simp only [inner_smul_right, ← dot_eq_inner_tensorToVec, Reindex.dot_eq_inner]
        rfl }
  exact (LinearEquiv.ofBijective linear
    ⟨Function.LeftInverse.injective vecToTensor_tensorToVec,
      Function.RightInverse.surjective tensorToVec_vecToTensor⟩).toContinuousLinearEquiv

/-- The bundled map preserves the tape's flattening order. -/
@[simp] theorem tensorVecEquiv_apply {shape : Shape} (x : Tensor ℝ shape) :
    tensorVecEquiv shape x = tensorToVec x := rfl

/-- Its inverse uses the existing tensor reconstruction. -/
@[simp] theorem tensorVecEquiv_symm_apply {shape : Shape} (v : Vec (Shape.size shape)) :
    (tensorVecEquiv shape).symm v = vecToTensor v := by
  apply (tensorVecEquiv shape).injective
  simp only [ContinuousLinearEquiv.apply_symm_apply, tensorVecEquiv_apply,
    tensorToVec_vecToTensor]

/-- Read a typed tensor block directly from a flattened context. -/
def CtxVec.getTensorCLM {Γ : List Shape} {shape : Shape} (input : Idx Γ shape) :
    CtxVec Γ →L[ℝ] Tensor ℝ shape :=
  (tensorVecEquiv shape).symm.toContinuousLinearMap.comp (getCLM input)

/-- The continuous projection agrees with tensor-pack indexing after reconstruction. -/
@[simp] theorem CtxVec.getTensorCLM_apply {Γ : List Shape} {shape : Shape}
    (input : Idx Γ shape) (x : CtxVec Γ) :
    getTensorCLM input x = getIdx (unflattenCtx x) input := by
  apply (tensorVecEquiv shape).injective
  change tensorVecEquiv shape ((tensorVecEquiv shape).symm (getCLM input x)) = _
  rw [ContinuousLinearEquiv.apply_symm_apply, getCLM_apply]
  exact (congrArg (get input) (flattenCtx_unflattenCtx x)).symm.trans
    (get_flattenCtx input (unflattenCtx x))

end

end Proofs.Autograd
