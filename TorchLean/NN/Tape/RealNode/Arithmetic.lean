/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealNode.Context
public import TorchLean.NN.FDeriv.Elementwise

/-!
# Affine tape node

The affine node `affine` applies a fixed continuous linear map and a bias to one context entry,
with its `FDerivCorrect` certificate `affineFDeriv`. We take pointwise arithmetic from the
operator arms of `Tape.LawfulNode.ofArm` and their `op_cert` certificates.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open TorchLean.Tensor

noncomputable section

open scoped BigOperators

namespace Tape.RealNode

/--
Fixed affine node over arbitrary tensor shapes.

It models a linear layer once all leading dimensions have been flattened into the vector
representation used by `CtxVec`. Because the shapes are arbitrary, an unbatched `Linear`, a
position-wise Transformer FFN map over a whole sequence, or any other fixed affine map can
instantiate the same theorem with the appropriate continuous linear map `A` and bias vector `b`.
-/
def affine {Γ : List Shape} {sIn sOut : Shape}
    (idx : Idx Γ sIn) (A : Vec (Shape.size sIn) →L[ℝ] Vec (Shape.size sOut))
    (b : Vec (Shape.size sOut)) : LawfulNode (α := ℝ) Unit Γ sOut :=
  LawfulNode.ofFn (Γ := Γ) (τ := sOut)
    (f := fun x => A (CtxVec.get (Γ := Γ) (s := sIn) idx x) + b)
    (jvp := fun _x dx => A (CtxVec.get (Γ := Γ) (s := sIn) idx dx))
    (vjp := fun _x δ => CtxVec.single (Γ := Γ) (s := sIn) idx (A.adjoint δ))
    (correct_inner := by
      intro _x dx δ
      rw [CtxVec.inner_get_single]
      simpa using
        (ContinuousLinearMap.adjoint_inner_right (A := A)
          (x := CtxVec.get (Γ := Γ) (s := sIn) idx dx) (y := δ)).symm)

/-- `FDerivCorrect` for a fixed affine map over arbitrary tensor shapes. -/
def affineFDeriv {Γ : List Shape} {sIn sOut : Shape}
    (idx : Idx Γ sIn) (A : Vec (Shape.size sIn) →L[ℝ] Vec (Shape.size sOut))
    (b : Vec (Shape.size sOut)) :
    FDerivCorrect (affine (Γ := Γ) (sIn := sIn) (sOut := sOut) idx A b).toNodeData () :=
  { deriv := fun _x => A.comp (CtxVec.getCLM (Γ := Γ) (s := sIn) idx)
    hasFDerivAt := by
      intro xV
      have hlin :
          HasFDerivAt
            (fun x : CtxVec Γ => A (CtxVec.get (Γ := Γ) (s := sIn) idx x))
            (A.comp (CtxVec.getCLM (Γ := Γ) (s := sIn) idx)) xV :=
        A.hasFDerivAt.comp xV (CtxVec.hasFDerivAt_get (Γ := Γ) (s := sIn) idx xV)
      simpa [affine, LawfulNode.forwardVec_ofFn, ContinuousLinearMap.comp_apply] using
        hlin.const_add b
    jvp_eq := by
      intro xV dxV
      simp [affine, LawfulNode.jvpVec_ofFn, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply] }

end Tape.RealNode

end

end TorchLean
