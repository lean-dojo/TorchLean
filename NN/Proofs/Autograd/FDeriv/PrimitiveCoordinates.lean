/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.PrimitiveSpecs

/-!
# Tensor coordinates for primitive derivatives

Flattening preserves the coordinate formulas of pointwise tensor operations. These lemmas connect
the actual tensor functions to the diagonal calculus in `PrimitiveSpecs`, without restricting the
tensor to one axis. A binary operation with a captured right argument then needs only the scalar
derivative with respect to its left argument.
-/

@[expose] public section

namespace Proofs.Autograd.PrimitiveSpecs

open Spec TorchLean TorchLean.Tensor

noncomputable section

/-- Coordinate evaluation removes the Euclidean wrapper around a pointwise function. -/
@[simp] theorem coordinatewise_apply {n : Nat} (f : Fin n → ℝ → ℝ)
    (x : Vec n) (i : Fin n) :
    coordinatewise f x i = f i (x i) := by
  simp [coordinatewise, euclideanEquiv]

/-- Row-major flattening commutes with a unary pointwise map. -/
@[simp] theorem tensorToVec_mapSpec_apply {s : Shape} (f : ℝ → ℝ)
    (x : Tensor ℝ s) (i : Fin (Shape.size s)) :
    tensorToVec (mapSpec f x) i = f (tensorToVec x i) := by
  change (getScalarE (flattenSpec (mapSpec f x))).ofLp i =
    f ((getScalarE (flattenSpec x)).ofLp i)
  rw [getScalarE_ofLp, getScalarE_ofLp, getScalar_eq_apply, getScalar_eq_apply]
  simp only [flattenSpec, Internal.Rep.reshape_apply_coordEquiv, mapSpec, Tensor.map,
    Internal.Rep.map_apply]

/-- Both operands of a binary pointwise map use the same flattened coordinate. -/
@[simp] theorem tensorToVec_map2Spec_apply {s : Shape} (f : ℝ → ℝ → ℝ)
    (x y : Tensor ℝ s) (i : Fin (Shape.size s)) :
    tensorToVec (map2Spec f x y) i = f (tensorToVec x i) (tensorToVec y i) := by
  change (getScalarE (flattenSpec (map2Spec f x y))).ofLp i =
    f ((getScalarE (flattenSpec x)).ofLp i) ((getScalarE (flattenSpec y)).ofLp i)
  rw [getScalarE_ofLp, getScalarE_ofLp, getScalarE_ofLp, getScalar_eq_apply,
    getScalar_eq_apply, getScalar_eq_apply]
  simp only [flattenSpec, Internal.Rep.reshape_apply_coordEquiv, map2Spec_apply]

/-- Every coordinate of a filled tensor is its fill value. -/
@[simp] theorem tensorToVec_full_apply (s : Shape) (value : ℝ)
    (i : Fin (Shape.size s)) :
    tensorToVec (Tensor.full s value) i = value := by
  change (getScalarE (flattenSpec (Tensor.full s value))).ofLp i = value
  rw [getScalarE_ofLp, getScalar_eq_apply]
  simp only [flattenSpec, Internal.Rep.reshape_apply_coordEquiv, Tensor.full_apply]

/-- Elementwise multiplication becomes multiplication of the corresponding real coordinates. -/
@[simp] theorem tensorToVec_mulSpec_apply {s : Shape} (x y : Tensor ℝ s)
    (i : Fin (Shape.size s)) :
    tensorToVec (mulSpec x y) i = tensorToVec x i * tensorToVec y i :=
  tensorToVec_map2Spec_apply (· * ·) x y i

/-- The captured-right constructor differentiates its actual tensor forward function. -/
theorem binaryElemOp_hasFDerivAt {s : Shape} (rhs : Tensor ℝ s)
    (f df : ℝ → ℝ → ℝ) (x : Vec (Shape.size s))
    (hf : ∀ i, HasDerivAt (fun z => f z (tensorToVec rhs i))
      (df (x i) (tensorToVec rhs i)) (x i)) :
    HasFDerivAt
      (fun y => tensorToVec ((Spec.binaryElemOp rhs f df).forward (vecToTensor y)))
      (coordinateDeriv fun i => df (x i) (tensorToVec rhs i)) x := by
  apply hasFDerivAt_of_coordinates (f := fun i z => f z (tensorToVec rhs i))
  · intro y
    ext i
    simp [Spec.binaryElemOp]
  · exact hf

/-- Scalar derivative evidence certifies the stored captured-right backward formula. -/
theorem binaryElemOp_backward_eq_adjoint {s : Shape} (rhs : Tensor ℝ s)
    (f df : ℝ → ℝ → ℝ) (x : Vec (Shape.size s))
    (hf : ∀ i, HasDerivAt (fun z => f z (tensorToVec rhs i))
      (df (x i) (tensorToVec rhs i)) (x i))
    (δ : Tensor ℝ s) :
    tensorToVec ((Spec.binaryElemOp rhs f df).backward (vecToTensor x) δ) =
      (fderiv ℝ
        (fun y => tensorToVec ((Spec.binaryElemOp rhs f df).forward (vecToTensor y)))
        x).adjoint (tensorToVec δ) := by
  apply backward_eq_adjoint_of_coordinates
    (f := fun i z => f z (tensorToVec rhs i))
    (coefficients := fun i => df (x i) (tensorToVec rhs i))
  · intro y
    ext i
    simp [Spec.binaryElemOp]
  · exact hf
  · intro tangent
    ext i
    simp [Spec.binaryElemOp]

end

end Proofs.Autograd.PrimitiveSpecs
