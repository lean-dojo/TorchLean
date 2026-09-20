/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.PrimitiveCoordinates

/-!
# Captured arithmetic derivatives

These theorems certify the actual forward and backward functions of the arithmetic `OpSpec`s.
The right-hand tensor is fixed, so every derivative is diagonal. This is the input-gradient
contract of these specifications; differentiating a second variable requires the corresponding
two-input rule.

Over the reals, division by a fixed denominator is a linear map even when that denominator is
zero: the totalized quotient is then constantly zero. The same fact applies to the fixed shifted
denominator in `safeDivOp`. It does not remove the pole of `safeInvOp`, whose denominator varies
with the differentiated input, or change the nonzero-denominator contract of floating-point
division.
-/

@[expose] public section

namespace Proofs.Autograd.PrimitiveSpecs

open Spec TorchLean

noncomputable section

/-- Adding a captured tensor has the identity coordinate derivative on every shape. -/
theorem addOp_hasFDerivAt {s : Shape} (rhs : Tensor ℝ s) (x : Vec (Shape.size s)) :
    HasFDerivAt (fun y => tensorToVec ((Spec.addOp rhs).forward (vecToTensor y)))
      (coordinateDeriv fun _ => 1) x :=
  binaryElemOp_hasFDerivAt rhs (· + ·) (fun _ _ => 1) x
    (fun i => (hasDerivAt_id (x i)).add_const (tensorToVec rhs i))

/-- The captured addition backward is the adjoint of its forward derivative. -/
theorem addOp_backward_eq_adjoint {s : Shape} (rhs : Tensor ℝ s)
    (x : Vec (Shape.size s)) (δ : Tensor ℝ s) :
    tensorToVec ((Spec.addOp rhs).backward (vecToTensor x) δ) =
      (fderiv ℝ (fun y => tensorToVec ((Spec.addOp rhs).forward (vecToTensor y)))
        x).adjoint (tensorToVec δ) :=
  binaryElemOp_backward_eq_adjoint rhs (· + ·) (fun _ _ => 1) x
    (fun i => (hasDerivAt_id (x i)).add_const (tensorToVec rhs i)) δ

/-- Subtracting a fixed tensor leaves the input derivative equal to the identity. -/
theorem subOp_hasFDerivAt {s : Shape} (rhs : Tensor ℝ s) (x : Vec (Shape.size s)) :
    HasFDerivAt (fun y => tensorToVec ((Spec.subOp rhs).forward (vecToTensor y)))
      (coordinateDeriv fun _ => 1) x :=
  binaryElemOp_hasFDerivAt rhs (· - ·) (fun _ _ => 1) x
    (fun i => (hasDerivAt_id (x i)).sub_const (tensorToVec rhs i))

/-- The captured subtraction backward has the positive input-gradient sign. -/
theorem subOp_backward_eq_adjoint {s : Shape} (rhs : Tensor ℝ s)
    (x : Vec (Shape.size s)) (δ : Tensor ℝ s) :
    tensorToVec ((Spec.subOp rhs).backward (vecToTensor x) δ) =
      (fderiv ℝ (fun y => tensorToVec ((Spec.subOp rhs).forward (vecToTensor y)))
        x).adjoint (tensorToVec δ) :=
  binaryElemOp_backward_eq_adjoint rhs (· - ·) (fun _ _ => 1) x
    (fun i => (hasDerivAt_id (x i)).sub_const (tensorToVec rhs i)) δ

/-- Multiplication by a fixed tensor scales each tangent by the matching captured entry. -/
theorem mulOp_hasFDerivAt {s : Shape} (rhs : Tensor ℝ s) (x : Vec (Shape.size s)) :
    HasFDerivAt (fun y => tensorToVec ((Spec.mulOp rhs).forward (vecToTensor y)))
      (coordinateDeriv fun i => tensorToVec rhs i) x :=
  binaryElemOp_hasFDerivAt rhs (· * ·) (fun _ y => y) x
    (fun i => hasDerivAt_mul_const (tensorToVec rhs i))

/-- The stored multiplication backward is the adjoint of the diagonal forward derivative. -/
theorem mulOp_backward_eq_adjoint {s : Shape} (rhs : Tensor ℝ s)
    (x : Vec (Shape.size s)) (δ : Tensor ℝ s) :
    tensorToVec ((Spec.mulOp rhs).backward (vecToTensor x) δ) =
      (fderiv ℝ (fun y => tensorToVec ((Spec.mulOp rhs).forward (vecToTensor y)))
        x).adjoint (tensorToVec δ) :=
  binaryElemOp_backward_eq_adjoint rhs (· * ·) (fun _ y => y) x
    (fun i => hasDerivAt_mul_const (tensorToVec rhs i)) δ

/-- A fixed real denominator gives a linear quotient, including the totalized zero case. -/
theorem divOp_hasFDerivAt {s : Shape} (rhs : Tensor ℝ s) (x : Vec (Shape.size s)) :
    HasFDerivAt (fun y => tensorToVec ((Spec.divOp rhs).forward (vecToTensor y)))
      (coordinateDeriv fun i => 1 / tensorToVec rhs i) x :=
  binaryElemOp_hasFDerivAt rhs (· / ·) (fun _ y => 1 / y) x
    (fun i => (hasDerivAt_id (x i)).div_const (tensorToVec rhs i))

/-- The captured quotient backward agrees with the adjoint for every real fixed denominator. -/
theorem divOp_backward_eq_adjoint {s : Shape} (rhs : Tensor ℝ s)
    (x : Vec (Shape.size s)) (δ : Tensor ℝ s) :
    tensorToVec ((Spec.divOp rhs).backward (vecToTensor x) δ) =
      (fderiv ℝ (fun y => tensorToVec ((Spec.divOp rhs).forward (vecToTensor y)))
        x).adjoint (tensorToVec δ) :=
  binaryElemOp_backward_eq_adjoint rhs (· / ·) (fun _ y => 1 / y) x
    (fun i => (hasDerivAt_id (x i)).div_const (tensorToVec rhs i)) δ

/-- The epsilon shift belongs to the fixed denominator, so it does not contribute a derivative. -/
theorem safeDivOp_hasFDerivAt {s : Shape} (rhs : Tensor ℝ s)
    (x : Vec (Shape.size s)) :
    HasFDerivAt (fun y => tensorToVec ((Spec.safeDivOp rhs).forward (vecToTensor y)))
      (coordinateDeriv fun i => 1 / (tensorToVec rhs i + Context.defaultEpsilon)) x := by
  apply hasFDerivAt_of_coordinates
    (f := fun i z => z / (tensorToVec rhs i + Context.defaultEpsilon))
  · intro y
    ext i
    simp [Spec.safeDivOp, TorchLean.Tensor.safedivSpec]
  · intro i
    exact (hasDerivAt_id (x i)).div_const
      (tensorToVec rhs i + Context.defaultEpsilon)

/-- The actual shifted-division backward uses the same fixed denominator as its forward map. -/
theorem safeDivOp_backward_eq_adjoint {s : Shape} (rhs : Tensor ℝ s)
    (x : Vec (Shape.size s)) (δ : Tensor ℝ s) :
    tensorToVec ((Spec.safeDivOp rhs).backward (vecToTensor x) δ) =
      (fderiv ℝ (fun y => tensorToVec ((Spec.safeDivOp rhs).forward (vecToTensor y)))
        x).adjoint (tensorToVec δ) := by
  rw [(safeDivOp_hasFDerivAt rhs x).fderiv, coordinateDeriv_adjoint_apply]
  ext i
  simp [Spec.safeDivOp]

end

end Proofs.Autograd.PrimitiveSpecs
