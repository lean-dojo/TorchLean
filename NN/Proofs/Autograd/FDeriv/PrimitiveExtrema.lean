/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.PrimitiveCoordinates

/-!
# Minimum and maximum

The captured minimum and maximum operations use the same comparison masks as their two-input
runtime counterparts. A strict winner receives the full upstream gradient, a strict loser
receives zero, and each input receives half at a tie. Capturing the right-hand tensor leaves
that selection unchanged.

We first identify the stored backward functions coordinate by coordinate, including their value
at ties. We then prove that these functions are the adjoints of the forward derivatives wherever
every input coordinate differs from its captured partner. The equality case has a selected
backward rule, but no classical derivative is claimed there.
-/

@[expose] public section

namespace Proofs.Autograd.PrimitiveSpecs

open Spec TorchLean TorchLean.Tensor Filter

noncomputable section

/-- With unequal arguments, minimum agrees locally with either the input or the constant. -/
theorem hasDerivAt_min_const_of_ne {x c : ℝ} (h : x ≠ c) :
    HasDerivAt (fun y => min y c)
      (if x < c then 1 else if c < x then 0 else 1 / 2) x := by
  rcases lt_or_gt_of_ne h with hlt | hgt
  · simp only [if_pos hlt]
    apply (hasDerivAt_id' x).congr_of_eventuallyEq
    filter_upwards [Iio_mem_nhds hlt] with y hy
    exact min_eq_left (le_of_lt hy)
  · simp only [if_neg (not_lt_of_ge hgt.le), if_pos hgt]
    apply (hasDerivAt_const x c).congr_of_eventuallyEq
    filter_upwards [Ioi_mem_nhds hgt] with y hy
    exact min_eq_right (le_of_lt hy)

/-- With unequal arguments, maximum has derivative one at a strict winner and zero otherwise. -/
theorem hasDerivAt_max_const_of_ne {x c : ℝ} (h : x ≠ c) :
    HasDerivAt (fun y => max y c)
      (if c < x then 1 else if x < c then 0 else 1 / 2) x := by
  rcases lt_or_gt_of_ne h with hlt | hgt
  · simp only [if_neg (not_lt_of_ge hlt.le), if_pos hlt]
    apply (hasDerivAt_const x c).congr_of_eventuallyEq
    filter_upwards [Iio_mem_nhds hlt] with y hy
    exact max_eq_right (le_of_lt hy)
  · simp only [if_pos hgt]
    apply (hasDerivAt_id' x).congr_of_eventuallyEq
    filter_upwards [Ioi_mem_nhds hgt] with y hy
    exact max_eq_left (le_of_lt hy)

/-- The actual minimum backward multiplies by its selected comparison mask at each coordinate. -/
theorem minOp_backward_apply {s : Shape} (rhs x δ : Tensor ℝ s)
    (i : Fin (Shape.size s)) :
    tensorToVec ((Spec.minOp rhs).backward x δ) i =
      (if tensorToVec x i < tensorToVec rhs i then 1
       else if tensorToVec rhs i < tensorToVec x i then 0 else 1 / 2) *
        tensorToVec δ i := by
  simp [Spec.minOp]

/-- The actual maximum backward keeps the same comparison order as the runtime selection. -/
theorem maxOp_backward_apply {s : Shape} (rhs x δ : Tensor ℝ s)
    (i : Fin (Shape.size s)) :
    tensorToVec ((Spec.maxOp rhs).backward x δ) i =
      (if tensorToVec rhs i < tensorToVec x i then 1
       else if tensorToVec x i < tensorToVec rhs i then 0 else 1 / 2) *
        tensorToVec δ i := by
  simp [Spec.maxOp]

/-- A minimum tie retains half the upstream gradient even when the other input is captured. -/
theorem minOp_backward_apply_of_eq {s : Shape} (rhs x δ : Tensor ℝ s)
    (i : Fin (Shape.size s)) (h : tensorToVec x i = tensorToVec rhs i) :
    tensorToVec ((Spec.minOp rhs).backward x δ) i = (1 / 2) * tensorToVec δ i := by
  rw [minOp_backward_apply]
  simp [h]

/-- A maximum tie uses the same half-gradient selection as minimum. -/
theorem maxOp_backward_apply_of_eq {s : Shape} (rhs x δ : Tensor ℝ s)
    (i : Fin (Shape.size s)) (h : tensorToVec x i = tensorToVec rhs i) :
    tensorToVec ((Spec.maxOp rhs).backward x δ) i = (1 / 2) * tensorToVec δ i := by
  rw [maxOp_backward_apply]
  simp [h]

/-- Away from all coordinate ties, the minimum mask is the derivative of its tensor forward. -/
theorem minOp_hasFDerivAt {s : Shape} (rhs : Tensor ℝ s) (x : Vec (Shape.size s))
    (hneq : ∀ i, x i ≠ tensorToVec rhs i) :
    HasFDerivAt (fun y => tensorToVec ((Spec.minOp rhs).forward (vecToTensor y)))
      (coordinateDeriv fun i =>
        if x i < tensorToVec rhs i then 1
        else if tensorToVec rhs i < x i then 0 else 1 / 2) x := by
  apply hasFDerivAt_of_coordinates (f := fun i z => min z (tensorToVec rhs i))
  · intro y
    ext i
    simp [Spec.minOp, TorchLean.Tensor.minSpec]
  · intro i
    exact hasDerivAt_min_const_of_ne (hneq i)

/-- Away from all coordinate ties, the maximum mask is the derivative of its tensor forward. -/
theorem maxOp_hasFDerivAt {s : Shape} (rhs : Tensor ℝ s) (x : Vec (Shape.size s))
    (hneq : ∀ i, x i ≠ tensorToVec rhs i) :
    HasFDerivAt (fun y => tensorToVec ((Spec.maxOp rhs).forward (vecToTensor y)))
      (coordinateDeriv fun i =>
        if tensorToVec rhs i < x i then 1
        else if x i < tensorToVec rhs i then 0 else 1 / 2) x := by
  apply hasFDerivAt_of_coordinates (f := fun i z => max z (tensorToVec rhs i))
  · intro y
    ext i
    simp [Spec.maxOp, TorchLean.Tensor.maxSpec]
  · intro i
    exact hasDerivAt_max_const_of_ne (hneq i)

/-- The stored minimum backward is the adjoint of the forward derivative away from ties. -/
theorem minOp_backward_eq_adjoint {s : Shape} (rhs : Tensor ℝ s)
    (x : Vec (Shape.size s)) (hneq : ∀ i, x i ≠ tensorToVec rhs i)
    (δ : Tensor ℝ s) :
    tensorToVec ((Spec.minOp rhs).backward (vecToTensor x) δ) =
      (fderiv ℝ (fun y => tensorToVec ((Spec.minOp rhs).forward (vecToTensor y)))
        x).adjoint (tensorToVec δ) := by
  rw [(minOp_hasFDerivAt rhs x hneq).fderiv, coordinateDeriv_adjoint_apply]
  ext i
  simp [Spec.minOp]

/-- The stored maximum backward is the adjoint of the forward derivative away from ties. -/
theorem maxOp_backward_eq_adjoint {s : Shape} (rhs : Tensor ℝ s)
    (x : Vec (Shape.size s)) (hneq : ∀ i, x i ≠ tensorToVec rhs i)
    (δ : Tensor ℝ s) :
    tensorToVec ((Spec.maxOp rhs).backward (vecToTensor x) δ) =
      (fderiv ℝ (fun y => tensorToVec ((Spec.maxOp rhs).forward (vecToTensor y)))
        x).adjoint (tensorToVec δ) := by
  rw [(maxOp_hasFDerivAt rhs x hneq).fderiv, coordinateDeriv_adjoint_apply]
  ext i
  simp [Spec.maxOp]

end

end Proofs.Autograd.PrimitiveSpecs
