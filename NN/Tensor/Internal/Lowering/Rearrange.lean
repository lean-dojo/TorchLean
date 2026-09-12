/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Semantics.Transform.RearrangeRepeat
public import NN.Tensor.Internal.Semantics.Transform -- shake: keep

/-!
# Primitive lowering for rearrange

A checked rearrange lowers directly to three tensor operations:

1. reshape physical input groups into elementary input axes;
2. reindex those axes into output order;
3. reshape elementary output axes into physical output groups.

The definitions operate on the existing checked transformation rather than
introducing an operator-specific program wrapper. The main theorem proves
that this explicit tensor program equals the independent coordinate
denotation for every scalar type and every valid checked rearrange.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

namespace Check.CheckedTransform

/-- A checked input shape has the same size as its elementary-axis shape. -/
theorem input_size_eq_elementary (checked : Check.CheckedTransform) :
    Shape.size checked.value.normalized.input =
      Shape.size
        (checked.value.normalized.inputAxes.map checked.value.axisLength) :=
  AxisTuple.size_eq_elementary_of_groupedShape_eq checked.value.axisLength
    checked.value.normalized.inputGroups checked.valid.input_shape

/-- A checked output shape has the same size as its elementary-axis shape. -/
theorem elementary_output_size_eq (checked : Check.CheckedTransform) :
    Shape.size
        (checked.value.normalized.outputAxes.map checked.value.axisLength) =
      Shape.size checked.value.output :=
  (AxisTuple.size_eq_elementary_of_groupedShape_eq checked.value.axisLength
    checked.value.normalized.outputGroups checked.valid.output_shape.symm).symm

end Check.CheckedTransform

namespace AxisTuple

/--
Ungrouping certified tensor axes is row-major reshape followed by conversion
to a named elementary-axis tuple.
-/
theorem groupedCoordEquivOfEq_eq_reshapeCoordEquiv {ι : Type u}
    {shape : Shape} (length : ι → Nat) (groups : List (List ι))
    (hShape : groupedShape length groups = shape) :
    groupedCoordEquivOfEq length groups hShape =
      (Rep.reshapeCoordEquiv
        (size_eq_elementary_of_groupedShape_eq length groups hShape)).symm.trans
          (coordEquiv length groups.flatten) := by
  cases hShape
  ext tensorCoordinate
  rfl

end AxisTuple

namespace Lowering

open Check

/--
Converting an input tensor coordinate to named elementary axes factors through
the row-major reshape used by lowering.
-/
theorem inputTensorCoordinateEquiv_eq_reshape (checked : CheckedTransform) :
    checked.inputTensorCoordinateEquiv =
      (Rep.reshapeCoordEquiv checked.input_size_eq_elementary).symm.trans
        (AxisTuple.coordEquiv checked.value.axisLength
          checked.value.normalized.inputAxes) := by
  simpa [Check.CheckedTransform.inputTensorCoordinateEquiv,
    Check.CheckedTransform.input_size_eq_elementary,
    Check.NormalizedTransform.inputAxes] using
      AxisTuple.groupedCoordEquivOfEq_eq_reshapeCoordEquiv
        checked.value.axisLength checked.value.normalized.inputGroups
        checked.valid.input_shape

/--
Converting an output tensor coordinate to named elementary axes factors
through the row-major reshape used by lowering.
-/
theorem outputTensorCoordinateEquiv_eq_reshape (checked : CheckedTransform) :
    checked.outputTensorCoordinateEquiv =
      (Rep.reshapeCoordEquiv checked.elementary_output_size_eq).trans
        (AxisTuple.coordEquiv checked.value.axisLength
          checked.value.normalized.outputAxes) := by
  have h :=
    AxisTuple.groupedCoordEquivOfEq_eq_reshapeCoordEquiv
      checked.value.axisLength checked.value.normalized.outputGroups
      checked.valid.output_shape.symm
  rw [← Rep.reshapeCoordEquiv_symm] at h
  simpa [Check.CheckedTransform.outputTensorCoordinateEquiv,
    Check.CheckedTransform.elementary_output_size_eq,
    Check.NormalizedTransform.outputAxes] using h

/-- Execute the explicit reshape, axis permutation, and reshape stages. -/
def rearrangeTensor {α : Type u} [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor α :=
  let axisPermutation :
      Coord
          (checked.value.normalized.outputAxes.map checked.value.axisLength) ≃
        Coord
          (checked.value.normalized.inputAxes.map checked.value.axisLength) :=
    (AxisTuple.coordEquiv checked.value.axisLength
        checked.value.normalized.outputAxes).trans <|
      (AxisTuple.selectEquiv
        checked.valid.normalization.input_nodup
        checked.valid.normalization.output_nodup
        (checked.valid.normalization.input_axes_subset_output_of_rearrange hKind)
        (checked.valid.normalization.output_axes_subset_input_of_rearrange hKind)).trans <|
          (AxisTuple.coordEquiv checked.value.axisLength
            checked.value.normalized.inputAxes).symm
  Rep.reshape checked.elementary_output_size_eq <|
    Rep.reindex axisPermutation <|
      Rep.reshape checked.input_size_eq_elementary inputTensor

/--
Primitive rearrange lowering is correct for every scalar type and input
tensor.
-/
@[grind =] theorem rearrangeTensor_correct {α : Type u}
    [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) :
    rearrangeTensor checked hKind inputTensor =
      Semantics.denoteRearrange checked hKind inputTensor := by
  let axisPermutation :
      Coord
          (checked.value.normalized.outputAxes.map checked.value.axisLength) ≃
        Coord
          (checked.value.normalized.inputAxes.map checked.value.axisLength) :=
    (AxisTuple.coordEquiv checked.value.axisLength
        checked.value.normalized.outputAxes).trans <|
      (AxisTuple.selectEquiv
        checked.valid.normalization.input_nodup
        checked.valid.normalization.output_nodup
        (checked.valid.normalization.input_axes_subset_output_of_rearrange hKind)
        (checked.valid.normalization.output_axes_subset_input_of_rearrange hKind)).trans <|
          (AxisTuple.coordEquiv checked.value.axisLength
            checked.value.normalized.inputAxes).symm
  unfold rearrangeTensor Semantics.denoteRearrange
  dsimp only
  rw [Rep.reshape_eq_reindex, Rep.reshape_eq_reindex,
    Rep.reindex_trans, Rep.reindex_trans]
  congr 1
  rw [Check.CheckedTransform.rearrangeCoordinateEquiv,
    outputTensorCoordinateEquiv_eq_reshape,
    inputTensorCoordinateEquiv_eq_reshape]
  ext outputTensorCoordinate
  rfl

end Lowering

end TorchLean.Tensor.Internal
