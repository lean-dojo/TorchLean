/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Semantics.Transform.RearrangeExpand -- shake: keep
public import TorchLean.Tensor.Einops.Semantics.Transform.Reduction -- shake: keep

/-!
# Primitive lowering for rearrange

A checked rearrange lowers to three tensor operations.

1. Reshape physical input groups into elementary input axes.
2. Reindex those axes into output order.
3. Reshape elementary output axes into physical output groups.

The definitions take their axes, lengths, and validity proofs from the `Transform`.
`rearrange_correct` proves that this three-stage tensor program equals the independent
coordinate denotation for every scalar type and every valid checked rearrange.
-/

@[expose] public section

namespace TorchLean.Einops

universe u

namespace Transform

/-- A checked input shape has the same size as its elementary-axis shape. -/
theorem input_size_eq_elementary (checked : Einops.Transform) :
    Shape.size checked.value.normalized.input =
      Shape.size
        (checked.value.normalized.inputAxes.map checked.value.axisLength) :=
  AxisTuple.size_eq_of_groupedShape_eq checked.value.axisLength
    checked.value.normalized.inputGroups checked.valid.input_shape

/-- A checked output shape has the same size as its elementary-axis shape. -/
theorem elementary_output_size_eq (checked : Einops.Transform) :
    Shape.size
        (checked.value.normalized.outputAxes.map checked.value.axisLength) =
      Shape.size checked.value.output :=
  (AxisTuple.size_eq_of_groupedShape_eq checked.value.axisLength
    checked.value.normalized.outputGroups checked.valid.output_shape.symm).symm

end Transform

namespace AxisTuple

/--
Ungrouping certified tensor axes is row-major reshape followed by conversion
to a named elementary-axis tuple.
-/
theorem groupedCoordEquivOfEq_eq {ι : Type u}
    {shape : Shape} (length : ι → Nat) (groups : List (List ι))
    (hShape : groupedShape length groups = shape) :
    groupedCoordEquivOfEq length groups hShape =
      (Shape.Coord.reshapeEquiv
        (size_eq_of_groupedShape_eq length groups hShape)).symm.trans
          (coordEquiv length groups.flatten) := by
  cases hShape
  ext tensorCoordinate
  rfl

end AxisTuple

section

open Tensor (Rep)
open Shape (Coord)

/--
Converting an input tensor coordinate to named elementary axes factors through
the row-major reshape used by lowering.
-/
theorem Transform.inputCoordEquiv_eq_reshape (checked : Transform) :
    checked.inputCoordEquiv =
      (Coord.reshapeEquiv checked.input_size_eq_elementary).symm.trans
        (AxisTuple.coordEquiv checked.value.axisLength
          checked.value.normalized.inputAxes) := by
  simpa [inputCoordEquiv,
    input_size_eq_elementary,
    NormalForm.inputAxes] using
      AxisTuple.groupedCoordEquivOfEq_eq
        checked.value.axisLength checked.value.normalized.inputGroups
        checked.valid.input_shape

/--
Converting an output tensor coordinate to named elementary axes factors
through the row-major reshape used by lowering.
-/
theorem Transform.outputCoordEquiv_eq_reshape (checked : Transform) :
    checked.outputCoordEquiv =
      (Coord.reshapeEquiv checked.elementary_output_size_eq).trans
        (AxisTuple.coordEquiv checked.value.axisLength
          checked.value.normalized.outputAxes) := by
  have h :=
    AxisTuple.groupedCoordEquivOfEq_eq
      checked.value.axisLength checked.value.normalized.outputGroups
      checked.valid.output_shape.symm
  rw [← Coord.reshapeEquiv_symm] at h
  simpa [outputCoordEquiv,
    elementary_output_size_eq,
    NormalForm.outputAxes] using h

/-- Execute the explicit reshape, axis permutation, and reshape stages. -/
def Lowering.rearrange {α : Type u} [Storage α]
    (checked : Transform)
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
        (checked.valid.normalization.input_subset_of_rearrange hKind)
        (checked.valid.normalization.output_subset_of_rearrange hKind)).trans <|
          (AxisTuple.coordEquiv checked.value.axisLength
            checked.value.normalized.inputAxes).symm
  Rep.reshape checked.elementary_output_size_eq <|
    Rep.reindex axisPermutation <|
      Rep.reshape checked.input_size_eq_elementary inputTensor

/--
Primitive rearrange lowering is correct for every scalar type and input
tensor.
-/
@[grind =, einops =, einops_simps] theorem Lowering.rearrange_correct {α : Type u}
    [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) :
    rearrange checked hKind inputTensor =
      Semantics.denoteRearrange checked hKind inputTensor := by
  unfold rearrange Semantics.denoteRearrange
  dsimp only
  rw [Rep.reshape_eq_reindex, Rep.reshape_eq_reindex,
    Rep.reindex_trans, Rep.reindex_trans]
  congr 1
  rw [Transform.rearrangeEquiv,
    Transform.outputCoordEquiv_eq_reshape,
    Transform.inputCoordEquiv_eq_reshape]
  ext outputTensorCoordinate
  rfl

end

end TorchLean.Einops
