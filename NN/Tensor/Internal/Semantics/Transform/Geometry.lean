/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Check.Transform
import Mathlib.Tactic.Bound.Init
public import NN.Tensor.Internal.Representation.Basic.Core
public import NN.Tensor.Internal.Representation.Fiber.Basic
public import NN.Tensor.Internal.Representation.Coordinate -- shake: keep
public import NN.Tensor.Internal.Representation.Fiber -- shake: keep
public import Mathlib.Algebra.BigOperators.GroupWithZero.Action -- shake: keep

/-!
# Coordinate geometry of checked transformations

This module turns the grouped axes of a checked transform into coordinate
maps, equivalences, and finite fibers. It contains no tensor denotation or
lowering program.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u

namespace Check.CheckedTransform

/-- The input tensor type of a checked transformation. -/
abbrev InputTensor (checked : CheckedTransform) (α : Type u)
    [Storage α] :=
  Rep α checked.value.normalized.input

/-- The output tensor type of a checked transformation. -/
abbrev OutputTensor (checked : CheckedTransform) (α : Type u)
    [Storage α] :=
  Rep α checked.value.output

/--
Convert a checked input tensor coordinate to coordinates of its elementary
axes.
-/
def inputTensorCoordinateEquiv (checked : CheckedTransform) :
    Coord checked.value.normalized.input ≃
      AxisTuple checked.value.axisLength checked.value.normalized.inputAxes :=
  AxisTuple.groupedCoordEquivOfEq checked.value.axisLength
    checked.value.normalized.inputGroups checked.valid.input_shape

/--
Convert a checked output tensor coordinate to coordinates of its elementary
axes.
-/
def outputTensorCoordinateEquiv (checked : CheckedTransform) :
    Coord checked.value.output ≃
      AxisTuple checked.value.axisLength checked.value.normalized.outputAxes :=
  AxisTuple.groupedCoordEquivOfEq checked.value.axisLength
    checked.value.normalized.outputGroups checked.valid.output_shape.symm

/--
The coordinate permutation certified by a checked rearrange plan.

Both normalized axis lists are duplicate-free and contain the same axes, so
selection in either direction gives mutually inverse coordinate maps.
-/
def rearrangeCoordinateEquiv (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange) :
    Coord checked.value.output ≃
      Coord checked.value.normalized.input :=
  checked.outputTensorCoordinateEquiv.trans <|
    (AxisTuple.selectEquiv
      checked.valid.normalization.input_nodup
      checked.valid.normalization.output_nodup
      (checked.valid.normalization.input_axes_subset_output_of_rearrange hKind)
      (checked.valid.normalization.output_axes_subset_input_of_rearrange hKind)).trans
        checked.inputTensorCoordinateEquiv.symm

/--
Map an output tensor coordinate to its input coordinate when every input axis
occurs in the output.

For repeat this forgets newly introduced output axes. For rearrange it
permutes the same elementary coordinates.
-/
def inputCoordinateOfOutput (checked : CheckedTransform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes) :
    Coord checked.value.output →
      Coord checked.value.normalized.input :=
  fun outputCoordinate =>
    checked.inputTensorCoordinateEquiv.symm <|
      AxisTuple.select hAxes
        (checked.outputTensorCoordinateEquiv outputCoordinate)

/--
Map an input tensor coordinate to its output coordinate when every output axis
occurs in the input.

For reduction this forgets the axes being aggregated. For rearrange it gives
the inverse axis permutation.
-/
def outputCoordinateOfInput (checked : CheckedTransform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.outputAxes →
          axis ∈ checked.value.normalized.inputAxes) :
    Coord checked.value.normalized.input →
      Coord checked.value.output :=
  fun inputCoordinate =>
    checked.outputTensorCoordinateEquiv.symm <|
      AxisTuple.select hAxes
        (checked.inputTensorCoordinateEquiv inputCoordinate)

/-- The rearrange equivalence applies the general output-to-input coordinate map. -/
@[simp] theorem rearrangeCoordinateEquiv_apply (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (outputCoordinate : Coord checked.value.output) :
    checked.rearrangeCoordinateEquiv hKind outputCoordinate =
      checked.inputCoordinateOfOutput
        (checked.valid.normalization.input_axes_subset_output_of_rearrange hKind)
        outputCoordinate :=
  rfl

/-- The inverse rearrange equivalence applies the input-to-output coordinate map. -/
@[simp] theorem rearrangeCoordinateEquiv_symm_apply
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputCoordinate : Coord checked.value.normalized.input) :
    (checked.rearrangeCoordinateEquiv hKind).symm inputCoordinate =
      checked.outputCoordinateOfInput
        (checked.valid.normalization.output_axes_subset_input_of_rearrange hKind)
        inputCoordinate :=
  rfl

/-- Mapping an input rearrange coordinate to the output and back is the identity. -/
@[simp] theorem inputCoordinateOfOutput_outputCoordinateOfInput
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputCoordinate : Coord checked.value.normalized.input) :
    checked.inputCoordinateOfOutput
        (checked.valid.normalization.input_axes_subset_output_of_rearrange hKind)
        (checked.outputCoordinateOfInput
          (checked.valid.normalization.output_axes_subset_input_of_rearrange hKind)
          inputCoordinate) =
      inputCoordinate :=
  (checked.rearrangeCoordinateEquiv hKind).apply_symm_apply inputCoordinate

/-- Mapping an output rearrange coordinate to the input and back is the identity. -/
@[simp] theorem outputCoordinateOfInput_inputCoordinateOfOutput
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (outputCoordinate : Coord checked.value.output) :
    checked.outputCoordinateOfInput
        (checked.valid.normalization.output_axes_subset_input_of_rearrange hKind)
        (checked.inputCoordinateOfOutput
          (checked.valid.normalization.input_axes_subset_output_of_rearrange hKind)
          outputCoordinate) =
      outputCoordinate :=
  (checked.rearrangeCoordinateEquiv hKind).symm_apply_apply outputCoordinate

/-- Rearrangement preserves the total number of tensor entries. -/
theorem rearrange_size_eq (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange) :
    Shape.size checked.value.output =
      Shape.size checked.value.normalized.input := by
  calc
    Shape.size checked.value.output =
        Fintype.card (Coord checked.value.output) :=
      (Coord.card checked.value.output).symm
    _ = Fintype.card (Coord checked.value.normalized.input) :=
      Fintype.card_congr (checked.rearrangeCoordinateEquiv hKind)
    _ = Shape.size checked.value.normalized.input :=
      Coord.card checked.value.normalized.input

/--
Every input coordinate has exactly one output coordinate under a checked
rearrangement. Thus rearrange neither drops nor duplicates tensor entries,
including for rank-zero and zero-size shapes.
-/
theorem rearrange_fiber_card (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputCoordinate : Coord checked.value.normalized.input) :
    Fintype.card
        (Fiber (checked.rearrangeCoordinateEquiv hKind) inputCoordinate) =
      1 :=
  Fiber.card_equiv (checked.rearrangeCoordinateEquiv hKind) inputCoordinate

/--
Every input coordinate of a checked repeat is selected once for each setting
of the newly introduced output axes.

The cardinality is therefore the product of their lengths. This also covers
the two boundary cases: no introduced axes give multiplicity one, while an
introduced zero-length axis gives an empty fiber.
-/
theorem repeat_fiber_card (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .repeat)
    (inputCoordinate : Coord checked.value.normalized.input) :
    Fintype.card
        (Fiber
          (checked.inputCoordinateOfOutput <|
            checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
          inputCoordinate) =
      ((checked.value.normalized.outputAxes.filter fun axis =>
          !checked.value.normalized.inputAxes.contains axis).map
        checked.value.axisLength).prod := by
  let hAxes :=
    checked.valid.normalization.input_axes_subset_output_of_repeat hKind
  let coordinateFiberEquiv :
      Fiber (checked.inputCoordinateOfOutput hAxes) inputCoordinate ≃
        Fiber (AxisTuple.select hAxes)
          (checked.inputTensorCoordinateEquiv inputCoordinate) :=
    { toFun := fun outputCoordinate =>
        ⟨checked.outputTensorCoordinateEquiv outputCoordinate.1, by
          apply checked.inputTensorCoordinateEquiv.symm.injective
          simpa only [inputCoordinateOfOutput, Equiv.symm_apply_apply] using
            outputCoordinate.2⟩
      invFun := fun outputAxisCoordinate =>
        ⟨checked.outputTensorCoordinateEquiv.symm outputAxisCoordinate.1, by
          simp only [inputCoordinateOfOutput, Equiv.apply_symm_apply,
            outputAxisCoordinate.2, Equiv.symm_apply_apply]⟩
      left_inv := fun outputCoordinate => by
        apply Subtype.ext
        exact checked.outputTensorCoordinateEquiv.symm_apply_apply
          outputCoordinate.1
      right_inv := fun outputAxisCoordinate => by
        apply Subtype.ext
        exact checked.outputTensorCoordinateEquiv.apply_symm_apply
          outputAxisCoordinate.1 }
  calc
    Fintype.card
          (Fiber (checked.inputCoordinateOfOutput hAxes) inputCoordinate) =
        Fintype.card
          (Fiber (AxisTuple.select hAxes)
            (checked.inputTensorCoordinateEquiv inputCoordinate)) :=
      Fintype.card_congr coordinateFiberEquiv
    _ =
        ((checked.value.normalized.outputAxes.filter fun axis =>
            !checked.value.normalized.inputAxes.contains axis).map
          checked.value.axisLength).prod :=
      AxisTuple.select_fiber_card
        checked.valid.normalization.input_nodup
        checked.valid.normalization.output_nodup
        hAxes
        (checked.inputTensorCoordinateEquiv inputCoordinate)

/-- Input axes removed by a checked reduction, in their original input order. -/
def reducedAxes (checked : CheckedTransform) : List AxisId :=
  checked.value.normalized.inputAxes.filter fun axis =>
    !checked.value.normalized.outputAxes.contains axis

/--
The physical shape traversed inside one reduction fiber.

Its axis order is inherited from the input pattern, so flat indices enumerate
the removed coordinates in the tensor's row-major order.
-/
def reductionShape (checked : CheckedTransform) : Shape :=
  checked.reducedAxes.map checked.value.axisLength

/--
The number of input coordinates aggregated into each output coordinate of a
checked reduction.

The value is one when no axis is removed and zero exactly when at least one
removed axis has length zero.
-/
def reductionFiberSize (checked : CheckedTransform) : Nat :=
  checked.reductionShape.prod

/-- The row-major reduction shape contains exactly one entry per fiber value. -/
@[simp] theorem reductionShape_size (checked : CheckedTransform) :
    Shape.size checked.reductionShape = checked.reductionFiberSize :=
  Shape.size_eq_prod checked.reductionShape

/--
One reduction-fiber coordinate is exactly one assignment of every input axis
removed from the output.

The equivalence is independent of tensor rank, grouping, and axis lengths.
It therefore also covers scalar reductions, reductions that remove no axes,
and empty fibers caused by zero-length dimensions.
-/
noncomputable def reductionFiberEquiv (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (outputCoordinate : Coord checked.value.output) :
    Fiber
        (checked.outputCoordinateOfInput <|
          checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
        outputCoordinate ≃
      AxisTuple checked.value.axisLength checked.reducedAxes := by
  classical
  let hAxes :=
    checked.valid.normalization.output_axes_subset_input_of_reduce hKind
  let coordinateFiberEquiv :
      Fiber (checked.outputCoordinateOfInput hAxes) outputCoordinate ≃
        Fiber (AxisTuple.select hAxes)
          (checked.outputTensorCoordinateEquiv outputCoordinate) :=
    { toFun := fun inputCoordinate =>
        ⟨checked.inputTensorCoordinateEquiv inputCoordinate.1, by
          apply checked.outputTensorCoordinateEquiv.symm.injective
          simpa only [outputCoordinateOfInput, Equiv.symm_apply_apply] using
            inputCoordinate.2⟩
      invFun := fun inputAxisCoordinate =>
        ⟨checked.inputTensorCoordinateEquiv.symm inputAxisCoordinate.1, by
          simp only [outputCoordinateOfInput, Equiv.apply_symm_apply,
            inputAxisCoordinate.2, Equiv.symm_apply_apply]⟩
      left_inv := fun inputCoordinate => by
        apply Subtype.ext
        exact checked.inputTensorCoordinateEquiv.symm_apply_apply
          inputCoordinate.1
      right_inv := fun inputAxisCoordinate => by
        apply Subtype.ext
        exact checked.inputTensorCoordinateEquiv.apply_symm_apply
          inputAxisCoordinate.1 }
  exact coordinateFiberEquiv.trans <|
    AxisTuple.selectFiberEquiv
      checked.valid.normalization.output_nodup
      checked.valid.normalization.input_nodup
      hAxes
      (checked.outputTensorCoordinateEquiv outputCoordinate)

/-- Every axis removed by a checked reduction occurs in its input axis list. -/
theorem reduction_axis_mem_input (checked : CheckedTransform) :
    ∀ ⦃axis⦄,
      axis ∈ checked.reducedAxes →
        axis ∈ checked.value.normalized.inputAxes :=
  fun _ hAxis => (List.mem_filter.mp hAxis).1

/--
The reduction-fiber equivalence reads the removed axes from the complete
input-axis assignment.
-/
@[simp] theorem reductionFiberEquiv_apply
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (outputCoordinate : Coord checked.value.output)
    (inputCoordinate :
      Fiber
        (checked.outputCoordinateOfInput <|
          checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
        outputCoordinate) :
    checked.reductionFiberEquiv hKind outputCoordinate inputCoordinate =
      AxisTuple.select checked.reduction_axis_mem_input
        (checked.inputTensorCoordinateEquiv inputCoordinate.1) :=
  rfl

/--
Every output coordinate of a checked reduction aggregates one input
coordinate for each setting of the removed axes.

The fiber cardinality is the product of the removed-axis lengths. Thus
reducing no axes gives a singleton fiber, while removing any zero-length axis
gives an empty fiber over every output coordinate.
-/
theorem reduce_fiber_card (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (outputCoordinate : Coord checked.value.output) :
    Fintype.card
        (Fiber
          (checked.outputCoordinateOfInput <|
            checked.valid.normalization.output_axes_subset_input_of_reduce hKind)
          outputCoordinate) =
      checked.reductionFiberSize := by
  classical
  calc
    Fintype.card
          (Fiber
            (checked.outputCoordinateOfInput <|
              checked.valid.normalization.output_axes_subset_input_of_reduce
                hKind)
            outputCoordinate) =
        Fintype.card
          (AxisTuple checked.value.axisLength checked.reducedAxes) :=
      Fintype.card_congr
        (checked.reductionFiberEquiv hKind outputCoordinate)
    _ = Fintype.card
          (Coord checked.reductionShape) :=
      (Fintype.card_congr
        (AxisTuple.coordEquiv checked.value.axisLength
          checked.reducedAxes)).symm
    _ = Shape.size checked.reductionShape :=
      Coord.card checked.reductionShape
    _ = checked.reductionShape.prod :=
      Shape.size_eq_prod checked.reductionShape
    _ = checked.reductionFiberSize := rfl

/--
A positive checked reduction-fiber size gives a nonempty coordinate fiber at
every output position.
-/
theorem reduce_fiber_card_pos (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize) :
    ∀ outputCoordinate,
      0 <
        Fintype.card
          (Fiber
            (checked.outputCoordinateOfInput <|
              checked.valid.normalization.output_axes_subset_input_of_reduce
                hKind)
            outputCoordinate) := by
  intro outputCoordinate
  rw [checked.reduce_fiber_card hKind outputCoordinate]
  exact hPositive

end Check.CheckedTransform

end TorchLean.Tensor.Internal
