/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Check.Transform
public import TorchLean.Tensor.Rep.Fiber.Mean -- shake: keep
public import TorchLean.Tensor.Rep.Fiber.Product -- shake: keep
public import TorchLean.Tensor.Rep.Fiber.Tie -- shake: keep
public import Mathlib.Algebra.BigOperators.GroupWithZero.Action -- shake: keep

/-!
# Coordinate geometry of checked transformations

The grouped axes of a checked transform give coordinate maps and equivalences
between its input and output tensors, together with the finite fibers of
repeat and reduction and their cardinalities. The module defines no tensor
denotation or lowering program.
-/

@[expose] public section

namespace TorchLean.Einops

open scoped BigOperators

universe u

namespace Transform

open Shape (Coord)
open Tensor (Rep)

/-- The input tensor type of a checked transformation. -/
abbrev InputTensor (checked : Transform) (α : Type u)
    [Storage α] :=
  Rep α checked.value.normalized.input

/-- The output tensor type of a checked transformation. -/
abbrev OutputTensor (checked : Transform) (α : Type u)
    [Storage α] :=
  Rep α checked.value.output

/--
Convert a checked input tensor coordinate to coordinates of its elementary
axes.
-/
def inputCoordEquiv (checked : Transform) :
    Coord checked.value.normalized.input ≃
      AxisTuple checked.value.axisLength checked.value.normalized.inputAxes :=
  AxisTuple.groupedCoordEquivOfEq checked.value.axisLength
    checked.value.normalized.inputGroups checked.valid.input_shape

/--
Convert a checked output tensor coordinate to coordinates of its elementary
axes.
-/
def outputCoordEquiv (checked : Transform) :
    Coord checked.value.output ≃
      AxisTuple checked.value.axisLength checked.value.normalized.outputAxes :=
  AxisTuple.groupedCoordEquivOfEq checked.value.axisLength
    checked.value.normalized.outputGroups checked.valid.output_shape.symm

/--
The coordinate permutation certified by a checked rearrange plan.

Both normalized axis lists are duplicate-free and contain the same axes, so
selection in either direction gives mutually inverse coordinate maps.
-/
def rearrangeEquiv (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange) :
    Coord checked.value.output ≃
      Coord checked.value.normalized.input :=
  checked.outputCoordEquiv.trans <|
    (AxisTuple.selectEquiv
      checked.valid.normalization.input_nodup
      checked.valid.normalization.output_nodup
      (checked.valid.normalization.input_subset_of_rearrange hKind)
      (checked.valid.normalization.output_subset_of_rearrange hKind)).trans
        checked.inputCoordEquiv.symm

/--
Map an output tensor coordinate to its input coordinate when every input axis
occurs in the output.

For repeat it forgets the output axes that the repeat adds, and for rearrange
it permutes the same elementary coordinates.
-/
def inputOfOutput (checked : Transform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.inputAxes →
          axis ∈ checked.value.normalized.outputAxes) :
    Coord checked.value.output →
      Coord checked.value.normalized.input :=
  fun outputCoordinate =>
    checked.inputCoordEquiv.symm <|
      AxisTuple.select hAxes
        (checked.outputCoordEquiv outputCoordinate)

/--
Map an input tensor coordinate to its output coordinate when every output axis
occurs in the input.

For reduction it forgets the aggregated axes, and for rearrange it gives the
inverse axis permutation.
-/
def outputOfInput (checked : Transform)
    (hAxes :
      ∀ ⦃axis⦄,
        axis ∈ checked.value.normalized.outputAxes →
          axis ∈ checked.value.normalized.inputAxes) :
    Coord checked.value.normalized.input →
      Coord checked.value.output :=
  fun inputCoordinate =>
    checked.outputCoordEquiv.symm <|
      AxisTuple.select hAxes
        (checked.inputCoordEquiv inputCoordinate)

/-- The rearrange equivalence applies the general output-to-input coordinate map. -/
@[simp] theorem rearrangeEquiv_apply (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (outputCoordinate : Coord checked.value.output) :
    checked.rearrangeEquiv hKind outputCoordinate =
      checked.inputOfOutput
        (checked.valid.normalization.input_subset_of_rearrange hKind)
        outputCoordinate :=
  rfl

/-- The inverse rearrange equivalence applies the input-to-output coordinate map. -/
@[simp] theorem rearrangeEquiv_symm_apply
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputCoordinate : Coord checked.value.normalized.input) :
    (checked.rearrangeEquiv hKind).symm inputCoordinate =
      checked.outputOfInput
        (checked.valid.normalization.output_subset_of_rearrange hKind)
        inputCoordinate :=
  rfl

/-- Mapping an input rearrange coordinate to the output and back is the identity. -/
@[simp] theorem inputOfOutput_outputOfInput
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputCoordinate : Coord checked.value.normalized.input) :
    checked.inputOfOutput
        (checked.valid.normalization.input_subset_of_rearrange hKind)
        (checked.outputOfInput
          (checked.valid.normalization.output_subset_of_rearrange hKind)
          inputCoordinate) =
      inputCoordinate :=
  (checked.rearrangeEquiv hKind).apply_symm_apply inputCoordinate

/-- Mapping an output rearrange coordinate to the input and back is the identity. -/
@[simp] theorem outputOfInput_inputOfOutput
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (outputCoordinate : Coord checked.value.output) :
    checked.outputOfInput
        (checked.valid.normalization.output_subset_of_rearrange hKind)
        (checked.inputOfOutput
          (checked.valid.normalization.input_subset_of_rearrange hKind)
          outputCoordinate) =
      outputCoordinate :=
  (checked.rearrangeEquiv hKind).symm_apply_apply outputCoordinate

/-- Rearrangement preserves the total number of tensor entries. -/
theorem rearrange_numel_eq (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange) :
    Shape.size checked.value.output =
      Shape.size checked.value.normalized.input := by
  calc
    Shape.size checked.value.output =
        Fintype.card (Coord checked.value.output) :=
      (Coord.card checked.value.output).symm
    _ = Fintype.card (Coord checked.value.normalized.input) :=
      Fintype.card_congr (checked.rearrangeEquiv hKind)
    _ = Shape.size checked.value.normalized.input :=
      Coord.card checked.value.normalized.input

/--
Every input coordinate has exactly one output coordinate under a checked
rearrangement. Thus rearrange neither drops nor duplicates tensor entries,
including for rank-zero and zero-size shapes.
-/
theorem rearrange_fiber_card (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputCoordinate : Coord checked.value.normalized.input) :
    Fintype.card
        (Fiber (checked.rearrangeEquiv hKind) inputCoordinate) =
      1 :=
  Fiber.card_equiv (checked.rearrangeEquiv hKind) inputCoordinate

/--
Every input coordinate of a checked repeat is read by one output coordinate for
each setting of the output axes absent from the input.

The fiber cardinality is therefore the product of those axis lengths. It is one
when the repeat adds no axis and zero when an added axis has length zero.
-/
theorem expand_fiber_card (checked : Transform)
    (hKind : checked.value.normalized.kind = .expand)
    (inputCoordinate : Coord checked.value.normalized.input) :
    Fintype.card
        (Fiber
          (checked.inputOfOutput <|
            checked.valid.normalization.input_subset_of_expand hKind)
          inputCoordinate) =
      ((checked.value.normalized.outputAxes.filter fun axis =>
          !checked.value.normalized.inputAxes.contains axis).map
        checked.value.axisLength).prod := by
  let hAxes :=
    checked.valid.normalization.input_subset_of_expand hKind
  calc
    Fintype.card
          (Fiber (checked.inputOfOutput hAxes) inputCoordinate) =
        Fintype.card
          (Fiber (checked.inputCoordEquiv.symm ∘ AxisTuple.select hAxes)
            inputCoordinate) := by
      simpa only [inputOfOutput, Function.comp_def,
        Equiv.apply_symm_apply] using
        Fiber.card_comp_equiv checked.outputCoordEquiv
          (checked.inputOfOutput hAxes) inputCoordinate
    _ =
        Fintype.card
          (Fiber (AxisTuple.select hAxes)
            (checked.inputCoordEquiv inputCoordinate)) :=
      (Fiber.card_target_equiv checked.inputCoordEquiv.symm
        (AxisTuple.select hAxes) inputCoordinate).symm
    _ =
        ((checked.value.normalized.outputAxes.filter fun axis =>
            !checked.value.normalized.inputAxes.contains axis).map
          checked.value.axisLength).prod :=
      AxisTuple.select_fiber_card
        checked.valid.normalization.input_nodup
        checked.valid.normalization.output_nodup
        hAxes
        (checked.inputCoordEquiv inputCoordinate)

/-- Input axes removed by a checked reduction, in their original input order. -/
def reducedAxes (checked : Transform) : List AxisId :=
  checked.value.normalized.inputAxes.filter fun axis =>
    !checked.value.normalized.outputAxes.contains axis

/--
The physical shape traversed inside one reduction fiber.

Its axis order is inherited from the input pattern, so flat indices enumerate
the removed coordinates in the tensor's row-major order.
-/
def reductionShape (checked : Transform) : Shape :=
  checked.reducedAxes.map checked.value.axisLength

/--
The number of input coordinates aggregated into each output coordinate of a
checked reduction.

The value is one when no axis is removed and zero exactly when at least one
removed axis has length zero.
-/
def reductionFiberSize (checked : Transform) : Nat :=
  checked.reductionShape.prod

/-- The row-major reduction shape contains exactly one entry per fiber value. -/
@[simp] theorem reductionShape_size (checked : Transform) :
    Shape.size checked.reductionShape = checked.reductionFiberSize :=
  Shape.size_eq_prod checked.reductionShape

/--
The fiber of an output coordinate of a checked reduction is equivalent to the
assignments of every input axis removed from the output.

The equivalence holds at every tensor rank, grouping and axis length, with no
separate case for scalar reductions, reductions that remove no axes, or
zero-length dimensions.
-/
noncomputable def reductionFiberEquiv (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (outputCoordinate : Coord checked.value.output) :
    Fiber
        (checked.outputOfInput <|
          checked.valid.normalization.output_subset_of_reduce hKind)
        outputCoordinate ≃
      AxisTuple checked.value.axisLength checked.reducedAxes :=
  AxisTuple.coordFiberEquiv checked.outputCoordEquiv checked.inputCoordEquiv
    checked.valid.normalization.output_nodup
    checked.valid.normalization.input_nodup
    (checked.valid.normalization.output_subset_of_reduce hKind)
    outputCoordinate

/-- Every axis removed by a checked reduction occurs in its input axis list. -/
theorem reduction_axis_mem_input (checked : Transform) :
    ∀ ⦃axis⦄,
      axis ∈ checked.reducedAxes →
        axis ∈ checked.value.normalized.inputAxes :=
  fun _ hAxis => (List.mem_filter.mp hAxis).1

/--
The reduction-fiber equivalence reads the removed axes from the complete
input-axis assignment.
-/
@[simp] theorem reductionFiberEquiv_apply
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (outputCoordinate : Coord checked.value.output)
    (inputCoordinate :
      Fiber
        (checked.outputOfInput <|
          checked.valid.normalization.output_subset_of_reduce hKind)
        outputCoordinate) :
    checked.reductionFiberEquiv hKind outputCoordinate inputCoordinate =
      AxisTuple.select checked.reduction_axis_mem_input
        (checked.inputCoordEquiv inputCoordinate.1) :=
  rfl

/--
Every output coordinate of a checked reduction aggregates one input
coordinate for each setting of the removed axes.

The fiber cardinality is the product of the removed-axis lengths, the same at
every output coordinate.
-/
theorem reduce_fiber_card (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (outputCoordinate : Coord checked.value.output) :
    Fintype.card
        (Fiber
          (checked.outputOfInput <|
            checked.valid.normalization.output_subset_of_reduce hKind)
          outputCoordinate) =
      checked.reductionFiberSize := by
  classical
  calc
    Fintype.card
          (Fiber
            (checked.outputOfInput <|
              checked.valid.normalization.output_subset_of_reduce
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
theorem reduce_fiber_card_pos (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize) :
    ∀ outputCoordinate,
      0 <
        Fintype.card
          (Fiber
            (checked.outputOfInput <|
              checked.valid.normalization.output_subset_of_reduce
                hKind)
            outputCoordinate) := by
  intro outputCoordinate
  rw [checked.reduce_fiber_card hKind outputCoordinate]
  exact hPositive

end Transform

end TorchLean.Einops
