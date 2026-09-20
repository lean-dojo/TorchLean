/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Semantics.Transform.Geometry

/-!
# Rearrange and repeat semantics

Independent coordinate denotations and algebraic laws for checked
rearrangements and repetitions.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

open scoped BigOperators

universe u v w

namespace Semantics

open Check

/--
Independent rearrange denotation: pull the input tensor along the certified
elementary-axis permutation.
-/
def denoteRearrange {α : Type u} [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor α :=
  Rep.reindex (checked.rearrangeCoordinateEquiv hKind) inputTensor

/-- Rearrange reads the input entry selected by its output coordinate. -/
@[simp, grind =] theorem denoteRearrange_apply {α : Type u}
    [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    denoteRearrange checked hKind inputTensor outputCoordinate =
      inputTensor
        (checked.inputCoordinateOfOutput
          (checked.valid.normalization.input_axes_subset_output_of_rearrange hKind)
          outputCoordinate) :=
  by
    simp [denoteRearrange]

/--
Reindexing a rearranged tensor by the inverse coordinate equivalence recovers
the original tensor.
-/
@[simp, grind =] theorem inverse_denoteRearrange {α : Type u}
    [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) :
    Rep.reindex (checked.rearrangeCoordinateEquiv hKind).symm
        (denoteRearrange checked hKind inputTensor) =
      inputTensor :=
  Rep.reindex_symm_reindex (checked.rearrangeCoordinateEquiv hKind) inputTensor

/-- Rearrange commutes with every pointwise scalar map. -/
@[grind =] theorem map_denoteRearrange {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (f : α → β) (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) :
    Rep.map f (denoteRearrange checked hKind inputTensor) =
      denoteRearrange checked hKind (Rep.map f inputTensor) :=
  Rep.map_reindex f (checked.rearrangeCoordinateEquiv hKind) inputTensor

/--
Rearrange commutes with every pointwise binary operator when both operands
have the same input shape.
-/
@[grind =] theorem zipWith_denoteRearrange {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage β] [Storage γ]
    (f : α → β → γ) (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (leftTensor : checked.InputTensor α)
    (rightTensor : checked.InputTensor β) :
    Rep.zipWith f
        (denoteRearrange checked hKind leftTensor)
        (denoteRearrange checked hKind rightTensor) =
      denoteRearrange checked hKind
        (Rep.zipWith f leftTensor rightTensor) :=
  Rep.zipWith_reindex f (checked.rearrangeCoordinateEquiv hKind)
    leftTensor rightTensor

/-- Rearrange preserves the total sum over any additive commutative monoid. -/
@[grind =] theorem sum_denoteRearrange {α : Type u} [Storage α]
    [AddCommMonoid α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) :
    (∑ outputCoordinate, denoteRearrange checked hKind inputTensor outputCoordinate) =
      ∑ inputCoordinate, inputTensor inputCoordinate :=
  Rep.sum_reindex (checked.rearrangeCoordinateEquiv hKind) inputTensor

/--
The adjoint of a checked rearrangement under the finite tensor pairing is
reindexing by its inverse coordinate equivalence.

This algebraic identity is the pattern-level core of the inverse-rearrange
VJP; connecting it to a differentiability framework is a separate theorem.
-/
@[grind =] theorem dot_denoteRearrange {R : Type u} [Storage R]
    [Semiring R]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor R)
    (outputTensor : checked.OutputTensor R) :
    Rep.dot (denoteRearrange checked hKind inputTensor) outputTensor =
      Rep.dot inputTensor
        (Rep.reindex
          (checked.rearrangeCoordinateEquiv hKind).symm outputTensor) :=
  Rep.dot_reindex_eq_dot_reindex_symm
    (checked.rearrangeCoordinateEquiv hKind) inputTensor outputTensor

/-- Rearranging both operands preserves their finite tensor pairing. -/
@[grind =] theorem dot_denoteRearrange_denoteRearrange {R : Type u}
    [Storage R] [Semiring R]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (leftTensor rightTensor : checked.InputTensor R) :
    Rep.dot
        (denoteRearrange checked hKind leftTensor)
        (denoteRearrange checked hKind rightTensor) =
      Rep.dot leftTensor rightTensor :=
  Rep.dot_reindex_reindex (checked.rearrangeCoordinateEquiv hKind)
    leftTensor rightTensor

/--
Independent repeat denotation: pull the input tensor along the projection that
forgets axes introduced on the output side.
-/
def denoteRepeat {α : Type u} [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .repeat)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor α :=
  Rep.pull
    (checked.inputCoordinateOfOutput <|
      checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
    inputTensor

/-- Repeat copies the input entry selected after new output axes are forgotten. -/
@[simp, grind =] theorem denoteRepeat_apply {α : Type u}
    [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .repeat)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    denoteRepeat checked hKind inputTensor outputCoordinate =
      inputTensor
        (checked.inputCoordinateOfOutput
          (checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
          outputCoordinate) :=
  by
    simp [denoteRepeat]

/--
Aggregating a repeated tensor back to its input coordinates multiplies every
entry by the number of settings of the introduced axes.

Natural-number scalar multiplication states the result over any additive
commutative monoid, including the zero-multiplicity case.
-/
@[grind =] theorem push_denoteRepeat {α : Type u} [Storage α]
    [AddCommMonoid α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .repeat)
    (inputTensor : checked.InputTensor α) :
    Rep.push
        (checked.inputCoordinateOfOutput <|
          checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
        (denoteRepeat checked hKind inputTensor) =
      Rep.map
        (fun value =>
          ((checked.value.normalized.outputAxes.filter fun axis =>
              !checked.value.normalized.inputAxes.contains axis).map
            checked.value.axisLength).prod • value)
        inputTensor := by
  classical
  ext inputCoordinate
  simp only [Rep.push_apply, denoteRepeat_apply, Rep.map_apply]
  calc
    _ =
        ∑ _outputCoordinate :
            Fiber
              (checked.inputCoordinateOfOutput <|
                checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
              inputCoordinate,
          inputTensor inputCoordinate := by
      apply Finset.sum_congr rfl
      intro outputCoordinate _
      rw [outputCoordinate.property]
    _ = _ := by
      rw [Finset.sum_const, Finset.card_univ,
        checked.repeat_fiber_card hKind inputCoordinate]

/--
Repeat scales the total additive sum by the product of the introduced axis
lengths.
-/
@[grind =] theorem sum_denoteRepeat {α : Type u} [Storage α]
    [AddCommMonoid α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .repeat)
    (inputTensor : checked.InputTensor α) :
    (∑ outputCoordinate,
        denoteRepeat checked hKind inputTensor outputCoordinate) =
      ((checked.value.normalized.outputAxes.filter fun axis =>
          !checked.value.normalized.inputAxes.contains axis).map
        checked.value.axisLength).prod •
        ∑ inputCoordinate, inputTensor inputCoordinate := by
  let repeatProjection :=
    checked.inputCoordinateOfOutput <|
      checked.valid.normalization.input_axes_subset_output_of_repeat hKind
  calc
    (∑ outputCoordinate,
          denoteRepeat checked hKind inputTensor outputCoordinate) =
        ∑ inputCoordinate,
          Rep.push repeatProjection
            (denoteRepeat checked hKind inputTensor) inputCoordinate :=
      (Rep.sum_push repeatProjection
        (denoteRepeat checked hKind inputTensor)).symm
    _ =
        ∑ inputCoordinate,
          Rep.map
            (fun value =>
              ((checked.value.normalized.outputAxes.filter fun axis =>
                  !checked.value.normalized.inputAxes.contains axis).map
                checked.value.axisLength).prod • value)
            inputTensor inputCoordinate := by
      rw [push_denoteRepeat]
    _ =
        ((checked.value.normalized.outputAxes.filter fun axis =>
            !checked.value.normalized.inputAxes.contains axis).map
          checked.value.axisLength).prod •
          ∑ inputCoordinate, inputTensor inputCoordinate := by
      simpa only [Rep.map_apply] using
        (Finset.smul_sum
          (s := Finset.univ)
          (f := inputTensor)
          (r :=
            ((checked.value.normalized.outputAxes.filter fun axis =>
              !checked.value.normalized.inputAxes.contains axis).map
              checked.value.axisLength).prod)).symm

/--
Fiber aggregation is the adjoint of repeat under the finite tensor pairing.

Commutativity of scalar multiplication is needed because `Rep.dot` records
the left operand first, whereas the general push/pull adjunction is stated
with the pushed tensor on the left.
-/
@[grind =] theorem dot_denoteRepeat {R : Type u} [Storage R]
    [CommSemiring R]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .repeat)
    (inputTensor : checked.InputTensor R)
    (outputTensor : checked.OutputTensor R) :
    Rep.dot (denoteRepeat checked hKind inputTensor) outputTensor =
      Rep.dot inputTensor
        (Rep.push
          (checked.inputCoordinateOfOutput <|
            checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
          outputTensor) := by
  have dot_comm {shape : Shape} (leftTensor rightTensor : Rep R shape) :
      Rep.dot leftTensor rightTensor =
        Rep.dot rightTensor leftTensor := by
    simp only [Rep.dot, mul_comm]
  calc
    Rep.dot (denoteRepeat checked hKind inputTensor) outputTensor =
        Rep.dot outputTensor (denoteRepeat checked hKind inputTensor) :=
      dot_comm _ _
    _ =
        Rep.dot
          (Rep.push
            (checked.inputCoordinateOfOutput <|
              checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
            outputTensor)
          inputTensor := by
      simpa only [denoteRepeat] using
        (Rep.dot_push_eq_dot_pull
          (checked.inputCoordinateOfOutput <|
            checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
          outputTensor inputTensor).symm
    _ =
        Rep.dot inputTensor
          (Rep.push
            (checked.inputCoordinateOfOutput <|
              checked.valid.normalization.input_axes_subset_output_of_repeat hKind)
            outputTensor) :=
      dot_comm _ _

end Semantics

end TorchLean.Tensor.Internal
