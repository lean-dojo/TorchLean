/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Semantics.Transform.Geometry

/-!
# Rearrange and repeat semantics

Independent coordinate denotations and algebraic laws for checked
rearrangements and repetitions. Rearrange reindexes its input along the
checked axis permutation, and repeat pulls its input back along the
projection that forgets the added output axes. The laws cover inverses,
pointwise maps, total sums and adjoints under the finite tensor pairing.
-/

@[expose] public section

namespace TorchLean.Einops

open scoped BigOperators

universe u v w

namespace Semantics

open Tensor (Rep)
open Shape (Coord)

/--
Independent rearrange denotation: pull the input tensor along the certified
elementary-axis permutation.
-/
def denoteRearrange {α : Type u} [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor α :=
  Rep.reindex (checked.rearrangeEquiv hKind) inputTensor

/-- Rearrange reads the input entry selected by its output coordinate. -/
@[simp, grind =] theorem denoteRearrange_apply {α : Type u}
    [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    denoteRearrange checked hKind inputTensor outputCoordinate =
      inputTensor
        (checked.inputOfOutput
          (checked.valid.normalization.input_subset_of_rearrange hKind)
          outputCoordinate) :=
  by
    simp [denoteRearrange]

/--
Reindexing a rearranged tensor by the inverse coordinate equivalence recovers
the original tensor.
-/
@[simp, grind =, einops =, einops_simps] theorem inverse_denoteRearrange {α : Type u}
    [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) :
    Rep.reindex (checked.rearrangeEquiv hKind).symm
        (denoteRearrange checked hKind inputTensor) =
      inputTensor :=
  Rep.reindex_symm_reindex (checked.rearrangeEquiv hKind) inputTensor

/-- Rearrange commutes with every pointwise scalar map. -/
@[grind =, einops =, einops_simps] theorem map_denoteRearrange {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (f : α → β) (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) :
    Tensor.map f (denoteRearrange checked hKind inputTensor) =
      denoteRearrange checked hKind (Tensor.map f inputTensor) :=
  Rep.map_reindex f (checked.rearrangeEquiv hKind) inputTensor

/--
Rearrange commutes with every pointwise binary operator when both operands
have the same input shape.
-/
@[grind =, einops =, einops_simps]
theorem zipWith_denoteRearrange {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage β] [Storage γ]
    (f : α → β → γ) (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (leftTensor : checked.InputTensor α)
    (rightTensor : checked.InputTensor β) :
    Tensor.zipWith f
        (denoteRearrange checked hKind leftTensor)
        (denoteRearrange checked hKind rightTensor) =
      denoteRearrange checked hKind
        (Tensor.zipWith f leftTensor rightTensor) :=
  Rep.zipWith_reindex f (checked.rearrangeEquiv hKind)
    leftTensor rightTensor

/-- Rearrange preserves the total sum over any additive commutative monoid. -/
@[grind =, einops =, einops_simps] theorem sum_denoteRearrange {α : Type u} [Storage α]
    [AddCommMonoid α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α) :
    (∑ outputCoordinate, denoteRearrange checked hKind inputTensor outputCoordinate) =
      ∑ inputCoordinate, inputTensor inputCoordinate :=
  Rep.sum_reindex (checked.rearrangeEquiv hKind) inputTensor

/--
The adjoint of a checked rearrangement under the finite tensor pairing is
reindexing by its inverse coordinate equivalence.

The identity is algebraic and forms the pattern-level core of the
inverse-rearrange VJP. Linking it to a differentiability framework is a
separate theorem.
-/
@[grind =, einops =, einops_simps] theorem dot_denoteRearrange {R : Type u} [Storage R]
    [AddCommMonoid R] [Mul R]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor R)
    (outputTensor : checked.OutputTensor R) :
    Rep.dot (denoteRearrange checked hKind inputTensor) outputTensor =
      Rep.dot inputTensor
        (Rep.reindex
          (checked.rearrangeEquiv hKind).symm outputTensor) :=
  Rep.dot_reindex_left
    (checked.rearrangeEquiv hKind) inputTensor outputTensor

/-- Rearranging both operands preserves their finite tensor pairing. -/
@[grind =, einops =, einops_simps] theorem dot_denoteRearrange_denoteRearrange {R : Type u}
    [Storage R] [AddCommMonoid R] [Mul R]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (leftTensor rightTensor : checked.InputTensor R) :
    Rep.dot
        (denoteRearrange checked hKind leftTensor)
        (denoteRearrange checked hKind rightTensor) =
      Rep.dot leftTensor rightTensor :=
  Rep.dot_reindex_reindex (checked.rearrangeEquiv hKind)
    leftTensor rightTensor

/--
Independent repeat denotation: pull the input tensor along the projection that
forgets axes introduced on the output side.
-/
def denoteExpand {α : Type u} [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .expand)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor α :=
  Rep.pull
    (checked.inputOfOutput <|
      checked.valid.normalization.input_subset_of_expand hKind)
    inputTensor

/-- Repeat copies the input entry selected once the added output axes are forgotten. -/
@[simp, grind =] theorem denoteExpand_apply {α : Type u}
    [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .expand)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    denoteExpand checked hKind inputTensor outputCoordinate =
      inputTensor
        (checked.inputOfOutput
          (checked.valid.normalization.input_subset_of_expand hKind)
          outputCoordinate) :=
  by
    simp [denoteExpand]

/--
Aggregating a repeated tensor back to its input coordinates multiplies every
entry by the number of settings of the introduced axes.

Natural-number scalar multiplication states the result over any additive
commutative monoid, including the zero-multiplicity case.
-/
@[grind =, einops =, einops_simps] theorem push_denoteExpand {α : Type u} [Storage α]
    [AddCommMonoid α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .expand)
    (inputTensor : checked.InputTensor α) :
    Rep.push
        (checked.inputOfOutput <|
          checked.valid.normalization.input_subset_of_expand hKind)
        (denoteExpand checked hKind inputTensor) =
      Tensor.map
        (fun value =>
          ((checked.value.normalized.outputAxes.filter fun axis =>
              !checked.value.normalized.inputAxes.contains axis).map
            checked.value.axisLength).prod • value)
        inputTensor := by
  classical
  ext inputCoordinate
  simp only [Rep.push_apply, denoteExpand_apply, Tensor.map_apply]
  calc
    _ =
        ∑ _outputCoordinate :
            Fiber
              (checked.inputOfOutput <|
                checked.valid.normalization.input_subset_of_expand hKind)
              inputCoordinate,
          inputTensor inputCoordinate := by
      apply Finset.sum_congr rfl
      intro outputCoordinate _
      rw [outputCoordinate.property]
    _ = _ := by
      rw [Finset.sum_const, Finset.card_univ,
        checked.expand_fiber_card hKind inputCoordinate]

/--
Repeat scales the total additive sum by the product of the introduced axis
lengths.
-/
@[grind =, einops =, einops_simps] theorem sum_denoteExpand {α : Type u} [Storage α]
    [AddCommMonoid α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .expand)
    (inputTensor : checked.InputTensor α) :
    (∑ outputCoordinate,
        denoteExpand checked hKind inputTensor outputCoordinate) =
      ((checked.value.normalized.outputAxes.filter fun axis =>
          !checked.value.normalized.inputAxes.contains axis).map
        checked.value.axisLength).prod •
        ∑ inputCoordinate, inputTensor inputCoordinate := by
  let repeatProjection :=
    checked.inputOfOutput <|
      checked.valid.normalization.input_subset_of_expand hKind
  calc
    (∑ outputCoordinate,
          denoteExpand checked hKind inputTensor outputCoordinate) =
        ∑ inputCoordinate,
          Rep.push repeatProjection
            (denoteExpand checked hKind inputTensor) inputCoordinate :=
      (Rep.sum_push repeatProjection
        (denoteExpand checked hKind inputTensor)).symm
    _ =
        ∑ inputCoordinate,
          Tensor.map
            (fun value =>
              ((checked.value.normalized.outputAxes.filter fun axis =>
                  !checked.value.normalized.inputAxes.contains axis).map
                checked.value.axisLength).prod • value)
            inputTensor inputCoordinate := by
      rw [push_denoteExpand]
    _ =
        ((checked.value.normalized.outputAxes.filter fun axis =>
            !checked.value.normalized.inputAxes.contains axis).map
          checked.value.axisLength).prod •
          ∑ inputCoordinate, inputTensor inputCoordinate := by
      simpa only [Tensor.map_apply] using
        (Finset.smul_sum
          (s := Finset.univ)
          (f := inputTensor)
          (r :=
            ((checked.value.normalized.outputAxes.filter fun axis =>
              !checked.value.normalized.inputAxes.contains axis).map
              checked.value.axisLength).prod)).symm

/--
Fiber aggregation is the adjoint of repeat under the finite tensor pairing.

Distributing the input value over each output fiber preserves the order of
the two scalar factors, so multiplication need not commute.
-/
@[grind =, einops =, einops_simps] theorem dot_denoteExpand {R : Type u} [Storage R]
    [NonUnitalNonAssocSemiring R]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .expand)
    (inputTensor : checked.InputTensor R)
    (outputTensor : checked.OutputTensor R) :
    Rep.dot (denoteExpand checked hKind inputTensor) outputTensor =
      Rep.dot inputTensor
        (Rep.push
          (checked.inputOfOutput <|
            checked.valid.normalization.input_subset_of_expand hKind)
          outputTensor) := by
  classical
  let repeatProjection :=
    checked.inputOfOutput <|
      checked.valid.normalization.input_subset_of_expand hKind
  simp only [Rep.dot, denoteExpand_apply, Rep.push_apply, Finset.mul_sum]
  symm
  calc
    (∑ inputCoordinate, ∑ outputCoordinate : Fiber repeatProjection inputCoordinate,
        inputTensor inputCoordinate * outputTensor outputCoordinate.1) =
        ∑ inputCoordinate, ∑ outputCoordinate : Fiber repeatProjection inputCoordinate,
          inputTensor (repeatProjection outputCoordinate.1) *
            outputTensor outputCoordinate.1 := by
      apply Finset.sum_congr rfl
      intro inputCoordinate _
      apply Finset.sum_congr rfl
      intro outputCoordinate _
      rw [outputCoordinate.property]
    _ = ∑ outputCoordinate,
        inputTensor (repeatProjection outputCoordinate) * outputTensor outputCoordinate :=
      Fintype.sum_fiberwise repeatProjection
        (fun outputCoordinate =>
          inputTensor (repeatProjection outputCoordinate) * outputTensor outputCoordinate)

end Semantics

end TorchLean.Einops
