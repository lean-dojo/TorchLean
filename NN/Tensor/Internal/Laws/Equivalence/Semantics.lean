/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Laws.Equivalence.Plan
public import NN.Tensor.Internal.Laws.Equivalence.Index -- shake: keep

/-!
# Semantic rearrangement equivalence

Coordinate-equivalent checked rearrangements have equal tensor denotations.
The converse is witnessed by natural-number coordinate tensors.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

open Rearrangement.Impl

namespace Semantics

open Check

/--
Equivalent rearrange coordinate maps produce equal tensors over every scalar
type. The casts only transport tensors across the supplied physical-shape
equalities.
-/
theorem denoteRearrange_eq_of_equivalent
    {α : Type u} [Storage α]
    (first second : CheckedTransform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (hInputShape :
      first.value.normalized.input =
        second.value.normalized.input)
    (hOutputShape : first.value.output = second.value.output)
    (hEquivalent :
      first.RearrangeEquivalent second hFirstKind hSecondKind
        hInputShape hOutputShape)
    (inputTensor : first.InputTensor α) :
    denoteRearrange first hFirstKind inputTensor =
      cast (congrArg (fun shape => Rep α shape) hOutputShape.symm)
        (denoteRearrange second hSecondKind
          (cast (congrArg (fun shape => Rep α shape) hInputShape)
            inputTensor)) := by
  rw [cast_tensor_eq_reindex hInputShape inputTensor]
  rw [cast_tensor_eq_reindex hOutputShape.symm]
  ext outputCoordinate
  simpa only [denoteRearrange, Rep.reindex_apply, cast_eq_equivCast] using
    congrArg inputTensor (hEquivalent outputCoordinate)

/--
Natural-number coordinate tensors separate rearrange maps. Thus equality on
all `Nat` tensors is not merely sufficient but also necessary for extensional
pattern equivalence.
-/
theorem rearrangeEquivalent_of_denoteRearrange_nat_eq
    (first second : CheckedTransform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (hInputShape :
      first.value.normalized.input =
        second.value.normalized.input)
    (hOutputShape : first.value.output = second.value.output)
    (hDenotations :
      ∀ inputTensor : first.InputTensor Nat,
        denoteRearrange first hFirstKind inputTensor =
          cast (congrArg (fun shape => Rep Nat shape) hOutputShape.symm)
            (denoteRearrange second hSecondKind
              (cast (congrArg (fun shape => Rep Nat shape) hInputShape)
                inputTensor))) :
    first.RearrangeEquivalent second hFirstKind hSecondKind
      hInputShape hOutputShape := by
  intro outputCoordinate
  apply Coord.linearize_injective
  apply Fin.ext
  let coordinateTensor : first.InputTensor Nat :=
    Rep.ofFn fun inputCoordinate => (Coord.linearize inputCoordinate).val
  have hAtCoordinate :=
    congrArg (fun tensor => tensor outputCoordinate)
      (hDenotations coordinateTensor)
  rw [cast_tensor_eq_reindex hInputShape coordinateTensor] at hAtCoordinate
  rw [cast_tensor_eq_reindex hOutputShape.symm] at hAtCoordinate
  simpa only [denoteRearrange, Rep.reindex_apply, coordinateTensor,
    Rep.get_ofFn, cast_eq_equivCast] using hAtCoordinate

/--
For fixed shapes, coordinate equivalence is exactly equality of rearrange
denotations on all natural-number tensors.
-/
theorem rearrangeEquivalent_iff_denoteRearrange_nat_eq
    {first second : CheckedTransform}
    {hFirstKind : first.value.normalized.kind = .rearrange}
    {hSecondKind : second.value.normalized.kind = .rearrange}
    {hInputShape :
      first.value.normalized.input =
        second.value.normalized.input}
    {hOutputShape : first.value.output = second.value.output} :
    first.RearrangeEquivalent second hFirstKind hSecondKind
        hInputShape hOutputShape ↔
      ∀ inputTensor : first.InputTensor Nat,
        denoteRearrange first hFirstKind inputTensor =
          cast (congrArg (fun shape => Rep Nat shape) hOutputShape.symm)
            (denoteRearrange second hSecondKind
              (cast (congrArg (fun shape => Rep Nat shape) hInputShape)
                inputTensor)) := by
  constructor
  · intro hEquivalent inputTensor
    exact denoteRearrange_eq_of_equivalent first second
      hFirstKind hSecondKind hInputShape hOutputShape hEquivalent inputTensor
  · exact rearrangeEquivalent_of_denoteRearrange_nat_eq first second
      hFirstKind hSecondKind hInputShape hOutputShape

/--
Applying a rearrangement after reindexing by its inverse coordinate map
recovers the output tensor. Together with `inverse_denoteRearrange`, this
states both inverse laws at the semantic level.
-/
@[simp] theorem denoteRearrange_inverse
    {α : Type u} [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (outputTensor : checked.OutputTensor α) :
    denoteRearrange checked hKind
        (Rep.reindex
          (checked.rearrangeCoordinateEquiv hKind).symm outputTensor) =
      outputTensor :=
  Rep.reindex_symm_reindex
    (checked.rearrangeCoordinateEquiv hKind).symm outputTensor

/--
Two successive checked rearrangements equal one reindexing by the composite
coordinate equivalence. The middle-shape cast is explicit because checked
plans store, rather than index over, their physical shapes.
-/
theorem denoteRearrange_comp
    {α : Type u} [Storage α]
    (first second : CheckedTransform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (hMiddleShape :
      first.value.output = second.value.normalized.input)
    (inputTensor : first.InputTensor α) :
    denoteRearrange second hSecondKind
        (cast (congrArg (fun shape => Rep α shape) hMiddleShape)
          (denoteRearrange first hFirstKind inputTensor)) =
      Rep.reindex
        ((second.rearrangeCoordinateEquiv hSecondKind).trans <|
          (Equiv.cast (congrArg Coord hMiddleShape.symm)).trans <|
            first.rearrangeCoordinateEquiv hFirstKind)
        inputTensor := by
  change
    Rep.reindex (second.rearrangeCoordinateEquiv hSecondKind)
        (cast (congrArg (fun shape => Rep α shape) hMiddleShape)
          (Rep.reindex
            (first.rearrangeCoordinateEquiv hFirstKind) inputTensor)) =
      _
  rw [cast_tensor_eq_reindex hMiddleShape
    (Rep.reindex
      (first.rearrangeCoordinateEquiv hFirstKind) inputTensor)]
  calc
    Rep.reindex (second.rearrangeCoordinateEquiv hSecondKind)
        (Rep.reindex
          (Equiv.cast (congrArg Coord hMiddleShape.symm))
          (Rep.reindex
            (first.rearrangeCoordinateEquiv hFirstKind) inputTensor)) =
      Rep.reindex (second.rearrangeCoordinateEquiv hSecondKind)
        (Rep.reindex
          ((Equiv.cast (congrArg Coord hMiddleShape.symm)).trans
            (first.rearrangeCoordinateEquiv hFirstKind))
          inputTensor) := by
            exact congrArg
              (Rep.reindex
                (second.rearrangeCoordinateEquiv hSecondKind))
              (Rep.reindex_trans
                (first.rearrangeCoordinateEquiv hFirstKind)
                (Equiv.cast (congrArg Coord hMiddleShape.symm))
                inputTensor)
    _ = _ :=
      Rep.reindex_trans
        ((Equiv.cast (congrArg Coord hMiddleShape.symm)).trans
          (first.rearrangeCoordinateEquiv hFirstKind))
        (second.rearrangeCoordinateEquiv hSecondKind)
        inputTensor

end Semantics

end TorchLean.Tensor.Internal
