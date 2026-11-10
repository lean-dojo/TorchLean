/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Equivalence.Plan

/-!
# Semantic rearrangement equivalence

Coordinate-equivalent checked rearrangements have equal tensor denotations.
The converse is witnessed by natural-number coordinate tensors.
-/

@[expose] public section

namespace TorchLean.Einops

universe u

open Tensor.Rep

section

open Tensor (Rep)
open Semantics (denoteRearrange)
open Shape (Coord)

/--
Equivalent rearrange coordinate maps produce equal tensors over every scalar
type. The casts only transport tensors across the supplied physical-shape
equalities.
-/
theorem Semantics.denoteRearrange_congr
    {α : Type u} [Storage α]
    (first second : Transform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (hInputShape :
      first.value.normalized.input =
        second.value.normalized.input)
    (hOutputShape : first.value.output = second.value.output)
    (hEquivalent :
      first.Equivalent second hFirstKind hSecondKind
        hInputShape hOutputShape)
    (inputTensor : first.InputTensor α) :
    denoteRearrange first hFirstKind inputTensor =
      cast (congrArg (fun shape => Rep α shape) hOutputShape.symm)
        (denoteRearrange second hSecondKind
          (cast (congrArg (fun shape => Rep α shape) hInputShape)
            inputTensor)) := by
  rw [cast_eq_reindex hInputShape inputTensor]
  rw [cast_eq_reindex hOutputShape.symm]
  ext outputCoordinate
  simpa only [denoteRearrange, Rep.reindex_apply, Equiv.cast_apply] using
    congrArg inputTensor (hEquivalent outputCoordinate)

/--
Natural-number coordinate tensors separate rearrange maps, so equality of the denotations on all
`Nat` tensors is sufficient for extensional pattern equivalence. By
`denoteRearrange_congr` it is also necessary.
-/
theorem Transform.equivalent_of_denoteRearrange
    (first second : Transform)
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
    first.Equivalent second hFirstKind hSecondKind
      hInputShape hOutputShape := by
  intro outputCoordinate
  apply Coord.linearize_injective
  apply Fin.ext
  let coordinateTensor : first.InputTensor Nat :=
    Rep.ofFn fun inputCoordinate => (Coord.linearize inputCoordinate).val
  have hAtCoordinate :=
    congrArg (fun tensor => tensor outputCoordinate)
      (hDenotations coordinateTensor)
  rw [cast_eq_reindex hInputShape coordinateTensor] at hAtCoordinate
  rw [cast_eq_reindex hOutputShape.symm] at hAtCoordinate
  simpa only [denoteRearrange, Rep.reindex_apply, coordinateTensor,
    Rep.get_ofFn, Equiv.cast_apply] using hAtCoordinate

/--
For fixed shapes, coordinate equivalence is exactly equality of rearrange
denotations on all natural-number tensors.
-/
theorem Transform.equivalent_iff_denoteRearrange
    {first second : Transform}
    {hFirstKind : first.value.normalized.kind = .rearrange}
    {hSecondKind : second.value.normalized.kind = .rearrange}
    {hInputShape :
      first.value.normalized.input =
        second.value.normalized.input}
    {hOutputShape : first.value.output = second.value.output} :
    first.Equivalent second hFirstKind hSecondKind
        hInputShape hOutputShape ↔
      ∀ inputTensor : first.InputTensor Nat,
        denoteRearrange first hFirstKind inputTensor =
          cast (congrArg (fun shape => Rep Nat shape) hOutputShape.symm)
            (denoteRearrange second hSecondKind
              (cast (congrArg (fun shape => Rep Nat shape) hInputShape)
                inputTensor)) := by
  constructor
  · intro hEquivalent inputTensor
    exact Semantics.denoteRearrange_congr first second
      hFirstKind hSecondKind hInputShape hOutputShape hEquivalent inputTensor
  · exact equivalent_of_denoteRearrange first second
      hFirstKind hSecondKind hInputShape hOutputShape

/--
Applying a rearrangement after reindexing by its inverse coordinate map
recovers the output tensor. Together with `inverse_denoteRearrange`, this
states both inverse laws at the semantic level.
-/
@[simp] theorem Semantics.denoteRearrange_inverse
    {α : Type u} [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (outputTensor : checked.OutputTensor α) :
    denoteRearrange checked hKind
        (Rep.reindex
          (checked.rearrangeEquiv hKind).symm outputTensor) =
      outputTensor :=
  Rep.reindex_symm_reindex
    (checked.rearrangeEquiv hKind).symm outputTensor

/--
Two successive checked rearrangements equal one reindexing by the composite
coordinate equivalence. The statement casts across the middle shape because checked plans store
their physical shapes as fields, and the two middle shapes are equal only by hypothesis.
-/
theorem Semantics.denoteRearrange_comp
    {α : Type u} [Storage α]
    (first second : Transform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (hMiddleShape :
      first.value.output = second.value.normalized.input)
    (inputTensor : first.InputTensor α) :
    denoteRearrange second hSecondKind
        (cast (congrArg (fun shape => Rep α shape) hMiddleShape)
          (denoteRearrange first hFirstKind inputTensor)) =
      Rep.reindex
        ((second.rearrangeEquiv hSecondKind).trans <|
          (Equiv.cast (congrArg Coord hMiddleShape.symm)).trans <|
            first.rearrangeEquiv hFirstKind)
        inputTensor := by
  change
    Rep.reindex (second.rearrangeEquiv hSecondKind)
        (cast (congrArg (fun shape => Rep α shape) hMiddleShape)
          (Rep.reindex
            (first.rearrangeEquiv hFirstKind) inputTensor)) =
      _
  rw [cast_eq_reindex hMiddleShape
    (Rep.reindex
      (first.rearrangeEquiv hFirstKind) inputTensor)]
  rw [Rep.reindex_trans (first.rearrangeEquiv hFirstKind),
    Rep.reindex_trans]

end

end TorchLean.Einops
