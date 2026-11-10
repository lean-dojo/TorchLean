/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Equivalence.Semantics

/-!
# Lowered rearrangement equivalence

Compiler-level congruence theorems connect checked row-major index
certificates to primitive tensor programs.
-/

@[expose] public section

namespace TorchLean.Einops

universe u

open Tensor.Rep

namespace Lowering

open Shape (Coord)
open Tensor (Rep)
open Shape.Coord (linearize_cast_val linearize_cast)

/--
A checked rearrangement takes its value at an output coordinate from any input coordinate whose
flat index is the one `Einops.Rearrangement.linearIndex` computes from the output's flat index.

The pointwise form avoids unfolding grouped coordinate equivalences in downstream correspondence
theorems and applies to arbitrary ranks, grouping, ungrouping, ellipses, and symbolic axis
lengths.
-/
theorem rearrange_apply_of_linearIndex
    {α : Type*} [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output)
    (inputCoordinate : Coord checked.value.normalized.input)
    (hLinearIndex :
      Rearrangement.linearIndex checked.value.axisLength
          checked.value.normalized.inputAxes
          checked.value.normalized.outputAxes
          (Coord.linearize outputCoordinate).val =
        (Coord.linearize inputCoordinate).val) :
    rearrange checked hKind inputTensor outputCoordinate =
      inputTensor inputCoordinate := by
  rw [rearrange_correct, Semantics.denoteRearrange_apply]
  congr 1
  apply Coord.linearize_injective
  apply Fin.ext
  calc
    (Coord.linearize
        (checked.rearrangeEquiv hKind outputCoordinate)).val =
        Rearrangement.linearIndex checked.value.axisLength
          checked.value.normalized.inputAxes
          checked.value.normalized.outputAxes
          (Coord.linearize outputCoordinate).val := by
      simpa only [Coord.unlinearize_linearize] using
        checked.rearrangeEquiv_linearize hKind
          (Coord.linearize outputCoordinate)
    _ = (Coord.linearize inputCoordinate).val := hLinearIndex

/--
Equivalent checked rearrangements compile to equal primitive tensor programs. Proof automation
uses this compiler-level congruence for patterns that are spelled differently but extensionally
equal.
-/
theorem rearrange_congr
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
    rearrange first hFirstKind inputTensor =
      cast (congrArg (fun shape => Rep α shape) hOutputShape.symm)
        (rearrange second hSecondKind
          (cast (congrArg (fun shape => Rep α shape) hInputShape)
            inputTensor)) := by
  rw [rearrange_correct, rearrange_correct]
  exact Semantics.denoteRearrange_congr first second
    hFirstKind hSecondKind hInputShape hOutputShape hEquivalent inputTensor

/--
Two checked rearrangements compile to the same tensor program when their
compact row-major index maps agree.

Unlike enumerating all concrete coordinates, the premise is a single
arithmetic formula over an arbitrary output index. This is suitable both for
general hand-written proofs and for reducing reflected literal plans inside
the `einops` tactic.
-/
theorem rearrange_eq_of_linearIndex
    {α : Type*} [Storage α]
    (first second : Transform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (hInputShape :
      first.value.normalized.input =
        second.value.normalized.input)
    (hOutputShape :
      first.value.output = second.value.output)
    (hLinearIndex :
      ∀ outputIndex : Fin (Shape.size first.value.output),
        Rearrangement.linearIndex first.value.axisLength
            first.value.normalized.inputAxes
            first.value.normalized.outputAxes outputIndex.val =
          Rearrangement.linearIndex second.value.axisLength
            second.value.normalized.inputAxes
            second.value.normalized.outputAxes
            (finCongr (congrArg Shape.size hOutputShape)
              outputIndex).val)
    (inputTensor : first.InputTensor α) :
    rearrange first hFirstKind inputTensor =
      cast (congrArg (fun shape => Rep α shape) hOutputShape.symm)
        (rearrange second hSecondKind
          (cast (congrArg (fun shape => Rep α shape) hInputShape)
            inputTensor)) := by
  apply rearrange_congr first second
    hFirstKind hSecondKind hInputShape hOutputShape
  intro outputCoordinate
  apply Coord.linearize_injective
  apply Fin.ext
  rw [linearize_cast_val hInputShape]
  rw [← Coord.unlinearize_linearize
    (cast (congrArg Coord hOutputShape) outputCoordinate)]
  rw [← Coord.unlinearize_linearize outputCoordinate]
  rw [Transform.rearrangeEquiv_linearize,
    Transform.rearrangeEquiv_linearize]
  rw [linearize_cast hOutputShape, Coord.linearize_unlinearize]
  exact hLinearIndex (Coord.linearize outputCoordinate)

/--
Two successive checked rearrangements equal one direct rearrangement when the
composite of their compact row-major index maps equals the direct map.

All three plans and all physical-shape transports are arbitrary. The theorem
therefore covers transpositions, grouping and ungrouping, ellipses, and their
compositions without introducing a separate representation of composed
plans.
-/
theorem rearrange_comp_of_linearIndex
    {α : Type*} [Storage α]
    (first second direct : Transform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (hDirectKind : direct.value.normalized.kind = .rearrange)
    (hMiddleShape :
      first.value.output =
        second.value.normalized.input)
    (hInputShape :
      first.value.normalized.input =
        direct.value.normalized.input)
    (hOutputShape :
      second.value.output = direct.value.output)
    (hLinearIndex :
      ∀ outputIndex : Fin (Shape.size second.value.output),
        Rearrangement.linearIndex first.value.axisLength
            first.value.normalized.inputAxes
            first.value.normalized.outputAxes
            (Rearrangement.linearIndex second.value.axisLength
              second.value.normalized.inputAxes
              second.value.normalized.outputAxes outputIndex.val) =
          Rearrangement.linearIndex direct.value.axisLength
            direct.value.normalized.inputAxes
            direct.value.normalized.outputAxes
            (finCongr (congrArg Shape.size hOutputShape)
              outputIndex).val)
    (inputTensor : first.InputTensor α) :
    rearrange second hSecondKind
        (cast (congrArg (fun shape => Rep α shape) hMiddleShape)
          (rearrange first hFirstKind inputTensor)) =
      cast (congrArg (fun shape => Rep α shape) hOutputShape.symm)
        (rearrange direct hDirectKind
          (cast (congrArg (fun shape => Rep α shape) hInputShape)
            inputTensor)) := by
  rw [rearrange_correct, rearrange_correct,
    rearrange_correct]
  rw [Semantics.denoteRearrange_comp first second
    hFirstKind hSecondKind hMiddleShape inputTensor]
  rw [cast_eq_reindex hInputShape inputTensor]
  rw [cast_eq_reindex hOutputShape.symm]
  ext outputCoordinate
  simp only [Semantics.denoteRearrange, Rep.reindex_apply]
  apply congrArg inputTensor
  apply Coord.linearize_injective
  apply Fin.ext
  rw [Equiv.cast_apply (congrArg Coord hInputShape.symm), linearize_cast_val hInputShape]
  rw [← Coord.unlinearize_linearize
    (Equiv.cast (congrArg Coord hOutputShape) outputCoordinate)]
  rw [← Coord.unlinearize_linearize outputCoordinate]
  rw [Transform.rearrangeEquiv_linearize]
  rw [Equiv.cast_apply (congrArg Coord hOutputShape), linearize_cast hOutputShape,
    Coord.linearize_unlinearize]
  rw [Coord.unlinearize_linearize]
  simp only [Equiv.trans_apply]
  rw [← Coord.unlinearize_linearize
    (Equiv.cast (congrArg Coord hMiddleShape.symm)
      (second.rearrangeEquiv hSecondKind outputCoordinate))]
  rw [Transform.rearrangeEquiv_linearize]
  rw [Equiv.cast_apply (congrArg Coord hMiddleShape.symm),
    linearize_cast_val hMiddleShape]
  rw [← Coord.unlinearize_linearize outputCoordinate]
  rw [Transform.rearrangeEquiv_linearize]
  simpa only [Coord.linearize_unlinearize] using
    hLinearIndex (Coord.linearize outputCoordinate)

end Lowering

end TorchLean.Einops
