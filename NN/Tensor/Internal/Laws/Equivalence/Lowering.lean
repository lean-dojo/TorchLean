/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Laws.Equivalence.Plan
public import NN.Tensor.Internal.Laws.Equivalence.Semantics
public import NN.Tensor.Internal.Laws.Equivalence.Index -- shake: keep

/-!
# Lowered rearrangement equivalence

Compiler-level congruence theorems connect checked row-major index
certificates to primitive tensor programs.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

open Rearrangement.Impl

namespace Lowering

open Check

/--
Evaluate a checked rearrangement at any input coordinate selected by its
verified row-major index map.

This pointwise form avoids unfolding grouped coordinate equivalences in
downstream correspondence theorems. It applies to arbitrary ranks, grouping,
ungrouping, ellipses, and symbolic axis lengths.
-/
theorem rearrangeTensor_apply_eq_of_linearIndex_eq
    {α : Type*} [Storage α]
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .rearrange)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output)
    (inputCoordinate : Coord checked.value.normalized.input)
    (hLinearIndex :
      rearrangeLinearIndex checked.value.axisLength
          checked.value.normalized.inputAxes
          checked.value.normalized.outputAxes
          (Coord.linearize outputCoordinate).val =
        (Coord.linearize inputCoordinate).val) :
    rearrangeTensor checked hKind inputTensor outputCoordinate =
      inputTensor inputCoordinate := by
  rw [rearrangeTensor_correct, Semantics.denoteRearrange_apply]
  congr 1
  apply Coord.linearize_injective
  apply Fin.ext
  calc
    (Coord.linearize
        (checked.rearrangeCoordinateEquiv hKind outputCoordinate)).val =
        rearrangeLinearIndex checked.value.axisLength
          checked.value.normalized.inputAxes
          checked.value.normalized.outputAxes
          (Coord.linearize outputCoordinate).val := by
      simpa only [Coord.unlinearize_linearize] using
        checked.rearrangeCoordinateEquiv_linearize hKind
          (Coord.linearize outputCoordinate)
    _ = (Coord.linearize inputCoordinate).val := hLinearIndex

/--
Equivalent checked rearrangements compile to equal primitive tensor
programs. This is the compiler-level congruence theorem used by proof
automation for differently spelled but extensionally equal patterns.
-/
theorem rearrangeTensor_eq_of_equivalent
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
    rearrangeTensor first hFirstKind inputTensor =
      cast (congrArg (fun shape => Rep α shape) hOutputShape.symm)
        (rearrangeTensor second hSecondKind
          (cast (congrArg (fun shape => Rep α shape) hInputShape)
            inputTensor)) := by
  rw [rearrangeTensor_correct, rearrangeTensor_correct]
  exact Semantics.denoteRearrange_eq_of_equivalent first second
    hFirstKind hSecondKind hInputShape hOutputShape hEquivalent inputTensor

/--
Two checked rearrangements compile to the same tensor program when their
compact row-major index maps agree.

Unlike enumerating all concrete coordinates, the premise is a single
arithmetic formula over an arbitrary output index. This is suitable both for
general hand-written proofs and for reducing reflected literal plans inside
the `einops` tactic.
-/
theorem rearrangeTensor_eq_of_linearIndex_eq
    {α : Type*} [Storage α]
    (first second : CheckedTransform)
    (hFirstKind : first.value.normalized.kind = .rearrange)
    (hSecondKind : second.value.normalized.kind = .rearrange)
    (hInputShape :
      first.value.normalized.input =
        second.value.normalized.input)
    (hOutputShape :
      first.value.output = second.value.output)
    (hLinearIndex :
      ∀ outputIndex : Fin (Shape.size first.value.output),
        rearrangeLinearIndex first.value.axisLength
            first.value.normalized.inputAxes
            first.value.normalized.outputAxes outputIndex.val =
          rearrangeLinearIndex second.value.axisLength
            second.value.normalized.inputAxes
            second.value.normalized.outputAxes
            (finCongr (congrArg Shape.size hOutputShape)
              outputIndex).val)
    (inputTensor : first.InputTensor α) :
    rearrangeTensor first hFirstKind inputTensor =
      cast (congrArg (fun shape => Rep α shape) hOutputShape.symm)
        (rearrangeTensor second hSecondKind
          (cast (congrArg (fun shape => Rep α shape) hInputShape)
            inputTensor)) := by
  apply rearrangeTensor_eq_of_equivalent first second
    hFirstKind hSecondKind hInputShape hOutputShape
  intro outputCoordinate
  apply Coord.linearize_injective
  apply Fin.ext
  rw [linearize_cast_val hInputShape]
  rw [← Coord.unlinearize_linearize
    (cast (congrArg Coord hOutputShape) outputCoordinate)]
  rw [← Coord.unlinearize_linearize outputCoordinate]
  rw [Check.CheckedTransform.rearrangeCoordinateEquiv_linearize,
    Check.CheckedTransform.rearrangeCoordinateEquiv_linearize]
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
theorem rearrangeTensor_comp_eq_of_linearIndex_eq
    {α : Type*} [Storage α]
    (first second direct : CheckedTransform)
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
        rearrangeLinearIndex first.value.axisLength
            first.value.normalized.inputAxes
            first.value.normalized.outputAxes
            (rearrangeLinearIndex second.value.axisLength
              second.value.normalized.inputAxes
              second.value.normalized.outputAxes outputIndex.val) =
          rearrangeLinearIndex direct.value.axisLength
            direct.value.normalized.inputAxes
            direct.value.normalized.outputAxes
            (finCongr (congrArg Shape.size hOutputShape)
              outputIndex).val)
    (inputTensor : first.InputTensor α) :
    rearrangeTensor second hSecondKind
        (cast (congrArg (fun shape => Rep α shape) hMiddleShape)
          (rearrangeTensor first hFirstKind inputTensor)) =
      cast (congrArg (fun shape => Rep α shape) hOutputShape.symm)
        (rearrangeTensor direct hDirectKind
          (cast (congrArg (fun shape => Rep α shape) hInputShape)
            inputTensor)) := by
  rw [rearrangeTensor_correct, rearrangeTensor_correct,
    rearrangeTensor_correct]
  rw [Semantics.denoteRearrange_comp first second
    hFirstKind hSecondKind hMiddleShape inputTensor]
  rw [cast_tensor_eq_reindex hInputShape inputTensor]
  rw [cast_tensor_eq_reindex hOutputShape.symm]
  ext outputCoordinate
  simp only [Semantics.denoteRearrange, Rep.reindex_apply]
  apply congrArg inputTensor
  apply Coord.linearize_injective
  apply Fin.ext
  rw [linearize_equivCast_val hInputShape]
  rw [← Coord.unlinearize_linearize
    (Equiv.cast (congrArg Coord hOutputShape) outputCoordinate)]
  rw [← Coord.unlinearize_linearize outputCoordinate]
  rw [Check.CheckedTransform.rearrangeCoordinateEquiv_linearize]
  rw [linearize_equivCast hOutputShape, Coord.linearize_unlinearize]
  rw [Coord.unlinearize_linearize]
  simp only [Equiv.trans_apply]
  rw [← Coord.unlinearize_linearize
    (Equiv.cast (congrArg Coord hMiddleShape.symm)
      (second.rearrangeCoordinateEquiv hSecondKind outputCoordinate))]
  rw [Check.CheckedTransform.rearrangeCoordinateEquiv_linearize]
  rw [linearize_equivCast_val hMiddleShape]
  rw [← Coord.unlinearize_linearize outputCoordinate]
  rw [Check.CheckedTransform.rearrangeCoordinateEquiv_linearize]
  simpa only [Coord.linearize_unlinearize] using
    hLinearIndex (Coord.linearize outputCoordinate)

end Lowering

end TorchLean.Tensor.Internal
