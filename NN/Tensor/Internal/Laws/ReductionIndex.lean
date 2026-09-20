/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Laws.RowMajor
public import NN.Tensor.Internal.Lowering.Reduce
public import NN.Tensor.Internal.Laws.Equivalence -- shake: keep

/-!
# Reduction index laws

Reduction lowering enumerates retained output coordinates and removed-axis
coordinates separately. These laws combine those counters into one row-major
coordinate and identify the resulting compact arithmetic with the independent
checked reduction reconstruction.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Lowering.Reduce.Impl

open Check

/--
Compute a selected row-major index by decoding retained and reduced counters
separately.

The retained axes occupy the high-order digits and the reduced axes occupy
the low-order digits. Keeping the counters separate gives native reduction
loops a direct affine index program without changing the checked coordinate
semantics.
-/
def separatedRearrangeLinearIndex {ι : Type*} [BEq ι]
    (length : ι → Nat) (source left right : List ι)
    (leftIndex rightIndex : Nat) : Nat :=
  Rearrangement.Impl.rowMajorIndex (source.map length) <|
    source.map fun axis =>
      (Rearrangement.Impl.rowMajorCoordinates
          (left.map length) leftIndex ++
        Rearrangement.Impl.rowMajorCoordinates
          (right.map length) rightIndex).getD
        ((left ++ right).idxOf axis) 0

/--
Separately decoded row-major counters select the same source index as their
single combined row-major counter.
-/
theorem separatedRearrangeLinearIndex_eq {ι : Type*}
    [BEq ι] [LawfulBEq ι]
    (length : ι → Nat) (source left right : List ι)
    (hSource : ∀ axis, axis ∈ source → axis ∈ left ++ right)
    (leftIndex : Fin (Shape.size (left.map length)))
    (rightIndex : Fin (Shape.size (right.map length))) :
    separatedRearrangeLinearIndex length source left right
        leftIndex.val rightIndex.val =
      rearrangeLinearIndex length source (left ++ right)
        (rightIndex.val +
          Shape.size (right.map length) * leftIndex.val) := by
  let leftCoordinate :=
    AxisTuple.coordEquiv length left (Coord.unlinearize leftIndex)
  let rightCoordinate :=
    AxisTuple.coordEquiv length right (Coord.unlinearize rightIndex)
  let combinedTuple :=
    AxisTuple.append left leftCoordinate rightCoordinate
  have hCoordinateValues :
      Rearrangement.Impl.rowMajorCoordinates
            (left.map length) leftIndex.val ++
          Rearrangement.Impl.rowMajorCoordinates
            (right.map length) rightIndex.val =
        List.ofFn fun index => (combinedTuple index).val := by
    rw [rowMajorCoordinates_axisTuple length left leftIndex,
      rowMajorCoordinates_axisTuple length right rightIndex]
    exact (AxisTuple.values_append leftCoordinate rightCoordinate).symm
  unfold separatedRearrangeLinearIndex
  rw [hCoordinateValues]
  rw [select_values source (left ++ right) hSource combinedTuple]
  rw [rowMajorIndex_axisTuple]
  let combinedCoordinate :
      Coord ((left ++ right).map length) :=
    (AxisTuple.coordEquiv length (left ++ right)).symm combinedTuple
  let combinedIndex := Coord.linearize combinedCoordinate
  have hCombinedTuple :
      AxisTuple.coordEquiv length (left ++ right)
          (Coord.unlinearize combinedIndex) =
        combinedTuple := by
    change
      AxisTuple.coordEquiv length (left ++ right)
          (Coord.unlinearize (Coord.linearize combinedCoordinate)) =
        combinedTuple
    rw [Coord.unlinearize_linearize]
    exact Equiv.apply_symm_apply _ combinedTuple
  have hSelected :=
    linearize_axisTupleSelect length source (left ++ right)
      hSource combinedIndex
  rw [hCombinedTuple] at hSelected
  have hCombinedIndex :
      combinedIndex.val =
        rightIndex.val +
          Shape.size (right.map length) * leftIndex.val := by
    dsimp only [combinedIndex, combinedCoordinate, combinedTuple]
    rw [AxisTuple.linearize_append_val]
    simp only [leftCoordinate, rightCoordinate, Equiv.symm_apply_apply,
      Coord.linearize_unlinearize]
  rw [hCombinedIndex] at hSelected
  exact hSelected

/--
Compute a reduction input index as one compact row-major selection.

The removed-axis counter occupies the low-order digits because reduction
fibers are traversed inside each retained output coordinate.
-/
def reductionLinearIndex (checked : CheckedTransform)
    (outputIndex : Fin (Shape.size checked.value.output))
    (reducedIndex : Fin (Shape.size checked.reductionShape)) : Nat :=
  rearrangeLinearIndex checked.value.axisLength
    checked.value.normalized.inputAxes
    (checked.value.normalized.outputAxes ++ checked.reducedAxes)
    (reducedIndex.val +
      Shape.size checked.reductionShape * outputIndex.val)

/--
The compact row-major reduction index equals the independent reconstructed
input coordinate.
-/
theorem reductionInputFlatIndex_val
    (checked : CheckedTransform)
    (outputIndex : Fin (Shape.size checked.value.output))
    (reducedIndex : Fin (Shape.size checked.reductionShape)) :
    (reductionInputFlatIndex checked
      (Coord.unlinearize outputIndex) reducedIndex).val =
      reductionLinearIndex checked outputIndex reducedIndex := by
  change
    Fin
      (Shape.size
        (checked.reducedAxes.map checked.value.axisLength))
    at reducedIndex
  unfold reductionLinearIndex Check.CheckedTransform.reductionShape
  let outputAxes := checked.value.normalized.outputAxes
  let reducedAxes := checked.reducedAxes
  let combinedTuple :
      AxisTuple checked.value.axisLength (outputAxes ++ reducedAxes) :=
    AxisTuple.append outputAxes
      (checked.outputTensorCoordinateEquiv (Coord.unlinearize outputIndex))
      (AxisTuple.coordEquiv checked.value.axisLength reducedAxes <|
        Coord.unlinearize reducedIndex)
  let combinedCoordinate :
      Coord ((outputAxes ++ reducedAxes).map checked.value.axisLength) :=
    (AxisTuple.coordEquiv checked.value.axisLength
      (outputAxes ++ reducedAxes)).symm combinedTuple
  let combinedIndex := Coord.linearize combinedCoordinate
  have hCombinedIndex :
      combinedIndex.val = reducedIndex.val +
        Shape.size
          (checked.reducedAxes.map checked.value.axisLength) *
            outputIndex.val := by
    change
      (Coord.linearize
        ((AxisTuple.coordEquiv checked.value.axisLength
          (outputAxes ++ reducedAxes)).symm combinedTuple)).val = _
    rw [AxisTuple.linearize_append_val]
    simp only [outputAxes, reducedAxes, Equiv.symm_apply_apply]
    rw [Lowering.outputTensorCoordinateEquiv_eq_reshape]
    simp only [Equiv.trans_apply, Equiv.symm_apply_apply]
    rw [linearize_reshapeCoordEquiv_val]
    rw [show
      (Coord.linearize (Coord.unlinearize reducedIndex)).val =
          reducedIndex.val
        from congrArg Fin.val (Coord.linearize_unlinearize reducedIndex)]
    rw [show
      (Coord.linearize (Coord.unlinearize outputIndex)).val =
          outputIndex.val
        from congrArg Fin.val (Coord.linearize_unlinearize outputIndex)]
  unfold reductionInputFlatIndex reconstructedInputCoordinate
  rw [Lowering.inputTensorCoordinateEquiv_eq_reshape]
  simp only [Equiv.symm_trans_apply, Equiv.symm_symm]
  rw [linearize_reshapeCoordEquiv_val]
  change
    (Coord.linearize
      ((AxisTuple.coordEquiv checked.value.axisLength
        checked.value.normalized.inputAxes).symm <|
        AxisTuple.select
          (inputAxes_subset_output_append_reduced checked) combinedTuple)).val =
      _
  have hSelected :=
    linearize_axisTupleSelect checked.value.axisLength
      checked.value.normalized.inputAxes
      (outputAxes ++ reducedAxes)
      (inputAxes_subset_output_append_reduced checked) combinedIndex
  have hTuple :
      AxisTuple.coordEquiv checked.value.axisLength
          (outputAxes ++ reducedAxes)
          (Coord.unlinearize combinedIndex) =
        combinedTuple := by
    change
      AxisTuple.coordEquiv checked.value.axisLength
          (outputAxes ++ reducedAxes)
          (Coord.unlinearize (Coord.linearize combinedCoordinate)) =
        combinedTuple
    rw [Coord.unlinearize_linearize]
    exact Equiv.apply_symm_apply _ combinedTuple
  rw [hTuple] at hSelected
  rw [hSelected, hCombinedIndex]

end TorchLean.Tensor.Internal.Lowering.Reduce.Impl
