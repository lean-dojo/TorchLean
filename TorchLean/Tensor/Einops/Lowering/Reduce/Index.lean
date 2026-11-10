/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.AxisTuple.RowMajor
public import TorchLean.Tensor.Einops.Lowering.Reduce
public import TorchLean.Tensor.Einops.Equivalence.Lowering -- shake: keep

/-!
# Reduction index laws

Reduction lowering enumerates retained output coordinates and removed-axis
coordinates with separate counters. These laws combine the two counters into one
row-major index and prove that the resulting arithmetic agrees with the checked
reconstruction of the input coordinate.
-/

@[expose] public section

namespace TorchLean.Einops.Lowering.Reduction

open Shape (Coord)
open Rearrangement (rowMajorCoordinates_axisTuple)

/--
Compute a selected row-major index by decoding retained and reduced counters
separately.

In the combined counter, retained axes occupy the high-order digits and reduced
axes occupy the low-order digits. The resulting coordinates are then selected
in source-axis order. Keeping the counters separate gives native reduction
loops an index program without changing the checked coordinate semantics.
-/
def separatedIndex {ι : Type*} [BEq ι]
    (length : ι → Nat) (source left right : List ι)
    (leftIndex rightIndex : Nat) : Nat :=
  Rearrangement.rowMajorIndex (source.map length) <|
    source.map fun axis =>
      (Rearrangement.rowMajorCoordinates
          (left.map length) leftIndex ++
        Rearrangement.rowMajorCoordinates
          (right.map length) rightIndex).getD
        ((left ++ right).idxOf axis) 0

/--
Separately decoded row-major counters select the same source index as their
single combined row-major counter.
-/
theorem separatedIndex_eq {ι : Type*}
    [BEq ι] [LawfulBEq ι]
    (length : ι → Nat) (source left right : List ι)
    (hSource : ∀ axis, axis ∈ source → axis ∈ left ++ right)
    (leftIndex : Fin (Shape.size (left.map length)))
    (rightIndex : Fin (Shape.size (right.map length))) :
    separatedIndex length source left right
        leftIndex.val rightIndex.val =
      Rearrangement.linearIndex length source (left ++ right)
        (rightIndex.val +
          Shape.size (right.map length) * leftIndex.val) := by
  let leftCoordinate :=
    AxisTuple.coordEquiv length left (Coord.unlinearize leftIndex)
  let rightCoordinate :=
    AxisTuple.coordEquiv length right (Coord.unlinearize rightIndex)
  let combinedTuple :=
    AxisTuple.append left leftCoordinate rightCoordinate
  have hCoordinateValues :
      Rearrangement.rowMajorCoordinates
            (left.map length) leftIndex.val ++
          Rearrangement.rowMajorCoordinates
            (right.map length) rightIndex.val =
        List.ofFn fun index => (combinedTuple index).val := by
    rw [rowMajorCoordinates_axisTuple length left leftIndex,
      rowMajorCoordinates_axisTuple length right rightIndex]
    exact (AxisTuple.values_append leftCoordinate rightCoordinate).symm
  unfold separatedIndex
  rw [hCoordinateValues]
  rw [AxisTuple.select_values source (left ++ right) hSource combinedTuple]
  rw [Rearrangement.rowMajorIndex_axisTuple]
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
    Rearrangement.linearize_select length source (left ++ right)
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
def linearIndex (checked : Transform)
    (outputIndex : Fin (Shape.size checked.value.output))
    (reducedIndex : Fin (Shape.size checked.reductionShape)) : Nat :=
  Rearrangement.linearIndex checked.value.axisLength
    checked.value.normalized.inputAxes
    (checked.value.normalized.outputAxes ++ checked.reducedAxes)
    (reducedIndex.val +
      Shape.size checked.reductionShape * outputIndex.val)

/--
The compact row-major reduction index equals the independent reconstructed
input coordinate.
-/
theorem inputFlatIndex_val
    (checked : Transform)
    (outputIndex : Fin (Shape.size checked.value.output))
    (reducedIndex : Fin (Shape.size checked.reductionShape)) :
    (inputFlatIndex checked
      (Coord.unlinearize outputIndex) reducedIndex).val =
      linearIndex checked outputIndex reducedIndex := by
  change
    Fin
      (Shape.size
        (checked.reducedAxes.map checked.value.axisLength))
    at reducedIndex
  unfold linearIndex Transform.reductionShape
  let outputAxes := checked.value.normalized.outputAxes
  let reducedAxes := checked.reducedAxes
  let combinedTuple :
      AxisTuple checked.value.axisLength (outputAxes ++ reducedAxes) :=
    AxisTuple.append outputAxes
      (checked.outputCoordEquiv (Coord.unlinearize outputIndex))
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
    rw [Transform.outputCoordEquiv_eq_reshape]
    simp only [Equiv.trans_apply, Equiv.symm_apply_apply]
    rw [Coord.linearize_reshapeEquiv_val]
    rw [show
      (Coord.linearize (Coord.unlinearize reducedIndex)).val =
          reducedIndex.val
        from congrArg Fin.val (Coord.linearize_unlinearize reducedIndex)]
    rw [show
      (Coord.linearize (Coord.unlinearize outputIndex)).val =
          outputIndex.val
        from congrArg Fin.val (Coord.linearize_unlinearize outputIndex)]
  unfold inputFlatIndex inputCoord
  rw [Transform.inputCoordEquiv_eq_reshape]
  simp only [Equiv.symm_trans_apply, Equiv.symm_symm]
  rw [Coord.linearize_reshapeEquiv_val]
  change
    (Coord.linearize
      ((AxisTuple.coordEquiv checked.value.axisLength
        checked.value.normalized.inputAxes).symm <|
        AxisTuple.select
          (inputAxes_subset_append checked) combinedTuple)).val =
      _
  have hSelected :=
    Rearrangement.linearize_select checked.value.axisLength
      checked.value.normalized.inputAxes
      (outputAxes ++ reducedAxes)
      (inputAxes_subset_append checked) combinedIndex
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

end TorchLean.Einops.Lowering.Reduction
