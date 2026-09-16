/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Mathlib.Algebra.BigOperators.Fin
public import NN.Tensor.Internal.Representation.Basic.Reindex -- shake: keep
public import NN.Tensor.Internal.Representation.Basic.Traversal -- shake: keep

/-!
# Tensor Segments Along an Arbitrary Axis

Packing is concatenation along one tensor axis, while unpacking is the
corresponding family of half-open slices. This module defines those operations
independently of pattern syntax and scalar algebra.

`Coord.appendEquiv` separates a coordinate into coordinates for a shape leadingShape
and trailingShape. The tensor operations use it to isolate the selected axis:

* `Rep.concatenateAxis` concatenates two tensors;
* `Rep.sliceAxis` reads one checked half-open interval;
* `Rep.concatenateAxes` concatenates a finite dependent family; and
* `Rep.splitAxis` partitions one axis according to a list of lengths.

The two family operations are proved mutually inverse, including empty
families and zero-length segments. Consequently these primitives preserve all
scalar values without requiring an operation on the scalar type.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

/--
Identify an index in a list of consecutive segments with its index in the
concatenated interval.

The first component selects a segment and the second selects a position
inside that segment. Mathlib's dependent finite-sum equivalence places the
segments in list order, including segments of length zero.
-/
def segmentIndexEquiv (lengths : List Nat) :
    ((segment : Fin lengths.length) × Fin (lengths.get segment)) ≃
      Fin lengths.sum :=
  finSigmaFinEquiv.trans <|
    finCongr <| by
      simp

namespace Coord

/--
Separate a coordinate of an appended shape into its leadingShape and trailingShape
coordinates.

The equivalence follows the recursive shape representation, so it preserves
the usual outermost-first coordinate order.
-/
def appendEquiv : (leadingShape trailingShape : Shape) →
    Coord (leadingShape ++ trailingShape) ≃ Coord leadingShape × Coord trailingShape
  | [], trailingShape => (Equiv.punitProd (Coord trailingShape)).symm
  | length :: leadingShape, trailingShape =>
      (Equiv.prodCongr (Equiv.refl (Fin length))
          (appendEquiv leadingShape trailingShape)).trans <|
        (Equiv.prodAssoc (Fin length) (Coord leadingShape) (Coord trailingShape)).symm

end Coord

namespace Rep

/--
Concatenate two tensors along the axis following `leadingShape`.

The dimensions in `leadingShape` and `trailingShape` are shared. Coordinates below
`leftLength` read the left tensor, and the remaining coordinates read the
right tensor after subtracting `leftLength`.
-/
def concatenateAxis {α : Type u} [Storage α]
    (leadingShape trailingShape : Shape)
    {leftLength rightLength : Nat}
    (leftTensor : Rep α (leadingShape ++ leftLength :: trailingShape))
    (rightTensor : Rep α (leadingShape ++ rightLength :: trailingShape)) :
    Rep α (leadingShape ++ (leftLength + rightLength) :: trailingShape) :=
  Rep.ofFn fun outputCoordinate =>
    let separated :=
      Coord.appendEquiv leadingShape ((leftLength + rightLength) :: trailingShape)
        outputCoordinate
    if hLeft : separated.2.1.val < leftLength then
      leftTensor <|
        (Coord.appendEquiv leadingShape (leftLength :: trailingShape)).symm
          (separated.1, ⟨⟨separated.2.1.val, hLeft⟩, separated.2.2⟩)
    else
      rightTensor <|
        (Coord.appendEquiv leadingShape (rightLength :: trailingShape)).symm
          (separated.1,
            ⟨⟨separated.2.1.val - leftLength, by omega⟩, separated.2.2⟩)

/--
Read a checked half-open interval from the axis following `leadingShape`.

The output coordinate `i` reads source coordinate `start + i`. The bound
proof prevents out-of-range slicing before tensor execution.
-/
def sliceAxis {α : Type u} [Storage α]
    (leadingShape trailingShape : Shape)
    {sourceLength : Nat} (start sliceLength : Nat)
    (hBounds : start + sliceLength ≤ sourceLength)
    (sourceTensor : Rep α (leadingShape ++ sourceLength :: trailingShape)) :
    Rep α (leadingShape ++ sliceLength :: trailingShape) :=
  Rep.ofFn fun outputCoordinate =>
    let separated :=
      Coord.appendEquiv leadingShape (sliceLength :: trailingShape) outputCoordinate
    sourceTensor <|
      (Coord.appendEquiv leadingShape (sourceLength :: trailingShape)).symm
        (separated.1,
          ⟨⟨start + separated.2.1.val, by omega⟩, separated.2.2⟩)

/-- The first slice of a binary concatenation recovers the left tensor. -/
@[simp] theorem sliceAxis_concatenateAxis_left {α : Type u}
    [Storage α]
    (leadingShape trailingShape : Shape) {leftLength rightLength : Nat}
    (leftTensor : Rep α (leadingShape ++ leftLength :: trailingShape))
    (rightTensor : Rep α (leadingShape ++ rightLength :: trailingShape)) :
    sliceAxis leadingShape trailingShape 0 leftLength (by omega)
        (concatenateAxis leadingShape trailingShape leftTensor rightTensor) =
      leftTensor := by
  ext outputCoordinate
  simp only [sliceAxis, concatenateAxis, zero_add, get_ofFn,
    Equiv.apply_symm_apply, Fin.is_lt, ↓reduceDIte, Fin.eta]
  congr 1
  exact
    (Coord.appendEquiv leadingShape (leftLength :: trailingShape)).symm_apply_apply
      outputCoordinate

/-- The second slice of a binary concatenation recovers the right tensor. -/
@[simp] theorem sliceAxis_concatenateAxis_right {α : Type u}
    [Storage α]
    (leadingShape trailingShape : Shape) {leftLength rightLength : Nat}
    (leftTensor : Rep α (leadingShape ++ leftLength :: trailingShape))
    (rightTensor : Rep α (leadingShape ++ rightLength :: trailingShape)) :
    sliceAxis leadingShape trailingShape leftLength rightLength (by omega)
        (concatenateAxis leadingShape trailingShape leftTensor rightTensor) =
      rightTensor := by
  ext outputCoordinate
  simp only [sliceAxis, concatenateAxis, get_ofFn, Equiv.apply_symm_apply,
    Nat.add_sub_cancel_left, Fin.eta]
  split
  · omega
  · congr 1
    exact
      (Coord.appendEquiv leadingShape (rightLength :: trailingShape)).symm_apply_apply
        outputCoordinate

/--
Concatenating two adjacent slices that partition an axis recovers the source
tensor.
-/
@[simp] theorem concatenateAxis_slices {α : Type u}
    [Storage α]
    (leadingShape trailingShape : Shape) (leftLength rightLength : Nat)
    (sourceTensor :
      Rep α (leadingShape ++ (leftLength + rightLength) :: trailingShape)) :
    concatenateAxis leadingShape trailingShape
        (sliceAxis leadingShape trailingShape 0 leftLength (by omega) sourceTensor)
        (sliceAxis leadingShape trailingShape leftLength rightLength (by omega) sourceTensor) =
      sourceTensor := by
  ext sourceCoordinate
  let separated :=
    Coord.appendEquiv leadingShape ((leftLength + rightLength) :: trailingShape)
      sourceCoordinate
  by_cases hLeft : separated.2.1.val < leftLength
  · simp only [concatenateAxis, sliceAxis, zero_add, get_ofFn,
      Equiv.apply_symm_apply, Fin.eta, hLeft, ↓reduceDIte, separated]
    congr 1
    exact
      (Coord.appendEquiv leadingShape
        ((leftLength + rightLength) :: trailingShape)).symm_apply_apply
          sourceCoordinate
  · have hLeftLe : leftLength ≤ separated.2.1.val :=
      Nat.le_of_not_gt hLeft
    have hRecombine :
        leftLength + (separated.2.1.val - leftLength) =
          separated.2.1.val :=
      Nat.add_sub_of_le hLeftLe
    simp only [concatenateAxis, sliceAxis, zero_add, get_ofFn,
      Equiv.apply_symm_apply, Fin.eta, hLeft, ↓reduceDIte, hRecombine,
      separated]
    congr 1
    exact
      (Coord.appendEquiv leadingShape
        ((leftLength + rightLength) :: trailingShape)).symm_apply_apply
          sourceCoordinate

/--
Identify a component coordinate in a segmented tensor family with its
coordinate after concatenation along the selected axis.

Leading and trailing coordinates are preserved. The local selected-axis
coordinate is placed in the component's consecutive segment by
`segmentIndexEquiv`.
-/
def concatenateAxesCoordinateEquiv (leadingShape trailingShape : Shape)
    (lengths : List Nat)
    : ((segment : Fin lengths.length) ×
        Coord (leadingShape ++ lengths.get segment :: trailingShape)) ≃
      Coord (leadingShape ++ lengths.sum :: trailingShape) :=
  let distribute :
      ((segment : Fin lengths.length) ×
        (Coord leadingShape ×
          (Fin (lengths.get segment) × Coord trailingShape))) ≃
        Coord leadingShape ×
          (((segment : Fin lengths.length) ×
              Fin (lengths.get segment)) × Coord trailingShape) :=
    { toFun := fun coordinate =>
        (coordinate.2.1,
          (⟨coordinate.1, coordinate.2.2.1⟩, coordinate.2.2.2))
      invFun := fun coordinate =>
        ⟨coordinate.2.1.1,
          (coordinate.1, coordinate.2.1.2, coordinate.2.2)⟩
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  (Equiv.sigmaCongrRight fun segment =>
      Coord.appendEquiv leadingShape
        (lengths.get segment :: trailingShape)).trans <|
    distribute |>.trans <|
      (Equiv.prodCongr (Equiv.refl (Coord leadingShape))
        (Equiv.prodCongr (segmentIndexEquiv lengths)
          (Equiv.refl (Coord trailingShape)))) |>.trans <|
        (Coord.appendEquiv leadingShape
          (lengths.sum :: trailingShape)).symm

/--
Concatenate a dependent family of tensors along one shared axis.

The family may be empty. In that case the result has a zero-length selected
axis and therefore no coordinates.
-/
def concatenateAxes {α : Type u} [Storage α]
    (leadingShape trailingShape : Shape)
    (lengths : List Nat)
    (segmentTensors :
      (segment : Fin lengths.length) →
        Rep α (leadingShape ++ lengths.get segment :: trailingShape)) :
    Rep α (leadingShape ++ lengths.sum :: trailingShape) :=
  Rep.ofFn fun outputCoordinate =>
    let segmentCoordinate :=
      (concatenateAxesCoordinateEquiv leadingShape trailingShape lengths).symm
        outputCoordinate
    segmentTensors segmentCoordinate.1 segmentCoordinate.2

/--
Split one tensor axis into a dependent family with the requested lengths.

Each segment coordinate is sent directly to the corresponding position in
the concatenated axis. This definition shares the same coordinate
equivalence as `concatenateAxes`.
-/
def splitAxis {α : Type u} [Storage α]
    (leadingShape trailingShape : Shape)
    (lengths : List Nat)
    (sourceTensor :
      Rep α (leadingShape ++ lengths.sum :: trailingShape))
    (segment : Fin lengths.length) :
    Rep α (leadingShape ++ lengths.get segment :: trailingShape) :=
  Rep.ofFn fun segmentTensorCoordinate =>
    sourceTensor <|
      concatenateAxesCoordinateEquiv leadingShape trailingShape lengths
        ⟨segment, segmentTensorCoordinate⟩

/-- Splitting a concatenated family recovers every original segment. -/
@[simp] theorem splitAxis_concatenateAxes {α : Type u}
    [Storage α]
    (leadingShape trailingShape : Shape) (lengths : List Nat)
    (segmentTensors :
      (segment : Fin lengths.length) →
        Rep α (leadingShape ++ lengths.get segment :: trailingShape)) :
    splitAxis leadingShape trailingShape lengths
        (concatenateAxes leadingShape trailingShape lengths segmentTensors) =
      segmentTensors := by
  funext segment
  ext segmentTensorCoordinate
  simpa only [splitAxis, concatenateAxes, get_ofFn] using
    congrArg
      (fun coordinate =>
        segmentTensors coordinate.1 coordinate.2)
      ((concatenateAxesCoordinateEquiv leadingShape trailingShape lengths)
        |>.symm_apply_apply ⟨segment, segmentTensorCoordinate⟩)

/-- Concatenating every segment of a complete split recovers the source tensor. -/
@[simp] theorem concatenateAxes_splitAxis {α : Type u}
    [Storage α]
    (leadingShape trailingShape : Shape) (lengths : List Nat)
    (sourceTensor : Rep α (leadingShape ++ lengths.sum :: trailingShape)) :
    concatenateAxes leadingShape trailingShape lengths
        (splitAxis leadingShape trailingShape lengths sourceTensor) =
      sourceTensor := by
  ext sourceCoordinate
  simpa only [concatenateAxes, splitAxis, get_ofFn] using
    congrArg sourceTensor
      ((concatenateAxesCoordinateEquiv leadingShape trailingShape lengths)
        |>.apply_symm_apply sourceCoordinate)

end Rep

end TorchLean.Tensor.Internal
