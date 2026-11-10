/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Mathlib.Algebra.BigOperators.Fin
public import TorchLean.Tensor.Rep.Reindex -- shake: keep
public import TorchLean.Tensor.Rep.Traversal -- shake: keep

/-!
# Tensor Segments Along an Arbitrary Axis

Packing concatenates tensors along one axis, and unpacking cuts that axis back into the
corresponding family of half-open slices. Neither operation depends on pattern syntax or on
scalar algebra.

`Shape.Coord.appendEquiv` splits a coordinate of `leadingShape ++ trailingShape` into a coordinate
of each part, and the tensor operations use it to isolate the selected axis.
`Tensor.Rep.concatenateAxes` concatenates a finite dependent family, and `Tensor.Rep.splitAxis`
partitions one axis according to a list of lengths. The two are proved mutually inverse, including
empty families and zero-length segments, so they preserve every scalar value and need no operation
on the scalar type.
-/

@[expose] public section

namespace TorchLean

universe u

/--
Identify an index in a list of consecutive segments with its index in the
concatenated interval.

The first component selects a segment and the second selects a position
inside that segment. Mathlib's dependent finite-sum equivalence places the
segments in list order, including segments of length zero.
-/
def Shape.Coord.segmentEquiv (lengths : List Nat) :
    ((segment : Fin lengths.length) × Fin (lengths.get segment)) ≃
      Fin lengths.sum :=
  finSigmaFinEquiv.trans <|
    finCongr <| by
      simp

namespace Shape.Coord

/--
Split a coordinate of an appended shape into its coordinates in `leadingShape` and
`trailingShape`.

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

/--
Appending two coordinates places the right coordinate in the low-order
row-major digits and the left coordinate in the high-order digits.
-/
theorem linearize_appendEquiv_symm_val :
    ∀ (left right : Shape) (leftCoordinate : Coord left) (rightCoordinate : Coord right),
      (linearize ((appendEquiv left right).symm (leftCoordinate, rightCoordinate))).val =
        (linearize rightCoordinate).val + Shape.size right * (linearize leftCoordinate).val
  | [], _, leftCoordinate, _ => by
      cases leftCoordinate
      rfl
  | _ :: left, right, (headCoordinate, leftCoordinate), rightCoordinate => by
      change (linearize (s := _ :: (left ++ right))
        (headCoordinate, (appendEquiv left right).symm (leftCoordinate, rightCoordinate))).val = _
      rw [linearize_cons_val, linearize_cons_val,
        linearize_appendEquiv_symm_val left right leftCoordinate rightCoordinate,
        Shape.size_append, Nat.mul_add, ← Nat.mul_assoc,
        Nat.mul_comm (Shape.size right) (Shape.size left), Nat.add_assoc]

end Shape.Coord

namespace Tensor.Rep

open Shape (Coord)

/--
Identify a component coordinate in a segmented tensor family with its
coordinate after concatenation along the selected axis.

The leading and trailing coordinates pass through unchanged, and `Coord.segmentEquiv` places the
local selected-axis coordinate inside the component's segment of the concatenated axis.
-/
def _root_.TorchLean.Shape.Coord.concatEquiv (leadingShape trailingShape : Shape)
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
        (Equiv.prodCongr (Coord.segmentEquiv lengths)
          (Equiv.refl (Coord trailingShape)))) |>.trans <|
        (Coord.appendEquiv leadingShape
          (lengths.sum :: trailingShape)).symm

/--
Concatenate a dependent family of tensors along one shared axis.

The family may be empty, in which case the selected axis has length zero and the result has no
coordinates.
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
      (Coord.concatEquiv leadingShape trailingShape lengths).symm
        outputCoordinate
    segmentTensors segmentCoordinate.1 segmentCoordinate.2

/--
Split one tensor axis into a dependent family with the requested lengths.

A coordinate of segment `segment` reads `sourceTensor` at its image under
`Shape.Coord.concatEquiv`, the equivalence that `concatenateAxes` applies in reverse.
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
      Coord.concatEquiv leadingShape trailingShape lengths
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
      ((Coord.concatEquiv leadingShape trailingShape lengths)
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
      ((Coord.concatEquiv leadingShape trailingShape lengths)
        |>.apply_symm_apply sourceCoordinate)

end Tensor.Rep

end TorchLean
