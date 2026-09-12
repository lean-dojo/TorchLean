/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Mathlib.Data.List.GetD
import Mathlib.Tactic.Bound.Init
public import NN.Tensor.Internal.Representation.Basic.Reindex
public import NN.Tensor.Internal.Representation.Coordinate
public import Mathlib.Data.List.OfFn -- shake: keep
public import NN.Tensor.Internal.Lowering.Rearrange -- shake: keep

/-!
# Compact row-major rearrangement indices

Natural-number encodings of checked coordinate maps and the transport lemmas
needed to connect them to shape-indexed tensor semantics.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

namespace Rearrangement.Impl

/--
Decode a row-major linear index into one natural-number coordinate per axis.

The coordinates are intentionally unbounded. Bounds come from the checked
plan when this function is connected to `Coord.unlinearize`; keeping them out
of the computation makes reducible rearrangement certificates small.
-/
@[simp] def rowMajorCoordinates : Shape → Nat → List Nat
  | [], _ => []
  | _ :: tailShape, linearIndex =>
      linearIndex / Shape.size tailShape ::
        rowMajorCoordinates tailShape (linearIndex % Shape.size tailShape)

/-- Encode natural-number axis coordinates as a row-major linear index. -/
@[simp] def rowMajorIndex : Shape → List Nat → Nat
  | [], _ => 0
  | _ :: _, [] => 0
  | _ :: tailShape, coordinate :: coordinates =>
      rowMajorIndex tailShape coordinates +
        Shape.size tailShape * coordinate

end Rearrangement.Impl

open Rearrangement.Impl

/--
Compute the input linear index selected by a rearrangement of elementary
axes.

This function is generic in the type used to identify axes. It is the compact
certificate language used by `einops`: checked plans whose dimensions reduce
become ordinary list, division, remainder, and multiplication calculations.
-/
def rearrangeLinearIndex {ι : Type*} [BEq ι] (length : ι → Nat)
    (inputAxes outputAxes : List ι) (outputLinearIndex : Nat) : Nat :=
  rowMajorIndex (inputAxes.map length) <|
    inputAxes.map fun axis =>
      (rowMajorCoordinates (outputAxes.map length)
        outputLinearIndex).getD (outputAxes.idxOf axis) 0

/-- Encoding coordinates through `AxisTuple.coordEquiv` agrees with `Coord.linearize`. -/
private theorem rowMajorIndex_coordEquiv {ι : Type*} (length : ι → Nat) :
    ∀ (axes : List ι)
      (coordinates : Coord (axes.map length)),
      rowMajorIndex (axes.map length)
          (List.ofFn fun axis =>
            ((AxisTuple.coordEquiv length axes coordinates) axis).val) =
        (Coord.linearize coordinates).val := by
  intro axes
  induction axes with
  | nil =>
      intro coordinates
      cases coordinates
      rfl
  | cons axis axes inductionHypothesis =>
      intro coordinates
      obtain ⟨headCoordinate, tailCoordinates⟩ := coordinates
      change
        rowMajorIndex (length axis :: axes.map length)
            (List.ofFn fun coordinateIndex =>
              ((AxisTuple.coordEquiv length (axis :: axes)
                (headCoordinate, tailCoordinates)) coordinateIndex).val) =
          (Coord.linearize (s := length axis :: axes.map length)
            (headCoordinate, tailCoordinates)).val
      rw [List.ofFn_succ]
      simp only [rowMajorIndex]
      change
        rowMajorIndex (axes.map length)
            (List.ofFn fun coordinateIndex =>
              ((AxisTuple.coordEquiv length axes tailCoordinates)
                coordinateIndex).val) +
            Shape.size (axes.map length) * headCoordinate.val =
          (Coord.linearize (s := length axis :: axes.map length)
            (headCoordinate, tailCoordinates)).val
      rw [inductionHypothesis]
      rw [Coord.linearize_cons_val]

/-- Encoding an axis tuple gives the linear index of its coordinate representation. -/
theorem rowMajorIndex_axisTuple {ι : Type*} (length : ι → Nat)
    (axes : List ι) (coordinates : AxisTuple length axes) :
    rowMajorIndex (axes.map length)
        (List.ofFn fun axis => (coordinates axis).val) =
      (Coord.linearize
        ((AxisTuple.coordEquiv length axes).symm coordinates)).val := by
  simpa using
    rowMajorIndex_coordEquiv length axes
      ((AxisTuple.coordEquiv length axes).symm coordinates)

/-- Row-major decoding recovers every component of the corresponding axis tuple. -/
theorem rowMajorCoordinates_axisTuple {ι : Type*} (length : ι → Nat) :
    ∀ (axes : List ι)
      (linearIndex : Fin (Shape.size (axes.map length))),
      rowMajorCoordinates (axes.map length) linearIndex.val =
        List.ofFn fun axis =>
          ((AxisTuple.coordEquiv length axes
            (Coord.unlinearize linearIndex)) axis).val := by
  intro axes
  induction axes with
  | nil =>
      intro linearIndex
      rfl
  | cons axis axes inductionHypothesis =>
      intro linearIndex
      change
        rowMajorCoordinates (length axis :: axes.map length)
            linearIndex.val =
          List.ofFn fun coordinateIndex =>
            ((AxisTuple.coordEquiv length (axis :: axes)
              (Coord.unlinearize linearIndex)) coordinateIndex).val
      rw [rowMajorCoordinates, List.ofFn_succ]
      congr 1
      exact inductionHypothesis
        ⟨linearIndex.val % Shape.size (axes.map length), by
          exact (finProdFinEquiv.symm linearIndex).2.isLt⟩

/-- List-based coordinate lookup computes the same tuple as `AxisTuple.select`. -/
theorem select_values {ι : Type*} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} :
    ∀ (source target : List ι)
      (h : ∀ axis, axis ∈ source → axis ∈ target)
      (coordinates : AxisTuple length target),
      source.map (fun axis =>
          (List.ofFn fun targetIndex =>
            (coordinates targetIndex).val).getD
              (target.idxOf axis) 0) =
        List.ofFn fun sourceIndex =>
          ((AxisTuple.select h coordinates) sourceIndex).val := by
  intro source
  induction source with
  | nil =>
      intro target h coordinates
      rfl
  | cons sourceAxis sourceAxes inductionHypothesis =>
      intro target h coordinates
      rw [List.map_cons, List.ofFn_succ]
      congr 1
      · let coordinateValues :=
          List.ofFn fun coordinateIndex =>
            (coordinates coordinateIndex).val
        let targetIndex : Fin coordinateValues.length :=
          ⟨target.idxOf sourceAxis,
            by
              simp only [coordinateValues, List.length_ofFn]
              exact List.idxOf_lt_length_iff.mpr
                (h sourceAxis (List.mem_cons_self))⟩
        change
          coordinateValues.getD targetIndex.val 0 =
            ((AxisTuple.select h coordinates) 0).val
        rw [List.getD_eq_get coordinateValues 0 targetIndex]
        rw [List.get_ofFn]
        rfl
      · exact inductionHypothesis target
          (fun axis hAxis => h axis (List.mem_cons_of_mem _ hAxis))
          coordinates

/-- Linearizing a selected tuple is exactly the executable rearrangement index. -/
theorem linearize_axisTupleSelect {ι : Type*} [BEq ι] [LawfulBEq ι]
    (length : ι → Nat) (source target : List ι)
    (h : ∀ axis, axis ∈ source → axis ∈ target)
    (linearIndex : Fin (Shape.size (target.map length))) :
    (Coord.linearize
      ((AxisTuple.coordEquiv length source).symm <|
        AxisTuple.select h <|
          AxisTuple.coordEquiv length target <|
            Coord.unlinearize linearIndex)).val =
      rearrangeLinearIndex length source target linearIndex.val := by
  rw [← rowMajorIndex_axisTuple]
  unfold rearrangeLinearIndex
  rw [rowMajorCoordinates_axisTuple]
  rw [select_values source target h]

/--
Swapping two row-major axes exchanges their coordinate contributions to the
linear index.
-/
theorem rearrangeLinearIndex_swap {ι : Type*} [BEq ι] [LawfulBEq ι]
    (length : ι → Nat) (first second : ι) (hDistinct : first ≠ second)
    (firstCoordinate secondCoordinate : Nat)
    (hFirst : firstCoordinate < length first) :
    rearrangeLinearIndex length [first, second] [second, first]
        (firstCoordinate + length first * secondCoordinate) =
      secondCoordinate + length second * firstCoordinate := by
  simp only [rearrangeLinearIndex, List.map_cons, List.map_nil,
    rowMajorCoordinates, Shape.size_cons, Shape.size_nil, mul_one,
    Nat.add_mul_mod_self_left, Nat.div_one, List.getD_eq_getElem?_getD, ne_eq,
    hDistinct.symm, not_false_eq_true, List.idxOf_cons_ne,
    List.idxOf_cons_self, Nat.succ_eq_add_one, zero_add,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some,
    rowMajorIndex, one_mul]
  rw [Nat.add_mul_div_left _ _ (Nat.zero_lt_of_lt hFirst),
    Nat.div_eq_of_lt hFirst, Nat.mod_eq_of_lt hFirst, Nat.zero_add]

/-- A reshape preserves the value of the row-major linear index. -/
theorem linearize_reshapeCoordEquiv_val
    {sourceShape targetShape : Shape}
    (hSize : Shape.size sourceShape = Shape.size targetShape)
    (targetCoordinate : Coord targetShape) :
    (Coord.linearize
      (Rep.reshapeCoordEquiv hSize targetCoordinate)).val =
      (Coord.linearize targetCoordinate).val := by
  change
    ((Coord.equivFin sourceShape)
      ((Coord.equivFin sourceShape).symm
        (finCongr hSize.symm
          ((Coord.equivFin targetShape) targetCoordinate)))).val =
      ((Coord.equivFin targetShape) targetCoordinate).val
  rw [Equiv.apply_symm_apply]
  rfl

namespace Rearrangement.Impl

/-- Reshaping an unlinearized index is unlinearization after finite-index transport. -/
theorem reshapeCoordEquiv_unlinearize
    {sourceShape targetShape : Shape}
    (hSize : Shape.size sourceShape = Shape.size targetShape)
    (targetIndex : Fin (Shape.size targetShape)) :
    Rep.reshapeCoordEquiv hSize (Coord.unlinearize targetIndex) =
      Coord.unlinearize (finCongr hSize.symm targetIndex) := by
  change
    (Coord.equivFin sourceShape).symm
        (finCongr hSize.symm
          ((Coord.equivFin targetShape)
            ((Coord.equivFin targetShape).symm targetIndex))) =
      (Coord.equivFin sourceShape).symm
        (finCongr hSize.symm targetIndex)
  rw [Equiv.apply_symm_apply]

end Rearrangement.Impl

/-- Transporting a coordinate across equal shapes preserves its linear-index value. -/
@[simp] theorem linearize_cast_val
    {sourceShape targetShape : Shape}
    (hShape : sourceShape = targetShape)
    (targetCoordinate : Coord targetShape) :
    (Coord.linearize
      (cast (congrArg Coord hShape.symm) targetCoordinate)).val =
      (Coord.linearize targetCoordinate).val := by
  cases hShape
  rfl

/-- Coordinate transport commutes with row-major linearization. -/
theorem linearize_cast
    {sourceShape targetShape : Shape}
    (hShape : sourceShape = targetShape)
    (sourceCoordinate : Coord sourceShape) :
    Coord.linearize
        (cast (congrArg Coord hShape) sourceCoordinate) =
      finCongr (congrArg Shape.size hShape)
        (Coord.linearize sourceCoordinate) := by
  cases hShape
  rfl

/-- `Equiv.cast` across equal shapes preserves the linear-index value. -/
theorem linearize_equivCast_val
    {sourceShape targetShape : Shape}
    (hShape : sourceShape = targetShape)
    (targetCoordinate : Coord targetShape) :
    (Coord.linearize
      (Equiv.cast (congrArg Coord hShape.symm)
        targetCoordinate)).val =
      (Coord.linearize targetCoordinate).val := by
  cases hShape
  rfl

/-- Row-major linearization commutes with shape transport through `Equiv.cast`. -/
theorem linearize_equivCast
    {sourceShape targetShape : Shape}
    (hShape : sourceShape = targetShape)
    (sourceCoordinate : Coord sourceShape) :
    Coord.linearize
        (Equiv.cast (congrArg Coord hShape) sourceCoordinate) =
      finCongr (congrArg Shape.size hShape)
        (Coord.linearize sourceCoordinate) := by
  cases hShape
  rfl

namespace Rearrangement.Impl

/-- Ordinary dependent transport and `Equiv.cast` are the same operation. -/
theorem cast_eq_equivCast {α β : Sort u}
    (h : α = β) (value : α) :
    cast h value = Equiv.cast h value := by
  cases h
  rfl

/-- Casting a tensor shape equals reindexing along the induced coordinate cast. -/
theorem cast_tensor_eq_reindex {α : Type u} [Storage α]
    {sourceShape targetShape : Shape}
    (hShape : sourceShape = targetShape)
    (tensor : Rep α sourceShape) :
    cast (congrArg (fun shape => Rep α shape) hShape) tensor =
      Rep.reindex
        (Equiv.cast (congrArg Coord hShape.symm)) tensor := by
  cases hShape
  ext coordinate
  simp

end Rearrangement.Impl

/--
Linearizing a composed coordinate map is unchanged when function composition
is exposed as nested application.

This small bridge lets proof-producing simplifiers normalize a stored
composition without reconstructing its dependent intermediate shape.
-/
theorem linearize_comp_apply_val
    {sourceShape intermediateShape outputShape : Shape}
    (outer : Coord intermediateShape → Coord sourceShape)
    (inner : Coord outputShape → Coord intermediateShape)
    (outputCoordinate : Coord outputShape) :
    (Coord.linearize ((outer ∘ inner) outputCoordinate)).val =
      (Coord.linearize (outer (inner outputCoordinate))).val := rfl

end TorchLean.Tensor.Internal
