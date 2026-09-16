/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Semantics.Einsum
public import Batteries.Data.Fin.Fold -- shake: keep
public import Batteries.Data.Fin.Lemmas -- shake: keep

/-!
# Fused native lowering for einsum

A checked einsum allocates only its final output tensor. For each output
entry, the executable kernel:

1. enumerates flat indices for only the contracted logical axes;
2. evaluates a precompiled row-major stride plan for each operand;
3. reads every operand directly from its native array, with diagonal selection
   and singleton broadcasting already encoded by that plan;
4. multiplies the operand values in source order; and
5. sums those products into the output entry.

No broadcast, pointwise-product, permutation, or contraction tensor is
materialized. The independent product tensor lives in the semantics module;
the lowering module contains only the executable flat-index plan and its
correctness bridge.

The executable loops need only addition, multiplication, and the scalar
literals zero and one. Their row-major order is part of the program, which
makes the same kernel available to IEEE floating-point types without
installing false algebraic instances. Stronger correctness theorems recover
the order-independent `Rep.push` denotation whenever the scalar operations
form additive and multiplicative monoids.

## References

The broadcast, ordered product, axis permutation, and reduction stages follow
the contraction behavior of einops v0.8.2 at commit
`8e911db71f2e693a0c434b041180388c685ed06f`.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

namespace Lowering

open Check
open scoped BigOperators

/-- Contracted axes inherit duplicate-freedom from the checked global axis list. -/
theorem contractedAxes_nodup (checked : CheckedEinsum) :
    checked.contractedAxes.Nodup :=
  checked.global_axes_nodup.filter _

/-- Contracted axes are precisely axes absent from the output. -/
private theorem contractedAxes_disjoint_output
    (checked : CheckedEinsum) :
    ∀ ⦃axis⦄,
      axis ∈ checked.contractedAxes →
        axis ∉ checked.outputAxes := by
  intro axis hAxis
  have hAbsent := (List.mem_filter.mp hAxis).2
  simpa using hAbsent

/-- Every global logical axis occurs in the output list or the contracted list. -/
private theorem globalAxes_subset_output_append_contracted
    (checked : CheckedEinsum) :
    ∀ ⦃axis⦄,
      axis ∈ checked.globalAxes →
        axis ∈ checked.outputAxes ++ checked.contractedAxes := by
  intro axis hGlobal
  by_cases hOutput : axis ∈ checked.outputAxes
  · exact List.mem_append_left _ hOutput
  · apply List.mem_append_right
    simp [Check.CheckedEinsum.contractedAxes, hGlobal, hOutput]

/-!
## Verified flat indexing

The fused kernel compiles each operand's named axes to a short list of
ordinary triples:

* whether the coordinate comes from the output or contraction coordinate;
* the position of that logical axis in its source coordinate; and
* the operand's physical row-major stride.

Singleton physical dimensions contribute no term. Evaluation then performs
only coordinate projection, multiplication, addition, and one direct array
read. The helper theorems below prove that this arithmetic index is exactly
the linearization of `inputCoordinateOfGlobal`; all shape and broadcasting
proofs are erased from generated code.
-/

/-- Read one component of a shape coordinate by its zero-based axis position. -/
def coordinateAt : (shape : Shape) → Coord shape → Nat → Nat
  | [], _, _ => 0
  | _ :: _, coordinate, 0 => coordinate.1.val
  | _ :: tailShape, coordinate, axisPosition + 1 =>
      coordinateAt tailShape coordinate.2 axisPosition

/-- Direct coordinate projection returns the matching named-axis coordinate. -/
private theorem coordinateAt_axisTuple {ι : Type*}
    (length : ι → Nat) :
    ∀ (axes : List ι)
      (coordinate : Coord (axes.map length))
      (axisPosition : Fin axes.length),
      coordinateAt (axes.map length) coordinate axisPosition.val =
        ((AxisTuple.coordEquiv length axes coordinate)
          axisPosition).val := by
  intro axes
  induction axes with
  | nil =>
      intro coordinate axisPosition
      exact Fin.elim0 axisPosition
  | cons axis axes inductionHypothesis =>
      intro coordinate axisPosition
      refine Fin.cases ?_ (fun tailPosition => ?_) axisPosition
      · rfl
      · exact inductionHypothesis coordinate.2 tailPosition

/--
Compile physical operand axes to source-position and input-stride triples.

The plan is built once per operand. Omitting singleton dimensions makes
broadcasting free inside the scalar loop.
-/
def inputFlatIndexPlan {ι : Type*} [BEq ι] [LawfulBEq ι]
    (outputAxes contractedAxes : List ι) :
    (inputAxes : List ι) → Shape → List (Bool × Nat × Nat)
  | [], _ => []
  | _ :: _, [] => []
  | axis :: inputAxes, physicalLength :: inputShape =>
      let remainingPlan :=
        inputFlatIndexPlan outputAxes contractedAxes inputAxes inputShape
      if physicalLength = 1 then
        remainingPlan
      else if axis ∈ outputAxes then
        (true, outputAxes.idxOf axis, Shape.size inputShape) :: remainingPlan
      else
        (false, contractedAxes.idxOf axis, Shape.size inputShape) :: remainingPlan

/--
Read one logical axis coordinate from the output index or the contracted index
according to whether that axis survives the contraction.
-/
private def plannedAxisCoordinate {ι : Type*} [BEq ι] [LawfulBEq ι]
    (outputAxes contractedAxes : List ι)
    (length : ι → Nat)
    (outputCoordinate : Coord (outputAxes.map length))
    (contractionCoordinate : Coord (contractedAxes.map length))
    (axis : ι) : Nat :=
  if axis ∈ outputAxes then
    coordinateAt (outputAxes.map length)
      outputCoordinate (outputAxes.idxOf axis)
  else
    coordinateAt (contractedAxes.map length)
      contractionCoordinate (contractedAxes.idxOf axis)

/-- Evaluate one compiled operand plan to its physical row-major array index. -/
def evaluateInputFlatIndexPlan
    (outputShape contractedShape : Shape)
    (outputCoordinate : Coord outputShape)
    (contractionCoordinate : Coord contractedShape) :
    List (Bool × Nat × Nat) → Nat
  | [] => 0
  | (fromOutput, sourcePosition, inputStride) :: remainingPlan =>
      let sourceCoordinate :=
        if fromOutput then
          coordinateAt outputShape outputCoordinate sourcePosition
        else
          coordinateAt contractedShape contractionCoordinate sourcePosition
      evaluateInputFlatIndexPlan outputShape contractedShape
          outputCoordinate contractionCoordinate remainingPlan +
        inputStride * sourceCoordinate

/--
Proof-only model of physical row-major linearization after singleton
broadcasting.
-/
private def broadcastFlatIndex :
    (physicalShape logicalShape : Shape) → Coord logicalShape → Nat
  | [], _, _ => 0
  | _ :: _, [], _ => 0
  | physicalLength :: physicalShape, _ :: logicalShape, coordinates =>
      broadcastFlatIndex physicalShape logicalShape coordinates.2 +
        if physicalLength = 1 then 0
        else Shape.size physicalShape * coordinates.1.val

/-- Evaluating a compiled index plan agrees with its recursive broadcast model. -/
private theorem evaluateInputFlatIndexPlan_eq_broadcastFlatIndex
    {ι : Type*} [BEq ι] [LawfulBEq ι]
    (outputAxes contractedAxes : List ι)
    (length : ι → Nat)
    (outputCoordinate : Coord (outputAxes.map length))
    (contractionCoordinate : Coord (contractedAxes.map length)) :
    ∀ (inputAxes : List ι) (inputShape : Shape)
      (coordinates : Coord (inputAxes.map length)),
      (∀ position,
        plannedAxisCoordinate outputAxes contractedAxes length
            outputCoordinate contractionCoordinate
            (inputAxes.get position) =
          ((AxisTuple.coordEquiv length inputAxes coordinates)
            position).val) →
      evaluateInputFlatIndexPlan
          (outputAxes.map length) (contractedAxes.map length)
          outputCoordinate contractionCoordinate
          (inputFlatIndexPlan outputAxes contractedAxes inputAxes inputShape) =
        broadcastFlatIndex inputShape (inputAxes.map length) coordinates := by
  intro inputAxes
  induction inputAxes with
  | nil =>
      intro inputShape coordinates hCoordinates
      cases inputShape <;> cases coordinates <;> rfl
  | cons axis inputAxes inductionHypothesis =>
      intro inputShape coordinates hCoordinates
      rcases coordinates with ⟨headCoordinate, tailCoordinate⟩
      cases inputShape with
      | nil =>
          simp [inputFlatIndexPlan, evaluateInputFlatIndexPlan,
            broadcastFlatIndex]
      | cons physicalLength inputShape =>
          have hHead :
              plannedAxisCoordinate outputAxes contractedAxes length
                  outputCoordinate contractionCoordinate axis =
                headCoordinate.val :=
            hCoordinates (0 : Fin (axis :: inputAxes).length)
          have hTail :
              ∀ position,
                plannedAxisCoordinate outputAxes contractedAxes length
                    outputCoordinate contractionCoordinate
                    (inputAxes.get position) =
                  ((AxisTuple.coordEquiv length inputAxes tailCoordinate)
                    position).val := by
            intro position
            exact hCoordinates position.succ
          have hRemaining :=
            inductionHypothesis inputShape tailCoordinate hTail
          change _ =
            broadcastFlatIndex (physicalLength :: inputShape)
              (length axis :: inputAxes.map length)
              (headCoordinate, tailCoordinate)
          by_cases hSingleton : physicalLength = 1
          · simpa [inputFlatIndexPlan,
              broadcastFlatIndex, hSingleton] using hRemaining
          · by_cases hOutput : axis ∈ outputAxes
            · simpa [inputFlatIndexPlan,
                evaluateInputFlatIndexPlan, broadcastFlatIndex, hSingleton,
                hOutput, plannedAxisCoordinate] using
                congrArg₂ Nat.add hRemaining
                  (congrArg (Shape.size inputShape * ·) hHead)
            · simpa [inputFlatIndexPlan,
                evaluateInputFlatIndexPlan, broadcastFlatIndex, hSingleton,
                hOutput, plannedAxisCoordinate] using
                congrArg₂ Nat.add hRemaining
                  (congrArg (Shape.size inputShape * ·) hHead)

/-- The recursive broadcast model equals linearization after coordinate broadcasting. -/
private theorem broadcastFlatIndex_eq_linearize_broadcast
    (physicalShape logicalShape : Shape)
    (hShape :
      List.Forall₂
        (fun physicalLength logicalLength =>
          physicalLength = logicalLength ∨ physicalLength = 1)
        physicalShape logicalShape)
    (coordinates : Coord logicalShape) :
    broadcastFlatIndex physicalShape logicalShape coordinates =
      (Coord.linearize
        (Coord.broadcast physicalShape logicalShape hShape
          coordinates)).val := by
  induction hShape with
  | nil =>
      cases coordinates
      rfl
  | @cons physicalLength logicalLength physicalShape logicalShape
      hLength hTail inductionHypothesis =>
      rcases coordinates with ⟨headCoordinate, tailCoordinate⟩
      have hInduction := inductionHypothesis tailCoordinate
      by_cases hSingleton : physicalLength = 1
      · subst physicalLength
        by_cases hLogicalOne : logicalLength = 1
        · subst logicalLength
          simp [broadcastFlatIndex, Coord.broadcast, hInduction]
        · have hNotEqual : ¬(1 = logicalLength) := by
            exact fun h => hLogicalOne h.symm
          simp [broadcastFlatIndex, Coord.broadcast, hNotEqual, hInduction]
      · have hEqual : physicalLength = logicalLength :=
          hLength.resolve_right hSingleton
        subst logicalLength
        simp [broadcastFlatIndex, Coord.broadcast, hSingleton, hInduction]

/-- An axis found in the left list reads its value from the left appended tuple. -/
private theorem append_idxOf_left_val {ι : Type*} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {left right : List ι}
    (hLeft : left.Nodup)
    (leftCoordinate : AxisTuple length left)
    (rightCoordinate : AxisTuple length right)
    {axis : ι} (hAxis : axis ∈ left) :
    ((AxisTuple.append left leftCoordinate rightCoordinate)
      ⟨(left ++ right).idxOf axis,
        List.idxOf_lt_length_iff.mpr
          (List.mem_append_left right hAxis)⟩).val =
      (leftCoordinate
        ⟨left.idxOf axis, List.idxOf_lt_length_iff.mpr hAxis⟩).val := by
  let leftIndex : Fin left.length :=
    ⟨left.idxOf axis, List.idxOf_lt_length_iff.mpr hAxis⟩
  have hGet : left.get leftIndex = axis :=
    List.idxOf_get leftIndex.isLt
  let selectedIndex : Fin (left ++ right).length :=
    ⟨(left ++ right).idxOf (left.get leftIndex),
      List.idxOf_lt_length_iff.mpr
        (List.mem_append_left right (List.get_mem left leftIndex))⟩
  let canonicalIndex : Fin (left ++ right).length :=
    ⟨(left ++ right).idxOf axis,
      List.idxOf_lt_length_iff.mpr
        (List.mem_append_left right hAxis)⟩
  have hSelectedIndex : selectedIndex = canonicalIndex := by
    apply Fin.ext
    simp only [selectedIndex, canonicalIndex, hGet]
  have hRecovered :=
    congrArg (fun coordinates => (coordinates leftIndex).val)
      (AxisTuple.select_append_left hLeft leftCoordinate rightCoordinate)
  change
    ((AxisTuple.append left leftCoordinate rightCoordinate)
        selectedIndex).val =
      (leftCoordinate leftIndex).val at hRecovered
  change
    ((AxisTuple.append left leftCoordinate rightCoordinate)
        canonicalIndex).val =
      (leftCoordinate leftIndex).val
  rw [← hSelectedIndex]
  exact hRecovered

/-- A right-only axis reads its value from the right appended tuple. -/
private theorem append_idxOf_right_val {ι : Type*} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat} {left right : List ι}
    (hRight : right.Nodup)
    (hDisjoint : ∀ axis, axis ∈ left → axis ∉ right)
    (leftCoordinate : AxisTuple length left)
    (rightCoordinate : AxisTuple length right)
    {axis : ι} (hAxis : axis ∈ right) :
    ((AxisTuple.append left leftCoordinate rightCoordinate)
      ⟨(left ++ right).idxOf axis,
        List.idxOf_lt_length_iff.mpr
          (List.mem_append_right left hAxis)⟩).val =
      (rightCoordinate
        ⟨right.idxOf axis, List.idxOf_lt_length_iff.mpr hAxis⟩).val := by
  let rightIndex : Fin right.length :=
    ⟨right.idxOf axis, List.idxOf_lt_length_iff.mpr hAxis⟩
  have hGet : right.get rightIndex = axis :=
    List.idxOf_get rightIndex.isLt
  let selectedIndex : Fin (left ++ right).length :=
    ⟨(left ++ right).idxOf (right.get rightIndex),
      List.idxOf_lt_length_iff.mpr
        (List.mem_append_right left (List.get_mem right rightIndex))⟩
  let canonicalIndex : Fin (left ++ right).length :=
    ⟨(left ++ right).idxOf axis,
      List.idxOf_lt_length_iff.mpr
        (List.mem_append_right left hAxis)⟩
  have hSelectedIndex : selectedIndex = canonicalIndex := by
    apply Fin.ext
    simp only [selectedIndex, canonicalIndex, hGet]
  have hRecovered :=
    congrArg (fun coordinates => (coordinates rightIndex).val)
      (AxisTuple.select_append_right hRight hDisjoint
        leftCoordinate rightCoordinate)
  change
    ((AxisTuple.append left leftCoordinate rightCoordinate)
        selectedIndex).val =
      (rightCoordinate rightIndex).val at hRecovered
  change
    ((AxisTuple.append left leftCoordinate rightCoordinate)
        canonicalIndex).val =
      (rightCoordinate rightIndex).val
  rw [← hSelectedIndex]
  exact hRecovered

/-- Planned decoding agrees with selecting from output and contraction coordinates. -/
private theorem plannedAxisCoordinate_eq_select_append
    {ι : Type*} [BEq ι] [LawfulBEq ι]
    {length : ι → Nat}
    (outputAxes contractedAxes inputAxes : List ι)
    (hOutput : outputAxes.Nodup)
    (hContracted : contractedAxes.Nodup)
    (hDisjoint : ∀ axis, axis ∈ outputAxes → axis ∉ contractedAxes)
    (hInput :
      ∀ axis, axis ∈ inputAxes → axis ∈ outputAxes ++ contractedAxes)
    (outputCoordinate : Coord (outputAxes.map length))
    (contractionCoordinate : Coord (contractedAxes.map length))
    (position : Fin inputAxes.length) :
    plannedAxisCoordinate outputAxes contractedAxes length
        outputCoordinate contractionCoordinate (inputAxes.get position) =
      ((AxisTuple.select hInput <|
        AxisTuple.append outputAxes
          (AxisTuple.coordEquiv length outputAxes outputCoordinate)
          (AxisTuple.coordEquiv length contractedAxes
            contractionCoordinate)) position).val := by
  let axis := inputAxes.get position
  have hAxisInput : axis ∈ inputAxes :=
    List.get_mem inputAxes position
  have hAxisCanonical : axis ∈ outputAxes ++ contractedAxes :=
    hInput axis hAxisInput
  let outputNamedCoordinate :=
    AxisTuple.coordEquiv length outputAxes outputCoordinate
  let contractionNamedCoordinate :=
    AxisTuple.coordEquiv length contractedAxes contractionCoordinate
  let canonicalCoordinate :=
    AxisTuple.append outputAxes outputNamedCoordinate
      contractionNamedCoordinate
  let selectedCanonicalIndex : Fin (outputAxes ++ contractedAxes).length :=
    ⟨(outputAxes ++ contractedAxes).idxOf axis,
      List.idxOf_lt_length_iff.mpr hAxisCanonical⟩
  have hSelectedCoordinate :
      ((AxisTuple.select hInput canonicalCoordinate) position).val =
        (canonicalCoordinate selectedCanonicalIndex).val := by
    rfl
  by_cases hAxisOutput : axis ∈ outputAxes
  · let sourcePosition : Fin outputAxes.length :=
      ⟨outputAxes.idxOf axis,
        List.idxOf_lt_length_iff.mpr hAxisOutput⟩
    have hDecoded :=
      coordinateAt_axisTuple length outputAxes outputCoordinate
        sourcePosition
    have hAppended :=
      append_idxOf_left_val hOutput outputNamedCoordinate
        contractionNamedCoordinate hAxisOutput
    change
      plannedAxisCoordinate outputAxes contractedAxes length
          outputCoordinate contractionCoordinate axis =
        ((AxisTuple.select hInput canonicalCoordinate) position).val
    rw [hSelectedCoordinate]
    simp only [plannedAxisCoordinate, ite_eq_left hAxisOutput]
    exact hDecoded.trans hAppended.symm
  · have hAxisContracted : axis ∈ contractedAxes := by
      rcases List.mem_append.mp hAxisCanonical with hImpossible | hContracted
      · exact False.elim (hAxisOutput hImpossible)
      · exact hContracted
    let sourcePosition : Fin contractedAxes.length :=
      ⟨contractedAxes.idxOf axis,
        List.idxOf_lt_length_iff.mpr hAxisContracted⟩
    have hDecoded :=
      coordinateAt_axisTuple length contractedAxes contractionCoordinate
        sourcePosition
    have hAppended :=
      append_idxOf_right_val hContracted hDisjoint outputNamedCoordinate
        contractionNamedCoordinate hAxisContracted
    change
      plannedAxisCoordinate outputAxes contractedAxes length
          outputCoordinate contractionCoordinate axis =
        ((AxisTuple.select hInput canonicalCoordinate) position).val
    rw [hSelectedCoordinate]
    simp only [plannedAxisCoordinate, ite_eq_right hAxisOutput]
    exact hDecoded.trans hAppended.symm

/-- The compiled operand plan computes the certified physical tensor index. -/
private theorem einsumInputFlatIndexPlan_correct
    (checked : CheckedEinsum)
    (operand : Fin checked.inputShapes.length)
    (outputCoordinate : Coord checked.output)
    (contractionCoordinate :
      Coord (checked.contractedAxes.map checked.axisLength)) :
    evaluateInputFlatIndexPlan
        checked.output
        (checked.contractedAxes.map checked.axisLength)
        outputCoordinate contractionCoordinate
        (inputFlatIndexPlan checked.outputAxes
          checked.contractedAxes
          (checked.operandAxes operand)
          (checked.inputShapes.get operand)) =
      (Coord.linearize
        (checked.inputCoordinateOfGlobal operand <|
          checked.reconstructedGlobalCoordinate
            outputCoordinate
            (AxisTuple.coordEquiv checked.axisLength
              checked.contractedAxes contractionCoordinate))).val := by
  let canonicalCoordinate :=
    AxisTuple.append checked.outputAxes
      (checked.outputTensorCoordinateEquiv outputCoordinate)
      (AxisTuple.coordEquiv checked.axisLength
        checked.contractedAxes contractionCoordinate)
  have hOperandCanonical :
      ∀ axis, axis ∈ checked.operandAxes operand →
        axis ∈ checked.outputAxes ++ checked.contractedAxes := by
    intro axis hAxis
    exact globalAxes_subset_output_append_contracted checked <|
      checked.input_axis_mem_global operand hAxis
  let selectedOperandCoordinate :=
    AxisTuple.select hOperandCanonical canonicalCoordinate
  let logicalOperandCoordinate :=
    (AxisTuple.coordEquiv checked.axisLength
      (checked.operandAxes operand)).symm selectedOperandCoordinate
  have hPlannedCoordinates :
      ∀ position,
        plannedAxisCoordinate checked.outputAxes
            checked.contractedAxes
            checked.axisLength outputCoordinate contractionCoordinate
            ((checked.operandAxes operand).get position) =
          ((AxisTuple.coordEquiv checked.axisLength
              (checked.operandAxes operand) logicalOperandCoordinate)
            position).val := by
    intro position
    rw [show
      AxisTuple.coordEquiv checked.axisLength (checked.operandAxes operand)
          logicalOperandCoordinate =
        selectedOperandCoordinate by
          exact Equiv.apply_symm_apply _ _]
    exact plannedAxisCoordinate_eq_select_append
      checked.outputAxes checked.contractedAxes
      (checked.operandAxes operand)
      checked.output_axes_nodup
      (contractedAxes_nodup checked)
      (fun axis hOutput hContracted =>
        contractedAxes_disjoint_output checked hContracted hOutput)
      hOperandCanonical outputCoordinate contractionCoordinate position
  calc
    evaluateInputFlatIndexPlan
          checked.output
          (checked.contractedAxes.map checked.axisLength)
          outputCoordinate contractionCoordinate
          (inputFlatIndexPlan checked.outputAxes
            checked.contractedAxes
            (checked.operandAxes operand)
            (checked.inputShapes.get operand)) =
        broadcastFlatIndex
          (checked.inputShapes.get operand)
          ((checked.operandAxes operand).map checked.axisLength)
          logicalOperandCoordinate :=
      evaluateInputFlatIndexPlan_eq_broadcastFlatIndex
        checked.outputAxes checked.contractedAxes
        checked.axisLength outputCoordinate contractionCoordinate
        (checked.operandAxes operand)
        (checked.inputShapes.get operand)
        logicalOperandCoordinate hPlannedCoordinates
    _ =
        (Coord.linearize
          (Coord.broadcast
            (checked.inputShapes.get operand)
            ((checked.operandAxes operand).map checked.axisLength)
            (checked.operand_shape_broadcastable operand)
            logicalOperandCoordinate)).val :=
      broadcastFlatIndex_eq_linearize_broadcast
        (checked.inputShapes.get operand)
        ((checked.operandAxes operand).map checked.axisLength)
        (checked.operand_shape_broadcastable operand)
        logicalOperandCoordinate
    _ =
        (Coord.linearize
          (checked.inputCoordinateOfGlobal operand <|
            checked.reconstructedGlobalCoordinate
              outputCoordinate
              (AxisTuple.coordEquiv checked.axisLength
                checked.contractedAxes contractionCoordinate))).val := by
      congr 2
      simp only [Check.CheckedEinsum.inputCoordinateOfGlobal,
        Check.CheckedEinsum.reconstructedGlobalCoordinate,
        Equiv.apply_symm_apply]
      rw [AxisTuple.select_comp]

/--
The certified physical row-major index read from one einsum operand.

This is the indexing contract shared by the generic kernel and
pattern-specialized elaboration. Its value is ordinary natural-number
arithmetic; the bound proof is erased from executable code.
-/
@[inline] def einsumInputFlatIndex
    (checked : CheckedEinsum)
    (operand : Fin checked.inputShapes.length)
    (outputCoordinate : Coord checked.output)
    (contractionCoordinate :
      Coord (checked.contractedAxes.map checked.axisLength)) :
    Fin (Shape.size (checked.inputShapes.get operand)) :=
  let inputIndexValue :=
    evaluateInputFlatIndexPlan
      checked.output
      (checked.contractedAxes.map checked.axisLength)
      outputCoordinate contractionCoordinate
      (inputFlatIndexPlan checked.outputAxes
        checked.contractedAxes
        (checked.operandAxes operand)
        (checked.inputShapes.get operand))
  ⟨inputIndexValue, by
    rw [show
      inputIndexValue =
        (Coord.linearize
          (checked.inputCoordinateOfGlobal operand <|
            checked.reconstructedGlobalCoordinate
              outputCoordinate
              (AxisTuple.coordEquiv checked.axisLength
                checked.contractedAxes contractionCoordinate))).val by
      exact einsumInputFlatIndexPlan_correct
        checked operand outputCoordinate contractionCoordinate]
    exact (Coord.linearize _).isLt⟩

/-!
The executable kernel uses nested `Fin.foldl` loops so contraction and operand
traversal allocate no temporary finite sets or lists. The row-major traversal
itself is `Semantics.coordinateSum`: there is one such loop in the library, and
the lemmas below are what connect it to the big-operator semantics.
-/

/-- Multiplicative finite folding preserves the source order of the list fold. -/
private theorem fin_foldl_mul_eq_foldl {R : Type u} [Mul R] [OfNat R 1]
    {n : Nat} (values : Fin n → R) :
    Fin.foldl n (fun product index => product * values index) 1 =
      (List.ofFn values).foldl (· * ·) 1 := by
  rw [Fin.foldl_eq_foldl_finRange, ← List.foldl_map,
    ← List.ofFn_eq_map]

/-- Pointwise-equal scalar kernels have equal multidimensional coordinate sums. -/
theorem coordinateSum_congr {R : Type u} [Add R] [OfNat R 0]
    (shape : Shape) (left right : Coord shape → R) (initial : R)
    (h : ∀ coordinate, left coordinate = right coordinate) :
    Semantics.coordinateSum shape left initial =
      Semantics.coordinateSum shape right initial := by
  apply congrArg (fun values => Semantics.coordinateSum shape values initial)
  funext coordinate
  exact h coordinate

/--
Every scalar kernel has the same coordinate sum over an empty coordinate
space. This is the certificate used when an einsum contracts a zero-length
axis, where the scalar kernels are intentionally never evaluated.
-/
theorem coordinateSum_congr_of_isEmpty {R : Type u} [Add R] [OfNat R 0]
    (shape : Shape) [IsEmpty (Coord shape)]
    (left right : Coord shape → R) (initial : R) :
    Semantics.coordinateSum shape left initial =
      Semantics.coordinateSum shape right initial :=
  coordinateSum_congr shape left right initial isEmptyElim

/--
Split an ordered multiplicative fold into three contiguous operand ranges.

The factors retain their source order; only associativity and the identity
laws of a monoid are used.
-/
theorem foldl_mul_split_three {R : Type u} [Monoid R]
    (left middle right : List R) :
    ((left.foldl (· * ·) 1 * middle.foldl (· * ·) 1) *
        right.foldl (· * ·) 1) =
      ((left ++ middle) ++ right).foldl (· * ·) 1 := by
  simp only [← List.prod_eq_foldl, List.prod_append]

/--
Move contraction-invariant factors from both ends of a coordinate sum.

Multiplication order is unchanged, so this law applies to noncommutative
semirings as well as ordinary numeric scalar types.
-/
theorem mul_coordinateSum_mul {R : Type u} [Semiring R]
    (shape : Shape) (left right : R) (values : Coord shape → R) :
    left * Semantics.coordinateSum shape values 0 * right =
      Semantics.coordinateSum shape
        (fun coordinate => left * values coordinate * right) 0 := by
  rw [Semantics.coordinateSum_eq_add_sum, zero_add,
    Semantics.coordinateSum_eq_add_sum, zero_add]
  rw [Finset.mul_sum, Finset.sum_mul]

/--
The generic scalar product used by `einsumTensor`.

The public elaborator may replace this function by code generated from the
checked literal. Its correctness obligation is pointwise equality with this
reference implementation, so every generated loop reuses the same semantic
bridge.
-/
def einsumInputProduct {R : Type u} [Storage R] [Mul R] [OfNat R 1]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R) :
    Coord checked.output →
      Coord (checked.contractedAxes.map checked.axisLength) → R :=
  let contractedAxes := checked.contractedAxes
  let contractedShape := contractedAxes.map checked.axisLength
  let inputFlatIndexPlans :=
    Array.ofFn fun operand =>
      inputFlatIndexPlan checked.outputAxes contractedAxes
        (checked.operandAxes operand) (checked.inputShapes.get operand)
  fun outputCoordinate contractionCoordinate =>
    Fin.foldl checked.inputShapes.length
      (fun product operand =>
        let operandPlan :=
          inputFlatIndexPlans[operand.val]'(by
            simp [inputFlatIndexPlans])
        let inputIndexValue :=
          evaluateInputFlatIndexPlan checked.output contractedShape
            outputCoordinate contractionCoordinate operandPlan
        let certifiedInputIndex :=
          einsumInputFlatIndex checked operand outputCoordinate
            contractionCoordinate
        let inputIndex :
            Fin (Shape.size (checked.inputShapes.get operand)) :=
          ⟨inputIndexValue, by
            simpa [certifiedInputIndex, einsumInputFlatIndex,
              inputIndexValue, operandPlan, inputFlatIndexPlans,
              contractedShape, contractedAxes] using
              certifiedInputIndex.isLt⟩
        product * (inputTensors operand).getFlat inputIndex)
      1

/--
The generic scalar product is the ordered finite fold of the certified input
reads.

Generated kernels use this theorem as their proof boundary: optimized index
arithmetic is certified operand by operand, then the resulting scalar product
is compared with this fold without unfolding `einsumInputProduct` in a client
module.
-/
theorem einsumInputProduct_eq_foldl {R : Type u}
    [Storage R] [Mul R] [OfNat R 1]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R)
    (outputCoordinate : Coord checked.output)
    (contractionCoordinate :
      Coord (checked.contractedAxes.map checked.axisLength)) :
    einsumInputProduct checked inputTensors outputCoordinate
        contractionCoordinate =
      Fin.foldl checked.inputShapes.length
        (fun product operand =>
          product *
            (inputTensors operand).getFlat
              (einsumInputFlatIndex checked operand outputCoordinate
                contractionCoordinate))
        1 := by
  simp only [einsumInputProduct, Array.getElem_ofFn]
  apply congrArg (fun step => Fin.foldl checked.inputShapes.length step 1)
  funext product operand
  congr 2

/--
Evaluate every contracted coordinate for one flat output position.

This reference executor is completely general. Literal syntax compiles it to
the same nested loops after reducing the checked pattern, which removes
coordinate-pair construction from the native scalar loop.
-/
def einsumOutput {R : Type u}
    [Storage R] [Add R] [Mul R] [OfNat R 0] [OfNat R 1]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R) :
    Fin (Shape.size checked.output) → R :=
  fun outputIndex =>
    Semantics.coordinateSum
      (checked.contractedAxes.map checked.axisLength)
      (einsumInputProduct checked inputTensors
        (Coord.unlinearize outputIndex))

/--
Execute a checked einsum using the general verified output function.

For each output entry, the executor enumerates only contracted-axis
coordinates, reads the corresponding values directly from every operand,
preserves source order during multiplication, and accumulates the products.
-/
@[inline] def einsumTensor {R : Type u}
    [Storage R] [Add R] [Mul R] [OfNat R 0] [OfNat R 1]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R) :
    checked.OutputTensor R :=
  Rep.ofFlatFn (einsumOutput checked inputTensors)

/--
Accept a compiler-generated einsum result through an explicit proof boundary.

The elaborator supplies a specialized scalar function and an output tensor,
which may have been assembled sequentially or from certified parallel chunks.
Erased certificates identify both with the general executor. Keeping this
boundary explicit avoids optional-argument wrappers inside generated proof
terms.
-/
@[inline] def einsumTensorKernel {R : Type u}
    [Storage R] [Add R] [Mul R] [OfNat R 0] [OfNat R 1]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R)
    (outputValues : Fin (Shape.size checked.output) → R)
    (_hOutputValues :
      ∀ outputIndex,
        outputValues outputIndex =
          einsumOutput checked inputTensors outputIndex)
    (outputTensor : checked.OutputTensor R)
    (_hOutputTensor :
      outputTensor = Rep.ofFlatFn outputValues) :
    checked.OutputTensor R :=
  outputTensor

/--
The compiler-generated einsum kernel equals the independent contraction
denotation for every checked pattern.

No algebraic laws are required: the theorem compares the exact source-ordered
operand fold and row-major contraction fold executed by both sides.
-/
@[grind =] theorem einsumTensorKernel_correct {R : Type u}
    [Storage R] [Add R] [Mul R] [OfNat R 0] [OfNat R 1]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R)
    (outputValues : Fin (Shape.size checked.output) → R)
    (hOutputValues :
      ∀ outputIndex,
        outputValues outputIndex =
          einsumOutput checked inputTensors outputIndex)
    (outputTensor : checked.OutputTensor R)
    (hOutputTensor : outputTensor = Rep.ofFlatFn outputValues) :
    einsumTensorKernel checked inputTensors outputValues hOutputValues
        outputTensor hOutputTensor =
      Semantics.denoteEinsum checked inputTensors := by
  rw [einsumTensorKernel, hOutputTensor]
  ext outputCoordinate
  simp only [Rep.get_ofFlatFn]
  rw [hOutputValues]
  simp only [einsumOutput, Coord.unlinearize_linearize,
    Semantics.denoteEinsum, Rep.get_ofFn]
  apply congrArg
    (fun values =>
      Semantics.coordinateSum
        (checked.contractedAxes.map checked.axisLength) values 0)
  funext contractionCoordinate
  rw [einsumInputProduct_eq_foldl, fin_foldl_mul_eq_foldl,
    Semantics.einsumProductTensor_get_ordered]
  apply congrArg
    (fun values : List R => values.foldl (fun left right => left * right) 1)
  rw [List.ofFn_inj]
  funext operand
  change
    (inputTensors operand).getFlat _ =
      (inputTensors operand).getFlat _
  congr 1
  apply Fin.ext
  exact einsumInputFlatIndexPlan_correct checked operand
    outputCoordinate contractionCoordinate

/--
The general einsum executor equals the independent contraction denotation.
-/
@[grind =] theorem einsumTensor_correct {R : Type u}
    [Storage R] [Add R] [Mul R] [OfNat R 0] [OfNat R 1]
    (checked : CheckedEinsum) (inputTensors : checked.InputTensors R) :
    einsumTensor checked inputTensors =
      Semantics.denoteEinsum checked inputTensors := by
  simpa only [einsumTensor, einsumTensorKernel] using
    einsumTensorKernel_correct checked inputTensors
      (einsumOutput checked inputTensors) (fun _ => rfl)
      (Rep.ofFlatFn (einsumOutput checked inputTensors)) rfl

end Lowering

end TorchLean.Tensor.Internal
