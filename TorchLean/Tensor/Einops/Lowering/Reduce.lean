/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Lowering.Rearrange -- shake: keep

/-!
# Fused native lowering for reduction

A checked reduction allocates no intermediate tensors. For each output coordinate, the executable
reference takes four steps.

1. It converts the retained output axes to elementary-axis coordinates.
2. It enumerates assignments of the removed axes alone, by their row-major flat index.
3. It reconstructs the corresponding original input coordinate.
4. It reads that scalar from the input tensor's storage.

There are no intermediate reshape, permutation, or reduction tensors. A generic reducer receives a
`Multiset` because its type promises a result independent of coordinate enumeration order, and that
multiset holds exactly one value per removed-axis assignment. The multiset path materializes those
values, and the accumulator path folds them as they are read. The lowering places no bound on
allocation inside a user-supplied reducer.

The correctness proof uses the abstract reduction-fiber equivalence in the theorem layer alone.
The executable definitions are computable and operate on native `Rep` storage.
-/

@[expose] public section

namespace TorchLean.Einops

universe u v w

namespace Lowering

open Shape (Coord)
open Tensor (Rep)

namespace Reduction

/-- Every input axis is either retained or assigned by the reduction fiber. -/
theorem inputAxes_subset_append
    (checked : Transform) :
    ∀ ⦃axis⦄,
      axis ∈ checked.value.normalized.inputAxes →
        axis ∈
          checked.value.normalized.outputAxes ++ checked.reducedAxes :=
  AxisTuple.mem_append_filter

/--
Reconstruct one original input coordinate from its retained output axes and
one assignment of all removed axes.
-/
def inputCoord
    (checked : Transform)
    (outputCoordinate : Coord checked.value.output)
    (reducedCoordinate :
      AxisTuple checked.value.axisLength checked.reducedAxes) :
    Coord checked.value.normalized.input :=
  checked.inputCoordEquiv.symm <|
    AxisTuple.merge checked.value.normalized.outputAxes
      (checked.outputCoordEquiv outputCoordinate) reducedCoordinate

/-- Reconstructed coordinates project back to the requested output coordinate. -/
private theorem outputOfInput_inputCoord
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (outputCoordinate : Coord checked.value.output)
    (reducedCoordinate :
      AxisTuple checked.value.axisLength checked.reducedAxes) :
    checked.outputOfInput
        (checked.valid.normalization.output_subset_of_reduce hKind)
        (inputCoord checked outputCoordinate
          reducedCoordinate) =
      outputCoordinate := by
  apply checked.outputCoordEquiv.injective
  simp only [Transform.outputOfInput,
    inputCoord, Equiv.apply_symm_apply]
  exact AxisTuple.select_merge_left
    checked.valid.normalization.output_nodup
    (checked.valid.normalization.output_subset_of_reduce hKind)
    (checked.outputCoordEquiv outputCoordinate)
    reducedCoordinate

/-- Selecting reduced axes from a reconstructed coordinate recovers their assignment. -/
private theorem select_inputCoord
    (checked : Transform)
    (outputCoordinate : Coord checked.value.output)
    (reducedCoordinate :
      AxisTuple checked.value.axisLength checked.reducedAxes) :
    AxisTuple.select checked.reduction_axis_mem_input
        (checked.inputCoordEquiv
          (inputCoord checked outputCoordinate
            reducedCoordinate)) =
      reducedCoordinate := by
  simp only [inputCoord, Equiv.apply_symm_apply]
  exact AxisTuple.select_merge_right
    checked.valid.normalization.input_nodup
    (checked.outputCoordEquiv outputCoordinate)
    reducedCoordinate

/--
Compute the row-major input index selected by one output coordinate and one
flat reduced-axis coordinate.
-/
def inputFlatIndex
    (checked : Transform)
    (outputCoordinate : Coord checked.value.output)
    (reducedFlatIndex : Fin (Shape.size checked.reductionShape)) :
    Fin (Shape.size checked.value.normalized.input) :=
  Coord.linearize <|
    inputCoord checked outputCoordinate
      (AxisTuple.coordEquiv checked.value.axisLength checked.reducedAxes <|
        Coord.unlinearize reducedFlatIndex)

/--
Collect exactly the input values belonging to one output coordinate through a
flat scalar reader.

The executable enumeration uses the row-major flat index of the removed-axis
shape. This avoids constructing the generic `Fintype` instance for a
dependent function tuple at every output coordinate.
-/
def valuesFromFlat {α : Type u}
    (checked : Transform)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (outputCoordinate : Coord checked.value.output) : Multiset α :=
  (Finset.univ :
      Finset (Fin (Shape.size checked.reductionShape))).val.map
    fun reducedFlatIndex =>
      read (inputFlatIndex checked outputCoordinate reducedFlatIndex)

/-- Collect one reduction fiber from an ordinary tensor's flat storage. -/
def values {α : Type u} [Storage α]
    (checked : Transform)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) : Multiset α :=
  valuesFromFlat checked inputTensor.getFlat outputCoordinate

/--
Fold one reduction fiber without materializing its value multiset.

The traversal is the same row-major enumeration used by `values`.
The enumeration itself keeps an accumulator rather than a collection of
scalars. An arbitrary `step` may still allocate or retain values in that
accumulator.
-/
def foldlFromFlat {α : Type u} {β : Type v}
    (step : β → α → β) (initial : β)
    (checked : Transform)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (outputCoordinate : Coord checked.value.output) : β :=
  Fin.foldl
    (Shape.size checked.reductionShape)
    (fun total reducedFlatIndex =>
      step total <|
        read (inputFlatIndex checked outputCoordinate reducedFlatIndex))
    initial

/-- Fold one reduction fiber by reading an ordinary tensor's flat storage. -/
def foldl {α : Type u} {β : Type v} [Storage α]
    (step : β → α → β) (initial : β)
    (checked : Transform)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) : β :=
  foldlFromFlat step initial checked inputTensor.getFlat
    outputCoordinate

/--
Fold a nonempty reduction fiber from its first row-major value.

Starting from an actual tensor entry avoids inventing sentinel values for
operations such as minimum and maximum. The remaining entries are visited in
the same order as `foldl`, so a step whose result depends on operand
order still has a fully specified result.
-/
def foldlNonemptyFromFlat {α : Type u}
    (step : α → α → α)
    (checked : Transform)
    (hPositive : 0 < checked.reductionFiberSize)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (outputCoordinate : Coord checked.value.output) : α :=
  have hReducedShape : 0 < Shape.size checked.reductionShape := by
    simpa using hPositive
  let value
      (reducedFlatIndex : Fin (Shape.size checked.reductionShape)) : α :=
    read (inputFlatIndex checked outputCoordinate reducedFlatIndex)
  Fin.foldl (Shape.size checked.reductionShape - 1)
    (fun total reducedFlatIndex =>
      step total <| value
        ⟨reducedFlatIndex.val + 1, by omega⟩)
    (value ⟨0, hReducedShape⟩)

/--
Fold a nonempty reduction fiber by reading an ordinary tensor's flat storage.
-/
def foldlNonempty {α : Type u} [Storage α]
    (step : α → α → α)
    (checked : Transform)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) : α :=
  foldlNonemptyFromFlat step checked hPositive inputTensor.getFlat
    outputCoordinate

/--
Direct fiber folding agrees with the order-independent fold of the semantic
value multiset.
-/
private theorem foldl_eq {α : Type u} {β : Type v}
    [Storage α]
    (step : β → α → β) [RightCommutative step] (initial : β)
    (checked : Transform)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    foldl step initial checked inputTensor outputCoordinate =
      Multiset.foldl step initial
        (values checked inputTensor outputCoordinate) := by
  let value (reducedFlatIndex : Fin (Shape.size checked.reductionShape)) : α :=
    inputTensor <|
      inputCoord checked outputCoordinate <|
        AxisTuple.coordEquiv checked.value.axisLength checked.reducedAxes <|
          Coord.unlinearize reducedFlatIndex
  unfold foldl values
  change
    Fin.foldl checked.reductionShape.size
        (fun total reducedFlatIndex => step total (value reducedFlatIndex))
        initial =
      Multiset.foldl step initial
        ((Finset.univ : Finset (Fin checked.reductionShape.size)).val.map value)
  rw [Fin.foldl_eq_foldl_finRange]
  rw [Finset.val_univ_fin]
  change
    List.foldl (fun total reducedFlatIndex => step total (value reducedFlatIndex))
        initial (List.finRange checked.reductionShape.size) =
      Multiset.foldl step initial
        (↑((List.finRange checked.reductionShape.size).map value) : Multiset α)
  rw [Multiset.coe_foldl]
  exact List.foldl_map.symm

/-- Direct reconstruction is the inverse of the semantic fiber equivalence. -/
private theorem inputCoord_eq_symm
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (outputCoordinate : Coord checked.value.output)
    (reducedCoordinate :
      AxisTuple checked.value.axisLength checked.reducedAxes) :
    inputCoord checked outputCoordinate reducedCoordinate =
      ((checked.reductionFiberEquiv hKind outputCoordinate).symm
        reducedCoordinate).1 := by
  let projection :=
    checked.outputOfInput <|
      checked.valid.normalization.output_subset_of_reduce hKind
  let candidate : Fiber projection outputCoordinate :=
    ⟨inputCoord checked outputCoordinate reducedCoordinate,
      outputOfInput_inputCoord checked hKind outputCoordinate
        reducedCoordinate⟩
  have hCandidate :
      checked.reductionFiberEquiv hKind outputCoordinate candidate =
        reducedCoordinate := by
    rw [checked.reductionFiberEquiv_apply hKind outputCoordinate]
    exact
      select_inputCoord checked outputCoordinate
        reducedCoordinate
  have hCandidateEq :
      candidate =
        (checked.reductionFiberEquiv hKind outputCoordinate).symm
          reducedCoordinate := by
    apply (checked.reductionFiberEquiv hKind outputCoordinate).injective
    rw [hCandidate]
    simpa [Transform.reducedAxes] using
      ((checked.reductionFiberEquiv hKind outputCoordinate).apply_symm_apply
        reducedCoordinate).symm
  exact congrArg Subtype.val hCandidateEq

/--
A fold over every index after zero is the left fold of the tail of `List.ofFn`.

The lemma isolates the finite-index bookkeeping of nonempty reductions and needs no algebraic
property of `step`.
-/
private theorem foldl_ofFn_tail {α : Type u} {n : Nat}
    (step : α → α → α) (value : Fin (n + 1) → α) :
    Fin.foldl n (fun total index => step total (value index.succ)) (value 0) =
      (List.ofFn value).tail.foldl step
        ((List.ofFn value).head (by simp)) := by
  have htail :
      (List.ofFn value).tail =
        List.ofFn (fun index => value index.succ) := by
    rw [List.ofFn_succ]
    rfl
  have hhead :
      (List.ofFn value).head (by simp) = value 0 := by
    rw [List.head_ofFn]
    congr
  rw [htail, hhead, Fin.foldl_eq_foldl_finRange, List.ofFn_eq_map,
    List.foldl_map]

/--
Positive finite ranges are successor ranges, so the tail-fold identity applies
without exposing a predecessor in callers.
-/
private theorem foldl_ofFn_tail_of_pos {α : Type u} {n : Nat}
    (hPositive : 0 < n) (step : α → α → α) (value : Fin n → α) :
    Fin.foldl (n - 1)
        (fun total index =>
          step total (value ⟨index.val + 1, by omega⟩))
        (value ⟨0, hPositive⟩) =
      (List.ofFn value).tail.foldl step
        ((List.ofFn value).head
          (by simpa using Nat.ne_of_gt hPositive)) := by
  obtain ⟨n, rfl⟩ :=
    Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hPositive)
  have hZero : (⟨0, hPositive⟩ : Fin n.succ) = 0 := Fin.ext rfl
  rw [hZero]
  simpa only [Nat.succ_sub_one, Fin.succ] using
    foldl_ofFn_tail step value

/-- Equal nonempty lists have equal first-value left folds. -/
private theorem foldl_tail_head_congr {α : Type u}
    (step : α → α → α) {leftValues rightValues : List α}
    (hValues : leftValues = rightValues)
    (hLeft : leftValues ≠ []) (hRight : rightValues ≠ []) :
    leftValues.tail.foldl step (leftValues.head hLeft) =
      rightValues.tail.foldl step (rightValues.head hRight) := by
  subst rightValues
  rfl

/--
The direct accumulator loop visits the independent ordered fiber list exactly
from left to right.
-/
private theorem foldl_eq_ordered
    {α : Type u} {β : Type v} [Storage α]
    (step : β → α → β) (initial : β)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    foldl step initial checked inputTensor outputCoordinate =
      (Semantics.orderedReductionValues checked hKind inputTensor
        outputCoordinate).foldl step initial := by
  let directValue
      (reducedFlatIndex : Fin (Shape.size checked.reductionShape)) : α :=
    inputTensor <|
      inputCoord checked outputCoordinate <|
        AxisTuple.coordEquiv checked.value.axisLength checked.reducedAxes <|
          Coord.unlinearize reducedFlatIndex
  let semanticValue
      (reducedFlatIndex : Fin (Shape.size checked.reductionShape)) : α :=
    inputTensor <|
      ((checked.reductionFiberEquiv hKind outputCoordinate).symm <|
        AxisTuple.coordEquiv checked.value.axisLength checked.reducedAxes <|
          Coord.unlinearize reducedFlatIndex).1
  have hValue : directValue = semanticValue := by
    funext reducedFlatIndex
    exact congrArg inputTensor <|
      inputCoord_eq_symm checked hKind
        outputCoordinate _
  change
    Fin.foldl checked.reductionShape.size
        (fun total reducedFlatIndex =>
          step total (directValue reducedFlatIndex))
        initial =
      (List.ofFn semanticValue).foldl step initial
  rw [Fin.foldl_eq_foldl_finRange, List.ofFn_eq_map, List.foldl_map,
    hValue]

/--
The nonempty direct loop starts from the first ordered fiber value and folds
the remaining values from left to right.
-/
private theorem foldlNonempty_eq_ordered
    {α : Type u} [Storage α]
    (step : α → α → α)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    foldlNonempty step checked hPositive inputTensor outputCoordinate =
      let values :=
        Semantics.orderedReductionValues checked hKind inputTensor
          outputCoordinate
      values.tail.foldl step <|
        values.head <|
          Semantics.orderedReductionValues_ne_nil checked hKind hPositive
            inputTensor outputCoordinate := by
  have hReducedShape : 0 < Shape.size checked.reductionShape := by
    simpa using hPositive
  let directValue
      (reducedFlatIndex : Fin (Shape.size checked.reductionShape)) : α :=
    inputTensor <|
      inputCoord checked outputCoordinate <|
        AxisTuple.coordEquiv checked.value.axisLength checked.reducedAxes <|
          Coord.unlinearize reducedFlatIndex
  have hValues :
      List.ofFn directValue =
        Semantics.orderedReductionValues checked hKind inputTensor
          outputCoordinate := by
    unfold Semantics.orderedReductionValues
    rw [List.ofFn_inj]
    funext reducedFlatIndex
    exact congrArg inputTensor <|
      inputCoord_eq_symm checked hKind
        outputCoordinate _
  unfold foldlNonempty
  change
    Fin.foldl (Shape.size checked.reductionShape - 1)
        (fun total reducedFlatIndex =>
          step total
            (directValue
              ⟨reducedFlatIndex.val + 1, by omega⟩))
        (directValue ⟨0, hReducedShape⟩) =
      _
  calc
    _ =
        (List.ofFn directValue).tail.foldl step
          ((List.ofFn directValue).head
            (by simpa using Nat.ne_of_gt hReducedShape)) :=
      foldl_ofFn_tail_of_pos hReducedShape step directValue
    _ = _ :=
      foldl_tail_head_congr step hValues _ _

/--
The fused removed-axis enumeration gives the same multiset as the independent
semantic input fiber.
-/
private theorem values_eq {α : Type u} [Storage α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    values checked inputTensor outputCoordinate =
      Fiber.values
        (checked.outputOfInput <|
          checked.valid.normalization.output_subset_of_reduce hKind)
        outputCoordinate inputTensor := by
  let reducedShape := checked.reductionShape
  let reducedCoordinateEquiv :
      Fin (Shape.size reducedShape) ≃
        AxisTuple checked.value.axisLength checked.reducedAxes :=
    (Coord.equivFin reducedShape).symm.trans <|
      AxisTuple.coordEquiv checked.value.axisLength checked.reducedAxes
  let fiberEquiv :
      Fiber
          (checked.outputOfInput <|
            checked.valid.normalization.output_subset_of_reduce hKind)
          outputCoordinate ≃
        AxisTuple checked.value.axisLength checked.reducedAxes :=
    checked.reductionFiberEquiv hKind outputCoordinate
  unfold values Fiber.values
  calc
    (Finset.univ :
        Finset (Fin (Shape.size reducedShape))).val.map
          (fun reducedFlatIndex =>
            inputTensor
              (inputCoord checked outputCoordinate
                (reducedCoordinateEquiv reducedFlatIndex))) =
        ((Finset.univ :
          Finset (Fin (Shape.size reducedShape))).val.map
            reducedCoordinateEquiv).map
          (fun reducedCoordinate =>
            inputTensor
              (inputCoord checked outputCoordinate
                reducedCoordinate)) := by
      simp only [Multiset.map_map, Function.comp_apply]
    _ =
        (Finset.univ :
          Finset
            (AxisTuple checked.value.axisLength
              checked.reducedAxes)).val.map
          (fun reducedCoordinate =>
            inputTensor
              (inputCoord checked outputCoordinate
                reducedCoordinate)) := by
      rw [Multiset.map_univ_val_equiv reducedCoordinateEquiv]
    _ =
        (Finset.univ :
          Finset
            (AxisTuple checked.value.axisLength checked.reducedAxes)).val.map
          (fun reducedCoordinate =>
            inputTensor (fiberEquiv.symm reducedCoordinate).1) := by
      apply Multiset.map_congr rfl
      intro reducedCoordinate _
      rw [inputCoord_eq_symm checked hKind
        outputCoordinate reducedCoordinate]
    _ =
        ((Finset.univ :
          Finset
            (AxisTuple checked.value.axisLength
              checked.reducedAxes)).val.map fiberEquiv.symm).map
          (fun inputCoordinate => inputTensor inputCoordinate.1) := by
      simp only [Multiset.map_map, Function.comp_apply]
    _ =
        (Finset.univ :
          Finset
            (Fiber
              (checked.outputOfInput <|
                checked.valid.normalization.output_subset_of_reduce
                  hKind)
              outputCoordinate)).val.map
          (fun inputCoordinate => inputTensor inputCoordinate.1) := by
      rw [Multiset.map_univ_val_equiv fiberEquiv.symm]

end Reduction

open Lowering.Reduction

/--
Execute a checked reduction with one final output allocation.

The aggregate receives one value for every assignment of the removed axes.
Its result type may differ from the input scalar type, and its empty-multiset
value determines empty-fiber behavior.
-/
def reduce {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : Multiset α → β)
    (checked : Transform)
    (_hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor β :=
  Rep.ofFlatFn fun flatIndex =>
    let outputCoordinate := Coord.unlinearize flatIndex
    aggregate (values checked inputTensor outputCoordinate)

/--
Fused native reduction equals the independent coordinate-fiber denotation for
every multiset aggregate, checked reduction plan, and input tensor.
-/
@[grind =, einops =, einops_simps] theorem reduce_correct {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : Multiset α → β)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) :
    reduce aggregate checked hKind inputTensor =
      Semantics.denoteReduce aggregate checked hKind inputTensor := by
  ext outputCoordinate
  simpa [reduce, Semantics.denoteReduce, Rep.reduce] using
    congrArg aggregate
      (values_eq checked hKind inputTensor outputCoordinate)

namespace Reduction

/-- The fused reduction enumerates exactly the certified number of fiber values. -/
theorem values_card {α : Type u} [Storage α]
    (checked : Transform)
    (inputTensor : checked.InputTensor α)
    (outputCoordinate : Coord checked.value.output) :
    (values checked inputTensor outputCoordinate).card =
      checked.reductionFiberSize := by
  simp only [values, valuesFromFlat, Multiset.card_map,
    Finset.card_val, Finset.card_univ, Fintype.card_fin,
    checked.reductionShape_size]

end Reduction

/--
Reduce every fiber through one accumulator loop and one scalar finalizer.

The finalizer also receives the certified fiber cardinality, which supports cardinality-dependent
reductions such as the arithmetic mean without a multiset built only to count entries.
-/
def reduceFold {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β)
    (finish : β → Nat → γ)
    (checked : Transform)
    (_hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor γ :=
  Rep.ofFlatFn fun flatIndex =>
    let outputCoordinate := Coord.unlinearize flatIndex
    finish
      (foldl step initial checked inputTensor outputCoordinate)
      checked.reductionFiberSize

/--
Reduce every nonempty fiber from its first row-major value.

The executor folds the entries in row-major order and builds no multiset, so it suits ordered
binary operations whose result may depend on traversal order.
-/
def reduceNonemptyFold {α : Type u} [Storage α]
    (step : α → α → α)
    (checked : Transform)
    (_hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor α :=
  Rep.ofFlatFn fun flatIndex =>
    foldlNonempty step checked hPositive inputTensor
      (Coord.unlinearize flatIndex)

/--
The direct accumulator executor implements the independent row-major ordered
reduction denotation for every scalar operation.

In particular, this theorem requires no associativity or commutativity and is
therefore suitable for IEEE floating-point arithmetic.
-/
@[grind =, einops =, einops_simps] theorem reduceFold_ordered_correct
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β) (finish : β → Nat → γ)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) :
    reduceFold step initial finish checked hKind inputTensor =
      Semantics.denoteOrderedReduce step initial finish checked hKind
        inputTensor := by
  ext outputCoordinate
  simp only [reduceFold, Rep.get_ofFlatFn,
    Coord.unlinearize_linearize, Semantics.denoteOrderedReduce,
    Rep.get_ofFn]
  rw [foldl_eq_ordered step initial checked hKind]

/--
The direct first-value executor implements the independent ordered nonempty
reduction denotation for every binary scalar operation.
-/
@[grind =, einops =, einops_simps] theorem reduceNonemptyFold_correct
    {α : Type u} [Storage α]
    (step : α → α → α)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α) :
    reduceNonemptyFold step checked hKind hPositive inputTensor =
      Semantics.denoteOrderedReduceNonempty step checked hKind hPositive
        inputTensor := by
  ext outputCoordinate
  simp only [reduceNonemptyFold, Rep.get_ofFlatFn,
    Coord.unlinearize_linearize, Semantics.denoteOrderedReduceNonempty,
    Rep.get_ofFn]
  exact foldlNonempty_eq_ordered step checked hKind
    hPositive inputTensor outputCoordinate

/--
The direct accumulator executor implements any multiset aggregate that
factors through an order-independent left fold and the multiset cardinality.
-/
theorem reduceFold_correct
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) [RightCommutative step] (initial : β)
    (finish : β → Nat → γ) (aggregate : Multiset α → γ)
    (hAggregate :
      ∀ values,
        aggregate values =
          finish (Multiset.foldl step initial values) values.card)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) :
    reduceFold step initial finish checked hKind inputTensor =
      Semantics.denoteReduce aggregate checked hKind inputTensor := by
  ext outputCoordinate
  simp only [reduceFold, Rep.get_ofFlatFn,
    Coord.unlinearize_linearize, Semantics.denoteReduce, Rep.reduce,
    Rep.get_ofFn]
  rw [foldl_eq, ← values_card checked inputTensor,
    ← hAggregate]
  exact congrArg aggregate <|
    values_eq checked hKind inputTensor outputCoordinate

/--
The accumulator executor also implements nonempty aggregates that factor
through an order-independent fold and the certified fiber cardinality.
-/
theorem reduceFold_nonempty_correct
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) [RightCommutative step] (initial : β)
    (finish : β → Nat → γ)
    (aggregate : (values : Multiset α) → values ≠ 0 → γ)
    (hAggregate :
      ∀ values hValues,
        aggregate values hValues =
          finish (Multiset.foldl step initial values) values.card)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α) :
    reduceFold step initial finish checked hKind inputTensor =
      Semantics.denoteReduceNonempty aggregate checked hKind hPositive
        inputTensor := by
  ext outputCoordinate
  simp only [reduceFold, Rep.get_ofFlatFn,
    Coord.unlinearize_linearize, Semantics.denoteReduceNonempty,
    Rep.reduceNonempty, Rep.get_ofFn]
  rw [foldl_eq, ← values_card checked inputTensor]
  change
    finish
        (Multiset.foldl step initial
          (Reduction.values checked inputTensor outputCoordinate))
        (Reduction.values checked inputTensor outputCoordinate).card =
      aggregate
        (Fiber.values
          (checked.outputOfInput <|
            checked.valid.normalization.output_subset_of_reduce hKind)
          outputCoordinate inputTensor) _
  have hValues :
      Reduction.values checked inputTensor outputCoordinate ≠ 0 :=
    Multiset.card_pos.mp <| by
      rw [values_card checked inputTensor outputCoordinate]
      exact hPositive
  rw [← hAggregate _ hValues]
  exact Rep.nonemptyAggregate_congr aggregate
    (values_eq checked hKind inputTensor outputCoordinate) _ _

/-- Additive reduction executes as one direct scalar fold per output entry. -/
@[grind =, einops =, einops_simps] theorem reduceFold_sum {α : Type u}
    [Storage α] [AddCommMonoid α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) :
    reduceFold (· + ·) 0 (fun total _ => total)
        checked hKind inputTensor =
      Semantics.denoteReduce Multiset.sum checked hKind inputTensor := by
  exact reduceFold_correct
    (fun total value => total + value) 0
    (fun total _ => total) Multiset.sum
    (fun values => Multiset.sum_eq_foldl values)
    checked hKind inputTensor

/-- Multiplicative reduction executes as one direct scalar fold per output entry. -/
@[grind =, einops =, einops_simps] theorem reduceFold_prod {α : Type u}
    [Storage α] [CommMonoid α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α) :
    reduceFold (· * ·) 1 (fun total _ => total)
        checked hKind inputTensor =
      Semantics.denoteReduce Multiset.prod checked hKind inputTensor := by
  exact reduceFold_correct
    (fun total value => total * value) 1
    (fun total _ => total) Multiset.prod
    (fun values => Multiset.prod_eq_foldl values)
    checked hKind inputTensor

/-- Boolean disjunction reduction executes without constructing a multiset. -/
@[grind =, einops =, einops_simps] theorem reduceFold_any
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor Bool) :
    reduceFold Bool.or false (fun total _ => total)
        checked hKind inputTensor =
      Semantics.denoteReduce Reduction.any checked hKind inputTensor := by
  exact reduceFold_correct Bool.or false
    (fun total _ => total) Reduction.any
    (fun values => Multiset.fold_eq_foldl Bool.or false values)
    checked hKind inputTensor

/-- Boolean conjunction reduction executes without constructing a multiset. -/
@[grind =, einops =, einops_simps] theorem reduceFold_all
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor Bool) :
    reduceFold Bool.and true (fun total _ => total)
        checked hKind inputTensor =
      Semantics.denoteReduce Reduction.all checked hKind inputTensor := by
  exact reduceFold_correct Bool.and true
    (fun total _ => total) Reduction.all
    (fun values => Multiset.fold_eq_foldl Bool.and true values)
    checked hKind inputTensor

/--
Over nonempty fibers, exact mean reduction accumulates the sum in one fold and divides once by
the certified fiber cardinality.
-/
@[grind =, einops =, einops_simps] theorem reduceFold_mean
    {α : Type u} [Storage α] [DivisionRing α] [CharZero α]
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α) :
    reduceFold (· + ·) 0
        (fun total cardinality => total / (cardinality : α))
        checked hKind inputTensor =
      Semantics.denoteReduceNonempty Reduction.mean checked hKind hPositive
        inputTensor := by
  exact reduceFold_nonempty_correct
    (fun total value => total + value) 0
    (fun total cardinality => total / (cardinality : α))
    Reduction.mean
    (fun values _ => by
      simp only [Reduction.mean, Multiset.sum_eq_foldl])
    checked hKind hPositive inputTensor

/--
Execute a reduction whose aggregate is defined only for nonempty multisets,
using the same single-allocation fused kernel.
-/
def reduceNonempty {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : (values : Multiset α) → values ≠ 0 → β)
    (checked : Transform)
    (_hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α) : checked.OutputTensor β :=
  Rep.ofFlatFn fun flatIndex =>
    let outputCoordinate := Coord.unlinearize flatIndex
    let values := Reduction.values checked inputTensor outputCoordinate
    aggregate values <| Multiset.card_pos.mp <| by
      change 0 < (Reduction.values checked inputTensor outputCoordinate).card
      rw [values_card checked inputTensor outputCoordinate]
      exact hPositive

/--
Fused nonempty reduction equals its independent coordinate-fiber denotation.
-/
@[grind =, einops =, einops_simps] theorem reduceNonempty_correct {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : (values : Multiset α) → values ≠ 0 → β)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α) :
    reduceNonempty aggregate checked hKind hPositive inputTensor =
      Semantics.denoteReduceNonempty aggregate checked hKind hPositive
        inputTensor := by
  ext outputCoordinate
  simp only [reduceNonempty, Rep.get_ofFlatFn,
    Coord.unlinearize_linearize, Semantics.denoteReduceNonempty,
    Rep.reduceNonempty, Rep.get_ofFn]
  change
    aggregate (Reduction.values checked inputTensor outputCoordinate) _ =
      aggregate
        (Fiber.values
          (checked.outputOfInput <|
            checked.valid.normalization.output_subset_of_reduce hKind)
          outputCoordinate inputTensor) _
  exact Rep.nonemptyAggregate_congr aggregate
    (values_eq checked hKind inputTensor outputCoordinate) _ _

end Lowering

end TorchLean.Einops
