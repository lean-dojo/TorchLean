/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Lowering.Reduce

/-!
# Flat input views for reduction

These kernels consume a certified flat scalar reader instead of requiring an
already materialized input tensor. The ordinary reduction definitions and
the fused transform-to-reduction path therefore share the same fiber
enumeration, accumulator order, and correctness proofs.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Lowering

universe u v w

open Check
open Reduce.Impl

/--
Apply a total multiset aggregate to every reduction fiber read through a flat
input view.
-/
def reduceTensorFromFlat {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : Multiset α → β)
    (checked : CheckedTransform)
    (_hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (_hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    checked.OutputTensor β :=
  Rep.ofFlatFn fun outputIndex =>
    aggregate <|
      reductionValuesFromFlat checked read
        (Coord.unlinearize outputIndex)

/--
Fold every reduction fiber through one flat scalar reader and one final output
allocation.
-/
def reduceFoldTensorFromFlat {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β)
    (finish : β → Nat → γ)
    (checked : CheckedTransform)
    (_hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (_hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    checked.OutputTensor γ :=
  Rep.ofFlatFn fun outputIndex =>
    finish
      (reductionFoldlFromFlat step initial checked read <|
        Coord.unlinearize outputIndex)
      checked.reductionFiberSize

/--
Fold every nonempty reduction fiber from its first flat-view value.
-/
def reduceNonemptyFoldTensorFromFlat {α : Type u} [Storage α]
    (step : α → α → α)
    (checked : CheckedTransform)
    (_hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (_hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    checked.OutputTensor α :=
  Rep.ofFlatFn fun outputIndex =>
    reductionFoldlNonemptyFromFlat step checked hPositive read
      (Coord.unlinearize outputIndex)

/--
Pointwise-equal flat readers produce the same multiset reduction tensor.
-/
theorem reduceTensorFromFlat_eq {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : Multiset α → β)
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceTensorFromFlat aggregate checked hKind inputTensor read hRead =
      reduceTensor aggregate checked hKind inputTensor := by
  ext outputCoordinate
  simp only [reduceTensorFromFlat, reduceTensor, Rep.get_ofFlatFn,
    Coord.unlinearize_linearize, reductionValues]
  apply congrArg aggregate
  unfold reductionValuesFromFlat
  apply Multiset.map_congr rfl
  intro inputIndex _
  exact hRead _

/--
Pointwise-equal flat readers produce the same ordered accumulator reduction.
-/
theorem reduceFoldTensorFromFlat_eq
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β)
    (finish : β → Nat → γ)
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceFoldTensorFromFlat step initial finish checked hKind inputTensor
        read hRead =
      reduceFoldTensor step initial finish checked hKind inputTensor := by
  ext outputCoordinate
  simp only [reduceFoldTensorFromFlat, reduceFoldTensor,
    Rep.get_ofFlatFn, Coord.unlinearize_linearize, reductionFoldl]
  apply congrArg (fun total => finish total checked.reductionFiberSize)
  unfold reductionFoldlFromFlat
  apply congrArg
    (fun update =>
      Fin.foldl (Shape.size checked.reductionShape) update initial)
  funext total inputIndex
  congr 1
  exact hRead _

/--
Pointwise-equal flat readers produce the same ordered nonempty reduction.
-/
theorem reduceNonemptyFoldTensorFromFlat_eq
    {α : Type u} [Storage α]
    (step : α → α → α)
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceNonemptyFoldTensorFromFlat step checked hKind hPositive inputTensor
        read hRead =
      reduceNonemptyFoldTensor step checked hKind hPositive inputTensor := by
  ext outputCoordinate
  simp only [reduceNonemptyFoldTensorFromFlat, reduceNonemptyFoldTensor,
    Rep.get_ofFlatFn, Coord.unlinearize_linearize,
    reductionFoldlNonempty]
  unfold reductionFoldlNonemptyFromFlat
  simp only [hRead]

/--
The fused flat-reader kernel implements the independent multiset reduction
semantics.
-/
@[grind =] theorem reduceTensorFromFlat_correct
    {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : Multiset α → β)
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceTensorFromFlat aggregate checked hKind inputTensor read hRead =
      Semantics.denoteReduce aggregate checked hKind inputTensor := by
  rw [reduceTensorFromFlat_eq, reduceTensor_correct]

/--
The fused flat-reader accumulator preserves the independent row-major ordered
reduction semantics.
-/
@[grind =] theorem reduceFoldTensorFromFlat_ordered_correct
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β)
    (finish : β → Nat → γ)
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceFoldTensorFromFlat step initial finish checked hKind inputTensor
        read hRead =
      Semantics.denoteOrderedReduce step initial finish checked hKind
        inputTensor := by
  rw [reduceFoldTensorFromFlat_eq, reduceFoldTensor_ordered_correct]

/--
The fused flat-reader nonempty accumulator preserves the independent ordered
nonempty reduction semantics.
-/
@[grind =] theorem reduceNonemptyFoldTensorFromFlat_ordered_correct
    {α : Type u} [Storage α]
    (step : α → α → α)
    (checked : CheckedTransform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceNonemptyFoldTensorFromFlat step checked hKind hPositive inputTensor
        read hRead =
      Semantics.denoteOrderedReduceNonempty step checked hKind hPositive
        inputTensor := by
  rw [reduceNonemptyFoldTensorFromFlat_eq,
    reduceNonemptyFoldTensor_ordered_correct]

end TorchLean.Tensor.Internal.Lowering
