/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Lowering.Reduce

/-!
# Flat input views for reduction

The kernels here read reduction inputs through a certified flat scalar reader,
so they need no materialized input tensor. The ordinary reduction definitions
and the fused transform-to-reduction path therefore share one fiber
enumeration, one accumulator order and the same correctness proofs.

The tensor argument supplies the reference semantics, and execution reads
through `read`. A pointwise equality proof connects the two without forcing or
caching the reference tensor.
-/

@[expose] public section

namespace TorchLean.Einops.Lowering

universe u v w

open Lowering.Reduction
open Shape (Coord)
open Tensor (Rep)

/--
Apply a total multiset aggregate to every reduction fiber read through a flat
input view.
-/
def reduceView {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : Multiset α → β)
    (checked : Transform)
    (_hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (_hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    checked.OutputTensor β :=
  Rep.ofFlatFn fun outputIndex =>
    aggregate <|
      valuesFromFlat checked read
        (Coord.unlinearize outputIndex)

/--
Fold every reduction fiber through one flat scalar reader and one final output
allocation.
-/
def reduceFoldView {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β)
    (finish : β → Nat → γ)
    (checked : Transform)
    (_hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (_hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    checked.OutputTensor γ :=
  Rep.ofFlatFn fun outputIndex =>
    finish
      (foldlFromFlat step initial checked read <|
        Coord.unlinearize outputIndex)
      checked.reductionFiberSize

/--
Fold every nonempty reduction fiber from its first flat-view value.
-/
def reduceNonemptyFoldView {α : Type u} [Storage α]
    (step : α → α → α)
    (checked : Transform)
    (_hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (_hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    checked.OutputTensor α :=
  Rep.ofFlatFn fun outputIndex =>
    foldlNonemptyFromFlat step checked hPositive read
      (Coord.unlinearize outputIndex)

/--
Pointwise-equal flat readers produce the same multiset reduction tensor.
-/
theorem reduceView_eq {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : Multiset α → β)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceView aggregate checked hKind inputTensor read hRead =
      reduce aggregate checked hKind inputTensor := by
  have hReadEq : read = inputTensor.getFlat := funext hRead
  subst read
  rfl

/--
Pointwise-equal flat readers produce the same ordered accumulator reduction.
-/
theorem reduceFoldView_eq
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β)
    (finish : β → Nat → γ)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceFoldView step initial finish checked hKind inputTensor
        read hRead =
      reduceFold step initial finish checked hKind inputTensor := by
  have hReadEq : read = inputTensor.getFlat := funext hRead
  subst read
  rfl

/--
Pointwise-equal flat readers produce the same ordered nonempty reduction.
-/
theorem reduceNonemptyFoldView_eq
    {α : Type u} [Storage α]
    (step : α → α → α)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceNonemptyFoldView step checked hKind hPositive inputTensor
        read hRead =
      reduceNonemptyFold step checked hKind hPositive inputTensor := by
  have hReadEq : read = inputTensor.getFlat := funext hRead
  subst read
  rfl

/--
The fused flat-reader kernel implements the independent multiset reduction
semantics.
-/
@[grind =, einops =, einops_simps] theorem reduceView_correct
    {α : Type u} {β : Type v}
    [Storage α] [Storage β]
    (aggregate : Multiset α → β)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceView aggregate checked hKind inputTensor read hRead =
      Semantics.denoteReduce aggregate checked hKind inputTensor := by
  rw [reduceView_eq, reduce_correct]

/--
The fused flat-reader accumulator preserves the independent row-major ordered
reduction semantics.
-/
@[grind =, einops =, einops_simps] theorem reduceFoldView_ordered_correct
    {α : Type u} {β : Type v} {γ : Type w}
    [Storage α] [Storage γ]
    (step : β → α → β) (initial : β)
    (finish : β → Nat → γ)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceFoldView step initial finish checked hKind inputTensor
        read hRead =
      Semantics.denoteOrderedReduce step initial finish checked hKind
        inputTensor := by
  rw [reduceFoldView_eq, reduceFold_ordered_correct]

/--
The fused flat-reader nonempty accumulator preserves the independent ordered
nonempty reduction semantics.
-/
@[grind =, einops =, einops_simps] theorem reduceNonemptyFoldView_correct
    {α : Type u} [Storage α]
    (step : α → α → α)
    (checked : Transform)
    (hKind : checked.value.normalized.kind = .reduce)
    (hPositive : 0 < checked.reductionFiberSize)
    (inputTensor : checked.InputTensor α)
    (read :
      Fin (Shape.size checked.value.normalized.input) → α)
    (hRead : ∀ inputIndex, read inputIndex = inputTensor.getFlat inputIndex) :
    reduceNonemptyFoldView step checked hKind hPositive inputTensor
        read hRead =
      Semantics.denoteOrderedReduceNonempty step checked hKind hPositive
        inputTensor := by
  rw [reduceNonemptyFoldView_eq,
    reduceNonemptyFold_correct]

end TorchLean.Einops.Lowering
