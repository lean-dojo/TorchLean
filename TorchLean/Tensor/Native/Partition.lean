/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Native.Fold -- shake: keep

/-!
# Certified contiguous output partitions

Task-parallel einsum output generation rests on one array theorem,
`OrderedFlatPartition.foldl_append_eq_ofFn`. For a partition into any
number of adjacent chunks, appending the chunk arrays in source order
reconstructs the original finite function.
-/

@[expose] public section

namespace TorchLean.Native

universe u

/--
One named physical output buffer together with its complete finite-function
semantics.

Generated sequential kernels name this record as one auxiliary definition.
Executable code projects `produce` and certificates project `toArray_produce`,
so neither side needs to unfold the complete native loop.
-/
structure CertifiedFlatBuffer
    {α : Type u} [storage : Storage α]
    (length : Nat) (values : Fin length → α) where
  /-- Produce the complete physical output buffer. -/
  produce : Unit → storage.Buffer
  /-- Observing the produced buffer gives the complete finite function. -/
  toArray_produce :
    storage.toArray (produce ()) = Array.ofFn values

/--
Select a contiguous subrange of a finite function.

The range proof is erased, so at runtime `Native.flatRange` adds `start` to the local
index and reads the corresponding value.
-/
@[inline] def flatRange
    {α : Type u} (total start length : Nat)
    (hRange : start + length ≤ total)
    (values : Fin total → α) : Fin length → α :=
  fun index => values ⟨start + index, by omega⟩

/--
One named parallel chunk together with the range of the finite function that
it produces.

Generated code seals each chunk as one auxiliary definition, with the same
split between `produce` and `toArray_produce` as `Native.CertifiedFlatBuffer`.
-/
structure CertifiedFlatPart
    {α : Type u} [storage : Storage α]
    (total start length : Nat)
    (hRange : start + length ≤ total)
    (values : Fin total → α) where
  /-- Produce the physical buffer for this contiguous range. -/
  produce : Unit → storage.Buffer
  /-- Observing the produced buffer gives exactly the selected range. -/
  toArray_produce :
    storage.toArray (produce ()) =
      Array.ofFn (flatRange total start length hRange values)

/--
An ordered list of chunk producers covering the flat indices from `start` up
to `total`.

Each producer returns one contiguous physical-buffer range, and the recursive
index leaves no gap or overlap between adjacent ranges. The final constructor
requires the accumulated start to equal `total`, so the partition is complete.
-/
inductive OrderedFlatPartition
    {α : Type u} [storage : Storage α]
    (total : Nat) (values : Fin total → α) :
    Nat → List (Unit → storage.Buffer) → Prop
  /-- The empty suffix completes a partition once its start reaches `total`. -/
  | done : OrderedFlatPartition total values total []
  /--
  Prepend one certified contiguous producer to a partition of the remaining
  interval.
  -/
  | next {start length : Nat}
      {parts : List (Unit → storage.Buffer)}
      (part : Unit → storage.Buffer)
      (hRange : start + length ≤ total)
      (hPart :
        storage.toArray (part ()) =
          Array.ofFn (flatRange total start length hRange values))
      (tail :
        OrderedFlatPartition total values (start + length) parts) :
      OrderedFlatPartition total values start (part :: parts)

/-- Append a complete physical source buffer to a physical output buffer. -/
@[inline] def appendBuffer
    {α : Type u} [storage : Storage α]
    (source output : storage.Buffer) : storage.Buffer :=
  storage.appendSlice source 0 (storage.size source) output

/-- Observing a complete physical-buffer append gives ordinary array append. -/
theorem toArray_appendBuffer
    {α : Type u} [storage : Storage α]
    (source output : storage.Buffer) :
    storage.toArray (appendBuffer source output) =
      storage.toArray output ++ storage.toArray source := by
  rw [appendBuffer, storage.toArray_appendSlice]
  congr 1
  rw [← storage.toArray_size]
  exact Array.extract_size

/-- Appending to a fresh empty physical buffer returns the source buffer. -/
theorem appendBuffer_empty
    {α : Type u} [storage : Storage α]
    (source : storage.Buffer) :
    appendBuffer source (storage.emptyWithCapacity 0) = source := by
  apply storage.toArray_injective
  rw [toArray_appendBuffer, storage.toArray_emptyWithCapacity]
  exact Array.empty_append

/--
Appending two adjacent finite-function ranges produces their combined range.
-/
private theorem array_ofFn_flatRange_append
    {α : Type u} (total start left right : Nat)
    (hRange : start + (left + right) ≤ total)
    (values : Fin total → α) :
    Array.ofFn
          (flatRange total start left (by omega) values) ++
        Array.ofFn
          (flatRange total (start + left) right (by omega) values) =
      Array.ofFn
        (flatRange total start (left + right) hRange values) := by
  rw [Array.ofFn_add]
  congr 1
  apply congrArg Array.ofFn
  funext index
  simp only [flatRange, Fin.val_natAdd]
  apply congrArg values
  apply Fin.ext
  simp [Nat.add_assoc]

/--
If `initial` already holds the entries of `values` before `start`, appending
every producer of an ordered partition reconstructs the complete
finite-function array.
-/
theorem OrderedFlatPartition.foldl_append_eq_ofFn
    {α : Type u} [storage : Storage α]
    {total start : Nat} {values : Fin total → α}
    {parts : List (Unit → storage.Buffer)}
    (partition : OrderedFlatPartition total values start parts)
    (initial : storage.Buffer)
    (hStart : start ≤ total)
    (hInitial :
      storage.toArray initial =
        Array.ofFn
          (flatRange total 0 start (by omega) values)) :
    storage.toArray
        (parts.foldl
          (fun output part => appendBuffer (part ()) output)
          initial) =
      Array.ofFn values := by
  induction partition generalizing initial with
  | done =>
      simp only [List.foldl_nil]
      rw [hInitial]
      apply congrArg Array.ofFn
      funext index
      simp only [flatRange, Nat.zero_add]
  | @next start length parts part hRange hPart tail inductionHypothesis =>
      simp only [List.foldl_cons]
      apply inductionHypothesis
        (appendBuffer (part ()) initial) hRange
      rw [toArray_appendBuffer, hInitial, hPart]
      convert
        array_ofFn_flatRange_append total 0 start length (by omega) values using 1
      congr 1
      apply congrArg Array.ofFn
      funext index
      simp only [flatRange, Nat.zero_add]

end TorchLean.Native
