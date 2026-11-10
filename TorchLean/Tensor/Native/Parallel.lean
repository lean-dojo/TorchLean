/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Native.Partition

/-!
# Certified parallel output assembly

Large concrete einsums may compute any ordered partition of the outer output
axis concurrently. `toArray_parallelBuffer` shows that running the
chunks as tasks and appending them in source order gives the same row-major
array as one sequential traversal.

Parallelism splits output positions among tasks, and each position keeps the
contraction order chosen by its scalar compiler. Task scheduling therefore
adds no parallel floating-point reduction.
-/

@[expose] public section

namespace TorchLean

open Native (flatRange)

universe u

/--
Run an arbitrary number of independent physical-buffer producers and append
their results in source order.

The first producer runs on the calling thread and may reserve capacity for the
complete result. Every remaining producer enters Lean's task pool before the
first chunk is evaluated.
-/
@[inline] def Native.parallelBuffer
    {α : Type u} [storage : Storage α] :
    List (Unit → storage.Buffer) → storage.Buffer
  | [] => storage.emptyWithCapacity 0
  | part0 :: parts =>
      let tasks := parts.map Task.spawn
      tasks.foldl
        (fun output task => appendBuffer task.get output)
        (part0 ())

/--
Task execution has the same value as evaluating and appending all chunk
producers sequentially.
-/
theorem Native.parallelBuffer_eq_foldl
    {α : Type u} [storage : Storage α]
    (parts : List (Unit → storage.Buffer)) :
    parallelBuffer parts =
      parts.foldl
        (fun output part => appendBuffer (part ()) output)
        (storage.emptyWithCapacity 0) := by
  cases parts with
  | nil => rfl
  | cons part parts =>
      simp only [parallelBuffer, List.foldl_cons, List.foldl_map, Task.spawn]
      rw [appendBuffer_empty]

/--
Folding one contiguous range of the outer axis constructs the corresponding
contiguous range of the full row-major output, independently of the reserved
buffer capacity.
-/
theorem Shape.Coord.toArray_foldl_push_outerRange
    {α : Type u} [storage : Storage α]
    (outer start count : Nat) (shape : Shape) (capacity : Nat)
    (hRange : start + count ≤ outer)
    (values : Fin (Shape.size (outer :: shape)) → α) :
    storage.toArray (foldl (count :: shape)
        (fun output coordinate =>
          storage.push output <|
            values <|
              Coord.linearize (s := outer :: shape)
                ((⟨start + coordinate.1, by omega⟩ : Fin outer),
                  coordinate.2))
        (storage.emptyWithCapacity capacity)) =
      Array.ofFn
        (flatRange (Shape.size (outer :: shape))
          (start * Shape.size shape) (count * Shape.size shape)
          (by
            change
              start * Shape.size shape + count * Shape.size shape ≤
                outer * Shape.size shape
            rw [← Nat.add_mul]
            exact Nat.mul_le_mul_right (Shape.size shape) hRange)
          values) := by
  let localValues : Fin (Shape.size (count :: shape)) → α :=
    fun index =>
      let coordinate := Coord.unlinearize index
      values <|
        Coord.linearize (s := outer :: shape)
          ((⟨start + coordinate.1, by omega⟩ : Fin outer),
            coordinate.2)
  calc
    _ = Array.ofFn localValues := by
      rw [foldl_eq_fin_foldl]
      exact Storage.toArray_fin_foldl_push_eq_ofFn capacity localValues
    _ = _ := by
      apply congrArg Array.ofFn
      funext index
      simp only [localValues, flatRange]
      congr 1
      apply Fin.ext
      have hLocal :=
        congrArg Fin.val (Coord.linearize_unlinearize index)
      change
        (Coord.linearize (s := count :: shape)
          ((Coord.unlinearize index).1,
            (Coord.unlinearize index).2)).val = index.val at hLocal
      simp only [Coord.linearize_cons_val, Shape.size_cons] at hLocal ⊢
      rw [Nat.mul_add, Nat.mul_comm (Shape.size shape) start]
      omega

/--
Any certified ordered partition, evaluated through `parallelBuffer`,
reconstructs the original finite-function array.
-/
theorem Native.toArray_parallelBuffer
    {α : Type u} [storage : Storage α]
    {total : Nat} {values : Fin total → α}
    {parts : List (Unit → storage.Buffer)}
    (partition : OrderedFlatPartition total values 0 parts) :
    storage.toArray (parallelBuffer parts) = Array.ofFn values := by
  rw [parallelBuffer_eq_foldl]
  apply partition.foldl_append_eq_ofFn
    (storage.emptyWithCapacity 0) (by omega)
  rw [storage.toArray_emptyWithCapacity]
  exact Array.ofFn_zero.symm

end TorchLean
