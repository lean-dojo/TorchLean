/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Storage

/-!
# Certified native finite loops

Generated tensor kernels use these primitives when a static traversal length
fits every Lean target. The executable loops carry `USize` counters, while
their theorems identify the results with standard `Fin.foldl` and
`Array.ofFn` semantics.

The primitives are independent of any one operation, so rearrangement,
repetition, reduction, packing, unpacking and einsum all share the same
traversal boundary.
-/

@[expose] public section

namespace TorchLean.Native

universe u

/--
Fill the physical buffer selected for `α` in increasing native-index order.

The storage dictionary is specialized at each scalar type, so for `Float` the
loop allocates and pushes into an unboxed `FloatArray`.
-/
@[inline] def bufferOfFn
    {α : Type u} [storage : Storage α]
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (values : (index : USize) → index.toNat < length → α) :
    storage.Buffer :=
  finFoldl length bound hBound
    (fun output index hIndex => storage.push output (values index hIndex))
    (storage.emptyWithCapacity length)

/--
The native physical-buffer builder observes as `Array.ofFn` when its callback
agrees with the semantic finite-index function.
-/
theorem bufferOfFn_toArray
    {α : Type u} [storage : Storage α]
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (nativeValues : (index : USize) → index.toNat < length → α)
    (values : Fin length → α)
    (hValues :
      ∀ (index : USize) (hIndex : index.toNat < length),
        nativeValues index hIndex = values ⟨index.toNat, hIndex⟩) :
    storage.toArray
        (bufferOfFn length bound hBound nativeValues) =
      Array.ofFn values := by
  rw [bufferOfFn]
  rw [finFoldl_eq_fin_foldl_of_eq length bound hBound
    (fun output index hIndex =>
      storage.push output (nativeValues index hIndex))
    (fun output index => storage.push output (values index))
    (storage.emptyWithCapacity length) (by
      intro output index hIndex
      rw [hValues index hIndex])]
  exact Storage.toArray_ofFn values

/-- The native physical-buffer builder creates the requested scalar count. -/
theorem bufferOfFn_size
    {α : Type u} [storage : Storage α]
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (values : (index : USize) → index.toNat < length → α) :
    storage.size (bufferOfFn length bound hBound values) = length := by
  let semanticValues : Fin length → α := fun index =>
    values
      (USize.ofNatLT index.val
        (Nat.lt_trans index.isLt (by
          rw [← hBound]
          exact USize.toNat_lt_size bound)))
      (by simp)
  rw [← storage.toArray_size,
    bufferOfFn_toArray length bound hBound values semanticValues]
  · exact Array.size_ofFn
  · intro index hIndex
    simp only [semanticValues, USize.ofNatLT_toNat]

/--
The storage gather observes as `Array.ofFn` when every copied source entry
agrees with the semantic finite-index function.
-/
theorem _root_.TorchLean.Storage.toArray_gather
    {α : Type u} [storage : Storage α]
    (source : storage.Buffer)
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (sourceIndices :
      (index : USize) → index.toNat < length → USize)
    (hSourceIndices :
      ∀ (index : USize) (hIndex : index.toNat < length),
        (sourceIndices index hIndex).toNat < storage.size source)
    (values : Fin length → α)
    (hValues :
      ∀ (index : USize) (hIndex : index.toNat < length),
        storage.uget source (sourceIndices index hIndex)
            (hSourceIndices index hIndex) =
          values ⟨index.toNat, hIndex⟩) :
    storage.toArray
        (storage.gather source length bound hBound
          sourceIndices hSourceIndices) =
      Array.ofFn values := by
  rw [storage.gather_eq_finFoldl]
  rw [finFoldl_eq_fin_foldl_of_eq length bound hBound
    (fun output index hIndex =>
      storage.copyAt source (sourceIndices index hIndex)
        (hSourceIndices index hIndex) output)
    (fun output index => storage.push output (values index))
    (storage.emptyWithCapacity length)]
  · exact Storage.toArray_ofFn values
  · intro output index hIndex
    apply storage.toArray_injective
    have hArray :
        (sourceIndices index hIndex).toNat <
            (storage.toArray source).size := by
      rw [storage.toArray_size]
      exact hSourceIndices index hIndex
    rw [storage.toArray_copyAt _ _ _ _ hArray, storage.toArray_push]
    congr 1
    rw [storage.toArray_uget source _ _ hArray]
    exact hValues index hIndex

/-- The storage gather creates the requested scalar count. -/
theorem _root_.TorchLean.Storage.size_gather
    {α : Type u} [storage : Storage α]
    (source : storage.Buffer)
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (sourceIndices :
      (index : USize) → index.toNat < length → USize)
    (hSourceIndices :
      ∀ (index : USize) (hIndex : index.toNat < length),
        (sourceIndices index hIndex).toNat < storage.size source) :
    storage.size
        (storage.gather source length bound hBound
          sourceIndices hSourceIndices) =
      length := by
  let values : Fin length → α := fun index =>
    storage.uget source
      (sourceIndices
        (USize.ofNatLT index.val
          (Nat.lt_trans index.isLt (by
            rw [← hBound]
            exact USize.toNat_lt_size bound)))
        (by simp))
      (hSourceIndices _ _)
  rw [← storage.toArray_size,
    Storage.toArray_gather source length bound hBound
      sourceIndices hSourceIndices values]
  · exact Array.size_ofFn
  · intro index hIndex
    simp only [values, USize.ofNatLT_toNat]

/--
Fill a physical buffer with an executable update callback.

The callback returns the updated physical buffer where `bufferOfFn`
expects a scalar, so movement kernels can copy boxed objects or unboxed native
entries without materializing a scalar at the callback ABI.
-/
@[inline] def bufferOfCopyFn
    {α : Type u} [storage : Storage α]
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (nativeStep :
      storage.Buffer → (index : USize) → index.toNat < length →
        storage.Buffer) :
    storage.Buffer :=
  finFoldl length bound hBound nativeStep
    (storage.emptyWithCapacity length)

/--
A physical-update buffer fill observes as `Array.ofFn` when every update is
the corresponding semantic push.
-/
theorem bufferOfCopyFn_toArray
    {α : Type u} [storage : Storage α]
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (nativeStep :
      storage.Buffer → (index : USize) → index.toNat < length →
        storage.Buffer)
    (values : Fin length → α)
    (hStep :
      ∀ (output : storage.Buffer) (index : USize)
          (hIndex : index.toNat < length),
        nativeStep output index hIndex =
          storage.push output (values ⟨index.toNat, hIndex⟩)) :
    storage.toArray
        (bufferOfCopyFn length bound hBound nativeStep) =
      Array.ofFn values := by
  rw [bufferOfCopyFn,
    finFoldl_eq_fin_foldl_of_eq length bound hBound
      nativeStep
      (fun output index =>
        storage.push output (values index))
      (storage.emptyWithCapacity length)]
  · exact Storage.toArray_ofFn values
  · exact hStep

/-- A physical-update buffer fill creates the requested scalar count. -/
theorem bufferOfCopyFn_size
    {α : Type u} [storage : Storage α]
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (nativeStep :
      storage.Buffer → (index : USize) → index.toNat < length →
        storage.Buffer)
    (values : Fin length → α)
    (hStep :
      ∀ (output : storage.Buffer) (index : USize)
          (hIndex : index.toNat < length),
        nativeStep output index hIndex =
          storage.push output (values ⟨index.toNat, hIndex⟩)) :
    storage.size
        (bufferOfCopyFn length bound hBound nativeStep) =
      length := by
  rw [← storage.toArray_size,
    bufferOfCopyFn_toArray length bound hBound
      nativeStep values hStep]
  exact Array.size_ofFn

end TorchLean.Native
