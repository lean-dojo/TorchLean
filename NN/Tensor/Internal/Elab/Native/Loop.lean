/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Storage

/-!
# Certified native finite loops

Generated tensor kernels use these primitives when a static traversal length
fits every Lean target. The executable loops carry `USize` counters, while
their theorems identify the results with standard `Fin.foldl` and
`Array.ofFn` semantics.

This module is operation-independent. Rearrangement, repetition, reduction,
packing, unpacking, and einsum all share the same traversal boundary.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Elab.Impl

universe u

/--
Combine branch-local equality proofs for a generated runtime decision.

Native segmented kernels use this theorem to assemble arbitrary-length
decision trees while keeping each leaf's semantic certificate local.
-/
theorem ite_eq_of_branch_eq {α : Type u} (condition : Prop)
    [Decidable condition] (thenValue elseValue expected : α)
    (hThen : condition → thenValue = expected)
    (hElse : ¬ condition → elseValue = expected) :
    (if condition then thenValue else elseValue) = expected := by
  by_cases h : condition
  · simpa [h] using hThen h
  · simpa [h] using hElse h

/--
Combine proof-dependent branches that both compute the same semantic value.

The branch evidence erases in generated native code, while each branch may
use it to certify subtraction, bounds, or direct buffer reads.
-/
theorem dite_eq_of_branch_eq {α : Type u} (condition : Prop)
    [Decidable condition]
    (thenValue : condition → α) (elseValue : ¬ condition → α)
    (expected : α)
    (hThen : ∀ proof, thenValue proof = expected)
    (hElse : ∀ proof, elseValue proof = expected) :
    (if proof : condition then thenValue proof else elseValue proof) =
      expected := by
  by_cases h : condition
  · simpa [h] using hThen h
  · simpa [h] using hElse h

/--
Fill one array in increasing native-index order.

The callback's bound proof is erased, so the executable builder contains one
allocation and one `USize` loop.
-/
@[inline] def nativeArrayOfFn
    {α : Type u} (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (values : (index : USize) → index.toNat < length → α) : Array α :=
  nativeFinFoldl length bound hBound
    (fun output index hIndex => output.push (values index hIndex))
    (Array.emptyWithCapacity length)

/--
The native array builder equals `Array.ofFn` whenever its native callback
agrees pointwise with the semantic finite-index function.
-/
theorem nativeArrayOfFn_eq_array_ofFn
    {α : Type u} (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (nativeValues : (index : USize) → index.toNat < length → α)
    (values : Fin length → α)
    (hValues :
      ∀ (index : USize) (hIndex : index.toNat < length),
        nativeValues index hIndex = values ⟨index.toNat, hIndex⟩) :
    nativeArrayOfFn length bound hBound nativeValues =
      Array.ofFn values := by
  rw [nativeArrayOfFn]
  rw [nativeFinFoldl_eq_fin_foldl_of_eq length bound hBound
    (fun output index hIndex =>
      output.push (nativeValues index hIndex))
    (fun output index => output.push (values index))
    (Array.emptyWithCapacity length) (by
      intro output index hIndex
      rw [hValues index hIndex])]
  exact Storage.fin_foldl_push_eq_array_ofFn length values

/-- The native builder produces the statically requested number of entries. -/
theorem nativeArrayOfFn_size
    {α : Type u} (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (values : (index : USize) → index.toNat < length → α) :
    (nativeArrayOfFn length bound hBound values).size = length := by
  let semanticValues : Fin length → α := fun index =>
    values
      (USize.ofNatLT index.val
        (Nat.lt_trans index.isLt (by
          rw [← hBound]
          exact USize.toNat_lt_size bound)))
      (by simp)
  rw [nativeArrayOfFn_eq_array_ofFn length bound hBound
    values semanticValues]
  · exact Array.size_ofFn
  · intro index hIndex
    have hNativeIndex :
        USize.ofNatLT index.toNat
            (Nat.lt_trans hIndex (by
              rw [← hBound]
              exact USize.toNat_lt_size bound)) =
          index := by
      apply USize.toNat.inj
      simp
    change values index hIndex =
      values
        (USize.ofNatLT index.toNat
          (Nat.lt_trans hIndex (by
            rw [← hBound]
            exact USize.toNat_lt_size bound)))
        _
    cases hNativeIndex
    rfl

/--
Fill the physical buffer selected for `α` in increasing native-index order.

The storage dictionary is specialized at each scalar type. For `Float`, this
loop therefore allocates and pushes directly into an unboxed `FloatArray`.
-/
@[inline] def nativeBufferOfFn
    {α : Type u} [storage : Storage α]
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (values : (index : USize) → index.toNat < length → α) :
    storage.Buffer :=
  nativeFinFoldl length bound hBound
    (fun output index hIndex => storage.push output (values index hIndex))
    (storage.emptyWithCapacity length)

/--
The native physical-buffer builder observes as `Array.ofFn` when its callback
agrees with the semantic finite-index function.
-/
theorem nativeBufferOfFn_toArray
    {α : Type u} [storage : Storage α]
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (nativeValues : (index : USize) → index.toNat < length → α)
    (values : Fin length → α)
    (hValues :
      ∀ (index : USize) (hIndex : index.toNat < length),
        nativeValues index hIndex = values ⟨index.toNat, hIndex⟩) :
    storage.toArray
        (nativeBufferOfFn length bound hBound nativeValues) =
      Array.ofFn values := by
  rw [nativeBufferOfFn]
  rw [nativeFinFoldl_eq_fin_foldl_of_eq length bound hBound
    (fun output index hIndex =>
      storage.push output (nativeValues index hIndex))
    (fun output index => storage.push output (values index))
    (storage.emptyWithCapacity length) (by
      intro output index hIndex
      rw [hValues index hIndex])]
  exact Storage.toArray_ofFn values

/-- The native physical-buffer builder creates the requested scalar count. -/
theorem nativeBufferOfFn_size
    {α : Type u} [storage : Storage α]
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (values : (index : USize) → index.toNat < length → α) :
    storage.size (nativeBufferOfFn length bound hBound values) = length := by
  let semanticValues : Fin length → α := fun index =>
    values
      (USize.ofNatLT index.val
        (Nat.lt_trans index.isLt (by
          rw [← hBound]
          exact USize.toNat_lt_size bound)))
      (by simp)
  rw [← storage.toArray_size,
    nativeBufferOfFn_toArray length bound hBound values semanticValues]
  · exact Array.size_ofFn
  · intro index hIndex
    have hNativeIndex :
        USize.ofNatLT index.toNat
            (Nat.lt_trans hIndex (by
              rw [← hBound]
              exact USize.toNat_lt_size bound)) =
          index := by
      apply USize.toNat.inj
      simp
    change values index hIndex =
      values
        (USize.ofNatLT index.toNat
          (Nat.lt_trans hIndex (by
            rw [← hBound]
            exact USize.toNat_lt_size bound)))
        _
    cases hNativeIndex
    rfl

/--
Fill a physical buffer by copying entries from another buffer.

The source-index callback never exposes `α`. Generic arrays can therefore
retain and transfer existing boxed values, while specialized scalar arrays
perform direct unboxed reads and writes.
-/
@[inline] def nativeBufferGather
    {α : Type u} [storage : Storage α]
    (source : storage.Buffer)
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (sourceIndices :
      (index : USize) → index.toNat < length → USize)
    (hSourceIndices :
      ∀ (index : USize) (hIndex : index.toNat < length),
        (sourceIndices index hIndex).toNat < storage.size source) :
    storage.Buffer :=
  storage.gather source length bound hBound
    sourceIndices hSourceIndices

/--
The native gather observes as `Array.ofFn` when every copied source entry
agrees with the semantic finite-index function.
-/
theorem nativeBufferGather_toArray
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
        (nativeBufferGather source length bound hBound
          sourceIndices hSourceIndices) =
      Array.ofFn values := by
  rw [nativeBufferGather, storage.gather_eq_nativeFinFoldl]
  rw [nativeFinFoldl_eq_fin_foldl_of_eq length bound hBound
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

/-- The native gather creates the requested scalar count. -/
theorem nativeBufferGather_size
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
        (nativeBufferGather source length bound hBound
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
    nativeBufferGather_toArray source length bound hBound
      sourceIndices hSourceIndices values]
  · exact Array.size_ofFn
  · intro index hIndex
    have hNativeIndex :
        USize.ofNatLT index.toNat
            (Nat.lt_trans hIndex (by
              rw [← hBound]
              exact USize.toNat_lt_size bound)) =
          index := by
      apply USize.toNat.inj
      simp
    change storage.uget source (sourceIndices index hIndex) _ =
      storage.uget source
        (sourceIndices
          (USize.ofNatLT index.toNat
            (Nat.lt_trans hIndex (by
              rw [← hBound]
              exact USize.toNat_lt_size bound)))
          _) _
    cases hNativeIndex
    rfl

/--
Fill a physical buffer with an executable update callback.

Unlike `nativeBufferOfFn`, the callback returns the updated physical buffer
instead of a scalar. Movement kernels can therefore copy boxed objects or
unboxed native entries without materializing a scalar at the callback ABI.
-/
@[inline] def nativeBufferOfCopyFn
    {α : Type u} [storage : Storage α]
    (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (nativeStep :
      storage.Buffer → (index : USize) → index.toNat < length →
        storage.Buffer) :
    storage.Buffer :=
  nativeFinFoldl length bound hBound nativeStep
    (storage.emptyWithCapacity length)

/--
A physical-update buffer fill observes as `Array.ofFn` when every update is
the corresponding semantic push.
-/
theorem nativeBufferOfCopyFn_toArray
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
        (nativeBufferOfCopyFn length bound hBound nativeStep) =
      Array.ofFn values := by
  rw [nativeBufferOfCopyFn,
    nativeFinFoldl_eq_fin_foldl_of_eq length bound hBound
      nativeStep
      (fun output index =>
        storage.push output (values index))
      (storage.emptyWithCapacity length)]
  · exact Storage.toArray_ofFn values
  · exact hStep

/-- A physical-update buffer fill creates the requested scalar count. -/
theorem nativeBufferOfCopyFn_size
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
        (nativeBufferOfCopyFn length bound hBound nativeStep) =
      length := by
  rw [← storage.toArray_size,
    nativeBufferOfCopyFn_toArray length bound hBound
      nativeStep values hStep]
  exact Array.size_ofFn

end TorchLean.Tensor.Internal.Elab.Impl
