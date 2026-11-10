/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Rep.Reindex -- shake: keep
public import TorchLean.Tensor.Rep.Traversal -- shake: keep

/-!
# Certified native tensor construction

Generated kernels build their output tensors through `tensorOfFlatFn` and the definitions
here, which lift the storage gather and the operation-independent buffer builder
`bufferOfCopyFn` to shaped tensors. A kernel fills one contiguous buffer with a `USize` loop,
and its correctness theorem rewrites the result to the `Rep.ofFlatFn` value (or the given
reference) that the rest of the library reasons about.
-/

@[expose] public section

namespace TorchLean.Native

open Tensor (Rep)

universe u v

/--
Build a tensor by copying physical entries from an existing tensor.

The source-index callback returns only native indices. Generic array storage
can therefore transfer existing boxed objects directly, while specialized
storage performs raw scalar-buffer reads and writes.
-/
@[inline] def tensorGather
    {α : Type u} [storage : Storage α]
    {sourceShape outputShape : Shape}
    (source : Rep α sourceShape)
    (bound : USize) (hBound : bound.toNat = Shape.size outputShape)
    (sourceIndices :
      (index : USize) → index.toNat < Shape.size outputShape → USize)
    (hSourceIndices :
      ∀ (index : USize) (hIndex : index.toNat < Shape.size outputShape),
        (sourceIndices index hIndex).toNat < Shape.size sourceShape)
    (values : Fin (Shape.size outputShape) → α)
    (_hValues :
      ∀ (index : USize) (hIndex : index.toNat < Shape.size outputShape),
        source.getFlatUSize (sourceIndices index hIndex)
            (hSourceIndices index hIndex) =
          values ⟨index.toNat, hIndex⟩) :
    Rep α outputShape :=
  let hBufferIndices :
      ∀ (index : USize) (hIndex : index.toNat < Shape.size outputShape),
        (sourceIndices index hIndex).toNat < storage.size source.buffer :=
    fun index hIndex => by
      simpa only [source.size_eq] using hSourceIndices index hIndex
  Rep.mk
    (storage.gather source.buffer (Shape.size outputShape)
      bound hBound sourceIndices hBufferIndices)
    (Storage.size_gather source.buffer (Shape.size outputShape)
      bound hBound sourceIndices hBufferIndices)

/-- A physical tensor gather equals its coordinate-level flat function. -/
theorem tensorGather_correct
    {α : Type u} [storage : Storage α]
    {sourceShape outputShape : Shape}
    (source : Rep α sourceShape)
    (bound : USize) (hBound : bound.toNat = Shape.size outputShape)
    (sourceIndices :
      (index : USize) → index.toNat < Shape.size outputShape → USize)
    (hSourceIndices :
      ∀ (index : USize) (hIndex : index.toNat < Shape.size outputShape),
        (sourceIndices index hIndex).toNat < Shape.size sourceShape)
    (values : Fin (Shape.size outputShape) → α)
    (hValues :
      ∀ (index : USize) (hIndex : index.toNat < Shape.size outputShape),
        source.getFlatUSize (sourceIndices index hIndex)
            (hSourceIndices index hIndex) =
          values ⟨index.toNat, hIndex⟩) :
    tensorGather source bound hBound sourceIndices hSourceIndices
        values hValues =
      Rep.ofFlatFn values := by
  apply Rep.mk_eq_ofFlatFn values
  exact Storage.toArray_gather source.buffer
    (Shape.size outputShape) bound hBound sourceIndices
    (fun index hIndex => by
      simpa only [source.size_eq] using hSourceIndices index hIndex)
    values (by
      intro index hIndex
      simpa only [Rep.getFlatUSize] using hValues index hIndex)

/-- Copy one native flat tensor entry into an output physical buffer. -/
@[inline] def tensorCopyAt
    {α : Type u} [storage : Storage α] {shape : Shape}
    (source : Rep α shape) (index : USize)
    (hIndex : index.toNat < Shape.size shape)
    (output : storage.Buffer) : storage.Buffer :=
  storage.copyAt source.buffer index
    (by simpa only [source.size_eq] using hIndex) output

/--
Copying a tensor entry is the same physical update as pushing its observed
native flat value.
-/
theorem tensorCopyAt_eq_push_of_eq
    {α : Type u} [storage : Storage α] {shape : Shape}
    (source : Rep α shape) (index : USize)
    (hIndex : index.toNat < Shape.size shape)
    (output : storage.Buffer) (value : α)
    (hValue : source.getFlatUSize index hIndex = value) :
    tensorCopyAt source index hIndex output =
      storage.push output value := by
  exact storage.copyAt_eq_push_of_uget_eq source.buffer index
    (by simpa only [source.size_eq] using hIndex)
    output value (by
      simpa only [Rep.getFlatUSize] using hValue)

/--
Build a tensor with a physical-buffer update callback.

This is the movement-oriented counterpart of `tensorOfFlatFn`: the
callback copies or appends physical entries directly instead of returning an
`α` value.
-/
@[inline] def tensorOfCopyFn
    {α : Type u} [storage : Storage α] {shape : Shape}
    (bound : USize) (hBound : bound.toNat = Shape.size shape)
    (nativeStep :
      storage.Buffer → (index : USize) →
        index.toNat < Shape.size shape → storage.Buffer)
    (values : Fin (Shape.size shape) → α)
    (hStep :
      ∀ (output : storage.Buffer) (index : USize)
          (hIndex : index.toNat < Shape.size shape),
        nativeStep output index hIndex =
          storage.push output (values ⟨index.toNat, hIndex⟩)) :
    Rep α shape :=
  Rep.mk
    (bufferOfCopyFn (Shape.size shape) bound hBound nativeStep)
    (bufferOfCopyFn_size (Shape.size shape) bound hBound
      nativeStep values hStep)

/-- A physical-update tensor fill equals its coordinate-level flat function. -/
theorem tensorOfCopyFn_correct
    {α : Type u} [storage : Storage α] {shape : Shape}
    (bound : USize) (hBound : bound.toNat = Shape.size shape)
    (nativeStep :
      storage.Buffer → (index : USize) →
        index.toNat < Shape.size shape → storage.Buffer)
    (values : Fin (Shape.size shape) → α)
    (hStep :
      ∀ (output : storage.Buffer) (index : USize)
          (hIndex : index.toNat < Shape.size shape),
        nativeStep output index hIndex =
          storage.push output (values ⟨index.toNat, hIndex⟩)) :
    tensorOfCopyFn bound hBound nativeStep values hStep =
      Rep.ofFlatFn values := by
  apply Rep.mk_eq_ofFlatFn values
  exact bufferOfCopyFn_toArray
    (Shape.size shape) bound hBound nativeStep values hStep

/--
Return a native tensor whose semantic reference appears only in an erased correctness
certificate.

The reference is an implicit argument fixed by the proof, so the call carries neither the
reference value nor the compiled value that `tensorKernel` takes. It fits kernels where
building either value would duplicate a large dependent plan or do avoidable strict work at
runtime.
-/
@[inline] def certifiedTensor
    {α : Type u} [Storage α] {shape : Shape}
    (implementation : Rep α shape)
    {reference : Rep α shape}
    (_hImplementation : implementation = reference) :
    Rep α shape :=
  implementation

/-- A tensor returned through an erased certificate equals its reference. -/
@[grind =] theorem certifiedTensor_correct
    {α : Type u} [Storage α] {shape : Shape}
    (implementation reference : Rep α shape)
    (hImplementation : implementation = reference) :
    certifiedTensor implementation hImplementation = reference :=
  hImplementation

/--
Evaluate a staged value before entering the consumer that uses it.

The `@[noinline]` attribute keeps the materialization separate from its consumer. The
transformation scheduler stages an intermediate tensor this way when materializing it is cheaper
than composing another flat-index program. Evaluation order rests on the compiler honoring
`@[noinline]`, and `Native.stage_eq` fixes only the returned value.
-/
@[noinline] def stage {α : Type u} {β : Type v}
    (value : α) (next : α → β) : β :=
  next value

/-- Staging moves evaluation and leaves the returned value unchanged. -/
@[grind =] theorem stage_eq {α : Type u} {β : Type v}
    (value : α) (next : α → β) :
    stage value next = next value :=
  rfl

/--
Execute a native tensor implementation while retaining semantic and compiled
references.

Both equality proofs are erased in generated code. Proofs about the call rewrite to the
independent semantic reference, and a downstream compiler can consume the compiled reference
without recovering executable callbacks or repeating the compilation proof.
-/
@[inline] def tensorKernel
    {α : Type u} [Storage α] {shape : Shape}
    (reference : Rep α shape)
    (compiled : Rep α shape)
    (implementation : Rep α shape)
    (_hCompiled : compiled = reference)
    (_hImplementation : implementation = compiled) :
    Rep α shape :=
  implementation

/-- A native tensor implementation equals its compact compiled reference. -/
theorem tensorKernel_eq_compiled
    {α : Type u} [Storage α] {shape : Shape}
    (reference compiled implementation : Rep α shape)
    (hCompiled : compiled = reference)
    (hImplementation : implementation = compiled) :
    tensorKernel reference compiled implementation hCompiled
        hImplementation =
      compiled :=
  hImplementation

/-- A native tensor kernel equals its semantic reference. -/
@[grind =] theorem tensorKernel_correct
    {α : Type u} [Storage α] {shape : Shape}
    (reference compiled implementation : Rep α shape)
    (hCompiled : compiled = reference)
    (hImplementation : implementation = compiled) :
    tensorKernel reference compiled implementation hCompiled
        hImplementation =
      reference :=
  hImplementation.trans hCompiled

end TorchLean.Native
