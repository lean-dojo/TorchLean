/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Basic.Reindex -- shake: keep
public import NN.Tensor.Internal.Representation.Basic.Traversal -- shake: keep

/-!
# Certified native tensor construction

This module lifts the operation-independent native array builder to shaped
tensors. Generated kernels fill one contiguous buffer with a `USize` loop,
while the correctness theorem exposes the ordinary `Rep.ofFlatFn`
semantics used throughout the library.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Elab.Impl

universe u v

/--
Build a tensor with one native row-major output loop.

The shape-size equality and callback bound proofs erase during code
generation.
-/
@[inline] def nativeTensorOfFlatFn
    {α : Type u} [storage : Storage α] {shape : Shape}
    (bound : USize) (hBound : bound.toNat = Shape.size shape)
    (nativeValues :
      (index : USize) → index.toNat < Shape.size shape → α)
    (values : Fin (Shape.size shape) → α)
    (_hValues :
      ∀ (index : USize) (hIndex : index.toNat < Shape.size shape),
        nativeValues index hIndex =
          values ⟨index.toNat, hIndex⟩) :
    Rep α shape :=
  Rep.mk
    (nativeBufferOfFn (Shape.size shape) bound hBound nativeValues)
    (nativeBufferOfFn_size (Shape.size shape) bound hBound nativeValues)

/--
The native tensor builder equals `Rep.ofFlatFn` whenever its callback
agrees pointwise with the semantic finite-index function.
-/
theorem nativeTensorOfFlatFn_correct
    {α : Type u} [storage : Storage α] {shape : Shape}
    (bound : USize) (hBound : bound.toNat = Shape.size shape)
    (nativeValues :
      (index : USize) → index.toNat < Shape.size shape → α)
    (values : Fin (Shape.size shape) → α)
    (hValues :
      ∀ (index : USize) (hIndex : index.toNat < Shape.size shape),
        nativeValues index hIndex =
          values ⟨index.toNat, hIndex⟩) :
    nativeTensorOfFlatFn bound hBound nativeValues values hValues =
      Rep.ofFlatFn values := by
  apply Rep.mk_eq_ofFlatFn values
  exact nativeBufferOfFn_toArray
    (Shape.size shape) bound hBound nativeValues values hValues

/--
Build a tensor by copying physical entries from an existing tensor.

The source-index callback returns only native indices. Generic array storage
can therefore transfer existing boxed objects directly, while specialized
storage performs raw scalar-buffer reads and writes.
-/
@[inline] def nativeTensorGather
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
    (nativeBufferGather source.buffer (Shape.size outputShape)
      bound hBound sourceIndices hBufferIndices)
    (nativeBufferGather_size source.buffer (Shape.size outputShape)
      bound hBound sourceIndices hBufferIndices)

/-- A physical tensor gather equals its coordinate-level flat function. -/
theorem nativeTensorGather_correct
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
    nativeTensorGather source bound hBound sourceIndices hSourceIndices
        values hValues =
      Rep.ofFlatFn values := by
  apply Rep.mk_eq_ofFlatFn values
  exact nativeBufferGather_toArray source.buffer
    (Shape.size outputShape) bound hBound sourceIndices
    (fun index hIndex => by
      simpa only [source.size_eq] using hSourceIndices index hIndex)
    values (by
      intro index hIndex
      simpa only [Rep.getFlatUSize] using hValues index hIndex)

/-- Copy one native flat tensor entry into an output physical buffer. -/
@[inline] def nativeTensorCopyAt
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
theorem nativeTensorCopyAt_eq_push_of_getFlatUSize_eq
    {α : Type u} [storage : Storage α] {shape : Shape}
    (source : Rep α shape) (index : USize)
    (hIndex : index.toNat < Shape.size shape)
    (output : storage.Buffer) (value : α)
    (hValue : source.getFlatUSize index hIndex = value) :
    nativeTensorCopyAt source index hIndex output =
      storage.push output value := by
  exact storage.copyAt_eq_push_of_uget_eq source.buffer index
    (by simpa only [source.size_eq] using hIndex)
    output value (by
      simpa only [Rep.getFlatUSize] using hValue)

/--
Build a tensor with a physical-buffer update callback.

This is the movement-oriented counterpart of `nativeTensorOfFlatFn`: the
callback copies or appends physical entries directly instead of returning an
`α` value.
-/
@[inline] def nativeTensorOfCopyFn
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
    (nativeBufferOfCopyFn (Shape.size shape) bound hBound nativeStep)
    (nativeBufferOfCopyFn_size (Shape.size shape) bound hBound
      nativeStep values hStep)

/-- A physical-update tensor fill equals its coordinate-level flat function. -/
theorem nativeTensorOfCopyFn_correct
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
    nativeTensorOfCopyFn bound hBound nativeStep values hStep =
      Rep.ofFlatFn values := by
  apply Rep.mk_eq_ofFlatFn values
  exact nativeBufferOfCopyFn_toArray
    (Shape.size shape) bound hBound nativeStep values hStep

/--
Return a native tensor whose semantic reference is carried only by an erased
correctness certificate.

Unlike `nativeTensorKernel`, this boundary has no ordinary reference value or
compiled value argument. It is useful when constructing either reference
would duplicate a large dependent plan or perform avoidable strict runtime
work.
-/
@[inline] def certifiedNativeTensor
    {α : Type u} [Storage α] {shape : Shape}
    (implementation : Rep α shape)
    {reference : Rep α shape}
    (_hImplementation : implementation = reference) :
    Rep α shape :=
  implementation

/-- A tensor returned through an erased certificate equals its reference. -/
@[grind =] theorem certifiedNativeTensor_correct
    {α : Type u} [Storage α] {shape : Shape}
    (implementation reference : Rep α shape)
    (hImplementation : implementation = reference) :
    certifiedNativeTensor implementation hImplementation = reference :=
  hImplementation

/--
Evaluate a staged value before entering the consumer that uses it.

Lean is strict, and `@[noinline]` prevents native code generation from
sinking `value` into a consumer closure. The transformation scheduler uses
this boundary when materializing an intermediate tensor is cheaper than
composing another flat-index program.
-/
@[noinline] def nativeStage {α : Type u} {β : Type v}
    (value : α) (next : α → β) : β :=
  next value

/-- Staging changes evaluation placement, not the returned value. -/
@[grind =] theorem nativeStage_eq {α : Type u} {β : Type v}
    (value : α) (next : α → β) :
    nativeStage value next = next value :=
  rfl

/--
Execute a native tensor implementation while retaining semantic and compiled
references.

Both equality proofs erase during code generation. Standalone reasoning uses
the independent semantic reference, while a downstream compiler may consume
the compiled reference without recovering executable callbacks or repeating
the compilation proof.
-/
@[inline] def nativeTensorKernel
    {α : Type u} [Storage α] {shape : Shape}
    (reference : Rep α shape)
    (compiled : Rep α shape)
    (implementation : Rep α shape)
    (_hCompiled : compiled = reference)
    (_hImplementation : implementation = compiled) :
    Rep α shape :=
  implementation

/-- A native tensor implementation equals its compact compiled reference. -/
theorem nativeTensorKernel_eq_compiled
    {α : Type u} [Storage α] {shape : Shape}
    (reference compiled implementation : Rep α shape)
    (hCompiled : compiled = reference)
    (hImplementation : implementation = compiled) :
    nativeTensorKernel reference compiled implementation hCompiled
        hImplementation =
      compiled :=
  hImplementation

/-- A certified native tensor implementation equals its compact reference. -/
@[grind =] theorem nativeTensorKernel_correct
    {α : Type u} [Storage α] {shape : Shape}
    (reference compiled implementation : Rep α shape)
    (hCompiled : compiled = reference)
    (hImplementation : implementation = compiled) :
    nativeTensorKernel reference compiled implementation hCompiled
        hImplementation =
      reference :=
  hImplementation.trans hCompiled

end TorchLean.Tensor.Internal.Elab.Impl
