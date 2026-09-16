/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Batteries.Data.Fin.Fold
public import Aesop.BuiltinRules
public import Mathlib.Data.Nat.Basic
public import Mathlib.Order.RelClasses
import Mathlib.Tactic.Attr.Core
import Mathlib.Tactic.Conv
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.Widget.Calc
public import Batteries.Data.Fin.Lemmas -- shake: keep
public import Init.Data.ByteArray.Lemmas -- shake: keep
public import Init.Data.FloatArray -- shake: keep
public import Mathlib.Data.List.OfFn -- shake: keep

/-!
# Physical Tensor Storage

`Storage α` selects the physical buffer used by tensors with scalar type
`α`. Arbitrary scalar types retain ordinary `Array α` storage. `Float` uses
Lean's unboxed `FloatArray` runtime representation, while `UInt8` uses
`ByteArray`.

The `toArray` laws are the proof boundary: the kernel reasons about an ordinary
array observation, while compiled code calls the specialized buffer operations
directly. Selected pointwise, slice, and transpose operations also have
proof-visible Lean definitions with native compiled replacements. The Lean
theorems describe those definitions; agreement of the external C bodies is a
native runtime boundary documented in `docs/TRUST_BOUNDARIES.md`.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

namespace Elab.Impl

/--
Run the tail-recursive engine used by `nativeFinFoldl`.

`hBound` and the callback's index proof are erased. The executable loop
therefore carries only an unboxed bound, counter, and accumulator.
-/
@[specialize] def nativeFinFoldlLoop
    {α : Type u} (length : Nat)
    (step : α → (index : USize) → index.toNat < length → α)
    (bound : USize) (hBound : bound.toNat = length)
    (index : USize) (value : α) : α :=
  if hIndex : index < bound then
    nativeFinFoldlLoop length step bound hBound (index + 1)
      (step value index (by
        have hIndexNat := USize.lt_iff_toNat_lt.mp hIndex
        lia))
  else
    value
termination_by length - index.toNat
decreasing_by
  have hIndexNat : index.toNat < bound.toNat :=
    USize.lt_iff_toNat_lt.mp hIndex
  have hNextFits : index.toNat + 1 < USize.size := by
    have hBoundFits := USize.toNat_lt_size bound
    lia
  simp only [USize.toNat_add, USize.toNat_one,
    Nat.mod_eq_of_lt hNextFits]
  lia

/--
Fold over `Fin length` with a platform-native loop counter.

The caller supplies a native bound together with an erased proof that it
represents the semantic length.
-/
@[inline] def nativeFinFoldl
    {α : Type u} (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (step : α → (index : USize) → index.toNat < length → α)
    (initial : α) : α :=
  nativeFinFoldlLoop length step bound hBound 0 initial

/--
Starting partway through `nativeFinFoldl` traverses exactly the corresponding
suffix of `List.finRange`.
-/
private theorem nativeFinFoldl_loop_eq_finRange
    {α : Type u} (length : Nat)
    (step : α → (index : USize) → index.toNat < length → α)
    (bound : USize) (hBound : bound.toNat = length)
    (index : USize) (value : α)
    (hIndex : index.toNat ≤ length) :
    nativeFinFoldlLoop length step bound hBound index value =
      ((List.finRange length).drop index.toNat).foldl
        (fun total coordinate =>
          step total
            (USize.ofNatLT coordinate.val
              (Nat.lt_trans coordinate.isLt (by
                rw [← hBound]
                exact USize.toNat_lt_size bound)))
            (by simp))
        value := by
  refine nativeFinFoldlLoop.induct length step bound hBound
    (motive := fun index value =>
      ∀ hIndex : index.toNat ≤ length,
        nativeFinFoldlLoop length step bound hBound index value =
          ((List.finRange length).drop index.toNat).foldl
            (fun total coordinate =>
              step total
                (USize.ofNatLT coordinate.val
                  (Nat.lt_trans coordinate.isLt (by
                    rw [← hBound]
                    exact USize.toNat_lt_size bound)))
                (by simp))
            value)
    ?_ ?_ index value hIndex
  · intro index value hIndexLt inductionHypothesis hIndexLe
    rw [nativeFinFoldlLoop.eq_1, dite_eq_left hIndexLt]
    have hIndexNat : index.toNat < length := by
      have := USize.lt_iff_toNat_lt.mp hIndexLt
      lia
    have hLengthFits : length < USize.size := by
      rw [← hBound]
      exact USize.toNat_lt_size bound
    have hNextFits : index.toNat + 1 < USize.size := by
      lia
    have hNextNat : (index + 1).toNat = index.toNat + 1 := by
      simp [USize.toNat_add, Nat.mod_eq_of_lt hNextFits]
    rw [inductionHypothesis (by lia)]
    conv_rhs =>
      rw [List.drop_eq_getElem_cons (by simpa using hIndexNat)]
    simp only [List.foldl_cons, List.getElem_finRange]
    rw [hNextNat]
    congr 1
  · intro index value hIndexNotLt hIndexLe
    rw [nativeFinFoldlLoop.eq_1, dite_eq_right hIndexNotLt]
    have hIndexNatNotLt : ¬index.toNat < length := by
      intro hIndexNat
      apply hIndexNotLt
      apply USize.lt_iff_toNat_lt.mpr
      lia
    have hIndexEq : index.toNat = length := by lia
    have hDrop : (List.finRange length).drop length = [] :=
      List.drop_eq_nil_iff.mpr (by simp)
    simp only [hIndexEq, hDrop, List.foldl_nil]

/--
`nativeFinFoldl` computes exactly `Fin.foldl`.

Generated kernels can execute with an unboxed counter while proofs continue
to reason about the standard finite enumeration.
-/
theorem nativeFinFoldl_eq_fin_foldl
    {α : Type u} (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (step : α → Fin length → α) (initial : α) :
    nativeFinFoldl length bound hBound
        (fun value index hIndex =>
          step value ⟨index.toNat, hIndex⟩)
        initial =
      Fin.foldl length step initial := by
  change nativeFinFoldlLoop length
    (fun value index hIndex =>
      step value ⟨index.toNat, hIndex⟩)
    bound hBound 0 initial = _
  calc
    _ = (List.finRange length).foldl step initial := by
      simpa using nativeFinFoldl_loop_eq_finRange length
        (fun value index hIndex =>
          step value ⟨index.toNat, hIndex⟩)
        bound hBound (0 : USize) initial (by simp)
    _ = _ := (Fin.foldl_eq_foldl_finRange _ _).symm

/--
A native callback computes the semantic finite fold when it agrees with the
semantic callback at every represented index.
-/
theorem nativeFinFoldl_eq_fin_foldl_of_eq
    {α : Type u} (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (nativeStep : α → (index : USize) → index.toNat < length → α)
    (step : α → Fin length → α) (initial : α)
    (hStep :
      ∀ value index hIndex,
        nativeStep value index hIndex =
          step value ⟨index.toNat, hIndex⟩) :
    nativeFinFoldl length bound hBound nativeStep initial =
      Fin.foldl length step initial := by
  rw [show nativeStep =
      (fun value index hIndex =>
        step value ⟨index.toNat, hIndex⟩) by
    funext value index hIndex
    exact hStep value index hIndex]
  exact nativeFinFoldl_eq_fin_foldl length bound hBound step initial

end Elab.Impl

end TorchLean.Tensor.Internal

namespace TorchLean

universe u

/--
Physical tensor storage for one scalar type.

The buffer operations are used by generated native kernels. `toArray` is their
proof-facing observation and may allocate when explicitly evaluated; optimized
code therefore stays on `Buffer`, `get`, `uget`, `push`, and `appendSlice`.
-/
class Storage (α : Type u) where
  /-- Physical row-major scalar buffer. -/
  Buffer : Type u
  /-- Allocate an empty buffer with room for the requested number of scalars. -/
  emptyWithCapacity : Nat → Buffer
  /-- Append one scalar. -/
  push : Buffer → α → Buffer
  /-- Append a contiguous half-open source interval to an output buffer. -/
  appendSlice : Buffer → Nat → Nat → Buffer → Buffer
  /-- Traverse the physical buffer in row-major order. -/
  foldl : {β : Type u} → (β → α → β) → β → Buffer → β
  /-- Number of scalar entries in the buffer. -/
  size : Buffer → Nat
  /--
  Copy one physical entry from a source buffer into an output buffer.

  Unlike `push output (uget source index ...)`, this primitive does not expose
  the scalar at the caller boundary. Generic arrays can therefore transfer an
  existing boxed object directly, while scalar arrays read and write their
  unboxed representation.
  -/
  copyAt : (source : Buffer) → (index : USize) →
    index.toNat < size source → Buffer → Buffer
  /--
  Build a buffer by gathering physical entries from one source buffer.

  The storage implementation owns the complete loop so backend selection
  happens once per tensor rather than once per copied scalar.
  -/
  gather : (source : Buffer) → (length : Nat) → (bound : USize) →
    bound.toNat = length →
    (sourceIndices : (index : USize) → index.toNat < length → USize) →
    (∀ (index : USize) (hIndex : index.toNat < length),
      (sourceIndices index hIndex).toNat < size source) →
    Buffer
  /-- Physical gathering is the storage-specific native finite loop. -/
  gather_eq_nativeFinFoldl (source : Buffer)
      (length : Nat) (bound : USize)
      (hBound : bound.toNat = length)
      (sourceIndices :
        (index : USize) → index.toNat < length → USize)
      (hSourceIndices :
        ∀ (index : USize) (hIndex : index.toNat < length),
          (sourceIndices index hIndex).toNat < size source) :
    gather source length bound hBound sourceIndices hSourceIndices =
      Tensor.Internal.Elab.Impl.nativeFinFoldl length bound hBound
        (fun output index hIndex =>
          copyAt source (sourceIndices index hIndex)
            (hSourceIndices index hIndex) output)
        (emptyWithCapacity length)
  /-- Read one scalar using a natural-number index. -/
  get : (buffer : Buffer) → (index : Nat) → index < size buffer → α
  /-- Read one scalar using a platform-native index. -/
  uget : (buffer : Buffer) → (index : USize) →
    index.toNat < size buffer → α
  /-- Observe the physical buffer as an ordinary logical array. -/
  toArray : Buffer → Array α
  /-- Construct physical storage from an ordinary array. -/
  ofArray : Array α → Buffer
  /-- Buffer observation preserves size. -/
  toArray_size (buffer : Buffer) :
    (toArray buffer).size = size buffer
  /-- Buffer observation preserves natural-number reads. -/
  toArray_get (buffer : Buffer) (index : Nat)
      (hBuffer : index < size buffer)
      (hArray : index < (toArray buffer).size) :
    (toArray buffer)[index]'hArray = get buffer index hBuffer
  /-- Buffer observation preserves platform-native reads. -/
  toArray_uget (buffer : Buffer) (index : USize)
      (hBuffer : index.toNat < size buffer)
      (hArray : index.toNat < (toArray buffer).size) :
    (toArray buffer).uget index hArray = uget buffer index hBuffer
  /-- An empty physical buffer observes as an empty ordinary array. -/
  toArray_emptyWithCapacity (capacity : Nat) :
    toArray (emptyWithCapacity capacity) = Array.emptyWithCapacity capacity
  /-- Appending one scalar commutes with buffer observation. -/
  toArray_push (buffer : Buffer) (value : α) :
    toArray (push buffer value) = (toArray buffer).push value
  /-- Copying one physical entry commutes with buffer observation. -/
  toArray_copyAt (source : Buffer) (index : USize)
      (hBuffer : index.toNat < size source) (output : Buffer)
      (hArray : index.toNat < (toArray source).size) :
    toArray (copyAt source index hBuffer output) =
      (toArray output).push ((toArray source).uget index hArray)
  /-- Appending a source interval commutes with buffer observation. -/
  toArray_appendSlice (source : Buffer) (start stop : Nat)
      (output : Buffer) :
    toArray (appendSlice source start stop output) =
      toArray output ++ (toArray source).extract start stop
  /-- Row-major traversal agrees with folding the ordinary array observation. -/
  toArray_foldl {β : Type u} (step : β → α → β)
      (initial : β) (buffer : Buffer) :
    foldl step initial buffer = (toArray buffer).foldl step initial
  /-- Converting from an ordinary array is inverse to observation. -/
  toArray_ofArray (values : Array α) :
    toArray (ofArray values) = values
  /-- The ordinary array observation uniquely determines physical storage. -/
  toArray_injective : Function.Injective toArray

namespace Storage

/--
Copy-on-write replacement of one physical scalar.

Every `Storage` receives the ordinary-array fallback below, so defining a
custom scalar storage does not require implementing this capability. Packed
backends can override it to retain their native representation.
-/
class Update (α : Type u) [storage : Storage α] where
  /-- Replace one in-bounds physical entry. -/
  set : (buffer : storage.Buffer) → (index : Nat) →
    index < storage.size buffer → α → storage.Buffer
  /-- Physical replacement agrees with ordinary-array replacement. -/
  toArray_set (buffer : storage.Buffer) (index : Nat)
      (hBuffer : index < storage.size buffer) (value : α) :
    storage.toArray (set buffer index hBuffer value) =
      (storage.toArray buffer).set index value (by
        rw [storage.toArray_size]
        exact hBuffer)

/--
Universal update support for any storage.

This fallback round-trips through the logical array observation. Specialized
packed storage overrides it below with native copy-on-write replacement.
-/
@[instance_reducible]
instance (priority := low) instUpdate
    (α : Type u) [storage : Storage α] : Update α where
  set buffer index hBuffer value :=
    storage.ofArray <|
      (storage.toArray buffer).set index value (by
        rw [storage.toArray_size]
        exact hBuffer)
  toArray_set := by
    intro buffer index hBuffer value
    rw [storage.toArray_ofArray]

/--
Copying a physical entry is the same buffer update as pushing its observed
scalar value.

Generated movement kernels use this theorem to reason about `copyAt` without
exposing the scalar at the executable callback boundary.
-/
theorem copyAt_eq_push_of_uget_eq
    {α : Type u} [storage : Storage α]
    (source : storage.Buffer) (index : USize)
    (hSource : index.toNat < storage.size source)
    (output : storage.Buffer) (value : α)
    (hValue : storage.uget source index hSource = value) :
    storage.copyAt source index hSource output =
      storage.push output value := by
  apply storage.toArray_injective
  have hArray :
      index.toNat < (storage.toArray source).size := by
    rw [storage.toArray_size]
    exact hSource
  rw [storage.toArray_copyAt source index hSource output hArray,
    storage.toArray_push, storage.toArray_uget source index hSource hArray,
    hValue]

/--
Construct a physical buffer from a finite scalar function.
-/
def ofFn {α : Type u} [storage : Storage α] {length : Nat}
    (values : Fin length → α) : storage.Buffer :=
  Fin.foldl length
    (fun output index => storage.push output (values index))
    (storage.emptyWithCapacity length)

/-- Observing a finite physical-buffer fold commutes with scalar appends. -/
private theorem toArray_finFoldl_push
    {α : Type u} [storage : Storage α]
    {length : Nat} (values : Fin length → α) (initial : storage.Buffer) :
    storage.toArray
        (Fin.foldl length
          (fun output index => storage.push output (values index))
          initial) =
      Fin.foldl length
        (fun output index => output.push (values index))
        (storage.toArray initial) := by
  induction length with
  | zero => rfl
  | succ length inductionHypothesis =>
      rw [Fin.foldl_succ_last, Fin.foldl_succ_last,
        storage.toArray_push, inductionHypothesis]

/-- Pushing finite-indexed values in order constructs `Array.ofFn`. -/
theorem fin_foldl_push_eq_array_ofFn
    {α : Type u} (length : Nat) (values : Fin length → α) :
    Fin.foldl length
        (fun output index => output.push (values index))
        (Array.emptyWithCapacity length) =
      Array.ofFn values := by
  induction length with
  | zero => simp
  | succ length inductionHypothesis =>
      rw [Fin.foldl_succ_last, Array.ofFn_succ]
      congr 1
      simpa using
        inductionHypothesis (fun index => values index.castSucc)

/--
Observing finite pushes from any reserved capacity gives the corresponding
ordinary function array.
-/
theorem toArray_finFoldl_push_eq_array_ofFn
    {α : Type u} [storage : Storage α]
    {length : Nat} (capacity : Nat) (values : Fin length → α) :
    storage.toArray
        (Fin.foldl length
          (fun output index => storage.push output (values index))
          (storage.emptyWithCapacity capacity)) =
      Array.ofFn values := by
  rw [toArray_finFoldl_push, storage.toArray_emptyWithCapacity,
    Array.emptyWithCapacity_eq]
  simpa only [Array.emptyWithCapacity_eq] using
    fin_foldl_push_eq_array_ofFn length values

/-- Finite construction observes as the corresponding ordinary array. -/
theorem toArray_ofFn {α : Type u} [storage : Storage α]
    {length : Nat} (values : Fin length → α) :
    storage.toArray (ofFn values) = Array.ofFn values := by
  rw [ofFn, toArray_finFoldl_push,
    storage.toArray_emptyWithCapacity,
    fin_foldl_push_eq_array_ofFn]

/-- Finite construction creates exactly the requested number of scalars. -/
theorem size_ofFn {α : Type u} [storage : Storage α]
    {length : Nat} (values : Fin length → α) :
    storage.size (ofFn values) = length := by
  rw [← storage.toArray_size, toArray_ofFn, Array.size_ofFn]

/-- An `Array.ofFn` observation determines the physical buffer size. -/
theorem size_eq_of_toArray_eq_ofFn
    {α : Type u} [storage : Storage α] {length : Nat}
    (buffer : storage.Buffer) (values : Fin length → α)
    (hData : storage.toArray buffer = Array.ofFn values) :
    storage.size buffer = length := by
  rw [← storage.toArray_size, hData, Array.size_ofFn]

/-- Empty physical storage contains no scalar entries. -/
theorem size_emptyWithCapacity
    {α : Type u} [storage : Storage α] (capacity : Nat) :
    storage.size (storage.emptyWithCapacity capacity) = 0 := by
  rw [← storage.toArray_size, storage.toArray_emptyWithCapacity]
  rfl

/-- Appending one scalar increases physical storage length by one. -/
theorem size_push
    {α : Type u} [storage : Storage α]
    (buffer : storage.Buffer) (value : α) :
    storage.size (storage.push buffer value) = storage.size buffer + 1 := by
  rw [← storage.toArray_size, storage.toArray_push, Array.size_push,
    storage.toArray_size]

/-- Constructing physical storage from an array preserves its length. -/
theorem size_ofArray
    {α : Type u} [storage : Storage α] (values : Array α) :
    storage.size (storage.ofArray values) = values.size := by
  rw [← storage.toArray_size, storage.toArray_ofArray]

end Storage

/-- Append one contiguous ordinary-array interval to an output array. -/
@[inline] def appendArraySlice {α : Type u}
    (source : Array α) (start stop : Nat) (output : Array α) : Array α :=
  source.foldl (fun result value => result.push value) output start stop

/-- Ordinary-array slice traversal is extraction followed by append. -/
theorem appendArraySlice_eq_append_extract {α : Type u}
    (source : Array α) (start stop : Nat) (output : Array α) :
    appendArraySlice source start stop output =
      output ++ source.extract start stop := by
  rw [appendArraySlice, Array.foldl_eq_foldl_extract]
  change
    (source.extract start stop).foldl
        (fun result value => result.push (id value)) output =
      output ++ source.extract start stop
  rw [Array.foldl_push_eq_append
    (as := source.extract start stop) (bs := output) (f := id) rfl]
  exact congrArg (output ++ ·) (Array.map_id _)

/--
Copy one ordinary-array entry without exposing its scalar at the caller
boundary.

Specializing a scalar-valued callback can unbox and immediately rebox values
such as `Float32`. Keeping the transfer inside this polymorphic primitive lets
Lean's runtime retain and move the existing `lean_object*` instead.
-/
@[inline, specialize] def copyArrayAt {α : Type u}
    (source : Array α) (index : USize)
    (hIndex : index.toNat < source.size)
    (output : Array α) : Array α :=
  output.push (source.uget index hIndex)

/-- Gather ordinary-array entries without exposing their scalar values. -/
@[specialize] def gatherArray {α : Type u}
    (source : Array α) (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (sourceIndices :
      (index : USize) → index.toNat < length → USize)
    (hSourceIndices :
      ∀ (index : USize) (hIndex : index.toNat < length),
        (sourceIndices index hIndex).toNat < source.size) :
    Array α :=
  Tensor.Internal.Elab.Impl.nativeFinFoldl length bound hBound
    (fun output index hIndex =>
      copyArrayAt source (sourceIndices index hIndex)
        (hSourceIndices index hIndex) output)
    (Array.emptyWithCapacity length)

/-- Ordinary polymorphic arrays are the universal tensor-storage fallback. -/
@[inline, instance_reducible]
instance (priority := low) instArrayStorage (α : Type u) :
    Storage α where
  Buffer := Array α
  emptyWithCapacity := Array.emptyWithCapacity
  push := Array.push
  copyAt := copyArrayAt
  gather := gatherArray
  gather_eq_nativeFinFoldl := by intros; rfl
  appendSlice := appendArraySlice
  foldl := fun step initial buffer => buffer.foldl step initial
  size := Array.size
  get := fun buffer index hIndex => buffer[index]'hIndex
  uget := Array.uget
  toArray := id
  ofArray := id
  toArray_size := by intros; rfl
  toArray_get := by intros; rfl
  toArray_uget := by intros; rfl
  toArray_emptyWithCapacity := by intros; rfl
  toArray_push := by intros; rfl
  toArray_copyAt := by intros; rfl
  toArray_appendSlice := appendArraySlice_eq_append_extract
  toArray_foldl := by intros; rfl
  toArray_ofArray := by intros; rfl
  toArray_injective := Function.injective_id

namespace Storage.Internal

/--
Proof-visible model of the native empty `ByteArray` allocator.

Compiled code calls Lean's built-in scalar-array runtime primitive.
-/
@[implemented_by ByteArray.emptyWithCapacity]
def byteBufferEmptyWithCapacity (capacity : @& Nat) : ByteArray :=
  ⟨Array.emptyWithCapacity capacity⟩

/-- Proof-visible model of native `ByteArray.push`. -/
@[implemented_by ByteArray.push]
def byteBufferPush : ByteArray → UInt8 → ByteArray
  | ⟨values⟩, value => ⟨values.push value⟩

/-- Proof-visible model of native `ByteArray.size`. -/
@[implemented_by ByteArray.size]
def byteBufferSize : (@& ByteArray) → Nat
  | ⟨values⟩ => values.size

/-- Fast native transfer between byte buffers. -/
@[inline] unsafe def byteBufferCopyAtFast
    (source : @& ByteArray) (index : USize)
    (_hIndex : index.toNat < byteBufferSize source)
    (output : ByteArray) : ByteArray :=
  output.push (source.uget index lcProof)

/-- Proof-visible model of one native byte-buffer transfer. -/
@[implemented_by byteBufferCopyAtFast]
def byteBufferCopyAt
    (source : @& ByteArray) (index : USize)
    (hIndex : index.toNat < byteBufferSize source)
    (output : ByteArray) : ByteArray :=
  ⟨output.data.push (source.data.uget index hIndex)⟩

/-- Runtime adapter from the proof-visible byte-buffer bound to `ByteArray.get`. -/
@[inline] unsafe def byteBufferGetFast
    (buffer : @& ByteArray) (index : @& Nat)
    (_hIndex : index < byteBufferSize buffer) : UInt8 :=
  buffer.get index lcProof

/-- Proof-visible model of native natural-number `ByteArray` lookup. -/
@[implemented_by byteBufferGetFast]
def byteBufferGet :
    (buffer : @& ByteArray) → (index : @& Nat) →
      index < byteBufferSize buffer → UInt8
  | ⟨values⟩, index, hIndex => values[index]'hIndex

/-- Fast native replacement in a packed byte buffer. -/
@[inline] unsafe def byteBufferSetFast
    (buffer : ByteArray) (index : @& Nat)
    (_hIndex : index < byteBufferSize buffer) (value : UInt8) : ByteArray :=
  buffer.set index value lcProof

/-- Proof-visible model of native byte-buffer replacement. -/
@[implemented_by byteBufferSetFast]
def byteBufferSet
    (buffer : ByteArray) (index : @& Nat)
    (hIndex : index < byteBufferSize buffer) (value : UInt8) : ByteArray :=
  ⟨buffer.data.set index value hIndex⟩

/-- Runtime adapter from the proof-visible byte-buffer bound to `ByteArray.uget`. -/
@[inline] unsafe def byteBufferUGetFast
    (buffer : @& ByteArray) (index : USize)
    (_hIndex : index.toNat < byteBufferSize buffer) : UInt8 :=
  buffer.uget index lcProof

/-- Proof-visible model of native `USize` `ByteArray` lookup. -/
@[implemented_by byteBufferUGetFast]
def byteBufferUGet :
    (buffer : @& ByteArray) → (index : USize) →
      index.toNat < byteBufferSize buffer → UInt8
  | ⟨values⟩, index, hIndex => values.uget index hIndex

/-- Gather native byte-buffer entries in one storage-specific loop. -/
@[inline] def byteBufferGather
    (source : @& ByteArray) (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (sourceIndices :
      (index : USize) → index.toNat < length → USize)
    (hSourceIndices :
      ∀ (index : USize) (hIndex : index.toNat < length),
        (sourceIndices index hIndex).toNat < byteBufferSize source) :
    ByteArray :=
  Tensor.Internal.Elab.Impl.nativeFinFoldl length bound hBound
    (fun output index hIndex =>
      byteBufferCopyAt source (sourceIndices index hIndex)
        (hSourceIndices index hIndex) output)
    (byteBufferEmptyWithCapacity length)

/--
Fast native implementation of contiguous byte-buffer copying.

Clamp before calling `ByteArray.copySlice`: its native allocator may reserve the requested length
before clipping the source interval. The model follows `Array.extract`, including huge naturals.
-/
@[inline] def byteBufferAppendSliceFast
    (source : ByteArray) (start stop : Nat)
    (output : ByteArray) : ByteArray :=
  let start := min start source.size
  let stop := min stop source.size
  source.copySlice start output output.size (stop - start) false

/--
Proof-visible model of contiguous native `ByteArray` traversal.

The logical body exposes ordinary array extraction. Native code delegates the
whole interval to Lean's built-in `ByteArray.copySlice` runtime primitive.
-/
@[implemented_by byteBufferAppendSliceFast]
def byteBufferAppendSlice
    (source : ByteArray) (start stop : Nat)
    (output : ByteArray) : ByteArray :=
  ⟨appendArraySlice source.data start stop output.data⟩

/-- Fast native traversal of an unboxed byte buffer. -/
@[inline] def byteBufferFoldlFast {β : Type u}
    (step : β → UInt8 → β) (initial : β) (buffer : ByteArray) : β :=
  buffer.foldl step initial

/--
Proof-visible model of native `ByteArray` traversal.

The logical body exposes the ordinary array fold. Native code traverses the
unboxed byte buffer directly.
-/
@[implemented_by byteBufferFoldlFast]
def byteBufferFoldl {β : Type u}
    (step : β → UInt8 → β) (initial : β) (buffer : ByteArray) : β :=
  buffer.data.foldl step initial

/--
Proof-visible model of the native empty `FloatArray` allocator.

Compiled code calls Lean's built-in scalar-array runtime primitive.
-/
@[implemented_by FloatArray.emptyWithCapacity]
def floatBufferEmptyWithCapacity (capacity : @& Nat) : FloatArray :=
  ⟨Array.emptyWithCapacity capacity⟩

/-- Proof-visible model of native `FloatArray.push`. -/
@[implemented_by FloatArray.push]
def floatBufferPush : FloatArray → Float → FloatArray
  | ⟨values⟩, value => ⟨values.push value⟩

/-- Proof-visible model of native `FloatArray.size`. -/
@[implemented_by FloatArray.size]
def floatBufferSize : (@& FloatArray) → Nat
  | ⟨values⟩ => values.size

/-- Fast native transfer between floating-point buffers. -/
@[inline] unsafe def floatBufferCopyAtFast
    (source : @& FloatArray) (index : USize)
    (_hIndex : index.toNat < floatBufferSize source)
    (output : FloatArray) : FloatArray :=
  output.push (source.uget index lcProof)

/-- Proof-visible model of one native floating-point-buffer transfer. -/
@[implemented_by floatBufferCopyAtFast]
def floatBufferCopyAt
    (source : @& FloatArray) (index : USize)
    (hIndex : index.toNat < floatBufferSize source)
    (output : FloatArray) : FloatArray :=
  ⟨output.data.push (source.data.uget index hIndex)⟩

/-- Runtime adapter from the proof-visible float-buffer bound to `FloatArray.get`. -/
@[inline] unsafe def floatBufferGetFast
    (buffer : @& FloatArray) (index : @& Nat)
    (_hIndex : index < floatBufferSize buffer) : Float :=
  buffer.get index lcProof

/-- Proof-visible model of native natural-number `FloatArray` lookup. -/
@[implemented_by floatBufferGetFast]
def floatBufferGet :
    (buffer : @& FloatArray) → (index : @& Nat) →
      index < floatBufferSize buffer → Float
  | ⟨values⟩, index, hIndex => values[index]'hIndex

/-- Runtime adapter from the proof-visible float-buffer bound to `FloatArray.uget`. -/
@[inline] unsafe def floatBufferUGetFast
    (buffer : @& FloatArray) (index : USize)
    (_hIndex : index.toNat < floatBufferSize buffer) : Float :=
  buffer.uget index lcProof

/-- Proof-visible model of native `USize` `FloatArray` lookup. -/
@[implemented_by floatBufferUGetFast]
def floatBufferUGet :
    (buffer : @& FloatArray) → (index : USize) →
      index.toNat < floatBufferSize buffer → Float
  | ⟨values⟩, index, hIndex => values.uget index hIndex

/-- Gather native floating-point-buffer entries in one storage-specific loop. -/
@[inline] def floatBufferGather
    (source : @& FloatArray) (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (sourceIndices :
      (index : USize) → index.toNat < length → USize)
    (hSourceIndices :
      ∀ (index : USize) (hIndex : index.toNat < length),
        (sourceIndices index hIndex).toNat < floatBufferSize source) :
    FloatArray :=
  Tensor.Internal.Elab.Impl.nativeFinFoldl length bound hBound
    (fun output index hIndex =>
      floatBufferCopyAt source (sourceIndices index hIndex)
        (hSourceIndices index hIndex) output)
    (floatBufferEmptyWithCapacity length)

/--
Proof-visible model of contiguous native `FloatArray` traversal.

The logical body exposes ordinary array extraction.
-/
def floatBufferAppendSlice
    (source : FloatArray) (start stop : Nat)
    (output : FloatArray) : FloatArray :=
  ⟨appendArraySlice source.data start stop output.data⟩

/--
Native implementation of contiguous packed floating-point copying.

The body calls the reference rather than restating it, so the `rfl` below cannot drift; compiled
code replaces the whole thing with `torchlean_float_array_append_slice`.
-/
@[extern "torchlean_float_array_append_slice"]
def floatBufferAppendSliceNative
    (source : @& FloatArray) (start stop : @& Nat)
    (output : FloatArray) : FloatArray :=
  floatBufferAppendSlice source start stop output

/--
Compile packed floating-point slice appends to one native bulk-copy call.

Lean evaluation and proofs continue to unfold `floatBufferAppendSlice`; only
generated code uses the external implementation.
-/
@[csimp] theorem floatBufferAppendSlice_eq_native :
    @floatBufferAppendSlice = @floatBufferAppendSliceNative := rfl

/-- Fast native traversal of an unboxed floating-point buffer. -/
@[inline] def floatBufferFoldlFast {β : Type u}
    (step : β → Float → β) (initial : β) (buffer : FloatArray) : β :=
  buffer.foldl step initial

/--
Proof-visible model of native `FloatArray` traversal.

The logical body exposes the ordinary array fold. Native code traverses the
unboxed floating-point buffer directly.
-/
@[implemented_by floatBufferFoldlFast]
def floatBufferFoldl {β : Type u}
    (step : β → Float → β) (initial : β) (buffer : FloatArray) : β :=
  buffer.data.foldl step initial

/-- Fast native replacement in an unboxed floating-point buffer. -/
@[inline] unsafe def floatBufferSetFast
    (buffer : FloatArray) (index : @& Nat)
    (_hIndex : index < floatBufferSize buffer) (value : Float) : FloatArray :=
  buffer.set index value lcProof

/-- Proof-visible model of native floating-point-buffer replacement. -/
@[implemented_by floatBufferSetFast]
def floatBufferSet
    (buffer : FloatArray) (index : @& Nat)
    (hIndex : index < floatBufferSize buffer) (value : Float) : FloatArray :=
  ⟨buffer.data.set index value hIndex⟩

end Storage.Internal

/-- `UInt8` tensors use Lean's packed native byte-array representation. -/
instance instUInt8Storage : Storage UInt8 where
  Buffer := ByteArray
  emptyWithCapacity := Storage.Internal.byteBufferEmptyWithCapacity
  push := Storage.Internal.byteBufferPush
  copyAt := Storage.Internal.byteBufferCopyAt
  gather := Storage.Internal.byteBufferGather
  gather_eq_nativeFinFoldl := by intros; rfl
  appendSlice := Storage.Internal.byteBufferAppendSlice
  foldl := Storage.Internal.byteBufferFoldl
  size := Storage.Internal.byteBufferSize
  get := Storage.Internal.byteBufferGet
  uget := Storage.Internal.byteBufferUGet
  toArray := fun ⟨values⟩ => values
  ofArray := fun values => ⟨values⟩
  toArray_size := by intros; rfl
  toArray_get := by intros; rfl
  toArray_uget := by intros; rfl
  toArray_emptyWithCapacity := by intros; rfl
  toArray_push := by intros; rfl
  toArray_copyAt := by intros; rfl
  toArray_appendSlice := by
    intro source start stop output
    cases source
    cases output
    exact appendArraySlice_eq_append_extract _ _ _ _
  toArray_foldl := by intros; rfl
  toArray_ofArray := by intros; rfl
  toArray_injective := by
    intro left right hData
    cases left
    cases right
    cases hData
    rfl

/-- `Float` tensors use Lean's unboxed native scalar-array representation. -/
instance instFloatStorage : Storage Float where
  Buffer := FloatArray
  emptyWithCapacity := Storage.Internal.floatBufferEmptyWithCapacity
  push := Storage.Internal.floatBufferPush
  copyAt := Storage.Internal.floatBufferCopyAt
  gather := Storage.Internal.floatBufferGather
  gather_eq_nativeFinFoldl := by intros; rfl
  appendSlice := Storage.Internal.floatBufferAppendSlice
  foldl := Storage.Internal.floatBufferFoldl
  size := Storage.Internal.floatBufferSize
  get := Storage.Internal.floatBufferGet
  uget := Storage.Internal.floatBufferUGet
  toArray := fun ⟨values⟩ => values
  ofArray := fun values => ⟨values⟩
  toArray_size := by intros; rfl
  toArray_get := by intros; rfl
  toArray_uget := by intros; rfl
  toArray_emptyWithCapacity := by intros; rfl
  toArray_push := by intros; rfl
  toArray_copyAt := by intros; rfl
  toArray_appendSlice := by
    intro source start stop output
    cases source
    cases output
    exact appendArraySlice_eq_append_extract _ _ _ _
  toArray_foldl := by intros; rfl
  toArray_ofArray := by intros; rfl
  toArray_injective := by
    intro left right hData
    cases left
    cases right
    cases hData
    rfl

/-- Byte tensors replace entries directly in their packed native buffer. -/
instance instUInt8StorageUpdate :
    @Storage.Update UInt8 instUInt8Storage where
  set buffer index hBuffer value :=
    Storage.Internal.byteBufferSet buffer index hBuffer value
  toArray_set := by
    intro buffer index hBuffer value
    cases buffer
    rfl

/-- Float tensors replace entries directly in their unboxed native buffer. -/
instance instFloatStorageUpdate :
    @Storage.Update Float instFloatStorage where
  set buffer index hBuffer value :=
    Storage.Internal.floatBufferSet buffer index hBuffer value
  toArray_set := by
    intro buffer index hBuffer value
    cases buffer
    rfl

/-
Native Float kernels reason through the exported `Storage` observation
laws, not by repeatedly normalizing the proof-visible extern models.
-/
attribute [irreducible]
  Storage.Internal.floatBufferEmptyWithCapacity
  Storage.Internal.floatBufferPush
  Storage.Internal.floatBufferSize
  Storage.Internal.floatBufferGet
  Storage.Internal.floatBufferUGet
  Storage.Internal.floatBufferSet

namespace Storage

end Storage

end TorchLean
