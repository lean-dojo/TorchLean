/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Batteries.Data.Fin.Fold
public import Mathlib.Data.List.OfFn -- shake: keep
import all Init.Data.FloatArray.Basic

/-!
# Physical Tensor Storage

`Storage α` selects the physical buffer used by tensors with scalar type `α`. Arbitrary scalar
types keep ordinary `Array α` storage, `Float` uses Lean's unboxed `FloatArray` runtime
representation and `UInt8` uses `ByteArray`.

The `toArray` laws are the proof boundary: proofs reason about an ordinary array observation,
while compiled code calls the specialized buffer operations. The `Float` instance stores core's
`FloatArray` operations and proves its laws from core's definitions, which the private
`import all Init.Data.FloatArray.Basic` makes visible in this module. Selected pointwise, slice and
transpose operations also have proof-visible Lean definitions with native compiled replacements.
The Lean theorems describe those definitions, and agreement of the external C bodies with them is
a native runtime boundary documented in `docs/TRUST_BOUNDARIES.md`.
-/

@[expose] public section

namespace TorchLean

universe u

/--
Run the tail-recursive loop behind `finFoldl` from `index` up to `bound`.

`hBound` and the callback's index proof are erased, so the compiled loop carries only the
accumulator and the unboxed bound and counter.
-/
@[specialize] def Native.finFoldlLoop
    {α : Type u} (length : Nat)
    (step : α → (index : USize) → index.toNat < length → α)
    (bound : USize) (hBound : bound.toNat = length)
    (index : USize) (value : α) : α :=
  if hIndex : index < bound then
    finFoldlLoop length step bound hBound (index + 1)
      (step value index (by
        have hIndexNat := USize.lt_iff_toNat_lt.mp hIndex
        lia))
  else
    value
termination_by length - index.toNat
decreasing_by
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
@[inline] def Native.finFoldl
    {α : Type u} (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (step : α → (index : USize) → index.toNat < length → α)
    (initial : α) : α :=
  finFoldlLoop length step bound hBound 0 initial

/--
The loop started at an `index` with `index.toNat ≤ length` folds the suffix of
`List.finRange length` that begins at position `index.toNat`.
-/
private theorem Native.finFoldl_loop_eq_finRange
    {α : Type u} (length : Nat)
    (step : α → (index : USize) → index.toNat < length → α)
    (bound : USize) (hBound : bound.toNat = length)
    (index : USize) (value : α)
    (hIndex : index.toNat ≤ length) :
    finFoldlLoop length step bound hBound index value =
      ((List.finRange length).drop index.toNat).foldl
        (fun total coordinate =>
          step total
            (USize.ofNatLT coordinate.val
              (Nat.lt_trans coordinate.isLt (by
                rw [← hBound]
                exact USize.toNat_lt_size bound)))
            (by simp))
        value := by
  refine finFoldlLoop.induct length step bound hBound
    (motive := fun index value =>
      ∀ hIndex : index.toNat ≤ length,
        finFoldlLoop length step bound hBound index value =
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
    rw [finFoldlLoop.eq_1, dite_eq_left hIndexLt]
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
    rw [finFoldlLoop.eq_1, dite_eq_right hIndexNotLt]
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
`finFoldl` with the callback `step value ⟨index.toNat, hIndex⟩` equals `Fin.foldl`.

Generated kernels run with an unboxed counter, and proofs reason about the standard finite
enumeration.
-/
theorem Native.finFoldl_eq_fin_foldl
    {α : Type u} (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (step : α → Fin length → α) (initial : α) :
    finFoldl length bound hBound
        (fun value index hIndex =>
          step value ⟨index.toNat, hIndex⟩)
        initial =
      Fin.foldl length step initial := by
  change finFoldlLoop length
    (fun value index hIndex =>
      step value ⟨index.toNat, hIndex⟩)
    bound hBound 0 initial = _
  calc
    _ = (List.finRange length).foldl step initial := by
      simpa using finFoldl_loop_eq_finRange length
        (fun value index hIndex =>
          step value ⟨index.toNat, hIndex⟩)
        bound hBound (0 : USize) initial (by simp)
    _ = _ := (Fin.foldl_eq_foldl_finRange _ _).symm

/--
A native callback computes the semantic finite fold when it agrees with the
semantic callback at every represented index.
-/
theorem Native.finFoldl_eq_fin_foldl_of_eq
    {α : Type u} (length : Nat) (bound : USize)
    (hBound : bound.toNat = length)
    (nativeStep : α → (index : USize) → index.toNat < length → α)
    (step : α → Fin length → α) (initial : α)
    (hStep :
      ∀ value index hIndex,
        nativeStep value index hIndex =
          step value ⟨index.toNat, hIndex⟩) :
    finFoldl length bound hBound nativeStep initial =
      Fin.foldl length step initial := by
  rw [show nativeStep =
      (fun value index hIndex =>
        step value ⟨index.toNat, hIndex⟩) by
    funext value index hIndex
    exact hStep value index hIndex]
  exact finFoldl_eq_fin_foldl length bound hBound step initial

/-- The quotient/remainder transpose address is inside the source rectangle. -/
theorem Native.transpose2DIndexValue_lt
    (rows columns index : Nat) (hIndex : index < columns * rows) :
    index / rows + columns * (index % rows) < rows * columns := by
  have hRows : 0 < rows := Nat.pos_of_ne_zero fun hZero => by simp [hZero] at hIndex
  have hQuotient : index / rows < columns :=
    Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at hIndex)
  have hRemainder : index % rows + 1 ≤ rows := Nat.mod_lt index hRows
  calc index / rows + columns * (index % rows)
      < columns + columns * (index % rows) := Nat.add_lt_add_right hQuotient _
    _ = columns * (index % rows + 1) := by rw [Nat.mul_succ, Nat.add_comm]
    _ ≤ columns * rows := Nat.mul_le_mul_left columns hRemainder
    _ = rows * columns := Nat.mul_comm columns rows

/--
Transpose a row-major `rows × columns` array into row-major `columns × rows` order.

Compiled code calls the tiled C loop `torchlean_array_transpose2d`, which moves entry pointers.
-/
@[extern "torchlean_array_transpose2d"]
def Native.arrayBufferTranspose2D
    {α : Type u} (source : Array α) (rows columns : @& Nat)
    (hSize : source.size = rows * columns) : Array α :=
  Array.ofFn fun outputIndex : Fin (columns * rows) =>
    source[outputIndex.val / rows + columns * (outputIndex.val % rows)]'(by
      rw [hSize]
      exact transpose2DIndexValue_lt rows columns outputIndex.val outputIndex.isLt)

/-- The ordinary-array transpose has exactly the transposed scalar count. -/
theorem Native.arrayBufferTranspose2D_size
    {α : Type u} (source : Array α) (rows columns : Nat)
    (hSize : source.size = rows * columns) :
    (arrayBufferTranspose2D source rows columns hSize).size =
      columns * rows := by
  simp [arrayBufferTranspose2D]

/--
Transpose a packed row-major `rows × columns` buffer into row-major `columns × rows` order.

Compiled code calls the tiled C loop `torchlean_float_array_transpose2d`.
-/
@[extern "torchlean_float_array_transpose2d"]
def Native.floatBufferTranspose2D
    (source : @& FloatArray) (rows columns : @& Nat)
    (hSize : source.data.size = rows * columns) : FloatArray :=
  ⟨arrayBufferTranspose2D source.data rows columns hSize⟩

/-- The platform-native source index that a row-major transpose reads for one output index. -/
def Einops.Elab.transpose2DSourceIndex (rows columns : Nat) (hFits : columns * rows < USize.size)
    (index : USize) (hIndex : index.toNat < columns * rows) : USize :=
  USize.ofNatLT (index.toNat / rows + columns * (index.toNat % rows)) (by
    have hValue := Native.transpose2DIndexValue_lt rows columns index.toNat hIndex
    rw [Nat.mul_comm rows columns] at hValue
    exact Nat.lt_trans hValue hFits)

/-- A transpose source index addresses an entry of the `rows × columns` source. -/
theorem Einops.Elab.transpose2DSourceIndex_lt (rows columns : Nat)
    (hFits : columns * rows < USize.size) (index : USize) (hIndex : index.toNat < columns * rows) :
    (transpose2DSourceIndex rows columns hFits index hIndex).toNat < rows * columns := by
  rw [transpose2DSourceIndex, USize.toNat_ofNatLT]
  exact Native.transpose2DIndexValue_lt rows columns index.toNat hIndex

end TorchLean

namespace TorchLean

universe u

/--
Physical tensor storage for one scalar type.

Generated native kernels call the buffer operations, and proofs observe them through `toArray`.
Evaluating `toArray` may allocate, so optimized code keeps to `Buffer`, `get`, `uget`, `push` and
`appendSlice`.
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
  Append one physical entry of a source buffer to an output buffer.

  The result equals `push output (uget source index ...)`, but the scalar never reaches the
  caller, so generic arrays move the existing boxed object and scalar arrays copy the unboxed
  value.
  -/
  copyAt : (source : Buffer) → (index : USize) →
    index.toNat < size source → Buffer → Buffer
  /--
  Build a buffer by gathering physical entries from one source buffer.

  The storage implementation owns the whole loop, so backend selection happens once per tensor.
  -/
  gather : (source : Buffer) → (length : Nat) → (bound : USize) →
    bound.toNat = length →
    (sourceIndices : (index : USize) → index.toNat < length → USize) →
    (∀ (index : USize) (hIndex : index.toNat < length),
      (sourceIndices index hIndex).toNat < size source) →
    Buffer :=
    fun source length bound hBound sourceIndices hSourceIndices =>
      Native.finFoldl length bound hBound
        (fun output index hIndex =>
          copyAt source (sourceIndices index hIndex)
            (hSourceIndices index hIndex) output)
        (emptyWithCapacity length)
  /-- Gathering is the `finFoldl` loop that appends each selected entry with `copyAt`. -/
  gather_eq_finFoldl (source : Buffer)
      (length : Nat) (bound : USize)
      (hBound : bound.toNat = length)
      (sourceIndices :
        (index : USize) → index.toNat < length → USize)
      (hSourceIndices :
        ∀ (index : USize) (hIndex : index.toNat < length),
          (sourceIndices index hIndex).toNat < size source) :
    gather source length bound hBound sourceIndices hSourceIndices =
      Native.finFoldl length bound hBound
        (fun output index hIndex =>
          copyAt source (sourceIndices index hIndex)
            (hSourceIndices index hIndex) output)
        (emptyWithCapacity length) := by
    intros; rfl
  /--
  Transpose a row-major `rows × columns` buffer into row-major `columns × rows` order.

  The default gathers every output entry from its source position.
  -/
  transpose2D : (source : Buffer) → (rows columns : Nat) → (bound : USize) →
    bound.toNat = columns * rows → size source = rows * columns → Buffer :=
    fun source rows columns bound hBound hSize =>
      gather source (columns * rows) bound hBound
        (Einops.Elab.transpose2DSourceIndex rows columns
          (hBound ▸ USize.toNat_lt_size bound))
        (fun index hIndex => by
          rw [hSize]
          exact Einops.Elab.transpose2DSourceIndex_lt rows columns _ index hIndex)
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
    (toArray buffer).size = size buffer := by intros; rfl
  /-- Buffer observation preserves natural-number reads. -/
  toArray_get (buffer : Buffer) (index : Nat)
      (hBuffer : index < size buffer)
      (hArray : index < (toArray buffer).size) :
    (toArray buffer)[index]'hArray = get buffer index hBuffer := by intros; rfl
  /-- Buffer observation preserves platform-native reads. -/
  toArray_uget (buffer : Buffer) (index : USize)
      (hBuffer : index.toNat < size buffer)
      (hArray : index.toNat < (toArray buffer).size) :
    (toArray buffer).uget index hArray = uget buffer index hBuffer := by intros; rfl
  /-- An empty physical buffer observes as an empty ordinary array. -/
  toArray_emptyWithCapacity (capacity : Nat) :
    toArray (emptyWithCapacity capacity) = Array.emptyWithCapacity capacity := by intros; rfl
  /-- Appending one scalar commutes with buffer observation. -/
  toArray_push (buffer : Buffer) (value : α) :
    toArray (push buffer value) = (toArray buffer).push value := by intros; rfl
  /-- Copying one physical entry commutes with buffer observation. -/
  toArray_copyAt (source : Buffer) (index : USize)
      (hBuffer : index.toNat < size source) (output : Buffer)
      (hArray : index.toNat < (toArray source).size) :
    toArray (copyAt source index hBuffer output) =
      (toArray output).push ((toArray source).uget index hArray) := by intros; rfl
  /-- Transposing a buffer commutes with buffer observation. -/
  toArray_transpose2D (source : Buffer) (rows columns : Nat) (bound : USize)
      (hBound : bound.toNat = columns * rows) (hSize : size source = rows * columns) :
    toArray (transpose2D source rows columns bound hBound hSize) =
      Native.arrayBufferTranspose2D (toArray source) rows columns
        ((toArray_size source).trans hSize) := by
    intro source rows columns bound hBound hSize
    symm
    exact Storage.arrayBufferTranspose2D_eq_finFoldl _ source rows columns
      bound hBound _ _ _ (by intros; rfl) (by rfl)
  /-- Appending a source interval commutes with buffer observation. -/
  toArray_appendSlice (source : Buffer) (start stop : Nat)
      (output : Buffer) :
    toArray (appendSlice source start stop output) =
      toArray output ++ (toArray source).extract start stop
  /-- Row-major traversal agrees with folding the ordinary array observation. -/
  toArray_foldl {β : Type u} (step : β → α → β)
      (initial : β) (buffer : Buffer) :
    foldl step initial buffer = (toArray buffer).foldl step initial := by intros; rfl
  /-- Converting from an ordinary array is inverse to observation. -/
  toArray_ofArray (values : Array α) :
    toArray (ofArray values) = values := by intros; rfl
  /-- The ordinary array observation uniquely determines physical storage. -/
  toArray_injective : Function.Injective toArray
  /--
  Optional decidable propositional equality on Lean's carrier for checked runtime optimizations.

  It is separate from numerical `BEq`. Lean's floating-point carrier equality distinguishes signed
  zeros and identifies every NaN regardless of sign or payload, and optimizations preserve it.
  Backends without a decision procedure use the uncompressed implementation of operations that
  need one.
  -/
  decEq? : Option (DecidableEq α) := none

namespace Storage

/--
Copy-on-write replacement of one physical scalar.

A custom `Storage` need not implement it: the low-priority `instUpdate` covers every storage, and
packed backends override it to keep their native representation.
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
Universal update support, by a round trip through the ordinary array observation.
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
If `uget source index` reads `value`, copying that entry onto `output` equals pushing `value`.

Generated movement kernels reason about `copyAt` through this theorem, while their executable
callbacks never handle the scalar.
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

/-- A natural-number read is the entry of the ordinary array observation. -/
theorem get_eq_toArray_getElem {α : Type u} [storage : Storage α] (buffer : storage.Buffer)
    (index : Nat) (hIndex : index < storage.size buffer) :
    storage.get buffer index hIndex =
      (storage.toArray buffer)[index]'(by rw [storage.toArray_size]; exact hIndex) :=
  (storage.toArray_get buffer index hIndex _).symm

/-- A platform-native read is the native read of the ordinary array observation. -/
theorem uget_eq_toArray_uget {α : Type u} [storage : Storage α] (buffer : storage.Buffer)
    (index : USize) (hIndex : index.toNat < storage.size buffer) :
    storage.uget buffer index hIndex =
      (storage.toArray buffer).uget index (by rw [storage.toArray_size]; exact hIndex) :=
  (storage.toArray_uget buffer index hIndex _).symm

/-- Copying a physical entry pushes the entry read from the source. -/
theorem copyAt_eq_push_uget {α : Type u} [storage : Storage α] (source : storage.Buffer)
    (index : USize) (hIndex : index.toNat < storage.size source) (output : storage.Buffer) :
    storage.copyAt source index hIndex output =
      storage.push output (storage.uget source index hIndex) :=
  copyAt_eq_push_of_uget_eq source index hIndex output _ rfl

/--
Construct a physical buffer from a finite scalar function.
-/
def ofFn {α : Type u} [storage : Storage α] {length : Nat}
    (values : Fin length → α) : storage.Buffer :=
  Fin.foldl length
    (fun output index => storage.push output (values index))
    (storage.emptyWithCapacity length)

/-- Observing a finite physical-buffer fold commutes with scalar appends. -/
private theorem toArray_fin_foldl_push
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
theorem fin_foldl_push_eq_ofFn
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
A native fold whose step appends the transposed source entry observes as the array transpose.
-/
theorem arrayBufferTranspose2D_eq_finFoldl {α Buffer : Type u}
    (toArray : Buffer → Array α) (source : Buffer) (rows columns : Nat) (bound : USize)
    (hBound : bound.toNat = columns * rows) (hSize : (toArray source).size = rows * columns)
    (step : Buffer → (index : USize) → index.toNat < columns * rows → Buffer)
    (initial : Buffer)
    (hStep : ∀ output index hIndex,
      toArray (step output index hIndex) =
        (toArray output).push ((toArray source).uget
          (Einops.Elab.transpose2DSourceIndex rows columns
            (hBound ▸ USize.toNat_lt_size bound) index hIndex)
          (by
            rw [hSize]
            exact Einops.Elab.transpose2DSourceIndex_lt rows columns _ index
              hIndex)))
    (hInitial : toArray initial = Array.emptyWithCapacity (columns * rows)) :
    Native.arrayBufferTranspose2D (toArray source) rows columns hSize =
      toArray (Native.finFoldl (columns * rows) bound hBound
        step initial) := by
  have hFits : columns * rows < USize.size := hBound ▸ USize.toNat_lt_size bound
  let entry : Fin (columns * rows) → α := fun index =>
    (toArray source)[index.val / rows + columns * (index.val % rows)]'(by
      rw [hSize]
      exact Native.transpose2DIndexValue_lt rows columns index.val
        index.isLt)
  let semanticStep : Buffer → Fin (columns * rows) → Buffer := fun output index =>
    step output (USize.ofNatLT index.val (Nat.lt_trans index.isLt hFits))
      (by rw [USize.toNat_ofNatLT]; exact index.isLt)
  have hSemanticStep : ∀ output index,
      (toArray output).push (entry index) = toArray (semanticStep output index) := by
    intro output index
    rw [hStep]
    rfl
  rw [Native.finFoldl_eq_fin_foldl_of_eq (columns * rows) bound hBound
      step semanticStep initial (by
        intro output index hIndex
        simp only [semanticStep, USize.ofNatLT_toNat]),
    Fin.foldl_eq_foldl_finRange,
    ← List.foldl_hom toArray (g₁ := semanticStep)
      (g₂ := fun output index => output.push (entry index)) hSemanticStep,
    hInitial, ← Fin.foldl_eq_foldl_finRange, fin_foldl_push_eq_ofFn]
  rfl

/--
Observing finite pushes from any reserved capacity gives the corresponding
ordinary function array.
-/
theorem toArray_fin_foldl_push_eq_ofFn
    {α : Type u} [storage : Storage α]
    {length : Nat} (capacity : Nat) (values : Fin length → α) :
    storage.toArray
        (Fin.foldl length
          (fun output index => storage.push output (values index))
          (storage.emptyWithCapacity capacity)) =
      Array.ofFn values := by
  rw [toArray_fin_foldl_push, storage.toArray_emptyWithCapacity,
    Array.emptyWithCapacity_eq]
  simpa only [Array.emptyWithCapacity_eq] using
    fin_foldl_push_eq_ofFn length values

/-- Finite construction observes as the corresponding ordinary array. -/
theorem toArray_ofFn {α : Type u} [storage : Storage α]
    {length : Nat} (values : Fin length → α) :
    storage.toArray (ofFn values) = Array.ofFn values := by
  exact toArray_fin_foldl_push_eq_ofFn length values

/-- A buffer built by `ofFn` has `length` entries. -/
theorem size_ofFn {α : Type u} [storage : Storage α]
    {length : Nat} (values : Fin length → α) :
    storage.size (ofFn values) = length := by
  rw [← storage.toArray_size, toArray_ofFn, Array.size_ofFn]

/-- Reading a buffer built from a finite function returns the indexed value. -/
@[simp] theorem get_ofFn {α : Type u} [storage : Storage α]
    {length : Nat} (values : Fin length → α) (index : Fin length)
    (hIndex : index.val < storage.size (ofFn values)) :
    storage.get (ofFn values) index.val hIndex = values index := by
  have hArray : index.val < (storage.toArray (ofFn values)).size := by
    simpa only [toArray_ofFn, Array.size_ofFn] using index.isLt
  have hGet := storage.toArray_get (ofFn values) index.val hIndex hArray
  simpa only [toArray_ofFn, Array.getElem_ofFn] using hGet.symm

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
@[inline] def Storage.appendArraySlice {α : Type u}
    (source : Array α) (start stop : Nat) (output : Array α) : Array α :=
  source.foldl (fun result value => result.push value) output start stop

/-- Ordinary-array slice traversal is extraction followed by append. -/
theorem Storage.appendArraySlice_eq_extract {α : Type u}
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
Lean's runtime retain and move the existing `lean_object*`.
-/
@[inline, specialize] def Storage.copyArrayAt {α : Type u}
    (source : Array α) (index : USize)
    (hIndex : index.toNat < source.size)
    (output : Array α) : Array α :=
  output.push (source.uget index hIndex)

/-- Ordinary polymorphic arrays are the universal tensor-storage fallback. -/
@[inline, instance_reducible]
instance (priority := low) instArrayStorage (α : Type u) :
    Storage α where
  Buffer := Array α
  emptyWithCapacity := Array.emptyWithCapacity
  push := Array.push
  copyAt := Storage.copyArrayAt
  transpose2D := fun source rows columns _ _ hSize =>
    Native.arrayBufferTranspose2D source rows columns hSize
  appendSlice := Storage.appendArraySlice
  foldl := fun step initial buffer => buffer.foldl step initial
  size := Array.size
  get := fun buffer index hIndex => buffer[index]'hIndex
  uget := Array.uget
  toArray := id
  ofArray := id
  toArray_appendSlice := Storage.appendArraySlice_eq_extract
  toArray_transpose2D := by intros; rfl
  toArray_injective := Function.injective_id

/-- Float32 retains ordinary array storage and exposes Lean's propositional carrier equality. -/
instance instFloat32Storage : Storage Float32 :=
  { instArrayStorage Float32 with decEq? := some inferInstance }

namespace Storage

/--
Append `source.data.extract start stop` to `output` with one `ByteArray.copySlice` call on `start`
and `stop` clamped to `source.size`, as `data_byteBufferAppendSlice` proves for all naturals.

Both bounds are clamped before `ByteArray.copySlice` runs because its native allocator may reserve
the requested length before clipping the source interval.
-/
@[inline] def byteBufferAppendSlice
    (source : ByteArray) (start stop : Nat)
    (output : ByteArray) : ByteArray :=
  let start := min start source.size
  let stop := min stop source.size
  source.copySlice start output output.size (stop - start) false

/--
The data of `byteBufferAppendSlice source start stop output` is
`output.data ++ source.data.extract start stop`, for all naturals `start` and `stop`.
-/
theorem data_byteBufferAppendSlice
    (source : ByteArray) (start stop : Nat) (output : ByteArray) :
    (byteBufferAppendSlice source start stop output).data =
      output.data ++ source.data.extract start stop := by
  simp only [byteBufferAppendSlice, ByteArray.copySlice, ← ByteArray.size_data,
    Array.extract_size]
  rw [Array.extract_eq_empty_of_le ((Nat.min_le_left _ _).trans (Nat.le_add_right _ _)),
    Array.append_empty]
  congr 1
  apply Array.ext
  · simp only [Array.size_extract]
    omega
  · intro k h₁ h₂
    simp only [Array.getElem_extract]
    congr 1
    simp only [Array.size_extract] at h₁ h₂
    omega

universe v

/--
A monadic left fold over a `ByteArray` from `start` to `stop` is the same fold over its `data`
array, in every monad.
-/
theorem _root_.TorchLean.ByteArray.foldlM_eq_foldlM_data {β : Type u}
    {m : Type u → Type v} [Monad m]
    {f : β → UInt8 → m β} {init : β} {as : ByteArray} {start stop : Nat} :
    as.foldlM f init start stop = as.data.foldlM f init start stop := by
  have loop (bound : Nat) (h₁ : bound ≤ as.size) (h₂ : bound ≤ as.data.size) (i j : Nat)
      (b : β) :
      ByteArray.foldlM.loop f as bound h₁ i j b = Array.foldlM.loop f as.data bound h₂ i j b := by
    induction i generalizing j b with
    | zero => unfold ByteArray.foldlM.loop Array.foldlM.loop; rfl
    | succ i ih =>
      unfold ByteArray.foldlM.loop Array.foldlM.loop
      by_cases hlt : j < bound
      · simp only [hlt, ↓reduceDIte, ih]
        rfl
      · simp only [hlt, ↓reduceDIte]
  unfold ByteArray.foldlM Array.foldlM
  by_cases hstop : stop ≤ as.size
  · simp only [ByteArray.size_data, hstop, ↓reduceDIte]
    exact loop ..
  · simp only [ByteArray.size_data, hstop, ↓reduceDIte]
    exact loop ..

/--
A monadic left fold over a `FloatArray` from `start` to `stop` is the same fold over its `data`
array, in every monad.
-/
theorem _root_.TorchLean.FloatArray.foldlM_eq_foldlM_data {β : Type u}
    {m : Type u → Type v} [Monad m]
    {f : β → Float → m β} {init : β} {as : FloatArray} {start stop : Nat} :
    as.foldlM f init start stop = as.data.foldlM f init start stop := by
  have loop (bound : Nat) (h₁ : bound ≤ as.size) (h₂ : bound ≤ as.data.size) (i j : Nat)
      (b : β) :
      FloatArray.foldlM.loop f as bound h₁ i j b = Array.foldlM.loop f as.data bound h₂ i j b := by
    induction i generalizing j b with
    | zero => unfold FloatArray.foldlM.loop Array.foldlM.loop; rfl
    | succ i ih =>
      unfold FloatArray.foldlM.loop Array.foldlM.loop
      by_cases hlt : j < bound
      · simp only [hlt, ↓reduceDIte, ih]
        rfl
      · simp only [hlt, ↓reduceDIte]
  have hsize : as.data.size = as.size := rfl
  unfold FloatArray.foldlM Array.foldlM
  by_cases hstop : stop ≤ as.size
  · simp only [hsize, hstop, ↓reduceDIte]
    exact loop ..
  · simp only [hsize, hstop, ↓reduceDIte]
    exact loop ..

/--
Append the entries of `source` from `start` to `stop`, clamped as by `Array.extract`, to `output`.

Compiled code calls the C bulk copy `torchlean_float_array_append_slice`, whose agreement with
this body is a trust boundary that `TorchLeanTest/Tensor/Storage.lean` tests for order, aliasing
and big-`Nat` bounds.
-/
@[extern "torchlean_float_array_append_slice"]
def floatBufferAppendSlice
    (source : @& FloatArray) (start stop : @& Nat)
    (output : FloatArray) : FloatArray :=
  ⟨appendArraySlice source.data start stop output.data⟩

end Storage

/-- `UInt8` tensors use Lean's packed native byte-array representation. -/
instance instUInt8Storage : Storage UInt8 where
  decEq? := some inferInstance
  Buffer := ByteArray
  emptyWithCapacity := ByteArray.emptyWithCapacity
  push := ByteArray.push
  copyAt source index hIndex output := output.push (source.uget index hIndex)
  -- We unfold `copyAt` in the class default, which saves the compiled instance one closure call
  -- per gathered entry.
  gather source length bound hBound sourceIndices hSourceIndices :=
    Native.finFoldl length bound hBound
      (fun output index hIndex =>
        output.push (source.uget (sourceIndices index hIndex) (hSourceIndices index hIndex)))
      (ByteArray.emptyWithCapacity length)
  appendSlice := Storage.byteBufferAppendSlice
  foldl := fun step initial buffer => buffer.foldl step initial
  size := ByteArray.size
  get buffer index hIndex := buffer.get index hIndex
  uget buffer index hIndex := buffer.uget index hIndex
  toArray := fun ⟨values⟩ => values
  ofArray := fun values => ⟨values⟩
  toArray_appendSlice := by
    intro source start stop output
    exact Storage.data_byteBufferAppendSlice source start stop output
  toArray_foldl := by
    intro β step initial buffer
    exact ByteArray.foldlM_eq_foldlM_data
  toArray_injective := by
    intro left right hData
    cases left
    cases right
    cases hData
    rfl

/--
`Float` tensors store an unboxed `FloatArray` and run core's `FloatArray` functions, except
`appendSlice` and `transpose2D`, which run C kernels. We prove the laws from core's definitions
and the Lean bodies of `floatBufferAppendSlice` and `floatBufferTranspose2D`.
-/
instance instFloatStorage : Storage Float where
  decEq? := some inferInstance
  Buffer := FloatArray
  emptyWithCapacity := FloatArray.emptyWithCapacity
  push := FloatArray.push
  copyAt source index hIndex output := output.push (source.uget index hIndex)
  -- We unfold `copyAt` in the class default, which saves the compiled instance one closure call
  -- per gathered entry.
  gather source length bound hBound sourceIndices hSourceIndices :=
    Native.finFoldl length bound hBound
      (fun output index hIndex =>
        output.push (source.uget (sourceIndices index hIndex) (hSourceIndices index hIndex)))
      (FloatArray.emptyWithCapacity length)
  transpose2D := fun source rows columns _ _ hSize =>
    Native.floatBufferTranspose2D source rows columns (by exact hSize)
  appendSlice := Storage.floatBufferAppendSlice
  foldl := fun step initial buffer => buffer.foldl step initial
  size := FloatArray.size
  get buffer index hIndex := buffer.get index hIndex
  uget buffer index hIndex := buffer.uget index hIndex
  toArray := fun ⟨values⟩ => values
  ofArray := fun values => ⟨values⟩
  toArray_transpose2D := by intros; rfl
  toArray_appendSlice := by
    intro source start stop output
    cases source
    cases output
    exact Storage.appendArraySlice_eq_extract _ _ _ _
  toArray_foldl := by
    intro β step initial buffer
    exact FloatArray.foldlM_eq_foldlM_data
  toArray_injective := by
    intro left right hData
    cases left
    cases right
    cases hData
    rfl

/-- Byte tensors replace entries in their packed native buffer with `ByteArray.set`. -/
instance instUInt8StorageUpdate :
    @Storage.Update UInt8 instUInt8Storage where
  set buffer index hBuffer value := buffer.set index value hBuffer
  toArray_set := by
    intro buffer index hBuffer value
    cases buffer
    rfl

/-- Float tensors replace entries in their unboxed native buffer with `FloatArray.set`. -/
instance instFloatStorageUpdate :
    @Storage.Update Float instFloatStorage where
  set buffer index hBuffer value := buffer.set index value hBuffer
  toArray_set := by
    intro buffer index hBuffer value
    cases buffer
    rfl

end TorchLean
