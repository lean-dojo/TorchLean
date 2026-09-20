/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Native.Loop
public import NN.Tensor.Internal.Representation.Shape

/-!
# Native shaped tensor storage

`Rep α shape` owns one contiguous row-major buffer selected by
`Storage α`. Its proof field certifies that the buffer length is exactly
`Shape.size shape`, while its coordinate-function view supplies the
mathematical observation semantics.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u

/--
A contiguous row-major tensor whose storage length is certified by its shape.

The proof is erased by code generation. At runtime a tensor is therefore its
selected native buffer, rather than a coordinate closure or a second semantic
representation.
-/
structure Rep (α : Type u) (s : Shape) [storage : Storage α] where
  /-- Contiguous row-major scalar storage. -/
  buffer : storage.Buffer
  /-- The storage has exactly one entry for every coordinate of the shape. -/
  size_eq : storage.size buffer = Shape.size s

namespace Rep

/-- Transport a tensor along an equality of its static shape. -/
def castShape {α : Type u} [Storage α]
    {sourceShape targetShape : Shape}
    (h : sourceShape = targetShape) (tensor : Rep α sourceShape) :
    Rep α targetShape :=
  h ▸ tensor

/-- Read one row-major entry from a tensor's native storage. -/
@[inline] def getFlat {α : Type u} [storage : Storage α]
    {s : Shape} (x : Rep α s)
    (i : Fin (Shape.size s)) : α :=
  storage.get x.buffer i.val (by simpa only [x.size_eq] using i.isLt)

/--
Read one row-major entry using a platform-native array index.

The explicit bound is the same safety certificate carried by `Fin` in
`getFlat`. This form is used by generated kernels whose index arithmetic has
already been proved to fit `USize`, avoiding boxed natural-number indexing in
the scalar loop.
-/
@[inline] def getFlatUSize {α : Type u} [storage : Storage α]
    {s : Shape} (x : Rep α s)
    (i : USize) (h : i.toNat < Shape.size s) : α :=
  storage.uget x.buffer i (by simpa only [x.size_eq] using h)

/--
Platform-native and finite flat reads agree at the same row-major position.

Generic arrays satisfy this definitionally. Specialized storage proves it
through the common ordinary-array observation.
-/
theorem getFlatUSize_eq_getFlat
    {α : Type u} [storage : Storage α] {s : Shape}
    (x : Rep α s) (i : USize) (h : i.toNat < Shape.size s) :
    x.getFlatUSize i h = x.getFlat ⟨i.toNat, h⟩ := by
  have hBuffer : i.toNat < storage.size x.buffer := by
    simpa only [x.size_eq] using h
  have hArray : i.toNat < (storage.toArray x.buffer).size := by
    rw [storage.toArray_size, x.size_eq]
    exact h
  calc
    storage.uget x.buffer i hBuffer =
        (storage.toArray x.buffer).uget i hArray :=
      (storage.toArray_uget x.buffer i hBuffer hArray).symm
    _ = (storage.toArray x.buffer)[i.toNat]'hArray := rfl
    _ = storage.get x.buffer i.toNat hBuffer :=
      storage.toArray_get x.buffer i.toNat hBuffer hArray

/--
Changing a native flat index along an equality does not change the observed
tensor entry. The two bounds proofs may differ because their types mention the
index; proof irrelevance removes that distinction after the indices agree.
-/
theorem getFlatUSize_congr {α : Type u} [Storage α]
    {s : Shape} (x : Rep α s)
    {i j : USize} (h : i = j)
    (hi : i.toNat < Shape.size s)
    (hj : j.toNat < Shape.size s) :
    x.getFlatUSize i hi = x.getFlatUSize j hj := by
  subst j
  rfl

/-- Read the entry at a multidimensional coordinate. -/
@[inline] def get {α : Type u} [Storage α]
    {s : Shape} (x : Rep α s) (i : Coord s) : α :=
  x.getFlat (Coord.linearize i)

/--
Replace one row-major entry while preserving the tensor's static shape.

Packed storage performs a native copy-on-write update. Other storage types use
the universal array-backed update capability supplied by `Storage.Update`.
-/
@[inline] def setFlat {α : Type u} [storage : Storage α]
    [update : Storage.Update α] {s : Shape} (x : Rep α s)
    (i : Fin (Shape.size s)) (value : α) : Rep α s where
  buffer :=
    update.set x.buffer i.val
      (by simpa only [x.size_eq] using i.isLt) value
  size_eq := by
    rw [← storage.toArray_size, update.toArray_set, Array.size_set,
      storage.toArray_size, x.size_eq]

/-- Replace the entry at a statically valid multidimensional coordinate. -/
@[inline] def set {α : Type u} [Storage α] [Storage.Update α]
    {s : Shape} (x : Rep α s) (i : Coord s) (value : α) : Rep α s :=
  x.setFlat (Coord.linearize i) value

/-- Transform the entry at one statically valid coordinate. -/
@[inline] def modify {α : Type u} [Storage α] [Storage.Update α]
    {s : Shape} (x : Rep α s) (i : Coord s) (f : α → α) : Rep α s :=
  x.set i (f (x.get i))

/-- A tensor acts as its coordinate observation function in statements and proofs. -/
instance {α : Type u} [Storage α] {s : Shape} :
    CoeFun (Rep α s) (fun _ => Coord s → α) where
  coe := get

/--
Render a tensor as shape-aware nested lists, with rank-zero tensors rendered
as their scalar value.

The recursion follows the coordinate type itself, so it works uniformly at
every rank and renders a zero-length axis as an empty list.
-/
def format {α : Type u} [Repr α] :
    (shape : Shape) → (Coord shape → α) → Std.Format
  | [], values => repr (values PUnit.unit)
  | length :: shape, values =>
      Std.Format.bracket "["
        (Std.Format.joinSep
          (List.ofFn fun index : Fin length =>
            format shape fun coordinate =>
              values (index, coordinate))
          ("," ++ Std.Format.line))
        "]"

/-- Print a tensor as a scalar or shape-aware nested row-major lists. -/
instance {α : Type u} [Storage α] {shape : Shape} [Repr α] :
    Repr (Rep α shape) where
  reprPrec tensor _ :=
    format shape fun coordinate => tensor coordinate

/-- Use the ordinary shape-aware tensor rendering in strings and interpolations. -/
instance {α : Type u} [Storage α] {shape : Shape} [Repr α] :
    ToString (Rep α shape) where
  toString tensor := reprStr tensor

/--
Print a finite dependent tensor family in component order.

This includes the result type returned by `unpack`; each component retains
its own statically checked shape and uses the ordinary tensor representation.
-/
instance {α : Type u} [Storage α] {componentCount : Nat}
    {shapes : Fin componentCount → Shape} [Repr α] :
    Repr ((component : Fin componentCount) → Rep α (shapes component)) where
  reprPrec tensors _ :=
    Std.Format.bracket "["
      (Std.Format.joinSep
        (List.ofFn fun component => repr (tensors component))
        ("," ++ Std.Format.line))
      "]"

/--
Build a tensor from a row-major function.

The selected storage backend emits one native allocation and fills it in
increasing flat-index order.
-/
def ofFlatFn {α : Type u} [storage : Storage α] {s : Shape}
    (values : Fin (Shape.size s) → α) : Rep α s where
  buffer := Storage.ofFn values
  size_eq := Storage.size_ofFn values

/--
Build a tensor with a platform-native flat-index loop when its static size
fits `USize`, and retain the universal finite-index construction otherwise.

The native and semantic callbacks are connected by an erased certificate.
Ordinary executable tensor shapes therefore use unboxed loop counters and
native storage reads, while the representation remains total for arbitrary
mathematical shapes.
-/
@[inline] def ofFlatNativeFn {α : Type u} [storage : Storage α] {s : Shape}
    (nativeValues :
      (index : USize) → index.toNat < Shape.size s → α)
    (values : Fin (Shape.size s) → α)
    (_hValues :
      ∀ (index : USize) (hIndex : index.toNat < Shape.size s),
        nativeValues index hIndex =
          values ⟨index.toNat, hIndex⟩) :
    Rep α s :=
  if hFits : Shape.size s < USize.size then
    let bound := USize.ofNatLT (Shape.size s) hFits
    Rep.mk
      (Tensor.Internal.Elab.Impl.nativeBufferOfFn
        (Shape.size s) bound (by simp [bound]) nativeValues)
      (Tensor.Internal.Elab.Impl.nativeBufferOfFn_size
        (Shape.size s) bound (by simp [bound]) nativeValues)
  else
    ofFlatFn values

/-- Build a tensor from an ordinary row-major array of the certified size. -/
def ofArray {α : Type u} [storage : Storage α] {s : Shape}
    (values : Array α) (hSize : values.size = Shape.size s) :
    Rep α s where
  buffer := storage.ofArray values
  size_eq := by
    rw [← storage.toArray_size, storage.toArray_ofArray, hSize]

/--
Rebuilding a tensor from a buffer equal to `Array.ofFn values` gives the
canonical flat-function tensor.

This theorem lets verified native builders replace their certified output
buffer without exposing proof-field equality to generated code.
-/
theorem mk_eq_ofFlatFn {α : Type u} [storage : Storage α] {s : Shape}
    (values : Fin (Shape.size s) → α)
    (buffer : storage.Buffer)
    (hSize : storage.size buffer = Shape.size s)
    (hData : storage.toArray buffer = Array.ofFn values) :
    Rep.mk buffer hSize = ofFlatFn values := by
  congr 1
  apply storage.toArray_injective
  rw [hData, Storage.toArray_ofFn]

/-- Native flat construction has the ordinary finite-index tensor semantics. -/
theorem ofFlatNativeFn_eq_ofFlatFn
    {α : Type u} [storage : Storage α] {s : Shape}
    (nativeValues :
      (index : USize) → index.toNat < Shape.size s → α)
    (values : Fin (Shape.size s) → α)
    (hValues :
      ∀ (index : USize) (hIndex : index.toNat < Shape.size s),
        nativeValues index hIndex =
          values ⟨index.toNat, hIndex⟩) :
    ofFlatNativeFn nativeValues values hValues = ofFlatFn values := by
  unfold ofFlatNativeFn
  split
  · apply mk_eq_ofFlatFn values
    exact Tensor.Internal.Elab.Impl.nativeBufferOfFn_toArray
      (Shape.size s) (USize.ofNatLT (Shape.size s) ‹_›)
      (by simp) nativeValues values hValues
  · rfl

/-- Build a native tensor by evaluating a coordinate function once per entry. -/
def ofFn {α : Type u} [Storage α] {s : Shape}
    (values : Coord s → α) : Rep α s :=
  ofFlatFn fun flatIndex => values (Coord.unlinearize flatIndex)

/-- Reading a generated tensor at a flat index returns the generated value. -/
@[simp, grind =] theorem getFlat_ofFlatFn {α : Type u}
    [storage : Storage α] {s : Shape}
    (values : Fin (Shape.size s) → α) (i : Fin (Shape.size s)) :
    getFlat (ofFlatFn values) i = values i := by
  change storage.get (Storage.ofFn values) i.val _ = values i
  have hData := Storage.toArray_ofFn values
  have hGet := storage.toArray_get (Storage.ofFn values) i.val
    (by simpa only [Storage.size_ofFn] using i.isLt)
    (by simpa only [hData, Array.size_ofFn] using i.isLt)
  simpa only [hData, Array.getElem_ofFn] using hGet.symm

/-- Reading a flat-generated tensor linearizes the requested coordinate. -/
@[simp, grind =] theorem get_ofFlatFn {α : Type u}
    [Storage α] {s : Shape}
    (values : Fin (Shape.size s) → α) (i : Coord s) :
    ofFlatFn values i = values (Coord.linearize i) := by
  change getFlat (ofFlatFn values) (Coord.linearize i) =
    values (Coord.linearize i)
  exact getFlat_ofFlatFn values (Coord.linearize i)

/-- Reading a generated tensor at a coordinate returns the generated value. -/
@[simp, grind =] theorem get_ofFn {α : Type u}
    [Storage α] {s : Shape}
    (values : Coord s → α) (i : Coord s) :
    ofFn values i = values i := by
  simp [ofFn, get, Coord.unlinearize_linearize]

/-- The native buffer has exactly the statically known tensor size. -/
def data {α : Type u} [storage : Storage α] {s : Shape}
    (x : Rep α s) : Array α :=
  storage.toArray x.buffer

/--
Executable tensors have decidable elementwise equality whenever their scalar type does.

The comparison uses the storage's certified ordinary-array observation. Equality of those
observations determines equality of the native buffers, while the size certificates are
proof-irrelevant.
-/
instance {α : Type u} [storage : Storage α] [DecidableEq α] {s : Shape} :
    DecidableEq (Rep α s) := fun left right =>
  if hData : left.data = right.data then
    isTrue (by
      cases left with
      | mk leftBuffer leftSize =>
          cases right with
          | mk rightBuffer rightSize =>
              simp only [data] at hData
              have hBuffer : leftBuffer = rightBuffer :=
                storage.toArray_injective hData
              cases hBuffer
              rfl)
  else
    isFalse (fun hTensor => hData (congrArg data hTensor))

/--
Traverse tensor entries in physical row-major order without materializing the
ordinary array observation.
-/
@[inline] def foldl {α β : Type u} [storage : Storage α] {s : Shape}
    (step : β → α → β) (initial : β) (x : Rep α s) : β :=
  storage.foldl step initial x.buffer

/-- Native tensor traversal agrees with folding the proof-facing observation. -/
theorem foldl_eq_data_foldl {α β : Type u} [storage : Storage α]
    {s : Shape} (step : β → α → β) (initial : β) (x : Rep α s) :
    x.foldl step initial = x.data.foldl step initial :=
  storage.toArray_foldl step initial x.buffer

/-- Observing a tensor built from an ordinary array returns that array. -/
@[simp] theorem data_ofArray {α : Type u} [storage : Storage α]
    {s : Shape} (values : Array α)
    (hSize : values.size = Shape.size s) :
    (ofArray values hSize).data = values :=
  storage.toArray_ofArray values

/-- Flat lookup into a tensor built from an ordinary array reads that array. -/
@[simp] theorem getFlat_ofArray {α : Type u} [storage : Storage α]
    {s : Shape} (values : Array α)
    (hSize : values.size = Shape.size s)
    (index : Fin (Shape.size s)) :
    (ofArray values hSize).getFlat index =
      values[index.val]'(by simpa only [hSize] using index.isLt) := by
  change storage.get (storage.ofArray values) index.val _ = _
  have hObserved := storage.toArray_get
    (storage.ofArray values) index.val
    (by
      rw [← storage.toArray_size, storage.toArray_ofArray, hSize]
      exact index.isLt)
    (by
      rw [storage.toArray_ofArray, hSize]
      exact index.isLt)
  simpa only [storage.toArray_ofArray] using hObserved.symm

/-- Coordinate lookup into an array-built tensor uses row-major linearization. -/
@[simp] theorem get_ofArray {α : Type u} [Storage α]
    {s : Shape} (values : Array α)
    (hSize : values.size = Shape.size s) (coordinate : Coord s) :
    ofArray values hSize coordinate =
      values[(Coord.linearize coordinate).val]'(by
        simpa only [hSize] using (Coord.linearize coordinate).isLt) := by
  exact getFlat_ofArray values hSize (Coord.linearize coordinate)

/--
The proof-facing ordinary array observation has the statically known tensor
size. Evaluating `data` may convert specialized storage, so native kernels use
`buffer`, `getFlat`, and `getFlatUSize` instead.
-/
@[simp] theorem data_size {α : Type u} [storage : Storage α]
    {s : Shape} (x : Rep α s) :
    x.data.size = Shape.size s :=
  (storage.toArray_size x.buffer).trans x.size_eq

/-- Reading the array observation agrees with native flat tensor lookup. -/
@[simp] theorem data_getFlat {α : Type u} [storage : Storage α]
    {s : Shape} (x : Rep α s)
    (index : Fin (Shape.size s)) :
    x.data[index.val]'(by simpa only [x.data_size] using index.isLt) =
      x.getFlat index := by
  exact storage.toArray_get x.buffer index.val
    (by simpa only [x.size_eq] using index.isLt)
    (by
      rw [storage.toArray_size, x.size_eq]
      exact index.isLt)

/-- Reading a flat entry immediately after replacing it returns the new value. -/
@[simp] theorem getFlat_setFlat_self {α : Type u}
    [storage : Storage α] [update : Storage.Update α]
    {s : Shape} (x : Rep α s) (i : Fin (Shape.size s)) (value : α) :
    (x.setFlat i value).getFlat i = value := by
  calc
    (x.setFlat i value).getFlat i =
        (x.setFlat i value).data[i.val]'(by
          simpa only [(x.setFlat i value).data_size] using i.isLt) :=
      (data_getFlat (x.setFlat i value) i).symm
    _ = value := by
      simp only [data, setFlat, update.toArray_set]
      exact Array.getElem_set_self
        (by
          rw [storage.toArray_size, x.size_eq]
          exact i.isLt)

/-- Reading a coordinate immediately after replacing it returns the new value. -/
@[simp] theorem get_set_self {α : Type u}
    [Storage α] [Storage.Update α]
    {s : Shape} (x : Rep α s) (i : Coord s) (value : α) :
    x.set i value i = value := by
  exact getFlat_setFlat_self x (Coord.linearize i) value

/-- Reading a coordinate after modifying it returns the transformed old value. -/
@[simp] theorem get_modify_self {α : Type u}
    [Storage α] [Storage.Update α]
    {s : Shape} (x : Rep α s) (i : Coord s) (f : α → α) :
    x.modify i f i = f (x i) := by
  exact get_set_self x i (f (x.get i))

/-- Two native tensors are equal when all of their coordinate observations agree. -/
@[ext, grind ext] theorem ext {α : Type u} [storage : Storage α]
    {s : Shape} {x y : Rep α s}
    (h : ∀ i, x i = y i) : x = y := by
  cases x with
  | mk xData xSize =>
    cases y with
    | mk yData ySize =>
      congr 1
      apply storage.toArray_injective
      apply Array.ext
      · simpa only [storage.toArray_size] using xSize.trans ySize.symm
      · intro index xBound yBound
        let flatIndex : Fin (Shape.size s) :=
          ⟨index, by
            rw [storage.toArray_size] at xBound
            simpa only [xSize] using xBound⟩
        have hObserved := h (Coord.unlinearize flatIndex)
        have hLeft := storage.toArray_get xData index
          (by simpa only [xSize] using flatIndex.isLt) xBound
        have hRight := storage.toArray_get yData index
          (by simpa only [ySize] using flatIndex.isLt) yBound
        rw [hLeft, hRight]
        simpa [get, getFlat, flatIndex] using hObserved

end Rep

end TorchLean.Tensor.Internal
