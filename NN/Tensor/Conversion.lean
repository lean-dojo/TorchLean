/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Constructors
public import NN.Spec.Core.Tensor -- shake: keep

/-!
# Tensor Conversion

Total in-memory conversion belongs here. Sources carry their intrinsic shape:
ordinary arrays and lists become vectors, while future structured sources may
provide richer shapes through `Tensor.From`.

Changing only the interpretation of a flat row-major buffer is `Tensor.reshape`
and requires a proof that the scalar count is unchanged. Validation of external
files and untrusted runtime metadata belongs in the corresponding loader.
-/

@[expose] public section

namespace TorchLean.Tensor

open Spec

/-- Intrinsic tensor shape described by an in-memory source value. -/
class SourceShape (source : Type) where
  /-- Shape intrinsically described by the source value. -/
  shape : source → Shape

/-- Extensible, total materialization of a shaped in-memory source. -/
class From (source : Type) (α : outParam Type) [TorchLean.Storage α]
    [SourceShape source] where
  /-- Construct the tensor without a runtime failure branch. -/
  make : (value : source) → Tensor α (SourceShape.shape value)

/-- Convert an in-memory value into a tensor with its intrinsic shape. -/
def «from» {source α : Type} [TorchLean.Storage α]
    [SourceShape source] [conversion : From source α] (value : source) :
    Tensor α (SourceShape.shape value) :=
  conversion.make value

/-- An `Array` describes one dimension, its length.

The shape is computed from the value rather than declared by the caller, which is what lets
`Tensor.from` be written without a shape annotation. -/
@[instance_reducible] instance arraySourceShape {α : Type} :
    SourceShape (Array α) where
  shape values := [values.size]

/-- The shape an `Array` describes is its length. -/
@[simp] theorem sourceShape_array {α : Type} (values : Array α) :
    SourceShape.shape values = [values.size] :=
  rfl

/-- Materialize an `Array` as a rank-one tensor; the buffer is already row-major. -/
@[instance_reducible] instance arrayFrom {α : Type} [TorchLean.Storage α] :
    From (Array α) α where
  make values :=
    TorchLean.Tensor.Internal.Rep.ofArray values (by
      simp [TorchLean.Tensor.Internal.Shape.size])

/-- A `List` behaves like an `Array`: one dimension, its length. -/
@[instance_reducible] instance listSourceShape {α : Type} :
    SourceShape (List α) where
  shape values := [values.length]

/-- The shape a `List` describes is its length. -/
@[simp] theorem sourceShape_list {α : Type} (values : List α) :
    SourceShape.shape values = [values.length] :=
  rfl

/-- Materialize a `List` as a rank-one tensor, copying once through `List.toArray`. -/
@[instance_reducible] instance listFrom {α : Type} [TorchLean.Storage α] :
    From (List α) α where
  make values :=
    TorchLean.Tensor.Internal.Rep.ofArray values.toArray (by
      simp [TorchLean.Tensor.Internal.Shape.size])

/-- A `Vector α n` carries its length in its type, so the shape is known without inspecting the
value at all. -/
@[instance_reducible] instance vectorSourceShape {α : Type} {n : Nat} :
    SourceShape (Vector α n) where
  shape _ := [n]

/-- The shape a `Vector α n` describes is `[n]`, independently of the value. -/
@[simp] theorem sourceShape_vector {α : Type} {n : Nat} (values : Vector α n) :
    SourceShape.shape values = [n] :=
  rfl

/-- Materialize a `Vector α n` as a `Tensor α [n]`; no length check is needed. -/
@[instance_reducible] instance vectorFrom
    {α : Type} [TorchLean.Storage α] {n : Nat} :
    From (Vector α n) α where
  make values :=
    TorchLean.Tensor.Internal.Rep.ofArray values.toArray (by
      simp [TorchLean.Tensor.Internal.Shape.size])

/-- A `FloatArray` describes one dimension, the length reported by its `Storage` instance. -/
@[instance_reducible] instance floatArraySourceShape : SourceShape FloatArray where
  shape values :=
    [@TorchLean.Storage.size Float TorchLean.instFloatStorage values]

/-- A `FloatArray` is already the unboxed `Float` storage, so materialization is a rewrap. -/
@[instance_reducible] instance floatArrayFrom :
    @From FloatArray Float TorchLean.instFloatStorage floatArraySourceShape where
  make values := by
    change Tensor Float
      [@TorchLean.Storage.size Float TorchLean.instFloatStorage values]
    exact
      { buffer := values
        size_eq := by simp [TorchLean.Tensor.Internal.Shape.size] }

/-- A `ByteArray` describes one dimension, the length reported by its `Storage` instance. -/
@[instance_reducible] instance byteArraySourceShape : SourceShape ByteArray where
  shape values :=
    [@TorchLean.Storage.size UInt8 TorchLean.instUInt8Storage values]

/-- A `ByteArray` is already the unboxed `UInt8` storage, so materialization is a rewrap. -/
@[instance_reducible] instance byteArrayFrom :
    @From ByteArray UInt8 TorchLean.instUInt8Storage byteArraySourceShape where
  make values := by
    change Tensor UInt8
      [@TorchLean.Storage.size UInt8 TorchLean.instUInt8Storage values]
    exact
      { buffer := values
        size_eq := by simp [TorchLean.Tensor.Internal.Shape.size] }

/--
Reinterpret the same contiguous row-major buffer at an equal-size shape.

The equality is the complete safety condition: reshape neither pads, truncates,
nor moves scalar data.

Example:
```lean
-- Same buffer, new shape. The size equality is the entire safety condition, and it is checked
-- here rather than trusted.
def matrix (flat : Tensor Float [12]) : Tensor Float [3, 4] :=
  Tensor.reshape flat [3, 4]
```
-/
def reshape {α : Type} [TorchLean.Storage α]
    {source : Shape} (tensor : Tensor α source) (target : Shape)
    (hSize : source.size = target.size := by decide) :
    Tensor α target :=
  TorchLean.Tensor.Internal.Rep.reshape (Shape.internalSize_congr hSize) tensor

/-- Extensible conversion from a tensor to a requested in-memory target type. -/
class To (α : Type) [TorchLean.Storage α] (shape : Shape)
    (target : Type) where
  /-- Materialize or expose the requested target representation. -/
  convert : Tensor α shape → target

/-- Convert a tensor to the requested target type. -/
abbrev «to» {α : Type} [TorchLean.Storage α] {shape : Shape}
    (tensor : Tensor α shape) (target : Type)
    [conversion : To α shape target] : target :=
  conversion.convert tensor

/-- Read a tensor out as its row-major array. -/
instance arrayTo
    {α : Type} [TorchLean.Storage α] {shape : Shape} :
    To α shape (Array α) where
  convert tensor := tensor.data

/-- Read a tensor out as its row-major list. -/
instance listTo
    {α : Type} [TorchLean.Storage α] {shape : Shape} :
    To α shape (List α) where
  convert tensor := tensor.data.toList

-- A rank-one natural-number tensor reads out as a shape through `listTo`, because a shape is a
-- list of dimensions. A separate instance for `Shape` would overlap with it.

/-- Read a tensor out as a length-indexed vector. -/
instance vectorTo
    {α : Type} [TorchLean.Storage α] {shape : Shape} :
    To α shape (Vector α (Spec.Shape.size shape)) where
  convert tensor :=
    ⟨tensor.data, by
      exact (TorchLean.Tensor.Internal.Rep.data_size tensor).trans
        (Shape.internalSize_eq shape)⟩

/-- Expose the native buffer of a `Float` tensor. -/
instance floatArrayTo {shape : Shape} :
    To Float shape FloatArray where
  convert tensor := tensor.buffer

/-- Expose the native buffer of a byte tensor. -/
instance byteArrayTo {shape : Shape} :
    To UInt8 shape ByteArray where
  convert tensor := tensor.buffer

/-- Converting to `Array` exposes the row-major observation, boxing packed storage if needed. -/
@[simp] theorem to_array_eq_data {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape) :
    Tensor.to tensor (Array α) = tensor.data :=
  rfl

/-- Converting to `List` is the array conversion followed by `Array.toList`. -/
@[simp] theorem to_list_eq_data {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape) :
    Tensor.to tensor (List α) = tensor.data.toList :=
  rfl

/-- Array in, array out: `Tensor.from` then `Tensor.to` is the identity.

The four round-trip lemmas that follow are the reason the conversion layer can be trusted at the
boundary. Nothing in the tensor representation reorders or pads the data, so importing and exporting
gives back exactly what was handed in. -/
@[simp] theorem to_array_from_array {α : Type} [TorchLean.Storage α]
    (values : Array α) :
    Tensor.to (Tensor.from values) (Array α) = values := by
  change
    (TorchLean.Tensor.Internal.Rep.ofArray values
      (by simp [TorchLean.Tensor.Internal.Shape.size])).data = values
  exact TorchLean.Tensor.Internal.Rep.data_ofArray values _

/-- Array in, list out. -/
@[simp] theorem to_list_from_array {α : Type} [TorchLean.Storage α]
    (values : Array α) :
    Tensor.to (Tensor.from values) (List α) = values.toList := by
  change
    (TorchLean.Tensor.Internal.Rep.ofArray values
      (by simp [TorchLean.Tensor.Internal.Shape.size])).data.toList =
      values.toList
  rw [TorchLean.Tensor.Internal.Rep.data_ofArray]

/-- List in, list out. -/
@[simp] theorem to_list_from_list {α : Type} [TorchLean.Storage α]
    (values : List α) :
    Tensor.to (Tensor.from values) (List α) = values := by
  change
    (TorchLean.Tensor.Internal.Rep.ofArray values.toArray
      (by simp [TorchLean.Tensor.Internal.Shape.size])).data.toList = values
  rw [TorchLean.Tensor.Internal.Rep.data_ofArray]

/-- A shape cast does not touch the data, so exporting before or after it gives the same array.

This is the statement that makes `castShape` free: it is a retyping, not a copy. -/
@[simp] theorem to_array_castShape {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    (hShape : source = target) :
    Tensor.to (Tensor.castShape tensor hShape) (Array α) =
      Tensor.to tensor (Array α) := by
  cases hShape
  rfl

/-- The list version of `to_array_castShape`. -/
@[simp] theorem to_list_castShape {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    (hShape : source = target) :
    Tensor.to (Tensor.castShape tensor hShape) (List α) =
      Tensor.to tensor (List α) := by
  cases hShape
  rfl

/-- Reshaping is likewise data-preserving: only the interpretation of the flat buffer changes.

Together with `to_array_castShape` this pins down the row-major convention. A reshape that permuted
elements would break this lemma, so it doubles as a regression test on the layout. -/
@[simp] theorem to_array_reshape {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    (hSize : source.size = target.size) :
    Tensor.to (tensor.reshape target hSize) (Array α) =
      Tensor.to tensor (Array α) := by
  rfl

/-- Converting a filled tensor to a list produces one value per scalar position. -/
@[simp] theorem to_list_full {α : Type} [TorchLean.Storage α]
    (shape : Shape) (value : α) :
    Tensor.to (Tensor.full shape value) (List α) =
      List.replicate shape.size value := by
  apply List.ext_getElem
  · simp [Shape.internalSize_eq]
  · intro index hLeft hRight
    simp only [to_list_eq_data]
    have hFlat : index < TorchLean.Tensor.Internal.Shape.size shape.toList := by
      rw [← TorchLean.Tensor.Internal.Rep.data_size (Tensor.full shape value),
        ← Array.length_toList]
      exact hLeft
    change
      (TorchLean.Tensor.Internal.Rep.data (Tensor.full shape value))[index]'_ =
        (List.replicate shape.size value)[index]
    calc
      _ = TorchLean.Tensor.Internal.Rep.getFlat (Tensor.full shape value) ⟨index, hFlat⟩ := by
        exact TorchLean.Tensor.Internal.Rep.data_getFlat (Tensor.full shape value) ⟨index, hFlat⟩
      _ = value := by simp [Tensor.full, TorchLean.Tensor.Internal.Rep.const]
      _ = (List.replicate shape.size value)[index] := by simp

/-- A filled natural-number vector describes the corresponding uniform shape. -/
@[simp] theorem to_shape_full (rank value : Nat) :
    Tensor.to (Tensor.full [rank] value) Shape =
      Shape.ofList (List.replicate rank value) := by
  change Shape.ofList (Tensor.to (Tensor.full [rank] value) (List Nat)) = _
  rw [to_list_full]
  simp [Shape.size]

/-- Native vector multiplication agrees with the product of its list conversion. -/
theorem prod_eq_to_list_prod {α : Type}
    [TorchLean.Storage α] [Monoid α]
    {n : Nat} (tensor : Tensor α [n]) :
    tensor.prod = (Tensor.to tensor (List α)).prod := by
  have foldl_mul (values : List α) (accumulator : α) :
      values.foldl (· * ·) accumulator = accumulator * values.prod := by
    induction values generalizing accumulator with
    | nil => simp
    | cons value values inductionHypothesis =>
        simp [inductionHypothesis, mul_assoc]
  rw [Tensor.prod, TorchLean.Tensor.Internal.Rep.foldl_eq_data_foldl,
    ← Array.foldl_toList, foldl_mul, one_mul]
  rfl

/-- Converting a natural-number vector to a shape preserves its product as the shape size. -/
@[simp] theorem size_to_shape {rank : Nat} (tensor : Tensor Nat [rank]) :
    (Tensor.to tensor Shape).size = tensor.prod := by
  change Shape.size (Tensor.to tensor (List Nat)) = tensor.prod
  rw [Shape.size_eq_prod, ← prod_eq_to_list_prod]

end TorchLean.Tensor
