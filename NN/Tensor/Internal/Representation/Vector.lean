/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Batteries.Data.Vector.Lemmas
public import NN.Tensor.Internal.Representation.Basic.Reindex
public import NN.Tensor.Internal.Representation.Basic.Traversal -- shake: keep

/-!
# Fixed-length vector and array interoperability

Lean's `Vector α n` is an `Array α` together with a proof that the array has
exactly `n` entries. This module identifies that executable one-dimensional
storage with the row-major entries of a tensor of any finite shape.

The equivalence is not restricted to rank one. A scalar tensor corresponds
to a vector of length one, a matrix corresponds to a vector whose length is
the product of its two dimensions, and a shape containing a zero-length axis
corresponds to the empty vector.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u v

namespace Rep

/--
The equivalence between a tensor and its fixed-length row-major storage.

The vector contains one entry for every coordinate in `shape`. Converting
back uses row-major linearization to recover the original multidimensional
coordinate semantics.
-/
def vectorEquiv (α : Type u) [storage : Storage α]
    (shape : Shape) :
    Rep α shape ≃ Vector α (Shape.size shape) where
  toFun tensor := ⟨tensor.data, tensor.data_size⟩
  invFun values := Rep.ofArray values.toArray values.size_toArray
  left_inv tensor := by
    apply Rep.ext
    intro coordinate
    rw [Rep.get_ofArray]
    rw [Rep.data_getFlat]
    change Rep.flatten tensor (Coord.linearize coordinate) =
      tensor coordinate
    rw [Rep.flatten_apply, Coord.unlinearize_linearize]
  right_inv values := by
    apply Vector.ext
    intro index hIndex
    change
      (Rep.ofArray values.toArray values.size_toArray).data[index]'_ =
        values.toArray[index]'_
    simp only [Rep.data_ofArray]

/--
Looking up a vector entry reads the tensor coordinate represented by the
same row-major flat index.
-/
@[simp] theorem vectorEquiv_apply {α : Type u} [Storage α]
    {shape : Shape}
    (tensor : Rep α shape) (flatIndex : Fin (Shape.size shape)) :
    (vectorEquiv α shape tensor).get flatIndex =
      tensor (Coord.unlinearize flatIndex) := by
  calc
    (vectorEquiv α shape tensor).get flatIndex =
        tensor.getFlat flatIndex :=
      Rep.data_getFlat tensor flatIndex
    _ = tensor (Coord.unlinearize flatIndex) :=
      flatten_apply tensor flatIndex

/--
Converting a vector back to a tensor reads each coordinate at its row-major
linear index.
-/
@[simp] theorem vectorEquiv_symm_apply {α : Type u} [Storage α]
    {shape : Shape}
    (values : Vector α (Shape.size shape)) (coordinate : Coord shape) :
    (vectorEquiv α shape).symm values coordinate =
      values.get (Coord.linearize coordinate) := by
  change
    (Rep.ofArray values.toArray values.size_toArray).get
        coordinate =
      values.get (Coord.linearize coordinate)
  change
    (Rep.ofArray values.toArray values.size_toArray).getFlat
        (Coord.linearize coordinate) =
      values.get (Coord.linearize coordinate)
  calc
    (Rep.ofArray values.toArray values.size_toArray).getFlat
        (Coord.linearize coordinate) =
        values.toArray[(Coord.linearize coordinate).val]'_ :=
      Rep.getFlat_ofArray values.toArray values.size_toArray
        (Coord.linearize coordinate)
    _ = values.get (Coord.linearize coordinate) := rfl

/-- Pointwise tensor maps become ordinary fixed-length vector maps. -/
theorem vectorEquiv_map {α : Type u} {β : Type v}
    [Storage α] [Storage β] {shape : Shape}
    (f : α → β) (tensor : Rep α shape) :
    vectorEquiv β shape (map f tensor) =
      (vectorEquiv α shape tensor).map f := by
  apply Vector.ext
  intro flatIndex hFlatIndex
  let index : Fin (Shape.size shape) := ⟨flatIndex, by
    simpa using hFlatIndex⟩
  change
    ((vectorEquiv β shape) (map f tensor)).get index =
      ((vectorEquiv α shape tensor).map f).get index
  calc
    ((vectorEquiv β shape) (map f tensor)).get index =
        map f tensor (Coord.unlinearize index) :=
      vectorEquiv_apply (map f tensor) index
    _ = f (tensor (Coord.unlinearize index)) :=
      Rep.map_apply f tensor (Coord.unlinearize index)
    _ = f ((vectorEquiv α shape tensor).get index) :=
      congrArg f (vectorEquiv_apply tensor index).symm
    _ = ((vectorEquiv α shape tensor).map f).get index :=
      by simp

/--
Reshape preserves the same row-major vector storage.

Only the proof-indexed vector length changes, so `Vector.cast` transports the
source vector along the equality between the source and target shape sizes.
-/
theorem vectorEquiv_reshape {α : Type u} [Storage α]
    {sourceShape targetShape : Shape}
    (hSize : Shape.size sourceShape = Shape.size targetShape)
    (tensor : Rep α sourceShape) :
    vectorEquiv α targetShape (reshape hSize tensor) =
      (vectorEquiv α sourceShape tensor).cast hSize := by
  apply Vector.ext
  intro flatIndex hFlatIndex
  rfl

/--
The array underlying a tensor's vector view has exactly the tensor's number
of entries.
-/
theorem vectorEquiv_toArray_size {α : Type u} [Storage α]
    {shape : Shape}
    (tensor : Rep α shape) :
    (vectorEquiv α shape tensor).toArray.size = Shape.size shape :=
  (vectorEquiv α shape tensor).size_toArray

end Rep

end TorchLean.Tensor.Internal
