/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Tactic.Bound.Init
public import TorchLean.Tensor.Datatype
public import TorchLean.Tensor.Conversion

/-!
# Shape-changing tensor operations

Lemmas about `Tensor.reshape`, which keeps the row-major sequence of scalars and changes only the
static shape, so `x.reshape [_] (Shape.size_vector _).symm` flattens `x`. Reshaping back undoes a
reshape (`reshape_reshape_self`), and flattening a reshaped tensor gives the flattened original
(`reshape_vector_reshape`). `sequenceFin` stacks a family `Fin n → Option (Tensor α shape)` of
optional slices into an optional `Tensor α (.dim n shape)`.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

/-- A shape and its one-axis flattening `[shape.size]` have the same number of entries. -/
theorem Shape.size_toList_flatten (shape : Shape) :
    Shape.size shape =
      Shape.size [shape.size] := by
  simp

/-- A coordinate of the one-axis shape `[n]` linearizes to its index. -/
theorem Shape.Coord.val_linearize_vector {n : Nat} (index : Fin n) :
    (Shape.Coord.linearize
      (s := [n]) (index, PUnit.unit)).val = index.val := by
  have hRowMajor :=
    Shape.Coord.linearize_cons_val
      (s := []) index PUnit.unit
  have hEmptyBound :
      (Shape.Coord.linearize
        (s := []) PUnit.unit).val < 1 :=
    (Shape.Coord.linearize
      (s := []) PUnit.unit).isLt
  simp only [Shape.size_nil, Nat.one_mul] at hRowMajor
  grind

namespace Tensor
section

open Shape (size_toList_flatten)

/--
Flattening a stack `Tensor.dim values` puts entry `inner` of the flattened slice `values outer` at
position `outer * shape.size + inner`, so each slice fills one contiguous row-major segment.
-/
theorem getScalar_reshape_dim {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape}
    (values : Fin n → Tensor α shape) (outer : Fin n)
    (inner : Fin shape.size)
    (hIndex :
      outer.val * shape.size + inner.val < (Shape.dim n shape).size) :
    getScalar ((Tensor.dim values).reshape [_] (Shape.size_vector _).symm)
        ⟨outer.val * shape.size + inner.val, hIndex⟩ =
      getScalar ((values outer).reshape [_] (Shape.size_vector _).symm) inner := by
  rw [getScalar_eq_apply, getScalar_eq_apply]
  unfold reshape
  rw [Tensor.Rep.reshape_apply_reshapeEquiv,
    Tensor.Rep.reshape_apply_reshapeEquiv]
  have hCoordinate :
      Shape.Coord.reshapeEquiv
          (size_toList_flatten (Shape.dim n shape))
          (⟨outer.val * shape.size + inner.val, hIndex⟩, PUnit.unit) =
        (outer,
          Shape.Coord.reshapeEquiv
            (size_toList_flatten shape) (inner, PUnit.unit)) := by
    apply Shape.Coord.linearize_injective
    apply Fin.ext
    rw [Shape.Coord.linearize_reshapeEquiv_val,
      Shape.Coord.linearize_cons_val,
      Shape.Coord.linearize_reshapeEquiv_val]
    rw [Shape.Coord.val_linearize_vector,
      Shape.Coord.val_linearize_vector]
    simp [Nat.mul_comm, Nat.add_comm]
  rw [hCoordinate]
  exact Tensor.dim_apply values
    (outer, Shape.Coord.reshapeEquiv
      (size_toList_flatten shape) (inner, PUnit.unit))

end
end Tensor

private theorem Tensor.cast_vector_buffer {α : Type}
    [TorchLean.Storage α] {sourceSize targetSize : Nat}
    (hSize : sourceSize = targetSize)
    (tensor : Tensor α [sourceSize]) :
    (hSize ▸ tensor).buffer = tensor.buffer := by
  cases hSize
  rfl

/-- Reshaping to an equal-size shape and back returns the original tensor. -/
@[simp] theorem Tensor.reshape_reshape_self {α : Type} [TorchLean.Storage α] {source target : Shape}
    (tensor : Tensor α source) (h : source.size = target.size)
    (h' : target.size = source.size) :
    (tensor.reshape target h).reshape source h' = tensor :=
  (Tensor.Rep.reshape_reshape h h' tensor).trans
    (Tensor.Rep.reshape_rfl tensor)

/-- Flattening a vector keeps each entry at its index. -/
theorem Tensor.getScalar_reshape_vector {α : Type} [TorchLean.Storage α]
    {n : Nat} (x : Tensor α [n]) (i : Fin (Shape.size [n]))
    (hsize : Shape.size [n] = n) :
    (x.reshape [_] (Shape.size_vector _).symm).getScalar i = x.getScalar (Fin.cast hsize i) := by
  rw [getScalar_eq_apply, getScalar_eq_apply, reshape,
    Tensor.Rep.reshape_apply]
  congr 1
  apply Shape.Coord.linearize_injective
  rw [Shape.Coord.linearize_unlinearize]
  apply Fin.ext
  rfl

/-- Flattening a reshaped tensor gives the flattened original, cast along `hSize`. -/
theorem Tensor.reshape_vector_reshape {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    (hSize : source.size = target.size) :
    (tensor.reshape target hSize).reshape [_] (Shape.size_vector _).symm =
      hSize ▸ tensor.reshape [_] (Shape.size_vector _).symm := by
  apply Tensor.Rep.data_injective
  change TorchLean.Storage.toArray tensor.buffer =
    TorchLean.Storage.toArray (hSize ▸ tensor.reshape [_] (Shape.size_vector _).symm).buffer
  rw [cast_vector_buffer hSize (tensor.reshape [_] (Shape.size_vector _).symm)]
  rfl

/--
`sequenceFin values` stacks the slices along a leading axis of extent `n` when every `values i` is
`some`, and is `none` when any slice is missing.

Each slice is evaluated once, in index order, and an empty family gives an empty tensor. The
successful slices are kept in an intermediate vector, so the output is assembled in one pass.
Rebuilding the tensor at every recursive step would copy the earlier slices again each time, which
is costly for large batches and mixture models.
-/
def Tensor.sequenceFin {α : Type} [TorchLean.Storage α]
    {shape : Shape} {n : Nat}
    (values : Fin n → Option (Tensor α shape)) :
    Option (Tensor α (.dim n shape)) := do
  let slices ← Tensor.sequenceFinM values
  pure (Tensor.dim slices)

end TorchLean
