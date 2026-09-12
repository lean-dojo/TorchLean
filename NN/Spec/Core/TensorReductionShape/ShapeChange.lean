/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Tactic.Bound.Init
public import NN.Spec.Core.Tensor.Core

/-!
# Shape-changing tensor operations

Flatten, unflatten, and reshape preserve the row-major scalar sequence. Since
`TorchLean.Tensor` owns a contiguous certified buffer, these operations are
zero-copy changes to the static shape proof.
-/

@[expose] public section

open Spec TorchLean

namespace TorchLean.Tensor

/-- The buffer and specification size functions agree on the flattened shape. -/
theorem size_toList_flatten (shape : Shape) :
    TorchLean.Tensor.Internal.Shape.size shape.toList =
      TorchLean.Tensor.Internal.Shape.size [shape.size] := by
  simp [Shape.internalSize_eq, Shape.toList, Shape.size]

/-- A reshape changes coordinates without changing their row-major index. -/
theorem reshapeCoordEquiv_linearize_val
    {source target : TorchLean.Tensor.Internal.Shape}
    (hSize : TorchLean.Tensor.Internal.Shape.size source =
      TorchLean.Tensor.Internal.Shape.size target)
    (coordinate : TorchLean.Tensor.Internal.Coord target) :
    (TorchLean.Tensor.Internal.Coord.linearize
      (TorchLean.Tensor.Internal.Rep.reshapeCoordEquiv hSize coordinate)).val =
      (TorchLean.Tensor.Internal.Coord.linearize coordinate).val := by
  change
    ((TorchLean.Tensor.Internal.Coord.equivFin source)
      ((TorchLean.Tensor.Internal.Coord.equivFin source).symm
        (finCongr hSize.symm
          ((TorchLean.Tensor.Internal.Coord.equivFin target) coordinate)))).val =
      ((TorchLean.Tensor.Internal.Coord.equivFin target) coordinate).val
  rw [Equiv.apply_symm_apply]
  rfl

/-- A one-axis coordinate linearizes to its sole finite index. -/
theorem vectorCoordinate_linearize_val {n : Nat} (index : Fin n) :
    (TorchLean.Tensor.Internal.Coord.linearize
      (s := [n]) (index, PUnit.unit)).val = index.val := by
  have hRowMajor :=
    TorchLean.Tensor.Internal.Coord.linearize_cons_val
      (s := []) index PUnit.unit
  have hEmptyBound :
      (TorchLean.Tensor.Internal.Coord.linearize
        (s := []) PUnit.unit).val < 1 :=
    (TorchLean.Tensor.Internal.Coord.linearize
      (s := []) PUnit.unit).isLt
  simp only [TorchLean.Tensor.Internal.Shape.size_nil, Nat.one_mul] at hRowMajor
  grind

/-- Reverse the coordinates of an arbitrary statically valid axis. -/
def reverseAxis {α : Type} [TorchLean.Storage α] :
    (axis : Nat) → {shape : Shape} → Tensor α shape →
      [_h : Shape.AxisInBounds axis shape] → Tensor α shape
  | 0, .dim length _, tensor, _ =>
      Tensor.dim fun index =>
        Tensor.unstack tensor ⟨length - 1 - index.val, by grind⟩
  | axis + 1, .dim _ rest, tensor, h =>
      have innerAxis : Shape.AxisInBounds axis rest :=
        ⟨by
          have := h.proof
          simp only [Shape.rank] at this
          grind⟩
      Tensor.dim fun index =>
        @reverseAxis α _ axis rest (Tensor.unstack tensor index) innerAxis

/--
Flatten a tensor into a one-dimensional row-major vector.

Execution reuses the original native buffer; only the certified static shape
changes.
-/
def flattenSpec {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape) :
    Tensor α [shape.size] :=
  TorchLean.Tensor.Internal.Rep.reshape (size_toList_flatten shape) tensor

/--
Restore a row-major vector to a specified shape with the same element count.

Execution reuses the vector's native buffer.
-/
def unflattenSpec {α : Type} [TorchLean.Storage α]
    (shape : Shape) (tensor : Tensor α [shape.size]) :
    Tensor α shape :=
  TorchLean.Tensor.Internal.Rep.reshape (size_toList_flatten shape).symm tensor

namespace ShapeChange
namespace Internal

/--
Flattening an outer dimension places each flattened slice in one contiguous
row-major segment.
-/
theorem flattenSpec_dim_apply {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape}
    (values : Fin n → Tensor α shape) (outer : Fin n)
    (inner : Fin shape.size)
    (hIndex :
      outer.val * shape.size + inner.val < (Shape.dim n shape).size) :
    getScalar (flattenSpec (Tensor.dim values))
        ⟨outer.val * shape.size + inner.val, hIndex⟩ =
      getScalar (flattenSpec (values outer)) inner := by
  rw [getScalar_eq_apply, getScalar_eq_apply]
  unfold flattenSpec
  rw [TorchLean.Tensor.Internal.Rep.reshape_apply_coordEquiv,
    TorchLean.Tensor.Internal.Rep.reshape_apply_coordEquiv]
  have hCoordinate :
      TorchLean.Tensor.Internal.Rep.reshapeCoordEquiv
          (size_toList_flatten (Shape.dim n shape))
          (⟨outer.val * shape.size + inner.val, hIndex⟩, PUnit.unit) =
        (outer,
          TorchLean.Tensor.Internal.Rep.reshapeCoordEquiv
            (size_toList_flatten shape) (inner, PUnit.unit)) := by
    apply TorchLean.Tensor.Internal.Coord.linearize_injective
    apply Fin.ext
    rw [reshapeCoordEquiv_linearize_val,
      TorchLean.Tensor.Internal.Coord.linearize_cons_val,
      reshapeCoordEquiv_linearize_val]
    rw [vectorCoordinate_linearize_val,
      vectorCoordinate_linearize_val]
    rw [Shape.internalSize_eq]
    simp [Nat.mul_comm, Nat.add_comm]
  rw [hCoordinate]
  exact TorchLean.Tensor.Internal.Rep.stack_apply values
    (outer, TorchLean.Tensor.Internal.Rep.reshapeCoordEquiv
      (size_toList_flatten shape) (inner, PUnit.unit))

end Internal
end ShapeChange

private theorem tensor_eq_of_buffer_eq {α : Type}
    [storage : TorchLean.Storage α] {shape : TorchLean.Tensor.Internal.Shape}
    {left right : TorchLean.Tensor.Internal.Rep α shape}
    (hBuffer : left.buffer = right.buffer) :
    left = right := by
  cases left
  cases right
  cases hBuffer
  rfl

private theorem cast_vector_buffer {α : Type}
    [TorchLean.Storage α] {sourceSize targetSize : Nat}
    (hSize : sourceSize = targetSize)
    (tensor : Tensor α [sourceSize]) :
    (hSize ▸ tensor).buffer = tensor.buffer := by
  cases hSize
  rfl

/-- Unflattening a flattened tensor returns the original tensor. -/
@[simp] theorem unflattenSpec_flattenSpec {α : Type}
    [TorchLean.Storage α] {shape : Shape} (tensor : Tensor α shape) :
    unflattenSpec shape (flattenSpec tensor) = tensor := by
  exact TorchLean.Tensor.Internal.Rep.reshape_symm_reshape (size_toList_flatten shape) tensor

/-- Flattening an unflattened vector returns the original vector. -/
@[simp] theorem flattenSpec_unflattenSpec {α : Type}
    [TorchLean.Storage α] {shape : Shape}
    (tensor : Tensor α [shape.size]) :
    flattenSpec (unflattenSpec shape tensor) = tensor := by
  exact TorchLean.Tensor.Internal.Rep.reshape_symm_reshape (size_toList_flatten shape).symm tensor

/-- Reshape a tensor while preserving its row-major scalar sequence. -/
def reshapeSpec {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    (hSize : source.size = target.size) :
    Tensor α target :=
  TorchLean.Tensor.Internal.Rep.reshape (Shape.internalSize_congr hSize) tensor

/-- Flattening a reshape returns the original flat data, transported by the size equality. -/
theorem flatten_reshapeSpec {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    (hSize : source.size = target.size) :
    flattenSpec (reshapeSpec tensor hSize) = hSize ▸ flattenSpec tensor := by
  apply tensor_eq_of_buffer_eq
  change tensor.buffer = (hSize ▸ flattenSpec tensor).buffer
  rw [cast_vector_buffer hSize (flattenSpec tensor)]
  rfl

/-- Reshaping to an equal-size shape and back preserves every tensor entry. -/
@[simp] theorem reshapeSpec_roundtrip {α : Type}
    [TorchLean.Storage α] {source target : Shape}
    (tensor : Tensor α source) (hSize : source.size = target.size) :
    reshapeSpec (reshapeSpec tensor hSize) hSize.symm = tensor := by
  exact TorchLean.Tensor.Internal.Rep.reshape_symm_reshape (Shape.internalSize_congr hSize) tensor

/-- Sequence a finite family of optional tensor slices. -/
def sequenceFin {α : Type} [TorchLean.Storage α]
    {shape : Shape} {n : Nat}
    (values : Fin n → Option (Tensor α shape)) :
    Option (Tensor α (.dim n shape)) :=
  match n with
  | 0 => some (Tensor.dim fun index => Fin.elim0 index)
  | n + 1 =>
      match values ⟨0, Nat.succ_pos n⟩ with
      | none => none
      | some first =>
          match sequenceFin (n := n) (fun index => values index.succ) with
          | none => none
          | some rest =>
              some (Tensor.dim fun index =>
                Fin.cases first (fun tail => Tensor.unstack rest tail) index)

end TorchLean.Tensor
