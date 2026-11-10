/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Ring.Nat -- shake: keep
public import TorchLean.Tensor.Algebra -- shake: keep
public import TorchLean.Tensor.ConcatSlice -- shake: keep
public import TorchLean.Tensor.Contract

/-!
# Tensor coordinates, flattening and reshapes

Coordinate formulas for vector addition, subtraction and scaling, the row-major flattening rules
and the constant-tensor reshape rule, for any storage scalar. `Shape.Coord.equivFin` identifies the
coordinates of a shape `s` with the flat indices below `Shape.size s`, the product of the extents
of `s`, in row-major order: the last axis varies fastest. This includes empty shapes with a
zero-length dimension.
-/

@[expose] public section

open TorchLean.Shape.Coord (linearize)

namespace TorchLean.Tensor

open Shape.Coord
open scoped BigOperators
open Shape (Coord)

/-! ## Vector coordinates -/

/-- `getScalar` distributes over pointwise addition (`+`). -/
theorem getScalar_add {α : Type} [Storage α] [Add α] {n : Nat} (x y : Tensor α [n]) :
    getScalar (x + y) = fun i => getScalar x i + getScalar y i := by
  funext i
  simp [getScalar_eq_apply, add_def]

/-- Coordinate `i` of a difference of vectors is the difference of the coordinates. -/
theorem getScalar_sub {α : Type} [Storage α] [Sub α] {n : Nat} (x y : Tensor α [n])
    (i : Fin n) : (x - y).getScalar i = x.getScalar i - y.getScalar i := by
  simp [getScalar_eq_apply, sub_def]

/-- `getScalar` distributes over pointwise scaling (`Tensor.scale`). -/
theorem getScalar_scale {α : Type} [Storage α] [Mul α] {n : Nat} (x : Tensor α [n]) (c : α) :
    getScalar (Tensor.scale x c) = fun i => getScalar x i * c := by
  funext i
  simp [getScalar_eq_apply, Tensor.scale]

/-- Reading a flattened tensor at the row-major position of a coordinate returns that entry. -/
theorem getScalar_reshape_toFin {α : Type} [Storage α] {s : Shape} (x : Tensor α s)
    (c : s.Coord) :
    getScalar (x.reshape [_] (Shape.size_vector _).symm) (Coord.linearize c) = x c := by
  rw [getScalar_eq_apply]
  unfold Tensor.reshape
  rw [Tensor.Rep.reshape_apply_reshapeEquiv]
  congr 1
  apply Coord.linearize_injective
  apply Fin.ext
  rw [Coord.linearize_reshapeEquiv_val, val_linearize_vector]

/-- A zero-padded read of a flattened tensor at the row-major position of a coordinate returns that
entry. -/
theorem getAtOrZero_reshape_toFin {α : Type} [Storage α] [Zero α] {s : Shape}
    (x : Tensor α s) (c : s.Coord) :
    getAtOrZero (x.reshape [_] (Shape.size_vector _).symm) [(Coord.linearize c).val] =
      x c := by
  rw [getAtOrZero_eq_getScalar]
  exact getScalar_reshape_toFin x c

/-- Entry `index` of the flattened tensor is the entry of `tensor` at the shaped coordinate whose
row-major position is `index`. -/
theorem getScalar_reshape_equivFin {α : Type} [Storage α]
    {shape : Shape} (tensor : Tensor α shape) (index : Fin shape.size) :
    Tensor.getScalar (Tensor.reshape tensor [_] (Shape.size_vector _).symm) index =
      tensor ((Coord.equivFin shape).symm index) := by
  obtain ⟨c, rfl⟩ := (Coord.equivFin shape).surjective index
  rw [Equiv.symm_apply_apply]
  exact Tensor.getScalar_reshape_toFin tensor c

/-- Reshaping the constant tensor `Tensor.full s a` to a shape `t` of the same size gives
`Tensor.full t a`. -/
theorem reshape_full {α : Type} [Storage α] {s t : Shape} (a : α) (h : s.size = t.size) :
    Tensor.reshape (Tensor.full s a) _ h = Tensor.full t a := by
  apply Tensor.Rep.ext
  intro c
  simp only [Tensor.reshape, Tensor.Rep.reshape_apply_reshapeEquiv,
    Tensor.full_apply]

end TorchLean.Tensor
