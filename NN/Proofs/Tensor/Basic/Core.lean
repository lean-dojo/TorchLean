/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import Mathlib.Algebra.BigOperators.Ring.Nat -- shake: keep
public import NN.Proofs.Tensor.Algebra -- shake: keep
public import NN.Spec.Core.TensorReductionShape.ConcatSlice -- shake: keep
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra -- shake: keep

/-!
# Tensor Coordinate and Shape Bridges

Coordinate formulas for real vector addition and scaling, together with row-major flattening and
constant-tensor reshape rules for any storage scalar. `Shape.Coord.equivFin` identifies shaped
coordinates with flat indices, including empty shapes with a zero-length dimension.

The generic matrix/vector indexing helpers are re-exported from `NN.Proofs.Tensor.Algebra` into
`Spec`. General fold lemmas retain their canonical `List` names.

The shape conventions match
[PyTorch flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html)
and [reshape](https://pytorch.org/docs/stable/generated/torch.reshape.html). `Spec.Shape.size`
corresponds to [numel](https://pytorch.org/docs/stable/generated/torch.Tensor.numel.html).
-/

@[expose] public section

open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open scoped BigOperators

export Proofs.TensorAlgebra
  (get2_eq get_eq getScalar_mat_vec_mul_spec getScalar_vec_mat_mul_spec)

/-! ## Vector coordinates -/

/-- `getScalar` distributes over pointwise addition (`addSpec`). -/
theorem getScalar_add_spec {n : Nat} (x y : Tensor ℝ [n]) :
    getScalar (addSpec x y) = fun i => getScalar x i + getScalar y i := by
  funext i
  simp [getScalar_eq_apply, addSpec, map2Spec]

/-- `getScalar` distributes over pointwise scaling (`scaleSpec`). -/
theorem getScalar_scale_spec {n : Nat} (x : Tensor ℝ [n]) (c : ℝ) :
    getScalar (scaleSpec x c) = fun i => getScalar x i * c := by
  funext i
  simp [getScalar_eq_apply, scaleSpec, Tensor.map]

/-- Row-major equivalence between the coordinates of a shape and flat indices. Its forward map is
`Shape.Coord.linearize` and its inverse is `Shape.Coord.unlinearize`. -/
def Shape.Coord.equivFin (s : Shape) : s.Coord ≃ Fin s.size where
  toFun := Shape.Coord.linearize
  invFun := Shape.Coord.unlinearize
  left_inv := Shape.Coord.unlinearize_linearize
  right_inv := Shape.Coord.linearize_unlinearize

/-- The forward map of `Shape.Coord.equivFin` is row-major linearization. -/
theorem Shape.Coord.equivFin_apply {s : Shape} (c : s.Coord) :
    Shape.Coord.equivFin s c = Shape.Coord.linearize c := rfl

/-- Reading a flattened tensor at the row-major position of a coordinate returns that entry. -/
theorem getScalar_flattenSpec_linearize {α : Type} [Storage α] {s : Shape} (x : Tensor α s)
    (c : s.Coord) : getScalar (flattenSpec x) (Shape.Coord.linearize c) = x c := by
  rw [getScalar_eq_apply]
  unfold flattenSpec
  rw [TorchLean.Tensor.Internal.Rep.reshape_apply_coordEquiv]
  congr 1
  apply TorchLean.Tensor.Internal.Coord.linearize_injective
  apply Fin.ext
  rw [reshapeCoordEquiv_linearize_val, vectorCoordinate_linearize_val]
  rfl

/-- A scalar in the flattened tensor is read at its row-major shaped coordinate. -/
theorem getScalar_flattenSpec {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape) (index : Fin shape.size) :
    Tensor.getScalar (Tensor.flattenSpec tensor) index =
      tensor (((Tensor.Internal.Coord.equivFin shape).trans
        (finCongr (Shape.internalSize_eq shape))).symm index) := by
  obtain ⟨c, rfl⟩ := ((Tensor.Internal.Coord.equivFin shape).trans
    (finCongr (Shape.internalSize_eq shape))).surjective index
  rw [Equiv.symm_apply_apply]
  exact Spec.getScalar_flattenSpec_linearize tensor c

/-- A shape change preserves a constant tensor's fill value. -/
theorem reshapeSpec_full {α : Type} [Storage α] {s t : Shape} (a : α) (h : s.size = t.size) :
    Tensor.reshapeSpec (Tensor.full s a) h = Tensor.full t a := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro c
  simp only [Tensor.reshapeSpec, TorchLean.Tensor.Internal.Rep.reshape_apply_coordEquiv,
    Tensor.full_apply]

end Spec
