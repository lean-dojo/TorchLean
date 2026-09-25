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
# Real Tensor Proof Toolkit

This file is the `ℝ`-specialized proof layer companion to the spec tensor layer.

The tensor proof folder has two layers:

- `NN.Proofs.Tensor.Algebra` is backend-generic and proves semiring facts about tensor dot
  products and executable folds.
- this file works in `Spec` over `ℝ`, where calculus, norms, Frobenius products, and model-analysis
  lemmas live.

The statements use PyTorch-shaped names where that helps readers:

- `flattenR` / `unflattenR` give a `Fin (Spec.Shape.size s) → ℝ` view of `Tensor ℝ s`.
- lemmas relate `getScalar` views to `addSpec`, `scaleSpec`, etc.

We re-export tensor-specific helpers from `NN.Proofs.Tensor.Algebra` into the `Spec` namespace.
General list-fold lemmas retain their canonical `List` names.

## PyTorch correspondence / citations

- Flatten / reshape: `torch.flatten`, `torch.reshape`, and `Tensor.view`.
  https://pytorch.org/docs/stable/generated/torch.flatten.html
  https://pytorch.org/docs/stable/generated/torch.reshape.html
  https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
- “numel”: `tensor.numel()` corresponds to `Spec.Shape.size`.
  https://pytorch.org/docs/stable/generated/torch.Tensor.numel.html
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open scoped BigOperators

-- Re-export generic helpers (defined once in `Proofs.TensorAlgebra`) into `Spec.*`.
export Proofs.TensorAlgebra
  (add_finRange_foldl_add_zero foldl_tensorScalar_mulAdd foldl_matvec_scalar get2_eq get_eq)

/-! ## Algebraic instances

The pointwise `AddCommGroup` and `Module` instances on `Tensor α s` live in
`NN.Spec.Core.Tensor.Core`; over `ℝ` they give `Module ℝ (Tensor ℝ s)` directly.
-/

/-! ## 1D helpers -/

/-- `getScalar` distributes over pointwise addition (`addSpec`). -/
theorem getScalar_add_spec {n : Nat} (x y : Tensor ℝ [n]) :
    getScalar (addSpec x y) = fun i => getScalar x i + getScalar y i := by
  funext i
  simp [getScalar_eq_apply, addSpec, map2Spec]

/-- `getScalar` distributes over pointwise scaling (`scaleSpec`). -/
theorem getScalar_scale_spec {n : Nat} (x : Tensor ℝ [n]) (c : ℝ) :
    getScalar (scaleSpec x c) = fun i => getScalar x i * c := by
  funext i
  simp [getScalar_eq_apply, scaleSpec, mapSpec, Tensor.map]

/--
Flatten a tensor of shape `s` into a 1D view `Fin (Spec.Shape.size s) → ℝ`.

This is the proof layer counterpart of `TorchLean.Tensor.flattenSpec` specialized to `ℝ`. In
PyTorch terms it is the functional analogue of flattening a tensor and then indexing it linearly
(`torch.flatten`, `tensor.view(-1)`). See the spec file `NN/Spec/Core/TensorReductionShape.lean`
for the definitional flatten/unflatten interface.

Citations:
https://pytorch.org/docs/stable/generated/torch.flatten.html
https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
-/
def flattenR {s : Shape} (x : Tensor ℝ s) : Fin (Spec.Shape.size s) → ℝ :=
  getScalar (flattenSpec (α:=ℝ) x)

/-- Row-major equivalence between the coordinates of a shape and flat indices. Its forward map is
`Shape.Coord.linearize` and its inverse is `Shape.Coord.unlinearize`. -/
def Shape.Coord.equivFin (s : Shape) : s.Coord ≃ Fin s.size where
  toFun := Shape.Coord.linearize
  invFun := Shape.Coord.unlinearize
  left_inv := Shape.Coord.unlinearize_linearize
  right_inv := Shape.Coord.linearize_unlinearize

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

/--
Unflatten a 1D view `Fin (Spec.Shape.size s) → ℝ` back into a tensor of shape `s`.

This is the proof layer counterpart of `TorchLean.Tensor.unflattenSpec` specialized to `ℝ`, and is
intended to round-trip with `flattenR` under the spec lemmas in
`NN/Spec/Core/TensorReductionShape.lean`.
-/
def unflattenR {s : Shape} (v : Fin (Spec.Shape.size s) → ℝ) : Tensor ℝ s :=
  unflattenSpec (α:=ℝ) s (ofFn v)

/-! ## Pointwise tensor algebra -/


end Spec
