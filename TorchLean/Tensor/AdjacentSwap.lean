/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Rep.Segment

@[expose] public section


namespace TorchLean.Tensor

/-!
# Adjacent-axis swaps

`permute` by `Shape.adjacentSwap s depth` exchanges the axes `depth` and `depth + 1`, and
`permute_adjacentSwap_zero` and `permute_adjacentSwap_succ` unfold it by recursion on the shape.
-/

/-- Exchanging the two leading axes moves entry `(i, j)` of `tensor` to `(j, i)`. -/
theorem permute_adjacentSwap_zero {β : Type} [TorchLean.Storage β] {m n : Nat} {s : Shape}
    (tensor : Tensor β (.dim m (.dim n s)))
    (hout : Shape.dim n (.dim m s) =
      Shape.permute (.dim m (.dim n s)) (Shape.adjacentSwap (.dim m (.dim n s)) 0)) :
    permute tensor (Shape.adjacentSwap (.dim m (.dim n s)) 0) (.dim n (.dim m s)) hout =
      .dim fun j => .dim fun i => Tensor.unstack (Tensor.unstack tensor i) j := by
  apply Rep.ext
  intro c
  rw [permute_adjacentSwap_zero_apply]
  simp

/-- Swapping at depth `depth + 1` swaps at depth `depth` inside every slice of the outer axis. -/
theorem permute_adjacentSwap_succ {β : Type} [TorchLean.Storage β] {n : Nat} {s : Shape}
    (tensor : Tensor β (.dim n s)) (depth : Nat)
    (hout : Shape.dim n (s.swapAdjacentAtDepth depth) =
      Shape.permute (.dim n s) (Shape.adjacentSwap (.dim n s) (depth + 1))) :
    permute tensor (Shape.adjacentSwap (.dim n s) (depth + 1))
        (.dim n (s.swapAdjacentAtDepth depth)) hout =
      .dim fun i =>
        permute (Tensor.unstack tensor i) (s.adjacentSwap depth) (s.swapAdjacentAtDepth depth) := by
  apply Rep.ext
  intro c
  rw [permute_adjacentSwap_apply, cast_eq]
  simp only [Tensor.dim_apply, permute_swapAdjacentAtDepth_apply,
    Tensor.unstack_apply]
  rfl

end TorchLean.Tensor
