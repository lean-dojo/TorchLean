/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Functional.Ops
public import NN.Runtime.Autograd.Batch

/-!
# Tensor Operations over Backend References

Arithmetic, shape changes, indexing, reductions, and seeded data for programs polymorphic over
`Ops`. `mapBatch` supplies the shared traversal when a backend has no fused batch operation.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

-- Re-export the capability operations; only argument adapters and compositions need bodies.
export Ops
  (dataConst mapData const add sub mul div neg scale abs sqrt clamp max min broadcastTo reshape
    swapAdjacentAtDepth reduceSum matmul concat slice detach randUniform bernoulliMask sum
    flatten)

variable {m : Type → Type} {α : Type} [Storage α] [Context α] [Monad m]
    [Ops (m := m) (α := α)]

/-- Restore an axis removed by a reduction and repeat the reduced value along that axis. -/
def broadcastAfterSum (s : Shape) (axis : Nat)
    (x : Ref (m := m) (α := α) (shapeAfterSum s axis)) :
    m (Ref (m := m) (α := α) s) := do
  let kept ← reshape (m := m) (α := α)
    (s₁ := shapeAfterSum s axis) (s₂ := shapeAfterSumKeepDim s axis) x
    (size_shapeAfterSumKeepDim s axis).symm
  broadcastTo (m := m) (α := α) (shapeAfterSumKeepDimBroadcast s axis) kept

@[inherit_doc Ops.reduceMean]
def reduceMean {s : Shape} (axis : Nat) (x : Ref (m := m) (α := α) s)
    [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s] :
    m (Ref (m := m) (α := α) (shapeAfterSum s axis)) :=
  Ops.reduceMean (m := m) (α := α) (s := s) axis x

@[inherit_doc Ops.select]
def select {s : Shape} (axis : Nat) (x : Ref (m := m) (α := α) s)
    [Shape.AxisInBounds axis s] (index : Fin (Shape.axisSize s axis)) :
    m (Ref (m := m) (α := α) (s.eraseAxis axis)) :=
  Ops.select (m := m) (α := α) (s := s) axis x index

@[inherit_doc Ops.indexSelect]
def indexSelect {s : Shape} (axis count : Nat) (x : Ref (m := m) (α := α) s)
    [Shape.AxisInBounds axis s]
    (indices : DataRef (m := m) (α := α) (Fin (Shape.axisSize s axis)) [count]) :
    m (Ref (m := m) (α := α) (s.replaceAxis axis count)) :=
  Ops.indexSelect (m := m) (α := α) (s := s) axis count x indices

@[inherit_doc Ops.scatterAdd]
def scatterAdd {s : Shape} (axis count : Nat) (base : Ref (m := m) (α := α) s)
    [Shape.AxisInBounds axis s]
    (source : Ref (m := m) (α := α) (s.replaceAxis axis count))
    (indices : DataRef (m := m) (α := α) (Fin (Shape.axisSize s axis)) [count]) :
    m (Ref (m := m) (α := α) s) :=
  Ops.scatterAdd (m := m) (α := α) (s := s) axis count base source indices

/--
Apply `f` to each leading-axis slice and concatenate the results in order.

An empty leading axis returns an empty reference without calling `f`.
-/
def mapBatch {σ τ : Shape}
    (f : Ref (m := m) (α := α) σ → m (Ref (m := m) (α := α) τ)) {n : Nat}
    (x : Ref (m := m) (α := α) (σ.prependDim n)) :
    m (Ref (m := m) (α := α) (τ.prependDim n)) :=
  Runtime.Autograd.mapBatch
    (const (m := m) (α := α) (s := τ.prependDim 0) (Tensor.full (τ.prependDim 0) (0 : α)))
    (fun x start len h => slice (m := m) (α := α) start len h x)
    (reshape (m := m) (α := α))
    (concat (m := m) (α := α))
    f x

end Runtime.Autograd.Torch
