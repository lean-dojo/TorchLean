/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Functional.Ops
public import NN.Runtime.Autograd.LeadingAxis

/-!
# Tensor Operations over Backend References

Arithmetic, shape changes, indexing, reductions, and seeded data for programs polymorphic over
`Ops`. `mapOuterAxis` supplies the shared traversal when a backend has no fused batch operation.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

variable {m : Type → Type} {α : Type} [Storage α] [Context α] [Monad m]
    [Ops (m := m) (α := α)]

@[inherit_doc Ops.dataConst]
def dataConst {β : Type} [Storage β] {s : Shape}
    (x : Tensor β s) : DataRef (m := m) (α := α) β s :=
  Ops.dataConst (m := m) (α := α) x

@[inherit_doc Ops.mapData]
def mapData {β γ : Type} [Storage β] [Storage γ]
    {s₁ s₂ : Shape} (f : Tensor β s₁ → Tensor γ s₂)
    (x : DataRef (m := m) (α := α) β s₁) : DataRef (m := m) (α := α) γ s₂ :=
  Ops.mapData (m := m) (α := α) f x

@[inherit_doc Ops.const]
def const {s : Shape} (t : Tensor α s) : m (Ref (m := m) (α := α) s) :=
  Ops.const (m := m) (α := α) t

@[inherit_doc Ops.add]
def add {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.add (m := m) (α := α) a b

@[inherit_doc Ops.sub]
def sub {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.sub (m := m) (α := α) a b

@[inherit_doc Ops.mul]
def mul {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.mul (m := m) (α := α) a b

@[inherit_doc Ops.scale]
def scale {s : Shape} (x : Ref (m := m) (α := α) s) (c : α) : m (Ref (m := m) (α := α) s) :=
  Ops.scale (m := m) (α := α) x c

@[inherit_doc Ops.abs]
def abs {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.abs (m := m) (α := α) x

@[inherit_doc Ops.sqrt]
def sqrt {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.sqrt (m := m) (α := α) x

@[inherit_doc Ops.clamp]
def clamp {s : Shape} (x : Ref (m := m) (α := α) s) (minVal maxVal : α) :
    m (Ref (m := m) (α := α) s) :=
  Ops.clamp (m := m) (α := α) (s := s) x minVal maxVal

@[inherit_doc Ops.max]
def max {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.max (m := m) (α := α) a b

@[inherit_doc Ops.min]
def min {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.min (m := m) (α := α) a b

@[inherit_doc Ops.broadcastTo]
def broadcastTo {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂)
    (x : Ref (m := m) (α := α) s₁) : m (Ref (m := m) (α := α) s₂) :=
  Ops.broadcastTo (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) cb x

@[inherit_doc Ops.reshape]
def reshape {s₁ s₂ : Shape} (x : Ref (m := m) (α := α) s₁)
    (h : Shape.size s₁ = Shape.size s₂) : m (Ref (m := m) (α := α) s₂) :=
  Ops.reshape (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) x h

/-- Restore an axis removed by a reduction and repeat the reduced value along that axis. -/
def broadcastAfterSum (s : Shape) (axis : Nat)
    (x : Ref (m := m) (α := α) (shapeAfterSum s axis)) :
    m (Ref (m := m) (α := α) s) := do
  let kept ← reshape (m := m) (α := α)
    (s₁ := shapeAfterSum s axis) (s₂ := shapeAfterSumKeepDim s axis) x
    (size_shapeAfterSumKeepDim s axis).symm
  broadcastTo (m := m) (α := α) (shapeAfterSumKeepDimBroadcast s axis) kept

@[inherit_doc Ops.swapAdjacentAtDepth]
def swapAdjacentAtDepth {s : Shape} (depth : Nat)
    (x : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) (s.swapAdjacentAtDepth depth)) :=
  Ops.swapAdjacentAtDepth (m := m) (α := α) (s := s) depth x

@[inherit_doc Ops.reduceSum]
def reduceSum {s : Shape} (axis : Nat) [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s]
    (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) (shapeAfterSum s axis)) :=
  Ops.reduceSum (m := m) (α := α) (s := s) axis x

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

@[inherit_doc Ops.matmul]
def matmul {batchA batchB batch : Shape} {mDim nDim pDim : Nat}
    [Shape.BroadcastTo batchA batch] [Shape.BroadcastTo batchB batch]
    (a : Ref (m := m) (α := α) (batchA.concat [mDim, nDim]))
    (b : Ref (m := m) (α := α) (batchB.concat [nDim, pDim])) :
    m (Ref (m := m) (α := α) (batch.concat [mDim, pDim])) :=
  Ops.matmul (m := m) (α := α) (batchA := batchA) (batchB := batchB)
    (batch := batch) (mDim := mDim) (nDim := nDim) (pDim := pDim) a b

@[inherit_doc Ops.concatLeadingAxis]
def concatLeadingAxis {nDim mDim : Nat} {s : Shape}
    (a : Ref (m := m) (α := α) (s.prependDim nDim))
    (b : Ref (m := m) (α := α) (s.prependDim mDim)) :
    m (Ref (m := m) (α := α) (s.prependDim (nDim + mDim))) :=
  Ops.concatLeadingAxis (m := m) (α := α) (nDim := nDim) (mDim := mDim) (s := s) a b

@[inherit_doc Ops.sliceLeadingAxisRange]
def sliceLeadingAxisRange {nDim : Nat} {s : Shape} (start len : Nat) (h : start + len ≤ nDim)
    (x : Ref (m := m) (α := α) (s.prependDim nDim)) :
    m (Ref (m := m) (α := α) (s.prependDim len)) :=
  Ops.sliceLeadingAxisRange (m := m) (α := α) (nDim := nDim) (s := s) start len h x

/--
Apply `f` to each leading-axis slice and concatenate the results in order.

An empty leading axis returns an empty reference without calling `f`.
-/
def mapOuterAxis {σ τ : Shape}
    (f : Ref (m := m) (α := α) σ → m (Ref (m := m) (α := α) τ)) {n : Nat}
    (x : Ref (m := m) (α := α) (σ.prependDim n)) :
    m (Ref (m := m) (α := α) (τ.prependDim n)) :=
  Runtime.Autograd.mapOuterAxisWith
    (const (m := m) (α := α) (s := τ.prependDim 0) (Tensor.full (τ.prependDim 0) (0 : α)))
    (fun x start len h => sliceLeadingAxisRange (m := m) (α := α) start len h x)
    (reshape (m := m) (α := α))
    (concatLeadingAxis (m := m) (α := α))
    f x

@[inherit_doc Ops.detach]
def detach {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.detach (m := m) (α := α) x

@[inherit_doc Ops.randUniform]
def randUniform {s : Shape} (seed : Nat) : m (Ref (m := m) (α := α) s) :=
  Ops.randUniform (m := m) (α := α) (s := s) seed

@[inherit_doc Ops.bernoulliMask]
def bernoulliMask {s : Shape} (keepProb : Ref (m := m) (α := α) Shape.scalar) (seed : Nat) :
    m (Ref (m := m) (α := α) s) :=
  Ops.bernoulliMask (m := m) (α := α) (s := s) keepProb seed

@[inherit_doc Ops.sum]
def sum {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.sum (m := m) (α := α) (s := s) x

@[inherit_doc Ops.flatten]
def flatten {s : Shape} (x : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) [Spec.Shape.size s]) :=
  Ops.flatten (m := m) (α := α) (s := s) x

end Runtime.Autograd.Torch
