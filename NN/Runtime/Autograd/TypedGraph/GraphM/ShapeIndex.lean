/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM.Core

/-!
# GraphM Shape And Indexing Ops

Reshape, transpose, broadcast, reduction, gather, and scatter builders for typed graphs.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TypedGraph
namespace GraphM

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/--
Flatten a tensor to a 1D vector (preserving total size).

PyTorch comparison: `torch.flatten(x)` (for a single tensor value).
-/
def flatten {α : Type} {Δ : Type} [TorchLean.Storage α] [Inhabited α] [Zero α]
  {Γ : List Shape} {s : Shape} (x : Var s) :
  MWith α Δ Γ (Var (.dim (Spec.Shape.size s) .scalar)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim (Spec.Shape.size s) .scalar
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d => flattenSpec (α := α) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        flattenSpec (α := α) (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TensorPack.single (α := α) (Γ := Γ ++ ss) (s := s) ix (unflattenSpec (α := α) s δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Reshape a tensor, given a proof that the total sizes match.

PyTorch comparison: `torch.reshape(x, new_shape)`.
-/
def reshape {α : Type} {Δ : Type} [TorchLean.Storage α] [Inhabited α] [Zero α]
  {Γ : List Shape} {s₁ s₂ : Shape} (x : Var s₁) (h : Spec.Shape.size s₁ = Spec.Shape.size s₂) :
    MWith α Δ Γ (Var s₂) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s₂ :=
    { forward := fun ctx _d =>
        TorchLean.Tensor.reshapeSpec (α := α) (source := s₁) (target := s₂)
          (getIdx (α := α) (xs := ctx) ix) h
      jvp := fun _ctx dctx _d =>
        TorchLean.Tensor.reshapeSpec (α := α) (source := s₁) (target := s₂)
          (getIdx (α := α) (xs := dctx) ix) h
      vjp := fun _ctx _d δ =>
        TensorPack.single (α := α) (Γ := Γ ++ ss) (s := s₁) ix
          (TorchLean.Tensor.reshapeSpec (α := α) (source := s₂) (target := s₁) δ h.symm) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s₂) g node

/--
Swap two adjacent axes at a given nesting `depth`.

This is the typed-graph primitive used to lower arbitrary permutations.
-/
def swapAdjacentAtDepth {α : Type} {Δ : Type} [TorchLean.Storage α] [Zero α]
  {Γ : List Shape} {s : Shape} (depth : Nat) (x : Var s) :
    MWith α Δ Γ (Var (s.swapAdjacentAtDepth depth)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := s.swapAdjacentAtDepth depth
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        TorchLean.Tensor.swapAdjacentAxes (tensor := getIdx (α := α) (xs := ctx) ix) depth
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        TorchLean.Tensor.swapAdjacentAxes (tensor := dx) depth
      vjp := fun _ctx _d δ =>
        let dx' := TorchLean.Tensor.swapAdjacentAxes (tensor := δ) depth
        let dx : Tensor α s :=
          Tensor.castShape dx' (by simp [outS])
        TensorPack.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Broadcast `x : s₁` to a larger shape `s₂` (given a `CanBroadcastTo` witness).

PyTorch comparison: `x.expand(...)` / broadcasting semantics in elementwise ops.
-/
def broadcastTo {α : Type} {Δ : Type} [TorchLean.Storage α] [Inhabited α] [Add α] [Zero α]
  {Γ : List Shape} {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) (x : Var s₁) :
  MWith α Δ Γ (Var s₂) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s₂ :=
    { forward := fun ctx _d =>
        TorchLean.Tensor.broadcastTo (α := α) cb (getIdx (α := α) (xs := ctx) ix)
      jvp := fun _ctx dctx _d =>
        TorchLean.Tensor.broadcastTo (α := α) cb (getIdx (α := α) (xs := dctx) ix)
      vjp := fun _ctx _d δ =>
        TensorPack.single (α := α) (Γ := Γ ++ ss) (s := s₁) ix
          (TorchLean.Tensor.reduceFromBroadcastTo (α := α) cb δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s₂) g node

/--
Reduce-sum along a given `axis`.

PyTorch comparison: `torch.sum(x, dim=axis)`.
-/
def reduceSum {α : Type} {Δ : Type} [TorchLean.Storage α] [Add α] [Zero α] [Inhabited α]
  {Γ : List Shape} {s : Shape} (axis : Nat)
  [_valid : Shape.HasNonemptyAxis axis s] [_wf : Shape.WellFormed s]
  (x : Var s) : MWith α Δ Γ (Var (shapeAfterSum s axis)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := shapeAfterSum s axis
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        TorchLean.Tensor.reduceSum (α := α) (s := s) axis
          (getIdx (α := α) (xs := ctx) ix) _valid.proof
      jvp := fun _ctx dctx _d =>
        TorchLean.Tensor.reduceSum (α := α) (s := s) axis
          (getIdx (α := α) (xs := dctx) ix) _valid.proof
      vjp := fun _ctx _d δ =>
        TensorPack.single (α := α) (Γ := Γ ++ ss) (s := s) ix
          (TorchLean.Tensor.broadcastAfterSum s axis δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Reduce-mean along a given `axis`.

PyTorch comparison: `torch.mean(x, dim=axis)`.
-/
def reduceMean {α : Type} {Δ : Type} [TorchLean.Storage α] [Context α]
  {Γ : List Shape} {s : Shape} (axis : Nat)
  [valid : Shape.HasNonemptyAxis axis s] [_wf : Shape.WellFormed s]
  (x : Var s) : MWith α Δ Γ (Var (shapeAfterSum s axis)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := shapeAfterSum s axis
  letI : Shape.AxisInBounds axis s := valid.proof.toAxisInBounds
  let denomNat := Shape.axisSize s axis
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let h := valid.proof
        TorchLean.Tensor.reduceMean (α := α) (s := s) axis xv h
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        let h := valid.proof
        TorchLean.Tensor.reduceMean (α := α) (s := s) axis dx h
      vjp := fun _ctx _d δ =>
        let dLdx := TorchLean.Tensor.broadcastAfterSum s axis δ
        let dLdx' := scaleSpec (α := α) (s := s) dLdx (1 / (denomNat : α))
        TensorPack.single (α := α) (Γ := Γ ++ ss) (s := s) ix dLdx' }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/-! ## Indexing -/

/-- Select one bounded coordinate from an arbitrary tensor axis. -/
def select {α : Type} {Δ : Type} [TorchLean.Storage α] [Zero α]
    {Γ : List Shape} {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Var s) (index : Fin (Shape.axisSize s axis)) :
    MWith α Δ Γ (Var (s.eraseAxis axis)) := do
  let ⟨ss, graph⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) (s.eraseAxis axis) :=
    { forward := fun context _ =>
        Tensor.selectSpec axis (getIdx (α := α) (xs := context) ix) index
      jvp := fun _ tangent _ =>
        Tensor.selectSpec axis (getIdx (α := α) (xs := tangent) ix) index
      vjp := fun _ _ delta =>
        TensorPack.single (α := α) (Γ := Γ ++ ss) (s := s) ix
          (Tensor.selectBackwardSpec axis index delta) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s.eraseAxis axis) graph node

/-- Select several bounded coordinates from an arbitrary tensor axis. -/
def indexSelect {α : Type} {Δ : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {Γ : List Shape} {s : Shape} (axis count : Nat) [Shape.AxisInBounds axis s]
    (x : Var s) (indices : Δ → Tensor (Fin (Shape.axisSize s axis)) [count]) :
    MWith α Δ Γ (Var (s.replaceAxis axis count)) := do
  let ⟨ss, graph⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) (s.replaceAxis axis count) :=
    { forward := fun context data =>
        Tensor.indexSelectSpec axis (getIdx (α := α) (xs := context) ix) (indices data)
      jvp := fun _ tangent data =>
        Tensor.indexSelectSpec axis (getIdx (α := α) (xs := tangent) ix) (indices data)
      vjp := fun _ data delta =>
        let dx := Tensor.scatterAddSpec axis (Tensor.full s (0 : α)) (indices data) delta
        TensorPack.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s.replaceAxis axis count) graph node

/-- Add source slices into an arbitrary tensor axis at bounded coordinates. -/
def scatterAdd {α : Type} {Δ : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {Γ : List Shape} {s : Shape} (axis count : Nat) [Shape.AxisInBounds axis s]
    (base : Var s) (source : Var (s.replaceAxis axis count))
    (indices : Δ → Tensor (Fin (Shape.axisSize s axis)) [count]) :
    MWith α Δ Γ (Var s) := do
  let ⟨ss, graph⟩ ← get
  let ibase ← liftM (mkIdx (_α := α) (Γ := Γ) ss base)
  let isource ← liftM (mkIdx (_α := α) (Γ := Γ) ss source)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun context data =>
        Tensor.scatterAddSpec axis (getIdx (α := α) (xs := context) ibase) (indices data)
          (getIdx (α := α) (xs := context) isource)
      jvp := fun _ tangent data =>
        Tensor.scatterAddSpec axis (getIdx (α := α) (xs := tangent) ibase) (indices data)
          (getIdx (α := α) (xs := tangent) isource)
      vjp := fun _ data delta =>
        TorchLean.TensorPack.add (α := α) (ss := Γ ++ ss)
          (TensorPack.single (α := α) (Γ := Γ ++ ss) (s := s) ibase delta)
          (TensorPack.single (α := α) (Γ := Γ ++ ss)
            (s := s.replaceAxis axis count) isource
            (Tensor.indexSelectSpec axis delta (indices data))) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) graph node

end GraphM
end TypedGraph
end Autograd
end Runtime
