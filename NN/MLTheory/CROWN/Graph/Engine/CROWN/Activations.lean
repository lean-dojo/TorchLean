/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Linear

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [TorchLean.Storage α] [Context α]
variable [BoundOps α]

open BoundOps

/-!
# CROWN Activation Relaxations

Affine transfer rules for nonlinear scalar activations.
-/

/-- Propagate CROWN bounds through ReLU using standard per-neuron triangle relaxations. -/
def propagateReluBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α := by
  let relaxHi0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  let relaxLo0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVectorLower (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let loAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxLo0 xLo
  let hiAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxHi0 xHi
  exact
    { inDim := xB.inDim
      outDim := preB.dim
      loAff := loAff
      hiAff := hiAff }

/-- Propagate ReLU bounds with externally supplied α slopes for crossing neurons. -/
def propagateReluBoundsWithAlpha
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim)
  (alpha : Tensor α [preB.dim]) : FlatAffineBounds α := by
  -- Upper relaxation: standard secant/tight bounds (independent of α).
  let relaxHi0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  -- Lower relaxation: for crossing bounds l < 0 < u, use a provided per-neuron α ∈ [0,1]
  -- (line y ≥ α x), which is always sound for ReLU; stable regions override α.
  let clamp01 (x : α) : α :=
    let x0 := if x > 0 then x else 0
    if x0 > 1 then 1 else x0
  let relaxLo0 : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [preB.dim] :=
    Tensor.ofFn fun i =>
      let l := Tensor.getScalar preB.lo i
      let u := Tensor.getScalar preB.hi i
      let a := Tensor.getScalar alpha i
      if u > 0 then
        if l > 0 then
          { slope := 1, bias := 0 }
        else
          { slope := clamp01 a, bias := 0 }
      else
        { slope := 0, bias := 0 }
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let loAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxLo0 xLo
  let hiAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxHi0 xHi
  exact
    { inDim := xB.inDim
      outDim := preB.dim
      loAff := loAff
      hiAff := hiAff }

end NN.MLTheory.CROWN.Graph
