/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM.Core
public import NN.Tensor.Conversion
public import NN.Spec.Layers.Pooling.Spatial

/-!
# GraphM Pooling Ops

Rank-generic pooling builders with forward, JVP, and VJP payloads.
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
N-D max pooling (channels-first) on a single sample tensor (no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on
the spatial rank `d`.

Forward-mode status: implemented as the selected-branch linearization from
`Spec.maxPoolLinearizationSpec`. At ties this follows the documented first-winner convention.
-/
def maxPool {α : Type} {Δ : Type} [TorchLean.Storage α] [Context α]
  {Γ : List Shape} {d C : Nat}
  {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
  (x : Var (Shape.ofList (C :: Tensor.to inSpatial (List Nat)))) :
  MWith α Δ Γ (Var (Shape.ofList
    (C :: Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))) := do
  if hKernel : (∀ i : Fin d, kernel.getScalar i ≠ 0) then
    if hStride : (∀ i : Fin d, stride.getScalar i ≠ 0) then
      let ⟨ss, g⟩ ← get
      let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
      let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
      let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
      let outShape : Shape := Shape.ofList (C :: Tensor.to outSpatial (List Nat))
      let inShape : Shape := Shape.ofList (C :: Tensor.to inSpatial (List Nat))
      let node : NodeData α Δ (Γ ++ ss) outShape :=
        { forward := fun ctx _d =>
            let xv := getIdx (α := α) (xs := ctx) ix
            Spec.maxPoolSpec (α := α) (d := d) (C := C)
              (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
              (layer := layer) xv
          jvp := fun ctx dctx _d =>
            let xv := getIdx (α := α) (xs := ctx) ix
            let dx := getIdx (α := α) (xs := dctx) ix
            Spec.maxPoolLinearizationSpec (α := α) (d := d) (C := C)
              (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
              (layer := layer) xv dx
          vjp := fun ctx _d δ =>
            let xv := getIdx (α := α) (xs := ctx) ix
            let dx :=
              Spec.maxPoolBackwardSpec (α := α) (d := d) (C := C)
                (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
                (layer := layer) (input := xv) (gradOutput := δ)
            TensorPack.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
      push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
    else
      throw "typed GraphM: max_pool requires stride > 0 on every spatial axis"
  else
    throw "typed GraphM: max_pool requires kernel_size > 0 on every spatial axis"

/--
N-D average pooling (channels-first) on a single sample tensor (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on
the spatial rank `d`.

  Forward-mode status: implemented. Average pooling is linear, so the JVP is the same average-pool
  map applied to the input tangent.
-/
def avgPool {α : Type} {Δ : Type} [TorchLean.Storage α] [Context α]
  {Γ : List Shape} {d C : Nat}
  {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
  (x : Var (Shape.ofList (C :: Tensor.to inSpatial (List Nat)))) :
  MWith α Δ Γ (Var (Shape.ofList
    (C :: Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))) := do
  if hKernel : (∀ i : Fin d, kernel.getScalar i ≠ 0) then
    if hStride : (∀ i : Fin d, stride.getScalar i ≠ 0) then
      let ⟨ss, g⟩ ← get
      let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
      let layer : Spec.AvgPoolSpec d kernel stride padding hKernel hStride := {}
      let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
      let outShape : Shape := Shape.ofList (C :: Tensor.to outSpatial (List Nat))
      let inShape : Shape := Shape.ofList (C :: Tensor.to inSpatial (List Nat))
      let node : NodeData α Δ (Γ ++ ss) outShape :=
        { forward := fun ctx _d =>
            let xv := getIdx (α := α) (xs := ctx) ix
            Spec.avgPoolSpec (α := α) (d := d) (C := C)
              (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
              (layer := layer) xv
          jvp := fun _ctx dctx _d =>
            let dx := getIdx (α := α) (xs := dctx) ix
            Spec.avgPoolSpec (α := α) (d := d) (C := C)
              (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
              (layer := layer) dx
          vjp := fun _ctx _d δ =>
            let dx :=
              Spec.avgPoolBackwardSpec (α := α) (d := d) (C := C)
                (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
                (layer := layer) (gradOutput := δ)
            TensorPack.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
      push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
    else
      throw "typed GraphM: avg_pool requires stride > 0 on every spatial axis"
  else
    throw "typed GraphM: avg_pool requires kernel_size > 0 on every spatial axis"

/--
N-D smooth max pooling (log-sum-exp surrogate) on a single sample tensor (no batch axis).

PyTorch comparison: there is no direct primitive; this is a differentiable approximation to
max pooling.

Forward-mode status: implemented. The JVP is the softmax-weighted tangent of the log-sum-exp
pooling window. Executable graphs require a finite, nonzero `beta` and at least one spatial
dimension.
-/
def smoothMaxPool {α : Type} {Δ : Type} [TorchLean.Storage α] [Context α] [DecidableEq α]
  {Γ : List Shape} {d C : Nat}
  {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
  (x : Var (Shape.ofList (C :: Tensor.to inSpatial (List Nat)))) (beta : α) :
  MWith α Δ Γ (Var (Shape.ofList
    (C :: Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))) := do
  -- `Context` has no generic `isFinite`; supported executable scalars self-subtract to zero exactly
  -- when finite, so this probe rejects NaN and infinities without adding a stronger scalar class.
  if beta == 0 then
    throw "typed GraphM: smooth_max_pool requires finite nonzero beta"
  if hBeta : beta ≠ 0 then
    if !(beta - beta == 0) then
      throw "typed GraphM: smooth_max_pool requires finite nonzero beta"
    if d = 0 then
      throw "typed GraphM: smooth_max_pool requires at least one spatial dimension"
    if hKernel : (∀ i : Fin d, kernel.getScalar i ≠ 0) then
      if hStride : (∀ i : Fin d, stride.getScalar i ≠ 0) then
        let ⟨ss, g⟩ ← get
        let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
        let layer : Spec.MaxPoolSpec d kernel stride padding hKernel hStride := {}
        let outSpatial := Spec.poolOutSpatialPad inSpatial kernel stride padding
        let outShape : Shape := Shape.ofList (C :: Tensor.to outSpatial (List Nat))
        let inShape : Shape := Shape.ofList (C :: Tensor.to inSpatial (List Nat))
        let node : NodeData α Δ (Γ ++ ss) outShape :=
          { forward := fun ctx _d =>
              let xv := getIdx (α := α) (xs := ctx) ix
              Spec.smoothMaxPoolSpec (α := α) (d := d) (C := C)
                (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
                (layer := layer) (beta := beta) (hBeta := hBeta) xv
            jvp := fun ctx dctx _d =>
              let xv := getIdx (α := α) (xs := ctx) ix
              let dx := getIdx (α := α) (xs := dctx) ix
              Spec.smoothMaxPoolJvpSpec (α := α) (d := d) (C := C)
                (inSpatial := inSpatial) (kernel := kernel) (stride := stride)
                (padding := padding) (layer := layer) (beta := beta) (hBeta := hBeta) xv dx
            vjp := fun ctx _d δ =>
              let xv := getIdx (α := α) (xs := ctx) ix
              let dx :=
                Spec.smoothMaxPoolBackwardSpec (α := α) (d := d) (C := C)
                  (inSpatial := inSpatial) (kernel := kernel) (stride := stride)
                  (padding := padding) (layer := layer) (beta := beta) (hBeta := hBeta)
                  (input := xv) (gradOutput := δ)
              TensorPack.single (α := α) (Γ := Γ ++ ss) (s := inShape) ix dx }
        push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node
      else
        throw "typed GraphM: smooth_max_pool requires stride > 0 on every spatial axis"
    else
      throw "typed GraphM: smooth_max_pool requires kernel_size > 0 on every spatial axis"
  else
    throw "typed GraphM: smooth_max_pool requires finite nonzero beta"

end GraphM
end TypedGraph
end Autograd
end Runtime
