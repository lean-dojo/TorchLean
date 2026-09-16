/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.Runtime.Autograd.Torch.Core.Functional -- shake: keep
public import NN.Runtime.Autograd.Torch.Core.Types
public import NN.Tensor -- shake: keep

/-!
# Operation-Polymorphic Tensor Programs

`Program` is model code abstract over a tensor-operation interpreter. Supplying the eager
interpreter executes operations immediately and records a dynamic tape; supplying the typed-graph
interpreter records reusable shape-indexed SSA data.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

export Runtime.Autograd.Torch (ExecutionMode Config Ops Ref RefList CurriedRef)

namespace Curried
export Runtime.Autograd.Torch.Curried (Fn curry uncurry)
end Curried

namespace CurriedRef
export Runtime.Autograd.Torch.CurriedRef (uncurry applyVarList)
end CurriedRef

namespace RefList
export Runtime.Autograd.Torch.RefList (append)
end RefList

/-! The execution-polymorphic surface shared by eager and typed-graph execution. -/
export Runtime.Autograd.Torch
  (const add sub mul scale abs sqrt clamp max min
   broadcastTo reshape swapAdjacentAtDepth
   reduceSum reduceMean
   select indexSelect scatterAdd
   matmul concatLeadingAxis sliceLeadingAxisRange
   relu silu gelu sigmoid tanh softmaxLast softplus
   exp sin cos log inv detach safeLog logSoftmaxLast
   sum flatten
   mseLoss batchNorm batchedMultiHeadAttention
   randUniform bernoulliMask)

/-! ## Operation-reference notation -/

/-- A tensor reference under the currently selected operation interpreter. -/
abbrev RefTy (m : Type → Type) (α : Type)
    [TorchLean.Storage α] [Context α] [Ops (m := m) (α := α)]
    (s : Shape) : Type :=
  Runtime.Autograd.Torch.Ops.Ref (m := m) (α := α) s

namespace LeadingAxis
namespace Internal

/-- Apply a single-sample operation independently along a leading axis. -/
def mapOuterAxis {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch : Nat} {s t : Shape}
    (x : RefTy (m := m) (α := α) (s.prependDim batch))
    (f : RefTy (m := m) (α := α) s → m (RefTy (m := m) (α := α) t)) :
    m (RefTy (m := m) (α := α) (t.prependDim batch)) :=
  Runtime.Autograd.Torch.mapOuterAxis (m := m) (α := α) f x

end Internal
end LeadingAxis

/-! ## Prefix-polymorphic derived operations -/

/-- Apply a single-sample operation independently over an arbitrary prefix shape. -/
def mapLeading {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (leadingShape : Shape) {s t : Shape}
    (x : RefTy (m := m) (α := α) (leadingShape.concat s))
    (f : RefTy (m := m) (α := α) s → m (RefTy (m := m) (α := α) t)) :
    m (RefTy (m := m) (α := α) (leadingShape.concat t)) := do
  let xFlat ← Runtime.Autograd.Torch.reshape (m := m) (α := α)
    (s₁ := leadingShape.concat s) (s₂ := s.prependDim leadingShape.size) x (by
      simp [Shape.size_concat, Shape.size])
  let yFlat ← LeadingAxis.Internal.mapOuterAxis (m := m) (α := α) xFlat f
  Runtime.Autograd.Torch.reshape (m := m) (α := α)
    (s₁ := t.prependDim leadingShape.size) (s₂ := leadingShape.concat t) yFlat (by
      simp [Shape.size_concat, Shape.size])

/-- Affine transformation of the final axis, independently over any prefix shape. -/
def linearEach {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {inDim outDim : Nat}
    (weight : RefTy (m := m) (α := α) [outDim, inDim])
    (bias : RefTy (m := m) (α := α) [outDim])
    (input : RefTy (m := m) (α := α) (leadingShape.concat [inDim])) :
    m (RefTy (m := m) (α := α) (leadingShape.concat [outDim])) :=
  mapLeading (m := m) (α := α) leadingShape input fun x =>
    Runtime.Autograd.Torch.linear (m := m) (α := α) weight bias x

/-- Rank-polymorphic convolution over channels-first inputs with any prefix shape. -/
def conv {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weight : RefTy (m := m) (α := α)
      (Shape.ofList (outC :: inC :: kernel.to (List Nat))))
    (bias : RefTy (m := m) (α := α) [outC])
    (input : RefTy (m := m) (α := α)
      (leadingShape.concat (Shape.ofList (inC :: inSpatial.to (List Nat))))) :
    m (RefTy (m := m) (α := α)
      (leadingShape.concat (Shape.ofList
        (outC ::
          (Spec.convOutSpatial inSpatial kernel stride padding).to (List Nat))))) :=
  mapLeading (m := m) (α := α) leadingShape input (fun x ↦
    Runtime.Autograd.Torch.conv (m := m) (α := α)
      (d := d) (inC := inC) (outC := outC) (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := inSpatial) weight bias x)

/-- Rank-polymorphic transpose convolution over channels-first inputs with any prefix shape. -/
def convTranspose {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (weight : RefTy (m := m) (α := α)
      (Shape.ofList (inC :: outC :: kernel.to (List Nat))))
    (bias : RefTy (m := m) (α := α) [outC])
    (input : RefTy (m := m) (α := α)
      (leadingShape.concat (Shape.ofList (inC :: inSpatial.to (List Nat))))) :
    m (RefTy (m := m) (α := α)
      (leadingShape.concat (Shape.ofList
        (outC ::
          (Spec.convTransposeOutSpatial inSpatial kernel stride padding).to (List Nat))))) :=
  mapLeading (m := m) (α := α) leadingShape input (fun x ↦
    Runtime.Autograd.Torch.convTranspose (m := m) (α := α)
      (d := d) (inC := inC) (outC := outC) (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := inSpatial) weight bias x)

/-- Rank-polymorphic max pooling over channels-first inputs with any prefix shape. -/
def maxPool {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {d channels : Nat}
    {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (input : RefTy (m := m) (α := α)
      (leadingShape.concat (Shape.ofList (channels :: inSpatial.to (List Nat))))) :
    m (RefTy (m := m) (α := α)
      (leadingShape.concat (Shape.ofList (channels ::
        (Spec.poolOutSpatialPad inSpatial kernel stride padding).to (List Nat))))) :=
  mapLeading (m := m) (α := α) leadingShape input (fun x ↦
    Runtime.Autograd.Torch.maxPool (m := m) (α := α)
      (d := d) (C := channels) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) x)

/-- Rank-polymorphic average pooling over channels-first inputs with any prefix shape. -/
def avgPool {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {d channels : Nat}
    {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (input : RefTy (m := m) (α := α)
      (leadingShape.concat (Shape.ofList (channels :: inSpatial.to (List Nat))))) :
    m (RefTy (m := m) (α := α)
      (leadingShape.concat (Shape.ofList (channels ::
        (Spec.poolOutSpatialPad inSpatial kernel stride padding).to (List Nat))))) :=
  mapLeading (m := m) (α := α) leadingShape input (fun x ↦
    Runtime.Autograd.Torch.avgPool (m := m) (α := α)
      (d := d) (C := channels) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) x)

/-- Batched rank-polymorphic smooth maximum pooling over channels-first inputs. -/
def smoothMaxPool {α : Type} [TorchLean.Storage α] [Context α] [DecidableEq α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {d channels : Nat}
    {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
    (input : RefTy (m := m) (α := α)
      (leadingShape.concat (Shape.ofList (channels :: inSpatial.to (List Nat))))) (temp : α) :
    m (RefTy (m := m) (α := α)
      (leadingShape.concat (Shape.ofList (channels ::
        (Spec.poolOutSpatialPad inSpatial kernel stride padding).to (List Nat))))) :=
  mapLeading (m := m) (α := α) leadingShape input (fun x ↦
    Runtime.Autograd.Torch.smoothMaxPool (m := m) (α := α)
      (d := d) (C := channels) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) x temp)

/--
Layer normalization over the final axis of an arbitrary tensor.

Backends expose a matrix-shaped fused primitive. This wrapper gives that primitive its
rank-polymorphic TorchLean interface by flattening the leading axes once and restoring them after
the operation. Empty leading axes produce the unique empty tensor without invoking a backend
kernel that requires a positive row count.
-/
def layerNorm {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading : Shape} {width : Nat} (hWidth : width > 0)
    (x : RefTy (m := m) (α := α) (leading.appendDim width))
    (gamma beta : RefTy (m := m) (α := α) [width])
    (epsilon : α := TorchLean.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (leading.appendDim width)) :=
  match hRows : leading.size with
  | 0 => Runtime.Autograd.Torch.const (m := m) (α := α)
      (s := leading.appendDim width) (Tensor.full (leading.appendDim width) (0 : α))
  | rows + 1 => do
      let matrixShape : Shape := [rows + 1, width]
      let xMatrix ← Runtime.Autograd.Torch.reshape (m := m) (α := α)
        (s₁ := leading.appendDim width) (s₂ := matrixShape) x (by
          simp [matrixShape, Shape.size_appendDim, Shape.size, hRows])
      let yMatrix ← Runtime.Autograd.Torch.layerNorm (m := m) (α := α)
        (seqLen := rows + 1) (embedDim := width) (Nat.succ_pos rows) hWidth xMatrix gamma beta
        (epsilon := epsilon)
      Runtime.Autograd.Torch.reshape (m := m) (α := α)
        (s₁ := matrixShape) (s₂ := leading.appendDim width) yMatrix (by
          simp [matrixShape, Shape.size_appendDim, Shape.size, hRows])

/-- Multi-head self-attention over any prefix shape. -/
def multiHeadAttention {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {n numHeads dModel headDim : Nat} (hN : n ≠ 0)
    (wq wk wv : RefTy (m := m) (α := α) [dModel, numHeads * headDim])
    (wo : RefTy (m := m) (α := α) [numHeads * headDim, dModel])
    (x : RefTy (m := m) (α := α) (leadingShape.concat [n, dModel]))
    (mask : Option (Tensor Bool [n, n]) := none) :
    m (RefTy (m := m) (α := α) (leadingShape.concat [n, dModel])) :=
  match batchEq : leadingShape.size with
  | 0 => Runtime.Autograd.Torch.const (m := m) (α := α)
      (s := leadingShape.concat [n, dModel])
      (Tensor.full (leadingShape.concat [n, dModel]) (0 : α))
  | batch + 1 => do
      let xFlat ← Runtime.Autograd.Torch.reshape (m := m) (α := α)
        (s₁ := leadingShape.concat [n, dModel]) (s₂ := [batch + 1, n, dModel]) x (by
          simp [Shape.size_concat, Shape.size, batchEq])
      let yFlat ← Runtime.Autograd.Torch.batchedMultiHeadAttention
        (m := m) (α := α) (batch := batch + 1) (n := n) (numHeads := numHeads)
        (dModel := dModel) (headDim := headDim) (by simp) hN wq wk wv wo xFlat (mask := mask)
      Runtime.Autograd.Torch.reshape (m := m) (α := α)
        (s₁ := [batch + 1, n, dModel]) (s₂ := leadingShape.concat [n, dModel]) yFlat (by
          simp [Shape.size_concat, Shape.size, batchEq])

/-- Multi-head attention followed by a trainable output-feature bias. -/
def multiHeadAttentionOutputBias {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leadingShape : Shape} {n numHeads dModel headDim : Nat} (hN : n ≠ 0)
    (wq wk wv : RefTy (m := m) (α := α) [dModel, numHeads * headDim])
    (wo : RefTy (m := m) (α := α) [numHeads * headDim, dModel])
    (bo : RefTy (m := m) (α := α) [dModel])
    (x : RefTy (m := m) (α := α) (leadingShape.concat [n, dModel]))
    (mask : Option (Tensor Bool [n, n]) := none) :
    m (RefTy (m := m) (α := α) (leadingShape.concat [n, dModel])) := do
  let y ← multiHeadAttention (m := m) (α := α) hN wq wk wv wo x mask
  mapLeading (m := m) (α := α) leadingShape y fun yi => do
    let boFull ← Runtime.Autograd.Torch.broadcastTo (m := m) (α := α)
      (s₁ := [dModel]) (s₂ := [n, dModel]) Shape.BroadcastTo.proof bo
    Runtime.Autograd.Torch.add (m := m) (α := α) (s := [n, dModel]) yi boFull

/-- An execution-polymorphic differentiable tensor program. -/
abbrev Program (α : Type) [TorchLean.Storage α] [Context α] (ss : List Shape) (τ : Shape) :
    Type 1 :=
  ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
    CurriedRef (fun s ↦ RefTy (m := m) (α := α) s) ss (m (RefTy (m := m) (α := α) τ))

/--
An execution-polymorphic program with differentiable tensors followed by non-differentiable data
tensors. The data element type is explicit; for example, token models may use `Fin vocab` so an
out-of-range token is unrepresentable.
-/
abbrev ProgramWithDataInputs (α β : Type) [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    (ss dataSs : List Shape) (τ : Shape) : Type 1 :=
  ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
    CurriedRef (fun s ↦ RefTy (m := m) (α := α) s) ss
      (CurriedRef (fun s ↦ Runtime.Autograd.Torch.DataRef
        (m := m) (α := α) β s) dataSs (m (RefTy (m := m) (α := α) τ)))

end Model
end Autograd
end Runtime
