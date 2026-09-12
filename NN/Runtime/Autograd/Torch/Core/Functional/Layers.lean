/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Functional.Ops

/-!
# Layer Operations over Backend References

Linear layers, normalization, attention, convolution, and pooling. These wrappers keep the backend
instance implicit while preserving the shape conventions of `Ops`.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

variable {m : Type → Type} {α : Type} [Storage α] [Context α] [Monad m]
    [Ops (m := m) (α := α)]

@[inherit_doc Ops.maxPool]
def maxPool {d C : Nat}
    {inSpatial kernel stride padding : Tensor Nat [d]}
    (x : Ref (m := m) (α := α) (Shape.ofList (C :: Tensor.to inSpatial (List Nat)))) :
    m (Ref (m := m) (α := α)
      (Shape.ofList
        (C :: Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))) :=
  Ops.maxPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    x

@[inherit_doc Ops.avgPool]
def avgPool {d C : Nat}
    {inSpatial kernel stride padding : Tensor Nat [d]}
    (x : Ref (m := m) (α := α) (Shape.ofList (C :: Tensor.to inSpatial (List Nat)))) :
    m (Ref (m := m) (α := α)
      (Shape.ofList
        (C :: Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))) :=
  Ops.avgPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    x

@[inherit_doc Ops.smoothMaxPool]
def smoothMaxPool {d C : Nat} [DecidableEq α]
    {inSpatial kernel stride padding : Tensor Nat [d]}
    (x : Ref (m := m) (α := α) (Shape.ofList (C :: Tensor.to inSpatial (List Nat))))
    (beta : α) :
    m (Ref (m := m) (α := α)
      (Shape.ofList
        (C :: Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))) :=
  Ops.smoothMaxPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    x beta

@[inherit_doc Ops.linear]
def linear {inDim outDim : Nat}
    (w : Ref (m := m) (α := α) [outDim, inDim])
    (b : Ref (m := m) (α := α) [outDim])
    (x : Ref (m := m) (α := α) [inDim]) :
    m (Ref (m := m) (α := α) [outDim]) :=
  Ops.linear (m := m) (α := α) (inDim := inDim) (outDim := outDim) w b x

@[inherit_doc Ops.mseLoss]
def mseLoss {s : Shape} (yhat target : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.mseLoss (m := m) (α := α) (s := s) yhat target

@[inherit_doc Ops.layerNorm]
def layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
    (x : Ref (m := m) (α := α) [seqLen, embedDim])
    (gamma : Ref (m := m) (α := α) [embedDim])
    (beta : Ref (m := m) (α := α) [embedDim])
    (epsilon : α := TorchLean.normalizationEpsilon) :
    m (Ref (m := m) (α := α) [seqLen, embedDim]) :=
  Ops.layerNorm (m := m) (α := α) (seqLen := seqLen) (embedDim := embedDim)
    h_seq_pos h_embed_pos x gamma beta (epsilon := epsilon)

@[inherit_doc Ops.batchNorm]
def batchNorm {channels : Nat} {sSpatial : Shape}
    (hWellFormed : (sSpatial.prependDim channels).wellFormed)
    (x : Ref (m := m) (α := α) (sSpatial.prependDim channels))
    (gamma : Ref (m := m) (α := α) [channels])
    (beta : Ref (m := m) (α := α) [channels])
    (epsilon : α := TorchLean.normalizationEpsilon) :
    m (Ref (m := m) (α := α) (sSpatial.prependDim channels)) :=
  Ops.batchNorm (m := m) (α := α) (channels := channels) (sSpatial := sSpatial)
    hWellFormed x gamma beta (epsilon := epsilon)

@[inherit_doc Ops.multiHeadAttention]
def multiHeadAttention {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
    (wq : Ref (m := m) (α := α) [dModel, numHeads * headDim])
    (wk : Ref (m := m) (α := α) [dModel, numHeads * headDim])
    (wv : Ref (m := m) (α := α) [dModel, numHeads * headDim])
    (wo : Ref (m := m) (α := α) [numHeads * headDim, dModel])
    (x : Ref (m := m) (α := α) [n, dModel])
    (mask : Option (Tensor Bool [n, n]) := none) :
    m (Ref (m := m) (α := α) [n, dModel]) :=
  Ops.multiHeadAttention (m := m) (α := α) (n := n) (numHeads := numHeads) (dModel := dModel)
    (headDim := headDim) h1 wq wk wv wo x mask

@[inherit_doc Ops.batchedMultiHeadAttention]
def batchedMultiHeadAttention {batch n numHeads dModel headDim : Nat}
    (hBatch : batch ≠ 0) (h1 : n ≠ 0)
    (wq : Ref (m := m) (α := α) [dModel, numHeads * headDim])
    (wk : Ref (m := m) (α := α) [dModel, numHeads * headDim])
    (wv : Ref (m := m) (α := α) [dModel, numHeads * headDim])
    (wo : Ref (m := m) (α := α) [numHeads * headDim, dModel])
    (x : Ref (m := m) (α := α) [batch, n, dModel])
    (mask : Option (Tensor Bool [n, n]) := none) :
    m (Ref (m := m) (α := α) [batch, n, dModel]) :=
  Ops.batchedMultiHeadAttention (m := m) (α := α)
    (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
    hBatch h1 wq wk wv wo x mask

@[inherit_doc Ops.conv]
def conv {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (weight : Ref (m := m) (α := α) (Shape.ofList (outC :: inC :: Tensor.to kernel (List Nat))))
    (bias : Ref (m := m) (α := α) [outC])
    (input : Ref (m := m) (α := α) (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))) :
    m (Ref (m := m) (α := α)
      (Shape.ofList
        (outC :: Tensor.to (Spec.convOutSpatial inSpatial kernel stride padding) (List Nat)))) :=
  Ops.conv (m := m) (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    weight bias input

@[inherit_doc Ops.convTranspose]
def convTranspose {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    (weight : Ref (m := m) (α := α) (Shape.ofList (inC :: outC :: Tensor.to kernel (List Nat))))
    (bias : Ref (m := m) (α := α) [outC])
    (input : Ref (m := m) (α := α) (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))) :
    m (Ref (m := m) (α := α)
      (Shape.ofList (outC ::
        Tensor.to (Spec.convTransposeOutSpatial inSpatial kernel stride padding) (List Nat)))) :=
  Ops.convTranspose (m := m) (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    weight bias input

end Runtime.Autograd.Torch
