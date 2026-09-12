/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Session.Types
public import NN.Runtime.Autograd.Torch.TypedGraphSession.ConvAttention
public import NN.Runtime.Autograd.Torch.TypedGraphSession.Neural

/-!
# Session Neural-Network Operations

This file contains higher-level neural-network session calls such as linear layers, normalization,
attention, and convolutional blocks. The operations share the same session dispatch discipline as
the elementary ops while preserving PyTorch-style call sites.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Session

/--
Fully-connected (affine) layer on vectors: $y=w\mathbin{\cdot}x+b$.

PyTorch analogue: `torch.nn.functional.linear` (weight shape `(outDim, inDim)`).
-/
def linear {α : Type} [TorchLean.Storage α] (s : Session α)
  [Inhabited α] [Add α] [Mul α] [Zero α]
  {inDim outDim : Nat}
  (w : Runtime.Autograd.Torch.TensorRef α [outDim, inDim])
  (b : Runtime.Autograd.Torch.TensorRef α [outDim])
  (x : Runtime.Autograd.Torch.TensorRef α [inDim]) :
  IO (Runtime.Autograd.Torch.TensorRef α [outDim]) := do
  match s.state with
  | .eager sess =>
      EagerSession.linear (α := α) sess (inDim := inDim) (outDim := outDim) w b x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.linear (α := α) sess
        (inDim := inDim) (outDim := outDim) w b x

/--
Mean squared error loss returning a scalar.

PyTorch analogue: `torch.nn.functional.mse_loss(..., reduction='mean')`.
-/
def mseLoss {α : Type} [TorchLean.Storage α] (s : Session α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [NatCast α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape}
  (yhat target : Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.state with
  | .eager sess => EagerSession.mseLoss (α := α) sess (sh := sh) yhat target
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.mseLoss (α := α) sess (sh := sh) yhat target

/--
LayerNorm over a `seqLen × embedDim` tensor.

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` applied per token.
-/
def layerNorm {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  [Runtime.Autograd.Torch.TensorTransfer α]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : Runtime.Autograd.Torch.TensorRef α [seqLen, embedDim])
  (gamma : Runtime.Autograd.Torch.TensorRef α [embedDim])
  (beta : Runtime.Autograd.Torch.TensorRef α [embedDim])
  (epsilon : α := TorchLean.normalizationEpsilon) :
  IO (Runtime.Autograd.Torch.TensorRef α [seqLen, embedDim]) := do
  match s.state with
  | .eager sess =>
      EagerSession.layerNorm (α := α) sess
        (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos :=
          h_embed_pos)
        x gamma beta (epsilon := epsilon)
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.layerNorm (α := α) sess
        (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos :=
          h_embed_pos)
        x gamma beta (epsilon := epsilon)

/-- Batch normalization over every spatial axis of a channel-first tensor. -/
def batchNorm {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
    [Runtime.Autograd.Torch.TensorTransfer α]
    {channels : Nat} {sSpatial : Shape}
    (hWellFormed : (Shape.dim channels sSpatial).wellFormed)
  (x : Runtime.Autograd.Torch.TensorRef α (.dim channels sSpatial))
  (gamma : Runtime.Autograd.Torch.TensorRef α [channels])
  (beta : Runtime.Autograd.Torch.TensorRef α [channels])
  (epsilon : α := TorchLean.normalizationEpsilon) :
  IO (Runtime.Autograd.Torch.TensorRef α (.dim channels sSpatial)) := do
  match s.state with
  | .eager sess =>
      EagerSession.batchNorm (α := α) sess
        (channels := channels) (sSpatial := sSpatial) hWellFormed x gamma beta
        (epsilon := epsilon)
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.batchNorm (α := α) sess
        (channels := channels) (sSpatial := sSpatial) hWellFormed x gamma beta
        (epsilon := epsilon)

/--
N-D convolution over a channels-first tensor `(inC, spatial...)`.

PyTorch analogue: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (w : Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: inC :: kernel.to (List Nat))))
  (b : Runtime.Autograd.Torch.TensorRef α [outC])
  (x : Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (inC :: inSpatial.to (List Nat)))) :
  IO (Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList
      (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).to (List Nat)))) := do
  match s.state with
  | .eager sess =>
      EagerSession.conv (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        w b x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.conv (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        w b x

/--
N-D transpose convolution over a channels-first tensor `(inC, spatial...)`.

PyTorch analogue: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (w : Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (inC :: outC :: kernel.to (List Nat))))
  (b : Runtime.Autograd.Torch.TensorRef α [outC])
  (x : Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (inC :: inSpatial.to (List Nat)))) :
  IO (Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList
      (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).to (List Nat))))
    := do
  match s.state with
  | .eager sess =>
      EagerSession.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        w b x
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        w b x

/--
Multi-head self-attention (single sequence, single batch).

This is a convenience op used by the transformer examples; it corresponds approximately to the
forward pass of `torch.nn.MultiheadAttention` in "self-attention" mode.
-/
def multiHeadAttention {α : Type} [TorchLean.Storage α] (s : Session α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wk : Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wv : Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wo : Runtime.Autograd.Torch.TensorRef α [numHeads * headDim, dModel])
  (x : Runtime.Autograd.Torch.TensorRef α [n, dModel])
  (mask : Option (Tensor Bool [n, n]) := none) :
  IO (Runtime.Autograd.Torch.TensorRef α [n, dModel]) := do
  match s.state with
  | .eager sess =>
      EagerSession.multiHeadAttention (α := α) sess
        (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        wq wk wv wo x (mask := mask)
  | .typedGraph sess =>
      Runtime.Autograd.Torch.Internal.TypedGraphSession.multiHeadAttention (α := α) sess
        (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        wq wk wv wo x (mask := mask)


end Session

end Model
end Autograd
end Runtime
