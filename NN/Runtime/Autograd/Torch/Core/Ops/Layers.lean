/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Dispatch
public import NN.Runtime.Autograd.Engine.Core.ActivationsLoss
public import NN.Runtime.Autograd.Engine.Core.Linear
public import NN.Runtime.Autograd.Engine.Core.Neural
public import NN.Runtime.Autograd.Engine.Cuda.Ops.Attention
public import NN.Runtime.Autograd.Engine.Cuda.Ops.Linear
public import NN.Runtime.Autograd.Engine.Cuda.Ops.NormSoftmax

/-!
# Eager Tensor Operations

PyTorch-style tensor operations backed by the eager CPU/CUDA tapes. These wrappers record runtime
nodes, dispatch CUDA kernels when requested, and preserve the typed `TensorRef` surface.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean TorchLean.Tensor

namespace Internal

namespace EagerSession

/-! ## Neural-network layers -/

/-- Fully-connected linear layer `y = w x + b`. PyTorch: `torch.nn.functional.linear`. -/
def linear {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Inhabited α] [Add α]
  [Mul α] [Zero α]
  {inDim outDim : Nat}
  (w : TensorRef α [outDim, inDim])
  (b : TensorRef α [outDim])
  (x : TensorRef α [inDim]) : IO (TensorRef α [outDim]) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <| Runtime.Autograd.Tape.linear (t := t0)
      (inDim := inDim) (outDim := outDim) w.id b.id x.id
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.linear (t := t0) (outDim := outDim) (inDim := inDim) w.id b.id x.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .linear #[w.identity?, b.identity?, x.identity?] cpu cuda

/-- Mean-squared-error loss returning a scalar. PyTorch: `torch.nn.functional.mse_loss`. -/
def mseLoss {α : Type} [TorchLean.Storage α] [TensorTransfer α] (s : EagerSession α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [NatCast α]
  {sh : Shape} (yhat target : TensorRef α sh) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.mseLoss (t := t0) (s := sh) yhat.id target.id
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.mseLoss (t := t0) (s := sh) yhat.id target.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .mseLoss #[yhat.identity?, target.identity?] cpu cuda

/-- Layer normalization over embedding dimension. PyTorch: `nn.LayerNorm` / `functional.layer_norm`.
  -/
def layerNorm {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [TensorTransfer α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : TensorRef α [seqLen, embedDim])
  (gamma : TensorRef α [embedDim])
  (beta : TensorRef α [embedDim])
  (epsilon : α := TorchLean.normalizationEpsilon) : IO (TensorRef α [seqLen, embedDim]) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <| Runtime.Autograd.Tape.layerNorm (t := t0)
      (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos)
      (h_embed_pos := h_embed_pos) x.id gamma.id beta.id (epsilon := epsilon)
    pure { id := id }
  let cuda := do
    let epsilonFloat ← TensorTransfer.toFloat (α := α) epsilon
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.layerNorm (t := t0)
      (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos)
      (h_embed_pos := h_embed_pos) x.id gamma.id beta.id (epsilon := epsilonFloat)
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .layerNorm #[x.identity?, gamma.identity?, beta.identity?] cpu cuda

/-- Batch normalization over every spatial axis of a channel-first tensor. -/
def batchNorm {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [TensorTransfer α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {channels : Nat} {sSpatial : Shape}
  (hWellFormed : (Shape.dim channels sSpatial).wellFormed)
  (x : TensorRef α (.dim channels sSpatial))
  (gamma : TensorRef α [channels])
  (beta : TensorRef α [channels])
  (epsilon : α := TorchLean.normalizationEpsilon) :
  IO (TensorRef α (.dim channels sSpatial)) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <| Runtime.Autograd.Tape.batchNorm (t := t0)
      (channels := channels) (sSpatial := sSpatial) hWellFormed
      x.id gamma.id beta.id (epsilon := epsilon)
    pure { id := id }
  let cuda : IO (Option (TensorRef α (.dim channels sSpatial))) :=
    do
      let epsilonFloat ← TensorTransfer.toFloat (α := α) epsilon
      let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
        Runtime.Autograd.Cuda.Tape.batchNorm (t := t0)
        (channels := channels) (spatial := sSpatial) hWellFormed x.id gamma.id beta.id
        (epsilon := epsilonFloat)
      pure (some { id := id })
  dispatchCudaOpt (α := α) s .batchNorm #[x.identity?, gamma.identity?, beta.identity?] cpu cuda

/-- Multi-head self-attention (typed, proof-friendly). PyTorch: `nn.MultiheadAttention`
  (conceptually). -/
def multiHeadAttention {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : TensorRef α [dModel, numHeads * headDim])
  (wk : TensorRef α [dModel, numHeads * headDim])
  (wv : TensorRef α [dModel, numHeads * headDim])
  (wo : TensorRef α [numHeads * headDim, dModel])
  (x : TensorRef α [n, dModel])
  (mask : Option (Tensor Bool [n, n]) := none) :
  IO (TensorRef α [n, dModel]) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.multiHeadAttention (t := t0)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
      wq.id wk.id wv.id wo.id x.id mask
    pure { id := id }
  let cuda := fun attentionCapsule => do
    let t0 ← s.cudaTape.get
    let result ← Runtime.Autograd.Cuda.Tape.multiHeadAttention (t := t0)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
      wq.id wk.id wv.id wo.id x.id (mask := mask) (attentionCapsule := attentionCapsule)
    let (t1, id) ← okOrThrow result
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaCapsuleOpt (α := α) s .scaledDotProductAttention
    #[wq.identity?, wk.identity?, wv.identity?, wo.identity?, x.identity?]
    #[.torchLean, .libTorch] cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
