/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.LeadingAxis
public import NN.Runtime.Autograd.Torch.Core.Ops.Layers
public import NN.Runtime.Autograd.Torch.Core.Ops.LinearAlgebra
public import NN.Runtime.Autograd.Torch.Core.Ops.ShapeReduction

/-!
# Batch-Aware Attention For The Eager Trainer

This module supplies the eager implementation used by the generic `Ops` instance for a leading
batch axis. The CPU path fixes the semantics by mapping the existing single-sample operation. The
CUDA path folds batch and head axes into one native launch while retaining TorchLean's local VJP.
-/

@[expose] public section

namespace Runtime.Autograd.Torch.Internal.EagerSession

open Spec TorchLean
open TorchLean TorchLean.Tensor

/--
CPU reference path for batch-aware attention.

It records the established single-sample attention node for each entry of the leading axis. The
CUDA path below can execute those samples together, while this definition fixes the exact
per-sample forward and backward meaning.
-/
def batchedMultiHeadAttentionCpuFallback {α : Type} [TorchLean.Storage α]
    (s : EagerSession α) [Context α] [TensorTransfer α]
    {batch n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
    (wq : TensorRef α [dModel, numHeads * headDim])
    (wk : TensorRef α [dModel, numHeads * headDim])
    (wv : TensorRef α [dModel, numHeads * headDim])
    (wo : TensorRef α [numHeads * headDim, dModel])
    (x : TensorRef α [batch, n, dModel])
    (mask : Option (Tensor Bool [n, n]) := none) :
    IO (TensorRef α [batch, n, dModel]) :=
  Runtime.Autograd.mapOuterAxisWith
    (EagerSession.const s <| Tensor.dim (fun i : Fin 0 => Fin.elim0 i))
    (fun x start len h => EagerSession.sliceLeadingAxisRange s x start len h)
    (fun x h => EagerSession.reshape s x h)
    (fun x y => EagerSession.concatLeadingAxis s x y)
    (fun sample => EagerSession.multiHeadAttention s h1 wq wk wv wo sample mask)
    x

/--
Batch-aware eager attention with a TorchLean-owned local VJP.

The CUDA executor folds `(batch, head)` into one BMM batch axis. Provider selection remains
explicit, and the checked default uses TorchLean's hard-masked softmax and backward rule.
-/
def batchedMultiHeadAttention {α : Type} [TorchLean.Storage α] (s : EagerSession α)
    [Context α] [TensorTransfer α]
    {batch n numHeads dModel headDim : Nat} (hBatch : batch ≠ 0) (h1 : n ≠ 0)
    (wq : TensorRef α [dModel, numHeads * headDim])
    (wk : TensorRef α [dModel, numHeads * headDim])
    (wv : TensorRef α [dModel, numHeads * headDim])
    (wo : TensorRef α [numHeads * headDim, dModel])
    (x : TensorRef α [batch, n, dModel])
    (mask : Option (Tensor Bool [n, n]) := none) :
    IO (TensorRef α [batch, n, dModel]) := do
  let cpu := batchedMultiHeadAttentionCpuFallback s h1 wq wk wv wo x mask
  let cuda := fun attentionCapsule => do
    let t0 ← s.cudaTape.get
    let result ← Runtime.Autograd.Cuda.Tape.batchedMultiHeadAttention (t := t0)
      (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
      hBatch h1 wq.id wk.id wv.id wo.id x.id (mask := mask)
      (attentionCapsule := attentionCapsule)
    let (t1, id) ← okOrThrow result
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaCapsuleOpt (α := α) s .scaledDotProductAttention
    #[wq.identity?, wk.identity?, wv.identity?, wo.identity?, x.identity?]
    #[.nativeCuda, .torchLean, .libTorch] cpu cuda

end Runtime.Autograd.Torch.Internal.EagerSession
