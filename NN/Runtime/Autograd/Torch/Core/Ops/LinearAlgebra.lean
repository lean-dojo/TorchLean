/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Dispatch
public import NN.Runtime.Autograd.Engine.Core.Linear
public import NN.Runtime.Autograd.Engine.Cuda.Ops.Indexing
public import NN.Runtime.Autograd.Engine.Cuda.Ops.Linear
public import NN.Runtime.Autograd.Engine.Cuda.Ops.Shape

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

/-! ## Linear algebra and concatenation -/

/-- Matrix multiplication with PyTorch-style broadcasting across batch prefixes. -/
def matmul {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {batchA batchB batch : Shape} {m n p : Nat}
  [broadcastA : Shape.BroadcastTo batchA batch]
  [broadcastB : Shape.BroadcastTo batchB batch]
  (a : TensorRef α (batchA.concat [m, n]))
  (b : TensorRef α (batchB.concat [n, p])) :
  IO (TensorRef α (batch.concat [m, p])) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.matmul (t := t0) (m := m) (n := n) (p := p) a.id b.id
        (batchA := batchA) (batchB := batchB) (batch := batch)
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let broadcastAFull :=
      TorchLean.Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [m, n] broadcastA.proof
    let broadcastBFull :=
      TorchLean.Tensor.LinearAlgebra.Internal.extendBroadcastSuffix [n, p] broadcastB.proof
    let (t1, commonAId) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.broadcastTo (t := t0) broadcastAFull a.id
    let (t2, commonBId) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.broadcastTo (t := t1) broadcastBFull b.id
    let (t3, flatAId) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reshape (t := t2) commonAId
        (TorchLean.Tensor.LinearAlgebra.Internal.flattenBatchMatrix_size batch m n)
    let (t4, flatBId) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reshape (t := t3) commonBId
        (TorchLean.Tensor.LinearAlgebra.Internal.flattenBatchMatrix_size batch n p)
    let (t5, flatOutId) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.Internal.matmulFlattened (t := t4) (batch := batch.size)
        (m := m) (n := n) (p := p) flatAId flatBId
    let (t6, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reshape (t := t5) flatOutId
        (TorchLean.Tensor.LinearAlgebra.Internal.flattenBatchMatrix_size batch m p).symm
    s.cudaTape.set t6
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .matmul #[a.identity?, b.identity?] cpu cuda

/-- Concatenate along dim 0 for tensors with leading dimension. PyTorch: `torch.cat(..., dim=0)`. -/
def concatLeadingAxis {α : Type} [TorchLean.Storage α] (s : EagerSession α)
  {n m : Nat} {sh : Shape}
  (a : TensorRef α (.dim n sh))
  (b : TensorRef α (.dim m sh)) :
  IO (TensorRef α (.dim (n + m) sh)) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      (Runtime.Autograd.Tape.concatLeadingAxis (α := α) (t := t0) (n := n) (m := m) (s := sh)
        a.id b.id)
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.concatLeadingAxis (t := t0) (n := n) (m := m) (s := sh) a.id b.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .concat #[a.identity?, b.identity?] cpu cuda

/-- Slice along dim 0: `x[start:start+len]`. PyTorch: standard slicing. -/
def sliceLeadingAxisRange {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Zero α]
  {n : Nat} {sh : Shape}
  (x : TensorRef α (.dim n sh)) (start len : Nat) (h : start + len ≤ n) :
  IO (TensorRef α (.dim len sh)) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      (Runtime.Autograd.Tape.sliceLeadingAxisRange (α := α) (t := t0) (n := n) (s := sh)
        x.id start len h)
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.sliceLeadingAxisRange (t := t0) (n := n) (s := sh) x.id start len h
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .slice #[x.identity?] cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
