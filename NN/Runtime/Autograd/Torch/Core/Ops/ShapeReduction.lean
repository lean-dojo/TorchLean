/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Dispatch
public import NN.Runtime.Autograd.Engine.Core.ActivationsLoss
public import NN.Runtime.Autograd.Engine.Core.Shape
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

/-! ## Shape and reduction operations -/

/-- Sum-reduce all elements to a scalar. PyTorch: `x.sum()`. -/
def sum {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Add α] [Zero α]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.sum (t := t0) (s := sh) x.id
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.sum (t := t0) (s := sh) x.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .reduceSum #[x.identity?] cpu cuda

/-- Flatten a tensor to a 1D vector. PyTorch: `torch.flatten`. -/
def flatten {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Inhabited α] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α [Spec.Shape.size sh]) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.flatten (t := t0) (s := sh) x.id
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.flatten (t := t0) (s := sh) x.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .reshape #[x.identity?] cpu cuda

/--
Reshape a tensor while preserving total number of elements.

PyTorch comparison: `torch.reshape` / `view` (when valid).
-/
def reshape {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Inhabited α] {sh1 sh2 : Shape}
  (x : TensorRef α sh1) (h : Spec.Shape.size sh1 = Spec.Shape.size sh2) : IO (TensorRef α sh2) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.reshape (t := t0) (s₁ := sh1) (s₂ := sh2) x.id h
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.reshape (t := t0) (s₁ := sh1) (s₂ := sh2) x.id h
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .reshape #[x.identity?] cpu cuda

/-- Swap two adjacent axes at a given depth. PyTorch analogue: `x.transpose(dim, dim+1)`. -/
def swapAdjacentAtDepth {α : Type} [TorchLean.Storage α] (s : EagerSession α) {sh : Shape}
  (depth : Nat) (x : TensorRef α sh) : IO (TensorRef α (sh.swapAdjacentAtDepth depth)) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.swapAdjacentAtDepth (t := t0) (s := sh) depth x.id
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.swapAdjacentAtDepth (t := t0) (s := sh) depth x.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .permute #[x.identity?] cpu cuda

/-- Broadcast a tensor to a larger shape. PyTorch: implicit broadcasting / `expand`. -/
def broadcastTo {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Inhabited α] [Add α]
  [Zero α]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : TensorRef α sh1) : IO (TensorRef α sh2)
    := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      (Runtime.Autograd.Tape.broadcastTo (α := α) (t := t0) (s₁ := sh1) (s₂ := sh2) cb x.id)
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.broadcastTo (t := t0) (s₁ := sh1) (s₂ := sh2) cb x.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .broadcast #[x.identity?] cpu cuda

/-- Sum-reduce along `axis`. PyTorch: `torch.sum(x, dim=axis)`. -/
def reduceSum {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Add α] [Zero α]
  [Inhabited α]
  {sh : Shape} (axis : Nat) [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.reduceSum (t := t0) (s := sh) axis x.id
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.reduceSum (s := sh) axis (t := t0) x.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .reduceSum #[x.identity?] cpu cuda

/-- Mean-reduce along `axis`. PyTorch: `torch.mean(x, dim=axis)`. -/
def reduceMean {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {sh : Shape} (axis : Nat) [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.reduceMean (t := t0) (s := sh) axis x.id
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.reduceMean (s := sh) axis (t := t0) x.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .reduceMean #[x.identity?] cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
