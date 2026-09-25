/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Dispatch
public import NN.Runtime.Autograd.Engine.Core.ConvPool
public import NN.Runtime.Autograd.Engine.Cuda.Ops.ConvPool

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

/-! ## Pooling operations -/

/--
N-D max pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on the
spatial rank `d`.
-/
def maxPool {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
  (x : TensorRef α (Shape.ofList (C :: Tensor.to inSpatial (List Nat)))) :
  IO (TensorRef α
    (Shape.ofList (C ::
      Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <| Runtime.Autograd.Tape.maxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) x.id
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.maxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) x.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .maxPool #[x.identity?] cpu cuda

/--
N-D average pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on the
spatial rank `d`.
-/
def avgPool {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
  (x : TensorRef α (Shape.ofList (C :: Tensor.to inSpatial (List Nat)))) :
  IO (TensorRef α
    (Shape.ofList (C ::
      Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <| Runtime.Autograd.Tape.avgPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) x.id
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.avgPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) x.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .avgPool #[x.identity?] cpu cuda

/--
N-D smooth max pooling (log-sum-exp surrogate) for channels-first tensors `(C, spatial...)`.

This is a differentiable approximation to max pooling; PyTorch does not expose it as a single
primitive, but it can be emulated with `logsumexp` over local windows. Executable backends require
at least one spatial dimension and a finite, nonzero `beta`; evaluation uses an input-space
max/min shift so the exponential weights remain stable for large finite values.
-/
def smoothMaxPool {α : Type} [TorchLean.Storage α] [TensorTransfer α] (s : EagerSession α)
  [Context α]
  [DecidableEq α]
  {d C : Nat} {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]}
  (x : TensorRef α (Shape.ofList (C :: Tensor.to inSpatial (List Nat)))) (beta : α) :
  IO (TensorRef α
    (Shape.ofList (C ::
      Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.smoothMaxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) x.id beta
    pure { id := id }
  let cuda := do
    let betaF ← TensorTransfer.toFloat (α := α) beta
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.smoothMaxPool (t := t0)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      x.id betaF
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .smoothMaxPool #[x.identity?] cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
