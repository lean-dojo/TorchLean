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

/-! ## Convolution operations -/

/--
N-D convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (w : TensorRef α (Shape.ofList (outC :: inC :: Tensor.to kernel (List Nat))))
  (b : TensorRef α [outC])
  (x : TensorRef α (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))) :
  IO (TensorRef α
    (Shape.ofList (outC ::
      Tensor.to (Spec.convOutSpatial inSpatial kernel stride padding) (List Nat)))) := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <| Runtime.Autograd.Tape.conv (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <| Runtime.Autograd.Cuda.Tape.conv (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .conv #[w.identity?, b.identity?, x.identity?] cpu cuda

/--
N-D transpose convolution for channels-first tensors `(inC, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} [TorchLean.Storage α] (s : EagerSession α) [Context α]
  {d inC outC : Nat}
  {kernel stride padding : TorchLean.Tensor Nat [d]}
  {inSpatial : TorchLean.Tensor Nat [d]}
  (w : TensorRef α (Shape.ofList (inC :: outC :: Tensor.to kernel (List Nat))))
  (b : TensorRef α [outC])
  (x : TensorRef α (Shape.ofList (inC :: Tensor.to inSpatial (List Nat)))) :
  IO (TensorRef α
    (Shape.ofList (outC ::
      Tensor.to (Spec.convTransposeOutSpatial inSpatial kernel stride padding) (List Nat))))
    := do
  let cpu := do
    let id ← s.recordCpu fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Tape.convTranspose (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id
    pure { id := id }
  let cuda := do
    let id ← s.recordCuda fun t0 => keepTapeOnError t0 <|
      Runtime.Autograd.Cuda.Tape.convTranspose (t := t0)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      w.id b.id x.id
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .convTranspose #[w.identity?, b.identity?, x.identity?] cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
