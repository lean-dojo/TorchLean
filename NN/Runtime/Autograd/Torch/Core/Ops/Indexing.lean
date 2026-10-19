/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Dispatch
public import NN.Runtime.Autograd.Engine.Core.Indexing
public import NN.Runtime.Autograd.Engine.LibTorch.Ops.Indexing

/-!
# Eager Tensor Operations

PyTorch-style tensor operations recorded on the session's tape. Each one runs the CPU or the CUDA
kernel of the operation, as the session's device selects, and preserves the typed `TensorRef`
surface.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean TorchLean.Tensor

namespace Internal

namespace EagerSession

/-! ## Indexing operations -/

/-- Select one bounded coordinate from an arbitrary tensor axis. -/
def select {α : Type} [TorchLean.Storage α] (session : EagerSession α) [Zero α]
    {shape : Shape} (axis : Nat) [Shape.AxisInBounds axis shape]
    (x : TensorRef α shape) (index : Fin (Shape.axisSize shape axis)) :
    IO (TensorRef α (shape.eraseAxis axis)) := do
  let cpu := do
    session.recordCpu fun tape => keepTapeOnError tape <|
      Runtime.Autograd.Tape.select (t := tape) x.id axis index
  let cuda := do
    session.recordCuda fun tape => keepTapeOnError tape <|
      Runtime.Autograd.LibTorch.Tape.select (t := tape) x.id axis index
  executeRecorded (α := α) session .gather #[x.identity?] cpu cuda

/-- Select several bounded coordinates from an arbitrary tensor axis. -/
def indexSelect {α : Type} [TorchLean.Storage α] (session : EagerSession α) [Add α] [Zero α]
    {shape : Shape} (axis count : Nat)
    [Shape.AxisInBounds axis shape] (x : TensorRef α shape)
    (indices : Tensor (Fin (Shape.axisSize shape axis)) [count]) :
    IO (TensorRef α (shape.replaceAxis axis count)) := do
  let cpu := do
    session.recordCpu fun tape => keepTapeOnError tape <|
      Runtime.Autograd.Tape.indexSelect (t := tape) x.id axis count indices
  let cuda := do
    session.recordCuda fun tape => keepTapeOnError tape <|
      Runtime.Autograd.LibTorch.Tape.indexSelect (t := tape) x.id axis count indices
  executeRecorded (α := α) session .gather #[x.identity?] cpu cuda

/-- Add source slices into an arbitrary tensor axis at bounded coordinates. -/
def scatterAdd {α : Type} [TorchLean.Storage α] (session : EagerSession α) [Add α] [Zero α]
    {shape : Shape} (axis count : Nat)
    [Shape.AxisInBounds axis shape] (base : TensorRef α shape)
    (source : TensorRef α (shape.replaceAxis axis count))
    (indices : Tensor (Fin (Shape.axisSize shape axis)) [count]) : IO (TensorRef α shape) := do
  let cpu := do
    session.recordCpu fun tape => keepTapeOnError tape <|
      Runtime.Autograd.Tape.scatterAdd (t := tape) base.id source.id axis count indices
  let cuda := do
    session.recordCuda fun tape => keepTapeOnError tape <|
      Runtime.Autograd.LibTorch.Tape.scatterAdd (t := tape) base.id source.id axis count indices
  executeRecorded (α := α) session .scatterAdd #[base.identity?, source.identity?] cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
