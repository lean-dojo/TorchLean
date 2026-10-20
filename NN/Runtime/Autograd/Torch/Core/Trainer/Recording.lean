/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Parameters
public import NN.Runtime.Autograd.Torch.Core.Trainer.EagerOps
public import NN.Runtime.Autograd.Torch.Core.BackwardOptim

/-!
# Recording Trainer Inputs

Keep parameter leaves in pack order so backward gradients and native optimizer state use
the same keys on every forward pass.
-/

public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

namespace Internal

/-- Read one tensor per recorded reference, in reference order. -/
def readRefs {α : Type} [TorchLean.Storage α]
    (read : {s : Shape} → TensorRef α s → IO (Tensor α s)) :
    {ss : List Shape} → RefList (TensorRef α) ss → IO (TorchLean.TensorPack α ss)
  | .nil, .nil => pure .nil
  | _ :: _, .cons ref refs => do
      let value ← read ref
      pure (.cons value (← readRefs read refs))

/-- Read parameter gradients in the same order as the recorded references. -/
def gradsOfRefs {α : Type} [TorchLean.Storage α] {ss : List Shape}
    (gradients : Array (Spec.SomeTensor α)) (refs : RefList (TensorRef α) ss) :
    IO (TorchLean.TensorPack α ss) :=
  readRefs (fun {s} ref => Internal.EagerSession.grad (α := α) (sh := s) gradients ref) refs

/-- Record parameter leaves in pack order, preserving each parameter's gradient setting. -/
def useParams {α : Type} [TorchLean.Storage α] [TensorTransfer α] :
    {ss : List Shape} → ParamList α ss → EagerM α (RefList (TensorRef α) ss)
  | .nil, .nil => pure .nil
  | s :: ss, .cons parameter parameters => fun session => do
      let ref ← Internal.EagerSession.use (α := α) (sh := s) session parameter
      let refs ← useParams (α := α) (ss := ss) parameters session
      pure (.cons ref refs)

/-- Record input tensors as tape leaves in pack order. -/
def useInputs {α : Type} [TorchLean.Storage α] [TensorTransfer α] :
    {ss : List Shape} → TorchLean.TensorPack α ss → EagerM α (RefList (TensorRef α) ss)
  | .nil, .nil => pure .nil
  | s :: ss, .cons input inputs => fun session => do
      let ref ← Internal.EagerSession.input (α := α) (sh := s) session input
      let refs ← useInputs (α := α) (ss := ss) inputs session
      pure (.cons ref refs)

end Internal

end Runtime.Autograd.Torch
