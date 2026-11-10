/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.ScalarTrainer.ParamList
public import TorchLean.NN.ScalarTrainer.EagerOps
public import TorchLean.NN.Session.Optimizer

/-!
# Recording Trainer Inputs

Keep parameter leaves in pack order so backward gradients and native optimizer state use
the same keys on every forward pass.
-/

public section

namespace TorchLean

open TorchLean.Tensor

/-- Read one tensor per recorded reference, in reference order. -/
def RefList.read {α : Type} [Storage α]
    (read : {s : Shape} → TensorRef α s → IO (Tensor α s)) :
    {ss : List Shape} → Pack (TensorRef α) ss → IO (TensorPack α ss)
  | .nil, .nil => pure .nil
  | _ :: _, .cons ref refs => do
      let value ← read ref
      pure (.cons value (← RefList.read read refs))

/-- Read parameter gradients in the same order as the recorded references. -/
def RefList.grads {α : Type} [Storage α] {ss : List Shape}
    (gradients : Array (SomeTensor α)) (refs : Pack (TensorRef α) ss) :
    IO (TensorPack α ss) :=
  read (fun {s} ref => Session.grad (α := α) (sh := s) gradients ref) refs

/-- Record parameter leaves in pack order, preserving each parameter's gradient setting. -/
def ParamList.use {α : Type} [Storage α] [TensorTransfer α] :
    {ss : List Shape} → ParamList α ss → SessionM α (Pack (TensorRef α) ss)
  | .nil, .nil => pure .nil
  | s :: ss, .cons parameter parameters => fun session => do
      let ref ← Session.use (α := α) (sh := s) session parameter
      let refs ← use (α := α) (ss := ss) parameters session
      pure (.cons ref refs)

/-- Record input tensors as tape leaves in pack order. -/
def Session.useInputs {α : Type} [Storage α] [TensorTransfer α] :
    {ss : List Shape} → TensorPack α ss → SessionM α (Pack (TensorRef α) ss)
  | .nil, .nil => pure .nil
  | s :: ss, .cons input inputs => fun session => do
      let ref ← Session.input (α := α) (sh := s) session input
      let refs ← useInputs (α := α) (ss := ss) inputs session
      pure (.cons ref refs)

end TorchLean
