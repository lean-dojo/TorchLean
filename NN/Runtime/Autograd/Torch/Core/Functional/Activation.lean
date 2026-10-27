/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Functional.Tensor

/-!
# Activations over Backend References

Elementwise nonlinearities and final-axis softmax through `Ops`.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

-- Re-export the capability operations; only argument adapters and compositions need bodies.
export Ops
  (relu sigmoid silu tanh softmaxLast softplus exp sin cos log inv logSoftmaxLast gelu topKMask)

variable {m : Type → Type} {α : Type} [Storage α] [Context α] [Monad m]
    [Ops (m := m) (α := α)]

@[inherit_doc Ops.safeLog]
def safeLog {s : Shape} (x : Ref (m := m) (α := α) s) (ε : α := Context.defaultEpsilon) :
    m (Ref (m := m) (α := α) s) :=
  Ops.safeLog (m := m) (α := α) (s := s) x ε

end Runtime.Autograd.Torch
