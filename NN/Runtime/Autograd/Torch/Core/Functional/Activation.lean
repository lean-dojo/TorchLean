/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Functional.Tensor

/-!
# Activations over Backend References

Elementwise nonlinearities and final-axis softmax through `Ops`. SiLU composes sigmoid and
multiplication.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

variable {m : Type → Type} {α : Type} [Storage α] [Context α] [Monad m]
    [Ops (m := m) (α := α)]

@[inherit_doc Ops.relu]
def relu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.relu (m := m) (α := α) x

@[inherit_doc Ops.sigmoid]
def sigmoid {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.sigmoid (m := m) (α := α) x

@[inherit_doc Ops.tanh]
def tanh {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.tanh (m := m) (α := α) x

@[inherit_doc Ops.softmaxLast]
def softmaxLast {s : Shape} (x : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) s) :=
  Ops.softmaxLast (m := m) (α := α) x

@[inherit_doc Ops.softplus]
def softplus {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.softplus (m := m) (α := α) x

@[inherit_doc Ops.exp]
def exp {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.exp (m := m) (α := α) x

@[inherit_doc Ops.sin]
def sin {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.sin (m := m) (α := α) x

@[inherit_doc Ops.cos]
def cos {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.cos (m := m) (α := α) x

@[inherit_doc Ops.log]
def log {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.log (m := m) (α := α) x

@[inherit_doc Ops.inv]
def inv {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.inv (m := m) (α := α) x

@[inherit_doc Ops.safeLog]
def safeLog {s : Shape} (x : Ref (m := m) (α := α) s) (ε : α := Context.defaultEpsilon) :
    m (Ref (m := m) (α := α) s) :=
  Ops.safeLog (m := m) (α := α) (s := s) x ε

@[inherit_doc Ops.logSoftmaxLast]
def logSoftmaxLast {s : Shape} (x : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) s) :=
  Ops.logSoftmaxLast (m := m) (α := α) (s := s) x

/-- SiLU (swish): `x * sigmoid(x)`. -/
def silu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := do
  let sx ← sigmoid (m := m) (α := α) (s := s) x
  mul (m := m) (α := α) (s := s) x sx

@[inherit_doc Ops.gelu]
def gelu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.gelu (m := m) (α := α) (s := s) x

end Runtime.Autograd.Torch
