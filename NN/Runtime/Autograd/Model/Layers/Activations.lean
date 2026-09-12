/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Core
public import NN.Runtime.Autograd.Model.Functional.Core
public import NN.Runtime.Autograd.Model.Functional.ShapeOps

/-!
# TorchLean NN: Activation and Shape Layers
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

namespace Layers

/--
ReLU activation layer (no parameters).

PyTorch analogues: `torch.nn.ReLU` / `torch.nn.functional.relu`.
-/
def relu {s : Shape} : Layer s s :=
  { kind := "ReLU"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.Model.relu (m := m) (α := α) (s := s) x
  }

/--
SiLU (a.k.a. swish) activation layer (no parameters).

PyTorch analogues: `torch.nn.SiLU` / `torch.nn.functional.silu`.
-/
def silu {s : Shape} : Layer s s :=
  { kind := "SiLU"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.Torch.silu (m := m) (α := α) (s := s) x
  }

/--
GELU activation layer (no parameters), using the tanh approximation.

PyTorch analogues: `torch.nn.GELU(approximate='tanh')` /
`torch.nn.functional.gelu(x, approximate='tanh')`. PyTorch's default `nn.GELU()` uses the exact
erf form, which TorchLean does not implement.
-/
def gelu {s : Shape} : Layer s s :=
  { kind := "GELU"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.Torch.gelu (m := m) (α := α) (s := s) x
  }

/--
Sigmoid activation layer (no parameters).

PyTorch analogy: `torch.sigmoid`.
-/
def sigmoid {s : Shape} : Layer s s :=
  { kind := "Sigmoid"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.Model.sigmoid (m := m) (α := α) (s := s) x
  }

/--
Hyperbolic tangent activation layer (no parameters).

PyTorch analogy: `torch.tanh`.
-/
def tanh {s : Shape} : Layer s s :=
  { kind := "Tanh"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.Model.tanh (m := m) (α := α) (s := s) x
  }

/-- Shape-preserving softmax layer along `axis`. -/
def softmax {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] : Layer s s :=
  { kind := "Softmax"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => F.softmax (m := m) (α := α) (s := s) axis x
  }

/-- Shape-preserving stable log-softmax layer along `axis`. -/
def logSoftmax {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] : Layer s s :=
  { kind := "LogSoftmax"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => F.logSoftmax (m := m) (α := α) (s := s) axis x
  }

/--
Pointwise square `x ↦ x^2` (no parameters).

PyTorch analogy: `torch.square(x)` / `x.square()`.
-/
def square {s : Shape} : Layer s s :=
  { kind := "Square"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.Model.F.square (m := m) (α := α) (s := s) x
  }

/--
Sum-reduce all elements of the input to a scalar (no parameters).

PyTorch analogy: `x.sum()`.
-/
def sum {s : Shape} : Layer s Shape.scalar :=
  { kind := "Sum"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.Model.sum (m := m) (α := α) (s := s) x
  }

/--
Flatten any tensor to a 1D vector of length `Spec.Shape.size s` (no parameters).

PyTorch analogy: `torch.flatten(x)` or `x.reshape(-1)`.
-/
def flatten {s : Shape} : Layer s [Spec.Shape.size s] :=
  { kind := "Flatten"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.Model.flatten (m := m) (α := α) (s := s) x
  }

/--
Dropout layer controlled by `Mode`.

- In `Mode.train`, randomly zeroes entries with probability `p`.
- In `Mode.eval`, it is the identity.
- At `p = 1`, training returns zero exactly rather than evaluating the undefined scale `1 / (1-p)`.

We store `p` as a scalar parameter tensor (with `requiresGrad := false`) so it can be threaded
through the unified parameter list without being optimized.

PyTorch analogy: `torch.nn.Dropout(p)` / `torch.nn.functional.dropout(x, p, training=...)`.
-/
def dropout {s : Shape} (p : Float) (seed : Nat := 0) : Layer s s :=
  let pShape : Shape := Shape.scalar
  let p0 : Tensor Float pShape := Tensor.scalar p
  { kind := s!"Dropout(p={p})"
    stateShapes := [pShape]
    initState := .cons p0 .nil
    runtimeInit := some (.cons (.flat (FloatArray.mk #[p])) .nil)
    requiresGrad := #[false]
    validateConfig :=
      if p.isFinite && 0.0 <= p && p <= 1.0 then
        pure ()
      else
        .error s!"Dropout: probability must be finite and in [0, 1], got {p}"
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun pRef x =>
          (show m (RefTy (m := m) (α := α) s) from
            if mode == .eval || p == 0.0 then
              pure x
            else if p == 1.0 then
              Runtime.Autograd.Torch.scale (m := m) (α := α) (s := s) x 0
            else
              Runtime.Autograd.Model.F.dropoutRefSeeded
                (m := m) (α := α) (s := s) x pRef seed)
  }
end Layers

end Model
end Autograd
end Runtime
