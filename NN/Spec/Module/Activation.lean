/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Module.Core

/-!
# Activation modules

Shape-preserving activation specifications packaged for module composition.
-/

@[expose] public section


namespace Spec.Module

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- ReLU as a shape-preserving module. -/
def relu {α : Type} [TorchLean.Storage α] [Zero α] [Max α] [BEq α]
    (s : Shape) : Spec.Module α s s :=
  { forward := Activation.reluSpec, kind := "ReLU", pythonExpr := "nn.ReLU()" }

/-- Sigmoid as a shape-preserving module. -/
def sigmoid (s : Shape) : Spec.Module α s s :=
  { forward := Activation.sigmoidSpec, kind := "Sigmoid", pythonExpr := "nn.Sigmoid()" }

/-- Hyperbolic tangent as a shape-preserving module. -/
def tanh (s : Shape) : Spec.Module α s s :=
  { forward := Activation.tanhSpec, kind := "Tanh", pythonExpr := "nn.Tanh()" }

/-- Softmax along an explicitly selected tensor dimension.

In PyTorch terms: `torch.softmax(x, dim=axis)`. -/
def softmax (s : Shape) (axis : Nat) [Shape.AxisInBounds axis s] :
    Spec.Module α s s :=
  { forward := Activation.softmaxSpec axis
    kind := "Softmax"
    pythonExpr := s!"nn.Softmax(dim={axis})" }

end Spec.Module
