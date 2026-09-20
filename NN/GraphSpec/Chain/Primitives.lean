/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Syntax
public import NN.Runtime.Autograd.Model.Layers.Activations
public import NN.Runtime.Autograd.Model.Layers.Recurrent

/-!
# Standard sequential GraphSpec primitives

This module provides dense linear, ReLU, and axis-wise softmax primitives together with their
corresponding `Chain` constructors. Each primitive supplies pure tensor semantics, executable
TorchLean lowering, and deterministic layer initialization.
-/

@[expose] public section

namespace NN
namespace GraphSpec

open Spec TorchLean
open TorchLean.Tensor

namespace Primitive

/--
The affine map $x \mapsto Wx + b$ from vectors of length `inDim` to vectors of length `outDim`.

Its parameter ABI is `[[outDim, inDim], [outDim]]`. The corresponding TorchLean layer initializes
the weight from the layer occurrence index and initializes the bias exactly to zero.
-/
def linear (inDim outDim : Nat) :
    Primitive
      [[outDim, inDim], [outDim]]
      [inDim] [outDim] :=
  { name := s!"linear({inDim},{outDim})"
    specFwd := fun {α} _storage _ctx params x =>
      match params with
      | .cons w (.cons b .nil) =>
          let lin : Spec.LinearSpec α inDim outDim := { weights := w, bias := b }
          Spec.linearSpec (α := α) lin x
    program := fun {α} _storage _ctx =>
      fun {m} _instM _instOps =>
        fun w b x =>
          Runtime.Autograd.Torch.linear (m := m) (α := α)
            (inDim := inDim) (outDim := outDim) w b x
    toLayerM? := some (fun i =>
      ⟨ Runtime.Autograd.Model.Layers.linear inDim outDim
          (weightSeed := i)
      , by rfl ⟩)
    countsAsLayer := true
  }

/-- Parameter-free elementwise ReLU on tensors of shape `s`. -/
def relu (s : Shape) : Primitive [] s s :=
  { name := "relu"
    specFwd := fun {α} _storage _ctx _params x => Activation.reluSpec (α := α) x
    program := fun {α} _storage _ctx =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.Model.relu (m := m) (α := α) (s := s) x
    toLayerM? := some (fun _i => ⟨Runtime.Autograd.Model.Layers.relu (s := s), by rfl⟩)
    countsAsLayer := false
  }

/--
Parameter-free softmax along `axis`. The axis bound is carried as an instance so ill-shaped
softmax nodes cannot be constructed.
-/
def softmax (s : Shape) (axis : Nat) [Spec.Shape.AxisInBounds axis s] : Primitive [] s s :=
  { name := "softmax"
    specFwd := fun {α} _storage _ctx _params x => Activation.softmaxSpec (α := α) axis x
    program := fun {α} _storage _ctx =>
      fun {m} _ _ =>
        fun x => Runtime.Autograd.Model.F.softmax (m := m) (α := α) axis x
    toLayerM? := some (fun _i =>
      ⟨Runtime.Autograd.Model.Layers.softmax (s := s) axis, by rfl⟩)
    countsAsLayer := false
  }

end Primitive

namespace Chain

/-- Chain constructor for `Primitive.linear`. -/
def linear (inDim outDim : Nat) :
    Chain [[outDim, inDim], [outDim]] [inDim] [outDim] :=
  .prim (Primitive.linear inDim outDim)

/-- Chain constructor for `Primitive.relu`. -/
def relu (s : Shape) : Chain [] s s :=
  .prim (Primitive.relu s)

/-- Chain constructor for `Primitive.softmax`. -/
def softmax (s : Shape) (axis : Nat) [Spec.Shape.AxisInBounds axis s] : Chain [] s s :=
  .prim (Primitive.softmax s axis)

end Chain

end GraphSpec
end NN
