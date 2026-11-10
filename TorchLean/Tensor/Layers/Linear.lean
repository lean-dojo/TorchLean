/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.AdjacentSwap
public import TorchLean.Tensor.Sequence
public import TorchLean.Tensor.Contract

/-!
# Linear layer (spec layer)

A fully connected layer maps an input vector `x` to `y = W x + b`. It is defined as a pure
shape-indexed function, once for a single vector and once (`forwardLeading`) for inputs with any
leading shape, each with its gradients ∂L/∂W, ∂L/∂b and ∂L/∂x. The `linear` arm of the operator
table (`TorchLean/Tensor/Op/Table/Rule.lean`) takes its forward, backward and tangent rules
from these definitions.

The gradients of `W` and `b` have the shapes of `W` and `b`, so they form a `Linear` themselves,
which `Linear.Gradients` pairs with the gradient of the input.

`SomeLinear` pairs a layer with its widths, for layers whose widths are read at run time
(checkpoints, graph payloads, bound queries).

On one input our `W x + b` is `torch.nn.functional.linear` (PyTorch 2.8 docs), `y = x Aᵀ + b`
with `A : (out_features, in_features)`. The sum is hand-written: output entry `i` adds
`W i k * x k` to zero in increasing `k`, and the bias comes after it
(`TorchLean.Tensor.getScalar_contract_matVec`). `Op.linear` runs `Linear.forwardLeading`
(`TorchLean.Op.rule?_linear_forward`).
-/

@[expose] public section


open TorchLean.Shape (size size_append)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α]
  [Add α] [Mul α] [Zero α]

/--
The parameters of a linear layer from `inDim` to `outDim` features: `weights` of shape
`[outDim, inDim]` and `bias` of shape `[outDim]`, the layout of PyTorch's `torch.nn.Linear` and
`torch.nn.functional.linear`.
-/
structure Linear (α : Type) [Storage α]
    (inDim outDim : Nat) where
  /-- Weight matrix with rows indexed by output features. -/
  weights : Tensor α [outDim, inDim]
  /-- Bias vector added to each output feature. -/
  bias    : Tensor α [outDim]
deriving DecidableEq, Repr

/-- A linear layer paired with the input and output widths that index its type. -/
structure SomeLinear (α : Type) [Storage α] where
  /-- Number of input features. -/
  inDim : Nat
  /-- Number of output features. -/
  outDim : Nat
  /-- The layer, indexed by its stored widths. -/
  spec : Linear α inDim outDim

/--
Unbatched forward pass: `y = W x + b`.

PyTorch analogue: `torch.nn.functional.linear`.
-/
def Linear.forward {inDim outDim : Nat}
  (m : Linear α inDim outDim)
  (input : Tensor α [inDim]) :
  Tensor α [outDim] :=
  Tensor.contract m.weights input _ _ [outDim] .matVec + m.bias

/-- Weight gradient of `y = W x + b`: the outer product `∂L/∂W = (∂L/∂y) ⊗ x`. -/
def Linear.weightsDeriv {inDim outDim : Nat}
  (input : Tensor α [inDim])
  (gradOutput : Tensor α [outDim]) :
  Tensor α [outDim, inDim] :=
  Tensor.zipWith (· * ·) (Tensor.broadcast gradOutput _ _ (Shape.broadcastDims_head _ _))
      (Tensor.broadcast input _ _ (Shape.broadcastDims_second _ _))

/--
Input gradient of `y = W x + b`: `∂L/∂x = Wᵀ (∂L/∂y)`, computed as the vector-matrix product of
`∂L/∂y` with `W`, so the transpose is never formed.
-/
def Linear.inputDeriv {inDim outDim : Nat}
  (weights : Tensor α [outDim, inDim])
  (gradOutput : Tensor α [outDim]) :
  Tensor α [inDim] :=
  Tensor.contract gradOutput weights _ _ [inDim] .vecMat

/--
Everything a linear backward pass produces: the two parameter gradients, as a `Linear` of the
same widths, and the gradient that travels on into the layer's input.

PyTorch analogue: `layer.weight.grad` and `layer.bias.grad` for `torch.nn.Linear`, with the
gradient of the input.
-/
structure Linear.Gradients (α : Type) [Storage α] (inDim outDim : Nat) where
  /-- Gradients of the weight matrix `W` and the bias vector `b`. -/
  parameters : Linear α inDim outDim
  /-- Gradient with respect to the layer input. -/
  inputGradient : Tensor α [inDim]
deriving Repr

/--
The affine map `W x + b` applied independently at every index of the leading shape `leading`.

PyTorch analogue: `torch.nn.functional.linear` on an input with leading axes.
-/
def Linear.forwardLeading (leading : Shape) {inDim outDim : Nat}
  (m : Linear α inDim outDim)
  (input : Tensor α (leading ++ [inDim])) :
  Tensor α (leading ++ [outDim]) :=
  Tensor.mapLeading leading (forward m) input

/--
The weight gradient of `forwardLeading`: the sum over the leading indices of the outer products of
the output cotangent and the primal input, which is `(gradOutputᵀ) · input` with the leading axes
flattened. Each entry is one left fold from zero in row-major order of the leading indices. At
`leading = []` that fold is `0 + g ⊗ x`, a different term from the vector rule
`weightsDeriv`, and an empty leading axis gives zero.
-/
def Linear.weightsDerivLeading (leading : Shape) {inDim outDim : Nat}
  (input : Tensor α (leading ++ [inDim]))
  (gradOutput : Tensor α (leading ++ [outDim])) :
  Tensor α [outDim, inDim] :=
  let inputFlat : Tensor α [Shape.size leading, inDim] :=
    Tensor.reshape input _ (by simp [Shape.size_append])
  let gradOutputFlat : Tensor α [Shape.size leading, outDim] :=
    Tensor.reshape gradOutput _ (by simp [Shape.size_append])
  Tensor.matmul (Tensor.permute gradOutputFlat
    (Shape.adjacentSwap [leading.size, outDim] 0) [outDim, leading.size]) inputFlat

/--
The bias gradient of `forwardLeading`: the sum of the output cotangent over the leading indices,
folded from zero in row-major order. At `leading = []` it is `0 + g`, where the vector rule
returns `g`, and an empty leading axis gives zero.
-/
def Linear.biasDerivLeading (leading : Shape) {outDim : Nat}
  (gradOutput : Tensor α (leading ++ [outDim])) :
  Tensor α [outDim] :=
  let gradOutputFlat : Tensor α [Shape.size leading, outDim] :=
    Tensor.reshape gradOutput _ (by simp [Shape.size_append])
  Tensor.castShape (Tensor.reduce .sum gradOutputFlat [⟨0, by simp⟩] false)
    (Shape.reduce_singleton _ _)

/--
The input gradient of `forwardLeading`: `Wᵀ g` at every index of the leading shape.
-/
def Linear.inputDerivLeading (leading : Shape) {inDim outDim : Nat}
  (weights : Tensor α [outDim, inDim])
  (gradOutput : Tensor α (leading ++ [outDim])) :
  Tensor α (leading ++ [inDim]) :=
  Tensor.mapLeading leading (inputDeriv weights) gradOutput

/--
The unbatched backward pass of a linear layer: from the parameters, the input `x` and the output
gradient `∂L/∂y` it returns ∂L/∂W, ∂L/∂b and ∂L/∂x. The bias gradient is `∂L/∂y` itself.
-/
def Linear.backward {inDim outDim : Nat}
  (layer : Linear α inDim outDim)
  (input : Tensor α [inDim])
  (gradOutput : Tensor α [outDim]) :
  Gradients α inDim outDim :=
  let d_weights := weightsDeriv input gradOutput
  let d_input := inputDeriv layer.weights gradOutput
  { parameters := { weights := d_weights, bias := gradOutput }, inputGradient := d_input }

end TorchLean
