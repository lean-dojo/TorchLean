/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.VJPOp
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Activation

/-!
# MLP (spec wiring example)

A 2-layer MLP, `Linear → ReLU → Linear`, composed from the spec layers `Linear.forward` and
`Tensor.relu`, with its hand-written backward pass.
`TorchLean.Examples.Sequential.MLP.eval_model` proves that the same model as an `nn.Sequential`
evaluates to `MLP.forward`. We wrote `MLP.backward` from the chain rule, with ReLU's slope `0` at
`z = 0`. On the torch side, the test `TorchLean.Test.PyTorchRoundtrip.run` compares a 2 → 3 → 1
model of this layout, built with `nn.linear`, with PyTorch in float64 at the input `[0.5, 0.8]`.

## Implementation status

No `nn.models` builder exists for this model. In the API it is a composition of `nn.linear` and
`nn.op .relu`, or `nn.mlp inDim outDim { hiddenWidths := [hidDim] }`, whose default activation is
ReLU. The sequential model and its results are in `TorchLeanExamples/DeepDives/Sequential/MLP.lean`.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

/-- A 2-layer MLP on one input vector: `Linear(inDim → hidDim)`, then `ReLU`, then
`Linear(hidDim → outDim)`, as PyTorch's `nn.Sequential(nn.Linear, nn.ReLU, nn.Linear)`. -/
def MLP.forward
  {α : Type} [Storage α] [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Linear α inDim hidDim)
  (l2 : Linear α hidDim outDim)
  (x : Tensor α [inDim]) :
  Tensor α [outDim] :=
  let z1 := Linear.forward (α:=α) l1 x
  let a1 := Tensor.relu z1
  Linear.forward (α:=α) l2 a1

/-- Backward pass for the 2-layer MLP with upstream gradient `dLdy`.
Returns (∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2, ∂L/∂x).

The ReLU derivative is `1` where the pre-activation is positive and `0` elsewhere, including at
zero.
-/
def MLP.backward
  {α : Type} [Storage α] [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Linear α inDim hidDim)
  (l2 : Linear α hidDim outDim)
  (x : Tensor α [inDim])
  (dLdy : Tensor α [outDim]) :
  ( Tensor α [hidDim, inDim]
  × Tensor α [hidDim]
  × Tensor α [outDim, hidDim]
  × Tensor α [outDim]
  × Tensor α [inDim] ) :=

  -- Forward intermediates
  let z1 := Linear.forward (α:=α) l1 x
  let a1 := Tensor.relu z1

  -- Layer 2 grads
  let dW2 := Linear.weightsDeriv (α:=α) a1 dLdy
  let da1 := Linear.inputDeriv (α:=α) l2.weights dLdy

  -- ReLU backprop
  let grad_relu := Tensor.map Activation.reluDeriv z1
  let dz1 := grad_relu * da1

  -- Layer 1 grads
  let dW1 := Linear.weightsDeriv (α:=α) x dz1
  let dX  := Linear.inputDeriv (α:=α) l1.weights dz1

  (dW1, dz1, dW2, dLdy, dX)

/-- `VJPOp` for the same 2-layer MLP, composed with `VJPOp.compose` from `VJPOp`s for the
two linear layers and ReLU.

An `VJPOp` pairs a forward definition with a reverse-mode definition, which suits symbolic
gradient checks. The backward pass returns the input gradient only, so the weights and biases act
as constants. `Examples.Sequential.MLP.mlpOpspec_backward` proves that this gradient is the last
component of `backward`.
-/
def MLP.vjpOp
  {α : Type} [Storage α] [Context α]
  {inDim hidDim outDim : Nat}
  (l1 : Linear α inDim hidDim)
  (l2 : Linear α hidDim outDim) :
  VJPOp α ([inDim]) ([outDim]) :=
  let lin1 : VJPOp α [inDim] [hidDim] :=
    { forward := Linear.forward (α := α) l1
      backward := fun _ g => Linear.inputDeriv (α := α) l1.weights g }
  let relu : VJPOp α [hidDim] [hidDim] :=
    { forward := Tensor.map Activation.relu
      backward := fun x g => Tensor.map Activation.reluDeriv x * g }
  let lin2 : VJPOp α [hidDim] [outDim] :=
    { forward := Linear.forward (α := α) l2
      backward := fun _ g => Linear.inputDeriv (α := α) l2.weights g }
  VJPOp.compose (α:=α)
    (VJPOp.compose (α:=α) lin1 relu)
    lin2

end TorchLean
