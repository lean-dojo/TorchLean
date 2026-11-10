/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Builder
public import TorchLean.Verify.CROWN.FlatBox.Transfer

/-!
# Sequential PINN Architecture

The PINN architecture class of the verification pipeline: fully-connected feed-forward networks
with one shared hidden activation between linear layers. That class covers the corridor networks of
the PINN and ODE checkers. Convolutional, residual, Fourier-feature and multi-branch physics models
are outside it and need their own architecture records.
-/

@[expose] public section


namespace TorchLean.Examples.PINN

/-- Supported hidden activation functions between linear layers. -/
inductive HiddenActivation where
  /-- The hyperbolic tangent, `Op.tanh`. -/
  | tanh
  /-- The rectified linear unit, `Op.relu`. -/
  | relu
  /-- The sine, `Op.sin`. -/
  | sin
  deriving DecidableEq, Repr

/-- Architectural description for a sequential fully-connected PINN/corridor network. -/
structure Architecture where
  /-- Input dimension. -/
  inputDim   : Nat
  /-- Hidden layer widths, in order. -/
  hiddenDims : Array Nat
  /-- Output dimension. -/
  outputDim  : Nat
  /-- Shared hidden activation function. -/
  activation : HiddenActivation := .tanh
  deriving DecidableEq, Repr

/-- Dimensions (input, output) for each linear layer, in order. -/
def Architecture.linearDims (arch : Architecture) : Array (Nat × Nat) :=
  let (_, dims) := (arch.hiddenDims.push arch.outputDim).foldl
    (fun (inDim, dims) outDim => (outDim, dims.push (inDim, outDim)))
    (arch.inputDim, #[])
  dims

namespace HiddenActivation

/-- The hidden activation as a registered operator on vectors of `n` entries. -/
def operator : HiddenActivation → (n : Nat) → IR.Operator
  | .tanh, n => ⟨_, _, Op.tanh (s := [n])⟩
  | .relu, n => ⟨_, _, Op.relu (s := [n])⟩
  | .sin, n => ⟨_, _, Op.sin (s := [n])⟩

end HiddenActivation

/-- The network of `arch` with the linear layers `layers`, in order, as a graph of registered
operators. Graph input `0` has `arch.inputDim` entries, each layer adds its weight and bias as
constants and an `Op.linear` node, and the hidden activation sits between consecutive layers. The
graph's one output is the last layer, and the result is an error unless the layers have the
dimensions of `arch.linearDims`. -/
def Architecture.model {α : Type} [Storage α] [Context α] (arch : Architecture)
    (layers : Array (SomeLinear α)) :
    Except String (IR.Graph × IR.Payload α) := do
  unless layers.map (fun l => (l.inDim, l.outDim)) == arch.linearDims do
    throw s!"PINN model: layer dimensions {layers.map fun l => (l.inDim, l.outDim)} are not the \
      architecture's {arch.linearDims}"
  IR.Graph.mlp (HiddenActivation.operator arch.activation) layers

end TorchLean.Examples.PINN
