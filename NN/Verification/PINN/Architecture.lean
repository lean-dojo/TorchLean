/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Core

/-!
# Sequential PINN Architecture

Sequential fully-connected PINN architecture helpers for TorchLean verification.

This module covers the PINN architecture class used by the verification pipeline:
fully-connected feed-forward networks with one shared hidden activation between linear layers.
That is enough for the corridor networks used by the PINN/ODE checkers, but it is not a complete
taxonomy of all PINN architectures.  Convolutional PINNs, residual PINNs, Fourier-feature PINNs,
and multi-branch physics models should get their own architecture records rather than overloading
this sequential MLP description.
-/

@[expose] public section

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph

namespace NN.Verification.PINN

/-- Supported hidden activation functions between linear layers. -/
inductive HiddenActivation where
  | tanh
  | relu
  | sin
  deriving DecidableEq, Repr

/-- Architectural description for a sequential fully-connected PINN/corridor network. -/
structure SequentialPINNArch where
  /-- Input dimension. -/
  inputDim   : Nat
  /-- Hidden layer widths, in order. -/
  hiddenDims : Array Nat
  /-- Output dimension. -/
  outputDim  : Nat
  /-- Shared hidden activation function. -/
  activation : HiddenActivation := .tanh
  deriving DecidableEq, Repr

namespace SequentialPINNArch

/-- Number of linear layers in the architecture. -/
def linearLayerCount (arch : SequentialPINNArch) : Nat :=
  arch.hiddenDims.size + 1

/-- Dimensions (input, output) for each linear layer, in order. -/
def linearDims (arch : SequentialPINNArch) : Array (Nat × Nat) :=
  let (_, dims) := (arch.hiddenDims.push arch.outputDim).foldl
    (fun (inDim, dims) outDim => (outDim, dims.push (inDim, outDim)))
    (arch.inputDim, #[])
  dims

/-- Node id assigned to the k-th linear layer (0-indexed). -/
def linearNodeId (idx : Nat) : Nat :=
  2 * idx + 1

/-- Id of the terminal linear layer (network output). -/
def outputNodeId (arch : SequentialPINNArch) : Nat :=
  match arch.linearLayerCount with
  | 0 => 0
  | Nat.succ n => linearNodeId n

namespace Internal

/-- Internal: map a `PINN.HiddenActivation` to the corresponding `NN.IR.OpKind`. -/
def activationOpKind : HiddenActivation → OpKind
  | .tanh => NN.IR.OpKind.tanh
  | .relu => NN.IR.OpKind.relu
  | .sin  => NN.IR.OpKind.sin

/--
Internal: worker for `buildGraph`.

Implementation note: exported definitions must not depend on private helpers.
-/
def buildNodes
    (activation : HiddenActivation)
    (remaining : Array (Nat × Nat))
    (prevId nextId : Nat)
    (acc : Array Node) : Array Node := Id.run do
  let mut previous := prevId
  let mut fresh := nextId
  let mut nodes := acc
  for i in [:remaining.size] do
    let (_, outDim) := remaining[i]!
    nodes := nodes.push
      { id := fresh
        parents := #[previous]
        kind := NN.IR.OpKind.linear
        outShape := .dim outDim .scalar }
    previous := fresh
    fresh := fresh + 1
    if i + 1 < remaining.size then
      nodes := nodes.push
        { id := fresh
          parents := #[previous]
          kind := activationOpKind activation
          outShape := .dim outDim .scalar }
      previous := fresh
      fresh := fresh + 1
  return nodes

end Internal

/-- Build a computation graph matching the supplied sequential PINN architecture. -/
def buildGraph (arch : SequentialPINNArch) : Graph :=
  let nodes : Array Node :=
    Internal.buildNodes arch.activation arch.linearDims 0 1
      #[{ id := 0, parents := #[], kind := NN.IR.OpKind.input,
          outShape := .dim arch.inputDim .scalar }]
  { nodes := nodes }

/-- Output node id for an arbitrary graph, assuming the last node is the network output. -/
def graphOutputId (g : Graph) : Nat :=
  match g.nodes.back? with
  | some node => node.id
  | none => 0

end SequentialPINNArch
end NN.Verification.PINN
