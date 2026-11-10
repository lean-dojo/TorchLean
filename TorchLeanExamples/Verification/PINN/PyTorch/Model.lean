/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.PINN.Architecture
public import TorchLeanExamples.Verification.PINN.PyTorch.Load

/-!
# PINN PyTorch model

The graph of registered operators of a PyTorch-trained PINN checkpoint.

The verification CLIs run PINNs as graphs of registered operators, and `model` builds the
checkpoint's graph with `Architecture.model` after converting its weights and biases to the
chosen scalar.
-/

@[expose] public section

namespace TorchLean
namespace Examples.PINN.Checkpoint


/--
The graph of registered operators of a loaded checkpoint over an arbitrary scalar `α`, each weight
and bias entry converted by `ofFloat`. It fails when the layer dimensions differ from
`sd.arch.linearDims`.

The same trained parameters thus serve the executable backends (`Float`,
`Binary32`) and, with an appropriate `ofFloat`, proof-level backends.
-/
def model {α : Type} [Storage α] [Context α] (ofFloat : Float → α) (sd : Checkpoint) :
    Except String (IR.Graph × IR.Payload α) :=
  sd.arch.model <| sd.layers.map fun layer =>
    { inDim := layer.inDim, outDim := layer.outDim
      spec := { weights := Tensor.map ofFloat layer.spec.weights
                bias := Tensor.map ofFloat layer.spec.bias } }

end Examples.PINN.Checkpoint
end TorchLean
