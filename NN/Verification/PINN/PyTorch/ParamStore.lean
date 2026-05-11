/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.CrownParamstore
public import NN.Verification.PINN.PyTorch.Load

/-!
# PINN PyTorch ParamStore Bridge

Build a CROWN-style graph parameter store from a PyTorch-trained PINN checkpoint.

The verification CLIs run PINNs through the graph backend. For that, we need a `ParamStore` keyed by
the node ids that `SequentialPINNArch.buildGraph` uses. Keeping this file under
`NN.Verification.PINN` makes the ownership clear: this is not a generic PyTorch example, it is the
checkpoint bridge for PINN verification.
-/

@[expose] public section


namespace Import
namespace PINNPyTorch

open Spec
open Tensor
open Shape

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.PINN

/- Convert a `PinnLayer` into the graph backend's `LinParams` container. -/
namespace Internal

/--
Convert a `PinnLayer` into the graph backend's `LinParams` container.

The declaration remains a named helper because exported PINN graph assembly refers to it directly.
-/
def layerToLinParams (layer : PinnLayer) : LinParams Float :=
  { m := layer.outDim, n := layer.inDim, w := layer.weights, b := layer.bias }

end Internal

/-- Build a `ParamStore Float` for the PINN graph from a loaded state. -/
def toParamStore (sd : PinnState) : ParamStore Float :=
  CROWNParamStore.ofLinearStack
    (nodeIdOfIndex := SequentialPINNArch.linearNodeId)
    (layers := sd.layers.map Internal.layerToLinParams)

/--
Convert a loaded float state dict to a `ParamStore` over an arbitrary scalar `α`.

This is useful when you want to reuse the same trained parameters for:

- executable backends (`Float`, `IEEE32Exec`), or
- proof-level backends (e.g. `ℝ`), by supplying an appropriate `ofFloat` cast.
-/
def toParamStoreWith {α : Type} [Context α] (ofFloat : Float → α) (sd : PinnState) : ParamStore α :=
  CROWNParamStore.ofLinearStackWith
    (α := α)
    (ofFloat := ofFloat)
    (nodeIdOfIndex := SequentialPINNArch.linearNodeId)
    (layers := sd.layers.map Internal.layerToLinParams)

/-- Build the computation graph corresponding to a loaded state. -/
def buildGraph (sd : PinnState) : Graph :=
  SequentialPINNArch.buildGraph sd.arch

end PINNPyTorch
end Import
