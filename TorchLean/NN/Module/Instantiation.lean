/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Module.Objective

/-!
# Module Instantiation

`ObjectiveDef.instantiate` turns a scalar-objective definition into a live `Objective`. The initial
state is a supplied tensor pack or the definition's `Float` initializers cast to the runtime
scalar. A storage-first initialization plan in the definition is only validated here.
`Init.applyPlan` fills parameter storage from such a plan.
`TorchLean.Module.RuntimeState.instantiate` applies it when no initial state is supplied.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

namespace ObjectiveDef

/--
Instantiate an `ObjectiveDef` under a runtime scalar and configuration.

The definition must carry one `requiresGrad` flag per state shape and pass `d.validate`. Its
runtime initialization plan, when present, is validated and not applied. A supplied
`initialState?` is the initial state as given, with no `Float` intermediate. Otherwise `cast`
converts the stored `Float` initializers of `d`.

`runtime` selects the device and arithmetic policy. Its default uses the standard
runtime settings.
-/
def instantiate {α β : Type} [Storage α] [Storage β]
    [Context α]
    [TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (d : ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (cast : Float → α) (runtime : Session.Config := {})
    (initialState? : Option (TensorPack α stateShapes) := none) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) := do
  IO.ofExcept (validateStatic "objective" d.requiresGrad d.validate d.runtimeInit)
  let initState : TensorPack α stateShapes :=
    match initialState? with
    | some state => state
    | none => Pack.map (fun tensor => Tensor.map cast tensor) d.initState
  Objective.create (α := α) (stateShapes := stateShapes) (inputShapes := inputShapes)
    (dataInputShapes := dataInputShapes)
    (runtime := runtime) (requiresGrad := d.requiresGrad)
    (validateDataInputs := d.validateDataInputs)
    (loss := d.loss (α := α)) initState

end ObjectiveDef

end TorchLean
