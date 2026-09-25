/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Module.Objective

/-!
# Module Instantiation

Constructors that instantiate scalar-objective definitions from semantic tensors or storage-first
runtime initialization plans.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

namespace Module

namespace ObjectiveDef

/--
Instantiate an `ObjectiveDef` under a runtime scalar and configuration.

An explicit `initialState?` is used directly, without a `Float` intermediate. Otherwise the model's
stored Float initializers are converted with `cast`.

This is the most general constructor. The shorter `instantiate` entrypoint chooses standard runtime
settings before calling this function.
-/
def instantiateWith {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
        [Runtime.Autograd.Torch.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (d : ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (cast : Float → α) (runtime : Torch.Config)
    (initialState? : Option (TorchLean.TensorPack α stateShapes) := none) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) := do
  unless d.requiresGrad.size = stateShapes.length do
    throw <| IO.userError
      s!"objective: expected {stateShapes.length} requiresGrad flags, got {d.requiresGrad.size}"
  match d.validate with
  | .error message => throw <| IO.userError message
  | .ok () => pure ()
  match d.runtimeInit with
  | some plan =>
      match plan.validate with
      | .error message => throw <| IO.userError message
      | .ok () => pure ()
  | none => pure ()
  let initState : TorchLean.TensorPack α stateShapes :=
    match initialState? with
    | some state => state
    | none => castPack (α := α) cast d.initState
  Objective.create (α := α) (stateShapes := stateShapes) (inputShapes := inputShapes)
    (dataInputShapes := dataInputShapes)
    (runtime := runtime) (requiresGrad := d.requiresGrad)
    (validateDataInputs := d.validateDataInputs)
    (loss := d.loss (α := α)) initState

/-- Convenience instantiator that chooses only the execution mode. -/
def instantiate {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    [Runtime.Autograd.Torch.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (d : ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (cast : Float → α) (execution : Torch.ExecutionMode := .eager)
    (initialState? : Option (TorchLean.TensorPack α stateShapes) := none) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) := do
  instantiateWith (α := α) (stateShapes := stateShapes) (inputShapes := inputShapes)
    (dataInputShapes := dataInputShapes)
    d cast { execution := execution } initialState?

end ObjectiveDef

end Module

end Model
end Autograd
end Runtime
