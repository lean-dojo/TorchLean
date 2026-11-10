/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- Downstream modules reach the marked imports through this module. Import shaking cannot see
-- those downstream uses, so keep the marked imports.
module -- shake: keep-downstream

public import TorchLean.NN.Session.Selection -- shake: keep
public import TorchLean.Tensor.Pack -- shake: keep
public import TorchLean.NN.Module.Evaluator -- shake: keep
public import TorchLean.NN.Module.Instantiation -- shake: keep

@[expose] public section

namespace TorchLean
namespace Module

/-!
# Module Execution

`ObjectiveDef` describes a scalar objective with its model state and
input shapes, and `instantiate` turns it into a mutable `Objective` at a chosen scalar type and
`Session.Config`. Its field `runtime` evaluates the loss, returns state gradients, takes optimizer
steps and reads or replaces its state. The shape lists stay in both types, so construction and
execution use the same state ordering. Command-line selection lives in
`TorchLean.NN.Module.Command`.
-/

/--
Executable state for a model and scalar objective. The operations `loss`, `grad`, `step` and
`state` of `TorchLean.Objective` run on its field `runtime`.
-/
structure Objective (α β : Type) [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    (stateShapes inputShapes : List Shape) (dataInputShapes : List Shape := []) where
  /-- The runtime objective, with its trainer and device machinery. -/
  runtime : TorchLean.Objective α β stateShapes inputShapes dataInputShapes

namespace Objective

/-- Try one backend-native mean-gradient update over a nonempty batch. -/
def tryNativeBatchStep {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes)
    (optimizer : Optimizer α stateShapes)
    (optimizerState : optimizer.State)
    (batch : Array (TensorPack α inputShapes × TensorPack β dataInputShapes))
    (readLoss : Bool) : IO (Option (optimizer.State × Option (Tensor α []))) := do
  let stepped ← optimizer.nativeBatchStep? (Objective.runtime module).trainer optimizerState
    batch readLoss
  pure (stepped.map (optimizerState, ·))

/--
Apply a generic optimizer update from an already-computed, shape-aligned state gradient.

On CUDA, native steps keep their moments on the device, apart from the `optimizerState` passed
here. The two paths therefore cannot share an optimizer history, and switching between them needs
a fresh `initOptimizer`.
-/
def update {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes)
    (optimizer : Optimizer α stateShapes)
    (optimizerState : optimizer.State)
    (gradients : TensorPack α stateShapes) :
    IO optimizer.State := do
  let runtimeObjective := runtime module
  runtimeObjective.trainer.useOptimizerPath .generic
  -- Explicit optimizers read host tensors, including parameters initialized in CUDA buffers.
  let _ ← runtimeObjective.trainer.getState
  optimizer.step optimizerState runtimeObjective.trainer.state
    gradients

end Objective

/-- Instantiate an objective with a caller-supplied `cast : Float → α` for scalar literals. -/
def instantiateWithCast
    {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    [TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (definition :
      ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (cast : Float → α) (runtime : Session.Config := {})
    (initialState? : Option (TensorPack α stateShapes) := none) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) := do
  let objective ←
    ObjectiveDef.instantiate
      (α := α) (β := β) (stateShapes := stateShapes) (inputShapes := inputShapes)
      (dataInputShapes := dataInputShapes) definition cast runtime
      initialState?
  pure ⟨objective⟩

/--
Instantiate an executable objective, optionally starting from exact typed state.

`initialState?` bypasses the model's stored Float initializers. Without it, initialization uses the
runtime arithmetic's standard `Float` conversion.

Custom losses and multi-input programs use this low-level constructor, while the higher-level
`nn.Module` and `Trainer` APIs suit ordinary sequential models better.
-/
def instantiate
    {β : Type} [TorchLean.Storage β]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (definition :
      ObjectiveDef β stateShapes inputShapes dataInputShapes)
    (runtime : Session.Config := {})
    (α : Type := Float32)
    [TorchLean.Storage α]
    [Context α] [FromFloat α]
    [TensorTransfer α]
    (initialState? : Option (TensorPack α stateShapes) := none)
    : IO (Objective α β stateShapes inputShapes dataInputShapes) :=
  instantiateWithCast
    (α := α) (β := β)
    (stateShapes := stateShapes) (inputShapes := inputShapes)
    (dataInputShapes := dataInputShapes)
    definition (ofFloat (α := α)) runtime initialState?

end Module
end TorchLean
