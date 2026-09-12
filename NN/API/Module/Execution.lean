/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.API.Arguments -- shake: keep
public import NN.API.Runtime -- shake: keep
public import NN.API.Neural.State -- shake: keep
public import NN.API.Arithmetic -- shake: keep
public import NN.Runtime.Autograd.Model.Module.Evaluator -- shake: keep
public import NN.Runtime.Autograd.Model.Module.Instantiation -- shake: keep
public import NN.Runtime.Autograd.Model.Module -- shake: keep

@[expose] public section

namespace TorchLean
namespace Module

/-!
# Module Execution

This file connects typed module definitions to executable scalar modules. It provides state
initialization and explicit execution settings. Command-line selection lives in
`NN.API.Module.Command`.

`ObjectiveDefinition` describes a scalar objective together with model state and input shapes.
Instantiating it produces a mutable `Objective` that can evaluate the objective, return explicit
state gradients, and update trainable entries. The shape lists remain part of both types, so
construction and execution use the same state ordering.
-/

/-- An immutable scalar objective together with its typed initial state. -/
abbrev ObjectiveDefinition (β : Type) [TorchLean.Storage β]
    (stateShapes inputShapes : List Spec.Shape)
    (dataInputShapes : List Spec.Shape := []) :=
  Runtime.Autograd.Model.Module.ObjectiveDef β stateShapes inputShapes dataInputShapes

/--
Executable state for a model and scalar objective.

The runtime trainer and device machinery are intentionally hidden. Public code interacts with an
objective through operations such as `loss`, `grad`, `step`, and `state`.
-/
structure Objective (α β : Type) [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    (stateShapes inputShapes : List Spec.Shape) (dataInputShapes : List Spec.Shape := []) where
  private mk ::
  private runtimeObjective :
    Runtime.Autograd.Model.Module.Objective α β stateShapes inputShapes dataInputShapes

/-- A reusable evaluator over an existing typed model state. -/
abbrev Evaluator (α β : Type) [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    (stateShapes inputShapes dataInputShapes : List Spec.Shape)
    (output : Spec.Shape) :=
  Runtime.Autograd.Model.Module.Evaluator α β stateShapes inputShapes dataInputShapes
    output

namespace RuntimeInit
export Runtime.Autograd.Model.Module.RuntimeInit
  (FloatInit Plan xavierUniformForShape kaimingUniformForShape)
end RuntimeInit

namespace Objective

namespace Internal

/-- Wrap a runtime objective behind the public execution boundary. -/
opaque fromRuntime {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (objective :
      Runtime.Autograd.Model.Module.Objective α β stateShapes inputShapes dataInputShapes) :
    Objective α β stateShapes inputShapes dataInputShapes :=
  ⟨objective⟩

/-- Reveal the runtime objective only to implementation code. -/
opaque runtime {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (objective : Objective α β stateShapes inputShapes dataInputShapes) :
    Runtime.Autograd.Model.Module.Objective α β stateShapes inputShapes dataInputShapes :=
  match objective with
  | ⟨runtimeObjective⟩ => runtimeObjective

end Internal

/-- Evaluate the scalar objective. -/
def loss {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes)
    (inputs : Arguments α inputShapes)
    (dataInputs : Arguments β dataInputShapes) :
    IO (Tensor α []) :=
  Runtime.Autograd.Model.Module.Objective.loss (Internal.runtime module)
    (Arguments.Internal.toTensorPack inputs)
    (Arguments.Internal.toTensorPack dataInputs)

/--
Differentiate the objective with respect to its state.

Entries that do not require gradients are zero. Set `value := true` to return
`(gradient, objectiveValue)` from one forward tape.
-/
def grad {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes)
    (inputs : Arguments α inputShapes)
    (dataInputs : Arguments β dataInputShapes) (value : Bool := false) :
    IO (match value with
      | false => nn.State α stateShapes
      | true => nn.State α stateShapes × Tensor α []) := by
  cases value with
  | false =>
      exact do
        let gradient ←
          Runtime.Autograd.Model.Module.Objective.grad (Internal.runtime module)
            (Arguments.Internal.toTensorPack inputs)
            (Arguments.Internal.toTensorPack dataInputs)
        pure (nn.State.Internal.fromTensorPack gradient)
  | true =>
      exact do
        let (gradient, objectiveValue) ←
          Runtime.Autograd.Model.Module.Objective.grad (Internal.runtime module)
            (Arguments.Internal.toTensorPack inputs)
            (Arguments.Internal.toTensorPack dataInputs)
            (value := true)
        pure (nn.State.Internal.fromTensorPack gradient, objectiveValue)

/--
Start a fresh optimizer history from the module's current parameters and buffers.

This clears backend-owned moments and permits switching between native and generic updates.
Use the returned optimizer state for subsequent steps.
-/
def initOptimizer {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes)
    (optimizer : Runtime.Autograd.Model.Optim.Optimizer α stateShapes) :
    IO optimizer.State :=
  Runtime.Autograd.Model.Module.Objective.initOptimizer (Internal.runtime module) optimizer

/--
Compute one gradient update.

By default this returns the next optimizer state. Set `loss := true` to return
`(nextOptimizerState, lossValue)` from the same training tape.
-/
def step {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes)
    (optimizer : Runtime.Autograd.Model.Optim.Optimizer α stateShapes)
    (optimizerState : optimizer.State)
    (inputs : Arguments α inputShapes)
    (dataInputs : Arguments β dataInputShapes) (loss : Bool := false) :
    IO (match loss with
      | false => optimizer.State
      | true => optimizer.State × Tensor α []) := by
  cases loss with
  | false =>
      exact
        Runtime.Autograd.Model.Module.Objective.step
          (Internal.runtime module) optimizer optimizerState
          (Arguments.Internal.toTensorPack inputs)
          (Arguments.Internal.toTensorPack dataInputs)
  | true =>
      exact
        Runtime.Autograd.Model.Module.Objective.step
          (Internal.runtime module) optimizer optimizerState
          (Arguments.Internal.toTensorPack inputs)
          (Arguments.Internal.toTensorPack dataInputs)
          (loss := true)

/-- Try one backend-native mean-gradient update over a nonempty batch. -/
def Internal.tryNativeBatchStep {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes)
    (optimizer : Runtime.Autograd.Model.Optim.Optimizer α stateShapes)
    (optimizerState : optimizer.State)
    (batch : Array (Arguments α inputShapes × Arguments β dataInputShapes))
    (readLoss : Bool) : IO (Option (optimizer.State × Option (Tensor α []))) :=
  optimizer.trainerBatchStep? (Internal.runtime module).trainer optimizerState
    (batch.map fun (inputs, dataInputs) =>
      (Arguments.Internal.toTensorPack inputs, Arguments.Internal.toTensorPack dataInputs))
    readLoss

/--
Apply a generic optimizer update from an already-computed, shape-aligned state gradient.

On CUDA, this cannot share an optimizer history with native steps: their device moments are
separate from the explicit optimizer state. Call `initOptimizer` before switching paths.
-/
def update {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes)
    (optimizer : Runtime.Autograd.Model.Optim.Optimizer α stateShapes)
    (optimizerState : optimizer.State)
    (gradients : nn.State α stateShapes) :
    IO optimizer.State := do
  let runtimeObjective := Internal.runtime module
  runtimeObjective.trainer.useOptimizerPath .generic
  -- Explicit optimizers read host tensors, including parameters initialized on the device.
  let _ ← runtimeObjective.trainer.getState
  optimizer.step optimizerState runtimeObjective.trainer.state
    (nn.State.Internal.toTensorPack gradients)

/-- Read the complete parameter-and-buffer state. -/
def state {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes) :
    IO (nn.State α stateShapes) := do
  let state ← Runtime.Autograd.Model.Module.Objective.state (Internal.runtime module)
  pure (nn.State.Internal.fromTensorPack state)

/-- Replace the complete parameter-and-buffer state. -/
def setState {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes)
    (state : nn.State α stateShapes) : IO Unit :=
  Runtime.Autograd.Model.Module.Objective.loadState (Internal.runtime module)
    (nn.State.Internal.toTensorPack state)

end Objective

namespace Internal

/-- Instantiate an objective with an explicit scalar-literal conversion. -/
def instantiate
    {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    [Runtime.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (definition : ObjectiveDefinition β stateShapes inputShapes dataInputShapes)
    (cast : Float → α) (runtime : Runtime.Config := {}) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) := do
  let objective ←
    Runtime.Autograd.Model.Module.ObjectiveDef.instantiateWith
      (α := α) (β := β) (stateShapes := stateShapes) (inputShapes := inputShapes)
      (dataInputShapes := dataInputShapes) definition cast runtime
  pure (Objective.Internal.fromRuntime objective)

end Internal

/--
Instantiate an executable objective using the runtime arithmetic's standard `Float` conversion.

This is the low-level constructor for custom losses and multi-input programs. The higher-level
`nn.Module` and `Trainer` APIs should be preferred for ordinary sequential models.
-/
def instantiate
    {β : Type} [TorchLean.Storage β]
    {stateShapes inputShapes dataInputShapes : List Spec.Shape}
    (definition : ObjectiveDefinition β stateShapes inputShapes dataInputShapes)
    (runtime : Runtime.Config := {})
    (α : Type := Float32)
    [TorchLean.Storage α]
    [Context α] [Runtime.FromFloat α]
    [Runtime.TensorTransfer α]
    : IO (Objective α β stateShapes inputShapes dataInputShapes) :=
  Internal.instantiate
    (α := α) (β := β)
    (stateShapes := stateShapes) (inputShapes := inputShapes)
    (dataInputShapes := dataInputShapes)
    definition (Runtime.ofFloat (α := α)) runtime

end Module
end TorchLean
