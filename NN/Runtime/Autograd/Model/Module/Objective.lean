/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Module.RuntimeInit
public import NN.Runtime.Autograd.Model.Optim
public import NN.Runtime.Autograd.Torch.Core.Trainer

/-!
# Scalar Objectives

Executable scalar-objective definitions and runtime state. This module provides loss and gradient
evaluation, explicit optimizer steps, state access, and optimizers bound to a live objective.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra

namespace Module

/--
An immutable scalar-objective definition:
- `initState` stores initial trainable parameters and persistent buffers as `Float` tensors,
- `loss` is *polymorphic in the scalar backend* (same code works for Float/configured binary32/…).

You can instantiate this definition as an `Objective` under a chosen execution mode and scalar.
-/
structure ObjectiveDef (β : Type) [TorchLean.Storage β] (stateShapes inputShapes : List Shape)
    (dataInputShapes : List Shape := []) where
  /-- Initial parameter-and-buffer state, cast from `Float` at instantiation time. -/
  initState : TorchLean.TensorPack Float stateShapes
  /--
  Optional storage-first initialization plan for executable `Float` runs.

  The ordinary tensors remain the semantic initial values. This plan records how a runtime may
  materialize the same initialization directly in backend storage without traversing a large
  nested Lean tensor first.
  -/
  runtimeInit : Option (RuntimeInit.Plan stateShapes) := none
  /-- Differentiability flags aligned with `stateShapes`; persistent buffers carry `false`. -/
  requiresGrad : Array Bool := Array.replicate stateShapes.length true
  /-- Validate static model and objective configuration before runtime allocation. -/
  validate : Except String Unit := pure ()
  /-- Validate non-differentiable inputs before they reach the runtime program. -/
  validateDataInputs : TorchLean.TensorPack β dataInputShapes → Except String Unit :=
    fun _ => pure ()
  /--
  Scalar loss over differentiable tensors followed by non-differentiable data tensors.

  The second curried input pack can carry labels, bounded token IDs, or gather indices without
  converting them through the model's floating-point scalar type.
  -/
  loss :
    ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
      ∀ {m : Type → Type}, [Monad m] → [Torch.Ops (m := m) (α := α)] →
        Torch.CurriedRef (fun s => Torch.Ops.Ref (m := m) (α := α) s)
          (stateShapes ++ inputShapes)
          (Torch.CurriedRef (fun s => Torch.Ops.DataRef (m := m) (α := α) β s)
            dataInputShapes (m (Torch.Ops.Ref (m := m) (α := α) Shape.scalar)))

/--
Runtime state for a model together with a scalar objective.

This is lower level than PyTorch's loss classes: it owns model state as well as the objective. It
wraps `Torch.ScalarTrainer` and exposes objective evaluation, explicit gradients, and updates.
-/
structure Objective (α β : Type) [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    (stateShapes inputShapes : List Shape) (dataInputShapes : List Shape := []) where
  /-- Trainer that owns trainable parameters and persistent buffers. -/
  trainer : Torch.ScalarTrainer α β stateShapes inputShapes dataInputShapes
  /-- Runtime configuration used to instantiate the module. -/
  runtime : Torch.Config
  /-- Concrete host/device tensor conversion selected when the module was instantiated. -/
  tensorTransfer : Runtime.Autograd.Torch.TensorTransfer α


namespace Objective

/--
Create a runtime objective from an explicit scalar program and initial model state.

This is the low-level constructor; public training code starts from an `ObjectiveDef` and calls
`ObjectiveDef.instantiate`.
-/
def create {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    [Runtime.Autograd.Torch.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (runtime : Torch.Config := {})
    (requiresGrad : Array Bool := Array.replicate stateShapes.length true)
    (validateDataInputs : TorchLean.TensorPack β dataInputShapes → Except String Unit :=
      fun _ => pure ())
    (loss :
      ∀ {m : Type → Type}, [Monad m] → [Torch.Ops (m := m) (α := α)] →
        Torch.CurriedRef (fun s => Torch.Ops.Ref (m := m) (α := α) s) (stateShapes ++ inputShapes)
          (Torch.CurriedRef (fun s => Torch.Ops.DataRef (m := m) (α := α) β s)
            dataInputShapes (m (Torch.Ops.Ref (m := m) (α := α) Shape.scalar))))
    (initState : TorchLean.TensorPack α stateShapes) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) := do
  unless requiresGrad.size = stateShapes.length do
    throw <| IO.userError
      s!"objective: expected {stateShapes.length} requiresGrad flags, got {requiresGrad.size}"
  let mkTr :=
    Torch.scalarTrainer (α := α) (paramShapes := stateShapes) (inputShapes := inputShapes)
      (dataInputShapes := dataInputShapes)
      (options := runtime) (initRequiresGrad := requiresGrad)
      (validateDataInputs := validateDataInputs) (loss := loss)
  let tr ← Torch.Curried.uncurry (α := α) (ss := stateShapes)
    (β := IO (Torch.ScalarTrainer α β stateShapes inputShapes dataInputShapes)) mkTr initState
  pure { trainer := tr, runtime, tensorTransfer := inferInstance }

/-- Evaluate the scalar objective. -/
def loss {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (xs : TorchLean.TensorPack α inputShapes)
    (dataInputs : TorchLean.TensorPack β dataInputShapes) :
    IO (Tensor α .scalar) :=
  Torch.ScalarTrainer.runLoss (α := α) (paramShapes := stateShapes) (inputShapes := inputShapes)
    m.trainer xs dataInputs

/--
Return state-shaped gradients, using zero for entries that do not require gradients.

Set `value := true` to return `(gradient, objectiveValue)` from one forward tape.
-/
def grad {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (xs : TorchLean.TensorPack α inputShapes)
    (dataInputs : TorchLean.TensorPack β dataInputShapes) (value : Bool := false) :
    IO (match value with
      | false => TorchLean.TensorPack α stateShapes
      | true => TorchLean.TensorPack α stateShapes × Tensor α []) := by
  cases value with
  | false =>
      exact
        Torch.ScalarTrainer.runGrad (α := α) (paramShapes := stateShapes)
          (inputShapes := inputShapes) m.trainer xs dataInputs
  | true =>
      exact do
        let (objectiveValue, gradient) ←
          Torch.ScalarTrainer.runDiff
            (α := α) (paramShapes := stateShapes) (inputShapes := inputShapes)
            m.trainer xs dataInputs
        pure (gradient, objectiveValue)

/--
Start a fresh optimizer history from this module's current state.

This also releases backend-owned moments and clears the selected update path. Use the returned
state for subsequent updates; reinitialization permits switching between generic and native steps.
-/
def initOptimizer {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (opt : Runtime.Autograd.Model.Optim.Optimizer α stateShapes) :
    IO opt.State := do
  -- Generic optimizer states are initialized from host tensors. Synchronize device-backed
  -- parameters first so a CUDA module cannot seed those states from stale host mirrors.
  let _ ← m.trainer.getState
  let initialState ← opt.init m.trainer.state
  m.trainer.resetOptimizerState
  pure initialState

/--
Run one optimizer step using an explicit optimizer and state.

This mirrors a PyTorch training step:
1. compute the explicit state gradient (`ScalarTrainer.runGrad`)
2. update parameters via `opt.step` and return the new optimizer state

Set `loss := true` to return `(nextOptimizerState, lossValue)` from the same training tape.
On CUDA, mixing native steps and generic updates within one optimizer history is rejected.
Call `initOptimizer` to start a fresh history before switching paths.
-/
def step {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (opt : Runtime.Autograd.Model.Optim.Optimizer α stateShapes) (st : opt.State)
    (xs : TorchLean.TensorPack α inputShapes)
    (dataInputs : TorchLean.TensorPack β dataInputShapes) (loss : Bool := false) :
    IO (match loss with
      | false => opt.State
      | true => opt.State × Tensor α []) := by
  cases loss with
  | false =>
      exact do
        match ← opt.trainerStep? m.trainer st xs dataInputs with
        | some st' =>
            pure st'
        | none =>
            m.trainer.useOptimizerPath .generic
            let grads ← Torch.ScalarTrainer.runGrad (α := α)
              (paramShapes := stateShapes) (inputShapes := inputShapes)
              m.trainer xs dataInputs
            -- The generic optimizer below reads host tensors. Synchronize any device-resident
            -- parameter updates before it does so.
            let _ ← m.trainer.getState
            opt.step st m.trainer.state grads
  | true =>
      exact do
        match ← opt.trainerStepWithLoss? m.trainer st xs dataInputs with
        | some result =>
            pure (result.optimizerState, result.loss)
        | none =>
            m.trainer.useOptimizerPath .generic
            let (lossValue, grads) ← Torch.ScalarTrainer.runDiff (α := α)
              (paramShapes := stateShapes) (inputShapes := inputShapes)
              m.trainer xs dataInputs
            -- See the no-loss fallback above: generic updates operate on current host values.
            let _ ← m.trainer.getState
            let nextState ← opt.step st m.trainer.state grads
            pure (nextState, lossValue)

/-- Read the complete parameter-and-buffer state as a shape-indexed list. -/
def state {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes) :
    IO (TorchLean.TensorPack α stateShapes) :=
  m.trainer.getState

/-- Replace the complete parameter-and-buffer state. -/
def loadState {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (ps : TorchLean.TensorPack α stateShapes) : IO Unit :=
  Torch.ParamList.setValues (α := α) (ss := stateShapes) m.trainer.state ps

end Objective

/-- Mutable optimizer state bound to one executable module. -/
structure BoundOptimizer (α β : Type)
    [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    (stateShapes inputShapes dataInputShapes : List Shape) (State : Type) where
  module : Objective α β stateShapes inputShapes dataInputShapes
  state : IO.Ref State
  step :
    TorchLean.TensorPack α inputShapes →
    TorchLean.TensorPack β dataInputShapes →
    IO Unit

/-- Initialize an optimizer and bind its state and update operation to `module`. -/
def bindOptimizer {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes)
    (optimizer : Optim.Optimizer α stateShapes) :
    IO (BoundOptimizer α β stateShapes inputShapes dataInputShapes optimizer.State) := do
  let initialState ← Objective.initOptimizer module optimizer
  let state ← IO.mkRef initialState
  let step
      (inputs : TorchLean.TensorPack α inputShapes)
      (dataInputs : TorchLean.TensorPack β dataInputShapes) :
      IO Unit := do
    let currentState ← state.get
    let nextState ← Objective.step module optimizer currentState inputs dataInputs
    state.set nextState
  pure { module, state, step }

end Module

end Model
end Autograd
end Runtime
