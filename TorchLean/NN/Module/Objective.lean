/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Init
public import TorchLean.NN.Program
public import TorchLean.NN.Optimizer
public import TorchLean.NN.ScalarTrainer.Construction

/-!
# Scalar Objectives

Executable scalar-objective definitions and their runtime state. An `Objective` evaluates its loss
and gradients, takes optimizer steps from a caller-held optimizer state, and reads or replaces its
parameter-and-buffer state. A `BoundOptimizer` keeps one optimizer state with one objective.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

/--
An immutable scalar-objective definition:
- `initState` stores initial trainable parameters and persistent buffers as `Float` tensors,
- `loss` is *polymorphic in the scalar backend* (same code works for Float/configured binary32/…).

`ObjectiveDef.instantiate` builds an `Objective` from it for a chosen scalar and runtime.
-/
structure ObjectiveDef (β : Type) [Storage β] (stateShapes inputShapes : List Shape)
    (dataInputShapes : List Shape := []) where
  /-- Initial parameter-and-buffer state, cast from `Float` at instantiation time. -/
  initState : TensorPack Float stateShapes
  /-- Optional runtime initialization plan, one initializer per state shape.
  `ObjectiveDef.instantiate` validates the plan and does not apply it. -/
  runtimeInit : Option (Init.Plan stateShapes) := none
  /-- Differentiability flags aligned with `stateShapes`, with `false` for persistent buffers. -/
  requiresGrad : Array Bool := Array.replicate stateShapes.length true
  /-- Validate static model and objective configuration before runtime allocation. -/
  validate : Except String Unit := pure ()
  /-- Validate non-differentiable inputs before they reach the runtime program. -/
  validateDataInputs : TensorPack β dataInputShapes → Except String Unit :=
    fun _ => pure ()
  /--
  Scalar loss over differentiable tensors followed by non-differentiable data tensors.

  The second curried input pack can carry labels, bounded token IDs, or gather indices without
  converting them through the model's floating-point scalar type.
  -/
  loss :
    ∀ {α : Type}, [Storage α] → [Context α] →
      Program.Data α β (stateShapes ++ inputShapes) dataInputShapes Shape.scalar

/--
Runtime state for a model together with a scalar objective.

A PyTorch loss class holds only the loss computation, and an `Objective` also owns the model
state. It wraps `ScalarTrainer` and exposes objective evaluation, state-shaped gradients
and optimizer updates.
-/
structure Objective (α β : Type) [Storage α] [Storage β]
    [Context α]
    (stateShapes inputShapes : List Shape) (dataInputShapes : List Shape := []) where
  /-- Trainer that owns trainable parameters and persistent buffers. -/
  trainer : ScalarTrainer α β stateShapes inputShapes dataInputShapes
  /-- Runtime configuration used to instantiate the module. -/
  runtime : Session.Config
  /-- Concrete host/device tensor conversion selected when the module was instantiated. -/
  tensorTransfer : TensorTransfer α


namespace Objective

/--
Create a runtime objective from a scalar program and an initial model state.

It is the low-level constructor, and it fails unless `requiresGrad` has one flag per state shape.
Public training code starts from an `ObjectiveDef` and calls `ObjectiveDef.instantiate`.
-/
def create {α β : Type} [Storage α] [Storage β]
    [Context α]
    [TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (runtime : Session.Config := {})
    (requiresGrad : Array Bool := Array.replicate stateShapes.length true)
    (validateDataInputs : TensorPack β dataInputShapes → Except String Unit :=
      fun _ => pure ())
    (loss : Program.Data α β (stateShapes ++ inputShapes) dataInputShapes Shape.scalar)
    (initState : TensorPack α stateShapes) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) := do
  unless requiresGrad.size = stateShapes.length do
    throw <| IO.userError
      s!"objective: expected {stateShapes.length} requiresGrad flags, got {requiresGrad.size}"
  let mkTr :=
    ScalarTrainer.create (α := α) (paramShapes := stateShapes) (inputShapes := inputShapes)
      (dataInputShapes := dataInputShapes)
      (options := runtime) (initRequiresGrad := requiresGrad)
      (validateDataInputs := validateDataInputs) (loss := loss)
  let tr ← CurriedRef.uncurry (Ref := (Tensor α ·)) (ss := stateShapes)
    (β := IO (ScalarTrainer α β stateShapes inputShapes dataInputShapes)) mkTr initState
  pure { trainer := tr, runtime, tensorTransfer := inferInstance }

/-- Evaluate the scalar objective. -/
def loss {α β : Type} [Storage α] [Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (xs : TensorPack α inputShapes)
    (dataInputs : TensorPack β dataInputShapes) :
    IO (Tensor α .scalar) :=
  ScalarTrainer.loss (α := α) (paramShapes := stateShapes) (inputShapes := inputShapes)
    m.trainer xs dataInputs

/--
Return state-shaped gradients, using zero for entries that do not require gradients.

Set `value := true` to return `(gradient, objectiveValue)` from one forward tape.
-/
def grad {α β : Type} [Storage α] [Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (xs : TensorPack α inputShapes)
    (dataInputs : TensorPack β dataInputShapes) (value : Bool := false) :
    IO (match value with
      | false => TensorPack α stateShapes
      | true => TensorPack α stateShapes × Tensor α []) :=
  ScalarTrainer.grad m.trainer xs dataInputs (value := value)

/--
Start a fresh optimizer history from this module's current state.

The call also releases backend-owned moments and clears the selected update path. Use the
returned state for subsequent updates, and reinitialize to switch between generic and native
steps.
-/
def initOptimizer {α β : Type} [Storage α] [Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (opt : Optimizer α stateShapes) :
    IO opt.State := do
  -- Generic optimizer states are initialized from host tensors. Synchronize device-backed
  -- parameters first so a CUDA module cannot seed those states from stale host mirrors.
  let _ ← m.trainer.getState
  let initialState ← opt.init m.trainer.state
  m.trainer.resetOptimizerState
  pure initialState

/--
Run one optimizer step with the optimizer `opt` and its state `st`.

When `opt.nativeBatchStep?` runs a native update on the one-sample batch, the step returns `st`
unchanged, since the moments live on the device. Otherwise it follows a PyTorch training step:
1. compute the state gradient (`ScalarTrainer.grad`)
2. update parameters via `opt.step` and return the new optimizer state

Set `loss := true` to return `(nextOptimizerState, lossValue)` from the same training tape. With
that flag, a native update that reports no loss throws. On CUDA, mixing native steps and generic
updates within one optimizer history is rejected. Call `initOptimizer` to start a fresh history
before switching paths.
-/
def step {α β : Type} [Storage α] [Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (opt : Optimizer α stateShapes) (st : opt.State)
    (xs : TensorPack α inputShapes)
    (dataInputs : TensorPack β dataInputShapes) (loss : Bool := false) :
    IO (match loss with
      | false => opt.State
      | true => opt.State × Tensor α []) := by
  cases loss with
  | false =>
      exact do
        match ← opt.nativeBatchStep? m.trainer st #[(xs, dataInputs)] false with
        | some _ =>
            pure st
        | none =>
            m.trainer.useOptimizerPath .generic
            let grads ← ScalarTrainer.grad (α := α)
              (paramShapes := stateShapes) (inputShapes := inputShapes)
              m.trainer xs dataInputs
            -- The generic optimizer below reads host tensors. Synchronize any device-resident
            -- parameter updates before it does so.
            let _ ← m.trainer.getState
            opt.step st m.trainer.state grads
  | true =>
      exact do
        match ← opt.nativeBatchStep? m.trainer st #[(xs, dataInputs)] true with
        | some (some lossValue) =>
            pure (st, lossValue)
        | some none =>
            throw <| IO.userError "torch: the native step returned no loss"
        | none =>
            m.trainer.useOptimizerPath .generic
            let (grads, lossValue) ← ScalarTrainer.grad (α := α)
              (paramShapes := stateShapes) (inputShapes := inputShapes)
              m.trainer xs dataInputs (value := true)
            -- See the no-loss fallback above: generic updates operate on current host values.
            let _ ← m.trainer.getState
            let nextState ← opt.step st m.trainer.state grads
            pure (nextState, lossValue)

/-- Read the complete parameter-and-buffer state as a shape-indexed tensor pack. -/
def state {α β : Type} [Storage α] [Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes) :
    IO (TensorPack α stateShapes) :=
  m.trainer.getState

/-- Replace the complete parameter-and-buffer state. -/
def loadState {α β : Type} [Storage α] [Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (ps : TensorPack α stateShapes) : IO Unit :=
  ParamList.setValues (α := α) (ss := stateShapes) m.trainer.state ps

end Objective

/-- Mutable optimizer state bound to one executable module. -/
structure BoundOptimizer (α β : Type)
    [Storage α] [Storage β] [Context α]
    (stateShapes inputShapes dataInputShapes : List Shape) (State : Type) where
  /-- The objective whose parameters the optimizer updates. -/
  module : Objective α β stateShapes inputShapes dataInputShapes
  /-- The current optimizer state. -/
  state : IO.Ref State
  /-- Run one optimizer step of `module` on the inputs and data inputs and store the result in
  `state`. -/
  step :
    TensorPack α inputShapes →
    TensorPack β dataInputShapes →
    IO Unit

/-- Initialize an optimizer and bind its state and update operation to `module`. -/
def Objective.bindOptimizer {α β : Type}
    [Storage α] [Storage β] [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (module : Objective α β stateShapes inputShapes dataInputShapes)
    (optimizer : Optimizer α stateShapes) :
    IO (BoundOptimizer α β stateShapes inputShapes dataInputShapes optimizer.State) := do
  let initialState ← Objective.initOptimizer module optimizer
  let state ← IO.mkRef initialState
  let step
      (inputs : TensorPack α inputShapes)
      (dataInputs : TensorPack β dataInputShapes) :
      IO Unit := do
    let currentState ← state.get
    let nextState ← Objective.step module optimizer currentState inputs dataInputs
    state.set nextState
  pure { module, state, step }

end TorchLean
