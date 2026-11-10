/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Module.Execution
public import TorchLean.NN.Builder.Indexed
public import TorchLean.NN.Sample
public import TorchLean.NN.Optimizer.Constructor -- shake: keep

/-!
# Executable Modules

`nn.Module` is the runtime counterpart of an `nn.Sequential` model. It owns one live set of
parameters and buffers and a train/eval flag, as a PyTorch `nn.Module` does, while the model
definition stays an immutable value for proofs and graph lowering. `nn.IndexedModule` does the same
for an `nn.IndexedModel`, whose input is a non-differentiable tensor such as a batch of token ids.
Both start in training mode.

The main operations are `instantiate`, `forward`, `train`, `eval`, `state` and `setState` on both
module types. For hand-written training loops over a `TorchLean.Module.Objective`,
`Objective.dataStep`, `Objective.dataLossEvaluator` and `Objective.indexedPredictor` return plain
tensor functions that run one update, one loss evaluation or one prediction.
-/

@[expose] public section

namespace TorchLean

open TorchLean (TensorTransfer)

namespace Module

/--
The live state behind `nn.Module` and `nn.IndexedModule`: parameter and buffer storage, the
runtime configuration, the train/eval mode and the counter of random operations.

The shape list stays in the type, so a replacement state must list the same tensor shapes in the
same order. All mutation happens in this object, and model and layer definitions stay immutable
values that can be lowered and reasoned about.
-/
structure RuntimeState (α : Type) [TorchLean.Storage α] [Context α]
    (stateShapes : List Shape) where
  /-- Parameter and buffer storage. -/
  stateRef : ParamList α stateShapes
  /-- The runtime configuration. -/
  runtime : Session.Config
  /-- The train/eval mode cell. -/
  modeRef : IO.Ref nn.Mode
  /--
  The counter of random operations, which keeps random draws in order across successive forwards
  of the same module.
  -/
  rngCounter : IO.Ref Nat

namespace RuntimeState

/--
Allocate runtime state. `initialState?` is used as given when present. Otherwise the runtime
initialization plan `runtimeInit` fills the storage, or, without a plan, `cast` converts `initial`
to `α`. A fresh state starts in training mode with its random counter at zero.
-/
def instantiate {α : Type} [TorchLean.Storage α] [Context α]
    [tensorTransfer : TensorTransfer α]
    {stateShapes : List Shape} (initial : TorchLean.TensorPack Float stateShapes)
    (runtimeInit : Option
      (Init.Plan stateShapes))
    (requiresGrad : Array Bool) (runtime : Session.Config) (cast : Float → α)
    (initialState? : Option (TorchLean.TensorPack α stateShapes) := none) :
    IO (RuntimeState α stateShapes) := do
  let stateRef ←
    match initialState? with
    | some values =>
        ParamList.ofPackWithRequiresGrad values requiresGrad
    | none =>
        match runtimeInit with
        | some plan => do
            let empty := TensorPack.fill (cast 0.0) (ss := stateShapes)
            let stateRef ← ParamList.ofPackWithRequiresGrad
              empty requiresGrad
            Init.applyPlan
              (α := α) cast runtime stateRef plan
            pure stateRef
        | none => do
            let values := Pack.map (fun tensor => Tensor.map cast tensor) initial
            ParamList.ofPackWithRequiresGrad values requiresGrad
  let modeRef ← IO.mkRef nn.Mode.train
  let rngCounter ← IO.mkRef 0
  pure ⟨stateRef, runtime, modeRef, rngCounter⟩

/-- Read the current behavior of training-sensitive layers. -/
def mode {α : Type} [TorchLean.Storage α] [Context α] {stateShapes : List Shape}
    (state : RuntimeState α stateShapes) : IO nn.Mode :=
  (modeRef state).get

/-- Enable training behavior. -/
def train {α : Type} [TorchLean.Storage α] [Context α] {stateShapes : List Shape}
    (state : RuntimeState α stateShapes) : IO Unit :=
  (modeRef state).set .train

/-- Enable evaluation behavior. -/
def eval {α : Type} [TorchLean.Storage α] [Context α] {stateShapes : List Shape}
    (state : RuntimeState α stateShapes) : IO Unit :=
  (modeRef state).set .eval

/-- Whether the state is in training mode. -/
def isTraining {α : Type} [TorchLean.Storage α] [Context α]
    {stateShapes : List Shape} (state : RuntimeState α stateShapes) : IO Bool := do
  pure ((← state.mode) == .train)

/-- Read all parameters and buffers, synchronizing device storage when necessary. -/
def state {α : Type} [TorchLean.Storage α] [Context α]
    [TensorTransfer α]
    {stateShapes : List Shape} (runtimeState : RuntimeState α stateShapes) :
    IO (TensorPack α stateShapes) := do
  let tensors ← ParamList.valuesSynced (stateRef runtimeState)
  pure tensors

/-- Replace all parameters and buffers with a shape-compatible state. -/
def setState {α : Type} [TorchLean.Storage α] [Context α] {stateShapes : List Shape}
    (runtimeState : RuntimeState α stateShapes)
    (values : TensorPack α stateShapes) : IO Unit :=
  ParamList.setValues (stateRef runtimeState)
    values

end RuntimeState

end Module

namespace nn

/--
An `nn.Sequential` model with live, mutable runtime state.

`nn.Sequential` stays the immutable, shape-checked model definition used by proofs and graph
lowering. An `nn.Module` is its runtime counterpart, in the role of a PyTorch `nn.Module`: it owns
one parameter set and a mutable train/eval flag. Executable code gets the familiar module
lifecycle, and the mathematical model stays free of mutation.

```lean
-- Instantiate in binary64, switch to evaluation mode, and predict once.
def predict (model : nn.Sequential [2] [1]) (input : Tensor Float [2]) :
    IO (Tensor Float [1]) := do
  let module ← nn.Module.instantiate model (α := Float)
  module.eval
  module.forward input
```
-/
structure Module (α : Type) [TorchLean.Storage α] [Context α]
    {σ τ : Shape} (model : nn.Sequential σ τ) where
  /-- The live runtime state of the module. -/
  runtimeState : TorchLean.Module.RuntimeState α (nn.Sequential.stateShapes model)

/--
An `nn.IndexedModel` with live, mutable runtime state. The input is a non-differentiable tensor
over `β`, such as token ids, and the parameters and outputs are in `α`.
-/
structure IndexedModule (α β : Type) [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {σ τ : Shape} (model : nn.IndexedModel σ τ β) where
  /-- The live runtime state of the module. -/
  runtimeState : TorchLean.Module.RuntimeState α model.stateShapes

namespace Module

/--
Instantiate a model with mutable parameter and buffer storage, after checking it with
`nn.Sequential.validate`. `runtime` holds the device, seed and other run settings, and the element
type `α` defaults to native binary32. Choose another supported type with
`nn.Module.instantiate model (α := Float)`.

`initialState?` supplies state already represented in `α` and skips the model's default
initialization, which starts from `Float` values.
-/
def instantiate {σ τ : Shape}
    (model : nn.Sequential σ τ)
    (runtime : Session.Config := {})
    (α : Type := Float32)
    [TorchLean.Storage α] [Context α] [FromFloat α]
    [tensorTransfer : TensorTransfer α]
    (initialState? : Option (TensorPack α (nn.Sequential.stateShapes model)) := none) :
    IO (Module α model) := do
  IO.ofExcept (nn.Sequential.validate model)
  let state ← Module.RuntimeState.instantiate
    (nn.initialState model)
    (nn.Sequential.runtimeInit? model) (nn.Sequential.requiresGrad model)
    runtime (ofFloat (α := α))
    (initialState? := initialState?)
  pure ⟨state⟩

/-- Read the mode, training or evaluation, that the module's training-sensitive layers use. -/
def mode {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ} (module : Module α model) :
    IO nn.Mode :=
  (runtimeState module).mode

/--
Switch to training mode, as PyTorch's `module.train()`: dropout is active, and batch normalization
uses batch statistics and updates its running buffers.
-/
def train {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ} (module : Module α model) : IO Unit :=
  (runtimeState module).train

/--
Switch to evaluation mode, as PyTorch's `module.eval()`: dropout is off and batch normalization
uses its running statistics, so forwards are deterministic.
-/
def eval {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ} (module : Module α model) : IO Unit :=
  (runtimeState module).eval

/-- Whether the module is in training mode (PyTorch: `module.training`). -/
def isTraining {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ} (module : Module α model) : IO Bool := do
  (runtimeState module).isTraining

/--
Read all parameters and buffers, synchronizing device storage when necessary. The PyTorch
analogue is `module.state_dict()`.
-/
def state {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    [TensorTransfer α]
    {model : nn.Sequential σ τ} (module : Module α model) :
    IO (TensorPack α (nn.Sequential.stateShapes model)) :=
  (runtimeState module).state

/--
Replace all parameters and buffers with `state`, whose type fixes their shapes. The PyTorch
analogue is `module.load_state_dict(state)`.
-/
def setState {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ} (module : Module α model)
    (state : TensorPack α (nn.Sequential.stateShapes model)) : IO Unit :=
  (runtimeState module).setState state

/--
Evaluate one input without recording a backward tape, as calling a PyTorch module under
`torch.no_grad()` does. The module's mode controls training-sensitive layers unless `mode`
overrides it for this call, which leaves the stored mode unchanged.

A forward in training mode updates running buffers and advances the module's random stream.
Differentiable model programs use `nn.Sequential.forward`.
-/
def forward {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    [tensorTransfer : TensorTransfer α]
    {model : nn.Sequential σ τ} (module : Module α model)
    (input : Tensor α σ) (mode : Option nn.Mode := none) : IO (Tensor α τ) := do
  let selectedMode ← match mode with
    | some value => pure value
    | none => module.mode
  let state := runtimeState module
  Sequential.forwardNoGrad
    (α := α) (tensorTransfer := tensorTransfer)
    (Module.RuntimeState.runtime state) model
    (Module.RuntimeState.stateRef state) input
    (mode := selectedMode)
    (rngCounter := some (Module.RuntimeState.rngCounter state))


end Module

namespace IndexedModule

/--
Instantiate an indexed model with mutable parameter and buffer storage, after checking it with
`nn.IndexedModel.validate`. The element type `α` defaults to native binary32.

`initialState?` supplies state already represented in `α` and skips the model's default
initialization, which starts from `Float` values.
-/
def instantiate {σ τ : Shape} {β : Type} [TorchLean.Storage β]
    (model : nn.IndexedModel σ τ β)
    (runtime : Session.Config := {})
    (α : Type := Float32)
    [TorchLean.Storage α] [Context α] [FromFloat α]
    [tensorTransfer : TensorTransfer α]
    (initialState? : Option (TensorPack α model.stateShapes) := none) :
    IO (IndexedModule α β model) := do
  IO.ofExcept model.validate
  let state ← Module.RuntimeState.instantiate
    model.initialState
    (nn.IndexedModel.initializationPlan model)
    model.requiresGrad
    runtime (ofFloat (α := α))
    (initialState? := initialState?)
  pure ⟨state⟩

/-- Read the module's mode, which training-sensitive layers such as dropout follow. -/
def mode {σ τ : Shape} {α β : Type} [TorchLean.Storage α]
    [TorchLean.Storage β] [Context α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model) :
    IO nn.Mode :=
  (runtimeState module).mode

/-- Switch to training mode (PyTorch: `module.train()`). -/
def train {σ τ : Shape} {α β : Type} [TorchLean.Storage α]
    [TorchLean.Storage β] [Context α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model) : IO Unit :=
  (runtimeState module).train

/-- Switch to evaluation mode (PyTorch: `module.eval()`). -/
def eval {σ τ : Shape} {α β : Type} [TorchLean.Storage α]
    [TorchLean.Storage β] [Context α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model) : IO Unit :=
  (runtimeState module).eval

/-- Whether the module is in training mode (PyTorch: `module.training`). -/
def isTraining {σ τ : Shape} {α β : Type} [TorchLean.Storage α]
    [TorchLean.Storage β] [Context α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model) : IO Bool :=
  (runtimeState module).isTraining

/--
Read all parameters and buffers, synchronizing device storage when necessary. The PyTorch
analogue is `module.state_dict()`.
-/
def state {σ τ : Shape} {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    [TensorTransfer α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model) :
    IO (TensorPack α model.stateShapes) :=
  (runtimeState module).state

/--
Replace all parameters and buffers with `state`, whose type fixes their shapes. The PyTorch
analogue is `module.load_state_dict(state)`.
-/
def setState {σ τ : Shape} {α β : Type} [TorchLean.Storage α]
    [TorchLean.Storage β] [Context α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model)
    (state : TensorPack α model.stateShapes) : IO Unit :=
  (runtimeState module).setState state

/--
Evaluate one index tensor without recording a backward tape. Fails when the input does not pass
the model's own input check. `mode` overrides the module's mode for this call and leaves the
stored mode unchanged.
-/
def forward {σ τ : Shape} {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    [tensorTransfer : TensorTransfer α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model)
    (input : Tensor β σ) (mode : Option nn.Mode := none) : IO (Tensor α τ) := do
  let selectedMode ← match mode with
    | some value => pure value
    | none => module.mode
  IO.ofExcept (nn.IndexedModel.validateInput model input)
  let state := runtimeState module
  let program : Program.Data α β
      (model.stateShapes ++ []) [σ] τ :=
    fun {m} _ _ => by
      simpa using
        (nn.IndexedModel.program model selectedMode (α := α) (m := m))
  let evaluator ← Evaluator.create
    (program := program)
    (Module.RuntimeState.runtime state)
    (Module.RuntimeState.stateRef state)
    (rngCounter := some (Module.RuntimeState.rngCounter state))
  Evaluator.run
    evaluator Pack.nil (Pack.singleton input)


end IndexedModule

end nn

namespace Module

/-- Bind an optimizer to the hidden runtime objective. -/
def bindOptimizer {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (objective : Objective α β stateShapes inputShapes dataInputShapes)
    (optimizer : Optimizer α stateShapes) :=
  Objective.bindOptimizer
    (Objective.runtime objective) optimizer

/--
Build a loss function for the data-only objective `definition` over the current parameters of
`objective`, for example a validation loss in evaluation mode while `objective` trains. The
returned function takes the two data tensors and returns the scalar loss. It reuses one eager
session, and callers never handle tensor packs.
-/
def Objective.dataLossEvaluator {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    [TensorTransfer α]
    {stateShapes : List Shape} {firstShape secondShape : Shape}
    (objective : Objective α β stateShapes [] [firstShape, secondShape])
    (definition :
      ObjectiveDef β stateShapes [] [firstShape, secondShape]) :
    IO (Tensor β firstShape → Tensor β secondShape → IO (Tensor α [])) := do
  let runtimeObjective := Objective.runtime objective
  let evaluator ←
    ObjectiveDef.evaluator
      definition runtimeObjective.runtime runtimeObjective.trainer.state
  pure fun first second =>
    Evaluator.run
      evaluator Pack.nil (Pack.pair first second)

/--
Build an evaluation-mode prediction function for `model` over the current parameters of
`objective`, for example to sample from a language model after training an indexed loss. Each
call checks its index tensor with the model's own input check. Fails when the state shapes of
`model` differ from those of `objective`.
-/
def Objective.indexedPredictor {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    [TensorTransfer α]
    {stateShapes : List Shape} {σ τ : Shape}
    (objective : Objective α β stateShapes [] [σ, σ])
    (model : nn.IndexedModel σ τ β) :
    IO (Tensor β σ → IO (Tensor α τ)) := do
  if hState : model.stateShapes = stateShapes then
    let runtimeObjective := Objective.runtime objective
    let program : Program.Data α β
        (stateShapes ++ []) [σ] τ := by
      subst stateShapes
      exact fun {m} _ _ => by
        simpa only [List.append_nil] using
          (nn.IndexedModel.program model .eval (α := α) (m := m))
    let evaluator ← Evaluator.create
      (program := program) runtimeObjective.runtime runtimeObjective.trainer.state
      (validateDataInputs := fun
        | .cons input .nil => nn.IndexedModel.validateInput model input)
    pure fun input =>
      Evaluator.run
        evaluator Pack.nil (Pack.singleton input)
  else
    throw <| IO.userError
      "indexed predictor model does not match the trained objective state"

/--
Check `config` with `optim.Optimizer.validateFor` for `α`, and also for binary32 when `runtime`
uses CUDA, then pass the matching runtime optimizer to `continuation`. Shared by
`packStep` and `Objective.dataStep`.
-/
def withConfiguredOptimizer {α Result : Type}
    [TorchLean.Storage α] [Context α] [FromFloat α]
    {stateShapes : List Shape} (runtime : Session.Config) (config : optim.Optimizer)
    (continuation : Optimizer α stateShapes → IO Result) :
    IO Result := do
  IO.ofExcept (config.validateFor (α := α))
  if runtime.usesCUDA then
    IO.ofExcept (config.validateFor (α := Float32) (scalarName := "binary32"))
  let cast := ofFloat (α := α)
  match optim.Optimizer.toView config with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        continuation (Optimizer.sgd (cast lr))
      else
        continuation (Optimizer.momentumSGD (cast lr) (cast momentum))
  | .adaGrad lr epsilon =>
      continuation (Optimizer.adagrad (cast lr) (cast epsilon))
  | .rmsProp lr decay epsilon =>
      continuation (Optimizer.rmsprop
        (cast lr) (cast decay) (cast epsilon))
  | .adam lr beta1 beta2 epsilon =>
      continuation (Optimizer.adam
        (cast lr) (cast beta1) (cast beta2) (cast epsilon))
  | .adaDelta lr rho epsilon =>
      continuation (Optimizer.adadelta
        (cast lr) (cast rho) (cast epsilon))
  | .adamW lr weightDecay beta1 beta2 epsilon =>
      continuation (Optimizer.adamw
        (cast lr) (cast weightDecay) (cast beta1) (cast beta2) (cast epsilon))

/--
Build a one-step update function from an `optim.*` config, the same record that
`Trainer.RunConfig` takes, for an objective with any input and data packs. Custom training loops
keep their own control flow and call the returned function once per update.
-/
def packStep {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] [Context α] [FromFloat α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (objective : Objective α β stateShapes inputShapes dataInputShapes)
    (config : optim.Optimizer) :
    IO (TensorPack α inputShapes → TensorPack β dataInputShapes → IO Unit) := do
  Module.withConfiguredOptimizer
      (Objective.runtime objective).runtime config fun optimizer => do
    let bound ← Module.bindOptimizer objective optimizer
    pure fun arguments dataArguments =>
      bound.step arguments
        dataArguments

/--
Build a training step for a data-only objective with two data inputs, such as inputs and targets.
Each call of the returned function computes the gradient on one pair of tensors and applies one
update of the optimizer that `config` selects, the work of `loss.backward()` and
`optimizer.step()` in PyTorch. The data tensors are in `β`, which may differ from the parameter
type `α`, and the optimizer state stays inside the returned function.

The settings in `config` must stay in their domains after conversion to `α`, as checked by
`optim.Optimizer.validateFor` through the `FromFloat α` instance. On CUDA they must also
be valid in binary32, even when `α` is `Float`.

```lean
let trainStep ← objective.dataStep (optim.adamW { lr := 3e-4 })
for batch in batches do
  trainStep batch.input batch.target
```
-/
def Objective.dataStep {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α] [FromFloat α]
    {stateShapes : List Shape} {firstShape secondShape : Shape}
    (objective : Objective α β stateShapes [] [firstShape, secondShape])
    (config : optim.Optimizer) :
    IO (Tensor β firstShape → Tensor β secondShape → IO Unit) := do
  withConfiguredOptimizer
      (Objective.runtime objective).runtime config fun optimizer => do
    let bound ← bindOptimizer objective optimizer
    pure fun first second => bound.step Pack.nil (Pack.pair first second)

end Module


end TorchLean
