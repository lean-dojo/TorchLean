/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Module.Execution
public import NN.API.Neural.Indexed
public import NN.API.Sample
public import NN.API.Optim -- shake: keep

/-!
# Executable Modules

Executable module operations for manual runtime and example code.
-/

@[expose] public section

namespace TorchLean

namespace Module

/--
Live parameter and buffer storage shared by executable modules.

The shape list remains in the type, so replacing the state cannot silently reorder parameters or
load tensors with incompatible dimensions. Mutation is confined to this runtime object; model and
layer definitions remain immutable values that can be lowered and reasoned about.
-/
structure RuntimeState (α : Type) [TorchLean.Storage α] [Context α]
    (stateShapes : List Shape) where
  private mk ::
  private stateRef : Runtime.Autograd.Torch.ParamList α stateShapes
  private runtime : Runtime.Config
  private modeRef : IO.Ref nn.Mode
  private rngCounter : IO.Ref Nat

namespace RuntimeState

/-
Everything below is `opaque` rather than `def` for one reason: the fields above are `private`, and
an `@[expose]` body may not mention a private declaration. `opaque` keeps the body out of the
exposed interface, which is what lets implementation code reach the fields while callers outside
cannot.
-/
namespace Internal

/-- Assemble runtime state from storage that has already been allocated. -/
opaque create {α : Type} [TorchLean.Storage α] [Context α]
    {stateShapes : List Shape}
    (stateRef : Runtime.Autograd.Torch.ParamList α stateShapes)
    (runtime : Runtime.Config) (modeRef : IO.Ref nn.Mode) (rngCounter : IO.Ref Nat) :
    RuntimeState α stateShapes :=
  ⟨stateRef, runtime, modeRef, rngCounter⟩

/-- Instantiate runtime state after converting semantic initializer tensors to `α`. -/
def instantiate {α : Type} [TorchLean.Storage α] [Context α]
    [tensorTransfer : Runtime.TensorTransfer α]
    {stateShapes : List Shape} (initial : TorchLean.TensorPack Float stateShapes)
    (runtimeInit : Option
      (Runtime.Autograd.Model.Module.RuntimeInit.Plan stateShapes))
    (requiresGrad : Array Bool) (runtime : Runtime.Config) (cast : Float → α) :
    IO (RuntimeState α stateShapes) := do
  let stateRef ←
    match runtimeInit with
    | some plan => do
        let empty := Runtime.Autograd.Model.Module.RuntimeInit.zeroPack
          (cast 0.0) (ss := stateShapes)
        let stateRef ← Runtime.Autograd.Torch.ParamList.ofPackWithRequiresGrad
          empty requiresGrad
        Runtime.Autograd.Model.Module.RuntimeInit.applyPlan
          (α := α) cast runtime stateRef plan
        pure stateRef
    | none => do
        let values := Runtime.Autograd.Model.Module.castPack cast initial
        Runtime.Autograd.Torch.ParamList.ofPackWithRequiresGrad values requiresGrad
  let modeRef ← IO.mkRef nn.Mode.train
  let rngCounter ← IO.mkRef 0
  pure (create stateRef runtime modeRef rngCounter)

/-- Reveal parameter storage only to executable-module implementation code. -/
opaque stateRef {α : Type} [TorchLean.Storage α] [Context α]
    {stateShapes : List Shape} (state : RuntimeState α stateShapes) :
    Runtime.Autograd.Torch.ParamList α stateShapes :=
  match state with
  | ⟨stateRef, _, _, _⟩ => stateRef

/-- Reveal runtime configuration only to executable-module implementation code. -/
opaque runtime {α : Type} [TorchLean.Storage α] [Context α]
    {stateShapes : List Shape} (state : RuntimeState α stateShapes) :
    Runtime.Config :=
  match state with
  | ⟨_, runtime, _, _⟩ => runtime

/-- Reveal the train/eval mode cell only to executable-module implementation code. -/
opaque modeRef {α : Type} [TorchLean.Storage α] [Context α]
    {stateShapes : List Shape} (state : RuntimeState α stateShapes) :
    IO.Ref nn.Mode :=
  match state with
  | ⟨_, _, modeRef, _⟩ => modeRef

/-- Preserve random-operation order across successive forwards of the same module. -/
opaque rngCounter {α : Type} [TorchLean.Storage α] [Context α]
    {stateShapes : List Shape} (state : RuntimeState α stateShapes) : IO.Ref Nat :=
  match state with
  | ⟨_, _, _, counter⟩ => counter

end Internal

/-- Read the current behavior of training-sensitive layers. -/
def mode {α : Type} [TorchLean.Storage α] [Context α] {stateShapes : List Shape}
    (state : RuntimeState α stateShapes) : IO nn.Mode :=
  (Internal.modeRef state).get

/-- Enable training behavior. -/
def train {α : Type} [TorchLean.Storage α] [Context α] {stateShapes : List Shape}
    (state : RuntimeState α stateShapes) : IO Unit :=
  (Internal.modeRef state).set .train

/-- Enable evaluation behavior. -/
def eval {α : Type} [TorchLean.Storage α] [Context α] {stateShapes : List Shape}
    (state : RuntimeState α stateShapes) : IO Unit :=
  (Internal.modeRef state).set .eval

/-- Return `true` exactly when the state uses training behavior. -/
def isTraining {α : Type} [TorchLean.Storage α] [Context α]
    {stateShapes : List Shape} (state : RuntimeState α stateShapes) : IO Bool := do
  pure ((← state.mode) == .train)

/-- Read all parameters and buffers, synchronizing device storage when necessary. -/
def state {α : Type} [TorchLean.Storage α] [Context α]
    [Runtime.TensorTransfer α]
    {stateShapes : List Shape} (runtimeState : RuntimeState α stateShapes) :
    IO (nn.State α stateShapes) := do
  let tensors ← Runtime.Autograd.Torch.ParamList.valuesSynced (Internal.stateRef runtimeState)
  pure (nn.State.Internal.fromTensorPack tensors)

/-- Replace all parameters and buffers with a shape-compatible state. -/
def setState {α : Type} [TorchLean.Storage α] [Context α] {stateShapes : List Shape}
    (runtimeState : RuntimeState α stateShapes)
    (values : nn.State α stateShapes) : IO Unit :=
  Runtime.Autograd.Torch.ParamList.setValues (Internal.stateRef runtimeState)
    (nn.State.Internal.toTensorPack values)

end RuntimeState

end Module

namespace nn

/--
An ordinary tensor model with live, mutable runtime state.

`nn.Sequential` remains the immutable, shape-checked model definition used by proofs and graph
lowering. An `nn.Module` is its runtime counterpart: it owns one parameter set and a mutable
train/eval flag. This separation gives executable code the familiar module lifecycle without
hiding mutation inside the mathematical model.
-/
structure Module (α : Type) [TorchLean.Storage α] [Context α]
    {σ τ : Shape} (model : nn.Sequential σ τ) where
  private mk ::
  private runtimeState : TorchLean.Module.RuntimeState α (nn.stateShapes model)

/-- An indexed-input model with live, mutable runtime state. -/
structure IndexedModule (α β : Type) [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {σ τ : Shape} (model : nn.IndexedModel σ τ β) where
  private mk ::
  private runtimeState : TorchLean.Module.RuntimeState α model.stateShapes

namespace Module

namespace Internal

/-- Wrap live runtime state behind the public module boundary. -/
opaque fromRuntimeState {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ}
    (state : TorchLean.Module.RuntimeState α (nn.stateShapes model)) :
    Module α model :=
  ⟨state⟩

/-- Reveal live runtime state only to module implementation code. -/
opaque runtimeState {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ}
    (module : Module α model) :
    TorchLean.Module.RuntimeState α (nn.stateShapes model) :=
  match module with
  | ⟨state⟩ => state

end Internal

/--
Instantiate a checked model with mutable parameter and buffer storage.

Native binary32 is the default. Select another supported element type with
`nn.Module.instantiate model (α := Float)`.
-/
def instantiate {σ τ : Shape}
    (model : nn.Sequential σ τ)
    (runtime : Runtime.Config := {})
    (α : Type := Float32)
    [TorchLean.Storage α] [Context α] [Runtime.FromFloat α]
  [tensorTransfer : Runtime.TensorTransfer α] :
    IO (Module α model) := do
  match nn.validate model with
  | .error message => throw <| IO.userError message
  | .ok () => pure ()
  let state ← TorchLean.Module.RuntimeState.Internal.instantiate
    (nn.State.Internal.toTensorPack (nn.initialState model))
    (nn.runtimeInit? model) (nn.requiresGrad model)
    runtime (Runtime.ofFloat (α := α))
  pure (Internal.fromRuntimeState state)

/-- Read whether training-sensitive layers currently use training or evaluation behavior. -/
def mode {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ} (module : Module α model) :
    IO nn.Mode :=
  (Internal.runtimeState module).mode

/-- Enable training behavior for dropout, normalization buffers, and similar layers. -/
def train {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ} (module : Module α model) : IO Unit :=
  (Internal.runtimeState module).train

/-- Enable deterministic evaluation behavior for training-sensitive layers. -/
def eval {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ} (module : Module α model) : IO Unit :=
  (Internal.runtimeState module).eval

/-- Return `true` exactly when this module uses training behavior. -/
def isTraining {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ} (module : Module α model) : IO Bool := do
  (Internal.runtimeState module).isTraining

/-- Read the complete parameter-and-buffer state, synchronizing device storage when necessary. -/
def state {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    [Runtime.TensorTransfer α]
    {model : nn.Sequential σ τ} (module : Module α model) :
    IO (nn.State α (nn.stateShapes model)) :=
  (Internal.runtimeState module).state

/-- Replace the complete shape-indexed parameter-and-buffer state. -/
def setState {σ τ : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    {model : nn.Sequential σ τ} (module : Module α model)
    (state : nn.State α (nn.stateShapes model)) : IO Unit :=
  (Internal.runtimeState module).setState state

namespace Internal

/-- Execute one ordinary module forward in an explicit mode without changing its mode cell. -/
def forwardWithMode {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    [tensorTransfer : Runtime.TensorTransfer α]
    {model : nn.Sequential σ τ} (module : Module α model)
    (mode : nn.Mode) (input : Tensor α σ) : IO (Tensor α τ) := do
  let state := Internal.runtimeState module
  Runtime.Autograd.Model.Layers.Seq.forwardNoGrad
    (α := α) (tensorTransfer := tensorTransfer)
    (TorchLean.Module.RuntimeState.Internal.runtime state) model
    (TorchLean.Module.RuntimeState.Internal.stateRef state) input
    (mode := mode) (rngCounter := some (TorchLean.Module.RuntimeState.Internal.rngCounter state))

end Internal

/--
Evaluate one concrete input without constructing a backward tape.

The module's mode controls training-sensitive layer behavior. Training forwards update running
buffers from their actual activations and advance the module's random stream. This concrete call
does not construct a backward tape; differentiable model programs use `nn.forward`.
-/
def forward {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    [tensorTransfer : Runtime.TensorTransfer α]
    {model : nn.Sequential σ τ} (module : Module α model)
    (input : Tensor α σ) : IO (Tensor α τ) := do
  Internal.forwardWithMode module (← module.mode) input

/--
Evaluation-mode inference without changing the module's persistent mode.

Keeping the override local to this call prevents concurrent forwards from observing a temporary
mode change.
-/
def predict {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    [tensorTransfer : Runtime.TensorTransfer α]
    {model : nn.Sequential σ τ} (module : Module α model)
    (input : Tensor α σ) : IO (Tensor α τ) :=
  Internal.forwardWithMode module .eval input

end Module

namespace IndexedModule

namespace Internal

/-- Wrap live runtime state behind the public indexed-module boundary. -/
opaque fromRuntimeState {σ τ : Shape} {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {model : nn.IndexedModel σ τ β}
    (state : TorchLean.Module.RuntimeState α model.stateShapes) :
    IndexedModule α β model :=
  ⟨state⟩

/-- Reveal live runtime state only to indexed-module implementation code. -/
opaque runtimeState {σ τ : Shape} {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {model : nn.IndexedModel σ τ β}
    (module : IndexedModule α β model) :
    TorchLean.Module.RuntimeState α model.stateShapes :=
  match module with
  | ⟨state⟩ => state

end Internal

/--
Instantiate an indexed model. Native binary32 is the default runtime element type.
-/
def instantiate {σ τ : Shape} {β : Type} [TorchLean.Storage β]
    (model : nn.IndexedModel σ τ β)
    (runtime : Runtime.Config := {})
    (α : Type := Float32)
    [TorchLean.Storage α] [Context α] [Runtime.FromFloat α]
    [tensorTransfer : Runtime.TensorTransfer α] :
    IO (IndexedModule α β model) := do
  match model.validate with
  | .error message => throw <| IO.userError message
  | .ok () => pure ()
  let state ← TorchLean.Module.RuntimeState.Internal.instantiate
    (nn.State.Internal.toTensorPack model.initialState)
    (nn.IndexedModel.Internal.initializationPlan model)
    (nn.IndexedModel.Internal.trainableMask model)
    runtime (Runtime.ofFloat (α := α))
  pure (Internal.fromRuntimeState state)

/-- Read the current behavior of training-sensitive layers. -/
def mode {σ τ : Shape} {α β : Type} [TorchLean.Storage α]
    [TorchLean.Storage β] [Context α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model) :
    IO nn.Mode :=
  (Internal.runtimeState module).mode

/-- Enable training behavior. -/
def train {σ τ : Shape} {α β : Type} [TorchLean.Storage α]
    [TorchLean.Storage β] [Context α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model) : IO Unit :=
  (Internal.runtimeState module).train

/-- Enable evaluation behavior. -/
def eval {σ τ : Shape} {α β : Type} [TorchLean.Storage α]
    [TorchLean.Storage β] [Context α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model) : IO Unit :=
  (Internal.runtimeState module).eval

/-- Return `true` exactly when the module uses training behavior. -/
def isTraining {σ τ : Shape} {α β : Type} [TorchLean.Storage α]
    [TorchLean.Storage β] [Context α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model) : IO Bool :=
  (Internal.runtimeState module).isTraining

/-- Read the complete parameter-and-buffer state. -/
def state {σ τ : Shape} {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    [Runtime.TensorTransfer α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model) :
    IO (nn.State α model.stateShapes) :=
  (Internal.runtimeState module).state

/-- Replace the complete shape-indexed parameter-and-buffer state. -/
def setState {σ τ : Shape} {α β : Type} [TorchLean.Storage α]
    [TorchLean.Storage β] [Context α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model)
    (state : nn.State α model.stateShapes) : IO Unit :=
  (Internal.runtimeState module).setState state

namespace Internal

/-- Execute one indexed module forward in an explicit mode without changing its mode cell. -/
def forwardWithMode {σ τ : Shape} {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    [tensorTransfer : Runtime.TensorTransfer α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model)
    (mode : nn.Mode) (input : Tensor β σ) : IO (Tensor α τ) := do
  match nn.IndexedModel.Internal.validateInput model input with
  | .error message => throw <| IO.userError message
  | .ok () =>
      let state := Internal.runtimeState module
      let program : Runtime.Autograd.Model.ProgramWithDataInputs α β
          (model.stateShapes ++ []) [σ] τ :=
        fun {m} _ _ => by
          simpa using
            (nn.IndexedModel.Internal.program model mode (α := α) (m := m))
      let evaluator ← Runtime.Autograd.Model.Module.Evaluator.withState
        (program := program)
        (TorchLean.Module.RuntimeState.Internal.runtime state)
        (TorchLean.Module.RuntimeState.Internal.stateRef state)
        (rngCounter := some (TorchLean.Module.RuntimeState.Internal.rngCounter state))
      Runtime.Autograd.Model.Module.Evaluator.run
        evaluator TensorPack.empty (TensorPack.singleton input)

end Internal

/-- Evaluate one validated index tensor without constructing a backward tape. -/
def forward {σ τ : Shape} {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    [tensorTransfer : Runtime.TensorTransfer α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model)
    (input : Tensor β σ) : IO (Tensor α τ) := do
  Internal.forwardWithMode module (← module.mode) input

/--
Evaluation-mode inference without changing the module's persistent mode.

Keeping the override local to this call prevents concurrent forwards from observing a temporary
mode change.
-/
def predict {σ τ : Shape} {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    [tensorTransfer : Runtime.TensorTransfer α]
    {model : nn.IndexedModel σ τ β} (module : IndexedModule α β model)
    (input : Tensor β σ) : IO (Tensor α τ) :=
  Internal.forwardWithMode module .eval input

end IndexedModule

end nn

namespace Module

namespace Supervised

/-- Run evaluation-mode prediction through a supervised runtime module. -/
def predict {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    [Runtime.TensorTransfer α]
    (options : Runtime.Config)
    (model : TorchLean.nn.Sequential σ τ)
    (objective : Objective α Unit (nn.stateShapes model) [σ, τ])
    (input : Tensor α σ) : IO (Tensor α τ) := do
  let runtimeObjective := Objective.Internal.runtime objective
  Runtime.Autograd.Model.Layers.Seq.predict
    (α := α) options model runtimeObjective.trainer.state input

end Supervised

namespace Internal

/--
Evaluate one supervised sample through a runtime module and return the scalar loss value.

This packages the common internal pattern `Module.Objective.loss ...; Tensor.item`.
-/
def sampleLoss {σ τ : Shape} {α : Type}
    [TorchLean.Storage α] [Context α]
    [Runtime.TensorTransfer α]
    (model : TorchLean.nn.Sequential σ τ)
    (objective : Objective α Unit (nn.stateShapes model) [σ, τ])
    (sample : Sample.Supervised α σ τ) : IO α := do
  let loss ← Objective.loss (α := α) objective
    (Sample.Internal.arguments sample) Arguments.empty
  pure (TorchLean.Tensor.item loss)

/-- Bind an optimizer to the hidden runtime objective. -/
def bindOptimizer {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (objective : Objective α β stateShapes inputShapes dataInputShapes)
    (optimizer : Runtime.Autograd.Model.Optim.Optimizer α stateShapes) :=
  Runtime.Autograd.Model.Module.bindOptimizer
    (Objective.Internal.runtime objective) optimizer

end Internal

/--
Create a reusable evaluator for a data-only objective that shares an executable module's state.

The returned function accepts ordinary tensors and hides both the heterogeneous pack representation
and the reusable eager session used underneath.
-/
def Objective.dataLossEvaluator {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    [Runtime.TensorTransfer α]
    {stateShapes : List Shape} {firstShape secondShape : Shape}
    (objective : Objective α β stateShapes [] [firstShape, secondShape])
    (definition : ObjectiveDefinition β stateShapes [] [firstShape, secondShape]) :
    IO (Tensor β firstShape → Tensor β secondShape → IO (Tensor α [])) := do
  let runtimeObjective := Objective.Internal.runtime objective
  let evaluator ←
    Runtime.Autograd.Model.Module.ObjectiveDef.evaluatorWithState
      definition runtimeObjective.runtime runtimeObjective.trainer.state
  pure fun first second =>
    Runtime.Autograd.Model.Module.Evaluator.run
      evaluator TensorPack.empty (TensorPack.pair first second)

/--
Create a reusable evaluation-mode predictor for an indexed model sharing this objective's state.

This is useful after training an indexed loss: prediction sees the updated parameters without
requiring callers to construct a runtime program, evaluator, or tensor pack.
-/
def Objective.indexedPredictor {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α]
    [Runtime.TensorTransfer α]
    {stateShapes : List Shape} {σ τ : Shape}
    (objective : Objective α β stateShapes [] [σ, σ])
    (model : nn.IndexedModel σ τ β) :
    IO (Tensor β σ → IO (Tensor α τ)) := do
  if hState : model.stateShapes = stateShapes then
    let runtimeObjective := Objective.Internal.runtime objective
    let program : Runtime.Autograd.Model.ProgramWithDataInputs α β
        (stateShapes ++ []) [σ] τ := by
      subst stateShapes
      exact fun {m} _ _ => by
        simpa only [List.append_nil] using
          (nn.IndexedModel.Internal.program model .eval (α := α) (m := m))
    let evaluator ← Runtime.Autograd.Model.Module.Evaluator.withState
      (program := program) runtimeObjective.runtime runtimeObjective.trainer.state
      (validateDataInputs := fun
        | .cons input .nil => nn.IndexedModel.Internal.validateInput model input)
    pure fun input =>
      Runtime.Autograd.Model.Module.Evaluator.run
        evaluator TensorPack.empty (TensorPack.singleton input)
  else
    throw <| IO.userError
      "indexed predictor model does not match the trained objective state"

/-- Validate and instantiate the optimizer shared by manual and mixed-dtype module loops. -/
def Internal.withConfiguredOptimizer {α Result : Type}
    [TorchLean.Storage α] [Context α] [Runtime.FromFloat α]
    {stateShapes : List Shape} (runtime : Runtime.Config) (config : optim.Optimizer)
    (continuation : Runtime.Autograd.Model.Optim.Optimizer α stateShapes → IO Result) :
    IO Result := do
  IO.ofExcept (config.validateFor (α := α))
  if runtime.usesCuda then
    IO.ofExcept config.validateFloat32
  let cast := Runtime.ofFloat (α := α)
  match optim.Optimizer.Internal.view config with
  | .sgd learningRate momentum =>
      if momentum == 0.0 then
        continuation (Runtime.Autograd.Model.Optim.sgd (cast learningRate))
      else
        continuation (Runtime.Autograd.Model.Optim.momentumSGD (cast learningRate) (cast momentum))
  | .adaGrad learningRate epsilon =>
      continuation (Runtime.Autograd.Model.Optim.adagrad (cast learningRate) (cast epsilon))
  | .rmsProp learningRate decay epsilon =>
      continuation (Runtime.Autograd.Model.Optim.rmsprop
        (cast learningRate) (cast decay) (cast epsilon))
  | .adam learningRate beta1 beta2 epsilon =>
      continuation (Runtime.Autograd.Model.Optim.adam
        (cast learningRate) (cast beta1) (cast beta2) (cast epsilon))
  | .adaDelta learningRate rho epsilon =>
      continuation (Runtime.Autograd.Model.Optim.adadelta
        (cast learningRate) (cast rho) (cast epsilon))
  | .adamW learningRate weightDecay beta1 beta2 epsilon =>
      continuation (Runtime.Autograd.Model.Optim.adamw
        (cast learningRate) (cast weightDecay) (cast beta1) (cast beta2) (cast epsilon))

/--
Create a one-step update function for any typed module input pack from the public optimizer config
used by the trainer API.

Generic bridge for custom training loops: richer examples can keep their own control flow while
still choosing a public `optim.*` config through the same API as `Trainer.RunConfig`.
-/
def Internal.packStep {α : Type}
    [TorchLean.Storage α] [Context α] [Runtime.FromFloat α]
    {stateShapes inputShapes : List Shape}
    (objective : Objective α Unit stateShapes inputShapes)
    (config : optim.Optimizer) :
    IO (Arguments α inputShapes → IO Unit) := do
  Internal.withConfiguredOptimizer
      (Objective.Internal.runtime objective).runtime config fun optimizer => do
    let bound ← Internal.bindOptimizer objective optimizer
    pure fun arguments =>
      bound.step (Arguments.Internal.toTensorPack arguments) TensorPack.empty

/--
Create a two-input update function for a mixed-dtype, data-only objective.

The returned function takes ordinary typed tensors. Optimizer state and heterogeneous runtime packs
remain implementation details. Configuration values must remain in their required domains after
conversion to the optimizer scalar, as checked by its `Runtime.FromFloat` instance. CUDA also
requires binary32-valid settings even when the host scalar is `Float`.
-/
def Objective.dataStep {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [Context α] [Runtime.FromFloat α]
    {stateShapes : List Shape} {firstShape secondShape : Shape}
    (objective : Objective α β stateShapes [] [firstShape, secondShape])
    (config : optim.Optimizer) :
    IO (Tensor β firstShape → Tensor β secondShape → IO Unit) := do
  Internal.withConfiguredOptimizer
      (Objective.Internal.runtime objective).runtime config fun optimizer => do
    let bound ← Internal.bindOptimizer objective optimizer
    pure fun first second => bound.step TensorPack.empty (TensorPack.pair first second)

end Module


end TorchLean
