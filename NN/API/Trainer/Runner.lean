/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Core
public import NN.API.Trainer.BatchInput
public import NN.API.Trainer.Scheduler
public import NN.Data.SampleStream

/-!
# Trainer Runtime Internals

Scalar-generic implementation of the trainer: an instantiated model with reusable forward and loss
evaluators (`Runner`), gradient accumulation and optimizer updates, and the stateful `Stepper` that
applies a configured optimizer and schedule.

Everything here is indexed by the runtime scalar `α`. `Trainer.openTyped` preserves that scalar at
the session boundary; the runtime-selected `Trainer.open`, `Trainer.train`, and `Trainer.predict`
paths expose `Float`.
-/

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Internal

/--
A checked model instantiated under one runtime scalar.

This bundles the imperative runtime objective (parameters and buffers stored in refs) and reusable
no-gradient forward and evaluation-loss evaluators over the same live state. Training forwards
update their buffers; evaluation forwards leave them unchanged.
-/
structure Runner (α : Type) [TorchLean.Storage α] [Context α]
    {σ τ : Spec.Shape} (model : TorchLean.nn.Sequential σ τ) where
  private mk ::
  private runtimeObjective :
    TorchLean.Module.Objective α Unit (TorchLean.nn.stateShapes model) [σ, τ]
  private trainingPredictor :
    Runtime.Autograd.Model.Module.Evaluator α Unit (TorchLean.nn.stateShapes model) [σ] [] τ
  private evaluationPredictor :
    Runtime.Autograd.Model.Module.Evaluator α Unit (TorchLean.nn.stateShapes model) [σ] [] τ
  private evaluationLossEvaluator :
    Runtime.Autograd.Model.Module.ObjectiveEvaluator α Unit
      (TorchLean.nn.stateShapes model) [σ, τ]

namespace Runner

/-- Construct a runner from its objective and evaluators. -/
opaque create {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runtimeObjective :
      TorchLean.Module.Objective α Unit (TorchLean.nn.stateShapes model) [σ, τ])
    (trainingPredictor :
      Runtime.Autograd.Model.Module.Evaluator α Unit (TorchLean.nn.stateShapes model) [σ] [] τ)
    (evaluationPredictor :
      Runtime.Autograd.Model.Module.Evaluator α Unit (TorchLean.nn.stateShapes model) [σ] [] τ)
    (evaluationLossEvaluator :
      Runtime.Autograd.Model.Module.ObjectiveEvaluator α Unit
        (TorchLean.nn.stateShapes model) [σ, τ]) :
    Runner α model :=
  ⟨runtimeObjective, trainingPredictor, evaluationPredictor, evaluationLossEvaluator⟩

/-- The executable runtime module behind a runner. -/
opaque objectiveModule {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) :
    TorchLean.Module.Objective α Unit (TorchLean.nn.stateShapes model) [σ, τ] :=
  match runner with
  | ⟨runtimeObjective, _, _, _⟩ => runtimeObjective

/-- The reusable no-gradient evaluator for one execution mode. -/
opaque predictor {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) (mode : nn.Mode) :
    Runtime.Autograd.Model.Module.Evaluator α Unit (TorchLean.nn.stateShapes model) [σ] [] τ :=
  match runner, mode with
  | ⟨_, trainingPredictor, _, _⟩, .train => trainingPredictor
  | ⟨_, _, evaluationPredictor, _⟩, .eval => evaluationPredictor

/-- Reusable evaluator for the evaluation-mode loss. -/
opaque evaluationEvaluator {σ τ : Spec.Shape}
    {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) :
    Runtime.Autograd.Model.Module.ObjectiveEvaluator α Unit
      (TorchLean.nn.stateShapes model) [σ, τ] :=
  match runner with
  | ⟨_, _, _, evaluator⟩ => evaluator

/-- Runtime configuration used by the runner's objective. -/
def runtime {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) : Runtime.Config :=
  (TorchLean.Module.Objective.Internal.runtime (objectiveModule runner)).runtime

/-- Finish runner construction once its executable objective has been instantiated. -/
def fromRuntimeObjective {σ τ : Spec.Shape}
    (model : TorchLean.nn.Sequential σ τ) (objective : Trainer.Objective τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    [Runtime.TensorTransfer α]
    (runtimeObjective :
      TorchLean.Module.Objective α Unit (TorchLean.nn.stateShapes model) [σ, τ]) :
    IO (Runner α model) := do
  let executableObjective := TorchLean.Module.Objective.Internal.runtime runtimeObjective
  let makePredictor (mode : nn.Mode) :=
    Runtime.Autograd.Model.Module.Evaluator.withState (β := Unit)
      (stateShapes := TorchLean.nn.stateShapes model) (inputShapes := [σ])
      (dataInputShapes := [])
      (program := Runtime.Autograd.Model.Layers.Seq.forward model mode (α := α))
      executableObjective.runtime executableObjective.trainer.state
  let trainingPredictor ← makePredictor .train
  let evaluationPredictor ← makePredictor .eval
  let evaluationLossEvaluator ←
    Runtime.Autograd.Model.Module.ObjectiveDef.evaluatorWithState
      (objective.definition model (mode := .eval))
      executableObjective.runtime executableObjective.trainer.state
  pure (create runtimeObjective trainingPredictor evaluationPredictor
    evaluationLossEvaluator)

/--
Instantiate a model and objective under a runtime scalar.

Explicit state is retained in `α`; otherwise stored Float initializers are cast with `ofFloat`.
-/
def instantiate {σ τ : Spec.Shape} (model : TorchLean.nn.Sequential σ τ)
    (objective : Trainer.Objective τ)
    (options : Runtime.Autograd.Torch.Config := {})
    (α : Type := Float32)
    [TorchLean.Storage α] [Context α]
    [TorchLean.Runtime.FromFloat α]
    [Runtime.TensorTransfer α]
    (initialState? : Option (nn.State α (TorchLean.nn.stateShapes model)) := none) :
    IO (Runner α model) := do
  let runtimeObjective ←
    TorchLean.Module.instantiate (α := α) (objective.definition model) options
      (initialState? := initialState?)
  fromRuntimeObjective model objective runtimeObjective

/-- Read the complete parameter-and-buffer state. -/
def state {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) :
    IO (nn.State α (TorchLean.nn.stateShapes model)) :=
  TorchLean.Module.Objective.state (objectiveModule runner)

/-- Replace the complete parameter-and-buffer state. -/
def setState {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) (state : nn.State α (TorchLean.nn.stateShapes model)) :
    IO Unit :=
  TorchLean.Module.Objective.setState (objectiveModule runner) state

/-- Initialize the state owned by a runtime optimizer for this runner. -/
def initOptimizer {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (optimizer : Runtime.Autograd.Model.Optim.Optimizer α (TorchLean.nn.stateShapes model)) :
    IO optimizer.State :=
  TorchLean.Module.Objective.initOptimizer (objectiveModule runner) optimizer

/-- Evaluate one input in the given mode (training by default). -/
def forward {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (input : Tensor α σ) (mode : nn.Mode := .train) : IO (Tensor α τ) :=
  Runtime.Autograd.Model.Module.Evaluator.run (predictor runner mode)
    (TorchLean.TensorPack.singleton input) TorchLean.TensorPack.empty

/--
Scalar loss of one sample in the given mode (training by default).

Training uses the objective's random stream and buffer updates. Evaluation uses a reusable
no-gradient evaluator over the live parameters without updating running buffers.
-/
def loss {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) (sample : TorchLean.Sample.Supervised α σ τ)
    (mode : nn.Mode := .train) : IO α := do
  let value ← match mode with
    | .train =>
      TorchLean.Module.Objective.loss (objectiveModule runner)
        (TorchLean.Sample.Internal.arguments sample) TorchLean.Arguments.empty
    | .eval =>
      Runtime.Autograd.Model.Module.Evaluator.run (evaluationEvaluator runner)
        (TorchLean.Arguments.Internal.toTensorPack (TorchLean.Sample.Internal.arguments sample))
        TorchLean.TensorPack.empty
  pure value.item

end Runner

/-! ## Optimizer binding -/

/-- Bind a configured optimizer, its initialized state, and its scheduled state updater. -/
def withBoundOptimizer {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α β : Type} [TorchLean.Storage α] [Context α]
    [TorchLean.Runtime.FromFloat α]
    (runner : Runner α model) (config : TorchLean.optim.Optimizer)
    (scheduler : Option TorchLean.Trainer.Scheduler.Config)
    (continuation :
      (optimizer : Runtime.Autograd.Model.Optim.Optimizer α (TorchLean.nn.stateShapes model)) →
      optimizer.State → (Nat → optimizer.State → optimizer.State) → IO β) : IO β := do
  match config.validateFor (α := α) with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message
  match scheduler with
  | some schedule => do
      IO.ofExcept <| TorchLean.Trainer.Scheduler.validateWith
        (TorchLean.Runtime.FromFloat.roundForValidation (α := α)) schedule
  | none => pure ()
  let objective := Runner.objectiveModule runner
  let learningRateAtStep (step : Nat) : Float :=
    scheduler.map (fun config => TorchLean.Trainer.Scheduler.learningRateAt config step)
      |>.getD config.learningRate
  let rec mapStateList
      {State : (α : Type) → [TorchLean.Storage α] → Spec.Shape → Type} :
      {shapes : List Spec.Shape} →
      ({s : Spec.Shape} → State α s → State α s) →
      Runtime.Autograd.Model.Optim.StateList State α shapes →
      Runtime.Autograd.Model.Optim.StateList State α shapes :=
    fun {_shapes} transform states =>
      match states with
      | Runtime.Autograd.Model.Optim.StateList.nil =>
          Runtime.Autograd.Model.Optim.StateList.nil
      | Runtime.Autograd.Model.Optim.StateList.cons shapeState remainingStates =>
          Runtime.Autograd.Model.Optim.StateList.cons
            (transform shapeState) (mapStateList transform remainingStates)
  let shapes := TorchLean.nn.stateShapes model
  -- Every optimizer state stores its learning rate per shape; `setRate` names that field.
  let scheduled {State : (α : Type) → [TorchLean.Storage α] → Spec.Shape → Type}
      (setRate : {s : Spec.Shape} → α → State α s → State α s) (step : Nat) :
      Runtime.Autograd.Model.Optim.StateList State α shapes →
      Runtime.Autograd.Model.Optim.StateList State α shapes :=
    mapStateList (setRate (TorchLean.Runtime.ofFloat (learningRateAtStep step)))
  match TorchLean.optim.Optimizer.Internal.view config with
  | .sgd learningRate momentum =>
      if momentum == 0.0 then
        let optimizer := Runtime.Autograd.Model.Optim.sgd (α := α) (paramShapes := shapes)
          (TorchLean.Runtime.ofFloat learningRate)
        let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
        continuation optimizer state
          (scheduled fun rate shapeState => { shapeState with learningRate := rate })
      else
        let optimizer := Runtime.Autograd.Model.Optim.momentumSGD
          (α := α) (paramShapes := shapes)
          (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat momentum)
        let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
        continuation optimizer state
          (scheduled fun rate shapeState => { shapeState with learningRate := rate })
  | .adaGrad learningRate epsilon =>
      let optimizer := Runtime.Autograd.Model.Optim.adagrad (α := α) (paramShapes := shapes)
        (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
      continuation optimizer state
        (scheduled fun rate shapeState => { shapeState with learningRate := rate })
  | .rmsProp learningRate decay epsilon =>
      let optimizer := Runtime.Autograd.Model.Optim.rmsprop (α := α) (paramShapes := shapes)
        (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat decay)
        (TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
      continuation optimizer state
        (scheduled fun rate shapeState => { shapeState with learningRate := rate })
  | .adam learningRate beta1 beta2 epsilon =>
      let optimizer := Runtime.Autograd.Model.Optim.adam (α := α) (paramShapes := shapes)
        (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat beta1)
        (TorchLean.Runtime.ofFloat beta2) (TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
      continuation optimizer state
        (scheduled fun rate shapeState => { shapeState with learningRate := rate })
  | .adamW learningRate weightDecay beta1 beta2 epsilon =>
      let optimizer := Runtime.Autograd.Model.Optim.adamw (α := α) (paramShapes := shapes)
        (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat weightDecay)
        (TorchLean.Runtime.ofFloat beta1) (TorchLean.Runtime.ofFloat beta2)
        (TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
      continuation optimizer state
        (scheduled fun rate shapeState => { shapeState with learningRate := rate })
  | .adaDelta learningRate rho epsilon =>
      let optimizer := Runtime.Autograd.Model.Optim.adadelta (α := α) (paramShapes := shapes)
        (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat rho)
        (TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
      continuation optimizer state
        (scheduled fun rate shapeState => { shapeState with learningRate := rate })

/-! ## Gradient accumulation and optimizer updates -/

/-- Add two shape-aligned model-state gradients. -/
def addGradients {α : Type} [TorchLean.Storage α] [Add α]
    {shapes : List Spec.Shape}
    (first second : nn.State α shapes) : nn.State α shapes :=
  first.zipWith second fun firstTensor secondTensor =>
    TorchLean.Tensor.add firstTensor secondTensor

/-- Scale every tensor in a model-state gradient. -/
def scaleGradients {α : Type} [TorchLean.Storage α] [Mul α]
    {shapes : List Spec.Shape} (factor : α)
    (gradient : nn.State α shapes) : nn.State α shapes :=
  gradient.map fun tensor => TorchLean.Tensor.scale tensor factor

/--
Mean parameter gradient and mean loss for a nonempty batch at one parameter point.

With `withLoss := false` the losses are not read back and the second component is `0`.
-/
def meanGradAndLoss {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (batch : Array (TorchLean.Sample.Supervised α σ τ)) (withLoss : Bool) :
    IO (nn.State α (TorchLean.nn.stateShapes model) × α) := do
  match batch[0]? with
  | none => throw <| IO.userError "Trainer.meanGrad: empty batch"
  | some firstSample =>
      let objective := Runner.objectiveModule runner
      let sampleGradient (sample : TorchLean.Sample.Supervised α σ τ) :
          IO (nn.State α (TorchLean.nn.stateShapes model) × α) := do
        let arguments := TorchLean.Sample.Internal.arguments sample
        if withLoss then
          let (gradient, lossValue) ← TorchLean.Module.Objective.grad objective
            arguments TorchLean.Arguments.empty (value := true)
          pure (gradient, TorchLean.Tensor.item lossValue)
        else
          pure (← TorchLean.Module.Objective.grad objective arguments TorchLean.Arguments.empty,
            0)
      let (firstGradient, firstLoss) ← sampleGradient firstSample
      let mut gradientSum := firstGradient
      let mut lossSum := firstLoss
      for sample in batch.drop 1 do
        let (gradient, lossValue) ← sampleGradient sample
        lossSum := lossSum + lossValue
        gradientSum := addGradients gradientSum gradient
      let reciprocalCount : α := 1 / (batch.size : α)
      pure (scaleGradients reciprocalCount gradientSum, lossSum * reciprocalCount)

/--
Compute mean parameter gradients for a nonempty batch at one parameter point.

Set `value := true` to return `(meanGradient, meanLoss)`.
-/
def meanGrad {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (batch : Array (TorchLean.Sample.Supervised α σ τ)) (value : Bool := false) :
    IO (match value with
      | false => nn.State α (TorchLean.nn.stateShapes model)
      | true => nn.State α (TorchLean.nn.stateShapes model) × α) := by
  cases value with
  | false => exact Prod.fst <$> meanGradAndLoss runner batch false
  | true => exact meanGradAndLoss runner batch true

/--
Apply one optimizer update to a nonempty batch.

When `useNative` is set, supported CUDA optimizers accumulate gradients on device and use the
same moment state for every batch size. Other optimizers average explicit per-sample gradients.
Set `loss := true` to return `(nextOptimizerState, meanLoss)`.
-/
def step {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (optimizer : Runtime.Autograd.Model.Optim.Optimizer α (TorchLean.nn.stateShapes model))
    (state : optimizer.State) (useNative : Bool)
    (batch : Array (TorchLean.Sample.Supervised α σ τ)) (loss : Bool := false) :
    IO (match loss with
      | false => optimizer.State
      | true => optimizer.State × α) := by
  -- Without `withLoss` the loss is never read back and the second component is `0`.
  let run (withLoss : Bool) : IO (optimizer.State × α) := do
    if batch.isEmpty then
      throw <| IO.userError "Trainer.step: empty batch"
    let objective := Runner.objectiveModule runner
    if useNative then
      if hSingleton : batch.size = 1 then
        let sample := batch[0]'(by simp [hSingleton])
        let arguments := TorchLean.Sample.Internal.arguments sample
        if withLoss then
          let (nextOptimizerState, lossValue) ←
            TorchLean.Module.Objective.step objective optimizer state
              arguments TorchLean.Arguments.empty (loss := true)
          return (nextOptimizerState, TorchLean.Tensor.item lossValue)
        else
          return (← TorchLean.Module.Objective.step objective optimizer state
            arguments TorchLean.Arguments.empty, 0)
      let arguments := batch.map fun sample =>
        (TorchLean.Sample.Internal.arguments sample, TorchLean.Arguments.empty)
      if let some (nextState, lossValue) ←
          TorchLean.Module.Objective.Internal.tryNativeBatchStep
            objective optimizer state arguments withLoss then
        if withLoss then
          match lossValue with
          | some lossValue => return (nextState, lossValue.item)
          | none =>
            throw <| IO.userError "Trainer.step: native update omitted requested loss"
        else
          return (nextState, 0)
    let (meanGradient, meanLoss) ← meanGradAndLoss runner batch withLoss
    let nextOptimizerState ←
      TorchLean.Module.Objective.update objective optimizer state meanGradient
    pure (nextOptimizerState, meanLoss)
  cases loss with
  | false => exact Prod.fst <$> run false
  | true => exact run true

/-! ## Stateful steppers -/

/-- Stateful optimizer step functions and completed-step counter for one runner. -/
structure Stepper (α : Type) [TorchLean.Storage α] [Context α]
    {σ τ : Spec.Shape} (model : TorchLean.nn.Sequential σ τ) where
  private mk ::
  private runBatch : Array (TorchLean.Sample.Supervised α σ τ) → IO α
  private runBatchSilently : Array (TorchLean.Sample.Supervised α σ τ) → IO Unit
  private stepRef : IO.Ref Nat

namespace Stepper

/-- Construct a stepper from its hidden batch-step functions and counter. -/
opaque create {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runBatch : Array (TorchLean.Sample.Supervised α σ τ) → IO α)
    (runBatchSilently : Array (TorchLean.Sample.Supervised α σ τ) → IO Unit)
    (stepRef : IO.Ref Nat) : Stepper α model :=
  ⟨runBatch, runBatchSilently, stepRef⟩

/-- The hidden loss-returning batch-step function. -/
opaque action {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model) :
    Array (TorchLean.Sample.Supervised α σ τ) → IO α :=
  match stepper with
  | ⟨runBatch, _, _⟩ => runBatch

/-- The hidden batch-step function that does not read the loss back. -/
opaque silentAction {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model) :
    Array (TorchLean.Sample.Supervised α σ τ) → IO Unit :=
  match stepper with
  | ⟨_, runBatchSilently, _⟩ => runBatchSilently

/-- The hidden completed-step counter. -/
opaque counter {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model) : IO.Ref Nat :=
  match stepper with
  | ⟨_, _, stepRef⟩ => stepRef

/--
Apply one optimizer update, optionally returning its scalar loss.

`batch := true` takes a nonempty array and averages gradients at the same parameter point. The
counter advances once for the whole batch. `loss := false` avoids reading the loss back.
-/
def step {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ} {Input : Type}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model) (sample : Input) (batch : Bool := false) (loss : Bool := false)
    [BatchInput (TorchLean.Sample.Supervised α σ τ)
      (Array (TorchLean.Sample.Supervised α σ τ)) batch Input] :
    IO (match loss with | false => Unit | true => α) := by
  have inputType := BatchInput.type_eq (single := TorchLean.Sample.Supervised α σ τ)
    (many := Array (TorchLean.Sample.Supervised α σ τ)) (batch := batch)
  subst Input
  let samples : Array (TorchLean.Sample.Supervised α σ τ) := by
    cases batch with
    | false => exact #[sample]
    | true => exact sample
  cases loss with
  | false =>
      change IO Unit
      exact do
        if samples.isEmpty then
          throw <| IO.userError "Stepper.step: batch must be nonempty"
        silentAction stepper samples
  | true =>
      change IO α
      exact do
        if samples.isEmpty then
          throw <| IO.userError "Stepper.step: batch must be nonempty"
        action stepper samples

/-- Read the number of completed optimizer steps. -/
def steps {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model) : IO Nat :=
  (counter stepper).get

end Stepper

/--
Construct a `Stepper` for a runner, optimizer config, and optional scheduler.

Every step trains through the runner's objective, applies the schedule for the current step index,
and advances the completed-step counter once per batch. A singleton batch uses the runtime's
native single-sample update.
-/
def Runner.stepper {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α] [TorchLean.Runtime.FromFloat α]
    (runner : Runner α model) (optimizer : TorchLean.optim.Optimizer)
    (scheduler : Option TorchLean.Trainer.Scheduler.Config := none) :
    IO (Stepper α model) := do
  let stepRef ← IO.mkRef 0
  withBoundOptimizer runner optimizer scheduler
      fun runtimeOptimizer initialState scheduleState => do
    let stateRef ← IO.mkRef initialState
    let runBatch := fun (batch : Array (TorchLean.Sample.Supervised α σ τ)) => do
      let stepIndex ← stepRef.get
      let state := scheduleState stepIndex (← stateRef.get)
      let (nextOptimizerState, lossValue) ←
        step runner runtimeOptimizer state true batch (loss := true)
      stateRef.set nextOptimizerState
      stepRef.set (stepIndex + 1)
      pure lossValue
    let runBatchSilently := fun (batch : Array (TorchLean.Sample.Supervised α σ τ)) => do
      let stepIndex ← stepRef.get
      let state := scheduleState stepIndex (← stateRef.get)
      let nextOptimizerState ←
        step runner runtimeOptimizer state true batch
      stateRef.set nextOptimizerState
      stepRef.set (stepIndex + 1)
    pure (Stepper.create runBatch runBatchSilently stepRef)

end Internal
end Trainer
end TorchLean
