/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Core
public import NN.API.Trainer.Scheduler
public import NN.Data.SampleStream

/-!
# Trainer Runtime Internals

Scalar-generic implementation of the trainer: an instantiated model with reusable forward and loss
evaluators (`Runner`), gradient accumulation and optimizer updates, and the stateful `Stepper` that
applies a configured optimizer and schedule.

Everything here is indexed by the runtime scalar `α`. The public API (`Trainer.Session`,
`Trainer.train`, `Trainer.predict`) wraps it and exposes `Float` only.
-/

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Internal

/--
A checked model instantiated under one runtime scalar.

This bundles the imperative runtime objective (parameters and buffers stored in refs), reusable
no-gradient forward and evaluation-loss evaluators over the same live state, and the current mode.
Training forwards update their buffers; evaluation forwards leave them unchanged.
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
  private modeRef : IO.Ref nn.Mode

namespace Runner

/-- Construct a runner from its objective, evaluators, and mode cell. -/
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
        (TorchLean.nn.stateShapes model) [σ, τ])
    (modeRef : IO.Ref nn.Mode) :
    Runner α model :=
  ⟨runtimeObjective, trainingPredictor, evaluationPredictor, evaluationLossEvaluator, modeRef⟩

/-- The executable runtime module behind a runner. -/
opaque objectiveModule {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) :
    TorchLean.Module.Objective α Unit (TorchLean.nn.stateShapes model) [σ, τ] :=
  match runner with
  | ⟨runtimeObjective, _, _, _, _⟩ => runtimeObjective

/-- The reusable no-gradient evaluator for one execution mode. -/
opaque predictor {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) (mode : nn.Mode) :
    Runtime.Autograd.Model.Module.Evaluator α Unit (TorchLean.nn.stateShapes model) [σ] [] τ :=
  match runner, mode with
  | ⟨_, trainingPredictor, _, _, _⟩, .train => trainingPredictor
  | ⟨_, _, evaluationPredictor, _, _⟩, .eval => evaluationPredictor

/-- Reusable evaluator for the evaluation-mode loss. -/
opaque evaluationEvaluator {σ τ : Spec.Shape}
    {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) :
    Runtime.Autograd.Model.Module.ObjectiveEvaluator α Unit
      (TorchLean.nn.stateShapes model) [σ, τ] :=
  match runner with
  | ⟨_, _, _, evaluator, _⟩ => evaluator

/-- The runner's mode cell. -/
opaque modeCell {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) : IO.Ref nn.Mode :=
  match runner with
  | ⟨_, _, _, _, modeRef⟩ => modeRef

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
  let modeRef : IO.Ref nn.Mode ← IO.mkRef .train
  pure (create runtimeObjective trainingPredictor evaluationPredictor
    evaluationLossEvaluator modeRef)

/-- Instantiate a model and objective under a runtime scalar, injecting literals with `ofFloat`. -/
def instantiate {σ τ : Spec.Shape} (model : TorchLean.nn.Sequential σ τ)
    (objective : Trainer.Objective τ)
    (options : Runtime.Autograd.Torch.Config := {})
    (α : Type := Float32)
    [TorchLean.Storage α] [Context α]
    [TorchLean.Runtime.FromFloat α]
    [Runtime.TensorTransfer α] :
    IO (Runner α model) := do
  let runtimeObjective ←
    TorchLean.Module.instantiate (α := α) (objective.definition model) options
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

/-- Read every tensor of a runtime pack back to host `Float`. -/
def readFloatPack {α : Type} [TorchLean.Storage α] [Context α]
    [Runtime.TensorTransfer α] :
    {shapes : List Shape} →
    TorchLean.TensorPack α shapes →
    IO (TorchLean.TensorPack Float shapes)
  | _, .nil => pure .nil
  | _, .cons tensor rest => do
      let floatTensor ← Runtime.readFloatTensor tensor
      let floatRest ← readFloatPack rest
      pure (.cons floatTensor floatRest)

/-- Read the state back as host `Float` tensors; exact for binary32 runtime scalars. -/
def floatState {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α] [Runtime.TensorTransfer α]
    (runner : Runner α model) :
    IO (nn.State Float (TorchLean.nn.stateShapes model)) := do
  let runtimeState ← state runner
  let tensors ← readFloatPack (nn.State.Internal.toTensorPack runtimeState)
  pure (nn.State.Internal.fromTensorPack tensors)

/-- Replace the state from host `Float` tensors, casting them into the runtime scalar. -/
def setFloatState {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α] [Runtime.FromFloat α]
    (runner : Runner α model) (floatState : nn.State Float (TorchLean.nn.stateShapes model)) :
    IO Unit :=
  setState runner <| floatState.map fun tensor =>
    TorchLean.Tensor.map (Runtime.ofFloat (α := α)) tensor

/-- Initialize the state owned by a runtime optimizer for this runner. -/
def initOptimizer {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (optimizer : Runtime.Autograd.Model.Optim.Optimizer α (TorchLean.nn.stateShapes model)) :
    IO optimizer.State :=
  TorchLean.Module.Objective.initOptimizer (objectiveModule runner) optimizer

/-- Read the runner's current mode (`.train` or `.eval`). -/
def mode {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) : IO nn.Mode :=
  (modeCell runner).get

/-- Set the runner mode (`.train` or `.eval`). -/
def setMode {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) (value : nn.Mode) : IO Unit :=
  (modeCell runner).set value

/-- Select training behavior for stateful layers. -/
def train {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) : IO Unit :=
  setMode runner .train

/-- Select evaluation behavior for stateful layers. -/
def eval {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) : IO Unit :=
  setMode runner .eval

/-- Evaluate one input tensor in an explicit mode without changing the runner's mode cell. -/
def forwardWithMode {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (selectedMode : nn.Mode) (input : Tensor α σ) : IO (Tensor α τ) := do
  Runtime.Autograd.Model.Module.Evaluator.run (predictor runner selectedMode)
    (TorchLean.TensorPack.singleton input) TorchLean.TensorPack.empty

/-- Evaluate one input tensor using the active mode (`.train` or `.eval`). -/
def forward {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (input : Tensor α σ) : IO (Tensor α τ) := do
  forwardWithMode runner (← mode runner) input

/-- Run evaluation-mode prediction without changing the runner's persistent mode. -/
def predict {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (input : Tensor α σ) : IO (Tensor α τ) :=
  forwardWithMode runner .eval input

/--
Scalar loss of one supervised sample in an explicit mode without changing the runner's mode cell.

Training uses the instantiated objective, including its random stream and buffer updates.
Evaluation uses a reusable no-gradient evaluator over the same live parameter objects and does
not update running buffers.
-/
def sampleLossWithMode {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) (selectedMode : nn.Mode)
    (sample : TorchLean.Sample.Supervised α σ τ) : IO α := do
  let loss ← match selectedMode with
    | .train =>
      TorchLean.Module.Objective.loss (objectiveModule runner)
        (TorchLean.Sample.Internal.arguments sample) TorchLean.Arguments.empty
    | .eval =>
      Runtime.Autograd.Model.Module.Evaluator.run (evaluationEvaluator runner)
        (TorchLean.Arguments.Internal.toTensorPack (TorchLean.Sample.Internal.arguments sample))
        TorchLean.TensorPack.empty
  pure loss.item

/-- Scalar loss of one supervised sample using the active mode. -/
def sampleLoss {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) (sample : TorchLean.Sample.Supervised α σ τ) : IO α := do
  sampleLossWithMode runner (← mode runner) sample

/-- Mean scalar loss over a finite sample stream in the active mode; `0` for an empty stream. -/
def meanLoss {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (samples : TorchLean.Data.SampleStream (TorchLean.Sample.Supervised α σ τ)) : IO α := do
  if samples.isEmpty then
    pure 0
  else
    let mut total : α := 0
    for h : i in [0:samples.size] do
      have hi : i < samples.size := h.2.1
      total := total + (← sampleLoss runner (samples.get ⟨i, hi⟩))
    pure (total / (samples.size : α))

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
  if (Runner.runtime runner).usesCuda then
    IO.ofExcept config.validateFloat32
  match scheduler with
  | some schedule =>
      match TorchLean.Trainer.Scheduler.validate schedule with
      | .ok () => pure ()
      | .error message => throw <| IO.userError message
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
  match TorchLean.optim.Optimizer.Internal.view config with
  | .sgd learningRate momentum =>
      if momentum == 0.0 then
        let optimizer := Runtime.Autograd.Model.Optim.sgd (α := α) (paramShapes := shapes)
          (TorchLean.Runtime.ofFloat learningRate)
        let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
        continuation optimizer state fun step state =>
          mapStateList (fun shapeState =>
            { shapeState with
              learningRate := TorchLean.Runtime.ofFloat (learningRateAtStep step) }) state
      else
        let optimizer := Runtime.Autograd.Model.Optim.momentumSGD
          (α := α) (paramShapes := shapes)
          (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat momentum)
        let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
        continuation optimizer state fun step state =>
          mapStateList (fun shapeState =>
            { shapeState with
              learningRate := TorchLean.Runtime.ofFloat (learningRateAtStep step) }) state
  | .adaGrad learningRate epsilon =>
      let optimizer := Runtime.Autograd.Model.Optim.adagrad (α := α) (paramShapes := shapes)
        (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
      continuation optimizer state fun step state =>
        mapStateList (fun shapeState =>
          { shapeState with
            learningRate := TorchLean.Runtime.ofFloat (learningRateAtStep step) }) state
  | .rmsProp learningRate decay epsilon =>
      let optimizer := Runtime.Autograd.Model.Optim.rmsprop (α := α) (paramShapes := shapes)
        (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat decay)
        (TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
      continuation optimizer state fun step state =>
        mapStateList (fun shapeState =>
          { shapeState with
            learningRate := TorchLean.Runtime.ofFloat (learningRateAtStep step) }) state
  | .adam learningRate beta1 beta2 epsilon =>
      let optimizer := Runtime.Autograd.Model.Optim.adam (α := α) (paramShapes := shapes)
        (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat beta1)
        (TorchLean.Runtime.ofFloat beta2) (TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
      continuation optimizer state fun step state =>
        mapStateList (fun shapeState =>
          { shapeState with
            learningRate := TorchLean.Runtime.ofFloat (learningRateAtStep step) }) state
  | .adamW learningRate weightDecay beta1 beta2 epsilon =>
      let optimizer := Runtime.Autograd.Model.Optim.adamw (α := α) (paramShapes := shapes)
        (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat weightDecay)
        (TorchLean.Runtime.ofFloat beta1) (TorchLean.Runtime.ofFloat beta2)
        (TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
      continuation optimizer state fun step state =>
        mapStateList (fun shapeState =>
          { shapeState with
            learningRate := TorchLean.Runtime.ofFloat (learningRateAtStep step) }) state
  | .adaDelta learningRate rho epsilon =>
      let optimizer := Runtime.Autograd.Model.Optim.adadelta (α := α) (paramShapes := shapes)
        (TorchLean.Runtime.ofFloat learningRate) (TorchLean.Runtime.ofFloat rho)
        (TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.Objective.initOptimizer objective optimizer
      continuation optimizer state fun step state =>
        mapStateList (fun shapeState =>
          { shapeState with
            learningRate := TorchLean.Runtime.ofFloat (learningRateAtStep step) }) state

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
  | false =>
      exact do
        match batch[0]? with
        | none => throw <| IO.userError "Trainer.meanGrad: empty batch"
        | some firstSample =>
            let objective := Runner.objectiveModule runner
            let mut gradientSum ← TorchLean.Module.Objective.grad objective
              (TorchLean.Sample.Internal.arguments firstSample) TorchLean.Arguments.empty
            for sample in batch.drop 1 do
              let gradient ← TorchLean.Module.Objective.grad objective
                (TorchLean.Sample.Internal.arguments sample) TorchLean.Arguments.empty
              gradientSum := addGradients gradientSum gradient
            let reciprocalCount : α := 1 / (batch.size : α)
            pure (scaleGradients reciprocalCount gradientSum)
  | true =>
      exact do
        match batch[0]? with
        | none => throw <| IO.userError "Trainer.meanGrad: empty batch"
        | some firstSample =>
            let objective := Runner.objectiveModule runner
            let (firstGradient, firstLoss) ←
              TorchLean.Module.Objective.grad objective
                (TorchLean.Sample.Internal.arguments firstSample) TorchLean.Arguments.empty
                (value := true)
            let mut lossSum := TorchLean.Tensor.item firstLoss
            let mut gradientSum := firstGradient
            for sample in batch.drop 1 do
              let (gradient, lossValue) ←
                TorchLean.Module.Objective.grad objective
                  (TorchLean.Sample.Internal.arguments sample) TorchLean.Arguments.empty
                  (value := true)
              lossSum := lossSum + TorchLean.Tensor.item lossValue
              gradientSum := addGradients gradientSum gradient
            let reciprocalCount : α := 1 / (batch.size : α)
            pure
              (scaleGradients reciprocalCount gradientSum,
                lossSum * reciprocalCount)

/--
Apply one optimizer update to a nonempty batch.

When `useNative` is set, supported CUDA optimizers accumulate gradients on device and use the
same moment state for every batch size. Other optimizers average explicit per-sample gradients.
Set `loss := true` to return `(nextOptimizerState, meanLoss)`.
-/
def stepBatch {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (optimizer : Runtime.Autograd.Model.Optim.Optimizer α (TorchLean.nn.stateShapes model))
    (state : optimizer.State) (useNative : Bool)
    (batch : Array (TorchLean.Sample.Supervised α σ τ)) (loss : Bool := false) :
    IO (match loss with
      | false => optimizer.State
      | true => optimizer.State × α) := by
  cases loss with
  | false =>
      exact do
        if batch.isEmpty then
          throw <| IO.userError "Trainer.stepBatch: empty batch"
        let objective := Runner.objectiveModule runner
        if useNative then
          if hSingleton : batch.size = 1 then
            let sample := batch[0]'(by simp [hSingleton])
            return ← TorchLean.Module.Objective.step objective optimizer state
              (TorchLean.Sample.Internal.arguments sample) TorchLean.Arguments.empty
          let arguments := batch.map fun sample =>
            (TorchLean.Sample.Internal.arguments sample, TorchLean.Arguments.empty)
          if let some (nextState, _) ←
              TorchLean.Module.Objective.Internal.tryNativeBatchStep
                objective optimizer state arguments false then
            return nextState
        let meanGradient ← meanGrad runner batch
        TorchLean.Module.Objective.update objective optimizer state meanGradient
  | true =>
      exact do
        if batch.isEmpty then
          throw <| IO.userError "Trainer.stepBatch: empty batch"
        let objective := Runner.objectiveModule runner
        if useNative then
          if hSingleton : batch.size = 1 then
            let sample := batch[0]'(by simp [hSingleton])
            let (nextOptimizerState, lossValue) ←
              TorchLean.Module.Objective.step objective optimizer state
                (TorchLean.Sample.Internal.arguments sample) TorchLean.Arguments.empty
                (loss := true)
            return (nextOptimizerState, TorchLean.Tensor.item lossValue)
          let arguments := batch.map fun sample =>
            (TorchLean.Sample.Internal.arguments sample, TorchLean.Arguments.empty)
          if let some (nextState, lossValue) ←
              TorchLean.Module.Objective.Internal.tryNativeBatchStep
                objective optimizer state arguments true then
            match lossValue with
            | some lossValue => return (nextState, lossValue.item)
            | none =>
              throw <| IO.userError "Trainer.stepBatch: native update omitted requested loss"
        let (meanGradient, meanLoss) ←
          meanGrad runner batch (value := true)
        let nextOptimizerState ←
          TorchLean.Module.Objective.update objective optimizer state meanGradient
        pure (nextOptimizerState, meanLoss)

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

/-- Run one optimizer step and return its scalar loss. -/
def step {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model)
    (sample : TorchLean.Sample.Supervised α σ τ) : IO α :=
  action stepper #[sample]

/--
Run one averaged-gradient optimizer step over a nonempty batch and return its mean scalar loss.

Every sample is differentiated at the same parameter point. The completed-step counter advances
once for the whole batch.
-/
def stepBatch {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model)
    (batch : Array (TorchLean.Sample.Supervised α σ τ)) : IO α := do
  if batch.isEmpty then
    throw <| IO.userError "Stepper.stepBatch: batch must be nonempty"
  action stepper batch

/-- Run one optimizer step over a nonempty batch without reading the loss back. -/
def update {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model)
    (batch : Array (TorchLean.Sample.Supervised α σ τ)) : IO Unit := do
  if batch.isEmpty then
    throw <| IO.userError "Stepper.update: batch must be nonempty"
  silentAction stepper batch

/-- Read the number of completed optimizer steps. -/
def steps {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model) : IO Nat :=
  (counter stepper).get

end Stepper

/--
Construct a `Stepper` for a runner, optimizer config, and optional scheduler.

Every step switches the runner to training mode, applies the schedule for the current step index,
and advances the completed-step counter once per batch. A singleton batch uses the runtime's
native single-sample update.
-/
def Runner.stepper {σ τ : Spec.Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α] [TorchLean.Runtime.FromFloat α]
    (runner : Runner α model) (optimizer : TorchLean.optim.Optimizer)
    (scheduler : Option TorchLean.Trainer.Scheduler.Config := none) :
    IO (Stepper α model) := do
  Runner.train runner
  let stepRef ← IO.mkRef 0
  withBoundOptimizer runner optimizer scheduler
      fun runtimeOptimizer initialState scheduleState => do
    let stateRef ← IO.mkRef initialState
    let runBatch := fun (batch : Array (TorchLean.Sample.Supervised α σ τ)) => do
      Runner.train runner
      let stepIndex ← stepRef.get
      let state := scheduleState stepIndex (← stateRef.get)
      let (nextOptimizerState, lossValue) ←
        stepBatch runner runtimeOptimizer state true batch (loss := true)
      stateRef.set nextOptimizerState
      stepRef.set (stepIndex + 1)
      pure lossValue
    let runBatchSilently := fun (batch : Array (TorchLean.Sample.Supervised α σ τ)) => do
      Runner.train runner
      let stepIndex ← stepRef.get
      let state := scheduleState stepIndex (← stateRef.get)
      let nextOptimizerState ←
        stepBatch runner runtimeOptimizer state true batch
      stateRef.set nextOptimizerState
      stepRef.set (stepIndex + 1)
    pure (Stepper.create runBatch runBatchSilently stepRef)

end Internal
end Trainer
end TorchLean
