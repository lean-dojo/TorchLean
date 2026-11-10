/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Trainer
public import TorchLean.NN.BatchInput
public import TorchLean.NN.Trainer.Scheduler
public import TorchLean.NN.Data.SampleStream
public import TorchLean.Tensor.Operations

/-!
# Trainer Runtime Internals

Scalar-generic implementation of the trainer: an instantiated model with reusable forward and loss
evaluators (`Runner`), gradient accumulation and optimizer updates, and the stateful `Stepper` that
applies a configured optimizer and schedule.

Everything here is indexed by the runtime scalar `α`. `Trainer.openTyped` keeps that scalar at the
session boundary, while the paths whose scalar the runtime selects (`Trainer.open`,
`Trainer.train` and `Trainer.predict`) expose `Float`.
-/

@[expose] public section

namespace TorchLean
namespace Trainer

/--
A checked model instantiated under one runtime scalar.

A runner bundles the imperative runtime objective, whose parameters and buffers live in refs, with
reusable no-gradient forward and evaluation-loss evaluators over the same live state. Training
forwards update the buffers, and evaluation forwards leave them unchanged.
-/
structure Runner (α : Type) [TorchLean.Storage α] [Context α]
    {σ τ : Shape} (model : TorchLean.nn.Sequential σ τ) where
  private mk ::
  private runtimeObjective :
    TorchLean.Module.Objective α Unit (nn.Sequential.stateShapes model) [σ, τ]
  private trainingPredictor :
    Evaluator α Unit (nn.Sequential.stateShapes model) [σ] [] τ
  private evaluationPredictor :
    Evaluator α Unit (nn.Sequential.stateShapes model) [σ] [] τ
  private evaluationLossEvaluator :
    ObjectiveEvaluator α Unit
      (nn.Sequential.stateShapes model) [σ, τ]

namespace Runner

/-- Construct a runner from its objective and evaluators. -/
opaque create {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runtimeObjective :
      TorchLean.Module.Objective α Unit (nn.Sequential.stateShapes model) [σ, τ])
    (trainingPredictor :
      Evaluator α Unit (nn.Sequential.stateShapes model) [σ] [] τ)
    (evaluationPredictor :
      Evaluator α Unit (nn.Sequential.stateShapes model) [σ] [] τ)
    (evaluationLossEvaluator :
      ObjectiveEvaluator α Unit
        (nn.Sequential.stateShapes model) [σ, τ]) :
    Runner α model :=
  ⟨runtimeObjective, trainingPredictor, evaluationPredictor, evaluationLossEvaluator⟩

/-- The executable runtime module behind a runner. -/
opaque objectiveModule {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) :
    TorchLean.Module.Objective α Unit (nn.Sequential.stateShapes model) [σ, τ] :=
  match runner with
  | ⟨runtimeObjective, _, _, _⟩ => runtimeObjective

/-- The reusable no-gradient evaluator for training or evaluation behavior. -/
opaque predictor {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) (mode : nn.Mode) :
    Evaluator α Unit (nn.Sequential.stateShapes model) [σ] [] τ :=
  match runner, mode with
  | ⟨_, trainingPredictor, _, _⟩, .train => trainingPredictor
  | ⟨_, _, evaluationPredictor, _⟩, .eval => evaluationPredictor

/-- Reusable evaluator for the evaluation-mode loss. -/
opaque lossEvaluator {σ τ : Shape}
    {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) :
    ObjectiveEvaluator α Unit
      (nn.Sequential.stateShapes model) [σ, τ] :=
  match runner with
  | ⟨_, _, _, evaluator⟩ => evaluator

/-- Runtime configuration used by the runner's objective. -/
def runtime {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) : Session.Config :=
  (Module.Objective.runtime (objectiveModule runner)).runtime

/-- Finish runner construction once its executable objective has been instantiated. -/
def fromRuntimeObjective {σ τ : Shape}
    (model : TorchLean.nn.Sequential σ τ) (objective : Trainer.Objective τ)
    {α : Type} [TorchLean.Storage α] [Context α]
    [TensorTransfer α]
    (runtimeObjective :
      TorchLean.Module.Objective α Unit (nn.Sequential.stateShapes model) [σ, τ]) :
    IO (Runner α model) := do
  let executableObjective := Module.Objective.runtime runtimeObjective
  let makePredictor (mode : nn.Mode) :=
    Evaluator.create (β := Unit)
      (stateShapes := nn.Sequential.stateShapes model) (inputShapes := [σ])
      (dataInputShapes := [])
      (program := nn.Sequential.forward model mode (α := α))
      executableObjective.runtime executableObjective.trainer.state
  let trainingPredictor ← makePredictor .train
  let evaluationPredictor ← makePredictor .eval
  let evaluationLossEvaluator ←
    ObjectiveDef.evaluator
      (objective.definition model (mode := .eval))
      executableObjective.runtime executableObjective.trainer.state
  pure (create runtimeObjective trainingPredictor evaluationPredictor
    evaluationLossEvaluator)

/--
Instantiate a model and objective under a runtime scalar.

A given `initialState?` is used as is in `α`, and otherwise the stored `Float` initializers are
cast with `ofFloat`.
-/
def instantiate {σ τ : Shape} (model : TorchLean.nn.Sequential σ τ)
    (objective : Trainer.Objective τ)
    (options : Session.Config := {})
    (α : Type := Float32)
    [TorchLean.Storage α] [Context α]
    [FromFloat α]
    [TensorTransfer α]
    (initialState? : Option (TensorPack α (nn.Sequential.stateShapes model)) := none) :
    IO (Runner α model) := do
  let runtimeObjective ←
    TorchLean.Module.instantiate (α := α) (objective.definition model) options
      (initialState? := initialState?)
  fromRuntimeObjective model objective runtimeObjective

/-- Read the complete parameter-and-buffer state. -/
def state {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) :
    IO (TensorPack α (nn.Sequential.stateShapes model)) :=
  (objectiveModule runner).runtime.state

/-- Replace the complete parameter-and-buffer state. -/
def setState {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) (state : TensorPack α (nn.Sequential.stateShapes model)) :
    IO Unit :=
  (objectiveModule runner).runtime.loadState state

/-- Evaluate one input in the given mode (training by default). -/
def forward {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (input : Tensor α σ) (mode : nn.Mode := .train) : IO (Tensor α τ) :=
  Evaluator.run (predictor runner mode)
    (TorchLean.Pack.singleton input) TorchLean.Pack.nil

/--
Scalar loss of one sample in the given mode (training by default).

Training uses the objective's random stream and buffer updates. Evaluation uses a reusable
no-gradient evaluator over the live parameters without updating running buffers.
-/
def loss {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model) (sample : TorchLean.Sample.Supervised α σ τ)
    (mode : nn.Mode := .train) : IO α := do
  let value ← match mode with
    | .train =>
      (objectiveModule runner).runtime.loss
        (Sample.arguments sample) TorchLean.Pack.nil
    | .eval =>
      Evaluator.run (lossEvaluator runner)
        (Sample.arguments sample)
        TorchLean.Pack.nil
  pure value.item

end Runner

/-! ## Optimizer binding -/

/-- Bind a configured optimizer, its initialized state, and its scheduled state updater. -/
def Runner.withBoundOptimizer {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α β : Type} [TorchLean.Storage α] [Context α]
    [FromFloat α]
    (runner : Runner α model) (config : TorchLean.optim.Optimizer)
    (scheduler : Option TorchLean.Trainer.Scheduler.Config)
    (continuation :
      (optimizer : Optimizer α (nn.Sequential.stateShapes model)) →
      optimizer.State → (Nat → optimizer.State → optimizer.State) → IO β) : IO β := do
  match scheduler with
  | some schedule => do
      IO.ofExcept <| schedule.validate
        (round := FromFloat.roundForValidation (α := α))
  | none => pure ()
  let rate (step : Nat) : Float :=
    scheduler.map (fun config => config.rate step)
      |>.getD config.lr
  Module.withConfiguredOptimizer (α := α)
    (stateShapes := nn.Sequential.stateShapes model) (Runner.runtime runner) config
    fun optimizer => do
      let state ← (Runner.objectiveModule runner).runtime.initOptimizer optimizer
      continuation optimizer state fun step state =>
        optimizer.setLr (ofFloat (rate step)) state

/-! ## Sample means -/

/-- Running totals for the mean of `count` per-sample values: their sum in sample order, the sum
of the values each divided by `count`, and their IEEE minimum and maximum. -/
structure Mean (α : Type) where
  /-- The sum of the values. -/
  sum : α
  /-- The sum of the values, each divided by the count. -/
  divided : α
  /-- The smallest value. -/
  low : α
  /-- The largest value. -/
  high : α

namespace Mean

/-- Whether a mean uses the divided sum and the range: IEEE scalars without derivative
coefficients. -/
def bounded (α : Type) [Context α] : Bool :=
  TorchLean.Numeric.QuotientArithmetic.supported (α := α) &&
    decide (TorchLean.Numeric.QuotientArithmetic.depth (α := α) = 0)

variable {α : Type} [Context α]

/-- The totals of a first value. -/
def single (count : Nat) (value : α) : Mean α :=
  ⟨value, Context.divNat value count, value, value⟩

/-- The totals with one more value. -/
def push (count : Nat) (totals : Mean α) (value : α) : Mean α :=
  ⟨totals.sum + value, totals.divided + Context.divNat value count,
    Context.minimum totals.low value, Context.maximum totals.high value⟩

/-- The mean of `count` values from their totals: the sum divided once by `count`
(`Context.divNat`). For an IEEE scalar, the divided sum replaces a non-finite sum whenever the
divided sum is finite, and a mean that is not a NaN is clamped to the range of the values. The
divided sum keeps the mean of equal large values finite, though cancellation between signs can
lose precision. -/
def value (count : Nat) (totals : Mean α) : α :=
  if bounded α then
    let mean := if !Context.isFinite totals.sum && Context.isFinite totals.divided then
      totals.divided else Context.divNat totals.sum count
    if Context.isNaN mean then mean
    else Context.minimum totals.high (Context.maximum totals.low mean)
  else Context.divNat totals.sum count

end Mean

/-! ## Gradient accumulation and optimizer updates -/

/--
Mean parameter gradient and mean loss for a nonempty batch at one parameter point.

With `withLoss := false` the losses are not read back and the second component is `0`. The loss
and every gradient coordinate are the `Mean.value` of their per-sample values, so the batch size is
never rounded into `α`.
-/
def Runner.meanGradAndLoss {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (batch : Array (TorchLean.Sample.Supervised α σ τ)) (withLoss : Bool) :
    IO (TensorPack α (nn.Sequential.stateShapes model) × α) := do
  match batch[0]? with
  | none => throw <| IO.userError "Trainer.Runner.meanGradAndLoss: empty batch"
  | some firstSample =>
      let objective := Runner.objectiveModule runner
      let sampleGradient (sample : TorchLean.Sample.Supervised α σ τ) :
          IO (TensorPack α (nn.Sequential.stateShapes model) × α) := do
        let arguments := Sample.arguments sample
        if withLoss then
          let (gradient, lossValue) ← objective.runtime.grad
            arguments TorchLean.Pack.nil (value := true)
          pure (gradient, TorchLean.Tensor.item lossValue)
        else
          pure (← objective.runtime.grad arguments TorchLean.Pack.nil, 0)
      let (firstGradient, firstLoss) ← sampleGradient firstSample
      let count := batch.size
      let bounded := Mean.bounded α
      let mut gradientSum := firstGradient
      let mut gradientDivided := if bounded then firstGradient.map (fun tensor =>
        Tensor.map (Context.divNat · count) tensor) else firstGradient
      let mut gradientMin := firstGradient
      let mut gradientMax := firstGradient
      let mut loss := Mean.single count firstLoss
      for sample in batch.drop 1 do
        let (gradient, lossValue) ← sampleGradient sample
        loss := loss.push count lossValue
        gradientSum := gradientSum.zipWith Tensor.add gradient
        if bounded then
          gradientDivided := gradientDivided.zipWith (fun total value =>
            Tensor.zipWith (fun total value => total + Context.divNat value count) total value)
            gradient
          gradientMin := gradientMin.zipWith (Tensor.minimum · ·) gradient
          gradientMax := gradientMax.zipWith (Tensor.maximum · ·) gradient
      let sums := gradientSum.zipWith (Tensor.zipWith Prod.mk) gradientDivided
      let ranges := gradientMin.zipWith (Tensor.zipWith Prod.mk) gradientMax
      let meanGradient := sums.zipWith (Tensor.zipWith fun (sum, divided) (low, high) =>
        Mean.value count ⟨sum, divided, low, high⟩) ranges
      pure (meanGradient, loss.value count)

/--
Apply one optimizer update to a nonempty batch.

The backend's native update is tried first: supported CUDA optimizers accumulate gradients on
device and use the same moment state for every batch size. Other optimizers fall back to the mean
of the per-sample gradients. Set `loss := true` to return `(nextOptimizerState, meanLoss)`.
-/
def Runner.step {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runner : Runner α model)
    (optimizer : Optimizer α (nn.Sequential.stateShapes model))
    (state : optimizer.State)
    (batch : Array (TorchLean.Sample.Supervised α σ τ)) (loss : Bool := false) :
    IO (match loss with
      | false => optimizer.State
      | true => optimizer.State × α) := by
  -- Without `withLoss` the loss is never read back and the second component is `0`.
  let run (withLoss : Bool) : IO (optimizer.State × α) := do
    if batch.isEmpty then
      throw <| IO.userError "Trainer.Runner.step: empty batch"
    let objective := Runner.objectiveModule runner
    if hSingleton : batch.size = 1 then
      let sample := batch[0]'(by simp [hSingleton])
      let arguments := Sample.arguments sample
      if withLoss then
        let (nextOptimizerState, lossValue) ←
          objective.runtime.step optimizer state
            arguments TorchLean.Pack.nil (loss := true)
        return (nextOptimizerState, TorchLean.Tensor.item lossValue)
      else
        return (← objective.runtime.step optimizer state
          arguments TorchLean.Pack.nil, 0)
    let arguments := batch.map fun sample =>
      (Sample.arguments sample, TorchLean.Pack.nil)
    if let some (nextState, lossValue) ←
        Module.Objective.tryNativeBatchStep
          objective optimizer state arguments withLoss then
      if withLoss then
        match lossValue with
        | some lossValue => return (nextState, lossValue.item)
        | none =>
          throw <| IO.userError "Trainer.Runner.step: native update omitted requested loss"
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
    {σ τ : Shape} (model : TorchLean.nn.Sequential σ τ) where
  private mk ::
  private runBatch : Array (TorchLean.Sample.Supervised α σ τ) → IO α
  private runBatchSilently : Array (TorchLean.Sample.Supervised α σ τ) → IO Unit
  private stepRef : IO.Ref Nat

namespace Stepper

/-- Construct a stepper from its hidden batch-step functions and counter. -/
opaque create {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (runBatch : Array (TorchLean.Sample.Supervised α σ τ) → IO α)
    (runBatchSilently : Array (TorchLean.Sample.Supervised α σ τ) → IO Unit)
    (stepRef : IO.Ref Nat) : Stepper α model :=
  ⟨runBatch, runBatchSilently, stepRef⟩

/-- The hidden loss-returning batch-step function. -/
opaque action {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model) :
    Array (TorchLean.Sample.Supervised α σ τ) → IO α :=
  match stepper with
  | ⟨runBatch, _, _⟩ => runBatch

/-- The hidden batch-step function that does not read the loss back. -/
opaque silentAction {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model) :
    Array (TorchLean.Sample.Supervised α σ τ) → IO Unit :=
  match stepper with
  | ⟨_, runBatchSilently, _⟩ => runBatchSilently

/-- The hidden completed-step counter. -/
opaque counter {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model) : IO.Ref Nat :=
  match stepper with
  | ⟨_, _, stepRef⟩ => stepRef

/--
Apply one optimizer update, optionally returning its scalar loss.

`batch := true` takes a nonempty array and averages gradients at the same parameter point. The
counter advances once for the whole batch. `loss := false` avoids reading the loss back.
-/
def step {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ} {Input : Type}
    {α : Type} [TorchLean.Storage α] [Context α]
    (stepper : Stepper α model) (sample : Input) (batch : Bool := false) (loss : Bool := false)
    [BatchInput (TorchLean.Sample.Supervised α σ τ)
      (Array (TorchLean.Sample.Supervised α σ τ)) batch Input] :
    IO (match loss with | false => Unit | true => α) := by
  have inputType := BatchInput.type_eq
    (single := TorchLean.Sample.Supervised α σ τ)
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
def steps {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
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
def Runner.stepper {σ τ : Shape} {model : TorchLean.nn.Sequential σ τ}
    {α : Type} [TorchLean.Storage α] [Context α] [FromFloat α]
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
        step runner runtimeOptimizer state batch (loss := true)
      stateRef.set nextOptimizerState
      stepRef.set (stepIndex + 1)
      pure lossValue
    let runBatchSilently := fun (batch : Array (TorchLean.Sample.Supervised α σ τ)) => do
      let stepIndex ← stepRef.get
      let state := scheduleState stepIndex (← stateRef.get)
      let nextOptimizerState ←
        step runner runtimeOptimizer state batch
      stateRef.set nextOptimizerState
      stepRef.set (stepIndex + 1)
    pure (Stepper.create runBatch runBatchSilently stepRef)

end Trainer
end TorchLean
