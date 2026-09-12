/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Training
public import NN.API.Trainer.Constructor
public import NN.API.Trainer.Runner
public import NN.API.Trainer.Train

/-!
# Gradient Accumulation

These checks cover mean-gradient updates, native-hook selection, and partial batches on the
trainer's internal runner. The probe optimizer records which route each update used without
coupling the test to a particular backend implementation.
-/

@[expose] public section

namespace NN.Tests.API.GradientAccumulation

open TorchLean.Trainer.Internal

def vector1 (x : Float) : TorchLean.Tensor Float [1] :=
  TorchLean.Tensor.ofFn fun _ => x

def vector1Value (x : TorchLean.Tensor Float [1]) : Float :=
  TorchLean.Tensor.item (TorchLean.Tensor.get x (0 : Fin 1))

def model : TorchLean.nn.Sequential [1] [1] :=
  Runtime.Autograd.Model.Layers.Seq.fromLayer <|
    Runtime.Autograd.Model.Layers.linear 1 1 17

def objective : TorchLean.Trainer.Objective [1] := .meanSquaredError

/-- Concrete state layout of the single affine layer used by this test. -/
theorem taskStateShapes :
    TorchLean.nn.stateShapes model = [([1, 1] : Spec.Shape), ([1] : Spec.Shape)] := by
  rfl

def sample (x y : Float) :
    TorchLean.Sample.Supervised Float [1] [1] :=
  { input := vector1 x
    target := vector1 y }

def dropoutModel : TorchLean.nn.Sequential [1] [1] :=
  TorchLean.nn.build 23 (TorchLean.nn.dropout (shape := [1]) 1.0)

def dropoutSample : TorchLean.Sample.Supervised Float [1] [1] :=
  { input := vector1 2.0
    target := vector1 0.0 }

def readLinearParams
    (state : TorchLean.nn.State Float (TorchLean.nn.stateShapes model)) :
    Float × Float :=
  let state := state.cast taskStateShapes
  let weight := state.get 0
  let bias := state.get 1
  ( TorchLean.Tensor.item <|
      TorchLean.Tensor.get
        (TorchLean.Tensor.get weight (0 : Fin 1)) (0 : Fin 1)
  , TorchLean.Tensor.item (TorchLean.Tensor.get bias (0 : Fin 1)) )

def close (x y : Float) : Bool :=
  Float.abs (x - y) ≤ 1e-5

/-- One-channel image used by the BatchNorm buffer regression. -/
def constantImage (value : Float) :
    TorchLean.Tensor Float [1, 1, 2] :=
  TorchLean.Tensor.generate [1, 1, 2] fun _ => value

def batchNormModel :
    Runtime.Autograd.Model.Layers.Seq
      [1, 1, 2]
      [1, 1, 2] :=
  Runtime.Autograd.Model.Layers.Seq.fromLayer <|
    Runtime.Autograd.Model.Layers.batchNorm 1 1 [2]
      (by decide) (momentum := 0.5)

/-- Concrete parameter-and-buffer layout of the BatchNorm layer used by this test. -/
theorem batchNormStateShapes :
    TorchLean.nn.stateShapes batchNormModel =
      [ ([1] : Spec.Shape)
      , ([1] : Spec.Shape)
      , ([1] : Spec.Shape)
      , ([1] : Spec.Shape)
      , ([] : Spec.Shape) ] := by
  rfl

def batchNormSample (value : Float) :
    TorchLean.Sample.Supervised Float [1, 1, 2] [1, 1, 2] :=
  { input := constantImage value
    target := constantImage 0.0 }

def readBatchNormBuffers
    (state : TorchLean.nn.State Float (TorchLean.nn.stateShapes batchNormModel)) :
    Float × Float :=
  let state := state.cast batchNormStateShapes
  let mean := state.get 2
  let variance := state.get 3
  ( TorchLean.Tensor.item (TorchLean.Tensor.get mean (0 : Fin 1))
  , TorchLean.Tensor.item (TorchLean.Tensor.get variance (0 : Fin 1)) )

def noOpOptimizer (shapes : List Spec.Shape) :
    Runtime.Autograd.Model.Optim.Optimizer Float shapes where
  State := Unit
  init := fun _ => pure ()
  step := fun _ _ _ => pure ()

/-- Counters shared by the probe optimizer and the assertions that follow a run. -/
structure ProbeCounters where
  nativeSteps : IO.Ref Nat
  lossSteps : IO.Ref Nat
  genericSteps : IO.Ref Nat

def newProbeCounters : IO ProbeCounters := do
  pure {
    nativeSteps := ← IO.mkRef 0
    lossSteps := ← IO.mkRef 0
    genericSteps := ← IO.mkRef 0
  }

/--
Optimizer that records native, loss-returning, and generic dispatch separately.

The no-loss hook completes the update itself. The loss hook returns `none` after recording the
attempt, allowing the normal same-tape fallback to produce the scalar when a caller asks for it.
-/
def probeOptimizer (counters : ProbeCounters) :
    Runtime.Autograd.Model.Optim.Optimizer Float (TorchLean.nn.stateShapes model) where
  State := Unit
  init := fun _ => pure ()
  step := fun _ _ _ => do
    counters.genericSteps.modify (· + 1)
  trainerStep? := fun _ _ _ _ => do
    counters.nativeSteps.modify (· + 1)
    pure (some ())
  trainerStepWithLoss? := fun _ _ _ _ => do
    counters.lossSteps.modify (· + 1)
    pure none

def expectNat (label : String) (actual expected : Nat) : IO Unit :=
  unless actual == expected do
    throw <| IO.userError s!"{label}: got {actual}, expected {expected}"

def expectFailure {α : Type} (label : String) (action : IO α) : IO Unit := do
  let failed ← try
    let _ ← action
    pure false
  catch _ =>
    pure true
  unless failed do
    throw <| IO.userError s!"{label}: expected failure"

def expectAccepted (label : String) (result : Except String Unit) : IO Unit :=
  match result with
  | .ok () => pure ()
  | .error message =>
      throw <| IO.userError s!"{label}: unexpectedly rejected: {message}"

def expectRejected (label : String) (result : Except String Unit) : IO Unit :=
  match result with
  | .error _ => pure ()
  | .ok () =>
      throw <| IO.userError s!"{label}: unexpectedly accepted"

def newRunner : IO (Runner Float model) :=
  Runner.instantiate model objective { execution := .typedGraph } (α := Float)

/--
Evaluation helpers select `.eval` for one call without mutating the mode used by subsequent
training work. Full-probability dropout makes the train/eval distinction deterministic.
-/
def checkEvaluationModeIsolation : IO Unit := do
  let runner ← Runner.instantiate dropoutModel .meanSquaredError
    { execution := .typedGraph } (α := Float)
  runner.train
  let trainingPrediction ← runner.forward dropoutSample.input
  let evaluationPrediction ← runner.predict dropoutSample.input
  unless close (vector1Value trainingPrediction) 0.0 do
    throw <| IO.userError "training-mode dropout did not zero its input"
  unless close (vector1Value evaluationPrediction) 2.0 do
    throw <| IO.userError "evaluation-mode dropout was not the identity"
  unless (← runner.mode) == .train do
    throw <| IO.userError "evaluation prediction changed the runner's training mode"
  let trainingLoss ← runner.sampleLoss dropoutSample
  let evaluationLoss ← runner.sampleLossWithMode .eval dropoutSample
  unless close trainingLoss 0.0 && close evaluationLoss 4.0 do
    throw <| IO.userError <|
      s!"dropout loss mode mismatch: training={trainingLoss}, evaluation={evaluationLoss}"
  unless (← runner.mode) == .train do
    throw <| IO.userError "evaluation loss changed the runner's training mode"

/-- A two-sample batch step applies the closed-form mean gradient once. -/
def checkClosedFormMeanGradient : IO Unit := do
  let runner ← newRunner
  let (weight, bias) := readLinearParams (← runner.state)
  let x₁ := 1.0
  let y₁ := 0.0
  let x₂ := 3.0
  let y₂ := 1.0
  let residual₁ := weight * x₁ + bias - y₁
  let residual₂ := weight * x₂ + bias - y₂
  let gradWeight := (2.0 * residual₁ * x₁ + 2.0 * residual₂ * x₂) / 2.0
  let gradBias := (2.0 * residual₁ + 2.0 * residual₂) / 2.0
  let lr := 0.1
  let stepper ← runner.stepper (TorchLean.optim.sgd { learningRate := lr })
  let _ ← stepper.stepBatch #[sample x₁ y₁, sample x₂ y₂]
  let (weight', bias') := readLinearParams (← runner.state)
  unless close weight' (weight - lr * gradWeight) && close bias' (bias - lr * gradBias) do
    throw <| IO.userError <|
      s!"minibatch update mismatch: got ({weight'}, {bias'}), expected "
        ++ s!"({weight - lr * gradWeight}, {bias - lr * gradBias})"

/-- The public Transformer schedule reaches its peak, midpoint, and floor at the stated updates. -/
def checkWarmupCosineSchedule : IO Unit := do
  let schedule := TorchLean.Trainer.Scheduler.warmupCosine 1.0 0.1 2 6
  let observed :=
    [ TorchLean.Trainer.Scheduler.learningRateAt schedule 0
    , TorchLean.Trainer.Scheduler.learningRateAt schedule 1
    , TorchLean.Trainer.Scheduler.learningRateAt schedule 2
    , TorchLean.Trainer.Scheduler.learningRateAt schedule 4
    , TorchLean.Trainer.Scheduler.learningRateAt schedule 6
    ]
  let expected := [0.5, 1.0, 1.0, 0.55, 0.1]
  unless List.all (List.zipWith close observed expected) id do
    throw <| IO.userError
      s!"warmup/cosine schedule mismatch: got {observed}, expected {expected}"
  let clamped := TorchLean.Trainer.Scheduler.warmupCosine 1.0 0.1 10 4
  let clampedObserved :=
    [ TorchLean.Trainer.Scheduler.learningRateAt clamped 0
    , TorchLean.Trainer.Scheduler.learningRateAt clamped 3
    , TorchLean.Trainer.Scheduler.learningRateAt clamped 4
    ]
  let clampedExpected := [0.25, 1.0, 0.1]
  unless List.all (List.zipWith close clampedObserved clampedExpected) id do
    throw <| IO.userError <|
      s!"clamped warm-up mismatch: got {clampedObserved}, expected {clampedExpected}"
  let empty := TorchLean.Trainer.Scheduler.warmupCosine 1.0 0.1 0 0
  unless close (TorchLean.Trainer.Scheduler.learningRateAt empty 0) 0.1 do
    throw <| IO.userError "zero-step warmup/cosine schedule did not remain at its floor"

/-- Schedules reject invalid numerical domains before optimizer state is allocated. -/
def checkSchedulerValidation : IO Unit := do
  let validate := TorchLean.Trainer.Scheduler.validate
  expectAccepted "constant scheduler" <| validate (.constant 0.1)
  expectAccepted "step scheduler" <| validate (.step 0.1 2 0.5)
  expectAccepted "exponential scheduler" <| validate (.exponential 0.1 0.9)
  expectAccepted "warmup scheduler" <| validate (.warmupCosine 0.1 0.01 2 10)
  expectRejected "negative learning rate" <| validate (.constant (-0.1))
  expectRejected "NaN learning rate" <| validate (.constant (0.0 / 0.0))
  expectRejected "infinite learning rate" <| validate (.constant (1.0 / 0.0))
  expectRejected "zero step size" <| validate (.step 0.1 0 0.5)
  expectRejected "negative decay" <| validate (.step 0.1 2 (-0.1))
  expectRejected "step decay above one" <| validate (.step 0.1 2 1.1)
  expectRejected "exponential decay above one" <| validate (.exponential 0.1 1.1)
  expectRejected "warmup minimum above peak" <| validate (.warmupCosine 0.1 0.2 2 10)

/-- Steppers reject empty batches and count one update per nonempty batch. -/
def checkStepperBatchBoundary : IO Unit := do
  let runner ← newRunner
  let stepper ← runner.stepper
    (TorchLean.optim.sgd { learningRate := 0.01 })
  let _ ← stepper.stepBatch #[sample 1.0 0.0, sample 2.0 1.0]
  expectNat "batched step count" (← stepper.steps) 1
  expectFailure "empty stepper batch" <| stepper.stepBatch #[]
  expectFailure "empty silent batch" <| stepper.update #[]
  expectNat "rejected batch must not advance the counter" (← stepper.steps) 1
  stepper.update #[sample 3.0 1.0]
  expectNat "silent update advances the counter" (← stepper.steps) 2

/-- Invalid training configurations fail before a sample or optimizer update is consumed. -/
def checkTrainingValidation : IO Unit := do
  let trainer := TorchLean.Trainer.new model
    { optimizer := TorchLean.optim.sgd { learningRate := 0.01 } }
  let data := TorchLean.Data.fromSamples #[sample 1.0 0.0]
  expectFailure "zero samples per step" <|
    trainer.train data { steps := 1, samplesPerStep := 0 }

  let optimizerRunner ← newRunner
  expectFailure "invalid optimizer" <|
    optimizerRunner.stepper
      (TorchLean.optim.sgd { learningRate := -0.01 })

  let schedulerRunner ← newRunner
  expectFailure "invalid scheduler" <|
    schedulerRunner.stepper
      (TorchLean.optim.sgd { learningRate := 0.01 })
      (some (.step 0.01 0))

  let invalidTrainer := TorchLean.Trainer.new model
    { optimizer := TorchLean.optim.sgd { learningRate := -0.01 } }
  expectFailure "session rejects an invalid optimizer" invalidTrainer.open

/-- A singleton batch keeps the native no-loss route unless the loss is requested. -/
def checkNoLossFastPath : IO Unit := do
  let runner ← newRunner
  let counters ← newProbeCounters
  let opt := probeOptimizer counters
  let state ← runner.initOptimizer opt
  runner.train
  let state ← stepBatch runner opt state true #[sample 1.0 0.0]
  expectNat "native no-loss steps" (← counters.nativeSteps.get) 1
  expectNat "loss-returning steps" (← counters.lossSteps.get) 0
  expectNat "generic steps" (← counters.genericSteps.get) 0
  let _ ← stepBatch runner opt state true #[sample 1.0 0.0] (loss := true)
  expectNat "loss-returning attempt" (← counters.lossSteps.get) 1
  expectNat "native no-loss steps after loss request" (← counters.nativeSteps.get) 1

/--
Multi-sample batches keep every update on the generic optimizer state, including a trailing
singleton batch when the native singleton route is disabled.
-/
def checkPartialBatchStateRoute : IO Unit := do
  let runner ← newRunner
  let counters ← newProbeCounters
  let opt := probeOptimizer counters
  let state ← runner.initOptimizer opt
  runner.train
  let state ← stepBatch runner opt state false #[sample 1.0 0.0, sample 2.0 0.0]
  let _ ← stepBatch runner opt state false #[sample 3.0 0.0]
  expectNat "partial-batch native steps" (← counters.nativeSteps.get) 0
  expectNat "partial-batch generic steps" (← counters.genericSteps.get) 2

/--
BatchNorm buffers advance once per accumulated item, and requesting a logged loss does not apply a
second buffer update.
-/
def checkBatchNormBuffers : IO Unit := do
  let batch := #[batchNormSample 2.0, batchNormSample 4.0]
  let runner ← Runner.instantiate batchNormModel .meanSquaredError
    { execution := .typedGraph } (α := Float)
  runner.train
  let opt := noOpOptimizer (TorchLean.nn.stateShapes batchNormModel)
  let state ← runner.initOptimizer opt
  let _ ← stepBatch runner opt state false batch
  let noLossBuffers := readBatchNormBuffers (← runner.state)

  let loggedRunner ← Runner.instantiate batchNormModel .meanSquaredError
    { execution := .typedGraph } (α := Float)
  loggedRunner.train
  let loggedState ← loggedRunner.initOptimizer opt
  let _ ← stepBatch loggedRunner opt loggedState false batch (loss := true)
  let loggedBuffers := readBatchNormBuffers (← loggedRunner.state)

  unless close noLossBuffers.1 2.5 && close noLossBuffers.2 0.25 do
    throw <| IO.userError <|
      s!"BatchNorm buffers: got mean={noLossBuffers.1}, variance={noLossBuffers.2}; "
        ++ "expected mean=2.5, variance=0.25"
  unless close loggedBuffers.1 noLossBuffers.1 && close loggedBuffers.2 noLossBuffers.2 do
    throw <| IO.userError <|
      s!"BatchNorm logging changed buffer updates: no-loss={noLossBuffers}, logged={loggedBuffers}"

def run : IO Unit := do
  checkEvaluationModeIsolation
  checkClosedFormMeanGradient
  checkWarmupCosineSchedule
  checkSchedulerValidation
  checkStepperBatchBoundary
  checkTrainingValidation
  checkNoLossFastPath
  checkPartialBatchStateRoute
  checkBatchNormBuffers

end NN.Tests.API.GradientAccumulation
