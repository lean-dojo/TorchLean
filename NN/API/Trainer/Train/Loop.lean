/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Memory
public import NN.API.Trainer.Session

/-!
# Prediction and Training

Prediction, dataset training, checkpoint restoration, and stream training on `TorchLean.Trainer`.

Every method here opens a `Session`, drives it, and finishes it. Programs that need a different
loop use `trainer.open` directly.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Internal

/-- Take the next cyclic group from a finite stream, evaluating only the requested samples. -/
def nextCyclicBatch {α : Type} (context : String)
    (samples : TorchLean.Data.SampleStream α)
    (cursorRef : IO.Ref Nat) (batchSize : Nat) : IO (Array α) := do
  if hEmpty : samples.size = 0 then
    throw <| IO.userError s!"{context}: empty sample stream"
  else
    have hSize : 0 < samples.size := Nat.pos_of_ne_zero hEmpty
    let mut cursor ← cursorRef.get
    let mut batch : Array α := #[]
    for _ in [0:batchSize] do
      let i := cursor % samples.size
      batch := batch.push (samples.get ⟨i, Nat.mod_lt _ hSize⟩)
      cursor := cursor + 1
    cursorRef.set cursor
    pure batch

/-- Print the predictions of every probe under a title. -/
def printProbes {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer) (probes : Array (Probe σ)) (title : String) : IO Unit := do
  unless probes.isEmpty do
    IO.println title
    for probe in probes do
      let prediction ← session.predict probe.input
      let expected :=
        match probe.expected with
        | some value => s!"  target={value}"
        | none => ""
      let inputText := if probe.inputText.isEmpty then "" else s!" {probe.inputText}"
      IO.println s!"  {probe.name}:{inputText}{expected}  prediction={Spec.pretty prediction}"

/-- Restore the optional checkpoint named by the training options. -/
def loadCheckpoint {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer) (path? : Option System.FilePath) : IO Unit := do
  match path? with
  | none => pure ()
  | some path => session.load path

/-- Write the optional checkpoint named by the training options. -/
def saveCheckpoint {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer) (path? : Option System.FilePath) : IO Unit := do
  match path? with
  | none => pure ()
  | some path =>
      session.save path
      IO.println s!"  wrote checkpoint: {path}"

/-- Reject training options that cannot describe a run. -/
def validateOptions (options : TrainOptions) : IO Unit :=
  match options.validate with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message

/--
Run the requested number of optimizer updates over a finite sample stream.

Batches cycle through the stream. Steps whose loss is logged read it back; every other step
updates without a host readback. CUDA allocator state is sampled at the configured cadence.
-/
def runSteps {σ τ : Shape} {trainer : TorchLean.Trainer σ τ}
    (session : Session trainer) (options : TrainOptions)
    (samples : Data.SampleStream (Sample.Supervised Float σ τ)) : IO Unit := do
  let runtime := trainer.runtime.executionSettings
  let watchEvery := Memory.cadence runtime options.steps options.cudaMemorySampleEvery
  let mut memorySample? ← Memory.sample runtime watchEvery options.steps 0 none
  unless samples.isEmpty do
    let cursorRef ← IO.mkRef 0
    for stepIndex in [0:options.steps] do
      let batch ←
        nextCyclicBatch "Trainer.train" samples cursorRef options.samplesPerStep
      let completedSteps := stepIndex + 1
      if Training.shouldReport options.logEvery completedSteps then
        let lossValue ← session.step (batch := true) batch (loss := true)
        IO.println s!"step {completedSteps}: loss={lossValue}"
      else
        session.step (batch := true) batch
      memorySample? ←
        Memory.sample runtime watchEvery options.steps completedSteps memorySample?

end Internal

/--
Predict one input using the trainer's current model and runtime settings.

Inference before any training call. After training, use the returned trained result's
`trained.predict` method to predict with the trained parameters.

Example:
```lean
-- This is inference with the freshly initialized parameters, which is what makes it a useful
-- baseline to print before training starts.
def baseline (trainer : TorchLean.Trainer [2] [1]) : IO (Tensor Float [1]) :=
  trainer.predict ([0.25, -0.75] : Tensor Float [2])
```
-/
def predict {σ τ inputShape : Shape}
    (trainer : TorchLean.Trainer σ τ) (input : Tensor Float inputShape)
    (batch : Bool := false) (batchSize : Nat := 1)
    [Trainer.Internal.BatchInput
      (Tensor Float σ) (Tensor Float (σ.prependDim batchSize)) batch (Tensor Float inputShape)] :
    IO (Tensor Float (match batch with | false => τ | true => τ.prependDim batchSize)) := do
  let session ← trainer.open
  session.predict input (batch := batch) (batchSize := batchSize)

/--
Train the model with the loss and runtime settings stored in `trainer`.

The signature uses `Tensor Float`, but the run itself executes in binary32: `.native` arithmetic
instantiates the model over `Float32` and `.ieee` over `ExecFloat.Binary 8 23`. Dataset samples are
converted
into that scalar as they are used, and results are read back to `Float`.

The result stores the trained parameters together with prediction, reporting, state access, and
verification methods. `trained.verify center (radius := r) (norm := .inf)` checks the model that
was actually trained. Add `(property := .topLabel label)` when the verification goal is a class
label. `trained.save path` writes the parameters; `trainer.load path data` restores them.

A positive step count requires a nonempty materialized dataset.

This is `trainer.open`, a loop of `step`, and `finish`; the reported losses are the evaluation-mode
mean losses over `data` before and after the loop.

Example:
```lean
-- `trained` owns the parameters the run produced; `trainer` still describes the untrained model.
def run (trainer : TorchLean.Trainer [2] [1])
    (data : Trainer.Dataset [2] [1]) : IO Unit := do
  let trained ← trainer.train data { steps := 200, logEvery := 25 }
  trained.printSummary
  let prediction ← trained.predict ([0.25, -0.75] : Tensor Float [2])
  IO.println s!"trained(heldout) = {reprStr prediction}"
```
-/
def train {σ τ : Shape}
    (trainer : TorchLean.Trainer σ τ)
    (data : Dataset σ τ) (trainOptions : TrainOptions)
    (probes : Array (Probe σ) := #[]) : IO (Result σ τ) := do
  Internal.validateOptions trainOptions
  let samples ← data.materialize (α := Float)
  if trainOptions.steps > 0 && samples.isEmpty then
    throw <| IO.userError
      "Trainer.train: no training samples; check the dataset and batch size (including drop_last)"
  let session ← trainer.open trainOptions.scheduler
  Internal.loadCheckpoint session trainOptions.loadCheckpoint?
  IO.println s!"dataset size = {samples.size}"
  let before ← session.loss samples (batch := true)
  IO.println s!"mean_loss(before training) = {before}"
  Internal.printProbes session probes "predictions before training"
  Internal.runSteps session trainOptions samples
  let after ← session.loss samples (batch := true)
  IO.println s!"mean_loss(after training) = {after}"
  Internal.printProbes session probes "predictions after training"
  Internal.saveCheckpoint session trainOptions.saveCheckpoint?
  let result ← session.finish { before, after }
  result.report.writeLog
    trainOptions.logDestination trainOptions.logTitle trainOptions.logNotes
  pure result

/--
Restore a checkpoint written by `Result.save` or `Checkpoint.State.save` and return it as a trained
result for this trainer's model, objective, and runtime settings.

The stored `Float` state is cast into the run's binary32 scalar, exactly when it came from
`Result.save`. `data` is evaluated once so that the result carries an honest report: `steps` is
`0` and both losses equal the mean loss of the restored parameters on `data`. To continue training
instead, pass `loadCheckpoint? := some path` in `TrainOptions`; the optimizer and schedule start
fresh at step zero.

Example:
```lean
-- Reads back what `trained.save` wrote and re-measures the report on `data`, so both reported
-- losses are the restored model's loss and the step count is `0`.
def restore (trainer : TorchLean.Trainer [2] [1])
    (data : Trainer.Dataset [2] [1]) : IO (Trainer.Result [2] [1]) :=
  trainer.load "checkpoints/mlp.state" data
```
-/
def load {σ τ : Shape}
    (trainer : TorchLean.Trainer σ τ)
    (path : System.FilePath)
    (data : Dataset σ τ) : IO (Result σ τ) := do
  let session ← trainer.open
  session.load path
  let loss ← session.eval data
  session.finish { before := loss, after := loss }

/--
Train a supervised model from a `Float` sample stream.

Generated-data examples use this when there is no fixed `Dataset` to hand to `trainer.train`. The
evaluation curve is measured on `evalSample`; `onEval` receives the completed step count, a label,
and a prediction function at every curve point.

Example:
```lean
-- No fixed dataset here: a sample is generated per step, and the loss curve is measured on one
-- held-out sample every `curveEvery` updates.
def sampleAt (step : Nat) : Sample.Supervised Float [1] [1] :=
  let x := step.toFloat * 0.01
  { input := [x], target := [2.0 * x] }

def run (trainer : TorchLean.Trainer [1] [1]) (runtime : Runtime.Config) :
    IO (Trainer.StreamResult [1] [1]) :=
  trainer.trainStream runtime sampleAt
    { input := [0.5], target := [1.0] }
    { steps := 500, logEvery := 100 }
    (curveEvery := 50)
```
-/
def trainStream {σ τ : Shape}
    (trainer : TorchLean.Trainer σ τ)
    (options : Runtime.Config)
    (sampleAt : Nat → Sample.Supervised Float σ τ)
    (evalSample : Sample.Supervised Float σ τ)
    (trainOptions : TrainOptions)
    (curveEvery : Nat := 0)
    (onEval : Nat → String →
      (Tensor Float σ → IO (Tensor Float τ)) → IO Unit :=
      fun _ _ _ => pure ()) :
    IO (StreamResult σ τ) := do
  Internal.validateOptions trainOptions
  let configured : TorchLean.Trainer σ τ :=
    { trainer with runtime := trainer.runtime.withRuntime options }
  let session ← configured.open trainOptions.scheduler
  Internal.loadCheckpoint session trainOptions.loadCheckpoint?
  let lossBefore ← session.loss evalSample
  let mut curve : Training.Curve := {}
  curve := curve.push 0 lossBefore
  onEval 0 "before training" session.predict
  let mut currentLoss := lossBefore
  let steps := trainOptions.steps
  let every : Nat := if curveEvery = 0 then Nat.max 1 (steps / 50) else curveEvery
  let runtime := configured.runtime.executionSettings
  let watchEvery := Memory.cadence runtime steps trainOptions.cudaMemorySampleEvery
  let mut memorySample? ← Memory.sample runtime watchEvery steps 0 none
  for stepIndex in [0:steps] do
    let mut batch : Array (Sample.Supervised Float σ τ) := #[]
    for offset in [0:trainOptions.samplesPerStep] do
      batch := batch.push (sampleAt (stepIndex * trainOptions.samplesPerStep + offset))
    let completedSteps := stepIndex + 1
    let logDue := Training.shouldReport trainOptions.logEvery completedSteps
    if logDue then
      let stepLoss ← session.step (batch := true) batch (loss := true)
      IO.println s!"step {completedSteps}: loss={stepLoss}"
    else
      session.step (batch := true) batch
    memorySample? ← Memory.sample runtime watchEvery steps completedSteps memorySample?
    if completedSteps < steps && Training.shouldReport every completedSteps then
      currentLoss ← session.loss evalSample
      curve := curve.push completedSteps currentLoss
      onEval completedSteps s!"step {completedSteps}" session.predict
  if steps > 0 then
    currentLoss ← session.loss evalSample
    curve := curve.push steps currentLoss
  onEval steps "after training" session.predict
  Internal.saveCheckpoint session trainOptions.saveCheckpoint?
  let trained ← session.finish { before := lossBefore, after := currentLoss }
  if trainOptions.logDestination.isEnabled then
    Training.writeLog trainOptions.logDestination
      (curve.toTrainLog trainOptions.logTitle "loss" (notes := trainOptions.logNotes))
  pure { trained := trained, curve := curve }

/--
Train two supervised models from coupled `Float` streams.

The first model receives one supervised sample per step, while the second may receive several.
`evalTotal predictFirst predictSecond` computes the scalar curve to record; it sees only prediction
functions. Both trained results report that paired value as their before/after losses.

This alternates independent supervised updates. It does not differentiate one model's loss through
the other model, so it cannot express an adversarial generator loss.
-/
def trainAlternating {σ₁ τ₁ σ₂ τ₂ : Shape}
    (first : TorchLean.Trainer σ₁ τ₁)
    (second : TorchLean.Trainer σ₂ τ₂)
    (options : Runtime.Config)
    (firstSampleAt : Nat → Sample.Supervised Float σ₁ τ₁)
    (secondSamplesAt : Nat → Array (Sample.Supervised Float σ₂ τ₂))
    (evalTotal :
      (Tensor Float σ₁ → IO (Tensor Float τ₁)) →
      (Tensor Float σ₂ → IO (Tensor Float τ₂)) →
      IO Float)
    (trainOptions : TrainOptions)
    (curveEvery : Nat := 1) :
    IO (AlternatingResult σ₁ τ₁ σ₂ τ₂) := do
  unless trainOptions.samplesPerStep = 1 do
    throw <| IO.userError <|
      "Trainer.trainAlternating: samplesPerStep must be 1; " ++
        "each callback already defines one model's update stream"
  match trainOptions.loadCheckpoint?, trainOptions.saveCheckpoint? with
  | none, none => pure ()
  | _, _ =>
      throw <| IO.userError
        "Trainer.trainAlternating: one checkpoint path cannot represent two model states"
  let firstConfigured : TorchLean.Trainer σ₁ τ₁ :=
    { first with runtime := first.runtime.withRuntime options }
  let secondConfigured : TorchLean.Trainer σ₂ τ₂ :=
    { second with runtime := second.runtime.withRuntime options }
  if firstConfigured.runtime.arithmetic != secondConfigured.runtime.arithmetic then
    throw <| IO.userError
      "Trainer.trainAlternating: both trainers must use the same arithmetic"
  let firstSession ← firstConfigured.open trainOptions.scheduler
  let secondSession ← secondConfigured.open trainOptions.scheduler
  let lossBefore ← evalTotal firstSession.predict secondSession.predict
  let mut curve : Training.Curve := {}
  curve := curve.push 0 lossBefore
  let mut currentLoss := lossBefore
  let steps := trainOptions.steps
  let every := Nat.max 1 curveEvery
  let runtime := firstConfigured.runtime.executionSettings
  let watchEvery := Memory.cadence runtime steps trainOptions.cudaMemorySampleEvery
  let mut memorySample? ← Memory.sample runtime watchEvery steps 0 none
  for stepIndex in [0:steps] do
    firstSession.step (firstSampleAt stepIndex)
    for sample in secondSamplesAt stepIndex do
      secondSession.step sample
    let completedSteps := stepIndex + 1
    memorySample? ← Memory.sample runtime watchEvery steps completedSteps memorySample?
    let curveDue := completedSteps < steps && Training.shouldReport every completedSteps
    let logDue :=
      completedSteps < steps && Training.shouldReport trainOptions.logEvery completedSteps
    if curveDue || logDue then
      currentLoss ← evalTotal firstSession.predict secondSession.predict
      if curveDue then
        curve := curve.push completedSteps currentLoss
      if logDue then
        IO.println s!"step {completedSteps}: loss={currentLoss}"
  if steps > 0 then
    currentLoss ← evalTotal firstSession.predict secondSession.predict
    curve := curve.push steps currentLoss
    if Training.shouldReport trainOptions.logEvery steps then
      IO.println s!"step {steps}: loss={currentLoss}"
  let progress : Training.LossProgress Float := { before := lossBefore, after := currentLoss }
  let firstResult ← firstSession.finish progress
  let secondResult ← secondSession.finish progress
  if trainOptions.logDestination.isEnabled then
    Training.writeLog trainOptions.logDestination
      (curve.toTrainLog trainOptions.logTitle "loss" (notes := trainOptions.logNotes))
  pure { first := firstResult, second := secondResult, curve := curve }

end Trainer

end TorchLean
