/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# Trainer Report API Tests

Regression checks for the numeric report returned by native and executable IEEE training.
-/

@[expose] public section

namespace NN.Tests.API.TrainerReport

open TorchLean

def fail {α : Type} (message : String) : IO α :=
  throw <| IO.userError s!"trainer report check failed: {message}"

def expect (message : String) (condition : Bool) : IO Unit := do
  unless condition do
    fail message

def expectFailure {α : Type} (message : String) (action : IO α) : IO Unit := do
  let failed ← try
    let _ ← action
    pure false
  catch _ =>
    pure true
  unless failed do
    fail s!"{message}: expected failure"

def close (left right : Float) : Bool :=
  Float.abs (left - right) ≤ 1e-5

def model : nn.Builder (nn.Sequential [1] [1]) :=
  nn.linear 1 1

def classifier : nn.Builder (nn.Sequential [1] [2]) :=
  nn.linear 1 2

def data : Trainer.Dataset [1] [1] :=
  Data.fromTensors
    ([[0.0], [1.0]] : Tensor Float [2, 1])
    ([[0.0], [1.0]] : Tensor Float [2, 1])

/-- Custom objective equivalent to the built-in mean-squared-error objective. -/
def customMse :
    ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
      Runtime.Autograd.Model.Program α
        [Shape.ofList [1], Shape.ofList [1]] [] :=
  fun {α} _ _ =>
    fun {m} _ _ =>
      fun prediction target =>
        Loss.mse (m := m) (α := α) (s := Shape.ofList [1]) prediction target

def regressionTrainer
    (objective : Trainer.Objective [1])
    (arithmetic : Runtime.Arithmetic := .native)
    (seed : Nat := 17) :
    TorchLean.Trainer [1] [1] :=
  Trainer.new model
    { objective
      optimizer := optim.sgd { learningRate := 0.01 }
      arithmetic
      seed }

def streamSample (index : Nat) : Sample.Supervised Float [1] [1] :=
  if index % 2 = 0 then
    { input := ([0.0] : Tensor Float [1])
      target := ([0.0] : Tensor Float [1]) }
  else
    { input := ([1.0] : Tensor Float [1])
      target := ([1.0] : Tensor Float [1]) }

def classificationSample (index : Nat) : Sample.Supervised Float [1] [2] :=
  if index % 2 = 0 then
    { input := ([0.0] : Tensor Float [1])
      target := ([1.0, 0.0] : Tensor Float [2]) }
  else
    { input := ([1.0] : Tensor Float [1])
      target := ([0.0, 1.0] : Tensor Float [2]) }

def check (arithmetic : Runtime.Arithmetic) : IO Unit := do
  let trainer := regressionTrainer .mse arithmetic
  let trained ← trainer.train data
    { steps := 0
      logDestination := .disabled }
  let report := trained.report
  expect "report should preserve the requested step count"
    (report.steps == 0)
  expect "report losses should be finite host values"
    (report.loss.before.isFinite && report.loss.after.isFinite)
  expect "zero updates should preserve the measured loss"
    (report.loss.before == report.loss.after)

  let log := report.toTrainLog "report regression" #["zero updates"]
  expect "report log should contain the before/after step indices"
    (log.steps == #[0, 0])
  expect "report log should contain one loss series"
    (log.series.size == 1 && log.series[0]!.name == "loss")
  expect "report log should preserve the numeric losses"
    (log.series[0]!.values == #[report.loss.before, report.loss.after])

  let prediction ← trained.predict ([0.5] : Tensor Float [1])
  expect "trained result should retain a finite prediction closure"
    prediction[0].isFinite

/-- Built-in and custom MSE use the same batching, scheduling, reporting, and prediction path. -/
def checkCustomParity : IO Unit := do
  let options : Trainer.TrainOptions :=
    { steps := 2
      samplesPerStep := 2
      scheduler := some (.step 0.01 1 0.5)
      logDestination := .disabled }
  let builtin ←
    (regressionTrainer .mse).train data options
  let custom ←
    (regressionTrainer (.custom customMse)).train data options
  expect "custom MSE should preserve the requested step count"
    (custom.report.steps == options.steps)
  expect "custom and built-in MSE should report the same initial loss"
    (close custom.report.loss.before builtin.report.loss.before)
  expect "custom and built-in MSE should report the same final loss"
    (close custom.report.loss.after builtin.report.loss.after)
  let input := ([0.25] : Tensor Float [1])
  let builtinPrediction ← builtin.predict input
  let customPrediction ← custom.predict input
  expect "custom and built-in MSE should produce the same trained prediction"
    (close builtinPrediction[0] customPrediction[0])

/-- Checkpoints apply uniformly to built-in and custom supervised objectives. -/
def checkCustomCheckpoint : IO Unit := do
  let path : System.FilePath := "/tmp/torchlean-trainer-report-custom-state.json"
  if ← path.pathExists then
    IO.FS.removeFile path
  let source ←
    (regressionTrainer (.custom customMse) (seed := 31)).train data
      { steps := 1
        saveCheckpoint? := some path
        logDestination := .disabled }
  expect "custom training should write its requested checkpoint" (← path.pathExists)
  let restored ←
    (regressionTrainer (.custom customMse) (seed := 999)).train data
      { steps := 0
        loadCheckpoint? := some path
        logDestination := .disabled }
  let input := ([0.75] : Tensor Float [1])
  let sourcePrediction ← source.predict input
  let restoredPrediction ← restored.predict input
  expect "loading a custom-objective checkpoint should restore the trained state"
    (close sourcePrediction[0] restoredPrediction[0])
  IO.FS.removeFile path

/-- Stream training supports custom and one-hot objectives with one callback per recorded step. -/
def checkStreamObjectives : IO Unit := do
  let callbackSteps ← IO.mkRef (#[] : Array Nat)
  let custom ←
    (regressionTrainer (.custom customMse)).trainStream
      { execution := .typedGraph }
      streamSample
      (streamSample 0)
      { steps := 2
        samplesPerStep := 2
        scheduler := some (.exponential 0.01 0.9)
        logDestination := .disabled }
      (curveEvery := 1)
      (onEval := fun step _ _ => callbackSteps.modify (·.push step))
  expect "stream callbacks should not duplicate the final step"
    ((← callbackSteps.get) == #[0, 1, 2])
  expect "stream curves should contain one point per requested reporting step"
    (custom.curve.steps == #[0, 1, 2])
  expect "stream reports should count optimizer updates, not source items"
    (custom.trained.report.steps == 2)
  let _ ←
    custom.trained.verify
      ([0.5] : Tensor Float [1])
      (radius := 0.01)
      (algorithm := .ibp)

  let classificationTrainer :=
    Trainer.new classifier
      { objective := .oneHotCrossEntropy 0
        optimizer := optim.sgd { learningRate := 0.01 }
        seed := 19 }
  let classification ←
    classificationTrainer.trainStream
      { execution := .typedGraph }
      classificationSample
      (classificationSample 0)
      { steps := 1
        logDestination := .disabled }
  let prediction ← classification.predict ([0.5] : Tensor Float [1])
  expect "one-hot stream training should return finite logits"
    (prediction[0].isFinite && prediction[1].isFinite)

/-- Alternating training reports actual per-model updates and rejects ambiguous shared options. -/
def checkAlternating : IO Unit := do
  let first := regressionTrainer .mse (seed := 41)
  let second := regressionTrainer (.custom customMse) (seed := 43)
  let runtime : Runtime.Config := { execution := .typedGraph }
  let firstSampleAt := streamSample
  let secondSamplesAt := fun step =>
    #[streamSample (2 * step), streamSample (2 * step + 1)]
  let evalTotal :=
    fun predictFirst predictSecond => do
      let firstPrediction ← predictFirst ([0.5] : Tensor Float [1])
      let secondPrediction ← predictSecond ([0.5] : Tensor Float [1])
      pure (firstPrediction[0] * firstPrediction[0] +
        secondPrediction[0] * secondPrediction[0])
  let run := fun (options : Trainer.TrainOptions) =>
    first.trainAlternating second runtime firstSampleAt secondSamplesAt evalTotal
      options (curveEvery := 1)

  let trained ← run { steps := 2, logDestination := .disabled }
  expect "the first alternating report should count its two optimizer updates"
    (trained.first.report.steps == 2)
  expect "the second alternating report should count all four optimizer updates"
    (trained.second.report.steps == 4)
  expect "the alternating curve should contain one final point"
    (trained.curve.steps == #[0, 1, 2])
  let _ ←
    trained.first.verify
      ([0.5] : Tensor Float [1])
      (radius := 0.01)
      (algorithm := .ibp)
  let _ ←
    trained.second.verify
      ([0.5] : Tensor Float [1])
      (radius := 0.01)
      (algorithm := .ibp)

  expectFailure "alternating samples per step" <|
    run { steps := 1, samplesPerStep := 2, logDestination := .disabled }
  expectFailure "alternating checkpoint path" <|
    run
      { steps := 1
        saveCheckpoint? := some "/tmp/unused-alternating-checkpoint.json"
        logDestination := .disabled }

/-- Persisted metrics must retain small values and all finite binary64 significand bits. -/
def checkMetricJsonPrecision : IO Unit := do
  let values : Array Float := #[0.0, 1e-9, -1e-9, 0.377564936876297, 1.0 / 3.0,
    Float.ofBits 1, Float.ofBits 0x000fffffffffffff, Float.ofBits 0x0010000000000000,
    Float.ofBits 0x7fefffffffffffff, Float.ofBits 0xffefffffffffffff]
  for value in values do
    let encoded := Runtime.Training.JsonCodec.floatToJson value
    let parsed ← match Lean.Json.parse encoded.compress with
      | .ok parsed => pure parsed
      | .error message => fail message
    let decoded ← match Runtime.Training.JsonCodec.floatOfJsonE "metric" parsed with
      | .ok decoded => pure decoded
      | .error message => fail message
    expect "metric JSON should round-trip the original finite value"
      (decoded.toBits == value.toBits)

/-- Both trainer runtimes reject hyperparameters made invalid by binary32 rounding. -/
def checkRuntimeHyperparameters : IO Unit := do
  for arithmetic in #[Runtime.Arithmetic.native, .ieee] do
    let invalid := Trainer.new model
      { optimizer := optim.adam { learningRate := 0.01, beta1 := 0.999999999 }
        arithmetic }
    expectFailure "optimizer beta rounds to one" invalid.open
    let valid := regressionTrainer .mse arithmetic
    expectFailure "scheduler rate overflows binary32" <|
      valid.open (scheduler := some (.constant 1e300))

/-- Live and frozen predictions enforce the same selected CPU capsule contract. -/
def checkSnapshotProfile (capsule : NN.Backend.KernelCapsule) (needle : String) : IO Unit := do
  let profile := NN.Backend.BackendProfile.checkedCpu.withCapsuleModules
    #[{ name := "reference", capsules := #[capsule] }]
  for arithmetic in #[Runtime.Arithmetic.native, .ieee] do
    let trainer : TorchLean.Trainer [1] [1] :=
      Trainer.new nn.relu { arithmetic, backendProfile? := some profile }
    let session ← trainer.open
    let frozen ← session.finish { before := 0.0, after := 0.0 }
    let input := ([1.0] : Tensor Float [1])
    for (label, action) in #[
        ("live", session.predict input), ("frozen", frozen.predict input)] do
      let failure ← try
        let _ ← action
        pure none
      catch error => pure (some error.toString)
      match failure with
      | none => fail s!"{label} prediction bypassed profile {capsule.name}"
      | some message =>
          expect s!"{label} prediction should preserve the profile rejection"
            (message.contains needle)
  IO.println s!"  frozen and live CPU profile rejection ({capsule.name}): passed"

/-- Snapshots preserve provider binding and the checked policy's value-contract guard. -/
def checkSnapshotProfiles : IO Unit := do
  checkSnapshotProfile
    { NN.Backend.Reference.relu with name := "mismatched.relu", provider := .torchLean }
    "no matching executable handler is linked"
  checkSnapshotProfile
    { NN.Backend.Reference.relu with
      name := "trusted_value.relu"
      valueContract := NN.Backend.ContractDescriptor.trusted (.valueRefinement .relu)
        "Test-only value contract resting on a trusted boundary."
        "trainer snapshot profile test" }
    "rejected `relu`"

def run : IO Unit := do
  checkMetricJsonPrecision
  checkRuntimeHyperparameters
  checkSnapshotProfiles
  check .native
  check .ieee
  checkCustomParity
  checkCustomCheckpoint
  checkStreamObjectives
  checkAlternating
  IO.println "  trainer report API: passed"

end NN.Tests.API.TrainerReport
