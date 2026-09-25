/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# Trainer Checkpoint API Tests

Round-trip checks for `Trainer.Result.state`, `Trainer.Result.save`, and `Trainer.load` under
both real arithmetic modes.
-/

@[expose] public section

namespace NN.Tests.API.TrainerCheckpoint

open TorchLean

def fail {α : Type} (message : String) : IO α :=
  throw <| IO.userError s!"trainer checkpoint check failed: {message}"

def expect (message : String) (condition : Bool) : IO Unit := do
  unless condition do
    fail message

def close (left right : Float) : Bool :=
  Float.abs (left - right) ≤ 1e-6

def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 3,
    nn.relu,
    nn.linear 3 1
  ]

def xs : Tensor Float [4, 2] :=
  [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

def ys : Tensor Float [4, 1] :=
  [[0.2], [1.0], [1.0], [1.8]]

def data : Trainer.Dataset [2] [1] := Data.fromTensors xs ys

def trainer (arithmetic : Runtime.Arithmetic) (seed : Nat) : TorchLean.Trainer [2] [1] :=
  Trainer.new model
    { objective := .mse
      optimizer := optim.sgd { learningRate := 0.05 }
      arithmetic
      seed }

def probe : Tensor Float [2] := [0.25, -0.75]

/-- Two equally shaped `Float` tensors agree up to a small absolute tolerance. -/
def tensorClose {shape : Shape} (left right : Tensor Float shape) : Bool :=
  Tensor.sum (Tensor.map Float.abs (Tensor.sub left right : Tensor Float shape)) ≤ 1e-6

/-- Every tensor of two equally laid out `Float` packs agrees elementwise. -/
def packClose : {shapes : List Shape} →
    TensorPack Float shapes → TensorPack Float shapes → Bool
  | _, .nil, .nil => true
  | _, .cons left leftRest, .cons right rightRest =>
      tensorClose left right && packClose leftRest rightRest

/-- Two `Float` states have the same layout and agree elementwise. -/
def statesClose {leftShapes rightShapes : List Shape}
    (left : nn.State Float leftShapes) (right : nn.State Float rightShapes) : Bool :=
  if sameShapes : leftShapes = rightShapes then
    packClose (nn.State.Internal.toTensorPack (left.cast sameShapes))
      (nn.State.Internal.toTensorPack right)
  else
    false

def check (arithmetic : Runtime.Arithmetic) : IO Unit := do
  let path : System.FilePath :=
    s!"/tmp/torchlean-trainer-checkpoint-{arithmetic.cliName}.state"
  if ← path.pathExists then
    IO.FS.removeFile path
  let source := trainer arithmetic 11
  let trained ← source.train data { steps := 3, logDestination := .disabled }
  expect "report should record the requested arithmetic"
    (trained.report.arithmetic == arithmetic)
  let built := nn.build 11 model
  let trainedState ← trained.state
  expect "three updates should move the parameters away from initialization"
    (!statesClose trainedState (nn.initialState built))

  trained.save path
  let reloadedState ← Checkpoint.State.load built path
  expect "saved state should round-trip through Checkpoint.State.load"
    (statesClose trainedState reloadedState)

  let restored ← (trainer arithmetic 999).load path data
  expect "restored report should count zero updates" (restored.report.steps == 0)
  expect "restored losses should be finite host values"
    (restored.report.loss.before.isFinite && restored.report.loss.after.isFinite)
  let trainedPrediction ← trained.predict probe
  let restoredPrediction ← restored.predict probe
  expect "restored model should reproduce the trained prediction"
    (close trainedPrediction[0] restoredPrediction[0])
  let restoredState ← restored.state
  expect "restored state should equal the saved state"
    (statesClose trainedState restoredState)
  IO.FS.removeFile path

/-- Continuing a session must not change a result that was already returned. -/
def checkSnapshot (arithmetic : Runtime.Arithmetic) : IO Unit := do
  let source := trainer arithmetic 11
  let session ← source.open
  let sample : Sample.Supervised Float [2] [1] :=
    { input := probe, target := [3.0] }
  let before ← session.loss sample
  session.step sample
  let after ← session.loss sample
  let result ← session.finish { before, after }
  let state ← result.state
  let prediction ← result.predict probe
  let verification ← result.verify probe (radius := 0.1) (algorithm := .ibp)
  for _ in [0:5] do
    session.step (batch := true) #[sample, sample]
  expect "continued updates should change the live session"
    (!statesClose state (← session.state))
  expect "a finished result should keep its step count" (result.report.steps == 1)
  expect "a finished result should keep its parameters"
    (statesClose state (← result.state))
  expect "a finished result should keep its predictions"
    (tensorClose prediction (← result.predict probe))
  let repeatedVerification ← result.verify probe (radius := 0.1) (algorithm := .ibp)
  expect "verification should use the same snapshot as prediction"
    (reprStr verification == reprStr repeatedVerification)
  let path : System.FilePath :=
    s!"/tmp/torchlean-trainer-snapshot-{arithmetic.cliName}.state"
  try
    result.save path
    let saved ← Checkpoint.State.load source.model path
    expect "saving a finished result should save its snapshot" (statesClose state saved)
    session.load path
    session.step sample
    expect "loading and updating the session should leave the result alone"
      (statesClose state (← result.state))
  finally
    if ← path.pathExists then IO.FS.removeFile path

/-- Batched evaluation preserves ordered means, leading axes, and the live state. -/
def checkBatchEvaluation (arithmetic : Runtime.Arithmetic) : IO Unit := do
  let source := trainer arithmetic 11
  let session ← source.open
  let state ← session.state
  let samples : Data.SampleStream (Sample.Supervised Float [2] [1]) :=
    Data.SampleStream.fromFunction 4 fun index =>
      { input := xs[index], target := ys[index] }
  let mut total : Float := 0
  for h : i in [0:samples.size] do
    total := total + (← session.loss (samples.get ⟨i, h.2.1⟩))
  let mean ← session.loss samples (batch := true)
  expect "stream loss should retain the ordered Float reduction"
    (mean == total / samples.size.toFloat)
  let empty := Data.SampleStream.fromArray (#[] : Array (Sample.Supervised Float [2] [1]))
  expect "empty stream loss should be zero" ((← session.loss empty (batch := true)) == 0)
  let predictions ← session.predict xs (batch := true) (batchSize := 4)
  for i in List.finRange 4 do
    expect "batch prediction should match evaluation of each leading slice"
      (tensorClose predictions[i] (← session.predict xs[i]))
  let emptyInputs : Tensor Float [0, 2] := Tensor.zeros [0, 2]
  let emptyPredictions ← session.predict emptyInputs (batch := true) (batchSize := 0)
  expect "empty prediction should have no elements"
    ((Tensor.to emptyPredictions (Array Float)).isEmpty)
  expect "evaluation should not update parameters or buffers" (statesClose state (← session.state))
  expect "evaluation should not count as an optimizer update" ((← session.steps) == 0)

def run : IO Unit := do
  check .native
  check .ieee
  checkSnapshot .native
  checkSnapshot .ieee
  checkBatchEvaluation .native
  checkBatchEvaluation .ieee
  IO.println "  trainer checkpoint API: passed"

end NN.Tests.API.TrainerCheckpoint
