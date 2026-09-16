/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Verification.Core
public import NN.API.Neural.State
public import NN.API.Trainer.Summary
public import NN.API.Trainer.Core -- shake: keep

/-!
# Training Results

Regression, cross-entropy, and custom losses all return the same trained-model type.

A `Result` keeps a parameter snapshot of the trained model. Prediction and verification use it
directly. `Result.state` reads the parameters back as `Float` tensors and `Result.save` writes
them with `Checkpoint.State.save`, so a trained model can be stored and later restored with
`Trainer.load`. Results are produced by `Session.finish`.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/--
A trained TorchLean model.

The result retains its parameter snapshot through prediction, state-reading, and verification
closures. Continued training of the source session does not change the result.

The runtime state lives in the binary32 scalar selected by `RunConfig.arithmetic` (`Float32` or
`ExecFloat.Binary 8 23`). `state` and `save` read it back as `Float`, which is exact because every
binary32
value is a binary64 value.
-/
structure Result (σ τ : Shape) where
  private mk ::
  private reportValue : Report
  /-- Shapes of the trained parameters and buffers, in the order used by `state` and `save`. -/
  stateShapes : List Shape
  private readState : IO (TorchLean.nn.State Float stateShapes)
  private saveState : System.FilePath → IO Unit
  private predictOne :
    Tensor Float σ → IO (Tensor Float τ)
  private runVerification :
    (center : Tensor Float σ) →
    (radius : Float) →
    Verification.Norm →
    Verification.Property →
    Verification.Algorithm →
    IO Verification.Report

namespace Result

namespace Internal

/-- Verification implementation used by training paths that cannot lower a verifier. -/
def verificationUnavailable {σ : Shape} :
    Tensor Float σ → Float → Verification.Norm →
      Verification.Property → Verification.Algorithm → IO Verification.Report :=
  fun _ _ _ _ _ =>
    throw <| IO.userError <|
      "verification is unavailable for this result; " ++
        "use the public trainer path with a supported model"

/-- Construct an opaque trained-model result at the trainer implementation boundary. -/
opaque create {σ τ : Shape}
    (report : Report)
    (stateShapes : List Shape)
    (readState : IO (TorchLean.nn.State Float stateShapes))
    (saveState : System.FilePath → IO Unit)
    (predict : Tensor Float σ → IO (Tensor Float τ))
    (verify := verificationUnavailable (σ := σ)) :
    Trainer.Result σ τ :=
  ⟨report, stateShapes, readState, saveState, predict, verify⟩

end Internal

/-- Loss progress, step count, and runtime arithmetic for the completed run. -/
opaque report {σ τ : Shape}
    (result : Result σ τ) : Report :=
  result.reportValue

/--
Read the trained parameters and persistent buffers as `Float` tensors.

The runtime holds them in binary32 (`Float32` or `ExecFloat.Binary 8 23`); reading them back to
binary64 is
exact. The layout `result.stateShapes` equals `nn.stateShapes` of the trained model, which is what
`Checkpoint.State.save` and `Checkpoint.State.load` expect.
-/
opaque state {σ τ : Shape}
    (result : Result σ τ) : IO (TorchLean.nn.State Float result.stateShapes) :=
  result.readState

/-- Save the trained state with `Checkpoint.State.save`; restore it with `Trainer.load`. -/
opaque save {σ τ : Shape}
    (result : Result σ τ) (path : System.FilePath) : IO Unit :=
  result.saveState path

/-- Run one `Float` input through the trained model. -/
opaque predict {σ τ : Shape}
    (result : Result σ τ)
    (input : Tensor Float σ) : IO (Tensor Float τ) :=
  result.predictOne input

/-- Run several `Float` inputs through the trained model. -/
opaque predictMany {σ τ : Shape} {batch : Nat}
    (result : Result σ τ)
    (inputs : Tensor Float (σ.prependDim batch)) :
    IO (Tensor Float (τ.prependDim batch)) :=
  Tensor.stackLeadingM fun index => result.predictOne inputs[index]

/--
Verify the trained model over a region around `center`.

`property` and `algorithm` have defaults, so checking output bounds needs only `center` and
`radius`.
-/
opaque verify {σ τ : Shape}
    (result : Result σ τ)
    (center : Tensor Float σ)
    (radius : Float)
    (norm : Verification.Norm := .inf)
    (property : Verification.Property := .bounds)
    (algorithm : Verification.Algorithm := .alphaBetaCrown) :
    IO Verification.Report :=
  result.runVerification center radius norm property algorithm

end Result

namespace Result

/-- One-line summary for the completed training run. -/
def summary {σ τ : Shape}
    (result : Result σ τ) : String :=
  result.report.summary

/-- Print the before/after training summary. -/
def printSummary {σ τ : Shape}
    (result : Result σ τ) : IO Unit :=
  IO.println result.summary

/-- Print one prediction with a caller-supplied label. -/
def printPrediction {σ τ : Shape}
    (result : Result σ τ) (label : String)
    (input : Tensor Float σ) : IO Unit := do
  let prediction ← result.predict input
  IO.println s!"{label} = {reprStr prediction}"

instance {σ τ : Shape} :
    ToString (Result σ τ) where
  toString := summary

end Result

/--
A trained model returned by step-indexed stream training.

Generated or resampled workloads may not have one static dataset to summarize. The ordinary training
result is paired with the evaluation curve collected from a caller-provided sample.
-/
structure StreamResult (σ τ : Shape) where
  /-- Trained model result. -/
  trained : Result σ τ
  /-- Evaluation loss curve recorded during stream training. -/
  curve : Training.Curve

namespace StreamResult

/-- One-line summary for the trained stream run. -/
def summary {σ τ : Shape}
    (result : StreamResult σ τ) : String :=
  result.trained.summary

/-- Print the stream training summary. -/
def printSummary {σ τ : Shape}
    (result : StreamResult σ τ) : IO Unit :=
  IO.println result.summary

/-- Run one prediction through the trained stream result. -/
def predict {σ τ : Shape}
    (result : StreamResult σ τ) (input : Tensor Float σ) :
    IO (Tensor Float τ) :=
  result.trained.predict input

/-- Run several predictions through the trained stream result. -/
def predictMany {σ τ : Shape} {batch : Nat}
    (result : StreamResult σ τ)
    (inputs : Tensor Float (σ.prependDim batch)) : IO (Tensor Float (τ.prependDim batch)) :=
  result.trained.predictMany inputs

instance {σ τ : Shape} :
    ToString (StreamResult σ τ) where
  toString := summary

end StreamResult

/-- Two trained regression models and the coupled metric recorded by alternating updates. -/
structure AlternatingResult
    (σ₁ τ₁ σ₂ τ₂ : Shape) where
  /-- Trained result for the first model. -/
  first : Result σ₁ τ₁
  /-- Trained result for the second model. -/
  second : Result σ₂ τ₂
  /-- Task-specific curve recorded by the caller-provided evaluation function. -/
  curve : Training.Curve

namespace AlternatingResult

/-- One-line summary for the two trained models. -/
def summary {σ₁ τ₁ σ₂ τ₂ : Shape}
    (result : AlternatingResult σ₁ τ₁ σ₂ τ₂) : String :=
  s!"first: {result.first.summary}; second: {result.second.summary}"

/-- Print the training summary for both models. -/
def printSummary {σ₁ τ₁ σ₂ τ₂ : Shape}
    (result : AlternatingResult σ₁ τ₁ σ₂ τ₂) : IO Unit :=
  IO.println result.summary

/-- Print the endpoints of the coupled metric curve. -/
def printCurveSummary {σ₁ τ₁ σ₂ τ₂ : Shape}
    (result : AlternatingResult σ₁ τ₁ σ₂ τ₂)
    (metric : String := "loss") : IO Unit := do
  let endpoints ←
    Training.Curve.endpoints result.curve "AlternatingResult.printCurveSummary"
  IO.println
    s!"  steps={endpoints.lastStep} {metric}={endpoints.firstValue} -> {endpoints.lastValue}"

instance {σ₁ τ₁ σ₂ τ₂ : Shape} :
    ToString (AlternatingResult σ₁ τ₁ σ₂ τ₂) where
  toString := summary

end AlternatingResult

end Trainer

end TorchLean
