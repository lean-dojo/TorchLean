/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Verification.Core
public import NN.API.Trainer.Core -- shake: keep
public import NN.API.Trainer.BatchInput

/-!
# Training Results

Regression, cross-entropy, and custom losses all return the same trained-model type.

A `Result` keeps a parameter snapshot of the trained model. Prediction and verification use it
directly. `Result.state` retains the session's public scalar and `Result.save` writes its exact
encoding with `Checkpoint.State.save`. Results are produced by `Session.finish`; verification is
available when the opening path supplies a verifier for its runtime scalar.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/--
A trained TorchLean model.

The result retains its parameter snapshot through prediction, state-reading, and verification
closures. Continued training of the source session does not change the result.

The default public scalar is `Float`: ordinary sessions read binary32 state back exactly. Typed
sessions retain their selected scalar through state, prediction, reporting, and checkpoint encoding.
-/
structure Result (σ τ : Shape) (α : Type := Float) [Storage α] where
  private mk ::
  private reportValue : Report α
  /-- Shapes of the trained parameters and buffers, in the order used by `state` and `save`. -/
  stateShapes : List Shape
  private readState : IO (TorchLean.nn.State α stateShapes)
  private saveState : System.FilePath → IO Unit
  private predictOne :
    Tensor α σ → IO (Tensor α τ)
  private runVerification? : Option (
    (center : Tensor Float σ) →
    (radius : Float) →
    Verification.Norm →
    Verification.Property →
    Verification.Algorithm →
    IO Verification.Report)

namespace Result

variable {α : Type} [Storage α]

namespace Internal

/-- Construct an opaque trained-model result at the trainer implementation boundary. -/
opaque create {σ τ : Shape}
    (report : Report α)
    (stateShapes : List Shape)
    (readState : IO (TorchLean.nn.State α stateShapes))
    (saveState : System.FilePath → IO Unit)
    (predict : Tensor α σ → IO (Tensor α τ))
    (verify? : Option (Tensor Float σ → Float → Verification.Norm →
      Verification.Property → Verification.Algorithm → IO Verification.Report) := none) :
    Trainer.Result σ τ α :=
  ⟨report, stateShapes, readState, saveState, predict, verify?⟩

end Internal

/-- Loss progress, step count, and runtime arithmetic for the completed run. -/
opaque report {σ τ : Shape}
    (result : Result σ τ α) : Report α :=
  result.reportValue

/--
Read the trained parameters and persistent buffers in the session's public scalar.

Ordinary sessions read binary32 state back to `Float` exactly; typed sessions preserve `α`.
The layout `result.stateShapes` equals `nn.stateShapes` of the trained model, which is what
`Checkpoint.State.save` and `Checkpoint.State.load` expect.
-/
opaque state {σ τ : Shape}
    (result : Result σ τ α) : IO (TorchLean.nn.State α result.stateShapes) :=
  result.readState

/--
Save the trained state with `Checkpoint.State.save`.

Restore it with `Session.load` on a session using the same scalar. For typed state, open that
session with `trainer.openTyped (α := α)` before loading the checkpoint.
-/
opaque save {σ τ : Shape}
    (result : Result σ τ α) (path : System.FilePath) : IO Unit :=
  result.saveState path

/--
Run evaluation-mode prediction through the trained snapshot.

`batch := true` maps over a leading axis of length `batchSize`, including an empty axis.
-/
opaque predict {σ τ inputShape : Shape}
    (result : Result σ τ α) (input : Tensor α inputShape)
    (batch : Bool := false) (batchSize : Nat := 1)
    [Trainer.Internal.BatchInput
      (Tensor α σ) (Tensor α (σ.prependDim batchSize)) batch (Tensor α inputShape)] :
    IO (Tensor α (match batch with | false => τ | true => τ.prependDim batchSize)) := by
  have inputType := Trainer.Internal.BatchInput.type_eq
    (single := Tensor α σ) (many := Tensor α (σ.prependDim batchSize)) (batch := batch)
  cases batch with
  | false => exact result.predictOne (inputType.mp input)
  | true =>
      change IO (Tensor α (τ.prependDim batchSize))
      let input : Tensor α (σ.prependDim batchSize) := inputType.mp input
      exact Tensor.stackLeadingM fun index => result.predictOne input[index]

/--
Verify the trained model over a region around `center`.

`property` and `algorithm` have defaults, so checking output bounds needs only `center` and
`radius`. This fails explicitly if the opening path supplied no verifier, including `openTyped`.
-/
opaque verify {σ τ : Shape}
    (result : Result σ τ α)
    (center : Tensor Float σ)
    (radius : Float)
    (norm : Verification.Norm := .inf)
    (property : Verification.Property := .bounds)
    (algorithm : Verification.Algorithm := .alphaBetaCrown) :
    IO Verification.Report :=
  match result.runVerification? with
  | some verify => verify center radius norm property algorithm
  | none => throw <| IO.userError "verification is unavailable for this result's scalar/runtime"

end Result

namespace Result

variable {α : Type} [Storage α]

/-- One-line summary for the completed training run. -/
def summary {σ τ : Shape} [ToString α]
    (result : Result σ τ α) : String :=
  result.report.summary

/-- Print the before/after training summary. -/
def printSummary {σ τ : Shape} [ToString α]
    (result : Result σ τ α) : IO Unit :=
  IO.println result.summary

/-- Print one prediction with a caller-supplied label. -/
def printPrediction {σ τ : Shape} [Repr α]
    (result : Result σ τ α) (label : String)
    (input : Tensor α σ) : IO Unit := do
  let prediction ← result.predict input
  IO.println s!"{label} = {reprStr prediction}"

instance {σ τ : Shape} [ToString α] :
    ToString (Result σ τ α) where
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

/-- Predict one input, or map a leading tensor batch, through the trained stream result. -/
def predict {σ τ inputShape : Shape}
    (result : StreamResult σ τ) (input : Tensor Float inputShape)
    (batch : Bool := false) (batchSize : Nat := 1)
    [Trainer.Internal.BatchInput
      (Tensor Float σ) (Tensor Float (σ.prependDim batchSize)) batch (Tensor Float inputShape)] :
    IO (Tensor Float (match batch with | false => τ | true => τ.prependDim batchSize)) :=
  result.trained.predict input (batch := batch) (batchSize := batchSize)

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
