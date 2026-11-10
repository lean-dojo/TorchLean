/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Trainer -- shake: keep
public import TorchLean.NN.IR.Imported
public import TorchLean.NN.BatchInput

/-!
# Training Results

Regression, cross-entropy and custom losses all return the same trained-model type, `Result`,
which keeps a parameter snapshot of the trained model for prediction.
`Result.state` reads the snapshot in the session's public scalar, and `Result.save` writes its
exact encoding with `Checkpoint.State.save`.

Results come from `Session.finish`. A result of `trainer.train`, or of a session from
`trainer.open`, also keeps the lowering of the model to the verifier IR, which `trained.verify`
(`TorchLean.Verify.Trainer`) runs on first use. `StreamResult` and `AlternatingResult` pair
trained results with the curve recorded during stream or alternating training.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/--
A trained model lowered to the verifier IR (`IR.Imported`) in the runtime scalar of its
session. The lowering runs once, on first use.
-/
inductive LoweredModel where
  /-- The lowering in binary32, for `native` arithmetic. -/
  | native (lowered : Thunk (Except String (IR.Imported Float32)))
  /-- The lowering in the binary32 `ExecFloat` model of IEEE 754, for `ieee` arithmetic. -/
  | ieee (lowered : Thunk (Except String (IR.Imported
      (FloatLib.Floats.ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)))))

/--
A trained TorchLean model. Its prediction and state-reading closures and the lowering for
`trained.verify` hold a parameter snapshot, so further training of the source session leaves the
result unchanged.

The public scalar defaults to `Float`, and ordinary sessions read their binary32 state back into
it exactly. Typed sessions keep their own scalar through state, prediction, reporting and
checkpoint encoding.
-/
structure Result (σ τ : Shape) (α : Type := Float) [Storage α] where
  private mk ::
  private reportValue : Report α
  /-- Shapes of the trained parameters and buffers, in the order used by `state` and `save`. -/
  stateShapes : List Shape
  private readState : IO (TorchLean.TensorPack α stateShapes)
  private saveState : System.FilePath → IO Unit
  private predictOne :
    Tensor α σ → IO (Tensor α τ)
  private loweredModel? : Option LoweredModel

/--
Apply `predictOne` to `input`, or with `batch := true` to each entry of its leading axis of
length `batchSize`, including an empty axis.
-/
def predictWith {α : Type} [Storage α] {σ τ inputShape : Shape}
    (predictOne : Tensor α σ → IO (Tensor α τ)) (input : Tensor α inputShape)
    (batch : Bool := false) (batchSize : Nat := 1)
    [BatchInput
      (Tensor α σ) (Tensor α (Shape.dim batchSize σ)) batch (Tensor α inputShape)] :
    IO (Tensor α (match batch with | false => τ | true => Shape.dim batchSize τ)) := by
  have inputType := BatchInput.type_eq
    (single := Tensor α σ) (many := Tensor α (Shape.dim batchSize σ)) (batch := batch)
  cases batch with
  | false => exact predictOne (inputType.mp input)
  | true =>
      change IO (Tensor α (Shape.dim batchSize τ))
      let input : Tensor α (Shape.dim batchSize σ) := inputType.mp input
      exact Tensor.stackLeadingM fun index => predictOne input[index]

namespace Result

variable {α : Type} [Storage α]

/-- Construct an opaque trained-model result at the trainer implementation boundary. -/
opaque create {σ τ : Shape}
    (report : Report α)
    (stateShapes : List Shape)
    (readState : IO (TorchLean.TensorPack α stateShapes))
    (saveState : System.FilePath → IO Unit)
    (predict : Tensor α σ → IO (Tensor α τ))
    (loweredModel? : Option Trainer.LoweredModel := none) :
    Trainer.Result σ τ α :=
  ⟨report, stateShapes, readState, saveState, predict, loweredModel?⟩

/-- The lowering kept by a result, `none` when its session supplied none. -/
opaque lowered? {σ τ : Shape} (result : Trainer.Result σ τ α) :
    Option Trainer.LoweredModel :=
  result.loweredModel?

/-- Loss progress, step count, and runtime arithmetic for the completed run. -/
opaque report {σ τ : Shape}
    (result : Result σ τ α) : Report α :=
  result.reportValue

/--
Read the trained parameters and persistent buffers in the session's public scalar.

Ordinary sessions read binary32 state back to `Float` exactly, and typed sessions keep `α`. The
layout `result.stateShapes` equals `nn.Sequential.stateShapes` of the trained model, which is what
`Checkpoint.State.save` and `Checkpoint.State.load` expect.
-/
opaque state {σ τ : Shape}
    (result : Result σ τ α) : IO (TorchLean.TensorPack α result.stateShapes) :=
  result.readState

/--
Save the trained state with `Checkpoint.State.save`.

Restore it with `Session.load` on a session in the same scalar, which for typed state means a
session opened with `trainer.openTyped (α := α)`.
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
    [BatchInput
      (Tensor α σ) (Tensor α (Shape.dim batchSize σ)) batch (Tensor α inputShape)] :
    IO (Tensor α (match batch with | false => τ | true => Shape.dim batchSize τ)) :=
  Trainer.predictWith result.predictOne input (batch := batch) (batchSize := batchSize)

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
A trained model returned by step-indexed stream training, paired with the evaluation curve
measured on a caller-provided sample.

Generated or resampled workloads may have no single static dataset to summarize, which is why the
curve comes from a fixed sample. Predictions keep the result's scalar `α`, while the curve records
`Float` metrics.
-/
structure StreamResult (σ τ : Shape) (α : Type := Float) [Storage α] where
  /-- Trained model result. -/
  trained : Result σ τ α
  /-- Evaluation loss curve recorded during stream training. -/
  curve : Training.Curve

namespace StreamResult

variable {α : Type} [Storage α]

/-- One-line summary for the trained stream run. -/
def summary {σ τ : Shape} [ToString α]
    (result : StreamResult σ τ α) : String :=
  result.trained.summary

/-- Print the stream training summary. -/
def printSummary {σ τ : Shape} [ToString α]
    (result : StreamResult σ τ α) : IO Unit :=
  IO.println result.summary

/-- Predict one input, or map a leading tensor batch, through the trained stream result. -/
def predict {σ τ inputShape : Shape}
    (result : StreamResult σ τ α) (input : Tensor α inputShape)
    (batch : Bool := false) (batchSize : Nat := 1)
    [BatchInput
      (Tensor α σ) (Tensor α (Shape.dim batchSize σ)) batch (Tensor α inputShape)] :
    IO (Tensor α (match batch with | false => τ | true => Shape.dim batchSize τ)) :=
  result.trained.predict input (batch := batch) (batchSize := batchSize)

instance {σ τ : Shape} [ToString α] :
    ToString (StreamResult σ τ α) where
  toString := summary

end StreamResult

/-- Two trained supervised models and the coupled metric recorded during alternating training. -/
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

instance {σ₁ τ₁ σ₂ τ₂ : Shape} :
    ToString (AlternatingResult σ₁ τ₁ σ₂ τ₂) where
  toString := summary

end AlternatingResult

end Trainer

end TorchLean
