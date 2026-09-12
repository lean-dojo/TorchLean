/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Summary
public import NN.API.Optim.Config
public import NN.API.Neural.Builders -- shake: keep
public import NN.API.Optim -- shake: keep
public import NN.API.Trainer.Reporting -- shake: keep
public import NN.Runtime.Autograd.Model -- shake: keep
public import NN.Spec.Core.Shape -- shake: keep
public import NN.API.Module -- shake: keep
public import NN.API.Neural.State -- shake: keep
public import NN.API.Loss -- shake: keep
public import NN.API.Trainer.Summary -- shake: keep

/-!
# Trainer

Create a trainer from a checked model, choose its loss and optimizer, then train or predict:

```lean
let trainer := Trainer.new model
  { objective := .meanSquaredError
    optimizer := optim.adam { learningRate := 0.03 } }
let y0 ← trainer.predict x
let trained ← trainer.train data { steps := 200, samplesPerStep := 16, logEvery := 25 }
trained.printSummary
trained.save "model.state"
```

Inputs, targets, and predictions cross the API as `Tensor Float`. Training itself never runs in
binary64: `.native` arithmetic uses `Float32` and `.ieee` uses `IEEE32Exec`, both binary32. The
`Float` values are converted at the boundary, and `Result.report` records which scalar ran.

`trainer.open` returns a `Trainer.Session` for programs that drive the optimizer loop themselves.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

universe u

/--
Runtime, device, arithmetic, and optimizer settings for a trainer or one training call.

Example:
```lean
-- The defaults train on the CPU in Lean's `Float32`. `.ieee` swaps in the bit-level reference
-- semantics, which is the setting to reach for when a result looks like a rounding artifact.
def settings : Trainer.RunConfig :=
  { optimizer := optim.sgd { learningRate := 0.01, momentum := 0.9 }
    arithmetic := .native
    execution := .eager
    device := .cpu }
```
-/
structure RunConfig where
  /-- Optimizer used unless a training call supplies another run configuration. -/
  optimizer : optim.Optimizer := optim.sgd { learningRate := 0.01 }
  /--
  Arithmetic semantics used for the run.

  `.native` trains in Lean's `Float32` and `.ieee` in the bit-level `IEEE32Exec` reference. Both
  are binary32; the `Tensor Float` values in the public signatures are converted at the boundary.
  Supervised training rejects `.complex`.
  -/
  arithmetic : Runtime.Arithmetic := .native
  /-- Immediate tape execution or reusable typed-graph execution. -/
  execution : Runtime.ExecutionMode := .eager
  /-- Device used for execution. -/
  device : Runtime.Device := .cpu
  /-- Optional advanced override for provider, assurance, and VJP policy. -/
  backendProfile? : Option NN.Backend.BackendProfile := none
  /-- Print each accepted backend capsule when it is first used. -/
  showBackend : Bool := false

/--
Loss used to train a model.

The output shape belongs to the model. The objective only decides how `(prediction, target)`
becomes a scalar objective; it does not need a separate input-shape index.

Example:
```lean
-- Regression scores a prediction against a target tensor.
def regression : Trainer.Objective [1] := .meanSquaredError

-- Classification needs the axis the logits live on, here the only axis of a ten-class output.
def classification : Trainer.Objective [10] := .oneHotCrossEntropy 0
```
-/
inductive Objective (output : Shape) where
  /-- Mean-squared-error supervised regression. -/
  | meanSquaredError (reduction : Loss.Reduction := .mean)
  /-- One-hot cross entropy over a class or structured logit tensor. -/
  | oneHotCrossEntropy (axis : Nat)
      (reduction : Loss.Reduction := .mean)
  /-- A checked TorchLean loss program supplied by the caller. -/
  | custom
      (loss : ∀ {α : Type}, [TorchLean.Storage α] → [Context α] →
        Runtime.Autograd.Model.Program α
          [output, output] ([] : Shape))

namespace Objective

/--
Lower an objective for a checked model into the runtime module definition that the trainer
instantiates.

Training mode is the default; `.eval` builds the graph that stateful layers use for inference.
-/
def definition {σ τ : Shape} (objective : Objective τ)
    (model : TorchLean.nn.Sequential σ τ) (mode : nn.Mode := .train) :
    TorchLean.Module.ObjectiveDefinition Unit (TorchLean.nn.stateShapes model) [σ, τ] :=
  match objective with
  | .meanSquaredError reduction =>
      Runtime.Autograd.Model.Layers.Seq.Objective.meanSquaredError
        (model := model) (reduction := reduction) (mode := mode)
  | .oneHotCrossEntropy axis reduction =>
      Runtime.Autograd.Model.Layers.Seq.Objective.oneHotCrossEntropy
        (model := model) axis (reduction := reduction) (mode := mode)
  | .custom loss =>
      Runtime.Autograd.Model.Layers.Seq.Objective.fromLoss model loss mode

end Objective

/--
Model-independent options accepted by `Trainer.new`.

The `RunConfig` fields select the optimizer and runtime. Although the trainer's public signatures
use `Tensor Float`, training runs in the binary32 scalar chosen by `arithmetic`: `Float32` for
`.native` and `IEEE32Exec` for `.ieee`. Trained parameters are read back to `Float` exactly.
-/
structure Config (input output : Shape) extends RunConfig where
  /-- Training objective attached to this trainer. -/
  objective : Objective output := .meanSquaredError
  /-- Seed used when the model is still a seedable `TorchLean.nn.Builder` builder. -/
  seed : Nat := 0

end Trainer

/--
A checked model together with its loss, runtime settings, and initialization seed.

Construct trainers with `Trainer.new`. The resulting value supports prediction, training, and model
inspection directly through dot notation.
-/
structure Trainer (input output : Shape) where
  /-- Checked TorchLean model. -/
  model : TorchLean.nn.Sequential input output
  /-- Supervised objective used by `train`. -/
  objective : Trainer.Objective output
  /-- Runtime, backend-contract, and optimizer choices carried by this trainer. -/
  runtime : Trainer.RunConfig := {}
  /-- Seed used to build this trainer when the input was a `TorchLean.nn.Builder` model builder. -/
  seed : Nat := 0

namespace Trainer

/-- Structured checked-model summary for this trainer. -/
def summary {σ τ : Shape}
    (trainer : TorchLean.Trainer σ τ) : Except String nn.ModelSummary :=
  nn.summary trainer.model

/-- Print the checked-model summary under a caller-chosen heading. -/
def printSummary {σ τ : Shape}
    (trainer : TorchLean.Trainer σ τ)
    (label : String := "model") : IO Unit := do
  IO.println s!"{label}:"
  match trainer.summary with
  | .ok details => IO.println details
  | .error message => throw <| IO.userError message

end Trainer

end TorchLean
