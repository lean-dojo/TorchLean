/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Trainer
public import TorchLean.NN.Trainer.Scheduler

/-!
# Training Configuration

Probes and per-call options for the trainer API. A `Probe` names an input whose prediction is
printed before and after training, and `TrainOptions` holds the step count, gradient accumulation,
schedule, logging, memory-sampling and checkpoint settings of one training call. Command-line
parsing lives in `TorchLean.NN.CLI.Trainer`, which keeps process flags out of these records.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/-- A small input probe evaluated at the start and end of training. -/
structure Probe (σ : Shape) where
  /-- Human-facing probe name. -/
  name : String
  /-- Human-facing input description. -/
  inputText : String := ""
  /-- Concrete input converted to the selected runtime arithmetic when the probe runs. -/
  input : Tensor Float σ
  /-- Optional expected value shown beside the prediction. -/
  expected : Option String := none

namespace Probe

/--
Build a named prediction probe from a tensor.

Example:
```lean
-- Probes print a named prediction before and after the run, so training shows its movement
-- without a separate evaluation script.
def probes : Array (Trainer.Probe [2]) :=
  #[Trainer.Probe.tensor "heldout" [0.25, -0.75] (inputText := "x = (0.25, -0.75)")]
```
-/
def tensor {σ : Shape} (name : String) (input : Tensor Float σ)
    (inputText : String := "") (expected : Option String := none) :
    Probe σ :=
  { name := name
    inputText := inputText
    input := input
    expected := expected }

end Probe


/--
Per-training-call options for the trainer API.

`steps` has no default so that `trainer.train data {}` cannot silently perform a single update.

Example:
```lean
-- 200 updates, a loss line every 25 of them, and the trained state saved at the end.
def options : Trainer.TrainOptions :=
  { steps := 200
    logEvery := 25
    saveCheckpoint? := some "checkpoints/mlp.state" }
```
-/
structure TrainOptions where
  /-- Number of optimizer updates. -/
  steps : Nat
  /--
  Number of dataset items whose gradients are accumulated into one optimizer update.

  The items are processed one after another and their gradients are averaged at the same
  parameter point, which is gradient accumulation. For a vectorized minibatch, give the model a
  batch axis, build the dataset with `Data.batch` (whose items are fixed-size tensor minibatches)
  and keep this option at `1`.
  -/
  samplesPerStep : Nat := 1
  /-- Optional learning-rate schedule, indexed by completed optimizer updates. -/
  scheduler : Option TorchLean.Trainer.Scheduler.Config := none
  /-- Print the step loss to stdout every `logEvery` updates, or never when it is `0`. -/
  logEvery : Nat := 0
  /-- CUDA allocator sampling cadence. `0` uses automatic sampling for runs of at least 1000
  updates and disables it for shorter runs. Non-CUDA runs do not sample the allocator. -/
  cudaMemorySampleEvery : Nat := 0
  /-- Optional TrainLog artifact destination. Use `.disabled` for stdout-only runs. -/
  logDestination : Training.LogDestination := .disabled
  /-- Title used when writing a TrainLog artifact. -/
  logTitle : String := "Training"
  /-- Free-form notes attached to the TrainLog artifact. -/
  logNotes : Array String := #[]
  /-- Optional model-state checkpoint loaded before training. The optimizer and schedule start
  fresh. -/
  loadCheckpoint? : Option System.FilePath := none
  /-- Optional model-state checkpoint written after training. -/
  saveCheckpoint? : Option System.FilePath := none
  /--
  Measure and print the evaluation-mode mean loss over the whole dataset before and after
  training. Each measurement is a forward pass over every sample, so a large streamed dataset may
  turn it off, and the report then records `NaN` for both losses.
  -/
  reportLoss : Bool := true

namespace TrainOptions

/-- Reject a zero `samplesPerStep`, which cannot describe a training run. -/
def validate (options : TrainOptions) : Except String Unit := do
  unless options.samplesPerStep > 0 do
    throw "training: samplesPerStep must be positive"

end TrainOptions

end Trainer

end TorchLean
