/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Core
public import NN.API.Trainer.Scheduler
public import NN.API.Trainer.Dataset -- shake: keep

/-!
# Training Configuration

Probes, run-configuration helpers, and per-training options for the trainer API. Command-line
parsing is available separately from `NN.API.CLI.Trainer`, keeping process flags out of these
configuration records.
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

namespace RunConfig

/-- Override the execution device using a maintained backend profile. -/
def withDevice (run : RunConfig) (device : Runtime.Device) : Except String RunConfig := do
  match NN.Backend.BackendProfile.maintainedForDevice? device with
  | some _ => pure { run with device := device, backendProfile? := none }
  | none =>
      throw <|
        s!"device `{device.cliName}` has no maintained runtime profile; " ++
          "provide an explicit backend profile"

/--
Select a complete backend contract profile.

The profile carries the device, provider preference, assurance policy, VJP ownership, and capsule
registry together. It can select, for example, LibTorch forward execution with a TorchLean-owned
backward pass.
-/
def withBackendProfile (run : RunConfig) (profile : NN.Backend.BackendProfile) : RunConfig :=
  { run with device := profile.policy.device, backendProfile? := some profile }

/-- Enable or disable first-use backend capsule reporting. -/
def withBackendReport (run : RunConfig) (enabled : Bool := true) : RunConfig :=
  { run with showBackend := enabled }

/-- Apply runtime execution settings to a persistent trainer run configuration. -/
def withRuntime (run : RunConfig) (runtime : Runtime.Config) : RunConfig :=
  { run with
      execution := runtime.execution
      device := runtime.device
      backendProfile? := runtime.backendProfile?
      showBackend := runtime.showBackend }

/-- Build a trainer run configuration from a runtime configuration and trainer choices. -/
def fromRuntime (runtime : Runtime.Config) (base : RunConfig := {}) : RunConfig :=
  base.withRuntime runtime

/-- Execution settings carried by this trainer configuration. -/
def executionSettings (run : RunConfig) : Runtime.Config :=
  { execution := run.execution
    device := run.device
    backendProfile? := run.backendProfile?
    showBackend := run.showBackend }

/-- Attach a training objective and initialization seed to these run settings. -/
def forObjective {σ τ : Shape}
    (run : RunConfig)
    (objective : Objective τ := .meanSquaredError)
    (seed : Nat := 0) :
    Config σ τ :=
  { run with objective := objective, seed := seed }

end RunConfig

/--
Per-training-call options for the trainer API.

`steps` has no default so that `trainer.train data {}` cannot silently perform a single update.

Example:
```lean
-- `steps` has no default on purpose: how long to train is not a library's decision.
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
  parameter point; this is gradient accumulation, not a vectorized minibatch. To run a vectorized
  minibatch, give the model an explicit batch axis and build the dataset with `Data.batch`, whose
  items are already fixed-size tensor minibatches, then keep this option at `1`.
  -/
  samplesPerStep : Nat := 1
  /-- Optional learning-rate schedule, indexed by completed optimizer updates. -/
  scheduler : Option TorchLean.Trainer.Scheduler.Config := none
  /-- Print step losses every `logEvery` updates; `0` disables stdout step logging. -/
  logEvery : Nat := 0
  /-- Sample CUDA allocator state every this many completed updates; `0` disables sampling. -/
  cudaMemorySampleEvery : Nat := 0
  /-- Optional TrainLog artifact destination. Use `.disabled` for stdout-only runs. -/
  logDestination : Training.LogDestination := .disabled
  /-- Title used when writing a TrainLog artifact. -/
  logTitle : String := "Training"
  /-- Free-form notes attached to the TrainLog artifact. -/
  logNotes : Array String := #[]
  /-- Optional model-state checkpoint loaded before training; optimizer and schedule start fresh. -/
  loadCheckpoint? : Option System.FilePath := none
  /-- Optional model-state checkpoint written after training. -/
  saveCheckpoint? : Option System.FilePath := none

namespace TrainOptions

/-- Reject option combinations that cannot describe a training run. -/
def validate (options : TrainOptions) : Except String Unit := do
  unless options.samplesPerStep > 0 do
    throw "training: samplesPerStep must be positive"

end TrainOptions

end Trainer

end TorchLean
