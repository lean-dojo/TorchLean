/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Training.Reporting -- shake: keep
public import TorchLean.NN.CLI.Command -- shake: keep

/-!
# Training Command-Line Options

Command-line options shared by the runnable training programs. `RunOptions` holds the step count,
batch size, log destination and CUDA allocator sampling, and `OptimizerOptions` adds a learning
rate. Model definitions and `Trainer.train` do not depend on these records.
-/

@[expose] public section

namespace TorchLean.CLI.Training

/-- Step, batching, logging, and allocator-reporting options for a training command. -/
structure RunOptions where
  /-- Number of optimizer updates. -/
  steps : Nat
  /-- Number of in-memory samples consumed by one optimizer update. -/
  batchSize : Nat := 1
  /-- Destination for the JSON training log. -/
  logDestination : Training.LogDestination
  /--
  Number of completed steps between CUDA allocator samples. With `0`, `Trainer.Memory.cadence`
  samples every `steps / 10` steps on CUDA runs of at least 1000 steps and never otherwise.
  -/
  cudaMemorySampleEvery : Nat := 0
deriving Repr

namespace RunOptions

/--
Parse `--log`, `--steps`, `--batch-size` and `--cuda-mem-watch` and return the other arguments.
A zero batch size is an error, and so is a zero step count unless `allowZeroSteps` is set.
-/
def parse
    (exeName : String)
    (arguments : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultBatchSize : Nat := 1)
    (allowZeroSteps : Bool := false) :
    Except String (RunOptions × List String) := do
  let (logRaw?, arguments) ← CLI.takeFlagValue? arguments "log"
  let (steps, arguments) ← CLI.takeNatFlag arguments "steps" (default := defaultSteps)
  let (batchSize?, arguments) ← CLI.takeNatFlag? arguments "batch-size"
  let (cudaMemorySampleEvery?, arguments) ← CLI.takeNatFlag? arguments "cuda-mem-watch"
  if !allowZeroSteps && steps = 0 then
    throw s!"{exeName}: --steps must be > 0"
  let batchSize := batchSize?.getD defaultBatchSize
  if batchSize = 0 then
    throw s!"{exeName}: --batch-size must be > 0"
  let logDestination :=
    Training.LogDestination.resolve (.jsonFile defaultLogPath) logRaw?
  pure
    ({ steps, batchSize, logDestination,
       cudaMemorySampleEvery := cudaMemorySampleEvery?.getD 0 }, arguments)

end RunOptions

/-- Training command options that also select a learning rate. -/
structure OptimizerOptions extends RunOptions where
  /-- Learning rate passed to the command's optimizer constructor. -/
  lr : Float
deriving Repr

namespace OptimizerOptions

/-- Parse the run options, then a finite positive `--lr` that defaults to `defaultLr`. -/
def parse
    (exeName : String)
    (arguments : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (defaultBatchSize : Nat := 1)
    (allowZeroSteps : Bool := false) :
    Except String (OptimizerOptions × List String) := do
  let (run, arguments) ←
    RunOptions.parse exeName arguments defaultLogPath
      (defaultSteps := defaultSteps)
      (defaultBatchSize := defaultBatchSize)
      (allowZeroSteps := allowZeroSteps)
  let (lr, arguments) ←
    CLI.takePositiveFloatFlag arguments exeName "lr" (default := defaultLr)
  pure
    ({ toRunOptions := run, lr }, arguments)

end OptimizerOptions

end TorchLean.CLI.Training
