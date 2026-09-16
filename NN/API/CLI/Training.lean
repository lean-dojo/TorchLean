/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Reporting -- shake: keep
public import NN.API.CLI -- shake: keep

/-!
# Training Command-Line Options

Command-line options shared by runnable training programs. These records describe process-level
choices such as step counts, log destinations, and CUDA allocator sampling. Model definitions and
the `Trainer.train` API do not depend on them.
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
  logDestination : Runtime.Training.LogDestination
  /-- Number of completed steps between CUDA allocator samples; `0` selects the default policy. -/
  cudaMemorySampleEvery : Nat := 0
deriving Repr

namespace RunOptions

/-- Parse the common options accepted by runnable training commands. -/
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
    Runtime.Training.LogDestination.resolve (.json defaultLogPath) logRaw?
  pure
    ({ steps, batchSize, logDestination,
       cudaMemorySampleEvery := cudaMemorySampleEvery?.getD 0 }, arguments)

end RunOptions

/-- Training command options that also select a learning rate. -/
structure OptimizerOptions where
  /-- Number of optimizer updates. -/
  steps : Nat
  /-- Number of in-memory samples consumed by one optimizer update. -/
  batchSize : Nat := 1
  /-- Destination for the JSON training log. -/
  logDestination : Runtime.Training.LogDestination
  /-- Number of completed steps between CUDA allocator samples; `0` selects the default policy. -/
  cudaMemorySampleEvery : Nat := 0
  /-- Learning rate passed to the command's optimizer constructor. -/
  learningRate : Float
deriving Repr

namespace OptimizerOptions

/-- Step, batching, and logging settings without the optimizer learning rate. -/
def toRunOptions (options : OptimizerOptions) : RunOptions :=
  { steps := options.steps
    batchSize := options.batchSize
    logDestination := options.logDestination
    cudaMemorySampleEvery := options.cudaMemorySampleEvery }

/-- Parse run options followed by a positive `--lr` value. -/
def parse
    (exeName : String)
    (arguments : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLearningRate : Float := 1e-3)
    (defaultBatchSize : Nat := 1)
    (allowZeroSteps : Bool := false) :
    Except String (OptimizerOptions × List String) := do
  let (run, arguments) ←
    RunOptions.parse exeName arguments defaultLogPath
      (defaultSteps := defaultSteps)
      (defaultBatchSize := defaultBatchSize)
      (allowZeroSteps := allowZeroSteps)
  let (learningRate, arguments) ←
    CLI.takePositiveFloatFlag arguments exeName "lr" (default := defaultLearningRate)
  pure
    ({ steps := run.steps
       batchSize := run.batchSize
       logDestination := run.logDestination
       cudaMemorySampleEvery := run.cudaMemorySampleEvery
       learningRate },
     arguments)

end OptimizerOptions

end TorchLean.CLI.Training
