/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI.Training
public import NN.API.Module.Command
public import NN.API.Trainer.Run

/-!
# Trainer Runtime Flags

Command-line parsing for `Trainer.RunConfig`. Import this module in applications that accept
training runtime settings. The flags are `--arithmetic`, `--execution`, `--device`, and
`--show-backend`; everything else is returned to the caller.
-/

@[expose] public section

namespace TorchLean.CLI.Trainer

open TorchLean

/-- Parse runtime flags over `base`, preserving unselected defaults and returning unused arguments.

An explicit device flag clears a programmatic backend profile so it cannot override the CLI choice.
-/
def parse (arguments : List String) (base : Trainer.RunConfig := {}) :
    Except String (Trainer.RunConfig × List String) := do
  let (selection, remainingArguments) ←
    TorchLean.Module.RuntimeSelection.parse arguments base.arithmetic
  if selection.arithmetic == .complex then
    throw <|
      "TorchLean.Trainer: supervised training supports native or IEEE arithmetic; " ++
        "complex arithmetic requires an explicit complex-valued training API"
  let hasFlag (name : String) :=
    arguments.any fun argument => argument == s!"--{name}" || argument.startsWith s!"--{name}="
  let device := if hasFlag "device" then selection.device else base.device
  let backendProfile? := if hasFlag "device" then none else base.backendProfile?
  if backendProfile?.isNone && (NN.Backend.BackendProfile.maintainedForDevice? device).isNone then
    throw <|
      s!"device `{device.cliName}` has no maintained runtime profile; " ++
        "use a programmatic backend profile"
  pure
    ({ base with
        arithmetic := selection.arithmetic
        execution := if hasFlag "execution" then selection.execution else base.execution
        device
        backendProfile?
        showBackend := selection.showBackend || base.showBackend },
      remainingArguments)

/-- Parse a complete command line, raising an `IO.userError` for invalid or unused arguments. -/
def parseCommandLine
    (exeName : String) (arguments : List String) (base : Trainer.RunConfig := {}) :
    IO Trainer.RunConfig := do
  let (config, remainingArguments) ← CLI.orThrow exeName (parse arguments base)
  CLI.requireNoArgs exeName remainingArguments
  pure config

/-- Render a run configuration as the command-line arguments `parse` accepts. -/
def cliArguments (run : Trainer.RunConfig) : List String :=
  ["--arithmetic", run.arithmetic.cliName] ++
  ["--execution", Runtime.ExecutionMode.cliName run.execution] ++
  ["--device", run.device.cliName] ++
  (if run.showBackend then ["--show-backend"] else [])

end TorchLean.CLI.Trainer

namespace TorchLean.CLI.Training.RunOptions

/-- Training options consumed by `Trainer.train`. -/
def trainOptions
    (runOptions : CLI.Training.RunOptions)
    (enableLog : Bool := true)
    (logEvery : Nat := 0)
    (logTitle : String := "Training")
    (logNotes : Array String := #[]) :
    Trainer.TrainOptions :=
  { steps := runOptions.steps
    samplesPerStep := runOptions.batchSize
    logDestination := if enableLog then runOptions.logDestination else .disabled
    logEvery
    cudaMemorySampleEvery := runOptions.cudaMemorySampleEvery
    logTitle
    logNotes }

end TorchLean.CLI.Training.RunOptions

namespace TorchLean.CLI.Training.OptimizerOptions

/-- Training options consumed by `Trainer.train`. -/
def trainOptions
    (optimizerOptions : CLI.Training.OptimizerOptions)
    (enableLog : Bool := true)
    (logEvery : Nat := 0)
    (logTitle : String := "Training")
    (logNotes : Array String := #[]) :
    Trainer.TrainOptions :=
  optimizerOptions.toRunOptions.trainOptions enableLog logEvery logTitle logNotes

end TorchLean.CLI.Training.OptimizerOptions
