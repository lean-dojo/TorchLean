/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.CLI.Training
public import TorchLean.NN.Module.Command
public import TorchLean.NN.Trainer.Run

/-!
# Trainer Runtime Flags

Command-line parsing for `Trainer.RunConfig`, for applications that accept training runtime
settings. `Trainer.RunConfig.parse` reads the flags `--arithmetic`, `--device` and `--show-backend`
and returns every other argument to the caller. `Trainer.RunConfig.parseCommandLine` parses a whole
command line, and `CLI.Training.RunOptions.trainOptions` builds the options that `Trainer.train`
takes.
-/

@[expose] public section

namespace TorchLean.Trainer.RunConfig

/-- Parse the runtime flags in `arguments` over `base` and return the configuration together with
the unused arguments. A setting without a flag keeps its value from `base`.

A `--device` flag also clears the backend profile of `base`, so that profile cannot override the
device chosen on the command line. Complex arithmetic is rejected.
-/
def parse (arguments : List String) (base : Trainer.RunConfig := {}) :
    Except String (Trainer.RunConfig × List String) := do
  let (selection, remainingArguments) ←
    Module.RuntimeFlags.parse arguments base.arithmetic
  if selection.arithmetic == .complex then
    throw <|
      "TorchLean.Trainer: supervised training supports native or IEEE arithmetic; " ++
        "complex arithmetic requires an explicit complex-valued training API"
  let hasDevice := CLI.hasFlagValue arguments "device"
  let device := if hasDevice then selection.device else base.device
  let backendProfile? := if hasDevice then none else base.backendProfile?
  pure
    ({ base with
        arithmetic := selection.arithmetic
        device
        backendProfile?
        showBackend := selection.showBackend || base.showBackend },
      remainingArguments)

/-- Parse a complete command line over `base`, raising an `IO.userError` prefixed with `exeName`
for invalid or unused arguments. -/
def parseCommandLine
    (exeName : String) (arguments : List String) (base : Trainer.RunConfig := {}) :
    IO Trainer.RunConfig := do
  let (config, remainingArguments) ← CLI.orThrow exeName (parse arguments base)
  CLI.requireNoArgs exeName remainingArguments
  pure config

end TorchLean.Trainer.RunConfig

namespace TorchLean.CLI.Training.RunOptions

/-- Build the `Trainer.TrainOptions` that `Trainer.train` takes from parsed run options. The step
count, batch size, log destination and CUDA memory sampling interval come from `runOptions`.
`enableLog := false` disables the log, and `logEvery`, `logTitle` and `logNotes` pass through. -/
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
