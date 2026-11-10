/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.CLI.Trainer

/-!
# Training Application Commands

A training application describes itself in a `Config`: defaults, a data-flag parser and a training
callback. `run` adds the shared runtime and training flags, the help text and the banner, and
`runParsed` is the variant for commands that parse their own training flags.
-/

@[expose] public section

namespace TorchLean.CLI.Training

/-- Run a training command that reads its own flags with `parseFlags` and prints the result. -/
def runParsed {φ ρ : Type}
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (φ × List String))
    (banner : Session.Config → String)
    (train : Session.Config → φ → IO ρ)
    (print : ρ → IO Unit)
    (usage? : Option String := none) :
    IO UInt32 :=
  Module.Command.run
    (config := { banner? := some banner, usage?, printSuccess := true })
    exeName args
    (fun runtime rest => do
      let (flags, rest) ← CLI.orThrow exeName <| parseFlags rest
      CLI.requireNoArgs exeName rest
      print (← train runtime flags))

/--
Configuration of a training command for the shared runner `run`.

The application supplies its defaults, a data-flag parser and the training callback. The runner
handles the runtime and training flags, the help output and the banner.
-/
structure Config (δ : Type) where
  /-- CLI subcommand name, for example `rnn`. -/
  exeName : String
  /-- Default JSON log path used when `--log` is omitted. -/
  defaultLogPath : System.FilePath
  /-- Default number of optimizer steps when `--steps` is omitted. -/
  defaultSteps : Nat
  /-- Learning rate used when `--lr` is omitted. -/
  defaultLr : Float
  /-- Model description used in banners. -/
  description : String
  /-- Command-specific data flags, rendered by `--help`. -/
  dataOptions : Array String := #[]
  /-- Parse the data flags and return the rest, which still holds the training flags. -/
  parseData : List String → Except String (δ × List String)
  /-- Training body, called once the runtime, data and training flags are parsed. -/
  train : Session.Config → δ → CLI.Training.OptimizerOptions → IO Unit

/-- The training flags `CLI.Training.OptimizerOptions.parse` accepts, with defaults when given. -/
def flags (defaultSteps? : Option Nat := none)
    (defaultLr? : Option Float := none) : Array String :=
  let steps := defaultSteps?.elim "" fun steps => s!" (default: {steps})"
  let rate := defaultLr?.elim "" fun rate => s!" (default: {rate})"
  #[ s!"  --steps N          optimizer updates{steps}"
   , "  --batch-size N     dataset items accumulated per update (default: 1)"
   , s!"  --lr X             learning rate{rate}"
   , "  --log PATH|false   write a TrainLog JSON, or disable logging"
   , "  --cuda-mem-watch N sample CUDA allocator state every N updates" ]

/--
Help text for a training command: its usage line, its data flags, the training flags, and the
runtime flags the shared runner parses. Every training command prints this layout.
-/
def helpText (exeName : String) (dataOptions : Array String)
    (trainingOptions : Array String := flags) (description? : Option String := none) :
    String :=
  let title := description?.elim #[] fun description => #[s!"{exeName}: {description}", ""]
  let dataSection := if dataOptions.isEmpty then #[] else #["", "Data:"] ++ dataOptions
  String.intercalate "\n" <|
    (title ++ #[s!"Usage: scripts/lake.sh exe torchlean {exeName} [options]"] ++ dataSection ++
      #["", "Training:"] ++ trainingOptions ++
      #[ ""
       , "Runtime:"
       , s!"  --device auto|{Backend.Device.help}"
       , "  --arithmetic native"
       , "  --seed N           runtime seed (default: 0)"
       , "  --show-backend     print backend capsules as they are selected" ]).toList

/-- Help text for `config`, showing its default step count and learning rate. -/
def usage {δ : Type} (config : Config δ) : String :=
  helpText config.exeName config.dataOptions
    (flags (some config.defaultSteps) (some config.defaultLr))
    (description? := some config.description)

/--
Run the training command that `config` describes, in native arithmetic only. After the runtime
flags, `parseData` reads the data flags and `CLI.Training.OptimizerOptions.parse` the training
flags, and a leftover argument is an error.
-/
def run {δ : Type} (config : Config δ) (args : List String) : IO UInt32 :=
  Module.Command.run
    (config := {
      banner? := some fun runtime =>
        s!"{config.exeName}: {config.description} (device={runtime.device.cliName})"
      usage? := some (usage config)
      printSuccess := true })
    config.exeName args
    (fun runtime rest => do
      let (dataArgs, rest) ← CLI.orThrow config.exeName <| config.parseData rest
      let (train, rest) ← CLI.orThrow config.exeName <|
        CLI.Training.OptimizerOptions.parse config.exeName rest config.defaultLogPath
          (defaultSteps := config.defaultSteps)
          (defaultLr := config.defaultLr)
      CLI.requireNoArgs config.exeName rest
      config.train runtime dataArgs train)

end TorchLean.CLI.Training
