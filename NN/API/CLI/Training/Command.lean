/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI.Trainer

/-!
# Training Application Commands

Compose an application-specific data parser and training callback with common CLI options.
-/

@[expose] public section

namespace TorchLean.CLI.Training.Command

/-- Run a parsed training command and print the trained result. -/
def runParsed {φ ρ : Type}
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (φ × List String))
    (banner : Runtime.Config → String)
    (train : Runtime.Config → φ → IO ρ)
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
Shared command runner for training applications with a caller-defined data parser.

Applications supply their data selection and training callback. The runner owns common flags,
help output, and runtime selection.
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
  /-- Parse data flags, then leave device/training flags for the shared parser. -/
  parseData : List String → Except String (δ × List String)
  /-- Run the actual training body after data, device, and training flags have been parsed. -/
  train : Runtime.Config → δ → CLI.Training.OptimizerOptions → IO Unit

/-- The training flags `CLI.Training.OptimizerOptions.parse` accepts, with defaults when given. -/
def trainingFlags (defaultSteps? : Option Nat := none)
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
    (trainingOptions : Array String := trainingFlags) (description? : Option String := none) :
    String :=
  let title := description?.elim #[] fun description => #[s!"{exeName}: {description}", ""]
  let dataSection := if dataOptions.isEmpty then #[] else #["", "Data:"] ++ dataOptions
  String.intercalate "\n" <|
    (title ++ #[s!"Usage: scripts/lake.sh exe torchlean {exeName} [options]"] ++ dataSection ++
      #["", "Training:"] ++ trainingOptions ++
      #[ ""
       , "Runtime:"
       , "  --device auto|cpu|gpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
       , "  --arithmetic native"
       , "  --seed N           runtime seed (default: 0)"
       , "  --show-backend     print backend capsules as they execute" ]).toList

/-- Usage text for training applications using the shared runner. -/
def usage {δ : Type} (config : Config δ) : String :=
  helpText config.exeName config.dataOptions
    (trainingFlags (some config.defaultSteps) (some config.defaultLr))
    (description? := some config.description)

/-- Run a public native-arithmetic training command. -/
def run {δ : Type} (config : Config δ) (args : List String) : IO UInt32 :=
  Module.Command.run
    (config := {
      banner? := some fun runtime =>
        s!"{config.exeName}: {config.description} (device={runtime.deviceName})"
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

end TorchLean.CLI.Training.Command
