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

/-- Run a parsed training command and finish with access to runtime flags and the result. -/
def runParsedWith {φ ρ : Type}
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (φ × List String))
    (banner : Runtime.Config → String)
    (train : Runtime.Config → φ → IO ρ)
    (finish : Runtime.Config → φ → ρ → IO Unit)
    (usage? : Option String := none) :
    IO UInt32 :=
  Module.Command.run
    (config := { banner? := some banner, usage?, printSuccess := true })
    exeName args
    (.native fun runtime rest => do
      let (flags, rest) ← CLI.orThrow exeName <| parseFlags rest
      CLI.requireNoArgs exeName rest
      let result ← train runtime flags
      finish runtime flags result)

/-- Run a parsed training command whose final printer only needs the trained result. -/
def runParsed {φ ρ : Type}
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (φ × List String))
    (banner : Runtime.Config → String)
    (train : Runtime.Config → φ → IO ρ)
    (print : ρ → IO Unit)
    (usage? : Option String := none) :
    IO UInt32 :=
  runParsedWith exeName args parseFlags banner train (fun _ _ trained => print trained) usage?

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
  defaultLearningRate : Float
  /-- Model description used in banners. -/
  description : String
  /-- Command-specific data flags, rendered by `--help`. -/
  dataOptions : Array String := #[]
  /-- Parse data flags, then leave device/training flags for the shared parser. -/
  parseData : List String → Except String (δ × List String)
  /-- Run the actual training body after data, device, and training flags have been parsed. -/
  train : Runtime.Config → δ → CLI.Training.OptimizerOptions → IO Unit

/-- Usage text for training applications using the shared runner. -/
def usage {δ : Type} (config : Config δ) : String :=
  let dataSection :=
    if config.dataOptions.isEmpty then #[]
    else #["", "Data:"] ++ config.dataOptions
  String.intercalate "\n" <|
    (#[ s!"{config.exeName}: {config.description}"
    , ""
    , "Usage:"
    , s!"  {config.exeName} [options]"
    ] ++ dataSection ++
    #[ ""
    , "Training:"
    , s!"  --steps N          optimizer updates (default: {config.defaultSteps})"
    , s!"  --lr X             learning rate (default: {config.defaultLearningRate})"
    , "  --batch-size N     dataset items accumulated per optimizer update (default: 1)"
    , "  --log PATH|false   write a TrainLog JSON, or disable logging"
    , ""
    , "Runtime:"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --execution eager|typed-graph"
    , "  --arithmetic native"
    , "  --show-backend     print backend capsules as they execute"
    ]).toList

/-- Run a public native-arithmetic training command. -/
def run {δ : Type} (config : Config δ) (args : List String) : IO UInt32 := do
  if args.contains "--help" || args.contains "-h" then
    IO.println (usage config)
    return 0
  Module.Command.run
    (config := {
      banner? := some fun runtime =>
        s!"{config.exeName}: {config.description} (device={runtime.deviceName})"
      printSuccess := true })
    config.exeName args
    (.native fun runtime rest => do
      let (dataArgs, rest) ← CLI.orThrow config.exeName <| config.parseData rest
      let (train, rest) ← CLI.orThrow config.exeName <|
        CLI.Training.OptimizerOptions.parse config.exeName rest config.defaultLogPath
          (defaultSteps := config.defaultSteps)
          (defaultLearningRate := config.defaultLearningRate)
      CLI.requireNoArgs config.exeName rest
      config.train runtime dataArgs train)

end TorchLean.CLI.Training.Command
