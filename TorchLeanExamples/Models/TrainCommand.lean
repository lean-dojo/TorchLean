/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.CLI.Training.Command
public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.NN.Data.Source
public import TorchLeanExamples.Models.Options

/-!
# Shared Model Training Commands

Command runners for the built-in example models, one each for a CSV file, paired NPY tensors and
forecasting windows. The trainer API supplies `Trainer.create`, `trainer.train` and trained
prediction handles. A runner adds the command line around them: help text, the seed and runtime
flags, the example's data flags, a banner, the caller's training callback and its result report.
-/

@[expose] public section

namespace TorchLean.Examples

/-- Training command over a supervised CSV file (`--csv PATH`). The caller's `train` picks the
objective, and `report` prints the result. -/
def CSV.train {Result : Type}
    (exeName : String)
    (args : List String)
    (defaultCsv : System.FilePath)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (banner : Session.Config → String)
    (train : Session.Config → Training.Options Options →
      IO Result)
    (report : Result → IO Unit) :
    IO UInt32 :=
  CLI.Training.runParsed exeName args
    (fun rest =>
      Training.Options.parse exeName rest defaultLogPath defaultSteps defaultLr
        (parseData := fun args => Options.parse args defaultCsv))
    banner train report
    (usage? := some <|
      CLI.Training.helpText exeName #["  --csv PATH         supervised CSV file"])

/-- Training command over paired NPY tensors (`--x`, `--y`, `--n-total`), read by the caller's
`parseFlags`. -/
def NPY.train {Result : Type}
    (exeName : String)
    (args : List String)
    (parseFlags : List String →
      Except String (Training.Options Options × List String))
    (banner : Session.Config → String)
    (train : Session.Config → Training.Options Options →
      IO Result)
    (report : Result → IO Unit)
    (target : String := "target") :
    IO UInt32 :=
  CLI.Training.runParsed exeName args parseFlags banner train report
    (usage? := some <| CLI.Training.helpText exeName #[
      "  --x PATH           feature/image NPY file",
      s!"  --y PATH           {target} NPY file",
      "  --n-total N        rows to load"
    ])

/-- Training command over NPY input and target windows (`--x`, `--y`, `--windows`,
`--report-offset`), read by the caller's `parseFlags`. -/
def Forecast.train {Result : Type}
    (exeName : String)
    (args : List String)
    (parseFlags :
      List String → Except String
        (Training.Options Forecast.Options × List String))
    (banner : Session.Config → String)
    (train : Session.Config → Training.Options Forecast.Options →
      IO Result)
    (report : Result → IO Unit) :
    IO UInt32 :=
  CLI.Training.runParsed exeName args parseFlags banner train report
    (usage? := some <| CLI.Training.helpText exeName #[
      "  --data-dir PATH    directory of the default --x and --y files",
      "  --x PATH           input-window NPY file",
      "  --y PATH           target-window NPY file",
      "  --windows N        windows to load",
      "  --report-offset N  window shown before and after training"
    ])


end TorchLean.Examples
