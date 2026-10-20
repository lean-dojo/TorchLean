/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Support

/-!
# Shared Model Training Commands

Example-side command runners for built-in runnable models.

The trainer API provides `Trainer.new`, `trainer.train`, and trained prediction handles. This
file owns repository command plumbing: parse example flags, check local files, run training, and
print the standard summary.
-/

@[expose] public section

namespace NN.Examples.Models.TrainCommand

open TorchLean

/-- CSV-backed training command. The caller chooses the objective and result reporting. -/
def csv {Result : Type}
    (exeName : String)
    (args : List String)
    (defaultCsv : System.FilePath)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (banner : Runtime.Config → String)
    (train : Runtime.Config → Support.Training.Options Support.Csv.Options →
      IO Result)
    (report : Result → IO Unit) :
    IO UInt32 :=
  CLI.Training.Command.runParsed exeName args
    (fun rest =>
      Support.Training.Options.parse exeName rest defaultLogPath defaultSteps defaultLr
        (parseData := fun args => Support.Csv.Options.parse args defaultCsv))
    banner train report
    (usage? := some <|
      CLI.Training.Command.helpText exeName #["  --csv PATH         supervised CSV file"])

/-- NPY-backed training command. The caller chooses the objective and result reporting. -/
def npy {Result : Type}
    (exeName : String)
    (args : List String)
    (parseFlags : List String →
      Except String (Support.Training.Options Support.Npy.Options × List String))
    (banner : Runtime.Config → String)
    (train : Runtime.Config → Support.Training.Options Support.Npy.Options →
      IO Result)
    (report : Result → IO Unit)
    (target : String := "target") :
    IO UInt32 :=
  CLI.Training.Command.runParsed exeName args parseFlags banner train report
    (usage? := some <| CLI.Training.Command.helpText exeName #[
      "  --x PATH           feature/image NPY file",
      s!"  --y PATH           {target} NPY file",
      "  --n-total N        rows to load"
    ])

/-- Forecasting command with caller-supplied training and result reporting. -/
def forecast {Result : Type}
    (exeName : String)
    (args : List String)
    (parseFlags :
      List String → Except String
        (Support.Training.Options Support.Forecast.Options × List String))
    (banner : Runtime.Config → String)
    (train : Runtime.Config → Support.Training.Options Support.Forecast.Options →
      IO Result)
    (report : Result → IO Unit) :
    IO UInt32 :=
  CLI.Training.Command.runParsed exeName args parseFlags banner train report
    (usage? := some <| CLI.Training.Command.helpText exeName #[
      "  --x PATH           input-window NPY file",
      "  --y PATH           target-window NPY file",
      "  --windows N        windows to load",
      "  --report-offset N  window shown before and after training"
    ])


end NN.Examples.Models.TrainCommand
