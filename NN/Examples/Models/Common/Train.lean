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

/-- Help text for model commands with caller-supplied data and training options. -/
def modelUsage
    (exeName : String)
    (dataOptions trainingOptions : Array String)
    (extraSections : Array String := #[]) : String :=
  String.intercalate "\n" <| (#[
    s!"Usage: scripts/lake.sh exe torchlean {exeName} [options]",
    "",
    "Data:"
  ] ++ dataOptions ++ #[
    "",
    "Training:"
  ] ++ trainingOptions ++ extraSections ++ #[
    "",
    "Runtime:",
    "  --device auto|cpu|gpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external",
    "  --arithmetic native",
    "  --seed N --show-backend"
  ]).toList

/-- Standard optimizer and logging flags accepted by most model commands. -/
def optimizerUsage (exeName : String) (dataOptions : Array String) : String :=
  modelUsage exeName dataOptions #[
    "  --steps N          optimizer updates",
    "  --batch-size N     dataset items accumulated per update",
    "  --lr X             learning rate",
    "  --log PATH|false   write a TrainLog JSON, or disable logging",
    "  --cuda-mem-watch N sample CUDA allocator state every N updates"
  ]

/-- CSV-backed training command. The caller chooses the objective and result reporting. -/
def csv {Result : Type}
    (exeName : String)
    (args : List String)
    (defaultCsv : System.FilePath)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLearningRate : Float := 1e-3)
    (banner : Runtime.Config → String)
    (train : Runtime.Config → Support.Training.Options Support.Csv.Options →
      IO Result)
    (report : Result → IO Unit) :
    IO UInt32 :=
  CLI.Training.Command.runParsed exeName args
    (fun rest =>
      Support.Training.Options.parse exeName rest defaultLogPath defaultSteps defaultLearningRate
        (parseData := fun args => Support.Csv.Options.parse args defaultCsv))
    banner train report
    (usage? := some <| optimizerUsage exeName #["  --csv PATH         supervised CSV file"])

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
    (usage? := some <| optimizerUsage exeName #[
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
    (usage? := some <| optimizerUsage exeName #[
      "  --x PATH           input-window NPY file",
      "  --y PATH           target-window NPY file",
      "  --windows N        windows to load",
      "  --report-offset N  window shown before and after training"
    ])


end NN.Examples.Models.TrainCommand
