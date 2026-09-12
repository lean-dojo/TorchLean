/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.API.CLI.Training.Command
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
    s!"Usage: lake exe torchlean {exeName} [options]",
    "",
    "Data:"
  ] ++ dataOptions ++ #[
    "",
    "Training:"
  ] ++ trainingOptions ++ extraSections ++ #[
    "",
    "Runtime:",
    "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external",
    "  --execution eager|typed-graph",
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

/-- CSV-backed regression command using the public trainer API. -/
def regressionCsv {σ τ : Shape}
    (exeName : String)
    (args : List String)
    (defaultCsv : System.FilePath)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLearningRate : Float := 1e-3)
    (banner : Runtime.Config → String)
    (train : Runtime.Config → Support.CsvTrainFlags →
      IO (Trainer.Result σ τ)) :
    IO UInt32 :=
  CLI.Training.Command.runParsed exeName args
    (fun rest =>
      Support.parseCsvTrainFlags
        exeName rest defaultCsv defaultLogPath defaultSteps defaultLearningRate)
    banner (fun runtime flags => train runtime { flags with seed := runtime.seed })
    (fun result => result.printSummary)
    (usage? := some <| optimizerUsage exeName #["  --csv PATH         supervised CSV file"])

/-- NPY-backed classifier command using the public trainer API. -/
def classificationNpy
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (Support.NpyModelTrainFlags × List String))
    (banner : Runtime.Config → String)
    (train : Runtime.Config → Support.NpyModelTrainFlags →
      IO Trainer.Report) :
    IO UInt32 :=
  CLI.Training.Command.runParsed exeName args parseFlags banner (fun runtime flags => train runtime
      { flags with data := { flags.data with seed := runtime.seed } })
    (fun report => report.printSummary)
    (usage? := some <| optimizerUsage exeName #[
      "  --x PATH           feature/image NPY file",
      "  --y PATH           class-label NPY file",
      "  --n-total N        rows to load"
    ])

/-- NPY-backed regression command using the public trainer API. -/
def regressionNpy {σ τ : Shape}
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (Support.NpyModelTrainFlags × List String))
    (banner : Runtime.Config → String)
    (train : Runtime.Config → Support.NpyModelTrainFlags →
      IO (Trainer.Result σ τ)) :
    IO UInt32 :=
  CLI.Training.Command.runParsed exeName args parseFlags banner (fun runtime flags => train runtime
      { flags with data := { flags.data with seed := runtime.seed } })
    (fun result => result.printSummary)
    (usage? := some <| optimizerUsage exeName #[
      "  --x PATH           feature/image NPY file",
      "  --y PATH           target NPY file",
      "  --n-total N        rows to load"
    ])

/-- Forecast-window regression command using the public trainer API. -/
def forecastWindow {σ τ : Shape}
    (exeName : String)
    (args : List String)
    (parseFlags :
      List String → Except String (Support.ForecastWindowModelTrainFlags × List String))
    (banner : Runtime.Config → String)
    (train : Runtime.Config → Support.ForecastWindowModelTrainFlags →
      IO (Trainer.Result σ τ)) :
    IO UInt32 :=
  CLI.Training.Command.runParsed exeName args parseFlags banner (fun runtime flags => train runtime
      { flags with data := { flags.data with seed := runtime.seed } })
    (fun result => result.printSummary)
    (usage? := some <| optimizerUsage exeName #[
      "  --x PATH           input-window NPY file",
      "  --y PATH           target-window NPY file",
      "  --windows N        windows to load",
      "  --report-offset N  window shown before and after training"
    ])


end NN.Examples.Models.TrainCommand
