/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --auto-mpg
  lake exe torchlean mlp --device cpu
  lake -R -K cuda=true exe torchlean mlp --device cuda
-/

module

public import NN.Examples.Models.Common

/-!
# MLP Tabular Regression

This example trains an MLP on the UCI Auto MPG regression task. The prepared CSV has seven
normalized numeric car features and one normalized target column for miles per gallon, so the model
is just ordinary supervised tabular regression:

`x1..x7 -> Linear -> ReLU -> Linear -> y`.

Prepare the CSV once:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
lake exe torchlean mlp --device cpu --steps 1
```

The downloader writes normalized columns `x1..x7,y`. If you want to try your own tabular regression
CSV, pass `--csv PATH` with the same columns.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Supervised.Mlp

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "mlp"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "mlp"

/-- Static minibatch size for the Auto MPG tabular loader. -/
def batchSize : Nat := 5

/-- Auto MPG has seven numeric predictors after dropping `car_name`. -/
def inputWidth : Nat := 7

/-- Hidden width of the one-hidden-layer MLP. -/
def hiddenWidth : Nat := 32

/-- Regression target width: normalized miles-per-gallon. -/
def outputWidth : Nat := 1

/-- Input shape: a minibatch of Auto MPG feature vectors. -/
abbrev input : Shape := [batchSize, inputWidth]

/-- Output shape: one scalar regression prediction per row. -/
abbrev output : Shape := [batchSize, outputWidth]

/-- One-hidden-layer ReLU MLP from the public block API. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.mlp inputWidth outputWidth { hiddenWidths := [hiddenWidth] } [batchSize]

/--
Auto MPG as a public TorchLean dataset.

The only dataset-specific details here are the CSV path, header convention, batch size, and feature
count. Runtime arithmetic selection stays inside `Trainer`, so the same dataset works for CPU,
CUDA, typed graph, eager, and IEEE-reference modes.
-/
def data (path : System.FilePath) (seed : Nat) :
    Trainer.Dataset input output :=
  Data.fromCsv path batchSize inputWidth outputWidth
    (csvOptions := { skipHeader := true })
    (shuffle := true) (seed := seed)

/-- Train the Auto MPG MLP with the public `Trainer` surface. -/
def train (runtime : Runtime.Config) (flags : Support.CsvTrainFlags) :
    IO (Trainer.Result input output) := do
  Data.requireFile exeName "CSV dataset" flags.csvPath RealData.missingAutoMpgHint
  let trainer :=
    Trainer.new model <|
      Trainer.RunConfig.forObjective
        (Trainer.RunConfig.fromRuntime runtime
          { optimizer := optim.adam { learningRate := flags.training.learningRate } })
        .mse
        (seed := flags.seed)
  trainer.train
    (data flags.csvPath flags.seed)
    (flags.training.trainOptions
      (logTitle := "MLP tabular training")
      (logNotes := #[s!"dataset={flags.csvPath}", s!"lr={flags.training.learningRate}",
        s!"steps={flags.training.steps}", s!"batch={batchSize}"]))

/-- CLI entrypoint for Auto MPG regression on CPU or CUDA. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionCsv exeName args
    NN.Examples.Data.RealPaths.autoMpgCsv defaultLogPath 1 1e-3
    (Support.bannerWithDevice exeName "Auto MPG MLP regression")
    train

end NN.Examples.Models.Supervised.Mlp
