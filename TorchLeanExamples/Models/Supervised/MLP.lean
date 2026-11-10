/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --auto-mpg
  scripts/lake.sh exe torchlean mlp --device cpu
  scripts/lake.sh -Kcuda=true exe torchlean mlp --device cuda
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLeanExamples.Models.Training

/-!
# MLP Tabular Regression

The `mlp` command trains an MLP on the UCI Auto MPG regression task. The prepared CSV holds seven
numeric car features and the miles-per-gallon target, each min-max normalized to $[0,1]$, and the
task is ordinary supervised tabular regression:

`x1..x7 -> Linear -> ReLU -> Linear -> y`.

Prepare the CSV once:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
scripts/lake.sh exe torchlean mlp --device cpu --steps 1
```

The downloader writes a header row `x1..x7,y` above the data. To try another tabular regression
CSV, pass `--csv PATH` with a header row and the same eight columns, seven features and one target.
-/

@[expose] public section

open TorchLean.Trainer.Objective

namespace TorchLean.Examples.MLP

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "mlp"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "mlp"

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
Auto MPG as a TorchLean dataset.

Only the CSV path, the header row, the batch size and the column counts are specific to Auto MPG.
The rows are shuffled with `seed` and grouped into batches of five. The trainer picks the device
and the arithmetic, native or IEEE-reference binary32, and converts the rows when it materializes
the dataset, so one dataset value serves CPU and CUDA runs.
-/
def data (path : System.FilePath) (seed : Nat) :
    Trainer.Dataset input output :=
  Data.fromCSV path batchSize inputWidth outputWidth
    (csvOptions := { skipHeader := true })
    (shuffle := true) (seed := seed)

/-- Train the Auto MPG MLP with Adam on the mean squared error. -/
def train (runtime : Session.Config) (flags : Training.Options CSV.Options) :
    IO (Trainer.Result input output) := do
  Data.requireFile exeName "CSV dataset" flags.data.path RealData.missingAutoMPGHint
  let trainer :=
    Trainer.create model <|
      { runtime with
          optimizer := optim.adam { lr := flags.training.lr }
          objective := mse }
  trainer.train
    (data flags.data.path runtime.seed)
    (flags.training.trainOptions
      (logTitle := "MLP tabular training")
      (logNotes := #[s!"dataset={flags.data.path}", s!"batch={batchSize}"]))

/-- Auto MPG CLI entrypoint for CPU or CUDA, by default one step at learning rate `1e-3`. -/
def main (args : List String) : IO UInt32 :=
  CSV.train exeName args
    Examples.RealData.autoMPGCsv defaultLogPath 1 1e-3
    (Training.banner exeName "Auto MPG MLP regression")
    train (fun result => result.printSummary)

end TorchLean.Examples.MLP
