/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --auto-mpg
  scripts/lake.sh exe torchlean kan --device cpu --steps 20
  scripts/lake.sh -Kcuda=true exe torchlean kan --device cuda --steps 20
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLean.NN.Models.KAN
public import TorchLeanExamples.Models.Training

/-!
# KAN Regression

The `kan` command trains a small Kolmogorov-Arnold Network on the prepared Auto MPG regression
CSV. The downloader normalizes every column to $[0,1]$, so the piecewise-linear KAN basis uses
$\mathtt{inputScale}=\mathtt{gridSize}-1$ to spread its knots across the data interval.

`nn.kan` builds the model, and the trainer chooses the task:

```lean
let trainer := Trainer.create model
  { objective := mse, optimizer := optim.adam { lr := 1e-3 } }
```

The edge basis is an ordinary config field. The example uses triangular piecewise-linear hats,
and a spline, polynomial or rational family would plug into the same `nn.KAN.EdgeFamily`
slot as a basis size plus a TorchLean model from `[inputWidth]` to `[inputWidth * basisSize]`.
-/

@[expose] public section

open TorchLean.Trainer.Objective

namespace TorchLean.Examples.KAN

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "kan"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "kan"

/-- Static minibatch size for the Auto MPG tabular loader. -/
def batchSize : Nat := 5

/-- Auto MPG has seven normalized numeric predictors after dropping the car name. -/
def inputWidth : Nat := 7

/-- Scalar regression target, the normalized miles per gallon. -/
def outputWidth : Nat := 1

/-- One KAN layer from the seven features to the target, with triangular bases on four knots. -/
abbrev modelConfig : nn.KAN.Config :=
  { inputWidth := inputWidth
    hiddenWidths := []
    outputWidth := outputWidth
    edge := nn.PiecewiseLinear.edgeFamily { gridSize := 4, inputScale := 3 } }

/-- Input shape for one batch, derived from the KAN config so the two cannot disagree. -/
abbrev input := modelConfig.inputShape [batchSize]
/-- Output shape for one batch. -/
abbrev output := modelConfig.outputShape [batchSize]

/-- The KAN model. The `Trainer` objective decides whether it does regression or classification. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.kan modelConfig [batchSize]

/-- The prepared Auto MPG CSV as shuffled batches of five rows, header skipped. -/
def data (path : System.FilePath) (seed : Nat) : Trainer.Dataset input output :=
  Data.fromCSV path batchSize inputWidth outputWidth
    (csvOptions := { skipHeader := true })
    (shuffle := true) (seed := seed)

/-- Train the KAN with Adam on mean squared error. -/
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
      (logTitle := "KAN Auto MPG regression")
      (logNotes := #[s!"data={flags.data.path}", s!"batch={batchSize}",
        s!"edge={modelConfig.edge.name}"]))

/-- CLI entry point for Auto MPG regression, defaulting to 20 steps at learning rate `1e-2`. -/
def main (args : List String) : IO UInt32 :=
  CSV.train exeName args
    Examples.RealData.autoMPGCsv defaultLogPath 20 1e-2
    (Training.banner exeName "Auto MPG KAN regression")
    train (fun result => result.printSummary)

end TorchLean.Examples.KAN
