/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --auto-mpg
  lake exe torchlean kan --device cpu --steps 20
  lake -R -K cuda=true exe torchlean kan --device cuda --steps 20
-/

module

public import NN.Examples.Models.Common

/-!
# KAN Regression

This example trains a small Kolmogorov-Arnold Network on the prepared Auto MPG tabular-regression
CSV. The downloader normalizes the columns to $[0,1]$, so the piecewise-linear KAN basis uses
$\mathtt{inputScale}=\mathtt{gridSize}-1$ to spread its knots across the data interval.

`KAN` is a model constructor. The task is chosen by the general trainer API:

```lean
let trainer := Trainer.new model
  { objective := .mse, optimizer := optim.adam { learningRate := 1e-3 } }
```

The edge basis is a normal config field. This example uses triangular piecewise-linear hats; a
spline, polynomial, or rational edge family would plug in through the same
`nn.models.KAN.EdgeFamily` slot, while the trainer continues to choose the task.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Supervised.Kan

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "kan"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "kan"

/-- Static minibatch size for the Auto MPG tabular loader. -/
def batchSize : Nat := 5

/-- Auto MPG has seven normalized numeric predictors after dropping the car name. -/
def inputWidth : Nat := 7

/-- Scalar regression target. -/
def outputWidth : Nat := 1

/-- KAN configuration using triangular edge bases over normalized tabular features. -/
abbrev modelConfig : nn.models.KAN.Config :=
  { inputWidth := inputWidth
    hiddenWidths := []
    outputWidth := outputWidth
    edge := nn.models.KAN.PiecewiseLinear.edgeFamily { gridSize := 4, inputScale := 3 } }

/-- Input shape for one batch, derived from the KAN config so the two cannot disagree. -/
abbrev input := modelConfig.inputShape [batchSize]
/-- Output shape for one batch. -/
abbrev output := modelConfig.outputShape [batchSize]

/-- Generic KAN model. Regression/classification is selected by `Trainer`, not by the model name. -/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.kan modelConfig [batchSize]

/-- Prepared Auto MPG CSV as a public trainer dataset. -/
def data (path : System.FilePath) (seed : Nat) : Trainer.Dataset input output :=
  Data.fromCsv path batchSize inputWidth outputWidth
    (csvOptions := { skipHeader := true })
    (shuffle := true) (seed := seed)

/-- Train the Auto MPG KAN with the public `Trainer` surface. -/
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
      (logTitle := "KAN Auto MPG regression")
      (logNotes := #[Support.deviceNote runtime, s!"data={flags.csvPath}",
        s!"lr={flags.training.learningRate}", s!"steps={flags.training.steps}",
        s!"batch={batchSize}", s!"edge={modelConfig.edge.name}"]))

/-- CLI entrypoint for Auto MPG regression with a KAN model. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionCsv exeName args
    NN.Examples.Data.RealPaths.autoMpgCsv defaultLogPath 20 1e-2
    (Support.bannerWithDevice exeName "Auto MPG KAN regression")
    train

end NN.Examples.Models.Supervised.Kan
