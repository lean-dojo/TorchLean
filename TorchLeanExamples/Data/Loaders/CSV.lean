/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Trainer.Constructor
public import TorchLean.NN.Trainer.Train
public import TorchLeanExamples.Data.SamplePaths
public import TorchLean.NN.CLI.Trainer
public import TorchLean.NN.Data.Dataset

/-!
# CSV loader tutorial

The tutorial mirrors the "data first" workflow people expect from PyTorch:

1. Load a dataset from disk (CSV).
2. Turn it into a fixed-size batched dataset.
3. Train through the public `Trainer` API.

Generate a small deterministic regression dataset with
`python3 TorchLeanExamples/Data/generate_small_data.py`:

- `TorchLeanExamples/Data/small_regression.csv` with rows `x1,x2,y` (25 samples).

Build:

- `scripts/lake.sh build TorchLeanExamples.Data.Loaders.CSV`

Run it with `scripts/lake.sh exe torchlean data_csv`. The command reads the CSV into shuffled
batches of two features and one target, trains the `2 -> 8 -> 1` MLP with Adam, and prints a
prediction for a constant held-out batch.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `TorchLeanExamples/Data`)
- `--csv PATH` (override the CSV file)
- `--seed S` (controls shuffling and model initialization)
- `--batch N`
- `--steps N`

Public API used here:

- `Data.fromCSV`
- `Trainer.create`
- `Trainer.RunConfig`
- `Trainer.TrainOptions`
- `trainer.train`
-/

@[expose] public section

namespace TorchLean.Examples.CSV

open Data Trainer.Objective

/-- Command name used in diagnostics and by the top-level example runner. -/
def exeName : String := "data_csv"

/-- Hint printed when the CSV file is missing, with the command that generates it. -/
def missingHint : String :=
  "Generate the small regression CSV with:\n" ++
  "  python3 TorchLeanExamples/Data/generate_small_data.py"

/-- Two input features, matching the generated regression CSV. -/
def inputWidth : Nat := 2
/-- Hidden width of the MLP, the same as in the NPY tutorial. -/
def hiddenWidth : Nat := 8
/-- One regression target. -/
def outputWidth : Nat := 1

/-- A small 2-layer batched MLP `2 -> 8 -> 1`. -/
def model {batchSize : Nat} :
    nn.Builder (nn.Sequential [batchSize, inputWidth] [batchSize, outputWidth]) :=
  nn.mlp inputWidth outputWidth { hiddenWidths := [hiddenWidth] } [batchSize]

/-- Command-line help for the CSV loader tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean CSV loader tutorial"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean data_csv [options]"
    , ""
    , "Options:"
    , "  --data-dir PATH"
    , "  --csv PATH"
    , "  --seed N"
    , "  --batch N"
    , "  --steps N"
    , "  --arithmetic native|ieee"
    , s!"  --device auto|{Backend.Device.help}"
    , "  --show-backend                    print backend capsules as they are selected"
    ]

/-- Entry point: read the CSV, build the loader, then train the `2 -> 8 -> 1` MLP on it. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return

  let (dataDir, args) ← CLI.orThrow exeName <| CLI.takePathFlag args "data-dir"
    (default := Examples.SampleData.defaultDataDir)
  let (seed, args) ← CLI.orThrow exeName <| CLI.takeSeed args (default := 0)
  let (steps, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args "steps" (default := 30)
  let (batchSize, args) ←
    CLI.orThrow exeName <| CLI.takePositiveNatFlag args exeName "batch" (default := 5)
  let (csvPath, args) ← CLI.orThrow exeName <|
    CLI.takePathFlag args "csv" (default := (Examples.SampleData.regressionCSV dataDir))

  let network := model (batchSize := batchSize)
  let run ← Trainer.RunConfig.parseCommandLine exeName args
    { optimizer := optim.adam { lr := 0.05 } }
  let trainer := Trainer.create network <|
    { run with objective := mse, seed }

  IO.println "== CSV loader training tutorial =="
  trainer.printSummary
  IO.println s!"data_dir = {dataDir}"
  IO.println s!"csv_path  = {csvPath}"
  IO.println s!"seed      = {seed}"
  IO.println
    (s!"train     = Adam(lr=0.05), steps={steps}, batch_size={batchSize}, " ++
      s!"shuffle=true, drop_last=true")

  let csvOptions : Data.CSV.Options := { skipHeader := true }
  let data :=
    Data.fromCSV csvPath batchSize inputWidth outputWidth
      (csvOptions := csvOptions) (shuffle := true) (seed := seed)
  Data.requireFile exeName "CSV dataset" csvPath missingHint
  let trained ← trainer.train data { steps := steps }
  trained.printSummary
  let heldout : Tensor Float [batchSize, inputWidth] :=
    Tensor.full [batchSize, inputWidth] 0.25
  trained.printPrediction "predict(batch=heldout)" heldout

end TorchLean.Examples.CSV
