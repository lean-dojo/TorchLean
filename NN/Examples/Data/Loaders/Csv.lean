/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Data.SamplePaths

/-!
# CSV loader tutorial

This tutorial mirrors the "data first" workflow people expect from PyTorch:

1. Load a dataset from disk (CSV).
2. Turn it into a fixed-size batched dataset.
3. Train through the public `Trainer` API.

Generate a small deterministic regression dataset with
`python3 NN/Examples/Data/generate_small_data.py`:

- `NN/Examples/Data/small_regression.csv` with rows `x1,x2,y` (25 samples).

Build:

- `lake build NN.Examples.Data.Loaders.Csv`

The tutorial code is compiled with the rest of TorchLean and is directly runnable as
`lake exe torchlean data_csv`. It keeps data loading and training together so the parsed CSV shape
can be checked before the first optimizer step.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `NN/Examples/Data`)
- `--csv PATH` (override the CSV file)
- `--seed S` (controls shuffling and model initialization)
- `--batch N`
- `--steps N`

Public API used here:

- `Data.fromCsv`
- `Trainer.new`
- `Trainer.RunConfig`
- `Trainer.TrainOptions`
- `trainer.train`
-/

@[expose] public section

namespace NN.Examples.Data.Loaders.Csv

open TorchLean

/-- Command name used in diagnostics and by the top-level example runner. -/
def exeName : String := "data_csv"

/--
Printed when the CSV is absent, so the reader knows how to produce it rather than just seeing a
file-not-found error.
-/
def missingCsvHint : String :=
  "Generate the small regression CSV with:\n" ++
  "  python3 NN/Examples/Data/generate_small_data.py"

/-- Two input features, matching the generated regression CSV. -/
def inputWidth : Nat := 2
/-- Hidden width; small enough that the printed parameter tensors fit on a screen. -/
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
    , "  lake exe torchlean data_csv [options]"
    , ""
    , "Options:"
    , "  --data-dir PATH"
    , "  --csv PATH"
    , "  --seed N"
    , "  --batch N"
    , "  --steps N"
    , "  --arithmetic native|ieee"
    , "  --execution eager|typed-graph"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --show-backend                    print backend capsules as they execute"
    ]

/-- Entry point: read the CSV, build the loader, then train the `2 -> 8 -> 1` MLP on it. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return

  let (dataDir, args) ← CLI.orThrow exeName <| TorchLean.CLI.takePathFlag args "data-dir"
    (default := NN.Examples.Data.SamplePaths.defaultDataDir)
  let (seed, args) ← CLI.orThrow exeName <| CLI.takeSeed args (default := 0)
  let (steps, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args "steps" (default := 30)
  let (batchSize, args) ←
    CLI.orThrow exeName <| CLI.takePositiveNatFlag args exeName "batch" (default := 5)
  let (csvPath, args) ← CLI.orThrow exeName <|
    CLI.takePathFlag args "csv" (default := (NN.Examples.Data.SamplePaths.regressionCsv dataDir))

  let network := model (batchSize := batchSize)
  let run ← TorchLean.CLI.Trainer.parseCommandLine exeName args
    { optimizer := optim.adam { learningRate := 0.05 } }
  let trainer := Trainer.new network <|
    Trainer.RunConfig.forObjective run .meanSquaredError (seed := seed)

  IO.println "== CSV loader training tutorial =="
  trainer.printSummary
  IO.println s!"data_dir = {dataDir}"
  IO.println s!"csv_path  = {csvPath}"
  IO.println s!"seed      = {seed}"
  IO.println
    (s!"train     = Adam(lr=0.05), steps={steps}, batch_size={batchSize}, " ++
      s!"shuffle=true, drop_last=true")

  let csvOptions : Data.CsvOptions := { skipHeader := true }
  let data :=
    Data.fromCsv csvPath batchSize inputWidth outputWidth
      (csvOptions := csvOptions) (shuffle := true) (seed := seed)
  Data.requireFile exeName "CSV dataset" csvPath missingCsvHint
  let trained ← trainer.train data { steps := steps }
  trained.printSummary
  let heldout : Tensor Float [batchSize, inputWidth] :=
    Tensor.full [batchSize, inputWidth] 0.25
  trained.printPrediction "predict(batch=heldout)" heldout

end NN.Examples.Data.Loaders.Csv
