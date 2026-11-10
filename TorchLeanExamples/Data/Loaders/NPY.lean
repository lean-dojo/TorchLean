/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Trainer.Constructor
public import TorchLean.NN.Trainer.Train
public import TorchLeanExamples.Data.Loaders.CSV
public import TorchLeanExamples.Data.SamplePaths
public import TorchLean.NN.CLI.Trainer
public import TorchLean.NN.Data.Dataset

/-!
# NPY loader tutorial (NumPy/PyTorch interop)

The tutorial trains from `.npy` files (NumPy arrays), as in a common PyTorch workflow: prepare
`X.npy` and `y.npy` in Python with NumPy or PyTorch, then load those files and train a model in
TorchLean.

Generate small deterministic `.npy` files with
`python3 TorchLeanExamples/Data/generate_small_data.py`:

- `TorchLeanExamples/Data/small_regression_X.npy`  (shape 25×2, dtype float32)
- `TorchLeanExamples/Data/small_regression_y.npy`  (shape 25×1, dtype float32)

Build:

- `scripts/lake.sh build TorchLeanExamples.Data.Loaders.NPY`

Run it with `scripts/lake.sh exe torchlean data_npy`. The command reads the metadata of both
arrays and checks the shapes `(N, 2)` and `(N, 1)` before it builds the typed dataset, then trains
through the same public trainer as the model examples.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `TorchLeanExamples/Data`)
- `--x PATH`, `--y PATH` (override the `.npy` files)
- `--seed S` (controls shuffling and model initialization)
- `--batch N`
- `--steps N`

Public API used here:

- `Data.NPY.read` (metadata)
- `Data.fromSupervisedSource`
- `Data.batch`
- `Trainer.create`
- `Trainer.RunConfig`
- `Trainer.TrainOptions`
- `trainer.train`
-/

@[expose] public section

namespace TorchLean.Examples.NPY

open Data Trainer.Objective

/-- Command name used in diagnostics and by the top-level example runner. -/
def exeName : String := "data_npy"

/-- Check that `X` has shape `(N, 2)` and `y` has shape `(N, 1)` for one `N`, and return `N`. -/
def rowCount
    (xShape yShape : Array Nat) :
    Except String Nat :=
  match xShape, yShape with
  | #[xRows, xWidth], #[yRows, yWidth] =>
    if xWidth != CSV.inputWidth then
      .error s!"X.npy: expected shape (N,{CSV.inputWidth}), got {xShape}"
    else if yWidth != CSV.outputWidth then
      .error s!"y.npy: expected shape (N,{CSV.outputWidth}), got {yShape}"
    else if xRows != yRows then
      .error s!"NPY row mismatch: X has {xRows} rows, y has {yRows}"
    else
      .ok xRows
  | _, _ =>
    .error <|
      s!"expected X.npy shape (N,{CSV.inputWidth}) and y.npy shape (N,{CSV.outputWidth}); " ++
      s!"got {xShape} and {yShape}"

/-- Command-line help for the NPY loader tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean NPY loader tutorial"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean data_npy [options]"
    , ""
    , "Options:"
    , "  --data-dir PATH"
    , "  --x PATH"
    , "  --y PATH"
    , "  --seed N"
    , "  --batch N"
    , "  --steps N"
    , "  --arithmetic native|ieee"
    , s!"  --device auto|{Backend.Device.help}"
    , "  --show-backend                    print backend capsules as they are selected"
    ]

/-- Entry point: load the NumPy arrays, then train the same MLP the CSV tutorial trains. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return

  let (dataDir, args) ← CLI.orThrow exeName <| CLI.takePathFlag args "data-dir"
    (default := Examples.SampleData.defaultDataDir)
  let (seed, args) ← CLI.orThrow exeName <| CLI.takeSeed args (default := 0)
  let (steps, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args "steps" (default := 20)
  let (batchSize, args) ←
    CLI.orThrow exeName <| CLI.takePositiveNatFlag args exeName "batch" (default := 5)
  let (xPath, args) ← CLI.orThrow exeName <| CLI.takePathFlag args "x"
    (default := Examples.SampleData.regressionX dataDir)
  let (yPath, args) ← CLI.orThrow exeName <| CLI.takePathFlag args "y"
    (default := Examples.SampleData.regressionY dataDir)

  let network := CSV.model (batchSize := batchSize)
  let run ← Trainer.RunConfig.parseCommandLine exeName args
    { optimizer := optim.adam { lr := 0.05 } }
  let trainer := Trainer.create network <|
    { run with objective := mse, seed }

  IO.println "== NPY loader training tutorial =="
  trainer.printSummary
  IO.println s!"data_dir = {dataDir}"
  IO.println s!"x_path   = {xPath}"
  IO.println s!"y_path   = {yPath}"
  IO.println s!"seed     = {seed}"
  IO.println
    (s!"train    = Adam(lr=0.05), steps={steps}, batch_size={batchSize}, " ++
      s!"shuffle=true, drop_last=true")
  let xMeta ← CLI.orThrow exeName <| (← Data.NPY.read xPath)
  let yMeta ← CLI.orThrow exeName <| (← Data.NPY.read yPath)
  IO.println s!"X.npy dtype={xMeta.dtype} shape={xMeta.shape}"
  IO.println s!"y.npy dtype={yMeta.dtype} shape={yMeta.shape}"
  let rowCount ← CLI.orThrow exeName <|
    rowCount xMeta.shape yMeta.shape

  let src : Data.SupervisedSource :=
    Data.SupervisedSource.fromFiles xPath yPath rowCount [CSV.inputWidth] [CSV.outputWidth]
  let samples := Data.fromSupervisedSource src
  let data := Data.batch batchSize samples
    (shuffle := true) (seed := seed)
  let trained ← trainer.train data { steps := steps }
  trained.printSummary
  let heldout : Tensor Float [batchSize, CSV.inputWidth] :=
    Tensor.full [batchSize, CSV.inputWidth] 0.25
  trained.printPrediction "predict(batch=heldout)" heldout

end TorchLean.Examples.NPY
