/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Data.SamplePaths

/-!
# NPY loader tutorial (NumPy/PyTorch interop)

This tutorial shows how to train from `.npy` files (NumPy arrays), similar to a common PyTorch
workflow where you:

1. prepare `X.npy` / `y.npy` in Python (NumPy / PyTorch),
2. then train a model in TorchLean by loading those files.

Generate small deterministic `.npy` files with
`python3 NN/Examples/Data/generate_small_data.py`:

- `NN/Examples/Data/small_regression_X.npy`  (shape 25×2, dtype float32)
- `NN/Examples/Data/small_regression_y.npy`  (shape 25×1, dtype float32)

Build:

- `lake build NN.Examples.Data.Loaders.Npy`

The tutorial code is compiled with the rest of TorchLean and is directly runnable as
`lake exe torchlean data_npy`. It checks the array metadata before constructing the typed dataset,
then trains through the same public trainer used by the model examples.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `NN/Examples/Data`)
- `--x PATH`, `--y PATH` (override the `.npy` files)
- `--seed S` (controls shuffling and model initialization)
- `--batch N`
- `--steps N`

Public API used here:

- `Data.readNpy` (metadata)
- `Data.fromSupervisedSource`
- `Data.batch`
- `Trainer.new`
- `Trainer.RunConfig`
- `Trainer.TrainOptions`
- `trainer.train`
-/

@[expose] public section

namespace NN.Examples.Data.Loaders.Npy

open TorchLean

/-- Command name used in diagnostics and by the top-level example runner. -/
def exeName : String := "data_npy"

/-- Two input features, matching the generated `.npy` arrays. -/
def inputWidth : Nat := 2
/-- Hidden width, kept the same as the CSV tutorial so the two can be compared directly. -/
def hiddenWidth : Nat := 8
/-- One regression target. -/
def outputWidth : Nat := 1

/-- A small 2-layer batched MLP `2 -> 8 -> 1`. -/
def model {batchSize : Nat} :
    nn.Builder (nn.Sequential [batchSize, inputWidth] [batchSize, outputWidth]) :=
  nn.mlp inputWidth outputWidth { hiddenWidths := [hiddenWidth] } [batchSize]

/-- Read exactly two matrix dimensions from untrusted NPY metadata. -/
def Internal.matrixDimensions? (shape : Array Nat) : Option (Nat × Nat) := do
  let rows ← shape[0]?
  let columns ← shape[1]?
  if shape.size = 2 then
    pure (rows, columns)
  else
    none

/-- Validate the two NPY shapes and return their shared leading-axis size. -/
def rowCountFromMetadata
    (xShape yShape : Array Nat) :
    Except String Nat :=
  match Internal.matrixDimensions? xShape, Internal.matrixDimensions? yShape with
  | some (xRows, xWidth), some (yRows, yWidth) =>
    if xWidth != inputWidth then
      .error s!"X.npy: expected shape (N,{inputWidth}), got {xShape}"
    else if yWidth != outputWidth then
      .error s!"y.npy: expected shape (N,{outputWidth}), got {yShape}"
    else if xRows != yRows then
      .error s!"NPY row mismatch: X has {xRows} rows, y has {yRows}"
    else
      .ok xRows
  | _, _ =>
    .error <|
      s!"expected X.npy shape (N,{inputWidth}) and y.npy shape (N,{outputWidth}); " ++
      s!"got {xShape} and {yShape}"

/-- Command-line help for the NPY loader tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean NPY loader tutorial"
    , ""
    , "Usage:"
    , "  lake exe torchlean data_npy [options]"
    , ""
    , "Options:"
    , "  --data-dir PATH"
    , "  --x PATH"
    , "  --y PATH"
    , "  --seed N"
    , "  --batch N"
    , "  --steps N"
    , "  --arithmetic native|ieee"
    , "  --execution eager|typed-graph"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --show-backend                    print backend capsules as they execute"
    ]

/-- Entry point: load the NumPy arrays, then train the same MLP the CSV tutorial trains. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return

  let (dataDir, args) ← CLI.orThrow exeName <| TorchLean.CLI.takePathFlag args "data-dir"
    (default := NN.Examples.Data.SamplePaths.defaultDataDir)
  let (seed, args) ← CLI.orThrow exeName <| CLI.takeSeed args (default := 0)
  let (steps, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args "steps" (default := 20)
  let (batchSize, args) ←
    CLI.orThrow exeName <| CLI.takePositiveNatFlag args exeName "batch" (default := 5)
  let (xPath, args) ← CLI.orThrow exeName <| CLI.takePathFlag args "x"
    (default := NN.Examples.Data.SamplePaths.regressionXNpy dataDir)
  let (yPath, args) ← CLI.orThrow exeName <| CLI.takePathFlag args "y"
    (default := NN.Examples.Data.SamplePaths.regressionYNpy dataDir)

  let network := model (batchSize := batchSize)
  let run ← TorchLean.CLI.Trainer.parseCommandLine exeName args
    { optimizer := optim.adam { learningRate := 0.05 } }
  let trainer := Trainer.new network <|
    Trainer.RunConfig.forObjective run .mse (seed := seed)

  IO.println "== NPY loader training tutorial =="
  trainer.printSummary
  IO.println s!"data_dir = {dataDir}"
  IO.println s!"x_path   = {xPath}"
  IO.println s!"y_path   = {yPath}"
  IO.println s!"seed     = {seed}"
  IO.println
    (s!"train    = Adam(lr=0.05), steps={steps}, batch_size={batchSize}, " ++
      s!"shuffle=true, drop_last=true")
  let xMeta ← CLI.orThrow exeName <| (← Data.readNpy xPath)
  let yMeta ← CLI.orThrow exeName <| (← Data.readNpy yPath)
  IO.println s!"X.npy dtype={xMeta.dtype} shape={xMeta.shape}"
  IO.println s!"y.npy dtype={yMeta.dtype} shape={yMeta.shape}"
  let rowCount ← CLI.orThrow exeName <|
    rowCountFromMetadata xMeta.shape yMeta.shape

  let src : Data.SupervisedSource :=
    Data.SupervisedSource.fromFiles xPath yPath rowCount [inputWidth] [outputWidth]
  let samples := Data.fromSupervisedSource src
  let data := Data.batch batchSize samples
    (shuffle := true) (seed := seed)
  let trained ← trainer.train data { steps := steps }
  trained.printSummary
  let heldout : Tensor Float [batchSize, inputWidth] :=
    Tensor.full [batchSize, inputWidth] 0.25
  trained.printPrediction "predict(batch=heldout)" heldout

end NN.Examples.Data.Loaders.Npy
