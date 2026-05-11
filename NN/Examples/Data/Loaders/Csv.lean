/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Examples.Data.ToyPaths
public import NN.API.Data.Transforms

/-!
# CSV loader tutorial (transforms + minibatches + scheduler)

This tutorial mirrors the "data first" workflow people expect from PyTorch:

1. Load a dataset from disk (CSV).
2. Build a transform pipeline (`Data.Transforms.Compose`).
3. Wrap the per-sample dataset in a minibatch loader (`Data.batchLoader`).
4. Train with a learning-rate scheduler.

Generate a small deterministic regression dataset with
`python3 NN/Examples/Data/generate_toy_data.py`:

- `NN/Examples/Data/toy_regression.csv` with rows `x1,x2,y` (25 samples).

Build:

- `lake build NN.Examples.Data.Loaders.Csv`

The tutorial code is compiled with the rest of TorchLean. For command-line model training, use the
`torchlean` executable examples in `NN/Examples/Models`.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `NN/Examples/Data`)
- `--csv PATH` (override the CSV file)
- `--seed S` (controls shuffling and model initialization)
- `--batch N`
- `--epochs E`

Public API used here:

- `Data.fromCsvSupervised`
- `Data.Transforms.Compose`
- `Data.batchLoader`
- `train.fitLoaderWith`
- `train.stepEpochLR`
-/

@[expose] public section


namespace NN.Examples.Data.Loaders.Csv

open Spec
open Tensor
open NN.Tensor
open NN.API

def inDim : Nat := 2
def outDim : Nat := 1

/-- A small 2-layer MLP `2 -> 8 -> 1`. -/
def mkModel {batch : Nat} : nn.M (nn.Sequential (Shape.Mat batch inDim) (Shape.Mat batch outDim)) :=
  nn.sequential![
    nn.linear inDim 8 (pfx := NN.Tensor.Shape.Vec batch),
    nn.relu,
    nn.linear 8 outDim (pfx := NN.Tensor.Shape.Vec batch)
  ]

/--
Load the CSV dataset, then apply a small input transform pipeline.

The transform pipeline is written once for the chosen scalar type `α`:

- normalize (here: mean=0, std=1, so it is an easy-to-read "template"), then
- scale inputs by `0.5`.
-/
def loadDataset (csvPath : System.FilePath)
    {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] :
    IO (Except String (Data.Dataset (sample.Supervised α (Shape.Vec inDim) (Shape.Vec outDim)))) :=
      do
  let opts : Data.CsvOptions := { skipHeader := true }
  let ds0 ← Data.fromCsvSupervised (α := α) csvPath inDim outDim (opts := opts)

  let xShape : Spec.Shape := Shape.Vec inDim
  let yShape : Spec.Shape := Shape.Vec outDim

  let xTransform : Spec.Tensor α xShape → Spec.Tensor α xShape :=
    Data.Transforms.Compose
      [ Data.Transforms.Lambda (Data.Transforms.normalizeTensorF (α := α) (s := xShape) 0.0 1.0)
      , Data.Transforms.Lambda (Data.Transforms.mapTensor (α := α) (s := xShape)
          (fun v => v * Runtime.ofFloat (α := α) 0.5))
      ]

  pure <|
    ds0.map (fun ds =>
      Data.Transforms.onSupervisedDatasetInput (α := α) (σ := xShape) (τ := yShape) xTransform ds)

def main (args : List String) : IO Unit := do
  let args := API.CLI.dropDashDash args

  let label := "Data.Loaders.Csv"
  let (dataDir, args) ← API.Common.orThrow label <| _root_.NN.Examples.Data.ToyPaths.takeDataDir args
  let (seed, args) ← API.Common.orThrow label <| API.CLI.takeSeed args 0
  let (eb, args) ← API.Common.orThrow label <| API.CLI.takeEpochBatch args 30 5
  let (csv?, args) ← API.Common.orThrow label <| API.CLI.takePathFlagOnce args "csv"

  let csvPath := csv?.getD (_root_.NN.Examples.Data.ToyPaths.regressionCsv dataDir)
  if eb.batch = 0 then
    throw <| IO.userError s!"{label}: --batch must be > 0"

  -- Train with a batched task: the model is written directly over `batch × Vec inDim` tensors.
  let task : train.Task (shape![eb.batch, inDim]) (shape![eb.batch, outDim]) :=
    train.regression (nn.build seed (mkModel (batch := eb.batch)))

  IO.println "== CSV loader training tutorial =="
  IO.println s!"data_dir = {dataDir}"
  IO.println s!"csv_path  = {csvPath}"
  IO.println s!"seed      = {seed}"
  IO.println s!"model     = MLP(2 -> 8 -> 1)"
  IO.println (s!"train     = Adam(lr=0.05), epochs={eb.epochs}, batch_size={eb.batch}, " ++
    s!"shuffle=true, drop_last=true")
  IO.println "scheduler = StepLR(step_size=10, gamma=0.5)"

  let _exitCode ← TorchLean.Module.run label args (.float (fun opts rest => do
    API.Common.orThrow label <| API.CLI.requireNoArgs rest
    let runner ← train.instantiateWithOptions (task := task) (α := Float) opts

    let dsE ← loadDataset (csvPath := csvPath) (α := Float)
    let ds ← API.Common.orThrow label dsE

    let loader := Data.batchLoader ds eb.batch (shuffle := true) (seed := seed) (dropLast := true)
    let batchedDs ← API.Common.orThrow label <| Data.BatchLoader.batchDataset loader

    let opt := optim.adam 0.05
    let cfg0 : train.LoaderFitConfig := { (train.epochs eb.epochs (optimizer := opt)) with logEvery
      := 0 }
    let cfg := train.stepEpochLR cfg0 (base := 0.05) (stepSize := 10) (gamma := 0.5)

    let hooks : train.Callbacks Float :=
      (train.onTrainStart do
        train.withMode runner .eval do
          train.Report.reportMeanLoss (task := task) runner batchedDs "before")
      ++ train.logLossEvery 5
      ++ (train.onTrainEnd (fun _ =>
        train.withMode runner .eval do
          train.Report.reportMeanLoss (task := task) runner batchedDs "after"))

    let (_report, _loader') ← train.fitLoaderWith (task := task) runner cfg loader hooks
    pure ())) { printOk := false }
  pure ()

end NN.Examples.Data.Loaders.Csv
