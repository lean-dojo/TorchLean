/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Examples.Data.ToyPaths

/-!
# Minibatch MLP training (batching is implicit)

This next-step file shows the intended "PyTorch-like" minibatch path in TorchLean:

1. Keep the dataset per-sample (`x : Vec inDim`, `y : Vec outDim`).
2. Use `Data.batchLoader` to collate minibatches.
3. Write the model and task over the minibatch shape (`batch × Vec inDim`).
4. Train one step per minibatch.

Run the loader tutorial instead when possible:

- `lake exe torchlean data_csv --epochs 1 --batch 5 --dtype float --backend eager`

Build this comparison module directly with:

- `lake build NN.Examples.Quickstart.MinibatchMlpTrain`

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `NN/Examples/Data`)
- `--csv PATH` (override the CSV file)
- `--seed S`
- `--batch N`
- `--epochs E`
-/

@[expose] public section


namespace NN.Examples.Quickstart.MinibatchMLPTrain

open Spec
open Tensor
open NN.Tensor
open NN.API

def inDim : Nat := 2
def hidDim : Nat := 8
def outDim : Nat := 1

/-- Batched MLP `2 -> 8 -> 1`. -/
def mkModel {batch : Nat} : nn.M (nn.Sequential (Shape.Mat batch inDim) (Shape.Mat batch outDim)) :=
  nn.sequential![
    nn.linear inDim hidDim (pfx := NN.Tensor.Shape.Vec batch),
    nn.relu,
    nn.linear hidDim outDim (pfx := NN.Tensor.Shape.Vec batch)
  ]

def loadDataset (csvPath : System.FilePath)
    {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] :
    IO (Except String (Data.Dataset (sample.Supervised α (Shape.Vec inDim) (Shape.Vec outDim)))) :=
      do
  let opts : Data.CsvOptions := { skipHeader := true }
  Data.fromCsvSupervised (α := α) csvPath inDim outDim (opts := opts)

def main (args : List String) : IO Unit := do
  let args := API.CLI.dropDashDash args

  let (dataDir, args) ← API.Common.orThrow "MinibatchMLPTrain" <| _root_.NN.Examples.Data.ToyPaths.takeDataDir args
  let (seed, args) ← API.Common.orThrow "MinibatchMLPTrain" <| API.CLI.takeSeed args 0
  let (eb, args) ← API.Common.orThrow "MinibatchMLPTrain" <| API.CLI.takeEpochBatch args 30 5
  let (csv?, args) ← API.Common.orThrow "MinibatchMLPTrain" <| API.CLI.takePathFlagOnce args "csv"
  let csvPath := csv?.getD (_root_.NN.Examples.Data.ToyPaths.regressionCsv dataDir)
  if eb.batch = 0 then
    throw <| IO.userError "MinibatchMLPTrain: --batch must be > 0"

  let task : train.Task (shape![eb.batch, inDim]) (shape![eb.batch, outDim]) :=
    train.regression (nn.build seed (mkModel (batch := eb.batch)))

  IO.println "== Quickstart next step: minibatch MLP training =="
  IO.println s!"data_dir = {dataDir}"
  IO.println s!"csv_path  = {csvPath}"
  IO.println s!"seed      = {seed}"
  IO.println s!"model     = MLP(2 -> 8 -> 1)"
  IO.println (s!"train     = Adam(lr=0.05), epochs={eb.epochs}, batch_size={eb.batch}, " ++
    s!"shuffle=true, drop_last=true")

  train.run task args (fun {α} _ _ _ _ runner rest => do
    API.Common.orThrow "MinibatchMLPTrain" <| API.CLI.requireNoArgs rest

    let dsE ← loadDataset (csvPath := csvPath) (α := α)
    let ds ← API.Common.orThrow "MinibatchMLPTrain" dsE

    let loader := Data.batchLoader ds eb.batch (shuffle := true) (seed := seed) (dropLast := true)
    let batchedDs ← API.Common.orThrow "MinibatchMLPTrain" <| Data.BatchLoader.batchDataset loader

    let cfg : train.LoaderFitConfig := { (train.epochs eb.epochs (optimizer := optim.adam 0.05))
      with logEvery := 0 }
    let hooks : train.Callbacks α :=
      (train.onTrainStart do
        train.withMode runner .eval do
          train.Report.reportMeanLoss (task := task) runner batchedDs "before")
      ++ train.logLossEvery 5
      ++ (train.onTrainEnd (fun _ =>
        train.withMode runner .eval do
          train.Report.reportMeanLoss (task := task) runner batchedDs "after"))

    let (_report, _loader') ← train.fitLoaderWith (task := task) runner cfg loader hooks
    pure ())

end NN.Examples.Quickstart.MinibatchMLPTrain
