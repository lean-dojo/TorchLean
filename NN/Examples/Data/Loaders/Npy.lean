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
# NPY loader tutorial (NumPy/PyTorch interop)

This tutorial shows how to train from `.npy` files (NumPy arrays), similar to a common PyTorch
workflow where you:

1. prepare `X.npy` / `y.npy` in Python (NumPy / PyTorch),
2. then train a model in TorchLean by loading those files.

Generate small deterministic `.npy` files with
`python3 NN/Examples/Data/generate_toy_data.py`:

- `NN/Examples/Data/toy_regression_X.npy`  (shape 25×2, dtype float32)
- `NN/Examples/Data/toy_regression_y.npy`  (shape 25×1, dtype float32)

Build:

- `lake build NN.Examples.Data.Loaders.Npy`

The tutorial code is compiled with the rest of TorchLean. For command-line model training, use the
`torchlean` executable examples in `NN/Examples/Models`.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `NN/Examples/Data`)
- `--x PATH`, `--y PATH` (override the `.npy` files)
- `--seed S` (controls shuffling and model initialization)
- `--batch N`
- `--epochs E`

Public API used here:

- `Data.fromNpy` (metadata)
- `Data.fromNpySupervised` (typed dataset from disk)
- `Data.Transforms.Compose`
- `Data.batchLoader`
- `train.fitLoaderWith`
-/

@[expose] public section


namespace NN.Examples.Data.Loaders.Npy

open Spec
open Tensor
open NN.Tensor
open NN.API

def nSamples : Nat := 25
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
Load the `.npy` tensors, print their metadata, then apply a small input transform.

This file is intentionally interop-first: it shows the path from NumPy/PyTorch exports on disk to
TorchLean's normal training API.
-/
def loadDataset (xPath yPath : System.FilePath)
    {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] :
    IO (Except String (Data.Dataset (sample.Supervised α (Shape.Vec inDim) (Shape.Vec outDim)))) :=
      do
  let xMeta ← Data.fromNpy xPath
  let yMeta ← Data.fromNpy yPath
  match xMeta with
  | .error e => pure (.error e)
  | .ok xMeta =>
      match yMeta with
      | .error e => pure (.error e)
      | .ok yMeta => do
          IO.println s!"X.npy dtype={xMeta.dtype} shape={xMeta.shape}"
          IO.println s!"y.npy dtype={yMeta.dtype} shape={yMeta.shape}"

          let ds0 ← Data.fromNpySupervised (α := α) xPath yPath nSamples [inDim] [outDim]

          let xShape : Spec.Shape := Shape.Vec inDim
          let yShape : Spec.Shape := Shape.Vec outDim

          -- Example transform: scale inputs down a bit.
          let xTransform : Spec.Tensor α xShape → Spec.Tensor α xShape :=
            Data.Transforms.Compose
              [ Data.Transforms.Lambda (Data.Transforms.mapTensor (α := α) (s := xShape)
                  (fun v => v * Runtime.ofFloat (α := α) 0.5))
              ]

          pure <|
            ds0.map (fun ds =>
              Data.Transforms.onSupervisedDatasetInput (α := α) (σ := xShape) (τ := yShape)
                xTransform ds)

def main (args : List String) : IO Unit := do
  let args := API.CLI.dropDashDash args

  let label := "Data.Loaders.Npy"
  let (dataDir, args) ← API.Common.orThrow label <| _root_.NN.Examples.Data.ToyPaths.takeDataDir args
  let (seed, args) ← API.Common.orThrow label <| API.CLI.takeSeed args 0
  let (eb, args) ← API.Common.orThrow label <| API.CLI.takeEpochBatch args 20 5

  let (x?, args) ← API.Common.orThrow label <| API.CLI.takePathFlagOnce args "x"
  let (y?, args) ← API.Common.orThrow label <| API.CLI.takePathFlagOnce args "y"

  let xPath := x?.getD (_root_.NN.Examples.Data.ToyPaths.regressionXNpy dataDir)
  let yPath := y?.getD (_root_.NN.Examples.Data.ToyPaths.regressionYNpy dataDir)
  if eb.batch = 0 then
    throw <| IO.userError s!"{label}: --batch must be > 0"

  -- Train with a batched task: the model is written directly over `batch × Vec inDim` tensors.
  let task : train.Task (shape![eb.batch, inDim]) (shape![eb.batch, outDim]) :=
    train.regression (nn.build seed (mkModel (batch := eb.batch)))

  IO.println "== NPY loader training tutorial =="
  IO.println s!"data_dir = {dataDir}"
  IO.println s!"x_path   = {xPath}"
  IO.println s!"y_path   = {yPath}"
  IO.println s!"seed     = {seed}"
  IO.println s!"model    = MLP(2 -> 8 -> 1)"
  IO.println (s!"train    = Adam(lr=0.05), epochs={eb.epochs}, batch_size={eb.batch}, " ++
    s!"shuffle=true, drop_last=true")
  IO.println "scheduler = ExponentialLR(gamma=0.9)"

  let _exitCode ← TorchLean.Module.run label args (.float (fun opts rest => do
    API.Common.orThrow label <| API.CLI.requireNoArgs rest
    let runner ← train.instantiateWithOptions (task := task) (α := Float) opts

    let dsE ← loadDataset (xPath := xPath) (yPath := yPath) (α := Float)
    let ds ← API.Common.orThrow label dsE

    let loader := Data.batchLoader ds eb.batch (shuffle := true) (seed := seed) (dropLast := true)
    let batchedDs ← API.Common.orThrow label <| Data.BatchLoader.batchDataset loader

    let opt := optim.adam 0.05
    let cfg0 : train.LoaderFitConfig := { (train.epochs eb.epochs (optimizer := opt)) with logEvery
      := 0 }
    let cfg := train.exponentialEpochLR cfg0 (base := 0.05) (gamma := 0.9)

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

end NN.Examples.Data.Loaders.Npy
