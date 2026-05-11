/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Examples.Data.RealPaths
public import NN.Examples.Data.ToyPaths

/-!
# CIFAR10-style image loader tutorial (NPY, offline)

This tutorial mirrors a classic PyTorch recipe:

1. Load a labeled image dataset from disk (`.npy` exported from NumPy/PyTorch).
2. Split into train/test.
3. Build a small CNN by explicitly stacking layers.
4. Train for multiple epochs over shuffled minibatches and report loss.

To keep this runnable without network downloads, generate a small deterministic
"CIFAR10-shaped" dataset locally:

- `NN/Examples/Data/toy_cifar10like_X.npy`: shape `(200, 3, 32, 32)`, dtype `float32`
- `NN/Examples/Data/toy_cifar10like_y.npy`: shape `(200,)`, dtype `float32` labels `0..9`

Generate it with:

`python3 NN/Examples/Data/generate_toy_data.py`

Build:

- `lake build NN.Examples.Data.Loaders.Cifar10Images`

For command-line CIFAR training, use `torchlean cnn`, `torchlean resnet`, or `torchlean vit` with
`--x`, `--y`, and `--n-total`.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `NN/Examples/Data`)
- `--real-cifar10` (use `data/real/cifar10/cifar10_train_*.npy`, as prepared by
  `scripts/datasets/download_example_data.py --cifar10`)
- `--x PATH`, `--y PATH` (override the `.npy` files)
- `--n-total N` (number of rows in the selected `.npy` files; default `200`)
- `--seed S` (controls split + shuffling + model initialization)
- `--batch N`
- `--epochs E`
- `--lr LR` (default: `0.001`)
- `--log-every N` (default: `1`; pass `0` to silence per-step loss)
- `--train-size N` (default: 160)
- `--quick` (load/split smoke test; skip the expensive eager CPU CNN training loop)

Why this tutorial matters:

- it shows the public `API.Data` file-loading path rather than only in-memory tensors;
- it keeps the model architecture familiar (Conv/ReLU/Pool stack + classifier head);
- it demonstrates the "offline artifact" workflow many PyTorch users already have, where arrays
  have been pre-exported to `.npy` and training happens without any dataset download step.
-/

@[expose] public section


namespace NN.Examples.Data.Loaders.Cifar10Images

open Spec
open Tensor
open NN.Tensor
open NN.API

abbrev classes : Nat := 10
abbrev channels : Nat := 3
abbrev height : Nat := 32
abbrev width : Nat := 32
abbrev nTotal : Nat := 200

/-- Small CNN (no BatchNorm): Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Linear(10). -/
def mkModel {batch : Nat} :
    nn.M (nn.Sequential (Shape.Images batch channels height width) (shape![batch, classes])) :=
  let outC1 : Nat := 16
  let outC2 : Nat := 32
  let conv1 : nn.Conv :=
    { outC := outC1, kH := 3, kW := 3, stride := 1, padding := 1 }
  let conv2 : nn.Conv :=
    { outC := outC2, kH := 3, kW := 3, stride := 1, padding := 1 }
  let pool : nn.MaxPool :=
    { kH := 2, kW := 2, stride := 2 }
  let h1 : Nat := (height + 2 * conv1.padding - conv1.kH) / conv1.stride + 1
  let w1 : Nat := (width + 2 * conv1.padding - conv1.kW) / conv1.stride + 1
  let h2 : Nat := (h1 - pool.kH) / pool.stride + 1
  let w2 : Nat := (w1 - pool.kW) / pool.stride + 1
  let h3 : Nat := (h2 + 2 * conv2.padding - conv2.kH) / conv2.stride + 1
  let w3 : Nat := (w2 + 2 * conv2.padding - conv2.kW) / conv2.stride + 1
  let h4 : Nat := (h3 - pool.kH) / pool.stride + 1
  let w4 : Nat := (w3 - pool.kW) / pool.stride + 1
  let featInner : Shape := Shape.Image outC2 h4 w4
  let featSize : Nat := Spec.Shape.size featInner
  nn.sequential![
    nn.conv (n := batch) (inC := channels) (inH := height) (inW := width) conv1,
    nn.relu,
    nn.maxPool (n := batch) (inC := outC1) (inH := h1) (inW := w1) pool,
    nn.conv (n := batch) (inC := outC1) (inH := h2) (inW := w2) conv2,
    nn.relu,
    nn.maxPool (n := batch) (inC := outC2) (inH := h3) (inW := w3) pool,
    nn.flattenBatch,
    nn.linear featSize classes (pfx := Shape.Vec batch)
  ]

/-- Load the offline CIFAR10-like `.npy` dataset at the runtime-selected scalar type `α`. -/
def loadDataset (xPath yPath : System.FilePath) (n : Nat)
    {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] :
    IO (Except String (Data.Dataset (sample.Supervised α (Shape.Image channels height width)
      (Shape.Vec classes)))) :=
  Data.fromNpyLabeled (α := α) xPath yPath n [channels, height, width] classes

def main (args : List String) : IO Unit := do
  let args0 := API.CLI.dropDashDash args
  let quick := args0.contains "--quick"
  let realCifar10 := args0.contains "--real-cifar10"
  let args := args0.filter (fun a => a != "--quick" && a != "--real-cifar10")

  let label := "Data.Loaders.Cifar10Images"
  let (dataDir, args) ← API.Common.orThrow label <| _root_.NN.Examples.Data.ToyPaths.takeDataDir
    args
  let (seed, args) ← API.Common.orThrow label <| API.CLI.takeSeed args 0
  let (eb, args) ← API.Common.orThrow label <| API.CLI.takeEpochBatch args 5 20
  let (trainSize?, args) ← API.Common.orThrow label <| API.CLI.takeNatFlagOnce args
    "train-size"
  let (nTotal?, args) ← API.Common.orThrow label <| API.CLI.takeNatFlagOnce args
    "n-total"
  let (lr?, args) ← API.Common.orThrow label <| API.CLI.takeFloatFlagOnce args "lr"
  let (logEvery?, args) ← API.Common.orThrow label <| API.CLI.takeNatFlagOnce args "log-every"
  let (x?, args) ← API.Common.orThrow label <| API.CLI.takePathFlagOnce args "x"
  let (y?, args) ← API.Common.orThrow label <| API.CLI.takePathFlagOnce args "y"

  if eb.batch = 0 then
    throw <| IO.userError s!"{label}: --batch must be > 0"

  let xPath :=
    match x? with
    | some p => p
    | none =>
        if realCifar10 then
          _root_.NN.Examples.Data.RealPaths.cifar10TrainX
        else
          _root_.NN.Examples.Data.ToyPaths.cifar10likeXNpy dataDir
  let yPath :=
    match y? with
    | some p => p
    | none =>
        if realCifar10 then
          _root_.NN.Examples.Data.RealPaths.cifar10TrainY
        else
          _root_.NN.Examples.Data.ToyPaths.cifar10likeYNpy dataDir
  let nRows := nTotal?.getD nTotal
  let trainSize := trainSize?.getD (if quick then Nat.min 16 nRows else Nat.min 160 nRows)
  let lr := lr?.getD 0.001
  let logEvery := logEvery?.getD 1

  let task : train.Task (Shape.Images eb.batch channels height width) (shape![eb.batch, classes]) :=
    train.classificationOneHot (nn.build seed (mkModel (batch := eb.batch)))

  IO.println "== CIFAR10-style NPY CNN tutorial =="
  IO.println s!"data_dir   = {dataDir}"
  IO.println s!"x_path     = {xPath}"
  IO.println s!"y_path     = {yPath}"
  IO.println s!"rows       = {nRows}"
  IO.println s!"seed       = {seed}"
  IO.println s!"train_size = {trainSize} / {nRows}"
  IO.println s!"model      = Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Linear({classes})"
  IO.println <|
    (s!"train      = Adam(lr={lr}), epochs={eb.epochs}, " ++
      s!"batch_size={eb.batch}, shuffle=true, drop_last=true, log_every={logEvery}")
  if quick then
    IO.println "mode       = --quick (load/split smoke test; skip CNN training)"
  (← IO.getStdout).flush

  let _exitCode ← TorchLean.Module.run label args (.float (fun opts rest => do
    API.Common.orThrow label <| API.CLI.requireNoArgs rest
    let module ← TorchLean.Module.instantiateWithOptions (α := Float) task.moduleDef id opts

    let dsE ← loadDataset (xPath := xPath) (yPath := yPath) (n := nRows) (α := Float)
    let dsAll ← API.Common.orThrow label dsE

    if trainSize > dsAll.size then
      throw <| IO.userError
        s!"{label}: --train-size {trainSize} exceeds dataset size {dsAll.size}"

    let (_seed', (dsTrain, dsTest)) := Data.randomSplitAt (seed := seed) trainSize dsAll

    if quick then
      IO.println s!"loaded     = {dsAll.size} image rows"
      IO.println s!"split      = train {dsTrain.size}, test {dsTest.size}"
      IO.println "quick      = dataset shape/path smoke passed"
      pure ()
    else

      let trainLoader := Data.batchLoader dsTrain eb.batch (shuffle := true) (seed := seed)
        (dropLast := true)

      let testLoader := Data.batchLoader dsTest eb.batch (shuffle := false) (seed := seed)
        (dropLast := true)
      let opt := TorchLean.Optim.adam (α := Float)
        (paramShapes := TorchLean.Supervised.paramShapes task)
        (lr := lr) (beta1 := 0.9) (beta2 := 0.999) (epsilon := 1e-8)

      -- The CIFAR tutorial uses the same public module-loader loop as the rest of the model
      -- zoo.  `Data.batchLoader` owns shuffling and epoch state; `train.fitModuleLoaderWith`
      -- streams raw minibatches, collates each one into typed tensors, and calls the runtime
      -- module's forward/backward/optimizer step.  That keeps this file focused on the dataset and
      -- model definition instead of carrying a local training loop.
      let hooks : train.Callbacks Float :=
        (train.onTrainStart (α := Float) do
          train.Report.reportMeanLossModuleLoader module trainLoader "train(before)"
          train.Report.reportMeanLossModuleLoader module testLoader "test(before)")
        ++ train.logLossEvery (α := Float) logEvery
        ++ train.onEpochEnd (α := Float) (fun ev =>
          train.Report.reportMeanLossModuleLoader module testLoader s!"test(epoch {ev.epoch + 1})")
        ++ train.onTrainEnd (α := Float) (fun _ => do
          train.Report.reportMeanLossModuleLoader module trainLoader "train(after)"
          train.Report.reportMeanLossModuleLoader module testLoader "test(after)")

      let (_report, _loader') ← train.fitModuleLoaderWith module opt eb.epochs trainLoader hooks
      pure ())) { printOk := false }
  pure ()

end NN.Examples.Data.Loaders.Cifar10Images
