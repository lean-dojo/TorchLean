/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Trainer.Constructor
public import TorchLean.NN.Trainer.Train
public import TorchLeanExamples.Data.RealPaths
public import TorchLeanExamples.Data.SamplePaths
public import TorchLean.NN.CLI.Trainer
public import TorchLean.NN.Data.Dataset

/-!
# CIFAR10-style image loader tutorial (NPY, offline)

The tutorial mirrors a classic PyTorch recipe:

1. Load a labeled image dataset from disk (`.npy` exported from NumPy/PyTorch).
2. Split into train/test.
3. Build a small CNN by stacking layers one by one.
4. Train for multiple epochs over shuffled minibatches through the public `Trainer` API.

To keep this runnable without network downloads, generate a small deterministic
"CIFAR10-shaped" dataset locally:

- `TorchLeanExamples/Data/small_cifar10like_X.npy`: shape `(200, 3, 32, 32)`, dtype `float32`
- `TorchLeanExamples/Data/small_cifar10like_y.npy`: shape `(200,)`, dtype `float32` labels `0..9`

Generate it with:

`python3 TorchLeanExamples/Data/generate_small_data.py`

Build:

- `scripts/lake.sh build TorchLeanExamples.Data.Loaders.CifarImages`

Run `scripts/lake.sh exe torchlean data_cifar10 --check-only` to validate the files, or leave out
`--check-only` to train and print a prediction for a blank image. The run reports the size of the
held-out split but computes no test accuracy.

For command-line CIFAR training, use `scripts/lake.sh exe torchlean cnn` or
`scripts/lake.sh exe torchlean vit` with `--x`, `--y`, and `--n-total`.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `TorchLeanExamples/Data`, or `data/real` with `--real-cifar10`)
- `--real-cifar10` (use `cifar10/cifar10_train_*.npy` under the data directory, as prepared by
  `scripts/datasets/download_example_data.py --cifar10`)
- `--x PATH`, `--y PATH` (override the `.npy` files)
- `--n-total N` (number of leading rows to load, default `200`)
- `--seed S` (controls split + shuffling + model initialization)
- `--batch N`
- `--epochs E`
- `--lr LR` (default: `0.001`)
- `--train-size N` (default: up to 160 rows, or up to 16 with `--check-only`)
- `--check-only` (validate paths, tensor shapes, and dataset splitting without training)

The data comes in through the public `Data` file-loading path, and the model is the familiar
Conv/ReLU/Pool stack with a classifier head. The arrays are pre-exported to `.npy`, the offline
workflow many PyTorch users already have, so training needs no dataset download step. Training
uses the same public `Trainer` surface as the model examples, without dropping to the callback
runner API, and the trained result makes a prediction right after its loss summary.
-/

@[expose] public section

namespace TorchLean.Examples.CifarImages

open Trainer.Objective

/-- Command name used in diagnostics and by the top-level example runner. -/
def exeName : String := "data_cifar10"

/-- CIFAR-10 has ten classes. -/
def classCount : Nat := 10
/-- RGB input. -/
def inputChannels : Nat := 3
/-- CIFAR-10 images are 32 by 32. -/
def imageHeight : Nat := 32
/-- Width, matching `imageHeight`. -/
def imageWidth : Nat := 32
/-- Default number of leading images loaded from pre-generated or downloaded files. -/
def defaultRowCount : Nat := 200

/-- Small CNN (no BatchNorm): Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Linear(10). -/
def model {batchSize : Nat} :
    nn.Builder (nn.Sequential
      [batchSize, inputChannels, imageHeight, imageWidth]
      [batchSize, classCount]) :=
  letI : NeZero inputChannels := ⟨by decide⟩
  let firstHiddenChannels : Nat := 16
  let secondHiddenChannels : Nat := 32
  let spatial0 : Tensor Nat [2] := [imageHeight, imageWidth]
  let conv1 : nn.Conv.Config 2 :=
    { outChannels := firstHiddenChannels
      kernelSize := [3, 3]
      padding := [1, 1] }
  let conv2 : nn.Conv.Config 2 :=
    { outChannels := secondHiddenChannels
      kernelSize := [3, 3]
      padding := [1, 1] }
  let pool : Window 2 :=
    { kernelSize := [2, 2]
      stride := [2, 2] }
  let spatial1 := conv1.convOutputSpatial spatial0
  let spatial2 := pool.poolOutputSpatial spatial1
  let spatial3 := conv2.convOutputSpatial spatial2
  let spatial4 := pool.poolOutputSpatial spatial3
  nn.Sequential![
    nn.conv spatial0 conv1 (batchShape := [batchSize]) (inputChannels := inputChannels),
    nn.op .relu,
    nn.maxPool spatial1 pool (batchShape := [batchSize]) (channels := firstHiddenChannels),
    nn.conv spatial2 conv2 (batchShape := [batchSize]) (inputChannels := firstHiddenChannels),
    nn.op .relu,
    nn.maxPool spatial3 pool (batchShape := [batchSize]) (channels := secondHiddenChannels),
    nn.classifierHead classCount (batchShape := [batchSize])
      (featureShape := Shape.dim secondHiddenChannels (spatial4.to Shape))
  ]

/--
Labeled image source over the first `rowCount` rows of `xPath` and `yPath`, read as `3 × 32 × 32`
images with ten classes.
-/
def source (xPath yPath : System.FilePath) (rowCount : Nat) : Data.LabeledSource :=
  Data.LabeledSource.fromFiles xPath yPath rowCount
    [inputChannels, imageHeight, imageWidth] classCount

/-- Command-line help for the CIFAR10-style NPY loader tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean CIFAR10-style NPY loader tutorial"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean data_cifar10 [options]"
    , ""
    , "Options:"
    , "  --data-dir PATH"
    , "  --real-cifar10"
    , "  --x PATH"
    , "  --y PATH"
    , "  --n-total N"
    , "  --train-size N"
    , "  --seed N"
    , "  --epochs N"
    , "  --batch N"
    , "  --lr LR"
    , "  --check-only"
    , "  --arithmetic native|ieee"
    , s!"  --device auto|{Backend.Device.help}"
    , "  --show-backend                    print backend capsules as they are selected"
    ]

/--
Entry point. With `--check-only` it loads the data and checks shapes and the split without
training, and `--real-cifar10` selects the prepared CIFAR-10 arrays under the data directory,
`data/real` unless `--data-dir` names another. Neither switch generates missing files, and both are
removed before the remaining flags reach the shared training-flag parser.
-/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let checkOnly := args.contains "--check-only"
  let realCifar10 := args.contains "--real-cifar10"
  let args := args.filter (fun a => a != "--check-only" && a != "--real-cifar10")

  let (dataDir, args) ← CLI.orThrow exeName <| CLI.takePathFlag args "data-dir"
    (default := if realCifar10 then Examples.RealData.defaultDataDir
      else Examples.SampleData.defaultDataDir)
  let (seed, args) ← CLI.orThrow exeName <| CLI.takeSeed args (default := 0)
  let (epochs, args) ← CLI.orThrow exeName <|
    CLI.takePositiveNatFlag args exeName "epochs" (default := 5)
  let (batchSize, args) ← CLI.orThrow exeName <|
    CLI.takePositiveNatFlag args exeName "batch" (default := 20)
  let (requestedTrainSize, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args
    "train-size" (default := 0)
  let (rowCount, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args
    "n-total" (default := defaultRowCount)
  let (lr, args) ← CLI.orThrow exeName <|
    CLI.takePositiveFloatFlag args exeName "lr" (default := 0.001)
  let defaultX :=
    if realCifar10 then
      Examples.RealData.cifar10TrainX dataDir
    else
      Examples.SampleData.cifar10LikeX dataDir
  let defaultY :=
    if realCifar10 then
      Examples.RealData.cifar10TrainY dataDir
    else
      Examples.SampleData.cifar10LikeY dataDir
  let (xPath, args) ← CLI.orThrow exeName <| CLI.takePathFlag args "x" (default := defaultX)
  let (yPath, args) ← CLI.orThrow exeName <| CLI.takePathFlag args "y" (default := defaultY)
  let trainSize :=
    if requestedTrainSize = 0 then
      (if checkOnly then Nat.min 16 rowCount else Nat.min 160 rowCount)
    else
      requestedTrainSize
  let trainSteps : Nat := epochs * (trainSize / batchSize)
  let run ← Trainer.RunConfig.parseCommandLine exeName args
    { optimizer := optim.adam { lr := lr } }
  let trainer := Trainer.create (model (batchSize := batchSize)) <|
    { run with objective := oneHotCrossEntropy 1, seed }

  IO.println "== CIFAR10-style NPY CNN tutorial =="
  IO.println s!"data_dir   = {dataDir}"
  IO.println s!"x_path     = {xPath}"
  IO.println s!"y_path     = {yPath}"
  IO.println s!"rows       = {rowCount}"
  IO.println s!"seed       = {seed}"
  IO.println s!"train_size = {trainSize} / {rowCount}"
  trainer.printSummary
  IO.println <|
    (s!"train      = Adam(lr={lr}), epochs={epochs}, " ++
      s!"batch_size={batchSize}, shuffle=true, drop_last=true, steps={trainSteps}")
  if checkOnly then
    IO.println "mode       = --check-only (validate paths, tensor shapes, and dataset split)"
  (← IO.getStdout).flush
  let samples ← Data.LabeledSource.load (α := Float) (source xPath yPath rowCount)

  if trainSize > samples.size then
    throw <| IO.userError
      s!"{exeName}: --train-size {trainSize} exceeds dataset size {samples.size}"

  let split := Data.SampleStream.randomSplitAt seed trainSize samples
  let trainSamples := split.selected
  let testSamples := split.remaining

  if checkOnly then
    IO.println s!"loaded     = {samples.size} image rows"
    IO.println s!"split      = train {trainSamples.size}, test {testSamples.size}"
    IO.println "check      = dataset shape/path runtime check passed"
    pure ()
  else
    if trainSize < batchSize then
      throw <| IO.userError
        s!"{exeName}: --train-size {trainSize} is smaller than --batch {batchSize}"
    let trainData :=
      Data.batch batchSize (Data.fromSamples trainSamples.toArray)
        (shuffle := true) (seed := seed)
    let trained ← trainer.train trainData
      { steps := trainSteps
        logTitle := "CIFAR10-style NPY CNN tutorial"
        logNotes :=
          #[s!"x={xPath}", s!"y={yPath}", s!"rows={rowCount}",
            s!"train_size={trainSize}", s!"test_size={testSamples.size}",
            s!"epochs={epochs}", s!"batch={batchSize}"] }
    trained.printSummary
    let blank : Tensor Float [inputChannels, imageHeight, imageWidth] :=
      Tensor.full [inputChannels, imageHeight, imageWidth] 0.0
    trained.printPrediction "blank" (Tensor.repeatAxis 0 batchSize blank)

end TorchLean.Examples.CifarImages
