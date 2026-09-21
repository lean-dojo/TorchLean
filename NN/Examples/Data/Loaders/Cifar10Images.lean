/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Data.RealPaths
public import NN.Examples.Data.SamplePaths

/-!
# CIFAR10-style image loader tutorial (NPY, offline)

This tutorial mirrors a classic PyTorch recipe:

1. Load a labeled image dataset from disk (`.npy` exported from NumPy/PyTorch).
2. Split into train/test.
3. Build a small CNN by explicitly stacking layers.
4. Train for multiple epochs over shuffled minibatches through the public `Trainer` API.

To keep this runnable without network downloads, generate a small deterministic
"CIFAR10-shaped" dataset locally:

- `NN/Examples/Data/small_cifar10like_X.npy`: shape `(200, 3, 32, 32)`, dtype `float32`
- `NN/Examples/Data/small_cifar10like_y.npy`: shape `(200,)`, dtype `float32` labels `0..9`

Generate it with:

`python3 NN/Examples/Data/generate_small_data.py`

Build:

- `lake build NN.Examples.Data.Loaders.Cifar10Images`

Run this tutorial with `lake exe torchlean data_cifar10 --check-only` to validate the files,
or omit `--check-only` to train and print a prediction for a blank image.
The held-out split is reported here; this tutorial does not compute test accuracy.

For command-line CIFAR training, use `torchlean cnn` or `torchlean vit` with
`--x`, `--y`, and `--n-total`.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `NN/Examples/Data`)
- `--real-cifar10` (use `data/real/cifar10/cifar10_train_*.npy`, as prepared by
  `scripts/datasets/download_example_data.py --cifar10`)
- `--x PATH`, `--y PATH` (override the `.npy` files)
- `--n-total N` (number of leading rows to load; default `200`)
- `--seed S` (controls split + shuffling + model initialization)
- `--batch N`
- `--epochs E`
- `--lr LR` (default: `0.001`)
- `--train-size N` (default: up to 160 rows, or up to 16 with `--check-only`)
- `--check-only` (validate paths, tensor shapes, and dataset splitting without training)

Why this tutorial matters:

- it shows the public `Data` file-loading path rather than only in-memory tensors;
- it keeps the model architecture familiar (Conv/ReLU/Pool stack + classifier head);
- it shows the "offline artifact" workflow many PyTorch users already have, where arrays
  have been pre-exported to `.npy` and training happens without any dataset download step.
- it stays on the same public `Trainer` surface as the model examples instead of dropping to the
  callback runner API.
- it shows that the trained result is still usable for immediate inference, not only for a terminal
  loss summary.
-/

@[expose] public section

namespace NN.Examples.Data.Loaders.Cifar10Images

open TorchLean

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
  let conv1 : nn.Convolution.Config 2 :=
    { outChannels := firstHiddenChannels
      kernelSize := [3, 3]
      padding := [1, 1] }
  let conv2 : nn.Convolution.Config 2 :=
    { outChannels := secondHiddenChannels
      kernelSize := [3, 3]
      padding := [1, 1] }
  let pool : nn.Pooling.Config 2 :=
    { kernelSize := [2, 2]
      stride := [2, 2] }
  let spatial1 := conv1.outputSpatial spatial0
  let spatial2 := pool.outputSpatial spatial1
  let spatial3 := conv2.outputSpatial spatial2
  let spatial4 := pool.outputSpatial spatial3
  nn.Sequential![
    nn.conv spatial0 conv1 (batchShape := [batchSize]) (inputChannels := inputChannels),
    nn.relu,
    nn.maxPool spatial1 pool (batchShape := [batchSize]) (channels := firstHiddenChannels),
    nn.conv spatial2 conv2 (batchShape := [batchSize]) (inputChannels := firstHiddenChannels),
    nn.relu,
    nn.maxPool spatial3 pool (batchShape := [batchSize]) (channels := secondHiddenChannels),
    nn.heads.classifier classCount (batchShape := [batchSize])
      (featureShape := (spatial4.to Shape).prependDim secondHiddenChannels)
  ]

/-- Shared offline CIFAR10-like tensor source used by this tutorial. -/
def source (xPath yPath : System.FilePath) (rowCount : Nat) : Data.LabeledSource :=
  Data.LabeledSource.fromFiles xPath yPath rowCount
    [inputChannels, imageHeight, imageWidth] classCount

/-- Command-line help for the CIFAR10-style NPY loader tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean CIFAR10-style NPY loader tutorial"
    , ""
    , "Usage:"
    , "  lake exe torchlean data_cifar10 [options]"
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
    , "  --execution eager|typed-graph"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --show-backend                    print backend capsules as they execute"
    ]

/--
Entry point. `--check-only` loads and validates the data and split without training.
`--real-cifar10` selects prepared real-data paths; neither mode generates missing files.
Both switches are stripped before the remaining flags reach the shared training-flag parser.
-/
def main (args : List String) : IO Unit := do
  let args0 := CLI.dropDashDash args
  if CLI.hasHelp args0 then
    IO.println usage
    return
  let checkOnly := args0.contains "--check-only"
  let realCifar10 := args0.contains "--real-cifar10"
  let args := args0.filter (fun a => a != "--check-only" && a != "--real-cifar10")

  let (dataDir, args) ← CLI.orThrow exeName <| TorchLean.CLI.takePathFlag args "data-dir"
    (default := NN.Examples.Data.SamplePaths.defaultDataDir)
  let (seed, args) ← CLI.orThrow exeName <| CLI.takeSeed args (default := 0)
  let (eb, args) ← CLI.orThrow exeName <|
    CLI.takePositiveEpochBatch args exeName (defaultEpochs := 5) (defaultBatch := 20)
  let (trainSize0, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args
    "train-size" (default := 0)
  let (rowCountOption, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args
    "n-total" (default := defaultRowCount)
  let (lr, args) ← CLI.orThrow exeName <|
    CLI.takePositiveFloatFlag args exeName "lr" (default := 0.001)
  let defaultX :=
    if realCifar10 then
      NN.Examples.Data.RealPaths.cifar10TrainX
    else
      NN.Examples.Data.SamplePaths.cifar10likeXNpy dataDir
  let defaultY :=
    if realCifar10 then
      NN.Examples.Data.RealPaths.cifar10TrainY
    else
      NN.Examples.Data.SamplePaths.cifar10likeYNpy dataDir
  let (xPath, args) ← CLI.orThrow exeName <| CLI.takePathFlag args "x" (default := defaultX)
  let (yPath, args) ← CLI.orThrow exeName <| CLI.takePathFlag args "y" (default := defaultY)
  let rowCount := rowCountOption
  let trainSize :=
    if trainSize0 = 0 then
      (if checkOnly then Nat.min 16 rowCount else Nat.min 160 rowCount)
    else
      trainSize0
  let trainSteps : Nat := eb.epochs * (trainSize / eb.batchSize)
  let run ← TorchLean.CLI.Trainer.parseCommandLine exeName args
    { optimizer := optim.adam { learningRate := lr } }
  let trainer := Trainer.new (model (batchSize := eb.batchSize)) <|
    Trainer.RunConfig.forObjective run
      (.oneHotCrossEntropy 1)
      (seed := seed)

  IO.println "== CIFAR10-style NPY CNN tutorial =="
  IO.println s!"data_dir   = {dataDir}"
  IO.println s!"x_path     = {xPath}"
  IO.println s!"y_path     = {yPath}"
  IO.println s!"rows       = {rowCount}"
  IO.println s!"seed       = {seed}"
  IO.println s!"train_size = {trainSize} / {rowCount}"
  trainer.printSummary
  IO.println <|
    (s!"train      = Adam(lr={lr}), epochs={eb.epochs}, " ++
      s!"batch_size={eb.batchSize}, shuffle=true, drop_last=true, steps={trainSteps}")
  if checkOnly then
    IO.println "mode       = --check-only (validate paths, tensor shapes, and dataset split)"
  (← IO.getStdout).flush
  let dsAll ← Data.LabeledSource.load (α := Float) (source xPath yPath rowCount)

  if trainSize > dsAll.size then
    throw <| IO.userError
      s!"{exeName}: --train-size {trainSize} exceeds dataset size {dsAll.size}"

  let split := Data.SampleStream.randomSplitAt seed trainSize dsAll
  let dsTrain := split.selected
  let dsTest := split.remaining

  if checkOnly then
    IO.println s!"loaded     = {dsAll.size} image rows"
    IO.println s!"split      = train {dsTrain.size}, test {dsTest.size}"
    IO.println "check      = dataset shape/path runtime check passed"
    pure ()
  else
    if trainSize < eb.batchSize then
      throw <| IO.userError
        s!"{exeName}: --train-size {trainSize} is smaller than --batch {eb.batchSize}"
    let trainData :=
      Data.batch eb.batchSize (Data.fromSamples dsTrain.toArray)
        (shuffle := true) (seed := seed)
    let trained ← trainer.train trainData
      { steps := trainSteps
        logTitle := "CIFAR10-style NPY CNN tutorial"
        logNotes :=
          #[s!"x={xPath}", s!"y={yPath}", s!"rows={rowCount}",
            s!"train_size={trainSize}", s!"test_size={dsTest.size}",
            s!"epochs={eb.epochs}", s!"batch={eb.batchSize}", s!"lr={lr}"] }
    trained.printSummary
    let blank : Tensor Float [inputChannels, imageHeight, imageWidth] :=
      Tensor.full [inputChannels, imageHeight, imageWidth] 0.0
    trained.printPrediction "blank" (Tensor.repeatAxis 0 eb.batchSize blank)

end NN.Examples.Data.Loaders.Cifar10Images
