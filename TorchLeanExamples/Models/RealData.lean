/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Data.RealPaths
public import TorchLeanExamples.Models.TrainCommand
public import TorchLean.NN.Data.Dataset
public import TorchLean.NN.Models.Generative
public import TorchLean.NN.Trainer.Constructor
public import TorchLean.NN.Trainer.Train

/-!
# Shared Real-Data Helpers for Model Examples

The model examples train on real data. The pieces they share live here: CIFAR-10 and ImageNet-style
NPY loaders and crops, the flag parsers for those datasets and for household-power forecasting
windows, a local text-corpus reader, and the "how to prepare data" hints.

`scripts/datasets/download_example_data.py` prepares the data files, except the ImageNet-style
arrays, which come from `scripts/datasets/torchlean_data_convert.py`. A missing input stops the
command with a preparation hint, so no example trains on synthetic tensors by accident.
-/

@[expose] public section

open TorchLean.Trainer.Objective

namespace TorchLean.Examples.RealData

/-- Number of channels in the prepared CIFAR-10 image tensors. -/
def cifarChannels : Nat := 3

/-- Height of the prepared CIFAR-10 image tensors. -/
def cifarHeight : Nat := 32

/-- Width of the prepared CIFAR-10 image tensors. -/
def cifarWidth : Nat := 32

/-- Number of CIFAR-10 classes, hence the width of one-hot targets. -/
def cifarClasses : Nat := 10

/-- Default row budget for CIFAR-10 model example commands. -/
def defaultCifarRows : Nat := 1

/-- Number of channels in converted ImageNet-style image tensors. -/
def imageNet64Channels : Nat := 3

/-- Height of converted ImageNet-style image tensors. -/
def imageNet64Height : Nat := 64

/-- Width of converted ImageNet-style image tensors. -/
def imageNet64Width : Nat := 64

/-- Number of ImageNet-style classes expected by the converted label path. -/
def imageNet64Classes : Nat := 1000

/-- Default row budget for ImageNet64 model example runs. -/
def defaultImageNet64Rows : Nat := 200

instance : NeZero cifarChannels := ⟨by decide⟩
instance : NeZero cifarHeight := ⟨by decide⟩
instance : NeZero cifarWidth := ⟨by decide⟩
instance : NeZero cifarClasses := ⟨by decide⟩
instance : NeZero imageNet64Channels := ⟨by decide⟩
instance : NeZero imageNet64Height := ⟨by decide⟩
instance : NeZero imageNet64Width := ⟨by decide⟩
instance : NeZero imageNet64Classes := ⟨by decide⟩

/-- Shape of one CIFAR-10 image after conversion to channel-first layout. -/
abbrev cifarImage : Shape :=
  [cifarChannels, cifarHeight, cifarWidth]

/-- One-hot CIFAR-10 target shape. -/
abbrev cifarTarget : Shape :=
  [cifarClasses]

/-- Top-left `cropHeight × cropWidth` crop of a CIFAR image batch, given proofs that it fits. -/
def cropCifarImagesOfLE (batchSize cropHeight cropWidth : Nat)
    (heightFits : cropHeight ≤ cifarHeight) (widthFits : cropWidth ≤ cifarWidth)
    (x : Tensor Float [batchSize, cifarChannels, cifarHeight, cifarWidth]) :
    Tensor Float [batchSize, cifarChannels, cropHeight, cropWidth] :=
  let rows : Tensor Float [batchSize, cifarChannels, cropHeight, cifarWidth] := by
    simpa using Tensor.take x 2 cropHeight heightFits
  Tensor.take rows 3 cropWidth widthFits

/--
Top-left `cropHeight × cropWidth` crop of a CIFAR image batch, or an error message when the crop is
larger than the 32 × 32 image.
-/
def cropCifarImages (batchSize cropHeight cropWidth : Nat)
    (x : Tensor Float [batchSize, cifarChannels, cifarHeight, cifarWidth]) :
    Except String (Tensor Float [batchSize, cifarChannels, cropHeight, cropWidth]) :=
  if heightFits : cropHeight ≤ cifarHeight then
    if widthFits : cropWidth ≤ cifarWidth then
      .ok (cropCifarImagesOfLE batchSize cropHeight cropWidth
        heightFits widthFits x)
    else
      .error s!"CIFAR crop width {cropWidth} exceeds image width {cifarWidth}"
  else
    .error s!"CIFAR crop height {cropHeight} exceeds image height {cifarHeight}"

/-- Crop a CIFAR minibatch while leaving the one-hot class labels unchanged. -/
def cropCifarBatch (batchSize cropHeight cropWidth : Nat)
    (sample : Sample.Batch Float batchSize cifarImage cifarTarget) :
    Except String (Sample.Supervised Float
      [batchSize, cifarChannels, cropHeight, cropWidth] [batchSize, cifarClasses]) := do
  let input ← cropCifarImages batchSize cropHeight cropWidth sample.input
  pure { input, target := sample.target }

/-- ImageNet-style converted image shape used by the higher-resolution diffusion example. -/
abbrev imageNet64Image : Shape :=
  [imageNet64Channels, imageNet64Height, imageNet64Width]

/--
One-hot target shape for ImageNet-style folders.

The diffusion example ignores labels, but reusing `Data.LabeledSource` keeps the data path identical
to the supervised examples and lets class-directory conversion catch malformed labels early.
-/
abbrev imageNet64Target : Shape :=
  [imageNet64Classes]

/-- Error message shown when a CIFAR-backed example cannot find the prepared arrays. -/
def missingCifarHint : String :=
  "Prepare real CIFAR-10 arrays with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --cifar10\n" ++
  "Then rerun the model command."

/-- Error message shown when an ImageNet64-backed example cannot find the prepared arrays. -/
def missingImageNet64Hint : String :=
  "Prepare an ImageNet-style 64x64 subset with:\n" ++
  "  python3 scripts/datasets/torchlean_data_convert.py image-folder \\\n" ++
  "    --input /path/to/imagenet/train \\\n" ++
  "    --x-output data/real/imagenet64/imagenet64_train_X.npy \\\n" ++
  "    --y-output data/real/imagenet64/imagenet64_train_y.npy \\\n" ++
  "    --height 64 --width 64 --labels-from-dirs --limit 2000\n" ++
  "This path expects a local ImageNet-style image-folder dataset."

/-- Error message shown when a text-model example cannot find a corpus. -/
def missingTextHint : String :=
  Examples.RealData.missingTinyShakespeareHint ++ "\nor pass --data-file PATH."

/-- Error message shown when the Auto MPG CSV is missing. -/
def missingAutoMPGHint : String :=
  "Prepare the Auto MPG CSV with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --auto-mpg\n" ++
  "Then rerun the model command."

/-- Error message shown when the household-power forecasting dataset is missing. -/
def missingHouseholdPowerHint : String :=
  "Prepare the household-power forecasting windows with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --household-power " ++
  "--household-power-windows 512\n" ++
  "Then rerun the model command."

/-- Parse the shared NPY data flags with CIFAR-10's default paths and row count filled in. -/
def parseCifar (args : List String) :
    Except String (NPY.Options × List String) :=
  NPY.Options.parse args
    Examples.RealData.cifar10TrainX
    Examples.RealData.cifar10TrainY
    defaultCifarRows

/--
Parse the shared flags for an ImageNet-style 64x64 NPY dataset, with 200 rows by default.

`scripts/datasets/torchlean_data_convert.py image-folder` produces the input. The converter handles
JPEG/PNG decoding, RGB conversion, resizing, class-directory labels and the final NCHW layout, so
Lean reads only plain `.npy` tensors.
-/
def parseImageNet64 (args : List String) :
    Except String (NPY.Options × List String) :=
  NPY.Options.parse args
    Examples.RealData.imageNet64TrainX
    Examples.RealData.imageNet64TrainY
    defaultImageNet64Rows

end TorchLean.Examples.RealData

namespace TorchLean.Examples.Forecast.Options

/--
Parse the household-power forecasting flags: `--data-dir`, a positive `--windows` (default
`defaultWindows`), `--report-offset` (default `defaultReportOffset`), and `--x` and `--y`, whose
defaults sit under `--data-dir`.
-/
def parse
    (exeName : String)
    (args : List String)
    (defaultWindows : Nat := 512)
    (defaultReportOffset : Nat := 96) :
    Except String (Forecast.Options × List String) := do
  let (dataDir, args) ← CLI.takePathFlag args "data-dir"
    (default := Examples.RealData.defaultDataDir)
  let (windows, args) ← CLI.takePositiveNatFlag args exeName "windows" (default := defaultWindows)
  let (reportOffset, args) ← CLI.takeNatFlag args "report-offset" (default := defaultReportOffset)
  let (xPath, args) ←
    CLI.takePathFlag args "x" (default := (Examples.RealData.householdPowerX dataDir))
  let (yPath, args) ←
    CLI.takePathFlag args "y" (default := (Examples.RealData.householdPowerY dataDir))
  pure ({ xPath := xPath
          yPath := yPath
          windows := windows
          reportOffset := reportOffset }, args)

end TorchLean.Examples.Forecast.Options

namespace TorchLean.Examples.RealData

/-- Check that both files exist, load `rowCount` labeled images and wrap them in a shuffled loader.
A failed load names `--n-total` and the dataset's recovery hint. -/
def loadImageLoader (imageShape : Shape) (classes : Nat)
    (datasetName missingHint recoveryHint exeName : String)
    (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Data.Loader Float batchSize imageShape [classes]) := do
  Data.requirePairedFiles exeName
    s!"{datasetName} images" xPath s!"{datasetName} labels" yPath missingHint
  let source := Data.LabeledSource.fromFiles xPath yPath rowCount imageShape classes
  let samples ←
    try
      source.load (α := Float)
    catch error =>
      throw <| IO.userError <|
        s!"{exeName}: failed to load {datasetName} arrays for --n-total {rowCount}.\n" ++
        s!"{error}\n" ++
        "If your local .npy files contain fewer rows, pass --n-total with that row count; " ++
        recoveryHint
  pure (Data.Loader.fromStream samples batchSize (shuffle := true) (seed := seed))

/--
Build a shuffled, batched CIFAR-10 loader from the image and label `.npy` files.

Both files must exist before loading starts, so a missing dataset fails at once with the hint that
names `download_example_data.py --cifar10`.
-/
def loadCifarLoader
    (exeName : String) (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Data.Loader Float batchSize cifarImage cifarTarget) :=
  loadImageLoader cifarImage cifarClasses "CIFAR-10" missingCifarHint
    ("to regenerate the prepared slice, run:\n" ++
      "  python3 scripts/datasets/download_example_data.py --cifar10")
    exeName batchSize rowCount seed xPath yPath

/-- TrainLog notes for NPY training: the dataset name, paths and row count, the batch size, then
`extra`. -/
def trainingNotes (dataset : String) (batchSize : Nat)
    (flags : Training.Options NPY.Options) (extra : Array String := #[]) :
    Array String :=
  NPY.Options.logNotes flags.data dataset ++
  #[s!"batch={batchSize}"]
  ++ extra

/-- Load one shuffled epoch of full CIFAR-10 minibatches from prepared `.npy` arrays. -/
def loadCifarBatches
    (exeName : String) (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Array (Sample.Batch Float batchSize cifarImage cifarTarget)) := do
  let loader ← loadCifarLoader exeName batchSize rowCount seed xPath yPath
  let epoch ← CLI.orThrow exeName <|
    Data.Loader.nextEpoch exeName loader (requireBatch := true)
  pure epoch.batches

/-- Train a cropped CIFAR classifier with the examples' shared Adam and cross-entropy policy. -/
def trainCifarClassifier (batchSize cropHeight cropWidth : Nat)
    (exeName logTitle : String)
    (model : nn.Builder (nn.Sequential
      [batchSize, cifarChannels, cropHeight, cropWidth] [batchSize, cifarClasses]))
    (runtime : Session.Config) (flags : Training.Options NPY.Options)
    (extraNotes : Array String := #[]) : IO Trainer.Report := do
  let batches ←
    loadCifarBatches exeName batchSize flags.data.nRows runtime.seed
      flags.data.xPath flags.data.yPath
  let batches ← batches.mapM fun sample =>
    CLI.orThrow exeName <| cropCifarBatch batchSize cropHeight cropWidth sample
  let trainer :=
    Trainer.create model <|
      { runtime with
          optimizer := optim.adam { lr := flags.training.lr }
          objective := oneHotCrossEntropy 1 }
  let trained ← trainer.train
    (Data.fromSamples batches)
    (flags.training.trainOptions
      (logTitle := logTitle)
      (logNotes := trainingNotes "cifar10" batchSize flags extraNotes))
  pure trained.report

/-- Load the first full CIFAR-10 minibatch from the shared CIFAR loader. -/
def loadCifarBatch
    (exeName : String) (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Sample.Batch Float batchSize cifarImage cifarTarget) := do
  let loader ← loadCifarLoader exeName batchSize rowCount seed xPath yPath
  CLI.orThrow exeName <| Data.Loader.firstBatch exeName loader

/--
Build a shuffled, batched loader over user-prepared ImageNet-style `64x64` arrays.

The loader reads prepared `.npy` arrays. The Python converter is the trust boundary for image
decoding and resizing, and the Lean side checks the tensor shape and class range of what it reads.
-/
def loadImageNet64Loader
    (exeName : String) (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Data.Loader Float batchSize imageNet64Image imageNet64Target) :=
  loadImageLoader imageNet64Image imageNet64Classes "ImageNet64" missingImageNet64Hint
    "to create ImageNet64 arrays, run the image-folder converter described in the error hint."
    exeName batchSize rowCount seed xPath yPath

/-- Load one shuffled epoch of full ImageNet64-style minibatches from prepared `.npy` arrays. -/
def loadImageNet64Batches
    (exeName : String) (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Array (Sample.Batch Float batchSize imageNet64Image imageNet64Target)) := do
  let loader ← loadImageNet64Loader exeName batchSize rowCount seed xPath yPath
  let epoch ← CLI.orThrow exeName <|
    Data.Loader.nextEpoch exeName loader (requireBatch := true)
  pure epoch.batches

/--
Load the first CIFAR minibatch, flatten each channel-first image and keep its first
`config.dataWidth` values, a plain prefix of the image buffer. A width above the 3072 values of a
full image is an error.
-/
def loadCifarFeatureBatch (batchSize : Nat) (config : nn.Generative.Config)
    (exeName : String) (xPath yPath : System.FilePath) (rowCount seed : Nat) :
    IO (Tensor Float (config.dataShape [batchSize])) := do
  let batchSample ← loadCifarBatch exeName batchSize rowCount seed xPath yPath
  if hData : config.dataWidth ≤ cifarImage.size then
    pure (Tensor.flattenThenTake [batchSize] config.dataWidth hData batchSample.input)
  else
    throw <| IO.userError
      (s!"{exeName}: requested feature width {config.dataWidth} "
        ++ s!"exceeds CIFAR image size {cifarImage.size}")

/--
One-sample dataset for compact vector generative models, built from a flattened CIFAR batch.

It loads one real CIFAR batch, keeps the first `config.dataWidth` values of each image and turns
them into one supervised sample with `sampleOfFeatures`. The sample is built in `Float`, and
`Data.defer` casts it with `ofFloat` to the runtime's scalar type when the trainer
materializes the dataset, so the command runs on the ordinary runtime backends.
-/
def cifarFeatureDataset {τ : Shape}
    (batchSize : Nat)
    (config : nn.Generative.Config)
    (exeName : String)
    (sampleOfFeatures : Tensor Float (config.dataShape [batchSize]) →
      Sample.Supervised Float (config.dataShape [batchSize]) τ)
    (xPath yPath : System.FilePath) (rowCount seed : Nat) :
    Trainer.Dataset (config.dataShape [batchSize]) τ :=
  Data.defer do
    let x ← loadCifarFeatureBatch batchSize config exeName xPath yPath rowCount seed
    pure (sampleOfFeatures x)

end TorchLean.Examples.RealData

namespace TorchLean.Examples.Corpus

/-- Help lines for the text-corpus flags `--data-file` and `--tiny-shakespeare`. -/
def help : Array String :=
  #[ "  --data-file PATH      read a local UTF-8 text corpus"
  , "  --tiny-shakespeare    use data/real/text/tiny_shakespeare.txt"
  ]

/--
Parse the shared `--data-file` flag of the local text-model examples, which defaults to the Tiny
Shakespeare corpus.

`--tiny-shakespeare` is accepted and dropped, so it selects that default unless `--data-file` is
also given.
-/
def parse (args : List String) :
    Except String (System.FilePath × List String) := do
  let args := args.filter (fun a => a != "--tiny-shakespeare")
  CLI.takePathFlag args "data-file" (default := Examples.RealData.tinyShakespeare)

/-- Read the selected text corpus. A missing file fails with `Examples.RealData.missingTextHint`,
and an empty file fails with its path. -/
def read (exeName : String) (path : System.FilePath) :
    IO String := do
  unless (← path.pathExists) do
    throw <| IO.userError s!"{exeName}: missing text corpus: {path}\n{RealData.missingTextHint}"
  let text ← IO.FS.readFile path
  if text.isEmpty then
    throw <| IO.userError s!"{exeName}: empty text corpus: {path}"
  pure text

/-- Text corpus selection plus the number of causal training windows to expose. -/
structure Options where
  /-- Selected UTF-8 corpus. -/
  corpus : System.FilePath
  /-- Number of approximately evenly spaced windows in the finite training dataset. -/
  windows : Nat
deriving Repr

section

open Corpus.Options (corpus windows)

/-- Help lines for the corpus flags plus `--windows N`, whose help shows `defaultWindows`. -/
def Options.help (defaultWindows : Nat) : Array String :=
  Corpus.help ++
    #[s!"  --windows N           corpus windows available to training (default: {defaultWindows})"]

/-- Parse a text corpus and a positive `--windows` count. -/
def Options.parse (exeName : String) (defaultWindows : Nat)
    (args : List String) :
    Except String (Corpus.Options × List String) := do
  let (corpus, args) ← Corpus.parse args
  let (windows, args) ←
    CLI.takePositiveNatFlag args exeName "windows" (default := defaultWindows)
  pure ({ corpus, windows }, args)

/-- Read the selected text corpus. -/
def Options.read (exeName : String) (flags : Corpus.Options) :
    IO String :=
  Corpus.read exeName flags.corpus

end

end TorchLean.Examples.Corpus
