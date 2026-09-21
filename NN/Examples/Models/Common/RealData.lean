/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Data.RealPaths
public import NN.Examples.Models.Common.Train

/-!
# Shared Real-Data Helpers for Model Examples

The model examples should exercise real data paths. We keep the shared pieces here:

- loading a prepared CIFAR-10 NPY minibatch,
- reading a local text corpus, and
- printing the same "how to prepare data" hint everywhere.

The data files are prepared by `scripts/datasets/download_example_data.py`; examples report missing
inputs explicitly instead of silently falling back to synthetic tensors.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.RealData

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
def imagenet64Channels : Nat := 3

/-- Height of converted ImageNet-style image tensors. -/
def imagenet64Height : Nat := 64

/-- Width of converted ImageNet-style image tensors. -/
def imagenet64Width : Nat := 64

/-- Number of ImageNet-style classes expected by the converted label path. -/
def imagenet64Classes : Nat := 1000

/-- Default row budget for ImageNet64 model example runs. -/
def defaultImageNet64Rows : Nat := 200

instance : NeZero cifarChannels := ⟨by decide⟩
instance : NeZero cifarHeight := ⟨by decide⟩
instance : NeZero cifarWidth := ⟨by decide⟩
instance : NeZero cifarClasses := ⟨by decide⟩
instance : NeZero imagenet64Channels := ⟨by decide⟩
instance : NeZero imagenet64Height := ⟨by decide⟩
instance : NeZero imagenet64Width := ⟨by decide⟩
instance : NeZero imagenet64Classes := ⟨by decide⟩

/-- Shape of one CIFAR-10 image after conversion to channel-first layout. -/
abbrev CifarImage : Shape :=
  [cifarChannels, cifarHeight, cifarWidth]

/-- One-hot CIFAR-10 target shape. -/
abbrev CifarTarget : Shape :=
  [cifarClasses]

namespace Internal

/-- Take the top-left `cropHeight × cropWidth` view of a CIFAR image batch. -/
def cropCifarImages (batchSize cropHeight cropWidth : Nat)
    (heightFits : cropHeight ≤ cifarHeight) (widthFits : cropWidth ≤ cifarWidth)
    (x : Tensor Float [batchSize, cifarChannels, cifarHeight, cifarWidth]) :
    Tensor Float [batchSize, cifarChannels, cropHeight, cropWidth] :=
  let rows : Tensor Float [batchSize, cifarChannels, cropHeight, cifarWidth] := by
    simpa [Spec.Shape.replaceAxis] using Tensor.take x 2 cropHeight heightFits
  Tensor.take rows 3 cropWidth widthFits

end Internal

/--
Take the top-left `cropHeight × cropWidth` view of a CIFAR image batch.

Invalid crop sizes are reported at the executable boundary.
-/
def cropCifarImages (batchSize cropHeight cropWidth : Nat)
    (x : Tensor Float [batchSize, cifarChannels, cifarHeight, cifarWidth]) :
    Except String (Tensor Float [batchSize, cifarChannels, cropHeight, cropWidth]) :=
  if heightFits : cropHeight ≤ cifarHeight then
    if widthFits : cropWidth ≤ cifarWidth then
      .ok (Internal.cropCifarImages batchSize cropHeight cropWidth
        heightFits widthFits x)
    else
      .error s!"CIFAR crop width {cropWidth} exceeds image width {cifarWidth}"
  else
    .error s!"CIFAR crop height {cropHeight} exceeds image height {cifarHeight}"

/-- Crop a CIFAR minibatch while leaving the one-hot class labels unchanged. -/
def cropCifarBatch (batchSize cropHeight cropWidth : Nat)
    (sample : Sample.Batch Float batchSize CifarImage CifarTarget) :
    Except String (Sample.Supervised Float
      [batchSize, cifarChannels, cropHeight, cropWidth] [batchSize, cifarClasses]) := do
  let input ← cropCifarImages batchSize cropHeight cropWidth sample.input
  pure { input, target := sample.target }

/-- ImageNet-style converted image shape used by the higher-resolution diffusion example. -/
abbrev ImageNet64Image : Shape :=
  [imagenet64Channels, imagenet64Height, imagenet64Width]

/--
One-hot target shape for ImageNet-style folders.

The diffusion example ignores labels, but reusing `Data.LabeledSource` keeps the data path identical
to the supervised examples and lets class-directory conversion catch malformed labels early.
-/
abbrev ImageNet64Target : Shape :=
  [imagenet64Classes]

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
  "Prepare the text corpus with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --tiny-shakespeare\n" ++
  "or pass --data-file PATH."

/-- Error message shown when the Auto MPG CSV is missing. -/
def missingAutoMpgHint : String :=
  "Prepare the Auto MPG CSV with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --auto-mpg\n" ++
  "Then rerun the model command."

/-- Error message shown when the household-power forecasting dataset is missing. -/
def missingHouseholdPowerHint : String :=
  "Prepare the household-power forecasting windows with:\n" ++
  "  python3 scripts/datasets/download_example_data.py --household-power " ++
  "--household-power-windows 512\n" ++
  "Then rerun the model command."

namespace NpyDatasets

/-- Parse the shared NPY data flags with CIFAR-10's default paths and row count filled in. -/
def parseCifar (args : List String) :
    Except String (Support.NpyDataFlags × List String) := do
  Support.NpyDataFlags.parse args
    NN.Examples.Data.RealPaths.cifar10TrainX
    NN.Examples.Data.RealPaths.cifar10TrainY
    defaultCifarRows

/--
Parse the shared flags for an ImageNet-style 64x64 NPY dataset.

The expected input is produced by `scripts/datasets/torchlean_data_convert.py image-folder`; that
converter handles JPEG/PNG decoding, RGB conversion, resizing, class-directory labels, and the final
NCHW layout. Lean then reads only the simple `.npy` tensors.
-/
def parseImageNet64 (args : List String) :
    Except String (Support.NpyDataFlags × List String) := do
  Support.NpyDataFlags.parse args
    NN.Examples.Data.RealPaths.imagenet64TrainX
    NN.Examples.Data.RealPaths.imagenet64TrainY
    defaultImageNet64Rows

end NpyDatasets

/-- Parsed CIFAR dataset and fixed-sample training flags for runnable model examples. -/
abbrev CifarLoggedTrainFlags := Support.NpyLoggedTrainFlags

/-- Parsed CIFAR dataset and optimizer/training flags for classifier examples. -/
abbrev CifarModelTrainFlags := Support.NpyModelTrainFlags

namespace CifarLoggedTrainFlags

/--
Parse the standard CIFAR plus fixed-step training flags and reject unused arguments.

Generative examples use the same prepared CIFAR arrays and the same loss-curve logging contract;
only the model and target construction differ.
-/
def parse (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 10) :
    Except String CifarLoggedTrainFlags :=
  Support.NpyLoggedTrainFlags.parse exeName args defaultLogPath defaultSteps
    (parseData := NpyDatasets.parseCifar)

end CifarLoggedTrainFlags

namespace CifarModelTrainFlags

/--
Parse the standard CIFAR plus optimizer/training flags.

Vision examples share the same CIFAR data boundary and optimizer controls; architecture files only
need to provide the model constructor and logging title. Any remaining arguments are preserved so
the caller can forward runtime flags such as `--device cpu`, `--device cuda`, or
`--execution typed-graph` to the public `Trainer.RunConfig` parser.
-/
def parse (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 1)
    (defaultLearningRate : Float := 1e-3) :
    Except String (CifarModelTrainFlags × List String) :=
  Support.NpyModelTrainFlags.parse exeName args defaultLogPath
    (defaultSteps := defaultSteps) (defaultLearningRate := defaultLearningRate)
    (parseData := NpyDatasets.parseCifar)

end CifarModelTrainFlags

namespace ForecastWindowDataFlags

/--
Parse the shared flags for household-power forecasting windows.

Forecasting commands share `--data-dir`, `--x`, `--y`, `--windows`, `--report-offset`, and `--seed`.
-/
def parse
    (exeName : String)
    (args : List String)
    (defaultWindows : Nat := 512)
    (defaultReportOffset : Nat := 96) :
    Except String (Support.ForecastWindowDataFlags × List String) := do
  let (dataDir, args) ← TorchLean.CLI.takePathFlag args "data-dir"
    (default := NN.Examples.Data.RealPaths.defaultDataDir)
  let (seed, args) ← CLI.takeSeed args (default := 0)
  let (windows, args) ← CLI.takePositiveNatFlag args exeName "windows" (default := defaultWindows)
  let (reportOffset, args) ← CLI.takeNatFlag args "report-offset" (default := defaultReportOffset)
  let (xPath, args) ←
    CLI.takePathFlag args "x" (default := (NN.Examples.Data.RealPaths.householdPowerX dataDir))
  let (yPath, args) ←
    CLI.takePathFlag args "y" (default := (NN.Examples.Data.RealPaths.householdPowerY dataDir))
  pure ({ xPath := xPath
          yPath := yPath
          windows := windows
          reportOffset := reportOffset
          seed := seed }, args)

end ForecastWindowDataFlags

/-- Parsed household-power forecasting data plus optimizer/training flags. -/
abbrev HouseholdPowerModelTrainFlags := Support.ForecastWindowModelTrainFlags

namespace HouseholdPowerModelTrainFlags

/--
Parse the standard household-power forecasting flags plus optimizer/training flags.

The forecasting command still owns the model and reporting logic, but the shared data/runtime flag
surface lives here with the other real-data code.
-/
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 100)
    (defaultLearningRate : Float := 0.01)
    (defaultWindows : Nat := 512)
    (defaultReportOffset : Nat := 96) :
    Except String (HouseholdPowerModelTrainFlags × List String) :=
  Support.ForecastWindowModelTrainFlags.parse exeName args defaultLogPath
    (defaultSteps := defaultSteps) (defaultLearningRate := defaultLearningRate)
    (parseData :=
      fun args => ForecastWindowDataFlags.parse exeName args defaultWindows defaultReportOffset)

end HouseholdPowerModelTrainFlags

/--
Build a batched CIFAR-10 loader from the image and label `.npy` files.

Both files are required up front, with a hint pointing at the converter script, so a missing dataset
fails with something actionable rather than a decode error halfway through the first epoch.
-/
def loadCifarLoader
    (exeName : String) (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Data.Loader Float batchSize CifarImage CifarTarget) := do
  Data.requirePairedFiles
    exeName
    "CIFAR-10 images" xPath
    "CIFAR-10 labels" yPath
    missingCifarHint
  let src := Data.LabeledSource.fromFiles xPath yPath rowCount
    [cifarChannels, cifarHeight, cifarWidth] cifarClasses
  let ds ←
    try
      src.load (α := Float)
    catch error =>
      let hint :=
        s!"{exeName}: failed to load CIFAR-10 arrays for --n-total {rowCount}.\n" ++
        s!"{error}\n" ++
        "If your local .npy files contain fewer rows, pass --n-total with that row count; " ++
        "to regenerate the prepared slice, run:\n" ++
        "  python3 scripts/datasets/download_example_data.py --cifar10"
      throw <| IO.userError hint
  -- Return the typed minibatch loader. Callers can take one batch for a fixed-sample check or pass
  -- the loader to the shared training code for shuffled multi-step training.
  let dl := Data.Loader.fromStream ds batchSize (shuffle := true) (seed := seed)
  pure dl

/-- Common training-log notes for CIFAR-backed classifier examples. -/
def cifarClassifierNotes (batchSize : Nat)
    (flags : CifarModelTrainFlags) (extra : Array String := #[]) : Array String :=
  Support.NpyDataFlags.trainLogNotes flags.data "cifar10" ++
  #[s!"lr={flags.training.learningRate}", s!"steps={flags.training.steps}", s!"batch={batchSize}"]
  ++ extra

/-- Load one shuffled epoch of full CIFAR-10 minibatches from prepared `.npy` arrays. -/
def loadCifarBatches
    (exeName : String) (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Array (Sample.Batch Float batchSize CifarImage CifarTarget)) := do
  let dl ← loadCifarLoader exeName batchSize rowCount seed xPath yPath
  let epoch ← CLI.orThrow exeName <|
    Data.Loader.nextNonemptyEpoch exeName dl
  pure epoch.batches

/-- Load the first full CIFAR-10 minibatch from the shared CIFAR loader. -/
def loadCifarBatch
    (exeName : String) (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Sample.Batch Float batchSize CifarImage CifarTarget) := do
  let dl ← loadCifarLoader exeName batchSize rowCount seed xPath yPath
  CLI.orThrow exeName <| Data.Loader.firstFullBatch exeName dl

/--
Load a user-prepared ImageNet-style `64x64` minibatch.

This loader reads prepared `.npy` arrays rather than JPEG files. The Python converter is the trust
boundary for filesystem image decoding and resizing; this Lean path checks the resulting tensor
shape and class range before handing the batch to examples.
-/
def loadImageNet64Loader
    (exeName : String) (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Data.Loader Float batchSize ImageNet64Image ImageNet64Target) := do
  Data.requirePairedFiles
    exeName
    "ImageNet64 images" xPath
    "ImageNet64 labels" yPath
    missingImageNet64Hint
  let src := Data.LabeledSource.fromFiles xPath yPath rowCount
    [imagenet64Channels, imagenet64Height, imagenet64Width] imagenet64Classes
  let ds ←
    try
      src.load (α := Float)
    catch error =>
      let hint :=
        s!"{exeName}: failed to load ImageNet64 arrays for --n-total {rowCount}.\n" ++
        s!"{error}\n" ++
        "If your local .npy files contain fewer rows, pass --n-total with that row count; " ++
        "to create ImageNet64 arrays, run the image-folder converter described in the error hint."
      throw <| IO.userError hint
  -- Same convention as CIFAR: this is the reusable loader for full-dataset loops.
  -- `loadImageNet64Batch` is for call sites that need a single fixed minibatch.
  let dl := Data.Loader.fromStream ds batchSize (shuffle := true) (seed := seed)
  pure dl

/-- Load one shuffled epoch of full ImageNet64-style minibatches from prepared `.npy` arrays. -/
def loadImageNet64Batches
    (exeName : String) (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Array (Sample.Batch Float batchSize ImageNet64Image ImageNet64Target)) := do
  let dl ← loadImageNet64Loader exeName batchSize rowCount seed xPath yPath
  let epoch ← CLI.orThrow exeName <|
    Data.Loader.nextNonemptyEpoch exeName dl
  pure epoch.batches

/-- Load the first full ImageNet64-style minibatch from the shared ImageNet64 loader. -/
def loadImageNet64Batch
    (exeName : String) (batchSize rowCount seed : Nat) (xPath yPath : System.FilePath) :
    IO (Sample.Batch Float batchSize ImageNet64Image ImageNet64Target) := do
  let dl ← loadImageNet64Loader exeName batchSize rowCount seed xPath yPath
  CLI.orThrow exeName <| Data.Loader.firstFullBatch exeName dl

/--
Load a CIFAR minibatch, flatten each channel-first image, and retain its first `config.dataWidth`
values. This is a prefix of the image buffer, not a spatial resize or learned feature extraction.
Widths larger than the full image are rejected.
-/
def loadCifarFeatureBatch (batchSize : Nat) (config : nn.models.Generative.Config)
    (exeName : String) (xPath yPath : System.FilePath) (rowCount seed : Nat) :
    IO (Tensor Float (config.dataShape [batchSize])) := do
  let batchSample ← loadCifarBatch exeName batchSize rowCount seed xPath yPath
  if hData : config.dataWidth ≤ CifarImage.size then
    pure (Tensor.flattenThenTake [batchSize] config.dataWidth hData batchSample.input)
  else
    throw <| IO.userError
      (s!"{exeName}: requested feature width {config.dataWidth} "
        ++ s!"exceeds CIFAR image size {CifarImage.size}")

/--
Public singleton dataset for compact vector generative examples over flattened CIFAR batches.

Autoencoder and supervised latent-bottleneck examples load one real CIFAR batch, flatten it to the
compact vector boundary, build one supervised sample, and hand that sample to the public trainer
API. The sample itself may be Float-specific; this dataset constructor casts it into the
runtime-selected arithmetic representation so the command still works across the ordinary public
runtime backends.
-/
def cifarFeatureDataset {τ : Shape}
    (batchSize : Nat)
    (config : nn.models.Generative.Config)
    (exeName : String)
    (sampleOfFeatures : Tensor Float (config.dataShape [batchSize]) →
      Sample.Supervised Float (config.dataShape [batchSize]) τ)
    (xPath yPath : System.FilePath) (rowCount seed : Nat) :
    Trainer.Dataset (config.dataShape [batchSize]) τ :=
  Data.defer do
    let x ← loadCifarFeatureBatch batchSize config exeName xPath yPath rowCount seed
    pure (sampleOfFeatures x)

/-- Shared text-corpus CLI/data boundary for local text-model examples. -/
abbrev TextCorpusFlags := text.CorpusPathOptions

namespace TextCorpusFlags

/-- Command-line options shared by local text-corpus examples. -/
def help : Array String :=
  #[ "  --data-file PATH      read a local UTF-8 text corpus"
  , "  --tiny-shakespeare    use data/real/text/tiny_shakespeare.txt"
  ]

/--
Parse the shared `--data-file` flag used by local text-model examples.

`--tiny-shakespeare` is accepted as an explicit shortcut for the default corpus path.
-/
def parse (args : List String) :
    Except String (TextCorpusFlags × List String) := do
  let args := args.filter (fun a => a != "--tiny-shakespeare")
  text.CorpusPathOptions.parse args NN.Examples.Data.RealPaths.tinyShakespeare

/-- Read the selected text corpus and fail with a shared preparation hint when it is missing. -/
def read (exeName : String) (flags : TextCorpusFlags) : IO String := do
  unless (← flags.path.pathExists) do
    throw <| IO.userError s!"{exeName}: missing text corpus: {flags.path}\n{missingTextHint}"
  let text ← IO.FS.readFile flags.path
  if text.isEmpty then
    throw <| IO.userError s!"{exeName}: empty text corpus: {flags.path}"
  pure text

end TextCorpusFlags

/-- Text corpus selection plus the number of causal training windows to expose. -/
structure TextWindowFlags where
  /-- Selected UTF-8 corpus. -/
  corpus : TextCorpusFlags
  /-- Number of approximately evenly spaced windows in the finite training dataset. -/
  windows : Nat
deriving Repr

namespace TextWindowFlags

/-- Command-line options shared by finite-window text examples. -/
def help (defaultWindows : Nat) : Array String :=
  TextCorpusFlags.help ++
    #[s!"  --windows N           corpus windows available to training (default: {defaultWindows})"]

/-- Parse a text corpus and a positive `--windows` count. -/
def parse (exeName : String) (defaultWindows : Nat) (args : List String) :
    Except String (TextWindowFlags × List String) := do
  let (corpus, args) ← TextCorpusFlags.parse args
  let (windows, args) ←
    CLI.takePositiveNatFlag args exeName "windows" (default := defaultWindows)
  pure ({ corpus, windows }, args)

/-- Read the selected text corpus. -/
def read (exeName : String) (flags : TextWindowFlags) : IO String :=
  TextCorpusFlags.read exeName flags.corpus

end TextWindowFlags

end NN.Examples.Models.RealData
