/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI.Trainer

/-!
# Runnable Example Commands

Command-line records used by TorchLean's runnable model examples.

These declarations live under example support because they describe repository
commands and local artifact formats, not the tensor, trainer, or runtime API. Applications can use
the public trainer directly without importing diffusion schedules, PPM paths, or prepared-dataset
defaults.
-/

@[expose] public section

namespace NN.Examples.Support

open TorchLean

/-! ## Generative-model artifacts -/

/-- Diffusion schedule parameters exposed by the runnable diffusion example. -/
structure DiffusionScheduleFlags where
  /-- Number of diffusion timesteps in the schedule. -/
  T : Nat
  /-- First beta value in the schedule. -/
  betaStart : Float
  /-- Final beta value in the schedule. -/
  betaEnd : Float
deriving Repr

namespace DiffusionScheduleFlags

/-- Parse `--T`, `--beta-start`, and `--beta-end`. -/
def parse
    (args : List String)
    (defaultT : Nat := 100)
    (defaultBetaStart : Float := 1e-4)
    (defaultBetaEnd : Float := 0.12) :
    Except String (DiffusionScheduleFlags × List String) := do
  let (T, args) ← CLI.takeNatFlag args "T" (default := defaultT)
  let (betaStart, args) ← CLI.takeFloatFlag args "beta-start" (default := defaultBetaStart)
  let (betaEnd, args) ← CLI.takeFloatFlag args "beta-end" (default := defaultBetaEnd)
  pure ({ T, betaStart, betaEnd }, args)

/-- Stable TrainLog metadata for a diffusion schedule. -/
def trainLogNotes (config : DiffusionScheduleFlags) : Array String :=
  #[s!"T={config.T}", s!"betaStart={config.betaStart}", s!"betaEnd={config.betaEnd}"]

end DiffusionScheduleFlags

/-- Optional image artifacts emitted by generation and reconstruction examples. -/
structure ImageArtifactFlags where
  /-- Optional timestep used for reconstruction-from-noise artifacts. -/
  reconstructStep? : Option Nat
  /-- Optional path for an unconditional sample image. -/
  samplePpm? : Option System.FilePath
  /-- Optional path for the clean reference image. -/
  referencePpm? : Option System.FilePath
  /-- Optional path for the noised or intermediate image. -/
  noisyPpm? : Option System.FilePath
  /-- Optional path for the reconstructed image. -/
  reconstructPpm? : Option System.FilePath
deriving Repr

namespace ImageArtifactFlags

/-- Parse image-artifact paths and the optional reconstruction timestep. -/
def parse (args : List String) : Except String (ImageArtifactFlags × List String) := do
  let (reconstructStep?, args) ← CLI.takeNatFlag? args "reconstruct-step"
  let (samplePpm?, args) ← CLI.takePathFlag? args "sample-ppm"
  let (referencePpm?, args) ← CLI.takePathFlag? args "reference-ppm"
  let (noisyPpm?, args) ← CLI.takePathFlag? args "noisy-ppm"
  let (reconstructPpm?, args) ← CLI.takePathFlag? args "reconstruct-ppm"
  pure
    ({ reconstructStep?, samplePpm?, referencePpm?, noisyPpm?, reconstructPpm? }, args)

/-- Stable TrainLog metadata for requested image artifacts. -/
def trainLogNotes (config : ImageArtifactFlags) : Array String :=
  (match config.reconstructStep? with | none => #[] | some t => #[s!"reconstructStep={t}"]) ++
  (match config.samplePpm? with | none => #[] | some p => #[s!"samplePpm={p}"]) ++
  (match config.referencePpm? with | none => #[] | some p => #[s!"referencePpm={p}"]) ++
  (match config.noisyPpm? with | none => #[] | some p => #[s!"noisyPpm={p}"]) ++
  (match config.reconstructPpm? with | none => #[] | some p => #[s!"reconstructPpm={p}"])

end ImageArtifactFlags

/-! ## Prepared-data and diagnostic artifacts -/

/-- Train/test tensor paths and row counts for paired-NPY scientific examples. -/
structure PairedNpyEvalFlags where
  /-- Number of rows loaded from the prepared training tensors. -/
  trainRows : Nat
  /-- Number of rows loaded from the prepared held-out tensors. -/
  testRows : Nat
  /-- Prefix length used for deterministic train/test loss reports. -/
  evalRows : Nat
  /-- Training input `.npy` path. -/
  trainX : System.FilePath
  /-- Training target `.npy` path. -/
  trainY : System.FilePath
  /-- Held-out input `.npy` path. -/
  testX : System.FilePath
  /-- Held-out target `.npy` path. -/
  testY : System.FilePath
deriving Repr

namespace PairedNpyEvalFlags

/-- Parse train/test paths and row counts for paired NPY tensors. -/
def parse
    (exeName : String)
    (args : List String)
    (defaultTrainX defaultTrainY defaultTestX defaultTestY : System.FilePath)
    (defaultTrainRows defaultTestRows : Nat)
    (defaultEvalRows : Nat := 16) :
    Except String (PairedNpyEvalFlags × List String) := do
  let (trainRows, args) ←
    CLI.takePositiveNatFlag args exeName "train-rows" (default := defaultTrainRows)
  let (testRows, args) ←
    CLI.takePositiveNatFlag args exeName "test-rows" (default := defaultTestRows)
  let (evalRows, args) ←
    CLI.takePositiveNatFlag args exeName "eval-rows" (default := defaultEvalRows)
  let (trainX, args) ← CLI.takePathFlag args "x" (default := defaultTrainX)
  let (trainY, args) ← CLI.takePathFlag args "y" (default := defaultTrainY)
  let (testX, args) ← CLI.takePathFlag args "test-x" (default := defaultTestX)
  let (testY, args) ← CLI.takePathFlag args "test-y" (default := defaultTestY)
  pure ({ trainRows, testRows, evalRows, trainX, trainY, testX, testY }, args)

/-- Stable TrainLog metadata for paired train/test NPY tensors. -/
def trainLogNotes (config : PairedNpyEvalFlags) : Array String :=
  #[
    s!"train_rows={config.trainRows}",
    s!"test_rows={config.testRows}",
    s!"eval_rows={config.evalRows}",
    s!"train_x={config.trainX}",
    s!"train_y={config.trainY}",
    s!"test_x={config.testX}",
    s!"test_y={config.testY}"
  ]

end PairedNpyEvalFlags

/-- Optional CSV artifact path for commands that emit one tabular diagnostic. -/
structure CsvArtifactFlags where
  /-- CSV path for the diagnostic artifact. -/
  plotCsv : System.FilePath
deriving Repr

namespace CsvArtifactFlags

/-- Parse the optional `--plot-csv` artifact path. -/
def parse
    (args : List String)
    (defaultPlotCsv : System.FilePath) :
    Except String (CsvArtifactFlags × List String) := do
  let (plotCsv, args) ← CLI.takePathFlag args "plot-csv" (default := defaultPlotCsv)
  pure ({ plotCsv }, args)

end CsvArtifactFlags

/-- Prepared NPY feature/target paths and their row budget. -/
structure NpyDataFlags where
  /-- Prepared feature or image tensor path. -/
  xPath : System.FilePath
  /-- Prepared label or target tensor path. -/
  yPath : System.FilePath
  /-- Number of rows to read from the arrays. -/
  nRows : Nat
  /-- Data-loader seed. -/
  seed : Nat
deriving Repr

namespace NpyDataFlags

/-- Parse `--seed`, `--n-total`, `--x`, and `--y` for an NPY-backed example. -/
def parse
    (args : List String)
    (defaultX defaultY : System.FilePath)
    (defaultRows : Nat) :
    Except String (NpyDataFlags × List String) := do
  let (seed, args) ← CLI.takeSeed args (default := 0)
  let (nRows, args) ← CLI.takeNatFlag args "n-total" (default := defaultRows)
  let (xPath, args) ← CLI.takePathFlag args "x" (default := defaultX)
  let (yPath, args) ← CLI.takePathFlag args "y" (default := defaultY)
  pure ({ xPath, yPath, nRows, seed }, args)

/-- Stable TrainLog metadata for an NPY-backed dataset branch. -/
def trainLogNotes (config : NpyDataFlags) (datasetName : String) : Array String :=
  #[s!"data={datasetName}", s!"x={config.xPath}", s!"y={config.yPath}", s!"nRows={config.nRows}"]

end NpyDataFlags

/-- Prepared image datasets understood by the built-in image-model commands. -/
inductive ImageDatasetChoice where
  /-- Prepared 64x64 RGB image tensors. -/
  | imagenet64
  /-- Prepared CIFAR-10 32x32 RGB tensors. -/
  | cifar10
deriving Repr, BEq

namespace ImageDatasetChoice

/-- Parse the dataset selector and reject ambiguous combinations. -/
def parse (args : List String) : Except String (ImageDatasetChoice × List String) := do
  let (dataset?, args) ← CLI.takeFlagValue? args "dataset"
  let (cifarFlag, args) ← CLI.takeBoolFlag args "cifar10"
  let (imagenetFlag, args) ← CLI.takeBoolFlag args "imagenet64"
  match dataset?, cifarFlag, imagenetFlag with
  | some _, true, _ | some _, _, true | none, true, true =>
      throw "choose only one dataset selector: --dataset, --cifar10, or --imagenet64"
  | some raw, false, false =>
      match raw with
      | "imagenet64" | "imagenet" | "imagenette64" => pure (.imagenet64, args)
      | "cifar10" | "cifar" => pure (.cifar10, args)
      | _ => throw s!"unknown --dataset {raw}; expected imagenet64 or cifar10"
  | none, true, false => pure (.cifar10, args)
  | none, false, true => pure (.imagenet64, args)
  | none, false, false => pure (.imagenet64, args)

end ImageDatasetChoice

/-- Prepared forecasting-window paths and report controls. -/
structure ForecastWindowDataFlags where
  /-- Prepared input-window tensor path. -/
  xPath : System.FilePath
  /-- Prepared target-window tensor path. -/
  yPath : System.FilePath
  /-- Number of forecasting windows to use. -/
  windows : Nat
  /-- Window index used for before/after forecast display. -/
  reportOffset : Nat
  /-- Data-loader seed. -/
  seed : Nat
deriving Repr

namespace ForecastWindowDataFlags

/-- Stable TrainLog metadata for forecasting-window datasets. -/
def trainLogNotes (config : ForecastWindowDataFlags) : Array String :=
  #[
    s!"windows={config.windows}",
    s!"report_index={config.reportOffset}",
    s!"x={config.xPath}",
    s!"y={config.yPath}"
  ]

end ForecastWindowDataFlags

/-! ## Data plus training controls -/

/-- Fixed-step training flags paired with an NPY dataset. -/
structure NpyLoggedTrainFlags where
  /-- Step, batching, and logging controls. -/
  training : TorchLean.CLI.Training.RunOptions
  /-- NPY paths, row budget, and data seed. -/
  data : NpyDataFlags
deriving Repr

namespace NpyLoggedTrainFlags

/-- Parse NPY data and logged-training flags, then reject unused command arguments. -/
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat)
    (parseData : List String → Except String (NpyDataFlags × List String)) :
    Except String NpyLoggedTrainFlags := do
  let (data, rest) ← parseData args
  let (train, rest) ←
    TorchLean.CLI.Training.RunOptions.parse exeName rest defaultLogPath
      (defaultSteps := defaultSteps)
  CLI.checkNoArgs rest
  pure { training := train, data := data }

end NpyLoggedTrainFlags

/-- Optimizer/training flags paired with an NPY dataset. -/
structure NpyModelTrainFlags where
  /-- Optimizer, step, batching, and logging controls. -/
  training : TorchLean.CLI.Training.OptimizerOptions
  /-- NPY paths, row budget, and data seed. -/
  data : NpyDataFlags
deriving Repr

namespace NpyModelTrainFlags

/-- Parse NPY data and the standard model-training flags. -/
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLearningRate : Float := 1e-3)
    (parseData : List String → Except String (NpyDataFlags × List String)) :
    Except String (NpyModelTrainFlags × List String) := do
  let (data, rest) ← parseData args
  let (train, rest) ←
    TorchLean.CLI.Training.OptimizerOptions.parse exeName rest defaultLogPath
      (defaultSteps := defaultSteps) (defaultLearningRate := defaultLearningRate)
  pure ({ training := train, data := data }, rest)

end NpyModelTrainFlags

/-- Optimizer/training flags paired with a forecasting-window dataset. -/
structure ForecastWindowModelTrainFlags
    where
  /-- Optimizer, step, batching, and logging controls. -/
  training : TorchLean.CLI.Training.OptimizerOptions
  /-- Prepared forecasting-window paths and reporting controls. -/
  data : ForecastWindowDataFlags
deriving Repr

namespace ForecastWindowModelTrainFlags

/-- Parse forecasting data and the standard model-training flags. -/
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 100)
    (defaultLearningRate : Float := 0.01)
    (parseData : List String → Except String (ForecastWindowDataFlags × List String)) :
    Except String (ForecastWindowModelTrainFlags × List String) := do
  let (data, rest) ← parseData args
  let (train, rest) ←
    TorchLean.CLI.Training.OptimizerOptions.parse exeName rest defaultLogPath
      (defaultSteps := defaultSteps) (defaultLearningRate := defaultLearningRate)
  pure ({ training := train, data := data }, rest)

end ForecastWindowModelTrainFlags

/-- Optimizer/training flags for a model command that reads one supervised CSV. -/
structure CsvTrainFlags where
  /-- Optimizer, step, batching, and logging controls. -/
  training : TorchLean.CLI.Training.OptimizerOptions
  /-- CSV file containing model inputs and targets. -/
  csvPath : System.FilePath
  /-- Seed used for model initialization and data shuffling. -/
  seed : Nat
deriving Repr

/-- Parse a CSV path, seed, and the standard model-training flags. -/
def parseCsvTrainFlags
    (exeName : String)
    (args : List String)
    (defaultCsv defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLearningRate : Float := 1e-3)
    (allowZeroSteps : Bool := false) :
    Except String (CsvTrainFlags × List String) := do
  let (csv?, args) ← CLI.takePathFlag? args "csv"
  let csvPath := csv?.getD defaultCsv
  let (seed, args) ← CLI.takeSeed args (default := 0)
  let (train, args) ←
    TorchLean.CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := defaultSteps) (defaultLearningRate := defaultLearningRate)
      (allowZeroSteps := allowZeroSteps)
  pure ({ training := train, csvPath, seed }, args)

end NN.Examples.Support
