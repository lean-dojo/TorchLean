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
structure Schedule.Options where
  /-- Number of diffusion timesteps in the schedule. -/
  T : Nat
  /-- First beta value in the schedule. -/
  betaStart : Float
  /-- Final beta value in the schedule. -/
  betaEnd : Float
deriving Repr

namespace Schedule.Options

/-- Parse `--T`, `--beta-start`, and `--beta-end`. -/
def parse
    (args : List String)
    (defaultT : Nat := 100)
    (defaultBetaStart : Float := 1e-4)
    (defaultBetaEnd : Float := 0.12) :
    Except String (Schedule.Options × List String) := do
  let (T, args) ← CLI.takeNatFlag args "T" (default := defaultT)
  let (betaStart, args) ← CLI.takeFloatFlag args "beta-start" (default := defaultBetaStart)
  let (betaEnd, args) ← CLI.takeFloatFlag args "beta-end" (default := defaultBetaEnd)
  pure ({ T, betaStart, betaEnd }, args)

/-- Stable TrainLog metadata for a diffusion schedule. -/
def logNotes (config : Schedule.Options) : Array String :=
  #[s!"T={config.T}", s!"betaStart={config.betaStart}", s!"betaEnd={config.betaEnd}"]

end Schedule.Options

/-- Optional image artifacts emitted by generation and reconstruction examples. -/
structure Image.OutputOptions where
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

namespace Image.OutputOptions

/-- Parse image-artifact paths and the optional reconstruction timestep. -/
def parse (args : List String) : Except String (Image.OutputOptions × List String) := do
  let (reconstructStep?, args) ← CLI.takeNatFlag? args "reconstruct-step"
  let (samplePpm?, args) ← CLI.takePathFlag? args "sample-ppm"
  let (referencePpm?, args) ← CLI.takePathFlag? args "reference-ppm"
  let (noisyPpm?, args) ← CLI.takePathFlag? args "noisy-ppm"
  let (reconstructPpm?, args) ← CLI.takePathFlag? args "reconstruct-ppm"
  pure
    ({ reconstructStep?, samplePpm?, referencePpm?, noisyPpm?, reconstructPpm? }, args)

/-- Stable TrainLog metadata for requested image artifacts. -/
def logNotes (config : Image.OutputOptions) : Array String :=
  (match config.reconstructStep? with | none => #[] | some t => #[s!"reconstructStep={t}"]) ++
  (match config.samplePpm? with | none => #[] | some p => #[s!"samplePpm={p}"]) ++
  (match config.referencePpm? with | none => #[] | some p => #[s!"referencePpm={p}"]) ++
  (match config.noisyPpm? with | none => #[] | some p => #[s!"noisyPpm={p}"]) ++
  (match config.reconstructPpm? with | none => #[] | some p => #[s!"reconstructPpm={p}"])

end Image.OutputOptions

/-! ## Prepared-data and diagnostic artifacts -/

/-- Train/test tensor paths and row counts for paired-NPY scientific examples. -/
structure Npy.SplitOptions where
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

namespace Npy.SplitOptions

/-- Parse train/test paths and row counts for paired NPY tensors. -/
def parse
    (exeName : String)
    (args : List String)
    (defaultTrainX defaultTrainY defaultTestX defaultTestY : System.FilePath)
    (defaultTrainRows defaultTestRows : Nat)
    (defaultEvalRows : Nat := 16) :
    Except String (Npy.SplitOptions × List String) := do
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
def logNotes (config : Npy.SplitOptions) : Array String :=
  #[
    s!"train_rows={config.trainRows}",
    s!"test_rows={config.testRows}",
    s!"eval_rows={config.evalRows}",
    s!"train_x={config.trainX}",
    s!"train_y={config.trainY}",
    s!"test_x={config.testX}",
    s!"test_y={config.testY}"
  ]

end Npy.SplitOptions

/-- Prepared NPY feature/target paths and their row budget. -/
structure Npy.Options where
  /-- Prepared feature or image tensor path. -/
  xPath : System.FilePath
  /-- Prepared label or target tensor path. -/
  yPath : System.FilePath
  /-- Number of rows to read from the arrays. -/
  nRows : Nat
deriving Repr

namespace Npy.Options

/-- Parse `--n-total`, `--x`, and `--y` for an NPY-backed example. -/
def parse
    (args : List String)
    (defaultX defaultY : System.FilePath)
    (defaultRows : Nat) :
    Except String (Npy.Options × List String) := do
  let (nRows, args) ← CLI.takeNatFlag args "n-total" (default := defaultRows)
  let (xPath, args) ← CLI.takePathFlag args "x" (default := defaultX)
  let (yPath, args) ← CLI.takePathFlag args "y" (default := defaultY)
  pure ({ xPath, yPath, nRows }, args)

/-- Stable TrainLog metadata for an NPY-backed dataset branch. -/
def logNotes (config : Npy.Options) (datasetName : String) : Array String :=
  #[s!"data={datasetName}", s!"x={config.xPath}", s!"y={config.yPath}", s!"nRows={config.nRows}"]

end Npy.Options

/-- Prepared image datasets understood by the built-in image-model commands. -/
inductive Image.Dataset where
  /-- Prepared 64x64 RGB image tensors. -/
  | imagenet64
  /-- Prepared CIFAR-10 32x32 RGB tensors. -/
  | cifar10
deriving Repr, BEq

namespace Image.Dataset

/-- Parse `--dataset cifar10|imagenet64`; without the flag the dataset is `imagenet64`. -/
def parse (args : List String) : Except String (Image.Dataset × List String) := do
  let (dataset?, args) ← CLI.takeFlagValue? args "dataset"
  match dataset? with
  | none | some "imagenet64" => pure (.imagenet64, args)
  | some "cifar10" => pure (.cifar10, args)
  | some raw => throw s!"unknown --dataset {raw}; expected imagenet64 or cifar10"

end Image.Dataset

/-- Prepared forecasting-window paths and report controls. -/
structure Forecast.Options where
  /-- Prepared input-window tensor path. -/
  xPath : System.FilePath
  /-- Prepared target-window tensor path. -/
  yPath : System.FilePath
  /-- Number of forecasting windows to use. -/
  windows : Nat
  /-- Window index used for before/after forecast display. -/
  reportOffset : Nat
deriving Repr

namespace Forecast.Options

/-- Stable TrainLog metadata for forecasting-window datasets. -/
def logNotes (config : Forecast.Options) : Array String :=
  #[
    s!"windows={config.windows}",
    s!"report_index={config.reportOffset}",
    s!"x={config.xPath}",
    s!"y={config.yPath}"
  ]

end Forecast.Options

/-! ## Data plus training controls -/

namespace Training

/-- Training controls paired with the caller's data options. -/
structure Options (Data : Type) where
  /-- Optimizer, step, batching, and logging controls. -/
  training : TorchLean.CLI.Training.OptimizerOptions
  /-- Dataset paths and data-specific controls. -/
  data : Data
deriving Repr

/-- Parse data options before the shared optimizer and logging options.
Unconsumed arguments remain available to the runtime parser. -/
def Options.parse {Data : Type}
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (parseData : List String → Except String (Data × List String)) :
    Except String (Options Data × List String) := do
  let (data, rest) ← parseData args
  let (training, rest) ←
    TorchLean.CLI.Training.OptimizerOptions.parse exeName rest defaultLogPath
      (defaultSteps := defaultSteps) (defaultLr := defaultLr)
  pure ({ training, data }, rest)

end Training

namespace Csv

/-- Data options for a command that reads one supervised CSV. -/
structure Options where
  /-- CSV file containing model inputs and targets. -/
  path : System.FilePath
deriving Repr

/-- Parse the CSV path, leaving training and runtime options to their parsers. -/
def Options.parse (args : List String) (defaultPath : System.FilePath) :
    Except String (Options × List String) := do
  let (path, args) ← CLI.takePathFlag args "csv" (default := defaultPath)
  pure ({ path }, args)

end Csv

end NN.Examples.Support
