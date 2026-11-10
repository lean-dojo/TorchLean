/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Native TorchLean 1D FNO on the Burgers operator:

  python3 TorchLeanExamples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 \
    --ntest 32
  scripts/lake.sh -Kcuda=true build
  scripts/lake.sh -Kcuda=true exe torchlean fno1d_burgers --device cuda --steps 700 --lr 0.003 \
    --plot-csv data/real/fno/predictions.csv --log data/real/fno/trainlog.json
  python3 TorchLeanExamples/Data/plot_fno1d_burgers.py --csv data/real/fno/predictions.csv
-/

module

public import TorchLeanExamples.Models.TrainCommand
public import TorchLean.NN.Models.FNO
public import TorchLean.NN.Trainer.Constructor
public import TorchLean.NN.Trainer.Train
public import TorchLeanExamples.Models.Training

/-!
# Native TorchLean FNO1D Burgers

A Fourier neural operator for the 1D Burgers equation. The model, loss, optimizer and training
loop are TorchLean code, and Python does two jobs only: it downloads and reshapes the public
`burgers_data_R10.mat` file, and it plots the prediction CSV.

The Burgers data are real-valued, so the Fourier arithmetic is real-split, with separate real and
imaginary tensors. CPU and GPU runs use the same full-spectrum model and spectral parameter
layout, and the runtime chooses how the operations execute (cuFFT on supported eager CUDA
interpreters, dense per-axis transforms otherwise).

The task is the standard FNO Burgers setup: learn the operator $u_0(x)\mapsto u(x,T)$ on a fixed
periodic grid. The default grid and row counts are small enough for a local run and exercise the
whole operator-learning path. Larger runs can raise `--steps`, export more rows and enlarge
`width`, `modes` or `blocks`.

References for the dataset/training convention:
- Li et al., “Fourier Neural Operator for Parametric Partial Differential Equations”, 2020/2021.
- MathWorks’ Burgers FNO example and the `burgers_data_R10.mat` public dataset.
- SciML FNO tutorials using fields `a` for initial conditions and `u` for final solutions.
-/

@[expose] public section

open TorchLean.Trainer.Objective TorchLean.Tensor

namespace TorchLean.Examples.FNOBurgers

/-- CLI subcommand name used in terminal banners and errors. -/
def exeName : String := "fno1d_burgers"

/-- Spatial grid resolution used by the prepared Burgers `.npy` slices. -/
def gridSize : Nat := 32

/-- Channel width inside the compact FNO block. -/
def width : Nat := 8

/--
Width of each retained frequency band.

The spectral mask keeps the Fourier coefficients below `modes` and those at least
`gridSize - modes`, 16 of the 32 here, on CPU and GPU alike.
-/
def modes : Nat := 8

/-- Number of spectral blocks used by the compact training run. -/
def blocks : Nat := 1

/-- Default number of training rows expected from the preparation script. -/
def defaultTrainRows : Nat := 128

/-- Default number of held-out rows expected from the preparation script. -/
def defaultTestRows : Nat := 32

/-- FNO configuration shared by the constructor and sample loaders. -/
abbrev modelConfig : nn.FNO.Config 1 :=
  { spatial := [gridSize]
    modes := [modes]
    width := width
    layerCount := blocks }

/-- Model input shape: one sampled initial condition on the fixed grid. -/
abbrev input : Shape := modelConfig.inputShape

/-- Model output shape: one predicted terminal solution on the same grid. -/
abbrev output : Shape := modelConfig.outputShape

/-- Directory where the preparation script writes Burgers tensors by default. -/
def defaultDir : System.FilePath := "data/real/fno"

/-- Default training input tensor path. -/
def trainXPath : System.FilePath := defaultDir / "burgers_train_X.npy"

/-- Default training target tensor path. -/
def trainYPath : System.FilePath := defaultDir / "burgers_train_y.npy"

/-- Default held-out input tensor path. -/
def testXPath : System.FilePath := defaultDir / "burgers_test_X.npy"

/-- Default held-out target tensor path. -/
def testYPath : System.FilePath := defaultDir / "burgers_test_y.npy"

/-- Default CSV path for the prediction-vs-target plot script. -/
def defaultPlotCSV : System.FilePath := defaultDir / "predictions.csv"

/-- Default JSON training-log path. -/
def defaultLogPath : System.FilePath := defaultDir / "trainlog.json"

/-- User-facing hint printed when the prepared Burgers tensors are missing. -/
def missingDataHint : String :=
  "Prepare the public Burgers FNO dataset with:\n" ++
  "  python3 TorchLeanExamples/Data/prepare_fno1d_burgers.py --download --grid 32 " ++
  "--ntrain 128 --ntest 32\n" ++
  "The .mat file is large; use --mat PATH if you already downloaded burgers_data_R10.mat."

/-- FNO Burgers command-line options: training flags, data paths and the prediction CSV path. -/
structure Options where
  /-- Optimizer, step, batching, and logging controls. -/
  training : CLI.Training.OptimizerOptions
  /-- Prepared train/test tensors and evaluation row counts. -/
  data : NPY.SplitOptions
  /-- Output path for the prediction diagnostic. -/
  plotCSV : System.FilePath
deriving Repr

/-- All required dataset files for this run. -/
def dataPaths (config : Options) : Array System.FilePath :=
  #[config.data.trainX, config.data.trainY, config.data.testX, config.data.testY]

namespace Options

/--
Parse the FNO Burgers command-line options, with defaults of 50 steps, learning rate `5e-3`, 128
training rows, 32 held-out rows and the default paths of this module.
-/
def parse (args : List String) :
    Except String (Options × List String) := do
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := 50) (defaultLr := 5e-3)
  let (data, args) ← NPY.SplitOptions.parse exeName args
    trainXPath trainYPath testXPath testYPath defaultTrainRows defaultTestRows
  let (plotCsv, args) ← CLI.takePathFlag args "plot-csv" (default := defaultPlotCSV)
  pure ({ training, data, plotCSV := plotCsv }, args)

/--
The experiment record of one run: the model and run settings as config entries, the train and test
curves, and the dataset files as notes.
-/
def experiment (config : Options) (spectralPath device : String)
    (history : Training.MetricHistory) : Training.ExperimentLog :=
  let entry (key value : String) : Training.ConfigEntry := { key, value }
  { run :=
      { name := "FNO1D Burgers (TorchLean)"
        tags := #["fno", "burgers"]
        config := #[
          entry "spectral_path" spectralPath,
          entry "device" device,
          entry "grid" (toString gridSize),
          entry "width" (toString width),
          entry "modes" (toString modes),
          entry "blocks" (toString blocks),
          entry "steps" (toString config.training.steps),
          entry "lr" (toString config.training.lr)] }
    history
    notes := NPY.SplitOptions.logNotes config.data }

end Options

/--
The Fourier neural operator this example trains, at the width, mode count and depth fixed by
`modelConfig`.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.fno modelConfig

/-- Load one fixed-grid Burgers split as supervised TorchLean samples. -/
def loadSplit
    (xPath yPath : System.FilePath) (n : Nat) :
    IO (Data.SampleStream (Sample.Supervised Float input output)) := do
  let source := Data.SupervisedSource.fromFiles xPath yPath n
    [gridSize] [gridSize]
  source.load (α := Float)

/--
Write one held-out prediction to CSV for `TorchLeanExamples/Data/plot_fno1d_burgers.py`, one row per
grid point with columns `i, x, input, target, prediction`.
-/
def writePrediction (plotCsv : System.FilePath)
    (x target prediction : Tensor Float input) : IO Unit := do
  let rows := (List.finRange gridSize).map (fun i =>
    let denom := Float.ofNat (Nat.max 1 gridSize)
    let xpos := Float.ofNat i.val / denom
    [toString i.val, toString xpos,
      toString (x.getScalar i), toString (target.getScalar i), toString (prediction.getScalar i)])
  if let some parent := plotCsv.parent then
    IO.FS.createDirAll parent
  let header := ["i", "x", "input", "target", "prediction"]
  let lines := String.intercalate "\n" ((String.intercalate "," header) :: rows.map
    (String.intercalate ",")) ++ "\n"
  IO.FS.writeFile plotCsv lines
  IO.println s!"  wrote prediction CSV: {plotCsv}"
  IO.println s!"  plot with: python3 TorchLeanExamples/Data/plot_fno1d_burgers.py --csv {plotCsv}"

/-- Two tracked curves, train and test MSE, with the colours the log viewer will use. -/
def metrics : Training.MetricHistory :=
  Training.MetricHistory.empty #[
    ("train_mse", "#4e79a7"),
    ("test_mse", "#f28e2b")
  ]

/-- Persist the train/test MSE history as an experiment record. -/
def writeLog (destination : Training.LogDestination) (history : Training.MetricHistory)
    (config : Options) (spectralPath device : String) : IO Unit := do
  Training.writeLog destination (config.experiment spectralPath device history).toTrainLog

/-- Loaded train/test splits before evaluation prefixes and cycling streams are derived. -/
structure Splits where
  /-- Training split as supervised samples. -/
  train : Data.SampleStream (Sample.Supervised Float input output)
  /-- Held-out split as supervised samples. -/
  test : Data.SampleStream (Sample.Supervised Float input output)

/-- Validate paths and load both Burgers splits. -/
def load (config : Options) :
    IO Splits := do
  Data.requireFiles exeName (dataPaths config) missingDataHint
  let train ← loadSplit config.data.trainX config.data.trainY config.data.trainRows
  let test ← loadSplit config.data.testX config.data.testY config.data.testRows
  pure { train, test }

/-- Deterministic evaluation prefixes and cycling stream derived from the loaded train/test sets. -/
structure Evaluation where
  /-- The first `evalRows` training rows, as input and target pairs. -/
  train : Data.SampleStream (Tensor Float input × Tensor Float output)
  /-- The first `evalRows` held-out rows, as input and target pairs. -/
  test : Data.SampleStream (Tensor Float input × Tensor Float output)
  /-- First training-prefix sample, on which the trainer measures its loss curve. -/
  sample : Sample.Supervised Float input output
  /-- Held-out sample used for the prediction CSV. -/
  probe : Tensor Float input × Tensor Float output
  /-- Training sample `i mod n` at step `i`, over the `n` training samples. -/
  next : Nat → Sample.Supervised Float input output

/--
Take the first `evalRows` training and held-out rows as evaluation prefixes and build a cycling
stream of training samples. An empty prefix or training split is an error.
-/
def prepare (config : Options) (data : Splits) : IO Evaluation := do
  let train := (data.train.splitAt config.data.evalRows).selected.map fun sample =>
    (sample.input, sample.target)
  let test := (data.test.splitAt config.data.evalRows).selected.map fun sample =>
    (sample.input, sample.target)
  let sample : Sample.Supervised Float input output ← if h : 0 < train.size then
    let (input, target) := train.get ⟨0, h⟩
    pure { input, target }
    else throw (IO.userError s!"{exeName}: empty Burgers training evaluation prefix")
  let probe ← if h : 0 < test.size then pure (test.get ⟨0, h⟩)
    else throw (IO.userError s!"{exeName}: empty Burgers held-out evaluation prefix")
  let next ← CLI.orThrow exeName <| data.train.cycleOrError "empty Burgers training dataset"
  pure { train, test, sample, probe, next }

/-- Push one train/test MSE point into the metric history and print the tagged report line. -/
def record
    (history : Training.MetricHistory)
    (step : Nat)
    (tag : String)
    (trainLoss testLoss : Float) : IO Training.MetricHistory := do
  IO.println s!"  {tag}: train_mse={trainLoss} test_mse={testLoss}"
  pure <| history.push step #[trainLoss, testLoss]

/--
Train the full-spectrum FNO with Adam on the MSE, on CPU or GPU, through the shared trainer.

Train and test MSE are evaluated about ten times per run. At the end the held-out probe goes to
the prediction CSV and the metric history to the experiment log.
-/
def run
    (runtime : Session.Config)
    (config : Options) :
    IO Unit := do
  -- Load the train/test arrays once, then keep the runtime loop purely over typed samples.
  let data ← load config
  let evaluation ← prepare config data
  let trainer :=
    Trainer.create model <|
      { runtime with
          optimizer := optim.adam { lr := config.training.lr }
          objective := mse }
  trainer.printSummary
  let historyRef ← IO.mkRef metrics
  let meanLoss (predict : Tensor Float input → IO (Tensor Float output))
      (samples : Data.SampleStream (Tensor Float input × Tensor Float output)) :
      IO Float := do
    let losses ← Tensor.generateFlatM [samples.size] fun index => do
      let (x, y) := samples.get ⟨index.val, by simpa using index.isLt⟩
      let yhat ← predict x
      pure (Tensor.meanSquaredError yhat y)
    pure (losses.sum / (Tensor.meanDenominator [samples.size] : Float))
  let evaluate
      (step : Nat) (tag : String)
      (predict : Tensor Float input → IO (Tensor Float output)) : IO Unit := do
    let trainLoss ← meanLoss predict evaluation.train
    let testLoss ← meanLoss predict evaluation.test
    let history ← historyRef.get
    historyRef.set (← record history step tag trainLoss testLoss)
  let progressEvery : Nat := Nat.max 1 (config.training.steps / 10)
  let trained ← trainer.trainStream runtime
    (fun step => evaluation.next (runtime.seed + step))
    evaluation.sample
    (config.training.trainOptions (enableLog := false))
    (curveEvery := progressEvery)
    (onEval := evaluate)
  let (x, y) := evaluation.probe
  let yhat ← trained.predict x
  writePrediction config.plotCSV x y yhat
  let history ← historyRef.get
  writeLog
    config.training.logDestination history config "shared full-spectrum Fourier model"
      (runtime.device.cliName)

/--
Print the run's configuration: device, model geometry, row counts and file paths.

An FNO run that silently loaded the wrong split can look like a training problem, and the printed
paths and row counts make that data error easy to spot.
-/
def printHeader (runtime : Session.Config) (config : Options) : IO Unit := do
  IO.println s!"{exeName}: native real-split FNO1D Burgers"
  IO.println s!"  {Training.deviceNote runtime}"
  IO.println
    s!"  grid={gridSize} width={width} modes={modes} blocks={blocks}"
  IO.println (s!"  rows train={config.data.trainRows} test={config.data.testRows} "
    ++ s!"eval_prefix={config.data.evalRows}")
  IO.println s!"  {Training.cudaMemoryNote runtime config.training.steps
    config.training.cudaMemorySampleEvery}"
  IO.println s!"  train={config.data.trainX} / {config.data.trainY}"
  IO.println s!"  test ={config.data.testX} / {config.data.testY}"
  IO.println s!"  log  ={config.training.logDestination}"

/-- Train and evaluate with the selected runtime and the shared FNO parameterization. -/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Training.banner exeName "native FNO1D Burgers"
      usage? := some <| CLI.Training.helpText exeName #[
        "  --x PATH           training input NPY file",
        "  --y PATH           training target NPY file",
        "  --test-x PATH      held-out input NPY file",
        "  --test-y PATH      held-out target NPY file",
        "  --train-rows N     training rows to load",
        "  --test-rows N      held-out rows to load",
        "  --eval-rows N      rows of each split used for the train and test reports",
        "  --plot-csv PATH    prediction/target CSV output"
      ]
      printSuccess := true })
    exeName args
    (fun runtime rest => do
      let (config, rest) ← CLI.orThrow exeName <| Options.parse rest
      CLI.requireNoArgs exeName rest
      printHeader runtime config
      run runtime config)

end TorchLean.Examples.FNOBurgers
