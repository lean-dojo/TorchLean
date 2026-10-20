/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Native TorchLean 1D FNO on the Burgers operator:

  python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32
  scripts/lake.sh -Kcuda=true build
  scripts/lake.sh -Kcuda=true exe torchlean fno1d_burgers --device cuda --steps 700 --lr 0.003 \
    --plot-csv data/real/fno/predictions.csv --log data/real/fno/trainlog.json
  python3 NN/Examples/Data/plot_fno1d_burgers.py --csv data/real/fno/predictions.csv
-/

module

public import NN.Examples.Models.Common.Train

/-!
# Native TorchLean FNO1D Burgers

Read this after the basic CNN/MLP examples if you want the operator-learning path. The Python
scripts do the two jobs Lean should not own here: download and reshape the public
`burgers_data_R10.mat` file, then plot the prediction CSV. The model, loss, optimizer, and training
loop stay in TorchLean.

This executable uses real-split Fourier arithmetic because the Burgers data are real-valued.
CPU and GPU execution use the same full-spectrum model and spectral parameter layout, with
separate real and imaginary tensors. The runtime selects how the operations are executed.

The training task follows the standard FNO Burgers setup: learn the operator
$u_0(x)\mapsto u(x,T)$ on a fixed periodic grid. The default grid and row counts are modest enough
for a local run while still exercising the real operator-learning path. Larger runs can raise
`--steps`, export more rows, and bump the constants below.

References for the dataset/training convention:
- Li et al., “Fourier Neural Operator for Parametric Partial Differential Equations”, 2020/2021.
- MathWorks’ Burgers FNO example and the `burgers_data_R10.mat` public dataset.
- SciML FNO tutorials using fields `a` for initial conditions and `u` for final solutions.
-/

@[expose] public section

open TorchLean TorchLean.Tensor

namespace NN.Examples.Models.Operators.Fno1dBurgers

/-- CLI subcommand name used in terminal banners and errors. -/
def exeName : String := "fno1d_burgers"

/-- Spatial grid resolution used by the prepared Burgers `.npy` slices. -/
def gridSize : Nat := 32

/-- Channel width inside the compact FNO block. -/
def width : Nat := 8

/--
Spectral mode budget.

The full-spectrum model uses this as the width of each end band on both CPU and GPU.
-/
def modes : Nat := 8

/-- Number of spectral blocks used by the compact training run. -/
def blocks : Nat := 1

/-- Default number of training rows expected from the preparation script. -/
def defaultTrainRows : Nat := 128

/-- Default number of held-out rows expected from the preparation script. -/
def defaultTestRows : Nat := 32

/-- FNO configuration shared by the constructor and sample loaders. -/
abbrev modelConfig : nn.models.FNO.Config 1 :=
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
def defaultPlotCsv : System.FilePath := defaultDir / "predictions.csv"

/-- Default JSON training-log path. -/
def defaultLogPath : System.FilePath := defaultDir / "trainlog.json"

/-- User-facing hint printed when the prepared Burgers tensors are missing. -/
def missingDataHint : String :=
  "Prepare the public Burgers FNO dataset with:\n" ++
  "  python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 " ++
  "--ntrain 128 --ntest 32\n" ++
  "The .mat file is large; use --mat PATH if you already downloaded burgers_data_R10.mat."

/--
FNO Burgers command-line options: training flags, data paths, and artifact paths.

The record keeps optimizer/log settings, tensor paths, and output artifacts as
separate named concerns.
-/
structure Options where
  /-- Optimizer, step, batching, and logging controls. -/
  training : CLI.Training.OptimizerOptions
  /-- Prepared train/test tensors and evaluation row counts. -/
  data : Support.Npy.SplitOptions
  /-- Output path for the prediction diagnostic. -/
  plotCsv : System.FilePath
deriving Repr

/-- All required dataset files for this run. -/
def dataPaths (config : Options) : Array System.FilePath :=
  #[config.data.trainX, config.data.trainY, config.data.testX, config.data.testY]

namespace Options

/-- Parse the FNO Burgers command-line options. -/
def parse (args : List String) :
    Except String (Options × List String) := do
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := 50) (defaultLr := 5e-3)
  let (data, args) ← Support.Npy.SplitOptions.parse exeName args
    trainXPath trainYPath testXPath testYPath defaultTrainRows defaultTestRows
  let (plotCsv, args) ← CLI.takePathFlag args "plot-csv" (default := defaultPlotCsv)
  pure ({ training, data, plotCsv }, args)

/-- Effective CUDA-memory-watch cadence for this run. -/
def memoryCadence (config : Options) (runtime : Runtime.Config) : Nat :=
  Trainer.Memory.cadence runtime config.training.steps config.training.cudaMemorySampleEvery

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
    notes := Support.Npy.SplitOptions.logNotes config.data }

end Options

/--
The Fourier neural operator this example trains, at the width, mode count and depth fixed by
`modelConfig`.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.fno modelConfig

/-- Load one fixed-grid Burgers split as supervised TorchLean samples. -/
def loadSplit
    (xPath yPath : System.FilePath) (n : Nat) :
    IO (Data.SampleStream (Sample.Supervised Float input output)) := do
  let source := Data.SupervisedSource.fromFiles xPath yPath n
    [gridSize] [gridSize]
  source.load (α := Float)

/-- Write one FNO prediction row to CSV for the companion plotting script. -/
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
  IO.println s!"  plot with: python3 NN/Examples/Data/plot_fno1d_burgers.py --csv {plotCsv}"

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
  train : Data.SampleStream (Tensor Float input × Tensor Float output)
  test : Data.SampleStream (Tensor Float input × Tensor Float output)
  /-- Training sample used to initialize the runtime's compiled evaluation path. -/
  sample : Sample.Supervised Float input output
  /-- Held-out sample used for the prediction CSV. -/
  probe : Tensor Float input × Tensor Float output
  next : Nat → Sample.Supervised Float input output

/--
Prepare deterministic evaluation prefixes and a cycling stream of training samples.
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

/-- Train the same full-spectrum model through the shared trainer on CPU or GPU. -/
def run
    (runtime : Runtime.Config)
    (config : Options) :
    IO Unit := do
  -- Load the train/test arrays once, then keep the runtime loop purely over typed samples.
  let data ← load config
  let evaluation ← prepare config data
  let trainer :=
    Trainer.new model <|
      { runtime with
          optimizer := optim.adam { lr := config.training.lr }
          objective := mse }
  trainer.printSummary
  let historyRef ← IO.mkRef metrics
  let meanLoss (predict : Tensor Float input → IO (Tensor Float output))
      (samples : Data.SampleStream (Tensor Float input × Tensor Float output)) :
      IO Float := do
    let losses ← Tensor.generateFlatM [samples.size] fun index => do
      let (x, y) := samples.get ⟨index.val, by simpa [Shape.size] using index.isLt⟩
      let yhat ← predict x
      pure (Tensor.meanSquaredError yhat y)
    pure losses.mean
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
  writePrediction config.plotCsv x y yhat
  let history ← historyRef.get
  writeLog
    config.training.logDestination history config "shared full-spectrum Fourier model"
      (runtime.deviceName)

/--
Print the run's configuration: device, model geometry, row counts and file paths.

An FNO run that silently loaded the wrong split can look like a training problem.
Printing the paths and row counts helps identify the data error.
-/
def printHeader (runtime : Runtime.Config) (config : Options) : IO Unit := do
  IO.println s!"{exeName}: native real-split FNO1D Burgers"
  IO.println s!"  {Support.deviceNote runtime}"
  IO.println
    s!"  grid={gridSize} width={width} modes={modes} blocks={blocks}"
  IO.println (s!"  rows train={config.data.trainRows} test={config.data.testRows} "
    ++ s!"eval_prefix={config.data.evalRows}")
  IO.println s!"  cuda_mem_watch={config.memoryCadence runtime}"
  IO.println s!"  train={config.data.trainX} / {config.data.trainY}"
  IO.println s!"  test ={config.data.testX} / {config.data.testY}"
  IO.println s!"  log  ={config.training.logDestination}"

/-- Train and evaluate with the selected runtime and the shared FNO parameterization. -/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Support.banner exeName "native FNO1D Burgers"
      usage? := some <| CLI.Training.Command.helpText exeName #[
        "  --x PATH           training input NPY file",
        "  --y PATH           training target NPY file",
        "  --test-x PATH      held-out input NPY file",
        "  --test-y PATH      held-out target NPY file",
        "  --train-rows N     training rows to load",
        "  --test-rows N      held-out rows to load",
        "  --eval-rows N      held-out rows used for reporting",
        "  --plot-csv PATH    prediction/target CSV output"
      ]
      printSuccess := true })
    exeName args
    (fun runtime rest => do
      let (config, rest) ← CLI.orThrow exeName <| Options.parse rest
      CLI.requireNoArgs exeName rest
      printHeader runtime config
      run runtime config)

end NN.Examples.Models.Operators.Fno1dBurgers
