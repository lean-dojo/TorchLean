/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Native TorchLean 1D FNO on the Burgers operator:

  python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32
  lake -R -K cuda=true build
  lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda --steps 700 --lr 0.003 \
    --plot-csv data/real/fno/predictions.csv --log data/real/fno/trainlog.json
  python3 NN/Examples/Data/plot_fno1d_burgers.py --csv data/real/fno/predictions.csv
-/

module

public import NN.API
public import NN.Examples.Support
public import NN.Examples.Models.Common.Train
public import NN.Runtime.Autograd.Engine.Cuda.Fno1dRfftFused

/-!
# Native TorchLean FNO1D Burgers

Read this after the basic CNN/MLP examples if you want the operator-learning path. The Python
scripts do the two jobs Lean should not own here: download and reshape the public
`burgers_data_R10.mat` file, then plot the prediction CSV. The model, loss, optimizer, and training
loop stay in TorchLean.

This executable uses real-split Fourier arithmetic because the Burgers data are real-valued. The
portable path evaluates the dense multidimensional DFT with separate real and imaginary tensors.
On CUDA, the command deliberately selects a specialized one-sided real-FFT model whose transforms
run through cuFFT. The two paths share the typed field-to-field boundary and training task, but
they do not share an identical spectral parameter layout.

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
open TorchLean

namespace NN.Examples.Models.Operators.Fno1dBurgers

/-- CLI subcommand name used in terminal banners and errors. -/
def exeName : String := "fno1d_burgers"

/-- Spatial grid resolution used by the prepared Burgers `.npy` slices. -/
def gridSize : Nat := 32

/-- Channel width inside the compact FNO block. -/
def width : Nat := 8

/--
Spectral mode budget.

The portable full-DFT path uses this as the width of each end band. The fused real-FFT path uses it
as the number of stored nonnegative-frequency bins.
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
abbrev input : Shape := modelConfig.input

/-- Model output shape: one predicted terminal solution on the same grid. -/
abbrev output : Shape := modelConfig.output

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

The record keeps optimizer/log settings, reproducibility, tensor paths, and output artifacts as
separate named concerns.
-/
structure Options where
  /-- Optimizer, step, batching, and logging controls. -/
  training : CLI.Training.OptimizerOptions
  /-- Seed used for initialization and training-row selection. -/
  seed : Nat
  /-- Prepared train/test tensors and evaluation row counts. -/
  data : Support.PairedNpyEvalFlags
  /-- Output path for the prediction diagnostic. -/
  artifacts : Support.CsvArtifactFlags
deriving Repr

/-- All required dataset files for this run. -/
def dataPaths (config : Options) : Array System.FilePath :=
  #[config.data.trainX, config.data.trainY, config.data.testX, config.data.testY]

namespace Options

/-- Parse the FNO Burgers command-line options. -/
def parse (args : List String) :
    Except String (Options × List String) := do
  let (seed, args) ← CLI.takeSeed args (default := 0)
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := 50) (defaultLearningRate := 5e-3)
  let (data, args) ← Support.PairedNpyEvalFlags.parse exeName args
    trainXPath trainYPath testXPath testYPath defaultTrainRows defaultTestRows
  let (artifacts, args) ← Support.CsvArtifactFlags.parse args defaultPlotCsv
  pure ({ training, seed, data, artifacts }, args)

/-- Effective CUDA-memory-watch cadence for this run. -/
def memoryCadence (config : Options) (runtime : Runtime.Config) : Nat :=
  Trainer.Memory.cadence runtime config.training.steps config.training.cudaMemorySampleEvery

/-- TrainLog note fields for the fused CUDA execution path. -/
def logNotes (config : Options) (spectralPath : String) (device : String) : Array String :=
  #[
    s!"model=fno",
    s!"spectral_path={spectralPath}",
    s!"device={device}",
    s!"grid={gridSize}",
    s!"width={width}",
    s!"modes={modes}",
    s!"blocks={blocks}",
    s!"steps={config.training.steps}",
    s!"lr={config.training.learningRate}"
  ] ++ Support.PairedNpyEvalFlags.trainLogNotes config.data

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
      toString (Tensor.item (Tensor.get x i)),
      toString (Tensor.item (Tensor.get target i)),
      toString (Tensor.item (Tensor.get prediction i))])
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

/-- Persist the train/test MSE history with model/data metadata attached. -/
def writeLog (destination : Training.LogDestination) (history : Training.MetricHistory)
    (config : Options) (spectralPath device : String) : IO Unit := do
  Training.MetricHistory.writeLog history destination "FNO1D Burgers (TorchLean)"
    (config.logNotes spectralPath device)

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
  /-- Held-out sample used for the prediction CSV emitted by both execution paths. -/
  probe : Tensor Float input × Tensor Float output
  next : Nat → Sample.Supervised Float input output

/--
Convert loaded Burgers datasets into the common runtime/evaluation view used by both execution
paths.

Both execution paths:
- evaluate on fixed deterministic prefixes,
- train by cycling through the finite dataset with `seed + step`, and
- emit the same train/test MSE metric history.
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

namespace FusedCuda

/-- Fused CUDA parameter packet for the real-FFT FNO kernel. -/
abbrev Param := Runtime.Autograd.Cuda.Fno1dRfftFused.Param

/-- Mean MSE over a finite evaluation prefix using the fused CUDA FNO implementation. -/
def meanLoss
    (parameters : Array Param)
    (samples : Data.SampleStream (Tensor Float input × Tensor Float output)) :
    IO Float := do
  let result ←
    Runtime.Autograd.Cuda.Fno1dRfftFused.meanLoss
      (grid := gridSize) (width := width)
      (modes := modes) (blocks := blocks) parameters samples
  Runtime.Autograd.okOrThrow result

/-- Train/test MSE pair for the current fused CUDA parameters. -/
def losses
    (trainEval testEval :
      Data.SampleStream (Tensor Float input × Tensor Float output))
    (parameters : Array Param) : IO (Float × Float) := do
  let trainLoss ← meanLoss parameters trainEval
  let testLoss ← meanLoss parameters testEval
  pure (trainLoss, testLoss)

/-- Append one fused-CUDA evaluation point to the metric history. -/
def record
    (trainEval testEval :
      Data.SampleStream (Tensor Float input × Tensor Float output))
    (history : Training.MetricHistory) (step : Nat) (parameters : Array Param) (tag : String) :
    IO Training.MetricHistory := do
  let (trainLoss, testLoss) ← losses trainEval testEval parameters
  Fno1dBurgers.record history step tag trainLoss testLoss

/-- Run the fused cuFFT/RFFT training path and emit its training and prediction artifacts. -/
def run (config : Options) : IO Unit := do
  let data ← load config
  let eval := ← prepare config data
  let mut parameters :=
    Runtime.Autograd.Cuda.Fno1dRfftFused.initParams
      (width := width)
      (modes := modes) (blocks := blocks) config.seed
  let mut adamState : Runtime.Autograd.Cuda.Fno1dRfftFused.AdamState := {}
  let mut history ←
    record eval.train eval.test metrics 0 parameters "before"
  let cudaOpts : Runtime.Autograd.Torch.Config :=
    { device := .cuda }
  let cudaMemorySampleEvery := config.memoryCadence cudaOpts
  let mut memWatch? ←
    Trainer.Memory.sample cudaOpts cudaMemorySampleEvery config.training.steps 0 none
  let progressEvery : Nat := Nat.max 1 (config.training.steps / 10)
  for step in [0:config.training.steps] do
    let current := eval.next (config.seed + step)
    let (updatedParameters, updatedAdamState) ←
      Runtime.Autograd.okOrThrow (← Runtime.Autograd.Cuda.Fno1dRfftFused.trainStep
        gridSize width modes blocks parameters current.input current.target
        config.training.learningRate adamState)
    parameters := updatedParameters
    adamState := updatedAdamState
    memWatch? ←
      Trainer.Memory.sample
        cudaOpts cudaMemorySampleEvery config.training.steps (step + 1) memWatch?
    if step + 1 < config.training.steps && Training.shouldReport progressEvery (step + 1) then
      history ←
        record eval.train eval.test history
          (step + 1) parameters s!"step {step + 1}"
  history ←
    record eval.train eval.test history
      config.training.steps parameters "after"
  let (input, target) := eval.probe
  let prediction ← Runtime.Autograd.okOrThrow (←
    Runtime.Autograd.Cuda.Fno1dRfftFused.predict gridSize width modes blocks parameters input)
  writePrediction config.artifacts.plotCsv input target prediction
  writeLog
    config.training.logDestination history config "fused cuFFT RFFT autograd op" "cuda"

end FusedCuda

/--
Train and evaluate using the portable dense DFT operations.

This is the path that runs anywhere. The fused cuFFT path above is faster but needs CUDA, and
keeping
both in the file means the two can be compared on the same data with the same seed.
-/
def runPortable
    (runtime : Runtime.Config)
    (config : Options) :
    IO Unit := do
  -- Load the train/test arrays once, then keep the runtime loop purely over typed samples.
  let data ← load config
  let eval := ← prepare config data
  let trainer :=
    Trainer.new model <|
      Trainer.RunConfig.forObjective
        (Trainer.RunConfig.fromRuntime runtime
          { optimizer := optim.adam { learningRate := config.training.learningRate } })
        .meanSquaredError
        (seed := config.seed)
  trainer.printSummary
  let histRef ← IO.mkRef metrics
  let meanPredMse (predict : Tensor Float input → IO (Tensor Float output))
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
    let trainLoss ← meanPredMse predict eval.train
    let testLoss ← meanPredMse predict eval.test
    let history ← histRef.get
    histRef.set (← record history step tag trainLoss testLoss)
  let progressEvery : Nat := Nat.max 1 (config.training.steps / 10)
  let trained ← trainer.trainStream runtime
    (fun step => eval.next (config.seed + step))
    eval.sample
    { steps := config.training.steps
      cudaMemorySampleEvery := config.training.cudaMemorySampleEvery
      logDestination := .disabled }
    (curveEvery := progressEvery)
    (onEval := evaluate)
  let (x, y) := eval.probe
  let yhat ← trained.predict x
  writePrediction config.artifacts.plotCsv x y yhat
  let history ← histRef.get
  writeLog
    config.training.logDestination history config "portable dense DFT ops"
      (runtime.deviceName)

/--
Print the run's configuration: device, execution mode, model geometry, row counts and file paths.

Worth the space, because an FNO run that silently loaded the wrong split looks like a training
problem
rather than a data problem.
-/
def printHeader (runtime : Runtime.Config) (config : Options) : IO Unit := do
  IO.println s!"{exeName}: native real-split FNO1D Burgers"
  let executionName :=
    match runtime.execution with
    | .eager => "eager"
    | .typedGraph => "typed-graph"
  IO.println s!"  {Support.deviceNote runtime} execution={executionName}"
  IO.println
    s!"  grid={gridSize} width={width} modes={modes} blocks={blocks}"
  IO.println (s!"  rows train={config.data.trainRows} test={config.data.testRows} "
    ++ s!"eval_prefix={config.data.evalRows}")
  IO.println s!"  cuda_mem_watch={config.memoryCadence runtime}"
  IO.println s!"  train={config.data.trainX} / {config.data.trainY}"
  IO.println s!"  test ={config.data.testX} / {config.data.testY}"
  IO.println s!"  log  ={config.training.logDestination}"

/--
Entry point; dispatches to the fused CUDA path when the device supports it and to `runPortable`
otherwise.
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Support.bannerWithDevice exeName "native FNO1D Burgers"
      usage? := some <| TrainCommand.optimizerUsage exeName #[
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
    (.native fun runtime rest => do
      let (config, rest) ← CLI.orThrow exeName <| Options.parse rest
      let config := { config with seed := runtime.seed }
      CLI.requireNoArgs exeName rest
      printHeader runtime config
      if runtime.usesCuda then
        if runtime.execution != .eager then
          throw <| IO.userError
            "fno1d_burgers: fused CUDA execution currently requires --execution eager"
        IO.println "  spectral path=fused cuFFT RFFT autograd op"
        FusedCuda.run config
      else
        IO.println "  spectral path=portable dense multidimensional DFT"
        runPortable runtime config)

end NN.Examples.Models.Operators.Fno1dBurgers
