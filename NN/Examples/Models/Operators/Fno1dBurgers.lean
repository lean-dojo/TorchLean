/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Native TorchLean 1D FNO on the Burgers operator:

  python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32
  lake build -R -K cuda=true
  lake exe torchlean fno1d_burgers --cuda --fast-kernels --steps 700 --lr 0.003 \
    --plot-csv data/real/fno/predictions.csv --log data/real/fno/trainlog.json
  python3 NN/Examples/Data/plot_fno1d_burgers.py --csv data/real/fno/predictions.csv
-/

module

public import NN
public import NN.API.Models.Fno1d
public import NN.Runtime.Autograd.TorchLean.Fno1d
public import NN.Runtime.Training.Log

import NN.Runtime.Autograd.TorchLean.Fno1d
public import NN.Runtime.Autograd.Engine.Cuda.Fno1dRfftFused

/-!
# Native TorchLean FNO1D Burgers

This file is the operator-learning tutorial we want people to read after the basic CNN/MLP
examples. The Python helpers do the two jobs Lean should not own here: download/reshape the public
`burgers_data_R10.mat` file, then plot the prediction CSV. The model, loss, optimizer, and training
loop stay in TorchLean.

Why we use the real-split FNO path in this executable:
- `NN.FNO1D.model` is the mathematically clean complex-domain implementation.
- The eager CUDA backend stores float32 buffers, not complex buffers.
- On CUDA this run uses the fused `spectralConv1dRfft` autograd primitive, which represents
  Fourier weights by real/imaginary float32 buffers and executes the real FFT path through cuFFT.
- On CPU it falls back to the dense DFT implementation. That is slower, but it is the useful
  reference path when someone wants to inspect the math without CUDA in the way.

The training task follows the standard FNO Burgers setup: learn the operator
`u₀(x) ↦ u(x,T)` on a fixed periodic grid. We keep the default grid and row counts modest because
the first run should answer one question quickly: "is my TorchLean/CUDA path wired correctly?" Once
that works, raise `--steps`, export more rows, and bump the constants below.

References for the dataset/training convention:
- Li et al., “Fourier Neural Operator for Parametric Partial Differential Equations”, 2020/2021.
- MathWorks’ Burgers FNO example and the `burgers_data_R10.mat` public dataset.
- SciML FNO tutorials using fields `a` for initial conditions and `u` for final solutions.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Operators.Fno1dBurgers


def exeName : String := "torchlean fno1d_burgers"

def grid : Nat := 32
def width : Nat := 8
def modes : Nat := 8
def blocks : Nat := 1
def defaultTrainRows : Nat := 128
def defaultTestRows : Nat := 32

def modelCfg : nn.models.Fno1dConfig :=
  { grid := grid, width := width, modes := modes, blocks := blocks, seed := 0 }

abbrev Vec (n : Nat) : Shape := Shape.Vec n
abbrev Mat (m n : Nat) : Shape := Shape.Mat m n

abbrev σ : Shape := nn.models.fno1dInShape modelCfg
abbrev τ : Shape := nn.models.fno1dOutShape modelCfg

def defaultDir : System.FilePath := "data/real/fno"
def trainXPath : System.FilePath := defaultDir / "burgers_train_X.npy"
def trainYPath : System.FilePath := defaultDir / "burgers_train_y.npy"
def testXPath : System.FilePath := defaultDir / "burgers_test_X.npy"
def testYPath : System.FilePath := defaultDir / "burgers_test_y.npy"
def defaultPlotCsv : System.FilePath := defaultDir / "predictions.csv"
def defaultLogJson : System.FilePath := defaultDir / "trainlog.json"

def missingDataHint : String :=
  "Prepare the public Burgers FNO dataset with:\n" ++
  "  python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32\n" ++
  "The .mat file is large; use --mat PATH if you already downloaded burgers_data_R10.mat."

structure TrainConfig where
  steps : Nat
  seed : Nat
  logEvery : Nat
  trainRows : Nat
  testRows : Nat
  evalRows : Nat
  lr : Float
deriving Repr

structure DataFiles where
  trainX : System.FilePath
  trainY : System.FilePath
  testX : System.FilePath
  testY : System.FilePath

namespace DataFiles

def paths (files : DataFiles) : List System.FilePath :=
  [files.trainX, files.trainY, files.testX, files.testY]

end DataFiles

structure RunSpec where
  train : TrainConfig
  files : DataFiles
  plotCsv : System.FilePath
  logJson : System.FilePath

def parseFlags (args : List String) :
    Except String (RunSpec × List String) := do
  let (steps, args) ← CLI.takeStepsOrEpochs args 50
  let (seed, args) ← CLI.takeSeed args 0
  let (logEvery?, args) ← CLI.takeNatFlagOnce args "log-every"
  let (trainRows?, args) ← CLI.takeNatFlagOnce args "train-rows"
  let (testRows?, args) ← CLI.takeNatFlagOnce args "test-rows"
  let (evalRows?, args) ← CLI.takeNatFlagOnce args "eval-rows"
  let (lr?, args) ← CLI.takeFloatFlagOnce args "lr"
  let (x?, args) ← CLI.takePathFlagOnce args "x"
  let (y?, args) ← CLI.takePathFlagOnce args "y"
  let (testX?, args) ← CLI.takePathFlagOnce args "test-x"
  let (testY?, args) ← CLI.takePathFlagOnce args "test-y"
  let (plotCsv?, args) ← CLI.takePathFlagOnce args "plot-csv"
  let (logJson?, args) ← CLI.takePathFlagOnce args "log"
  let train : TrainConfig :=
    { steps := steps
      seed := seed
      logEvery := logEvery?.getD 10
      trainRows := trainRows?.getD defaultTrainRows
      testRows := testRows?.getD defaultTestRows
      evalRows := evalRows?.getD 16
      lr := lr?.getD 5e-3 }
  let files : DataFiles :=
    { trainX := x?.getD trainXPath
      trainY := y?.getD trainYPath
      testX := testX?.getD testXPath
      testY := testY?.getD testYPath }
  let spec : RunSpec :=
    { train := train
      files := files
      plotCsv := plotCsv?.getD defaultPlotCsv
      logJson := logJson?.getD defaultLogJson }
  pure (spec, args)

def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.fno1dReal modelCfg

def loadDataset {α : Type} [Runtime.Scalar α]
    (xPath yPath : System.FilePath) (n : Nat) :
    IO (Data.Dataset (sample.Supervised α σ τ)) := do
  let src := Data.SupervisedSource.ofPaths .npy xPath yPath n [grid] [grid]
  let dsE ← src.load (α := α)
  Common.orThrow exeName dsE

def sampleToInputTarget {α : Type} {σ τ : Shape} :
    sample.Supervised α σ τ → Tensor α σ × Tensor α τ
  | .cons x (.cons y .nil) => (x, y)

def writePredictionProbe (plotCsv : System.FilePath)
    (x target prediction : Tensor Float σ) : IO Unit := do
  Data.writePredictionCsv1D plotCsv x target prediction
  IO.println s!"  wrote prediction CSV: {plotCsv}"
  IO.println s!"  plot with: python3 NN/Examples/Data/plot_fno1d_burgers.py --csv {plotCsv}"

def metricHistory : _root_.Runtime.Training.MetricHistory :=
  _root_.Runtime.Training.MetricHistory.empty #[
    ("train_mse", "#4e79a7"),
    ("test_mse", "#f28e2b")
  ]

def trainLogNotes (spec : RunSpec) (spectralPath : String) (device : String) : Array String :=
  #[
    s!"model=fno1d_real",
    s!"spectral_path={spectralPath}",
    s!"device={device}",
    s!"grid={grid}",
    s!"width={width}",
    s!"modes={modes}",
    s!"blocks={blocks}",
    s!"steps={spec.train.steps}",
    s!"lr={spec.train.lr}",
    s!"train_rows={spec.train.trainRows}",
    s!"test_rows={spec.train.testRows}",
    s!"eval_rows={spec.train.evalRows}",
    s!"train_x={spec.files.trainX}",
    s!"train_y={spec.files.trainY}",
    s!"test_x={spec.files.testX}",
    s!"test_y={spec.files.testY}"
  ]

def writeMetricLog (path : System.FilePath) (hist : _root_.Runtime.Training.MetricHistory)
    (spec : RunSpec) (spectralPath device : String) : IO Unit := do
  let log := hist.toTrainLog
    (title := "FNO1D Burgers (TorchLean)")
    (notes := trainLogNotes spec spectralPath device)
  _root_.Runtime.Training.TrainLog.writeJson path log
  IO.println s!"  wrote TrainLog JSON: {path}"

structure LoadedData (α : Type) where
  train : Data.Dataset (sample.Supervised α σ τ)
  test : Data.Dataset (sample.Supervised α σ τ)

def loadData {α : Type} [Runtime.Scalar α] (files : DataFiles) (cfg : TrainConfig) :
    IO (LoadedData α) := do
  Data.requireFiles exeName files.paths missingDataHint
  let train ← loadDataset (α := α) files.trainX files.trainY cfg.trainRows
  let test ← loadDataset (α := α) files.testX files.testY cfg.testRows
  pure { train, test }

namespace FusedCuda

abbrev Param := _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.Param

def meanLoss (ps : Array Param) (samples : List (Tensor Float σ × Tensor Float τ)) :
    IO Float :=
  _root_.Runtime.Autograd.okOrThrow <|
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.meanLoss
      (grid := grid) (width := width) (modes := modes) (blocks := blocks) ps samples

def evalLosses (trainEval testEval : List (Tensor Float σ × Tensor Float τ))
    (ps : Array Param) : IO (Float × Float) := do
  let trainLoss ← meanLoss ps trainEval
  let testLoss ← meanLoss ps testEval
  pure (trainLoss, testLoss)

def recordEval (trainEval testEval : List (Tensor Float σ × Tensor Float τ))
    (hist : _root_.Runtime.Training.MetricHistory) (step : Nat) (ps : Array Param) (tag : String) :
    IO _root_.Runtime.Training.MetricHistory := do
  let (trainLoss, testLoss) ← evalLosses trainEval testEval ps
  IO.println s!"  {tag}: train_mse={trainLoss} test_mse={testLoss}"
  pure <| hist.push step #[trainLoss, testLoss]

def predict (ps : Array Param) (x : Tensor Float σ) : IO (Tensor Float τ) := do
  let fw ← _root_.Runtime.Autograd.okOrThrow <|
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.forward
      (grid := grid) (width := width) (modes := modes) (blocks := blocks) ps x none
  _root_.Runtime.Autograd.okOrThrow <|
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.predFromTape (grid := grid) fw.tape fw.predId

def trainStep (lr : Float)
    (ps : Array Param)
    (adamSt : _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.AdamState)
    (sample : Tensor Float σ × Tensor Float τ) :
    IO (Array Param × _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.AdamState) := do
  let (x, y) := sample
  let fw ← _root_.Runtime.Autograd.okOrThrow <|
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.forward
      (grid := grid) (width := width) (modes := modes) (blocks := blocks) ps x (some y)
  _root_.Runtime.Autograd.okOrThrow <|
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.updateParamsAdam ps fw lr adamSt

def run (spec : RunSpec) : IO Unit := do
  let cfg := spec.train
  let files := spec.files
  let data ← loadData (α := Float) files cfg
  let trainDs := data.train
  let testDs := data.test
  let trainSamples := Data.toList trainDs |>.map sampleToInputTarget
  let testSamples := Data.toList testDs |>.map sampleToInputTarget
  let reportTrainSamples := trainSamples.take cfg.evalRows
  let reportTestSamples := testSamples.take cfg.evalRows
  let trainCycle ← Common.orThrow exeName <|
    Data.cycleListOrError trainSamples "empty Burgers training dataset"
  let mut ps :=
    _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.initParams
      (grid := grid) (width := width) (modes := modes) (blocks := blocks) cfg.seed
  let mut adamSt : _root_.Runtime.Autograd.Cuda.Fno1dRfftFused.AdamState := {}
  let mut hist ← recordEval reportTrainSamples reportTestSamples metricHistory 0 ps "before"
  for step in [0:cfg.steps] do
    let sample := trainCycle (cfg.seed + step)
    let (ps', adamSt') ← trainStep cfg.lr ps adamSt sample
    ps := ps'
    adamSt := adamSt'
    if cfg.logEvery != 0 && (step + 1) % cfg.logEvery == 0 then
      hist ← recordEval reportTrainSamples reportTestSamples hist (step + 1) ps s!"step {step + 1}"
  hist ← recordEval reportTrainSamples reportTestSamples hist cfg.steps ps "after"
  match Data.toList testDs with
  | [] => pure ()
  | sample :: _ =>
      let (x, y) := sampleToInputTarget sample
      let yhat ← predict ps x
      writePredictionProbe spec.plotCsv x y yhat
  writeMetricLog spec.logJson hist spec "fused cuFFT RFFT autograd op" "cuda"

end FusedCuda

def runPortableDense
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (spec : RunSpec) :
    IO Unit := do
  let cfg := spec.train
  let files := spec.files
  -- Load the train/test arrays once, then keep the runtime loop purely over typed samples.
  let data ← loadData (α := Float) files cfg
  nn.withModel mkModel fun model => do
    let modDef := _root_.Runtime.Autograd.TorchLean.NN.Seq.mseScalarModuleDef model
    let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith (α := Float) modDef id opts
    let trainSamples := Data.toList data.train
    let testSamples := Data.toList data.test
    -- Evaluation uses fixed prefixes so before/after metrics are deterministic and cheap.
    let reportTrainSamples := trainSamples.take cfg.evalRows
    let reportTestSamples := testSamples.take cfg.evalRows
    -- Training cycles through the dataset by seed/step instead of materializing a repeated epoch
    -- list; this keeps long runs memory-stable.
    let trainCycle ← Common.orThrow exeName <|
      Data.cycleListOrError trainSamples "empty Burgers training dataset"
    let opt :=
      _root_.Runtime.Autograd.TorchLean.Optim.adam (α := Float) (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes model)
        cfg.lr 0.9 0.999 1e-8
    let mut st ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModule.initOptim m opt
    let evalLosses : IO (Float × Float) := do
      let trainLoss ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModule.meanLoss m reportTrainSamples
      let testLoss ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModule.meanLoss m reportTestSamples
      pure (trainLoss, testLoss)
    -- The metric history becomes the JSON training curve consumed by the website.
    let recordEval (hist : _root_.Runtime.Training.MetricHistory) (step : Nat) (tag : String) := do
      let (trainLoss, testLoss) ← evalLosses
      IO.println s!"  {tag}: train_mse={trainLoss} test_mse={testLoss}"
      pure <| hist.push step #[trainLoss, testLoss]
    let mut hist ← recordEval metricHistory 0 "before"
    for step in [0:cfg.steps] do
      st ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModule.stepWith m opt st
        (trainCycle (cfg.seed + step))
      if cfg.logEvery != 0 && (step + 1) % cfg.logEvery == 0 then
        hist ← recordEval hist (step + 1) s!"step {step + 1}"
    hist ← recordEval hist cfg.steps "after"

    -- Save one prediction probe so the example reports both scalar loss curves and a field-level
    -- Burgers trajectory comparison.
    let params ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModule.params m
    let compiled ← _root_.Runtime.Autograd.TorchLean.NN.Seq.compileOut model (α := Float)
    match testSamples with
    | [] => pure ()
    | sample :: _ =>
        let (x, y) := sampleToInputTarget sample
        let yhat := _root_.Runtime.Autograd.TorchLean.NN.Seq.predict1 model compiled params x
        writePredictionProbe spec.plotCsv x y yhat
    writeMetricLog spec.logJson hist spec "portable dense DFT ops" (if opts.useGpu then "cuda" else "cpu")

def logRunHeader (opts : _root_.Runtime.Autograd.Torch.Options) (spec : RunSpec) : IO Unit := do
  IO.println s!"{exeName}: native real-split FNO1D Burgers"
  if opts.useGpu && opts.fastKernels then
    IO.println "  fast-kernels=on"
  let backendName :=
    match opts.backend with
    | .eager => "eager"
    | .compiled => "compiled"
  IO.println s!"  device={if opts.useGpu then "cuda" else "cpu"} backend={backendName}"
  IO.println s!"  grid={grid} width={width} modes={modes} blocks={blocks}"
  IO.println s!"  rows train={spec.train.trainRows} test={spec.train.testRows} eval_prefix={spec.train.evalRows}"
  IO.println s!"  train={spec.files.trainX} / {spec.files.trainY}"
  IO.println s!"  test ={spec.files.testX} / {spec.files.testY}"
  IO.println s!"  log  ={spec.logJson}"

def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (spec, rest) ← Common.orThrow exeName <| parseFlags rest
      if spec.train.steps = 0 then
        throw <| IO.userError s!"{exeName}: --steps/--epochs must be > 0"
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let opts :=
        if opts.useGpu && !opts.fastKernels then
          { opts with fastKernels := true }
        else
          opts
      logRunHeader opts spec
      if opts.useGpu then
        IO.println "  spectral path=fused cuFFT RFFT autograd op"
        if opts.backend != .eager then
          IO.println "  note: fused CUDA path uses the eager CUDA tape (ignoring --backend compiled)"
        FusedCuda.run spec
      else
        IO.println "  spectral path=portable dense DFT ops"
        runPortableDense opts spec))
    { banner? := some (fun opts =>
        s!"{exeName}: native FNO1D Burgers (device={if opts.useGpu then "cuda" else "cpu"})")
      printOk := true }

end NN.Examples.Models.Operators.Fno1dBurgers
