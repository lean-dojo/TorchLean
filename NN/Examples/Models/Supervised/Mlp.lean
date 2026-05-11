/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --auto-mpg
  lake exe torchlean mlp --cpu
  lake build -R -K cuda=true && lake exe torchlean mlp --cuda
-/

module


public import NN
public import NN.API.Models.Mlp
public import NN.Examples.Data.RealPaths

/-!
# MLP Tabular Regression

We use UCI Auto MPG because it is small, public, predictable, and entirely numeric: seven car
features predict miles per gallon. That makes this a clean first supervised example without
inventing data in Lean.

Prepare the CSV once:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
lake exe torchlean mlp --cpu --steps 1
```

The downloader writes normalized columns `x1..x7,y`. If you want to try your own tabular regression
CSV, pass `--csv PATH` with the same columns.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Supervised.Mlp

def exeName : String := "torchlean mlp"
def defaultLogJson : System.FilePath := "data/model_zoo/mlp_trainlog.json"

def batch : Nat := 5
/-- Auto MPG has seven numeric predictors after dropping `car_name`. -/
def inDim : Nat := 7
def hidDim : Nat := 32
def outDim : Nat := 1

def cfg : nn.models.Mlp1Config :=
  { batch := batch, inDim := inDim, hidDim := hidDim, outDim := outDim }

abbrev σ : Shape :=
  nn.models.mlp1InShape cfg

abbrev τ : Shape :=
  nn.models.mlp1OutShape cfg

def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.mlp1Relu cfg

def loadCsvLoader {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (path : System.FilePath) (seed : Nat) : IO (Data.BatchLoader α batch (Shape.Vec inDim)
      (Shape.Vec outDim)) := do
  unless (← path.pathExists) do
    throw <| IO.userError
      s!"{exeName}: missing CSV dataset: {path}\nRun: python3 scripts/datasets/download_example_data.py --auto-mpg"
  let src : Data.TabularSupervisedSource :=
    { path := path, inDim := inDim, outDim := outDim, csvOptions := { skipHeader := true } }
  let dsE ← src.load (α := α)
  let ds ← Common.orThrow exeName dsE
  let dl := Data.batchLoader ds batch (shuffle := true) (seed := seed) (dropLast := true)
  pure dl

def main (args : List String) : IO UInt32 := do
  -- `runAnyOrFloat` keeps the same training script usable for scalar-polymorphic checks and for
  -- Float/CUDA runs. The Float branch records a loss curve because those runs are the ones used for
  -- plotted training artifacts.
  Common.runAnyOrFloat exeName args
    (preferFloat := fun args => args.contains "--cuda" || CLI.hasFlagValue args "log")
    (banner := fun opts =>
      s!"{exeName}: Auto MPG MLP regression (device={if opts.useGpu then "cuda" else "cpu"})")
    (anyK := fun {α} _ _ _ _ cast opts rest => do
      let (csv?, rest) ← Common.orThrow exeName <| CLI.takePathFlagOnce rest "csv"
      let csvPath := csv?.getD (_root_.NN.Examples.Data.RealPaths.autoMpgCsv)
      let (seed, rest) ← Common.orThrow exeName <| CLI.takeSeed rest 0
      let (trainCfg, rest) ← Common.orThrow exeName <|
        Common.parseModelTrainFlags exeName rest defaultLogJson 1 1e-3
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let loader ← loadCsvLoader (α := α) csvPath seed
      nn.withModel mkModel fun model => do
        -- Build the module, optimizer, and reporting hooks in the same order as a PyTorch loop:
        -- instantiate model, attach optimizer, fit the loader, then compare before/after loss.
        let modDef := nn.mseScalarModuleDef model
        let module ← TorchLean.Module.instantiateWithOptions (α := α) modDef cast opts
        let opt := Common.adamOptimizer (α := α) cast (nn.paramShapes model) trainCfg.lr
        let hooks : train.Callbacks α :=
          (train.onTrainStart (α := α) do
            train.Report.reportMeanLossModuleLoader module loader "train(before)")
          ++ train.onTrainEnd (α := α) (fun _ =>
            train.Report.reportMeanLossModuleLoader module loader "train(after)")
        let (report, _loader') ← train.fitModuleLoaderWith module opt trainCfg.train.steps loader hooks
        IO.println s!"  epochs={trainCfg.train.steps} loss0={report.before} loss1={report.after}"
      pure ())
    (floatK := fun opts rest => do
      let (csv?, rest) ← Common.orThrow exeName <| CLI.takePathFlagOnce rest "csv"
      let csvPath := csv?.getD (_root_.NN.Examples.Data.RealPaths.autoMpgCsv)
      let (seed, rest) ← Common.orThrow exeName <| CLI.takeSeed rest 0
      let (trainCfg, rest) ← Common.orThrow exeName <|
        Common.parseModelTrainFlags exeName rest defaultLogJson 1 1e-3
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let loader ← loadCsvLoader (α := Float) csvPath seed
      nn.withModel mkModel fun model => do
        -- The Float path mirrors the generic path and additionally records per-step loss for the
        -- website training curve.
        let modDef := nn.mseScalarModuleDef model
        let module ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
        let opt := Common.adamOptimizer (α := Float) id (nn.paramShapes model) trainCfg.lr
        let curveRef ← IO.mkRef ({} : _root_.Runtime.Training.Curve)
        let hooks : train.Callbacks Float :=
          (train.onTrainStart (α := Float) do
            train.Report.reportMeanLossModuleLoader module loader "train(before)")
          ++ train.onStep (α := Float) (fun ev =>
            curveRef.modify (fun c => c.push ev.step ev.loss))
          ++ train.onTrainEnd (α := Float) (fun _ =>
            train.Report.reportMeanLossModuleLoader module loader "train(after)")
        let (report, _loader') ← train.fitModuleLoaderWith module opt trainCfg.train.steps loader hooks
        let curve ← curveRef.get
        IO.println s!"  epochs={trainCfg.train.steps} loss0={report.before} loss1={report.after}"
        Common.writeCurveLogTo trainCfg.train.log "MLP tabular training" curve "loss"
          #[s!"dataset={csvPath}", s!"device={if opts.useGpu then "cuda" else "cpu"}",
            s!"lr={trainCfg.lr}", s!"epochs={trainCfg.train.steps}", s!"batch={batch}"])

end NN.Examples.Models.Supervised.Mlp
