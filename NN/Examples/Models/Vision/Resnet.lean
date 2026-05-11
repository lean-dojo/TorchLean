/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe torchlean resnet --cpu
  lake build -R -K cuda=true && lake exe torchlean resnet --cuda

This runs a medium ResNet-style model built from `API.nn.resnetBasicBlock` (Conv+BN+residual).
-/

module


public import NN
public import NN.API.Models.Resnet
public import NN.Examples.Models.Common.RealData

/-!
# ResNet Real-Data Example

Runnable `torchlean resnet` example. It trains a compact ResNet-style classifier built from
`API.nn.resnetBasicBlock` on a prepared CIFAR-10 minibatch.

The reusable model wiring lives in `NN.API.Models.Resnet` (`nn.models.resnet`). This file is the
runnable wrapper (CIFAR loader construction + multi-epoch training loop).

```bash
python3 scripts/datasets/download_example_data.py --cifar10
lake build -R -K cuda=true && lake exe torchlean resnet --cuda --n-total 200 --steps 1
```

Tip: the defaults are set for a quick sanity run. For a longer run:

```bash
lake build -R -K cuda=true
lake exe torchlean resnet --cuda --fast-kernels --n-total 5000 --steps 200
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Vision.Resnet

def exeName : String := "torchlean resnet"
def defaultLogJson : System.FilePath := "data/model_zoo/resnet_trainlog.json"

def batch : Nat := 2
def inC : Nat := 3
def inH : Nat := RealData.cifarHeight
def inW : Nat := RealData.cifarWidth

def stemC : Nat := 8
def numClasses : Nat := RealData.cifarClasses

def cfg : nn.models.ResnetConfig :=
  { batch := batch, inC := inC, inH := inH, inW := inW, stemC := stemC, numClasses := numClasses }

abbrev σ : Shape :=
  nn.models.resnetInShape cfg

abbrev τ : Shape :=
  nn.models.resnetOutShape cfg

def mkModel : nn.M (nn.Sequential σ τ) := do
  nn.models.resnet cfg

def loadCifarLoader {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO (Data.BatchLoader α batch RealData.CifarImage RealData.CifarTarget) := do
  RealData.loadCifarLoader (α := α) exeName batch nRows seed xPath yPath

def main (args : List String) : IO UInt32 := do
  -- The generic branch checks that the same model/training loop is scalar-polymorphic; the Float
  -- branch is the runtime path that writes curve artifacts for CUDA/CPU demos.
  Common.runAnyOrFloat exeName args
    (preferFloat := fun args => args.contains "--cuda" || CLI.hasFlagValue args "log")
    (banner := fun opts =>
      s!"{exeName}: ResNet CIFAR training (device={if opts.useGpu then "cuda" else "cpu"})")
    (anyK := fun {α} _ _ _ _ cast opts rest => do
      let (xPath, yPath, nRows, seed, rest) ← Common.orThrow exeName <|
        RealData.parseCifarFlags rest
      let (trainCfg, rest) ← Common.orThrow exeName <|
        Common.parseModelTrainFlags exeName rest defaultLogJson 1 1e-3
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let loader ← loadCifarLoader (α := α) xPath yPath nRows seed
      nn.withModel mkModel fun model => do
        -- Build a standard image-classification loop: cross-entropy module, Adam optimizer,
        -- before/after loader evaluation, and a multi-epoch fit call.
        let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
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
      let (xPath, yPath, nRows, seed, rest) ← Common.orThrow exeName <|
        RealData.parseCifarFlags rest
      let (trainCfg, rest) ← Common.orThrow exeName <|
        Common.parseModelTrainFlags exeName rest defaultLogJson 1 1e-3
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let loader ← loadCifarLoader (α := Float) xPath yPath nRows seed
      nn.withModel mkModel fun model => do
        -- The Float branch records the same training signal as a TrainLog curve for the website.
        let modDef := nn.crossEntropyOneHotScalarModuleDef model (reduction := .mean)
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
        Common.writeCurveLogTo trainCfg.train.log "ResNet CIFAR training" curve "loss"
          #[s!"data=cifar10", s!"x={xPath}", s!"y={yPath}", s!"nRows={nRows}",
            s!"device={if opts.useGpu then "cuda" else "cpu"}", s!"lr={trainCfg.lr}",
            s!"epochs={trainCfg.train.steps}", s!"batch={batch}"])

end NN.Examples.Models.Vision.Resnet
