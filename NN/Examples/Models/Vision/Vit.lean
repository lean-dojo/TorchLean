/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe torchlean vit --cpu
  lake build -R -K cuda=true && lake exe torchlean vit --cuda

This is a real-data ViT-style CIFAR-10 minibatch run:
- patch embedding via Conv2d,
- reshape + transpose to tokens,
- one Transformer encoder block,
- flatten + linear head.
-/

module


public import NN
public import NN.API.Models.Vit
public import NN.Examples.Models.Common.RealData

/-!
# ViT-Style Real-Data Example

Runnable `torchlean vit` example. It trains a compact ViT-style image classifier on a
prepared CIFAR-10 minibatch: patch embedding by convolution, token reshape, transformer block, and
linear head.

The reusable model wiring lives in `NN.API.Models.Vit` (`nn.models.vit1`). This file is the
runnable wrapper (CIFAR loader construction + multi-epoch training loop).

```bash
python3 scripts/datasets/download_example_data.py --cifar10
lake build -R -K cuda=true && lake exe torchlean vit --cuda --n-total 200 --steps 1
```

Tip: the defaults are set for a quick sanity run. For a longer run, bump `--steps` and
`--n-total`, and enable CUDA fused kernels:

```bash
lake build -R -K cuda=true
lake exe torchlean vit --cuda --fast-kernels --n-total 2000 --steps 50
```
-/

@[expose] public section

open _root_.Spec
open _root_.Spec.Tensor
open NN.API

namespace NN.Examples.Models.Vision.Vit

def exeName : String := "torchlean vit"
def defaultLogJson : System.FilePath := "data/model_zoo/vit_trainlog.json"

def batch : Nat := 2
def inC : Nat := 3
def inH : Nat := RealData.cifarHeight
def inW : Nat := RealData.cifarWidth

def patchH : Nat := 4
def patchW : Nat := 4
def stride : Nat := 4
def padding : Nat := 0

-- CIFAR images are 32×32; 4×4 patches with stride 4 produce an 8×8 token grid. The attention
-- width is `dModel = numHeads * headDim`, matching the API-level Transformer block contract.
def dModel : Nat := 32
def outDim : Nat := RealData.cifarClasses

def numHeads : Nat := 4
def headDim : Nat := 8
def ffnHidden : Nat := 128

def cfg : nn.models.VitConfig :=
  { batch := batch
    inC := inC
    inH := inH
    inW := inW
    patchH := patchH
    patchW := patchW
    stride := stride
    padding := padding
    dModel := dModel
    outDim := outDim
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden }

abbrev σ : Shape :=
  nn.models.vitInShape cfg

abbrev τ : Shape :=
  nn.models.vitOutShape cfg

def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.vit1 cfg

def loadCifarLoader {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO (Data.BatchLoader α batch RealData.CifarImage RealData.CifarTarget) := do
  RealData.loadCifarLoader (α := α) exeName batch nRows seed xPath yPath

def main (args : List String) : IO UInt32 := do
  Common.runAnyOrFloat exeName args
    (preferFloat := fun args => args.contains "--cuda" || CLI.hasFlagValue args "log")
    (banner := fun opts =>
      s!"{exeName}: ViT CIFAR training (device={if opts.useGpu then "cuda" else "cpu"})")
    (anyK := fun {α} _ _ _ _ cast opts rest => do
        let (xPath, yPath, nRows, seed, rest) ← Common.orThrow exeName <|
          RealData.parseCifarFlags rest
        let (trainCfg, rest) ← Common.orThrow exeName <|
          Common.parseModelTrainFlags exeName rest defaultLogJson 1 1e-3
        Common.orThrow exeName <| CLI.requireNoArgs rest
        let loader ← loadCifarLoader (α := α) xPath yPath nRows seed
        nn.withModel mkModel fun model => do
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
          Common.writeCurveLogTo trainCfg.train.log "ViT CIFAR training" curve "loss"
            #[s!"x={xPath}", s!"y={yPath}", s!"nRows={nRows}",
              s!"device={if opts.useGpu then "cuda" else "cpu"}", s!"lr={trainCfg.lr}",
              s!"epochs={trainCfg.train.steps}", s!"batch={batch}"])

end NN.Examples.Models.Vision.Vit
