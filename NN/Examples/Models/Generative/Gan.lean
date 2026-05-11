/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe -K cuda=true torchlean gan --cuda --steps 10
-/

module

public import NN
public import NN.API.Models.Generative
public import NN.Examples.Models.Common.RealData
public import NN.Spec.Models.Gan
public import NN.MLTheory.Generative.Latent.GAN

/-!
# GAN CIFAR Example

Small LSGAN-style executable path.

This trains:
- a generator `z -> image` toward the current CIFAR minibatch as a stable warm-up objective;
- a discriminator on real CIFAR images (`1`) and deterministic noise images (`0`).

The formal LSGAN objective decomposition lives in `NN.Spec.Models.Gan` and
`NN.MLTheory.Generative.Latent.GAN`. A full alternating adversarial trainer can reuse the same
generator/discriminator constructors and data path.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Generative.Gan

def exeName : String := "torchlean gan"
def defaultLogJson : System.FilePath := "data/model_zoo/gan_trainlog.json"
def cfg : nn.models.VectorGenerativeConfig := nn.models.compactImageConfig

abbrev Z : Shape := nn.models.vectorLatentShape cfg
abbrev X : Shape := nn.models.vectorDataShape cfg
abbrev S : Shape := NN.Tensor.Shape.Mat cfg.batch 1

def mkGenerator : nn.M (nn.Sequential Z X) :=
  nn.models.vectorGanGenerator cfg

def mkDiscriminator : nn.M (nn.Sequential X S) :=
  nn.models.vectorGanDiscriminator cfg

def trainCurve (opts : TorchLean.Options) (xPath yPath : System.FilePath)
    (nRows seed steps : Nat) : IO _root_.Runtime.Training.Curve := do
  nn.withModel mkGenerator fun gen => do
  nn.withModel mkDiscriminator fun disc => do
    let genDef := nn.mseScalarModuleDef gen
    let discDef := nn.mseScalarModuleDef disc
    let genM ← TorchLean.Module.instantiateWithOptions (α := Float) genDef id opts
    let discM ← TorchLean.Module.instantiateWithOptions (α := Float) discDef id opts
    let realX ← RealData.loadCifarVectorBatch cfg (by decide) exeName xPath yPath nRows seed
    let z := nn.models.latentNoise cfg seed
    let noiseX := nn.models.dataNoise cfg (seed + 17)
    let genSample : API.sample.Supervised Float Z X := API.sample.mk z realX
    let discReal : API.sample.Supervised Float X S := API.sample.mk realX (nn.models.onesScore cfg)
    let discFake : API.sample.Supervised Float X S := API.sample.mk noiseX (nn.models.zerosScore cfg)
    let genOpt :=
      TorchLean.Optim.adam (α := Float) (paramShapes := nn.paramShapes gen)
        (lr := 1e-3) (beta1 := 0.9) (beta2 := 0.999) (epsilon := 1e-8)
    let discOpt :=
      TorchLean.Optim.adam (α := Float) (paramShapes := nn.paramShapes disc)
        (lr := 1e-3) (beta1 := 0.9) (beta2 := 0.999) (epsilon := 1e-8)
    let genH ← TorchLean.Optim.handle (α := Float) genM genOpt
    let discH ← TorchLean.Optim.handle (α := Float) discM discOpt
    let mut curve : _root_.Runtime.Training.Curve := {}
    let g0 ← TorchLean.Module.forward (α := Float) genM genSample
    let d0r ← TorchLean.Module.forward (α := Float) discM discReal
    let d0f ← TorchLean.Module.forward (α := Float) discM discFake
    let mut last := Tensor.toScalar g0 + Tensor.toScalar d0r + Tensor.toScalar d0f
    curve := curve.push 0 last
    for step in [0:steps] do
      genH.step genSample
      discH.step discReal
      discH.step discFake
      let g ← TorchLean.Module.forward (α := Float) genM genSample
      let dr ← TorchLean.Module.forward (α := Float) discM discReal
      let df ← TorchLean.Module.forward (α := Float) discM discFake
      last := Tensor.toScalar g + Tensor.toScalar dr + Tensor.toScalar df
      curve := curve.push (step + 1) last
    IO.println s!"  steps={steps} totalLoss0={Tensor.toScalar g0 + Tensor.toScalar d0r + Tensor.toScalar d0f} totalLoss{steps}={last}"
    pure curve

def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (xPath, yPath, nRows, seed, rest) ← Common.orThrow exeName <| RealData.parseCifarFlags rest
      let (train, rest) ← Common.orThrow exeName <|
        Common.parseLoggedTrainFlags exeName rest defaultLogJson 10
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let curve ← trainCurve opts xPath yPath nRows seed train.steps
      Common.writeCurveLogTo train.log "GAN-style CIFAR training" curve "total_loss"
        #[s!"data=cifar10", s!"latentDim={cfg.latentDim}", s!"nRows={nRows}",
          s!"device={if opts.useGpu then "cuda" else "cpu"}"]
    ))
    { banner? := some (fun opts =>
        s!"{exeName}: CIFAR LSGAN-style training (device={if opts.useGpu then "cuda" else "cpu"})")
      printOk := true }

end NN.Examples.Models.Generative.Gan
