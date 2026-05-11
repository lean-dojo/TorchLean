/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe -K cuda=true torchlean vqvae --cuda --steps 10
-/

module

public import NN
public import NN.API.Models.Generative
public import NN.API.Models.TrainFixed
public import NN.Examples.Models.Common.RealData
public import NN.Spec.Models.VqVae
public import NN.MLTheory.Generative.Latent.VQVAE

/-!
# VQ-VAE-Style CIFAR Example

Trains a small vector reconstruction model with a narrow `tanh` bottleneck, paired with the
VQ-VAE spec/theory modules. The theorem-facing codebook objective lives in `NN.Spec.Models.VqVae`;
this runtime example is the executable reconstruction path.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Generative.VqVae

def exeName : String := "torchlean vqvae"
def defaultLogJson : System.FilePath := "data/model_zoo/vqvae_trainlog.json"
def cfg : nn.models.VectorGenerativeConfig := nn.models.compactImageConfig

abbrev σ : Shape := nn.models.vectorDataShape cfg
abbrev τ : Shape := nn.models.vectorDataShape cfg

def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.vectorVqVae cfg

def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (xPath, yPath, nRows, seed, rest) ← Common.orThrow exeName <| RealData.parseCifarFlags rest
      let (train, rest) ← Common.orThrow exeName <|
        Common.parseLoggedTrainFlags exeName rest defaultLogJson 10
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let x ← RealData.loadCifarVectorBatch cfg (by decide) exeName xPath yPath nRows seed
      let sample := nn.models.reconstructionSample cfg x
      let curve ←
        _root_.NN.API.Models.TrainFixed.curveFloat
          (mkModel := mkModel)
          (mkModuleDef := fun model => nn.mseScalarModuleDef model)
          (mkOptim := fun ps =>
            TorchLean.Optim.adam (α := Float) (paramShapes := ps)
              (lr := 1e-3) (beta1 := 0.9) (beta2 := 0.999) (epsilon := 1e-8))
          (opts := opts) (sample := sample) (steps := train.steps)
      let loss0 := curve.values.getD 0 0.0
      let lossN := curve.values.getD (curve.values.size - 1) loss0
      IO.println s!"  steps={train.steps} loss0={loss0} loss{train.steps}={lossN}"
      Common.writeCurveLogTo train.log "VQ-VAE-style CIFAR reconstruction" curve "loss"
        #[s!"data=cifar10", s!"latentDim={cfg.latentDim}", s!"nRows={nRows}",
          s!"device={if opts.useGpu then "cuda" else "cpu"}"]
    ))
    { banner? := some (fun opts =>
        s!"{exeName}: CIFAR VQ-VAE-style training (device={if opts.useGpu then "cuda" else "cpu"})")
      printOk := true }

end NN.Examples.Models.Generative.VqVae
