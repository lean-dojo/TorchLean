/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe -K cuda=true torchlean mae --cuda --steps 25
-/

module

public import NN.API.Common
public import NN.API.Data
public import NN.API.Models.SelfSupervised
public import NN.API.Models.TrainFixed
public import NN.API.SelfSupervised
public import NN.Examples.Models.Common.RealData

/-!
# Masked Autoencoder CIFAR Example

This is the smallest ViT-MAE-style training path in TorchLean.

The data path is intentionally concrete:

1. load real CIFAR-10 `.npy` arrays through `NN.API.Data`;
2. take a typed image batch with shape `[batch, channels, height, width]`;
3. hide deterministic image patches with `ssl.imagePatchMaeSample`;
4. run a ViT encoder over patch tokens;
5. train a decoder head to reconstruct the original image vector.

This is still intentionally small: one transformer encoder block and a linear pixel decoder rather
than a large asymmetric MAE decoder. The important pieces are the MAE pieces exercised by the
example: image patch masking, patch embedding, transformer tokens, and reconstruction of the
original image.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Generative.Mae

/-- Command name used in error messages and CLI output. -/
def exeName : String := "torchlean mae"

/-- Default training curve location. `data/` is intentionally ignored by git. -/
def defaultLogJson : System.FilePath := "data/model_zoo/mae_trainlog.json"

def batch : Nat := 1
def inC : Nat := RealData.cifarChannels
def inH : Nat := RealData.cifarHeight
def inW : Nat := RealData.cifarWidth
def patchH : Nat := 16
def patchW : Nat := 16
def stride : Nat := 16
def padding : Nat := 0
def dModel : Nat := 8
def numHeads : Nat := 1
def headDim : Nat := 8
def ffnHidden : Nat := 32
def reconDim : Nat := 256

/--
Small ViT-MAE configuration.

CIFAR-10 with `16×16` patches gives `2×2 = 4` patch tokens. The model embeds patches into
`dModel = 8`, runs one transformer encoder block, then decodes the flattened token state back to a
256-pixel prefix of the original image. This is intentionally compact: it keeps the example runnable
while still exercising a real patch-token transformer path. For full-image reconstruction, set
`reconDim := inC * inH * inW`.
-/
def cfg : nn.models.VitMaeConfig :=
  { batch := batch
    inC := inC
    inH := inH
    inW := inW
    patchH := patchH
    patchW := patchW
    stride := stride
    padding := padding
    dModel := dModel
    reconDim := reconDim
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden }

/--
Hide one patch-index class every four patch positions.

The image remains an image tensor; the mask zeros whole patch regions before patch embedding.
-/
def maskPeriod : Nat := 4
def maskOffset : Nat := 0

/-- Input shape: a real batched CIFAR image tensor. -/
abbrev σ : Shape := nn.models.vitMaeInShape cfg

/-- Output shape: flattened image reconstruction. -/
abbrev τ : Shape := nn.models.vitMaeOutShape cfg

/-- CIFAR-10 images are stored as `3 × 32 × 32` tensors. -/
def cifarClasses : Nat := RealData.cifarClasses

/--
Construct the trainable model.

The architecture lives in the public API (`NN.API.Models.SelfSupervised`); the example only chooses
a config and trains it.
-/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.vitMaskedAutoencoder cfg

/--
Load one CIFAR minibatch as an image tensor batch.

This function deliberately stops at the data boundary: it returns CIFAR as typed image tensors.  The
self-supervised conversion happens in `mkMaeSample`, using the public SSL API, so the loader does
not secretly define the model's representation.
-/
def loadCifarBatch (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO (sample.Batch Float cfg.batch RealData.CifarImage RealData.CifarTarget) :=
  RealData.loadCifarBatch (α := Float) exeName cfg.batch nRows seed xPath yPath

/--
Turn a typed CIFAR image batch into the compact MAE training sample.

The input stays an image tensor with some patches zeroed out. The target is the original image
flattened to a vector because the current decoder head predicts a batched matrix.
-/
def mkMaeSample
    (b : sample.Batch Float cfg.batch RealData.CifarImage RealData.CifarTarget) :
    sample.Supervised Float σ τ :=
  ssl.imagePatchMaeSample cfg.batch cfg.inC cfg.inH cfg.inW cfg.reconDim cfg.patchH cfg.patchW
    maskPeriod maskOffset (by decide) (sample.x b)

/--
Train and return a loss curve.

The curve is written by `main` using TorchLean's general training-log JSON format, so plotting and
dashboard tools can consume it the same way they consume the other model examples.
-/
def trainCurve (opts : TorchLean.Options) (xPath yPath : System.FilePath)
    (nRows seed steps : Nat) : IO _root_.Runtime.Training.Curve := do
  let batch ← loadCifarBatch xPath yPath nRows seed
  let sample := mkMaeSample batch
  _root_.NN.API.Models.TrainFixed.curveFloat
    (mkModel := mkModel)
    (mkModuleDef := fun model => nn.mseScalarModuleDef model)
    (mkOptim := fun ps =>
      TorchLean.Optim.adam (α := Float) (paramShapes := ps)
        (lr := 1e-3) (beta1 := 0.9) (beta2 := 0.999) (epsilon := 1e-8))
    (opts := opts) (sample := sample) (steps := steps)

/--
CLI entrypoint.

Useful flags:
- `--cuda` runs the eager training loop on the CUDA runtime.
- `--steps <n>` or `--epochs <n>` controls optimization steps.
- `--x <path> --y <path>` selects custom CIFAR-style `.npy` arrays.
- `--log <path>` writes the training curve JSON.
-/
def main (args : List String) : IO UInt32 := do
  let (rt, rest) ← Common.orThrow exeName <| TorchLean.Module.ExecConfig.parseAndStrip args
  TorchLean.Module.ExecConfig.log rt
  let opts : TorchLean.Options :=
    { backend := rt.backend
      useGpu := rt.useGpu
      fastKernels := rt.fastKernels
      fastGpuMatmulPrecision := rt.fastGpuMatmulPrecision }
  IO.println s!"{exeName}: CIFAR masked reconstruction (device={if opts.useGpu then "cuda" else "cpu"})"
  let (xPath, yPath, nRows, seed, rest) ← Common.orThrow exeName <| RealData.parseCifarFlags rest
  let (train, rest) ← Common.orThrow exeName <|
    Common.parseLoggedTrainFlags exeName rest defaultLogJson 10
  Common.orThrow exeName <| CLI.requireNoArgs rest
  let curve ← trainCurve opts xPath yPath nRows seed train.steps
  let loss0 := curve.values.getD 0 0.0
  let lossN := curve.values.getD (curve.values.size - 1) loss0
  IO.println s!"  steps={train.steps} loss0={loss0} loss{train.steps}={lossN}"
  Common.writeCurveLogTo train.log "MAE CIFAR masked reconstruction" curve "loss"
    #[s!"data=cifar10", s!"x={xPath}", s!"y={yPath}", s!"nRows={nRows}",
      s!"maskPeriod={maskPeriod}", s!"maskOffset={maskOffset}",
      s!"device={if opts.useGpu then "cuda" else "cpu"}"]
  IO.println "OK"
  pure 0

end NN.Examples.Models.Generative.Mae
