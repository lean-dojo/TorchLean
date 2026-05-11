/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.API.Models.Diffusion
public import NN.Examples.Models.Common.RealData
public import NN.Runtime.Autograd.TorchLean.NN

/-!
# Diffusion Training Example

Runnable `torchlean diffusion` example.

This is the maintained diffusion command. It supports two real-data modes:

- `--dataset imagenet64` (default): user-provided ImageNet/Imagenette/Tiny-ImageNet-style images
  converted to `(N,3,64,64)` `.npy` tensors.
- `--dataset cifar10`: prepared CIFAR-10 `(N,3,32,32)` arrays.

The command is one public entrypoint, but the implementation keeps separate typed branches because
Lean tracks image height and width in the tensor type.

## Why unconditional samples are still rough

The default epsilon predictor is a small same-resolution residual CNN with a broadcast time channel.
That is enough to validate real image loading, CUDA training, logging, reconstruction diagnostics,
and DDIM replay from Lean.  It is deliberately not advertised as a high-fidelity image generator:
good unconditional samples need a full U-Net with multiscale skips, richer timestep embeddings, EMA,
more training, more timesteps, and runtime support that avoids eager-autograd buffer blow-up for
wider models.

## Examples

Prepare ImageNet-style data:

```bash
python3 scripts/datasets/torchlean_data_convert.py image-folder \
  --input /path/to/imagenet/train \
  --x-output data/real/imagenet64/imagenet64_train_X.npy \
  --y-output data/real/imagenet64/imagenet64_train_y.npy \
  --height 64 --width 64 --labels-from-dirs --limit 800
```

Train on ImageNet64 and save visual artifacts:

```bash
lake build -R -K cuda=true
CUDA_VISIBLE_DEVICES=0 lake exe -K cuda=true torchlean diffusion --cuda --fast-kernels \
  --dataset imagenet64 --n-total 800 --steps 1000 --hidden-c 8 --T 100 --beta-end 0.12 \
  --reference-ppm data/model_zoo/diffusion_reference.ppm \
  --noisy-ppm data/model_zoo/diffusion_noisy.ppm \
  --reconstruct-ppm data/model_zoo/diffusion_reconstruct.ppm \
  --sample-ppm data/model_zoo/diffusion_sample.ppm
```

CIFAR smoke path:

```bash
python3 scripts/datasets/download_example_data.py --cifar10
lake exe torchlean diffusion --dataset cifar10 --cuda --fast-kernels --steps 200
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Generative.Diffusion

def exeName : String := "torchlean diffusion"

def defaultLogJson : System.FilePath := "data/model_zoo/diffusionlog.json"

def batch : Nat := 4

abbrev x0Shape (c h w : Nat) : Shape :=
  NN.Tensor.Shape.NCHW batch c h w

abbrev xInShape (c h w : Nat) : Shape :=
  NN.Tensor.Shape.NCHW batch (c + 1) h w

def cfgFor (c h w hiddenC : Nat) : nn.models.EpsConvNetConfig :=
  { batch := batch, dataC := c, h := h, w := w, hiddenC := hiddenC }

/--
Build the default epsilon predictor for a specific typed image shape.

We use the residual CNN from `NN.API.Models.Diffusion`: it is still small enough for tutorial-scale
CUDA runs, but the skip paths train much better than the plain convolution chain.  The plain
`epsConvNet` remains in the API as the smaller baseline; this example uses the residual default so
the documented command matches the maintained training path.
-/
def mkModel (c h w hiddenC : Nat)
    [NeZero c] [NeZero h] [NeZero w] (h_hiddenC : hiddenC ≠ 0) :
    nn.M (nn.Sequential (xInShape c h w) (x0Shape c h w)) :=
  nn.models.epsResidualConvNet (cfgFor c h w hiddenC)
    (h_batch := by simp [cfgFor, batch])
    (h_dataC := by exact NeZero.ne c)
    (h_inC := by exact Nat.succ_ne_zero c)
    (h_h := by exact NeZero.ne h)
    (h_w := by exact NeZero.ne w)
    (h_hiddenC := h_hiddenC)

/--
Map converted image tensors from `[0,1]` into the standard diffusion range `[-1,1]`.

The input is already NCHW because the dataset converter and `RealData` loaders enforce that layout.
-/
def toDiffusionRange {c h w : Nat} (x01 : Tensor Float (x0Shape c h w)) :
    Tensor Float (x0Shape c h w) :=
  Spec.Tensor.mapSpec (fun x => 2.0 * x - 1.0) x01

def cifarBatchX0
    (batchSample : sample.Batch Float batch RealData.CifarImage RealData.CifarTarget) :
    Tensor Float (x0Shape RealData.cifarChannels RealData.cifarHeight RealData.cifarWidth) := by
  let x01 : Tensor Float (x0Shape RealData.cifarChannels RealData.cifarHeight RealData.cifarWidth) := by
    simpa [x0Shape, NN.Tensor.Shape.NCHW, RealData.CifarImage, Shape.Image] using
      (API.sample.x batchSample)
  exact toDiffusionRange x01

def imageNet64BatchX0
    (batchSample : sample.Batch Float batch RealData.ImageNet64Image RealData.ImageNet64Target) :
    Tensor Float (x0Shape RealData.imagenet64Channels RealData.imagenet64Height
      RealData.imagenet64Width) := by
  let x01 : Tensor Float (x0Shape RealData.imagenet64Channels RealData.imagenet64Height
      RealData.imagenet64Width) := by
    simpa [x0Shape, NN.Tensor.Shape.NCHW, RealData.ImageNet64Image, Shape.Image] using
      (API.sample.x batchSample)
  exact toDiffusionRange x01

def loadCifarX0Batches (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO (List (Tensor Float (x0Shape RealData.cifarChannels RealData.cifarHeight
      RealData.cifarWidth))) := do
  unless (← xPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing CIFAR-10 images: {xPath}\n{RealData.missingCifarHint}"
  unless (← yPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing CIFAR-10 labels: {yPath}\n{RealData.missingCifarHint}"
  let src := Data.LabeledSource.ofPaths .npy xPath yPath nRows
    [RealData.cifarChannels, RealData.cifarHeight, RealData.cifarWidth] RealData.cifarClasses
  let dsE ← src.load (α := Float)
  let ds ←
    match dsE with
    | .ok ds => pure ds
    | .error msg =>
        throw <| IO.userError s!"{exeName}: failed to load CIFAR-10 arrays for --n-total {nRows}.\n{msg}"
  let dl := Data.batchLoader ds batch (shuffle := true) (seed := seed) (dropLast := true)
  let (_dl', batches) ← Common.orThrow exeName <| Data.BatchLoader.epoch exeName dl
  match batches.map cifarBatchX0 with
  | [] =>
      throw <| IO.userError
        s!"{exeName}: no full CIFAR-10 minibatch available (batch={batch}, rows={Data.size ds})"
  | xs => pure xs

def loadImageNet64X0Batches (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO (List (Tensor Float (x0Shape RealData.imagenet64Channels RealData.imagenet64Height
      RealData.imagenet64Width))) := do
  unless (← xPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing ImageNet64 images: {xPath}\n{RealData.missingImageNet64Hint}"
  unless (← yPath.pathExists) do
    throw <| IO.userError s!"{exeName}: missing ImageNet64 labels: {yPath}\n{RealData.missingImageNet64Hint}"
  let src := Data.LabeledSource.ofPaths .npy xPath yPath nRows
    [RealData.imagenet64Channels, RealData.imagenet64Height, RealData.imagenet64Width]
    RealData.imagenet64Classes
  let dsE ← src.load (α := Float)
  let ds ←
    match dsE with
    | .ok ds => pure ds
    | .error msg =>
        throw <| IO.userError
          s!"{exeName}: failed to load ImageNet64 arrays for --n-total {nRows}.\n{msg}"
  let dl := Data.batchLoader ds batch (shuffle := true) (seed := seed) (dropLast := true)
  let (_dl', batches) ← Common.orThrow exeName <| Data.BatchLoader.epoch exeName dl
  match batches.map imageNet64BatchX0 with
  | [] =>
      throw <| IO.userError
        s!"{exeName}: no full ImageNet64 minibatch available (batch={batch}, rows={Data.size ds})"
  | xs => pure xs

def randomEps {c h w : Nat} (seed step : Nat) : Tensor Float (x0Shape c h w) :=
  let key : UInt64 := _root_.Runtime.Autograd.TorchLean.Random.keyOf (seed := seed) (counter := step)
  _root_.Runtime.Autograd.TorchLean.Random.normal (α := Float) key (s := x0Shape c h w)

def mkNoisedSample {c h w : Nat} (alphaBars : Array Float) (T : Nat)
    (x0 : Tensor Float (x0Shape c h w)) (seed step : Nat) :
    sample.Supervised Float (xInShape c h w) (x0Shape c h w) :=
  NN.API.diffusion.noisedSampleFromEps alphaBars T x0 (randomEps (c := c) (h := h) (w := w) seed step)
    (seed + step)

def reverseDdim {c h w : Nat}
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential (xInShape c h w) (x0Shape c h w))
    (params : _root_.Runtime.Autograd.Torch.ParamList Float (nn.paramShapes model))
    (alphaBars : Array Float) (T : Nat) (xStart : Tensor Float (x0Shape c h w)) :
    IO (Tensor Float (x0Shape c h w)) := do
  let mut x_t := xStart
  if T > 1 then
    for tRev in [0:T] do
      let tIdx : Nat := (T - 1) - tRev
      let ab : Float := alphaBars.getD tIdx 1.0
      let abPrev : Float := if tIdx = 0 then 1.0 else alphaBars.getD (tIdx - 1) 1.0
      let tNorm : Float := Float.ofNat tIdx / Float.ofNat (T - 1)
      let epsHat ← nn.eval1 (α := Float) opts model params
        (NN.API.diffusion.appendTimeChannel x_t tNorm)
      x_t := NN.API.diffusion.ddimPrev abPrev ab x_t epsHat
  pure x_t

/--
Reverse DDIM from a chosen timestep for reconstruction diagnostics.

This is intentionally separate from unconditional sampling.  It lets us corrupt a real image to a
moderate timestep, denoise from there, and check whether reconstruction improves over the noisy
input.
-/
def reverseDdimFrom {c h w : Nat}
    (opts : Runtime.Autograd.Torch.Options)
    (model : nn.Sequential (xInShape c h w) (x0Shape c h w))
    (params : _root_.Runtime.Autograd.Torch.ParamList Float (nn.paramShapes model))
    (alphaBars : Array Float) (T tStart : Nat) (xStart : Tensor Float (x0Shape c h w)) :
    IO (Tensor Float (x0Shape c h w)) := do
  let mut x_t := xStart
  for tRev in [0:tStart + 1] do
    let tIdx : Nat := tStart - tRev
    let ab : Float := alphaBars.getD tIdx 1.0
    let abPrev : Float := if tIdx = 0 then 1.0 else alphaBars.getD (tIdx - 1) 1.0
    let tNorm : Float := if T <= 1 then 0.0 else Float.ofNat tIdx / Float.ofNat (T - 1)
    let epsHat ← nn.eval1 (α := Float) opts model params
      (NN.API.diffusion.appendTimeChannel x_t tNorm)
    x_t := NN.API.diffusion.ddimPrev abPrev ab x_t epsHat
  pure x_t

structure TrainConfig where
  steps : Nat
  logEvery : Nat
  lr : Float
  T : Nat
  hiddenC : Nat
  betaStart : Float
  betaEnd : Float
  reconstructStep? : Option Nat
  samplePpm? : Option System.FilePath
  referencePpm? : Option System.FilePath
  noisyPpm? : Option System.FilePath
  reconstructPpm? : Option System.FilePath

/--
Shared training loop for both CIFAR-10 and ImageNet64 branches.

The loop optimizes epsilon prediction and can emit four visual artifacts:

- `reference-ppm`: clean evaluation image,
- `noisy-ppm`: clean image after forward diffusion to `reconstruct-step`,
- `reconstruct-ppm`: DDIM denoising from that timestep,
- `sample-ppm`: unconditional DDIM sample from Gaussian noise.
-/
def trainCurveFloat {c h w : Nat} [NeZero c] [NeZero h] [NeZero w]
    (opts : Runtime.Autograd.Torch.Options)
    (loadBatches : IO (List (Tensor Float (x0Shape c h w))))
    (cfg : TrainConfig) (h_hiddenC : cfg.hiddenC ≠ 0) :
    IO _root_.Runtime.Training.Curve := do
  let batches ← loadBatches
  let batchAt ← Common.orThrow exeName <|
    Data.cycleListOrError batches s!"{exeName}: no training minibatches available"
  let evalX0 := batchAt 0
  let alphaBars := NN.API.diffusion.alphaBarsLinear cfg.T cfg.betaStart cfg.betaEnd
  nn.withModel (mkModel c h w cfg.hiddenC h_hiddenC) fun model => do
    let modDef := nn.mseScalarModuleDef model
    let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    let opt := TorchLean.Optim.adam (α := Float)
      (paramShapes := nn.paramShapes model)
      (lr := cfg.lr) (beta1 := 0.9) (beta2 := 0.999) (epsilon := 1e-8)
    let optH ← TorchLean.Optim.handle (α := Float) m opt
    let evalStep := if cfg.T = 0 then 0 else cfg.T / 2
    let evalSample := mkNoisedSample alphaBars cfg.T evalX0 (seed := opts.seed) (step := evalStep)
    let loss0 ← TorchLean.Module.forward (α := Float) m evalSample
    let L0 := Tensor.toScalar loss0
    let mut curve : _root_.Runtime.Training.Curve := {}
    curve := curve.push 0 L0
    let mut last := L0
    for step in [0:cfg.steps] do
      let x0 := batchAt step
      let s := mkNoisedSample alphaBars cfg.T x0 (seed := opts.seed) (step := step + 1)
      optH.step s
      let done := step + 1
      if cfg.logEvery != 0 && done % cfg.logEvery == 0 then
        let loss ← TorchLean.Module.forward (α := Float) m evalSample
        last := Tensor.toScalar loss
        curve := curve.push done last
    if cfg.logEvery == 0 || cfg.steps % cfg.logEvery != 0 then
      let loss ← TorchLean.Module.forward (α := Float) m evalSample
      last := Tensor.toScalar loss
      curve := curve.push cfg.steps last
    IO.println s!"  steps={cfg.steps} loss0={L0} lastLoss={last}"
    match cfg.referencePpm? with
    | none => pure ()
    | some path => NN.API.diffusion.writeFirstRgbNchwPpm path evalX0
    match cfg.samplePpm? with
    | none => pure ()
    | some path => do
        let x_T := randomEps (c := c) (h := h) (w := w) (seed := opts.seed) (step := 999)
        let x_t ← reverseDdim opts model m.trainer.params alphaBars cfg.T x_T
        NN.API.diffusion.writeFirstRgbNchwPpm path x_t
    match cfg.reconstructPpm? with
    | none => pure ()
    | some path => do
        let tIdx : Nat := if cfg.T = 0 then 0 else Nat.min (cfg.reconstructStep?.getD (cfg.T / 4)) (cfg.T - 1)
        let ab : Float := alphaBars.getD tIdx 1.0
        let sqrtAb : Float := MathFunctions.sqrt (Max.max ab 0.0)
        let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (1.0 - ab) 0.0)
        let eps := randomEps (c := c) (h := h) (w := w) (seed := opts.seed) (step := 1001)
        let xNoisy : Tensor Float (x0Shape c h w) :=
          Spec.Tensor.addSpec
            (Spec.Tensor.scaleSpec evalX0 sqrtAb)
            (Spec.Tensor.scaleSpec eps sqrtOneMinusAb)
        match cfg.noisyPpm? with
        | none => pure ()
        | some noisyPath => NN.API.diffusion.writeFirstRgbNchwPpm noisyPath xNoisy
        let x_t ← reverseDdimFrom opts model m.trainer.params alphaBars cfg.T tIdx xNoisy
        NN.API.diffusion.writeFirstRgbNchwPpm path x_t
    pure curve

inductive DatasetChoice where
  | imagenet64
  | cifar10
deriving Repr, BEq

namespace DatasetChoice

def parse (args : List String) : Except String (DatasetChoice × List String) := do
  let (dataset?, args) ← CLI.takeFlagValueOnce args "dataset"
  let (cifarFlag, args) ← CLI.takeBoolFlagOnce args "cifar10"
  let (imagenetFlag, args) ← CLI.takeBoolFlagOnce args "imagenet64"
  match dataset?, cifarFlag, imagenetFlag with
  | some _, true, _ | some _, _, true | none, true, true =>
      throw "choose only one dataset selector: --dataset, --cifar10, or --imagenet64"
  | some raw, false, false =>
      match raw with
      | "imagenet64" | "imagenet" | "imagenette64" => pure (.imagenet64, args)
      | "cifar10" | "cifar" => pure (.cifar10, args)
      | _ => throw s!"unknown --dataset {raw}; expected imagenet64 or cifar10"
  | none, true, false => pure (.cifar10, args)
  | none, false, true => pure (.imagenet64, args)
  | none, false, false => pure (.imagenet64, args)

end DatasetChoice

def parseTrainConfig (args : List String) :
    Except String (TrainConfig × _root_.Runtime.Training.LogDestination × List String) := do
  let (train, rest) ← Common.parseLoggedTrainFlags exeName args defaultLogJson 50
  let (logEvery?, rest) ← CLI.takeNatFlagOnce rest "log-every"
  let (lr?, rest) ← CLI.takeFloatFlagOnce rest "lr"
  let (T?, rest) ← CLI.takeNatFlagOnce rest "T"
  let (hiddenC?, rest) ← CLI.takeNatFlagOnce rest "hidden-c"
  let (betaStart?, rest) ← CLI.takeFloatFlagOnce rest "beta-start"
  let (betaEnd?, rest) ← CLI.takeFloatFlagOnce rest "beta-end"
  let (reconstructStep?, rest) ← CLI.takeNatFlagOnce rest "reconstruct-step"
  let (samplePpm?, rest) ← CLI.takePathFlagOnce rest "sample-ppm"
  let (referencePpm?, rest) ← CLI.takePathFlagOnce rest "reference-ppm"
  let (noisyPpm?, rest) ← CLI.takePathFlagOnce rest "noisy-ppm"
  let (reconstructPpm?, rest) ← CLI.takePathFlagOnce rest "reconstruct-ppm"
  let logEveryDefault := Nat.max 1 (train.steps / 50)
  pure ({ steps := train.steps,
          logEvery := logEvery?.getD logEveryDefault,
          lr := lr?.getD 1e-3,
          T := T?.getD 100,
          hiddenC := hiddenC?.getD 16,
          betaStart := betaStart?.getD 1e-4,
          -- Short tutorial chains need a stronger terminal noise level than the usual 1000-step
          -- DDPM beta end. With `T=100`, `betaEnd=0.12` leaves `sqrt(alpha_bar_T) ≈ 0.044`,
          -- so unconditional sampling from Gaussian noise is at least distributionally plausible.
          betaEnd := betaEnd?.getD 0.12,
          reconstructStep? := reconstructStep?,
          samplePpm? := samplePpm?,
          referencePpm? := referencePpm?,
          noisyPpm? := noisyPpm?,
          reconstructPpm? := reconstructPpm? },
        train.log,
        rest)

def writeTrainingLog (log : _root_.Runtime.Training.LogDestination) (dataset : String)
    (sourceNotes : Array String) (cfg : TrainConfig) (opts : Runtime.Autograd.Torch.Options)
    (curve : _root_.Runtime.Training.Curve) : IO Unit :=
  Common.writeCurveLogTo log "Diffusion training" curve "loss"
    (sourceNotes ++
      #[s!"dataset={dataset}",
        s!"device={if opts.useGpu then "cuda" else "cpu"}",
        s!"lr={cfg.lr}",
        s!"T={cfg.T}",
        s!"hiddenC={cfg.hiddenC}",
        s!"betaStart={cfg.betaStart}",
        s!"betaEnd={cfg.betaEnd}",
        s!"logEvery={cfg.logEvery}"] ++
      (match cfg.reconstructStep? with | none => #[] | some t => #[s!"reconstructStep={t}"]) ++
      (match cfg.samplePpm? with | none => #[] | some p => #[s!"samplePpm={p}"]) ++
      (match cfg.referencePpm? with | none => #[] | some p => #[s!"referencePpm={p}"]) ++
      (match cfg.noisyPpm? with | none => #[] | some p => #[s!"noisyPpm={p}"]) ++
      (match cfg.reconstructPpm? with | none => #[] | some p => #[s!"reconstructPpm={p}"]))

def runImageNet64 (opts : Runtime.Autograd.Torch.Options) (args : List String) : IO Unit := do
  let (xPath, yPath, nRows, seed, args) ← Common.orThrow exeName <| RealData.parseImageNet64Flags args
  let (cfg, log, rest) ← Common.orThrow exeName <| parseTrainConfig args
  Common.orThrow exeName <| CLI.requireNoArgs rest
  match cfg.hiddenC with
  | 0 => throw <| IO.userError s!"{exeName}: --hidden-c must be > 0"
  | hc + 1 =>
      let cfg := { cfg with hiddenC := hc + 1 }
      let sourceNotes :=
        #[s!"data=imagenet64", s!"x={xPath}", s!"y={yPath}", s!"nRows={nRows}"]
      let load := loadImageNet64X0Batches xPath yPath nRows seed
      let curve ← trainCurveFloat opts load cfg (Nat.succ_ne_zero hc)
      writeTrainingLog log "imagenet64" sourceNotes cfg opts curve

def runCifar10 (opts : Runtime.Autograd.Torch.Options) (args : List String) : IO Unit := do
  let (xPath, yPath, nRows, seed, args) ← Common.orThrow exeName <| RealData.parseCifarFlags args
  let (cfg, log, rest) ← Common.orThrow exeName <| parseTrainConfig args
  Common.orThrow exeName <| CLI.requireNoArgs rest
  match cfg.hiddenC with
  | 0 => throw <| IO.userError s!"{exeName}: --hidden-c must be > 0"
  | hc + 1 =>
      let cfg := { cfg with hiddenC := hc + 1 }
      let sourceNotes :=
        #[s!"data=cifar10", s!"x={xPath}", s!"y={yPath}", s!"nRows={nRows}"]
      let load := loadCifarX0Batches xPath yPath nRows seed
      let curve ← trainCurveFloat opts load cfg (Nat.succ_ne_zero hc)
      writeTrainingLog log "cifar10" sourceNotes cfg opts curve

def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (choice, rest) ← Common.orThrow exeName <| DatasetChoice.parse rest
      match choice with
      | .imagenet64 => runImageNet64 opts rest
      | .cifar10 => runCifar10 opts rest))
    { banner? := some (fun opts =>
        s!"{exeName}: diffusion trainer (device={if opts.useGpu then "cuda" else "cpu"})")
      printOk := true }

end NN.Examples.Models.Generative.Diffusion
