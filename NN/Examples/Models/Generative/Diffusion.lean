/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.API.Data.Image
public import NN.API.Models.Diffusion.Sampling
public import NN.Examples.Models.Common.RealData

/-!
# Diffusion Training Example

Runnable `torchlean diffusion` example.

This is the maintained diffusion command. It supports two real-data modes:

- `--dataset imagenet64` (default): user-provided ImageNet/Imagenette/Tiny-ImageNet-style images
  converted to `(N,3,64,64)` `.npy` tensors.
- `--dataset cifar10`: prepared CIFAR-10 `(N,3,32,32)` arrays.

The command is one public entrypoint, but the implementation keeps separate typed branches because
Lean tracks image height and width in the tensor type.

## Why unconditional samples are still modest

The default epsilon predictor is a compact same-resolution residual CNN with a broadcast time
channel. That is enough to validate real image loading, CUDA training, logging, reconstruction
diagnostics, and DDIM replay from Lean. High-fidelity unconditional samples require more machinery:
a full U-Net with multiscale skips, richer timestep embeddings, EMA, more training, more timesteps,
and runtime support that avoids eager-autograd buffer blow-up for wider models.

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
lake -R -K cuda=true build
CUDA_VISIBLE_DEVICES=0 lake -R -K cuda=true exe torchlean diffusion --device cuda \
  --dataset imagenet64 --n-total 800 --steps 1000 --hidden-c 8 --T 100 --beta-end 0.12 \
  --log data/examples/diffusion_trainlog.json \
  --reference-ppm data/examples/diffusion_reference.ppm \
  --noisy-ppm data/examples/diffusion_noisy.ppm \
  --reconstruct-ppm data/examples/diffusion_reconstruct.ppm \
  --sample-ppm data/examples/diffusion_sample.ppm
```

CIFAR run:

```bash
python3 scripts/datasets/download_example_data.py --cifar10
lake -R -K cuda=true exe torchlean diffusion --device cuda --dataset cifar10 --n-total 1 \
  --steps 1 --hidden-c 2 --T 2
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Generative.Diffusion

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "diffusion"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "diffusion"

/-- Static minibatch size used by both CIFAR-10 and ImageNet64 typed branches. -/
def batchSize : Nat := 1

/-- Cropped CIFAR height for the compact runnable diffusion example. -/
def cifarCropHeight : Nat := 4

/-- Cropped CIFAR width for the compact runnable diffusion example. -/
def cifarCropWidth : Nat := 4

local instance : NeZero cifarCropHeight := ⟨by decide⟩
local instance : NeZero cifarCropWidth := ⟨by decide⟩

/-- Clean image batch dimensions $x_0$: NCHW with the fixed command batch size. -/
abbrev output (c h w : Nat) : Shape :=
  [batchSize, c, h, w]

/-- Epsilon-model input dimensions: image channels plus one broadcast timestep channel. -/
abbrev input (c h w : Nat) : Shape :=
  [batchSize, c + 1, h, w]

/--
Architecture of the epsilon predictor for a particular image size.

A kernel radius of one means every convolution uses a `3 x 3` same-padding kernel.
-/
def config (c h w hiddenChannels : Nat) :
    nn.models.Diffusion.NoisePredictor.Config 2 :=
  { dataChannels := c
    spatial := [h, w]
    hiddenChannels := hiddenChannels
    kernelRadius := [1, 1] }

/--
Build the residual epsilon predictor for a specific typed image shape.

Every `3×3` convolution preserves the spatial extent, and the residual blocks mix information
between neighboring pixels while remaining small enough for the runnable CUDA check.
-/
def model (c h w hiddenChannels : Nat) :
    nn.Builder (nn.Sequential (input c h w) (output c h w)) :=
  by
    let built :=
      nn.models.Diffusion.NoisePredictor.residual
        (config c h w hiddenChannels) (batchShape := [batchSize])
    rw [nn.models.Diffusion.NoisePredictor.Config.input,
      nn.models.Diffusion.NoisePredictor.Config.output] at built
    simpa [input, output, config,
      Shape.ofList, Shape.concat] using built

/--
Convert one typed CIFAR minibatch into diffusion-space clean images.

The loader returns images in $[0,1]$; diffusion training uses $[-1,1]$, so this function performs
the range conversion after Lean has established the CIFAR NCHW shape.
-/
def cifarBatch
    (batchSample : Sample.Batch Float batchSize RealData.CifarImage RealData.CifarTarget) :
    Except String (Tensor Float
      (output RealData.cifarChannels cifarCropHeight cifarCropWidth)) := do
  let cropped ←
    RealData.cropCifarBatch batchSize cifarCropHeight cifarCropWidth batchSample
  let unitImage :
      Tensor Float (output RealData.cifarChannels cifarCropHeight cifarCropWidth) := by
    simpa [output] using cropped.input
  pure (diffusion.unitToSignedUnit unitImage)

/--
Convert one typed ImageNet64 minibatch into diffusion-space clean images.

This mirrors `cifarBatch` but keeps the ImageNet64 height/width/channel constants in the type.
-/
def imageNet64Batch
    (batchSample :
      Sample.Batch Float batchSize RealData.ImageNet64Image RealData.ImageNet64Target) :
    Tensor Float (output RealData.imagenet64Channels RealData.imagenet64Height
      RealData.imagenet64Width) := by
  let unitImage : Tensor Float (output RealData.imagenet64Channels RealData.imagenet64Height
      RealData.imagenet64Width) := by
    simpa [output, RealData.ImageNet64Image] using
      batchSample.input
  exact diffusion.unitToSignedUnit unitImage

/--
Load CIFAR-10 batches as a tensor of clean diffusion minibatches.

The function validates the `.npy` paths, builds a typed `Data.Loader`, drops incomplete final
batches, and returns NCHW tensors already mapped into $[-1,1]$.
-/
def loadCifar (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO ((count : Nat) × Tensor Float
      ((output RealData.cifarChannels cifarCropHeight cifarCropWidth).prependDim count)) := do
  let batches ← RealData.loadCifarBatches exeName batchSize nRows seed xPath yPath
  let images ← Tensor.stackLeadingM fun index : Fin batches.size =>
    CLI.orThrow exeName (cifarBatch batches[index])
  pure ⟨batches.size, images⟩

/--
Load ImageNet64-style batches as a tensor of clean diffusion minibatches.

The converter accepts ImageNet/Imagenette/Tiny-ImageNet-style folders ahead of time; this Lean path
only consumes the prepared `.npy` arrays and keeps the tensor shapes explicit.
-/
def loadImageNet64 (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO ((count : Nat) × Tensor Float
      ((output RealData.imagenet64Channels RealData.imagenet64Height
        RealData.imagenet64Width).prependDim count)) := do
  let batches ← RealData.loadImageNet64Batches exeName batchSize nRows seed xPath yPath
  pure ⟨batches.size, Tensor.stackLeading fun index => imageNet64Batch batches[index]⟩

/-- Adapt image-and-time conditioning to the dataset-independent DDIM sampler. -/
def conditionedPredictor {c h w T : Nat}
    (predict : Tensor Float (input c h w) → IO (Tensor Float (output c h w)))
    (index : Fin T) (sample : Tensor Float (output c h w)) :
    IO (Tensor Float (output c h w)) :=
  let time := if T <= 1 then 0.0 else Float.ofNat index.val / Float.ofNat (T - 1)
  let conditioned : Tensor Float (input c h w) := by
    simpa [input, output] using
      diffusion.appendTimeChannel [batchSize] ([h, w] : Tensor Nat [2])
        (by simpa [output] using sample) time
  predict conditioned

/--
Diffusion command-line options after parsing.

The inherited pieces make the CLI shape explicit: ordinary training flags come from `Support`,
diffusion math lives in `Support.DiffusionScheduleFlags`, visual outputs live in
`Support.ImageArtifactFlags`, and the epsilon-network width is the model-specific knob.
-/
structure Options where
  /-- Optimizer, step, batching, and logging controls. -/
  training : CLI.Training.OptimizerOptions
  /-- Diffusion timestep and beta schedule. -/
  schedule : Support.DiffusionScheduleFlags
  /-- Optional generated and reconstructed image paths. -/
  artifacts : Support.ImageArtifactFlags
  /-- Hidden channel width of the epsilon predictor. -/
  hiddenChannels : Nat
deriving Repr

/--
Shared training loop for both CIFAR-10 and ImageNet64 branches.

The loop optimizes epsilon prediction and can emit four visual artifacts:

- `reference-ppm`: clean evaluation image,
- `noisy-ppm`: clean image after forward diffusion to `reconstruct-step`,
- `reconstruct-ppm`: DDIM denoising from that timestep,
- `sample-ppm`: unconditional DDIM sample from Gaussian noise.
-/
def Training.run
    {c h w : Nat} [NeZero c] [NeZero h] [NeZero w]
    (runtime : Runtime.Config)
    (loadBatches : IO ((count : Nat) × Tensor Float ((output c h w).prependDim count)))
    (config : Options)
    (schedule : diffusion.Schedule config.schedule.T) :
    IO Training.Curve := do
  let ⟨count, batches⟩ ← loadBatches
  if empty : count = 0 then
    throw (IO.userError s!"{exeName}: no training minibatches available")
  else
  let batchAt (step : Nat) := batches.get ⟨step % count, Nat.mod_lt _ (Nat.pos_of_ne_zero empty)⟩
  let evalX0 := batchAt 0
  let alphaBars := schedule.alphaBars
  let evalStep := config.schedule.T / 2
  let evalSample : Sample.Supervised Float (input c h w) (output c h w) := by
    simpa [input, output] using
      diffusion.noisedSample [batchSize] ([h, w] : Tensor Nat [2])
        schedule (by simpa [output] using evalX0) (seed := runtime.seed)
        (step := evalStep)
  let trainer :=
    Trainer.new (model c h w config.hiddenChannels) <|
      Trainer.RunConfig.forObjective
        (Trainer.RunConfig.fromRuntime runtime
          { optimizer := optim.adam { learningRate := config.training.learningRate } })
        .meanSquaredError
        (seed := runtime.seed)
  trainer.printSummary
  let curveEvery : Nat := Nat.max 1 (config.training.steps / 50)
  let trained ← trainer.trainStream runtime
    (fun step =>
      let x0 := batchAt step
      show Sample.Supervised Float (input c h w) (output c h w) from by
        simpa [input, output] using
          diffusion.noisedSample [batchSize] ([h, w] : Tensor Nat [2])
            schedule (by simpa [output] using x0)
            (seed := runtime.seed) (step := step + 1))
    evalSample
    { steps := config.training.steps
      cudaMemorySampleEvery := config.training.cudaMemorySampleEvery
      logDestination := .disabled }
    (curveEvery := curveEvery)
  let curve := trained.curve
  trained.printSummary
  match config.artifacts.referencePpm? with
  | none => pure ()
  | some path => Data.Image.writeFirstRgbPpm path (evalX0.map fun x => (x + 1.0) / 2.0)
  match config.artifacts.samplePpm? with
  | none => pure ()
  | some path => do
      let x_T := diffusion.normalNoise (shape := output c h w)
        (seed := runtime.seed) (step := 999)
      let x_t ← diffusion.reverseDdim (conditionedPredictor trained.predict) alphaBars x_T
      Data.Image.writeFirstRgbPpm path (x_t.map fun x => (x + 1.0) / 2.0)
  match config.artifacts.reconstructPpm? with
  | none => pure ()
  | some path => do
      let tIdxNat :=
        Nat.min
          (config.artifacts.reconstructStep?.getD (config.schedule.T / 4))
          (config.schedule.T - 1)
      let tIdx := schedule.index tIdxNat
      let ab : Float := alphaBars[tIdx]
      let sqrtAb : Float := MathFunctions.sqrt (Max.max ab 0.0)
      let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (1.0 - ab) 0.0)
      let eps := diffusion.normalNoise (shape := output c h w)
        (seed := runtime.seed) (step := 1001)
      let noisyImage : Tensor Float (output c h w) :=
        evalX0.scale sqrtAb + eps.scale sqrtOneMinusAb
      match config.artifacts.noisyPpm? with
      | none => pure ()
      | some noisyPath =>
          Data.Image.writeFirstRgbPpm noisyPath (noisyImage.map fun x => (x + 1.0) / 2.0)
      let x_t ← diffusion.reverseDdimFrom
        (conditionedPredictor trained.predict) alphaBars tIdx noisyImage
      Data.Image.writeFirstRgbPpm path (x_t.map fun x => (x + 1.0) / 2.0)
  pure curve

/-- Train the diffusion example after rejecting an empty timestep schedule. -/
def train {c h w : Nat} [NeZero c] [NeZero h] [NeZero w]
    (runtime : Runtime.Config)
    (loadBatches : IO ((count : Nat) × Tensor Float ((output c h w).prependDim count)))
    (config : Options) :
    IO Training.Curve := do
  let schedule ← CLI.orThrow exeName <|
    diffusion.Schedule.linear config.schedule.T
      config.schedule.betaStart config.schedule.betaEnd
  Training.run runtime loadBatches config schedule

namespace Options

/--
Parse diffusion-specific training flags after runtime/device flags and dataset flags.

The shared parser handles `--steps`, `--log`, and `--cuda-mem-watch`; this parser handles diffusion
schedule parameters, model width, and optional PPM artifact paths.
-/
def parse (args : List String) :
    Except String (Options × List String) := do
  let (train, rest) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := 50) (defaultLearningRate := 1e-3)
  let (hiddenChannels, rest) ←
    CLI.takePositiveNatFlag rest exeName "hidden-c" (default := 16)
  let (schedule, rest) ← Support.DiffusionScheduleFlags.parse rest
  let (artifacts, rest) ← Support.ImageArtifactFlags.parse rest
  pure ({ training := train,
          schedule,
          artifacts,
          hiddenChannels },
        rest)

/-- Dataset/source note fields shared by the CIFAR-10 and ImageNet64 branches. -/
def sourceNotes
    (datasetName : String)
    (data : Support.NpyDataFlags) : Array String :=
  Support.NpyDataFlags.trainLogNotes data datasetName

/-- TrainLog note fields shared by all diffusion dataset branches. -/
def logNotes
    (config : Options)
    (dataset : String)
    (runtime : Runtime.Config)
    (sourceNotes : Array String := #[]) : Array String :=
  sourceNotes ++
    #[s!"dataset={dataset}",
      Support.deviceNote runtime,
      s!"lr={config.training.learningRate}",
      s!"hiddenChannels={config.hiddenChannels}"] ++
    Support.DiffusionScheduleFlags.trainLogNotes config.schedule ++
    Support.ImageArtifactFlags.trainLogNotes config.artifacts

end Options

/-- Write the diffusion loss curve plus dataset, schedule, model, and artifact metadata. -/
def writeLog (log : Training.LogDestination) (dataset : String)
    (sourceNotes : Array String) (config : Options) (runtime : Runtime.Config)
    (curve : Training.Curve) : IO Unit :=
  Training.Curve.writeLog curve log "Diffusion training" "loss"
    (notes := config.logNotes dataset runtime sourceNotes)

/--
Run one typed diffusion dataset branch.

The CIFAR-10 and ImageNet64 commands differ in their shape-level loader and default `.npy` paths,
but after parsing those inputs they follow the same command flow: parse training flags, reject
unused args, require a positive hidden-channel count, train the epsilon predictor, then write the
same curve log.
-/
def runDataset {c h w : Nat} [NeZero c] [NeZero h] [NeZero w]
    (runtime : Runtime.Config) (args : List String)
    (datasetName : String)
    (parseData : List String → Except String (Support.NpyDataFlags × List String))
    (loadBatches : System.FilePath → System.FilePath → Nat → Nat →
      IO ((count : Nat) × Tensor Float ((output c h w).prependDim count))) : IO Unit := do
  let (data, args) ← CLI.orThrow exeName <| parseData args
  let data := { data with seed := runtime.seed }
  let (config, rest) ← CLI.orThrow exeName <| Options.parse args
  CLI.requireNoArgs exeName rest
  let sourceNotes := Options.sourceNotes datasetName data
  let load := loadBatches data.xPath data.yPath data.nRows data.seed
  let curve ← train runtime load config
  writeLog
    config.training.logDestination datasetName sourceNotes config runtime curve

/-- Run the ImageNet64 branch with shape-specialized model construction. -/
def runImageNet64 (runtime : Runtime.Config) (args : List String) : IO Unit :=
  runDataset
    (c := RealData.imagenet64Channels)
    (h := RealData.imagenet64Height)
    (w := RealData.imagenet64Width)
    runtime args "imagenet64"
    RealData.NpyDatasets.parseImageNet64
    loadImageNet64

/-- Run the CIFAR-10 branch with shape-specialized model construction. -/
def runCifar10 (runtime : Runtime.Config) (args : List String) : IO Unit :=
  runDataset
    (c := RealData.cifarChannels)
    (h := cifarCropHeight)
    (w := cifarCropWidth)
    runtime args "cifar10"
    RealData.NpyDatasets.parseCifar
    loadCifar

/--
Executable entrypoint for diffusion training.

The runtime parser selects CPU/CUDA and eager/typed-graph settings first; the remaining arguments
select the dataset branch and diffusion training configuration.
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Support.bannerWithDevice exeName "diffusion trainer"
      usage? := some <| TrainCommand.optimizerUsage exeName #[
        "  --dataset cifar10|imagenet64",
        "  --x PATH           image NPY file",
        "  --y PATH           label NPY file",
        "  --n-total N        images to load",
        "  --hidden-c N       denoiser channel width",
        "  --T N              diffusion timesteps",
        "  --beta-start X     initial noise variance",
        "  --beta-end X       final noise variance",
        "  --sample-ppm PATH --reference-ppm PATH",
        "  --noisy-ppm PATH --reconstruct-ppm PATH",
        "  --reconstruct-step N"
      ]
      printSuccess := true })
    exeName args
    (.native fun runtime rest => do
      let (choice, rest) ← CLI.orThrow exeName <| Support.ImageDatasetChoice.parse rest
      match choice with
      | .imagenet64 => runImageNet64 runtime rest
      | .cifar10 => runCifar10 runtime rest)

end NN.Examples.Models.Generative.Diffusion
