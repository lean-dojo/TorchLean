/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Models.RealData
public import TorchLean.NN.Data.Image
public import TorchLean.NN.Models.Diffusion.NoisePredictor
public import TorchLeanExamples.Models.Training

/-!
# Diffusion Training Example

The `torchlean diffusion` command trains a diffusion epsilon predictor on real images and can
write DDIM samples and reconstructions as PPM images. It has two data modes:

- `--dataset imagenet64` (default): user-provided ImageNet/Imagenette/Tiny-ImageNet-style images
  converted to `(N,3,64,64)` `.npy` tensors.
- `--dataset cifar10`: prepared CIFAR-10 `(N,3,32,32)` arrays, cropped to `4×4` pixels.

Both modes share one entrypoint, and each has its own typed branch because Lean tracks the image
height and width in the tensor type.

## Why unconditional samples are still modest

The default epsilon predictor is a small residual CNN that keeps the input resolution and reads the
timestep from one broadcast channel. It is enough to exercise real image loading, CUDA training,
logging, reconstruction diagnostics and DDIM sampling from Lean. Good unconditional samples need
more machinery: a full U-Net with multiscale skips, richer timestep embeddings, EMA, longer
training, more timesteps, and runtime support that avoids eager-autograd buffer blow-up for wider
models.

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
scripts/lake.sh -Kcuda=true build
CUDA_VISIBLE_DEVICES=0 scripts/lake.sh -Kcuda=true exe torchlean diffusion --device cuda \
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
scripts/lake.sh -Kcuda=true exe torchlean diffusion --device cuda --dataset cifar10 --n-total 1 \
  --steps 1 --hidden-c 2 --T 2
```
-/

@[expose] public section

open TorchLean.Trainer.Objective

namespace TorchLean.Examples.Diffusion

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "diffusion"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Training.artifactPath "diffusion"

/-- Static minibatch size used by both CIFAR-10 and ImageNet64 typed branches. -/
def batchSize : Nat := 1

/-- Cropped CIFAR height for the compact runnable diffusion example. -/
def cifarCropHeight : Nat := 4

/-- Cropped CIFAR width for the compact runnable diffusion example. -/
def cifarCropWidth : Nat := 4

local instance : NeZero cifarCropHeight := ⟨by decide⟩
local instance : NeZero cifarCropWidth := ⟨by decide⟩

/-- Shape of a clean image batch $x_0$: NCHW with the fixed `batchSize`. -/
abbrev output (c h w : Nat) : Shape :=
  [batchSize, c, h, w]

/-- Epsilon-model input dimensions: image channels plus one broadcast timestep channel. -/
abbrev input (c h w : Nat) : Shape :=
  [batchSize, c + 1, h, w]

/--
Epsilon-predictor configuration for images with `c` channels and size `h × w`.

A kernel radius of one on both axes gives every convolution a `3×3` same-padding kernel.
-/
def config (c h w hiddenChannels : Nat) :
    nn.NoisePredictor.Config 2 :=
  { dataChannels := c
    spatial := [h, w]
    hiddenChannels := hiddenChannels
    kernelRadius := [1, 1] }

/--
Residual epsilon predictor for one typed image shape: a stem convolution, two residual blocks and
an output convolution, with ReLU between them.

Every convolution is `3×3` with same padding, so the predictor maps `c + 1` input channels to `c`
noise channels at the input resolution.
-/
def model (c h w hiddenChannels : Nat) :
    nn.Builder (nn.Sequential (input c h w) (output c h w)) :=
  by
    let built :=
      nn.NoisePredictor.residual
        (config c h w hiddenChannels) (batchShape := [batchSize])
    rw [nn.NoisePredictor.Config.inputShape,
      nn.NoisePredictor.Config.outputShape] at built
    simpa [input, output, config] using built

/--
Crop one typed CIFAR minibatch to `cifarCropHeight × cifarCropWidth` and map it into diffusion
space.

The loader returns pixel values in $[0,1]$ and diffusion training works in $[-1,1]$, so the crop
goes through `Diffusion.unitToSignedUnit`, the map $x \mapsto 2x - 1$.
-/
def cifarBatch
    (batchSample : Sample.Batch Float batchSize RealData.cifarImage RealData.cifarTarget) :
    Except String (Tensor Float
      (output RealData.cifarChannels cifarCropHeight cifarCropWidth)) := do
  let cropped ←
    RealData.cropCifarBatch batchSize cifarCropHeight cifarCropWidth batchSample
  let unitImage :
      Tensor Float (output RealData.cifarChannels cifarCropHeight cifarCropWidth) := by
    simpa [output] using cropped.input
  pure (Diffusion.unitToSignedUnit unitImage)

/--
Convert one typed ImageNet64 minibatch into diffusion-space clean images.

The images keep their full `64×64` size, and only the value range changes, from $[0,1]$ to
$[-1,1]$.
-/
def imageNet64Batch
    (batchSample :
      Sample.Batch Float batchSize RealData.imageNet64Image RealData.imageNet64Target) :
    Tensor Float (output RealData.imageNet64Channels RealData.imageNet64Height
      RealData.imageNet64Width) := by
  let unitImage : Tensor Float (output RealData.imageNet64Channels RealData.imageNet64Height
      RealData.imageNet64Width) := by
    simpa [output, RealData.imageNet64Image] using
      batchSample.input
  exact Diffusion.unitToSignedUnit unitImage

/--
Load one shuffled epoch of CIFAR-10 minibatches as a tensor of clean diffusion batches.

Both `.npy` files must exist, an incomplete final batch is dropped, and `cifarBatch` crops each
batch and maps it into $[-1,1]$.
-/
def loadCifar (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO ((count : Nat) × Tensor Float
      (Shape.dim count (output RealData.cifarChannels cifarCropHeight cifarCropWidth))) := do
  let batches ← RealData.loadCifarBatches exeName batchSize nRows seed xPath yPath
  let images ← Tensor.stackLeadingM fun index : Fin batches.size =>
    CLI.orThrow exeName (cifarBatch batches[index])
  pure ⟨batches.size, images⟩

/--
Load one shuffled epoch of ImageNet64-style minibatches as a tensor of clean diffusion batches.

`scripts/datasets/torchlean_data_convert.py` turns ImageNet, Imagenette or Tiny-ImageNet folders
into `.npy` arrays ahead of time, and this loader reads only those prepared arrays.
-/
def loadImageNet64 (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO ((count : Nat) × Tensor Float
      (Shape.dim count (output RealData.imageNet64Channels RealData.imageNet64Height
        RealData.imageNet64Width))) := do
  let batches ← RealData.loadImageNet64Batches exeName batchSize nRows seed xPath yPath
  pure ⟨batches.size, Tensor.dim fun index => imageNet64Batch batches[index]⟩

/--
Turn a predictor on time-conditioned inputs into the `(index, sample)` function the DDIM sampler
calls: the sample gets one extra channel holding the time of `index`.
-/
def conditionedPredictor {c h w T : Nat}
    (predict : Tensor Float (input c h w) → IO (Tensor Float (output c h w)))
    (index : Fin T) (sample : Tensor Float (output c h w)) :
    IO (Tensor Float (output c h w)) :=
  let time : Float := Diffusion.ImageDDIM.timeOfIndex index
  let conditioned : Tensor Float (input c h w) := by
    simpa [input, output] using
      Diffusion.appendTimeChannel [batchSize] ([h, w] : Tensor Nat [2])
        (by simpa [output] using sample) time
  predict conditioned

/-- Training, schedule, and image-output options for the diffusion example. -/
structure Options where
  /-- Optimizer, step, batching, and logging controls. -/
  training : CLI.Training.OptimizerOptions
  /-- Diffusion timestep and beta schedule. -/
  schedule : DiffusionSchedule.Options
  /-- Optional generated and reconstructed image paths. -/
  artifacts : Image.OutputOptions
  /-- Hidden channel width of the epsilon predictor. -/
  hiddenChannels : Nat
deriving Repr

/--
Shared training loop for the CIFAR-10 and ImageNet64 branches.

Each step takes the next minibatch in turn, draws seeded noise and cycles the timestep through the
schedule, one timestep further on each pass over the minibatches, so that no minibatch keeps one
timestep. Adam minimizes the mean squared error of the predicted noise. After training the loop
can write four PPM images:

- `reference-ppm`: clean evaluation image,
- `noisy-ppm`: clean image after forward diffusion to `reconstruct-step` (default `T / 4`),
- `reconstruct-ppm`: DDIM denoising from that timestep,
- `sample-ppm`: unconditional DDIM sample from Gaussian noise.
-/
def trainLoop
    {c h w : Nat} [NeZero c] [NeZero h] [NeZero w]
    (runtime : Session.Config)
    (loadBatches : IO ((count : Nat) × Tensor Float (Shape.dim count (output c h w))))
    (config : Options)
    (schedule : Diffusion.Schedule config.schedule.timesteps) :
    IO Training.Curve := do
  let ⟨count, batches⟩ ← loadBatches
  if empty : count = 0 then
    throw (IO.userError s!"{exeName}: no training minibatches available")
  else
  let batchAt (step : Nat) :=
    batches.unstack ⟨step % count, Nat.mod_lt _ (Nat.pos_of_ne_zero empty)⟩
  let evalX0 := batchAt 0
  let alphaBars := schedule.alphaBars
  let evalStep := config.schedule.timesteps / 2
  let evalSample : Sample.Supervised Float (input c h w) (output c h w) := by
    simpa [input, output] using
      Diffusion.sample [batchSize] ([h, w] : Tensor Nat [2])
        schedule (by simpa [output] using evalX0) (seed := runtime.seed)
        (step := evalStep)
  let trainer :=
    Trainer.create (model c h w config.hiddenChannels) <|
      { runtime with
          optimizer := optim.adam { lr := config.training.lr }
          objective := mse }
  trainer.printSummary
  let curveEvery : Nat := Nat.max 1 (config.training.steps / 50)
  let trained ← trainer.trainStream runtime
    (fun step =>
      let x0 := batchAt step
      show Sample.Supervised Float (input c h w) (output c h w) from by
        simpa [input, output] using
          Diffusion.sample [batchSize] ([h, w] : Tensor Nat [2])
            schedule (by simpa [output] using x0)
            (step := step % count + step / count + 1)
            (noise := Diffusion.normalNoise runtime.seed (step + 1)))
    evalSample
    (config.training.trainOptions (enableLog := false))
    (curveEvery := curveEvery)
  let curve := trained.curve
  trained.printSummary
  match config.artifacts.referencePPM? with
  | none => pure ()
  | some path => Data.Image.writeFirstPPM path (evalX0.map fun x => (x + 1.0) / 2.0)
  match config.artifacts.samplePPM? with
  | none => pure ()
  | some path => do
      let x_T := Diffusion.normalNoise (shape := output c h w)
        (seed := runtime.seed) (step := 999)
      let x_t ← Diffusion.ImageDDIM.sample 1e-12 alphaBars
        (conditionedPredictor trained.predict) x_T
      Data.Image.writeFirstPPM path (x_t.map fun x => (x + 1.0) / 2.0)
  match config.artifacts.noisyPPM?, config.artifacts.reconstructPPM? with
  | none, none => pure ()
  | noisyPath?, reconstructionPath? => do
      let tIdxNat :=
        Nat.min
          (config.artifacts.reconstructStep?.getD (config.schedule.timesteps / 4))
          (config.schedule.timesteps - 1)
      let tIdx := schedule.index tIdxNat
      let eps := Diffusion.normalNoise (shape := output c h w)
        (seed := runtime.seed) (step := 1001)
      let noisyImage : Tensor Float (output c h w) :=
        Diffusion.qSampleAt alphaBars[tIdx] evalX0 eps
      match noisyPath? with
      | none => pure ()
      | some noisyPath =>
          Data.Image.writeFirstPPM noisyPath (noisyImage.map fun x => (x + 1.0) / 2.0)
      match reconstructionPath? with
      | none => pure ()
      | some path => do
          let x_t ← Diffusion.ImageDDIM.sample 1e-12 alphaBars
            (conditionedPredictor trained.predict) noisyImage (start := some tIdx)
          Data.Image.writeFirstPPM path (x_t.map fun x => (x + 1.0) / 2.0)
  pure curve

/--
Build the linear beta schedule from the options and run `trainLoop`. A zero timestep count, or
a beta endpoint that is not finite or lies outside $[0,1)$, is an error.
-/
def train {c h w : Nat} [NeZero c] [NeZero h] [NeZero w]
    (runtime : Session.Config)
    (loadBatches : IO ((count : Nat) × Tensor Float (Shape.dim count (output c h w))))
    (config : Options) :
    IO Training.Curve := do
  let schedule ← CLI.orThrow exeName <|
    Diffusion.Schedule.linear config.schedule.timesteps
      config.schedule.betaStart config.schedule.betaEnd
  trainLoop runtime loadBatches config schedule

namespace Options

/--
Parse the training flags that follow the runtime and dataset flags.

The shared optimizer parser takes `--steps`, `--batch-size`, `--log`, `--cuda-mem-watch` and
`--lr` (defaults 50 steps and `1e-3`), and this parser adds the schedule flags, `--hidden-c`
(default 16) and the optional PPM artifact paths.
-/
def parse (args : List String) :
    Except String (Options × List String) := do
  let (train, rest) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogPath
      (defaultSteps := 50) (defaultLr := 1e-3)
  let (hiddenChannels, rest) ←
    CLI.takePositiveNatFlag rest exeName "hidden-c" (default := 16)
  let (schedule, rest) ← DiffusionSchedule.Options.parse rest
  let (artifacts, rest) ← Image.OutputOptions.parse rest
  pure ({ training := train,
          schedule,
          artifacts,
          hiddenChannels },
        rest)

/-- TrainLog note fields shared by all diffusion dataset branches. -/
def logNotes
    (config : Options)
    (dataset : String)
    (runtime : Session.Config)
    (sourceNotes : Array String := #[]) : Array String :=
  sourceNotes ++
    #[s!"dataset={dataset}",
      Training.deviceNote runtime,
      s!"lr={config.training.lr}",
      s!"hiddenChannels={config.hiddenChannels}"] ++
    DiffusionSchedule.Options.logNotes config.schedule ++
    Image.OutputOptions.logNotes config.artifacts

end Options

/--
Run one typed dataset branch of the diffusion command.

The CIFAR-10 and ImageNet64 branches differ in their typed loader and default `.npy` paths. Both
parse the training flags (the hidden width must be positive), reject leftover arguments, train the
epsilon predictor and write the loss curve to the training log.
-/
def runDataset {c h w : Nat} [NeZero c] [NeZero h] [NeZero w]
    (runtime : Session.Config) (args : List String)
    (datasetName : String)
    (parseData : List String → Except String (NPY.Options × List String))
    (loadBatches : System.FilePath → System.FilePath → Nat → Nat →
      IO ((count : Nat) × Tensor Float (Shape.dim count (output c h w)))) : IO Unit := do
  let (data, args) ← CLI.orThrow exeName <| parseData args
  let (config, rest) ← CLI.orThrow exeName <| Options.parse args
  CLI.requireNoArgs exeName rest
  let sourceNotes := NPY.Options.logNotes data datasetName
  let load := loadBatches data.xPath data.yPath data.nRows runtime.seed
  let curve ← train runtime load config
  Training.Curve.writeLog curve config.training.logDestination "Diffusion training" "loss"
    (notes := config.logNotes datasetName runtime sourceNotes)

/--
Executable entrypoint for diffusion training.

The runtime parser picks the CPU or CUDA backend first, and `--dataset` then selects the branch
that parses the remaining flags.
-/
def main (args : List String) : IO UInt32 := do
  Module.Command.run
    (config := {
      banner? := some <| Training.banner exeName "diffusion trainer"
      usage? := some <| CLI.Training.helpText exeName #[
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
    (fun runtime rest => do
      let (choice, rest) ← CLI.orThrow exeName <| Image.Dataset.parse rest
      match choice with
      | .imageNet64 =>
          runDataset runtime rest "imagenet64" RealData.parseImageNet64 loadImageNet64
      | .cifar10 =>
          runDataset runtime rest "cifar10" RealData.parseCifar loadCifar)

end TorchLean.Examples.Diffusion
