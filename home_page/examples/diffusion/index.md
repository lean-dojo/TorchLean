---
title: Diffusion Walkthrough
---

In this example, we'll add noise to images and train a small network to predict the noise we
added. Then we'll use its predictions to reconstruct or generate an image. We'll work in Lean
for the model, training code, sampler, and mathematical definitions.

<div class="media-slab">
  <img src="{{ '/assets/media/examples/diffusion_imagenette64_real_vs_generated_plot.png' | relative_url }}" alt="Real Imagenette image, diffusion-generated sample, and diffusion reconstruction"/>
</div>

The saved comparison above uses Imagenette-derived 64×64 images, enlarged for display. Its
generated sample is still noisy, and it isn't the output of the one-step CIFAR-10 run below.

## Run It First

Let's run a small version first, then go through how it works. We'll prepare the CIFAR-10 arrays
and train a tiny model for one step. For the CUDA commands you'll need a compatible LibTorch SDK
and GPU (see the [installation guide]({{ '/installation/' | relative_url }})):

```bash
python3 scripts/datasets/download_example_data.py --cifar10
scripts/lake.sh -Kcuda=true exe torchlean diffusion --device cuda \
  --dataset cifar10 --n-total 1 --steps 1 --hidden-c 1 --T 2
```

For a longer CUDA run, use 800 images and 50 training steps. This command saves the loss history
as JSON and a generated image as a PPM file:

```bash
python3 scripts/datasets/download_example_data.py --cifar10 --cifar10-limit-train 800

scripts/lake.sh -Kcuda=true exe torchlean diffusion --device cuda \
  --dataset cifar10 --n-total 800 --steps 50 --hidden-c 8 \
  --log data/examples/diffusion_trainlog.json \
  --sample-ppm data/examples/diffusion_sample.ppm
```

## Data: Where Images Come From

TorchLean doesn't decode JPEG/PNG inside Lean. Image decoding and resizing happen before Lean, and
the Lean side receives fixed-layout `.npy` tensors and checks their shapes.

Once the arrays exist, Lean loads tensors with axes ordered as
`[batch, channel, height, width]`, checks their shapes and class range, and batches them. This is a
layout convention for the dataset, not a separate image tensor type.

`TorchLeanExamples.Models.Generative.Diffusion` has two dataset modes:

- CIFAR-10 as $(N,3,32,32)$ arrays, cropped to $4\times4$ patches by this compact example.
- ImageNet-style folders converted to $(N,3,64,64)$ arrays (Imagenette, Tiny-ImageNet, or any
  folder with class subdirectories).

To prepare a batch for training, we:

1. Build a labeled source over `.npy` files.
2. Load it into a dataset.
3. Create a shuffled batch loader.
4. Map pixel values from $[0,1]$ into the diffusion range $[-1,1]$.

For the last step, we rescale the cropped minibatch with one call:

```lean
pure (Diffusion.unitToSignedUnit unitImage)
```

`Diffusion.unitToSignedUnit` keeps the tensor shape and only rescales values from $[0,1]$ to
$[-1,1]$.

## The diffusion equations

The definitions in `TorchLean.NN.Models.Diffusion` and its submodules describe denoising diffusion
probabilistic models (DDPMs). We add noise to an image at timestep $t$, train a model to predict
the noise that was added, then run reverse steps that use the model’s noise prediction to move back
toward a clean image.

In the excerpts that follow, we work in the `TorchLean.Diffusion` namespace with
`open TorchLean.Tensor`, and the arithmetic definitions assume `[Storage α] [Context α]` and take
`T` and `s` as implicit parameters. `EpsModel` describes a denoiser that takes a noisy tensor and a
timestep and predicts the noise:

```lean
structure EpsModel (α : Type) (s : Shape) [TorchLean.Storage α] where
  eps : Tensor α s → α → Tensor α s
```

We add noise with the standard DDPM formula, expressed here as a total tensor definition at one
cumulative coefficient and then at a schedule timestep:

```lean
def qSampleAt (alphaBar : α) (x0 eps : Tensor α s) : Tensor α s :=
  Tensor.scale x0 (sqrtNonneg alphaBar) + Tensor.scale eps (sqrtNonneg (1 - alphaBar))

def qSample (sched : VPSchedule α T)
    (x0 : Tensor α s) (t : Fin (T + 1)) (eps : Tensor α s) :
    Tensor α s :=
  qSampleAt (sched.alphaBar t) x0 eps
```

To measure how well our denoiser predicts the added noise, we use mean squared error:

```lean
def epsPredLoss (sched : VPSchedule α T) (model : EpsModel α s)
    (x0 : Tensor α s) (t : Fin (T + 1)) (eps : Tensor α s) : α :=
  let x_t := qSample sched x0 t eps
  let tScalar : α := VPSchedule.timeOfIndex (α := α) (T := T) t
  let epsHat := model.eps x_t tScalar
  Tensor.meanSquaredError epsHat eps
```

Then the surrounding spec modules define:

- a discrete VP schedule `VPSchedule` with $\beta_t$, $\alpha_t=1-\beta_t$, and cumulative products
  $\bar{\alpha}_t$,
- two reverse samplers:

  - DDPM: a stochastic reverse step with per-step noise inputs,
  - DDIM ($\eta=0$): `ImageDDIM.sample`, a deterministic reverse loop that calls the noise
    predictor once per step and adds no noise.

The sampler theory packages one DDIM step, with a pure predictor, as the dynamical system
`ImageDDIM.stepSystem`, so the dynamical-system definitions and theorems apply to it.

## The Runtime Layer: The Model That Runs

Instead of working directly with `EpsModel`, the runnable diffusion command instantiates a concrete
neural network and then uses the public data API to build training samples.

For our model, we'll make two choices:

1. For the epsilon predictor, we use a residual CNN that preserves resolution
   (`nn.NoisePredictor.residual`).
2. We feed time to the model as an extra channel: when the data has $c$ channels, the input has
   $c+1$ channels, and the last one is the normalized timestep broadcast across spatial positions.

To append the time channel, provide the leading dimensions and spatial extents, and the helper
returns a tensor with one additional channel:

```lean
let modelInput :=
  Diffusion.appendTimeChannel [batchSize] ([h, w] : Tensor Nat [2]) x_t tNorm
```

We'll set the image size and hidden channel count below, then let the constructor build the
convolutions and residual blocks and initialize their parameters from a seed:

```lean
def config (c h w hiddenChannels : Nat) :
    nn.NoisePredictor.Config 2 :=
  { dataChannels := c
    spatial := [h, w]
    hiddenChannels := hiddenChannels
    kernelRadius := [1, 1] }

def model (c h w hiddenChannels : Nat) :
    nn.Builder (nn.Sequential (input c h w) (output c h w)) := by
  let built := nn.NoisePredictor.residual
    (config c h w hiddenChannels) (batchShape := [batchSize])
  rw [nn.NoisePredictor.Config.inputShape,
    nn.NoisePredictor.Config.outputShape] at built
  simpa [input, output, config] using built
```

Here `2` is the spatial rank, and a kernel radius of one means every convolution is a `3 x 3`
same-padding kernel. The contract is dimension-general: the input has arbitrary leading axes, one
channel axis and any number of spatial axes. Our runnable image example instantiates this with
batch, channel, height and width axes, and its output predicts noise with the original data shape.
The `rw`/`simpa` step makes the constructor's computed shapes line up with the example's `input`
and `output` abbreviations.

## Training: What Gets Optimized

We're training the model to predict $\varepsilon$, the noise added to the image. At each step, we:

1. pick a real image batch $x_0$,
2. pick a timestep $t$,
3. sample noise $\varepsilon$,
4. pair `appendTimeChannel x_t tNorm` with the target noise $\varepsilon$,
5. take an optimizer step on MSE.

TorchLean represents the supervised pair as a `Sample.Supervised` value.
`TorchLean.Diffusion.sample` constructs $x_t$ from $x_0$ and $\varepsilon$ by applying the spec's
`qSampleAt` to the stored coefficient of the selected timestep.

That helper performs the schedule lookup, noising formula, time-channel append and supervised-pair
construction. We can choose a seed or supply the noise tensor ourselves:

```lean
let sample :=
  Diffusion.sample [batchSize] ([h, w] : Tensor Nat [2]) schedule x0
    (seed := runtime.seed) (step := step)
let sampleFromNoise :=
  Diffusion.sample [batchSize] ([h, w] : Tensor Nat [2]) schedule x0 step (noise := eps)
```

Schedule length is part of the type, so a `Schedule T` for `T` diffusion steps can't be used with a
different timestep count. Before constructing its first sample, the runnable command rejects
`T = 0`. Each `(seed, step)` pair determines the sampled noise, and `step` also selects the
schedule coefficient.

## Sampling: DDIM Replay In Lean

Now we can use the denoiser to sample an image. We'll use deterministic DDIM, so the starting
noise and denoiser determine the reverse path.

For the reverse update, the runnable example uses `TorchLean.Diffusion.ImageDDIM.stepFromEps`. Its
default step is:

- estimate $\hat{x}_0$ from $x_t$ and $\hat{\varepsilon}$,
- clamp it to $[-1,1]$,
- recombine it using the previous schedule coefficients.

The step takes the denominator floor, the adjacent schedule coefficients, the current sample, and
the predicted noise:

```lean
let previous := ImageDDIM.stepFromEps 1e-12 abPrev ab x_t epsHat
```

With the optional `postprocess` argument, we can change the operation applied to the estimated
clean image before remixing (it must preserve the tensor shape). For an unclipped reconstruction:

```lean
let previous := ImageDDIM.stepFromEps 1e-12 abPrev ab x_t epsHat
  (postprocess := fun reconstructed => reconstructed)
```

The denominator uses $\sqrt{\bar\alpha_t}$ when that value is strictly greater than
`denominatorFloor`, and the floor otherwise. The runnable example passes `1e-12`.
`ImageDDIM.sample` passes both choices through each step. A different postprocessor or floor
changes the sampler's function, even when the model and schedule are unchanged.

The runnable sampler is the image-DDIM specification itself, run in `IO` with the trained network
as its predictor. Sharing the specification says nothing about denoiser accuracy or sample quality.

We also get the “three pictures” view from the sampler:

- a reference image (the real $x_0$),
- a noisy image at a chosen timestep,
- a reconstruction from DDIM reverse steps starting at that timestep.

## What To Look At After A Run

Once the run finishes, let's look at the saved images as well as the loss log. A lower noise-prediction loss does not by itself
tell you whether the generated images look better. Keep both outputs when comparing sampling
settings, noise schedules, or model widths.

For interactive inspection, open `TorchLeanExamples.Models.Generative.Diffusion` in VS Code with
the Lean Infoview enabled. Its widgets can display tensor summaries, graph and shape views, and
saved JSON logs next to the source.

Source entry points:

- [`TorchLeanExamples.Models.Generative.Diffusion`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/Models/Generative/Diffusion.lean)
- [`TorchLeanExamples/Models/Generative`](https://github.com/lean-dojo/TorchLean/tree/main/TorchLeanExamples/Models/Generative)
- [Generative Models]({{ '/blueprint/Examples-and-Applications/Generative-Models/' | relative_url }})
