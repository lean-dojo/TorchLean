---
title: Diffusion Walkthrough
---

In this example, we'll add noise to images and train a small network to predict the noise we
added. Then we'll use its predictions to reconstruct or generate an image. We'll work in Lean
for the model, training code, sampler, and mathematical definitions.

<div class="media-slab">
  <img src="{{ '/assets/media/examples/diffusion_imagenette64_real_vs_generated_plot.png' | relative_url }}" alt="Real Imagenette image, diffusion-generated sample, and diffusion reconstruction"/>
</div>

The saved comparison above uses Imagenette-derived 64×64 images, enlarged for display.
Its generated sample is still noisy; it is not the output of the one-step CIFAR-10 run below.

## Run It First

Let's run a small version first, then go through how it works. We'll prepare the CIFAR-10 arrays
and train a tiny model for one step. The CUDA commands require
a compatible LibTorch SDK and GPU, as described in the
[installation guide]({{ '/installation/' | relative_url }}):

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

TorchLean does not decode JPEG/PNG inside Lean. Image decoding and resizing happen before Lean; the
Lean side receives fixed-layout `.npy` tensors and checks their shapes.

Once the arrays exist, Lean loads tensors with axes ordered as
`[batch, channel, height, width]`, checks their shapes and class range, and batches them. This is a
layout convention for the dataset, not a separate image tensor type.

The two dataset modes in `NN.Examples.Models.Generative.Diffusion` are:

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
pure (diffusion.unitToSignedUnit unitImage)
```

`diffusion.unitToSignedUnit` keeps the tensor shape and only rescales values from $[0,1]$ to
$[-1,1]$.

## The diffusion equations

The definitions in `NN.Spec.Generative.Diffusion.*` describe denoising diffusion probabilistic
models (DDPMs). We add noise to an image at timestep
$t$, train a model to predict the noise that was added, then run reverse steps that use the model’s
noise prediction to move back toward a clean image.

The following excerpts use the `Generative.Diffusion` namespace, with `open Spec TorchLean`.
The arithmetic definitions assume `[Storage α] [Context α]` and take `T` and `s` as implicit
parameters. `EpsModel` describes a denoiser that takes a noisy tensor and a timestep and predicts
the noise:

```lean
structure EpsModel (α : Type) (s : Shape) [TorchLean.Storage α] where
  eps : Tensor α s → α → Tensor α s
```

We add noise with the standard DDPM formula, expressed here as a total tensor definition:

```lean
def qSample (sched : VPSchedule α T)
    (x0 : Tensor α s) (t : Fin (T + 1)) (eps : Tensor α s) :
    Tensor α s :=
  let αbar : α := sched.alphaBar t
  let c0 : α := sqrtNonneg αbar
  let c1 : α := sqrtNonneg (1 - αbar)
  Tensor.scaleSpec x0 c0 + Tensor.scaleSpec eps c1
```

To measure how well our denoiser predicts the added noise, we use mean squared error:

```lean
def epsPredLoss (sched : VPSchedule α T) (model : EpsModel α s)
    (x0 : Tensor α s) (t : Fin (T + 1)) (eps : Tensor α s) : α :=
  let x_t := qSample sched x0 t eps
  let tScalar : α := VPSchedule.timeOfIndex (α := α) (T := T) t
  let epsHat := model.eps x_t tScalar
  Spec.mseSpec epsHat eps
```

Then the surrounding spec modules define:

- a discrete VP schedule `VPSchedule` with $\beta_t$, $\alpha_t=1-\beta_t$, and cumulative products
  $\bar{\alpha}_t$;
- two reverse samplers:

  - DDPM: a stochastic reverse step with explicit per-step noise inputs,
  - DDIM ($\eta=0$): a deterministic reverse step that reuses the same denoiser but drops the noise.

`ddimStepSystem` expresses the DDIM sampler as a discrete dynamical system, so its steps can be
used with the corresponding dynamical-system definitions and theorems.

## The Runtime Layer: The Model That Runs

The runnable diffusion command does not work directly with `EpsModel`. It instantiates a concrete
neural network and then uses the public data API to build training samples.

For our model, we'll make two choices:

1. The epsilon predictor is a residual CNN that preserves resolution
   (`nn.models.Diffusion.NoisePredictor.residual`).
2. Time is fed to the model as an extra channel: when the data has $c$ channels, the input has
   $c+1$ channels. The last channel is the normalized timestep broadcast across spatial positions.

To append the time channel, provide the
leading dimensions and spatial extents; the helper returns a tensor with one additional channel:

```lean
let modelInput :=
  diffusion.appendTimeChannel [batchSize] ([h, w] : Tensor Nat [2]) x_t tNorm
```

We'll set the image size and hidden channel count below. The constructor builds
the convolutions and residual blocks and initializes their parameters from a seed:

```lean
def config (c h w hiddenChannels : Nat) :
    nn.models.Diffusion.NoisePredictor.Config 2 :=
  { dataChannels := c
    spatial := [h, w]
    hiddenChannels := hiddenChannels
    kernelRadius := [1, 1] }

def model (c h w hiddenChannels : Nat) :
    nn.Builder (nn.Sequential (input c h w) (output c h w)) := by
  let built := nn.models.Diffusion.NoisePredictor.residual
    (config c h w hiddenChannels) (batchShape := [batchSize])
  rw [nn.models.Diffusion.NoisePredictor.Config.inputShape,
    nn.models.Diffusion.NoisePredictor.Config.outputShape] at built
  simpa [input, output, config, Shape.ofList, Shape.concat] using built
```

The `2` is the spatial rank, and a kernel radius of one means every convolution is a `3 x 3`
same-padding kernel. The contract is dimension-general: the input has arbitrary leading axes, one
channel axis, and any number of spatial axes. The runnable image example instantiates this with
batch, channel, height, and width axes. Its output predicts noise with the original data shape.
The `rw`/`simpa` step makes the constructor's computed shapes
line up with the example's `input` and `output` abbreviations.

## Training: What Gets Optimized

We're training the model to predict $\varepsilon$, the noise added to the image. At each step, we:

1. pick a real image batch $x_0$,
2. pick a timestep $t$,
3. sample noise $\varepsilon$,
4. pair `appendTimeChannel x_t tNorm` with the target noise $\varepsilon$,
5. take an optimizer step on MSE.

TorchLean makes the supervised pair explicit as a `Sample.Supervised` value. The operation that
constructs $x_t$ from $x_0$ and $\varepsilon$ is
`TorchLean.diffusion.sample`; it is the runtime version of the same formula used by
`qSample` in the spec layer.

The helper performs the schedule lookup, noising formula, time-channel append, and supervised-pair
construction. We can choose a seed or pass the noise tensor directly:

```lean
let sample :=
  diffusion.sample [batchSize] ([h, w] : Tensor Nat [2]) schedule x0
    (seed := runtime.seed) (step := step)
let sampleFromNoise :=
  diffusion.sample [batchSize] ([h, w] : Tensor Nat [2]) schedule x0 step (noise := eps)
```

The schedule length is part of the type. A `Schedule T` for `T` diffusion steps cannot be used
with a different timestep count, and the runnable command rejects `T = 0` before constructing its
first sample. The `(seed, step)` pair determines the sampled noise; `step` also selects the
schedule coefficient.

## Sampling: DDIM Replay In Lean

Now we can use the denoiser to sample an image. We'll use deterministic DDIM, so the starting
noise and denoiser determine the reverse path.

The reverse update used by the runnable example is `TorchLean.diffusion.ddimPrev`. Its default
step is:

- estimate $\hat{x}_0$ from $x_t$ and $\hat{\varepsilon}$,
- clamp it to $[-1,1]$,
- recombine it using the previous schedule coefficients.

The public helper takes the current sample, predicted noise, and adjacent schedule coefficients:

```lean
let previous := diffusion.ddimPrev abPrev ab x_t epsHat
```

The optional `postprocess` argument changes the operation applied to the estimated clean image
before remixing; it must preserve the tensor shape. For an unclipped reconstruction:

```lean
let previous := diffusion.ddimPrev abPrev ab x_t epsHat
  (postprocess := fun reconstructed => reconstructed)
  (denominatorFloor := 1e-12)
```

The denominator uses $\sqrt{\bar\alpha_t}$ when that value is strictly greater than
`denominatorFloor`, and the floor otherwise. Its default is `1e-12`. `diffusion.ddim`
passes both choices through each step. A different postprocessor or floor changes
the sampler's function, even when the model and schedule are unchanged.

`Generative.Diffusion.ImageDDIM.ddimPrev_eq_stepFromEps` relates this API update to
the image-DDIM specification with the same postprocessor and floor. It preserves the written
floating-point expression order; it is not a claim about denoiser accuracy or sample quality.

The sampler also produces the “three pictures” view:

- a reference image (the real $x_0$),
- a noisy image at a chosen timestep,
- a reconstruction from DDIM reverse steps starting at that timestep.

## What To Look At After A Run

Once the run finishes, let's look at the saved images as well as the loss log. A lower noise-prediction loss does not by itself
tell you whether the generated images look better. Keep both outputs when comparing sampling
settings, noise schedules, or model widths.

For interactive inspection, open `NN.Examples.Models.Generative.Diffusion` in VS Code with the Lean
Infoview enabled. The widgets can display tensor summaries, graph and shape views, and saved JSON
logs next to the source.

Source entry points:

- [`NN.Examples.Models.Generative.Diffusion`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Diffusion.lean)
- [`NN.Examples.Models.Generative`](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/Models/Generative)
- [Generative Models]({{ '/blueprint/Examples-and-Applications/Generative-Models/' | relative_url }})
