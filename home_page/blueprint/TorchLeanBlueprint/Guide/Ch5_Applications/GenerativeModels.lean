import VersoManual
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.CLI.Training.Command
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.ResNet
import TorchLean.NN.Models.UNet
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.Generative
import TorchLean.NN.Models.FNO
import TorchLean.NN.Models.PPO
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.RL.PPO.ActorCritic
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
import TorchLean.NN.Data.Image
import TorchLean.NN.Models.Diffusion.Schedule
import TorchLean.NN.Models.Diffusion.ForwardProcess
import TorchLean.NN.Models.Diffusion.Loss
import TorchLean.NN.Models.Diffusion.ImageDDIM.Training
import TorchLean.NN.Models.VAE
import TorchLean.NN.Models.VQVAE
import TorchLean.NN.Models.VAE.Elbo
import TorchLean.NN.Models.VQVAE.Loss
import TorchLean.NN.Models.GAN.Loss
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean Backend.Device
open Diffusion
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Generative Models" =>
%%%
tag := "generative-models"
%%%

An exact noise prediction need not give an exact reconstruction. In the oracle-denoiser example
below, the prediction loss is zero, but the sampler run with a denominator floor of `0.5` returns
`0.467440` from an original value of `0.5`. Supplying the noise ourselves lets us isolate that
arithmetic before asking what a learned denoiser can do.

I spell out the inputs in the latent-model examples for the same reason: we can calculate a KL term
or a codebook loss by hand, then compare it with the definition and its printed value. When this
guide is built, its Lean blocks are elaborated, and their output blocks record Lean's evaluations
and type checks. Command-line and PyTorch transcripts are recorded examples from earlier runs.
Nothing in the guide build executes them, and they may differ with the data, version, or backend.

# Implementation Layers

Three layers appear in these examples. Apart from the commands and `TorchLean/NN/Latent.lean`, their
modules sit in `TorchLean/NN/Models`, and the table gives paths relative to it. The module, not the
directory, distinguishes the code a command runs from the definitions a theorem concerns:

:::table +header
*
  * Layer
  * Location
  * What it gives you
*
  * runnable API
  * `Generative.lean`, `Diffusion/NoisePredictor.lean`, and `TorchLeanExamples/Models/Generative`
  * runtime constructors and helpers for schedules, training samples, and denoiser architectures
*
  * executable spec
  * `Diffusion.lean`, `Diffusion/Schedule.lean`, `Diffusion/ForwardProcess.lean`,
    `Diffusion/ReverseDDPM.lean`, `Diffusion/ReverseDDIM.lean`, `Diffusion/PFODE.lean`,
    `Diffusion/ImageDDIM.lean`, `Diffusion/Loss.lean`, `VAE.lean`, `VQVAE.lean`, `GAN.lean`, and
    `TorchLean/NN/Latent.lean`
  * scalar-parametric model definitions, total and side-effect free
*
  * theory
  * `Diffusion/Samplers.lean`, `Diffusion/ImageDDIM/Training.lean`, `VAE/Elbo.lean`,
    `VQVAE/Loss.lean`, and `GAN/Loss.lean`
  * theorems, in Mathlib's vocabulary, about the spec-layer or real-valued objects
:::

Spec-layer functions are executable, and most evaluations below call them at `Float`. Relating
those values to a training run requires matching the definitions and scalar arithmetic, then
accounting for any implementation refinement.

A generative model is useful when the desired output is not a single label attached to an input. An
image can have many plausible completions, and many images can belong to the same class. These
examples separate two ways of representing that variation: diffusion moves between noise levels in
observation space, while latent models decode a smaller continuous or discrete representation. That
distinction determines what the network consumes, what its loss compares, and how we could use a
trained network to produce a new observation.

# The Noise Schedule

Diffusion needs a variance schedule before it needs a network. Let $`\beta_t` be the per-step
variance, $`\alpha_t=1-\beta_t`, and

$$`\bar\alpha_t=\prod_{s=0}^{t}\alpha_s.`

The runnable layer computes the cumulative coefficients in one pass:

```lean (name := genAlphaBars)
-- Five variances produce five noisy-state coefficients in
-- the runnable convention.
#eval (Schedule.linear 5 0.1 0.5).map
  (fun s => s.alphaBars)
```
```leanOutput genAlphaBars
Except.ok [0.900000, 0.720000, 0.504000, 0.302400, 0.151200]
```

Here the betas are $`0.1,0.2,0.3,0.4,0.5`, so the coefficients are the running products $`0.9`,
$`0.9\cdot 0.8`, $`0.9\cdot 0.8\cdot 0.7`, and so on. PyTorch spells the same computation with
`cumprod` {Informal.citep pytorch2019}[]:

```
# Keep the per-step schedule fixed while comparing
# cumulative products.
betas = torch.linspace(0.1, 0.5, 5)
print(torch.cumprod(1.0 - betas, 0).tolist())
```

```
[0.8999999761581421, 0.7199999690055847, 0.5040000081062317,
 0.30239999294281006, 0.15119999647140503]
```

Both outputs agree to the precision each side prints: TorchLean evaluated in Lean's `Float`
(binary64), and the transcript above is binary32.

For this variance-preserving process, cumulative coefficients must stay in $`[0,1]` and must not
increase. TorchLean's diffusion code guards square roots by applying `max x 0` first, in
`sqrtNonneg` from {src "TorchLean/NN/Models/Diffusion.lean"}[the spec] , which the
runnable noising and DDIM steps call. A negative finite input to that guarded expression therefore
becomes zero before the square root is taken (a property of the guard, not of native `Float.sqrt`).
But clamping alone would conceal an invalid schedule, so the runnable constructor validates the
coefficients:

```lean (name := genSchedErrors)
-- Reject invalid variances and coefficients that would
-- increase the retained signal.
#eval (Schedule.linear 5 (-0.5) 0.5).map
  (fun s => s.alphaBars)

#eval (Schedule.from
    ([0.5, 0.9] : Tensor Float [2])).map
  (fun s => s.alphaBars)
```
```leanOutput genSchedErrors
Except.error "Diffusion.Schedule: beta start must be in [0, 1)"
```
```leanOutput genSchedErrors
Except.error "Diffusion.Schedule: cumulative coefficients must be nonincreasing"
```

`Schedule` has a private constructor, so its public constructors validate the coefficient tensor at
the boundary. Downstream functions can use that validated schedule, but they keep their local
numerical guards, including clamping square-root arguments and guarding division in DDIM.

Two conveniences come with the validated schedule. Timesteps cycle, so a training loop can pass its
own step counter without a modulus, and the normalized time that feeds the network's time channel
is derived from the same index:

```lean (name := genSchedUse)
-- The training counter cycles through the schedule, and
-- time is normalized after indexing.
#eval show IO Unit from do
  let sched ← IO.ofExcept
    (Schedule.linear 5 0.1 0.5)
  IO.println s!"alphaBar 2  = {sched.alphaBar 2}"
  let time : Float := ImageDDIM.timeOfIndex (sched.index 2)
  IO.println s!"time 2      = {time}"
  IO.println s!"index 7     = {sched.index 7}"
  IO.println s!"alphaBar 7  = {sched.alphaBar 7}"
```
```leanOutput genSchedUse
alphaBar 2  = 0.504000
time 2      = 0.500000
index 7     = 2
alphaBar 7  = 0.504000
```

Step seven reuses timestep two because $`7 \bmod 5 = 2`. Cycling lets the loop treat "step" as a
global counter, so a run with
`--steps 1000` and `--T 20` is well defined.

These decreasing coefficients describe how much of the original signal remains (before any weights
have been trained). At the first printed coefficient, the multiplier on the signal is
$`\sqrt{0.9}`. By the last it is $`\sqrt{0.1512}`. On the noise side, the complementary square root
controls the multiplier. Their squares add to one, which explains the variance-preserving
terminology when the clean data and independent noise each have unit variance. That terminology
doesn't say that every individual pixel, or every realized noise tensor, has unit magnitude.

## Schedule Indexing Conventions

The spec layer uses a different index convention. Printing both the per-step variances and their
cumulative coefficients makes the distinction visible:

```lean (name := genSpecSched)
-- Include the clean state at index zero when inspecting the
-- specification schedule.
def genSched : VPSchedule Float 4 :=
  VPSchedule.linear 4 0.1 0.5

#eval genSched.betas

#eval genSched.alphaBarTensor
```
```leanOutput genSpecSched
[0.100000, 0.233333, 0.366667, 0.500000]
```
```leanOutput genSpecSched
[1.000000, 0.900000, 0.690000, 0.437000, 0.218500]
```

Four betas give five cumulative coefficients, because the spec sets $`\bar\alpha_0=1` and defines
$`\bar\alpha_{t+1}=\bar\alpha_t\alpha_t` (a product over $`s<t`). In the runnable layer,
$`\bar\alpha_t` is the product over $`s\le t`, so its first entry is already $`1-\beta_0`. Both are
standardand both are self-consistent. That relationship is recorded in the image-sampling
specification. In {src "TorchLean/NN/Models/Diffusion/ImageDDIM.lean"}[`ImageDDIM.lean`],
`VPSchedule.noisyAlphaBars` drops the clean-state entry, and `ImageDDIM.alphaBar_noisyAlphaBars`
proves that putting it back recovers the original cumulative coefficient at every state index. Then
the API bridge's `schedule_alphaBar_eq` accounts for the runnable schedule's cyclic training index.
The shift belongs in the correspondence, not in an informal convention a caller must remember.

This correspondence starts from matching coefficient tables. The runnable `Schedule.linear`
builds its table from `VPSchedule.linear`: after validating the endpoints, it takes the running
products of that schedule's $`\alpha_t` in one `cumprod` pass. A one-step schedule therefore has
$`\beta_0=\beta_{\mathrm{start}}` in both layers. A schedule loaded with `Schedule.from` is related
to a spec schedule only through its table.

# Forward Noising

The DDPM forward process can jump to any timestep in one step
{Informal.citep ddpm2020}[]:

$$`x_t
=\sqrt{\bar\alpha_t}\,x_0
+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).`

We use `Diffusion.sample` for both cases: give it a seed to draw reproducible noise, or pass
`noise := ...` to use a tensor we already have. The training sample pairs the noised image (with
its time channel attached) against the noise that produced it:

```lean (name := genForward)
-- A one-channel, two-pixel image makes both the noise and
-- added time channel visible.
def genClean : Tensor Float [1, 1, 2] := [[[0.5, -0.25]]]
def genNoise : Tensor Float [1, 1, 2] := [[[1.0, -1.0]]]

#eval show IO Unit from do
  let sched ← IO.ofExcept
    (Schedule.linear 5 0.1 0.5)
  let sample :=
    Diffusion.sample [1]
      ([2] : Tensor Nat [1]) sched genClean 2
      (noise := genNoise)
  IO.println s!"input  = {sample.input}"
  IO.println s!"target = {sample.target}"
```
```leanOutput genForward
input  = [[[1.059237, -0.881755], [0.500000, 0.500000]]]
target = [[[1.000000, -1.000000]]]
```

Read the input as a one-image batch with two channels over a two-pixel row. Its first channel is
$`x_2`, and the second is the constant $`0.5` (the normalized timestep broadcast across the
pixels). And the target is the noise, unchanged: this is epsilon prediction, so the network's job
is to recover $`\epsilon`, not the clean image.

For the first pixel, $`\bar\alpha_2=0.504`. Its clean-image contribution is
$`\sqrt{0.504}\cdot0.5\approx0.354965`, and its noise contribution is
$`\sqrt{0.496}\cdot1\approx0.704273`. Their sum, before the terms are rounded for display, is
approximately $`1.059237`. PyTorch agrees on both entries:

```
# Reconstruct the two pixel values before appending any
# model-conditioning channel.
x0 = torch.tensor([0.5, -0.25])
eps = torch.tensor([1.0, -1.0])
a = torch.tensor(0.504)
print((a.sqrt() * x0 + (1 - a).sqrt() * eps).tolist())
```

```
[1.0592373609542847, -0.8817553520202637]
```

A separate reusable function adds the time channel, which is why the denoiser input has one channel
more than the data:

```lean (name := genTimeChannel)
-- Preserve the image channel and append a spatially
-- constant timestep channel.
#eval Diffusion.appendTimeChannel [1]
  ([2] : Tensor Nat [1]) genClean 0.5
```
```leanOutput genTimeChannel
[[[0.500000, -0.250000], [0.500000, 0.500000]]]
```

For batch $`B`, data channels $`C`, and spatial extent $`S`, the denoiser contract is therefore

$$`B\times(C+1)\times S
\longrightarrow
B\times C\times S,`

The builders in
{src "TorchLean/NN/Models/Diffusion/NoisePredictor.lean"}[
`TorchLean/NN/Models/Diffusion/NoisePredictor.lean`] are
indexed by the config, so `genConf.inputShape [1]` and `genConf.outputShape [1]` below compute this
contract from the same configuration that builds the network.

With the time channel, the network gets information that the corrupted pixels can't reliably supply
on their own. A given observed pixel value can arise from a clean dark pixel with little noise or
from a brighter pixel with a larger negative perturbation. Conditioning on the scheduled noise
level lets one network learn different corrections for those cases. Here time is repeated at every
spatial position, so ordinary convolutions can access it alongside each image patch. Only the
injected noise goes into the target, and the model isn't asked to reproduce its own time channel.

## Seeded Noise Generation

`normalNoise` takes a seed and a step as arguments:

```lean (name := genNoiseKey)
-- Hold the seed fixed and advance the step to select
-- another reproducible noise tensor.
#eval (Diffusion.normalNoise (shape := [4]) 2026 0 :
  Tensor Float [4])

#eval (Diffusion.normalNoise (shape := [4]) 2026 1 :
  Tensor Float [4])
```
```leanOutput genNoiseKey
[-0.371478, 0.424449, -2.588055, -0.399177]
```
```leanOutput genNoiseKey
[-0.711314, -0.448065, 0.971581, 1.495110]
```

Reusing a `(seed, step)` pair reproduces the same tensor. Its key is `Random.keyOf seed step`, which
applies `splitmix64` to `seed + step * Random.counterStep` in `UInt64`. Two distinct seeds at most
`10 ^ 5` apart give `splitmix64` the same input only at steps at least `87435955166711` apart.
Above, the two step values give different tensors (although distinct keys aren't guaranteed to do
so). Replaying a whole training run also requires its data, configuration, checkpoint state, and
backend behavior.

Supplying the noise to `Diffusion.sample` ourselves lets us study the noising formula independently
of the generator. In {ref "probability-and-gradients"}[the probability chapter], the formal
Gaussian law concerns measures. No proof connects that law to the generator's output bits.

# Reverse Sampling

Deterministic DDIM sampling reads the forward formula backwards
{Informal.citep ddim2021}[]. From $`x_t` and a predicted noise $`\widehat\epsilon_t`, estimate the
clean image, apply a postprocessor $`P`, and remix it at the previous timestep. For coefficients
in $`[0,1]`, the runnable step `ImageDDIM.stepFromEps` uses

$$`d_t=\begin{cases}
\sqrt{\bar\alpha_t}&\text{if }\sqrt{\bar\alpha_t}>\delta,\\
\delta&\text{otherwise},
\end{cases}`

$$`\widehat x_0
=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\widehat\epsilon_t}
       {d_t},`

$$`x_{t-1}
=\sqrt{\bar\alpha_{t-1}}\,P(\widehat x_0)
+\sqrt{1-\bar\alpha_{t-1}}\,\widehat\epsilon_t.`

The runnable example passes $`\delta=10^{-12}`, and the default postprocessor is
$`P(x)=\operatorname{clip}(x,-1,1)`. A custom postprocessor must return the same tensor shape.
Postprocessing changes the reconstructed image before remixing, while the noise prediction in the
second term stays the one supplied by the caller. `ImageDDIM.sample` passes the chosen
postprocessor and floor to every step. Use `start := some index` to start from a particular
schedule entry instead of the last one.

To test this formula, supply the noise that was used. Over real arithmetic, with a positive signal
coefficient and an inactive denominator floor, $`\widehat x_0` is exactly $`x_0`. If clipping also
leaves that value unchanged, one step from $`t` to a timestep with
$`\bar\alpha_{t-1}=1` should return the original image:

```lean (name := genDdimPrev)
-- Compare an exact noise prediction with a deliberately
-- oversized prediction.
#eval show IO Unit from do
  let sched ← IO.ofExcept
    (Schedule.linear 5 0.1 0.5)
  let ab := sched.alphaBar 2
  let xt := qSampleAt ab genClean genNoise
  let back := ImageDDIM.stepFromEps 1e-12 1.0 ab xt genNoise
  let bad := Tensor.scale genNoise 3.0
  let saturated := ImageDDIM.stepFromEps 1e-12 1.0 ab xt bad
  IO.println s!"x_2       = {xt}"
  IO.println s!"recovered = {back}"
  IO.println s!"bad eps   = {saturated}"
```
```leanOutput genDdimPrev
x_2       = [[[1.059237, -0.881755]]]
recovered = [[[0.500000, -0.250000]]]
bad eps   = [[[-1.000000, 1.000000]]]
```

The middle line recovers the input to the printed precision. In the last line, a noise prediction
three times too large drives $`\widehat x_0` outside the data range, so clipping returns the
endpoints $`-1` and $`1`. Clamping bounds the image estimate without correcting the inaccurate
noise prediction. This prevents large channel values from propagating through the reverse process,
but a saturated output can still indicate a poor denoiser.

For this example, the destination coefficient is one. Its noise multiplier is therefore zero,
so the printed result exposes the clipped clean estimate directly. At an intermediate destination,
the same estimate would be mixed with a nonzero amount of predicted noise. This is why a
one-step reconstruction and a full reverse trajectory answer different questions: the latter
feeds each resulting sample back into a denoiser at another noise level.

For an image representation with a different range, we can supply another postprocessor. With
identity postprocessing, a clean estimate of two remains two:

```lean (name := genDdimPostprocess)
#eval ImageDDIM.stepFromEps 1e-8 1.0 1.0
  ([2.0] : Tensor Float [1]) ([0.0] : Tensor Float [1])
  (postprocess := id)
```
```leanOutput genDdimPostprocess
[2.000000]
```

The runnable sampler `ImageDDIM.sample` calls this `ImageDDIM.stepFromEps` at every step, so the
evaluations above use the same `Float` operations, in the same order, as the example's sampler.
The step places no positivity condition on the floor: choosing zero or a negative value gives no
safe-division guarantee. The forward-noising theorem in
{src "TorchLean/NN/Models/Diffusion/ImageDDIM/Training.lean"}[the image-DDIM bridge] identifies
the training input, including its appended time channel, when the coefficient tables agree. Neither
the step nor that theorem asserts that floating-point cancellation exactly recovers every clean
image.

## The Denominator Floor in the Reverse Loop

`ImageDDIM.sample` runs the whole reverse loop, and its predictor is an argument. We can pass an
oracle that returns the exact noise, run the four steps of `genSched`, and see whether the data
comes back. The second run raises the floor $`\delta` from the runnable $`10^{-12}` to $`0.5`:

```lean (name := genOracle)
-- Fix the denoiser output so the reverse loop isolates the
-- sampler arithmetic.
def genFlat : Tensor Float [3] := [0.5, -0.25, 0.0]

def genEps : Tensor Float [3] := [1.0, -1.0, 2.0]

/-- A denoiser that returns the exact noise used for
the forward pass. -/
def genOracleModel : EpsModel Float [3] :=
  { eps := fun _ _ => genEps }

#eval show IO Unit from do
  let xT := qSample genSched genFlat 4 genEps
  let oracle (index : Fin 4) (x : Tensor Float [3]) :
      IO (Tensor Float [3]) :=
    pure (genOracleModel.eps x
      (ImageDDIM.timeOfIndex index))
  let alphaBars := genSched.noisyAlphaBars
  let back ← ImageDDIM.sample 1e-12 alphaBars oracle xT
  let floored ← ImageDDIM.sample 0.5 alphaBars oracle xT
  let loss :=
    epsPredLoss genSched genOracleModel genFlat 2 genEps
  IO.println s!"x_4       = {xT}"
  IO.println s!"recovered = {back}"
  IO.println s!"floor 0.5 = {floored}"
  IO.println s!"loss      = {loss}"
```
```leanOutput genOracle
x_4       = [1.117745, -1.000885, 1.768050]
recovered = [0.500000, -0.250000, 0.000000]
floor 0.5 = [0.467440, -0.233720, 0.000000]
loss      = 0.000000
```

In the last line, the epsilon-prediction loss is zero, as it must be for a perfect denoiser. At the
runnable floor the loop returns the data to the printed precision: every square root in this
schedule exceeds $`10^{-12}`, so each step divides by $`\sqrt{\bar\alpha_t}` and the sole error is
rounding.

The floor $`0.5` is active at the first step, where
$`\sqrt{\bar\alpha_4}=\sqrt{0.2185}\approx0.467440`. There $`d_4=0.5`, which multiplies the clean
estimate by $`0.467440/0.5\approx0.934880`. The three later square roots exceed the floor, so each
later step reconstructs, from the exact noise, the scaled estimate it receives, and the shrinkage
reaches the output unchanged. When $`0<\sqrt{\bar\alpha_t}\le\delta`, the floor multiplies that
step's estimate, before the clip, by $`\sqrt{\bar\alpha_t}/\delta`.

## Sampler Stability Bounds

That floor example compares two settings of one step, while the sampler bounds in
{src "TorchLean/NN/Models/Diffusion/Samplers.lean"}[`Samplers.lean`] hold one step map fixed and
ask how far apart it can send two inputs. For an Euler update at a fixed time $`t` and step size
$`\Delta t`, write $`E(x)=x+\Delta t\,f(x,t)`. If the vector field $`f` is $`L`-Lipschitz at that
time, the triangle inequality gives

$$`\begin{aligned}
\|E(x)-E(y)\|_2
&\le \|x-y\|_2+|\Delta t|\,\|f(x,t)-f(y,t)\|_2\\
&\le (1+|\Delta t|L)\|x-y\|_2.
\end{aligned}`

Thus inputs a distance $`\delta` apart emerge no farther than $`(1+|\Delta t|L)\delta` apart.
`eulerStep_l2_lipschitz` proves this bound over the reals, with the Lipschitz
bound on the vector field as its hypothesis.

For DDIM, `ImageDDIM.stepSystem_contracts_of_step_contracts` assumes a contraction bound for the
image step at one index, with a pure predictor, and carries that bound into its `Dynamics.System`
wrapper `ImageDDIM.stepSystem`. It does not prove that the step contracts or compose the changing
timesteps of `ImageDDIM.sample`. Applying either result requires a bound for the particular model
and step being used.

A useful stability estimate must also make its timestep dependence visible. Even if a model has a
uniform Lipschitz bound, the coefficient multiplying its prediction can change strongly near a
small cumulative signal coefficient. A bound for one comfortably conditioned timestep doesn't
establish the same constant near the end of a schedule. In the oracle example, the prediction is
fixed to study arithmetic, while a learned denoiser introduces input sensitivity as a separate
quantity that these hypotheses must control.

# The Denoiser

An ordinary TorchLean model predicts $`\widehat\epsilon`. Its architecture is described by a
config, and the config computes the input and output shapes, so the extra time channel can't be
forgotten:

```lean (name := genConfShapes)
-- RGB data needs a fourth input channel for time, but still
-- predicts three noise channels.
def genConf : nn.NoisePredictor.Config 2 :=
  { dataChannels := 3
    spatial := [2, 2]
    hiddenChannels := 2
    kernelRadius := [1, 1] }

#eval (genConf.inputShape [1], genConf.outputShape [1])
```
```leanOutput genConfShapes
([1, 4, 2, 2], [1, 3, 2, 2])
```

Three data channels and one time channel give the four input channels, and the prediction has one
channel for each of the three data channels. A kernel radius of one gives every convolution a
$`3\times3` same-padding kernel, which preserves the pixel grid through each residual branch.

The reusable constructor is
{src "TorchLean/NN/Models/Diffusion/NoisePredictor.lean"}[`nn.NoisePredictor.residual`].
Building it gives a parameter list that exposes the six convolutions:

```lean (name := genDenoiserShapes)
-- Inspect convolution parameters separately from the image
-- and batch dimensions.
abbrev genIn : Shape := genConf.inputShape [1]
abbrev genOut : Shape := genConf.outputShape [1]

def genDenoiser : nn.Builder (nn.Sequential genIn genOut) :=
  nn.NoisePredictor.residual genConf [1]

#eval nn.Sequential.stateShapes (nn.build 7 genDenoiser)
```
```leanOutput genDenoiserShapes (whitespace := lax)
[[2, 4, 3, 3], [2], [2, 2, 3, 3], [2], [2, 2, 3, 3], [2],
 [2, 2, 3, 3], [2], [2, 2, 3, 3], [2], [3, 2, 3, 3], [3]]
```

Six convolutions with their biases: a stem taking four channels to two, four hidden convolutions
inside two residual blocks, and an output convolution back to three channels. Nothing in that list
depends on the image size, because a convolution's parameters don't. Running the denoiser once on a
constant input confirms the output shape and measures the initial prediction:

```lean (name := genDenoiserRun)
-- Initialize once and run a constant input through the
-- complete denoiser.
#eval do
  let m ← nn.Module.instantiate (nn.build 7 genDenoiser)
    { device := cpu }
  m.eval
  let out ← m.forward (Tensor.full genIn 0.25)
  IO.println s!"{out}"
```
```leanOutput genDenoiserRun (whitespace := lax)
[[[[0.000689, -0.000936], [0.000979, 0.000871]],
  [[0.000113, 0.001545], [-0.000432, -0.001128]],
  [[-0.000220, -0.001236], [-0.001358, -0.000401]]]]
```

For seed 7 and this constant input, every displayed prediction is near zero. That observation is
specific to the initialized model and input: a residual skip connection alone wouldn't explain a
near-zero output. Training updates the parameters through the same model interface used in
{ref "modern-models"}[the architecture chapter].

Because the residual constructor is parameterized by spatial rank, a single definition serves one,
two, or three spatial axes. It has no downsampling, upsampling, multi-scale skip concatenation,
attention blocks, or learned timestep embeddings. And the example has no exponential moving average
of the weights. Those choices limit comparisons with diffusion systems that use a multiscale U-Net
and weight averaging.

# Diffusion Training And Sampling

We maintain a {src "TorchLeanExamples/Models/Generative/Diffusion.lean"}[`Diffusion` command] that
supports prepared CIFAR-10 arrays and converted $`64\times64` image folders. Its CIFAR branch crops
images to $`4\times4` for a fast check of `.npy` loading, typed minibatch construction,
optimization, and artifact writing.

Prepare CIFAR and run one CPU update:

```terminal
# Prepare the arrays consumed by this one-update CPU run.
python3 scripts/datasets/download_example_data.py --cifar10

scripts/lake.sh exe torchlean diffusion --device cpu \
  --dataset cifar10 --n-total 1 \
  --steps 1 --hidden-c 2 --T 2 \
  --log /tmp/diffusion-trainlog.json
```

```terminal +output
[TorchLean] arithmetic: native binary32
[TorchLean] device: cpu
diffusion: diffusion trainer (device=cpu)
model:
Sequential: [1, 4, 4, 4] -> [1, 3, 4, 4], layers=7, params=283, state=283
  [0] Conv(rank=2, in=4, out=2): [1, 4, 4, 4] -> [1, 2, 4, 4] params=74 ...
  [1] relu: [1, 2, 4, 4] -> [1, 2, 4, 4] params=0, state=0 []
  [2] Residual: [1, 2, 4, 4] -> [1, 2, 4, 4] params=76 ...
  [3] relu: [1, 2, 4, 4] -> [1, 2, 4, 4] params=0, state=0 []
  [4] Residual: [1, 2, 4, 4] -> [1, 2, 4, 4] params=76 ...
  [5] relu: [1, 2, 4, 4] -> [1, 2, 4, 4] params=0, state=0 []
  [6] Conv(rank=2, in=2, out=3): [1, 2, 4, 4] -> [1, 3, 4, 4] params=57 ...
steps=1 arithmetic=native scalar=Float32 loss=1.041324 -> 1.039106
  wrote TrainLog JSON: /tmp/diffusion-trainlog.json
diffusion: ok
```

Four of those lines end in `...` where the per-layer state shapes were trimmed to fit this page
(they're the same shapes the `stateShapes` evaluation printed above). All other lines preserve that
recorded run.

The banner connects the model configuration to this run. Here the input is `[1, 4, 4, 4]` and the
output is `[1, 3, 4, 4]`: four channels in because of the time channel, three out because the target
is the noise on three colour channels. At 283 parameters, the whole model keeps this CPU wiring
check small. After one optimizer update on one image, the loss moved from `1.041324` to `1.039106`.
This checks that training changes the objective, but it provides no evidence about sample quality.

The JSON log records some settings alongside the curve. That's useful provenance, but it doesn't
contain the dataset, complete model state, or every setting needed to reproduce a run:

```
{"title": "Diffusion training",
 "steps": [0, 1],
 "series":
 [{"values": [1.04132401943206787109375, 1.03910577297210693359375],
   "name": "loss",
   "color": "#4e79a7"}],
 "notes":
 ["data=cifar10",
  "x=data/real/cifar10/cifar10_train_X.npy",
  "y=data/real/cifar10/cifar10_train_y.npy",
  "nRows=1",
  "dataset=cifar10",
  "device=cpu",
  "lr=0.001000",
  "hiddenChannels=2",
  "T=2",
  "betaStart=0.000100",
  "betaEnd=0.120000"]}
```

The default $`\beta` range, $`10^{-4}` to $`0.12`, has a larger upper variance than the linear
schedule of {Informal.citet ddpm2020}[] (which used $`10^{-4}` to $`0.02` over a thousand steps). A
short schedule may need larger variances to make its terminal distribution resemble pure noise. In
the two-step run above, that doesn't happen: its final cumulative coefficient is approximately
0.88, so unconditional sampling isn't justified by terminal noising alone. That's why the variance
range has to be considered together with the number of diffusion steps.

Increase `--T` while keeping `--steps 1`, and the optimizer still takes one update while the reverse
artifact needs more model evaluations. Then keep `--T` fixed and raise `--hidden-c`: the schedule is
unchanged while the network becomes wider. These changes separate the cost of repeated sampling
steps from the cost of each network evaluation.

You can ask the command for four images, and they answer different questions:

- `--reference-ppm` is the clean input, so you can see what the model was aiming at.
- `--noisy-ppm` is that input after forward noising to the chosen timestep.
- `--reconstruct-ppm` is DDIM denoising starting from the noisy image, which asks whether the model
  improves on its input.
- `--sample-ppm` is an unconditional sample from pure noise, which asks the much harder question of
  whether the model has learned the data distribution.

A short CUDA run that writes all four:

```terminal
# Save all four image stages so a longer run can be
# inspected beyond its scalar loss.
scripts/lake.sh -R -K cuda=true exe torchlean diffusion --device cuda \
  --dataset cifar10 --n-total 8 \
  --steps 20 --hidden-c 4 --T 20 \
  --reference-ppm /tmp/reference.ppm \
  --noisy-ppm /tmp/noisy.ppm \
  --reconstruct-ppm /tmp/reconstruct.ppm \
  --sample-ppm /tmp/sample.ppm
```

# Autoencoders

A plain autoencoder maps an observation through a latent representation and trains the decoder to
reconstruct it. This gives a reference point for the VAE, which samples its latent, and the VQ-VAE,
which selects its latent from a finite codebook:

$$`x
\xrightarrow{\mathrm{encoder}}z
\xrightarrow{\mathrm{decoder}}\widehat x,\qquad
L_{\mathrm{recon}}=\|x-\widehat x\|_2^2.`

The reusable backbone has widths
`dataWidth -> hiddenWidth -> latentWidth -> hiddenWidth -> dataWidth`, with ReLU hidden activations
and an unconstrained output. The runnable CIFAR example appends `nn.op .sigmoid` because its
normalized targets lie in $`[0,1]`. A caller with a different data domain picks a different output
activation, which is why the constructor does not choose one.

Its batch argument is a `Shape`, and the widths become part of the built model's type:

```lean (name := genAeShapes)
-- Use one width configuration for reconstruction, latent
-- input, and scalar scoring.
def genAeConf : nn.Generative.Config :=
  { dataWidth := 16, hiddenWidth := 8, latentWidth := 4 }

#eval (genAeConf.dataShape [1], genAeConf.latentShape [1],
  genAeConf.scoreShape [1])
```
```leanOutput genAeShapes
([1, 16], [1, 4], [1, 1])
```

The runnable
{src "TorchLeanExamples/Models/Generative/Autoencoder.lean"}[`autoencoder` command]
trains this backbone on flattened CIFAR features, so the numbers below come from real `.npy` data:

```terminal
# Train on four selected examples and retain the loss trace
# as JSON.
scripts/lake.sh exe torchlean autoencoder --device cpu \
  --steps 2 --n-total 4 --log /tmp/autoencoder-trainlog.json
```

```terminal +output
[TorchLean] arithmetic: native binary32
[TorchLean] device: cpu
autoencoder: CIFAR vector reconstruction (device=cpu)
dataset size = 1
mean_loss(before training) = 0.120622
mean_loss(after training) = 0.119943
  wrote TrainLog JSON: /tmp/autoencoder-trainlog.json
steps=2 arithmetic=native scalar=Float32 loss=0.120622 -> 0.119943
autoencoder: ok
```

`dataset size = 1` counts minibatches, not images. This example uses a singleton dataset of one
batch, and `--n-total` says how many rows to load before the batch is drawn from them. The printed
loss is the mean reconstruction error for this data and initialization. Normalizing targets and
constraining outputs to $`[0,1]` bounds each squared pixel error by one, but it doesn't determine
how small the initial mean error will be. A comparison with another implementation must also match
the selected pixels, initialization, and loss reduction.

This run checks data loading, flattening, reconstruction, and optimization. No probability law is
attached to the latent vector.

The bottleneck changes what the model has to preserve. Sixteen observed coordinates pass through
four latent coordinates, so reconstruction rewards whatever information survives that compression.
Merely evaluating the decoder on an arbitrary four-vector is possible, but the reconstruction
objective hasn't specified which such vectors should be sampled. In a VAE, the posterior and prior
address that missing distributional choice, while a discrete codebook restricts which latent
vectors the decoder encounters.

# The VAE Objective

A VAE adds an approximate posterior $`q_\phi(z\mid x)` and a prior $`p(z)`. When $`\beta=1`, the
objective below is the negative evidence lower bound, and varying $`\beta` reweights the KL penalty
relative to reconstruction {Informal.citep vae2014}[]:

$$`\mathcal L_{\mathrm{VAE}}
=
\mathbb E_{q_\phi(z\mid x)}
  [-\log p_\theta(x\mid z)]
+\beta\,D_{\mathrm{KL}}
  \left(q_\phi(z\mid x)\,\|\,p(z)\right).`

For diagonal Gaussian posterior parameters $`\mu_i` and $`\sigma_i^2`, the KL against a standard
normal has a closed form:

$$`D_{\mathrm{KL}}
=\frac12\sum_i
\left(\mu_i^2+\sigma_i^2-\log\sigma_i^2-1\right)\ge0.`

TorchLean has no runnable VAE command. Its executable spec pairs an encoder returning
$`(\mu,\log\sigma^2)` with a decoder. As in the diffusion example, reparameterization noise is
passed in as an argument:

```lean (name := genVaeRun)
-- An identity reconstruction at zero noise separates
-- sampling from decoder error.
def genVae : VAE Float [2] [2] :=
  { encoder :=
      { mean := fun x => Tensor.scale x 0.5
        logvar := fun _ => [0.0, 0.0] }
    decoder := fun z => Tensor.scale z 2.0 }

def genObs : Tensor Float [2] := [1.0, -0.5]

#eval show IO Unit from do
  let quiet : Tensor Float [2] := [0.0, 0.0]
  let z := VAE.sampleLatent genVae genObs quiet
  let out := VAE.forward genVae genObs quiet
  let jittered :=
    VAE.forward genVae genObs [0.5, 0.5]
  IO.println s!"latent      = {z}"
  IO.println s!"decoded     = {out}"
  IO.println s!"with noise  = {jittered}"
```
```leanOutput genVaeRun
latent      = [0.500000, -0.250000]
decoded     = [1.000000, -0.500000]
with noise  = [2.000000, 0.500000]
```

With zero noise this model reconstructs its input exactly, because the decoder undoes the encoder.
With noise the reconstruction moves, and how far it moves is set by $`\log\sigma^2`: here the
log-variance is zero, so $`\sigma=1` and half a standard deviation of noise costs a full unit in
the output. Separately, the KL term penalizes departure of the posterior from the standard normal
prior. With zero supplied noise, the loss separates as follows:

```lean (name := genVaeLoss)
-- Zero sampled noise removes reconstruction error while
-- leaving the posterior KL penalty.
#eval show IO Unit from do
  let quiet : Tensor Float [2] := [0.0, 0.0]
  let rec' :=
    VAE.reconstructionLoss genVae genObs quiet
  let kl := VAE.klLoss genVae genObs
  let total :=
    VAE.loss genVae 1.0 genObs quiet
  IO.println s!"reconstruction = {rec'}"
  IO.println s!"kl             = {kl}"
  IO.println s!"total          = {total}"
```
```leanOutput genVaeLoss
reconstruction = 0.000000
kl             = 0.078125
total          = 0.078125
```

Even though reconstruction is exact, the nonzero total is informative. This encoder places the
posterior mean away from zero, so the latent distribution differs from the standard-normal prior.
Setting the sampled noise to zero chooses one point from the reparameterized expression, but it
doesn't remove the posterior's declared variance or its KL cost. Conversely, the noisy forward
example changes the sampled latent and decoded value while leaving the same posterior parameters in
place.

## KL Loss Reduction

PyTorch exposes both the per-coordinate KL values and their sum and mean for the same posterior:

```
# Show both reductions; TorchLean averages these two
# coordinate contributions.
mu = torch.tensor([0.5, -0.25])
q = torch.distributions.Normal(mu, torch.ones(2))
p = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
per = torch.distributions.kl_divergence(q, p)
print(per.tolist(), per.sum().item(), per.mean().item())
```

```
[0.125, 0.03125] 0.15625 0.078125
```

Per coordinate the values are $`\mu^2/2`, namely $`0.125` and $`0.03125`. Their sum, $`0.15625`, is
the KL of the joint diagonal Gaussian, which is the quantity in the displayed formula. TorchLean's
executable `klLoss` returns their mean, $`0.078125`, using a mean over latent coordinates. Other
objectives can use sums or weighted reductions. Check their definitions before comparing scales.

For a fixed positive latent width, scaling the KL term alone preserves its own minimizers. In a
reconstruction-plus-KL objective, however, replacing the sum by a mean changes its weight relative
to reconstruction unless $`\beta` is adjusted. To compare the executable mean against a PyTorch
implementation that sums coordinates, multiply by the latent width. Reparameterization also explains
why noise is passed as a value. For fixed noise, $`z_i=\mu_i+\exp(\mathrm{logvar}_i/2)\epsilon_i` is
a deterministic function of the encoder outputs. Over real arithmetic its derivatives with respect
to those outputs are one and $`\tfrac12\exp(\mathrm{logvar}_i/2)\epsilon_i`, respectively. So the
same supplied noise can be used when comparing two evaluations or tracing a derivative. Drawing
fresh noise between them would change the function input as well as the posterior parameters. This
pathwise calculation is separate from establishing the distribution of a generator that supplies the
noise.

The nonnegativity theorem below concerns the spec's KL term `VAE.meanKLToStandard` over
$`\mathbb R`, the mean that `klLoss` returns:

```lean (name := genVaeKlType)
-- The theorem concerns the spec's KL mean over ℝ for
-- arbitrary posterior parameters.
#check @VAE.meanKLToStandard_nonneg
```
```leanOutput genVaeKlType (whitespace := lax)
@VAE.meanKLToStandard_nonneg :
  ∀ {latent : Shape} (mu logvar : Tensor ℝ latent),
    0 ≤ VAE.meanKLToStandard mu logvar
```

Over $`\mathbb R` the stabilized small-variance branch and the final clamp of the spec agree with
the closed form, so the theorem covers the definition that `klLoss` evaluates. Applying it to a
`Float` evaluation needs an argument about rounding.

The equality case identifies the posterior at which the regularizer vanishes:

```lean (name := genVaeKlZero)
-- Inspect both directions of the characterization of zero
-- KL.
#check @VAE.meanKLToStandard_eq_zero_iff
```
```leanOutput genVaeKlZero (whitespace := lax)
@VAE.meanKLToStandard_eq_zero_iff :
  ∀ {latent : Shape} (mu logvar : Tensor ℝ latent),
    VAE.meanKLToStandard mu logvar = 0 ↔ mu = 0 ∧ logvar = 0
```

The KL term vanishes exactly when every coordinate is already standard normal. The proof is the
elementary inequality $`e^x\ge 1+x` applied coordinatewise, with equality only at zero.

# Discrete Codes

A VQ-VAE replaces the continuous latent with a lookup into a finite codebook
{Informal.citep vqvae2017}[]. Its encoder produces $`z_e(x)`, the nearest code $`e_k` replaces it,
and the decoder sees only the code:

$$`\mathcal L_{\mathrm{VQ}}
=\underbrace{\|x-\mathrm{dec}(e_k)\|^2}_{\text{reconstruction}}
+\underbrace{\|e_k-z_e(x)\|^2}_{\text{codebook}}
+\beta\underbrace{\|z_e(x)-e_k\|^2}_{\text{commitment}}.`

Of the three terms, the second and third have the same scalar value but train different parameters.
The codebook term moves the selected embedding toward a fixed encoder output, while the commitment
term moves the encoder output toward a fixed embedding. TorchLean keeps these scalar objectives in
`VQVAE` and implements their distinct gradient paths in
{src "TorchLean/NN/Models/VQVAE/Autograd.lean"}[`VQVAE.Autograd`]. Its
`codebookLoss` detaches the encoder output, while `commitmentLoss` detaches the selected code.

Reconstruction needs a third choice. Here the decoder receives
`detach(selected) + (encoded - detach(encoded))`, whose forward value over the reals is the
selected code. Finite floating-point evaluation can change a signed zero, and nonfinite encoder
values can produce NaN. The incoming decoder cotangent passes to the encoder, while the codebook
receives no reconstruction gradient through this expression. This is the straight-through
estimator, whose derivative rule is deliberately specified separately from hard nearest-code
selection. There is no claim that a hard argmin has this classical derivative.

For a latent with $`n` coordinates and an incoming reconstruction cotangent $`g`, the specified
encoder contribution is $`g + (2\beta/n)(z_e-e_k)`, and the selected code receives
$`(2/n)(e_k-z_e)`. Unselected code rows receive zero from that assignment. In the spec,
`straightThroughGrads` spells out these roles. The evaluations below inspect forward loss values,
and those values alone can't distinguish a correctly detached implementation from one that sends
both auxiliary gradients through both arguments.

A three-entry codebook over a two-dimensional latent, with an identity encoder and decoder so the
arithmetic stays visible:

```lean (name := genVqDist)
-- Three candidate codes and an identity decoder keep
-- selection and loss arithmetic explicit.
def genBook : Codebook Float 3 [2] :=
  { embedding := fun i =>
      match i with
      | 0 => [0.0, 0.0]
      | 1 => [1.0, 1.0]
      | 2 => [-1.0, 0.5] }

def genVq : VQVAE Float [2] [2] 3 :=
  { encoder := fun x => x
    codebook := genBook
    decoder := fun z => z }

/-- Nearest code by mean squared distance, with the first
index winning ties. -/
def genNearest {n : Nat} [NeZero n] {latent : Shape}
    (book : Codebook Float n latent)
    (z : Tensor Float latent) : Fin n :=
  let dist (i : Fin n) : Float :=
    Tensor.meanSquaredError (book.embedding i) z
  (List.finRange n).foldl
    (fun best i => if dist i < dist best then i else best)
    ⟨0, Nat.pos_of_neZero n⟩

#eval show IO Unit from do
  let dists := (List.finRange 3).map (fun i =>
    Tensor.meanSquaredError (genBook.embedding i) genObs)
  IO.println s!"distances = {dists}"
  IO.println s!"nearest   = {genNearest genBook genObs}"
```
```leanOutput genVqDist
distances = [0.625000, 1.125000, 2.500000]
nearest   = 0
```

We write the argmin here instead of importing it, because the spec deliberately doesn't contain
one. `VQVAE.quantized` takes the index as an argument, so the model file says nothing
about how it was chosen. Taking the index as an argument separates code selection from decoding,
just as the diffusion functions separate noise generation from noising.

PyTorch computes the same three distances the same way:

```
# Compute the same mean distances before selecting the first
# minimum.
book = torch.tensor([[0., 0.], [1., 1.], [-1., 0.5]])
z = torch.tensor([1.0, -0.5])
d = ((book - z) ** 2).mean(dim=1)
print(d.tolist(), int(d.argmin()))
```

```
[0.625, 1.125, 2.5] 0
```

Feeding the winning index back gives the three loss terms:

```lean (name := genVqLoss)
-- Reuse the selected index in reconstruction and both
-- latent penalties.
#eval show IO Unit from do
  let idx := genNearest genBook genObs
  let vq := VQVAE.quantized genVq idx
  let recon :=
    VQVAE.reconstructionLoss genVq genObs idx
  let cb := VQVAE.codebookLoss genVq genObs idx
  let total := VQVAE.loss genVq 0.25 genObs idx
  IO.println s!"z_q    = {vq}"
  IO.println s!"recon  = {recon}"
  IO.println s!"book   = {cb}"
  IO.println s!"total  = {total}"
```
```leanOutput genVqLoss
z_q    = [0.000000, 0.000000]
recon  = 0.625000
book   = 0.625000
total  = 1.406250
```

All three terms coincide at $`0.625` because the encoder and decoder are identities here, so the
total is $`(1+1+\beta)\cdot0.625` with $`\beta=0.25`. Selecting code one instead increases the loss
in this example:

```lean (name := genVqWrong)
-- Force a different code to expose the effect of assignment
-- on the total value.
#eval show IO Unit from do
  let one : Fin 3 := 1
  let vq := VQVAE.quantized genVq one
  let total := VQVAE.loss genVq 0.25 genObs one
  IO.println s!"z_q   = {vq}"
  IO.println s!"total = {total}"
```
```leanOutput genVqWrong
z_q   = [1.000000, 1.000000]
total = 2.531250
```

With a learned nonlinear decoder, nearest latent distance need not select the smallest
reconstruction error.

Changing latent width affects the training scale even when nearest-code ordering is unchanged. A
mean squared distance divides each coordinate's contribution by the number of latent coordinates,
and the auxiliary gradient formulas above contain that same divisor. So keeping the commitment
weight fixed while changing from a sum to a mean changes the update relative to reconstruction.
Codebook and commitment losses share a scalar value, but that removes neither this scaling choice
nor their different stop-gradient roles.

## The Nearest-Code Hypothesis

The theory layer does not prove that any particular argmin is correct. It states the nearest-code
predicate with the Euclidean distance and proves that a nonempty codebook always has a nearest
code:

```lean (name := genVqThm)
-- A nonempty codebook has a nearest code for every
-- encoder output.
#check @VQVAE.exists_isNearestCode
```
```leanOutput genVqThm (whitespace := lax)
@VQVAE.exists_isNearestCode :
  ∀ {numCodes d : ℕ} [NeZero numCodes] (embedding : Fin numCodes → Fin d → ℝ)
    (z : Fin d → ℝ), ∃ idx, VQVAE.IsNearestCode embedding z idx
```

`IsNearestCode embedding z idx` says that no code is closer to `z` than code `idx`. A CUDA argmin
kernel, a NumPy `argmin`, or the Lean fold above would need a separate argument that its result
satisfies `IsNearestCode`. Tied nearest codes have the same
quantization distance, but their embeddings can differ, so passing them through a decoder can
produce different reconstructions and reconstruction losses.

`IsNearestCode` compares Euclidean distances, whereas the executable `codebookLoss` uses
`Tensor.meanSquaredError` (the squared distance divided by the latent size). Over real arithmetic
and a nonempty latent shape, both give the same nearest codes, since the mean squared error is the
squared distance divided by a positive constant. Using `IsNearestCode` for the executable mean
requires spelling out that relationship and the change of scalar arithmetic.

# Adversarial Training

A GAN replaces the explicit likelihood with a second network. TorchLean's spec uses the
least-squares form {Informal.citep lsgan2017}[], which regresses scores onto targets instead of
using a log-loss:

$$`\mathcal L_D=(D(x)-1)^2+D(G(z))^2,
\qquad
\mathcal L_G=(D(G(z))-1)^2.`

Both networks come from the same shape config as the autoencoder, since a generator is a decoder
from latent space and a discriminator is an encoder onto a single score:

```lean (name := genGanRun)
-- Keep generator and discriminator parameter initialization
-- separate.
def genGanConf : nn.Generative.Config :=
  { dataWidth := 4, hiddenWidth := 4, latentWidth := 2 }

abbrev genLat : Shape := genGanConf.latentShape [1]
abbrev genData : Shape := genGanConf.dataShape [1]
abbrev genScore : Shape := genGanConf.scoreShape [1]

def genG : nn.Builder (nn.Sequential genLat genData) :=
  nn.Generative.generator genGanConf [1]

def genD : nn.Builder (nn.Sequential genData genScore) :=
  nn.Generative.discriminator genGanConf [1]

#eval do
  let g ← nn.Module.instantiate (nn.build 1 genG)
    { device := cpu }
  let d ← nn.Module.instantiate (nn.build 2 genD)
    { device := cpu }
  g.eval
  d.eval
  let z : Tensor Float32 [1, 2] := [[0.5, -0.5]]
  let real : Tensor Float32 [1, 4] := [[1.0, 0.0, 1.0, 0.0]]
  let fake ← g.forward z
  let scoreFake ← d.forward fake
  let scoreReal ← d.forward real
  let sf := Tensor.at scoreFake (0, (0, ()))
  let sr := Tensor.at scoreReal (0, (0, ()))
  IO.println s!"G z     = {fake}"
  IO.println s!"D (G z) = {scoreFake}"
  IO.println s!"D x     = {scoreReal}"
  let lossG := (sf - 1.0) * (sf - 1.0)
  let lossD := (sr - 1.0) * (sr - 1.0) + sf * sf
  IO.println s!"L_G     = {lossG}"
  IO.println s!"L_D     = {lossD}"
```
```leanOutput genGanRun
G z     = [[0.469412, 0.508243, 0.458092, 0.125636]]
D (G z) = [[-0.199216]]
D x     = [[-0.042200]]
L_G     = 1.438119
L_D     = 1.125867
```

For this pair of inputs, the discriminator gives scores about $`-0.0422` and $`-0.1992`. From the
fake score, the generator loss is $`(-0.1992-1)^2\approx1.4381`, and the discriminator loss adds
$`(-0.0422-1)^2` and $`(-0.1992)^2`, giving approximately 1.1259. These are point evaluations, not
measurements over either distribution. Scores 1 on the real input and 0 on the generated input
instead give:

```
# Perfect discriminator targets need not minimize the
# generator objective.
d_real, d_fake = torch.tensor(1.0), torch.tensor(0.0)
print(((d_real - 1) ** 2 + d_fake ** 2).item(), ((d_fake - 1) ** 2).item())
```

```
0.0 1.0
```

On this fixed pair, those scores minimize $`\mathcal L_D` at zero but leave $`\mathcal L_G=1`. The
generator seeks to move its score toward 1, while the discriminator seeks to keep generated scores
near 0. Changes to either network change the other network's objective. A single loss value doesn't
establish sample quality.

This theorem from
{src "TorchLean/NN/Models/GAN/Loss.lean"}[`TorchLean/NN/Models/GAN/Loss.lean`]
states the discriminator loss at perfect scores:

```lean (name := genGanThm)
-- Read the score hypotheses that make every discriminator
-- squared error vanish.
#check @GAN.discriminatorLoss_eq_zero
```
```leanOutput genGanThm (whitespace := lax)
@GAN.discriminatorLoss_eq_zero :
  ∀ {latent obs : Shape} (model : GAN ℝ latent obs) (xReal : Tensor ℝ obs)
    (z : Tensor ℝ latent),
  model.discriminator xReal = GAN.realTarget →
    model.fakeScore z = GAN.fakeTarget →
      model.discriminatorLoss xReal z = 0
```

Evaluating `D (G z)` checks the forward composition. Adversarial training also needs the
generator's gradient to flow through the discriminator, alternating parameter updates, and control
over which network receives gradients during each update. Here the API builders are two independent
supervised models. As the spec docstring states, they aren't constructed from the GAN records, and
no theorem relates the two. There's no runnable GAN training command.

Neither objective is a probability, and a negative initialized score is valid for this
unconstrained scalar output. When comparing GAN runs, identify whose update produced a loss value.

# Masked Autoencoding

Masked autoencoding trains a reconstruction model on partially hidden inputs
{Informal.citep mae2022}[]. Hide part of the image, reconstruct what was hidden, and score only the
hidden coordinates:

$$`L_{\mathrm{MAE}}
=\frac1{|M|}\sum_{i\in M}
\left(\widehat x_i-x_i\right)^2,`

where $`M` is the hidden coordinate set. Both the mask and the loss must use the same set $`M`. If
the loss averages over all pixels, a model can score well by copying the visible ones.

TorchLean makes the mask a first-class value. `SelfSupervised.BlockMask.apply` zeros whole blocks of
a tensor according to a per-axis block policy, a period, and a phase. On a $`4\times4`
single-channel image with $`2\times2` blocks and period four, exactly one block is hidden:

```lean (name := genMask)
-- Mask spatial blocks together while preserving the channel
-- axis.
def genMaeBlocks : Tensor (Option Nat) [3] :=
  [none, some 2, some 2]

def genImage : Tensor Float [1, 4, 4] :=
  [[[1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]]]

#eval show IO Unit from do
  let masked :=
    SelfSupervised.BlockMask.apply genImage genMaeBlocks 4 0
  IO.println s!"{masked}"
```
```leanOutput genMask (whitespace := lax)
[[[0.000000, 0.000000, 3.000000, 4.000000],
  [0.000000, 0.000000, 7.000000, 8.000000],
  [9.000000, 10.000000, 11.000000, 12.000000],
  [13.000000, 14.000000, 15.000000, 16.000000]]]
```

Putting `none` on the channel axis excludes it from block indexing. A hidden spatial block
therefore removes all color channels at those pixels. Hiding only one channel would leave the other
channels at the same location visible and define a different reconstruction task.

That same policy produces the loss support, and it's returned as flat indices so the loss can
weight a flattened prediction:

```lean (name := genMaskIdx)
-- Compare flattened support for one channel and for three
-- repeated channel planes.
#eval show IO Unit from do
  let small := SelfSupervised.BlockMAE.hiddenIndices
    (dataShape := [1, 4, 4]) genMaeBlocks 4 0
  let cifar := SelfSupervised.BlockMAE.hiddenIndices
    (dataShape := [3, 4, 4]) genMaeBlocks 4 0
  IO.println s!"one channel  = {small.map (fun i => i.val)}"
  IO.println s!"three chans  = {cifar.map (fun i => i.val)}"
  IO.println s!"count        = {cifar.size} of 48"
```
```leanOutput genMaskIdx (whitespace := lax)
one channel  = #[0, 1, 4, 5]
three chans  = #[0, 1, 4, 5, 16, 17, 20, 21, 32, 33, 36, 37]
count        = 12 of 48
```

Twelve of forty-eight coordinates (four per channel), offset by sixteen because each channel is a
$`4\times4` plane. The runnable `mae` command builds its loss weights from these same indices,
giving each hidden coordinate weight $`1/12` and every visible coordinate weight zero, so this
command derives its masking and scoring support from one policy.

`SelfSupervised.BlockMAE.sample` pairs the masked input with a target drawn from the original image:

```lean (name := genMaeSample)
-- The target is gathered from the original image, before
-- its input is masked.
#eval show IO Unit from do
  let pair ← IO.ofExcept
    (SelfSupervised.BlockMAE.sample []
      (dataShape := [1, 4, 4]) 8
      genMaeBlocks 4 0 genImage)
  IO.println s!"target = {pair.target}"
```
```leanOutput genMaeSample (whitespace := lax)
target = [1.000000, 2.000000, 3.000000, 4.000000, 5.000000,
  6.000000, 7.000000, 8.000000]
```

Because the input is masked and the target is the original, coordinates 0, 1, 4 and 5 appear in the
target with their true values 1, 2, 5 and 6 while the model sees zeros there. Since the width
argument truncates the target to a flat prefix, it's validated: asking for more coordinates than
the data holds returns an error, not a silently padded tensor.

The runnable command builds a compact ViT around this: $`2\times2` patches with stride two, a
four-wide token stream, two attention heads, one encoder block, and a linear pixel decoder. In the
published MAE the architecture is asymmetric: its encoder processes only visible patches, and a
lightweight decoder reconstructs the image from encoded patches and mask tokens
{Informal.citep mae2022}[]. Here the compact command instead feeds a zero-masked image through its
ViT. It exercises patch extraction, token attention, and masked-only scoring, but doesn't reproduce
the published encoder's omission of masked patches.

```terminal
# Run one reconstruction update with the command’s default
# masking configuration.
scripts/lake.sh exe torchlean mae --device cpu --steps 1 --n-total 1 --log false
```

```terminal +output
[TorchLean] arithmetic: native binary32
[TorchLean] device: cpu
mae: CIFAR masked reconstruction (device=cpu)
dataset size = 1
mean_loss(before training) = 0.473153
mean_loss(after training) = 0.497867
steps=1 arithmetic=native scalar=Float32 loss=0.473153 -> 0.497867
mae: ok
```

A single optimizer update need not lower the loss, and this one raises the masked reconstruction
error from `0.473153` to `0.497867`.

The MAE and autoencoder losses have different architectures, normalization, and scoring support.
Their differing magnitudes don't establish that masking is implemented correctly. The hidden-index
calculation above is the relevant check that the mask and scoring support agree.

# Result Scope

Each result's source determines its scope:

:::table +header
*
  * Claim
  * What produced it
  * What it covers
*
  * `alphaBars` values, DDIM roundtrip, VAE and VQ-VAE losses
  * evaluation during the guide build
  * these inputs, at `Float`, in the build's runtime
*
  * `stateShapes`, input and output shapes
  * type-level computation
  * every model built from that config
*
  * losses before and after training, PPM or JSON written
  * one CLI run
  * one optimization trajectory, one artifact
*
  * sampler Lipschitz bounds
  * a proof over $`\mathbb R`
  * every step satisfying the stated hypotheses
*
  * KL nonnegativity and its equality case
  * a proof over $`\mathbb R`
  * the spec's KL term `VAE.meanKLToStandard`, which `klLoss` evaluates
*
  * existence of a nearest code
  * a proof over $`\mathbb R`
  * every nonempty codebook and encoder output
*
  * backend capsule
  * runtime provenance record
  * how one accelerated operation was provided
:::

# Implementation And Proof Gaps

These examples expose three missing connections:

- The image-DDIM bridge relates the cyclic API index, the clean-state coefficient, and the time
  channel. The runnable reverse step is `ImageDDIM.stepFromEps` itself. Comparing a loaded schedule
  requires matching its coefficient table, and the sampler stability results take the step's
  Lipschitz or contraction bound as a hypothesis.
- Over $`\mathbb R`, the KL theorems are about the spec's `VAE.meanKLToStandard`, the mean that
  `klLoss` returns. The evaluations above run at `Float`, and no theorem bounds the rounding between
  those values and the real-valued ones.
- The API's generator and discriminator builders are two independent supervised models. The spec's
  docstring records that they aren't built from its GAN records and have no correspondence theorem.
  Adding an adversarial trainer requires implementing and testing the alternating updates and
  stop-gradient paths.

# Further Reading

- {Informal.citet ddpm2020}[] is the paper the forward process and the training objective come
  from, and its appendix contains the closed-form $`q(x_t\mid x_0)` used above.
- {Informal.citet ddim2021}[] introduces the deterministic sampler that `ImageDDIM.stepFromEps`
  implements.
- {Informal.citet vae2014}[] is the source of the reparameterization trick, which is why the noise
  is an argument in every function on this page.
- {Informal.citet vqvae2017}[] introduces the codebook and the three-term objective.
- {Informal.citet lsgan2017}[] is the least-squares adversarial objective the spec uses instead of
  the original log-loss.
- {Informal.citet mae2022}[] is the masked-image objective the `mae` command is modelled on.
- {Informal.citet pytorch2019}[] documents the framework every comparison here was run against.
