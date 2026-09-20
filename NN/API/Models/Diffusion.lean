/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Diffusion Models

Config-style diffusion model constructors plus reusable, dataset-independent DDPM/DDIM helpers.

The runnable examples decide where data comes from (CIFAR-10, ImageNet-style folders, synthetic
artifacts).  The definitions here are shape-parametric and can be reused by tests, examples, and
future proof layer specifications.
-/

@[expose] public section

namespace TorchLean


open Spec TorchLean TorchLean.Tensor

namespace nn
namespace models

/-- Configuration for a convolutional diffusion-noise predictor. -/
structure Diffusion.NoisePredictor.Config (d : Nat) where
  /-- Number of channels in the denoised sample. -/
  dataChannels : Nat
  /-- Size of each sample axis. Values such as `[32, 32]` work directly. -/
  spatial : Tensor Nat [d]
  /-- Hidden channel width. -/
  hiddenChannels : Nat := 32
  /--
  Radius of the same-padding convolution kernel on each axis.

  A radius of `1` gives the usual kernel size `3`; every residual branch therefore preserves the
  sample grid by construction.
  -/
  kernelRadius : Tensor Nat [d] := Tensor.ones [d]

namespace Diffusion.NoisePredictor.Config

/-- Validate the complete epsilon-predictor geometry before allocating convolution parameters. -/
def validate {d : Nat} (config : Diffusion.NoisePredictor.Config d) :
    Except String Unit := do
  if config.dataChannels = 0 then
    throw "Diffusion.NoisePredictor: data channel count must be positive"
  if config.spatial.prod = 0 then
    throw "Diffusion.NoisePredictor: input spatial dimensions must be positive"
  if config.hiddenChannels = 0 then
    throw "Diffusion.NoisePredictor: hidden channel count must be positive"

/-- Input shape, with one extra channel carrying diffusion time. -/
abbrev input {d : Nat} (config : Diffusion.NoisePredictor.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat
    ((config.spatial.to Shape).prependDim (config.dataChannels + 1))

/-- Output shape matching the denoised data channels. -/
abbrev output {d : Nat} (config : Diffusion.NoisePredictor.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat ((config.spatial.to Shape).prependDim config.dataChannels)

end Diffusion.NoisePredictor.Config

namespace Internal

/-- Implementation helper for the shape-preserving convolutions in the epsilon predictors. -/
def noisePredictorConvolution {d : Nat}
    (config : Diffusion.NoisePredictor.Config d) (batchShape : Shape)
    (inputChannels outputChannels : Nat) :
    nn.Builder (nn.Sequential
      (batchShape.concat ((config.spatial.to Shape).prependDim inputChannels))
      (batchShape.concat ((config.spatial.to Shape).prependDim outputChannels))) := by
  if config.spatial.prod = 0 then
    exact pure <| nn.Internal.invalidConfiguration
      (batchShape.concat ((config.spatial.to Shape).prependDim inputChannels))
      (batchShape.concat ((config.spatial.to Shape).prependDim outputChannels))
      "Diffusion.NoisePredictor"
      "Diffusion.NoisePredictor: input spatial dimensions must be positive"
  else
    let geometry := Convolution.Geometry.samePadding config.kernelRadius
    have preservesSize : geometry.output config.spatial = config.spatial :=
      Convolution.Geometry.output_samePadding config.spatial config.kernelRadius
    simpa [preservesSize] using
      (nn.conv config.spatial (geometry.convolution outputChannels)
        (batchShape := batchShape) (inputChannels := inputChannels))

end Internal

/--
Build a minimal epsilon-predictor conv net:
`conv -> relu -> conv -> relu -> conv -> relu -> conv`.

This stays compact enough for the eager CUDA example while giving the CIFAR trainer more denoising
capacity than a bare two-layer network.
-/
def Diffusion.NoisePredictor.basic {d : Nat}
    (config : Diffusion.NoisePredictor.Config d) (batchShape : Shape := []) :
    nn.Builder
      (nn.Sequential (config.input batchShape) (config.output batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.input batchShape) (config.output batchShape)
        "Diffusion.NoisePredictor" message
  | .ok () =>
      nn.Sequential![
        Internal.noisePredictorConvolution config batchShape
          (config.dataChannels + 1) config.hiddenChannels,
        relu,
        Internal.noisePredictorConvolution config batchShape
          config.hiddenChannels config.hiddenChannels,
        relu,
        Internal.noisePredictorConvolution config batchShape
          config.hiddenChannels config.hiddenChannels,
        relu,
        Internal.noisePredictorConvolution config batchShape
          config.hiddenChannels config.dataChannels
      ]

/--
Build a stronger same-resolution residual epsilon predictor.

Architecture:

`stem conv -> relu -> residual block -> relu -> residual block -> relu -> output conv`

Each residual block preserves `hiddenChannels :: spatial` and computes
$x+\operatorname{conv}(\operatorname{relu}(\operatorname{conv}(x)))$.  This compact residual
denoiser omits U-Net downsampling, upsampling,
and multi-scale skip concatenation. It is still a useful compact architecture because
residual paths make the denoising problem much easier than a plain conv chain while staying within
the eager CUDA memory envelope used by examples.
-/
def Diffusion.NoisePredictor.residual {d : Nat}
    (config : Diffusion.NoisePredictor.Config d)
    (batchShape : Shape := []) :
    nn.Builder
      (nn.Sequential (config.input batchShape) (config.output batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.input batchShape) (config.output batchShape)
        "Diffusion.NoisePredictor" message
  | .ok () =>
      nn.Sequential![
        Internal.noisePredictorConvolution config batchShape
          (config.dataChannels + 1) config.hiddenChannels,
        relu,
        (do
          let block ←
            nn.Sequential![
              Internal.noisePredictorConvolution config batchShape
                config.hiddenChannels config.hiddenChannels,
              relu,
              Internal.noisePredictorConvolution config batchShape
                config.hiddenChannels config.hiddenChannels
            ]
          pure (nn.residual block)),
        relu,
        (do
          let block ←
            nn.Sequential![
              Internal.noisePredictorConvolution config batchShape
                config.hiddenChannels config.hiddenChannels,
              relu,
              Internal.noisePredictorConvolution config batchShape
                config.hiddenChannels config.hiddenChannels
            ]
          pure (nn.residual block)),
        relu,
        Internal.noisePredictorConvolution config batchShape
          config.hiddenChannels config.dataChannels
      ]

end models
end nn

namespace diffusion

/-- Map a tensor from the unit interval to the signed unit interval. -/
def unitToSignedUnit {shape : Shape} (x01 : Tensor Float shape) : Tensor Float shape :=
  TorchLean.Tensor.map (fun value => 2.0 * value - 1.0) x01

/--
Deterministic Gaussian epsilon tensor for an arbitrary diffusion shape.

The `(seed, step)` pair is turned into the runtime RNG key, so examples and artifact generation can
reproduce the same noising path without ambient randomness.
-/
def normalNoise {shape : Shape} (seed step : Nat) : Tensor Float shape :=
  let key : UInt64 := Spec.Random.keyOf (seed := seed) (counter := step)
  Spec.Random.normal (α := Float) key (s := shape)

/-- Diffusion sample layout: arbitrary batch axes followed by channels and spatial axes. -/
@[simp] abbrev sampleShape (batchShape : Shape) (channels : Nat)
    {d : Nat} (spatial : Tensor Nat [d]) : Shape :=
  batchShape.concat ((spatial.to Shape).prependDim channels)

namespace Internal

/-- Linear beta-schedule worker for a natural-number timestep. -/
def linearBetaAt (T : Nat) (betaStart betaEnd : Float) (t : Nat) : Float :=
  if T <= 1 then
    betaStart
  else
    let u := Float.ofNat t / Float.ofNat (T - 1)
    betaStart + u * (betaEnd - betaStart)

/-- Cumulative linear-schedule worker for a natural-number timestep. -/
def linearAlphaBarAt (T : Nat) (betaStart betaEnd : Float) (t : Nat) : Float :=
  Tensor.foldl (· * ·) 1.0 <|
    Tensor.generateFlat [t + 1] fun s => 1.0 - linearBetaAt T betaStart betaEnd s

/-- Whether a loaded cumulative diffusion coefficient is finite and probabilistically valid. -/
def validAlphaBar (value : Float) : Bool :=
  value.isFinite && 0.0 <= value && value <= 1.0

/-- Cumulative diffusion coefficients cannot increase as noise is added. -/
def alphaBarsNonincreasing {steps : Nat} (values : Tensor Float [steps]) : Bool :=
  (Tensor.foldl
    (fun (state : Option Float × Bool) value =>
      match state with
      | (none, valid) => (some value, valid)
      | (some previous, valid) => (some value, valid && value <= previous))
    (none, true) values).2

end Internal

/-- Linear beta-schedule value at a timestep that belongs to the schedule. -/
def linearBeta {T : Nat} (betaStart betaEnd : Float) (t : Fin T) : Float :=
  Internal.linearBetaAt T betaStart betaEnd t.val

/-- The cumulative coefficient $\bar\alpha_t=\prod_{s=0}^{t}(1-\beta_s)$. -/
def linearAlphaBar {T : Nat} (betaStart betaEnd : Float) (t : Fin T) : Float :=
  Internal.linearAlphaBarAt T betaStart betaEnd t.val

/--
The `T` cumulative coefficients of a linear beta schedule.

The length belongs to the return type, so a consumer cannot pair the coefficients with a different
timestep count.
-/
def linearAlphaBars (T : Nat) (betaStart betaEnd : Float) : Tensor Float [T] :=
  let betas : Tensor Float [T] :=
    Tensor.generateFlat [T] (Internal.linearBetaAt T betaStart betaEnd)
  Tensor.scanl (fun alpha beta => alpha * (1.0 - beta)) 1.0 betas

/--
A validated, nonempty diffusion schedule.

The constructor is private so runnable code never carries a separate proof that the coefficient
tensor can be indexed. Use `Schedule.from` for loaded coefficients or `Schedule.linear` for the
standard linear beta schedule.
-/
structure Schedule (steps : Nat) where
  private mk ::
  /-- Cumulative coefficients indexed by diffusion timestep. -/
  alphaBars : Tensor Float [steps]
  /-- Internal invariant used to cycle natural-number training steps safely. -/
  nonempty : steps ≠ 0

namespace Schedule

/-- Validate a coefficient tensor as a runnable diffusion schedule. -/
opaque «from» {steps : Nat} (alphaBars : Tensor Float [steps]) :
    Except String (Schedule steps) :=
  if h : steps = 0 then
    .error "Diffusion.Schedule: must contain at least one step"
  else
    let valid :=
      Tensor.foldl (fun valid value => valid && Internal.validAlphaBar value) true alphaBars
    if !valid then
      .error "Diffusion.Schedule: coefficients must be finite values in [0, 1]"
    else if !Internal.alphaBarsNonincreasing alphaBars then
      .error "Diffusion.Schedule: cumulative coefficients must be nonincreasing"
    else
      .ok ⟨alphaBars, h⟩

/-- Build and validate the standard linear beta schedule. -/
def linear (steps : Nat) (betaStart betaEnd : Float) :
    Except String (Schedule steps) := do
  unless betaStart.isFinite && betaEnd.isFinite do
    throw "Diffusion.Schedule: beta endpoints must be finite"
  unless 0.0 <= betaStart && betaStart < 1.0 do
    throw "Diffusion.Schedule: beta start must be in [0, 1)"
  unless 0.0 <= betaEnd && betaEnd < 1.0 do
    throw "Diffusion.Schedule: beta end must be in [0, 1)"
  Schedule.from (linearAlphaBars steps betaStart betaEnd)

/-- Cycle an arbitrary logical step through this schedule. -/
def index {steps : Nat} (schedule : Schedule steps) (step : Nat) : Fin steps :=
  ⟨step % steps, Nat.mod_lt step (Nat.pos_of_ne_zero schedule.nonempty)⟩

/-- Cumulative coefficient selected by a logical training step. -/
def alphaBar {steps : Nat} (schedule : Schedule steps) (step : Nat) : Float :=
  schedule.alphaBars[schedule.index step]

/-- Normalize a logical training step to the unit interval used by the time channel. -/
def normalizedTime {steps : Nat} (schedule : Schedule steps) (step : Nat) : Float :=
  let timestep := schedule.index step
  -- Use the scalar context's zero and natural-number conversion, as in the schedule spec.
  -- This keeps the time expression shared even when a native conversion has an opaque model.
  if steps <= 1 then
    @Zero.zero Float Context.toZero
  else
    @Nat.cast Float Context.toNatCast timestep.val /
      @Nat.cast Float Context.toNatCast (steps - 1)

end Schedule

namespace Internal

/-- Recursive worker for `diffusion.appendTimeChannel`. -/
def appendTimeChannel (batchShape : Shape) {d c : Nat} (spatial : Tensor Nat [d])
    (x : Tensor Float (sampleShape batchShape c spatial)) (tNorm : Float) :
    Tensor Float (sampleShape batchShape (c + 1) spatial) :=
  match batchShape with
  | .scalar =>
      TorchLean.Tensor.concatAfter [] x <|
        TorchLean.Tensor.full
          ((spatial.to Shape).prependDim 1) tNorm
  | .dim _ rest =>
      TorchLean.Tensor.stackLeading fun index =>
        appendTimeChannel rest spatial (x.unstack index) tNorm

end Internal

/--
Append a constant time channel to every sample in `batchShape`.

The input layout is `batchShape × channels × spatial`. The result preserves the batch and spatial
axes and changes only the channel count from `c` to `c + 1`.
-/
def appendTimeChannel (batchShape : Shape) {d c : Nat} (spatial : Tensor Nat [d])
    (x : Tensor Float (sampleShape batchShape c spatial))
    (tNorm : Float) :
    Tensor Float (sampleShape batchShape (c + 1) spatial) :=
  Internal.appendTimeChannel batchShape spatial x tNorm

/--
Build an epsilon-prediction training sample from explicit noise.

The caller supplies `eps`, usually from the runtime RNG.  Keeping randomness outside this helper
makes the transformation reusable:

$x_t=\sqrt{\bar{\alpha}_t}\,x_0+\sqrt{1-\bar{\alpha}_t}\,\varepsilon$, with target
$\varepsilon$.
-/
def noisedSampleFromNoise (batchShape : Shape) {d c T : Nat}
    (spatial : Tensor Nat [d]) (schedule : Schedule T)
    (x0 eps : Tensor Float (sampleShape batchShape c spatial)) (step : Nat) :
    TorchLean.Sample.Supervised Float
      (sampleShape batchShape (c + 1) spatial)
      (sampleShape batchShape c spatial) :=
  -- The forward process and the reverse step use the same scalar dictionary as the spec.
  -- Keep alpha first in each maximum: changing operand order is not a Float algebraic law.
  let zero := @Zero.zero Float Context.toZero
  let one := @One.one Float Context.toOne
  let _ : Max Float := Context.toMax
  let ab := schedule.alphaBar step
  let sqrtAb : Float := MathFunctions.sqrt (Max.max ab zero)
  let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (one - ab) zero)
  let x_t : Tensor Float (sampleShape batchShape c spatial) :=
    TorchLean.Tensor.add
      (TorchLean.Tensor.scale x0 sqrtAb)
      (TorchLean.Tensor.scale eps sqrtOneMinusAb)
  { input := appendTimeChannel batchShape spatial x_t (schedule.normalizedTime step)
    target := eps }

/--
Build a deterministic epsilon-prediction training sample.

This is the common DDPM training step used by examples: draw reproducible Gaussian noise from
`(seed, step)`, corrupt $x_0$, and use that same noise as the target.
-/
def noisedSample (batchShape : Shape) {d c T : Nat}
    (spatial : Tensor Nat [d]) (schedule : Schedule T)
    (x0 : Tensor Float (sampleShape batchShape c spatial)) (seed step : Nat) :
    TorchLean.Sample.Supervised Float
      (sampleShape batchShape (c + 1) spatial)
      (sampleShape batchShape c spatial) :=
  noisedSampleFromNoise batchShape spatial schedule x0
    (normalNoise (shape := sampleShape batchShape c spatial) seed step)
    step

/--
One deterministic DDIM reverse update ($\eta=0$).

Given $x_t$, predicted epsilon, and adjacent schedule values, this estimates $x_0$ and remixes it to
the previous timestep.

We clamp the intermediate $x_0$ estimate to the training image range $[-1,1]$.  This is the standard
"clipped denoised" stabilizer used by many DDPM/DDIM samplers: without it, a compact model can
drive one color channel far outside the data range and the final PPM exporter merely clips the
damage into saturated color blobs.
-/
def ddimPrev {shape : Shape}
    (abPrev ab : Float)
    (x_t epsHat : Tensor Float shape) : Tensor Float shape :=
  -- Obtain constants and branch operations from the spec's scalar dictionary. In particular,
  -- retain max(alpha, zero) and the strict root > floor test with the floor in the else branch.
  -- Their operand order matters for Float zeros and non-finite inputs.
  let zero := @Zero.zero Float Context.toZero
  let one := @One.one Float Context.toOne
  let _ : Max Float := Context.toMax
  let _ : LT Float := Context.toLT
  let _ : DecidableRel ((· > ·) : Float → Float → Prop) := Context.decidableGT
  let sqrtAb : Float := MathFunctions.sqrt (Max.max ab zero)
  let sqrtAbPrev : Float := MathFunctions.sqrt (Max.max abPrev zero)
  let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (one - ab) zero)
  let sqrtOneMinusAbPrev : Float := MathFunctions.sqrt (Max.max (one - abPrev) zero)
  let x0Hat : Tensor Float shape :=
    TorchLean.Tensor.scale
      (TorchLean.Tensor.sub x_t (TorchLean.Tensor.scale epsHat sqrtOneMinusAb))
      (one / (if sqrtAb > 1e-12 then sqrtAb else 1e-12))
  let x0Clipped : Tensor Float shape :=
    TorchLean.Tensor.clamp x0Hat (-one) one
  TorchLean.Tensor.add
    (TorchLean.Tensor.scale x0Clipped sqrtAbPrev)
    (TorchLean.Tensor.scale epsHat sqrtOneMinusAbPrev)

end diffusion

end TorchLean
