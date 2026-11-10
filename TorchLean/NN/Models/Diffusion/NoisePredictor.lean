/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder
public import TorchLean.NN.Sample
public import TorchLean.Tensor.Cumulative
public import TorchLean.NN.Models.Diffusion.ForwardProcess
public import TorchLean.NN.Models.Diffusion.ImageDDIM
public import TorchLean.Tensor.Operations
public import TorchLean.Tensor.Reduce.Fold

/-!
# Diffusion Models

`nn.NoisePredictor.plain` and `nn.NoisePredictor.residual` build convolutional
epsilon predictors from a `nn.NoisePredictor.Config`. The `Diffusion` namespace holds the
dataset-independent `Schedule` and the training-sample builder `Diffusion.sample`. DDIM sampling
runs the spec's `TorchLean.Diffusion.ImageDDIM.sample`. `Schedule`, `normalNoise`,
`unitToSignedUnit` and `sample` work over `Float`, while the spec definitions they call, such as
`VPSchedule` and `qSampleAt`, are generic in the scalar.

The runnable examples decide where data comes from (CIFAR-10, ImageNet-style folders, synthetic
artifacts). The predictors and the training-sample builder are shape-parametric and can be reused
by tests and examples.

The predictors are ours and stay small. Ho, Jain and Abbeel (2020) use a U-Net and give it `t`
through a Transformer sinusoidal embedding (their Section 4), and `NoisePredictor` appends `t` as
one constant input channel. `unitToSignedUnit` is the `[-1, 1]` data scaling of their Section 3.3,
for data already in `[0, 1]`. `Diffusion.sample` builds the pair `(x_t, ε)` of their Algorithm 1,
which draws `t` uniformly where `sample` cycles it as `step % T`. Entry `i` of a `Schedule` is
their step `t = i + 1`, because `Schedule.linear` stores the inclusive running products of the
alphas.

References:
- Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020), "Denoising Diffusion Probabilistic Models",
  NeurIPS 2020, arXiv:2006.11239, Sections 3.3 and 4, Algorithm 1.
-/

@[expose] public section

namespace TorchLean


open TorchLean.Tensor

namespace nn

/-- Configuration for a convolutional diffusion-noise predictor. -/
structure NoisePredictor.Config (d : Nat) where
  /-- Number of channels in the denoised sample. -/
  dataChannels : Nat
  /-- Size of each sample axis, written as a literal such as `[32, 32]`. -/
  spatial : Tensor Nat [d]
  /-- Hidden channel width. -/
  hiddenChannels : Nat := 32
  /--
  Radius of the same-padding convolution kernel on each axis.

  A radius of `1` gives the usual kernel size `3`. Same padding keeps every convolution on the
  sample grid, so each residual branch preserves the grid by construction.
  -/
  kernelRadius : Tensor Nat [d] := Tensor.ones [d]

namespace NoisePredictor.Config

/-- Validate the complete epsilon-predictor geometry before allocating convolution parameters. -/
def validate {d : Nat} (config : NoisePredictor.Config d) :
    Except String Unit := do
  if config.dataChannels = 0 then
    throw "Diffusion.NoisePredictor: data channel count must be positive"
  if config.spatial.prod = 0 then
    throw "Diffusion.NoisePredictor: input spatial dimensions must be positive"
  if config.hiddenChannels = 0 then
    throw "Diffusion.NoisePredictor: hidden channel count must be positive"

/-- Input shape, with one extra channel carrying diffusion time. -/
abbrev inputShape {d : Nat} (config : NoisePredictor.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++
    (Shape.dim (config.dataChannels + 1) (config.spatial.to Shape))

/-- Output shape matching the denoised data channels. -/
abbrev outputShape {d : Nat} (config : NoisePredictor.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ (Shape.dim config.dataChannels (config.spatial.to Shape))

end NoisePredictor.Config

namespace NoisePredictor

/--
Same-padding convolution from `inputChannels` to `outputChannels` channels on the sample grid. An
empty grid gives the invalid-configuration placeholder that model validation rejects.
-/
def conv {d : Nat}
    (config : NoisePredictor.Config d) (batchShape : Shape)
    (inputChannels outputChannels : Nat) :
    nn.Builder (nn.Sequential
      (batchShape ++ (Shape.dim inputChannels (config.spatial.to Shape)))
      (batchShape ++ (Shape.dim outputChannels (config.spatial.to Shape)))) := by
  if config.spatial.prod = 0 then
    exact pure <| nn.Sequential.invalid
      (batchShape ++ (Shape.dim inputChannels (config.spatial.to Shape)))
      (batchShape ++ (Shape.dim outputChannels (config.spatial.to Shape)))
      "Diffusion.NoisePredictor"
      "Diffusion.NoisePredictor: input spatial dimensions must be positive"
  else
    let geometry := Window.samePadding config.kernelRadius
    have preservesSize : geometry.convOutputSpatial config.spatial = config.spatial :=
      Window.convOutputSpatial_samePadding config.spatial config.kernelRadius
    simpa [preservesSize] using
      (nn.conv config.spatial (geometry.convolution outputChannels)
        (batchShape := batchShape) (inputChannels := inputChannels))

/-- Build one independently seeded residual block whose branch is conv, relu and conv. -/
def residualBlock {d : Nat}
    (config : NoisePredictor.Config d) (batchShape : Shape) :
    nn.Builder (nn.Sequential
      (batchShape ++ (Shape.dim config.hiddenChannels (config.spatial.to Shape)))
      (batchShape ++ (Shape.dim config.hiddenChannels (config.spatial.to Shape)))) := do
  let block ← nn.Sequential![
    conv config batchShape config.hiddenChannels config.hiddenChannels,
    nn.op .relu,
    conv config batchShape config.hiddenChannels config.hiddenChannels
  ]
  pure (nn.residual block)

/--
Build a minimal epsilon-predictor conv net:
`conv -> relu -> conv -> relu -> conv -> relu -> conv`.

The first three convolutions output `hiddenChannels` channels, and the last returns
`dataChannels` noise values at every spatial position.
-/
def plain {d : Nat}
    (config : NoisePredictor.Config d) (batchShape : Shape := []) :
    nn.Builder
      (nn.Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.outputShape batchShape)
        "Diffusion.NoisePredictor" message
  | .ok () =>
      nn.Sequential![
        conv config batchShape
          (config.dataChannels + 1) config.hiddenChannels,
        nn.op .relu,
        conv config batchShape
          config.hiddenChannels config.hiddenChannels,
        nn.op .relu,
        conv config batchShape
          config.hiddenChannels config.hiddenChannels,
        nn.op .relu,
        conv config batchShape
          config.hiddenChannels config.dataChannels
      ]

/--
Build a same-resolution residual epsilon predictor.

Architecture:

`stem conv -> relu -> residual block -> relu -> residual block -> relu -> output conv`

Each residual block preserves `hiddenChannels :: spatial` and computes
$x+\operatorname{conv}(\operatorname{relu}(\operatorname{conv}(x)))$. The skip adds the block's
input unchanged, while its branch learns a correction at the same resolution. There is no U-Net
downsampling, upsampling or multi-scale skip concatenation.
-/
def residual {d : Nat}
    (config : NoisePredictor.Config d)
    (batchShape : Shape := []) :
    nn.Builder
      (nn.Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.outputShape batchShape)
        "Diffusion.NoisePredictor" message
  | .ok () =>
      nn.Sequential![
        conv config batchShape
          (config.dataChannels + 1) config.hiddenChannels,
        nn.op .relu,
        residualBlock config batchShape,
        nn.op .relu,
        residualBlock config batchShape,
        nn.op .relu,
        conv config batchShape
          config.hiddenChannels config.dataChannels
      ]

end NoisePredictor

end nn

namespace Diffusion

/-- Map a tensor from the unit interval to the signed unit interval. -/
def unitToSignedUnit {shape : Shape} (x01 : Tensor Float shape) : Tensor Float shape :=
  TorchLean.Tensor.map (fun value => 2.0 * value - 1.0) x01

/--
Deterministic Gaussian epsilon tensor for an arbitrary diffusion shape.

The key comes from `Random.keyOf seed step`, so examples and artifact generation can
reproduce the same noising path without ambient randomness.
-/
def normalNoise {shape : Shape} (seed step : Nat) : Tensor Float shape :=
  let key : UInt64 := Random.keyOf (seed := seed) (counter := step)
  Random.normal (α := Float) key (s := shape)

/-- Diffusion sample layout: arbitrary batch axes followed by channels and spatial axes. -/
@[simp] abbrev sampleShape (batchShape : Shape) (channels : Nat)
    {d : Nat} (spatial : Tensor Nat [d]) : Shape :=
  batchShape ++ (Shape.dim channels (spatial.to Shape))

/-- Whether each cumulative coefficient is at most the one before it, as adding noise requires. -/
def alphaBarsNonincreasing {steps : Nat} (values : Tensor Float [steps]) : Bool :=
  (Tensor.foldl
    (fun (state : Option Float × Bool) value =>
      match state with
      | (none, valid) => (some value, valid)
      | (some previous, valid) => (some value, valid && value <= previous))
    (none, true) values).2

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
  /-- Internal invariant that lets `index` cycle natural-number training steps. -/
  nonempty : steps ≠ 0

namespace Schedule

/--
Accept a coefficient tensor as a schedule when it has at least one step and its values are finite,
lie in `[0, 1]` and never increase.
-/
@[no_expose] def «from» {steps : Nat} (alphaBars : Tensor Float [steps]) :
    Except String (Schedule steps) :=
  if h : steps = 0 then
    .error "Diffusion.Schedule: must contain at least one step"
  else
    let valid :=
      Tensor.foldl (fun valid value => valid && nn.functional.inUnitInterval value) true alphaBars
    if !valid then
      .error "Diffusion.Schedule: coefficients must be finite values in [0, 1]"
    else if !alphaBarsNonincreasing alphaBars then
      .error "Diffusion.Schedule: cumulative coefficients must be nonincreasing"
    else
      .ok ⟨alphaBars, h⟩

/-- A schedule built by `Schedule.from` stores the supplied coefficients unchanged. -/
theorem alphaBars_of_from_eq_ok {steps : Nat} {alphaBars : Tensor Float [steps]}
    {schedule : Schedule steps} (h : Schedule.from alphaBars = .ok schedule) :
    schedule.alphaBars = alphaBars := by
  unfold Schedule.from at h
  by_cases hs : steps = 0
  · simp [hs] at h
  · by_cases hv :
        Tensor.foldl (fun valid value => valid && nn.functional.inUnitInterval value) true alphaBars
    · by_cases hn : alphaBarsNonincreasing alphaBars
      · simp only [hs, hv, hn, dite_false, Bool.not_true, Bool.false_eq_true, ite_false,
          Except.ok.injEq] at h
        rw [← h]
      · simp [hs, hv, hn] at h
    · simp [hs, hv] at h

/--
Build the standard linear beta schedule from finite beta endpoints in `[0, 1)`. The stored
coefficients are the running products of `VPSchedule.linear`'s alpha values, taken in one
`Tensor.cumprod` and checked by `Schedule.from`.
-/
def linear (steps : Nat) (betaStart betaEnd : Float) :
    Except String (Schedule steps) := do
  unless betaStart.isFinite && betaEnd.isFinite do
    throw "Diffusion.Schedule: beta endpoints must be finite"
  unless 0.0 <= betaStart && betaStart < 1.0 do
    throw "Diffusion.Schedule: beta start must be in [0, 1)"
  unless 0.0 <= betaEnd && betaEnd < 1.0 do
    throw "Diffusion.Schedule: beta end must be in [0, 1)"
  let vp := Diffusion.VPSchedule.linear steps betaStart betaEnd
  Schedule.from (Tensor.cumprod (Tensor.ofFn vp.alpha) ⟨0, Nat.one_pos⟩)

/-- Cycle an arbitrary logical step through this schedule. -/
def index {steps : Nat} (schedule : Schedule steps) (step : Nat) : Fin steps :=
  ⟨step % steps, Nat.mod_lt step (Nat.pos_of_ne_zero schedule.nonempty)⟩

/-- Cumulative coefficient selected by a logical training step. -/
def alphaBar {steps : Nat} (schedule : Schedule steps) (step : Nat) : Float :=
  schedule.alphaBars[schedule.index step]

end Schedule

/--
Append a constant time channel to every sample in `batchShape`.

The input layout is `batchShape × channels × spatial`. The result preserves the batch and spatial
axes and changes only the channel count from `c` to `c + 1`.
-/
def appendTimeChannel {α : Type} [Storage α]
    (batchShape : Shape) {d c : Nat} (spatial : Tensor Nat [d])
    (x : Tensor α (sampleShape batchShape c spatial))
    (tNorm : α) :
    Tensor α (sampleShape batchShape (c + 1) spatial) :=
  Tensor.concatAfter batchShape x (Tensor.full (sampleShape batchShape 1 spatial) tNorm)

/--
Build an epsilon-prediction training sample whose input carries
`ImageDDIM.timeOfIndex (schedule.index step)` as one extra constant channel.

By default `(seed, step)` determines reproducible Gaussian noise, and `noise := ...` passes a
fixed tensor. Both use the same noising formula:

$x_t=\sqrt{\bar{\alpha}_t}\,x_0+\sqrt{1-\bar{\alpha}_t}\,\varepsilon$, with target
$\varepsilon$.
-/
def sample (batchShape : Shape) {d c T : Nat}
    (spatial : Tensor Nat [d]) (schedule : Schedule T)
    (x0 : Tensor Float (sampleShape batchShape c spatial)) (step : Nat) (seed : Nat := 0)
    (noise : Tensor Float (sampleShape batchShape c spatial) := normalNoise seed step) :
    TorchLean.Sample.Supervised Float
      (sampleShape batchShape (c + 1) spatial)
      (sampleShape batchShape c spatial) :=
  { input := appendTimeChannel batchShape spatial
      (Diffusion.qSampleAt (schedule.alphaBar step) x0 noise)
      (Diffusion.ImageDDIM.timeOfIndex (schedule.index step))
    target := noise }

end Diffusion

end TorchLean
