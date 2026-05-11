/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# Diffusion Model Helpers (API)

Config-style diffusion model constructors plus reusable, dataset-independent DDPM/DDIM helpers.

The runnable examples decide where data comes from (CIFAR-10, ImageNet-style folders, synthetic
fixtures).  The definitions here are shape-parametric and can be reused by tests, examples, and
future proof-facing specifications.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-- Configuration for a minimal epsilon-predictor conv net. -/
structure EpsConvNetConfig where
  batch : Nat
  /-- Data channels (e.g. `3` for RGB). The model input has one extra channel for time. -/
  dataC : Nat
  h : Nat
  w : Nat
  /-- Hidden channel width. -/
  hiddenC : Nat := 32

abbrev epsConvNetInShape (cfg : EpsConvNetConfig) : Shape :=
  NN.Tensor.Shape.NCHW cfg.batch (cfg.dataC + 1) cfg.h cfg.w

abbrev epsConvNetOutShape (cfg : EpsConvNetConfig) : Shape :=
  NN.Tensor.Shape.NCHW cfg.batch cfg.dataC cfg.h cfg.w

/--
Build a minimal epsilon-predictor conv net:
`conv -> relu -> conv -> relu -> conv -> relu -> conv`.

This stays compact enough for the eager CUDA example while giving the CIFAR trainer more denoising
capacity than a bare two-layer smoke-test network.
-/
def epsConvNet (cfg : EpsConvNetConfig)
    (h_batch : cfg.batch ≠ 0 := by decide)
    (h_dataC : cfg.dataC ≠ 0 := by decide)
    (h_inC : (cfg.dataC + 1) ≠ 0 := by decide)
    (h_h : cfg.h ≠ 0 := by decide)
    (h_w : cfg.w ≠ 0 := by decide)
    (h_hiddenC : cfg.hiddenC ≠ 0 := by decide) :
    nn.M (nn.Sequential (epsConvNetInShape cfg) (epsConvNetOutShape cfg)) :=
  letI : NeZero cfg.batch := ⟨h_batch⟩
  letI : NeZero cfg.dataC := ⟨h_dataC⟩
  letI : NeZero (cfg.dataC + 1) := ⟨h_inC⟩
  letI : NeZero cfg.h := ⟨h_h⟩
  letI : NeZero cfg.w := ⟨h_w⟩
  letI : NeZero cfg.hiddenC := ⟨h_hiddenC⟩
  nn.sequential![
    withSeeds2 (fun seedK seedB =>
      _root_.NN.API.nn.pure.blocks.conv3x3SameImages
        (n := cfg.batch) (inC := cfg.dataC + 1) (outC := cfg.hiddenC) (h := cfg.h) (w := cfg.w)
        (seedK := seedK) (seedB := seedB)
        (kInit := .uniform (-0.1) 0.1)),
    nn.relu,
    withSeeds2 (fun seedK seedB =>
      _root_.NN.API.nn.pure.blocks.conv3x3SameImages
        (n := cfg.batch) (inC := cfg.hiddenC) (outC := cfg.hiddenC) (h := cfg.h) (w := cfg.w)
        (seedK := seedK) (seedB := seedB)
        (kInit := .uniform (-0.1) 0.1)),
    nn.relu,
    withSeeds2 (fun seedK seedB =>
      _root_.NN.API.nn.pure.blocks.conv3x3SameImages
        (n := cfg.batch) (inC := cfg.hiddenC) (outC := cfg.hiddenC) (h := cfg.h) (w := cfg.w)
        (seedK := seedK) (seedB := seedB)
        (kInit := .uniform (-0.1) 0.1)),
    nn.relu,
    withSeeds2 (fun seedK seedB =>
      _root_.NN.API.nn.pure.blocks.conv3x3SameImages
        (n := cfg.batch) (inC := cfg.hiddenC) (outC := cfg.dataC) (h := cfg.h) (w := cfg.w)
        (seedK := seedK) (seedB := seedB)
        (kInit := .uniform (-0.1) 0.1))
  ]

/--
Build a stronger same-resolution residual epsilon predictor.

Architecture:

`stem conv -> relu -> residual block -> relu -> residual block -> relu -> output conv`

Each residual block has shape `hiddenC×H×W -> hiddenC×H×W` and computes
`x + conv(relu(conv(x)))`.  This compact residual denoiser omits U-Net downsampling, upsampling,
and multi-scale skip concatenation. It is still a useful tutorial-scale architecture because
residual paths make the denoising problem much easier than a plain conv chain while staying within
the eager CUDA memory envelope used by examples.
-/
def epsResidualConvNet (cfg : EpsConvNetConfig)
    (h_batch : cfg.batch ≠ 0 := by decide)
    (h_dataC : cfg.dataC ≠ 0 := by decide)
    (h_inC : (cfg.dataC + 1) ≠ 0 := by decide)
    (h_h : cfg.h ≠ 0 := by decide)
    (h_w : cfg.w ≠ 0 := by decide)
    (h_hiddenC : cfg.hiddenC ≠ 0 := by decide) :
    nn.M (nn.Sequential (epsConvNetInShape cfg) (epsConvNetOutShape cfg)) :=
  letI : NeZero cfg.batch := ⟨h_batch⟩
  letI : NeZero cfg.dataC := ⟨h_dataC⟩
  letI : NeZero (cfg.dataC + 1) := ⟨h_inC⟩
  letI : NeZero cfg.h := ⟨h_h⟩
  letI : NeZero cfg.w := ⟨h_w⟩
  letI : NeZero cfg.hiddenC := ⟨h_hiddenC⟩
  nn.sequential![
    withSeeds2 (fun seedK seedB =>
      _root_.NN.API.nn.pure.blocks.conv3x3SameImages
        (n := cfg.batch) (inC := cfg.dataC + 1) (outC := cfg.hiddenC) (h := cfg.h) (w := cfg.w)
        (seedK := seedK) (seedB := seedB)
        (kInit := .uniform (-0.1) 0.1)),
    nn.relu,
    (do
      let block ←
        nn.sequential![
          withSeeds2 (fun seedK seedB =>
            _root_.NN.API.nn.pure.blocks.conv3x3SameImages
              (n := cfg.batch) (inC := cfg.hiddenC) (outC := cfg.hiddenC)
              (h := cfg.h) (w := cfg.w) (seedK := seedK) (seedB := seedB)
              (kInit := .uniform (-0.1) 0.1)),
          nn.relu,
          withSeeds2 (fun seedK seedB =>
            _root_.NN.API.nn.pure.blocks.conv3x3SameImages
              (n := cfg.batch) (inC := cfg.hiddenC) (outC := cfg.hiddenC)
              (h := cfg.h) (w := cfg.w) (seedK := seedK) (seedB := seedB)
              (kInit := .uniform (-0.1) 0.1))
        ]
      pure (nn.pure.blocks.residual block)),
    nn.relu,
    (do
      let block ←
        nn.sequential![
          withSeeds2 (fun seedK seedB =>
            _root_.NN.API.nn.pure.blocks.conv3x3SameImages
              (n := cfg.batch) (inC := cfg.hiddenC) (outC := cfg.hiddenC)
              (h := cfg.h) (w := cfg.w) (seedK := seedK) (seedB := seedB)
              (kInit := .uniform (-0.1) 0.1)),
          nn.relu,
          withSeeds2 (fun seedK seedB =>
            _root_.NN.API.nn.pure.blocks.conv3x3SameImages
              (n := cfg.batch) (inC := cfg.hiddenC) (outC := cfg.hiddenC)
              (h := cfg.h) (w := cfg.w) (seedK := seedK) (seedB := seedB)
              (kInit := .uniform (-0.1) 0.1))
        ]
      pure (nn.pure.blocks.residual block)),
    nn.relu,
    withSeeds2 (fun seedK seedB =>
      _root_.NN.API.nn.pure.blocks.conv3x3SameImages
        (n := cfg.batch) (inC := cfg.hiddenC) (outC := cfg.dataC) (h := cfg.h) (w := cfg.w)
        (seedK := seedK) (seedB := seedB)
        (kInit := .uniform (-0.1) 0.1))
  ]

end models
end nn

namespace diffusion

/-- Linear beta schedule value at timestep `t`. -/
def linearBeta (T : Nat) (betaStart betaEnd : Float) (t : Nat) : Float :=
  if T <= 1 then
    betaEnd
  else
    let u := Float.ofNat t / Float.ofNat (T - 1)
    betaStart + u * (betaEnd - betaStart)

/--
Compute cumulative products `ᾱ_t = ∏_{s≤t} (1 - β_s)` for a linear beta schedule.

These values connect clean data `x₀`, noised data `x_t`, and the epsilon target used by DDPM-style
training.
-/
def alphaBarsLinear (T : Nat) (betaStart betaEnd : Float) : Array Float :=
  Id.run do
    let mut a : Float := 1.0
    let mut out : Array Float := Array.mkEmpty T
    for t in [0:T] do
      let beta := linearBeta T betaStart betaEnd t
      let alpha := 1.0 - beta
      a := a * alpha
      out := out.push a
    return out

/--
Append a constant time channel to an NCHW image batch.

The epsilon predictor consumes `(data channels + 1)` channels: noisy image channels plus a scalar
timestep broadcast over spatial positions.
-/
def appendTimeChannel {batch c h w : Nat}
    (x : Spec.Tensor Float (NN.Tensor.Shape.NCHW batch c h w)) (tNorm : Float) :
    Spec.Tensor Float (NN.Tensor.Shape.NCHW batch (c + 1) h w) :=
  Tensor.dim (fun bi =>
    Tensor.dim (fun ci =>
      if hci : ci.1 < c then
        let ci' : Fin c := ⟨ci.1, hci⟩
        x[bi][ci']
      else
        Spec.fill tNorm (Spec.Shape.dim h (Spec.Shape.dim w Spec.Shape.scalar))))

/--
Build an epsilon-prediction training sample from explicit noise.

The caller supplies `eps`, usually from the runtime RNG.  Keeping randomness outside this helper
makes the transformation reusable:

`x_t = sqrt(ᾱ_t) * x₀ + sqrt(1 - ᾱ_t) * eps`, target `eps`.
-/
def noisedSampleFromEps {batch c h w : Nat}
    (alphaBars : Array Float) (T : Nat)
    (x0 eps : Spec.Tensor Float (NN.Tensor.Shape.NCHW batch c h w)) (step : Nat) :
    sample.Supervised Float
      (NN.Tensor.Shape.NCHW batch (c + 1) h w)
      (NN.Tensor.Shape.NCHW batch c h w) :=
  let tIdx : Nat := if T = 0 then 0 else step % T
  let ab : Float := alphaBars.getD tIdx 1.0
  let sqrtAb : Float := MathFunctions.sqrt (Max.max ab 0.0)
  let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (1.0 - ab) 0.0)
  let x_t : Spec.Tensor Float (NN.Tensor.Shape.NCHW batch c h w) :=
    Spec.Tensor.addSpec
      (Spec.Tensor.scaleSpec x0 sqrtAb)
      (Spec.Tensor.scaleSpec eps sqrtOneMinusAb)
  let tNorm : Float := if T <= 1 then 0.0 else Float.ofNat tIdx / Float.ofNat (T - 1)
  sample.mk (appendTimeChannel x_t tNorm) eps

/--
One deterministic DDIM reverse update (`η = 0`).

Given `x_t`, predicted epsilon, and adjacent schedule values, this estimates `x₀` and remixes it to
the previous timestep.

We clamp the intermediate `x₀` estimate to the training image range `[-1,1]`.  This is the standard
"clipped denoised" stabilizer used by many DDPM/DDIM samplers: without it, a small tutorial model can
drive one color channel far outside the data range and the final PPM exporter merely clips the
damage into saturated color blobs.
-/
def ddimPrev {batch c h w : Nat}
    (abPrev ab : Float)
    (x_t epsHat : Spec.Tensor Float (NN.Tensor.Shape.NCHW batch c h w)) :
    Spec.Tensor Float (NN.Tensor.Shape.NCHW batch c h w) :=
  let sqrtAb : Float := MathFunctions.sqrt (Max.max ab 0.0)
  let sqrtAbPrev : Float := MathFunctions.sqrt (Max.max abPrev 0.0)
  let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (1.0 - ab) 0.0)
  let sqrtOneMinusAbPrev : Float := MathFunctions.sqrt (Max.max (1.0 - abPrev) 0.0)
  let x0Hat : Spec.Tensor Float (NN.Tensor.Shape.NCHW batch c h w) :=
    Spec.Tensor.scaleSpec
      (Spec.Tensor.subSpec x_t (Spec.Tensor.scaleSpec epsHat sqrtOneMinusAb))
      (1.0 / (if sqrtAb > 1e-12 then sqrtAb else 1e-12))
  let x0Clipped : Spec.Tensor Float (NN.Tensor.Shape.NCHW batch c h w) :=
    Spec.Tensor.clampSpec x0Hat (-1.0) 1.0
  Spec.Tensor.addSpec
    (Spec.Tensor.scaleSpec x0Clipped sqrtAbPrev)
    (Spec.Tensor.scaleSpec epsHat sqrtOneMinusAbPrev)

/--
Write the first image in an RGB NCHW batch as an ASCII PPM.

This dependency-free writer is for example artifacts and quick visual checks, not high-throughput
image export.
-/
def writeFirstRgbNchwPpm {batch c h w : Nat}
    (path : System.FilePath) (x : Spec.Tensor Float (NN.Tensor.Shape.NCHW batch c h w)) : IO Unit := do
  if c < 3 then
    throw <| IO.userError "diffusion PPM export requires at least 3 channels"
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let clamp01 (v : Float) : Float :=
    if v < 0.0 then 0.0 else if v > 1.0 then 1.0 else v
  let toByte (v : Float) : Nat :=
    let v01 := clamp01 ((v + 1.0) / 2.0)
    Nat.min 255 ((v01 * 255.0).toUInt64.toNat)
  let hOut ← IO.FS.Handle.mk path IO.FS.Mode.write
  hOut.putStr s!"P3\n{w} {h}\n255\n"
  let getPx (ci hi wi : Nat) : Float :=
    (Spec.getSpec (α := Float) (s := NN.Tensor.Shape.NCHW batch c h w) x [0, ci, hi, wi]).getD 0.0
  for hi in [0:h] do
    for wi in [0:w] do
      hOut.putStr s!"{toByte (getPx 0 hi wi)} {toByte (getPx 1 hi wi)} {toByte (getPx 2 hi wi)}\n"

end diffusion

end API
end NN
