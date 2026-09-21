/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.Runtime.Autograd.Model.FnoRfft

/-!
# Fourier Neural Operators

`fno` is polymorphic in spatial rank and uses separable FFTs or a dense multidimensional DFT with
the same real/imaginary weights. `fnoRfft` uses learned nonnegative-frequency weights and chooses a
dense reference and native cuFFT with the same checkpoint. Its default ReLU and parameter layout
match the specialized CUDA Burgers model. The two constructors have different spectral weights;
switching between them requires an explicit model conversion.

Reference: Zongyi Li et al., *Fourier Neural Operator for Parametric Partial Differential
Equations*, ICLR 2021.
-/

@[expose] public section

namespace TorchLean.nn.models

/-- Configuration for a scalar-field FNO over `d` spatial axes. -/
structure FNO.Config (d : Nat) where
  /-- Size of each sampled grid axis. -/
  spatial : Tensor Nat [d]
  /--
  Width of the low- and high-frequency bands retained along each full-DFT axis.

  The bands use ordinary FFT indexing: coordinates below `modes` and coordinates at least
  `spatial - modes` are retained. Overlapping bands retain the entire axis.
  -/
  modes : Tensor Nat [d]
  /-- Width of the latent channel representation. -/
  width : Nat
  /-- Number of spectral residual blocks. -/
  layerCount : Nat
  /-- Activation applied after each spectral residual block. -/
  activation : Activation.Kind := .tanh
  /-- Per-axis FFT execution or the dense full-grid reference, with the same parameters. -/
  spectralPath : Runtime.Autograd.Model.F.SpectralPath := .automatic

namespace FNO.Config

/-- Validate the complete operator geometry before allocating any spectral parameters. -/
def validate {d : Nat} (config : FNO.Config d) : Except String Unit := do
  if config.spatial.prod = 0 then
    throw "FNO: spatial grid must contain at least one point"
  if config.width = 0 then
    throw "FNO: width must be positive"

end FNO.Config

/-- Scalar-field input shape with an arbitrary batch shape. -/
abbrev FNO.Config.inputShape {d : Nat} (config : FNO.Config d)
    (batchShape : Spec.Shape := []) : Spec.Shape :=
  batchShape.concat (config.spatial.to Spec.Shape)

/-- Scalar-field output shape with the same batch shape as the input. -/
abbrev FNO.Config.outputShape {d : Nat} (config : FNO.Config d)
    (batchShape : Spec.Shape := []) : Spec.Shape :=
  batchShape.concat (config.spatial.to Spec.Shape)

/--
Build a multidimensional FNO model, independently of spatial rank and batch shape.

`automatic` uses separable transforms, with cuFFT on supported eager CUDA interpreters and dense
per-axis transforms otherwise. `denseReference` uses the full-grid DFT matrices. Both paths retain
the full-spectrum parameterization, frequency mask, activation, and checkpoint layout.
-/
def fno {d : Nat} (config : FNO.Config d) (batchShape : Spec.Shape := []) :
    nn.Builder (nn.Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Internal.invalidConfiguration
        (config.inputShape batchShape) (config.outputShape batchShape) "FNO" message
  | .ok () =>
      let grid := Runtime.Autograd.Model.Layers.FNO.gridSize config.spatial
      let field := Runtime.Autograd.Model.Layers.FNO.fieldShape config.spatial config.width
      let rec buildBlocks (remaining : Nat) :
          nn.Builder (nn.Sequential field field) :=
        match remaining with
        | 0 => pure <| nn.Sequential.identity field
        | count + 1 =>
            nn.withSeed fun spectralRealSeed =>
              nn.withSeed fun spectralImagSeed =>
                nn.withSeed fun skipWeightSeed => do
                  let rest ← buildBlocks count
                  let current :=
                    Runtime.Autograd.Model.Layers.FNO.block
                      config.spatial config.modes config.width config.activation
                      spectralRealSeed spectralImagSeed skipWeightSeed (path := config.spectralPath)
                  pure <| Runtime.Autograd.Model.Layers.Seq.cons current rest
      nn.withSeed fun liftWeightSeed => do
        let operators ← buildBlocks config.layerCount
        nn.withSeed fun projectWeightSeed =>
          let lift :=
            Runtime.Autograd.Model.Layers.FNO.pointwiseAffine
              grid 1 config.width liftWeightSeed
          let project :=
            Runtime.Autograd.Model.Layers.FNO.pointwiseAffine
              grid config.width 1 projectWeightSeed
          let model :=
            Runtime.Autograd.Model.Layers.Seq.cons
              (Runtime.Autograd.Model.Layers.FNO.Internal.addScalarChannel config.spatial) <|
            Runtime.Autograd.Model.Layers.Seq.cons
              (Runtime.Autograd.Model.Layers.FNO.Internal.flattenSpatial config.spatial) <|
            Runtime.Autograd.Model.Layers.Seq.cons lift <|
            Runtime.Autograd.Model.Layers.Seq.cons
              (Runtime.Autograd.Model.Layers.FNO.Internal.restoreSpatial config.spatial) <|
            Runtime.Autograd.Model.Layers.Seq.comp operators <|
            Runtime.Autograd.Model.Layers.Seq.cons
              (Runtime.Autograd.Model.Layers.FNO.Internal.flattenSpatial config.spatial) <|
            Runtime.Autograd.Model.Layers.Seq.cons project <|
            Runtime.Autograd.Model.Layers.Seq.cons
              (Runtime.Autograd.Model.Layers.FNO.Internal.restoreSpatial config.spatial) <|
            Runtime.Autograd.Model.Layers.Seq.cons
              (Runtime.Autograd.Model.Layers.FNO.Internal.removeScalarChannel config.spatial)
              (nn.Sequential.identity _)
          pure (nn.mapLeading batchShape model)

/-- Configuration for the one-dimensional, one-sided spectral FNO. -/
structure FNO.RFFTConfig where
  /-- Number of spatial samples per scalar field. -/
  grid : Nat
  /-- Number of nonnegative frequencies with learned channel maps, including DC. -/
  modes : Nat
  /-- Number of hidden channels at each spatial sample. -/
  width : Nat
  /-- Number of spectral residual blocks. -/
  layerCount : Nat
  /-- Execution choice; both paths use the same weights and Fourier normalization. -/
  spectralPath : Runtime.Autograd.Model.F.SpectralPath := .automatic
  /-- Activation after each spectral residual block. -/
  activation : Activation.Kind := .relu

/--
Build an FNO whose dense and native paths have identical one-sided spectral semantics.

Parameters are ordered as lift weight/bias, each block's real weight/imaginary weight/skip/bias,
and project weight/bias. This is the layout used by the specialized CUDA model. To compare the
runners, supply the same parameter tensors and loss, rather than assuming equal seeds suffice.
-/
def fnoRfft (config : FNO.RFFTConfig) (batchShape : Spec.Shape := []) :
    nn.Builder (nn.Sequential (batchShape.concat [config.grid])
      (batchShape.concat [config.grid])) :=
  if valid : 0 < config.grid ∧ 0 < config.width ∧ config.modes ≤ config.grid / 2 + 1 then
    let spatial : Tensor Nat [1] := [config.grid]
    let rec buildBlocks (remaining : Nat) :
        nn.Builder (nn.Sequential [config.grid, config.width] [config.grid, config.width]) :=
      match remaining with
      | 0 => pure <| nn.Sequential.identity _
      | count + 1 =>
          nn.withSeed fun realSeed =>
            nn.withSeed fun imagSeed =>
              nn.withSeed fun skipSeed => do
                let rest ← buildBlocks count
                let block := Runtime.Autograd.Model.Layers.FNO.rfftBlock
                  config.grid config.width config.modes valid.1 valid.2.1 valid.2.2
                  config.spectralPath config.activation realSeed imagSeed skipSeed
                pure <| Runtime.Autograd.Model.Layers.Seq.cons block rest
    nn.withSeed fun liftSeed => do
      let blocks ← buildBlocks config.layerCount
      nn.withSeed fun projectSeed =>
        let model :=
          Runtime.Autograd.Model.Layers.Seq.cons
            (Runtime.Autograd.Model.Layers.FNO.Internal.addScalarChannel spatial) <|
          Runtime.Autograd.Model.Layers.Seq.cons
            (Runtime.Autograd.Model.Layers.FNO.pointwiseAffine
              config.grid 1 config.width liftSeed) <|
          Runtime.Autograd.Model.Layers.Seq.comp blocks <|
          Runtime.Autograd.Model.Layers.Seq.cons
            (Runtime.Autograd.Model.Layers.FNO.pointwiseAffine
              config.grid config.width 1 projectSeed) <|
          Runtime.Autograd.Model.Layers.Seq.cons
            (Runtime.Autograd.Model.Layers.FNO.Internal.removeScalarChannel spatial)
            (nn.Sequential.identity _)
        pure (nn.mapLeading batchShape model)
  else
    pure <| nn.Internal.invalidConfiguration
      (batchShape.concat [config.grid]) (batchShape.concat [config.grid]) "FNO RFFT"
      "FNO RFFT: grid and width must be positive, and modes must be at most grid / 2 + 1"

end TorchLean.nn.models
