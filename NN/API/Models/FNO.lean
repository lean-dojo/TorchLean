/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.Runtime.Autograd.Model.Fno

/-!
# Fourier Neural Operators

`fno` is polymorphic in spatial rank and uses separable FFTs or a dense multidimensional DFT with
the same real/imaginary weights.

Reference: Zongyi Li et al., *Fourier Neural Operator for Parametric Partial Differential
Equations*, ICLR 2021.
-/

@[expose] public section

namespace TorchLean.nn.models

/-- Configuration for an FNO over `d` spatial axes with tensor-valued fields. -/
structure FNO.Config (d : Nat) where
  /-- Size of each sampled grid axis. -/
  spatial : Tensor Nat [d]
  /--
  Width of the low- and high-frequency bands retained along each full-DFT axis.

  The bands use ordinary FFT indexing: coordinates below `modes` and coordinates at least
  `spatial - modes` are retained. Overlapping bands retain the entire axis.
  -/
  modes : Tensor Nat [d]
  /-- Feature axes at each input grid point; `[]` is a scalar field. -/
  inputFeatures : Spec.Shape := []
  /-- Feature axes at each output grid point; `[]` is a scalar field. -/
  outputFeatures : Spec.Shape := []
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
  if config.inputFeatures.size = 0 then
    throw "FNO: input features must contain at least one value"
  if config.outputFeatures.size = 0 then
    throw "FNO: output features must contain at least one value"

end FNO.Config

/-- Batch axes followed by spatial axes and input feature axes. -/
abbrev FNO.Config.inputShape {d : Nat} (config : FNO.Config d)
    (batchShape : Spec.Shape := []) : Spec.Shape :=
  batchShape.concat ((config.spatial.to Spec.Shape).concat config.inputFeatures)

/-- Batch axes followed by spatial axes and output feature axes. -/
abbrev FNO.Config.outputShape {d : Nat} (config : FNO.Config d)
    (batchShape : Spec.Shape := []) : Spec.Shape :=
  batchShape.concat ((config.spatial.to Spec.Shape).concat config.outputFeatures)

/--
Build a multidimensional FNO model, independently of spatial rank, feature shape and batch shape.

The pointwise lift and projection flatten only the feature axes. Spectral blocks still transform
the spatial axes separately. Empty feature shapes select scalar fields, preserving the default
parameter layout; feature shapes containing a zero extent are rejected before initialization.

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
            Runtime.Autograd.Model.Layers.FNO.linear
              grid config.inputFeatures.size config.width liftWeightSeed
          let project :=
            Runtime.Autograd.Model.Layers.FNO.linear
              grid config.width config.outputFeatures.size projectWeightSeed
          let model :=
            Runtime.Autograd.Model.Layers.Seq.comp
              (nn.Impl.reshape (config.inputShape []) [grid, config.inputFeatures.size]) <|
            Runtime.Autograd.Model.Layers.Seq.cons lift <|
            Runtime.Autograd.Model.Layers.Seq.cons
              (Runtime.Autograd.Model.Layers.FNO.Internal.restoreSpatial config.spatial) <|
            Runtime.Autograd.Model.Layers.Seq.comp operators <|
            Runtime.Autograd.Model.Layers.Seq.cons
              (Runtime.Autograd.Model.Layers.FNO.Internal.flattenSpatial config.spatial) <|
            Runtime.Autograd.Model.Layers.Seq.cons project <|
              nn.Impl.reshape [grid, config.outputFeatures.size] (config.outputShape [])
          pure (nn.mapLeading batchShape model)

end TorchLean.nn.models
