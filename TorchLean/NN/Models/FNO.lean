/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder
public import TorchLean.NN.Models.FNO.Block

/-!
# Fourier Neural Operators

`fno` is polymorphic in spatial rank and uses separable FFTs or a dense multidimensional DFT with
the same real/imaginary weights.

`fno` is laid out as in Li et al. (2021), Section 3: a pointwise lift `P`, a stack of Fourier layers
and a pointwise projection `Q`. Their `P` is usually a shallow fully connected network, and our `P`
and `Q` are single affine maps. Their experiments stack four Fourier layers with `d_v = 64` in 1-d
and `d_v = 32` in 2-d (Section 5), which is `layerCount = 4` with that `width` here. No theorem or
torch test covers `fno`.

Reference: Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik
Bhattacharya, Andrew Stuart, and Anima Anandkumar (2021), *Fourier Neural Operator for Parametric
Partial Differential Equations*, ICLR 2021, arXiv:2010.08895, Sections 3 to 5.
-/

@[expose] public section

open TorchLean.Shape (size size_append)

namespace TorchLean.nn

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
  /-- Feature axes at each input grid point, with `[]` for a scalar field. -/
  inputFeatures : Shape := []
  /-- Feature axes at each output grid point, with `[]` for a scalar field. -/
  outputFeatures : Shape := []
  /-- Width of the latent channel representation. -/
  width : Nat
  /-- Number of spectral residual blocks. -/
  layerCount : Nat
  /-- Activation applied after each spectral residual block. -/
  activation : Activation.Kind := .tanh
  /-- Per-axis FFT execution or the dense full-grid reference, with the same parameters. -/
  spectralPath : functional.SpectralPath := .automatic

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
    (batchShape : Shape := []) : Shape :=
  batchShape ++ ((config.spatial.to Shape) ++ config.inputFeatures)

/-- Batch axes followed by spatial axes and output feature axes. -/
abbrev FNO.Config.outputShape {d : Nat} (config : FNO.Config d)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ ((config.spatial.to Shape) ++ config.outputFeatures)

/-- The input field and its matrix view `[spatial.prod, inputFeatures.size]` have the same
number of entries. -/
theorem FNO.Config.size_inputShape {d : Nat} (config : FNO.Config d) :
    (config.inputShape []).size =
      Shape.size [config.spatial.prod, config.inputFeatures.size] := by
  rw [Shape.size_append, Shape.size_append, Tensor.size_to_shape]
  simp

/-- The output field and its matrix view `[spatial.prod, outputFeatures.size]` have the same
number of entries. -/
theorem FNO.Config.size_outputShape {d : Nat} (config : FNO.Config d) :
    (config.outputShape []).size =
      Shape.size [config.spatial.prod, config.outputFeatures.size] := by
  rw [Shape.size_append, Shape.size_append, Tensor.size_to_shape]
  simp

/--
Build a multidimensional FNO model for any spatial rank, feature shape and batch shape.

The pointwise lift and projection treat the feature axes as one flat channel vector, while the
spectral blocks keep the spatial axes separate. An empty feature shape is a scalar field with the
default parameter layout, and a feature shape with a zero extent is rejected before initialization.

`automatic` uses separable transforms, with cuFFT on supported eager CUDA interpreters and dense
per-axis transforms otherwise. `denseReference` uses the full-grid DFT matrices. Both paths keep
the full-spectrum parameterization, frequency mask, activation and checkpoint layout.
-/
def fno {d : Nat} (config : FNO.Config d) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  match config.validate with
  | .error message =>
      pure <| nn.Sequential.invalid
        (config.inputShape batchShape) (config.outputShape batchShape) "FNO" message
  | .ok () =>
      let grid := config.spatial.prod
      let field := FNO.fieldShape config.spatial config.width
      let rec
        /-- A sequence of `remaining` spectral blocks with their own seeds, the identity for `0`. -/
        buildBlocks (remaining : Nat) :
            nn.Builder (nn.Sequential field field) :=
          match remaining with
          | 0 => pure <| Sequential.id field
          | count + 1 =>
              nn.withSeed fun spectralRealSeed =>
                nn.withSeed fun spectralImagSeed =>
                  nn.withSeed fun skipWeightSeed => do
                    let rest ← buildBlocks count
                    let current :=
                      FNO.block
                        config.spatial config.modes config.width config.activation
                        spectralRealSeed spectralImagSeed skipWeightSeed
                        (path := config.spectralPath)
                    pure <| Sequential.cons current rest
      nn.withSeed fun liftWeightSeed => do
        let operators ← buildBlocks config.layerCount
        nn.withSeed fun projectWeightSeed =>
          let lift :=
            FNO.linear
              grid config.inputFeatures.size config.width liftWeightSeed
          let project :=
            FNO.linear
              grid config.width config.outputFeatures.size projectWeightSeed
          let model :=
            Sequential.comp
              (nn.Sequential.fromLayer (Layer.op
                (.reshape config.size_inputShape))) <|
            Sequential.cons lift <|
            Sequential.cons
              (Layer.op (.reshape
                (FNO.size_fieldShape config.spatial _).symm)) <|
            Sequential.comp operators <|
            Sequential.cons
              (Layer.op (.reshape
                (FNO.size_fieldShape config.spatial _))) <|
            Sequential.cons project <|
              nn.Sequential.fromLayer (Layer.op
                (.reshape config.size_outputShape.symm))
          pure (nn.mapLeading batchShape model)

end TorchLean.nn
