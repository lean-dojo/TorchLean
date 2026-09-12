/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Layers.Core
import Mathlib.Algebra.Order.Algebra

/-!
# Fourier Neural Operators over Arbitrary Spatial Rank

This module implements a dense, correctness-oriented Fourier layer over any finite collection of
spatial axes. The transform phase is the sum of the per-axis phases, so this is the tensor-product
multidimensional DFT rather than a one-dimensional DFT of flattened storage. Real and imaginary
parts are represented by separate tensors, allowing the model to run over ordinary real scalar
backends. The spectral linear map is applied independently at every retained frequency.
-/

@[expose] public section

namespace Runtime.Autograd.Model.Layers.FNO

open Spec TorchLean TorchLean.Tensor

/-- Shape of an `m × n` matrix. -/
abbrev matrixShape (m n : Nat) : Shape := [m, n]

/-- Shape of a vector with `n` entries. -/
abbrev vectorShape (n : Nat) : Shape := [n]

/-- Reshape `[modes, width]` for mode-wise batched matrix multiplication. -/
def reshapeModesForMatmul (modes width : Nat) :
    Layer [modes, width] [modes, 1, width] :=
  let source : Shape := [modes, width]
  let target : Shape := [modes, 1, width]
  have sameSize : Shape.size source = Shape.size target := by
    simp [source, target, Shape.size]
  { stateShapes := []
    initState := .nil
    requiresGrad := #[]
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      Runtime.Autograd.Model.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

/-- Restore the matrix view produced by `reshapeModesForMatmul`. -/
def restoreModesAfterMatmul (modes width : Nat) :
    Layer [modes, 1, width] [modes, width] :=
  let source : Shape := [modes, 1, width]
  let target : Shape := [modes, width]
  have sameSize : Shape.size source = Shape.size target := by
    simp [source, target, Shape.size]
  { stateShapes := []
    initState := .nil
    requiresGrad := #[]
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      Runtime.Autograd.Model.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

/-- Pointwise affine channel map over a flattened spatial grid. -/
def pointwiseAffine (grid inChannels outChannels : Nat) (weightSeed : Nat := 0) :
    Layer [grid, inChannels] [grid, outChannels] :=
  let weightShape : Shape := [inChannels, outChannels]
  let biasShape : Shape := [outChannels]
  let weight : Tensor Float weightShape :=
    Torch.Init.tensor (s := weightShape) (sch := .uniform (-0.1) 0.1) (seed := weightSeed)
  let bias : Tensor Float biasShape := Tensor.zeros (α := Float) biasShape
  { kind := s!"PointwiseLinear({inChannels}, {outChannels})"
    stateShapes := [weightShape, biasShape]
    initState := .cons weight (.cons bias .nil)
    runtimeInit := some (.cons (.uniform (-0.1) 0.1 weightSeed) (.cons .zeros .nil))
    requiresGrad := #[true, true]
    validateConfig := do
      if grid = 0 then
        throw "FNO: spatial grid must contain at least one point"
      if inChannels = 0 then
        throw "FNO: input width must be positive"
      if outChannels = 0 then
        throw "FNO: output width must be positive"
    forward := fun _ {α} _ _ => fun {m} _ _ => fun w b x =>
      (show m (RefTy (m := m) (α := α) [grid, outChannels]) from do
        let y ← Runtime.Autograd.Model.matmul (m := m) (α := α)
          (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
          (mDim := grid) (nDim := inChannels) (pDim := outChannels) x w
        let expandedBias ← Runtime.Autograd.Model.broadcastTo (m := m) (α := α)
          (s₁ := biasShape) (s₂ := [grid, outChannels]) Shape.BroadcastTo.proof b
        Runtime.Autograd.Model.add (m := m) (α := α) (s := [grid, outChannels]) y expandedBias) }

/-- Tensor shape `spatial... × channels`. -/
abbrev fieldShape {d : Nat} (spatial : TorchLean.Tensor Nat [d]) (channels : Nat) : Shape :=
  (spatial.to Shape).appendDim channels

/-- Tensor shape of a scalar field over the spatial grid. -/
abbrev scalarFieldShape {d : Nat} (spatial : TorchLean.Tensor Nat [d]) : Shape :=
  spatial.to Shape

/-- Number of spatial grid points. -/
def gridSize {d : Nat} (spatial : TorchLean.Tensor Nat [d]) : Nat :=
  spatial.prod

/-- Matrix view that flattens the spatial axes and preserves the channel axis. -/
abbrev flatFieldShape {d : Nat} (spatial : TorchLean.Tensor Nat [d]) (channels : Nat) : Shape :=
  [gridSize spatial, channels]

/-- Learned frequency-wise channel maps, one `channels × channels` matrix per grid frequency. -/
abbrev spectralWeightShape {d : Nat} (spatial : TorchLean.Tensor Nat [d]) (channels : Nat) :
    Shape :=
  [gridSize spatial, channels, channels]

namespace Internal

/-- Decode a row-major flat index into coordinates for the given axis extents. -/
def coordinates : List Nat → Nat → List Nat
  | .nil, _ => []
  | .cons _ extents, flat =>
      let stride := extents.prod
      let coordinate := if stride = 0 then 0 else flat / stride
      let remainder := if stride = 0 then 0 else flat % stride
      coordinate :: coordinates extents remainder

/-- Sum `inputᵢ * frequencyᵢ / extentᵢ` over matching spatial coordinates. -/
def phaseFraction {α : Type} [TorchLean.Storage α] [Context α] : List Nat → List Nat → List Nat → α
  | .cons extent extents, .cons input inputs, .cons frequency frequencies =>
      (input : α) * (frequency : α) / (extent : α) +
        phaseFraction extents inputs frequencies
  | _, _, _ => 0

/-- Tensor-product DFT phase for two flattened spatial coordinates. -/
def phase {α : Type} [TorchLean.Storage α] [Context α] (extents : List Nat)
    (input frequency : Nat) : α :=
  phaseFraction extents (coordinates extents input) (coordinates extents frequency)

/-- Cosine part of the dense multidimensional DFT matrix. -/
def dftCosMatrix {α : Type} [TorchLean.Storage α] [Context α] {d : Nat}
    (spatial : TorchLean.Tensor Nat [d]) :
    Tensor α (matrixShape (gridSize spatial) (gridSize spatial)) :=
  Tensor.dim (fun frequency =>
    Tensor.dim (fun input =>
      let angle := 2 * MathFunctions.pi *
        phase (α := α) (spatial.to (List Nat)) input.val frequency.val
      Tensor.scalar (MathFunctions.cos angle)))

/-- Negative-sine part of the dense multidimensional DFT matrix. -/
def dftNegSinMatrix {α : Type} [TorchLean.Storage α] [Context α] {d : Nat}
    (spatial : TorchLean.Tensor Nat [d]) :
    Tensor α (matrixShape (gridSize spatial) (gridSize spatial)) :=
  Tensor.dim (fun frequency =>
    Tensor.dim (fun input =>
      let angle := 2 * MathFunctions.pi *
        phase (α := α) (spatial.to (List Nat)) input.val frequency.val
      Tensor.scalar (0 - MathFunctions.sin angle)))

/-- Normalized cosine part of the dense multidimensional inverse DFT matrix. -/
def idftCosMatrix {α : Type} [TorchLean.Storage α] [Context α] {d : Nat}
    (spatial : TorchLean.Tensor Nat [d]) :
    Tensor α (matrixShape (gridSize spatial) (gridSize spatial)) :=
  Tensor.dim (fun input =>
    Tensor.dim (fun frequency =>
      let angle := 2 * MathFunctions.pi *
        phase (α := α) (spatial.to (List Nat)) input.val frequency.val
      Tensor.scalar (MathFunctions.cos angle / (gridSize spatial : α))))

/-- Normalized sine part of the dense multidimensional inverse DFT matrix. -/
def idftSinMatrix {α : Type} [TorchLean.Storage α] [Context α] {d : Nat}
    (spatial : TorchLean.Tensor Nat [d]) :
    Tensor α (matrixShape (gridSize spatial) (gridSize spatial)) :=
  Tensor.dim (fun input =>
    Tensor.dim (fun frequency =>
      let angle := 2 * MathFunctions.pi *
        phase (α := α) (spatial.to (List Nat)) input.val frequency.val
      Tensor.scalar (MathFunctions.sin angle / (gridSize spatial : α))))

/-- Whether one coordinate lies in the retained low- or high-frequency bands. -/
def keepCoordinate (extent modes coordinate : Nat) : Bool :=
  coordinate < modes || extent - modes ≤ coordinate

/--
Test membership in the rectangular frequency set selected by `modes`.

The flat index is decoded in the same outermost-first row-major order used by `Shape.ofList`.
-/
def keepFrequency : List Nat → List Nat → Nat → Bool
  | .nil, .nil, _ => true
  | .cons extent extents, .cons modes remainingModes, flat =>
      let stride := extents.prod
      let coordinate := if stride = 0 then 0 else flat / stride
      let remainder := if stride = 0 then 0 else flat % stride
      keepCoordinate extent modes coordinate && keepFrequency extents remainingModes remainder
  | _, _, _ => false

/-- Pointwise mask for the retained multidimensional Fourier modes. -/
def frequencyMask {α : Type} [TorchLean.Storage α] [Context α] {d channels : Nat}
    (spatial modes : TorchLean.Tensor Nat [d]) : Tensor α (flatFieldShape spatial channels) :=
  Tensor.dim (fun frequency =>
    Tensor.dim (fun _ =>
      Tensor.scalar
        (if keepFrequency (spatial.to (List Nat)) (modes.to (List Nat)) frequency.val then 1
        else 0)))

/-- Reshape a spatial field to its matrix view. -/
def flattenSpatial {d channels : Nat} (spatial : TorchLean.Tensor Nat [d]) :
    Layer (fieldShape spatial channels) (flatFieldShape spatial channels) :=
  let source : Shape := fieldShape spatial channels
  let target : Shape := flatFieldShape spatial channels
  have sameSize : Shape.size source = Shape.size target := by
    show Shape.size ((spatial.to Shape).appendDim channels)
      = Shape.size [gridSize spatial, channels]
    -- The spatial vector reads out as a shape whose size is its product, which is `gridSize`.
    rw [Shape.size_appendDim, Tensor.size_to_shape]
    simp [gridSize, Shape.size]
  { kind := "ReshapeSpatial"
    stateShapes := []
    initState := .nil
    requiresGrad := #[]
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      Runtime.Autograd.Model.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

/-- Restore a matrix view to its spatial axes. -/
def restoreSpatial {d channels : Nat} (spatial : TorchLean.Tensor Nat [d]) :
    Layer (flatFieldShape spatial channels) (fieldShape spatial channels) :=
  let source : Shape := flatFieldShape spatial channels
  let target : Shape := fieldShape spatial channels
  have sameSize : Shape.size source = Shape.size target := by
    show Shape.size [gridSize spatial, channels]
      = Shape.size ((spatial.to Shape).appendDim channels)
    rw [Shape.size_appendDim, Tensor.size_to_shape]
    simp [gridSize, Shape.size]
  { kind := "RestoreSpatial"
    stateShapes := []
    initState := .nil
    requiresGrad := #[]
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      Runtime.Autograd.Model.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

/-- Add the singleton channel axis used inside an FNO model. -/
def addScalarChannel {d : Nat} (spatial : TorchLean.Tensor Nat [d]) :
    Layer (scalarFieldShape spatial) (fieldShape spatial 1) :=
  let source : Shape := scalarFieldShape spatial
  let target : Shape := fieldShape spatial 1
  have sameSize : Shape.size source = Shape.size target := by
    simp [source, target, fieldShape, Shape.size_appendDim]
  { kind := "AddScalarChannel"
    stateShapes := []
    initState := .nil
    requiresGrad := #[]
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      Runtime.Autograd.Model.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

/-- Remove the singleton channel axis after the output projection. -/
def removeScalarChannel {d : Nat} (spatial : TorchLean.Tensor Nat [d]) :
    Layer (fieldShape spatial 1) (scalarFieldShape spatial) :=
  let source : Shape := fieldShape spatial 1
  let target : Shape := scalarFieldShape spatial
  have sameSize : Shape.size source = Shape.size target := by
    simp [source, target, fieldShape, Shape.size_appendDim]
  { kind := "RemoveScalarChannel"
    stateShapes := []
    initState := .nil
    requiresGrad := #[]
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      Runtime.Autograd.Model.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

end Internal

/--
One multidimensional FNO block.

The first `d` axes are transformed. A learned channel map is applied at every retained frequency,
the discarded frequency rectangle is set to zero, and a pointwise affine skip is added after the
inverse transform.
-/
def block {d : Nat} (spatial modes : TorchLean.Tensor Nat [d]) (width : Nat)
    (activation : Activation.Kind := .tanh)
    (spectralRealSeed spectralImagSeed skipWeightSeed : Nat := 0) :
    Layer (fieldShape spatial width) (fieldShape spatial width) :=
  let grid := gridSize spatial
  let field : Shape := fieldShape spatial width
  let flat : Shape := flatFieldShape spatial width
  let spectralShape : Shape := spectralWeightShape spatial width
  let skipShape : Shape := matrixShape width width
  let biasShape : Shape := vectorShape width
  let spectralReal0 : Tensor Float spectralShape :=
    Torch.Init.tensor (s := spectralShape) (sch := .uniform (-0.05) 0.05)
      (seed := spectralRealSeed)
  let spectralImag0 : Tensor Float spectralShape :=
    Torch.Init.tensor (s := spectralShape) (sch := .uniform (-0.05) 0.05)
      (seed := spectralImagSeed)
  let skip0 : Tensor Float skipShape :=
    Torch.Init.tensor (s := skipShape) (sch := .uniform (-0.05) 0.05)
      (seed := skipWeightSeed)
  let bias0 : Tensor Float biasShape := Tensor.zeros (α := Float) biasShape
  { kind := "FNOBlock"
    stateShapes := [spectralShape, spectralShape, skipShape, biasShape]
    initState := .cons spectralReal0 (.cons spectralImag0 (.cons skip0 (.cons bias0 .nil)))
    runtimeInit := some <| .cons (.uniform (-0.05) 0.05 spectralRealSeed) <|
      .cons (.uniform (-0.05) 0.05 spectralImagSeed) <|
      .cons (.uniform (-0.05) 0.05 skipWeightSeed) <| .cons .zeros .nil
    requiresGrad := #[true, true, true, true]
    validateConfig := do
      if grid = 0 then
        throw "FNO: spatial grid must contain at least one point"
      if width = 0 then
        throw "FNO: width must be positive"
    forward := fun mode {α} _ _ => fun {m} _ _ => fun spectralReal spectralImag skip bias x =>
      (show m (RefTy (m := m) (α := α) field) from do
        let xMatrix ← (Internal.flattenSpatial spatial).forward mode (α := α) (m := m) x
        let transformShape : Shape := matrixShape grid grid
        let cosRef ← Runtime.Autograd.Model.const (m := m) (α := α) (s := transformShape)
          (Internal.dftCosMatrix (α := α) spatial)
        let negSinRef ← Runtime.Autograd.Model.const (m := m) (α := α) (s := transformShape)
          (Internal.dftNegSinMatrix (α := α) spatial)
        let inverseCosRef ← Runtime.Autograd.Model.const (m := m) (α := α) (s := transformShape)
          (Internal.idftCosMatrix (α := α) spatial)
        let inverseSinRef ← Runtime.Autograd.Model.const (m := m) (α := α) (s := transformShape)
          (Internal.idftSinMatrix (α := α) spatial)
        let xReal ← Runtime.Autograd.Model.matmul (m := m) (α := α)
          (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
          (mDim := grid) (nDim := grid) (pDim := width) cosRef xMatrix
        let xImag ← Runtime.Autograd.Model.matmul (m := m) (α := α)
          (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
          (mDim := grid) (nDim := grid) (pDim := width) negSinRef xMatrix
        let xRealBatched ← (reshapeModesForMatmul grid width).forward mode
          (α := α) (m := m) xReal
        let xImagBatched ← (reshapeModesForMatmul grid width).forward mode
          (α := α) (m := m) xImag
        let realReal ← Runtime.Autograd.Model.matmul (m := m) (α := α)
          (batchA := [grid]) (batchB := [grid]) (batch := [grid])
          (mDim := 1) (nDim := width) (pDim := width) xRealBatched spectralReal
        let imagImag ← Runtime.Autograd.Model.matmul (m := m) (α := α)
          (batchA := [grid]) (batchB := [grid]) (batch := [grid])
          (mDim := 1) (nDim := width) (pDim := width) xImagBatched spectralImag
        let realImag ← Runtime.Autograd.Model.matmul (m := m) (α := α)
          (batchA := [grid]) (batchB := [grid]) (batch := [grid])
          (mDim := 1) (nDim := width) (pDim := width) xRealBatched spectralImag
        let imagReal ← Runtime.Autograd.Model.matmul (m := m) (α := α)
          (batchA := [grid]) (batchB := [grid]) (batch := [grid])
          (mDim := 1) (nDim := width) (pDim := width) xImagBatched spectralReal
        let transformedRealBatched ← Runtime.Autograd.Model.sub (m := m) (α := α)
          (s := [grid, 1, width]) realReal imagImag
        let transformedImagBatched ← Runtime.Autograd.Model.add (m := m) (α := α)
          (s := [grid, 1, width]) realImag imagReal
        let transformedReal ← (restoreModesAfterMatmul grid width).forward mode
          (α := α) (m := m) transformedRealBatched
        let transformedImag ← (restoreModesAfterMatmul grid width).forward mode
          (α := α) (m := m) transformedImagBatched
        let mask : Tensor α flat := Internal.frequencyMask (channels := width) spatial modes
        let maskRef ← Runtime.Autograd.Model.const (m := m) (α := α) (s := flat) mask
        let retainedReal ← Runtime.Autograd.Model.mul (m := m) (α := α) (s := flat)
          transformedReal maskRef
        let retainedImag ← Runtime.Autograd.Model.mul (m := m) (α := α) (s := flat)
          transformedImag maskRef
        let inverseReal ← Runtime.Autograd.Model.matmul (m := m) (α := α)
          (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
          (mDim := grid) (nDim := grid) (pDim := width) inverseCosRef retainedReal
        let inverseImag ← Runtime.Autograd.Model.matmul (m := m) (α := α)
          (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
          (mDim := grid) (nDim := grid) (pDim := width) inverseSinRef retainedImag
        let spectralMatrix ← Runtime.Autograd.Model.sub (m := m) (α := α) (s := flat)
          inverseReal inverseImag
        let spectralResult ← (Internal.restoreSpatial spatial).forward mode
          (α := α) (m := m) spectralMatrix
        let skipMatrix ← Runtime.Autograd.Model.matmul (m := m) (α := α)
          (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
          (mDim := grid) (nDim := width) (pDim := width) xMatrix skip
        let biasBroadcast ← Runtime.Autograd.Model.broadcastTo (m := m) (α := α)
          (s₁ := biasShape) (s₂ := flat) Shape.BroadcastTo.proof bias
        let skipBiased ← Runtime.Autograd.Model.add (m := m) (α := α) (s := flat)
          skipMatrix biasBroadcast
        let skipField ← (Internal.restoreSpatial spatial).forward mode
          (α := α) (m := m) skipBiased
        let y ← Runtime.Autograd.Model.add (m := m) (α := α) (s := field) spectralResult skipField
        match activation with
        | .tanh => Runtime.Autograd.Model.tanh (m := m) (α := α) (s := field) y
        | .relu => Runtime.Autograd.Model.relu (m := m) (α := α) (s := field) y
        | .gelu => Runtime.Autograd.Torch.gelu (m := m) (α := α) (s := field) y
        | .silu => Runtime.Autograd.Torch.silu (m := m) (α := α) (s := field) y
        | .sigmoid => Runtime.Autograd.Model.sigmoid (m := m) (α := α) (s := field) y) }

end Runtime.Autograd.Model.Layers.FNO
