/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Layer.Activation
public import TorchLean.NN.Layer
public import TorchLean.NN.Functional.Fourier.Transform

/-!
# Fourier Neural Operators over Arbitrary Spatial Rank

`block` is one Fourier layer over any finite collection of spatial axes. The transform uses
separable per-axis FFTs or a dense reference DFT. Its phase is the sum of the per-axis phases,
which makes it the tensor-product multidimensional DFT of the grid. Real and imaginary parts are
separate tensors, so the model runs over ordinary real scalar backends. The spectral linear map is
applied independently at every retained frequency.

We wrote `block` by hand from Li et al. (2021). Their eq. (2) is the update `v ↦ σ(W v + K v)`,
and eqs. (4) and (5) give `K v = F⁻¹(R · F v)`. As in their discrete case (Section 4), `R` spans
the whole grid, `[spatial.prod, width, width]` here in real and imaginary halves, and only the
corner bands survive the mask. Our bands are one coordinate short of their `Z_kmax`: `modes = 12`
on a 64-point axis keeps coordinates `0..11` and `52..63`, and their set with `k_max,j = 12` also
keeps `12`. A few choices are ours. The skip `W` carries a bias. They impose conjugate symmetry
on `R`, and we keep the real part of the inverse transform. The default activation is `tanh`,
where their experiments use ReLU with batch normalization (Section 5). No theorem or torch test
covers the block.

Reference: Zongyi Li et al. (2021), *Fourier Neural Operator for Parametric Partial Differential
Equations*, ICLR 2021, arXiv:2010.08895, eqs. (2), (4) and (5), Sections 4 and 5.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size)

namespace TorchLean.FNO

open TorchLean.Tensor
open Ops
open nn (Layer)

/-- Pointwise affine channel map over a flattened spatial grid: the matrix product of the input
with a weight of shape `[inChannels, outChannels]` (the transpose of `nn.Layer.linear`'s layout,
initial scheme `.uniform (-0.1) 0.1`) plus a bias broadcast over the grid, which starts at zero. -/
def linear (grid inChannels outChannels : Nat) (weightSeed : Nat := 0) :
    Layer [grid, inChannels] [grid, outChannels] :=
  let weightShape : Shape := [inChannels, outChannels]
  let biasShape : Shape := [outChannels]
  let weight : Tensor Float weightShape :=
    Init.tensor (shape := weightShape) (scheme := .uniform (-0.1) 0.1)
      (seed := weightSeed)
  let bias : Tensor Float biasShape := Tensor.zeros (α := Float) biasShape
  { kind := s!"PointwiseLinear({inChannels}, {outChannels})"
    stateShapes := [weightShape, biasShape]
    initState := .cons weight (.cons bias .nil)
    runtimeInit := some (.cons (.scheme (.uniform (-0.1) 0.1) weightSeed)
      (.cons (.scheme .zeros) .nil))
    requiresGrad := #[true, true]
    validateConfig := do
      if grid = 0 then
        throw "FNO: spatial grid must contain at least one point"
      if inChannels = 0 then
        throw "FNO: input width must be positive"
      if outChannels = 0 then
        throw "FNO: output width must be positive"
    forward := fun _ {α} _ _ => fun {m} _ _ => fun w b x =>
      (show m (Ref (m := m) (α := α) [grid, outChannels]) from do
        let y ← Ops.apply (m := m) (α := α)
            (.matmul (batchA := .scalar) (batchB := .scalar) (batch := .scalar) (m := grid)
              (n := inChannels) (p := outChannels) false false (fun h => nomatch h)
              (fun h => nomatch h) Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (x, w, ())
        let expandedBias ← Ops.apply (m := m) (α := α)
            (.broadcast (s := biasShape) (t := [grid, outChannels]) (Shape.BroadcastTo.proof).dims
              (Shape.BroadcastTo.proof).broadcastDims) (b, ())
        Ops.apply (m := m) (α := α) (.add (s := [grid, outChannels]))
            (y, expandedBias, ())) }

/-- Tensor shape `spatial... × channels`. -/
abbrev fieldShape {d : Nat} (spatial : Tensor Nat [d]) (channels : Nat) : Shape :=
  spatial.to Shape ++ [channels]

/-- Matrix view that flattens the spatial axes and preserves the channel axis. -/
abbrev flatFieldShape {d : Nat} (spatial : Tensor Nat [d]) (channels : Nat) : Shape :=
  [spatial.prod, channels]

/-- A field and its matrix view `flatFieldShape` have the same number of entries. -/
theorem size_fieldShape {d : Nat} (spatial : Tensor Nat [d]) (channels : Nat) :
    Shape.size (fieldShape spatial channels) = Shape.size (flatFieldShape spatial channels) := by
  show Shape.size (spatial.to Shape ++ [channels]) = Shape.size [spatial.prod, channels]
  -- The spatial vector reads out as a shape whose size is its product.
  rw [Shape.size_append_singleton, Tensor.size_to_shape]
  simp

/-- Learned frequency-wise channel maps, one `channels × channels` matrix per grid frequency. -/
abbrev spectralWeightShape {d : Nat} (spatial : Tensor Nat [d]) (channels : Nat) :
    Shape :=
  [spatial.prod, channels, channels]

/-- Decode a row-major flat index into coordinates for the given axis extents. -/
def coordinates : Shape → Nat → Shape
  | .nil, _ => []
  | .cons _ extents, flat =>
      let stride := extents.prod
      let coordinate := if stride = 0 then 0 else flat / stride
      let remainder := if stride = 0 then 0 else flat % stride
      coordinate :: coordinates extents remainder

/-- Sum `inputᵢ * frequencyᵢ / extentᵢ` over matching spatial coordinates. -/
def phaseFraction {α : Type} [Storage α] [Context α] : Shape → Shape → Shape → α
  | .cons extent extents, .cons input inputs, .cons frequency frequencies =>
      (input : α) * (frequency : α) / (extent : α) +
        phaseFraction extents inputs frequencies
  | _, _, _ => 0

/-- Tensor-product DFT phase for two flattened spatial coordinates. -/
def phase {α : Type} [Storage α] [Context α] (extents : Shape)
    (input frequency : Nat) : α :=
  phaseFraction extents (coordinates extents input) (coordinates extents frequency)

/-- Cosine part of the dense multidimensional DFT matrix. -/
def dftCosMatrix {α : Type} [Storage α] [Context α] {d : Nat}
    (spatial : Tensor Nat [d]) :
    Tensor α [spatial.prod, spatial.prod] :=
  Tensor.dim (fun frequency =>
    Tensor.dim (fun input =>
      let angle := 2 * MathFunctions.pi *
        phase (α := α) (spatial.to Shape) input.val frequency.val
      Tensor.scalar (MathFunctions.cos angle)))

/-- Negative-sine part of the dense multidimensional DFT matrix. -/
def dftNegSinMatrix {α : Type} [Storage α] [Context α] {d : Nat}
    (spatial : Tensor Nat [d]) :
    Tensor α [spatial.prod, spatial.prod] :=
  Tensor.dim (fun frequency =>
    Tensor.dim (fun input =>
      let angle := 2 * MathFunctions.pi *
        phase (α := α) (spatial.to Shape) input.val frequency.val
      Tensor.scalar (0 - MathFunctions.sin angle)))

/-- Normalized cosine part of the dense multidimensional inverse DFT matrix. -/
def idftCosMatrix {α : Type} [Storage α] [Context α] {d : Nat}
    (spatial : Tensor Nat [d]) :
    Tensor α [spatial.prod, spatial.prod] :=
  Tensor.dim (fun input =>
    Tensor.dim (fun frequency =>
      let angle := 2 * MathFunctions.pi *
        phase (α := α) (spatial.to Shape) input.val frequency.val
      Tensor.scalar (MathFunctions.cos angle / (spatial.prod : α))))

/-- Normalized sine part of the dense multidimensional inverse DFT matrix. -/
def idftSinMatrix {α : Type} [Storage α] [Context α] {d : Nat}
    (spatial : Tensor Nat [d]) :
    Tensor α [spatial.prod, spatial.prod] :=
  Tensor.dim (fun input =>
    Tensor.dim (fun frequency =>
      let angle := 2 * MathFunctions.pi *
        phase (α := α) (spatial.to Shape) input.val frequency.val
      Tensor.scalar (MathFunctions.sin angle / (spatial.prod : α))))

/-- Whether one coordinate lies in the retained low- or high-frequency bands. -/
def keepCoordinate (extent modes coordinate : Nat) : Bool :=
  coordinate < modes || extent - modes ≤ coordinate

/--
Test membership in the rectangular frequency set selected by `modes`.

The flat index is decoded in the outermost-first row-major order that tensors use.
-/
def keepFrequency : Shape → Shape → Nat → Bool
  | .nil, .nil, _ => true
  | .cons extent extents, .cons modes remainingModes, flat =>
      let stride := extents.prod
      let coordinate := if stride = 0 then 0 else flat / stride
      let remainder := if stride = 0 then 0 else flat % stride
      keepCoordinate extent modes coordinate && keepFrequency extents remainingModes remainder
  | _, _, _ => false

/-- Pointwise mask for the retained multidimensional Fourier modes. -/
def frequencyMask {α : Type} [Storage α] [Context α] {d channels : Nat}
    (spatial modes : Tensor Nat [d]) : Tensor α (flatFieldShape spatial channels) :=
  Tensor.dim (fun frequency =>
    Tensor.dim (fun _ =>
      Tensor.scalar
        (if keepFrequency (spatial.to Shape) (modes.to Shape) frequency.val then 1
        else 0)))

/--
One multidimensional FNO block.

The first `d` axes are transformed. A learned channel map is applied at every retained frequency,
the discarded frequencies are multiplied by a zero mask, and a pointwise affine skip is added
after the inverse transform. The mask zeroes finite values, and a NaN or infinite coefficient
times `0` gives NaN.
-/
def block {d : Nat} (spatial modes : Tensor Nat [d]) (width : Nat)
    (activation : Activation.Kind := .tanh)
    (spectralRealSeed spectralImagSeed skipWeightSeed : Nat := 0)
    (path : nn.functional.SpectralPath := .automatic) :
    Layer (fieldShape spatial width) (fieldShape spatial width) :=
  let grid := spatial.prod
  let field : Shape := fieldShape spatial width
  let flat : Shape := flatFieldShape spatial width
  let spectralShape : Shape := spectralWeightShape spatial width
  let skipShape : Shape := [width, width]
  let biasShape : Shape := [width]
  let spectralReal0 : Tensor Float spectralShape :=
    Init.tensor (shape := spectralShape) (scheme := .uniform (-0.05) 0.05)
      (seed := spectralRealSeed)
  let spectralImag0 : Tensor Float spectralShape :=
    Init.tensor (shape := spectralShape) (scheme := .uniform (-0.05) 0.05)
      (seed := spectralImagSeed)
  let skip0 : Tensor Float skipShape :=
    Init.tensor (shape := skipShape) (scheme := .uniform (-0.05) 0.05)
      (seed := skipWeightSeed)
  let bias0 : Tensor Float biasShape := Tensor.zeros (α := Float) biasShape
  { kind := "FNOBlock"
    stateShapes := [spectralShape, spectralShape, skipShape, biasShape]
    initState := .cons spectralReal0 (.cons spectralImag0 (.cons skip0 (.cons bias0 .nil)))
    runtimeInit := some <| .cons (.scheme (.uniform (-0.05) 0.05) spectralRealSeed) <|
      .cons (.scheme (.uniform (-0.05) 0.05) spectralImagSeed) <|
      .cons (.scheme (.uniform (-0.05) 0.05) skipWeightSeed) <| .cons (.scheme .zeros) .nil
    requiresGrad := #[true, true, true, true]
    validateConfig := do
      if grid = 0 then
        throw "FNO: spatial grid must contain at least one point"
      if width = 0 then
        throw "FNO: width must be positive"
    forward := fun mode {α} _ _ => fun {m} _ _ => fun spectralReal spectralImag skip bias x =>
      (show m (Ref (m := m) (α := α) field) from do
        let xMatrix ← Ops.apply (m := m) (α := α)
          (.reshape (size_fieldShape spatial width)) (x, ())
        have grid_eq : grid = (spatial.to Shape).prod := Tensor.prod_eq_to_list_prod spatial
        let (xReal, xImag) ← if positive : 0 < grid ∧ path = .automatic then do
          let zero ← Ops.const (Tensor.zeros (α := α) flat)
          let transformed ← nn.functional.fft (α := α) (m := m) (channels := width)
            (spatial.to Shape) (by simpa only [← grid_eq] using positive.1)
            (by simpa only [← grid_eq] using xMatrix) (by simpa only [← grid_eq] using zero)
          pure (by simpa only [← grid_eq] using transformed)
        else do
          let cosRef ← Ops.const (dftCosMatrix (α := α) spatial)
          let negSinRef ← Ops.const (dftNegSinMatrix (α := α) spatial)
          let realPart : Ref (m := m) (α := α) [grid, width] ←
            Ops.apply
              (.matmul (batchA := []) (batchB := []) (batch := []) false false (fun h => nomatch h)
                (fun h => nomatch h) Shape.BroadcastTo.proof Shape.BroadcastTo.proof)
              (cosRef, xMatrix, ())
          let imagPart : Ref (m := m) (α := α) [grid, width] ←
            Ops.apply
              (.matmul (batchA := []) (batchB := []) (batch := []) false false (fun h => nomatch h)
                (fun h => nomatch h) Shape.BroadcastTo.proof Shape.BroadcastTo.proof)
              (negSinRef, xMatrix, ())
          pure (realPart, imagPart)
        let xRealBatched ← Ops.apply
            (.reshape (s := [grid, width]) (t := [grid, 1, width]) (by simp))
            (xReal, ())
        let xImagBatched ← Ops.apply
            (.reshape (s := [grid, width]) (t := [grid, 1, width]) (by simp))
            (xImag, ())
        let realReal ← Ops.apply (m := m) (α := α)
            (.matmul (batchA := [grid]) (batchB := [grid]) (batch := [grid]) (m := 1) (n := width)
              (p := width) false false (fun h => nomatch h) (fun h => nomatch h)
              Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (xRealBatched, spectralReal, ())
        let imagImag ← Ops.apply (m := m) (α := α)
            (.matmul (batchA := [grid]) (batchB := [grid]) (batch := [grid]) (m := 1) (n := width)
              (p := width) false false (fun h => nomatch h) (fun h => nomatch h)
              Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (xImagBatched, spectralImag, ())
        let realImag ← Ops.apply (m := m) (α := α)
            (.matmul (batchA := [grid]) (batchB := [grid]) (batch := [grid]) (m := 1) (n := width)
              (p := width) false false (fun h => nomatch h) (fun h => nomatch h)
              Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (xRealBatched, spectralImag, ())
        let imagReal ← Ops.apply (m := m) (α := α)
            (.matmul (batchA := [grid]) (batchB := [grid]) (batch := [grid]) (m := 1) (n := width)
              (p := width) false false (fun h => nomatch h) (fun h => nomatch h)
              Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (xImagBatched, spectralReal, ())
        let transformedRealBatched ← Ops.apply (m := m) (α := α)
            (.sub (s := [grid, 1, width])) (realReal, imagImag, ())
        let transformedImagBatched ← Ops.apply (m := m) (α := α)
            (.add (s := [grid, 1, width])) (realImag, imagReal, ())
        let transformedReal ← Ops.apply
            (.reshape (t := [grid, width]) (by simp)) (transformedRealBatched, ())
        let transformedImag ← Ops.apply
            (.reshape (t := [grid, width]) (by simp)) (transformedImagBatched, ())
        let mask : Tensor α flat := frequencyMask (channels := width) spatial modes
        let maskRef ← Ops.const (m := m) (α := α) (s := flat) mask
        let retainedReal ← Ops.apply (m := m) (α := α) (.mul (s := flat))
            (transformedReal, maskRef, ())
        let retainedImag ← Ops.apply (m := m) (α := α) (.mul (s := flat))
            (transformedImag, maskRef, ())
        let spectralMatrix ← if positive : 0 < grid ∧ path = .automatic then do
          let (realPart, _) ← nn.functional.fft (α := α) (m := m) (channels := width)
            (spatial.to Shape)
            (by simpa only [← grid_eq] using positive.1)
            (by simpa only [← grid_eq] using retainedReal)
            (by simpa only [← grid_eq] using retainedImag)
            (inverse := true)
          pure (by simpa only [← grid_eq] using realPart)
        else do
          let inverseCosRef ← Ops.const (idftCosMatrix (α := α) spatial)
          let inverseSinRef ← Ops.const (idftSinMatrix (α := α) spatial)
          let realPart : Ref (m := m) (α := α) [grid, width] ←
            Ops.apply
              (.matmul (batchA := []) (batchB := []) (batch := []) false false (fun h => nomatch h)
                (fun h => nomatch h) Shape.BroadcastTo.proof Shape.BroadcastTo.proof)
              (inverseCosRef, retainedReal, ())
          let imagPart : Ref (m := m) (α := α) [grid, width] ←
            Ops.apply
              (.matmul (batchA := []) (batchB := []) (batch := []) false false (fun h => nomatch h)
                (fun h => nomatch h) Shape.BroadcastTo.proof Shape.BroadcastTo.proof)
              (inverseSinRef, retainedImag, ())
          Ops.apply .sub (realPart, imagPart, ())
        let spectralResult ← Ops.apply (m := m) (α := α)
          (.reshape (size_fieldShape spatial width).symm) (spectralMatrix, ())
        let skipMatrix ← Ops.apply (m := m) (α := α)
            (.matmul (batchA := .scalar) (batchB := .scalar) (batch := .scalar) (m := grid)
              (n := width) (p := width) false false (fun h => nomatch h) (fun h => nomatch h)
              Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (xMatrix, skip, ())
        let biasBroadcast ← Ops.apply (m := m) (α := α)
            (.broadcast (s := biasShape) (t := flat) (Shape.BroadcastTo.proof).dims
              (Shape.BroadcastTo.proof).broadcastDims) (bias, ())
        let skipBiased ← Ops.apply (m := m) (α := α) (.add (s := flat))
            (skipMatrix, biasBroadcast, ())
        let skipField ← Ops.apply (m := m) (α := α)
          (.reshape (size_fieldShape spatial width).symm) (skipBiased, ())
        let y ← Ops.apply (m := m) (α := α) (.add (s := field))
            (spectralResult, skipField, ())
        (Layer.op activation.op activation.need_op).forward mode (α := α) (m := m) y) }

end TorchLean.FNO
