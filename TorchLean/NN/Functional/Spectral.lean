/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Functional.Fourier

/-!
# One-sided spectral convolution

Both execution paths use the same weights, indexed by `[frequency, input channel, output channel]`,
and only the stored nonnegative frequencies are learned. The normalized inverse supplies the
conjugate negative frequencies and, in exact arithmetic, ignores the imaginary DC and Nyquist
coordinates.

The dense path is a complete implementation of its own, so the two paths can be compared: changing
`SpectralPath` changes the implementation and keeps the parameters and the real-valued spectral
map. Floating-point results can differ, and a NaN or an infinity in a coordinate that the dense
path multiplies by zero reaches the output as NaN.
-/

@[expose] public section

namespace TorchLean.nn.functional

open Ops

variable {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

namespace Fourier

/-- Append `frequencies - modes` zero rows for the discarded frequencies before the inverse. -/
def padFrequencies {modes frequencies width : Nat} (h : modes ≤ frequencies)
    (x : Ref (m := m) (α := α) [modes, width]) :
    m (Ref (m := m) (α := α) [frequencies, width]) := do
  let zeros ← Ops.const (Tensor.zeros (α := α) [frequencies - modes, width])
  let padded : Ref (m := m) (α := α) [modes + (frequencies - modes), width] ←
    Ops.apply
      (.cat (s := .dim 0 [width]) ⟨0, by simp⟩ [modes, frequencies - modes]) (x, zeros, ())
  pure (by simpa [Nat.add_sub_of_le h] using padded)

end Fourier

/--
Apply learned channel maps to the first `modes` nonnegative Fourier bins of a `[grid, width]` input.

Both weights have shape `[modes, width, width]`, and the four real matrix products compute the
complex product with the weight unconjugated. Padding comes after the product, so discarded
frequencies have no parameters. The arbitrary-rank full-DFT FNO has a different parameterization
and cannot share these weights. `denseReference` and `automatic` share a checkpoint, because only
the transforms select native primitives: the tape composes their packed-real adjoints with the
channel maps and the indexing operations.
-/
def spectralConv {grid width modes : Nat} (_hgrid : 0 < grid)
    (_hwidth : 0 < width) (hmodes : modes ≤ grid / 2 + 1)
    (x : Ref (m := m) (α := α) [grid, width])
    (realWeight imagWeight : Ref (m := m) (α := α) [modes, width, width])
    (path : SpectralPath := .automatic) :
    m (Ref (m := m) (α := α) [grid, width]) := do
  let channels ← Ops.apply
      (.permute (s := [grid, width]) (Shape.adjacentSwap [grid, width] 0)
        (Shape.swapAdjacentAtDepth [grid, width] 0) (by tensor_shape)) (x, ())
  let transformed ← functional.rfft (batch := [width]) _hgrid channels path
  let realChannels ← Ops.apply
      (.indexAt (s := [width, grid / 2 + 1, 2]) ⟨2, Shape.AxisInBounds.proof⟩
        ⟨0, by change 0 < 2; decide⟩) (transformed, ())
  let imagChannels ← Ops.apply
      (.indexAt (s := [width, grid / 2 + 1, 2]) ⟨2, Shape.AxisInBounds.proof⟩
        ⟨1, by change 1 < 2; decide⟩) (transformed, ())
  let realFrequencies ← Ops.apply
      (.permute (s := [width, grid / 2 + 1]) (Shape.adjacentSwap [width, grid / 2 + 1] 0)
        (Shape.swapAdjacentAtDepth [width, grid / 2 + 1] 0) (by tensor_shape)) (realChannels, ())
  let imagFrequencies ← Ops.apply
      (.permute (s := [width, grid / 2 + 1]) (Shape.adjacentSwap [width, grid / 2 + 1] 0)
        (Shape.swapAdjacentAtDepth [width, grid / 2 + 1] 0) (by tensor_shape)) (imagChannels, ())
  let realModes : Ref (m := m) (α := α) [modes, width] ←
      Ref.cast
          (Shape.slice_singleton [grid / 2 + 1, width] ⟨0, by simp⟩ 0 modes) <$>
        Ops.apply
          (.slice (s := [grid / 2 + 1, width]) [⟨0, by simp⟩] [0] [0 + modes] [1]
            (Shape.sliceValid_singleton _ ⟨0, by simp⟩ 0 modes
              (by show 0 + modes ≤ grid / 2 + 1; omega))) (realFrequencies, ())
  let imagModes : Ref (m := m) (α := α) [modes, width] ←
      Ref.cast
          (Shape.slice_singleton [grid / 2 + 1, width] ⟨0, by simp⟩ 0 modes) <$>
        Ops.apply
          (.slice (s := [grid / 2 + 1, width]) [⟨0, by simp⟩] [0] [0 + modes] [1]
            (Shape.sliceValid_singleton _ ⟨0, by simp⟩ 0 modes
              (by show 0 + modes ≤ grid / 2 + 1; omega))) (imagFrequencies, ())
  let realRows ← Ops.apply
      (.reshape (t := [modes, 1, width]) (by simp)) (realModes, ())
  let imagRows ← Ops.apply
      (.reshape (t := [modes, 1, width]) (by simp)) (imagModes, ())
  -- The four products of the rows with the weights, one row matrix per kept mode.
  let product := Op.matmul (batchA := [modes]) (batchB := [modes]) (batch := [modes])
    (m := 1) (n := width) (p := width) false false (fun h => nomatch h) (fun h => nomatch h)
    Shape.BroadcastTo.proof Shape.BroadcastTo.proof
  let rr : Ref (m := m) (α := α) [modes, 1, width] ←
    Ops.apply product (realRows, realWeight, ())
  let ii : Ref (m := m) (α := α) [modes, 1, width] ←
    Ops.apply product (imagRows, imagWeight, ())
  let ri : Ref (m := m) (α := α) [modes, 1, width] ←
    Ops.apply product (realRows, imagWeight, ())
  let ir : Ref (m := m) (α := α) [modes, 1, width] ←
    Ops.apply product (imagRows, realWeight, ())
  let realProduct ← Ops.apply .sub (rr, ii, ())
  let imagProduct ← Ops.apply .add (ri, ir, ())
  let realMatrix ← Ops.apply
      (.reshape (t := [modes, width]) (by simp)) (realProduct, ())
  let imagMatrix ← Ops.apply
      (.reshape (t := [modes, width]) (by simp)) (imagProduct, ())
  let realPadded ← Fourier.padFrequencies hmodes realMatrix
  let imagPadded ← Fourier.padFrequencies hmodes imagMatrix
  let realPlane ← Ops.apply
      (.reshape (t := [1, grid / 2 + 1, width]) (by simp)) (realPadded, ())
  let imagPlane ← Ops.apply
      (.reshape (t := [1, grid / 2 + 1, width]) (by simp)) (imagPadded, ())
  let planes : Ref (m := m) (α := α) [2, grid / 2 + 1, width] ←
    Ops.apply
      (.cat (s := .dim 0 [grid / 2 + 1, width]) ⟨0, by simp⟩ [1, 1]) (realPlane, imagPlane, ())
  let frequencyPlanes ← Ops.apply
      (.permute (s := [2, grid / 2 + 1, width]) (Shape.adjacentSwap [2, grid / 2 + 1, width] 0)
        (Shape.swapAdjacentAtDepth [2, grid / 2 + 1, width] 0) (by tensor_shape)) (planes, ())
  let frequencyChannels ← Ops.apply
      (.permute (s := [grid / 2 + 1, 2, width]) (Shape.adjacentSwap [grid / 2 + 1, 2, width] 1)
        (Shape.swapAdjacentAtDepth [grid / 2 + 1, 2, width] 1) (by tensor_shape))
      (frequencyPlanes, ())
  let packed ← Ops.apply
      (.permute (s := [grid / 2 + 1, width, 2]) (Shape.adjacentSwap [grid / 2 + 1, width, 2] 0)
        (Shape.swapAdjacentAtDepth [grid / 2 + 1, width, 2] 0) (by tensor_shape))
      (frequencyChannels, ())
  let result ← functional.irfft (batch := [width]) _hgrid packed path
  Ops.apply
      (.permute (s := [width, grid]) (Shape.adjacentSwap [width, grid] 0)
        (Shape.swapAdjacentAtDepth [width, grid] 0) (by tensor_shape)) (result, ())

end TorchLean.nn.functional
