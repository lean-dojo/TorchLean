/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Functional.Fourier

/-!
# One-sided spectral convolution

Both execution paths use the same weights, indexed by `[frequency, input channel, output channel]`.
Only the stored nonnegative frequencies are learned. The normalized inverse supplies the conjugate
negative frequencies and ignores imaginary DC and Nyquist coordinates.

Keeping the dense path explicit makes comparisons meaningful: changing `SpectralPath` changes the
implementation, while preserving the parameters, activation, and function being differentiated.
-/

@[expose] public section

namespace Runtime.Autograd.Model.F

open Spec TorchLean

variable {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

namespace Fourier

/-- Restore discarded frequencies as zeros before the inverse transform. -/
def padFrequencies {modes frequencies width : Nat} (h : modes ≤ frequencies)
    (x : RefTy m α [modes, width]) : m (RefTy m α [frequencies, width]) := do
  let zeros ← const (Tensor.zeros (α := α) [frequencies - modes, width])
  let padded ← concatLeadingAxis x zeros
  pure (by simpa [Nat.add_sub_of_le h] using padded)

/--
Dense reference for the cuFFT spectral primitive, with exactly the same weight layout.

The four real matrix products implement ordinary complex multiplication, without conjugating the
weight. Padding occurs after that multiplication, so discarded frequencies have no parameters.
-/
def spectralConv1dRfftReference {grid width modes : Nat} (hmodes : modes ≤ grid / 2 + 1)
    (x : RefTy m α [grid, width])
    (realWeight imagWeight : RefTy m α [modes, width, width]) :
    m (RefTy m α [grid, width]) := do
  let channels ← swapAdjacentAtDepth 0 x
  let transformed ← rfft1dReference channels
  let realChannels ← select 2 transformed ⟨0, by change 0 < 2; decide⟩
  let imagChannels ← select 2 transformed ⟨1, by change 1 < 2; decide⟩
  let realFrequencies ← swapAdjacentAtDepth 0 realChannels
  let imagFrequencies ← swapAdjacentAtDepth 0 imagChannels
  let realModes ← sliceLeadingAxisRange 0 modes (by omega) realFrequencies
  let imagModes ← sliceLeadingAxisRange 0 modes (by omega) imagFrequencies
  let realRows ← reshape (s₂ := [modes, 1, width]) realModes
    (by simp [Shape.eraseAxis, Shape.size])
  let imagRows ← reshape (s₂ := [modes, 1, width]) imagModes
    (by simp [Shape.eraseAxis, Shape.size])
  let rr ← matmul (batchA := [modes]) (batchB := [modes]) (batch := [modes])
    realRows realWeight
  let ii ← matmul (batchA := [modes]) (batchB := [modes]) (batch := [modes])
    imagRows imagWeight
  let ri ← matmul (batchA := [modes]) (batchB := [modes]) (batch := [modes])
    realRows imagWeight
  let ir ← matmul (batchA := [modes]) (batchB := [modes]) (batch := [modes])
    imagRows realWeight
  let realProduct ← sub rr ii
  let imagProduct ← add ri ir
  let realMatrix ← reshape (s₂ := [modes, width]) realProduct (by simp [Shape.size])
  let imagMatrix ← reshape (s₂ := [modes, width]) imagProduct (by simp [Shape.size])
  let realPadded ← padFrequencies hmodes realMatrix
  let imagPadded ← padFrequencies hmodes imagMatrix
  let realPlane ← reshape (s₂ := [1, grid / 2 + 1, width]) realPadded
    (by simp [Shape.size])
  let imagPlane ← reshape (s₂ := [1, grid / 2 + 1, width]) imagPadded
    (by simp [Shape.size])
  let planes ← concatLeadingAxis realPlane imagPlane
  let frequencyPlanes ← swapAdjacentAtDepth 0 planes
  let frequencyChannels ← swapAdjacentAtDepth 1 frequencyPlanes
  let packed ← swapAdjacentAtDepth 0 frequencyChannels
  let result ← irfft1dReference (n := grid) packed
  swapAdjacentAtDepth 0 result

end Fourier

/--
Apply learned channel maps to the first `modes` nonnegative Fourier bins.

Inputs and outputs have shape `[grid, width]`; real and imaginary weights both have shape
`[modes, width, width]`. `denseReference` and `automatic` can share a checkpoint directly.
The arbitrary-rank full-DFT FNO uses a different parameterization and cannot share these weights.
-/
def spectralConv1dRfft {grid width modes : Nat}
    (_hgrid : 0 < grid) (_hwidth : 0 < width) (hmodes : modes ≤ grid / 2 + 1)
    (x : RefTy m α [grid, width])
    (realWeight imagWeight : RefTy m α [modes, width, width])
    (path : SpectralPath := .automatic) : m (RefTy m α [grid, width]) := do
  if path == .automatic then
    if let some native :=
        _root_.Runtime.Autograd.Torch.Ops.spectralConv1dRfftNative? (m := m) (α := α) then
      if let some result ← native x realWeight imagWeight then return result
  Fourier.spectralConv1dRfftReference hmodes x realWeight imagWeight

end Runtime.Autograd.Model.F
