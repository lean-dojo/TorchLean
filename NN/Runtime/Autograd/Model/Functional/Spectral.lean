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
implementation, while preserving the parameters, activation, and real-valued spectral map.
Floating-point results can differ; dense zero-coefficient products do not suppress NaN or infinity.
-/

@[expose] public section

namespace Runtime.Autograd.Model.F

open Spec TorchLean

variable {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

namespace Fourier

/-- Restore discarded frequencies as zeros before the inverse transform. -/
def padFrequencies {modes frequencies width : Nat} (h : modes ≤ frequencies)
    (x : Ref (m := m) (α := α) [modes, width]) :
    m (Ref (m := m) (α := α) [frequencies, width]) := do
  let zeros ← const (Tensor.zeros (α := α) [frequencies - modes, width])
  let padded ← concat x zeros
  pure (by simpa [Nat.add_sub_of_le h] using padded)

end Fourier

/--
Apply learned channel maps to the first `modes` nonnegative Fourier bins.

Inputs and outputs have shape `[grid, width]`; both weights have shape `[modes, width, width]`.
The arbitrary-rank full-DFT FNO has a different parameterization and cannot share these weights.

The four real matrix products implement ordinary complex multiplication, without conjugating the
weight. Padding occurs after that multiplication, so discarded frequencies have no parameters.
`denseReference` and `automatic` share a checkpoint. Only the transforms select native primitives;
the tape composes their packed-real adjoints with the channel maps and indexing operations.
-/
def spectralConv {grid width modes : Nat} (_hgrid : 0 < grid)
    (_hwidth : 0 < width) (hmodes : modes ≤ grid / 2 + 1)
    (x : Ref (m := m) (α := α) [grid, width])
    (realWeight imagWeight : Ref (m := m) (α := α) [modes, width, width])
    (path : SpectralPath := .automatic) :
    m (Ref (m := m) (α := α) [grid, width]) := do
  let channels ← swapAdjacentAtDepth 0 x
  let transformed ← Runtime.Autograd.Model.F.rfft (batch := [width]) _hgrid channels path
  let realChannels ← select 2 transformed ⟨0, by change 0 < 2; decide⟩
  let imagChannels ← select 2 transformed ⟨1, by change 1 < 2; decide⟩
  let realFrequencies ← swapAdjacentAtDepth 0 realChannels
  let imagFrequencies ← swapAdjacentAtDepth 0 imagChannels
  let realModes ← slice 0 modes (by omega) realFrequencies
  let imagModes ← slice 0 modes (by omega) imagFrequencies
  let realRows ← reshape (s₂ := [modes, 1, width]) realModes
    (by simp [Shape.size])
  let imagRows ← reshape (s₂ := [modes, 1, width]) imagModes
    (by simp [Shape.size])
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
  let realPadded ← Fourier.padFrequencies hmodes realMatrix
  let imagPadded ← Fourier.padFrequencies hmodes imagMatrix
  let realPlane ← reshape (s₂ := [1, grid / 2 + 1, width]) realPadded
    (by simp [Shape.size])
  let imagPlane ← reshape (s₂ := [1, grid / 2 + 1, width]) imagPadded
    (by simp [Shape.size])
  let planes ← concat realPlane imagPlane
  let frequencyPlanes ← swapAdjacentAtDepth 0 planes
  let frequencyChannels ← swapAdjacentAtDepth 1 frequencyPlanes
  let packed ← swapAdjacentAtDepth 0 frequencyChannels
  let result ← Runtime.Autograd.Model.F.irfft (batch := [width]) _hgrid packed path
  swapAdjacentAtDepth 0 result

end Runtime.Autograd.Model.F
