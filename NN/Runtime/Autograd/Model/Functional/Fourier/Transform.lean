/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Functional.Fourier

/-!
# Separable multidimensional Fourier transforms

Real and imaginary components share a row-major spatial shape and a trailing channel axis.
Each spatial axis is transformed independently; flattening is only a storage view, never a change
from a multidimensional DFT to a one-dimensional DFT. Native execution uses the existing real FFT
hooks twice per complex axis transform. Completing each real spectrum by conjugate symmetry gives
the full complex transform, including odd lengths and unrestricted complex inputs.

The inverse conjugates the input and output and divides by each axis length. Thus its total
normalization is the reciprocal of the number of spatial points. All indexing, permutation, and
arithmetic operations retain their ordinary JVP and VJP rules.
-/

@[expose] public section

namespace Runtime.Autograd.Model.F

open Spec TorchLean

variable {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

namespace Fourier

/-- Complete a real transform by reflecting its stored nonnegative-frequency bins. -/
def fullSpectrum {batch n : Nat} (hn : 0 < n) (x : RefTy m α [batch, n])
    (path : SpectralPath) : m (RefTy m α [batch, n] × RefTy m α [batch, n]) := do
  let packed ← if path == .automatic then rfft1d hn x else rfft1dReference x
  let realPart : RefTy m α [batch, n / 2 + 1] ← select 2 packed ⟨0, by change 0 < 2; decide⟩
  let imagPart : RefTy m α [batch, n / 2 + 1] ← select 2 packed ⟨1, by change 1 < 2; decide⟩
  let indices : Tensor (Fin (n / 2 + 1)) [n] := Id.run <|
    Tensor.generateFlatM [n] fun k => do
      have bound : k.val < n := by simpa [Shape.size] using k.isLt
      pure <| if h : k.val ≤ n / 2 then ⟨k.val, by omega⟩ else ⟨n - k.val, by omega⟩
  let indicesRef := Torch.dataConst (m := m) (α := α) indices
  let realFull ← indexSelect 1 n realPart indicesRef
  let imagFull ← indexSelect 1 n imagPart indicesRef
  let signs : Tensor α [n] := Tensor.generateFlat [n] fun k =>
    if k ≤ n / 2 then 1 else -1
  let signRef ← const signs
  let expanded ← broadcastTo (s₂ := [batch, n]) Shape.BroadcastTo.proof signRef
  let imaginary ← mul imagFull expanded
  pure (realFull, imaginary)

/-- Transform independent complex rows; inverse normalization is local to this axis. -/
def rows {batch n : Nat} (hn : 0 < n)
    (realPart imagPart : RefTy m α [batch, n]) (inverse : Bool) (path : SpectralPath) :
    m (RefTy m α [batch, n] × RefTy m α [batch, n]) := do
  let imaginary ← if inverse then scale imagPart (-1) else pure imagPart
  let (ar, ai) ← fullSpectrum hn realPart path
  let (br, bi) ← fullSpectrum hn imaginary path
  let realResult ← sub ar bi
  let imagResult ← add ai br
  if inverse then
    pure (← scale realResult (1 / (n : α)), ← scale imagResult (-1 / (n : α)))
  else pure (realResult, imagResult)

/--
Transform the remaining spatial axes, retaining the already transformed prefix in the row count.

The reshapes expose one axis at a time and batch together every other coordinate and channel.
An empty spatial shape is the identity; channels may be empty.
-/
def axes {channels : Nat} (outer : Nat) :
    (spatial : List Nat) → 0 < spatial.prod →
    RefTy m α [outer * spatial.prod, channels] →
    RefTy m α [outer * spatial.prod, channels] → Bool → SpectralPath →
    m (RefTy m α [outer * spatial.prod, channels] ×
      RefTy m α [outer * spatial.prod, channels])
  | [], _, realPart, imagPart, _, _ => pure (realPart, imagPart)
  | n :: rest, positive, realPart, imagPart, inverse, path => do
    have lengths : 0 < n ∧ 0 < rest.prod :=
      ⟨Nat.pos_of_mul_pos_right positive, Nat.pos_of_mul_pos_left positive⟩
    let toRows (value : RefTy m α [outer * (n :: rest).prod, channels]) := do
      let tensor ← reshape (s₂ := [outer, n, rest.prod * channels]) value
        (by simp [Shape.size, Nat.mul_assoc])
      let transposed ← swapAdjacentAtDepth 1 tensor
      reshape (s₂ := [outer * (rest.prod * channels), n]) transposed
        (by simp [Shape.size, Nat.mul_comm, Nat.mul_left_comm])
    let realRows ← toRows realPart
    let imagRows ← toRows imagPart
    let (transformedReal, transformedImag) ← rows lengths.1 realRows imagRows inverse path
    let fromRows (value : RefTy m α [outer * (rest.prod * channels), n]) := do
      let tensor ← reshape (s₂ := [outer, rest.prod * channels, n]) value
        (by simp [Shape.size, Nat.mul_assoc])
      let transposed ← swapAdjacentAtDepth 1 tensor
      reshape (s₂ := [outer * n * rest.prod, channels]) transposed
        (by simp [Shape.size, Nat.mul_assoc])
    let restoredReal ← fromRows transformedReal
    let restoredImag ← fromRows transformedImag
    let result ← axes (outer * n) rest lengths.2 restoredReal restoredImag inverse path
    pure (by simpa [List.prod_cons, Nat.mul_assoc] using result)

end Fourier

/--
Complex Fourier transform over any finite spatial shape, with a trailing channel axis.

Inputs are separate real/imaginary matrices whose first dimension is the flattened spatial grid.
The phase is `exp(-2*pi*i*sum_j(k_j*x_j/n_j))`, not a transform of the flattened index. Set
`inverse := true` for the normalized inverse. Empty spatial shape means identity; all supplied
spatial extents must be positive. `automatic` selects native per-axis FFT hooks when available.
-/
def fft (spatial : List Nat) (positive : 0 < spatial.prod) {channels : Nat}
    (realPart imagPart : RefTy m α [spatial.prod, channels])
    (inverse : Bool := false) (path : SpectralPath := .automatic) :
    m (RefTy m α [spatial.prod, channels] × RefTy m α [spatial.prod, channels]) := do
  let result ← Fourier.axes 1 spatial positive
    (by simpa using realPart) (by simpa using imagPart) inverse path
  pure (by simpa using result)

end Runtime.Autograd.Model.F
