/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Functional.Fourier

/-!
# Separable multidimensional Fourier transforms

Real and imaginary components share a row-major spatial shape and a trailing channel axis. Each
spatial axis is transformed on its own, and flattening the grid is only a storage view, so the
result is the multidimensional DFT. A complex axis transform runs two real FFTs, one per component
and native where the interpreter has the hook, then completes each real spectrum by conjugate
symmetry. That covers odd lengths and arbitrary complex inputs.

The inverse conjugates the input and the output and divides by each axis length, so its total
normalization is the reciprocal of the number of spatial points. Every indexing, permutation and
arithmetic operation keeps its ordinary JVP and VJP rules.
-/

@[expose] public section

namespace TorchLean.nn.functional

open Ops

variable {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

namespace Fourier

/--
Full length-`n` spectrum of real rows, as real and imaginary parts: the stored bins of `rfft`,
extended by conjugate symmetry (bin `n - k` is the conjugate of bin `k`).
-/
def fullSpectrum {batch n : Nat} (hn : 0 < n) (x : Ref (m := m) (α := α) [batch, n])
    (path : SpectralPath) :
    m (Ref (m := m) (α := α) [batch, n] × Ref (m := m) (α := α) [batch, n]) := do
  let packed ← functional.rfft (batch := [batch]) hn x path
  let realPart : Ref (m := m) (α := α) [batch, n / 2 + 1] ←
    Ops.apply
        (.indexAt (s := [batch, n / 2 + 1, 2]) ⟨2, Shape.AxisInBounds.proof⟩
          ⟨0, by change 0 < 2; decide⟩) (packed, ())
  let imagPart : Ref (m := m) (α := α) [batch, n / 2 + 1] ←
    Ops.apply
        (.indexAt (s := [batch, n / 2 + 1, 2]) ⟨2, Shape.AxisInBounds.proof⟩
          ⟨1, by change 1 < 2; decide⟩) (packed, ())
  let indices : Tensor (Fin (n / 2 + 1)) [n] := Id.run <|
    Tensor.generateFlatM [n] fun k => do
      have bound : k.val < n := by simpa using k.isLt
      pure <| if h : k.val ≤ n / 2 then ⟨k.val, by omega⟩ else ⟨n - k.val, by omega⟩
  -- Both parts are read at `indices` along axis 1, one gather each.
  let selection := Op.gather
    (Shape.selectDims [batch, n / 2 + 1] ⟨1, Shape.AxisInBounds.proof⟩)
    (Shape.selectBatch [batch, n / 2 + 1] ⟨1, Shape.AxisInBounds.proof⟩ n)
    (.select [batch, n / 2 + 1] ⟨1, Shape.AxisInBounds.proof⟩ n)
  let binIndices := Ops.dataConst (m := m) (α := α)
    (Tensor.map (fun tuple => tuple ⟨0, Nat.one_pos⟩)
      (Tensor.selectIndices (s := [batch, n / 2 + 1]) ⟨1, Shape.AxisInBounds.proof⟩ indices))
  let realFull : Ref (m := m) (α := α) [batch, n] ←
    Ref.cast (m := m) (α := α)
        (by rw [Shape.eraseAxes_selectBatch, List.append_nil]; rfl) <$>
      Ops.apply selection (realPart, binIndices, ())
  let imagFull : Ref (m := m) (α := α) [batch, n] ←
    Ref.cast (m := m) (α := α)
        (by rw [Shape.eraseAxes_selectBatch, List.append_nil]; rfl) <$>
      Ops.apply selection (imagPart, binIndices, ())
  let signs : Tensor α [n] := Tensor.generateFlat [n] fun k =>
    if k ≤ n / 2 then 1 else -1
  let signRef ← Ops.const signs
  let expanded ← Ops.apply
      (.broadcast (t := [batch, n]) (Shape.BroadcastTo.proof).dims
        (Shape.BroadcastTo.proof).broadcastDims) (signRef, ())
  let imaginary ← Ops.apply .mul (imagFull, expanded, ())
  pure (realFull, imaginary)

/-- Transform independent complex rows of length `n`. The inverse divides by this axis length. -/
def rows {batch n : Nat} (hn : 0 < n)
    (realPart imagPart : Ref (m := m) (α := α) [batch, n]) (inverse : Bool) (path : SpectralPath) :
    m (Ref (m := m) (α := α) [batch, n] × Ref (m := m) (α := α) [batch, n]) := do
  let imaginary ← if inverse then
      Ops.apply .scale
        (imagPart, .data (Ops.dataConst (Tensor.scalar (-1))), ())
    else pure imagPart
  let (ar, ai) ← fullSpectrum hn realPart path
  let (br, bi) ← fullSpectrum hn imaginary path
  let realResult ← Ops.apply .sub (ar, bi, ())
  let imagResult ← Ops.apply .add (ai, br, ())
  if inverse then
    pure (← Ops.apply .scale
          (realResult, .data (Ops.dataConst
            (Tensor.scalar (Context.divNat 1 n))), ()), ← Ops.apply .scale
          (imagResult, .data (Ops.dataConst
            (Tensor.scalar (Context.divNat (-1) n))), ()))
  else pure (realResult, imagResult)

/--
Transform the remaining spatial axes, with the axes already transformed folded into the `outer`
row count.

The reshapes expose one axis at a time and batch every other coordinate and channel together. An
empty spatial shape is the identity, and `channels` may be zero.
-/
def axes {channels : Nat} (outer : Nat) :
    (spatial : List Nat) → 0 < spatial.prod →
    Ref (m := m) (α := α) [outer * spatial.prod, channels] →
    Ref (m := m) (α := α) [outer * spatial.prod, channels] → Bool → SpectralPath →
    m (Ref (m := m) (α := α) [outer * spatial.prod, channels] ×
      Ref (m := m) (α := α) [outer * spatial.prod, channels])
  | [], _, realPart, imagPart, _, _ => pure (realPart, imagPart)
  | n :: rest, positive, realPart, imagPart, inverse, path => do
    have lengths : 0 < n ∧ 0 < rest.prod :=
      ⟨Nat.pos_of_mul_pos_right positive, Nat.pos_of_mul_pos_left positive⟩
    let toRows (value : Ref (m := m) (α := α) [outer * (n :: rest).prod, channels]) := do
      let tensor ← Ops.apply
          (.reshape (t := [outer, n, rest.prod * channels]) (by simp [Nat.mul_assoc]))
          (value, ())
      let transposed ← Ops.apply
          (.permute (s := [outer, n, rest.prod * channels])
            (Shape.adjacentSwap [outer, n, rest.prod * channels] 1)
            (Shape.swapAdjacentAtDepth [outer, n, rest.prod * channels] 1) (by tensor_shape))
          (tensor, ())
      Ops.apply
          (.reshape (t := [outer * (rest.prod * channels), n])
            (by simp [Nat.mul_comm, Nat.mul_left_comm])) (transposed, ())
    let realRows ← toRows realPart
    let imagRows ← toRows imagPart
    let (transformedReal, transformedImag) ← rows lengths.1 realRows imagRows inverse path
    let fromRows (value : Ref (m := m) (α := α) [outer * (rest.prod * channels), n]) := do
      let tensor ← Ops.apply
          (.reshape (t := [outer, rest.prod * channels, n]) (by simp [Nat.mul_assoc]))
          (value, ())
      let transposed ← Ops.apply
          (.permute (s := [outer, rest.prod * channels, n])
            (Shape.adjacentSwap [outer, rest.prod * channels, n] 1)
            (Shape.swapAdjacentAtDepth [outer, rest.prod * channels, n] 1) (by tensor_shape))
          (tensor, ())
      Ops.apply
          (.reshape (t := [outer * n * rest.prod, channels]) (by simp [Nat.mul_assoc]))
          (transposed, ())
    let restoredReal ← fromRows transformedReal
    let restoredImag ← fromRows transformedImag
    let result ← axes (outer * n) rest lengths.2 restoredReal restoredImag inverse path
    pure (by simpa [List.prod_cons, Nat.mul_assoc] using result)

end Fourier

/--
Complex Fourier transform over any finite spatial shape, with a trailing channel axis.

The real and imaginary parts are separate matrices whose first axis is the flattened spatial grid,
and the phase is `exp(-2*pi*i*sum_j(k_j*x_j/n_j))` over the spatial axes. Set `inverse := true`
for the normalized inverse. An empty spatial shape is the identity, and every spatial extent must
be positive. `automatic` selects native per-axis FFT hooks when available.

PyTorch analogue: `torch.fft.fftn` over the spatial axes.
-/
def fft (spatial : List Nat) (positive : 0 < spatial.prod) {channels : Nat}
    (realPart imagPart : Ref (m := m) (α := α) [spatial.prod, channels])
    (inverse : Bool := false) (path : SpectralPath := .automatic) :
    m (Ref (m := m) (α := α) [spatial.prod, channels] ×
      Ref (m := m) (α := α) [spatial.prod, channels]) := do
  let result ← Fourier.axes 1 spatial positive
    (by simpa using realPart) (by simpa using imagPart) inverse path
  pure (by simpa using result)

end TorchLean.nn.functional
