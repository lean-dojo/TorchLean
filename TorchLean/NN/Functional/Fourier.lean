/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Program
public import TorchLean.Tensor.Layers.RealFFT

/-!
# Differentiable real Fourier transforms

Real and imaginary coordinates sit in a final axis of length two, so models can use real FFTs
without changing the scalar type. Under `SpectralPath.automatic`, eager CUDA execution runs the
native cuFFT transform and its packed-real adjoint. Every other case, eager CPU execution
included, multiplies by the specification matrices, so the ordinary matrix JVP and VJP rules cover
these transforms and their higher derivatives. The two paths compute the same transform with
different floating-point operations, and their results can differ in rounding.
-/

@[expose] public section

open TorchLean.Shape (size size_append)

namespace TorchLean.nn.functional

open Ops

/-- Execution path of a Fourier transform. The choice changes neither normalization nor spectral
parameters. -/
inductive SpectralPath where
  /-- Offer `Op.rfft` and `Op.irfft` to the interpreter's `applyNative?`, which eager CUDA
  execution alone accepts, and record the dense reference transforms where it declines. -/
  | automatic
  /-- Always use dense reference transforms. -/
  | denseReference
  deriving BEq, DecidableEq, Repr

variable {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

namespace Fourier

/--
Dense reference transform, a product with `RealFFT.forwardMatrix n`, which `nn.functional.rfft`
records unless the interpreter's `applyNative?` records `Op.rfft`.
-/
def rfft {batch n : Nat} (x : Ref (m := m) (α := α) [batch, n]) :
    m (Ref (m := m) (α := α) [batch, n / 2 + 1, 2]) := do
  let basis ← Ops.const (m := m) (RealFFT.forwardMatrix (α := α) n)
  let packed : Ref (m := m) (α := α) [batch, (n / 2 + 1) * 2] ← Ops.apply
      (.matmul (batchA := []) (batchB := []) (batch := []) false false (fun h => nomatch h)
        (fun h => nomatch h) Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (x, basis, ())
  Ops.apply (.reshape (by simp)) (packed, ())

/-- Dense normalized inverse, a product with `RealFFT.inverseMatrix n` for output length `n`. -/
def irfft {batch n : Nat} (x : Ref (m := m) (α := α) [batch, n / 2 + 1, 2]) :
    m (Ref (m := m) (α := α) [batch, n]) := do
  let packed ← Ops.apply
      (.reshape (t := [batch, (n / 2 + 1) * 2]) (by simp)) (x, ())
  let basis ← Ops.const (m := m) (RealFFT.inverseMatrix (α := α) n)
  Ops.apply
      (.matmul (batchA := []) (batchB := []) (batch := []) false false (fun h => nomatch h)
        (fun h => nomatch h) Shape.BroadcastTo.proof Shape.BroadcastTo.proof) (packed, basis, ())

end Fourier

/--
Unnormalized real Fourier transform along the last axis, with every leading batch axis kept.

The last coordinate is `(real, imaginary)` and the phase convention is `exp(-2*pi*i*k*t/n)`, as in
`torch.fft.rfft`. The witness `0 < n` rules out the undefined length-zero transform, and an empty
batch is allowed. The result is differentiable on both paths.
-/
def rfft {batch : Shape} {n : Nat} (hn : 0 < n)
    (x : Ref (m := m) (α := α) (batch ++ [n])) (path : SpectralPath := .automatic) :
    m (Ref (m := m) (α := α) (batch ++ [n / 2 + 1, 2])) := do
  let rows ← Ops.apply
      (.reshape (t := [Shape.size batch, n]) (by simp [Shape.size_append])) (x, ())
  let packed ← (show m (Ref (m := m) (α := α) [Shape.size batch, n / 2 + 1, 2]) from do
    if path == .automatic then
      if let some native := Ops.applyNative? (m := m) (α := α) then
        if let some result ← native (.rfft (s := [Shape.size batch, n]) ⟨1, Nat.one_lt_two⟩
            n hn) (rows, ()) then
          return result
    Fourier.rfft rows)
  Ops.apply (.reshape (by simp [Shape.size_append])) (packed, ())

/--
Normalized inverse real transform with output length `n`, as `torch.fft.irfft` with `n`.

Interior bins are completed by conjugate symmetry. In exact real arithmetic the imaginary DC
coordinate and, for even `n`, the imaginary Nyquist coordinate contribute zero and have zero
derivatives. The dense path multiplies them by zero, so a NaN or an infinity there makes the
output NaN. The length is a parameter because lengths such as four and five store the same number
of bins.
-/
def irfft {batch : Shape} {n : Nat} (hn : 0 < n)
    (x : Ref (m := m) (α := α) (batch ++ [n / 2 + 1, 2]))
    (path : SpectralPath := .automatic) : m (Ref (m := m) (α := α) (batch ++ [n])) := do
  let rows ← Ops.apply
      (.reshape (t := [Shape.size batch, n / 2 + 1, 2]) (by simp [Shape.size_append]))
      (x, ())
  let values ← (show m (Ref (m := m) (α := α) [Shape.size batch, n]) from do
    if path == .automatic then
      if let some native := Ops.applyNative? (m := m) (α := α) then
        if let some result ← native (.irfft (s := [Shape.size batch, n]) ⟨1, Nat.one_lt_two⟩
            n hn) (rows, ()) then
          return result
    Fourier.irfft rows)
  Ops.apply (.reshape (by simp [Shape.size_append])) (values, ())

end TorchLean.nn.functional
