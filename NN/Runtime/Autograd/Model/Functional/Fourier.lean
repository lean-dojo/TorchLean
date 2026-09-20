/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Program
public import NN.Spec.Layers.RealFFT

/-!
# Differentiable real Fourier transforms

These operations keep real and imaginary coordinates in a final axis of length two. Models can
therefore use real FFTs without changing scalar type. Eager CUDA execution uses the native
transform and its packed-real adjoint. Other interpreters multiply by the specification matrices,
so their existing matrix JVP and VJP rules also cover these transforms and higher derivatives.

The portable implementation is a dense reference transform. The native implementation uses
cuFFT; selecting the same mathematical operation does not imply the same floating-point order.
-/

@[expose] public section

namespace Runtime.Autograd.Model.F

open Spec TorchLean

variable {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

namespace Fourier

/-- Dense real-linear reference transform, used when the interpreter has no native FFT hook. -/
def rfft1dReference {batch n : Nat} (x : RefTy m α [batch, n]) :
    m (RefTy m α [batch, n / 2 + 1, 2]) := do
  let basis ← const (m := m) (RealFFT.forwardMatrix (α := α) n)
  let packed ← matmul (batchA := []) (batchB := []) (batch := []) x basis
  reshape packed (by simp [Shape.size])

/-- Dense normalized inverse, with explicit length so odd and even inputs remain distinct. -/
def irfft1dReference {batch n : Nat} (x : RefTy m α [batch, n / 2 + 1, 2]) :
    m (RefTy m α [batch, n]) := do
  let packed ← reshape (s₂ := [batch, (n / 2 + 1) * 2]) x
    (by simp [Shape.size])
  let basis ← const (m := m) (RealFFT.inverseMatrix (α := α) n)
  matmul (batchA := []) (batchB := []) (batch := []) packed basis

end Fourier

/--
Unnormalized real Fourier transform of each row, returned as `[batch, n / 2 + 1, 2]`.

The last coordinate is `(real, imaginary)` and the phase convention is `exp(-2*pi*i*k*t/n)`.
The positive-length witness rules out an undefined transform; an empty batch is allowed. This
operation records differentiable references, including on interpreters that use the dense fallback.
-/
def rfft1d {batch n : Nat} (_hn : 0 < n) (x : RefTy m α [batch, n]) :
    m (RefTy m α [batch, n / 2 + 1, 2]) := do
  if let some native := _root_.Runtime.Autograd.Torch.Ops.rfft1dNative? (m := m) (α := α) then
    if let some result ← native x then return result
  Fourier.rfft1dReference x

/--
Normalized inverse real transform with output length `n`.

Interior bins are completed by conjugate symmetry. Imaginary DC and even-length Nyquist inputs
are ignored, and their derivatives are zero. Passing `n` explicitly distinguishes lengths such
as four and five, which have the same number of stored bins.
-/
def irfft1d {batch n : Nat} (_hn : 0 < n) (x : RefTy m α [batch, n / 2 + 1, 2]) :
    m (RefTy m α [batch, n]) := do
  if let some native := _root_.Runtime.Autograd.Torch.Ops.irfft1dNative? (m := m) (α := α) then
    if let some result ← native x then return result
  Fourier.irfft1dReference x

end Runtime.Autograd.Model.F
