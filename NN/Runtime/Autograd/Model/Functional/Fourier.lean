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

/-- Choose Fourier execution without changing normalization or spectral parameters. -/
inductive SpectralPath where
  /-- Use native FFT hooks when supported, otherwise differentiable reference operations. -/
  | automatic
  /-- Always use dense reference transforms. -/
  | denseReference
  deriving BEq, DecidableEq, Repr

variable {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

namespace Fourier

/-- Dense real-linear reference transform, used when the interpreter has no native FFT hook. -/
def rfft {batch n : Nat} (x : Ref (m := m) (α := α) [batch, n]) :
    m (Ref (m := m) (α := α) [batch, n / 2 + 1, 2]) := do
  let basis ← const (m := m) (RealFFT.forwardMatrix (α := α) n)
  let packed ← matmul (batchA := []) (batchB := []) (batch := []) x basis
  reshape packed (by simp [Shape.size])

/-- Dense normalized inverse, with explicit length so odd and even inputs remain distinct. -/
def irfft {batch n : Nat} (x : Ref (m := m) (α := α) [batch, n / 2 + 1, 2]) :
    m (Ref (m := m) (α := α) [batch, n]) := do
  let packed ← reshape (s₂ := [batch, (n / 2 + 1) * 2]) x
    (by simp [Shape.size])
  let basis ← const (m := m) (RealFFT.inverseMatrix (α := α) n)
  matmul (batchA := []) (batchB := []) (batch := []) packed basis

end Fourier

/--
Unnormalized real Fourier transform along the last axis, preserving all leading batch dimensions.

The last coordinate is `(real, imaginary)` and the phase convention is `exp(-2*pi*i*k*t/n)`.
The positive-length witness rules out an undefined transform; an empty batch is allowed. This
operation records differentiable references, including on interpreters that use the dense fallback.
-/
def rfft {batch : Shape} {n : Nat} (_hn : 0 < n)
    (x : Ref (m := m) (α := α) (batch.concat [n])) (path : SpectralPath := .automatic) :
    m (Ref (m := m) (α := α) (batch.concat [n / 2 + 1, 2])) := do
  let rows ← reshape (s₂ := [Shape.size batch, n]) x
    (by simp [Shape.size_concat, Shape.size])
  let packed ← (show m (Ref (m := m) (α := α) [Shape.size batch, n / 2 + 1, 2]) from do
    if path == .automatic then
      if let some native := _root_.Runtime.Autograd.Torch.Ops.rfft1dNative? (m := m) (α := α) then
        if let some result ← native rows then return result
    Fourier.rfft rows)
  reshape packed (by simp [Shape.size_concat, Shape.size])

/--
Normalized inverse real transform with output length `n`.

Interior bins are completed by conjugate symmetry. In exact real arithmetic, imaginary DC and
even-length Nyquist coordinates contribute zero and have zero derivatives. The dense path still
multiplies these coordinates by zero, so it does not suppress NaN or infinity. Passing `n`
explicitly distinguishes lengths such as four and five, which store the same number of bins.
-/
def irfft {batch : Shape} {n : Nat} (_hn : 0 < n)
    (x : Ref (m := m) (α := α) (batch.concat [n / 2 + 1, 2]))
    (path : SpectralPath := .automatic) : m (Ref (m := m) (α := α) (batch.concat [n])) := do
  let rows ← reshape (s₂ := [Shape.size batch, n / 2 + 1, 2]) x
    (by simp [Shape.size_concat, Shape.size])
  let values ← (show m (Ref (m := m) (α := α) [Shape.size batch, n]) from do
    if path == .automatic then
      if let some native := _root_.Runtime.Autograd.Torch.Ops.irfft1dNative? (m := m) (α := α) then
        if let some result ← native rows then return result
    Fourier.irfft rows)
  reshape values (by simp [Shape.size_concat, Shape.size])

end Runtime.Autograd.Model.F
