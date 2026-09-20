/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# Packed real Fourier transforms

A real transform keeps frequencies `0` through `n / 2`. The final axis stores the real and
imaginary components, so the scalar type stays real throughout a differentiable program.
The forward transform has no normalization; the inverse divides by `n`. Its interior bins
represent a conjugate pair and therefore contribute twice. DC and the even-length Nyquist bin
contribute once, and their imaginary coordinates have no effect.

The matrices below specify the real linear maps used by the portable runtime. In particular,
their ordinary real transposes are the adjoints for the packed Euclidean pairing. Taking an
inverse transform alone is not the forward transform's adjoint.
-/

@[expose] public section

namespace Spec.RealFFT

open TorchLean

/-- Whether a stored frequency is its own conjugate: DC, or Nyquist for an even length. -/
def isEndpoint (n frequency : Nat) : Bool :=
  frequency == 0 || (n % 2 == 0 && frequency == n / 2)

/-- Number of full-spectrum frequencies represented by a stored real-transform bin. -/
def multiplicity (n frequency : Nat) : Nat :=
  if isEndpoint n frequency then 1 else 2

/--
Forward real-transform matrix, with interleaved real/imaginary columns.

Endpoint imaginary columns are exactly zero. Evaluating `sin(pi * integer)` instead would
introduce a small, artificial derivative in coordinates that the real transform ignores.
-/
def forwardMatrix {α : Type} [Storage α] [Context α] (n : Nat) :
    Tensor α [n, (n / 2 + 1) * 2] :=
  Tensor.dim fun time => Tensor.dim fun packed =>
    let frequency := packed.val / 2
    let angle := 2 * MathFunctions.pi * (time.val : α) * (frequency : α) / (n : α)
    Tensor.scalar <|
      if packed.val % 2 == 0 then MathFunctions.cos angle
      else if isEndpoint n frequency then 0
      else -MathFunctions.sin angle

/-- Normalized inverse matrix, including the conjugate-pair multiplicity of interior bins. -/
def inverseMatrix {α : Type} [Storage α] [Context α] (n : Nat) :
    Tensor α [(n / 2 + 1) * 2, n] :=
  Tensor.dim fun packed => Tensor.dim fun time =>
    let frequency := packed.val / 2
    let angle := 2 * MathFunctions.pi * (time.val : α) * (frequency : α) / (n : α)
    let weight := (multiplicity n frequency : α) / (n : α)
    Tensor.scalar <|
      if packed.val % 2 == 0 then weight * MathFunctions.cos angle
      else if isEndpoint n frequency then 0
      else -weight * MathFunctions.sin angle

end Spec.RealFFT
