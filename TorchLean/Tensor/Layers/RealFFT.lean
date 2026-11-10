/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Rep.Segment

/-!
# Packed real Fourier transforms

A real transform keeps frequencies `0` through `n / 2`. The final axis stores the real and
imaginary components, so the scalar type stays real throughout a differentiable program.
The forward transform has no normalization, and the inverse divides by `n`. The interior bins
represent a conjugate pair and therefore contribute twice to the inverse. DC and the even-length
Nyquist bin contribute once, and in the real maps their imaginary coordinates have no effect. In
floating point a NaN or infinity in one of those coordinates makes the result NaN through its zero
coefficient.

The matrices below specify the real linear maps used by the portable runtime. Their ordinary real
transposes are the adjoints for the packed Euclidean pairing, and the inverse transform is not the
adjoint of the forward one.

`Tensor.rfft dim n` and `Tensor.irfft dim n` apply these maps along any axis `dim` of length `n`,
with the (real, imaginary) pair on an appended last axis of length `2`, and `rfftBackward` and
`irfftBackward` apply their transposes.

`norm="backward"`, the default of `torch.fft.rfft` and `torch.fft.irfft` (PyTorch 2.8 docs), is
the scaling above, and the Python importer brings `"ortho"` and `"forward"` in as a factor applied
after the transform. We built both matrices by hand from the DFT. Their sums run term by term in
the order the docstrings give, where torch runs an FFT schedule, so the results can round
differently. The `irfft` page says an imaginary part in the zero-frequency term is always ignored,
and the zero coefficient above ignores it too on finite values.
`TorchLean.RealFFT.get2_forwardMatrix` and `TorchLean.RealFFT.get2_inverseMatrix`
(`TorchLean/Tensor/Layers/RealFFT/Complex.lean`) prove the entries over `ℝ` for `n ≠ 0` from the
kernels of Mathlib's `ZMod.dft`. In `scripts/checks/test_ir_capture.py`,
`test_real_fourier_transforms_within_tolerance` of `PyTorchGeneralOperatorTests` runs `Op.rfft`
and `Op.irfft` against torch at all three norms in binary32 and binary64, within `1e-4` relative
and `1e-5` absolute.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.RealFFT

/-- Whether a stored frequency is its own conjugate: DC, or Nyquist for an even length. -/
def isEndpoint (n frequency : Nat) : Bool :=
  frequency == 0 || (n % 2 == 0 && frequency == n / 2)

/-- Number of full-spectrum frequencies represented by a stored real-transform bin. -/
def multiplicity (n frequency : Nat) : Nat :=
  if isEndpoint n frequency then 1 else 2

/--
Forward real-transform matrix, with interleaved real/imaginary columns.

Endpoint imaginary columns are zero by definition, since a rounded `sin(pi * integer)` would put a
small, artificial derivative in coordinates that the real transform ignores. The angle is
`2 * pi * (r / n)`, with `r = time * frequency mod n` reduced exactly in `Nat` and the ratio `r / n`
cast once into the scalar. Over the reals it is `2 * pi * time * frequency / n` up to a whole turn,
and a binary backend rounds the ratio once and never forms the product `t * f` in the scalar type.
-/
def forwardMatrix {α : Type} [Storage α] [Context α] (n : Nat) :
    Tensor α [n, (n / 2 + 1) * 2] :=
  Tensor.dim fun time => Tensor.dim fun packed =>
    let frequency := packed.val / 2
    let angle := 2 * MathFunctions.pi * (((time.val * frequency % n : Nat) / n : Rat) : α)
    Tensor.scalar <|
      if packed.val % 2 == 0 then MathFunctions.cos angle
      else if isEndpoint n frequency then 0
      else -MathFunctions.sin angle

/-- Normalized inverse matrix, including the conjugate-pair multiplicity of interior bins. The angle
is formed as in `forwardMatrix`. -/
def inverseMatrix {α : Type} [Storage α] [Context α] (n : Nat) :
    Tensor α [(n / 2 + 1) * 2, n] :=
  Tensor.dim fun packed => Tensor.dim fun time =>
    let frequency := packed.val / 2
    let angle := 2 * MathFunctions.pi * (((time.val * frequency % n : Nat) / n : Rat) : α)
    let weight := Context.divNat (multiplicity n frequency : α) n
    Tensor.scalar <|
      if packed.val % 2 == 0 then weight * MathFunctions.cos angle
      else if isEndpoint n frequency then 0
      else -weight * MathFunctions.sin angle

end TorchLean.RealFFT

namespace TorchLean.Tensor

variable {α : Type} [Storage α] [Context α] {s : Shape}

/-- The real Fourier transform of `x` along `dim`, of length `n`: frequency `k` of each fiber is
`Σ t, x t * RealFFT.forwardMatrix n t (2 k + j)` for its real (`j = 0`) and imaginary (`j = 1`)
part, added from zero in increasing `t`, with the pair on an appended last axis. -/
def rfft (dim : Fin s.rank) (n : Nat) (x : Tensor α (s.set dim n)) :
    Tensor α (s.set dim (n / 2 + 1) ++ [2]) :=
  let basis := RealFFT.forwardMatrix (α := α) n
  Rep.ofFn fun c =>
    let parts := Shape.Coord.appendEquiv (s.set dim (n / 2 + 1)) [2] c
    let split := Shape.Coord.setEquiv dim.val (shape := s) (n / 2 + 1) parts.1
    (List.finRange n).foldl (fun acc t =>
      acc + x ((Shape.Coord.setEquiv dim.val (shape := s) n).symm (t, split.2)) *
        basis (t, ⟨2 * split.1.val + parts.2.1.val, by omega⟩, ())) 0

/-- The pullback of `rfft dim n x` at `g`: entry `t` of each fiber receives
`Σ k j, g k j * RealFFT.forwardMatrix n t (2 k + j)`, added from zero in increasing `2 k + j`. -/
def rfftBackward (dim : Fin s.rank) (n : Nat)
    (g : Tensor α (s.set dim (n / 2 + 1) ++ [2])) : Tensor α (s.set dim n) :=
  let basis := RealFFT.forwardMatrix (α := α) n
  Rep.ofFn fun c =>
    let split := Shape.Coord.setEquiv dim.val (shape := s) n c
    (List.finRange ((n / 2 + 1) * 2)).foldl (fun acc p =>
      acc + g ((Shape.Coord.appendEquiv (s.set dim (n / 2 + 1)) [2]).symm
        ((Shape.Coord.setEquiv dim.val (shape := s) (n / 2 + 1)).symm
          (⟨p.val / 2, by have := p.isLt; omega⟩, split.2), (⟨p.val % 2, by omega⟩, ()))) *
        basis (split.1, p, ())) 0

/-- The normalized inverse real Fourier transform of `x` along `dim`, with output length `n`:
entry `t` of each fiber is `Σ k j, x k j * RealFFT.inverseMatrix n (2 k + j) t`, added from zero in
increasing `2 k + j`. -/
def irfft (dim : Fin s.rank) (n : Nat) (x : Tensor α (s.set dim (n / 2 + 1) ++ [2])) :
    Tensor α (s.set dim n) :=
  let basis := RealFFT.inverseMatrix (α := α) n
  Rep.ofFn fun c =>
    let split := Shape.Coord.setEquiv dim.val (shape := s) n c
    (List.finRange ((n / 2 + 1) * 2)).foldl (fun acc p =>
      acc + x ((Shape.Coord.appendEquiv (s.set dim (n / 2 + 1)) [2]).symm
        ((Shape.Coord.setEquiv dim.val (shape := s) (n / 2 + 1)).symm
          (⟨p.val / 2, by have := p.isLt; omega⟩, split.2), (⟨p.val % 2, by omega⟩, ()))) *
        basis (p, split.1, ())) 0

/-- The pullback of `irfft dim n x` at `g`: frequency `k`, part `j` of each fiber receives
`Σ t, g t * RealFFT.inverseMatrix n (2 k + j) t`, added from zero in increasing `t`. -/
def irfftBackward (dim : Fin s.rank) (n : Nat) (g : Tensor α (s.set dim n)) :
    Tensor α (s.set dim (n / 2 + 1) ++ [2]) :=
  let basis := RealFFT.inverseMatrix (α := α) n
  Rep.ofFn fun c =>
    let parts := Shape.Coord.appendEquiv (s.set dim (n / 2 + 1)) [2] c
    let split := Shape.Coord.setEquiv dim.val (shape := s) (n / 2 + 1) parts.1
    (List.finRange n).foldl (fun acc t =>
      acc + g ((Shape.Coord.setEquiv dim.val (shape := s) n).symm (t, split.2)) *
        basis (⟨2 * split.1.val + parts.2.1.val, by omega⟩, t, ())) 0

end TorchLean.Tensor
