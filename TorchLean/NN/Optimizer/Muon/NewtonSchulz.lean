/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Optimizer.Muon
public import TorchLean.Tensor.Contract

/-!
# Newton-Schulz Muon Backends

The column-oriented Newton-Schulz polynomial gives a Muon orthogonalizer over any scalar type. With
a residual post-check it is an approximate backend, and on a buffer that has exact column Gram and
is fixed by one step it is an exact backend. The real-valued facts about one step at an exactly
column-orthogonal matrix live in `TorchLean.NN.Optimizer.Muon.QR`, next to the entrywise
matrix API they need.

We wrote the step by hand from `newtonschulz5` in Jordan et al. (2024), "Muon: An optimizer for
hidden layers in neural networks", blog post, https://kellerjordan.github.io/posts/muon/. That
function casts the matrix to bfloat16, divides it by its Frobenius norm plus `1e-7`, transposes a
tall input and runs 5 steps of $A=XX^\mathsf{T}$, $B=bA+cA^2$, $X\mapsto aX+BX$. Over ℝ our
column form is the same polynomial in either orientation, since
$(XX^\mathsf{T})^kX=X(X^\mathsf{T}X)^k$. We take the scalar type, coefficients and step count from
the caller and don't normalize, so scaling is the caller's job: under the blog's coefficients
`(3.4445, -4.7750, 2.0315)` one step sends a singular value of 2 to about 33.7.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

namespace Muon

variable {α : Type} [Storage α] [Context α]

/--
Coefficients $(a, b, c)$ of the odd Newton-Schulz polynomial used by Muon-style orthogonalization.

The column-oriented step is $X\mapsto aX+bX(X^\mathsf{T}X)+cX(X^\mathsf{T}X)^2$, which matches the
$Q^\mathsf{T}Q$ certificate `HasExactColumnGram`. At an exactly orthonormal $X$ the step returns
$(a+b+c)X$. Muon's tuned quintic $(3.4445, -4.7750, 2.0315)$ sums to $0.701$ and shrinks such an
$X$, so the exact fixed-point backend does not apply to it. The classical cubic $(3/2, -1/2, 0)$
sums to one.
-/
structure NewtonSchulz.Coeffs (α : Type) where
  /-- Linear coefficient. -/
  a : α
  /-- Cubic coefficient. -/
  b : α
  /-- Quintic coefficient. -/
  c : α

/-- One column-oriented Newton-Schulz step $aX+bXG+cXG^2$, where $G=X^\mathsf{T}X$ is the column
Gram matrix. -/
def NewtonSchulz.step {m n : Nat} (coeffs : Coeffs α)
    (X : MatrixTensor α m n) : MatrixTensor α m n :=
  let G := columnGram X
  let XG := Tensor.matmul X G
  let XG2 := Tensor.matmul XG G
  (Tensor.scale X coeffs.a + Tensor.scale XG coeffs.b) + Tensor.scale XG2 coeffs.c

/-- `Muon.NewtonSchulz.iter coeffs steps` applies `Muon.NewtonSchulz.step coeffs` `steps` times. -/
def NewtonSchulz.iter {m n : Nat} (coeffs : Coeffs α) :
    Nat → MatrixTensor α m n → MatrixTensor α m n :=
  fun steps => (step coeffs)^[steps]

/-- The Muon orthogonalizer that runs `steps` column-oriented Newton-Schulz steps on the buffer. -/
def NewtonSchulz.orthogonalizer {m n : Nat}
    (coeffs : Coeffs α) (steps : Nat) :
    Orthogonalizer α (.dim m (.dim n .scalar)) :=
  { apply := fun buffer => iter coeffs steps buffer }

/--
Wraps any orthogonalizer as a checked approximate backend whose success predicate is the
Gram-residual bound on its output.
-/
def Orthogonalizer.residualChecked {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))) :
    CheckedApproxOrthogonalizer α m n eps :=
  { orthogonalizer := orthogonalizer
    Success := ApproxOrthogonalizesBuffer eps orthogonalizer
    certified := fun _ hresidual => hresidual }

/--
Newton-Schulz packaged as a checked approximate backend: the orthogonalizer is
`Muon.NewtonSchulz.orthogonalizer coeffs steps`, and the success predicate is a post-check that its
output meets the Gram-residual bound.
-/
def NewtonSchulz.residualChecked {m n : Nat}
    (coeffs : Coeffs α) (steps : Nat) (eps : α) :
    CheckedApproxOrthogonalizer α m n eps :=
  Orthogonalizer.residualChecked eps
    (orthogonalizer (α := α) (m := m) (n := n) coeffs steps)

/--
One column-oriented Newton-Schulz step leaves `buffer` unchanged.

The condition is local. It is the algebraic fact an exact backend needs once the iteration has
reached a stable direction, and nothing here proves that the iteration converges to one.
-/
def NewtonSchulz.FixedPoint {m n : Nat}
    (coeffs : Coeffs α) (buffer : MatrixTensor α m n) : Prop :=
  step coeffs buffer = buffer

/--
If one Newton-Schulz step fixes a buffer, then any finite number of Newton-Schulz iterations fixes
the same buffer.
-/
theorem NewtonSchulz.FixedPoint.iter_eq {m n : Nat}
    (coeffs : Coeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hfixed : FixedPoint coeffs buffer) :
    iter coeffs steps buffer = buffer :=
  Function.iterate_fixed hfixed steps

/-- A Newton-Schulz fixed point is returned unchanged by the corresponding orthogonalizer. -/
theorem NewtonSchulz.FixedPoint.orthogonalizer_apply {m n : Nat}
    (coeffs : Coeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hfixed : FixedPoint coeffs buffer) :
    (orthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer =
      buffer :=
  iter_eq coeffs steps buffer hfixed

/--
An exact-column-orthogonal buffer that is fixed by one Newton-Schulz step is exactly orthogonalized
by every finite Newton-Schulz iteration.
-/
theorem NewtonSchulz.exactOrthogonalizesBuffer {m n : Nat}
    (coeffs : Coeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hgram : HasExactColumnGram buffer)
    (hfixed : FixedPoint coeffs buffer) :
    ExactOrthogonalizesBuffer
      (orthogonalizer (α := α) (m := m) (n := n) coeffs steps) buffer := by
  unfold ExactOrthogonalizesBuffer
  rw [FixedPoint.orthogonalizer_apply coeffs steps buffer hfixed]
  exact hgram

/--
Newton-Schulz packaged as a checked exact backend for directions that are already stable.

The success predicate asks that the input buffer have exact column Gram and be a fixed point of one
Newton-Schulz step, so every number of steps returns the buffer itself. Over `ℝ` with $a+b+c=1$, the
fixed-point half follows from exact column Gram
(`Muon.NewtonSchulz.fixedPointChecked_success`).
-/
def NewtonSchulz.fixedPointChecked {m n : Nat}
    (coeffs : Coeffs α) (steps : Nat) :
    CheckedExactOrthogonalizer α m n :=
  { orthogonalizer := orthogonalizer (α := α) (m := m) (n := n) coeffs steps
    Success := fun buffer => HasExactColumnGram buffer ∧ FixedPoint coeffs buffer
    certified := fun buffer hsuccess =>
      exactOrthogonalizesBuffer coeffs steps buffer
        hsuccess.1 hsuccess.2 }

end Muon

end TorchLean
