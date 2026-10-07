/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.Muon.Core

/-!
# Newton-Schulz Muon Backends

The column-oriented Newton-Schulz polynomial orthogonalizer, its residual-checked packaging, and
the fixed-point condition under which it is an exact backend. Real-valued facts about the
polynomial step at an exactly column-orthogonal matrix live in `NN.MLTheory.Optimization.Muon.QR`,
next to the entrywise matrix API they need.
-/

@[expose] public section

namespace Optim

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Muon

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Coefficients for the odd Newton-Schulz polynomial used by Muon-style orthogonalization.

The column-oriented shape is
$X\mapsto aX+bX(X^\mathsf{T}X)+cX(X^\mathsf{T}X)^2$, matching the $Q^\mathsf{T}Q$
certificate `HasExactColumnGram`. TorchLean keeps the coefficients explicit so experiments and
backend-specific proofs can choose the polynomial they actually use.
-/
structure NewtonSchulzCoeffs (α : Type) where
  /-- Linear coefficient. -/
  a : α
  /-- Cubic coefficient. -/
  b : α
  /-- Quintic coefficient. -/
  c : α

/-- One column-oriented Newton-Schulz polynomial step using the column Gram matrix
$X^\mathsf{T}X$. -/
def newtonSchulzStep {m n : Nat} (coeffs : NewtonSchulzCoeffs α)
    (X : MatrixTensor α m n) : MatrixTensor α m n :=
  let G := columnGram X
  let XG := matMulSpec X G
  let XG2 := matMulSpec XG G
  addSpec (addSpec (scaleSpec X coeffs.a) (scaleSpec XG coeffs.b)) (scaleSpec XG2 coeffs.c)

/-- Iterate the column-oriented Newton-Schulz polynomial step a fixed number of times. -/
def newtonSchulzIter {m n : Nat} (coeffs : NewtonSchulzCoeffs α) :
    Nat → MatrixTensor α m n → MatrixTensor α m n :=
  fun steps => (newtonSchulzStep coeffs)^[steps]

/-- Column-oriented Newton-Schulz-shaped orthogonalizer backend. -/
def newtonSchulzOrthogonalizer {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) :
    Orthogonalizer α (.dim m (.dim n .scalar)) :=
  { apply := fun buffer => newtonSchulzIter coeffs steps buffer }

/--
Turn any orthogonalizer into a checked approximate backend by using the Gram-residual bound itself
as the success predicate.
-/
def residualCheckedApproxOrthogonalizer {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))) :
    CheckedApproxOrthogonalizer α m n eps :=
  { orthogonalizer := orthogonalizer
    Success := ApproxOrthogonalizesBuffer eps orthogonalizer
    certified := fun _ hresidual => hresidual }

/--
Newton-Schulz packaged as a checked approximate backend.

The backend is the explicit Newton-Schulz tensor program, and the success predicate is the
post-check that its returned direction satisfies the requested Gram-residual bound.
-/
def newtonSchulzResidualCheckedOrthogonalizer {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (eps : α) :
    CheckedApproxOrthogonalizer α m n eps :=
  residualCheckedApproxOrthogonalizer eps
    (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)

/--
A buffer is a fixed point of one column-oriented Newton-Schulz step.

This is a deliberately local condition.  It does not assert convergence from arbitrary inputs; it
records the exact algebraic fact needed when a backend has already reached a stable direction.
-/
def NewtonSchulzFixedPoint {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (buffer : MatrixTensor α m n) : Prop :=
  newtonSchulzStep coeffs buffer = buffer

/--
If one Newton-Schulz step fixes a buffer, then any finite number of Newton-Schulz iterations fixes
the same buffer.
-/
theorem newtonSchulzIter_fixed_of_step_fixed {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hfixed : NewtonSchulzFixedPoint coeffs buffer) :
    newtonSchulzIter coeffs steps buffer = buffer :=
  Function.iterate_fixed hfixed steps

/-- A Newton-Schulz fixed point is returned unchanged by the corresponding orthogonalizer. -/
theorem newtonSchulzOrthogonalizer_fixed_apply {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hfixed : NewtonSchulzFixedPoint coeffs buffer) :
    (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer =
      buffer :=
  newtonSchulzIter_fixed_of_step_fixed coeffs steps buffer hfixed

/--
An exact-column-orthogonal buffer that is fixed by one Newton-Schulz step is exactly orthogonalized
by every finite Newton-Schulz iteration.
-/
theorem newtonSchulzOrthogonalizer_exactOrthogonalizesBuffer_of_fixedPoint {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hgram : HasExactColumnGram buffer)
    (hfixed : NewtonSchulzFixedPoint coeffs buffer) :
    ExactOrthogonalizesBuffer
      (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps) buffer := by
  unfold ExactOrthogonalizesBuffer
  rw [newtonSchulzOrthogonalizer_fixed_apply coeffs steps buffer hfixed]
  exact hgram

/--
Newton-Schulz packaged as a checked exact backend for already-stable directions.

The success predicate says that the input buffer already has exact column Gram and is a fixed point
of one Newton-Schulz step.  Under that explicit condition, every finite Newton-Schulz iteration
returns the same exact-column-orthogonal direction.
-/
def newtonSchulzFixedPointCheckedExactOrthogonalizer {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) :
    CheckedExactOrthogonalizer α m n :=
  { orthogonalizer := newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps
    Success := fun buffer => HasExactColumnGram buffer ∧ NewtonSchulzFixedPoint coeffs buffer
    certified := fun buffer hsuccess =>
      newtonSchulzOrthogonalizer_exactOrthogonalizesBuffer_of_fixedPoint coeffs steps buffer
        hsuccess.1 hsuccess.2 }

end Muon

end Optim
