/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.InnerProductSpace.Adjoint
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Notation for analytic (Fréchet) derivatives in TorchLean autograd proofs

Mathlib already provides two very useful scoped notations in this area:

- `open scoped InnerProduct` enables postfix `†` for adjoints of bounded operators
  (`ContinuousLinearMap.adjoint`).
- `open scoped Gradient` enables `∇ f x` for the gradient of a scalar function (in Hilbert spaces).

TorchLean’s autograd proofs are primarily **VJP-first**: the central analytic object is the
adjoint of the Fréchet derivative `(fderiv ℝ f x)†`, which is exactly the vector-Jacobian product
(Jacobian-transpose product) that PyTorch-style reverse-mode computes.

This file defines a small scoped notation `VJP[f, x]` for that operator.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open scoped InnerProduct

noncomputable section

variable {E F : Type}
variable [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
variable [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F]

/-- The Jacobian (Fréchet derivative) of `f` at `x`, packaged as a bounded linear map. -/
noncomputable abbrev jacobian (f : E → F) (x : E) : E →L[ℝ] F :=
  fderiv ℝ f x

@[inherit_doc Proofs.Autograd.jacobian]
scoped[Autograd] notation "J[" f ", " x "]" => Proofs.Autograd.jacobian f x

/--
The **vector-Jacobian product** operator (VJP) of `f` at `x`.

`VJP[f, x] : F →L[ℝ] E` is the adjoint of the Fréchet derivative `(fderiv ℝ f x) : E →L[ℝ] F`.

When `f` is a scalar loss (`F = ℝ`), the gradient is `VJP[f, x] 1` (equivalently `∇ f x`).
-/
noncomputable abbrev vjp (f : E → F) (x : E) : F →L[ℝ] E :=
  (jacobian f x)†

@[inherit_doc Proofs.Autograd.vjp]
scoped[Autograd] notation "VJP[" f ", " x "]" => Proofs.Autograd.vjp f x

end

end Autograd
end Proofs
