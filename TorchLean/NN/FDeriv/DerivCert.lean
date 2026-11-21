/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Calculus.FDeriv.Basic
public import Mathlib.Analysis.InnerProductSpace.Adjoint

/-!
# The statement of a derivative certificate

`DerivCert f x δ v` says that `f` is differentiable at `x` and that `v` is the adjoint of its
derivative at `x` applied to `δ`. `op_cert Op.<operator> derivative` registers a theorem only when
its statement is this proposition with `f` the operator's forward in the listed inputs, `x` a point
of those inputs, `δ` a cotangent of its value outputs and `v` the operator's pullback of `δ`, the
other inputs fixed (`TorchLean.Tensor.Op.Cert`).

When the inputs of an operator are a mapped list `l.map g`, as for `Op.cat`, the point is the
family of the inputs by position (`TProd.parts`), in the L2 product of their tangent spaces.
-/

@[expose] public section

namespace TorchLean

/-- `f` is differentiable at `x`, and `v` is the adjoint of its derivative at `x` applied to `δ`:
the statement of a `derivative` certificate. -/
def DerivCert {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F]
    (f : E → F) (x : E) (δ : F) (v : E) : Prop :=
  DifferentiableAt ℝ f x ∧ v = (fderiv ℝ f x).adjoint δ

end TorchLean
