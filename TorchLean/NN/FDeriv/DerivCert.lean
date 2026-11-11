/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Calculus.FDeriv.Basic
public import Mathlib.Analysis.InnerProductSpace.Adjoint
public import Mathlib.Data.Prod.TProd

/-!
# The statement of a derivative certificate

`DerivCert f x δ v` says that `f` is differentiable at `x` and that `v` is the adjoint of its
derivative at `x` applied to `δ`. `op_cert Op.<operator> derivative` registers a theorem only when
its statement is this proposition with `f` the operator's forward in the listed inputs, `x` a point
of those inputs, `δ` a cotangent of its value outputs and `v` the operator's pullback of `δ`, the
other inputs fixed (`TorchLean.Tensor.Op.Cert`).

When the inputs of an operator are a mapped list `l.map g`, as for `Op.cat`, the point is the
family of the inputs by position, in the L2 product of their tangent spaces. `TProd.parts` reads a
tuple of `List.TProd` over `l.map g` as that family, and `TProd.ofParts` builds the tuple from it.
-/

@[expose] public section

namespace TorchLean

/-- `f` is differentiable at `x`, and `v` is the adjoint of its derivative at `x` applied to `δ`:
the statement of a `derivative` certificate. -/
def DerivCert {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F]
    (f : E → F) (x : E) (δ : F) (v : E) : Prop :=
  DifferentiableAt ℝ f x ∧ v = (fderiv ℝ f x).adjoint δ

namespace TProd

variable {ι κ : Type*} {β : κ → Type*} (g : ι → κ)

/-- The components of a tuple over `l.map g`, by position in `l`. -/
def parts : (l : List ι) → List.TProd β (l.map g) → (i : Fin l.length) → β (g (l.get i))
  | [], _, i => i.elim0
  | _ :: _, (x, _), ⟨0, _⟩ => x
  | _ :: rest, (_, xs), ⟨k + 1, h⟩ => parts rest xs ⟨k, Nat.lt_of_succ_lt_succ h⟩

/-- The tuple over `l.map g` whose component at position `i` of `l` is `xs i`. -/
def ofParts : (l : List ι) → ((i : Fin l.length) → β (g (l.get i))) → List.TProd β (l.map g)
  | [], _ => PUnit.unit
  | _ :: rest, xs => (xs ⟨0, Nat.zero_lt_succ _⟩, ofParts rest fun i => xs i.succ)

end TProd

end TorchLean
