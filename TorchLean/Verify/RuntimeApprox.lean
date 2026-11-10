/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Robustness
public import TorchLean.Verify.RuntimeApprox.Tolerance
public import TorchLean.Tensor.Context.Real

/-!
# Approximation of spec tensors by runtime tensors

Predicates saying that a runtime tensor approximates a spec tensor within an error bound. The
runtime tensor is mapped into `Real` entry by entry and compared with the spec under a chosen norm,
against a plain bound `eps` (`ApproxWith`) or an absolute and relative tolerance (`ApproxWithTol`).

Trust boundary:
- The predicates only state approximation. An end-to-end theorem needs an approximation lemma for
  each operation and a composition argument.
- Lean supplies a logical model for `Float`, but this module proves no per-operation
  approximation lemmas for that model, and native execution needs a separate provider agreement.
  Neither connection is assumed here.
- The approximation lemmas are proved for rounding-model backends such as FloatLib's `NF`, whose
  rounding error bounds are stated as formulas that compose.
-/

@[expose] public section

namespace TorchLean

open Robustness

noncomputable section

/-- Map a runtime tensor into a spec tensor, entry by entry, with `toSpec`. -/
def Tensor.toReal {α : Type} [Storage α] {s : Shape} (toSpec : α → ℝ)
    (t : Tensor α s) : Tensor ℝ s :=
  Tensor.map toSpec t

/-- `ApproxWith toSpec norm spec runtime eps`: the `norm` of the difference between `spec` and the
converted `runtime` is at most `eps`. -/
def ApproxWith {α : Type} [Storage α] {s : Shape}
    (toSpec : α → ℝ)
    (norm : ∀ {s : Shape}, Tensor ℝ s → ℝ)
    (spec : Tensor ℝ s)
    (runtime : Tensor α s)
    (eps : ℝ) : Prop :=
  distance (α := ℝ) norm spec (Tensor.toReal toSpec runtime) ≤ eps

/-- The `norm` of the difference between `spec` and the converted `runtime` is within the budget
`ApproxTol.bound tol` at the norms of the two tensors, so the relative part scales with the larger
norm. -/
def ApproxWithTol {α : Type} [Storage α] {s : Shape}
    (toSpec : α → ℝ)
    (norm : ∀ {s : Shape}, Tensor ℝ s → ℝ)
    (spec : Tensor ℝ s)
    (runtime : Tensor α s)
    (tol : ApproxTol) : Prop :=
  let runtimeS := Tensor.toReal toSpec runtime
  distance (α := ℝ) norm spec runtimeS ≤
    ApproxTol.bound tol (norm spec) (norm runtimeS)

/-- `ApproxWithTol` under the `L∞` norm `linfNorm`. -/
def ApproxTensorWithTol {α : Type} [Storage α] {s : Shape}
    (toSpec : α → ℝ)
    (spec : Tensor ℝ s)
    (runtime : Tensor α s)
    (tol : ApproxTol) : Prop :=
  ApproxWithTol (toSpec := toSpec) (norm := linfNorm) spec runtime tol

/-- A plain `eps` bound is an absolute-only tolerance bound.

Clamping by `Real.toNNReal` can only raise the budget, so this direction needs no sign hypothesis
on `eps`, while the converse `approxWithTol_absOnly_iff` needs `0 ≤ eps`. -/
theorem ApproxWith.approxWithTol_absOnly {α : Type} [Storage α] {s : Shape}
    {toSpec : α → ℝ}
    {norm : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {spec : Tensor ℝ s} {runtime : Tensor α s} (eps : ℝ)
    (h : ApproxWith (toSpec := toSpec) (norm := norm) spec runtime eps) :
    ApproxWithTol (toSpec := toSpec) (norm := norm) spec runtime (ApproxTol.absOnly eps) := by
  dsimp [ApproxWithTol]
  rw [ApproxTol.bound_absOnly]
  exact le_trans h (Real.le_coe_toNNReal eps)

/-- `ApproxWith.approxWithTol_absOnly` for the `L∞` relation `ApproxTensorWithTol`. -/
theorem ApproxWith.approxTensorWithTol_absOnly {α : Type} [Storage α] {s : Shape}
    {toSpec : α → ℝ}
    {spec : Tensor ℝ s} {runtime : Tensor α s} (eps : ℝ)
    (h : ApproxWith (toSpec := toSpec) (norm := linfNorm) spec runtime eps) :
    ApproxTensorWithTol (toSpec := toSpec) spec runtime (ApproxTol.absOnly eps) :=
  approxWithTol_absOnly (toSpec := toSpec) (norm := linfNorm) eps h

/-- A tolerance bound is a plain bound at the tolerance's evaluated budget. -/
theorem ApproxWithTol.approxWith {α : Type} [Storage α] {s : Shape}
    {toSpec : α → ℝ}
    {norm : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {spec : Tensor ℝ s} {runtime : Tensor α s} {tol : ApproxTol}
    (h : ApproxWithTol (toSpec := toSpec) (norm := norm) spec runtime tol) :
    ApproxWith (toSpec := toSpec) (norm := norm) spec runtime
      (ApproxTol.bound tol (norm spec) (norm (Tensor.toReal toSpec runtime))) :=
  h

/-- Tensor approximation is preserved when the tolerance is weakened in any field. -/
theorem ApproxWithTol.mono {α : Type} [Storage α] {s : Shape}
    {toSpec : α → ℝ}
    {norm : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {spec : Tensor ℝ s} {runtime : Tensor α s} {tol₁ tol₂ : ApproxTol}
    (habs : tol₁.abs ≤ tol₂.abs) (hrel : tol₁.rel ≤ tol₂.rel) (hslack : tol₁.slack ≤ tol₂.slack)
    (h : ApproxWithTol (toSpec := toSpec) (norm := norm) spec runtime tol₁) :
    ApproxWithTol (toSpec := toSpec) (norm := norm) spec runtime tol₂ :=
  le_trans h (ApproxTol.bound_mono habs hrel hslack _ _)

/-- For nonnegative `eps`, the absolute-only tolerance bound and the plain `eps` bound are
equivalent, so a proof can use whichever is convenient at each step. -/
theorem approxWithTol_absOnly_iff {α : Type} [Storage α] {s : Shape}
    {toSpec : α → ℝ}
    {norm : ∀ {s : Shape}, Tensor ℝ s → ℝ}
    {spec : Tensor ℝ s} {runtime : Tensor α s} {eps : ℝ} (heps : 0 ≤ eps) :
    ApproxWithTol (toSpec := toSpec) (norm := norm) spec runtime (ApproxTol.absOnly eps) ↔
      ApproxWith (toSpec := toSpec) (norm := norm) spec runtime eps := by
  -- With no relative part, `absOnly eps` evaluates to the constant budget `eps`.
  simp [ApproxWithTol, ApproxWith, ApproxTol.bound_absOnly, Real.coe_toNNReal eps heps]

/-! ## Notation

After `open scoped ApproxTol`, `spec ≈ᵀ[toSpec, tol] runtime` stands for
`ApproxTensorWithTol toSpec spec runtime tol`.
-/

scoped[TorchLean.ApproxTol] notation:50 spec " ≈ᵀ[" toSpec ", " tol "] " runtime =>
  ApproxTensorWithTol (toSpec := toSpec) spec runtime tol

end

end TorchLean
