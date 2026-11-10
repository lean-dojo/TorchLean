/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Elementwise

/-!
# Approximate equality with an absolute tolerance

Finite absolute-tolerance comparisons for artifact replay and leaf witness-margin bookkeeping.
IBP interval containment and CROWN binary32 replay use separate, tolerance-free comparisons.
An approximate match here is not a proof that an interval encloses every real execution.

`Numeric.approxEq` is the scalar `torch.allclose` (PyTorch 2.8 docs) with `rtol = 0` and
`atol = tol`. PyTorch tests `|input - other| ≤ atol + rtol * |other|`, whose relative term reads
`other` alone, and with that term gone our test is symmetric in `x` and `y` and keeps one absolute
margin at every magnitude. Equal infinities are close for `torch.isclose` (PyTorch 2.8 docs), and
with an infinite `atol` PyTorch 2.8.0 accepts any pair whose computed difference is finite, but
both cases give `false` here. The default `tol` is `Context.defaultEpsilon`, `1e-6` at `Float`,
where PyTorch defaults to `rtol = 1e-05` and `atol = 1e-08`.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

/-- Whether `|x - y| ≤ tol` for a tolerance that is not infinite, compared through `Context`. At
`Float` a NaN or an infinity among `x`, `y` and `tol` gives `false`, and at a lawful `Context`
such as the exact one of `ℚ` it is the ordered-field comparison (`approxEq_eq_true_iff`). -/
def Numeric.approxEq {α : Type} [Context α] (x y : α) (tol : α := Context.defaultEpsilon) : Bool :=
  Context.leBool (MathFunctions.abs (x - y)) tol && !Context.isInf tol

/-- At a lawful `Context`, `approxEq x y tol` holds exactly when `|x - y| ≤ tol` and `tol` is not
infinite. -/
theorem approxEq_eq_true_iff {α : Type} [Field α] [LinearOrder α] [IsStrictOrderedRing α]
    {context : Context α} [@LawfulContext α context _ _ _] (x y tol : α) :
    Numeric.approxEq x y tol = true ↔ |x - y| ≤ tol ∧ Context.isInf tol = false := by
  simp only [Numeric.approxEq, Context.leBool, Context.gtBool, Bool.and_eq_true, Bool.or_eq_true,
    decide_eq_true_eq, Bool.not_eq_true', LawfulContext.beq_iff, LawfulContext.abs_eq,
    LawfulContext.lt_iff, LawfulContext.le_iff, LawfulContext.sub_eq, ← le_iff_lt_or_eq]

end TorchLean
