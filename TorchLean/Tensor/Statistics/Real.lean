/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.BigOperators.Expect
public import Mathlib.Basic.Real.Basic

/-!
# Population variance

`TorchLean.var x` is the population variance `𝔼 i, (x i - 𝔼 j, x j) * (x i - 𝔼 j, x j)` of a family
`x` of reals over a finite type, where `𝔼` is Mathlib's `Finset.expect`. The normalization proofs
take the variance of a row as `TorchLean.var`.
-/

@[expose] public section

namespace TorchLean

open scoped BigOperators

noncomputable section

/-- The population variance `𝔼 i, (x i - 𝔼 j, x j) * (x i - 𝔼 j, x j)` of a family `x` of reals
over a finite type. -/
protected def var {ι : Type*} [Fintype ι] (x : ι → ℝ) : ℝ :=
  𝔼 i, (x i - 𝔼 j, x j) * (x i - 𝔼 j, x j)

/-- The population variance of a family of reals over a finite type is nonnegative. -/
theorem var_nonneg {ι : Type*} [Fintype ι] (x : ι → ℝ) : 0 ≤ TorchLean.var x :=
  Finset.expect_nonneg fun _ _ => mul_self_nonneg _

end

end TorchLean
