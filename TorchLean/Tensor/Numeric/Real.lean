/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Numeric
public import TorchLean.Tensor.Numeric.InverseTrig.Real
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# `MathFunctions` helper lemmas

Specifications call `MathFunctions.*` to stay polymorphic in `α`, and proofs specialize to `ℝ`.
Over `ℝ` the methods are definitionally the `Real.*` functions, and the `rfl` lemmas `exp_real`,
`sinh_real`, `cosh_real`, `tanh_real` and `sqrt_real` state this for `exp`, `sinh`, `cosh`, `tanh`
and `sqrt`, so proof scripts rewrite uniformly without local copies. `sqrt_max_zero` drops the
clamp at zero under a real square root, and `MathFunctions.sqrt_max_zero` states it for
`MathFunctions.sqrt`.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

/-- `MathFunctions.exp` is definitionally equal to `Real.exp` for `ℝ`. -/
theorem MathFunctions.exp_real (x : ℝ) : MathFunctions.exp x = Real.exp x := rfl

/-- `MathFunctions.sinh` is definitionally equal to `Real.sinh` for `ℝ`. -/
theorem MathFunctions.sinh_real (x : ℝ) : MathFunctions.sinh x = Real.sinh x := rfl

/-- `MathFunctions.cosh` is definitionally equal to `Real.cosh` for `ℝ`. -/
theorem MathFunctions.cosh_real (x : ℝ) : MathFunctions.cosh x = Real.cosh x := rfl

/-- `MathFunctions.tanh` is definitionally equal to `Real.tanh` for `ℝ`. -/
theorem MathFunctions.tanh_real (x : ℝ) : MathFunctions.tanh x = Real.tanh x := rfl

/-- `MathFunctions.sqrt` is definitionally equal to `Real.sqrt` for `ℝ`. -/
theorem MathFunctions.sqrt_real (x : ℝ) : MathFunctions.sqrt x = Real.sqrt x := rfl

/-- Clamping at zero does not change a real square root, since `Real.sqrt` is zero at the numbers
below zero. -/
theorem sqrt_max_zero (x : ℝ) : Real.sqrt (max x 0) = Real.sqrt x := by
  rcases le_total x 0 with h | h
  · rw [max_eq_right h, Real.sqrt_zero, Real.sqrt_eq_zero_of_nonpos h]
  · rw [max_eq_left h]

/-- Over `ℝ`, `MathFunctions.sqrt (max x 0)` is `Real.sqrt x`. -/
theorem MathFunctions.sqrt_max_zero (x : ℝ) : MathFunctions.sqrt (max x 0) = Real.sqrt x :=
  TorchLean.sqrt_max_zero x

end TorchLean
