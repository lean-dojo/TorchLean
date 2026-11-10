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
and `sqrt`, so proof scripts rewrite uniformly without local copies.
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

end TorchLean
