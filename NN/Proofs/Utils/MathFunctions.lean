/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context

/-!
# `MathFunctions` helper lemmas

TorchLean sometimes writes scalar specs using `MathFunctions.*` (to keep the spec polymorphic over
`α`) but then specializes to `ℝ` in proofs.

For `ℝ`, the `MathFunctions` methods are definitionally equal to their `Real.*` counterparts. We
keep named lemmas here so proof scripts can rewrite uniformly without repeating the same `rfl`
helpers across modules.
-/

@[expose] public section

namespace Proofs

/-- `MathFunctions.exp` is definitional equal to `Real.exp` for `ℝ`. -/
lemma mathfunc_exp_eq_rexp (x : ℝ) : MathFunctions.exp x = Real.exp x := rfl

/-- `MathFunctions.sinh` is definitional equal to `Real.sinh` for `ℝ`. -/
lemma mathfunc_sinh_eq_rsinh (x : ℝ) : MathFunctions.sinh x = Real.sinh x := rfl

/-- `MathFunctions.cosh` is definitional equal to `Real.cosh` for `ℝ`. -/
lemma mathfunc_cosh_eq_rcosh (x : ℝ) : MathFunctions.cosh x = Real.cosh x := rfl

end Proofs
