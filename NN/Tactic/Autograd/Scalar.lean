/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tactic.Init
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
public meta import Mathlib.Tactic.Ring

/-!
# Scalar derivative rules

We reuse mathlib's calculus rules, then normalize the resulting derivative algebraically.
Keeping this rule set separate from the tensor rules prevents the normalization step from
recursively calling itself. The unindexed rules also match eta-reduced functions such as `(3 * ·)`.
Strict recursive ring normalization also identifies equivalent arguments inside nonlinear
functions, such as `tanh (x*x)` and `tanh (x^2)`, before comparing derivative formulas.
-/

public meta section

attribute [aesop safe apply (rule_sets := [AutogradDeriv]) (index := [unindexed])]
  hasDerivAt_id' hasDerivAt_const
  HasDerivAt.fun_add HasDerivAt.fun_sub HasDerivAt.fun_neg
  HasDerivAt.fun_mul HasDerivAt.fun_div HasDerivAt.fun_pow HasDerivAt.inv
  HasDerivAt.exp HasDerivAt.log HasDerivAt.sqrt HasDerivAt.sin HasDerivAt.cos
  HasDerivAt.sinh HasDerivAt.cosh
  HasDerivAt.cexp HasDerivAt.csin HasDerivAt.ccos HasDerivAt.csinh HasDerivAt.ccosh

/-- Register a derivative lemma or an autograd certificate for `autograd`.

Rules are applied with backtracking. Prefer a conclusion that names the operation being proved;
domain restrictions belong in the hypotheses, not in the tactic implementation.
-/
macro "autograd" : attr =>
  `(attr| aesop unsafe 90% apply
    (rule_sets := [$(Lean.mkIdent `AutogradDeriv):ident, $(Lean.mkIdent `Autograd):ident])
    (index := [unindexed]))

/-- Register a proved rewrite for use inside autograd expressions.
For jet laws, orient rules from mathematical jets to runtime operations.
Keep domain conditions in the hypotheses.
Unlike `@[simp]`, this registration affects only the autograd simplifier.
-/
macro "autograd" " simp" : attr => `(attr| autograd_simps)

add_aesop_rules safe tactic (rule_sets := [Autograd]) (by
  apply HasDerivAt.congr_deriv
  · aesop (config := { terminal := true, enableSimp := false })
      (rule_sets := [AutogradDeriv, -default])
  · first | ring1_nf | (simp <;> ring1))
