/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tactic.Init
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
public meta import Mathlib.Tactic.Ring

/-!
# Scalar derivative rules

The scalar rules reuse mathlib's calculus lemmas and then normalize the resulting derivative
algebraically. The derivative lemmas sit in their own rule set, apart from the tensor rules, so the
normalization step cannot call itself recursively. Because they are unindexed, the rules also
match eta-reduced functions such as `(3 * ·)`. Strict recursive ring normalization identifies
equivalent arguments inside nonlinear functions, such as `tanh (x*x)` and `tanh (x^2)`, before the
derivative formulas are compared.
-/

public meta section

attribute [aesop safe apply (rule_sets := [AutogradDeriv]) (index := [unindexed])]
  hasDerivAt_id' hasDerivAt_const
  HasDerivAt.fun_add HasDerivAt.fun_sub HasDerivAt.fun_neg
  HasDerivAt.fun_mul HasDerivAt.fun_div HasDerivAt.fun_pow HasDerivAt.inv
  HasDerivAt.exp HasDerivAt.log HasDerivAt.sqrt HasDerivAt.sin HasDerivAt.cos
  HasDerivAt.sinh HasDerivAt.cosh
  HasDerivAt.cexp HasDerivAt.csin HasDerivAt.ccos HasDerivAt.csinh HasDerivAt.ccosh

namespace TorchLean

/-- Register a derivative lemma or an autograd certificate for `autograd`.

Rules are applied with backtracking. Prefer a conclusion that names the operation being proved,
and state domain restrictions as hypotheses of the rule.
-/
macro "autograd" : attr =>
  `(attr| aesop unsafe 90% apply
    (rule_sets := [$(Lean.mkIdent `AutogradDeriv):ident, $(Lean.mkIdent `Autograd):ident])
    (index := [unindexed]))

/-- Register a proved rewrite that the autograd simplifier applies inside expressions, leaving the
default simp set unchanged. Orient jet laws from mathematical jets to runtime operations, and keep
domain conditions in the hypotheses.
-/
macro "autograd" " simp" : attr => `(attr| autograd_simps)

add_aesop_rules safe tactic (rule_sets := [Autograd]) (by
  apply HasDerivAt.congr_deriv
  · aesop (config := { terminal := true, enableSimp := false })
      (rule_sets := [AutogradDeriv, -default])
  · first | ring1_nf | (simp <;> ring1))

end TorchLean
