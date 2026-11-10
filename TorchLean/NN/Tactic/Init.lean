/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Aesop
import Lean.Meta.Tactic.Simp.RegisterCommand

/-!
# Rule sets of the NN proof tactics

We declare the Aesop rule sets `AutogradDeriv` and `Autograd` used by `autograd`, the rule set
`Converges` used by `converges`, the rule set `OpDeriv` of the scalar derivatives that `op_deriv`
adds to `AutogradDeriv`, the simp set `autograd_simps` behind `@[autograd simp]`, and the simp set
`op_deriv_simps` of the real semantics that `op_deriv` unfolds.
A rule set or simp set is available only in modules that import its declaration, so
`Autograd/Scalar.lean` and `Converges.lean` import this module, and `Autograd.lean` and
`OpDeriv.lean` reach it through `Autograd/Scalar.lean`.
-/

declare_aesop_rule_sets [AutogradDeriv, Autograd, Converges, OpDeriv]

/-- Proved operation identities used inside `autograd`, without changing ordinary `simp`. -/
register_simp_attr autograd_simps

/-- Proved equations that write an operation over the reals through the operations `op_deriv`
differentiates, without changing ordinary `simp`. -/
register_simp_attr op_deriv_simps
