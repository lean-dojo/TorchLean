/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Aesop
import Lean.Meta.Tactic.Simp.RegisterCommand

/-! Shared rule-set declarations for TorchLean's proof tactics. -/

declare_aesop_rule_sets [AutogradDeriv, Autograd, Verify, Converges]

/-- Proved operation identities used inside `autograd`, without changing ordinary `simp`. -/
register_simp_attr autograd_simps
