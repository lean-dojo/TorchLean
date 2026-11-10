/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Lean.Meta.Tactic.Grind.RegisterCommand
public meta import Lean.Meta.Tactic.Simp.Attr
import Lean.Meta.Tactic.Simp.RegisterCommand

/-!
# Lemma sets of the `einops` tactic

`einops` calls `grind only` on the `einops` set and `simp only` on the `einops_simps` set. A lemma
joins a set by an attribute at its declaration: `@[einops =]` adds an equation that `grind`
instantiates from its left-hand side, and `@[einops_simps]` adds a rewrite rule.
-/

/-- Equations that the `einops` tactic instantiates with `grind`. -/
register_grind_attr einops

/-- Rewrite rules of the simplification pass of the `einops` tactic. -/
register_simp_attr einops_simps
