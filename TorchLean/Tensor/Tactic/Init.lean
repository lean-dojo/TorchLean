/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Aesop

/-!
# Shared declarations of the tensor proof tactics

We declare the Aesop rule set `Verify` used by `verify`, and the expression helper
`instantiateOuterLets` used by the einops tactics and by
`TorchLean.Verify.RuntimeApprox.Roundoff.Coverage`.
-/

declare_aesop_rule_sets [Verify]

public meta section

namespace TorchLean.Einops

open Lean

/-- Substitute the outer `let` and `have` bindings of an expression. -/
partial def instantiateOuterLets : Expr → Expr
  | .letE _ _ value body _ =>
      instantiateOuterLets (body.instantiate1 value)
  | expression => expression

end TorchLean.Einops
