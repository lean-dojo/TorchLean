/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Tactic.Init

/-!
# From checked evidence to semantic guarantees

`verify` applies registered soundness theorems and discharges executable side conditions by
kernel reduction. A checker result alone is not a semantic theorem: each supported checker must
provide a proved rule, tagged `@[verify]`, connecting its result to the requested property.

The tactic depends on no particular certificate format, optimizer or external solver, and it runs
no external process. Domain modules register their own rules.
-/

public meta section

namespace TorchLean

/-- Register a proved soundness rule for `verify`, which applies it backwards. The rule's premises,
such as the checker's success and domain conditions, become goals. -/
macro "verify" : attr =>
  `(attr| aesop unsafe 90% apply (rule_sets := [$(Lean.mkIdent `Verify):ident]))

-- Checker results are closed by kernel reduction, which has no heartbeat limit, so a large
-- decidable side condition can run for a long time. Checked evidence should be executable checks.
add_aesop_rules safe tactic (rule_sets := [Verify]) (by decide +kernel)

/-- Prove a semantic guarantee from the registered soundness rules and checked evidence. The search
fails unless it closes every goal. It skips Aesop's `intros` rule, so introduce universally
quantified hypotheses with `intro` first.
-/
macro "verify" : tactic =>
  `(tactic| aesop (config := { terminal := true, enableSimp := false })
    (rule_sets := [$(Lean.mkIdent `Verify):ident, -default])
    (erase Aesop.BuiltinRules.intros))

/-- Show the soundness rules and checks used by `verify`. -/
@[tactic_alt tacticVerify]
macro "verify?" : tactic =>
  `(tactic| aesop? (config := { terminal := true, enableSimp := false })
    (rule_sets := [$(Lean.mkIdent `Verify):ident, -default])
    (erase Aesop.BuiltinRules.intros))

end TorchLean
