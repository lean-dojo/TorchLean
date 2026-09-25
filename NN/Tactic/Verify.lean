/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tactic.Init

/-!
# From checked evidence to semantic guarantees

`verify` applies registered soundness theorems and discharges executable side conditions by
kernel reduction. A checker result alone is not a semantic theorem: each supported checker must
provide a proved rule, tagged `@[verify]`, connecting its result to the requested property.

This module has no dependency on a particular certificate format, optimizer, or external solver.
Domain modules register their own rules. No external process is run by the tactic.
-/

public meta section

/-- Register a proved soundness rule. Unproved success and domain conditions remain obligations. -/
macro "verify" : attr =>
  `(attr| aesop unsafe 90% apply (rule_sets := [$(Lean.mkIdent `Verify):ident]))

-- Checker results are closed by kernel reduction, which has no heartbeat limit. A large decidable
-- side condition can therefore run for a long time; keep checked evidence to executable checks.
add_aesop_rules safe tactic (rule_sets := [Verify]) (by decide +kernel)

/-- Prove a semantic guarantee using registered soundness rules and checked evidence.
Fails unless every obligation is solved. Introduce universal hypotheses explicitly when needed.
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
