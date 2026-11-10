/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tactic.Init
public import TorchLean.NN.Optimizer.GD.Convergence
public import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Tactic.Positivity.Finset

/-!
# Convergence from proved bounds

`converges` applies convergence and error-bound theorems, using the local hypotheses. It supports
contractive fixed-point iterations and strongly monotone, Lipschitz gradient-descent operators.
Side conditions such as strong monotonicity come from hypotheses or from `positivity` and
`linarith`. Use `converges?` to inspect the proof, or register a proved rule with `@[converges]`.
-/

public meta section

namespace TorchLean

/-- Register a convergence or error-bound theorem, retaining all its hypotheses. -/
macro "converges" : attr =>
  `(attr| aesop unsafe 90% apply (rule_sets := [$(Lean.mkIdent `Converges):ident]))

attribute [converges]
  ContractingWith.tendsto_iterate_fixedPoint
  ContractingWith.apriori_dist_iterate_fixedPoint_le
  ContractingWith.aposteriori_dist_iterate_fixedPoint_le
  GD.tendsto_iterate_of_q_lt_one
  GD.dist_sq_iterate_le_of_q_nonneg
  GD.q_nonneg_of_le
  GD.q_lt_one_of_mul_sq_lt
  GD.q_lt_one_of_lt_div

add_aesop_rules safe tactic (rule_sets := [Converges]) (by positivity)
add_aesop_rules safe tactic (rule_sets := [Converges]) (by linarith)

/-- Prove a supported convergence or rate goal, or fail if any side condition is left unproved. -/
macro "converges" : tactic =>
  `(tactic| aesop (config := { terminal := true, enableSimp := false })
    (rule_sets := [$(Lean.mkIdent `Converges):ident, -default])
    (erase Aesop.BuiltinRules.intros))

/-- Show the convergence theorem and side-condition proofs found by `converges`. -/
@[tactic_alt tacticConverges]
macro "converges?" : tactic =>
  `(tactic| aesop? (config := { terminal := true, enableSimp := false })
    (rule_sets := [$(Lean.mkIdent `Converges):ident, -default])
    (erase Aesop.BuiltinRules.intros))

end TorchLean
