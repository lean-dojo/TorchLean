/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tactic.Init
public import NN.MLTheory.Optimization.StronglyConvexGD
public import Mathlib.Topology.MetricSpace.Contracting
import Mathlib.Tactic.Positivity.Finset

/-!
# Convergence from proved bounds

`converges` applies convergence and error-bound theorems, using the hypotheses in the goal.
It supports contractive fixed-point iterations and strongly monotone, Lipschitz gradient-descent
operators. It does not infer convexity from model syntax or claim arbitrary training converges.
Use `converges?` to inspect the proof, or register a proved rule with `@[converges]`.
-/

public meta section

/-- Register a convergence or error-bound theorem, retaining all its hypotheses. -/
macro "converges" : attr =>
  `(attr| aesop unsafe 90% apply (rule_sets := [$(Lean.mkIdent `Converges):ident]))

attribute [converges]
  ContractingWith.tendsto_iterate_fixedPoint
  ContractingWith.apriori_dist_iterate_fixedPoint_le
  ContractingWith.aposteriori_dist_iterate_fixedPoint_le
  Optim.GD.tendsto_iterate_of_q_lt_one
  Optim.GD.dist_sq_iterate_le_of_q_nonneg
  Optim.GD.q_nonneg_of_le
  Optim.GD.q_lt_one_of_mul_sq_lt
  Optim.GD.q_lt_one_of_lt_div

add_aesop_rules safe tactic (rule_sets := [Converges]) (by positivity)
add_aesop_rules safe tactic (rule_sets := [Converges]) (by linarith)

/-- Prove supported convergence or rate goals; fail if any hypothesis remains unproved. -/
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
