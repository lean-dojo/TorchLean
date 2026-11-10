/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tactic.Autograd.Scalar
public import TorchLean.NN.Tape.RealNode.Arithmetic
public import TorchLean.NN.Tape.RealNode.Arm
public import TorchLean.NN.Tape.RealNode.Elementwise
public import TorchLean.NN.Tape.RealNode.Reductions
public import TorchLean.NN.Op.Pointwise
public import TorchLean.NN.Op.Softmax
public import TorchLean.NN.Tape.Lowering.HigherOrderReverse
public import TorchLean.NN.FDeriv.Interchange
public import TorchLean.NN.Dual.Domain

/-!
# Autograd proofs

`autograd` combines registered derivative rules and lifts scalar derivatives to elementwise tensor
nodes. It builds `RealNode.FDerivCorrect` certificates for the recorded graph nodes from proved
rules alone. For a node built by `Tape.LawfulNode.ofArm` from an operator arm, `autograd_arm_cert`
applies the operator's `derivative` certificates registered with `op_cert`.

`@[autograd]` registers a proved rule for an operation, and `autograd?` shows the proof script.
Nonsmooth operations require their pointwise domain hypotheses. `@[autograd simp]` registers
proved identities for rewriting inside expressions without changing the ordinary simp set. These
rewrites retain their hypotheses and can be scoped locally.

For nested evaluation, the tactic combines higher-order `Dual.jet` rules and mathlib smoothness
proofs, then lifts them through tensor coordinates and recorded graph nodes. These rules cover
arbitrary finite order, tensor shape, and fixed coordinate reindexing. The tactic also assembles
jet laws for ordered sums and products, matrix multiplication, and reverse accumulation.
The smooth nonlinear rules include tanh and the mutually differentiating sinh/cosh pair.
Local rules cover division, logarithms, and square roots away from zero, including compositions
with the smooth activation rules, and need smoothness only near the evaluation point.
Tensor outputs, reductions, and matrix products use the same local rules. Congruence carries tensor
identities through surrounding functions and retains membership hypotheses inside ordered folds.
Interchange rules identify derivatives of a pullback with pullbacks of higher derivatives when
the direction tuple and cotangent are fixed and the required smoothness is proved.
Identifying a reverse pass with the adjoint of the forward derivative takes the separate
first-order certificate, and general model lowering and checked IO need their own correctness
connections.
-/

public meta section

open TorchLean Tape RealNode
open FloatLib.Numerics (MathFunctions)

namespace TorchLean.FoldCongr

-- A fold only evaluates its update at visited entries, where domain hypotheses are available.
-- Scoped so that only the autograd rewrites below use this congruence.
attribute [scoped congr] List.foldl_ext

end TorchLean.FoldCongr

-- Failed matches must not unfold concrete iterated derivatives or the adjoint implementation.
attribute [aesop safe apply (transparency := reducible) (rule_sets := [Autograd])]
  iteratedFDeriv_fderiv_apply iteratedFDeriv_fderiv iteratedFDeriv_adjoint_fderiv

attribute [aesop safe apply (rule_sets := [AutogradDeriv]) (index := [unindexed])]
  HasDerivAt.tanh

attribute [aesop safe apply (rule_sets := [Autograd])]
  RealNode.elemwiseFDeriv RealNode.elemwiseFDerivAt
  RealNode.affineFDeriv RealNode.sumFDeriv

namespace TorchLean

open Lean Meta in
/-- The `TorchLean.Op` constructor under a tape node built by `LawfulNode.ofArm`, unfolding the
node's definitions one step at a time. -/
private partial def armOperator? (node : Expr) : MetaM (Option Name) := do
  let node := (← instantiateMVars node).headBeta
  if node.isAppOfArity ``LawfulNode.ofArm 9 then
    return (node.getArg! 4).getAppFn.constName?
  if node.isAppOf ``LawfulNode.toNodeData then
    return ← armOperator? node.appArg!
  match ← unfoldDefinition? node with
  | some node => armOperator? node
  | none => return none

open Lean Elab Tactic in
/-- Prove `FDerivCorrect` or `FDerivCorrectAt` for a node built by `LawfulNode.ofArm`
from a `derivative` certificate registered with `op_cert` for its operator. -/
elab "autograd_arm_cert" : tactic => withMainContext do
  let target ← instantiateMVars (← getMainTarget)
  unless target.isAppOfArity ``FDerivCorrect 5 ||
      target.isAppOfArity ``FDerivCorrectAt 6 do
    throwError "autograd_arm_cert: the goal is not a node derivative certificate"
  let some operator ← armOperator? (target.getArg! 3)
    | throwError "autograd_arm_cert: the node is not built from an operator arm"
  let side ← `(tactic| first
    | assumption
    | exact (CtxVec.forall_getTensorCLM_iff _ _ (· ≠ 0)).2 (by assumption)
    | exact (CtxVec.forall_getTensorCLM_iff _ _ (0 < ·)).2 (by assumption))
  for entry in Op.certificates (← getEnv) operator do
    unless entry.kind == `derivative do continue
    let cert := mkIdent entry.certificate
    for attempt in [← `(tactic| (apply FDerivCorrect.ofArm₂; intro _ _ _)),
        ← `(tactic| (apply FDerivCorrectAt.ofArm₂; intro _)),
        ← `(tactic| (apply FDerivCorrect.ofArm₁; intro _ _)),
        ← `(tactic| (apply FDerivCorrectAt.ofArm₁; intro _))] do
      let saved ← saveState
      try
        focus do
          evalTactic attempt
          evalTactic (← `(tactic| apply $cert))
          evalTactic (← `(tactic| all_goals try (first | (intro _; rfl) | (intro _ _; rfl))))
          evalTactic (← `(tactic| all_goals $side:tactic))
          done
        return
      catch _ => saved.restore
  throwError "autograd_arm_cert: no derivative certificate of {operator} closes the goal"

add_aesop_rules safe tactic (rule_sets := [Autograd]) (by autograd_arm_cert)

attribute [aesop unsafe 50% apply (rule_sets := [Autograd])] FDerivCorrect.at

attribute [aesop safe apply (rule_sets := [Autograd])]
  GraphData.PreservesJet.nil GraphData.PreservesJet.snoc
  GraphData.PreservesPullbackJet.nil GraphData.PreservesPullbackJet.snoc

attribute [aesop safe apply (rule_sets := [Autograd]) (index := [unindexed])]
  NodeData.PreservesJet.map NodeData.PreservesJet.zipWith
  NodeData.PreservesJet.get
  NodeData.PreservesPullbackJet.single NodeData.PreservesPullbackJet.add
  NodeData.PreservesPullbackJet.mul_left NodeData.PreservesPullbackJet.mul_right

add_aesop_rules safe tactic (rule_sets := [Autograd]) (by
  fun_prop [MathFunctions.tanh])

add_aesop_rules safe tactic (rule_sets := [Autograd]) (by
  open TorchLean.FoldCongr in
  simp (disch := first
    | assumption
    | fun_prop (disch :=
        first | solve_by_elim |
          (simp only [Tensor.Rep.pull_apply]; solve_by_elim))
    | solve_by_elim
    | (intro i hi; fun_prop (disch :=
        first | solve_by_elim |
          (simp only [Tensor.Rep.pull_apply]; solve_by_elim)))) only
    [autograd_simps, funext MathFunctions.exp_real,
      show (MathFunctions.sin : ℝ → ℝ) = Real.sin from rfl,
      show (MathFunctions.cos : ℝ → ℝ) = Real.cos from rfl,
      show (MathFunctions.tanh : ℝ → ℝ) = Real.tanh from rfl,
      show (MathFunctions.sinh : ℝ → ℝ) = Real.sinh from rfl,
      show (MathFunctions.cosh : ℝ → ℝ) = Real.cosh from rfl,
      show (MathFunctions.log : ℝ → ℝ) = Real.log from rfl,
      show (MathFunctions.sqrt : ℝ → ℝ) = Real.sqrt from rfl,
      ← Dual.tangent_jet,
      DualTensor.jet_apply,
      DualTensor.jet_coordinate,
      DualTensor.jet_sum,
      DualTensor.jet_sum_comp,
      DualTensor.jet_dot,
      DualTensor.jet_id,
      Dual.Nested.seedTensor_apply,
      Dual.jet_one,
      Dual.jet_add, Dual.jet_mul,
      Dual.jet_sub, Dual.jet_div,
      Dual.jet_log, Dual.jet_log_comp,
      Dual.jet_sqrt, Dual.jet_sqrt_comp,
      Dual.jet_one_div,
      Dual.jet_exp_comp, Dual.jet_sin_comp,
      Dual.jet_cos_comp, Dual.jet_tanh_comp,
      Dual.jet_sinh_comp, Dual.jet_cosh_comp,
      Dual.jet_foldl_add,
      Dual.jet_foldl_mul,
      Dual.Nested.ofPrimal_zero,
      Dual.jet_exp, Dual.jet_neg,
      Dual.jet_sin, Dual.jet_cos,
      Dual.jet_tanh,
      Dual.jet_sinh, Dual.jet_cosh,
      Dual.jet_id, Dual.jet_linear,
      Dual.jet_const])

add_aesop_rules safe tactic (rule_sets := [Autograd]) (by
  rw [← DualTensor.tangent_jet _
      (by fun_prop (disch :=
        first | solve_by_elim |
          (simp only [Tensor.Rep.pull_apply]; solve_by_elim)))]
  apply congrArg Dual.Nested.tangentTensor
  try open TorchLean.FoldCongr in simp (disch := fun_prop (disch :=
    first | solve_by_elim |
      (simp only [Tensor.Rep.pull_apply]; solve_by_elim))) only
    [autograd_simps, DualTensor.jet_matmul,
      DualTensor.jet_mul,
      DualTensor.jet_id,
      DualTensor.jet_const]
  all_goals (
    apply Tensor.Rep.ext
    intro i
    simp only [TorchLean.Tensor.map_apply,
      TorchLean.Tensor.zipWith_apply, Tensor.Rep.pull_apply,
      DualTensor.jet_apply]))

-- Argument equality can be stronger than output equality, so keep this step backtrackable.
add_aesop_rules unsafe 50% tactic (rule_sets := [Autograd]) (by
  congr! (sameFun := true)
  all_goals (
    apply Tensor.Rep.ext
    intro i
    simp only [TorchLean.Tensor.map_apply,
      TorchLean.Tensor.zipWith_apply, Tensor.Rep.pull_apply,
      DualTensor.jet_apply]))

add_aesop_rules safe tactic (rule_sets := [Autograd]) (by
  first
  | change PUnit
    exact PUnit.unit
  | change TorchLean.Tape.RealGraph.FDerivCorrect _ _ × TorchLean.Tape.RealNode.FDerivCorrect _ _
    constructor
  | change TorchLean.Tape.RealGraph.FDerivCorrectAt _ _ _ ×
      TorchLean.Tape.RealNode.FDerivCorrectAt _ _ _
    constructor)

/-- Prove a scalar derivative or a registered autograd certificate, including its side conditions.
Fails unless the whole goal is solved. Extend goal-directed rules with `@[autograd]`, or register
rewrites inside expressions with `@[autograd simp]`.
-/
macro "autograd" : tactic =>
  `(tactic| aesop (config := { terminal := true, enableSimp := false })
    (rule_sets := [$(Lean.mkIdent `Autograd):ident, -default]))

/-- Show the proof script found by `autograd`. -/
@[tactic_alt tacticAutograd]
macro "autograd?" : tactic =>
  `(tactic| aesop? (config := { terminal := true, enableSimp := false })
    (rule_sets := [$(Lean.mkIdent `Autograd):ident, -default]))

end TorchLean
