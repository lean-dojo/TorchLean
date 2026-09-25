/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tactic.Autograd.Scalar
public import NN.Proofs.Autograd.Tape.Nodes.Arithmetic
public import NN.Proofs.Autograd.Tape.Nodes.Elementwise
public import NN.Proofs.Autograd.Tape.Nodes.Reductions
public import NN.Proofs.Autograd.Runtime.Link.HigherOrderReverse
public import NN.Proofs.Autograd.FDeriv.Interchange
public import NN.Proofs.Autograd.Dual.Domain

/-!
# Autograd proofs

`autograd` combines registered derivative rules and lifts scalar derivatives to elementwise
tensor nodes. It constructs the existing `NodeFDerivCorrect` certificates; it does not introduce
a second graph representation or certify an arbitrary runtime program by inspecting its name.

Use `@[autograd]` to register a proved rule for a new operation. Use `autograd?` to inspect the
proof script. Nonsmooth operations still require their pointwise domain hypotheses.
`@[autograd simp]` registers proved identities for rewriting inside expressions without changing
the ordinary simp set. These rewrites retain their hypotheses and can be scoped locally.

For nested evaluation, the tactic combines higher-order `Dual.jet` rules and mathlib smoothness
proofs, then lifts them through tensor coordinates and recorded graph nodes. These rules cover
arbitrary finite order, tensor shape, and fixed coordinate reindexing. The tactic also assembles
jet laws for ordered sums and products, matrix multiplication, and reverse accumulation.
The smooth nonlinear rules include tanh and the mutually differentiating sinh/cosh pair.
Local rules cover division, logarithms, and square roots away from zero, including compositions
with the smooth activation rules. They require smoothness near the evaluation point, not everywhere.
Tensor outputs, reductions, and matrix products use the same local rules. Congruence carries tensor
identities through surrounding functions and retains membership hypotheses inside ordered folds.
Interchange rules identify derivatives of a pullback with pullbacks of higher derivatives when
the direction tuple and cotangent are fixed and the required smoothness is proved.
The first-order adjoint certificate remains independent;
general model lowering and checked IO require their own correctness connections.
-/

public meta section

open Proofs.Autograd

namespace Proofs.Autograd.FoldCongr

-- A fold only evaluates its update at visited entries, where domain hypotheses are available.
-- Scoped so that only the autograd rewrites below use this congruence.
attribute [scoped congr] List.foldl_ext

end Proofs.Autograd.FoldCongr

-- Failed matches must not unfold concrete iterated derivatives or the adjoint implementation.
attribute [aesop safe apply (transparency := reducible) (rule_sets := [Autograd])]
  iteratedFDeriv_fderiv_apply iteratedFDeriv_fderiv iteratedFDeriv_adjoint_fderiv

attribute [aesop safe apply (rule_sets := [AutogradDeriv]) (index := [unindexed])]
  HasDerivAt.tanh

attribute [aesop safe apply (rule_sets := [Autograd])]
  TapeNodes.elemwiseFderiv TapeNodes.elemwiseFderivAt
  TapeNodes.affineFderiv TapeNodes.addFderiv TapeNodes.subFderiv
  TapeNodes.scaleFderiv TapeNodes.mulFderiv TapeNodes.squareFderiv
  TapeNodes.sumFderiv TapeNodes.broadcastToFderiv
  TapeNodes.reduceSumFderiv TapeNodes.reduceMeanFderiv
  TapeNodes.logisticFderiv TapeNodes.sigmoidFderiv TapeNodes.tanhFderiv
  TapeNodes.softplusFderiv TapeNodes.siluFderiv TapeNodes.geluFderiv
  TapeNodes.expFderiv TapeNodes.sinhFderiv TapeNodes.coshFderiv
  TapeNodes.reluFderivAt TapeNodes.absFderivAt TapeNodes.logFderivAt
  TapeNodes.invFderivAt TapeNodes.sqrtFderivAt TapeNodes.divFderivAt

attribute [aesop unsafe 50% apply (rule_sets := [Autograd])] NodeFDerivCorrect.at

attribute [aesop safe apply (rule_sets := [Autograd])]
  Algebra.GraphData.PreservesJet.nil Algebra.GraphData.PreservesJet.snoc
  Algebra.GraphData.PreservesPullbackJet.nil Algebra.GraphData.PreservesPullbackJet.snoc

attribute [aesop safe apply (rule_sets := [Autograd]) (index := [unindexed])]
  Algebra.NodeData.PreservesJet.map Algebra.NodeData.PreservesJet.map2
  Algebra.NodeData.PreservesJet.get
  Algebra.NodeData.PreservesPullbackJet.single Algebra.NodeData.PreservesPullbackJet.add
  Algebra.NodeData.PreservesPullbackJet.mul_left Algebra.NodeData.PreservesPullbackJet.mul_right

add_aesop_rules safe tactic (rule_sets := [Autograd]) (by
  fun_prop [Activation.Math.tanhSpec, Activation.Math.tanhDerivSpec, MathFunctions.tanh])

add_aesop_rules safe tactic (rule_sets := [Autograd]) (by
  open Proofs.Autograd.FoldCongr in
  simp (disch := first
    | assumption
    | fun_prop (disch :=
        first | solve_by_elim |
          (simp only [TorchLean.Tensor.Internal.Rep.pull_apply]; solve_by_elim))
    | solve_by_elim
    | (intro i hi; fun_prop (disch :=
        first | solve_by_elim |
          (simp only [TorchLean.Tensor.Internal.Rep.pull_apply]; solve_by_elim)))) only
    [autograd_simps, Activation.Math.tanhSpec, Activation.Math.tanhDerivSpec,
      funext Proofs.mathfunc_exp_eq_rexp,
      show (MathFunctions.sin : ℝ → ℝ) = Real.sin from rfl,
      show (MathFunctions.cos : ℝ → ℝ) = Real.cos from rfl,
      show (MathFunctions.tanh : ℝ → ℝ) = Real.tanh from rfl,
      show (MathFunctions.sinh : ℝ → ℝ) = Real.sinh from rfl,
      show (MathFunctions.cosh : ℝ → ℝ) = Real.cosh from rfl,
      show (MathFunctions.log : ℝ → ℝ) = Real.log from rfl,
      show (MathFunctions.sqrt : ℝ → ℝ) = Real.sqrt from rfl,
      ← Runtime.Autograd.Model.Dual.tangent_jet_at,
      Runtime.Autograd.Model.DualTensor.jet_apply,
      Runtime.Autograd.Model.DualTensor.jet_coordinate,
      Runtime.Autograd.Model.DualTensor.jet_sumSpec,
      Runtime.Autograd.Model.DualTensor.jet_sumSpec_comp_at,
      Runtime.Autograd.Model.DualTensor.jet_dotSpec_at,
      Runtime.Autograd.Model.DualTensor.jet_id,
      Runtime.Autograd.Model.Dual.Nested.seedTensor_apply,
      Runtime.Autograd.Model.Dual.jet_one,
      Runtime.Autograd.Model.Dual.jet_add_at, Runtime.Autograd.Model.Dual.jet_mul_at,
      Runtime.Autograd.Model.Dual.jet_sub_at, Runtime.Autograd.Model.Dual.jet_div_at,
      Runtime.Autograd.Model.Dual.jet_log, Runtime.Autograd.Model.Dual.jet_log_comp_at,
      Runtime.Autograd.Model.Dual.jet_sqrt, Runtime.Autograd.Model.Dual.jet_sqrt_comp_at,
      Runtime.Autograd.Model.Dual.jet_one_div,
      Runtime.Autograd.Model.Dual.jet_exp_comp_at, Runtime.Autograd.Model.Dual.jet_sin_comp_at,
      Runtime.Autograd.Model.Dual.jet_cos_comp_at, Runtime.Autograd.Model.Dual.jet_tanh_comp_at,
      Runtime.Autograd.Model.Dual.jet_sinh_comp_at, Runtime.Autograd.Model.Dual.jet_cosh_comp_at,
      Runtime.Autograd.Model.Dual.jet_foldl_add_at,
      Runtime.Autograd.Model.Dual.jet_foldl_mul_at,
      Runtime.Autograd.Model.Dual.Nested.ofPrimal_zero,
      Runtime.Autograd.Model.Dual.jet_exp, Runtime.Autograd.Model.Dual.jet_neg,
      Runtime.Autograd.Model.Dual.jet_sin, Runtime.Autograd.Model.Dual.jet_cos,
      Runtime.Autograd.Model.Dual.jet_tanh,
      Runtime.Autograd.Model.Dual.jet_sinh, Runtime.Autograd.Model.Dual.jet_cosh,
      Runtime.Autograd.Model.Dual.jet_id, Runtime.Autograd.Model.Dual.jet_linear,
      Runtime.Autograd.Model.Dual.jet_const])

-- Expand the shared steps at registration so `autograd?` prints replayable tactics.
set_option hygiene false in
run_cmd do
  let normalizeTensorCoordinates ← `(tactic| (
    apply TorchLean.Tensor.Internal.Rep.ext
    intro i
    simp only [TorchLean.Tensor.map, TorchLean.Tensor.Internal.Rep.map_apply,
      TorchLean.Tensor.map2Spec_apply, TorchLean.Tensor.Internal.Rep.pull_apply,
      Runtime.Autograd.Model.DualTensor.jet_apply]))
  Lean.Elab.Command.elabCommand (← `(command|
    add_aesop_rules safe tactic (rule_sets := [$(Lean.mkIdent `Autograd):ident]) (by
      rw [← Runtime.Autograd.Model.DualTensor.tangent_jet_at _
          (by fun_prop (disch :=
            first | solve_by_elim |
              (simp only [TorchLean.Tensor.Internal.Rep.pull_apply]; solve_by_elim)))]
      apply congrArg Runtime.Autograd.Model.Dual.Nested.tangentTensor
      try open Proofs.Autograd.FoldCongr in simp (disch := fun_prop (disch :=
        first | solve_by_elim |
          (simp only [TorchLean.Tensor.Internal.Rep.pull_apply]; solve_by_elim))) only
        [autograd_simps, Runtime.Autograd.Model.DualTensor.jet_matMulSpec_at,
          Runtime.Autograd.Model.DualTensor.jet_mul_at,
          Runtime.Autograd.Model.DualTensor.jet_id,
          Runtime.Autograd.Model.DualTensor.jet_const]
      all_goals $normalizeTensorCoordinates:tactic)))
  -- Argument equality can be stronger than output equality, so keep this step backtrackable.
  Lean.Elab.Command.elabCommand (← `(command|
    add_aesop_rules unsafe 50% tactic (rule_sets := [$(Lean.mkIdent `Autograd):ident]) (by
      congr! (sameFun := true)
      all_goals $normalizeTensorCoordinates:tactic)))

add_aesop_rules safe tactic (rule_sets := [Autograd]) (by
  first
  | change PUnit
    exact PUnit.unit
  | change GraphFDerivCorrect _ × NodeFDerivCorrect _
    constructor
  | change GraphFDerivCorrectAt _ _ × NodeFDerivCorrectAt _ _
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
