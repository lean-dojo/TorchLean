/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Backend.Grouping
public import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate.Contracts

/-!
# Numerical certificate policy guards

Successful fixed-left transfers use the numerical policy of the actual selected node capsule.
The lookup uses the first matching node id; the statements do not assume that ids are distinct.
They establish the consumer's policy gate, not numerical correctness of a native implementation.
-/

@[expose] public section

namespace Proofs.RuntimeApprox.NumericalCertificate

open NN.Backend
open NN.IR

/-- The reduction guard succeeds exactly when the selected node policy declares `fixedLeft`. -/
theorem requireFixedLeftReduction_eq_ok_iff
    (plan : AcceptedGraphKernelPlan) (node : Node) :
    requireFixedLeftReduction plan node = .ok () ↔
      ∃ policy, nodeNumericalPolicy plan node.id = some policy ∧
        policy.reduction = .fixedLeft := by
  unfold requireFixedLeftReduction
  cases hpolicy : nodeNumericalPolicy plan node.id with
  | none => simp
  | some policy => simp [Pure.pure, Except.pure]

/--
A successful reduction guard uses the actual first matching capsule, with its fixed-left
declaration and its complete accepted-plan policy and evidence gate.
-/
theorem requireFixedLeftReduction_selected_capsule
    {plan : AcceptedGraphKernelPlan} {node : Node}
    (hguard : requireFixedLeftReduction plan node = .ok ()) :
    ∃ kernel,
      plan.graphPlan.kernels.find? (fun kernel => kernel.nodeId == node.id) = some kernel ∧
      kernel.nodeId = node.id ∧
      kernel.capsule.numericalPolicy.reduction = .fixedLeft ∧
      ({ op := kernel.op, capsule := kernel.capsule } : PlannedKernel).acceptable
        plan.policy = true := by
  rcases (requireFixedLeftReduction_eq_ok_iff plan node).mp hguard with
    ⟨policy, hpolicy, hreduction⟩
  unfold nodeNumericalPolicy at hpolicy
  cases hfind : plan.graphPlan.kernels.find? (fun kernel => kernel.nodeId == node.id) with
  | none => simp [hfind] at hpolicy
  | some kernel =>
      have heq : kernel.capsule.numericalPolicy = policy := by
        simpa [hfind] using hpolicy
      have hmem := Array.mem_of_find?_eq_some hfind
      refine ⟨kernel, rfl, ?_, ?_, plan.node_acceptable kernel hmem⟩
      · simpa using Array.find?_some hfind
      · rw [heq]
        exact hreduction

/-- A successful matrix-product range transfer must have passed the fixed-left policy guard. -/
theorem matmulContract_derive_requires_fixedLeft
    {context : NumericalRangeContext} {node : Node} {result : RangeTransferResult}
    (hderive : matmulContract.derive context node = .ok result) :
    requireFixedLeftReduction context.plan node = .ok () := by
  dsimp only [matmulContract] at hderive
  generalize hparents : node.parents = parents at hderive
  rcases parents with ⟨parents⟩
  cases parents with
  | nil =>
      change Except.error _ = .ok result at hderive
      cases hderive
  | cons left parents =>
      cases parents with
      | nil =>
          change Except.error _ = .ok result at hderive
          cases hderive
      | cons right parents =>
          cases parents with
          | nil =>
              cases hguard : requireFixedLeftReduction context.plan node with
              | error err =>
                  rw [hguard] at hderive
                  change Except.error err = .ok result at hderive
                  cases hderive
              | ok value =>
                  cases value
                  rfl
          | cons next parents =>
              change Except.error _ = .ok result at hderive
              cases hderive

/-- The matrix-product range consumer inherits the selected capsule's accepted fixed-left policy. -/
theorem matmulContract_derive_selected_capsule
    {context : NumericalRangeContext} {node : Node} {result : RangeTransferResult}
    (hderive : matmulContract.derive context node = .ok result) :
    ∃ kernel,
      context.plan.graphPlan.kernels.find? (fun kernel => kernel.nodeId == node.id) =
        some kernel ∧
      kernel.nodeId = node.id ∧
      kernel.capsule.numericalPolicy.reduction = .fixedLeft ∧
      ({ op := kernel.op, capsule := kernel.capsule } : PlannedKernel).acceptable
        context.plan.policy = true :=
  requireFixedLeftReduction_selected_capsule (matmulContract_derive_requires_fixedLeft hderive)

end Proofs.RuntimeApprox.NumericalCertificate
