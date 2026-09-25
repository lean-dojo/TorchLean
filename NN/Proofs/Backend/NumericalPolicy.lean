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

Every fixed-left range transfer (matrix product, whole and axis sums and means, average pooling,
mean squared error, and LayerNorm) succeeds only after the reduction guard accepts the node, and
the guard reads the numerical policy of a selected capsule. The lookup takes the first kernel with
a matching node id. `AcceptedGraphKernelPlan` does not require ids to be distinct, so a later
kernel with the same id is not checked. The statements establish the consumer's policy gate, not
numerical correctness of a native implementation.
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

/-- A computation that starts with the fixed-left guard succeeds only if the guard succeeds. -/
theorem requireFixedLeftReduction_of_bind_ok {β : Type} {plan : AcceptedGraphKernelPlan}
    {node : Node} {rest : Unit → Except String β} {result : β}
    (h : (requireFixedLeftReduction plan node >>= rest) = .ok result) :
    requireFixedLeftReduction plan node = .ok () := by
  cases hguard : requireFixedLeftReduction plan node with
  | error err => rw [hguard] at h; cases h
  | ok value => rfl

/-- A successful matrix-product range transfer must have passed the fixed-left policy guard. -/
theorem matmulContract_derive_requires_fixedLeft
    {context : NumericalRangeContext} {node : Node} {result : RangeTransferResult}
    (hderive : matmulContract.derive context node = .ok result) :
    requireFixedLeftReduction context.plan node = .ok () := by
  dsimp only [matmulContract] at hderive
  generalize node.parents = parents at hderive
  rcases parents with ⟨_ | ⟨a, _ | ⟨b, _ | ⟨c, rest⟩⟩⟩⟩ <;>
    first
    | exact requireFixedLeftReduction_of_bind_ok hderive
    | (change Except.error _ = .ok _ at hderive; cases hderive)


/-- A successful whole-tensor sum transfer must have passed the fixed-left policy guard. -/
theorem sumContract_derive_requires_fixedLeft
    {context : NumericalRangeContext} {node : Node} {result : RangeTransferResult}
    (hderive : sumContract.derive context node = .ok result) :
    requireFixedLeftReduction context.plan node = .ok () := by
  dsimp only [sumContract] at hderive
  generalize node.parents = parents at hderive
  rcases parents with ⟨_ | ⟨a, _ | ⟨b, _ | ⟨c, rest⟩⟩⟩⟩ <;>
    first
    | exact requireFixedLeftReduction_of_bind_ok hderive
    | (change Except.error _ = .ok _ at hderive; cases hderive)


/-- A successful mean-squared-error transfer must have passed the fixed-left policy guard. -/
theorem mseContract_derive_requires_fixedLeft
    {context : NumericalRangeContext} {node : Node} {result : RangeTransferResult}
    (hderive : mseContract.derive context node = .ok result) :
    requireFixedLeftReduction context.plan node = .ok () := by
  dsimp only [mseContract] at hderive
  generalize node.parents = parents at hderive
  rcases parents with ⟨_ | ⟨a, _ | ⟨b, _ | ⟨c, rest⟩⟩⟩⟩ <;>
    first
    | exact requireFixedLeftReduction_of_bind_ok hderive
    | (change Except.error _ = .ok _ at hderive; cases hderive)


/-- A successful average-pooling transfer must have passed the fixed-left policy guard. -/
theorem averagePoolContract_derive_requires_fixedLeft (padded : Bool)
    {context : NumericalRangeContext} {node : Node} {result : RangeTransferResult}
    (hderive : (averagePoolContract padded).derive context node = .ok result) :
    requireFixedLeftReduction context.plan node = .ok () := by
  dsimp only [averagePoolContract] at hderive
  generalize node.kind = kind at hderive
  generalize node.parents = parents at hderive
  rcases parents with ⟨_ | ⟨a, _ | ⟨b, rest⟩⟩⟩ <;> cases kind <;>
    first
    | exact requireFixedLeftReduction_of_bind_ok hderive
    | (change Except.error _ = .ok _ at hderive; cases hderive)


/-- A successful axis sum or mean transfer must have passed the fixed-left policy guard. -/
theorem axisReductionContract_derive_requires_fixedLeft (mean : Bool)
    {context : NumericalRangeContext} {node : Node} {result : RangeTransferResult}
    (hderive : (axisReductionContract mean).derive context node = .ok result) :
    requireFixedLeftReduction context.plan node = .ok () := by
  dsimp only [axisReductionContract] at hderive
  generalize node.kind = kind at hderive
  generalize node.parents = parents at hderive
  rcases parents with ⟨_ | ⟨a, _ | ⟨b, rest⟩⟩⟩ <;> cases kind <;>
    first
    | exact requireFixedLeftReduction_of_bind_ok hderive
    | (change Except.error _ = .ok _ at hderive; cases hderive)


/-- A successful LayerNorm transfer must have passed the fixed-left policy guard. -/
theorem layerNormContract_derive_requires_fixedLeft
    {context : NumericalRangeContext} {node : Node} {result : RangeTransferResult}
    (hderive : layerNormContract.derive context node = .ok result) :
    requireFixedLeftReduction context.plan node = .ok () := by
  dsimp only [layerNormContract] at hderive
  generalize node.kind = kind at hderive
  generalize node.parents = parents at hderive
  rcases parents with ⟨_ | ⟨a, _ | ⟨b, rest⟩⟩⟩ <;> cases kind <;>
    first
    | exact requireFixedLeftReduction_of_bind_ok hderive
    | (change Except.error _ = .ok _ at hderive; cases hderive)


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
