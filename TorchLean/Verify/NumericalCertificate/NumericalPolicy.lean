/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Grouping.Coalescing
public import TorchLean.Verify.NumericalCertificate.Contracts

/-!
# Numerical certificate policy guards

`deriveNodeRange` passes every node whose range transfer folds a reduction from the left through
the reduction guard before its transfer, and the guard reads the numerical policy of a selected
capsule. The guard covers the nodes of `fixedLeftNode`: the matrix product, every `Op.reduce` (only
the sum has a transfer), means, average pooling, mean squared error, and LayerNorm. The lookup takes
the first kernel with a matching node id. `AcceptedGraphKernelPlan` allows repeated ids, and a later
kernel with the same id plays no part in the guard. Every kernel of the plan, the selected one
included, passes the plan's policy and evidence gate. The statements establish the consumer's policy
gate, not numerical correctness of a native implementation.
-/

@[expose] public section

namespace TorchLean.NumericalCertificate

open Backend
open IR

/-- The reduction guard succeeds exactly when the selected node policy declares `fixedLeft`. -/
theorem fixedLeftGuard_eq_ok_iff
    (plan : AcceptedGraphKernelPlan) (id : Nat) (node : Node) :
    fixedLeftGuard plan id node = .ok () ↔
      ∃ policy, nodePolicy plan id = some policy ∧
        policy.reduction = .fixedLeft := by
  unfold fixedLeftGuard
  cases hpolicy : nodePolicy plan id with
  | none => simp
  | some policy => simp [Pure.pure, Except.pure]

/--
A successful reduction guard selects the first kernel with the node id. Its capsule declares
fixed-left reduction and passes the plan's complete policy and evidence gate.
-/
theorem fixedLeftGuard_capsule
    {plan : AcceptedGraphKernelPlan} {id : Nat} {node : Node}
    (hguard : fixedLeftGuard plan id node = .ok ()) :
    ∃ kernel,
      plan.graphPlan.kernels.find? (fun kernel => kernel.nodeId == id) = some kernel ∧
      kernel.nodeId = id ∧
      kernel.capsule.numericalPolicy.reduction = .fixedLeft ∧
      ({ op := kernel.op, capsule := kernel.capsule } : PlannedKernel).acceptable
        plan.policy = true := by
  rcases (fixedLeftGuard_eq_ok_iff plan id node).mp hguard with
    ⟨policy, hpolicy, hreduction⟩
  unfold nodePolicy at hpolicy
  cases hfind : plan.graphPlan.kernels.find? (fun kernel => kernel.nodeId == id) with
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
theorem fixedLeftGuard_of_bind_ok {β : Type} {plan : AcceptedGraphKernelPlan}
    {id : Nat} {node : Node} {rest : Unit → Except String β} {result : β}
    (h : (fixedLeftGuard plan id node >>= rest) = .ok result) :
    fixedLeftGuard plan id node = .ok () := by
  cases hguard : fixedLeftGuard plan id node with
  | error err => rw [hguard] at h; cases h
  | ok value => rfl

/-- A node range derived for a node whose transfer folds a reduction from the left passed the
reduction guard: the first selected kernel with the node's id declares the fixed-left policy and
passes the accepted plan's policy and evidence gate. -/
theorem deriveNodeRange_capsule
    {sources : Array CheckedSourceRange} {plan : AcceptedGraphKernelPlan}
    {ranges : Array CheckedNodeRange} {id : Nat} {node : Node} {result : RangeTransferResult}
    (hnode : fixedLeftNode node = true)
    (hderive : deriveNodeRange sources plan ranges id node = .ok result) :
    ∃ kernel,
      plan.graphPlan.kernels.find? (fun kernel => kernel.nodeId == id) = some kernel ∧
      kernel.nodeId = id ∧
      kernel.capsule.numericalPolicy.reduction = .fixedLeft ∧
      ({ op := kernel.op, capsule := kernel.capsule } : PlannedKernel).acceptable
        plan.policy = true := by
  unfold deriveNodeRange at hderive
  split at hderive
  · cases hderive
  · simp only [hnode, ↓reduceIte] at hderive
    exact fixedLeftGuard_capsule
      (fixedLeftGuard_of_bind_ok hderive)

end TorchLean.NumericalCertificate
