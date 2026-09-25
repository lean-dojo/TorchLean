/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Backend.Dispatch

/-!
# Lossless backend audit grouping

Coalescing preserves every node id, operation, and complete capsule, in order and with repeats.
Consequently, a grouped acceptance proof applies to every node in its source graph plan.
These statements concern the planner and its declared contracts, not native kernel execution.
-/

@[expose] public section

namespace NN.Backend

/-- A group may absorb a node only when its operation and complete capsule agree. -/
theorem KernelGroup.canAppend_iff (group : KernelGroup) (kernel : IR.PlannedNodeKernel) :
    group.canAppend kernel = true ↔
      group.op = kernel.op ∧ group.capsule = kernel.capsule := by
  simp [KernelGroup.canAppend]

private theorem pushKernel_expands
    (groups : Array KernelGroup) (kernel : IR.PlannedNodeKernel) :
    (IR.GraphKernelPlan.pushKernel groups kernel).flatMap
        (fun group => group.nodeIds.map fun id => (id, group.op, group.capsule)) =
      (groups.flatMap
        (fun group => group.nodeIds.map fun id => (id, group.op, group.capsule))).push
          (kernel.nodeId, kernel.op, kernel.capsule) := by
  by_cases hempty : groups = #[]
  · subst groups
    simp [IR.GraphKernelPlan.pushKernel, IR.PlannedNodeKernel.toSingletonGroup]
  · obtain ⟨groups, group, rfl⟩ := Array.exists_push_of_ne_empty hempty
    simp only [IR.GraphKernelPlan.pushKernel, Array.back?_push]
    by_cases hsame : group.canAppend kernel = true
    · rcases (group.canAppend_iff kernel).mp hsame with ⟨hop, hcapsule⟩
      simp [hsame, KernelGroup.appendNode, hop, hcapsule]
    · simp [hsame, IR.PlannedNodeKernel.toSingletonGroup]

private theorem foldl_expands
    (kernels : List IR.PlannedNodeKernel) (groups : Array KernelGroup) :
    (kernels.foldl IR.GraphKernelPlan.pushKernel groups).flatMap
        (fun group => group.nodeIds.map fun id => (id, group.op, group.capsule)) =
      groups.flatMap (fun group => group.nodeIds.map fun id => (id, group.op, group.capsule)) ++
        (kernels.map fun kernel => (kernel.nodeId, kernel.op, kernel.capsule)).toArray := by
  induction kernels generalizing groups with
  | nil => simp
  | cons kernel kernels ih =>
      rw [List.foldl_cons, ih, pushKernel_expands]
      simp [-Array.append_singleton, Array.push_eq_append, Array.append_assoc]

/--
Expanding coalesced audit groups reproduces the original node ids, operations, and full capsules.
Array equality preserves order and multiplicity, including repeated node ids.
-/
theorem IR.GraphKernelPlan.toCoalescedGroups_expands (plan : IR.GraphKernelPlan) :
    plan.toCoalescedGroups.groups.flatMap
        (fun group => group.nodeIds.map fun id => (id, group.op, group.capsule)) =
      plan.kernels.map (fun kernel => (kernel.nodeId, kernel.op, kernel.capsule)) := by
  rcases plan with ⟨⟨kernels⟩⟩
  simpa [IR.GraphKernelPlan.toCoalescedGroups] using foldl_expands kernels #[]

/-- Every original node occurs in a group with exactly its operation and capsule. -/
theorem IR.GraphKernelPlan.node_mem_coalesced
    (plan : IR.GraphKernelPlan) (kernel : IR.PlannedNodeKernel)
    (hkernel : kernel ∈ plan.kernels) :
    ∃ group ∈ plan.toCoalescedGroups.groups,
      kernel.nodeId ∈ group.nodeIds ∧
      kernel.op = group.op ∧ kernel.capsule = group.capsule := by
  have hentry :
      (kernel.nodeId, kernel.op, kernel.capsule) ∈
        plan.toCoalescedGroups.groups.flatMap
          (fun group => group.nodeIds.map fun id => (id, group.op, group.capsule)) := by
    rw [plan.toCoalescedGroups_expands]
    exact Array.mem_map.mpr ⟨kernel, hkernel, rfl⟩
  rcases Array.mem_flatMap.mp hentry with ⟨group, hgroup, hentry⟩
  rcases Array.mem_map.mp hentry with ⟨id, hid, heq⟩
  rcases Prod.mk.inj heq with ⟨hid_eq, hfields⟩
  rcases Prod.mk.inj hfields with ⟨hop, hcapsule⟩
  exact ⟨group, hgroup, hid_eq ▸ hid, hop.symm, hcapsule.symm⟩

/-- Group acceptance checks every projected operation and complete capsule. -/
theorem GroupedKernelPlan.acceptable_iff
    (plan : GroupedKernelPlan) (policy : KernelPolicy) :
    plan.acceptable policy = true ↔
      ∀ group ∈ plan.groups, group.toPlannedKernel.acceptable policy = true := by
  rw [GroupedKernelPlan.acceptable, Array.all_eq_true_iff_forall_mem]
  constructor
  · intro h group hgroup
    exact h group.toPlannedKernel (Array.mem_map.mpr ⟨group, hgroup, rfl⟩)
  · intro h kernel hkernel
    rcases Array.mem_map.mp hkernel with ⟨group, hgroup, rfl⟩
    exact h group hgroup

/-- Accepting the coalesced plan guarantees acceptance of each original node's capsule. -/
theorem IR.GraphKernelPlan.node_acceptable
    (plan : IR.GraphKernelPlan) (policy : KernelPolicy)
    (haccepted : plan.toCoalescedGroups.acceptable policy = true)
    (kernel : IR.PlannedNodeKernel) (hkernel : kernel ∈ plan.kernels) :
    ({ op := kernel.op, capsule := kernel.capsule } : PlannedKernel).acceptable policy = true := by
  rcases plan.node_mem_coalesced kernel hkernel with
    ⟨group, hgroup, _, hop, hcapsule⟩
  have haccepted := (plan.toCoalescedGroups.acceptable_iff policy).mp haccepted group hgroup
  simpa [KernelGroup.toPlannedKernel, hop, hcapsule] using haccepted

/-- Every node in an accepted graph plan passes the same complete policy and evidence gate. -/
theorem AcceptedGraphKernelPlan.node_acceptable
    (plan : AcceptedGraphKernelPlan) (kernel : IR.PlannedNodeKernel)
    (hkernel : kernel ∈ plan.graphPlan.kernels) :
    ({ op := kernel.op, capsule := kernel.capsule } : PlannedKernel).acceptable
      plan.policy = true := by
  have haccepted : plan.graphPlan.toCoalescedGroups.acceptable plan.policy = true := by
    rw [← plan.grouped_eq]
    exact plan.accepted
  exact plan.graphPlan.node_acceptable plan.policy haccepted kernel hkernel

end NN.Backend
