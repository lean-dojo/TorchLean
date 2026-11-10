/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Planner.Soundness

/-!
# Lossless backend audit grouping

Coalescing a graph plan into audit groups preserves every node id, operator and complete capsule,
in order and with repeats. Accepting the grouped plan therefore accepts every node of the source
graph plan. The results concern the planner and its declared contracts, and native kernel
execution is outside their scope.
-/

@[expose] public section

namespace TorchLean.Backend

/-- A group can absorb a node if and only if the node has the group's operation and complete
capsule. -/
theorem KernelGroup.canAppend_iff (group : KernelGroup) (kernel : PlannedNodeKernel) :
    group.canAppend kernel = true ↔
      group.op = kernel.op ∧ group.capsule = kernel.capsule := by
  simp [KernelGroup.canAppend]

private theorem pushKernel_expands
    (groups : Array KernelGroup) (kernel : PlannedNodeKernel) :
    (GraphKernelPlan.pushKernel groups kernel).flatMap
        (fun group => group.nodeIds.map fun id => (id, group.op, group.capsule)) =
      (groups.flatMap
        (fun group => group.nodeIds.map fun id => (id, group.op, group.capsule))).push
          (kernel.nodeId, kernel.op, kernel.capsule) := by
  by_cases hempty : groups = #[]
  · subst groups
    simp [GraphKernelPlan.pushKernel, PlannedNodeKernel.toSingletonGroup]
  · obtain ⟨groups, group, rfl⟩ := Array.exists_push_of_ne_empty hempty
    simp only [GraphKernelPlan.pushKernel, Array.back?_push]
    by_cases hsame : group.canAppend kernel = true
    · rcases (group.canAppend_iff kernel).mp hsame with ⟨hop, hcapsule⟩
      simp [hsame, KernelGroup.appendNode, hop, hcapsule]
    · simp [hsame, PlannedNodeKernel.toSingletonGroup]

private theorem foldl_expands
    (kernels : List PlannedNodeKernel) (groups : Array KernelGroup) :
    (kernels.foldl GraphKernelPlan.pushKernel groups).flatMap
        (fun group => group.nodeIds.map fun id => (id, group.op, group.capsule)) =
      groups.flatMap (fun group => group.nodeIds.map fun id => (id, group.op, group.capsule)) ++
        (kernels.map fun kernel => (kernel.nodeId, kernel.op, kernel.capsule)).toArray := by
  induction kernels generalizing groups with
  | nil => simp
  | cons kernel kernels ih =>
      rw [List.foldl_cons, ih, pushKernel_expands]
      simp [-Array.append_singleton, Array.push_eq_append, Array.append_assoc]

/--
Expanding the coalesced audit groups gives back the original node ids, operations and full
capsules as an array, so order and multiplicity carry over, repeated node ids included.
-/
theorem GraphKernelPlan.toCoalescedGroups_expands (plan : GraphKernelPlan) :
    plan.toCoalescedGroups.groups.flatMap
        (fun group => group.nodeIds.map fun id => (id, group.op, group.capsule)) =
      plan.kernels.map (fun kernel => (kernel.nodeId, kernel.op, kernel.capsule)) := by
  rcases plan with ⟨⟨kernels⟩⟩
  simpa [GraphKernelPlan.toCoalescedGroups] using foldl_expands kernels #[]

/-- Every original node occurs in a group with exactly its operation and capsule. -/
theorem GraphKernelPlan.node_mem_coalesced
    (plan : GraphKernelPlan) (kernel : PlannedNodeKernel)
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

/-- A grouped plan passes the acceptance gate if and only if every group, projected to a planned
kernel, passes it. -/
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

/-- If the coalesced groups pass the policy and evidence gate, each node of the original plan
passes it with its own operation and capsule. -/
theorem GraphKernelPlan.node_acceptable
    (plan : GraphKernelPlan) (policy : KernelPolicy)
    (haccepted : plan.toCoalescedGroups.acceptable policy = true)
    (kernel : PlannedNodeKernel) (hkernel : kernel ∈ plan.kernels) :
    ({ op := kernel.op, capsule := kernel.capsule } : PlannedKernel).acceptable policy = true := by
  rcases plan.node_mem_coalesced kernel hkernel with
    ⟨group, hgroup, _, hop, hcapsule⟩
  have haccepted := (plan.toCoalescedGroups.acceptable_iff policy).mp haccepted group hgroup
  simpa [KernelGroup.toPlannedKernel, hop, hcapsule] using haccepted

/-- Every node in an accepted graph plan passes the same complete policy and evidence gate. -/
theorem AcceptedGraphKernelPlan.node_acceptable
    (plan : AcceptedGraphKernelPlan) (kernel : PlannedNodeKernel)
    (hkernel : kernel ∈ plan.graphPlan.kernels) :
    ({ op := kernel.op, capsule := kernel.capsule } : PlannedKernel).acceptable
      plan.policy = true := by
  have haccepted : plan.graphPlan.toCoalescedGroups.acceptable plan.policy = true := by
    rw [← plan.grouped_eq]
    exact plan.accepted
  exact plan.graphPlan.node_acceptable plan.policy haccepted kernel hkernel

end TorchLean.Backend
