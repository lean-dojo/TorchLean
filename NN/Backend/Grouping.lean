/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Audit
public import NN.Backend.IR

/-!
# Backend Kernel Groups

`GraphKernelPlan` chooses a capsule for each runtime-relevant IR node. A `GroupedKernelPlan` places
adjacent choices with the same operation and complete capsule in one audit group, so a graph with
twenty consecutive ReLU nodes produces one audit row instead of twenty. Grouping does not translate
graph semantics, fuse operations, or claim that a group executes as one kernel launch.
-/

@[expose] public section

namespace NN
namespace Backend

/-- One audit group, with source IR node ids retained for diagnostics. -/
structure KernelGroup where
  nodeIds : Array Nat
  op : BackendOp
  capsule : KernelCapsule
  deriving Repr

/-- A node-level graph plan grouped for trust-boundary audit. -/
structure GroupedKernelPlan where
  groups : Array KernelGroup
  deriving Repr

namespace KernelGroup

/-- Project the operation and capsule selected for this group. -/
def toPlannedKernel (g : KernelGroup) : PlannedKernel :=
  { op := g.op, capsule := g.capsule }

/--
Whether a planned node can share this audit group without losing contract evidence.

Matching dispatch identities alone is insufficient: a caller can supply capsules with the same
name, operation, provider, and device but different trust or numerical policies.
-/
def canAppend (g : KernelGroup) (k : IR.PlannedNodeKernel) : Bool :=
  g.op == k.op && decide (g.capsule = k.capsule)

/-- Append a node to an existing group, preserving source-node provenance. -/
def appendNode (g : KernelGroup) (k : IR.PlannedNodeKernel) : KernelGroup :=
  { g with nodeIds := g.nodeIds.push k.nodeId }

end KernelGroup

namespace GroupedKernelPlan

/-- Source IR node ids covered by the grouped plan, in graph order. -/
def nodeIds (p : GroupedKernelPlan) : Array Nat :=
  p.groups.flatMap (·.nodeIds)

/-- Selected capsule names, in group order. -/
def capsuleNames (p : GroupedKernelPlan) : Array String :=
  p.groups.map fun g => g.capsule.name

/-- Project groups to the capsule rows consumed by the contract check. -/
def toKernelPlan (p : GroupedKernelPlan) : KernelPlan :=
  { kernels := p.groups.map KernelGroup.toPlannedKernel }

/-- Audit the selected backend boundaries of the grouped plan. -/
def audit (p : GroupedKernelPlan) : KernelPlanAudit :=
  p.toKernelPlan.audit

end GroupedKernelPlan

namespace IR

namespace PlannedNodeKernel

/-- Place one graph-planned node in its own group. -/
def toSingletonGroup (k : PlannedNodeKernel) : KernelGroup :=
  { nodeIds := #[k.nodeId], op := k.op, capsule := k.capsule }

end PlannedNodeKernel

namespace GraphKernelPlan

/-- Add one planned node, sharing the preceding group when the boundary is identical. -/
def pushKernel (groups : Array KernelGroup) (k : PlannedNodeKernel) : Array KernelGroup :=
  match groups.back? with
  | some g =>
      if g.canAppend k then groups.pop.push (g.appendNode k) else groups.push k.toSingletonGroup
  | none => #[k.toSingletonGroup]

/-- Adjacent nodes with the same backend boundary share one audit group. -/
def toCoalescedGroups (p : GraphKernelPlan) : GroupedKernelPlan :=
  { groups := p.kernels.foldl pushKernel #[] }

end GraphKernelPlan

end IR

end Backend
end NN
