/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Audit

/-!
# Backend Kernel Groups

A `GraphKernelPlan` holds one capsule for each IR node that needs a kernel. `GroupedKernelPlan`
puts adjacent kernels with the same wire record (operator, shapes and attributes) and the same
full capsule in one audit group, so twenty consecutive ReLU nodes of one shape give one audit row
in place of twenty. Grouping affects the audit alone and leaves the graph's semantics and its
kernel launches as they are: nothing is fused.
-/

@[expose] public section

namespace TorchLean
namespace Backend

/-- One audit group: the ids of its source IR nodes, kept for diagnostics, and the wire record and
capsule they share. -/
structure KernelGroup where
  /-- Ids of the source IR nodes, kept for diagnostics. -/
  nodeIds : Array Nat
  /-- Wire record the nodes share. -/
  op : Op.Wire
  /-- Capsule the nodes share. -/
  capsule : KernelCapsule
  deriving Repr

/-- A graph kernel plan whose adjacent identical kernels share groups, for the trust-boundary
audit. -/
structure GroupedKernelPlan where
  /-- The audit groups, in graph order. -/
  groups : Array KernelGroup
  deriving Repr

namespace KernelGroup

/-- The group's wire record and capsule as a `PlannedKernel`. -/
def toPlannedKernel (g : KernelGroup) : PlannedKernel :=
  { op := g.op, capsule := g.capsule }

/--
Whether a planned node can join this audit group without losing contract evidence: it must have the
group's wire record and the identical capsule.

Capsule identity (`KernelCapsule.sameIdentity`) is too weak a test, because a caller can supply
capsules with the same name, operator, provider and device but different trust levels or numerical
policies.
-/
def canAppend (g : KernelGroup) (k : PlannedNodeKernel) : Bool :=
  g.op == k.op && decide (g.capsule = k.capsule)

/-- Append the node's id to the group's source node ids. -/
def appendNode (g : KernelGroup) (k : PlannedNodeKernel) : KernelGroup :=
  { g with nodeIds := g.nodeIds.push k.nodeId }

end KernelGroup

namespace GroupedKernelPlan

/-- Source IR node ids covered by the grouped plan, in graph order. -/
def nodeIds (p : GroupedKernelPlan) : Array Nat :=
  p.groups.flatMap (·.nodeIds)

/-- Selected capsule names, in group order. -/
def capsuleNames (p : GroupedKernelPlan) : Array String :=
  p.groups.map fun g => g.capsule.name

/-- One `PlannedKernel` per group: the rows the contract check reads. -/
def toKernelPlan (p : GroupedKernelPlan) : KernelPlan :=
  { kernels := p.groups.map KernelGroup.toPlannedKernel }

end GroupedKernelPlan

namespace PlannedNodeKernel

/-- Place one graph-planned node in its own group. -/
def toSingletonGroup (k : PlannedNodeKernel) : KernelGroup :=
  { nodeIds := #[k.nodeId], op := k.op, capsule := k.capsule }

end PlannedNodeKernel

namespace GraphKernelPlan

/-- Add a planned node to the last group when `KernelGroup.canAppend` allows it, and as a group of
its own otherwise. -/
def pushKernel (groups : Array KernelGroup) (k : PlannedNodeKernel) : Array KernelGroup :=
  match groups.back? with
  | some g =>
      if g.canAppend k then groups.pop.push (g.appendNode k) else groups.push k.toSingletonGroup
  | none => #[k.toSingletonGroup]

/-- Group the kernels of `p` from left to right with `pushKernel`, so adjacent nodes with the same
wire record and capsule share one audit group. -/
def toCoalescedGroups (p : GraphKernelPlan) : GroupedKernelPlan :=
  { groups := p.kernels.foldl pushKernel #[] }

end GraphKernelPlan

end Backend
end TorchLean
