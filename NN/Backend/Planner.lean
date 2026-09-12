/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Availability

/-!
# Backend Planner

The planner is the bridge between a semantic graph and backend capsules.

Given a kernel policy, an operation tag, and a capsule registry, it chooses an admissible capsule
or explains why none is available. Graph-aware layers recover operation tags from `NN.IR.OpKind`
and call the same selection per node.
-/

@[expose] public section

namespace NN
namespace Backend

/-- A selected kernel capsule for one graph operation. -/
structure PlannedKernel where
  op : BackendOp
  capsule : KernelCapsule
  deriving Repr

/-- One selected kernel capsule per requested operation. This record does not execute them. -/
structure KernelPlan where
  kernels : Array PlannedKernel
  deriving Repr

namespace KernelPlan

/-- Names of the selected backend capsules, useful for audits and logs. -/
def capsuleNames (p : KernelPlan) : Array String :=
  p.kernels.map fun k => k.capsule.name

end KernelPlan

/-- Choose a backend capsule for one operation. -/
def planOp (policy : KernelPolicy) (registry : Array KernelCapsule)
    (op : BackendOp) : Except String PlannedKernel := do
  match chooseCapsuleFor? policy op registry with
  | some capsule => pure { op, capsule }
  | none =>
      throw s!"no admissible kernel capsule for op {op.name} on device {policy.device.cliName}"

/-- Choose backend capsules for a sequence of operations. -/
def planOps (policy : KernelPolicy) (registry : Array KernelCapsule)
    (ops : Array BackendOp) : Except String KernelPlan := do
  let kernels ← ops.mapM (planOp policy registry)
  pure { kernels }

/-- Choose backend capsules after filtering the registry by build availability. -/
def planOpsAvailable (policy : KernelPolicy) (availability : Availability)
    (registry : Array KernelCapsule) (ops : Array BackendOp) : Except String KernelPlan :=
  planOps policy (availability.filterCapsules registry) ops

end Backend
end NN
