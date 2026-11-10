/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Availability

/-!
# Backend Planner

The planner connects the semantic graph to backend capsules. Given a kernel policy, a registered
operator with its attributes and a list of implementations, `planOp` picks an admissible capsule
among those offered for the operator, or fails with a message saying why none is available.
`planOps` plans a sequence of operators, and graph planning (`TorchLean.NN.Backend.Graph`) calls
`planOp` for the operator of each graph node. A `KernelPlan` or `GraphKernelPlan` records the
choices and runs nothing.
-/

@[expose] public section

namespace TorchLean
namespace Backend

/-- A selected kernel capsule for one operator, named by its wire record. -/
structure PlannedKernel where
  /-- Wire record of the operator. -/
  op : Op.Wire
  /-- The selected capsule. -/
  capsule : KernelCapsule
  deriving DecidableEq, Repr

/-- One selected kernel capsule per requested operation, in request order. -/
structure KernelPlan where
  /-- The selected kernels, in request order. -/
  kernels : Array PlannedKernel
  deriving DecidableEq, Repr

namespace KernelPlan

/-- Names of the selected capsules, in plan order, for audits and logs. -/
def capsuleNames (p : KernelPlan) : Array String :=
  p.kernels.map fun k => k.capsule.name

end KernelPlan

/-- The name of the first capsule in `capsules` whose identity (`KernelCapsule.sameIdentity`)
occurs more than once, if there is one. -/
def firstDuplicateCapsuleIdentity? (capsules : Array KernelCapsule) : Option String :=
  capsules.findSome? fun capsule =>
    if 1 < capsules.countP capsule.sameIdentity then
      some capsule.name
    else
      none

/-- Choose a backend capsule for one operator with `chooseCapsuleFor?`, failing when none is
admissible. Two offered capsules with the same name, operator, provider and device are rejected,
since their order would decide the choice. -/
def planOp (policy : KernelPolicy) (impls : Array Implementation) {ins outs : List Port}
    (op : Op ins outs) : Except String PlannedKernel := do
  if let some name := firstDuplicateCapsuleIdentity? (Implementation.candidates impls op) then
    throw s!"duplicate backend capsule identity `{name}`"
  match chooseCapsuleFor? policy op impls with
  | some capsule => pure { op := Op.encode ⟨ins, outs, op⟩, capsule }
  | none =>
      throw <| s!"no admissible kernel capsule for op {(Op.encode ⟨ins, outs, op⟩).tag} " ++
        s!"on device {policy.device.cliName}"

/-- Choose backend capsules for a sequence of operators, failing at the first one `planOp`
rejects. -/
def planOps (policy : KernelPolicy) (impls : Array Implementation)
    (ops : Array (Σ ins outs, Op ins outs)) : Except String KernelPlan := do
  let kernels ← ops.mapM fun e => planOp policy impls e.2.2
  pure { kernels }

/-- Choose backend capsules after filtering the implementations by build availability. -/
def planOpsAvailable (policy : KernelPolicy) (availability : Availability)
    (impls : Array Implementation) (ops : Array (Σ ins outs, Op ins outs)) :
    Except String KernelPlan :=
  planOps policy (availability.filter impls) ops

/-- Backend choice for one graph node: its id, its operator's wire record and the capsule. -/
structure PlannedNodeKernel where
  /-- Id of the source IR node. -/
  nodeId : Nat
  /-- Wire record of the node's operator. -/
  op : Op.Wire
  /-- The selected capsule. -/
  capsule : KernelCapsule
  deriving Repr

/-- The kernels selected for a graph, each with the id of its source IR node. -/
structure GraphKernelPlan where
  /-- The selected kernels, in graph order. -/
  kernels : Array PlannedNodeKernel
  deriving Repr

namespace GraphKernelPlan

/-- Node ids covered by backend kernels, in graph order. -/
def nodeIds (p : GraphKernelPlan) : Array Nat :=
  p.kernels.map (·.nodeId)

/-- Selected capsule names, in graph order. -/
def capsuleNames (p : GraphKernelPlan) : Array String :=
  p.kernels.map fun k => k.capsule.name

end GraphKernelPlan

end Backend
end TorchLean
