/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Graph
public import NN.Backend.Planner

/-!
# Kernel Selection for IR Graphs

Adapter from TorchLean's op-tagged IR to backend capsules.

This is intentionally a planning layer, not an evaluator. The graph remains the semantic object;
the planner selects which backend contract should implement each node or node family.
-/

@[expose] public section

namespace NN
namespace Backend
namespace IR

/--
Backend operation tag for an IR op, or `none` for a node that needs no backend kernel.

These tags are the same vocabulary used by backend capsules and runtime reports.
If an op maps to a tag that has no capsule for the selected profile, planning fails at the backend
boundary instead of silently using a broader bucket. Pointwise operators map only to a backend
operation with the same meaning: negation, true division and the tanh GELU. `planNode?` rejects
the others.
-/
def op? : NN.IR.OpKind → Option BackendOp
  | .input => none
  | .const .. => none
  | .custom .. => some .custom
  | .detach => none
  | .randUniform .. => some .randUniform
  | .bernoulliMask .. => some .bernoulliMask
  | .add => some .add
  | .sub => some .sub
  | .mulElem => some .mul
  | .abs => some .abs
  | .sqrt => some .sqrt
  | .inv => some .inv
  | .maxElem => some .max
  | .minElem => some .min
  | .relu => some .relu
  | .tanh => some .tanh
  | .sigmoid => some .sigmoid
  | .silu => some .silu
  | .softplus => some .softplus
  | .safeLog => some .safeLog
  | .exp => some .exp
  | .log => some .log
  | .sin => some .sin
  | .cos => some .cos
  | .mseLoss => some .mseLoss
  | .matmul => some .matmul
  | .scaledMatmul => some .scaledMatmul
  | .linear => some .linear
  | .conv .. => some .conv
  | .maxPool .. => some .maxPool
  | .avgPool .. => some .avgPool
  | .broadcastTo .. => some .broadcast
  | .reduceSum .. => some .reduceSum
  | .reduceMean .. => some .reduceMean
  | .sum => some .reduceSum
  | .softmax .. => some .softmax
  | .hardMaskedSoftmax .. => some .hardMaskedSoftmax
  | .topKMask .. => some .topKMask
  | .layernorm .. => some .layerNorm
  | .reshape .. => some .reshape
  | .flatten .. => some .reshape
  | .concat .. => some .concat
  | .slice .. => some .slice
  | .transpose .. => some .permute
  | .permute .. => some .permute
  | .batchNormEval .. => some .batchNorm
  | .pointwise .neg => some .neg
  | .pointwise (.div .none) => some .div
  | .pointwise (.gelu .tanh) => some .gelu
  | .pointwise _ => none

/-- Backend choice for one concrete IR node. -/
structure PlannedNodeKernel where
  nodeId : Nat
  kind : NN.IR.OpKind
  op : BackendOp
  capsule : KernelCapsule
  deriving Repr

/-- Selected kernel capsules with the source IR node identity preserved. -/
structure GraphKernelPlan where
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

/-- Plan a single IR node when it corresponds to runtime work.

`available` is the capsule registry already restricted by `Availability.filterCapsules`, so a graph
planner filters once rather than once per node. -/
def planNode? (policy : KernelPolicy) (available : Array KernelCapsule) (n : NN.IR.Node) :
    Except String (Option PlannedNodeKernel) := do
  match op? n.kind with
  | none =>
      -- Pointwise operators are always runtime work; one without a backend operation fails here.
      if let .pointwise op := n.kind then
        throw s!"backend planning: node {n.id}: {op.describe} has no backend operation"
      pure none
  | some op =>
      let k ← planOp policy available op
      pure <| some
        { nodeId := n.id
          kind := n.kind
          op := k.op
          capsule := k.capsule }

/-- Plan every runtime-relevant node in graph order. -/
def planGraph (policy : KernelPolicy) (availability : Availability)
    (registry : Array KernelCapsule) (g : NN.IR.Graph) : Except String GraphKernelPlan := do
  let available := availability.filterCapsules registry
  let mut kernels : Array PlannedNodeKernel := #[]
  for n in g.nodes do
    match (← planNode? policy available n) with
    | none => pure ()
    | some k => kernels := kernels.push k
  pure { kernels }

/-- Check graph well-formedness, then plan every runtime-relevant node. -/
def checkedPlanGraph (policy : KernelPolicy) (availability : Availability)
    (registry : Array KernelCapsule) (g : NN.IR.Graph) : Except String GraphKernelPlan := do
  g.checkWellFormed
  planGraph policy availability registry g

end IR
end Backend
end NN
