/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Profile
public import TorchLean.NN.IR.Eval

/-!
# Kernel Selection for IR Graphs

`TorchLean.IR.Graph.plan` checks an operator graph with `TorchLean.IR.Graph.check` and then selects
a capsule for the registered operator of each node through `planOp`, the selection eager execution
uses, so each planned kernel carries its operator's wire record. Planning picks kernels and
evaluates nothing: the graph is the semantic object. Inputs, constants and `detach` need no kernel.
Every other operator node gets a capsule, or planning fails with the error `planOp` reports for it.
`IR.Graph.accept` plans under a backend profile, groups the plan and accepts it only when every
group passes the profile's contract check.
-/

@[expose] public section

namespace TorchLean.IR

open Backend GraphKernelPlan

/-- The kernel of the node at index `id`: `none` for an input, a constant or `Op.detach`, which
need no kernel, and otherwise `id` with the wire record and capsule that `planOp` selects for the
node's operator. The caller has already restricted `impls` with `Availability.filter`. -/
def Node.plan? (policy : KernelPolicy) (impls : Array Implementation) (id : Nat) :
    Node → Except String (Option PlannedNodeKernel)
  | .input .. | .const _ => pure none
  | .op ⟨_, _, .detach⟩ _ => pure none
  | .op ⟨_, _, o⟩ _ => do
      let k ← planOp policy impls o
      pure (some { nodeId := id, op := k.op, capsule := k.capsule })

/-- The kernel plan of `g`: once `Graph.check` passes, every node is planned in order against the
implementations `availability` admits. `IR.Graph.plan_sound` states what each planned kernel
records. -/
def Graph.plan (g : Graph) (policy : KernelPolicy) (availability : Availability)
    (impls : Array Implementation) : Except String GraphKernelPlan := do
  g.check
  let available := availability.filter impls
  let kernels ← g.nodes.zipIdx.filterMapM fun (n, id) => n.plan? policy available id
  pure { kernels }

/-- Plan, group and contract-check `g` under the profile `p`, after `Registry.validate` has
rejected duplicate implementation names. -/
def Graph.accept (g : Graph) (p : Profile) : Except String GraphKernelPlanResult := do
  Registry.validate p.implementations
  pure (GraphKernelPlan.accept (← g.plan p.policy p.availability p.implementations) p.policy)

end TorchLean.IR
