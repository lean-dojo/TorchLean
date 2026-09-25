/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Check
public import NN.Runtime.Autograd.IRExec.Lowering

/-!
# IR To Executable Graph Lowering

Public entrypoint for validating and lowering a complete shared IR graph.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open TorchLean.Tensor
open Proofs.Autograd.Algebra
open NN.IR

/--
Lower an op-tagged IR graph into an executable `ForwardGraph`.

Requirements:
- Node id 0 must be `.input`.
- The graph must satisfy `Graph.checkWellFormed`.
- The external payload must contain entries for every `.const`/`.linear`/`.conv` node id.

This returns a `ForwardGraph` whose `eval` computes all node values in topological order. The
artifact is intentionally forward-only; it is distinct from the differentiable `Torch.TypedGraph`
used by the autograd lowering path.

This is the main API consumed by runtime callers that want executable evaluation while remaining
aligned with the shared `NN.IR.Graph` semantics.
-/
def lowerToForwardGraph
    {α : Type} [Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) : Except String (ForwardGraph α) := do
  g.checkWellFormed
  let n0 ← g.getNode 0
  match n0.kind with
  | .input =>
      let inShape := n0.outShape
      let stFinal ← Internal.buildFrom (α := α) (g := g) (payload := payload)
        (inShape := inShape) (i := 1) (st := (⟨[], .nil⟩ : Internal.State α inShape))
      let ⟨ss, gd⟩ := stFinal
      pure { inShape := inShape, ss := ss, body := gd }
  | _ =>
      throw s!"IRExec: node 0 is not `.input` (got {n0.kind.tag})"


/-- Validate an IR graph and evaluate its selected output on a shape-checked input.

Structural, declared-shape, payload, and lowering errors are returned to the caller. The input
shape is checked against node zero before execution. This evaluates the forward graph; it does
not compare floating-point results with a second semantics or assert numerical equivalence.

Each call lowers the graph again. For repeated execution, call `lowerToForwardGraph` once and
reuse the resulting `ForwardGraph.eval` or `ForwardGraph.denoteAll`. The array execution
optimization applies to that already-lowered artifact; lowering's shape-list appends can still
take quadratic time in the number of nodes.
-/
def evaluate
    {α : Type} [Storage α] [Context α] {σ : Shape}
    (g : NN.IR.Graph) (payload : Payload α) (x : Tensor α σ)
    (outputId : Fin g.nodes.size) : Except String (Spec.SomeTensor α) := do
  let graph ← lowerToForwardGraph g payload
  g.checkShapes
  let input ←
    if h : σ = graph.inShape then
      pure (Tensor.castShape x h)
    else
      throw s!"IRExec: input shape mismatch: tensor={repr σ}, graph={repr graph.inShape}"
  let values := graph.denoteAll input
  if h : outputId.val < values.size then
    pure (values[outputId.val]'h)
  else
    throw s!"IRExec: output index {outputId.val} exceeds value table size {values.size}"
