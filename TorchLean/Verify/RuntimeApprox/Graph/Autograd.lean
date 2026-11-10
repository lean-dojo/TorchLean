/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Graph.Backward

/-!
# Link `TorchLean.RevGraph` to the executable `TorchLean.Tape.GraphData`

`RevGraph.toGraphData` erases the spec and bound data of a graph and keeps its runtime forward and
VJP closures. On the erased graph, `GraphData.eval` and `GraphData.backpropCtx` agree with
`RevGraph.evalRuntime` and `RevGraph.backpropRuntime`. Both sides hold contexts as
`TorchLean.TensorPack`, so the bridge needs no conversion between context representations.

Only forward evaluation and reverse accumulation carry over. The required `NodeData.jvp` field holds
the forward map and ignores its tangent argument, so forward-mode differentiation must not use it.
The bridge proves neither correspondence with native kernels nor that the supplied reverse maps
are derivatives.

`TorchLean.Tape.GraphData` threads an extra non-differentiable environment `Δ` through
evaluation. The runtime-approximation graphs use none, so the executable layer is instantiated at
`Δ := Unit` and evaluation and backprop receive `()`.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

noncomputable section

open Tape

namespace RevNode

variable {α : Type} {toSpec : α → ℝ}

/--
Erase a `RevNode` into an executable `Tape.NodeData` with no environment
(`Δ := Unit`). The `jvp` field repeats the forward map as a placeholder, so the conversion supports
forward evaluation and VJP-based backprop only.
-/
def toNodeData {Γ : List Shape} {τ : Shape}
    (node : RevNode (α := α) toSpec Γ τ) :
    NodeData α Unit Γ τ :=
  { forward := fun ctxA _d => node.forwardRuntime ctxA
    jvp := fun ctxA _dctxA _d => node.forwardRuntime ctxA
    vjp := fun ctxA _d δ => node.vjpRuntime ctxA δ }

end RevNode

namespace RevGraph

variable {α : Type} {toSpec : α → ℝ}

/--
Erase a `RevGraph` into an executable `Tape.GraphData` with `Δ := Unit`,
converting each node with `RevNode.toNodeData`. The spec and bound data are dropped, and the runtime
`forward` and `vjp` closures are kept.
-/
def toGraphData {Γ : List Shape} {ss : List Shape}
    (g : RevGraph (α := α) toSpec Γ ss) : GraphData α Unit Γ ss :=
  match g with
  | .nil => .nil
  | .snoc g node => .snoc (toGraphData (ss := _) g) (RevNode.toNodeData (α := α) (toSpec := toSpec)
    node)

/--
Erasing a `RevGraph` preserves its runtime forward semantics: `GraphData.eval` of `toGraphData g`
at `xR` and `()` equals `RevGraph.evalRuntime g xR`.
-/
theorem eval_toGraphData {Γ : List Shape} :
    {ss : List Shape} →
    (g : RevGraph (α := α) toSpec Γ ss) →
    (xR : TensorPack α Γ) →
      GraphData.eval (α := α) (Δ := Unit) (Γ := Γ) (ss := ss)
          (g := toGraphData (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g) xR ()
        =
      RevGraph.evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss :=
        ss) g xR := by
  intro ss g xR
  induction g with
  | nil =>
      simp [toGraphData, GraphData.eval, RevGraph.evalRuntime,
        RevGraph.toFwdGraph, FwdGraph.evalRuntime]
  | snoc g node ih =>
      simp [toGraphData, GraphData.eval, RevGraph.evalRuntime,
        RevGraph.toFwdGraph, FwdGraph.evalRuntime, RevNode.toNodeData,
          ih]

/--
Erasing a `RevGraph` preserves its runtime backward semantics: for the same input context `xR` and
seed cotangents `seedR`, `GraphData.backpropCtx` of `toGraphData g` equals
`RevGraph.backpropRuntime g xR seedR`.
-/
theorem backpropCtx_toGraphData {Γ : List Shape} [Add α] :
    {ss : List Shape} →
    (g : RevGraph (α := α) toSpec Γ ss) →
    (xR : TensorPack α Γ) →
    (seedR : TensorPack α (Γ ++ ss)) →
      GraphData.backpropCtx (α := α) (Δ := Unit) (Γ := Γ) (ss := ss)
          (g := toGraphData (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g) xR () seedR
        =
      RevGraph.backpropRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss
        := ss) g xR seedR := by
  intro ss g xR seedR
  induction g with
  | nil =>
      simp [toGraphData, GraphData.backpropCtx,
        RevGraph.backpropRuntime]
  | snoc g node ih =>
      have hEval :=
        eval_toGraphData (α := α) (toSpec := toSpec) (Γ := Γ) (ss := _) g xR
      simp [toGraphData, GraphData.backpropCtx,
        RevGraph.backpropRuntime,
        RevNode.toNodeData, ih, hEval]

end RevGraph

end
end TorchLean
