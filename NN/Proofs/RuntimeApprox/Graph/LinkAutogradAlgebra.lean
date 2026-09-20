/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.BackwardApprox

/-!
# Link `Proofs.RuntimeApprox.RevGraph` to the executable `Proofs.Autograd.Algebra.GraphData`

This is a definitional/structural bridge:

- `RevGraph.toGraphData` erases spec/bound metadata and keeps the runtime forward/VJP closures.
- We show `GraphData.eval` / `GraphData.backpropCtx` coincide with
  `RevGraph.evalRuntime` / `RevGraph.backpropRuntime`.

Both sides use `TorchLean.TensorPack` directly, so the bridge does not convert between
different context representations.

Note on environments (`Δ`):
`Proofs.Autograd.Algebra.GraphData` is parameterized by an extra (non-differentiable) environment
`Δ` threaded through evaluation. The runtime-approximation graphs here do not use such an
environment, so we instantiate the executable layer at `Δ := Unit` and pass `()` during
evaluation/backprop.
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

namespace LinkAutogradAlgebra

open Proofs.Autograd.Algebra

namespace RevNode

variable {α : Type} {toSpec : α → SpecScalar}

/--
Erase a `RuntimeApprox.RevNode` into an executable `Autograd.Algebra.NodeData`.

We take `Δ := Unit` (no extra environment) and ignore the JVP input, since this bridge is only
used for forward evaluation and VJP-based backprop.
-/
def toNodeData {Γ : List Shape} {τ : Shape}
    (node : Proofs.RuntimeApprox.RevNode (α := α) toSpec Γ τ) :
    NodeData α Unit Γ τ :=
  { forward := fun ctxA _d => node.forwardRuntime ctxA
    jvp := fun ctxA _dctxA _d => node.forwardRuntime ctxA
    vjp := fun ctxA _d δ => node.vjpRuntime ctxA δ }

end RevNode

namespace RevGraph

variable {α : Type} {toSpec : α → SpecScalar}

/--
Erase a `RuntimeApprox.RevGraph` into executable `Autograd.Algebra.GraphData` (with `Δ := Unit`).

This forgets all spec/bound metadata and keeps only the runtime `forward` and `vjp` closures.
-/
def toGraphData {Γ : List Shape} {ss : List Shape}
    (g : Proofs.RuntimeApprox.RevGraph (α := α) toSpec Γ ss) : GraphData α Unit Γ ss :=
  match g with
  | .nil => .nil
  | .snoc g node => .snoc (toGraphData (ss := _) g) (RevNode.toNodeData (α := α) (toSpec := toSpec)
    node)

/--
Erasing a `RevGraph` into executable `GraphData` preserves the runtime **forward** semantics.

Informally: if you forget the spec and bound metadata and keep only the runtime closures, you still
evaluate to the same runtime context.
-/
theorem evalRuntime_of_toGraphData {Γ : List Shape} :
    {ss : List Shape} →
    (g : Proofs.RuntimeApprox.RevGraph (α := α) toSpec Γ ss) →
    (xR : TorchLean.TensorPack α Γ) →
      GraphData.eval (α := α) (Δ := Unit) (Γ := Γ) (ss := ss)
          (g := toGraphData (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g) xR ()
        =
      Proofs.RuntimeApprox.RevGraph.evalRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss :=
        ss) g xR := by
  intro ss g xR
  induction g with
  | nil =>
      simp [toGraphData, GraphData.eval, Proofs.RuntimeApprox.RevGraph.evalRuntime,
        Proofs.RuntimeApprox.RevGraph.toFwdGraph, FwdGraph.evalRuntime]
  | snoc g node ih =>
      simp [toGraphData, GraphData.eval, Proofs.RuntimeApprox.RevGraph.evalRuntime,
        Proofs.RuntimeApprox.RevGraph.toFwdGraph, FwdGraph.evalRuntime, RevNode.toNodeData,
          ih]

/--
Erasing a `RevGraph` into executable `GraphData` preserves the runtime **backward** semantics.

Informally: the executable `GraphData.backpropCtx` computes the same reverse-mode accumulation as
`RevGraph.backpropRuntime` when given the same runtime input context and seed cotangents.
-/
theorem backpropRuntime_of_toGraphData {Γ : List Shape} [Add α] :
    {ss : List Shape} →
    (g : Proofs.RuntimeApprox.RevGraph (α := α) toSpec Γ ss) →
    (xR : TorchLean.TensorPack α Γ) →
    (seedR : TorchLean.TensorPack α (Γ ++ ss)) →
      GraphData.backpropCtx (α := α) (Δ := Unit) (Γ := Γ) (ss := ss)
          (g := toGraphData (α := α) (toSpec := toSpec) (Γ := Γ) (ss := ss) g) xR () seedR
        =
      Proofs.RuntimeApprox.RevGraph.backpropRuntime (α := α) (toSpec := toSpec) (Γ := Γ) (ss
        := ss) g xR seedR := by
  intro ss g xR seedR
  induction g with
  | nil =>
      simp [toGraphData, GraphData.backpropCtx,
        Proofs.RuntimeApprox.RevGraph.backpropRuntime]
  | snoc g node ih =>
      have hEval :=
        evalRuntime_of_toGraphData (α := α) (toSpec := toSpec) (Γ := Γ) (ss := _) g xR
      simp [toGraphData, GraphData.backpropCtx,
        Proofs.RuntimeApprox.RevGraph.backpropRuntime,
        RevNode.toNodeData, ih, hEval]

end RevGraph

end LinkAutogradAlgebra

end
end RuntimeApprox
end Proofs
