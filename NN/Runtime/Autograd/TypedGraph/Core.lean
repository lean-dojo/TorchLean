/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.BackwardGraphData

/-!
# Typed Graph Core

Shape-indexed graph execution and its runtime-tape lowering.

This module exposes the "approach (a)" workflow:
1) Build an executable SSA/DAG graph (`Proofs.Autograd.Algebra.GraphData`).
2) Lower it to a runtime `Tape` with `Graph.lowerGraphDataToTape`.
3) Run `Tape.backwardDenseFrom` / `Tape.backwardDenseAll`.

`GraphData` is executable data, not a derivative-correctness certificate. The lowering theorem
shows that the tape implements `GraphData.backpropAllCtx`. To prove that this operation is the
adjoint of the graph JVP, construct the proof-carrying `Proofs.Autograd.Algebra.Graph`, whose nodes
include their local adjointness laws.

Notes / trust boundaries:
- If you instantiate `α := Float` or `α := FloatLib.Floats.ExecFloat.Binary 8 23`, you get an
  executable engine,
  but connecting those runs to real hardware semantics is treated as a trusted interface.
- The proof-carrying graph (`Proofs.Autograd.Algebra.Graph`) is available for backends
  where you can actually discharge algebraic/calc correctness assumptions (e.g. `ℝ`, `ℚ`).

## Main declarations

- `NN.Runtime.Autograd.TypedGraph.GraphM` is the small authoring DSL for typed graphs.
- `NN.Runtime.Autograd.IRExec` bridges `NN.IR.Graph` to executable graph data.
- `NN.Runtime.Autograd.IRExec.Correctness` proves the forward-correctness lemmas.

See also:
- Equality between lowered-tape backprop and executable graph backprop:
  `NN/Proofs/Autograd/Runtime/Link.lean`
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TypedGraph

open Spec TorchLean
open TorchLean

open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/--
Executable SSA/DAG graph for typed graph execution.

This is `Proofs.Autograd.Algebra.GraphData` specialized to:
- `Δ := Unit` (no extra opaque environment threaded through evaluation), and
- the `Runtime.Autograd.TypedGraph` namespace.
-/
abbrev GraphData (α : Type) [TorchLean.Storage α] (Γ : List Shape) (ss : List Shape) :=
  Proofs.Autograd.Algebra.GraphData α Unit Γ ss

/--
Lower an executable `GraphData` into a runtime tape.

This is the bridge from the shape-indexed SSA representation to the runtime tape engine:
`Graph.lowerGraphDataToTape` emits a `Runtime.Autograd.Tape` whose nodes replay the graph and whose
backward closures implement the graph's VJP rules.

The graph remains the persistent artifact; the tape contains the runtime closures needed for one
execution and reverse pass.
-/
def lowerToTape {α : Type} [TorchLean.Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Γ ss) (x : TorchLean.TensorPack α Γ) :
    Runtime.Autograd.Tape α × TorchLean.TensorPack α (Γ ++ ss) :=
  Proofs.Autograd.Algebra.Graph.lowerGraphDataToTape (α := α) (Δ := Unit) (Γ := Γ) (ss := ss) g x ()

/-- Validate and evaluate nodes in one pass while constructing their reverse-mode tape. -/
def lowerToTapeChecked {α Δ : Type} [TorchLean.Storage α]
    {Γ ss : List Shape} (graph : Proofs.Autograd.Algebra.GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) :
    Runtime.Autograd.Result
      (Runtime.Autograd.Tape α × TorchLean.TensorPack α (Γ ++ ss)) :=
  match graph with
  | .nil =>
      pure (Graph.addLeaves Runtime.Autograd.Tape.empty inputs,
        TorchLean.TensorPack.cast (h := (List.append_nil Γ).symm) inputs)
  | .snoc (ss := previousShapes) (τ := outputShape) previous node => do
      let (tape, context) ← lowerToTapeChecked previous inputs data
      node.validate context data
      let value := node.forward context data
      let (nextTape, _) := tape.addNode (Graph.lowerNode (some "typed-graph") node context data)
      let nextContext := TorchLean.TensorPack.cast
        (h := List.append_assoc Γ previousShapes [outputShape])
        (TorchLean.TensorPack.snoc context value)
      pure (nextTape, nextContext)

/-- Successful checked lowering produces exactly the existing pure runtime tape and context. -/
theorem lowerToTapeChecked_eq {α Δ : Type} [TorchLean.Storage α]
    {Γ ss : List Shape} (graph : Proofs.Autograd.Algebra.GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ)
    (result : Runtime.Autograd.Tape α × TorchLean.TensorPack α (Γ ++ ss))
    (checked : lowerToTapeChecked graph inputs data = .ok result) :
    result = Graph.lowerGraphDataToTape graph inputs data := by
  induction graph with
  | nil =>
      change Except.ok _ = Except.ok result at checked
      simpa only [Graph.lowerGraphDataToTape] using (Except.ok.inj checked).symm
  | snoc previous node ih =>
      cases previousResult : lowerToTapeChecked previous inputs data with
      | error message =>
          simp only [lowerToTapeChecked, previousResult, Bind.bind, Except.bind] at checked
          cases checked
      | ok pair =>
          obtain ⟨tape, context⟩ := pair
          have same := ih (tape, context) previousResult
          simp only [lowerToTapeChecked, previousResult, Bind.bind, Except.bind] at checked
          cases validation : node.validate context data with
          | error message =>
              simp only [validation] at checked
              cases checked
          | ok done =>
              cases done
              simp only [validation, Pure.pure, Except.pure, Except.ok.injEq] at checked
              subst result
              simp only [Graph.lowerGraphDataToTape, ← same]

/-- Validate runtime domains while evaluating the primal and tangent contexts together. -/
def jvpChecked {α Δ : Type} [TorchLean.Storage α]
    {Γ ss : List Shape} (graph : Proofs.Autograd.Algebra.GraphData α Δ Γ ss)
    (inputs tangents : TorchLean.TensorPack α Γ) (data : Δ) :
    Runtime.Autograd.Result
      (TorchLean.TensorPack α (Γ ++ ss) × TorchLean.TensorPack α (Γ ++ ss)) :=
  match graph with
  | .nil =>
      pure (TorchLean.TensorPack.cast (h := (List.append_nil Γ).symm) inputs,
        TorchLean.TensorPack.cast (h := (List.append_nil Γ).symm) tangents)
  | .snoc (ss := previousShapes) (τ := outputShape) previous node => do
      let (context, tangentContext) ← jvpChecked previous inputs tangents data
      node.validate context data
      let value := node.forward context data
      let tangent := node.jvp context tangentContext data
      let extend := fun (context : TorchLean.TensorPack α (Γ ++ previousShapes))
          (value : Tensor α outputShape) =>
        TorchLean.TensorPack.cast (h := List.append_assoc Γ previousShapes [outputShape])
          (TorchLean.TensorPack.snoc context value)
      pure (extend context value, extend tangentContext tangent)

/-- Successful checked JVP evaluation agrees with the pure primal and tangent semantics. -/
theorem jvpChecked_eq {α Δ : Type} [TorchLean.Storage α]
    {Γ ss : List Shape} (graph : Proofs.Autograd.Algebra.GraphData α Δ Γ ss)
    (inputs tangents : TorchLean.TensorPack α Γ) (data : Δ)
    (result : TorchLean.TensorPack α (Γ ++ ss) × TorchLean.TensorPack α (Γ ++ ss))
    (checked : jvpChecked graph inputs tangents data = .ok result) :
    result = (graph.eval inputs data, graph.jvpCtx inputs tangents data) := by
  induction graph with
  | nil =>
      change Except.ok _ = Except.ok result at checked
      simpa only [Proofs.Autograd.Algebra.GraphData.eval,
        Proofs.Autograd.Algebra.GraphData.jvpCtx] using (Except.ok.inj checked).symm
  | snoc previous node ih =>
      cases previousResult : jvpChecked previous inputs tangents data with
      | error message =>
          simp only [jvpChecked, previousResult, Bind.bind, Except.bind] at checked
          cases checked
      | ok pair =>
          obtain ⟨context, tangentContext⟩ := pair
          have same := ih (context, tangentContext) previousResult
          simp only [jvpChecked, previousResult, Bind.bind, Except.bind] at checked
          cases validation : node.validate context data with
          | error message =>
              simp only [validation] at checked
              cases checked
          | ok done =>
              cases done
              simp only [validation, Pure.pure, Except.pure, Except.ok.injEq] at checked
              subst result
              obtain ⟨rfl, rfl⟩ := Prod.mk.inj same
              rfl

/--
Run reverse-mode backpropagation from a typed output reference and cotangent seed.

The output may be an original graph input or any recorded node; it need not be the final node.
The full typed context is seeded explicitly, matching `GraphData.backpropCtx` and the tape-lowering
theorem. In particular, this path does not rely on reachability pruning or an unstated assumption
that every stored VJP maps a zero cotangent to zero.
-/
def backwardDenseAllFrom {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {Γ : List Shape} {ss : List Shape} {τ : Shape}
    (t : Runtime.Autograd.Tape α) (output : Idx (Γ ++ ss) τ) (seed : Tensor α τ) :
    Runtime.Autograd.Result (Array (Spec.SomeTensor α)) :=
  Runtime.Autograd.Tape.backwardDenseFrom (t := t)
    (grads0 := TorchLean.TensorPack.toShapeErasedArray
      (Proofs.Autograd.Algebra.TensorPack.single output seed))

/--
Lowering a typed graph to the runtime tape preserves reverse mode from any typed output reference.

The result covers outputs that are inputs or intermediate nodes, not only the final recorded node.
It states fidelity to the executable VJP stored in `GraphData`; derivative correctness requires the
separate local laws carried by `Proofs.Autograd.Algebra.Node`.
-/
theorem backwardDenseAllFrom_lowerToTape_eq_backpropAllCtx
    {α : Type} [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} {τ : Shape}
    (g : GraphData α Γ ss) (x : TorchLean.TensorPack α Γ) (output : Idx (Γ ++ ss) τ)
    (seed : Tensor α τ) :
    backwardDenseAllFrom (lowerToTape g x).1 output seed =
      .ok
        (TorchLean.TensorPack.toShapeErasedArray
          (Proofs.Autograd.Algebra.GraphData.backpropAllCtx
            g x () (Proofs.Autograd.Algebra.TensorPack.single output seed))) := by
  simpa [backwardDenseAllFrom, lowerToTape] using
    (Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx
      (α := α) (Δ := Unit) (Γ := Γ) (ss := ss) g x ()
      (Proofs.Autograd.Algebra.TensorPack.single output seed))

end TypedGraph
end Autograd
end Runtime
