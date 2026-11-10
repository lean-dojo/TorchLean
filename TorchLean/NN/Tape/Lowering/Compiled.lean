/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.WithData
public import TorchLean.NN.Tape.SavedBackward
public import TorchLean.NN.Tape.Lowering.BackwardGraph

/-!
# Correctness of compiled typed-graph execution

These theorems relate checked compilation (`TorchLean.TypedGraph.compileChecked`) to the
lowered tape and to graph backpropagation. Their proofs use lemmas of the autograd proofs, which the
runtime modules do not import.

## Main declarations

- `TorchLean.Tape.SavedGraph.ofGraph_toTape`: the saved programs of a graph adapt to the
  tape of `GraphData.toTape`.
- `TorchLean.TypedGraph.compileChecked_context`: a successful compilation holds the context
  `GraphData.eval`.
- `TorchLean.TypedGraph.Compiled.backwardDenseFrom_eq`: its dense pass returns the
  gradients of `GraphData.backpropAllCtx`.
- `TorchLean.TypedGraph.Compiled.backwardDenseFrom_eq_tape`: the dense pass agrees with
  `Tape.backwardDenseFrom` on the lowered tape.
- `TorchLean.TypedGraph.WithData.vjpChecked_eq_of_lowerChecked`: a successful domain
  check returns the output and VJP of pure evaluation.
-/

@[expose] public section

namespace TorchLean.Tape.SavedGraph

variable {α : Type} [Storage α] {Γ : List Shape}

/-- Saved programs produce the same runtime tape, including its closures. -/
theorem ofGraph_toTape {Δ : Type} {ss : List Shape} (graph : GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) :
    (ofGraph graph inputs data).1.toTape inputs =
      (GraphData.toTape graph inputs data).1 := by
  induction graph with
  | nil => rfl
  | snoc previous node ih =>
      simp only [ofGraph, toTape, GraphData.toTape, ih, ofGraph_context_eq,
        NodeData.prepare_value_ofPack, TypedGraph.lowerPreparedNode_ofPack,
        GraphData.toTape_ctx_eq_eval]

end TorchLean.Tape.SavedGraph

namespace TorchLean.TypedGraph

open Tape
open TorchLean (Idx)

/-- Successful compilation has the original complete forward context. -/
theorem compileChecked_context {α Δ : Type} [Storage α] {Γ ss : List Shape}
    (graph : Tape.GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) (compiled : Compiled α Γ ss)
    (checked : compileChecked graph inputs data = .ok compiled) :
    compiled.context.toPack = graph.eval inputs data := by
  cases outcome : SavedGraph.lowerChecked graph inputs data with
  | error message => simp [compileChecked, outcome, Except.map] at checked
  | ok result =>
      simp only [compileChecked, outcome, Except.map, Except.ok.injEq] at checked
      subst compiled
      rw [SavedGraph.lowerChecked_eq _ _ _ _ outcome, SavedGraph.ofGraph_context_eq,
        TensorContext.toPack_ofPack]

/-- Compiled reverse execution agrees with full graph backpropagation for every seed pack. -/
theorem Compiled.backwardDenseFrom_eq {α Δ : Type} [Storage α] [Add α]
    {Γ ss : List Shape} (graph : Tape.GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) (compiled : Compiled α Γ ss)
    (checked : compileChecked graph inputs data = .ok compiled)
    (seed : TensorPack α (Γ ++ ss)) :
    compiled.backwardDenseFrom seed =
      (graph.backpropAllCtx inputs data seed).erase := by
  cases outcome : SavedGraph.lowerChecked graph inputs data with
  | error message => simp [compileChecked, outcome, Except.map] at checked
  | ok result =>
      simp only [compileChecked, outcome, Except.map, Except.ok.injEq] at checked
      subst compiled
      simp only [Compiled.backwardDenseFrom, SavedGraph.lowerChecked_eq _ _ _ _ outcome]
      have same := SavedGraph.ofGraph_backpropStored graph inputs data seed
      have arrays := congrArg TensorPack.erase same
      have materialized (ctx : TensorContext α (Γ ++ ss)) :
          ctx.toPack.erase = ctx.values :=
        congrArg TensorContext.values (TensorContext.ofPack_toPack ctx)
      rw [materialized] at arrays
      exact arrays

/-- `Compiled.backwardDenseFrom` agrees with `Tape.backwardDenseFrom` on the lowered tape. -/
theorem Compiled.backwardDenseFrom_eq_tape
    {α Δ : Type} [Storage α] [Add α] {Γ ss : List Shape}
    (graph : Tape.GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) (compiled : Compiled α Γ ss)
    (checked : compileChecked graph inputs data = .ok compiled)
    (seed : TensorPack α (Γ ++ ss)) :
    .ok (compiled.backwardDenseFrom seed) =
      Tape.backwardDenseFrom (GraphData.toTape graph inputs data).1
        seed.erase := by
  rw [backwardDenseFrom_eq graph inputs data compiled checked,
    GraphData.backwardDenseFrom_toTape]

end TorchLean.TypedGraph

namespace TorchLean
namespace TypedGraph.WithData

open TorchLean.Tensor
open Tape
open TorchLean (Idx)
open TensorPack (getIdx)

/-- A successful domain check returns the same output and dependency-aware VJP as pure
evaluation. This does not identify single-output traversal with the all-seeds dense reference. -/
theorem vjpChecked_eq_of_lowerChecked {α Δ : Type} [Storage α] [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape} (graph : WithData α Δ Γ τ)
    (inputs : TensorPack α Γ) (data : Δ) (seed : Tensor α τ)
    (execution : SavedGraph α Γ graph.nodeShapes × TensorContext α (Γ ++ graph.nodeShapes))
    (checked : SavedGraph.lowerChecked graph.data inputs data = .ok execution) :
    graph.vjpChecked inputs data seed = .ok (graph.vjp inputs data seed,
      graph.forward inputs data) := by
  have same := SavedGraph.lowerChecked_eq graph.data inputs data execution checked
  simp only [vjpChecked, TypedGraph.compileChecked, checked,
    Except.map, Bind.bind, Except.bind, Pure.pure, Except.pure]
  rw [same]
  simp only [vjp, forward, SavedGraph.ofGraph_context_eq, TensorContext.lookup_ofPack,
    TensorLookup.ofPack]

end TypedGraph.WithData
end TorchLean
