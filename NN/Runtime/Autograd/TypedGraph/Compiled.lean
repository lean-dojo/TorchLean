/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.Core
public import NN.Proofs.Autograd.Tape.Algebra.SavedBackward

/-!
Checked TypedGraph compilation saves local node programs and an indexed primal context. Reverse
execution uses the certified compact contributions when available and preserves the complete dense
gradient semantics. The explicit Tape adapter retains compatibility with consumers of raw tapes.
-/

@[expose] public section

namespace Proofs.Autograd.Algebra.SavedGraph

open Spec TorchLean

variable {α : Type} [Storage α] {Γ : List Shape}

/-- Adapt saved programs to the original runtime Tape representation. -/
def toTape {ss : List Shape} (saved : SavedGraph α Γ ss)
    (inputs : TorchLean.TensorPack α Γ) : Runtime.Autograd.Tape α :=
  match saved with
  | .nil => Graph.addLeaves Runtime.Autograd.Tape.empty inputs
  | .snoc previous prepared value =>
      ((previous.toTape inputs).addNode
        (Runtime.Autograd.TypedGraph.lowerPreparedNode prepared value)).1

/-- The compatibility adapter produces the exact original Tape, including its closures. -/
theorem ofGraph_toTape {Δ : Type} {ss : List Shape} (graph : GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) :
    (ofGraph graph inputs data).1.toTape inputs =
      (Graph.lowerGraphDataToTape graph inputs data).1 := by
  induction graph with
  | nil => rfl
  | snoc previous node ih =>
      simp only [ofGraph, toTape, Graph.lowerGraphDataToTape, ih, ofGraph_context_eq,
        NodeData.prepare_value_ofPack, Runtime.Autograd.TypedGraph.lowerPreparedNode_ofPack,
        Graph.lowerGraphDataToTape_ctx_eq_eval]

/-- Checked saved programs and indexed lowering have the same errors, tape and context. -/
theorem lowerChecked_toTape {Δ : Type} {ss : List Shape} (graph : GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) :
    (lowerChecked graph inputs data).map (fun result => (result.1.toTape inputs, result.2)) =
      Runtime.Autograd.TypedGraph.lowerToArrayChecked graph inputs data := by
  induction graph with
  | nil => rfl
  | snoc previous node ih =>
      simp only [lowerChecked, Runtime.Autograd.TypedGraph.lowerToArrayChecked, ← ih]
      cases outcome : lowerChecked previous inputs data with
      | error message => rfl
      | ok result =>
          obtain ⟨saved, context⟩ := result
          simp only [Except.map, Bind.bind, Except.bind]
          cases validation : (node.prepare context data).validate () with
          | error message => rfl
          | ok done => cases done; rfl

end Proofs.Autograd.Algebra.SavedGraph

namespace Runtime.Autograd.TypedGraph

open Spec TorchLean Proofs.Autograd.Algebra
open Proofs (Idx)

/-- One checked execution, retaining locally saved VJPs and indexed primal tensors. -/
structure Compiled (α : Type) [Storage α] (Γ ss : List Shape) where
  /-- Original input tensors, needed only by the raw Tape adapter. -/
  inputs : TorchLean.TensorPack α Γ
  /-- Programs for this execution in graph order. -/
  saved : SavedGraph α Γ ss
  /-- Input and intermediate values in graph order. -/
  context : TensorContext α (Γ ++ ss)

/-- Validate and evaluate each node once while recording its certified local reverse program. -/
def compileChecked {α Δ : Type} [Storage α] {Γ ss : List Shape}
    (graph : Proofs.Autograd.Algebra.GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) : Result (Compiled α Γ ss) :=
  (SavedGraph.lowerChecked graph inputs data).map
    (fun result => ⟨inputs, result.1, result.2⟩)

namespace Compiled

variable {α : Type} [Storage α] {Γ ss : List Shape}

/-- Materialize the original runtime Tape when required by another engine consumer. -/
def toTape (compiled : Compiled α Γ ss) : Tape α :=
  compiled.saved.toTape compiled.inputs

/-- Preserve the original checked lowering result type at an explicit compatibility boundary. -/
def asLegacy (compiled : Compiled α Γ ss) :
    Tape α × TorchLean.TensorPack α (Γ ++ ss) :=
  (compiled.toTape, compiled.context.toPack)

/-- Compute every gradient from a typed seed pack, retaining the dense floating-point order. -/
def backwardDenseFrom [Add α] (compiled : Compiled α Γ ss)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) : Array (Spec.SomeTensor α) :=
  (compiled.saved.backpropStored seed).values

/-- Seed any input or intermediate output and compute the complete gradient context. -/
def backwardDenseAllFrom [Add α] [Zero α] {shape : Shape}
    (compiled : Compiled α Γ ss) (output : Idx (Γ ++ ss) shape)
    (seed : Tensor α shape) : Array (Spec.SomeTensor α) :=
  compiled.backwardDenseFrom (TensorPack.single output seed)

end Compiled

/-- Compiled execution and legacy checked lowering agree on both failures and complete results. -/
theorem compileChecked_asLegacy {α Δ : Type} [Storage α] {Γ ss : List Shape}
    (graph : Proofs.Autograd.Algebra.GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) :
    (compileChecked graph inputs data).map Compiled.asLegacy =
      lowerToTapeChecked graph inputs data := by
  have same := SavedGraph.lowerChecked_toTape graph inputs data
  have mapped := congrArg (fun result =>
    result.map (fun pair => (pair.1, pair.2.toPack))) same
  simp only [lowerToArrayChecked_eq] at mapped
  cases saved : SavedGraph.lowerChecked graph inputs data <;>
    cases legacy : lowerToTapeChecked graph inputs data <;>
      simpa only [saved, legacy, compileChecked, Compiled.asLegacy, Compiled.toTape,
        Except.map, TensorContext.toPack_ofPack] using mapped

/-- Successful compilation has the original complete forward context. -/
theorem compileChecked_context {α Δ : Type} [Storage α] {Γ ss : List Shape}
    (graph : Proofs.Autograd.Algebra.GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) (compiled : Compiled α Γ ss)
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
theorem compileChecked_backwardDenseFrom {α Δ : Type} [Storage α] [Add α]
    {Γ ss : List Shape} (graph : Proofs.Autograd.Algebra.GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) (compiled : Compiled α Γ ss)
    (checked : compileChecked graph inputs data = .ok compiled)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    compiled.backwardDenseFrom seed =
      (graph.backpropAllCtx inputs data seed).toShapeErasedArray := by
  cases outcome : SavedGraph.lowerChecked graph inputs data with
  | error message => simp [compileChecked, outcome, Except.map] at checked
  | ok result =>
      simp only [compileChecked, outcome, Except.map, Except.ok.injEq] at checked
      subst compiled
      simp only [Compiled.backwardDenseFrom, SavedGraph.lowerChecked_eq _ _ _ _ outcome]
      have same := SavedGraph.ofGraph_backpropStored graph inputs data seed
      have arrays := congrArg TorchLean.TensorPack.toShapeErasedArray same
      have materialized (ctx : TensorContext α (Γ ++ ss)) :
          ctx.toPack.toShapeErasedArray = ctx.values :=
        congrArg TensorContext.values (TensorContext.ofPack_toPack ctx)
      rw [materialized] at arrays
      exact arrays

/-- The fast backward interface also agrees with the existing dense runtime Tape engine. -/
theorem compileChecked_backwardDenseFrom_eq_tape
    {α Δ : Type} [Storage α] [Add α] {Γ ss : List Shape}
    (graph : Proofs.Autograd.Algebra.GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) (compiled : Compiled α Γ ss)
    (checked : compileChecked graph inputs data = .ok compiled)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    .ok (compiled.backwardDenseFrom seed) =
      Tape.backwardDenseFrom (Graph.lowerGraphDataToTape graph inputs data).1
        seed.toShapeErasedArray := by
  rw [compileChecked_backwardDenseFrom graph inputs data compiled checked,
    Graph.backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx]

end Runtime.Autograd.TypedGraph
