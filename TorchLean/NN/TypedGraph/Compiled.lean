/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph
public import TorchLean.NN.TypedGraph.Saved

/-!
Checked compilation of a typed graph validates and evaluates each node once, saving its local
reverse program and an indexed context of primal values. The all-seeds reverse pass over these
programs computes `GraphData.backpropAllCtx` for every seed pack. A single-output pass
(`compiled.saved.backpropFrom`) visits the dependencies of that output and no other node when the
storage provides decidable equality and the nodes certify a compact support. An opaque VJP makes its
whole prefix reachable, and a carrier without storage equality runs the dense pass.
`Compiled.toTape` materializes the runtime tape, on which `Tape.backwardDenseFrom` computes the same
dense gradients. These equalities, and the agreement of a compiled execution's context with
`GraphData.eval`, are proved in `TorchLean.NN.Tape.Lowering.Compiled`.
-/

@[expose] public section

namespace TorchLean.Tape.SavedGraph

variable {α : Type} [Storage α] {Γ : List Shape}

/-- The runtime tape of saved programs: the input leaves, then one tape node per saved program. -/
def toTape {ss : List Shape} (saved : SavedGraph α Γ ss)
    (inputs : TensorPack α Γ) : Tape (SomeTensor α) :=
  match saved with
  | .nil => addLeaves Tape.empty inputs
  | .snoc previous prepared value =>
      ((previous.toTape inputs).addNode
        (TypedGraph.lowerPreparedNode prepared value)).1

/-- `lowerChecked` followed by `toTape` agrees with `lowerToArrayChecked` on errors, the tape and
the context. -/
theorem lowerChecked_toTape {Δ : Type} {ss : List Shape} (graph : GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) :
    (lowerChecked graph inputs data).map (fun result => (result.1.toTape inputs, result.2)) =
      TypedGraph.lowerToArrayChecked graph inputs data := by
  induction graph with
  | nil => rfl
  | snoc previous node ih =>
      simp only [lowerChecked, TypedGraph.lowerToArrayChecked, ← ih]
      cases outcome : lowerChecked previous inputs data with
      | error message => rfl
      | ok result =>
          obtain ⟨saved, context⟩ := result
          simp only [Except.map, Bind.bind, Except.bind]
          cases validation : (node.prepare context data).validate () with
          | error message => rfl
          | ok done => cases done; rfl

end TorchLean.Tape.SavedGraph

namespace TorchLean.TypedGraph

open Tape
open TorchLean (Idx)

/-- One checked execution, with its locally saved VJPs and indexed primal tensors. -/
structure Compiled (α : Type) [Storage α] (Γ ss : List Shape) where
  /-- The input tensors, which only `Compiled.toTape` reads. -/
  inputs : TensorPack α Γ
  /-- Programs for this execution in graph order. -/
  saved : SavedGraph α Γ ss
  /-- Input and intermediate values in graph order. -/
  context : TensorContext α (Γ ++ ss)

/-- Validate and evaluate each node once while recording its certified local reverse program. -/
def compileChecked {α Δ : Type} [Storage α] {Γ ss : List Shape}
    (graph : Tape.GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) : Result (Compiled α Γ ss) :=
  (SavedGraph.lowerChecked graph inputs data).map
    (fun result => ⟨inputs, result.1, result.2⟩)

namespace Compiled

variable {α : Type} [Storage α] {Γ ss : List Shape}

/-- Materialize the runtime tape for consumers of the raw tape engine. -/
def toTape (compiled : Compiled α Γ ss) : Tape (SomeTensor α) :=
  compiled.saved.toTape compiled.inputs

/-- Compute every gradient from a typed seed pack, keeping the dense floating-point order. -/
def backwardDenseFrom [Add α] (compiled : Compiled α Γ ss)
    (seed : TensorPack α (Γ ++ ss)) : Array (SomeTensor α) :=
  (compiled.saved.backpropStored seed).values

end Compiled

/-- Saved execution and tape lowering agree on failures, the tape and every forward value. -/
theorem compileChecked_eq {α Δ : Type} [Storage α] {Γ ss : List Shape}
    (graph : Tape.GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) :
    (compileChecked graph inputs data).map (fun compiled =>
      (compiled.toTape, compiled.context.toPack)) =
      lowerToTapeChecked graph inputs data := by
  have same := SavedGraph.lowerChecked_toTape graph inputs data
  have mapped := congrArg (fun result =>
    result.map (fun pair => (pair.1, pair.2.toPack))) same
  simp only [lowerToArrayChecked_eq] at mapped
  cases saved : SavedGraph.lowerChecked graph inputs data <;>
    cases lowered : lowerToTapeChecked graph inputs data <;>
      simpa only [saved, lowered, compileChecked, Compiled.toTape,
        Except.map, TensorContext.toPack_ofPack] using mapped

end TorchLean.TypedGraph
