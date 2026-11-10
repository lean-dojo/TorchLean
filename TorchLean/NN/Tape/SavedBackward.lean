/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.SavedGraph
public import TorchLean.NN.TypedGraph.Saved

/-!
# Correctness of the saved reverse passes

The reverse passes of `TorchLean.NN.TypedGraph.Saved` run over saved node programs
with indexed gradients and deferred uniform updates. We prove the all-seeds pass equal to dense
backpropagation without assuming associativity, zero identities, or that a VJP maps the zero
cotangent to zero. The single-output pass `SavedGraph.backpropFrom` omits unrelated nodes, and for
arbitrary user-defined VJPs the two passes need not agree.

## Main declarations

- `SavedGraph.toPack_backpropArray` and `SavedGraph.toPack_backpropStored`: the indexed passes
  return the gradients of `SavedGraph.backpropDense`.
- `SavedGraph.ofGraph_backpropStored`: the stored pass on the saved programs of a graph returns
  `GraphData.backpropAllCtx`.
-/

@[expose] public section

namespace TorchLean.Tape

namespace SavedGraph

variable {α : Type} [Storage α] [Add α] {Γ : List Shape}

/-- Indexed reverse traversal returns the gradients of the dense reference pass, position for
position. -/
theorem toPack_backpropArray [DecidableEq α] {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : GradientContext α (Γ ++ ss)) :
    (saved.backpropArray seed).toPack = saved.backpropDense seed.toPack := by
  induction saved with
  | nil => simp only [backpropArray, backpropDense, TensorContext.toPack_ofPack]
  | @snoc previousShapes shape previous prepared value ih =>
      have parts := GradientContext.toPack_pop
        (seed.cast (List.append_assoc Γ previousShapes [shape]).symm)
      rw [GradientContext.toPack_cast] at parts
      simp only [backpropArray, backpropDense, TensorContext.toPack_cast,
        TensorContext.toPack_push, ih, PreparedNode.toPack_accumulate]
      rw [← parts]

/-- Prepared array backpropagation preserves full graph semantics for arbitrary seed packs. -/
theorem ofGraph_backpropArray [DecidableEq α] {Δ : Type} {ss : List Shape}
    (graph : GraphData α Δ Γ ss) (inputs : TensorPack α Γ) (data : Δ)
    (seed : TensorPack α (Γ ++ ss)) :
    ((ofGraph graph inputs data).1.backpropArray (GradientContext.ofPack seed)).toPack =
      graph.backpropAllCtx inputs data seed := by
  rw [toPack_backpropArray, GradientContext.toPack_ofPack, ofGraph_backpropDense]

/-- Whichever path the storage selects, the result is the dense reference pass. -/
theorem toPack_backpropStored {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : TensorPack α (Γ ++ ss)) :
    (saved.backpropStored seed).toPack = saved.backpropDense seed := by
  unfold backpropStored
  split
  · rw [toPack_backpropArray, GradientContext.toPack_ofPack]
  · exact TensorContext.toPack_ofPack _

/-- Stored reverse execution on the graph saved by `ofGraph` equals `graph.backpropAllCtx` for
every seed and scalar backend. -/
theorem ofGraph_backpropStored {Δ : Type} {ss : List Shape}
    (graph : GraphData α Δ Γ ss) (inputs : TensorPack α Γ) (data : Δ)
    (seed : TensorPack α (Γ ++ ss)) :
    ((ofGraph graph inputs data).1.backpropStored seed).toPack =
      graph.backpropAllCtx inputs data seed := by
  rw [toPack_backpropStored, ofGraph_backpropDense]

end SavedGraph

end TorchLean.Tape
