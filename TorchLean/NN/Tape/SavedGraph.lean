/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Lowering
public import TorchLean.NN.TypedGraph.Saved
import TorchLean.Tensor.Tactic.Except

/-!
# Correctness of saved graphs

The saved graphs of `TorchLean.NN.TypedGraph.Saved` hold the tensors and locally
prepared VJPs of one forward execution. We prove that their dense reference pass equals
`GraphData.backpropAllCtx` for every seed pack, including intermediate seeds and nodes whose
incoming cotangent is zero.

## Main declarations

- `SavedGraph.ofGraph_backpropDense`: the dense reference pass is full graph backpropagation.
- `SavedGraph.lowerChecked_eq`: a successful checked preparation produces the saved programs of
  `SavedGraph.ofGraph`.
-/

@[expose] public section

namespace TorchLean.Tape

namespace SavedGraph

variable {α : Type} [Storage α] {Γ : List Shape}

/-- Successful checked preparation produces the same saved programs as pure preparation. -/
theorem lowerChecked_eq {Δ : Type} {ss : List Shape} (graph : GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ)
    (result : SavedGraph α Γ ss × TensorContext α (Γ ++ ss))
    (checked : lowerChecked graph inputs data = .ok result) :
    result = ofGraph graph inputs data := by
  induction graph with
  | nil =>
      change Except.ok _ = Except.ok result at checked
      exact (Except.ok.inj checked).symm
  | snoc previous node ih =>
      simp only [lowerChecked] at checked
      except_cases previousResult : lowerChecked previous inputs data using checked with pair =>
        obtain ⟨saved, context⟩ := pair
        have same := ih (saved, context) previousResult
        simp only [previousResult, Bind.bind, Except.bind] at checked
        except_cases validation : (node.prepare context data).validate () using checked with done =>
          cases done
          simp only [validation, Pure.pure, Except.pure, Except.ok.injEq] at checked
          subst result
          simp only [ofGraph, ← same]

/-- Backpropagation through the saved VJPs gives `GraphData.backpropAllCtx` for every seed pack. -/
theorem ofGraph_backpropDense [Add α] {Δ : Type} {ss : List Shape}
    (graph : GraphData α Δ Γ ss) (inputs : TensorPack α Γ) (data : Δ)
    (seed : TensorPack α (Γ ++ ss)) :
    (ofGraph graph inputs data).1.backpropDense seed =
      graph.backpropAllCtx inputs data seed := by
  induction graph with
  | nil => rfl
  | snoc previous node ih =>
      simp only [ofGraph, backpropDense, GraphData.backpropAllCtx, ofGraph_context_eq,
        NodeData.prepare_vjp_ofPack, ih]

end SavedGraph

end TorchLean.Tape
