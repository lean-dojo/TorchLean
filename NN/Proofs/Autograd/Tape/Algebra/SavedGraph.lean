/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Core
import NN.Tactic.Except

/-!
A saved graph contains the tensors and locally prepared VJPs from one forward execution. Its dense
reference pass is exactly the existing full graph backpropagation, including intermediate seeds and
nodes whose incoming cotangent is zero.
-/

@[expose] public section

namespace Proofs.Autograd.Algebra

open Spec TorchLean

/-- Locally saved node programs and their computed primal tensors in evaluation order. -/
inductive SavedGraph (α : Type) [Storage α] (Γ : List Shape) : List Shape → Type where
  | nil : SavedGraph α Γ []
  | snoc {ss : List Shape} {τ : Shape} (previous : SavedGraph α Γ ss)
      (prepared : PreparedNode α (Γ ++ ss) τ) (value : Tensor α τ) :
      SavedGraph α Γ (ss ++ [τ])

namespace SavedGraph

variable {α : Type} [Storage α] {Γ : List Shape}

/-- Prepare a graph with one array context, saving only the inputs requested by each node. -/
def ofGraph {Δ : Type} {ss : List Shape} (graph : GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) :
    SavedGraph α Γ ss × TensorContext α (Γ ++ ss) :=
  match graph with
  | .nil => (.nil, (TensorContext.ofPack inputs).cast (List.append_nil Γ).symm)
  | .snoc (ss := previousShapes) (τ := shape) previous node =>
      let (saved, context) := ofGraph previous inputs data
      let prepared := node.prepare context data
      let value := prepared.value ()
      (.snoc saved prepared value,
        (context.push value).cast (List.append_assoc Γ previousShapes [shape]))

/-- Checked preparation validates a node before evaluating or recording its primal. -/
def lowerChecked {Δ : Type} {ss : List Shape} (graph : GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) :
    Except String (SavedGraph α Γ ss × TensorContext α (Γ ++ ss)) :=
  match graph with
  | .nil => pure (.nil, (TensorContext.ofPack inputs).cast (List.append_nil Γ).symm)
  | .snoc (ss := previousShapes) (τ := shape) previous node => do
      let (saved, context) ← lowerChecked previous inputs data
      let prepared := node.prepare context data
      prepared.validate ()
      let value := prepared.value ()
      pure (.snoc saved prepared value,
        (context.push value).cast (List.append_assoc Γ previousShapes [shape]))

/-- Successful checked preparation produces the same saved programs as pure preparation. -/
theorem lowerChecked_eq {Δ : Type} {ss : List Shape} (graph : GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ)
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

/-- Prepared graph primals are the original graph evaluator's complete context. -/
theorem ofGraph_context_eq {Δ : Type} {ss : List Shape} (graph : GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ) :
    (ofGraph graph inputs data).2 = TensorContext.ofPack (graph.eval inputs data) := by
  induction graph with
  | nil => simp only [ofGraph, GraphData.eval, TensorContext.cast_ofPack]
  | snoc previous node ih =>
      simp only [ofGraph, GraphData.eval, ih, NodeData.prepare_value_ofPack,
        TensorContext.push_ofPack, TensorContext.cast_ofPack]

/-- The dense reference reverse pass preserves the original node and local addition order. -/
def backpropDense [Add α] {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) : TorchLean.TensorPack α (Γ ++ ss) :=
  match saved with
  | .nil => seed
  | .snoc (ss := previousShapes) (τ := shape) previous prepared _ =>
      let parts := (TorchLean.TensorPack.cast
        (List.append_assoc Γ previousShapes [shape]).symm seed).unsnoc
      let contribution := prepared.vjp parts.2
      let gradients := previous.backpropDense (TorchLean.TensorPack.add parts.1 contribution)
      TorchLean.TensorPack.cast (List.append_assoc Γ previousShapes [shape])
        (gradients.snoc parts.2)

/-- Saved VJPs compute precisely full graph backpropagation for every seed pack. -/
theorem ofGraph_backpropDense [Add α] {Δ : Type} {ss : List Shape}
    (graph : GraphData α Δ Γ ss) (inputs : TorchLean.TensorPack α Γ) (data : Δ)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    (ofGraph graph inputs data).1.backpropDense seed =
      graph.backpropAllCtx inputs data seed := by
  induction graph with
  | nil => rfl
  | snoc previous node ih =>
      simp only [ofGraph, backpropDense, GraphData.backpropAllCtx, ofGraph_context_eq,
        NodeData.prepare_vjp_ofPack, ih]

end SavedGraph

end Proofs.Autograd.Algebra
