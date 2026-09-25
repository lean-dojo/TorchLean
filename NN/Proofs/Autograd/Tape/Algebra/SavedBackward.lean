/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.SavedGraph
public import NN.Proofs.Autograd.Tape.Algebra.Gradients

/-!
Reverse execution over saved node programs uses indexed gradients and deferred uniform updates.
Its result is proved equal to dense backpropagation without assuming associativity, zero identities,
or that a VJP maps the zero cotangent to zero.
-/

@[expose] public section

namespace Proofs.Autograd.Algebra

open Spec TorchLean

namespace PreparedNode

variable {α : Type} [Storage α] [Add α] [DecidableEq α]
variable {shapes : List Shape} {shape : Shape}

/-- Evaluate every VJP, selecting its certified compact program when available. -/
def accumulate (node : PreparedNode α shapes shape) (seed : Tensor α shape)
    (ctx : GradientContext α shapes) : GradientContext α shapes :=
  match node.compact? with
  | some implementation => ctx.addContributions (implementation.val seed)
  | none => ctx.addDense (node.vjp seed)

/-- Either execution path adds exactly the complete dense contribution. -/
theorem toPack_accumulate (node : PreparedNode α shapes shape) (seed : Tensor α shape)
    (ctx : GradientContext α shapes) :
    (node.accumulate seed ctx).toPack = TorchLean.TensorPack.add ctx.toPack (node.vjp seed) := by
  unfold accumulate
  split
  next implementation _ =>
    rw [GradientContext.toPack_addContributions, implementation.property]
  next _ => exact GradientContext.toPack_addDense _ _

end PreparedNode

namespace SavedGraph

variable {α : Type} [Storage α] [Add α] {Γ : List Shape}

/-- Reverse traversal finalizes one output gradient before continuing with the active prefix. -/
def backpropArray [DecidableEq α] {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : GradientContext α (Γ ++ ss)) : TensorContext α (Γ ++ ss) :=
  match saved with
  | .nil => TensorContext.ofPack seed.toPack
  | .snoc (ss := previousShapes) (τ := shape) previous prepared _ =>
      let parts := (seed.cast (List.append_assoc Γ previousShapes [shape]).symm).pop
      let accumulated := prepared.accumulate parts.2 parts.1
      let gradients := previous.backpropArray accumulated
      (gradients.push parts.2).cast (List.append_assoc Γ previousShapes [shape])

/-- Indexed reverse traversal returns every dense gradient in precisely the same order. -/
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
    (graph : GraphData α Δ Γ ss) (inputs : TorchLean.TensorPack α Γ) (data : Δ)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    ((ofGraph graph inputs data).1.backpropArray (GradientContext.ofPack seed)).toPack =
      graph.backpropAllCtx inputs data seed := by
  rw [toPack_backpropArray, GradientContext.toPack_ofPack, ofGraph_backpropDense]

/--
Use certified equality when the scalar storage provides it. Other carriers retain the exact
dense program, so the generic interface needs no additional scalar hypotheses.
-/
def backpropStored {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) : TensorContext α (Γ ++ ss) :=
  match Storage.decEq? (α := α) with
  | some equality =>
      letI := equality
      saved.backpropArray (GradientContext.ofPack seed)
  | none => TensorContext.ofPack (saved.backpropDense seed)

/-- Backend selection preserves the complete reverse program for every scalar storage. -/
theorem toPack_backpropStored {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    (saved.backpropStored seed).toPack = saved.backpropDense seed := by
  unfold backpropStored
  split
  · rw [toPack_backpropArray, GradientContext.toPack_ofPack]
  · exact TensorContext.toPack_ofPack _

/-- Stored reverse execution agrees with the original graph for every seed and scalar backend. -/
theorem ofGraph_backpropStored {Δ : Type} {ss : List Shape}
    (graph : GraphData α Δ Γ ss) (inputs : TorchLean.TensorPack α Γ) (data : Δ)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    ((ofGraph graph inputs data).1.backpropStored seed).toPack =
      graph.backpropAllCtx inputs data seed := by
  rw [toPack_backpropStored, ofGraph_backpropDense]

/-- Dense inputs-only reference, retaining the original local accumulation order. -/
def backpropInputsDense {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) : TorchLean.TensorPack α Γ :=
  match saved with
  | .nil => TorchLean.TensorPack.cast (List.append_nil Γ) seed
  | .snoc (ss := previousShapes) (τ := shape) previous prepared _ =>
      let parts := (TorchLean.TensorPack.cast
        (List.append_assoc Γ previousShapes [shape]).symm seed).unsnoc
      previous.backpropInputsDense (TorchLean.TensorPack.add parts.1 (prepared.vjp parts.2))

/-- Saving primals also preserves the original inputs-only VJP for every seed. -/
theorem ofGraph_backpropInputsDense {Δ : Type} {ss : List Shape}
    (graph : GraphData α Δ Γ ss) (inputs : TorchLean.TensorPack α Γ) (data : Δ)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    (ofGraph graph inputs data).1.backpropInputsDense seed =
      graph.backpropCtx inputs data seed := by
  induction graph with
  | nil => rfl
  | snoc previous node ih =>
      simp only [ofGraph, backpropInputsDense, GraphData.backpropCtx, ofGraph_context_eq,
        NodeData.prepare_vjp_ofPack, ih]

/-- Inputs-only reverse traversal releases finalized intermediate gradients immediately. -/
def backpropInputsArray [DecidableEq α] {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : GradientContext α (Γ ++ ss)) : TorchLean.TensorPack α Γ :=
  match saved with
  | .nil => TorchLean.TensorPack.cast (List.append_nil Γ) seed.toPack
  | .snoc (ss := previousShapes) (τ := shape) previous prepared _ =>
      let parts := (seed.cast (List.append_assoc Γ previousShapes [shape]).symm).pop
      previous.backpropInputsArray (prepared.accumulate parts.2 parts.1)

/-- Dropping finalized intermediates preserves the complete inputs-only result. -/
theorem backpropInputsArray_eq [DecidableEq α] {ss : List Shape}
    (saved : SavedGraph α Γ ss) (seed : GradientContext α (Γ ++ ss)) :
    saved.backpropInputsArray seed = saved.backpropInputsDense seed.toPack := by
  induction saved with
  | nil => rfl
  | @snoc previousShapes shape previous prepared value ih =>
      have parts := GradientContext.toPack_pop
        (seed.cast (List.append_assoc Γ previousShapes [shape]).symm)
      rw [GradientContext.toPack_cast] at parts
      simp only [backpropInputsArray, backpropInputsDense, ih, PreparedNode.toPack_accumulate]
      rw [← parts]

/-- Select the certified indexed implementation without strengthening the scalar interface. -/
def backpropInputsStored {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) : TorchLean.TensorPack α Γ :=
  match Storage.decEq? (α := α) with
  | some equality =>
      letI := equality
      saved.backpropInputsArray (GradientContext.ofPack seed)
  | none => saved.backpropInputsDense seed

/-- Both storage choices compute the same ordered inputs-only reverse program. -/
theorem backpropInputsStored_eq {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    saved.backpropInputsStored seed = saved.backpropInputsDense seed := by
  unfold backpropInputsStored
  split
  · rw [backpropInputsArray_eq, GradientContext.toPack_ofPack]
  · rfl

end SavedGraph

namespace GraphData

/-- Save one primal execution and compute its input pullback using indexed gradients. -/
def backpropCtxSaved {α : Type} [Storage α] {Δ : Type} {Γ : List Shape} [Add α]
    {ss : List Shape} (graph : GraphData α Δ Γ ss)
    (inputs : TorchLean.TensorPack α Γ) (data : Δ)
    (seed : TorchLean.TensorPack α (Γ ++ ss)) : TorchLean.TensorPack α Γ :=
  (SavedGraph.ofGraph graph inputs data).1.backpropInputsStored seed

/-- The ordinary pure VJP API compiles to the proved saved-program implementation. -/
@[csimp] theorem backpropCtx_eq_saved : @backpropCtx = @backpropCtxSaved := by
  funext α storage Δ Γ add ss graph inputs data seed
  simp only [backpropCtxSaved, SavedGraph.backpropInputsStored_eq,
    SavedGraph.ofGraph_backpropInputsDense]

end GraphData

end Proofs.Autograd.Algebra
