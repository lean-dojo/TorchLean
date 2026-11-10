/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.Data
public import TorchLean.NN.TypedGraph.Gradients

/-!
# Saved graphs and their reverse passes

A `SavedGraph` holds the tensors and locally prepared VJPs of one forward execution, built by
`SavedGraph.ofGraph` or, validating each node first, by `SavedGraph.lowerChecked`. Its reverse
passes accumulate into a `GradientContext`. `SavedGraph.backpropStored` takes a seed for every
context entry, and `SavedGraph.backpropFrom` seeds one entry and visits its certified dependencies
when storage equality is available. For arbitrary VJPs the two passes need not agree. Their theorems
are in `TorchLean.NN.Tape.SavedGraph` and
`TorchLean.NN.Tape.SavedBackward`.

## Main declarations

- `TorchLean.Tape.SavedGraph`, with the dense reference pass `SavedGraph.backpropDense` and
  the indexed passes `SavedGraph.backpropStored`, `SavedGraph.backpropFrom` and
  `SavedGraph.backpropInputsStored`.
- `TorchLean.Tape.GraphData.backpropCtxSaved`, the saved-program implementation of
  `GraphData.backpropCtx` (`csimp` lemma `GraphData.backpropCtx_eq_saved`).
-/

@[expose] public section

namespace TorchLean.Tape

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
    (inputs : TensorPack α Γ) (data : Δ) :
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
    (inputs : TensorPack α Γ) (data : Δ) :
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

/-- Prepared graph primals are the original graph evaluator's complete context. -/
theorem ofGraph_context_eq {Δ : Type} {ss : List Shape} (graph : GraphData α Δ Γ ss)
    (inputs : TensorPack α Γ) (data : Δ) :
    (ofGraph graph inputs data).2 = TensorContext.ofPack (graph.eval inputs data) := by
  induction graph with
  | nil => simp only [ofGraph, GraphData.eval, TensorContext.cast_ofPack]
  | snoc previous node ih =>
      simp only [ofGraph, GraphData.eval, ih, NodeData.prepare_value_ofPack,
        TensorContext.push_ofPack, TensorContext.cast_ofPack]

/-- The dense reference reverse pass preserves the original node and local addition order. -/
def backpropDense [Add α] {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : TensorPack α (Γ ++ ss)) : TensorPack α (Γ ++ ss) :=
  match saved with
  | .nil => seed
  | .snoc (ss := previousShapes) (τ := shape) previous prepared _ =>
      let parts := (Pack.cast
        (List.append_assoc Γ previousShapes [shape]).symm seed).unsnoc
      let contribution := prepared.vjp parts.2
      let gradients := previous.backpropDense (TensorPack.add parts.1 contribution)
      Pack.cast (List.append_assoc Γ previousShapes [shape])
        (gradients.snoc parts.2)

end SavedGraph

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
    (node.accumulate seed ctx).toPack = TensorPack.add ctx.toPack (node.vjp seed) := by
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

/--
Use certified equality when the scalar storage provides it. Other carriers retain the exact
dense program, so the generic interface needs no additional scalar hypotheses.
-/
def backpropStored {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : TensorPack α (Γ ++ ss)) : TensorContext α (Γ ++ ss) :=
  match Storage.decEq? (α := α) with
  | some equality =>
      letI := equality
      saved.backpropArray (GradientContext.ofPack seed)
  | none => TensorContext.ofPack (saved.backpropDense seed)

/-- Visit only dependencies of the seeded output when a node has a certified compact support.
An opaque dense VJP conservatively makes every preceding position reachable. Reachability follows
the prepared VJP's support: a reached node still runs when its cotangent happens to be zero.
-/
def backpropReachableArray [DecidableEq α] [Zero α] {ss : List Shape}
    (saved : SavedGraph α Γ ss) (seed : GradientContext α (Γ ++ ss))
    (reached : Array Bool) : TensorContext α (Γ ++ ss) :=
  match saved with
  | .nil => TensorContext.ofPack seed.toPack
  | .snoc (ss := previousShapes) (τ := shape) previous prepared _ =>
      let parts := (seed.cast (List.append_assoc Γ previousShapes [shape]).symm).pop
      let prefixLength := (Γ ++ previousShapes).length
      let (accumulated, reached) := if reached[prefixLength]? == some true then
          match prepared.compact? with
          | some implementation =>
              let contribution := implementation.val parts.2
              let reached := if contribution.uniform = 0 then
                  contribution.support.foldl
                    (fun reached entry => reached.set! entry.2.i.val true) reached
                else Array.replicate prefixLength true
              (parts.1.addContributions contribution, reached)
          | none =>
              (parts.1.addDense (prepared.vjp parts.2), Array.replicate prefixLength true)
        else (parts.1, reached)
      let gradients := backpropReachableArray previous accumulated reached
      (gradients.push parts.2).cast (List.append_assoc Γ previousShapes [shape])

/-- Backpropagate from one output using certified support when storage equality is available.

Storage equality compares carrier encodings, not the scalar's numerical `BEq`. IEEE signed zeros
and NaNs therefore do not invalidate the support check. Carriers without decidable storage
equality retain the full dense program. `backpropStored` remains the explicit all-seeds reference.
Opaque VJPs conservatively visit the full prefix. Custom compact VJPs supply their own support.
This traversal makes no separate claim that it captures the syntactic dependencies of an arbitrary
user-defined forward function.
-/
def backpropFrom [Zero α] {ss : List Shape} {shape : Shape}
    (saved : SavedGraph α Γ ss) (output : Idx (Γ ++ ss) shape) (seed : Tensor α shape) :
    TensorContext α (Γ ++ ss) :=
  let gradients := TensorPack.single output seed
  match Storage.decEq? (α := α) with
  | some equality =>
      letI := equality
      let reached := (Array.replicate (Γ ++ ss).length false).set! output.i.val true
      backpropReachableArray saved (GradientContext.ofPack gradients) reached
  | none => saved.backpropStored gradients

/-- Dense inputs-only reference, accumulating in the order of `GraphData.backpropCtx`. -/
def backpropInputsDense {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : TensorPack α (Γ ++ ss)) : TensorPack α Γ :=
  match saved with
  | .nil => Pack.cast (List.append_nil Γ) seed
  | .snoc (ss := previousShapes) (τ := shape) previous prepared _ =>
      let parts := (Pack.cast
        (List.append_assoc Γ previousShapes [shape]).symm seed).unsnoc
      previous.backpropInputsDense (TensorPack.add parts.1 (prepared.vjp parts.2))

/-- Saving primals also preserves the inputs-only VJP `graph.backpropCtx` for every seed. -/
theorem ofGraph_backpropInputsDense {Δ : Type} {ss : List Shape}
    (graph : GraphData α Δ Γ ss) (inputs : TensorPack α Γ) (data : Δ)
    (seed : TensorPack α (Γ ++ ss)) :
    (ofGraph graph inputs data).1.backpropInputsDense seed =
      graph.backpropCtx inputs data seed := by
  induction graph with
  | nil => rfl
  | snoc previous node ih =>
      simp only [ofGraph, backpropInputsDense, GraphData.backpropCtx, ofGraph_context_eq,
        NodeData.prepare_vjp_ofPack, ih]

/-- Inputs-only reverse traversal releases finalized intermediate gradients immediately. -/
def backpropInputsArray [DecidableEq α] {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : GradientContext α (Γ ++ ss)) : TensorPack α Γ :=
  match saved with
  | .nil => Pack.cast (List.append_nil Γ) seed.toPack
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
    (seed : TensorPack α (Γ ++ ss)) : TensorPack α Γ :=
  match Storage.decEq? (α := α) with
  | some equality =>
      letI := equality
      saved.backpropInputsArray (GradientContext.ofPack seed)
  | none => saved.backpropInputsDense seed

/-- Both storage choices compute the same ordered inputs-only reverse program. -/
theorem backpropInputsStored_eq {ss : List Shape} (saved : SavedGraph α Γ ss)
    (seed : TensorPack α (Γ ++ ss)) :
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
    (inputs : TensorPack α Γ) (data : Δ)
    (seed : TensorPack α (Γ ++ ss)) : TensorPack α Γ :=
  (SavedGraph.ofGraph graph inputs data).1.backpropInputsStored seed

/-- The ordinary pure VJP API compiles to the proved saved-program implementation. -/
@[csimp] theorem backpropCtx_eq_saved : @backpropCtx = @backpropCtxSaved := by
  funext α storage Δ Γ add ss graph inputs data seed
  simp only [backpropCtxSaved, SavedGraph.backpropInputsStored_eq,
    SavedGraph.ofGraph_backpropInputsDense]

end GraphData

end TorchLean.Tape
