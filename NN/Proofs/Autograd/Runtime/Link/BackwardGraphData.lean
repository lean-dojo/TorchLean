/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.BackwardLeaves
public import NN.Proofs.Autograd.Runtime.Link.BackwardSnoc
public import NN.Proofs.Autograd.Runtime.Link.Invariants

/-!
# GraphData Backward Pass Link

This file states the dense-backward correctness theorem for `GraphData`, where the
forward/backward closures carry an additional payload such as parameters or configuration data.

The proof has the same two cases as `backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx` and
uses the same shared lemmas, `backwardDenseFrom_addLeaves_empty` and
`backwardDenseFrom_addNode_lowerNode`; only the forward-pass facts about the lowered prefix come
from the `GraphData` lemmas of `Link.Core` and `Link.Invariants`.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Algebra

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Graph

open Runtime
open Runtime.Autograd

/--
Variant of `backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx` for the `GraphData` interface.

This is useful when a graph carries extra payload `Δ` (e.g. parameters/config) through forward and
backward closures.
-/
theorem backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx {α : Type} {Δ : Type}
  [TorchLean.Storage α] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d0 : Δ) (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    Runtime.Autograd.Tape.backwardDenseFrom
      (t := (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1)
        (grads0 := TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss) seed) =
      .ok
        (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss)
          (Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss :=
            ss) g x d0 seed)) := by
  induction g with
  | nil =>
      simpa [lowerGraphDataToTape, GraphData.backpropAllCtx] using
        backwardDenseFrom_addLeaves_empty (α := α) x
          (TorchLean.TensorPack.cast (α := α) (h := List.append_nil Γ) seed)
  | snoc g node ih =>
      rename_i ssPrev τ
      rcases hprev : lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 with
        ⟨tPrev, ctxPrev⟩
      -- Forward-pass facts about the lowered prefix.
      have hctx : ctxPrev = GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 := by
        simpa [hprev] using
          lowerGraphDataToTape_ctx_eq_eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      have hsize : tPrev.nodes.size = (Γ ++ ssPrev).length := by
        simpa [hprev] using
          lowerGraphDataToTape_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      have hreq : ∀ i (hi : i < tPrev.nodes.size), (tPrev.nodes[i]'hi).requiresGrad = true := by
        simpa [hprev] using
          lowerGraphDataToTape_requires_grad_true (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      have hvals :
          tPrev.nodes.map (fun n => n.value) =
            TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev)
              ctxPrev := by
        simpa [hprev] using
          lowerGraphDataToTape_values_eq (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      have hpids : BackwardPidsLt tPrev := by
        have h :=
          lowerGraphDataToTape_backward_pids_lt_id (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
        rw [hprev] at h
        exact h
      have hbp : ∀ s : TorchLean.TensorPack α (Γ ++ ssPrev),
          Runtime.Autograd.Tape.backwardDenseFrom (t := tPrev)
              (TorchLean.TensorPack.toShapeErasedArray (α := α) s) =
            .ok (TorchLean.TensorPack.toShapeErasedArray (α := α)
              (GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 s)) := by
        intro s
        simpa [hprev] using ih s
      -- The lowered `snoc` graph is the prefix tape plus one lowered node.
      have hTape :
          (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ])
            (.snoc (ss := ssPrev) (τ := τ) g node) x d0).1 =
            (tPrev.addNode (lowerNode (α := α) (some "typed-graph") node ctxPrev d0)).1 := by
        simp [lowerGraphDataToTape, hprev]
      -- Split the seed into its prefix and its last entry, then apply the generic `snoc` step.
      have hseed :
          TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ (ssPrev ++ [τ]))
              seed =
            TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := (Γ ++ ssPrev) ++ [τ])
              (TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
                (TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
                  (TorchLean.TensorPack.cast (α := α)
                    (h := (List.append_assoc Γ ssPrev [τ]).symm) seed)).1
                (TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
                  (TorchLean.TensorPack.cast (α := α)
                    (h := (List.append_assoc Γ ssPrev [τ]).symm) seed)).2) := by
        rw [TorchLean.TensorPack.snoc_unsnoc,
          TorchLean.TensorPack.toShapeErasedArray_cast]
      rw [hTape, hseed, hctx]
      simp only [GraphData.backpropAllCtx, TorchLean.TensorPack.toShapeErasedArray_cast]
      subst hctx
      exact backwardDenseFrom_addNode_lowerNode tPrev _ node _ d0
        (GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0) hsize hreq hvals
        hpids hbp _ _

end Graph

end Algebra
end Autograd
end Proofs
