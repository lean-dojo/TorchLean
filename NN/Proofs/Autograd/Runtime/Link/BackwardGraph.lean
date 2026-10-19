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
# Dense Runtime Backward Pass Link

This file proves that the executable dense backward loop on a lowered tape agrees with the
proof-level full backpropagation `backpropAllCtx`. It is the main bridge between the runtime tape
engine and the algebraic reverse-mode model.

The theorem is stated for `GraphData`, which stores executable forward and backward functions and
needs only `Storage` and `Add` on the scalar. The proof is an induction on the graph: the `nil`
case is `backwardDenseFrom_addLeaves_empty` (the loop is the identity on a leaf-only tape) and the
`snoc` case is `backwardDenseFrom_addNode_lowerNode`, instantiated with the forward-pass facts about
the lowered prefix from `Link.Core` and `Link.Invariants`. A proved `Graph` lowers through
`Graph.toData`, so its theorem is the `GraphData` one applied to `g.toData`.
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
**Main runtime/link theorem**: running the runtime dense backward loop on a tape produced by
`lowerGraphDataToTape` matches the proved full backpropagation `GraphData.backpropAllCtx`.

This is the formal statement that the executable engine implements the same reverse-mode
accumulation semantics as the proved tape model.
-/
theorem backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx {α : Type} {Δ : Type}
    [TorchLean.Storage α] [Add α]
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

/-- The same statement for a proved graph, which lowers through `g.toData`. -/
theorem backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx {α : Type} {Δ : Type}
    [TorchLean.Storage α] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d0 : Δ) (seed : TorchLean.TensorPack α (Γ ++ ss)) :
    Runtime.Autograd.Tape.backwardDenseFrom
      (t := (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1)
        (grads0 := TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss) seed) =
      .ok (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss)
        (backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0 seed)) :=
  backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx g.toData x d0 seed

end Graph

end Algebra
end Autograd
end Proofs
