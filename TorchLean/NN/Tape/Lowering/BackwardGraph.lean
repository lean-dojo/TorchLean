/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Lowering.BackwardLeaves
public import TorchLean.NN.Tape.Lowering.BackwardSnoc
public import TorchLean.NN.Tape.Lowering.Invariants

/-!
# Dense Runtime Backward Pass Link

On a lowered tape, the executable dense backward loop agrees with the proof-level full
backpropagation `backpropAllCtx`. That theorem is the main bridge between the runtime tape engine
and the algebraic reverse-mode model.

The theorem is stated for `GraphData`, which stores executable forward and backward functions and
needs only `Storage` and `Add` on the scalar. The proof is an induction on the graph: the `nil`
case is `backwardDenseFrom_addLeaves_empty` (the loop is the identity on a leaf-only tape) and the
`snoc` case is `NodeData.backwardDenseFrom_toNode`, instantiated with the forward-pass facts about
the lowered prefix from `TorchLean.NN.Tape.Lowering` and `TorchLean.NN.Tape.Lowering.Invariants`.
For a proof-carrying graph `g` the theorem applies to its executable graph `g.toData`.
-/

@[expose] public section

namespace TorchLean
namespace Tape

open TorchLean.Tensor

namespace GraphData

/--
**Main runtime/link theorem**: running the runtime dense backward loop on a tape produced by
`toTape` matches the proved full backpropagation `GraphData.backpropAllCtx`.

For every seed, the executable engine accumulates cotangents with the reverse-mode semantics of
the proved tape model.
-/
theorem backwardDenseFrom_toTape {α : Type} {Δ : Type}
    [Storage α] [Add α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TensorPack α Γ)
    (d0 : Δ) (seed : TensorPack α (Γ ++ ss)) :
    Tape.backwardDenseFrom
      (t := (toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1)
        (grads0 := TensorPack.erase (α := α) (ss := Γ ++ ss) seed) =
      .ok
        (TensorPack.erase (α := α) (ss := Γ ++ ss)
          (GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss :=
            ss) g x d0 seed)) := by
  induction g with
  | nil =>
      simpa [toTape, GraphData.backpropAllCtx] using
        backwardDenseFrom_addLeaves_empty (α := α) x
          (Pack.cast (h := List.append_nil Γ) seed)
  | snoc g node ih =>
      rename_i ssPrev τ
      rcases hprev : toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 with
        ⟨tPrev, ctxPrev⟩
      -- Forward-pass facts about the lowered prefix.
      have hctx : ctxPrev = GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 := by
        simpa [hprev] using
          toTape_ctx_eq_eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      have hsize : tPrev.nodes.size = (Γ ++ ssPrev).length := by
        simpa [hprev] using
          toTape_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      have hreq : ∀ i (hi : i < tPrev.nodes.size), (tPrev.nodes[i]'hi).requiresGrad = true := by
        simpa [hprev] using
          requiresGrad_toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      have hvals :
          tPrev.nodes.map (fun n => n.value) =
            TensorPack.erase (α := α) (ss := Γ ++ ssPrev)
              ctxPrev := by
        simpa [hprev] using
          toTape_values_eq (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      have hpids : BackwardPidsLt tPrev := by
        have h :=
          backwardPidsLt_toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
        rw [hprev] at h
        exact h
      have hbp : ∀ s : TensorPack α (Γ ++ ssPrev),
          Tape.backwardDenseFrom (t := tPrev)
              (TensorPack.erase (α := α) s) =
            .ok (TensorPack.erase (α := α)
              (GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 s)) := by
        intro s
        simpa [hprev] using ih s
      -- The lowered `snoc` graph is the prefix tape plus one lowered node.
      have hTape :
          (toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ])
            (.snoc (ss := ssPrev) (τ := τ) g node) x d0).1 =
            (tPrev.addNode (NodeData.toNode (α := α) (some "typed-graph") node ctxPrev d0)).1 := by
        simp [toTape, hprev]
      -- Split the seed into its prefix and its last entry, then apply the generic `snoc` step.
      have hseed :
          TensorPack.erase (α := α) (ss := Γ ++ (ssPrev ++ [τ]))
              seed =
            TensorPack.erase (α := α) (ss := (Γ ++ ssPrev) ++ [τ])
              (Pack.snoc (ss := Γ ++ ssPrev) (τ := τ)
                (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ)
                  (Pack.cast
                    (h := (List.append_assoc Γ ssPrev [τ]).symm) seed)).1
                (Pack.unsnoc (ss := Γ ++ ssPrev) (τ := τ)
                  (Pack.cast
                    (h := (List.append_assoc Γ ssPrev [τ]).symm) seed)).2) := by
        rw [Pack.snoc_unsnoc,
          TensorPack.erase_cast]
      rw [hTape, hseed, hctx]
      simp only [GraphData.backpropAllCtx, TensorPack.erase_cast]
      subst hctx
      exact NodeData.backwardDenseFrom_toNode tPrev _ node _ d0
        (GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0) hsize hreq hvals
        hpids hbp _ _

end GraphData

end Tape
end TorchLean
