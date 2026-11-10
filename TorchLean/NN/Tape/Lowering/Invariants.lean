/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Lowering

/-!
# Tape invariants of lowered graphs

Lowering a graph marks every runtime node as a gradient accumulation slot and emits backward
contributions only to earlier node ids (`BackwardPidsLt`). The dense reverse sweep proofs in
`BackwardGraph` consume both invariants.
-/

@[expose] public section

namespace TorchLean
namespace Tape

open TorchLean.Tensor

/-- `addLeaves` keeps every tape node eligible for gradient accumulation. -/
theorem all_requiresGrad_addLeaves {α : Type} [Storage α] (t : Tape (SomeTensor α))
    (ht : t.nodes.all (fun n => n.requiresGrad) = true)
    {Γ : List Shape} (xs : TensorPack α Γ) :
    (addLeaves (α := α) (t := t) (Γ := Γ) xs).nodes.all (fun n => n.requiresGrad) = true := by
  rw [nodes_addLeaves, Array.all_append, ht, Bool.true_and, Array.all_map]
  exact Array.all_eq_true.mpr (fun _ _ => rfl)

/--
All nodes produced by `toTape` have `requiresGrad = true`.

Lowering marks every node, input leaves included, as eligible for gradient accumulation, so on a
lowered tape the dense reverse loop runs every node's VJP and drops no contribution.
-/
theorem GraphData.all_requiresGrad_toTape {α : Type} {Δ : Type}
    [Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TensorPack α Γ)
    (d : Δ) :
    ((toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.all (fun n =>
      n.requiresGrad)) = true := by
  induction g with
  | nil =>
      have h0 :
          (Tape.empty (V := SomeTensor α)).nodes.all
              (fun n => n.requiresGrad) = true := by
        simp [Tape.empty]
      simpa [toTape] using
        all_requiresGrad_addLeaves (Tape.empty (V := SomeTensor α)) h0 x
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [toTape, Tape.addNode, ih]

/-- Pointwise form of `all_requiresGrad_toTape`, convenient for array
indexing: every node of the lowered tape has `requiresGrad = true`. -/
theorem GraphData.requiresGrad_toTape {α : Type} {Δ : Type}
    [Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TensorPack α Γ)
    (d : Δ) :
    let t := (toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1
    ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requiresGrad = true := by
  intro t i hi
  have hall :
      t.nodes.all (fun n => n.requiresGrad) = true := by
    simpa [t] using
      all_requiresGrad_toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d
  have := (Array.all_eq_true).1 hall i hi
  simpa using this

/--
Backward closure safety for `toTape`: every contribution that a node of the lowered
tape emits targets a strictly smaller node id (`Tape.BackwardPidsLt`).

Leaves emit nothing, and a lowered node numbers its contributions from `0` across its input
context, which consists of the nodes before it. A reverse sweep in decreasing id order therefore
never sends a contribution to a node it has already visited.
-/
theorem GraphData.backwardPidsLt_toTape {α : Type} {Δ : Type}
    [Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TensorPack α Γ)
    (d0 : Δ) :
    BackwardPidsLt (toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1 := by
  unfold BackwardPidsLt
  induction g with
  | nil =>
      intro id n hn d contribs hback pid pg hmem
      -- `toTape nil` produces leaves with empty backward arrays.
      have hn' :
          ((TensorPack.erase (α := α) (ss := Γ) x).map Node.leaf)[id]? = some n := by
        simpa [toTape, Tape.getNode?, nodes_addLeaves,
          Tape.empty] using hn
      cases hx : (TensorPack.erase (α := α) (ss := Γ) x)[id]? with
      | none =>
          simp [Array.getElem?_map, hx] at hn'
      | some v =>
          have hnEq : n = Node.leaf v := by
            symm
            simpa [Array.getElem?_map, hx] using hn'
          subst hnEq
          -- A leaf contributes no parent cotangents.
          have hcontribs : contribs = #[] := (Except.ok.inj hback).symm
          subst hcontribs
          simp at hmem
  | snoc g node ih =>
      rename_i ssPrev τ
      intro id n hn d contribs hback pid pg hmem
      let prev := toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      let tPrev := prev.1
      let ctxPrev := prev.2
      let runtimeNode : Node (SomeTensor α) :=
        NodeData.toNode (some "typed-graph") node ctxPrev d0
      have hnNodes :
          (tPrev.nodes.push runtimeNode)[id]? = some n := by
        simpa [toTape, prev, tPrev, ctxPrev, runtimeNode,
          Tape.getNode?, Tape.addNode] using hn
      by_cases hlast : id = tPrev.nodes.size
      · subst hlast
        have hnEq : n = runtimeNode := by
          symm
          simpa [Array.getElem?_push] using hnNodes
        subst hnEq
        have hd : d.shape = τ := by
          by_contra hne
          have : runtimeNode.backward d = .error "autograd: upstream gradient shape mismatch" :=
            NodeData.toNode_backward_of_ne _ _ _ _ d hne
          simp [this] at hback
        have hret : runtimeNode.backward d =
            .ok (TensorPack.eraseIndexed (α := α) (ss := Γ ++ ssPrev)
              (node.vjp ctxPrev d0 (d.cast hd)) 0) :=
          NodeData.toNode_backward_of_shape _ _ _ _ d hd
        have hcontribs :
            contribs =
              TensorPack.eraseIndexed (α := α) (ss := Γ ++ ssPrev)
                (node.vjp ctxPrev d0 (d.cast hd)) 0 :=
          (Except.ok.inj (hret.symm.trans hback)).symm
        subst hcontribs
        have hpidlt :=
          TensorPack.mem_eraseIndexed_lt
            (α := α) (ss := Γ ++ ssPrev)
            (node.vjp ctxPrev d0 (d.cast hd)) 0 (pid := pid) (pg := pg) hmem
        -- `hpidlt` bounds `pid` by `0 + (Γ ++ ssPrev).length`, the size of the prefix tape.
        have htPrev :
            tPrev.nodes.size = (Γ ++ ssPrev).length := by
          simpa [prev] using
            toTape_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
        simpa [htPrev] using hpidlt
      · have hnPrev : Tape.getNode? (t := tPrev) id = some n := by
          have : tPrev.nodes[id]? = some n := by
            simpa [Array.getElem?_push, hlast] using hnNodes
          simpa [Tape.getNode?, tPrev] using this
        exact ih id n (by simpa [prev, tPrev] using hnPrev) d contribs hback hmem

end Tape
end TorchLean
