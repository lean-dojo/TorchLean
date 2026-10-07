/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Core

/-!
# Tape invariants of lowered graphs

Lowering a graph marks every runtime node as a gradient accumulation slot and only emits backward
contributions to earlier node ids (`BackwardPidsLt`). The dense reverse sweep proofs in
`BackwardGraph` and `BackwardGraphData` consume both invariants.
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

/-- `addLeaves` keeps every tape node eligible for gradient accumulation. -/
theorem all_requiresGrad_addLeaves {α : Type} [TorchLean.Storage α] (t : Tape α)
    (ht : t.nodes.all (fun n => n.requiresGrad) = true)
    {Γ : List Shape} (xs : TorchLean.TensorPack α Γ) :
    (addLeaves (α := α) (t := t) (Γ := Γ) xs).nodes.all (fun n => n.requiresGrad) = true := by
  rw [nodes_addLeaves, Array.all_append, ht, Bool.true_and, Array.all_map]
  exact Array.all_eq_true.mpr (fun _ _ => rfl)

/--
All nodes produced by `lowerGraphDataToTape` have `requiresGrad = true`.

This is a simplifying invariant: the lowered tape is meant for correctness proofs, so we mark
every node as eligible for gradient accumulation (including leaves for inputs).
-/
theorem lowerGraphDataToTape_all_requires_grad_true {α : Type} {Δ : Type}
    [TorchLean.Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d : Δ) :
    ((lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.all (fun n =>
      n.requiresGrad)) = true := by
  induction g with
  | nil =>
      have h0 :
          (Runtime.Autograd.Tape.empty (α := α)).nodes.all (fun n => n.requiresGrad) = true := by
        simp [Runtime.Autograd.Tape.empty]
      simpa [lowerGraphDataToTape] using
        all_requiresGrad_addLeaves (Runtime.Autograd.Tape.empty (α := α)) h0 x
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [lowerGraphDataToTape, Runtime.Autograd.Tape.addNode, ih]

/-- Pointwise form of `lowerGraphDataToTape_all_requires_grad_true`, convenient for array
indexing: every node of the lowered tape has `requiresGrad = true`. -/
theorem lowerGraphDataToTape_requires_grad_true {α : Type} {Δ : Type}
    [TorchLean.Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d : Δ) :
    let t := (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1
    ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requiresGrad = true := by
  intro t i hi
  have hall :
      t.nodes.all (fun n => n.requiresGrad) = true := by
    simpa [t] using
      lowerGraphDataToTape_all_requires_grad_true (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d
  have := (Array.all_eq_true).1 hall i hi
  simpa using this

/--
Backward closure safety for `lowerGraphDataToTape`: parent ids produced by any node are strictly
smaller than the node id.

This is the “edges point backwards” invariant required by the runtime reverse loop: when processing
node `id`, every contribution targets an earlier node (`pid < id`), so accumulation is well-founded.
-/
theorem lowerGraphDataToTape_backward_pids_lt_id {α : Type} {Δ : Type}
    [TorchLean.Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d0 : Δ) :
    BackwardPidsLt (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1 := by
  unfold BackwardPidsLt
  induction g with
  | nil =>
      intro id n hn d contribs hback pid pg hmem
      -- `lowerGraphDataToTape nil` produces leaves with empty backward arrays.
      have hn' :
          ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x).map
            (leafNodeOfSomeTensor (α := α)))[id]? = some n := by
        simpa [lowerGraphDataToTape, Runtime.Autograd.Tape.getNode?, nodes_addLeaves,
          Runtime.Autograd.Tape.empty] using hn
      cases hx : (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x)[id]? with
      | none =>
          simp [Array.getElem?_map, hx] at hn'
      | some v =>
          have hnEq : n = leafNodeOfSomeTensor (α := α) v := by
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
      let prev := lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      let tPrev := prev.1
      let ctxPrev := prev.2
      let runtimeNode : Runtime.Autograd.Node α :=
        lowerNode (some "typed-graph") node ctxPrev d0
      have hnNodes :
          (tPrev.nodes.push runtimeNode)[id]? = some n := by
        simpa [lowerGraphDataToTape, prev, tPrev, ctxPrev, runtimeNode,
          Runtime.Autograd.Tape.getNode?, Runtime.Autograd.Tape.addNode] using hn
      by_cases hlast : id = tPrev.nodes.size
      · subst hlast
        have hnEq : n = runtimeNode := by
          symm
          simpa [Array.getElem?_push] using hnNodes
        subst hnEq
        have hd : d.shape = τ := by
          by_contra hne
          have : runtimeNode.backward d = .error "autograd: upstream gradient shape mismatch" :=
            lowerNode_backward_of_ne _ _ _ _ d hne
          simp [this] at hback
        have hret : runtimeNode.backward d =
            .ok (TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := Γ ++ ssPrev)
              (node.vjp ctxPrev d0 (d.cast hd)) 0) :=
          lowerNode_backward_of_shape _ _ _ _ d hd
        have hcontribs :
            contribs =
              TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := Γ ++ ssPrev)
                (node.vjp ctxPrev d0 (d.cast hd)) 0 :=
          (Except.ok.inj (hret.symm.trans hback)).symm
        subst hcontribs
        have hpidlt :=
          TorchLean.TensorPack.mem_toIndexedShapeErasedArray_lt
            (α := α) (ss := Γ ++ ssPrev)
            (node.vjp ctxPrev d0 (d.cast hd)) 0 (pid := pid) (pg := pg) hmem
        -- `0 + (Γ ++ ssPrev).length = tPrev.nodes.size`
        have htPrev :
            tPrev.nodes.size = (Γ ++ ssPrev).length := by
          -- by the size lemma for the lowered `GraphData` prefix
          simpa [prev] using
            lowerGraphDataToTape_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
        simpa [htPrev] using hpidlt
      · have hnPrev : Runtime.Autograd.Tape.getNode? (t := tPrev) id = some n := by
          have : tPrev.nodes[id]? = some n := by
            simpa [Array.getElem?_push, hlast] using hnNodes
          simpa [Runtime.Autograd.Tape.getNode?, tPrev] using this
        exact ih id n (by simpa [prev, tPrev] using hnPrev) d contribs hback hmem

/--
All nodes produced by `lowerGraphToTape` have `requiresGrad = true`.

This mirrors `lowerGraphDataToTape_all_requires_grad_true` for the `Graph` interface.
-/
theorem lowerGraphToTape_all_requires_grad_true {α : Type} {Δ : Type}
    [TorchLean.Storage α] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d0 : Δ) :
    ((lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1.nodes.all (fun n =>
      n.requiresGrad)) = true := by
  induction g with
  | nil =>
      have h0 :
          (Runtime.Autograd.Tape.empty (α := α)).nodes.all (fun n => n.requiresGrad) = true := by
        simp [Runtime.Autograd.Tape.empty]
      simpa [lowerGraphToTape] using
        all_requiresGrad_addLeaves (Runtime.Autograd.Tape.empty (α := α)) h0 x
  | snoc g node ih =>
      rename_i ssPrev τ
      -- `lowerGraphToTape` appends a node with `requiresGrad = true`.
      simp [lowerGraphToTape, Runtime.Autograd.Tape.addNode, ih]

/-- Pointwise form of `lowerGraphToTape_all_requires_grad_true`. -/
theorem lowerGraphToTape_requires_grad_true {α : Type} {Δ : Type}
    [TorchLean.Storage α] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d0 : Δ) :
    let t := (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1
    ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requiresGrad = true := by
  intro t i hi
  have hall :
      t.nodes.all (fun n => n.requiresGrad) = true := by
    simpa [t] using
      lowerGraphToTape_all_requires_grad_true (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0
  -- `Array.all_eq_true` gives the pointwise result.
  have := (Array.all_eq_true).1 hall i hi
  simpa using this

/--
Backward closure safety for `lowerGraphToTape`: parent ids produced by any node are strictly
smaller than the node id.

This mirrors `lowerGraphDataToTape_backward_pids_lt_id` for the `Graph` interface.
-/
theorem lowerGraphToTape_backward_pids_lt_id {α : Type} {Δ : Type}
    [TorchLean.Storage α] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d0 : Δ) :
    BackwardPidsLt (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1 := by
  unfold BackwardPidsLt
  induction g with
  | nil =>
      intro id n hn d contribs hback pid pg hmem
      -- `lowerGraphToTape nil` produces leaves with empty backward arrays.
      have hn' :
          ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x).map
            (leafNodeOfSomeTensor (α := α)))[id]? = some n := by
        simpa [lowerGraphToTape, Runtime.Autograd.Tape.getNode?, nodes_addLeaves,
          Runtime.Autograd.Tape.empty] using hn
      cases hx : (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x)[id]? with
      | none =>
          simp [Array.getElem?_map, hx] at hn'
      | some v =>
          have hnEq : n = leafNodeOfSomeTensor (α := α) v := by
            -- `getElem?_map` turns this into `some (leafNodeOfSomeTensor v) = some n`.
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
      let prev := lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      let tPrev := prev.1
      let ctxPrev := prev.2
      let runtimeNode : Runtime.Autograd.Node α :=
        lowerNode (some "proof-carrying-graph") node.toNodeData ctxPrev d0
      have hnNodes :
          (tPrev.nodes.push runtimeNode)[id]? = some n := by
        simpa [lowerGraphToTape, prev, tPrev, ctxPrev, runtimeNode, Runtime.Autograd.Tape.getNode?,
          Runtime.Autograd.Tape.addNode] using hn
      by_cases hlast : id = tPrev.nodes.size
      · subst hlast
        have hnEq : n = runtimeNode := by
          -- `getElem?_push` at `size` yields `some runtimeNode`.
          symm
          simpa [Array.getElem?_push] using hnNodes
        subst hnEq
        have hd : d.shape = τ := by
          by_contra hne
          have : runtimeNode.backward d = .error "autograd: upstream gradient shape mismatch" :=
            lowerNode_backward_of_ne _ _ _ _ d hne
          simp [this] at hback
        have hret : runtimeNode.backward d =
            .ok (TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := Γ ++ ssPrev)
              (node.vjp ctxPrev d0 (d.cast hd)) 0) :=
          lowerNode_backward_of_shape _ _ _ _ d hd
        have hcontribs :
            contribs =
              TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := Γ ++ ssPrev)
                (node.vjp ctxPrev d0 (d.cast hd)) 0 :=
          (Except.ok.inj (hret.symm.trans hback)).symm
        subst hcontribs
        have hpidlt :=
          TorchLean.TensorPack.mem_toIndexedShapeErasedArray_lt
            (α := α) (ss := Γ ++ ssPrev)
            (node.vjp ctxPrev d0 (d.cast hd)) 0 hmem
        have hlen : (Γ ++ ssPrev).length = tPrev.nodes.size := by
          have : tPrev.nodes.size = Γ.length + ssPrev.length := by
            simpa [tPrev, prev] using
              (lowerGraphToTape_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)
          simp [List.length_append, this]
        simpa [Nat.zero_add, hlen] using hpidlt
      · have hnPrev : Runtime.Autograd.Tape.getNode? (t := tPrev) id = some n := by
          have : tPrev.nodes[id]? = some n := by
            simpa [Array.getElem?_push, hlast] using hnNodes
          simpa [Runtime.Autograd.Tape.getNode?, tPrev] using this
        exact ih id n (by simpa [prev, tPrev] using hnPrev) d contribs hback hmem

end Graph

end Algebra
end Autograd
end Proofs
