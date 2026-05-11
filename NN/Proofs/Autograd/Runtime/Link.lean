/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Any

/-!
# Link

Link the executable runtime tape (`Runtime.Autograd.Tape`) to the proved SSA/DAG tape model
(`Proofs.Autograd.Algebra.Graph`).

This file provides a small compiler from proved graphs to runtime tapes. The compiler bakes the
proved `vjp` into each runtime node's `backward` closure.

## What is proved here

- Forward-pass correspondence: `compileAux{,Data}` produces the same values as the proved
  `Graph{,Data}.eval`, and the runtime tape stores those values in the same order
  (`compileAux{,Data}_ctx_eq_eval`, `compileAux{,Data}_values_eq`).
- Backward-pass correspondence: running the runtime dense reverse loop
  `Tape.backwardDenseFrom` on a compiled tape matches the proved “full backpropagation”
  `backpropAllCtx` (`backwardDenseFrom_compileAux_eq_backpropAllCtx` and its `GraphData` variant).

The core invariant making the runtime reverse loop well-founded is that compiled nodes only emit
contributions to earlier node ids (`pid < id`).

## PyTorch correspondence / citations
This is analogous to taking a proven “graph IR” and compiling it to an executable autograd tape
whose nodes carry a backward closure (PyTorch does this internally for the eager autograd engine).
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Algebra

open Spec
open Tensor

namespace Graph

open Runtime
open Runtime.Autograd

/--
Extend a tape with leaf nodes for every tensor in the input context `Γ`.

Each leaf has `requires_grad = true` and `backward = ok []`, so the runtime backward loop treats
them as gradient accumulation slots but never produces parent contributions from them.
-/
def addLeaves {α : Type} (t : Tape α) : {Γ : List Shape} → TList α Γ → Tape α
  | [], .nil => t
  | _ :: Γ, .cons x xs =>
      let (t', _id) := Tape.leaf (t := t) x
      addLeaves (t := t') (Γ := Γ) xs

/--
Turn a value-only `AnyTensor` into a runtime leaf node.

This is the node-level counterpart of `addLeaves`: it has no parents and contributes nothing in
backward.
-/
private def leafNodeOfAny {α : Type} (v : Runtime.AnyTensor α) : Runtime.Autograd.Node α :=
  { name := none
    value := v
    requires_grad := true
    parents := []
    backward := fun _ => .ok [] }

/-- `addLeaves` grows the tape by exactly `Γ.length` nodes. -/
private theorem size_addLeaves {α : Type} (t : Tape α) :
    {Γ : List Shape} → (x : TList α Γ) → (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes.size =
      t.nodes.size + Γ.length
  | [], .nil => by simp [addLeaves]
  | _ :: Γ, .cons x xs => by
      simp [addLeaves, Tape.leaf, Tape.addNode, size_addLeaves (t := { nodes := t.nodes.push _ }) (x
        := xs),
        Nat.add_assoc, Nat.add_comm, Array.size_push]

/-- `addLeaves` appends `leafNodeOfAny` nodes for each element of the input context, in order. -/
private theorem nodes_addLeaves {α : Type} (t : Tape α) :
    {Γ : List Shape} → (x : TList α Γ) →
      (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes =
        t.nodes ++ (TList.toAnyArray (α := α) (ss := Γ) x).map (leafNodeOfAny (α := α))
  | [], .nil => by
      simp [addLeaves, TList.toAnyArray, TList.toAnyList]
  | _ :: Γ, .cons x xs => by
      simp [addLeaves, Tape.leaf, Tape.addNode,
        nodes_addLeaves (t := { nodes := t.nodes.push _ }) (Γ := Γ) (x := xs),
        leafNodeOfAny, TList.toAnyArray_cons (α := α) (ss := Γ) x xs,
        Array.map_append, Array.append_singleton_assoc]

/-- Value projection of `nodes_addLeaves`: `node.value` agrees with `toAnyArray` for added leaves.
  -/
private theorem addLeaves_values {α : Type} (t : Tape α) :
    {Γ : List Shape} → (x : TList α Γ) →
      (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes.map (fun node => node.value) =
        t.nodes.map (fun node => node.value) ++ TList.toAnyArray (α := α) (ss := Γ) x
  | [], .nil => by
      simp [addLeaves, TList.toAnyArray, TList.toAnyList]
  | _ :: Γ, .cons x xs => by
      -- unfold one `leaf` push and use the induction hypothesis on the remaining leaves
      simp [addLeaves, Tape.leaf, Tape.addNode,
        addLeaves_values (t := { nodes := t.nodes.push _ }) (Γ := Γ) (x := xs),
        TList.toAnyArray, TList.toAnyList]

/--
Compile an executable graph (`GraphData`) to a runtime tape by evaluating forward nodes and baking
in each node’s proved `vjp` into its runtime `backward` closure.

PyTorch analogy: this corresponds to building a tape of autograd nodes during the forward pass,
where each node stores enough information to compute parent contributions when given an upstream
cotangent.
-/
def compileAuxData {α : Type} {Δ : Type} [DecidableEq Shape]
  {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
  Tape α × TList α (Γ ++ ss) :=
  match g with
  | .nil =>
      let t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x
      (t, TList.cast (α := α) (h := (List.append_nil Γ).symm) x)
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let (tPrev, ctxPrev) := compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let y := node.forward ctxPrev d
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-compiled"
          value := Runtime.Autograd.AnyTensor.mk y
          requires_grad := true
          parents := []
          backward := fun dLdyAny => by
            if h : dLdyAny.s = τ then
              let dLdy : Tensor α τ := Tensor.castShape dLdyAny.t h
              let contribs := node.vjp ctxPrev d dLdy
              exact .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      let (tNext, _id) := Tape.addNode (t := tPrev) runtimeNode
      let ctxNext :=
        TList.cast (α := α) (h := List.append_assoc Γ ssPrev [τ])
          (TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) ctxPrev y)
      (tNext, ctxNext)

/-!
### Forward-pass correspondence

The next lemmas show that `compileAuxData` preserves the proved forward semantics, and that the
resulting runtime tape contains exactly the evaluated context (erased to `AnyTensor`) in order.
-/

/-- The context returned by `compileAuxData` agrees with the proved `GraphData.eval`. -/
theorem compileAuxData_ctx_eq_eval {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 =
      GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d := by
  induction g with
  | nil =>
      simp [compileAuxData, GraphData.eval]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [compileAuxData, GraphData.eval, ih]

/-- The compiled tape’s `.value` array is `GraphData.eval` erased to `AnyTensor`, in the same order.
  -/
theorem compileAuxData_values_eq {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.map (fun node =>
      node.value) =
      TList.toAnyArray (α := α) (ss := Γ ++ ss) (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss :=
        ss) g x d).2 := by
  induction g with
  | nil =>
      -- only leaves
      simp [compileAuxData, addLeaves_values, Runtime.Autograd.Tape.empty]
  | snoc g _node ih =>
      rename_i ssPrev τ
      simp [compileAuxData, Runtime.Autograd.Tape.addNode, ih]

/-- Size bookkeeping: the compiled tape contains one runtime node for each element of `Γ ++ ss`. -/
theorem compileAuxData_nodes_size {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.size = Γ.length + ss.length
      := by
  induction g with
  | nil =>
      -- only leaves
      simp [compileAuxData, size_addLeaves, Runtime.Autograd.Tape.empty]
  | snoc g _node ih =>
      rename_i ssPrev τ
      simp [compileAuxData, Runtime.Autograd.Tape.addNode, ih, Array.size_push, Nat.add_assoc,
        ]

/--
Compile a proved graph (`Graph`) to a runtime tape by evaluating forward nodes and baking in each
node’s proved `vjp`.

Compared to `compileAuxData`, this uses the pure graph interface (no explicit `GraphData` payload).
-/
def compileAux {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
  {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ) :
  Tape α × TList α (Γ ++ ss) :=
  match g with
  | .nil =>
      let t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x
      (t, TList.cast (α := α) (h := (List.append_nil Γ).symm) x)
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let (tPrev, ctxPrev) := compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let y := node.forward ctxPrev d
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-compiled"
          value := Runtime.Autograd.AnyTensor.mk y
          requires_grad := true
          parents := []
          backward := fun dLdyAny => by
            if h : dLdyAny.s = τ then
              let dLdy : Tensor α τ := Tensor.castShape dLdyAny.t h
              let contribs := node.vjp ctxPrev d dLdy
              exact .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      let (tNext, _id) := Tape.addNode (t := tPrev) runtimeNode
      let ctxNext :=
        TList.cast (α := α) (h := List.append_assoc Γ ssPrev [τ])
          (TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) ctxPrev y)
      (tNext, ctxNext)

/-- The context returned by `compileAux` agrees with the proved `Graph.eval`. -/
theorem compileAux_ctx_eq_eval {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 =
      Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d := by
  induction g with
  | nil =>
      simp [compileAux, Graph.eval]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [compileAux, Graph.eval, ih]

/-- The compiled tape’s `.value` array is `Graph.eval` erased to `AnyTensor`, in the same order. -/
theorem compileAux_values_eq {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.map (fun node => node.value) =
      TList.toAnyArray (α := α) (ss := Γ ++ ss) (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g
        x d).2 := by
  induction g with
  | nil =>
      -- only leaves
      simp [compileAux, addLeaves_values, Runtime.Autograd.Tape.empty]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [compileAux, Runtime.Autograd.Tape.addNode, ih]

/-- Size bookkeeping: `compileAux` produces `Γ.length + ss.length` runtime nodes. -/
theorem compileAux_nodes_size {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.size = Γ.length + ss.length :=
      by
  induction g with
  | nil =>
      simp [compileAux, size_addLeaves, Runtime.Autograd.Tape.empty]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [compileAux, Runtime.Autograd.Tape.addNode, ih, Array.size_push, Nat.add_assoc]

/-!
### Full backpropagation (dense) for proofs and runtime

The runtime engine computes a *dense* gradient array, accumulating cotangents for every node in the
tape (inputs and intermediates). The following definition and theorems connect that behavior to the
proved backpropagation semantics.
-/

/-- A "full" backpropagation that returns gradients for *all* values (`Γ ++ ss`), not just `Γ`. -/
def backpropAllCtx {α : Type} {Δ : Type} [CommSemiring α]
  {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ)
  (seed : TList α (Γ ++ ss)) :
  TList α (Γ ++ ss) :=
  match g with
  | .nil => seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) := TList.cast (α := α) (h := assoc.symm) seed
      let seedPrev : TList α (Γ ++ ssPrev) := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
        seed').1
      let seedOut : Tensor α τ := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := TList.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev := backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d seedPrev'
      TList.cast (α := α) (h := assoc)
        (TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) gradsPrev seedOut)

/--
“Full” backpropagation for `GraphData` that returns gradients for *all* values (`Γ ++ ss`), not just
  inputs.

This is the `GraphData`-analogue of `backpropAllCtx` above. We keep both definitions because:
- `Graph` uses `[CommSemiring α]` (so it can express dot products and semiring-based accumulation),
  while
- `GraphData` only needs `[Add α]` here (it just adds contributions).

Both follow the same reverse-mode accumulation structure: peel off the last node, apply its VJP to
the seed on that node, add into the previous seed, and recurse.
-/
def _root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx {α : Type} {Δ : Type} [Add α]
  {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ)
  (seed : TList α (Γ ++ ss)) :
  TList α (Γ ++ ss) :=
  match g with
  | .nil => seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) := TList.cast (α := α) (h := assoc.symm) seed
      let seedPrev : TList α (Γ ++ ssPrev) := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
        seed').1
      let seedOut : Tensor α τ := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := TList.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev := backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d seedPrev'
      TList.cast (α := α) (h := assoc)
        (TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) gradsPrev seedOut)

/-!
## Runtime link: `compileAux` + `Tape.backwardDenseFrom`

`compileAux` produces a runtime tape whose node ids correspond to positions in the proof context
`Γ ++ ss`, and bakes the proved `vjp` into each node’s runtime `backward` closure.

The theorem `backwardDenseFrom_compileAux_eq_backpropAllCtx` states that executing the runtime
reverse-mode loop on this compiled tape matches the proved `backpropAllCtx`.
-/

/--
All nodes produced by `compileAuxData` have `requires_grad = true`.

This is a simplifying invariant: the compiled tape is meant for correctness proofs, so we mark
every node as eligible for gradient accumulation (including leaves for inputs).
-/
private theorem compileAuxData_all_requires_grad_true {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
    ((compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.all (fun n =>
      n.requires_grad)) = true := by
  -- Helper: if the current tape has `.all requires_grad = true`, `addLeaves` preserves it.
  have addLeaves_all :
      ∀ (t : Tape α),
        t.nodes.all (fun n => n.requires_grad) = true →
          ∀ {Γ : List Shape} (xs : TList α Γ),
            (addLeaves (α := α) (t := t) (Γ := Γ) xs).nodes.all (fun n => n.requires_grad) = true :=
              by
    intro t ht Γ xs
    induction xs generalizing t with
    | nil =>
        simpa [addLeaves] using ht
    | cons x xs ih =>
        -- push one leaf (which has `requires_grad = true`) and recurse
        let t' : Tape α := (Runtime.Autograd.Tape.leaf (t := t) x).1
        have ht' : t'.nodes.all (fun n => n.requires_grad) = true := by
          simpa [t', Runtime.Autograd.Tape.leaf, Runtime.Autograd.Tape.addNode, Array.all_push]
            using ht
        simpa [addLeaves, t', Runtime.Autograd.Tape.leaf, Runtime.Autograd.Tape.addNode] using ih (t
          := t') ht'

  induction g with
  | nil =>
      have h0 : (Runtime.Autograd.Tape.empty (α := α)).nodes.all (fun n => n.requires_grad) = true
        := by
        simp [Runtime.Autograd.Tape.empty]
      simpa [compileAuxData] using addLeaves_all (t := Runtime.Autograd.Tape.empty (α := α)) h0 (Γ
        := Γ) x
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [compileAuxData, Runtime.Autograd.Tape.addNode, ih]

/--
Pointwise form of `compileAuxData_all_requires_grad_true`: every node index is `requires_grad =
  true`.

This is often more convenient than the `.all` formulation when reasoning about array indexing.
-/
private theorem compileAuxData_requires_grad_true {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
    let t := (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1
    ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requires_grad = true := by
  intro t i hi
  have hall :
      t.nodes.all (fun n => n.requires_grad) = true := by
    simpa [t] using compileAuxData_all_requires_grad_true (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x
      d
  have := (Array.all_eq_true).1 hall i hi
  simpa using this

/--
Backward closure safety for `compileAuxData`: parent ids produced by any node are strictly smaller
  than the node id.

This is the “edges point backwards” invariant required by the runtime reverse loop: when processing
node `id`, every contribution targets an earlier node (`pid < id`), so accumulation is well-founded.
-/
private theorem compileAuxData_backward_pids_lt_id {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d0 : Δ) :
    ∀ id (n : Runtime.Autograd.Node α),
      (Runtime.Autograd.Tape.getNode? (t := (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g
        x d0).1) id = some n) →
      ∀ (d : Runtime.AnyTensor α) (contribs : List (Nat × Runtime.AnyTensor α)),
        n.backward d = .ok contribs →
          ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ contribs → pid < id := by
  induction g with
  | nil =>
      intro id n hn d contribs hback pid pg hmem
      -- `compileAuxData nil` produces only leaves with `backward = ok []`.
      have hn' :
          ((TList.toAnyArray (α := α) (ss := Γ) x).map (leafNodeOfAny (α := α)))[id]? = some n := by
        simpa [compileAuxData, Runtime.Autograd.Tape.getNode?, nodes_addLeaves,
          Runtime.Autograd.Tape.empty] using hn
      cases hx : (TList.toAnyArray (α := α) (ss := Γ) x)[id]? with
      | none =>
          simp [Array.getElem?_map, hx] at hn'
      | some v =>
          have hnEq : n = leafNodeOfAny (α := α) v := by
            symm
            simpa [Array.getElem?_map, hx] using hn'
          subst hnEq
          have hcontribs : contribs = [] := by
            have := congrArg (fun r => match r with | .ok l => l | .error _ => []) hback
            simpa [leafNodeOfAny] using this
          subst hcontribs
          cases hmem
  | snoc g node ih =>
      rename_i ssPrev τ
      intro id n hn d contribs hback pid pg hmem
      let prev := compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      let tPrev := prev.1
      let ctxPrev := prev.2
      let y := node.forward ctxPrev d0
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-compiled"
          value := Runtime.Autograd.AnyTensor.mk y
          requires_grad := true
          parents := []
          backward := fun dLdyAny => by
            if h : dLdyAny.s = τ then
              let dLdy : Tensor α τ := Tensor.castShape dLdyAny.t h
              let contribs := node.vjp ctxPrev d0 dLdy
              exact .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      have hnNodes :
          (tPrev.nodes.push runtimeNode)[id]? = some n := by
        simpa [compileAuxData, prev, tPrev, ctxPrev, y, runtimeNode, Runtime.Autograd.Tape.getNode?,
          Runtime.Autograd.Tape.addNode] using hn
      by_cases hlast : id = tPrev.nodes.size
      · subst hlast
        have hnEq : n = runtimeNode := by
          symm
          simpa [Array.getElem?_push] using hnNodes
        subst hnEq
        have hd : d.s = τ := by
          by_contra hne
          have : runtimeNode.backward d = .error "autograd: upstream gradient shape mismatch" := by
            have : d.s ≠ τ := hne
            simp [runtimeNode, this]
          simp [this]  at hback
        have hcontribs :
            contribs =
              TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev)
                (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0 := by
          let listExpr :=
            TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev)
              (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0
          have hret : runtimeNode.backward d = .ok listExpr := by
            simp [runtimeNode, hd, listExpr]
          have hok :
              (.ok listExpr : Result (List (Nat × Runtime.AnyTensor α))) = .ok contribs := by
            calc
              (.ok listExpr : Result (List (Nat × Runtime.AnyTensor α))) = runtimeNode.backward d :=
                by
                simpa using hret.symm
              _ = .ok contribs := hback
          have := congrArg (fun r => match r with | .ok l => l | .error _ => []) hok
          simpa [listExpr] using this.symm
        subst hcontribs
        have hpidlt :=
          TList.mem_toIndexedAnyList_lt (α := α) (ss := Γ ++ ssPrev)
            (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0 (pid := pid) (pg := pg) hmem
        -- `0 + (Γ ++ ssPrev).length = tPrev.nodes.size`
        have htPrev :
            tPrev.nodes.size = (Γ ++ ssPrev).length := by
          -- by the size lemma for compiled GraphData prefix
          simpa [prev] using
            compileAuxData_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
        simpa [htPrev] using hpidlt
      · have hidPrev : id < tPrev.nodes.size := by
          have hidPush : id < (tPrev.nodes.push runtimeNode).size := by
            rcases Array.getElem_of_getElem? hnNodes with ⟨hid, _⟩
            exact hid
          have hidLe : id ≤ tPrev.nodes.size := by
            have : id < tPrev.nodes.size + 1 := by
              simpa [Array.size_push] using hidPush
            exact Nat.le_of_lt_succ this
          exact Nat.lt_of_le_of_ne hidLe hlast
        have hnPrev : Runtime.Autograd.Tape.getNode? (t := tPrev) id = some n := by
          have : tPrev.nodes[id]? = some n := by
            simpa [Array.getElem?_push, hlast] using hnNodes
          simpa [Runtime.Autograd.Tape.getNode?, tPrev] using this
        exact ih id n (by simpa [prev, tPrev] using hnPrev) d contribs hback hmem

/--
All nodes produced by `compileAux` have `requires_grad = true`.

This mirrors `compileAuxData_all_requires_grad_true` for the `Graph` interface.
-/
private theorem compileAux_all_requires_grad_true {α : Type} {Δ : Type} [DecidableEq Shape]
  [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d0 : Δ) :
    ((compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1.nodes.all (fun n =>
      n.requires_grad)) = true := by
  -- Helper: if the current tape has `.all requires_grad = true`, `addLeaves` preserves it.
  have addLeaves_all :
      ∀ (t : Tape α),
        t.nodes.all (fun n => n.requires_grad) = true →
          ∀ {Γ : List Shape} (xs : TList α Γ),
            (addLeaves (α := α) (t := t) (Γ := Γ) xs).nodes.all (fun n => n.requires_grad) = true :=
              by
    intro t ht Γ xs
    induction xs generalizing t with
    | nil =>
        simpa [addLeaves] using ht
    | cons x xs ih =>
        -- push one leaf (which has `requires_grad = true`) and recurse
        let t' : Tape α := (Runtime.Autograd.Tape.leaf (t := t) x).1
        have ht' : t'.nodes.all (fun n => n.requires_grad) = true := by
          -- `leaf` pushes a node with `requires_grad = true`, so `.all` is preserved
          simpa [t', Runtime.Autograd.Tape.leaf, Runtime.Autograd.Tape.addNode, Array.all_push]
            using ht
        simpa [addLeaves, t', Runtime.Autograd.Tape.leaf, Runtime.Autograd.Tape.addNode] using ih (t
          := t') ht'

  induction g with
  | nil =>
      -- Start from the empty tape where `.all _ = true`.
      have h0 : (Runtime.Autograd.Tape.empty (α := α)).nodes.all (fun n => n.requires_grad) = true
        := by
        simp [Runtime.Autograd.Tape.empty]
      simpa [compileAux] using addLeaves_all (t := Runtime.Autograd.Tape.empty (α := α)) h0 (Γ := Γ)
        x
  | snoc g node ih =>
      rename_i ssPrev τ
      -- `compileAux` appends a node with `requires_grad = true`.
      simp [compileAux, Runtime.Autograd.Tape.addNode, ih]

/-- Pointwise form of `compileAux_all_requires_grad_true`. -/
private theorem compileAux_requires_grad_true {α : Type} {Δ : Type} [DecidableEq Shape]
  [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d0 : Δ) :
    let t := (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1
    ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requires_grad = true := by
  intro t i hi
  have hall :
      t.nodes.all (fun n => n.requires_grad) = true := by
    simpa [t] using compileAux_all_requires_grad_true (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0
  -- `Array.all_eq_true` gives the pointwise result.
  have := (Array.all_eq_true).1 hall i hi
  simpa using this

/--
Backward closure safety for `compileAux`: parent ids produced by any node are strictly smaller than
  the node id.

This mirrors `compileAuxData_backward_pids_lt_id` for the `Graph` interface.
-/
private theorem compileAux_backward_pids_lt_id {α : Type} {Δ : Type} [DecidableEq Shape]
  [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d0 : Δ) :
    ∀ id (n : Runtime.Autograd.Node α),
      (Runtime.Autograd.Tape.getNode? (t := (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x
        d0).1) id = some n) →
      ∀ (d : Runtime.AnyTensor α) (contribs : List (Nat × Runtime.AnyTensor α)),
        n.backward d = .ok contribs →
          ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ contribs → pid < id := by
  induction g with
  | nil =>
      intro id n hn d contribs hback pid pg hmem
      -- `compileAux nil` produces only leaves with `backward = ok []`.
      have hn' :
          ((TList.toAnyArray (α := α) (ss := Γ) x).map (leafNodeOfAny (α := α)))[id]? = some n := by
        simpa [compileAux, Runtime.Autograd.Tape.getNode?, nodes_addLeaves,
          Runtime.Autograd.Tape.empty] using hn
      cases hx : (TList.toAnyArray (α := α) (ss := Γ) x)[id]? with
      | none =>
          simp [Array.getElem?_map, hx] at hn'
      | some v =>
          have hnEq : n = leafNodeOfAny (α := α) v := by
            -- `getElem?_map` turns this into `some (leafNodeOfAny v) = some n`.
            symm
            simpa [Array.getElem?_map, hx] using hn'
          subst hnEq
          -- `leafNodeOfAny.backward = ok []`, so `contribs = []`.
          have hcontribs : contribs = [] := by
            have := congrArg (fun r => match r with | .ok l => l | .error _ => []) hback
            simpa [leafNodeOfAny] using this
          subst hcontribs
          cases hmem
  | snoc g node ih =>
      rename_i ssPrev τ
      intro id n hn d contribs hback pid pg hmem
      let prev := compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      let tPrev := prev.1
      let ctxPrev := prev.2
      let y := node.forward ctxPrev d0
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-compiled"
          value := Runtime.Autograd.AnyTensor.mk y
          requires_grad := true
          parents := []
          backward := fun dLdyAny => by
            if h : dLdyAny.s = τ then
              let dLdy : Tensor α τ := Tensor.castShape dLdyAny.t h
              let contribs := node.vjp ctxPrev d0 dLdy
              exact .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      have hnNodes :
          (tPrev.nodes.push runtimeNode)[id]? = some n := by
        simpa [compileAux, prev, tPrev, ctxPrev, y, runtimeNode, Runtime.Autograd.Tape.getNode?,
          Runtime.Autograd.Tape.addNode] using hn
      by_cases hlast : id = tPrev.nodes.size
      · subst hlast
        have hnEq : n = runtimeNode := by
          -- `getElem?_push` at `size` yields `some runtimeNode`.
          symm
          simpa [Array.getElem?_push] using hnNodes
        subst hnEq
        have hd : d.s = τ := by
          by_contra hne
          have : runtimeNode.backward d = .error "autograd: upstream gradient shape mismatch" := by
            have : d.s ≠ τ := hne
            simp [runtimeNode, this]
          simp [this]  at hback
        have hcontribs :
            contribs =
              TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev)
                (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0 := by
          let listExpr :=
            TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev)
              (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0
          have hret : runtimeNode.backward d = .ok listExpr := by
            simp [runtimeNode, hd, listExpr]
          have hok :
              (.ok listExpr : Result (List (Nat × Runtime.AnyTensor α))) = .ok contribs := by
            calc
              (.ok listExpr : Result (List (Nat × Runtime.AnyTensor α))) = runtimeNode.backward d :=
                by
                simpa using hret.symm
              _ = .ok contribs := hback
          have := congrArg (fun r => match r with | .ok l => l | .error _ => []) hok
          simpa [listExpr] using this.symm
        subst hcontribs
        have hpidlt :=
          TList.mem_toIndexedAnyList_lt (α := α) (ss := Γ ++ ssPrev)
            (node.vjp ctxPrev d0 (Tensor.castShape d.t hd)) 0 hmem
        have hlen : (Γ ++ ssPrev).length = tPrev.nodes.size := by
          have : tPrev.nodes.size = Γ.length + ssPrev.length := by
            simpa [tPrev, prev] using
              (compileAux_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)
          simp [List.length_append, this]
        simpa [Nat.zero_add, hlen] using hpidlt
      · have hnPrev : Runtime.Autograd.Tape.getNode? (t := tPrev) id = some n := by
          have : tPrev.nodes[id]? = some n := by
            simpa [Array.getElem?_push, hlast] using hnNodes
          simpa [Runtime.Autograd.Tape.getNode?, tPrev] using this
        exact ih id n (by simpa [prev, tPrev] using hnPrev) d contribs hback hmem

/--
Key accumulation lemma for the runtime dense gradient array:

Folding `Tape.addGradAll` over the contributions corresponding to a `TList` (via `toIndexedAnyList`)
is equivalent to pointwise addition of the typed contexts (`TList.add`), embedded back into the
array layout `pref ++ seed ++ suffix`.

This is the “runtime accumulation matches proved addition” bridge.
-/
private theorem foldlM_addGradAll_toIndexedAnyList_eq_add {α : Type} [Add α] [DecidableEq Shape]
    (t : Runtime.Autograd.Tape α) :
    ∀ {ss : List Shape} (pref : Array (Runtime.AnyTensor α)) (seed contrib : TList α ss)
      (suffix : Array (Runtime.AnyTensor α)),
      (∀ i (hi : i < ss.length),
        let id := pref.size + i
        ∃ node : Runtime.Autograd.Node α,
          t.getNode? id = some node ∧
            node.requires_grad = true ∧
            node.value.s =
              ((TList.toAnyArray (α := α) (ss := ss) seed)[i]'(by
                  simpa [TList.size_toAnyArray] using hi)).s) →
      (TList.toIndexedAnyList (α := α) (ss := ss) contrib pref.size).foldlM
          (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid pg)
          (pref ++ TList.toAnyArray (α := α) (ss := ss) seed ++ suffix) =
        .ok (pref ++
              TList.toAnyArray (α := α) (ss := ss) (TList.add (α := α) (ss := ss) seed contrib) ++
              suffix) := by
  intro ss pref seed contrib suffix hnodes
  induction ss generalizing pref with
  | nil =>
      cases seed; cases contrib
      simp [TList.toIndexedAnyList, TList.toAnyArray, TList.toAnyList, TList.add]
      rfl
  | cons s ss ih =>
      cases seed with
      | cons seedHead seedTail =>
        cases contrib with
        | cons contribHead contribTail =>
          let seedHeadAny : Runtime.AnyTensor α := Runtime.Autograd.AnyTensor.mk seedHead
          let contribHeadAny : Runtime.AnyTensor α := Runtime.Autograd.AnyTensor.mk contribHead
          let newHeadAny : Runtime.AnyTensor α := Runtime.Autograd.AnyTensor.mk (addSpec seedHead
            contribHead)

          have hseedArr :
              TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead seedTail) =
                #[seedHeadAny] ++ TList.toAnyArray (α := α) (ss := ss) seedTail := by
            simpa [seedHeadAny] using
              (TList.toAnyArray_cons (α := α) (s := s) (ss := ss) seedHead seedTail)

          have hacc0 :
              pref ++ TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead seedTail) ++
                suffix =
                (pref.push seedHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++ suffix
                  := by
            -- Avoid `simp` loops between `push` and `++ #[x]`.
            -- Expand the seed array, reassociate, then rewrite `pref ++ #[x]` as `pref.push x`.
            rw [hseedArr]
            simp [Array.append_assoc]

          have h0 := hnodes 0 (by simp [List.length_cons])
          rcases h0 with ⟨node0, hnode0, hreq0, hshape0'⟩

          have hshape0 : node0.value.s = seedHeadAny.s := by
            have : ((TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead
              seedTail))[0]'(by
              simp [TList.size_toAnyArray, List.length_cons])).s = seedHeadAny.s := by
              simp [hseedArr, seedHeadAny]
            exact hshape0'.trans this

          have hgetExisting :
              ((pref.push seedHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                suffix)[pref.size]? =
                some seedHeadAny := by
            have hlt : pref.size < (pref.push seedHeadAny).size := by
              simp
            simp [Array.getElem?_append]

          have hsummed : Runtime.Autograd.AnyTensor.add seedHeadAny contribHeadAny = .ok newHeadAny
            := by
            -- Reduce the shape-cast using definitional equality of shapes.
            have hs :
                (Runtime.Autograd.AnyTensor.mk seedHead).s =
                  (Runtime.Autograd.AnyTensor.mk contribHead).s := by
              rfl
            cases hs
            simp [Runtime.Autograd.AnyTensor.add, Runtime.Autograd.AnyTensor.mk, Tensor.castShape,
              seedHeadAny, contribHeadAny, newHeadAny]

          have hset :
              ((pref.push seedHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                suffix).set
                  pref.size newHeadAny
                  (by
                    simp [Array.size_append, Nat.add_assoc]) =
                (pref.push newHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++ suffix :=
                  by
            have hlt : pref.size < (pref.push seedHeadAny).size := by
              simp
            simp [Array.set_append_left (xs := pref.push seedHeadAny)
                  (ys := TList.toAnyArray (α := α) (ss := ss) seedTail ++ suffix)
                  (i := pref.size) (x := newHeadAny) hlt,
              Array.set_push]

          have hadd0 :
              Runtime.Autograd.Tape.addGradAll (t := t)
                  (pref ++ TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead seedTail)
                    ++ suffix)
                  pref.size contribHeadAny =
                .ok ((pref.push newHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                  suffix) := by
              have hidAcc :
                  pref.size <
                  ((pref.push seedHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                    suffix).size := by
                simp [Array.size_append, Nat.add_assoc]
              have hshapeG : contribHeadAny.s = node0.value.s := by
                simpa [contribHeadAny, seedHeadAny] using hshape0.symm
              have hshapeExisting : seedHeadAny.s = node0.value.s := by
                simpa using hshape0.symm
              have hnode0' : t.getNode? pref.size = some node0 := by
                simpa [Nat.add_zero] using hnode0
              have hgetExisting' :
                  ((pref.push seedHeadAny) ++ (TList.toAnyArray (α := α) (ss := ss) seedTail ++
                    suffix))[pref.size]? =
                    some seedHeadAny := by
                simpa [Array.append_assoc] using hgetExisting
              have hidAcc' :
                  pref.size <
                    ((pref.push seedHeadAny) ++ (TList.toAnyArray (α := α) (ss := ss) seedTail ++
                      suffix)).size := by
                simpa [Array.append_assoc] using hidAcc

              have : Runtime.Autograd.Tape.addGradAll (t := t)
                  ((pref.push seedHeadAny) ++ (TList.toAnyArray (α := α) (ss := ss) seedTail ++
                    suffix))
                  pref.size contribHeadAny =
                    .ok ((pref.push newHeadAny) ++
                      (TList.toAnyArray (α := α) (ss := ss) seedTail ++ suffix)) := by
                -- After rewriting `node0.value.s = seedHeadAny.s`, all shape casts become
                -- definitional.
                cases hshape0
                -- Reduce the dependent shape-casts by eliminating the equality proofs.
                cases hshapeExisting
                cases hshapeG

                have hid :
                    pref.size < pref.size + 1 + (ss.length + suffix.size) := by
                  -- `pref.size < pref.size + 1` and adding to the RHS preserves `<`.
                  exact Nat.lt_of_lt_of_le (Nat.lt_succ_self pref.size)
                    (Nat.le_add_right (pref.size + 1) (ss.length + suffix.size))

                have hseedShape : (Runtime.Autograd.AnyTensor.mk seedHead).s = node0.value.s := by
                  rfl
                have hcontribShape :
                    (Runtime.Autograd.AnyTensor.mk contribHead).s = node0.value.s := by
                  rfl

                -- Now `addGradAll` is a straight-line computation: fetch node, check flags/shapes,
                -- add, and overwrite the `pref.size` slot.
                -- Keep `Tensor.cast_shape` opaque here: in Lean 4.29, unfolding it too early tends
                -- to leave behind `cast` terms that make later rewrites brittle.
                simp [Runtime.Autograd.Tape.addGradAll, hnode0', hreq0, hseedShape, hcontribShape,
                  hid, seedHeadAny, contribHeadAny, newHeadAny, Array.set_push]

                have hsummed' :
                    Runtime.Autograd.AnyTensor.add
                        { s := node0.value.s
                          t := Tensor.castShape (Runtime.Autograd.AnyTensor.mk seedHead).t
                            hseedShape }
                        { s := node0.value.s
                          t := Tensor.castShape (Runtime.Autograd.AnyTensor.mk contribHead).t
                            hcontribShape } =
                      .ok newHeadAny := by
                  cases hseedShape
                  cases hcontribShape
                  simpa [seedHeadAny, contribHeadAny] using hsummed

                rw [hsummed']
                simp [newHeadAny]

              -- Rewrite back to the original associative form for the outer goal.
              simpa [Array.append_assoc, hacc0] using this

          have hnodesTail :
              ∀ i (hi : i < ss.length),
                let id := (pref.push newHeadAny).size + i
                ∃ node : Runtime.Autograd.Node α,
                  t.getNode? id = some node ∧ node.requires_grad = true ∧
                    node.value.s =
                      ((TList.toAnyArray (α := α) (ss := ss) seedTail)[i]'(by
                          simpa [TList.size_toAnyArray] using hi)).s := by
            intro i hi
            have h' :=
              hnodes (i + 1) (by
                simpa [List.length_cons] using Nat.succ_lt_succ hi)
            rcases h' with ⟨node, hnode, hreq, hshape⟩
            refine ⟨node, ?_, hreq, ?_⟩
            · simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hnode
            ·
              have hiFull :
                  i + 1 < (TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead
                    seedTail)).size := by
                have : i + 1 < (s :: ss).length := by
                  simpa [List.length_cons] using Nat.succ_lt_succ hi
                simpa [TList.size_toAnyArray] using this
              have hiTail :
                  i < (TList.toAnyArray (α := α) (ss := ss) seedTail).size := by
                simpa [TList.size_toAnyArray] using hi
              have hidx :
                  (TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead seedTail))[i +
                    1]'hiFull =
                    (TList.toAnyArray (α := α) (ss := ss) seedTail)[i]'hiTail := by
                have hcons :=
                  (TList.toAnyArray_cons (α := α) (s := s) (ss := ss) seedHead seedTail)
                have hxs : (#[(Runtime.Autograd.AnyTensor.mk seedHead)] : Array (Runtime.AnyTensor
                  α)).size ≤ i + 1 := by
                  simp
                have : (i + 1) - (#[(Runtime.Autograd.AnyTensor.mk seedHead)] : Array
                  (Runtime.AnyTensor α)).size = i := by
                  simp
                simp [hcons]
              have hshape' :
                  ((TList.toAnyArray (α := α) (ss := s :: ss) (TList.cons seedHead seedTail))[i +
                    1]'hiFull).s =
                    ((TList.toAnyArray (α := α) (ss := ss) seedTail)[i]'hiTail).s := by
                simpa using congrArg Runtime.AnyTensor.s hidx
              simpa [hshape'] using hshape

          have htail :=
            ih (pref := pref.push newHeadAny) (seed := seedTail) (contrib := contribTail) hnodesTail

          -- The `foldlM` over the cons list runs one `addGradAll` step, then continues with the
          -- tail.
          -- We need the "push-form" of `hadd0` to rewrite that first step.
          have hadd0Push :
              Runtime.Autograd.Tape.addGradAll (t := t)
                  ((pref.push seedHeadAny) ++ (TList.toAnyArray (α := α) (ss := ss) seedTail ++
                    suffix))
                  pref.size contribHeadAny =
                .ok ((pref.push newHeadAny) ++ (TList.toAnyArray (α := α) (ss := ss) seedTail ++
                  suffix)) := by
            have hadd0' :
                Runtime.Autograd.Tape.addGradAll (t := t)
                    ((pref.push seedHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                      suffix)
                    pref.size contribHeadAny =
                  .ok ((pref.push newHeadAny) ++ TList.toAnyArray (α := α) (ss := ss) seedTail ++
                    suffix) := by
              simpa [hacc0] using hadd0
            simpa [Array.append_assoc] using hadd0'

          simpa [TList.toIndexedAnyList, List.foldlM, hadd0Push, TList.add, TList.toAnyArray_cons,
            Array.append_assoc, seedHeadAny, contribHeadAny, newHeadAny]
            using htail

/--
`Tape.addGradAll` never changes the size of the dense gradient array in the `.ok` case.

This is a structural property needed to show the runtime reverse loop preserves array sizes.
-/
private theorem addGradAll_size_preserved {α : Type} [Add α] [DecidableEq Shape]
    (t : Runtime.Autograd.Tape α) (grads : Array (Runtime.AnyTensor α)) (id : Nat) (g :
      Runtime.AnyTensor α) :
    match Runtime.Autograd.Tape.addGradAll (t := t) grads id g with
    | .ok grads' => grads'.size = grads.size
    | .error _ => True := by
  cases hnode : Runtime.Autograd.Tape.getNode? (t := t) id with
  | none =>
      -- Reduce `addGradAll` to its initial `throw`, then simplify the match.
      have hadd :
          Runtime.Autograd.Tape.addGradAll (t := t) grads id g =
            .error "autograd: invalid parent id during backward" := by
        simp [Runtime.Autograd.Tape.addGradAll, hnode, throw, throwThe, MonadExceptOf.throw]
        rfl
      simp [hadd]
  | some node =>
      by_cases hreq : node.requires_grad = false
      · -- No-op case: `pure grads`.
        have hadd : Runtime.Autograd.Tape.addGradAll (t := t) grads id g = .ok grads := by
          simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq]
          rfl
        simp [hadd]
      · by_cases hshape : g.s = node.value.s
        ·
          let g' : Runtime.AnyTensor α := { s := node.value.s, t := Tensor.castShape g.t hshape }
          cases hgrad : grads[id]? with
          | none =>
              simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq, hshape, hgrad,
                throw, throwThe, MonadExceptOf.throw]
          | some existing =>
              by_cases hex : existing.s = node.value.s
              ·
                let existing' : Runtime.AnyTensor α :=
                  { s := node.value.s, t := Tensor.castShape existing.t hex }
                cases hadd : Runtime.Autograd.AnyTensor.add existing' g' with
                | error e =>
                    -- In the `AnyTensor.add = .error e` case, `addGradAll` errors too, so the size
                    -- statement is `True`.
                    have haddAll :
                        Runtime.Autograd.Tape.addGradAll (t := t) grads id g = .error e := by
                      simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq, hshape, hgrad, hex, g',
                        existing', hadd,
                        throw, throwThe, MonadExceptOf.throw]
                      simp [Bind.bind, Except.bind]
                    simp [haddAll]
                | ok summed =>
                    -- `grads[id]? = some existing` implies `id < grads.size`, so `set` is
                    -- in-bounds.
                    rcases
                      (Array.getElem_of_getElem? (xs := grads) (i := id) (a := existing) hgrad) with
                      ⟨hid, hget⟩

                    -- Help `simp` pick the correct shape-check branch:
                    -- `grads[id].s = existing.s = node.value.s`.
                    have hs : grads[id].s = node.value.s := by
                      simpa [hget] using hex

                    -- Now the `do`-block in `addGradAll` reduces all the way to an in-bounds `set`.
                    have haddAll :
                        Runtime.Autograd.Tape.addGradAll (t := t) grads id g =
                          .ok (grads.set id summed (h := hid)) := by
                      -- First reduce the control flow of `addGradAll` down to the final `map/set`
                      -- line.
                      simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq, hshape,
                        hid, hs, throw, throwThe, MonadExceptOf.throw]

                      -- Identify the `AnyTensor.add` call produced by the reduced code with our
                      -- `hadd`.
                      have hadd2 :
                          Runtime.Autograd.AnyTensor.add
                              { s := node.value.s, t := Tensor.castShape grads[id].t hs }
                              { s := node.value.s, t := Tensor.castShape g.t hshape } =
                            Except.ok summed := by
                        -- The first argument is `existing'` up to proof-irrelevant `cast_shape`.
                        have harg1 :
                            ({ s := node.value.s, t := Tensor.castShape grads[id].t hs } :
                              Runtime.AnyTensor α) =
                              existing' := by
                          -- Rewrite `grads[id] = existing`, then use proof irrelevance on the
                          -- shape-cast proof.
                          cases hget
                          simp [existing']
                        simpa [g', harg1] using hadd

                      simp [hadd2]

                    simp [haddAll, Array.size_set]
              ·
                simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq, hshape, hgrad, hex,
                  throw, throwThe, MonadExceptOf.throw]
        ·
          simp [Runtime.Autograd.Tape.addGradAll, hnode, hreq, hshape, throw, throwThe,
            MonadExceptOf.throw]

/-- If `addGradAll` returns `.ok grads'`, then `grads'.size = grads.size`. -/
private theorem addGradAll_ok_size {α : Type} [Add α] [DecidableEq Shape]
    (t : Runtime.Autograd.Tape α) :
    ∀ {grads : Array (Runtime.AnyTensor α)} {id : Nat} {g : Runtime.AnyTensor α}
      {grads' : Array (Runtime.AnyTensor α)},
      Runtime.Autograd.Tape.addGradAll (t := t) grads id g = .ok grads' →
        grads'.size = grads.size := by
  intro grads id g grads' h
  simpa [h] using addGradAll_size_preserved (t := t) grads id g

/--
If one step of the runtime dense backward loop succeeds, it preserves the accumulator array size.

This is proved by showing the internal `foldlM addGradAll` preserves size, then splitting on the
control flow of `backwardDenseFromStep`.
-/
private theorem backwardDenseFromStep_ok_size {α : Type} [Add α] [DecidableEq Shape]
    (t : Runtime.Autograd.Tape α) :
    ∀ {acc : Array (Runtime.AnyTensor α)} {id : Nat} {acc' : Array (Runtime.AnyTensor α)},
      Runtime.Autograd.Tape.backwardDenseFromStep (t := t) acc id = .ok acc' →
        acc'.size = acc.size := by
  intro acc id acc' h
  -- `foldlM` over `addGradAll` preserves `size` in the `.ok` case.
  have fold_ok_size :
      ∀ (contribs : List (Nat × Runtime.AnyTensor α)) (acc0 accOut : Array (Runtime.AnyTensor α)),
        (contribs.foldlM (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid
          pg) acc0 =
            .ok accOut) →
          accOut.size = acc0.size := by
    intro contribs acc0 accOut hfold
    induction contribs generalizing acc0 accOut with
    | nil =>
        simp [List.foldlM] at hfold
        cases hfold
        rfl
    | cons head tail ih =>
        cases head with
        | mk pid pg =>
            cases h1 : Runtime.Autograd.Tape.addGradAll (t := t) acc0 pid pg with
            | error e =>
                simp [List.foldlM, h1] at hfold
                cases hfold
            | ok acc1 =>
                have htail :
                    tail.foldlM
                        (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid
                          pg) acc1 =
                      .ok accOut := by
                  simpa [List.foldlM, h1] using hfold
                have hs1 : acc1.size = acc0.size :=
                  addGradAll_ok_size (t := t) (grads := acc0) (id := pid) (g := pg) (grads' := acc1)
                    (by
                    simpa using h1)
                have := ih (acc0 := acc1) (accOut := accOut) htail
                simpa [hs1] using this

  cases hnode : Runtime.Autograd.Tape.getNode? (t := t) id with
  | none =>
      simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode] at h
      cases h
  | some node =>
      by_cases hreq : node.requires_grad = false
      · simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, hreq] at h
        cases h
        simp
      · cases hgrad : acc[id]? with
        | none =>
            simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, hreq, hgrad] at h
            cases h
          | some dLdyAny =>
              by_cases hshape : dLdyAny.s = node.value.s
              ·
                let dLdy : Runtime.AnyTensor α :=
                  { s := node.value.s, t := Tensor.castShape dLdyAny.t hshape }
                cases hback : node.backward dLdy with
                | error e =>
                    simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, hreq, hgrad, hshape,
                      dLdy, hback] at h
                    cases h
                | ok contribs =>
                    have hfold :
                        contribs.foldlM
                            (fun acc2 (pid, pg) =>
                              Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid pg)
                            acc =
                          .ok acc' := by
                      simpa
                        [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, hreq, hgrad, hshape,
                          dLdy, hback]
                        using h
                    simpa using fold_ok_size contribs acc acc' hfold
              ·
                have : False := by
                  simp
                    [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, hreq, hgrad, hshape]
                    at h
                cases this

/--
**Main runtime/link theorem**: running the runtime dense backward loop on a tape produced by
`compileAux` matches the proved “full backpropagation” `backpropAllCtx`.

This is the formal statement that the executable engine implements the same reverse-mode
accumulation semantics as the proved tape model.
-/
theorem backwardDenseFrom_compileAux_eq_backpropAllCtx {α : Type} {Δ : Type} [DecidableEq Shape]
  [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d0 : Δ)
    (seed : TList α (Γ ++ ss)) :
    Runtime.Autograd.Tape.backwardDenseFrom (t := (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss)
      g x d0).1)
        (grads0 := TList.toAnyArray (α := α) (ss := Γ ++ ss) seed) =
      .ok (TList.toAnyArray (α := α) (ss := Γ ++ ss)
        (backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0 seed)) := by
  induction g with
  | nil =>
      -- Only leaf nodes; `backwardDenseFromLoop` does nothing because every leaf's `backward` is
      -- `[]`.
      -- (We still need to show the dense array length check passes and every per-node shape check
      -- passes.)
      have hnsize :
          Γ.length =
            (addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x).nodes.size
              := by
        simp [size_addLeaves, Runtime.Autograd.Tape.empty]

      -- Main loop fact for leaf tapes.
      have hloop :
          Runtime.Autograd.Tape.backwardDenseFromLoop
              (t := addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x)
              (addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x).nodes.size
              (TList.toAnyArray (α := α) (ss := Γ) (TList.cast (α := α) (h := (List.append_nil Γ))
                seed)) =
            Except.ok
              (TList.toAnyArray (α := α) (ss := Γ)
                (TList.cast (α := α) (h := (List.append_nil Γ)) seed)) := by
        -- Put the tape in a convenient form: it's exactly the `leafNodeOfAny` image of
        -- `x.toAnyArray`.
        let t :=
          addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x
        have hnodes :
            t.nodes =
              (TList.toAnyArray (α := α) (ss := Γ) x).map (leafNodeOfAny (α := α)) := by
          -- `nodes_addLeaves` for an empty prefix tape.
          simp [t, nodes_addLeaves, Runtime.Autograd.Tape.empty]

        -- The loop just runs identity steps in reverse order.
        let seedArr :=
          TList.toAnyArray (α := α) (ss := Γ) (TList.cast (α := α) (h := List.append_nil Γ) seed)
        have htlen : t.nodes.size = Γ.length := by
          -- `t` is `addLeaves empty x`
          simpa [t] using hnsize.symm

        -- A small lemma: for any `n ≤ t.size`, the loop is the identity on the dense array.
        have loop_id :
            ∀ n, n ≤ t.nodes.size →
              Runtime.Autograd.Tape.backwardDenseFromLoop (t := t) n seedArr = Except.ok seedArr :=
                by
          intro n hnle
          induction n with
          | zero =>
              rfl
          | succ n ihn =>
              have hnlt : n < t.nodes.size :=
                Nat.lt_of_lt_of_le (Nat.lt_succ_self n) hnle
              have hnle' : n ≤ t.nodes.size :=
                Nat.le_trans (Nat.le_succ n) hnle
              have hidSeed : n < seedArr.size := by
                have : n < Γ.length := by simpa [htlen] using hnlt
                simpa [seedArr, TList.size_toAnyArray] using this
              have hidX : n < (TList.toAnyArray (α := α) (ss := Γ) x).size := by
                have : n < Γ.length := by simpa [htlen] using hnlt
                simpa [TList.size_toAnyArray] using this

              -- Identify the node at `n`: it is a leaf node.
              have hnode :
                  t.getNode? n =
                    some
                      (leafNodeOfAny (α := α)
                        ((TList.toAnyArray (α := α) (ss := Γ) x)[n]'hidX)) := by
                simp [Runtime.Autograd.Tape.getNode?, hnodes, Array.getElem?_map, leafNodeOfAny,
                  Array.getElem?_eq_getElem (xs := TList.toAnyArray (α := α) (ss := Γ) x) (i := n)
                    hidX]

              -- Shape check at `n`: both entries have the same shape.
              have hshape :
                  (seedArr[n]'hidSeed).s =
                    ((TList.toAnyArray (α := α) (ss := Γ) x)[n]'hidX).s := by
                let i : Fin Γ.length := ⟨n, by
                  have : n < Γ.length := by simpa [htlen] using hnlt
                  exact this⟩
                have hx_s :
                    ((TList.toAnyArray (α := α) (ss := Γ) x)[n]'hidX).s = Γ.get i := by
                  simpa [i] using congrArg Runtime.AnyTensor.s (TList.get_toAnyArray (α := α) (ss :=
                    Γ) x i)
                have hseed_s :
                    (seedArr[n]'hidSeed).s = Γ.get i := by
                  simpa [seedArr, i] using congrArg Runtime.AnyTensor.s
                    (TList.get_toAnyArray (α := α) (ss := Γ)
                      (TList.cast (α := α) (h := List.append_nil Γ) seed) i)
                exact hseed_s.trans hx_s.symm

              have hstepn :
                  Runtime.Autograd.Tape.backwardDenseFromStep (t := t) seedArr n = Except.ok seedArr
                    := by
                have hidSeed0 : n < (TList.toAnyArray (α := α) (ss := Γ ++ []) seed).size := by
                  have : n < Γ.length := by simpa [htlen] using hnlt
                  simpa [TList.size_toAnyArray] using this
                -- unfold `backwardDenseFromStep`; `leafNodeOfAny.backward = []`, so the step is the
                -- identity.
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, leafNodeOfAny, seedArr,
                  Array.getElem?_eq_getElem (xs := (TList.toAnyArray (α := α) (ss := Γ ++ []) seed))
                    (i := n) hidSeed0]
                have hcond : seed.toAnyArray[n].s = x.toAnyArray[n].s := by
                  simpa [seedArr] using hshape
                -- `leafNodeOfAny.backward` contributes no parent gradients, so the fold is a no-op.
                -- After choosing the `if` branch via `hcond`, this is definitional.
                simp [hcond]
                rfl

              -- Unfold one loop iteration and use the IH.
              simpa [Runtime.Autograd.Tape.backwardDenseFromLoop, hstepn] using (ihn hnle')

        -- Specialize to `n = t.size`.
        simpa [t, seedArr] using loop_id t.nodes.size (le_rfl)

      -- Put it all together.
      -- `backpropAllCtx` is the identity in the nil case.
      -- Use the size check (`hnsize`) and rewrite away the `cast` on the seed array.
      simpa [compileAux, backpropAllCtx, Runtime.Autograd.Tape.backwardDenseFrom, hnsize,
        TList.toAnyArray_cast] using hloop
  | snoc g node ih =>
      rename_i ssPrev τ
      -- Unpack the compilation of the prefix graph.
      rcases hprev : compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 with ⟨tPrev,
        ctxPrev⟩
      have hctxPrev :
          ctxPrev = Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 := by
        simpa [hprev] using
          (compileAux_ctx_eq_eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)
      have htPrevSize : tPrev.nodes.size = Γ.length + ssPrev.length := by
        simpa [hprev] using
          (compileAux_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)

      -- Unpack the seed into `(seedPrev, seedOut)` matching the snoc structure.
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) := TList.cast (α := α) (h := assoc.symm) seed
      let seedPrev : TList α (Γ ++ ssPrev) :=
        (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').1
      let seedOut : Tensor α τ :=
        (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      have hseed' :
          TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedPrev seedOut = seed' := by
        simpa [seedPrev, seedOut] using
          (TList.snoc_unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) (xs := seed'))

      -- The output gradient seed is the last entry of the dense array.
      let outAny : Runtime.AnyTensor α := Runtime.Autograd.AnyTensor.mk seedOut

      -- Define the runtime tape for the snoc graph explicitly.
      let y := node.forward ctxPrev d0
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-compiled"
          value := Runtime.Autograd.AnyTensor.mk y
          requires_grad := true
          parents := []
          backward := fun dLdyAny => by
            if h : dLdyAny.s = τ then
              let dLdy : Tensor α τ := Tensor.castShape dLdyAny.t h
              let contribs := node.vjp ctxPrev d0 dLdy
              exact .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      let tNext : Runtime.Autograd.Tape α := (Runtime.Autograd.Tape.addNode (t := tPrev)
        runtimeNode).1
      have htNextNodes :
          tNext.nodes = tPrev.nodes.push runtimeNode := by
        simp [tNext, Runtime.Autograd.Tape.addNode]
      have htNextSize :
          tNext.nodes.size = tPrev.nodes.size + 1 := by
        simp [htNextNodes]

      -- Rewrite the initial gradient array as a push of the prefix part plus `seedOut`.
      have hseedArr :
          TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed =
            (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev).push outAny := by
        -- First rewrite `seed` to the assoc-cast form `seed'`, then use `toAnyArray_snoc`.
        have hcast :
            TList.toAnyArray (α := α) (ss := (Γ ++ ssPrev) ++ [τ]) seed' =
              TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed := by
          simp [seed']
        -- Replace the LHS by `seed'`, then rewrite `seed'` as a `snoc`.
        rw [← hcast]
        -- `seed' = snoc seedPrev seedOut`
        have : seed' = TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedPrev seedOut := by
          simpa using hseed'.symm
        -- Now `toAnyArray_snoc` gives the pushed array.
        simp [this, outAny, TList.toAnyArray_snoc]

      -- From here on we unfold `backwardDenseFrom` into the loop+step decomposition.
      -- `backpropAllCtx` peels off the last seed and recurses on the prefix graph.
      have hsizeCheck :
          (TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed).size = tNext.nodes.size :=
            by
        -- LHS: `(Γ ++ ssPrev ++ [τ]).length`, RHS: `tPrev.size + 1`.
        simp [hseedArr, htNextSize, htPrevSize, TList.size_toAnyArray, Nat.add_assoc]

      -- Expand both sides (`compileAux`, `backpropAllCtx`, and `backwardDenseFrom`) and reduce to:
      -- 1) one runtime step for the last node (adding `vjp` contributions to the prefix),
      -- 2) then the IH on the prefix graph.
      let ctx := Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      let contrib := node.vjp ctx d0 seedOut
      let seedPrev' := TList.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev := backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 seedPrev'
      -- Now compute the runtime `backwardDenseFrom` explicitly:
      -- 1) run one step for the last node, adding `contrib` into the prefix grads,
      -- 2) run the prefix loop, which matches the IH on `g`,
      -- 3) the last gradient entry `seedOut` is never modified afterwards.

      -- Normalize `compileAux` and `backpropAllCtx` to our explicit `tNext`/`seedPrev`
      -- decomposition.
      have hTape :
          (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ]) (.snoc (ss := ssPrev) (τ :=
            τ) g node) x d0).1 =
            tNext := by
        simp [compileAux, hprev, tNext, y, runtimeNode]

      have hBackpropArr :
          TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ]))
              (backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ])
                (.snoc (ss := ssPrev) (τ := τ) g node) x d0 seed) =
            (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) gradsPrev).push outAny := by
        -- unfold `backpropAllCtx` at the snoc constructor and use `toAnyArray_snoc`
        simp [backpropAllCtx, seed', seedPrev, seedOut, ctx, contrib, seedPrev', gradsPrev, outAny,
          TList.toAnyArray_cast, TList.toAnyArray_snoc]

      -- Reduce the main goal to the loop over `tNext`.
      -- After rewriting, the goal is:
      -- `tNext.backwardDenseFrom seedArr0 = .ok (gradsPrevArr.push outAny)`.
      -- The size check passes by `hsizeCheck`.
      have hmain :
          Runtime.Autograd.Tape.backwardDenseFrom (t := tNext)
              (grads0 := TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed) =
            .ok ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) gradsPrev).push outAny) := by
        -- Rewrite the initial array into the pushed form.
        -- Also rewrite `backwardDenseFrom` to its core loop using the size check.
        simp [Runtime.Autograd.Tape.backwardDenseFrom, hseedArr, htNextSize, htPrevSize]
        -- Remaining proof: compute the loop body.
        -- Set up convenient shorthands for the prefix size and gradient arrays.
        let n : Nat := tPrev.nodes.size
        let seedPrevArr : Array (Runtime.AnyTensor α) :=
          TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev
        let seedPrevArr' : Array (Runtime.AnyTensor α) :=
          TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev'
        let gradsPrevArr : Array (Runtime.AnyTensor α) :=
          TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) gradsPrev

        -- A small helper: the prefix gradient array has size `n`.
        have hsizeSeedPrevArr : seedPrevArr.size = n := by
          simp [seedPrevArr, n, TList.size_toAnyArray, htPrevSize, List.length_append]

        -- First, compute the last-node step (`id = n`), which just adds `contrib` into the prefix
        -- grads.
        have hnodeLast : Runtime.Autograd.Tape.getNode? (t := tNext) n = some runtimeNode := by
          -- `tNext.nodes = tPrev.nodes.push runtimeNode`
          have : tNext.nodes[n]? = some runtimeNode := by
            simp [htNextNodes, n]
          simpa [Runtime.Autograd.Tape.getNode?, tNext] using this

        have haccLast : (seedPrevArr.push outAny)[n]? = some outAny := by
          have : (seedPrevArr.push outAny)[seedPrevArr.size]? = some outAny := by
            simp
          simpa [hsizeSeedPrevArr] using this

        have hshapeLast : outAny.s = runtimeNode.value.s := by
          rfl

        -- Show the `addGradAll` fold for the last node matches `TList.add` on the prefix, leaving
        -- `[outAny]` untouched.
        have hreqAll :
            ∀ i (hi : i < tPrev.nodes.size), (tPrev.nodes[i]'hi).requires_grad = true := by
          -- Unfold the `let t := ...` binder in `compileAux_requires_grad_true` and rewrite by
          -- `hprev`.
          have hreq0 :
              ∀ i (hi : i < (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x
                d0).1.nodes.size),
                (((compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x
                  d0).1.nodes[i]'hi).requires_grad = true) := by
            simpa using (compileAux_requires_grad_true (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x
              d0)
          simpa [hprev] using hreq0

        have hnodes0 :
            ∀ i (hi : i < (Γ ++ ssPrev).length),
              let id := (0 : Nat) + i
              ∃ node : Runtime.Autograd.Node α,
                tNext.getNode? id = some node ∧ node.requires_grad = true ∧
                  node.value.s =
                    ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                        simpa [TList.size_toAnyArray] using hi)).s := by
          intro i hi
          have hiT : i < tPrev.nodes.size := by
            -- `tPrev.nodes.size = (Γ ++ ssPrev).length`
            simpa [htPrevSize, List.length_append, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
              using hi
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[i]'hiT
          have hgetPrev : tPrev.getNode? i = some nodeAt := by
            simp [Runtime.Autograd.Tape.getNode?, nodeAt, Array.getElem?_eq_getElem (xs :=
              tPrev.nodes) (i := i) hiT]
          have hgetNext : tNext.getNode? i = some nodeAt := by
            -- index `< tPrev.nodes.size`, so `push` doesn't change it
            have : (tPrev.nodes.push runtimeNode)[i]? = some (tPrev.nodes[i]'hiT) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hiT)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this
          have hreq : nodeAt.requires_grad = true := by
            simpa [nodeAt] using hreqAll i hiT
          -- Shapes: both are the `i`th shape in `Γ ++ ssPrev`.
          have hseedShape :
              nodeAt.value.s =
                ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                    simpa [TList.size_toAnyArray] using hi)).s := by
            let fi : Fin (Γ ++ ssPrev).length := ⟨i, hi⟩
            -- `tPrev.nodes.map value = ctxPrev.toAnyArray`
            have hvals :
                tPrev.nodes.map (fun nd => nd.value) =
                  TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev := by
              simpa [hprev] using
                (compileAux_values_eq (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)
            have hvalOpt := congrArg (fun a => a[i]?) hvals
            -- Evaluate both sides at `i`.
            have hnodeVal :
                nodeAt.value = (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                  -- `i < ctxPrev.toAnyArray.size` because it matches `tPrev.nodes.size`
                  simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc] using
                    hiT) := by
              -- Left: map+index gives `some nodeAt.value`
              have hleft :
                  (tPrev.nodes.map (fun nd => nd.value))[i]? = some nodeAt.value := by
                have : tPrev.nodes[i]? = some nodeAt := by
                  simp [nodeAt, Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := i) hiT]
                simp [Array.getElem?_map, this, nodeAt]
              -- Right: in-bounds `getElem?` is `some _`
              have hright :
                  (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]? =
                    some ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                      simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc]
                        using hiT)) := by
                have hiCtx :
                    i < (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev).size := by
                  simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc] using
                    hiT
                simp [Array.getElem?_eq_getElem (xs := (TList.toAnyArray (α := α) (ss := Γ ++
                  ssPrev) ctxPrev)) (i := i) hiCtx]
              -- Combine and extract the value equality.
              have : some nodeAt.value =
                  some ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                    simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc]
                      using hiT)) := by
                -- rewrite both sides of `hvalOpt` using `hleft`/`hright`
                simpa [hleft, hright] using hvalOpt
              simpa using congrArg (fun o => o.getD nodeAt.value) this
            have hnode_s :
                nodeAt.value.s = (Γ ++ ssPrev).get fi := by
              have hiCtx :
                  i < (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev).size := by
                simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc] using
                  hiT
              have hctx_s :
                  ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'hiCtx).s =
                    (Γ ++ ssPrev).get fi := by
                -- `ctxPrev.get fi : Tensor α ((Γ ++ ssPrev).get fi)`, so the RHS shape is
                -- definitional.
                simpa [fi] using
                  congrArg Runtime.AnyTensor.s
                    (TList.get_toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev fi)
              -- rewrite the LHS using `hnodeVal`
              simpa [hnodeVal] using hctx_s
            have hseed_s :
                ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                    simpa [TList.size_toAnyArray] using hi)).s = (Γ ++ ssPrev).get fi := by
              simpa [fi] using congrArg Runtime.AnyTensor.s
                (TList.get_toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev fi)
            exact hnode_s.trans hseed_s.symm

          refine ⟨nodeAt, ?_, ?_, ?_⟩
          · simpa [Nat.zero_add] using hgetNext
          · exact hreq
          · simpa using hseedShape

        have hfoldLast :
              (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0).foldlM
                  (fun acc2 (pid, pg) =>
                    Runtime.Autograd.Tape.addGradAll (t := tNext) acc2 pid pg)
                  (seedPrevArr.push outAny) =
                .ok (seedPrevArr'.push outAny) := by
            -- Apply the generic fold lemma with `pref = #[]` and `suffix = #[outAny]`.
            have hfold :=
              foldlM_addGradAll_toIndexedAnyList_eq_add (α := α) (t := tNext)
                (pref := (#[] : Array (Runtime.AnyTensor α)))
                (seed := seedPrev) (contrib := contrib) (suffix := #[outAny]) hnodes0
            -- Simplify the array concatenations and rewrite `seedPrev'`.
            simpa [seedPrevArr, seedPrevArr', seedPrev', Array.append_assoc, Array.append_empty,
              Array.empty_append, Array.append_singleton] using hfold

        have hstepLast :
              Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (seedPrevArr.push outAny) n =
                .ok (seedPrevArr'.push outAny) := by
            -- Unfold the step: pick out the last node, check shapes, run `backward`, then fold
            -- `addGradAll`.
            -- Eliminate the shape equality so all casts become definitional.
            cases hshapeLast
            have hreqLast : runtimeNode.requires_grad = true := by
              rfl
            have hout : outAny.s = τ := by
              rfl
            have hshapeNode : outAny.s = runtimeNode.value.s := by
              rfl
            have hbackLast :
                runtimeNode.backward
                    { s := runtimeNode.value.s
                      t := Tensor.castShape outAny.t hshapeNode } =
                  .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0) := by
              simp [runtimeNode, outAny, hctxPrev, ctx, contrib, Tensor.castShape,
                Runtime.Autograd.AnyTensor.mk]
            -- Unfold the step and reduce the control flow (`getNode?`, `requires_grad`, `acc[id]?`,
            -- shape check).
            simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodeLast, hreqLast, haccLast]
            -- Select the shape-check branch, but keep `Tensor.cast_shape` folded so we can rewrite
            -- via `hbackLast`.
            simp [hshapeNode]
            -- Rewrite the `backward` call to its concrete list of contributions.
            rw [hbackLast]
            -- The remaining `foldlM` is exactly `hfoldLast`.
            simpa [seedPrevArr, seedPrevArr', outAny] using hfoldLast

        -- Run the remaining prefix loop (ids `< n`). This matches the IH on `g` and leaves `outAny`
        -- untouched.
        have ihPrev :
            Runtime.Autograd.Tape.backwardDenseFrom (t := tPrev) (grads0 := seedPrevArr') =
              .ok gradsPrevArr := by
          simpa [hprev, seedPrevArr', gradsPrevArr, gradsPrev] using (ih seedPrev')

        have hsizeSeedPrevArr' : seedPrevArr'.size = n := by
          simp [seedPrevArr', n, TList.size_toAnyArray, htPrevSize, List.length_append]

        have ihPrevLoop :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tPrev) n seedPrevArr' =
              .ok gradsPrevArr := by
          have hsize : seedPrevArr'.size = tPrev.nodes.size := by
            simpa [n] using hsizeSeedPrevArr'
          simpa [Runtime.Autograd.Tape.backwardDenseFrom, hsize, n] using ihPrev

        -- Helper: `addGradAll` commutes with pushing an unused last slot.
        have haddGradAllPush :
            ∀ (acc : Array (Runtime.AnyTensor α)) (hacc : acc.size = n)
              (pid : Nat) (pg : Runtime.AnyTensor α),
              pid < n →
              Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outAny) pid pg =
                Except.map (fun a => a.push outAny)
                  (Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg) := by
          intro acc hacc pid pg hpid
          have hpidPrev : pid < tPrev.nodes.size := by
            simpa [n] using hpid
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[pid]'hpidPrev
          have hnodePrev : Runtime.Autograd.Tape.getNode? (t := tPrev) pid = some nodeAt := by
            simp [Runtime.Autograd.Tape.getNode?, nodeAt,
              Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := pid) hpidPrev]
          have hnodeNext : Runtime.Autograd.Tape.getNode? (t := tNext) pid = some nodeAt := by
            have : (tPrev.nodes.push runtimeNode)[pid]? = some (tPrev.nodes[pid]'hpidPrev) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hpidPrev)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this

          have hpidAcc : pid < acc.size := by
            simpa [hacc] using hpid
          let existing : Runtime.AnyTensor α := acc[pid]'hpidAcc
          have hgetPrev : acc[pid]? = some existing := by
            simp [existing]
          have hgetNext : (acc.push outAny)[pid]? = some existing := by
            simpa [existing] using (Array.getElem?_push_lt (xs := acc) (x := outAny) hpidAcc)

          cases hreq : nodeAt.requires_grad with
          | false =>
              -- If the node does not require grad, `addGradAll` is the identity.
              have hlhs :
                  Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outAny) pid pg =
                    .ok (acc.push outAny) := by
                simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, throw, throwThe,
                  MonadExceptOf.throw]
                rfl
              have hrhs :
                  Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg = .ok acc :=
                    by
                simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, throw, throwThe,
                  MonadExceptOf.throw]
                rfl
              simp [hlhs, hrhs, Except.map]
          | true =>
              by_cases hshape : pg.s = nodeAt.value.s
              · by_cases hex : existing.s = nodeAt.value.s
                ·
                    let pg' : Runtime.AnyTensor α :=
                      { s := nodeAt.value.s, t := Tensor.castShape pg.t hshape }
                    let existing' : Runtime.AnyTensor α :=
                      { s := nodeAt.value.s, t := Tensor.castShape existing.t hex }
                    have hidPrev : pid < acc.size := hpidAcc
                    have hidNext : pid < (acc.push outAny).size := by
                      simpa [Array.size_push] using Nat.lt_trans hidPrev (Nat.lt_succ_self acc.size)
                    cases hadd : Runtime.Autograd.AnyTensor.add existing' pg' with
                    | error e =>
                        have hprevRes :
                            Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg =
                              .error e := by
                          simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, hshape, hgetPrev,
                            hex, pg',
                            existing', hadd, throw, throwThe, MonadExceptOf.throw]
                          rfl
                        have hnextRes :
                            Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outAny)
                              pid pg =
                              .error e := by
                          simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, hshape, hgetNext,
                            hex, pg',
                            existing', hadd, throw, throwThe, MonadExceptOf.throw]
                          rfl
                        simp [hprevRes, hnextRes, Except.map]
                    | ok summed =>
                        have hpid_lt_succ : pid < acc.size + 1 :=
                          Nat.lt_trans hidPrev (Nat.lt_succ_self acc.size)
                        have hprevRes :
                            Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg =
                              .ok (acc.set pid summed (h := hidPrev)) := by
                          have hcond : acc[pid].s = nodeAt.value.s := by
                            simpa [existing] using hex
                          have haddAcc :
                              Runtime.Autograd.AnyTensor.add
                                  { s := nodeAt.value.s, t := Tensor.castShape acc[pid].t hcond }
                                  { s := nodeAt.value.s, t := Tensor.castShape pg.t hshape } =
                                .ok summed := by
                            simpa [pg', existing', existing, hcond] using hadd
                          simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, hshape, hcond,
                            haddAcc, hidPrev,
                            throw, throwThe, MonadExceptOf.throw]
                        have hnextRes :
                            Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outAny)
                              pid pg =
                              .ok ((acc.set pid summed (h := hidPrev)).push outAny) := by
                          have hget : (acc.push outAny)[pid] = acc[pid] := by
                            simpa using
                              (Array.getElem_push_lt (xs := acc) (x := outAny) (i := pid) hidPrev)
                          have hcond : acc[pid].s = nodeAt.value.s := by
                            simpa [existing] using hex
                          have haddAcc :
                              Runtime.Autograd.AnyTensor.add
                                  { s := nodeAt.value.s, t := Tensor.castShape acc[pid].t hcond }
                                  { s := nodeAt.value.s, t := Tensor.castShape pg.t hshape } =
                                .ok summed := by
                            simpa [pg', existing', existing, hcond] using hadd
                          simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, hshape,
                            haddAcc, hpid_lt_succ, Array.set_push, hidPrev, hget, hcond, throw,
                              throwThe,
                            MonadExceptOf.throw]
                        simp [hprevRes, hnextRes, Except.map]
                · simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hnodeNext, hreq, hshape,
                  hgetPrev, hgetNext, hex,
                    Except.map, throw, throwThe, MonadExceptOf.throw]
              ·
                  have hprevRes :
                      Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg =
                        .error "autograd: gradient contribution has wrong shape for parent" := by
                    simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, hshape, throw,
                      throwThe,
                      MonadExceptOf.throw]
                  have hnextRes :
                      Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outAny) pid
                        pg =
                        .error "autograd: gradient contribution has wrong shape for parent" := by
                    simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, hshape, throw,
                      throwThe,
                      MonadExceptOf.throw]
                  simp [hprevRes, hnextRes, Except.map]

        -- Helper: `backwardDenseFromStep` commutes with pushing an unused last slot for ids `< n`.
        have hstepPush :
            ∀ (id : Nat) (hid : id < n) (acc : Array (Runtime.AnyTensor α)),
              acc.size = n →
              Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (acc.push outAny) id =
                Except.map (fun a => a.push outAny)
                  (Runtime.Autograd.Tape.backwardDenseFromStep (t := tPrev) acc id) := by
          intro id hid acc hacc
          have hidPrev : id < tPrev.nodes.size := by
            simpa [n] using hid
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[id]'hidPrev
          have hnodePrev : Runtime.Autograd.Tape.getNode? (t := tPrev) id = some nodeAt := by
            simp [Runtime.Autograd.Tape.getNode?, nodeAt,
              Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := id) hidPrev]
          have hnodeNext : Runtime.Autograd.Tape.getNode? (t := tNext) id = some nodeAt := by
            have : (tPrev.nodes.push runtimeNode)[id]? = some (tPrev.nodes[id]'hidPrev) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hidPrev)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this
          have hidAcc : id < acc.size := by
            simpa [hacc] using hid
          have hgetAcc : acc[id]? = some (acc[id]'hidAcc) := by
            simp
          have hgetAccPush : (acc.push outAny)[id]? = some (acc[id]'hidAcc) := by
            simpa using (Array.getElem?_push_lt (xs := acc) (x := outAny) hidAcc)
          cases hreq : nodeAt.requires_grad with
          | false =>
              have hlhs :
                  Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (acc.push outAny) id =
                    .ok (acc.push outAny) := by
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodeNext, hreq, throw, throwThe,
                  MonadExceptOf.throw]
                rfl
              have hrhs :
                  Runtime.Autograd.Tape.backwardDenseFromStep (t := tPrev) acc id = .ok acc := by
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hreq, throw, throwThe,
                  MonadExceptOf.throw]
                rfl
              simp [hlhs, hrhs, Except.map]
          | true =>
              by_cases hshape : (acc[id]'hidAcc).s = nodeAt.value.s
              · -- shape ok, split on `backward` result
                let dLdy : Runtime.AnyTensor α :=
                  { s := nodeAt.value.s, t := Tensor.castShape (acc[id]'hidAcc).t hshape }
                cases hback : nodeAt.backward dLdy with
                | error e =>
                    simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                      hgetAcc, hgetAccPush,
                      hshape, dLdy, hback, Except.map, throw, throwThe, MonadExceptOf.throw]
                    rfl
                | ok contribs =>
                    have hpids :
                          ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ contribs → pid < id
                            := by
                        intro pid pg hmem
                        have hgetComp :
                            Runtime.Autograd.Tape.getNode? (t := (compileAux (α := α) (Δ := Δ) (Γ :=
                              Γ) (ss := ssPrev) g x d0).1)
                                id =
                              some nodeAt := by
                          simpa [hprev] using hnodePrev
                        exact
                          compileAux_backward_pids_lt_id (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g
                            x d0 id nodeAt hgetComp dLdy
                              contribs hback hmem

                    have hfoldAux :
                        ∀ (cs : List (Nat × Runtime.AnyTensor α)) (acc0 accOut : Array
                          (Runtime.AnyTensor α)),
                          acc0.size = n →
                          (∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ cs → pid < n) →
                          cs.foldlM (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t :=
                            tNext) acc2 pid pg)
                              (acc0.push outAny) =
                            Except.map (fun a => a.push outAny)
                              (cs.foldlM
                                (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := tPrev)
                                  acc2 pid pg) acc0) := by
                      intro cs
                      induction cs with
                      | nil =>
                          intro acc0 accOut _hsize _hpids
                          simp [List.foldlM, Except.map]
                          rfl
                      | cons hd tl ih =>
                          intro acc0 accOut hsize hpids
                          rcases hd with ⟨pid, pg⟩
                          have hpid : pid < n := by
                            exact hpids (pid := pid) (pg := pg) (by simp)
                          have hadd :=
                            haddGradAllPush (acc := acc0) (hacc := hsize) (pid := pid) (pg := pg)
                              hpid
                          cases hret : Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc0)
                            pid pg with
                          | error e =>
                              -- both folds error at the first step
                              simp [List.foldlM, hret, hadd, Except.map]
                              rfl
                          | ok acc1 =>
                              have hret' :
                                  Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc0.push
                                    outAny) pid pg =
                                    .ok (acc1.push outAny) := by
                                -- unfold `Except.map` in `hadd`
                                simpa [Except.map, hret] using hadd
                              have hsize1 : acc1.size = n := by
                                have := addGradAll_ok_size (t := tPrev) (grads := acc0) (id := pid)
                                  (g := pg)
                                  (grads' := acc1) (by simpa using hret)
                                simpa [hsize] using this
                              have hpids_tl :
                                  ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ tl → pid < n
                                    := by
                                intro pid pg hmem
                                exact hpids (pid := pid) (pg := pg) (by simp [hmem])
                              have ih' :=
                                ih (acc0 := acc1) (accOut := accOut) hsize1 hpids_tl
                              -- unfold the `foldlM` for the cons case on both sides
                              simpa [List.foldlM, hret, hret', Except.map] using ih'

                    have hpids_n :
                        ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ contribs → pid < n :=
                          by
                      intro pid pg hmem
                      exact Nat.lt_trans (hpids (pid := pid) (pg := pg) hmem) hid

                    -- Apply the fold lemma.
                    have hfold :=
                      hfoldAux contribs acc acc hacc hpids_n
                    -- Unfold the step and rewrite using the fold lemma.
                    simpa [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                      hgetAcc, hgetAccPush,
                      hshape, dLdy, hback, Except.map, throw, throwThe, MonadExceptOf.throw] using
                        hfold
              · -- shape mismatch
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                  hgetAcc, hgetAccPush, hshape,
                  Except.map, throw, throwThe, MonadExceptOf.throw]

        -- The loop itself commutes with pushing an unused last slot.
        have hloopPush :
              ∀ m (hm : m ≤ n) (acc : Array (Runtime.AnyTensor α)),
                acc.size = n →
                Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) m (acc.push outAny) =
                  Except.map (fun a => a.push outAny)
                    (Runtime.Autograd.Tape.backwardDenseFromLoop (t := tPrev) m acc) := by
            intro m hm acc hacc
            induction m generalizing acc with
            | zero =>
                simp [Runtime.Autograd.Tape.backwardDenseFromLoop, Except.map]
                rfl
            | succ m ihm =>
                have hm' : m ≤ n := Nat.le_trans (Nat.le_succ m) hm
                have hmid : m < n := Nat.lt_of_lt_of_le (Nat.lt_succ_self m) hm
                have hstep := hstepPush (id := m) (hid := hmid) (acc := acc) hacc
                cases hret : Runtime.Autograd.Tape.backwardDenseFromStep (t := tPrev) acc m with
                | error e =>
                    -- both loops error on this step
                    simp [Runtime.Autograd.Tape.backwardDenseFromLoop, hret, hstep, Except.map]
                    rfl
                | ok acc1 =>
                    have hstep' :
                        Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (acc.push outAny) m
                          =
                          .ok (acc1.push outAny) := by
                      simpa [Except.map, hret] using hstep
                    have hsize1 : acc1.size = n := by
                      have := backwardDenseFromStep_ok_size (t := tPrev) (acc := acc) (id := m)
                        (acc' := acc1)
                        (by simpa using hret)
                      simpa [hacc] using this
                    have ih' := ihm (acc := acc1) hm' hsize1
                    simpa [Runtime.Autograd.Tape.backwardDenseFromLoop, hret, hstep', Except.map]
                      using ih'

        have hloopFinal :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) n (seedPrevArr'.push outAny) =
              .ok (gradsPrevArr.push outAny) := by
          have h := hloopPush n (le_rfl) seedPrevArr' hsizeSeedPrevArr'
          simpa [ihPrevLoop, Except.map] using h

        -- Use `hstepLast` to reduce the initial step, then apply `hloopFinal`.
        have hloopAll :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) (n + 1) (seedPrevArr.push
              outAny) =
              .ok (gradsPrevArr.push outAny) := by
          -- Unfold the loop one step, then rewrite the step result via `hstepLast`.
          -- The remaining goal is exactly `hloopFinal`.
          simp [Runtime.Autograd.Tape.backwardDenseFromLoop, hstepLast]
          simpa using hloopFinal
        simpa [n, htPrevSize, Nat.add_assoc] using hloopAll

      -- Finish: rewrite both sides back to the original statement.
      simpa [hTape, hBackpropArr] using hmain

/--
Variant of `backwardDenseFrom_compileAux_eq_backpropAllCtx` for the `GraphData` interface.

This is useful when a graph carries extra payload `Δ` (e.g. parameters/config) through forward and
backward closures.
-/
theorem backwardDenseFrom_compileAuxData_eq_backpropAllCtx {α : Type} {Δ : Type} [DecidableEq Shape]
  [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d0 : Δ)
    (seed : TList α (Γ ++ ss)) :
    Runtime.Autograd.Tape.backwardDenseFrom (t := (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss :=
      ss) g x d0).1)
        (grads0 := TList.toAnyArray (α := α) (ss := Γ ++ ss) seed) =
      .ok
        (TList.toAnyArray (α := α) (ss := Γ ++ ss)
          (_root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss :=
            ss) g x d0 seed)) := by
  induction g with
  | nil =>
      -- Only leaf nodes; `backwardDenseFromLoop` does nothing because every leaf's `backward` is
      -- `[]`.
      have hnsize :
          Γ.length =
            (addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x).nodes.size
              := by
        simp [size_addLeaves, Runtime.Autograd.Tape.empty]

      have hloop :
          Runtime.Autograd.Tape.backwardDenseFromLoop
              (t := addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x)
              (addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x).nodes.size
              (TList.toAnyArray (α := α) (ss := Γ) (TList.cast (α := α) (h := (List.append_nil Γ))
                seed)) =
            Except.ok
              (TList.toAnyArray (α := α) (ss := Γ)
                (TList.cast (α := α) (h := (List.append_nil Γ)) seed)) := by
        let t :=
          addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x
        have hnodes :
            t.nodes =
              (TList.toAnyArray (α := α) (ss := Γ) x).map (leafNodeOfAny (α := α)) := by
          simp [t, nodes_addLeaves, Runtime.Autograd.Tape.empty]

        let seedArr :=
          TList.toAnyArray (α := α) (ss := Γ) (TList.cast (α := α) (h := List.append_nil Γ) seed)
        have htlen : t.nodes.size = Γ.length := by
          simpa [t] using hnsize.symm

        have loop_id :
            ∀ n, n ≤ t.nodes.size →
              Runtime.Autograd.Tape.backwardDenseFromLoop (t := t) n seedArr = Except.ok seedArr :=
                by
          intro n hnle
          induction n with
          | zero =>
              rfl
          | succ n ihn =>
              have hnlt : n < t.nodes.size :=
                Nat.lt_of_lt_of_le (Nat.lt_succ_self n) hnle
              have hnle' : n ≤ t.nodes.size :=
                Nat.le_trans (Nat.le_succ n) hnle
              have hidSeed : n < seedArr.size := by
                have : n < Γ.length := by simpa [htlen] using hnlt
                simpa [seedArr, TList.size_toAnyArray] using this
              have hidX : n < (TList.toAnyArray (α := α) (ss := Γ) x).size := by
                have : n < Γ.length := by simpa [htlen] using hnlt
                simpa [TList.size_toAnyArray] using this

              have hnode :
                  t.getNode? n =
                    some
                      (leafNodeOfAny (α := α)
                        ((TList.toAnyArray (α := α) (ss := Γ) x)[n]'hidX)) := by
                simp [Runtime.Autograd.Tape.getNode?, hnodes, Array.getElem?_map, leafNodeOfAny,
                  Array.getElem?_eq_getElem (xs := TList.toAnyArray (α := α) (ss := Γ) x) (i := n)
                    hidX]

              have hshape :
                  (seedArr[n]'hidSeed).s =
                    ((TList.toAnyArray (α := α) (ss := Γ) x)[n]'hidX).s := by
                let i : Fin Γ.length := ⟨n, by
                  have : n < Γ.length := by simpa [htlen] using hnlt
                  exact this⟩
                have hx_s :
                    ((TList.toAnyArray (α := α) (ss := Γ) x)[n]'hidX).s = Γ.get i := by
                  simpa [i] using congrArg Runtime.AnyTensor.s (TList.get_toAnyArray (α := α) (ss :=
                    Γ) x i)
                have hseed_s :
                    (seedArr[n]'hidSeed).s = Γ.get i := by
                  simpa [seedArr, i] using congrArg Runtime.AnyTensor.s
                    (TList.get_toAnyArray (α := α) (ss := Γ)
                      (TList.cast (α := α) (h := List.append_nil Γ) seed) i)
                exact hseed_s.trans hx_s.symm

              have hstepn :
                  Runtime.Autograd.Tape.backwardDenseFromStep (t := t) seedArr n = Except.ok seedArr
                    := by
                have hidSeed0 : n < (TList.toAnyArray (α := α) (ss := Γ ++ []) seed).size := by
                  have : n < Γ.length := by simpa [htlen] using hnlt
                  simpa [TList.size_toAnyArray] using this
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, leafNodeOfAny, seedArr,
                  Array.getElem?_eq_getElem (xs := (TList.toAnyArray (α := α) (ss := Γ ++ []) seed))
                    (i := n) hidSeed0]
                have hcond : seed.toAnyArray[n].s = x.toAnyArray[n].s := by
                  simpa [seedArr] using hshape
                simp [hcond]
                rfl

              simpa [Runtime.Autograd.Tape.backwardDenseFromLoop, hstepn] using (ihn hnle')

        simpa [t, seedArr] using loop_id t.nodes.size (le_rfl)

      simpa [compileAuxData, _root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx,
        Runtime.Autograd.Tape.backwardDenseFrom, hnsize, TList.toAnyArray_cast] using hloop
  | snoc g node ih =>
      rename_i ssPrev τ
      rcases hprev : compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 with ⟨tPrev,
        ctxPrev⟩
      have hctxPrev :
          ctxPrev = _root_.Proofs.Autograd.Algebra.GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss :=
            ssPrev) g x d0 := by
        simpa [hprev] using
          (compileAuxData_ctx_eq_eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)
      have htPrevSize : tPrev.nodes.size = Γ.length + ssPrev.length := by
        simpa [hprev] using
          (compileAuxData_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)

      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) := TList.cast (α := α) (h := assoc.symm) seed
      let seedPrev : TList α (Γ ++ ssPrev) :=
        (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').1
      let seedOut : Tensor α τ :=
        (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      have hseed' :
          TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedPrev seedOut = seed' := by
        simpa [seedPrev, seedOut] using
          (TList.snoc_unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) (xs := seed'))

      let outAny : Runtime.AnyTensor α := Runtime.Autograd.AnyTensor.mk seedOut

      let y := node.forward ctxPrev d0
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-compiled"
          value := Runtime.Autograd.AnyTensor.mk y
          requires_grad := true
          parents := []
          backward := fun dLdyAny => by
            if h : dLdyAny.s = τ then
              let dLdy : Tensor α τ := Tensor.castShape dLdyAny.t h
              let contribs := node.vjp ctxPrev d0 dLdy
              exact .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      let tNext : Runtime.Autograd.Tape α := (Runtime.Autograd.Tape.addNode (t := tPrev)
        runtimeNode).1
      have htNextNodes :
          tNext.nodes = tPrev.nodes.push runtimeNode := by
        simp [tNext, Runtime.Autograd.Tape.addNode]
      have htNextSize :
          tNext.nodes.size = tPrev.nodes.size + 1 := by
        simp [htNextNodes]

      have hseedArr :
          TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed =
            (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev).push outAny := by
        have hcast :
            TList.toAnyArray (α := α) (ss := (Γ ++ ssPrev) ++ [τ]) seed' =
              TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed := by
          simp [seed']
        rw [← hcast]
        have : seed' = TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedPrev seedOut := by
          simpa using hseed'.symm
        simp [this, outAny, TList.toAnyArray_snoc]

      have hsizeCheck :
          (TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed).size = tNext.nodes.size :=
            by
        simp [hseedArr, htNextSize, htPrevSize, TList.size_toAnyArray, Nat.add_assoc]

      let ctx := _root_.Proofs.Autograd.Algebra.GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss :=
        ssPrev) g x d0
      let contrib := node.vjp ctx d0 seedOut
      let seedPrev' := TList.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev :=
        _root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss :=
          ssPrev) g x d0 seedPrev'

      have hTape :
          (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ]) (.snoc (ss := ssPrev) (τ
            := τ) g node) x d0).1 =
            tNext := by
        simp [compileAuxData, hprev, tNext, y, runtimeNode]

      have hBackpropArr :
          TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ]))
              (_root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ)
                (ss := ssPrev ++ [τ])
                (.snoc (ss := ssPrev) (τ := τ) g node) x d0 seed) =
            (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) gradsPrev).push outAny := by
        simp [_root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx, seed', seedPrev, seedOut,
          ctx, contrib,
          seedPrev', gradsPrev, outAny, TList.toAnyArray_cast, TList.toAnyArray_snoc]

      have hmain :
          Runtime.Autograd.Tape.backwardDenseFrom (t := tNext)
              (grads0 := TList.toAnyArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed) =
            .ok ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) gradsPrev).push outAny) := by
        simp [Runtime.Autograd.Tape.backwardDenseFrom, hseedArr, htNextSize, htPrevSize]
        let n : Nat := tPrev.nodes.size
        let seedPrevArr : Array (Runtime.AnyTensor α) :=
          TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev
        let seedPrevArr' : Array (Runtime.AnyTensor α) :=
          TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev'
        let gradsPrevArr : Array (Runtime.AnyTensor α) :=
          TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) gradsPrev

        have hsizeSeedPrevArr : seedPrevArr.size = n := by
          simp [seedPrevArr, n, TList.size_toAnyArray, htPrevSize, List.length_append]

        have hnodeLast :
            Runtime.Autograd.Tape.getNode? (t := tNext) n = some runtimeNode := by
          simp [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, n]

        have hreqLast : runtimeNode.requires_grad = true := by rfl

        have hshapeLast : outAny.s = runtimeNode.value.s := by
          simp [outAny, runtimeNode]
          rfl

        have hpids :
            ∀ {pid : Nat} {pg : Runtime.AnyTensor α},
              (pid, pg) ∈ (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0) → pid < n
                := by
          intro pid pg hmem
          have hback :
              runtimeNode.backward outAny =
                .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0) := by
            have hτ : outAny.s = τ := by rfl
            have hcastAny : ∀ h : outAny.s = τ, Tensor.castShape outAny.t h = seedOut := by
              intro h
              cases h
              rfl
            simp [runtimeNode, outAny, hτ, hcastAny, ctx, contrib, hctxPrev]
          have hpidlt :=
            compileAuxData_backward_pids_lt_id (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ])
              (.snoc (ss := ssPrev) (τ := τ) g node) x d0
              n runtimeNode (by
                simpa [compileAuxData, hprev, tNext, runtimeNode, Runtime.Autograd.Tape.getNode?,
                  htNextNodes, n])
              outAny _ hback hmem
          simpa [n] using hpidlt

        have hnodes0 :
            ∀ i (hi : i < (Γ ++ ssPrev).length),
              let id := (0 : Nat) + i
              ∃ nodeAt : Runtime.Autograd.Node α,
                tNext.getNode? id = some nodeAt ∧ nodeAt.requires_grad = true ∧
                  nodeAt.value.s =
                    ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                        simpa [TList.size_toAnyArray] using hi)).s := by
          intro i hi
          have hiT : i < tPrev.nodes.size := by
            -- `tPrev.nodes.size = (Γ ++ ssPrev).length`
            simpa [htPrevSize, List.length_append, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
              using hi
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[i]'hiT
          have hgetNext : tNext.getNode? i = some nodeAt := by
            -- index `< tPrev.nodes.size`, so `push` doesn't change it
            have : (tPrev.nodes.push runtimeNode)[i]? = some (tPrev.nodes[i]'hiT) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hiT)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this
          have hreq : nodeAt.requires_grad = true := by
            have hreq' :=
              (compileAuxData_requires_grad_true (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0) i
                (by
                simpa [hprev] using hiT)
            simpa [hprev, nodeAt] using hreq'
          -- Shapes: both are the `i`th shape in `Γ ++ ssPrev`.
          have hseedShape :
              nodeAt.value.s =
                ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                    simpa [TList.size_toAnyArray] using hi)).s := by
            let fi : Fin (Γ ++ ssPrev).length := ⟨i, hi⟩
            -- `tPrev.nodes.map value = ctxPrev.toAnyArray`
            have hvals :
                tPrev.nodes.map (fun nd => nd.value) =
                  TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev := by
              simpa [hprev] using
                (compileAuxData_values_eq (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)
            have hvalOpt := congrArg (fun a => a[i]?) hvals
            -- Evaluate both sides at `i`.
            have hnodeVal :
                nodeAt.value = (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                  -- `i < ctxPrev.toAnyArray.size` because it matches `tPrev.nodes.size`
                  simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc] using
                    hiT) := by
              -- Left: map+index gives `some nodeAt.value`
              have hleft :
                  (tPrev.nodes.map (fun nd => nd.value))[i]? = some nodeAt.value := by
                have : tPrev.nodes[i]? = some nodeAt := by
                  simp [nodeAt,
                    Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := i) hiT]
                simp [Array.getElem?_map, this, nodeAt]
              -- Right: in-bounds `getElem?` is `some _`
              have hright :
                  (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]? =
                    some ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                      simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc]
                        using hiT)) := by
                have hiCtx :
                    i < (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev).size := by
                  simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc] using
                    hiT
                simp [Array.getElem?_eq_getElem (xs := (TList.toAnyArray (α := α) (ss := Γ ++
                  ssPrev) ctxPrev)) (i := i) hiCtx]
              -- Combine and extract the value equality.
              have : some nodeAt.value =
                  some ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                    simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc]
                      using hiT)) := by
                -- rewrite both sides of `hvalOpt` using `hleft`/`hright`
                simpa [hleft, hright] using hvalOpt
              simpa using congrArg (fun o => o.getD nodeAt.value) this
            have hnode_s :
                nodeAt.value.s = (Γ ++ ssPrev).get fi := by
              have hiCtx :
                  i < (TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev).size := by
                simpa [TList.size_toAnyArray, htPrevSize, List.length_append, Nat.add_assoc] using
                  hiT
              have hctx_s :
                  ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'hiCtx).s =
                    (Γ ++ ssPrev).get fi := by
                -- `ctxPrev.get fi : Tensor α ((Γ ++ ssPrev).get fi)`, so the RHS shape is
                -- definitional.
                simpa [fi] using
                  congrArg Runtime.AnyTensor.s
                    (TList.get_toAnyArray (α := α) (ss := Γ ++ ssPrev) ctxPrev fi)
              -- rewrite the LHS using `hnodeVal`
              simpa [hnodeVal] using hctx_s
            have hseed_s :
                ((TList.toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                    simpa [TList.size_toAnyArray] using hi)).s = (Γ ++ ssPrev).get fi := by
              simpa [fi] using congrArg Runtime.AnyTensor.s
                (TList.get_toAnyArray (α := α) (ss := Γ ++ ssPrev) seedPrev fi)
            exact hnode_s.trans hseed_s.symm

          -- discharge the `let id := 0 + i`
          refine ⟨nodeAt, ?_, hreq, hseedShape⟩
          simpa [Nat.zero_add] using hgetNext

        have hstepLast :
            Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (seedPrevArr.push outAny) n =
              .ok (seedPrevArr'.push outAny) := by
          have haccLast : (seedPrevArr.push outAny)[n]? = some outAny := by
            have : (seedPrevArr.push outAny)[seedPrevArr.size]? = some outAny := by
              simp
            simpa [hsizeSeedPrevArr] using this

          -- Show the `addGradAll` fold for the last node matches `TList.add` on the prefix, leaving
          -- `[outAny]` untouched.
          have hfoldLast :
              (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0).foldlM
                  (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := tNext) acc2 pid pg)
                  (seedPrevArr.push outAny) =
                .ok (seedPrevArr'.push outAny) := by
            have hfold :=
              foldlM_addGradAll_toIndexedAnyList_eq_add (α := α) (t := tNext)
                (ss := Γ ++ ssPrev) (pref := #[]) (seed := seedPrev) (contrib := contrib) (suffix :=
                  #[outAny])
                (by
                  intro i hi
                  have := hnodes0 i hi
                  simpa using this)
            -- Simplify the array concatenations and rewrite `seedPrev'`.
            simpa [seedPrevArr, seedPrevArr', seedPrev', Array.append_assoc, Array.append_empty,
              Array.empty_append,
              Array.append_singleton, TList.toAnyArray_cast] using hfold

          -- Unfold the step and rewrite the `backward` call using `hfoldLast`.
          cases hshapeLast
          have hreqLast : runtimeNode.requires_grad = true := by rfl
          have hout : outAny.s = τ := by rfl
          have hshapeNode : outAny.s = runtimeNode.value.s := by rfl
          have hcast : Tensor.castShape seedOut hout = seedOut := by
            cases hout
            rfl

          have hbackLast :
              runtimeNode.backward outAny =
                .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0) := by
            simp [runtimeNode, outAny, hctxPrev, ctx, contrib, Tensor.castShape,
              Runtime.Autograd.AnyTensor.mk]

          have hbackLast2 :
              runtimeNode.backward
                  { s := runtimeNode.value.s
                    t := Tensor.castShape outAny.t hshapeNode } =
                .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contrib 0) := by
            -- Keep `Tensor.cast_shape` folded so this rewrite matches the `backwardDenseFromStep`
            -- unfolding.
            cases hshapeNode
            simpa [Tensor.castShape] using hbackLast
          -- Unfold the step and reduce the control flow (`getNode?`, `requires_grad`, `acc[id]?`,
          -- shape check),
          -- then rewrite the `backward` call and finish with the pre-proved fold lemma.
          simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodeLast, hreqLast, haccLast]
          simp [hshapeNode]
          rw [hbackLast2]
          simpa [seedPrevArr, seedPrevArr', outAny] using hfoldLast

        have ihPrevLoop :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tPrev) n seedPrevArr' =
              .ok gradsPrevArr := by
          have ihPrev :
              Runtime.Autograd.Tape.backwardDenseFrom (t := tPrev) (grads0 := seedPrevArr') =
                .ok gradsPrevArr := by
            have h := ih (seed := seedPrev')
            simpa [hprev, seedPrevArr', gradsPrevArr, gradsPrev] using h
          have hsizeSeedPrevArr' : seedPrevArr'.size = n := by
            simp [seedPrevArr', n, TList.size_toAnyArray, htPrevSize, List.length_append]
          have hsize : seedPrevArr'.size = tPrev.nodes.size := by
            simpa [n] using hsizeSeedPrevArr'
          simpa [Runtime.Autograd.Tape.backwardDenseFrom, hsize, n] using ihPrev

        -- Helper: `addGradAll` commutes with pushing an unused last slot.
        have haddGradAllPush :
            ∀ (acc : Array (Runtime.AnyTensor α)) (hacc : acc.size = n)
              (pid : Nat) (pg : Runtime.AnyTensor α),
              pid < n →
              Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outAny) pid pg =
                Except.map (fun a => a.push outAny)
                  (Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg) := by
          intro acc hacc pid pg hpid
          have hpidPrev : pid < tPrev.nodes.size := by
            simpa [n] using hpid
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[pid]'hpidPrev
          have hnodePrev : Runtime.Autograd.Tape.getNode? (t := tPrev) pid = some nodeAt := by
            simp [Runtime.Autograd.Tape.getNode?, nodeAt,
              Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := pid) hpidPrev]
          have hnodeNext : Runtime.Autograd.Tape.getNode? (t := tNext) pid = some nodeAt := by
            have : (tPrev.nodes.push runtimeNode)[pid]? = some (tPrev.nodes[pid]'hpidPrev) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hpidPrev)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this

          have hpidAcc : pid < acc.size := by
            simpa [hacc] using hpid
          have hgetPrev : acc[pid]? = some (acc[pid]'hpidAcc) := by
            simp
          have hgetNext : (acc.push outAny)[pid]? = some (acc[pid]'hpidAcc) := by
            simpa using (Array.getElem?_push_lt (xs := acc) (x := outAny) hpidAcc)

          cases hreq : nodeAt.requires_grad with
          | false =>
              simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hnodeNext, hreq, Except.map]
              rfl
          | true =>
              by_cases hshape : pg.s = nodeAt.value.s
              · by_cases hex : (acc[pid]'hpidAcc).s = nodeAt.value.s
                ·
                  let pg' : Runtime.AnyTensor α :=
                    { s := nodeAt.value.s, t := Tensor.castShape pg.t hshape }
                  let existing' : Runtime.AnyTensor α :=
                    { s := nodeAt.value.s, t := Tensor.castShape (acc[pid]'hpidAcc).t hex }
                  cases hadd : Runtime.Autograd.AnyTensor.add existing' pg' with
                  | error e =>
                      have hprev :
                          Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg =
                            .error e := by
                        simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, hshape, hgetPrev,
                          hex, pg', existing', hadd,
                          throw, throwThe, MonadExceptOf.throw]
                        simp [Bind.bind, Except.bind]
                      have hnext :
                          Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outAny)
                            pid pg = .error e := by
                        simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, hshape, hgetNext,
                          hex, pg', existing', hadd,
                          throw, throwThe, MonadExceptOf.throw]
                        simp [Bind.bind, Except.bind]
                      simp [hprev, hnext, Except.map]
                  | ok summed =>
                      have hpidAccPush : pid < (acc.push outAny).size := by
                        simpa [Array.size_push] using Nat.lt_trans hpidAcc (Nat.lt_succ_self
                          acc.size)
                      have hprev :
                          Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg =
                            .ok (acc.set pid summed (h := hpidAcc)) := by
                        simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, hshape, hex, pg',
                          existing', hadd,
                          hpidAcc, throw, throwThe, MonadExceptOf.throw]
                      have hnext :
                          Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outAny)
                            pid pg =
                            .ok ((acc.set pid summed (h := hpidAcc)).push outAny) := by
                        have hpid_le : pid ≤ acc.size := Nat.le_of_lt hpidAcc
                        have hget : (acc.push outAny)[pid] = acc[pid] := by
                          simpa using
                            (Array.getElem_push_lt (xs := acc) (x := outAny) (i := pid) hpidAcc)
                        simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, hshape, hex, pg',
                          existing', hadd,
                          hpidAcc, hpid_le, hget, Array.set_push, throw, throwThe,
                            MonadExceptOf.throw]
                      simp [hprev, hnext, Except.map]
                ·
                  simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hnodeNext, hreq, hshape,
                    hgetPrev, hgetNext, hex,
                    Except.map, throw, throwThe, MonadExceptOf.throw]
              ·
                simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hnodeNext, hreq, hshape,
                  Except.map,
                  throw, throwThe, MonadExceptOf.throw]

        -- Helper: `backwardDenseFromStep` commutes with pushing an unused last slot for ids `< n`.
        have hstepPush :
            ∀ (id : Nat) (hid : id < n) (acc : Array (Runtime.AnyTensor α)),
              acc.size = n →
              Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (acc.push outAny) id =
                Except.map (fun a => a.push outAny)
                  (Runtime.Autograd.Tape.backwardDenseFromStep (t := tPrev) acc id) := by
          intro id hid acc hacc
          have hidPrev : id < tPrev.nodes.size := by
            simpa [n] using hid
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[id]'hidPrev
          have hnodePrev : Runtime.Autograd.Tape.getNode? (t := tPrev) id = some nodeAt := by
            simp [Runtime.Autograd.Tape.getNode?, nodeAt,
              Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := id) hidPrev]
          have hnodeNext : Runtime.Autograd.Tape.getNode? (t := tNext) id = some nodeAt := by
            have : (tPrev.nodes.push runtimeNode)[id]? = some (tPrev.nodes[id]'hidPrev) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hidPrev)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this
          have hidAcc : id < acc.size := by
            simpa [hacc] using hid
          have hgetAcc : acc[id]? = some (acc[id]'hidAcc) := by
            simp
          have hgetAccPush : (acc.push outAny)[id]? = some (acc[id]'hidAcc) := by
            simpa using (Array.getElem?_push_lt (xs := acc) (x := outAny) hidAcc)
          cases hreq : nodeAt.requires_grad with
          | false =>
              simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                hgetAcc, hgetAccPush,
                Except.map]
              rfl
          | true =>
              by_cases hshape : (acc[id]'hidAcc).s = nodeAt.value.s
              · -- shape ok, split on `backward` result
                let dLdy : Runtime.AnyTensor α :=
                  { s := nodeAt.value.s, t := Tensor.castShape (acc[id]'hidAcc).t hshape }
                cases hback : nodeAt.backward dLdy with
                | error e =>
                    simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                      hgetAcc, hgetAccPush,
                      hshape, dLdy, hback, Except.map]
                    rfl
                | ok contribs =>
                    have hpids :
                          ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ contribs → pid < id
                            := by
                        intro pid pg hmem
                        have hgetComp :
                            Runtime.Autograd.Tape.getNode?
                                (t := (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x
                                  d0).1)
                                id =
                              some nodeAt := by
                          simpa [hprev] using hnodePrev
                        exact
                          compileAuxData_backward_pids_lt_id (α := α) (Δ := Δ) (Γ := Γ) (ss :=
                            ssPrev) g x d0 id nodeAt
                              hgetComp dLdy contribs hback hmem

                    have hfoldAux :
                        ∀ (cs : List (Nat × Runtime.AnyTensor α)) (acc0 accOut : Array
                          (Runtime.AnyTensor α)),
                          acc0.size = n →
                          (∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ cs → pid < n) →
                          cs.foldlM (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t :=
                            tNext) acc2 pid pg)
                              (acc0.push outAny) =
                            Except.map (fun a => a.push outAny)
                              (cs.foldlM
                                (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := tPrev)
                                  acc2 pid pg) acc0) := by
                      intro cs
                      induction cs with
                      | nil =>
                          intro acc0 accOut _hsize _hpids
                          simp [List.foldlM, Except.map]
                          rfl
                      | cons hd tl ih =>
                          intro acc0 accOut hsize hpids
                          rcases hd with ⟨pid, pg⟩
                          have hpid : pid < n := by
                            exact hpids (pid := pid) (pg := pg) (by simp)
                          have hadd :=
                            haddGradAllPush (acc := acc0) (hacc := hsize) (pid := pid) (pg := pg)
                              hpid
                          cases hret : Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc0)
                            pid pg with
                          | error e =>
                              -- both folds error at the first step
                              simp [List.foldlM, hret, hadd, Except.map]
                              rfl
                          | ok acc1 =>
                              have hret' :
                                  Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc0.push
                                    outAny) pid pg =
                                    .ok (acc1.push outAny) := by
                                -- unfold `Except.map` in `hadd`
                                simpa [Except.map, hret] using hadd
                              have hsize1 : acc1.size = n := by
                                have := addGradAll_ok_size (t := tPrev) (grads := acc0) (id := pid)
                                  (g := pg)
                                  (grads' := acc1) (by simpa using hret)
                                simpa [hsize] using this
                              have hpids_tl :
                                  ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ tl → pid < n
                                    := by
                                intro pid pg hmem
                                exact hpids (pid := pid) (pg := pg) (by simp [hmem])
                              have ih' :=
                                ih (acc0 := acc1) (accOut := accOut) hsize1 hpids_tl
                              -- unfold the `foldlM` for the cons case on both sides
                              simpa [List.foldlM, hret, hret', Except.map] using ih'

                    have hpids_n :
                        ∀ {pid : Nat} {pg : Runtime.AnyTensor α}, (pid, pg) ∈ contribs → pid < n :=
                          by
                      intro pid pg hmem
                      exact Nat.lt_trans (hpids (pid := pid) (pg := pg) hmem) hid

                    -- Apply the fold lemma.
                    have hfold :=
                      hfoldAux contribs acc (accOut := by
                        exact acc) hacc hpids_n
                    -- Unfold the step definitions, then discharge the remaining fold goal via
                    -- `hfold`.
                    simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                      hgetAcc, hgetAccPush,
                      hshape, dLdy, hback]
                    simpa [Except.map] using hfold
              · -- shape mismatch
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                  hgetAcc, hgetAccPush, hshape,
                  Except.map, throw, throwThe, MonadExceptOf.throw]

        -- The loop itself commutes with pushing an unused last slot.
        have hloopPush :
            ∀ m (hm : m ≤ n) (acc : Array (Runtime.AnyTensor α)),
              acc.size = n →
              Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) m (acc.push outAny) =
                Except.map (fun a => a.push outAny)
                  (Runtime.Autograd.Tape.backwardDenseFromLoop (t := tPrev) m acc) := by
            intro m hm acc hacc
            induction m generalizing acc with
            | zero =>
                simp [Runtime.Autograd.Tape.backwardDenseFromLoop, Except.map]
                rfl
            | succ m ihm =>
                have hm' : m ≤ n := Nat.le_trans (Nat.le_succ m) hm
                have hmid : m < n := Nat.lt_of_lt_of_le (Nat.lt_succ_self m) hm
                have hstep := hstepPush (id := m) (hid := hmid) (acc := acc) hacc
                cases hret : Runtime.Autograd.Tape.backwardDenseFromStep (t := tPrev) acc m with
                | error e =>
                    -- both loops error on this step
                    simp [Runtime.Autograd.Tape.backwardDenseFromLoop, hret, hstep, Except.map]
                    rfl
                | ok acc1 =>
                    have hstep' :
                        Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (acc.push outAny) m
                          =
                          .ok (acc1.push outAny) := by
                      simpa [Except.map, hret] using hstep
                    have hsize1 : acc1.size = n := by
                      have := backwardDenseFromStep_ok_size (t := tPrev) (acc := acc) (id := m)
                        (acc' := acc1)
                        (by simpa using hret)
                      simpa [hacc] using this
                    have ih' := ihm (acc := acc1) hm' hsize1
                    simpa [Runtime.Autograd.Tape.backwardDenseFromLoop, hret, hstep', Except.map]
                      using ih'

        have hloopFinal :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) n (seedPrevArr'.push outAny) =
              .ok (gradsPrevArr.push outAny) := by
          have h := hloopPush n (le_rfl) seedPrevArr' (by
            simp [seedPrevArr', n, TList.size_toAnyArray, htPrevSize, List.length_append])
          simpa [ihPrevLoop, Except.map] using h

        have hloopAll :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) (n + 1) (seedPrevArr.push
              outAny) =
              .ok (gradsPrevArr.push outAny) := by
          -- Unfold the loop one step, rewrite via `hstepLast`, then discharge with `hloopFinal`.
          simp [Runtime.Autograd.Tape.backwardDenseFromLoop, hstepLast]
          simpa using hloopFinal
        simpa [n, htPrevSize, Nat.add_assoc] using hloopAll

      simpa [hTape, hBackpropArr] using hmain
  end Graph

  end Algebra
  end Autograd
  end Proofs
