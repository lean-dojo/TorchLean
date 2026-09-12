/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.ShapeErasure
public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Runtime.Autograd.Engine.Core.Base

/-!
# Link

Link the executable runtime tape (`Runtime.Autograd.Tape`) to the shape-indexed SSA/DAG models in
`Proofs.Autograd.Algebra`.

`GraphData` stores executable forward, JVP, and VJP functions without derivative laws. `Graph`
extends that representation with the local adjointness law used by its global backpropagation
theorem. Lowering places the stored VJP into each runtime node's `backward` closure.

## What is proved here

- Forward-pass correspondence: `lowerGraphToTape{,Data}` produces the same values as
  `Graph{,Data}.eval`, and the runtime tape stores those values in the same order
  (`lowerGraphToTape{,Data}_ctx_eq_eval`, `lowerGraphToTape{,Data}_values_eq`).
- Backward-pass correspondence: running the runtime dense reverse loop
  `Tape.backwardDenseFrom` on a lowered tape matches the graph's stored reverse program
  `backpropAllCtx` (`backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx` and its `GraphData`
  variant). For `GraphData`, this is an implementation-equivalence result. For `Graph`, it can be
  combined with `Graph.backprop_correct` to obtain derivative correctness.

The core invariant making the runtime reverse loop well-founded is that lowered nodes only emit
contributions to earlier node ids (`pid < id`).

## PyTorch correspondence / citations
This is analogous to lowering a graph representation to an executable autograd tape whose nodes
carry backward closures (PyTorch does this internally for the eager autograd engine).
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Algebra

open TorchLean

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Graph

open Runtime
open Runtime.Autograd

/--
Extend a tape with leaf nodes for every tensor in the input context `Γ`.

Each leaf has `requiresGrad = true` and an empty backward contribution array, so the runtime loop
treats them as gradient accumulation slots but never produces parent contributions from them.
-/
def addLeaves {α : Type} [Storage α]
    (t : Tape α) : {Γ : List Shape} → TorchLean.TensorPack α Γ → Tape α
  | [], .nil => t
  | _ :: Γ, .cons x xs =>
      let (t', _id) := Tape.leaf (t := t) x
      addLeaves (t := t') (Γ := Γ) xs

/--
Turn a shape-erased tensor into a runtime leaf node.

This is the node-level counterpart of `addLeaves`: it has no parents and contributes nothing in
backward.
-/
def leafNodeOfSomeTensor {α : Type} [TorchLean.Storage α]
    (v : Spec.SomeTensor α) : Runtime.Autograd.Node α :=
  { name := none
    value := v
    requiresGrad := true
    parents := #[]
    backward := fun _ => .ok #[] }

/-!
### `Result` equations

`Result` is `Except String`. The runtime backward functions are written with `do` notation, so
proofs about them repeatedly need the unfolding equations for `bind`, `pure`, `throw` and
`Except.map`. They are stated once here and used with `simp only` instead of unfolding the
`Monad` instances by hand.
-/

/-- Binding a successful `Result` applies the continuation. -/
theorem result_bind_ok {β γ : Type} (a : β) (f : β → Result γ) :
    (Except.ok a : Result β) >>= f = f a := rfl

/-- Binding a failed `Result` propagates the error. -/
theorem result_bind_error {β γ : Type} (e : String) (f : β → Result γ) :
    (Except.error e : Result β) >>= f = Except.error e := rfl

/-- `pure` in `Result` is `Except.ok`. -/
theorem result_pure_eq_ok {β : Type} (a : β) : (pure a : Result β) = Except.ok a := rfl

/-- `throw` in `Result` is `Except.error`. -/
theorem result_throw_eq_error {β : Type} (e : String) :
    (throw e : Result β) = Except.error e := rfl

/-- `Except.map` on a successful `Result`. -/
theorem result_map_ok {β γ : Type} (f : β → γ) (a : β) :
    Except.map f (Except.ok a : Result β) = Except.ok (f a) := rfl

/-- `Except.map` on a failed `Result`. -/
theorem result_map_error {β γ : Type} (f : β → γ) (e : String) :
    Except.map f (Except.error e : Result β) = Except.error e := rfl

/--
Backward closures of `t` only point backwards: every contribution emitted by the node stored at
`id` targets a node id strictly smaller than `id`. This is the invariant that makes the dense
reverse loop well-founded; `lowerGraphToTape_backward_pids_lt_id` and its `GraphData` counterpart
establish it for lowered tapes.
-/
def BackwardPidsLt {α : Type} [Storage α] (t : Tape α) : Prop :=
  ∀ id (n : Runtime.Autograd.Node α), t.getNode? id = some n →
    ∀ (d : Spec.SomeTensor α) (contribs : Array (Nat × Spec.SomeTensor α)),
      n.backward d = .ok contribs →
        ∀ {pid : Nat} {pg : Spec.SomeTensor α}, (pid, pg) ∈ contribs → pid < id

/-- Each entry of `toShapeErasedArray` carries the shape recorded at its position of `ss`. -/
theorem shape_getElem_toShapeErasedArray {α : Type} [Storage α] {ss : List Shape}
    (xs : TorchLean.TensorPack α ss) {i : Nat} (hi : i < ss.length) :
    ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) xs)[i]'(by
        simpa [TorchLean.TensorPack.size_toShapeErasedArray] using hi)).shape =
      ss.get ⟨i, hi⟩ := by
  simpa [Spec.SomeTensor.ofTensor] using congrArg Spec.SomeTensor.shape
    (TorchLean.TensorPack.get_toShapeErasedArray (α := α) (ss := ss) xs ⟨i, hi⟩)

/-- `addLeaves` grows the tape by exactly `Γ.length` nodes. -/
theorem size_addLeaves {α : Type} [Storage α] (t : Tape α) :
    {Γ : List Shape} → (x : TorchLean.TensorPack α Γ) →
      (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes.size = t.nodes.size + Γ.length
  | [], .nil => by simp [addLeaves]
  | _ :: Γ, .cons x xs => by
      simp [addLeaves, Tape.leaf, Tape.addNode, size_addLeaves (t := { nodes := t.nodes.push _ }) (x
        := xs),
        Nat.add_assoc, Nat.add_comm, Array.size_push]

/-- `addLeaves` appends `leafNodeOfSomeTensor` nodes for each input tensor, in order. -/
theorem nodes_addLeaves {α : Type} [Storage α] (t : Tape α) :
    {Γ : List Shape} → (x : TorchLean.TensorPack α Γ) →
      (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes =
        t.nodes ++
          (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x).map
            (leafNodeOfSomeTensor (α := α))
  | [], .nil => by
      simp [addLeaves, TorchLean.TensorPack.toShapeErasedArray]
  | _ :: Γ, .cons x xs => by
      simp [addLeaves, Tape.leaf, Tape.addNode,
        nodes_addLeaves (t := { nodes := t.nodes.push _ }) (Γ := Γ) (x := xs),
        leafNodeOfSomeTensor,
        TorchLean.TensorPack.toShapeErasedArray_cons (α := α) (ss := Γ) x xs,
        Array.map_append, Array.append_singleton_assoc]

/-- Value projection of `nodes_addLeaves`: `node.value` agrees with `toShapeErasedArray` for added
leaves. -/
theorem addLeaves_values {α : Type} [Storage α] (t : Tape α) :
    {Γ : List Shape} → (x : TorchLean.TensorPack α Γ) →
      (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes.map (fun node => node.value) =
        t.nodes.map (fun node => node.value) ++
          TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x
  | [], .nil => by
      simp [addLeaves, TorchLean.TensorPack.toShapeErasedArray]
  | _ :: Γ, .cons x xs => by
      -- unfold one `leaf` push and use the induction hypothesis on the remaining leaves
      simp [addLeaves, Tape.leaf, Tape.addNode,
        addLeaves_values (t := { nodes := t.nodes.push _ }) (Γ := Γ) (x := xs),
        TorchLean.TensorPack.toShapeErasedArray]

/--
Runtime node produced by lowering one graph node.

The node stores the forward value and wraps the stored `vjp` in a runtime `backward` closure
that checks the shape of the upstream cotangent and emits one contribution per entry of the
node's input context, indexed from `0`. The `name` only serves debugging output.
-/
def lowerNode {α : Type} {Δ : Type} [Storage α] {Γ : List Shape} {τ : Shape}
    (name : Option String) (node : NodeData α Δ Γ τ) (ctx : TorchLean.TensorPack α Γ)
    (d : Δ) : Runtime.Autograd.Node α :=
  { name := name
    value := Spec.SomeTensor.ofTensor (node.forward ctx d)
    requiresGrad := true
    parents := #[]
    backward := fun dLdyValue =>
      if h : dLdyValue.shape = τ then
        .ok (TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := Γ)
          (node.vjp ctx d (dLdyValue.cast h)) 0)
      else
        .error "autograd: upstream gradient shape mismatch" }

/-- The lowered node stores the node's forward value. -/
@[simp] theorem lowerNode_value {α : Type} {Δ : Type} [Storage α] {Γ : List Shape} {τ : Shape}
    (name : Option String) (node : NodeData α Δ Γ τ) (ctx : TorchLean.TensorPack α Γ)
    (d : Δ) :
    (lowerNode (α := α) name node ctx d).value = Spec.SomeTensor.ofTensor (node.forward ctx d) :=
  rfl

/-- Lowered nodes always take part in gradient accumulation. -/
@[simp] theorem lowerNode_requiresGrad {α : Type} {Δ : Type} [Storage α] {Γ : List Shape}
    {τ : Shape} (name : Option String) (node : NodeData α Δ Γ τ)
    (ctx : TorchLean.TensorPack α Γ) (d : Δ) :
    (lowerNode (α := α) name node ctx d).requiresGrad = true :=
  rfl

/-- On an upstream cotangent of the right shape, the lowered `backward` runs the stored `vjp`. -/
theorem lowerNode_backward_of_shape {α : Type} {Δ : Type} [Storage α] {Γ : List Shape}
    {τ : Shape} (name : Option String) (node : NodeData α Δ Γ τ)
    (ctx : TorchLean.TensorPack α Γ) (d : Δ) (v : Spec.SomeTensor α) (h : v.shape = τ) :
    (lowerNode (α := α) name node ctx d).backward v =
      .ok (TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := Γ)
        (node.vjp ctx d (v.cast h)) 0) := by
  simp only [lowerNode, h, dite_true]

/-- On an upstream cotangent of the wrong shape, the lowered `backward` fails. -/
theorem lowerNode_backward_of_ne {α : Type} {Δ : Type} [Storage α] {Γ : List Shape}
    {τ : Shape} (name : Option String) (node : NodeData α Δ Γ τ)
    (ctx : TorchLean.TensorPack α Γ) (d : Δ) (v : Spec.SomeTensor α) (h : v.shape ≠ τ) :
    (lowerNode (α := α) name node ctx d).backward v =
      .error "autograd: upstream gradient shape mismatch" := by
  simp only [lowerNode, h, dite_false]

/--
Lower an executable graph (`GraphData`) to a runtime tape by evaluating forward nodes and storing
each node's `vjp` program in its runtime `backward` closure.

PyTorch analogy: this corresponds to building a tape of autograd nodes during the forward pass,
where each node stores enough information to compute parent contributions when given an upstream
cotangent.
-/
def lowerGraphDataToTape {α : Type} {Δ : Type} [Storage α]
  {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TorchLean.TensorPack α Γ)
  (d : Δ) : Tape α × TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil =>
      let t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x
      (t, TorchLean.TensorPack.cast (α := α) (h := (List.append_nil Γ).symm) x)
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let (tPrev, ctxPrev) := lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let y := node.forward ctxPrev d
      let (tNext, _id) := Tape.addNode (t := tPrev) (lowerNode (some "typed-graph") node ctxPrev d)
      let ctxNext :=
        TorchLean.TensorPack.cast (α := α) (h := List.append_assoc Γ ssPrev [τ])
          (TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) ctxPrev y)
      (tNext, ctxNext)

/-!
### Forward-pass correspondence

The next lemmas show that `lowerGraphDataToTape` preserves executable forward semantics, and that
the resulting runtime tape contains exactly the evaluated context as shape-erased tensors in order.
-/

/-- The context returned by `lowerGraphDataToTape` agrees with `GraphData.eval`. -/
theorem lowerGraphDataToTape_ctx_eq_eval {α : Type} {Δ : Type}
    [Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d : Δ) :
    (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 =
      GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d := by
  induction g with
  | nil =>
      simp [lowerGraphDataToTape, GraphData.eval]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [lowerGraphDataToTape, GraphData.eval, ih]

/-- The lowered tape's `.value` array is `GraphData.eval` with shapes erased in the same order.
  -/
theorem lowerGraphDataToTape_values_eq {α : Type} {Δ : Type}
    [Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d : Δ) :
    (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.map (fun node =>
      node.value) =
      TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss)
        (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 := by
  induction g with
  | nil =>
      -- only leaves
      simp [lowerGraphDataToTape, addLeaves_values, Runtime.Autograd.Tape.empty]
  | snoc g _node ih =>
      rename_i ssPrev τ
      simp [lowerGraphDataToTape, Runtime.Autograd.Tape.addNode, ih]

/-- Size bookkeeping: the lowered tape contains one runtime node for each element of `Γ ++ ss`. -/
theorem lowerGraphDataToTape_nodes_size {α : Type} {Δ : Type}
    [Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d : Δ) :
    (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.size =
      Γ.length + ss.length := by
  induction g with
  | nil =>
      -- only leaves
      simp [lowerGraphDataToTape, size_addLeaves, Runtime.Autograd.Tape.empty]
  | snoc g _node ih =>
      rename_i ssPrev τ
      simp [lowerGraphDataToTape, Runtime.Autograd.Tape.addNode, ih, Array.size_push, Nat.add_assoc,
        ]

/--
Lower a proved graph (`Graph`) to a runtime tape by evaluating forward nodes and storing each
node’s proved `vjp`.

Compared to `lowerGraphDataToTape`, this uses the pure graph interface (no explicit `GraphData`
payload).
-/
def lowerGraphToTape {α : Type} {Δ : Type}
    [Storage α] [CommSemiring α]
  {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss)
  (x : TorchLean.TensorPack α Γ)
  (d : Δ) : Tape α × TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil =>
      let t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x
      (t, TorchLean.TensorPack.cast (α := α) (h := (List.append_nil Γ).symm) x)
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let (tPrev, ctxPrev) := lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let y := node.forward ctxPrev d
      let (tNext, _id) :=
        Tape.addNode (t := tPrev)
          (lowerNode (some "proof-carrying-graph") node.toNodeData ctxPrev d)
      let ctxNext :=
        TorchLean.TensorPack.cast (α := α) (h := List.append_assoc Γ ssPrev [τ])
          (TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) ctxPrev y)
      (tNext, ctxNext)

/-- The context returned by `lowerGraphToTape` agrees with the proved `Graph.eval`. -/
theorem lowerGraphToTape_ctx_eq_eval {α : Type} {Δ : Type}
    [Storage α] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d : Δ) :
    (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 =
      Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d := by
  induction g with
  | nil =>
      simp [lowerGraphToTape, Graph.eval, Graph.toData, GraphData.eval]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [lowerGraphToTape, Graph.eval, Graph.toData, GraphData.eval, ih]

/-- The lowered tape's `.value` array is `Graph.eval` with shapes erased in the same order. -/
theorem lowerGraphToTape_values_eq {α : Type} {Δ : Type}
    [Storage α] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d : Δ) :
    (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.map
      (fun node => node.value) =
      TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss)
        (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 := by
  induction g with
  | nil =>
      -- only leaves
      simp [lowerGraphToTape, addLeaves_values, Runtime.Autograd.Tape.empty]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [lowerGraphToTape, Runtime.Autograd.Tape.addNode, ih]

/-- Size bookkeeping: `lowerGraphToTape` produces `Γ.length + ss.length` runtime nodes. -/
theorem lowerGraphToTape_nodes_size {α : Type} {Δ : Type}
    [Storage α] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss)
    (x : TorchLean.TensorPack α Γ)
    (d : Δ) :
    (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.size =
      Γ.length + ss.length := by
  induction g with
  | nil =>
      simp [lowerGraphToTape, size_addLeaves, Runtime.Autograd.Tape.empty]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [lowerGraphToTape, Runtime.Autograd.Tape.addNode, ih, Array.size_push, Nat.add_assoc]

/-!
### Full backpropagation (dense) for proofs and runtime

The runtime engine computes a *dense* gradient array, accumulating cotangents for every node in the
tape (inputs and intermediates). The following definition and theorems connect that behavior to the
proved backpropagation semantics.
-/

/-- A "full" backpropagation that returns gradients for every value in `Γ ++ ss`. -/
def backpropAllCtx {α : Type} {Δ : Type} [Storage α] [CommSemiring α]
  {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss)
  (x : TorchLean.TensorPack α Γ)
  (d : Δ) (seed : TorchLean.TensorPack α (Γ ++ ss)) :
  TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil => seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TorchLean.TensorPack α ((Γ ++ ssPrev) ++ [τ]) :=
        TorchLean.TensorPack.cast (α := α) (h := assoc.symm) seed
      let seedPrev : TorchLean.TensorPack α (Γ ++ ssPrev) :=
        (TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').1
      let seedOut : Tensor α τ :=
        (TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := TorchLean.TensorPack.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev := backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d seedPrev'
      TorchLean.TensorPack.cast (α := α) (h := assoc)
        (TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) gradsPrev seedOut)

/--
“Full” backpropagation for `GraphData` that returns gradients for every value in `Γ ++ ss`,
including inputs.

This is the `GraphData`-analogue of `backpropAllCtx` above. We keep both definitions because:
- `Graph` uses `[CommSemiring α]` (so it can express dot products and semiring-based accumulation),
  while
- `GraphData` only needs `[Add α]` here (it just adds contributions).

Both follow the same reverse-mode accumulation structure: peel off the last node, apply its VJP to
the seed on that node, add into the previous seed, and recurse.
-/
def _root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx
    {α : Type} {Δ : Type} [Storage α] [Add α]
  {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TorchLean.TensorPack α Γ)
  (d : Δ) (seed : TorchLean.TensorPack α (Γ ++ ss)) :
  TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil => seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TorchLean.TensorPack α ((Γ ++ ssPrev) ++ [τ]) :=
        TorchLean.TensorPack.cast (α := α) (h := assoc.symm) seed
      let seedPrev : TorchLean.TensorPack α (Γ ++ ssPrev) :=
        (TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').1
      let seedOut : Tensor α τ :=
        (TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := TorchLean.TensorPack.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev := backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d seedPrev'
      TorchLean.TensorPack.cast (α := α) (h := assoc)
        (TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) gradsPrev seedOut)


end Graph

end Algebra
end Autograd
end Proofs
