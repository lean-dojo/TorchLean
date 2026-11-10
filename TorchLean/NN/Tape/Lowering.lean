/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Graph
public import TorchLean.NN.Tape.CPU
public import TorchLean.NN.TypedGraph

/-!
# Lowering

Link the executable runtime tape `TorchLean.Tape` to the shape-indexed graphs `Tape.GraphData`
(`TorchLean.NN.TypedGraph.Data`) and `Tape.Graph` (`TorchLean.NN.Tape.Graph`).

`GraphData` stores executable forward, JVP, and VJP functions without derivative laws. `Graph`
pairs such a graph with the proof that every node satisfies the local adjointness law used by its
global backpropagation theorem. The lowering `Tape.GraphData.toTape` of `TorchLean.NN.TypedGraph`
places the stored VJP into each runtime node's `backward` closure.

## What is proved here

- Forward-pass correspondence: `Tape.GraphData.toTape` produces the same values as
  `Tape.GraphData.eval`. Its `ctx_eq_eval` and `values_eq` lemmas show that the returned context
  and runtime tape store those values in the same order.
- Backward-pass correspondence: `GraphData.backpropAllCtx` of
  `TorchLean.NN.TypedGraph.Data` is the graph's full reverse program, and
  `BackwardGraph` proves that the runtime dense reverse loop `Tape.backwardDenseFrom` computes it
  on a lowered tape (`GraphData.backwardDenseFrom_toTape`). This is an implementation-equivalence
  result. For a proof-carrying graph `g` it combines with `Tape.Graph.jvpCtx_backpropCtx_adjoint`,
  stated on `g.toData`, to give derivative correctness.

Lowered nodes emit contributions only to earlier node ids (`pid < id`, recorded as
`BackwardPidsLt`), so a reverse sweep in decreasing id order has collected every contribution to a
node before it visits that node.

## PyTorch correspondence / citations
PyTorch's eager autograd engine builds the same structure internally: an executable tape whose
nodes carry backward closures.
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

/-!
### `Result` equations

`Result` is `Except String`. The runtime backward functions are written with `do` notation, so
proofs about them repeatedly need the unfolding equations for `bind`, `pure` and `throw`. Stating
them once here lets those proofs rewrite with `simp only` and leave the `Monad` instances folded.
-/

/-- Binding a successful `Result` applies the continuation. -/
theorem Result.bind_ok {β γ : Type} (a : β) (f : β → Result γ) :
    (Except.ok a : Result β) >>= f = f a := rfl

/-- Binding a failed `Result` propagates the error. -/
theorem Result.bind_error {β γ : Type} (e : String) (f : β → Result γ) :
    (Except.error e : Result β) >>= f = Except.error e := rfl

/-- `pure` in `Result` is `Except.ok`. -/
theorem Result.pure_eq_ok {β : Type} (a : β) : (pure a : Result β) = Except.ok a := rfl

/-- `throw` in `Result` is `Except.error`. -/
theorem Result.throw_eq_error {β : Type} (e : String) :
    (throw e : Result β) = Except.error e := rfl

/--
Backward closures of `t` only point backwards: every contribution that the node stored at `id`
emits on a successful `backward` call targets a node id strictly smaller than `id`.

A reverse sweep in decreasing id order has therefore received every contribution to a node by the
time it visits that node. The proofs use the invariant to keep the contributions of prefix nodes
inside the prefix (`Tape.backwardDenseFromStep_addNode_push`).
`GraphData.backwardPidsLt_toTape` establishes it for lowered tapes, including those of
proved graphs, which lower through `Graph.toData`.
-/
def Tape.BackwardPidsLt {α : Type} [Storage α] (t : Tape (SomeTensor α)) : Prop :=
  ∀ id (n : Node (SomeTensor α)), t.getNode? id = some n →
    ∀ (d : SomeTensor α) (contribs : Array (Nat × SomeTensor α)),
      n.backward d = .ok contribs →
        ∀ {pid : Nat} {pg : SomeTensor α}, (pid, pg) ∈ contribs → pid < id

/-- Each entry of `TensorPack.erase` carries the shape recorded at its position of `ss`. -/
theorem TensorPack.shape_getElem_erase {α : Type} [Storage α] {ss : List Shape}
    (xs : TensorPack α ss) {i : Nat} (hi : i < ss.length) :
    ((TensorPack.erase (α := α) (ss := ss) xs)[i]'(by
        simpa [TensorPack.size_erase] using hi)).shape =
      ss.get ⟨i, hi⟩ := by
  simpa [SomeTensor.ofTensor] using congrArg SomeTensor.shape
    (TensorPack.get_erase (α := α) (ss := ss) xs ⟨i, hi⟩)

/-- `addLeaves` grows the tape by exactly `Γ.length` nodes. -/
theorem Tape.size_addLeaves {α : Type} [Storage α] (t : Tape (SomeTensor α)) :
    {Γ : List Shape} → (x : TensorPack α Γ) →
      (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes.size = t.nodes.size + Γ.length
  | [], .nil => by simp [addLeaves]
  | _ :: Γ, .cons x xs => by
      simp [addLeaves, Tape.leaf, Tape.addNode,
        size_addLeaves (t := { nodes := t.nodes.push _ }) (x := xs),
        Nat.add_assoc, Nat.add_comm, Array.size_push]

/-- `addLeaves` appends `Node.leaf` nodes for each input tensor, in order. -/
theorem Tape.nodes_addLeaves {α : Type} [Storage α] (t : Tape (SomeTensor α)) :
    {Γ : List Shape} → (x : TensorPack α Γ) →
      (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes =
        t.nodes ++
          (TensorPack.erase (α := α) (ss := Γ) x).map Node.leaf
  | [], .nil => by
      simp [addLeaves, TensorPack.erase]
  | _ :: Γ, .cons x xs => by
      simp [addLeaves, Tape.leaf, Tape.addNode,
        nodes_addLeaves (t := { nodes := t.nodes.push _ }) (Γ := Γ) (x := xs),
        Node.leaf,
        TensorPack.erase_cons (α := α) (ss := Γ) x xs,
        Array.map_append, Array.append_singleton_assoc]

/-- Value projection of `nodes_addLeaves`: `node.value` agrees with `TensorPack.erase` for added
leaves. -/
theorem Tape.addLeaves_values {α : Type} [Storage α] (t : Tape (SomeTensor α)) {Γ : List Shape}
    (x : TensorPack α Γ) :
    (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes.map (fun node => node.value) =
      t.nodes.map (fun node => node.value) ++
        TensorPack.erase (α := α) (ss := Γ) x := by
  -- A leaf stores its tensor unchanged, so projecting `value` off the leaf layout is the identity.
  simp [nodes_addLeaves, Array.map_map, Function.comp_def, Node.leaf]

/-- The lowered node stores the node's forward value. -/
@[simp] theorem Tape.NodeData.toNode_value {α : Type} {Δ : Type} [Storage α]
    {Γ : List Shape} {τ : Shape}
    (name : Option String) (node : NodeData α Δ Γ τ) (ctx : TensorPack α Γ)
    (d : Δ) :
    (toNode (α := α) name node ctx d).value = SomeTensor.ofTensor (node.forward ctx d) :=
  rfl

/-- Lowered nodes always take part in gradient accumulation. -/
@[simp] theorem Tape.NodeData.toNode_requiresGrad {α : Type} {Δ : Type} [Storage α] {Γ : List Shape}
    {τ : Shape} (name : Option String) (node : NodeData α Δ Γ τ)
    (ctx : TensorPack α Γ) (d : Δ) :
    (toNode (α := α) name node ctx d).requiresGrad = true :=
  rfl

/-- On an upstream cotangent of the right shape, the lowered `backward` runs the stored `vjp`. -/
theorem Tape.NodeData.toNode_backward_of_shape {α : Type} {Δ : Type} [Storage α] {Γ : List Shape}
    {τ : Shape} (name : Option String) (node : NodeData α Δ Γ τ)
    (ctx : TensorPack α Γ) (d : Δ) (v : SomeTensor α) (h : v.shape = τ) :
    (toNode (α := α) name node ctx d).backward v =
      .ok (TensorPack.eraseIndexed (α := α) (ss := Γ)
        (node.vjp ctx d (v.cast h)) 0) := by
  simp only [toNode, h, dite_true]

/-- On an upstream cotangent of the wrong shape, the lowered `backward` fails. -/
theorem Tape.NodeData.toNode_backward_of_ne {α : Type} {Δ : Type} [Storage α] {Γ : List Shape}
    {τ : Shape} (name : Option String) (node : NodeData α Δ Γ τ)
    (ctx : TensorPack α Γ) (d : Δ) (v : SomeTensor α) (h : v.shape ≠ τ) :
    (toNode (α := α) name node ctx d).backward v =
      .error "autograd: upstream gradient shape mismatch" := by
  simp only [toNode, h, dite_false]

/-!
### Forward-pass correspondence

`Tape.GraphData.toTape` preserves the executable forward semantics, and the runtime tape it builds
holds the evaluated context as shape-erased tensors, in order.
-/

/-- The context returned by `toTape` agrees with `GraphData.eval`. -/
theorem Tape.GraphData.toTape_ctx_eq_eval {α : Type} {Δ : Type}
    [Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TensorPack α Γ)
    (d : Δ) :
    (toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 =
      GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d := by
  induction g with
  | nil =>
      simp [toTape, GraphData.eval]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [toTape, GraphData.eval, ih]

/-- The lowered tape's `.value` array is `GraphData.eval` with shapes erased in the same order. -/
theorem Tape.GraphData.toTape_values_eq {α : Type} {Δ : Type}
    [Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TensorPack α Γ)
    (d : Δ) :
    (toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.map (fun node =>
      node.value) =
      TensorPack.erase (α := α) (ss := Γ ++ ss)
        (toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 := by
  induction g with
  | nil =>
      -- An empty graph lowers to leaves only.
      simp [toTape, addLeaves_values, Tape.empty]
  | snoc g _node ih =>
      rename_i ssPrev τ
      simp [toTape, Tape.addNode, ih]

/-- Size bookkeeping: the lowered tape contains one runtime node for each element of `Γ ++ ss`. -/
theorem Tape.GraphData.toTape_nodes_size {α : Type} {Δ : Type}
    [Storage α]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TensorPack α Γ)
    (d : Δ) :
    (toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.size =
      Γ.length + ss.length := by
  induction g with
  | nil =>
      -- An empty graph lowers to leaves only.
      simp [toTape, size_addLeaves, Tape.empty]
  | snoc g _node ih =>
      rename_i ssPrev τ
      simp [toTape, Tape.addNode, ih, Array.size_push, Nat.add_assoc]

end TorchLean
