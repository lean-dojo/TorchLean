/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Operator

/-!
# IR Graph

`NN.IR.Graph` is TorchLean’s canonical *op-tagged* DAG IR.

Today it is used as the shared target for:
- TorchLean to verifier lowering (`NN/Verification/Builtin/Lowering.lean`),
- bound-propagation / verification tooling (CROWN/LiRPA) (`NN/MLTheory/CROWN/Graph.lean`),
- IR → PyTorch emission (`NN/Runtime/PyTorch/Export/IRPyTorch.lean`),
- compact example graphs (e.g. `NN/Examples/DeepDives/GraphSpec/Tutorial.lean`).

`NN.IR.Operator` defines the operations and their static attributes. This file adds node identities,
dependencies, and declared output shapes. Parameter payloads (weights, biases, and constants) live
in backend-specific stores keyed by node id. This split keeps one graph format usable across:

- verification (where parameters often carry additional metadata like bounds or perturbation sets),
- export (where parameters may be emitted as PyTorch `nn.Parameter`s or ONNX initializers),
- and runtime execution/tracing (where parameters may already live in a separate module state).

Like a PyTorch FX graph or TorchScript IR, nodes are operations, edges are data dependencies, and
execution follows topological order. TorchLean additionally attaches explicit *shape* metadata to
every node for verification and proofs.

References / related systems:
- PyTorch FX docs: https://pytorch.org/docs/stable/fx.html
- TorchScript overview: https://pytorch.org/docs/stable/jit.html
- ONNX (graph + initializers as separate parameter store): https://onnx.ai/

## Conventions (important)

- **Topo order**: a node only references parents with smaller ids.
- **Id discipline**: checked graphs require `node.id` to equal its index in `Graph.nodes`.
  TorchLean lowering uses `freshId := nodes.size` and then appends.
- **External parameters**:
  - `OpKind.const` stores its `valueShape` here, but the constant value is stored externally
    (e.g. in a verifier `ParamStore` keyed by node id).
  - Some ops (notably `OpKind.linear` and `OpKind.conv`) typically use *external* parameter stores
    keyed by node id; in those cases the node’s `parents` array only contains the *runtime inputs*
    (e.g. the activation input `x`), not the weights/bias tensors.

This file does **not** implement evaluation or shape inference. Those live in:
- `NN/IR/Semantics.lean` (evaluation semantics for a chosen scalar backend),
- `NN/IR/Infer.lean` / `NN/IR/Check.lean` (shape inference/checking utilities),
- and backend-specific passes (verification/export) that interpret `OpKind` in their own setting.
-/

@[expose] public section


namespace NN.IR

open Spec TorchLean

/-- Node in the graph. Edges are implicit via parent indices. -/
structure Node where
  /-- Node id. Structural validation requires this to equal the index in `Graph.nodes`. -/
  id       : Nat
  /-- Parent node ids, i.e. data dependencies. Each parent must be smaller than `id`. -/
  parents  : Array Nat
  /-- Operation tag and any operation-local metadata. -/
  kind     : OpKind
  /-- Declared output shape. `NN.IR.Infer` can recompute/check this from parents. -/
  outShape : Shape
  deriving Repr

namespace Node

/-- Check the basic parent-count convention for this node kind. -/
def hasValidArity (n : Node) : Bool :=
  let p := n.parents.size
  match n.kind.maxParents? with
  | some hi => (n.kind.minParents ≤ p) && (p ≤ hi)
  | none => (n.kind.minParents ≤ p)

/--
Check that every parent id is strictly smaller than this node id (topological order).

This is the single most important invariant for the IR:
- it guarantees acyclicity,
- it makes evaluation/inference a simple left-to-right pass,
- and it makes backends predictable (no hidden recursion or “graph rewriting during execution”).
-/
def parentsBelow (n : Node) : Bool :=
  n.parents.all (fun pid => pid < n.id)

/-- Render a compact, user-facing summary (useful in error messages). -/
def summary (n : Node) : String :=
  s!"Node(id={n.id}, kind={n.kind.describe}, parents={n.parents}, outShape={repr n.outShape})"

end Node

/-- Return the sole parent id when an IR node has unary arity. -/
def unaryParent? (parents : Array Nat) : Option Nat :=
  if parents.size = 1 then parents[0]? else none

/-- Return both parent ids when an IR node has binary arity. -/
def binaryParents? (parents : Array Nat) : Option (Nat × Nat) :=
  if parents.size = 2 then some (parents[0]!, parents[1]!) else none

/-- The parent returned by the unary decoder belongs to the source array. -/
theorem mem_of_unaryParent?_eq_some {parents : Array Nat} {parent : Nat}
    (h : unaryParent? parents = some parent) : parent ∈ parents := by
  simp only [unaryParent?] at h
  split at h
  next => exact Array.mem_iff_getElem?.2 ⟨0, h⟩
  next => simp_all

/-- The first parent returned by the binary decoder belongs to the source array. -/
theorem fst_mem_of_binaryParents?_eq_some {parents : Array Nat} {left right : Nat}
    (h : binaryParents? parents = some (left, right)) : left ∈ parents := by
  simp only [binaryParents?] at h
  split at h
  next hsize =>
    have hp : (parents[0]!, parents[1]!) = (left, right) := Option.some.inj h
    have hleft : parents[0]! = left := by simpa using congrArg Prod.fst hp
    have hzero : 0 < parents.size := by simp [hsize]
    have hleft' : parents[0] = left := by simpa [getElem!_pos parents 0 hzero] using hleft
    rw [← hleft']
    exact Array.getElem_mem hzero
  next => simp_all

/-- The second parent returned by the binary decoder belongs to the source array. -/
theorem snd_mem_of_binaryParents?_eq_some {parents : Array Nat} {left right : Nat}
    (h : binaryParents? parents = some (left, right)) : right ∈ parents := by
  simp only [binaryParents?] at h
  split at h
  next hsize =>
    have hp : (parents[0]!, parents[1]!) = (left, right) := Option.some.inj h
    have hright : parents[1]! = right := by simpa using congrArg Prod.snd hp
    have hone : 1 < parents.size := by simp [hsize]
    have hright' : parents[1] = right := by simpa [getElem!_pos parents 1 hone] using hright
    rw [← hright']
    exact Array.getElem_mem hone
  next => simp_all

/-- Entire graph as an array of nodes. Parents must have smaller ids (topo order). -/
structure Graph where
  /-- The nodes, in topological order: every parent id is strictly smaller than the index of the
  node referencing it. Evaluation is then a single left-to-right pass with no scheduling step. -/
  nodes : Array Node
  deriving Repr

namespace Graph

/-- Number of nodes in the graph. -/
def size (g : Graph) : Nat :=
  g.nodes.size

/-- Safe node lookup by id (treating ids as array indices). -/
def getNode? (g : Graph) (id : Nat) : Option Node :=
  g.nodes[id]?

/--
Total node lookup that enforces the common "id discipline" invariant
$\mathrm{nodes}[i].\mathrm{id}=i$.

This is convenient for backends that treat node ids as array indices (verifiers, exporters, pretty
printers). The error message is meant to point to a builder bug rather than a user error.
-/
def getNode (g : Graph) (id : Nat) : Except String Node := do
  match g.getNode? id with
  | none => throw s!"IR graph: node id out of bounds: {id}"
  | some n =>
      if n.id != id then
        throw s!"IR graph: internal error: nodes[{id}].id = {n.id} (expected {id})"
      pure n

/-- A successful checked lookup returns a node whose stored id is the requested array index. -/
theorem getNode_id_eq {g : Graph} {id : Nat} {node : Node}
    (h : g.getNode id = .ok node) : node.id = id := by
  unfold getNode at h
  split at h
  · contradiction
  · rename_i found hFound
    split at h
    · contradiction
    · rename_i hId
      have hn : found.id = id := by simpa using hId
      change Except.ok found = Except.ok node at h
      injection h with hNode
      subst node
      exact hn

/--
Explain why `Node.hasValidArity` failed.

This returns a human-facing message rather than structured data; callers use it for diagnostics.
-/
def arityError (n : Node) : String :=
  let got := n.parents.size
  match n.kind.maxParents? with
  | some hi =>
      s!"bad parent count for {n.kind.tag}: expected {n.kind.minParents}..{hi}, got {got}"
  | none =>
      s!"bad parent count for {n.kind.tag}: expected at least {n.kind.minParents}, got {got}"

/--
Basic well-formedness check used by verifier code paths.

This checks:
- node ids match array indices (common construction invariant),
- each node respects its op arity convention, and
- all parent ids are strictly smaller than the node id (topological order).

We keep this as a boolean predicate because some passes want a fast “yes/no” filter. If you need a
human-facing error, use `checkWellFormed`.
-/
def wellFormed (g : Graph) : Bool :=
  (List.finRange g.nodes.size).all (fun i =>
    match g.nodes[i]? with
    | none => false
    | some n => (n.id = i) && n.hasValidArity && n.parentsBelow)

/--
Like `wellFormed`, but returns a helpful error message on failure.

This is useful when you want a *clean* user error rather than a silent `false`.
-/
def checkWellFormed (g : Graph) : Except String Unit := do
  for i in [0:g.nodes.size] do
    match g.nodes[i]? with
    | none =>
        throw s!"IR graph: internal error: missing node at index {i}"
    | some n =>
        if n.id != i then
          throw s!"IR graph: id discipline violated at index {i}: nodes[{i}].id = {n.id}"
        if !n.hasValidArity then
          throw s!"IR graph: node {i}: {arityError n} ({n.summary})"
        -- Because we have `n.id = i`, checking `pid < n.id` also implies `pid` is in-bounds.
        for pid in n.parents do
          if pid ≥ n.id then
            throw s!"IR graph: node {i}: parent id {pid} is not < {n.id} ({n.summary})"

end Graph

/-- Default node used only to satisfy generic container APIs; real graphs should not rely on it. -/
instance : Inhabited Node where
  default := { id := 0, parents := #[], kind := OpKind.input, outShape := Shape.scalar }

end NN.IR
