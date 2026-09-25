/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.SomeTensor

/-!
# Engine Core

A small dynamic (DAG) autograd engine.

This is intended to be the "runtime" counterpart to `Spec.OpSpec`: instead of manually writing
backward passes end-to-end, you build a tape during a forward pass and then call `backward`.

Design goals:
- Works for arbitrary `Tensor α s` shapes using `Spec.SomeTensor` values.
- Supports DAGs (shared subexpressions): gradients are accumulated by summation.
- Preserves typed tensors at operator boundaries and reports dynamic shape mismatches as errors.

This module defines and executes the tape; it does not by itself prove every registered VJP.
Analytic correctness theorems for supported operations live under `NN.Proofs.Autograd`, while
runtime checks cover the integration between those operations and the tape. For the imperative
frontend over this engine, see `NN.Runtime.Autograd.Torch.Core`.

References (PyTorch / background reading):
- PyTorch docs: `torch.autograd` and "Autograd mechanics":
  https://pytorch.org/docs/stable/autograd.html
  https://pytorch.org/docs/stable/notes/autograd.html
- "micrograd" (small autograd engine, useful for intuition):
  https://github.com/karpathy/micrograd
-/

@[expose] public section


namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

/-!
## Core Types

The eager autograd engine is built out of a few small pieces:

* `Result`: a pure error monad used throughout the tape API.
* `Spec.SomeTensor`: a shape-erased tensor used to store heterogeneous node values on one tape.
* `Node`: one recorded computation step, with a local backward (VJP) rule.
* `Tape`: a grow-only array of nodes; reverse-mode traversals walk it in reverse order.
-/

/--
Runtime error monad for the eager autograd engine.

We use plain `Except String` (instead of `IO` exceptions) so the tape constructors remain pure and
easy to test. Front-ends that prefer exceptions can use `okOrThrow`.
-/
abbrev Result (α : Type) := Except String α

/--
Convert an `Autograd.Result` into an `IO` action by throwing `IO.userError` on failure.

This is mainly used by the imperative Torch/TorchLean front-ends to keep their code readable.
-/
def okOrThrow {α : Type} : Result α → IO α
  | .ok a => pure a
  | .error e => throw <| IO.userError e

namespace SomeTensor

/--
Accumulate two packed tensors by elementwise addition, with a dynamic shape check.

This is the heart of DAG support: if two different paths contribute gradients to the same parent,
we sum the contributions.
-/
def add {α : Type} [TorchLean.Storage α] [Add α]
  (a b : Spec.SomeTensor α) : Result (Spec.SomeTensor α) := by
  if h : a.shape = b.shape then
    let b' : Tensor α a.shape := b.cast h.symm
    exact .ok (Spec.SomeTensor.ofTensor (addSpec a.tensor b'))
  else
    exact .error "autograd: gradient shape mismatch during accumulation"

end SomeTensor

/--
A tape node representing a single tensor value in the recorded computation graph.

Fields:
- `value`: the forward value (shape-erased).
- `parents`: ids of parent nodes in the tape.
- `backward`: a local VJP rule. Given an upstream cotangent for `value`, it returns a list of
  `(parentId, parentCotangent)` contributions (one per parent, usually).

PyTorch comparison: analogous to an autograd `Function` instance + saved tensors, but here we store
the backward closure directly.
-/
structure Node (α : Type) [TorchLean.Storage α] where
  /-- Optional node name used for debugging and pretty-printing. -/
  name : Option String := none
  /-- Forward value computed at this node (shape-erased). -/
  value : Spec.SomeTensor α
  /--
  Whether reverse-mode propagation should visit this node.

  If `false`, reverse-mode traversal skips this node and does not accumulate gradients into it.
  Eager operations inherit the disjunction of their differentiable parents' flags; leaves retain
  the caller's explicit flag.
  -/
  requiresGrad : Bool := true
  /-- Parent node ids (dependencies) in the tape. -/
  parents : Array Nat := #[]
  /--
  Local VJP rule for this node.

  Given an upstream cotangent for `value`, return an array of `(parentId, parentCotangent)`
  contributions. If multiple children contribute to the same parent, the engine will sum
  contributions via `SomeTensor.add`.
  -/
  backward : Spec.SomeTensor α → Result (Array (Nat × Spec.SomeTensor α))

/--
Autograd tape: a grow-only array of nodes.

Node ids are array indices (`Nat`). All ops append exactly one node and return its id.
This makes it easy to implement reverse-mode by traversing ids in reverse order.
-/
structure Tape (α : Type) [TorchLean.Storage α] where
  /--
  Tape nodes in evaluation order.

  Node ids are array indices (`Nat`). Each tape op appends exactly one node and returns its id.
  -/
  nodes : Array (Node α) := #[]

/-!
## Tape Construction

The `Tape` namespace provides *pure* constructors for building a recorded computation graph.
Each op appends exactly one node and returns the updated tape plus the new node id.

If you prefer an implicit tape-threading style, see `NN.Runtime.Autograd.Engine.TapeM`.
-/

namespace Tape

variable {α : Type} [TorchLean.Storage α]

/-- Empty tape (no nodes). -/
def empty : Tape α := {}

/-- Number of nodes stored in the tape. -/
def size (t : Tape α) : Nat := t.nodes.size

/-- Read a node by id (returns `none` if out of bounds). -/
def getNode? (t : Tape α) (id : Nat) : Option (Node α) :=
  t.nodes[id]?

/-- Read just the stored forward value for a node id. -/
def getValue? (t : Tape α) (id : Nat) : Option (Spec.SomeTensor α) :=
  (t.getNode? id).map (·.value)

/--
Append a node and return its id.

Invariant: the returned id is `t.size`, the pre-append size of the tape.
-/
def addNode (t : Tape α) (node : Node α) : Tape α × Nat :=
  let id := t.nodes.size
  ({ nodes := t.nodes.push node }, id)

/-- `addNode` returns the current tape size as the fresh node id. -/
@[simp] theorem addNode_id (t : Tape α) (node : Node α) :
    (t.addNode node).2 = t.size := by
  simp [addNode, size]

/-- Appending a node increases the tape size by one. -/
@[simp] theorem size_addNode (t : Tape α) (node : Node α) :
    (t.addNode node).1.size = t.size + 1 := by
  simp [addNode, size]

/--
Add a leaf node (no parents).

PyTorch comparison: a tensor that enters the graph as a leaf (e.g. input or parameter value).
-/
def leaf {α : Type} [TorchLean.Storage α] {s : Shape}
  (t : Tape α) (value : Tensor α s) (name : Option String := none) (requiresGrad : Bool := true) :
  Tape α × Nat :=
  t.addNode {
    name := name,
    value := Spec.SomeTensor.ofTensor value,
    requiresGrad := requiresGrad,
    parents := #[],
    backward := fun _ => .ok #[]
  }

/--
Read a typed tensor value from a tape node id.

This is the main "dynamic check" boundary in the eager runtime:
- fails if the id is invalid, or
- fails if the stored runtime shape doesn't match the expected dependent shape `s`.
-/
def requireValue {α : Type} [TorchLean.Storage α] {s : Shape}
  (t : Tape α) (id : Nat) : Result (Tensor α s) := by
  match t.getValue? id with
  | none => exact .error "autograd: invalid node id"
  | some any =>
    if h : any.shape = s then
      exact .ok (any.cast h)
    else
      exact .error "autograd: shape mismatch"

/--
Read a typed upstream gradient tensor from a packed tensor.

This is the backward analogue of `Tape.requireValue`: it checks that the upstream gradient has the
expected shape `τ` and then performs the dependent cast.
-/
def requireGrad {α : Type} [TorchLean.Storage α] {τ : Shape}
    (dLdyAny : Spec.SomeTensor α) : Result (Tensor α τ) := by
  if h : dLdyAny.shape = τ then
    exact .ok (dLdyAny.cast h)
  else
    exact .error "autograd: upstream gradient shape mismatch"

/--
Generic constructor for unary ops.

You provide:
- `forward : Tensor α σ → Tensor α τ`
- `backward : Tensor α σ → Tensor α τ → Tensor α σ` (a VJP rule; note it may depend on the input)

The returned node inherits the parent's gradient flag and stores the forward value and a backward
closure that checks the upstream gradient's shape and returns the parent contribution.
-/
@[inline] def unary {α : Type} [TorchLean.Storage α] {σ τ : Shape}
  (t : Tape α) (opName : String) (xId : Nat)
  (forward : Tensor α σ → Tensor α τ)
  (backward : Tensor α σ → Tensor α τ → Tensor α σ) :
  Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=σ) xId
  let y := forward x
  let node : Node α :=
    { name := some opName
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := τ) dLdyAny
        let dLdx : Tensor α σ := backward x dLdy
        pure #[(xId, Spec.SomeTensor.ofTensor dLdx)]
    }
  pure (t.addNode node)
