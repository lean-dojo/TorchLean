/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Base

@[expose] public section

namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-!
## Backpropagation

Reverse-mode is implemented by traversing node ids in reverse order. Each node's `backward`
closure produces parent-gradient contributions, which we accumulate by elementwise summation.

Two traversal variants live here, and it matters which one a caller runs:

* `backwardDense` (and its totalized form `backwardDenseAll`) is what the eager trainer executes.
  It keeps an `Option` per node and runs a node's VJP only when that node has received a
  cotangent, so disconnected nodes are never visited. `backwardDenseAll` then fills the
  unvisited slots with explicit zero tensors. Skipping is deliberate: on `Float`, feeding a
  synthetic zero cotangent through the VJP of a singular value can produce `NaN` via `0 * (1/0)`.
* `backwardDenseFrom` starts from a total gradient array and runs every node's VJP. It is the
  variant the proof layer reasons about directly
  (`Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx`).

The two agree whenever every VJP on the tape sends a zero cotangent to zero contributions of the
parents' shapes; that is `Proofs.Autograd.Algebra.Graph.ZeroPreserving` in
`NN.Proofs.Autograd.Runtime.Link.BackwardDense`, where `backwardDenseAll_eq_backwardDenseFrom`
is proved, and `NN.Proofs.Autograd.Runtime.Link.BackwardDenseGraph`, where it is instantiated for
every tape produced by `lowerGraphToTape`.
-/

/--
 Internal helper: add a single parent gradient contribution into the dense optional gradient array.

 This is where we implement PyTorch-style accumulation for DAGs: if multiple children contribute
 to the same parent id, we sum the contributions.

 The dense array entry is `none` until we first reach a node during reverse traversal.
 -/
def addGradDense
  {α : Type} [TorchLean.Storage α] [Add α]
  (t : Tape α) (grads : Array (Option (Spec.SomeTensor α)))
  (id : Nat) (g : Spec.SomeTensor α) : Result (Array (Option (Spec.SomeTensor α))) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: invalid parent id during backward"
  if node.requiresGrad = false then
    pure grads
  else if h : g.shape = node.value.shape then
    let g' : Spec.SomeTensor α := Spec.SomeTensor.ofTensor (g.cast h)
    if hid : id < grads.size then
      match grads[id]'hid with
      | none =>
          pure (grads.set id (some g') (h := hid))
      | some existing =>
          let summed ← SomeTensor.add existing g'
          pure (grads.set id (some summed) (h := hid))
    else
      throw "autograd: internal error (gradient array out of bounds)"
  else
    throw "autograd: gradient contribution has wrong shape for parent"

/--
Reverse-mode backpropagation producing a dense array of optional gradients.

- The result array has length `t.nodes.size`.
- Entry `id` is `some g` if the node was reached from `outId` during reverse traversal, otherwise
  `none`.
- When multiple paths contribute to the same node, we sum gradients via `SomeTensor.add`.
- A node's VJP runs only if the node was reached; see the section docstring for why. The proof
  layer names this per-node step `backwardDenseStep` and proves
  `backwardDense` is the reverse fold of it.

This is the variant the eager trainer executes (through `backwardDenseAll`). It is loosely
analogous to PyTorch's autograd engine walking the dynamic graph and accumulating `.grad` for
leaf tensors, but we keep gradients for every node id rather than leaves alone. That makes the
runtime easier to debug and gives proof-bridge code direct access to intermediate cotangents.

Reference (PyTorch): https://pytorch.org/docs/stable/notes/autograd.html
-/
def backwardDense {α : Type} [TorchLean.Storage α] [Add α]
  (t : Tape α) (outId : Nat) (seed : Spec.SomeTensor α) :
  Result (Array (Option (Spec.SomeTensor α))) := do
  let outNode ← match t.getNode? outId with
    | some n => pure n
    | none => throw "autograd: invalid output id"
  if h : seed.shape = outNode.value.shape then
    let seed' : Spec.SomeTensor α := Spec.SomeTensor.ofTensor (seed.cast h)
    let mut grads : Array (Option (Spec.SomeTensor α)) := Array.replicate t.nodes.size none
    if hout : outId < grads.size then
      grads := grads.set outId (some seed') (h := hout)
    else
      throw "autograd: invalid output id"
    let ids := (List.range t.nodes.size).reverse
    ids.foldlM (fun acc id => do
      match acc[id]? with
      | none => throw "autograd: internal error (gradient array out of bounds)"
      | some none => pure acc
      | some (some dLdy) =>
        let node ← match t.getNode? id with
          | some n => pure n
          | none => throw "autograd: internal error (node missing)"
        if node.requiresGrad = false then
          pure acc
        else
          let contribs ← node.backward dLdy
          contribs.foldlM (fun acc2 (pid, pg) => addGradDense (t:=t) acc2 pid pg) acc
    ) grads
  else
    throw "autograd: seed gradient shape mismatch for output"

/--
Internal helper: like `addGradDense`, but assumes the gradient array is total (no `Option`).

This is used by the proof-friendly `backwardDenseFrom*` variants, which start from an explicit
gradient tensor for every node.
-/
def addGradAll
  {α : Type} [TorchLean.Storage α] [Add α]
  (t : Tape α) (grads : Array (Spec.SomeTensor α))
  (id : Nat) (g : Spec.SomeTensor α) : Result (Array (Spec.SomeTensor α)) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: invalid parent id during backward"
  if node.requiresGrad = false then
    pure grads
  else if h : g.shape = node.value.shape then
    let g' : Spec.SomeTensor α := Spec.SomeTensor.ofTensor (g.cast h)
    match grads[id]? with
    | none => throw "autograd: internal error (gradient array out of bounds)"
      | some existing =>
          if hex : existing.shape = node.value.shape then
            let existing' : Spec.SomeTensor α :=
              Spec.SomeTensor.ofTensor (existing.cast hex)
            let summed ← SomeTensor.add existing' g'
            if hid : id < grads.size then
              pure (grads.set id summed (h := hid))
            else
              throw "autograd: internal error (gradient array out of bounds)"
          else
            throw "autograd: gradient array has wrong shape for node"
  else
    throw "autograd: gradient contribution has wrong shape for parent"

/--
One reverse-mode backprop step at a single node id, updating a total dense gradient array.

Precondition by convention: `acc` has one entry per tape node, and every entry has the matching
node shape. The function checks those conditions dynamically and returns an error if a caller
violates them. This makes it suitable as the small proof-friendly step used by
`backwardDenseFromLoop`.
-/
def backwardDenseFromStep {α : Type} [TorchLean.Storage α] [Add α]
  (t : Tape α) (acc : Array (Spec.SomeTensor α)) (id : Nat) :
  Result (Array (Spec.SomeTensor α)) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: internal error (node missing)"
  if node.requiresGrad = false then
    pure acc
  else
    let dLdyAny ← match acc[id]? with
      | some g => pure g
      | none => throw "autograd: internal error (gradient array out of bounds)"
    if hshape : dLdyAny.shape = node.value.shape then
      let dLdy : Spec.SomeTensor α := Spec.SomeTensor.ofTensor (dLdyAny.cast hshape)
      let contribs ← node.backward dLdy
      contribs.foldlM (fun acc2 (pid, pg) => addGradAll (t := t) acc2 pid pg) acc
    else
      throw "autograd: gradient array has wrong shape for node"

/--
Reverse-mode accumulation over the first `n` nodes in reverse order.

The recursion visits node ids `n-1, n-2, ..., 0`. Passing `n = t.nodes.size` therefore traverses the
entire tape. This structurally recursive loop is also used by typed graph sessions after lowering.
-/
def backwardDenseFromLoop {α : Type} [TorchLean.Storage α] [Add α]
  (t : Tape α) : Nat → Array (Spec.SomeTensor α) → Result (Array (Spec.SomeTensor α))
  | 0, acc => pure acc
  | n + 1, acc => do
      let acc' ← backwardDenseFromStep (t := t) acc n
      backwardDenseFromLoop (t := t) n acc'

/--
Reverse-mode accumulation starting from an explicit dense gradient array.

This is the variant the proofs reason about directly: it always runs every node's VJP (in
reverse order) and keeps a gradient tensor for every node id. The eager trainer does not call
it; it calls `backwardDenseAll`, which agrees with this function on zero-preserving tapes
(`Proofs.Autograd.Algebra.Graph.backwardDenseAll_eq_backwardDenseFrom`).
-/
def backwardDenseFrom {α : Type} [TorchLean.Storage α] [Add α]
  (t : Tape α) (grads0 : Array (Spec.SomeTensor α)) :
  Result (Array (Spec.SomeTensor α)) := do
  if grads0.size = t.nodes.size then
    backwardDenseFromLoop (t := t) t.nodes.size grads0
  else
    throw "autograd: initial dense gradient array has wrong length"

/-- Reverse-mode accumulation that returns a dense gradient array for every node id.

Propagation uses `backwardDense`, so local VJP closures run only for nodes reached from `outId`.
The optional result is then totalized with explicit zero tensors for disconnected nodes. This is
necessary at singular forward values: applying a disconnected VJP to a synthetic
zero cotangent can manufacture `NaN` through expressions such as `0 * (1 / 0)`, even though the
mathematical gradient of the selected output with respect to that node is zero.

This is the entry point the eager trainer executes. On zero-preserving tapes (in particular
every tape produced by `lowerGraphToTape`) it returns exactly what `backwardDenseFrom` returns
from the one-hot seed array; see `NN.Proofs.Autograd.Runtime.Link.BackwardDense`.
-/
def backwardDenseAll {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
  (t : Tape α) (outId : Nat) (seed : Spec.SomeTensor α) :
  Result (Array (Spec.SomeTensor α)) := do
  let reached ← backwardDense (t := t) outId seed
  pure <| t.nodes.mapIdx fun id node =>
    match reached[id]? with
    | some (some grad) => grad
    | _ => Spec.SomeTensor.ofTensor (Tensor.full node.value.shape (0 : α))

/--
Convert the optional dense gradient array returned by `backwardDense` into a sparse `HashMap`.

Only entries that are present (`some (some g)`) are kept. The result records exactly the nodes
reached by reverse-mode propagation.
-/
def denseToHashMap {α : Type} [TorchLean.Storage α]
  (grads : Array (Option (Spec.SomeTensor α))) :
  Std.HashMap Nat (Spec.SomeTensor α) :=
  (List.range grads.size).foldl (fun acc id =>
    match grads[id]? with
    | some (some g) => acc.insert id g
    | _ => acc
  ) (Std.HashMap.emptyWithCapacity)

/--
Reverse-mode backpropagation returning a `HashMap` of only the nodes that received gradients.

This is the sparse public form of `backwardDense`: it computes dense gradients first, then drops
nodes that did not receive a gradient.
-/
def backward {α : Type} [TorchLean.Storage α] [Add α]
  (t : Tape α) (outId : Nat) (seed : Spec.SomeTensor α) :
  Result (Std.HashMap Nat (Spec.SomeTensor α)) := do
  let dense ← backwardDense (t := t) outId seed
  pure (denseToHashMap dense)

/--
Backpropagate from a scalar output with seed gradient `1`.

PyTorch analogy: `loss.backward()` when `loss` is a scalar.
-/
def backwardScalar {α : Type} [TorchLean.Storage α] [Add α] [One α]
  (t : Tape α) (outId : Nat) : Result (Std.HashMap Nat (Spec.SomeTensor α)) :=
  backward (t:=t) outId (Spec.SomeTensor.ofTensor (Tensor.scalar (1 : α)))

end Tape
end Autograd
end Runtime
