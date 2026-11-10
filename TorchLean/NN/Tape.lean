/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Shape

/-!
# The autograd tape

One tape serves every device. A `Tape V` is a grow-only array of nodes whose forward values have
type `V`: the CPU stores `SomeTensor α` and CUDA stores `LibTorch.SomeBuffer`. Each node
keeps its parents and a local VJP, and the reverse sweeps below walk node ids from last to first,
summing the contributions that several children send to one parent.

The dense sweeps are shared between devices. `GradValue` supplies the device-specific gradient
checks, addition and buffer ownership, which on the CPU are pure tensor operations. CUDA training
also uses the sparse `LibTorch.Tape.backwardSparse` traversal in
`TorchLean.NN.LibTorch.Tape`, which releases unretained gradients in `IO`.

Two traversals live here, and it matters which one a caller runs:
* `backwardDense` (and its totalized form `backwardDenseAll`) keeps an `Option` per node and runs a
  node's VJP only when that node has received a cotangent, so disconnected nodes are never
  visited. Skipping is deliberate: on `Float`, feeding a synthetic zero cotangent through the VJP
  of a singular value can produce `NaN` via `0 * (1/0)`.
* `backwardDenseFrom` starts from a total gradient array and runs the VJP of every node that
  requires a gradient. It is the variant the proof layer reasons about
  (`TorchLean.Tape.GraphData.backwardDenseFrom_toTape`).

Over a scalar whose zero is an additive identity (`AddZeroClass`), the two agree when the VJP of
every node that requires a gradient sends a zero cotangent to zero contributions of the parents'
shapes. A VJP that is linear over a semiring and returns the parents' shapes does. That condition is
`TorchLean.Tape.ZeroPreserving`, and the agreement is
`TorchLean.Tape.backwardDenseAll_eq_backwardDenseFrom`, both in
`TorchLean.NN.Tape.Lowering.BackwardDense`. `Float` is not such a scalar: `+0 + -0 = +0` under
round-to-nearest (IEEE 754-2019 6.3), so on `Float` the two can return opposite signs of zero even
on a zero-preserving tape.
-/

@[expose] public section

namespace TorchLean

/--
Runtime error monad for the autograd tape.

Errors are plain `Except String` values, so the tape constructors are pure and easy to test, and
front-ends that want `IO` exceptions can use Lean's `IO.ofExcept`.
-/
abbrev Result (α : Type) := Except String α

/--
What the reverse sweep needs from the values a tape stores.

A gradient for a node has the node's shape. `retag` re-expresses a gradient whose shape has been
checked against a value, `check` performs any further device check (CUDA compares dtypes and
buffer lengths), and `add` sums two gradients of one node. Backward rules return fresh
contributions, so a node keeps its first contribution as it is. `drop` retires a contribution that
no node keeps, which on CUDA releases the device buffer at once.
-/
class GradValue (V : Type) where
  /-- Runtime shape of a value. -/
  shape : V → Shape
  /-- `g`, given the shape of `v`, once the two shapes are known to agree. -/
  retag : (g v : V) → shape g = shape v → V
  /-- Device checks of a gradient `g` against the value `v` it belongs to. -/
  check : (g v : V) → Result Unit := fun _ _ => pure ()
  /-- Sum two gradients of one node. -/
  add : V → V → Result V
  /-- Retire a contribution that no node keeps, then continue with `rest`. -/
  drop {β : Type} : V → β → β := fun _ rest => rest

/-- A zero gradient of the same shape as a value, for nodes the sweep never reached. -/
class GradZero (V : Type) where
  /-- A zero gradient shaped like `v`. -/
  zerosLike : V → V

/--
One recorded computation step.

`value` is the forward value, `parents` the ids of its inputs, and `backward` its local VJP: given
the upstream cotangent of `value`, it returns `(parentId, contribution)` pairs. When several
children contribute to one parent, the sweep sums the contributions with `GradValue.add`.

PyTorch comparison: an autograd `Function` instance with its saved tensors, where the node stores
the backward closure itself.
-/
structure Tape.Node (V : Type) where
  /-- Optional node name used for debugging and pretty-printing. -/
  name : Option String := none
  /-- Forward value computed at this node. -/
  value : V
  /--
  Whether this node owns its forward value. On CUDA, shape views and detached references borrow a
  parent's buffer, and releasing a borrowed value would also invalidate the parent.
  -/
  ownsValue : Bool := true
  /--
  Whether reverse-mode propagation should visit this node.

  If `false`, the reverse sweeps skip this node and accumulate no gradient into it. An operation's
  flag is the disjunction of its differentiable parents' flags, and a leaf takes the flag its
  caller passes.
  -/
  requiresGrad : Bool := true
  /-- Parent node ids (dependencies) in the tape. -/
  parents : Array Nat := #[]
  /--
  Forward workspace kept only because `backward` may need it. CUDA releases these buffers when the
  tape is reset, so long training loops do not wait on finalizers for large intermediates.
  -/
  cleanup : Array V := #[]
  /-- Local VJP rule for this node. -/
  backward : V → Result (Array (Nat × V))

/--
Autograd tape: a grow-only array of nodes.

Node ids are array indices (`Nat`). Recording an operation appends at most one node, after the
nodes it reads, so reverse mode traverses ids in reverse order.
-/
structure Tape (V : Type) where
  /-- Tape nodes in evaluation order (id = index). -/
  nodes : Array (Tape.Node V) := #[]

namespace Tape

variable {V : Type}

/-- Empty tape (no nodes). -/
def empty : Tape V := {}

/-- Number of nodes stored in the tape. -/
def size (t : Tape V) : Nat := t.nodes.size

/-- Read a node by id (returns `none` if out of bounds). -/
def getNode? (t : Tape V) (id : Nat) : Option (Node V) :=
  t.nodes[id]?

/-- Read just the stored forward value for a node id. -/
def getValue? (t : Tape V) (id : Nat) : Option V :=
  (t.getNode? id).map (·.value)

/--
Append a node and return its id.

Invariant: the returned id is `t.size`, the pre-append size of the tape.
-/
@[inline] def addNode (t : Tape V) (node : Node V) : Tape V × Nat :=
  let id := t.nodes.size
  ({ nodes := t.nodes.push node }, id)

/-- `addNode` returns the current tape size as the fresh node id. -/
@[simp] theorem addNode_id (t : Tape V) (node : Node V) :
    (t.addNode node).2 = t.size := by
  simp [addNode, size]

/-- Appending a node increases the tape size by one. -/
@[simp] theorem size_addNode (t : Tape V) (node : Node V) :
    (t.addNode node).1.size = t.size + 1 := by
  simp [addNode, size]

/-- The leaf node holding `value`: it has no parents and contributes nothing in backward. -/
@[inline] def Node.leaf (value : V) (name : Option String := none) (requiresGrad : Bool := true) :
    Node V :=
  { name := name
    value := value
    requiresGrad := requiresGrad
    parents := #[]
    backward := fun _ => .ok #[] }

/-- Append a leaf node holding `value`, with no parents, such as an input or a parameter value. -/
@[inline] def leaf (t : Tape V) (value : V) (name : Option String := none)
    (requiresGrad : Bool := true) : Tape V × Nat :=
  t.addNode (Node.leaf value name requiresGrad)

/-!
## Reverse sweeps
-/

variable [GradValue V]

/--
Add one parent contribution into the optional gradient array.

The entry is `none` until the sweep first reaches the node. The first contribution is stored as it
is, and later ones are summed in, in the order the sweep produces them. A contribution to a node
with `requiresGrad = false` is dropped.
-/
def addGradDense (t : Tape V) (grads : Array (Option V)) (id : Nat) (g : V) :
    Result (Array (Option V)) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: invalid parent id during backward"
  if node.requiresGrad = false then
    pure (GradValue.drop g grads)
  else if h : GradValue.shape g = GradValue.shape node.value then
    GradValue.check g node.value
    let g' := GradValue.retag g node.value h
    if hid : id < grads.size then
      match grads[id]'hid with
      | none =>
          pure (grads.set id (some g') (h := hid))
      | some existing =>
          let summed ← GradValue.add existing g'
          pure (grads.set id (some summed) (h := hid))
    else
      throw "autograd: internal error (gradient array out of bounds)"
  else
    throw "autograd: gradient contribution has wrong shape for parent"

/--
Visit one node in the optional-gradient reverse sweep.

An unreached node, or one with `requiresGrad = false`, is skipped without running its VJP. A
reached node passes each contribution of its VJP to `addGradDense`, in order.
-/
def backwardDenseStep (t : Tape V) (acc : Array (Option V)) (id : Nat) :
    Result (Array (Option V)) := do
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
      contribs.foldlM (fun acc2 (pid, pg) => addGradDense (t := t) acc2 pid pg) acc

/--
Reverse-mode backpropagation producing a dense array of optional gradients.

- The result array has length `t.nodes.size`.
- Entry `id` is `some g` if the node was reached from `outId` during reverse traversal, otherwise
  `none`.
- When multiple paths contribute to the same node, the contributions are summed.
- A node's VJP runs only if the node was reached (the module doc explains why).

It is loosely analogous to PyTorch's autograd engine walking the dynamic graph and accumulating
`.grad`, but it keeps a gradient for every node id, intermediate nodes included. That makes the
runtime easier to debug and gives proof-bridge code access to the intermediate cotangents.
-/
def backwardDense (t : Tape V) (outId : Nat) (seed : V) : Result (Array (Option V)) := do
  let outNode ← match t.getNode? outId with
    | some n => pure n
    | none => throw "autograd: invalid output id"
  if h : GradValue.shape seed = GradValue.shape outNode.value then
    GradValue.check seed outNode.value
    let seed' := GradValue.retag seed outNode.value h
    let mut grads : Array (Option V) := Array.replicate t.nodes.size none
    if hout : outId < grads.size then
      grads := grads.set outId (some seed') (h := hout)
    else
      throw "autograd: invalid output id"
    (List.range t.nodes.size).reverse.foldlM (backwardDenseStep t) grads
  else
    throw "autograd: seed gradient shape mismatch for output"

/--
Add one parent contribution into a total gradient array.

The proof-friendly `backwardDenseFrom*` variants use it, because they start with a gradient for
every node.
-/
def addGradAll (t : Tape V) (grads : Array V) (id : Nat) (g : V) : Result (Array V) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: invalid parent id during backward"
  if node.requiresGrad = false then
    pure (GradValue.drop g grads)
  else if h : GradValue.shape g = GradValue.shape node.value then
    GradValue.check g node.value
    let g' := GradValue.retag g node.value h
    match hget : grads[id]? with
    | none => throw "autograd: internal error (gradient array out of bounds)"
    | some existing =>
        if hex : GradValue.shape existing = GradValue.shape node.value then
          let existing' := GradValue.retag existing node.value hex
          let summed ← GradValue.add existing' g'
          have hid : id < grads.size := (Array.getElem?_eq_some_iff.mp hget).1
          pure (grads.set id summed (h := hid))
        else
          throw "autograd: gradient array has wrong shape for node"
  else
    throw "autograd: gradient contribution has wrong shape for parent"

/--
One reverse-mode step at a single node id, updating a total dense gradient array.

By convention `acc` has one entry per tape node, each of its node's shape. The step checks every
entry it reads and returns an error if one is missing or has the wrong shape.
`backwardDenseFromLoop` iterates this small, proof-friendly step.
-/
def backwardDenseFromStep (t : Tape V) (acc : Array V) (id : Nat) : Result (Array V) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: internal error (node missing)"
  if node.requiresGrad = false then
    pure acc
  else
    let dLdyAny ← match acc[id]? with
      | some g => pure g
      | none => throw "autograd: internal error (gradient array out of bounds)"
    if hshape : GradValue.shape dLdyAny = GradValue.shape node.value then
      GradValue.check dLdyAny node.value
      let dLdy := GradValue.retag dLdyAny node.value hshape
      let contribs ← node.backward dLdy
      contribs.foldlM (fun acc2 (pid, pg) => addGradAll (t := t) acc2 pid pg) acc
    else
      throw "autograd: gradient array has wrong shape for node"

/--
Reverse-mode accumulation over the first `n` nodes in reverse order.

The recursion visits node ids `n-1, n-2, ..., 0`, so `n = t.nodes.size` traverses the entire tape.
-/
def backwardDenseFromLoop (t : Tape V) : Nat → Array V → Result (Array V)
  | 0, acc => pure acc
  | n + 1, acc => do
      let acc' ← backwardDenseFromStep (t := t) acc n
      backwardDenseFromLoop (t := t) n acc'

/--
Reverse-mode accumulation from a dense gradient array `grads0` with one entry per node.

This runs the VJP of every node that needs a gradient (in reverse order) and keeps a gradient for
every node id. The eager runtime calls `backwardDenseAll` instead, and the two agree on
zero-preserving tapes over an `AddZeroClass` scalar
(`TorchLean.Tape.backwardDenseAll_eq_backwardDenseFrom`). On `Float` they can
differ, for example in the sign of a zero gradient.
-/
def backwardDenseFrom (t : Tape V) (grads0 : Array V) : Result (Array V) := do
  if grads0.size = t.nodes.size then
    backwardDenseFromLoop (t := t) t.nodes.size grads0
  else
    throw "autograd: initial dense gradient array has wrong length"

/--
Fill in the gradient of every node the sweep did not reach with a zero of the node's shape.
Reached nodes keep their computed gradient.
-/
def totalize [GradZero V] (t : Tape V) (grads : Array (Option V)) : Array V :=
  t.nodes.mapIdx fun id node =>
    match grads[id]? with
    | some (some grad) => grad
    | _ => GradZero.zerosLike node.value

/--
Reverse-mode accumulation that returns a gradient for every node id.

Propagation uses `backwardDense`, so VJP closures run only for nodes reached from `outId`, and
`totalize` then gives each unreached node a zero gradient. Running a disconnected VJP on a
synthetic zero cotangent at a singular forward value could manufacture `NaN` through expressions
such as `0 * (1 / 0)`, even though the mathematical gradient of the selected output with respect
to that node is zero.

Over an `AddZeroClass` scalar, on zero-preserving tapes (in particular the tape that
`GraphData.toTape` produces from the data `g.toData` of a proof-carrying `Tape.Graph` over a
commutative semiring) it returns exactly what `backwardDenseFrom` returns from the one-hot
seed array (`TorchLean.Tape.backwardDenseAll_eq_backwardDenseFrom`). On `Float` they can
differ, for example in the sign of a zero gradient.
-/
def backwardDenseAll [GradZero V] (t : Tape V) (outId : Nat) (seed : V) : Result (Array V) := do
  let reached ← backwardDense (t := t) outId seed
  pure (totalize t reached)

end Tape
end TorchLean
