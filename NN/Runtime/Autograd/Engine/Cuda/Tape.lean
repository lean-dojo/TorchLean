/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA-native autograd tape (float32 buffers).

This is the GPU/runtime analogue of `NN.Runtime.Autograd.Engine.Core`, but specialized to
`Runtime.Autograd.Cuda.Buffer` values plus runtime `Spec.Shape` metadata.

Notes:
- The tape is pure: we use `Except String` for errors (no `IO` exceptions).
- Buffers are assumed to be contiguous float32 arrays of length `Shape.size s`.
- Backprop here is "dense": it returns a gradient buffer for every node id (zeros for unused).
- This module intentionally focuses on tape machinery; differentiable CUDA ops live in
  `NN.Runtime.Autograd.Engine.Cuda.Ops`.
-/

module


public import NN.Runtime.Context
public import NN.Runtime.Autograd.Engine.Cuda.Buffer

/-!
# CUDA Autograd Tape

Shape-erased CUDA tape machinery for float32 `Cuda.Buffer` values. This is the GPU/runtime analogue
of the CPU autograd tape: it records node ids, runtime shapes, parent links, and backward callbacks,
then performs dense reverse-mode accumulation over buffers.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec

/-- Pure error monad for the CUDA tape. Mirrors `Engine/Core`. -/
abbrev Result (α : Type) := Except String α

/--
Runtime, shape-erased CUDA buffer.

This plays the same role as `Runtime.AnyTensor` in the CPU tape: it pairs runtime `Shape`
metadata with an opaque `Cuda.Buffer` handle.
-/
structure AnyBuffer where
  /-- Runtime shape metadata for the buffer. -/
  s : Shape
  /-- Device buffer (float32). Length is expected to be `Shape.size s`. -/
  buf : Buffer

namespace AnyBuffer

/-- Checked conversion of a `Nat` length to `UInt32`, erroring on overflow. -/
def natToU32Checked (n : Nat) : Result UInt32 := do
  let u := UInt32.ofNat n
  if u.toNat = n then
    pure u
  else
    throw "autograd: tensor too large (numel does not fit in UInt32)"

/-- Number of scalar elements as a `UInt32` (checked). -/
def numelU32 (s : Shape) : Result UInt32 :=
  natToU32Checked (Shape.size s)

/-- Allocate a zero-filled buffer for a given shape. -/
def zeros (s : Shape) : Result AnyBuffer := do
  let n ← numelU32 s
  pure { s := s, buf := Buffer.zeros n }

/--
Accumulate two `AnyBuffer` values by elementwise addition, with a dynamic shape check.

This is used by backprop to sum gradient contributions in DAGs.
-/
def add (a b : AnyBuffer) : Result AnyBuffer := by
  if decide (a.s = b.s) then
    let expected := Shape.size a.s
    if _hExpected : expected ≤ UInt32.size then
      let expectedU32 : UInt32 := UInt32.ofNat expected
      let aSize := Buffer.size a.buf
      let bSize := Buffer.size b.buf
      if aSize = expectedU32 && bSize = expectedU32 then
        exact .ok { s := a.s, buf := Buffer.add a.buf b.buf }
      else
        exact .error
          s!"autograd: native gradient buffer size mismatch during accumulation \
             (shape elements={expected}, left={aSize.toNat}, right={bSize.toNat})"
    else
      exact .error "autograd: tensor too large for CUDA gradient accumulation"
  else
    exact .error "autograd: gradient shape mismatch during accumulation"

end AnyBuffer

/--
CUDA tape node representing one recorded computation step.

Fields mirror the CPU `Engine/Core` node:
- `value` holds the forward buffer.
- `parents` are tape node ids.
- `backward` is a local VJP rule producing parent gradient contributions.
-/
structure Node where
  /-- Optional node name for debugging/pretty-printing. -/
  name : Option String := none
  /-- Forward value computed at this node. -/
  value : AnyBuffer
  /-- Whether reverse-mode propagation should visit this node. -/
  requires_grad : Bool := true
  /-- Parent node ids (dependencies) in the tape. -/
  parents : List Nat := []
  /--
  Forward scratch buffers retained only because this node's backward closure may need them.

  The eager runtime releases these buffers explicitly after backprop consumes the tape. This keeps
  long CUDA training loops from waiting on Lean external-object finalizers for large temporary
  allocations.
  -/
  cleanup : List Buffer := []
  /--
  Local VJP rule for this node.

  Given an upstream cotangent for `value`, return a list of `(parentId, parentCotangent)`
  contributions (one per parent, usually).
  -/
  backward : AnyBuffer → Result (List (Nat × AnyBuffer))

/-- CUDA autograd tape: a grow-only array of nodes. Node ids are array indices. -/
structure Tape where
  /-- Tape nodes in evaluation order (id = index). -/
  nodes : Array Node := #[]

namespace Tape

/-- Empty tape (no nodes). -/
def empty : Tape := {}

/-- Number of nodes stored in the tape. -/
def size (t : Tape) : Nat := t.nodes.size

/-- Read a node by id (returns `none` if out of bounds). -/
def getNode? (t : Tape) (id : Nat) : Option Node :=
  t.nodes[id]?

/-- Read just the stored forward value for a node id. -/
def getValue? (t : Tape) (id : Nat) : Option AnyBuffer :=
  (t.getNode? id).map (·.value)

/--
Append a node and return its id.

Invariant: the returned id is `t.size`, the pre-append size of the tape.
-/
def addNode (t : Tape) (node : Node) : Tape × Nat :=
  let id := t.nodes.size
  ({ nodes := t.nodes.push node }, id)

/-- `addNode` returns the current tape size as the fresh node id. -/
@[simp] theorem addNode_id (t : Tape) (node : Node) :
    (t.addNode node).2 = t.size := by
  simp [addNode, size]

/-- Appending a node increases the tape size by one. -/
@[simp] theorem size_addNode (t : Tape) (node : Node) :
    (t.addNode node).1.size = t.size + 1 := by
  simp [addNode, size]

/--
Add a leaf node (no parents).

PyTorch comparison: a tensor that enters the graph as a leaf (e.g. input or parameter value).
-/
def leaf (t : Tape) (value : AnyBuffer) (name : Option String := none) (requires_grad : Bool := true) :
    Tape × Nat :=
  t.addNode
    { name := name
      value := value
      requires_grad := requires_grad
      parents := []
      backward := fun _ => .ok [] }

/--
Read a buffer value from a tape node id, requiring a specific runtime shape.

Fails if:
- the id is invalid, or
- the stored shape does not match `s`.
-/
def requireValue (t : Tape) (id : Nat) (s : Shape) : Result Buffer := do
  match t.getValue? id with
  | none => throw "autograd: invalid node id"
  | some any =>
      if decide (any.s = s) then
        pure any.buf
      else
        throw "autograd: shape mismatch"

/--
Require that an upstream gradient matches an expected runtime shape.

This is used inside backward closures to validate/cast the incoming cotangent.

Error message must remain identical to the pre-refactor in all call sites.
-/
def requireGrad (dLdyAny : AnyBuffer) (expected : Shape) : Result AnyBuffer := do
  if decide (dLdyAny.s = expected) then
    pure { s := expected, buf := dLdyAny.buf }
  else
    throw "autograd: upstream gradient shape mismatch"

/--
Generic constructor for unary ops.

You provide:
- `forward : Buffer → Buffer`
- `backward : Buffer → Buffer → Buffer` (VJP; given input `x` and upstream `dLdy`, return `dLdx`)

Shapes are explicit and checked dynamically.
-/
def unary
    (t : Tape) (opName : String) (xId : Nat) (σ τ : Shape)
    (forward : Buffer → Buffer)
    (backward : Buffer → Buffer → Buffer) :
    Result (Tape × Nat) := do
  let x ← requireValue (t := t) xId σ
  let y := forward x
  let node : Node :=
    { name := some opName
      value := { s := τ, buf := y }
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny τ
        let dx := backward x dLdy.buf
        pure [(xId, { s := σ, buf := dx })] }
  pure (t.addNode node)

/--
Generic constructor for binary ops.

You provide:
- `forward : Buffer → Buffer → Buffer`
- `backward : Buffer → Buffer → Buffer → (Buffer × Buffer)` (VJP; given inputs `a`, `b`, and
  upstream `dLdy`, return `(dLda, dLdb)`)

Shapes are explicit and checked dynamically.
-/
def binary
    (t : Tape) (opName : String) (aId bId : Nat) (σ₁ σ₂ τ : Shape)
    (forward : Buffer → Buffer → Buffer)
    (backward : Buffer → Buffer → Buffer → (Buffer × Buffer)) :
    Result (Tape × Nat) := do
  let a ← requireValue (t := t) aId σ₁
  let b ← requireValue (t := t) bId σ₂
  let y := forward a b
  let node : Node :=
    { name := some opName
      value := { s := τ, buf := y }
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny τ
        let (da, db) := backward a b dLdy.buf
        pure [(aId, { s := σ₁, buf := da }), (bId, { s := σ₂, buf := db })] }
  pure (t.addNode node)

/-
Backpropagation (dense gradients)
-/

/--
Internal helper: add a gradient contribution `g` into the dense gradient array at `id`.

This checks:
- `id` is a valid node id,
- the parent requires gradients,
- the contribution shape matches the parent's value shape,
then accumulates via `AnyBuffer.add`.

When the parent does not require gradients, we still explicitly release the native contribution
buffer. The small branch below makes the returned flag observable to Lean, which prevents the FFI
release call from being optimized away as an unused pure value.
-/
def addGradAll (t : Tape) (grads : Array AnyBuffer) (id : Nat) (g : AnyBuffer) :
    Result (Array AnyBuffer) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: invalid parent id during backward"
  if node.requires_grad = false then
    let released := Buffer.release g.buf
    if released == 0 || released != 0 then
      pure grads
    else
      pure grads
  else if decide (g.s = node.value.s) then
    let g' : AnyBuffer := { s := node.value.s, buf := g.buf }
    let expected := Shape.size node.value.s
    if _hExpected : expected ≤ UInt32.size then
      let expectedU32 : UInt32 := UInt32.ofNat expected
      let gSize := Buffer.size g'.buf
      if gSize != expectedU32 then
        throw
          s!"autograd: native gradient contribution size mismatch \
             (parent id={id}, parent name={node.name}, shape elements={expected}, got={gSize.toNat})"
    else
      throw "autograd: tensor too large for CUDA gradient accumulation"
    let existing ← match grads[id]? with
      | some e => pure e
      | none => throw "autograd: internal error (gradient array out of bounds)"
    if decide (existing.s = node.value.s) then
      let existing' : AnyBuffer := { s := node.value.s, buf := existing.buf }
      let existingSize := Buffer.size existing'.buf
      let expected := Shape.size node.value.s
      if _hExpected : expected ≤ UInt32.size then
        let expectedU32 : UInt32 := UInt32.ofNat expected
        if existingSize != expectedU32 then
          throw
            s!"autograd: native accumulated gradient size mismatch \
               (parent id={id}, parent name={node.name}, shape elements={expected}, got={existingSize.toNat})"
      else
        throw "autograd: tensor too large for CUDA gradient accumulation"
      let summedRaw ← AnyBuffer.add existing' g'
      let summed : AnyBuffer :=
        { s := summedRaw.s
          buf := Buffer.releaseThen existing'.buf <| Buffer.releaseThen g'.buf summedRaw.buf }
      if hid : id < grads.size then
        pure (grads.set id summed (h := hid))
      else
        throw "autograd: internal error (gradient array out of bounds)"
    else
      throw "autograd: gradient array has wrong shape for node"
  else
    throw "autograd: gradient contribution has wrong shape for parent"

/--
One reverse-mode backprop step at a single node id, updating the dense gradient array.

The incoming array is total: every tape node has a gradient buffer, initialized to zero unless a
later node has already contributed to it. That total representation is convenient for the CUDA
runtime because every slot has a concrete device buffer that can be released deterministically.
-/
def backwardDenseFromStep (t : Tape) (acc : Array AnyBuffer) (id : Nat) : Result (Array AnyBuffer) :=
  do
    let node ← match t.getNode? id with
      | some n => pure n
      | none => throw "autograd: internal error (node missing)"
    if node.requires_grad = false then
      pure acc
    else
      let dLdyAny ← match acc[id]? with
        | some g => pure g
        | none => throw "autograd: internal error (gradient array out of bounds)"
      if decide (dLdyAny.s = node.value.s) then
        let dLdy : AnyBuffer := { s := node.value.s, buf := dLdyAny.buf }
        let contribs ← node.backward dLdy
        contribs.foldlM (fun acc2 (pid, pg) => addGradAll (t := t) acc2 pid pg) acc
      else
        throw "autograd: gradient array has wrong shape for node"

/--
Reverse-mode accumulation over the first `n` nodes in reverse order.

The recursion visits `n-1, n-2, ..., 0`; using `n = t.size` runs the full tape. We keep this as a
structural loop rather than a list fold so proof-facing callers can reason about one node step at a
time.
-/
def backwardDenseFromLoop (t : Tape) : Nat → Array AnyBuffer → Result (Array AnyBuffer)
  | 0, acc => pure acc
  | n + 1, acc => do
      let acc' ← backwardDenseFromStep (t := t) acc n
      backwardDenseFromLoop (t := t) n acc'

/--
Reverse-mode accumulation starting from an explicit dense gradient array.

This expects `grads0.size = t.size`. The function is useful for callers that already seeded
multiple outputs or want to run a custom cotangent initialization.
-/
def backwardDenseFrom (t : Tape) (grads0 : Array AnyBuffer) : Result (Array AnyBuffer) := do
  if grads0.size = t.nodes.size then
    backwardDenseFromLoop (t := t) t.nodes.size grads0
  else
    throw "autograd: initial dense gradient array has wrong length"

/--
Reverse-mode accumulation that returns a dense gradient buffer for every node id.

All gradients are initialized to zeros (using each node's runtime `Shape`), then we seed the
output node with `seed` and traverse the tape in reverse order.
-/
def backwardDenseAll (t : Tape) (outId : Nat) (seed : AnyBuffer) : Result (Array AnyBuffer) := do
  let outNode ← match t.getNode? outId with
    | some n => pure n
    | none => throw "autograd: invalid output id"
  if decide (seed.s = outNode.value.s) then
    let seed' : AnyBuffer := { s := outNode.value.s, buf := seed.buf }
    let mut grads : Array AnyBuffer := #[]
    for node in t.nodes do
      let z ← AnyBuffer.zeros node.value.s
      grads := grads.push z
    if hout : outId < grads.size then
      let previousSeedSlot := grads[outId]
      let released := Buffer.release previousSeedSlot.buf
      if released == 0 || released != 0 then
        grads := grads.set outId seed' (h := hout)
      else
        throw "autograd: internal error (buffer release impossible branch)"
    else
      throw "autograd: invalid output id"
    backwardDenseFrom (t := t) grads
  else
    throw "autograd: seed gradient shape mismatch for output"

end Tape

end Cuda
end Autograd
end Runtime
