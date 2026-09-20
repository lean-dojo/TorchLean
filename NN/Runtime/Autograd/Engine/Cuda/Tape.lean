/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA-native autograd tape (float32 buffers).

This is the GPU/runtime analogue of `NN.Runtime.Autograd.Engine.Core`, but specialized to
`Runtime.Autograd.Cuda.Buffer` values plus runtime `Spec.Shape` metadata.

Notes:
- The tape is pure: we use `Except String` for errors (no `IO` exceptions).
- Buffers are contiguous float32 arrays. Shape-erased inputs are checked against
  `Spec.Shape.size s` before shape-derived indexing or reverse-mode propagation.
- Backprop supports both a dense result for diagnostics and a sparse, explicitly owned result for
  training paths that only retain parameter gradients.
- This module contains tape machinery; differentiable CUDA ops live in
  `NN.Runtime.Autograd.Engine.Cuda.Ops`.
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Buffer
import Mathlib.Tactic.Bound.Init
public import NN.Spec.Core.Shape

/-!
# CUDA Autograd Tape

Shape-erased CUDA tape machinery for float32 `Cuda.Buffer` values. This is the GPU/runtime analogue
of the CPU autograd tape: it records node ids, runtime shapes, parent links, and backward callbacks,
then performs reverse-mode accumulation over buffers.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec TorchLean

/-- Pure error monad for the CUDA tape. Mirrors `Engine/Core`. -/
abbrev Result (α : Type) := Except String α

/--
Runtime, shape-erased CUDA buffer.

This plays the same role as `Spec.SomeTensor` in the CPU tape: it pairs runtime `Shape`
metadata with an opaque `Cuda.Buffer` handle.
-/
structure AnyBuffer where
  /-- Runtime shape metadata for the buffer. -/
  s : Shape
  /-- Device buffer (float32). Length is expected to be `Spec.Shape.size s`. -/
  buf : Buffer

namespace AnyBuffer

/-- Checked conversion of a `Nat` length to `UInt32`, erroring on overflow. -/
def natToU32Checked (n : Nat) : Result UInt32 := do
  let u := UInt32.ofNat n
  if u.toNat = n then
    pure u
  else
    throw "autograd: tensor too large (numel does not fit in UInt32)"

/-- Number of scalar elements as a `UInt32`, with every individual shape dimension checked too. -/
def numelU32 (s : Shape) : Result UInt32 := do
  for dim in s.toList do
    let _ ← natToU32Checked dim
  natToU32Checked (Spec.Shape.size s)

/-- Allocate a zero-filled buffer for a given shape. -/
def zeros (s : Shape) : Result AnyBuffer := do
  let n ← numelU32 s
  pure { s := s, buf := Buffer.zeros n }

/--
Check that runtime shape metadata agrees with the native CUDA buffer length.

`AnyBuffer` is intentionally shape-erased, so its structure alone cannot enforce this invariant.
Call this at boundaries that accept buffers assembled outside the typed tensor conversion path,
before any kernel uses shape-derived indexing.
-/
def validate (value : AnyBuffer) : Result AnyBuffer := do
  let expected ← numelU32 value.s
  let actual := Buffer.size value.buf
  if actual = expected then
    pure value
  else
    throw
      s!"autograd: CUDA buffer size mismatch \
         (shape elements={expected.toNat}, native elements={actual.toNat})"

/--
Accumulate two `AnyBuffer` values by elementwise addition, with a dynamic shape check.

This is used by backprop to sum gradient contributions in DAGs. The addition runs only after the
shape metadata agree and both native buffer lengths match that shape.
-/
def add (a b : AnyBuffer) : Result AnyBuffer := by
  if decide (a.s = b.s) then
    let expected := Spec.Shape.size a.s
    match numelU32 a.s with
    | .error _ =>
        exact .error "autograd: tensor too large for CUDA gradient accumulation"
    | .ok expectedU32 =>
        let aSize := Buffer.size a.buf
        let bSize := Buffer.size b.buf
        if aSize = expectedU32 && bSize = expectedU32 then
          exact .ok { s := a.s, buf := Buffer.add a.buf b.buf }
        else
          exact .error
            s!"autograd: native gradient buffer size mismatch during accumulation \
               (shape elements={expected}, left={aSize.toNat}, right={bSize.toNat})"
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
  /--
  Whether this node owns its forward allocation.

  Shape views and detached references borrow a parent's buffer. Releasing a borrowed value would
  also invalidate the parent, including parameter mirrors that must survive a tape reset.
  -/
  ownsValue : Bool := true
  /-- Whether reverse-mode propagation should visit this node. -/
  requiresGrad : Bool := true
  /-- Parent node ids (dependencies) in the tape. -/
  parents : Array Nat := #[]
  /--
  Forward workspace buffers retained only because this node's backward closure may need them.

  The eager runtime releases these buffers explicitly after backprop consumes the tape, so long CUDA
  training loops do not wait on Lean external-object finalizers for large intermediate allocations.
  -/
  cleanup : Array Buffer := #[]
  /--
  Local VJP rule for this node.

  Given an upstream cotangent for `value`, return an array of `(parentId, parentCotangent)`
  contributions (one per parent, usually).
  -/
  backward : AnyBuffer → Result (Array (Nat × AnyBuffer))

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

This low-level constructor records `value` without validating its external buffer. Code that wraps
an externally supplied `Buffer` must call `AnyBuffer.validate` first; tape operations validate
their operands again when they retrieve values.
-/
def leaf (t : Tape) (value : AnyBuffer) (name : Option String := none)
    (requiresGrad : Bool := true) : Tape × Nat :=
  t.addNode
    { name := name
      value := value
      requiresGrad := requiresGrad
      parents := #[]
      backward := fun _ => .ok #[] }

/--
Read a buffer value from a tape node id, requiring a specific runtime shape.

Fails if:
- the id is invalid, or
- the stored shape does not match `s`,
- the shape cannot be represented by the `UInt32` CUDA ABI, or
- the native buffer length differs from `Shape.size s`.
-/
def requireValue (t : Tape) (id : Nat) (s : Shape) : Result Buffer := do
  match t.getValue? id with
  | none => throw "autograd: invalid node id"
  | some any =>
      if decide (any.s = s) then
        let checked ← AnyBuffer.validate { s := s, buf := any.buf }
        pure checked.buf
      else
        throw "autograd: shape mismatch"

/--
Require that an upstream gradient matches an expected runtime shape.

This is used inside backward closures to validate/cast the incoming cotangent.
Both the shape tag and native buffer length are checked before a VJP reads the cotangent.
-/
def requireGrad (dLdyAny : AnyBuffer) (expected : Shape) : Result AnyBuffer := do
  if decide (dLdyAny.s = expected) then
    AnyBuffer.validate { s := expected, buf := dLdyAny.buf }
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
    (backward : Buffer → Buffer → Buffer)
    (ownsValue : Bool := true) :
    Result (Tape × Nat) := do
  let x ← requireValue (t := t) xId σ
  let y := forward x
  let node : Node :=
    { name := some opName
      value := { s := τ, buf := y }
      ownsValue := ownsValue
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny τ
        let dx := backward x dLdy.buf
        pure #[(xId, { s := σ, buf := dx })] }
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
      requiresGrad := true
      parents := #[aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny τ
        let (da, db) := backward a b dLdy.buf
        pure #[(aId, { s := σ₁, buf := da }), (bId, { s := σ₂, buf := db })] }
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

When the parent does not require gradients, the contribution is not stored anywhere. We still have
to release its device buffer. The release is threaded through the existing zero slot with
`releaseThen`, so the cleanup is part of the returned gradient array instead of a dead pure call.
-/
def addGradAll (t : Tape) (grads : Array AnyBuffer) (id : Nat) (g : AnyBuffer) :
    Result (Array AnyBuffer) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: invalid parent id during backward"
  if node.requiresGrad = false then
    let existing ← match grads[id]? with
      | some e => pure e
      | none => throw "autograd: internal error (gradient array out of bounds)"
    if hid : id < grads.size then
      let existing' : AnyBuffer :=
        { existing with buf := Buffer.releaseThen g.buf existing.buf }
      pure (grads.set id existing' (h := hid))
    else
      throw "autograd: internal error (gradient array out of bounds)"
  else if decide (g.s = node.value.s) then
    let g' : AnyBuffer := { s := node.value.s, buf := g.buf }
    let expected := Spec.Shape.size node.value.s
    let expectedU32 ← match AnyBuffer.numelU32 node.value.s with
      | .ok expectedU32 => pure expectedU32
      | .error _ => throw "autograd: tensor too large for CUDA gradient accumulation"
    let gSize := Buffer.size g'.buf
    if gSize != expectedU32 then
      throw
        s!"autograd: native gradient contribution size mismatch \
           (parent id={id}, parent name={node.name}, shape elements={expected}, got={gSize.toNat})"
    let existing ← match grads[id]? with
      | some e => pure e
      | none => throw "autograd: internal error (gradient array out of bounds)"
    if decide (existing.s = node.value.s) then
      let existing' : AnyBuffer := { s := node.value.s, buf := existing.buf }
      let existingSize := Buffer.size existing'.buf
      let expected := Spec.Shape.size node.value.s
      let expectedU32 ← match AnyBuffer.numelU32 node.value.s with
        | .ok expectedU32 => pure expectedU32
        | .error _ => throw "autograd: tensor too large for CUDA gradient accumulation"
      if existingSize != expectedU32 then
        throw
          s!"autograd: native accumulated gradient size mismatch \
             (parent id={id}, parent name={node.name}, shape elements={expected}, \
             got={existingSize.toNat})"
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
def backwardDenseFromStep (t : Tape) (acc : Array AnyBuffer) (id : Nat) :
    Result (Array AnyBuffer) := do
    let node ← match t.getNode? id with
      | some n => pure n
      | none => throw "autograd: internal error (node missing)"
    if node.requiresGrad = false then
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
structural loop rather than a list fold so proof layer callers can reason about one node step at a
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
multiple outputs or want to run a custom cotangent initialization. Before traversal, every slot is
checked against its node's shape and native buffer length.
-/
def backwardDenseFrom (t : Tape) (grads0 : Array AnyBuffer) : Result (Array AnyBuffer) := do
  if grads0.size = t.nodes.size then
    for i in [0:t.nodes.size] do
      let node ← match t.getNode? i with
        | some node => pure node
        | none => throw "autograd: internal error (node missing)"
      let grad ← match grads0[i]? with
        | some grad => pure grad
        | none => throw "autograd: internal error (gradient array out of bounds)"
      if decide (grad.s = node.value.s) then
        let _ ← AnyBuffer.validate { s := node.value.s, buf := grad.buf }
      else
        throw "autograd: initial dense gradient has wrong shape for node"
    backwardDenseFromLoop (t := t) t.nodes.size grads0
  else
    throw "autograd: initial dense gradient array has wrong length"

/--
Run one dense reverse step only when `id` has been reached from the selected output, and mark the
parents that receive concrete VJP contributions. The gradient array remains total so consumed CUDA
buffers continue to follow the ownership discipline in `addGradAll`.
-/
def backwardDenseReachableStep
    (t : Tape) (reachable : Array Bool) (grads : Array AnyBuffer) (id : Nat) :
    Result (Array Bool × Array AnyBuffer) := do
  let isReachable ← match reachable[id]? with
    | some value => pure value
    | none => throw "autograd: internal error (reachability array out of bounds)"
  if !isReachable then
    pure (reachable, grads)
  else
    let node ← match t.getNode? id with
      | some n => pure n
      | none => throw "autograd: internal error (node missing)"
    if node.requiresGrad = false then
      pure (reachable, grads)
    else
      let dLdyAny ← match grads[id]? with
        | some g => pure g
        | none => throw "autograd: internal error (gradient array out of bounds)"
      if decide (dLdyAny.s = node.value.s) then
        let dLdy : AnyBuffer := { s := node.value.s, buf := dLdyAny.buf }
        let contribs ← node.backward dLdy
        let mut reachable' := reachable
        let mut grads' := grads
        for (parentId, contribution) in contribs do
          grads' ← addGradAll (t := t) grads' parentId contribution
          if hParent : parentId < reachable'.size then
            reachable' := reachable'.set parentId true (h := hParent)
          else
            throw "autograd: internal error (reachability array out of bounds)"
        pure (reachable', grads')
      else
        throw "autograd: gradient array has wrong shape for node"

/-- Reverse traversal used by `backwardDenseAll`, carrying the set of reached node ids. -/
def backwardDenseReachableLoop
    (t : Tape) : Nat → Array Bool → Array AnyBuffer → Result (Array AnyBuffer)
  | 0, _reachable, grads => pure grads
  | n + 1, reachable, grads => do
      let (reachable', grads') ←
        backwardDenseReachableStep (t := t) reachable grads n
      backwardDenseReachableLoop (t := t) n reachable' grads'

/--
Reverse-mode accumulation that returns a dense gradient buffer for every node id.

All gradients are initialized to zeros (using each node's runtime `Shape`), then the output node is
seeded and only nodes reached by VJP propagation are executed. Disconnected slots stay explicit
zero buffers; their backward closures are not called.
-/
def backwardDenseAll (t : Tape) (outId : Nat) (seed : AnyBuffer) : Result (Array AnyBuffer) := do
  let outNode ← match t.getNode? outId with
    | some n => pure n
    | none => throw "autograd: invalid output id"
  if decide (seed.s = outNode.value.s) then
    let _ ← AnyBuffer.validate outNode.value
    let seed' ← AnyBuffer.validate { s := outNode.value.s, buf := seed.buf }
    let mut grads : Array AnyBuffer := #[]
    for node in t.nodes do
      let z ← AnyBuffer.zeros node.value.s
      grads := grads.push z
    if hout : outId < grads.size then
      let previousSeedSlot := grads[outId]
      let seed' : AnyBuffer :=
        { seed' with buf := Buffer.releaseThen previousSeedSlot.buf seed'.buf }
      grads := grads.set outId seed' (h := hout)
      let mut reachable := Array.replicate t.nodes.size false
      if hReachable : outId < reachable.size then
        reachable := reachable.set outId true (h := hReachable)
        backwardDenseReachableLoop (t := t) t.nodes.size reachable grads
      else
        throw "autograd: internal error (reachability array out of bounds)"
    else
      throw "autograd: invalid output id"
  else
    throw "autograd: seed gradient shape mismatch for output"

/-!
Sparse gradients

Training only needs cotangents for parameter leaves. The sparse traversal below releases each
activation cotangent immediately after its local VJP has propagated it and retains owned copies
only for node ids selected by the caller.
-/

/-- Device gradients retained for selected tape node ids. -/
abbrev SparseGradMap := Std.HashMap Nat AnyBuffer

/-- Sequence a native buffer release inside `IO`. -/
def releaseSparseBuffer (b : Buffer) : IO Unit := do
  discard <| Buffer.releaseIO b

/-- Release every buffer owned by a sparse gradient map. -/
def releaseSparseGrads (grads : SparseGradMap) : IO Unit := do
  for (_id, grad) in grads.toList do
    releaseSparseBuffer grad.buf

/--
Insert or accumulate an owned VJP contribution into the sparse gradient map.

CUDA backward rules return fresh contribution buffers. The sparse map copies a first contribution
so its retained values have uniform ownership, then retires the consumed contribution immediately.
-/
def addSparseGrad (t : Tape) (gradsRef : IO.Ref SparseGradMap)
    (id : Nat) (g : AnyBuffer) : IO Unit := do
  let node ← match t.getNode? id with
    | some node => pure node
    | none =>
        releaseSparseBuffer g.buf
        throw <| IO.userError "autograd: invalid parent id during sparse CUDA backward"
  if !node.requiresGrad then
    releaseSparseBuffer g.buf
  else if _h : g.s = node.value.s then
    let contribution ← match AnyBuffer.validate { s := node.value.s, buf := g.buf } with
      | .ok contribution => pure contribution
      | .error msg =>
          releaseSparseBuffer g.buf
          throw <| IO.userError msg
    let grads ← gradsRef.get
    match grads.get? id with
    | none =>
        -- VJP rules may pass through an upstream handle. Copy-on-insert gives the map an
        -- unambiguous owner independently of local aliasing inside a backward rule.
        let owned : AnyBuffer := {
          s := contribution.s
          buf := Buffer.copyAndRelease contribution.buf }
        gradsRef.set (grads.insert id owned)
    | some old =>
        if _hOld : old.s = node.value.s then
          let old' : AnyBuffer := { s := node.value.s, buf := old.buf }
          let summed ← match AnyBuffer.add old' contribution with
            | .ok summed => pure summed
            | .error msg =>
                releaseSparseBuffer contribution.buf
                throw <| IO.userError msg
          releaseSparseBuffer old'.buf
          gradsRef.set (grads.insert id summed)
          releaseSparseBuffer contribution.buf
        else
          releaseSparseBuffer contribution.buf
          throw <| IO.userError "autograd: sparse CUDA gradient shape mismatch"
  else
    releaseSparseBuffer g.buf
    throw <| IO.userError "autograd: sparse CUDA gradient contribution has wrong shape"

/--
Run reverse mode while retaining gradients only for node ids accepted by `retain`.

This function consumes `seed`. The returned buffers are owned by the map and must be released with
`releaseSparseGrads` after the optimizer has consumed them. All other activation gradients are
retired during the traversal.
-/
def backwardSparse (t : Tape) (outId : Nat) (seed : AnyBuffer)
    (retain : Nat → Bool) : IO SparseGradMap := do
  let outNode ← match t.getNode? outId with
    | some node => pure node
    | none =>
        releaseSparseBuffer seed.buf
        throw <| IO.userError "autograd: invalid output id"
  if _h : seed.s = outNode.value.s then
    let _ ← match AnyBuffer.validate outNode.value with
      | .ok value => pure value
      | .error msg =>
          releaseSparseBuffer seed.buf
          throw <| IO.userError msg
    let seedChecked ← match AnyBuffer.validate { s := outNode.value.s, buf := seed.buf } with
      | .ok value => pure value
      | .error msg =>
          releaseSparseBuffer seed.buf
          throw <| IO.userError msg
    let seedOwned : AnyBuffer := {
      s := outNode.value.s
      buf := Buffer.copyAndRelease seedChecked.buf }
    let gradsRef ← IO.mkRef
      ((Std.HashMap.emptyWithCapacity).insert outId seedOwned : SparseGradMap)
    try
      for offset in [0:t.nodes.size] do
        let id := t.nodes.size - 1 - offset
        let grads ← gradsRef.get
        match grads.get? id with
        | none => pure ()
        | some upstream =>
            let node ← match t.getNode? id with
              | some node => pure node
              | none => throw <| IO.userError "autograd: internal sparse CUDA node missing"
            if node.requiresGrad then
              let expectedSize := Spec.Shape.size node.value.s
              let actualSize := (← Buffer.sizeIO upstream.buf).toNat
              if actualSize != expectedSize then
                let nodeName := node.name.getD "<unnamed>"
                throw <| IO.userError
                  s!"autograd: sparse CUDA gradient buffer size mismatch \
                     at node {id} ({nodeName}): expected {expectedSize}, got {actualSize}"
              let contributions ← match node.backward upstream with
                | .ok contributions => pure contributions
                | .error msg => throw <| IO.userError msg
              try
                for (parentId, contribution) in contributions do
                  addSparseGrad t gradsRef parentId contribution
              catch e =>
                -- Contributions already consumed by `addSparseGrad` have null native handles, so
                -- releasing the whole list is idempotent and also retires every unvisited tail.
                for (_parentId, contribution) in contributions do
                  releaseSparseBuffer contribution.buf
                throw e
            if !retain id then
              let current ← gradsRef.get
              match current.get? id with
              | none => pure ()
              | some stale =>
                  releaseSparseBuffer stale.buf
                  gradsRef.set (current.erase id)
      gradsRef.get
    catch e =>
      releaseSparseGrads (← gradsRef.get)
      throw e
  else
    releaseSparseBuffer seed.buf
    throw <| IO.userError "autograd: seed gradient shape mismatch for sparse output"

end Tape

end Cuda
end Autograd
end Runtime
