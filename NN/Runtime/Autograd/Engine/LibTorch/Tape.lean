/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Buffer
public import NN.Runtime.Autograd.Engine.Tape

/-!
# CUDA tape values

The CUDA tape is `Tape AnyBuffer`: the one tape of `NN.Runtime.Autograd.Engine.Tape`, storing
device buffers with their runtime shapes. This module gives those values their `GradValue`
instance, whose accumulation also releases the device buffers it consumes, and the buffer-level
constructors every CUDA operation is built from.

Notes:
- Buffers retain their scalar dtype. Shape-erased inputs are checked against
  `Spec.Shape.size s` before shape-derived indexing or reverse-mode propagation.
- Training uses `backwardSparse`, which keeps only the gradients the caller retains and releases
  every other activation gradient during the traversal.
- Differentiable CUDA ops live in `NN.Runtime.Autograd.Engine.LibTorch.Ops`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

open Spec TorchLean

/--
Runtime, shape-erased CUDA buffer.

This plays the same role as `Spec.SomeTensor` in the CPU tape: it pairs runtime `Shape`
metadata with an opaque `LibTorch.Buffer` handle.
-/
structure AnyBuffer where
  /-- Runtime shape metadata for the buffer. -/
  s : Shape
  /-- Device buffer. Length is expected to be `Spec.Shape.size s`; dtype is retained natively. -/
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
def zeros (s : Shape) (dtype : Dtype := .float32) : Result AnyBuffer := do
  let n ← numelU32 s
  pure { s := s, buf := Buffer.zeros n dtype }

/-- Reject an implicit promotion at a value or cotangent boundary. -/
def checkDtype (value reference : AnyBuffer) : Result Unit := do
  unless Buffer.dtype value.buf == Buffer.dtype reference.buf &&
      Buffer.format? value.buf == Buffer.format? reference.buf do
    throw "autograd: native buffer dtype mismatch"

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
def add (a b : AnyBuffer) : Result AnyBuffer := do
  unless a.s == b.s do
    throw "autograd: gradient shape mismatch during accumulation"
  checkDtype a b
  let a ← validate a
  let b ← validate b
  pure { s := a.s, buf := Buffer.add a.buf b.buf }

end AnyBuffer

/-- A device workspace buffer kept on a node, shaped by its length. -/
def AnyBuffer.ofWorkspace (buf : Buffer) : AnyBuffer :=
  { s := [(Buffer.size buf).toNat], buf }

/--
CUDA gradients are device buffers. A sum consumes both of its inputs, and a contribution that no
node keeps is released at once, so long training loops do not wait on finalizers.
-/
instance : GradValue AnyBuffer where
  shape b := b.s
  retag g v _ := { s := v.s, buf := g.buf }
  check g v := do
    AnyBuffer.checkDtype g v
    discard <| AnyBuffer.validate { s := v.s, buf := g.buf }
  add a b := do
    let summed ← AnyBuffer.add a b
    pure { summed with buf := Buffer.releaseThen a.buf <| Buffer.releaseThen b.buf summed.buf }
  drop g rest := Buffer.releaseThen g.buf rest

/-- The CUDA zero gradient of a value. -/
instance : GradZero AnyBuffer where
  zerosLike v := { s := v.s, buf := Buffer.zerosLike v.buf }

namespace Tape

/--
Add a leaf node (no parents).

PyTorch comparison: a tensor that enters the graph as a leaf (e.g. input or parameter value).

This low-level constructor records `value` without validating its external buffer. Code that wraps
an externally supplied `Buffer` must call `AnyBuffer.validate` first; tape operations validate
their operands again when they retrieve values.
-/
@[inline] def leaf (t : Tape AnyBuffer) (value : AnyBuffer) (name : Option String := none)
    (requiresGrad : Bool := true) : Tape AnyBuffer × Nat :=
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
def requireValue (t : Tape AnyBuffer) (id : Nat) (s : Shape) : Result Buffer := do
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

Shapes are explicit and checked dynamically. Reverse mode visits the result only when the input
requires gradients, so constant subexpressions do not execute their backward closures.
-/
@[inline] def unary
    (t : Tape AnyBuffer) (opName : String) (xId : Nat) (σ τ : Shape)
    (forward : Buffer → Buffer)
    (backward : Buffer → Buffer → Buffer)
    (ownsValue : Bool := true) :
    Result (Tape AnyBuffer × Nat) := do
  let x ← requireValue (t := t) xId σ
  let y := forward x
  let node : Node AnyBuffer :=
    { name := some opName
      value := { s := τ, buf := y }
      ownsValue := ownsValue
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
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

Shapes are explicit and checked dynamically. The result requires gradients if either input does.
-/
@[inline] def binary
    (t : Tape AnyBuffer) (opName : String) (aId bId : Nat) (σ₁ σ₂ τ : Shape)
    (forward : Buffer → Buffer → Buffer)
    (backward : Buffer → Buffer → Buffer → (Buffer × Buffer)) :
    Result (Tape AnyBuffer × Nat) := do
  let a ← requireValue (t := t) aId σ₁
  let b ← requireValue (t := t) bId σ₂
  unless Buffer.dtype a == Buffer.dtype b && Buffer.format? a == Buffer.format? b do
    throw "autograd: binary operands have different dtypes"
  let y := forward a b
  let node : Node AnyBuffer :=
    { name := some opName
      value := { s := τ, buf := y }
      requiresGrad :=
        (t.getNode? aId).any (·.requiresGrad) || (t.getNode? bId).any (·.requiresGrad)
      parents := #[aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny τ
        let (da, db) := backward a b dLdy.buf
        pure #[(aId, { s := σ₁, buf := da }), (bId, { s := σ₂, buf := db })] }
  pure (t.addNode node)

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
def addSparseGrad (t : Tape AnyBuffer) (gradsRef : IO.Ref SparseGradMap)
    (id : Nat) (g : AnyBuffer) : IO Unit := do
  let node ← match t.getNode? id with
    | some node => pure node
    | none =>
        releaseSparseBuffer g.buf
        throw <| IO.userError "autograd: invalid parent id during sparse CUDA backward"
  if !node.requiresGrad then
    releaseSparseBuffer g.buf
  else if _h : g.s = node.value.s then
    match AnyBuffer.checkDtype g node.value with
    | .ok () => pure ()
    | .error message =>
        releaseSparseBuffer g.buf
        throw <| IO.userError message
    let contribution ← match AnyBuffer.validate { s := node.value.s, buf := g.buf } with
      | .ok contribution => pure contribution
      | .error msg =>
          releaseSparseBuffer g.buf
          throw <| IO.userError msg
    let grads ← gradsRef.get
    match grads.get? id with
    | none =>
        -- Backward rules return fresh buffers, so the copy consumes the contribution and leaves
        -- the map the only owner of what it keeps.
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
def backwardSparse (t : Tape AnyBuffer) (outId : Nat) (seed : AnyBuffer)
    (retain : Nat → Bool) : IO SparseGradMap := do
  let outNode ← match t.getNode? outId with
    | some node => pure node
    | none =>
        releaseSparseBuffer seed.buf
        throw <| IO.userError "autograd: invalid output id"
  if _h : seed.s = outNode.value.s then
    match AnyBuffer.checkDtype seed outNode.value with
    | .ok () => pure ()
    | .error message =>
        releaseSparseBuffer seed.buf
        throw <| IO.userError message
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

end LibTorch
end Autograd
end Runtime
