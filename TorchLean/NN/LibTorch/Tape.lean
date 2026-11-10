/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

-/

module

public import TorchLean.CUDA.LibTorch.Buffer
public import TorchLean.NN.Tape

/-!
# CUDA tape values

The CUDA tape is `Tape SomeBuffer`: the one tape of `TorchLean.NN.Tape`,
storing device buffers with their runtime shapes. This module gives those values their `GradValue`
instance, whose accumulation also releases the device buffers it consumes, and the buffer-level
constructors every CUDA operation is built from.

Notes:
- Buffers retain their scalar dtype. Shape-erased inputs are checked against
  `Shape.size s` before shape-derived indexing or reverse-mode propagation.
- Training uses `backwardSparse`, which keeps only the gradients the caller retains and releases
  every other activation gradient during the traversal.
- Differentiable CUDA operations are the entries of `TorchLean.LibTorch.Kernel.ofOp`
  (`TorchLean.NN.LibTorch.Kernel`).
- `TorchLean.Tape.leaf` records a buffer without checking it: code that wraps an
  externally supplied `Buffer` must call `SomeBuffer.validate` first. Tape operations check their
  operands again when they read them.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean
namespace LibTorch

/--
Runtime, shape-erased CUDA buffer.

Like `SomeTensor` on the CPU tape, it pairs runtime `Shape` metadata with an opaque
`LibTorch.Buffer` handle.
-/
structure SomeBuffer where
  /-- Runtime shape metadata for the buffer. -/
  s : Shape
  /-- Device buffer. Its length is expected to be `Shape.size s` and its dtype is retained
  natively. -/
  buf : Buffer

namespace SomeBuffer

/-- The error for a length that does not fit in the `UInt32` counts of the CUDA buffer API. -/
def tooLarge : String := "autograd: tensor too large (numel does not fit in UInt32)"

/-- Number of scalar elements as a `UInt32`, with every individual shape dimension checked too. -/
def numelU32 (s : Shape) : Result UInt32 := do
  for dim in s do
    if (Bytes.toUInt32? dim).isNone then throw tooLarge
  let some numel := Bytes.toUInt32? (Shape.size s) | throw tooLarge
  pure numel

/-- Reject an implicit promotion at a value or cotangent boundary. -/
def checkDtype (value reference : SomeBuffer) : Result Unit := do
  unless Buffer.dtype value.buf == Buffer.dtype reference.buf &&
      Buffer.format? value.buf == Buffer.format? reference.buf do
    throw "autograd: native buffer dtype mismatch"

/--
Check that runtime shape metadata agrees with the native CUDA buffer length.

`SomeBuffer` is intentionally shape-erased, so its structure alone cannot enforce this invariant.
Call this at boundaries that accept buffers assembled outside the typed tensor conversion path,
before any kernel uses shape-derived indexing.
-/
def validate (value : SomeBuffer) : Result SomeBuffer := do
  let expected ← numelU32 value.s
  let actual := Buffer.size value.buf
  if actual = expected then
    pure value
  else
    throw
      s!"autograd: CUDA buffer size mismatch \
         (shape elements={expected.toNat}, native elements={actual.toNat})"

/--
Accumulate two `SomeBuffer` values by elementwise addition, with a dynamic shape check.

Backprop uses it to sum gradient contributions in DAGs. The addition runs only after the shape
metadata, dtypes and formats agree and both native buffer lengths match that shape.
-/
def add (a b : SomeBuffer) : Result SomeBuffer := do
  unless a.s == b.s do
    throw "autograd: gradient shape mismatch during accumulation"
  checkDtype a b
  let a ← validate a
  let b ← validate b
  pure { s := a.s, buf := Buffer.add a.buf b.buf }

end SomeBuffer

/-- A device workspace buffer kept on a node, shaped by its length. -/
def SomeBuffer.ofWorkspace (buf : Buffer) : SomeBuffer :=
  { s := [(Buffer.size buf).toNat], buf }

/--
CUDA gradients are device buffers. A sum consumes both of its inputs, and a contribution that no
node keeps is released at once, so long training loops do not wait on finalizers.
-/
instance : GradValue SomeBuffer where
  shape b := b.s
  retag g v _ := { s := v.s, buf := g.buf }
  check g v := do
    SomeBuffer.checkDtype g v
    discard <| SomeBuffer.validate { s := v.s, buf := g.buf }
  add a b := do
    let summed ← SomeBuffer.add a b
    pure { summed with buf := Buffer.releaseThen a.buf <| Buffer.releaseThen b.buf summed.buf }
  drop g rest := Buffer.releaseThen g.buf rest

/-- The CUDA zero gradient of a value. -/
instance : GradZero SomeBuffer where
  zerosLike v := { s := v.s, buf := Buffer.zerosLike v.buf }

namespace Tape

open TorchLean.Tape (Node)

/--
Read a buffer value from a tape node id, requiring a specific runtime shape.

Fails if:
- the id is invalid,
- the stored shape does not match `s`,
- the shape cannot be represented by the `UInt32` CUDA ABI, or
- the native buffer length differs from `Shape.size s`.
-/
def requireValue (t : Tape SomeBuffer) (id : Nat) (s : Shape) : Result Buffer := do
  match t.getValue? id with
  | none => throw "autograd: invalid node id"
  | some any =>
      if decide (any.s = s) then
        let checked ← SomeBuffer.validate { s := s, buf := any.buf }
        pure checked.buf
      else
        throw "autograd: shape mismatch"

/--
Require that an upstream gradient matches an expected runtime shape.

Backward closures use it to check and cast the incoming cotangent. Both the shape tag and the
native buffer length are checked before a VJP reads the cotangent.
-/
def requireGrad (dLdyAny : SomeBuffer) (expected : Shape) : Result SomeBuffer := do
  if decide (dLdyAny.s = expected) then
    SomeBuffer.validate { s := expected, buf := dLdyAny.buf }
  else
    throw "autograd: upstream gradient shape mismatch"

/--
Generic constructor for unary ops.

The caller supplies:
- `forward : Buffer → Buffer`
- `backward : Buffer → Buffer → Buffer`, the VJP, which takes the input `x` and the upstream
  `dLdy` and returns `dLdx`

Shapes are passed as arguments and checked at run time. Reverse mode visits the result only when
the input requires gradients, so constant subexpressions do not execute their backward closures.
-/
@[inline] def unary
    (t : Tape SomeBuffer) (opName : String) (xId : Nat) (σ τ : Shape)
    (forward : Buffer → Buffer)
    (backward : Buffer → Buffer → Buffer)
    (ownsValue : Bool := true) :
    Result (Tape SomeBuffer × Nat) := do
  let x ← requireValue (t := t) xId σ
  let y := forward x
  let node : Node SomeBuffer :=
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

The caller supplies:
- `forward : Buffer → Buffer → Buffer`
- `backward : Buffer → Buffer → Buffer → (Buffer × Buffer)`, the VJP, which takes the inputs `a`
  and `b` and the upstream `dLdy` and returns `(dLda, dLdb)`

Shapes are passed as arguments and checked at run time. The result requires gradients if either
input does.
-/
@[inline] def binary
    (t : Tape SomeBuffer) (opName : String) (aId bId : Nat) (σ₁ σ₂ τ : Shape)
    (forward : Buffer → Buffer → Buffer)
    (backward : Buffer → Buffer → Buffer → (Buffer × Buffer)) :
    Result (Tape SomeBuffer × Nat) := do
  let a ← requireValue (t := t) aId σ₁
  let b ← requireValue (t := t) bId σ₂
  unless Buffer.dtype a == Buffer.dtype b && Buffer.format? a == Buffer.format? b do
    throw "autograd: binary operands have different dtypes"
  let y := forward a b
  let node : Node SomeBuffer :=
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
abbrev SparseGradMap := Std.HashMap Nat SomeBuffer

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
def addSparseGrad (t : Tape SomeBuffer) (gradsRef : IO.Ref SparseGradMap)
    (id : Nat) (g : SomeBuffer) : IO Unit := do
  let node ← match t.getNode? id with
    | some node => pure node
    | none =>
        releaseSparseBuffer g.buf
        throw <| IO.userError "autograd: invalid parent id during sparse CUDA backward"
  if !node.requiresGrad then
    releaseSparseBuffer g.buf
  else if _h : g.s = node.value.s then
    match SomeBuffer.checkDtype g node.value with
    | .ok () => pure ()
    | .error message =>
        releaseSparseBuffer g.buf
        throw <| IO.userError message
    let contribution ← match SomeBuffer.validate { s := node.value.s, buf := g.buf } with
      | .ok contribution => pure contribution
      | .error msg =>
          releaseSparseBuffer g.buf
          throw <| IO.userError msg
    let grads ← gradsRef.get
    match grads.get? id with
    | none =>
        -- Backward rules return fresh buffers, so the copy consumes the contribution and leaves
        -- the map the only owner of what it keeps.
        let owned : SomeBuffer := {
          s := contribution.s
          buf := Buffer.copyAndRelease contribution.buf }
        gradsRef.set (grads.insert id owned)
    | some old =>
        if _hOld : old.s = node.value.s then
          let old' : SomeBuffer := { s := node.value.s, buf := old.buf }
          let summed ← match SomeBuffer.add old' contribution with
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

The call consumes `seed`. The map owns the returned buffers, which must be released with
`releaseSparseGrads` after the optimizer has consumed them. All other activation gradients are
retired during the traversal.
-/
def backwardSparse (t : Tape SomeBuffer) (outId : Nat) (seed : SomeBuffer)
    (retain : Nat → Bool) : IO SparseGradMap := do
  let outNode ← match t.getNode? outId with
    | some node => pure node
    | none =>
        releaseSparseBuffer seed.buf
        throw <| IO.userError "autograd: invalid output id"
  if _h : seed.s = outNode.value.s then
    match SomeBuffer.checkDtype seed outNode.value with
    | .ok () => pure ()
    | .error message =>
        releaseSparseBuffer seed.buf
        throw <| IO.userError message
    let _ ← match SomeBuffer.validate outNode.value with
      | .ok value => pure value
      | .error msg =>
          releaseSparseBuffer seed.buf
          throw <| IO.userError msg
    let seedChecked ← match SomeBuffer.validate { s := outNode.value.s, buf := seed.buf } with
      | .ok value => pure value
      | .error msg =>
          releaseSparseBuffer seed.buf
          throw <| IO.userError msg
    let seedOwned : SomeBuffer := {
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
              let expectedSize := Shape.size node.value.s
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
end TorchLean
